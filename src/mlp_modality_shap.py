#!/usr/bin/env python3

import os
import argparse
import random
import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import shap

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
MAX_EPOCHS = 40
K_FEATURES = 512

BASE = "/lustre/orion/gen006/scratch/sireesiru/miniffinity/all_emb"
WORK = os.path.join(BASE, "XGB")

AFFINITY_FILE = (
    "/lustre/orion/gen006/scratch/sireesiru/miniffinity/"
    "pdbbind_canonical_affinities.csv"
)

OUTDIR = os.path.join(WORK, "mlp_shap")
os.makedirs(OUTDIR, exist_ok=True)


PROTEIN_MAP = [
    ("esm2", "esm2_embeddings_mean.npy"),
    ("prostt5", "prostt5_embeddings_mean.npy"),
    ("progen2", "progen2_embeddings.npy"),
    ("boltz", "boltz_mp.npy"),
    ("protgpt2", "pdbbind_protgpt2_embeddings.npy"),
]


# ============================================================
# FROZEN BEST CONFIGS FROM ORIGINAL RAY OPTIMIZATION
# NO RAY TUNING IS PERFORMED IN THIS SCRIPT
# ============================================================

BEST_CONFIGS = {

    "esm2": {
        "hidden_dim": 512,
        "dropout": 0.18339,
        "lr": 0.0007060,
        "batch_size": 128,
    },

    "prostt5": {
        "hidden_dim": 256,
        "dropout": 0.11341,
        "lr": 0.0045398,
        "batch_size": 128,
    },

    "progen2": {
        "hidden_dim": 1024,
        "dropout": 0.16128,
        "lr": 0.0001782,
        "batch_size": 64,
    },

    "boltz": {
        "hidden_dim": 512,
        "dropout": 0.39088,
        "lr": 0.0070943,
        "batch_size": 32,
    },

    "protgpt2": {
        "hidden_dim": 512,
        "dropout": 0.15957,
        "lr": 0.0031925,
        "batch_size": 32,
    },
}

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# MODEL
# ============================================================

class AffinityPredictionNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout_rate
    ):

        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.BatchNorm1d(
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout_rate
            ),
            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),

            nn.BatchNorm1d(
                hidden_dim // 2
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout_rate
            ),

            nn.Linear(
                hidden_dim // 2,
                1
            ),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# LOAD SAVED SEED-42 SPLIT
# ============================================================

def load_saved_split(model_name):

    split_file = os.path.join(
        WORK,
        f"pdbbind_split_seed42_{model_name}_protein.npz"
    )

    if not os.path.exists(split_file):

        raise FileNotFoundError(
            f"Saved split not found:\n{split_file}"
        )

    split = np.load(
        split_file,
        allow_pickle=True
    )

    print("\nUsing saved split:")
    print(split_file)

    print("Keys:", split.files)

    # --------------------------------------------------------
    # Support likely key naming schemes
    # --------------------------------------------------------

    key_sets = [

        (
            "train_idx",
            "val_idx",
            "test_idx"
        ),

        (
            "train_indices",
            "val_indices",
            "test_indices"
        ),

        (
            "train_index",
            "val_index",
            "test_index"
        ),

        (
            "train",
            "val",
            "test"
        ),
    ]

    for train_key, val_key, test_key in key_sets:

        if (
            train_key in split.files
            and val_key in split.files
            and test_key in split.files
        ):

            train_idx = np.asarray(
                split[train_key],
                dtype=int
            )

            val_idx = np.asarray(
                split[val_key],
                dtype=int
            )

            test_idx = np.asarray(
                split[test_key],
                dtype=int
            )

            print(
                "Using keys:",
                train_key,
                val_key,
                test_key
            )

            return (
                train_idx,
                val_idx,
                test_idx
            )

    raise KeyError(
        "Could not identify train/val/test indices.\n"
        f"Available keys: {split.files}"
    )

# ============================================================
# TRAIN FINAL MODEL
# Does NOT perform Ray tuning. Uses the ORIGINAL representation-specific Ray optimum.
# ============================================================

def train_model(
    X_train,
    y_train,
    config,
    device
):

    set_seed(SEED)

    dataset = TensorDataset(

        torch.tensor(
            X_train,
            dtype=torch.float32
        ),

        torch.tensor(
            y_train,
            dtype=torch.float32
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader = DataLoader(

        dataset,

        batch_size=int(
            config["batch_size"]
        ),

        shuffle=True,

        generator=generator,
    )

    model = AffinityPredictionNetwork(

        input_dim=X_train.shape[1],

        hidden_dim=int(
            config["hidden_dim"]
        ),

        dropout_rate=float(
            config["dropout"]
        ),

    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(
            config["lr"]
        ),
        weight_decay=1e-4,
    )
    criterion = nn.MSELoss()
    for epoch in range(
        MAX_EPOCHS
    ):
        model.train()

        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)

            loss = criterion(
                pred,
                yb
            )
            loss.backward()
            optimizer.step()

    model.eval()
    return model

# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_id",
        type=int,
        required=True
    )

    args = parser.parse_args()

    if args.task_id < 0 or args.task_id >= len(PROTEIN_MAP):

        raise ValueError(
            "task_id must be between 0 and 4."
        )

    set_seed(SEED)

    model_name, filename = (
        PROTEIN_MAP[
            args.task_id
        ]
    )

    print("\n" + "=" * 70)
    print("MODEL:", model_name.upper())
    print("=" * 70)

    # ========================================================
    # AFFINITY
    # ========================================================

    df = pl.read_csv(
        AFFINITY_FILE
    )

    names_raw = (
        df["name"]
        .to_numpy()
    )

    y_raw = (
        df[
            "neg_log10_affinity_M"
        ]
        .to_numpy()
    )

    bad = {
        "3ag9",
        "5dyw"
    }

    mask = np.array([

        name not in bad

        for name in names_raw
    ])

    names = names_raw[
        mask
    ]

    y = (
        y_raw[
            mask
        ]
        .astype(
            np.float32
        )
    )

    print(
        "Filtered samples:",
        len(y)
    )

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    X_lig_raw = np.load(

        os.path.join(

            BASE,

            "pdbbind_txgemma_ligand_embeddings.npy"
        )
    )

    X_lig = X_lig_raw[
        mask
    ]

    Xp_raw = np.load(

        os.path.join(
            BASE,
            filename
        )
    )

    # Some protein embedding files were already filtered.
    if Xp_raw.shape[0] == mask.sum():

        print(
            "Protein embeddings already filtered."
        )

        X_prot = Xp_raw

    elif Xp_raw.shape[0] == len(mask):

        X_prot = Xp_raw[
            mask
        ]

    else:

        raise ValueError(
            f"Unexpected number of protein rows: "
            f"{Xp_raw.shape[0]}"
        )

    assert (
        len(X_prot)
        == len(X_lig)
        == len(y)
    )

    protein_dim = (
        X_prot.shape[1]
    )

    ligand_dim = (
        X_lig.shape[1]
    )

    print(
        "Protein shape:",
        X_prot.shape
    )

    print(
        "Ligand shape:",
        X_lig.shape
    )

    # ========================================================
    # USE EXISTING SEED-42 SPLIT
    # ========================================================

    (
        train_idx,
        val_idx,
        test_idx
    ) = load_saved_split(
        model_name
    )

    print("\nSPLIT SIZES")

    print(
        "Train:",
        len(train_idx)
    )

    print(
        "Validation:",
        len(val_idx)
    )

    print(
        "Test:",
        len(test_idx)
    )

    # Safety
    max_idx = max(

        np.max(train_idx),

        np.max(val_idx),

        np.max(test_idx)
    )

    if max_idx >= len(y):

        raise ValueError(
            "Saved split indices exceed "
            "the filtered dataset size."
        )

    y_train = (
        y[train_idx]
        .reshape(-1, 1)
    )

    y_val = (
        y[val_idx]
        .reshape(-1, 1)
    )

    y_test = (
        y[test_idx]
        .reshape(-1, 1)
    )

    # ========================================================
    # TRAIN-ONLY SCALING
    # ========================================================

    sp = StandardScaler()

    sl = StandardScaler()

    Xp_train = sp.fit_transform(
        X_prot[
            train_idx
        ]
    )

    Xp_val = sp.transform(
        X_prot[
            val_idx
        ]
    )

    Xp_test = sp.transform(
        X_prot[
            test_idx
        ]
    )

    Xl_train = sl.fit_transform(
        X_lig[
            train_idx
        ]
    )

    Xl_val = sl.transform(
        X_lig[
            val_idx
        ]
    )

    Xl_test = sl.transform(
        X_lig[
            test_idx
        ]
    )

    # ========================================================
    # ORIGINAL FUSION
    # ========================================================

    X_train_full = np.concatenate(

        [
            Xp_train,
            Xl_train
        ],

        axis=1
    )

    X_val_full = np.concatenate(

        [
            Xp_val,
            Xl_val
        ],

        axis=1
    )

    X_test_full = np.concatenate(

        [
            Xp_test,
            Xl_test
        ],

        axis=1
    )

    # ========================================================
    # RECOMPUTE MI ON SAME SAVED TRAINING SPLIT
    # We deliberately do NOT reuse the old selected_indices.
    # This verifies/reconstructs the original MI pipeline
    # from the saved seed-42 training data.
    # ========================================================

    print(
        "\nRecomputing MI selection "
        "on saved seed-42 TRAIN split..."
    )

    selector = SelectKBest(

        score_func=
            mutual_info_regression,

        k=min(
            K_FEATURES,
            X_train_full.shape[1]
        ),
    )

    X_train = selector.fit_transform(

        X_train_full,

        y_train.ravel()

    ).astype(
        np.float32
    )

    X_val = selector.transform(

        X_val_full

    ).astype(
        np.float32
    )

    X_test = selector.transform(

        X_test_full

    ).astype(
        np.float32
    )

    selected_indices = (
        selector.get_support(
            indices=True
        )
    )

    # Save separately so old files are not overwritten.
    np.save(

        os.path.join(

            OUTDIR,

            f"selected_indices_"
            f"{model_name}_"
            f"mi_recomputed.npy"
        ),

        selected_indices
    )

    # ========================================================
    # MI FEATURE COMPOSITION
    # ========================================================

    selected_is_protein = (
        selected_indices
        < protein_dim
    )

    n_protein_selected = int(

        selected_is_protein.sum()
    )

    n_ligand_selected = int(

        (~selected_is_protein).sum()
    )

    mi_protein_pct = (

        100.0
        * n_protein_selected
        / len(selected_indices)
    )

    mi_ligand_pct = (

        100.0
        * n_ligand_selected
        / len(selected_indices)
    )

    print(
        "\nMI SELECTED FEATURES"
    )

    print(
        f"Protein: "
        f"{n_protein_selected} "
        f"({mi_protein_pct:.2f}%)"
    )

    print(
        f"Ligand : "
        f"{n_ligand_selected} "
        f"({mi_ligand_pct:.2f}%)"
    )

    # ========================================================
    # TRAIN FINAL MODEL
    # IMPORTANT:NO RAY TUNING HERE.
    # We use the previously discovered Ray optimum.
    # ========================================================

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    config = (
        BEST_CONFIGS[
            model_name
        ]
    )

    print(
        "\nUsing frozen original Ray config:"
    )

    print(
        config
    )

    model = train_model(

        X_train,

        y_train,

        config,

        device
    )

    # ========================================================
    # TEST PERFORMANCE
    # ========================================================

    with torch.no_grad():

        pred = model(

            torch.tensor(

                X_test,

                dtype=torch.float32,

                device=device
            )

        ).cpu().numpy()

    r2 = r2_score(
        y_test,
        pred
    )

    rmse = np.sqrt(

        mean_squared_error(
            y_test,
            pred
        )
    )

    mae = mean_absolute_error(
        y_test,
        pred
    )

    print(
        "\nTEST"
    )

    print(
        f"R2   : {r2:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"MAE  : {mae:.4f}"
    )

    # ========================================================
    # MLP SHAP
    # ========================================================

    print(
        "\nCalculating MLP SHAP..."
    )

    rng = (
        np.random.default_rng(
            SEED
        )
    )

    bg_n = min(
        256,
        len(X_train)
    )

    explain_n = min(
        512,
        len(X_test)
    )

    bg_idx = rng.choice(

        len(X_train),

        size=bg_n,

        replace=False
    )

    ex_idx = rng.choice(

        len(X_test),

        size=explain_n,

        replace=False
    )

    background = torch.tensor(

        X_train[
            bg_idx
        ],

        dtype=torch.float32,

        device=device
    )

    explain_data = torch.tensor(

        X_test[
            ex_idx
        ],

        dtype=torch.float32,

        device=device
    )

    model.eval()

    explainer = (
        shap.GradientExplainer(

            model,

            background
        )
    )

    shap_values = (
        explainer.shap_values(
            explain_data
        )
    )

    if isinstance(
        shap_values,
        list
    ):

        shap_values = (
            shap_values[0]
        )

    shap_values = np.asarray(
        shap_values
    )

    # PyTorch SHAP may return:
    # samples x features x 1
    if (
        shap_values.ndim == 3
        and
        shap_values.shape[-1] == 1
    ):

        shap_values = (
            shap_values[
                :, :, 0
            ]
        )

    if shap_values.ndim != 2:

        raise ValueError(
            "Unexpected SHAP shape: "
            f"{shap_values.shape}"
        )

    mean_abs_shap = np.mean(

        np.abs(
            shap_values
        ),

        axis=0
    )

    if (
        len(mean_abs_shap)
        != len(selected_indices)
    ):

        raise ValueError(
            "SHAP feature count does not "
            "match MI-selected feature count."
        )

    protein_shap = (

        mean_abs_shap[
            selected_is_protein
        ]
        .sum()
    )

    ligand_shap = (

        mean_abs_shap[
            ~selected_is_protein
        ]
        .sum()
    )

    total_shap = (
        protein_shap
        + ligand_shap
    )

    protein_pct = (

        100.0
        * protein_shap
        / total_shap
    )

    ligand_pct = (

        100.0
        * ligand_shap
        / total_shap
    )

    print(
        "\nMLP SHAP ATTRIBUTION"
    )

    print(
        f"Protein: "
        f"{protein_pct:.2f}%"
    )

    print(
        f"Ligand : "
        f"{ligand_pct:.2f}%"
    )

    # ========================================================
    # SAVE
    # ========================================================

    result = pd.DataFrame([{

        "representation":
            model_name,

        "protein_dim":
            protein_dim,

        "ligand_dim":
            ligand_dim,

        "mi_protein_features":
            n_protein_selected,

        "mi_ligand_features":
            n_ligand_selected,

        "mi_protein_pct":
            mi_protein_pct,

        "mi_ligand_pct":
            mi_ligand_pct,

        "mlp_shap_protein_pct":
            protein_pct,

        "mlp_shap_ligand_pct":
            ligand_pct,

        "test_r2":
            r2,

        "test_rmse":
            rmse,

        "test_mae":
            mae,
    }])

    outfile = os.path.join(

        OUTDIR,

        f"{model_name}_"
        f"mlp_shap.csv"
    )

    result.to_csv(
        outfile,
        index=False
    )

    print(
        "\nSaved:",
        outfile
    )


if __name__ == "__main__":
    main()