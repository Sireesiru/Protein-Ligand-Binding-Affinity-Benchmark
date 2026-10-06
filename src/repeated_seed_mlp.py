#!/usr/bin/env python3

import os
import sys
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# PATHS / SETTINGS
# ============================================================

BASE = Path(
    "/lustre/orion/gen006/scratch/sireesiru/miniffinity/all_emb"
)

AFFINITY_FILE = Path(
    "/lustre/orion/gen006/scratch/sireesiru/miniffinity/"
    "pdbbind_canonical_affinities.csv"
)

OUTPUT_DIR = BASE / "XGB" / "logs" / "repeated_seed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_EPOCHS = 40
K_FEATURES = 512

SEEDS = [
    42, 43, 44, 45, 46,
    47, 48, 49, 50, 51
]


# ============================================================
# PROTEIN EMBEDDINGS
# ============================================================

PROTEIN_MODELS = [
    ("esm2", "esm2_embeddings_mean.npy"),
    ("prostt5", "prostt5_embeddings_mean.npy"),
    ("progen2", "progen2_embeddings.npy"),
    ("boltz", "boltz_mp.npy"),
    ("protgpt2", "pdbbind_protgpt2_embeddings.npy"),
]


# ============================================================
# FROZEN HYPERPARAMETERS FROM ORIGINAL RAY TUNING
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
# Same architecture as original Ray experiment
# ============================================================

class AffinityPredictionNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout_rate=0.2
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# TRAIN + TEST ONE MODEL
# ============================================================

def train_and_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    config,
    seed
):

    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator
    )

    model = AffinityPredictionNetwork(
        input_dim=X_train.shape[1],
        hidden_dim=int(config["hidden_dim"]),
        dropout_rate=config["dropout"]
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for epoch in range(MAX_EPOCHS):

        model.train()

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions = model(batch_X)

            loss = criterion(
                predictions,
                batch_y
            )

            loss.backward()
            optimizer.step()

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    model.eval()

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False
    )

    preds = []
    targets = []

    with torch.no_grad():

        for batch_X, batch_y in test_loader:

            batch_X = batch_X.to(device)

            prediction = (
                model(batch_X)
                .cpu()
                .numpy()
            )

            preds.append(prediction)
            targets.append(batch_y.numpy())

    y_pred = np.vstack(preds).ravel()
    y_true = np.vstack(targets).ravel()

    r2 = r2_score(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mae = mean_absolute_error(
        y_true, y_pred
    )

    return r2, rmse, mae, y_true, y_pred


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(
    X,
    train_idx,
    test_idx,
    y_train,
    k_features=K_FEATURES
):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X[train_idx]
    )

    X_test = scaler.transform(
        X[test_idx]
    )

    # If modality has fewer than 512 dimensions,
    # simply retain all dimensions.

    k = min(
        k_features,
        X_train.shape[1]
    )

    selector = SelectKBest(
        score_func=mutual_info_regression,
        k=k
    )

    X_train = selector.fit_transform(
        X_train,
        y_train.ravel()
    ).astype(np.float32)

    X_test = selector.transform(
        X_test
    ).astype(np.float32)

    return X_train, X_test


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

    task_id = args.task_id

    # --------------------------------------------------------
    # Map array ID 0-49 to:
    # 5 models x 10 seeds
    # --------------------------------------------------------

    model_index = task_id // len(SEEDS)
    seed_index = task_id % len(SEEDS)

    if model_index >= len(PROTEIN_MODELS):
        raise ValueError(
            f"Invalid task_id {task_id}"
        )

    model_name, protein_file = (
        PROTEIN_MODELS[model_index]
    )

    seed = SEEDS[seed_index]

    config = BEST_CONFIGS[model_name]

    output_file = (
        OUTPUT_DIR /
        f"{model_name}_seed{seed}.csv"
    )

    prediction_file = (
        OUTPUT_DIR /
        f"{model_name}_seed{seed}_predictions.csv"
    )

    # --------------------------------------------------------
    # RESTART SAFETY
    # --------------------------------------------------------

    if output_file.exists():

        print(
            f"\nAlready completed:\n{output_file}"
        )

        print("Skipping task.")

        sys.exit(0)

    print("=" * 70)
    print("REPEATED-SEED MLP EXPERIMENT")
    print("=" * 70)

    print(f"Task ID : {task_id}")
    print(f"Model   : {model_name}")
    print(f"Seed    : {seed}")
    print(f"Config  : {config}")

    set_seed(seed)

    # ========================================================
    # LOAD AFFINITY DATA
    # ========================================================

    df = pl.read_csv(
        str(AFFINITY_FILE)
    )

    names = df[
        "name"
    ].to_numpy()

    y_raw = df[
        "neg_log10_affinity_M"
    ].to_numpy()

    bad = {
        "3ag9",
        "5dyw"
    }

    mask = np.array([
        name not in bad
        for name in names
    ])

    names_filtered = names[mask]

    y = (
        y_raw[mask]
        .astype(np.float32)
    )

    # ========================================================
    # LOAD EMBEDDINGS
    # ========================================================

    X_lig = np.load(
        BASE /
        "pdbbind_txgemma_ligand_embeddings.npy"
    )[mask]

    X_prot_raw = np.load(
        BASE / protein_file
    )

    if X_prot_raw.shape[0] == mask.sum():

        X_prot = X_prot_raw

    else:

        X_prot = X_prot_raw[mask]

    assert len(X_prot) == len(X_lig)
    assert len(X_prot) == len(y)

    print(
        "Protein shape:",
        X_prot.shape
    )

    print(
        "Ligand shape :",
        X_lig.shape
    )

    # ========================================================
    # EXACT 70 / 15 / 15 SPLIT
    # ========================================================

    split1 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.70,
        random_state=seed
    )

    train_idx, temp_idx = next(
        split1.split(
            X_prot,
            y,
            groups=names_filtered
        )
    )

    split2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=seed
    )

    val_relative_idx, test_relative_idx = next(
        split2.split(
            X_prot[temp_idx],
            y[temp_idx],
            groups=names_filtered[temp_idx]
        )
    )

    val_idx = temp_idx[
        val_relative_idx
    ]

    test_idx = temp_idx[
        test_relative_idx
    ]

    print("\nSPLIT")
    print("Train:", len(train_idx))
    print("Val  :", len(val_idx))
    print("Test :", len(test_idx))

    # We deliberately do not use validation here.
    # Hyperparameters are already frozen from Ray.

    y_train = (
        y[train_idx]
        .reshape(-1, 1)
    )

    y_test = (
        y[test_idx]
        .reshape(-1, 1)
    )

    # ========================================================
    # EXPERIMENT 1: PROTEIN ONLY
    # ========================================================

    print("\n[1/3] Protein-only")

    Xp_train, Xp_test = prepare_features(
        X_prot,
        train_idx,
        test_idx,
        y_train
    )

    protein_metrics = train_and_evaluate(
        Xp_train,
        y_train,
        Xp_test,
        y_test,
        config,
        seed
    )

    # ========================================================
    # EXPERIMENT 2: LIGAND ONLY
    # ========================================================

    print("\n[2/3] TxGemma ligand-only")

    Xl_train, Xl_test = prepare_features(
        X_lig,
        train_idx,
        test_idx,
        y_train
    )

    ligand_metrics = train_and_evaluate(
        Xl_train,
        y_train,
        Xl_test,
        y_test,
        config,
        seed
    )

    # ========================================================
    # EXPERIMENT 3: PROTEIN + LIGAND
    #
    # Scale modalities independently first, exactly as in
    # original combined pipeline.
    # ========================================================

    print("\n[3/3] Protein + ligand")

    scaler_prot = StandardScaler()
    scaler_lig = StandardScaler()

    Xp_train_scaled = scaler_prot.fit_transform(
        X_prot[train_idx]
    )

    Xp_test_scaled = scaler_prot.transform(
        X_prot[test_idx]
    )

    Xl_train_scaled = scaler_lig.fit_transform(
        X_lig[train_idx]
    )

    Xl_test_scaled = scaler_lig.transform(
        X_lig[test_idx]
    )

    X_comb_train = np.concatenate(
        [
            Xp_train_scaled,
            Xl_train_scaled
        ],
        axis=1
    )

    X_comb_test = np.concatenate(
        [
            Xp_test_scaled,
            Xl_test_scaled
        ],
        axis=1
    )

    selector = SelectKBest(
        score_func=mutual_info_regression,
        k=min(
            K_FEATURES,
            X_comb_train.shape[1]
        )
    )

    X_comb_train = selector.fit_transform(
        X_comb_train,
        y_train.ravel()
    ).astype(np.float32)

    X_comb_test = selector.transform(
        X_comb_test
    ).astype(np.float32)

    combined_metrics = train_and_evaluate(
        X_comb_train,
        y_train,
        X_comb_test,
        y_test,
        config,
        seed
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    rows = []

    for modality, result in [

        ("protein", protein_metrics),

        ("ligand", ligand_metrics),

        ("combined", combined_metrics),

    ]:

        r2, rmse, mae, _, _ = result

        rows.append({

            "model": model_name,
            "seed": seed,
            "modality": modality,

            "r2": r2,
            "rmse": rmse,
            "mae": mae,

            "hidden_dim":
                config["hidden_dim"],

            "dropout":
                config["dropout"],

            "lr":
                config["lr"],

            "batch_size":
                config["batch_size"],

            "n_train":
                len(train_idx),

            "n_val":
                len(val_idx),

            "n_test":
                len(test_idx),
        })

    result_df = pd.DataFrame(rows)

    result_df.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # SAVE TEST PREDICTIONS
    #
    # Important for bootstrap CIs and paired tests later.
    # ========================================================

    pred_df = pd.DataFrame({

        "pdb_id":
            names_filtered[test_idx],

        "seed":
            seed,

        "model":
            model_name,

        "y_true":
            protein_metrics[3],

        "pred_protein":
            protein_metrics[4],

        "pred_ligand":
            ligand_metrics[4],

        "pred_combined":
            combined_metrics[4],
    })

    pred_df.to_csv(
        prediction_file,
        index=False
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 70)

    print(
        result_df[
            [
                "model",
                "seed",
                "modality",
                "r2",
                "rmse",
                "mae"
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(output_file)
    print(prediction_file)

    print("=" * 70)


if __name__ == "__main__":
    main()