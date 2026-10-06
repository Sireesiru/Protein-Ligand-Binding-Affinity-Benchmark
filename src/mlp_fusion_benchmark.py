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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session


# ============================================================
# GLOBAL SETTINGS
# ============================================================

SEED = 42
MAX_EPOCHS = 40
K_FEATURES = 512
N_TRIALS = 30

BASE = "/lustre/orion/gen006/scratch/sireesiru/miniffinity/all_emb"
WORK = os.path.join(BASE, "XGB")

AFFINITY_FILE = (
    "/lustre/orion/gen006/scratch/sireesiru/miniffinity/"
    "pdbbind_canonical_affinities.csv"
)

LIGAND_FILE = os.path.join(
    BASE, "pdbbind_txgemma_ligand_embeddings.npy")

OUTDIR = os.path.join(
    WORK,
    "fusion_results"
)

os.makedirs(
    OUTDIR,
    exist_ok=True
)

# ============================================================
# PROTEIN REPRESENTATIONS
# ============================================================

PROTEIN_MAP = [("esm2","esm2_embeddings_mean.npy"),
    ("prostt5","prostt5_embeddings_mean.npy"),
    ("progen2","progen2_embeddings.npy"),
    ("boltz","boltz_mp.npy"),
    ("protgpt2","pdbbind_protgpt2_embeddings.npy"]
# ============================================================
# FUSION / FEATURE-SELECTION STRATEGIES
# ============================================================

STRATEGIES = {
    0: "mi512",
    1: "balanced_mi",
    2: "rf_shap512",
    3: "all_features"}
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
# Same architecture as original MLP study
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
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# LOAD EXISTING SEED-42 SPLIT
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

    print(
        "Split keys:",
        split.files
    )

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
        )
    ]

    for train_key, val_key, test_key in key_sets:

        if (
            train_key in split.files
            and
            val_key in split.files
            and
            test_key in split.files
        ):

            print(
                "Using split keys:",
                train_key,
                val_key,
                test_key
            )

            return (

                np.asarray(
                    split[train_key],
                    dtype=int
                ),

                np.asarray(
                    split[val_key],
                    dtype=int
                ),

                np.asarray(
                    split[test_key],
                    dtype=int
                )
            )

    raise KeyError(
        "Could not identify train/val/test keys. "
        f"Available keys: {split.files}"
    )


# ============================================================
# RAY TRAINING PIPELINE
#
# Receives TRAIN + VALIDATION only.
# TEST is never visible to Ray.
# ============================================================

def train_affinity_pipeline(
    config,
    data_ref
):

    set_seed(SEED)

    (
        X_train,
        X_val,
        y_train,
        y_val
    ) = data_ref

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = TensorDataset(

        torch.tensor(
            X_train,
            dtype=torch.float32
        ),

        torch.tensor(
            y_train,
            dtype=torch.float32
        )
    )

    val_dataset = TensorDataset(

        torch.tensor(
            X_val,
            dtype=torch.float32
        ),

        torch.tensor(
            y_val,
            dtype=torch.float32
        )
    )

    # --------------------------------------------------------
    # REPRODUCIBLE TRAIN SHUFFLING
    # --------------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(

        train_dataset,

        batch_size=int(
            config["batch_size"]
        ),

        shuffle=True,

        generator=generator
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=256,

        shuffle=False
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = AffinityPredictionNetwork(

        input_dim=X_train.shape[1],

        hidden_dim=int(
            config["hidden_dim"]
        ),

        dropout_rate=float(
            config["dropout"]
        )

    ).to(device)

    optimizer = optim.AdamW(

        model.parameters(),

        lr=float(
            config["lr"]
        ),

        weight_decay=1e-4
    )

    criterion = nn.MSELoss()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for epoch in range(MAX_EPOCHS):

        model.train()

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)

            loss = criterion(
                pred,
                yb
            )

            loss.backward()

            optimizer.step()

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_pred = []
        val_true = []

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(device)

                pred = model(
                    xb
                )

                val_pred.append(
                    pred.cpu().numpy()
                )

                val_true.append(
                    yb.numpy()
                )

        val_pred = np.concatenate(
            val_pred
        ).ravel()

        val_true = np.concatenate(
            val_true
        ).ravel()

        val_r2 = r2_score(
            val_true,
            val_pred
        )

        val_rmse = np.sqrt(
            mean_squared_error(
                val_true,
                val_pred
            )
        )

        val_mae = mean_absolute_error(
            val_true,
            val_pred
        )

        session.report({

            "r2_score":
                val_r2,

            "rmse":
                val_rmse,

            "mae":
                val_mae
        })


# ============================================================
# TRAIN FINAL MODEL
#
# Fresh model using best Ray hyperparameters.
# Train only.
# Untouched TEST used only after training.
# ============================================================

def train_final_model(
    config,
    X_train,
    y_train,
    X_test,
    y_test
):

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = AffinityPredictionNetwork(

        input_dim=X_train.shape[1],

        hidden_dim=int(
            config["hidden_dim"]
        ),

        dropout_rate=float(
            config["dropout"]
        )

    ).to(device)

    optimizer = optim.AdamW(

        model.parameters(),

        lr=float(
            config["lr"]
        ),

        weight_decay=1e-4
    )

    criterion = nn.MSELoss()

    dataset = TensorDataset(

        torch.tensor(
            X_train,
            dtype=torch.float32
        ),

        torch.tensor(
            y_train,
            dtype=torch.float32
        )
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader = DataLoader(

        dataset,

        batch_size=int(
            config["batch_size"]
        ),

        shuffle=True,

        generator=generator
    )

    # --------------------------------------------------------
    # SAME 40-EPOCH FINAL TRAINING
    # --------------------------------------------------------

    for epoch in range(MAX_EPOCHS):

        model.train()

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)

            loss = criterion(
                pred,
                yb
            )

            loss.backward()

            optimizer.step()

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        pred = model(

            torch.tensor(
                X_test,
                dtype=torch.float32,
                device=device
            )

        ).cpu().numpy().ravel()

    y_true = y_test.ravel()

    test_r2 = r2_score(
        y_true,
        pred
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_true,
            pred
        )
    )

    test_mae = mean_absolute_error(
        y_true,
        pred
    )

    return (
        test_r2,
        test_rmse,
        test_mae
    )


# ============================================================
# APPLY FUSION / FEATURE-SELECTION STRATEGY
#
# ALL selection is fitted using TRAIN only.
# ============================================================

def apply_strategy(
    strategy,
    Xp_train,
    Xp_val,
    Xp_test,
    Xl_train,
    Xl_val,
    Xl_test,
    y_train,
    model_name
):

    protein_dim = Xp_train.shape[1]
    ligand_dim = Xl_train.shape[1]

    # --------------------------------------------------------
    # CONCATENATED FULL REPRESENTATION
    # Protein first, ligand second.
    # --------------------------------------------------------

    X_train_full = np.concatenate(
        [
            Xp_train,
            Xl_train
        ],
        axis=1
    ).astype(np.float32)

    X_val_full = np.concatenate(
        [
            Xp_val,
            Xl_val
        ],
        axis=1
    ).astype(np.float32)

    X_test_full = np.concatenate(
        [
            Xp_test,
            Xl_test
        ],
        axis=1
    ).astype(np.float32)

    # ========================================================
    # STRATEGY 0
    # ORIGINAL MI TOP-512
    # ========================================================

    if strategy == "mi512":

        print(
            "\nSTRATEGY: MI TOP-512"
        )

        selector = SelectKBest(

            score_func=
                mutual_info_regression,

            k=min(
                K_FEATURES,
                X_train_full.shape[1]
            )
        )

        X_train = selector.fit_transform(

            X_train_full,

            y_train.ravel()

        ).astype(np.float32)

        X_val = selector.transform(

            X_val_full

        ).astype(np.float32)

        X_test = selector.transform(

            X_test_full

        ).astype(np.float32)

        selected = selector.get_support(
            indices=True
        )


    # ========================================================
    # STRATEGY 1
    # BALANCED MI
    #
    # Top 256 protein dimensions
    # Top 256 ligand dimensions
    # ========================================================

    elif strategy == "balanced_mi":

        print(
            "\nSTRATEGY: BALANCED MI 256P + 256L"
        )

        kp = min(
            K_FEATURES // 2,
            protein_dim
        )

        kl = min(
            K_FEATURES // 2,
            ligand_dim
        )

        selector_p = SelectKBest(

            score_func=
                mutual_info_regression,

            k=kp
        )

        selector_l = SelectKBest(

            score_func=
                mutual_info_regression,

            k=kl
        )

        # Protein selection
        P_train = selector_p.fit_transform(

            Xp_train,

            y_train.ravel()

        ).astype(np.float32)

        P_val = selector_p.transform(
            Xp_val
        ).astype(np.float32)

        P_test = selector_p.transform(
            Xp_test
        ).astype(np.float32)

        # Ligand selection
        L_train = selector_l.fit_transform(

            Xl_train,

            y_train.ravel()

        ).astype(np.float32)

        L_val = selector_l.transform(
            Xl_val
        ).astype(np.float32)

        L_test = selector_l.transform(
            Xl_test
        ).astype(np.float32)

        # Recombine
        X_train = np.concatenate(
            [
                P_train,
                L_train
            ],
            axis=1
        ).astype(np.float32)

        X_val = np.concatenate(
            [
                P_val,
                L_val
            ],
            axis=1
        ).astype(np.float32)

        X_test = np.concatenate(
            [
                P_test,
                L_test
            ],
            axis=1
        ).astype(np.float32)

        # Original-space selected indices
        p_idx = selector_p.get_support(
            indices=True
        )

        l_idx = selector_l.get_support(
            indices=True
        )

        selected = np.concatenate(
            [
                p_idx,
                protein_dim + l_idx
            ]
        )


    # ========================================================
    # STRATEGY 2
    # RANDOM FOREST -> SHAP -> TOP-512
    #
    # Same feature-selection concept as revised XGB analysis.
    # RF and SHAP see TRAIN only.
    # ========================================================

    elif strategy == "rf_shap512":

        print(
            "\nSTRATEGY: RF-SHAP TOP-512"
        )

        rf = RandomForestRegressor(

            n_estimators=50,

            random_state=SEED,

            n_jobs=-1
        )

        rf.fit(

            X_train_full,

            y_train.ravel()
        )

        # ----------------------------------------------------
        # SHAP sample from TRAIN only
        # ----------------------------------------------------

        rng = np.random.default_rng(
            SEED
        )

        shap_n = min(
            1000,
            len(X_train_full)
        )

        shap_idx = rng.choice(

            len(X_train_full),

            size=shap_n,

            replace=False
        )

        explainer = shap.TreeExplainer(
            rf
        )

        shap_values = explainer.shap_values(

            X_train_full[
                shap_idx
            ]
        )

        if isinstance(
            shap_values,
            list
        ):

            shap_values = shap_values[0]

        shap_values = np.asarray(
            shap_values
        )

        if (
            shap_values.ndim == 3
            and
            shap_values.shape[-1] == 1
        ):

            shap_values = (
                shap_values[:, :, 0]
            )

        mean_abs_shap = np.mean(

            np.abs(
                shap_values
            ),

            axis=0
        )

        # Highest mean |SHAP|
        selected = np.argsort(
            mean_abs_shap
        )[
            -min(
                K_FEATURES,
                len(mean_abs_shap)
            ):
        ]

        # Preserve original feature order
        selected = np.sort(
            selected
        )

        X_train = X_train_full[
            :, selected
        ].astype(np.float32)

        X_val = X_val_full[
            :, selected
        ].astype(np.float32)

        X_test = X_test_full[
            :, selected
        ].astype(np.float32)


    # ========================================================
    # STRATEGY 3
    # ALL FEATURES
    # No feature selection
    # ========================================================

    elif strategy == "all_features":

        print(
            "\nSTRATEGY: ALL FEATURES"
        )

        X_train = (
            X_train_full
        )

        X_val = (
            X_val_full
        )

        X_test = (
            X_test_full
        )

        selected = np.arange(
            X_train_full.shape[1]
        )


    else:

        raise ValueError(
            f"Unknown strategy: {strategy}"
        )


    # ========================================================
    # FEATURE COMPOSITION
    # ========================================================

    n_protein = int(
        np.sum(
            selected < protein_dim
        )
    )

    n_ligand = int(
        np.sum(
            selected >= protein_dim
        )
    )

    total_selected = (
        n_protein
        + n_ligand
    )

    protein_pct = (
        100.0
        * n_protein
        / total_selected
    )

    ligand_pct = (
        100.0
        * n_ligand
        / total_selected
    )

    print(
        "\nSELECTED FEATURE COMPOSITION"
    )

    print(
        f"Protein: {n_protein} "
        f"({protein_pct:.2f}%)"
    )

    print(
        f"Ligand : {n_ligand} "
        f"({ligand_pct:.2f}%)"
    )

    print(
        "Final feature dimension:",
        X_train.shape[1]
    )

    # --------------------------------------------------------
    # SAVE SELECTED INDICES
    # --------------------------------------------------------

    selected_file = os.path.join(

        OUTDIR,

        f"selected_indices_"
        f"{model_name}_"
        f"{strategy}.npy"
    )

    np.save(
        selected_file,
        selected
    )

    return (
        X_train,
        X_val,
        X_test,
        n_protein,
        n_ligand,
        protein_pct,
        ligand_pct
    )


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

    parser.add_argument(
        "--strategy",
        type=int,
        required=True,
        choices=[0, 1, 2, 3]
    )

    args = parser.parse_args()

    if not 0 <= args.task_id < len(PROTEIN_MAP):

        raise ValueError(
            "task_id must be 0-4"
        )

    set_seed(SEED)

    (
        model_name,
        protein_filename
    ) = PROTEIN_MAP[
        args.task_id
    ]

    strategy = STRATEGIES[
        args.strategy
    ]

    print("\n" + "=" * 70)

    print(
        "MODEL:",
        model_name.upper()
    )

    print(
        "STRATEGY:",
        strategy
    )

    print("=" * 70)


    # ========================================================
    # LOAD AFFINITY DATA
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

    y = (
        y_raw[
            mask
        ]
        .astype(
            np.float32
        )
    )


    # ========================================================
    # LOAD LIGAND EMBEDDINGS
    # ========================================================

    X_lig_raw = np.load(
        LIGAND_FILE
    )

    X_lig = X_lig_raw[
        mask
    ]


    # ========================================================
    # LOAD PROTEIN EMBEDDINGS
    # ========================================================

    X_prot_raw = np.load(

        os.path.join(
            BASE,
            protein_filename
        )
    )

    if (
        X_prot_raw.shape[0]
        == mask.sum()
    ):

        print(
            "Protein embeddings already filtered."
        )

        X_prot = (
            X_prot_raw
        )

    elif (
        X_prot_raw.shape[0]
        == len(mask)
    ):

        X_prot = (
            X_prot_raw[
                mask
            ]
        )

    else:

        raise ValueError(

            "Unexpected protein "
            "embedding row count: "
            f"{X_prot_raw.shape[0]}"
        )

    assert (
        len(X_prot)
        == len(X_lig)
        == len(y)
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
    # LOAD SAVED SEED-42 SPLIT
    # ========================================================

    (
        train_idx,
        val_idx,
        test_idx
    ) = load_saved_split(
        model_name
    )

    print(
        "\nSPLIT SIZES"
    )

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

    max_idx = max(

        np.max(train_idx),

        np.max(val_idx),

        np.max(test_idx)
    )

    if max_idx >= len(y):

        raise ValueError(
            "Saved split indices exceed "
            "filtered dataset size."
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

    scaler_prot = (
        StandardScaler()
    )

    scaler_lig = (
        StandardScaler()
    )

    Xp_train = (
        scaler_prot.fit_transform(
            X_prot[
                train_idx
            ]
        )
    )

    Xp_val = (
        scaler_prot.transform(
            X_prot[
                val_idx
            ]
        )
    )

    Xp_test = (
        scaler_prot.transform(
            X_prot[
                test_idx
            ]
        )
    )

    Xl_train = (
        scaler_lig.fit_transform(
            X_lig[
                train_idx
            ]
        )
    )

    Xl_val = (
        scaler_lig.transform(
            X_lig[
                val_idx
            ]
        )
    )

    Xl_test = (
        scaler_lig.transform(
            X_lig[
                test_idx
            ]
        )
    )


    # ========================================================
    # APPLY FUSION STRATEGY
    # ========================================================

    (
        X_train,
        X_val,
        X_test,
        n_protein,
        n_ligand,
        protein_feature_pct,
        ligand_feature_pct
    ) = apply_strategy(

        strategy,

        Xp_train,
        Xp_val,
        Xp_test,

        Xl_train,
        Xl_val,
        Xl_test,

        y_train,

        model_name
    )


    # ========================================================
    # SAME RAY SEARCH SPACE AS ORIGINAL STUDY
    # ========================================================

    search_space = {

        "hidden_dim":

            tune.choice(
                [
                    256,
                    512,
                    1024
                ]
            ),

        "dropout":

            tune.uniform(
                0.1,
                0.4
            ),

        "lr":

            tune.loguniform(
                1e-4,
                1e-2
            ),

        "batch_size":

            tune.choice(
                [
                    32,
                    64,
                    128
                ]
            )
    }


    # ========================================================
    # SAME ASHA CONFIGURATION
    # ========================================================

    scheduler = ASHAScheduler(

        metric="r2_score",

        mode="max",

        max_t=MAX_EPOCHS,

        grace_period=10,

        reduction_factor=2
    )


    # ========================================================
    # TRAIN + VALIDATION ONLY
    # ========================================================

    data_ref = (

        X_train,

        X_val,

        y_train,

        y_val
    )


    # ========================================================
    # RAY TUNER
    # ========================================================

    tuner = tune.Tuner(

        tune.with_resources(

            tune.with_parameters(

                train_affinity_pipeline,

                data_ref=data_ref
            ),

            resources={

                "cpu": 2,

                "gpu": 1
            }
        ),

        tune_config=tune.TuneConfig(

            scheduler=scheduler,

            num_samples=N_TRIALS,

            max_concurrent_trials=1
        ),

        param_space=search_space
    )


    # ========================================================
    # RUN RAY
    # ========================================================

    results = tuner.fit()


    # ========================================================
    # BEST VALIDATION CONFIGURATION
    # ========================================================

    best_result = (
        results.get_best_result(

            metric="r2_score",

            mode="max"
        )
    )

    best_config = (
        best_result.config
    )

    best_val_r2 = (
        best_result.metrics[
            "r2_score"
        ]
    )

    best_val_rmse = (
        best_result.metrics[
            "rmse"
        ]
    )

    best_val_mae = (
        best_result.metrics[
            "mae"
        ]
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "BEST RAY RESULT"
    )

    print("=" * 70)

    print(
        "Model:",
        model_name
    )

    print(
        "Strategy:",
        strategy
    )

    print(
        f"Validation R2: "
        f"{best_val_r2:.4f}"
    )

    print(
        f"Validation RMSE: "
        f"{best_val_rmse:.4f}"
    )

    print(
        f"Validation MAE: "
        f"{best_val_mae:.4f}"
    )

    print(
        "\nBest hyperparameters:"
    )

    for key, value in (
        best_config.items()
    ):

        print(
            f"{key}: {value}"
        )


    # ========================================================
    # FRESH FINAL MODEL
    # ========================================================

    print(
        "\nTraining fresh final MLP..."
    )

    (
        test_r2,
        test_rmse,
        test_mae
    ) = train_final_model(

        best_config,

        X_train,

        y_train,

        X_test,

        y_test
    )


    # ========================================================
    # FINAL TEST RESULT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL UNTOUCHED TEST"
    )

    print("=" * 70)

    print(
        f"R2   = {test_r2:.4f}"
    )

    print(
        f"RMSE = {test_rmse:.4f}"
    )

    print(
        f"MAE  = {test_mae:.4f}"
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    result = pd.DataFrame([{

        "representation":
            model_name,

        "strategy":
            strategy,

        "n_features":
            X_train.shape[1],

        "protein_features":
            n_protein,

        "ligand_features":
            n_ligand,

        "protein_feature_pct":
            protein_feature_pct,

        "ligand_feature_pct":
            ligand_feature_pct,

        "best_val_r2":
            best_val_r2,

        "best_val_rmse":
            best_val_rmse,

        "best_val_mae":
            best_val_mae,

        "best_hidden_dim":
            best_config[
                "hidden_dim"
            ],

        "best_dropout":
            best_config[
                "dropout"
            ],

        "best_lr":
            best_config[
                "lr"
            ],

        "best_batch_size":
            best_config[
                "batch_size"
            ],

        "test_r2":
            test_r2,

        "test_rmse":
            test_rmse,

        "test_mae":
            test_mae
    }])


    outfile = os.path.join(

        OUTDIR,

        f"{model_name}_"
        f"{strategy}.csv"
    )


    result.to_csv(

        outfile,

        index=False
    )


    print(
        "\nSaved:"
    )

    print(
        outfile
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()