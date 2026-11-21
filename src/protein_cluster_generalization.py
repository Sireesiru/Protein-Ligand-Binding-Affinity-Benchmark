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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session


# ============================================================
# GLOBAL SETTINGS
# ============================================================

SEED = 42

MAX_EPOCHS = 40
N_TRIALS = 30
SMOKE_TEST = False

BASE = (
    "/lustre/orion/gen006/scratch/sireesiru/"
    "miniffinity/all_emb"
)

WORK = os.path.join(BASE, "XGB")

AFFINITY_FILE = (
    "/lustre/orion/gen006/scratch/sireesiru/miniffinity/"
    "pdbbind_canonical_affinities.csv"
)

LIGAND_FILE = os.path.join(
    BASE,
    "pdbbind_txgemma_ligand_embeddings.npy"
)

# Frozen 30%-identity MMseqs cluster-disjoint split
SPLIT_FILE = os.path.join(
    WORK,
    "protein_clustering",
    "pdbbind_split_cluster30_seed42.npz"
)

# Keep generalization results separate from original fusion results
OUTDIR = os.path.join(
    WORK,
    "protein_cluster_generalization"
)

os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# PROTEIN REPRESENTATIONS
# ============================================================

PROTEIN_MAP = [
    ("esm2", "esm2_embeddings_mean.npy"),
    ("prostt5", "prostt5_embeddings_mean.npy"),
    ("progen2", "progen2_embeddings.npy"),
    ("boltz", "boltz_mp.npy"),
    ("protgpt2", "pdbbind_protgpt2_embeddings.npy"),
]


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
# Same architecture as original MLP benchmark
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
# LOAD FROZEN PROTEIN-CLUSTER SPLIT
# ============================================================

def load_cluster_split():

    if not os.path.exists(SPLIT_FILE):

        raise FileNotFoundError(
            f"Cluster split not found:\n{SPLIT_FILE}"
        )

    split = np.load(
        SPLIT_FILE,
        allow_pickle=True
    )

    required = {
        "train_idx",
        "val_idx",
        "test_idx"
    }

    if not required.issubset(split.files):

        raise KeyError(
            "Expected train_idx, val_idx, test_idx. "
            f"Found: {split.files}"
        )

    train_idx = np.asarray(
        split["train_idx"],
        dtype=int
    )

    val_idx = np.asarray(
        split["val_idx"],
        dtype=int
    )

    test_idx = np.asarray(
        split["test_idx"],
        dtype=int
    )

    print("\nUsing frozen protein-cluster split:")
    print(SPLIT_FILE)

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

    return (
        train_idx,
        val_idx,
        test_idx
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
# TRAIN FRESH FINAL MODEL
#
# Preserve original benchmark behavior:
# final model is trained on TRAIN only.
# TEST remains untouched until final evaluation.
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
    # FINAL TEST
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
# MAIN
# ============================================================

def main():

    global MAX_EPOCHS
    global N_TRIALS
    global SMOKE_TEST

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_id",
        type=int,
        required=True,
        help=(
            "Protein representation: "
            "0=ESM2, 1=ProstT5, 2=ProGen2, "
            "3=Boltz, 4=ProtGPT2"
        )
    )

    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help=(
            "Run small end-to-end test before "
            "full SLURM submission"
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # SMOKE TEST SETTINGS
    # --------------------------------------------------------

    SMOKE_TEST = args.smoke_test

    if SMOKE_TEST:

        MAX_EPOCHS = 2
        N_TRIALS = 1

        print("\n" + "=" * 70)
        print("*** SMOKE TEST MODE ***")
        print("300 train / 100 validation / 100 test")
        print("1 Ray trial / 2 epochs")
        print("=" * 70)

    else:

        print("\n*** PRODUCTION MODE ***")
        print(
            f"Ray trials: {N_TRIALS}"
        )
        print(
            f"Training epochs: {MAX_EPOCHS}"
        )

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

    print("\n" + "=" * 70)

    print(
        "PROTEIN REPRESENTATION:",
        model_name.upper()
    )

    print(
        "GENERALIZATION SPLIT:",
        "30% MMseqs protein-cluster-disjoint"
    )

    print(
        "FUSION:",
        "ALL FEATURES"
    )

    print("=" * 70)

    # ========================================================
    # CHECK INPUT FILES
    # ========================================================

    protein_file = os.path.join(
        BASE,
        protein_filename
    )

    required_files = {

        "Affinity file":
            AFFINITY_FILE,

        "Ligand embeddings":
            LIGAND_FILE,

        "Protein embeddings":
            protein_file,

        "Cluster split":
            SPLIT_FILE,
    }

    print(
        "\nChecking required input files..."
    )

    for label, path in required_files.items():

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"{label} not found:\n{path}"
            )

        print(
            f"OK: {label}"
        )

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

        str(name).lower() not in bad

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

    print(
        "\nFiltered affinity samples:",
        len(y)
    )

    if len(y) != 19116:

        raise ValueError(
            f"Expected 19116 filtered complexes, "
            f"found {len(y)}"
        )

    # ========================================================
    # LOAD LIGAND EMBEDDINGS
    # ========================================================

    print(
        "\nLoading ligand embeddings..."
    )

    X_lig_raw = np.load(
        LIGAND_FILE
    )

    if X_lig_raw.shape[0] == len(mask):

        X_lig = X_lig_raw[
            mask
        ]

    elif X_lig_raw.shape[0] == mask.sum():

        print(
            "Ligand embeddings already filtered."
        )

        X_lig = X_lig_raw

    else:

        raise ValueError(
            "Unexpected ligand embedding row count: "
            f"{X_lig_raw.shape[0]}"
        )

    # ========================================================
    # LOAD PROTEIN EMBEDDINGS
    # ========================================================

    print(
        "Loading protein embeddings..."
    )

    X_prot_raw = np.load(
        protein_file
    )

    if (
        X_prot_raw.shape[0]
        == mask.sum()
    ):

        print(
            "Protein embeddings already filtered."
        )

        X_prot = X_prot_raw

    elif (
        X_prot_raw.shape[0]
        == len(mask)
    ):

        X_prot = X_prot_raw[
            mask
        ]

    else:

        raise ValueError(
            "Unexpected protein embedding row count: "
            f"{X_prot_raw.shape[0]}"
        )

    if not (
        len(X_prot)
        == len(X_lig)
        == len(y)
        == 19116
    ):

        raise ValueError(
            "Protein, ligand, and affinity row counts "
            "do not match after filtering."
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
    # LOAD FROZEN CLUSTER-DISJOINT SPLIT
    # ========================================================

    (
        train_idx,
        val_idx,
        test_idx
    ) = load_cluster_split()

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if not (
        len(train_idx) == 13381
        and len(val_idx) == 2867
        and len(test_idx) == 2868
    ):

        raise ValueError(
            "Unexpected frozen split sizes."
        )

    if (
        len(np.intersect1d(train_idx, val_idx)) > 0
        or len(np.intersect1d(train_idx, test_idx)) > 0
        or len(np.intersect1d(val_idx, test_idx)) > 0
    ):

        raise ValueError(
            "Index leakage detected between splits."
        )

    all_idx = np.concatenate(
        [
            train_idx,
            val_idx,
            test_idx
        ]
    )

    if len(np.unique(all_idx)) != len(y):

        raise ValueError(
            "Frozen split does not cover the full dataset."
        )

    if (
        np.min(all_idx) < 0
        or np.max(all_idx) >= len(y)
    ):

        raise ValueError(
            "Frozen split contains invalid indices."
        )

    print(
        "Frozen split sanity checks: PASSED"
    )

    # ========================================================
    # SMOKE TEST SUBSAMPLING
    #
    # Preserve cluster partition boundaries.
    # This is only an execution test, not a scientific result.
    # ========================================================

    if SMOKE_TEST:

        rng = np.random.default_rng(
            SEED
        )

        train_idx = np.sort(
            rng.choice(
                train_idx,
                size=min(
                    300,
                    len(train_idx)
                ),
                replace=False
            )
        )

        val_idx = np.sort(
            rng.choice(
                val_idx,
                size=min(
                    100,
                    len(val_idx)
                ),
                replace=False
            )
        )

        test_idx = np.sort(
            rng.choice(
                test_idx,
                size=min(
                    100,
                    len(test_idx)
                ),
                replace=False
            )
        )

        print(
            "\nSMOKE TEST SPLIT SIZES"
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

    # ========================================================
    # TARGET ARRAYS
    # ========================================================

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
    #
    # Protein and ligand representations are standardized
    # separately exactly as in the original benchmark.
    # ========================================================

    scaler_prot = StandardScaler()
    scaler_lig = StandardScaler()

    Xp_train = scaler_prot.fit_transform(
        X_prot[
            train_idx
        ]
    )

    Xp_val = scaler_prot.transform(
        X_prot[
            val_idx
        ]
    )

    Xp_test = scaler_prot.transform(
        X_prot[
            test_idx
        ]
    )

    Xl_train = scaler_lig.fit_transform(
        X_lig[
            train_idx
        ]
    )

    Xl_val = scaler_lig.transform(
        X_lig[
            val_idx
        ]
    )

    Xl_test = scaler_lig.transform(
        X_lig[
            test_idx
        ]
    )

    print(
        "\nScaling completed successfully."
    )

    # ========================================================
    # ALL-FEATURE FUSION
    #
    # No feature selection.
    # Protein first, ligand second.
    # ========================================================

    X_train = np.concatenate(
        [
            Xp_train,
            Xl_train
        ],
        axis=1
    ).astype(
        np.float32
    )

    X_val = np.concatenate(
        [
            Xp_val,
            Xl_val
        ],
        axis=1
    ).astype(
        np.float32
    )

    X_test = np.concatenate(
        [
            Xp_test,
            Xl_test
        ],
        axis=1
    ).astype(
        np.float32
    )

    protein_dim = Xp_train.shape[1]
    ligand_dim = Xl_train.shape[1]

    print(
        "\nALL-FEATURE FUSION"
    )

    print(
        "Protein features:",
        protein_dim
    )

    print(
        "Ligand features:",
        ligand_dim
    )

    print(
        "Total features:",
        X_train.shape[1]
    )

    print(
        "Train shape:",
        X_train.shape
    )

    print(
        "Validation shape:",
        X_val.shape
    )

    print(
        "Test shape:",
        X_test.shape
    )

    # ========================================================
    # SAME RAY SEARCH SPACE AS ORIGINAL BENCHMARK
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
    # SAME ASHA SETTINGS
    # ========================================================

    scheduler = ASHAScheduler(

        metric="r2_score",

        mode="max",

        max_t=MAX_EPOCHS,

        grace_period=(
            1 if SMOKE_TEST else 10
        ),

        reduction_factor=2
    )

    data_ref = (
        X_train,
        X_val,
        y_train,
        y_val
    )

    # ========================================================
    # RAY TUNER
    # ========================================================

    print(
        "\nStarting Ray..."
    )

    tuner = tune.Tuner(

        tune.with_resources(

            tune.with_parameters(

                train_affinity_pipeline,

                data_ref=data_ref
            ),

            resources={
                "cpu": 2
            }
        ),

        tune_config=tune.TuneConfig(

            scheduler=scheduler,

            num_samples=N_TRIALS,

            max_concurrent_trials=1
        ),

        param_space=search_space
    )

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

    best_config = best_result.config

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

    print(
        "=" * 70
    )

    print(
        "Representation:",
        model_name
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

    for key, value in best_config.items():

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
    # FINAL CLUSTER-DISJOINT TEST RESULT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL 30% PROTEIN-CLUSTER-DISJOINT TEST"
        if not SMOKE_TEST
        else "SMOKE TEST FINAL EVALUATION"
    )

    print(
        "=" * 70
    )

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

        "split":
            "mmseqs30_cluster_disjoint",

        "seed":
            SEED,

        "train_n":
            len(train_idx),

        "val_n":
            len(val_idx),

        "test_n":
            len(test_idx),

        "protein_features":
            protein_dim,

        "ligand_features":
            ligand_dim,

        "total_features":
            X_train.shape[1],

        "fusion":
            "all_features",

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

    if SMOKE_TEST:

        outfile = os.path.join(
            OUTDIR,
            f"TEST_{model_name}_cluster30_all_features.csv"
        )

    else:

        outfile = os.path.join(
            OUTDIR,
            f"{model_name}_cluster30_all_features.csv"
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

    if SMOKE_TEST:

        print(
            "\n" + "=" * 70
        )

        print(
            "*** SMOKE TEST COMPLETED SUCCESSFULLY ***"
        )

        print(
            "load -> frozen cluster split -> "
            "train-only scaling -> all-feature fusion -> "
            "Ray -> final MLP -> evaluation -> save"
        )

        print(
            "=" * 70
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
