import os

# ============================================================
# FRONTIER / RAY SETUP
# ============================================================

os.environ["RAY_memory_monitor_refresh_ms"] = "0"

if "CUDA_VISIBLE_DEVICES" in os.environ and "ROCR_VISIBLE_DEVICES" not in os.environ:
    os.environ["ROCR_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import random
import numpy as np
import polars as pl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import ray
from ray import tune
from ray.tune import report
from ray.tune.schedulers import ASHAScheduler

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# GLOBAL SETTINGS
# ============================================================

SEED = 42
MAX_EPOCHS = 40
K_FEATURES = 512
N_TRIALS = 30

BASE = "/lustre/orion/gen006/scratch/sireesiru/miniffinity/all_emb"

AFFINITY_FILE = (
    "/lustre/orion/gen006/scratch/sireesiru/miniffinity/"
    "pdbbind_canonical_affinities.csv"
)


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

    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):

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
# RAY TRAINING FUNCTION
# ============================================================

def train_affinity_pipeline(config, data_ref):

    set_seed(SEED)

    X_train, X_val, y_train, y_val = data_ref

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Worker running on: {device}")

    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False
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

    # ========================================================
    # TRAIN / VALIDATE
    # ========================================================

    for epoch in range(MAX_EPOCHS):

        model.train()

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions = model(batch_X)

            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():

            for batch_X, batch_y in val_loader:

                predictions = (
                    model(batch_X.to(device))
                    .cpu()
                    .numpy()
                )

                all_preds.append(predictions)
                all_targets.append(batch_y.numpy())

        y_true = np.vstack(all_targets)
        y_pred = np.vstack(all_preds)

        val_r2 = r2_score(y_true, y_pred)

        val_rmse = np.sqrt(
            mean_squared_error(y_true, y_pred)
        )

        val_mae = mean_absolute_error(
            y_true, y_pred
        )

        report({
            "r2_score": val_r2,
            "rmse": val_rmse,
            "mae": val_mae
        })


# ============================================================
# FINAL MODEL
# ============================================================

def train_final_model(
    config,
    X_train,
    y_train,
    X_test,
    y_test,
    seed=SEED
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
    # FINAL TRAINING
    # --------------------------------------------------------

    for epoch in range(MAX_EPOCHS):

        model.train()

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions = model(batch_X)

            loss = criterion(predictions, batch_y)

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

    all_preds = []
    all_targets = []

    with torch.no_grad():

        for batch_X, batch_y in test_loader:

            predictions = (
                model(batch_X.to(device))
                .cpu()
                .numpy()
            )

            all_preds.append(predictions)
            all_targets.append(batch_y.numpy())

    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)

    test_r2 = r2_score(y_true, y_pred)

    test_rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    test_mae = mean_absolute_error(
        y_true, y_pred
    )

    return test_r2, test_rmse, test_mae


# ============================================================
# MAIN
# ============================================================

def run_production_tuning():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="Protein model index: 0-4"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["ligand", "protein", "combined"],
        help="Input modality to evaluate"
    )

    args = parser.parse_args()

    set_seed(SEED)

    # ========================================================
    # INITIALIZE RAY
    # ========================================================

    if not ray.is_initialized():

        ray.init(
            num_cpus=16,
            num_gpus=1,
            ignore_reinit_error=True
        )

    # ========================================================
    # LOAD AFFINITY DATA
    # ========================================================

    df = pl.read_csv(AFFINITY_FILE)

    names = df["name"].to_numpy()

    y_raw = df[
        "neg_log10_affinity_M"
    ].to_numpy()

    # Same exclusions as previous experiment
    bad = {
        "3ag9",
        "5dyw"
    }

    mask = np.array([
        name not in bad
        for name in names
    ])

    names_filtered = names[mask]
    y_filtered = y_raw[mask]

    print("\nDATA FILTERING")
    print(f"Original samples : {len(names)}")
    print(f"After filtering  : {len(names_filtered)}")
    print(f"Removed          : {len(names) - len(names_filtered)}")


    # ========================================================
    # PROTEIN MODELS
    # ========================================================

    protein_map = [
        ("esm2", "esm2_embeddings_mean.npy"),
        ("prostt5", "prostt5_embeddings_mean.npy"),
        ("progen2", "progen2_embeddings.npy"),
        ("boltz", "boltz_mp.npy"),
        ("protgpt2", "pdbbind_protgpt2_embeddings.npy")
    ]

    model_name, filename = protein_map[
        args.task_id
    ]

    mode = args.mode

    print("\n" + "=" * 60)

    print(
        f"MODEL: {model_name.upper()} | "
        f"MODE: {mode.upper()}"
    )

    print("=" * 60)


    # ========================================================
    # LOAD EMBEDDINGS
    # ========================================================

    X_lig = np.load(
        os.path.join(
            BASE,
            "pdbbind_txgemma_ligand_embeddings.npy"
        )
    )[mask]

    X_prot_raw = np.load(
        os.path.join(BASE, filename)
    )

    if X_prot_raw.shape[0] == mask.sum():

        print(
            f"{model_name} embeddings already "
            f"pre-filtered ({X_prot_raw.shape[0]} rows)"
        )

        X_prot = X_prot_raw

    else:

        X_prot = X_prot_raw[mask]

    assert len(X_prot) == len(X_lig)
    assert len(X_prot) == len(y_filtered)

    print(
        f"Protein shape : {X_prot.shape}"
    )

    print(
        f"Ligand shape  : {X_lig.shape}"
    )


    # ========================================================
    # SPLIT
    # EXACT SAME 70 / 15 / 15 SPLIT
    # ========================================================

    split1 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.70,
        random_state=SEED
    )

    train_idx, temp_idx = next(
        split1.split(
            X_prot,
            y_filtered,
            groups=names_filtered
        )
    )

    split2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=SEED
    )

    val_rel_idx, test_rel_idx = next(
        split2.split(
            X_prot[temp_idx],
            y_filtered[temp_idx],
            groups=names_filtered[temp_idx]
        )
    )

    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]
    print("\nDATA SPLIT")
    print(
        f"Train      : {len(train_idx)} "
        f"({len(train_idx)/len(X_prot):.1%})"
    )

    print(
        f"Validation : {len(val_idx)} "
        f"({len(val_idx)/len(X_prot):.1%})"
    )

    print(
        f"Test       : {len(test_idx)} "
        f"({len(test_idx)/len(X_prot):.1%})"
    )
    
    np.savez(f"pdbbind_split_seed42_{model_name}_{mode}.npz",train_idx=train_idx,val_idx=val_idx,test_idx=test_idx)
    # ========================================================
    # TARGETS
    # ========================================================

    y_train = (
        y_filtered[train_idx]
        .astype(np.float32)
        .reshape(-1, 1)
    )

    y_val = (
        y_filtered[val_idx]
        .astype(np.float32)
        .reshape(-1, 1)
    )

    y_test = (
        y_filtered[test_idx]
        .astype(np.float32)
        .reshape(-1, 1)
    )


    # ========================================================
    # SCALE PROTEIN AND LIGAND SEPARATELY
    # ========================================================

    scaler_prot = StandardScaler()
    scaler_lig = StandardScaler()

    Xp_train = scaler_prot.fit_transform(
        X_prot[train_idx]
    )

    Xp_val = scaler_prot.transform(
        X_prot[val_idx]
    )

    Xp_test = scaler_prot.transform(
        X_prot[test_idx]
    )

    Xl_train = scaler_lig.fit_transform(
        X_lig[train_idx]
    )

    Xl_val = scaler_lig.transform(
        X_lig[val_idx]
    )

    Xl_test = scaler_lig.transform(
        X_lig[test_idx]
    )


    # ========================================================
    # CHOOSE MODALITY
    # ========================================================

    if mode == "protein":

        X_train_full = Xp_train
        X_val_full = Xp_val
        X_test_full = Xp_test

    elif mode == "ligand":

        X_train_full = Xl_train
        X_val_full = Xl_val
        X_test_full = Xl_test

    elif mode == "combined":

        X_train_full = np.concatenate(
            [Xp_train, Xl_train],
            axis=1
        )

        X_val_full = np.concatenate(
            [Xp_val, Xl_val],
            axis=1
        )

        X_test_full = np.concatenate(
            [Xp_test, Xl_test],
            axis=1
        )

    X_train_full = X_train_full.astype(
        np.float32
    )

    X_val_full = X_val_full.astype(
        np.float32
    )

    X_test_full = X_test_full.astype(
        np.float32
    )

    print("\nINPUT REPRESENTATION")

    print(
        "Mode:",
        mode
    )

    print(
        "Before feature selection:",
        X_train_full.shape
    )


    # ========================================================
    # MUTUAL INFORMATION FEATURE SELECTION
    #
    # Same target K=512.
    #
    # If the representation has <512 dimensions,
    # use all available dimensions.
    # ========================================================

    actual_k = min(
        K_FEATURES,
        X_train_full.shape[1]
    )

    print(
        f"Selecting {actual_k} features..."
    )

    if actual_k < X_train_full.shape[1]:

        selector = SelectKBest(
            score_func=mutual_info_regression,
            k=actual_k
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

        np.save(
            f"selected_indices_{model_name}_{mode}.npy",
            selector.get_support(indices=True)
        )

    else:

        # No feature selection needed because the
        # representation already has <=512 dimensions.

        X_train = X_train_full
        X_val = X_val_full
        X_test = X_test_full

        np.save(
            f"selected_indices_{model_name}_{mode}.npy",
            np.arange(X_train_full.shape[1])
        )

    print(
        "Final input shape:",
        X_train.shape
    )


    # ========================================================
    # SEND TRAIN + VALIDATION TO RAY
    # ========================================================

    data_ref = ray.put(
        (
            X_train,
            X_val,
            y_train,
            y_val
        )
    )


    # ========================================================
    # SAME HYPERPARAMETER SPACE
    # ========================================================

    search_space = {

        "hidden_dim": tune.choice(
            [256, 512, 1024]
        ),

        "dropout": tune.uniform(
            0.1, 0.4
        ),

        "lr": tune.loguniform(
            1e-4, 1e-2
        ),

        "batch_size": tune.choice(
            [32, 64, 128]
        )
    }


    # ========================================================
    # ASHA
    # ========================================================

    scheduler = ASHAScheduler(
        metric="r2_score",
        mode="max",
        max_t=MAX_EPOCHS,
        grace_period=10,
        reduction_factor=2
    )


    # ========================================================
    # RAY TUNING
    #
    # IMPORTANT:
    # metric/mode are defined in ASHA only.
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
    # RUN TUNING
    # ========================================================

    results = tuner.fit()


    # ========================================================
    # BEST VALIDATION RESULT
    # ========================================================

    best_result = results.get_best_result(
        metric="r2_score",
        mode="max"
    )

    best_config = best_result.config

    print("\n" + "=" * 60)

    print(
        f"BEST VALIDATION RESULT"
    )

    print(
        f"MODEL : {model_name.upper()}"
    )

    print(
        f"MODE  : {mode.upper()}"
    )

    print("=" * 60)

    print(
        f"Validation R2   : "
        f"{best_result.metrics['r2_score']:.4f}"
    )

    print(
        f"Validation RMSE : "
        f"{best_result.metrics['rmse']:.4f}"
    )

    print(
        f"Validation MAE  : "
        f"{best_result.metrics['mae']:.4f}"
    )

    print("\nBest hyperparameters:")

    for key, value in best_config.items():
        print(f"{key}: {value}")


    # ========================================================
    # FINAL MODEL ? UNTOUCHED TEST
    # ========================================================

    print(
        "\nTraining final model..."
    )

    test_r2, test_rmse, test_mae = train_final_model(

        best_config,

        X_train,
        y_train,

        X_test,
        y_test,

        seed=SEED
    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL UNTOUCHED TEST RESULT")

    print(
        f"Protein model : {model_name}"
    )

    print(
        f"Input mode    : {mode}"
    )

    print(
        f"Features      : {X_train.shape[1]}"
    )

    print("-" * 60)

    print(
        f"Test R2   : {test_r2:.4f}"
    )

    print(
        f"Test RMSE : {test_rmse:.4f}"
    )

    print(
        f"Test MAE  : {test_mae:.4f}"
    )

    print("=" * 60)

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_production_tuning()