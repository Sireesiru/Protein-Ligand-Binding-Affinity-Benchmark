import os
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (r2_score,mean_squared_error,mean_absolute_error)
import xgboost as xgb
import shap


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BASE = "/lustre/orion/gen006/scratch/sireesiru/miniffinity/all_emb"

AFFINITY_FILE = (
    "/lustre/orion/gen006/scratch/sireesiru/miniffinity/"
    "pdbbind_canonical_affinities.csv"
)

# Point split_file to the SAME split used by the MLP experiments.Use same 512 Top_K as used in MLP.  
SPLIT_FILE = (
    "/lustre/orion/gen006/scratch/sireesiru/miniffinity/"
    "all_emb/XGB/pdbbind_split_seed42_esm2_protein.npz"
)

TOP_K = 512
SHAP_SAMPLES = 1000

# ============================================================
# LOAD AFFINITY DATA
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST + SHAP MODALITY ANALYSIS")
print("=" * 70)
df = pl.read_csv(AFFINITY_FILE)
names = df["name"].to_numpy()
y_raw = df[
    "neg_log10_affinity_M"
].to_numpy()
# ============================================================
# REMOVE SAME PROBLEMATIC ENTRIES
# ============================================================

bad = {
    "3ag9",
    "5dyw"
}

mask = np.array([
    name not in bad
    for name in names
])

names_filtered = names[mask]

y = y_raw[mask].astype(np.float32)

print("\nDATA FILTERING")
print(f"Original samples : {len(names)}")
print(f"After filtering  : {len(names_filtered)}")
print(f"Removed          : {len(names) - len(names_filtered)}")

# ============================================================
# LOAD EXACT SAME 70 / 15 / 15 SPLIT
# ============================================================

print("\nLoading saved train/validation/test split:")
print(SPLIT_FILE)
split_data = np.load(SPLIT_FILE)
train_idx = split_data[
    "train_idx"
]
val_idx = split_data[
    "val_idx"
]
test_idx = split_data[
    "test_idx"
]
print("\nDATA SPLIT")
print(
    f"Train      : {len(train_idx)} "
    f"({len(train_idx)/len(y):.1%})"
)
print(
    f"Validation : {len(val_idx)} "
    f"({len(val_idx)/len(y):.1%})"
)

print(
    f"Test       : {len(test_idx)} "
    f"({len(test_idx)/len(y):.1%})"
)


# ============================================================
# SANITY CHECK SPLIT
# ============================================================

all_idx = np.concatenate([
    train_idx,
    val_idx,
    test_idx
])

assert len(all_idx) == len(y), (
    "Split size does not match filtered dataset."
)

assert len(np.unique(all_idx)) == len(y), (
    "Duplicate indices detected in split."
)

assert set(train_idx).isdisjoint(
    set(val_idx)
)

assert set(train_idx).isdisjoint(
    set(test_idx)
)

assert set(val_idx).isdisjoint(
    set(test_idx)
)

print(
    "Split integrity check: PASSED"
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading embeddings...")

embedding_files = {

    "esm2":
        "esm2_embeddings_mean.npy",

    "prostt5":
        "prostt5_embeddings_mean.npy",

    "progen2":
        "progen2_embeddings.npy",

    "boltz":
        "boltz_mp.npy",

    "protgpt2":
        "pdbbind_protgpt2_embeddings.npy"
}


protein_embeddings = {}

for model_name, filename in embedding_files.items():

    X_raw = np.load(
        os.path.join(
            BASE,
            filename
        )
    )

    # Some embedding arrays may already have
    # the two problematic entries removed.

    if X_raw.shape[0] == mask.sum():

        X = X_raw

        print(
            f"{model_name:10s}: "
            f"already filtered {X.shape}"
        )

    elif X_raw.shape[0] == len(mask):

        X = X_raw[mask]

        print(
            f"{model_name:10s}: "
            f"filtered now {X.shape}"
        )

    else:

        raise ValueError(
            f"Unexpected number of rows for "
            f"{model_name}: {X_raw.shape[0]}"
        )

    protein_embeddings[
        model_name
    ] = X


# ============================================================
# LOAD TXGEMMA LIGAND EMBEDDINGS
# ============================================================

ligand_file = os.path.join(
    BASE,
    "pdbbind_txgemma_ligand_embeddings.npy"
)

X_lig_raw = np.load(
    ligand_file
)

if X_lig_raw.shape[0] == mask.sum():

    X_lig = X_lig_raw

elif X_lig_raw.shape[0] == len(mask):

    X_lig = X_lig_raw[mask]

else:

    raise ValueError(
        "Unexpected TxGemma embedding rows: "
        f"{X_lig_raw.shape[0]}"
    )

print(
    f"txgemma   : {X_lig.shape}"
)

assert len(X_lig) == len(y)


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# MAIN LOOP
#
# ALL FIVE PROTEIN REPRESENTATIONS
# ============================================================

for emb_name in [

    "esm2",
    "prostt5",
    "progen2",
    "boltz",
    "protgpt2"

]:

    print("\n")
    print("=" * 70)

    print(
        f"PROCESSING: {emb_name.upper()} + TXGEMMA"
    )

    print("=" * 70)

    X_prot = protein_embeddings[
        emb_name
    ]

    assert len(X_prot) == len(y)


    # ========================================================
    # SCALE EACH MODALITY SEPARATELY
    #
    # FIT ONLY ON TRAINING SET
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
    # CONCATENATE PROTEIN + LIGAND
    # ========================================================

    X_train = np.concatenate(
        [
            Xp_train,
            Xl_train
        ],
        axis=1
    )

    X_val = np.concatenate(
        [
            Xp_val,
            Xl_val
        ],
        axis=1
    )

    X_test = np.concatenate(
        [
            Xp_test,
            Xl_test
        ],
        axis=1
    )


    prot_dim = Xp_train.shape[1]

    total_dim = X_train.shape[1]

    print("\nEMBEDDING DIMENSIONS")

    print(
        f"Protein dimensions : {prot_dim}"
    )

    print(
        f"Ligand dimensions  : {Xl_train.shape[1]}"
    )

    print(
        f"Combined dimensions: {total_dim}"
    )


    # ========================================================
    # RANDOM FOREST
    #
    # Used ONLY for SHAP feature importance.
    # Training data ONLY.
    # ========================================================

    print(
        "\nTraining Random Forest for "
        "SHAP feature analysis..."
    )

    rf = RandomForestRegressor(

        n_estimators=50,

        n_jobs=-1,

        random_state=SEED
    )

    rf.fit(
        X_train,
        y[train_idx]
    )


    # ========================================================
    # SHAP
    #
    # SHAP calculated ONLY from training samples.
    # ========================================================

    print(
        "Calculating SHAP values..."
    )

    explainer = shap.TreeExplainer(
        rf
    )

    rng = np.random.default_rng(
        SEED
    )

    shap_indices = rng.choice(

        len(X_train),

        size=min(
            SHAP_SAMPLES,
            len(X_train)
        ),

        replace=False
    )


    shap_values = explainer.shap_values(
        X_train[
            shap_indices
        ]
    )


    shap_importance = (
        np.abs(
            shap_values
        )
        .mean(axis=0)
    )


    # ========================================================
    # SELECT TOP FEATURES
    # ========================================================

    actual_k = min(
        TOP_K,
        len(shap_importance)
    )

    top_idx = np.argsort(
        shap_importance
    )[-actual_k:]


    # ========================================================
    # MODALITY CONTRIBUTION
    #
    # Protein features:
    # index < prot_dim
    #
    # Ligand features:
    # index >= prot_dim
    # ========================================================

    protein_mask_top = (
        top_idx < prot_dim
    )

    ligand_mask_top = (
        top_idx >= prot_dim
    )


    # --------------------------------------------------------
    # NUMBER OF SELECTED FEATURES
    # --------------------------------------------------------

    n_protein_features = np.sum(
        protein_mask_top
    )

    n_ligand_features = np.sum(
        ligand_mask_top
    )


    # --------------------------------------------------------
    # SUM SHAP IMPORTANCE
    # --------------------------------------------------------

    protein_shap = np.sum(

        shap_importance[
            top_idx[
                protein_mask_top
            ]
        ]
    )


    ligand_shap = np.sum(

        shap_importance[
            top_idx[
                ligand_mask_top
            ]
        ]
    )


    total_shap = (
        protein_shap +
        ligand_shap
    )


    protein_pct = (
        100.0 *
        protein_shap /
        total_shap
    )


    ligand_pct = (
        100.0 *
        ligand_shap /
        total_shap
    )


    # ========================================================
    # DETERMINE MODALITY BIAS
    # ========================================================

    if protein_pct > ligand_pct:

        dominant_modality = (
            "PROTEIN"
        )

    elif ligand_pct > protein_pct:

        dominant_modality = (
            "LIGAND"
        )

    else:

        dominant_modality = (
            "BALANCED"
        )


    print("\n" + "-" * 70)

    print(
        "SHAP MODALITY CONTRIBUTION"
    )

    print("-" * 70)

    print(
        f"Protein contribution : "
        f"{protein_pct:.2f}%"
    )

    print(
        f"Ligand contribution  : "
        f"{ligand_pct:.2f}%"
    )

    print(
        f"Dominant modality     : "
        f"{dominant_modality}"
    )


    print("\nTOP FEATURE COMPOSITION")

    print(
        f"Protein features : "
        f"{n_protein_features} / "
        f"{actual_k}"
    )

    print(
        f"Ligand features  : "
        f"{n_ligand_features} / "
        f"{actual_k}"
    )


    # ========================================================
    # FINAL XGBOOST MODEL
    #
    # SAME MODEL SETTINGS AS ORIGINAL ANALYSIS
    # ========================================================

    print(
        "\nTraining XGBoost..."
    )

    model = xgb.XGBRegressor(

        n_estimators=300,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=SEED,

        n_jobs=-1
    )


    model.fit(

        X_train[
            :,
            top_idx
        ],

        y[
            train_idx
        ]
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    val_pred = model.predict(

        X_val[
            :,
            top_idx
        ]
    )


    val_r2 = r2_score(
        y[val_idx],
        val_pred
    )


    val_rmse = np.sqrt(

        mean_squared_error(
            y[val_idx],
            val_pred
        )
    )


    val_mae = mean_absolute_error(

        y[val_idx],
        val_pred
    )


    print("\nVALIDATION PERFORMANCE")

    print(
        f"R2   : {val_r2:.4f}"
    )

    print(
        f"RMSE : {val_rmse:.4f}"
    )

    print(
        f"MAE  : {val_mae:.4f}"
    )


    # ========================================================
    # UNTOUCHED TEST SET
    # ========================================================

    test_pred = model.predict(

        X_test[
            :,
            top_idx
        ]
    )


    test_r2 = r2_score(
        y[test_idx],
        test_pred
    )


    test_rmse = np.sqrt(

        mean_squared_error(
            y[test_idx],
            test_pred
        )
    )


    test_mae = mean_absolute_error(

        y[test_idx],
        test_pred
    )


    print("\nFINAL UNTOUCHED TEST PERFORMANCE")

    print(
        f"R2   : {test_r2:.4f}"
    )

    print(
        f"RMSE : {test_rmse:.4f}"
    )

    print(
        f"MAE  : {test_mae:.4f}"
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    results.append({

        "model":
            emb_name,

        "protein_pct":
            protein_pct,

        "ligand_pct":
            ligand_pct,

        "dominant_modality":
            dominant_modality,

        "protein_features":
            int(
                n_protein_features
            ),

        "ligand_features":
            int(
                n_ligand_features
            ),

        "val_r2":
            val_r2,

        "test_r2":
            test_r2,

        "test_rmse":
            test_rmse,

        "test_mae":
            test_mae
    })


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n\n")
print("=" * 100)

print(
    "FINAL XGBOOST + SHAP MODALITY SUMMARY"
)

print("=" * 100)

print(
    f"{'Model':<12}"
    f"{'Protein %':>12}"
    f"{'Ligand %':>12}"
    f"{'Bias':>12}"
    f"{'Prot Feat':>12}"
    f"{'Lig Feat':>12}"
    f"{'Val R2':>10}"
    f"{'Test R2':>10}"
)

print("-" * 100)


for result in results:

    print(

        f"{result['model']:<12}"

        f"{result['protein_pct']:>11.2f}%"

        f"{result['ligand_pct']:>11.2f}%"

        f"{result['dominant_modality']:>12}"

        f"{result['protein_features']:>12}"

        f"{result['ligand_features']:>12}"

        f"{result['val_r2']:>10.4f}"

        f"{result['test_r2']:>10.4f}"
    )


print("=" * 100)


# ============================================================
# SAVE SUMMARY AS CSV
# ============================================================

output_file = (
    "xgb_shap_modality_summary.csv"
)

results_df = pl.DataFrame(
    results
)

results_df.write_csv(
    output_file
)

print(
    f"\nSaved summary to: "
    f"{output_file}"
)

print("\nAnalysis complete.")