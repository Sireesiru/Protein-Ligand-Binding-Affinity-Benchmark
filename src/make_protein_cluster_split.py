import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42

CLUSTER_FILE = Path("pdbbind_cluster30_cluster.tsv")

AFFINITY_FILE = Path(
    "/lustre/orion/gen006/scratch/sireesiru/miniffinity/"
    "pdbbind_canonical_affinities.csv"
)

OUTPUT_FILE = Path("pdbbind_split_cluster30_seed42.npz")
OUTPUT_CSV = Path("pdbbind_split_cluster30_seed42.csv")


# ------------------------------------------------------------
# Load the same filtered PDBbind dataset used in the benchmark
# ------------------------------------------------------------

df = pd.read_csv(AFFINITY_FILE)

bad = {"3ag9", "5dyw"}

mask = ~df["name"].astype(str).str.lower().isin(bad)
df = df.loc[mask].reset_index(drop=True)

df["pdb_id"] = df["name"].astype(str).str.lower()

print("Filtered complexes:", len(df))

if len(df) != 19116:
    raise ValueError(f"Expected 19116 complexes, found {len(df)}")


# ------------------------------------------------------------
# Load MMseqs clusters
# ------------------------------------------------------------

clusters = pd.read_csv(
    CLUSTER_FILE,
    sep="\t",
    header=None,
    names=["cluster_id", "member"]
)

clusters["pdb_id"] = (
    clusters["member"]
    .str.split("|")
    .str[-1]
    .str.lower()
)

print("MMseqs assignments:", len(clusters))
print("Unique clusters:", clusters["cluster_id"].nunique())
print("Unique PDB IDs:", clusters["pdb_id"].nunique())


# ------------------------------------------------------------
# Match clusters to benchmark rows
# ------------------------------------------------------------

mapping = clusters[["pdb_id", "cluster_id"]]

merged = df.merge(
    mapping,
    on="pdb_id",
    how="left",
    validate="one_to_one"
)

if merged["cluster_id"].isna().any():
    missing = merged.loc[
        merged["cluster_id"].isna(), "pdb_id"
    ].tolist()

    raise ValueError(
        f"{len(missing)} benchmark complexes have no cluster: "
        f"{missing[:10]}"
    )

print("All benchmark complexes mapped to clusters.")


# ------------------------------------------------------------
# Cluster sizes
# ------------------------------------------------------------

cluster_sizes = (
    merged.groupby("cluster_id")
    .size()
    .sort_values(ascending=False)
)

print("\nLargest clusters:")
print(cluster_sizes.head(10))


# ------------------------------------------------------------
# Target sizes
# ------------------------------------------------------------

N = len(merged)

targets = {
    "train": round(0.70 * N),
    "val": round(0.15 * N),
}

targets["test"] = N - targets["train"] - targets["val"]

print("\nTargets:", targets)


# ------------------------------------------------------------
# Size-balanced greedy assignment
#
# Large clusters are assigned first.
# At each step assign to the split with the greatest
# remaining capacity relative to its target.
# ------------------------------------------------------------

rng = np.random.default_rng(SEED)

# Randomize equal-size clusters reproducibly
cluster_table = cluster_sizes.reset_index()
cluster_table.columns = ["cluster_id", "size"]

cluster_table["_random"] = rng.random(len(cluster_table))

cluster_table = cluster_table.sort_values(
    ["size", "_random"],
    ascending=[False, True]
).reset_index(drop=True)


current = {
    "train": 0,
    "val": 0,
    "test": 0
}

assignment = {}


for _, row in cluster_table.iterrows():

    cluster_id = row["cluster_id"]
    size = int(row["size"])

    # Fraction of target still unfilled
    remaining_fraction = {
        split:
            (targets[split] - current[split]) / targets[split]
        for split in current
    }

    destination = max(
        remaining_fraction,
        key=remaining_fraction.get
    )

    assignment[cluster_id] = destination
    current[destination] += size


print("\nActual split sizes:")
for split in ["train", "val", "test"]:
    print(
        f"{split:5s}: "
        f"{current[split]:5d} "
        f"(target {targets[split]})"
    )


# ------------------------------------------------------------
# Assign every complex
# ------------------------------------------------------------

merged["split"] = merged["cluster_id"].map(assignment)


# ------------------------------------------------------------
# Leakage check
# ------------------------------------------------------------

train_clusters = set(
    merged.loc[merged["split"] == "train", "cluster_id"]
)

val_clusters = set(
    merged.loc[merged["split"] == "val", "cluster_id"]
)

test_clusters = set(
    merged.loc[merged["split"] == "test", "cluster_id"]
)

assert train_clusters.isdisjoint(val_clusters)
assert train_clusters.isdisjoint(test_clusters)
assert val_clusters.isdisjoint(test_clusters)

print("\nCluster leakage check: PASSED")


# ------------------------------------------------------------
# Recover indices corresponding to the filtered 19,116-row
# benchmark arrays
# ------------------------------------------------------------

train_idx = merged.index[
    merged["split"] == "train"
].to_numpy(dtype=int)

val_idx = merged.index[
    merged["split"] == "val"
].to_numpy(dtype=int)

test_idx = merged.index[
    merged["split"] == "test"
].to_numpy(dtype=int)


# ------------------------------------------------------------
# Index leakage / coverage checks
# ------------------------------------------------------------

assert len(
    np.intersect1d(train_idx, val_idx)
) == 0

assert len(
    np.intersect1d(train_idx, test_idx)
) == 0

assert len(
    np.intersect1d(val_idx, test_idx)
) == 0

all_indices = np.concatenate(
    [train_idx, val_idx, test_idx]
)

assert len(np.unique(all_indices)) == N

print("Index leakage check: PASSED")
print("All complexes assigned: PASSED")


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

np.savez(
    OUTPUT_FILE,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx
)

merged[
    [
        "pdb_id",
        "cluster_id",
        "split",
        "neg_log10_affinity_M"
    ]
].to_csv(
    OUTPUT_CSV,
    index=False
)


print("\nSaved:")
print(OUTPUT_FILE)
print(OUTPUT_CSV)

print("\nFinal:")
print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))
