# Protein–Ligand Binding Affinity Representation Benchmark

This repository benchmarks how different pretrained protein representations contribute to protein–ligand binding-affinity prediction when paired with a fixed ligand representation.

The central question is:

> **How does the choice of pretrained protein representation affect binding-affinity prediction, particularly its ability to complement a fixed ligand representation?**

Rather than treating pretrained embeddings as interchangeable feature sources, the study evaluates their standalone predictive information, incremental value in multimodal prediction, interaction with feature-selection and downstream learning strategies, and robustness across repeated train/validation/test splits.

## Representations

### Protein representations

Five protein representations are compared:

- **ESM2** — protein sequence representation
- **ProstT5** — sequence/structure-aware protein representation
- **ProGen2** — generative protein language-model representation
- **ProtGPT2** — generative protein language-model representation
- **Boltz** — structure-derived protein representation

### Ligand representation

- **TxGemma** — fixed ligand representation used across all protein benchmarks

Keeping the ligand representation fixed allows differences in multimodal performance to be associated with the choice of protein representation and its compatibility with the ligand representation.

## Experimental Design

### 1. Protein and ligand ablation

Protein-only and ligand-only models quantify the predictive information available from each modality independently.

### 2. Protein + ligand prediction

Protein and TxGemma ligand representations are combined to determine whether each protein representation improves prediction beyond the ligand representation alone.

Incremental contribution is measured as:

**ΔR² = R²(protein + ligand) − R²(ligand)**

A strong protein-only representation therefore does not necessarily imply strong multimodal complementarity.

### 3. Repeated-split robustness

Protein-only, ligand-only, and combined models are evaluated across 10 random seeds to determine whether representation differences are robust to train/validation/test partitioning.

### 4. RF-SHAP feature selection + XGBoost

Random Forest SHAP values are used to rank features from the concatenated protein–ligand representation. The top 512 features are then used for XGBoost affinity prediction.

SHAP values are aggregated by modality to characterize model attribution to protein versus ligand features.

These percentages describe **feature attribution/model reliance**, not the fraction of R² explained by each modality.

### 5. MLP modality attribution

MLP-based SHAP analysis examines protein-versus-ligand attribution under the neural-network prediction pipeline.

*Results are currently pending.*

### 6. Fusion-strategy benchmark

Additional experiments compare:

- mutual-information selection from concatenated features
- modality-balanced mutual-information selection
- RF-SHAP feature selection
- all-feature fusion

*Results are currently pending.*

## Current Results

Repeated-seed experiments show clear differences in how effectively protein representations complement the fixed ligand representation.

| Protein representation | Protein R² | Ligand R² | Combined R² | ΔR² |
|---|---:|---:|---:|---:|
| Boltz | 0.323 | 0.395 | **0.505** | **+0.110** |
| ProstT5 | 0.343 | 0.377 | **0.471** | **+0.095** |
| ProtGPT2 | 0.342 | 0.376 | **0.458** | **+0.083** |
| ESM2 | 0.315 | 0.382 | **0.444** | **+0.062** |
| ProGen2 | 0.154 | 0.376 | **0.150** | **−0.225** |

Values are means across 10 random splits.

The results demonstrate that **unimodal representation quality and multimodal utility are distinct properties**. Boltz, ProstT5, ProtGPT2, and ESM2 provide positive incremental information when combined with TxGemma, whereas ProGen2 reduces predictive performance under the evaluated MLP pipeline.

## RF-SHAP + XGBoost Results

| Protein representation | Protein attribution | Ligand attribution | Test R² |
|---|---:|---:|---:|
| ESM2 | 33.0% | 67.0% | 0.522 |
| ProstT5 | 49.9% | 50.1% | **0.552** |
| ProGen2 | 11.5% | 88.5% | 0.497 |
| Boltz | 42.2% | 57.8% | 0.543 |
| ProtGPT2 | 45.7% | 54.3% | 0.547 |

These attribution values are obtained from the Random Forest SHAP feature-analysis stage preceding XGBoost and should not be interpreted as a decomposition of predictive R².

## Repository Structure

```text
protein-ligand-binding-affinity-benchmark/
├── README.md
├── data/
│   └── local PDBbind-derived input tables (not tracked)
├── embeddings/
│   └── generated embedding matrices (not tracked)
├── embedding_generation/
│   ├── TxGemma_embedings.ipynb
│   ├── protgpt2_embedding_generation.py
│   ├── worker_decoder_progen2.py
│   ├── worker_encoder_ESM2.py
│   └── worker_encoder_prostT5.py
├── src/
│   ├── FS_ray_tuning_ablation.py
│   ├── FS_ray_tuning_modified.py
│   ├── repeated_seed_mlp.py
│   ├── revised_pipeline_shap_xgb.py
│   ├── mlp_modality_shap.py
│   └── mlp_fusion_benchmark.py
├── slurm/
│   └── Frontier SLURM launch scripts
├── splits/
│   └── saved train/validation/test split indices
└── results/
    ├── ablation/
    ├── combined/
    ├── repeated_seeds/
    ├── xgb_shap/
    ├── modality_shap/
    └── fusion/
```

## Data Preparation

The benchmark uses protein–ligand complexes and affinity measurements derived from **PDBbind 2020**.

The PDBbind-derived source tables are not distributed through this repository. Users should obtain the appropriate PDBbind data independently under the applicable access and usage terms.

The modeling workflows use two prepared input tables:

```text
data/
├── pdbbind_canonical_affinities.csv
└── pdbbind_sequences.csv
```

The affinity table contains the complex identifiers, ligand information including canonical SMILES, and binding-affinity values used for modeling. The sequence table provides the corresponding protein sequences.

The final modeling dataset contains 19,116 complexes after exclusion of two problematic entries (`3ag9` and `5dyw`) from the original 19,118-row dataset.

## Embedding Generation

Large precomputed embedding matrices are not distributed through Git. Instead, the code used to generate the representations is provided in `embedding_generation/`.

| Representation | Modality | Generation code |
|---|---|---|
| ESM2 | Protein | `worker_encoder_ESM2.py` |
| ProstT5 | Protein | `worker_encoder_prostT5.py` |
| ProGen2 | Protein | `worker_decoder_progen2.py` |
| ProtGPT2 | Protein | `protgpt2_embedding_generation.py` |
| TxGemma | Ligand | `TxGemma_embedings.ipynb` |
| Boltz | Protein | Precomputed representation; provenance to be documented |

The corresponding HPC launch scripts are provided in `slurm/`.

The TxGemma notebook is the original notebook used to generate the ligand representations used in this benchmark.

### Expected generated embeddings

After generation, the modeling workflows expect:

```text
embeddings/
├── boltz_mp.npy
├── esm2_embeddings_mean.npy
├── pdbbind_protgpt2_embeddings.npy
├── pdbbind_txgemma_ligand_embeddings.npy
├── progen2_embeddings.npy
└── prostt5_embeddings_mean.npy
```

These generated matrices are intentionally excluded from Git tracking.

### Row alignment

Correct row alignment is essential.

The affinity table, protein sequence table, protein embedding matrices, and ligand embedding matrix must correspond to the same PDBbind complexes in the same order before model training.

Saved split-index files under `splits/` preserve the exact seed-42 train/validation/test partition used for the primary benchmark.

## Benchmark Workflows

| Script | Purpose |
|---|---|
| `src/FS_ray_tuning_ablation.py` | Protein-only and ligand-only ablation |
| `src/FS_ray_tuning_modified.py` | Original protein + ligand MLP benchmark |
| `src/repeated_seed_mlp.py` | Repeated-split robustness analysis |
| `src/revised_pipeline_shap_xgb.py` | RF-SHAP feature attribution + XGBoost |
| `src/mlp_modality_shap.py` | MLP protein-versus-ligand SHAP attribution |
| `src/mlp_fusion_benchmark.py` | Alternative multimodal fusion strategies |

## Primary MLP Pipeline

The original multimodal MLP benchmark uses:

1. A 70/15/15 train/validation/test split
2. Train-only standardization of protein and ligand representations
3. Protein–ligand feature concatenation
4. Train-only mutual-information feature selection to 512 features
5. Ray Tune hyperparameter optimization with ASHA
6. Representation-specific optimal configurations selected using validation performance
7. Final MLP training
8. Evaluation on the untouched test set

All protein representations were evaluated using the same hyperparameter optimization protocol and search space.

Repeated-seed experiments reuse the previously selected representation-specific configurations rather than rerunning hyperparameter optimization for every random split.

## Reproducibility

The repository provides the embedding-generation code, Frontier SLURM launch scripts, benchmark and analysis code, exact seed-42 split indices, repeated-seed metrics, combined and ablation results, and RF-SHAP/XGBoost attribution results.

The project was developed and evaluated on the **OLCF Frontier** system using SLURM-based execution.

The PDBbind-derived input tables and large generated embedding matrices remain outside Git. They can be recreated from the source data and embedding-generation workflows described above.

## Key Takeaway

> **The usefulness of a pretrained protein representation is determined not only by the information it contains independently, but also by how effectively that information complements the ligand representation under a particular fusion, feature-selection, and downstream learning strategy.**
