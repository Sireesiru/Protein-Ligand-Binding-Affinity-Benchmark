# Protein–Ligand Binding Affinity Representation Benchmark

This repository benchmarks how different pretrained protein representations contribute to protein–ligand binding-affinity prediction when paired with a fixed ligand representation.

The central question is:

> **How does the choice of pretrained protein representation affect binding-affinity prediction, particularly its ability to complement a fixed ligand representation?**

Rather than treating pretrained embeddings as interchangeable feature sources, the study evaluates their standalone predictive information, their incremental value in multimodal prediction, their interaction with feature-selection and downstream learning strategies, and the robustness of these effects across repeated train/validation/test splits.

## Representations

### Protein representations

Five pretrained protein representations are compared:

- **ESM2** — protein sequence representation
- **ProstT5** — protein representation incorporating sequence/structure-aware pretraining
- **ProGen2** — generative protein language-model representation
- **ProtGPT2** — generative protein language-model representation
- **Boltz** — structure-derived protein representation

### Ligand representation

- **TxGemma** — fixed ligand representation used across protein benchmarks

Keeping the ligand representation fixed allows differences in multimodal performance to be associated with the choice of protein representation and its compatibility with the ligand representation.

## Experimental Design

The benchmark evaluates several complementary questions.

### 1. Protein and ligand ablation

Protein-only and ligand-only models quantify the predictive information available from each modality independently.

### 2. Protein + ligand prediction

Protein and TxGemma ligand representations are combined to determine whether each protein representation improves prediction beyond the ligand representation alone.

The incremental contribution is measured as:

**ΔR² = R²(protein + ligand) − R²(ligand)**

A strong protein-only representation therefore does not necessarily imply strong multimodal complementarity.

### 3. Repeated-split robustness

The protein-only, ligand-only, and combined models are evaluated across 10 random seeds to determine whether observed representation differences are robust to train/validation/test partitioning.

### 4. RF-SHAP feature selection + XGBoost

Random Forest SHAP values are used to rank features from the concatenated protein–ligand representation. The top 512 features are then used for XGBoost affinity prediction.

SHAP values are aggregated by modality to characterize model attribution to protein versus ligand features.

These percentages describe **feature attribution/model reliance**, not the fraction of R² explained by each modality.

### 5. MLP modality attribution

MLP-based SHAP analysis is included to examine protein-versus-ligand attribution under the neural-network prediction pipeline.

*Results are currently pending.*

### 6. Fusion-strategy benchmark

Additional experiments compare alternative multimodal fusion/feature-selection strategies, including:

- mutual-information selection from concatenated features
- modality-balanced mutual-information selection
- RF-SHAP feature selection
- all-feature fusion

*Results are currently pending.*

## Current Results

Repeated-seed experiments show clear differences in how effectively the protein representations complement the fixed ligand representation.

| Protein representation | Protein R² | Ligand R² | Combined R² | ΔR² |
|---|---:|---:|---:|---:|
| Boltz | 0.323 | 0.395 | **0.505** | **+0.110** |
| ProstT5 | 0.343 | 0.377 | **0.471** | **+0.095** |
| ProtGPT2 | 0.342 | 0.376 | **0.458** | **+0.083** |
| ESM2 | 0.315 | 0.382 | **0.444** | **+0.062** |
| ProGen2 | 0.154 | 0.376 | **0.150** | **−0.225** |

Values are means across 10 random splits.

The results demonstrate that **unimodal representation quality and multimodal utility are distinct properties**. Boltz, ProstT5, ProtGPT2, and ESM2 provide positive incremental information when combined with TxGemma, whereas ProGen2 substantially reduces predictive performance under the evaluated MLP pipeline.

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
│   └── README.md
├── embeddings/
│   └── README.md
├── splits/
│   └── saved train/validation/test split indices
├── src/
│   ├── FS_ray_tuning_ablation.py
│   ├── repeated_seed_mlp.py
│   ├── revised_pipeline_shap_xgb.py
│   ├── mlp_modality_shap.py
│   └── mlp_fusion_benchmark.py
├── slurm/
│   └── Frontier SLURM launch scripts
└── results/
    ├── ablation/
    ├── combined/
    ├── repeated_seeds/
    ├── xgb_shap/
    ├── modality_shap/
    └── fusion/

