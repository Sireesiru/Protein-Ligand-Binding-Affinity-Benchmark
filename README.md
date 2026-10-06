# Protein–Ligand Binding Affinity Representation Benchmark

## Study Question

This study asks:

> **How does the choice of pretrained protein representation affect protein–ligand binding-affinity prediction, particularly its ability to complement a fixed ligand representation?**

We combined five pretrained protein sequence and structural representations : **ESM2, ProstT5, ProGen2, ProtGPT2, and Boltz**  while keeping the ligand representation (**TxGemma**) fixed and compared the results. Rather than asking only which representation gives the highest prediction accuracy, the benchmark follows six connected questions:

| | Question | What it tests |
|---|---|---|
| **Q1** | **How informative is each protein representation on its own?** | Standalone affinity-relevant information |
| **Q2** | **How much does each protein representation add beyond the ligand?** | Multimodal complementarity with fixed TxGemma |
| **Q3** | **Are the observed differences robust?** | Stability across 10 random data splits |
| **Q4** | **Do the conclusions depend on the downstream learner?** | MLP vs. RF-SHAP + XGBoost |
| **Q5** | **How much does the model rely on protein versus ligand information?** | Protein–ligand modality attribution |
| **Q6** | **Does the fusion strategy matter?** | Alternative feature-selection and fusion strategies |

Together, these experiments distinguish representation quality, multimodal complementarity, and dependence on the downstream modeling strategy.

## Study Workflow
<p align="Left">
  <img src="Protein–Ligand Representation Benchmark Workflow.png"
       alt="Protein–Ligand Representation Benchmark Workflow"
       width="100%"
       height="600">
</p>
---

## Representations used
### Protein
| Representation | Representation type |
|---|---|
| ESM2 | Protein sequence representation |
| ProstT5 | Sequence/structure-aware protein representation |
| ProGen2 | Generative protein language-model representation |
| ProtGPT2 | Generative protein language-model representation |
| Boltz | Structure-derived protein representation |

### Ligand
**TxGemma** is used as the fixed ligand representation throughout the benchmark.
Keeping the ligand representation fixed allows differences in multimodal performance to be associated with the protein representation and its compatibility with the ligand representation.

---
## Results

### Q1–Q3: Representation information, complementarity, and robustness

The table below summarizes performance across 10 random train/validation/test splits.

| Protein representation | Protein R² | Ligand R² | Protein + Ligand R² | ΔR² |
|---|---:|---:|---:|---:|
| Boltz | 0.323 | 0.395 | **0.505** | **+0.110** |
| ProstT5 | 0.343 | 0.377 | **0.471** | **+0.095** |
| ProtGPT2 | 0.342 | 0.376 | **0.458** | **+0.083** |
| ESM2 | 0.315 | 0.382 | **0.444** | **+0.062** |
| ProGen2 | 0.154 | 0.376 | **0.150** | **−0.225** |

The results show that **unimodal representation quality and multimodal complementarity are distinct properties**.

Boltz, ProstT5, ProtGPT2, and ESM2 provide positive incremental information when combined with TxGemma. Under the evaluated MLP pipeline, ProGen2 does not provide the same complementary benefit.

### Q4: Does performance depend on the downstream learner?

The original MLP results were compared with an alternative pipeline using Random Forest SHAP feature selection followed by XGBoost.

| Protein representation | MLP P+L R² | RF-SHAP + XGB R² | Difference |
|---|---:|---:|---:|
| ESM2 | 0.450 | 0.522 | +0.072 |
| ProstT5 | 0.506 | 0.552 | +0.047 |
| ProGen2 | 0.183 | 0.497 | **+0.314** |
| Boltz | 0.531 | 0.543 | +0.012 |
| ProtGPT2 | 0.492 | 0.547 | +0.055 |

These are seed-42 results and therefore complement, rather than replace, the repeated-split MLP analysis above.

The large change observed for ProGen2 illustrates an important point: **the apparent utility of a pretrained representation can depend strongly on the feature-selection and downstream learning strategy.**

### Q5: Protein versus ligand attribution

Random Forest SHAP values were aggregated by modality before XGBoost training.

| Protein representation | Protein attribution | Ligand attribution | XGB Test R² |
|---|---:|---:|---:|
| ESM2 | 33.0% | 67.0% | 0.522 |
| ProstT5 | 49.9% | 50.1% | **0.552** |
| ProGen2 | 11.5% | 88.5% | 0.497 |
| Boltz | 42.2% | 57.8% | 0.543 |
| ProtGPT2 | 45.7% | 54.3% | 0.547 |

These percentages describe **feature attribution/model reliance in the Random Forest SHAP analysis**. They should not be interpreted as the percentage of predictive R² explained by each modality.
MLP-based modality SHAP analysis is also included in the benchmark.

### Q6: Fusion strategy

The fusion benchmark evaluates whether multimodal performance changes when protein and ligand information are integrated using different strategies:

- mutual-information selection from concatenated features
- modality-balanced mutual-information selection
- RF-SHAP feature selection
- all-feature fusion
Results will be added after completion of the benchmark.
---

## Modeling Pipelines used for benchmarking 

### Multi Layer Percepteron(MLP)-based 
The primary MLP workflow uses:

1. 70/15/15 train/validation/test splitting
2. train-only standardization of protein and ligand embeddings
3. protein and ligand feature concatenation
4. train-only mutual-information feature selection to 512 features
5. Ray Tune hyperparameter optimization with ASHA
6. representation-specific optimal configurations selected using validation performance
7. final MLP training
8. evaluation on the untouched test set

All protein representations use the same hyperparameter search protocol and search space.
For reproducibility, we did repeated-split experiment. The previously selected representation-specific configurations are reused here rather than re-running hyperparameter optimization for every split.

### RF-SHAP + XGBoost-based pipeline 

The alternative modeling pipeline uses:

1. The same input representations and train/validation/test partition
2. Train-only feature standardization
3. Protein and ligand concatenation
4. Random Forest modeling
5. SHAP-based feature ranking
6. Selection of the top 512 features
7. XGBoost affinity prediction
8. Protein- Vs. ligand attribution analysis

This provides an alternative downstream learner and feature-selection strategy for testing whether representation rankings are specific to the MLP pipeline.
---
## Repository Contents

```text
protein-ligand-binding-affinity-benchmark/
├── README.md
│
├── data/
│   └── local PDBbind-derived datasets (not distributed here)
│
├── embedding_generation/
│   ├── TxGemma_embedings.ipynb
│   ├── protgpt2_embedding_generation.py
│   ├── worker_decoder_progen2.py
│   ├── worker_encoder_ESM2.py
│   └── worker_encoder_prostT5.py
│
├── embeddings/
│   └── generated embedding matrices (not tracked)
├── splits/
│   └── saved train/validation/test split indices
│
├── src/
│   ├── FS_ray_tuning_ablation.py
│   ├── FS_ray_tuning_modified.py
│   ├── repeated_seed_mlp.py
│   ├── revised_pipeline_shap_xgb.py
│   ├── mlp_modality_shap.py
│   └── mlp_fusion_benchmark.py
│
├── slurm/
│   └── Frontier SLURM launch scripts
│
└── results/
    ├── ablation/
    ├── combined/
    ├── repeated_seeds/
    ├── xgb_shap/
    ├── modality_shap/
    └── fusion/
```

### What the folders contain

- **`embedding_generation/`** — code used to generate the pretrained protein and ligand representations.
- **`embeddings/`** — local generated embedding matrices. These large files are excluded from Git.
- **`splits/`** — saved indices preserving the exact seed-42 train/validation/test split.
- **`src/`** — modeling, ablation, robustness, SHAP, XGBoost, and fusion-analysis code.
- **`slurm/`** — SLURM launch scripts used for execution on OLCF Frontier.
- **`results/ablation/`** — protein-only and ligand-only results.
- **`results/combined/`** — original protein + ligand MLP results.
- **`results/repeated_seeds/`** — performance across seeds 42–51.
- **`results/xgb_shap/`** — RF-SHAP modality attribution, selected features, and XGBoost results.
- **`results/modality_shap/`** — MLP modality-attribution analysis.
- **`results/fusion/`** — multimodal fusion-strategy benchmark.
---
## Reproducing the Benchmark 
### 1. Prepare the source data
The study uses protein–ligand complexes and affinity measurements derived from **PDBbind 2020**. The PDBbind-derived input tables are **not distributed with this repository**.
The workflows require two locally prepared inputs:

- `pdbbind_canonical_affinities.csv`
- `pdbbind_sequences.csv`

The final modeling dataset contains 19,116 complexes after exclusion of two problematic entries (`3ag9` and `5dyw`) from the original 19,118-row dataset.
### 2. Generate representations

Embeddings generation code is provided for four protein sequence models and one ligand model as below

| Representation | Code |
|---|---|
| ESM2 | `embedding_generation/worker_encoder_ESM2.py` |
| ProstT5 | `embedding_generation/worker_encoder_prostT5.py` |
| ProGen2 | `embedding_generation/worker_decoder_progen2.py` |
| ProtGPT2 | `embedding_generation/protgpt2_embedding_generation.py` |
| TxGemma | `embedding_generation/TxGemma_embedings.ipynb` |

Corresponding HPC launch scripts are provided under `slurm/`.
The Boltz representation used in the current benchmark are calculated in-house for another study. They are obtained separately. Its provenance/access procedure will be documented soon. Generated matrices are stored locally under `embeddings/` and are excluded from Git.

### 3. Preserve sample alignment

The affinity table, protein sequences, protein embedding matrices, and TxGemma ligand embedding matrix must correspond to the same complexes in the same row order.

### 4. Run the experiments

| Scientific question | Script |
|---|---|
| Protein/ligand ablation | `src/FS_ray_tuning_ablation.py` |
| Protein + ligand MLP | `src/FS_ray_tuning_modified.py` |
| Repeated-split robustness | `src/repeated_seed_mlp.py` |
| RF-SHAP + XGBoost | `src/revised_pipeline_shap_xgb.py` |
| MLP modality attribution | `src/mlp_modality_shap.py` |
| Fusion strategies | `src/mlp_fusion_benchmark.py` |

---
## Reproducibility

Saved split files preserve the exact seed-42 partition used for the primary benchmark.Repeated-split results preserve performance across seeds 42–51.
The project was developed and evaluated on the **OLCF Frontier** system using SLURM-based execution.
Large generated embedding matrices and PDBbind-derived source tables are intentionally kept outside the Git repository.

---
## Key Finding

> **A pretrained protein representation should not be judged only by how informative it is independently. Its usefulness for multimodal affinity prediction also depends on how well it complements the ligand representation and how the combined representations are selected, fused, and learned.**
