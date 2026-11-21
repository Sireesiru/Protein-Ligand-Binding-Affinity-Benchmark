# Protein–Ligand Binding Affinity Representation Benchmark

For binding-affinity prediction, the best-performing protein embedding depends on how protein and ligand features are fused and how well the model generalizes to unseen protein clusters. Boltz leads under global and balanced mutual information, ProstT5 leads under RF-SHAP, and ProtGPT2 leads with all features on the conventional random split. With all features held fixed, ESM2 leads on the 30% MMseqs protein-cluster-disjoint split.

![Protein–ligand representation benchmark workflow](<Protein–Ligand Representation Benchmark Workflow.png>)

## 1. How does feature selection and fusion affect performance?

The benchmark compares five pretrained protein representations. The ligand representation is fixed as TxGemma (2,304 dimensions).

| Protein representation | Dimensions |
|---|---:|
| ESM2 | 1,280 |
| ProstT5 (ProtT5) | 1,024 |
| ProGen2 | 2,560 |
| Boltz protein representation | 384 |
| ProtGPT2 | 1,280 |

The four integration conditions use the same conventional random split and an MLP for affinity prediction.

- **Global MI** ranks concatenated protein and ligand features by mutual information with affinity and retains the top 512.
- **Modality-balanced MI** retains the top 256 features by mutual information within each modality.
- **RF-SHAP** retains the top 512 features ranked by SHAP importance from a Random Forest.
- **All-feature concatenation** passes every protein and ligand dimension to the MLP.

### Test R² by fusion strategy

| Protein | Global MI | Balanced MI | RF-SHAP | All features |
|---|---:|---:|---:|---:|
| ESM2 | 0.450 | 0.507 | 0.501 | 0.507 |
| ProstT5 | 0.506 | 0.531 | **0.540** | 0.539 |
| ProGen2 | 0.183 | 0.423 | 0.422 | 0.449 |
| Boltz | **0.531** | **0.538** | 0.520 | 0.529 |
| ProtGPT2 | 0.492 | 0.522 | 0.534 | **0.546** |

### Retained protein and ligand features

P denotes protein features and L denotes ligand features.

| Protein | Global MI | Balanced MI | RF-SHAP | All features |
|---|---|---|---|---|
| ESM2 | 25P / 487L | 256P / 256L | 233P / 279L | 1280P / 2304L |
| ProstT5 | 484P / 28L | 256P / 256L | 274P / 238L | 1024P / 2304L |
| ProGen2 | 512P / 0L | 256P / 256L | 82P / 430L | 2560P / 2304L |
| Boltz | 206P / 306L | 256P / 256L | 224P / 288L | 384P / 2304L |
| ProtGPT2 | 490P / 22L | 256P / 256L | 303P / 209L | 1280P / 2304L |

Global MI selected very different proportions of protein and ligand features depending on the embedding. It retained 25 protein and 487 ligand features for ESM2, and 512 protein and zero ligand features for ProGen2. ProGen2 was particularly sensitive to the integration choice. Its test R² was 0.183 with global MI and ranged from 0.422 to 0.449 across the three conditions that retained both modalities. Scores across the five embeddings were also more consistent within each of those three conditions. Boltz performed well with global MI and had the highest score under balanced MI.

[Fusion result CSVs](results/fusion/)

## 2. How well do representations generalize to unseen protein space?

Both split conditions use all-feature concatenation, so feature selection is held fixed. The conventional random split is compared with a **30% MMseqs protein-cluster-disjoint split**, which assigns whole protein clusters to train, validation, or test.

| Protein | Random R² | 30% cluster R² | R² drop | Retention |
|---|---:|---:|---:|---:|
| ESM2 | 0.507 | **0.2833** | 0.2237 | 55.9% |
| ProstT5 | 0.539 | 0.1781 | 0.3609 | 33.0% |
| ProGen2 | 0.449 | 0.2353 | 0.2137 | 52.4% |
| Boltz | 0.529 | 0.2394 | 0.2896 | 45.3% |
| ProtGPT2 | **0.546** | 0.2575 | 0.2885 | 47.2% |

R² drop is random-split R² minus cluster-split R². Retention is cluster-split R² divided by random-split R², expressed as a percentage.

- **Random split** — ProtGPT2 > ProstT5 > Boltz > ESM2 > ProGen2
- **30% MMseqs protein-cluster-disjoint split** — ESM2 > ProtGPT2 > Boltz ≈ ProGen2 > ProstT5

ESM2 moved from fourth on the conventional random split to first on the cluster-disjoint benchmark. ProstT5 showed the largest performance decrease, with a drop of 0.3609 R².

These results come from one frozen cluster split, so small pairwise differences, such as the gap between Boltz and ProGen2, should be interpreted cautiously. The 30% value refers to the MMseqs clustering threshold. Pairwise identity across train and test proteins would require a separate audit.

The planned generalization gradient is **Random → 50% → 40% → 30%**. The 50% and 40% evaluations are planned.

[Protein generalization result CSVs](results/protein_generalization/) · [Frozen cluster assignments and split files](splits/protein_clustering/)

## Dataset and modeling

The benchmark uses **PDBbind 2020** affinity measurements and includes **19,116 complexes** after excluding `3ag9` and `5dyw`. The train, validation, and test sets contain **13,381 / 2,867 / 2,868** complexes, respectively.

Standardization is fit on training data and applied to validation and test data. Feature selection, where applicable, is also fit on training data. Ray Tune with ASHA tunes the MLP using validation performance. The selected configuration is used to train the final MLP, and the held-out test set is used for evaluation.

## Repository structure and reproducibility

| Path | Contents |
|---|---|
| `data/` | Location for locally prepared PDBbind-derived inputs |
| `embedding_generation/` | Scripts for ESM2, ProstT5, ProGen2, and ProtGPT2 and a notebook for TxGemma |
| `splits/` | Saved seed-42 random split indices and the frozen cluster split under `protein_clustering/` |
| `src/` | Benchmark, split-generation, and supporting analysis scripts |
| `slurm/` | Job launchers used on OLCF Frontier |
| `results/` | Fusion, cluster generalization, ablation, combined MLP, repeated-split, and XGBoost artifacts |

1. Prepare `pdbbind_canonical_affinities.csv` and `pdbbind_sequences.csv` locally. These source tables are distributed separately from the repository.
2. Generate the embeddings with the scripts and notebook under `embedding_generation/`. The Boltz embeddings were generated in-house for a separate study and must be obtained separately. Store matrices under the local `embeddings/` directory. Keep the affinity records, sequences, and all embedding matrices in the same complex order.
3. Use the saved splits for the reported runs. The frozen cluster split is available as both CSV and NPZ. The cluster membership TSV is the input to `src/make_protein_cluster_split.py` when regenerating that split.
4. Use `src/mlp_fusion_benchmark.py` for the random-split fusion comparison and `src/protein_cluster_generalization.py` for the cluster-disjoint benchmark. Use `slurm/mlp_fusion_benchmark.slurm` and `slurm/protein_cluster_generalization.slurm` as launcher references. Adjust the original Frontier paths and environment settings for your run.

## Supporting analyses

The protein-only and ligand-only ablations, combined MLP results, and repeated-split ligand complementarity analysis across seeds 42–51 are available under `results/ablation/`, `results/combined/`, and `results/repeated_seeds/`. The seed-42 RF-SHAP + XGBoost comparison and Random Forest modality attribution are under `results/xgb_shap/`. The XGBoost comparison uses a different final learner from the RF-SHAP condition in the fusion table. CKA and learned protein–ligand alignment are outside the two comparisons reported here.
