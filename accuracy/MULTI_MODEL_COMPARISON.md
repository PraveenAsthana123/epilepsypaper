# Multi-Model + Ensemble - CHB-MIT LOSO (real)

Patient-independent (24 subjects), same features.

## Global accuracy @ threshold 0.5
| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | AUC | MCC |
|---|---|---|---|---|---|---|---|
| SVM_RBF | 87.8% | 36.6% | 94.2% | 43.9% | 0.399 | 0.779 | 0.333 |
| RandomForest | 89.5% | 27.5% | 97.3% | 55.9% | 0.369 | 0.778 | 0.343 |
| XGBoost | 87.5% | 35.2% | 94.0% | 42.3% | 0.384 | 0.770 | 0.317 |
| MLP | 83.7% | 45.1% | 88.5% | 32.9% | 0.380 | 0.716 | 0.294 |
| Ensemble_Voting | 88.6% | 35.3% | 95.3% | 48.4% | 0.408 | 0.789 | 0.353 |

## Subjective (per-subject, n=24) - AUC mean +/- SD [95% CI], worst subject
| Model | AUC mean+/-SD | 95% CI | worst subject (AUC) |
|---|---|---|---|
| SVM_RBF | 0.830 +/- 0.181 | [0.754, 0.906] | chb14 (0.421) |
| RandomForest | 0.846 +/- 0.160 | [0.779, 0.914] | chb15 (0.489) |
| XGBoost | 0.837 +/- 0.168 | [0.766, 0.907] | chb14 (0.504) |
| MLP | 0.801 +/- 0.174 | [0.728, 0.875] | chb20 (0.375) |
| Ensemble_Voting | 0.851 +/- 0.169 | [0.780, 0.923] | chb15 (0.527) |

## Statistical - Wilcoxon paired AUC vs RandomForest (across subjects)
| Model | mean AUC delta vs RF | p-value | significant? |
|---|---|---|---|
| SVM_RBF | -0.017 | 1.000 | no |
| RandomForest | - | - | reference model |
| XGBoost | -0.010 | 0.422 | no |
| MLP | -0.045 | 0.012 | yes |
| Ensemble_Voting | +0.005 | 0.068 | no |

> Honest read: ~90% accuracy is imbalance-driven; compare on **AUC + MCC + per-subject spread**. The worst-subject column is the deployability check (subject-wise, section 83).