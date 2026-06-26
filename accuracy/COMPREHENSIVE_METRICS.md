# Comprehensive Accuracy + Statistical Report — CHB-MIT LOSO (real)

Derived from `chbmit_loso_results.json` (per-subject tp/tn/fp/fn, 24 subjects).
Model: RandomForest(300,balanced). No fabricated numbers (§57.7).

## Aggregate confusion matrix (summed over 24 LOSO folds)
|              | Pred no-seizure | Pred seizure |
|--------------|----------------:|-------------:|
| **Actual no-seizure** | 12,297 (TN) | 343 (FP) |
| **Actual seizure**    | 1,145 (FN) | 435 (TP) |

## All accuracy metrics (aggregate)
| Metric | Value |
|---|---|
| **Accuracy** | **89.54%** |
| Precision (PPV) | 55.91% |
| Recall (Sensitivity) | 27.53% |
| Specificity | 97.29% |
| NPV | 91.48% |
| F1-score | 0.3690 |
| Balanced accuracy | 62.41% |
| MCC | 0.3430 |
| Cohen's kappa | 0.3190 |
| AUC (mean) | 0.8464 |

## Statistical (per-subject, n=24)
| Metric | Mean ± SD | 95% CI | Median [IQR] |
|---|---|---|---|
| accuracy | 90.0% ± 8.5 | [86.4, 93.6] | 91.2% [89.1–94.5] |
| sensitivity | 35.1% ± 29.6 | [22.6, 47.6] | 24.4% [7.7–62.0] |
| specificity | 96.9% ± 8.9 | [93.1, 100.6] | 99.7% [98.4–100.0] |
| ppv | 70.3% ± 37.8 | [54.3, 86.2] | 88.1% [55.0–100.0] |
| npv | 92.3% ± 3.4 | [90.9, 93.7] | 91.0% [89.2–95.4] |
| f1 | 41.2% ± 31.5 | [27.9, 54.5] | 34.1% [9.8–75.2] |
| auc | 84.6% ± 16.0 | [77.9, 91.4] | 93.5% [71.2–96.1] |

**Significance:** accuracy is significantly
above chance (one-sample t vs 0.5: t=23.16, df=23, p<0.05).

## Honest caveat (§162 invariant)
Accuracy is high (89.5%) because non-seizure epochs dominate. **Sensitivity
is 27.5%** — the deployable signal. High accuracy alone is misleading on
imbalanced seizure data; report sensitivity + AUC + MCC, not accuracy alone.

Plots: confusion_matrix.png, per_subject_auc.png (in `images/`).
