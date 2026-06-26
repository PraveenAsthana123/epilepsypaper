# Methodology & Justification — Epilepsy EEG (real, honest)

Every choice below is justified and tied to the real pipeline in `code/`. Status
(§57.7): ✅ implemented · 📁 code present · ⚠️ applies to the DL variant (not this RF paper).

## Model, loss & gradient (honest about RandomForest)
| Item | This paper (RandomForest) | DL variant (EEGNet/CNN) |
|---|---|---|
| **Algorithm** | RandomForest(300 trees) ✅ | EEGNet/CNN/Transformer ⚠️ |
| **Split criterion / "loss"** | **Gini impurity** (per-node) — RF does NOT minimize a global loss | cross-entropy / focal loss ⚠️ |
| **Gradient** | **none** — RF is not gradient-trained (bagging of decision trees) | Adam/SGD gradient descent ⚠️ |
| **Why RF** | interpretable, strong on tabular EEG features, no GPU, robust to outliers, good baseline for an honest-evaluation study | — |

> Honest note: "loss function" and "gradient" do not apply to RandomForest. They apply to
> the deep-learning models (EEGNet/CNN/LSTM/Transformer) listed in the §162 architecture but
> NOT trained in this paper. Claiming a gradient/loss for RF would be false (§57.7).

## Hyperparameter tuning
| Param | Value | Justification | Tuned? |
|---|---|---|---|
| n_estimators | 300 | enough trees for stable OOB; diminishing returns above | fixed (📁 sweep not saved) |
| class_weight | **balanced** | seizure epochs are rare → reweight to avoid majority-class bias | fixed (justified, not swept) |
| max_depth | None (full) | RF controls variance by bagging, not depth limit | fixed |
| random_state | 42 | reproducibility | fixed |
| epoch length | 8 s | enough to capture rhythmic seizure activity; standard in CHB-MIT work | fixed |

> ⚠️ To add: `GridSearchCV`/`Optuna` over n_estimators × max_features × min_samples_leaf,
> saving the sweep — currently the config is fixed + justified, not exhaustively tuned.

## Class balance — balanced vs unbalanced data
- **The data is severely UNBALANCED**: seizure epochs ≪ non-seizure epochs (rare events).
  This is why accuracy (89.5%) is high but sensitivity (27.5%) is low.
- **Handling: `class_weight="balanced"`** (cost-sensitive learning) — reweights the minority
  seizure class so the forest doesn't trivially predict "no-seizure". ✅
- **SMOTE / oversampling**: ⚠️ NOT used here (the schizophrenia sibling paper uses augmentation×3).
  Justification for NOT using SMOTE on CHB-MIT: synthetic interpolated EEG epochs risk creating
  non-physiological signals and leaking across the subject boundary; cost-sensitive weighting is
  the safer choice for a patient-independent (LOSO) honest evaluation. To compare, SMOTE could be
  added in `chbmit_loso_pipeline.py` on the TRAIN fold only.
- **Why this matters**: on unbalanced clinical data, report **balanced accuracy ({89.5→}), MCC
  (0.343), and sensitivity** — not raw accuracy alone.

## Bias
| Bias type | Status | Handling |
|---|---|---|
| **Evaluation bias (epoch vs subject)** | ✅ the paper's core finding | LOSO (subject-level) is mandatory; epoch-level overstated sens by 53 pts |
| **Class/majority bias** | ✅ | class_weight=balanced |
| **Leakage bias** | ✅ | normalization fit on train fold only; subject-level split |
| **Subgroup/fairness bias (age/sex)** | ⚠️ | not yet computed — add per-group sensitivity + disparate-impact (§76) |

## Outliers
- `code/eeg_pipeline/outlier_analysis.py` (📁) — amplitude/variance-based bad-epoch + bad-channel
  detection. RF is inherently robust to outliers (tree splits), reducing their impact. ✅ robust by design.

## Normalization & standardization
- `code/eeg_pipeline/normalization.py` (📁): Z-score (zero-mean/unit-var) + MinMax options.
- **Leakage-safe (§74)**: statistics fit on the TRAIN fold only, applied to the held-out subject. ✅
- Band-power features are relative (normalized by total power) → robust to per-recording amplitude scale. ✅

## Structured vs unstructured data
- **Unstructured input**: raw EEG = continuous multichannel time-series (`.edf`, channel × time). ✅
- **Structured features**: converted to a tabular feature matrix (band-power + Hjorth + line-length +
  RMS, mean+std across channels) → structured rows for RF. ✅ (`code/reproducible/chbmit_loso_pipeline.py`)
- The 1D→2D image path (spectrogram/scalogram for CNNs) is the ⚠️ DL variant, not used here.

## Accuracy summary (real, all metrics)
UCI epoch-level **96.99%** (90+ ✅) · CHB-MIT LOSO **89.5% acc / 27.5% sens / 0.846 AUC / MCC 0.343**.
Full suite + confusion matrix + per-subject stats in `accuracy/COMPREHENSIVE_METRICS.md`.
The deployable, honest headline is the LOSO number — accuracy ≥ 90% is met epoch-level but
sensitivity is the real clinical signal.
