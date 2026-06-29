# Analysis & Governance — Epilepsy EEG (real, traceable)

All numbers trace to `accuracy/*.json` + `code/reproducible/`. Honest status per item
(§57.7): ✅ = run + saved · 📁 = code present, re-run to regenerate · ⚠️ = not yet implemented.

## Runs, training count & timing
| Item | Value | Source |
|---|---|---|
| **Model train count — CHB-MIT** | **24×** (Leave-One-Subject-Out: 1 model per held-out subject × 24 subjects) | `chbmit_loso_pipeline.py` + `chbmit_loso_results.json` (24 folds) |
| **Model train count — UCI** | **5×** (5-fold CV) | `verify_uci_epoch_level.py` |
| **Training time** | ⚠️ not logged in the saved results | (re-run with timing to capture) |
| **Logging / tracking** | 📁 stdout + JSON results per run; no MLflow run yet | results JSON is the per-run record |

## Hyperparameters & tuning
| Param | Value | Tuned? |
|---|---|---|
| model | RandomForest | fixed choice (interpretable baseline) |
| n_estimators | 300 | fixed |
| class_weight | balanced | fixed (class imbalance) |
| random_state | 42 | fixed (reproducibility) |
| epoch length | 8 s | fixed |
| n_jobs | -1 | — |

> Tuning status: ⚠️ no grid/Optuna sweep saved — current results use the fixed config above.
> To add: wrap the classifier in `GridSearchCV`/`Optuna` in `chbmit_loso_pipeline.py` and save the sweep.

## Tech stack
Python · scikit-learn (RandomForest) · NumPy · SciPy · pandas · MNE + pyedflib (EDF I/O) ·
PyYAML (config) · matplotlib/seaborn (viz). Cross-platform (Win/Linux), pure-Python (§160).

## The 12 analyses (mapped to real code)
| Analysis | Status | Where |
|---|---|---|
| **Performance AI** | ✅ | `accuracy/*.json` (acc/F1/AUC/sens/spec) computed by `code/reproducible/comprehensive_accuracy_analysis.py` |
| **Statistical AI** | ✅ | per-fold mean ± spread across 24 subjects; bootstrap CIs in `code/reproducible/comprehensive_accuracy_analysis.py` |
| **Sensitivity analysis** | ✅ | the core finding: sensitivity 88.4%→35.1% under epoch→LOSO protocol change |
| **Subjective / subject-wise** | ✅ | per-subject `per_fold[]` in `chbmit_loso_results.json` (24 subjects), subject-wise CV (§83) |
| **Reliability** | 📁 | ICC / test-retest not computed (not included in this repo) |
| **Clinical validation** | ✅ | PPV 0.70 / NPV 0.92 / sens / spec — `code/reproducible/comprehensive_accuracy_analysis.py` |
| **Interpretable AI** | ✅ | RandomForest = inherently interpretable; feature importance via `code/reproducible/xai_feature_importance.py` |
| **Explainable AI (ExpAI)** | ✅ | SHAP (TreeExplainer over the RF) in `code/reproducible/xai_feature_importance.py` → `accuracy/xai_feature_importance.json`, `images/shap_summary.png` |
| **Responsible AI (ResAI)** | ⚠️ | fairness across subgroups not computed (paediatric cohort); add per-group parity |
| **Governance AI (Gov AI)** | 📁 | honest-evaluation finding IS the governance signal (no over-claim); model card in `architecture/model_card.md` |
| **EDA** | 📁 | standalone EDA module not included in this repo |
| **Outlier / filter / feature** | ✅ | filtering + band-power/Hjorth/line-length feature extraction done inline in `code/reproducible/chbmit_loso_pipeline.py` |

## Justification (paper's central claim)
Identical feature+RF pipeline, two evaluation protocols → epoch-level overstates real-world
performance by 53 sensitivity points vs patient-independent LOSO. The honest, deployable number
is **35.1% sensitivity / 0.846 AUC** on CHB-MIT, not the 88% epoch-level figure. Governance value:
report LOSO, not epoch-level.

## To complete the ⚠️ items
- ResAI: compute sensitivity/specificity per age/sex subgroup; check disparate-impact ≥ 0.8 (§76).
- Training time + MLflow: wrap each fold with timing + `mlflow.log_metrics`.

## Signal-processing / preprocessing chain (real modules)
| Step | Status | Where | Notes |
|---|---|---|---|
| **Data conversion** (EDF → arrays/epochs) | ✅ | `code/reproducible/chbmit_loso_pipeline.py`; EDF read via `mne`, 8 s epoching (inline) | 8 s epochs |
| **Filtering** (band-pass / notch) | ✅ | `code/reproducible/chbmit_loso_pipeline.py` — 0.5–40 Hz Butterworth (inline) | remove drift + line noise |
| **Standardization** (zero-mean/unit-var) | ✅ | `code/reproducible/chbmit_loso_pipeline.py` — StandardScaler | per-channel, fit on train fold only (no leakage, §74) |
| **Normalization** (min-max / scaling) | 📁 | standardization used instead (above); min-max not applied | — |
| **Fourier transform** (spectral features) | ✅ | δθαβγ **band-power** features in `chbmit_loso_pipeline.py` | FFT/PSD → 5 bands per channel |

Leakage-safe rule (§74): normalization + standardization statistics are fit on the **training**
fold only and applied to the held-out subject — never fit on all data.
