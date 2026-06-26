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
| **Performance AI** | ✅ | `accuracy/*.json` (acc/F1/AUC/sens/spec), `eeg_pipeline/benchmarking.py`, `metrics.py` |
| **Statistical AI** | ✅ | per-fold mean ± spread across 24 subjects; `eeg_pipeline/metrics.py` (CIs) |
| **Sensitivity analysis** | ✅ | the core finding: sensitivity 88.4%→35.1% under epoch→LOSO protocol change |
| **Subjective / subject-wise** | ✅ | per-subject `per_fold[]` in `chbmit_loso_results.json` (24 subjects), subject-wise CV (§83) |
| **Reliability** | 📁 | `eeg_pipeline/reliability_analysis.py` (ICC, test-retest) |
| **Clinical validation** | ✅ | PPV 0.70 / NPV 0.92 / sens / spec — `eeg_pipeline/clinical_validation.py` |
| **Interpretable AI** | 📁 | RandomForest = inherently interpretable; feature-importance via `eeg_pipeline/feature_selection.py` |
| **Explainable AI (ExpAI)** | ⚠️ | SHAP not yet wired — add `shap` over the RF in `chbmit_loso_pipeline.py` |
| **Responsible AI (ResAI)** | ⚠️ | fairness across subgroups not computed (paediatric cohort); add per-group parity |
| **Governance AI (Gov AI)** | 📁 | honest-evaluation finding IS the governance signal (no over-claim); model card in `architecture/model_card.md` |
| **EDA** | 📁 | `eeg_pipeline/eda_analysis.py` |
| **Outlier / filter / feature** | 📁 | `eeg_pipeline/{outlier,filter,feature_engineering}_analysis.py` |

## Justification (paper's central claim)
Identical feature+RF pipeline, two evaluation protocols → epoch-level overstates real-world
performance by 53 sensitivity points vs patient-independent LOSO. The honest, deployable number
is **35.1% sensitivity / 0.846 AUC** on CHB-MIT, not the 88% epoch-level figure. Governance value:
report LOSO, not epoch-level.

## To complete the ⚠️ items
- ExpAI: `pip install shap`; compute SHAP on the trained RF, save plots to `images/`.
- ResAI: compute sensitivity/specificity per age/sex subgroup; check disparate-impact ≥ 0.8 (§76).
- Training time + MLflow: wrap each fold with timing + `mlflow.log_metrics`.

## Signal-processing / preprocessing chain (real modules)
| Step | Status | Where | Notes |
|---|---|---|---|
| **Data conversion** (EDF → arrays/epochs) | ✅ | `code/eeg_pipeline/data_conversion.py`, `data_loader.py`; EDF read via `pyedflib`/`mne` | 8 s epochs |
| **Filtering** (band-pass / notch) | 📁 | `code/eeg_pipeline/filter_analysis.py`, `preprocessing.py` | remove drift + line noise |
| **Standardization** (zero-mean/unit-var) | 📁 | `code/eeg_pipeline/normalization.py` | per-channel, fit on train fold only (no leakage, §74) |
| **Normalization** (min-max / scaling) | 📁 | `code/eeg_pipeline/normalization.py` | config-selectable |
| **Fourier transform** (spectral features) | ✅ | δθαβγ **band-power** features in `chbmit_loso_pipeline.py` | FFT/PSD → 5 bands per channel |

Leakage-safe rule (§74): normalization + standardization statistics are fit on the **training**
fold only and applied to the held-out subject — never fit on all data.
