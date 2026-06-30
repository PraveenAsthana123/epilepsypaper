# Reproducible Evidence — EEG Epilepsy Honest-Evaluation Paper

Every number in the paper traces to real data + code here.

## Datasets (real, public)
- **UCI Epileptic Seizure Recognition** — 11,500 segments, 178 features
- **CHB-MIT Scalp EEG** — 24 paediatric subjects, 619 EDF files (PhysioNet)

## Verified results
| Protocol | Dataset | Accuracy | Sensitivity | AUC | Code |
|---|---|---|---|---|---|
| Epoch-level 5-fold | UCI | 96.99% | 88.39% | 0.9958 | `code/verify_uci_epoch_level.py` |
| LOSO (patient-independent) | CHB-MIT | 90.0% | **35.1%** | 0.846 | `code/chbmit_loso_pipeline.py` |

**Finding:** identical feature+RF pipeline; sensitivity collapses 88.4% → 35.1% (53 points) when
evaluation switches from epoch-level to leave-one-subject-out. The high epoch-level number is a
measurement artefact, not deployable performance.

## Reproduce
```bash
python code/verify_uci_epoch_level.py     # → 96.99% / 88.4% sens
python code/chbmit_loso_pipeline.py       # → 35.1% sens (needs CHB-MIT EDF)
```

## Full 23-step pipeline (each step runnable, verified on real data)
| Step | Script | Produces |
|---|---|---|
| 8/9 Time-frequency + 2D images | `step08_time_frequency.py` | STFT spectrogram, CWT scalogram, PSD → `images/`, `accuracy/time_frequency.json` |
| 12 Feature evaluation (statistical) | `step12_feature_evaluation.py` | ANOVA F-test + mutual information + correlation → `accuracy/feature_evaluation.json` |
| 13 Feature selection | `step13_feature_selection.py` | LASSO + RFE + SelectKBest + PCA, LOSO-validated → `accuracy/feature_selection.json` |
| 14-16 Multi-model benchmark | `multi_pipeline_benchmark.py` | Statistical vs ML vs DL × normalize/standardize × LOSO/epoch on **2 datasets** → `accuracy/MULTI_PIPELINE_COMPARISON.md` |
| 18-22 RAG report | `step18_rag_report.py` | TF-IDF index + hybrid retrieval + doctor/patient report → `accuracy/rag_report.md` |
| 23 Governance | `step23_governance.py` | model version, PSI data-drift, perf monitor, audit log → `accuracy/governance.json` |

```bash
# the multi-model comparison (needs feature_cache.npz from chbmit_loso_pipeline.py + UCI_CSV)
python code/reproducible/step12_feature_evaluation.py
python code/reproducible/step13_feature_selection.py
UCI_CSV="/path/Epileptic Seizure Recognition.csv" python code/reproducible/multi_pipeline_benchmark.py
python code/reproducible/step18_rag_report.py "inter-patient seizure detection"
python code/reproducible/step23_governance.py
```

**Benchmark headline:** the epoch→LOSO sensitivity collapse holds for **every** model family
(statistical, ML, DL) — it is a property of the evaluation protocol, not the model. On UCI epoch-level,
RandomForest reproduces the paper's 96.99% / 88.4% sens / 0.9958 AUC exactly.
