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
