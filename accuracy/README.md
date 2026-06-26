# Accuracy / Results — REAL, reproducible

Every number traces to real public data + the code in `code/reproducible/`.

| Protocol | Dataset | Accuracy | Sensitivity | AUC | Code |
|---|---|---|---|---|---|
| Epoch-level 5-fold | UCI Epileptic (11,500 seg) | 96.99% | 88.39% | 0.9958 | `verify_uci_epoch_level.py` |
| **LOSO** (patient-independent) | CHB-MIT (24 subjects) | 90.0% | **35.1%** | 0.846 | `chbmit_loso_pipeline.py` |

- Model: `RandomForest(300, balanced)`
- Features: per-channel δθαβγ band-power + Hjorth(activity/mobility/complexity) + line-length + RMS (mean+std across channels)
- Raw per-fold + mean metrics: `chbmit_loso_results.json`, `uci_epoch_verified.json`

**Justification / key finding:** the identical feature+RF pipeline drops from 88.4% → 35.1% sensitivity
(−53 pts) when evaluation switches epoch-level → leave-one-subject-out. The high epoch-level number is a
measurement artefact, not deployable performance. This is the paper's central honest-evaluation claim.

## Reproduce
```bash
python code/reproducible/verify_uci_epoch_level.py   # → 96.99% / 88.4% sens
python code/reproducible/chbmit_loso_pipeline.py     # → 35.1% sens (needs CHB-MIT EDF)
```
