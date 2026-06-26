# Epilepsy DL Review — Paper, Code, Data Samples

Deep-learning review of EEG-based epilepsy / seizure detection.

## Layout
| Folder | Contents |
|---|---|
| `pdf/` | Compiled paper — `epilepsy-dl-review.pdf` |
| `code/paper/` | LaTeX source (`.tex` + `.bib`) — **edit in TeXstudio**, recompile to PDF |
| `code/eeg_pipeline/` | EEG ML pipeline (preprocessing → features → training → benchmarking → validation) |
| `architecture/` | Model card + data-split + normalization-stats configs |
| `data-samples/` | **10-row sample per dataset** (CHB-MIT, Bonn, Siena) — illustrative only |
| `accuracy/` | How metrics are computed (results in the paper) |
| `model/` | Training entry point (no trained binary committed) |
| `images/` | Figures (this review is text-only — none) |

## Datasets
- **CHB-MIT** Scalp EEG (`.edf`) — sample: 10 time-samples × 23 channels
- **Bonn** University EEG (`.csv`) — sample: 10 rows
- **Siena** Scalp EEG / PhysioNet (`.edf`) — sample: 10 time-samples × 35 channels

> Samples are 10 rows for inspection only — not the full datasets (which are large + access-controlled). No patient-identifying clinical reports are included.

## Edit the paper (TeXstudio)
```
texstudio code/paper/epilepsy-dl-review.tex
```

## Cross-platform (Windows + Linux)
The pipeline is pure-Python and OS-agnostic: it uses `pathlib.Path` for all paths
(no hardcoded `/media` or `C:\`), no `os.system`/shell calls. All dependencies
(`numpy`, `scipy`, `scikit-learn`, `pandas`, `mne`, `PyYAML`) ship wheels for both
Windows and Linux.

## Setup (Windows or Linux)
```bash
python -m venv venv
# Linux/macOS:  source venv/bin/activate
# Windows:      venv\Scripts\activate
pip install -r requirements.txt
```

## Config
Pipeline behaviour is driven by `code/eeg_pipeline/configs/spec.yaml`
(data splits, normalization, features, model). Edit it to change the run —
no code changes needed.

## Run
```bash
python code/eeg_pipeline/example_pipeline.py   # end-to-end on the data samples
```

## Backend / frontend compatible
The pipeline is a plain importable package (`from eeg_pipeline import ...`) with no
UI or server assumptions — call it from any FastAPI/Flask backend, a notebook, or a
CLI. Inputs/outputs are arrays + JSON, so a frontend can consume the results directly.

## Architecture (23-step EEG→AI→RAG, §162)
See `ARCHITECTURE.md` for the full 23-step flow + Mermaid diagram mapped to this repo's code.
This paper implements the classical-ML path (steps 1–16); DL-image, XAI, and RAG steps are
documented and marked ⚠️ (not implemented in this honest-evaluation study).

## Documentation map
| Doc | Covers |
|---|---|
| `ARCHITECTURE.md` | 23-step flow + Mermaid + code mapping |
| `METHODOLOGY.md` | model/loss/gradient · hyperparameters · SMOTE · bias · outliers · normalization · structured/unstructured · balanced/unbalanced — with justification |
| `ANALYSIS.md` | runs/training-count · 12 analyses · signal chain |
| `accuracy/COMPREHENSIVE_METRICS.md` | ALL metrics (acc/precision/recall/F1/AUC/spec/MCC/kappa) + confusion matrix + per-subject statistics |
| `OUTPUT_EVALUATION.md` | ORF output evaluation + RAGAS contract |
