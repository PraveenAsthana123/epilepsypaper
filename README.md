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
