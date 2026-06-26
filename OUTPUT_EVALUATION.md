# Output Evaluation (ORF) + RAGAS — mandatory metrics (§59.4 / §79)

The §162 architecture includes a RAG report layer (steps 18–20). **This epilepsy paper is
classical-ML (RandomForest on EEG features) and does NOT implement RAG** — so the RAGAS
metrics below are documented as the mandatory contract, marked ⚠️ (not yet measured here),
not fabricated (§57.7). The RAG reference implementation lives in the sibling `eeg-stress-rag`.

## Output-Relevancy-First (ORF) — model-output evaluation (✅ applies now)
| What | Metric | Value (real) |
|---|---|---|
| Prediction correctness | Accuracy / Sensitivity / AUC / MCC | see `accuracy/COMPREHENSIVE_METRICS.md` |
| Calibration | per-subject spread (24-fold) | mean ± SD + 95% CI computed |
| Honest deployability | LOSO sensitivity (not epoch-level) | 27.5% — the relevant output signal |

## RAGAS — RAG report evaluation (⚠️ gap for this paper · mandatory if RAG added)
| Metric | Threshold (§59.4) | Status |
|---|---|---|
| Faithfulness | ≥ 0.85 | ⚠️ no RAG layer in this paper |
| Context precision | ≥ 0.75 | ⚠️ |
| Context recall | ≥ 0.75 | ⚠️ |
| Answer relevance | ≥ 0.80 | ⚠️ |
| **Citation accuracy** | **100%** | ⚠️ (every claim must trace to a retrieved chunk, §48.5) |

To implement: index epilepsy papers + SOPs + guidelines (step 18), hybrid retrieval (step 19),
generate the doctor/patient report (step 20–22), then gate on RAGAS in CI (`ragas` + `deepeval`).
