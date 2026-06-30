# GLOBAL PAPER POLICY — binding gate for ALL epilepsy papers

This is the **single mandatory alignment policy** for every paper in this project
(`q1_full_2col` = with RGAIG, `q1_noRGAIG_2col` = without RGAIG, `review_full_2col`).
It consolidates and supersedes ad-hoc guidance. Detailed checklists live in
[PAPER_TOP1_POLICY.md](PAPER_TOP1_POLICY.md), [PAPER_ANALYSIS_INVENTORY.md](PAPER_ANALYSIS_INVENTORY.md),
and [PAPER_FIGURES_TABLES_FORMULAS.md](PAPER_FIGURES_TABLES_FORMULAS.md). **No paper may be
called submission-ready until it passes every MANDATORY rule below.**

## M. MANDATORY (a paper fails the gate if any is violated)
1. **No leakage claim laundering.** Never present epoch-level / random-split numbers as
   clinical or patient-independent performance. Epoch-level results must be explicitly
   labelled "leaky / non-deployable".
2. **Same-dataset leakage proof is the centre.** The headline comparison must be
   **CHB-MIT epoch-level CV vs CHB-MIT LOSO on the identical features+model**, NOT
   Bonn/UCI-vs-CHB-MIT (which confounds protocol with dataset/population/montage). The
   cross-dataset (Bonn vs CHB-MIT) result may appear only as a secondary, clearly-labelled
   "protocol-plus-dataset" point.
3. **Report pooled AND macro metrics, both labelled.** Pooled-confusion sensitivity
   (0.2753) and macro per-subject sensitivity (0.3508/35.1%) must both appear; state which
   is the headline (macro per-subject), and show the per-subject distribution.
4. **Mandatory metric set:** accuracy, sensitivity, specificity, PPV, NPV, F1, **MCC**,
   balanced accuracy, ROC-AUC, **PR-AUC**, with **subject-level bootstrap CIs** (resample
   subjects, not epochs).
5. **RGAIG / RAG / MCP / governance are NOT empirical contributions** unless actually
   evaluated. They live in a short "Deployment Implications" section. They never appear in
   the title, and only minimally in the abstract.
6. **No self-promotional language:** remove "Q1-readiness", "top-1%", and over-repeated
   "honest figure of merit". Reviewers punish self-certification.
7. **Real availability line:** cite the actual repository
   `https://github.com/PraveenAsthana123/epilepsypaper` (+ archived DOI before submission),
   not "code released".
8. **Honesty (Sec. 57.7):** every number traces to a committed script + result JSON. No
   fabricated, placeholder, or future-dated values.

## R. REQUIRED before submission (strongly enforced)
9. Confusion matrix / TP-FP-FN-TN totals for the LOSO result.
10. Threshold sweep (0.1-0.9) reporting sens/spec/PPV/NPV/F1/MCC, and **sensitivity at
    fixed specificity 90/95/99%**.
11. Per-subject sensitivity bar plot + worst-subject failure analysis (chb14/15/20/04).
12. Calibration: Brier + ECE (+ reliability curve); calibrate inside training folds if needed.
13. Multi-model LOSO benchmark on one common split (majority, LogReg, SVM, RF, HistGB, MLP),
    plus comparison to >=3 prior CHB-MIT LOSO papers (epoch-level papers listed separately
    and marked non-deployable).
14. Limitations table: pediatric-only CHB-MIT, public data, no prospective trial, low
    deployable sensitivity, dataset differences.
15. Nested/subject-wise hyperparameter & threshold tuning ONLY (never random-epoch tuning
    presented as patient-independent).

## F. FUTURE / nice-to-have (state as future work if not done)
16. Adult external validation (Siena / TUH-TUSZ); cross-dataset train/test.
17. False-alarms-per-hour and detection latency.
18. Montage/channel-reduction robustness; temporal post-processing (k-of-n windows).

## Review-paper-specific
19. Self-contained build (all `\input` files present in the build folder).
20. Exact PRISMA (query strings, databases, dates, dedupe counts, exclusion reasons) — not "~220".
21. Per-paper evidence table (dataset/subjects/task/split/LOSO?/code?/limitation) + risk-of-bias.
22. Verified BibTeX (no AUTHOR-TO-VERIFY, no unverified future-year entries); remove off-topic papers.

## I. PAPER IDENTITY (each paper = ONE contribution; no overlap)
- **q1_noRGAIG (empirical):** validation-aware benchmark ONLY. Sections allowed: intro,
  *short* related work, data, preprocessing, features, validation, same-dataset comparison,
  model benchmark, error analysis, metrics/calibration, limitations, conclusion.
  **FORBIDDEN here:** RGAIG/MCP/RAG/AIOps/governance/agentic, architecture-family survey
  tables, dataset-reliance survey tables, long future-work lists, deployment architecture.
- **q1_full (RGAIG = systems/deployment paper):** governance/deployment framework. Keep the
  honest LOSO result only as *motivation*; do NOT repeat full data/feature/benchmark detail or
  claim RGAIG improves accuracy. Focus: architecture, audit, drift, human escalation, KPIs.
- **review (systematic review):** PRISMA + evidence synthesis. The taxonomy/survey tables live
  HERE, not in the empirical paper.

## X. RECURRING MISTAKES — do not repeat (brutal log)
1. **Scope-creep into the empirical paper.** Adding review surveys, governance, RL/RAG, or a
   15-figure kitchen sink to q1_noRGAIG. Result: float-overload build hangs + all 3 papers read
   the same (salami-slicing). FIX: the empirical paper subtracts, never pads; survey content
   belongs to the review.
2. **Padding for page count.** Never add breadth to hit "10 pages". Pages come from REAL
   analyses (calibration, benchmark, error analysis), or the paper is shorter. Honesty > length.
3. **Self-promotional language** ("top 1%", "Q1-ready", "clinical-grade", "state-of-the-art").
4. **Leftover cross-references** (e.g. `\ref{sec:rgaig}` in the no-RGAIG paper) — grep before commit.
5. **Dataset/protocol confound** presented as pure protocol effect. Same-dataset control is the centre.
6. **Committing LaTeX build artifacts** (.aux/.log/.out) — gitignored.
7. **Unverified citations** (AUTHOR-TO-VERIFY, 2026 preprints) shipped in a submission bib.

## Per-paper alignment status (updated as fixes land)
| Rule | q1_full (RGAIG) | q1_noRGAIG | review |
|---|---|---|---|
| M2 same-dataset proof | pending | pending | n/a |
| M3 pooled+macro | pending | pending | n/a |
| M4 MCC/PR-AUC/CI | pending | pending | n/a |
| M5 RGAIG demoted | pending | partial | n/a |
| M6 no self-promo | pending | pending | pending |
| M7 repo link | pending | pending | pending |
| R19 self-contained build | n/a | n/a | pending |
