# Paper Quality Policy for Epilepsy EEG Manuscripts

This policy must be followed before any epilepsy EEG paper in this repository is treated as submission-ready. The goal is not to make the paper look stronger; the goal is to make every claim defensible under expert review.

## Scope

This policy applies to:

- `code/paper/q1_noRGAIG_2col.tex`
- `code/paper/q1_full_2col.tex`
- `code/paper/review_full_2col.tex`
- All accuracy, training, data, benchmarking, and model claims used in those papers

## Non-Negotiable Submission Gate

A paper must not be submitted unless all of the following are true:

- Every reported result is traceable to a script, data split, random seed, and output file.
- Patient-independent performance is reported using leave-one-subject-out or an equivalent subject-disjoint protocol.
- Epoch-level cross-validation is clearly separated from subject-level validation and is never presented as deployment evidence.
- The same dataset, same features, and same model are used when comparing validation protocols.
- No RGAIG, RAG, MCP, agentic, or deployment-framework claim is presented as an empirical contribution unless it is actually evaluated.
- Accuracy is never the lead metric unless sensitivity, specificity, AUC, PR-AUC, MCC, false alarms, and confidence intervals are also reported.
- Limitations are explicit, specific, and tied to the actual experiment.

## Required Empirical Analyses

The empirical paper must include these analyses before it is considered strong:

1. Same-dataset leakage comparison: CHB-MIT epoch-level split versus CHB-MIT subject-disjoint LOSO using the same features and model.
2. Nested LOSO hyperparameter tuning: tune only on training subjects inside each outer fold.
3. Threshold operating-point analysis: show sensitivity, specificity, precision, F1, MCC, and false-alarm tradeoffs across thresholds.
4. Sensitivity at fixed specificity: report sensitivity at 90%, 95%, and 99% specificity.
5. PR-AUC: mandatory because seizure epochs are minority-class events.
6. Calibration: report Brier score, ECE, and reliability curves.
7. Per-subject failure analysis: identify best, median, and worst subjects and explain why performance differs.
8. Confidence intervals: report 95% CIs for all primary metrics.
9. Statistical comparison: compare models with paired subject-level tests or bootstrap CIs.
10. Ablation study: feature groups, classifier choices, threshold tuning, class balancing, and temporal post-processing.
11. Learning curve: show performance versus number of training subjects or training epochs.
12. Robustness analysis: noise, missing channels, montage changes, artifact-heavy samples, and seizure-boundary uncertainty.
13. Detection-latency analysis: report delay from seizure onset to detection.
14. False-alarm-rate analysis: report false alarms per hour, not only false positives per epoch.
15. Clinical utility analysis: include an operating point that a clinician could plausibly accept.

## Required Data Reporting

The paper must include a complete data audit:

- Dataset name, version/source, access date, and citation.
- Number of subjects, recordings, channels, seizure events, seizure epochs, non-seizure epochs, and total duration.
- Per-subject counts, not only aggregate counts.
- Inclusion and exclusion criteria.
- Any skipped, corrupt, short, or unusable files.
- Window length, overlap, seizure-boundary labeling rule, and preictal/postictal handling.
- Channel/montage harmonization method.
- Artifact handling and filtering steps.
- Split manifest proving no subject leakage.

## Required Model Benchmarking

At minimum, the empirical paper must benchmark:

- Majority or simple prior baseline.
- Logistic regression.
- SVM.
- Random forest.
- Gradient boosting or XGBoost/HistGradientBoosting.
- MLP or shallow neural model.
- At least one raw-signal deep model such as EEGNet, CNN, or TCN if raw EEG is available.
- Final proposed model under the same subject-disjoint protocol.

All models must use the same outer folds. Hyperparameters must be selected without touching the held-out subject.

## Required Metrics

Every main results table must include:

- Accuracy
- Balanced accuracy
- Sensitivity/recall
- Specificity
- Precision/PPV
- NPV
- F1
- MCC
- ROC-AUC
- PR-AUC
- False alarms per hour
- Detection latency
- 95% confidence interval

If space is limited, the main paper may contain the most clinically important metrics, but the supplement or repository must contain the full table.

## Claim Control Rules

Use this language discipline:

- If a result comes from epoch-level cross-validation, call it `epoch-level performance`, not `clinical`, `generalizable`, or `deployment-ready`.
- If a result comes from LOSO, call it `patient-independent` only if subjects are fully disjoint.
- If sensitivity is low, do not hide it behind high accuracy.
- If RGAIG or RAG is not evaluated, describe it only as a possible deployment direction.
- Do not use phrases such as `top 1%`, `Q1-ready`, `state-of-the-art`, or `clinical-grade` unless the evidence directly supports them.
- Do not cite unrelated disease results as epilepsy evidence.

## Review Paper Requirements

The review paper must not be submitted until it has:

- A complete PRISMA flow with exact database counts and exclusion reasons.
- Search strings, databases, search dates, and screening protocol.
- Inclusion and exclusion criteria.
- Inter-rater agreement or a clear screening quality-control process.
- A structured table for every included paper.
- Dataset, validation protocol, leakage risk, model type, metrics, and limitations for every included study.
- Separate analysis for epoch-level, subject-level, cross-dataset, and clinical deployment studies.
- A risk-of-bias assessment.
- A synthesis that explains why reported high accuracies often do not transfer to patient-independent settings.

## Reproducibility Requirements

Before submission, the repository must contain:

- Exact commands to reproduce every table and figure.
- Frozen dependency file or environment file.
- Filled model card; no placeholder fields.
- Real split manifest; no empty subject lists.
- Raw output JSON/CSV for every reported experiment.
- Buildable paper source with no missing included files.
- A `make` target, script, or README section that rebuilds the main results.

## Final Acceptance Checklist

Mark each item before submission:

- [ ] Same-dataset epoch-level versus LOSO comparison completed.
- [ ] Nested LOSO hyperparameter tuning completed.
- [ ] Threshold tuning completed and reported.
- [ ] PR-AUC, MCC, calibration, false alarms/hour, and latency reported.
- [ ] Per-subject results and failure analysis included.
- [ ] Confidence intervals included.
- [ ] Statistical model comparison included.
- [ ] Data audit table included.
- [ ] Split manifest included and verified.
- [ ] Model card completed.
- [ ] RGAIG/RAG claims either evaluated or moved to future work.
- [ ] Review paper PRISMA files complete and buildable.
- [ ] Paper builds from source without missing files.
- [ ] All claims match the actual experimental evidence.

## Current Repository Status Notes

Based on the current repository audit:

- The strongest empirical candidate is `q1_noRGAIG_2col.tex`.
- `q1_full_2col.tex` should not lead with RGAIG unless RGAIG is evaluated.
- The current CHB-MIT LOSO sensitivity is too low to support clinical-deployment claims.
- The current UCI/Bonn epoch-level result and CHB-MIT LOSO result should not be compared as if only the validation protocol changed.
- The review source under `code/paper/` is not fully buildable because required included files are missing there.
- The model card and split manifest need completion before submission.
