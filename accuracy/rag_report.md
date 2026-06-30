# RAG Clinical Decision-Support Report (epilepsy / seizure detection)

> **Generated offline** (TF-IDF retrieval, no external LLM). For research use; **requires human review**.

**Query:** inter-patient subject-wise seizure detection

## 1. Model prediction (REAL, CHB-MIT LOSO)
- Accuracy: **0.8954**  |  Mean per-subject sensitivity: **0.35083986465958633**  |  Specificity: **0.9729**  |  AUC: **0.8464**
- Honest figure of merit = subject-wise sensitivity (not epoch-level accuracy).

## 2. Key EEG biomarkers (top by ANOVA F-test)
hjorth_activity_mean, rms_mean, line_length_mean, theta_std, theta_mean

## 3. Retrieved evidence (hybrid TF-IDF + keyword, top 5 of 50 indexed)
1. [ep250515203] {EEG-Based Inter-Patient Epileptic Seizure Detection Combining Domain Adversarial Training with CNN-BiLSTM Network (2025) — score 0.3789
2. [ep220800025] {Six-center Assessment of CNN-Transformer with Belief Matching Loss for Patient-independent Seizure Detection in EEG (2022) — score 0.2198
3. [ep231018767] {Enhancing Epileptic Seizure Detection with EEG Feature Embeddings (2023) — score 0.1983
4. [ep250905190] {Accuracy-Constrained CNN Pruning for Efficient and Reliable EEG-Based Seizure Detection (2025) — score 0.1903
5. [ep220305950] {CNN-Aided Factor Graphs with Estimated Mutual Information Features for Seizure Detection (2022) — score 0.1894

## 4. Doctor-facing summary
- Risk-support: model flags seizure epochs with high specificity but **modest, patient-variable
  sensitivity** — use as triage-with-escalation, not autonomous decision.
- Key abnormality drivers: hjorth_activity_mean, rms_mean, line_length_mean, theta_std, theta_mean.
- Evidence: see retrieved citations above.
- **Limitations:** patient-independent sensitivity ~35%; some subjects near 0% — escalate uncertain cases.
- **Recommended next step:** clinician review of flagged epochs + standard EEG read.

## 5. Patient-facing summary
- This is an automated **screening aid**, not a diagnosis.
- It highlights EEG segments a specialist should look at.
- Please follow up with your neurologist for any clinical decision.

## 6. Governance gate (Step 21)
- [ ] Reviewed by psychiatrist / neurophysiologist
- [ ] Approve / reject / request more assessment
- Audit + drift monitoring: see step23_governance.py
