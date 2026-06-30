# Epilepsy EEG Paper Analysis Inventory

This is the master inventory of analyses, methods, architectures, preprocessing choices, and benchmarking items that should be considered for a top-tier epilepsy EEG paper. Not every item must become a main contribution, but every important omission must be justified.

## 1. Data Comparison

- UCI/Bonn versus CHB-MIT comparison.
- CHB-MIT epoch-level split versus CHB-MIT subject-level LOSO split.
- Patient-specific versus patient-independent evaluation.
- Pediatric dataset versus adult dataset comparison if available.
- Cross-dataset training/testing: train on one dataset, test on another.
- Per-subject seizure/non-seizure imbalance table.
- Per-recording duration, seizure count, and channel availability.
- Dataset quality comparison: missing channels, corrupt files, artifacts, annotation quality.
- Window-length comparison: 1 s, 2 s, 4 s, 8 s, 16 s.
- Window-overlap comparison: 0%, 25%, 50%, 75%.
- Preictal, ictal, postictal, and interictal labeling comparison.
- Easy dataset versus difficult dataset analysis; do not mix them as if they are equivalent.

## 2. Preprocessing

- Bandpass filtering.
- Notch filtering for powerline noise.
- Artifact rejection.
- Channel selection.
- Montage harmonization.
- Resampling strategy.
- Baseline correction.
- Missing-channel handling.
- Signal clipping/outlier handling.
- Normalization per epoch.
- Normalization per subject.
- Standardization using train-fold statistics only.
- Min-max scaling using train-fold statistics only.
- Robust scaling using median/IQR.
- Z-score standardization.
- Log transform for skewed spectral features.
- One-hot encoding for categorical metadata if metadata is used.

## 3. Signal Transformations

- Raw 1D EEG time-series input.
- Fourier transform / FFT.
- Short-time Fourier transform.
- Wavelet transform.
- Spectrogram representation.
- Time-frequency maps.
- 1D-to-2D conversion using spectrograms.
- 1D-to-2D conversion using channel-by-time matrices.
- 1D-to-2D conversion using topographic EEG maps if channel geometry is available.
- Power spectral density.
- Bandpower extraction.
- Hilbert transform features.
- Entropy-based transformations.

## 4. Feature Engineering

- Time-domain features: mean, variance, RMS, skewness, kurtosis.
- Hjorth parameters: activity, mobility, complexity.
- Line length.
- Zero-crossing rate.
- Peak-to-peak amplitude.
- Frequency-band power: delta, theta, alpha, beta, gamma.
- Bandpower ratios.
- Spectral entropy.
- Shannon entropy.
- Approximate entropy.
- Sample entropy.
- Permutation entropy.
- Fractal dimension.
- Hurst exponent.
- Wavelet energy.
- Cross-channel correlation.
- Coherence.
- Phase-locking value.
- Functional connectivity features.
- Seizure-boundary-aware temporal features.

## 5. Feature Evaluation

- Univariate statistical testing.
- Mutual information ranking.
- ANOVA F-score.
- Correlation with target.
- Redundancy check between features.
- Feature distribution plots by class.
- Feature distribution plots by subject.
- Stability of feature importance across LOSO folds.
- Feature leakage check.
- Clinical interpretability check.

## 6. Feature Selection

- Filter methods: ANOVA, mutual information, correlation pruning.
- Wrapper methods: recursive feature elimination.
- Embedded methods: L1 logistic regression, tree-based importance.
- SHAP-based feature ranking.
- Permutation importance.
- Boruta-style selection.
- PCA.
- ICA.
- Autoencoder-based representation learning.
- Compare full feature set versus selected feature set.
- Select features inside each training fold only.

## 7. Classical ML Approaches

- Majority baseline.
- Logistic regression.
- Naive Bayes.
- k-nearest neighbors.
- Linear SVM.
- RBF SVM.
- Decision tree.
- Random forest.
- Extra trees.
- Gradient boosting.
- HistGradientBoosting.
- XGBoost / LightGBM if available.
- AdaBoost.
- Balanced random forest.
- EasyEnsemble / imbalance-aware ensemble.
- Calibrated classifier.
- Stacking ensemble.
- Voting ensemble.

## 8. Deep Learning Approaches

- MLP on engineered features.
- 1D CNN on raw EEG.
- 2D CNN on spectrograms.
- EEGNet.
- DeepConvNet.
- ShallowConvNet.
- TCN / temporal convolutional network.
- CNN-LSTM.
- CNN-GRU.
- Autoencoder plus classifier.
- Variational autoencoder for representation learning.
- Contrastive/self-supervised pretraining.
- Multi-channel raw EEG model.
- Subject-adaptive neural model.

## 9. RNN Approaches

- Vanilla RNN.
- LSTM.
- BiLSTM.
- GRU.
- BiGRU.
- Attention-LSTM.
- CNN-LSTM.
- CNN-GRU.
- Sequence-to-label seizure detection.
- Sequence-to-sequence temporal labeling.
- Stateful versus stateless RNN comparison.

## 10. Transformer Approaches

- Vanilla transformer encoder.
- Time-series transformer.
- Informer.
- Autoformer.
- FEDformer.
- PatchTST.
- Temporal Fusion Transformer.
- Conformer-style EEG model.
- Channel attention transformer.
- Spectrogram vision transformer.
- Hybrid CNN-transformer.
- Self-supervised transformer pretraining.
- Positional encoding comparison.
- Patch size and sequence length sensitivity.

## 11. Time-Series Approaches

- Sliding-window classification.
- Event-level seizure detection.
- Temporal smoothing.
- Majority voting over windows.
- Hidden Markov model post-processing.
- Conditional random field post-processing.
- Change-point detection.
- Early detection analysis.
- Detection latency analysis.
- False alarms per hour.
- Preictal exclusion analysis.
- Postictal exclusion analysis.
- Subject-specific temporal calibration.

## 12. Cross-Validation Approaches

- Random epoch-level k-fold CV.
- Stratified epoch-level k-fold CV.
- Group k-fold by subject.
- Leave-one-subject-out CV.
- Leave-one-recording-out CV.
- Leave-one-seizure-out CV.
- Nested LOSO for hyperparameter tuning.
- Temporal train/test split within subject.
- Cross-dataset validation.
- External validation.
- Bootstrap confidence intervals.
- Repeated CV with different seeds.

## 13. Statistical Approaches

- Confusion matrix.
- Confidence intervals for every metric.
- McNemar test for paired classification errors.
- Wilcoxon signed-rank test across subject-level metrics.
- Paired bootstrap comparison.
- DeLong test for ROC-AUC where appropriate.
- Effect size reporting.
- Multiple-comparison correction.
- Class imbalance analysis.
- Calibration analysis.
- Reliability curve.
- Brier score.
- Expected calibration error.
- Decision-curve analysis.
- Clinical utility analysis.

## 14. Loss Functions

- Binary cross-entropy.
- Weighted binary cross-entropy.
- Focal loss.
- Class-balanced focal loss.
- Dice loss.
- Tversky loss.
- Focal Tversky loss.
- Hinge loss.
- Squared hinge loss.
- AUC surrogate loss.
- Contrastive loss.
- Triplet loss.
- Supervised contrastive loss.
- Reconstruction loss for autoencoders.
- Multi-task loss if predicting seizure state plus auxiliary labels.

## 15. Optimization and Gradient Descent

- Batch gradient descent.
- Stochastic gradient descent.
- Mini-batch gradient descent.
- SGD with momentum.
- Nesterov momentum.
- Adam.
- AdamW.
- RMSProp.
- Adagrad.
- Learning-rate schedules.
- Cosine decay.
- Reduce-on-plateau.
- Warmup.
- Weight decay.
- Gradient clipping.
- Early stopping.
- Batch size sensitivity.
- Random seed sensitivity.

## 16. Accuracy and Performance Metrics

- Accuracy.
- Balanced accuracy.
- Sensitivity / recall.
- Specificity.
- Precision / PPV.
- NPV.
- F1 score.
- F2 score if recall is prioritized.
- MCC.
- Cohen's kappa.
- ROC-AUC.
- PR-AUC.
- False-positive rate.
- False-negative rate.
- False alarms per hour.
- Detection latency.
- Time-to-detection.
- Event-level sensitivity.
- Epoch-level sensitivity.
- Subject-level macro average.
- Pooled micro average.
- 95% confidence intervals.

## 17. Imbalance Learning

- Class weighting.
- Threshold tuning.
- Random undersampling.
- Random oversampling.
- SMOTE.
- Borderline-SMOTE.
- ADASYN.
- Tomek links.
- EasyEnsemble.
- Balanced random forest.
- Focal loss.
- Cost-sensitive learning.
- Per-subject imbalance analysis.
- Compare metrics before and after imbalance handling.

## 18. Human-in-the-Loop and Feedback

- Clinician review of false positives.
- Clinician review of false negatives.
- Human feedback for uncertain windows.
- Active learning for difficult subjects.
- Expert correction of seizure boundaries.
- Feedback-driven threshold adjustment.
- Confidence-based triage.
- Explainability dashboard for clinician review.
- Audit trail for human decisions.
- Inter-rater agreement if multiple reviewers are used.

## 19. Architecture List

- Feature-based ML pipeline.
- Raw-signal deep learning pipeline.
- Spectrogram-based vision pipeline.
- Hybrid feature plus raw-signal model.
- Ensemble architecture.
- Calibration layer.
- Temporal post-processing layer.
- Human-review layer.
- Monitoring and drift-detection layer.
- Reproducible training pipeline.
- Production inference pipeline.
- Model registry and versioning.
- Data validation pipeline.
- Clinical review interface.

## 20. General Approach List

- Start with clean baselines.
- Prove leakage-free validation.
- Compare epoch-level and patient-independent protocols.
- Optimize sensitivity under fixed specificity.
- Report clinically meaningful metrics.
- Add model complexity only after baselines are strong.
- Use nested validation for tuning.
- Use external validation if possible.
- Analyze failure cases deeply.
- Keep claims matched to evidence.

## 21. Benchmarking

- Compare against simple baseline.
- Compare against classical ML models.
- Compare against deep learning models.
- Compare against transformer/time-series models if data volume supports them.
- Compare patient-specific and patient-independent results.
- Compare against prior literature using the same validation type.
- Separate easy epoch-level papers from hard subject-level papers.
- Include runtime, memory, and inference latency.
- Include ablation results.
- Include statistical significance.

## 22. Production Readiness

- Reproducible environment.
- Fixed random seeds.
- Data versioning.
- Model versioning.
- Split manifest.
- Model card.
- Datasheet for dataset.
- Inference-time preprocessing parity with training.
- Calibration monitoring.
- Drift detection.
- False-alarm monitoring.
- Human override workflow.
- Logging and audit trail.
- Privacy/security review.
- Failure-mode documentation.
- Rollback plan.
- Clinical safety disclaimer.

## 23. Explainability

- Feature importance.
- Permutation importance.
- SHAP.
- LIME if appropriate.
- Saliency maps for deep models.
- Integrated gradients.
- Attention visualization for transformers.
- Per-subject explanation comparison.
- False-positive explanation.
- False-negative explanation.
- Clinician-readable explanation table.

## 24. Must-Have Paper Tables

- Dataset summary table.
- Per-subject data table.
- Preprocessing table.
- Feature list table.
- Model benchmark table.
- Validation protocol comparison table.
- Confusion matrix table.
- Threshold operating-point table.
- Per-subject performance table.
- Ablation table.
- Statistical comparison table.
- Literature comparison table.
- Limitations and mitigation table.

## 25. Must-Have Figures

- Study workflow diagram.
- Data split diagram.
- Class imbalance plot.
- Per-subject seizure count plot.
- ROC curve.
- PR curve.
- Calibration curve.
- Threshold tradeoff curve.
- Per-subject performance plot.
- False-alarm versus sensitivity curve.
- Detection latency distribution.
- Feature importance plot.
- Error analysis examples.

## 26. Red-Line Rules

- Do not report only accuracy.
- Do not compare UCI/Bonn epoch CV against CHB-MIT LOSO as if the dataset stayed constant.
- Do not tune hyperparameters on the test subject.
- Do not select features before the split.
- Do not standardize using all data before the split.
- Do not oversample before the split.
- Do not call epoch-level results deployment-ready.
- Do not claim RGAIG, agentic AI, RAG, or human feedback improves accuracy unless tested.
- Do not hide weak sensitivity behind strong specificity.
- Do not use placeholder model cards or empty split manifests.
