# Paper Figures, Charts, Tables, Architectures, and Formulas Checklist

Use this checklist to decide what must appear in the epilepsy EEG paper, supplement, or repository. A top-tier paper should not only report accuracy; it must show data quality, validation rigor, clinical usefulness, and reproducibility.

## 1. Required Graphs and Charts

- Dataset class imbalance bar chart: seizure versus non-seizure epochs.
- Per-subject seizure count chart.
- Per-subject total recording duration chart.
- Per-subject seizure-duration chart.
- Per-subject performance chart for sensitivity, specificity, and AUC.
- ROC curve for each main model.
- PR curve for each main model.
- Calibration curve / reliability plot.
- Threshold tradeoff curve: threshold versus sensitivity, specificity, precision, F1, MCC.
- Sensitivity at fixed specificity plot.
- False alarms per hour versus sensitivity plot.
- Detection latency distribution plot.
- Confusion matrix heatmap.
- Feature importance bar chart.
- Feature correlation heatmap.
- Feature distribution plots for top features.
- Ablation study bar chart.
- Model comparison bar chart.
- Learning curve: number of training subjects versus performance.
- Runtime/inference-latency comparison chart.
- Error analysis chart: false positives and false negatives by subject.
- External validation comparison chart if another dataset is used.

## 2. Required Comparison Figures

- UCI/Bonn epoch-level CV versus CHB-MIT LOSO comparison.
- CHB-MIT epoch-level CV versus CHB-MIT LOSO comparison using the same model and features.
- Patient-specific versus patient-independent comparison.
- Classical ML versus DL versus transformer comparison.
- Feature-based model versus raw-signal model comparison.
- Full feature set versus selected feature set comparison.
- Threshold 0.5 versus tuned threshold comparison.
- Before and after class-imbalance handling comparison.
- Before and after temporal smoothing comparison.
- Internal validation versus external validation comparison.

## 3. Required Tables

- Dataset summary table.
- Per-subject data table.
- Train/test split table.
- Preprocessing configuration table.
- Feature list table.
- Feature selection result table.
- Model architecture table.
- Hyperparameter table.
- Main model benchmark table.
- Validation protocol comparison table.
- Confusion matrix table.
- Per-subject performance table.
- Threshold operating-point table.
- Ablation table.
- Statistical significance table.
- Calibration metric table.
- Runtime and memory table.
- Literature comparison table.
- Error/failure mode table.
- Limitations and mitigation table.
- Reproducibility checklist table.

## 4. Architecture Diagrams

- Full study workflow diagram.
- Data ingestion and preprocessing pipeline.
- Feature extraction pipeline.
- Classical ML pipeline.
- Deep learning pipeline.
- Transformer/time-series pipeline.
- Cross-validation and LOSO split diagram.
- Training and inference architecture.
- Temporal post-processing architecture.
- Human-in-the-loop review architecture.
- Production deployment architecture.
- Monitoring and drift-detection architecture.
- RGAIG/RAG architecture only if it is evaluated; otherwise keep it as future work.

## 5. Mathematical Formulas

Include formulas where they clarify the method or metric.

### Classification Metrics

- Accuracy:
  `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

- Sensitivity / recall:
  `Sensitivity = TP / (TP + FN)`

- Specificity:
  `Specificity = TN / (TN + FP)`

- Precision / PPV:
  `Precision = TP / (TP + FP)`

- Negative predictive value:
  `NPV = TN / (TN + FN)`

- F1 score:
  `F1 = 2 * (Precision * Recall) / (Precision + Recall)`

- Balanced accuracy:
  `Balanced Accuracy = (Sensitivity + Specificity) / 2`

- Matthews correlation coefficient:
  `MCC = ((TP * TN) - (FP * FN)) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))`

### Calibration Metrics

- Brier score:
  `Brier = (1/N) * sum((p_i - y_i)^2)`

- Expected calibration error:
  `ECE = sum((n_m / N) * |acc(B_m) - conf(B_m)|)`

### Signal Processing

- Z-score standardization:
  `z = (x - mean_train) / std_train`

- Min-max normalization:
  `x_scaled = (x - min_train) / (max_train - min_train)`

- Discrete Fourier transform:
  `X_k = sum(x_n * exp(-j * 2*pi*k*n/N))`

- Power spectral density:
  `PSD(f) = |X(f)|^2`

- Bandpower:
  `Bandpower = integral PSD(f) df over the selected frequency band`

### Loss Functions

- Binary cross-entropy:
  `BCE = -[y log(p) + (1-y) log(1-p)]`

- Weighted binary cross-entropy:
  `WBCE = -[w_pos*y log(p) + w_neg*(1-y) log(1-p)]`

- Focal loss:
  `FL = -alpha * (1 - p_t)^gamma * log(p_t)`

- Contrastive loss:
  `L = y*d^2 + (1-y)*max(0, margin-d)^2`

### Optimization

- Gradient descent update:
  `theta = theta - learning_rate * gradient(loss)`

- Weight decay:
  `L_total = L_task + lambda * ||theta||_2^2`

## 6. Accuracy Reporting Rules

- Report accuracy only with sensitivity and specificity.
- Report pooled and per-subject metrics separately.
- Report macro average across subjects.
- Report micro/pooled average across all epochs.
- Report 95% confidence intervals.
- Report PR-AUC for imbalanced seizure detection.
- Report MCC because accuracy can be misleading.
- Report false alarms per hour for clinical relevance.
- Report detection latency for event-level seizure detection.
- Never hide weak sensitivity behind high accuracy.

## 7. Minimum Main-Paper Package

If page space is limited, the main paper should contain at least:

- Dataset summary table.
- Workflow architecture diagram.
- Validation protocol diagram.
- Main benchmark table.
- Per-subject performance plot.
- ROC and PR curves.
- Threshold operating-point table.
- Confusion matrix.
- Feature importance chart.
- Literature comparison table.
- Limitations table.

The supplement or repository should contain the full extended set.
