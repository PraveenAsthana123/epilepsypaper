# EEG Pipeline: 11-Phase Methodology Implementation

A rigorous, leakage-safe EEG ML pipeline for neurological disease detection.

## Features

This pipeline addresses all critical gaps identified in the codebase review:

| Gap | Implementation |
|-----|---------------|
| Subject-wise splits | `SubjectWiseSplitter` with leakage verification |
| Train-only normalization | `LeakageSafeNormalizer` with saved scalers |
| Nested CV for HPO | `NestedCV` with GroupKFold |
| Riemannian baseline | `RiemannianFeatures` (pyriemann) |
| Probability calibration | `TemperatureScaling`, `PlattScaling` |
| Statistical tests | `McNemar`, `paired bootstrap`, `DeLong` |
| Bootstrap CI | `BootstrapCI` with percentile/BCa methods |
| Model cards | `ModelCardGenerator` |
| **25+ Metrics** | `ComprehensiveMetrics` with clinical, LOSO metrics |
| **Clinical validation** | `ClinicalValidationReportGenerator` |
| **Reliability analysis** | `ReliabilityReportGenerator` |
| **Feature analysis** | `FeatureAnalysisReportGenerator` |

## Quick Start

```python
from eeg_pipeline import (
    EEGDataLoader,
    SubjectWiseSplitter,
    LeakageSafeNormalizer,
    EEGTrainer,
    BootstrapCI
)

# 1. Load data with subject-wise splits
loader = EEGDataLoader('./datasets')
X, y, subjects, metadata = loader.load_dataset('depression')

splitter = SubjectWiseSplitter()
train, val, test = splitter.split(subjects, y)

# 2. Normalize (train-only)
normalizer = LeakageSafeNormalizer()
normalizer.fit(X[train.indices])
X_train_norm = normalizer.transform(X[train.indices])
X_test_norm = normalizer.transform(X[test.indices])

# 3. Train with nested CV
trainer = EEGTrainer()
results = trainer.train_baseline_ladder(X_train_norm, y[train.indices], subjects[train.indices])

# 4. Test with CIs
bootstrap = BootstrapCI(n_bootstrap=1000)
cis = bootstrap.compute_all_metrics_ci(y_test, y_pred, y_prob)
```

## Pipeline Structure

```
eeg_pipeline/
├── __init__.py                      # Package exports
├── configs/
│   ├── spec.yaml                    # Project specification (Phase 1)
│   ├── splits_template.json
│   └── norm_stats_template.json
├── templates/
│   └── model_card_template.md       # Model card template
│
│ # Core Pipeline Modules
├── data_loader.py                   # Phase 2: Subject-wise loading & splitting
├── preprocessing.py                 # Phase 3: Filtering, artifact rejection
├── normalization.py                 # Phase 4: Leakage-safe normalization
├── feature_selection.py             # Phase 5-6: Riemannian + selection
├── training.py                      # Phase 7: Nested CV, calibration
├── validation.py                    # Phase 8: Validation suite, leakage audit
├── testing.py                       # Phase 9: Bootstrap CI, statistical tests
├── benchmarking.py                  # Phase 10: Model cards, reports
│
│ # Extended Analysis Modules
├── metrics.py                       # Comprehensive metrics (25+ types)
├── clinical_validation.py           # Clinical validation suite
├── reliability_analysis.py          # Reliability & robustness testing
├── feature_engineering_analysis.py  # Feature importance & ablation
│
├── example_pipeline.py              # Complete example
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## Critical Leakage Prevention

1. **Subject-wise splits**: All samples from one subject stay in the same split
2. **Train-only normalization**: Statistics computed ONLY from training data
3. **Nested CV**: Inner loop for HPO, outer loop for unbiased evaluation
4. **Verification**: Automated leakage checks at every stage

## Usage

Run the complete pipeline:

```bash
python -m eeg_pipeline.example_pipeline \
    --disease depression \
    --data_dir ./datasets \
    --output_dir ./results
```

## Config Templates

### spec.yaml (Phase 1 - freeze before modeling)

```yaml
project:
  name: "EEG Disease Detection"
  experiment_id: "EXP_001"

split_strategy:
  method: "subject_wise"
  ratios: {train: 0.70, val: 0.15, test: 0.15}

evaluation:
  primary_metric: "macro_f1"
  clinical_constraints:
    minimum_sensitivity: 0.85
```

### splits.json (Phase 2 - verify no overlap)

```json
{
  "splits": {
    "train": {"subject_ids": ["S001", "S002", ...]},
    "validation": {"subject_ids": ["S010", "S011", ...]},
    "test": {"subject_ids": ["S020", "S021", ...]}
  },
  "_leakage_checks": {
    "train_val_overlap": [],
    "all_checks_passed": true
  }
}
```

## Requirements

```
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
pyriemann>=0.3.0  # For Riemannian features
```

Install:
```bash
pip install -r requirements.txt
```

## Extended Analysis Modules

### Comprehensive Metrics (`metrics.py`)

25+ performance metrics organized by category:

```python
from eeg_pipeline import ComprehensiveMetrics, ClinicalMetrics

# Compute all metrics at once
metrics_calc = ComprehensiveMetrics(include_clinical=True, prevalence=0.1)
results = metrics_calc.compute_all(y_true, y_pred, y_prob, subject_ids)

# Generate formatted report
report = metrics_calc.generate_report(results, format='markdown')
```

**Metric Categories:**
- **Classification**: Accuracy, Precision, Recall, Specificity, F1, F2, MCC, Kappa, G-Mean, Youden's J
- **Probability**: ROC-AUC, PR-AUC, Log Loss, Brier Score, ECE, MCE
- **Clinical**: LR+, LR-, NND, CUI+, CUI-, NPV, DOR, Post-test probability
- **Subject-wise**: Per-subject accuracy, sensitivity, specificity with aggregation

### Clinical Validation (`clinical_validation.py`)

Comprehensive clinical validation suite:

```python
from eeg_pipeline import ClinicalValidationReportGenerator

validator = ClinicalValidationReportGenerator(model, 'RandomForest', 'depression')

report = validator.generate_full_report(
    X_test, y_test,
    demographics={'age_group': age_groups, 'sex': sex_labels},
    timestamps=recording_dates,
    external_datasets=[(X_ext, y_ext, 'External_Dataset')]
)

validator.export_report(report, 'clinical_validation.json')
```

**Features:**
- Cross-dataset generalization analysis
- Demographic subgroup analysis with disparity detection
- Temporal stability testing (performance drift)
- Failure mode analysis
- Equalized odds fairness metrics
- Regulatory compliance notes

### Reliability Analysis (`reliability_analysis.py`)

Test model reliability and robustness:

```python
from eeg_pipeline import ReliabilityReportGenerator

reliability = ReliabilityReportGenerator(model, sampling_rate=256.0)

report = reliability.generate_report(
    X_test, y_test,
    X_retest=X_session2,  # Optional: for test-retest
    subject_ids=subjects
)
```

**Tests:**
- **Test-retest reliability**: ICC, Pearson correlation, Bland-Altman analysis
- **Noise robustness**: Gaussian noise, EMG artifacts, eye movements, electrode drift, powerline
- **Channel robustness**: Random channel dropout, specific channel loss
- **Signal quality analysis**: SQI-performance relationship, quality thresholds

### Feature Engineering Analysis (`feature_engineering_analysis.py`)

Comprehensive feature importance and validation:

```python
from eeg_pipeline import FeatureAnalysisReportGenerator

analyzer = FeatureAnalysisReportGenerator(model, feature_names, disease='depression')

report = analyzer.generate_report(X_train, y_train, X_test, y_test)
```

**Analysis:**
- **Importance methods**: Permutation, effect size (Cohen's d), model-based
- **Stability analysis**: Bootstrap selection frequency, Kuncheva index
- **Ablation studies**: Group ablation, progressive ablation curves
- **Domain validation**: Validate against known EEG biomarkers
- **Interpretability**: Generate human-readable feature interpretations

### EDA Analysis (`eda_analysis.py`)

Comprehensive 20-point EDA framework:

```python
from eeg_pipeline import EDAReportGenerator

eda = EDAReportGenerator(feature_names=feature_names)
report = eda.generate_full_report(X, y, subject_ids, timestamps, dataset_name='MyEEG')
eda.export_report(report, 'eda_report.md', format='markdown')
```

**Analysis Categories (20 points):**
- **Dataset Overview (1-4)**: Schema validation, missing values, duplicates
- **Distribution (5-8)**: Statistics, skewness, outliers, range validity
- **Target & Correlation (9-14)**: Class balance, feature correlation, multicollinearity
- **Temporal & Group (15-16)**: Drift detection, subject heterogeneity
- **Quality Assessment (17-20)**: Noise, leakage suspicion, risk register

### Outlier Analysis (`outlier_analysis.py`)

Comprehensive 20-point outlier detection:

```python
from eeg_pipeline import OutlierReportGenerator

outlier_analyzer = OutlierReportGenerator(sampling_rate=256.0)
report = outlier_analyzer.generate_full_report(X, y, channel_names, y_pred)
```

**Detection Methods (20 points):**
- **Amplitude-based (1-4)**: Physiological limits, Z-score, channel-wise, spikes
- **Frequency-based (5-8)**: Band power anomalies, drift, flatline, saturation
- **Feature-space (14-17)**: Isolation Forest, Mahalanobis, class-conditional
- **Handling strategies**: Remove, winsorize, robust methods

### Filter Analysis (`filter_analysis.py`)

Comprehensive 20-point filter analysis:

```python
from eeg_pipeline import FilterReportGenerator, FilterConfig

filter_analyzer = FilterReportGenerator(sampling_rate=256.0)
config = FilterConfig(filter_type='bandpass', low_freq=0.5, high_freq=45.0)
report = filter_analyzer.generate_full_report(X, config)
```

**Analysis Points:**
- **Sampling validation (1-2)**: Nyquist, baseline wander
- **Filter design (3-7)**: Cutoff selection, notch filter, type comparison
- **Quality analysis (8-14)**: Phase distortion, edge artifacts, SNR improvement

### Data Conversion (`data_conversion.py`)

1D to 2D transformation methods:

```python
from eeg_pipeline import ConversionPipeline, ConversionAnalyzer

# Compare methods
analyzer = ConversionAnalyzer(sampling_rate=256.0)
comparisons = analyzer.compare_methods(X, y, methods=['fft', 'psd', 'spectrogram', 'gaf'])

# Apply conversion
pipeline = ConversionPipeline(sampling_rate=256.0)
X_spec = pipeline.fit_transform(X, method='spectrogram')
X_test_spec = pipeline.transform(X_test, method='spectrogram')
```

**Conversion Methods:**
- **Frequency domain**: FFT features, PSD, STFT spectrogram
- **Wavelet**: CWT scalogram, band-power images
- **Advanced**: Gramian Angular Field (GAF), Markov Transition Field (MTF), Recurrence plots

## Complete Module Summary

| Module | Analysis Points | Framework |
|--------|-----------------|-----------|
| `metrics.py` | 25+ metrics | Classification, Clinical, LOSO |
| `clinical_validation.py` | Cross-dataset, subgroup, temporal | Clinical Validation Matrix |
| `reliability_analysis.py` | Test-retest, noise, channel | Reliability Matrix |
| `feature_engineering_analysis.py` | 20 analysis types | Feature Engineering Framework |
| `eda_analysis.py` | 20 analysis types | EDA 4-Column Framework |
| `outlier_analysis.py` | 20 analysis types | Outlier 4-Column Framework |
| `filter_analysis.py` | 20 analysis types | Filter 4-Column Framework |
| `data_conversion.py` | 20 conversion types | 1D→2D 4-Column Framework |

## License

MIT
