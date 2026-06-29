# Model Card: {{MODEL_NAME}} for {{DISEASE}} Detection

## Model Details

- **Model Name**: {{MODEL_NAME}}
- **Version**: {{VERSION}}
- **Date**: {{DATE}}
- **Type**: EEG Classification
- **Target**: {{DISEASE}} Detection
- **Framework**: scikit-learn / PyTorch

## Intended Use

### Primary Use Cases
- Clinical decision support for {{DISEASE}} screening
- Research applications in neurological disease detection
- Educational purposes in EEG analysis

### Out-of-Scope Uses
- Standalone clinical diagnosis without expert review
- Pediatric populations (unless specifically validated)
- Real-time monitoring without proper clinical oversight

## Training Data

| Property | Value |
|----------|-------|
| **Dataset(s)** | {{DATASETS}} |
| **Total Subjects** | {{N_SUBJECTS}} |
| **Total Samples** | {{N_SAMPLES}} |
| **Class Distribution** | {{CLASS_DISTRIBUTION}} |
| **Split Method** | Subject-wise (no leakage) |

### Preprocessing
- Sampling Rate: {{SAMPLING_RATE}} Hz
- Filtering: {{FILTER_BAND}} Hz bandpass
- Reference: {{REFERENCE_METHOD}}
- Artifact Rejection: {{ARTIFACT_METHOD}}

## Performance Metrics

| Metric | Value | 95% CI |
|--------|-------|--------|
| Accuracy | {{ACCURACY}} | [{{ACC_CI_LOW}}, {{ACC_CI_HIGH}}] |
| F1 (macro) | {{F1_MACRO}} | [{{F1_CI_LOW}}, {{F1_CI_HIGH}}] |
| Sensitivity | {{SENSITIVITY}} | [{{SENS_CI_LOW}}, {{SENS_CI_HIGH}}] |
| Specificity | {{SPECIFICITY}} | [{{SPEC_CI_LOW}}, {{SPEC_CI_HIGH}}] |
| ROC-AUC | {{ROC_AUC}} | [{{AUC_CI_LOW}}, {{AUC_CI_HIGH}}] |
| PR-AUC | {{PR_AUC}} | [{{PR_CI_LOW}}, {{PR_CI_HIGH}}] |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| {{CLASS_0}} | {{P0}} | {{R0}} | {{F0}} | {{S0}} |
| {{CLASS_1}} | {{P1}} | {{R1}} | {{F1}} | {{S1}} |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Random Seed | {{RANDOM_SEED}} |
| Cross-Validation | {{CV_METHOD}} ({{N_FOLDS}} folds) |
| HPO Method | {{HPO_METHOD}} |
| Scoring Metric | {{SCORING}} |
| Class Balancing | {{CLASS_WEIGHT}} |
| Calibration | {{CALIBRATION_METHOD}} |

### Best Hyperparameters

```
{{BEST_PARAMS}}
```

## Limitations and Biases

### Known Limitations
- Model trained on specific EEG hardware/protocol ({{DEVICE}})
- Performance may vary across different patient populations
- Not validated for clinical diagnosis without expert review
- Requires preprocessing consistent with training pipeline

### Potential Biases
- Dataset demographics: {{DEMOGRAPHICS}}
- Potential selection bias in training data
- Performance not validated across all age groups

## Ethical Considerations

- Model outputs should be reviewed by qualified healthcare professionals
- Not to be used as sole basis for clinical decisions
- Patient consent required for data collection and model application
- Results should be interpreted in clinical context
- Regular monitoring for performance drift recommended

## Validation Checks

| Check | Status |
|-------|--------|
| Subject leakage test | {{LEAKAGE_STATUS}} |
| Shuffled label sanity | {{SHUFFLE_STATUS}} |
| Calibration (ECE) | {{CALIBRATION_STATUS}} |
| Robustness to noise | {{ROBUSTNESS_STATUS}} |

## Reproducibility

- **Code Version**: {{GIT_HASH}}
- **Data Version**: {{DATA_HASH}}
- **Environment**: See `requirements.txt`
- **Random Seed**: {{RANDOM_SEED}}

To reproduce results:
```bash
# CHB-MIT Leave-One-Subject-Out (seed fixed at 42 in-script)
CHBMIT_DIR=/path/to/chbmit_edf python code/reproducible/chbmit_loso_pipeline.py
python code/reproducible/comprehensive_accuracy_analysis.py   # -> accuracy/comprehensive_metrics.json
```

## Citation

If using this model, please cite:

```bibtex
@misc{{{MODEL_NAME_LOWER}}_{{DISEASE_LOWER}}_{{YEAR}},
  title={{EEG-based {{DISEASE}} Detection Model}},
  author={{NeuroDiseaseAI Team}},
  year={{{{YEAR}}}},
  note={{Model Card v{{VERSION}}}}
}
```

## Model Files

| File | Description |
|------|-------------|
| `{{MODEL_NAME}}_bundle.pkl` | Complete model bundle (model + scaler + config) |
| `normalizer.pkl` | Fitted normalizer (train-only stats) |
| `splits.json` | Subject-wise split manifest |
| `test_results.json` | Final test metrics with CIs |

## Contact

For questions or issues:
- Repository: {{REPO_URL}}
- Issue Tracker: {{ISSUES_URL}}

---
*Model Card Version: 1.0*
*Generated: {{GENERATION_DATE}}*
