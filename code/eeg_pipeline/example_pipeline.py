#!/usr/bin/env python
"""
Complete EEG Pipeline Example
==============================

This script demonstrates the full 11-phase EEG methodology pipeline
for disease detection. It addresses all the gaps identified in the
codebase review.

Usage:
    python example_pipeline.py --disease depression --data_dir ./datasets/depression_real

"""

import argparse
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Import pipeline modules
from data_loader import EEGDataLoader, SubjectWiseSplitter, GroupKFoldCV, verify_no_leakage
from preprocessing import EEGPreprocessor, PreprocessingConfig
from normalization import LeakageSafeNormalizer, NormalizationQA
from feature_selection import FeatureSelector, StabilitySelection, RiemannianFeatures
from training import EEGTrainer, TrainingConfig, NestedCV
from validation import ValidationSuite, LeakageAuditor, CalibrationValidator
from testing import BootstrapCI, StatisticalTester, TestExecutor
from benchmarking import BenchmarkRunner, ModelCardGenerator, create_executive_summary


def run_pipeline(
    disease: str,
    data_dir: str,
    output_dir: str,
    random_seed: int = 42
):
    """
    Run the complete EEG pipeline.

    Parameters
    ----------
    disease : str
        Target disease (e.g., 'depression', 'epilepsy')
    data_dir : str
        Path to data directory
    output_dir : str
        Path to output directory
    random_seed : int
        Random seed for reproducibility
    """
    print("=" * 70)
    print(f"EEG Pipeline for {disease.upper()} Detection")
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PHASE 2: Data Loading with Subject-Wise Splitting
    # =========================================================================
    print("\n[Phase 2] Loading data with subject-wise splitting...")

    data_loader = EEGDataLoader(data_dir)

    try:
        X, y, subject_ids, metadata = data_loader.load_dataset(disease)
        print(f"  Loaded {len(y)} samples from {metadata.total_subjects} subjects")
        print(f"  Class distribution: {metadata.class_distribution}")
    except FileNotFoundError:
        print(f"  Dataset not found. Creating synthetic data for demonstration...")
        # Create synthetic data for demonstration
        n_subjects = 50
        n_samples_per_subject = 20
        n_channels = 19
        n_timepoints = 1024

        X = np.random.randn(n_subjects * n_samples_per_subject, n_channels, n_timepoints)
        y = np.repeat([0, 1], n_subjects // 2 * n_samples_per_subject)
        subject_ids = np.repeat([f"S{i:03d}" for i in range(n_subjects)], n_samples_per_subject)

    # Subject-wise split
    splitter = SubjectWiseSplitter(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=random_seed,
        stratify=True
    )

    train_info, val_info, test_info = splitter.split(subject_ids, y)

    print(f"  Train: {train_info.n_subjects} subjects, {train_info.n_windows} samples")
    print(f"  Val: {val_info.n_subjects} subjects, {val_info.n_windows} samples")
    print(f"  Test: {test_info.n_subjects} subjects, {test_info.n_windows} samples")

    # Verify no leakage
    verify_no_leakage(
        train_info.subject_ids,
        val_info.subject_ids,
        test_info.subject_ids
    )

    # Save split manifest
    manifest_path = output_dir / "splits.json"
    data_loader.create_split_manifest(
        disease, train_info, val_info, test_info, str(manifest_path)
    )

    # Get data for each split
    X_train = X[train_info.indices]
    y_train = y[train_info.indices]
    groups_train = subject_ids[train_info.indices]

    X_val = X[val_info.indices]
    y_val = y[val_info.indices]
    groups_val = subject_ids[val_info.indices]

    X_test = X[test_info.indices]
    y_test = y[test_info.indices]
    groups_test = subject_ids[test_info.indices]

    # =========================================================================
    # PHASE 3: Preprocessing
    # =========================================================================
    print("\n[Phase 3] Preprocessing...")

    preproc_config = PreprocessingConfig(
        sampling_rate=256.0,
        filter_low=0.5,
        filter_high=45.0,
        notch_freq=60.0,
        epoch_duration=4.0,
        epoch_overlap=0.5
    )

    preprocessor = EEGPreprocessor(preproc_config)

    # Note: In a real scenario, you would preprocess each file
    # Here we assume data is already epoched
    print(f"  Data shape: {X_train.shape}")
    print(f"  Preprocessing config: {preproc_config}")

    # =========================================================================
    # PHASE 4: Normalization (Train-Only)
    # =========================================================================
    print("\n[Phase 4] Fitting normalizer on TRAIN DATA ONLY...")

    normalizer = LeakageSafeNormalizer(
        method='zscore',
        scope='channel_wise'
    )

    # Fit on training data only
    normalizer.fit(X_train)

    # Transform all splits
    X_train_norm = normalizer.transform(X_train)
    X_val_norm = normalizer.transform(X_val, verify_not_train=True)
    X_test_norm = normalizer.transform(X_test, verify_not_train=True)

    # Save normalizer
    normalizer.save(str(output_dir / "normalizer"))

    # QA check
    qa_results = NormalizationQA.verify_train_stats(X_train_norm)
    print(f"  QA Results: {qa_results}")

    # =========================================================================
    # PHASE 5 & 6: Feature Extraction
    # =========================================================================
    print("\n[Phase 5-6] Extracting features...")

    # Flatten for simple features
    X_train_flat = X_train_norm.reshape(len(X_train_norm), -1)
    X_val_flat = X_val_norm.reshape(len(X_val_norm), -1)
    X_test_flat = X_test_norm.reshape(len(X_test_norm), -1)

    print(f"  Feature shape: {X_train_flat.shape}")

    # Feature selection with stability
    print("  Running feature selection...")
    base_selector = FeatureSelector(method='effect_size', n_features=100)
    stability_selector = StabilitySelection(
        base_selector=base_selector,
        n_bootstrap=50,
        threshold=0.6
    )

    stability_selector.fit(X_train_flat, y_train)

    X_train_selected = stability_selector.transform(X_train_flat)
    X_val_selected = stability_selector.transform(X_val_flat)
    X_test_selected = stability_selector.transform(X_test_flat)

    print(f"  Selected features: {X_train_selected.shape[1]}")

    # Try Riemannian features if pyriemann is available
    try:
        print("  Computing Riemannian features...")
        riemann = RiemannianFeatures(estimator='lwf', metric='riemann')
        X_train_riemann = riemann.fit_transform(X_train_norm, y_train)
        X_val_riemann = riemann.transform(X_val_norm)
        X_test_riemann = riemann.transform(X_test_norm)
        has_riemannian = True
        print(f"  Riemannian features: {X_train_riemann.shape[1]}")
    except ImportError:
        print("  pyriemann not available, skipping Riemannian features")
        has_riemannian = False

    # =========================================================================
    # PHASE 7: Model Training with Nested CV
    # =========================================================================
    print("\n[Phase 7] Training models with nested CV...")

    training_config = TrainingConfig(
        random_seed=random_seed,
        n_outer_folds=5,
        n_inner_folds=3,
        scoring='f1_macro',
        class_weight='balanced',
        calibrate=True
    )

    trainer = EEGTrainer(training_config)

    # Train baseline ladder
    baseline_results = trainer.train_baseline_ladder(
        X_train_selected, y_train, groups_train
    )

    # Select best model
    best_model_name = trainer.select_best_model()

    # Train final model
    final_model = trainer.train_final_model(
        best_model_name,
        X_train_selected, y_train,
        X_val_selected, y_val
    )

    # =========================================================================
    # PHASE 8: Validation
    # =========================================================================
    print("\n[Phase 8] Running validation suite...")

    validation_suite = ValidationSuite(
        final_model, X_val_selected, y_val, groups_val
    )
    validation_results = validation_suite.run_all_checks()

    print("  Validation Results:")
    for name, result in validation_results.items():
        status = "✓" if result.passed else "✗"
        print(f"    {status} {name}: {result.value:.4f}")

    # Leakage audit
    print("\n  Running leakage audit...")
    shuffled_test = LeakageAuditor.shuffled_label_test(
        final_model, X_train_selected, y_train
    )

    # =========================================================================
    # PHASE 9: Final Testing (ONE TIME ONLY)
    # =========================================================================
    print("\n[Phase 9] Running FINAL TEST (one-time execution)...")

    test_executor = TestExecutor(str(output_dir / "test_results"))
    test_results = test_executor.execute_test(
        final_model,
        X_test_selected,
        y_test,
        groups_test,
        best_model_name
    )

    # Bootstrap confidence intervals
    bootstrap_ci = BootstrapCI(n_bootstrap=1000, confidence_level=0.95)
    y_pred_test = final_model.predict(X_test_selected)
    y_prob_test = None
    if hasattr(final_model, 'predict_proba'):
        y_prob_test = final_model.predict_proba(X_test_selected)
        if y_prob_test.shape[1] == 2:
            y_prob_test = y_prob_test[:, 1]

    cis = bootstrap_ci.compute_all_metrics_ci(y_test, y_pred_test, y_prob_test)

    print("\n  Test Results with 95% CI:")
    for metric_name, ci in cis.items():
        print(f"    {metric_name}: {ci.estimate:.4f} [{ci.lower:.4f}, {ci.upper:.4f}]")

    # =========================================================================
    # PHASE 10: Benchmarking & Reporting
    # =========================================================================
    print("\n[Phase 10] Generating reports...")

    # Generate model card
    card_generator = ModelCardGenerator(str(output_dir / "reports"))
    card_generator.generate(
        model_name=best_model_name,
        disease=disease,
        metrics={name: ci.estimate for name, ci in cis.items()},
        confidence_intervals={name: (ci.lower, ci.upper) for name, ci in cis.items()},
        training_config={
            'random_seed': random_seed,
            'cv_method': 'GroupKFold',
            'n_folds': 5,
            'scoring': 'F1 (macro)',
            'class_weight': 'balanced'
        },
        dataset_info={
            'datasets': disease,
            'n_subjects': len(np.unique(subject_ids)),
            'n_samples': len(y),
            'class_distribution': str(dict(zip(*np.unique(y, return_counts=True))))
        }
    )

    # Save model bundle
    trainer.save_model_bundle(
        str(output_dir / "model_bundle"),
        best_model_name,
        normalizer=normalizer,
        feature_selector=stability_selector
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nBest Model: {best_model_name}")
    print(f"Test F1 (macro): {cis['f1'].estimate:.4f} [{cis['f1'].lower:.4f}, {cis['f1'].upper:.4f}]")
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - splits.json: Subject-wise split manifest")
    print(f"  - normalizer.pkl: Fitted normalizer (train-only)")
    print(f"  - test_results/: Final test results")
    print(f"  - reports/: Model card and reports")
    print(f"  - model_bundle/: Complete model bundle for deployment")

    return {
        'best_model': best_model_name,
        'test_metrics': {name: ci.estimate for name, ci in cis.items()},
        'confidence_intervals': {name: (ci.lower, ci.upper) for name, ci in cis.items()}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Pipeline for Disease Detection")
    parser.add_argument("--disease", type=str, default="depression",
                        help="Target disease")
    parser.add_argument("--data_dir", type=str, default="./datasets",
                        help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./output/pipeline_results",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    results = run_pipeline(
        disease=args.disease,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        random_seed=args.seed
    )
