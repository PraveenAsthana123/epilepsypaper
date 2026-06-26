"""
Model Validation Suite (Phase 8)
=================================

This module implements:
- Validation protocol management
- Calibration validation
- Robustness checks
- Error analysis
- Leakage auditing
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class ValidationResult:
    """Container for validation results."""
    metric_name: str
    value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    passed: bool = True
    threshold: Optional[float] = None


class ValidationSuite:
    """
    Comprehensive validation suite for EEG models.

    Implements all validation checks from Phase 8 of the EEG strategy.
    """

    def __init__(
        self,
        model,
        X_val: np.ndarray,
        y_val: np.ndarray,
        groups_val: np.ndarray
    ):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.groups_val = groups_val

        self._results = {}

    def run_all_checks(self) -> Dict[str, ValidationResult]:
        """Run all validation checks."""
        results = {}

        # 1. Basic performance metrics
        results.update(self._compute_metrics())

        # 2. Per-subject analysis
        results.update(self._per_subject_analysis())

        # 3. Calibration check
        results.update(self._calibration_check())

        # 4. Confusion matrix analysis
        results.update(self._confusion_analysis())

        self._results = results
        return results

    def _compute_metrics(self) -> Dict[str, ValidationResult]:
        """Compute standard metrics."""
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            roc_auc_score, average_precision_score
        )

        y_pred = self.model.predict(self.X_val)
        y_prob = None
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(self.X_val)
            if y_prob.shape[1] == 2:
                y_prob = y_prob[:, 1]

        results = {
            'accuracy': ValidationResult(
                metric_name='accuracy',
                value=accuracy_score(self.y_val, y_pred)
            ),
            'f1_macro': ValidationResult(
                metric_name='f1_macro',
                value=f1_score(self.y_val, y_pred, average='macro')
            ),
            'precision_macro': ValidationResult(
                metric_name='precision_macro',
                value=precision_score(self.y_val, y_pred, average='macro', zero_division=0)
            ),
            'recall_macro': ValidationResult(
                metric_name='recall_macro',
                value=recall_score(self.y_val, y_pred, average='macro', zero_division=0)
            )
        }

        if y_prob is not None:
            try:
                results['roc_auc'] = ValidationResult(
                    metric_name='roc_auc',
                    value=roc_auc_score(self.y_val, y_prob)
                )
                results['pr_auc'] = ValidationResult(
                    metric_name='pr_auc',
                    value=average_precision_score(self.y_val, y_prob)
                )
            except ValueError:
                pass

        return results

    def _per_subject_analysis(self) -> Dict[str, ValidationResult]:
        """Analyze performance per subject."""
        y_pred = self.model.predict(self.X_val)

        subject_accuracies = []
        for subj in np.unique(self.groups_val):
            mask = self.groups_val == subj
            if np.sum(mask) > 0:
                acc = np.mean(y_pred[mask] == self.y_val[mask])
                subject_accuracies.append(acc)

        return {
            'subject_accuracy_mean': ValidationResult(
                metric_name='subject_accuracy_mean',
                value=np.mean(subject_accuracies)
            ),
            'subject_accuracy_std': ValidationResult(
                metric_name='subject_accuracy_std',
                value=np.std(subject_accuracies)
            ),
            'subject_accuracy_min': ValidationResult(
                metric_name='subject_accuracy_min',
                value=np.min(subject_accuracies),
                passed=np.min(subject_accuracies) >= 0.5  # Threshold
            ),
            'n_subjects': ValidationResult(
                metric_name='n_subjects',
                value=len(subject_accuracies)
            )
        }

    def _calibration_check(self) -> Dict[str, ValidationResult]:
        """Check probability calibration."""
        if not hasattr(self.model, 'predict_proba'):
            return {}

        y_prob = self.model.predict_proba(self.X_val)
        if y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
        else:
            return {}  # Multi-class calibration is more complex

        # Expected Calibration Error
        from .training import CalibrationMetrics
        ece = CalibrationMetrics.expected_calibration_error(self.y_val, y_prob)
        brier = CalibrationMetrics.brier_score(self.y_val, y_prob)

        return {
            'ece': ValidationResult(
                metric_name='expected_calibration_error',
                value=ece,
                passed=ece < 0.15,  # Threshold
                threshold=0.15
            ),
            'brier_score': ValidationResult(
                metric_name='brier_score',
                value=brier,
                passed=brier < 0.25,
                threshold=0.25
            )
        }

    def _confusion_analysis(self) -> Dict[str, ValidationResult]:
        """Analyze confusion matrix."""
        from sklearn.metrics import confusion_matrix

        y_pred = self.model.predict(self.X_val)
        cm = confusion_matrix(self.y_val, y_pred)

        # For binary classification
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

            return {
                'sensitivity': ValidationResult(
                    metric_name='sensitivity',
                    value=sensitivity,
                    passed=sensitivity >= 0.8,
                    threshold=0.8
                ),
                'specificity': ValidationResult(
                    metric_name='specificity',
                    value=specificity,
                    passed=specificity >= 0.7,
                    threshold=0.7
                ),
                'false_positive_rate': ValidationResult(
                    metric_name='false_positive_rate',
                    value=fp / (fp + tn) if (fp + tn) > 0 else 0
                ),
                'false_negative_rate': ValidationResult(
                    metric_name='false_negative_rate',
                    value=fn / (fn + tp) if (fn + tp) > 0 else 0
                )
            }

        return {}

    def get_results_table(self) -> 'pd.DataFrame':
        """Get validation results as DataFrame."""
        import pandas as pd

        rows = []
        for name, result in self._results.items():
            rows.append({
                'Metric': result.metric_name,
                'Value': f"{result.value:.4f}" if isinstance(result.value, float) else result.value,
                'Passed': '✓' if result.passed else '✗',
                'Threshold': result.threshold
            })

        return pd.DataFrame(rows)


class CalibrationValidator:
    """Validate probability calibration on held-out data."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def validate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """
        Validate calibration quality.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_prob : np.ndarray
            Predicted probabilities

        Returns
        -------
        metrics : Dict[str, float]
            Calibration metrics
        """
        from .training import CalibrationMetrics

        ece = CalibrationMetrics.expected_calibration_error(y_true, y_prob, self.n_bins)
        brier = CalibrationMetrics.brier_score(y_true, y_prob)

        # Maximum calibration error (worst bin)
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        max_error = 0.0

        for i in range(self.n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(y_true[mask])
                bin_confidence = np.mean(y_prob[mask])
                error = np.abs(bin_accuracy - bin_confidence)
                max_error = max(max_error, error)

        return {
            'ece': ece,
            'mce': max_error,
            'brier_score': brier,
            'calibration_quality': 'good' if ece < 0.1 else 'moderate' if ece < 0.2 else 'poor'
        }


class LeakageAuditor:
    """
    Audit for data leakage in the pipeline.

    Implements sanity checks to detect leakage.
    """

    @staticmethod
    def shuffled_label_test(
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int = 5,
        random_seed: int = 42
    ) -> Dict[str, float]:
        """
        Test model performance with shuffled labels.

        If performance is significantly above chance with shuffled labels,
        there may be leakage in the pipeline.

        Parameters
        ----------
        model : sklearn estimator
            Model to test
        X : np.ndarray
            Features
        y : np.ndarray
            True labels
        n_iterations : int
            Number of shuffle iterations
        random_seed : int
            Random seed

        Returns
        -------
        results : Dict[str, float]
            Shuffled label test results
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.base import clone

        rng = np.random.RandomState(random_seed)
        n_classes = len(np.unique(y))
        chance_level = 1.0 / n_classes

        # Real label performance
        real_scores = cross_val_score(clone(model), X, y, cv=5, scoring='accuracy')
        real_mean = np.mean(real_scores)

        # Shuffled label performance
        shuffled_scores = []
        for _ in range(n_iterations):
            y_shuffled = rng.permutation(y)
            scores = cross_val_score(clone(model), X, y_shuffled, cv=5, scoring='accuracy')
            shuffled_scores.append(np.mean(scores))

        shuffled_mean = np.mean(shuffled_scores)
        shuffled_std = np.std(shuffled_scores)

        # Check if shuffled performance is above chance + 2 std
        leakage_suspected = shuffled_mean > (chance_level + 2 * shuffled_std)

        results = {
            'real_accuracy': real_mean,
            'shuffled_accuracy_mean': shuffled_mean,
            'shuffled_accuracy_std': shuffled_std,
            'chance_level': chance_level,
            'leakage_suspected': leakage_suspected
        }

        if leakage_suspected:
            warnings.warn(
                f"POTENTIAL LEAKAGE: Shuffled label accuracy ({shuffled_mean:.3f}) "
                f"is above chance ({chance_level:.3f}) + 2*std ({shuffled_std:.3f})"
            )
        else:
            print("✓ Shuffled label test passed - no obvious leakage")

        return results

    @staticmethod
    def subject_id_prediction_test(
        X: np.ndarray,
        subject_ids: np.ndarray,
        random_seed: int = 42
    ) -> Dict[str, float]:
        """
        Test if features can predict subject IDs.

        If features strongly predict subject IDs, the model may be
        learning subject-specific patterns rather than disease patterns.

        Parameters
        ----------
        X : np.ndarray
            Features
        subject_ids : np.ndarray
            Subject IDs
        random_seed : int
            Random seed

        Returns
        -------
        results : Dict[str, float]
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder

        # Encode subject IDs
        le = LabelEncoder()
        y_subjects = le.fit_transform(subject_ids)

        n_subjects = len(np.unique(y_subjects))
        chance_level = 1.0 / n_subjects

        # Train classifier to predict subject
        clf = RandomForestClassifier(n_estimators=50, random_state=random_seed, max_depth=10)

        # Use limited features to avoid overfitting
        X_limited = X[:, :min(50, X.shape[1])]

        scores = cross_val_score(clf, X_limited, y_subjects, cv=min(5, n_subjects), scoring='accuracy')
        mean_score = np.mean(scores)

        # High accuracy in predicting subjects suggests subject-specific features
        high_subject_predictability = mean_score > (chance_level + 0.3)

        results = {
            'subject_prediction_accuracy': mean_score,
            'chance_level': chance_level,
            'n_subjects': n_subjects,
            'high_subject_predictability': high_subject_predictability
        }

        if high_subject_predictability:
            warnings.warn(
                f"Features strongly predict subject ID ({mean_score:.3f} vs chance {chance_level:.3f}). "
                "Model may be learning subject-specific patterns."
            )
        else:
            print("✓ Subject ID prediction test passed")

        return results


class RobustnessChecker:
    """Check model robustness under various perturbations."""

    def __init__(self, model):
        self.model = model

    def noise_robustness(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]
    ) -> Dict[str, float]:
        """
        Test model robustness to Gaussian noise.

        Parameters
        ----------
        X : np.ndarray
            Clean features
        y : np.ndarray
            Labels
        noise_levels : List[float]
            Noise standard deviations as fraction of feature std

        Returns
        -------
        results : Dict[str, float]
            Accuracy at each noise level
        """
        from sklearn.metrics import accuracy_score

        results = {}

        # Baseline (no noise)
        y_pred = self.model.predict(X)
        baseline_acc = accuracy_score(y, y_pred)
        results['baseline'] = baseline_acc

        # Add noise
        feature_std = np.std(X, axis=0)

        for noise_level in noise_levels:
            noise = np.random.randn(*X.shape) * feature_std * noise_level
            X_noisy = X + noise

            y_pred = self.model.predict(X_noisy)
            acc = accuracy_score(y, y_pred)
            results[f'noise_{noise_level}'] = acc

            # Check degradation
            degradation = baseline_acc - acc
            if degradation > 0.1:
                warnings.warn(f"Large degradation ({degradation:.3f}) at noise level {noise_level}")

        return results

    def missing_channel_robustness(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_channels: int,
        drop_fractions: List[float] = [0.1, 0.2, 0.3]
    ) -> Dict[str, float]:
        """
        Test robustness to missing channels (set to zero).

        Parameters
        ----------
        X : np.ndarray
            Features (n_samples, n_features)
        y : np.ndarray
            Labels
        n_channels : int
            Number of EEG channels
        drop_fractions : List[float]
            Fraction of channels to drop

        Returns
        -------
        results : Dict[str, float]
        """
        from sklearn.metrics import accuracy_score

        results = {}

        # Baseline
        y_pred = self.model.predict(X)
        baseline_acc = accuracy_score(y, y_pred)
        results['baseline'] = baseline_acc

        for drop_frac in drop_fractions:
            n_drop = max(1, int(n_channels * drop_frac))

            # Drop random channels (set features to zero)
            X_dropped = X.copy()

            # Assuming features are organized by channel
            features_per_channel = X.shape[1] // n_channels

            for _ in range(10):  # Average over multiple random drops
                drop_channels = np.random.choice(n_channels, n_drop, replace=False)
                for ch in drop_channels:
                    start_idx = ch * features_per_channel
                    end_idx = start_idx + features_per_channel
                    if end_idx <= X.shape[1]:
                        X_dropped[:, start_idx:end_idx] = 0

            y_pred = self.model.predict(X_dropped)
            acc = accuracy_score(y, y_pred)
            results[f'drop_{drop_frac}'] = acc

        return results
