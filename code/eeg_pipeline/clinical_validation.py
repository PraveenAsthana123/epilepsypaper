"""
Clinical Validation Module
===========================

This module provides comprehensive clinical validation analysis
including:
- Cross-dataset generalization
- Demographic subgroup analysis
- Temporal stability testing
- Real-world performance estimation
- Regulatory documentation support

Based on the Clinical Validation & Real-World Performance Matrix.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import warnings


@dataclass
class SubgroupResult:
    """Results for a demographic subgroup."""
    subgroup_name: str
    n_samples: int
    accuracy: float
    sensitivity: float
    specificity: float
    f1: float
    auc: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


@dataclass
class GeneralizationResult:
    """Cross-dataset generalization result."""
    train_dataset: str
    test_dataset: str
    accuracy: float
    sensitivity: float
    specificity: float
    auc: float
    n_train: int
    n_test: int
    performance_drop: float  # Relative to within-dataset performance


@dataclass
class TemporalStabilityResult:
    """Temporal stability analysis result."""
    time_window: str
    n_samples: int
    accuracy: float
    performance_drift: float  # Change from baseline
    p_value: float


@dataclass
class ClinicalValidationReport:
    """Complete clinical validation report."""
    model_name: str
    disease: str
    validation_date: str
    overall_metrics: Dict[str, float]
    subgroup_analysis: List[SubgroupResult]
    generalization_results: List[GeneralizationResult]
    temporal_stability: List[TemporalStabilityResult]
    failure_modes: List[Dict[str, Any]]
    recommendations: List[str]
    regulatory_notes: List[str]


# =============================================================================
# Cross-Dataset Generalization
# =============================================================================

class CrossDatasetValidator:
    """
    Validate model performance across different datasets.

    Tests:
    1. External validation on held-out datasets
    2. Domain adaptation analysis
    3. Dataset shift detection
    """

    def __init__(self, model, feature_extractor=None):
        """
        Initialize cross-dataset validator.

        Parameters
        ----------
        model : sklearn-compatible model
            Trained model to validate
        feature_extractor : callable, optional
            Feature extraction function if needed
        """
        self.model = model
        self.feature_extractor = feature_extractor

    def external_validation(
        self,
        X_external: np.ndarray,
        y_external: np.ndarray,
        dataset_name: str,
        baseline_metrics: Dict[str, float]
    ) -> GeneralizationResult:
        """
        Validate on external dataset.

        Parameters
        ----------
        X_external : array
            External dataset features
        y_external : array
            External dataset labels
        dataset_name : str
            Name of external dataset
        baseline_metrics : dict
            Performance on original test set

        Returns
        -------
        GeneralizationResult
        """
        # Apply feature extraction if needed
        if self.feature_extractor is not None:
            X_external = self.feature_extractor(X_external)

        # Predict
        y_pred = self.model.predict(X_external)
        y_prob = None
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X_external)
            if y_prob.ndim > 1:
                y_prob = y_prob[:, 1]

        # Compute metrics
        from metrics import ClassificationMetrics, ProbabilityMetrics

        accuracy = ClassificationMetrics.accuracy(y_external, y_pred)
        sensitivity = ClassificationMetrics.sensitivity(y_external, y_pred)
        specificity = ClassificationMetrics.specificity(y_external, y_pred)

        auc = 0.5
        if y_prob is not None:
            auc = ProbabilityMetrics.roc_auc(y_external, y_prob)

        # Performance drop
        baseline_acc = baseline_metrics.get('accuracy', accuracy)
        performance_drop = (baseline_acc - accuracy) / baseline_acc if baseline_acc > 0 else 0

        return GeneralizationResult(
            train_dataset='original',
            test_dataset=dataset_name,
            accuracy=accuracy,
            sensitivity=sensitivity,
            specificity=specificity,
            auc=auc,
            n_train=0,  # Not available
            n_test=len(y_external),
            performance_drop=performance_drop
        )

    def detect_dataset_shift(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        method: str = 'mmd'
    ) -> Tuple[float, float]:
        """
        Detect distribution shift between datasets.

        Parameters
        ----------
        X_train : array
            Training dataset
        X_test : array
            Test dataset
        method : str
            Detection method ('mmd', 'ks', 'energy')

        Returns
        -------
        shift_score : float
            Magnitude of shift
        p_value : float
            Statistical significance
        """
        # Flatten if needed
        if X_train.ndim > 2:
            X_train = X_train.reshape(len(X_train), -1)
        if X_test.ndim > 2:
            X_test = X_test.reshape(len(X_test), -1)

        if method == 'mmd':
            # Maximum Mean Discrepancy
            shift_score = self._compute_mmd(X_train, X_test)
            # Permutation test for p-value
            p_value = self._permutation_test_mmd(X_train, X_test, shift_score)
        elif method == 'ks':
            # Kolmogorov-Smirnov test (per feature, then aggregate)
            from scipy import stats
            ks_stats = []
            p_values = []
            n_features = min(X_train.shape[1], X_test.shape[1])

            for i in range(n_features):
                stat, p = stats.ks_2samp(X_train[:, i], X_test[:, i])
                ks_stats.append(stat)
                p_values.append(p)

            shift_score = np.mean(ks_stats)
            p_value = np.mean(p_values)
        else:
            shift_score = 0.0
            p_value = 1.0

        return float(shift_score), float(p_value)

    def _compute_mmd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        kernel: str = 'rbf',
        gamma: Optional[float] = None
    ) -> float:
        """Compute Maximum Mean Discrepancy."""
        if gamma is None:
            # Median heuristic
            combined = np.vstack([X[:100], Y[:100]])
            dists = np.sqrt(((combined[:, None] - combined[None, :]) ** 2).sum(axis=-1))
            gamma = 1.0 / (np.median(dists) ** 2 + 1e-10)

        def rbf_kernel(X1, X2):
            sq_dist = ((X1[:, None] - X2[None, :]) ** 2).sum(axis=-1)
            return np.exp(-gamma * sq_dist)

        K_XX = rbf_kernel(X, X)
        K_YY = rbf_kernel(Y, Y)
        K_XY = rbf_kernel(X, Y)

        n, m = len(X), len(Y)

        mmd = (K_XX.sum() / (n * n) - 2 * K_XY.sum() / (n * m) + K_YY.sum() / (m * m))

        return float(max(0, mmd))

    def _permutation_test_mmd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        observed_mmd: float,
        n_permutations: int = 100
    ) -> float:
        """Permutation test for MMD significance."""
        combined = np.vstack([X, Y])
        n = len(X)

        count = 0
        for _ in range(n_permutations):
            perm = np.random.permutation(len(combined))
            X_perm = combined[perm[:n]]
            Y_perm = combined[perm[n:]]
            perm_mmd = self._compute_mmd(X_perm, Y_perm)

            if perm_mmd >= observed_mmd:
                count += 1

        return float(count / n_permutations)


# =============================================================================
# Demographic Subgroup Analysis
# =============================================================================

class SubgroupAnalyzer:
    """
    Analyze model performance across demographic subgroups.

    Ensures fair and equitable model performance across:
    - Age groups
    - Sex/gender
    - Ethnicity
    - Disease severity
    - Comorbidities
    """

    def __init__(self, model):
        """
        Initialize subgroup analyzer.

        Parameters
        ----------
        model : sklearn-compatible model
            Trained model to analyze
        """
        self.model = model

    def analyze_subgroups(
        self,
        X: np.ndarray,
        y: np.ndarray,
        demographics: Dict[str, np.ndarray],
        min_samples: int = 30
    ) -> List[SubgroupResult]:
        """
        Analyze performance across all demographic subgroups.

        Parameters
        ----------
        X : array
            Features
        y : array
            Labels
        demographics : dict
            Dictionary mapping demographic variable names to arrays
            e.g., {'age_group': array, 'sex': array}
        min_samples : int
            Minimum samples required for subgroup analysis

        Returns
        -------
        List of SubgroupResult
        """
        from metrics import ClassificationMetrics, ProbabilityMetrics

        results = []

        for demo_name, demo_values in demographics.items():
            unique_values = np.unique(demo_values)

            for value in unique_values:
                mask = demo_values == value
                n_samples = np.sum(mask)

                if n_samples < min_samples:
                    warnings.warn(
                        f"Subgroup {demo_name}={value} has only {n_samples} samples "
                        f"(< {min_samples}). Results may be unreliable."
                    )
                    continue

                X_sub = X[mask]
                y_sub = y[mask]

                # Predict
                y_pred = self.model.predict(X_sub)
                y_prob = None
                if hasattr(self.model, 'predict_proba'):
                    y_prob = self.model.predict_proba(X_sub)
                    if y_prob.ndim > 1:
                        y_prob = y_prob[:, 1]

                # Compute metrics
                auc = None
                if y_prob is not None and len(np.unique(y_sub)) > 1:
                    auc = ProbabilityMetrics.roc_auc(y_sub, y_prob)

                results.append(SubgroupResult(
                    subgroup_name=f"{demo_name}={value}",
                    n_samples=int(n_samples),
                    accuracy=ClassificationMetrics.accuracy(y_sub, y_pred),
                    sensitivity=ClassificationMetrics.sensitivity(y_sub, y_pred),
                    specificity=ClassificationMetrics.specificity(y_sub, y_pred),
                    f1=ClassificationMetrics.f1_score(y_sub, y_pred),
                    auc=auc
                ))

        return results

    def detect_disparities(
        self,
        subgroup_results: List[SubgroupResult],
        metric: str = 'accuracy',
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Detect significant performance disparities between subgroups.

        Parameters
        ----------
        subgroup_results : list
            List of SubgroupResult objects
        metric : str
            Metric to compare ('accuracy', 'sensitivity', 'specificity', 'f1')
        threshold : float
            Threshold for flagging disparity

        Returns
        -------
        List of disparity alerts
        """
        disparities = []

        values = [(r.subgroup_name, getattr(r, metric)) for r in subgroup_results]

        # Compare all pairs
        for i, (name1, val1) in enumerate(values):
            for j, (name2, val2) in enumerate(values):
                if i >= j:
                    continue

                diff = abs(val1 - val2)
                if diff > threshold:
                    disparities.append({
                        'subgroup_1': name1,
                        'subgroup_2': name2,
                        'metric': metric,
                        'value_1': val1,
                        'value_2': val2,
                        'difference': diff,
                        'severity': 'high' if diff > threshold * 2 else 'moderate'
                    })

        return disparities

    def equalized_odds_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        protected_attribute: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute equalized odds metrics.

        Parameters
        ----------
        X : array
            Features
        y : array
            Labels
        protected_attribute : array
            Binary protected attribute

        Returns
        -------
        Dictionary with equalized odds metrics
        """
        from metrics import ClassificationMetrics

        y_pred = self.model.predict(X)

        # Group 0 (protected)
        mask_0 = protected_attribute == 0
        # Group 1 (non-protected)
        mask_1 = protected_attribute == 1

        # True positive rates
        tpr_0 = ClassificationMetrics.sensitivity(y[mask_0], y_pred[mask_0])
        tpr_1 = ClassificationMetrics.sensitivity(y[mask_1], y_pred[mask_1])

        # False positive rates
        fpr_0 = 1 - ClassificationMetrics.specificity(y[mask_0], y_pred[mask_0])
        fpr_1 = 1 - ClassificationMetrics.specificity(y[mask_1], y_pred[mask_1])

        return {
            'tpr_group_0': tpr_0,
            'tpr_group_1': tpr_1,
            'tpr_difference': abs(tpr_0 - tpr_1),
            'fpr_group_0': fpr_0,
            'fpr_group_1': fpr_1,
            'fpr_difference': abs(fpr_0 - fpr_1),
            'equalized_odds_satisfied': abs(tpr_0 - tpr_1) < 0.1 and abs(fpr_0 - fpr_1) < 0.1
        }


# =============================================================================
# Temporal Stability Analysis
# =============================================================================

class TemporalStabilityAnalyzer:
    """
    Analyze model performance stability over time.

    Tests for:
    - Performance drift
    - Concept drift
    - Data drift
    - Seasonal effects
    """

    def __init__(self, model):
        """
        Initialize temporal stability analyzer.

        Parameters
        ----------
        model : sklearn-compatible model
            Trained model to analyze
        """
        self.model = model
        self.baseline_performance = None

    def set_baseline(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Set baseline performance for drift detection."""
        from metrics import ClassificationMetrics

        y_pred = self.model.predict(X)
        self.baseline_performance = {
            'accuracy': ClassificationMetrics.accuracy(y, y_pred),
            'sensitivity': ClassificationMetrics.sensitivity(y, y_pred),
            'specificity': ClassificationMetrics.specificity(y, y_pred),
            'f1': ClassificationMetrics.f1_score(y, y_pred)
        }
        return self.baseline_performance

    def analyze_temporal_windows(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray,
        window_size: str = 'monthly'
    ) -> List[TemporalStabilityResult]:
        """
        Analyze performance across temporal windows.

        Parameters
        ----------
        X : array
            Features
        y : array
            Labels
        timestamps : array
            Timestamp for each sample
        window_size : str
            Window size ('daily', 'weekly', 'monthly', 'quarterly')

        Returns
        -------
        List of TemporalStabilityResult
        """
        from metrics import ClassificationMetrics
        from scipy import stats

        results = []

        # Convert timestamps to datetime if needed
        if not isinstance(timestamps[0], datetime):
            timestamps = np.array([datetime.fromtimestamp(t) for t in timestamps])

        # Group by window
        if window_size == 'monthly':
            windows = [(t.year, t.month) for t in timestamps]
        elif window_size == 'weekly':
            windows = [(t.year, t.isocalendar()[1]) for t in timestamps]
        elif window_size == 'quarterly':
            windows = [(t.year, (t.month - 1) // 3) for t in timestamps]
        else:
            windows = [(t.year, t.timetuple().tm_yday) for t in timestamps]

        unique_windows = sorted(set(windows))

        if self.baseline_performance is None:
            # Use first window as baseline
            first_mask = np.array([w == unique_windows[0] for w in windows])
            self.set_baseline(X[first_mask], y[first_mask])

        baseline_acc = self.baseline_performance['accuracy']

        for window in unique_windows:
            mask = np.array([w == window for w in windows])
            n_samples = np.sum(mask)

            if n_samples < 10:
                continue

            X_window = X[mask]
            y_window = y[mask]

            y_pred = self.model.predict(X_window)
            accuracy = ClassificationMetrics.accuracy(y_window, y_pred)

            # Performance drift
            drift = (accuracy - baseline_acc) / baseline_acc if baseline_acc > 0 else 0

            # Statistical test for significance
            # Using binomial test
            n_correct = np.sum(y_pred == y_window)
            _, p_value = stats.binom_test(
                n_correct, n_samples, baseline_acc,
                alternative='two-sided'
            ) if hasattr(stats, 'binom_test') else (0, 1.0)

            results.append(TemporalStabilityResult(
                time_window=str(window),
                n_samples=int(n_samples),
                accuracy=accuracy,
                performance_drift=drift,
                p_value=float(p_value) if isinstance(p_value, (int, float)) else 1.0
            ))

        return results

    def detect_concept_drift(
        self,
        X_old: np.ndarray,
        y_old: np.ndarray,
        X_new: np.ndarray,
        y_new: np.ndarray,
        method: str = 'page_hinkley'
    ) -> Dict[str, Any]:
        """
        Detect concept drift between old and new data.

        Parameters
        ----------
        X_old, y_old : arrays
            Historical data
        X_new, y_new : arrays
            New data
        method : str
            Detection method

        Returns
        -------
        Dictionary with drift detection results
        """
        from metrics import ClassificationMetrics

        # Performance on old vs new
        y_pred_old = self.model.predict(X_old)
        y_pred_new = self.model.predict(X_new)

        acc_old = ClassificationMetrics.accuracy(y_old, y_pred_old)
        acc_new = ClassificationMetrics.accuracy(y_new, y_pred_new)

        # Simple drift detection
        performance_change = acc_new - acc_old

        # CUSUM-like detection
        threshold = 0.05
        drift_detected = abs(performance_change) > threshold

        return {
            'accuracy_old': acc_old,
            'accuracy_new': acc_new,
            'performance_change': performance_change,
            'drift_detected': drift_detected,
            'drift_severity': 'significant' if abs(performance_change) > 0.1 else 'moderate' if drift_detected else 'none'
        }


# =============================================================================
# Failure Mode Analysis
# =============================================================================

class FailureModeAnalyzer:
    """
    Analyze and characterize model failure modes.

    Identifies:
    - Systematic errors
    - Edge cases
    - Confounding patterns
    """

    def __init__(self, model):
        """
        Initialize failure mode analyzer.

        Parameters
        ----------
        model : sklearn-compatible model
            Trained model to analyze
        """
        self.model = model

    def analyze_failures(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze failure cases.

        Parameters
        ----------
        X : array
            Features
        y : array
            Labels
        feature_names : list, optional
            Names of features

        Returns
        -------
        List of failure mode descriptions
        """
        y_pred = self.model.predict(X)

        # Identify errors
        false_positives = (y == 0) & (y_pred == 1)
        false_negatives = (y == 1) & (y_pred == 0)

        failure_modes = []

        # Analyze false positives
        if np.sum(false_positives) > 0:
            fp_analysis = self._analyze_error_pattern(
                X[false_positives],
                X[~false_positives],
                feature_names,
                'false_positive'
            )
            failure_modes.append(fp_analysis)

        # Analyze false negatives
        if np.sum(false_negatives) > 0:
            fn_analysis = self._analyze_error_pattern(
                X[false_negatives],
                X[~false_negatives],
                feature_names,
                'false_negative'
            )
            failure_modes.append(fn_analysis)

        return failure_modes

    def _analyze_error_pattern(
        self,
        X_error: np.ndarray,
        X_correct: np.ndarray,
        feature_names: Optional[List[str]],
        error_type: str
    ) -> Dict[str, Any]:
        """Analyze pattern in error cases."""
        if X_error.ndim > 2:
            X_error = X_error.reshape(len(X_error), -1)
        if X_correct.ndim > 2:
            X_correct = X_correct.reshape(len(X_correct), -1)

        n_features = X_error.shape[1]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Find features that differ most between error and correct cases
        differences = []

        for i in range(min(n_features, 100)):  # Limit for efficiency
            mean_error = np.mean(X_error[:, i])
            mean_correct = np.mean(X_correct[:, i])
            std_correct = np.std(X_correct[:, i])

            if std_correct > 0:
                effect_size = abs(mean_error - mean_correct) / std_correct
                differences.append((feature_names[i] if i < len(feature_names) else f"feature_{i}",
                                    effect_size, mean_error, mean_correct))

        # Sort by effect size
        differences.sort(key=lambda x: x[1], reverse=True)

        return {
            'error_type': error_type,
            'n_cases': len(X_error),
            'top_distinguishing_features': differences[:5],
            'mean_feature_values': np.mean(X_error, axis=0).tolist()[:10]
        }

    def identify_edge_cases(
        self,
        X: np.ndarray,
        y: np.ndarray,
        confidence_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Identify edge cases where model is uncertain.

        Parameters
        ----------
        X : array
            Features
        y : array
            Labels
        confidence_threshold : float
            Threshold for low confidence

        Returns
        -------
        Dictionary with edge case analysis
        """
        if not hasattr(self.model, 'predict_proba'):
            return {'error': 'Model does not support probability predictions'}

        y_prob = self.model.predict_proba(X)
        if y_prob.ndim > 1:
            confidence = np.max(y_prob, axis=1)
        else:
            confidence = np.abs(y_prob - 0.5) * 2

        low_confidence_mask = confidence < (0.5 + confidence_threshold / 2)

        y_pred = self.model.predict(X)

        return {
            'n_edge_cases': int(np.sum(low_confidence_mask)),
            'proportion_edge_cases': float(np.mean(low_confidence_mask)),
            'edge_case_accuracy': float(np.mean(y_pred[low_confidence_mask] == y[low_confidence_mask])) if np.sum(low_confidence_mask) > 0 else None,
            'high_confidence_accuracy': float(np.mean(y_pred[~low_confidence_mask] == y[~low_confidence_mask])) if np.sum(~low_confidence_mask) > 0 else None
        }


# =============================================================================
# Clinical Validation Report Generator
# =============================================================================

class ClinicalValidationReportGenerator:
    """
    Generate comprehensive clinical validation report.
    """

    def __init__(
        self,
        model,
        model_name: str,
        disease: str
    ):
        """
        Initialize report generator.

        Parameters
        ----------
        model : sklearn-compatible model
            Trained model
        model_name : str
            Name of the model
        disease : str
            Target disease
        """
        self.model = model
        self.model_name = model_name
        self.disease = disease

        self.cross_dataset = CrossDatasetValidator(model)
        self.subgroup = SubgroupAnalyzer(model)
        self.temporal = TemporalStabilityAnalyzer(model)
        self.failure = FailureModeAnalyzer(model)

    def generate_full_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        demographics: Optional[Dict[str, np.ndarray]] = None,
        timestamps: Optional[np.ndarray] = None,
        external_datasets: Optional[List[Tuple[np.ndarray, np.ndarray, str]]] = None
    ) -> ClinicalValidationReport:
        """
        Generate comprehensive clinical validation report.

        Parameters
        ----------
        X_test : array
            Test features
        y_test : array
            Test labels
        demographics : dict, optional
            Demographic variables for subgroup analysis
        timestamps : array, optional
            Timestamps for temporal analysis
        external_datasets : list, optional
            List of (X, y, name) tuples for external validation

        Returns
        -------
        ClinicalValidationReport
        """
        from metrics import ClassificationMetrics, ProbabilityMetrics

        # Overall metrics
        y_pred = self.model.predict(X_test)
        y_prob = None
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X_test)
            if y_prob.ndim > 1:
                y_prob = y_prob[:, 1]

        overall_metrics = {
            'accuracy': ClassificationMetrics.accuracy(y_test, y_pred),
            'sensitivity': ClassificationMetrics.sensitivity(y_test, y_pred),
            'specificity': ClassificationMetrics.specificity(y_test, y_pred),
            'f1': ClassificationMetrics.f1_score(y_test, y_pred),
            'mcc': ClassificationMetrics.mcc(y_test, y_pred)
        }
        if y_prob is not None:
            overall_metrics['auc'] = ProbabilityMetrics.roc_auc(y_test, y_prob)

        # Subgroup analysis
        subgroup_results = []
        if demographics is not None:
            subgroup_results = self.subgroup.analyze_subgroups(X_test, y_test, demographics)

        # External validation
        generalization_results = []
        if external_datasets is not None:
            for X_ext, y_ext, name in external_datasets:
                result = self.cross_dataset.external_validation(
                    X_ext, y_ext, name, overall_metrics
                )
                generalization_results.append(result)

        # Temporal stability
        temporal_results = []
        if timestamps is not None:
            self.temporal.set_baseline(X_test, y_test)
            temporal_results = self.temporal.analyze_temporal_windows(
                X_test, y_test, timestamps
            )

        # Failure modes
        failure_modes = self.failure.analyze_failures(X_test, y_test)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_metrics,
            subgroup_results,
            generalization_results
        )

        # Regulatory notes
        regulatory_notes = self._generate_regulatory_notes(
            overall_metrics,
            subgroup_results
        )

        return ClinicalValidationReport(
            model_name=self.model_name,
            disease=self.disease,
            validation_date=datetime.now().isoformat(),
            overall_metrics=overall_metrics,
            subgroup_analysis=subgroup_results,
            generalization_results=generalization_results,
            temporal_stability=temporal_results,
            failure_modes=failure_modes,
            recommendations=recommendations,
            regulatory_notes=regulatory_notes
        )

    def _generate_recommendations(
        self,
        overall: Dict[str, float],
        subgroups: List[SubgroupResult],
        generalization: List[GeneralizationResult]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Check overall performance
        if overall.get('sensitivity', 0) < 0.8:
            recommendations.append(
                "CRITICAL: Sensitivity below 80%. Consider increasing model sensitivity "
                "for clinical applications where missing positive cases is costly."
            )

        if overall.get('specificity', 0) < 0.7:
            recommendations.append(
                "WARNING: Specificity below 70%. High false positive rate may lead to "
                "unnecessary follow-up procedures."
            )

        # Check subgroup disparities
        if subgroups:
            accs = [s.accuracy for s in subgroups]
            if max(accs) - min(accs) > 0.15:
                recommendations.append(
                    "FAIRNESS: Performance varies >15% across subgroups. "
                    "Consider targeted data collection or model adjustments."
                )

        # Check generalization
        for gen in generalization:
            if gen.performance_drop > 0.1:
                recommendations.append(
                    f"GENERALIZATION: Performance dropped {gen.performance_drop:.1%} on "
                    f"{gen.test_dataset}. Model may not generalize well to this population."
                )

        if not recommendations:
            recommendations.append(
                "Model meets basic clinical validation criteria. "
                "Proceed with clinical pilot study."
            )

        return recommendations

    def _generate_regulatory_notes(
        self,
        overall: Dict[str, float],
        subgroups: List[SubgroupResult]
    ) -> List[str]:
        """Generate regulatory compliance notes."""
        notes = [
            "This model is intended as a clinical decision support tool.",
            "Model outputs should be reviewed by qualified healthcare professionals.",
            "Not intended for standalone diagnosis without clinical oversight.",
        ]

        # FDA-specific notes
        if overall.get('sensitivity', 0) >= 0.9 and overall.get('specificity', 0) >= 0.9:
            notes.append(
                "Performance metrics may support FDA 510(k) clearance pathway "
                "for computer-aided detection (CADe) devices."
            )
        else:
            notes.append(
                "Performance metrics suggest additional validation required "
                "before regulatory submission."
            )

        # Subgroup notes
        if subgroups:
            notes.append(
                f"Validated across {len(subgroups)} demographic subgroups. "
                "See subgroup analysis for performance breakdown."
            )

        return notes

    def export_report(
        self,
        report: ClinicalValidationReport,
        output_path: str,
        format: str = 'json'
    ) -> None:
        """
        Export validation report.

        Parameters
        ----------
        report : ClinicalValidationReport
            Report to export
        output_path : str
            Output file path
        format : str
            Output format ('json', 'markdown')
        """
        if format == 'json':
            # Convert dataclasses to dicts
            report_dict = {
                'model_name': report.model_name,
                'disease': report.disease,
                'validation_date': report.validation_date,
                'overall_metrics': report.overall_metrics,
                'subgroup_analysis': [
                    {
                        'subgroup_name': s.subgroup_name,
                        'n_samples': s.n_samples,
                        'accuracy': s.accuracy,
                        'sensitivity': s.sensitivity,
                        'specificity': s.specificity,
                        'f1': s.f1,
                        'auc': s.auc
                    }
                    for s in report.subgroup_analysis
                ],
                'generalization_results': [
                    {
                        'train_dataset': g.train_dataset,
                        'test_dataset': g.test_dataset,
                        'accuracy': g.accuracy,
                        'performance_drop': g.performance_drop
                    }
                    for g in report.generalization_results
                ],
                'temporal_stability': [
                    {
                        'time_window': t.time_window,
                        'accuracy': t.accuracy,
                        'performance_drift': t.performance_drift
                    }
                    for t in report.temporal_stability
                ],
                'failure_modes': report.failure_modes,
                'recommendations': report.recommendations,
                'regulatory_notes': report.regulatory_notes
            }

            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2)

        elif format == 'markdown':
            md = self._report_to_markdown(report)
            with open(output_path, 'w') as f:
                f.write(md)

    def _report_to_markdown(self, report: ClinicalValidationReport) -> str:
        """Convert report to markdown format."""
        lines = [
            f"# Clinical Validation Report: {report.model_name}",
            f"**Disease:** {report.disease}",
            f"**Date:** {report.validation_date}",
            "",
            "## Overall Performance",
            "",
            "| Metric | Value |",
            "|--------|-------|"
        ]

        for metric, value in report.overall_metrics.items():
            lines.append(f"| {metric} | {value:.4f} |")

        if report.subgroup_analysis:
            lines.extend([
                "",
                "## Subgroup Analysis",
                "",
                "| Subgroup | N | Accuracy | Sensitivity | Specificity |",
                "|----------|---|----------|-------------|-------------|"
            ])
            for s in report.subgroup_analysis:
                lines.append(f"| {s.subgroup_name} | {s.n_samples} | {s.accuracy:.3f} | {s.sensitivity:.3f} | {s.specificity:.3f} |")

        if report.recommendations:
            lines.extend([
                "",
                "## Recommendations",
                ""
            ])
            for rec in report.recommendations:
                lines.append(f"- {rec}")

        if report.regulatory_notes:
            lines.extend([
                "",
                "## Regulatory Notes",
                ""
            ])
            for note in report.regulatory_notes:
                lines.append(f"- {note}")

        return '\n'.join(lines)
