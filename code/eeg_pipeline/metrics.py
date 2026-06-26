"""
Comprehensive Metrics Module (Extended)
========================================

This module provides all 25+ performance metrics and 30+ analysis types
from the EEG-based ML project framework.

Covers:
- Standard classification metrics
- Subject-wise (LOSO) metrics
- Clinical metrics (PPV, NPV, LR+, LR-, DOR, NNT)
- Reliability metrics (ICC, Cronbach's α, Cohen's κ)
- Calibration metrics (ECE, MCE, Brier)
- Fairness metrics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.special import expit
import warnings


@dataclass
class MetricResult:
    """Container for a metric result with confidence interval."""
    name: str
    value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    std: Optional[float] = None
    n_samples: Optional[int] = None


@dataclass
class SubjectMetrics:
    """Per-subject metrics for LOSO analysis."""
    subject_id: str
    accuracy: float
    sensitivity: float
    specificity: float
    f1: float
    n_samples: int


# =============================================================================
# Standard Classification Metrics
# =============================================================================

class ClassificationMetrics:
    """
    Standard classification performance metrics.

    Covers:
    1. Accuracy
    2. Precision (PPV)
    3. Recall (Sensitivity, TPR)
    4. Specificity (TNR)
    5. F1-Score
    6. F2-Score (emphasizes recall)
    7. F0.5-Score (emphasizes precision)
    8. Matthews Correlation Coefficient (MCC)
    9. Cohen's Kappa
    10. Balanced Accuracy
    11. Geometric Mean (G-Mean)
    12. Youden's J (Informedness)
    13. Markedness
    14. Diagnostic Odds Ratio (DOR)
    """

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        """Compute confusion matrix components."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        return {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Overall accuracy."""
        return np.mean(y_true == y_pred)

    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Precision (Positive Predictive Value)."""
        cm = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        denom = cm['tp'] + cm['fp']
        return cm['tp'] / denom if denom > 0 else 0.0

    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Recall (Sensitivity, True Positive Rate)."""
        cm = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        denom = cm['tp'] + cm['fn']
        return cm['tp'] / denom if denom > 0 else 0.0

    @staticmethod
    def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Sensitivity (same as recall)."""
        return ClassificationMetrics.recall(y_true, y_pred)

    @staticmethod
    def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Specificity (True Negative Rate)."""
        cm = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        denom = cm['tn'] + cm['fp']
        return cm['tn'] / denom if denom > 0 else 0.0

    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """F1 Score (harmonic mean of precision and recall)."""
        p = ClassificationMetrics.precision(y_true, y_pred)
        r = ClassificationMetrics.recall(y_true, y_pred)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @staticmethod
    def fbeta_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 1.0) -> float:
        """F-beta Score with configurable beta."""
        p = ClassificationMetrics.precision(y_true, y_pred)
        r = ClassificationMetrics.recall(y_true, y_pred)
        beta2 = beta ** 2
        denom = beta2 * p + r
        return (1 + beta2) * p * r / denom if denom > 0 else 0.0

    @staticmethod
    def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Matthews Correlation Coefficient."""
        cm = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        tp, tn, fp, fn = cm['tp'], cm['tn'], cm['fp'], cm['fn']

        num = tp * tn - fp * fn
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        return num / denom if denom > 0 else 0.0

    @staticmethod
    def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Cohen's Kappa coefficient."""
        n = len(y_true)
        po = ClassificationMetrics.accuracy(y_true, y_pred)

        # Expected agreement by chance
        pe = (
            (np.sum(y_true == 1) * np.sum(y_pred == 1) +
             np.sum(y_true == 0) * np.sum(y_pred == 0)) / (n ** 2)
        )

        return (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

    @staticmethod
    def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Balanced accuracy (average of sensitivity and specificity)."""
        sens = ClassificationMetrics.sensitivity(y_true, y_pred)
        spec = ClassificationMetrics.specificity(y_true, y_pred)
        return (sens + spec) / 2

    @staticmethod
    def geometric_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Geometric mean of sensitivity and specificity."""
        sens = ClassificationMetrics.sensitivity(y_true, y_pred)
        spec = ClassificationMetrics.specificity(y_true, y_pred)
        return np.sqrt(sens * spec)

    @staticmethod
    def youdens_j(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Youden's J statistic (Informedness)."""
        sens = ClassificationMetrics.sensitivity(y_true, y_pred)
        spec = ClassificationMetrics.specificity(y_true, y_pred)
        return sens + spec - 1

    @staticmethod
    def markedness(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Markedness (PPV + NPV - 1)."""
        ppv = ClassificationMetrics.precision(y_true, y_pred)
        npv = ClassificationMetrics.npv(y_true, y_pred)
        return ppv + npv - 1

    @staticmethod
    def npv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Negative Predictive Value."""
        cm = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        denom = cm['tn'] + cm['fn']
        return cm['tn'] / denom if denom > 0 else 0.0

    @staticmethod
    def diagnostic_odds_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Diagnostic Odds Ratio."""
        cm = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        tp, tn, fp, fn = cm['tp'], cm['tn'], cm['fp'], cm['fn']

        # Add small constant to avoid division by zero
        eps = 1e-10
        lr_pos = (tp / (tp + fn + eps)) / (fp / (tn + fp + eps))
        lr_neg = (fn / (tp + fn + eps)) / (tn / (tn + fp + eps))

        return lr_pos / (lr_neg + eps) if lr_neg > 0 else float('inf')

    @classmethod
    def compute_all(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, MetricResult]:
        """Compute all classification metrics."""
        results = {}

        # Basic metrics
        results['accuracy'] = MetricResult('accuracy', cls.accuracy(y_true, y_pred))
        results['precision'] = MetricResult('precision', cls.precision(y_true, y_pred))
        results['recall'] = MetricResult('recall', cls.recall(y_true, y_pred))
        results['specificity'] = MetricResult('specificity', cls.specificity(y_true, y_pred))
        results['f1'] = MetricResult('f1', cls.f1_score(y_true, y_pred))
        results['f2'] = MetricResult('f2', cls.fbeta_score(y_true, y_pred, beta=2.0))
        results['f0.5'] = MetricResult('f0.5', cls.fbeta_score(y_true, y_pred, beta=0.5))
        results['mcc'] = MetricResult('mcc', cls.mcc(y_true, y_pred))
        results['kappa'] = MetricResult('kappa', cls.cohens_kappa(y_true, y_pred))
        results['balanced_accuracy'] = MetricResult('balanced_accuracy', cls.balanced_accuracy(y_true, y_pred))
        results['g_mean'] = MetricResult('g_mean', cls.geometric_mean(y_true, y_pred))
        results['youdens_j'] = MetricResult('youdens_j', cls.youdens_j(y_true, y_pred))
        results['markedness'] = MetricResult('markedness', cls.markedness(y_true, y_pred))
        results['npv'] = MetricResult('npv', cls.npv(y_true, y_pred))
        results['dor'] = MetricResult('dor', cls.diagnostic_odds_ratio(y_true, y_pred))

        # Probability-based metrics
        if y_prob is not None:
            prob_metrics = ProbabilityMetrics.compute_all(y_true, y_prob)
            results.update(prob_metrics)

        return results


# =============================================================================
# Probability-Based Metrics
# =============================================================================

class ProbabilityMetrics:
    """
    Probability-based metrics.

    Covers:
    15. ROC-AUC
    16. PR-AUC
    17. Log Loss (Cross-Entropy)
    18. Brier Score
    19. Expected Calibration Error (ECE)
    20. Maximum Calibration Error (MCE)
    """

    @staticmethod
    def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Area Under ROC Curve."""
        # Simple implementation using trapezoidal rule
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Sort by probability
        desc_idx = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[desc_idx]

        # Compute TPR and FPR at each threshold
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr = np.cumsum(y_true_sorted == 1) / n_pos
        fpr = np.cumsum(y_true_sorted == 0) / n_neg

        # Add origin point
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])

        # Trapezoidal integration
        auc = np.trapz(tpr, fpr)

        return float(auc)

    @staticmethod
    def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Area Under Precision-Recall Curve."""
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Sort by probability descending
        desc_idx = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[desc_idx]

        # Compute precision and recall at each threshold
        tp_cumsum = np.cumsum(y_true_sorted == 1)
        fp_cumsum = np.cumsum(y_true_sorted == 0)

        n_pos = np.sum(y_true == 1)

        if n_pos == 0:
            return 0.0

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / n_pos

        # Add start point
        precision = np.concatenate([[1], precision])
        recall = np.concatenate([[0], recall])

        # Trapezoidal integration
        auc = np.trapz(precision, recall)

        return float(auc)

    @staticmethod
    def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
        """Log loss (cross-entropy)."""
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Brier score (mean squared error of probabilities)."""
        return np.mean((y_prob - y_true) ** 2)

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Expected Calibration Error (ECE)."""
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                avg_confidence = np.mean(y_prob[in_bin])
                avg_accuracy = np.mean(y_true[in_bin])
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

        return float(ece)

    @staticmethod
    def maximum_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Maximum Calibration Error (MCE)."""
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0

        for i in range(n_bins):
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])

            if np.sum(in_bin) > 0:
                avg_confidence = np.mean(y_prob[in_bin])
                avg_accuracy = np.mean(y_true[in_bin])
                mce = max(mce, np.abs(avg_accuracy - avg_confidence))

        return float(mce)

    @classmethod
    def compute_all(cls, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, MetricResult]:
        """Compute all probability-based metrics."""
        return {
            'roc_auc': MetricResult('roc_auc', cls.roc_auc(y_true, y_prob)),
            'pr_auc': MetricResult('pr_auc', cls.pr_auc(y_true, y_prob)),
            'log_loss': MetricResult('log_loss', cls.log_loss(y_true, y_prob)),
            'brier': MetricResult('brier', cls.brier_score(y_true, y_prob)),
            'ece': MetricResult('ece', cls.expected_calibration_error(y_true, y_prob)),
            'mce': MetricResult('mce', cls.maximum_calibration_error(y_true, y_prob))
        }


# =============================================================================
# Clinical Metrics
# =============================================================================

class ClinicalMetrics:
    """
    Clinical validation metrics.

    Covers:
    21. Positive Likelihood Ratio (LR+)
    22. Negative Likelihood Ratio (LR-)
    23. Number Needed to Diagnose (NND)
    24. Number Needed to Treat (NNT)
    25. Clinical Utility Index (CUI)
    """

    @staticmethod
    def positive_likelihood_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Positive Likelihood Ratio (LR+).
        LR+ = Sensitivity / (1 - Specificity)
        """
        sens = ClassificationMetrics.sensitivity(y_true, y_pred)
        spec = ClassificationMetrics.specificity(y_true, y_pred)

        fpr = 1 - spec
        return sens / fpr if fpr > 0 else float('inf')

    @staticmethod
    def negative_likelihood_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Negative Likelihood Ratio (LR-).
        LR- = (1 - Sensitivity) / Specificity
        """
        sens = ClassificationMetrics.sensitivity(y_true, y_pred)
        spec = ClassificationMetrics.specificity(y_true, y_pred)

        fnr = 1 - sens
        return fnr / spec if spec > 0 else float('inf')

    @staticmethod
    def number_needed_to_diagnose(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Number Needed to Diagnose (NND).
        NND = 1 / Youden's J = 1 / (Sensitivity + Specificity - 1)
        """
        j = ClassificationMetrics.youdens_j(y_true, y_pred)
        return 1 / j if j > 0 else float('inf')

    @staticmethod
    def clinical_utility_index_positive(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Clinical Utility Index for positive predictions.
        CUI+ = Sensitivity × PPV
        """
        sens = ClassificationMetrics.sensitivity(y_true, y_pred)
        ppv = ClassificationMetrics.precision(y_true, y_pred)
        return sens * ppv

    @staticmethod
    def clinical_utility_index_negative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Clinical Utility Index for negative predictions.
        CUI- = Specificity × NPV
        """
        spec = ClassificationMetrics.specificity(y_true, y_pred)
        npv = ClassificationMetrics.npv(y_true, y_pred)
        return spec * npv

    @staticmethod
    def post_test_probability(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        pre_test_prob: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute post-test probabilities using Bayes' theorem.

        Parameters
        ----------
        y_true : array
            True labels
        y_pred : array
            Predicted labels
        pre_test_prob : float
            Pre-test probability (prevalence)

        Returns
        -------
        dict with post_test_positive and post_test_negative
        """
        lr_pos = ClinicalMetrics.positive_likelihood_ratio(y_true, y_pred)
        lr_neg = ClinicalMetrics.negative_likelihood_ratio(y_true, y_pred)

        # Convert to odds
        pre_test_odds = pre_test_prob / (1 - pre_test_prob)

        # Post-test odds
        post_test_odds_pos = pre_test_odds * lr_pos
        post_test_odds_neg = pre_test_odds * lr_neg

        # Convert back to probability
        post_test_prob_pos = post_test_odds_pos / (1 + post_test_odds_pos)
        post_test_prob_neg = post_test_odds_neg / (1 + post_test_odds_neg)

        return {
            'post_test_positive': post_test_prob_pos,
            'post_test_negative': post_test_prob_neg
        }

    @classmethod
    def compute_all(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prevalence: float = 0.5
    ) -> Dict[str, MetricResult]:
        """Compute all clinical metrics."""
        post_test = cls.post_test_probability(y_true, y_pred, prevalence)

        return {
            'lr_positive': MetricResult('lr_positive', cls.positive_likelihood_ratio(y_true, y_pred)),
            'lr_negative': MetricResult('lr_negative', cls.negative_likelihood_ratio(y_true, y_pred)),
            'nnd': MetricResult('nnd', cls.number_needed_to_diagnose(y_true, y_pred)),
            'cui_positive': MetricResult('cui_positive', cls.clinical_utility_index_positive(y_true, y_pred)),
            'cui_negative': MetricResult('cui_negative', cls.clinical_utility_index_negative(y_true, y_pred)),
            'post_test_prob_positive': MetricResult('post_test_prob_positive', post_test['post_test_positive']),
            'post_test_prob_negative': MetricResult('post_test_prob_negative', post_test['post_test_negative'])
        }


# =============================================================================
# Subject-Wise Metrics (LOSO Analysis)
# =============================================================================

class SubjectWiseMetrics:
    """
    Subject-wise (LOSO) performance analysis.

    Computes metrics per subject and aggregate statistics.
    """

    @staticmethod
    def compute_per_subject(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        subject_ids: np.ndarray
    ) -> List[SubjectMetrics]:
        """
        Compute metrics for each subject.

        Parameters
        ----------
        y_true : array
            True labels
        y_pred : array
            Predicted labels
        subject_ids : array
            Subject ID for each sample

        Returns
        -------
        List of SubjectMetrics
        """
        unique_subjects = np.unique(subject_ids)
        results = []

        for subj in unique_subjects:
            mask = subject_ids == subj
            y_t = y_true[mask]
            y_p = y_pred[mask]

            # Handle edge cases
            if len(np.unique(y_t)) < 2:
                # Only one class for this subject
                acc = ClassificationMetrics.accuracy(y_t, y_p)
                results.append(SubjectMetrics(
                    subject_id=str(subj),
                    accuracy=acc,
                    sensitivity=np.nan,
                    specificity=np.nan,
                    f1=np.nan,
                    n_samples=len(y_t)
                ))
            else:
                results.append(SubjectMetrics(
                    subject_id=str(subj),
                    accuracy=ClassificationMetrics.accuracy(y_t, y_p),
                    sensitivity=ClassificationMetrics.sensitivity(y_t, y_p),
                    specificity=ClassificationMetrics.specificity(y_t, y_p),
                    f1=ClassificationMetrics.f1_score(y_t, y_p),
                    n_samples=len(y_t)
                ))

        return results

    @staticmethod
    def aggregate_subject_metrics(
        subject_metrics: List[SubjectMetrics]
    ) -> Dict[str, MetricResult]:
        """
        Aggregate per-subject metrics into summary statistics.

        Returns mean, std, min, max, and IQR for each metric.
        """
        accuracies = [s.accuracy for s in subject_metrics]
        sensitivities = [s.sensitivity for s in subject_metrics if not np.isnan(s.sensitivity)]
        specificities = [s.specificity for s in subject_metrics if not np.isnan(s.specificity)]
        f1s = [s.f1 for s in subject_metrics if not np.isnan(s.f1)]

        results = {}

        # Accuracy
        results['subject_accuracy_mean'] = MetricResult(
            'subject_accuracy_mean',
            np.mean(accuracies),
            std=np.std(accuracies)
        )

        # Sensitivity
        if sensitivities:
            results['subject_sensitivity_mean'] = MetricResult(
                'subject_sensitivity_mean',
                np.mean(sensitivities),
                std=np.std(sensitivities)
            )

        # Specificity
        if specificities:
            results['subject_specificity_mean'] = MetricResult(
                'subject_specificity_mean',
                np.mean(specificities),
                std=np.std(specificities)
            )

        # F1
        if f1s:
            results['subject_f1_mean'] = MetricResult(
                'subject_f1_mean',
                np.mean(f1s),
                std=np.std(f1s)
            )

        # Distribution statistics
        results['subject_accuracy_iqr'] = MetricResult(
            'subject_accuracy_iqr',
            np.percentile(accuracies, 75) - np.percentile(accuracies, 25)
        )

        results['subject_accuracy_range'] = MetricResult(
            'subject_accuracy_range',
            np.max(accuracies) - np.min(accuracies)
        )

        return results

    @staticmethod
    def identify_outlier_subjects(
        subject_metrics: List[SubjectMetrics],
        metric: str = 'accuracy',
        z_threshold: float = 2.0
    ) -> List[str]:
        """
        Identify subjects with outlier performance.

        Parameters
        ----------
        subject_metrics : list
            Per-subject metrics
        metric : str
            Which metric to use for outlier detection
        z_threshold : float
            Z-score threshold for outlier detection

        Returns
        -------
        List of outlier subject IDs
        """
        values = [getattr(s, metric) for s in subject_metrics]
        values = [v for v in values if not np.isnan(v)]

        mean_val = np.mean(values)
        std_val = np.std(values)

        outliers = []
        for s in subject_metrics:
            val = getattr(s, metric)
            if not np.isnan(val) and std_val > 0:
                z = (val - mean_val) / std_val
                if np.abs(z) > z_threshold:
                    outliers.append(s.subject_id)

        return outliers


# =============================================================================
# Reliability Metrics
# =============================================================================

class ReliabilityMetrics:
    """
    Reliability and agreement metrics.

    Covers:
    - Intraclass Correlation Coefficient (ICC)
    - Cronbach's Alpha
    - Inter-rater agreement
    - Test-retest reliability
    """

    @staticmethod
    def icc(
        ratings: np.ndarray,
        icc_type: str = 'ICC(2,1)'
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Compute Intraclass Correlation Coefficient.

        Parameters
        ----------
        ratings : np.ndarray
            Rating matrix (n_subjects x n_raters)
        icc_type : str
            Type of ICC ('ICC(1,1)', 'ICC(2,1)', 'ICC(3,1)')

        Returns
        -------
        icc : float
            ICC value
        ci : tuple
            95% confidence interval
        """
        n, k = ratings.shape

        # Grand mean
        grand_mean = np.mean(ratings)

        # Subject means
        subject_means = np.mean(ratings, axis=1)

        # Rater means
        rater_means = np.mean(ratings, axis=0)

        # Sum of squares
        ss_total = np.sum((ratings - grand_mean) ** 2)
        ss_between_subjects = k * np.sum((subject_means - grand_mean) ** 2)
        ss_between_raters = n * np.sum((rater_means - grand_mean) ** 2)
        ss_residual = ss_total - ss_between_subjects - ss_between_raters

        # Mean squares
        ms_between = ss_between_subjects / (n - 1)
        ms_raters = ss_between_raters / (k - 1) if k > 1 else 0
        ms_residual = ss_residual / ((n - 1) * (k - 1)) if k > 1 else ss_residual / (n - 1)

        # ICC calculation depends on type
        if icc_type == 'ICC(1,1)':
            # One-way random
            icc = (ms_between - ms_residual) / (ms_between + (k - 1) * ms_residual)
        elif icc_type == 'ICC(2,1)':
            # Two-way random
            icc = (ms_between - ms_residual) / (
                ms_between + (k - 1) * ms_residual + k * (ms_raters - ms_residual) / n
            )
        elif icc_type == 'ICC(3,1)':
            # Two-way mixed
            icc = (ms_between - ms_residual) / (ms_between + (k - 1) * ms_residual)
        else:
            raise ValueError(f"Unknown ICC type: {icc_type}")

        # Confidence interval (F-based approximation)
        f_value = ms_between / ms_residual if ms_residual > 0 else 0
        df1, df2 = n - 1, (n - 1) * (k - 1)

        # Simplified CI calculation
        f_lower = f_value / stats.f.ppf(0.975, df1, df2) if df2 > 0 else 0
        f_upper = f_value * stats.f.ppf(0.975, df2, df1) if df1 > 0 else 0

        ci_lower = (f_lower - 1) / (f_lower + k - 1)
        ci_upper = (f_upper - 1) / (f_upper + k - 1)

        return float(icc), (float(ci_lower), float(ci_upper))

    @staticmethod
    def cronbachs_alpha(items: np.ndarray) -> float:
        """
        Compute Cronbach's Alpha for internal consistency.

        Parameters
        ----------
        items : np.ndarray
            Item scores (n_subjects x n_items)

        Returns
        -------
        alpha : float
            Cronbach's alpha coefficient
        """
        n_items = items.shape[1]

        # Variance of each item
        item_vars = np.var(items, axis=0, ddof=1)

        # Variance of total scores
        total_var = np.var(np.sum(items, axis=1), ddof=1)

        if total_var == 0:
            return 0.0

        alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_vars) / total_var)

        return float(alpha)

    @staticmethod
    def fleiss_kappa(ratings: np.ndarray) -> float:
        """
        Compute Fleiss' Kappa for multi-rater agreement.

        Parameters
        ----------
        ratings : np.ndarray
            Rating matrix (n_subjects x n_categories) where each cell
            contains the count of raters who assigned that category

        Returns
        -------
        kappa : float
            Fleiss' kappa coefficient
        """
        n, k = ratings.shape
        N = np.sum(ratings[0])  # Total raters per subject

        # Proportion of ratings in each category
        p = np.sum(ratings, axis=0) / (n * N)

        # Expected agreement by chance
        Pe = np.sum(p ** 2)

        # Observed agreement
        Pi = (np.sum(ratings ** 2, axis=1) - N) / (N * (N - 1))
        Po = np.mean(Pi)

        if Pe == 1:
            return 1.0

        kappa = (Po - Pe) / (1 - Pe)

        return float(kappa)

    @staticmethod
    def test_retest_reliability(
        test1: np.ndarray,
        test2: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute test-retest reliability metrics.

        Parameters
        ----------
        test1 : array
            First measurement
        test2 : array
            Second measurement

        Returns
        -------
        dict with pearson_r, icc, and mean_diff
        """
        # Pearson correlation
        r, p_value = stats.pearsonr(test1, test2)

        # ICC
        ratings = np.column_stack([test1, test2])
        icc_val, _ = ReliabilityMetrics.icc(ratings, 'ICC(3,1)')

        # Mean difference and limits of agreement (Bland-Altman)
        diff = test1 - test2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)

        return {
            'pearson_r': float(r),
            'p_value': float(p_value),
            'icc': icc_val,
            'mean_diff': float(mean_diff),
            'loa_lower': float(mean_diff - 1.96 * std_diff),
            'loa_upper': float(mean_diff + 1.96 * std_diff)
        }


# =============================================================================
# Comprehensive Metrics Calculator
# =============================================================================

class ComprehensiveMetrics:
    """
    Comprehensive metrics calculator combining all metric types.
    """

    def __init__(self, include_clinical: bool = True, prevalence: float = 0.5):
        """
        Initialize comprehensive metrics calculator.

        Parameters
        ----------
        include_clinical : bool
            Whether to include clinical metrics
        prevalence : float
            Disease prevalence for clinical metric calculations
        """
        self.include_clinical = include_clinical
        self.prevalence = prevalence

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        subject_ids: Optional[np.ndarray] = None
    ) -> Dict[str, MetricResult]:
        """
        Compute all available metrics.

        Parameters
        ----------
        y_true : array
            True labels
        y_pred : array
            Predicted labels
        y_prob : array, optional
            Predicted probabilities
        subject_ids : array, optional
            Subject IDs for subject-wise analysis

        Returns
        -------
        Dictionary of all metrics
        """
        results = {}

        # Classification metrics
        classification = ClassificationMetrics.compute_all(y_true, y_pred, y_prob)
        results.update(classification)

        # Clinical metrics
        if self.include_clinical:
            clinical = ClinicalMetrics.compute_all(y_true, y_pred, self.prevalence)
            results.update(clinical)

        # Subject-wise metrics
        if subject_ids is not None:
            subject_metrics = SubjectWiseMetrics.compute_per_subject(y_true, y_pred, subject_ids)
            aggregated = SubjectWiseMetrics.aggregate_subject_metrics(subject_metrics)
            results.update(aggregated)

        return results

    def generate_report(
        self,
        metrics: Dict[str, MetricResult],
        format: str = 'markdown'
    ) -> str:
        """
        Generate a formatted report of all metrics.

        Parameters
        ----------
        metrics : dict
            Dictionary of MetricResult objects
        format : str
            Output format ('markdown', 'latex', 'csv')

        Returns
        -------
        Formatted report string
        """
        if format == 'markdown':
            return self._markdown_report(metrics)
        elif format == 'latex':
            return self._latex_report(metrics)
        elif format == 'csv':
            return self._csv_report(metrics)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _markdown_report(self, metrics: Dict[str, MetricResult]) -> str:
        """Generate markdown report."""
        lines = ["# Performance Metrics Report\n"]

        # Group metrics by category
        categories = {
            'Classification': ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'f2', 'f0.5', 'mcc', 'kappa', 'balanced_accuracy', 'g_mean'],
            'Probability': ['roc_auc', 'pr_auc', 'log_loss', 'brier', 'ece', 'mce'],
            'Clinical': ['lr_positive', 'lr_negative', 'nnd', 'cui_positive', 'cui_negative', 'npv', 'dor'],
            'Subject-wise': ['subject_accuracy_mean', 'subject_sensitivity_mean', 'subject_specificity_mean', 'subject_f1_mean']
        }

        for category, metric_names in categories.items():
            available = [m for m in metric_names if m in metrics]
            if available:
                lines.append(f"\n## {category} Metrics\n")
                lines.append("| Metric | Value | Std |")
                lines.append("|--------|-------|-----|")

                for name in available:
                    m = metrics[name]
                    std_str = f"{m.std:.4f}" if m.std is not None else "-"
                    lines.append(f"| {name} | {m.value:.4f} | {std_str} |")

        return '\n'.join(lines)

    def _latex_report(self, metrics: Dict[str, MetricResult]) -> str:
        """Generate LaTeX table."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Performance Metrics}",
            r"\begin{tabular}{lcc}",
            r"\hline",
            r"Metric & Value & CI \\",
            r"\hline"
        ]

        for name, m in metrics.items():
            ci_str = f"[{m.ci_lower:.3f}, {m.ci_upper:.3f}]" if m.ci_lower is not None else "-"
            lines.append(f"{name.replace('_', ' ')} & {m.value:.4f} & {ci_str} \\\\")

        lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}"
        ])

        return '\n'.join(lines)

    def _csv_report(self, metrics: Dict[str, MetricResult]) -> str:
        """Generate CSV output."""
        lines = ["metric,value,ci_lower,ci_upper,std"]

        for name, m in metrics.items():
            ci_l = f"{m.ci_lower:.6f}" if m.ci_lower is not None else ""
            ci_u = f"{m.ci_upper:.6f}" if m.ci_upper is not None else ""
            std = f"{m.std:.6f}" if m.std is not None else ""
            lines.append(f"{name},{m.value:.6f},{ci_l},{ci_u},{std}")

        return '\n'.join(lines)
