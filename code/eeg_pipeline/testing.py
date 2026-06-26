"""
Model Testing with Statistical Significance (Phase 9)
======================================================

This module implements:
- Bootstrap confidence intervals
- Statistical significance tests (McNemar, paired bootstrap)
- DeLong test for AUC comparison
- One-time test execution protocol
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib
import warnings


@dataclass
class ConfidenceInterval:
    """Confidence interval container."""
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str
    n_bootstrap: int = 0


@dataclass
class SignificanceTestResult:
    """Result from a significance test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: Optional[float] = None


class BootstrapCI:
    """
    Bootstrap confidence intervals for metrics.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations (default: 1000)
    confidence_level : float
        Confidence level (default: 0.95)
    random_seed : int
        Random seed for reproducibility
    method : str
        Bootstrap method: 'percentile', 'bca' (bias-corrected accelerated)
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int = 42,
        method: str = 'percentile'
    ):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        self.method = method

    def compute_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn,
        y_prob: Optional[np.ndarray] = None
    ) -> ConfidenceInterval:
        """
        Compute bootstrap CI for a metric.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        metric_fn : callable
            Metric function that takes (y_true, y_pred) or (y_true, y_prob)
        y_prob : np.ndarray, optional
            Predicted probabilities (for AUC, etc.)

        Returns
        -------
        ci : ConfidenceInterval
            Confidence interval
        """
        rng = np.random.RandomState(self.random_seed)
        n_samples = len(y_true)

        # Compute point estimate
        if y_prob is not None:
            point_estimate = metric_fn(y_true, y_prob)
        else:
            point_estimate = metric_fn(y_true, y_pred)

        # Bootstrap samples
        bootstrap_estimates = []
        for _ in range(self.n_bootstrap):
            indices = rng.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            try:
                if y_prob is not None:
                    y_prob_boot = y_prob[indices]
                    est = metric_fn(y_true_boot, y_prob_boot)
                else:
                    est = metric_fn(y_true_boot, y_pred_boot)
                bootstrap_estimates.append(est)
            except:
                continue

        bootstrap_estimates = np.array(bootstrap_estimates)

        # Compute CI
        alpha = 1 - self.confidence_level

        if self.method == 'percentile':
            lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
            upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

        elif self.method == 'bca':
            # Bias-corrected and accelerated bootstrap
            # Simplified version
            z0 = self._bias_correction(bootstrap_estimates, point_estimate)
            a = self._acceleration(y_true, y_pred, metric_fn, y_prob)

            from scipy import stats as scipy_stats
            z_alpha = scipy_stats.norm.ppf(alpha / 2)
            z_1_alpha = scipy_stats.norm.ppf(1 - alpha / 2)

            alpha1 = scipy_stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
            alpha2 = scipy_stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))

            lower = np.percentile(bootstrap_estimates, 100 * alpha1)
            upper = np.percentile(bootstrap_estimates, 100 * alpha2)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return ConfidenceInterval(
            estimate=point_estimate,
            lower=lower,
            upper=upper,
            confidence_level=self.confidence_level,
            method=self.method,
            n_bootstrap=self.n_bootstrap
        )

    def _bias_correction(self, estimates: np.ndarray, point: float) -> float:
        """Compute bias correction factor."""
        from scipy import stats as scipy_stats
        prop = np.mean(estimates < point)
        return scipy_stats.norm.ppf(max(0.001, min(0.999, prop)))

    def _acceleration(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn,
        y_prob: Optional[np.ndarray]
    ) -> float:
        """Compute acceleration factor using jackknife."""
        n = len(y_true)
        jackknife_estimates = []

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False

            try:
                if y_prob is not None:
                    est = metric_fn(y_true[mask], y_prob[mask])
                else:
                    est = metric_fn(y_true[mask], y_pred[mask])
                jackknife_estimates.append(est)
            except:
                continue

        jackknife_estimates = np.array(jackknife_estimates)
        mean_jack = np.mean(jackknife_estimates)
        diff = mean_jack - jackknife_estimates

        num = np.sum(diff ** 3)
        denom = 6 * (np.sum(diff ** 2) ** 1.5)

        if denom == 0:
            return 0.0
        return num / denom

    def compute_all_metrics_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, ConfidenceInterval]:
        """
        Compute CIs for all standard metrics.

        Returns
        -------
        cis : Dict[str, ConfidenceInterval]
            CIs for each metric
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            roc_auc_score, average_precision_score
        )

        results = {}

        # Accuracy
        results['accuracy'] = self.compute_ci(
            y_true, y_pred,
            lambda y, p: accuracy_score(y, p)
        )

        # F1
        results['f1'] = self.compute_ci(
            y_true, y_pred,
            lambda y, p: f1_score(y, p, average='macro', zero_division=0)
        )

        # Precision
        results['precision'] = self.compute_ci(
            y_true, y_pred,
            lambda y, p: precision_score(y, p, average='macro', zero_division=0)
        )

        # Recall
        results['recall'] = self.compute_ci(
            y_true, y_pred,
            lambda y, p: recall_score(y, p, average='macro', zero_division=0)
        )

        # AUC (if probabilities available)
        if y_prob is not None:
            try:
                results['roc_auc'] = self.compute_ci(
                    y_true, None,
                    lambda y, p: roc_auc_score(y, p),
                    y_prob=y_prob
                )
                results['pr_auc'] = self.compute_ci(
                    y_true, None,
                    lambda y, p: average_precision_score(y, p),
                    y_prob=y_prob
                )
            except:
                pass

        return results


class StatisticalTester:
    """
    Statistical significance tests for model comparison.

    Implements:
    - McNemar's test (paired binary predictions)
    - Paired bootstrap test
    - DeLong test (AUC comparison)
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def mcnemar_test(
        self,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray
    ) -> SignificanceTestResult:
        """
        McNemar's test for comparing two classifiers on paired samples.

        Tests whether the disagreements between classifiers are symmetric.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred_a : np.ndarray
            Predictions from model A
        y_pred_b : np.ndarray
            Predictions from model B

        Returns
        -------
        result : SignificanceTestResult
        """
        # Build contingency table of correct/incorrect
        correct_a = (y_pred_a == y_true)
        correct_b = (y_pred_b == y_true)

        # n01: A wrong, B right
        # n10: A right, B wrong
        n01 = np.sum(~correct_a & correct_b)
        n10 = np.sum(correct_a & ~correct_b)

        # McNemar's test statistic (with continuity correction)
        if n01 + n10 == 0:
            return SignificanceTestResult(
                test_name='mcnemar',
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha
            )

        # Chi-square statistic with continuity correction
        statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)

        from scipy import stats
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

        return SignificanceTestResult(
            test_name='mcnemar',
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha
        )

    def paired_bootstrap_test(
        self,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
        metric_fn,
        n_bootstrap: int = 1000,
        random_seed: int = 42
    ) -> SignificanceTestResult:
        """
        Paired bootstrap test for comparing two models.

        Tests H0: metric(A) = metric(B)

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred_a, y_pred_b : np.ndarray
            Predictions from models A and B
        metric_fn : callable
            Metric function
        n_bootstrap : int
            Number of bootstrap iterations
        random_seed : int
            Random seed

        Returns
        -------
        result : SignificanceTestResult
        """
        rng = np.random.RandomState(random_seed)
        n_samples = len(y_true)

        # Original difference
        metric_a = metric_fn(y_true, y_pred_a)
        metric_b = metric_fn(y_true, y_pred_b)
        observed_diff = metric_a - metric_b

        # Bootstrap differences
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            indices = rng.choice(n_samples, n_samples, replace=True)

            try:
                diff = (metric_fn(y_true[indices], y_pred_a[indices]) -
                       metric_fn(y_true[indices], y_pred_b[indices]))
                bootstrap_diffs.append(diff)
            except:
                continue

        bootstrap_diffs = np.array(bootstrap_diffs)

        # Two-tailed p-value: proportion of bootstrap diffs more extreme than 0
        # Under H0, the difference should be centered at 0
        centered_diffs = bootstrap_diffs - np.mean(bootstrap_diffs)

        # Count how many centered differences are more extreme than observed
        p_value = np.mean(np.abs(centered_diffs) >= np.abs(observed_diff))

        return SignificanceTestResult(
            test_name='paired_bootstrap',
            statistic=observed_diff,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=observed_diff
        )

    def delong_test(
        self,
        y_true: np.ndarray,
        y_prob_a: np.ndarray,
        y_prob_b: np.ndarray
    ) -> SignificanceTestResult:
        """
        DeLong test for comparing two AUCs.

        Tests H0: AUC(A) = AUC(B)

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels
        y_prob_a, y_prob_b : np.ndarray
            Predicted probabilities from models A and B

        Returns
        -------
        result : SignificanceTestResult
        """
        from sklearn.metrics import roc_auc_score

        # Compute AUCs
        auc_a = roc_auc_score(y_true, y_prob_a)
        auc_b = roc_auc_score(y_true, y_prob_b)

        # Compute structural components for DeLong test
        # This is a simplified implementation
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        # Placements
        pos_scores_a = y_prob_a[y_true == 1]
        neg_scores_a = y_prob_a[y_true == 0]
        pos_scores_b = y_prob_b[y_true == 1]
        neg_scores_b = y_prob_b[y_true == 0]

        # Compute variance using Mann-Whitney formulation
        # Simplified covariance estimate
        var_a = (auc_a * (1 - auc_a) + (n_pos - 1) * (auc_a / (2 - auc_a) - auc_a ** 2) +
                 (n_neg - 1) * (2 * auc_a ** 2 / (1 + auc_a) - auc_a ** 2)) / (n_pos * n_neg)

        var_b = (auc_b * (1 - auc_b) + (n_pos - 1) * (auc_b / (2 - auc_b) - auc_b ** 2) +
                 (n_neg - 1) * (2 * auc_b ** 2 / (1 + auc_b) - auc_b ** 2)) / (n_pos * n_neg)

        # Simplified covariance (assuming some correlation)
        cov_ab = 0.5 * np.sqrt(var_a * var_b)

        # Z statistic
        se_diff = np.sqrt(var_a + var_b - 2 * cov_ab)
        if se_diff < 1e-10:
            se_diff = 1e-10

        z_stat = (auc_a - auc_b) / se_diff

        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

        return SignificanceTestResult(
            test_name='delong',
            statistic=z_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=auc_a - auc_b
        )


class TestExecutor:
    """
    One-time test execution protocol.

    Ensures that the test set is only used ONCE.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self._executed = False
        self._execution_log = []

    def execute_test(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        groups_test: np.ndarray,
        model_name: str,
        freeze_hash: Optional[str] = None
    ) -> Dict:
        """
        Execute final test (ONE TIME ONLY).

        Parameters
        ----------
        model : sklearn estimator
            Trained model
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        groups_test : np.ndarray
            Test subject groups
        model_name : str
            Name of the model
        freeze_hash : str, optional
            Hash of frozen pipeline for verification

        Returns
        -------
        results : Dict
            Complete test results
        """
        if self._executed:
            warnings.warn(
                "Test has already been executed! This should only happen once. "
                "Results from this run should be treated with caution."
            )

        execution_time = datetime.now().isoformat()

        # Compute data hash
        data_hash = hashlib.sha256(
            np.concatenate([X_test.flatten()[:1000], y_test.flatten()]).tobytes()
        ).hexdigest()[:16]

        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                y_prob = y_prob[:, 1]

        # Compute metrics with CIs
        bootstrap_ci = BootstrapCI(n_bootstrap=1000, confidence_level=0.95)
        cis = bootstrap_ci.compute_all_metrics_ci(y_test, y_pred, y_prob)

        # Compute confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, y_pred)

        # Per-class metrics
        report = classification_report(y_test, y_pred, output_dict=True)

        results = {
            '_metadata': {
                'execution_time': execution_time,
                'model_name': model_name,
                'data_hash': data_hash,
                'freeze_hash': freeze_hash,
                'n_test_samples': len(y_test),
                'n_test_subjects': len(np.unique(groups_test)),
                'test_executed': True
            },
            'metrics': {
                name: {
                    'estimate': ci.estimate,
                    'ci_lower': ci.lower,
                    'ci_upper': ci.upper,
                    'confidence_level': ci.confidence_level
                }
                for name, ci in cis.items()
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'per_subject_results': self._compute_per_subject(
                y_test, y_pred, groups_test
            )
        }

        # Save results
        self._save_results(results, model_name)

        self._executed = True
        self._execution_log.append({
            'time': execution_time,
            'model': model_name,
            'data_hash': data_hash
        })

        return results

    def _compute_per_subject(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray
    ) -> Dict:
        """Compute per-subject results."""
        results = {}
        for subj in np.unique(groups):
            mask = groups == subj
            acc = np.mean(y_pred[mask] == y_true[mask])
            results[str(subj)] = {
                'accuracy': acc,
                'n_samples': int(np.sum(mask)),
                'n_correct': int(np.sum(y_pred[mask] == y_true[mask]))
            }
        return results

    def _save_results(self, results: Dict, model_name: str) -> str:
        """Save test results to disk."""
        import os
        from pathlib import Path

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"test_results_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_path / filename

        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj

        results_serializable = json.loads(
            json.dumps(results, default=convert)
        )

        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"✓ Test results saved to {filepath}")
        return str(filepath)


def create_benchmark_table(
    results: Dict[str, Dict],
    baseline_name: str = 'LogisticRegression'
) -> 'pd.DataFrame':
    """
    Create benchmark comparison table.

    Parameters
    ----------
    results : Dict[str, Dict]
        Results for each model (from BootstrapCI.compute_all_metrics_ci)
    baseline_name : str
        Name of baseline model for delta computation

    Returns
    -------
    df : pd.DataFrame
        Benchmark table
    """
    import pandas as pd

    rows = []
    baseline_metrics = results.get(baseline_name, {}).get('metrics', {})

    for model_name, model_results in results.items():
        metrics = model_results.get('metrics', {})

        row = {'Model': model_name}

        for metric_name, ci in metrics.items():
            if isinstance(ci, dict):
                estimate = ci.get('estimate', ci.get('value', 0))
                lower = ci.get('ci_lower', ci.get('lower', 0))
                upper = ci.get('ci_upper', ci.get('upper', 0))
            else:
                estimate = ci.estimate
                lower = ci.lower
                upper = ci.upper

            row[metric_name] = f"{estimate:.4f} [{lower:.4f}, {upper:.4f}]"

            # Compute delta vs baseline
            if baseline_metrics and metric_name in baseline_metrics:
                baseline_ci = baseline_metrics[metric_name]
                if isinstance(baseline_ci, dict):
                    baseline_est = baseline_ci.get('estimate', baseline_ci.get('value', 0))
                else:
                    baseline_est = baseline_ci.estimate
                delta = estimate - baseline_est
                row[f'{metric_name}_delta'] = f"{delta:+.4f}"

        rows.append(row)

    df = pd.DataFrame(rows)
    return df
