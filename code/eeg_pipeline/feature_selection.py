"""
Feature Selection with Riemannian Geometry (Phase 6)
=====================================================

This module implements:
- Riemannian geometry features (strong EEG baseline)
- Filter methods (effect size, mutual information)
- Stability selection
- Feature ablation framework
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from scipy import stats
import warnings

# Try to import pyriemann for Riemannian geometry
try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.classification import MDM, TSclassifier
    PYRIEMANN_AVAILABLE = True
except ImportError:
    PYRIEMANN_AVAILABLE = False
    warnings.warn(
        "pyriemann not installed. Riemannian features will not be available. "
        "Install with: pip install pyriemann"
    )


@dataclass
class FeatureImportance:
    """Container for feature importance scores."""
    feature_names: List[str]
    scores: np.ndarray
    method: str
    threshold: Optional[float] = None
    selected_features: Optional[List[str]] = None


class RiemannianFeatures:
    """
    Riemannian geometry-based features for EEG.

    This is a STRONG baseline for EEG classification that is:
    - Montage-robust (works across different channel configurations)
    - Requires no feature engineering
    - Often competitive with deep learning

    Parameters
    ----------
    estimator : str
        Covariance estimator: 'scm' (sample), 'lwf' (Ledoit-Wolf), 'oas' (Oracle)
    metric : str
        Riemannian metric: 'riemann', 'logeuclid', 'euclid'
    regularization : float
        Regularization for covariance estimation (default: 1e-6)
    """

    def __init__(
        self,
        estimator: str = 'lwf',
        metric: str = 'riemann',
        regularization: float = 1e-6
    ):
        if not PYRIEMANN_AVAILABLE:
            raise ImportError(
                "pyriemann is required for Riemannian features. "
                "Install with: pip install pyriemann"
            )

        self.estimator = estimator
        self.metric = metric
        self.regularization = regularization

        self._cov_estimator = Covariances(estimator=estimator)
        self._tangent_space = TangentSpace(metric=metric)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RiemannianFeatures':
        """
        Fit Riemannian feature extractor.

        Parameters
        ----------
        X : np.ndarray
            EEG data (n_samples, n_channels, n_timepoints)
        y : np.ndarray
            Labels

        Returns
        -------
        self : RiemannianFeatures
        """
        # Compute covariance matrices
        covmats = self._compute_covariances(X)

        # Fit tangent space projection
        self._tangent_space.fit(covmats, y)

        self._n_channels = X.shape[1]
        self._n_features = self._n_channels * (self._n_channels + 1) // 2
        self._fitted = True

        print(f"✓ RiemannianFeatures fitted: {self._n_features} tangent space features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform EEG to Riemannian tangent space features.

        Parameters
        ----------
        X : np.ndarray
            EEG data (n_samples, n_channels, n_timepoints)

        Returns
        -------
        features : np.ndarray
            Tangent space features (n_samples, n_features)
        """
        if not self._fitted:
            raise RuntimeError("RiemannianFeatures not fitted!")

        covmats = self._compute_covariances(X)
        features = self._tangent_space.transform(covmats)

        return features

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def _compute_covariances(self, X: np.ndarray) -> np.ndarray:
        """Compute regularized covariance matrices."""
        covmats = self._cov_estimator.fit_transform(X)

        # Add regularization for numerical stability
        n_channels = covmats.shape[1]
        reg_matrix = self.regularization * np.eye(n_channels)
        covmats = covmats + reg_matrix

        return covmats

    def get_feature_names(self) -> List[str]:
        """Get names of tangent space features."""
        if not self._fitted:
            raise RuntimeError("RiemannianFeatures not fitted!")

        names = []
        for i in range(self._n_channels):
            for j in range(i, self._n_channels):
                names.append(f"ts_cov_{i}_{j}")
        return names


class RiemannianClassifier:
    """
    Riemannian geometry-based classifier.

    Options:
    - MDM: Minimum Distance to Mean (simple, robust)
    - TSclassifier: Tangent Space + any sklearn classifier
    """

    def __init__(
        self,
        method: str = 'ts_lr',
        metric: str = 'riemann',
        estimator: str = 'lwf'
    ):
        if not PYRIEMANN_AVAILABLE:
            raise ImportError("pyriemann is required")

        self.method = method
        self.metric = metric
        self.estimator = estimator

        if method == 'mdm':
            self._clf = MDM(metric=metric)
        elif method == 'ts_lr':
            self._clf = TSclassifier(
                metric=metric,
                clf=LogisticRegression(max_iter=1000)
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        self._cov_estimator = Covariances(estimator=estimator)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RiemannianClassifier':
        """Fit classifier on covariance matrices."""
        covmats = self._cov_estimator.fit_transform(X)
        self._clf.fit(covmats, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        covmats = self._cov_estimator.transform(X)
        return self._clf.predict(covmats)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (if available)."""
        covmats = self._cov_estimator.transform(X)
        if hasattr(self._clf, 'predict_proba'):
            return self._clf.predict_proba(covmats)
        else:
            # For MDM, use softmax of negative distances
            distances = self._clf.transform(covmats)
            return self._softmax(-distances)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class FeatureSelector:
    """
    Feature selection with multiple methods.

    Methods:
    - effect_size: Cohen's d for each feature
    - mutual_info: Mutual information with labels
    - anova: F-statistic from ANOVA
    - l1: L1-regularized logistic regression coefficients
    """

    def __init__(
        self,
        method: str = 'effect_size',
        n_features: Optional[int] = None,
        threshold: Optional[float] = None
    ):
        self.method = method
        self.n_features = n_features
        self.threshold = threshold

        self._importance = None
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'FeatureSelector':
        """
        Fit feature selector on training data.

        Parameters
        ----------
        X : np.ndarray
            Features (n_samples, n_features)
        y : np.ndarray
            Labels
        feature_names : List[str], optional
            Names of features

        Returns
        -------
        self : FeatureSelector
        """
        n_samples, n_features = X.shape

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        if self.method == 'effect_size':
            scores = self._compute_effect_size(X, y)
        elif self.method == 'mutual_info':
            scores = mutual_info_classif(X, y, random_state=42)
        elif self.method == 'anova':
            scores, _ = f_classif(X, y)
        elif self.method == 'l1':
            scores = self._compute_l1_importance(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Handle NaN scores
        scores = np.nan_to_num(scores, nan=0.0)

        # Determine selected features
        if self.n_features is not None:
            selected_idx = np.argsort(scores)[-self.n_features:]
        elif self.threshold is not None:
            selected_idx = np.where(scores >= self.threshold)[0]
        else:
            selected_idx = np.arange(n_features)

        selected_features = [feature_names[i] for i in selected_idx]

        self._importance = FeatureImportance(
            feature_names=feature_names,
            scores=scores,
            method=self.method,
            threshold=self.threshold,
            selected_features=selected_features
        )

        self._selected_idx = selected_idx
        self._fitted = True

        print(f"✓ FeatureSelector fitted: {len(selected_features)}/{n_features} features selected")
        return self

    def _compute_effect_size(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute Cohen's d effect size for each feature."""
        classes = np.unique(y)
        if len(classes) != 2:
            warnings.warn("Effect size computed for binary classification only. Using first two classes.")
            classes = classes[:2]

        mask0 = y == classes[0]
        mask1 = y == classes[1]

        mean0 = np.mean(X[mask0], axis=0)
        mean1 = np.mean(X[mask1], axis=0)
        std0 = np.std(X[mask0], axis=0)
        std1 = np.std(X[mask1], axis=0)

        # Pooled standard deviation
        n0 = np.sum(mask0)
        n1 = np.sum(mask1)
        pooled_std = np.sqrt(((n0 - 1) * std0**2 + (n1 - 1) * std1**2) / (n0 + n1 - 2))
        pooled_std = np.maximum(pooled_std, 1e-10)  # Prevent division by zero

        # Cohen's d
        cohens_d = np.abs(mean1 - mean0) / pooled_std

        return cohens_d

    def _compute_l1_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute feature importance from L1-regularized logistic regression."""
        from sklearn.preprocessing import StandardScaler

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit L1-regularized model
        clf = LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=42)
        clf.fit(X_scaled, y)

        # Use absolute coefficient values as importance
        if len(clf.coef_.shape) > 1:
            importance = np.mean(np.abs(clf.coef_), axis=0)
        else:
            importance = np.abs(clf.coef_)

        return importance

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select features."""
        if not self._fitted:
            raise RuntimeError("FeatureSelector not fitted!")
        return X[:, self._selected_idx]

    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_importance_table(self) -> 'pd.DataFrame':
        """Get feature importance as a DataFrame."""
        import pandas as pd

        if not self._fitted:
            raise RuntimeError("FeatureSelector not fitted!")

        df = pd.DataFrame({
            'feature': self._importance.feature_names,
            'score': self._importance.scores,
            'selected': [f in self._importance.selected_features for f in self._importance.feature_names]
        })
        df = df.sort_values('score', ascending=False).reset_index(drop=True)

        return df


class StabilitySelection:
    """
    Stability selection for robust feature selection.

    Runs feature selection on bootstrap samples and keeps
    features that are consistently selected.

    Parameters
    ----------
    base_selector : FeatureSelector
        Base feature selector to use
    n_bootstrap : int
        Number of bootstrap iterations (default: 100)
    threshold : float
        Minimum selection frequency to keep feature (default: 0.7)
    sample_fraction : float
        Fraction of data to use in each bootstrap (default: 0.8)
    random_seed : int
        Random seed for reproducibility
    """

    def __init__(
        self,
        base_selector: FeatureSelector,
        n_bootstrap: int = 100,
        threshold: float = 0.7,
        sample_fraction: float = 0.8,
        random_seed: int = 42
    ):
        self.base_selector = base_selector
        self.n_bootstrap = n_bootstrap
        self.threshold = threshold
        self.sample_fraction = sample_fraction
        self.random_seed = random_seed

        self._selection_frequencies = None
        self._stable_features = None
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'StabilitySelection':
        """
        Fit stability selection.

        Parameters
        ----------
        X : np.ndarray
            Features (n_samples, n_features)
        y : np.ndarray
            Labels
        feature_names : List[str], optional
            Names of features

        Returns
        -------
        self : StabilitySelection
        """
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_seed)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Track selection counts
        selection_counts = np.zeros(n_features)

        for i in range(self.n_bootstrap):
            # Bootstrap sample
            n_sample = int(n_samples * self.sample_fraction)
            indices = rng.choice(n_samples, size=n_sample, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Fit selector
            selector = FeatureSelector(
                method=self.base_selector.method,
                n_features=self.base_selector.n_features,
                threshold=self.base_selector.threshold
            )
            selector.fit(X_boot, y_boot, feature_names)

            # Count selections
            selection_counts[selector._selected_idx] += 1

        # Compute selection frequencies
        self._selection_frequencies = selection_counts / self.n_bootstrap

        # Select stable features
        stable_idx = np.where(self._selection_frequencies >= self.threshold)[0]
        self._stable_features = [feature_names[i] for i in stable_idx]
        self._stable_idx = stable_idx
        self._feature_names = feature_names
        self._fitted = True

        print(f"✓ StabilitySelection: {len(self._stable_features)}/{n_features} stable features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select stable features."""
        if not self._fitted:
            raise RuntimeError("StabilitySelection not fitted!")
        return X[:, self._stable_idx]

    def get_stability_table(self) -> 'pd.DataFrame':
        """Get stability results as DataFrame."""
        import pandas as pd

        if not self._fitted:
            raise RuntimeError("StabilitySelection not fitted!")

        df = pd.DataFrame({
            'feature': self._feature_names,
            'selection_frequency': self._selection_frequencies,
            'stable': self._selection_frequencies >= self.threshold
        })
        df = df.sort_values('selection_frequency', ascending=False).reset_index(drop=True)

        return df


class FeatureAblation:
    """
    Feature ablation study framework.

    Removes features/feature families and measures impact on performance.
    """

    def __init__(
        self,
        model,
        scorer,
        cv=5
    ):
        """
        Parameters
        ----------
        model : sklearn estimator
            Model to evaluate
        scorer : callable
            Scoring function (higher is better)
        cv : int or cross-validator
            Cross-validation strategy
        """
        self.model = model
        self.scorer = scorer
        self.cv = cv

    def run_ablation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_groups: Dict[str, List[int]],
        groups: Optional[np.ndarray] = None
    ) -> Dict[str, Dict]:
        """
        Run ablation study by removing feature groups.

        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        feature_groups : Dict[str, List[int]]
            Mapping from group name to feature indices
        groups : np.ndarray, optional
            Subject groups for GroupKFold

        Returns
        -------
        results : Dict[str, Dict]
            Ablation results with scores and deltas
        """
        from sklearn.model_selection import cross_val_score, GroupKFold

        # Baseline with all features
        if groups is not None:
            cv = GroupKFold(n_splits=self.cv)
            cv_iter = cv.split(X, y, groups)
        else:
            cv_iter = self.cv

        baseline_scores = cross_val_score(
            self.model, X, y, cv=cv_iter, scoring=self.scorer
        )
        baseline_mean = np.mean(baseline_scores)

        results = {
            "baseline": {
                "score_mean": baseline_mean,
                "score_std": np.std(baseline_scores),
                "n_features": X.shape[1]
            }
        }

        # Ablate each group
        for group_name, feature_indices in feature_groups.items():
            # Create mask to remove these features
            mask = np.ones(X.shape[1], dtype=bool)
            mask[feature_indices] = False
            X_ablated = X[:, mask]

            if groups is not None:
                cv_iter = cv.split(X_ablated, y, groups)
            else:
                cv_iter = self.cv

            ablated_scores = cross_val_score(
                self.model, X_ablated, y, cv=cv_iter, scoring=self.scorer
            )
            ablated_mean = np.mean(ablated_scores)

            results[f"without_{group_name}"] = {
                "score_mean": ablated_mean,
                "score_std": np.std(ablated_scores),
                "n_features": X_ablated.shape[1],
                "delta": baseline_mean - ablated_mean,
                "delta_pct": (baseline_mean - ablated_mean) / baseline_mean * 100
            }

        return results

    def get_ablation_table(self, results: Dict[str, Dict]) -> 'pd.DataFrame':
        """Convert ablation results to DataFrame."""
        import pandas as pd

        rows = []
        for name, data in results.items():
            rows.append({
                "configuration": name,
                "score": f"{data['score_mean']:.4f} ± {data['score_std']:.4f}",
                "n_features": data['n_features'],
                "delta": data.get('delta', 0),
                "delta_pct": f"{data.get('delta_pct', 0):.1f}%"
            })

        return pd.DataFrame(rows)


# Feature families for EEG (typical)
EEG_FEATURE_FAMILIES = {
    "time_domain": {
        "features": ["mean", "variance", "skewness", "kurtosis", "rms", "hjorth_activity",
                     "hjorth_mobility", "hjorth_complexity", "zero_crossings", "peak_to_peak"],
        "description": "Statistical features from time series"
    },
    "frequency_domain": {
        "features": ["delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
                     "total_power", "peak_frequency", "spectral_entropy", "spectral_edge"],
        "description": "Power spectral features"
    },
    "connectivity": {
        "features": ["coherence", "plv", "pli", "wpli", "correlation"],
        "description": "Inter-channel connectivity"
    },
    "nonlinear": {
        "features": ["sample_entropy", "approximate_entropy", "lzc", "hurst_exponent",
                     "fractal_dimension", "dfa"],
        "description": "Complexity and nonlinear dynamics"
    },
    "riemannian": {
        "features": ["tangent_space"],
        "description": "Riemannian geometry (covariance-based)"
    }
}
