"""
EEG Model Training with Nested CV (Phase 7)
============================================

This module implements:
- Nested cross-validation for unbiased HPO
- Probability calibration (temperature scaling)
- Training with class imbalance handling
- Model selection with proper validation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
from datetime import datetime
import warnings

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    StratifiedGroupKFold, GroupKFold, cross_val_score, cross_val_predict
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve,
    average_precision_score, make_scorer
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import scipy.optimize as optimize


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    random_seed: int = 42
    n_outer_folds: int = 5
    n_inner_folds: int = 3
    scoring: str = 'f1_macro'
    class_weight: Optional[str] = 'balanced'
    early_stopping: bool = True
    patience: int = 15
    calibrate: bool = True
    calibration_method: str = 'temperature'


@dataclass
class TrainingResult:
    """Results from model training."""
    model_name: str
    outer_scores: np.ndarray
    mean_score: float
    std_score: float
    best_params: Optional[Dict] = None
    calibration_error: Optional[float] = None
    training_time: float = 0.0
    config: Optional[TrainingConfig] = None


class NestedCV:
    """
    Nested Cross-Validation for unbiased hyperparameter optimization.

    CRITICAL: This prevents "validation set overfitting" by using:
    - Outer loop: Model evaluation (report these scores)
    - Inner loop: Hyperparameter tuning (select best params)

    Parameters
    ----------
    estimator : sklearn estimator
        Base estimator to optimize
    param_grid : Dict
        Hyperparameter search space
    n_outer_folds : int
        Number of outer CV folds (default: 5)
    n_inner_folds : int
        Number of inner CV folds (default: 3)
    scoring : str or callable
        Scoring metric (default: 'f1_macro')
    random_seed : int
        Random seed for reproducibility
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: Dict,
        n_outer_folds: int = 5,
        n_inner_folds: int = 3,
        scoring: str = 'f1_macro',
        random_seed: int = 42
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.scoring = scoring
        self.random_seed = random_seed

        self._outer_scores = []
        self._best_params_per_fold = []
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray
    ) -> 'NestedCV':
        """
        Run nested cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        groups : np.ndarray
            Subject groups for GroupKFold

        Returns
        -------
        self : NestedCV
        """
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        outer_cv = StratifiedGroupKFold(
            n_splits=self.n_outer_folds,
            shuffle=True,
            random_state=self.random_seed
        )

        self._outer_scores = []
        self._best_params_per_fold = []

        print(f"Running nested CV: {self.n_outer_folds} outer × {self.n_inner_folds} inner folds")

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]

            # Verify no group leakage
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            if train_groups & test_groups:
                raise ValueError(f"Fold {fold_idx}: Group leakage detected!")

            # Inner CV for hyperparameter tuning
            inner_cv = StratifiedGroupKFold(
                n_splits=self.n_inner_folds,
                shuffle=True,
                random_state=self.random_seed
            )

            # Determine search method based on param_grid size
            n_combinations = np.prod([len(v) for v in self.param_grid.values()])

            if n_combinations <= 50:
                search = GridSearchCV(
                    clone(self.estimator),
                    self.param_grid,
                    cv=inner_cv,
                    scoring=self.scoring,
                    n_jobs=-1,
                    refit=True
                )
            else:
                search = RandomizedSearchCV(
                    clone(self.estimator),
                    self.param_grid,
                    n_iter=50,
                    cv=inner_cv,
                    scoring=self.scoring,
                    n_jobs=-1,
                    refit=True,
                    random_state=self.random_seed
                )

            # Fit on training data
            search.fit(X_train, y_train, groups=groups_train)

            # Evaluate on held-out test fold
            y_pred = search.predict(X_test)

            if self.scoring == 'f1_macro':
                score = f1_score(y_test, y_pred, average='macro')
            elif self.scoring == 'accuracy':
                score = accuracy_score(y_test, y_pred)
            elif self.scoring == 'roc_auc':
                if hasattr(search, 'predict_proba'):
                    y_proba = search.predict_proba(X_test)
                    if len(np.unique(y)) == 2:
                        score = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        score = roc_auc_score(y_test, y_proba, multi_class='ovr')
                else:
                    score = accuracy_score(y_test, y_pred)
            else:
                score = accuracy_score(y_test, y_pred)

            self._outer_scores.append(score)
            self._best_params_per_fold.append(search.best_params_)

            print(f"  Fold {fold_idx + 1}: {self.scoring}={score:.4f}, best_params={search.best_params_}")

        self._fitted = True
        return self

    def get_results(self) -> TrainingResult:
        """Get nested CV results."""
        if not self._fitted:
            raise RuntimeError("NestedCV not fitted!")

        scores = np.array(self._outer_scores)

        # Most common best params across folds
        from collections import Counter
        param_strs = [str(p) for p in self._best_params_per_fold]
        most_common = Counter(param_strs).most_common(1)[0][0]
        best_params = eval(most_common)

        return TrainingResult(
            model_name=type(self.estimator).__name__,
            outer_scores=scores,
            mean_score=np.mean(scores),
            std_score=np.std(scores),
            best_params=best_params
        )


class TemperatureScaling:
    """
    Temperature scaling for probability calibration.

    Learns a single temperature parameter T to calibrate probabilities:
    calibrated_proba = softmax(logits / T)

    This is a post-hoc calibration method that preserves accuracy
    while improving probability estimates.
    """

    def __init__(self):
        self.temperature = 1.0
        self._fitted = False

    def fit(
        self,
        logits: np.ndarray,
        y_true: np.ndarray
    ) -> 'TemperatureScaling':
        """
        Fit temperature parameter on validation data.

        Parameters
        ----------
        logits : np.ndarray
            Pre-softmax logits from model (n_samples, n_classes)
        y_true : np.ndarray
            True labels

        Returns
        -------
        self : TemperatureScaling
        """
        def nll_loss(T):
            """Negative log-likelihood with temperature scaling."""
            scaled_logits = logits / T
            probs = self._softmax(scaled_logits)
            # Clip for numerical stability
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            # Cross-entropy loss
            n_samples = len(y_true)
            log_probs = np.log(probs[np.arange(n_samples), y_true])
            return -np.mean(log_probs)

        # Optimize temperature
        result = optimize.minimize_scalar(
            nll_loss,
            bounds=(0.1, 10.0),
            method='bounded'
        )

        self.temperature = result.x
        self._fitted = True

        print(f"✓ Temperature scaling fitted: T={self.temperature:.4f}")
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        if not self._fitted:
            raise RuntimeError("TemperatureScaling not fitted!")

        scaled_logits = logits / self.temperature
        return self._softmax(scaled_logits)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class PlattScaling:
    """
    Platt scaling for probability calibration.

    Fits a logistic regression on the decision function outputs.
    Works well for binary classification.
    """

    def __init__(self):
        self._lr = LogisticRegression(max_iter=1000)
        self._fitted = False

    def fit(
        self,
        decision_values: np.ndarray,
        y_true: np.ndarray
    ) -> 'PlattScaling':
        """
        Fit Platt scaling on validation data.

        Parameters
        ----------
        decision_values : np.ndarray
            Decision function values (n_samples,) or (n_samples, 1)
        y_true : np.ndarray
            True labels

        Returns
        -------
        self : PlattScaling
        """
        if len(decision_values.shape) == 1:
            decision_values = decision_values.reshape(-1, 1)

        self._lr.fit(decision_values, y_true)
        self._fitted = True

        print("✓ Platt scaling fitted")
        return self

    def calibrate(self, decision_values: np.ndarray) -> np.ndarray:
        """Apply Platt scaling."""
        if not self._fitted:
            raise RuntimeError("PlattScaling not fitted!")

        if len(decision_values.shape) == 1:
            decision_values = decision_values.reshape(-1, 1)

        return self._lr.predict_proba(decision_values)


class CalibrationMetrics:
    """Metrics for evaluating probability calibration."""

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE measures the average gap between predicted probability
        and actual accuracy across probability bins.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_prob : np.ndarray
            Predicted probabilities for positive class
        n_bins : int
            Number of bins for calibration

        Returns
        -------
        ece : float
            Expected calibration error (lower is better)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_samples = len(y_true)

        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            if np.sum(mask) == 0:
                continue

            bin_accuracy = np.mean(y_true[mask])
            bin_confidence = np.mean(y_prob[mask])
            bin_size = np.sum(mask) / total_samples

            ece += bin_size * np.abs(bin_accuracy - bin_confidence)

        return ece

    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Compute Brier score.

        Brier score measures the mean squared error of probability estimates.
        Lower is better, range [0, 1].
        """
        return np.mean((y_prob - y_true) ** 2)

    @staticmethod
    def reliability_diagram(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute data for reliability diagram.

        Returns
        -------
        bin_centers : np.ndarray
            Center of each bin
        bin_accuracies : np.ndarray
            Accuracy in each bin
        bin_counts : np.ndarray
            Number of samples in each bin
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_counts = []

        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            if np.sum(mask) == 0:
                continue

            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accuracies.append(np.mean(y_true[mask]))
            bin_counts.append(np.sum(mask))

        return np.array(bin_centers), np.array(bin_accuracies), np.array(bin_counts)


class EEGTrainer:
    """
    EEG Model Trainer with proper validation.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self._results = {}
        self._best_model = None
        self._calibrator = None

    def train_baseline_ladder(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray
    ) -> Dict[str, TrainingResult]:
        """
        Train baseline model ladder.

        Ladder:
        1. Logistic Regression
        2. SVM (RBF kernel)
        3. Random Forest
        4. Gradient Boosting

        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        groups : np.ndarray
            Subject groups

        Returns
        -------
        results : Dict[str, TrainingResult]
            Results for each baseline
        """
        baselines = {
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=1000, class_weight=self.config.class_weight),
                'params': {
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['saga']
                }
            },
            'SVM_RBF': {
                'model': SVC(probability=True, class_weight=self.config.class_weight),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale', 'auto', 0.01, 0.1]
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(
                    class_weight=self.config.class_weight,
                    random_state=self.config.random_seed
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=self.config.random_seed),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }
        }

        results = {}

        for name, config in baselines.items():
            print(f"\n{'='*50}")
            print(f"Training {name}")
            print(f"{'='*50}")

            nested_cv = NestedCV(
                estimator=config['model'],
                param_grid=config['params'],
                n_outer_folds=self.config.n_outer_folds,
                n_inner_folds=self.config.n_inner_folds,
                scoring=self.config.scoring,
                random_seed=self.config.random_seed
            )

            nested_cv.fit(X, y, groups)
            results[name] = nested_cv.get_results()

            print(f"Result: {results[name].mean_score:.4f} ± {results[name].std_score:.4f}")

        self._results = results
        return results

    def select_best_model(self) -> str:
        """Select best model based on mean score."""
        if not self._results:
            raise RuntimeError("No results available. Run train_baseline_ladder first.")

        best_name = max(self._results, key=lambda k: self._results[k].mean_score)
        print(f"\n✓ Best model: {best_name} ({self._results[best_name].mean_score:.4f})")
        return best_name

    def train_final_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> BaseEstimator:
        """
        Train final model with best parameters and calibration.

        Parameters
        ----------
        model_name : str
            Name of model to train
        X_train, y_train : np.ndarray
            Training data
        X_val, y_val : np.ndarray
            Validation data for calibration

        Returns
        -------
        model : BaseEstimator
            Trained and calibrated model
        """
        if model_name not in self._results:
            raise ValueError(f"Unknown model: {model_name}")

        result = self._results[model_name]
        best_params = result.best_params

        # Create model with best params
        if model_name == 'LogisticRegression':
            model = LogisticRegression(
                max_iter=1000,
                class_weight=self.config.class_weight,
                **best_params
            )
        elif model_name == 'SVM_RBF':
            model = SVC(
                probability=True,
                class_weight=self.config.class_weight,
                **best_params
            )
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(
                class_weight=self.config.class_weight,
                random_state=self.config.random_seed,
                **best_params
            )
        elif model_name == 'GradientBoosting':
            model = GradientBoostingClassifier(
                random_state=self.config.random_seed,
                **best_params
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Train on full training data
        model.fit(X_train, y_train)

        # Calibrate on validation data
        if self.config.calibrate:
            if hasattr(model, 'predict_proba'):
                y_prob_val = model.predict_proba(X_val)
                if len(y_prob_val.shape) > 1 and y_prob_val.shape[1] == 2:
                    # Binary classification - use positive class probability
                    y_prob_val = y_prob_val[:, 1]

                    ece_before = CalibrationMetrics.expected_calibration_error(y_val, y_prob_val)
                    print(f"ECE before calibration: {ece_before:.4f}")

                    # Apply Platt scaling
                    decision_vals = model.decision_function(X_val) if hasattr(model, 'decision_function') else y_prob_val
                    self._calibrator = PlattScaling()
                    self._calibrator.fit(decision_vals, y_val)

        self._best_model = model
        return model

    def save_model_bundle(
        self,
        output_dir: str,
        model_name: str,
        normalizer=None,
        feature_selector=None
    ) -> str:
        """
        Save complete model bundle for deployment.

        Parameters
        ----------
        output_dir : str
            Output directory
        model_name : str
            Name of the model
        normalizer : optional
            Fitted normalizer
        feature_selector : optional
            Fitted feature selector

        Returns
        -------
        bundle_path : str
            Path to saved bundle
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        bundle = {
            'model': self._best_model,
            'calibrator': self._calibrator,
            'normalizer': normalizer,
            'feature_selector': feature_selector,
            'config': self.config,
            'results': self._results.get(model_name),
            'created_at': datetime.now().isoformat()
        }

        bundle_path = output_dir / f"{model_name}_bundle.pkl"
        with open(bundle_path, 'wb') as f:
            pickle.dump(bundle, f)

        # Save metadata as JSON
        metadata = {
            'model_name': model_name,
            'config': {
                'random_seed': self.config.random_seed,
                'scoring': self.config.scoring,
                'calibrate': self.config.calibrate
            },
            'performance': {
                'mean_score': float(self._results[model_name].mean_score),
                'std_score': float(self._results[model_name].std_score)
            } if model_name in self._results else {},
            'created_at': datetime.now().isoformat()
        }

        metadata_path = output_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Model bundle saved to {bundle_path}")
        return str(bundle_path)


def get_baseline_results_table(results: Dict[str, TrainingResult]) -> 'pd.DataFrame':
    """Convert training results to a comparison table."""
    import pandas as pd

    rows = []
    for name, result in results.items():
        rows.append({
            'Model': name,
            'Mean Score': f"{result.mean_score:.4f}",
            'Std': f"± {result.std_score:.4f}",
            'Best Params': str(result.best_params)
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('Mean Score', ascending=False)

    return df
