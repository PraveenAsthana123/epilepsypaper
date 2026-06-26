"""
Feature Engineering Analysis Module
====================================

Comprehensive feature analysis for EEG-based ML models.

Covers:
- Feature importance analysis
- Feature stability testing
- Ablation studies
- Interpretability analysis
- Domain-specific feature validation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class FeatureImportanceResult:
    """Container for feature importance result."""
    feature_name: str
    importance_score: float
    std: Optional[float] = None
    rank: Optional[int] = None
    domain: Optional[str] = None  # e.g., 'spectral', 'temporal', 'spatial'


@dataclass
class AblationResult:
    """Result from feature ablation study."""
    feature_group: str
    n_features: int
    baseline_performance: float
    ablated_performance: float
    performance_drop: float
    relative_importance: float


@dataclass
class StabilityResult:
    """Feature stability analysis result."""
    feature_name: str
    selection_frequency: float
    bootstrap_mean: float
    bootstrap_std: float
    stable: bool


# =============================================================================
# Feature Importance Analysis
# =============================================================================

class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis.

    Methods:
    - Permutation importance
    - SHAP-like analysis
    - Model-based importance (for tree models)
    - Effect size analysis
    """

    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Initialize importance analyzer.

        Parameters
        ----------
        model : sklearn-compatible model
            Trained model
        feature_names : list, optional
            Names of features
        """
        self.model = model
        self.feature_names = feature_names

    def permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        scoring: str = 'accuracy'
    ) -> List[FeatureImportanceResult]:
        """
        Compute permutation importance.

        Parameters
        ----------
        X : array
            Features
        y : array
            Labels
        n_repeats : int
            Number of permutation repeats
        scoring : str
            Scoring metric

        Returns
        -------
        List of FeatureImportanceResult sorted by importance
        """
        n_features = X.shape[1]

        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(n_features)]

        # Baseline score
        baseline_score = self._compute_score(X, y, scoring)

        importances = []
        importance_std = []

        for feat_idx in range(n_features):
            scores = []

            for _ in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[:, feat_idx] = np.random.permutation(X_permuted[:, feat_idx])
                score = self._compute_score(X_permuted, y, scoring)
                scores.append(baseline_score - score)

            importances.append(np.mean(scores))
            importance_std.append(np.std(scores))

        # Create results
        results = []
        sorted_idx = np.argsort(importances)[::-1]

        for rank, idx in enumerate(sorted_idx):
            results.append(FeatureImportanceResult(
                feature_name=self.feature_names[idx],
                importance_score=importances[idx],
                std=importance_std[idx],
                rank=rank + 1,
                domain=self._infer_domain(self.feature_names[idx])
            ))

        return results

    def effect_size_importance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> List[FeatureImportanceResult]:
        """
        Compute feature importance based on effect size (Cohen's d).

        Parameters
        ----------
        X : array
            Features
        y : array
            Labels (binary)

        Returns
        -------
        List of FeatureImportanceResult
        """
        n_features = X.shape[1]

        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(n_features)]

        class_0_mask = y == 0
        class_1_mask = y == 1

        results = []

        for feat_idx in range(n_features):
            feat_0 = X[class_0_mask, feat_idx]
            feat_1 = X[class_1_mask, feat_idx]

            # Cohen's d
            pooled_std = np.sqrt(
                ((len(feat_0) - 1) * np.var(feat_0, ddof=1) +
                 (len(feat_1) - 1) * np.var(feat_1, ddof=1)) /
                (len(feat_0) + len(feat_1) - 2)
            )

            if pooled_std > 0:
                cohens_d = abs(np.mean(feat_1) - np.mean(feat_0)) / pooled_std
            else:
                cohens_d = 0

            results.append(FeatureImportanceResult(
                feature_name=self.feature_names[feat_idx],
                importance_score=cohens_d,
                domain=self._infer_domain(self.feature_names[feat_idx])
            ))

        # Sort by importance
        results.sort(key=lambda x: x.importance_score, reverse=True)

        for rank, result in enumerate(results):
            result.rank = rank + 1

        return results

    def model_based_importance(self) -> Optional[List[FeatureImportanceResult]]:
        """
        Get model-based feature importance (for tree-based models).

        Returns
        -------
        List of FeatureImportanceResult or None if not available
        """
        if not hasattr(self.model, 'feature_importances_'):
            return None

        importances = self.model.feature_importances_
        n_features = len(importances)

        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(n_features)]

        results = []
        sorted_idx = np.argsort(importances)[::-1]

        for rank, idx in enumerate(sorted_idx):
            results.append(FeatureImportanceResult(
                feature_name=self.feature_names[idx],
                importance_score=importances[idx],
                rank=rank + 1,
                domain=self._infer_domain(self.feature_names[idx])
            ))

        return results

    def _compute_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scoring: str
    ) -> float:
        """Compute model score."""
        y_pred = self.model.predict(X)

        if scoring == 'accuracy':
            return np.mean(y_pred == y)
        elif scoring == 'f1':
            from metrics import ClassificationMetrics
            return ClassificationMetrics.f1_score(y, y_pred)
        else:
            return np.mean(y_pred == y)

    def _infer_domain(self, feature_name: str) -> str:
        """Infer feature domain from name."""
        name_lower = feature_name.lower()

        if any(x in name_lower for x in ['alpha', 'beta', 'gamma', 'theta', 'delta', 'power', 'psd', 'band']):
            return 'spectral'
        elif any(x in name_lower for x in ['variance', 'mean', 'std', 'kurtosis', 'skew', 'entropy']):
            return 'temporal'
        elif any(x in name_lower for x in ['coherence', 'correlation', 'connectivity', 'plv']):
            return 'connectivity'
        elif any(x in name_lower for x in ['fp', 'f3', 'f4', 'c3', 'c4', 'p3', 'p4', 'o1', 'o2', 'channel']):
            return 'spatial'
        else:
            return 'unknown'


# =============================================================================
# Feature Stability Analysis
# =============================================================================

class FeatureStabilityAnalyzer:
    """
    Analyze stability of feature selection across bootstrap samples.
    """

    def __init__(self, selector, feature_names: Optional[List[str]] = None):
        """
        Initialize stability analyzer.

        Parameters
        ----------
        selector : sklearn-compatible selector
            Feature selector with fit_transform method
        feature_names : list, optional
            Names of features
        """
        self.selector = selector
        self.feature_names = feature_names

    def analyze_stability(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 100,
        sample_ratio: float = 0.8
    ) -> List[StabilityResult]:
        """
        Analyze feature selection stability across bootstrap samples.

        Parameters
        ----------
        X : array
            Features
        y : array
            Labels
        n_bootstrap : int
            Number of bootstrap iterations
        sample_ratio : float
            Proportion of samples per bootstrap

        Returns
        -------
        List of StabilityResult
        """
        n_samples, n_features = X.shape
        n_select = int(n_samples * sample_ratio)

        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(n_features)]

        # Track selection frequency
        selection_counts = np.zeros(n_features)

        for _ in range(n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(n_samples, n_select, replace=True)
            X_boot = X[idx]
            y_boot = y[idx]

            # Fit selector
            try:
                self.selector.fit(X_boot, y_boot)

                # Get selected features
                if hasattr(self.selector, 'get_support'):
                    selected = self.selector.get_support()
                elif hasattr(self.selector, 'selected_features_'):
                    selected = np.zeros(n_features, dtype=bool)
                    selected[self.selector.selected_features_] = True
                else:
                    # Try transform and infer from shape
                    X_transformed = self.selector.transform(X_boot)
                    selected = np.zeros(n_features, dtype=bool)
                    selected[:X_transformed.shape[1]] = True

                selection_counts += selected.astype(int)
            except Exception as e:
                warnings.warn(f"Bootstrap iteration failed: {e}")

        # Compute selection frequency
        selection_freq = selection_counts / n_bootstrap

        # Create results
        results = []
        for i in range(n_features):
            results.append(StabilityResult(
                feature_name=self.feature_names[i],
                selection_frequency=selection_freq[i],
                bootstrap_mean=selection_freq[i],
                bootstrap_std=np.sqrt(selection_freq[i] * (1 - selection_freq[i]) / n_bootstrap),
                stable=selection_freq[i] >= 0.5
            ))

        # Sort by stability
        results.sort(key=lambda x: x.selection_frequency, reverse=True)

        return results

    def compute_stability_index(
        self,
        stability_results: List[StabilityResult]
    ) -> float:
        """
        Compute overall stability index (Kuncheva index approximation).

        Parameters
        ----------
        stability_results : list
            Results from analyze_stability

        Returns
        -------
        Stability index (0-1, higher is better)
        """
        frequencies = [r.selection_frequency for r in stability_results]

        # Stability index based on consistency
        # High if features are either always selected or never selected
        stability = np.mean([max(f, 1 - f) for f in frequencies])

        return float(stability)


# =============================================================================
# Ablation Studies
# =============================================================================

class AblationStudyRunner:
    """
    Run systematic feature ablation studies.
    """

    def __init__(self, model, feature_groups: Optional[Dict[str, List[int]]] = None):
        """
        Initialize ablation study runner.

        Parameters
        ----------
        model : sklearn-compatible model
            Model with fit and predict methods
        feature_groups : dict, optional
            Mapping from group name to feature indices
        """
        self.model = model
        self.feature_groups = feature_groups

    def run_group_ablation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        groups: Optional[Dict[str, List[int]]] = None
    ) -> List[AblationResult]:
        """
        Run ablation study by feature groups.

        Parameters
        ----------
        X_train, y_train : arrays
            Training data
        X_test, y_test : arrays
            Test data
        groups : dict, optional
            Feature groups to ablate

        Returns
        -------
        List of AblationResult
        """
        if groups is None:
            groups = self.feature_groups

        if groups is None:
            # Create default groups (split into quarters)
            n_features = X_train.shape[1]
            quarter = n_features // 4
            groups = {
                'group_1': list(range(0, quarter)),
                'group_2': list(range(quarter, 2 * quarter)),
                'group_3': list(range(2 * quarter, 3 * quarter)),
                'group_4': list(range(3 * quarter, n_features))
            }

        results = []

        # Baseline performance
        self.model.fit(X_train, y_train)
        baseline_acc = np.mean(self.model.predict(X_test) == y_test)

        for group_name, feature_indices in groups.items():
            # Create ablated data (zero out features)
            X_train_ablated = X_train.copy()
            X_test_ablated = X_test.copy()

            X_train_ablated[:, feature_indices] = 0
            X_test_ablated[:, feature_indices] = 0

            # Train and evaluate
            self.model.fit(X_train_ablated, y_train)
            ablated_acc = np.mean(self.model.predict(X_test_ablated) == y_test)

            performance_drop = baseline_acc - ablated_acc

            results.append(AblationResult(
                feature_group=group_name,
                n_features=len(feature_indices),
                baseline_performance=baseline_acc,
                ablated_performance=ablated_acc,
                performance_drop=performance_drop,
                relative_importance=performance_drop / baseline_acc if baseline_acc > 0 else 0
            ))

        # Sort by importance
        results.sort(key=lambda x: x.performance_drop, reverse=True)

        return results

    def run_progressive_ablation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        importance_ranking: List[int],
        steps: int = 10
    ) -> Dict[str, List[float]]:
        """
        Run progressive feature ablation (remove features in order of importance).

        Parameters
        ----------
        X_train, y_train : arrays
            Training data
        X_test, y_test : arrays
            Test data
        importance_ranking : list
            Feature indices sorted by importance (most to least)
        steps : int
            Number of ablation steps

        Returns
        -------
        Dictionary with ablation curve data
        """
        n_features = X_train.shape[1]
        features_per_step = n_features // steps

        results = {
            'n_features_removed': [],
            'performance': [],
            'features_removed': []
        }

        # Baseline
        self.model.fit(X_train, y_train)
        baseline_acc = np.mean(self.model.predict(X_test) == y_test)

        results['n_features_removed'].append(0)
        results['performance'].append(baseline_acc)
        results['features_removed'].append([])

        # Progressive ablation
        features_to_remove = []
        for step in range(1, steps + 1):
            # Add next batch of features to remove
            start_idx = (step - 1) * features_per_step
            end_idx = min(step * features_per_step, n_features)
            features_to_remove.extend(importance_ranking[start_idx:end_idx])

            # Ablate
            X_train_ablated = X_train.copy()
            X_test_ablated = X_test.copy()
            X_train_ablated[:, features_to_remove] = 0
            X_test_ablated[:, features_to_remove] = 0

            self.model.fit(X_train_ablated, y_train)
            acc = np.mean(self.model.predict(X_test_ablated) == y_test)

            results['n_features_removed'].append(len(features_to_remove))
            results['performance'].append(acc)
            results['features_removed'].append(features_to_remove.copy())

        return results


# =============================================================================
# Domain-Specific Feature Validation
# =============================================================================

class EEGFeatureValidator:
    """
    Validate EEG features against domain knowledge.
    """

    # Known EEG frequency bands
    FREQUENCY_BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }

    # Expected patterns for various diseases
    DISEASE_PATTERNS = {
        'depression': {
            'frontal_alpha_asymmetry': 'reduced left > right',
            'theta_power': 'increased',
            'alpha_power': 'often reduced',
            'beta_power': 'variable'
        },
        'alzheimer': {
            'alpha_power': 'reduced',
            'theta_power': 'increased',
            'delta_power': 'increased',
            'coherence': 'reduced'
        },
        'epilepsy': {
            'spike_frequency': 'increased',
            'theta_power': 'increased during seizure',
            'gamma_power': 'increased during seizure'
        },
        'schizophrenia': {
            'gamma_power': 'reduced',
            'alpha_power': 'variable',
            'connectivity': 'altered'
        }
    }

    def __init__(self, disease: str):
        """
        Initialize EEG feature validator.

        Parameters
        ----------
        disease : str
            Target disease for validation
        """
        self.disease = disease.lower()
        self.expected_patterns = self.DISEASE_PATTERNS.get(self.disease, {})

    def validate_top_features(
        self,
        importance_results: List[FeatureImportanceResult],
        n_top: int = 20
    ) -> Dict[str, Any]:
        """
        Validate top features against domain knowledge.

        Parameters
        ----------
        importance_results : list
            Feature importance results
        n_top : int
            Number of top features to validate

        Returns
        -------
        Validation report
        """
        top_features = importance_results[:n_top]

        report = {
            'validated_features': [],
            'unexpected_features': [],
            'missing_expected_features': [],
            'domain_consistency_score': 0.0
        }

        # Check each top feature
        for feat in top_features:
            name_lower = feat.feature_name.lower()
            domain = feat.domain

            is_expected = False

            # Check against disease-specific patterns
            for pattern_name, expected in self.expected_patterns.items():
                if pattern_name.replace('_', '') in name_lower.replace('_', ''):
                    is_expected = True
                    report['validated_features'].append({
                        'feature': feat.feature_name,
                        'expected_pattern': f"{pattern_name}: {expected}",
                        'importance_rank': feat.rank
                    })
                    break

            # Check against general EEG knowledge
            if not is_expected:
                for band_name in self.FREQUENCY_BANDS.keys():
                    if band_name in name_lower:
                        is_expected = True
                        report['validated_features'].append({
                            'feature': feat.feature_name,
                            'expected_pattern': f"Known EEG frequency band: {band_name}",
                            'importance_rank': feat.rank
                        })
                        break

            if not is_expected:
                report['unexpected_features'].append({
                    'feature': feat.feature_name,
                    'domain': domain,
                    'importance_rank': feat.rank
                })

        # Check for missing expected features
        found_patterns = [f['expected_pattern'].split(':')[0]
                         for f in report['validated_features']]

        for pattern_name in self.expected_patterns.keys():
            if not any(pattern_name in f for f in found_patterns):
                report['missing_expected_features'].append(pattern_name)

        # Compute consistency score
        n_validated = len(report['validated_features'])
        n_unexpected = len(report['unexpected_features'])
        n_missing = len(report['missing_expected_features'])

        if n_top > 0:
            report['domain_consistency_score'] = (
                n_validated / n_top -
                0.1 * n_missing / max(len(self.expected_patterns), 1)
            )
            report['domain_consistency_score'] = max(0, min(1, report['domain_consistency_score']))

        return report

    def generate_feature_interpretation(
        self,
        feature_name: str
    ) -> str:
        """
        Generate human-readable interpretation of a feature.

        Parameters
        ----------
        feature_name : str
            Name of the feature

        Returns
        -------
        Interpretation string
        """
        name_lower = feature_name.lower()

        interpretations = []

        # Frequency band
        for band, (low, high) in self.FREQUENCY_BANDS.items():
            if band in name_lower:
                interpretations.append(f"{band} band ({low}-{high} Hz)")

        # Channel/region
        regions = {
            'fp': 'prefrontal',
            'f3': 'left frontal',
            'f4': 'right frontal',
            'c3': 'left central',
            'c4': 'right central',
            'p3': 'left parietal',
            'p4': 'right parietal',
            'o1': 'left occipital',
            'o2': 'right occipital',
            't3': 'left temporal',
            't4': 'right temporal'
        }

        for code, region in regions.items():
            if code in name_lower:
                interpretations.append(f"{region} region")

        # Feature type
        if 'power' in name_lower or 'psd' in name_lower:
            interpretations.append("spectral power")
        elif 'coherence' in name_lower:
            interpretations.append("inter-regional coherence")
        elif 'asymmetry' in name_lower:
            interpretations.append("hemispheric asymmetry")
        elif 'entropy' in name_lower:
            interpretations.append("signal complexity")
        elif 'variance' in name_lower:
            interpretations.append("signal variability")

        if interpretations:
            return f"Measures {' in '.join(interpretations)}"
        else:
            return "Feature interpretation not available"


# =============================================================================
# Comprehensive Feature Analysis Report
# =============================================================================

class FeatureAnalysisReportGenerator:
    """
    Generate comprehensive feature analysis report.
    """

    def __init__(
        self,
        model,
        feature_names: List[str],
        disease: str = 'unknown'
    ):
        """
        Initialize report generator.

        Parameters
        ----------
        model : sklearn-compatible model
            Trained model
        feature_names : list
            Names of features
        disease : str
            Target disease
        """
        self.model = model
        self.feature_names = feature_names
        self.disease = disease

        self.importance_analyzer = FeatureImportanceAnalyzer(model, feature_names)
        self.validator = EEGFeatureValidator(disease)

    def generate_report(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generate comprehensive feature analysis report.

        Parameters
        ----------
        X_train, y_train : arrays
            Training data
        X_test, y_test : arrays
            Test data

        Returns
        -------
        Complete feature analysis report
        """
        report = {
            'feature_importance': {},
            'domain_validation': {},
            'ablation_study': {},
            'interpretations': {},
            'recommendations': []
        }

        # Feature importance (multiple methods)
        perm_importance = self.importance_analyzer.permutation_importance(X_test, y_test)
        report['feature_importance']['permutation'] = [
            {
                'feature': r.feature_name,
                'importance': r.importance_score,
                'std': r.std,
                'rank': r.rank,
                'domain': r.domain
            }
            for r in perm_importance[:20]  # Top 20
        ]

        effect_importance = self.importance_analyzer.effect_size_importance(X_train, y_train)
        report['feature_importance']['effect_size'] = [
            {
                'feature': r.feature_name,
                'cohens_d': r.importance_score,
                'rank': r.rank,
                'domain': r.domain
            }
            for r in effect_importance[:20]
        ]

        # Model-based if available
        model_importance = self.importance_analyzer.model_based_importance()
        if model_importance:
            report['feature_importance']['model_based'] = [
                {
                    'feature': r.feature_name,
                    'importance': r.importance_score,
                    'rank': r.rank
                }
                for r in model_importance[:20]
            ]

        # Domain validation
        report['domain_validation'] = self.validator.validate_top_features(perm_importance)

        # Ablation study
        ablation_runner = AblationStudyRunner(self.model)

        # Create domain-based groups
        spectral_features = [i for i, f in enumerate(self.feature_names) if 'power' in f.lower() or 'psd' in f.lower()]
        temporal_features = [i for i, f in enumerate(self.feature_names) if 'var' in f.lower() or 'mean' in f.lower()]
        connectivity_features = [i for i, f in enumerate(self.feature_names) if 'coh' in f.lower() or 'corr' in f.lower()]

        groups = {}
        if spectral_features:
            groups['spectral'] = spectral_features
        if temporal_features:
            groups['temporal'] = temporal_features
        if connectivity_features:
            groups['connectivity'] = connectivity_features

        if groups:
            ablation_results = ablation_runner.run_group_ablation(
                X_train, y_train, X_test, y_test, groups
            )
            report['ablation_study'] = [
                {
                    'group': r.feature_group,
                    'n_features': r.n_features,
                    'performance_drop': r.performance_drop,
                    'relative_importance': r.relative_importance
                }
                for r in ablation_results
            ]

        # Interpretations for top features
        for feat_result in perm_importance[:10]:
            interpretation = self.validator.generate_feature_interpretation(feat_result.feature_name)
            report['interpretations'][feat_result.feature_name] = interpretation

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Check domain consistency
        consistency = report['domain_validation'].get('domain_consistency_score', 0)
        if consistency < 0.5:
            recommendations.append(
                "WARNING: Low domain consistency score. Important features may not align "
                "with established EEG biomarkers for this disease. Verify feature engineering."
            )

        # Check unexpected features
        unexpected = report['domain_validation'].get('unexpected_features', [])
        if len(unexpected) > 5:
            recommendations.append(
                f"NOTE: {len(unexpected)} unexpected features in top importance. "
                "Consider investigating these for potential novel biomarkers or artifacts."
            )

        # Check missing expected features
        missing = report['domain_validation'].get('missing_expected_features', [])
        if missing:
            recommendations.append(
                f"Consider adding features related to: {', '.join(missing)}. "
                "These are established biomarkers that may improve model performance."
            )

        # Check ablation results
        ablation = report.get('ablation_study', [])
        if ablation:
            top_group = ablation[0] if ablation else None
            if top_group and top_group['relative_importance'] > 0.3:
                recommendations.append(
                    f"The '{top_group['group']}' feature group accounts for "
                    f"{top_group['relative_importance']:.1%} of model performance. "
                    "Consider ensuring robustness in this feature domain."
                )

        if not recommendations:
            recommendations.append(
                "Feature analysis shows good alignment with domain knowledge. "
                "Model appears to be learning from relevant EEG biomarkers."
            )

        return recommendations
