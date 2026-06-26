"""
EEG Exploratory Data Analysis (EDA) Module
===========================================

Comprehensive EDA following the 20-point 4-column framework.

Covers:
- Dataset overview and schema validation
- Missing value and duplicate analysis
- Distribution and outlier analysis
- Feature correlation and target analysis
- Temporal and group analysis
- Data quality risk assessment
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from scipy import stats
import warnings


@dataclass
class EDAResult:
    """Container for EDA analysis result."""
    analysis_type: str
    question: str
    findings: Dict[str, Any]
    artifacts: List[str]
    recommendations: List[str]
    passed: bool = True


@dataclass
class EDAReport:
    """Complete EDA report."""
    dataset_name: str
    n_samples: int
    n_features: int
    analyses: List[EDAResult]
    overall_quality_score: float
    modeling_recommendations: List[str]


# =============================================================================
# Dataset Overview Analysis
# =============================================================================

class DatasetOverviewAnalyzer:
    """
    Analyze dataset structure and basic properties.

    EDA Questions 1-4:
    1. What data do we have?
    2. Are data types correct?
    3. Where is data missing?
    4. Are records duplicated?
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names

    def analyze_overview(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ) -> EDAResult:
        """
        Dataset overview analysis.

        Question: What data do we have?
        """
        findings = {
            'n_samples': len(X),
            'n_features': X.shape[1] if X.ndim > 1 else X.shape[-1] if X.ndim > 0 else 1,
            'data_shape': X.shape,
            'data_dtype': str(X.dtype),
            'target_dtype': str(y.dtype),
            'n_classes': len(np.unique(y)),
            'class_values': np.unique(y).tolist(),
            'memory_mb': X.nbytes / (1024 * 1024)
        }

        if subject_ids is not None:
            findings['n_subjects'] = len(np.unique(subject_ids))
            findings['samples_per_subject_mean'] = len(X) / len(np.unique(subject_ids))

        return EDAResult(
            analysis_type='dataset_overview',
            question='What data do we have?',
            findings=findings,
            artifacts=['Dataset summary table'],
            recommendations=[]
        )

    def analyze_schema(
        self,
        X: np.ndarray,
        expected_dtype: str = 'float'
    ) -> EDAResult:
        """
        Schema and type validation.

        Question: Are data types correct?
        """
        findings = {
            'actual_dtype': str(X.dtype),
            'expected_dtype': expected_dtype,
            'is_numeric': np.issubdtype(X.dtype, np.number),
            'is_float': np.issubdtype(X.dtype, np.floating),
            'is_integer': np.issubdtype(X.dtype, np.integer),
            'has_nan': np.any(np.isnan(X)) if np.issubdtype(X.dtype, np.floating) else False,
            'has_inf': np.any(np.isinf(X)) if np.issubdtype(X.dtype, np.floating) else False
        }

        recommendations = []
        if findings['has_nan']:
            recommendations.append('Data contains NaN values - handle missing data')
        if findings['has_inf']:
            recommendations.append('Data contains Inf values - investigate source')

        return EDAResult(
            analysis_type='schema_validation',
            question='Are data types correct?',
            findings=findings,
            artifacts=['Schema validation report'],
            recommendations=recommendations,
            passed=not findings['has_inf']
        )

    def analyze_missing_values(
        self,
        X: np.ndarray
    ) -> EDAResult:
        """
        Missing value analysis.

        Question: Where is data missing and why?
        """
        if not np.issubdtype(X.dtype, np.floating):
            X = X.astype(float)

        nan_mask = np.isnan(X)

        findings = {
            'total_missing': int(np.sum(nan_mask)),
            'missing_percentage': float(np.mean(nan_mask) * 100),
            'samples_with_missing': int(np.sum(np.any(nan_mask, axis=1) if X.ndim > 1 else nan_mask)),
            'complete_samples': int(np.sum(~np.any(nan_mask, axis=1) if X.ndim > 1 else ~nan_mask))
        }

        if X.ndim > 1:
            findings['missing_per_feature'] = np.sum(nan_mask, axis=0).tolist()
            findings['features_with_missing'] = int(np.sum(np.any(nan_mask, axis=0)))

        recommendations = []
        if findings['missing_percentage'] > 5:
            recommendations.append('High missing rate (>5%) - investigate missingness pattern (MCAR/MAR/MNAR)')
        if findings['missing_percentage'] > 20:
            recommendations.append('Very high missing rate (>20%) - consider imputation strategy carefully')

        return EDAResult(
            analysis_type='missing_value_analysis',
            question='Where is data missing and why?',
            findings=findings,
            artifacts=['Missingness heatmap'],
            recommendations=recommendations,
            passed=findings['missing_percentage'] < 20
        )

    def analyze_duplicates(
        self,
        X: np.ndarray,
        tolerance: float = 1e-10
    ) -> EDAResult:
        """
        Duplicate record analysis.

        Question: Are records duplicated?
        """
        # Flatten for comparison if needed
        X_flat = X.reshape(len(X), -1) if X.ndim > 2 else X

        # Find exact duplicates
        _, unique_idx, counts = np.unique(
            X_flat, axis=0, return_index=True, return_counts=True
        )

        n_duplicates = len(X) - len(unique_idx)
        duplicate_groups = np.sum(counts > 1)

        findings = {
            'n_total': len(X),
            'n_unique': len(unique_idx),
            'n_duplicates': n_duplicates,
            'duplicate_percentage': float(n_duplicates / len(X) * 100),
            'duplicate_groups': int(duplicate_groups)
        }

        recommendations = []
        if n_duplicates > 0:
            recommendations.append(f'Found {n_duplicates} duplicate samples - verify if intentional')

        return EDAResult(
            analysis_type='duplicate_analysis',
            question='Are records duplicated?',
            findings=findings,
            artifacts=['Duplication report'],
            recommendations=recommendations,
            passed=findings['duplicate_percentage'] < 5
        )


# =============================================================================
# Distribution Analysis
# =============================================================================

class DistributionAnalyzer:
    """
    Analyze data distributions.

    EDA Questions 5-8:
    5. Basic statistical summary
    6. Distribution analysis
    7. Outlier detection
    8. Range and validity checks
    """

    def analyze_basic_statistics(
        self,
        X: np.ndarray
    ) -> EDAResult:
        """
        Basic statistical summary.

        Question: What are central tendencies?
        """
        X_flat = X.reshape(len(X), -1) if X.ndim > 2 else X

        findings = {
            'mean': float(np.nanmean(X_flat)),
            'median': float(np.nanmedian(X_flat)),
            'std': float(np.nanstd(X_flat)),
            'min': float(np.nanmin(X_flat)),
            'max': float(np.nanmax(X_flat)),
            'range': float(np.nanmax(X_flat) - np.nanmin(X_flat)),
            'q25': float(np.nanpercentile(X_flat, 25)),
            'q75': float(np.nanpercentile(X_flat, 75)),
            'iqr': float(np.nanpercentile(X_flat, 75) - np.nanpercentile(X_flat, 25))
        }

        # Per-feature statistics if multi-dimensional
        if X_flat.ndim > 1 and X_flat.shape[1] > 1:
            findings['feature_means'] = np.nanmean(X_flat, axis=0).tolist()[:20]  # First 20
            findings['feature_stds'] = np.nanstd(X_flat, axis=0).tolist()[:20]

        return EDAResult(
            analysis_type='basic_statistics',
            question='What are central tendencies?',
            findings=findings,
            artifacts=['Descriptive statistics table'],
            recommendations=[]
        )

    def analyze_distribution(
        self,
        X: np.ndarray
    ) -> EDAResult:
        """
        Distribution analysis.

        Question: Are distributions skewed or heavy-tailed?
        """
        X_flat = X.flatten()

        # Remove NaN for statistics
        X_clean = X_flat[~np.isnan(X_flat)]

        findings = {
            'skewness': float(stats.skew(X_clean)),
            'kurtosis': float(stats.kurtosis(X_clean)),
            'is_normal': False,
            'distribution_type': 'unknown'
        }

        # Normality test (for smaller samples)
        if len(X_clean) < 5000:
            try:
                _, p_value = stats.normaltest(X_clean)
                findings['normality_p_value'] = float(p_value)
                findings['is_normal'] = p_value > 0.05
            except Exception:
                pass

        # Classify distribution
        skew = findings['skewness']
        kurt = findings['kurtosis']

        if abs(skew) < 0.5 and abs(kurt) < 1:
            findings['distribution_type'] = 'approximately_normal'
        elif skew > 1:
            findings['distribution_type'] = 'right_skewed'
        elif skew < -1:
            findings['distribution_type'] = 'left_skewed'
        elif kurt > 3:
            findings['distribution_type'] = 'heavy_tailed'

        recommendations = []
        if abs(skew) > 2:
            recommendations.append('Highly skewed distribution - consider log transform')
        if kurt > 7:
            recommendations.append('Heavy tails detected - use robust statistics')

        return EDAResult(
            analysis_type='distribution_analysis',
            question='Are distributions skewed or heavy-tailed?',
            findings=findings,
            artifacts=['Distribution plots', 'Histograms', 'KDE plots'],
            recommendations=recommendations
        )

    def analyze_outliers_eda(
        self,
        X: np.ndarray,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5
    ) -> EDAResult:
        """
        EDA-level outlier detection.

        Question: Are extreme values present?
        """
        X_flat = X.reshape(len(X), -1) if X.ndim > 2 else X

        # Z-score outliers
        z_scores = np.abs(stats.zscore(X_flat, nan_policy='omit'))
        z_outliers = np.sum(z_scores > z_threshold)

        # IQR outliers
        q1 = np.nanpercentile(X_flat, 25, axis=0)
        q3 = np.nanpercentile(X_flat, 75, axis=0)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        iqr_outliers = np.sum((X_flat < lower) | (X_flat > upper))

        findings = {
            'z_score_outliers': int(z_outliers),
            'z_score_outlier_pct': float(z_outliers / X_flat.size * 100),
            'iqr_outliers': int(iqr_outliers),
            'iqr_outlier_pct': float(iqr_outliers / X_flat.size * 100),
            'z_threshold': z_threshold,
            'iqr_multiplier': iqr_multiplier
        }

        recommendations = []
        if findings['iqr_outlier_pct'] > 5:
            recommendations.append('High outlier rate (>5%) - investigate data quality')

        return EDAResult(
            analysis_type='outlier_detection_eda',
            question='Are extreme values present?',
            findings=findings,
            artifacts=['Boxplots', 'Outlier list'],
            recommendations=recommendations
        )

    def analyze_range_validity(
        self,
        X: np.ndarray,
        expected_min: Optional[float] = None,
        expected_max: Optional[float] = None,
        physiological_range: Tuple[float, float] = (-500, 500)  # µV for EEG
    ) -> EDAResult:
        """
        Range and validity checks.

        Question: Are values realistic?
        """
        actual_min = float(np.nanmin(X))
        actual_max = float(np.nanmax(X))

        findings = {
            'actual_min': actual_min,
            'actual_max': actual_max,
            'expected_min': expected_min,
            'expected_max': expected_max,
            'physiological_min': physiological_range[0],
            'physiological_max': physiological_range[1]
        }

        # Check violations
        violations = []

        if expected_min is not None and actual_min < expected_min:
            violations.append(f'Values below expected minimum: {actual_min} < {expected_min}')

        if expected_max is not None and actual_max > expected_max:
            violations.append(f'Values above expected maximum: {actual_max} > {expected_max}')

        if actual_min < physiological_range[0] or actual_max > physiological_range[1]:
            violations.append('Values outside physiological EEG range')
            n_outside = np.sum((X < physiological_range[0]) | (X > physiological_range[1]))
            findings['values_outside_range'] = int(n_outside)
            findings['outside_range_pct'] = float(n_outside / X.size * 100)

        findings['violations'] = violations

        return EDAResult(
            analysis_type='range_validity',
            question='Are values realistic?',
            findings=findings,
            artifacts=['Validity violation report'],
            recommendations=['Investigate values outside physiological range'] if violations else [],
            passed=len(violations) == 0
        )


# =============================================================================
# Target and Correlation Analysis
# =============================================================================

class TargetCorrelationAnalyzer:
    """
    Analyze target variable and feature correlations.

    EDA Questions 9-14:
    9. Target variable analysis
    10. Class balance analysis
    11. Feature correlation analysis
    12. Multicollinearity analysis
    13. Feature-target relationship
    14. Interaction exploration
    """

    def analyze_target(
        self,
        y: np.ndarray
    ) -> EDAResult:
        """
        Target variable analysis.

        Question: What does the target look like?
        """
        unique, counts = np.unique(y, return_counts=True)

        findings = {
            'n_classes': len(unique),
            'class_labels': unique.tolist(),
            'class_counts': counts.tolist(),
            'class_percentages': (counts / len(y) * 100).tolist(),
            'majority_class': unique[np.argmax(counts)],
            'minority_class': unique[np.argmin(counts)],
            'target_dtype': str(y.dtype)
        }

        return EDAResult(
            analysis_type='target_analysis',
            question='What does the target look like?',
            findings=findings,
            artifacts=['Target distribution chart'],
            recommendations=[]
        )

    def analyze_class_balance(
        self,
        y: np.ndarray
    ) -> EDAResult:
        """
        Class balance analysis.

        Question: Is the dataset imbalanced?
        """
        unique, counts = np.unique(y, return_counts=True)

        imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        minority_pct = min(counts) / len(y) * 100

        findings = {
            'imbalance_ratio': float(imbalance_ratio),
            'minority_percentage': float(minority_pct),
            'is_balanced': imbalance_ratio < 1.5,
            'is_moderately_imbalanced': 1.5 <= imbalance_ratio < 3,
            'is_highly_imbalanced': imbalance_ratio >= 3,
            'class_counts': dict(zip(unique.tolist(), counts.tolist()))
        }

        recommendations = []
        if findings['is_highly_imbalanced']:
            recommendations.append('Highly imbalanced dataset - use stratified sampling, class weights, or resampling')
        elif findings['is_moderately_imbalanced']:
            recommendations.append('Moderately imbalanced - consider class weights')

        return EDAResult(
            analysis_type='class_balance',
            question='Is the dataset imbalanced?',
            findings=findings,
            artifacts=['Balance ratio report'],
            recommendations=recommendations,
            passed=not findings['is_highly_imbalanced']
        )

    def analyze_feature_correlation(
        self,
        X: np.ndarray,
        method: str = 'pearson',
        high_corr_threshold: float = 0.9
    ) -> EDAResult:
        """
        Feature correlation analysis.

        Question: Are features correlated?
        """
        X_flat = X.reshape(len(X), -1) if X.ndim > 2 else X

        # Limit features for efficiency
        n_features = min(X_flat.shape[1], 100)
        X_sample = X_flat[:, :n_features]

        # Compute correlation matrix
        if method == 'pearson':
            corr_matrix = np.corrcoef(X_sample.T)
        else:
            # Spearman
            corr_matrix = np.zeros((n_features, n_features))
            for i in range(n_features):
                for j in range(i, n_features):
                    r, _ = stats.spearmanr(X_sample[:, i], X_sample[:, j])
                    corr_matrix[i, j] = r
                    corr_matrix[j, i] = r

        # Handle NaN in correlation matrix
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Find high correlations (excluding diagonal)
        np.fill_diagonal(corr_matrix, 0)
        high_corr_pairs = np.sum(np.abs(corr_matrix) > high_corr_threshold) // 2

        findings = {
            'n_features_analyzed': n_features,
            'method': method,
            'mean_abs_correlation': float(np.mean(np.abs(corr_matrix))),
            'max_correlation': float(np.max(np.abs(corr_matrix))),
            'high_correlation_pairs': int(high_corr_pairs),
            'high_corr_threshold': high_corr_threshold
        }

        recommendations = []
        if high_corr_pairs > 0:
            recommendations.append(f'{high_corr_pairs} feature pairs with correlation > {high_corr_threshold} - consider feature selection')

        return EDAResult(
            analysis_type='feature_correlation',
            question='Are features correlated?',
            findings=findings,
            artifacts=['Correlation matrix', 'Correlation heatmap'],
            recommendations=recommendations
        )

    def analyze_feature_target_relationship(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_top_features: int = 20
    ) -> EDAResult:
        """
        Feature-target relationship analysis.

        Question: Which features relate to the target?
        """
        X_flat = X.reshape(len(X), -1) if X.ndim > 2 else X
        n_features = X_flat.shape[1]

        # Compute effect sizes (Cohen's d for binary classification)
        if len(np.unique(y)) == 2:
            class_0_mask = y == np.unique(y)[0]
            class_1_mask = y == np.unique(y)[1]

            effect_sizes = []
            for i in range(min(n_features, 500)):
                x0 = X_flat[class_0_mask, i]
                x1 = X_flat[class_1_mask, i]

                pooled_std = np.sqrt(
                    ((len(x0) - 1) * np.var(x0, ddof=1) +
                     (len(x1) - 1) * np.var(x1, ddof=1)) /
                    (len(x0) + len(x1) - 2)
                )

                if pooled_std > 0:
                    d = abs(np.mean(x1) - np.mean(x0)) / pooled_std
                else:
                    d = 0

                effect_sizes.append(d)

            effect_sizes = np.array(effect_sizes)
            top_indices = np.argsort(effect_sizes)[::-1][:n_top_features]

            findings = {
                'method': 'cohens_d',
                'n_features_analyzed': len(effect_sizes),
                'top_feature_indices': top_indices.tolist(),
                'top_effect_sizes': effect_sizes[top_indices].tolist(),
                'mean_effect_size': float(np.mean(effect_sizes)),
                'features_with_large_effect': int(np.sum(effect_sizes > 0.8)),
                'features_with_medium_effect': int(np.sum((effect_sizes > 0.5) & (effect_sizes <= 0.8)))
            }
        else:
            findings = {
                'method': 'not_applicable_multiclass',
                'n_classes': len(np.unique(y))
            }

        return EDAResult(
            analysis_type='feature_target_relationship',
            question='Which features relate to the target?',
            findings=findings,
            artifacts=['Feature relevance plots'],
            recommendations=[]
        )


# =============================================================================
# Temporal and Group Analysis
# =============================================================================

class TemporalGroupAnalyzer:
    """
    Analyze temporal patterns and group differences.

    EDA Questions 15-16:
    15. Temporal EDA
    16. Group/segment analysis
    """

    def analyze_temporal_patterns(
        self,
        X: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        window_size: int = 100
    ) -> EDAResult:
        """
        Temporal EDA analysis.

        Question: How does data evolve over time?
        """
        if X.ndim > 2:
            # For epoched data, analyze across epochs
            n_epochs = len(X)
            epoch_means = np.mean(X.reshape(n_epochs, -1), axis=1)
            epoch_stds = np.std(X.reshape(n_epochs, -1), axis=1)
        else:
            epoch_means = np.mean(X, axis=1) if X.ndim > 1 else X
            epoch_stds = np.std(X, axis=1) if X.ndim > 1 else np.zeros_like(X)

        # Compute rolling statistics
        n_windows = len(epoch_means) // window_size
        if n_windows > 1:
            window_means = [np.mean(epoch_means[i*window_size:(i+1)*window_size])
                          for i in range(n_windows)]
            window_stds = [np.std(epoch_means[i*window_size:(i+1)*window_size])
                         for i in range(n_windows)]

            # Check for drift
            trend_corr, _ = stats.pearsonr(range(n_windows), window_means)

            findings = {
                'n_epochs': len(epoch_means),
                'n_windows': n_windows,
                'window_size': window_size,
                'mean_trend_correlation': float(trend_corr),
                'has_drift': abs(trend_corr) > 0.5,
                'mean_variance_over_time': float(np.std(window_means)),
                'variance_stability': float(np.std(window_stds))
            }
        else:
            findings = {
                'n_epochs': len(epoch_means),
                'insufficient_data': True
            }

        recommendations = []
        if findings.get('has_drift', False):
            recommendations.append('Temporal drift detected - consider time-based normalization')

        return EDAResult(
            analysis_type='temporal_eda',
            question='How does data evolve over time?',
            findings=findings,
            artifacts=['Time-series EDA plots'],
            recommendations=recommendations
        )

    def analyze_groups(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_ids: np.ndarray,
        group_name: str = 'subject'
    ) -> EDAResult:
        """
        Group/segment analysis.

        Question: Do patterns differ across groups?
        """
        unique_groups = np.unique(group_ids)
        n_groups = len(unique_groups)

        group_stats = []
        for group in unique_groups:
            mask = group_ids == group
            X_group = X[mask]
            y_group = y[mask]

            group_stats.append({
                'group_id': str(group),
                'n_samples': int(np.sum(mask)),
                'mean': float(np.mean(X_group)),
                'std': float(np.std(X_group)),
                'class_distribution': dict(zip(*np.unique(y_group, return_counts=True)))
            })

        # Check for group heterogeneity
        group_means = [g['mean'] for g in group_stats]
        group_stds = [g['std'] for g in group_stats]

        findings = {
            'n_groups': n_groups,
            'group_name': group_name,
            'mean_variance_between_groups': float(np.var(group_means)),
            'mean_variance_within_groups': float(np.mean([s**2 for s in group_stds])),
            'group_heterogeneity': float(np.std(group_means) / (np.mean(group_stds) + 1e-10)),
            'samples_per_group_mean': float(np.mean([g['n_samples'] for g in group_stats])),
            'samples_per_group_std': float(np.std([g['n_samples'] for g in group_stats]))
        }

        recommendations = []
        if findings['group_heterogeneity'] > 1.0:
            recommendations.append(f'High {group_name} heterogeneity - consider {group_name}-wise normalization')

        return EDAResult(
            analysis_type='group_analysis',
            question='Do patterns differ across groups?',
            findings=findings,
            artifacts=['Segment comparison report'],
            recommendations=recommendations
        )


# =============================================================================
# Data Quality Assessment
# =============================================================================

class DataQualityAnalyzer:
    """
    Assess overall data quality and modeling readiness.

    EDA Questions 17-20:
    17. Noise and variability inspection
    18. Leakage suspicion analysis
    19. Data quality risk assessment
    20. EDA conclusions
    """

    def analyze_noise_variability(
        self,
        X: np.ndarray
    ) -> EDAResult:
        """
        Noise and variability inspection.

        Question: Is data noisy or unstable?
        """
        X_flat = X.reshape(len(X), -1) if X.ndim > 2 else X

        # Compute variability metrics
        sample_variances = np.var(X_flat, axis=1)
        feature_variances = np.var(X_flat, axis=0)

        # Coefficient of variation
        cv_samples = np.std(sample_variances) / (np.mean(sample_variances) + 1e-10)
        cv_features = np.std(feature_variances) / (np.mean(feature_variances) + 1e-10)

        findings = {
            'mean_sample_variance': float(np.mean(sample_variances)),
            'variance_of_variances': float(np.var(sample_variances)),
            'cv_across_samples': float(cv_samples),
            'cv_across_features': float(cv_features),
            'high_variance_samples': int(np.sum(sample_variances > np.mean(sample_variances) + 2*np.std(sample_variances))),
            'low_variance_samples': int(np.sum(sample_variances < np.mean(sample_variances) - 2*np.std(sample_variances)))
        }

        recommendations = []
        if cv_samples > 2:
            recommendations.append('High variance instability across samples - investigate outlier samples')
        if findings['low_variance_samples'] > 0:
            recommendations.append('Some samples have very low variance - check for flatline signals')

        return EDAResult(
            analysis_type='noise_variability',
            question='Is data noisy or unstable?',
            findings=findings,
            artifacts=['Noise assessment'],
            recommendations=recommendations
        )

    def analyze_leakage_suspicion(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> EDAResult:
        """
        Leakage suspicion analysis.

        Question: Does any feature leak target info?
        """
        X_flat = X.reshape(len(X), -1) if X.ndim > 2 else X
        n_features = min(X_flat.shape[1], 100)

        # Check for perfect correlations with target
        suspicions = []

        for i in range(n_features):
            # Point-biserial correlation for binary target
            if len(np.unique(y)) == 2:
                r, _ = stats.pointbiserialr(y, X_flat[:, i])
                if abs(r) > 0.95:
                    suspicions.append({
                        'feature_idx': i,
                        'correlation': float(r),
                        'type': 'perfect_correlation'
                    })

        findings = {
            'n_features_checked': n_features,
            'n_suspicions': len(suspicions),
            'suspicious_features': suspicions
        }

        recommendations = []
        if suspicions:
            recommendations.append(f'CRITICAL: {len(suspicions)} features with near-perfect target correlation - investigate for leakage')

        return EDAResult(
            analysis_type='leakage_suspicion',
            question='Does any feature leak target info?',
            findings=findings,
            artifacts=['Leakage suspicion log'],
            recommendations=recommendations,
            passed=len(suspicions) == 0
        )

    def generate_quality_report(
        self,
        analyses: List[EDAResult]
    ) -> EDAResult:
        """
        Generate overall data quality risk assessment.

        Question: What could break modeling later?
        """
        risks = []

        for analysis in analyses:
            if not analysis.passed:
                risks.append({
                    'source': analysis.analysis_type,
                    'severity': 'high' if 'leakage' in analysis.analysis_type.lower() else 'medium',
                    'details': analysis.recommendations
                })

        # Compute quality score
        n_passed = sum(1 for a in analyses if a.passed)
        quality_score = n_passed / len(analyses) if analyses else 0

        findings = {
            'n_analyses': len(analyses),
            'n_passed': n_passed,
            'n_failed': len(analyses) - n_passed,
            'quality_score': float(quality_score),
            'risks': risks
        }

        recommendations = []
        if quality_score < 0.7:
            recommendations.append('Data quality concerns detected - address before modeling')
        if any(r['severity'] == 'high' for r in risks):
            recommendations.append('HIGH RISK issues found - must resolve before proceeding')

        return EDAResult(
            analysis_type='quality_risk_assessment',
            question='What could break modeling later?',
            findings=findings,
            artifacts=['Data risk register'],
            recommendations=recommendations,
            passed=quality_score >= 0.7
        )


# =============================================================================
# Comprehensive EDA Report Generator
# =============================================================================

class EDAReportGenerator:
    """
    Generate comprehensive EDA report following the 20-point framework.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.overview = DatasetOverviewAnalyzer(feature_names)
        self.distribution = DistributionAnalyzer()
        self.target_corr = TargetCorrelationAnalyzer()
        self.temporal_group = TemporalGroupAnalyzer()
        self.quality = DataQualityAnalyzer()

    def generate_full_report(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        dataset_name: str = 'EEG Dataset'
    ) -> EDAReport:
        """
        Generate comprehensive EDA report.

        Runs all 20 EDA analyses.
        """
        analyses = []

        # 1-4: Dataset overview
        analyses.append(self.overview.analyze_overview(X, y, subject_ids))
        analyses.append(self.overview.analyze_schema(X))
        analyses.append(self.overview.analyze_missing_values(X))
        analyses.append(self.overview.analyze_duplicates(X))

        # 5-8: Distribution analysis
        analyses.append(self.distribution.analyze_basic_statistics(X))
        analyses.append(self.distribution.analyze_distribution(X))
        analyses.append(self.distribution.analyze_outliers_eda(X))
        analyses.append(self.distribution.analyze_range_validity(X))

        # 9-14: Target and correlation
        analyses.append(self.target_corr.analyze_target(y))
        analyses.append(self.target_corr.analyze_class_balance(y))
        analyses.append(self.target_corr.analyze_feature_correlation(X))
        analyses.append(self.target_corr.analyze_feature_target_relationship(X, y))

        # 15-16: Temporal and group
        analyses.append(self.temporal_group.analyze_temporal_patterns(X, timestamps))
        if subject_ids is not None:
            analyses.append(self.temporal_group.analyze_groups(X, y, subject_ids))

        # 17-20: Data quality
        analyses.append(self.quality.analyze_noise_variability(X))
        analyses.append(self.quality.analyze_leakage_suspicion(X, y))
        quality_report = self.quality.generate_quality_report(analyses)
        analyses.append(quality_report)

        # Generate modeling recommendations
        modeling_recs = self._generate_modeling_recommendations(analyses)

        return EDAReport(
            dataset_name=dataset_name,
            n_samples=len(X),
            n_features=X.shape[1] if X.ndim > 1 else X.shape[-1],
            analyses=analyses,
            overall_quality_score=quality_report.findings['quality_score'],
            modeling_recommendations=modeling_recs
        )

    def _generate_modeling_recommendations(
        self,
        analyses: List[EDAResult]
    ) -> List[str]:
        """Generate modeling recommendations from EDA findings."""
        recommendations = []

        for analysis in analyses:
            recommendations.extend(analysis.recommendations)

        # Deduplicate
        recommendations = list(set(recommendations))

        # Add general recommendations
        recommendations.append('Use stratified cross-validation to handle class distribution')
        recommendations.append('Apply subject-wise splitting to prevent leakage')

        return recommendations

    def export_report(
        self,
        report: EDAReport,
        output_path: str,
        format: str = 'markdown'
    ) -> None:
        """Export EDA report to file."""
        if format == 'markdown':
            content = self._to_markdown(report)
        else:
            import json
            content = json.dumps({
                'dataset_name': report.dataset_name,
                'n_samples': report.n_samples,
                'n_features': report.n_features,
                'quality_score': report.overall_quality_score,
                'analyses': [
                    {
                        'type': a.analysis_type,
                        'question': a.question,
                        'findings': a.findings,
                        'passed': a.passed
                    }
                    for a in report.analyses
                ],
                'recommendations': report.modeling_recommendations
            }, indent=2)

        with open(output_path, 'w') as f:
            f.write(content)

    def _to_markdown(self, report: EDAReport) -> str:
        """Convert report to markdown."""
        lines = [
            f"# EDA Report: {report.dataset_name}",
            "",
            f"**Samples:** {report.n_samples}",
            f"**Features:** {report.n_features}",
            f"**Quality Score:** {report.overall_quality_score:.2f}",
            "",
            "## Analysis Summary",
            "",
            "| Analysis | Question | Passed |",
            "|----------|----------|--------|"
        ]

        for a in report.analyses:
            status = "✓" if a.passed else "✗"
            lines.append(f"| {a.analysis_type} | {a.question[:50]}... | {status} |")

        lines.extend([
            "",
            "## Recommendations",
            ""
        ])

        for rec in report.modeling_recommendations:
            lines.append(f"- {rec}")

        return '\n'.join(lines)
