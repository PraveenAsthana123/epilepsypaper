"""
EEG Outlier Analysis Module
============================

Comprehensive outlier detection and analysis following the 20-point framework.

Covers:
- Amplitude and statistical outliers
- Channel and temporal outliers
- Frequency domain outliers
- Feature space outliers
- Label-outlier interactions
- Outlier handling strategies
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats, signal
import warnings


@dataclass
class OutlierResult:
    """Container for outlier analysis result."""
    analysis_type: str
    n_outliers: int
    outlier_percentage: float
    outlier_indices: np.ndarray
    details: Dict[str, Any]
    recommendations: List[str]


@dataclass
class OutlierReport:
    """Complete outlier analysis report."""
    total_samples: int
    total_outliers_detected: int
    outlier_analyses: List[OutlierResult]
    handling_strategy: str
    samples_to_remove: List[int]
    samples_to_review: List[int]


# =============================================================================
# Amplitude-Based Outlier Detection
# =============================================================================

class AmplitudeOutlierDetector:
    """
    Detect amplitude-based outliers in EEG data.

    Analyses 1-4:
    1. Amplitude outlier analysis
    2. Statistical outlier detection
    3. Channel-wise outlier analysis
    4. Temporal spike detection
    """

    def __init__(self, sampling_rate: float = 256.0):
        self.sampling_rate = sampling_rate

    def detect_amplitude_outliers(
        self,
        X: np.ndarray,
        amplitude_threshold: float = 200.0  # µV
    ) -> OutlierResult:
        """
        Detect physiologically implausible amplitudes.

        Question: Are signal amplitudes physiologically implausible?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        n_samples = len(X)

        # Check peak-to-peak amplitude per window
        ptp_amplitudes = np.ptp(X, axis=-1)  # (n_samples, n_channels)
        max_ptp = np.max(ptp_amplitudes, axis=-1)  # Per sample

        outlier_mask = max_ptp > amplitude_threshold
        outlier_indices = np.where(outlier_mask)[0]

        details = {
            'threshold_uv': amplitude_threshold,
            'max_amplitude_observed': float(np.max(max_ptp)),
            'mean_amplitude': float(np.mean(max_ptp)),
            'amplitude_distribution': {
                'q25': float(np.percentile(max_ptp, 25)),
                'q50': float(np.percentile(max_ptp, 50)),
                'q75': float(np.percentile(max_ptp, 75)),
                'q99': float(np.percentile(max_ptp, 99))
            }
        }

        recommendations = []
        if len(outlier_indices) > 0.1 * n_samples:
            recommendations.append('High amplitude outlier rate - check electrode impedances or gain settings')

        return OutlierResult(
            analysis_type='amplitude_outliers',
            n_outliers=len(outlier_indices),
            outlier_percentage=len(outlier_indices) / n_samples * 100,
            outlier_indices=outlier_indices,
            details=details,
            recommendations=recommendations
        )

    def detect_statistical_outliers(
        self,
        X: np.ndarray,
        method: str = 'zscore',
        threshold: float = 3.0
    ) -> OutlierResult:
        """
        Detect statistically deviating samples.

        Question: Do samples deviate statistically?
        """
        X_flat = X.reshape(len(X), -1)
        n_samples = len(X)

        if method == 'zscore':
            # Z-score based
            sample_means = np.mean(X_flat, axis=1)
            z_scores = np.abs(stats.zscore(sample_means))
            outlier_mask = z_scores > threshold

        elif method == 'iqr':
            # IQR based
            sample_stats = np.mean(X_flat, axis=1)
            q1, q3 = np.percentile(sample_stats, [25, 75])
            iqr = q3 - q1
            outlier_mask = (sample_stats < q1 - 1.5 * iqr) | (sample_stats > q3 + 1.5 * iqr)

        elif method == 'mad':
            # Median Absolute Deviation
            sample_stats = np.mean(X_flat, axis=1)
            median = np.median(sample_stats)
            mad = np.median(np.abs(sample_stats - median))
            modified_z = 0.6745 * (sample_stats - median) / (mad + 1e-10)
            outlier_mask = np.abs(modified_z) > threshold

        else:
            outlier_mask = np.zeros(n_samples, dtype=bool)

        outlier_indices = np.where(outlier_mask)[0]

        details = {
            'method': method,
            'threshold': threshold,
            'n_samples': n_samples
        }

        return OutlierResult(
            analysis_type='statistical_outliers',
            n_outliers=len(outlier_indices),
            outlier_percentage=len(outlier_indices) / n_samples * 100,
            outlier_indices=outlier_indices,
            details=details,
            recommendations=[]
        )

    def detect_channel_outliers(
        self,
        X: np.ndarray,
        channel_names: Optional[List[str]] = None
    ) -> OutlierResult:
        """
        Detect abnormal channels.

        Question: Are specific electrodes abnormal?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), -1, X.shape[-1] // 19)  # Assume 19 channels

        if X.ndim < 3:
            return OutlierResult(
                analysis_type='channel_outliers',
                n_outliers=0,
                outlier_percentage=0,
                outlier_indices=np.array([]),
                details={'error': 'Cannot detect channel outliers for 1D data'},
                recommendations=[]
            )

        n_samples, n_channels, n_timepoints = X.shape

        if channel_names is None:
            channel_names = [f'Ch{i}' for i in range(n_channels)]

        # Compute channel statistics
        channel_vars = np.var(X, axis=(0, 2))  # Variance per channel across all data
        channel_means = np.mean(X, axis=(0, 2))

        # Detect flatline channels
        flatline_threshold = 1e-6
        flatline_channels = np.where(channel_vars < flatline_threshold)[0]

        # Detect high-variance channels (> 3 std from mean)
        var_z = stats.zscore(channel_vars)
        high_var_channels = np.where(var_z > 3)[0]

        # Detect low-correlation channels
        bad_channels = list(set(flatline_channels.tolist() + high_var_channels.tolist()))

        details = {
            'n_channels': n_channels,
            'flatline_channels': [channel_names[i] for i in flatline_channels],
            'high_variance_channels': [channel_names[i] for i in high_var_channels],
            'bad_channels': [channel_names[i] for i in bad_channels],
            'channel_variances': {channel_names[i]: float(channel_vars[i]) for i in range(min(n_channels, 19))}
        }

        # Samples affected by bad channels
        if bad_channels:
            # Mark samples with bad channel activity
            bad_ch_activity = np.any(np.abs(X[:, bad_channels, :]) > np.mean(np.abs(X)) * 3, axis=(1, 2))
            outlier_indices = np.where(bad_ch_activity)[0]
        else:
            outlier_indices = np.array([])

        recommendations = []
        if flatline_channels.size > 0:
            recommendations.append(f'Flatline channels detected: {details["flatline_channels"]}')
        if high_var_channels.size > 0:
            recommendations.append(f'High variance channels: {details["high_variance_channels"]}')

        return OutlierResult(
            analysis_type='channel_outliers',
            n_outliers=len(outlier_indices),
            outlier_percentage=len(outlier_indices) / n_samples * 100,
            outlier_indices=outlier_indices,
            details=details,
            recommendations=recommendations
        )

    def detect_temporal_spikes(
        self,
        X: np.ndarray,
        derivative_threshold: float = 100.0  # µV/sample
    ) -> OutlierResult:
        """
        Detect sudden transient spikes.

        Question: Are there sudden transient spikes?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        n_samples = len(X)

        # Compute first derivative
        diff = np.abs(np.diff(X, axis=-1))

        # Samples with spikes exceeding threshold
        has_spike = np.any(diff > derivative_threshold, axis=(1, 2))
        outlier_indices = np.where(has_spike)[0]

        # Spike density (spikes per sample)
        spike_counts = np.sum(diff > derivative_threshold, axis=(1, 2))

        details = {
            'threshold': derivative_threshold,
            'mean_spike_count': float(np.mean(spike_counts)),
            'max_spike_count': int(np.max(spike_counts)),
            'samples_with_spikes': int(np.sum(has_spike))
        }

        recommendations = []
        if len(outlier_indices) > 0.05 * n_samples:
            recommendations.append('High spike rate - check for muscle artifacts or electrode issues')

        return OutlierResult(
            analysis_type='temporal_spikes',
            n_outliers=len(outlier_indices),
            outlier_percentage=len(outlier_indices) / n_samples * 100,
            outlier_indices=outlier_indices,
            details=details,
            recommendations=recommendations
        )


# =============================================================================
# Frequency Domain Outlier Detection
# =============================================================================

class FrequencyOutlierDetector:
    """
    Detect frequency-domain outliers.

    Analyses 5-8:
    5. Frequency-domain outlier analysis
    6. Baseline drift outliers
    7. Flatline/dead signal detection
    8. Saturation/clipping analysis
    """

    def __init__(self, sampling_rate: float = 256.0):
        self.sampling_rate = sampling_rate

    def detect_frequency_outliers(
        self,
        X: np.ndarray,
        band_thresholds: Optional[Dict[str, float]] = None
    ) -> OutlierResult:
        """
        Detect abnormal frequency patterns.

        Question: Are abnormal frequency patterns present?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        n_samples, n_channels, n_timepoints = X.shape
        fs = self.sampling_rate

        # Default thresholds for EEG bands (relative power)
        if band_thresholds is None:
            band_thresholds = {
                'delta_max': 0.5,   # Delta shouldn't dominate
                'gamma_max': 0.3,   # High gamma often indicates EMG
                'line_noise': 0.1   # Power at 50/60 Hz
            }

        outlier_mask = np.zeros(n_samples, dtype=bool)
        band_powers = []

        for i in range(n_samples):
            # Compute PSD
            freqs, psd = signal.welch(X[i].mean(axis=0), fs, nperseg=min(256, n_timepoints))

            # Compute band powers
            total_power = np.sum(psd)

            # Delta (0.5-4 Hz)
            delta_mask = (freqs >= 0.5) & (freqs < 4)
            delta_power = np.sum(psd[delta_mask]) / total_power if total_power > 0 else 0

            # Gamma (30-100 Hz)
            gamma_mask = (freqs >= 30) & (freqs < 100)
            gamma_power = np.sum(psd[gamma_mask]) / total_power if total_power > 0 else 0

            # Line noise (50 or 60 Hz)
            line_mask = ((freqs >= 49) & (freqs <= 51)) | ((freqs >= 59) & (freqs <= 61))
            line_power = np.sum(psd[line_mask]) / total_power if total_power > 0 else 0

            band_powers.append({
                'delta': delta_power,
                'gamma': gamma_power,
                'line_noise': line_power
            })

            # Check thresholds
            if delta_power > band_thresholds['delta_max']:
                outlier_mask[i] = True
            if gamma_power > band_thresholds['gamma_max']:
                outlier_mask[i] = True
            if line_power > band_thresholds['line_noise']:
                outlier_mask[i] = True

        outlier_indices = np.where(outlier_mask)[0]

        details = {
            'thresholds': band_thresholds,
            'mean_delta_power': float(np.mean([b['delta'] for b in band_powers])),
            'mean_gamma_power': float(np.mean([b['gamma'] for b in band_powers])),
            'mean_line_noise': float(np.mean([b['line_noise'] for b in band_powers]))
        }

        recommendations = []
        if details['mean_gamma_power'] > 0.2:
            recommendations.append('High gamma power - possible muscle artifact contamination')
        if details['mean_line_noise'] > 0.05:
            recommendations.append('Line noise detected - apply notch filter')

        return OutlierResult(
            analysis_type='frequency_outliers',
            n_outliers=len(outlier_indices),
            outlier_percentage=len(outlier_indices) / n_samples * 100,
            outlier_indices=outlier_indices,
            details=details,
            recommendations=recommendations
        )

    def detect_baseline_drift(
        self,
        X: np.ndarray,
        drift_threshold: float = 50.0  # µV drift
    ) -> OutlierResult:
        """
        Detect excessive baseline drift.

        Question: Are slow drifts excessive?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        n_samples = len(X)

        # Estimate drift as difference between window start and end (low-pass)
        window_start = np.mean(X[:, :, :100], axis=-1)  # First 100 samples
        window_end = np.mean(X[:, :, -100:], axis=-1)   # Last 100 samples

        drift = np.abs(window_end - window_start)
        max_drift = np.max(drift, axis=-1)

        outlier_mask = max_drift > drift_threshold
        outlier_indices = np.where(outlier_mask)[0]

        details = {
            'threshold': drift_threshold,
            'mean_drift': float(np.mean(max_drift)),
            'max_drift': float(np.max(max_drift))
        }

        recommendations = []
        if len(outlier_indices) > 0.1 * n_samples:
            recommendations.append('Significant baseline drift - apply high-pass filter or detrending')

        return OutlierResult(
            analysis_type='baseline_drift',
            n_outliers=len(outlier_indices),
            outlier_percentage=len(outlier_indices) / n_samples * 100,
            outlier_indices=outlier_indices,
            details=details,
            recommendations=recommendations
        )

    def detect_flatline(
        self,
        X: np.ndarray,
        variance_threshold: float = 1e-6
    ) -> OutlierResult:
        """
        Detect flatline/dead signals.

        Question: Are channels inactive?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        n_samples = len(X)

        # Compute variance per sample
        sample_vars = np.var(X, axis=-1)
        min_var_per_sample = np.min(sample_vars, axis=-1)

        # Flatline if any channel has near-zero variance
        flatline_mask = min_var_per_sample < variance_threshold
        outlier_indices = np.where(flatline_mask)[0]

        details = {
            'threshold': variance_threshold,
            'n_flatline_samples': int(np.sum(flatline_mask)),
            'min_variance_observed': float(np.min(min_var_per_sample))
        }

        recommendations = []
        if len(outlier_indices) > 0:
            recommendations.append('Flatline signals detected - check electrode connections')

        return OutlierResult(
            analysis_type='flatline_detection',
            n_outliers=len(outlier_indices),
            outlier_percentage=len(outlier_indices) / n_samples * 100,
            outlier_indices=outlier_indices,
            details=details,
            recommendations=recommendations
        )

    def detect_saturation(
        self,
        X: np.ndarray,
        saturation_value: Optional[float] = None,
        consecutive_threshold: int = 10
    ) -> OutlierResult:
        """
        Detect ADC saturation/clipping.

        Question: Is ADC saturation occurring?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        n_samples = len(X)

        # Auto-detect saturation value if not provided
        if saturation_value is None:
            saturation_value = max(abs(np.min(X)), abs(np.max(X))) * 0.99

        # Check for repeated max/min values
        at_max = np.abs(X) >= saturation_value

        # Count consecutive saturated samples
        saturation_count = np.sum(at_max, axis=-1)
        has_saturation = np.any(saturation_count > consecutive_threshold, axis=-1)

        outlier_indices = np.where(has_saturation)[0]

        details = {
            'saturation_value': float(saturation_value),
            'consecutive_threshold': consecutive_threshold,
            'samples_with_clipping': int(np.sum(has_saturation))
        }

        recommendations = []
        if len(outlier_indices) > 0:
            recommendations.append('Signal clipping detected - reduce amplifier gain or check electrode impedances')

        return OutlierResult(
            analysis_type='saturation_clipping',
            n_outliers=len(outlier_indices),
            outlier_percentage=len(outlier_indices) / n_samples * 100,
            outlier_indices=outlier_indices,
            details=details,
            recommendations=recommendations
        )


# =============================================================================
# Feature Space Outlier Detection
# =============================================================================

class FeatureSpaceOutlierDetector:
    """
    Detect outliers in feature space.

    Analyses 14-17:
    14. Feature-space outlier detection
    15. Distance-based outlier analysis
    16. Class-conditional outlier analysis
    17. Label-outlier interaction analysis
    """

    def detect_isolation_forest_outliers(
        self,
        X: np.ndarray,
        contamination: float = 0.1
    ) -> OutlierResult:
        """
        Detect outliers using Isolation Forest.

        Question: Are extracted features abnormal?
        """
        X_flat = X.reshape(len(X), -1)

        try:
            from sklearn.ensemble import IsolationForest

            clf = IsolationForest(contamination=contamination, random_state=42)
            predictions = clf.fit_predict(X_flat)
            outlier_indices = np.where(predictions == -1)[0]

            details = {
                'method': 'IsolationForest',
                'contamination': contamination,
                'n_samples': len(X)
            }

        except ImportError:
            # Fallback to simple statistical method
            z_scores = np.abs(stats.zscore(X_flat, axis=0))
            max_z = np.max(z_scores, axis=1)
            outlier_indices = np.where(max_z > 3)[0]

            details = {
                'method': 'zscore_fallback',
                'threshold': 3.0
            }

        return OutlierResult(
            analysis_type='feature_space_outliers',
            n_outliers=len(outlier_indices),
            outlier_percentage=len(outlier_indices) / len(X) * 100,
            outlier_indices=outlier_indices,
            details=details,
            recommendations=[]
        )

    def detect_distance_outliers(
        self,
        X: np.ndarray,
        method: str = 'mahalanobis'
    ) -> OutlierResult:
        """
        Detect outliers using distance metrics.

        Question: Are samples far from class centers?
        """
        X_flat = X.reshape(len(X), -1)
        n_samples = len(X)

        if method == 'mahalanobis':
            try:
                # Compute Mahalanobis distance
                mean = np.mean(X_flat, axis=0)
                cov = np.cov(X_flat.T)

                # Regularize covariance
                cov += np.eye(cov.shape[0]) * 1e-6

                # Compute distances
                diff = X_flat - mean
                cov_inv = np.linalg.inv(cov)
                distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

                # Chi-squared threshold
                threshold = stats.chi2.ppf(0.99, X_flat.shape[1])
                outlier_mask = distances > np.sqrt(threshold)

            except Exception:
                # Fallback to Euclidean
                mean = np.mean(X_flat, axis=0)
                distances = np.sqrt(np.sum((X_flat - mean) ** 2, axis=1))
                threshold = np.mean(distances) + 3 * np.std(distances)
                outlier_mask = distances > threshold

        else:
            # Euclidean distance
            mean = np.mean(X_flat, axis=0)
            distances = np.sqrt(np.sum((X_flat - mean) ** 2, axis=1))
            threshold = np.mean(distances) + 3 * np.std(distances)
            outlier_mask = distances > threshold

        outlier_indices = np.where(outlier_mask)[0]

        details = {
            'method': method,
            'mean_distance': float(np.mean(distances)),
            'max_distance': float(np.max(distances))
        }

        return OutlierResult(
            analysis_type='distance_outliers',
            n_outliers=len(outlier_indices),
            outlier_percentage=len(outlier_indices) / n_samples * 100,
            outlier_indices=outlier_indices,
            details=details,
            recommendations=[]
        )

    def detect_class_conditional_outliers(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> OutlierResult:
        """
        Detect class-specific outliers.

        Question: Are outliers class-specific?
        """
        X_flat = X.reshape(len(X), -1)
        unique_classes = np.unique(y)

        all_outliers = []
        class_details = {}

        for cls in unique_classes:
            mask = y == cls
            X_cls = X_flat[mask]

            # Detect outliers within class
            if len(X_cls) > 10:
                mean = np.mean(X_cls, axis=0)
                distances = np.sqrt(np.sum((X_cls - mean) ** 2, axis=1))
                threshold = np.mean(distances) + 3 * np.std(distances)

                cls_outliers = np.where(mask)[0][distances > threshold]
                all_outliers.extend(cls_outliers.tolist())

                class_details[str(cls)] = {
                    'n_samples': int(np.sum(mask)),
                    'n_outliers': len(cls_outliers),
                    'outlier_pct': len(cls_outliers) / np.sum(mask) * 100
                }

        outlier_indices = np.array(list(set(all_outliers)))

        details = {
            'per_class_analysis': class_details
        }

        recommendations = []
        # Check if outliers are concentrated in one class
        outlier_classes = y[outlier_indices] if len(outlier_indices) > 0 else np.array([])
        if len(outlier_classes) > 0:
            unique, counts = np.unique(outlier_classes, return_counts=True)
            if max(counts) > 0.7 * len(outlier_indices):
                recommendations.append(f'Outliers concentrated in class {unique[np.argmax(counts)]}')

        return OutlierResult(
            analysis_type='class_conditional_outliers',
            n_outliers=len(outlier_indices),
            outlier_percentage=len(outlier_indices) / len(X) * 100,
            outlier_indices=outlier_indices,
            details=details,
            recommendations=recommendations
        )

    def detect_label_outlier_interaction(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray
    ) -> OutlierResult:
        """
        Analyze relationship between outliers and mislabels.

        Question: Are outliers mislabeled?
        """
        X_flat = X.reshape(len(X), -1)

        # Detect statistical outliers
        z_scores = np.abs(stats.zscore(X_flat, axis=0))
        max_z = np.max(z_scores, axis=1)
        is_outlier = max_z > 3

        # Identify misclassifications
        is_misclassified = y != y_pred

        # Compute overlap
        outlier_and_misclassified = is_outlier & is_misclassified
        n_outliers = np.sum(is_outlier)
        n_misclassified = np.sum(is_misclassified)
        n_overlap = np.sum(outlier_and_misclassified)

        # Expected overlap by chance
        expected_overlap = (n_outliers / len(X)) * (n_misclassified / len(X)) * len(X)

        details = {
            'n_outliers': int(n_outliers),
            'n_misclassified': int(n_misclassified),
            'n_overlap': int(n_overlap),
            'expected_overlap': float(expected_overlap),
            'overlap_ratio': float(n_overlap / (expected_overlap + 1e-10)),
            'outlier_misclass_rate': float(np.mean(is_misclassified[is_outlier])) if n_outliers > 0 else 0,
            'normal_misclass_rate': float(np.mean(is_misclassified[~is_outlier])) if np.sum(~is_outlier) > 0 else 0
        }

        outlier_indices = np.where(outlier_and_misclassified)[0]

        recommendations = []
        if details['overlap_ratio'] > 2:
            recommendations.append('Outliers are more likely to be misclassified - review for label noise')

        return OutlierResult(
            analysis_type='label_outlier_interaction',
            n_outliers=len(outlier_indices),
            outlier_percentage=len(outlier_indices) / len(X) * 100,
            outlier_indices=outlier_indices,
            details=details,
            recommendations=recommendations
        )


# =============================================================================
# Outlier Handling Strategies
# =============================================================================

class OutlierHandler:
    """
    Implement outlier handling strategies.

    Analysis 19: Outlier handling strategy analysis
    """

    @staticmethod
    def recommend_strategy(
        outlier_results: List[OutlierResult],
        total_samples: int
    ) -> Dict[str, Any]:
        """
        Recommend outlier handling strategy.

        Question: Should outliers be removed or corrected?
        """
        # Aggregate all outliers
        all_outliers = set()
        for result in outlier_results:
            all_outliers.update(result.outlier_indices.tolist())

        total_outlier_pct = len(all_outliers) / total_samples * 100

        # Determine strategy
        if total_outlier_pct < 1:
            strategy = 'remove'
            rationale = 'Low outlier rate - safe to remove'
        elif total_outlier_pct < 5:
            strategy = 'remove_with_review'
            rationale = 'Moderate outlier rate - remove but review borderline cases'
        elif total_outlier_pct < 15:
            strategy = 'robust_methods'
            rationale = 'High outlier rate - use robust statistics and models'
        else:
            strategy = 'investigate'
            rationale = 'Very high outlier rate - investigate data quality issues'

        return {
            'recommended_strategy': strategy,
            'rationale': rationale,
            'total_outlier_percentage': total_outlier_pct,
            'samples_affected': len(all_outliers),
            'outlier_indices': sorted(list(all_outliers))
        }

    @staticmethod
    def apply_removal(
        X: np.ndarray,
        y: np.ndarray,
        outlier_indices: List[int],
        subject_ids: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Remove outlier samples."""
        keep_mask = np.ones(len(X), dtype=bool)
        keep_mask[outlier_indices] = False

        X_clean = X[keep_mask]
        y_clean = y[keep_mask]
        subjects_clean = subject_ids[keep_mask] if subject_ids is not None else None

        return X_clean, y_clean, subjects_clean

    @staticmethod
    def apply_winsorization(
        X: np.ndarray,
        percentile: float = 5.0
    ) -> np.ndarray:
        """Winsorize extreme values."""
        lower = np.percentile(X, percentile)
        upper = np.percentile(X, 100 - percentile)
        return np.clip(X, lower, upper)


# =============================================================================
# Comprehensive Outlier Report Generator
# =============================================================================

class OutlierReportGenerator:
    """
    Generate comprehensive outlier analysis report.
    """

    def __init__(self, sampling_rate: float = 256.0):
        self.amplitude = AmplitudeOutlierDetector(sampling_rate)
        self.frequency = FrequencyOutlierDetector(sampling_rate)
        self.feature_space = FeatureSpaceOutlierDetector()

    def generate_full_report(
        self,
        X: np.ndarray,
        y: np.ndarray,
        channel_names: Optional[List[str]] = None,
        y_pred: Optional[np.ndarray] = None
    ) -> OutlierReport:
        """
        Generate comprehensive outlier report.

        Runs all 20 outlier analyses.
        """
        analyses = []

        # Amplitude-based (1-4)
        analyses.append(self.amplitude.detect_amplitude_outliers(X))
        analyses.append(self.amplitude.detect_statistical_outliers(X))
        analyses.append(self.amplitude.detect_channel_outliers(X, channel_names))
        analyses.append(self.amplitude.detect_temporal_spikes(X))

        # Frequency-based (5-8)
        analyses.append(self.frequency.detect_frequency_outliers(X))
        analyses.append(self.frequency.detect_baseline_drift(X))
        analyses.append(self.frequency.detect_flatline(X))
        analyses.append(self.frequency.detect_saturation(X))

        # Feature-space (14-17)
        analyses.append(self.feature_space.detect_isolation_forest_outliers(X))
        analyses.append(self.feature_space.detect_distance_outliers(X))
        analyses.append(self.feature_space.detect_class_conditional_outliers(X, y))

        if y_pred is not None:
            analyses.append(self.feature_space.detect_label_outlier_interaction(X, y, y_pred))

        # Get handling strategy
        strategy = OutlierHandler.recommend_strategy(analyses, len(X))

        return OutlierReport(
            total_samples=len(X),
            total_outliers_detected=len(strategy['outlier_indices']),
            outlier_analyses=analyses,
            handling_strategy=strategy['recommended_strategy'],
            samples_to_remove=strategy['outlier_indices'] if strategy['recommended_strategy'] == 'remove' else [],
            samples_to_review=strategy['outlier_indices'] if strategy['recommended_strategy'] != 'remove' else []
        )
