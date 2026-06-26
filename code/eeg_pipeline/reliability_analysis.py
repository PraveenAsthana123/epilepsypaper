"""
Reliability Analysis Module
============================

Comprehensive reliability and robustness analysis for EEG models.

Covers:
- Test-retest reliability
- Intraclass correlation
- Signal quality impact analysis
- Noise robustness testing
- Missing data handling
- Hardware invariance testing
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats, signal
import warnings


@dataclass
class ReliabilityResult:
    """Container for reliability analysis result."""
    metric_name: str
    value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    interpretation: str = ""


@dataclass
class RobustnessResult:
    """Container for robustness test result."""
    test_name: str
    baseline_performance: float
    perturbed_performance: float
    degradation: float
    passed: bool
    threshold: float


# =============================================================================
# Test-Retest Reliability
# =============================================================================

class TestRetestAnalyzer:
    """
    Analyze test-retest reliability of model predictions.

    Measures consistency of model outputs across repeated measurements
    of the same subjects.
    """

    def __init__(self, model):
        """
        Initialize test-retest analyzer.

        Parameters
        ----------
        model : sklearn-compatible model
            Trained model to analyze
        """
        self.model = model

    def analyze_reliability(
        self,
        X_session1: np.ndarray,
        X_session2: np.ndarray,
        subject_ids: np.ndarray
    ) -> Dict[str, ReliabilityResult]:
        """
        Analyze reliability between two sessions.

        Parameters
        ----------
        X_session1 : array
            Features from first session
        X_session2 : array
            Features from second session (same subjects)
        subject_ids : array
            Subject IDs (should be same for both)

        Returns
        -------
        Dictionary of reliability metrics
        """
        results = {}

        # Get predictions
        y_prob1 = self._get_probabilities(X_session1)
        y_prob2 = self._get_probabilities(X_session2)

        if y_prob1 is None or y_prob2 is None:
            warnings.warn("Model does not support probability predictions")
            y_pred1 = self.model.predict(X_session1)
            y_pred2 = self.model.predict(X_session2)

            # Agreement for binary predictions
            agreement = np.mean(y_pred1 == y_pred2)
            results['prediction_agreement'] = ReliabilityResult(
                'prediction_agreement',
                agreement,
                interpretation=self._interpret_agreement(agreement)
            )
            return results

        # Aggregate by subject (mean probability per subject)
        unique_subjects = np.unique(subject_ids)
        subj_prob1 = []
        subj_prob2 = []

        for subj in unique_subjects:
            mask = subject_ids == subj
            subj_prob1.append(np.mean(y_prob1[mask]))
            subj_prob2.append(np.mean(y_prob2[mask]))

        subj_prob1 = np.array(subj_prob1)
        subj_prob2 = np.array(subj_prob2)

        # Pearson correlation
        r, p_value = stats.pearsonr(subj_prob1, subj_prob2)
        results['pearson_r'] = ReliabilityResult(
            'pearson_r',
            r,
            interpretation=self._interpret_correlation(r)
        )

        # ICC
        icc_value, icc_ci = self._compute_icc(subj_prob1, subj_prob2)
        results['icc'] = ReliabilityResult(
            'icc',
            icc_value,
            icc_ci[0],
            icc_ci[1],
            interpretation=self._interpret_icc(icc_value)
        )

        # Bland-Altman analysis
        ba_results = self._bland_altman(subj_prob1, subj_prob2)
        results['mean_difference'] = ReliabilityResult(
            'mean_difference',
            ba_results['mean_diff'],
            interpretation=f"Systematic bias: {ba_results['mean_diff']:.3f}"
        )

        results['limits_of_agreement'] = ReliabilityResult(
            'limits_of_agreement',
            ba_results['loa_range'],
            interpretation=f"95% LoA: [{ba_results['loa_lower']:.3f}, {ba_results['loa_upper']:.3f}]"
        )

        return results

    def _get_probabilities(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Get probability predictions if available."""
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)
            if probs.ndim > 1:
                return probs[:, 1]
            return probs
        return None

    def _compute_icc(
        self,
        session1: np.ndarray,
        session2: np.ndarray
    ) -> Tuple[float, Tuple[float, float]]:
        """Compute ICC(3,1) for two sessions."""
        ratings = np.column_stack([session1, session2])
        n, k = ratings.shape

        # Grand mean
        grand_mean = np.mean(ratings)

        # Subject means
        subject_means = np.mean(ratings, axis=1)

        # Rater means
        rater_means = np.mean(ratings, axis=0)

        # Sum of squares
        ss_total = np.sum((ratings - grand_mean) ** 2)
        ss_between = k * np.sum((subject_means - grand_mean) ** 2)
        ss_raters = n * np.sum((rater_means - grand_mean) ** 2)
        ss_residual = ss_total - ss_between - ss_raters

        # Mean squares
        ms_between = ss_between / (n - 1)
        ms_residual = ss_residual / ((n - 1) * (k - 1)) if k > 1 else ss_residual / (n - 1)

        # ICC(3,1)
        icc = (ms_between - ms_residual) / (ms_between + (k - 1) * ms_residual)

        # Confidence interval
        f_value = ms_between / ms_residual if ms_residual > 0 else 0
        df1, df2 = n - 1, (n - 1) * (k - 1)

        try:
            f_lower = f_value / stats.f.ppf(0.975, df1, df2)
            f_upper = f_value * stats.f.ppf(0.975, df2, df1)
            ci_lower = (f_lower - 1) / (f_lower + k - 1)
            ci_upper = (f_upper - 1) / (f_upper + k - 1)
        except Exception:
            ci_lower, ci_upper = 0.0, 1.0

        return float(icc), (float(ci_lower), float(ci_upper))

    def _bland_altman(
        self,
        session1: np.ndarray,
        session2: np.ndarray
    ) -> Dict[str, float]:
        """Bland-Altman analysis."""
        diff = session1 - session2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)

        return {
            'mean_diff': float(mean_diff),
            'std_diff': float(std_diff),
            'loa_lower': float(mean_diff - 1.96 * std_diff),
            'loa_upper': float(mean_diff + 1.96 * std_diff),
            'loa_range': float(3.92 * std_diff)
        }

    def _interpret_correlation(self, r: float) -> str:
        """Interpret Pearson correlation."""
        if r >= 0.9:
            return "Excellent reliability"
        elif r >= 0.75:
            return "Good reliability"
        elif r >= 0.5:
            return "Moderate reliability"
        else:
            return "Poor reliability"

    def _interpret_icc(self, icc: float) -> str:
        """Interpret ICC value."""
        if icc >= 0.9:
            return "Excellent reliability (ICC >= 0.9)"
        elif icc >= 0.75:
            return "Good reliability (0.75 <= ICC < 0.9)"
        elif icc >= 0.5:
            return "Moderate reliability (0.5 <= ICC < 0.75)"
        else:
            return "Poor reliability (ICC < 0.5)"

    def _interpret_agreement(self, agreement: float) -> str:
        """Interpret prediction agreement."""
        if agreement >= 0.9:
            return "Excellent agreement (>= 90%)"
        elif agreement >= 0.8:
            return "Good agreement (80-90%)"
        elif agreement >= 0.7:
            return "Moderate agreement (70-80%)"
        else:
            return "Poor agreement (< 70%)"


# =============================================================================
# Noise Robustness Testing
# =============================================================================

class NoiseRobustnessTester:
    """
    Test model robustness to various noise types.

    Tests:
    - Gaussian noise
    - EMG artifacts
    - Eye movement artifacts
    - Electrode drift
    - Powerline interference
    """

    def __init__(self, model, sampling_rate: float = 256.0):
        """
        Initialize noise robustness tester.

        Parameters
        ----------
        model : sklearn-compatible model
            Trained model to test
        sampling_rate : float
            EEG sampling rate in Hz
        """
        self.model = model
        self.sampling_rate = sampling_rate

    def run_all_tests(
        self,
        X: np.ndarray,
        y: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None
    ) -> List[RobustnessResult]:
        """
        Run all robustness tests.

        Parameters
        ----------
        X : array
            Test features
        y : array
            Test labels
        thresholds : dict, optional
            Performance degradation thresholds

        Returns
        -------
        List of RobustnessResult
        """
        if thresholds is None:
            thresholds = {
                'gaussian_noise': 0.1,
                'emg_artifact': 0.15,
                'eye_movement': 0.15,
                'electrode_drift': 0.1,
                'powerline': 0.05
            }

        results = []

        # Baseline performance
        baseline_acc = self._compute_accuracy(X, y)

        # Test each noise type
        noise_tests = [
            ('gaussian_noise', self.add_gaussian_noise, {'snr_db': 10}),
            ('emg_artifact', self.add_emg_artifact, {'intensity': 0.3}),
            ('eye_movement', self.add_eye_movement, {'amplitude': 50}),
            ('electrode_drift', self.add_electrode_drift, {'drift_rate': 0.1}),
            ('powerline', self.add_powerline_noise, {'amplitude': 20})
        ]

        for test_name, noise_func, params in noise_tests:
            X_noisy = noise_func(X.copy(), **params)
            noisy_acc = self._compute_accuracy(X_noisy, y)
            degradation = baseline_acc - noisy_acc

            threshold = thresholds.get(test_name, 0.1)

            results.append(RobustnessResult(
                test_name=test_name,
                baseline_performance=baseline_acc,
                perturbed_performance=noisy_acc,
                degradation=degradation,
                passed=degradation <= threshold,
                threshold=threshold
            ))

        return results

    def _compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute model accuracy."""
        y_pred = self.model.predict(X)
        return float(np.mean(y_pred == y))

    def add_gaussian_noise(
        self,
        X: np.ndarray,
        snr_db: float = 10
    ) -> np.ndarray:
        """
        Add Gaussian noise at specified SNR.

        Parameters
        ----------
        X : array
            EEG data
        snr_db : float
            Signal-to-noise ratio in dB

        Returns
        -------
        Noisy data
        """
        signal_power = np.mean(X ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(*X.shape) * np.sqrt(noise_power)
        return X + noise

    def add_emg_artifact(
        self,
        X: np.ndarray,
        intensity: float = 0.3
    ) -> np.ndarray:
        """
        Add simulated EMG (muscle) artifacts.

        Parameters
        ----------
        X : array
            EEG data
        intensity : float
            Artifact intensity (0-1)

        Returns
        -------
        Contaminated data
        """
        # EMG is high-frequency noise
        fs = self.sampling_rate
        nyq = fs / 2

        # Generate broadband noise
        emg = np.random.randn(*X.shape)

        # High-pass filter to simulate EMG (> 20 Hz)
        if nyq > 20:
            b, a = signal.butter(4, 20 / nyq, btype='high')
            if X.ndim == 3:
                for i in range(len(X)):
                    for j in range(X.shape[1]):
                        emg[i, j] = signal.filtfilt(b, a, emg[i, j])
            else:
                emg = signal.filtfilt(b, a, emg, axis=-1)

        # Scale and add
        emg_power = np.std(emg)
        signal_power = np.std(X)
        emg = emg * (intensity * signal_power / emg_power)

        return X + emg

    def add_eye_movement(
        self,
        X: np.ndarray,
        amplitude: float = 50
    ) -> np.ndarray:
        """
        Add simulated eye movement artifacts (blinks, saccades).

        Parameters
        ----------
        X : array
            EEG data
        amplitude : float
            Artifact amplitude in microvolts

        Returns
        -------
        Contaminated data
        """
        X_out = X.copy()

        if X.ndim == 3:
            n_epochs, n_channels, n_samples = X.shape
        else:
            n_epochs = 1
            X_out = X_out.reshape(1, *X.shape)
            n_channels, n_samples = X_out.shape[1:]

        # Add blink artifacts to ~10% of epochs
        n_blinks = max(1, int(0.1 * n_epochs))
        blink_epochs = np.random.choice(n_epochs, n_blinks, replace=False)

        for epoch_idx in blink_epochs:
            # Blink at random time
            blink_center = np.random.randint(100, n_samples - 100)
            blink_duration = int(0.2 * self.sampling_rate)  # 200ms blink

            # Gaussian envelope for blink
            t = np.arange(n_samples)
            blink_shape = amplitude * np.exp(-0.5 * ((t - blink_center) / (blink_duration / 4)) ** 2)

            # Add to frontal channels (first 4)
            for ch in range(min(4, n_channels)):
                X_out[epoch_idx, ch] += blink_shape

        if X.ndim < 3:
            return X_out.reshape(X.shape)
        return X_out

    def add_electrode_drift(
        self,
        X: np.ndarray,
        drift_rate: float = 0.1
    ) -> np.ndarray:
        """
        Add slow electrode drift artifact.

        Parameters
        ----------
        X : array
            EEG data
        drift_rate : float
            Drift rate per second

        Returns
        -------
        Contaminated data
        """
        if X.ndim == 3:
            n_epochs, n_channels, n_samples = X.shape
        else:
            n_samples = X.shape[-1]

        # Linear drift
        t = np.arange(n_samples) / self.sampling_rate
        drift = drift_rate * t * np.std(X)

        # Add random direction per channel
        drift_directions = np.random.choice([-1, 1], size=X.shape[:-1])

        if X.ndim == 3:
            return X + drift_directions[..., np.newaxis] * drift
        else:
            return X + drift_directions[..., np.newaxis] * drift

    def add_powerline_noise(
        self,
        X: np.ndarray,
        amplitude: float = 20,
        freq: float = 60.0
    ) -> np.ndarray:
        """
        Add powerline interference.

        Parameters
        ----------
        X : array
            EEG data
        amplitude : float
            Noise amplitude in microvolts
        freq : float
            Powerline frequency (50 or 60 Hz)

        Returns
        -------
        Contaminated data
        """
        n_samples = X.shape[-1]
        t = np.arange(n_samples) / self.sampling_rate

        # Sinusoidal powerline interference
        powerline = amplitude * np.sin(2 * np.pi * freq * t)

        return X + powerline


# =============================================================================
# Missing Channel Robustness
# =============================================================================

class MissingChannelTester:
    """
    Test model robustness to missing or bad channels.
    """

    def __init__(self, model, channel_names: Optional[List[str]] = None):
        """
        Initialize missing channel tester.

        Parameters
        ----------
        model : sklearn-compatible model
            Trained model to test
        channel_names : list, optional
            Names of EEG channels
        """
        self.model = model
        self.channel_names = channel_names

    def test_channel_dropout(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dropout_rates: List[float] = [0.05, 0.1, 0.2, 0.3]
    ) -> List[RobustnessResult]:
        """
        Test performance with random channel dropout.

        Parameters
        ----------
        X : array
            Test features (n_samples, n_channels, n_timepoints) or flattened
        y : array
            Test labels
        dropout_rates : list
            Proportion of channels to drop

        Returns
        -------
        List of RobustnessResult
        """
        results = []

        # Baseline
        baseline_acc = self._compute_accuracy(X, y)

        for rate in dropout_rates:
            X_dropped = self._dropout_channels(X, rate)
            dropped_acc = self._compute_accuracy(X_dropped, y)

            results.append(RobustnessResult(
                test_name=f'channel_dropout_{int(rate*100)}pct',
                baseline_performance=baseline_acc,
                perturbed_performance=dropped_acc,
                degradation=baseline_acc - dropped_acc,
                passed=(baseline_acc - dropped_acc) <= 0.15,
                threshold=0.15
            ))

        return results

    def test_specific_channel_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        critical_channels: Optional[List[int]] = None
    ) -> Dict[str, RobustnessResult]:
        """
        Test impact of losing specific channels.

        Parameters
        ----------
        X : array
            Test features
        y : array
            Test labels
        critical_channels : list, optional
            Indices of channels to test

        Returns
        -------
        Dictionary mapping channel to RobustnessResult
        """
        results = {}
        baseline_acc = self._compute_accuracy(X, y)

        if critical_channels is None:
            # Test all channels
            if X.ndim == 3:
                critical_channels = list(range(X.shape[1]))
            else:
                return results

        for ch_idx in critical_channels:
            X_dropped = self._drop_channel(X, ch_idx)
            dropped_acc = self._compute_accuracy(X_dropped, y)

            ch_name = self.channel_names[ch_idx] if self.channel_names else f'ch_{ch_idx}'

            results[ch_name] = RobustnessResult(
                test_name=f'drop_{ch_name}',
                baseline_performance=baseline_acc,
                perturbed_performance=dropped_acc,
                degradation=baseline_acc - dropped_acc,
                passed=(baseline_acc - dropped_acc) <= 0.1,
                threshold=0.1
            )

        return results

    def _compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute model accuracy."""
        # Flatten if needed for model
        if X.ndim > 2:
            X = X.reshape(len(X), -1)
        y_pred = self.model.predict(X)
        return float(np.mean(y_pred == y))

    def _dropout_channels(
        self,
        X: np.ndarray,
        rate: float
    ) -> np.ndarray:
        """Drop random channels by zeroing them."""
        X_out = X.copy()

        if X.ndim == 3:
            n_channels = X.shape[1]
            n_drop = int(rate * n_channels)
            drop_idx = np.random.choice(n_channels, n_drop, replace=False)
            X_out[:, drop_idx, :] = 0
        elif X.ndim == 2:
            # Assume flattened: try to identify channel structure
            n_features = X.shape[1]
            n_drop = int(rate * n_features)
            drop_idx = np.random.choice(n_features, n_drop, replace=False)
            X_out[:, drop_idx] = 0

        return X_out

    def _drop_channel(
        self,
        X: np.ndarray,
        ch_idx: int
    ) -> np.ndarray:
        """Drop specific channel by zeroing."""
        X_out = X.copy()

        if X.ndim == 3:
            X_out[:, ch_idx, :] = 0
        elif X.ndim == 2:
            # Assume channel data is contiguous
            n_timepoints = X.shape[1] // 19  # Assume 19 channels
            start = ch_idx * n_timepoints
            end = start + n_timepoints
            X_out[:, start:end] = 0

        return X_out


# =============================================================================
# Signal Quality Impact Analysis
# =============================================================================

class SignalQualityAnalyzer:
    """
    Analyze relationship between signal quality and model performance.
    """

    def __init__(self, model, sampling_rate: float = 256.0):
        """
        Initialize signal quality analyzer.

        Parameters
        ----------
        model : sklearn-compatible model
            Trained model to analyze
        sampling_rate : float
            EEG sampling rate
        """
        self.model = model
        self.sampling_rate = sampling_rate

    def compute_sqi(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Signal Quality Index for each sample.

        Parameters
        ----------
        X : array
            EEG data

        Returns
        -------
        Array of SQI values per sample
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        sqi_values = []

        for i in range(len(X)):
            sample = X[i]

            # Multiple SQI components
            scores = []

            # 1. Amplitude range score
            ptp = np.ptp(sample, axis=-1)
            amp_score = 1.0 - np.clip(np.mean(ptp) / 200.0, 0, 1)
            scores.append(amp_score)

            # 2. Variance stability
            var = np.var(sample, axis=-1)
            var_score = 1.0 - np.clip(np.std(var) / (np.mean(var) + 1e-10), 0, 1)
            scores.append(var_score)

            # 3. High-frequency noise ratio
            fs = self.sampling_rate
            nyq = fs / 2
            if nyq > 40:
                b, a = signal.butter(4, 40 / nyq, btype='high')
                hf = signal.filtfilt(b, a, sample, axis=-1)
                hf_power = np.mean(hf ** 2)
                total_power = np.mean(sample ** 2)
                hf_ratio = hf_power / (total_power + 1e-10)
                hf_score = 1.0 - np.clip(hf_ratio * 10, 0, 1)
                scores.append(hf_score)

            sqi_values.append(np.mean(scores))

        return np.array(sqi_values)

    def analyze_sqi_performance_relationship(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_bins: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze how performance varies with signal quality.

        Parameters
        ----------
        X : array
            Test features
        y : array
            Test labels
        n_bins : int
            Number of SQI bins

        Returns
        -------
        Dictionary with performance per SQI bin
        """
        sqi = self.compute_sqi(X)
        y_pred = self.model.predict(X if X.ndim == 2 else X.reshape(len(X), -1))

        # Bin by SQI
        bin_edges = np.percentile(sqi, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-10  # Include max value

        results = {
            'bins': [],
            'accuracies': [],
            'n_samples': [],
            'correlation': None
        }

        for i in range(n_bins):
            mask = (sqi >= bin_edges[i]) & (sqi < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(y_pred[mask] == y[mask])
                results['bins'].append(f'SQI {bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}')
                results['accuracies'].append(float(bin_acc))
                results['n_samples'].append(int(np.sum(mask)))

        # Correlation between SQI and correctness
        correct = (y_pred == y).astype(float)
        corr, p_value = stats.pointbiserialr(correct, sqi)
        results['correlation'] = {
            'r': float(corr),
            'p_value': float(p_value),
            'interpretation': 'Higher SQI associated with better performance' if corr > 0 else 'No clear relationship'
        }

        return results

    def identify_quality_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        min_accuracy: float = 0.8
    ) -> float:
        """
        Find minimum SQI threshold for acceptable performance.

        Parameters
        ----------
        X : array
            Test features
        y : array
            Test labels
        min_accuracy : float
            Minimum acceptable accuracy

        Returns
        -------
        SQI threshold
        """
        sqi = self.compute_sqi(X)
        y_pred = self.model.predict(X if X.ndim == 2 else X.reshape(len(X), -1))

        # Sort by SQI
        sorted_idx = np.argsort(sqi)
        sqi_sorted = sqi[sorted_idx]
        correct_sorted = (y_pred[sorted_idx] == y[sorted_idx])

        # Find threshold using cumulative accuracy from high SQI
        for i in range(len(sqi_sorted) - 10, 0, -1):
            acc = np.mean(correct_sorted[i:])
            if acc < min_accuracy:
                return float(sqi_sorted[i + 1])

        return float(sqi_sorted[0])


# =============================================================================
# Comprehensive Reliability Report
# =============================================================================

class ReliabilityReportGenerator:
    """
    Generate comprehensive reliability report.
    """

    def __init__(self, model, sampling_rate: float = 256.0):
        """
        Initialize report generator.

        Parameters
        ----------
        model : sklearn-compatible model
            Trained model
        sampling_rate : float
            EEG sampling rate
        """
        self.model = model
        self.noise_tester = NoiseRobustnessTester(model, sampling_rate)
        self.channel_tester = MissingChannelTester(model)
        self.sqi_analyzer = SignalQualityAnalyzer(model, sampling_rate)

    def generate_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_retest: Optional[np.ndarray] = None,
        subject_ids: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive reliability report.

        Parameters
        ----------
        X_test : array
            Test features
        y_test : array
            Test labels
        X_retest : array, optional
            Retest data for same subjects
        subject_ids : array, optional
            Subject IDs

        Returns
        -------
        Dictionary with all reliability metrics
        """
        report = {
            'noise_robustness': [],
            'channel_robustness': [],
            'signal_quality_analysis': None,
            'test_retest': None,
            'overall_reliability_score': None
        }

        # Noise robustness
        noise_results = self.noise_tester.run_all_tests(X_test, y_test)
        report['noise_robustness'] = [
            {
                'test': r.test_name,
                'baseline': r.baseline_performance,
                'perturbed': r.perturbed_performance,
                'degradation': r.degradation,
                'passed': r.passed
            }
            for r in noise_results
        ]

        # Channel robustness
        channel_results = self.channel_tester.test_channel_dropout(X_test, y_test)
        report['channel_robustness'] = [
            {
                'test': r.test_name,
                'degradation': r.degradation,
                'passed': r.passed
            }
            for r in channel_results
        ]

        # Signal quality analysis
        report['signal_quality_analysis'] = self.sqi_analyzer.analyze_sqi_performance_relationship(
            X_test, y_test
        )

        # Test-retest if available
        if X_retest is not None and subject_ids is not None:
            retest_analyzer = TestRetestAnalyzer(self.model)
            retest_results = retest_analyzer.analyze_reliability(
                X_test, X_retest, subject_ids
            )
            report['test_retest'] = {
                name: {'value': r.value, 'interpretation': r.interpretation}
                for name, r in retest_results.items()
            }

        # Overall reliability score
        noise_passed = sum(1 for r in noise_results if r.passed) / len(noise_results)
        channel_passed = sum(1 for r in channel_results if r.passed) / len(channel_results)

        report['overall_reliability_score'] = {
            'noise_robustness_score': noise_passed,
            'channel_robustness_score': channel_passed,
            'combined_score': (noise_passed + channel_passed) / 2,
            'interpretation': self._interpret_overall_score((noise_passed + channel_passed) / 2)
        }

        return report

    def _interpret_overall_score(self, score: float) -> str:
        """Interpret overall reliability score."""
        if score >= 0.9:
            return "Excellent reliability - model suitable for clinical deployment"
        elif score >= 0.75:
            return "Good reliability - model suitable for clinical research"
        elif score >= 0.5:
            return "Moderate reliability - additional validation recommended"
        else:
            return "Poor reliability - model requires significant improvement"
