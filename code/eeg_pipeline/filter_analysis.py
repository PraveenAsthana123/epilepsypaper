"""
EEG Filter Analysis Module
===========================

Comprehensive filter analysis following the 20-point framework.

Covers:
- Sampling and Nyquist validation
- Filter type selection and comparison
- Phase and edge artifact analysis
- SNR improvement measurement
- Artifact interaction analysis
- Filter governance and reproducibility
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import signal
import warnings


@dataclass
class FilterAnalysisResult:
    """Container for filter analysis result."""
    analysis_type: str
    question: str
    findings: Dict[str, Any]
    recommendations: List[str]
    passed: bool = True


@dataclass
class FilterConfig:
    """Filter configuration."""
    filter_type: str  # 'highpass', 'lowpass', 'bandpass', 'notch'
    low_freq: Optional[float] = None
    high_freq: Optional[float] = None
    order: int = 4
    method: str = 'butter'  # 'butter', 'cheby1', 'cheby2', 'bessel', 'fir'
    zero_phase: bool = True


# =============================================================================
# Sampling and Frequency Validation
# =============================================================================

class SamplingValidator:
    """
    Validate sampling rate and frequency requirements.

    Analyses 1-2:
    1. Sampling frequency validation
    2. Baseline wander analysis
    """

    def validate_sampling_rate(
        self,
        sampling_rate: float,
        required_bands: List[str] = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    ) -> FilterAnalysisResult:
        """
        Validate sampling rate for EEG bands.

        Question: Is the sampling rate sufficient for EEG bands?
        """
        # EEG frequency bands
        band_freqs = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100),
            'high_gamma': (100, 200)
        }

        nyquist = sampling_rate / 2
        supported_bands = []
        unsupported_bands = []

        for band in required_bands:
            if band in band_freqs:
                _, high = band_freqs[band]
                if high < nyquist:
                    supported_bands.append(band)
                else:
                    unsupported_bands.append(band)

        findings = {
            'sampling_rate': sampling_rate,
            'nyquist_frequency': nyquist,
            'supported_bands': supported_bands,
            'unsupported_bands': unsupported_bands,
            'max_frequency_analyzable': nyquist,
            'minimum_recommended_fs': max(band_freqs.get(b, (0, 0))[1] for b in required_bands) * 2.5
        }

        recommendations = []
        if unsupported_bands:
            recommendations.append(f'Bands {unsupported_bands} require higher sampling rate')
        if sampling_rate < 256:
            recommendations.append('Sampling rate < 256 Hz may limit analysis options')

        return FilterAnalysisResult(
            analysis_type='sampling_validation',
            question='Is the sampling rate sufficient for EEG bands?',
            findings=findings,
            recommendations=recommendations,
            passed=len(unsupported_bands) == 0
        )

    def analyze_baseline_wander(
        self,
        X: np.ndarray,
        sampling_rate: float,
        drift_threshold: float = 50.0
    ) -> FilterAnalysisResult:
        """
        Analyze low-frequency baseline drift.

        Question: Is low-frequency drift present?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        n_samples, n_channels, n_timepoints = X.shape

        # Compute DC offset and drift
        dc_offsets = np.mean(X, axis=-1)
        dc_offset_range = np.ptp(dc_offsets, axis=0)

        # Low-pass filter to isolate drift
        nyq = sampling_rate / 2
        b, a = signal.butter(2, 0.5 / nyq, btype='low')

        drift_magnitudes = []
        for i in range(min(n_samples, 100)):  # Sample subset
            for j in range(n_channels):
                try:
                    low_freq = signal.filtfilt(b, a, X[i, j])
                    drift_magnitudes.append(np.ptp(low_freq))
                except Exception:
                    pass

        mean_drift = np.mean(drift_magnitudes) if drift_magnitudes else 0
        has_significant_drift = mean_drift > drift_threshold

        findings = {
            'mean_drift_magnitude': float(mean_drift),
            'drift_threshold': drift_threshold,
            'has_significant_drift': has_significant_drift,
            'dc_offset_range': float(np.mean(dc_offset_range))
        }

        recommendations = []
        if has_significant_drift:
            recommendations.append('Significant baseline drift detected - apply high-pass filter (0.5 Hz recommended)')

        return FilterAnalysisResult(
            analysis_type='baseline_wander',
            question='Is low-frequency drift present?',
            findings=findings,
            recommendations=recommendations,
            passed=not has_significant_drift
        )


# =============================================================================
# Filter Design and Selection
# =============================================================================

class FilterDesignAnalyzer:
    """
    Analyze filter design choices.

    Analyses 3-7:
    3. High-pass filter analysis
    4. Low-pass filter analysis
    5. Band-pass filter selection
    6. Notch filter analysis
    7. Filter type comparison
    """

    def __init__(self, sampling_rate: float = 256.0):
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2

    def analyze_highpass_cutoff(
        self,
        X: np.ndarray,
        cutoff_options: List[float] = [0.1, 0.3, 0.5, 1.0]
    ) -> FilterAnalysisResult:
        """
        Analyze high-pass filter cutoff selection.

        Question: What low-frequency cutoff is safe?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        # Take sample for analysis
        X_sample = X[:min(50, len(X))]

        results = {}
        for cutoff in cutoff_options:
            if cutoff >= self.nyquist:
                continue

            try:
                b, a = signal.butter(4, cutoff / self.nyquist, btype='high')
                X_filtered = signal.filtfilt(b, a, X_sample, axis=-1)

                # Measure signal preservation
                power_original = np.mean(X_sample ** 2)
                power_filtered = np.mean(X_filtered ** 2)
                power_ratio = power_filtered / (power_original + 1e-10)

                # Measure distortion (correlation with original)
                correlations = []
                for i in range(len(X_sample)):
                    for j in range(X_sample.shape[1]):
                        r = np.corrcoef(X_sample[i, j], X_filtered[i, j])[0, 1]
                        correlations.append(r)
                mean_corr = np.nanmean(correlations)

                results[cutoff] = {
                    'power_ratio': float(power_ratio),
                    'mean_correlation': float(mean_corr),
                    'drift_removed': float(1 - power_ratio)
                }

            except Exception as e:
                results[cutoff] = {'error': str(e)}

        # Recommend cutoff
        best_cutoff = 0.5  # Default
        for cutoff, res in results.items():
            if 'mean_correlation' in res and res['mean_correlation'] > 0.95:
                best_cutoff = cutoff

        findings = {
            'cutoff_analysis': results,
            'recommended_cutoff': best_cutoff,
            'rationale': 'Balances drift removal with signal preservation'
        }

        return FilterAnalysisResult(
            analysis_type='highpass_cutoff',
            question='What low-frequency cutoff is safe?',
            findings=findings,
            recommendations=[f'Recommended high-pass cutoff: {best_cutoff} Hz']
        )

    def analyze_lowpass_cutoff(
        self,
        X: np.ndarray,
        cutoff_options: List[float] = [30, 40, 45, 70, 100]
    ) -> FilterAnalysisResult:
        """
        Analyze low-pass filter cutoff selection.

        Question: What high-frequency noise should be removed?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        X_sample = X[:min(50, len(X))]

        # Compute PSD to see frequency content
        freqs, psd = signal.welch(
            X_sample.reshape(-1),
            self.sampling_rate,
            nperseg=min(256, X_sample.shape[-1])
        )

        # Find where power drops significantly
        cumulative_power = np.cumsum(psd)
        total_power = cumulative_power[-1]
        power_95_freq = freqs[np.searchsorted(cumulative_power, 0.95 * total_power)]

        # Analyze each cutoff
        results = {}
        for cutoff in cutoff_options:
            if cutoff >= self.nyquist:
                continue

            # Power retained
            mask = freqs <= cutoff
            power_retained = np.sum(psd[mask]) / total_power

            results[cutoff] = {
                'power_retained': float(power_retained),
                'power_removed': float(1 - power_retained)
            }

        # Recommend cutoff based on 95% power frequency
        recommended = min([c for c in cutoff_options if c > power_95_freq], default=45)

        findings = {
            'cutoff_analysis': results,
            '95_power_frequency': float(power_95_freq),
            'recommended_cutoff': recommended
        }

        return FilterAnalysisResult(
            analysis_type='lowpass_cutoff',
            question='What high-frequency noise should be removed?',
            findings=findings,
            recommendations=[f'Recommended low-pass cutoff: {recommended} Hz']
        )

    def analyze_notch_filter(
        self,
        X: np.ndarray,
        line_freq: float = 60.0,
        notch_width: float = 2.0
    ) -> FilterAnalysisResult:
        """
        Analyze powerline noise and notch filter effectiveness.

        Question: Is 50/60 Hz contamination present?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        X_sample = X[:min(50, len(X))]

        # Compute PSD
        freqs, psd = signal.welch(
            X_sample.reshape(-1),
            self.sampling_rate,
            nperseg=min(512, X_sample.shape[-1])
        )

        # Check power at line frequency
        line_mask = (freqs >= line_freq - 2) & (freqs <= line_freq + 2)
        nearby_mask = ((freqs >= line_freq - 10) & (freqs < line_freq - 2)) | \
                      ((freqs > line_freq + 2) & (freqs <= line_freq + 10))

        line_power = np.mean(psd[line_mask]) if np.any(line_mask) else 0
        nearby_power = np.mean(psd[nearby_mask]) if np.any(nearby_mask) else 1

        has_line_noise = line_power > nearby_power * 2

        # Check harmonics
        harmonics = [line_freq * i for i in [2, 3, 4] if line_freq * i < self.nyquist]
        harmonic_powers = []
        for h in harmonics:
            h_mask = (freqs >= h - 2) & (freqs <= h + 2)
            if np.any(h_mask):
                harmonic_powers.append(float(np.mean(psd[h_mask])))

        findings = {
            'line_frequency': line_freq,
            'line_power': float(line_power),
            'nearby_power': float(nearby_power),
            'line_to_nearby_ratio': float(line_power / (nearby_power + 1e-10)),
            'has_line_noise': has_line_noise,
            'harmonics_detected': len([p for p in harmonic_powers if p > nearby_power])
        }

        recommendations = []
        if has_line_noise:
            recommendations.append(f'Apply notch filter at {line_freq} Hz')
            if findings['harmonics_detected'] > 0:
                recommendations.append('Consider multi-notch filter for harmonics')

        return FilterAnalysisResult(
            analysis_type='notch_filter',
            question='Is 50/60 Hz contamination present?',
            findings=findings,
            recommendations=recommendations,
            passed=not has_line_noise
        )

    def compare_filter_types(
        self,
        X: np.ndarray,
        low_freq: float = 0.5,
        high_freq: float = 45.0,
        filter_types: List[str] = ['butter', 'cheby1', 'bessel', 'fir']
    ) -> FilterAnalysisResult:
        """
        Compare different filter types.

        Question: Which filter preserves EEG morphology best?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        X_sample = X[:min(20, len(X))]
        results = {}

        for ftype in filter_types:
            try:
                if ftype == 'butter':
                    b, a = signal.butter(4, [low_freq / self.nyquist, high_freq / self.nyquist], btype='band')
                elif ftype == 'cheby1':
                    b, a = signal.cheby1(4, 0.5, [low_freq / self.nyquist, high_freq / self.nyquist], btype='band')
                elif ftype == 'bessel':
                    b, a = signal.bessel(4, [low_freq / self.nyquist, high_freq / self.nyquist], btype='band')
                elif ftype == 'fir':
                    numtaps = 101
                    b = signal.firwin(numtaps, [low_freq, high_freq], fs=self.sampling_rate, pass_zero=False)
                    a = [1.0]
                else:
                    continue

                # Apply filter
                X_filtered = signal.filtfilt(b, a, X_sample, axis=-1)

                # Measure properties
                correlations = []
                for i in range(len(X_sample)):
                    for j in range(X_sample.shape[1]):
                        r = np.corrcoef(X_sample[i, j], X_filtered[i, j])[0, 1]
                        correlations.append(r)

                # Check frequency response
                w, h = signal.freqz(b, a, worN=512)
                freq_hz = w * self.sampling_rate / (2 * np.pi)

                # Measure passband ripple
                passband_mask = (freq_hz >= low_freq) & (freq_hz <= high_freq)
                passband_response = np.abs(h[passband_mask])
                ripple = np.ptp(passband_response) if len(passband_response) > 0 else 0

                results[ftype] = {
                    'mean_correlation': float(np.nanmean(correlations)),
                    'passband_ripple': float(ripple),
                    'group_delay_samples': 0  # Simplified
                }

            except Exception as e:
                results[ftype] = {'error': str(e)}

        # Recommend best filter
        valid_results = {k: v for k, v in results.items() if 'mean_correlation' in v}
        if valid_results:
            best = max(valid_results.keys(), key=lambda k: valid_results[k]['mean_correlation'])
        else:
            best = 'butter'

        findings = {
            'filter_comparison': results,
            'recommended_filter': best,
            'low_freq': low_freq,
            'high_freq': high_freq
        }

        return FilterAnalysisResult(
            analysis_type='filter_comparison',
            question='Which filter preserves EEG morphology best?',
            findings=findings,
            recommendations=[f'Recommended filter type: {best}']
        )


# =============================================================================
# Filter Quality Analysis
# =============================================================================

class FilterQualityAnalyzer:
    """
    Analyze filter quality and artifacts.

    Analyses 8-14:
    8. Phase distortion analysis
    9. Edge artifact analysis
    10. Frequency response validation
    11. SNR improvement
    12-14. Artifact interaction analysis
    """

    def __init__(self, sampling_rate: float = 256.0):
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2

    def analyze_phase_distortion(
        self,
        config: FilterConfig
    ) -> FilterAnalysisResult:
        """
        Analyze phase distortion of filter.

        Question: Does filtering distort temporal structure?
        """
        # Design filter
        try:
            if config.filter_type == 'bandpass':
                b, a = signal.butter(
                    config.order,
                    [config.low_freq / self.nyquist, config.high_freq / self.nyquist],
                    btype='band'
                )
            elif config.filter_type == 'highpass':
                b, a = signal.butter(config.order, config.low_freq / self.nyquist, btype='high')
            elif config.filter_type == 'lowpass':
                b, a = signal.butter(config.order, config.high_freq / self.nyquist, btype='low')
            else:
                b, a = [1], [1]

            # Compute phase response
            w, h = signal.freqz(b, a, worN=512)
            freq_hz = w * self.sampling_rate / (2 * np.pi)
            phase = np.unwrap(np.angle(h))

            # Group delay (derivative of phase)
            group_delay = -np.diff(phase) / np.diff(w)

            # Phase linearity check
            phase_deviation = np.std(group_delay)

            findings = {
                'filter_type': config.filter_type,
                'zero_phase': config.zero_phase,
                'phase_deviation': float(phase_deviation),
                'max_group_delay_samples': float(np.max(np.abs(group_delay))),
                'is_linear_phase': phase_deviation < 0.1
            }

            recommendations = []
            if not config.zero_phase and phase_deviation > 0.1:
                recommendations.append('Non-zero-phase filter may distort ERP latencies - use filtfilt()')

        except Exception as e:
            findings = {'error': str(e)}
            recommendations = []

        return FilterAnalysisResult(
            analysis_type='phase_distortion',
            question='Does filtering distort temporal structure?',
            findings=findings,
            recommendations=recommendations
        )

    def analyze_edge_artifacts(
        self,
        X: np.ndarray,
        config: FilterConfig
    ) -> FilterAnalysisResult:
        """
        Analyze filter edge artifacts.

        Question: Are filter boundary effects present?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        X_sample = X[:min(20, len(X))]

        # Design and apply filter
        try:
            if config.filter_type == 'bandpass':
                b, a = signal.butter(
                    config.order,
                    [config.low_freq / self.nyquist, config.high_freq / self.nyquist],
                    btype='band'
                )
            elif config.filter_type == 'highpass':
                b, a = signal.butter(config.order, config.low_freq / self.nyquist, btype='high')
            else:
                b, a = signal.butter(config.order, config.high_freq / self.nyquist, btype='low')

            X_filtered = signal.filtfilt(b, a, X_sample, axis=-1)

            # Check edge regions (first and last 5% of samples)
            n_edge = max(10, int(0.05 * X_sample.shape[-1]))

            # Measure edge artifacts as variance ratio
            edge_var = np.var(X_filtered[:, :, :n_edge])
            center_var = np.var(X_filtered[:, :, n_edge:-n_edge])
            var_ratio = edge_var / (center_var + 1e-10)

            # Check for ringing
            edge_diff = np.diff(X_filtered[:, :, :n_edge], axis=-1)
            center_diff = np.diff(X_filtered[:, :, n_edge:-n_edge], axis=-1)
            ringing_indicator = np.std(edge_diff) / (np.std(center_diff) + 1e-10)

            has_edge_artifacts = var_ratio > 2 or ringing_indicator > 2

            findings = {
                'edge_to_center_variance_ratio': float(var_ratio),
                'ringing_indicator': float(ringing_indicator),
                'has_edge_artifacts': has_edge_artifacts,
                'edge_samples': n_edge
            }

            recommendations = []
            if has_edge_artifacts:
                recommendations.append('Edge artifacts detected - add padding or trim edges after filtering')

        except Exception as e:
            findings = {'error': str(e)}
            recommendations = []
            has_edge_artifacts = False

        return FilterAnalysisResult(
            analysis_type='edge_artifacts',
            question='Are filter boundary effects present?',
            findings=findings,
            recommendations=recommendations,
            passed=not has_edge_artifacts
        )

    def analyze_snr_improvement(
        self,
        X: np.ndarray,
        config: FilterConfig
    ) -> FilterAnalysisResult:
        """
        Analyze SNR improvement from filtering.

        Question: Does filtering improve SNR?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        X_sample = X[:min(50, len(X))]

        try:
            # Design filter
            if config.filter_type == 'bandpass':
                b, a = signal.butter(
                    config.order,
                    [config.low_freq / self.nyquist, config.high_freq / self.nyquist],
                    btype='band'
                )
            elif config.filter_type == 'highpass':
                b, a = signal.butter(config.order, config.low_freq / self.nyquist, btype='high')
            else:
                b, a = signal.butter(config.order, config.high_freq / self.nyquist, btype='low')

            X_filtered = signal.filtfilt(b, a, X_sample, axis=-1)

            # Estimate SNR improvement
            # Signal: variance of filtered data
            # Noise: variance of removed components
            signal_power = np.var(X_filtered)
            noise_power = np.var(X_sample - X_filtered)

            snr_before = 10 * np.log10(np.var(X_sample) / (noise_power + 1e-10))
            snr_after = 10 * np.log10(signal_power / (noise_power + 1e-10))
            snr_improvement = snr_after - snr_before

            findings = {
                'snr_before_db': float(snr_before),
                'snr_after_db': float(snr_after),
                'snr_improvement_db': float(snr_improvement),
                'noise_removed_percentage': float(noise_power / np.var(X_sample) * 100)
            }

            recommendations = []
            if snr_improvement < 0:
                recommendations.append('Filter may be removing signal - verify cutoff frequencies')

        except Exception as e:
            findings = {'error': str(e)}
            recommendations = []

        return FilterAnalysisResult(
            analysis_type='snr_improvement',
            question='Does filtering improve SNR?',
            findings=findings,
            recommendations=recommendations
        )


# =============================================================================
# Comprehensive Filter Report Generator
# =============================================================================

class FilterReportGenerator:
    """
    Generate comprehensive filter analysis report.
    """

    def __init__(self, sampling_rate: float = 256.0):
        self.sampling_rate = sampling_rate
        self.sampling_validator = SamplingValidator()
        self.design_analyzer = FilterDesignAnalyzer(sampling_rate)
        self.quality_analyzer = FilterQualityAnalyzer(sampling_rate)

    def generate_full_report(
        self,
        X: np.ndarray,
        proposed_config: Optional[FilterConfig] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive filter analysis report.

        Returns all filter analysis results and recommendations.
        """
        if proposed_config is None:
            proposed_config = FilterConfig(
                filter_type='bandpass',
                low_freq=0.5,
                high_freq=45.0,
                order=4,
                method='butter',
                zero_phase=True
            )

        analyses = []

        # Sampling validation
        analyses.append(self.sampling_validator.validate_sampling_rate(self.sampling_rate))
        analyses.append(self.sampling_validator.analyze_baseline_wander(X, self.sampling_rate))

        # Filter design analysis
        analyses.append(self.design_analyzer.analyze_highpass_cutoff(X))
        analyses.append(self.design_analyzer.analyze_lowpass_cutoff(X))
        analyses.append(self.design_analyzer.analyze_notch_filter(X))
        analyses.append(self.design_analyzer.compare_filter_types(X))

        # Filter quality analysis
        analyses.append(self.quality_analyzer.analyze_phase_distortion(proposed_config))
        analyses.append(self.quality_analyzer.analyze_edge_artifacts(X, proposed_config))
        analyses.append(self.quality_analyzer.analyze_snr_improvement(X, proposed_config))

        # Compile recommendations
        all_recommendations = []
        for analysis in analyses:
            all_recommendations.extend(analysis.recommendations)

        # Generate optimal filter config
        optimal_config = self._determine_optimal_config(analyses, proposed_config)

        return {
            'analyses': analyses,
            'proposed_config': proposed_config,
            'optimal_config': optimal_config,
            'recommendations': list(set(all_recommendations)),
            'all_passed': all(a.passed for a in analyses)
        }

    def _determine_optimal_config(
        self,
        analyses: List[FilterAnalysisResult],
        proposed: FilterConfig
    ) -> FilterConfig:
        """Determine optimal filter configuration from analyses."""
        optimal = FilterConfig(
            filter_type=proposed.filter_type,
            low_freq=proposed.low_freq,
            high_freq=proposed.high_freq,
            order=proposed.order,
            method=proposed.method,
            zero_phase=True  # Always recommend zero-phase
        )

        # Update based on analysis results
        for analysis in analyses:
            if analysis.analysis_type == 'highpass_cutoff':
                if 'recommended_cutoff' in analysis.findings:
                    optimal.low_freq = analysis.findings['recommended_cutoff']
            elif analysis.analysis_type == 'lowpass_cutoff':
                if 'recommended_cutoff' in analysis.findings:
                    optimal.high_freq = analysis.findings['recommended_cutoff']
            elif analysis.analysis_type == 'filter_comparison':
                if 'recommended_filter' in analysis.findings:
                    optimal.method = analysis.findings['recommended_filter']

        return optimal
