"""
EEG Data Conversion Module (1D → 2D)
=====================================

Comprehensive data conversion following the 20-point framework.

Covers:
- FFT and PSD conversion
- Spectrogram (STFT) conversion
- Wavelet transforms (CWT/DWT)
- Topographic mapping
- Gramian Angular Fields (GAF)
- Markov Transition Fields (MTF)
- Resolution and normalization analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from scipy import signal
import warnings


@dataclass
class ConversionResult:
    """Container for conversion result."""
    method: str
    output_shape: Tuple
    output_type: str  # 'image' or 'numerical'
    data: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class ConversionAnalysis:
    """Analysis result for a conversion method."""
    method: str
    question: str
    findings: Dict[str, Any]
    recommendations: List[str]
    suitable_for: List[str]  # Model types


# =============================================================================
# Frequency Domain Conversions
# =============================================================================

class FrequencyDomainConverter:
    """
    Convert EEG to frequency domain representations.

    Methods 3-5:
    3. FFT Spectrum Conversion
    4. Power Spectral Density (PSD)
    5. STFT Spectrogram
    """

    def __init__(self, sampling_rate: float = 256.0):
        self.sampling_rate = sampling_rate

    def to_fft_features(
        self,
        X: np.ndarray,
        n_fft: Optional[int] = None,
        max_freq: float = 100.0
    ) -> ConversionResult:
        """
        Convert to FFT magnitude features (numerical).

        Question: Does frequency magnitude capture signal?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        n_samples, n_channels, n_timepoints = X.shape

        if n_fft is None:
            n_fft = n_timepoints

        # Compute FFT
        fft_data = np.fft.rfft(X, n=n_fft, axis=-1)
        fft_magnitude = np.abs(fft_data)

        # Get frequencies
        freqs = np.fft.rfftfreq(n_fft, 1 / self.sampling_rate)

        # Limit to max_freq
        freq_mask = freqs <= max_freq
        fft_magnitude = fft_magnitude[:, :, freq_mask]
        freqs = freqs[freq_mask]

        # Flatten to (n_samples, n_channels * n_freqs)
        output = fft_magnitude.reshape(n_samples, -1)

        return ConversionResult(
            method='fft_features',
            output_shape=output.shape,
            output_type='numerical',
            data=output,
            metadata={
                'n_fft': n_fft,
                'frequency_bins': len(freqs),
                'frequency_range': (float(freqs[0]), float(freqs[-1])),
                'resolution_hz': float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0
            }
        )

    def to_psd_features(
        self,
        X: np.ndarray,
        nperseg: int = 256,
        bands: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> ConversionResult:
        """
        Convert to Power Spectral Density features (numerical).

        Question: Is power distribution meaningful?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        n_samples, n_channels, n_timepoints = X.shape

        # Default EEG bands
        if bands is None:
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
            }

        # Compute PSD
        freqs, psd = signal.welch(
            X, self.sampling_rate,
            nperseg=min(nperseg, n_timepoints),
            axis=-1
        )

        # Extract band powers
        band_powers = np.zeros((n_samples, n_channels, len(bands)))

        for i, (band_name, (low, high)) in enumerate(bands.items()):
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                band_powers[:, :, i] = np.mean(psd[:, :, mask], axis=-1)

        # Flatten
        output = band_powers.reshape(n_samples, -1)

        # Also include relative powers
        total_power = np.sum(psd, axis=-1, keepdims=True)
        relative_powers = band_powers / (total_power + 1e-10)
        relative_output = relative_powers.reshape(n_samples, -1)

        # Combine absolute and relative
        combined_output = np.hstack([output, relative_output])

        return ConversionResult(
            method='psd_features',
            output_shape=combined_output.shape,
            output_type='numerical',
            data=combined_output,
            metadata={
                'bands': list(bands.keys()),
                'n_channels': n_channels,
                'features_per_channel': len(bands) * 2,  # Absolute + relative
                'nperseg': nperseg
            }
        )

    def to_spectrogram(
        self,
        X: np.ndarray,
        nperseg: int = 64,
        noverlap: Optional[int] = None,
        output_size: Tuple[int, int] = (128, 128)
    ) -> ConversionResult:
        """
        Convert to STFT spectrogram images.

        Question: Does time-frequency structure emerge?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        n_samples, n_channels, n_timepoints = X.shape

        if noverlap is None:
            noverlap = nperseg // 2

        # Compute spectrograms for each sample and channel
        spectrograms = []

        for i in range(n_samples):
            channel_specs = []
            for j in range(n_channels):
                f, t, Sxx = signal.spectrogram(
                    X[i, j],
                    self.sampling_rate,
                    nperseg=nperseg,
                    noverlap=noverlap
                )

                # Log transform
                Sxx_log = 10 * np.log10(Sxx + 1e-10)

                # Resize if needed
                if Sxx_log.shape != output_size:
                    Sxx_log = self._resize_image(Sxx_log, output_size)

                channel_specs.append(Sxx_log)

            # Stack channels
            spectrograms.append(np.stack(channel_specs, axis=0))

        output = np.array(spectrograms)  # (n_samples, n_channels, freq, time)

        return ConversionResult(
            method='spectrogram',
            output_shape=output.shape,
            output_type='image',
            data=output,
            metadata={
                'nperseg': nperseg,
                'noverlap': noverlap,
                'frequency_bins': output.shape[2],
                'time_bins': output.shape[3],
                'log_scale': True
            }
        )

    def _resize_image(
        self,
        img: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Resize 2D array using bilinear interpolation."""
        from scipy.ndimage import zoom

        zoom_factors = (target_size[0] / img.shape[0], target_size[1] / img.shape[1])
        return zoom(img, zoom_factors, order=1)


# =============================================================================
# Wavelet Transforms
# =============================================================================

class WaveletConverter:
    """
    Convert EEG using wavelet transforms.

    Methods 7-8:
    7. Wavelet Transform (CWT/DWT)
    8. Band-Power Image Mapping
    """

    def __init__(self, sampling_rate: float = 256.0):
        self.sampling_rate = sampling_rate

    def to_cwt_scalogram(
        self,
        X: np.ndarray,
        wavelet: str = 'morl',
        n_scales: int = 64,
        freq_range: Tuple[float, float] = (1, 100)
    ) -> ConversionResult:
        """
        Convert to Continuous Wavelet Transform scalogram.

        Question: Are multi-scale dynamics preserved?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        n_samples, n_channels, n_timepoints = X.shape

        # Define scales (frequencies)
        freqs = np.linspace(freq_range[0], freq_range[1], n_scales)
        scales = self.sampling_rate / (2 * freqs)  # Convert to scales

        scalograms = []

        for i in range(n_samples):
            channel_scalograms = []
            for j in range(n_channels):
                # Compute CWT using scipy
                try:
                    cwt_matrix = signal.cwt(X[i, j], signal.morlet2, scales)
                    cwt_power = np.abs(cwt_matrix) ** 2
                except Exception:
                    # Fallback: use ricker wavelet
                    cwt_matrix = signal.cwt(X[i, j], signal.ricker, scales)
                    cwt_power = np.abs(cwt_matrix) ** 2

                channel_scalograms.append(cwt_power)

            scalograms.append(np.stack(channel_scalograms, axis=0))

        output = np.array(scalograms)

        return ConversionResult(
            method='cwt_scalogram',
            output_shape=output.shape,
            output_type='image',
            data=output,
            metadata={
                'wavelet': wavelet,
                'n_scales': n_scales,
                'frequency_range': freq_range,
                'scales': scales.tolist()[:10]  # First 10
            }
        )

    def to_band_power_image(
        self,
        X: np.ndarray,
        window_size: int = 64,
        overlap: float = 0.5,
        bands: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> ConversionResult:
        """
        Convert to band-power heatmap image.

        Question: Do EEG bands form stable 2D patterns?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        n_samples, n_channels, n_timepoints = X.shape

        if bands is None:
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
            }

        step = int(window_size * (1 - overlap))
        n_windows = (n_timepoints - window_size) // step + 1

        band_images = []

        for i in range(n_samples):
            # Shape: (n_channels, n_bands, n_windows)
            sample_image = np.zeros((n_channels, len(bands), n_windows))

            for w in range(n_windows):
                start = w * step
                end = start + window_size
                segment = X[i, :, start:end]

                # Compute band powers for each channel
                for b, (band_name, (low, high)) in enumerate(bands.items()):
                    # Bandpass filter
                    nyq = self.sampling_rate / 2
                    if high < nyq:
                        try:
                            sos = signal.butter(4, [low / nyq, high / nyq], btype='band', output='sos')
                            filtered = signal.sosfilt(sos, segment, axis=-1)
                            power = np.mean(filtered ** 2, axis=-1)
                        except Exception:
                            power = np.zeros(n_channels)
                    else:
                        power = np.zeros(n_channels)

                    sample_image[:, b, w] = power

            band_images.append(sample_image)

        output = np.array(band_images)

        return ConversionResult(
            method='band_power_image',
            output_shape=output.shape,
            output_type='image',
            data=output,
            metadata={
                'bands': list(bands.keys()),
                'n_windows': n_windows,
                'window_size': window_size,
                'overlap': overlap
            }
        )


# =============================================================================
# Advanced Transformations
# =============================================================================

class AdvancedConverter:
    """
    Advanced transformation methods.

    Methods 11-13:
    11. Recurrence Plot
    12. Gramian Angular Field (GAF)
    13. Markov Transition Field (MTF)
    """

    def __init__(self, sampling_rate: float = 256.0):
        self.sampling_rate = sampling_rate

    def to_gramian_angular_field(
        self,
        X: np.ndarray,
        method: str = 'summation',
        image_size: int = 64
    ) -> ConversionResult:
        """
        Convert to Gramian Angular Field image.

        Question: Does angular encoding preserve dynamics?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        n_samples, n_channels, n_timepoints = X.shape

        gaf_images = []

        for i in range(n_samples):
            channel_gafs = []
            for j in range(n_channels):
                # Normalize to [-1, 1]
                x = X[i, j]
                x_min, x_max = np.min(x), np.max(x)
                x_norm = 2 * (x - x_min) / (x_max - x_min + 1e-10) - 1
                x_norm = np.clip(x_norm, -1, 1)

                # Downsample if needed
                if len(x_norm) > image_size:
                    indices = np.linspace(0, len(x_norm) - 1, image_size, dtype=int)
                    x_norm = x_norm[indices]

                # Convert to polar coordinates (angle)
                phi = np.arccos(x_norm)

                # Compute GAF
                if method == 'summation':
                    # GASF: cos(phi_i + phi_j)
                    gaf = np.cos(np.add.outer(phi, phi))
                else:
                    # GADF: sin(phi_i - phi_j)
                    gaf = np.sin(np.subtract.outer(phi, phi))

                channel_gafs.append(gaf)

            gaf_images.append(np.stack(channel_gafs, axis=0))

        output = np.array(gaf_images)

        return ConversionResult(
            method=f'gaf_{method}',
            output_shape=output.shape,
            output_type='image',
            data=output,
            metadata={
                'method': method,
                'image_size': image_size
            }
        )

    def to_markov_transition_field(
        self,
        X: np.ndarray,
        n_bins: int = 8,
        image_size: int = 64
    ) -> ConversionResult:
        """
        Convert to Markov Transition Field image.

        Question: Do state transitions add value?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        n_samples, n_channels, n_timepoints = X.shape

        mtf_images = []

        for i in range(n_samples):
            channel_mtfs = []
            for j in range(n_channels):
                x = X[i, j]

                # Quantize to bins
                bin_edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
                bin_edges[-1] += 1e-10  # Include max
                quantized = np.digitize(x, bin_edges) - 1
                quantized = np.clip(quantized, 0, n_bins - 1)

                # Compute transition matrix
                trans_matrix = np.zeros((n_bins, n_bins))
                for k in range(len(quantized) - 1):
                    trans_matrix[quantized[k], quantized[k + 1]] += 1

                # Normalize rows
                row_sums = trans_matrix.sum(axis=1, keepdims=True)
                trans_matrix = trans_matrix / (row_sums + 1e-10)

                # Build MTF
                # Downsample time series if needed
                if len(quantized) > image_size:
                    indices = np.linspace(0, len(quantized) - 1, image_size, dtype=int)
                    quantized_ds = quantized[indices]
                else:
                    quantized_ds = quantized
                    indices = np.arange(len(quantized))

                mtf = np.zeros((len(quantized_ds), len(quantized_ds)))
                for row in range(len(quantized_ds)):
                    for col in range(len(quantized_ds)):
                        mtf[row, col] = trans_matrix[quantized_ds[row], quantized_ds[col]]

                channel_mtfs.append(mtf)

            mtf_images.append(np.stack(channel_mtfs, axis=0))

        output = np.array(mtf_images)

        return ConversionResult(
            method='mtf',
            output_shape=output.shape,
            output_type='image',
            data=output,
            metadata={
                'n_bins': n_bins,
                'image_size': image_size
            }
        )

    def to_recurrence_plot(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None,
        image_size: int = 64
    ) -> ConversionResult:
        """
        Convert to recurrence plot image.

        Question: Are temporal recurrences informative?
        """
        if X.ndim == 2:
            X = X.reshape(len(X), 1, -1)

        n_samples, n_channels, n_timepoints = X.shape

        rp_images = []

        for i in range(n_samples):
            channel_rps = []
            for j in range(n_channels):
                x = X[i, j]

                # Downsample if needed
                if len(x) > image_size:
                    indices = np.linspace(0, len(x) - 1, image_size, dtype=int)
                    x = x[indices]

                # Compute distance matrix
                dist_matrix = np.abs(x[:, np.newaxis] - x[np.newaxis, :])

                # Threshold
                if threshold is None:
                    threshold = np.percentile(dist_matrix, 10)

                rp = (dist_matrix < threshold).astype(float)
                channel_rps.append(rp)

            rp_images.append(np.stack(channel_rps, axis=0))

        output = np.array(rp_images)

        return ConversionResult(
            method='recurrence_plot',
            output_shape=output.shape,
            output_type='image',
            data=output,
            metadata={
                'threshold': float(threshold) if threshold else 'auto',
                'image_size': image_size
            }
        )


# =============================================================================
# Comprehensive Conversion Analyzer
# =============================================================================

class ConversionAnalyzer:
    """
    Analyze and compare different conversion methods.
    """

    def __init__(self, sampling_rate: float = 256.0):
        self.sampling_rate = sampling_rate
        self.freq_converter = FrequencyDomainConverter(sampling_rate)
        self.wavelet_converter = WaveletConverter(sampling_rate)
        self.advanced_converter = AdvancedConverter(sampling_rate)

    def compare_methods(
        self,
        X: np.ndarray,
        y: np.ndarray,
        methods: List[str] = ['fft', 'psd', 'spectrogram', 'gaf', 'mtf']
    ) -> Dict[str, ConversionAnalysis]:
        """
        Compare different conversion methods.

        Question: Which conversion best suits the task?
        """
        results = {}

        for method in methods:
            try:
                if method == 'fft':
                    conv = self.freq_converter.to_fft_features(X)
                elif method == 'psd':
                    conv = self.freq_converter.to_psd_features(X)
                elif method == 'spectrogram':
                    conv = self.freq_converter.to_spectrogram(X)
                elif method == 'gaf':
                    conv = self.advanced_converter.to_gramian_angular_field(X)
                elif method == 'mtf':
                    conv = self.advanced_converter.to_markov_transition_field(X)
                elif method == 'cwt':
                    conv = self.wavelet_converter.to_cwt_scalogram(X)
                elif method == 'band_power':
                    conv = self.wavelet_converter.to_band_power_image(X)
                else:
                    continue

                # Analyze conversion
                analysis = self._analyze_conversion(conv, y)
                results[method] = analysis

            except Exception as e:
                results[method] = ConversionAnalysis(
                    method=method,
                    question=f'Error in {method} conversion',
                    findings={'error': str(e)},
                    recommendations=[],
                    suitable_for=[]
                )

        return results

    def _analyze_conversion(
        self,
        conv: ConversionResult,
        y: np.ndarray
    ) -> ConversionAnalysis:
        """Analyze a single conversion result."""
        data = conv.data
        n_samples = len(data)

        # Flatten for analysis
        data_flat = data.reshape(n_samples, -1)

        # Check for class separability (simple metric)
        if len(np.unique(y)) == 2:
            class_0 = data_flat[y == np.unique(y)[0]]
            class_1 = data_flat[y == np.unique(y)[1]]

            # Mean effect size across features
            effect_sizes = []
            for i in range(min(data_flat.shape[1], 100)):
                m0, m1 = np.mean(class_0[:, i]), np.mean(class_1[:, i])
                pooled_std = np.sqrt((np.var(class_0[:, i]) + np.var(class_1[:, i])) / 2)
                if pooled_std > 0:
                    effect_sizes.append(abs(m1 - m0) / pooled_std)

            mean_effect = np.mean(effect_sizes) if effect_sizes else 0
        else:
            mean_effect = 0

        # Determine suitable models
        suitable_for = []
        if conv.output_type == 'image':
            suitable_for.extend(['CNN', 'ResNet', 'ViT'])
        else:
            suitable_for.extend(['RandomForest', 'SVM', 'XGBoost', 'MLP'])

        if conv.output_shape[-1] > 100:  # Sequential
            suitable_for.append('LSTM')

        findings = {
            'output_shape': conv.output_shape,
            'output_type': conv.output_type,
            'n_features': np.prod(conv.output_shape[1:]),
            'mean_effect_size': float(mean_effect),
            'memory_mb': float(data.nbytes / (1024 * 1024)),
            'metadata': conv.metadata
        }

        recommendations = []
        if mean_effect < 0.2:
            recommendations.append('Low class separability - may need feature selection')
        if findings['n_features'] > 10000:
            recommendations.append('High dimensionality - consider dimensionality reduction')

        return ConversionAnalysis(
            method=conv.method,
            question=f'Is {conv.method} suitable for this task?',
            findings=findings,
            recommendations=recommendations,
            suitable_for=suitable_for
        )

    def recommend_conversion(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_model: str = 'CNN'
    ) -> str:
        """
        Recommend best conversion method for target model.

        Question: Which representation suits this model?
        """
        comparisons = self.compare_methods(X, y)

        best_method = None
        best_score = -1

        for method, analysis in comparisons.items():
            if target_model in analysis.suitable_for:
                score = analysis.findings.get('mean_effect_size', 0)
                if score > best_score:
                    best_score = score
                    best_method = method

        return best_method or 'psd'


# =============================================================================
# Conversion Pipeline
# =============================================================================

class ConversionPipeline:
    """
    Pipeline for applying EEG data conversions.
    """

    def __init__(self, sampling_rate: float = 256.0):
        self.sampling_rate = sampling_rate
        self.analyzer = ConversionAnalyzer(sampling_rate)
        self._fitted_params = {}

    def fit_transform(
        self,
        X: np.ndarray,
        method: str = 'spectrogram',
        **kwargs
    ) -> np.ndarray:
        """
        Fit and transform EEG data.

        Stores normalization parameters for consistent transform.
        """
        # Get converter based on method
        if method in ['fft', 'psd', 'spectrogram']:
            converter = self.analyzer.freq_converter
        elif method in ['cwt', 'band_power']:
            converter = self.analyzer.wavelet_converter
        else:
            converter = self.analyzer.advanced_converter

        # Apply conversion
        if method == 'fft':
            result = converter.to_fft_features(X, **kwargs)
        elif method == 'psd':
            result = converter.to_psd_features(X, **kwargs)
        elif method == 'spectrogram':
            result = converter.to_spectrogram(X, **kwargs)
        elif method == 'cwt':
            result = converter.to_cwt_scalogram(X, **kwargs)
        elif method == 'band_power':
            result = converter.to_band_power_image(X, **kwargs)
        elif method == 'gaf':
            result = converter.to_gramian_angular_field(X, **kwargs)
        elif method == 'mtf':
            result = converter.to_markov_transition_field(X, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Store normalization parameters
        data = result.data
        self._fitted_params[method] = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data)
        }

        return data

    def transform(
        self,
        X: np.ndarray,
        method: str = 'spectrogram',
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Transform EEG data using fitted parameters.
        """
        # Get converter
        if method in ['fft', 'psd', 'spectrogram']:
            converter = self.analyzer.freq_converter
        elif method in ['cwt', 'band_power']:
            converter = self.analyzer.wavelet_converter
        else:
            converter = self.analyzer.advanced_converter

        # Apply conversion
        if method == 'fft':
            result = converter.to_fft_features(X, **kwargs)
        elif method == 'psd':
            result = converter.to_psd_features(X, **kwargs)
        elif method == 'spectrogram':
            result = converter.to_spectrogram(X, **kwargs)
        elif method == 'cwt':
            result = converter.to_cwt_scalogram(X, **kwargs)
        elif method == 'band_power':
            result = converter.to_band_power_image(X, **kwargs)
        elif method == 'gaf':
            result = converter.to_gramian_angular_field(X, **kwargs)
        elif method == 'mtf':
            result = converter.to_markov_transition_field(X, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        data = result.data

        # Apply normalization using fitted parameters
        if normalize and method in self._fitted_params:
            params = self._fitted_params[method]
            data = (data - params['mean']) / (params['std'] + 1e-10)

        return data
