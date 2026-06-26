"""
EEG Preprocessing (Phase 3)
============================

This module provides preprocessing utilities that integrate with
the existing preprocessing infrastructure while adding:
- Standardized preprocessing pipeline
- QC reporting
- Artifact rejection with logging
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import signal
import warnings


@dataclass
class PreprocessingConfig:
    """Configuration for EEG preprocessing."""
    sampling_rate: float = 256.0
    filter_low: float = 0.5
    filter_high: float = 45.0
    notch_freq: float = 60.0  # 50 for Europe, 60 for North America
    notch_width: float = 2.0
    epoch_duration: float = 4.0
    epoch_overlap: float = 0.5
    artifact_threshold_uv: float = 100.0
    reference_method: str = 'average'  # 'average', 'cz', 'linked_mastoids'


@dataclass
class QCReport:
    """Quality control report for preprocessing."""
    n_channels: int
    n_samples: int
    sampling_rate: float
    n_epochs: int
    n_epochs_rejected: int
    rejection_rate: float
    channels_rejected: List[str]
    sqi_mean: float
    sqi_std: float
    passed: bool


class EEGPreprocessor:
    """
    EEG Preprocessing pipeline with QC.

    This class wraps preprocessing functions with:
    - Consistent configuration
    - Quality control checks
    - Artifact logging
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._qc_reports = []

    def preprocess(
        self,
        raw_data: np.ndarray,
        channel_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, QCReport]:
        """
        Run full preprocessing pipeline.

        Parameters
        ----------
        raw_data : np.ndarray
            Raw EEG data (n_channels, n_samples) or (n_samples, n_channels)
        channel_names : List[str], optional
            Names of EEG channels

        Returns
        -------
        processed : np.ndarray
            Preprocessed data (n_epochs, n_channels, n_timepoints)
        qc_report : QCReport
            Quality control report
        """
        # Ensure correct shape (n_channels, n_samples)
        if raw_data.shape[0] > raw_data.shape[1]:
            raw_data = raw_data.T

        n_channels, n_samples = raw_data.shape

        if channel_names is None:
            channel_names = [f"Ch{i}" for i in range(n_channels)]

        # 1. Check raw data quality
        self._check_raw_quality(raw_data, channel_names)

        # 2. Re-reference
        data = self._rereference(raw_data)

        # 3. Notch filter (remove powerline)
        data = self._notch_filter(data)

        # 4. Bandpass filter
        data = self._bandpass_filter(data)

        # 5. Epoch the data
        epochs, epoch_quality = self._epoch_data(data)

        # 6. Reject bad epochs
        good_epochs, n_rejected = self._reject_artifacts(epochs, epoch_quality)

        # 7. Check bad channels
        bad_channels = self._check_bad_channels(good_epochs, channel_names)

        # Generate QC report
        sqi_values = np.mean(epoch_quality, axis=1)  # Mean SQI per epoch
        qc_report = QCReport(
            n_channels=n_channels,
            n_samples=n_samples,
            sampling_rate=self.config.sampling_rate,
            n_epochs=len(epochs),
            n_epochs_rejected=n_rejected,
            rejection_rate=n_rejected / len(epochs) if len(epochs) > 0 else 0,
            channels_rejected=bad_channels,
            sqi_mean=np.mean(sqi_values),
            sqi_std=np.std(sqi_values),
            passed=n_rejected / len(epochs) < 0.3 if len(epochs) > 0 else False
        )

        self._qc_reports.append(qc_report)

        return good_epochs, qc_report

    def _check_raw_quality(
        self,
        data: np.ndarray,
        channel_names: List[str]
    ) -> None:
        """Check raw data for obvious issues."""
        # Check for flat channels
        channel_vars = np.var(data, axis=1)
        flat_channels = np.where(channel_vars < 1e-10)[0]

        if len(flat_channels) > 0:
            flat_names = [channel_names[i] for i in flat_channels]
            warnings.warn(f"Flat channels detected: {flat_names}")

        # Check for clipping
        max_vals = np.max(np.abs(data), axis=1)
        clipped = np.where(max_vals > 500)[0]  # > 500 uV likely clipped

        if len(clipped) > 0:
            clipped_names = [channel_names[i] for i in clipped]
            warnings.warn(f"Possible clipping in channels: {clipped_names}")

    def _rereference(self, data: np.ndarray) -> np.ndarray:
        """Apply re-referencing."""
        if self.config.reference_method == 'average':
            # Common Average Reference (CAR)
            ref = np.mean(data, axis=0)
            return data - ref
        elif self.config.reference_method == 'cz':
            # Assume Cz is at index n_channels // 2 (approximate)
            ref_idx = data.shape[0] // 2
            return data - data[ref_idx:ref_idx+1, :]
        else:
            return data

    def _notch_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filter to remove powerline interference."""
        fs = self.config.sampling_rate
        f0 = self.config.notch_freq
        Q = f0 / self.config.notch_width

        b, a = signal.iirnotch(f0, Q, fs)
        filtered = signal.filtfilt(b, a, data, axis=1)

        return filtered

    def _bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter."""
        fs = self.config.sampling_rate
        low = self.config.filter_low
        high = self.config.filter_high

        # Design filter
        order = 4
        nyq = fs / 2
        low_norm = low / nyq
        high_norm = high / nyq

        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        filtered = signal.filtfilt(b, a, data, axis=1)

        return filtered

    def _epoch_data(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Epoch continuous data into fixed-length segments."""
        fs = self.config.sampling_rate
        epoch_samples = int(self.config.epoch_duration * fs)
        overlap_samples = int(epoch_samples * self.config.epoch_overlap)
        step = epoch_samples - overlap_samples

        n_channels, n_samples = data.shape
        n_epochs = (n_samples - epoch_samples) // step + 1

        epochs = np.zeros((n_epochs, n_channels, epoch_samples))
        epoch_quality = np.zeros((n_epochs, n_channels))

        for i in range(n_epochs):
            start = i * step
            end = start + epoch_samples
            epochs[i] = data[:, start:end]

            # Compute per-channel quality (inverse of variance as simple SQI)
            for ch in range(n_channels):
                var = np.var(epochs[i, ch])
                # SQI: lower variance is better (more stable signal)
                epoch_quality[i, ch] = 1.0 / (1.0 + var / 100.0)

        return epochs, epoch_quality

    def _reject_artifacts(
        self,
        epochs: np.ndarray,
        epoch_quality: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Reject epochs with artifacts."""
        threshold = self.config.artifact_threshold_uv
        n_epochs = len(epochs)

        good_mask = np.ones(n_epochs, dtype=bool)

        for i in range(n_epochs):
            epoch = epochs[i]

            # Check peak-to-peak amplitude
            ptp = np.max(epoch, axis=1) - np.min(epoch, axis=1)
            if np.any(ptp > threshold):
                good_mask[i] = False
                continue

            # Check for sudden jumps
            diff = np.abs(np.diff(epoch, axis=1))
            if np.any(diff > threshold / 2):
                good_mask[i] = False
                continue

        good_epochs = epochs[good_mask]
        n_rejected = n_epochs - np.sum(good_mask)

        return good_epochs, n_rejected

    def _check_bad_channels(
        self,
        epochs: np.ndarray,
        channel_names: List[str]
    ) -> List[str]:
        """Identify bad channels based on statistics."""
        if len(epochs) == 0:
            return []

        # Compute channel statistics across epochs
        channel_vars = np.var(epochs, axis=(0, 2))  # Variance per channel
        channel_corrs = []

        for ch in range(epochs.shape[1]):
            # Correlation with mean of other channels
            other_mean = np.mean(np.delete(epochs, ch, axis=1), axis=1)
            corr = np.corrcoef(epochs[:, ch].flatten(), other_mean.flatten())[0, 1]
            channel_corrs.append(corr)

        channel_corrs = np.array(channel_corrs)

        # Bad channels: high variance or low correlation
        var_threshold = np.median(channel_vars) + 3 * np.std(channel_vars)
        corr_threshold = 0.3

        bad_idx = np.where(
            (channel_vars > var_threshold) |
            (channel_corrs < corr_threshold) |
            np.isnan(channel_corrs)
        )[0]

        return [channel_names[i] for i in bad_idx]

    def get_qc_summary(self) -> Dict:
        """Get summary of all QC reports."""
        if not self._qc_reports:
            return {}

        return {
            'n_files_processed': len(self._qc_reports),
            'mean_rejection_rate': np.mean([r.rejection_rate for r in self._qc_reports]),
            'mean_sqi': np.mean([r.sqi_mean for r in self._qc_reports]),
            'n_passed': sum(1 for r in self._qc_reports if r.passed),
            'pass_rate': sum(1 for r in self._qc_reports if r.passed) / len(self._qc_reports)
        }


def compute_sqi(data: np.ndarray, fs: float = 256.0) -> float:
    """
    Compute Signal Quality Index for EEG segment.

    Parameters
    ----------
    data : np.ndarray
        EEG segment (n_channels, n_samples) or (n_samples,)
    fs : float
        Sampling rate

    Returns
    -------
    sqi : float
        Signal quality index (0-1, higher is better)
    """
    if len(data.shape) == 1:
        data = data.reshape(1, -1)

    # Multiple SQI components
    scores = []

    # 1. Amplitude range score
    ptp = np.ptp(data, axis=1)
    amp_score = 1.0 - np.clip(np.mean(ptp) / 200.0, 0, 1)  # Good if < 200 uV
    scores.append(amp_score)

    # 2. Variance stability score
    var = np.var(data, axis=1)
    var_score = 1.0 - np.clip(np.std(var) / np.mean(var), 0, 1)
    scores.append(var_score)

    # 3. High-frequency noise score (EMG)
    nyq = fs / 2
    high_cutoff = min(40.0 / nyq, 0.99)
    b, a = signal.butter(4, high_cutoff, btype='high')
    high_freq = signal.filtfilt(b, a, data, axis=1)
    hf_power = np.mean(high_freq ** 2)
    total_power = np.mean(data ** 2)
    hf_ratio = hf_power / (total_power + 1e-10)
    hf_score = 1.0 - np.clip(hf_ratio * 10, 0, 1)  # Good if low HF
    scores.append(hf_score)

    # 4. Flatline score
    diff = np.abs(np.diff(data, axis=1))
    flatline_ratio = np.mean(diff < 0.1) / data.shape[1]
    flatline_score = 1.0 - flatline_ratio
    scores.append(flatline_score)

    return float(np.mean(scores))
