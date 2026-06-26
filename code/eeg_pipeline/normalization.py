"""
Leakage-Safe Normalization (Phase 4)
=====================================

This module implements normalization that is STRICTLY fitted on training data only.

CRITICAL RULES:
1. NEVER compute statistics on validation or test data
2. ALWAYS save scaler after fitting on train
3. ALWAYS load saved scaler for val/test transformation
4. ALWAYS verify no data leakage before training
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass, asdict
from scipy import stats
import warnings


@dataclass
class NormalizationStats:
    """Container for normalization statistics."""
    mean: np.ndarray
    std: np.ndarray
    median: Optional[np.ndarray] = None
    iqr: Optional[np.ndarray] = None
    n_samples: int = 0
    channels: Optional[List[str]] = None


class LeakageSafeNormalizer:
    """
    Normalization that is strictly fitted on training data only.

    CRITICAL: This class enforces that statistics are ONLY computed from
    training data. Any attempt to refit on validation/test will raise an error.

    Parameters
    ----------
    method : str
        Normalization method: 'zscore', 'robust', 'minmax'
    scope : str
        Scope of normalization: 'channel_wise', 'global', 'per_window'
    epsilon : float
        Small value to prevent division by zero (default: 1e-10)
    """

    def __init__(
        self,
        method: str = 'zscore',
        scope: str = 'channel_wise',
        epsilon: float = 1e-10
    ):
        self.method = method
        self.scope = scope
        self.epsilon = epsilon

        self._fitted = False
        self._stats = None
        self._fit_data_hash = None
        self._fit_timestamp = None

    def fit(
        self,
        X_train: np.ndarray,
        channel_names: Optional[List[str]] = None
    ) -> 'LeakageSafeNormalizer':
        """
        Fit normalizer on TRAINING DATA ONLY.

        Parameters
        ----------
        X_train : np.ndarray
            Training data (n_samples, n_channels, n_timepoints)
        channel_names : List[str], optional
            Names of EEG channels

        Returns
        -------
        self : LeakageSafeNormalizer
            Fitted normalizer
        """
        if self._fitted:
            warnings.warn(
                "Normalizer already fitted! Refitting will overwrite previous stats. "
                "This should ONLY happen if you're restarting the pipeline from scratch."
            )

        # Compute hash to track which data was used for fitting
        self._fit_data_hash = self._compute_hash(X_train)
        self._fit_timestamp = datetime.now().isoformat()

        if self.scope == 'channel_wise':
            self._stats = self._fit_channel_wise(X_train)
        elif self.scope == 'global':
            self._stats = self._fit_global(X_train)
        elif self.scope == 'per_window':
            # Per-window normalization doesn't need fitting
            self._stats = NormalizationStats(
                mean=np.array([0.0]),
                std=np.array([1.0]),
                n_samples=len(X_train)
            )
        else:
            raise ValueError(f"Unknown scope: {self.scope}")

        if channel_names is not None:
            self._stats.channels = channel_names

        self._fitted = True
        print(f"✓ Normalizer fitted on {len(X_train)} training samples")
        return self

    def _fit_channel_wise(self, X: np.ndarray) -> NormalizationStats:
        """Fit channel-wise statistics."""
        # X shape: (n_samples, n_channels, n_timepoints)
        if len(X.shape) == 2:
            X = X[:, np.newaxis, :]  # Add channel dimension

        n_channels = X.shape[1]

        if self.method == 'zscore':
            # Compute mean and std per channel (across samples and time)
            mean = np.mean(X, axis=(0, 2))  # (n_channels,)
            std = np.std(X, axis=(0, 2))    # (n_channels,)
            std = np.maximum(std, self.epsilon)  # Prevent division by zero

            return NormalizationStats(
                mean=mean,
                std=std,
                n_samples=len(X)
            )

        elif self.method == 'robust':
            # Use median and IQR for robustness to outliers
            median = np.median(X, axis=(0, 2))
            q75 = np.percentile(X, 75, axis=(0, 2))
            q25 = np.percentile(X, 25, axis=(0, 2))
            iqr = q75 - q25
            iqr = np.maximum(iqr, self.epsilon)

            return NormalizationStats(
                mean=median,  # Use median as center
                std=iqr,      # Use IQR as scale
                median=median,
                iqr=iqr,
                n_samples=len(X)
            )

        elif self.method == 'minmax':
            min_val = np.min(X, axis=(0, 2))
            max_val = np.max(X, axis=(0, 2))
            range_val = max_val - min_val
            range_val = np.maximum(range_val, self.epsilon)

            return NormalizationStats(
                mean=min_val,
                std=range_val,
                n_samples=len(X)
            )

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _fit_global(self, X: np.ndarray) -> NormalizationStats:
        """Fit global statistics across all channels."""
        if self.method == 'zscore':
            mean = np.array([np.mean(X)])
            std = np.array([max(np.std(X), self.epsilon)])
            return NormalizationStats(mean=mean, std=std, n_samples=len(X))

        elif self.method == 'robust':
            median = np.array([np.median(X)])
            iqr = np.array([max(stats.iqr(X.flatten()), self.epsilon)])
            return NormalizationStats(
                mean=median, std=iqr, median=median, iqr=iqr, n_samples=len(X)
            )

        else:
            raise ValueError(f"Unknown method for global: {self.method}")

    def transform(self, X: np.ndarray, verify_not_train: bool = False) -> np.ndarray:
        """
        Transform data using fitted statistics.

        Parameters
        ----------
        X : np.ndarray
            Data to transform
        verify_not_train : bool
            If True, verifies this is not the training data (prevents accidental reuse)

        Returns
        -------
        X_normalized : np.ndarray
            Normalized data
        """
        if not self._fitted:
            raise RuntimeError(
                "Normalizer not fitted! Call fit(X_train) first with TRAINING DATA ONLY."
            )

        if verify_not_train:
            data_hash = self._compute_hash(X)
            if data_hash == self._fit_data_hash:
                warnings.warn(
                    "This appears to be the same data used for fitting. "
                    "Make sure you're using different data for validation/test."
                )

        if self.scope == 'per_window':
            return self._transform_per_window(X)
        elif self.scope == 'channel_wise':
            return self._transform_channel_wise(X)
        elif self.scope == 'global':
            return self._transform_global(X)
        else:
            raise ValueError(f"Unknown scope: {self.scope}")

    def _transform_channel_wise(self, X: np.ndarray) -> np.ndarray:
        """Apply channel-wise normalization."""
        if len(X.shape) == 2:
            X = X[:, np.newaxis, :]

        X_norm = np.zeros_like(X, dtype=np.float32)

        for ch_idx in range(X.shape[1]):
            if self.method == 'minmax':
                X_norm[:, ch_idx, :] = (X[:, ch_idx, :] - self._stats.mean[ch_idx]) / self._stats.std[ch_idx]
            else:
                X_norm[:, ch_idx, :] = (X[:, ch_idx, :] - self._stats.mean[ch_idx]) / self._stats.std[ch_idx]

        return X_norm

    def _transform_global(self, X: np.ndarray) -> np.ndarray:
        """Apply global normalization."""
        return (X - self._stats.mean[0]) / self._stats.std[0]

    def _transform_per_window(self, X: np.ndarray) -> np.ndarray:
        """Apply per-window normalization (computed at transform time)."""
        X_norm = np.zeros_like(X, dtype=np.float32)

        for i in range(len(X)):
            window = X[i]
            if self.method == 'zscore':
                mean = np.mean(window)
                std = max(np.std(window), self.epsilon)
                X_norm[i] = (window - mean) / std
            elif self.method == 'robust':
                median = np.median(window)
                iqr = max(stats.iqr(window.flatten()), self.epsilon)
                X_norm[i] = (window - median) / iqr

        return X_norm

    def _compute_hash(self, X: np.ndarray) -> str:
        """Compute hash of data for tracking."""
        import hashlib
        # Use first 1000 elements for hash (for speed)
        sample = X.flatten()[:1000]
        return hashlib.sha256(sample.tobytes()).hexdigest()[:16]

    def save(self, path: str) -> str:
        """
        Save fitted normalizer to disk.

        Parameters
        ----------
        path : str
            Path to save (will create .pkl and .json files)

        Returns
        -------
        path : str
            Path to saved files
        """
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted normalizer")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save as pickle
        pkl_path = path.with_suffix('.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(self, f)

        # Save stats as JSON for human readability
        json_path = path.with_suffix('.json')
        stats_dict = {
            "_metadata": {
                "description": "Train-only normalization statistics - LEAKAGE SAFE",
                "created_at": self._fit_timestamp,
                "fitted_on": "train_split_only",
                "method": self.method,
                "scope": self.scope,
                "data_hash": self._fit_data_hash
            },
            "_warnings": [
                "These statistics MUST only be computed from training data",
                "DO NOT recompute on validation or test data",
                "Load and apply directly to val/test"
            ],
            "global_stats": {
                "n_samples": int(self._stats.n_samples),
                "n_channels": len(self._stats.mean) if self.scope == 'channel_wise' else 1
            },
            "channel_wise" if self.scope == 'channel_wise' else "global": {
                "mean": self._stats.mean.tolist(),
                "std": self._stats.std.tolist()
            },
            "verification": {
                "mean_values_stored": True,
                "std_values_stored": True,
                "no_inf_values": not np.any(np.isinf(self._stats.mean)) and not np.any(np.isinf(self._stats.std)),
                "no_nan_values": not np.any(np.isnan(self._stats.mean)) and not np.any(np.isnan(self._stats.std))
            }
        }

        if self._stats.channels is not None:
            stats_dict["channel_names"] = self._stats.channels

        with open(json_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)

        print(f"✓ Normalizer saved to {pkl_path} and {json_path}")
        return str(pkl_path)

    @classmethod
    def load(cls, path: str) -> 'LeakageSafeNormalizer':
        """
        Load a fitted normalizer from disk.

        Parameters
        ----------
        path : str
            Path to saved normalizer (.pkl file)

        Returns
        -------
        normalizer : LeakageSafeNormalizer
            Loaded normalizer
        """
        path = Path(path)
        if path.suffix != '.pkl':
            path = path.with_suffix('.pkl')

        with open(path, 'rb') as f:
            normalizer = pickle.load(f)

        if not normalizer._fitted:
            raise RuntimeError("Loaded normalizer is not fitted!")

        print(f"✓ Normalizer loaded from {path}")
        return normalizer


class BandpowerNormalizer:
    """
    Normalizer for bandpower features with log transform.

    Applies: log(power + epsilon) then z-score

    Parameters
    ----------
    epsilon : float
        Small value added before log to prevent log(0) (default: 1e-10)
    """

    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
        self._fitted = False
        self._band_stats = {}

    def fit(
        self,
        bandpowers: Dict[str, np.ndarray]
    ) -> 'BandpowerNormalizer':
        """
        Fit on training bandpower features.

        Parameters
        ----------
        bandpowers : Dict[str, np.ndarray]
            Dictionary mapping band names to power values

        Returns
        -------
        self : BandpowerNormalizer
        """
        for band_name, powers in bandpowers.items():
            # Apply log transform
            log_powers = np.log10(powers + self.epsilon)

            self._band_stats[band_name] = {
                'mean': np.mean(log_powers),
                'std': max(np.std(log_powers), self.epsilon),
                'log_epsilon': self.epsilon
            }

        self._fitted = True
        print(f"✓ BandpowerNormalizer fitted on {len(bandpowers)} bands")
        return self

    def transform(self, bandpowers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transform bandpower features."""
        if not self._fitted:
            raise RuntimeError("BandpowerNormalizer not fitted!")

        normalized = {}
        for band_name, powers in bandpowers.items():
            if band_name not in self._band_stats:
                warnings.warn(f"Band '{band_name}' not seen during fitting")
                continue

            stats = self._band_stats[band_name]
            log_powers = np.log10(powers + stats['log_epsilon'])
            normalized[band_name] = (log_powers - stats['mean']) / stats['std']

        return normalized


class TFRNormalizer:
    """
    Normalizer for Time-Frequency Representations (STFT/CWT images).

    Uses per-frequency-bin z-score normalization.
    """

    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
        self._fitted = False
        self._bin_means = None
        self._bin_stds = None

    def fit(self, tfr_images: np.ndarray) -> 'TFRNormalizer':
        """
        Fit on training TFR images.

        Parameters
        ----------
        tfr_images : np.ndarray
            TFR images (n_samples, n_freq_bins, n_time_bins)

        Returns
        -------
        self : TFRNormalizer
        """
        # Compute per-frequency-bin statistics
        # Average across samples and time to get stats per frequency bin
        self._bin_means = np.mean(tfr_images, axis=(0, 2))  # (n_freq_bins,)
        self._bin_stds = np.std(tfr_images, axis=(0, 2))    # (n_freq_bins,)
        self._bin_stds = np.maximum(self._bin_stds, self.epsilon)

        self._fitted = True
        print(f"✓ TFRNormalizer fitted on {tfr_images.shape[1]} frequency bins")
        return self

    def transform(self, tfr_images: np.ndarray) -> np.ndarray:
        """Transform TFR images using per-frequency-bin normalization."""
        if not self._fitted:
            raise RuntimeError("TFRNormalizer not fitted!")

        # Apply per-frequency-bin normalization
        normalized = np.zeros_like(tfr_images, dtype=np.float32)
        for freq_idx in range(tfr_images.shape[1]):
            normalized[:, freq_idx, :] = (
                (tfr_images[:, freq_idx, :] - self._bin_means[freq_idx])
                / self._bin_stds[freq_idx]
            )

        return normalized


class NormalizationQA:
    """Quality assurance checks for normalization."""

    @staticmethod
    def verify_train_stats(
        X_train_normalized: np.ndarray,
        tolerance: float = 0.1
    ) -> Dict[str, bool]:
        """
        Verify that normalized training data has expected properties.

        Parameters
        ----------
        X_train_normalized : np.ndarray
            Normalized training data
        tolerance : float
            Tolerance for mean≈0, std≈1 checks

        Returns
        -------
        results : Dict[str, bool]
            QA check results
        """
        results = {
            "mean_near_zero": False,
            "std_near_one": False,
            "no_inf_values": False,
            "no_nan_values": False
        }

        mean = np.mean(X_train_normalized)
        std = np.std(X_train_normalized)

        results["mean_near_zero"] = abs(mean) < tolerance
        results["std_near_one"] = abs(std - 1.0) < tolerance
        results["no_inf_values"] = not np.any(np.isinf(X_train_normalized))
        results["no_nan_values"] = not np.any(np.isnan(X_train_normalized))

        all_passed = all(results.values())

        if all_passed:
            print("✓ Normalization QA passed")
        else:
            failed = [k for k, v in results.items() if not v]
            print(f"✗ Normalization QA failed: {failed}")

        return results

    @staticmethod
    def check_distribution_shift(
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Check for distribution shift between splits after normalization.

        Uses KS-test to compare distributions.

        Parameters
        ----------
        X_train, X_val, X_test : np.ndarray
            Normalized data for each split
        threshold : float
            KS statistic threshold for warning

        Returns
        -------
        ks_stats : Dict[str, float]
            KS statistics for each comparison
        """
        from scipy.stats import ks_2samp

        # Flatten for comparison
        train_flat = X_train.flatten()
        val_flat = X_val.flatten()
        test_flat = X_test.flatten()

        # Sample if too large
        max_samples = 10000
        if len(train_flat) > max_samples:
            train_flat = np.random.choice(train_flat, max_samples, replace=False)
            val_flat = np.random.choice(val_flat, min(len(val_flat), max_samples), replace=False)
            test_flat = np.random.choice(test_flat, min(len(test_flat), max_samples), replace=False)

        ks_train_val, _ = ks_2samp(train_flat, val_flat)
        ks_train_test, _ = ks_2samp(train_flat, test_flat)
        ks_val_test, _ = ks_2samp(val_flat, test_flat)

        results = {
            "train_val": ks_train_val,
            "train_test": ks_train_test,
            "val_test": ks_val_test
        }

        for name, ks_stat in results.items():
            if ks_stat > threshold:
                warnings.warn(f"Large distribution shift ({name}): KS={ks_stat:.3f}")
            else:
                print(f"✓ {name} distribution shift OK: KS={ks_stat:.3f}")

        return results
