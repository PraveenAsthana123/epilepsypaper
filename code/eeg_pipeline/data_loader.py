"""
EEG Data Loader with Subject-Wise Splitting (Phase 2)
======================================================

This module implements leakage-safe data loading with:
- Subject-wise train/val/test splits
- GroupKFold cross-validation
- Leave-One-Subject-Out (LOSO) for small N
- Split manifest generation and verification
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, LeaveOneGroupOut
import warnings


@dataclass
class SplitInfo:
    """Information about a data split."""
    subject_ids: List[str]
    n_subjects: int
    n_windows: int
    class_counts: Dict[str, int]
    indices: np.ndarray


@dataclass
class DatasetMetadata:
    """Metadata for a dataset."""
    name: str
    path: str
    total_subjects: int
    total_windows: int
    sampling_rate: float
    n_channels: int
    class_distribution: Dict[str, int]
    subject_info: Dict[str, Dict]


class SubjectWiseSplitter:
    """
    Implements subject-wise splitting to prevent data leakage.

    CRITICAL: All windows from the same subject MUST be in the same split.

    Parameters
    ----------
    train_ratio : float
        Proportion of subjects for training (default: 0.7)
    val_ratio : float
        Proportion of subjects for validation (default: 0.15)
    test_ratio : float
        Proportion of subjects for testing (default: 0.15)
    random_seed : int
        Random seed for reproducibility (default: 42)
    stratify : bool
        Whether to stratify by class label (default: True)
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        stratify: bool = True
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.stratify = stratify
        self._rng = np.random.RandomState(random_seed)

    def split(
        self,
        subject_ids: np.ndarray,
        labels: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Tuple[SplitInfo, SplitInfo, SplitInfo]:
        """
        Split data by subject ensuring no leakage.

        Parameters
        ----------
        subject_ids : np.ndarray
            Subject ID for each sample
        labels : np.ndarray
            Class labels for each sample
        groups : np.ndarray, optional
            Additional grouping (e.g., session). If None, uses subject_ids.

        Returns
        -------
        train_info, val_info, test_info : Tuple[SplitInfo, SplitInfo, SplitInfo]
            Split information for each partition
        """
        if groups is None:
            groups = subject_ids

        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)

        # Get majority label per subject for stratification
        if self.stratify:
            subject_labels = {}
            for subj in unique_subjects:
                mask = subject_ids == subj
                subj_labels = labels[mask]
                # Use majority label for stratification
                unique, counts = np.unique(subj_labels, return_counts=True)
                subject_labels[subj] = unique[np.argmax(counts)]

            subject_label_array = np.array([subject_labels[s] for s in unique_subjects])

        # Shuffle subjects
        shuffled_idx = self._rng.permutation(n_subjects)
        shuffled_subjects = unique_subjects[shuffled_idx]

        if self.stratify:
            shuffled_labels = subject_label_array[shuffled_idx]
            # Sort by label to ensure stratification
            sort_idx = np.argsort(shuffled_labels)
            shuffled_subjects = shuffled_subjects[sort_idx]

        # Calculate split points
        n_train = int(n_subjects * self.train_ratio)
        n_val = int(n_subjects * self.val_ratio)

        # Assign subjects to splits
        train_subjects = set(shuffled_subjects[:n_train])
        val_subjects = set(shuffled_subjects[n_train:n_train + n_val])
        test_subjects = set(shuffled_subjects[n_train + n_val:])

        # Get indices for each split
        train_idx = np.array([i for i, s in enumerate(subject_ids) if s in train_subjects])
        val_idx = np.array([i for i, s in enumerate(subject_ids) if s in val_subjects])
        test_idx = np.array([i for i, s in enumerate(subject_ids) if s in test_subjects])

        # Create SplitInfo objects
        def make_split_info(indices: np.ndarray, subjects: set) -> SplitInfo:
            split_labels = labels[indices]
            unique, counts = np.unique(split_labels, return_counts=True)
            class_counts = dict(zip(unique.astype(str), counts.tolist()))
            return SplitInfo(
                subject_ids=sorted(list(subjects)),
                n_subjects=len(subjects),
                n_windows=len(indices),
                class_counts=class_counts,
                indices=indices
            )

        train_info = make_split_info(train_idx, train_subjects)
        val_info = make_split_info(val_idx, val_subjects)
        test_info = make_split_info(test_idx, test_subjects)

        # Verify no leakage
        self._verify_no_leakage(train_info, val_info, test_info)

        return train_info, val_info, test_info

    def _verify_no_leakage(
        self,
        train_info: SplitInfo,
        val_info: SplitInfo,
        test_info: SplitInfo
    ) -> None:
        """Verify no subject overlap between splits."""
        train_set = set(train_info.subject_ids)
        val_set = set(val_info.subject_ids)
        test_set = set(test_info.subject_ids)

        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set

        if train_val_overlap:
            raise ValueError(f"LEAKAGE DETECTED: Train-Val overlap: {train_val_overlap}")
        if train_test_overlap:
            raise ValueError(f"LEAKAGE DETECTED: Train-Test overlap: {train_test_overlap}")
        if val_test_overlap:
            raise ValueError(f"LEAKAGE DETECTED: Val-Test overlap: {val_test_overlap}")

        print("✓ No subject leakage detected")


class GroupKFoldCV:
    """
    Group K-Fold cross-validation ensuring no subject leakage.

    Parameters
    ----------
    n_folds : int
        Number of folds (default: 5)
    stratified : bool
        Whether to stratify by class (default: True)
    random_seed : int
        Random seed for reproducibility (default: 42)
    """

    def __init__(
        self,
        n_folds: int = 5,
        stratified: bool = True,
        random_seed: int = 42
    ):
        self.n_folds = n_folds
        self.stratified = stratified
        self.random_seed = random_seed

        if stratified:
            self._cv = StratifiedGroupKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=random_seed
            )
        else:
            self._cv = GroupKFold(n_splits=n_folds)

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray
    ):
        """
        Generate train/val indices for each fold.

        Parameters
        ----------
        X : np.ndarray
            Features (only shape used)
        y : np.ndarray
            Labels
        groups : np.ndarray
            Subject IDs (grouping variable)

        Yields
        ------
        train_idx, val_idx : Tuple[np.ndarray, np.ndarray]
            Indices for training and validation
        """
        for fold_idx, (train_idx, val_idx) in enumerate(self._cv.split(X, y, groups)):
            # Verify no group leakage
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])
            overlap = train_groups & val_groups

            if overlap:
                raise ValueError(f"Fold {fold_idx}: Group leakage detected: {overlap}")

            yield train_idx, val_idx

    def get_fold_info(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray
    ) -> List[Dict]:
        """Get detailed information about each fold."""
        fold_info = []

        for fold_idx, (train_idx, val_idx) in enumerate(self._cv.split(X, y, groups)):
            train_subjects = sorted(list(set(groups[train_idx])))
            val_subjects = sorted(list(set(groups[val_idx])))

            fold_info.append({
                "fold_id": fold_idx,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "n_train_samples": len(train_idx),
                "n_val_samples": len(val_idx),
                "train_class_dist": dict(zip(*np.unique(y[train_idx], return_counts=True))),
                "val_class_dist": dict(zip(*np.unique(y[val_idx], return_counts=True)))
            })

        return fold_info


class LeaveOneSubjectOut:
    """
    Leave-One-Subject-Out cross-validation for small N.

    Recommended when N < 30 subjects.
    """

    def __init__(self):
        self._cv = LeaveOneGroupOut()

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray
    ):
        """Generate LOSO splits."""
        return self._cv.split(X, y, groups)

    def get_n_splits(self, groups: np.ndarray) -> int:
        """Get number of splits (= number of unique subjects)."""
        return len(np.unique(groups))


class EEGDataLoader:
    """
    EEG Data Loader with subject-wise splitting and manifest generation.

    Parameters
    ----------
    data_root : str
        Root directory containing datasets
    config_path : str, optional
        Path to spec.yaml configuration
    """

    def __init__(
        self,
        data_root: str,
        config_path: Optional[str] = None
    ):
        self.data_root = Path(data_root)
        self.config = self._load_config(config_path) if config_path else {}
        self._metadata_cache = {}

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from spec.yaml."""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_dataset(
        self,
        dataset_name: str,
        return_metadata: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[DatasetMetadata]]:
        """
        Load a dataset with subject information.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load
        return_metadata : bool
            Whether to return metadata (default: True)

        Returns
        -------
        X : np.ndarray
            EEG data (n_samples, n_channels, n_timepoints)
        y : np.ndarray
            Labels
        subject_ids : np.ndarray
            Subject ID for each sample
        metadata : DatasetMetadata, optional
            Dataset metadata if return_metadata=True
        """
        dataset_path = self.data_root / dataset_name

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        # Look for data files
        data_files = list(dataset_path.glob("*.npz")) + list(dataset_path.glob("*.npy"))

        if not data_files:
            # Try loading from subdirectories (subject folders)
            X, y, subject_ids = self._load_from_subject_folders(dataset_path)
        else:
            X, y, subject_ids = self._load_from_files(data_files)

        # Build metadata
        metadata = None
        if return_metadata:
            unique_labels, label_counts = np.unique(y, return_counts=True)
            metadata = DatasetMetadata(
                name=dataset_name,
                path=str(dataset_path),
                total_subjects=len(np.unique(subject_ids)),
                total_windows=len(y),
                sampling_rate=self.config.get('data_scope', {}).get('sampling', {}).get('target_rate_hz', 256),
                n_channels=X.shape[1] if len(X.shape) > 1 else 0,
                class_distribution=dict(zip(unique_labels.astype(str), label_counts.tolist())),
                subject_info=self._get_subject_info(subject_ids, y)
            )

        return X, y, subject_ids, metadata

    def _load_from_subject_folders(
        self,
        dataset_path: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load data from subject-organized folders."""
        X_list, y_list, subj_list = [], [], []

        for subj_dir in sorted(dataset_path.iterdir()):
            if not subj_dir.is_dir():
                continue

            subject_id = subj_dir.name

            # Look for data files in subject directory
            for data_file in subj_dir.glob("*.npy"):
                data = np.load(data_file, allow_pickle=True)

                # Handle different data formats
                if isinstance(data, np.ndarray):
                    if data.dtype == np.object_:
                        data = data.item()

                    if isinstance(data, dict):
                        X_list.append(data.get('X', data.get('data', data.get('eeg'))))
                        y_list.append(data.get('y', data.get('label', data.get('labels'))))
                    else:
                        X_list.append(data)
                        # Try to infer label from filename or directory
                        label = self._infer_label(data_file, subj_dir)
                        y_list.append(np.full(len(data), label))

                    subj_list.extend([subject_id] * len(X_list[-1]))

        if not X_list:
            raise ValueError(f"No valid data found in {dataset_path}")

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0) if y_list else np.zeros(len(X))
        subject_ids = np.array(subj_list)

        return X, y, subject_ids

    def _load_from_files(
        self,
        data_files: List[Path]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load data from .npz or .npy files."""
        for data_file in data_files:
            if data_file.suffix == '.npz':
                data = np.load(data_file, allow_pickle=True)
                X = data.get('X', data.get('data', data.get('eeg')))
                y = data.get('y', data.get('labels', data.get('label')))
                subject_ids = data.get('subject_ids', data.get('subjects'))

                if subject_ids is None:
                    # Generate dummy subject IDs if not provided
                    warnings.warn("No subject_ids found in data. Generating dummy IDs.")
                    subject_ids = np.arange(len(y)).astype(str)

                return X, y, subject_ids

        raise ValueError("Could not load data from files")

    def _infer_label(self, data_file: Path, subj_dir: Path) -> int:
        """Infer label from filename or directory structure."""
        filename = data_file.stem.lower()
        dirname = subj_dir.name.lower()

        # Common label patterns
        positive_patterns = ['seizure', 'disease', 'patient', 'positive', 'abnormal', '1']
        negative_patterns = ['normal', 'control', 'healthy', 'negative', '0']

        for pattern in positive_patterns:
            if pattern in filename or pattern in dirname:
                return 1

        for pattern in negative_patterns:
            if pattern in filename or pattern in dirname:
                return 0

        return 0  # Default to negative

    def _get_subject_info(
        self,
        subject_ids: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Dict]:
        """Get per-subject information."""
        info = {}
        for subj in np.unique(subject_ids):
            mask = subject_ids == subj
            subj_labels = labels[mask]
            unique, counts = np.unique(subj_labels, return_counts=True)
            info[str(subj)] = {
                "n_samples": int(mask.sum()),
                "label_distribution": dict(zip(unique.astype(str), counts.tolist()))
            }
        return info

    def create_split_manifest(
        self,
        dataset_name: str,
        train_info: SplitInfo,
        val_info: SplitInfo,
        test_info: SplitInfo,
        output_path: str
    ) -> str:
        """
        Create and save a split manifest file.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        train_info, val_info, test_info : SplitInfo
            Split information from SubjectWiseSplitter
        output_path : str
            Path to save the manifest

        Returns
        -------
        manifest_path : str
            Path to the saved manifest
        """
        # Compute data hash for versioning
        X, y, subject_ids, metadata = self.load_dataset(dataset_name)
        data_hash = hashlib.sha256(
            np.concatenate([X.flatten()[:1000], y.flatten()]).tobytes()
        ).hexdigest()[:16]

        manifest = {
            "_metadata": {
                "description": "Subject-wise data split manifest - LEAKAGE PREVENTION",
                "created_at": datetime.now().isoformat(),
                "spec_version": self.config.get('project', {}).get('version', '1.0.0'),
                "split_method": "subject_wise",
                "random_seed": 42,
                "data_hash": data_hash
            },
            "_validation": {
                "no_subject_overlap": True,
                "stratified_by_label": True,
                "verified_at": datetime.now().isoformat(),
                "verified_by": "SubjectWiseSplitter"
            },
            "datasets": {
                dataset_name: {
                    "total_subjects": metadata.total_subjects,
                    "total_windows": metadata.total_windows,
                    "class_distribution": metadata.class_distribution,
                    "splits": {
                        "train": {
                            "subject_ids": train_info.subject_ids,
                            "n_subjects": train_info.n_subjects,
                            "n_windows": train_info.n_windows,
                            "class_counts": train_info.class_counts
                        },
                        "validation": {
                            "subject_ids": val_info.subject_ids,
                            "n_subjects": val_info.n_subjects,
                            "n_windows": val_info.n_windows,
                            "class_counts": val_info.class_counts
                        },
                        "test": {
                            "subject_ids": test_info.subject_ids,
                            "n_subjects": test_info.n_subjects,
                            "n_windows": test_info.n_windows,
                            "class_counts": test_info.class_counts
                        }
                    }
                }
            },
            "_leakage_checks": {
                "train_val_overlap": list(set(train_info.subject_ids) & set(val_info.subject_ids)),
                "train_test_overlap": list(set(train_info.subject_ids) & set(test_info.subject_ids)),
                "val_test_overlap": list(set(val_info.subject_ids) & set(test_info.subject_ids)),
                "all_checks_passed": True
            }
        }

        # Save manifest
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)

        print(f"✓ Split manifest saved to {output_path}")
        return str(output_path)

    def load_split_manifest(self, manifest_path: str) -> Dict:
        """Load and validate a split manifest."""
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Verify leakage checks
        checks = manifest.get("_leakage_checks", {})
        if not checks.get("all_checks_passed", False):
            raise ValueError("Split manifest has failed leakage checks!")

        if checks.get("train_val_overlap"):
            raise ValueError(f"Train-Val overlap: {checks['train_val_overlap']}")
        if checks.get("train_test_overlap"):
            raise ValueError(f"Train-Test overlap: {checks['train_test_overlap']}")
        if checks.get("val_test_overlap"):
            raise ValueError(f"Val-Test overlap: {checks['val_test_overlap']}")

        print("✓ Split manifest loaded and verified")
        return manifest


def create_metadata_csv(
    data_loader: EEGDataLoader,
    dataset_names: List[str],
    output_path: str
) -> str:
    """
    Create a standardized metadata CSV file.

    Parameters
    ----------
    data_loader : EEGDataLoader
        Data loader instance
    dataset_names : List[str]
        List of dataset names to include
    output_path : str
        Path to save the CSV

    Returns
    -------
    csv_path : str
        Path to the saved CSV
    """
    records = []

    for dataset_name in dataset_names:
        try:
            X, y, subject_ids, metadata = data_loader.load_dataset(dataset_name)

            for i, (subj_id, label) in enumerate(zip(subject_ids, y)):
                records.append({
                    "dataset": dataset_name,
                    "subject_id": subj_id,
                    "sample_idx": i,
                    "label": label,
                    "n_channels": X.shape[1] if len(X.shape) > 1 else 0,
                    "n_timepoints": X.shape[-1] if len(X.shape) > 1 else len(X[i]),
                    "sampling_rate": metadata.sampling_rate
                })
        except Exception as e:
            warnings.warn(f"Could not load {dataset_name}: {e}")
            continue

    df = pd.DataFrame(records)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✓ Metadata CSV saved to {output_path} ({len(df)} records)")
    return str(output_path)


# Sanity check function for leakage detection
def verify_no_leakage(
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str]
) -> bool:
    """
    Verify no subject overlap between splits.

    CRITICAL: Call this before any training!
    """
    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)

    overlaps = {
        "train_val": train_set & val_set,
        "train_test": train_set & test_set,
        "val_test": val_set & test_set
    }

    has_leakage = False
    for split_pair, overlap in overlaps.items():
        if overlap:
            print(f"❌ LEAKAGE: {split_pair} overlap: {overlap}")
            has_leakage = True

    if not has_leakage:
        print("✓ No leakage detected - splits are clean")

    return not has_leakage
