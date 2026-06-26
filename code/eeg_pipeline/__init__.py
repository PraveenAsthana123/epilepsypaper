"""
EEG Pipeline - 11-Phase Methodology Implementation
===================================================

This package implements a rigorous, leakage-safe EEG ML pipeline
following the 11-phase methodology for neurological disease detection.

Phases:
    1. Project Framing (spec.yaml)
    2. Data Acquisition (data_loader.py)
    3. Preprocessing (preprocessing.py)
    4. Normalization (normalization.py)
    5. EDA (eda.py)
    6. Feature Selection (feature_selection.py)
    7. Model Training (training.py)
    8. Validation (validation.py)
    9. Testing (testing.py)
    10. Benchmarking (benchmarking.py)
    11. Deployment (deployment.py)

Key Features:
    - Subject-wise splits to prevent leakage
    - Train-only normalization with saved scalers
    - Nested CV for hyperparameter optimization
    - Riemannian geometry features
    - Probability calibration
    - Bootstrap confidence intervals
    - Statistical significance testing
    - Comprehensive metrics (25+ classification, clinical, subject-wise)
    - Clinical validation (cross-dataset, subgroup, temporal)
    - Reliability analysis (test-retest, noise robustness, SQI)
    - Feature engineering analysis (importance, stability, ablation)
"""

__version__ = "1.1.0"
__author__ = "EEG Pipeline Team"

# Core pipeline modules
from .data_loader import EEGDataLoader, SubjectWiseSplitter
from .preprocessing import EEGPreprocessor
from .normalization import LeakageSafeNormalizer
from .feature_selection import FeatureSelector, RiemannianFeatures
from .training import EEGTrainer, NestedCV
from .validation import ValidationSuite, CalibrationValidator
from .testing import StatisticalTester, BootstrapCI
from .benchmarking import BenchmarkRunner

# Extended analysis modules
from .metrics import (
    ClassificationMetrics,
    ProbabilityMetrics,
    ClinicalMetrics,
    SubjectWiseMetrics,
    ReliabilityMetrics,
    ComprehensiveMetrics,
    MetricResult
)

from .clinical_validation import (
    CrossDatasetValidator,
    SubgroupAnalyzer,
    TemporalStabilityAnalyzer,
    FailureModeAnalyzer,
    ClinicalValidationReportGenerator
)

from .reliability_analysis import (
    TestRetestAnalyzer,
    NoiseRobustnessTester,
    MissingChannelTester,
    SignalQualityAnalyzer,
    ReliabilityReportGenerator
)

from .feature_engineering_analysis import (
    FeatureImportanceAnalyzer,
    FeatureStabilityAnalyzer,
    AblationStudyRunner,
    EEGFeatureValidator,
    FeatureAnalysisReportGenerator
)

# EDA and specialized analysis modules
from .eda_analysis import (
    DatasetOverviewAnalyzer,
    DistributionAnalyzer,
    TargetCorrelationAnalyzer,
    TemporalGroupAnalyzer,
    DataQualityAnalyzer,
    EDAReportGenerator
)

from .outlier_analysis import (
    AmplitudeOutlierDetector,
    FrequencyOutlierDetector,
    FeatureSpaceOutlierDetector,
    OutlierHandler,
    OutlierReportGenerator
)

from .filter_analysis import (
    SamplingValidator,
    FilterDesignAnalyzer,
    FilterQualityAnalyzer,
    FilterReportGenerator,
    FilterConfig
)

from .data_conversion import (
    FrequencyDomainConverter,
    WaveletConverter,
    AdvancedConverter,
    ConversionAnalyzer,
    ConversionPipeline
)

__all__ = [
    # Core pipeline
    'EEGDataLoader',
    'SubjectWiseSplitter',
    'EEGPreprocessor',
    'LeakageSafeNormalizer',
    'FeatureSelector',
    'RiemannianFeatures',
    'EEGTrainer',
    'NestedCV',
    'ValidationSuite',
    'CalibrationValidator',
    'StatisticalTester',
    'BootstrapCI',
    'BenchmarkRunner',

    # Comprehensive metrics
    'ClassificationMetrics',
    'ProbabilityMetrics',
    'ClinicalMetrics',
    'SubjectWiseMetrics',
    'ReliabilityMetrics',
    'ComprehensiveMetrics',
    'MetricResult',

    # Clinical validation
    'CrossDatasetValidator',
    'SubgroupAnalyzer',
    'TemporalStabilityAnalyzer',
    'FailureModeAnalyzer',
    'ClinicalValidationReportGenerator',

    # Reliability analysis
    'TestRetestAnalyzer',
    'NoiseRobustnessTester',
    'MissingChannelTester',
    'SignalQualityAnalyzer',
    'ReliabilityReportGenerator',

    # Feature analysis
    'FeatureImportanceAnalyzer',
    'FeatureStabilityAnalyzer',
    'AblationStudyRunner',
    'EEGFeatureValidator',
    'FeatureAnalysisReportGenerator',

    # EDA analysis
    'DatasetOverviewAnalyzer',
    'DistributionAnalyzer',
    'TargetCorrelationAnalyzer',
    'TemporalGroupAnalyzer',
    'DataQualityAnalyzer',
    'EDAReportGenerator',

    # Outlier analysis
    'AmplitudeOutlierDetector',
    'FrequencyOutlierDetector',
    'FeatureSpaceOutlierDetector',
    'OutlierHandler',
    'OutlierReportGenerator',

    # Filter analysis
    'SamplingValidator',
    'FilterDesignAnalyzer',
    'FilterQualityAnalyzer',
    'FilterReportGenerator',
    'FilterConfig',

    # Data conversion
    'FrequencyDomainConverter',
    'WaveletConverter',
    'AdvancedConverter',
    'ConversionAnalyzer',
    'ConversionPipeline',
]
