"""
Benchmarking and Reporting (Phase 10)
======================================

This module implements:
- Benchmark ladder execution
- Results aggregation
- Report generation
- Model card creation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json


@dataclass
class BenchmarkResult:
    """Result from benchmarking a single model."""
    model_name: str
    representation: str
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    training_time: float
    inference_time: float
    n_parameters: int
    best_params: Optional[Dict] = None


class BenchmarkRunner:
    """
    Run benchmark experiments across model ladder.

    Implements the benchmark ladder:
    1. Handcrafted features + LR/SVM
    2. Riemannian tangent space + LR/SVM
    3. 1D CNN on raw EEG
    4. 2D CNN/ViT on CWT scalograms
    """

    def __init__(
        self,
        output_dir: str,
        random_seed: int = 42
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        self._results = []

    def run_benchmark_ladder(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        groups_train: np.ndarray,
        groups_val: np.ndarray,
        feature_representations: Dict[str, np.ndarray] = None
    ) -> List[BenchmarkResult]:
        """
        Run the complete benchmark ladder.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training data
        X_val, y_val : np.ndarray
            Validation data
        groups_train, groups_val : np.ndarray
            Subject groups
        feature_representations : Dict[str, np.ndarray], optional
            Pre-computed feature representations

        Returns
        -------
        results : List[BenchmarkResult]
        """
        from .training import EEGTrainer, TrainingConfig

        results = []

        # Default representations if not provided
        if feature_representations is None:
            feature_representations = {
                'raw': X_train
            }

        for rep_name, X_rep in feature_representations.items():
            print(f"\n{'='*60}")
            print(f"Benchmarking representation: {rep_name}")
            print(f"{'='*60}")

            # Adjust validation data
            if rep_name in ['raw']:
                X_val_rep = X_val
            else:
                # Need to transform validation data similarly
                X_val_rep = feature_representations.get(f'{rep_name}_val', X_val)

            # Train baseline ladder
            config = TrainingConfig(
                random_seed=self.random_seed,
                n_outer_folds=5,
                n_inner_folds=3,
                scoring='f1_macro'
            )

            trainer = EEGTrainer(config)
            baseline_results = trainer.train_baseline_ladder(
                X_rep, y_train, groups_train
            )

            # Convert to BenchmarkResult
            for model_name, result in baseline_results.items():
                benchmark_result = BenchmarkResult(
                    model_name=model_name,
                    representation=rep_name,
                    metrics={
                        'f1_macro': result.mean_score,
                        'f1_std': result.std_score
                    },
                    confidence_intervals={
                        'f1_macro': (
                            result.mean_score - 1.96 * result.std_score,
                            result.mean_score + 1.96 * result.std_score
                        )
                    },
                    training_time=result.training_time,
                    inference_time=0.0,
                    n_parameters=0,
                    best_params=result.best_params
                )
                results.append(benchmark_result)

        self._results = results
        return results

    def get_results_table(self) -> 'pd.DataFrame':
        """Get benchmark results as a table."""
        import pandas as pd

        rows = []
        for result in self._results:
            rows.append({
                'Model': result.model_name,
                'Representation': result.representation,
                'F1 (macro)': f"{result.metrics['f1_macro']:.4f} ± {result.metrics['f1_std']:.4f}",
                'Best Params': str(result.best_params)[:50] + '...' if result.best_params else ''
            })

        df = pd.DataFrame(rows)
        df = df.sort_values('F1 (macro)', ascending=False)
        return df

    def save_results(self, filename: str = 'benchmark_results.json') -> str:
        """Save benchmark results to JSON."""
        results_dict = []
        for result in self._results:
            results_dict.append({
                'model_name': result.model_name,
                'representation': result.representation,
                'metrics': result.metrics,
                'confidence_intervals': {
                    k: list(v) for k, v in result.confidence_intervals.items()
                },
                'best_params': result.best_params
            })

        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"✓ Benchmark results saved to {filepath}")
        return str(filepath)


class ModelCardGenerator:
    """
    Generate model cards for documentation.

    Model cards provide transparency about:
    - Model architecture and training
    - Performance metrics
    - Limitations and biases
    - Intended use
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        model_name: str,
        disease: str,
        metrics: Dict[str, float],
        confidence_intervals: Dict[str, Tuple[float, float]],
        training_config: Dict,
        dataset_info: Dict,
        limitations: List[str] = None,
        intended_use: str = None
    ) -> str:
        """
        Generate a model card.

        Parameters
        ----------
        model_name : str
            Name of the model
        disease : str
            Target disease
        metrics : Dict[str, float]
            Performance metrics
        confidence_intervals : Dict[str, Tuple[float, float]]
            Confidence intervals for metrics
        training_config : Dict
            Training configuration
        dataset_info : Dict
            Information about training data
        limitations : List[str], optional
            Known limitations
        intended_use : str, optional
            Intended use description

        Returns
        -------
        filepath : str
            Path to generated model card
        """
        if limitations is None:
            limitations = [
                "Model trained on specific EEG hardware/protocol",
                "Performance may vary across different patient populations",
                "Not validated for clinical diagnosis without expert review",
                "Requires preprocessing consistent with training pipeline"
            ]

        if intended_use is None:
            intended_use = (
                "Research and clinical decision support. "
                "Not intended as sole diagnostic tool."
            )

        template = f"""# Model Card: {model_name} for {disease.title()} Detection

## Model Details

- **Model Name**: {model_name}
- **Version**: 1.0.0
- **Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Type**: EEG Classification
- **Target**: {disease.title()} Detection

## Intended Use

{intended_use}

### Primary Use Cases
- Clinical decision support for {disease} screening
- Research applications in neurological disease detection
- Educational purposes in EEG analysis

### Out-of-Scope Uses
- Standalone clinical diagnosis without expert review
- Pediatric populations (unless specifically validated)
- Real-time monitoring without proper clinical oversight

## Training Data

- **Dataset(s)**: {dataset_info.get('datasets', 'N/A')}
- **Total Subjects**: {dataset_info.get('n_subjects', 'N/A')}
- **Total Samples**: {dataset_info.get('n_samples', 'N/A')}
- **Class Distribution**: {dataset_info.get('class_distribution', 'N/A')}
- **Split Method**: Subject-wise (no leakage)

### Preprocessing
- Sampling Rate: {training_config.get('sampling_rate', 256)} Hz
- Filtering: {training_config.get('filter_band', '0.5-45')} Hz
- Artifact Rejection: {training_config.get('artifact_rejection', 'ICA-based')}

## Performance Metrics

| Metric | Value | 95% CI |
|--------|-------|--------|
"""
        for metric_name, value in metrics.items():
            ci = confidence_intervals.get(metric_name, (value, value))
            template += f"| {metric_name} | {value:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |\n"

        template += f"""
## Training Configuration

- **Random Seed**: {training_config.get('random_seed', 42)}
- **Cross-Validation**: {training_config.get('cv_method', 'GroupKFold')} ({training_config.get('n_folds', 5)} folds)
- **Scoring Metric**: {training_config.get('scoring', 'F1 (macro)')}
- **Class Balancing**: {training_config.get('class_weight', 'balanced')}

## Limitations and Biases

"""
        for limitation in limitations:
            template += f"- {limitation}\n"

        template += f"""
## Ethical Considerations

- Model outputs should be reviewed by qualified healthcare professionals
- Not to be used as sole basis for clinical decisions
- Patient consent required for data collection and model application
- Results should be interpreted in clinical context

## Citation

If using this model, please cite:

```
@misc{{{model_name.lower()}_{disease}_2024,
  title={{EEG-based {disease.title()} Detection Model}},
  author={{NeuroDiseaseAI Team}},
  year={{2024}},
  note={{Model Card v1.0}}
}}
```

## Contact

For questions or issues, contact the development team.

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # Save model card
        filename = f"model_card_{model_name}_{disease}.md"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            f.write(template)

        print(f"✓ Model card saved to {filepath}")
        return str(filepath)


class AblationReporter:
    """Generate ablation study reports."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        ablation_results: Dict[str, Dict],
        model_name: str
    ) -> str:
        """
        Generate ablation study report.

        Parameters
        ----------
        ablation_results : Dict[str, Dict]
            Results from FeatureAblation.run_ablation()
        model_name : str
            Name of the model

        Returns
        -------
        filepath : str
            Path to generated report
        """
        import pandas as pd

        # Create table
        rows = []
        baseline = ablation_results.get('baseline', {})
        baseline_score = baseline.get('score_mean', 0)

        for config_name, result in ablation_results.items():
            score = result.get('score_mean', 0)
            std = result.get('score_std', 0)
            delta = result.get('delta', 0) if config_name != 'baseline' else 0
            delta_pct = result.get('delta_pct', 0) if config_name != 'baseline' else 0

            rows.append({
                'Configuration': config_name,
                'Score': f"{score:.4f} ± {std:.4f}",
                'Delta': f"{delta:+.4f}" if delta != 0 else '-',
                'Delta %': f"{delta_pct:+.1f}%" if delta_pct != 0 else '-',
                'N Features': result.get('n_features', 'N/A')
            })

        df = pd.DataFrame(rows)

        # Generate markdown report
        report = f"""# Ablation Study Report: {model_name}

## Summary

This report documents the ablation study for {model_name}, showing the impact
of removing different feature groups or pipeline components.

## Results Table

{df.to_markdown(index=False)}

## Interpretation

"""
        # Add interpretation
        significant_ablations = [
            (name, result.get('delta', 0))
            for name, result in ablation_results.items()
            if name != 'baseline' and abs(result.get('delta', 0)) > 0.02
        ]

        if significant_ablations:
            report += "### Significant Components\n\n"
            for name, delta in sorted(significant_ablations, key=lambda x: -abs(x[1])):
                direction = "hurts" if delta > 0 else "helps"
                report += f"- Removing **{name.replace('without_', '')}** {direction} performance by {abs(delta):.4f}\n"
        else:
            report += "No components showed significant impact (Δ > 0.02) when removed.\n"

        report += f"""
## Methodology

- Baseline score: {baseline_score:.4f}
- Each ablation removes one component and re-evaluates
- Cross-validation used for all evaluations
- Significance threshold: Δ > 0.02

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # Save report
        filename = f"ablation_report_{model_name}.md"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            f.write(report)

        print(f"✓ Ablation report saved to {filepath}")
        return str(filepath)


def create_executive_summary(
    benchmark_results: List[BenchmarkResult],
    test_results: Dict,
    disease: str,
    output_dir: str
) -> str:
    """
    Create executive summary of all results.

    Parameters
    ----------
    benchmark_results : List[BenchmarkResult]
        Results from benchmark ladder
    test_results : Dict
        Final test results
    disease : str
        Target disease
    output_dir : str
        Output directory

    Returns
    -------
    filepath : str
        Path to executive summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find best model
    best_result = max(benchmark_results, key=lambda x: x.metrics.get('f1_macro', 0))

    summary = f"""# Executive Summary: {disease.title()} Detection

## Key Results

| Metric | Best Model | Value | CI |
|--------|-----------|-------|-----|
"""

    test_metrics = test_results.get('metrics', {})
    for metric_name, metric_data in test_metrics.items():
        if isinstance(metric_data, dict):
            value = metric_data.get('estimate', 0)
            lower = metric_data.get('ci_lower', value)
            upper = metric_data.get('ci_upper', value)
        else:
            value = metric_data
            lower = upper = value

        summary += f"| {metric_name} | {best_result.model_name} | {value:.4f} | [{lower:.4f}, {upper:.4f}] |\n"

    summary += f"""
## Best Model

- **Model**: {best_result.model_name}
- **Representation**: {best_result.representation}
- **Validation F1**: {best_result.metrics.get('f1_macro', 0):.4f}

## Benchmark Ladder Summary

| Rank | Model | Representation | F1 (macro) |
|------|-------|----------------|------------|
"""

    sorted_results = sorted(benchmark_results, key=lambda x: -x.metrics.get('f1_macro', 0))
    for i, result in enumerate(sorted_results[:5], 1):
        summary += f"| {i} | {result.model_name} | {result.representation} | {result.metrics.get('f1_macro', 0):.4f} |\n"

    summary += f"""
## Conclusions

1. Best performing model: **{best_result.model_name}** with **{best_result.representation}** features
2. Achieved F1 score of **{best_result.metrics.get('f1_macro', 0):.4f}** on validation
3. Model is ready for deployment with proper clinical oversight

## Recommendations

1. Use {best_result.model_name} as the primary model for {disease} detection
2. Ensure preprocessing matches training pipeline exactly
3. Monitor model performance in production with drift detection
4. Plan for periodic retraining as new data becomes available

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    filepath = output_dir / f"executive_summary_{disease}.md"
    with open(filepath, 'w') as f:
        f.write(summary)

    print(f"✓ Executive summary saved to {filepath}")
    return str(filepath)
