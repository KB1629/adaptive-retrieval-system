"""Tests for experiment tracking and reproducibility."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from src.experiment.tracker import (
    ExperimentTracker,
    set_random_seed,
)
from src.experiment.export import (
    export_to_latex,
    export_to_csv,
    export_comparison_table,
)
from src.models.config import ExperimentConfig, ExperimentResult
from src.models.results import MetricsResult, LatencyResult


# Fixtures


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def tracker(temp_output_dir):
    """Create experiment tracker with temp directory."""
    return ExperimentTracker(output_dir=temp_output_dir)


@pytest.fixture
def sample_config():
    """Create sample experiment configuration."""
    return ExperimentConfig(
        experiment_id="exp_test_001",
        router_type="heuristic",
        vision_model="colpali",
        text_model="nomic-embed-text",
        lora_weights_path=None,
        vector_db_backend="qdrant",
        batch_size=32,
        random_seed=42,
        created_at=datetime.now(),
    )


@pytest.fixture
def sample_result(sample_config):
    """Create sample experiment result."""
    metrics = MetricsResult(
        recall_at_1=0.85,
        recall_at_5=0.92,
        recall_at_10=0.95,
        mrr=0.88,
        ndcg=0.91,
    )

    latency = LatencyResult(
        mean_ms=250.0,
        median_ms=240.0,
        p95_ms=350.0,
        std_ms=45.0,
    )

    return ExperimentResult(
        config=sample_config,
        metrics=metrics,
        latency=latency,
        throughput_pages_per_sec=4.0,
        router_accuracy=0.96,
        completed_at=datetime.now(),
    )


# Unit Tests


def test_create_experiment(tracker):
    """Test experiment creation."""
    config = tracker.create_experiment(
        router_type="heuristic",
        vision_model="colpali",
        text_model="nomic-embed-text",
        vector_db_backend="qdrant",
        batch_size=32,
        random_seed=42,
    )

    assert config.experiment_id.startswith("exp_")
    assert config.router_type == "heuristic"
    assert config.vision_model == "colpali"
    assert config.random_seed == 42


def test_save_and_load_config(tracker, sample_config):
    """Test configuration save and load."""
    # Save
    config_path = tracker.save_config(sample_config)
    assert config_path.exists()

    # Load
    loaded_config = tracker.load_config(sample_config.experiment_id)

    assert loaded_config.experiment_id == sample_config.experiment_id
    assert loaded_config.router_type == sample_config.router_type
    assert loaded_config.vision_model == sample_config.vision_model
    assert loaded_config.random_seed == sample_config.random_seed


def test_save_and_load_result(tracker, sample_result):
    """Test result save and load."""
    # Save
    result_path = tracker.save_result(sample_result)
    assert result_path.exists()

    # Load
    loaded_result = tracker.load_result(sample_result.config.experiment_id)

    assert loaded_result.config.experiment_id == sample_result.config.experiment_id
    assert loaded_result.metrics.recall_at_1 == sample_result.metrics.recall_at_1
    assert loaded_result.latency.mean_ms == sample_result.latency.mean_ms
    assert loaded_result.throughput_pages_per_sec == sample_result.throughput_pages_per_sec


def test_list_experiments(tracker, sample_config):
    """Test listing experiments."""
    # Initially empty
    assert len(tracker.list_experiments()) == 0

    # Save config
    tracker.save_config(sample_config)

    # Should have one experiment
    experiments = tracker.list_experiments()
    assert len(experiments) == 1
    assert sample_config.experiment_id in experiments


def test_set_random_seed():
    """Test random seed setting."""
    seed = 42

    # Set seed
    set_random_seed(seed)

    # Generate random numbers
    py_rand1 = np.random.rand()
    torch_rand1 = torch.rand(1).item()

    # Reset seed
    set_random_seed(seed)

    # Should get same numbers
    py_rand2 = np.random.rand()
    torch_rand2 = torch.rand(1).item()

    assert py_rand1 == py_rand2
    assert torch_rand1 == torch_rand2


def test_export_to_latex(sample_result, temp_output_dir):
    """Test LaTeX export."""
    output_path = Path(temp_output_dir) / "results.tex"

    latex_content = export_to_latex(
        results=[sample_result],
        output_path=str(output_path),
        caption="Test Results",
        label="tab:test",
    )

    # Check file created
    assert output_path.exists()

    # Check LaTeX structure
    assert "\\begin{table}" in latex_content
    assert "\\end{table}" in latex_content
    assert "\\begin{tabular}" in latex_content
    assert "\\end{tabular}" in latex_content
    assert "\\caption{Test Results}" in latex_content
    assert "\\label{tab:test}" in latex_content

    # Check data present
    assert "0.850" in latex_content  # recall_at_1
    assert "250.0" in latex_content  # mean latency


def test_export_to_csv(sample_result, temp_output_dir):
    """Test CSV export."""
    output_path = Path(temp_output_dir) / "results.csv"

    export_to_csv(
        results=[sample_result],
        output_path=str(output_path),
    )

    # Check file created
    assert output_path.exists()

    # Read and verify content
    with open(output_path) as f:
        lines = f.readlines()

    # Check header
    assert "experiment_id" in lines[0]
    assert "recall_at_1" in lines[0]

    # Check data
    assert sample_result.config.experiment_id in lines[1]
    assert "0.85" in lines[1]


def test_export_comparison_table(sample_result, temp_output_dir):
    """Test comparison table export."""
    output_path = Path(temp_output_dir) / "comparison.txt"

    table_content = export_comparison_table(
        results=[sample_result],
        baseline_name="ColPali",
        output_path=str(output_path),
    )

    # Check file created
    assert output_path.exists()

    # Check table structure
    assert "System" in table_content
    assert "R@1" in table_content
    assert "Latency" in table_content

    # Check baseline present
    assert "ColPali" in table_content

    # Check experiment present
    assert "0.850" in table_content


def test_export_empty_results_raises_error(temp_output_dir):
    """Test that exporting empty results raises error."""
    output_path = Path(temp_output_dir) / "results.tex"

    with pytest.raises(ValueError, match="No results to export"):
        export_to_latex(results=[], output_path=str(output_path))

    with pytest.raises(ValueError, match="No results to export"):
        export_to_csv(results=[], output_path=str(output_path))

    with pytest.raises(ValueError, match="No results to export"):
        export_comparison_table(results=[], output_path=str(output_path))


# Property-Based Tests


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10000))
def test_property_18_experiment_reproducibility_with_seeds(seed):
    """Property 18: Experiment Reproducibility with Seeds.

    Feature: adaptive-retrieval-system, Property 18: For any experiment configuration
    with a fixed random seed, running the experiment twice SHALL produce identical
    results (same metrics, same model outputs for same inputs).

    Validates: Requirements 10.2
    """
    # Set seed and generate random values
    set_random_seed(seed)
    np_values_1 = [np.random.rand() for _ in range(5)]
    torch_values_1 = [torch.rand(1).item() for _ in range(5)]

    # Reset seed and generate again
    set_random_seed(seed)
    np_values_2 = [np.random.rand() for _ in range(5)]
    torch_values_2 = [torch.rand(1).item() for _ in range(5)]

    # Should be identical
    assert np_values_1 == np_values_2
    assert torch_values_1 == torch_values_2


@settings(max_examples=10, deadline=None)
@given(
    router_type=st.sampled_from(["heuristic", "ml"]),
    vision_model=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))),
    text_model=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))),
    batch_size=st.integers(min_value=1, max_value=128),
    random_seed=st.integers(min_value=0, max_value=10000),
)
def test_property_19_experiment_persistence_completeness(
    router_type, vision_model, text_model, batch_size, random_seed
):
    """Property 19: Experiment Persistence Completeness.

    Feature: adaptive-retrieval-system, Property 19: For any completed experiment,
    the saved results SHALL include the complete ExperimentConfig (experiment_id,
    router_type, vision_model, text_model, hyperparameters) and ExperimentResult
    (metrics, latency, throughput).

    Validates: Requirements 10.1, 10.3
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(output_dir=tmpdir)

        # Create experiment
        config = tracker.create_experiment(
            router_type=router_type,
            vision_model=vision_model,
            text_model=text_model,
            vector_db_backend="qdrant",
            batch_size=batch_size,
            random_seed=random_seed,
        )

        # Create result
        metrics = MetricsResult(
            recall_at_1=0.85,
            recall_at_5=0.92,
            recall_at_10=0.95,
            mrr=0.88,
            ndcg=0.91,
        )

        latency = LatencyResult(
            mean_ms=250.0,
            median_ms=240.0,
            p95_ms=350.0,
            std_ms=45.0,
        )

        result = ExperimentResult(
            config=config,
            metrics=metrics,
            latency=latency,
            throughput_pages_per_sec=4.0,
            router_accuracy=0.96,
            completed_at=datetime.now(),
        )

        # Save result
        result_path = tracker.save_result(result)

        # Load and verify completeness
        loaded_result = tracker.load_result(config.experiment_id)

        # Verify all config fields
        assert loaded_result.config.experiment_id == config.experiment_id
        assert loaded_result.config.router_type == router_type
        assert loaded_result.config.vision_model == vision_model
        assert loaded_result.config.text_model == text_model
        assert loaded_result.config.batch_size == batch_size
        assert loaded_result.config.random_seed == random_seed

        # Verify all result fields
        assert loaded_result.metrics.recall_at_1 == metrics.recall_at_1
        assert loaded_result.metrics.recall_at_5 == metrics.recall_at_5
        assert loaded_result.metrics.recall_at_10 == metrics.recall_at_10
        assert loaded_result.metrics.mrr == metrics.mrr
        assert loaded_result.metrics.ndcg == metrics.ndcg

        assert loaded_result.latency.mean_ms == latency.mean_ms
        assert loaded_result.latency.median_ms == latency.median_ms
        assert loaded_result.latency.p95_ms == latency.p95_ms
        assert loaded_result.latency.std_ms == latency.std_ms

        assert loaded_result.throughput_pages_per_sec == result.throughput_pages_per_sec
        assert loaded_result.router_accuracy == result.router_accuracy


@settings(max_examples=10, deadline=None)
@given(
    num_results=st.integers(min_value=1, max_value=5),
)
def test_property_20_latex_export_validity(num_results):
    """Property 20: LaTeX Export Validity.

    Feature: adaptive-retrieval-system, Property 20: For any experiment results
    exported to LaTeX format, the output SHALL be valid LaTeX that compiles without
    errors and produces a properly formatted table.

    Validates: Requirements 10.5
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample results
        results = []
        for i in range(num_results):
            config = ExperimentConfig(
                experiment_id=f"exp_test_{i:03d}",
                router_type="heuristic",
                vision_model="colpali",
                text_model="nomic-embed-text",
                lora_weights_path=None,
                vector_db_backend="qdrant",
                batch_size=32,
                random_seed=42,
                created_at=datetime.now(),
            )

            metrics = MetricsResult(
                recall_at_1=0.80 + i * 0.01,
                recall_at_5=0.90 + i * 0.01,
                recall_at_10=0.95 + i * 0.005,
                mrr=0.85 + i * 0.01,
                ndcg=0.88 + i * 0.01,
            )

            latency = LatencyResult(
                mean_ms=200.0 + i * 10,
                median_ms=190.0 + i * 10,
                p95_ms=300.0 + i * 20,
                std_ms=40.0 + i * 5,
            )

            result = ExperimentResult(
                config=config,
                metrics=metrics,
                latency=latency,
                throughput_pages_per_sec=4.0 - i * 0.1,
                router_accuracy=0.95 + i * 0.005,
                completed_at=datetime.now(),
            )

            results.append(result)

        # Export to LaTeX
        output_path = Path(tmpdir) / "results.tex"
        latex_content = export_to_latex(
            results=results,
            output_path=str(output_path),
            caption="Test Results",
            label="tab:test",
        )

        # Verify LaTeX structure
        assert latex_content.startswith("\\begin{table}")
        assert latex_content.endswith("\\end{table}")
        assert "\\begin{tabular}" in latex_content
        assert "\\end{tabular}" in latex_content
        assert "\\toprule" in latex_content
        assert "\\midrule" in latex_content
        assert "\\bottomrule" in latex_content

        # Verify all results present
        for result in results:
            assert result.config.experiment_id.replace("_", "\\_") in latex_content

        # Verify table has correct number of data rows (count lines with \\\\)
        lines = latex_content.split("\n")
        data_lines = [line for line in lines if "\\\\" in line and "toprule" not in line and "midrule" not in line and "bottomrule" not in line]
        # Subtract 1 for header row
        data_rows = len(data_lines) - 1
        assert data_rows == num_results
