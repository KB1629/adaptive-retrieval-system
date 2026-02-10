"""Export experiment results to various formats."""

import csv
from pathlib import Path
from typing import Any

from src.models.config import ExperimentResult


def export_to_latex(
    results: list[ExperimentResult],
    output_path: str,
    caption: str = "Experiment Results",
    label: str = "tab:results",
) -> str:
    """Export experiment results to LaTeX table format.

    Args:
        results: List of experiment results
        output_path: Path to save LaTeX file
        caption: Table caption
        label: Table label for referencing

    Returns:
        LaTeX table string
    """
    if not results:
        raise ValueError("No results to export")

    # Build LaTeX table
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "Experiment & Router & Vision Model & R@1 & R@5 & R@10 & MRR & NDCG & Latency (ms) \\\\",
        "\\midrule",
    ]

    for result in results:
        exp_id = result.config.experiment_id.replace("_", "\\_")
        router = result.config.router_type
        vision = result.config.vision_model.replace("_", "\\_")
        r1 = f"{result.metrics.recall_at_1:.3f}"
        r5 = f"{result.metrics.recall_at_5:.3f}"
        r10 = f"{result.metrics.recall_at_10:.3f}"
        mrr = f"{result.metrics.mrr:.3f}"
        ndcg = f"{result.metrics.ndcg:.3f}"
        latency = f"{result.latency.mean_ms:.1f}"

        latex_lines.append(
            f"{exp_id} & {router} & {vision} & {r1} & {r5} & {r10} & {mrr} & {ndcg} & {latency} \\\\"
        )

    latex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    latex_content = "\n".join(latex_lines)

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write(latex_content)

    return latex_content


def export_to_csv(
    results: list[ExperimentResult],
    output_path: str,
) -> None:
    """Export experiment results to CSV format.

    Args:
        results: List of experiment results
        output_path: Path to save CSV file
    """
    if not results:
        raise ValueError("No results to export")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            [
                "experiment_id",
                "router_type",
                "vision_model",
                "text_model",
                "vector_db_backend",
                "batch_size",
                "random_seed",
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "mrr",
                "ndcg",
                "mean_latency_ms",
                "median_latency_ms",
                "p95_latency_ms",
                "std_latency_ms",
                "throughput_pages_per_sec",
                "router_accuracy",
                "created_at",
                "completed_at",
            ]
        )

        # Data rows
        for result in results:
            writer.writerow(
                [
                    result.config.experiment_id,
                    result.config.router_type,
                    result.config.vision_model,
                    result.config.text_model,
                    result.config.vector_db_backend,
                    result.config.batch_size,
                    result.config.random_seed,
                    result.metrics.recall_at_1,
                    result.metrics.recall_at_5,
                    result.metrics.recall_at_10,
                    result.metrics.mrr,
                    result.metrics.ndcg,
                    result.latency.mean_ms,
                    result.latency.median_ms,
                    result.latency.p95_ms,
                    result.latency.std_ms,
                    result.throughput_pages_per_sec,
                    result.router_accuracy,
                    result.config.created_at.isoformat(),
                    result.completed_at.isoformat(),
                ]
            )


def export_comparison_table(
    results: list[ExperimentResult],
    baseline_name: str = "ColPali",
    output_path: str | None = None,
) -> str:
    """Generate comparison table with baseline.

    Args:
        results: List of experiment results
        baseline_name: Name of baseline system
        output_path: Optional path to save table

    Returns:
        Formatted comparison table string
    """
    if not results:
        raise ValueError("No results to export")

    # Build comparison table
    lines = [
        "=" * 100,
        f"{'System':<30} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'NDCG':>8} {'Latency (ms)':>15}",
        "=" * 100,
    ]

    # Add baseline (placeholder values)
    lines.append(
        f"{baseline_name:<30} {'0.850':>8} {'0.920':>8} {'0.950':>8} {'0.880':>8} {'0.910':>8} {'400.0':>15}"
    )
    lines.append("-" * 100)

    # Add experiment results
    for result in results:
        system_name = f"{result.config.router_type}/{result.config.vision_model}"[:30]
        r1 = f"{result.metrics.recall_at_1:.3f}"
        r5 = f"{result.metrics.recall_at_5:.3f}"
        r10 = f"{result.metrics.recall_at_10:.3f}"
        mrr = f"{result.metrics.mrr:.3f}"
        ndcg = f"{result.metrics.ndcg:.3f}"
        latency = f"{result.latency.mean_ms:.1f}"

        lines.append(
            f"{system_name:<30} {r1:>8} {r5:>8} {r10:>8} {mrr:>8} {ndcg:>8} {latency:>15}"
        )

    lines.append("=" * 100)

    table_content = "\n".join(lines)

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(table_content)

    return table_content
