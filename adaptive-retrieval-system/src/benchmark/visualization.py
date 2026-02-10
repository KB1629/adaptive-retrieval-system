"""
Result visualization utilities.

This module provides tools for visualizing benchmark results:
- Comparison tables
- Performance charts
- Ablation study visualization

Requirements: 6.6, 6.7
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from ..models.results import MetricsResult, LatencyResult
from .runner import BenchmarkResult

logger = logging.getLogger(__name__)


def create_comparison_table(
    results: Dict[str, BenchmarkResult],
    metrics: List[str] = None,
) -> str:
    """
    Create ASCII comparison table for multiple benchmark results.
    
    Args:
        results: Dictionary mapping system names to BenchmarkResult
        metrics: List of metrics to include (default: all)
        
    Returns:
        ASCII table string
        
    Example:
        >>> results = {
        ...     "ColPali": colpali_result,
        ...     "Adaptive": adaptive_result,
        ... }
        >>> table = create_comparison_table(results)
        >>> print(table)
    """
    if not results:
        return "No results to display"
    
    if metrics is None:
        metrics = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg"]
    
    # Header
    header = "| System" + " " * 15 + "|"
    for metric in metrics:
        header += f" {metric:>12} |"
    
    separator = "|" + "-" * 21 + "|" + ("-" * 15 + "|") * len(metrics)
    
    # Rows
    rows = []
    for system_name, result in results.items():
        row = f"| {system_name:<20} |"
        for metric in metrics:
            value = getattr(result.metrics, metric, 0.0)
            row += f" {value:12.4f} |"
        rows.append(row)
    
    # Combine
    table = "\n".join([header, separator] + rows)
    return table


def create_latex_table(
    results: Dict[str, BenchmarkResult],
    caption: str = "Benchmark Results",
    label: str = "tab:results",
) -> str:
    """
    Create LaTeX table for publication.
    
    Args:
        results: Dictionary mapping system names to BenchmarkResult
        caption: Table caption
        label: LaTeX label for referencing
        
    Returns:
        LaTeX table string
        
    Example:
        >>> latex = create_latex_table(results, caption="Performance Comparison")
        >>> with open("results.tex", "w") as f:
        ...     f.write(latex)
    """
    if not results:
        return "% No results to display"
    
    # LaTeX table header
    latex = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l|ccccc}",
        "\\hline",
        "System & R@1 & R@5 & R@10 & MRR & NDCG \\\\",
        "\\hline",
    ]
    
    # Data rows
    for system_name, result in results.items():
        m = result.metrics
        row = (
            f"{system_name} & "
            f"{m.recall_at_1:.3f} & "
            f"{m.recall_at_5:.3f} & "
            f"{m.recall_at_10:.3f} & "
            f"{m.mrr:.3f} & "
            f"{m.ndcg:.3f} \\\\"
        )
        latex.append(row)
    
    # Table footer
    latex.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    return "\n".join(latex)


def create_latency_comparison_table(
    results: Dict[str, BenchmarkResult],
    components: List[str] = None,
) -> str:
    """
    Create comparison table for latency measurements.
    
    Args:
        results: Dictionary mapping system names to BenchmarkResult
        components: List of components to include (default: all)
        
    Returns:
        ASCII table string
    """
    if not results:
        return "No results to display"
    
    # Collect all components if not specified
    if components is None:
        components = set()
        for result in results.values():
            if result.latency:
                components.update(result.latency.keys())
        components = sorted(components)
    
    if not components:
        return "No latency data available"
    
    # Header
    header = "| System" + " " * 15 + "|"
    for component in components:
        header += f" {component:>15} |"
    
    separator = "|" + "-" * 21 + "|" + ("-" * 17 + "|") * len(components)
    
    # Rows
    rows = []
    for system_name, result in results.items():
        row = f"| {system_name:<20} |"
        for component in components:
            if result.latency and component in result.latency:
                mean_ms = result.latency[component].mean_ms
                row += f" {mean_ms:13.2f}ms |"
            else:
                row += " " * 15 + "- |"
        rows.append(row)
    
    # Combine
    table = "\n".join([header, separator] + rows)
    return table


def create_speedup_table(
    baseline_name: str,
    baseline_result: BenchmarkResult,
    optimized_results: Dict[str, BenchmarkResult],
) -> str:
    """
    Create speedup comparison table.
    
    Args:
        baseline_name: Name of baseline system
        baseline_result: Baseline benchmark result
        optimized_results: Dictionary of optimized system results
        
    Returns:
        ASCII table with speedup factors
    """
    if not baseline_result.latency:
        return "No latency data in baseline"
    
    components = list(baseline_result.latency.keys())
    
    # Header
    header = "| System" + " " * 15 + "|"
    for component in components:
        header += f" {component:>15} |"
    header += " Overall Speedup |"
    
    separator = "|" + "-" * 21 + "|" + ("-" * 17 + "|") * len(components) + "-" * 18 + "|"
    
    # Baseline row
    rows = [f"| {baseline_name:<20} |" + " " * 15 + "1.00x |" * len(components) + " " * 14 + "1.00x |"]
    
    # Optimized system rows
    for system_name, result in optimized_results.items():
        if not result.latency:
            continue
        
        row = f"| {system_name:<20} |"
        speedups = []
        
        for component in components:
            if component in result.latency and component in baseline_result.latency:
                baseline_ms = baseline_result.latency[component].mean_ms
                optimized_ms = result.latency[component].mean_ms
                speedup = baseline_ms / optimized_ms if optimized_ms > 0 else 0.0
                speedups.append(speedup)
                row += f" {speedup:13.2f}x |"
            else:
                row += " " * 15 + "- |"
        
        # Overall speedup (average)
        if speedups:
            overall_speedup = sum(speedups) / len(speedups)
            row += f" {overall_speedup:14.2f}x |"
        else:
            row += " " * 16 + "- |"
        
        rows.append(row)
    
    # Combine
    table = "\n".join([header, separator] + rows)
    return table


def create_ablation_table(
    ablation_results: Dict[str, BenchmarkResult],
    baseline_name: str,
) -> str:
    """
    Create ablation study table.
    
    Shows impact of removing/adding components.
    
    Args:
        ablation_results: Dictionary mapping configuration names to results
        baseline_name: Name of the full system configuration
        
    Returns:
        ASCII table showing ablation results
        
    Example:
        >>> ablation_results = {
        ...     "Full System": full_result,
        ...     "No Router": no_router_result,
        ...     "No LoRA": no_lora_result,
        ... }
        >>> table = create_ablation_table(ablation_results, "Full System")
    """
    if not ablation_results or baseline_name not in ablation_results:
        return "Invalid ablation results"
    
    baseline = ablation_results[baseline_name]
    
    # Header
    header = "| Configuration" + " " * 8 + "| R@1   | R@5   | R@10  | MRR   | NDCG  | Δ R@5  |"
    separator = "|" + "-" * 22 + "|" + ("-" * 7 + "|") * 6
    
    # Rows
    rows = []
    for config_name, result in ablation_results.items():
        m = result.metrics
        delta_r5 = m.recall_at_5 - baseline.metrics.recall_at_5
        
        row = (
            f"| {config_name:<21} | "
            f"{m.recall_at_1:.3f} | "
            f"{m.recall_at_5:.3f} | "
            f"{m.recall_at_10:.3f} | "
            f"{m.mrr:.3f} | "
            f"{m.ndcg:.3f} | "
            f"{delta_r5:+.3f} |"
        )
        rows.append(row)
    
    # Combine
    table = "\n".join([header, separator] + rows)
    return table


def print_summary_statistics(result: BenchmarkResult):
    """
    Print summary statistics for a benchmark result.
    
    Args:
        result: Benchmark result to summarize
    """
    print("\n" + "="*60)
    print(f"Summary: {result.config.name}")
    print("="*60)
    
    # Metrics
    print("\nRetrieval Metrics:")
    print(f"  Recall@1:  {result.metrics.recall_at_1:.4f}")
    print(f"  Recall@5:  {result.metrics.recall_at_5:.4f}")
    print(f"  Recall@10: {result.metrics.recall_at_10:.4f}")
    print(f"  MRR:       {result.metrics.mrr:.4f}")
    print(f"  NDCG:      {result.metrics.ndcg:.4f}")
    
    # Latency
    if result.latency:
        print("\nLatency (mean ± std):")
        for component, stats in result.latency.items():
            print(f"  {component}: {stats.mean_ms:.2f} ± {stats.std_ms:.2f} ms")
    
    # Throughput
    if result.throughput:
        print(f"\nThroughput: {result.throughput.pages_per_second:.2f} pages/sec")
    
    print("="*60 + "\n")


def export_results_for_plotting(
    results: Dict[str, BenchmarkResult],
    output_file: str,
):
    """
    Export results in format suitable for plotting.
    
    Creates a CSV file with all metrics for easy plotting with
    matplotlib, seaborn, or other tools.
    
    Args:
        results: Dictionary mapping system names to BenchmarkResult
        output_file: Path to output CSV file
    """
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "System", "Recall@1", "Recall@5", "Recall@10", 
            "MRR", "NDCG", "Latency_Mean", "Throughput"
        ])
        
        # Data rows
        for system_name, result in results.items():
            m = result.metrics
            
            # Average latency across components
            avg_latency = 0.0
            if result.latency:
                latencies = [stats.mean_ms for stats in result.latency.values()]
                avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            
            throughput = result.throughput.pages_per_second if result.throughput else 0.0
            
            writer.writerow([
                system_name,
                m.recall_at_1,
                m.recall_at_5,
                m.recall_at_10,
                m.mrr,
                m.ndcg,
                avg_latency,
                throughput,
            ])
    
    logger.info(f"Exported results to {output_file}")
