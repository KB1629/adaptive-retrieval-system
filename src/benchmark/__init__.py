"""
Benchmark and evaluation framework.

This module provides tools for evaluating retrieval performance:
- Retrieval metrics (Recall@K, MRR, NDCG)
- Latency measurement
- Throughput calculation
- Benchmark runner
- Result visualization

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

from .metrics import compute_recall_at_k, compute_mrr, compute_ndcg, evaluate_retrieval
from .latency import measure_latency, LatencyProfiler
from .throughput import measure_throughput, ThroughputResult
from .runner import BenchmarkRunner, BenchmarkConfig, BenchmarkResult
from .visualization import (
    create_comparison_table,
    create_latex_table,
    create_speedup_table,
    create_ablation_table,
)

__all__ = [
    # Metrics
    "compute_recall_at_k",
    "compute_mrr",
    "compute_ndcg",
    "evaluate_retrieval",
    # Latency
    "measure_latency",
    "LatencyProfiler",
    # Throughput
    "measure_throughput",
    "ThroughputResult",
    # Runner
    "BenchmarkRunner",
    "BenchmarkConfig",
    "BenchmarkResult",
    # Visualization
    "create_comparison_table",
    "create_latex_table",
    "create_speedup_table",
    "create_ablation_table",
]
