"""
Latency measurement utilities.

This module provides tools for measuring and analyzing component latency:
- Per-component timing
- Statistical analysis (mean, median, P95, std dev)
- Latency profiling

Requirements: 6.2
"""

import logging
import time
from typing import Callable, Any, List
from contextlib import contextmanager
import numpy as np

from ..models.results import LatencyResult

logger = logging.getLogger(__name__)


@contextmanager
def measure_time():
    """
    Context manager for measuring execution time.
    
    Yields elapsed time in milliseconds.
    
    Example:
        >>> with measure_time() as timer:
        ...     # Do some work
        ...     result = expensive_function()
        >>> print(f"Took {timer()} ms")
    """
    start = time.perf_counter()
    elapsed_ms = lambda: (time.perf_counter() - start) * 1000
    yield elapsed_ms


def measure_latency(
    func: Callable,
    *args,
    iterations: int = 100,
    warmup: int = 5,
    **kwargs,
) -> LatencyResult:
    """
    Measure function latency over multiple iterations.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments for func
        iterations: Number of iterations to run
        warmup: Number of warmup iterations (not counted)
        **kwargs: Keyword arguments for func
        
    Returns:
        LatencyResult with statistics
        
    Example:
        >>> def my_function(x):
        ...     return x * 2
        >>> result = measure_latency(my_function, 42, iterations=100)
        >>> print(f"Mean: {result.mean_ms:.2f}ms")
    """
    if iterations <= 0:
        raise ValueError(f"iterations must be > 0, got {iterations}")
    
    logger.info(f"Measuring latency for {func.__name__} ({iterations} iterations)")
    
    # Warmup iterations
    for _ in range(warmup):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Warmup iteration failed: {e}")
    
    # Actual measurements
    measurements = []
    for i in range(iterations):
        start = time.perf_counter()
        try:
            func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            measurements.append(elapsed_ms)
        except Exception as e:
            logger.error(f"Iteration {i} failed: {e}")
            # Continue with remaining iterations
    
    if not measurements:
        logger.error("All iterations failed, returning empty result")
        return LatencyResult()
    
    # Compute statistics
    result = LatencyResult.from_measurements(measurements)
    
    logger.info(
        f"Latency for {func.__name__}: "
        f"mean={result.mean_ms:.2f}ms, median={result.median_ms:.2f}ms, "
        f"P95={result.p95_ms:.2f}ms, std={result.std_ms:.2f}ms"
    )
    
    return result


def measure_component_latency(
    components: dict[str, Callable],
    iterations: int = 100,
) -> dict[str, LatencyResult]:
    """
    Measure latency for multiple components.
    
    Args:
        components: Dictionary mapping component names to functions
        iterations: Number of iterations per component
        
    Returns:
        Dictionary mapping component names to LatencyResult
        
    Example:
        >>> components = {
        ...     "router": lambda: router.classify(page),
        ...     "text_embed": lambda: text_embedder.embed(text),
        ... }
        >>> results = measure_component_latency(components)
        >>> print(results["router"].mean_ms)
    """
    results = {}
    
    for name, func in components.items():
        logger.info(f"Measuring {name}...")
        try:
            result = measure_latency(func, iterations=iterations)
            results[name] = result
        except Exception as e:
            logger.error(f"Failed to measure {name}: {e}")
            results[name] = LatencyResult()
    
    return results


class LatencyProfiler:
    """
    Context manager for profiling latency of code sections.
    
    Example:
        >>> profiler = LatencyProfiler()
        >>> with profiler.profile("data_loading"):
        ...     data = load_data()
        >>> with profiler.profile("processing"):
        ...     result = process(data)
        >>> stats = profiler.get_statistics()
        >>> print(stats["data_loading"].mean_ms)
    """
    
    def __init__(self):
        """Initialize profiler."""
        self.measurements: dict[str, List[float]] = {}
    
    @contextmanager
    def profile(self, section_name: str):
        """
        Profile a code section.
        
        Args:
            section_name: Name of the section being profiled
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            if section_name not in self.measurements:
                self.measurements[section_name] = []
            self.measurements[section_name].append(elapsed_ms)
    
    def get_statistics(self) -> dict[str, LatencyResult]:
        """
        Get latency statistics for all profiled sections.
        
        Returns:
            Dictionary mapping section names to LatencyResult
        """
        stats = {}
        for section_name, measurements in self.measurements.items():
            stats[section_name] = LatencyResult.from_measurements(measurements)
        return stats
    
    def reset(self):
        """Clear all measurements."""
        self.measurements.clear()
    
    def print_summary(self):
        """Print summary of all measurements."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("Latency Profile Summary")
        print("="*60)
        
        for section_name, result in stats.items():
            print(f"\n{section_name}:")
            print(f"  Mean:   {result.mean_ms:8.2f} ms")
            print(f"  Median: {result.median_ms:8.2f} ms")
            print(f"  P95:    {result.p95_ms:8.2f} ms")
            print(f"  Std:    {result.std_ms:8.2f} ms")
            print(f"  Min:    {result.min_ms:8.2f} ms")
            print(f"  Max:    {result.max_ms:8.2f} ms")
            print(f"  Samples: {result.num_samples}")
        
        print("="*60 + "\n")


def compare_latencies(
    baseline: LatencyResult,
    optimized: LatencyResult,
) -> dict[str, float]:
    """
    Compare two latency results.
    
    Args:
        baseline: Baseline latency measurements
        optimized: Optimized latency measurements
        
    Returns:
        Dictionary with comparison metrics:
        - speedup: Ratio of baseline to optimized (>1 means faster)
        - reduction_percent: Percentage reduction in latency
        - mean_diff_ms: Absolute difference in mean latency
        
    Example:
        >>> baseline = LatencyResult(mean_ms=400, ...)
        >>> optimized = LatencyResult(mean_ms=200, ...)
        >>> comparison = compare_latencies(baseline, optimized)
        >>> print(f"Speedup: {comparison['speedup']:.2f}x")
        2.00x
    """
    if baseline.mean_ms == 0:
        return {
            "speedup": float('inf') if optimized.mean_ms < baseline.mean_ms else 1.0,
            "reduction_percent": 0.0,
            "mean_diff_ms": 0.0,
        }
    
    speedup = baseline.mean_ms / optimized.mean_ms if optimized.mean_ms > 0 else float('inf')
    reduction_percent = ((baseline.mean_ms - optimized.mean_ms) / baseline.mean_ms) * 100
    mean_diff_ms = baseline.mean_ms - optimized.mean_ms
    
    return {
        "speedup": speedup,
        "reduction_percent": reduction_percent,
        "mean_diff_ms": mean_diff_ms,
    }
