"""
Throughput measurement utilities.

This module provides tools for measuring processing throughput:
- Pages per second calculation
- End-to-end pipeline throughput
- Batch processing throughput

Requirements: 6.3
"""

import logging
import time
from typing import Callable, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ThroughputResult:
    """
    Result from throughput measurement.
    
    Attributes:
        pages_per_second: Processing throughput
        total_pages: Number of pages processed
        total_time_seconds: Total processing time
        avg_time_per_page_ms: Average time per page
    """
    pages_per_second: float
    total_pages: int
    total_time_seconds: float
    avg_time_per_page_ms: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pages_per_second": self.pages_per_second,
            "total_pages": self.total_pages,
            "total_time_seconds": self.total_time_seconds,
            "avg_time_per_page_ms": self.avg_time_per_page_ms,
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"Throughput: {self.pages_per_second:.2f} pages/sec "
            f"({self.total_pages} pages in {self.total_time_seconds:.2f}s, "
            f"avg {self.avg_time_per_page_ms:.2f}ms/page)"
        )


def measure_throughput(
    process_func: Callable[[Any], Any],
    items: List[Any],
    batch_size: int = 1,
) -> ThroughputResult:
    """
    Measure processing throughput.
    
    Args:
        process_func: Function that processes a single item or batch
        items: List of items to process
        batch_size: Number of items to process per call (1 = sequential)
        
    Returns:
        ThroughputResult with statistics
        
    Example:
        >>> def process_page(page):
        ...     return embed_page(page)
        >>> pages = load_pages()
        >>> result = measure_throughput(process_page, pages)
        >>> print(f"Throughput: {result.pages_per_second:.2f} pages/sec")
    """
    if not items:
        logger.warning("Empty items list, returning zero throughput")
        return ThroughputResult(
            pages_per_second=0.0,
            total_pages=0,
            total_time_seconds=0.0,
            avg_time_per_page_ms=0.0,
        )
    
    total_pages = len(items)
    logger.info(f"Measuring throughput for {total_pages} items (batch_size={batch_size})")
    
    start_time = time.perf_counter()
    
    # Process items
    if batch_size == 1:
        # Sequential processing
        for item in items:
            try:
                process_func(item)
            except Exception as e:
                logger.error(f"Failed to process item: {e}")
    else:
        # Batch processing
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            try:
                process_func(batch)
            except Exception as e:
                logger.error(f"Failed to process batch: {e}")
    
    total_time = time.perf_counter() - start_time
    
    # Calculate throughput
    pages_per_second = total_pages / total_time if total_time > 0 else 0.0
    avg_time_per_page_ms = (total_time * 1000) / total_pages if total_pages > 0 else 0.0
    
    result = ThroughputResult(
        pages_per_second=pages_per_second,
        total_pages=total_pages,
        total_time_seconds=total_time,
        avg_time_per_page_ms=avg_time_per_page_ms,
    )
    
    logger.info(str(result))
    
    return result


def measure_pipeline_throughput(
    pipeline_func: Callable[[List[Any]], List[Any]],
    documents: List[Any],
) -> ThroughputResult:
    """
    Measure end-to-end pipeline throughput.
    
    Args:
        pipeline_func: Function that processes a list of documents
        documents: List of documents to process
        
    Returns:
        ThroughputResult with statistics
        
    Example:
        >>> def pipeline(docs):
        ...     # Router -> Embed -> Store
        ...     return process_documents(docs)
        >>> docs = load_documents()
        >>> result = measure_pipeline_throughput(pipeline, docs)
    """
    if not documents:
        logger.warning("Empty documents list, returning zero throughput")
        return ThroughputResult(
            pages_per_second=0.0,
            total_pages=0,
            total_time_seconds=0.0,
            avg_time_per_page_ms=0.0,
        )
    
    total_pages = len(documents)
    logger.info(f"Measuring pipeline throughput for {total_pages} documents")
    
    start_time = time.perf_counter()
    
    try:
        pipeline_func(documents)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
    
    total_time = time.perf_counter() - start_time
    
    # Calculate throughput
    pages_per_second = total_pages / total_time if total_time > 0 else 0.0
    avg_time_per_page_ms = (total_time * 1000) / total_pages if total_pages > 0 else 0.0
    
    result = ThroughputResult(
        pages_per_second=pages_per_second,
        total_pages=total_pages,
        total_time_seconds=total_time,
        avg_time_per_page_ms=avg_time_per_page_ms,
    )
    
    logger.info(f"Pipeline throughput: {result}")
    
    return result


def compare_throughput(
    baseline: ThroughputResult,
    optimized: ThroughputResult,
) -> dict[str, float]:
    """
    Compare two throughput results.
    
    Args:
        baseline: Baseline throughput measurements
        optimized: Optimized throughput measurements
        
    Returns:
        Dictionary with comparison metrics:
        - speedup: Ratio of optimized to baseline throughput
        - improvement_percent: Percentage improvement
        - throughput_diff: Absolute difference in pages/sec
        
    Example:
        >>> baseline = ThroughputResult(pages_per_second=2.5, ...)
        >>> optimized = ThroughputResult(pages_per_second=7.7, ...)
        >>> comparison = compare_throughput(baseline, optimized)
        >>> print(f"Speedup: {comparison['speedup']:.2f}x")
        3.08x
    """
    if baseline.pages_per_second == 0:
        return {
            "speedup": float('inf') if optimized.pages_per_second > 0 else 1.0,
            "improvement_percent": 0.0,
            "throughput_diff": optimized.pages_per_second,
        }
    
    speedup = optimized.pages_per_second / baseline.pages_per_second
    improvement_percent = ((optimized.pages_per_second - baseline.pages_per_second) / 
                          baseline.pages_per_second) * 100
    throughput_diff = optimized.pages_per_second - baseline.pages_per_second
    
    return {
        "speedup": speedup,
        "improvement_percent": improvement_percent,
        "throughput_diff": throughput_diff,
    }


class ThroughputMonitor:
    """
    Monitor for tracking throughput over time.
    
    Example:
        >>> monitor = ThroughputMonitor()
        >>> for page in pages:
        ...     with monitor.track():
        ...         process_page(page)
        >>> result = monitor.get_result()
        >>> print(result.pages_per_second)
    """
    
    def __init__(self):
        """Initialize monitor."""
        self.start_time = None
        self.page_count = 0
    
    def start(self):
        """Start monitoring."""
        self.start_time = time.perf_counter()
        self.page_count = 0
    
    def track(self):
        """
        Context manager for tracking a single page.
        
        Example:
            >>> with monitor.track():
            ...     process_page(page)
        """
        class TrackContext:
            def __init__(self, monitor):
                self.monitor = monitor
            
            def __enter__(self):
                if self.monitor.start_time is None:
                    self.monitor.start()
                return self
            
            def __exit__(self, *args):
                self.monitor.page_count += 1
        
        return TrackContext(self)
    
    def get_result(self) -> ThroughputResult:
        """
        Get current throughput result.
        
        Returns:
            ThroughputResult with current statistics
        """
        if self.start_time is None:
            return ThroughputResult(
                pages_per_second=0.0,
                total_pages=0,
                total_time_seconds=0.0,
                avg_time_per_page_ms=0.0,
            )
        
        elapsed_time = time.perf_counter() - self.start_time
        pages_per_second = self.page_count / elapsed_time if elapsed_time > 0 else 0.0
        avg_time_per_page_ms = (elapsed_time * 1000) / self.page_count if self.page_count > 0 else 0.0
        
        return ThroughputResult(
            pages_per_second=pages_per_second,
            total_pages=self.page_count,
            total_time_seconds=elapsed_time,
            avg_time_per_page_ms=avg_time_per_page_ms,
        )
    
    def reset(self):
        """Reset monitor."""
        self.start_time = None
        self.page_count = 0
