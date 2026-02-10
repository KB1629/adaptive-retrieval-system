"""
Dataset splitting utilities.

Provides train/validation/test splitting with configurable ratios
and ensures no data leakage between splits.

Requirements: 7.6
"""

import random
from dataclasses import dataclass
from typing import Tuple

from src.models import BenchmarkDataset, Page


@dataclass
class SplitConfig:
    """
    Configuration for dataset splitting.
    
    Attributes:
        train_ratio: Proportion for training set (0.0 to 1.0)
        val_ratio: Proportion for validation set (0.0 to 1.0)
        test_ratio: Proportion for test set (0.0 to 1.0)
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle before splitting
    """
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    shuffle: bool = True
    
    def __post_init__(self):
        """Validate split ratios."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        
        for name, ratio in [
            ("train_ratio", self.train_ratio),
            ("val_ratio", self.val_ratio),
            ("test_ratio", self.test_ratio),
        ]:
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {ratio}")


class DatasetSplitter:
    """
    Splits datasets into train/validation/test sets.
    
    Ensures no data leakage by splitting at document level
    when documents span multiple pages.
    """
    
    def __init__(self, config: SplitConfig = None):
        """Initialize splitter with configuration."""
        self.config = config or SplitConfig()
    
    def split(
        self, 
        dataset: BenchmarkDataset
    ) -> Tuple[BenchmarkDataset, BenchmarkDataset, BenchmarkDataset]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            dataset: Dataset to split
            
        Returns:
            Tuple of (train, validation, test) BenchmarkDatasets
        """
        # Group pages by document to prevent leakage
        doc_pages = self._group_by_document(dataset.pages)
        doc_ids = list(doc_pages.keys())
        
        # Shuffle if configured
        if self.config.shuffle:
            rng = random.Random(self.config.seed)
            rng.shuffle(doc_ids)
        
        # Calculate split indices
        n_docs = len(doc_ids)
        train_end = int(n_docs * self.config.train_ratio)
        val_end = train_end + int(n_docs * self.config.val_ratio)
        
        # Split document IDs
        train_docs = set(doc_ids[:train_end])
        val_docs = set(doc_ids[train_end:val_end])
        test_docs = set(doc_ids[val_end:])
        
        # Collect pages for each split
        train_pages = []
        val_pages = []
        test_pages = []
        
        for page in dataset.pages:
            doc_id = page.source_document
            if doc_id in train_docs:
                train_pages.append(page)
            elif doc_id in val_docs:
                val_pages.append(page)
            else:
                test_pages.append(page)
        
        # Split queries based on relevant pages
        train_queries = self._filter_queries(dataset.queries, dataset.labels, train_docs)
        val_queries = self._filter_queries(dataset.queries, dataset.labels, val_docs)
        test_queries = self._filter_queries(dataset.queries, dataset.labels, test_docs)
        
        # Create split datasets
        train_dataset = BenchmarkDataset(
            name=f"{dataset.name}_train",
            pages=train_pages,
            queries=train_queries[0],
            labels=train_queries[1],
        )
        
        val_dataset = BenchmarkDataset(
            name=f"{dataset.name}_val",
            pages=val_pages,
            queries=val_queries[0],
            labels=val_queries[1],
        )
        
        test_dataset = BenchmarkDataset(
            name=f"{dataset.name}_test",
            pages=test_pages,
            queries=test_queries[0],
            labels=test_queries[1],
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def _group_by_document(self, pages: list[Page]) -> dict[str, list[Page]]:
        """Group pages by their source document."""
        doc_pages = {}
        for page in pages:
            doc_id = page.source_document
            if doc_id not in doc_pages:
                doc_pages[doc_id] = []
            doc_pages[doc_id].append(page)
        return doc_pages
    
    def _filter_queries(
        self, 
        queries: list[str], 
        labels: list[str],
        doc_ids: set[str]
    ) -> Tuple[list[str], list[str]]:
        """Filter queries to only include those with labels in doc_ids."""
        filtered_queries = []
        filtered_labels = []
        
        for query, label in zip(queries, labels):
            # Label format is typically "doc_id:page_num" or just "doc_id"
            doc_id = label.split(":")[0] if ":" in label else label
            if doc_id in doc_ids:
                filtered_queries.append(query)
                filtered_labels.append(label)
        
        return filtered_queries, filtered_labels
