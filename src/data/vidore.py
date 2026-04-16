"""
ViDoRe benchmark loader.

Loads the Visual Document Retrieval benchmark used for ColPali evaluation.
Source: vidore/vidore-benchmark on HuggingFace

Requirements: 7.3, 7.4
"""

import logging
from typing import Literal, Optional

import numpy as np

from src.models import BenchmarkDataset, Page
from .loader import BaseDataLoader, DataLoaderConfig

logger = logging.getLogger(__name__)


# Available ViDoRe benchmark subsets
VIDORE_SUBSETS = [
    "vidore/arxivqa_test_subsampled",
    "vidore/docvqa_test_subsampled", 
    "vidore/infovqa_test_subsampled",
    "vidore/tabfquad_test_subsampled",
    "vidore/tatdqa_test",
    "vidore/shiftproject_test",
]


class ViDoReLoader(BaseDataLoader):
    """
    Loader for ViDoRe benchmark datasets.
    
    ViDoRe (Visual Document Retrieval) is the benchmark used to
    evaluate ColPali and similar vision-based retrieval models.
    """
    
    def __init__(
        self,
        subset: str = "vidore/docvqa_test_subsampled",
        config: Optional[DataLoaderConfig] = None
    ):
        """
        Initialize ViDoRe loader.
        
        Args:
            subset: Which ViDoRe subset to load
            config: Optional loader configuration
        """
        super().__init__(config)
        self.subset = subset
        
        # Validate subset
        if subset not in VIDORE_SUBSETS:
            logger.warning(
                f"Unknown subset: {subset}. Known subsets: {VIDORE_SUBSETS}. "
                "Attempting to load anyway..."
            )
    
    @property
    def name(self) -> str:
        """Return dataset name."""
        # Extract short name from full path
        short_name = self.subset.split("/")[-1] if "/" in self.subset else self.subset
        return f"ViDoRe_{short_name}"
    
    @property
    def description(self) -> str:
        """Return dataset description."""
        return f"ViDoRe benchmark subset: {self.subset}"
    
    def _load_raw(self) -> dict:
        """
        Load raw data from HuggingFace Hub.
        
        Returns:
            Raw dataset dictionary
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        logger.info(f"Loading {self.subset} from HuggingFace Hub...")
        
        dataset = load_dataset(
            self.subset,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
        )
        
        return {"hf_dataset": dataset}
    
    def _normalize(self, raw_data: dict) -> BenchmarkDataset:
        """
        Normalize HuggingFace dataset to BenchmarkDataset format.
        
        Args:
            raw_data: Dictionary containing 'hf_dataset'
            
        Returns:
            Normalized BenchmarkDataset
        """
        hf_dataset = raw_data["hf_dataset"]
        
        pages = []
        queries = []
        labels = []
        seen_images = set()
        
        # ViDoRe datasets typically have a 'test' split
        split_data = hf_dataset.get("test", hf_dataset)
        if hasattr(split_data, "__iter__"):
            data_iter = split_data
        else:
            # If it's a DatasetDict, get the first available split
            for split_name in ["test", "train", "validation"]:
                if split_name in hf_dataset:
                    data_iter = hf_dataset[split_name]
                    break
            else:
                data_iter = list(hf_dataset.values())[0] if hf_dataset else []
        
        for idx, item in enumerate(data_iter):
            # Extract image
            image = item.get("image")
            if image is None:
                continue
            
            if hasattr(image, "convert"):
                image_array = np.array(image.convert("RGB"))
            else:
                image_array = np.array(image)
            
            # Extract document ID
            doc_id = item.get("docid", item.get("doc_id", f"vidore_{idx}"))
            
            # Only add unique pages
            if doc_id not in seen_images:
                seen_images.add(doc_id)
                page = Page.from_array(
                    image=image_array,
                    page_number=1,
                    source_document=str(doc_id),
                )
                pages.append(page)
            
            # Extract query
            query = item.get("query", item.get("question", ""))
            if query:
                queries.append(query)
                # Label is the relevant document ID
                labels.append(str(doc_id))
        
        logger.info(f"Normalized {len(pages)} pages, {len(queries)} queries")
        
        return BenchmarkDataset(
            name=self.name,
            pages=pages,
            queries=queries,
            labels=labels,
        )


def load_vidore(
    subset: str = "vidore/docvqa_test_subsampled",
    config: Optional[DataLoaderConfig] = None
) -> BenchmarkDataset:
    """
    Convenience function to load ViDoRe benchmark subset.
    
    Args:
        subset: Which ViDoRe subset to load
        config: Optional loader configuration
        
    Returns:
        BenchmarkDataset with ViDoRe data
    """
    loader = ViDoReLoader(subset=subset, config=config)
    return loader.load()


def load_all_vidore(config: Optional[DataLoaderConfig] = None) -> list[BenchmarkDataset]:
    """
    Load all ViDoRe benchmark subsets.
    
    Args:
        config: Optional loader configuration
        
    Returns:
        List of BenchmarkDatasets for all ViDoRe subsets
    """
    datasets = []
    for subset in VIDORE_SUBSETS:
        try:
            dataset = load_vidore(subset=subset, config=config)
            datasets.append(dataset)
        except Exception as e:
            logger.warning(f"Failed to load {subset}: {e}")
    return datasets


def get_available_subsets() -> list[str]:
    """Return list of available ViDoRe subsets."""
    return VIDORE_SUBSETS.copy()
