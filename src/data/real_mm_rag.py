"""
REAL-MM-RAG dataset loader.

Loads TechReport and TechSlides datasets from HuggingFace Hub.
Source: ibm-research/REAL-MM-RAG_TechReport, ibm-research/REAL-MM-RAG_TechSlides

Requirements: 7.1, 7.4
"""

import logging
from typing import Literal, Optional

import numpy as np

from src.models import BenchmarkDataset, Page
from .loader import BaseDataLoader, DataLoaderConfig

logger = logging.getLogger(__name__)


class RealMMRAGLoader(BaseDataLoader):
    """
    Loader for REAL-MM-RAG datasets (TechReport and TechSlides).
    
    These datasets contain technical documentation pages with
    associated queries and relevance labels for retrieval evaluation.
    """
    
    # HuggingFace dataset identifiers
    DATASETS = {
        "techreport": "ibm-research/REAL-MM-RAG_TechReport",
        "techslides": "ibm-research/REAL-MM-RAG_TechSlides",
    }
    
    def __init__(
        self, 
        variant: Literal["techreport", "techslides"] = "techreport",
        config: Optional[DataLoaderConfig] = None
    ):
        """
        Initialize REAL-MM-RAG loader.
        
        Args:
            variant: Which dataset variant to load
            config: Optional loader configuration
        """
        super().__init__(config)
        self.variant = variant.lower()
        if self.variant not in self.DATASETS:
            raise ValueError(f"Unknown variant: {variant}. Must be one of {list(self.DATASETS.keys())}")
        
        self._hf_dataset_id = self.DATASETS[self.variant]
    
    @property
    def name(self) -> str:
        """Return dataset name."""
        return f"REAL-MM-RAG_{self.variant.capitalize()}"
    
    @property
    def description(self) -> str:
        """Return dataset description."""
        if self.variant == "techreport":
            return "Technical reports dataset (~2.2k pages) for multimodal RAG evaluation"
        return "Technical slides dataset (~2.6k pages) for multimodal RAG evaluation"
    
    def _load_raw(self) -> dict:
        """
        Load raw data from HuggingFace Hub.
        
        Returns:
            Raw dataset dictionary with 'pages', 'queries', 'labels'
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        logger.info(f"Loading {self._hf_dataset_id} from HuggingFace Hub...")
        
        # Load dataset - will cache automatically
        dataset = load_dataset(
            self._hf_dataset_id,
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
        
        # Process corpus (pages)
        if "corpus" in hf_dataset:
            corpus = hf_dataset["corpus"]
            for idx, item in enumerate(corpus):
                page = self._convert_page(item, idx)
                if page is not None:
                    pages.append(page)
        
        # Process queries
        if "queries" in hf_dataset:
            query_data = hf_dataset["queries"]
            for item in query_data:
                if "query" in item:
                    queries.append(item["query"])
                elif "text" in item:
                    queries.append(item["text"])
        
        # Process relevance labels (qrels)
        if "qrels" in hf_dataset:
            qrels = hf_dataset["qrels"]
            for item in qrels:
                # Format: query_id -> doc_id mapping
                if "query_id" in item and "doc_id" in item:
                    labels.append(f"{item['doc_id']}")
        
        logger.info(f"Normalized {len(pages)} pages, {len(queries)} queries")
        
        return BenchmarkDataset(
            name=self.name,
            pages=pages,
            queries=queries,
            labels=labels,
        )
    
    def _convert_page(self, item: dict, idx: int) -> Optional[Page]:
        """
        Convert a single HuggingFace item to Page.
        
        Args:
            item: HuggingFace dataset item
            idx: Item index
            
        Returns:
            Page object or None if conversion fails
        """
        try:
            # Extract image - may be PIL Image or numpy array
            image = item.get("image")
            if image is None:
                logger.warning(f"Item {idx} has no image, skipping")
                return None
            
            # Convert PIL Image to numpy array if needed
            if hasattr(image, "convert"):
                image = np.array(image.convert("RGB"))
            elif not isinstance(image, np.ndarray):
                image = np.array(image)
            
            # Extract document ID and page number
            doc_id = item.get("doc_id", item.get("document_id", f"doc_{idx}"))
            page_num = item.get("page_num", item.get("page_number", 1))
            
            # Ensure page_num is at least 1
            if isinstance(page_num, int) and page_num < 1:
                page_num = 1
            
            return Page.from_array(
                image=image,
                page_number=page_num,
                source_document=str(doc_id),
            )
        except Exception as e:
            logger.warning(f"Failed to convert item {idx}: {e}")
            return None


def load_techreport(config: Optional[DataLoaderConfig] = None) -> BenchmarkDataset:
    """
    Convenience function to load TechReport dataset.
    
    Args:
        config: Optional loader configuration
        
    Returns:
        BenchmarkDataset with TechReport data
    """
    loader = RealMMRAGLoader(variant="techreport", config=config)
    return loader.load()


def load_techslides(config: Optional[DataLoaderConfig] = None) -> BenchmarkDataset:
    """
    Convenience function to load TechSlides dataset.
    
    Args:
        config: Optional loader configuration
        
    Returns:
        BenchmarkDataset with TechSlides data
    """
    loader = RealMMRAGLoader(variant="techslides", config=config)
    return loader.load()
