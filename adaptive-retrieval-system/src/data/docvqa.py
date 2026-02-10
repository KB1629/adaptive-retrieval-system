"""
DocVQA and InfographicVQA dataset loaders.

Loads document visual question answering datasets from HuggingFace Hub.
Source: lmms-lab/DocVQA

Requirements: 7.2, 7.4
"""

import logging
from typing import Literal, Optional

import numpy as np

from src.models import BenchmarkDataset, Page
from .loader import BaseDataLoader, DataLoaderConfig

logger = logging.getLogger(__name__)


class DocVQALoader(BaseDataLoader):
    """
    Loader for DocVQA dataset.
    
    DocVQA contains document images with question-answer pairs
    for visual question answering evaluation.
    """
    
    HF_DATASET_ID = "lmms-lab/DocVQA"
    
    def __init__(
        self,
        split: Literal["train", "validation", "test"] = "validation",
        config: Optional[DataLoaderConfig] = None
    ):
        """
        Initialize DocVQA loader.
        
        Args:
            split: Which split to load
            config: Optional loader configuration
        """
        super().__init__(config)
        self.split = split
    
    @property
    def name(self) -> str:
        """Return dataset name."""
        return f"DocVQA_{self.split}"
    
    @property
    def description(self) -> str:
        """Return dataset description."""
        return "Document Visual Question Answering dataset (~10.5k samples)"
    
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
        
        logger.info(f"Loading {self.HF_DATASET_ID} ({self.split}) from HuggingFace Hub...")
        
        dataset = load_dataset(
            self.HF_DATASET_ID,
            split=self.split,
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
        seen_images = set()  # Track unique images
        
        for idx, item in enumerate(hf_dataset):
            # Extract image
            image = item.get("image")
            if image is None:
                continue
            
            # Convert PIL Image to numpy array
            if hasattr(image, "convert"):
                image_array = np.array(image.convert("RGB"))
            else:
                image_array = np.array(image)
            
            # Create unique document ID based on image hash or index
            doc_id = item.get("questionId", item.get("question_id", f"docvqa_{idx}"))
            
            # Only add unique pages
            image_key = f"{doc_id}"
            if image_key not in seen_images:
                seen_images.add(image_key)
                page = Page.from_array(
                    image=image_array,
                    page_number=1,
                    source_document=str(doc_id),
                )
                pages.append(page)
            
            # Extract question and answer
            question = item.get("question", "")
            answers = item.get("answers", item.get("answer", []))
            
            if question:
                queries.append(question)
                # Use first answer as label
                if isinstance(answers, list) and answers:
                    labels.append(str(answers[0]))
                elif isinstance(answers, str):
                    labels.append(answers)
                else:
                    labels.append(str(doc_id))
        
        logger.info(f"Normalized {len(pages)} pages, {len(queries)} queries")
        
        return BenchmarkDataset(
            name=self.name,
            pages=pages,
            queries=queries,
            labels=labels,
        )


class InfographicVQALoader(BaseDataLoader):
    """
    Loader for InfographicVQA dataset.
    
    Contains infographic images with question-answer pairs,
    useful for testing visual understanding of charts and diagrams.
    """
    
    HF_DATASET_ID = "lmms-lab/InfographicVQA"
    
    def __init__(
        self,
        split: Literal["train", "validation", "test"] = "validation",
        config: Optional[DataLoaderConfig] = None
    ):
        """
        Initialize InfographicVQA loader.
        
        Args:
            split: Which split to load
            config: Optional loader configuration
        """
        super().__init__(config)
        self.split = split
    
    @property
    def name(self) -> str:
        """Return dataset name."""
        return f"InfographicVQA_{self.split}"
    
    @property
    def description(self) -> str:
        """Return dataset description."""
        return "Infographic Visual Question Answering dataset (~6k samples)"
    
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
        
        logger.info(f"Loading {self.HF_DATASET_ID} ({self.split}) from HuggingFace Hub...")
        
        dataset = load_dataset(
            self.HF_DATASET_ID,
            split=self.split,
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
        
        for idx, item in enumerate(hf_dataset):
            image = item.get("image")
            if image is None:
                continue
            
            if hasattr(image, "convert"):
                image_array = np.array(image.convert("RGB"))
            else:
                image_array = np.array(image)
            
            doc_id = item.get("questionId", item.get("question_id", f"infovqa_{idx}"))
            
            image_key = f"{doc_id}"
            if image_key not in seen_images:
                seen_images.add(image_key)
                page = Page.from_array(
                    image=image_array,
                    page_number=1,
                    source_document=str(doc_id),
                )
                pages.append(page)
            
            question = item.get("question", "")
            answers = item.get("answers", item.get("answer", []))
            
            if question:
                queries.append(question)
                if isinstance(answers, list) and answers:
                    labels.append(str(answers[0]))
                elif isinstance(answers, str):
                    labels.append(answers)
                else:
                    labels.append(str(doc_id))
        
        logger.info(f"Normalized {len(pages)} pages, {len(queries)} queries")
        
        return BenchmarkDataset(
            name=self.name,
            pages=pages,
            queries=queries,
            labels=labels,
        )


def load_docvqa(
    split: str = "validation",
    config: Optional[DataLoaderConfig] = None
) -> BenchmarkDataset:
    """
    Convenience function to load DocVQA dataset.
    
    Args:
        split: Which split to load
        config: Optional loader configuration
        
    Returns:
        BenchmarkDataset with DocVQA data
    """
    loader = DocVQALoader(split=split, config=config)
    return loader.load()


def load_infographicvqa(
    split: str = "validation",
    config: Optional[DataLoaderConfig] = None
) -> BenchmarkDataset:
    """
    Convenience function to load InfographicVQA dataset.
    
    Args:
        split: Which split to load
        config: Optional loader configuration
        
    Returns:
        BenchmarkDataset with InfographicVQA data
    """
    loader = InfographicVQALoader(split=split, config=config)
    return loader.load()
