"""Main pipeline orchestrator combining all components."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.router.heuristic import HeuristicRouter
from src.embedding.text_path import TextEmbeddingPath
from src.embedding.vision_path import VisionEmbeddingPath
from src.storage.base import VectorDBInterface
from src.storage.qdrant_backend import QdrantBackend
from src.storage.lancedb_backend import LanceDBBackend
from src.retrieval.retriever import Retriever
from src.models.data import Page, Document, EmbeddingResult
from src.models.results import ClassificationResult, QueryResult

logger = logging.getLogger(__name__)


class AdaptiveRetrievalPipeline:
    """End-to-end adaptive retrieval pipeline.
    
    Orchestrates: Router → Embedding Paths → Vector DB → Retrieval
    """

    def __init__(
        self,
        router_type: str = "heuristic",
        vector_db_backend: str = "qdrant",
        vector_db_config: dict[str, Any] | None = None,
        text_model: str = "nomic-embed-text",
        vision_model: str = "colpali",
        lora_weights_path: str | None = None,
        device: str = "auto",
    ):
        """Initialize pipeline.

        Args:
            router_type: Type of router ("heuristic" or "ml")
            vector_db_backend: Vector DB backend ("qdrant" or "lancedb")
            vector_db_config: Configuration for vector database
            text_model: Text embedding model name
            vision_model: Vision embedding model name
            lora_weights_path: Optional path to LoRA weights
            device: Device to use ("auto", "mps", "cuda", "cpu")
        """
        self.router_type = router_type
        self.vector_db_backend = vector_db_backend
        self.text_model = text_model
        self.vision_model = vision_model
        self.lora_weights_path = lora_weights_path
        self.device = device

        # Initialize components
        self._init_router()
        self._init_embedding_paths()
        self._init_vector_db(vector_db_config or {})
        self._init_retriever()

        logger.info(
            f"Pipeline initialized: router={router_type}, "
            f"vector_db={vector_db_backend}, device={device}"
        )

    def _init_router(self) -> None:
        """Initialize router component."""
        if self.router_type == "heuristic":
            self.router = HeuristicRouter()
        elif self.router_type == "ml":
            # ML router not implemented yet, fallback to heuristic
            logger.warning("ML router not implemented, using heuristic")
            self.router = HeuristicRouter()
        else:
            raise ValueError(f"Unknown router type: {self.router_type}")

    def _init_embedding_paths(self) -> None:
        """Initialize text and vision embedding paths."""
        self.text_path = TextEmbeddingPath()
        self.vision_path = VisionEmbeddingPath(
            lora_weights_path=self.lora_weights_path,
        )

    def _init_vector_db(self, config: dict[str, Any]) -> None:
        """Initialize vector database.

        Args:
            config: Vector database configuration
        """
        if self.vector_db_backend == "qdrant":
            self.vector_db: VectorDBInterface = QdrantBackend(
                collection_name=config.get("collection_name", "adaptive_retrieval"),
                host=config.get("host", "localhost"),
                port=config.get("port", 6333),
                embedding_dim=config.get("embedding_dim", 768),
            )
        elif self.vector_db_backend == "lancedb":
            self.vector_db = LanceDBBackend(
                db_path=config.get("db_path", "data/lancedb"),
                table_name=config.get("table_name", "adaptive_retrieval"),
                embedding_dim=config.get("embedding_dim", 768),
            )
        else:
            raise ValueError(f"Unknown vector DB backend: {self.vector_db_backend}")

    def _init_retriever(self) -> None:
        """Initialize retriever component."""
        self.retriever = Retriever(vector_db=self.vector_db)

    def process_document(self, document: Document) -> dict[str, Any]:
        """Process a document through the pipeline.

        Args:
            document: Document to process

        Returns:
            Processing statistics
        """
        stats = {
            "total_pages": len(document.pages),
            "text_path_count": 0,
            "vision_path_count": 0,
            "errors": 0,
        }

        for page in document.pages:
            try:
                # Classify page
                classification = self.router.classify(page.image)

                # Route to appropriate embedding path
                if classification.modality == "text-heavy":
                    embedding_result = self._process_text_path(page, document.doc_id)
                    stats["text_path_count"] += 1
                else:
                    embedding_result = self._process_vision_path(page, document.doc_id)
                    stats["vision_path_count"] += 1

                # Store embedding
                self._store_embedding(
                    embedding_result=embedding_result,
                    doc_id=document.doc_id,
                    page_number=page.page_number,
                    modality=classification.modality,
                    source_file=document.source_path,
                )

            except Exception as e:
                logger.error(
                    f"Error processing page {page.page_number} of {document.doc_id}: {e}"
                )
                stats["errors"] += 1

        logger.info(
            f"Processed document {document.doc_id}: "
            f"{stats['text_path_count']} text, "
            f"{stats['vision_path_count']} vision, "
            f"{stats['errors']} errors"
        )

        return stats

    def _process_text_path(self, page: Page, doc_id: str) -> EmbeddingResult:
        """Process page through text embedding path.

        Args:
            page: Page to process
            doc_id: Document ID

        Returns:
            Embedding result
        """
        try:
            return self.text_path.process_page(page.image)
        except Exception as e:
            logger.warning(
                f"Text path failed for {doc_id} page {page.page_number}, "
                f"escalating to vision path: {e}"
            )
            return self.vision_path.process_page(page.image)

    def _process_vision_path(self, page: Page, doc_id: str) -> EmbeddingResult:
        """Process page through vision embedding path.

        Args:
            page: Page to process
            doc_id: Document ID

        Returns:
            Embedding result
        """
        return self.vision_path.process_page(page.image)

    def _store_embedding(
        self,
        embedding_result: EmbeddingResult,
        doc_id: str,
        page_number: int,
        modality: str,
        source_file: str,
    ) -> str:
        """Store embedding in vector database.

        Args:
            embedding_result: Embedding result
            doc_id: Document ID
            page_number: Page number
            modality: Page modality
            source_file: Source file path

        Returns:
            Embedding ID
        """
        metadata = {
            "doc_id": doc_id,
            "page_number": page_number,
            "modality": modality,
            "source_file": source_file,
            "model_name": embedding_result.model_name,
        }

        return self.vector_db.insert(
            embedding=embedding_result.vector,
            metadata=metadata,
        )

    def query(self, query_text: str, top_k: int = 10) -> list[QueryResult]:
        """Execute retrieval query.

        Args:
            query_text: Query text
            top_k: Number of results to return

        Returns:
            List of query results
        """
        retrieval_result = self.retriever.query(query_text, top_k=top_k)

        # Convert to QueryResult format
        query_results = []
        for search_result in retrieval_result.results:
            query_results.append(
                QueryResult(
                    doc_id=search_result.doc_id,
                    page_number=search_result.page_number,
                    relevance_score=search_result.score,
                    modality=search_result.modality,
                    snippet=None,  # Could add snippet extraction
                )
            )

        return query_results

    def index_documents(self, documents: list[Document]) -> dict[str, Any]:
        """Index multiple documents.

        Args:
            documents: List of documents to index

        Returns:
            Overall statistics
        """
        overall_stats = {
            "total_documents": len(documents),
            "total_pages": 0,
            "text_path_count": 0,
            "vision_path_count": 0,
            "errors": 0,
        }

        for doc in documents:
            stats = self.process_document(doc)
            overall_stats["total_pages"] += stats["total_pages"]
            overall_stats["text_path_count"] += stats["text_path_count"]
            overall_stats["vision_path_count"] += stats["vision_path_count"]
            overall_stats["errors"] += stats["errors"]

        logger.info(
            f"Indexed {overall_stats['total_documents']} documents, "
            f"{overall_stats['total_pages']} pages"
        )

        return overall_stats

    def clear_index(self) -> None:
        """Clear all indexed documents."""
        # This would need to be implemented in vector DB backends
        logger.warning("Clear index not fully implemented")

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics.

        Returns:
            Pipeline statistics
        """
        return {
            "router_type": self.router_type,
            "vector_db_backend": self.vector_db_backend,
            "text_model": self.text_model,
            "vision_model": self.vision_model,
            "device": self.device,
        }
