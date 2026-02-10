# Requirements Document

## Introduction

The Adaptive Retrieval System is a research project targeting publication in an academic conference/journal. The system optimizes latency in Technical RAG (Retrieval-Augmented Generation) through Semantic Routing and Domain-Specific Vision Fine-Tuning. The core innovation is a modality-adaptive pipeline that intelligently routes document pages to either fast text embedding or high-accuracy vision embedding based on visual complexity, achieving >90% of ColPali's retrieval accuracy with 50% less latency for technical documentation in Automotive, Aerospace, and Industrial Machinery domains.

## Glossary

- **Router**: A lightweight classifier component that analyzes page visual density and routes pages to the appropriate embedding path
- **Text_Embedding_Path**: The fast processing path using PyMuPDF for text extraction and nomic-embed-text for embedding generation (~30-50ms per page)
- **Vision_Embedding_Path**: The high-accuracy processing path using ColPali or fine-tuned SigLIP for visual document understanding (~300-400ms per page)
- **Visual_Density**: A metric combining OCR text density and image area ratio to classify page modality
- **ColPali**: State-of-the-art vision-based document retrieval model that generates ~1,000 vector embeddings per page
- **LoRA**: Low-Rank Adaptation technique for efficient fine-tuning of large models
- **Recall@K**: Retrieval metric measuring the proportion of relevant documents found in top K results
- **MRR**: Mean Reciprocal Rank - average of reciprocal ranks of first relevant result
- **NDCG**: Normalized Discounted Cumulative Gain - measures ranking quality
- **Vector_Database**: Storage system (Qdrant/LanceDB) for document embeddings enabling similarity search
- **MPS_Backend**: Metal Performance Shaders - Apple's GPU acceleration framework for PyTorch on M1/M2 Macs

## Requirements

### Requirement 1: Page Classification Router

**User Story:** As a researcher, I want to automatically classify document pages by visual complexity, so that each page is processed by the most efficient embedding method.

#### Acceptance Criteria

1. WHEN a PDF page is submitted to the Router, THE Router SHALL analyze the page and return a classification of either "text-heavy" or "visual-critical" within 50ms
2. WHEN classifying pages, THE Router SHALL achieve at least 95% recall on visual-critical pages to prevent accuracy loss from misrouting
3. WHEN a page contains more than 70% text area with minimal diagrams, THE Router SHALL classify it as "text-heavy"
4. WHEN a page contains wiring diagrams, schematics, charts, or complex visual elements, THE Router SHALL classify it as "visual-critical"
5. IF the Router encounters a corrupted or unreadable page, THEN THE Router SHALL log the error and default to the Vision_Embedding_Path to preserve accuracy
6. THE Router SHALL support both heuristic-based classification (OCR density + image area ratio) and optional ML-based classification (DistilBERT)
7. WHEN the Router processes a batch of pages, THE Router SHALL return classification results with confidence scores for each page

### Requirement 2: Text Embedding Path

**User Story:** As a researcher, I want to efficiently process text-heavy pages, so that I can achieve fast embedding generation without sacrificing retrieval quality for simple documents.

#### Acceptance Criteria

1. WHEN a page is routed to the Text_Embedding_Path, THE Text_Embedding_Path SHALL extract text using PyMuPDF and generate embeddings using nomic-embed-text
2. WHEN processing a text-heavy page, THE Text_Embedding_Path SHALL complete embedding generation within 50ms on average
3. WHEN extracting text, THE Text_Embedding_Path SHALL preserve document structure including headings, paragraphs, and lists
4. WHEN generating embeddings, THE Text_Embedding_Path SHALL produce vectors compatible with the Vector_Database schema
5. IF text extraction fails or returns empty content, THEN THE Text_Embedding_Path SHALL escalate the page to the Vision_Embedding_Path
6. THE Text_Embedding_Path SHALL support batch processing of multiple pages for throughput optimization

### Requirement 3: Vision Embedding Path

**User Story:** As a researcher, I want to accurately process visual-critical pages, so that spatial information in diagrams and schematics is preserved for retrieval.

#### Acceptance Criteria

1. WHEN a page is routed to the Vision_Embedding_Path, THE Vision_Embedding_Path SHALL generate embeddings using ColPali or fine-tuned SigLIP model
2. WHEN processing a visual-critical page, THE Vision_Embedding_Path SHALL complete embedding generation within 400ms on average
3. THE Vision_Embedding_Path SHALL preserve spatial relationships in wiring diagrams, schematics, and technical illustrations
4. WHEN generating embeddings, THE Vision_Embedding_Path SHALL produce vectors compatible with the Vector_Database schema
5. THE Vision_Embedding_Path SHALL support both off-the-shelf ColPali and LoRA fine-tuned variants
6. IF the vision model encounters an out-of-memory error, THEN THE Vision_Embedding_Path SHALL log the error and attempt processing with reduced batch size

### Requirement 4: Vector Database Integration

**User Story:** As a researcher, I want to store and retrieve document embeddings efficiently, so that I can perform similarity search across the document corpus.

#### Acceptance Criteria

1. THE Vector_Database SHALL store embeddings from both Text_Embedding_Path and Vision_Embedding_Path in a unified schema
2. WHEN storing embeddings, THE Vector_Database SHALL associate metadata including source document, page number, modality type, and processing timestamp
3. WHEN a query is submitted, THE Vector_Database SHALL return top-K similar documents with relevance scores within 100ms for corpora up to 10,000 pages
4. THE Vector_Database SHALL support both Qdrant and LanceDB backends with a common interface
5. WHEN embeddings are stored, THE Vector_Database SHALL validate vector dimensions match the expected schema
6. THE Vector_Database SHALL support incremental indexing for adding new documents without full re-indexing

### Requirement 5: Unified Retrieval Interface

**User Story:** As a researcher, I want a single interface for document retrieval, so that I can query the system without worrying about underlying embedding modalities.

#### Acceptance Criteria

1. WHEN a text query is submitted, THE Retrieval_Interface SHALL generate query embeddings and search across all stored document embeddings
2. THE Retrieval_Interface SHALL return results ranked by relevance score regardless of original embedding modality
3. WHEN returning results, THE Retrieval_Interface SHALL include document ID, page number, relevance score, and modality type
4. THE Retrieval_Interface SHALL support configurable top-K parameter (default K=10)
5. WHEN a query returns no results above threshold, THE Retrieval_Interface SHALL return an empty result set with appropriate status message
6. THE Retrieval_Interface SHALL log query latency and result count for benchmarking purposes

### Requirement 6: Benchmark and Evaluation Framework

**User Story:** As a researcher, I want to systematically evaluate system performance, so that I can compare against baselines and validate research claims.

#### Acceptance Criteria

1. THE Benchmark_Framework SHALL compute Recall@1, Recall@5, Recall@10, MRR, and NDCG metrics against ground truth datasets
2. THE Benchmark_Framework SHALL measure per-page latency, average latency, and P95 latency for each pipeline component
3. THE Benchmark_Framework SHALL measure throughput in pages per second for end-to-end processing
4. WHEN running benchmarks, THE Benchmark_Framework SHALL compare results against ColPali baseline and HPC-ColPali where available
5. THE Benchmark_Framework SHALL support evaluation on REAL-MM-RAG TechReport, TechSlides, DocVQA, and ViDoRe datasets
6. THE Benchmark_Framework SHALL generate result tables and visualizations suitable for publication
7. THE Benchmark_Framework SHALL support ablation studies by allowing selective component disabling

### Requirement 7: Data Loading and Preprocessing

**User Story:** As a researcher, I want to easily load and preprocess benchmark datasets, so that I can focus on model development rather than data wrangling.

#### Acceptance Criteria

1. THE Data_Loader SHALL support loading REAL-MM-RAG TechReport and TechSlides datasets from HuggingFace
2. THE Data_Loader SHALL support loading DocVQA and InfographicVQA datasets from HuggingFace
3. THE Data_Loader SHALL support loading ViDoRe benchmark for ColPali evaluation
4. WHEN loading datasets, THE Data_Loader SHALL normalize data into a common format with page images, ground truth queries, and relevance labels
5. THE Data_Loader SHALL support caching of downloaded datasets to avoid repeated downloads
6. THE Data_Loader SHALL provide utilities for creating train/validation/test splits with configurable ratios
7. IF a dataset download fails, THEN THE Data_Loader SHALL retry with exponential backoff and provide clear error messages

### Requirement 8: Model Fine-Tuning Pipeline

**User Story:** As a researcher, I want to fine-tune vision models on domain-specific data, so that I can improve retrieval accuracy for technical documentation.

#### Acceptance Criteria

1. THE Fine_Tuning_Pipeline SHALL support LoRA-based fine-tuning of SigLIP and ColPali models
2. THE Fine_Tuning_Pipeline SHALL be compatible with Google Colab and Kaggle free-tier T4 GPUs
3. WHEN fine-tuning, THE Fine_Tuning_Pipeline SHALL checkpoint model weights at configurable intervals
4. THE Fine_Tuning_Pipeline SHALL support synthetic QA pair generation for training data augmentation
5. WHEN training completes, THE Fine_Tuning_Pipeline SHALL export LoRA weights in a format loadable by the Vision_Embedding_Path
6. THE Fine_Tuning_Pipeline SHALL log training metrics including loss, learning rate, and validation performance
7. IF training is interrupted, THEN THE Fine_Tuning_Pipeline SHALL support resuming from the latest checkpoint

### Requirement 9: Hardware Compatibility

**User Story:** As a researcher, I want the system to run efficiently on available hardware, so that I can develop locally and scale to cloud for intensive tasks.

#### Acceptance Criteria

1. THE System SHALL support local development on MacBook M1 Pro using MPS backend for PyTorch
2. THE System SHALL support cloud fine-tuning on Google Colab and Kaggle with T4 GPU
3. WHEN running on MPS backend, THE Router and Text_Embedding_Path SHALL execute without GPU memory issues
4. THE System SHALL automatically detect available hardware and configure appropriate backends
5. WHEN GPU memory is insufficient, THE System SHALL gracefully fall back to CPU processing with appropriate warnings
6. THE System SHALL provide configuration options for batch sizes optimized for each hardware target

### Requirement 10: Experiment Tracking and Reproducibility

**User Story:** As a researcher, I want to track experiments and ensure reproducibility, so that I can validate results and prepare for publication.

#### Acceptance Criteria

1. THE System SHALL log all experiment configurations including model versions, hyperparameters, and dataset versions
2. THE System SHALL support setting random seeds for reproducible results
3. WHEN an experiment completes, THE System SHALL save results in a structured format suitable for analysis
4. THE System SHALL generate comparison tables between experiment runs
5. THE System SHALL support exporting results in formats suitable for LaTeX tables and publication figures
6. THE System SHALL maintain a changelog of significant architecture decisions and assumption changes
