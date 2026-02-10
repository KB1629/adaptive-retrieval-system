# Implementation Plan: Adaptive Retrieval System

## Overview

This implementation plan breaks down the Adaptive Retrieval System into discrete coding tasks organized by phase. Each task builds incrementally on previous work, with property-based tests integrated close to implementation to catch errors early. The plan follows the research project phases while ensuring all 10 requirements and 20 correctness properties are covered.

**Code Philosophy**: Keep code simple, readable, and well-organized. Avoid over-engineering. Each module should do one thing well.

**Testing Strategy**: Property-based tests (Hypothesis, 100+ iterations) for universal properties; unit tests for specific examples and edge cases.

## Project Structure

```
adaptive-retrieval-system/
├── README.md                    # Project overview, updated at milestones
├── CHANGELOG.md                 # Architecture decisions and changes
├── pyproject.toml               # Dependencies and project config
├── config.yaml                  # Default configuration
├── references/                  # Research papers and baselines
│   ├── README.md                # Paper summaries and links
│   ├── colpali.pdf              # Main baseline (arXiv:2407.01449)
│   ├── hpc-colpali.pdf          # Competition (arXiv:2506.21601)
│   ├── real-mm-rag.pdf          # Benchmark dataset (arXiv:2502.12342)
│   └── multimodal-rag-comparison.pdf  # Validation (arXiv:2511.16654)
├── data/                        # Downloaded datasets (auto-cached)
│   ├── .gitkeep
│   └── cache/                   # HuggingFace cache directory
├── src/                         # Source code (simple, modular)
│   ├── __init__.py
│   ├── cli.py                   # Command-line interface
│   ├── models/                  # Data structures
│   ├── router/                  # Page classification
│   ├── embedding/               # Text and vision paths
│   ├── storage/                 # Vector database
│   ├── retrieval/               # Query interface
│   ├── benchmark/               # Evaluation framework
│   ├── finetuning/              # LoRA training
│   ├── experiment/              # Tracking and export
│   ├── pipeline/                # Orchestration
│   └── utils/                   # Hardware, logging
├── tests/                       # All tests
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures
│   ├── test_router.py
│   ├── test_embedding.py
│   ├── test_storage.py
│   ├── test_retrieval.py
│   ├── test_benchmark.py
│   └── test_integration.py
├── notebooks/                   # Jupyter notebooks
│   ├── finetuning.ipynb         # Colab/Kaggle training
│   └── benchmark.ipynb          # Evaluation and viz
├── configs/                     # Configuration variants
│   ├── dev.yaml                 # Local development
│   ├── colab.yaml               # Colab fine-tuning
│   └── benchmark.yaml           # Full evaluation
└── outputs/                     # Experiment results
    ├── checkpoints/             # Model checkpoints
    ├── results/                 # Benchmark results
    └── figures/                 # Generated visualizations
```

## Tasks

- [x] 1. Project Setup and Research References
  - [x] 1.1 Create project directory structure and configuration files
  - [x] 1.2 Set up research references directory
  - [x] 1.3 Implement hardware detection and backend configuration
  - [x] 1.4 Write property test for hardware detection
  - [x] 1.5 Update README with setup instructions

- [x] 2. Data Models and Core Types
  - [x] 2.1 Implement core data structures
  - [x] 2.2 Write property test for data model serialization

- [x] 3. Checkpoint - Ensure all tests pass

- [x] 4. Data Loading and Preprocessing
  - [x] 4.1 Implement base data loader interface
  - [x] 4.2 Implement REAL-MM-RAG dataset loader
  - [x] 4.3 Implement DocVQA and InfographicVQA loaders
  - [x] 4.4 Implement ViDoRe benchmark loader
  - [x] 4.5 Implement dataset splitting utilities
  - [x] 4.6 Write property tests for data loading
  - [x] 4.7 Update README with data loading instructions

- [x] 5. Checkpoint - Ensure all tests pass

- [x] 6. Router Component
  - [x] 6.1 Implement heuristic router
  - [x] 6.2 Implement router interface and batch processing
  - [x] 6.3 Implement optional ML-based router (DistilBERT)
  - [x] 6.4 Write property tests for router

- [x] 7. Checkpoint - Ensure all tests pass and update README

- [x] 8. Text Embedding Path
  - [x] 8.1 Implement text extraction with PyMuPDF
  - [x] 8.2 Implement nomic-embed-text embedding
  - [x] 8.3 Implement TextEmbeddingPath pipeline
  - [x] 8.4 Write property tests for text embedding path

- [x] 9. Vision Embedding Path
  - [x] 9.1 Implement ColPali embedding
  - [x] 9.2 Implement LoRA weight loading
  - [x] 9.3 Implement VisionEmbeddingPath pipeline
  - [x] 9.4 Write property tests for vision embedding path

- [x] 10. Checkpoint - Ensure all tests pass and update README

- [x] 11. Vector Database Integration
  - [x] 11.1 Implement vector database interface
  - [x] 11.2 Implement Qdrant backend
  - [x] 11.3 Implement LanceDB backend
  - [x] 11.4 Implement incremental indexing
  - [x] 11.5 Write property tests for vector database

- [x] 12. Unified Retrieval Interface
  - [x] 12.1 Implement query encoder
  - [x] 12.2 Implement retrieval interface
  - [x] 12.3 Implement result formatting and logging
  - [x] 12.4 Write property tests for retrieval

- [x] 13. Checkpoint - Ensure all tests pass and update README

- [x] 14. Benchmark and Evaluation Framework
  - [x] 14.1 Implement retrieval metrics
  - [x] 14.2 Implement latency measurement
  - [x] 14.3 Implement throughput measurement
  - [x] 14.4 Implement benchmark runner
  - [x] 14.5 Implement result visualization
  - [x] 14.6 Write property tests for benchmark framework

- [x] 15. Checkpoint - Ensure all tests pass and update README

- [x] 16. Fine-Tuning Pipeline
  - [x] 16.1 Implement LoRA fine-tuning setup
  - [x] 16.2 Implement checkpointing
  - [x] 16.3 Implement synthetic QA generation
  - [x] 16.4 Implement LoRA weight export
  - [x] 16.5 Write property tests for fine-tuning

- [x] 17. Experiment Tracking and Reproducibility
  - [x] 17.1 Implement experiment configuration logging
  - [x] 17.2 Implement random seed management
  - [x] 17.3 Implement result persistence
  - [x] 17.4 Implement LaTeX export
  - [x] 17.5 Create changelog template
  - [x] 17.6 Write property tests for experiment tracking

- [x] 18. Pipeline Integration
  - [x] 18.1 Implement main pipeline orchestrator
  - [x] 18.2 Implement CLI interface
  - [x] 18.3 Write integration tests

- [x] 19. Final Checkpoint - Ensure all tests pass

- [x] 20. Documentation and Notebooks
  - [x] 20.1 Create Colab/Kaggle fine-tuning notebook
  - [x] 20.2 Create benchmark notebook
  - [x] 20.3 Update README with usage instructions


## Task Details

### Task 1: Project Setup and Research References

#### 1.1 Create project directory structure and configuration files
- Create directory structure as shown above
- Set up `pyproject.toml` with dependencies (torch, transformers, pymupdf, qdrant-client, lancedb, hypothesis)
- Create `config.yaml` with default configuration schema
- Initialize `README.md` with project overview
- **Requirements**: 9.1, 9.4

#### 1.2 Set up research references directory
- Create `references/README.md` with paper summaries and arXiv links
- Document baseline papers: ColPali (main target), HPC-ColPali (competition), REAL-MM-RAG (dataset), Multimodal RAG Comparison
- Include links to download PDFs from arXiv
- Add notes on key metrics and claims from each paper
- **Requirements**: 6.4

#### 1.3 Implement hardware detection and backend configuration
- Create `src/utils/hardware.py` with device detection (MPS, CUDA, CPU)
- Implement automatic backend selection based on available hardware
- Add configuration options for batch sizes per hardware target
- **Requirements**: 9.4, 9.5, 9.6

#### 1.4 Write property test for hardware detection
- **Property 17: Hardware Detection Correctness**
- **Validates**: Requirements 9.4

#### 1.5 Update README with setup instructions
- Add installation steps and environment setup
- Document hardware requirements (M1 Pro local, T4 cloud)
- **Requirements**: All

### Task 2: Data Models and Core Types

#### 2.1 Implement core data structures
- Create `src/models/data.py` with Page, Document, EmbeddingResult dataclasses
- Create `src/models/results.py` with ClassificationResult, SearchResult, QueryResult
- Create `src/models/config.py` with ExperimentConfig, ExperimentResult
- **Requirements**: 4.2, 5.3, 10.1

#### 2.2 Write property test for data model serialization
- Test round-trip serialization of all dataclasses
- **Validates**: Requirements 4.2, 10.3

### Task 4: Data Loading and Preprocessing

#### 4.1 Implement base data loader interface
- Create `src/data/loader.py` with DataLoader protocol
- **Dataset Storage Strategy**:
  - Primary: HuggingFace `datasets` library with local cache (`~/.cache/huggingface/` or custom `data/cache/`)
  - Cache persists until manually cleared (not auto-deleted)
  - Alternative: Direct download to `data/raw/` for offline use
- Implement caching mechanism with configurable cache directory
- Add retry logic with exponential backoff for downloads
- **Requirements**: 7.5, 7.7

#### 4.2 Implement REAL-MM-RAG dataset loader
- Create `src/data/real_mm_rag.py` for TechReport (2.2k pages) and TechSlides (2.6k pages)
- **Source**: HuggingFace Hub (`ibm-research/REAL-MM-RAG_TechReport`, `ibm-research/REAL-MM-RAG_TechSlides`)
- Downloads via `datasets.load_dataset()` - cached locally after first download
- Normalize to BenchmarkDataset format with page images, queries, labels
- **Requirements**: 7.1, 7.4

#### 4.3 Implement DocVQA and InfographicVQA loaders
- Create `src/data/docvqa.py` for DocVQA dataset (10.5k samples)
- Create `src/data/infographicvqa.py` for InfographicVQA dataset (6k samples)
- **Source**: HuggingFace Hub (`lmms-lab/DocVQA`)
- Downloads via `datasets.load_dataset()` - cached locally
- **Requirements**: 7.2, 7.4

#### 4.4 Implement ViDoRe benchmark loader
- Create `src/data/vidore.py` for ViDoRe benchmark (ColPali evaluation)
- **Source**: GitHub illuin-tech/vidore-benchmark or HuggingFace
- Downloads benchmark files and caches locally
- **Requirements**: 7.3, 7.4

#### 4.5 Implement dataset splitting utilities
- Add train/validation/test split functionality with configurable ratios
- Ensure no data leakage between splits
- **Requirements**: 7.6

#### 4.6 Write property tests for data loading
- **Property 11: Dataset Normalization Consistency**
- **Property 12: Dataset Caching Round-Trip**
- **Property 13: Dataset Split Proportions**
- **Validates**: Requirements 7.4, 7.5, 7.6

#### 4.7 Update README with data loading instructions
- Document available datasets and their sizes
- Add examples for loading and inspecting data
- Include instructions for offline use (pre-download datasets)
- **Requirements**: 7.1, 7.2, 7.3

### Task 6: Router Component

#### 6.1 Implement heuristic router
- Create `src/router/heuristic.py` with OCR density and image area ratio computation
- Implement threshold-based classification logic
- Add confidence score calculation
- **Requirements**: 1.1, 1.3, 1.4, 1.6

#### 6.2 Implement router interface and batch processing
- Create `src/router/base.py` with RouterInterface protocol
- Implement batch classification with parallel processing
- Add error handling for corrupted pages (fallback to vision path)
- **Requirements**: 1.5, 1.7

#### 6.3 Implement optional ML-based router (DistilBERT)
- Create `src/router/ml_router.py` with DistilBERT classifier
- Add model loading and inference logic
- Implement fallback to heuristic if ML model unavailable
- **Requirements**: 1.6

#### 6.4 Write property tests for router
- **Property 1: Router Classification Correctness**
- **Property 2: Batch Classification Output Consistency**
- **Validates**: Requirements 1.2, 1.3, 1.4, 1.7

### Task 8: Text Embedding Path

#### 8.1 Implement text extraction with PyMuPDF
- Create `src/embedding/text_extractor.py` with structure-preserving extraction
- Handle headings, paragraphs, lists, and tables
- Add fallback for extraction failures
- **Requirements**: 2.1, 2.3, 2.5

#### 8.2 Implement nomic-embed-text embedding
- Create `src/embedding/text_embedder.py` with nomic-embed-text integration
- Implement batch processing for throughput optimization
- Add token limit handling with truncation
- **Requirements**: 2.1, 2.4, 2.6

#### 8.3 Implement TextEmbeddingPath pipeline
- Create `src/embedding/text_path.py` combining extraction and embedding
- Wire together extractor and embedder with error handling
- Implement escalation to vision path on failure
- **Requirements**: 2.1, 2.5

#### 8.4 Write property tests for text embedding path
- **Property 3: Text Extraction Structure Preservation**
- **Property 4: Embedding Dimension Consistency** (text path portion)
- **Validates**: Requirements 2.3, 2.4

### Task 9: Vision Embedding Path

#### 9.1 Implement ColPali embedding
- Create `src/embedding/vision_embedder.py` with ColPali model loading
- Implement image preprocessing and embedding generation
- Add batch processing with memory management
- **Requirements**: 3.1, 3.4

#### 9.2 Implement LoRA weight loading
- Add LoRA weight loading functionality to vision embedder
- Support both base ColPali and fine-tuned variants
- Handle incompatible weights gracefully
- **Requirements**: 3.5

#### 9.3 Implement VisionEmbeddingPath pipeline
- Create `src/embedding/vision_path.py` with full pipeline
- Add OOM handling with batch size reduction
- Implement CPU fallback for memory issues
- **Requirements**: 3.1, 3.6

#### 9.4 Write property tests for vision embedding path
- **Property 4: Embedding Dimension Consistency** (vision path portion)
- **Validates**: Requirements 3.4

### Task 11: Vector Database Integration

#### 11.1 Implement vector database interface
- Create `src/storage/base.py` with VectorDBInterface protocol
- Define common operations (insert, search, delete)
- Add metadata schema validation
- **Requirements**: 4.1, 4.5

#### 11.2 Implement Qdrant backend
- Create `src/storage/qdrant_backend.py` with Qdrant client integration
- Implement collection creation and management
- Add connection retry logic
- **Requirements**: 4.4

#### 11.3 Implement LanceDB backend
- Create `src/storage/lancedb_backend.py` with LanceDB integration
- Implement table creation and management
- **Requirements**: 4.4

#### 11.4 Implement incremental indexing
- Add support for adding documents without full re-indexing
- Implement batch insertion for efficiency
- **Requirements**: 4.6

#### 11.5 Write property tests for vector database
- **Property 5: Vector Database Storage Round-Trip**
- **Property 6: Embedding Dimension Validation**
- **Validates**: Requirements 4.1, 4.2, 4.5

### Task 12: Unified Retrieval Interface

#### 12.1 Implement query encoder
- Create `src/retrieval/query_encoder.py` for query embedding generation
- Support text queries with nomic-embed-text
- **Requirements**: 5.1

#### 12.2 Implement retrieval interface
- Create `src/retrieval/retriever.py` with unified search
- Implement top-K retrieval with configurable K
- Add result ranking by relevance score
- **Requirements**: 5.1, 5.2, 5.4

#### 12.3 Implement result formatting and logging
- Add result formatting with all required fields
- Implement query latency logging for benchmarking
- Handle empty result sets appropriately
- **Requirements**: 5.3, 5.5, 5.6

#### 12.4 Write property tests for retrieval
- **Property 7: Retrieval Result Correctness**
- **Validates**: Requirements 5.1, 5.2, 5.3, 5.4

### Task 14: Benchmark and Evaluation Framework

#### 14.1 Implement retrieval metrics
- Create `src/benchmark/metrics.py` with Recall@K, MRR, NDCG computation
- Ensure correct handling of edge cases (empty predictions, perfect recall)
- **Requirements**: 6.1

#### 14.2 Implement latency measurement
- Create `src/benchmark/latency.py` with per-component timing
- Compute mean, median, P95, and standard deviation
- **Requirements**: 6.2

#### 14.3 Implement throughput measurement
- Add throughput calculation (pages/second)
- Support end-to-end pipeline measurement
- **Requirements**: 6.3

#### 14.4 Implement benchmark runner
- Create `src/benchmark/runner.py` for full evaluation runs
- Support evaluation on all configured datasets
- Add baseline comparison (ColPali, HPC-ColPali)
- **Requirements**: 6.4, 6.5

#### 14.5 Implement result visualization
- Create `src/benchmark/visualization.py` for charts and tables
- Support ablation study configuration
- **Requirements**: 6.6, 6.7

#### 14.6 Write property tests for benchmark framework
- **Property 8: Recall Metric Computation**
- **Property 9: Latency Statistics Computation**
- **Property 10: Throughput Computation**
- **Validates**: Requirements 6.1, 6.2, 6.3

### Task 16: Fine-Tuning Pipeline

#### 16.1 Implement LoRA fine-tuning setup
- Create `src/finetuning/lora_trainer.py` with LoRA configuration
- Support SigLIP and ColPali model fine-tuning
- Optimize for T4 GPU memory constraints
- **Requirements**: 8.1, 8.2

#### 16.2 Implement checkpointing
- Add checkpoint saving at configurable intervals
- Implement checkpoint loading for training resumption
- **Requirements**: 8.3, 8.7

#### 16.3 Implement synthetic QA generation
- Create `src/finetuning/synthetic_qa.py` for training data augmentation
- Generate QA pairs from document content
- **Requirements**: 8.4

#### 16.4 Implement LoRA weight export
- Add weight export in format compatible with Vision_Embedding_Path
- Include training metrics logging
- **Requirements**: 8.5, 8.6

#### 16.5 Write property tests for fine-tuning
- **Property 14: Checkpoint Interval Consistency**
- **Property 15: LoRA Weights Round-Trip**
- **Property 16: Training Resume from Checkpoint**
- **Validates**: Requirements 8.3, 8.5, 8.7

### Task 17: Experiment Tracking and Reproducibility

#### 17.1 Implement experiment configuration logging
- Create `src/experiment/tracker.py` for configuration persistence
- Log model versions, hyperparameters, dataset versions
- **Requirements**: 10.1

#### 17.2 Implement random seed management
- Add seed setting for PyTorch, NumPy, and Python random
- Ensure reproducible results across runs
- **Requirements**: 10.2

#### 17.3 Implement result persistence
- Save experiment results in structured JSON format
- Support comparison tables between runs
- **Requirements**: 10.3, 10.4

#### 17.4 Implement LaTeX export
- Create `src/experiment/export.py` for publication-ready output
- Generate LaTeX tables and figure data
- **Requirements**: 10.5

#### 17.5 Create changelog template
- Add `CHANGELOG.md` for architecture decisions
- Document assumptions and direction changes
- **Requirements**: 10.6

#### 17.6 Write property tests for experiment tracking
- **Property 18: Experiment Reproducibility with Seeds**
- **Property 19: Experiment Persistence Completeness**
- **Property 20: LaTeX Export Validity**
- **Validates**: Requirements 10.2, 10.1, 10.3, 10.5

### Task 18: Pipeline Integration

#### 18.1 Implement main pipeline orchestrator
- Create `src/pipeline/orchestrator.py` combining all components
- Wire Router → Embedding Paths → Vector DB → Retrieval
- Add end-to-end error handling
- **Requirements**: 1.1, 2.1, 3.1, 4.1, 5.1

#### 18.2 Implement CLI interface
- Create `src/cli.py` with commands for indexing, querying, benchmarking
- Add configuration file loading
- **Requirements**: All

#### 18.3 Write integration tests
- Test full pipeline: load dataset → process pages → store → query → verify
- Test benchmark run on small dataset subset
- **Requirements**: All

### Task 20: Documentation and Notebooks

#### 20.1 Create Colab/Kaggle fine-tuning notebook
- Create `notebooks/finetuning.ipynb` for cloud fine-tuning
- Include setup instructions for T4 GPU
- **Requirements**: 8.2

#### 20.2 Create benchmark notebook
- Create `notebooks/benchmark.ipynb` for running evaluations
- Include visualization examples
- **Requirements**: 6.5, 6.6

#### 20.3 Update README with usage instructions
- Document installation, configuration, and usage
- Include example commands and expected outputs
- **Requirements**: All

## Notes

- All tasks including tests are required for comprehensive coverage
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation and README updates
- Property tests validate universal correctness properties (20 properties total)
- Unit tests validate specific examples and edge cases
- Fine-tuning tasks (16.x) are designed for cloud execution on Colab/Kaggle
- Local development focuses on router, text path, and retrieval components
- **Code simplicity**: Each module should be readable and do one thing well
- **Research papers**: Stored in `references/` with summaries in README

## Dataset Strategy

**How datasets are obtained:**
1. **HuggingFace `datasets` library** (primary method):
   - `datasets.load_dataset("ibm-research/REAL-MM-RAG_TechReport")` etc.
   - Downloads on first call, caches to `~/.cache/huggingface/datasets/`
   - Cache persists indefinitely until manually cleared (`rm -rf ~/.cache/huggingface/datasets/`)
   - Works locally and on Colab/Kaggle

2. **Offline/Pre-download option**:
   - Can pre-download to `data/raw/` for air-gapped environments
   - Useful for Kaggle competitions with limited internet

3. **Cache location** (configurable):
   - Default: `~/.cache/huggingface/datasets/`
   - Custom: Set `HF_DATASETS_CACHE` environment variable
   - Project-local: `data/cache/` if preferred

**Dataset sizes (approximate):**
- REAL-MM-RAG TechReport: ~2GB
- REAL-MM-RAG TechSlides: ~3GB
- DocVQA: ~5GB
- InfographicVQA: ~2GB
- ViDoRe: ~1GB
