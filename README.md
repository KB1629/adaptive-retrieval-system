# Adaptive Retrieval System

**Optimizing Latency in Technical RAG via Semantic Routing and Domain-Specific Vision Fine-Tuning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This research project implements a **modality-adaptive RAG pipeline** that intelligently routes document pages to either fast text embedding or high-accuracy vision embedding based on visual complexity.

**Core Objective**: Achieve >90% of ColPali's retrieval accuracy with 50% less latency for technical documentation.

### ⚠️ Current Status (Feb 4, 2026)

**IMPORTANT:** This project is undergoing a major architecture revision based on recent findings. See [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) for details.

**What's Working:**
- ✅ ColPali vision model (~4s/page on M1 Pro)
- ✅ PaddleOCR integrated (configuration being optimized)
- ✅ Basic pipeline infrastructure
- ✅ 100 DocVQA pages for benchmarking

**What's Being Fixed:**
- 🔧 **Router:** Replacing heuristic router with vision-based router (Microsoft DiT)
- 🔧 **PaddleOCR:** Optimizing configuration (removing unnecessary preprocessing modules)
- 🔧 **Benchmarks:** Re-running with corrected configuration

**Expected Timeline:** 3-4 hours to complete fixes and re-run benchmarks.

### Key Innovation

Traditional approaches force a single retrieval method on all pages:
- **OCR-based RAG**: Fast but loses spatial information in diagrams ("Diagram Blindness")
- **Vision RAG (ColPali)**: Accurate but computationally expensive (~1,000 embeddings per page)

Our **hybrid approach** uses a lightweight vision router to classify pages:
- **Text-heavy pages** → Fast OCR + text embedding (~1-5s)
- **Visual-critical pages** → Vision embedding (~4s)

This achieves significant latency reduction while preserving accuracy on complex technical diagrams.

## Target Domain

- Automotive technical manuals
- Aerospace documentation
- Industrial machinery guides
- Engineering schematics and wiring diagrams

## Architecture

**Current Architecture (Being Revised):**

```
PDF Document
     │
     ▼
┌──────────────────┐
│  Vision Router   │ ← Microsoft DiT (document layout classifier)
│  (50-100ms)      │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌──────────┐
│  Text   │ │  Vision  │
│  Path   │ │  Path    │
│ (1-5s)  │ │  (~4s)   │
└────┬────┘ └────┬─────┘
     │           │
     ▼           ▼
┌──────────┐ ┌──────────┐
│PaddleOCR │ │ ColPali  │
│(det+rec) │ │ (vision) │
└────┬─────┘ └────┬─────┘
     │            │
     ▼            ▼
┌──────────┐ ┌──────────┐
│   Text   │ │  Vision  │
│Embeddings│ │Embeddings│
└────┬─────┘ └────┬─────┘
     │            │
     └─────┬──────┘
           ▼
    ┌─────────────┐
    │  Vector DB  │
    │  (Unified)  │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Retrieval  │
    └─────────────┘
```

**Key Components:**
1. **Vision Router (DiT):** Classifies pages as text-heavy vs visual-critical
2. **Text Path:** PaddleOCR (det+rec only) → text embeddings
3. **Vision Path:** ColPali → vision embeddings
4. **Vector Store:** Unified storage for both embedding types

See [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) for detailed architecture decisions.

## Installation

### Prerequisites

- Python 3.10+
- macOS with M1/M2/M3 chip (for local development) OR
- Google Colab/Kaggle (for fine-tuning)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/username/adaptive-retrieval-system.git
cd adaptive-retrieval-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, MPS: {torch.backends.mps.is_available()}')"
```

### Hardware-Specific Setup

**For M1/M2/M3 Mac (MPS backend):**
```bash
pip install -e ".[dev]"
# MPS is automatically detected
```

**For Colab/Kaggle (CUDA backend):**
```bash
pip install -e ".[dev,finetuning]"
```

### Verify Hardware Detection

```python
from src.utils.hardware import detect_device

device = detect_device()
print(f"Using device: {device}")
```

## Usage

### 1. Data Loading

Load benchmark datasets for evaluation:

```python
from src.data.real_mm_rag import load_real_mm_rag_dataset
from src.data.docvqa import load_docvqa_dataset

# Load REAL-MM-RAG (IBM Research benchmark)
dataset = load_real_mm_rag_dataset(
    variant="TechReport",  # or "TechSlides"
    cache_dir="./data/cache"
)

print(f"Loaded {len(dataset.documents)} documents")
print(f"Queries: {len(dataset.queries)}")
```

### 2. Router Classification

Classify pages as text-heavy or visual-critical:

```python
from src.router.heuristic import HeuristicRouter
from PIL import Image

# Initialize router
router = HeuristicRouter(
    text_density_threshold=0.7,
    image_area_threshold=0.3
)

# Classify a page
page_image = Image.open("page.png")
result = router.classify(page_image)

print(f"Modality: {result.modality}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Features: {result.features}")
```

### 3. Text Embedding Path

Process text-heavy pages with fast embedding:

```python
from src.embedding.text_path import TextEmbeddingPath

# Initialize text path
text_path = TextEmbeddingPath(device="mps")

# Process a page
result = text_path.process_page(page_image)

print(f"Embedding shape: {result.embedding.shape}")
print(f"Processing time: {result.processing_time_ms:.2f}ms")
```

### 4. Vision Embedding Path

Process visual-critical pages with ColPali:

```python
from src.embedding.vision_path import VisionEmbeddingPath

# Initialize vision path
vision_path = VisionEmbeddingPath(
    model_name="vidore/colpali",
    device="mps"
)

# Optional: Load LoRA fine-tuned weights
vision_path.load_lora_weights("path/to/lora_weights.pt")

# Process a page
result = vision_path.process_page(page_image)

print(f"Embedding shape: {result.embedding.shape}")
print(f"Processing time: {result.processing_time_ms:.2f}ms")
```

### 5. Vector Storage

Store embeddings in vector database:

```python
from src.storage.qdrant_backend import QdrantBackend
from src.storage.base import DocumentMetadata
from datetime import datetime

# Initialize Qdrant
vector_db = QdrantBackend(
    collection_name="adaptive_retrieval",
    embedding_dim=768,
    host="localhost",
    port=6333
)

# Store embedding
metadata = DocumentMetadata(
    doc_id="manual_001",
    page_number=1,
    modality="text-heavy",
    source_file="manual.pdf",
    processed_at=datetime.now(),
    model_name="nomic-embed-text",
    embedding_dim=768
)

embedding_id = vector_db.insert(embedding, metadata)
```

### 6. Retrieval

Query the system:

```python
from src.retrieval.retriever import Retriever

# Initialize retriever
retriever = Retriever(vector_db=vector_db, default_top_k=5)

# Query
result = retriever.retrieve("How to replace the fuel pump?")

print(f"Found {len(result.results)} results in {result.query_latency_ms:.2f}ms")
for i, doc in enumerate(result.results, 1):
    print(f"{i}. {doc.doc_id} (page {doc.page_number}) - Score: {doc.score:.3f}")
```

### 7. Benchmark Evaluation

Evaluate performance against ColPali:

```python
from src.benchmark.runner import BenchmarkRunner, BenchmarkConfig
from src.benchmark.metrics import evaluate_retrieval

# Create benchmark config
config = BenchmarkConfig(
    name="adaptive-vs-colpali",
    datasets=["REAL-MM-RAG"],
    k_values=[1, 5, 10],
    measure_latency=True,
    measure_throughput=True
)

# Run benchmark
runner = BenchmarkRunner(retrieval_func=retriever.retrieve)
result = runner.run(config, datasets)

# Print results
runner.print_summary(result)

# Output:
# Overall Metrics:
#   Recall@1:  0.7234
#   Recall@5:  0.8912
#   Recall@10: 0.9456
#   MRR:       0.8123
#   NDCG:      0.8734
#
# Latency (mean):
#   query: 45.23ms
#
# Throughput: 7.8 pages/sec
```

### 8. Visualization

Create comparison tables:

```python
from src.benchmark.visualization import (
    create_comparison_table,
    create_latex_table,
    create_speedup_table
)

# Compare with baseline
results = {
    "ColPali": colpali_result,
    "Adaptive (Ours)": adaptive_result,
}

# ASCII table
print(create_comparison_table(results))

# LaTeX table for paper
latex = create_latex_table(results, caption="Performance Comparison")
with open("results.tex", "w") as f:
    f.write(latex)

# Speedup analysis
speedup_table = create_speedup_table("ColPali", colpali_result, {"Adaptive": adaptive_result})
print(speedup_table)
```

## Project Structure

```
adaptive-retrieval-system/
├── src/
│   ├── router/          # Page classification
│   ├── embedding/       # Text and vision paths
│   ├── storage/         # Vector database backends
│   ├── retrieval/       # Query interface
│   ├── benchmark/       # Evaluation framework
│   ├── data/            # Dataset loaders
│   ├── models/          # Data structures
│   └── utils/           # Hardware detection, logging
├── tests/               # Comprehensive test suite
├── configs/             # Configuration files
├── notebooks/           # Jupyter notebooks
└── references/          # Research papers
```

## Benchmarks

### Current Status

**Baseline Completed:**
- Pure ColPali: 2820.56 ms/page (100 DocVQA pages, M1 Pro)

**Adaptive System:** In progress - being re-run with optimized configuration

### Datasets

- **DocVQA** (ViDoRe benchmark): 100 pages downloaded for testing
- **REAL-MM-RAG** (IBM Research): Technical reports and slides
- **ViDoRe**: ColPali evaluation benchmark

### Expected Results (After Fixes)

| System | Recall@5 | Latency (mean) | Speedup |
|--------|----------|----------------|---------|
| ColPali | 0.912 | 2820ms (M1 Pro) | 1.0x |
| **Adaptive (Target)** | **>0.82** | **1000-2000ms** | **1.5-2.8x** |

*Note: Benchmarks will be updated once router and PaddleOCR fixes are complete.*

### Research Context

| Paper | arXiv | Role |
|-------|-------|------|
| ColPali | [2407.01449](https://arxiv.org/abs/2407.01449) | Main baseline (vision-based retrieval) |
| HPC-ColPali | [2506.21601](https://arxiv.org/abs/2506.21601) | Competition (pruning approach) |
| REAL-MM-RAG | [2502.12342](https://arxiv.org/abs/2502.12342) | Benchmark dataset |
| DiT | [2203.02378](https://arxiv.org/abs/2203.02378) | Document layout analysis (our router) |

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_router.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Property-Based Testing

We use Hypothesis for property-based testing to validate correctness properties:

```bash
# Run property tests
pytest tests/ -k "property" -v

# Run with more examples
pytest tests/ -k "property" --hypothesis-profile=thorough
```

## Research Papers

### Main Baseline
- **ColPali** (arXiv:2407.01449): Vision-based document retrieval with late interaction

### Competition
- **HPC-ColPali** (arXiv:2506.21601): Pruning-based optimization

### Validation
- **REAL-MM-RAG** (arXiv:2502.12342): Benchmark dataset
- **Multimodal RAG Comparison** (arXiv:2511.16654): Vision vs text for diagrams

## Citation

```bibtex
@article{adaptive-retrieval-2026,
  title={Adaptive Retrieval System: Optimizing Latency in Technical RAG via Semantic Routing},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- ColPali team for the baseline vision retrieval model
- IBM Research for the REAL-MM-RAG benchmark dataset
- HuggingFace for model hosting and datasets library

```python
from src.utils.hardware import get_hardware_config

config = get_hardware_config()
print(f"Device: {config.device.value}")
print(f"Device Name: {config.device_name}")
print(f"Max Memory: {config.max_memory_gb:.1f} GB")
print(f"Recommended batch sizes: {config.recommended_batch_sizes}")
```

Expected output on M1 Pro:
```
Device: mps
Device Name: Apple Silicon (MPS)
Max Memory: 8.0 GB
Recommended batch sizes: {'text_embedding': 16, 'vision_embedding': 2, 'router': 32}
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run property-based tests only
pytest tests/ -v -k "property"
```

## Project Structure

```
adaptive-retrieval-system/
├── src/                    # Source code
│   ├── models/             # Data structures
│   ├── data/               # Dataset loaders
│   ├── router/             # Page classification
│   ├── embedding/          # Text and vision paths
│   ├── storage/            # Vector database
│   ├── retrieval/          # Query interface
│   ├── benchmark/          # Evaluation framework
│   └── utils/              # Hardware, logging
├── tests/                  # Test suite
├── notebooks/              # Jupyter notebooks
├── configs/                # Configuration variants
├── references/             # Research papers
└── outputs/                # Experiment results
```

## Router Component

The router classifies document pages as either "text-heavy" or "visual-critical":

```python
from src.router import HeuristicRouter, RouterConfig
from src.models import Page
import numpy as np

# Create router with custom configuration
config = RouterConfig(
    text_threshold=0.6,      # Score above this = text-heavy
    min_confidence=0.5,      # Minimum confidence threshold
    fallback_modality="visual-critical",  # Default on low confidence
)
router = HeuristicRouter(config)

# Classify a single page
image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
page = Page.from_array(image, page_number=1, source_document="doc.pdf")

result = router.classify(page)
print(f"Modality: {result.modality}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Features: {result.features}")

# Batch classification
pages = [page1, page2, page3]
results = router.classify_batch(pages)
```

### Router Features

The heuristic router extracts these visual features:
- **text_density**: Estimated text coverage (0-1)
- **image_ratio**: Non-text visual content ratio (0-1)
- **edge_density**: Edge pixel density (indicates diagrams)
- **color_variance**: Color variation (high = images/diagrams)
- **white_ratio**: Background coverage

## Embedding Paths

### Text Embedding Path

Fast processing for text-heavy pages (~30-50ms per page):

```python
from src.embedding import TextEmbeddingPath
import numpy as np

# Initialize text embedding path
text_path = TextEmbeddingPath()

# Process a page
page_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
result = text_path.process_page(page_image)

print(f"Modality: {result.modality}")  # "text-heavy"
print(f"Embedding shape: {result.vector.shape}")  # (768,)
print(f"Processing time: {result.processing_time_ms:.2f}ms")
print(f"Extracted text length: {len(result.extracted_text)}")
```

**Features:**
- PyMuPDF text extraction with structure preservation
- nomic-embed-text model (768 dimensions)
- Automatic escalation to vision path on failure
- Batch processing support

### Vision Embedding Path

High-accuracy processing for visual-critical pages (~300-400ms per page):

```python
from src.embedding import VisionEmbeddingPath
import numpy as np

# Initialize vision embedding path
vision_path = VisionEmbeddingPath()

# Process a page
page_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
result = vision_path.process_page(page_image)

print(f"Modality: {result.modality}")  # "visual-critical"
print(f"Embedding shape: {result.vector.shape}")
print(f"Processing time: {result.processing_time_ms:.2f}ms")
```

**Features:**
- ColPali/SigLIP vision models
- LoRA fine-tuned weight support
- OOM handling with batch size reduction
- CPU fallback for memory issues

### Loading LoRA Weights

```python
from src.embedding import VisionEmbeddingPath

# Load with fine-tuned LoRA weights
vision_path = VisionEmbeddingPath(
    lora_weights_path="./outputs/checkpoints/lora_weights.pt"
)

# Check if LoRA is loaded
info = vision_path.get_pipeline_info()
print(f"LoRA loaded: {info['embedder']['lora_loaded']}")
```

## Research Papers

This project builds upon and compares against:

| Paper | arXiv | Role |
|-------|-------|------|
| ColPali | [2407.01449](https://arxiv.org/abs/2407.01449) | Main baseline |
| HPC-ColPali | [2506.21601](https://arxiv.org/abs/2506.21601) | Competition (pruning) |
| REAL-MM-RAG | [2502.12342](https://arxiv.org/abs/2502.12342) | Benchmark dataset |

## Datasets

This project uses publicly available datasets from HuggingFace Hub:

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| REAL-MM-RAG TechReport | `ibm-research/REAL-MM-RAG_TechReport` | ~2.2k pages, ~2GB | Primary benchmark |
| REAL-MM-RAG TechSlides | `ibm-research/REAL-MM-RAG_TechSlides` | ~2.6k pages, ~3GB | Secondary benchmark |
| DocVQA | `lmms-lab/DocVQA` | ~10.5k samples, ~5GB | Document QA baseline |
| InfographicVQA | `lmms-lab/InfographicVQA` | ~6k samples, ~2GB | Infographic understanding |
| ViDoRe | `vidore/*` | Various | ColPali evaluation benchmark |

### Loading Datasets

Datasets are automatically downloaded and cached on first use:

```python
from src.data import load_techreport, load_docvqa, load_vidore

# Load REAL-MM-RAG TechReport
dataset = load_techreport()
print(f"Loaded {dataset.num_pages} pages, {dataset.num_queries} queries")

# Load DocVQA validation split
docvqa = load_docvqa(split="validation")

# Load ViDoRe benchmark subset
vidore = load_vidore(subset="vidore/docvqa_test_subsampled")
```

### Dataset Caching

Datasets are cached in `~/.cache/huggingface/datasets/` by default:

```python
from src.data import DataLoaderConfig, RealMMRAGLoader

# Custom cache directory
config = DataLoaderConfig(cache_dir="./data/cache")
loader = RealMMRAGLoader(variant="techreport", config=config)
dataset = loader.load()

# Check if cached
print(f"Is cached: {loader.is_cached()}")
```

### Dataset Splitting

Split datasets for training/validation/testing:

```python
from src.data import DatasetSplitter, SplitConfig

# Configure split ratios
config = SplitConfig(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,  # For reproducibility
)

splitter = DatasetSplitter(config)
train, val, test = splitter.split(dataset)

print(f"Train: {train.num_pages} pages")
print(f"Val: {val.num_pages} pages")
print(f"Test: {test.num_pages} pages")
```

### Available ViDoRe Subsets

```python
from src.data import get_available_subsets

subsets = get_available_subsets()
# ['vidore/arxivqa_test_subsampled', 'vidore/docvqa_test_subsampled', ...]
```

## Expected Results

| Metric | ColPali (Baseline) | Our System (Target) |
|--------|-------------------|---------------------|
| Recall@10 | 100% | >90% |
| Avg Latency | ~350ms/page | ~100ms/page |
| Latency Reduction | - | 50-70% |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Command Line Interface

The system provides a CLI for common operations:

```bash
# Index documents
python -m src.cli index ./documents/ --config config.yaml

# Query the system
python -m src.cli query "How to replace the fuel pump?" --top-k 5

# Run benchmark
python -m src.cli benchmark --dataset real-mm-rag --config configs/benchmark.yaml

# List experiments
python -m src.cli experiment list --output-dir outputs/experiments

# Show experiment results
python -m src.cli experiment show --experiment-id exp_20260204_120000
```

### Configuration

Create a `config.yaml` file:

```yaml
router:
  type: heuristic
  text_density_threshold: 0.7
  image_area_threshold: 0.3

text_embedding:
  model: nomic-embed-text
  max_tokens: 8192
  batch_size: 32

vision_embedding:
  model: colpali
  lora_weights: null  # or path to weights
  batch_size: 4

vector_db:
  backend: qdrant  # or lancedb
  collection_name: adaptive_retrieval
  host: localhost
  port: 6333

hardware:
  device: auto  # mps, cuda, or cpu

experiment:
  random_seed: 42
  checkpoint_interval: 100
```

## Notebooks

Interactive notebooks for experimentation:

- **`notebooks/finetuning.ipynb`** - LoRA fine-tuning on Colab/Kaggle
- **`notebooks/benchmark.ipynb`** - Evaluation and visualization

## Citation

```bibtex
@misc{adaptive-retrieval-2025,
  title={Adaptive Retrieval System: Optimizing Latency in Technical RAG via Semantic Routing},
  author={Kabeleswar P E},
  year={2025},
  note={Research project - CB.SC.I5DAS21032}
}
```

## Acknowledgments

- ColPali team for the baseline vision retrieval model
- IBM Research for the REAL-MM-RAG benchmark
- HuggingFace for model hosting and datasets
