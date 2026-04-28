# Adaptive Retrieval System for Technical Documents

**Design and Implementation of an Adaptive Retrieval System for Technical Documents using Dynamic Dual-Path Routing**

> Final Project — Integrated M.Sc. in Data Science  
> Kabeleswar P E · CB.SC.I5DAS21032  
> Amrita School of Physical Sciences, Coimbatore · Amrita Vishwa Vidyapeetham  
> Project Advisor: Dr. P. Sriramakrishnan, Assistant Professor, Department of Mathematics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

Standard Retrieval-Augmented Generation (RAG) pipelines apply either OCR or a Vision Language Model (VLM) uniformly to every page of a document. OCR destroys spatial structure — tables collapse, wiring diagrams become garbled text. Applying ColPali (a VLM) uniformly costs ~2.05 seconds per page, making it infeasible for large corpora (11+ hours for 20,000 pages).

This project solves that by building a **smart page-level router** using Microsoft DiT (Document Image Transformer) that classifies every page in under 100ms and routes it to the right pipeline:

- **Text-heavy pages** → PaddleOCR → Nomic Embed (768-D) → LanceDB text index
- **Visually complex pages** → ColPali v1.2 → Mean Pooling (128-D) → LanceDB vision index

At query time, both indexes are searched in parallel and merged using **Reciprocal Rank Fusion (RRF)**, which elegantly sidesteps the problem of incompatible cosine score distributions between the two embedding spaces.

---

## Key Results

Benchmarked on a **500-page subset of the REAL-MM-RAG TechReport** (IBM Research, arXiv:2502.12342) with **100 natural language queries** on Apple M1 Pro (16GB Unified Memory).

| System | R@1 | R@5 | R@10 | MRR | Query Latency |
|--------|-----|-----|------|-----|---------------|
| OCR + Nomic Text Only | 0.04 | 0.18 | 0.31 | 0.10 | 95 ms |
| Mean-pooled ColPali | 0.07 | 0.35 | 0.60 | 0.21 | 483 ms |
| **Adaptive System (Ours)** | **0.07** | **0.35** | **0.60** | **0.21** | **483 ms** |
| ColPali MaxSim (external)* | 0.09 | 0.41 | 0.81 | 0.23 | 2,050 ms |

> \* ColPali MaxSim figures from the ViDoRe benchmark, not measured on our 500-page subset. The Recall@10 gap (0.60 vs 0.81) is explained by the mean-pooling approximation.

**Key achievements:**
- **4.24× query speedup** — 483ms vs 2,050ms (MaxSim baseline)
- **~10% indexing latency reduction** — 1.86s vs 2.05s per page
- **Recall@10 = 0.60, MRR = 0.21** — matches full mean-pooled ColPali
- **12.2% of pages routed to the fast text path** — eliminating that fraction of costly VLM encoding calls
- **Self-healing OOM recovery** — auto-halves batch size on memory overflow and retries without crash

---

## System Architecture

```
PDF Document
     │
     ▼
Microsoft DiT Router  ──── < 100ms per page ─── ViT-Base, 16-class Softmax
     │                                           Pre-trained on RVL-CDIP (400k docs)
     ├──── Text-Heavy (12.2%) ──────────────────────────────────────┐
     │                                                               │
     │     PaddleOCR v5                                             │
     │     (disabled 3 unused preprocessors: 200s → ~3s/page)      │
     │          │                                                    │
     │     Nomic Embed Text v1.5                                    │
     │     (768-dim, 8192-token context window)                     │
     │          │                                                    │
     │     LanceDB Text Index ◄──────────────────────────────────── ┘
     │
     └──── Visually Complex (87.8%) ────────────────────────────────┐
                                                                     │
           ColPali v1.2  (SigLIP + PaliGemma)                      │
           (1024 patch embeddings per page)                          │
                │                                                    │
           Mean Pooling → 128-dim constant-size vector              │
                │                                                    │
           LanceDB Vision Index ◄──────────────────────────────────┘

                    ▼ At Query Time ▼

     Both indexes searched in parallel
           │
     Reciprocal Rank Fusion  [RRF(d) = Σ 1/(k+rank), k=60]
           │
     Final Ranked Output
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Apple Silicon (M1/M2/M3) recommended — the pipeline uses PyTorch MPS for ColPali inference

### Installation

```bash
# Clone the repository
git clone https://github.com/KB1629/adaptive-retrieval-system.git
cd adaptive-retrieval-system

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install all dependencies
pip install -e ".[dev]"
```

Verify hardware detection:
```python
from src.utils.hardware import detect_device
print(f"Using device: {detect_device()}")
```

### Running the Benchmark

```bash
# Full accuracy benchmark on REAL-MM-RAG TechReport (100 queries, 500 pages)
python scripts/run_accuracy_benchmark.py

# Run on the full REAL-MM-RAG dataset
python scripts/run_real_benchmark.py

# Run ColPali baseline only (for comparison)
python scripts/run_colpali_baseline.py

# Download the REAL-MM-RAG dataset pages
python scripts/download_real_pages.py
```

---

## Repository Structure — Explained for Reviewers

```
adaptive-retrieval-system/
├── src/                        ← All source code
│   ├── router/                 ← Page classification (Microsoft DiT)
│   ├── embedding/              ← Both embedding paths (text + vision)
│   ├── retrieval/              ← RRF fusion and query engine
│   ├── storage/                ← LanceDB vector database backend
│   ├── benchmark/              ← Evaluation metrics (Recall@K, MRR, latency)
│   ├── pipeline/               ← End-to-end orchestrator
│   ├── data/                   ← REAL-MM-RAG dataset loaders
│   ├── models/                 ← Dataclass definitions for configs/results
│   ├── experiment/             ← Experiment tracking and export utilities
│   ├── finetuning/             ← LoRA fine-tuning (future work)
│   └── utils/                  ← Apple M1 MPS hardware utilities
│
├── scripts/                    ← Entry-point scripts to run the system
│   ├── run_accuracy_benchmark.py   ← MAIN: runs full R@K, MRR evaluation
│   ├── run_real_benchmark.py       ← Runs on full REAL-MM-RAG dataset
│   ├── run_colpali_baseline.py     ← Standalone ColPali baseline comparison
│   ├── download_real_dataset.py    ← Downloads REAL-MM-RAG metadata
│   └── download_real_pages.py      ← Downloads page images from HuggingFace
│
├── data/                       ← Dataset
│   └── docvqa_sample/          ← 100 page images (page_000.png–page_099.png)
│                                  + metadata.json (query–page ground truth)
│
├── configs/                    ← YAML configuration files
│   ├── benchmark.yaml          ← Benchmark run settings
│   ├── dev.yaml                ← Development/local settings
│   └── colab.yaml              ← Google Colab settings
│
├── outputs/                    ← All benchmark result files (auto-generated)
│   ├── benchmark_results/      ← JSON/CSV result files with timestamps
│   ├── checkpoints/            ← Model checkpoints (if fine-tuned)
│   ├── figures/                ← Auto-generated charts and plots
│   └── results/                ← Summarized result exports
│
├── tests/                      ← Unit tests for all modules
├── notebooks/                  ← Jupyter notebooks for exploration
│   ├── benchmark.ipynb         ← Interactive benchmark exploration
│   └── finetuning.ipynb        ← Fine-tuning experiments notebook
│
├── generated-diagrams/         ← Architecture diagrams (PNG)
├── references/                 ← Reference notes
├── docs/                       ← Final project deliverables
│   ├── Adaptive_Retrieval_FinalReview.pptx   ← Final viva presentation
│   ├── ADAPTIVE RETRIEVAL SYSTEM.docx        ← Project report (Word)
│   ├── kabi proj report 30 pages.pdf         ← Project report (PDF)
│   └── Adaptive_Retrieval_System_Paper.pdf   ← IEEE-format research paper
│
├── config.yaml                 ← Root configuration file
├── pyproject.toml              ← Python project dependencies
└── .gitignore                  ← Git ignore rules
```

### Module Details

#### `src/router/` — The Routing Gate
The core innovation. `vision_router.py` wraps Microsoft DiT (a ViT-Base model pre-trained on 400k document images from RVL-CDIP) to classify each page into one of 16 document layout classes. Pages below a confidence threshold τ = 0.05 for visual content are routed to the fast text path; all others go to ColPali.

#### `src/embedding/` — Dual Embedding Paths
- `text_path.py` + `text_extractor.py` + `text_embedder.py` → PaddleOCR extracts text, Nomic Embed Text v1.5 encodes it into a 768-dim dense vector
- `vision_path.py` + `vision_embedder.py` → ColPali v1.2 encodes each page as 1024 patch embeddings, which are mean-pooled to a constant 128-dim vector for indexing

#### `src/retrieval/` — RRF Query Engine
`retriever.py` encodes the user query through both embedding models simultaneously, retrieves the top-10 candidates from each LanceDB index, and merges the ranked lists using Reciprocal Rank Fusion: `RRF(d) = Σ 1/(60 + rank)`.

#### `src/storage/` — Vector Database
`lancedb_backend.py` manages two separate Apache Arrow-format indexes on disk — one for 768-D text vectors and one for 128-D vision vectors — with cosine similarity search and zero network latency.

#### `src/benchmark/` — Evaluation Framework
`metrics.py` computes Recall@1, Recall@5, Recall@10, and MRR. `latency.py` measures per-page indexing time and per-query retrieval time. `runner.py` orchestrates a full benchmark run across all 100 queries.

---

## Engineering Challenges Solved

| Challenge | Problem | Solution |
|-----------|---------|----------|
| PaddleOCR latency | Package silently loaded 5 neural nets on import → 200s/page | Disabled `use_angle_cls`, `use_doc_preprocessor`, `use_doc_unwarping` → ~3s/page |
| Double classification bug | DiT router re-ran classification inside benchmark timing loop → 160× slowdown | Pre-computed all classifications before timing loop |
| M1 memory bottleneck | Batching 4 ColPali images saturated 16GB unified memory → 2.30s/page | Sequential ColPali (batch=1) + batched Nomic (batch=4) |
| OOM crashes | PyTorch MPS out-of-memory on large pages | Custom `RuntimeError` trap that auto-halves batch size and retries |
| Incompatible score spaces | Nomic (768-D) and ColPali (128-D) cosine scores are not comparable | Reciprocal Rank Fusion uses only integer ranks, not raw scores |

---

## References

1. V. Faysse et al., *ColPali: Efficient Document Retrieval with Vision Language Models*, arXiv:2407.01449, 2024.
2. J. Li et al., *DiT: Self-supervised Pre-training for Document Image Transformer*, ACM Multimedia, 2022.
3. Z. Nussbaum et al., *Nomic Embed: Training a Reproducible Long Context Text Embedder*, arXiv:2402.01613, 2024.
4. G. V. Cormack et al., *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*, SIGIR, 2009.
5. R. Chen et al., *REAL-MM-RAG: A Real-World Multimodal Retrieval Benchmark*, arXiv:2502.12342, 2025.
6. PaddlePaddle Authors, *PaddleOCR*, GitHub, 2020. https://github.com/PaddlePaddle/PaddleOCR
7. LanceDB Authors, *LanceDB: Developer-friendly, Serverless Vector Database*, GitHub, 2024. https://github.com/lancedb/lancedb

---

## Authors

- **Kabeleswar P E** — Student Researcher, Integrated M.Sc. Data Science, Amrita Vishwa Vidyapeetham
- **Dr. P. Sriramakrishnan** — Project Advisor, Assistant Professor, Department of Mathematics, Amrita Vishwa Vidyapeetham

---

## License

MIT License — see `LICENSE` for details.
