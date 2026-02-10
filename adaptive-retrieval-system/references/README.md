# Research References

This directory contains summaries and links to key research papers that inform this project.

## Baseline Papers

### 1. ColPali: Efficient Document Retrieval with Vision Language Models (2024)
**arXiv**: [2407.01449](https://arxiv.org/abs/2407.01449)
**Authors**: Faysse et al.
**Status**: Main baseline to beat

**Key Contributions**:
- Introduced vision-based document retrieval using Vision Language Models
- Late interaction mechanism for efficient multi-vector retrieval
- ~1,000 embeddings per page for fine-grained matching

**Key Metrics** (from paper):
- State-of-the-art on ViDoRe benchmark
- Outperforms OCR-based pipelines on visually rich documents
- ~350ms per page on GPU

**Relevance to Our Work**:
- Our vision path uses ColPali as the base model
- We aim to achieve >90% of ColPali's accuracy with 50% less latency
- Our router decides when to use ColPali vs. faster text embedding

**Download**: [PDF](https://arxiv.org/pdf/2407.01449.pdf)

---

### 2. HPC-ColPali: Efficient Multi-Vector Document Retrieval with Dynamic Pruning (2025)
**arXiv**: [2506.21601](https://arxiv.org/abs/2506.21601)
**Status**: Competition (different optimization approach)

**Key Contributions**:
- Dynamic pruning of "useless" vectors to reduce computation
- Quantization for memory efficiency
- 30-50% latency reduction on ViDoRe and SEC-Filings datasets

**Key Metrics** (from paper):
- 30-50% lower query latency under HNSW indexing
- Maintains high retrieval precision
- 30% reduction in hallucination rates in RAG pipeline

**Relevance to Our Work**:
- Complementary approach: they prune vectors, we route pages
- Could potentially combine both approaches
- Must compare our results against HPC-ColPali

**Download**: [PDF](https://arxiv.org/pdf/2506.21601.pdf)

---

### 3. REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark (2025)
**arXiv**: [2502.12342](https://arxiv.org/abs/2502.12342)
**Authors**: IBM Research
**Status**: Primary benchmark dataset

**Key Contributions**:
- Multi-modal retrieval benchmark with realistic queries
- Four key properties: multi-modal documents, enhanced difficulty, realistic queries, accurate labeling
- Includes TechReport and TechSlides datasets

**Datasets**:
- **TechReport**: ~2.2k pages of technical reports
- **TechSlides**: ~2.6k pages of technical presentations
- **FinReport**: Financial reports (not used in our work)
- **FinSlides**: Financial presentations (not used in our work)

**Relevance to Our Work**:
- Primary benchmark for evaluation
- TechReport and TechSlides match our target domain
- Provides ground truth for retrieval metrics

**HuggingFace**:
- `ibm-research/REAL-MM-RAG_TechReport`
- `ibm-research/REAL-MM-RAG_TechSlides`

**Download**: [PDF](https://arxiv.org/pdf/2502.12342.pdf)

---

### 4. Comparison of Text-Based and Image-Based Retrieval in Multimodal RAG (2025)
**arXiv**: [2511.16654](https://arxiv.org/abs/2511.16654)
**Status**: Validation for our approach

**Key Contributions**:
- Statistical comparison of text vs. image retrieval
- Vision retrieval is 13-30% more accurate than text retrieval
- Justifies using vision models for diagram-heavy pages

**Key Findings**:
- Image-based retrieval significantly outperforms text-based on visually rich documents
- OCR-based approaches lose spatial information
- Hybrid approaches can balance accuracy and efficiency

**Relevance to Our Work**:
- Validates our hypothesis that vision is needed for diagrams
- Justifies the 80/20 split (text-heavy vs. visual-critical)
- Supports our router-based approach

**Download**: [PDF](https://arxiv.org/pdf/2511.16654.pdf)

---

### 5. Fine-Tuning Vision-Language Model for Automated Engineering Drawing Information Extraction (2024)
**Source**: ResearchGate
**Status**: Fine-tuning methodology reference

**Key Contributions**:
- Fine-tuning Florence-2 for engineering drawings
- Domain-specific adaptation for technical documents
- LoRA-based efficient fine-tuning

**Relevance to Our Work**:
- Practical guide for Phase 4 (fine-tuning)
- Engineering drawings are similar to our target domain
- LoRA approach matches our hardware constraints

---

## Additional References

### ViDoRe Benchmark
**GitHub**: [illuin-tech/vidore-benchmark](https://github.com/illuin-tech/vidore-benchmark)
**Purpose**: ColPali evaluation benchmark

### DocVQA Dataset
**HuggingFace**: `lmms-lab/DocVQA`
**Purpose**: Document QA baseline (~10.5k samples)

### nomic-embed-text
**HuggingFace**: `nomic-ai/nomic-embed-text-v1.5`
**Purpose**: Fast text embedding model for text path

---

## Key Metrics to Track

| Metric | Description | Target |
|--------|-------------|--------|
| Recall@1 | % of queries where correct doc is top result | >85% |
| Recall@5 | % of queries where correct doc is in top 5 | >92% |
| Recall@10 | % of queries where correct doc is in top 10 | >95% |
| MRR | Mean Reciprocal Rank | >0.85 |
| NDCG | Normalized Discounted Cumulative Gain | >0.90 |
| Latency (mean) | Average processing time per page | <100ms |
| Latency (P95) | 95th percentile latency | <200ms |
| Throughput | Pages processed per second | >10 |

---

## Download Instructions

To download papers locally:

```bash
# ColPali
wget https://arxiv.org/pdf/2407.01449.pdf -O references/colpali.pdf

# HPC-ColPali
wget https://arxiv.org/pdf/2506.21601.pdf -O references/hpc-colpali.pdf

# REAL-MM-RAG
wget https://arxiv.org/pdf/2502.12342.pdf -O references/real-mm-rag.pdf

# Multimodal RAG Comparison
wget https://arxiv.org/pdf/2511.16654.pdf -O references/multimodal-rag-comparison.pdf
```

Note: PDFs are not tracked in git due to size. Download as needed.
