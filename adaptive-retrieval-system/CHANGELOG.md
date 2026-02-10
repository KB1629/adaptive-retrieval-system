# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and configuration files
- Research references directory with paper summaries
- Hardware detection for MPS/CUDA/CPU backends
- Core data models and type definitions

### Architecture Decisions

#### AD-001: Hybrid Router Architecture (2025-02-03)
**Decision**: Use a two-tier router with heuristic-first approach and optional ML classifier.

**Context**: Need to classify pages as text-heavy or visual-critical with minimal latency (<50ms).

**Rationale**:
- Heuristic router (OCR density + image area) is fast and interpretable
- ML classifier (DistilBERT) can be added for higher accuracy if needed
- Fallback to vision path on errors preserves accuracy

**Consequences**:
- Simple implementation, easy to debug
- May need ML classifier for edge cases
- Router accuracy directly impacts overall system performance

#### AD-002: Vector Database Abstraction (2025-02-03)
**Decision**: Support both Qdrant and LanceDB with a common interface.

**Context**: Need flexible storage that works locally (dev) and in production.

**Rationale**:
- Qdrant: Production-ready, supports filtering, good for large corpora
- LanceDB: Simpler, embedded, good for local development
- Common interface allows easy switching

**Consequences**:
- Slightly more complex codebase
- Flexibility in deployment options
- Need to test both backends

#### AD-003: Dataset Strategy (2025-02-03)
**Decision**: Use HuggingFace datasets library with local caching.

**Context**: Need to load multiple benchmark datasets efficiently.

**Rationale**:
- HuggingFace provides unified API for all target datasets
- Automatic caching avoids repeated downloads
- Works on both local and cloud environments

**Consequences**:
- Dependency on HuggingFace infrastructure
- Large initial download (~13GB total)
- Cache management needed for disk space

### Assumptions

1. **Page Distribution**: Assumed 80% text-heavy, 20% visual-critical pages in technical manuals
2. **Latency Targets**: Text path <50ms, Vision path <400ms based on ColPali benchmarks
3. **Hardware**: Primary development on M1 Pro (16GB), fine-tuning on T4 GPU (16GB VRAM)
4. **Accuracy Target**: >90% of ColPali's Recall@10 is acceptable trade-off for latency reduction

### Experiment Log

#### Experiment Template
Use this template for documenting experiments:

```
#### Experiment: [EXP_ID] - [Brief Description] (YYYY-MM-DD)

**Objective**: What are we testing?

**Configuration**:
- Router: [heuristic/ml]
- Vision Model: [colpali/siglip]
- Text Model: [nomic-embed-text]
- LoRA Weights: [path or "none"]
- Dataset: [REAL-MM-RAG/DocVQA/ViDoRe]
- Random Seed: [42]

**Results**:
- Recall@1: [0.XXX]
- Recall@5: [0.XXX]
- Recall@10: [0.XXX]
- MRR: [0.XXX]
- NDCG: [0.XXX]
- Mean Latency: [XXX ms]
- Throughput: [XX pages/sec]

**Observations**:
- Key findings
- Unexpected behaviors
- Performance bottlenecks

**Next Steps**:
- What to try next
- Hypotheses to test
```

---

## [0.1.0] - 2025-02-03

### Added
- Project initialization
- Basic directory structure
- Configuration schema
- README with project overview
