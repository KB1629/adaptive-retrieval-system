# Adaptive Retrieval System - Project Steering Guide

## Project Context

This is a **research project** targeting academic publication (conference/journal). The goal is to optimize latency in Technical RAG via Semantic Routing and Domain-Specific Vision Fine-Tuning.

**Core Objective**: Achieve >90% of ColPali's retrieval accuracy with 50% less latency.

**Primary Spec Reference**: `#[[file:.kiro/specs/adaptive-retrieval-system/requirements.md]]`
**Design Reference**: `#[[file:.kiro/specs/adaptive-retrieval-system/design.md]]`
**Task Reference**: `#[[file:.kiro/specs/adaptive-retrieval-system/tasks.md]]`

---

## Project Guidelines

### Code Structure & Organization

1. **Follow the project structure** defined in `tasks.md`:
   - Source code in `src/` with modular subdirectories
   - Tests in `tests/` with pytest and Hypothesis
   - Notebooks in `notebooks/` for Colab/Kaggle
   - Configuration in `configs/` with YAML files
   - Research papers in `references/`

2. **Keep code simple and readable**:
   - Each module should do ONE thing well
   - Avoid over-engineering
   - Use type hints consistently
   - Write docstrings for public functions

3. **Naming conventions**:
   - Files: `snake_case.py`
   - Classes: `PascalCase`
   - Functions/variables: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`

### Documentation Requirements

1. **Update README.md** at each checkpoint (tasks 3, 5, 7, 10, 13, 15, 19):
   - Add new features and usage examples
   - Update installation instructions if dependencies change
   - Include performance expectations

2. **Maintain CHANGELOG.md** for:
   - Architecture decisions and rationale
   - Assumption changes
   - Direction pivots
   - Breaking changes

3. **Document assumptions** in code comments when:
   - Making design trade-offs
   - Choosing between alternatives
   - Implementing workarounds

### Testing Requirements

1. **Property-Based Tests** (Hypothesis):
   - 20 properties defined in design.md
   - Minimum 100 iterations per property
   - Tag format: `# Feature: adaptive-retrieval-system, Property N: [text]`

2. **Unit Tests**:
   - Test specific examples and edge cases
   - Co-locate with source when possible (`.test.ts` pattern)
   - Focus on core logic, not mocks

3. **Run tests before commits**:
   - All tests must pass at checkpoints
   - Ask user if tests fail repeatedly

### Hardware Considerations

1. **Local Development (M1 Pro)**:
   - Use MPS backend for PyTorch
   - Router and text embedding run locally
   - Vision inference possible but slower

2. **Cloud Fine-Tuning (Colab/Kaggle T4)**:
   - LoRA fine-tuning only
   - Optimize batch sizes for 16GB VRAM
   - Save checkpoints frequently

3. **Automatic detection**:
   - System auto-detects MPS/CUDA/CPU
   - Graceful fallback to CPU if needed

---

## Research Paper References

When implementing, refer to these baseline papers:

| Paper | arXiv | Purpose |
|-------|-------|---------|
| ColPali | 2407.01449 | Main baseline to beat |
| HPC-ColPali | 2506.21601 | Competition (pruning approach) |
| REAL-MM-RAG | 2502.12342 | Benchmark dataset |
| Multimodal RAG Comparison | 2511.16654 | Validates vision > text for diagrams |

**Key metrics to track**:
- Recall@1, Recall@5, Recall@10
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
- Latency (mean, median, P95)
- Throughput (pages/second)

---

## Dataset Information

**Primary datasets** (HuggingFace):
- `ibm-research/REAL-MM-RAG_TechReport` (~2.2k pages, ~2GB)
- `ibm-research/REAL-MM-RAG_TechSlides` (~2.6k pages, ~3GB)
- `lmms-lab/DocVQA` (~10.5k samples, ~5GB)
- ViDoRe benchmark (GitHub illuin-tech)

**Caching strategy**:
- Use HuggingFace `datasets` library
- Cache persists in `~/.cache/huggingface/datasets/`
- Can set `HF_DATASETS_CACHE` for custom location

---

## Implementation Priorities

### Phase 1: Foundation (Tasks 1-5)
- Project setup, data models, data loading
- **Focus**: Get datasets loading correctly

### Phase 2: Core Pipeline (Tasks 6-13)
- Router, embedding paths, vector DB, retrieval
- **Focus**: End-to-end pipeline working

### Phase 3: Evaluation (Tasks 14-15)
- Benchmark framework, metrics
- **Focus**: Reproducible evaluation

### Phase 4: Optimization (Tasks 16-17)
- Fine-tuning, experiment tracking
- **Focus**: Beat ColPali baseline

### Phase 5: Polish (Tasks 18-20)
- Integration, CLI, documentation
- **Focus**: Publication-ready code

---

## Common Patterns

### Error Handling
```python
# Fallback pattern for router
try:
    result = router.classify(page)
except Exception as e:
    logger.error(f"Router failed: {e}")
    result = ClassificationResult(modality="visual-critical", confidence=0.0)
```

### Configuration Loading
```python
# Use YAML configs
import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)
```

### Hardware Detection
```python
# Auto-detect device
import torch
device = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Quality Checklist

Before completing any task:
- [ ] Code follows project structure
- [ ] Type hints added
- [ ] Docstrings for public functions
- [ ] Tests written and passing
- [ ] No hardcoded paths (use config)
- [ ] Error handling implemented
- [ ] Logging added for debugging

Before each checkpoint:
- [ ] All tests pass
- [ ] README updated if needed
- [ ] CHANGELOG updated for decisions
- [ ] Code reviewed for simplicity

---

## Evolving Requirements

This is a **long-running research project**. Requirements may evolve as:
- Experiments reveal new insights
- Baselines change (new papers)
- Hardware constraints discovered
- Publication requirements clarify

**When requirements change**:
1. Update `requirements.md` with new criteria
2. Update `design.md` if architecture affected
3. Update `tasks.md` with new/modified tasks
4. Document change in `CHANGELOG.md`
5. Notify user of impact on timeline

---

## Contact Points

- **Spec files**: `.kiro/specs/adaptive-retrieval-system/`
- **This steering guide**: `.kiro/steering/adaptive-retrieval-system.md`
- **Project root**: `adaptive-retrieval-system/` (to be created)

Always refer to spec files as the source of truth for requirements and design decisions.
