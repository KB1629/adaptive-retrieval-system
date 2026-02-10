# Adaptive Retrieval System - Implementation Plan
## Based on Today's Research & Debugging (Feb 4, 2026)

---

## Executive Summary

Today we discovered critical issues with our current implementation and identified the correct path forward. The core problem: **our heuristic router is inaccurate** (misclassifying text pages with dark backgrounds) and **PaddleOCR is misconfigured** (loading 5 unnecessary models, taking 200+ seconds per page instead of milliseconds).

**Solution:** Implement a proper vision-based router + optimized OCR pipeline.

---

## Current Status

### What Works ✅
- ColPali vision model: ~4 seconds/page on M1 Pro
- PaddleOCR installed and functional (when it finally loads)
- Basic pipeline architecture in place
- 100 DocVQA pages downloaded for benchmarking

### What's Broken ❌
1. **Heuristic Router** - Feature-based classification is unreliable
   - Page 8 (text-only with dark background) → classified as "visual-critical"
   - Uses simple features (white ratio, edge density) that fail on real documents
   
2. **PaddleOCR Configuration** - Loading unnecessary preprocessing modules
   - PP-LCNet_x1_0_doc_ori (document orientation)
   - UVDoc (document unwarping)
   - PP-LCNet_x1_0_textline_ori (text line orientation)
   - PP-OCRv5_server_det (detection) ← NEED THIS
   - en_PP-OCRv5_mobile_rec (recognition) ← NEED THIS
   - **Result:** 200+ seconds per page (should be milliseconds to few seconds)

3. **Router Threshold** - text_threshold=0.5 is arbitrary and ineffective

---

## Research Findings

### Best Vision Router Options

Based on research, here are the top candidates for document layout classification:

#### Option 1: Microsoft DiT (Document Image Transformer) ⭐ RECOMMENDED
- **Model:** `microsoft/dit-base-finetuned-rvlcdip` (document classification)
- **Size:** ~350MB (base model)
- **Speed:** ~50-100ms per image on CPU, <10ms on GPU
- **Accuracy:** 92.69% on RVL-CDIP (document classification)
- **Use Case:** Perfect for text-heavy vs visual-critical classification
- **HuggingFace:** Available, easy to integrate
- **Pros:** 
  - Specifically trained for document understanding
  - Fast inference
  - Pre-trained on document layouts
  - Works well on M1 Pro MPS
- **Cons:** Slightly larger than MobileNet

#### Option 2: LayoutLMv3
- **Model:** `microsoft/layoutlmv3-base`
- **Size:** ~500MB
- **Speed:** ~100-200ms per image
- **Accuracy:** Excellent for document understanding
- **Pros:** Multimodal (text + layout + image)
- **Cons:** Overkill for simple routing, slower than DiT

#### Option 3: MobileNetV3 (Generic Vision)
- **Model:** `google/mobilenet_v3_small_100_224`
- **Size:** ~10MB
- **Speed:** ~20-30ms per image
- **Pros:** Extremely lightweight and fast
- **Cons:** Not document-specific, would need fine-tuning

**DECISION: Use Microsoft DiT** - Best balance of speed, accuracy, and document-specific training.

---

## Implementation Plan

### Phase 1: Fix PaddleOCR Configuration (CRITICAL)

**File:** `adaptive-retrieval-system/src/embedding/text_extractor.py`

**Changes:**
```python
# In _load_model():
self._ocr = PaddleOCR(
    lang=self.lang,
    use_angle_cls=False,  # Disable angle classification
    use_gpu=False,  # M1 Pro uses MPS, not CUDA
    show_log=False,
    # CRITICAL: Disable document preprocessing
    use_doc_preprocessor=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

# In extract_from_image():
result = self._ocr.ocr(image_input, cls=False)  # Disable per-call angle classification
```

**Expected Result:** OCR latency drops from 200s → 1-5s per page

---

### Phase 2: Implement Vision Router

**New File:** `adaptive-retrieval-system/src/router/vision_router.py`

**Implementation:**
```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

class VisionRouter(BaseRouter):
    """
    Vision-based router using Microsoft DiT for document classification.
    
    Classifies pages as text-heavy vs visual-critical based on layout analysis.
    """
    
    def __init__(self, config: RouterConfig = None):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._device = None
        self._load_model()
    
    def _load_model(self):
        """Load DiT model once at startup."""
        device_type = detect_device()
        self._device = get_device_string(device_type)
        
        # Use DiT fine-tuned for document classification
        model_name = "microsoft/dit-base-finetuned-rvlcdip"
        
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModelForImageClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()
    
    def _classify_impl(self, page: Page) -> ClassificationResult:
        """Classify using DiT model."""
        # Preprocess
        inputs = self._processor(images=page.image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Map DiT classes to text-heavy/visual-critical
        # DiT has 16 document classes - we map them to our 2 classes
        text_heavy_classes = [0, 1, 2, 3, 4]  # letter, form, email, handwriting, memo
        visual_classes = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # advertisement, budget, invoice, presentation, etc.
        
        text_prob = probs[0][text_heavy_classes].sum().item()
        visual_prob = probs[0][visual_classes].sum().item()
        
        if text_prob > visual_prob:
            return ClassificationResult(
                modality="text-heavy",
                confidence=text_prob,
                features={"text_prob": text_prob, "visual_prob": visual_prob}
            )
        else:
            return ClassificationResult(
                modality="visual-critical",
                confidence=visual_prob,
                features={"text_prob": text_prob, "visual_prob": visual_prob}
            )
```

**Expected Result:** Accurate classification of text vs visual pages, ~50-100ms per page

---

### Phase 3: Update Benchmark Script

**File:** `adaptive-retrieval-system/scripts/run_real_benchmark.py`

**Changes:**
```python
# Replace HeuristicRouter with VisionRouter
from src.router.vision_router import VisionRouter

# In __init__:
self.router = VisionRouter(config=router_config)
```

---

### Phase 4: Run Benchmarks

1. **Pure ColPali Baseline** (already done)
   - Result: 2820.56 ms/page on 100 pages

2. **Adaptive System with Vision Router + Optimized PaddleOCR**
   - Expected: 1000-2000 ms/page (50-70% faster than pure ColPali)
   - Text path: ~1-5s (OCR + embedding)
   - Vision path: ~4s (ColPali)
   - Router: ~0.05-0.1s (DiT)

3. **Generate Comparison Report**
   - Use existing scripts: `generate_comparison_report.py` and `generate_comparison_charts.py`

---

## Alternative: Keep Heuristic Router (Not Recommended)

If we can't use DiT due to model size/speed constraints, we can improve the heuristic router:

**Changes to `heuristic.py`:**
1. Lower text_threshold from 0.5 → 0.4
2. Adjust weights to prioritize text_density over white_ratio
3. Add special handling for dark backgrounds

**Pros:** No new dependencies, very fast (<1ms)
**Cons:** Still inaccurate, will misclassify edge cases

---

## Final Architecture

```
┌─────────────────┐
│  Document Page  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Vision Router (DiT)    │  ← 50-100ms
│  Text-heavy vs Visual   │
└────────┬────────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌──────────┐
│  Text  │  │  Visual  │
│  Path  │  │  Path    │
└───┬────┘  └────┬─────┘
    │            │
    ▼            ▼
┌────────────┐  ┌──────────────┐
│ PaddleOCR  │  │   ColPali    │
│ (det+rec)  │  │   (vision)   │
│  1-5s      │  │    ~4s       │
└─────┬──────┘  └──────┬───────┘
      │                │
      ▼                ▼
┌────────────┐  ┌──────────────┐
│    Text    │  │    Vision    │
│ Embeddings │  │  Embeddings  │
└─────┬──────┘  └──────┬───────┘
      │                │
      └────────┬───────┘
               ▼
      ┌────────────────┐
      │  Vector Store  │
      │   (Unified)    │
      └────────────────┘
```

---

## Success Metrics

1. **Router Accuracy:** >90% correct classification on DocVQA
2. **OCR Speed:** <5 seconds per page (down from 200s)
3. **Overall Speedup:** 1.5-2x faster than pure ColPali on mixed documents
4. **Text Path:** Faster than vision path for text-heavy pages

---

## Next Steps

1. ✅ Fix PaddleOCR configuration (DONE - code updated)
2. ⏳ Implement VisionRouter with DiT
3. ⏳ Update benchmark script to use VisionRouter
4. ⏳ Run benchmarks on 100 pages
5. ⏳ Generate comparison report
6. ⏳ Update README with findings

---

## Timeline

- **Phase 1 (PaddleOCR fix):** COMPLETED
- **Phase 2 (Vision Router):** 1-2 hours
- **Phase 3 (Benchmark update):** 30 minutes
- **Phase 4 (Run benchmarks):** 10-15 minutes
- **Total:** ~3-4 hours to complete implementation

---

## References

- [DiT Paper](https://arxiv.org/abs/2203.02378) - Microsoft Research
- [DiT HuggingFace](https://huggingface.co/microsoft/dit-base-finetuned-rvlcdip)
- [PaddleOCR Docs](https://github.com/PaddlePaddle/PaddleOCR)
- [ColPali Paper](https://arxiv.org/abs/2407.01449)
