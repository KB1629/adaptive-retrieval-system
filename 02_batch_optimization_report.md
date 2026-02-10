# Part 2: Batch Processing Optimization

## 1. Goal
The objective was to beat the Sequential Baseline (2.05s/page) by processing pages in batches (Batch Size = 4), theorizing that parallelizing the Vision Path would yield significant speedups (similar to GPU behavior).

## 2. The M1 Pro "Bandwidth Wall"
We implemented batching for both the Baseline and the Adaptive System.
*   **Sequential Baseline**: 2.05s / page
*   **Batched Baseline (Batch=4)**: ~2.30s / page (SLOWER)

**Discovery**: On M1 Pro (Unified Memory), processing 4 high-resolution images in parallel saturates the memory bandwidth. The overhead of padding and collating batches makes it *slower* than processing them one by one.
*   **Constraint**: Hardware limitation of the testing device.

## 3. The Text Path Breakthrough
While the Vision Path struggled with batching, the **Text Path (Tesseract + SentenceTransformer)** excelled.
*   **Text Path Latency (Batch=4)**: **0.83s / page** (Average)
*   **Speedup**: >2x faster than sequential text processing.
*   **Why**: CPU-bound text extraction and smaller embedding vectors are handled efficiently by the M1 CPU/MPS, scaling well with batching.

## 4. Final Adaptive Result
The Adaptive System combines the "slower" batched vision path with the "super-fast" batched text path.
*   **Traffic Split**: 49% Text-Heavy / 51% Visual-Critical.
*   **Calculation**: (49% × 0.83s) + (51% × 2.30s) ≈ 1.58s (Theoretical) -> **1.86s** (Real World).

## 5. Final Verdict (Batch=4)
*   **Mean Latency**: **1.86 seconds / page**.
*   **Speedup**: **10% Faster** than the best Sequential Baseline (2.05s).
*   **Conclusion**: Even though vision batching failed on this hardware, the **Adaptive System won** because it intelligently routed half the workload to a path that *could* be optimized.

---

### Reference Artifacts
*   **Baseline Data**: `outputs/benchmark_results/colpali_baseline_20260208_142035.json`
*   **Adaptive Data**: `outputs/benchmark_results/real_benchmark_20260208_142537.json`
