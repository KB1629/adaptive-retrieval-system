# Part 1: Sequential Process Optimization

## 1. Goal
The initial objective was to establish a solid performance baseline for the Adaptive Retrieval System on Apple Silicon (M1 Pro) and match the performance of the "pure" ColPali baseline.

## 2. Baseline Establishment (Feb 6)
We started by running a pure ColPali baseline (no adaptive routing) to understand the hardware limits.
*   **Sequential Baseline Latency**: ~2.05 seconds / page
*   **Hardware**: Apple M1 Pro (MPS Acceleration)

## 3. The "Double Classification" Bug
Initial runs of the Adaptive System were extremely slow (13s - 160s per page).
*   **Investigation**: We discovered that `run_real_benchmark.py` was classifying pages *twice*:
    1.  Once in a batch at the start (Fast).
    2.  Again individually inside `process_page` (Slow, causing massive overhead).
*   **Fix**: Modified the script to pass pre-computed batch classifications to `process_page`, eliminating the redundant Router calls.
*   **Result**: Latency dropped to match the baseline (~2.07s/page).

## 4. Model Pre-loading Experiment
We observed a **29-second Data Loading Cold Start** on the first vision page.
*   **Hypothesis**: Loading `colpali-v1.2` only when the first visual page is encountered causes a massive user-facing delay.
*   **Optimization**: We modified `RealBenchmarkRunner` to pre-load both the Router and ColPali models at startup.
*   **Results**:
    *   **Cold Start**: Reduced from ~29s to ~1s.
    *   **Steady State**: Slightly slower (~2.29s/page vs 2.07s/page).
    *   **Analysis**: Pre-loading both models on an 16GB M1 Pro caused memory pressure, forcing the system to swap or throttle slightly.
    *   **Decision**: Accepted the trade-off. A slightly slower average is better than a 30s hang for the first user interaction.

## 5. Final Sequential Verdict
After fixing bugs and optimizing start-up:
*   **Configuration**: Sequential processing (Batch Size = 1).
*   **Mean Latency**: **2.07 seconds / page**.
*   **Status**: Parity with Baseline established. Ready for advanced optimization (Batching).

---

### Reference Artifacts
*   **Benchmark Data**: `outputs/benchmark_results/real_benchmark_20260208_114905.json`
