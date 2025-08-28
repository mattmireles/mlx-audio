# Memory Anomaly Investigation Plan

**Andy Hertzfeld**

### 1. Executive Summary

We have observed a counter-intuitive memory behavior when running inference with the Kokoro TTS models: quantized models (4-bit, 6-bit, 8-bit) are exhibiting a *higher* peak memory footprint than their `bfloat16` full-precision counterpart. This is unexpected, as quantization is designed to reduce memory usage.

**Primary Hypothesis:** The MLX framework is performing on-the-fly de-quantization of the model weights into a higher-precision format (e.g., `bfloat16`) before computation. This creates a temporary state where both the compact quantized weights and the larger de-quantized weights exist in memory simultaneously, resulting in a higher peak memory watermark than loading the `bfloat16` model directly.

**Objective:** To design and execute a controlled experiment that validates this hypothesis, determines if there is a memory leak, and provides clear data on the memory behavior of quantized models during repeated inference.

---

### 2. Action Plan

This investigation will be conducted by creating a dedicated, focused profiling script. This isolates the test from the complexity of the full benchmark suite and provides a clean environment for measurement.

#### Step 1: Create a Controlled Test Environment

*   **Action:** Create a new Python script named `memory_profiler.py` in the root of the project.
*   **Purpose:** To provide a minimal, reproducible test case for analyzing memory usage without the confounding variables of the larger benchmark application.

#### Step 2: Implement a Repetitive Inference Loop

*   **Action:** Within `memory_profiler.py`, implement the following logic:
    1.  Select a single, representative quantized model (e.g., `mlx-community/Kokoro-82M-4bit`).
    2.  Load the model and its associated pipeline once, outside of the loop.
    3.  Define a fixed, short text input for synthesis.
    4.  Create a `for` loop that runs inference on this text 10-15 times.

#### Step 3: Instrument Memory Profiling Inside the Loop

*   **Action:** Within the `for` loop, immediately following each inference call, perform a rigorous sequence of memory management and measurement operations.
*   **Instrumentation Code (to be run each iteration):**
    ```python
    import gc
    import mlx.core as mx

    # 1. Force Python's garbage collector to clean up any dangling references.
    gc.collect()

    # 2. Clear all MLX internal memory caches.
    # This is critical for getting an accurate reading of managed memory.
    mx.clear_cache()

    # 3. Measure and log memory usage.
    # Note: get_active_memory() is specific to the Metal backend.
    peak_mem = mx.get_peak_memory() / 1e9  # Peak since start of program (GB)
    # Prefer top-level API; metal.get_active_memory is deprecated
    active_mem = mx.get_active_memory() / 1e9  # Currently used by MLX tensors (GB)

    print(f"Iteration {i+1}: Active Memory = {active_mem:.3f} GB | Peak Memory = {peak_mem:.3f} GB")
    ```

#### Step 4: Analyze the Results

*   **Action:** Execute the script (`python memory_profiler.py`) and analyze the logged output.
*   **Interpretation Guide:**
    *   **If `active_mem` remains relatively stable across iterations:** This indicates there is **no memory leak**. The memory is being successfully reclaimed after each run. The high `peak_mem` value would then confirm our hypothesis of a temporary spike due to de-quantization.
    *   **If `active_mem` consistently increases with each iteration:** This strongly suggests a **memory leak**. Some objects are not being de-allocated correctly between runs.

---

### 3. Expected Outcomes

*   A clear, data-backed conclusion on the cause of the high memory usage in quantized models.
*   Definitive confirmation of whether a memory leak exists in the inference pipeline.
*   A solid foundation for the next steps, which could be:
    *   Accepting the memory spike as a known trade-off for faster inference.
    *   Investigating the implementation of true quantized computation kernels to avoid the de-quantization step.
    *   Filing a bug report with the MLX team if a framework-level leak is discovered.

---

### 4. Code Template for `memory_profiler.py`

To accelerate this process, use the following code as a starting point:

```python
import gc
import time
import sys
from pathlib import Path

import mlx.core as mx

# Add mlx_audio to path
sys.path.insert(0, str(Path(__file__).parent))
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.models.kokoro import KokoroPipeline

def run_inference_and_profile(pipeline, text, voice):
    """Runs a single inference and returns audio."""
    # Note: The pipeline itself is a generator. We need to consume it.
    audio_segments = [result.audio for result in pipeline(text, voice=voice)]
    return mx.concatenate(audio_segments, axis=0) if audio_segments else None

def main():
    MODEL_NAME = "mlx-community/Kokoro-82M-4bit"
    TEST_TEXT = "This is a memory profiling test for the Kokoro TTS model."
    VOICE = "af_heart"
    NUM_ITERATIONS = 15

    print(f"🚀 Starting memory profiling for: {MODEL_NAME}")
    print("-" * 50)

    # 1. Load the model and pipeline once
    try:
        model = load_model(model_path=MODEL_NAME)
        pipeline = KokoroPipeline(
            lang_code="a",
            model=model,
            repo_id=MODEL_NAME
        )
        print("✅ Model and pipeline loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Initial memory state
    mx.clear_cache()
    initial_active_mem = mx.metal.get_active_memory() / 1e9
    print(f"Initial Active Memory: {initial_active_mem:.3f} GB")
    print("-" * 50)

    # 2. Run inference in a loop
    for i in range(NUM_ITERATIONS):
        print(f"Running iteration {i+1}/{NUM_ITERATIONS}...")
        start_time = time.perf_counter()

        # Run inference
        audio = run_inference_and_profile(pipeline, TEST_TEXT, VOICE)
        mx.eval(audio) # Ensure computation is complete

        inference_time = time.perf_counter() - start_time
        print(f"   Inference time: {inference_time:.2f}s")

        # 3. Force GC, clear cache, and measure memory
        gc.collect()
        mx.clear_cache()

        active_mem = mx.metal.get_active_memory() / 1e9
        peak_mem = mx.get_peak_memory() / 1e9

        print(f"   Memory after run {i+1}: Active={active_mem:.3f}GB | Peak={peak_mem:.3f}GB")
        print("-" * 50)

    print("✨ Profiling complete.")

if __name__ == "__main__":
    main()

```

