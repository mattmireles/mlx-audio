#!/usr/bin/env python3
"""
Memory profiler for Kokoro TTS models (MLX)

Runs repeated inference and logs active vs peak memory across iterations
for both a full-precision (bf16) model and a quantized (4-bit) model.

Outputs:
- Console logs per iteration
- Optional JSON summary written to benchmark_results/memory_profile.json
"""

import gc
import json
import time
import sys
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from mlx_audio.tts.utils import load_model
from mlx_audio.tts.models.kokoro import KokoroPipeline

RESULTS_DIR = REPO_ROOT / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = [
    {
        "name": "mlx-community/Kokoro-82M-bf16",
        "label": "bf16",
    },
    {
        "name": "mlx-community/Kokoro-82M-4bit",
        "label": "4bit",
    },
]

TEST_TEXT = "This is a memory profiling test for the Kokoro TTS model."
VOICE = "af_heart"
NUM_ITERATIONS = 10


def run_one_iteration(pipeline: KokoroPipeline, text: str, voice: str) -> np.ndarray:
    # Consume pipeline generator and concatenate the audio segments
    segments = []
    for res in pipeline(text, voice=voice):
        if res.audio is not None:
            seg = np.asarray(res.audio).reshape(-1)
            segments.append(seg)
    if not segments:
        return np.empty((0,), dtype=np.float32)
    audio = np.concatenate(segments, axis=0)
    # Ensure compute completes on device
    mx.eval(mx.array(audio))
    return audio


def profile_model(model_name: str, label: str) -> Dict:
    print(f"\n=== Profiling: {label} -> {model_name} ===")

    # Load once
    t0 = time.perf_counter()
    model = load_model(model_path=model_name)
    pipe = KokoroPipeline(lang_code="a", model=model, repo_id=model_name)
    load_s = time.perf_counter() - t0
    print(f"Model+Pipeline loaded in {load_s:.2f}s")

    # Baseline memory
    gc.collect()
    mx.clear_cache()
    baseline_active = getattr(mx.metal, "get_active_memory", lambda: 0)() / 1e9
    baseline_peak = mx.get_peak_memory() / 1e9
    print(f"Baseline: Active={baseline_active:.3f} GB | Peak={baseline_peak:.3f} GB")

    iters: List[Dict] = []

    for i in range(NUM_ITERATIONS):
        print(f"Iteration {i+1}/{NUM_ITERATIONS}...")
        t1 = time.perf_counter()
        audio = run_one_iteration(pipe, TEST_TEXT, VOICE)
        infer_s = time.perf_counter() - t1

        # GC + clear caches
        gc.collect()
        mx.clear_cache()

        active_gb = getattr(mx.metal, "get_active_memory", lambda: 0)() / 1e9
        peak_gb = mx.get_peak_memory() / 1e9
        rel_peak_gb = max(0.0, peak_gb - baseline_peak)

        dur_s = audio.shape[0] / 24000.0 if audio.size > 0 else 0.0
        rtf = (infer_s / dur_s) if dur_s > 0 else float("inf")

        print(
            f"  time={infer_s:.2f}s, audio={dur_s:.2f}s, RTF={rtf:.2f}, "
            f"Active={active_gb:.3f} GB, Peak={peak_gb:.3f} GB (Δ={rel_peak_gb:.3f} GB)"
        )

        iters.append(
            {
                "iteration": i + 1,
                "inference_s": infer_s,
                "audio_s": dur_s,
                "rtf": None if dur_s == 0 else infer_s / dur_s,
                "active_gb": active_gb,
                "peak_gb": peak_gb,
                "peak_delta_gb": rel_peak_gb,
                "samples": int(audio.shape[0]),
            }
        )

    return {
        "model": model_name,
        "label": label,
        "load_s": load_s,
        "baseline_active_gb": baseline_active,
        "baseline_peak_gb": baseline_peak,
        "iterations": iters,
    }


def main():
    all_results = []
    for m in MODELS:
        res = profile_model(m["name"], m["label"])
        all_results.append(res)

    out_path = RESULTS_DIR / "memory_profile.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved detailed results to {out_path}")

    # Quick comparison summary
    print("\n=== Summary (Active vs Peak) ===")
    for r in all_results:
        label = r["label"]
        last = r["iterations"][-1]
        print(
            f"{label:>5}: Active={last['active_gb']:.3f} GB | Peak={last['peak_gb']:.3f} GB | Δ={last['peak_delta_gb']:.3f} GB"
        )


if __name__ == "__main__":
    main()
