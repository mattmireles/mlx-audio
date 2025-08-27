#!/usr/bin/env python3
"""
Kokoro TTS Benchmarking Tool

Tests different Kokoro TTS models (full precision and quantized) for:
- Performance metrics (latency, throughput)
- Audio quality comparison
- Memory usage analysis
- Real-time factors

Usage:
    python kokoro_benchmark.py
"""

import time
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

import mlx.core as mx
import numpy as np
import soundfile as sf

# Add mlx_audio to path if running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from mlx_audio.tts.audio_player import AudioPlayer
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.models.kokoro import KokoroPipeline

# Available Kokoro models with their descriptions
AVAILABLE_MODELS = {
    "1": {
        "name": "mlx-community/Kokoro-82M-bf16",
        "description": "Kokoro 82M (bfloat16 - Full Precision)",
        "precision": "bf16",
        "quantization": None
    },
    "2": {
        "name": "mlx-community/Kokoro-82M-8bit", 
        "description": "Kokoro 82M (8-bit Quantized)",
        "precision": "8bit",
        "quantization": "8bit"
    },
    "3": {
        "name": "mlx-community/Kokoro-82M-6bit",
        "description": "Kokoro 82M (6-bit Quantized)", 
        "precision": "6bit",
        "quantization": "6bit"
    },
    "4": {
        "name": "mlx-community/Kokoro-82M-4bit",
        "description": "Kokoro 82M (4-bit Quantized)",
        "precision": "4bit", 
        "quantization": "4bit"
    }
}

# Available voices
AVAILABLE_VOICES = [
    "af_heart", "af_bella", "af_sarah", "af_nicole", "af_sky", "af_river",
    "am_michael", "am_adam", "am_eric", "am_liam", "am_onyx", "am_puck"
]

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    model_name: str
    model_description: str
    text: str
    voice: str
    
    # Timing metrics
    model_load_time: float
    inference_time: float
    total_time: float
    
    # Audio metrics
    audio_duration: float
    real_time_factor: float
    samples_per_sec: float
    
    # Memory metrics
    peak_memory_gb: float
    
    # Generated audio
    audio_data: Optional[np.ndarray] = None
    sample_rate: int = 24000


class KokoroBenchmark:
    """Main benchmarking class for Kokoro TTS models"""
    
    def __init__(self):
        self.models_cache: Dict[str, any] = {}
        self.results: List[BenchmarkResult] = []
        self.output_dir = Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def print_banner(self):
        """Print the application banner"""
        print("=" * 70)
        print("🎤 KOKORO TTS BENCHMARK TOOL")
        print("=" * 70)
        print("Compare performance and quality across different Kokoro models")
        print()

    def select_model(self) -> str:
        """Interactive model selection"""
        print("📋 Available Models:")
        print()
        
        for key, model_info in AVAILABLE_MODELS.items():
            print(f"  {key}. {model_info['description']}")
        
        print()
        while True:
            try:
                choice = input("Select model (1-4) or 'all' for comparison: ").strip().lower()
                
                if choice == 'all':
                    return 'all'
                elif choice in AVAILABLE_MODELS:
                    return AVAILABLE_MODELS[choice]["name"]
                else:
                    print("❌ Invalid selection. Please try again.")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                sys.exit(0)
    
    def select_voice(self) -> str:
        """Interactive voice selection"""
        print("\n🎭 Available Voices:")
        print()
        
        # Group voices by gender
        female_voices = [v for v in AVAILABLE_VOICES if v.startswith('af_')]
        male_voices = [v for v in AVAILABLE_VOICES if v.startswith('am_')]
        
        print("Female voices:", ", ".join(female_voices))
        print("Male voices:  ", ", ".join(male_voices))
        print()
        
        while True:
            voice = input(f"Select voice (default: af_heart): ").strip()
            
            if not voice:
                return "af_heart"
            elif voice in AVAILABLE_VOICES:
                return voice
            else:
                print("❌ Invalid voice. Please select from the list above.")

    def get_test_text(self) -> str:
        """Get text input for TTS generation"""
        print("\n📝 Text Input:")
        print("Enter text to synthesize (or press Enter for default):")
        
        text = input("> ").strip()
        
        if not text:
            text = ("Hello, this is a benchmark test of the Kokoro text-to-speech system. "
                   "We are testing different model quantization levels for performance and quality.")
            print(f"Using default text: {text}")
        
        return text

    def load_model_with_timing(self, model_name: str) -> Tuple[any, float]:
        """Load model and measure load time"""
        if model_name in self.models_cache:
            return self.models_cache[model_name], 0.0
        
        print(f"🔄 Loading model: {model_name}...")
        
        start_time = time.perf_counter()
        
        try:
            model = load_model(model_path=model_name)
            load_time = time.perf_counter() - start_time
            
            self.models_cache[model_name] = model
            
            print(f"✅ Model loaded in {load_time:.2f}s")
            return model, load_time
            
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {e}")
            raise

    def benchmark_single_model(self, model_name: str, text: str, voice: str) -> BenchmarkResult:
        """Benchmark a single model"""
        model_info = None
        for info in AVAILABLE_MODELS.values():
            if info["name"] == model_name:
                model_info = info
                break
        
        if not model_info:
            raise ValueError(f"Unknown model: {model_name}")
            
        print(f"\n🚀 Benchmarking: {model_info['description']}")
        print("-" * 50)
        
        # Load model with timing
        model, load_time = self.load_model_with_timing(model_name)
        
        # Create pipeline
        pipeline = KokoroPipeline(
            lang_code="a",  # American English
            model=model,
            repo_id=model_name
        )
        
        # Measure generation time
        print("🎵 Generating audio...")
        
        # Clear memory cache before inference
        mx.clear_cache()
        
        # Get peak memory using non-deprecated API
        memory_before = mx.get_peak_memory() / (1024**3)  # Convert to GB
        
        start_time = time.perf_counter()
        
        # Generate audio 
        results_gen = pipeline(text, voice=voice, speed=1.0)
        
        # Collect all audio segments
        audio_segments = []
        total_samples = 0
        
        for result in results_gen:
            if result.audio is not None:
                seg = np.asarray(result.audio).reshape(-1)
                audio_segments.append(seg)
                total_samples += seg.shape[0]
        
        inference_time = time.perf_counter() - start_time
        
        # Get peak memory after inference
        memory_after = mx.get_peak_memory() / (1024**3)  # Convert to GB
        peak_memory = memory_after
        
        # Concatenate audio segments (flatten to 1D per segment)
        if audio_segments:
            audio_data = np.concatenate(audio_segments, axis=0)
        else:
            raise RuntimeError("No audio generated")
            
        # Calculate metrics
        audio_duration = audio_data.shape[0] / 24000  # Assuming 24kHz sample rate
        real_time_factor = inference_time / audio_duration if audio_duration > 0 else float('inf')
        samples_per_sec = total_samples / inference_time if inference_time > 0 else 0
        total_time = load_time + inference_time
        
        result = BenchmarkResult(
            model_name=model_name,
            model_description=model_info['description'],
            text=text,
            voice=voice,
            model_load_time=load_time,
            inference_time=inference_time, 
            total_time=total_time,
            audio_duration=audio_duration,
            real_time_factor=real_time_factor,
            samples_per_sec=samples_per_sec,
            peak_memory_gb=peak_memory,
            audio_data=audio_data,
            sample_rate=24000
        )
        
        # Print results
        print(f"⏱️  Model Load Time: {load_time:.2f}s")
        print(f"⏱️  Inference Time: {inference_time:.2f}s") 
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print(f"🎵 Audio Duration: {audio_duration:.2f}s")
        print(f"⚡ Real-time Factor: {real_time_factor:.2f}x")
        print(f"📈 Samples/sec: {samples_per_sec:.1f}")
        print(f"🧠 Peak Memory: {peak_memory:.2f}GB")
        
        return result

    def play_audio(self, result: BenchmarkResult) -> bool:
        """Play generated audio"""
        if result.audio_data is None:
            return False
            
        try:
            print(f"\n🔊 Playing audio from {result.model_description}...")
            
            player = AudioPlayer(sample_rate=result.sample_rate, verbose=False)
            player.queue_audio(result.audio_data)
            player.wait_for_drain()
            player.stop()
            
            return True
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
            return False

    def save_audio_files(self, results: List[BenchmarkResult]):
        """Save generated audio files for comparison"""
        print(f"\n💾 Saving audio files to {self.output_dir}/")
        
        for i, result in enumerate(results):
            if result.audio_data is not None:
                # Create safe filename
                model_name = result.model_name.split("/")[-1]
                filename = f"{i+1:02d}_{model_name}_{result.voice}.wav"
                filepath = self.output_dir / filename
                
                sf.write(filepath, result.audio_data, result.sample_rate)
                print(f"  📁 {filename}")

    def save_benchmark_results(self, results: List[BenchmarkResult]):
        """Save benchmark results to JSON file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        results_data = []
        for result in results:
            result_dict = {
                "model_name": result.model_name,
                "model_description": result.model_description,
                "text": result.text,
                "voice": result.voice,
                "model_load_time": result.model_load_time,
                "inference_time": result.inference_time,
                "total_time": result.total_time,
                "audio_duration": result.audio_duration,
                "real_time_factor": result.real_time_factor,
                "samples_per_sec": result.samples_per_sec,
                "peak_memory_gb": result.peak_memory_gb,
                "sample_rate": result.sample_rate
            }
            results_data.append(result_dict)
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        print(f"📊 Results saved to {filename}")

    def print_comparison_table(self, results: List[BenchmarkResult]):
        """Print comparison table of all results"""
        if len(results) <= 1:
            return
            
        print("\n" + "=" * 100)
        print("📊 BENCHMARK COMPARISON")
        print("=" * 100)
        
        # Table headers
        print(f"{'Model':<25} {'Load(s)':<8} {'Infer(s)':<9} {'Total(s)':<9} {'RTF':<6} {'Samples/s':<10} {'Memory(GB)':<11}")
        print("-" * 100)
        
        # Sort by total time for comparison
        sorted_results = sorted(results, key=lambda r: r.total_time)
        
        for result in sorted_results:
            model_short = result.model_name.split("/")[-1][:24]
            print(f"{model_short:<25} {result.model_load_time:<8.2f} {result.inference_time:<9.2f} "
                  f"{result.total_time:<9.2f} {result.real_time_factor:<6.2f} "
                  f"{result.samples_per_sec:<10.0f} {result.peak_memory_gb:<11.2f}")
        
        print("-" * 100)
        
        # Find best performer
        fastest = min(results, key=lambda r: r.inference_time)
        print(f"🏆 Fastest inference: {fastest.model_description} ({fastest.inference_time:.2f}s)")
        
        most_efficient = min(results, key=lambda r: r.real_time_factor) 
        print(f"⚡ Most efficient (lowest RTF): {most_efficient.model_description} ({most_efficient.real_time_factor:.2f}x)")

    def run_benchmark(self):
        """Main benchmark execution"""
        self.print_banner()
        
        try:
            # Get user selections
            model_choice = self.select_model()
            voice = self.select_voice() 
            text = self.get_test_text()
            
            # Determine models to test
            if model_choice == 'all':
                models_to_test = [info["name"] for info in AVAILABLE_MODELS.values()]
            else:
                models_to_test = [model_choice]
            
            # Run benchmarks
            results = []
            for model_name in models_to_test:
                try:
                    result = self.benchmark_single_model(model_name, text, voice)
                    results.append(result)
                    self.results.append(result)
                except Exception as e:
                    print(f"❌ Failed to benchmark {model_name}: {e}")
                    continue
                    
                # Ask if user wants to hear the audio
                if len(models_to_test) == 1:  # Only for single model
                    play_choice = input("\n🔊 Play generated audio? (y/n): ").strip().lower()
                    if play_choice == 'y':
                        self.play_audio(result)
            
            # Show comparison if multiple models
            if len(results) > 1:
                self.print_comparison_table(results)
                
                # Ask about playing audio samples
                play_choice = input("\n🔊 Play audio samples for comparison? (y/n): ").strip().lower()
                if play_choice == 'y':
                    for result in results:
                        print(f"\nPlaying: {result.model_description}")
                        self.play_audio(result)
                        input("Press Enter to continue to next sample...")
            
            # Save results
            if results:
                save_choice = input(f"\n💾 Save results to {self.output_dir}? (y/n): ").strip().lower()
                if save_choice == 'y':
                    self.save_audio_files(results)
                    self.save_benchmark_results(results)
                    print("✅ Results saved successfully!")
            
            print("\n✨ Benchmark complete!")
            
        except KeyboardInterrupt:
            print("\n\n👋 Benchmark interrupted by user. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point"""
    benchmark = KokoroBenchmark()
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()