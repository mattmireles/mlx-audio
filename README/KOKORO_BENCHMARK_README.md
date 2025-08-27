# 🎤 Kokoro TTS Benchmark Tool

A comprehensive benchmarking tool for testing and comparing different Kokoro TTS models (full precision and quantized versions) on performance, latency, and audio quality.

## 🚀 Features

- **Interactive Model Selection**: Choose from 4 different Kokoro model variants
- **Performance Metrics**: Detailed timing analysis including load time, inference time, and real-time factors
- **Memory Usage Tracking**: Monitor peak memory consumption during generation
- **Audio Playback**: Automatic playback of generated audio for quality comparison
- **Batch Comparison**: Test all models with the same text/voice for direct comparison
- **Results Export**: Save audio files and benchmark metrics to JSON for analysis

## 📋 Available Models

1. **Kokoro 82M (bfloat16)** - `mlx-community/Kokoro-82M-bf16`
   - Full precision model for highest quality
2. **Kokoro 82M (8-bit)** - `mlx-community/Kokoro-82M-8bit` 
   - 8-bit quantized for balanced quality/performance
3. **Kokoro 82M (6-bit)** - `mlx-community/Kokoro-82M-6bit`
   - 6-bit quantized for better performance
4. **Kokoro 82M (4-bit)** - `mlx-community/Kokoro-82M-4bit`
   - 4-bit quantized for fastest inference

## 🎭 Available Voices

**Female**: af_heart, af_bella, af_sarah, af_nicole, af_sky, af_river  
**Male**: am_michael, am_adam, am_eric, am_liam, am_onyx, am_puck

## 📦 Prerequisites

Make sure you have the mlx-audio package installed with all dependencies:

```bash
# Install mlx-audio with dependencies
pip install -e .

# Ensure MLX and other dependencies are available
pip install mlx soundfile sounddevice numpy
```

## 🔧 Usage

### Basic Usage

```bash
python kokoro_benchmark.py
```

### Interactive Flow

1. **Select Model**: Choose a specific model (1-4) or 'all' for comparison
2. **Choose Voice**: Select from available female/male voices (default: af_heart)
3. **Enter Text**: Input text to synthesize (or use default benchmark text)
4. **Review Results**: View detailed performance metrics
5. **Listen to Audio**: Optional playback of generated speech
6. **Save Results**: Export audio files and metrics for further analysis

### Example Session

```
🎤 KOKORO TTS BENCHMARK TOOL
=======================================

📋 Available Models:
  1. Kokoro 82M (bfloat16 - Full Precision)
  2. Kokoro 82M (8-bit Quantized)
  3. Kokoro 82M (6-bit Quantized)
  4. Kokoro 82M (4-bit Quantized)

Select model (1-4) or 'all' for comparison: all

🎭 Available Voices:
Female voices: af_heart, af_bella, af_sarah, af_nicole, af_sky, af_river
Male voices: am_michael, am_adam, am_eric, am_liam, am_onyx, am_puck

Select voice (default: af_heart): af_heart

📝 Text Input:
Enter text to synthesize (or press Enter for default):
> Hello world, this is a test of the Kokoro TTS system.
```

## 📊 Metrics Explained

### Performance Metrics
- **Model Load Time**: Time to load model weights and initialize
- **Inference Time**: Time for actual text-to-speech generation
- **Total Time**: Combined load + inference time
- **Real-time Factor (RTF)**: Inference time / Audio duration (lower is better)
- **Samples/sec**: Audio samples generated per second

### Quality Metrics
- **Audio Duration**: Length of generated speech
- **Sample Rate**: Audio sampling frequency (typically 24kHz)
- **Peak Memory**: Maximum memory usage during generation

### Comparison Table Example
```
📊 BENCHMARK COMPARISON
====================================================================================================
Model                     Load(s)  Infer(s)  Total(s)  RTF    Samples/s  Memory(GB)
----------------------------------------------------------------------------------------------------
Kokoro-82M-4bit          2.34     0.45      2.79      0.23   53333      1.85      
Kokoro-82M-6bit          2.41     0.52      2.93      0.26   46154      2.12      
Kokoro-82M-8bit          2.38     0.58      2.96      0.29   41379      2.45      
Kokoro-82M-bf16          2.42     0.71      3.13      0.35   33803      3.21      
----------------------------------------------------------------------------------------------------
🏆 Fastest inference: Kokoro 82M (4-bit Quantized) (0.45s)
⚡ Most efficient (lowest RTF): Kokoro 82M (4-bit Quantized) (0.23x)
```

## 💾 Output Files

Results are saved to `benchmark_results/`:
- **Audio files**: `01_Kokoro-82M-bf16_af_heart.wav`, etc.
- **Metrics JSON**: `benchmark_results_YYYYMMDD_HHMMSS.json`

### JSON Structure
```json
[
  {
    "model_name": "mlx-community/Kokoro-82M-bf16",
    "model_description": "Kokoro 82M (bfloat16 - Full Precision)",
    "text": "Hello world...",
    "voice": "af_heart",
    "model_load_time": 2.42,
    "inference_time": 0.71,
    "total_time": 3.13,
    "audio_duration": 2.05,
    "real_time_factor": 0.35,
    "samples_per_sec": 33803,
    "peak_memory_gb": 3.21,
    "sample_rate": 24000
  }
]
```

## 🎯 Use Cases

### Model Selection
Compare quantized vs full precision models to find the best balance of quality and performance for your use case.

### Performance Analysis
Identify bottlenecks in your TTS pipeline and optimize for specific hardware constraints.

### Quality Assessment
Listen to side-by-side audio comparisons to evaluate quality trade-offs with quantization.

### Research & Development
Generate reproducible benchmarks for model optimization and comparison studies.

## 🔍 Troubleshooting

### Import Errors
- Ensure `mlx-audio` is installed: `pip install -e .`
- Check MLX installation: `python -c "import mlx.core; print('MLX OK')"`

### Memory Issues
- Use smaller models (4-bit, 6-bit) for devices with limited memory
- Clear MLX cache between runs if needed

### Audio Issues
- Verify `sounddevice` is installed for playback
- Check system audio settings if no sound

## 📝 Notes

- Models are downloaded from HuggingFace Hub on first use
- Model caching prevents re-downloading across runs
- All timing measurements use `time.perf_counter()` for accuracy
- Memory measurements require MLX Metal backend