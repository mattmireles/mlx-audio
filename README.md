# MLX-Audio

A text-to-speech (TTS) and Speech-to-Speech (STS) library built on Apple's MLX framework, providing efficient speech synthesis on Apple Silicon.

## Features

- **Fast inference** on Apple Silicon (M series chips)
- **Multiple TTS models**: Kokoro, OuteTTS, Spark, CSM, and more
- **Native iOS/macOS support** with Swift framework implementation  
- **Multiple language support** including English, Japanese, Chinese
- **Advanced quantization**: 4-bit, 6-bit, 8-bit, and full precision options
- **Voice customization** and cloning capabilities
- **Adjustable speech speed** control (0.5x to 2.0x)
- **Interactive web interface** with 3D audio visualization
- **REST API** for TTS generation
- **Benchmarking tools** for performance analysis and model comparison
- **Streaming-first pipeline**: sentence-by-sentence generation with fast TTFA
- **Speech-to-text** support with Voxtral and other STT models
- **Direct file access** via Finder/Explorer integration

## Installation

### Python Package Installation

```bash
# Install the package
pip install mlx-audio

# For web interface, API, and benchmarking dependencies
pip install -r requirements.txt

# For development and testing
pip install -e .
```

### iOS/macOS Swift Framework

For native iOS/macOS development:

1. Clone the repository
2. Open `mlx_audio_swift/tts/Swift-TTS.xcodeproj` in Xcode
3. Follow the setup instructions in the [iOS/macOS Native Support](#iosmacOS-native-support) section

### Verification

Verify your installation works:

```bash
# Test basic TTS functionality
mlx_audio.tts.generate --text "Installation successful!" --file_prefix test

# Test web server
mlx_audio.server --help

# Run benchmarking tool (optional)
python kokoro_benchmark.py
```

### Quick Start

To generate audio with an LLM use:

```bash
# Basic usage
mlx_audio.tts.generate --text "Hello, world"

# Specify prefix for output file
mlx_audio.tts.generate --text "Hello, world" --file_prefix hello

# Adjust speaking speed (0.5-2.0)
mlx_audio.tts.generate --text "Hello, world" --speed 1.4
```

### How to call from python

To generate audio with an LLM use:

```python
from mlx_audio.tts.generate import generate_audio

# Example: Generate an audiobook chapter as mp3 audio
generate_audio(
    text=("In the beginning, the universe was created...\n"
        "...or the simulation was booted up."),
    model_path="prince-canuma/Kokoro-82M",
    voice="af_heart",
    speed=1.2,
    lang_code="a", # Kokoro: (a)f_heart, or comment out for auto
    file_prefix="audiobook_chapter1",
    audio_format="wav",
    sample_rate=24000,
    join_audio=True,
    verbose=True  # Set to False to disable print messages
)

print("Audiobook chapter successfully generated!")

```

### Web Interface & API Server

MLX-Audio includes a web interface with a 3D visualization that reacts to audio frequencies. The interface allows you to:

1. Generate TTS with different voices and speed settings
2. Upload and play your own audio files
3. Visualize audio with an interactive 3D orb
4. Automatically saves generated audio files to the outputs directory in the current working folder
5. Open the output folder directly from the interface (when running locally)

#### Features

- **Multiple Voice Options**: Choose from different voice styles (AF Heart, AF Nova, AF Bella, BF Emma)
- **Adjustable Speech Speed**: Control the speed of speech generation with an interactive slider (0.5x to 2.0x)
- **Real-time 3D Visualization**: A responsive 3D orb that reacts to audio frequencies
- **Audio Upload**: Play and visualize your own audio files
- **Auto-play Option**: Automatically play generated audio
- **Output Folder Access**: Convenient button to open the output folder in your system's file explorer

To start the web interface and API server:

```bash
# Using the command-line interface
mlx_audio.server

# With custom host and port
mlx_audio.server --host 0.0.0.0 --port 9000

# With verbose logging
mlx_audio.server --verbose
```

Available command line arguments:
- `--host`: Host address to bind the server to (default: 127.0.0.1)
- `--port`: Port to bind the server to (default: 8000)

Then open your browser and navigate to:
```
http://127.0.0.1:8000
```

#### API Endpoints

The server provides the following REST API endpoints:

- `POST /tts`: Generate TTS audio
  - Parameters (form data):
    - `text`: The text to convert to speech (required)
    - `voice`: Voice to use (default: "af_heart")
    - `speed`: Speech speed from 0.5 to 2.0 (default: 1.0)
  - Returns: JSON with filename of generated audio

- `GET /audio/{filename}`: Retrieve generated audio file

- `POST /play`: Play audio directly from the server
  - Parameters (form data):
    - `filename`: The filename of the audio to play (required)
  - Returns: JSON with status and filename

- `POST /stop`: Stop any currently playing audio
  - Returns: JSON with status

- `POST /open_output_folder`: Open the output folder in the system's file explorer
  - Returns: JSON with status and path
  - Note: This feature only works when running the server locally

> Note: Generated audio files are stored in `~/.mlx_audio/outputs` by default, or in a fallback directory if that location is not writable.

## Models

### Kokoro

Kokoro is a multilingual TTS model that supports various languages and voice styles.

#### Example Usage

```python
from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.utils import load_model
from IPython.display import Audio
import soundfile as sf

# Initialize the model
model_id = 'prince-canuma/Kokoro-82M'
model = load_model(model_id)

# Create a pipeline with American English
pipeline = KokoroPipeline(lang_code='a', model=model, repo_id=model_id)

# Generate audio (streams sentence-by-sentence)
text = "The MLX King lives. Let him cook!"
for _, _, audio in pipeline(text, voice='af_heart', speed=1):
    # Display audio in notebook (if applicable)
    display(Audio(data=audio, rate=24000, autoplay=0))

    # Save audio to file
    sf.write('audio.wav', audio[0], 24000)
```

#### Language Options

- 🇺🇸 `'a'` - American English
- 🇬🇧 `'b'` - British English
- 🇯🇵 `'j'` - Japanese (requires `pip install misaki[ja]`)
- 🇨🇳 `'z'` - Mandarin Chinese (requires `pip install misaki[zh]`)

### CSM (Conversational Speech Model)

CSM is a model from Sesame that allows you text-to-speech and to customize voices using reference audio samples.

#### Example Usage

```bash
# Generate speech using CSM-1B model with reference audio
python -m mlx_audio.tts.generate --model mlx-community/csm-1b --text "Hello from Sesame." --play --ref_audio ./conversational_a.wav
```

You can pass any audio to clone the voice from or download sample audio file from [here](https://huggingface.co/mlx-community/csm-1b/tree/main/prompts).

### OuteTTS

OuteTTS is a state-of-the-art text-to-speech model that provides high-quality voice synthesis with advanced customization options.

#### Example Usage

```python
from mlx_audio.tts.models.outetts import OuteTTSPipeline
from mlx_audio.tts.utils import load_model

# Initialize the model
model_id = 'OuteTTS/OuteTTS-0.1-350M'
model = load_model(model_id)

# Create pipeline
pipeline = OuteTTSPipeline(model=model, repo_id=model_id)

# Generate audio with custom voice
text = "Hello from OuteTTS!"
for _, _, audio in pipeline(text, voice='default', speed=1.0):
    # Process audio output
    pass
```

### Spark

Spark is an advanced audio codec and TTS model featuring sophisticated audio tokenization and high-fidelity speech synthesis.

#### Example Usage

```python
from mlx_audio.tts.models.spark import SparkPipeline
from mlx_audio.tts.utils import load_model

# Initialize the model
model_id = 'spark-community/spark-tts'
model = load_model(model_id)

# Generate speech
pipeline = SparkPipeline(model=model)
audio_output = pipeline.generate(text="Spark TTS synthesis", voice_id="default")
```

### Additional TTS Models

MLX-Audio now supports several additional TTS models for various use cases:

- **Dia**: Multilingual TTS model with strong voice quality
- **IndexTTS**: Advanced model with conformer architecture for improved pronunciation
- **Llama**: Experimental LLM-based approach to text-to-speech

### Voxtral (Speech-to-Text)

Voxtral is a high-performance speech-to-text model optimized for Apple Silicon.

#### Example Usage

```python
from mlx_audio.stt.models.voxtral import Model
import mlx.core as mx

# Load model
model = Model.from_pretrained("mlx-community/voxtral")

# Process audio file
audio_data = mx.load("audio.wav")
transcription = model.transcribe(audio_data)
print(transcription)
```

## Advanced Features

### Benchmarking

MLX-Audio includes a comprehensive benchmarking tool for testing and comparing different Kokoro TTS models, including quantized versions.

#### Kokoro TTS Benchmark Tool

The benchmarking tool provides detailed performance analysis including timing, memory usage, and audio quality comparison.

**Available Models for Benchmarking:**
1. **Kokoro 82M (bfloat16)** - `mlx-community/Kokoro-82M-bf16` (Full precision)
2. **Kokoro 82M (8-bit)** - `mlx-community/Kokoro-82M-8bit` (8-bit quantized)
3. **Kokoro 82M (6-bit)** - `mlx-community/Kokoro-82M-6bit` (6-bit quantized)
4. **Kokoro 82M (4-bit)** - `mlx-community/Kokoro-82M-4bit` (4-bit quantized)

#### Usage

```bash
# Run the interactive benchmark tool (streams audio during inference)
python kokoro_benchmark.py

# The tool will guide you through:
# 1. Model selection (individual or all models for comparison)
# 2. Voice selection from available options
# 3. Text input for synthesis
# 4. Performance analysis and audio playback
```

#### Features

- **Performance Metrics**: Model load time, inference time, time-to-first-audio (TTFA), real-time factor (RTF)
- **Memory Usage**: Peak memory consumption tracking
- **Audio Quality**: Direct playback for quality comparison
- **Results Export**: Save audio files and metrics to JSON
- **Batch Comparison**: Test all models with identical input for direct comparison

#### Example Output

```
📊 BENCHMARK COMPARISON
====================================================================================================
Model                     Load(s)  Infer(s)  TTFA(s)  Total(s)  RTF    Samples/s  Memory(GB)
----------------------------------------------------------------------------------------------------
Kokoro-82M-4bit          2.34     0.45      0.18     2.79      0.23   53333      1.85      
Kokoro-82M-6bit          2.41     0.52      0.21     2.93      0.26   46154      2.12      
Kokoro-82M-8bit          2.38     0.58      0.24     2.96      0.29   41379      2.45      
Kokoro-82M-bf16          2.42     0.71      0.27     3.13      0.35   33803      3.21      
----------------------------------------------------------------------------------------------------
🏆 Fastest inference: Kokoro 82M (4-bit Quantized) (0.45s)
⚡ Most efficient (lowest RTF): Kokoro 82M (4-bit Quantized) (0.23x)
```

Notes:
- The benchmark now preloads the selected voice and performs a tiny warm‑up before timing to reduce cold‑start bias.
- Playback buffers are tuned for fast audible start; you’ll hear sentence‑by‑sentence streaming during runs.

**Results are saved to `benchmark_results/` with audio files and detailed JSON metrics for further analysis.**

### Quantization

MLX-Audio supports multiple quantization levels for optimized performance and reduced memory usage. Choose the right balance of quality and performance for your use case.

#### Available Quantization Levels

**Pre-quantized Models (Recommended)**
- **4-bit**: `mlx-community/Kokoro-82M-4bit` - Fastest inference, lowest memory usage
- **6-bit**: `mlx-community/Kokoro-82M-6bit` - Balanced performance and quality
- **8-bit**: `mlx-community/Kokoro-82M-8bit` - Good quality with improved performance
- **bfloat16**: `mlx-community/Kokoro-82M-bf16` - Full precision, highest quality

#### Performance Comparison
| Quantization | Memory Usage | Inference Speed | Quality | Use Case |
|-------------|--------------|-----------------|---------|----------|
| 4-bit       | Lowest       | Fastest         | Good    | Resource-constrained devices |
| 6-bit       | Low          | Fast            | Better  | Balanced mobile/desktop apps |
| 8-bit       | Medium       | Good            | High    | Desktop applications |
| bfloat16    | Highest      | Baseline        | Highest | Quality-first applications |

#### Custom Quantization

You can also quantize models yourself for fine-tuned control:

```python
from mlx_audio.tts.utils import quantize_model, load_model
import json
import mlx.core as mx

# Load full precision model
model = load_model(repo_id='prince-canuma/Kokoro-82M')
config = model.config

# Quantize to different bit depths
group_size = 64

# 4-bit quantization (most aggressive)
bits = 4
weights_4bit, config_4bit = quantize_model(model, config, group_size, bits)

# 6-bit quantization (balanced)
bits = 6
weights_6bit, config_6bit = quantize_model(model, config, group_size, bits)

# 8-bit quantization (conservative)
bits = 8
weights_8bit, config_8bit = quantize_model(model, config, group_size, bits)

# Save quantized models
for bits, weights, config_q in [(4, weights_4bit, config_4bit), 
                               (6, weights_6bit, config_6bit),
                               (8, weights_8bit, config_8bit)]:
    output_dir = f'./{bits}bit'
    with open(f'{output_dir}/config.json', 'w') as f:
        json.dump(config_q, f)
    
    mx.save_safetensors(f"{output_dir}/kokoro-v1_0_bf16.safetensors", weights, 
                       metadata={"format": "mlx"})
```

#### Usage with Quantized Models

```python
from mlx_audio.tts.generate import generate_audio

# Use pre-quantized 4-bit model for fastest inference
generate_audio(
    text="Fast inference with quantized model",
    model_path="mlx-community/Kokoro-82M-4bit",  # 4-bit quantized
    voice="af_heart",
    file_prefix="quantized_output"
)

# Compare different quantization levels
for model, suffix in [
    ("mlx-community/Kokoro-82M-4bit", "4bit"),
    ("mlx-community/Kokoro-82M-6bit", "6bit"), 
    ("mlx-community/Kokoro-82M-8bit", "8bit"),
    ("mlx-community/Kokoro-82M-bf16", "bf16")
]:
    generate_audio(
        text="Comparing quantization levels",
        model_path=model,
        voice="af_heart", 
        file_prefix=f"comparison_{suffix}"
    )
```

### iOS/macOS Native Support

MLX-Audio includes native Swift implementations for running TTS models directly on iOS and macOS devices using MLX Swift.

#### Supported Models
- **Kokoro TTS**: Full implementation with eSpeak NG integration
- **Orpheus TTS**: Advanced model with voice expressions support

#### Features
- Native Apple Silicon optimization using MLX Swift
- On-device inference with no network requirements  
- Support for multiple voices and expressions
- Optimized for M1 chips and newer

#### Requirements
- **Hardware**: Apple Silicon Mac (M1 or newer)
- **OS**: macOS 14.0+ / iOS 16.0+
- **Xcode**: 15.0+ (with Metal Toolchain for Beta versions)
- **Dependencies**: MLX Swift framework (automatically resolved)

#### Setup & Installation

1. **Open the Xcode Project**
   ```bash
   open mlx_audio_swift/tts/Swift-TTS.xcodeproj
   ```

2. **Configure Model Files**
   
   **For Kokoro:**
   - Copy `kokoro-v1_0_bf16.safetensors` to `Kokoro/Resources/`
   - Voice JSON files are already included in the repository
   
   **For Orpheus:**
   - Copy required files to `Orpheus/Resources/`:
     - `orpheus-3b-0.1-ft-4bit.safetensors`
     - `config.json`
     - `model.safetensors.index.json` 
     - `snac_model.safetensors`
     - `snac_config.json`
     - `tokenizer_config.json`
     - `tokenizer.json`

3. **Configure Code Signing**
   - Open project settings in Xcode
   - Navigate to "Signing & Capabilities"
   - Set your development team and bundle identifier

4. **Build and Run**
   ```bash
   # Command line build
   xcodebuild -scheme Swift-TTS -destination platform=macOS,arch=arm64 build
   
   # Or build and run through Xcode GUI
   ```

#### Available Voices & Expressions

**Kokoro Voices:**
- **Female**: af_heart, af_bella, af_sarah, af_nicole, af_sky, af_river
- **Male**: am_michael, am_adam, am_eric, am_liam, am_onyx, am_puck

**Orpheus Voices:**
- tara, leah, jess, leo, dan, mia, zac, zoe

**Orpheus Expressions:**
- `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`

#### Performance Notes
- Kokoro: Optimized performance on Apple Silicon
- Orpheus: Approximately 0.1x real-time speed on M1 (be patient for generation)

#### Troubleshooting

**Metal Toolchain Error (Xcode Beta)**
```bash
# Install Metal Toolchain for Xcode Beta
env DEVELOPER_DIR=/Applications/Xcode-beta.app xcodebuild -downloadComponent MetalToolchain
```

**Build Failures**
- Ensure correct Xcode version and Metal Toolchain installation
- Verify all required model files are in the correct Resources folders
- Check code signing configuration

For detailed troubleshooting, see `XCODE_BUILD_TROUBLESHOOTING.md`.

## Requirements

### Core Requirements
- **MLX**: Apple's machine learning framework
- **Python**: 3.8+ (3.10+ recommended for best compatibility)
- **Hardware**: Apple Silicon Mac (M1, M2, M3 series) for optimal performance
- **macOS**: 12.0+ (macOS 14.0+ recommended for Swift framework features)

### Python Dependencies
- **Core**: `mlx`, `numpy`, `soundfile` 
- **TTS Models**: `transformers`, `tokenizers`, `huggingface_hub`
- **Audio Processing**: `librosa`, `scipy`
- **Web Interface & API**: `fastapi`, `uvicorn`, `python-multipart`
- **Audio Playback**: `sounddevice` (optional, for benchmarking tool)

### iOS/macOS Swift Framework Requirements
- **Hardware**: Apple Silicon Mac (M1 or newer)
- **OS**: macOS 14.0+ / iOS 16.0+
- **Xcode**: 15.0+ (with Metal Toolchain for Beta versions)
- **Dependencies**: MLX Swift framework (automatically resolved via Swift Package Manager)

### Optional Dependencies
- **Benchmarking**: `sounddevice` (for audio playback during benchmarking)
- **Advanced Features**: Various model-specific dependencies loaded on-demand
- **Development**: `pytest` (for running tests)
  
## License

[MIT License](LICENSE)

## Acknowledgements

- Thanks to the Apple MLX team for providing a great framework for building TTS and STS models.
- This project uses the Kokoro model architecture for text-to-speech synthesis.
- The 3D visualization uses Three.js for rendering.
