#Getting Started with MLX-Audio

## Installation

```py linenums="1"
# Install the package
pip install mlx-audio

# For web interface and API dependencies
pip install -r requirements.txt
```

## Quick Start

Generate an audio from text using default settings:

```py

mlx_audio.tts.generate --text "Hello, world"

```

Customize the output file name for the generated audio using `--file_prefix`

```py

mlx_audio.tts.generate --text "Hello, world" --file_prefix hello

```

Control the speech rate using `--speed` which accepts values between 0.5-2.0

```py 

mlx_audio.tts.generate --text "Hello, world" --speed 1.4

```

## How to call from Python

To generate an audio using an LLM use the following code:

```py
from mlx_audio.tts.generate import generate_audio

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