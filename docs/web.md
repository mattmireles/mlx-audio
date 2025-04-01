MLX-Audio includes a web interface with a 3D visualization that reacts to audio frequencies. The interface allows you to:

1. Generate TTS with different voices and speed settings
2. Upload and play your own audio files
3. Visualize audio with an interactive 3D orb
4. Automatically saves generated audio files to the outputs directory in the current working folder
5. Open the output folder directly from the interface (when running locally)

## Features
- Multiple Voice Options: Choose from different voice styles (e.g. AF Heart, AF Nova, AF Bella, BF Emma)
- Adjustable Speech Speed: Control the speed of speech generation with an interactive slider (0.5x to 2.0x)
- Real-time 3D Visualization: A responsive 3D orb that reacts to audio frequencies
- Audio Upload: Play and visualize your own audio files
- Auto-play Option: Automatically play generated audio
- Output Folder Access: Convenient button to open the output folder in your system's file explorer

## Getting Started with the Web and API Server

Use code:
```py
# Using the command-line interface
mlx_audio.server

# With custom host and port
mlx_audio.server --host 0.0.0.0 --port 9000

# With verbose logging
mlx_audio.server --verbose
```

Available command line arguments:

`--host`: Host address to bind the server to (default: 127.0.0.1)

`--port`: Port to bind the server to (default: 8000)


Then open your browser and navigate to:

```py
http://127.0.0.1:8000

```

## API Endpoints
The server provides the following REST API endpoints:

- POST /tts: Generate TTS audio
    -   Parameters (form data):
        -   `text`: The text to convert to speech (required)
        -   `voice`: Voice to use (default: "af_heart")
        -   `speed`: Speech speed from 0.5 to 2.0 (default: 1.0)
    -   Returns: JSON with filename of generated audio
- GET /audio/{filename}: Retrieve generated audio file

- POST /play: Play audio directly from the server
    -   Parameters (form data):
        -   `filename`: The filename of the audio to play (required)
    -   Returns: JSON with status and filename

- POST /stop: Stop any currently playing audio
    -   Returns: JSON with status

- POST /open_output_folder: Open the output folder in the system's file explorer
    -   Returns: JSON with status and path
    -   Note: This feature only works when running the server locally


!!! note "Note"
    Generated audio files are stored in `~/.mlx_audio/outputs` by default, or in a fallback directory if that location is not writable.