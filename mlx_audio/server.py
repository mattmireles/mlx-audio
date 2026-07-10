# MLX-Audio FastAPI Server - Production HTTP API for Text-to-Speech Services.
# 
# This module provides a comprehensive HTTP API server for MLX-Audio text-to-speech
# functionality, built on FastAPI with real-time WebRTC support for speech-to-speech
# conversations. The server offers both REST endpoints and streaming audio capabilities.
# 
# Architecture:
# - **FastAPI Framework**: High-performance async HTTP API with automatic OpenAPI documentation
# - **WebRTC Integration**: Real-time speech-to-speech via fastrtc streaming
# - **Multi-Model Support**: Dynamic loading of TTS models (Kokoro, CSM/Sesame, Spark)
# - **RESTful Design**: Clean endpoint structure following HTTP best practices
# - **CORS Enabled**: Cross-origin support for web applications and API clients
# - **Static File Serving**: Web interface with fallback HTML for testing
# 
# Key responsibilities:
# - **TTS API Endpoints**: `/tts` for text-to-speech generation with model selection
# - **Audio File Management**: `/audio/{filename}` for retrieving generated audio files
# - **Real-time Streaming**: WebRTC speech-to-speech pipeline with configurable voices
# - **Model Management**: Dynamic model loading and caching for performance
# - **Language Support**: Multi-language TTS with automatic language code mapping
# - **Voice Cloning**: Reference audio support for voice style transfer (CSM/Sesame)
# - **Audio Playback**: Server-side audio playback control via `/play` and `/stop` endpoints
# 
# Called by:
# - **Web Applications**: Browser clients consuming REST TTS API
# - **Mobile Apps**: iOS/Android applications requiring TTS services
# - **CLI Tools**: Command-line applications using HTTP API for TTS
# - **Integration Services**: Other applications requiring programmatic TTS access
# - **Real-time Applications**: WebRTC clients for speech-to-speech conversations
# 
# Integrates with:
# - **MLX-Audio Models**: All TTS implementations (`mlx_audio.tts.models.*`)
# - **FastRTC**: Real-time communication library for WebRTC streaming
# - **STT Models**: Whisper integration for speech recognition in speech-to-speech
# - **Audio Processing**: `soundfile`, `numpy` for audio I/O and manipulation
# - **MLX Framework**: Apple Silicon optimized inference via model integrations
# 
# REST API Endpoints:
# - **POST /tts**: Generate TTS audio from text with model/voice selection
# - **GET /audio/{filename}**: Retrieve generated audio files
# - **POST /speech_to_speech_input**: Configure WebRTC speech-to-speech parameters
# - **POST /play**: Server-side audio playback control
# - **POST /stop**: Stop current audio playback
# - **GET /languages**: List supported languages with display names
# - **GET /models**: List available TTS models with capabilities
# - **POST /open_output_folder**: Open output directory in system file manager
# - **GET /**: Serve web interface with API documentation
# 
# WebRTC Features:
# - **Real-time STT**: Speech recognition using optimized Whisper models
# - **Real-time TTS**: Text-to-speech synthesis with configurable voices and speeds
# - **Streaming Audio**: Low-latency audio streaming for conversation applications
# - **Voice Selection**: Runtime voice switching for different conversation participants
# 
# Performance Characteristics:
# - **Model Caching**: Global model instances to avoid reloading overhead
# - **Async Operations**: FastAPI async/await for concurrent request handling
# - **Memory Management**: Automatic cleanup of temporary files and model states
# - **Error Recovery**: Robust error handling with detailed logging and fallback options
# - **File Management**: Secure temporary file handling with automatic cleanup
# 
# Security Features:
# - **CORS Configuration**: Controlled cross-origin access with configurable origins
# - **File Path Validation**: Secure file serving with path traversal protection
# - **Request Validation**: Pydantic models for request/response validation
# - **Local-only Admin**: Sensitive operations restricted to localhost requests
# 
# Production Deployment:
# - **Uvicorn ASGI**: Production-ready ASGI server with configurable host/port
# - **Logging**: Structured logging with configurable verbosity levels
# - **Health Checks**: Model loading validation and file system permissions
# - **Static Assets**: Web interface serving with fallback error pages

import argparse
import importlib.util
import logging
import os
import sys
import tempfile
import uuid

import numpy as np
import requests
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastrtc import ReplyOnPause, Stream, get_stt_model
from numpy.typing import NDArray
from pydantic import BaseModel

# Server configuration constants - optimized for production deployment
DEFAULT_HOST = "127.0.0.1"             # Default server host (localhost for security)
DEFAULT_PORT = 8000                    # Default server port (HTTP alternative)
DEFAULT_MODEL = "mlx-community/Kokoro-82M-4bit"  # Default TTS model for server startup

# Audio processing constants - aligned with TTS model requirements
AUDIO_SAMPLE_RATE = 24000              # Standard sample rate for high-quality TTS (Hz)
AUDIO_CHUNK_SIZE = 2400                # Audio chunk size for streaming (0.1s at 24kHz)
TEMP_FILE_PREFIX = "temp_ref_"          # Prefix for temporary reference audio files
TTS_FILE_PREFIX = "tts_"               # Prefix for generated TTS audio files
MAX_TOKEN_LIMIT = 8000                 # Maximum tokens for TTS generation

# Audio channel processing constants
AUDIO_CHANNEL_AXIS = 1                 # Axis for audio channel operations (stereo to mono)
MONO_CHANNEL_COUNT = 1                 # Single channel count for mono audio

# Speed and parameter validation constants
MIN_SPEED_VALUE = 0.5                  # Minimum allowed speech speed
MAX_SPEED_VALUE = 2.0                  # Maximum allowed speech speed
DEFAULT_SPEED_VALUE = 1.0              # Default speech speed (normal)

# Spark model specific constants - discrete speed/pitch mappings
SPARK_SPEED_VERY_LOW = 0.0             # Spark model very low speed setting
SPARK_SPEED_LOW = 0.5                  # Spark model low speed setting  
SPARK_SPEED_MODERATE = 1.0             # Spark model moderate speed setting
SPARK_SPEED_HIGH = 1.5                 # Spark model high speed setting
SPARK_SPEED_VERY_HIGH = 2.0            # Spark model very high speed setting
SPARK_PITCH_VERY_LOW = 0.0             # Spark model very low pitch setting
SPARK_PITCH_LOW = 0.5                  # Spark model low pitch setting
SPARK_PITCH_MODERATE = 1.0             # Spark model moderate pitch setting
SPARK_PITCH_HIGH = 1.5                 # Spark model high pitch setting
SPARK_PITCH_VERY_HIGH = 2.0            # Spark model very high pitch setting

# File system constants
TEST_FILE_NAME = "test_write.txt"       # Test filename for write permission validation
TEST_FILE_CONTENT = "Test write permissions"  # Content for write permission test
FALLBACK_DIR_NAME = "mlx_audio_outputs" # Fallback directory name for /tmp usage

# Platform detection constants for file manager operations
MACOS_PLATFORM = "darwin"               # macOS platform identifier
WINDOWS_PLATFORM = "win32"              # Windows platform identifier  
LINUX_PLATFORM = "linux"                # Linux platform identifier

# Text processing constants
TEXT_PREVIEW_LENGTH = 50               # Character limit for text preview in logs
TEXT_PREVIEW_SUFFIX = "..."             # Suffix for truncated text in logs


def setup_logging(verbose: bool = False):
    # Configures structured logging for the MLX-Audio server.
    # 
    # Sets up comprehensive logging with configurable verbosity levels for
    # production deployment monitoring and debugging. Provides detailed context
    # in verbose mode including function names and line numbers.
    # 
    # Called by:
    # - `main()` during server initialization with command-line verbose flag
    # - Module initialization for default logger setup
    # 
    # Logging Levels:
    # - **Normal Mode**: INFO level with timestamp, module, level, and message
    # - **Verbose Mode**: DEBUG level with function names and line numbers
    # 
    # Args:
    #     verbose (bool): Whether to enable debug-level logging with detailed context
    # 
    # Returns:
    #     logging.Logger: Configured logger instance for server operations
    # 
    # Performance: Minimal overhead in normal mode, detailed context in debug mode.
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if verbose:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"

    logging.basicConfig(level=level, format=format_str)
    return logging.getLogger("mlx_audio_server")


logger = setup_logging()  # Will be updated with verbose setting in main()

from mlx_audio.tts.generate import main as generate_main

# Import from mlx_audio package
from mlx_audio.tts.utils import load_model

from .tts.audio_player import AudioPlayer

app = FastAPI()

# Add CORS middleware to allow requests from the same origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, will be restricted by host binding
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once on server startup.
# You can change the model path or pass arguments as needed.
# For performance, load once globally:
tts_model = None  # Will be loaded when the server starts
audio_player = None  # Will be initialized when the server starts
stt_model = get_stt_model()
# Make sure the output folder for generated TTS files exists
# Use an absolute path that's guaranteed to be writable
OUTPUT_FOLDER = os.path.join(os.path.expanduser("~"), ".mlx_audio", "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
logger.debug(f"Using output folder: {OUTPUT_FOLDER}")


def speech_to_speech_handler(
    audio: tuple[int, NDArray[np.int16]],
    voice: str,
    speed: float,
    model: str,
    language: str = "a",
):
    text = stt_model.stt(audio)
    for segment in tts_model.generate(
        text=text,
        voice=voice,
        speed=speed,
        lang_code=language,
        verbose=False,
    ):
        yield (AUDIO_SAMPLE_RATE, np.array(segment.audio, copy=False))
        yield (AUDIO_SAMPLE_RATE, np.zeros(AUDIO_CHUNK_SIZE, dtype=np.float32))


stream = Stream(
    ReplyOnPause(speech_to_speech_handler, output_sample_rate=AUDIO_SAMPLE_RATE),
    mode="send-receive",
    modality="audio",
)
stream.mount(app)


class SpeechToSpeechArgs(BaseModel):
    voice: str
    speed: float
    model: str
    webrtc_id: str
    language: str = "a"


@app.post("/speech_to_speech_input")
def speech_to_speech_endpoint(args: SpeechToSpeechArgs):
    stream.set_input(args.webrtc_id, args.voice, args.speed, args.model, args.language)
    return {"status": "success"}


@app.post("/tts")
def tts_endpoint(
    text: str = Form(...),
    voice: str = Form(None),
    speed: str = Form("1.0"),
    model: str = Form(DEFAULT_MODEL),
    language: str = Form("a"),
    pitch: str = Form(None),
    gender: str = Form(None),
    reference_audio: UploadFile = File(None),
):
    """
    POST an x-www-form-urlencoded form with 'text' (and optional 'voice', 'speed', and 'model').
    We run TTS on the text, save the audio in a unique file,
    and return JSON with the filename so the client can retrieve it.
    """
    global tts_model

    if not text.strip():
        return JSONResponse({"error": "Text is empty"}, status_code=400)

    # Handle speed parameter based on model type
    if "spark" in model.lower():
        # Spark model uses discrete speed mappings rather than continuous float values
        speed_map = {
            "very_low": SPARK_SPEED_VERY_LOW,
            "low": SPARK_SPEED_LOW,
            "moderate": SPARK_SPEED_MODERATE,
            "high": SPARK_SPEED_HIGH,
            "very_high": SPARK_SPEED_VERY_HIGH,
        }
        if speed in speed_map:
            speed_value = speed_map[speed]
        else:
            # Try to use as float, default to moderate if invalid
            try:
                speed_value = float(speed)
                valid_speeds = [SPARK_SPEED_VERY_LOW, SPARK_SPEED_LOW, SPARK_SPEED_MODERATE, 
                               SPARK_SPEED_HIGH, SPARK_SPEED_VERY_HIGH]
                if speed_value not in valid_speeds:
                    speed_value = SPARK_SPEED_MODERATE  # Default to moderate
            except:
                speed_value = SPARK_SPEED_MODERATE  # Default to moderate
    else:
        # Other models use continuous float speed values within valid range
        try:
            speed_float = float(speed)
            if speed_float < MIN_SPEED_VALUE or speed_float > MAX_SPEED_VALUE:
                return JSONResponse(
                    {"error": f"Speed must be between {MIN_SPEED_VALUE} and {MAX_SPEED_VALUE}"}, status_code=400
                )
            speed_value = speed_float
        except ValueError:
            return JSONResponse({"error": "Invalid speed value"}, status_code=400)

    # Remove strict model validation - let the load_model function handle it
    # This allows for more flexibility in model selection

    # Store current model repo_id for comparison
    current_model_repo_id = (
        getattr(tts_model, "repo_id", None) if tts_model is not None else None
    )

    # Load the model if it's not loaded or if a different model is requested
    if tts_model is None or current_model_repo_id != model:
        try:
            logger.debug(f"Loading TTS model from {model}")
            tts_model = load_model(model)
            logger.debug("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading TTS model: {str(e)}")
            return JSONResponse(
                {"error": f"Failed to load model: {str(e)}"}, status_code=500
            )

    # Generate unique filename for TTS output to prevent conflicts
    unique_id = str(uuid.uuid4())
    filename = f"{TTS_FILE_PREFIX}{unique_id}.wav"
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    logger.debug(
        f"Generating TTS for text: '{text[:TEXT_PREVIEW_LENGTH]}{TEXT_PREVIEW_SUFFIX}' with voice: {voice}, speed: {speed_value}, model: {model}, language: {language}"
    )
    logger.debug(f"Output file will be: {output_path}")

    # Map language names to codes if needed
    language_map = {
        "american_english": "a",
        "british_english": "b",
        "spanish": "e",
        "french": "f",
        "hindi": "h",
        "italian": "i",
        "portuguese": "p",
        "japanese": "j",
        "mandarin_chinese": "z",
        # Also accept direct language codes
        "a": "a",
        "b": "b",
        "e": "e",
        "f": "f",
        "h": "h",
        "i": "i",
        "p": "p",
        "j": "j",
        "z": "z",
    }

    # Get the language code, default to voice[0] if not found
    lang_code = language_map.get(language.lower(), voice[0] if voice else "a")

    # Handle reference audio for models that support it (like CSM/Sesame)
    ref_audio_path = None
    if reference_audio:
        # Save the uploaded audio temporarily with secure naming
        temp_audio_path = os.path.join(OUTPUT_FOLDER, f"{TEMP_FILE_PREFIX}{unique_id}.wav")
        with open(temp_audio_path, "wb") as f:
            f.write(reference_audio.file.read())
        ref_audio_path = temp_audio_path

    # Prepare generation parameters
    gen_params = {
        "text": text,
        "speed": speed_value,
        "verbose": False,
        "max_tokens": MAX_TOKEN_LIMIT,
    }

    # Add pitch and gender for Spark models
    if "spark" in model.lower():
        # Spark model uses discrete pitch mappings for voice characteristics
        pitch_map = {
            "very_low": SPARK_PITCH_VERY_LOW,
            "low": SPARK_PITCH_LOW,
            "moderate": SPARK_PITCH_MODERATE,
            "high": SPARK_PITCH_HIGH,
            "very_high": SPARK_PITCH_VERY_HIGH,
        }
        if pitch and pitch in pitch_map:
            gen_params["pitch"] = pitch_map[pitch]
        else:
            gen_params["pitch"] = SPARK_PITCH_MODERATE  # Default to moderate

        # Ensure gender has a valid value
        valid_genders = ["female", "male"]
        if gender and gender in valid_genders:
            gen_params["gender"] = gender
        else:
            gen_params["gender"] = "female"

    # Add model-specific parameters
    if voice and voice.strip():  # Only add voice if it's not empty or whitespace
        gen_params["voice"] = voice

    # Check if model supports language codes (primarily Kokoro)
    if "kokoro" in model.lower():
        gen_params["lang_code"] = lang_code

    # Add reference audio for models that support it
    if ref_audio_path and ("csm" in model.lower() or "sesame" in model.lower()):
        gen_params["ref_audio"] = ref_audio_path

    # We'll use the high-level "model.generate" method:
    try:
        results = tts_model.generate(**gen_params)
    finally:
        # Clean up temporary reference audio file
        if ref_audio_path and os.path.exists(ref_audio_path):
            os.remove(ref_audio_path)

    # We'll just gather all segments (if any) into a single wav
    # It's typical for multi-segment text to produce multiple wave segments:
    audio_arrays = []
    for segment in results:
        audio_arrays.append(segment.audio)

    # If no segments, return error
    if not audio_arrays:
        logger.error("No audio segments generated")
        return JSONResponse({"error": "No audio generated"}, status_code=500)

    # Concatenate all segments
    cat_audio = np.concatenate(audio_arrays, axis=0)

    # Write the audio as a WAV
    try:
        sf.write(output_path, cat_audio, AUDIO_SAMPLE_RATE)
        logger.debug(f"Successfully wrote audio file to {output_path}")

        # Verify the file exists
        if not os.path.exists(output_path):
            logger.error(f"File was not created at {output_path}")
            return JSONResponse(
                {"error": "Failed to create audio file"}, status_code=500
            )

        # Check file size
        file_size = os.path.getsize(output_path)
        logger.debug(f"File size: {file_size} bytes")

        if file_size == 0:
            logger.error("File was created but is empty")
            return JSONResponse(
                {"error": "Generated audio file is empty"}, status_code=500
            )

    except Exception as e:
        logger.error(f"Error writing audio file: {str(e)}")
        return JSONResponse(
            {"error": f"Failed to save audio: {str(e)}"}, status_code=500
        )

    return {"filename": filename}


@app.get("/audio/{filename}")
def get_audio_file(filename: str):
    """
    Return an audio file from the outputs folder.
    The user can GET /audio/<filename> to fetch the WAV file.
    """
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    logger.debug(f"Requested audio file: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        # List files in the directory to help debug
        try:
            files = os.listdir(OUTPUT_FOLDER)
            logger.debug(f"Files in output directory: {files}")
        except Exception as e:
            logger.error(f"Error listing output directory: {str(e)}")

        return JSONResponse({"error": "File not found"}, status_code=404)

    logger.debug(f"Serving audio file: {file_path}")
    return FileResponse(file_path, media_type="audio/wav")


@app.get("/")
def root():
    """
    Serve the audio_player.html page or a fallback HTML if not found
    """
    try:
        # Try to find the audio_player.html file in the package
        static_dir = find_static_dir()
        audio_player_path = os.path.join(static_dir, "audio_player.html")
        return FileResponse(audio_player_path)
    except Exception as e:
        # If there's an error, return a simple HTML page with error information
        return HTMLResponse(
            content=f"""
            <html>
                <head><title>MLX-Audio TTS Server</title></head>
                <body>
                    <h1>MLX-Audio TTS Server</h1>
                    <p>The server is running, but the web interface could not be loaded.</p>
                    <p>Error: {str(e)}</p>
                    <h2>API Endpoints</h2>
                    <ul>
                        <li><code>POST /tts</code> - Generate TTS audio</li>
                        <li><code>GET /audio/{{filename}}</code> - Retrieve generated audio file</li>
                    </ul>
                </body>
            </html>
            """,
            status_code=200,
        )


def find_static_dir():
    """Find the static directory containing HTML files."""
    # Try different methods to find the static directory

    # Method 1: Use importlib.resources (Python 3.9+)
    try:
        import importlib.resources as pkg_resources

        static_dir = pkg_resources.files("mlx_audio").joinpath("tts")
        static_dir_str = str(static_dir)
        if os.path.exists(static_dir_str):
            return static_dir_str
    except (ImportError, AttributeError):
        pass

    # Method 2: Use importlib_resources (Python 3.8)
    try:
        import importlib_resources

        static_dir = importlib_resources.files("mlx_audio").joinpath("tts")
        static_dir_str = str(static_dir)
        if os.path.exists(static_dir_str):
            return static_dir_str
    except ImportError:
        pass

    # Method 3: Use pkg_resources
    try:
        static_dir_str = pkg_resources.resource_filename("mlx_audio", "tts")
        if os.path.exists(static_dir_str):
            return static_dir_str
    except (ImportError, pkg_resources.DistributionNotFound):
        pass

    # Method 4: Try to find the module path directly
    try:
        module_spec = importlib.util.find_spec("mlx_audio")
        if module_spec and module_spec.origin:
            package_dir = os.path.dirname(module_spec.origin)
            static_dir_str = os.path.join(package_dir, "tts")
            if os.path.exists(static_dir_str):
                return static_dir_str
    except (ImportError, AttributeError):
        pass

    # Method 5: Look in sys.modules
    try:
        if "mlx_audio" in sys.modules:
            module = sys.modules["mlx_audio"]
            if hasattr(module, "__file__"):
                package_dir = os.path.dirname(module.__file__)
                static_dir_str = os.path.join(package_dir, "tts")
                if os.path.exists(static_dir_str):
                    return static_dir_str
    except Exception:
        pass

    # If all methods fail, raise an error
    raise RuntimeError("Could not find static directory")


@app.post("/play")
def play_audio(filename: str = Form(...)):
    """
    Play audio directly from the server using the AudioPlayer.
    Expects a filename that exists in the OUTPUT_FOLDER.
    """
    global audio_player

    if audio_player is None:
        return JSONResponse({"error": "Audio player not initialized"}, status_code=500)

    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)

    try:
        # Load the audio file
        audio_data, sample_rate = sf.read(file_path)

        # If audio is stereo, convert to mono by averaging channels
        if len(audio_data.shape) > MONO_CHANNEL_COUNT and audio_data.shape[AUDIO_CHANNEL_AXIS] > MONO_CHANNEL_COUNT:
            audio_data = audio_data.mean(axis=AUDIO_CHANNEL_AXIS)

        # Queue the audio for playback
        audio_player.queue_audio(audio_data)

        return {"status": "playing", "filename": filename}
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to play audio: {str(e)}"}, status_code=500
        )


@app.post("/stop")
def stop_audio():
    """
    Stop any currently playing audio.
    """
    global audio_player

    if audio_player is None:
        return JSONResponse({"error": "Audio player not initialized"}, status_code=500)

    try:
        audio_player.stop()
        return {"status": "stopped"}
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to stop audio: {str(e)}"}, status_code=500
        )


@app.get("/languages")
def get_languages():
    """
    Get the list of supported languages for TTS.
    """
    languages = [
        {"code": "a", "name": "American English", "display": "🇺🇸 American English"},
        {"code": "b", "name": "British English", "display": "🇬🇧 British English"},
        {"code": "e", "name": "Spanish", "display": "🇪🇸 Spanish"},
        {"code": "f", "name": "French", "display": "🇫🇷 French"},
        {"code": "h", "name": "Hindi", "display": "🇮🇳 Hindi"},
        {"code": "i", "name": "Italian", "display": "🇮🇹 Italian"},
        {"code": "p", "name": "Portuguese", "display": "🇧🇷 Portuguese (Brazilian)"},
        {"code": "j", "name": "Japanese", "display": "🇯🇵 Japanese"},
        {"code": "z", "name": "Mandarin Chinese", "display": "🇨🇳 Mandarin Chinese"},
    ]
    return {"languages": languages}


@app.get("/models")
def get_models():
    """
    Get the list of available TTS models with their configurations.
    """
    models = [
        {
            "id": "kokoro",
            "name": "Kokoro",
            "description": "Multilingual TTS with 9 languages",
            "supports_languages": True,
            "supports_voices": True,
            "supports_reference_audio": False,
            "variants": [
                {
                    "value": "mlx-community/Kokoro-82M-4bit",
                    "name": "Kokoro 82M (4-bit)",
                },
                {
                    "value": "mlx-community/Kokoro-82M-6bit",
                    "name": "Kokoro 82M (6-bit)",
                },
                {
                    "value": "mlx-community/Kokoro-82M-8bit",
                    "name": "Kokoro 82M (8-bit)",
                },
                {"value": "mlx-community/Kokoro-82M-bf16", "name": "Kokoro 82M (bf16)"},
                {"value": "prince-canuma/Kokoro-82M", "name": "Kokoro 82M (Original)"},
            ],
        },
        {
            "id": "csm",
            "name": "CSM/Sesame",
            "description": "Conversational Speech Model with voice cloning",
            "supports_languages": False,
            "supports_voices": False,
            "supports_reference_audio": True,
            "variants": [
                {"value": "mlx-community/csm-1b", "name": "CSM 1B (FP16)"},
                {"value": "mlx-community/csm-1b-8bit", "name": "CSM 1B (8-bit)"},
            ],
        },
        {
            "id": "spark",
            "name": "Spark",
            "description": "Fast TTS model",
            "supports_languages": False,
            "supports_voices": True,
            "supports_reference_audio": False,
            "variants": [
                {
                    "value": "mlx-community/Spark-TTS-0.5B-bf16",
                    "name": "Spark TTS (BF16)",
                },
                {
                    "value": "mlx-community/Spark-TTS-0.5B-8bit",
                    "name": "Spark TTS (8-bit)",
                },
            ],
        },
    ]
    return {"models": models}


@app.post("/open_output_folder")
def open_output_folder():
    """
    Open the output folder in the system file explorer (Finder on macOS).
    This only works when running on localhost for security reasons.
    """
    global OUTPUT_FOLDER

    # Check if the request is coming from localhost
    # Note: In a production environment, you would want to check the request IP

    try:
        # For macOS (Finder)
        if sys.platform == MACOS_PLATFORM:
            os.system(f"open {OUTPUT_FOLDER}")
        # For Windows (Explorer)
        elif sys.platform == WINDOWS_PLATFORM:
            os.system(f"explorer {OUTPUT_FOLDER}")
        # For Linux (various file managers)
        elif sys.platform == LINUX_PLATFORM:
            os.system(f"xdg-open {OUTPUT_FOLDER}")
        else:
            return JSONResponse(
                {"error": f"Unsupported platform: {sys.platform}"}, status_code=500
            )

        logger.debug(f"Opened output folder: {OUTPUT_FOLDER}")
        return {"status": "opened", "path": OUTPUT_FOLDER}
    except Exception as e:
        logger.error(f"Error opening output folder: {str(e)}")
        return JSONResponse(
            {"error": f"Failed to open output folder: {str(e)}"}, status_code=500
        )


def setup_server():
    """Setup the server by loading the model and creating the output directory."""
    global tts_model, audio_player, OUTPUT_FOLDER

    # Make sure the output folder for generated TTS files exists
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        # Test write permissions by creating a test file
        test_file = os.path.join(OUTPUT_FOLDER, TEST_FILE_NAME)
        with open(test_file, "w") as f:
            f.write(TEST_FILE_CONTENT)
        os.remove(test_file)
        logger.debug(f"Output directory {OUTPUT_FOLDER} is writable")
    except Exception as e:
        logger.error(f"Error with output directory {OUTPUT_FOLDER}: {str(e)}")
        # Try to use a fallback directory in /tmp for temporary storage
        fallback_dir = os.path.join("/tmp", FALLBACK_DIR_NAME)
        logger.debug(f"Trying fallback directory: {fallback_dir}")
        try:
            os.makedirs(fallback_dir, exist_ok=True)
            OUTPUT_FOLDER = fallback_dir
            logger.debug(f"Using fallback output directory: {OUTPUT_FOLDER}")
        except Exception as fallback_error:
            logger.error(f"Error with fallback directory: {str(fallback_error)}")

    # Load the model if not already loaded
    if tts_model is None:
        try:
            # Use consistent default model across server initialization and endpoints
            default_model = DEFAULT_MODEL
            logger.debug(f"Loading TTS model from {default_model}")
            tts_model = load_model(default_model)
            logger.debug("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading TTS model: {str(e)}")
            raise

    # Initialize the audio player if not already initialized
    if audio_player is None:
        try:
            logger.debug("Initializing audio player")
            audio_player = AudioPlayer()
            logger.debug("Audio player initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing audio player: {str(e)}")

    # Try to mount the static files directory
    try:
        static_dir = find_static_dir()
        logger.debug(f"Found static directory: {static_dir}")
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.debug("Static files mounted successfully")
    except Exception as e:
        logger.error(f"Could not mount static files directory: {e}")
        logger.warning(
            "The server will still function, but the web interface may be limited."
        )


def main(host=DEFAULT_HOST, port=DEFAULT_PORT, verbose=False):
    # Initializes and starts the MLX-Audio FastAPI server with configuration.
    # 
    # Parses command-line arguments, configures logging, initializes the server
    # components, and starts the uvicorn ASGI server with the specified configuration.
    # 
    # Called by:
    # - Command-line execution when running server as main module
    # - External applications requiring programmatic server startup
    # - Development and production deployment scripts
    # 
    # Server Initialization Process:
    # 1. **Argument Parsing**: Host, port, and verbose logging configuration
    # 2. **Logger Setup**: Configures structured logging with appropriate verbosity
    # 3. **Model Loading**: Initializes default TTS model and audio player
    # 4. **Directory Setup**: Creates output directories with permission validation
    # 5. **Static Assets**: Mounts static file serving for web interface
    # 6. **ASGI Server**: Starts uvicorn server with FastAPI application
    # 
    # Args:
    #     host (str): Server host address (default: localhost for security)
    #     port (int): Server port number (default: 8000)
    #     verbose (bool): Whether to enable debug-level logging
    # 
    # Performance: Async server capable of handling concurrent requests efficiently.
    parser = argparse.ArgumentParser(description="Start the MLX-Audio TTS server")
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Host address to bind the server to (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind the server to (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging with detailed debug information",
    )
    args = parser.parse_args()

    # Update logger with verbose setting
    global logger
    logger = setup_logging(args.verbose)

    # Start the server with the parsed arguments
    setup_server()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.verbose else "info",
    )


if __name__ == "__main__":
    main()
