# Text-to-Speech Audio Generation Pipeline for MLX-Audio.
# 
# This module provides comprehensive TTS audio generation functionality including
# audio preprocessing, speech synthesis, and post-processing utilities optimized
# for Apple Silicon using the MLX framework.
# 
# Key responsibilities:
# - **Audio Processing**: Loading, resampling, normalization, and segment extraction
# - **TTS Generation**: High-level interface for text-to-speech synthesis 
# - **Voice Cloning**: Reference audio processing for voice style transfer
# - **Output Management**: Audio file writing, streaming, and playback orchestration
# - **CLI Interface**: Complete command-line interface for TTS generation
# 
# Architecture:
# - **Pipeline Design**: Modular audio processing with configurable parameters
# - **Model Agnostic**: Works with multiple TTS architectures (Kokoro, Bark, etc.)
# - **Performance Optimized**: MLX array operations for Apple Silicon acceleration
# - **Memory Efficient**: Streaming processing for long-form audio generation
# 
# Called by:
# - `mlx_audio.server` for HTTP API TTS generation endpoints
# - Command-line interface for direct audio synthesis
# - `mlx_audio.tts.models.*` for audio preprocessing and post-processing
# - Integration applications requiring programmatic TTS access
# 
# Integrates with:
# - **MLX Framework**: Apple's ML framework for optimized tensor operations
# - **Audio Models**: All TTS model implementations in `mlx_audio.tts.models.*`
# - **STT Models**: Whisper integration for reference audio transcription
# - **Audio Codecs**: SoundFile and scipy for audio I/O and signal processing
# 
# Performance Characteristics:
# - **Streaming Support**: Real-time audio generation with configurable intervals
# - **Batch Processing**: Efficient handling of long text sequences
# - **Memory Management**: Automatic cleanup and cache clearing for sustained operation
# - **Quality Control**: Advanced volume normalization and audio enhancement

import argparse
import os
import random
import sys
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import resample

from .audio_player import AudioPlayer
from .utils import load_model

# Audio processing constants - optimized for speech synthesis
DEFAULT_SAMPLE_RATE = 24000           # Standard sample rate for high-quality TTS (Hz)
LENGTH_ASSERTION_TOLERANCE = 1000     # Acceptable length difference for audio validation
AUDIO_CHANNEL_AXIS = 1                # Axis index for audio channel operations

# Volume normalization constants - for consistent audio levels
DEFAULT_VOLUME_COEFF = 0.2            # Target volume coefficient for normalization
VOLUME_SCALE_THRESHOLD = 0.1          # Threshold for audio scaling decisions
MIN_SCALING_FACTOR = 1e-3             # Minimum scaling factor to prevent division by zero
NOISE_THRESHOLD = 0.01                # Threshold for noise filtering in volume analysis
MIN_SIGNIFICANT_VALUES = 10           # Minimum significant values for volume calculation
VOLUME_PERCENTILE_LOW = 0.9           # Lower percentile for volume analysis (90%)
VOLUME_PERCENTILE_HIGH = 0.99         # Upper percentile for volume analysis (99%)
MIN_VOLUME_SCALE = 0.1                # Minimum volume scale factor
MAX_VOLUME_SCALE = 10                 # Maximum volume scale factor
MAX_AUDIO_AMPLITUDE = 1               # Maximum allowed audio amplitude

# Speech detection constants - for silence removal and boundary detection
DEFAULT_WINDOW_DURATION = 0.1         # Default window duration for speech detection (100ms)
DEFAULT_ENERGY_THRESHOLD = 0.01       # Default RMS energy threshold for speech detection
DEFAULT_MARGIN_FACTOR = 2             # Margin factor for speech boundary expansion
STEP_SIZE_DIVISOR = 10                # Step size divisor for sliding window analysis

# Mel scale constants - imported from established DSP literature
# These should match the constants in utils.py for consistency
HTK_MEL_SCALE_FACTOR = 2595           # HTK mel scale conversion factor
HTK_FREQ_REFERENCE = 700              # HTK frequency reference point (Hz)

# TTS model default parameters - optimized for quality and performance
DEFAULT_MAX_TOKENS = 1200             # Maximum tokens for TTS generation
DEFAULT_SPEED = 1.0                   # Baseline speech speed (no modification)
DEFAULT_PITCH = 1.0                   # Baseline pitch (no modification)
DEFAULT_TEMPERATURE = 0.7             # Default sampling temperature for model generation
DEFAULT_STREAMING_INTERVAL = 2.0      # Default streaming interval in seconds

# Generation sampling parameters - for neural model output control
DEFAULT_TOP_P = 0.9                   # Default top-p (nucleus) sampling threshold
DEFAULT_TOP_K = 50                    # Default top-k sampling limit
DEFAULT_REPETITION_PENALTY = 1.1      # Default repetition penalty to reduce loops

# File naming and formatting constants
FILE_INDEX_PADDING = 3                # Zero-padding for file index formatting (e.g., 001, 002)
DECIMAL_PRECISION = 1                  # Decimal places for performance metrics display
DECIMAL_PRECISION_EXTENDED = 2         # Extended decimal places for detailed metrics


def load_audio(
    audio_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    length: int = None,
    volume_normalize: bool = False,
    segment_duration: int = None,
) -> mx.array:
    # Loads and preprocesses audio files for TTS processing.
    # 
    # Provides comprehensive audio loading with resampling, normalization, and
    # segmentation capabilities. Handles multi-channel audio by converting to mono
    # and supports various preprocessing options for TTS model compatibility.
    # 
    # Called by:
    # - `generate_audio()` for reference audio loading in voice cloning
    # - TTS pipelines requiring audio preprocessing
    # - Voice matching and style transfer applications
    # 
    # Audio Processing Pipeline:
    # 1. **File Loading**: Reads audio using soundfile with automatic format detection
    # 2. **Channel Processing**: Converts multi-channel audio to mono by averaging
    # 3. **Resampling**: Resamples to target sample rate using high-quality interpolation
    # 4. **Segmentation**: Optional random segment extraction for training/testing
    # 5. **Normalization**: Optional volume normalization for consistent levels
    # 6. **Length Control**: Padding or truncation to exact length requirements
    # 
    # Args:
    #     audio_path (str): Path to audio file (supports WAV, FLAC, MP3, etc.)
    #     sample_rate (int): Target sample rate in Hz (default optimized for TTS)
    #     length (int): Exact output length in samples (None for original length)
    #     volume_normalize (bool): Whether to apply volume normalization
    #     segment_duration (int): Duration in seconds for random segment extraction
    # 
    # Returns:
    #     mx.array: Preprocessed audio as MLX array of shape (samples,)
    # 
    # Performance: Uses scipy resampling for high-quality results.
    samples, orig_sample_rate = sf.read(audio_path)
    shape = samples.shape

    # Collapse multi channel as mono
    if len(shape) > 1:
        samples = samples.sum(axis=AUDIO_CHANNEL_AXIS)
        # Divide summed samples by channel count to maintain amplitude range
        samples = samples / shape[AUDIO_CHANNEL_AXIS]
    if sample_rate != orig_sample_rate:
        print(f"Resampling from {orig_sample_rate} to {sample_rate}")
        duration = samples.shape[0] / orig_sample_rate
        num_samples = int(duration * sample_rate)
        samples = resample(samples, num_samples)

    if segment_duration is not None:
        seg_length = int(sample_rate * segment_duration)
        samples = random_select_audio_segment(samples, seg_length)

    # Audio volume normalize
    if volume_normalize:
        samples = audio_volume_normalize(samples)

    if length is not None:
        # Validate that requested length is reasonable (within tolerance)
        assert abs(samples.shape[0] - length) < LENGTH_ASSERTION_TOLERANCE
        if samples.shape[0] > length:
            samples = samples[:length]
        else:
            samples = np.pad(samples, (0, int(length - samples.shape[0])))

    audio = mx.array(samples, dtype=mx.float32)

    return audio


def audio_volume_normalize(audio: np.ndarray, coeff: float = DEFAULT_VOLUME_COEFF) -> np.ndarray:
    """
    Normalize the volume of an audio signal.

    Parameters:
        audio (numpy array): Input audio signal array.
        coeff (float): Target coefficient for normalization, default is 0.2.

    Returns:
        numpy array: The volume-normalized audio signal.
    """
    # Sort the absolute values of the audio signal
    temp = np.sort(np.abs(audio))

    # If the maximum value is below threshold, scale to minimum audible level
    if temp[-1] < VOLUME_SCALE_THRESHOLD:
        scaling_factor = max(
            temp[-1], MIN_SCALING_FACTOR
        )  # Prevent division by zero with a small constant
        audio = audio / scaling_factor * VOLUME_SCALE_THRESHOLD

    # Filter out noise values below threshold from temp
    temp = temp[temp > NOISE_THRESHOLD]
    L = temp.shape[0]  # Length of the filtered array

    # If there are insufficient significant values, skip statistical normalization
    if L <= MIN_SIGNIFICANT_VALUES:
        return audio

    # Compute the average of the top percentile range for robust volume estimation
    volume = np.mean(temp[int(VOLUME_PERCENTILE_LOW * L) : int(VOLUME_PERCENTILE_HIGH * L)])

    # Normalize the audio to the target coefficient level with safe scale factor bounds
    audio = audio * np.clip(coeff / volume, a_min=MIN_VOLUME_SCALE, a_max=MAX_VOLUME_SCALE)

    # Ensure the maximum absolute value in the audio does not exceed digital limit
    max_value = np.max(np.abs(audio))
    if max_value > MAX_AUDIO_AMPLITUDE:
        audio = audio / max_value

    return audio


def random_select_audio_segment(audio: np.ndarray, length: int) -> np.ndarray:
    # Extracts a random audio segment of specified length for training or testing.
    # 
    # Selects a random contiguous segment from the input audio, padding with zeros
    # if the input is shorter than the requested length. Useful for data augmentation
    # and creating fixed-length training samples.
    # 
    # Called by:
    # - `load_audio()` when segment_duration parameter is specified
    # - Data preprocessing pipelines for model training
    # - Audio augmentation and dataset preparation utilities
    # 
    # Args:
    #     audio (np.ndarray): Input audio signal of shape (samples,)
    #     length (int): Desired segment length in samples (sample_rate * duration)
    # 
    # Returns:
    #     np.ndarray: Random audio segment of exact specified length
    # 
    # Performance: Uses random.randint for efficient random sampling.
    if audio.shape[0] < length:
        audio = np.pad(audio, (0, int(length - audio.shape[0])))
    start_index = random.randint(0, audio.shape[0] - length)
    end_index = int(start_index + length)

    return audio[start_index:end_index]


def detect_speech_boundaries(
    wav: np.ndarray,
    sample_rate: int,
    window_duration: float = DEFAULT_WINDOW_DURATION,
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
    margin_factor: int = DEFAULT_MARGIN_FACTOR,
) -> Tuple[int, int]:
    """Detect the start and end points of speech in an audio signal using RMS energy.

    Args:
        wav: Input audio signal array with values in [-1, 1]
        sample_rate: Audio sample rate in Hz
        window_duration: Duration of detection window in seconds
        energy_threshold: RMS energy threshold for speech detection
        margin_factor: Factor to determine extra margin around detected boundaries

    Returns:
        tuple: (start_index, end_index) of speech segment

    Raises:
        ValueError: If the audio contains only silence
    """
    window_size = int(window_duration * sample_rate)
    margin = margin_factor * window_size
    step_size = window_size // STEP_SIZE_DIVISOR

    # Create sliding windows using stride tricks to avoid loops
    windows = sliding_window_view(wav, window_size)[::step_size]

    # Calculate RMS energy for each window
    energy = np.sqrt(np.mean(windows**2, axis=1))
    speech_mask = energy >= energy_threshold

    if not np.any(speech_mask):
        raise ValueError("No speech detected in audio (only silence)")

    start = max(0, np.argmax(speech_mask) * step_size - margin)
    end = min(
        len(wav),
        (len(speech_mask) - 1 - np.argmax(speech_mask[::-1])) * step_size + margin,
    )

    return start, end


def remove_silence_on_both_ends(
    wav: np.ndarray,
    sample_rate: int,
    window_duration: float = DEFAULT_WINDOW_DURATION,
    volume_threshold: float = DEFAULT_ENERGY_THRESHOLD,
) -> np.ndarray:
    """Remove silence from both ends of an audio signal.

    Args:
        wav: Input audio signal array
        sample_rate: Audio sample rate in Hz
        window_duration: Duration of detection window in seconds
        volume_threshold: Amplitude threshold for silence detection

    Returns:
        np.ndarray: Audio signal with silence removed from both ends

    Raises:
        ValueError: If the audio contains only silence
    """
    start, end = detect_speech_boundaries(
        wav, sample_rate, window_duration, volume_threshold
    )
    return wav[start:end]


def hertz_to_mel(pitch: float) -> float:
    """
    Converts a frequency from the Hertz scale to the Mel scale.

    Parameters:
    - pitch: float or ndarray
        Frequency in Hertz.

    Returns:
    - mel: float or ndarray
        Frequency in Mel scale.
    """
    mel = HTK_MEL_SCALE_FACTOR * np.log10(1 + pitch / HTK_FREQ_REFERENCE)
    return mel


def generate_audio(
    text: str,
    model_path: str = "prince-canuma/Kokoro-82M",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    voice: str = "af_heart",
    speed: float = DEFAULT_SPEED,
    lang_code: str = "a",
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    stt_model: str = "mlx-community/whisper-large-v3-turbo",
    file_prefix: str = "audio",
    audio_format: str = "wav",
    join_audio: bool = False,
    play: bool = False,
    verbose: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    stream: bool = False,
    streaming_interval: float = DEFAULT_STREAMING_INTERVAL,
    **kwargs,
) -> None:
    """
    Generates audio from text using a specified TTS model.

    Parameters:
    - text (str): The input text to be converted to speech.
    - model (str): The TTS model to use.
    - voice (str): The voice style to use.
    - temperature (float): The temperature for the model.
    - speed (float): Playback speed multiplier.
    - lang_code (str): The language code.
    - ref_audio (mx.array): Reference audio you would like to clone the voice from.
    - ref_text (str): Caption for reference audio.
    - stt_model (str): A mlx whisper model to use to transcribe.
    - file_prefix (str): The output file path without extension.
    - audio_format (str): Output audio format (e.g., "wav", "flac").
    - join_audio (bool): Whether to join multiple audio files into one.
    - play (bool): Whether to play the generated audio.
    - verbose (bool): Whether to print status messages.
    Returns:
    - None: The function writes the generated audio to a file.
    """
    try:
        play = play or stream

        # Load model
        model = load_model(model_path=model_path)

        # Load reference audio for voice matching if specified
        if ref_audio:
            if not os.path.exists(ref_audio):
                raise FileNotFoundError(f"Reference audio file not found: {ref_audio}")

            normalize = False
            if hasattr(model, "model_type") and model.model_type() == "spark":
                normalize = True

            ref_audio = load_audio(
                ref_audio, sample_rate=model.sample_rate, volume_normalize=normalize
            )
            if not ref_text:
                print("Ref_text not found. Transcribing ref_audio...")
                from mlx_audio.stt.models.whisper import Model as Whisper

                stt_model = Whisper.from_pretrained(path_or_hf_repo=stt_model)
                ref_text = stt_model.generate(ref_audio).text
                print("Ref_text", ref_text)

                # clear memory
                del stt_model
                mx.clear_cache()

        # Load AudioPlayer
        player = AudioPlayer(sample_rate=model.sample_rate) if play else None

        print(
            f"\n\033[94mModel:\033[0m {model_path}\n"
            f"\033[94mText:\033[0m {text}\n"
            f"\033[94mVoice:\033[0m {voice}\n"
            f"\033[94mSpeed:\033[0m {speed}x\n"
            f"\033[94mLanguage:\033[0m {lang_code}"
        )

        results = model.generate(
            text=text,
            voice=voice,
            speed=speed,
            lang_code=lang_code,
            ref_audio=ref_audio,
            ref_text=ref_text,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            stream=stream,
            streaming_interval=streaming_interval,
            **kwargs,
        )

        audio_list = []
        file_name = f"{file_prefix}.{audio_format}"
        for i, result in enumerate(results):
            if play:
                player.queue_audio(result.audio)

            if join_audio:
                audio_list.append(result.audio)
            elif not stream:
                file_name = f"{file_prefix}_{i:0{FILE_INDEX_PADDING}d}.{audio_format}"
                sf.write(file_name, result.audio, result.sample_rate)
                print(f"✅ Audio successfully generated and saving as: {file_name}")

            if verbose:

                print("==========")
                print(f"Duration:              {result.audio_duration}")
                print(
                    f"Samples/sec:           {result.audio_samples['samples-per-sec']:.{DECIMAL_PRECISION}f}"
                )
                print(
                    f"Prompt:                {result.token_count} tokens, {result.prompt['tokens-per-sec']:.{DECIMAL_PRECISION}f} tokens-per-sec"
                )
                print(
                    f"Audio:                 {result.audio_samples['samples']} samples, {result.audio_samples['samples-per-sec']:.{DECIMAL_PRECISION}f} samples-per-sec"
                )
                print(f"Real-time factor:      {result.real_time_factor:.{DECIMAL_PRECISION_EXTENDED}f}x")
                print(f"Processing time:       {result.processing_time_seconds:.{DECIMAL_PRECISION_EXTENDED}f}s")
                print(f"Peak memory usage:     {result.peak_memory_usage:.{DECIMAL_PRECISION_EXTENDED}f}GB")

        if join_audio and not stream:
            if verbose:
                print(f"Joining {len(audio_list)} audio files")
            audio = mx.concatenate(audio_list, axis=0)
            sf.write(
                f"{file_prefix}.{audio_format}",
                audio,
                model.sample_rate,
            )
            if verbose:
                print(f"✅ Audio successfully generated and saving as: {file_name}")

        if play:
            player.wait_for_drain()
            player.stop()

    except ImportError as e:
        print(f"Import error: {e}")
        print(
            "This might be due to incorrect Python path. Check your project structure."
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback

        traceback.print_exc()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate audio from text using TTS.")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Kokoro-82M-bf16",
        help="Path or repo id of the model",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to generate (leave blank to input via stdin)",
    )
    parser.add_argument("--voice", type=str, default=None, help="Voice name")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="Speed of the audio")
    parser.add_argument(
        "--gender", type=str, default="male", help="Gender of the voice [male, female]"
    )
    parser.add_argument("--pitch", type=float, default=DEFAULT_PITCH, help="Pitch of the voice")
    parser.add_argument("--lang_code", type=str, default="a", help="Language code")
    parser.add_argument(
        "--file_prefix", type=str, default="audio", help="Output file name prefix"
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--join_audio", action="store_true", help="Join all audio files into one"
    )
    parser.add_argument("--play", action="store_true", help="Play the output audio")
    parser.add_argument(
        "--audio_format", type=str, default="wav", help="Output audio format"
    )
    parser.add_argument(
        "--ref_audio", type=str, default=None, help="Path to reference audio"
    )
    parser.add_argument(
        "--ref_text", type=str, default=None, help="Caption for reference audio"
    )
    parser.add_argument(
        "--stt_model",
        type=str,
        default="mlx-community/whisper-large-v3-turbo",
        help="STT model to use to transcribe reference audio",
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Temperature for the model"
    )
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P, help="Top-p for the model")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Top-k for the model")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
        help="Repetition penalty for the model",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the audio as segments instead of saving to a file",
    )
    parser.add_argument(
        "--streaming_interval",
        type=float,
        default=DEFAULT_STREAMING_INTERVAL,
        help="The time interval in seconds for streaming segments",
    )

    args = parser.parse_args()

    if args.text is None:
        if not sys.stdin.isatty():
            args.text = sys.stdin.read().strip()
        else:
            print("Please enter the text to generate:")
            args.text = input("> ").strip()

    return args


def main():
    args = parse_args()
    generate_audio(model_path=args.model, **vars(args))


if __name__ == "__main__":
    main()
