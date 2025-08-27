# Base classes and utilities for Text-to-Speech (TTS) models in MLX-Audio.
# 
# This module provides foundational components shared across all TTS model
# implementations, including configuration handling, audio processing utilities,
# and standardized result structures.
# 
# Cross-file Dependencies:
# - Used by all TTS model implementations in `mlx_audio.tts.models.*`
# - `BaseModelArgs` inherited by model-specific configuration classes
# - `GenerationResult` returned by all TTS generation methods
# - Audio utilities used by synthesis pipelines
# 
# Key Components:
# - Configuration base class with parameter filtering
# - Audio speed adjustment with linear interpolation
# - Array shape validation for neural network layers
# - Standardized generation result format

import inspect
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

# Audio processing constants
MIN_AUDIO_DIMENSIONS = 3              # Minimum dimensions for valid audio arrays
SPEED_FACTOR_FASTER = 1.0            # Baseline speed factor (no change)
INTERPOLATION_WEIGHT_BASE = 1.0       # Base weight for linear interpolation
ARRAY_RESHAPE_CHANNEL_DIM = -1        # Channel dimension for reshaping


@dataclass
class BaseModelArgs:
    # Base configuration class for all TTS model architectures.
    # 
    # Provides common functionality for TTS model configuration including
    # parameter validation, dictionary conversion, and inheritance support.
    # All model-specific configuration classes should inherit from this base.
    # 
    # The `from_dict` method enables safe loading of configurations from
    # JSON files, YAML files, or other dictionary sources while automatically
    # filtering out invalid parameters.
    # 
    # Used by:
    # - `ModelConfig` in Kokoro TTS implementation
    # - All other TTS model configuration classes
    # - Model loading and initialization functions
    # 
    # Features:
    # - Parameter validation through introspection
    # - Safe dictionary conversion with invalid key filtering
    # - Inheritance-friendly design for model-specific extensions
    
    @classmethod
    def from_dict(cls, params):
        # Creates model configuration from dictionary with parameter validation.
        # 
        # Safely constructs configuration object from dictionary by filtering
        # out keys that don't correspond to valid dataclass fields. This prevents
        # errors when loading configurations with extra or deprecated parameters.
        # 
        # Called by:
        # - Model loading functions when parsing config files
        # - Configuration factories and builders
        # - API endpoints accepting configuration dictionaries
        # 
        # Args:
        #     params (dict): Configuration parameters dictionary
        # 
        # Returns:
        #     BaseModelArgs: Validated configuration instance
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def check_array_shape(arr):
    # Validates array shape for neural network layer compatibility.
    # 
    # Verifies that input arrays have the expected 3D shape structure
    # with proper channel and spatial dimension relationships. This is
    # typically used for validating convolutional layer inputs.
    # 
    # Expected shape: (out_channels, height, width) where:
    # - out_channels is the largest dimension (feature channels)
    # - height and width are equal (square spatial dimensions)
    # 
    # Called by:
    # - TTS model initialization for parameter validation
    # - Layer construction functions for tensor shape verification
    # - Audio processing pipelines for format checking
    # 
    # Args:
    #     arr (mx.array): Input array to validate
    # 
    # Returns:
    #     bool: True if array shape is valid for neural network processing
    shape = arr.shape

    # Verify array has exactly 3 dimensions as expected
    if len(shape) != MIN_AUDIO_DIMENSIONS:
        return False

    out_channels, kH, KW = shape

    # Validate dimension relationships: channels >= spatial dims and square spatial
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False


def adjust_speed(audio_array, speed_factor):
    # Adjusts audio playback speed using linear interpolation resampling.
    # 
    # Changes the temporal duration of audio while preserving pitch by
    # resampling the waveform. Uses linear interpolation to compute intermediate
    # sample values, providing smooth speed adjustment without artifacts.
    # 
    # Speed Factor Effects:
    # - speed_factor > 1.0: Faster playback (shorter duration)
    # - speed_factor < 1.0: Slower playback (longer duration)  
    # - speed_factor = 1.0: No change (original speed)
    # 
    # Note: This is time-domain speed adjustment, not pitch-preserving
    # time-stretching. For pitch preservation, use specialized algorithms.
    # 
    # Called by:
    # - TTS synthesis pipelines for speech rate control
    # - Audio post-processing utilities
    # - Real-time audio applications requiring speed adjustment
    # 
    # Args:
    #     audio_array (mx.array | np.array): Input audio waveform
    #     speed_factor (float): Speed multiplier (>1 faster, <1 slower)
    # 
    # Returns:
    #     mx.array: Speed-adjusted audio with new temporal length
    
    # Ensure consistent MLX array format
    if not isinstance(audio_array, mx.array):
        audio_array = mx.array(audio_array)

    # Calculate target length after speed adjustment
    old_length = audio_array.shape[0]
    new_length = int(old_length / speed_factor)

    # Generate time indices for original and target sampling points
    old_indices = mx.arange(old_length)
    new_indices = mx.linspace(0, old_length - 1, new_length)

    # Implement linear interpolation manually (MLX doesn't have built-in interp)
    indices_floor = mx.floor(new_indices).astype(mx.int32)  # Lower sample indices
    indices_ceil = mx.minimum(indices_floor + 1, old_length - 1)  # Upper sample indices
    
    # Calculate interpolation weights
    weights_ceil = new_indices - indices_floor  # Weight for upper samples
    weights_floor = INTERPOLATION_WEIGHT_BASE - weights_ceil  # Weight for lower samples

    # Perform linear interpolation: result = w1*sample1 + w2*sample2
    result = (
        weights_floor.reshape(ARRAY_RESHAPE_CHANNEL_DIM, 1) * audio_array[indices_floor]
        + weights_ceil.reshape(ARRAY_RESHAPE_CHANNEL_DIM, 1) * audio_array[indices_ceil]
    )

    return result


@dataclass
class GenerationResult:
    # Standardized result structure for TTS model generation.
    # 
    # Contains comprehensive metadata about TTS synthesis including the generated
    # audio, performance metrics, and processing details. Provides consistent
    # interface across all TTS model implementations.
    # 
    # Used by:
    # - All TTS model `generate()` methods as return type
    # - TTS evaluation and benchmarking utilities
    # - Audio analysis and quality assessment tools
    # - Performance monitoring and optimization
    # 
    # Audio Data:
    #     audio (mx.array): Generated audio waveform
    #     samples (int): Number of audio samples generated
    #     sample_rate (int): Audio sampling rate in Hz
    #     audio_duration (str): Human-readable duration string
    # 
    # Generation Metadata:
    #     segment_idx (int): Index of current segment in batch processing
    #     token_count (int): Number of text/phoneme tokens processed
    #     prompt (dict): Input prompt and conditioning information
    # 
    # Performance Metrics:
    #     real_time_factor (float): Generation speed vs. real-time playback
    #     processing_time_seconds (float): Total processing time
    #     peak_memory_usage (float): Maximum memory used during generation
    
    # Core audio output
    audio: mx.array
    samples: int 
    sample_rate: int
    audio_duration: str
    
    # Generation metadata  
    segment_idx: int
    token_count: int
    prompt: dict
    
    # Performance metrics
    real_time_factor: float
    processing_time_seconds: float 
    peak_memory_usage: float
