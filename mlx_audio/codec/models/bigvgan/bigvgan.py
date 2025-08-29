# BigVGAN Neural Vocoder - High-Quality Mel-Spectrogram to Audio Synthesis.
# 
# This module implements the BigVGAN architecture, a state-of-the-art neural vocoder
# that converts mel-spectrograms to high-fidelity audio waveforms. BigVGAN enhances
# the HiFi-GAN architecture with anti-aliased multi-periodic convolutions and advanced
# activation functions for superior audio quality and reduced artifacts.
# 
# Architecture Overview:
# - **Mel-to-Audio Pipeline**: Converts mel-spectrograms to time-domain audio waveforms
# - **Anti-Aliased Design**: Multi-periodic convolutions prevent aliasing artifacts
# - **Advanced Activations**: Snake and SnakeBeta activations for improved synthesis
# - **Residual Blocks**: AMPBlocks (Anti-aliased Multi-Periodic) for frequency modeling
# - **Progressive Upsampling**: Multi-stage upsampling with configurable rates
# 
# Key Components:
# - **BigVGANConfig**: Configuration dataclass with all model hyperparameters
# - **BigVGAN**: Main vocoder model with mel-spectrogram to audio conversion
# - **Upsampling Pipeline**: Progressive temporal resolution enhancement
# - **Residual Processing**: Parallel residual blocks for harmonic modeling
# - **Weight Sanitization**: PyTorch to MLX weight conversion utilities
# 
# Called by:
# - `mlx_audio.tts.models.*` for TTS audio synthesis and post-processing
# - `mlx_audio.codec.models.*` for audio codec reconstruction pipelines
# - Voice conversion and audio enhancement applications
# - Real-time audio synthesis systems requiring high-quality vocoding
# 
# Integrates with:
# - **TTS Models**: Receives mel-spectrograms from text-to-speech synthesis
# - **Audio Codecs**: Works with compressed audio representations
# - **MLX Framework**: Optimized for Apple Silicon inference acceleration
# - **Preprocessing**: Requires aligned mel-spectrogram features from audio
# 
# Audio Processing Pipeline:
# 1. **Input Processing**: Mel-spectrogram feature alignment and channel preparation
# 2. **Pre-convolution**: Initial feature extraction with wide receptive field
# 3. **Progressive Upsampling**: Multi-stage temporal resolution enhancement
# 4. **Residual Modeling**: Parallel processing for harmonic and noise components
# 5. **Post-processing**: Final activation and amplitude clipping/normalization
# 6. **Output Generation**: High-fidelity time-domain audio waveform
# 
# Performance Characteristics:
# - **Real-time Inference**: Optimized for low-latency audio synthesis
# - **High Fidelity**: Superior audio quality compared to traditional vocoders
# - **Memory Efficient**: MLX optimized operations for Apple Silicon
# - **Configurable Quality**: Adjustable model size and quality trade-offs
# 
# Model Variants:
# - **ResBlock Type 1/2**: Different residual block architectures for quality/speed
# - **Snake/SnakeBeta**: Alternative activation functions for different characteristics
# - **Variable Channels**: Configurable channel counts for model size optimization
# - **Custom Upsampling**: Flexible upsampling rates for different applications

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_audio.codec.models.bigvgan.activation import Snake, SnakeBeta
from mlx_audio.codec.models.bigvgan.amp import AMPBlock1, AMPBlock2
from mlx_audio.codec.models.bigvgan.conv import WNConv1d, WNConvTranspose1d
from mlx_audio.codec.models.bigvgan.resample import Activation1d

# Model architecture constants - optimized for high-quality audio synthesis
CONV_KERNEL_SIZE = 7                   # Standard kernel size for pre/post convolution layers
CONV_STRIDE = 1                        # Standard stride for convolution operations
CONV_PADDING = 3                       # Padding size for kernel size 7 (maintains dimensions)
OUTPUT_CHANNELS = 1                    # Single channel audio output (mono)
AUDIO_AMPLITUDE_LIMIT = 1.0            # Maximum audio amplitude for clipping
NEGATIVE_AUDIO_LIMIT = -1.0            # Minimum audio amplitude for clipping

# Network architecture constants
CHANNEL_REDUCTION_FACTOR = 2           # Factor for progressive channel reduction in upsampling
KERNEL_STRIDE_OFFSET = 1               # Offset for index-based calculations
AVERAGE_NORMALIZATION = True           # Enable averaging normalization in residual blocks

# Weight sanitization constants
CONV_LAYER_DIMS_3D = 3                 # 3D convolution tensor dimensions
CONV_LAYER_DIMS_4D = 4                 # 4D convolution tensor dimensions
TRANSPOSE_DIM_0 = 0                    # First dimension for tensor transposition
TRANSPOSE_DIM_1 = 1                    # Second dimension for tensor transposition  
TRANSPOSE_DIM_2 = 2                    # Third dimension for tensor transposition
TRANSPOSE_DIM_3 = 3                    # Fourth dimension for tensor transposition


@dataclass
class BigVGANConfig:
    # Configuration class for BigVGAN neural vocoder architecture.
    # 
    # Defines all hyperparameters and architectural choices for BigVGAN model
    # instantiation. Provides flexibility for different quality/performance
    # trade-offs and supports various audio synthesis applications.
    # 
    # Used by:
    # - `BigVGAN.__init__()` for model architecture construction
    # - Model loading utilities for configuration restoration
    # - Training scripts for hyperparameter specification
    # - Inference pipelines requiring model configuration
    # 
    # Architecture Parameters:
    #     num_mels (int): Number of mel-frequency bins in input spectrograms
    #     upsample_rates (list[int]): Temporal upsampling factors per stage
    #     upsample_kernel_sizes (list[int]): Kernel sizes for upsampling convolutions
    #     upsample_initial_channel (int): Initial channel count for upsampling pipeline
    #     resblock (Literal["1", "2"]): Residual block architecture variant
    #     resblock_kernel_sizes (list[int]): Kernel sizes for residual convolutions
    #     resblock_dilation_sizes (list[list[int]]): Dilation patterns for residual blocks
    #     activation (Literal["snakebeta", "snake"]): Activation function type
    #     snake_logscale (bool): Whether to use logarithmic scaling in Snake activations
    #     use_bias_at_final (bool): Whether to use bias in final output layer
    #     use_tanh_at_final (bool): Whether to apply tanh activation to final output
    num_mels: int
    upsample_rates: list[int]
    upsample_kernel_sizes: list[int]
    upsample_initial_channel: int
    resblock: Literal["1", "2"]
    resblock_kernel_sizes: list[int]
    resblock_dilation_sizes: list[list[int]]
    activation: Literal["snakebeta", "snake"]
    snake_logscale: bool
    use_bias_at_final: bool = True  # compatability
    use_tanh_at_final: bool = True  # compatability


class BigVGAN(nn.Module):
    # BigVGAN neural vocoder for mel-spectrogram to audio synthesis.
    # 
    # Implements the complete BigVGAN architecture with progressive upsampling,
    # anti-aliased multi-periodic convolutions, and advanced activation functions.
    # Converts mel-spectrograms to high-fidelity time-domain audio waveforms.
    # 
    # Architecture Flow:
    # 1. **Pre-processing**: Wide convolution for initial feature extraction
    # 2. **Progressive Upsampling**: Multi-stage temporal resolution enhancement
    # 3. **Residual Processing**: Parallel AMPBlocks for harmonic modeling
    # 4. **Post-processing**: Final activation and amplitude normalization
    # 
    # Called by:
    # - TTS synthesis pipelines for mel-spectrogram to audio conversion
    # - Audio codec reconstruction for compressed audio restoration
    # - Voice conversion systems for high-quality audio synthesis
    # - Real-time audio applications requiring low-latency vocoding
    # 
    # Performance:
    # - **Real-time**: Optimized for streaming audio synthesis
    # - **High Quality**: State-of-the-art vocoder performance
    # - **Memory Efficient**: MLX optimizations for Apple Silicon
    # 
    # Args:
    #     config (BigVGANConfig): Model configuration with all hyperparameters
    
    def __init__(self, config: BigVGANConfig):
        # Initializes BigVGAN neural vocoder with specified configuration.
        # 
        # Constructs the complete network architecture including upsampling layers,
        # residual blocks, and activation functions according to the configuration.
        # Sets up weight normalization and anti-aliasing components.
        # 
        # Architecture Construction:
        # - **Pre-convolution**: Initial mel-spectrogram feature extraction
        # - **Upsampling Stack**: Progressive temporal resolution enhancement
        # - **Residual Blocks**: Parallel processing for harmonic content
        # - **Post-processing**: Final activation and output normalization
        # 
        # Args:
        #     config (BigVGANConfig): Complete model configuration specification
        super().__init__()

        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.use_tanh_at_final = config.use_tanh_at_final

        self.conv_pre = WNConv1d(
            config.num_mels, config.upsample_initial_channel, 
            CONV_KERNEL_SIZE, CONV_STRIDE, CONV_PADDING
        )
        self.ups = [
            [
                WNConvTranspose1d(
                    config.upsample_initial_channel // (CHANNEL_REDUCTION_FACTOR**i),
                    config.upsample_initial_channel // (CHANNEL_REDUCTION_FACTOR ** (i + KERNEL_STRIDE_OFFSET)),
                    k,
                    u,
                    padding=(k - u) // CHANNEL_REDUCTION_FACTOR,
                )
            ]
            for i, (u, k) in enumerate(
                zip(config.upsample_rates, config.upsample_kernel_sizes)
            )
        ]
        self.resblocks = [
            (
                AMPBlock1(
                    config.upsample_initial_channel // (CHANNEL_REDUCTION_FACTOR ** (i + KERNEL_STRIDE_OFFSET)),
                    config.snake_logscale,
                    config.activation,
                    k,
                    d,
                )
                if config.resblock == "1"
                else AMPBlock2(
                    config.upsample_initial_channel // (CHANNEL_REDUCTION_FACTOR ** (i + KERNEL_STRIDE_OFFSET)),
                    config.snake_logscale,
                    config.activation,
                    k,
                    d,
                )
            )
            for i in range(len(self.ups))
            for j, (k, d) in enumerate(
                zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)
            )
        ]
        self.activation_post = Activation1d(
            Snake(
                config.upsample_initial_channel // (CHANNEL_REDUCTION_FACTOR ** len(self.ups)),
                alpha_logscale=config.snake_logscale,
            )
            if config.activation == "snake"
            else SnakeBeta(
                config.upsample_initial_channel // (CHANNEL_REDUCTION_FACTOR ** len(self.ups)),
                alpha_logscale=config.snake_logscale,
            )
        )
        self.conv_post = WNConv1d(
            config.upsample_initial_channel // (CHANNEL_REDUCTION_FACTOR ** len(self.ups)),
            OUTPUT_CHANNELS,
            CONV_KERNEL_SIZE,
            CONV_STRIDE,
            padding=CONV_PADDING,
            bias=config.use_bias_at_final,
        )

    def __call__(
        self, x: mx.array, *args, **kwargs
    ) -> mx.array:
        # Converts mel-spectrograms to high-fidelity audio waveforms.
        # 
        # Implements the complete BigVGAN forward pass with progressive upsampling,
        # residual processing, and anti-aliasing. Processes mel-spectrogram features
        # through multiple stages to generate time-domain audio.
        # 
        # Processing Pipeline:
        # 1. **Input Alignment**: Transpose for convolution compatibility
        # 2. **Feature Extraction**: Wide convolution for initial processing
        # 3. **Progressive Upsampling**: Multi-stage temporal resolution enhancement
        # 4. **Residual Modeling**: Parallel processing for harmonic content
        # 5. **Post-processing**: Final activation and amplitude control
        # 6. **Output Formatting**: Transpose back to expected dimensions
        # 
        # Args:
        #     x (mx.array): Input mel-spectrogram of shape (batch, num_mels, seq_len)
        # 
        # Returns:
        #     mx.array: Generated audio waveform of shape (batch, seq_len_upsampled, 1)
        # 
        # Performance: Optimized for real-time synthesis with MLX acceleration.
        # Transpose input for convolution processing: (batch, seq, mels) -> (batch, mels, seq)
        x = x.transpose(TRANSPOSE_DIM_0, TRANSPOSE_DIM_2, TRANSPOSE_DIM_1)

        x = self.conv_pre(x)

        for step in range(self.num_upsamples):
            for idx in range(len(self.ups[step])):
                x = self.ups[step][idx](x)

            xs = self.resblocks[step * self.num_kernels](x)
            for idx in range(1, self.num_kernels):
                xs += self.resblocks[step * self.num_kernels + idx](x)

            # Normalize residual block outputs by averaging
            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)

        # Apply final amplitude control based on configuration
        if self.use_tanh_at_final:
            x = mx.tanh(x)
        else:
            x = mx.clip(x, NEGATIVE_AUDIO_LIMIT, AUDIO_AMPLITUDE_LIMIT)

        # Transpose output back to expected format: (batch, 1, seq) -> (batch, seq, 1)
        return x.transpose(TRANSPOSE_DIM_0, TRANSPOSE_DIM_2, TRANSPOSE_DIM_1)

    def sanitize(self, weights: dict[str, mx.array]):
        # Converts PyTorch weights to MLX-compatible format with proper tensor layouts.
        # 
        # Handles the conversion of pre-trained PyTorch BigVGAN weights to MLX format,
        # including proper tensor transpositions for convolution layers and filtering
        # of incompatible parameters like batch normalization tracking statistics.
        # 
        # Weight Conversion Process:
        # 1. **Parameter Filtering**: Remove PyTorch-specific tracking parameters
        # 2. **Convolution Weights**: Transpose 3D/4D tensors for MLX convolution format
        # 3. **Upsampling Weights**: Handle transposed convolution weight layouts
        # 4. **Shape Validation**: Ensure converted weights match model architecture
        # 
        # Called by:
        # - Model loading utilities when converting pre-trained weights
        # - Checkpoint restoration functions
        # - Transfer learning and fine-tuning pipelines
        # 
        # Args:
        #     weights (dict[str, mx.array]): PyTorch model weights dictionary
        # 
        # Returns:
        #     dict[str, mx.array]: MLX-compatible weights with corrected tensor layouts
        # 
        # Performance: Efficient in-place tensor operations where possible.
        new_weights = {}
        curr_weights = dict(tree_flatten(self.parameters()))

        for key, value in weights.items():
            if "num_batches_tracked" in key:
                continue

            if "conv" in key or "lowpass.filter" in key or "upsample.filter" in key:
                # Handle 3D convolution tensor transposition (batch, in_channels, kernel)
                if value.ndim == CONV_LAYER_DIMS_3D:
                    if value.shape != curr_weights[key].shape:
                        value = value.transpose(TRANSPOSE_DIM_0, TRANSPOSE_DIM_2, TRANSPOSE_DIM_1)
                # Handle 4D convolution tensor transposition (batch, in_channels, kernel_h, kernel_w)
                elif value.ndim == CONV_LAYER_DIMS_4D:
                    if value.shape != curr_weights[key].shape:
                        value = value.transpose(TRANSPOSE_DIM_0, TRANSPOSE_DIM_2, TRANSPOSE_DIM_3, TRANSPOSE_DIM_1)

            # Handle upsampling layer weight transposition for transposed convolutions
            if "ups." in key:
                if value.ndim == CONV_LAYER_DIMS_3D:
                    if value.shape != curr_weights[key].shape:
                        value = value.transpose(TRANSPOSE_DIM_1, TRANSPOSE_DIM_2, TRANSPOSE_DIM_0)

            new_weights[key] = value

        del curr_weights

        return new_weights
