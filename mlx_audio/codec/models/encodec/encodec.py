# EnCodec neural audio codec implementation for MLX.
# 
# This module implements the complete EnCodec architecture from Meta's
# "High Fidelity Neural Audio Compression" paper, providing state-of-the-art
# audio compression with perceptual quality preservation.
# 
# EnCodec uses a convolutional encoder-decoder architecture with residual
# vector quantization (RVQ) for efficient audio compression at various bitrates.
# The model supports both causal and non-causal convolutions for streaming
# and non-streaming applications.
# 
# Cross-file Dependencies:
# - Uses `mlx_audio.utils` for STFT/ISTFT and windowing functions
# - Called by `mlx_audio.codec.tests.test_encodec` for validation
# - Used by applications requiring high-quality audio compression
# 
# Key Features:
# - Variable bitrate compression (1.5-24 kbps)
# - Streaming and non-streaming modes
# - Perceptual loss optimization
# - Residual vector quantization
# - Custom MLX LSTM implementation with Metal kernels

import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download

# EnCodec model constants
DEFAULT_SAMPLING_RATE = 24000        # Standard sampling rate for EnCodec
DEFAULT_NUM_FILTERS = 32             # Base number of filters in conv layers
DEFAULT_KERNEL_SIZE = 7              # Default convolutional kernel size
DEFAULT_CODEBOOK_SIZE = 1024         # Vector quantization codebook size
DEFAULT_CODEBOOK_DIM = 128           # Dimension of each codebook vector
DEFAULT_HIDDEN_SIZE = 128            # Hidden dimension for LSTM layers
DEFAULT_NUM_LSTM_LAYERS = 2          # Number of LSTM layers in bottleneck
RESIDUAL_KERNEL_SIZE = 3             # Kernel size for residual connections
DEFAULT_DILATION_GROWTH_RATE = 2     # Growth rate for dilated convolutions
DEFAULT_COMPRESSION_FACTOR = 2       # Default temporal compression ratio
LAST_KERNEL_SIZE = 7                 # Final layer kernel size
TRIM_RIGHT_RATIO = 1.0               # Right padding trim ratio for causal mode

# LSTM Metal kernel constants
LSTM_GATE_COUNT = 4                  # Number of LSTM gates (i, f, g, o)
LSTM_THREADGROUP_SIZE = 256          # Metal threadgroup size for LSTM kernel


def filter_dataclass_fields(data_dict, dataclass_type):
    # Filters dictionary to only include valid dataclass fields.
    # 
    # Helper function for safe dataclass instantiation from potentially
    # untrusted configuration dictionaries. Only includes keys that match
    # actual dataclass field names, preventing errors from extra keys.
    # 
    # Called by:
    # - Model loading functions when parsing configuration files
    # - Configuration validation and sanitization
    # 
    # Args:
    #     data_dict (dict): Input dictionary with potential extra keys
    #     dataclass_type: Dataclass type to validate against
    # 
    # Returns:
    #     dict: Filtered dictionary with only valid field names
    valid_fields = {f.name for f in dataclass_type.__dataclass_fields__.values()}
    return {k: v for k, v in data_dict.items() if k in valid_fields}


@dataclass
class EncodecConfig:
    # Complete configuration specification for EnCodec model architecture.
    # 
    # Defines all hyperparameters and architectural choices for the EnCodec
    # neural audio codec. These parameters control model size, compression
    # ratio, quality, and computational requirements.
    # 
    # Used by:
    # - `EncodecModel.__init__()` for model construction
    # - Model loading and saving functions
    # - Configuration validation and parameter tuning
    # 
    # Architecture Parameters:
    #     model_type (str): Model identifier ("encodec")
    #     audio_channels (int): Number of audio channels (1=mono, 2=stereo)
    #     num_filters (int): Base number of convolutional filters
    #     kernel_size (int): Convolutional kernel size
    #     num_residual_layers (int): Number of residual blocks per stage
    #     dilation_growth_rate (int): Dilation growth factor for temporal modeling
    # 
    # Quantization Parameters:
    #     codebook_size (int): Size of vector quantization codebook
    #     codebook_dim (int): Dimension of quantization vectors
    #     compress (int): Temporal compression factor
    # 
    # Bottleneck Parameters:
    #     hidden_size (int): LSTM hidden dimension
    #     num_lstm_layers (int): Number of LSTM layers in bottleneck
    # 
    # Processing Parameters:
    #     use_causal_conv (bool): Whether to use causal convolutions (for streaming)
    #     normalize (bool): Whether to normalize audio input
    #     pad_mode (str): Padding mode ("reflect", "constant", etc.)
    #     norm_type (str): Normalization type ("weight_norm", "time_group_norm")
    #     sampling_rate (int): Expected audio sampling rate in Hz
    model_type: str = "encodec"
    audio_channels: int = 1
    num_filters: int = DEFAULT_NUM_FILTERS
    kernel_size: int = DEFAULT_KERNEL_SIZE
    num_residual_layers: int = 1
    dilation_growth_rate: int = DEFAULT_DILATION_GROWTH_RATE
    codebook_size: int = DEFAULT_CODEBOOK_SIZE
    codebook_dim: int = DEFAULT_CODEBOOK_DIM
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_lstm_layers: int = DEFAULT_NUM_LSTM_LAYERS
    residual_kernel_size: int = RESIDUAL_KERNEL_SIZE
    use_causal_conv: bool = True
    normalize: bool = False
    pad_mode: str = "reflect"
    norm_type: str = "weight_norm"
    last_kernel_size: int = LAST_KERNEL_SIZE
    trim_right_ratio: float = TRIM_RIGHT_RATIO
    compress: int = DEFAULT_COMPRESSION_FACTOR
    upsampling_ratios: List[int] = None
    target_bandwidths: List[float] = None
    sampling_rate: int = DEFAULT_SAMPLING_RATE
    chunk_length_s: Optional[float] = None
    overlap: Optional[float] = None
    architectures: List[str] = None


def preprocess_audio(
    raw_audio: Union[mx.array, List[mx.array]],
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    chunk_length: Optional[int] = None,
    chunk_stride: Optional[int] = None,
):
    # Preprocesses raw audio for EnCodec model input with batching and padding.
    # 
    # Converts raw audio arrays to model-compatible format with consistent
    # batch dimensions and proper padding. Handles both single audio arrays
    # and lists of arrays for batch processing.
    # 
    # The function ensures all audio has proper channel dimension (adds if mono)
    # and pads to consistent length within batch for efficient processing.
    # Optional chunking support for processing long audio files.
    # 
    # Called by:
    # - `EncodecModel.encode()` for input preprocessing
    # - Audio preprocessing pipelines
    # - Batch inference utilities
    # 
    # Args:
    #     raw_audio: Single audio array or list of arrays to preprocess
    #     sampling_rate (int): Target sampling rate for audio processing
    #     chunk_length (int, optional): Fixed chunk length for segmentation
    #     chunk_stride (int, optional): Stride for overlapping chunks
    # 
    # Returns:
    #     tuple: (batched_inputs, attention_masks)
    #            - batched_inputs (mx.array): Batched audio of shape (B, T, C)
    #            - attention_masks (mx.array): Valid sample masks of shape (B, T)
    if not isinstance(raw_audio, list):
        raw_audio = [raw_audio]

    raw_audio = [x[..., None] if x.ndim == 1 else x for x in raw_audio]

    max_length = max(array.shape[0] for array in raw_audio)
    if chunk_length is not None:
        max_length += chunk_length - (max_length % chunk_stride)

    inputs = []
    masks = []
    for x in raw_audio:
        length = x.shape[0]
        mask = mx.ones((length,), dtype=mx.bool_)
        difference = max_length - length
        if difference > 0:
            mask = mx.pad(mask, (0, difference))
            x = mx.pad(x, ((0, difference), (0, 0)))
        inputs.append(x)
        masks.append(mask)
    return mx.stack(inputs), mx.stack(masks)


# Custom Metal compute kernel for optimized LSTM computation on Apple Silicon.
# 
# This Metal kernel implements the core LSTM cell computation with maximum
# efficiency on Apple's GPU architecture. The kernel processes all 4 LSTM gates
# (input, forget, cell candidate, output) in parallel across the batch dimension.
# 
# Performance Benefits:
# - Direct Metal GPU execution bypassing MLX overhead
# - Parallel processing across batch dimension
# - Optimized memory access patterns
# - Fused sigmoid and tanh operations
# 
# The kernel computes the standard LSTM equations:
# i_t = sigmoid(W_xi * x_t + W_hi * h_{t-1} + b_i)  # Input gate
# f_t = sigmoid(W_xf * x_t + W_hf * h_{t-1} + b_f)  # Forget gate
# g_t = tanh(W_xg * x_t + W_hg * h_{t-1} + b_g)     # Cell candidate
# o_t = sigmoid(W_xo * x_t + W_ho * h_{t-1} + b_o)  # Output gate
# c_t = f_t * c_{t-1} + i_t * g_t                   # Cell state
# h_t = o_t * tanh(c_t)                             # Hidden state
_lstm_kernel = mx.fast.metal_kernel(
    name="lstm",
    input_names=["x", "h_in", "cell", "hidden_size", "time_step", "num_time_steps"],
    output_names=["hidden_state", "cell_state"],
    header="""
    template <typename T>
    T sigmoid(T x) {
        auto y = 1 / (1 + metal::exp(-metal::abs(x)));
        return (x < 0) ? 1 - y : y;
    }
    """,
    source="""
        uint b = thread_position_in_grid.x;
        uint d = hidden_size * 4;

        uint elem = b * d + thread_position_in_grid.y;
        uint index = elem;
        uint x_index = b * num_time_steps * d + time_step * d + index;

        auto i = sigmoid(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto f = sigmoid(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto g = metal::precise::tanh(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto o = sigmoid(h_in[index] + x[x_index]);

        cell_state[elem] = f * cell[elem] + i * g;
        hidden_state[elem] = o * metal::precise::tanh(cell_state[elem]);
    """,
)


def lstm_custom(x, h_in, cell, time_step):
    # Invokes the custom Metal LSTM kernel for efficient computation.
    # 
    # Wrapper function that configures and dispatches the Metal LSTM kernel
    # with proper tensor shapes, data types, and grid configuration for
    # optimal GPU utilization on Apple Silicon.
    # 
    # Called by:
    # - `LSTM.__call__()` for each time step in sequence processing
    # - EnCodec LSTM layers during encoding and decoding
    # 
    # Args:
    #     x (mx.array): Input tensor of shape (batch, seq_len, 4*hidden_size)
    #     h_in (mx.array): Previous hidden state of shape (batch, 4*hidden_size)
    #     cell (mx.array): Previous cell state of shape (batch, hidden_size)
    #     time_step (int): Current time step index in sequence
    # 
    # Returns:
    #     tuple: (new_hidden_state, new_cell_state) both of shape (batch, hidden_size)
    # 
    # Raises:
    #     AssertionError: If input tensor doesn't have exactly 3 dimensions
    assert x.ndim == 3, "Input to LSTM must have 3 dimensions."
    out_shape = cell.shape
    return _lstm_kernel(
        inputs=[x, h_in, cell, out_shape[-1], time_step, x.shape[-2]],
        output_shapes=[out_shape, out_shape],
        output_dtypes=[h_in.dtype, h_in.dtype],
        grid=(x.shape[0], h_in.size // LSTM_GATE_COUNT, 1),  # Grid dimensions for parallel processing
        threadgroup=(LSTM_THREADGROUP_SIZE, 1, 1),          # Metal threadgroup size
    )


class LSTM(nn.Module):
    # Custom LSTM implementation optimized for MLX with Metal kernel acceleration.
    # 
    # Implements Long Short-Term Memory recurrent neural network with custom
    # Metal compute kernels for optimal performance on Apple Silicon. Uses
    # standard LSTM gating mechanism with input, forget, cell, and output gates.
    # 
    # The implementation is optimized for the EnCodec bottleneck where LSTM
    # layers model temporal dependencies in compressed audio representations.
    # Metal kernels provide significant speedup over standard MLX operations.
    # 
    # Used by:
    # - `EncodecLSTM` as the core recurrent component
    # - EnCodec encoder-decoder bottleneck for temporal modeling
    # 
    # Architecture:
    # - 4 gates per cell: input (i), forget (f), cell candidate (g), output (o)
    # - Sigmoid activation for gates, tanh for cell state
    # - Optional bias terms for all linear transformations
    # 
    # Args:
    #     input_size (int): Input feature dimension
    #     hidden_size (int): Hidden state and cell state dimension
    #     bias (bool): Whether to include bias terms in linear layers
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        # Weight matrices: 4 gates × hidden_size for LSTM
        self.Wx = mx.zeros((LSTM_GATE_COUNT * hidden_size, input_size))  # Input-to-hidden weights
        self.Wh = mx.zeros((LSTM_GATE_COUNT * hidden_size, hidden_size)) # Hidden-to-hidden weights
        self.bias = mx.zeros((LSTM_GATE_COUNT * hidden_size,)) if bias else None  # Bias terms

    def __call__(self, x, hidden=None, cell=None):
        if self.bias is not None:
            x = mx.addmm(self.bias, x, self.Wx.T)
        else:
            x = x @ self.Wx.T

        all_hidden = []

        B = x.shape[0]
        cell = cell or mx.zeros((B, self.hidden_size), x.dtype)
        for t in range(x.shape[-2]):
            if hidden is None:
                hidden = mx.zeros((B, self.hidden_size * 4), x.dtype)
            else:
                hidden = hidden @ self.Wh.T
            hidden, cell = lstm_custom(x, hidden, cell, t)
            all_hidden.append(hidden)

        return mx.stack(all_hidden, axis=-2)


class EncodecConv1d(nn.Module):
    # 1D convolutional layer with EnCodec-specific padding and normalization.
    # 
    # Implements 1D convolution with flexible padding strategies to support
    # both causal (streaming) and non-causal (offline) processing modes.
    # Includes optional time-domain group normalization for stable training.
    # 
    # Features:
    # - Asymmetric padding for proper temporal alignment
    # - Causal padding for streaming applications
    # - Reflect padding for boundary artifact reduction
    # - Optional group normalization in time domain
    # - Dilated convolutions for expanded receptive fields
    # 
    # Used by:
    # - `EncodecEncoder` convolutional downsampling layers
    # - `EncodecDecoder` convolutional upsampling layers
    # - Residual blocks throughout the EnCodec architecture
    # 
    # Args:
    #     config (EncodecConfig): Model configuration with padding and norm settings
    #     in_channels (int): Number of input channels
    #     out_channels (int): Number of output channels
    #     kernel_size (int): Convolution kernel size
    #     stride (int): Convolution stride for temporal downsampling
    #     dilation (int): Dilation rate for expanded receptive field
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.causal = config.use_causal_conv
        self.pad_mode = config.pad_mode
        self.norm_type = config.norm_type

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, dilation=dilation
        )
        if self.norm_type == "time_group_norm":
            self.norm = nn.GroupNorm(1, out_channels, pytorch_compatible=True)

        self.stride = stride

        # Effective kernel size accounting for dilation
        self.kernel_size = (kernel_size - 1) * dilation + 1

        # Total padding needed for proper output alignment
        self.padding_total = kernel_size - stride

    def _get_extra_padding_for_conv1d(
        self,
        hidden_states: mx.array,
    ) -> mx.array:
        length = hidden_states.shape[1]
        n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        n_frames = int(math.ceil(n_frames)) - 1
        ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        return ideal_length - length

    def _pad1d(
        self,
        hidden_states: mx.array,
        paddings: Tuple[int, int],
        mode: str = "zero",
        value: float = 0.0,
    ):
        if mode != "reflect":
            return mx.pad(
                hidden_states, paddings, mode="constant", constant_values=value
            )

        length = hidden_states.shape[1]
        prefix = hidden_states[:, 1 : paddings[0] + 1][:, ::-1]
        suffix = hidden_states[:, max(length - (paddings[1] + 1), 0) : -1][:, ::-1]
        return mx.concatenate([prefix, hidden_states, suffix], axis=1)

    def __call__(self, hidden_states):
        extra_padding = self._get_extra_padding_for_conv1d(hidden_states)

        if self.causal:
            # Left padding for causal
            hidden_states = self._pad1d(
                hidden_states, (self.padding_total, extra_padding), mode=self.pad_mode
            )
        else:
            # Asymmetric padding required for odd strides
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right
            hidden_states = self._pad1d(
                hidden_states,
                (padding_left, padding_right + extra_padding),
                mode=self.pad_mode,
            )

        hidden_states = self.conv(hidden_states)

        if self.norm_type == "time_group_norm":
            hidden_states = self.norm(hidden_states)

        return hidden_states


class EncodecConvTranspose1d(nn.Module):
    # Transposed 1D convolution for EnCodec upsampling with proper padding.
    # 
    # Implements transposed convolution (deconvolution) for temporal upsampling
    # in the EnCodec decoder. Handles both causal and non-causal modes with
    # appropriate padding to maintain proper temporal alignment.
    # 
    # The layer performs upsampling to reconstruct higher temporal resolution
    # from compressed representations, essential for audio reconstruction.
    # 
    # Features:
    # - Transposed convolution for temporal upsampling
    # - Causal and non-causal padding strategies
    # - Configurable right-side trimming for causal mode
    # - Optional time-domain group normalization
    # 
    # Used by:
    # - `EncodecDecoder` upsampling layers
    # - Audio reconstruction pathway in EnCodec model
    # 
    # Args:
    #     config (EncodecConfig): Model configuration with mode and norm settings
    #     in_channels (int): Number of input channels
    #     out_channels (int): Number of output channels  
    #     kernel_size (int): Transposed convolution kernel size
    #     stride (int): Upsampling stride factor
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super().__init__()
        self.causal = config.use_causal_conv
        self.trim_right_ratio = config.trim_right_ratio
        self.norm_type = config.norm_type
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        if config.norm_type == "time_group_norm":
            self.norm = nn.GroupNorm(1, out_channels, pytorch_compatible=True)
        self.padding_total = kernel_size - stride

    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        if self.norm_type == "time_group_norm":
            hidden_states = self.norm(hidden_states)

        if self.causal:
            padding_right = math.ceil(self.padding_total * self.trim_right_ratio)
        else:
            padding_right = self.padding_total // 2

        padding_left = self.padding_total - padding_right

        end = hidden_states.shape[1] - padding_right
        hidden_states = hidden_states[:, padding_left:end, :]
        return hidden_states


class EncodecLSTM(nn.Module):
    def __init__(self, config, dimension):
        super().__init__()
        self.lstm = [LSTM(dimension, dimension) for _ in range(config.num_lstm_layers)]

    def __call__(self, hidden_states):
        h = hidden_states
        for lstm in self.lstm:
            h = lstm(h)
        return h + hidden_states


class EncodecResnetBlock(nn.Module):
    """
    Residual block from SEANet model as used by EnCodec.
    """

    def __init__(self, config, dim: int, dilations: List[int]):
        super().__init__()
        kernel_sizes = (config.residual_kernel_size, 1)
        if len(kernel_sizes) != len(dilations):
            raise ValueError("Number of kernel sizes should match number of dilations")

        hidden = dim // config.compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [nn.ELU()]
            block += [
                EncodecConv1d(config, in_chs, out_chs, kernel_size, dilation=dilation)
            ]
        self.block = block

        if getattr(config, "use_conv_shortcut", True):
            self.shortcut = EncodecConv1d(config, dim, dim, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def __call__(self, hidden_states):
        residual = hidden_states
        for layer in self.block:
            hidden_states = layer(hidden_states)

        return self.shortcut(residual) + hidden_states


class EncodecEncoder(nn.Module):
    """SEANet encoder as used by EnCodec."""

    def __init__(self, config):
        super().__init__()
        model = [
            EncodecConv1d(
                config, config.audio_channels, config.num_filters, config.kernel_size
            )
        ]
        scaling = 1

        for ratio in reversed(config.upsampling_ratios):
            current_scale = scaling * config.num_filters
            for j in range(config.num_residual_layers):
                model += [
                    EncodecResnetBlock(
                        config, current_scale, [config.dilation_growth_rate**j, 1]
                    )
                ]
            model += [nn.ELU()]
            model += [
                EncodecConv1d(
                    config,
                    current_scale,
                    current_scale * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            ]
            scaling *= 2

        model += [EncodecLSTM(config, scaling * config.num_filters)]
        model += [nn.ELU()]
        model += [
            EncodecConv1d(
                config,
                scaling * config.num_filters,
                config.hidden_size,
                config.last_kernel_size,
            )
        ]

        self.layers = model

    def __call__(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class EncodecDecoder(nn.Module):
    """SEANet decoder as used by EnCodec."""

    def __init__(self, config):
        super().__init__()
        scaling = int(2 ** len(config.upsampling_ratios))
        model = [
            EncodecConv1d(
                config,
                config.hidden_size,
                scaling * config.num_filters,
                config.kernel_size,
            )
        ]

        model += [EncodecLSTM(config, scaling * config.num_filters)]

        for ratio in config.upsampling_ratios:
            current_scale = scaling * config.num_filters
            model += [nn.ELU()]
            model += [
                EncodecConvTranspose1d(
                    config,
                    current_scale,
                    current_scale // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            ]
            for j in range(config.num_residual_layers):
                model += [
                    EncodecResnetBlock(
                        config, current_scale // 2, (config.dilation_growth_rate**j, 1)
                    )
                ]
            scaling //= 2

        model += [nn.ELU()]
        model += [
            EncodecConv1d(
                config,
                config.num_filters,
                config.audio_channels,
                config.last_kernel_size,
            )
        ]
        self.layers = model

    def __call__(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class EncodecEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""

    def __init__(self, config):
        super().__init__()
        self.embed = mx.zeros((config.codebook_size, config.codebook_dim))

    def quantize(self, hidden_states):
        embed = self.embed.T
        scaled_states = hidden_states.square().sum(axis=1, keepdims=True)
        dist = -(
            scaled_states
            - 2 * hidden_states @ embed
            + embed.square().sum(axis=0, keepdims=True)
        )
        embed_ind = dist.argmax(axis=-1)
        return embed_ind

    def encode(self, hidden_states):
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape((-1, shape[-1]))
        embed_ind = self.quantize(hidden_states)
        embed_ind = embed_ind.reshape(*shape[:-1])
        return embed_ind

    def decode(self, embed_ind):
        return self.embed[embed_ind]


class EncodecVectorQuantization(nn.Module):
    """
    Vector quantization implementation. Currently supports only euclidean distance.
    """

    def __init__(self, config):
        super().__init__()
        self.codebook = EncodecEuclideanCodebook(config)

    def encode(self, hidden_states):
        return self.codebook.encode(hidden_states)

    def decode(self, embed_ind):
        return self.codebook.decode(embed_ind)


class EncodecResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer."""

    def __init__(self, config):
        super().__init__()
        self.codebook_size = config.codebook_size

        hop_length = np.prod(config.upsampling_ratios)
        self.frame_rate = math.ceil(config.sampling_rate / hop_length)
        self.num_quantizers = int(
            1000 * config.target_bandwidths[-1] // (self.frame_rate * 10)
        )
        self.layers = [
            EncodecVectorQuantization(config) for _ in range(self.num_quantizers)
        ]

    def get_num_quantizers_for_bandwidth(
        self, bandwidth: Optional[float] = None
    ) -> int:
        """Return num_quantizers based on specified target bandwidth."""
        bw_per_q = math.log2(self.codebook_size) * self.frame_rate
        num_quantizers = self.num_quantizers
        if bandwidth is not None and bandwidth > 0.0:
            num_quantizers = int(max(1, math.floor(bandwidth * 1000 / bw_per_q)))
        return num_quantizers

    def encode(
        self, embeddings: mx.array, bandwidth: Optional[float] = None
    ) -> mx.array:
        """
        Encode a given input array with the specified frame rate at the given
        bandwidth. The RVQ encode method sets the appropriate number of
        quantizers to use and returns indices for each quantizer.
        """
        num_quantizers = self.get_num_quantizers_for_bandwidth(bandwidth)
        residual = embeddings
        all_indices = []
        for layer in self.layers[:num_quantizers]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = mx.stack(all_indices, axis=1)
        return out_indices

    def decode(self, codes: mx.array) -> mx.array:
        """Decode the given codes to the quantized representation."""
        quantized_out = None
        for i, indices in enumerate(codes.split(codes.shape[1], axis=1)):
            layer = self.layers[i]
            quantized = layer.decode(indices.squeeze(1))
            if quantized_out is None:
                quantized_out = quantized
            else:
                quantized_out = quantized + quantized_out
        return quantized_out


class Encodec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = EncodecEncoder(self.config)
        self.decoder = EncodecDecoder(self.config)
        self.quantizer = EncodecResidualVectorQuantizer(self.config)

    def _encode_frame(
        self, input_values: mx.array, bandwidth: float, padding_mask: mx.array
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Encodes the given input using the underlying VQVAE.
        """
        length = input_values.shape[1]
        duration = length / self.config.sampling_rate

        if (
            self.config.chunk_length_s is not None
            and duration > 1e-5 + self.config.chunk_length_s
        ):
            raise RuntimeError(
                f"Duration of frame ({duration}) is longer than chunk {self.config.chunk_length_s}"
            )

        scale = None
        if self.config.normalize:
            # if the padding is non zero
            input_values = input_values * padding_mask[..., None]
            mono = mx.sum(input_values, axis=2, keepdims=True) / input_values.shape[2]
            scale = mono.square().mean(axis=1, keepdims=True).sqrt() + 1e-8
            input_values = input_values / scale

        embeddings = self.encoder(input_values)
        codes = self.quantizer.encode(embeddings, bandwidth)
        return codes, scale

    def encode(
        self,
        input_values: mx.array,
        padding_mask: mx.array = None,
        bandwidth: Optional[float] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (mx.array): The input audio waveform with shape
                ``(batch_size, channels, sequence_length)``.
            padding_mask (mx.array): Padding mask used to pad the ``input_values``.
            bandwidth (float, optional): The target bandwidth. Must be one of
                ``config.target_bandwidths``. If ``None``, uses the smallest
                possible bandwidth. bandwidth is represented as a thousandth of
                what it is, e.g. 6kbps bandwidth is represented as bandwidth == 6.0

        Returns:
            A list of frames containing the discrete encoded codes for the
            input audio waveform, along with rescaling factors for each chunk
            when ``config.normalize==True``. Each frame is a tuple ``(codebook,
            scale)``, with ``codebook`` of shape ``(batch_size, num_codebooks,
            frames)``.
        """

        if bandwidth is None:
            bandwidth = self.config.target_bandwidths[0]
        if bandwidth not in self.config.target_bandwidths:
            raise ValueError(
                f"This model doesn't support the bandwidth {bandwidth}. Select one of {self.config.target_bandwidths}."
            )

        _, input_length, channels = input_values.shape

        if channels < 1 or channels > 2:
            raise ValueError(
                f"Number of audio channels must be 1 or 2, but got {channels}"
            )

        chunk_length = self.chunk_length
        if chunk_length is None:
            chunk_length = input_length
            stride = input_length
        else:
            stride = self.chunk_stride

        if padding_mask is None:
            padding_mask = mx.ones(input_values.shape[:2], dtype=mx.bool_)
        encoded_frames = []
        scales = []

        step = chunk_length - stride
        if (input_length % stride) != step:
            raise ValueError(
                "The input length is not properly padded for batched chunked encoding. Make sure to pad the input correctly."
            )

        for offset in range(0, input_length - step, stride):
            mask = padding_mask[:, offset : offset + chunk_length].astype(mx.bool_)
            frame = input_values[:, offset : offset + chunk_length]
            encoded_frame, scale = self._encode_frame(frame, bandwidth, mask)
            encoded_frames.append(encoded_frame)
            scales.append(scale)

        encoded_frames = mx.stack(encoded_frames)

        return (encoded_frames, scales)

    @staticmethod
    def _linear_overlap_add(frames: List[mx.array], stride: int):
        if len(frames) == 0:
            raise ValueError("`frames` cannot be an empty list.")

        dtype = frames[0].dtype
        N, frame_length, C = frames[0].shape
        total_size = stride * (len(frames) - 1) + frames[-1].shape[1]

        time_vec = mx.linspace(0, 1, frame_length + 2, dtype=dtype)[1:-1]
        weight = 0.5 - (time_vec - 0.5).abs()

        weight = weight[:, None]
        sum_weight = mx.zeros((total_size, 1), dtype=dtype)
        out = mx.zeros((N, total_size, C), dtype=dtype)
        offset = 0

        for frame in frames:
            frame_length = frame.shape[1]
            out[:, offset : offset + frame_length] += weight[:frame_length] * frame
            sum_weight[offset : offset + frame_length] += weight[:frame_length]
            offset += stride

        return out / sum_weight

    def _decode_frame(
        self, codes: mx.array, scale: Optional[mx.array] = None
    ) -> mx.array:
        embeddings = self.quantizer.decode(codes)
        outputs = self.decoder(embeddings)
        if scale is not None:
            outputs = outputs * scale
        return outputs

    @property
    def channels(self):
        return self.config.audio_channels

    @property
    def sampling_rate(self):
        return self.config.sampling_rate

    @property
    def chunk_length(self):
        if self.config.chunk_length_s is None:
            return None
        else:
            return int(self.config.chunk_length_s * self.config.sampling_rate)

    @property
    def chunk_stride(self):
        if self.config.chunk_length_s is None or self.config.overlap is None:
            return None
        else:
            return max(1, int((1.0 - self.config.overlap) * self.chunk_length))

    @classmethod
    def from_pretrained(cls, path_or_repo: str):
        """
        Load the model and audo preprocessor.
        """
        path = Path(path_or_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_repo,
                    allow_patterns=["*.json", "*.safetensors", "*.model"],
                )
            )

        with open(path / "config.json", "r") as f:
            config = json.load(f)

        filtered_config = filter_dataclass_fields(config, EncodecConfig)
        config = EncodecConfig(**filtered_config)
        model = cls(config)
        model.load_weights(str(path / "model.safetensors"))
        processor = functools.partial(
            preprocess_audio,
            sampling_rate=config.sampling_rate,
            chunk_length=model.chunk_length,
            chunk_stride=model.chunk_stride,
        )
        mx.eval(model)
        return model, processor

    def decode(
        self,
        audio_codes: mx.array,
        audio_scales: Union[mx.array, List[mx.array]],
        padding_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that
        case, any extra steps at the end should be trimmed.

        Args:
            audio_codes (mx.array): Discret code embeddings of shape
                ``(batch_size, nb_chunks, chunk_length)``.
            audio_scales (mx.array): Scaling factor for each input.
            padding_mask (mx.array): Padding mask.
        """
        chunk_length = self.chunk_length
        if chunk_length is None:
            if audio_codes.shape[1] != 1:
                raise ValueError(f"Expected one frame, got {len(audio_codes)}")
            audio_values = self._decode_frame(audio_codes[:, 0], audio_scales[0])
        else:
            decoded_frames = []

            for frame, scale in zip(audio_codes, audio_scales):
                frames = self._decode_frame(frame, scale)
                decoded_frames.append(frames)

            audio_values = self._linear_overlap_add(
                decoded_frames, self.chunk_stride or 1
            )

        # truncate based on padding mask
        if padding_mask is not None and padding_mask.shape[1] < audio_values.shape[1]:
            audio_values = audio_values[:, : padding_mask.shape[1]]
        return audio_values
