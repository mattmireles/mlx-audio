# Copyright © 2023 Apple Inc.

# Whisper speech-to-text model implementation for MLX.
# 
# This module provides the complete Whisper architecture including audio encoder,
# text decoder, and high-level transcription interface. The model follows the
# OpenAI Whisper paper architecture with MLX-specific optimizations.
# 
# Cross-file Dependencies:
# - `.audio`: Provides mel-spectrogram computation and audio preprocessing
# - `.decoding`: Implements beam search and language detection algorithms  
# - `.timing`: Handles word-level timestamp extraction using cross-attention
# - `.tokenizer`: Manages multilingual tokenization and vocabulary
# - `mlx_audio.utils`: Uses STFT and mel filterbank utilities for preprocessing
# 
# Used by:
# - `mlx_audio.stt.generate` for high-level transcription API
# - `mlx_audio.server` for real-time speech recognition serving
# - Applications requiring multilingual speech recognition

import base64
import gzip
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import tqdm
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from .audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from .decoding import DecodingOptions, DecodingResult
from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .timing import add_word_timestamps
from .tokenizer import LANGUAGES, get_tokenizer

# Timestamp formatting constants
MILLISECONDS_PER_SECOND = 1000
MILLISECONDS_PER_MINUTE = 60_000
MILLISECONDS_PER_HOUR = 3_600_000

# Vocabulary size thresholds for multilingual model detection
MULTILINGUAL_VOCAB_THRESHOLD = 51865  # Minimum vocab size for multilingual models
BASE_VOCAB_SIZE = 51765               # Base vocabulary size before language tokens

# Positional embedding constants
DEFAULT_MAX_TIMESCALE = 10000         # Maximum timescale for sinusoidal positional encoding

# Attention scaling constants
ATTENTION_SCALE_FACTOR = -0.25        # Scaling factor for query/key in attention (1/sqrt(sqrt(d_k)))
MLP_EXPANSION_FACTOR = 4              # Hidden dimension expansion in MLP blocks

# Transcription quality thresholds
DEFAULT_COMPRESSION_RATIO_THRESHOLD = 2.4    # Gzip compression ratio threshold for repetition detection
DEFAULT_LOGPROB_THRESHOLD = -1.0            # Average log probability threshold for quality
DEFAULT_NO_SPEECH_THRESHOLD = 0.6           # No-speech probability threshold
DEFAULT_TEMPERATURE_SCHEDULE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # Temperature fallback schedule

# Word timestamp anomaly detection thresholds
WORD_PROBABILITY_THRESHOLD = 0.15     # Minimum word probability for anomaly scoring
MIN_WORD_DURATION = 0.133            # Minimum reasonable word duration (seconds)
MAX_WORD_DURATION = 2.0              # Maximum reasonable word duration (seconds)
WORD_DURATION_PENALTY_FACTOR = 15    # Penalty factor for very short words
ANOMALY_SCORE_THRESHOLD = 3          # Threshold for marking segment as anomalous
ANOMALY_SCORE_PER_WORD_THRESHOLD = 0.01  # Per-word anomaly score threshold
MAX_ANOMALY_WORDS_CHECKED = 8        # Maximum words to check for anomalies

# Hallucination detection constants
MIN_SEGMENT_DISTANCE = 2.0           # Minimum time distance for segment boundaries (seconds)
HALLUCINATION_SKIP_OFFSET = 1        # Time offset when skipping hallucinations (seconds)

# High temperature prompt conditioning threshold
HIGH_TEMPERATURE_THRESHOLD = 0.5     # Temperature above which to reset prompt conditioning


def _format_timestamp(seconds: float):
    # Formats a timestamp in seconds to human-readable HH:MM:SS.mmm format.
    # 
    # Converts floating-point seconds to a standard timestamp string format
    # suitable for subtitle files (SRT, VTT) and transcription display.
    # Hours are omitted when zero for cleaner output.
    # 
    # Called by:
    # - `Model.generate()` for verbose transcription output
    # - External subtitle generation utilities
    # - Transcription result formatting
    # 
    # Args:
    #     seconds (float): Time in seconds, must be non-negative
    # 
    # Returns:
    #     str: Formatted timestamp string (e.g., "01:23:45.678" or "23:45.678")
    # 
    # Raises:
    #     AssertionError: If seconds is negative
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * MILLISECONDS_PER_SECOND)

    hours = milliseconds // MILLISECONDS_PER_HOUR
    milliseconds -= hours * MILLISECONDS_PER_HOUR

    minutes = milliseconds // MILLISECONDS_PER_MINUTE
    milliseconds -= minutes * MILLISECONDS_PER_MINUTE

    seconds = milliseconds // MILLISECONDS_PER_SECOND
    milliseconds -= seconds * MILLISECONDS_PER_SECOND

    hours_marker = f"{hours:02d}:" if hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _get_end(segments: List[dict]) -> Optional[float]:
    # Extracts the latest end timestamp from a list of transcription segments.
    # 
    # Searches through segments in reverse order to find the end timestamp of
    # the last word. Falls back to segment-level end timestamp if no word-level
    # timestamps are available. Used for timestamp-based seeking and progress tracking.
    # 
    # Called by:
    # - `Model.generate()` for tracking speech end times
    # - Word timestamp processing in transcription pipeline
    # - Hallucination detection for timing-based quality control
    # 
    # Args:
    #     segments (List[dict]): List of transcription segments, each containing
    #                           optional "words" array with "end" timestamps
    # 
    # Returns:
    #     Optional[float]: Latest end timestamp in seconds, or None if no segments
    return next(
        (w["end"] for s in reversed(segments) for w in reversed(s["words"])),
        segments[-1]["end"] if segments else None,
    )


@dataclass
class STTOutput:
    # Complete speech-to-text transcription result with metadata.
    # 
    # Standardized output format for all Whisper transcription operations,
    # containing the full transcribed text plus optional segmentation and
    # language detection results.
    # 
    # Used by:
    # - `Model.generate()` as primary return type
    # - `mlx_audio.stt.generate` for API responses
    # - Client applications consuming transcription results
    # 
    # Attributes:
    #     text (str): Complete transcribed text from the entire audio
    #     segments (List[dict]): Optional list of time-segmented transcription chunks,
    #                           each containing start/end times, text, tokens, and metadata
    #     language (str): Optional detected or specified language code (e.g., "en", "es")
    text: str
    segments: List[dict] = None
    language: str = None


@dataclass
class ModelDimensions:
    # Whisper model architecture configuration parameters.
    # 
    # Defines the complete architecture specification for a Whisper model,
    # including both audio encoder and text decoder dimensions. These parameters
    # determine model size, computational requirements, and capabilities.
    # 
    # Used by:
    # - `Model.__init__()` for architecture construction
    # - `Model.from_pretrained()` for loading model configurations
    # - Model size calculation and memory planning
    # 
    # Audio Encoder Parameters:
    #     n_mels (int): Number of mel-frequency channels (typically 80)
    #     n_audio_ctx (int): Audio context length in tokens (typically 1500)
    #     n_audio_state (int): Audio encoder hidden dimension
    #     n_audio_head (int): Number of attention heads in audio encoder
    #     n_audio_layer (int): Number of transformer layers in audio encoder
    # 
    # Text Decoder Parameters:
    #     n_vocab (int): Vocabulary size (50257 for English, 51865+ for multilingual)
    #     n_text_ctx (int): Text context length in tokens (typically 448)
    #     n_text_state (int): Text decoder hidden dimension
    #     n_text_head (int): Number of attention heads in text decoder
    #     n_text_layer (int): Number of transformer layers in text decoder
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


def sinusoids(length, channels, max_timescale=DEFAULT_MAX_TIMESCALE):
    # Generates sinusoidal positional embeddings for transformer models.
    # 
    # Creates position-dependent sinusoidal embeddings that allow the model to
    # understand sequence order without explicit position tokens. Uses alternating
    # sine and cosine functions with different frequencies for each dimension.
    # 
    # Based on "Attention Is All You Need" paper positional encoding scheme.
    # The frequency decreases exponentially across dimensions, allowing the model
    # to learn to attend by relative positions.
    # 
    # Called by:
    # - `AudioEncoder.__init__()` for audio sequence position encoding
    # - Any transformer component requiring positional information
    # 
    # Args:
    #     length (int): Sequence length (number of positions to encode)
    #     channels (int): Embedding dimension (must be even for sin/cos pairs)
    #     max_timescale (int): Maximum timescale for frequency range (default 10000)
    # 
    # Returns:
    #     mx.array: Positional embeddings of shape (length, channels)
    # 
    # Raises:
    #     AssertionError: If channels is not even
    assert channels % 2 == 0
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_timescale_increment * mx.arange(channels // 2))
    scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)


class MultiHeadAttention(nn.Module):
    # Multi-head self-attention mechanism for transformer architectures.
    # 
    # Implements scaled dot-product attention with multiple attention heads,
    # allowing the model to attend to different representation subspaces
    # simultaneously. Used in both audio encoder and text decoder.
    # 
    # Architecture follows "Attention Is All You Need" with MLX optimizations.
    # Key projections use no bias following common transformer practices.
    # 
    # Used by:
    # - `ResidualAttentionBlock` for both self-attention and cross-attention
    # - `AudioEncoder` blocks for audio sequence modeling
    # - `TextDecoder` blocks for causal text generation and cross-attention to audio
    # 
    # Args:
    #     n_state (int): Hidden dimension of the model (embedding size)
    #     n_head (int): Number of attention heads (n_state must be divisible by n_head)
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)        # Query projection with bias
        self.key = nn.Linear(n_state, n_state, bias=False)    # Key projection (no bias)
        self.value = nn.Linear(n_state, n_state)        # Value projection with bias
        self.out = nn.Linear(n_state, n_state)          # Output projection

    def __call__(
        self,
        x,
        xa=None,
        mask=None,
        kv_cache=None,
    ):
        # Forward pass for multi-head attention with optional cross-attention and caching.
        # 
        # Supports both self-attention (xa=None) and cross-attention (xa provided).
        # Implements key-value caching for efficient autoregressive generation.
        # 
        # Args:
        #     x (mx.array): Input queries of shape (batch, seq_len, n_state)
        #     xa (mx.array, optional): Cross-attention keys/values for encoder-decoder attention.
        #                              If None, performs self-attention using x.
        #     mask (mx.array, optional): Attention mask to prevent attending to certain positions
        #     kv_cache (tuple, optional): Cached (key, value) tensors for efficiency
        # 
        # Returns:
        #     tuple: (attention_output, updated_kv_cache, attention_weights)
        #            - attention_output (mx.array): Attended features of shape (batch, seq_len, n_state)
        #            - updated_kv_cache (tuple): Updated (key, value) cache for next step
        #            - attention_weights (mx.array): Raw attention scores for analysis
        q = self.query(x)

        if xa is None:
            # Self-attention: keys and values come from the same input
            k = self.key(x)
            v = self.value(x)
            if kv_cache is not None:
                # Concatenate with cached keys/values for autoregressive generation
                k = mx.concatenate([kv_cache[0], k], axis=1)
                v = mx.concatenate([kv_cache[1], v], axis=1)
        elif kv_cache is None:
            # Cross-attention without cache: keys and values from encoder
            k = self.key(xa)
            v = self.value(xa)
        else:
            # Cross-attention with cache: use cached encoder keys/values
            k, v = kv_cache

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), (k, v), qk

    def qkv_attention(self, q, k, v, mask=None):
        # Core scaled dot-product attention computation with multi-head support.
        # 
        # Implements the scaled dot-product attention mechanism:
        # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
        # 
        # Uses double square root scaling (-0.25 power) for numerical stability
        # as recommended in some transformer variants. Supports causal masking
        # for autoregressive generation.
        # 
        # Args:
        #     q (mx.array): Query tensor of shape (batch, seq_q, n_state)
        #     k (mx.array): Key tensor of shape (batch, seq_k, n_state)
        #     v (mx.array): Value tensor of shape (batch, seq_k, n_state)
        #     mask (mx.array, optional): Additive attention mask
        # 
        # Returns:
        #     tuple: (attention_output, attention_weights)
        #            - attention_output (mx.array): Weighted value aggregation
        #            - attention_weights (mx.array): Raw attention scores (pre-softmax)
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** ATTENTION_SCALE_FACTOR  # 1/sqrt(sqrt(d_k)) scaling
        
        # Reshape for multi-head attention: (batch, heads, seq, head_dim)
        q = q.reshape(*q.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3) * scale
        k = k.reshape(*k.shape[:2], self.n_head, -1).transpose(0, 2, 3, 1) * scale  # Transpose for matmul
        v = v.reshape(*v.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)

        # Compute attention scores
        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        # Apply softmax and compute weighted values
        w = mx.softmax(qk, axis=-1, precise=True)  # Use precise softmax for stability
        out = (w @ v).transpose(0, 2, 1, 3)        # Transpose back to (batch, seq, heads, head_dim)
        out = out.reshape(n_batch, n_ctx, n_state)  # Merge heads
        return out, qk


class ResidualAttentionBlock(nn.Module):
    # Transformer block with self-attention, optional cross-attention, and MLP.
    # 
    # Implements a standard transformer layer with pre-layer normalization,
    # residual connections, and optional encoder-decoder cross-attention.
    # The MLP uses GELU activation and 4x hidden dimension expansion.
    # 
    # Architecture follows GPT-style pre-norm design for better training stability.
    # Cross-attention is only present in decoder layers for encoder-decoder models.
    # 
    # Used by:
    # - `AudioEncoder` blocks (without cross-attention)
    # - `TextDecoder` blocks (with cross-attention to audio features)
    # 
    # Args:
    #     n_state (int): Hidden dimension of the model
    #     n_head (int): Number of attention heads
    #     cross_attention (bool): Whether to include cross-attention for encoder-decoder
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        # Self-attention components
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        # Optional cross-attention for encoder-decoder (decoder blocks only)
        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        # MLP with 4x expansion ratio (standard transformer design)
        n_mlp = n_state * MLP_EXPANSION_FACTOR
        self.mlp1 = nn.Linear(n_state, n_mlp)    # Expansion layer
        self.mlp2 = nn.Linear(n_mlp, n_state)    # Projection back to model dimension
        self.mlp_ln = nn.LayerNorm(n_state)      # Pre-MLP layer normalization

    def __call__(self, x, xa=None, mask=None, kv_cache=None):
        # Forward pass through transformer block with residual connections.
        # 
        # Applies self-attention, optional cross-attention, and MLP with residual
        # connections and pre-layer normalization. Returns updated cache for
        # efficient autoregressive generation.
        # 
        # Args:
        #     x (mx.array): Input tensor of shape (batch, seq_len, n_state)
        #     xa (mx.array, optional): Encoder features for cross-attention
        #     mask (mx.array, optional): Causal mask for self-attention
        #     kv_cache (tuple, optional): Cached (self_kv, cross_kv) for generation
        # 
        # Returns:
        #     tuple: (output, updated_cache, cross_attention_weights)
        #            - output (mx.array): Block output with same shape as input
        #            - updated_cache (tuple): Updated (self_kv, cross_kv) cache
        #            - cross_attention_weights: Cross-attention scores for analysis
        kv, cross_kv = kv_cache if kv_cache else (None, None)
        
        # Self-attention with residual connection
        y, kv, _ = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv)
        x += y
        
        # Optional cross-attention with residual connection
        cross_qk = None
        if self.cross_attn:
            y, cross_kv, cross_qk = self.cross_attn(
                self.cross_attn_ln(x), xa, kv_cache=cross_kv
            )
            x += y
            
        # MLP with GELU activation and residual connection
        x = x + self.mlp2(nn.gelu(self.mlp1(self.mlp_ln(x))))
        return x, (kv, cross_kv), cross_qk


class AudioEncoder(nn.Module):
    # Whisper audio encoder that transforms mel-spectrograms to contextual features.
    # 
    # Encodes mel-spectrogram audio features into contextualized representations
    # using convolutional preprocessing followed by transformer self-attention layers.
    # The architecture downsamples the time dimension by 2x with strided convolution.
    # 
    # Architecture:
    # 1. Two 1D convolutions with GELU activation (2x temporal downsampling)
    # 2. Positional embeddings added to conv features
    # 3. Stack of transformer blocks with self-attention
    # 4. Final layer normalization
    # 
    # Used by:
    # - `Model.encoder` as the audio processing frontend
    # - `Model.embed_audio()` for encoding audio features
    # - Cross-attention in `TextDecoder` for encoder-decoder attention
    # 
    # Args:
    #     n_mels (int): Number of mel-frequency channels (typically 80)
    #     n_ctx (int): Audio context length after conv downsampling (typically 1500)
    #     n_state (int): Hidden dimension for transformer layers
    #     n_head (int): Number of attention heads in each transformer layer
    #     n_layer (int): Number of transformer layers (6-32 depending on model size)
    #     dtype (mx.Dtype): Data type for computation (float16 for efficiency)
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()
        # Convolutional preprocessing: mel-spectrogram -> hidden features
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)        # Project to hidden dim
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)  # 2x downsample
        
        # Sinusoidal positional embeddings for sequence modeling
        self._positional_embedding = sinusoids(n_ctx, n_state).astype(dtype)

        # Transformer blocks for contextual modeling
        self.blocks = [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        self.ln_post = nn.LayerNorm(n_state)  # Final output normalization

    def __call__(self, x):
        # Encodes mel-spectrogram features into contextualized audio representations.
        # 
        # Processes input mel-spectrograms through convolutional preprocessing
        # and transformer layers to produce rich audio features for cross-attention.
        # 
        # Args:
        #     x (mx.array): Input mel-spectrogram of shape (batch, n_mels, time_steps)
        # 
        # Returns:
        #     mx.array: Encoded audio features of shape (batch, n_ctx, n_state)
        #              where n_ctx = time_steps // 2 due to stride-2 convolution
        # 
        # Raises:
        #     AssertionError: If input shape doesn't match expected audio context length
        
        # Convolutional preprocessing with GELU activation
        x = nn.gelu(self.conv1(x))  # (batch, n_mels, time) -> (batch, n_state, time)
        x = nn.gelu(self.conv2(x))  # (batch, n_state, time) -> (batch, n_state, time//2)
        
        # Validate shape matches positional embedding dimensions
        assert x.shape[1:] == self._positional_embedding.shape, "incorrect audio shape"
        
        # Add positional embeddings for sequence understanding
        x = x + self._positional_embedding

        # Process through transformer blocks (self-attention only)
        for block in self.blocks:
            x, _, _ = block(x)  # No cross-attention in encoder blocks

        # Final normalization
        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    # Whisper autoregressive text decoder with cross-attention to audio features.
    # 
    # Generates text tokens autoregressively using causal self-attention and
    # cross-attention to encoded audio features. Implements the decoder side
    # of the encoder-decoder transformer architecture.
    # 
    # Architecture:
    # 1. Token and positional embeddings
    # 2. Stack of transformer blocks with causal self-attention and cross-attention
    # 3. Layer normalization and output projection through embedding weights
    # 
    # Used by:
    # - `Model.decoder` for text generation
    # - `Model.logits()` for single forward pass
    # - `Model.forward_with_cross_qk()` for attention analysis
    # 
    # Args:
    #     n_vocab (int): Vocabulary size including special tokens
    #     n_ctx (int): Maximum text context length (typically 448)
    #     n_state (int): Hidden dimension for transformer layers
    #     n_head (int): Number of attention heads in each layer
    #     n_layer (int): Number of decoder transformer layers
    #     dtype (mx.Dtype): Data type for computation (float16 for efficiency)
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()

        # Embedding layers
        self.token_embedding = nn.Embedding(n_vocab, n_state)      # Token -> hidden
        self.positional_embedding = mx.zeros((n_ctx, n_state))     # Learned positional embeddings

        # Transformer decoder blocks with cross-attention enabled
        self.blocks = [
            ResidualAttentionBlock(n_state, n_head, cross_attention=True)
            for _ in range(n_layer)
        ]
        
        self.ln = nn.LayerNorm(n_state)  # Pre-output normalization
        
        # Causal attention mask for autoregressive generation
        self._mask = nn.MultiHeadAttention.create_additive_causal_mask(n_ctx).astype(
            dtype
        )

    def __call__(self, x, xa, kv_cache=None):
        # Decodes text tokens autoregressively with cross-attention to audio.
        # 
        # Generates text logits by processing input tokens through causal self-attention
        # and cross-attention to encoded audio features. Supports key-value caching
        # for efficient beam search and autoregressive generation.
        # 
        # The positional embeddings are offset-aware for incremental generation,
        # allowing efficient continuation from cached states.
        # 
        # Args:
        #     x (mx.array): Input token IDs of shape (batch_size, seq_len)
        #                   where seq_len <= n_ctx
        #     xa (mx.array): Encoded audio features from AudioEncoder
        #                    of shape (batch_size, n_audio_ctx, n_audio_state)
        #     kv_cache (list, optional): Cached key-value states for each layer
        # 
        # Returns:
        #     tuple: (logits, updated_kv_cache, cross_attention_weights)
        #            - logits (mx.array): Next-token logits of shape (batch, seq_len, n_vocab)
        #            - updated_kv_cache (list): Updated cache for each layer
        #            - cross_attention_weights (list): Cross-attention scores for analysis
        
        # Calculate positional embedding offset for incremental decoding
        offset = kv_cache[0][0][0].shape[1] if kv_cache else 0
        
        # Apply token and positional embeddings
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )

        # Initialize cache and cross-attention storage
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)
        cross_qk = [None] * len(self.blocks)
        
        # Process through transformer decoder blocks
        for e, block in enumerate(self.blocks):
            x, kv_cache[e], cross_qk[e] = block(
                x, xa, mask=self._mask, kv_cache=kv_cache[e]
            )

        # Apply final normalization and project to vocabulary
        x = self.ln(x)
        return self.token_embedding.as_linear(x), kv_cache, cross_qk


class Model(nn.Module):
    # Complete Whisper model with audio encoder, text decoder, and transcription interface.
    # 
    # Implements the full Whisper architecture as described in the OpenAI Whisper paper.
    # Combines AudioEncoder and TextDecoder with high-level transcription methods,
    # language detection, and word-level timestamp extraction.
    # 
    # The model supports both single-pass transcription and streaming generation
    # with configurable quality thresholds and fallback strategies.
    # 
    # Used by:
    # - `mlx_audio.stt.generate` for high-level transcription API
    # - `mlx_audio.server` for real-time speech recognition
    # - Applications requiring multilingual speech-to-text
    # 
    # Key Features:
    # - Multilingual support with automatic language detection
    # - Word-level timestamp extraction using cross-attention alignment
    # - Quality-based fallback with temperature scheduling
    # - Efficient caching for beam search and generation
    def __init__(self, dims: ModelDimensions, dtype: mx.Dtype = mx.float16):
        super().__init__()
        self.dims = dims
        self.dtype = dtype
        
        # Initialize encoder-decoder architecture
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            dtype,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            dtype,
        )
        
        # Initialize alignment heads for word timestamp extraction
        # Use the last half of decoder layers by default (better temporal alignment)
        all_heads = np.zeros(
            (self.dims.n_text_layer, self.dims.n_text_head), dtype=bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.alignment_heads = mx.array(np.asarray(all_heads.nonzero()).T)

    def set_alignment_heads(self, dump: Union[bytes, np.ndarray]):
        # Configures which attention heads to use for word timestamp extraction.
        # 
        # Sets the attention heads used for computing word-level timestamps via
        # cross-attention alignment analysis. Supports both direct numpy arrays
        # and compressed base85-encoded byte arrays for efficient storage.
        # 
        # Called by:
        # - Model loading code when alignment head data is available
        # - Applications requiring custom timestamp extraction strategies
        # - Research tools for attention head analysis
        # 
        # Args:
        #     dump (Union[bytes, np.ndarray]): Either a numpy array of head indices
        #           or base85-encoded compressed boolean mask of shape (n_text_layer, n_text_head)
        # 
        # Raises:
        #     ValueError: If dump is not a supported type
        if isinstance(dump, np.ndarray):
            self.alignment_heads = mx.array(dump)
        elif isinstance(dump, bytes):
            # Decompress and decode base85-encoded alignment head mask
            array = np.frombuffer(
                gzip.decompress(base64.b85decode(dump)), dtype=bool
            ).copy()
            mask = array.reshape(self.dims.n_text_layer, self.dims.n_text_head)
            self.alignment_heads = mx.array(np.asarray(mask.nonzero()).T)
        else:
            raise ValueError(
                f"Invalid type for `dump`: {type(dump)}. Expected a np.ndarray or base85-encoded bytes containing"
                " alignment_head information"
            )

    def embed_audio(self, mel):
        # Encodes mel-spectrogram to audio feature representations.
        # 
        # Convenience method that applies the audio encoder to convert
        # mel-spectrograms into contextual audio features for cross-attention.
        # 
        # Args:
        #     mel (mx.array): Mel-spectrogram of shape (batch, n_mels, time_steps)
        # 
        # Returns:
        #     mx.array: Encoded audio features of shape (batch, n_audio_ctx, n_audio_state)
        return self.encoder(mel)

    def logits(self, tokens, audio_features):
        # Computes next-token logits given tokens and audio context.
        # 
        # Single forward pass through the text decoder with pre-encoded audio features.
        # Used for non-autoregressive evaluation and analysis.
        # 
        # Args:
        #     tokens (mx.array): Input token sequence of shape (batch, seq_len)
        #     audio_features (mx.array): Encoded audio of shape (batch, n_audio_ctx, n_audio_state)
        # 
        # Returns:
        #     mx.array: Logits over vocabulary of shape (batch, seq_len, n_vocab)
        return self.decoder(tokens, audio_features)[0]

    def forward_with_cross_qk(self, mel, tokens):
        # Forward pass returning both logits and cross-attention weights.
        # 
        # Computes model output while capturing cross-attention patterns between
        # text tokens and audio features. Used for word timestamp extraction
        # and attention visualization.
        # 
        # Args:
        #     mel (mx.array): Input mel-spectrogram
        #     tokens (mx.array): Input token sequence
        # 
        # Returns:
        #     tuple: (logits, cross_attention_weights)
        #            - logits: Next-token predictions
        #            - cross_attention_weights: Cross-attention scores for alignment
        logits, _, cross_qk = self.decoder(tokens, self.encoder(mel))
        return logits, cross_qk

    def __call__(self, mel, tokens):
        # Default forward pass for training and inference.
        # 
        # Standard encoder-decoder forward pass returning next-token logits.
        # 
        # Args:
        #     mel (mx.array): Input mel-spectrogram
        #     tokens (mx.array): Input token sequence
        # 
        # Returns:
        #     mx.array: Next-token logits over vocabulary
        return self.decoder(tokens, self.encoder(mel))[0]

    @property
    def is_multilingual(self):
        # Determines if this is a multilingual model based on vocabulary size.
        # 
        # Multilingual Whisper models have larger vocabularies to accommodate
        # language-specific tokens and multilingual text generation.
        # 
        # Returns:
        #     bool: True if model supports multiple languages
        return self.dims.n_vocab >= MULTILINGUAL_VOCAB_THRESHOLD

    @property
    def num_languages(self):
        # Calculates the number of languages supported by this model.
        # 
        # Derives language count from vocabulary size by subtracting base tokens
        # and accounting for the multilingual model structure.
        # 
        # Returns:
        #     int: Number of supported languages (0 for English-only models)
        return self.dims.n_vocab - BASE_VOCAB_SIZE - int(self.is_multilingual)

    detect_language = detect_language_function
    decode = decode_function

    @classmethod
    def from_pretrained(
        cls,
        path_or_hf_repo: str = "mlx-community/whisper-tiny",
        dtype: mx.Dtype = mx.float16,
    ) -> "Whisper":
        model_path = Path(path_or_hf_repo)
        if not model_path.exists():
            model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

        with open(str(model_path / "config.json"), "r") as f:
            config = json.loads(f.read())
            config.pop("model_type", None)
            quantization = config.pop("quantization", None)

        model_args = ModelDimensions(**config)

        wf = model_path / "weights.safetensors"
        if not wf.exists():
            wf = model_path / "weights.npz"
        weights = mx.load(str(wf))

        model = Model(model_args, dtype)

        if quantization is not None:
            class_predicate = (
                lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
                and f"{p}.scales" in weights
            )
            nn.quantize(model, **quantization, class_predicate=class_predicate)

        weights = tree_unflatten(list(weights.items()))
        model.update(weights)
        mx.eval(model.parameters())
        return model

    def generate(
        self,
        audio: Union[str, np.ndarray, mx.array],
        *,
        verbose: Optional[bool] = None,
        generation_stream: bool = False,
        temperature: Union[float, Tuple[float, ...]] = DEFAULT_TEMPERATURE_SCHEDULE,
        compression_ratio_threshold: Optional[float] = DEFAULT_COMPRESSION_RATIO_THRESHOLD,
        logprob_threshold: Optional[float] = DEFAULT_LOGPROB_THRESHOLD,
        no_speech_threshold: Optional[float] = DEFAULT_NO_SPEECH_THRESHOLD,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        clip_timestamps: Union[str, List[float]] = "0",
        hallucination_silence_threshold: Optional[float] = None,
        **decode_options,
    ):
        # High-level speech-to-text transcription with quality controls and word timestamps.
        # 
        # Transcribes audio using Whisper with configurable quality thresholds, temperature
        # fallback, and optional word-level timestamps. Implements hallucination detection
        # and progressive quality control for robust transcription.
        # 
        # Called by:
        # - `mlx_audio.stt.generate` for high-level transcription API
        # - Applications requiring robust speech-to-text with quality control
        # - Batch processing pipelines
        # 
        """
        Transcribe an audio file using Whisper

        Parameters
        ----------
        audio: Union[str, np.ndarray, mx.array]
            The path to the audio file to open, or the audio waveform

        verbose: bool
            Whether to display the text being decoded to the console. If True, displays all the details,
            If False, displays minimal details. If None, does not display anything

        temperature: Union[float, Tuple[float, ...]]
            Temperature for sampling. It can be a tuple of temperatures, which will be successively used
            upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

        compression_ratio_threshold: float
            If the gzip compression ratio is above this value, treat as failed

        logprob_threshold: float
            If the average log probability over sampled tokens is below this value, treat as failed

        no_speech_threshold: float
            If the no_speech probability is higher than this value AND the average log probability
            over sampled tokens is below `logprob_threshold`, consider the segment as silent

        condition_on_previous_text: bool
            if True, the previous output of the model is provided as a prompt for the next window;
            disabling may make the text inconsistent across windows, but the model becomes less prone to
            getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

        word_timestamps: bool
            Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
            and include the timestamps for each word in each segment.

        prepend_punctuations: str
            If word_timestamps is True, merge these punctuation symbols with the next word

        append_punctuations: str
            If word_timestamps is True, merge these punctuation symbols with the previous word

        initial_prompt: Optional[str]
            Optional text to provide as a prompt for the first window. This can be used to provide, or
            "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
            to make it more likely to predict those word correctly.

        decode_options: dict
            Keyword arguments to construct `DecodingOptions` instances

        clip_timestamps: Union[str, List[float]]
            Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process.
            The last end timestamp defaults to the end of the file.

        hallucination_silence_threshold: Optional[float]
            When word_timestamps is True, skip silent periods longer than this threshold (in seconds)
            when a possible hallucination is detected

        Returns
        -------
        A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
        the spoken language ("language"), which is detected when `decode_options["language"]` is None.
        """

        decode_options.pop("max_tokens", None)
        decode_options.pop("generation_stream", None)

        # Pad 30-seconds of silence to the input audio, for slicing
        mel = log_mel_spectrogram(audio, n_mels=self.dims.n_mels, padding=N_SAMPLES)
        content_frames = mel.shape[-2] - N_FRAMES
        content_duration = float(content_frames * HOP_LENGTH / SAMPLE_RATE)

        if verbose:
            system_encoding = sys.getdefaultencoding()
            if system_encoding != "utf-8":
                make_safe = lambda x: x.encode(
                    system_encoding, errors="replace"
                ).decode(system_encoding)
            else:
                make_safe = lambda x: x

        if decode_options.get("language", None) is None:
            if not self.is_multilingual:
                decode_options["language"] = "en"
            else:
                if verbose:
                    print(
                        "Detecting language using up to the first 30 seconds. "
                        "Use the `language` decoding option to specify the language"
                    )
                mel_segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(self.dtype)
                _, probs = self.detect_language(mel_segment)
                decode_options["language"] = max(probs, key=probs.get)
                if verbose is not None:
                    print(
                        f"Detected language: {LANGUAGES[decode_options['language']].title()}"
                    )

        language: str = decode_options["language"]
        task: str = decode_options.get("task", "transcribe")
        tokenizer = get_tokenizer(
            self.is_multilingual,
            num_languages=self.num_languages,
            language=language,
            task=task,
        )

        if isinstance(clip_timestamps, str):
            clip_timestamps = [
                float(ts)
                for ts in (clip_timestamps.split(",") if clip_timestamps else [])
            ]
        seek_points: List[int] = [
            round(ts * FRAMES_PER_SECOND) for ts in clip_timestamps
        ]
        if len(seek_points) == 0:
            seek_points.append(0)
        if len(seek_points) % 2 == 1:
            seek_points.append(content_frames)
        else:
            seek_points[-1] = min(content_frames, seek_points[-1])
        seek_clips: List[Tuple[int, int]] = list(
            zip(seek_points[::2], seek_points[1::2])
        )

        punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"

        if word_timestamps and task == "translate":
            warnings.warn("Word-level timestamps on translations may not be reliable.")

        def decode_with_fallback(segment: mx.array) -> DecodingResult:
            temperatures = (
                [temperature] if isinstance(temperature, (int, float)) else temperature
            )
            decode_result = None

            for t in temperatures:
                kwargs = {**decode_options}
                if t > 0:
                    # disable beam_size and patience when t > 0
                    kwargs.pop("beam_size", None)
                    kwargs.pop("patience", None)
                else:
                    # disable best_of when t == 0
                    kwargs.pop("best_of", None)

                options = DecodingOptions(**kwargs, temperature=t)
                decode_result = self.decode(segment, options)

                needs_fallback = False
                if (
                    compression_ratio_threshold is not None
                    and decode_result.compression_ratio > compression_ratio_threshold
                ):
                    needs_fallback = True  # too repetitive
                if (
                    logprob_threshold is not None
                    and decode_result.avg_logprob < logprob_threshold
                ):
                    needs_fallback = True  # average log probability is too low
                if (
                    no_speech_threshold is not None
                    and decode_result.no_speech_prob > no_speech_threshold
                ):
                    needs_fallback = False  # silence
                if not needs_fallback:
                    break

            return decode_result

        clip_idx = 0
        seek = seek_clips[clip_idx][0]
        input_stride = (
            N_FRAMES // self.dims.n_audio_ctx
        )  # mel frames per output token: 2
        time_precision = (
            input_stride * HOP_LENGTH / SAMPLE_RATE
        )  # time per output token: 0.02 (seconds)
        all_tokens = []
        all_segments = []
        prompt_reset_since = 0

        if initial_prompt is not None:
            initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
            all_tokens.extend(initial_prompt_tokens)
        else:
            initial_prompt_tokens = []

        def new_segment(
            *, start: float, end: float, tokens: mx.array, result: DecodingResult
        ):
            tokens = tokens.tolist()
            text_tokens = [token for token in tokens if token < tokenizer.eot]
            return {
                "seek": seek,
                "start": start,
                "end": end,
                "text": tokenizer.decode(text_tokens),
                "tokens": tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
            }

        # show the progress bar when verbose is False (if True, transcribed text will be printed)
        with tqdm.tqdm(
            total=content_frames, unit="frames", disable=verbose is not False
        ) as pbar:
            last_speech_timestamp = 0.0
            for seek_clip_start, seek_clip_end in seek_clips:
                while seek < seek_clip_end:
                    time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
                    window_end_time = float(
                        (seek + N_FRAMES) * HOP_LENGTH / SAMPLE_RATE
                    )
                    segment_size = min(
                        N_FRAMES, content_frames - seek, seek_clip_end - seek
                    )
                    mel_segment = mel[seek : seek + segment_size]
                    segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
                    mel_segment = pad_or_trim(mel_segment, N_FRAMES, axis=-2).astype(
                        self.dtype
                    )

                    decode_options["prompt"] = all_tokens[prompt_reset_since:]
                    result: DecodingResult = decode_with_fallback(mel_segment)

                    tokens = np.array(result.tokens)

                    if no_speech_threshold is not None:
                        # no voice activity check
                        should_skip = result.no_speech_prob > no_speech_threshold
                        if (
                            logprob_threshold is not None
                            and result.avg_logprob > logprob_threshold
                        ):
                            # don't skip if the logprob is high enough, despite the no_speech_prob
                            should_skip = False

                        if should_skip:
                            seek += segment_size  # fast-forward to the next segment boundary
                            continue

                    previous_seek = seek
                    current_segments = []

                    # anomalous words are very long/short/improbable
                    def word_anomaly_score(word: dict) -> float:
                        probability = word.get("probability", 0.0)
                        duration = word["end"] - word["start"]
                        score = 0.0
                        if probability < 0.15:
                            score += 1.0
                        if duration < 0.133:
                            score += (0.133 - duration) * 15
                        if duration > 2.0:
                            score += duration - 2.0
                        return score

                    def is_segment_anomaly(segment: Optional[dict]) -> bool:
                        if segment is None or not segment["words"]:
                            return False
                        words = [
                            w for w in segment["words"] if w["word"] not in punctuation
                        ]
                        words = words[:8]
                        score = sum(word_anomaly_score(w) for w in words)
                        return score >= 3 or score + 0.01 >= len(words)

                    def next_words_segment(segments: List[dict]) -> Optional[dict]:
                        return next((s for s in segments if s["words"]), None)

                    timestamp_tokens = tokens >= tokenizer.timestamp_begin
                    single_timestamp_ending = timestamp_tokens[-2:].tolist() == [
                        False,
                        True,
                    ]

                    consecutive = np.where(
                        np.logical_and(timestamp_tokens[:-1], timestamp_tokens[1:])
                    )[0]
                    consecutive += 1
                    if len(consecutive) > 0:
                        # if the output contains two consecutive timestamp tokens
                        slices = consecutive.tolist()
                        if single_timestamp_ending:
                            slices.append(len(tokens))

                        last_slice = 0
                        for current_slice in slices:
                            sliced_tokens = tokens[last_slice:current_slice]
                            start_timestamp_pos = (
                                sliced_tokens[0].item() - tokenizer.timestamp_begin
                            )
                            end_timestamp_pos = (
                                sliced_tokens[-1].item() - tokenizer.timestamp_begin
                            )
                            current_segments.append(
                                new_segment(
                                    start=time_offset
                                    + start_timestamp_pos * time_precision,
                                    end=time_offset
                                    + end_timestamp_pos * time_precision,
                                    tokens=sliced_tokens,
                                    result=result,
                                )
                            )
                            last_slice = current_slice

                        if single_timestamp_ending:
                            # single timestamp at the end means no speech after the last timestamp.
                            seek += segment_size
                        else:
                            # otherwise, ignore the unfinished segment and seek to the last timestamp
                            last_timestamp_pos = (
                                tokens[last_slice - 1].item()
                                - tokenizer.timestamp_begin
                            )
                            seek += last_timestamp_pos * input_stride
                    else:
                        duration = segment_duration
                        timestamps = tokens[timestamp_tokens.nonzero()[0]]
                        if (
                            len(timestamps) > 0
                            and timestamps[-1].item() != tokenizer.timestamp_begin
                        ):
                            # no consecutive timestamps but it has a timestamp; use the last one.
                            last_timestamp_pos = (
                                timestamps[-1].item() - tokenizer.timestamp_begin
                            )
                            duration = last_timestamp_pos * time_precision

                        current_segments.append(
                            new_segment(
                                start=time_offset,
                                end=time_offset + duration,
                                tokens=tokens,
                                result=result,
                            )
                        )
                        seek += segment_size

                    if word_timestamps:
                        add_word_timestamps(
                            segments=current_segments,
                            model=self,
                            tokenizer=tokenizer,
                            mel=mel_segment,
                            num_frames=segment_size,
                            prepend_punctuations=prepend_punctuations,
                            append_punctuations=append_punctuations,
                            last_speech_timestamp=last_speech_timestamp,
                        )

                        if not single_timestamp_ending:
                            last_word_end = _get_end(current_segments)
                            if (
                                last_word_end is not None
                                and last_word_end > time_offset
                            ):
                                seek = round(last_word_end * FRAMES_PER_SECOND)

                        # skip silence before possible hallucinations
                        if hallucination_silence_threshold is not None:
                            threshold = hallucination_silence_threshold
                            if not single_timestamp_ending:
                                last_word_end = _get_end(current_segments)
                                if (
                                    last_word_end is not None
                                    and last_word_end > time_offset
                                ):
                                    remaining_duration = window_end_time - last_word_end
                                    if remaining_duration > threshold:
                                        seek = round(last_word_end * FRAMES_PER_SECOND)
                                    else:
                                        seek = previous_seek + segment_size

                            # if first segment might be a hallucination, skip leading silence
                            first_segment = next_words_segment(current_segments)
                            if first_segment is not None and is_segment_anomaly(
                                first_segment
                            ):
                                gap = first_segment["start"] - time_offset
                                if gap > threshold:
                                    seek = previous_seek + round(
                                        gap * FRAMES_PER_SECOND
                                    )
                                    continue

                            # skip silence before any possible hallucination that is surrounded
                            # by silence or more hallucinations
                            hal_last_end = last_speech_timestamp
                            for si in range(len(current_segments)):
                                segment = current_segments[si]
                                if not segment["words"]:
                                    continue
                                if is_segment_anomaly(segment):
                                    next_segment = next_words_segment(
                                        current_segments[si + 1 :]
                                    )
                                    if next_segment is not None:
                                        hal_next_start = next_segment["words"][0][
                                            "start"
                                        ]
                                    else:
                                        hal_next_start = time_offset + segment_duration
                                    silence_before = (
                                        segment["start"] - hal_last_end > threshold
                                        or segment["start"] < threshold
                                        or segment["start"] - time_offset < 2.0
                                    )
                                    silence_after = (
                                        hal_next_start - segment["end"] > threshold
                                        or is_segment_anomaly(next_segment)
                                        or window_end_time - segment["end"] < 2.0
                                    )
                                    if silence_before and silence_after:
                                        seek = round(
                                            max(time_offset + 1, segment["start"])
                                            * FRAMES_PER_SECOND
                                        )
                                        if (
                                            content_duration - segment["end"]
                                            < threshold
                                        ):
                                            seek = content_frames
                                        current_segments[si:] = []
                                        break
                                hal_last_end = segment["end"]

                        last_word_end = _get_end(current_segments)
                        if last_word_end is not None:
                            last_speech_timestamp = last_word_end

                    if verbose:
                        for segment in current_segments:
                            start, end, text = (
                                segment["start"],
                                segment["end"],
                                segment["text"],
                            )
                            line = f"[{_format_timestamp(start)} --> {_format_timestamp(end)}] {text}"
                            print(make_safe(line))

                    # if a segment is instantaneous or does not contain text, clear it
                    for i, segment in enumerate(current_segments):
                        if (
                            segment["start"] == segment["end"]
                            or segment["text"].strip() == ""
                        ):
                            segment["text"] = ""
                            segment["tokens"] = []
                            segment["words"] = []

                    all_segments.extend(
                        [
                            {"id": i, **segment}
                            for i, segment in enumerate(
                                current_segments, start=len(all_segments)
                            )
                        ]
                    )
                    all_tokens.extend(
                        [
                            token
                            for segment in current_segments
                            for token in segment["tokens"]
                        ]
                    )

                    if not condition_on_previous_text or result.temperature > HIGH_TEMPERATURE_THRESHOLD:
                        # do not feed the prompt tokens if a high temperature was used
                        prompt_reset_since = len(all_tokens)

                    # update progress bar
                    pbar.update(min(content_frames, seek) - previous_seek)

        # Clear cache after each segment to avoid memory leaks
        mx.clear_cache()

        if verbose:
            print("\n\033[94mSegments:\033[0m")
            if hasattr(all_segments, "segments"):
                print(all_segments)
            elif hasattr(all_segments, "tokens"):
                print(all_segments.tokens)
            else:
                print(all_segments)

        return STTOutput(
            text=tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]),
            segments=all_segments,
            language=language,
        )
