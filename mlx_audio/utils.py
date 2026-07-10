# Core audio processing utilities for MLX-Audio.
# 
# This module provides fundamental signal processing functions used throughout
# the MLX-Audio library, including window functions, STFT/ISTFT transforms,
# and mel-scale filterbank generation.
# 
# Used by:
# - `mlx_audio.stt.models.whisper.audio` for spectrogram computation
# - `mlx_audio.codec.models.*` for audio encoding/decoding preprocessing
# - `mlx_audio.tts.models.*` for audio feature extraction
# 
# All functions are optimized for MLX array operations and support batching.

import math
from functools import lru_cache
from typing import Optional

import mlx.core as mx

# Window function coefficients - following DSP literature standards
HANNING_AMPLITUDE_FACTOR = 0.5  # Standard Hanning window amplitude coefficient
HAMMING_ALPHA = 0.54            # Hamming window alpha coefficient (industry standard)
HAMMING_BETA = 0.46             # Hamming window beta coefficient (1 - alpha)
BLACKMAN_A0 = 0.42              # Blackman window a0 coefficient
BLACKMAN_A1 = 0.5               # Blackman window a1 coefficient
BLACKMAN_A2 = 0.08              # Blackman window a2 coefficient

# STFT default parameters - optimized for speech processing
DEFAULT_N_FFT = 800             # Default FFT size for speech analysis (25ms at 32kHz)
HOP_LENGTH_DIVISOR = 4           # Standard 75% overlap (hop = n_fft // 4)

# Mel scale conversion constants - HTK and Slaney scale standards
HTK_MEL_SCALE_FACTOR = 2595.0    # HTK mel scale conversion factor
HTK_FREQ_REFERENCE = 700.0       # HTK frequency reference point (Hz)
SLANEY_FREQ_STEP = 200.0 / 3     # Slaney linear frequency step
SLANEY_LOG_THRESHOLD = 1000.0    # Slaney logarithmic scale threshold (Hz)
SLANEY_LOG_STEP_FACTOR = 6.4     # Slaney logarithmic step factor
SLANEY_LOG_STEP_DIVISOR = 27.0   # Slaney logarithmic step normalization
SLANEY_NORM_FACTOR = 2.0         # Slaney normalization amplitude factor

# STFT/ISTFT processing constants
SAMPLE_INDEX_OFFSET = 1           # Index offset for signal boundary padding
FREQUENCY_BIN_MULTIPLIER = 2      # RFFT frequency bins = (n_fft // 2) + 1
NYQUIST_DIVISOR = 2               # Nyquist frequency = sample_rate / 2
WINDOW_BOUNDARY_OFFSET = 1        # Window boundary offset for reconstruction
PADDING_SIZE_DIVISOR = 2          # Center padding = n_fft // 2
MEL_FILTER_EDGE_OFFSET = 2        # Offset for mel filter edge calculation

# Common window functions for audio analysis


@lru_cache(maxsize=None)
def hanning(size):
    # Generates a Hanning (raised cosine) window for audio analysis.
    # 
    # The Hanning window is the most commonly used window function in audio
    # processing due to its good frequency resolution and low spectral leakage.
    # Formula: w[n] = 0.5 * (1 - cos(2*pi * n / (N-1)))
    # 
    # Called by:
    # - `stft()` when window="hann" or "hanning" (default)
    # - `istft()` for reconstruction windowing
    # - Various TTS models for spectral analysis
    # 
    # Args:
    #     size (int): Length of the window in samples. Must be positive.
    # 
    # Returns:
    #     mx.array: Hanning window coefficients of shape (size,)
    # 
    # Performance: Cached for repeated calls with same size.
    return mx.array(
        [HANNING_AMPLITUDE_FACTOR * (1 - math.cos(2 * math.pi * n / (size - 1))) for n in range(size)]
    )


@lru_cache(maxsize=None)
def hamming(size):
    # Generates a Hamming window for audio analysis with reduced spectral leakage.
    # 
    # The Hamming window provides better stopband attenuation than Hanning
    # at the cost of slightly wider mainlobe. Commonly used in speech recognition.
    # Formula: w[n] = 0.54 - 0.46 * cos(2*pi * n / (N-1))
    # 
    # Called by:
    # - `stft()` when window="hamming"
    # - `istft()` for reconstruction windowing
    # - Speech recognition models requiring low sidelobe levels
    # 
    # Args:
    #     size (int): Length of the window in samples. Must be positive.
    # 
    # Returns:
    #     mx.array: Hamming window coefficients of shape (size,)
    # 
    # Performance: Cached for repeated calls with same size.
    return mx.array(
        [HAMMING_ALPHA - HAMMING_BETA * math.cos(2 * math.pi * n / (size - 1)) for n in range(size)]
    )


@lru_cache(maxsize=None)
def blackman(size):
    # Generates a Blackman window for high-precision audio analysis.
    # 
    # The Blackman window offers excellent stopband attenuation (better than 70 decibels) making
    # it ideal for high-quality spectral analysis at the cost of wider mainlobe.
    # Formula: w[n] = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
    # 
    # Called by:
    # - `stft()` when window="blackman"
    # - High-quality audio codec preprocessing
    # - Precision measurement applications
    # 
    # Args:
    #     size (int): Length of the window in samples. Must be positive.
    # 
    # Returns:
    #     mx.array: Blackman window coefficients of shape (size,)
    # 
    # Performance: Cached for repeated calls with same size.
    return mx.array(
        [
            BLACKMAN_A0
            - BLACKMAN_A1 * math.cos(2 * math.pi * n / (size - 1))
            + BLACKMAN_A2 * math.cos(4 * math.pi * n / (size - 1))
            for n in range(size)
        ]
    )


@lru_cache(maxsize=None)
def bartlett(size):
    # Generates a Bartlett (triangular) window for basic audio windowing.
    # 
    # The Bartlett window is the simplest window function, linearly decreasing
    # from center to edges. Provides moderate spectral characteristics.
    # Formula: w[n] = 1 - 2 * |n - (N-1)/2| / (N-1)
    # 
    # Called by:
    # - `stft()` when window="bartlett"
    # - Simple audio processing applications
    # - Educational and prototyping scenarios
    # 
    # Args:
    #     size (int): Length of the window in samples. Must be positive.
    # 
    # Returns:
    #     mx.array: Bartlett window coefficients of shape (size,)
    # 
    # Performance: Cached for repeated calls with same size.
    return mx.array([1 - 2 * abs(n - (size - 1) / 2) / (size - 1) for n in range(size)])


# Window function lookup table for string-based window selection.
# 
# Maps standard audio processing window names to their implementation functions.
# Supports both common abbreviations and full names for compatibility with
# PyTorch, librosa, and other audio libraries.
# 
# Used by:
# - `stft()` for automatic window function selection
# - `istft()` for reconstruction window selection
# - External APIs requiring string-based window specification
STR_TO_WINDOW_FN = {
    "hann": hanning,        # Short form, matches PyTorch
    "hanning": hanning,     # Full name, matches librosa
    "hamming": hamming,     # Standard name
    "blackman": blackman,   # Standard name  
    "bartlett": bartlett,   # Standard name
}

# Short-Time Fourier Transform (STFT) and Inverse STFT

def stft(
    x,
    n_fft=DEFAULT_N_FFT,
    hop_length=None,
    win_length=None,
    window: mx.array | str = "hann",
    center=True,
    pad_mode="reflect",
):
    # Computes the Short-Time Fourier Transform of an audio signal.
    # 
    # The STFT is the foundation of most audio processing in this library,
    # converting time-domain audio into time-frequency representation for
    # analysis, modification, and reconstruction.
    # 
    # Called by:
    # - `mlx_audio.stt.models.whisper.audio.log_mel_spectrogram()`
    # - `mlx_audio.codec.models.*.encode()` for frequency-domain processing
    # - `mlx_audio.tts.models.*.forward()` for spectral feature extraction
    # 
    # Args:
    #     x (mx.array): Input audio signal of shape (T,) where T is time samples
    #     n_fft (int): FFT size, determines frequency resolution (default 800 = 25 ms frame)
    #     hop_length (int): Frame advance in samples (default n_fft//4 = 75% overlap)
    #     win_length (int): Window length in samples (default = n_fft)
    #     window (mx.array | str): Window function or name ("hann", "hamming", etc.)
    #     center (bool): Whether to center-pad input for boundary frame analysis
    #     pad_mode (str): Padding mode - "reflect" (default) or "constant"
    # 
    # Returns:
    #     mx.array: Complex STFT coefficients of shape (n_frames, n_fft//2 + 1)
    # 
    # Performance: Uses MLX optimized FFT with strided array views for efficiency.
    if hop_length is None:
        hop_length = n_fft // HOP_LENGTH_DIVISOR  # 75% overlap standard
    if win_length is None:
        win_length = n_fft

    if isinstance(window, str):
        window_fn = STR_TO_WINDOW_FN.get(window.lower())
        if window_fn is None:
            raise ValueError(f"Unknown window function: {window}")
        w = window_fn(win_length)
    else:
        w = window

    if w.shape[0] < n_fft:
        pad_size = n_fft - w.shape[0]
        w = mx.concatenate([w, mx.zeros((pad_size,))], axis=0)

    def _pad(x, padding, pad_mode="reflect"):
        # Pads input signal for boundary frame analysis.
        # 
        # Internal helper function that handles different padding strategies
        # to enable STFT analysis near signal boundaries without artifacts.
        # 
        # Args:
        #     x (mx.array): Input signal to pad
        #     padding (int): Number of samples to pad on each side
        #     pad_mode (str): "reflect" for mirror padding, "constant" for zero padding
        # 
        # Returns:
        #     mx.array: Padded signal with shape (T + 2*padding,)
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[SAMPLE_INDEX_OFFSET : padding + SAMPLE_INDEX_OFFSET][::-1]
            suffix = x[-(padding + SAMPLE_INDEX_OFFSET) : -SAMPLE_INDEX_OFFSET][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")

    if center:
        x = _pad(x, n_fft // PADDING_SIZE_DIVISOR, pad_mode)

    # Calculate number of frames with proper boundary handling
    # Formula: 1 frame minimum + additional frames from hop length spacing
    num_frames = SAMPLE_INDEX_OFFSET + (x.shape[0] - n_fft) // hop_length
    if num_frames <= 0:
        raise ValueError(
            f"Input is too short (length={x.shape[0]}) for n_fft={n_fft} with "
            f"hop_length={hop_length} and center={center}."
        )

    shape = (num_frames, n_fft)
    strides = (hop_length, 1)
    frames = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(frames * w)


def istft(
    x,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    length=None,
):
    # Reconstructs time-domain audio from STFT coefficients using overlap-add.
    # 
    # The Inverse STFT reconstructs audio from frequency-domain representation,
    # essential for audio synthesis and modification pipelines throughout the library.
    # 
    # Called by:
    # - `mlx_audio.codec.models.*.decode()` for audio reconstruction
    # - `mlx_audio.tts.models.*.synthesize()` for audio generation
    # - Audio effect processing pipelines
    # 
    # Args:
    #     x (mx.array): Complex STFT coefficients of shape (n_frames, n_freq_bins)
    #     hop_length (int): Frame advance used in forward STFT (default derived from x.shape)
    #     win_length (int): Window length for reconstruction (default derived from x.shape)
    #     window (str | mx.array): Window function name or coefficients (must match forward STFT)
    #     center (bool): Whether input was center-padded (must match forward STFT)
    #     length (int): Desired output length in samples (default auto-computed)
    # 
    # Returns:
    #     mx.array: Reconstructed time-domain audio signal
    # 
    # Performance: Uses optimized overlap-add with MLX scatter operations.
    if win_length is None:
        # Derive window length from RFFT size: n_fft = 2 * (n_freq_bins - 1)
        win_length = (x.shape[1] - SAMPLE_INDEX_OFFSET) * FREQUENCY_BIN_MULTIPLIER
    if hop_length is None:
        hop_length = win_length // HOP_LENGTH_DIVISOR  # Match STFT default

    if isinstance(window, str):
        window_fn = STR_TO_WINDOW_FN.get(window.lower())
        if window_fn is None:
            raise ValueError(f"Unknown window function: {window}")
        w = window_fn(win_length + WINDOW_BOUNDARY_OFFSET)[:-WINDOW_BOUNDARY_OFFSET]
    else:
        w = window

    if w.shape[0] < win_length:
        w = mx.concatenate([w, mx.zeros((win_length - w.shape[0],))], axis=0)

    num_frames = x.shape[1]
    # Calculate total reconstruction length with proper frame overlap
    t = (num_frames - SAMPLE_INDEX_OFFSET) * hop_length + win_length

    reconstructed = mx.zeros(t)
    window_sum = mx.zeros(t)

    # inverse FFT of each frame
    frames_time = mx.fft.irfft(x, axis=0).transpose(1, 0)

    # get the position in the time-domain signal to add the frame
    frame_offsets = mx.arange(num_frames) * hop_length
    indices = frame_offsets[:, None] + mx.arange(win_length)
    indices_flat = indices.flatten()

    updates_reconstructed = (frames_time * w).flatten()
    updates_window = mx.tile(w, (num_frames,)).flatten()

    # overlap-add the inverse transformed frame, scaled by the window
    reconstructed = reconstructed.at[indices_flat].add(updates_reconstructed)
    window_sum = window_sum.at[indices_flat].add(updates_window)

    # normalize by the sum of the window values
    reconstructed = mx.where(window_sum != 0, reconstructed / window_sum, reconstructed)

    if center and length is None:
        # Remove center padding applied during forward STFT
        padding_samples = win_length // PADDING_SIZE_DIVISOR
        reconstructed = reconstructed[padding_samples : -padding_samples]

    if length is not None:
        reconstructed = reconstructed[:length]

    return reconstructed


# Mel-scale filterbank generation for perceptual audio analysis

@lru_cache(maxsize=None)
def mel_filters(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0,
    f_max: Optional[float] = None,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
) -> mx.array:
    # Generates mel-scale triangular filterbank for perceptual audio analysis.
    # 
    # Creates a bank of overlapping triangular filters spaced on the mel scale,
    # which approximates human auditory perception. Essential for speech recognition
    # and audio understanding models throughout this library.
    # 
    # Called by:
    # - `mlx_audio.stt.models.whisper.audio.log_mel_spectrogram()`
    # - `mlx_audio.stt.models.*.preprocess()` for feature extraction
    # - `mlx_audio.tts.models.*.encode()` for perceptual audio encoding
    # 
    # Args:
    #     sample_rate (int): Audio sampling rate in Hz (e.g., 16000, 24000)
    #     n_fft (int): FFT size, determines frequency resolution
    #     n_mels (int): Number of mel filters to generate (typically 80-128)
    #     f_min (float): Minimum frequency in Hz (default 0)
    #     f_max (float): Maximum frequency in Hz (default sample_rate/2)
    #     norm (str): Normalization type - "slaney" or None
    #     mel_scale (str): Scale type - "htk" (default) or "slaney"
    # 
    # Returns:
    #     mx.array: Filterbank matrix of shape (n_mels, n_fft//2 + 1)
    # 
    # Performance: Cached for repeated calls with same parameters.
    def hz_to_mel(freq, mel_scale="htk"):
        # Converts frequency in Hz to mel scale value.
        # 
        # Internal helper implementing both HTK and Slaney mel scale conversions.
        # HTK scale is used by most speech recognition systems, Slaney scale
        # matches Auditory Toolbox and some librosa configurations.
        # 
        # Args:
        #     freq (float): Frequency in Hz to convert
        #     mel_scale (str): "htk" for HTK scale, "slaney" for Slaney scale
        # 
        # Returns:
        #     float: Corresponding mel scale value
        if mel_scale == "htk":
            return HTK_MEL_SCALE_FACTOR * math.log10(1.0 + freq / HTK_FREQ_REFERENCE)

        # Slaney scale implementation - matches Malcolm Slaney's Auditory Toolbox
        # Slaney scale linear region parameters
        f_min, f_sp = 0.0, SLANEY_FREQ_STEP
        mels = (freq - f_min) / f_sp
        min_log_hz = SLANEY_LOG_THRESHOLD
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(SLANEY_LOG_STEP_FACTOR) / SLANEY_LOG_STEP_DIVISOR
        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz) / logstep
        return mels

    def mel_to_hz(mels, mel_scale="htk"):
        # Converts mel scale values back to frequency in Hz.
        # 
        # Internal helper implementing inverse mel scale conversion for both
        # HTK and Slaney scales. Used to compute filter center frequencies.
        # 
        # Args:
        #     mels (mx.array | float): Mel scale values to convert
        #     mel_scale (str): "htk" for HTK scale, "slaney" for Slaney scale
        # 
        # Returns:
        #     mx.array | float: Corresponding frequencies in Hz
        if mel_scale == "htk":
            return HTK_FREQ_REFERENCE * (10.0 ** (mels / HTK_MEL_SCALE_FACTOR) - 1.0)

        # Slaney scale inverse conversion
        # Slaney scale linear region parameters for inverse conversion
        f_min, f_sp = 0.0, SLANEY_FREQ_STEP
        freqs = f_min + f_sp * mels
        min_log_hz = SLANEY_LOG_THRESHOLD
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(SLANEY_LOG_STEP_FACTOR) / SLANEY_LOG_STEP_DIVISOR
        freqs = mx.where(
            mels >= min_log_mel,
            min_log_hz * mx.exp(logstep * (mels - min_log_mel)),
            freqs,
        )
        return freqs

    f_max = f_max or sample_rate / NYQUIST_DIVISOR

    # generate frequency points

    # Calculate frequency bins for RFFT output
    n_freqs = n_fft // NYQUIST_DIVISOR + SAMPLE_INDEX_OFFSET
    all_freqs = mx.linspace(0, sample_rate // NYQUIST_DIVISOR, n_freqs)

    # convert frequencies to mel and back to hz

    m_min = hz_to_mel(f_min, mel_scale)
    m_max = hz_to_mel(f_max, mel_scale)
    # Generate mel scale points with edge filters (+2 for boundary filters)
    m_pts = mx.linspace(m_min, m_max, n_mels + MEL_FILTER_EDGE_OFFSET)
    f_pts = mel_to_hz(m_pts, mel_scale)

    # compute slopes for filterbank

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = mx.expand_dims(f_pts, 0) - mx.expand_dims(all_freqs, 1)

    # calculate overlapping triangular filters

    down_slopes = (-slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    filterbank = mx.maximum(
        mx.zeros_like(down_slopes), mx.minimum(down_slopes, up_slopes)
    )

    if norm == "slaney":
        # Apply Slaney normalization - area under each filter sums to 1
        # Apply Slaney normalization using edge filter boundaries
        enorm = SLANEY_NORM_FACTOR / (f_pts[MEL_FILTER_EDGE_OFFSET : n_mels + MEL_FILTER_EDGE_OFFSET] - f_pts[:n_mels])
        filterbank *= mx.expand_dims(enorm, 0)

    filterbank = filterbank.moveaxis(0, 1)
    return filterbank
