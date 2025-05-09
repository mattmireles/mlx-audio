from typing import Optional

import mlx.core as mx
import numpy as np
import soundfile as sf
from scipy import signal

SAMPLE_RATE = 16000


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = signal.resample_poly(audio, up, down, padtype="edge")
    return resampled


def load_audio(
    file: str = Optional[str],
    sr: int = SAMPLE_RATE,
    from_stdin=False,
    dtype: mx.Dtype = mx.float32,
):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    audio, sample_rate = sf.read(file, always_2d=True)
    if sample_rate != sr:
        audio = resample_audio(audio, sample_rate, sr)
    return mx.array(audio, dtype=dtype).mean(axis=1)
