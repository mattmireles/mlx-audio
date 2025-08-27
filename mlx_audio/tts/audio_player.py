import os
import platform
import shutil
import subprocess
import tempfile
import time
import sys
import contextlib
from collections import deque
from threading import Event, Lock

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioPlayer:
    """Robust audio playback helper with graceful fallbacks.

    Primary path uses sounddevice/PortAudio for low-latency streaming.
    If the output device rejects the stream (common on some systems/sample rates),
    we fall back to spawning a system audio player (afplay/paplay/aplay/ffplay)
    on a temporary WAV file. This ensures playback works on any machine.
    """

    # with respect to real-time, not the sample rate
    min_buffer_seconds = 1.5
    measure_window = 0.25
    ema_alpha = 0.25

    def __init__(self, sample_rate: int = 24_000, buffer_size: int = 2048, verbose: bool = False):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.verbose = verbose

        self.audio_buffer = deque()
        self.buffer_lock = Lock()
        self.stream: sd.OutputStream | None = None
        self.playing = False
        self.drain_event = Event()

        self.window_sample_count = 0
        self.window_start = time.perf_counter()
        # arrival_rate measured in the stream's sample-rate domain
        self.arrival_rate = sample_rate

        # Playback sample-rate may differ if we need to adapt to device
        self.playback_sample_rate: int | None = None
        self.channels: int = 1

        # Fallback path state
        self._fallback_mode = False
        self._fallback_segments: list[np.ndarray] = []
        self._fallback_started = False

    def callback(self, outdata, frames, time, status):
        outdata.fill(0)  # initialize the frame with silence
        filled = 0

        with self.buffer_lock:
            while filled < frames and self.audio_buffer:
                buf = self.audio_buffer[0]
                to_copy = min(frames - filled, len(buf))
                # Broadcast mono buffer to all output channels
                outdata[filled : filled + to_copy, :self.channels] = buf[:to_copy, None]
                filled += to_copy

                if to_copy == len(buf):
                    self.audio_buffer.popleft()
                else:
                    self.audio_buffer[0] = buf[to_copy:]

            if not self.audio_buffer and filled < frames:
                self.drain_event.set()
                self.playing = False
                raise sd.CallbackStop()

    def start_stream(self):
        """Attempt to start a PortAudio stream; gracefully degrade on failure.

        Tries the requested sample rate first, then common safe rates. If all
        attempts fail, enable system-player fallback without raising.
        """
        if self._fallback_mode:
            return  # already in fallback

        if self.verbose:
            print("\nStarting audio stream...")
        candidate_rates = []
        # Prefer the requested rate
        candidate_rates.append(self.sample_rate)
        # Add common hardware rates for macOS/Linux/Windows
        for r in (48_000, 44_100):
            if r not in candidate_rates:
                candidate_rates.append(r)

        last_error: Exception | None = None
        for rate in candidate_rates:
            for ch in (1, 2):
                try:
                    # Suppress PortAudio's stderr spam while probing
                    with self._suppress_stderr():
                        self.stream = sd.OutputStream(
                            samplerate=rate,
                            channels=ch,
                            callback=self.callback,
                            blocksize=self.buffer_size,
                            dtype='float32',
                        )
                        self.stream.start()
                    self.playing = True
                    self.drain_event.clear()
                    self.playback_sample_rate = rate
                    self.channels = ch
                    if rate != self.sample_rate and self.verbose:
                        print(f"sounddevice: using {rate} Hz for playback (resampling from {self.sample_rate} Hz)")
                        # Resample already-buffered chunks to the device rate
                        with self.buffer_lock:
                            if self.audio_buffer:
                                rebuffed = deque()
                                while self.audio_buffer:
                                    rebuffed.append(
                                        self._resample_linear(self.audio_buffer.popleft(), self.sample_rate, rate)
                                    )
                                self.audio_buffer = rebuffed
                        # Reset arrival rate/window in the device domain
                        self.arrival_rate = rate
                        self.window_sample_count = 0
                        self.window_start = time.perf_counter()
                    return
                except Exception as e:  # pragma: no cover - hardware dependent
                    last_error = e
                    continue

        # If we reach here, PortAudio failed
        if self.verbose:
            print(f"sounddevice OutputStream unavailable, falling back to system audio player. Reason: {last_error}")
        # Transfer any already-buffered samples into fallback segments
        with self.buffer_lock:
            while self.audio_buffer:
                self._fallback_segments.append(self.audio_buffer.popleft())
        self._fallback_mode = True
        self.playing = False
        self.drain_event.clear()

    def stop_stream(self):
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None
            self.playing = False

    def buffered_samples(self) -> int:
        return sum(map(len, self.audio_buffer))

    def queue_audio(self, samples):
        """Queue mono float samples in range [-1, 1] for playback."""
        if samples is None:
            return
        if not len(samples):
            return

        now = time.perf_counter()

        # Fallback path: accumulate for later one-shot playback
        if self._fallback_mode:
            self._fallback_segments.append(np.asarray(samples, dtype=np.float32))
            return

        # Resample if the device uses a different playback rate
        target_rate = self.playback_sample_rate or self.sample_rate
        chunk = np.asarray(samples, dtype=np.float32)
        if target_rate != self.sample_rate:
            chunk = self._resample_linear(chunk, self.sample_rate, target_rate)

        # arrival-rate statistics in the device's sample-rate domain
        self.window_sample_count += len(chunk)
        if now - self.window_start >= self.measure_window:
            inst_rate = self.window_sample_count / (now - self.window_start)
            self.arrival_rate = (
                inst_rate if self.arrival_rate is None else self.ema_alpha * inst_rate + (1 - self.ema_alpha) * self.arrival_rate
            )
            self.window_sample_count = 0
            self.window_start = now

        with self.buffer_lock:
            self.audio_buffer.append(chunk)

        # start playback only when we have enough buffered audio
        needed = int(self.arrival_rate * self.min_buffer_seconds)
        if not self.playing and self.buffered_samples() >= needed:
            self.start_stream()

    def wait_for_drain(self):
        if self._fallback_mode:
            # For fallback, do a blocking system-player invocation once
            if not self._fallback_started:
                # If there are still samples in the stream buffer, move them
                with self.buffer_lock:
                    while self.audio_buffer:
                        self._fallback_segments.append(self.audio_buffer.popleft())

                if self._fallback_segments:
                    self._fallback_started = True
                    try:
                        self._play_with_system_player(np.concatenate(self._fallback_segments))
                    finally:
                        self._fallback_segments.clear()
                        self.drain_event.set()
            return True
        return self.drain_event.wait()

    def stop(self):
        if self.playing:
            self.wait_for_drain()
            sd.sleep(100)

            self.stop_stream()
            self.playing = False
        # fallback path has nothing to stop explicitly (we block in wait_for_drain)

    def flush(self):
        """Discard everything and stop playback immediately."""
        if self._fallback_mode:
            self._fallback_segments.clear()
            self._fallback_started = False
            self.drain_event.set()
            return

        if not self.playing:
            return

        with self.buffer_lock:
            self.audio_buffer.clear()
        self.stop_stream()
        self.playing = False
        self.drain_event.set()

    # ---- helpers ----

    @staticmethod
    def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Lightweight linear resampler for 1D audio.

        Avoids adding scipy as a dependency. Good enough for monitoring.
        """
        if src_rate == dst_rate or samples.size == 0:
            return samples
        duration = samples.size / src_rate
        dst_len = max(1, int(round(duration * dst_rate)))
        x_src = np.linspace(0.0, 1.0, samples.size, endpoint=False, dtype=np.float32)
        x_dst = np.linspace(0.0, 1.0, dst_len, endpoint=False, dtype=np.float32)
        return np.interp(x_dst, x_src, samples).astype(np.float32)

    def _play_with_system_player(self, samples: np.ndarray):
        """Write to a temp WAV and play via a system player.

        Uses platform-appropriate tools:
        - macOS: afplay
        - Linux: paplay → aplay → ffplay → play
        - Windows: PowerShell SoundPlayer
        """
        # Clip and convert to 16-bit PCM for maximal compatibility
        clipped = np.clip(samples, -1.0, 1.0)
        int16 = (clipped * 32767.0).astype(np.int16)

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, int16, self.sample_rate, subtype='PCM_16')

            system = platform.system()
            if system == 'Darwin':
                cmd = shutil.which('afplay')
                if cmd is None:
                    # Fallback to opening via default app (blocking)
                    subprocess.run(['open', path], check=False)
                else:
                    subprocess.run([cmd, path], check=False)
            elif system == 'Windows':
                # Play synchronously via PowerShell SoundPlayer
                ps = (
                    f"$p=New-Object Media.SoundPlayer '{path}';"
                    f"$p.PlaySync();"
                )
                subprocess.run(['powershell', '-NoProfile', '-Command', ps], check=False)
            else:
                # Linux / other Unixes
                cmd = (
                    shutil.which('paplay')
                    or shutil.which('aplay')
                    or shutil.which('ffplay')
                    or shutil.which('play')
                )
                if cmd is None:
                    # Last resort: try xdg-open (may open a GUI player)
                    subprocess.run(['xdg-open', path], check=False)
                else:
                    if os.path.basename(cmd) == 'ffplay':
                        subprocess.run([cmd, '-autoexit', '-nodisp', '-loglevel', 'error', path], check=False)
                    elif os.path.basename(cmd) == 'play':
                        subprocess.run([cmd, '-q', path], check=False)
                    else:
                        subprocess.run([cmd, path], check=False)
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

    # Redirect C-level stderr temporarily (to silence PortAudio warnings)
    @staticmethod
    @contextlib.contextmanager
    def _suppress_stderr():
        try:
            stderr_fd = sys.stderr.fileno()
        except Exception:
            # If not a real file descriptor (e.g., in some environments), do nothing
            yield
            return
        # Duplicate original stderr
        with os.fdopen(os.dup(stderr_fd), 'wb') as saved_stderr:
            with open(os.devnull, 'wb') as devnull:
                try:
                    os.dup2(devnull.fileno(), stderr_fd)
                    yield
                finally:
                    try:
                        os.dup2(saved_stderr.fileno(), stderr_fd)
                    except Exception:
                        pass
