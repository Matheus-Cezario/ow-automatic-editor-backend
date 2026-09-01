"""Audio reading and the reduced waveform the app draws.

It lives here, and not in the rhythm service, because there are now **two**
audios the editor draws: the music the user chose and the match's own audio --
the shot, the explosion, the noise of the play. Both become the same thing: a
few thousand peaks between 0 and 1.

Sending the waveform instead of the audio is the point: the app needs it to
*draw*, and downloading minutes of sound just to find where the chorus is would
be expensive on a phone.

**Everything here reads in blocks.** The previous version did
`np.frombuffer(w.readframes(all)).astype(np.float32)`, which puts the whole
file in memory three times at once -- the raw bytes, the float32 copy and the
result of the mono downmix. Measured on a 53 MB WAV (20 min of match audio, as
the preprocessor extracts it), the peak was **265 MB**, plus another 212 MB on
top to reduce that to 6000 points. To draw a ruler. Reading in blocks costs the
same time and a fixed memory ceiling.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

#: How many peaks per second the waveform keeps. 40 is enough for the drawing
#: to look like the sound on a phone; keeping sample resolution would mean
#: sending the whole audio as JSON.
PEAKS_PER_S = 40

#: Ceiling of peaks per track. A 20-minute match does not need 48 thousand
#: points to fit a ruler a thousand pixels wide.
MAX_PEAKS = 6000

#: How many audio frames to read at a time. 1 Mi samples is 2 MB of raw bytes
#: plus 4 MB in float32 -- large enough for numpy to work in efficient batches,
#: and small enough that the peak does not depend on the file size. It is this
#: number, and not the audio duration, that fixes the ceiling.
READ_BLOCK = 1 << 20


def _header(w: wave.Wave_read) -> tuple[int, int, int, int]:
    """`(sample_rate, channels, width, frames)` -- with the width checked.

    16-bit PCM only: it is what `extract_audio` and the music decoder produce,
    and silently guessing another width would give a wrong waveform instead of
    an error.
    """
    width = w.getsampwidth()
    if width != 2:
        raise ValueError(f"esperado PCM 16 bits, recebi {width * 8} bits")
    return w.getframerate(), w.getnchannels(), width, w.getnframes()


def _blocks(w: wave.Wave_read, channels: int, at_a_time: int):
    """Walks the WAV in chunks already converted to mono float32."""
    while True:
        raw = w.readframes(at_a_time)
        if not raw:
            return
        block = np.frombuffer(raw, dtype="<i2")
        if channels > 1:
            block = block.reshape(-1, channels).mean(axis=1, dtype=np.float32)
            yield block / 32768.0
        else:
            # `astype` already produces the float32 copy; dividing in place
            # avoids the second temporary matrix of `block / 32768.0`
            out = block.astype(np.float32)
            out /= 32768.0
            yield out


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """PCM WAV -> samples in -1..1, mono.

    Returns the whole signal, because the caller (the beat tracker) needs it
    whole. What is avoided here is the **transient**: the result is written
    straight into a pre-allocated array instead of being born from two full
    copies.
    """
    with wave.open(str(path), "rb") as w:
        sr, channels, _width, frames = _header(w)
        data = np.empty(frames, dtype=np.float32)
        end = 0
        for block in _blocks(w, channels, READ_BLOCK):
            n = min(block.size, data.size - end)
            if n <= 0:
                break
            data[end : end + n] = block[:n]
            end += n
    return data[:end], sr


def peaks_for(duration_s: float) -> int:
    """How many points to use for a track of this duration."""
    return int(min(MAX_PEAKS, max(64, duration_s * PEAKS_PER_S)))


def _normalize(blocks: np.ndarray) -> list[float]:
    """Envelope normalized by the track's peak, rounded for the JSON.

    Normalizing by the maximum, rather than by an absolute value, is what makes
    a quiet recording draw like a loud one -- the drawing exists to find the
    moment by eye, not to measure volume.
    """
    if blocks.size == 0:
        return []
    top = float(blocks.max())
    if top <= 0:
        return [0.0] * int(blocks.size)
    return [round(float(v), 3) for v in (blocks / top)]


def waveform(data: np.ndarray, n: int) -> list[float]:
    """Envelope of a signal already in memory, in `n` points."""
    if data.size == 0 or n <= 0:
        return []
    n = min(n, data.size)
    width = data.size // n
    if width == 0:
        return []
    # in batches: `np.abs(data[:cut])` over the whole thing doubled the memory
    # peak just to reduce everything to a few thousand numbers
    blocks = np.empty(n, dtype=np.float32)
    per_batch = max(1, READ_BLOCK // width)
    for i in range(0, n, per_batch):
        j = min(n, i + per_batch)
        chunk = data[i * width : j * width].reshape(j - i, width)
        np.abs(chunk).max(axis=1, out=blocks[i:j])
    return _normalize(blocks)


def waveform_of(path: Path) -> tuple[list[float], float]:
    """The waveform of a WAV, with the duration it covers.

    Reads the file **in blocks** and never holds the whole signal: the audio of
    a 20-minute match becomes 6000 numbers without ever occupying more than a
    few megabytes. It is the difference between a preprocessor that fits in a
    small container and one that bursts it.

    Returns `([], 0.0)` for an empty or unreadable file: a track with no
    waveform is one drawing fewer, not an error worth failing the job over.
    """
    try:
        with wave.open(str(Path(path)), "rb") as w:
            sr, channels, _width, frames = _header(w)
            if sr <= 0 or frames <= 0:
                return [], 0.0
            duration = frames / sr
            n = min(peaks_for(duration), frames)
            width = frames // n
            if width == 0:
                return [], 0.0

            blocks = np.empty(n, dtype=np.float32)
            # read an exact multiple of the block width, so every peak comes
            # from a whole window and no sample straddles two reads
            at_a_time = max(width, (READ_BLOCK // width) * width)
            leftover = np.empty(0, dtype=np.float32)
            done = 0
            for chunk in _blocks(w, channels, at_a_time):
                if done >= n:
                    break
                signal = (
                    np.concatenate((leftover, chunk)) if leftover.size else chunk
                )
                fit = min(signal.size // width, n - done)
                if fit:
                    window = signal[: fit * width].reshape(fit, width)
                    np.abs(window).max(axis=1, out=blocks[done : done + fit])
                    done += fit
                leftover = signal[fit * width :].copy()
    except (OSError, wave.Error, ValueError):
        return [], 0.0
    return _normalize(blocks[:done]), round(duration, 3)
