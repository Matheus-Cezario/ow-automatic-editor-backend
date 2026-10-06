"""The sound effects library: whooshes, hits, risers -- the punctuation of a
montage.

The sounds are **synthesised here**, not downloaded and not bundled as
recordings. "Nothing paid, no external API" rules out a stock library, and a
folder of WAVs from the internet would bring a licence question along with
every file. A whoosh is filtered noise with a sweep; a hit is a falling sine
with a click on top -- a few lines each, the same bytes on every machine, and
no question about who owns them.

It is pure Python on purpose: the gateway serves this, and its image has no
numpy. A missing import there is how the preprocessor once sat in a restart
loop for a whole phase.

An effect enters a montage the way any sound does -- as an audio item of the
match library (see the gateway's `POST /api/jobs/{id}/sfx`). From there on it
is a block on an audio layer like the music, and the render needs nothing new.
"""

from __future__ import annotations

import io
import math
import random
import struct
import wave
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

#: 44.1 kHz mono, 16-bit: what `read_wav` reads, and plenty for a hit or a
#: whoosh. The render resamples to the mix's rate anyway.
SR = 44100

#: Peak each effect is normalised to, a little under full scale so a hit on top
#: of the music does not clip the mix before `loudnorm` sees it.
PEAK = 0.89

Signal = list[float]

#: The waveform's density -- the same numbers as `owcore.audio`, copied rather
#: than imported, because that module needs numpy.
PEAKS_PER_S = 40
MIN_PEAKS = 64


# ── building blocks ─────────────────────────────────────────────────────────


def _n(duration_s: float) -> int:
    return int(round(duration_s * SR))


def _noise(n: int, seed: int) -> Signal:
    rnd = random.Random(seed)
    return [rnd.uniform(-1.0, 1.0) for _ in range(n)]


def _lowpass(x: Signal, cutoff: Callable[[float], float]) -> Signal:
    """One-pole low-pass whose cutoff (Hz) moves with the fraction of the
    sound already played -- the sweep is what turns noise into a whoosh."""
    out: Signal = []
    y = 0.0
    n = len(x)
    for i, v in enumerate(x):
        fc = max(20.0, cutoff(i / n))
        a = 1.0 - math.exp(-2.0 * math.pi * fc / SR)
        y += a * (v - y)
        out.append(y)
    return out


def _sine(n: int, freq: Callable[[float], float], phase: float = 0.0) -> Signal:
    """A sine whose frequency follows `freq(fraction)`, integrated sample by
    sample so a sweep has no jumps."""
    out: Signal = []
    for i in range(n):
        phase += 2.0 * math.pi * freq(i / n) / SR
        out.append(math.sin(phase))
    return out


def _saw(n: int, freq: float) -> Signal:
    step = freq / SR
    out: Signal = []
    p = 0.0
    for _ in range(n):
        out.append(2.0 * p - 1.0)
        p = (p + step) % 1.0
    return out


def _envelope(x: Signal, env: Callable[[float], float]) -> Signal:
    n = len(x)
    return [v * env(i / n) for i, v in enumerate(x)]


def _mix(*parts: tuple[Signal, float]) -> Signal:
    n = max(len(p) for p, _ in parts)
    out = [0.0] * n
    for p, gain in parts:
        for i, v in enumerate(p):
            out[i] += v * gain
    return out


def _decay(rate: float) -> Callable[[float], float]:
    return lambda t: math.exp(-rate * t)


def _swell(t: float) -> float:
    """Rises and falls, peaking a little past the middle -- the shape of
    something passing by."""
    return math.sin(math.pi * min(1.0, t)) ** 2


def _finish(x: Signal) -> Signal:
    """Normalises and softens the first and last few milliseconds: a sound
    that starts or stops on a non-zero sample clicks, and a click on every
    effect would be heard as a defect, not as the effect."""
    top = max((abs(v) for v in x), default=0.0)
    gain = PEAK / top if top > 0 else 0.0
    out = [v * gain for v in x]
    # a millisecond in, so a hit keeps its attack; a little longer out
    attack, release = int(0.001 * SR), min(len(x) // 4, int(0.004 * SR))
    for i in range(attack):
        out[i] *= i / attack
    for i in range(release):
        out[-1 - i] *= i / release
    return out


# ── the effects ─────────────────────────────────────────────────────────────


def _whoosh() -> Signal:
    n = _n(0.7)
    return _envelope(
        _lowpass(_noise(n, 1), lambda t: 300 + 5000 * _swell(t)), _swell
    )


def _swipe() -> Signal:
    n = _n(0.35)
    return _envelope(
        _lowpass(_noise(n, 2), lambda t: 1500 + 9000 * t),
        lambda t: _swell(t) * (1 - t * 0.3),
    )


def _hit() -> Signal:
    n = _n(0.5)
    body = _envelope(_sine(n, lambda t: 50 + 110 * math.exp(-12 * t)), _decay(7))
    click = _envelope(_lowpass(_noise(n, 3), lambda t: 6000), _decay(60))
    return _mix((body, 1.0), (click, 0.6))


def _boom() -> Signal:
    n = _n(1.6)
    sub = _envelope(_sine(n, lambda t: 35 + 70 * math.exp(-6 * t)), _decay(3))
    rumble = _envelope(_lowpass(_noise(n, 4), lambda t: 400 * (1 - t) + 80), _decay(4))
    return _mix((sub, 1.0), (rumble, 0.8))


def _riser() -> Signal:
    n = _n(2.0)
    air = _envelope(
        _lowpass(_noise(n, 5), lambda t: 400 + 9000 * t * t), lambda t: t**2
    )
    tone = _envelope(_sine(n, lambda t: 200 + 900 * t * t), lambda t: t**3)
    return _mix((air, 0.8), (tone, 0.35))


def _downlifter() -> Signal:
    n = _n(1.5)
    air = _envelope(
        _lowpass(_noise(n, 6), lambda t: 8000 * (1 - t) ** 2 + 200), _decay(2.5)
    )
    tone = _envelope(_sine(n, lambda t: 900 * (1 - t) ** 2 + 60), _decay(2))
    return _mix((air, 0.8), (tone, 0.4))


def _ding() -> Signal:
    n = _n(1.2)
    partials = [(1.0, 1.0), (2.76, 0.45), (5.4, 0.25), (8.93, 0.12)]
    return _mix(*(
        (_envelope(_sine(n, lambda t, r=r: 880 * r), _decay(4 + 3 * r)), g)
        for r, g in partials
    ))


def _beep() -> Signal:
    """The censor beep: what goes over a word that should not be heard."""
    n = _n(0.5)
    return _envelope(_sine(n, lambda t: 1000), lambda t: 1.0)


def _tick() -> Signal:
    n = _n(0.25)
    return _mix(
        (_envelope(_sine(n, lambda t: 2400), _decay(40)), 1.0),
        (_envelope(_noise(n, 7), _decay(90)), 0.3),
    )


def _zap() -> Signal:
    n = _n(0.4)
    return _envelope(_sine(n, lambda t: 2200 * math.exp(-5 * t) + 120), _decay(5))


def _glitch() -> Signal:
    """Stutters of noise and square bursts, cut on a grid -- the digital
    tear that goes with a glitch transition."""
    n = _n(0.6)
    rnd = random.Random(8)
    out: Signal = []
    slice_n = _n(0.03)
    while len(out) < n:
        kind = rnd.random()
        freq = rnd.choice((220, 440, 880, 1760))
        gain = rnd.uniform(0.3, 1.0)
        for i in range(slice_n):
            if kind < 0.35:
                out.append(0.0)
            elif kind < 0.7:
                out.append(gain * (1.0 if (i * freq // SR) % 2 else -1.0))
            else:
                out.append(gain * rnd.uniform(-1.0, 1.0))
    return out[:n]


def _airhorn() -> Signal:
    n = _n(1.3)
    chord = _mix(*((_saw(n, f), 1.0) for f in (415.0, 523.0, 622.0)))
    shaped = _lowpass(chord, lambda t: 3500)
    return _envelope(
        shaped, lambda t: min(1.0, t * 25) * (1.0 if t < 0.8 else (1 - t) / 0.2)
    )


def _heartbeat() -> Signal:
    n = _n(1.0)

    def thump(at: float, gain: float) -> Signal:
        start, length = _n(at), _n(0.25)
        body = _envelope(_sine(length, lambda t: 45 + 30 * math.exp(-10 * t)), _decay(14))
        return [0.0] * start + [v * gain for v in body]

    return _mix((thump(0.0, 1.0), 1.0), (thump(0.28, 0.7), 1.0), ([0.0] * n, 0.0))


def _level_up() -> Signal:
    notes = (523.25, 659.25, 783.99, 1046.5)
    step = _n(0.1)
    tail = _n(0.35)
    out: Signal = []
    for k, f in enumerate(notes):
        length = step if k < len(notes) - 1 else tail
        tone = _mix((_saw(length, f), 0.35), (_sine(length, lambda t, f=f: f), 1.0))
        out += _envelope(tone, _decay(6 if k < len(notes) - 1 else 5))
    return out


@dataclass(frozen=True)
class Effect:
    id: str
    name: str
    #: how the app groups them: transition, impact, tension, ui, meme
    category: str
    make: Callable[[], Signal]


#: The catalogue, in the order the app shows it. The ids are part of the
#: montage's history (the library item remembers where it came from), so an
#: id is never reused for a different sound.
EFFECTS: tuple[Effect, ...] = (
    Effect("whoosh", "Whoosh", "transition", _whoosh),
    Effect("swipe", "Swipe", "transition", _swipe),
    Effect("riser", "Riser", "transition", _riser),
    Effect("downlifter", "Downlifter", "transition", _downlifter),
    Effect("hit", "Hit", "impact", _hit),
    Effect("boom", "Boom", "impact", _boom),
    Effect("heartbeat", "Heartbeat", "impact", _heartbeat),
    Effect("glitch", "Glitch", "impact", _glitch),
    Effect("ding", "Ding", "ui", _ding),
    Effect("tick", "Tick", "ui", _tick),
    Effect("zap", "Zap", "ui", _zap),
    Effect("level-up", "Level up", "ui", _level_up),
    Effect("airhorn", "Air horn", "meme", _airhorn),
    Effect("beep", "Censor beep", "meme", _beep),
)

CATEGORIES = ("transition", "impact", "ui", "meme")


def catalog() -> dict[str, Effect]:
    return {e.id: e for e in EFFECTS}


@lru_cache(maxsize=None)
def _signal(effect_id: str) -> tuple[float, ...]:
    effect = catalog().get(effect_id)
    if effect is None:
        raise KeyError(effect_id)
    return tuple(_finish(effect.make()))


def duration_s(effect_id: str) -> float:
    return round(len(_signal(effect_id)) / SR, 3)


@lru_cache(maxsize=None)
def wav_bytes(effect_id: str) -> bytes:
    """The effect as a 16-bit mono WAV. Raises `KeyError` for an unknown id."""
    samples = _signal(effect_id)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(
            struct.pack(
                f"<{len(samples)}h",
                *(int(max(-1.0, min(1.0, v)) * 32767) for v in samples),
            )
        )
    return buf.getvalue()


@lru_cache(maxsize=None)
def peaks(effect_id: str) -> tuple[float, ...]:
    """The waveform the app draws, in the same format `owcore.audio` gives a
    song: 40 points a second, never fewer than 64, between 0 and 1."""
    samples = _signal(effect_id)
    n = min(max(MIN_PEAKS, int(len(samples) / SR * PEAKS_PER_S)), len(samples))
    width = len(samples) // n
    blocks = [
        max(abs(v) for v in samples[i * width : (i + 1) * width]) for i in range(n)
    ]
    top = max(blocks, default=0.0)
    return tuple(round(b / top, 3) if top > 0 else 0.0 for b in blocks)
