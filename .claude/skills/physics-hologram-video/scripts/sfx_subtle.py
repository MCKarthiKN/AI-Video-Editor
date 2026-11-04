"""Subtle, wide, event-driven sound palette (shimmer, flutter, atom swishes, vibration drones).

Library functions return stereo arrays (n, 2). Used by score scripts that place cues on a timeline.
"""
import math
import numpy as np
from scipy import signal

SR = 48000
rng = np.random.default_rng(21)


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def sstep(a, b, x):
    u = np.clip((x - a) / (b - a), 0, 1)
    return u * u * (3 - 2 * u)


def bandnoise(dur, f0, f1, order=2):
    x = rng.normal(size=int(dur * SR))
    sos = signal.butter(order, [f0, f1], 'band', fs=SR, output='sos')
    y = signal.sosfilt(sos, x)
    return y / (np.abs(y).max() + 1e-9)


def wide(mono, width=1.0, delay_ms=11.0):
    """Decorrelate a mono signal into a wide stereo image (Haas + mid/side)."""
    d = int(delay_ms * SR / 1000)
    l = mono
    r = np.concatenate([np.zeros(d), mono[:-d]]) if d else mono
    mid, side = (l + r) / 2, (l - r) / 2
    side *= 1 + width
    return np.stack([mid + side, mid - side], 1)


def pan(mono, p):
    """p: scalar or per-sample array in -1..1 (constant-power)."""
    a = (np.asarray(p) + 1) * math.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)], 1)


def flutter(n, rate=11.0, depth=0.6, jitter=0.25):
    t = np.arange(n) / SR
    r = rate * (1 + jitter * np.sin(2 * np.pi * 0.37 * t + rng.uniform(0, 6)))
    ph = 2 * np.pi * np.cumsum(r) / SR
    return 1 - depth * (0.5 + 0.5 * np.sin(ph))


def shimmer(dur, f0=1750.0, spread=0.06, n=7, rate=11.0, air=0.5, env=None):
    """Glassy fluttering tone cluster + breathing air noise, wide stereo."""
    t = tt(dur)
    out = np.zeros((len(t), 2))
    for k in range(n):
        f = f0 * (1 + rng.uniform(-spread, spread))
        tone = np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28) + 0.003 * f / 1000 * np.sin(2 * np.pi * 5 * t))
        tone *= flutter(len(t), rate * rng.uniform(0.8, 1.25), 0.7)
        out += pan(tone, rng.uniform(-0.9, 0.9)) / n
    a = bandnoise(dur, 3000, 12000) * flutter(len(t), rate * 0.6, 0.5)
    out += wide(a, 1.2) * air * 0.35
    if env is not None:
        out *= env[:, None]
    return out


def swish(dur, f0, f1, p0, p1, bw=0.5):
    """Soft moving whoosh: band sweeps f0->f1 and pans p0->p1 with a bell envelope."""
    n = int(dur * SR)
    x = rng.normal(size=n)
    f, ts, Z = signal.stft(x, SR, nperseg=1024)
    u = np.clip(ts / dur, 0, 1)
    fc = f0 * (f1 / f0) ** u
    g = np.exp(-(np.log(np.maximum(f, 1)[:, None] / fc[None, :])) ** 2 / (2 * bw ** 2))
    _, y = signal.istft(Z * g, SR, nperseg=1024)
    y = y[:n] / (np.abs(y[:n]).max() + 1e-9)
    tt_ = np.arange(n) / n
    y *= np.sin(np.pi * tt_) ** 2
    return pan(y, p0 + (p1 - p0) * tt_)


def vibration(dur, f0=640.0, beat=1.6, env=None):
    """String-like sustained 'vibration': detuned partials beating, slow vibrato, wide."""
    t = tt(dur)
    vib = 1 + 0.004 * np.sin(2 * np.pi * 4.8 * t)
    y = np.zeros(len(t))
    for h, a in ((1, 1.0), (2, 0.45), (3, 0.25), (4, 0.12)):
        for d in (-beat / 2, beat / 2):
            y += a * np.sin(2 * np.pi * np.cumsum((f0 * h + d) * vib) / SR + rng.uniform(0, 6.28))
    y /= np.abs(y).max() + 1e-9
    out = wide(y, 1.5, 17)
    if env is not None:
        out *= env[:, None]
    return out


def low_swell(dur, peak_at=0.7):
    n = int(dur * SR)
    y = signal.sosfilt(signal.butter(2, [35, 140], 'band', fs=SR, output='sos'), rng.normal(size=n))
    y /= np.abs(y).max() + 1e-9
    u = np.arange(n) / n
    e = np.where(u < peak_at, (u / peak_at) ** 2, np.exp(-(u - peak_at) / (1 - peak_at) * 3))
    return wide(y * e, 0.6, 5)


def sparkle(dur, density=60, fmin=3000, fmax=9000):
    """Granular glitter: many tiny high pings scattered across the stereo field."""
    n = int(dur * SR)
    out = np.zeros((n, 2))
    for _ in range(int(density * dur)):
        st = rng.integers(0, max(1, n - 4000))
        g = np.arange(4000) / SR
        f = rng.uniform(fmin, fmax)
        gr = np.sin(2 * np.pi * f * g) * np.exp(-g / rng.uniform(0.008, 0.03))
        out[st:st + 4000] += pan(gr, rng.uniform(-1, 1)) * rng.uniform(0.3, 1)
    return out / (np.abs(out).max() + 1e-9)


def bloom(dur=2.5, f0=880.0):
    """Soft glassy swell (instead of a hard 'ding'): partials fade in fast-ish and ring out."""
    t = tt(dur)
    y = np.zeros(len(t))
    for k, r in enumerate((1, 1.5, 2.0, 3.01)):
        y += np.sin(2 * np.pi * f0 * r * t + rng.uniform(0, 6)) * (0.6 ** k)
    e = (1 - np.exp(-t / 0.08)) * np.exp(-t / (dur * 0.35))
    return wide(y * e / 2, 1.3, 13)


def reverb(x, secs=2.4, wet=0.35):
    n = int(secs * SR)
    t = np.arange(n) / SR
    out = np.zeros_like(x)
    for c in range(2):
        ir = signal.sosfilt(signal.butter(2, 6000, 'low', fs=SR, output='sos'), rng.normal(size=n)) * np.exp(-t / (secs * 0.3))
        ir /= np.sqrt((ir ** 2).sum())
        out[:, c] = x[:, c] + wet * signal.fftconvolve(x[:, c], ir)[:len(x)]
    return out


class Timeline:
    def __init__(self, dur):
        self.buf = np.zeros((int(dur * SR), 2))

    def add(self, sig, start, gain_db=0.0):
        i0 = int(start * SR)
        if i0 >= len(self.buf):
            return
        if i0 < 0:
            sig = sig[-i0:]; i0 = 0
        n = min(len(sig), len(self.buf) - i0)
        self.buf[i0:i0 + n] += sig[:n] * 10 ** (gain_db / 20)
