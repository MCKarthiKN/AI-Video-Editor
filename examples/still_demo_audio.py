"""Generate the sound design for the 10 s demo. usage: python3 audio.py out.wav"""
import sys, math
import numpy as np
from scipy import signal
from scipy.io import wavfile

SR, DUR = 48000, 10.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(3)
L = np.zeros(N); R = np.zeros(N)

def sstep(a, b, x):
    u = np.clip((x - a) / (b - a), 0, 1)
    return u * u * (3 - 2 * u)

def place(sig, start, pan=0.0, gain=1.0):
    """Add a mono signal at time `start` with constant-power pan (-1 left .. 1 right)."""
    i0 = int(start * SR)
    n = min(len(sig), N - i0)
    if n <= 0:
        return
    a = (pan + 1) * math.pi / 4
    L[i0:i0 + n] += sig[:n] * math.cos(a) * gain
    R[i0:i0 + n] += sig[:n] * math.sin(a) * gain

def sweep_noise(dur, f0, f1, width=0.35, curve='exp'):
    """Noise whose band centre glides from f0 to f1 (log-gaussian band in the STFT domain)."""
    n = int(dur * SR)
    x = rng.normal(size=n)
    f, ts, Z = signal.stft(x, SR, nperseg=1024)
    u = np.clip(ts / dur, 0, 1)
    fc = f0 * (f1 / f0) ** u if curve == 'exp' else f0 + (f1 - f0) * u
    gain = np.exp(-(np.log(np.maximum(f, 1)[:, None] / fc[None, :])) ** 2 / (2 * width ** 2))
    _, y = signal.istft(Z * gain, SR, nperseg=1024)
    y = y[:n]
    return y / (np.abs(y).max() + 1e-9)

def chime(freqs, dur, tau, vib=0.0):
    tt = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(tt)
    for k, f in enumerate(freqs):
        ph = rng.uniform(0, 6.28)
        y += np.sin(2 * np.pi * f * tt + ph + vib * np.sin(2 * np.pi * 5.5 * tt)) * np.exp(-tt / (tau * (1 - 0.15 * k / len(freqs))))
    att = np.clip(tt / 0.008, 0, 1)
    return y * att / len(freqs)

def boom(dur, f0, f1, tau):
    tt = np.arange(int(dur * SR)) / SR
    f = f1 + (f0 - f1) * np.exp(-tt / (dur * 0.3))
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-tt / tau) * np.clip(tt / 0.005, 0, 1)

def lowpass(x, fc, order=2):
    sos = signal.butter(order, fc, 'low', fs=SR, output='sos')
    return signal.sosfilt(sos, x)

def bandpass(x, f0, f1, order=2):
    sos = signal.butter(order, [f0, f1], 'band', fs=SR, output='sos')
    return signal.sosfilt(sos, x)

# 1) ambient drone bed (the original never goes fully quiet)
env = sstep(0.7, 2.0, t) * (1 + 0.5 * sstep(4.8, 6.3, t) - 0.3 * sstep(6.3, 7.0, t)) * (1 - sstep(9.2, 10.0, t))
lfo = 1 + 0.25 * np.sin(2 * np.pi * 0.15 * t)
for f, a in ((55, 1.0), (82.41, 0.6), (110, 0.5), (164.81, 0.22)):
    L += 0.05 * a * env * lfo * np.sin(2 * np.pi * (f - 0.2) * t)
    R += 0.05 * a * env * lfo * np.sin(2 * np.pi * (f + 0.2) * t + 1.0)
air = bandpass(rng.normal(size=N), 250, 900) * 0.025 * env * (1 + 0.3 * np.sin(2 * np.pi * 0.11 * t))
L += air; R += np.roll(air, 900)

# 2) lights out
place(boom(2.0, 95, 38, 0.5), 0.65, 0, 0.45)
place(lowpass(rng.normal(size=int(0.4 * SR)), 300) * np.exp(-np.arange(int(0.4 * SR)) / SR / 0.06), 0.65, 0, 0.35)

# 3) hand ignitions: whoosh + shimmer, left then right
for start, pan in ((1.8, -0.7), (1.95, 0.7)):
    w = sweep_noise(0.9, 700, 4500, 0.4)
    tt = np.arange(len(w)) / SR
    place(w * np.clip(tt / 0.04, 0, 1) * np.exp(-tt / 0.22), start - 0.03, pan, 0.18)
    place(chime([1567.98, 2349.32, 3135.96], 2.0, 0.8, 0.002), start, pan, 0.16)

# 4) orange appears: bloom chime + soft whoosh, then a hum while it is present
place(chime([659.25, 987.77, 1318.51, 1975.53], 3.0, 1.3), 2.2, 0, 0.22)
w = sweep_noise(0.7, 300, 2500, 0.45); tt = np.arange(len(w)) / SR
place(w * np.clip(tt / 0.1, 0, 1) * np.exp(-tt / 0.25), 2.12, 0, 0.12)
hum_env = sstep(2.3, 2.9, t) * (1 - sstep(5.9, 6.3, t))
trem = 1 + 0.3 * np.sin(2 * np.pi * 4.5 * t)
hum = (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 330.5 * t)) * hum_env * trem * 0.025
L += hum; R += hum

# 5) riser into the magnify, cut at the flash
rd = 1.5
rs = sweep_noise(rd, 400, 7000, 0.3)
tt = np.arange(len(rs)) / SR
cres = (tt / rd) ** 2.5 * np.clip((rd - tt) / 0.02, 0, 1)
glide = np.sin(2 * np.pi * np.cumsum(110 * 6 ** (tt / rd)) / SR) * cres
place(rs * cres, 4.8, 0, 0.3)
place(glide, 4.8, 0, 0.1)

# 6) impact at the flash
place(boom(2.5, 72, 32, 0.6), 6.3, 0, 0.6)
nb = lowpass(rng.normal(size=int(1.0 * SR)), 2200); tt = np.arange(len(nb)) / SR
place(nb * np.exp(-tt / 0.18) / np.abs(nb).max(), 6.3, 0, 0.3)
place(chime([880.0, 1318.51, 1760.0, 2637.02], 3.5, 1.8), 6.3, -0.2, 0.2)
place(chime([1108.73, 1661.22, 2217.46], 3.5, 1.6), 6.32, 0.2, 0.14)

# 7) electron pings (pentatonic), wandering across the stereo field
notes = [1046.5, 1174.66, 1318.51, 1567.98, 1760.0, 1318.51, 1567.98]
for k, f in enumerate(notes):
    st = 6.95 + k * 0.32
    tt = np.arange(int(0.9 * SR)) / SR
    p = (np.sin(2 * np.pi * f * tt) + 0.2 * np.sin(2 * np.pi * 2 * f * tt)) * np.exp(-tt / 0.35) * np.clip(tt / 0.004, 0, 1)
    place(p, st, math.sin(k * 1.7) * 0.8, 0.07)

# 8) simple reverb + master
ir_n = int(2.2 * SR)
irt = np.arange(ir_n) / SR
ir = lowpass(rng.normal(size=ir_n), 5000) * np.exp(-irt / 0.55)
ir /= np.sqrt((ir ** 2).sum())
wetL = signal.fftconvolve(L, ir)[:N]; wetR = signal.fftconvolve(R, np.roll(ir, 311))[:N]
outL = L + 0.35 * wetL; outR = R + 0.35 * wetR
fade = 1 - sstep(9.2, 10.0, t)
out = np.stack([outL * fade, outR * fade], 1)
out = signal.sosfilt(signal.butter(2, 28, 'high', fs=SR, output='sos'), out, axis=0)
out *= 10 ** (-1 / 20) / np.abs(out).max()
wavfile.write(sys.argv[1], SR, (out * 32767).astype(np.int16))
rms = 20 * np.log10(np.sqrt((out ** 2).mean()))
print(f'wrote {sys.argv[1]}  peak -1.0 dBFS  rms {rms:.1f} dBFS')
