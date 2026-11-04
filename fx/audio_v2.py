"""Mix the speaker's voice with a generated sound-design layer.
usage: python3 audio_v2.py voice.wav out.wav"""
import sys, math
import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000
sr_in, v = wavfile.read(sys.argv[1])
v = v.astype(np.float64) / 32768.0
if v.ndim == 2:
    v = v.mean(1)
assert sr_in == SR
N = len(v)
DUR = N / SR
t = np.arange(N) / SR
rng = np.random.default_rng(9)
L = np.zeros(N); R = np.zeros(N)


def sstep(a, b, x):
    u = np.clip((x - a) / (b - a), 0, 1)
    return u * u * (3 - 2 * u)


def place(sig, start, pan=0.0, gain=1.0):
    i0 = int(start * SR)
    if i0 >= N:
        return
    n = min(len(sig), N - i0)
    a = (pan + 1) * math.pi / 4
    L[i0:i0 + n] += sig[:n] * math.cos(a) * gain
    R[i0:i0 + n] += sig[:n] * math.sin(a) * gain


def sweep_noise(dur, f0, f1, width=0.35):
    n = int(dur * SR)
    x = rng.normal(size=n)
    f, ts, Z = signal.stft(x, SR, nperseg=1024)
    u = np.clip(ts / dur, 0, 1)
    fc = f0 * (f1 / f0) ** u
    g = np.exp(-(np.log(np.maximum(f, 1)[:, None] / fc[None, :])) ** 2 / (2 * width ** 2))
    _, y = signal.istft(Z * g, SR, nperseg=1024)
    y = y[:n]
    return y / (np.abs(y).max() + 1e-9)


def chime(freqs, dur, tau, vib=0.0):
    tt = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(tt)
    for k, f in enumerate(freqs):
        y += np.sin(2 * np.pi * f * tt + rng.uniform(0, 6.28) + vib * np.sin(2 * np.pi * 5.5 * tt)) * np.exp(-tt / (tau * (1 - 0.15 * k / len(freqs))))
    return y * np.clip(tt / 0.006, 0, 1) / len(freqs)


def boom(dur, f0, f1, tau):
    tt = np.arange(int(dur * SR)) / SR
    f = f1 + (f0 - f1) * np.exp(-tt / (dur * 0.3))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / tau) * np.clip(tt / 0.005, 0, 1)


def env_shape(sig, att, tau):
    tt = np.arange(len(sig)) / SR
    return sig * np.clip(tt / att, 0, 1) * np.exp(-tt / tau)


def lowpass(x, fc):
    return signal.sosfilt(signal.butter(2, fc, 'low', fs=SR, output='sos'), x)


def bandpass(x, f0, f1):
    return signal.sosfilt(signal.butter(2, [f0, f1], 'band', fs=SR, output='sos'), x)


# ---------- voice ----------
v = signal.sosfilt(signal.butter(2, 75, 'high', fs=SR, output='sos'), v)
def movavg(x, n):
    c = np.concatenate([[0.0], np.cumsum(x)])
    h = n // 2
    i = np.arange(len(x))
    lo, hi = np.clip(i - h, 0, len(x)), np.clip(i + h + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


frame = int(0.05 * SR)
rms = np.sqrt(movavg(v ** 2, frame))
speech = rms > 10 ** (-38 / 20)
v *= 10 ** (-20 / 20) / np.sqrt((v[speech] ** 2).mean())        # speech RMS -> -20 dBFS
act = movavg(speech.astype(float), int(0.25 * SR))
duck = 1 - 0.55 * np.clip(act * 1.5, 0, 1)                        # bed dips under the voice

# ---------- bed ----------
bed_env = sstep(0.4, 1.6, t) * (1 + 0.4 * sstep(16.4, 17.5, t) - 0.3 * sstep(17.6, 18.5, t)) * (1 - sstep(22.4, DUR, t))
lfo = 1 + 0.25 * np.sin(2 * np.pi * 0.13 * t)
for f, a in ((55, 1.0), (82.41, 0.6), (110, 0.55), (164.81, 0.25)):
    L += 0.032 * a * bed_env * lfo * duck * np.sin(2 * np.pi * (f - 0.2) * t)
    R += 0.032 * a * bed_env * lfo * duck * np.sin(2 * np.pi * (f + 0.2) * t + 1.0)
air = bandpass(rng.normal(size=N), 250, 900) * 0.02 * bed_env * duck
L += air; R += np.roll(air, 900)

# ---------- cues ----------
place(boom(2.0, 95, 38, 0.45), 0.35, 0, 0.4)                                  # lights out
notes = [1046.5, 1174.66, 1318.51, 1567.98, 1760.0, 2093.0]
for k in range(10):                                                           # equations appear
    place(chime([notes[k % 6], notes[k % 6] * 2.01], 1.2, 0.45), 1.0 + 0.25 * k, math.sin(k * 2.1) * 0.7, 0.07)
w = sweep_noise(0.6, 500, 5000, 0.4)                                          # hand swipe
place(env_shape(w, 0.05, 0.18), 5.0, -0.6, 0.22)
for k in range(70):                                                           # glassy shatter
    st = 5.12 + rng.uniform(0, 0.45)
    f = rng.uniform(2200, 7000)
    tt = np.arange(int(0.25 * SR)) / SR
    place(np.sin(2 * np.pi * f * tt) * np.exp(-tt / rng.uniform(0.03, 0.12)), st, -0.8 + 1.6 * (st - 5.12) / 0.45, 0.035)
nb = bandpass(rng.normal(size=int(0.8 * SR)), 2000, 9000)
place(env_shape(nb / np.abs(nb).max(), 0.005, 0.2), 5.12, 0, 0.16)
for st, pan in ((10.68, -0.7), (10.72, 0.7)):                                 # hands ignite
    w = sweep_noise(0.9, 700, 4500, 0.4)
    place(env_shape(w, 0.04, 0.22), st, pan, 0.16)
    place(chime([1567.98, 2349.32, 3135.96], 2.0, 0.8, 0.002), st + 0.03, pan, 0.13)
g = sweep_noise(0.8, 3000, 400, 0.4)                                          # dust gathers into the hands
tt = np.arange(len(g)) / SR
place(g * (tt / 0.8) ** 2 * np.clip((0.8 - tt) / 0.03, 0, 1), 10.9, 0, 0.12)
place(chime([659.25, 987.77, 1318.51, 1975.53], 3.0, 1.3), 12.0, 0, 0.18)    # orb appears
w = sweep_noise(0.7, 300, 2500, 0.45)
place(env_shape(w, 0.1, 0.25), 11.92, 0, 0.1)
hum_env = sstep(12.2, 12.8, t) * (1 - sstep(19.2, 19.6, t))
hf = 196 * (1 + 0.12 * sstep(14.5, 17.5, t))
hum = (np.sin(2 * np.pi * np.cumsum(hf) / SR) + 0.5 * np.sin(2 * np.pi * np.cumsum(hf * 1.5) / SR)) * hum_env * (1 + 0.3 * np.sin(2 * np.pi * 4.5 * t)) * 0.02 * duck
L += hum; R += hum
rd = 1.15                                                                     # riser into "it became clear"
rs = sweep_noise(rd, 400, 6000, 0.3); tt = np.arange(len(rs)) / SR
cres = (tt / rd) ** 2.5 * np.clip((rd - tt) / 0.02, 0, 1)
place(rs * cres, 17.55 - rd, 0, 0.16)
place(np.sin(2 * np.pi * np.cumsum(220 * 3 ** (tt / rd)) / SR) * cres, 17.55 - rd, 0, 0.05)
place(boom(2.0, 70, 34, 0.5), 17.55, 0, 0.35)                                 # the hit (in the pause)
place(chime([880.0, 1318.51, 1760.0, 2637.02], 3.0, 1.5), 17.55, -0.2, 0.15)
s = sweep_noise(0.55, 600, 3500, 0.35); tt = np.arange(len(s)) / SR           # orb compresses
place(s * (tt / 0.55) ** 2, 19.0, 0, 0.08)
z = sweep_noise(0.3, 1500, 7000, 0.3); tt = np.arange(len(z)) / SR            # zip into the head
place(z * np.sin(np.pi * tt / 0.3), 19.55, 0, 0.14)
place(chime([1318.51, 1975.53, 2637.02, 3951.07], 2.5, 1.0), 19.85, 0, 0.22)  # ding
place(boom(1.5, 80, 40, 0.3), 19.85, 0, 0.2)
for k in range(9):                                                            # hologram flicker
    st = 20.0 + k * 0.05 + rng.uniform(0, 0.02)
    tt = np.arange(int(0.035 * SR)) / SR
    f = rng.choice([1200, 1800, 2400, 3000])
    place(np.sign(np.sin(2 * np.pi * f * tt)) * 0.5 * np.exp(-tt / 0.02), st, rng.uniform(-0.5, 0.5), 0.05)
place(chime([523.25, 783.99, 1046.5, 1567.98], 3.0, 1.6, 0.003), 20.45, 0, 0.12)

# ---------- reverb on the effects only ----------
ir_n = int(2.0 * SR); irt = np.arange(ir_n) / SR
ir = lowpass(rng.normal(size=ir_n), 5000) * np.exp(-irt / 0.5); ir /= np.sqrt((ir ** 2).sum())
fxL = L + 0.3 * signal.fftconvolve(L, ir)[:N]
fxR = R + 0.3 * signal.fftconvolve(R, np.roll(ir, 311))[:N]
fade = 1 - sstep(DUR - 0.45, DUR, t)
out = np.stack([(v + fxL) * fade, (v + fxR) * fade], 1)
out = signal.sosfilt(signal.butter(2, 28, 'high', fs=SR, output='sos'), out, axis=0)
pk = np.abs(out).max()
if pk > 0.95:
    out *= 0.95 / pk
wavfile.write(sys.argv[2], SR, (out * 32767).astype(np.int16))
print('wrote', sys.argv[2], f'{DUR:.2f}s')
