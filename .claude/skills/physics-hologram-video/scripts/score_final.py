"""Subtle score for the apple / time video. usage: python3 score_final.py <work_dir> voice.wav out.wav"""
import sys, os, math
import numpy as np
from scipy.io import wavfile
from scipy import signal
sys.path.insert(0, os.path.dirname(__file__))
import sfx_subtle as S

V = sys.argv[1]
sr, v = wavfile.read(sys.argv[2]); v = v.astype(np.float64) / 32768
if v.ndim == 2: v = v.mean(1)
DUR = len(v) / sr
spk = np.abs(v) > 10 ** (-40 / 20)
v *= 10 ** (-20 / 20) / np.sqrt((v ** 2)[np.convolve(spk, np.ones(2400), 'same') > 0].mean())
tl = S.Timeline(DUR)
T_PLUCK, T_RELEASE, T_KNOB0, T_KNOB1, T_DIST, T_STOP, T_RING, T_ARROW, T_VQ = 1.30, 4.22, 2.30, 3.20, 8.8, 13.7, 15.2, 17.45, 18.35
env = lambda d, a, r: S.sstep(0, a, S.tt(d)) * (1 - S.sstep(d - r, d, S.tt(d)))
def rate(t):
    if t < T_KNOB0: return 1.0
    if t < T_KNOB1 + 0.3: return 1 - 0.65 * S.sstep(T_KNOB0, T_KNOB1, t)
    if t < T_STOP: return 0.35 - 0.12 * S.sstep(4.4, 9.0, t)
    return 0.0

# lights out + tree projection
tl.add(S.low_swell(1.6, 0.3), 0.0, -8)
sp = S.sparkle(1.0, 120) * np.linspace(0.2, 1, int(1.0 * S.SR))[:, None]
tl.add(sp, 0.15, -13)
# leaf rustle driven by the raised hand's speed while it is inside the crown (until the time stop)
P = np.load(V + '/palms.npy')
spd = np.r_[0, np.linalg.norm(np.diff(P[:, 0], axis=0), axis=1) * 25]
inside = (P[:, 0, 1] < 760) & (P[:, 0, 0] < 930)
e = np.clip(spd / 900, 0, 1) * inside
e = np.interp(np.arange(len(tl.buf)) / S.SR * 25, np.arange(len(e)), e)
e = np.convolve(e, np.ones(2400) / 2400, 'same') * (np.arange(len(e)) / S.SR < T_STOP)
rust = S.bandnoise(DUR, 2500, 9000) * S.flutter(len(e), 17, 0.8)
tl.buf += S.wide(rust * e * 0.5, 1.3, 9)
# pluck: tiny snap + glassy bloom; camera punch-in swish
g = S.tt(0.03); tl.add(S.pan(np.sin(2 * np.pi * 3800 * g) * np.exp(-g / 0.004), -0.3), T_PLUCK, -10)
tl.add(S.bloom(2.2, 1568.0), T_PLUCK, -12)
tl.add(S.swish(0.35, 800, 4500, -0.4, 0.2, 0.4), 1.27, -11)
# camera swing to the knob and back
tl.add(S.swish(0.35, 900, 3500, -0.3, 0.6, 0.4), 2.12, -12)
tl.add(S.swish(0.45, 2800, 700, 0.6, -0.2, 0.4), 3.3, -13)
# knob: ratchet clicks while it turns, pitch-down sweep, amber ripples
for k in range(18):
    t0 = T_KNOB0 + (T_KNOB1 - T_KNOB0) * (k / 17) ** 0.9
    g = S.tt(0.02); tl.add(S.pan(np.sin(2 * np.pi * (2400 - 60 * k) * g) * np.exp(-g / 0.003), 0.55), t0, -15)
tl.add(S.swish(1.0, 3000, 250, 0.4, 0.0, 0.45), 2.35, -9)
# stopwatch ticks: one per simulated second-step, getting further apart as time slows, stopping at the freeze
ts = 0.0; t = 2.1; nxt = 0.25
while t < T_STOP:
    ts += rate(t) / 200; t += 1 / 200
    if ts >= nxt:
        nxt += 0.25
        g = S.tt(0.02)
        tl.add(S.pan(np.sin(2 * np.pi * (3200 if int(nxt * 4) % 2 else 2700) * g) * np.exp(-g / 0.004), 0.75), t, -19)
# drop: soft downward swish + camera pull-out
tl.add(S.swish(0.6, 2500, 400, -0.2, -0.2, 0.45), 4.15, -10)
tl.add(S.swish(0.4, 3000, 900, 0.2, -0.4, 0.4), 4.25, -14)
# the slow fall: fluttering shimmer bed; a soft ping as each copy is left behind
d = T_STOP - 4.4; tl.add(S.shimmer(d, 1300, n=6, rate=7, air=0.4, env=env(d, 0.8, 0.05) * 0.5), 4.4, -12)
Y0, YF = 318.0, 775.0
for k, gt in enumerate([5.6, 6.6, 7.6, 9.4]):
    g = S.tt(1.0)
    tl.add(S.wide(np.sin(2 * np.pi * (1760 - 110 * k) * g) * np.exp(-g / 0.35) * 0.6, 1.0, 9), gt, -17)
# the dotted scale draws in ("டிஸ்டன்ஸ்")
sp = S.sparkle(0.8, 90); sp *= np.linspace(1, 0.2, len(sp))[:, None]
tl.add(sp, T_DIST, -13)
# time stop: reverse swell -> hit -> hush, then a frozen sustained tone
rs = S.swish(0.6, 300, 4000, 0, 0, 0.5); tt_ = np.arange(len(rs)) / len(rs)
tl.add(rs * (tt_ ** 2)[:, None], T_STOP - 0.6, -8)
tl.add(S.bloom(3.0, 880.0), T_STOP, -8)
tl.add(S.low_swell(1.2, 0.08), T_STOP - 0.05, -7)
tl.add(S.swish(0.3, 1200, 6000, 0, 0, 0.35), T_STOP - 0.1, -12)            # snap-zoom
d = DUR - T_STOP - 0.3
fz = S.wide(sum(np.sin(2 * np.pi * f * S.tt(d) + k) for k, f in enumerate((1318.5, 1975.5, 2637.0))) / 3 * env(d, 0.6, 0.6), 1.4, 13)
tl.add(fz, T_STOP + 0.3, -21)
cut = np.ones(len(tl.buf)); i0 = int(T_STOP * S.SR)
cut[i0:] = 0.35 + 0.65 * np.exp(-np.arange(len(cut) - i0) / S.SR / 0.25)   # hush everything else after the freeze
# ring, arrow, question
tl.add(S.bloom(2.0, 1046.5), T_RING, -12)
tl.add(S.swish(0.5, 2500, 600, 0, 0, 0.4), T_ARROW, -11)
g = S.tt(1.2); tl.add(S.wide(np.sin(2 * np.pi * 1568 * g) * np.exp(-g / 0.4) * 0.7, 1.0, 9), T_VQ, -12)
tl.add(S.swish(0.6, 2000, 500, 0, 0, 0.4), 17.0, -16)                          # slow pull-out

fx = S.reverb(tl.buf, 2.4, 0.4)
fx[:len(cut)] *= cut[:len(fx), None] if len(cut) >= len(fx) else 1
act = np.convolve(spk.astype(float), np.ones(4800) / 4800, 'same')[:len(fx)]
fx *= 10 ** (-19 / 20) * np.sqrt((v ** 2).mean() * 4) / (np.sqrt((fx ** 2).mean()) + 1e-12)
fx *= (1 - 0.3 * np.clip(act * 2, 0, 1))[:, None]
v = v[:len(fx)]
mix = fx + v[:, None]
n = len(mix); f = np.ones(n); f[-int(0.3 * S.SR):] = np.linspace(1, 0, int(0.3 * S.SR))
mix *= f[:, None]; mix *= 0.89 / np.abs(mix).max()
wavfile.write(sys.argv[3], S.SR, (mix * 32767).astype(np.int16))
print('wrote', sys.argv[3])
