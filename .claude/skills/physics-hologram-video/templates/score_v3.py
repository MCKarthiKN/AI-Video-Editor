"""Subtle, event-driven score for the 'picture in the head' clip. usage: python3 score_v3.py voice.wav out.wav"""
import sys, os, math
import numpy as np
from scipy.io import wavfile
sys.path.insert(0, os.path.dirname(__file__))
import sfx_subtle as S

sr, v = wavfile.read(sys.argv[1]); v = v.astype(np.float64) / 32768
if v.ndim == 2: v = v.mean(1)
DUR = len(v) / sr
spk = np.abs(v) > 10 ** (-40 / 20)
v *= 10 ** (-20 / 20) / np.sqrt((v ** 2)[np.convolve(spk, np.ones(2400), 'same') > 0].mean())
tl = S.Timeline(DUR)
env = lambda d, a, r: S.sstep(0, a, S.tt(d)) * (1 - S.sstep(d - r, d, S.tt(d)))

# lights go out
tl.add(S.low_swell(1.6, 0.35), 0.05, -8)
# emotion: warm, slowly beating pad + soft heartbeat while the rings pulse
d = 5.0
tl.add(S.vibration(d, 220, 1.2, env=env(d, 1.0, 0.8)), 0.9, -16)
for k in range(int((5.0 - 2.9) * 1.2) + 1):          # heartbeat at 72 bpm from "எமோஷன்"
    for off, gdb in ((0.0, -9), (0.16, -13)):
        tt = S.tt(0.25)
        th = np.sin(2 * np.pi * 52 * tt) * np.exp(-tt / 0.06)
        tl.add(S.wide(th, 0.2, 3), 2.9 + k / 1.2 + off, gdb)
# heart rises into the head, soft bloom on arrival ("மண்டை")
tl.add(S.swish(0.8, 500, 4500, 0.0, 0.0, 0.4), 5.0, -8)
tl.add(S.bloom(2.8, 1046.5), 5.75, -10)
# the picture opens ("பிக்சர்"): glitter, then a quiet fluttering shimmer that lives with it
sp = S.sparkle(0.8, 100); sp *= S.flutter(len(sp), 14, 0.7)[:, None] * np.sin(np.linspace(0, math.pi, len(sp)))[:, None]
tl.add(sp, 6.2, -10)
d = DUR - 6.5
tl.add(S.shimmer(d, 1750, env=0.45 * env(d, 1.0, 0.6)), 6.5, -10)
# the person appears in the picture
tl.add(S.bloom(2.0, 1568.0), 8.1, -15)
# speaks to us ("பேசுவாரு"): three wide whooshes travelling out towards the listener
for k, ts in enumerate((13.35, 13.65, 13.95)):
    tl.add(S.swish(1.2, 2600, 500, -0.2 * (k - 1), 0.9 * (1 if k % 2 else -1), 0.5), ts - 0.15, -8 - 2 * k)
# our picture: frame appears, then slow "pixel" ticks while it assembles; ticks thin out as it stalls
tl.add(S.sparkle(0.5, 80), 14.6, -14)
rng = np.random.default_rng(4)
t = 15.5
while t < 20.4:
    rate = 9 if t < 17.6 else 4
    g = np.arange(int(0.03 * S.SR)) / S.SR
    tl.add(S.pan(np.sin(2 * np.pi * rng.uniform(2500, 4500) * g) * np.exp(-g / 0.006), 0.6 + rng.uniform(-0.2, 0.2)), t, -20)
    t += rng.exponential(1 / rate)
# clock ("டைம்ல"): quiet tick-tock on the right
for k in range(int((22.2 - 19.8) * 4)):
    g = np.arange(int(0.02 * S.SR)) / S.SR
    f = 3200 if k % 2 == 0 else 2700
    tl.add(S.pan(np.sin(2 * np.pi * f * g) * np.exp(-g / 0.004), 0.65), 19.8 + k * 0.25, -17)
# "ரொம்ப டிலே": glitchy flutter + low swell under the stall
gl = S.shimmer(0.8, 900, n=4, rate=22, air=0.9, env=env(0.8, 0.05, 0.4))
tl.add(gl, 21.2, -11)
tl.add(S.low_swell(1.4, 0.3), 20.9, -9)
# "அதனாலதான்…": freeze swell and fade
tl.add(S.swish(0.9, 300, 3000, 0.0, 0.0, 0.5), 21.9, -8)
tl.add(S.low_swell(1.0, 0.5), 22.0, -8)

fx = S.reverb(tl.buf, 2.4, 0.4)
act = np.convolve(spk.astype(float), np.ones(4800) / 4800, 'same')[:len(fx)]
fx *= 10 ** (-20 / 20) * np.sqrt((v ** 2).mean() * 4) / (np.sqrt((fx ** 2).mean()) + 1e-12)
fx *= (1 - 0.3 * np.clip(act * 2, 0, 1))[:, None]                     # dip slightly under the words
v = v[:len(fx)]
mix = fx + v[:, None]
n = len(mix); fade = np.ones(n); fade[-int(0.45 * S.SR):] = np.linspace(1, 0, int(0.45 * S.SR))
mix *= fade[:, None]
mix *= 0.89 / np.abs(mix).max()
wavfile.write(sys.argv[2], S.SR, (mix * 32767).astype(np.int16))
print('wrote', sys.argv[2])
