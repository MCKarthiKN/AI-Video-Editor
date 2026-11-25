"""Voice edit + event-driven subtle score for Kinematics Ep 1. Writes voice_edit.wav and mix.wav (48 kHz stereo)."""
import sys, os, json, subprocess, math
import numpy as np
from scipy.io import wavfile
from scipy import signal
sys.path.insert(0, '/home/user/AI-Video-Editor/.claude/skills/physics-hologram-video/scripts')
import sfx_subtle as S
def mavg(x, n):
    c = np.cumsum(np.r_[0.0, x]); y = (c[n:] - c[:-n]) / n; pad = len(x) - len(y); return np.r_[np.full(pad // 2, y[0]), y, np.full(pad - pad // 2, y[-1])]
from fx import EDL, T, beat_span, D
SR = S.SR

# ---------------------------------------------------------------- voice: cut per EDL with 12 ms fades
cache, out = {}, []
for p in EDL['pieces']:
    c = p['clip']
    if c not in cache:
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', f"{D}/in/{c}.mov", '-vn', '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True).stdout
        cache[c] = np.frombuffer(raw, np.float32).astype(np.float64)
    n = int(round((p['t'] + p['o'] - p['i']) * 25)) - int(round(p['t'] * 25))      # match the video frame count exactly
    i0 = int(p['i'] * SR); x = cache[c][i0:i0 + int(n / 25 * SR)].copy()
    if len(x) < int(n / 25 * SR): x = np.r_[x, np.zeros(int(n / 25 * SR) - len(x))]
    f = int(.012 * SR); x[:f] *= np.linspace(0, 1, f); x[-f:] *= np.linspace(1, 0, f); out.append(x)
v = np.concatenate(out)
v = signal.sosfilt(signal.butter(2, 80, 'high', fs=SR, output='sos'), v)
# gentle levelling: 400 ms RMS follower -> soft gain toward a target (keeps quiet words up, loud ones down)
win = int(.4 * SR); rms = np.sqrt(mavg(v ** 2, win) + 1e-9)
spk = rms > 10 ** (-42 / 20)
tgt = np.sqrt((v[spk] ** 2).mean())
g = np.clip((tgt / rms) ** .35, .6, 1.8); g[~spk] = 1.0
g = signal.sosfiltfilt(signal.butter(1, 3, 'low', fs=SR, output='sos'), g)
v *= g; v *= 10 ** (-20 / 20) / np.sqrt((v[spk] ** 2).mean())
wavfile.write(f'{D}/voice_edit.wav', SR, (np.clip(v, -1, 1) * 32767).astype(np.int16))
DUR = len(v) / SR
print('voice', DUR)

# ---------------------------------------------------------------- cues
tl = S.Timeline(DUR + 2)
def ping(t, f=1760, db=-17, d=1.0):
    g = S.tt(d); tl.add(S.wide(np.sin(2 * np.pi * f * g) * np.exp(-g / .3) * .6, 1.0, 9), t, db)
def bloom(t, f=880, db=-13, d=2.2): tl.add(S.bloom(d, f), t, db)
def swish(t, d=.6, f0=600, f1=4000, db=-14, p0=-.5, p1=.5): tl.add(S.swish(d, f0, f1, p0, p1, .5), t, db)
def sparkle(t, d=.8, db=-15): sp = S.sparkle(d, 90); sp *= np.linspace(1, .2, len(sp))[:, None]; tl.add(sp, t, db)
def swell(t, d=1.6, db=-9): tl.add(S.low_swell(d, .4), t, db)
def bed(t0, t1, f=1300, db=-22):
    d = t1 - t0; e = S.sstep(0, .8, S.tt(d)) * (1 - S.sstep(d - .8, d, S.tt(d)))
    tl.add(S.shimmer(d, f, n=6, rate=7, air=.4, env=e * .5), t0, db)
def ticks(t0, t1, rate0, rate1, db=-24):
    t = t0
    while t < t1:
        u = (t - t0) / (t1 - t0); g = S.tt(.05)
        tl.add(S.wide(signal.sosfilt(signal.butter(2, [2500, 7000], 'band', fs=SR, output='sos'), np.random.default_rng(int(t * 100)).normal(size=len(g))) * np.exp(-g / .008), .6, 5), t, db)
        t += 1 / (rate0 + (rate1 - rate0) * u)
NOTES = [1318.5, 1568, 1760, 1975.5, 2093]
E = lambda b, c: T(b, c)
# opening
swell(.1, 1.8, -8); sparkle(.3, 1.0, -16)
bloom(E('opening', 13.3), 659.3, -13); sparkle(E('opening', 13.4), 1.2, -17)
ticks(E('opening', 19.6), E('opening', 21.6), 14, 4); ticks(E('opening', 21.6), E('opening', 30.6), 2, 2, -27)
for k, c in enumerate((23.4, 25.0, 26.7, 28.6)): ping(E('opening', c), NOTES[k], -18)
swish(E('opening', 30.9), 2.2, 2500, 400, -12, .6, -.6)
bloom(E('opening', 36.7), 523.3, -15); sparkle(E('opening', 39.4), 1.4, -16)
eo = 28.7; swell(eo, 2.2, -9); bed(eo + .5, eo + 15.2, 1100, -22)
swish(eo + 1.4, 3.4, 300, 1800, -15, -.4, .4)                     # push-in
bed(eo + 5.3, eo + 9.1, 1650, -20)                                # the 10-minute trace
bloom(eo + 9.1, 880, -11); sparkle(eo + 9.1, .9, -14)              # 250 KM
swish(eo + 12.0, 3.0, 2000, 400, -14, .4, -.4)                     # pull-back
# guess
swish(E('guess', 12.4), .5, 800, 3000, -15); ping(E('guess', 13.5), 1318.5, -18); ping(E('guess', 14.4), 1568, -18)
ping(E('guess', 16.4), 2093, -19); sparkle(E('guess', 24.4), .9, -15); bloom(E('guess', 24.5), 784, -15)
# train: rumble + wheel joints while riding, wipe, pass-by on the platform
t0, t1 = beat_span('train')
d = t1 - t0; rum = signal.sosfilt(signal.butter(2, [45, 180], 'band', fs=SR, output='sos'), np.random.default_rng(3).normal(size=int(d * SR)))
tt_ = S.tt(d); clack = sum(np.exp(-((tt_ - k) % 1.0) / .03) * (k % 2 * .3 + .7) for k in range(2)) * .5
e = np.where(tt_ < 14.2, .6 + .4 * clack, np.exp(-((tt_ - 21.8) / 2.6) ** 2) * 1.6 + .08) * S.sstep(0, .8, tt_) * (1 - S.sstep(d - .8, d, tt_))
tl.add(S.wide(rum * e * .5, 1.2, 11), t0, -14)
swish(t0 + 13.9, .7, 500, 5000, -12)
pb = S.swish(3.0, 300, 1500, -1, 1, .6); tl.add(pb, t0 + 21.8 - 1.5, -9)
for c in (10.2, 18.6, 24.6): ping(t0 + c, 1760 if c != 18.6 else 1568, -18)
bloom(t0 + 29.4, 659.3, -15)
# toy split
bloom(E('toy', 61.0), 784, -15); swish(E('toy', 67.6), .8, 600, 3500, -13)
g = S.tt(3.0); tl.add(S.wide(np.sin(2 * np.pi * np.cumsum(500 + 500 * g / 3) / SR) * S.sstep(0, .5, g) * (1 - S.sstep(2.5, 3, g)) * .3, 1.0, 11), E('toy', 82.5), -19)
ping(E('toy', 77.0), 1568, -18); ping(E('toy', 84.0), 2093, -18)
# big idea
bloom(E('idea', 94.4), 523.3, -14)
for k, c in enumerate((97.6, 101.0, 101.6)): ping(E('idea', c), NOTES[k + 1], -19)
sparkle(E('idea', 108.5), .9, -16); bloom(E('idea', 110.0), 659.3, -13)
# frame of reference kit + textbook
sparkle(E('frame', 5.9), 1.2, -15); bloom(E('frame', 6.0), 523.3, -14)
for k in range(3): ping(E('frame', 14.7) + k * .25, NOTES[k], -20)
for k, c in enumerate((16.6, 31.4, 36.4)): bloom(E('frame', c), [659.3, 784, 987.8][k], -16); ping(E('frame', c), NOTES[k + 1], -19)
swish(E('frame', 40.2), .6, 700, 3200, -15); bloom(E('frame', 48.0), 523.3, -14)
sparkle(E('book', 86.3), 1.6, -14); bloom(E('book', 87.6), 659.3, -14)
# number line
for k, c in enumerate((10.0, 18.6, 34.3, 46.6, 61.6, 68.5, 72.9, 90.9, 112.4, 115.1, 117.7, 141.3)): ping(E('position', c), NOTES[k % 5], -19)
for k in range(4): ping(E('position', 42.4) + k * .12, 2637 - k * 100, -24, .6)
for k in range(4): ping(E('position', 82.2) + k * 1.2, 2637 - k * 100, -24, .6)
bloom(E('position', 18.8), 784, -16); bloom(E('position', 106.6), 523.3, -13); bloom(E('position', 129.0), 659.3, -15)
ping(E('position', 145.4), 2093, -17); bloom(E('position', 150.5), 880, -12); sparkle(E('position', 150.5), .8, -15)
# two origins
for k, c in enumerate((7.1, 9.6, 25.6, 44.9)): ping(E('origins', c), NOTES[k], -18)
swish(E('origins', 49.6), 1.6, 400, 2200, -13, -.6, .6)
bloom(E('origins', 68.8), 659.3, -14); bloom(E('origins', 74.6), 659.3, -14); sparkle(E('origins', 75.0), .8, -16)
swish(E('origins', 79.3), .5, 700, 3000, -16); bloom(E('origins', 90.6), 523.3, -13); bloom(E('origins', 94.0), 784, -14)
# ending: cosmic cutaways
eb = beat_span('ending')[0]
bloom(E('ending', 10.8), 784, -16)
swell(eb + 5.8, 2.0, -11); bed(eb + 6.2, eb + 14.4, 1200, -23)
swell(eb + 14.1, 2.4, -8); bed(eb + 14.4, eb + 30.0, 1000, -21); swish(eb + 17.0, 3.5, 300, 2000, -14, -.6, .6)
bloom(eb + 17.9, 880, -12); ping(eb + 24.8, 1318.5, -19); ping(eb + 25.6, 1975.5, -17)
swell(eb + 30.0, 2.0, -10); sparkle(eb + 30.2, 2.5, -17); bed(eb + 30.2, eb + 36.6, 1600, -22)
bloom(E('ending', 55.2), 523.3, -14); bloom(E('ending', 64.2), 659.3, -15)
bloom(E('ending', 93.5), 523.3, -12); sparkle(E('ending', 93.6), 1.0, -15)
swish(E('ending', 98.4), .6, 600, 3000, -15); ping(E('ending', 107.0), 1568, -18)
bloom(E('ending', 113.5), 659.3, -14); ping(E('ending', 118.4), 1975.5, -18)
bloom(E('ending', 125.1), 523.3, -11); sparkle(E('ending', 125.2), 2.0, -16)

fx = S.reverb(tl.buf, 2.4, .4)[:len(v)]
act = mavg(spk.astype(float), 4800)
fx *= (1 - .3 * np.clip(act * 2, 0, 1))[:, None]          # tuck effects under speech
mix = fx * 10 ** (-1 / 20) + v[:, None]
f = np.ones(len(mix)); f[-int(.8 * SR):] = np.linspace(1, 0, int(.8 * SR)); mix *= f[:, None]
mix *= .89 / np.abs(mix).max()
wavfile.write(f'{D}/mix.wav', SR, (mix * 32767).astype(np.int16)); print('mix', len(mix) / SR)
