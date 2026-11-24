"""Reel audio: episode mix cut per reel plan + rewind swish + CTA bloom."""
import sys, subprocess
import numpy as np
from scipy.io import wavfile
sys.path.insert(0, '/home/user/AI-Video-Editor/.claude/skills/physics-hologram-video/scripts')
import sfx_subtle as S
from reel import plan, DUR
from fx import D
SR = 48000
raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', f'{D}/E01_X_final.mp4', '-vn', '-ac', '2', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True).stdout
A = np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)
out = np.zeros((int(round(DUR * 25)) * SR // 25 + SR, 2))
f = int(.015 * SR)
for p in plan:
    i0 = int(round(p['r0'] * 25)) * SR // 25; n = (int(round(p['r1'] * 25)) - int(round(p['r0'] * 25))) * SR // 25
    if p['name'] == 'rewind':
        x = S.swish(n / SR, 4000, 300, .6, -.6, .5) * .5; out[i0:i0 + len(x)] += x[:n]; continue
    if p['name'] == 'cta':
        x = S.bloom(2.6, 523.3) * .5 + np.r_[S.sparkle(2.0, 70), np.zeros((int(.6 * SR), 2))] * .25; out[i0:i0 + min(len(x), len(out) - i0)] += x[:len(out) - i0]; continue
    a0 = int(p['a'] * SR); x = A[a0:a0 + n].copy()
    x[:f] *= np.linspace(0, 1, f)[:, None]; x[-f:] *= np.linspace(1, 0, f)[:, None]
    out[i0:i0 + len(x)] += x
out = out[:int(round(DUR * 25)) * SR // 25]
out *= .9 / np.abs(out).max()
wavfile.write(f'{D}/reel/reel_audio.wav', SR, (out * 32767).astype(np.int16)); print('ok', len(out) / SR)
