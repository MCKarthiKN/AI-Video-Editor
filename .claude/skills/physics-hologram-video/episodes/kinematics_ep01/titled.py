"""Episode with heading title sequences between sections + outro buffer.
Per boundary: 1.0 s hold of the last frame (gentle dim + glow) -> heading card (0.5 s ease-in, >=1.5 s static) ->
0.5 s cross-dissolve into the next section. End: 2.0 s hold, then 1.0 s fade to black."""
import sys, math, subprocess
import numpy as np, cv2
from scipy.io import wavfile
sys.path.insert(0, '/home/user/AI-Video-Editor/.claude/skills/physics-hologram-video/scripts')
import sfx_subtle as S
from fx import *

O = f'{D}/out'
SECTIONS = [  # (sources [(file, first_frame, n_frames or None)], audio frames, heading label, heading, subtitle)
    ([(f'{O}/opening.mp4', 0, None)], 1101, None, None, None),
    ([(f'{O}/guess.mp4', 0, None)], 481, 'PART 2 · GUESS', 'AT REST OR MOVING?', 'PICK ONE BEFORE YOU WATCH ON'),
    ([(f'{D}/r3d/train.mp4', 0, None), (f'{O}/toy.mp4', 0, None), (f'{O}/idea.mp4', 0, None)], 793 + 837 + 441,
     'PART 3 · SEE IT', 'ONE FRIEND, TWO ANSWERS', 'A TRAIN · A TOY · A TABLE'),
    ([(f'{O}/frame.mp4', 0, None), (f'{O}/book.mp4', 0, None)], 785 + 431, 'PART 4 · NAME IT', 'FRAME OF REFERENCE', 'OBSERVER · RULER · CLOCK'),
    ([(f'{O}/position.mp4', 0, None)], 3592, 'PART 5 · SAY IT IN MATHS', 'POSITION', 'ORIGIN · DIRECTION · UNIT'),
    ([(f'{O}/origins.mp4', 0, None)], 2423, 'PART 6 · TEST IT', 'TWO ORIGINS, ONE CAR', 'CHEEMS · PERRO · ONE CAR'),
    ([(f'{O}/ending_a.mp4', 0, None), (f'{O}/ending_b.mp4', 0, 872), (f'{O}/ending_c.mp4', 0, None)], 2403,
     'PART 7 · WONDER AGAIN', 'IS ANYTHING TRULY AT REST?', 'EARTH · SUN · GALAXY'),
]
HOLD, EASE, STATIC, DISS, OUT_HOLD, OUT_FADE = 25, 12, 38, 13, 50, 25      # frames @25 fps

def frames(path, first=0, n=None):
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', path, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
    k = 0
    while True:
        b = p.stdout.read(W * H * 3)
        if len(b) < W * H * 3: break
        if k >= first and (n is None or k < first + n): yield np.frombuffer(b, np.uint8).reshape(H, W, 3)
        k += 1
        if n is not None and k >= first + n: break
    p.kill()

BG = void(W / 2, H / 2, 720, .7)
def card(label, head, sub, u):
    """u: 0..1 ease-in progress (1 = settled, static)"""
    e = eout(u); img = BG.copy(); G = np.zeros_like(img)
    G += (np.exp(-((XX - W / 2) ** 2 / (2 * 520 ** 2) + (YY - H / 2) ** 2 / (2 * 200 ** 2))) * .22 * e)[..., None] * np.array(CYAN, np.float32)
    dy = 26 * (1 - e)
    T_(G, label, (W / 2, 410 + dy), 34, AMBER, 'Medium', amp=e, track=.22)
    size = 96 if len(head) < 18 else 76
    T_(G, head, (W / 2, 520 + dy), size, WHITE, 'Bold', amp=e, scale=.96 + .04 * e, track=.06, halo=.7, sblur=7)
    m = M(); L = 380 * e; cv2.line(m, (int(W / 2 - L), 600), (int(W / 2 + L), 600), 255, 2, cv2.LINE_AA); glow(G, m, CYAN, .9, .9, 5, e)
    T_(G, sub, (W / 2, 650 + dy * .6), 30, CYAN, 'Light', amp=e * seg(u, .3, 1), track=.16)
    g = np.clip(G, 0, 2); out = 1 - (1 - img) * (1 - np.clip(g, 0, 1)) + np.clip(g - 1, 0, 1) * .25
    return (np.clip(out, 0, 1) * 255 + .5).astype(np.uint8)
def hold_frame(last, k, n):
    """gentle fade + ambient glow out on the frozen last frame"""
    u = (k + 1) / n; f = last.astype(np.float32) / 255
    bloom = cv2.GaussianBlur(f, (0, 0), 18)
    out = f * (1 - .55 * ease(u)) + bloom * .35 * math.sin(math.pi * min(1, u * 1.2)) + BG * .5 * ease(u)
    return (np.clip(out, 0, 1) * 255 + .5).astype(np.uint8)

wr = Writer(f'{D}/titled_video.mp4')
plan = []      # audio plan: ('src', audio_frame_start, n) or ('gap', n, card_start_offset)
afr = 0; nfr = 0
for si, (srcs, an, label, head, sub) in enumerate(SECTIONS):
    if label:
        for k in range(HOLD): wr.write(hold_frame(last, k, HOLD))
        for k in range(EASE): wr.write(card(label, head, sub, (k + 1) / EASE))
        static = card(label, head, sub, 1.0)
        for k in range(STATIC): wr.write(static)
        plan.append(('gap', HOLD + EASE + STATIC)); nfr += HOLD + EASE + STATIC
    k = 0
    for path, first, n in srcs:
        for f in frames(path, first, n):
            if label and k < DISS:
                a = ease((k + 1) / (DISS + 1)); f = (static.astype(np.float32) * (1 - a) + f.astype(np.float32) * a + .5).astype(np.uint8)
            wr.write(f); last = f; k += 1
    print(si, 'section frames', k, 'audio frames', an, flush=True)
    plan.append(('src', afr, k)); afr += an; nfr += k
for k in range(OUT_HOLD): wr.write(last)
for k in range(OUT_FADE): wr.write((last.astype(np.float32) * (1 - (k + 1) / OUT_FADE)).astype(np.uint8))
nfr += OUT_HOLD + OUT_FADE
wr.close()

# ---------------------------------------------------------------- audio
SR = 48000; fr = SR // 25
sr, mix = wavfile.read(f'{D}/mix.wav'); mix = mix.astype(np.float64) / 32768
out = np.zeros((nfr * fr + SR, 2)); pos = 0; ff = int(.02 * SR)
for item in plan:
    if item[0] == 'gap':
        n = item[1] * fr
        out[pos + HOLD * fr:pos + HOLD * fr + int(3.0 * SR)] += S.bloom(3.0, 523.3)[:int(3.0 * SR)] * 10 ** (-16 / 20)        # soft bloom as the heading eases in
        sw = S.swish(.8, 300, 1800, -.4, .4, .5); out[pos + int((HOLD - 8) * fr):pos + int((HOLD - 8) * fr) + len(sw)] += sw * 10 ** (-22 / 20)
        pos += n
    else:
        _, a0, n = item; x = mix[a0 * fr:(a0 + n) * fr].copy()
        if len(x) < n * fr: x = np.r_[x, np.zeros((n * fr - len(x), 2))]
        x[:ff] *= np.linspace(0, 1, ff)[:, None]; x[-ff:] *= np.linspace(1, 0, ff)[:, None]
        out[pos:pos + n * fr] += x; pos += n * fr
# outro: ambient shimmer under the 2 s hold, fading with the picture
d = (OUT_HOLD + OUT_FADE) / 25; e = 1 - S.sstep(OUT_HOLD / 25, d, S.tt(d))
out[pos:pos + int(d * SR)] += S.shimmer(d, 1046.5, n=5, rate=6, air=.3, env=e * .4)[:int(d * SR)] * 10 ** (-24 / 20)
out = out[:nfr * fr]
wavfile.write(f'{D}/titled_audio.wav', SR, (np.clip(out, -1, 1) * 32767).astype(np.int16))
print('total frames', nfr, nfr / 25)
