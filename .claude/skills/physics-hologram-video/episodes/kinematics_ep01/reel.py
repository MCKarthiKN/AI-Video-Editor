"""9:16 reel (1080x1920) cut from the finished episode. usage: python3 reel.py out.mp4 | python3 reel.py --stills r1,r2,..
Layout: top band (hook question + progress bar), middle 1080x1215 (episode footage, crop 960 wide), bottom band (keyword pops)."""
import sys, math, subprocess
import numpy as np, cv2
import fx
from fx import T, cl, seg, ease, eout, back, env, text_mask, CYAN, AMBER, WHITE, GOLD, GREEN, D

RW, RH, MID0, MIDH = 1080, 1920, 352, 1215
SRC = f'{D}/E01_X_final.mp4'
# (name, F0, F1, crop centre x, top question)  - F = seconds in the final episode
SEGS = [('hook', 35.72, 40.02, 1000, 'I MOVED 250 KM'),
        ('rewind', None, None, 1000, 'WAIT… HOW?'),
        ('setup', 9.5, 19.07, 1180, '10 MINUTES · ZERO STEPS'),
        ('pull', 20.27, 40.0, 1000, 'WHERE AM I, REALLY?'),
        ('seat', 71.74, 77.08, 1020, 'YOUR FRIEND ON A TRAIN'),
        ('platform', 77.48, 86.98, 1000, 'SAME FRIEND · OTHER VIEW'),
        ('idea1', 131.17, 134.87, 650, 'THE BIG IDEA'),
        ('idea2', 140.57, 146.03, 1245, 'THE BIG IDEA'),
        ('cta', 29.62, 31.62, 960, 'SO… ARE YOU AT REST RIGHT NOW?')]
REW = .6
plan, R = [], 0.0
for name, a, b, cx, q in SEGS:
    d = REW if name == 'rewind' else b - a
    plan.append(dict(name=name, a=a, b=b, cx=cx, q=q, r0=R, r1=R + d)); R += d
DUR = R
def F2R(name, F):
    p = next(p for p in plan if p['name'] == name); return p['r0'] + F - p['a']
# keyword pops (reel seconds) from word timings
O, TR, ID = (lambda c: T('opening', c)), (lambda c: T('train', c)), (lambda c: T('idea', c))
KW = [(0.15, 'WITHOUT A SINGLE STEP', WHITE),
      (F2R('setup', O(19.6)), '10 MINUTES', CYAN), (F2R('setup', O(23.4)), 'ZERO STEPS', CYAN), (F2R('setup', O(25.0)), 'NOT EVEN A STRETCH', CYAN),
      (F2R('setup', O(26.7)), 'HOW FAR DID I TRAVEL?', AMBER), (F2R('setup', O(28.6)), 'ZERO… RIGHT?', WHITE),
      (F2R('pull', O(30.8)), 'PULL THE CAMERA BACK', CYAN), (F2R('pull', O(37.0)), 'A ROOM', CYAN), (F2R('pull', O(39.5)), 'A CITY', CYAN),
      (F2R('pull', O(42.2)), 'A PLANET', AMBER), (F2R('pull', O(44.0)), 'SPINNING ONCE A DAY', CYAN), (F2R('pull', O(46.6)), 'IN THOSE 10 MINUTES…', WHITE),
      (F2R('pull', O(54.4)), '250 KM!', AMBER),
      (F2R('seat', TR(17.5)), 'FROM YOUR SEAT', CYAN), (F2R('seat', TR(19.2)), 'NOT MOVING', CYAN),
      (F2R('platform', TR(24.0)), 'FROM THE PLATFORM', AMBER), (F2R('platform', TR(28.8)), '80 KM/H!', AMBER), (F2R('platform', TR(32.6)), 'BOTH ARE RIGHT', WHITE),
      (F2R('idea1', ID(95.8)), 'REST & MOTION', WHITE), (F2R('idea1', ID(97.9)), 'ARE NOT PROPERTIES', AMBER),
      (F2R('idea2', ID(107.4)), 'OBJECT ↔ OBSERVER', CYAN), (F2R('idea2', ID(110.0)), "IT'S A RELATIONSHIP", GOLD),
      (plan[-1]['r0'] + .2, 'FULL EPISODE ON THE CHANNEL', GOLD)]
KW.sort()

YY, XX = np.mgrid[0:RH, 0:RW].astype(np.float32)
NAVY = np.array((.075, .035, .018), np.float32)
BG = np.ones((RH, RW, 3), np.float32) * NAVY + (np.exp(-((XX - RW / 2) ** 2 + (YY - RH / 2) ** 2) / (2 * 800 ** 2)))[..., None] * np.array((.16, .09, .04), np.float32)
_tc = {}
def text(G, s, xy, size, col, w='Bold', amp=1.0, scale=1.0, maxw=1000, track=.06):
    if amp <= .002: return
    k = (s, size, w)
    if k not in _tc:
        m = text_mask(s, size, w, track)
        if m.shape[1] > maxw: m = cv2.resize(m, (maxw, int(m.shape[0] * maxw / m.shape[1])), interpolation=cv2.INTER_AREA)
        _tc[k] = m
    m = _tc[k]
    if scale != 1: m = cv2.resize(m, None, fx=scale, fy=scale)
    h, w_ = m.shape; x0, y0 = int(xy[0] - w_ / 2), int(xy[1] - h / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(RW, x0 + w_), min(RH, y0 + h)
    if xb <= xa or yb <= ya: return
    f = np.zeros((RH, RW), np.float32); f[ya:yb, xa:xb] = m[ya - y0:yb - y0, xa - x0:xb - x0] / 255
    G += ((f + cv2.GaussianBlur(f, (0, 0), 5) * .6) * amp)[..., None] * np.array(col, np.float32)

def frame(r, src):
    p = next(q for q in plan if q['r0'] <= r < q['r1'] + 1e-6) if r < DUR else plan[-1]
    img = BG.copy(); G = np.zeros_like(img)
    # middle: crop 960x1080 around cx, gentle push-in per segment + punch on each keyword
    u = (r - p['r0']) / max(.1, p['r1'] - p['r0'])
    kick = sum(.035 * math.exp(-((r - k) / .12) ** 2) for k, *_ in KW if 0 <= r - k < .5)
    z = 1.0 + .05 * u + kick
    cw, ch = 960 / z, 1080 / z; x0 = cl(p['cx'] - cw / 2, 0, 1920 - cw); y0 = (1080 - ch) / 2
    A = np.float32([[RW / cw, 0, -x0 * RW / cw], [0, MIDH / ch, -y0 * MIDH / ch + MID0]])
    mid = cv2.warpAffine(src, A, (RW, RH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = np.zeros((RH, RW), np.float32); mask[MID0:MID0 + MIDH] = 1
    fe = np.clip(np.minimum(YY - MID0, MID0 + MIDH - YY) / 60, 0, 1) * mask          # feather into the bands
    img = img * (1 - fe[..., None]) + mid * fe[..., None]
    if p['name'] == 'rewind':                                                         # VHS-style rewind look
        k = seg(r, p['r0'], p['r1']); img = img * (.75 + .25 * math.sin(k * 40))
        for y in range(MID0, MID0 + MIDH, 90): img[y + int(30 * math.sin(r * 50 + y)):y + int(30 * math.sin(r * 50 + y)) + 6] += .25
        m = np.zeros((RH, RW), np.uint8); cy_ = MID0 + MIDH // 2
        for ox in (-70, 50): cv2.fillPoly(m, [np.array([(RW // 2 + ox + 60, cy_ - 70), (RW // 2 + ox + 60, cy_ + 70), (RW // 2 + ox - 50, cy_)], np.int32)], 255, cv2.LINE_AA)
        f = m.astype(np.float32) / 255; G += (f + cv2.GaussianBlur(f, (0, 0), 8))[..., None] * np.array(WHITE, np.float32) * .9
    # top band: running question
    tq = seg(r, p['r0'], p['r0'] + .3)
    text(G, p['q'], (RW / 2, 215), 70 if len(p['q']) < 18 else 54, WHITE if p['name'] != 'hook' else AMBER, amp=tq, scale=.85 + .15 * back(tq))
    text(G, 'KINEMATICS · EP 1', (RW / 2, 105), 30, CYAN, 'Medium', amp=.85, track=.2)
    # progress bar
    pb = np.zeros((RH, RW), np.float32); pb[22:30, 40:40 + int((RW - 80) * r / DUR)] = 1; pb[22:30, 40:RW - 40] += .18
    G += (pb + cv2.GaussianBlur(pb, (0, 0), 4) * .8)[..., None] * np.array(CYAN, np.float32)
    # bottom band: latest keyword pops
    cur = [k for k in KW if k[0] <= r]
    if cur:
        k0, s, col = cur[-1]; nxt = [k for k in KW if k[0] > r]; end = min(nxt[0][0], k0 + 3.2) if nxt else k0 + 3.2
        a = seg(r, k0, k0 + .25) * (1 - seg(r, end - .2, end)); sc = back(seg(r, k0, k0 + .3))
        text(G, s, (RW / 2, 1745), 84 if len(s) < 14 else 62, col, amp=a, scale=max(.1, sc))
    g = np.clip(G, 0, 2)
    out = 1 - (1 - np.clip(img, 0, 1)) * (1 - np.clip(g, 0, 1)) + np.clip(g - 1, 0, 1) * .2
    return (np.clip(out, 0, 1) * 255 + .5).astype(np.uint8)

def read_at(F):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{F:.3f}', '-i', SRC, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3).astype(np.float32) / 255
def src_time(r):
    p = next(q for q in plan if q['r0'] <= r < q['r1'] + 1e-6) if r < DUR else plan[-1]
    if p['name'] == 'rewind': h = plan[0]; return h['b'] - (h['b'] - 25.0) * seg(r, p['r0'], p['r1'])   # fast rewind through the Earth shot
    return p['a'] + r - p['r0']

if __name__ == '__main__':
    if sys.argv[1] == '--stills':
        for i, r in enumerate(float(x) for x in sys.argv[2].split(',')):
            cv2.imwrite(f'{D}/reel/still_{i}.jpg', frame(r, read_at(src_time(r))))
        print('dur', DUR); sys.exit()
    out = sys.argv[1]; n = int(round(DUR * 25))
    wr = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{RW}x{RH}', '-r', '25', '-i', '-',
                           '-c:v', 'libx264', '-crf', '16', '-preset', 'medium', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    i = 0
    for p in plan:
        n_p = int(round(p['r1'] * 25)) - int(round(p['r0'] * 25))
        if p['name'] == 'rewind':
            h = plan[0]
            for k in range(n_p):
                r = (int(round(p['r0'] * 25)) + k) / 25
                src = read_at(h['b'] - (h['b'] - 25.0) * (k + 1) / n_p)
                wr.stdin.write(frame(r, src).tobytes())
            continue
        pr = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f"{p['a']:.3f}", '-i', SRC, '-t', f"{p['b'] - p['a'] + .2:.3f}", '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
        last = None
        for k in range(n_p):
            r = (int(round(p['r0'] * 25)) + k) / 25
            buf = pr.stdout.read(1920 * 1080 * 3)
            if len(buf) == 1920 * 1080 * 3: last = np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3)
            wr.stdin.write(frame(min(r, DUR - 1e-3), last.astype(np.float32) / 255).tobytes())
        pr.kill()
    wr.stdin.close(); wr.wait(); print('dur', DUR, 'frames', n)
