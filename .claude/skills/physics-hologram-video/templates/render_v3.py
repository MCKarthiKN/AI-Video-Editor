"""Speech-driven scene: emotion -> picture in the head -> lived in it -> speaks to us -> our picture is delayed.

usage:
  python3 render_v3.py <work_dir> <assets_dir> --stills t1,t2,...
  python3 render_v3.py <work_dir> <assets_dir> --range f0 f1 out.mp4
"""
import sys, os, json, math, subprocess
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from fxlib import Canvas, sstep, fs, ease_out_back, blur

D, AS = sys.argv[1], sys.argv[2]
FPS, SC, W, H = 30.0, 1.5, 1080, 1920
cv = Canvas(W, H)
FR = json.load(open(f'{D}/track.json'))['frames']
N = len(FR)
T_FREEZE = 22.2


def gsmooth(a, s):
    r = int(3 * s) + 1
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * s * s)); k /= k.sum()
    pad = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)], 0)
    return sum(w * pad[i:i + len(a)] for i, w in enumerate(k))


def P(name):
    return np.array([f['pose'][name][:2] for f in FR], np.float64) * SC


NOSE = gsmooth(P('nose'), 2.0)
CHEST = gsmooth((P('l_shoulder') + P('r_shoulder')) / 2 + [0, 90], 3.0)
FACE = NOSE + [0, -15]
FOREHEAD = NOSE + [0, -120]
HOLO_X = gsmooth(NOSE[:, 0], 12.0)
HOLO_CY, HOLO_HW, HOLO_HH = 290, 300, 175
LIS_C, LIS_HW, LIS_HH = np.array([872.0, 640.0]), 165, 120


def fidx(t):
    return int(min(max(round(t * FPS), 0), N - 1))


# ---------- static maps ----------
yy2, xx2 = np.mgrid[0:2 * H, 0:2 * W].astype(np.float32)
KEYBIG = 0.26 + 0.9 * np.exp(-((xx2 - W) ** 2 + ((yy2 - H) * 1.05) ** 2) / (2 * 360.0 ** 2))
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
RN = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - 800) / H) ** 2); RN = (RN / RN.max()).astype(np.float32)
NAVY = (np.array([0.05, 0.02, 0.006], np.float32) * np.exp(-((xx - W / 2) ** 2 + (yy - 1000) ** 2) / (2 * 650.0 ** 2))[..., None]).astype(np.float32)


def tone(x, k=2.0):
    return (1 - np.exp(-k * x)) / (1 - math.exp(-k))


# ---------- icons ----------
ICON = {}
for nme in ('heart', 'sun', 'tree', 'home', 'walk', 'clock'):
    a = cv2.imread(f'{AS}/icons/{nme}.png', cv2.IMREAD_UNCHANGED)[..., 3].astype(np.float32) / 255
    ICON[nme] = a


def icon(G, name, cx, cy, size, col, amp, glow=1.0):
    """Glowing line icon centred at (cx,cy), `size` px tall."""
    if amp <= 0.01 or size < 4:
        return
    a = ICON[name]
    s = int(size)
    m = cv2.resize(a, (s, s), interpolation=cv2.INTER_AREA)
    pad = int(s * 0.35) + 8
    C = np.zeros((s + 2 * pad, s + 2 * pad), np.float32)
    C[pad:pad + s, pad:pad + s] = m
    g = C * 1.0 + cv2.GaussianBlur(C, (0, 0), max(1.5, s * 0.02)) * 0.9 * glow + cv2.GaussianBlur(C, (0, 0), max(4, s * 0.08)) * 0.6 * glow
    x0, y0 = int(cx - s / 2 - pad), int(cy - s / 2 - pad)
    hh, ww = g.shape
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0, dx1, dy1 = max(0, x0), max(0, y0), min(W, x0 + ww), min(H, y0 + hh)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    G[dy0:dy1, dx0:dx1] += (g[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0] * amp)[..., None] * np.asarray(col, np.float32)


def rrect(C, x0, y0, x1, y1, r, col, th=3):
    r = int(max(1, r))
    for p, q in (((x0 + r, y0), (x1 - r, y0)), ((x0 + r, y1), (x1 - r, y1)), ((x0, y0 + r), (x0, y1 - r)), ((x1, y0 + r), (x1, y1 - r))):
        cv2.line(C, (int(p[0] * 16), int(p[1] * 16)), (int(q[0] * 16), int(q[1] * 16)), col, th, cv2.LINE_AA, 4)
    for cx, cy, a0 in ((x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90)):
        cv2.ellipse(C, (int(cx * 16), int(cy * 16)), (r * 16, r * 16), 0, a0, a0 + 90, col, th, cv2.LINE_AA, 4)


def frame_box(G, cx, cy, hw, hh, a, col=(1.0, 0.85, 0.45), fill=0.06, scan=True):
    if a <= 0.01 or hw < 3:
        return
    x0, x1, y0, y1 = cv.win(cx, cy, max(hw, hh) + 50)
    C = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    rrect(C, cx - hw - x0, cy - hh - y0, cx + hw - x0, cy + hh - y0, 24 * min(1, hw / 200), col)
    ry0, ry1 = int(max(cy - hh - y0, 0)), int(min(cy + hh - y0, C.shape[0]))
    rx0, rx1 = int(max(cx - hw - x0, 0)), int(min(cx + hw - x0, C.shape[1]))
    if ry1 > ry0 and rx1 > rx0:
        C[ry0:ry1, rx0:rx1] += np.array([1.0, 0.6, 0.25], np.float32) * fill
        if scan:
            C[ry0:ry1:6, rx0:rx1] += np.array([1.0, 0.75, 0.4], np.float32) * 0.03
    G[y0:y1, x0:x1] += (C + blur(C, 4) * 0.9 + blur(C, 16) * 0.6) * a


def scene(G, cx, cy, sc, a, t, person=0.0, dim=1.0):
    """The memory picture: sun, tree, house, ground line, and the person who lived in it."""
    if a <= 0.01:
        return
    sky = (1.0, 0.85, 0.5)
    icon(G, 'sun', cx + 175 * sc, cy - 85 * sc, 90 * sc, (0.35, 0.8, 1.0), a * dim * (0.9 + 0.1 * math.sin(t * 2)))
    icon(G, 'tree', cx - 190 * sc, cy + 25 * sc, 150 * sc, (0.45, 0.95, 0.55), a * dim)
    icon(G, 'home', cx + 40 * sc, cy + 30 * sc, 130 * sc, sky, a * dim)
    gy = int(cy + 100 * sc)
    x0, x1 = int(cx - 270 * sc), int(cx + 270 * sc)
    if 0 <= gy < H:
        line = np.zeros((1, max(1, x1 - x0)), np.float32) + 1
        G[gy:gy + 2, max(0, x0):min(W, x1)] += 0.5 * a * dim * np.array([1.0, 0.8, 0.45], np.float32)
    if person > 0.01:
        px = cx - 95 * sc + 75 * sc * math.sin(t * 0.9)
        cv.add_gauss(G, px, cy + 40 * sc, 70 * sc, (0.4, 0.7, 1.0), 0.25 * person * a)
        icon(G, 'walk', px, cy + 38 * sc, 120 * sc, (0.45, 0.78, 1.0), person * a * 1.2)


def pixel_scene(G, cx, cy, hw, hh, prog, a, t):
    """Listener's picture assembling block by block (prog 0..1)."""
    if a <= 0.01:
        return
    sc = hw / 300
    tmp = np.zeros((H, W, 3), np.float32)
    scene(tmp, cx, cy, sc, 1.0, t, person=1.0)
    x0, x1, y0, y1 = int(cx - hw), int(cx + hw), int(cy - hh), int(cy + hh)
    sub = tmp[y0:y1, x0:x1]
    bs = 22
    gh, gw = (y1 - y0 + bs - 1) // bs, (x1 - x0 + bs - 1) // bs
    order = np.random.default_rng(3).permutation(gh * gw).reshape(gh, gw) / (gh * gw)
    mask = (order < prog).astype(np.float32)
    m = cv2.resize(mask, (gw * bs, gh * bs), interpolation=cv2.INTER_NEAREST)[:y1 - y0, :x1 - x0]
    small = cv2.resize(sub, (max(1, (x1 - x0) // 10), max(1, (y1 - y0) // 10)), interpolation=cv2.INTER_AREA)
    pix = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    G[y0:y1, x0:x1] += pix * m[..., None] * a * 1.2
    # loading bar
    by = int(cy + hh + 34)
    bx0, bx1 = int(cx - hw), int(cx + hw)
    bar = np.zeros((14, bx1 - bx0, 3), np.float32)
    bar[:, :] = np.array([1.0, 0.7, 0.35], np.float32) * 0.12
    bar[:, :int((bx1 - bx0) * prog)] = np.array([1.0, 0.85, 0.5], np.float32) * 0.9
    G[by:by + 14, bx0:bx1] += (bar + blur(bar, 4) * 0.6) * a


def progress(t):
    if t < 15.5:
        return 0.0
    if t < 17.6:
        return 0.33 * fs(15.5, 17.6, t)
    if t < 20.4:
        return 0.33 + 0.12 * fs(18.0, 20.4, t)
    return 0.45 + 0.01 * math.sin(t * 20) * (t > 21.2)


def rings(G, cx, cy, t, t0, t1, period, rmax, col, amp, life=1.6):
    k0 = int(max(0, (t - t0 - life) // period))
    k = k0
    while True:
        ts = t0 + k * period
        if ts > t or ts > t1:
            break
        u = (t - ts) / life
        if 0 <= u < 1:
            r = 40 + (rmax - 40) * (1 - (1 - u) ** 2)
            x0, x1, y0, y1 = cv.win(cx, cy, r + 30)
            C = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
            cv2.circle(C, (int((cx - x0) * 16), int((cy - y0) * 16)), int(r * 16), col, 3, cv2.LINE_AA, 4)
            G[y0:y1, x0:x1] += (C * 0.5 + blur(C, 5) * 0.8) * amp * (1 - u) ** 1.3
        k += 1


def clock(G, cx, cy, size, a, t):
    if a <= 0.01:
        return
    icon(G, 'clock', cx, cy, size, (1.0, 0.85, 0.5), a)
    ang = (t - 19.8) * 7.0 - math.pi / 2
    x0, x1, y0, y1 = cv.win(cx, cy, size)
    C = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    ex, ey = cx + math.cos(ang) * size * 0.3 - x0, cy + math.sin(ang) * size * 0.3 - y0
    cv2.line(C, (int((cx - x0) * 16), int((cy - y0) * 16)), (int(ex * 16), int(ey * 16)), (0.5, 0.85, 1.0), 4, cv2.LINE_AA, 4)
    G[y0:y1, x0:x1] += (C + blur(C, 3)) * a


def render(n, t):
    A = cv2.imread(f'{D}/frames/{n + 1:05d}.jpg')
    A = cv2.resize(A, (W, H), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255
    A = np.clip(A + 0.35 * (A - cv2.GaussianBlur(A, (0, 0), 1.2)), 0, 1)
    a = cv2.resize(cv2.imread(f'{D}/masks_rvm/{n + 1:05d}.png', 0), (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    MS = np.clip((a - 0.02) / 0.96, 0, 1)
    Fg = cv2.resize(cv2.imread(f'{D}/fgr_rvm/{n + 1:05d}.png'), (W, H), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255
    Fg = np.clip(Fg + 0.35 * (Fg - cv2.GaussianBlur(Fg, (0, 0), 1.2)), 0, 1)
    PSRC = np.where(MS[..., None] < 0.98, Fg, A)
    p = fs(0.15, 0.85, t)
    fx, fy = FACE[n]
    KEY = KEYBIG[int(H - fy):int(H - fy) + H, int(W - fx):int(W - fx) + W]
    PG = tone(PSRC * 0.95) * KEY[..., None] * np.array([1.08, 0.98, 0.9], np.float32) \
        + np.array([0.02, 0.006, 0.0], np.float32) * (1 - np.clip(KEY, 0, 1))[..., None]
    if p < 1:
        k = np.clip(1.6 * p - (1 - RN) * 0.6, 0, 1); k = (k * k * (3 - 2 * k))[..., None]
        bg = A * (1 - k) + NAVY * k
        person = A * (1 - p) + PG * p
    else:
        bg, person = NAVY.copy(), PG
    G = np.zeros((H, W, 3), np.float32)
    spill = np.zeros((H, W, 3), np.float32)
    cx_, cy_ = CHEST[n]
    hx, hy = FOREHEAD[n]
    pink = (0.6, 0.45, 1.0)

    # 1) emotion: rings from the chest, heart from "எமோஷன்" (2.9 s)
    if 0.9 <= t < 6.0:
        ea = fs(0.9, 1.4, t) * (1 - fs(5.0, 5.6, t))
        rings(G, cx_, cy_, t, 1.0, 5.0, 0.6, 430, pink, 1.8 * ea * (0.6 + 0.4 * fs(2.8, 3.2, t)))
        cv.add_gauss(spill, cx_, cy_, 260, (0.5, 0.45, 1.0), 0.35 * ea)
    if 2.85 <= t < 5.9:
        beat = 1 + 0.18 * max(0.0, math.sin(t * 2 * math.pi * 1.2)) ** 6
        sz = 165 * ease_out_back((t - 2.85) / 0.4) * beat
        u = fs(5.0, 5.75, t)                                   # the heart rises into the head
        x = cx_ + (hx - cx_) * u; y = cy_ + (hy - cy_) * u - 90 * math.sin(math.pi * u)
        icon(G, 'heart', x, y, sz * (1 - 0.6 * u), pink, 1.6)
        cv.add_gauss(G, x, y, 75, pink, 0.45)
        cv.add_gauss(spill, x, y, 180, (0.5, 0.45, 1.0), 0.5)
    if 5.75 <= t < 7.0:                                        # lands in the head ("மண்டை" 5.8)
        fl = math.exp(-(t - 5.75) / 0.18)
        cv.add_gauss(G, hx, hy, 40, (1, 1, 1), 1.0 * fl)
        cv.add_streak(G, hx, hy, 650, 3, (1.0, 0.75, 0.45), 1.0 * fl)
        cv.add_gauss(spill, hx, hy, 200, (1.0, 0.7, 0.45), 1.0 * fl)

    # 2) the picture in the head ("பிக்சர்" 6.5 s)
    hcx = float(HOLO_X[n])
    if t >= 6.2:
        g = ease_out_back((t - 6.2) / 0.5)
        flick = 0.55 + 0.45 * (np.random.default_rng(int(t * 1000)).random() > 0.3) if t < 6.65 else 1.0
        hb = fs(6.2, 6.6, t)
        # beam from forehead up to the frame
        L = (hy - 60) - (HOLO_CY + HOLO_HH + 30)
        if L > 10:
            y0, y1 = int(HOLO_CY + HOLO_HH + 30), int(hy - 60)
            ys = np.arange(y0, y1, dtype=np.float32)[:, None]
            along = (y1 - ys) / L
            xs = np.arange(W, dtype=np.float32)[None, :]
            cxl = hx + (hcx - hx) * along
            hwid = 30 + along * 280
            I = np.exp(-2.2 * ((xs - cxl) / hwid) ** 2) * (1 - along * 0.6)
            G[y0:y1] += (I * 0.14 * hb)[..., None] * np.array([1.0, 0.72, 0.45], np.float32)
        frame_box(G, hcx, HOLO_CY, HOLO_HW * g, HOLO_HH * g, flick)
        if t < 6.8:
            u = fs(6.2, 6.8, t); sy = HOLO_CY - HOLO_HH + 2 * HOLO_HH * u
            G[int(sy):int(sy) + 3, int(hcx - HOLO_HW):int(hcx + HOLO_HW)] += np.array([1.0, 0.9, 0.7], np.float32) * 0.8
        person_a = fs(8.1, 8.6, t)
        scene(G, hcx, HOLO_CY + 5, 0.95 * g, flick * fs(6.4, 7.0, t), t, person=person_a, dim=1 - 0.35 * person_a)
        if 8.1 <= t < 9.0:
            cv.add_gauss(G, hcx - 95, HOLO_CY + 40, 30, (1, 1, 1), 0.8 * math.exp(-(t - 8.1) / 0.2))

    # 3) speaks to us ("பேசுவாரு" 13.5 s): waves travel out of the picture towards the viewer
    if 13.3 <= t < 15.6:
        for k, ts in enumerate((13.35, 13.65, 13.95)):
            u = (t - ts) / 1.4
            if 0 <= u < 1:
                r = 120 + 1100 * u ** 1.4
                x0, x1, y0, y1 = cv.win(hcx, HOLO_CY, r + 40)
                C = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
                cv2.ellipse(C, (int((hcx - x0) * 16), int((HOLO_CY - y0) * 16)), (int(r * 16), int(r * 0.75 * 16)), 0, 0, 360,
                            (1.0, 0.8, 0.5), int(3 + 8 * u), cv2.LINE_AA, 4)
                G[y0:y1, x0:x1] += (C * 0.3 + blur(C, 6 + 10 * u) * 0.9) * (1 - u) ** 1.5 * 0.9

    # 4) our picture is delayed: listener frame, assembling, clock, stall
    if t >= 14.6:
        la = ease_out_back((t - 14.6) / 0.45)
        gl = 0.0
        if 21.2 <= t < 22.0:
            gl = (np.random.default_rng(int(t * 60)).random() - 0.5) * 30 * (1 - fs(21.2, 22.0, t))
        frame_box(G, LIS_C[0] + gl, LIS_C[1], LIS_HW * la, LIS_HH * la, 0.85, col=(1.0, 0.75, 0.4), fill=0.04)
        pixel_scene(G, LIS_C[0] + gl, LIS_C[1], LIS_HW - 6, LIS_HH - 6, progress(t), fs(14.8, 15.2, t), t)
    if t >= 19.7:
        clock(G, LIS_C[0] + 110, LIS_C[1] - LIS_HH - 80, 100 * ease_out_back((t - 19.7) / 0.4), 1.0, t)

    # composite
    person = person + PSRC * spill * 0.9
    base = person * MS[..., None] + bg * (1 - MS[..., None])
    out = 1 - (1 - base) * (1 - (1 - np.exp(-G)))
    if t >= T_FREEZE:                                         # freeze punch-in
        z = 1 + 0.06 * fs(T_FREEZE, 22.9, t)
        c = np.array([W / 2, 760.0])
        M = np.array([[z, 0, c[0] - z * c[0]], [0, z, c[1] - z * c[1]]], np.float32)
        out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = out * (1 - fs(22.45, 22.9, t))
    out = out + np.random.default_rng(n).normal(0, 0.007, (H, W, 1)).astype(np.float32)
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def frame_at(i):
    t = i / FPS
    n = fidx(min(t, T_FREEZE))
    return render(n, t)


if __name__ == '__main__':
    if '--stills' in sys.argv:
        os.makedirs(f'{D}/stills3', exist_ok=True)
        for t in [float(x) for x in sys.argv[sys.argv.index('--stills') + 1].split(',')]:
            cv2.imwrite(f'{D}/stills3/t{t:05.2f}.jpg', frame_at(round(t * FPS)), [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        i = sys.argv.index('--range')
        f0, f1, outp = int(sys.argv[i + 1]), int(sys.argv[i + 2]), sys.argv[i + 3]
        ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '30',
                               '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p', outp],
                              stdin=subprocess.PIPE)
        for k in range(f0, min(f1, N)):
            ff.stdin.write(frame_at(k).tobytes())
        ff.stdin.close(); ff.wait()
        print(f'[{f0}-{f1}] done', flush=True)
