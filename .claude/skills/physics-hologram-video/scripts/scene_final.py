"""Final scene: hologram apple tree, pluck, time-slow knob, slow free fall with dotted scale, time stop, velocity question.
Composited at 2880x1620 (from 4K), then a virtual camera (zoom / pan / motion blur) outputs 1920x1080.

usage: python3 scene_final.py <work_dir> <out.mp4> <first_frame> <last_frame>   (or --stills t1,t2,...)
"""
import sys, os, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

V = sys.argv[1]
A = V + '/assets'
K = 1.5                                   # design (1920x1080) -> canvas
W, H = 2880, 1620
OW, OH = 1920, 1080
FPS = 25
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
CYAN, AMBER = np.array((1.0, 0.85, 0.45), np.float32), np.array((0.35, 0.75, 1.0), np.float32)


def sstep(a, b, x):
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def gsm(a, s):
    r = int(3 * s) + 1
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * s * s)); k /= k.sum()
    pad = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)], 0)
    return sum(w * pad[i:i + len(a)] for i, w in enumerate(k))


P1 = gsm(np.load(V + '/palms.npy').reshape(-1, 4), 1.0).reshape(-1, 2, 2)     # design coords
N = len(P1)
PALM = P1 * K
T_PLUCK, T_RELEASE, T_KNOB0, T_KNOB1, T_DIST, T_STOP, T_RING, T_ARROW, T_VQ = 1.30, 4.22, 2.30, 3.20, 8.8, 13.7, 15.2, 17.45, 18.35
DUR = N / FPS

# ---------------- time rate (slowed by the knob, stopped at T_STOP) ----------------
def rate(t):
    if t < T_KNOB0: return 1.0
    if t < T_KNOB1 + 0.3: return 1 - 0.65 * sstep(T_KNOB0, T_KNOB1, t)
    if t < T_STOP: return 0.35 - 0.12 * sstep(4.4, 9.0, t)
    return 0.0
TS = np.concatenate([[0.0], np.cumsum([rate(i / FPS) / FPS for i in range(N + 1)])])
def tsim(t): return float(np.interp(t * FPS, np.arange(len(TS)), TS))

# ---------------- tree ----------------
T = cv2.imread(A + '/tree2.png', cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
th, tw = T.shape[:2]
b_, g_, r_, a_ = cv2.split(T)
fm = ((((r_ > 0.65) & (g_ < 0.5)) | ((r_ > 0.7) & (g_ > 0.55) & (b_ < 0.45))) & (a_ > 0.5)).astype(np.uint8)
fm = cv2.morphologyEx(fm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
nlab, lab, st, cen = cv2.connectedComponentsWithStats(fm)
fruits = [(cen[i][0], cen[i][1], i) for i in range(1, nlab) if 150 < st[i][4] < 4000]
SCT = 0.8 * K
target = (P1[int(T_PLUCK * FPS), 0] + np.array([4.0, -16.0])) * K
cands = [f for f in fruits if f[1] < 0.32 * th]
fx, fy, fid = max(cands, key=lambda f: f[0] - 0.6 * f[1])
OX, OY = target[0] - fx * SCT, target[1] - fy * SCT


def to_canvas(img):
    return cv2.warpAffine(img, np.array([[SCT, 0, OX], [0, SCT, OY]], np.float32), (W, H), flags=cv2.INTER_AREA)


TREE = to_canvas(T)
Tp = T.copy()
m = cv2.dilate((lab == fid).astype(np.uint8), np.ones((9, 9), np.uint8))
Tp[..., :3] = cv2.inpaint((np.clip(Tp[..., :3], 0, 1) * 255).astype(np.uint8), m, 9, cv2.INPAINT_TELEA).astype(np.float32) / 255
TREE_P = to_canvas(Tp)
FRUIT = np.array([OX + fx * SCT, OY + fy * SCT])
AT = TREE[..., 3]
dt_ = cv2.distanceTransform((cv2.GaussianBlur(AT, (0, 0), 6) > 0.2).astype(np.uint8), cv2.DIST_L2, 5)
DEPTH = cv2.GaussianBlur(dt_ / (dt_.max() + 1e-6), (0, 0), 30)
TOPW = np.clip(1 - (yy - OY) / (th * SCT * 0.8), 0, 1) ** 2
EDGE = cv2.GaussianBlur(cv2.morphologyEx((AT > 0.3).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 4)
CXT = OX + tw * SCT / 2
TREE_BASE_Y = OY + th * SCT - 15
noise = cv2.GaussianBlur(np.random.default_rng(7).random((H // 12, W // 12)).astype(np.float32), (0, 0), 2.2)
noise = cv2.resize((noise - noise.min()) / (noise.max() - noise.min()), (W, H), interpolation=cv2.INTER_CUBIC)
FRONT = np.clip((noise - 0.50) / 0.14, 0, 1) * (yy < OY + th * SCT * 0.70) * np.clip((960 * K - xx) / (80 * K), 0, 1)
FRONT *= 1 - np.exp(-(((xx - FRUIT[0]) ** 2 + (yy - FRUIT[1]) ** 2) / (2 * (55 * K) ** 2)))
FRONT = cv2.GaussianBlur(FRONT.astype(np.float32), (0, 0), 12)

# ---------------- fall path (design coords) ----------------
REL = P1[int(T_RELEASE * FPS), 0]
X0, Y0 = REL[0] + 4, REL[1] - 16
YF = 775.0
Y76 = P1[int(7.6 * FPS), 0][1] - 16


def apple_y(t):
    t = min(t, T_STOP)
    if t < 7.6:
        return Y0 + (Y76 - Y0) * ((t - T_RELEASE) / (7.6 - T_RELEASE)) ** 1.6
    return Y76 + (YF - Y76) * (1 - math.exp(-(t - 7.6) / 2.2)) / (1 - math.exp(-(T_STOP - 7.6) / 2.2))


GHOSTS = [Y0 + (YF - Y0) * (k / 6) ** 2 for k in range(2, 6)]
def ghost_time(g):
    ts = np.linspace(T_RELEASE, T_STOP, 2000)
    return float(ts[np.argmax([apple_y(x) >= g for x in ts])])
GT = [ghost_time(g) for g in GHOSTS]
LANE = np.clip(1 - np.abs(xx - (X0 - 45) * K) / (190 * K), 0, 1) ** 0.8 * np.clip((yy - (Y0 - 90) * K) / (60 * K), 0, 1)
LANE = cv2.GaussianBlur(LANE.astype(np.float32), (0, 0), 35)

# ---------------- props ----------------
APPLE = cv2.imread(A + '/apple.png').astype(np.float32) / 255
GHOST = APPLE.max(2, keepdims=True) * np.array([1.0, 0.92, 0.7], np.float32)
KNOB = cv2.imread(A + '/knob.png').astype(np.float32) / 255
SW = cv2.imread(A + '/stopwatch.png').astype(np.float32) / 255
SW_C = np.array([228.0, 342.0]) / SW.shape[0]        # dial centre as a fraction of the image height
FP = P1[int(T_KNOB0 * FPS), 1]
KNOB_XY = np.array([FP[0] + 5, FP[1] - 55])
SW_XY, SW_H = np.array([1690.0, 300.0]), 400.0
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', int(19 * K))
FONT_BIG = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', int(52 * K))


def text_mask(s, font):
    l, t_, r, b = font.getbbox(s)
    im = Image.new('L', (r - l + 8, b - t_ + 8), 0)
    ImageDraw.Draw(im).text((4 - l, 4 - t_), s, fill=255, font=font)
    return np.asarray(im, np.float32) / 255


LABELS = {k: text_mask('d%d' % k, FONT) for k in range(1, 8)}
VQ = text_mask('v = ?', FONT_BIG)


def add_img(G, img, cx, cy, h, amp=1.0, rot=0.0):
    """cx, cy, h in canvas px; additive."""
    if amp <= 0.003 or h < 2:
        return
    s = h / img.shape[0]
    im = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(h))), interpolation=cv2.INTER_AREA)
    if rot:
        im = cv2.warpAffine(im, cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] / 2), rot, 1), (im.shape[1], im.shape[0]))
    if im.ndim == 2:
        im = im[..., None]
    x0, y0 = int(cx - im.shape[1] / 2), int(cy - im.shape[0] / 2)
    sx, sy, dx, dy = max(0, -x0), max(0, -y0), max(0, x0), max(0, y0)
    ex, ey = min(W, x0 + im.shape[1]), min(H, y0 + im.shape[0])
    if ex > dx and ey > dy:
        G[dy:ey, dx:ex] += im[sy:sy + ey - dy, sx:sx + ex - dx] * amp


def add_mask(G, m, x, y, col, amp, glow=4):
    h_, w_ = m.shape
    x0, y0 = int(x), int(y)
    if x0 < 0 or y0 < 0 or x0 + w_ > W or y0 + h_ > H:
        return
    g = m + cv2.GaussianBlur(m, (0, 0), glow) * 0.6
    G[y0:y0 + h_, x0:x0 + w_] += (g * amp)[..., None] * col


def ring(G, cx, cy, r, col, amp, th=3, blur=5):
    if amp <= 0.01 or r < 2:
        return
    x0, x1 = int(max(0, cx - r - 40)), int(min(W, cx + r + 40)); y0, y1 = int(max(0, cy - r - 40)), int(min(H, cy + r + 40))
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.circle(m, (int((cx - x0) * 16), int((cy - y0) * 16)), int(r * 16), 255, th, cv2.LINE_AA, 4)
    f = m.astype(np.float32) / 255
    G[y0:y1, x0:x1] += ((f * 0.6 + cv2.GaussianBlur(f, (0, 0), blur)) * amp)[..., None] * np.asarray(col, np.float32)


# ---------------- living layers ----------------
rng = np.random.default_rng(12)
FIRE = [dict(x=rng.uniform(40, 900), y=rng.uniform(90, 820), sp=rng.uniform(8, 22), ph=rng.uniform(0, 6.3),
             r=rng.uniform(1.4, 3.2), c=CYAN if rng.random() < 0.5 else AMBER) for _ in range(48)]
LEAVES = [dict(x=rng.uniform(150, 800), y0=rng.uniform(250, 600), t0=rng.uniform(-6, 12), sp=rng.uniform(35, 60),
               ph=rng.uniform(0, 6.3)) for _ in range(9)]
FLOOR = np.zeros((H, W), np.uint8)
TBX, TBY = CXT, TREE_BASE_Y
for rx, ry, th_ in ((250, 26, 2), (390, 42, 2), (540, 58, 2)):
    cv2.ellipse(FLOOR, (int(TBX), int(TBY)), (int(rx * K), int(ry * K)), 0, 0, 360, 255, th_, cv2.LINE_AA)
for k in range(16):
    a = k * math.pi / 8
    for u in np.linspace(0.18, 1.0, 14):
        cv2.circle(FLOOR, (int(TBX + 540 * K * u * math.cos(a)), int(TBY + 58 * K * u * math.sin(a))), 2, 200, -1, cv2.LINE_AA)
FLOOR = FLOOR.astype(np.float32) / 255
FLOOR = FLOOR * 0.55 + cv2.GaussianBlur(FLOOR, (0, 0), 7) * 0.9


def lively(G, ts, amp, reveal):
    G += (FLOOR * amp * reveal)[..., None] * CYAN
    for f in FIRE:
        y = f['y'] - f['sp'] * ts; y = 90 + ((y - 90) % 730); x = f['x'] + 14 * math.sin(ts * 0.6 + f['ph'])
        tw_ = 0.55 + 0.45 * math.sin(ts * 3 + f['ph'])
        cx, cy = int(x * K), int(y * K)
        if 6 <= cx < W - 6 and 6 <= cy < H - 6:
            sub = np.zeros((25, 25), np.float32); cv2.circle(sub, (12, 12), int(f['r'] * K), 1.0, -1, cv2.LINE_AA)
            sub = cv2.GaussianBlur(sub, (0, 0), 3) * 1.6 * tw_ * amp * reveal
            G[cy - 12:cy + 13, cx - 12:cx + 13] += sub[..., None] * f['c']
    for L in LEAVES:
        tt = ts - L['t0']
        if tt < 0:
            continue
        y = L['y0'] + L['sp'] * tt; x = L['x'] + 40 * math.sin(tt * 1.3 + L['ph'])
        if y > TREE_BASE_Y / K - 20:
            continue
        ang = math.degrees(math.sin(tt * 1.7 + L['ph'])) * 0.6 + L['ph'] * 30
        cx, cy = int(x * K), int(y * K)
        if not (60 <= cx < W - 60 and 60 <= cy < H - 60):
            continue
        sub = np.zeros((120, 120), np.uint8)
        pts = cv2.ellipse2Poly((60, 60), (int(16 * K), int(6 * K)), int(ang), 0, 360, 20)
        cv2.polylines(sub, [pts], True, 255, 2, cv2.LINE_AA)
        ar = math.radians(ang)
        cv2.line(sub, (int(60 - 15 * K * math.cos(ar)), int(60 - 15 * K * math.sin(ar))), (int(60 + 15 * K * math.cos(ar)), int(60 + 15 * K * math.sin(ar))), 180, 1, cv2.LINE_AA)
        f = sub.astype(np.float32) / 255
        G[cy - 60:cy + 60, cx - 60:cx + 60] += ((f * 0.8 + cv2.GaussianBlur(f, (0, 0), 4) * 0.8) * amp * reveal)[..., None] * np.array((0.45, 1.0, 0.55), np.float32)


# ---------------- springs (hand -> leaves) ----------------
GS = 12
gh, gw = H // GS, W // GS
gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float32) * GS
KS = (2 * math.pi * 2.0) ** 2; CS = 2 * 0.22 * math.sqrt(KS)
LEAFG = cv2.resize(AT, (gw, gh), interpolation=cv2.INTER_AREA)


class Springs:
    def __init__(self):
        self.pos = np.zeros((gh, gw, 2), np.float32); self.vel = np.zeros((gh, gw, 2), np.float32)

    def step(self, n):
        t = n / FPS
        if t >= T_STOP:
            return
        r = rate(t)
        dtf = 1 / FPS
        for hand in range(2):
            p0, p1 = PALM[max(n - 1, 0), hand], PALM[n, hand]
            v = (p1 - p0) / dtf
            if np.linalg.norm(v) < 30 * K:
                continue
            w = np.exp(-((gx - p1[0]) ** 2 + (gy - p1[1]) ** 2) / (2 * (95 * K) ** 2)) * LEAFG
            self.vel += (w[..., None] * v[None, None, :] * 0.9).astype(np.float32)
        if abs(t - T_PLUCK) < 0.5 / FPS:
            w = np.exp(-((gx - FRUIT[0]) ** 2 + (gy - FRUIT[1]) ** 2) / (2 * (70 * K) ** 2)) * LEAFG
            self.vel[..., 1] += w * -420 * K
        for _ in range(4):
            h = dtf / 4 * max(r, 0.05)
            acc = -KS * self.pos - CS * self.vel
            self.vel += acc * h; self.pos += self.vel * h
        np.clip(self.pos, -45 * K, 45 * K, out=self.pos)


# ---------------- camera ----------------
CAM = [  # t, zoom, cx, cy (design coords)
    (0.00, 1.00, 960, 540), (1.15, 1.08, 830, 480), (1.30, 1.08, 830, 480), (1.55, 1.32, 730, 320), (1.95, 1.32, 730, 320),
    (2.20, 1.05, 960, 560), (2.42, 1.26, 1180, 700), (3.25, 1.26, 1180, 700), (3.65, 1.12, 820, 470), (4.18, 1.12, 820, 470),
    (4.45, 1.00, 960, 540), (5.50, 1.00, 960, 540), (11.0, 1.12, 790, 600), (13.62, 1.12, 790, 600), (13.82, 1.36, X0 + 40, YF - 40),
    (15.00, 1.36, X0 + 40, YF - 40), (17.20, 1.08, 930, 600), (DUR, 1.08, 930, 600)]
CT = np.array([c[0] for c in CAM])


def cam(t):
    i = int(np.clip(np.searchsorted(CT, t) - 1, 0, len(CAM) - 2))
    t0, z0, x0, y0 = CAM[i]; t1, z1, x1, y1 = CAM[i + 1]
    u = sstep(t0, t1, t) if t1 > t0 else 1.0
    z, cx, cy = z0 + (z1 - z0) * u, x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
    hw, hh = 960 / z, 540 / z
    return z, float(np.clip(cx, hw, 1920 - hw)), float(np.clip(cy, hh, 1080 - hh))


def camera_out(canvas, t):
    samples = [t - (1 - k / 6) * 0.5 / FPS for k in range(7)]
    params = [cam(s) for s in samples]
    zs = [p[0] for p in params]; cs = np.array([p[1:] for p in params])
    moving = (max(zs) - min(zs)) > 0.004 or np.abs(cs[-1] - cs[0]).max() > 2.5
    use = params if moving else [params[-1]]
    acc = np.zeros((OH, OW, 3), np.float32)
    for z, cx, cy in use:
        sc = z / K
        src = cv2.GaussianBlur(canvas, (0, 0), 0.45 / sc) if sc < 0.9 else canvas
        M = np.array([[sc, 0, 960 - cx * z], [0, sc, 540 - cy * z]], np.float32)
        acc += cv2.warpAffine(src, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / len(use)


# ---------------- per-frame ----------------
NAVY = (np.array([0.05, 0.02, 0.006], np.float32) * np.exp(-((xx - 900 * K) ** 2 + (yy - 560 * K) ** 2) / (2 * (700 * K) ** 2))[..., None]).astype(np.float32)
RN = np.sqrt(((xx - 1050 * K) / W) ** 2 + ((yy - 480 * K) / H) ** 2); RN = (RN / RN.max()).astype(np.float32)
KEYL = 0.3 + 0.85 * np.exp(-((xx - 1050 * K) ** 2 + ((yy - 470 * K) * 1.05) ** 2) / (2 * (330 * K) ** 2))


def desat(img, s):
    if s >= 0.999:
        return img
    g = img.mean(2, keepdims=True)
    return img * s + g * (1 - s) * np.array([1.08, 1.0, 0.88], np.float32)


def render(n, springs):
    t = n / FPS
    ts = tsim(t)
    frozen = t >= T_STOP
    sat = 1 - 0.78 * sstep(T_STOP, T_STOP + 0.3, t)
    Aimg = cv2.imread(V + '/frames_hi/%05d.jpg' % (n + 1)).astype(np.float32) / 255
    al = cv2.resize(cv2.imread(V + '/masks_rvm/%05d.png' % (n + 1), 0), (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    al = np.clip((al - 0.02) / 0.96, 0, 1)
    fg = cv2.resize(cv2.imread(V + '/fgr_rvm/%05d.png' % (n + 1)), (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    PS = np.where(al[..., None] < 0.98, fg, Aimg)
    PG = (1 - np.exp(-2 * PS * 0.95)) / (1 - math.exp(-2)) * KEYL[..., None] * np.array([1.06, 0.98, 0.92], np.float32)
    p = sstep(0.0, 0.6, t)
    kk = np.clip(1.6 * p - (1 - RN) * 0.6, 0, 1)[..., None]
    bg = Aimg * (1 - kk) + NAVY * kk
    person = Aimg * (1 - p) + PG * p

    # tree with wind sway, hologram turn, hand springs
    d = cv2.resize(springs.pos, (W, H), interpolation=cv2.INTER_LINEAR)
    yaw = 0.05 * math.sin(2 * math.pi * ts / 7.0)
    wind = (6.0 * math.sin(2 * math.pi * ts / 3.1) + 2.5 * math.sin(2 * math.pi * ts / 1.7)) * K
    mx = (xx - d[..., 0] - yaw * (DEPTH * 140 * K + (xx - CXT) * 0.2 * DEPTH) - wind * TOPW).astype(np.float32)
    my = (yy - d[..., 1]).astype(np.float32)
    tr = cv2.remap(TREE_P if t >= T_PLUCK else TREE, mx, my, cv2.INTER_LINEAR)
    front = cv2.remap(FRONT, mx, my, cv2.INTER_LINEAR)
    edge = cv2.remap(EDGE, mx, my, cv2.INTER_LINEAR)
    rv = sstep(0.15, 0.95, t)
    ry_ = OY + th * SCT * (1 - rv * 1.12)
    rev = np.clip((yy - ry_) / (40 * K), 0, 1)
    lane = 0.72 * sstep(T_RELEASE + 0.1, T_RELEASE + 0.9, t)
    ta = tr[..., 3] * rev * 0.93 * (1 - lane * LANE) * (1 - 0.2 * sstep(T_STOP, T_STOP + 0.3, t))
    rgb = tr[..., :3]
    lum = rgb.max(2, keepdims=True)
    rgb = rgb * 0.85 + lum * CYAN * 0.15
    tf = T_STOP if frozen else t
    sl = 0.88 + 0.12 * np.sin(yy * (2 * math.pi / (4.0 * K)) + tf * 6)
    band_y = OY + th * SCT * (1 - ((tf * 0.3) % 1.3))
    band = 1 + 0.45 * np.exp(-((yy - band_y) / (16 * K)) ** 2)
    fl = 1.0 if frozen else 0.95 + 0.05 * math.sin(t * 37) * math.sin(t * 11)
    rgb = desat(rgb * (sl * band * fl)[..., None], sat)
    glow = cv2.GaussianBlur(rgb * ta[..., None], (0, 0), 14) * 0.45 + edge[..., None] * CYAN * 0.3 * rev[..., None]
    if rv < 1:
        glow += (np.exp(-((yy - ry_) / (4 * K)) ** 2) * (tr[..., 3] > 0.05))[..., None] * np.array([1.0, 0.9, 0.6], np.float32)
    back = bg * (1 - ta[..., None]) + rgb * ta[..., None]
    back = 1 - (1 - back) * (1 - np.clip(glow, 0, 1))
    comp = person * al[..., None] + back * (1 - al[..., None])
    fa = (ta * front * 0.9)[..., None]
    comp = comp * (1 - fa) + rgb * fa
    if frozen:
        comp = comp * 0.92

    G = np.zeros((H, W, 3), np.float32)
    lively(G, ts, (0.9 if not frozen else 0.55) * (1 - 0.25 * sstep(T_RELEASE, T_RELEASE + 1, t)), rv)
    if frozen:
        G = desat(G, sat)

    # stopwatch with a sweeping second hand (slows with time, stops at T_STOP)
    swa = sstep(2.05, 2.45, t)
    if swa > 0:
        h_ = SW_H * K * (0.85 + 0.15 * swa)
        add_img(G, SW, SW_XY[0] * K, SW_XY[1] * K, h_, 0.95 * swa)
        cxs = SW_XY[0] * K - SW.shape[1] / SW.shape[0] * h_ / 2 + SW_C[0] * h_
        cys = SW_XY[1] * K - h_ / 2 + SW_C[1] * h_
        ang = 2 * math.pi * (ts - tsim(2.05)) / 4.0 - math.pi / 2
        sub = np.zeros((int(h_), int(h_)), np.uint8); c0 = int(h_ / 2)
        cv2.line(sub, (c0, c0), (int(c0 + math.cos(ang) * h_ * 0.30), int(c0 + math.sin(ang) * h_ * 0.30)), 255, 3, cv2.LINE_AA)
        cv2.circle(sub, (c0, c0), 5, 255, -1, cv2.LINE_AA)
        f = sub.astype(np.float32) / 255
        add_img(G, ((f + cv2.GaussianBlur(f, (0, 0), 4) * 0.8)[..., None] * np.array([0.6, 0.9, 1.0], np.float32)), cxs, cys, h_, swa)
    # knob: appears, turns down while time slows, then rests
    kna = sstep(2.05, 2.35, t) * (1 - 0.45 * sstep(3.6, 4.4, t)) * (1 - sstep(4.4, 5.2, t))
    if kna > 0:
        rot = 150 * sstep(T_KNOB0, T_KNOB1, t)
        add_img(G, KNOB, KNOB_XY[0] * K, KNOB_XY[1] * K, 190 * K * (0.8 + 0.2 * sstep(2.05, 2.35, t)), kna, rot=rot)
        for t0r in (2.45, 2.85, 3.2):
            u = (t - t0r) / 1.2
            if 0 <= u < 1:
                ring(G, KNOB_XY[0] * K, KNOB_XY[1] * K, (120 + 420 * u) * K, (1.0, 0.75, 0.3), 0.9 * (1 - u) ** 1.5)

    # apple
    if t >= T_PLUCK:
        if t < T_RELEASE:
            hp = P1[n, 0]; ax, ay, arot = hp[0] + 4, hp[1] - 16, 0.0
        else:
            ax, ay = X0, apple_y(t)
            arot = 40 * math.sin(0.9 * (tsim(min(t, T_STOP)) - tsim(T_RELEASE)))
        grow = sstep(T_PLUCK, T_PLUCK + 0.25, t)
        add_img(G, APPLE, ax * K, ay * K, 74 * K * (0.75 + 0.25 * grow), 1.0, rot=arot)
        fl_ = math.exp(-max(t - T_PLUCK, 0) / 0.18)
        if fl_ > 0.02:
            G += (np.exp(-((xx - ax * K) ** 2 + (yy - ay * K) ** 2) / (2 * (30 * K) ** 2)) * fl_ * 0.9)[..., None] * np.array([0.7, 0.9, 1.0], np.float32)
    # ghosts + dotted scale
    if t >= T_RELEASE:
        cur = apple_y(t)
        for g, gt in zip(GHOSTS, GT):
            if t >= gt + 0.05:
                add_img(G, GHOST, X0 * K, g * K, 68 * K, 0.8 * sstep(gt, gt + 0.4, t))
        sa = sstep(T_DIST, T_DIST + 0.7, t)
        if sa > 0:
            X = (X0 - 120) * K
            ys = [Y0] + [g for g, gt in zip(GHOSTS, GT) if t >= gt] + ([cur] if frozen else [])
            ylim = Y0 + (cur - Y0) * sa
            sm = np.zeros((H, W), np.uint8); lm = np.zeros((H, W), np.float32)
            for yv in np.arange(Y0 * K, ylim * K, 11 * K):
                cv2.circle(sm, (int(X), int(yv)), 2, 255, -1, cv2.LINE_AA)
            for yv in ys:
                if yv > ylim + 1:
                    continue
                xa, xb = X - 12 * K, (X0 - 34) * K
                xb = xa + (xb - xa) * sa
                x_ = xa
                while x_ < xb:
                    cv2.line(sm, (int(x_), int(yv * K)), (int(min(x_ + 9 * K, xb)), int(yv * K)), 255, 2, cv2.LINE_AA); x_ += 16 * K
                cv2.circle(sm, (int(X), int(yv * K)), int(4 * K), 255, -1, cv2.LINE_AA)
            f = sm.astype(np.float32) / 255
            G += ((f * 0.75 + cv2.GaussianBlur(f, (0, 0), 4) * 0.7) * sa)[..., None] * CYAN
            kk_ = 0
            for a1, b1 in zip(ys[:-1], ys[1:]):
                if b1 - a1 < 14 or b1 > ylim + 1:
                    continue
                kk_ += 1
                mk = LABELS[min(kk_, 7)]
                add_mask(G, mk, X - 50 * K, (a1 + b1) / 2 * K - mk.shape[0] / 2, AMBER, 0.95 * sa, 3)
    # time stop: white ripples from the apple
    if t >= T_STOP - 0.1:
        for t0r, rmax in ((T_STOP, 560), (T_STOP + 0.12, 300)):
            u = (t - t0r) / 0.9
            if 0 <= u < 1:
                ring(G, X0 * K, YF * K, (40 + rmax * u) * K, (1.0, 0.95, 0.85), 1.1 * (1 - u) ** 1.3, 4, 8)
        if t >= T_STOP:
            G += math.exp(-(t - T_STOP) / 0.08) * 0.12
    # velocity question
    if t >= T_RING:
        ra = sstep(T_RING, T_RING + 0.35, t)
        ring(G, X0 * K, YF * K, 62 * K * (0.6 + 0.4 * ra) + 3 * K * math.sin(t * 4), (1.0, 0.85, 0.4), 1.1 * ra)
    if t >= T_ARROW:
        u = sstep(T_ARROW, T_ARROW + 0.5, t)
        sub = np.zeros((H, W), np.uint8)
        y_a, y_b = int((YF + 70) * K), int((YF + 70 + 160 * u) * K)
        if y_b - y_a > 10:
            cv2.arrowedLine(sub, (int(X0 * K), y_a), (int(X0 * K), y_b), 255, int(6 * K), cv2.LINE_AA, tipLength=min(0.35, 40 * K / (y_b - y_a)))
        f = sub.astype(np.float32) / 255
        G += (f + cv2.GaussianBlur(f, (0, 0), 8) * 0.8)[..., None] * np.array([0.45, 0.75, 1.0], np.float32)
    if t >= T_VQ:
        u = sstep(T_VQ, T_VQ + 0.3, t)
        add_mask(G, VQ, (X0 + 45) * K, (YF + 120) * K, np.array([0.6, 0.9, 1.0], np.float32), u, 6)
    out = 1 - (1 - comp) * (1 - np.clip(G, 0, 1.0))
    return out


def frame(n, springs):
    t = n / FPS
    out = camera_out(render(n, springs), t)
    out *= 1 - sstep(DUR - 0.3, DUR, t)
    out += np.random.default_rng(n).normal(0, 0.006, (OH, OW, 1)).astype(np.float32)
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    springs = Springs()
    if '--stills' in sys.argv:
        ts_ = [float(x) for x in sys.argv[sys.argv.index('--stills') + 1].split(',')]
        want = sorted(int(round(x * FPS)) for x in ts_)
        os.makedirs(V + '/stills_f', exist_ok=True)
        for n in range(max(want) + 1):
            springs.step(n)
            if n in want:
                cv2.imwrite(V + '/stills_f/t%05.2f.jpg' % (n / FPS), frame(n, springs), [cv2.IMWRITE_JPEG_QUALITY, 90])
        sys.exit()
    OUT, F0, F1 = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-',
                           '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p', OUT], stdin=subprocess.PIPE)
    for n in range(min(F1, N)):
        springs.step(n)
        if n >= F0:
            ff.stdin.write(frame(n, springs).tobytes())
    ff.stdin.close(); ff.wait(); print(f'[{F0}-{F1}] done', flush=True)
