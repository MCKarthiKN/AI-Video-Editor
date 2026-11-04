"""Interactive hologram tree: leaves in front of / behind the speaker, hand-driven spring physics, fruit pluck.

usage: python3 tree_scene.py <work_dir> <out.mp4> <first_frame> <last_frame>
"""
import sys, os, math, subprocess
import numpy as np, cv2

V, OUT, F0, F1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
A = V + '/assets'
W, H, FPS = 1920, 1080, 25
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)


def sstep(a, b, x):
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def gsm(a, s):
    r = int(3 * s) + 1
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * s * s)); k /= k.sum()
    pad = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)], 0)
    return sum(w * pad[i:i + len(a)] for i, w in enumerate(k))


PALM = gsm(np.load(V + '/palms.npy').reshape(-1, 4), 1.0).reshape(-1, 2, 2)
N = len(PALM)
T_PLUCK, T_RELEASE = 1.30, 4.22

# ---------------- tree placement ----------------
T = cv2.imread(A + '/tree2.png', cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
th, tw = T.shape[:2]
# clean the cut-out: drop near-invisible pixels and isolated specks
al = np.clip((T[..., 3] - 0.18) / 0.82, 0, 1)
al[:6], al[-6:], al[:, :6], al[:, -6:] = 0, 0, 0, 0                       # source image border line
solid = cv2.morphologyEx((al > 0.05).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
ncc, lcc, scc, _ = cv2.connectedComponentsWithStats(solid)
keep = np.isin(lcc, [i for i in range(1, ncc) if scc[i][4] > 5000])
keep = cv2.dilate(keep.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
T[..., 3] = al * keep
b_, g_, r_, a_ = cv2.split(T)
fruitmask = ((((r_ > 0.65) & (g_ < 0.5)) | ((r_ > 0.7) & (g_ > 0.55) & (b_ < 0.45))) & (a_ > 0.5)).astype(np.uint8)
fruitmask = cv2.morphologyEx(fruitmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
nlab, lab, st, cen = cv2.connectedComponentsWithStats(fruitmask)
fruits = [(cen[i][0], cen[i][1], i) for i in range(1, nlab) if 150 < st[i][4] < 4000]
SC = 0.80
hand_pl = PALM[int(T_PLUCK * FPS), 0]
target = hand_pl + np.array([4.0, -16.0])                 # the fruit sits just above the closing fingers
# pick a fruit in the upper-right crown, then position the tree so it lands at the hand
cands = [f for f in fruits if f[1] < 0.32 * th]
fx, fy, fid = max(cands, key=lambda f: f[0] - 0.6 * f[1]) if cands else (0.8 * tw, 0.2 * th, None)
OX, OY = target[0] - fx * SC, target[1] - fy * SC
print('tree offset', round(OX), round(OY), 'fruits', len(fruits), flush=True)


def to_frame(img):
    """Place a tree-sized image into a full-frame canvas at (OX,OY), scale SC."""
    M = np.array([[SC, 0, OX], [0, SC, OY]], np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)


TREE = to_frame(T)                                     # BGRA, frame size
# plucked version: inpaint the chosen fruit away
Tp = T.copy()
if fid is not None:
    m = cv2.dilate((lab == fid).astype(np.uint8), np.ones((9, 9), np.uint8))
    rgb8 = (np.clip(Tp[..., :3], 0, 1) * 255).astype(np.uint8)
    Tp[..., :3] = cv2.inpaint(rgb8, m, 9, cv2.INPAINT_TELEA).astype(np.float32) / 255
    fruit_r = math.sqrt(st[fid][4] / math.pi) * SC
else:
    fruit_r = 22
TREE_P = to_frame(Tp)
FRUIT_XY = np.array([OX + fx * SC, OY + fy * SC])
ALPHA_T = TREE[..., 3]
# pseudo depth for a slow holographic turn
dt = cv2.distanceTransform((cv2.GaussianBlur(ALPHA_T, (0, 0), 6) > 0.2).astype(np.uint8), cv2.DIST_L2, 5)
DEPTH = cv2.GaussianBlur(dt / (dt.max() + 1e-6), (0, 0), 25)
TOPW = np.clip(1 - (yy - OY) / (th * SC * 0.8), 0, 1) ** 2
EDGE = cv2.GaussianBlur(cv2.morphologyEx((ALPHA_T > 0.3).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 3)
CX_T = OX + tw * SC / 2
# leaves in front of the speaker: soft low-frequency blobs over the crown, kept clear around the pluck point
noise = cv2.GaussianBlur(np.random.default_rng(7).random((H // 8, W // 8)).astype(np.float32), (0, 0), 2.2)
noise = cv2.resize((noise - noise.min()) / (noise.max() - noise.min()), (W, H), interpolation=cv2.INTER_CUBIC)
FRONT = np.clip((noise - 0.50) / 0.14, 0, 1) * (yy < OY + th * SC * 0.70) * np.clip((960 - xx) / 80, 0, 1)
FRONT *= 1 - np.exp(-(((xx - FRUIT_XY[0]) ** 2 + (yy - FRUIT_XY[1]) ** 2) / (2 * 55.0 ** 2)))
FRONT = cv2.GaussianBlur(FRONT.astype(np.float32), (0, 0), 9)

APPLE = cv2.imread(A + '/apple.png').astype(np.float32) / 255

# ---------------- spring field driven by the hands ----------------
GS = 8
gh, gw = H // GS, W // GS
gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float32) * GS
K = (2 * math.pi * 2.0) ** 2
C = 2 * 0.22 * math.sqrt(K)
pos = np.zeros((gh, gw, 2), np.float32); vel = np.zeros((gh, gw, 2), np.float32)
leafmask = cv2.resize(ALPHA_T, (gw, gh), interpolation=cv2.INTER_AREA)


def step_springs(n):
    global pos, vel
    dtf = 1 / FPS
    for hand in range(2):
        p0, p1 = PALM[max(n - 1, 0), hand], PALM[n, hand]
        v = (p1 - p0) / dtf
        sp = np.linalg.norm(v)
        if sp < 30:
            continue
        w = np.exp(-((gx - p1[0]) ** 2 + (gy - p1[1]) ** 2) / (2 * 95.0 ** 2)) * leafmask
        vel += (w[..., None] * v[None, None, :] * 0.9).astype(np.float32)
    t = n / FPS
    if abs(t - T_PLUCK) < 0.5 / FPS:                       # the branch springs back when the fruit is pulled
        w = np.exp(-((gx - FRUIT_XY[0]) ** 2 + (gy - FRUIT_XY[1]) ** 2) / (2 * 70.0 ** 2)) * leafmask
        vel[..., 1] += w * -420
    for _ in range(4):
        h = dtf / 4
        acc = -K * pos - C * vel
        vel += acc * h; pos += vel * h
    np.clip(pos, -45, 45, out=pos)


def displaced(t, ts):
    d = cv2.resize(pos, (W, H), interpolation=cv2.INTER_LINEAR)
    yaw = 0.05 * math.sin(2 * math.pi * ts / 7.0)
    wind = 6.0 * math.sin(2 * math.pi * ts / 3.1) + 2.5 * math.sin(2 * math.pi * ts / 1.7)
    dx = d[..., 0] + yaw * (DEPTH * 140 + (xx - CX_T) * 0.2 * DEPTH) + wind * TOPW
    dy = d[..., 1]
    return (xx - dx).astype(np.float32), (yy - dy).astype(np.float32)


def holo_tint(rgb, t):
    lum = rgb.max(2, keepdims=True)
    out = rgb * 0.85 + lum * np.array([1.0, 0.85, 0.4], np.float32) * 0.15
    sl = 0.88 + 0.12 * np.sin(yy * (2 * math.pi / 4.0) + t * 6)
    band_y = OY + th * SC * (1 - ((t * 0.3) % 1.3))
    band = 1 + 0.45 * np.exp(-((yy - band_y) / 16) ** 2)
    fl = 0.95 + 0.05 * math.sin(t * 37) * math.sin(t * 11)
    return out * (sl * band * fl)[..., None]


def place_add(G, img, cx, cy, h, amp=1.0):
    s = h / img.shape[0]
    im = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(h))), interpolation=cv2.INTER_AREA)
    x0, y0 = int(cx - im.shape[1] / 2), int(cy - im.shape[0] / 2)
    sx0, sy0 = max(0, -x0), max(0, -y0); dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + im.shape[1]), min(H, y0 + im.shape[0])
    if dx1 > dx0 and dy1 > dy0:
        G[dy0:dy1, dx0:dx1] += im[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0] * amp


NAVY = (np.array([0.05, 0.02, 0.006], np.float32) * np.exp(-((xx - 900) ** 2 + (yy - 560) ** 2) / (2 * 700.0 ** 2))[..., None]).astype(np.float32)
RN = np.sqrt(((xx - 1050) / W) ** 2 + ((yy - 480) / H) ** 2); RN = (RN / RN.max()).astype(np.float32)
KEY = 0.3 + 0.85 * np.exp(-((xx - 1050) ** 2 + ((yy - 470) * 1.05) ** 2) / (2 * 330.0 ** 2))

ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                       '-c:v', 'libx264', '-crf', '17', '-pix_fmt', 'yuv420p', OUT], stdin=subprocess.PIPE)
for n in range(0, min(F1, N)):
    step_springs(n)                                     # the simulation always runs from frame 0
    if n < F0:
        continue
    t = n / FPS
    Aimg = cv2.imread(V + '/frames/%05d.jpg' % (n + 1)).astype(np.float32) / 255
    a = cv2.imread(V + '/masks_rvm/%05d.png' % (n + 1), 0).astype(np.float32) / 255
    fg = cv2.imread(V + '/fgr_rvm/%05d.png' % (n + 1)).astype(np.float32) / 255
    PG = (1 - np.exp(-2 * fg * 0.95)) / (1 - math.exp(-2)) * KEY[..., None] * np.array([1.06, 0.98, 0.92], np.float32)
    p = sstep(0.0, 0.6, t)
    k = np.clip(1.6 * p - (1 - RN) * 0.6, 0, 1)[..., None]
    bg = Aimg * (1 - k) + NAVY * k
    person = Aimg * (1 - p) + PG * p
    # tree (plucked version after the pluck), displaced by wind, hologram turn and the hand springs
    src = TREE_P if t >= T_PLUCK else TREE
    mx, my = displaced(t, t)
    tr = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    front = cv2.remap(FRONT, mx, my, cv2.INTER_LINEAR)
    edge = cv2.remap(EDGE, mx, my, cv2.INTER_LINEAR)
    rv = sstep(0.15, 0.95, t)                            # projector build-up from the base
    reveal_y = OY + th * SC * (1 - rv * 1.12)
    rev = np.clip((yy - reveal_y) / 40, 0, 1)
    ta = tr[..., 3] * rev * 0.93
    rgb = holo_tint(tr[..., :3], t)
    glow = cv2.GaussianBlur(rgb * ta[..., None], (0, 0), 10) * 0.45 + edge[..., None] * np.array([1.0, 0.85, 0.4], np.float32) * 0.3 * rev[..., None]
    if rv < 1:
        glow += (np.exp(-((yy - reveal_y) / 4) ** 2) * (tr[..., 3] > 0.05))[..., None] * np.array([1.0, 0.9, 0.6], np.float32)
    back = bg * (1 - ta[..., None]) + rgb * ta[..., None]
    back = 1 - (1 - back) * (1 - np.clip(glow, 0, 1))
    comp = person * a[..., None] + back * (1 - a[..., None])
    fa = (ta * front * 0.9)[..., None]                   # leaves in front of the speaker
    comp = comp * (1 - fa) + rgb * fa
    # apple in the hand, then a first free fall (test only)
    G = np.zeros((H, W, 3), np.float32)
    if t >= T_PLUCK:
        hp = PALM[n, 0]
        if t < T_RELEASE:
            ax, ay = hp[0] + 4, hp[1] - 16
        else:
            rp = PALM[int(T_RELEASE * FPS), 0]
            tau = (t - T_RELEASE) * 0.35
            ax, ay = rp[0] + 4, rp[1] - 16 + 0.5 * 980 * tau ** 2
        grow = sstep(T_PLUCK, T_PLUCK + 0.25, t)
        place_add(G, APPLE, ax, ay, max(72.0, fruit_r * 4.5) * (0.75 + 0.25 * grow))
        fl = math.exp(-max(t - T_PLUCK, 0) / 0.18)
        G += (np.exp(-((xx - ax) ** 2 + (yy - ay) ** 2) / (2 * 30.0 ** 2)) * fl * 0.9)[..., None] * np.array([0.7, 0.9, 1.0], np.float32)
    out = 1 - (1 - comp) * (1 - np.clip(G, 0, 1))
    ff.stdin.write(np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8).tobytes())
ff.stdin.close(); ff.wait(); print('done', flush=True)
