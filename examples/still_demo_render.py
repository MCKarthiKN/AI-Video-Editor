"""Render the 'glowing physics' demo onto a still frame of the user's clip.

usage: python3 render.py <work_dir> [--stills t1,t2,...]
work_dir must hold src/NN.png frames, mask_pose.png and landmarks.json.
"""
import sys, os, json, math, subprocess
import numpy as np, cv2

D = sys.argv[1]
STILLS = None
if '--stills' in sys.argv:
    STILLS = [float(x) for x in sys.argv[sys.argv.index('--stills') + 1].split(',')]

FPS, DUR = 30, 10.0
NF = int(FPS * DUR)
STILL_IDX = 13

A = cv2.imread(f'{D}/src/{STILL_IDX:02d}.png').astype(np.float32) / 255
H, W = A.shape[:2]
lm = json.load(open(f'{D}/landmarks.json'))['pose']
P = {k: np.array(v[:2], np.float32) for k, v in lm.items()}
FACE = P['nose'] + [0, -12]
HANDS = []
for w_, i_ in (('r_wrist', 'r_index'), ('l_wrist', 'l_index')):
    HANDS.append(P[i_] + 0.2 * (P[i_] - P[w_]) + [0, 8])
ORB = np.array([(P['l_shoulder'][0] + P['r_shoulder'][0]) / 2, P['l_shoulder'][1] + 90], np.float32)
ZOOM_C = np.array([ORB[0], (FACE[1] + ORB[1]) / 2 + 20], np.float32)

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

# ---------- helpers ----------
def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)

def ease_out_back(t, s=1.7):
    t = min(max(t, 0.0), 1.0) - 1
    return t * t * ((s + 1) * t + s) + 1

def win(cx, cy, r):
    x0, x1 = max(0, int(cx - r)), min(W, int(cx + r) + 1)
    y0, y1 = max(0, int(cy - r)), min(H, int(cy + r) + 1)
    return x0, x1, y0, y1

def add_gauss(G, cx, cy, s, col, amp):
    if amp <= 1e-4 or s <= 0:
        return
    x0, x1, y0, y1 = win(cx, cy, 3 * s + 1)
    if x0 >= x1 or y0 >= y1:
        return
    xs = np.arange(x0, x1, dtype=np.float32) - cx
    ys = np.arange(y0, y1, dtype=np.float32) - cy
    g = np.exp(-ys[:, None] ** 2 / (2 * s * s)) * np.exp(-xs[None, :] ** 2 / (2 * s * s)) * amp
    G[y0:y1, x0:x1] += g[..., None] * np.asarray(col, np.float32)

def add_streak(G, cx, cy, sx, sy, col, amp):
    """Horizontal anamorphic lens streak."""
    if amp <= 1e-4:
        return
    y0, y1 = max(0, int(cy - 3 * sy - 1)), min(H, int(cy + 3 * sy) + 2)
    if y0 >= y1:
        return
    xs = np.arange(W, dtype=np.float32) - cx
    ys = np.arange(y0, y1, dtype=np.float32) - cy
    gx = 0.6 * np.exp(-xs ** 2 / (2 * sx * sx)) + 0.4 * np.exp(-np.abs(xs) / (sx * 0.35))
    g = np.exp(-ys ** 2 / (2 * sy * sy))[:, None] * gx[None, :] * amp
    G[y0:y1] += g[..., None] * np.asarray(col, np.float32)

def add_rays(G, cx, cy, length, width, n, rot, col, amp):
    if amp <= 1e-4:
        return
    x0, x1, y0, y1 = win(cx, cy, length * 2.5)
    if x0 >= x1 or y0 >= y1:
        return
    dx = np.arange(x0, x1, dtype=np.float32)[None, :] - cx
    dy = np.arange(y0, y1, dtype=np.float32)[:, None] - cy
    acc = np.zeros((y1 - y0, x1 - x0), np.float32)
    for k in range(n):
        th = rot + k * math.pi / n
        c, s = math.cos(th), math.sin(th)
        perp = -s * dx + c * dy
        along = np.abs(c * dx + s * dy)
        acc += np.exp(-perp ** 2 / (2 * width ** 2)) * np.exp(-along / length)
    G[y0:y1, x0:x1] += (acc * amp)[..., None] * np.asarray(col, np.float32)

def blur(img, s):
    if s < 0.5:
        return img
    if s > 10:  # big blurs at quarter resolution
        f = 4
        small = cv2.resize(img, (img.shape[1] // f, img.shape[0] // f), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), s / f)
        return cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(img, (0, 0), s)

# ---------- matte + plate prep ----------
m = cv2.imread(f'{D}/mask_pose.png', 0).astype(np.float32) / 255
mb = cv2.erode((m > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
MS = cv2.GaussianBlur(mb, (0, 0), 1.6)
inner = cv2.erode((m > 0.97).astype(np.float32), np.ones((11, 11), np.uint8))
num = cv2.GaussianBlur(A * inner[..., None], (0, 0), 9)
den = cv2.GaussianBlur(inner, (0, 0), 9)[..., None]
fill = num / np.maximum(den, 1e-4)
w_edge = np.clip(MS * 1.3, 0, 1)[..., None]
PSRC = fill * (1 - w_edge) + A * w_edge          # edge colours pulled from inside the body
MS3 = MS[..., None]

def tone(x, k=2.0):
    return (1 - np.exp(-k * x)) / (1 - math.exp(-k))

dyk = (yy - FACE[1]) * np.where(yy > FACE[1], 1.3, 0.8)
d2 = (xx - FACE[0]) ** 2 + dyk ** 2
KEY = 0.18 + 1.0 * np.exp(-d2 / (2 * 240.0 ** 2))   # light on the face, falling off down the body
LUM = (PSRC @ np.array([0.114, 0.587, 0.299], np.float32))[..., None]
HICUT = 1 - 0.42 * sstep(0.45, 0.95, LUM)            # pull bright clothing down
PGRADE = tone(PSRC * 0.92 * HICUT) * KEY[..., None]
PGRADE = PGRADE * np.array([1.10, 0.97, 0.86], np.float32) + np.array([0.025, 0.008, 0.0], np.float32) * (1 - KEY)[..., None]
PGRADE = np.clip(PGRADE, 0, 1)

rn = np.sqrt(((xx - FACE[0]) / W) ** 2 + ((yy - FACE[1]) / H) ** 2)
RN = (rn / rn.max()).astype(np.float32)
NAVY = (np.array([0.05, 0.02, 0.006], np.float32) * np.exp(-((xx - ORB[0]) ** 2 + (yy - ORB[1]) ** 2) / (2 * 600.0 ** 2))[..., None]).astype(np.float32)

# ---------- scene elements ----------
rng = np.random.default_rng(7)
NDOT = 460
i = np.arange(NDOT) + 0.5
phi = np.arccos(1 - 2 * i / NDOT)
theta = math.pi * (1 + 5 ** 0.5) * i
SPH = np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], 1).astype(np.float32)

NUC = []
pts = rng.normal(size=(14, 3)); pts /= np.linalg.norm(pts, axis=1, keepdims=True)
pts *= rng.uniform(0.25, 1.0, (14, 1)) ** 0.5
for k, p in enumerate(pts):
    NUC.append((p, (0.42, 0.32, 1.0) if k % 2 == 0 else (1.0, 0.38, 0.22)))
ORBITS = [(0.0, 2.3, 0.0), (math.radians(60), -1.9, 2.1), (math.radians(-60), 2.7, 4.2)]
BOKEH = [dict(x=rng.uniform(0, W), y=rng.uniform(250, H - 250), vx=rng.uniform(-18, 18), vy=rng.uniform(-14, 6),
              r=rng.uniform(5, 22), a=rng.uniform(0.06, 0.2), ph=rng.uniform(0, 6.28),
              c=(1.0, 0.8, 0.6) if rng.random() < 0.6 else (0.85, 0.6, 1.0)) for _ in range(38)]
SPARK = [dict(rad=rng.uniform(1.25, 2.1), ang=rng.uniform(0, 6.28), w=rng.uniform(-1.2, 1.2), ph=rng.uniform(0, 6.28)) for _ in range(26)]

def orb_radius(t):
    if t < 2.2:
        return 0.0
    if t < 2.8:
        return 95 * ease_out_back((t - 2.2) / 0.6)
    if t < 5.2:
        return 95 + 45 * (t - 2.8) / 2.4
    return 140 * math.exp(2.0 * (t - 5.2))

def zoom_at(t):
    z = 1.0 + 0.12 * float(sstep(2.6, 5.2, t))
    if t >= 5.2:
        u = min((t - 5.2) / 1.1, 1.0)
        z = 1.12 + 0.20 * u * u
    if t >= 6.3:
        z = 1.14 + 0.06 * float(sstep(6.3, 9.4, t))
    return z

def draw_sphere(G, t, R, alpha):
    if R < 1 or alpha <= 0.01:
        return
    a = 0.65 * t
    ca, sa_ = math.cos(a), math.sin(a)
    tx = 0.35
    ct, st = math.cos(tx), math.sin(tx)
    x, y, z = SPH[:, 0], SPH[:, 1], SPH[:, 2]
    x1, z1 = ca * x + sa_ * z, -sa_ * x + ca * z
    y2, z2 = ct * y - st * z1, st * y + ct * z1
    order = np.argsort(z2)
    big = min(max((R - 160) / 900, 0.0), 1.0)
    col = np.array([0.18 + 0.12 * big, 0.42 + 0.33 * big, 1.0], np.float32)
    margin = int(max(30, R * 0.2) * 3)
    x0, x1_, y0, y1 = win(ORB[0], ORB[1], R + margin)
    if x0 >= x1_ or y0 >= y1:
        return
    C = np.zeros((y1 - y0, x1_ - x0, 3), np.float32)
    dot_r = max(1.6, R * 0.034)
    for k in order:
        depth = (z2[k] + 1) / 2
        px, py = ORB[0] + x1[k] * R - x0, ORB[1] + y2[k] * R - y0
        b = 0.18 + 0.82 * depth ** 1.6
        r = dot_r * (0.65 + 0.35 * depth)
        cv2.circle(C, (int(px * 16), int(py * 16)), max(1, int(r * 16)), (col * b).tolist(), -1, cv2.LINE_AA, 4)
    if big > 0:
        C = blur(C, R * 0.012)
    glow = C + blur(C, max(3, R * 0.035)) * 1.3 + blur(C, max(12, R * 0.16)) * 0.7
    G[y0:y1, x0:x1_] += glow * alpha
    add_gauss(G, ORB[0], ORB[1], R * 0.75, col, 0.35 * alpha * min(1.0, 400 / R))
    add_gauss(G, ORB[0], ORB[1], R * 1.5, (1.0, 0.55, 0.22), 0.28 * alpha * min(1.0, 250 / R) ** 0.8)

def draw_cone(G, R, amp):
    if amp <= 0.01:
        return
    top = FACE[1] - 330
    y0, y1 = int(max(0, top)), int(ORB[1])
    x0, x1 = int(max(0, ORB[0] - 560)), int(min(W, ORB[0] + 560))
    ys = np.arange(y0, y1, dtype=np.float32)[:, None]
    xs = np.arange(x0, x1, dtype=np.float32)[None, :] - ORB[0]
    along = (ORB[1] - ys) / (ORB[1] - top)
    hw = R * 0.7 + along * 400
    I = np.exp(-2.2 * (xs / hw) ** 2) * np.clip(1 - along, 0, 1) ** 1.3 * sstep(0.0, 0.12, along)
    G[y0:y1, x0:x1] += (I * amp)[..., None] * np.array([1.0, 0.72, 0.45], np.float32)

def draw_hand_flare(G, h, a, t, k):
    if a <= 0.01:
        return
    flick = 1 + 0.07 * math.sin(t * 23 + k * 2) + 0.05 * math.sin(t * 37 + k)
    a *= flick
    add_gauss(G, h[0], h[1], 7, (1, 1, 1), 0.9 * a)
    add_gauss(G, h[0], h[1], 26, (1.0, 0.62, 0.28), 0.75 * a)
    add_gauss(G, h[0], h[1], 95, (1.0, 0.45, 0.15), 0.32 * a)
    add_streak(G, h[0], h[1], 520, 2.6, (1.0, 0.72, 0.42), 0.85 * a)
    add_streak(G, h[0], h[1], 300, 11, (1.0, 0.55, 0.25), 0.22 * a)
    add_rays(G, h[0], h[1], 70, 1.6, 3, 0.25 * t + k, (1.0, 0.88, 0.75), 0.45 * a)

def hand_env(t, t0):
    if t < t0:
        return 0.0
    e = 1.5 * math.exp(-(t - t0) / 0.22) + 0.55 * float(sstep(t0, t0 + 0.35, t))
    e += 0.35 * float(sstep(4.6, 6.0, t))
    e *= 1 - float(sstep(6.25, 6.9, t))
    return e

def draw_ball(S, Sa, cx, cy, r, col):
    x0, x1, y0, y1 = win(cx, cy, r + 2)
    if x0 >= x1 or y0 >= y1:
        return
    dx = np.arange(x0, x1, dtype=np.float32)[None, :] - cx
    dy = np.arange(y0, y1, dtype=np.float32)[:, None] - cy
    d = np.sqrt(dx * dx + dy * dy)
    nz = np.sqrt(np.clip(1 - (d / r) ** 2, 0, 1))
    nx, ny = dx / r, dy / r
    L = np.array([-0.45, -0.55, 0.70]); L /= np.linalg.norm(L)
    lam = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    shade = 0.22 + 0.78 * lam
    spec = 0.75 * lam ** 24
    c = np.asarray(col, np.float32)[None, None, :] * shade[..., None] + spec[..., None]
    a = np.clip(r - d + 0.5, 0, 1)[..., None]
    S[y0:y1, x0:x1] = c * a + S[y0:y1, x0:x1] * (1 - a)
    Sa[y0:y1, x0:x1] = a + Sa[y0:y1, x0:x1] * (1 - a)

def draw_atom(G, S, Sa, t):
    sa = ease_out_back((t - 6.3) / 0.9)
    if sa <= 0.01:
        return
    fade = float(sstep(6.3, 6.7, t))
    # nucleus
    ang = 0.6 * t
    c, s = math.cos(ang), math.sin(ang)
    balls = []
    for p, col in NUC:
        x = c * p[0] + s * p[2]; z = -s * p[0] + c * p[2]
        balls.append((z, ORB[0] + x * 54 * sa, ORB[1] + p[1] * 54 * sa, col))
    balls.sort(key=lambda b: b[0])
    for z, bx, by, col in balls:
        draw_ball(S, Sa, bx, by, 27 * sa * (0.9 + 0.1 * (z + 1) / 2), col)
    Sa *= fade; S *= fade
    add_gauss(G, ORB[0], ORB[1], 100 * sa, (0.75, 0.45, 1.0), 0.5 * fade)
    add_gauss(G, ORB[0], ORB[1], 280 * sa, (1.0, 0.5, 0.2), 0.22 * fade)
    # orbits + electrons
    if t < 6.5:
        return
    ef = float(sstep(6.5, 7.0, t))
    rx, ry = 340 * sa, 100 * sa
    x0, x1, y0, y1 = win(ORB[0], ORB[1], rx + 40)
    C = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    for phi_, w, ph in ORBITS:
        cv2.ellipse(C, (int((ORB[0] - x0) * 16), int((ORB[1] - y0) * 16)), (int(rx * 16), int(ry * 16)),
                    math.degrees(phi_), 0, 360, (1.0, 0.85, 0.45), 3, cv2.LINE_AA, 4)
    G[y0:y1, x0:x1] += (C * 0.10 + blur(C, 5) * 0.35) * ef
    for phi_, w, ph in ORBITS:
        cp, sp = math.cos(phi_), math.sin(phi_)
        for k in range(14):
            th = w * t + ph - math.copysign(k * 0.07, w)
            lx, ly = rx * math.cos(th), ry * math.sin(th)
            ex, ey = ORB[0] + cp * lx - sp * ly, ORB[1] + sp * lx + cp * ly
            depth = 1.0 if math.sin(th) > 0 else 0.5
            if k == 0:
                add_gauss(G, ex, ey, 8, (1, 1, 1), 1.3 * ef * depth)
                add_gauss(G, ex, ey, 26, (1.0, 0.8, 0.35), 0.7 * ef * depth)
            else:
                add_gauss(G, ex, ey, 6, (1.0, 0.85, 0.5), 0.55 * (1 - k / 14) * ef * depth)

def render(t):
    p = float(sstep(0.6, 1.8, t))
    G = np.zeros((H, W, 3), np.float32)
    R = orb_radius(t)
    # background: lights go out from the edges inward
    if p < 1:
        k = np.clip(1.6 * p - (1 - RN) * 0.6, 0, 1)
        k = (k * k * (3 - 2 * k))[..., None]
        bg = A * (1 - k) + NAVY * k
    else:
        bg = NAVY
    person = A * (1 - p) + PGRADE * p if p < 1 else PGRADE.copy()
    # orb light spilling onto the body
    spill = np.zeros((H, W, 3), np.float32)
    orb_a = float(sstep(2.2, 2.5, t)) * (1 - float(sstep(6.15, 6.45, t)))
    if R > 0 and orb_a > 0:
        add_gauss(spill, ORB[0], ORB[1] - R * 0.4, R * 1.6 + 70, (0.35, 0.6, 1.0), 1.1 * orb_a * min(1, 140 / max(R, 1)) ** 0.3)
        add_gauss(spill, ORB[0], ORB[1] - 120, 260, (1.0, 0.6, 0.3), 0.5 * orb_a)
    for kk, (h, t0) in enumerate(zip(HANDS, (1.8, 1.95))):
        add_gauss(spill, h[0], h[1], 110, (1.0, 0.6, 0.3), 1.0 * min(hand_env(t, t0), 1.2))
    person = person + PSRC * spill * 0.9
    dim = 1 - 0.72 * float(sstep(5.4, 6.1, t)) + 0.67 * float(sstep(6.4, 7.3, t))
    person *= dim
    base = person * MS3 + bg * (1 - MS3)
    occ = np.zeros((H, W), np.float32)
    if R > 0 and orb_a > 0:
        oa = 0.82 * orb_a * (1 - float(sstep(300, 700, R)))
        if oa > 0.01:
            x0, x1, y0, y1 = win(ORB[0], ORB[1], R * 1.3)
            dd = np.sqrt((xx[y0:y1, x0:x1] - ORB[0]) ** 2 + (yy[y0:y1, x0:x1] - ORB[1]) ** 2)
            occ[y0:y1, x0:x1] = np.maximum(occ[y0:y1, x0:x1], oa * sstep(R * 1.08, R * 0.75, dd))
    if t >= 6.3:
        ofa = 0.6 * float(sstep(6.3, 6.7, t)) * ease_out_back((t - 6.3) / 0.9)
        x0, x1, y0, y1 = win(ORB[0], ORB[1], 330)
        dd2 = (xx[y0:y1, x0:x1] - ORB[0]) ** 2 + (yy[y0:y1, x0:x1] - ORB[1]) ** 2
        occ[y0:y1, x0:x1] = np.maximum(occ[y0:y1, x0:x1], ofa * np.exp(-dd2 / (2 * 110.0 ** 2)))
    base = base * (1 - occ[..., None])

    # hands
    for kk, (h, t0) in enumerate(zip(HANDS, (1.8, 1.95))):
        draw_hand_flare(G, h, hand_env(t, t0), t, kk)
    # orb, cone, appear-flare, sparkles
    if R > 0:
        draw_sphere(G, t, R, orb_a)
        draw_cone(G, R, 0.32 * orb_a * (1 - float(sstep(5.3, 5.9, t))))
        fa = 1.4 * math.exp(-max(t - 2.2, 0) / 0.3) * (t >= 2.2)
        add_gauss(G, ORB[0], ORB[1], 30, (1, 1, 1), 0.8 * fa)
        add_streak(G, ORB[0], ORB[1], 600, 3, (1.0, 0.7, 0.4), 0.9 * fa)
        add_rays(G, ORB[0], ORB[1], 110, 2.0, 3, 0.4, (1.0, 0.85, 0.7), 0.5 * fa)
        sp_a = float(sstep(2.6, 3.0, t)) * (1 - float(sstep(5.0, 5.4, t)))
        for s_ in SPARK:
            a_ = s_['ang'] + s_['w'] * t
            rr = R * s_['rad']
            tw = 0.5 + 0.5 * math.sin(t * 6 + s_['ph'])
            add_gauss(G, ORB[0] + rr * math.cos(a_), ORB[1] + rr * math.sin(a_) * 0.8, 2.6, (0.6, 0.85, 1.0), 0.8 * tw * sp_a)
    # flash at the peak of the magnify
    fl = 0.0
    if 5.9 < t < 7.5:
        fl = math.exp(-((t - 6.3) / 0.08) ** 2) if t < 6.3 else math.exp(-(t - 6.3) / 0.075)
        G += np.array([1.0, 0.9, 0.8], np.float32) * (0.85 * fl)
        add_streak(G, ORB[0], ORB[1], 900, 6, (1.0, 0.7, 0.4), 1.5 * fl)
    # atom + floating particles
    S = np.zeros((H, W, 3), np.float32); Sa = np.zeros((H, W, 1), np.float32)
    if t >= 6.3:
        draw_atom(G, S, Sa, t)
        bf = float(sstep(6.4, 7.2, t))
        for b in BOKEH:
            bx = b['x'] + b['vx'] * (t - 6.3)
            by = b['y'] + b['vy'] * (t - 6.3) + 6 * math.sin(t * 0.9 + b['ph'])
            add_gauss(G, bx, by, b['r'] * 0.6, b['c'], b['a'] * bf)
    base = S + base * (1 - Sa)
    Gs = 1 - np.exp(-G)
    out = 1 - (1 - base) * (1 - Gs)
    # camera push-in
    z = zoom_at(t)
    if z > 1.0005:
        u = min((z - 1) / 0.3, 1)
        c = np.array([W / 2, H / 2]) * (1 - u) + ZOOM_C * u
        M = np.array([[z, 0, W / 2 - z * c[0]], [0, z, H / 2 - z * c[1]]], np.float32)
        out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out *= 1 - float(sstep(9.2, 10.0, t))
    out += np.random.default_rng(int(t * 1000)).normal(0, 0.008, (H, W, 1)).astype(np.float32)
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)

def raw_frame(n):
    return cv2.imread(f'{D}/src/{n:02d}.png')

if STILLS:
    os.makedirs(f'{D}/stills', exist_ok=True)
    for t in STILLS:
        cv2.imwrite(f'{D}/stills/t{t:05.2f}.jpg', render(t), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print('still', t, flush=True)
    sys.exit()

os.makedirs(f'{D}/out', exist_ok=True)
ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS),
                       '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p',
                       f'{D}/out/video_only.mp4'], stdin=subprocess.PIPE)
for n in range(NF):
    t = n / FPS
    if n < STILL_IDX:  # play the real clip first, then hold the sharpest frame
        frame = raw_frame(n + 1)
    else:
        frame = render(t)
    ff.stdin.write(frame.tobytes())
    if n % 30 == 0:
        print('frame', n, flush=True)
ff.stdin.close(); ff.wait()
print('done')
