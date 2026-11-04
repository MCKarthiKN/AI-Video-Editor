"""Physics-explainer effects on a talking-head clip (tracked hands, dark stage, glowing graphics).

usage:
  python3 render_v2.py <work_dir> --stills t1,t2,...
  python3 render_v2.py <work_dir> --range <f0> <f1> <out.mp4>
"""
import sys, os, json, math, subprocess
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from fxlib import Canvas, sstep, fs, ease_out_back, blur, draw_sphere, draw_atom

D = sys.argv[1]
FPS = 30.0
SC = 1.5                      # 720x1280 source -> 1080x1920 output
W, H = 1080, 1920
cv = Canvas(W, H)
tr = json.load(open(f'{D}/track.json'))
FR = tr['frames']
N = len(FR)
DUR = N / FPS


def gsmooth(a, s):
    if s <= 0:
        return a
    r = int(3 * s) + 1
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * s * s)); k /= k.sum()
    pad = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)], 0)
    out = np.zeros_like(a, dtype=np.float64)
    for i, w in enumerate(k):
        out += w * pad[i:i + len(a)]
    return out


# ---------------- tracking data ----------------
PALM_RAW = np.load(f'{D}/palms.npy').astype(np.float64) * SC          # N x 2 x 2
PALM = gsmooth(PALM_RAW.reshape(N, 4), 1.2).reshape(N, 2, 2)
MID = gsmooth(PALM_RAW.mean(1), 3.0)
SPREAD = gsmooth(np.linalg.norm(PALM_RAW[:, 0] - PALM_RAW[:, 1], axis=1), 3.0)
NOSE = gsmooth(np.array([f['pose']['nose'][:2] for f in FR], np.float64) * SC, 2.0)
FACE = NOSE + [0, -15]
FOREHEAD = NOSE + [0, -120]
HOLO_X = gsmooth(NOSE[:, 0], 10.0)


def fidx(t):
    return int(min(max(round(t * FPS), 0), N - 1))


# ---------------- static maps ----------------
yy2, xx2 = np.mgrid[0:2 * H, 0:2 * W].astype(np.float32)
KEYBIG = 0.26 + 0.9 * np.exp(-((xx2 - W) ** 2 + ((yy2 - H) * 1.05) ** 2) / (2 * 360.0 ** 2))
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
RN = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - 800) / H) ** 2); RN = (RN / RN.max()).astype(np.float32)
NAVY = (np.array([0.05, 0.02, 0.006], np.float32) * np.exp(-((xx - W / 2) ** 2 + (yy - 1000) ** 2) / (2 * 650.0 ** 2))[..., None]).astype(np.float32)


def tone(x, k=2.0):
    return (1 - np.exp(-k * x)) / (1 - math.exp(-k))


# ---------------- equations ----------------
EQS = [("F = ma", 90, 230, 2.1), ("E = mc^2", 600, 200, 2.3), ("v = u + at", 40, 470, 1.4),
       ("PV = nRT", 720, 440, 1.5), ("p = mv", 420, 95, 1.6), ("s = ut + 1/2 at^2", 20, 650, 1.0),
       ("V = IR", 800, 680, 1.5), ("KE = 1/2 mv^2", 300, 390, 1.3), ("a = dv/dt", 30, 860, 1.2),
       ("W = Fd", 840, 900, 1.3)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def make_sprite(text, scale):
    segs, i = [], 0
    while i < len(text):
        if text[i] == '^' and i + 1 < len(text):
            segs.append((text[i + 1], True)); i += 2
        else:
            j = i
            while j < len(text) and text[j] != '^':
                j += 1
            segs.append((text[i:j], False)); i = j
    th = max(1, round(2 * scale))
    (tw, tht), _ = cv2.getTextSize('Hg', FONT, scale, th)
    width = 0
    for s, sup in segs:
        width += cv2.getTextSize(s, FONT, scale * (0.6 if sup else 1), th)[0][0] + 2
    pad = 34
    C = np.zeros((tht * 2 + 2 * pad, width + 2 * pad), np.uint8)
    x = pad; base = pad + int(tht * 1.4)
    for s, sup in segs:
        sc_ = scale * (0.6 if sup else 1)
        yb = base - (int(tht * 0.6) if sup else 0)
        cv2.putText(C, s, (x, yb), FONT, sc_, 255, max(1, round(th * (0.75 if sup else 1))), cv2.LINE_AA)
        x += cv2.getTextSize(s, FONT, sc_, th)[0][0] + 2
    core = C.astype(np.float32) / 255
    spr = (core[..., None] * np.array([1.0, 0.97, 0.9], np.float32) * 1.1
           + cv2.GaussianBlur(core, (0, 0), 3.0)[..., None] * np.array([1.0, 0.8, 0.45], np.float32) * 1.3
           + cv2.GaussianBlur(core, (0, 0), 12)[..., None] * np.array([1.0, 0.6, 0.25], np.float32) * 1.0)
    return spr, core, pad, base


rng = np.random.default_rng(5)
SPRITES = []
T_SHATTER0 = 5.12
for k, (txt, x, y, s) in enumerate(EQS):
    spr, core, pad, base = make_sprite(txt, s)
    t_in = 1.0 + 0.25 * k
    v = rng.uniform(-9, 9, 2)
    ph = rng.uniform(0, 6.28)
    t_sh = T_SHATTER0 + 0.28 * (x / W) + 0.05 * rng.random()
    SPRITES.append(dict(spr=spr, core=core, x=x - pad, y=y - base, t_in=t_in, v=v, ph=ph, t_sh=t_sh))


def eq_pos(e, t):
    dt = t - 1.0
    return (e['x'] + e['v'][0] * dt + 5 * math.sin(0.9 * t + e['ph']),
            e['y'] + e['v'][1] * dt + 6 * math.sin(0.7 * t + e['ph'] * 1.3))


# particles from the shattered equations
HAND_SWEEP = PALM[fidx(5.3), 0]
PTS = []
for e in SPRITES:
    ys_, xs_ = np.nonzero(e['core'] > 0.5)
    if len(xs_) == 0:
        continue
    sel = rng.choice(len(xs_), size=min(150, len(xs_)), replace=False)
    ex, ey = eq_pos(e, e['t_sh'])
    for i in sel:
        p0 = np.array([ex + xs_[i], ey + ys_[i]])
        d = p0 - HAND_SWEEP; d /= (np.linalg.norm(d) + 1e-6)
        perp = np.array([-d[1], d[0]])
        v0 = d * rng.uniform(110, 400) + perp * rng.normal(0, 60)
        dust = rng.random() < 0.3
        PTS.append((p0[0], p0[1], v0[0], v0[1], e['t_sh'], rng.uniform(0.6, 1.5), dust,
                    rng.uniform(-12, 12), rng.uniform(-14, 6), rng.uniform(0, 6.28), rng.uniform(0, 0.5)))
PTS = np.array(PTS, np.float64)
(P_X, P_Y, P_VX, P_VY, P_T0, P_LIFE, P_DUST, P_DX, P_DY, P_PH, P_DEL) = PTS.T
P_DUST = P_DUST.astype(bool)
T_GATHER = 10.7
P_TARGET = np.zeros(len(PTS), int)   # which palm collects the dust


def particles(t):
    """Return positions and brightness of the equation particles at time t."""
    dt = t - P_T0
    live = dt > 0
    kdr = 0.55
    x = P_X + P_VX * kdr * (1 - np.exp(-np.maximum(dt, 0) / kdr)) + P_DX * np.maximum(dt, 0) * P_DUST
    y = P_Y + P_VY * kdr * (1 - np.exp(-np.maximum(dt, 0) / kdr)) + P_DY * np.maximum(dt, 0) * P_DUST
    a = np.where(P_DUST, 0.75 * np.exp(-np.maximum(dt, 0) / 0.5) + 0.32, np.clip(1 - dt / P_LIFE, 0, 1) ** 1.5)
    a *= 0.75 + 0.25 * np.sin(t * 9 + P_PH)
    a = np.where(live, a, 0)
    if t > T_GATHER:
        n = fidx(t)
        u = np.clip((t - T_GATHER - P_DEL) / 0.7, 0, 1)
        u = u * u * (3 - 2 * u)
        tx = PALM[n, P_TARGET, 0]; ty = PALM[n, P_TARGET, 1]
        x = x + (tx - x) * u ** 1.5
        y = y + (ty - y) * u ** 1.5
        a = np.where(P_DUST, a * (1 + 2 * u) * (1 - np.clip((u - 0.85) / 0.15, 0, 1)), a)
    return x, y, a


# assign each dust particle to the nearer palm at gather time
_gx, _gy, _ = particles(T_GATHER)
_pn = PALM[fidx(T_GATHER)]
P_TARGET = (np.hypot(_gx - _pn[1, 0], _gy - _pn[1, 1]) < np.hypot(_gx - _pn[0, 0], _gy - _pn[0, 1])).astype(int)


def splat(G, x, y, a, col, s1=1.4, s2=5.0):
    buf = np.zeros((H, W), np.float32)
    ok = (a > 0.003) & (x >= 0) & (x < W - 1) & (y >= 0) & (y < H - 1)
    if not ok.any():
        return
    np.add.at(buf, (y[ok].astype(int), x[ok].astype(int)), a[ok].astype(np.float32))
    img = cv2.GaussianBlur(buf, (0, 0), s1) * 18 + cv2.GaussianBlur(buf, (0, 0), s2) * 25
    G += img[..., None] * np.asarray(col, np.float32)


# ---------------- orb / head / hologram timing ----------------
T_ORB, T_COMPRESS, T_FLY, T_ARRIVE, T_HOLO = 12.0, 19.0, 19.55, 19.85, 20.0
T_HANDS = 10.72
HOLO_CY, HOLO_HW, HOLO_HH = 285, 300, 175
SPARKS = [dict(off=rng.uniform(-1, 1), sp=rng.uniform(0.6, 1.2), ph=rng.uniform(0, 1)) for _ in range(28)]


def orb_state(t):
    """(alive, x, y, R, bright)"""
    if t < T_ORB or t >= T_ARRIVE:
        return False, 0, 0, 0, 0
    n = fidx(t)
    R = float(np.clip(0.32 * SPREAD[n], 70, 280)) * ease_out_back((t - T_ORB) / 0.5)
    R = min(R, max(60.0, float(MID[n][1] - (NOSE[n][1] + 140))))   # never cover the mouth
    R = R + (16 - R) * fs(T_COMPRESS, T_FLY, t)
    x, y = MID[n]
    if t >= T_FLY:
        u = fs(T_FLY, T_ARRIVE, t)
        x0, y0 = MID[fidx(T_FLY)]
        fx, fy = FOREHEAD[n]
        x, y = x0 + (fx - x0) * u, y0 + (fy - y0) * u - 120 * math.sin(math.pi * u)
    bright = 1 + 0.08 * math.sin(t * 5) + 0.7 * math.exp(-((t - 17.5) / 0.14) ** 2)
    return True, float(x), float(y), R, bright


def hands_env(t):
    if t < T_HANDS:
        return 0.0
    e = 1.5 * math.exp(-(t - T_HANDS) / 0.22) + 0.55 * fs(T_HANDS, T_HANDS + 0.35, t)
    return e * (1 - fs(12.0, 12.7, t))


def draw_cone(G, ox, oy, r0, tx, ty, spread, amp, col=(1.0, 0.72, 0.45)):
    """Soft beam of light from (ox,oy) towards (tx,ty), widening by `spread` px."""
    if amp <= 0.01:
        return
    L = math.hypot(tx - ox, ty - oy)
    if L < 5:
        return
    x0, x1, y0, y1 = cv.win((ox + tx) / 2, (oy + ty) / 2, L / 2 + spread + r0 + 10)
    dx = xx[y0:y1, x0:x1] - ox; dy = yy[y0:y1, x0:x1] - oy
    ux, uy = (tx - ox) / L, (ty - oy) / L
    along = (dx * ux + dy * uy) / L
    perp = -dx * uy + dy * ux
    hw = r0 + np.clip(along, 0, 1) * spread
    I = np.exp(-2.2 * (perp / hw) ** 2) * np.clip(1 - along, 0, 1) ** 1.2 * sstep(0.0, 0.1, along)
    G[y0:y1, x0:x1] += (I * amp)[..., None] * np.asarray(col, np.float32)


def draw_holo(G, S, Sa, t, n):
    g = ease_out_back((t - T_HOLO) / 0.5)
    if g <= 0.01:
        return
    cx, cy = float(HOLO_X[n]), HOLO_CY
    hw, hh = HOLO_HW * g, HOLO_HH * g
    flick = 1.0
    if t < T_HOLO + 0.45:
        flick = 0.5 + 0.5 * (np.random.default_rng(int(t * 1000)).random() > 0.35)
    a = flick * (1 - fs(22.45, 22.9, t))
    x0, x1, y0, y1 = cv.win(cx, cy, max(hw, hh) + 50)
    C = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    lx0, ly0 = cx - hw - x0, cy - hh - y0
    lx1, ly1 = cx + hw - x0, cy + hh - y0
    rr = int(26 * g)
    col = (1.0, 0.85, 0.45)
    pts = [((lx0 + rr, ly0), (lx1 - rr, ly0)), ((lx0 + rr, ly1), (lx1 - rr, ly1)),
           ((lx0, ly0 + rr), (lx0, ly1 - rr)), ((lx1, ly0 + rr), (lx1, ly1 - rr))]
    for p, q in pts:
        cv2.line(C, (int(p[0] * 16), int(p[1] * 16)), (int(q[0] * 16), int(q[1] * 16)), col, 3, cv2.LINE_AA, 4)
    for (ccx, ccy, a0) in ((lx0 + rr, ly0 + rr, 180), (lx1 - rr, ly0 + rr, 270), (lx1 - rr, ly1 - rr, 0), (lx0 + rr, ly1 - rr, 90)):
        cv2.ellipse(C, (int(ccx * 16), int(ccy * 16)), (rr * 16, rr * 16), 0, a0, a0 + 90, col, 3, cv2.LINE_AA, 4)
    # fill + scanlines
    ry0, ry1 = int(max(ly0, 0)), int(min(ly1, C.shape[0]))
    rx0, rx1 = int(max(lx0, 0)), int(min(lx1, C.shape[1]))
    if ry1 > ry0 and rx1 > rx0:
        fill = np.zeros((ry1 - ry0, rx1 - rx0, 3), np.float32) + np.array([1.0, 0.6, 0.25], np.float32) * 0.06
        fill[::6] += np.array([1.0, 0.75, 0.4], np.float32) * 0.035
        C[ry0:ry1, rx0:rx1] += fill
    glow = C + blur(C, 4) * 0.9 + blur(C, 16) * 0.6
    G[y0:y1, x0:x1] += glow * a
    # scan bar while materialising
    if t < T_HOLO + 0.6:
        u = fs(T_HOLO, T_HOLO + 0.6, t)
        sy = cy - hh + 2 * hh * u
        ys = np.arange(max(0, int(sy - 12)), min(H, int(sy + 13)))
        if len(ys):
            xs0, xs1 = int(max(0, cx - hw)), int(min(W, cx + hw))
            band = np.exp(-(ys - sy) ** 2 / (2 * 3.0 ** 2))[:, None] * 0.9 * (1 - u * 0.5)
            G[ys[0]:ys[-1] + 1, xs0:xs1] += band[..., None] * np.array([1.0, 0.9, 0.7], np.float32)
    # content: the atom
    aa = a * fs(T_HOLO + 0.25, T_HOLO + 0.7, t)
    draw_atom(cv, G, S, Sa, t, cx, cy + 5, 0.6 * g, aa * 0.95)


def render(n):
    t = n / FPS
    A = cv2.imread(f'{D}/frames/{n + 1:05d}.jpg')
    A = cv2.resize(A, (W, H), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255
    A = np.clip(A + 0.35 * (A - cv2.GaussianBlur(A, (0, 0), 1.2)), 0, 1)
    p = fs(0.35, 1.1, t)
    if p <= 0:
        return A
    rvm_a = f'{D}/masks_rvm/{n + 1:05d}.png'
    if os.environ.get('MATTE', 'rvm') == 'rvm' and os.path.exists(rvm_a):
        # Robust Video Matting: soft alpha + background-free foreground colour
        a = cv2.resize(cv2.imread(rvm_a, 0), (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
        MS = np.clip((a - 0.02) / 0.96, 0, 1)
        Fg = cv2.resize(cv2.imread(f'{D}/fgr_rvm/{n + 1:05d}.png'), (W, H), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255
        Fg = np.clip(Fg + 0.35 * (Fg - cv2.GaussianBlur(Fg, (0, 0), 1.2)), 0, 1)
        PSRC = Fg * (MS[..., None] < 0.98) + A * (MS[..., None] >= 0.98)
    else:
        m = cv2.resize(cv2.imread(f'{D}/masks/{n + 1:05d}.png', 0), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        mb = cv2.erode((m > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
        MS = cv2.GaussianBlur(mb, (0, 0), 1.6)
        inner = cv2.erode((m > 0.97).astype(np.float32), np.ones((11, 11), np.uint8))
        num = cv2.GaussianBlur(A * inner[..., None], (0, 0), 9)
        den = cv2.GaussianBlur(inner, (0, 0), 9)[..., None]
        we = np.clip(MS * 1.3, 0, 1)[..., None]
        PSRC = (num / np.maximum(den, 1e-4)) * (1 - we) + A * we
    fx, fy = FACE[n]
    ox, oy = int(W - fx), int(H - fy)
    KEY = KEYBIG[oy:oy + H, ox:ox + W]
    PG = tone(PSRC * 0.95) * KEY[..., None] * np.array([1.08, 0.98, 0.9], np.float32) \
        + np.array([0.02, 0.006, 0.0], np.float32) * (1 - np.clip(KEY, 0, 1))[..., None]
    if p < 1:
        k = np.clip(1.6 * p - (1 - RN) * 0.6, 0, 1); k = (k * k * (3 - 2 * k))[..., None]
        bg = A * (1 - k) + NAVY * k
        person = A * (1 - p) + PG * p
    else:
        bg = NAVY.copy()
        person = PG

    G = np.zeros((H, W, 3), np.float32)
    EQB = np.zeros((H, W, 3), np.float32)       # equations live behind the person
    spill = np.zeros((H, W, 3), np.float32)
    occ = np.zeros((H, W), np.float32)
    S = np.zeros((H, W, 3), np.float32); Sa = np.zeros((H, W, 1), np.float32)

    # --- equations ---
    if 1.0 <= t < 7.0:
        for e in SPRITES:
            if t < e['t_in']:
                continue
            a = fs(e['t_in'], e['t_in'] + 0.6, t) * (0.88 + 0.12 * math.sin(t * 7 + e['ph']))
            a *= 1 - fs(e['t_sh'], e['t_sh'] + 0.12, t)
            if a <= 0.01:
                continue
            ex, ey = eq_pos(e, t)
            spr = e['spr']; h_, w_ = spr.shape[:2]
            x0, y0 = int(ex), int(ey)
            sx0, sy0 = max(0, -x0), max(0, -y0)
            dx0, dy0 = max(0, x0), max(0, y0)
            dx1, dy1 = min(W, x0 + w_), min(H, y0 + h_)
            if dx1 > dx0 and dy1 > dy0:
                EQB[dy0:dy1, dx0:dx1] += spr[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0] * a
            ap = 1.2 * math.exp(-max(t - e['t_in'], 0) / 0.25)
            if ap > 0.02:
                cv.add_gauss(G, ex + spr.shape[1] / 2, ey + spr.shape[0] / 2, 14, (1.0, 0.85, 0.6), 0.5 * ap)
    # --- shatter particles + dust ---
    if T_SHATTER0 <= t < 12.2:
        px, py, pa = particles(t)
        splat(G, px, py, pa, (1.0, 0.85, 0.6))
    if 5.0 <= t < 6.0:   # the swiping hand flares as it hits the equations
        a = 1.3 * math.exp(-((t - 5.3) / 0.18) ** 2)
        hx, hy = PALM[n, 0]
        cv.hand_flare(G, hx, hy, a, t, 0)
        cv.add_gauss(spill, hx, hy, 140, (1.0, 0.6, 0.3), a)

    # --- hand flares: "one thing became clear" ---
    he = hands_env(t)
    if he > 0.01:
        for k in range(2):
            hx, hy = PALM[n, k]
            cv.hand_flare(G, hx, hy, he, t, k)
            cv.add_gauss(spill, hx, hy, 130, (1.0, 0.6, 0.3), min(he, 1.2))

    # --- orb between the hands ---
    alive, ox_, oy_, R, br = orb_state(t)
    if alive:
        oa = 1 - fs(T_FLY + 0.15, T_ARRIVE, t) * 0.3
        draw_sphere(cv, G, t, ox_, oy_, R, oa, col=(0.3, 0.74, 1.0), bright=br)
        if t < T_FLY:
            od = 0.78 * oa * (1 - fs(T_COMPRESS, T_FLY, t))
            x0, x1, y0, y1 = cv.win(ox_, oy_, R * 1.3)
            dd = np.sqrt((xx[y0:y1, x0:x1] - ox_) ** 2 + (yy[y0:y1, x0:x1] - oy_) ** 2)
            occ[y0:y1, x0:x1] = np.maximum(occ[y0:y1, x0:x1], od * sstep(R * 1.08, R * 0.75, dd))
            draw_cone(G, ox_, oy_ - R * 0.3, R * 0.6, ox_ + (fx - ox_) * 0.2, fy - 380, 380,
                      0.2 * fs(T_ORB, T_ORB + 0.4, t) * (1 - fs(18.6, T_COMPRESS, t)))
        cv.add_gauss(spill, ox_, oy_ - R * 0.3, R * 1.6 + 80, (0.35, 0.62, 1.0), 1.1 * oa * br)
        fa = 1.3 * math.exp(-max(t - T_ORB, 0) / 0.3)
        cv.add_gauss(G, ox_, oy_, 30, (1, 1, 1), 0.8 * fa)
        cv.add_streak(G, ox_, oy_, 600, 3, (1.0, 0.7, 0.4), 0.9 * fa)
        ca = 1.1 * math.exp(-((t - 17.55) / 0.12) ** 2) if t < 17.55 else 1.1 * math.exp(-(t - 17.55) / 0.3)
        if ca > 0.02:
            cv.add_streak(G, ox_, oy_, 700, 3.5, (1.0, 0.75, 0.45), ca)
            cv.add_rays(G, ox_, oy_, 120, 2.0, 3, 0.3, (1.0, 0.9, 0.75), 0.6 * ca)
        if t >= T_FLY:
            for k in range(1, 12):
                _, tx_, ty_, _, _ = orb_state(t - k * 0.012)
                cv.add_gauss(G, tx_, ty_, 10, (0.4, 0.8, 1.0), 0.5 * (1 - k / 12))

    # --- the idea lands in the head ---
    if t >= T_ARRIVE:
        fx2, fy2 = FOREHEAD[n]
        fl = math.exp(-(t - T_ARRIVE) / 0.15)
        cv.add_gauss(G, fx2, fy2, 40, (1, 1, 1), 1.1 * fl)
        cv.add_streak(G, fx2, fy2, 700, 3, (1.0, 0.75, 0.45), 1.1 * fl)
        cv.add_rays(G, fx2, fy2, 110, 2.0, 3, 0.2, (1.0, 0.9, 0.75), 0.6 * fl)
        hg = 0.3 * fl + 0.16 * fs(T_ARRIVE, T_ARRIVE + 0.3, t)
        cv.add_gauss(G, fx2, fy2 - 40, 150, (1.0, 0.65, 0.3), hg)
        cv.add_gauss(spill, fx2, fy2, 200, (1.0, 0.7, 0.45), 1.2 * fl + 0.15)

    # --- the picture inside the head ---
    if t >= T_HOLO:
        hb = fs(T_HOLO, T_HOLO + 0.4, t)
        cx = float(HOLO_X[n])
        draw_cone(G, FOREHEAD[n][0], FOREHEAD[n][1] - 60, 30, cx, HOLO_CY + HOLO_HH + 40, 290, 0.16 * hb)
        draw_holo(G, S, Sa, t, n)
        for s_ in SPARKS:
            u = ((t - T_HOLO) * s_['sp'] + s_['ph']) % 1.0
            sx = FOREHEAD[n][0] + s_['off'] * 40 * (1 - u) + (cx - FOREHEAD[n][0]) * u + s_['off'] * 220 * u
            sy = FOREHEAD[n][1] - 60 + (HOLO_CY + HOLO_HH - FOREHEAD[n][1] + 60) * u
            cv.add_gauss(G, sx, sy, 3, (1.0, 0.85, 0.55), 0.8 * hb * math.sin(math.pi * u))

    # --- composite ---
    bg = 1 - (1 - bg) * (1 - (1 - np.exp(-EQB)))
    person = person + PSRC * spill * 0.9
    base = person * MS[..., None] + bg * (1 - MS[..., None])
    base = base * (1 - occ[..., None])
    base = S + base * (1 - Sa)
    out = 1 - (1 - base) * (1 - (1 - np.exp(-G)))
    z = 1 + 0.07 * fs(13.3, 17.4, t) - 0.07 * fs(18.0, 19.8, t)
    if z > 1.0005:
        c = np.array([W / 2, 900.0])
        M = np.array([[z, 0, W / 2 - z * c[0]], [0, z, H / 2 - z * c[1]]], np.float32)
        out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = out * (1 - fs(22.45, 22.9, t))
    out = out + np.random.default_rng(n).normal(0, 0.007, (H, W, 1)).astype(np.float32)
    return out


def to8(img):
    return np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    if '--stills' in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index('--stills') + 1].split(',')]
        os.makedirs(f'{D}/stills', exist_ok=True)
        for t in ts:
            cv2.imwrite(f'{D}/stills/t{t:05.2f}.jpg', to8(render(fidx(t))), [cv2.IMWRITE_JPEG_QUALITY, 90])
            print('still', t, flush=True)
    elif '--range' in sys.argv:
        i = sys.argv.index('--range')
        f0, f1, outp = int(sys.argv[i + 1]), int(sys.argv[i + 2]), sys.argv[i + 3]
        ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                               '-r', '30', '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '16',
                               '-pix_fmt', 'yuv420p', outp], stdin=subprocess.PIPE)
        for n in range(f0, min(f1, N)):
            ff.stdin.write(to8(render(n)).tobytes())
            if (n - f0) % 30 == 0:
                print(f'[{f0}-{f1}] frame {n}', flush=True)
        ff.stdin.close(); ff.wait()
        print(f'[{f0}-{f1}] done', flush=True)
