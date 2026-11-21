"""Storyboard mock panels for Kinematics Ep 1. usage: python3 sb_ep01.py <ep01_dir>"""
import sys, os, math, json
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

D = sys.argv[1]; OUT = f'{D}/sb'; os.makedirs(OUT, exist_ok=True)
W, H = 1920, 1080
CYAN, AMBER, WHITE, GOLD = (1.0, .85, .45), (.35, .75, 1.0), (1.0, .95, .9), (.3, .8, 1.0)
GREEN, ORANGE, VIOLET = (.55, 1.0, .5), (.3, .55, 1.0), (1.0, .55, .8)
F_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
F_R = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
rng = np.random.default_rng(3)

def M(): return np.zeros((H, W), np.uint8)
def glow(G, m, col, core=.9, halo=1.0, s=6, amp=1.0):
    f = m.astype(np.float32) / 255
    G += ((f * core + cv2.GaussianBlur(f, (0, 0), s) * halo) * amp)[..., None] * np.array(col, np.float32)
def txt(m, s, xy, size, font=F_B, anchor='la', fill=255):
    p = Image.fromarray(m); ImageDraw.Draw(p).text(xy, s, fill=fill, font=ImageFont.truetype(font, size), anchor=anchor)
    m[:] = np.array(p)
def T(G, s, xy, size, col, font=F_B, anchor='la', s_=4, amp=1.0):
    m = M(); txt(m, s, xy, size, font, anchor); glow(G, m, col, 1.0, .6, s_, amp)
def P(pt): return (int(round(pt[0])), int(round(pt[1])))
def dotted(m, a, b, step=12, r=2):
    n = int(math.dist(a, b) / step)
    for i in range(n + 1):
        u = i / max(n, 1); cv2.circle(m, P((a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)), r, 255, -1, cv2.LINE_AA)
def dashed(m, a, b, dash=14, gap=9, th=2):
    L = math.dist(a, b); d = 0
    while d < L:
        u0, u1 = d / L, min(d + dash, L) / L
        cv2.line(m, P((a[0] + (b[0] - a[0]) * u0, a[1] + (b[1] - a[1]) * u0)), P((a[0] + (b[0] - a[0]) * u1, a[1] + (b[1] - a[1]) * u1)), 255, th, cv2.LINE_AA)
        d += dash + gap
def arrow(m, a, b, th=3, tip=0.18):
    cv2.arrowedLine(m, P(a), P(b), 255, th, cv2.LINE_AA, tipLength=tip)
def rrect(m, x0, y0, x1, y1, r, th=2, fill=None):
    p = Image.fromarray(m); d = ImageDraw.Draw(p)
    if fill is not None: d.rounded_rectangle((x0, y0, x1, y1), r, fill=fill)
    d.rounded_rectangle((x0, y0, x1, y1), r, outline=255, width=th); m[:] = np.array(p)
def card(img, G, x0, y0, x1, y1, col=CYAN, a=.55):
    """frosted dark glass panel with glowing rim"""
    reg = img[y0:y1, x0:x1]; img[y0:y1, x0:x1] = cv2.GaussianBlur(reg, (0, 0), 12) * (1 - a) + np.array((.09, .05, .03), np.float32) * a
    m = M(); rrect(m, x0, y0, x1, y1, 18, 2); glow(G, m, col, .7, .5, 5)
def scan(G):
    G *= (1 - .16 * ((YY.astype(int) % 4) < 2))[..., None]
def finish(img, G, name, vign=True):
    scan(G)
    out = 1 - (1 - np.clip(img, 0, 1)) * (1 - np.clip(G, 0, 1.5) / 1.5 * 1.0)   # screen blend
    out = out + np.clip(G - 1, 0, 1) * .3
    if vign:
        v = 1 - .35 * (((XX - W / 2) / (W / 2)) ** 2 + ((YY - H / 2) / (H / 2)) ** 2)
        out *= np.clip(v, .4, 1)[..., None]
    for x0, x1 in ((0, W),):
        lens = np.zeros((H, W), np.float32)
    cv2.imwrite(f'{OUT}/{name}.jpg', (np.clip(out, 0, 1) * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('wrote', name)

NAVY = np.array((.075, .035, .018), np.float32)
def void(glow_xy=(W / 2, H / 2), r=700, amt=.10):
    img = np.ones((H, W, 3), np.float32) * NAVY
    g = np.exp(-((XX - glow_xy[0]) ** 2 + (YY - glow_xy[1]) ** 2) / (2 * r * r))
    img += g[..., None] * np.array((.20, .11, .05), np.float32) * amt * 4
    return img
def stars(img, n=260, amp=.8):
    for _ in range(n):
        x, y = rng.uniform(0, W), rng.uniform(0, H); b = rng.uniform(.2, 1) ** 2 * amp
        cv2.circle(img, P((x, y)), 1, (b, b * .95, b * .9), -1, cv2.LINE_AA)
def stage(n, scale=1.0, dx=0, dy=0):
    """presenter (RVM cutout) in the dark void, key light on the face"""
    fr = cv2.imread(f'{D}/pres/{n}/fgr_rvm/00040.png').astype(np.float32) / 255
    a = cv2.imread(f'{D}/pres/{n}/masks_rvm/00040.png', 0).astype(np.float32) / 255
    if scale != 1.0 or dx or dy:
        A = np.float32([[scale, 0, W / 2 - scale * W / 2 + dx], [0, scale, H - scale * H + dy]])
        fr = cv2.warpAffine(fr, A, (W, H)); a = cv2.warpAffine(a, A, (W, H))
    rows = np.where(a.max(1) > .5)[0]; top = rows[0] if len(rows) else 200
    band = a[top:top + int(180 * scale)]; cx = (band.sum(0) * np.arange(W)).sum() / max(band.sum(), 1)
    face = (cx, top + 110 * scale)
    key = .45 + .75 * np.exp(-((XX - face[0]) ** 2 + ((YY - face[1]) * 1.2) ** 2) / (2 * (330 * scale) ** 2))
    person = fr * key[..., None]
    person = person * np.array((1.06, 1.0, .93), np.float32)                     # slight cool grade
    img = void((face[0], face[1] + 150), 650)
    return img * (1 - a[..., None]) + person * a[..., None], a, face
def plate(name, dim=.42, tint=True):
    im = cv2.imread(f'{D}/plates/{name}.jpg').astype(np.float32) / 255
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None]
    im = (im * .55 + g * .45) * dim
    if tint: im *= np.array((1.25, 1.0, .82), np.float32)
    return im
def floor(G, cx, cy, amp=.8, w=520):
    m = M()
    for k, (rx, ry) in enumerate(((w * .45, w * .05), (w * .72, w * .08), (w, w * .11))):
        cv2.ellipse(m, P((cx, cy)), P((rx, ry)), 0, 0, 360, 255, 2 if k < 2 else 1, cv2.LINE_AA)
    for k in range(16):
        a = k * math.pi / 8
        for u in np.linspace(.2, 1, 12):
            cv2.circle(m, P((cx + w * u * math.cos(a), cy + w * .11 * u * math.sin(a))), 1, 200, -1, cv2.LINE_AA)
    glow(G, m, CYAN, .5, .8, 5, amp)
def fireflies(G, n=40, amp=1.0, box=(0, 0, W, H)):
    m = np.zeros((H, W), np.uint8)
    for _ in range(n):
        x, y = rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])
        cv2.circle(m, P((x, y)), int(rng.uniform(1, 3)), int(rng.uniform(120, 255)), -1, cv2.LINE_AA)
    glow(G, m, CYAN, .6, 1.6, 3, amp)

GEO = json.load(open(f'{D}/geo.json'))
def globe(G, cx, cy, R, lon0, lat0, amp=1.0, india=True, grat=True):
    p0 = math.radians(lat0)
    def proj(lon, lat):
        lam, phi = math.radians(lon - lon0), math.radians(lat)
        x = math.cos(phi) * math.sin(lam); y = math.cos(p0) * math.sin(phi) - math.sin(p0) * math.cos(phi) * math.cos(lam)
        z = math.sin(p0) * math.sin(phi) + math.cos(p0) * math.cos(phi) * math.cos(lam)
        return (cx + R * x, cy - R * y), z
    m = M(); cv2.circle(m, P((cx, cy)), int(R), 255, 2, cv2.LINE_AA); glow(G, m, CYAN, .8, 1.6, 10, amp)
    # atmosphere
    d = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    G += (np.exp(-((d - R) / (R * .06)) ** 2) * .35 * amp)[..., None] * np.array(CYAN, np.float32)
    G += ((d < R) * .05 * amp)[..., None] * np.array(CYAN, np.float32)
    if grat:
        m = M()
        for lat in range(-60, 90, 30):
            for lon in range(-180, 180, 3):
                (pt, z) = proj(lon, lat)
                if z > 0: cv2.circle(m, P(pt), 1, 150, -1, cv2.LINE_AA)
        for lon in range(-180, 180, 30):
            for lat in range(-88, 90, 3):
                (pt, z) = proj(lon, lat)
                if z > 0: cv2.circle(m, P(pt), 1, 150, -1, cv2.LINE_AA)
        glow(G, m, CYAN, .5, .4, 2, amp)
    m = M()
    for ring in GEO['land']:
        prev = None
        for lon, lat in ring:
            pt, z = proj(lon, lat)
            if z > 0 and prev is not None: cv2.line(m, P(prev), P(pt), 255, 2, cv2.LINE_AA)
            prev = pt if z > 0 else None
    glow(G, m, CYAN, .9, .9, 4, amp)
    if india:
        m = M()
        for ring in GEO['india']:
            pts = [proj(lo, la) for lo, la in ring]
            if all(z > 0 for _, z in pts): cv2.fillPoly(m, [np.array([P(p) for p, _ in pts], np.int32)], 150, cv2.LINE_AA)
        glow(G, m, AMBER, .8, .8, 5, amp)
    return proj
def eye(m, c, s):
    cx, cy = c
    pts = [(cx + s * math.cos(t), cy - s * .55 * math.sin(t) ** 1) for t in np.linspace(0, math.pi, 30)]
    pts += [(cx + s * math.cos(t), cy + s * .55 * math.sin(t - math.pi) * -1) for t in np.linspace(math.pi, 2 * math.pi, 30)]
    cv2.polylines(m, [np.array([P(p) for p in pts], np.int32)], True, 255, 3, cv2.LINE_AA)
    cv2.circle(m, P(c), int(s * .32), 255, 3, cv2.LINE_AA); cv2.circle(m, P(c), int(s * .12), 255, -1, cv2.LINE_AA)
def clock(m, c, r, hh=10, mm=10, th=3):
    cv2.circle(m, P(c), int(r), 255, th, cv2.LINE_AA)
    for k in range(12):
        a = k * math.pi / 6; u = .82 if k % 3 else .72
        cv2.line(m, P((c[0] + r * u * math.cos(a), c[1] + r * u * math.sin(a))), P((c[0] + r * .92 * math.cos(a), c[1] + r * .92 * math.sin(a))), 255, 2, cv2.LINE_AA)
    for val, L, w in ((hh / 12 + mm / 720, .5, 4), (mm / 60, .75, 3)):
        a = val * 2 * math.pi - math.pi / 2
        cv2.line(m, P(c), P((c[0] + r * L * math.cos(a), c[1] + r * L * math.sin(a))), 255, w, cv2.LINE_AA)
def ruler(m, x0, y0, L, h, ang=0):
    R = cv2.getRotationMatrix2D((0, 0), -ang, 1)
    def t(x, y): v = R @ np.array([x, y, 1.0]); return (x0 + v[0], y0 + v[1])
    cv2.polylines(m, [np.array([P(t(0, 0)), P(t(L, 0)), P(t(L, h)), P(t(0, h))], np.int32)], True, 255, 3, cv2.LINE_AA)
    for k in range(1, 20):
        u = k * L / 20; hh = h * (.5 if k % 5 == 0 else .28)
        cv2.line(m, P(t(u, 0)), P(t(u, hh)), 255, 2, cv2.LINE_AA)
def stick(m, x, y, s, sit=False, th=3):
    cv2.circle(m, P((x, y - s * 1.6)), int(s * .28), 255, th, cv2.LINE_AA)
    cv2.line(m, P((x, y - s * 1.3)), P((x, y - s * .5)), 255, th, cv2.LINE_AA)
    cv2.line(m, P((x, y - s * 1.1)), P((x - s * .4, y - s * .7)), 255, th, cv2.LINE_AA)
    cv2.line(m, P((x, y - s * 1.1)), P((x + s * .4, y - s * .7)), 255, th, cv2.LINE_AA)
    if sit:
        cv2.line(m, P((x, y - s * .5)), P((x + s * .5, y - s * .5)), 255, th, cv2.LINE_AA)
        cv2.line(m, P((x + s * .5, y - s * .5)), P((x + s * .5, y)), 255, th, cv2.LINE_AA)
    else:
        cv2.line(m, P((x, y - s * .5)), P((x - s * .3, y)), 255, th, cv2.LINE_AA)
        cv2.line(m, P((x, y - s * .5)), P((x + s * .3, y)), 255, th, cv2.LINE_AA)
def lens_streak(G, x, y, col=CYAN, amp=.5, L=500):
    s = np.exp(-((YY - y) / 2.2) ** 2) * np.exp(-((XX - x) / L) ** 2)
    G += (s * amp)[..., None] * np.array(col, np.float32)

# ---------------------------------------------------------------- panels
def p01():
    img, a, face = stage('op13'); G = np.zeros_like(img)
    floor(G, face[0], 1010, .6, 620); fireflies(G, 30, .6)
    T(G, 'K I N E M A T I C S', (1480, 430), 56, CYAN, anchor='mm'); T(G, 'Episode 1', (1480, 500), 30, AMBER, F_R, 'mm')
    T(G, 'Is anything truly at rest?', (1480, 560), 30, WHITE, F_R, 'mm', amp=.8)
    lens_streak(G, 1480, 465, CYAN, .5, 420)
    finish(img, G, 'p01')
def p02():
    img, a, face = stage('op24'); G = np.zeros_like(img)
    c = (1500, 470); m = M(); clock(m, c, 170, 0, 50); glow(G, m, CYAN, .9, 1, 6)
    m = M(); cv2.ellipse(m, c, (182, 182), -90, 0, 300, 255, 5, cv2.LINE_AA); glow(G, m, AMBER, .9, 1, 6)
    T(G, '10:00', (c[0], c[1] + 230), 52, WHITE, anchor='mm')
    T(G, 'steps taken   0', (c[0], c[1] + 300), 30, CYAN, F_R, 'mm')
    T(G, 'distance travelled   ?', (c[0], c[1] + 345), 30, AMBER, F_R, 'mm')
    floor(G, face[0], 1010, .5, 600); fireflies(G, 25, .5)
    finish(img, G, 'p02')
def p03():
    img, a, face = stage('op39', .42, 0, -170); G = np.zeros_like(img)
    # room box (perspective cube) around the small presenter, city blocks beyond
    cx, cy = W / 2, 560
    f, b = (430, 300), (300, 210)
    m = M()
    F = [(cx - f[0], cy - f[1]), (cx + f[0], cy - f[1]), (cx + f[0], cy + f[1]), (cx - f[0], cy + f[1])]
    B = [(cx - b[0], cy - b[1] - 30), (cx + b[0], cy - b[1] - 30), (cx + b[0], cy + b[1] - 30), (cx - b[0], cy + b[1] - 30)]
    cv2.polylines(m, [np.array([P(p) for p in B], np.int32)], True, 255, 2, cv2.LINE_AA)
    for p, q in zip(F, B): cv2.line(m, P(p), P(q), 255, 2, cv2.LINE_AA)
    glow(G, m, CYAN, .8, .8, 4)
    m = M()
    for k in range(26):
        x = rng.uniform(60, W - 60)
        if abs(x - cx) < f[0] + 40: continue
        h = rng.uniform(120, 420); w = rng.uniform(50, 110); y = 900
        cv2.rectangle(m, P((x, y - h)), P((x + w, y)), 255, 1, cv2.LINE_AA)
        for yy in np.arange(y - h + 15, y - 10, 22):
            for xx in np.arange(x + 10, x + w - 8, 18): cv2.circle(m, P((xx, yy)), 1, 200, -1)
    glow(G, m, CYAN, .5, .6, 3, .7)
    m = M()
    for x in range(-2000, 4000, 120): cv2.line(m, P((cx + (x - cx) * .3, 900)), P((x, 1080)), 255, 1, cv2.LINE_AA)
    for y in (920, 960, 1015): cv2.line(m, (0, y), (W, y), 255, 1, cv2.LINE_AA)
    glow(G, m, CYAN, .4, .4, 2, .6)
    # progress chips
    labels = ['room', 'city', 'India', 'Earth']
    for k, s in enumerate(labels):
        x = 660 + k * 200; col = AMBER if k == 1 else CYAN
        m = M(); cv2.circle(m, (x, 1035), 7, 255, -1 if k <= 1 else 2, cv2.LINE_AA); glow(G, m, col)
        T(G, s, (x, 1000), 24, col, F_R, 'mm', amp=1 if k <= 1 else .5)
    finish(img, G, 'p03')
def p04():
    img = void((W / 2, H / 2), 900, .06); stars(img); G = np.zeros_like(img)
    proj = globe(G, 820, 560, 420, 80, 18)
    (p, z) = proj(80.0, 13.0)
    m = M(); cv2.circle(m, P(p), 7, 255, -1, cv2.LINE_AA); glow(G, m, WHITE, 1, 1.2, 6)
    # magnifier callout: 250 km eastward along the latitude in 10 min
    q = (1440, 380)
    m = M(); dashed(m, p, (q[0] - 220, q[1] + 40)); cv2.circle(m, q, 220, 255, 2, cv2.LINE_AA); glow(G, m, CYAN, .8, .6, 4)
    m = M(); cv2.ellipse(m, (q[0], q[1] + 260), (520, 260), 0, 230, 310, 255, 3, cv2.LINE_AA); glow(G, m, CYAN, .7, .6, 4, .6)
    a0, a1 = (q[0] - 140, q[1] + 30), (q[0] + 140, q[1] + 30)
    m = M(); dotted(m, a0, a1, 14, 3); glow(G, m, AMBER, 1, 1, 4)
    m = M(); arrow(m, (a1[0] - 40, a1[1]), (a1[0] + 10, a1[1]), 4, .9); cv2.circle(m, P(a0), 9, 255, 2, cv2.LINE_AA); cv2.circle(m, P(a1), 9, 255, -1, cv2.LINE_AA); glow(G, m, WHITE)
    T(G, '≈ 250 km', (q[0], q[1] - 40), 54, AMBER, anchor='mm'); T(G, 'in 10 minutes · without a step', (q[0], q[1] + 100), 26, WHITE, F_R, 'mm')
    m = M(); cv2.ellipse(m, (820, 560), (470, 90), -12, 200, 340, 255, 3, cv2.LINE_AA); arrow(m, (1240, 540), (1268, 510), 3, .9); glow(G, m, CYAN, .8, .8, 4, .7)
    T(G, 'Earth spins once a day', (820, 1035), 28, CYAN, F_R, 'mm')
    finish(img, G, 'p04')
def p05(both=False):
    img, a, face = stage('gu25' if both else 'gu14'); G = np.zeros_like(img)
    x0, y0, x1, y1 = 1180, 300, 1800, 760
    card(img, G, x0, y0, x1, y1)
    T(G, 'Right now, am I…', ((x0 + x1) // 2, y0 + 60), 36, WHITE, F_R, 'mm')
    for k, (s, col) in enumerate((('At rest', CYAN), ('Moving', AMBER))):
        yy = y0 + 150 + k * 140; m = M(); rrect(m, x0 + 50, yy, x1 - 50, yy + 100, 50, 3); glow(G, m, col, .9, .8, 5)
        T(G, s, (x0 + 110, yy + 50), 42, col, anchor='lm')
        if both:
            m = M(); cv2.polylines(m, [np.array([(x1 - 150, yy + 52), (x1 - 128, yy + 75), (x1 - 88, yy + 28)], np.int32)], False, 255, 7, cv2.LINE_AA); glow(G, m, GREEN, 1, 1.2, 6)
    if both: T(G, 'both are correct', ((x0 + x1) // 2, y1 - 40), 32, GREEN, F_B, 'mm')
    else:
        m = M(); cv2.rectangle(m, (1470, 712), (1484, 748), 255, -1); cv2.rectangle(m, (1496, 712), (1510, 748), 255, -1); glow(G, m, WHITE, 1, .8, 4, .8)
        T(G, 'pause & choose', (1530, 730), 26, WHITE, F_R, 'lm', amp=.8)
    floor(G, face[0], 1010, .45, 560)
    finish(img, G, 'p06' if both else 'p05')
def p07():
    img = void((W / 2, 520), 900, .07); G = np.zeros_like(img)
    x0, x1, y0, y1 = 220, 1700, 300, 640
    m = M(); rrect(m, x0, y0, x1, y1, 50, 3)
    for k in range(7):
        wx = x0 + 70 + k * 205; cv2.rectangle(m, (wx, y0 + 50), (wx + 150, y0 + 170), 255, 2, cv2.LINE_AA)
    for wx in (x0 + 140, x1 - 140):
        cv2.circle(m, (wx, y1 + 30), 32, 255, 2, cv2.LINE_AA); cv2.circle(m, (wx + 120, y1 + 30) if wx < W / 2 else (wx - 120, y1 + 30), 32, 255, 2, cv2.LINE_AA)
    glow(G, m, CYAN, .9, .8, 5)
    m = M(); cv2.line(m, (0, y1 + 70), (W, y1 + 70), 255, 2, cv2.LINE_AA)
    for x in range(0, W, 60): cv2.line(m, (x, y1 + 70), (x + 25, y1 + 90), 200, 2, cv2.LINE_AA)
    glow(G, m, CYAN, .6, .6, 3, .7)
    # platform + observer
    m = M(); cv2.rectangle(m, (0, y1 + 150), (W, y1 + 175), 255, 2, cv2.LINE_AA); stick(m, 1500, y1 + 150, 90); glow(G, m, WHITE, .9, .8, 4, .8)
    # passengers
    m = M(); stick(m, 830, y1 - 30, 95, True); glow(G, m, CYAN)
    m = M(); stick(m, 1080, y1 - 30, 95, True); cv2.rectangle(m, (1040, y1 - 150), (1060, y1 - 118), 255, -1); glow(G, m, AMBER)
    T(G, 'you', (830, y1 + 10 - 230), 26, CYAN, F_R, 'mm'); T(G, 'friend', (1080, y1 + 10 - 230), 26, AMBER, F_R, 'mm')
    m = M(); arrow(m, (1720, 470), (1880, 470), 5, .3); glow(G, m, AMBER)
    T(G, '80 km/h', (1800, 420), 34, AMBER, anchor='mm')
    for y in (380, 450, 520, 590):                    # speed streaks behind the train
        m = M(); cv2.line(m, (20, y), (190, y), 255, 2, cv2.LINE_AA); glow(G, m, CYAN, .6, .6, 3, .5)
    x0c = 360; card(img, G, x0c, 120, x0c + 560, 230, CYAN)
    T(G, 'from your seat:  friend is AT REST', (x0c + 280, 175), 26, CYAN, F_B, 'mm')
    card(img, G, 1180, 880, 1820, 990, AMBER)
    T(G, 'from the platform:  friend is MOVING', (1500, 935), 26, AMBER, F_B, 'mm')
    finish(img, G, 'p07')
def p08():
    L = plate('tc62', .62)[:, 480:1440]; R = plate('tw62', .62)[:, 480:1440]
    R = cv2.imread(f'{D}/plates/tw62.jpg').astype(np.float32) / 255
    R = cv2.resize(R[0:1080, 900:1860], (960, 1080)) * .62 * np.array((1.2, 1.0, .85), np.float32)
    img = np.hstack([L, R]).astype(np.float32); G = np.zeros_like(img)
    m = M(); cv2.line(m, (960, 0), (960, H), 255, 2); glow(G, m, CYAN, .8, 1.2, 8)
    # left: dotted level lines, toy fixed
    m = M(); dashed(m, (120, 130), (840, 130)); glow(G, m, CYAN, .8, .6, 3, .8)
    # right: trail of the rising toy
    m = M(); dashed(m, (1000, 470), (1900, 470)); glow(G, m, AMBER, .8, .6, 3, .6)
    m = M(); arrow(m, (1480, 470), (1480, 230), 5, .2); glow(G, m, AMBER)
    for (x, s1, s2, col) in ((480, 'relative to the table', 'AT REST', CYAN), (1440, 'relative to the room', 'MOVING', AMBER)):
        card(img, G, x - 300, 900, x + 300, 1030, col)
        T(G, s1, (x, 940), 28, WHITE, F_R, 'mm'); T(G, s2, (x, 992), 44, col, F_B, 'mm')
    T(G, 'table camera', (40, 50), 24, CYAN, F_R, amp=.7); T(G, 'room camera', (1000, 50), 24, AMBER, F_R, amp=.7)
    finish(img, G, 'p08', False)
def p09():
    img = plate('tc108', .4); G = np.zeros_like(img)
    m = M(); eye(m, (1700, 300), 80); glow(G, m, AMBER, 1, 1, 6)
    T(G, 'observer', (1700, 410), 28, AMBER, F_R, 'mm')
    m = M(); cv2.circle(m, (966, 300), 150, 255, 2, cv2.LINE_AA); glow(G, m, CYAN, .8, .8, 5)
    T(G, 'object', (966, 490), 28, CYAN, F_R, 'mm')
    m = M(); dotted(m, (1125, 300), (1605, 300), 16, 3); glow(G, m, WHITE, 1, 1, 5)
    T(G, 'rest / motion', (1365, 255), 30, WHITE, F_B, 'mm'); T(G, 'a relationship', (1365, 345), 26, WHITE, F_R, 'mm', amp=.8)
    card(img, G, 90, 160, 560, 470, CYAN)
    T(G, 'properties of the object', (325, 210), 24, CYAN, F_R, 'mm')
    for k, s in enumerate(('colour', 'mass')): T(G, '✓  ' + s, (150, 280 + k * 60), 32, GREEN, F_R, 'lm')
    T(G, '✗  rest / motion', (150, 400), 32, ORANGE, F_R, 'lm')
    finish(img, G, 'p09')
def p10():
    img = plate('fr37', .4); G = np.zeros_like(img)
    items = (((348, 470), 'observer', 'who looks', AMBER, 'eye'), ((1061, 580), 'ruler', 'where', CYAN, 'ruler'), ((1557, 640), 'clock', 'when', CYAN, 'clock'))
    for (c, s1, s2, col, kind) in items:
        m = M(); cv2.circle(m, c, 170, 255, 2, cv2.LINE_AA); glow(G, m, col, .7, .8, 6, .8)
        m = M(); ic = (c[0], 170)
        if kind == 'eye': eye(m, ic, 70)
        elif kind == 'ruler': ruler(m, ic[0] - 110, ic[1] - 25, 220, 50)
        else: clock(m, ic, 60)
        glow(G, m, col, 1, 1, 5)
        m = M(); dashed(m, (ic[0], ic[1] + 85), (c[0], c[1] - 180)); glow(G, m, col, .7, .5, 3, .7)
        T(G, s1, (ic[0], ic[1] + 105), 30, col, F_B, 'mm'); T(G, s2, (ic[0] + 0, ic[1] + 140), 24, WHITE, F_R, 'mm', amp=.7)
    card(img, G, 610, 900, 1310, 1020, GOLD)
    T(G, 'FRAME OF REFERENCE', (960, 942), 40, GOLD, F_B, 'mm'); T(G, 'observer + ruler + clock', (960, 990), 24, WHITE, F_R, 'mm', amp=.8)
    finish(img, G, 'p10')
def p11():
    img = plate('fr86', .35); G = np.zeros_like(img)
    card(img, G, 360, 110, 1560, 380, CYAN, .7)
    T(G, '"A car moves at 20 m/s"', (960, 190), 52, WHITE, F_R, 'mm')
    T(G, '…relative to the ground', (960, 300), 52, AMBER, F_B, 'mm')
    m = M(); dashed(m, (560, 345), (1360, 345), 10, 8, 2); glow(G, m, AMBER, .8, .6, 3, .7)
    T(G, 'the hidden phrase', (1520, 420), 24, AMBER, F_R, 'rm', amp=.7)
    finish(img, G, 'p11')
# number-line geometry on the overhead whiteboard (position clip)
O = np.array((799, 572.)); U = np.array((162, -43.))
def NL(v): return O + U * v
def p12():
    img = plate('po47', .5); G = np.zeros_like(img)
    m = M(); cv2.line(m, P(NL(-0.2)), P(NL(4.6)), 255, 3, cv2.LINE_AA); glow(G, m, CYAN, .9, 1, 6)
    f = NL(0); m = M(); cv2.line(m, P(f), P((f[0], f[1] - 170)), 255, 3, cv2.LINE_AA)
    cv2.fillPoly(m, [np.array([P((f[0], f[1] - 170)), P((f[0] + 80, f[1] - 145)), P((f[0], f[1] - 120))], np.int32)], 255, cv2.LINE_AA); glow(G, m, AMBER, 1, 1, 5)
    T(G, 'ORIGIN  (0)', (f[0] - 20, f[1] - 210), 32, AMBER, F_B, 'mm')
    m = M(); arrow(m, NL(3.6), NL(4.8), 6, .4); glow(G, m, GREEN, 1, 1, 6)
    T(G, '+ positive direction', NL(4.4) + (0, -60), 28, GREEN, F_B, 'mm')
    a, b = NL(0) + (0, 40), NL(1) + (0, 40)
    m = M(); cv2.line(m, P(a), P(b), 255, 2, cv2.LINE_AA); cv2.line(m, P(a + (0, -12)), P(a + (0, 12)), 255, 2); cv2.line(m, P(b + (0, -12)), P(b + (0, 12)), 255, 2); glow(G, m, CYAN)
    T(G, 'unit: 1 m', (a + b) / 2 + (0, 50), 26, CYAN, F_R, 'mm')
    finish(img, G, 'p12')
def p13():
    img = plate('po146', .5); G = np.zeros_like(img)
    m = M(); cv2.line(m, P(NL(-3.6)), P(NL(4.6)), 255, 3, cv2.LINE_AA); glow(G, m, CYAN, .9, .9, 6, .8)
    for v, s, col, pp in ((3, '+3 m', VIOLET, (1289, 486)), (-2, '−2 m', ORANGE, (497, 594))):
        p = np.array(pp, float); m = M(); cv2.circle(m, P(p), 46, 255, 3, cv2.LINE_AA); dashed(m, p + (0, -50), p + (0, -150)); glow(G, m, col, 1, 1, 5)
        T(G, s, p + (0, -185), 52, col, F_B, 'mm')
    T(G, 'car', np.array((1289, 486.)) + (60, 40), 24, WHITE, F_R, 'lm', amp=.7); T(G, 'child', np.array((497, 594.)) + (-60, 50), 24, WHITE, F_R, 'rm', amp=.7)
    card(img, G, 1080, 850, 1860, 1010, GOLD)
    T(G, 'origin + direction + unit', (1470, 900), 34, WHITE, F_R, 'mm'); T(G, '= POSITION', (1470, 960), 42, GOLD, F_B, 'mm')
    finish(img, G, 'p13')
def p14():
    img = plate('po146', .5); G = np.zeros_like(img)
    m = M(); cv2.line(m, P(NL(-4.6)), P(NL(4.2)), 255, 3, cv2.LINE_AA)
    for v in range(-4, 5): p = NL(v); cv2.line(m, P(p + (0, -14)), P(p + (0, 14)), 255, 2, cv2.LINE_AA)
    glow(G, m, CYAN, .9, .9, 5, .8)
    p = NL(-4); m = M(); cv2.circle(m, P(p), 40, 255, 3, cv2.LINE_AA); glow(G, m, AMBER, 1, 1.2, 6)
    m = M(); cv2.line(m, P(NL(0) + (0, 60)), P(NL(-4) + (0, 60)), 255, 3, cv2.LINE_AA); arrow(m, NL(-3.5) + (0, 60), NL(-4) + (0, 60), 3, .5); glow(G, m, AMBER, 1, .8, 4)
    T(G, '4 to the left', (NL(-2) + (0, 110)), 28, AMBER, F_R, 'mm')
    T(G, '−4', p + (0, -110), 80, AMBER, F_B, 'mm')
    card(img, G, 1180, 120, 1800, 260, CYAN)
    T(G, 'Quick check: where is the pen?', (1490, 190), 30, WHITE, F_R, 'mm')
    finish(img, G, 'p14')
# whiteboard (two origins) - hand-drawn tick anchors measured on the plate
WB_X = {0: 168, 10: 970, 15: 1421, 20: 1704}; WB_Y = lambda x: 583 + (x - 168) / (1704 - 168) * 11
def wbx(v):
    ks = sorted(WB_X);
    for a, b in zip(ks, ks[1:]):
        if a <= v <= b: return WB_X[a] + (WB_X[b] - WB_X[a]) * (v - a) / (b - a)
    return WB_X[20] + (v - 20) * 57
def p15(after=True):
    img = plate('wb64', .48); G = np.zeros_like(img)
    car = 20 if after else 15
    for (zero, yoff, name, col) in ((0, -140, 'Cheems', GREEN), (10, 170, 'Perro', ORANGE)):
        y = WB_Y(wbx(zero)) + yoff
        m = M(); cv2.line(m, P((wbx(zero) - (40 if zero == 0 else 300), y)), P((wbx(21), y)), 255, 3, cv2.LINE_AA)
        for v in range(zero - (0 if zero == 0 else 3), 21):
            x = wbx(v); h = 14 if (v - zero) % 5 == 0 else 8
            cv2.line(m, P((x, y - h)), P((x, y + h)), 255, 2, cv2.LINE_AA)
        glow(G, m, col, .9, .8, 4, .9)
        m = M(); cv2.circle(m, P((wbx(zero), y)), 10, 255, -1, cv2.LINE_AA); glow(G, m, col, 1, 1.2, 5)
        T(G, f"{name}'s 0", (wbx(zero), y + (-40 if yoff < 0 else 45)), 26, col, F_B, 'mm')
        for v, a in ((15, .45 if after else 1), (20, 1 if after else 0)):
            if a == 0: continue
            x = wbx(v); m = M(); dashed(m, (x, WB_Y(x) - 40), (x, y)); glow(G, m, col, .7, .5, 3, .7 * a)
            T(G, f'{v - zero} m', (x + 16, y + (-28 if yoff < 0 else 30)), 34, col, F_B, 'lm', amp=a)
        if after:
            ya = y + (-70 if yoff < 0 else 75)
            m = M(); cv2.line(m, P((wbx(15), ya)), P((wbx(20), ya)), 255, 3, cv2.LINE_AA); arrow(m, (wbx(19), ya), (wbx(20), ya), 3, .6); glow(G, m, GOLD, 1, 1, 5)
            T(G, 'change = 5 m', ((wbx(15) + wbx(20)) / 2, ya + (-28 if yoff < 0 else 30)), 28, GOLD, F_B, 'mm')
    x = wbx(car); m = M(); cv2.rectangle(m, P((x - 40, WB_Y(x) - 75)), P((x + 40, WB_Y(x) - 40)), 255, 2, cv2.LINE_AA)
    cv2.circle(m, P((x - 22, WB_Y(x) - 36)), 9, 255, 2, cv2.LINE_AA); cv2.circle(m, P((x + 22, WB_Y(x) - 36)), 9, 255, 2, cv2.LINE_AA); glow(G, m, WHITE, 1, 1, 4)
    if after:
        x0 = wbx(15); m = M(); cv2.rectangle(m, P((x0 - 40, WB_Y(x0) - 75)), P((x0 + 40, WB_Y(x0) - 40)), 255, 1, cv2.LINE_AA); glow(G, m, WHITE, .5, .4, 3, .5)
        card(img, G, 120, 900, 820, 1030, GOLD)
        T(G, 'positions differ · the change is the same', (470, 965), 26, GOLD, F_R, 'mm')
    finish(img, G, 'p16' if after else 'p15')
def p17():
    img, a, face = stage('en16'); G = np.zeros_like(img)
    proj = globe(G, 1480, 470, 260, 80, 15, .9)
    m = M(); cv2.ellipse(m, (1480, 470), (330, 70), -8, 190, 350, 255, 3, cv2.LINE_AA); arrow(m, (1785, 455), (1805, 430), 3, .9); glow(G, m, AMBER, .9, .8, 4)
    T(G, '≈ 1,670 km/h', (1480, 800), 54, AMBER, F_B, 'mm'); T(G, 'at the equator · relative to Earth\'s centre', (1480, 855), 24, WHITE, F_R, 'mm', amp=.8)
    T(G, 'relative to the table: 0', (400, 160), 30, CYAN, F_R, 'mm')
    floor(G, face[0], 1010, .4, 560)
    finish(img, G, 'p17')
def p18():
    img = void((W / 2, H / 2), 800, .05); stars(img, 300); G = np.zeros_like(img)
    sun = (760, 540)
    d = np.sqrt((XX - sun[0]) ** 2 + (YY - sun[1]) ** 2)
    G += (np.exp(-(d / 70) ** 2) * 1.6 + np.exp(-(d / 260) ** 2) * .5)[..., None] * np.array((.35, .8, 1.0), np.float32)
    m = M(); cv2.ellipse(m, sun, (560, 330), 0, 0, 360, 255, 2, cv2.LINE_AA); glow(G, m, CYAN, .7, .6, 4, .8)
    e = (sun[0] + 560 * math.cos(-.6), sun[1] + 330 * math.sin(-.6))
    m = M(); cv2.circle(m, P(e), 22, 255, -1, cv2.LINE_AA); glow(G, m, CYAN, 1, 1.2, 6)
    tdir = np.array((-560 * math.sin(-.6), 330 * math.cos(-.6))); tdir /= np.linalg.norm(tdir)
    m = M(); arrow(m, e, np.array(e) + tdir * 170, 5, .25); glow(G, m, AMBER)
    T(G, '30 km every second', (np.array(e) + tdir * 170 + (40, -10)), 36, AMBER, F_B, 'lm')
    card(img, G, 1300, 760, 1860, 1010, CYAN)
    T(G, 'fastest bullet', (1330, 820), 26, WHITE, F_R, 'lm'); m = M(); cv2.rectangle(m, (1560, 808), (1560 + 16, 832), 255, -1); glow(G, m, WHITE, .9, .5, 3)
    T(G, '≈ 1.7 km/s', (1590, 820), 24, WHITE, F_R, 'lm', amp=.7)
    T(G, 'Earth', (1330, 900), 26, AMBER, F_B, 'lm'); m = M(); cv2.rectangle(m, (1460, 888), (1460 + 290, 912), 255, -1); glow(G, m, AMBER, .9, .6, 3)
    T(G, '30 km/s', (1330, 960), 30, AMBER, F_B, 'lm')
    finish(img, G, 'p18')
def p19():
    img, a, face = stage('en70'); G = np.zeros_like(img)
    c = (1480, 420); m = np.zeros((H, W), np.uint8)
    for arm in range(2):
        for k in range(900):
            t = k / 900 * 3.6 * math.pi; r = 22 * math.exp(.19 * t) + rng.normal(0, 9)
            ang = t + arm * math.pi + rng.normal(0, .12)
            x, y = c[0] + r * math.cos(ang), c[1] + r * math.sin(ang) * .42
            if 0 <= x < W and 0 <= y < H: cv2.circle(m, P((x, y)), 1, int(rng.uniform(90, 255)), -1, cv2.LINE_AA)
    glow(G, m, CYAN, .8, 1.4, 3)
    G += (np.exp(-(((XX - c[0]) / 70) ** 2 + ((YY - c[1]) / 30) ** 2)))[..., None] * np.array((.45, .8, 1.0), np.float32)
    sp = (c[0] + 250, c[1] + 40); m = M(); cv2.circle(m, P(sp), 7, 255, -1, cv2.LINE_AA); arrow(m, sp, (sp[0] - 60, sp[1] + 60), 3, .3); glow(G, m, AMBER, 1, 1.2, 5)
    T(G, 'Sun', (sp[0] + 20, sp[1] - 20), 24, AMBER, F_B, 'lm')
    T(G, 'Where is "still"?', (1480, 760), 50, WHITE, F_B, 'mm'); T(G, 'no place the whole universe agrees on', (1480, 820), 26, CYAN, F_R, 'mm')
    floor(G, face[0], 1010, .4, 560)
    finish(img, G, 'p19')
def p20():
    img, a, face = stage('en116'); G = np.zeros_like(img)
    card(img, G, 1160, 280, 1820, 700, AMBER)
    T(G, 'NEXT EPISODE', (1490, 340), 28, AMBER, F_B, 'mm')
    T(G, 'Walk 5 km…', (1490, 430), 50, WHITE, F_R, 'mm'); T(G, '…and go nowhere?', (1490, 510), 50, WHITE, F_R, 'mm')
    m = M(); cv2.ellipse(m, (1490, 610), (150, 45), 0, 0, 360, 255, 2, cv2.LINE_AA); dotted(m, (1340, 610), (1340, 610)); cv2.circle(m, (1340, 610), 8, 255, -1, cv2.LINE_AA); glow(G, m, CYAN, .9, .9, 4)
    floor(G, face[0], 1010, .45, 560); fireflies(G, 25, .5)
    finish(img, G, 'p20')

for f in [globals()[n] for n in sys.argv[2:]] if len(sys.argv) > 2 else (p01, p02, p03, p04, p05, lambda: p05(True), p07, p08, p09, p10, p11, p12, p13, p14, lambda: p15(False), p15, p17, p18, p19, p20):
    f()
