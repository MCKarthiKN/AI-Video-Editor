"""Presenter beats (opening, guess, ending): RVM cutout on the dark stage + holograms + 3D cutaways.
usage: python3 beats_pres.py <beat> <out.mp4> [stills t1,t2,...]"""
import sys, math
import numpy as np, cv2
from fx import *
from beats_over import eye, clock, ruler, pop_text, pop_ring

S0 = {'E01_A_0000_opening': 10.0, 'E01_A_0100_guess': 10.0, 'E01_A_1030_ending': 4.5}   # first extracted frame (clip s)
CLIP = {'opening': 'E01_A_0000_opening', 'guess': 'E01_A_0100_guess', 'ending': 'E01_A_1030_ending'}
def load(clip, ct):
    n = int(round((ct - S0[clip]) * FPS)) + 1; d = f'{D}/rvm/{clip}'
    fr = cv2.imread(f'{d}/frames/{n:05d}.jpg'); a = cv2.imread(f'{d}/masks_rvm/{n:05d}.png', 0); fg = cv2.imread(f'{d}/fgr_rvm/{n:05d}.png')
    return fr.astype(np.float32) / 255, a.astype(np.float32) / 255, fg.astype(np.float32) / 255
class Vid:
    """sequential reader of a rendered 3D clip, indexed by local time"""
    def __init__(self, path): self.cap = cv2.VideoCapture(path); self.i = -1; self.f = None
    def at(self, tl):
        k = int(round(tl * FPS))
        if k < self.i: self.cap.set(cv2.CAP_PROP_POS_FRAMES, k); self.i = k - 1
        while self.i < k:
            ok, f = self.cap.read()
            if not ok: break
            self.f = f; self.i += 1
        return None if self.f is None else self.f.astype(np.float32) / 255
_face = {}
def face_of(a):
    rows = np.where(a.max(1) > .5)[0]
    if not len(rows): return (W / 2, 300)
    top = rows[0]; band = a[top:top + 170]; cx = (band.sum(0) * np.arange(W)).sum() / max(band.sum(), 1)
    return (cx, top + 115)

def stage(fr, a, fg, zoom, cx, cy, t, lights=1.0):
    """person (zoomed about cx,cy) on the void with a key light on the face; lights<1 blends the real room back in"""
    a = a * np.clip((H - 10 - YY) / 230, 0, 1) ** 1.5           # bottom of the cutout dissolves into the floor glow
    A = np.float32([[zoom, 0, cx - zoom * cx], [0, zoom, cy - zoom * cy]])
    fgz = cv2.warpAffine(fg, A, (W, H), flags=cv2.INTER_LINEAR); az = cv2.warpAffine(a, A, (W, H))[..., None]
    fx_, fy_ = face_of(az[..., 0])
    key = .5 + .72 * np.exp(-((XX - fx_) ** 2 + ((YY - fy_) * 1.15) ** 2) / (2 * (330 * zoom) ** 2))
    person = fgz * key[..., None] * np.array((1.05, 1.0, .94), np.float32)
    bg = void(fx_, fy_ + 260, 680, .55)
    out = bg * (1 - az) + person * az
    if lights < 1:      # lights-out wipe: edges go dark first
        room = cv2.warpAffine(fr, A, (W, H))
        r = np.sqrt(((XX - W / 2) / (W / 2)) ** 2 + ((YY - H / 2) / (H / 2)) ** 2) / 1.42
        m = np.clip((r - (1 - lights) * 1.2) * 6 + .5, 0, 1)[..., None] * (1 - lights) ** .5
        m = np.clip((1 - lights) * 1.0 - (1 - r) * 0, 0, 1) if False else np.clip(1.4 * (1 - lights) - r + .2, 0, 1)[..., None] * (1 - az) + (1 - lights) * az * .6
        out = out * (1 - m) + room * m
    return out, (fx_, fy_)

def cam_for(beat, t):
    """alternate zoom per kept piece to hide jump cuts; slow push inside each piece"""
    for k, p in enumerate(pieces(beat)):
        if p['t'] <= t < p['t'] + p['o'] - p['i'] + 1e-6:
            u = (t - p['t']) / max(.5, p['o'] - p['i'])
            return (1.0 if k % 2 == 0 else 1.1) + .035 * u
    return 1.0

# ------------------------------------------------------------------ overlays
def ov_opening(t, img, G, face, E):
    floor_rings(G, face[0], 1035, t, .5 * seg(t, .6, 1.6)); fireflies(G, t, .45 * seg(t, .8, 2))
    # title
    a = env(t, E(13.2), E(18.6), .6, .6)
    if a > 0:
        u = seg(t, E(13.2), E(14.4)); s = 'KINEMATICS'; k = max(1, int(len(s) * eout(u)))
        T_(G, s[:k], (1450, 420), 84, WHITE, 'Bold', amp=a, track=.25)
        m = M(); cv2.line(m, (int(1450 - 330 * eout(u)), 480), (int(1450 + 330 * eout(u)), 480), 255, 2, cv2.LINE_AA); glow(G, m, CYAN, .9, .8, 4, a)
        T_(G, 'EPISODE 1', (1450, 520), 30, AMBER, 'Regular', amp=a * seg(t, E(14.4), E(15)))
        T_(G, 'IS ANYTHING TRULY AT REST?', (1450, 570), 28, CYAN, 'Light', amp=a * seg(t, E(15.0), E(15.6)))
    # stopwatch: ten minutes, zero steps
    b = env(t, E(19.6), E(30.9), .5, .5)
    if b > 0:
        c = (1460, 430); sw = ease(seg(t, E(19.6), E(21.6)))
        m = M(); clock(m, c, 150, sw * 2 * math.pi * 10); glow(G, m, CYAN, 1, .9, 6, b)
        m = M(); cv2.ellipse(m, c, (164, 164), -90, 0, 360 * sw, 255, 5, cv2.LINE_AA); glow(G, m, AMBER, 1, 1, 6, b)
        T_(G, f'{int(10 * sw):02d}:00', (c[0], c[1] + 210), 54, WHITE, 'Bold', amp=b)
        pop_text(G, 'STEPS · 0', (c[0], c[1] + 275), 28, CYAN, t, E(23.4), 'Regular', amp=b)
        pop_text(G, 'STRETCHES · 0', (c[0], c[1] + 315), 28, CYAN, t, E(25.0), 'Regular', amp=b)
        pop_text(G, 'DISTANCE TRAVELLED?', (c[0], c[1] + 365), 30, AMBER, t, E(26.7), 'Medium', amp=b)
        pop_text(G, 'ZERO?', (c[0], c[1] + 420), 44, WHITE, t, E(28.6), 'Bold', amp=b)
    # zoom-out: room box, then the city
    rb = env(t, E(36.6), E(41.6), .6, .3)
    if rb > 0:
        cx, cy = W / 2, 560; f, bk = (520 * eout(seg(t, E(36.6), E(37.6))), 330), (360, 230)
        m = M()
        F = [(cx - f[0], cy - f[1]), (cx + f[0], cy - f[1]), (cx + f[0], cy + f[1]), (cx - f[0], cy + f[1])]
        B = [(cx - bk[0], cy - bk[1] - 30), (cx + bk[0], cy - bk[1] - 30), (cx + bk[0], cy + bk[1] - 30), (cx - bk[0], cy + bk[1] - 30)]
        cv2.polylines(m, [np.array([P(p) for p in B], np.int32)], True, 255, 2, cv2.LINE_AA)
        for p, q in zip(F, B): cv2.line(m, P(p), P(q), 255, 2, cv2.LINE_AA)
        glow(G, m, CYAN, .8, .8, 4, rb); T_(G, 'ROOM', (cx, cy - 300), 28, CYAN, amp=rb)
    cb = env(t, E(39.3), E(41.6), .7, .3)
    if cb > 0:
        rng = np.random.default_rng(4); m = M(); grow = eout(seg(t, E(39.3), E(40.6)))
        for k in range(30):
            x = rng.uniform(40, W - 140)
            if abs(x + 40 - W / 2) < 420: continue
            h = rng.uniform(120, 440) * grow; w = rng.uniform(50, 110); y = 930
            cv2.rectangle(m, P((x, y - h)), P((x + w, y)), 255, 1, cv2.LINE_AA)
            for yy in np.arange(y - h + 15, y - 10, 22):
                for xx in np.arange(x + 10, x + w - 8, 18): cv2.circle(m, P((xx, yy)), 1, 200, -1)
        for x in range(-2000, 4000, 120): cv2.line(m, P((W / 2 + (x - W / 2) * .3, 930)), P((x, 1080)), 160, 1, cv2.LINE_AA)
        glow(G, m, CYAN, .6, .6, 3, cb); T_(G, 'CITY', (W / 2, 150), 30, CYAN, amp=cb)
    for k, s in enumerate(('ROOM', 'CITY', 'INDIA', 'EARTH')):
        on = seg(t, [E(36.8), E(39.5), E(41.3), E(42.2)][k], [E(36.8), E(39.5), E(41.3), E(42.2)][k] + .3)
        ca = env(t, E(36.6), E(41.8), .4, .3)
        if ca > 0:
            x = 660 + k * 200; m = M(); cv2.circle(m, (x, 1040), 7, 255, -1 if on > .5 else 2, cv2.LINE_AA)
            glow(G, m, AMBER if on > .5 else CYAN, 1, 1, 4, ca); T_(G, s, (x, 1005), 20, AMBER if on > .5 else CYAN, 'Regular', amp=ca * (.5 + .5 * on))
def ov_guess(t, img, G, face, E):
    floor_rings(G, face[0], 1035, t, .45); fireflies(G, t, .4)
    a = env(t, E(12.4), E(31.4), .5, .5)
    if a <= 0: return
    x0, y0, x1, y1 = 90, 270, 730, 790; card(img, G, x0, y0, x1, y1, CYAN, a, seg(t, E(12.4), E(12.9)))
    T_(G, 'RIGHT NOW, AM I…', ((x0 + x1) // 2, y0 + 62), 34, WHITE, 'Regular', amp=a)
    both = seg(t, E(24.4), E(24.9))
    for k, (s, col, te) in enumerate((('AT REST', CYAN, E(13.5)), ('MOVING', AMBER, E(14.4)))):
        u = seg(t, te, te + .4)
        if u <= 0: continue
        yy = y0 + 150 + k * 150; m = M(); rrect(m, x0 + 50, yy, x1 - 50, yy + 108, 54, 3); glow(G, m, col, .9, .8, 5, a * u)
        T_(G, s, (x0 + 110, yy + 54), 46, col, 'Bold', 'lm', amp=a * u)
        if both > 0:
            m = M(); cv2.polylines(m, [np.array([(x1 - 160, yy + 56), (x1 - 134, yy + 82), (x1 - 90, yy + 30)], np.int32)], False, 255, 7, cv2.LINE_AA)
            glow(G, m, GREEN, 1, 1.2, 6, a * both)
    ps = env(t, E(16.3), E(19.0), .3, .3)
    if ps > 0:
        m = M(); cv2.rectangle(m, (190, 720), (208, 770), 255, -1); cv2.rectangle(m, (226, 720), (244, 770), 255, -1); glow(G, m, WHITE, 1, .9, 5, a * ps)
        T_(G, 'PAUSE · CHOOSE ONE', (270, 745), 26, WHITE, 'Regular', 'lm', amp=a * ps)
    T_(G, 'REMEMBER YOUR ANSWER', ((x0 + x1) // 2, y1 - 40), 26, CYAN, 'Light', amp=a * env(t, E(19.4), E(23.8), .3, .3))
    T_(G, 'BOTH ARE CORRECT', ((x0 + x1) // 2, y1 - 40), 32, GREEN, 'Bold', amp=a * env(t, E(24.6), E(28.4), .3, .3))
    T_(G, 'AND NO CONTRADICTION', ((x0 + x1) // 2, y1 - 40), 28, AMBER, 'Medium', amp=a * seg(t, E(28.5), E(29.0)))
def ov_ending(t, img, G, face, E):
    floor_rings(G, face[0], 1035, t, .45); fireflies(G, t, .4)
    a = env(t, E(9.0), E(12.2), .4, .4)
    if a > 0:
        card(img, G, 1270, 380, 1850, 560, CYAN, a, seg(t, E(9.0), E(9.4)))
        T_(G, 'RELATIVE TO THE TABLE', (1560, 430), 24, WHITE, 'Light', amp=a); pop_text(G, '0 KM/H · AT REST', (1560, 500), 42, CYAN, t, E(10.8), 'Bold', amp=a)
    q = env(t, E(55.0), E(62.2), .5, .4)
    if q > 0:
        T_(G, 'IS ANYTHING, ANYWHERE,', (1555, 400), 40, WHITE, 'Medium', amp=q); T_(G, 'TRULY AT REST?', (1555, 465), 64, AMBER, 'Bold', amp=q)
    n = env(t, E(62.4), E(77.2), .5, .5)
    if n > 0:
        card(img, G, 1250, 300, 1870, 620, CYAN, n, seg(t, E(62.4), E(62.9)))
        T_(G, 'ACCORDING TO PHYSICS', (1560, 350), 24, CYAN, 'Light', amp=n)
        T_(G, 'THERE IS NO PLACE', (1560, 425), 44, WHITE, 'Medium', amp=n); T_(G, 'THE WHOLE UNIVERSE', (1560, 485), 44, WHITE, 'Medium', amp=n)
        pop_text(G, 'AGREES IS STILL', (1560, 555), 50, AMBER, t, E(64.2), 'Bold', amp=n)
    f = env(t, E(81.2), E(96.0), .5, .5)
    if f > 0:
        T_(G, 'MOTION IS ALWAYS MEASURED', (1555, 330), 34, WHITE, 'Medium', amp=f * (1 - seg(t, E(93.4), E(93.8))))
        T_(G, 'AGAINST SOMETHING', (1555, 385), 34, CYAN, 'Medium', amp=f * (1 - seg(t, E(93.4), E(93.8))))
        u = seg(t, E(93.5), E(94.1))
        if u > 0:
            T_(G, 'FRAME OF REFERENCE', (1555, 360), 60, GOLD, 'Bold', amp=f, scale=back(u))
            for k, (kind, col) in enumerate((('eye', AMBER), ('ruler', CYAN), ('clock', CYAN))):
                c = (1375 + k * 180, 500); m = M(); uu = seg(t, E(93.8) + k * .15, E(93.8) + k * .15 + .4)
                if uu <= 0: continue
                if kind == 'eye': eye(m, c, 46 * back(uu))
                elif kind == 'ruler': ruler(m, c, 110 * back(uu), 30)
                else: clock(m, c, 40 * back(uu), t)
                glow(G, m, col, 1, 1, 5, f)
    b = env(t, E(98.4), E(112.6), .5, .5)
    if b > 0:
        card(img, G, 1250, 250, 1870, 720, AMBER, b, seg(t, E(98.4), E(98.9)))
        T_(G, 'TRY THIS ON A BUS', (1560, 305), 34, AMBER, 'Bold', amp=b)
        # bus window with a tree outside and a passenger inside
        m = M(); rrect(m, 1300, 360, 1820, 560, 18, 3)
        tx = 1790 - ((t - E(98.4)) * 160) % 460
        cv2.line(m, P((tx, 540)), P((tx, 470)), 255, 3, cv2.LINE_AA); cv2.circle(m, P((tx, 440)), 34, 255, 3, cv2.LINE_AA)
        glow(G, m, GREEN, .9, .8, 4, b * env(t, E(103.6), E(112.6), .4, .4))
        m = M(); rrect(m, 1300, 360, 1820, 560, 18, 3); glow(G, m, CYAN, .9, .8, 4, b)
        m = M(); cv2.circle(m, (1400, 440), 26, 255, 3, cv2.LINE_AA); cv2.line(m, (1400, 466), (1400, 545), 255, 3, cv2.LINE_AA); glow(G, m, CYAN, .9, .8, 4, b * seg(t, E(105.0), E(105.4)))
        pop_text(G, 'TREE OUTSIDE · PASSENGER INSIDE', (1560, 605), 24, WHITE, t, E(104.0), 'Regular', amp=b)
        pop_text(G, 'MOVING? COMPARED TO WHAT?', (1560, 665), 34, AMBER, t, E(107.0), 'Bold', amp=b)
    nx = env(t, E(113.4), E(124.8), .5, .5)
    if nx > 0:
        card(img, G, 1250, 300, 1870, 700, GOLD, nx, seg(t, E(113.4), E(113.9)))
        T_(G, 'NEXT EPISODE', (1560, 355), 28, GOLD, 'Bold', amp=nx)
        pop_text(G, 'WALK 5 KM…', (1560, 440), 52, WHITE, t, E(115.8), 'Medium', amp=nx)
        pop_text(G, '…AND GO NOWHERE?', (1560, 515), 46, AMBER, t, E(118.4), 'Bold', amp=nx)
        u = seg(t, E(116.0), E(120.0)); m = M()
        cv2.ellipse(m, (1560, 610), (190, 50), 0, 180, 180 + 360 * u, 255, 3, cv2.LINE_AA); cv2.circle(m, (1370, 610), 9, 255, -1, cv2.LINE_AA)
        a2 = math.radians(180 + 360 * u); cv2.circle(m, P((1560 + 190 * math.cos(a2), 610 + 50 * math.sin(a2))), 8, 255, -1, cv2.LINE_AA)
        glow(G, m, CYAN, 1, 1, 5, nx)
    h = env(t, E(125.0), E(129.6), .6, .1)
    if h > 0: T_(G, 'HAPPY LEARNING', (1540, 470), 70, WHITE, 'Bold', amp=h, track=.2)

CUT = {'opening': [('earth_open', 28.7, 44.06, 'full')],
       'ending': [('globe', 435.2 + 5.8, 435.2 + 14.6, 'screen'), ('sun', 435.2 + 14.1, 435.2 + 30.1, 'full'), ('galaxy', 435.2 + 30.0, 435.2 + 36.8, 'full')]}
OV = {'opening': ov_opening, 'guess': ov_guess, 'ending': ov_ending}

def frame_at(beat, t, vids):
    clip = CLIP[beat]; p, ct = clip_at(beat, t); E = lambda c: T(beat, c)
    fr, a, fg = load(clip, ct); fa = face_of(a)
    z = cam_for(beat, t)
    if beat == 'opening':        # pull-back: "move the camera back" -> room -> city
        z *= 1 - .45 * ease(seg(t, E(30.9), E(37.4))) - .22 * ease(seg(t, E(39.3), E(41.2)))
    lights = seg(t, .15, 1.0) if beat == 'opening' else 1.0
    ax, ay = fa[0], fa[1]
    if z < 1: k = min(1, (1 - z) / .45); ax, ay = ax + (W / 2 - ax) * k, ay + (H * .78 - ay) * k   # shrink into the room box
    img, face = stage(fr, a, fg, z, ax, ay, t, lights)
    G = np.zeros_like(img); OV[beat](t, img, G, face, E)
    out = finish(img, G, t).astype(np.float32) / 255
    for name, a0, a1, mode in CUT.get(beat, []):
        if a0 - .01 <= t <= a1 + .01:
            v = vids[name].at(t - a0)
            if v is None: continue
            if mode == 'full':
                k = seg(t, a0, a0 + .5) * (1 - seg(t, a1 - .4, a1)) if beat != 'opening' else seg(t, a0, a0 + .7)
                out = out * (1 - k) + v * k
            else:                 # small hologram globe beside the presenter
                g = cv2.resize(v, (W // 2, H // 2)); canvas = np.zeros_like(out); canvas[150:150 + H // 2, 940:940 + W // 2] = g
                out = 1 - (1 - out) * (1 - canvas)
    if beat == 'ending': out *= 1 - .9 * seg(t, E(128.2), E(129.1))
    return (np.clip(out, 0, 1) * 255 + .5).astype(np.uint8)

def render(beat, out, stills=None):
    vids = {n: Vid(f'{D}/r3d/{n}.mp4') for n, *_ in CUT.get(beat, [])}
    if stills:
        for k, t in enumerate(stills): cv2.imwrite(f'{out}_{k}.jpg', frame_at(beat, t, vids))
        return
    t0, t1 = beat_span(beat); wr = Writer(out)
    for i in range(int(round(t0 * FPS)), int(round(t1 * FPS))): wr.write(frame_at(beat, i / FPS, vids))
    wr.close()

def render_range(beat, out, a, b):
    vids = {n: Vid(f'{D}/r3d/{n}.mp4') for n, *_ in CUT.get(beat, [])}; wr = Writer(out)
    for i in range(a, b): wr.write(frame_at(beat, i / FPS, vids))
    wr.close()

if __name__ == '__main__':
    if len(sys.argv) > 3 and sys.argv[3].startswith('frames:'):
        a, b = map(int, sys.argv[3][7:].split('-')); render_range(sys.argv[1], sys.argv[2], a, b)
    else:
        render(sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else None)
