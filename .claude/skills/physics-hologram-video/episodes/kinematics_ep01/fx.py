"""Shared helpers for the Kinematics Ep 1 compositor (1920x1080, 25 fps)."""
import os, json, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

D = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1920, 1080, 25
CYAN, AMBER, WHITE, GOLD = (1.0, .85, .45), (.35, .72, 1.0), (1.0, .96, .92), (.32, .80, 1.0)
GREEN, ORANGE, VIOLET = (.55, 1.0, .55), (.30, .55, 1.0), (1.0, .55, .85)
FONT = {k: f'{D}/fontzip/LEMONMILK-{k}.otf' for k in ('Light', 'Regular', 'Medium', 'Bold')}
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
EDL = json.load(open(f'{D}/edl.json'))
SIL = {}
for _l in open(f'{D}/silences.txt'):
    _c, _r, _ = _l.split(' ', 2); _a, _b = map(float, _r.split('-')); SIL.setdefault(_c, []).append((_a, _b))

# ---------------------------------------------------------------- timing
def pieces(beat): return [p for p in EDL['pieces'] if p['beat'] == beat]
def beat_span(beat):
    ps = pieces(beat); return ps[0]['t'], ps[-1]['t'] + ps[-1]['o'] - ps[-1]['i']
def T(beat, clip_t):
    """timeline second of a clip moment inside a beat (snaps into the nearest kept piece)"""
    best = None
    for p in pieces(beat):
        if p['i'] <= clip_t <= p['o']: return p['t'] + clip_t - p['i']
        d = min(abs(clip_t - p['i']), abs(clip_t - p['o']))
        tt = p['t'] if clip_t < p['i'] else p['t'] + p['o'] - p['i']
        if best is None or d < best[0]: best = (d, tt)
    return best[1]
def clip_at(beat, t):
    for p in pieces(beat):
        if p['t'] <= t < p['t'] + p['o'] - p['i'] + 1e-6: return p, p['i'] + t - p['t']
    p = pieces(beat)[-1]; return p, p['o'] - 1e-3

# ---------------------------------------------------------------- easing
def cl(x, a=0., b=1.): return min(b, max(a, x))
def seg(t, a, b): return cl((t - a) / (b - a)) if b != a else float(t >= a)
def ease(x): x = cl(x); return 4 * x * x * x if x < .5 else 1 - (-2 * x + 2) ** 3 / 2
def eout(x): x = cl(x); return 1 - (1 - x) ** 3
def back(x, c=1.8): x = cl(x); return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2
def env(t, a, b, fi=.35, fo=.35): return eout(seg(t, a, a + fi)) * (1 - ease(seg(t, b - fo, b)))

# ---------------------------------------------------------------- drawing
def M(): return np.zeros((H, W), np.uint8)
def glow(G, m, col, core=.95, halo=.9, s=6, amp=1.0):
    if amp <= 0.002: return
    f = m.astype(np.float32) / 255
    G += ((f * core + cv2.GaussianBlur(f, (0, 0), s) * halo) * amp)[..., None] * np.array(col, np.float32)
def P(p): return (int(round(p[0])), int(round(p[1])))
_fc = {}
def font(w, size):
    k = (w, size)
    if k not in _fc: _fc[k] = ImageFont.truetype(FONT[w], size)
    return _fc[k]
def text_mask(s, size, w='Medium', track=0.08):
    """tight uint8 mask of a text line (Lemon Milk, letter-spaced)"""
    f = font(w, size); sp = size * track
    widths = [f.getlength(ch) for ch in s]; tw = int(sum(widths) + sp * (len(s) - 1)) + 8
    asc, desc = f.getmetrics(); im = Image.new('L', (max(tw, 1), asc + desc + 8)); d = ImageDraw.Draw(im); x = 4
    for ch, cw in zip(s, widths): d.text((x, 4), ch, fill=255, font=f); x += cw + sp
    a = np.array(im); ys = np.where(a.max(1) > 0)[0]
    return a[max(0, ys[0] - 2):ys[-1] + 3] if len(ys) else a
_tc = {}
def T_(G, s, xy, size, col, w='Medium', anchor='mm', amp=1.0, scale=1.0, track=.08, halo=.55, sblur=4):
    """glowing hologram text; anchor l/m/r + t/m/b"""
    if amp <= .002: return
    k = (s, size, w, track)
    if k not in _tc: _tc[k] = text_mask(s, size, w, track)
    a = _tc[k]
    if scale != 1.0: a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w_ = a.shape; x, y = xy
    x0 = x - (0 if anchor[0] == 'l' else w_ / 2 if anchor[0] == 'm' else w_)
    y0 = y - (0 if anchor[1] == 't' else h / 2 if anchor[1] == 'm' else h)
    x0, y0 = int(round(x0)), int(round(y0))
    m = M(); xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w_), min(H, y0 + h)
    if xb <= xa or yb <= ya: return
    m[ya:yb, xa:xb] = a[ya - y0:yb - y0, xa - x0:xb - x0]
    glow(G, m, col, 1.0, halo, sblur, amp)
def dashed(m, a, b, dash=14, gap=9, th=2, v=255):
    L = math.dist(a, b); d = 0
    while d < L:
        u0, u1 = d / L, min(d + dash, L) / L
        cv2.line(m, P((a[0] + (b[0] - a[0]) * u0, a[1] + (b[1] - a[1]) * u0)), P((a[0] + (b[0] - a[0]) * u1, a[1] + (b[1] - a[1]) * u1)), v, th, cv2.LINE_AA)
        d += dash + gap
def dotted(m, a, b, step=12, r=2, v=255):
    n = max(1, int(math.dist(a, b) / step))
    for i in range(n + 1):
        u = i / n; cv2.circle(m, P((a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)), r, v, -1, cv2.LINE_AA)
def partial_line(m, a, b, u, th=3, v=255):
    if u <= 0: return
    cv2.line(m, P(a), P((a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)), v, th, cv2.LINE_AA)
def arrow(m, a, b, th=3, tip=.2, v=255): cv2.arrowedLine(m, P(a), P(b), v, th, cv2.LINE_AA, tipLength=tip)
def rrect(m, x0, y0, x1, y1, r, th=2, v=255):
    p = Image.fromarray(m); ImageDraw.Draw(p).rounded_rectangle((x0, y0, x1, y1), r, outline=v, width=th); m[:] = np.array(p)
def card(img, G, x0, y0, x1, y1, col=CYAN, amp=1.0, build=1.0):
    """frosted glass card; build 0..1 grows it open from the centre line"""
    if amp <= .002: return
    cy = (y0 + y1) / 2; hh = (y1 - y0) / 2 * eout(build); y0b, y1b = int(cy - hh), int(cy + hh)
    if y1b - y0b < 4: return
    x0, x1 = int(x0), int(x1); reg = img[y0b:y1b, x0:x1]
    frost = cv2.GaussianBlur(reg, (0, 0), 14) * .55 + np.array((.10, .05, .02), np.float32) * .45
    img[y0b:y1b, x0:x1] = reg * (1 - .8 * amp) + frost * .8 * amp
    m = M(); rrect(m, x0, y0b, x1, y1b, 16, 2); glow(G, m, col, .8, .6, 5, amp)
def scanlines(G, t):
    G *= (1 - .14 * ((YY.astype(np.int32) % 4) < 2))[..., None]
    band = np.exp(-((YY - ((t * 260) % (H + 400) - 200)) / 40) ** 2)[..., None]
    G *= (1 + .35 * band)
def flicker(t): return .94 + .06 * math.sin(t * 37.0) * math.sin(t * 11.3)

def finish(img, G, t=0.0, vign=.32):
    """screen-blend the hologram layer, cool grade, vignette -> uint8 BGR"""
    scanlines(G, t); G *= flicker(t)
    g = np.clip(G, 0, 2.5)
    out = 1 - (1 - np.clip(img, 0, 1)) * (1 - np.clip(g, 0, 1))
    out += np.clip(g - 1, 0, 1.5) * .25
    if vign:
        v = 1 - vign * (((XX - W / 2) / (W / 2)) ** 2 + ((YY - H / 2) / (H / 2)) ** 2)
        out *= np.clip(v, .45, 1)[..., None]
    return (np.clip(out, 0, 1) * 255 + .5).astype(np.uint8)

# ---------------------------------------------------------------- footage
def ext(c): return 'mp4' if 'toy_wide' in c else 'mov'
class Reader:
    """sequential frame reader for a clip time range at 25 fps; returns float32 BGR 0..1 at (w,h)"""
    def __init__(self, clip, t0, dur, w=W, h=H, vf=''):
        self.w, self.h = w, h
        f = f'fps=25,scale={w}:{h}:flags=lanczos' + (',' + vf if vf else '')
        self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{t0:.3f}', '-i', f'{D}/in/{clip}.{ext(clip)}', '-t', f'{dur + .2:.3f}',
                                   '-vf', f, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
        self.last = None
    def read(self):
        b = self.p.stdout.read(self.w * self.h * 3)
        if len(b) == self.w * self.h * 3: self.last = np.frombuffer(b, np.uint8).reshape(self.h, self.w, 3)
        return None if self.last is None else self.last.astype(np.float32) / 255
    def close(self):
        try: self.p.kill()
        except Exception: pass
class Writer:
    def __init__(self, path):
        self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '25', '-i', '-',
                                   '-c:v', 'libx264', '-crf', '15', '-preset', 'medium', '-pix_fmt', 'yuv420p', path], stdin=subprocess.PIPE)
    def write(self, u8): self.p.stdin.write(u8.tobytes())
    def close(self): self.p.stdin.close(); self.p.wait()

def dark_grade(im, dim=.42, sat=.55, cool=True):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None]
    out = (im * sat + g * (1 - sat)) * dim
    if cool: out = out * np.array((1.22, 1.0, .84), np.float32)
    return out
NAVY = np.array((.075, .035, .018), np.float32)
def void(cx=W / 2, cy=H * .55, r=700, amt=.4):
    img = np.ones((H, W, 3), np.float32) * NAVY
    g = np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * r * r))
    return img + g[..., None] * np.array((.20, .11, .05), np.float32) * amt
def floor_rings(G, cx, cy, t, amp=.6, w=560):
    m = M()
    for k, (rx, ry) in enumerate(((w * .45, w * .05), (w * .72, w * .08), (w, w * .11))):
        cv2.ellipse(m, P((cx, cy)), P((rx, ry)), 0, 0, 360, 255, 2 if k < 2 else 1, cv2.LINE_AA)
    for k in range(16):
        a = k * math.pi / 8 + t * .05
        for u in np.linspace(.2, 1, 12):
            cv2.circle(m, P((cx + w * u * math.cos(a), cy + w * .11 * u * math.sin(a))), 1, 200, -1, cv2.LINE_AA)
    glow(G, m, CYAN, .45, .7, 5, amp)
_FF = np.random.default_rng(5)
FIRE = [dict(x=_FF.uniform(0, W), y=_FF.uniform(0, H), sp=_FF.uniform(6, 20), ph=_FF.uniform(0, 6.3), r=_FF.uniform(1.2, 2.8),
             c=CYAN if _FF.random() < .6 else AMBER) for _ in range(55)]
def fireflies(G, t, amp=.6):
    m1, m2 = M(), M()
    for f in FIRE:
        y = (f['y'] - f['sp'] * t) % H; x = f['x'] + 16 * math.sin(t * .5 + f['ph'])
        tw = int(150 + 105 * math.sin(t * 2.6 + f['ph']))
        cv2.circle(m1 if f['c'] is CYAN else m2, P((x, y)), int(f['r'] + .5), tw, -1, cv2.LINE_AA)
    glow(G, m1, CYAN, .5, 1.5, 3, amp); glow(G, m2, AMBER, .5, 1.5, 3, amp)
