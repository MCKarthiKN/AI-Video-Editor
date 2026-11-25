"""Overhead / toy beats. usage: python3 beats_over.py <beat> <out.mp4> [stills t1,t2,...]"""
import sys, math
import numpy as np, cv2
from fx import *

PROXY = {'E01_B_0200_toy_close': 40, 'E01_A_0200_toy_wide': 50, 'E01_A_0400_frame': 0, 'E01_A_0600_position': 1, 'E01_B_0830_whiteboard': 1}
class PReader(Reader):
    def __init__(self, clip, t0, dur):
        s = PROXY[clip]; self.w, self.h = W, H; self.last = None
        self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{t0 - s:.3f}', '-i', f'{D}/proxy/{clip}_{s}.mp4', '-t', f'{dur + .2:.3f}',
                                   '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
def still(clip, ct):
    r = PReader(clip, ct, .1); f = r.read(); r.close(); return f

def pop_ring(G, c, r, t, t0, col, th=3, amp=1.0):
    u = seg(t, t0, t0 + .45)
    if u <= 0: return
    m = M(); cv2.circle(m, P(c), int(r * back(u)), 255, th, cv2.LINE_AA); glow(G, m, col, 1, 1, 6, amp)
    if u < 1:   # expanding shock ring
        m = M(); cv2.circle(m, P(c), int(r * (1 + 1.2 * u)), 255, 2, cv2.LINE_AA); glow(G, m, col, .6, .6, 4, amp * (1 - u))
def pop_text(G, s, xy, size, col, t, t0, w='Medium', anchor='mm', amp=1.0):
    u = seg(t, t0, t0 + .4)
    if u <= 0: return
    T_(G, s, xy, size, col, w, anchor, amp * min(1, u * 3), scale=max(.05, back(u)))

# ================================================================ number line on the whiteboard (position clip)
O = np.array((799, 572.)); U = np.array((160, -27.))          # origin pixel and mean +1 m step (fitted to the drawn ticks)
UP, UN = np.array((163, -29.)), np.array((148, -22.))
def NL(v): return O + (UP if v >= 0 else UN) * v
def nl_ticks(G, vs, t, t0, col=CYAN, amp=1.0, labels=True, stagger=.12):
    for k, v in enumerate(vs):
        tt = t0 + k * stagger; u = seg(t, tt, tt + .3)
        if u <= 0: continue
        p = NL(v); n = np.array((-U[1], U[0])) / np.linalg.norm(U)
        m = M(); cv2.line(m, P(p - n * 16 * back(u)), P(p + n * 16 * back(u)), 255, 3, cv2.LINE_AA); glow(G, m, col, 1, .8, 4, amp)
        if labels: pop_text(G, f'{v:+d}' if v else '0', p + n * 50, 30, col, t, tt, amp=amp)
def beat_position(t, img, G):
    E = lambda ct: T('position', ct)
    # road lane: sweeps on as "this is the road"
    u = ease(seg(t, E(10.0), E(11.6)))
    m = M(); partial_line(m, NL(0), NL(4.7), u, 3); glow(G, m, CYAN, .9, .8, 5, .85)
    un = ease(seg(t, E(82.0), E(83.5)))
    m = M(); partial_line(m, NL(0), NL(-4.6), un, 3); glow(G, m, CYAN, .9, .8, 5, .85)
    T_(G, 'THE ROAD', NL(4.2) + (0, -70), 28, CYAN, 'Regular', amp=env(t, E(10.6), E(17.5)))
    # origin flag
    f = NL(0); uf = seg(t, E(18.6), E(19.1))
    if uf > 0:
        h = 150 * back(uf); m = M(); cv2.line(m, P(f), P((f[0], f[1] - h)), 255, 3, cv2.LINE_AA)
        wv = 8 * math.sin(t * 4)
        cv2.fillPoly(m, [np.array([P((f[0], f[1] - h)), P((f[0] + 78, f[1] - h + 22 + wv)), P((f[0], f[1] - h + 46))], np.int32)], 255, cv2.LINE_AA)
        glow(G, m, AMBER, 1, 1, 6); pop_text(G, 'ORIGIN', (f[0], f[1] - h - 34), 34, AMBER, t, E(18.8))
        pop_ring(G, f, 22, t, E(18.6), AMBER)
    # positive direction
    a0, a1 = NL(3.4), NL(4.9); ua = ease(seg(t, E(34.3), E(35.0)))
    if ua > 0:
        m = M(); arrow(m, a0, a0 + (a1 - a0) * ua, 7, .35); glow(G, m, GREEN, 1, 1.2, 7)
        T_(G, '+ POSITIVE DIRECTION', NL(4.6) + (40, -150), 28, GREEN, amp=min(1, ua * 2))
    # unit: one gap = 1 m
    ub = seg(t, E(46.6), E(47.4))
    if ub > 0:
        n = np.array((-U[1], U[0])) / np.linalg.norm(U); a, b = NL(0) + n * 46, NL(0) + n * 46 + U * ease(ub)
        m = M(); cv2.line(m, P(a), P(b), 255, 2, cv2.LINE_AA)
        for q in (a, b): cv2.line(m, P(q - n * 10), P(q + n * 10), 255, 2, cv2.LINE_AA)
        glow(G, m, CYAN, 1, .8, 4); T_(G, '1 UNIT = 1 M', (a + b) / 2 + n * 46, 26, CYAN, amp=ub)
    nl_ticks(G, [1, 2, 3, 4], t, E(42.4), CYAN, labels=True)
    nl_ticks(G, [-1, -2, -3, -4], t, E(82.2), CYAN, labels=True, stagger=1.2)
    # car at +3 (violet coin) and child at -2 (orange coin)
    car, kid = np.array((1289, 486.)), np.array((497, 594.))
    pop_ring(G, car, 48, t, E(61.6), VIOLET)
    pop_text(G, '+3 M', car + (0, -110), 56, VIOLET, t, E(68.5), 'Bold')
    pop_text(G, 'CAR', car + (78, 34), 24, WHITE, t, E(60.0), 'Regular', 'lm', amp=.8)
    pop_ring(G, kid, 48, t, E(72.9), ORANGE)
    pop_text(G, 'CHILD', kid + (-70, 50), 24, WHITE, t, E(72.9), 'Regular', 'rm', amp=.8)
    pop_text(G, '−2 M', kid + (0, -110), 56, ORANGE, t, E(90.9), 'Bold')
    # "minus is not bad news - it's a direction"
    am = env(t, E(76.6), E(81.5))
    if am > 0:
        card(img, G, 1180, 100, 1840, 240, CYAN, am, seg(t, E(76.6), E(77.0)))
        T_(G, 'MINUS = DIRECTION', (1510, 150), 38, WHITE, 'Medium', amp=am)
        T_(G, 'NOT BAD NEWS', (1510, 200), 24, CYAN, 'Light', amp=am)
    # final card: origin + direction + unit = position
    ac = env(t, E(104.6), E(131.6), .5, .6)
    if ac > 0:
        x0, y0, x1, y1 = 1000, 820, 1880, 1040; card(img, G, x0, y0, x1, y1, GOLD, ac, seg(t, E(104.6), E(105.2)))
        terms = [('ORIGIN', E(112.4)), ('DIRECTION', E(115.1)), ('UNIT', E(117.7))]
        xs = [1090, 1360, 1600]
        miss = env(t, E(124.5), E(128.4), .2, .3)
        for k, ((s, te), x) in enumerate(zip(terms, xs)):
            fl = 1 - .7 * miss * (k == int(t * 2) % 3)
            pop_text(G, s, (x + 40, 880), 30, WHITE, t, te, amp=ac * fl)
            if k < 2: pop_text(G, '+', ((1210, 1515)[k], 880), 30, GOLD, t, te + .2, amp=ac)
        pop_text(G, '= POSITION', (1440, 960), 50, GOLD, t, E(106.6), 'Bold', amp=ac)
        T_(G, 'MISS ONE → THE NUMBER MEANS NOTHING', (1440, 1012), 20, AMBER, 'Light', amp=ac * miss)
    T_(G, 'THE FRAME OF REFERENCE, WRITTEN IN MATHS', (960, 70), 30, GOLD, amp=env(t, E(129.0), E(135.0)))
    # quick check: pen 4 to the left
    aq = env(t, E(136.0), E(151.4), .4, .2)
    if aq > 0:
        card(img, G, 1240, 90, 1840, 230, CYAN, aq, seg(t, E(136.0), E(136.5)))
        T_(G, 'QUICK CHECK', (1540, 135), 34, WHITE, amp=aq); T_(G, 'WHERE IS THE PEN?', (1540, 185), 24, CYAN, 'Light', amp=aq)
        up = ease(seg(t, E(141.3), E(143.2)))
        if up > 0:
            n = np.array((-U[1], U[0])) / np.linalg.norm(U); a = NL(0) + n * 70
            m = M(); partial_line(m, a, a - U * 4, up, 4); glow(G, m, AMBER, 1, 1, 5, aq)
            for k in range(1, 5):
                if up * 4 >= k: m = M(); cv2.circle(m, P(a - U * k), 7, 255, -1, cv2.LINE_AA); glow(G, m, AMBER, 1, 1, 4, aq)
            pop_text(G, '4 STEPS LEFT', a - U * 2 + n * 46, 24, AMBER, t, E(142.4), 'Regular', amp=aq)
        pop_text(G, '?', NL(-4) + (0, -110), 80, WHITE, t, E(145.4), 'Bold', amp=aq * (1 - seg(t, E(149.4), E(149.6))))
        pop_text(G, '−4', NL(-4) + (0, -110), 90, AMBER, t, E(150.5), 'Bold', amp=aq)
        pop_ring(G, NL(-4), 34, t, E(150.5), AMBER, amp=aq)

# ================================================================ two origins (whiteboard clip)
WB_X = {0: 168, 10: 970, 15: 1421, 20: 1704}
def wbx(v):
    ks = sorted(WB_X)
    for a, b in zip(ks, ks[1:]):
        if a <= v <= b: return WB_X[a] + (WB_X[b] - WB_X[a]) * (v - a) / (b - a)
    return WB_X[20] + (v - 20) * 57 if v > 20 else WB_X[0] + v * 80
def wby(x): return 583 + (x - 168) / (1704 - 168) * 11
def beat_origins(t, img, G):
    E = lambda ct: T('origins', ct)
    T_(G, 'TWO OBSERVERS · ONE ROAD', (960, 60), 34, WHITE, amp=env(t, E(3.0), E(11.6)))
    lines = (('CHEEMS', 0, -150, GREEN, E(7.1)), ('PERRO', 10, 165, ORANGE, E(9.6)))
    carv = 15 + 5 * ease(seg(t, E(49.6), E(52.6)))
    for name, zero, yoff, col, t0 in lines:
        y = wby(wbx(zero)) + yoff; ul = ease(seg(t, t0 + .2, t0 + 1.2))
        pop_ring(G, (wbx(zero), wby(wbx(zero))), 52, t, t0, col)
        pop_text(G, f"{name}'S ORIGIN", (wbx(zero), wby(wbx(zero)) + (-88 if yoff < 0 else 96)), 22, col, t, t0 + .1, 'Regular')
        if ul <= 0: continue
        xa, xb = wbx(zero) - (20 if zero == 0 else 260), wbx(21.5)
        m = M(); cv2.line(m, P((xa, y)), P((xa + (xb - xa) * ul, y)), 255, 3, cv2.LINE_AA)
        for v in range(zero - (0 if zero == 0 else 3), 22):
            x = wbx(v)
            if x > xa + (xb - xa) * ul: break
            h = 14 if (v - zero) % 5 == 0 else 7; cv2.line(m, P((x, y - h)), P((x, y + h)), 255, 2, cv2.LINE_AA)
        glow(G, m, col, .95, .8, 4)
        m = M(); cv2.circle(m, P((wbx(zero), y)), 9, 255, -1, cv2.LINE_AA); glow(G, m, col, 1, 1.2, 5)
        T_(G, '0', (wbx(zero), y + (-32 if yoff < 0 else 34)), 24, col, amp=ul)
        # reading of the car on this line
        tr = E(25.6) if zero == 0 else E(44.9)
        ar = seg(t, tr, tr + .4)
        if ar > 0:
            x = wbx(carv); m = M(); dashed(m, (x, wby(x) - 30), (x, y)); glow(G, m, col, .7, .5, 3, .8 * ar)
            val = carv - zero; s = f'{val:.0f} M'
            T_(G, s, (x + 18, y + (-34 if yoff < 0 else 36)), 40, col, 'Bold', 'lm', amp=min(1, ar * 3), scale=back(ar) if ar < 1 else 1)
        # ghost of the old reading after the move + change bracket
        if t > E(49.6):
            x0 = wbx(15); m = M(); dashed(m, (x0, wby(x0) - 30), (x0, y), 6, 8, 1); glow(G, m, col, .5, .3, 3, .45)
            T_(G, f'{15 - zero} M', (x0 - 14, y + (-34 if yoff < 0 else 36)), 28, col, 'Regular', 'rm', amp=.45)
        tc = E(68.8) if zero == 0 else E(74.6)
        uc = seg(t, tc, tc + .6)
        if uc > 0:
            yy = y + (-84 if yoff < 0 else 88); xa_, xb_ = wbx(15), wbx(15) + (wbx(20) - wbx(15)) * ease(uc)
            m = M(); cv2.line(m, P((xa_, yy)), P((xb_, yy)), 255, 3, cv2.LINE_AA)
            for q in (xa_, xb_): cv2.line(m, P((q, yy - 10)), P((q, yy + 10)), 255, 2, cv2.LINE_AA)
            glow(G, m, GOLD, 1, 1, 5)
            s = '20 − 15 = 5 M' if zero == 0 else '10 − 5 = 5 M'
            pop_text(G, s, ((wbx(15) + wbx(20)) / 2, yy + (-34 if yoff < 0 else 36)), 32, GOLD, t, tc + .3, 'Bold')
    # the car itself on the road
    x = wbx(carv); yc = wby(x) - 58; m = M()
    cv2.rectangle(m, P((x - 44, yc - 22)), P((x + 44, yc + 14)), 255, 2, cv2.LINE_AA)
    cv2.line(m, P((x - 26, yc - 22)), P((x - 14, yc - 40)), 255, 2, cv2.LINE_AA); cv2.line(m, P((x - 14, yc - 40)), P((x + 20, yc - 40)), 255, 2, cv2.LINE_AA)
    cv2.line(m, P((x + 20, yc - 40)), P((x + 32, yc - 22)), 255, 2, cv2.LINE_AA)
    for dx in (-24, 24): cv2.circle(m, P((x + dx, yc + 16)), 10, 255, 2, cv2.LINE_AA)
    glow(G, m, WHITE, 1, 1, 5, seg(t, E(24.9), E(25.4)))
    if E(49.6) < t < E(53.0):
        for k in range(1, 6):
            m = M(); cv2.line(m, P((x - 50 - k * 22, yc - 10)), P((x - 50 - k * 22 - 18, yc - 10)), 255, 2, cv2.LINE_AA); glow(G, m, WHITE, .7, .5, 3, .6 * (1 - k / 6))
    ad = env(t, E(79.3), E(88.0))
    if ad > 0:
        card(img, G, 560, 860, 1360, 1000, CYAN, ad, seg(t, E(79.3), E(79.8)))
        T_(G, 'SAME CAR · DIFFERENT POSITIONS', (960, 910), 34, WHITE, amp=ad); T_(G, 'IT DEPENDS ON WHERE YOU PUT ZERO', (960, 960), 22, CYAN, 'Light', amp=ad)
    af = env(t, E(90.6), E(101.0), .5, .5)
    if af > 0:
        card(img, G, 460, 840, 1460, 1020, GOLD, af, seg(t, E(90.6), E(91.1)))
        T_(G, 'CHANGE THE ORIGIN →', (960, 890), 26, WHITE, 'Regular', amp=af)
        T_(G, 'ALL POSITIONS CHANGE', (960, 935), 34, AMBER, amp=af)
        pop_text(G, 'THE STORY OF THE MOTION DOES NOT', (960, 985), 34, GOLD, t, E(94.0), 'Bold', amp=af)

BEATS = {'position': ('E01_A_0600_position', beat_position), 'origins': ('E01_B_0830_whiteboard', beat_origins)}

def render(beat, out, stills=None):
    clip, fn = BEATS[beat]; t0b, t1b = beat_span(beat)
    if stills:
        for k, t in enumerate(stills):
            p, ct = clip_at(beat, t); img = dark_grade(still(clip, ct), .5, .6); G = np.zeros_like(img)
            fn(t, img, G); cv2.imwrite(f'{out}_{k}.jpg', finish(img, G, t))
        return
    wr = Writer(out)
    for p in pieces(beat):
        n = int(round((p['t'] + p['o'] - p['i']) * FPS)) - int(round(p['t'] * FPS))
        r = PReader(clip, p['i'], p['o'] - p['i'])
        for k in range(n):
            t = (int(round(p['t'] * FPS)) + k) / FPS; f = r.read()
            img = dark_grade(f, .5, .6); G = np.zeros_like(img); fn(t, img, G); wr.write(finish(img, G, t))
        r.close()
    wr.close()

# ================================================================ frame of reference kit (overhead table)
def eye(m, c, s, th=3):
    cx, cy = c; pts = [(cx + s * math.cos(a), cy - s * .55 * math.sin(a)) for a in np.linspace(0, math.pi, 30)]
    pts += [(cx + s * math.cos(a), cy - s * .55 * math.sin(a)) for a in np.linspace(math.pi, 2 * math.pi, 30)]
    cv2.polylines(m, [np.array([P(p) for p in pts], np.int32)], True, 255, th, cv2.LINE_AA)
    cv2.circle(m, P(c), int(s * .32), 255, th, cv2.LINE_AA); cv2.circle(m, P(c), int(s * .12), 255, -1, cv2.LINE_AA)
def clock(m, c, r, ang, th=3):
    cv2.circle(m, P(c), int(r), 255, th, cv2.LINE_AA)
    for k in range(12):
        a = k * math.pi / 6; u = .80 if k % 3 else .70
        cv2.line(m, P((c[0] + r * u * math.cos(a), c[1] + r * u * math.sin(a))), P((c[0] + r * .92 * math.cos(a), c[1] + r * .92 * math.sin(a))), 255, 2, cv2.LINE_AA)
    for a, L, w in ((ang / 12, .5, 4), (ang, .78, 3)):
        cv2.line(m, P(c), P((c[0] + r * L * math.sin(a), c[1] - r * L * math.cos(a))), 255, w, cv2.LINE_AA)
def ruler(m, c, L, h, th=3):
    x0, y0 = c[0] - L / 2, c[1] - h / 2; cv2.rectangle(m, P((x0, y0)), P((x0 + L, y0 + h)), 255, th, cv2.LINE_AA)
    for k in range(1, 20):
        u = x0 + k * L / 20; hh = h * (.55 if k % 5 == 0 else .3); cv2.line(m, P((u, y0)), P((u, y0 + hh)), 255, 2, cv2.LINE_AA)
SLOTS = [((356, 560), 'OBSERVER', 'WHO LOOKS', AMBER, 'eye'), ((1061, 600), 'RULER', 'WHERE', CYAN, 'ruler'), ((1557, 612), 'CLOCK', 'WHEN', CYAN, 'clock')]
def beat_frame(t, img, G):
    E = lambda ct: T('frame', ct)
    at = env(t, E(5.9), E(16.0), .6, .5)
    if at > 0:
        u = seg(t, E(5.9), E(6.8)); s = 'FRAME OF REFERENCE'; k = int(len(s) * eout(u))
        T_(G, s[:k] if k < len(s) else s, (960, 300), 74, WHITE, 'Bold', amp=at)
        m = M(); cv2.line(m, (int(960 - 560 * eout(u)), 360), (int(960 + 560 * eout(u)), 360), 255, 2, cv2.LINE_AA); glow(G, m, CYAN, .9, .8, 4, at)
        T_(G, 'THINK OF IT AS A KIT', (960, 410), 30, CYAN, 'Light', amp=env(t, E(9.4), E(16.0)))
    for k, (c, s1, s2, col, kind) in enumerate(SLOTS):
        ts = E(14.7) + k * .25; us = seg(t, ts, ts + .5)
        if us <= 0: continue
        ic = (c[0], 200); ti = [E(16.6), E(31.4), E(36.4)][k]; ui = seg(t, ti, ti + .5)
        m = M(); cv2.circle(m, P(ic), int(92 * back(us)), 255, 2, cv2.LINE_AA); glow(G, m, col, .8, .8, 5, .8)
        T_(G, str(k + 1), (ic[0] - 120, ic[1] - 70), 26, col, 'Regular', amp=us * (1 - ui))
        if ui > 0:
            m = M()
            if kind == 'eye': eye(m, ic, 62 * back(ui))
            elif kind == 'ruler': ruler(m, ic, 150 * back(ui), 40)
            else: clock(m, ic, 52 * back(ui), t * 1.3)
            glow(G, m, col, 1, 1, 5)
            T_(G, s1, (ic[0], ic[1] + 132), 32, col, 'Bold', amp=ui); T_(G, s2, (ic[0], ic[1] + 172), 22, WHITE, 'Light', amp=ui * .85)
            # beam down to the real object once it is on the table
            ub = seg(t, max(ti, E(31.0)), max(ti, E(31.0)) + .6)
            if ub > 0:
                m = M(); dashed(m, (c[0], ic[1] + 195), (c[0], ic[1] + 195 + (c[1] - 110 - ic[1] - 195) * ease(ub)), 10, 8, 2); glow(G, m, col, .8, .6, 3, .7)
                m = M(); cv2.ellipse(m, P(c), (int(130 * eout(ub)), int(60 * eout(ub))), 0, 0, 360, 255, 2, cv2.LINE_AA); glow(G, m, col, .8, .8, 5, .7)
    # different passengers use different kits
    ak = env(t, E(40.2), E(50.2), .5, .5)
    if ak > 0:
        for x, s1, col in ((560, "PASSENGER'S KIT", CYAN), (1360, "PLATFORM'S KIT", AMBER)):
            card(img, G, x - 330, 850, x + 330, 1000, col, ak, seg(t, E(40.2), E(40.7)))
            T_(G, s1, (x, 895), 32, col, amp=ak); T_(G, 'OBSERVER · RULER · CLOCK', (x, 948), 20, WHITE, 'Light', amp=ak * .85)
        pop_text(G, 'DIFFERENT KITS → DIFFERENT ANSWERS', (960, 790), 34, GOLD, t, E(48.0), 'Bold', amp=ak)
def beat_book(t, img, G):
    E = lambda ct: T('book', ct)
    a = env(t, E(77.8), E(92.2), .5, .5)
    if a <= 0: return
    card(img, G, 330, 100, 1590, 400, CYAN, a, seg(t, E(77.8), E(78.4)))
    T_(G, 'TEXTBOOK', (960, 145), 22, CYAN, 'Light', amp=a)
    T_(G, '"A CAR MOVES AT 20 M/S"', (960, 215), 54, WHITE, 'Medium', amp=a)
    # hidden phrase decodes like invisible ink
    u = seg(t, E(86.3), E(87.9)); s = '…RELATIVE TO THE GROUND'
    if u > 0:
        rng = np.random.default_rng(int(t * 25)); k = int(len(s) * u)
        shown = ''.join(ch if (i < k or ch in ' .…') else '·#*/'[rng.integers(4)] for i, ch in enumerate(s))
        T_(G, shown, (960, 315), 52, AMBER, 'Bold', amp=a)
        m = M(); dashed(m, (470, 362), (470 + 980 * u, 362), 10, 8, 2); glow(G, m, AMBER, .8, .6, 3, a * .8)
    pop_text(G, 'THE HIDDEN PHRASE', (1450, 440), 24, AMBER, t, E(84.7), 'Regular', amp=a)

# ================================================================ toy demo: table camera vs room camera
TOY_C = (966, 470)          # toy centre in the table camera
def beat_toy(t, img, G, split=0.0, wide=None):
    E = lambda ct: T('toy', ct)
    # level line at the toy's head (table camera) - it never moves
    a1 = env(t, E(60.3), E(130), .5, .1)
    if a1 > 0:
        xL = 0 if split < .5 else 0; xR = W if split < .01 else W / 2 * (1 - 0) - 20 if split > .99 else W - (W / 2 + 20) * split
        m = M(); dashed(m, (60, 150), (xR - 40, 150), 16, 10, 2); glow(G, m, CYAN, .8, .6, 3, a1 * .8)
        T_(G, 'SAME HEIGHT', (80, 120), 20, CYAN, 'Light', 'lm', amp=a1 * .8)
    ac = env(t, E(61.0), E(130), .5, .1)
    cx = 480 if split > .5 else 960
    if ac > 0:
        card(img, G, cx - 300, 880, cx + 300, 1030, CYAN, ac, seg(t, E(61.0), E(61.5)))
        T_(G, 'RELATIVE TO THE TABLE', (cx, 925), 24, WHITE, 'Light', amp=ac); T_(G, 'AT REST', (cx, 980), 50, CYAN, 'Bold', amp=ac)
    if split > .5:
        m = M(); cv2.line(m, (960, 0), (960, H), 255, 2); glow(G, m, CYAN, .9, 1.2, 8, split)
        T_(G, 'TABLE CAMERA', (40, 50), 22, CYAN, 'Regular', 'lm', amp=split * .8); T_(G, 'ROOM CAMERA', (1000, 50), 22, AMBER, 'Regular', 'lm', amp=split * .8)
        am = env(t, E(77.0), E(130), .5, .1)
        if am > 0:
            card(img, G, 1440 - 300, 880, 1440 + 300, 1030, AMBER, am, seg(t, E(77.0), E(77.5)))
            T_(G, 'RELATIVE TO THE ROOM', (1440, 925), 24, WHITE, 'Light', amp=am); T_(G, 'MOVING', (1440, 980), 50, AMBER, 'Bold', amp=am)
            m = M(); dashed(m, (1000, 600), (1900, 600), 16, 10, 2); glow(G, m, AMBER, .7, .5, 3, am * .6)
            T_(G, 'START LEVEL', (1880, 570), 20, AMBER, 'Light', 'rm', amp=am * .7)
            up = ease(seg(t, E(82.5), E(85.5)))
            m = M(); arrow(m, (1700, 600), (1700, 600 - 330 * up), 6, .18); glow(G, m, AMBER, 1, 1.1, 6, am * min(1, up * 3))
            pop_text(G, '≈ 30–40 CM', (1560, 600 - 330 * up), 40, AMBER, t, E(84.0), 'Bold', 'rm', amp=am)
    T_(G, 'TOY · ON A BOOK · ON MY TABLE', (960, 1000), 26, WHITE, 'Regular', amp=env(t, E(47.4), E(49.6), .4, .3))
def beat_idea(t, img, G):
    E = lambda ct: T('idea', ct)
    T_(G, 'THE BIG IDEA', (960, 70), 40, GOLD, 'Bold', amp=env(t, E(94.4), E(110.9)))
    a = env(t, E(95.6), E(110.9))
    if a > 0:
        card(img, G, 70, 160, 610, 500, CYAN, a, seg(t, E(95.6), E(96.1)))
        T_(G, 'PROPERTIES OF AN OBJECT', (340, 205), 22, CYAN, 'Light', amp=a)
        for k, (s, te, ok) in enumerate((('COLOUR', E(101.0), True), ('MASS', E(101.6), True), ('REST / MOTION', E(97.6), False))):
            u = seg(t, te, te + .4)
            if u <= 0: continue
            y = 280 + k * 70; m = M()
            if ok: cv2.polylines(m, [np.array([(130, y), (148, y + 20), (180, y - 18)], np.int32)], False, 255, 5, cv2.LINE_AA)
            else: cv2.line(m, (132, y - 16), (168, y + 20), 255, 5, cv2.LINE_AA); cv2.line(m, (168, y - 16), (132, y + 20), 255, 5, cv2.LINE_AA)
            glow(G, m, GREEN if ok else ORANGE, 1, 1, 5, a * u)
            T_(G, s, (210, y + 2), 34, GREEN if ok else ORANGE, 'Medium', 'lm', amp=a * u)
    b = env(t, E(105.2), E(110.9), .5, .3)
    if b > 0:
        oc, ec = np.array((966, 330.)), np.array((1640, 330.))
        m = M(); cv2.circle(m, P(oc), int(170 * back(seg(t, E(105.2), E(105.7)))), 255, 2, cv2.LINE_AA); glow(G, m, CYAN, .8, .8, 5, b)
        T_(G, 'OBJECT', oc + (0, 205), 26, CYAN, 'Regular', amp=b)
        ue = seg(t, E(107.4), E(107.9))
        if ue > 0: m = M(); eye(m, ec, 80 * back(ue)); glow(G, m, AMBER, 1, 1, 6, b); T_(G, 'OBSERVER', ec + (0, 110), 26, AMBER, 'Regular', amp=b * ue)
        ul = ease(seg(t, E(108.5), E(109.6)))
        if ul > 0:
            m = M(); dotted(m, oc + (180, 0), oc + (180, 0) + (ec - oc - (280, 0)) * ul, 16, 3); glow(G, m, WHITE, 1, 1, 5, b)
            pop_text(G, 'RELATIONSHIP', (oc + ec) / 2 + (0, -48), 40, WHITE, t, E(110.0), 'Bold', amp=b)

BEATS.update({'frame': ('E01_A_0400_frame', beat_frame), 'book': ('E01_A_0400_frame', beat_book), 'idea': ('E01_B_0200_toy_close', beat_idea)})

def render_toy(out, stills=None):
    """toy_close full frame, then a split with the room camera (rise played from the wide clip)"""
    E = lambda ct: T('toy', ct)
    ts = E(67.6)                                     # "zoom out and look" -> split
    WIDE0, CT0 = 57.0, 69.0                          # wide clip time shown at close-clip time CT0
    def frame(t, fc, fw):
        sp = ease(seg(t, ts, ts + .8))
        img = dark_grade(fc, .55, .65)
        if sp > 0 and fw is not None:
            L = dark_grade(fc[:, 486:1446], .55, .65); R = dark_grade(fw[:, 340:1300], .55, .65)
            split = np.hstack([L, R]); img = img * (1 - sp) + split * sp
        G = np.zeros_like(img); beat_toy(t, img, G, sp); return finish(img, G, t)
    if stills:
        for k, t in enumerate(stills):
            p, ct = clip_at('toy', t); fc = still('E01_B_0200_toy_close', ct)
            fw = still('E01_A_0200_toy_wide', min(79.5, WIDE0 + max(0, ct - CT0))) if t > ts else None
            cv2.imwrite(f'{out}_{k}.jpg', frame(t, fc, fw))
        return
    wr = Writer(out)
    rw = PReader('E01_A_0200_toy_wide', WIDE0, 30); wide_t = WIDE0; fw = None
    for p in pieces('toy'):
        n = int(round((p['t'] + p['o'] - p['i']) * FPS)) - int(round(p['t'] * FPS)); r = PReader('E01_B_0200_toy_close', p['i'], p['o'] - p['i'])
        for k in range(n):
            t = (int(round(p['t'] * FPS)) + k) / FPS; ct = p['i'] + k / FPS; fc = r.read()
            if ct >= CT0 and wide_t < 79.4:
                fw = rw.read(); wide_t += 1 / FPS
            elif t > ts and fw is None:
                fw = still('E01_A_0200_toy_wide', WIDE0)
            wr.write(frame(t, fc, fw))
        r.close()
    wr.close()

if __name__ == '__main__' and sys.argv[1] == 'toy':
    render_toy(sys.argv[2], [float(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else None); sys.exit()

if __name__ == '__main__':
    beat, out = sys.argv[1], sys.argv[2]
    render(beat, out, [float(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else None)
