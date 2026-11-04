"""Shared glow / flare / particle helpers for the physics-explainer effects."""
import math
import numpy as np, cv2


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def fs(a, b, x):
    return float(sstep(a, b, x))


def ease_out_back(t, s=1.7):
    t = min(max(t, 0.0), 1.0) - 1
    return t * t * ((s + 1) * t + s) + 1


class Canvas:
    def __init__(self, W, H):
        self.W, self.H = W, H

    def win(self, cx, cy, r):
        x0, x1 = max(0, int(cx - r)), min(self.W, int(cx + r) + 1)
        y0, y1 = max(0, int(cy - r)), min(self.H, int(cy + r) + 1)
        return x0, x1, y0, y1

    def add_gauss(self, G, cx, cy, s, col, amp):
        if amp <= 1e-4 or s <= 0:
            return
        x0, x1, y0, y1 = self.win(cx, cy, 3 * s + 1)
        if x0 >= x1 or y0 >= y1:
            return
        xs = np.arange(x0, x1, dtype=np.float32) - cx
        ys = np.arange(y0, y1, dtype=np.float32) - cy
        g = np.exp(-ys[:, None] ** 2 / (2 * s * s)) * np.exp(-xs[None, :] ** 2 / (2 * s * s)) * amp
        G[y0:y1, x0:x1] += g[..., None] * np.asarray(col, np.float32)

    def add_streak(self, G, cx, cy, sx, sy, col, amp):
        if amp <= 1e-4:
            return
        y0, y1 = max(0, int(cy - 3 * sy - 1)), min(self.H, int(cy + 3 * sy) + 2)
        if y0 >= y1:
            return
        xs = np.arange(self.W, dtype=np.float32) - cx
        ys = np.arange(y0, y1, dtype=np.float32) - cy
        gx = 0.6 * np.exp(-xs ** 2 / (2 * sx * sx)) + 0.4 * np.exp(-np.abs(xs) / (sx * 0.35))
        g = np.exp(-ys ** 2 / (2 * sy * sy))[:, None] * gx[None, :] * amp
        G[y0:y1] += g[..., None] * np.asarray(col, np.float32)

    def add_rays(self, G, cx, cy, length, width, n, rot, col, amp):
        if amp <= 1e-4:
            return
        x0, x1, y0, y1 = self.win(cx, cy, length * 2.5)
        if x0 >= x1 or y0 >= y1:
            return
        dx = np.arange(x0, x1, dtype=np.float32)[None, :] - cx
        dy = np.arange(y0, y1, dtype=np.float32)[:, None] - cy
        acc = np.zeros((y1 - y0, x1 - x0), np.float32)
        for k in range(n):
            th = rot + k * math.pi / n
            c, s = math.cos(th), math.sin(th)
            acc += np.exp(-(-s * dx + c * dy) ** 2 / (2 * width ** 2)) * np.exp(-np.abs(c * dx + s * dy) / length)
        G[y0:y1, x0:x1] += (acc * amp)[..., None] * np.asarray(col, np.float32)

    def hand_flare(self, G, x, y, a, t, k, scale=1.0):
        if a <= 0.01:
            return
        a *= 1 + 0.07 * math.sin(t * 23 + k * 2) + 0.05 * math.sin(t * 37 + k)
        s = scale
        self.add_gauss(G, x, y, 7 * s, (1, 1, 1), 0.9 * a)
        self.add_gauss(G, x, y, 26 * s, (1.0, 0.62, 0.28), 0.75 * a)
        self.add_gauss(G, x, y, 95 * s, (1.0, 0.45, 0.15), 0.32 * a)
        self.add_streak(G, x, y, 520 * s, 2.6 * s, (1.0, 0.72, 0.42), 0.85 * a)
        self.add_streak(G, x, y, 300 * s, 11 * s, (1.0, 0.55, 0.25), 0.22 * a)
        self.add_rays(G, x, y, 70 * s, 1.6 * s, 3, 0.25 * t + k, (1.0, 0.88, 0.75), 0.45 * a)


def blur(img, s):
    if s < 0.5:
        return img
    if s > 10:
        f = 4
        small = cv2.resize(img, (img.shape[1] // f, img.shape[0] // f), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), s / f)
        return cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(img, (0, 0), s)


_i = np.arange(460) + 0.5
_phi = np.arccos(1 - 2 * _i / 460)
_th = math.pi * (1 + 5 ** 0.5) * _i
SPH = np.stack([np.cos(_th) * np.sin(_phi), np.cos(_phi), np.sin(_th) * np.sin(_phi)], 1).astype(np.float32)


def draw_sphere(cv, G, t, cx, cy, R, alpha, col=(0.25, 0.62, 1.0), bright=1.0):
    """Rotating sphere of glowing dots (the 'orange' look)."""
    if R < 1 or alpha <= 0.01:
        return
    a = 0.65 * t
    ca, sa_ = math.cos(a), math.sin(a)
    ct, st = math.cos(0.35), math.sin(0.35)
    x, y, z = SPH[:, 0], SPH[:, 1], SPH[:, 2]
    x1, z1 = ca * x + sa_ * z, -sa_ * x + ca * z
    y2, z2 = ct * y - st * z1, st * y + ct * z1
    order = np.argsort(z2)
    col = np.asarray(col, np.float32)
    margin = int(max(30, R * 0.2) * 3)
    x0, x1_, y0, y1 = cv.win(cx, cy, R + margin)
    if x0 >= x1_ or y0 >= y1:
        return
    C = np.zeros((y1 - y0, x1_ - x0, 3), np.float32)
    dot_r = max(1.6, R * 0.034)
    for k in order:
        depth = (z2[k] + 1) / 2
        px, py = cx + x1[k] * R - x0, cy + y2[k] * R - y0
        b = (0.18 + 0.82 * depth ** 1.6) * bright
        r = dot_r * (0.65 + 0.35 * depth)
        cv2.circle(C, (int(px * 16), int(py * 16)), max(1, int(r * 16)), (col * b).tolist(), -1, cv2.LINE_AA, 4)
    glow = C + blur(C, max(3, R * 0.035)) * 1.3 + blur(C, max(12, R * 0.16)) * 0.7
    G[y0:y1, x0:x1_] += glow * alpha
    cv.add_gauss(G, cx, cy, R * 0.75, col, 0.35 * alpha * bright)
    cv.add_gauss(G, cx, cy, R * 1.5, (1.0, 0.55, 0.22), 0.28 * alpha)


def draw_ball(cv, S, Sa, cx, cy, r, col, alpha=1.0):
    x0, x1, y0, y1 = cv.win(cx, cy, r + 2)
    if x0 >= x1 or y0 >= y1:
        return
    dx = np.arange(x0, x1, dtype=np.float32)[None, :] - cx
    dy = np.arange(y0, y1, dtype=np.float32)[:, None] - cy
    d = np.sqrt(dx * dx + dy * dy)
    nz = np.sqrt(np.clip(1 - (d / r) ** 2, 0, 1))
    L = np.array([-0.45, -0.55, 0.70]); L /= np.linalg.norm(L)
    lam = np.clip(dx / r * L[0] + dy / r * L[1] + nz * L[2], 0, 1)
    c = np.asarray(col, np.float32)[None, None, :] * (0.22 + 0.78 * lam)[..., None] + (0.75 * lam ** 24)[..., None]
    a = (np.clip(r - d + 0.5, 0, 1) * alpha)[..., None]
    S[y0:y1, x0:x1] = c * a + S[y0:y1, x0:x1] * (1 - a)
    Sa[y0:y1, x0:x1] = a + Sa[y0:y1, x0:x1] * (1 - a)


_rng = np.random.default_rng(11)
_p = _rng.normal(size=(14, 3)); _p /= np.linalg.norm(_p, axis=1, keepdims=True)
_p *= _rng.uniform(0.25, 1.0, (14, 1)) ** 0.5
NUC = [(p, (0.42, 0.32, 1.0) if k % 2 == 0 else (1.0, 0.38, 0.22)) for k, p in enumerate(_p)]
ORBITS = [(0.0, 2.3, 0.0), (math.radians(60), -1.9, 2.1), (math.radians(-60), 2.7, 4.2)]


def draw_atom(cv, G, S, Sa, t, cx, cy, sc, alpha):
    """Nucleus of shaded balls + three glowing electron orbits. sc = overall scale."""
    if sc <= 0.01 or alpha <= 0.01:
        return
    ang = 0.6 * t
    c, s = math.cos(ang), math.sin(ang)
    balls = []
    for p, col in NUC:
        x = c * p[0] + s * p[2]; z = -s * p[0] + c * p[2]
        balls.append((z, cx + x * 54 * sc, cy + p[1] * 54 * sc, col))
    balls.sort(key=lambda b: b[0])
    for z, bx, by, col in balls:
        draw_ball(cv, S, Sa, bx, by, 27 * sc * (0.9 + 0.1 * (z + 1) / 2), col, alpha)
    cv.add_gauss(G, cx, cy, 100 * sc, (0.75, 0.45, 1.0), 0.5 * alpha)
    cv.add_gauss(G, cx, cy, 280 * sc, (1.0, 0.5, 0.2), 0.22 * alpha)
    rx, ry = 340 * sc, 100 * sc
    x0, x1, y0, y1 = cv.win(cx, cy, rx + 40)
    if x0 >= x1 or y0 >= y1:
        return
    C = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    for phi_, w, ph in ORBITS:
        cv2.ellipse(C, (int((cx - x0) * 16), int((cy - y0) * 16)), (int(rx * 16), int(ry * 16)),
                    math.degrees(phi_), 0, 360, (1.0, 0.85, 0.45), 3, cv2.LINE_AA, 4)
    G[y0:y1, x0:x1] += (C * 0.10 + blur(C, 5) * 0.35) * alpha
    for phi_, w, ph in ORBITS:
        cp, sp = math.cos(phi_), math.sin(phi_)
        for k in range(14):
            th = w * t + ph - math.copysign(k * 0.07, w)
            lx, ly = rx * math.cos(th), ry * math.sin(th)
            ex, ey = cx + cp * lx - sp * ly, cy + sp * lx + cp * ly
            depth = 1.0 if math.sin(th) > 0 else 0.5
            if k == 0:
                cv.add_gauss(G, ex, ey, 8 * sc / 0.8, (1, 1, 1), 1.3 * alpha * depth)
                cv.add_gauss(G, ex, ey, 26 * sc / 0.8, (1.0, 0.8, 0.35), 0.7 * alpha * depth)
            else:
                cv.add_gauss(G, ex, ey, 6 * sc / 0.8, (1.0, 0.85, 0.5), 0.55 * (1 - k / 14) * alpha * depth)
