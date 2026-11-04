"""Turn a flat glowing image into a projected 2.5D hologram (depth parallax, sway, scanlines, projector beam)."""
import math
import numpy as np, cv2


class Hologram:
    def __init__(self, img_bgr, height, seed=1):
        s = height / img_bgr.shape[0]
        self.img = cv2.resize(img_bgr, (int(img_bgr.shape[1] * s), int(height)), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        self.h, self.w = self.img.shape[:2]
        lum = self.img.max(2)
        self.alpha = np.clip(lum * 3, 0, 1)
        # pseudo-depth: nearer towards the middle of the shape, farther at its edges
        m = (cv2.GaussianBlur(self.alpha, (0, 0), 6) > 0.15).astype(np.uint8)
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        self.depth = cv2.GaussianBlur(dt / (dt.max() + 1e-6), (0, 0), 25).astype(np.float32)
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        self.yy, self.xx = yy, xx
        self.top = np.clip(1 - yy / (self.h * 0.85), 0, 1) ** 2          # sway grows towards the crown
        e = cv2.morphologyEx((self.alpha > 0.2).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
        self.edge = cv2.GaussianBlur(e, (0, 0), 2.5)
        self.rng = np.random.default_rng(seed)
        self.dust = [dict(x=self.rng.uniform(0.25, 0.75), sp=self.rng.uniform(0.05, 0.15), ph=self.rng.uniform(0, 1),
                          r=self.rng.uniform(1.2, 2.8)) for _ in range(70)]

    def frame(self, t, speed=1.0, tsim=None, freeze=False, alpha=1.0):
        """Return (rgb, depth_offset) of the hologram at time t. `tsim` is animation time (slowed/stopped by the knob)."""
        ts = t if tsim is None else tsim
        yaw = 0.07 * math.sin(2 * math.pi * ts / 7.0)
        sway = 7.0 * math.sin(2 * math.pi * ts / 3.1) + 3.0 * math.sin(2 * math.pi * ts / 1.7)
        cx = self.w / 2
        dx = yaw * (self.depth * 160 + (self.xx - cx) * 0.25 * self.depth) + sway * self.top
        dy = -abs(yaw) * self.depth * 20
        mx = (self.xx - dx).astype(np.float32); my = (self.yy - dy).astype(np.float32)
        img = cv2.remap(self.img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        edge = cv2.remap(self.edge, mx, my, cv2.INTER_LINEAR)
        # chromatic fringe
        b, g, r = cv2.split(img)
        b = np.roll(b, 2, 1); r = np.roll(r, -2, 1)
        img = cv2.merge([b, g, r])
        # holographic tint + outline
        lum = img.max(2, keepdims=True)
        img = img * 0.78 + lum * np.array([1.0, 0.85, 0.35], np.float32) * 0.22
        img += edge[..., None] * np.array([1.0, 0.85, 0.4], np.float32) * 0.35
        # scanlines + rising scan band
        sl = 0.86 + 0.14 * np.sin(self.yy * (2 * math.pi / 4.0) + t * 6)
        band_y = self.h * (1 - ((t * 0.35) % 1.3))
        band = 1 + 0.55 * np.exp(-((self.yy - band_y) / 18) ** 2)
        img = img * (sl * band)[..., None]
        # flicker + rare glitch rows
        fl = 0.93 + 0.07 * math.sin(t * 37) * math.sin(t * 11)
        if not freeze:
            r = np.random.default_rng(int(t * 25))
            if r.random() < 0.06:
                y0 = int(r.uniform(0, self.h - 40)); hh = int(r.uniform(6, 30))
                img[y0:y0 + hh] = np.roll(img[y0:y0 + hh], int(r.uniform(-25, 25)), 1)
        img *= fl * alpha
        # projector beam from the base rings + rising dust
        beam = np.exp(-((self.xx - cx) / (self.w * (0.10 + 0.32 * (self.yy / self.h) ** 0.0 * (1 - self.yy / self.h)))) ** 2)
        beam *= np.clip((self.yy / self.h - 0.15) / 0.85, 0, 1) ** 1.5 * 0.10
        img += beam[..., None] * np.array([1.0, 0.8, 0.35], np.float32) * alpha
        dimg = np.zeros((self.h, self.w), np.float32)
        for d in self.dust:
            u = (d['ph'] + (ts * d['sp'])) % 1.0
            x = int(self.w * d['x'] + 25 * math.sin(ts * 0.7 + d['ph'] * 9)); y = int(self.h * (1 - u))
            if 0 <= x < self.w and 0 <= y < self.h:
                cv2.circle(dimg, (x, y), int(d['r']), 0.6 * math.sin(math.pi * u), -1, cv2.LINE_AA)
        img += cv2.GaussianBlur(dimg, (0, 0), 1.5)[..., None] * np.array([1.0, 0.9, 0.6], np.float32) * alpha
        return np.clip(img, 0, 1.5)
