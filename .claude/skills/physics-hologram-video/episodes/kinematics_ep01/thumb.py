import cv2, numpy as np, math, sys
from fx import *
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1300
d = 'rvm/E01_A_1030_ending'
fg = cv2.imread(f'{d}/fgr_rvm/{n:05d}.png').astype(np.float32) / 255
a = cv2.imread(f'{d}/masks_rvm/{n:05d}.png', 0).astype(np.float32) / 255
a *= np.clip((H - 10 - YY) / 260, 0, 1) ** 1.4
# place the presenter big on the left
s, dx, dy = 1.06, -470, 75
A = np.float32([[s, 0, W / 2 - s * W / 2 + dx], [0, s, H - s * H + dy]])
fg = cv2.warpAffine(fg, A, (W, H)); a = cv2.warpAffine(a, A, (W, H))[..., None]
rows = np.where(a[..., 0].max(1) > .5)[0]; top = rows[0]; band = a[top:top + 220, :, 0]
fx_ = (band.sum(0) * np.arange(W)).sum() / band.sum(); fy_ = top + 150
key = .55 + .8 * np.exp(-((XX - fx_) ** 2 + ((YY - fy_) * 1.1) ** 2) / (2 * 380 ** 2))
person = np.clip(fg * key[..., None] * np.array((1.04, 1.0, .95), np.float32) * 1.08, 0, 1)
img = void(1350, 560, 900, .8)
# hologram Earth (last frame of the opening cutaway) on the right
e = cv2.imread('gfx/f_earth_open/00020.png').astype(np.float32) / 255
e = cv2.warpAffine(e, np.float32([[1.25, 0, 1400 - 1.25 * 960], [0, 1.25, 600 - 1.25 * 540]]), (W, H))
img = 1 - (1 - img) * (1 - e * np.clip((XX - 700) / 300, 0, 1)[..., None])
img = img * (1 - a) + person * a
G = np.zeros_like(img)
# glowing 250 km trail on India (exaggerated for readability) + the fixed point
P0, P1 = np.array((1452., 712.)), np.array((1640., 690.))     # fixed glowing point, START ring
pts = [P0 + (P1 - P0) * u + np.array((0, -26 * math.sin(math.pi * u))) for u in np.linspace(0, 1, 60)]
for k in range(len(pts) - 1):
    u = k / (len(pts) - 1); m = M(); cv2.line(m, P(pts[k]), P(pts[k + 1]), 255, 7, cv2.LINE_AA); glow(G, m, AMBER, .8 + .9 * (1 - u) ** 3, 1.0 + 1.2 * (1 - u) ** 3, 7)
m = M(); cv2.circle(m, P(P0), 15, 255, -1, cv2.LINE_AA); glow(G, m, WHITE, 1.5, 2.6, 16)
m = M(); cv2.circle(m, P(P1), 18, 255, 3, cv2.LINE_AA); glow(G, m, CYAN, 1, 1.1, 6)
T_(G, 'START', P1 + (0, 46), 24, CYAN, 'Medium', amp=.9)
T_(G, '250 KM', (1390, 200), 190, AMBER, 'Bold', amp=1.25, track=.04, halo=.8, sblur=10)
T_(G, 'WITHOUT MOVING?', (1390, 330), 78, WHITE, 'Bold', amp=1.15, track=.06, halo=.6, sblur=6)
T_(G, 'KINEMATICS · EP 1', (70, 70), 40, CYAN, 'Medium', 'lm', amp=1.0, track=.18)
T_(G, 'IS ANYTHING TRULY AT REST?', (1390, 1010), 40, CYAN, 'Medium', amp=1.0, track=.12)
out = 1 - (1 - np.clip(img, 0, 1)) * (1 - np.clip(G, 0, 1)); out += np.clip(G - 1, 0, 1) * .25
v = 1 - .25 * (((XX - W / 2) / (W / 2)) ** 2 + ((YY - H / 2) / (H / 2)) ** 2); out *= np.clip(v, .5, 1)[..., None]
o = (np.clip(out, 0, 1) * 255).astype(np.uint8)
cv2.imwrite('E01_X_thumbnail_1080.png', o)
cv2.imwrite('E01_X_thumbnail.png', cv2.resize(o, (1280, 720), interpolation=cv2.INTER_AREA))
