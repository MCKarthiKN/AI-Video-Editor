"""Prepare user-supplied hologram assets.
- PNG/WebP with transparency: kept as-is (best: e.g. the tree).
- Glowing art on a black background: soft-threshold the faint grid away; optional crop and keep-circle.
usage: python3 cut_assets.py black <in> <out.png> [x0 y0 x1 y1]   |   python3 cut_assets.py alpha <in> <out.png>"""
import sys, cv2, numpy as np
mode, src, dst = sys.argv[1:4]
if mode == 'alpha':
    im = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    assert im.ndim == 3 and im.shape[2] == 4, 'no alpha channel: use "black" mode or remove the background first'
    cv2.imwrite(dst, im)
else:
    im = cv2.imread(src)
    if len(sys.argv) == 8:
        x0, y0, x1, y1 = map(int, sys.argv[4:8]); im = im[y0:y1, x0:x1]
    x = im.astype(np.float32) / 255
    a = np.clip((x.max(2) - 0.16) / 0.12, 0, 1)          # removes the dim background grid, keeps the glow
    cv2.imwrite(dst, np.clip(x * a[..., None] * 255, 0, 255).astype(np.uint8))
print('wrote', dst)
