"""Count rendered PNGs in a folder tree and confirm every one is 1080x1350.

Usage: python3 -I check_images.py QUOTE_CARDS/output/<book>
"""
import os, struct, sys
n = bad = 0
for root, _, files in os.walk(sys.argv[1]):
    for f in files:
        if f.endswith(".png"):
            with open(os.path.join(root, f), "rb") as fh:
                w, h = struct.unpack(">II", fh.read(24)[16:24])
            n += 1
            if (w, h) != (1080, 1350):
                bad += 1
                print("wrong size:", f, w, h)
print(n, "pngs, wrong size:", bad)
sys.exit(1 if bad else 0)
