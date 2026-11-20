"""Search an extracted book text (with [[P<n>]] markers) and print hits with page numbers.

Usage:
  python3 -I find_text.py BOOK.txt [--ctx 400] [--max 1] [--pages 20-180] [--loose] PATTERN [PATTERN ...]

  PATTERN   a regex (case-insensitive). With --loose, a plain phrase whose words
            may be glued together or split oddly (OCR / bad PDFs), e.g.
            --loose "there were no marigolds".
  --pages   only search this page range (skip the contents page and index,
            which otherwise steal the first hit).
Prints "NOT FOUND" for misses: never cite a fact that does not show up here.
"""
import argparse, re

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("patterns", nargs="+")
ap.add_argument("--ctx", type=int, default=400)
ap.add_argument("--max", type=int, default=1)
ap.add_argument("--pages", default=None)
ap.add_argument("--loose", action="store_true")
a = ap.parse_args()

t = open(a.file, encoding="utf-8").read()
parts = re.split(r"\[\[P(\d+)\]\]", t)
pages = [(0, re.sub(r"\s+", " ", parts[0]))]
pages += [(int(parts[i]), re.sub(r"\s+", " ", parts[i + 1])) for i in range(1, len(parts), 2)]
lo, hi = 0, 10**9
if a.pages:
    lo, hi = (int(x) for x in a.pages.split("-"))
for pat in a.patterns:
    rx = r"\s*".join(re.escape(w) for w in pat.split()) if a.loose else pat
    n = 0
    for pg, tx in pages:
        if not lo <= pg <= hi:
            continue
        for m in re.finditer(rx, tx, flags=re.I):
            s = max(0, m.start() - a.ctx // 3)
            print("[%s | p%d] …%s…\n" % (pat[:30], pg, tx[s:m.end() + a.ctx]))
            n += 1
            if n >= a.max:
                break
        if n >= a.max:
            break
    if n == 0:
        print("[%s] NOT FOUND\n" % pat)
