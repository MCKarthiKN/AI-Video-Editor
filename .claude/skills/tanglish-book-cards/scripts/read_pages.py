"""Print pages FIRST..LAST of an extracted book text, each cut to MAXCHARS.

Usage: python3 -I read_pages.py BOOK.txt FIRST LAST [MAXCHARS]
Use it to skim a chapter quickly (e.g. 1200 chars per page) before writing cards.
"""
import re, sys
t = open(sys.argv[1], encoding="utf-8").read()
a, b = int(sys.argv[2]), int(sys.argv[3])
mx = int(sys.argv[4]) if len(sys.argv) > 4 else 10**6
parts = re.split(r"\[\[P(\d+)\]\]", t)
for i in range(1, len(parts), 2):
    pg = int(parts[i])
    if a <= pg <= b:
        tx = re.sub(r"-\s+(?=[a-z])", "", re.sub(r"\s+", " ", parts[i + 1])).strip()
        print("[p%d] %s" % (pg, tx[:mx]))
