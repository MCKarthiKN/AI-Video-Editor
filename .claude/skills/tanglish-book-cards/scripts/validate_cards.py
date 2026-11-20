"""Check a cards file before rendering.

Usage: python3 -I validate_cards.py cards/<book>.txt

Checks each card (separated by a line containing only ---):
  * 18–62 words, exactly one *highlight*, and the highlight is the last thing
  * none of the banned spellings/words: thoonum, panuga, nee, da, dei
  * no em dash, no Tamil script or other stray non-ASCII (OCR/typing slips)
  * the "vandhu" filler at most once
and reports the total, which should be a multiple of 4 so collages come out even.
Exit code 1 if anything needs fixing.
"""
import re, sys
t = open(sys.argv[1], encoding="utf-8").read()
cards = [c.strip() for c in re.split(r"^\s*---\s*$", t, flags=re.M) if c.strip()]
BANNED = re.compile(r"\b(thoonum|panuga|nee|da|dei)\b", re.I)
ALLOWED = set("–→…₹")
bad = 0
for i, c in enumerate(cards, 1):
    problems = []
    w = len(c.split())
    if not 18 <= w <= 62:
        problems.append("%d words" % w)
    hl = re.findall(r"\*[^*\n]+\*", c)
    if len(hl) != 1:
        problems.append("%d highlights" % len(hl))
    elif not c.rstrip().endswith("*"):
        problems.append("highlight is not the last line")
    if BANNED.search(c):
        problems.append("banned: " + ",".join(sorted(set(m.lower() for m in BANNED.findall(c)))))
    if "—" in c:
        problems.append("em dash")
    odd = sorted(set(ch for ch in c if ord(ch) > 127 and ch not in ALLOWED))
    if odd:
        problems.append("non-ASCII: " + "".join(odd))
    if len(re.findall(r"\bvandhu\b", c, flags=re.I)) > 1:
        problems.append("vandhu x%d" % len(re.findall(r"\bvandhu\b", c, flags=re.I)))
    if problems:
        bad += 1
        print("card %d: %s | %s" % (i, "; ".join(problems), c.splitlines()[0][:60]))
ws = [len(c.split()) for c in cards]
print("%d cards, words min/avg/max %d/%d/%d" % (len(cards), min(ws), sum(ws) // len(ws), max(ws)))
if len(cards) % 4:
    print("Note: %d is not a multiple of 4, so the last collage will be partly empty." % len(cards))
sys.exit(1 if bad else 0)
