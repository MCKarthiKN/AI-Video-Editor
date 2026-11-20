"""Build the Reels captions markdown from a caption source file.

Usage:
  python3 -I make_captions.py SOURCE.txt CONFIG.json OUT.md

SOURCE.txt: one record per card, separated by a line with only ===
  NN|Short title|CHAPTER_KEY|PAGES|#tag1 #tag2[|H]
  <caption body: hook, book story, everyday tie-in, the card's punchline>
  Q: <one question for the comments>
  ===
  * NN must run 01, 02, ... in card order (it matches sans-card-NN).
  * CHAPTER_KEY is a key from CONFIG "chapters", or "-" for none.
  * PAGES is free text, e.g. "41" or "86–87" or "92; Ch. 9, p. 203".
  * Add |H for heavy topics: a "talk to a trusted adult or counsellor" line is added.

CONFIG.json example:
  {"title": "Flow", "author": "Mihaly Csikszentmihalyi", "year": 1990,
   "source_note": "Page numbers are the PDF pages of your copy (Flow.pdf).",
   "extra_note": "",
   "page_label": "p.",
   "base_hashtags": "#flow #tanglish #tamilmotivation #booksummary #tamilreels",
   "chapters": {"1": "Ch. 1: Happiness Revisited"}}
Checks numbering and banned words (nee, da, dei, thoonum, panuga) and exits 1 on problems.
"""
import json, re, sys

src, cfg_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(cfg_path, encoding="utf-8"))
chapters = cfg.get("chapters", {})
SAVE = ["Save pannunga. Thevai padumbodhu thirumba paarunga.",
        "Idhu yaarukku thevai-nu thoonudho, avangalukku share pannunga.",
        "Save pannunga. Oru naal idhu ungalukku use aagum.",
        "Ungalukku pidicha oruthar kitta idha anuppunga."]
CARE = "Idhu romba kanamaa irundhaa, thaniya thaangaadheenga. Nambikkaiyaana oru periyavar, illa counsellor kitta pesunga. 🤍"
BANNED = re.compile(r"\b(thoonum|panuga|nee|da|dei)\b", re.I)

recs = [r.strip("\n") for r in re.split(r"^===\s*$", open(src, encoding="utf-8").read(), flags=re.M) if r.strip()]
book = "%s by %s (%s)" % (cfg["title"], cfg["author"], cfg.get("year", "")) if cfg.get("year") else "%s by %s" % (cfg["title"], cfg["author"])
o = ["# %s: Reels captions (all %d cards)" % (cfg["title"], len(recs)), "",
     "Each caption matches `sans-card-NN` / `serif-card-NN` with the same number. Copy everything inside the box.",
     "Every story comes from **%s**. %s" % (book, cfg.get("source_note", "")), ""]
if cfg.get("extra_note"):
    o += [cfg["extra_note"], ""]
o += ["Base hashtags (added to each): `%s`" % cfg["base_hashtags"], ""]
problems = 0
for i, r in enumerate(recs, 1):
    lines = r.split("\n")
    head = [x.strip() for x in lines[0].split("|")]
    n, title, ch, pg, tags = head[:5]
    heavy = len(head) > 5 and head[5] == "H"
    if int(n) != i:
        print("record %d is numbered %s" % (i, n)); problems += 1
    qi = max((j for j, l in enumerate(lines) if l.startswith("Q: ")), default=None)
    if qi is None:
        print("record %s has no Q: line" % n); problems += 1; continue
    body = "\n".join(lines[1:qi]).strip()
    q = lines[qi][3:].strip()
    if BANNED.search(body + " " + q):
        print("record %s has banned words: %s" % (n, BANNED.findall(body + " " + q))); problems += 1
    src_line = "📖 %s, %s" % (cfg["title"], cfg["author"])
    where = chapters.get(ch, "") if ch != "-" else ""
    loc = "%s (%s %s)" % (where, cfg.get("page_label", "p."), pg) if where else "(%s %s)" % (cfg.get("page_label", "p."), pg)
    o += ["---", "", "## %s · %s" % (n, title), "```", body, ""]
    if heavy:
        o += [CARE, ""]
    o += [src_line, loc, "", q + " 👇", SAVE[i % 4], "", tags + " " + cfg["base_hashtags"], "```", ""]
open(out, "w", encoding="utf-8").write("\n".join(o))
ws = [len(r.split("\n", 1)[1].split()) for r in recs]
print(len(recs), "captions, words min/avg/max %d/%d/%d" % (min(ws), sum(ws) // len(ws), max(ws)))
sys.exit(1 if problems else 0)
