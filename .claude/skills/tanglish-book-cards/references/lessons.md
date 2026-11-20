# Lessons and gotchas from past books

## Environment
- Python run with the Desktop folder as the working directory fails in this
  sandbox. Run scripts with `cd <scratchpad>` (or use absolute paths) and
  `python3 -I` for anything that reads untrusted files.
- Shell cwd drifts between calls: always use absolute paths.
- zsh: an unmatched glob like `rm -f dir/*.png` errors with "no matches found".
  Use `find dir -name '*.png' -delete`.
- Inside heredocs, a line of `===` is fine, but `echo ===` on its own fails in
  zsh; use `printf '===\n'`.
- The scratchpad can be wiped between turns: recreate the venv, server and
  site copy whenever they are missing.
- pypdf: `python3 -m venv <scratch>/venv && <scratch>/venv/bin/pip install -q pypdf`.
- Scanned PDFs (0 words from pypdf): build `ocr_pdf.swift` with `swiftc -O`
  and run it in the background (~240 pages took a few minutes). OCR can
  misread digits: double-check every number you put on a card.
- Bad PDFs glue words together ("Quietas it's kept"): use `find_text.py --loose`.
- The contents page matches every chapter keyword first: use `--pages` to
  search only the body.
- The upload server runs as a background task; when you kill it with pkill, the
  task reports "failed (exit 144)". That is expected, not an error.
- Adding a book to `all-quote-cards.zip` (100+ MB) can take over 2 minutes:
  run that zip in the background.

## Sizes that worked
- "Create cards" / "as many as possible": 44–96 cards, always a multiple of 4
  (collages are 2×2). Typical: 64–72 for a self-help book, 44 for a novel or
  history book.
- Rendering 180 PNGs takes ~10 seconds in the browser.

## Sensitive books
- Novels and history (The Bluest Eye, An Era of Darkness) need care: leave out
  child abuse, graphic violence, slurs, party politics and communal framing.
  Present "divide and rule" as a pattern, not aimed at a community.
- Note when a book is dated (e.g. laws that changed after publication).
- The user's **students follow the page**. Keep cards and captions safe for
  them; avoid romance-heavy cards in the main plan; never invent classroom
  stories; heavy cards get the counsellor line.

## The user's audience goals (Instagram)
- Goal is a real, engaged audience, not follower count.
- What worked: practical tools/formulas (saves), "this is me" mirror lines
  (likes), lines people have heard ("Yen appove sollala?") for shares, and
  posts in his own voice (comments).
- Plan used: one card a day at a fixed time (7:30 pm), share to story with a
  sticker, reply to every real comment with a question back, a names list,
  one own-voice video every few days, review on Day 7 and Day 14.
- Optional English versions: translate the card text, render in the same
  style, and pair with the Tanglish card in a carousel using
  `assets/swipe_line.js` ("English version →").
