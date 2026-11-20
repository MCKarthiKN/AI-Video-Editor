---
name: tanglish-book-cards
description: Turn a book (PDF, scanned PDF, epub, audiobook transcript) into MCkarthiKN's Tanglish Dan Koe-style quote cards in his own voice, render them as black 1080x1350 Instagram cards in sans and serif with 2x2 collages, zip them, and write Reels captions with page references. Use this whenever the user drops a book file or transcript and says anything like "create cards for this book", "do this for this book too", "create as many quotes as possible", "scratch through this book", "make captions", "English version of the cards", or asks to plan Instagram posts from his quote cards, even if he does not say "skill" or "cards".
---

# Tanglish book cards (MCkarthiKN)

The user is an electronics/physics teacher in Madurai who grows an Instagram
audience with quote cards from books he reads, written in his own Tanglish
voice. For each book he sends, the full deliverable is:

1. `QUOTE CARDS/cards/<book>.txt` (cards separated by `---`, one `*highlight*` each)
2. Rendered PNGs in `QUOTE CARDS/output/<book>/sans/` and `.../serif/`
   (`sans-card-NN.png`, `sans-collage-NN.png`, same for serif) plus `cards.txt`
3. `QUOTE CARDS/cards/<book>-captions.md`, copied to `output/<book>/captions.md`
4. `QUOTE CARDS/zips/<book>.zip`, and the book folder added to `zips/all-quote-cards.zip`

He has asked for captions after every book, so do all four in one go unless
he says otherwise.

**Paths.** The project folder is usually
`~/Desktop/Oppourtunities/AI VIDEO EDITOR/QUOTE CARDS` (note the spelling
"Oppourtunities"). The card renderer is `QUOTE CARDS/index.html`.
This skill's helpers are in `~/.claude/skills/tanglish-book-cards/`
(`scripts/`, `assets/`, `references/`); below, `$SK` means that folder.
Work in the session's scratchpad (`$S`) for everything temporary.

Before writing, read `references/voice.md` (his voice and hard rules), and
skim `references/lessons.md` (gotchas from earlier books). Read
`references/captions.md` before writing captions.

## Cloud session (claude.ai/code, Linux) differences

The steps below were written for the Mac desktop app. In a cloud session
(this repo checked out in a Linux container) change only these things:

- **Paths.** `SK="$(git rev-parse --show-toplevel)/.claude/skills/tanglish-book-cards"`,
  `Q="$(git rev-parse --show-toplevel)/work/QUOTE CARDS"` (`work/` is gitignored and
  the user can open files there from the Claude app), then
  `mkdir -p "$Q/cards" "$Q/output" "$Q/zips"`. `$S` is the scratchpad.
- **Card renderer.** `QUOTE CARDS/index.html` lives on the user's Mac. Use
  `$SK/assets/index.html` if it exists, otherwise ask the user to upload
  `index.html` (and `MY_VOICE.md` if they have one) and copy them into `$Q/`.
  Do not render with a look-alike page.
- **Getting the book.** A chat upload lands under `/root/.claude/uploads/...`;
  a Google Drive share link downloads with
  `curl -L -o "$S/in/BOOK.pdf" "https://drive.usercontent.google.com/download?id=<ID>&export=download&confirm=t"`.
  Copy it into its own new folder under `$S` before extracting.
- **Scanned PDF.** There is no Swift or Vision here. Use tesseract instead:
  `apt-get install -y tesseract-ocr`, test with
  `bash $SK/scripts/ocr_pdf_linux.sh BOOK.pdf $S/book.txt 1 12`, then run all pages
  in the background (`run_in_background`). Output has the same `[[P<n>]]` markers.
- **Render (Step 4).** There is no built-in browser. After setting up `$S/site`
  (same `cp` lines as Step 4), run
  `node $SK/scripts/render_cloud.js $S/site $S/out_<slug>` instead of starting
  `upload_server.py` yourself. It starts and stops the server, runs
  `render_job.js` in headless Chromium (Playwright is preinstalled), prints
  the `window.__p` JSON and exits 0 when `done` = 2 × (cards + collages).
  Skip the `pkill` line.
- **Zips (Step 6).** Make `$Q/zips/<slug>.zip` as usual. `all-quote-cards.zip`
  lives on the Mac, so skip it unless the user uploads it. Send the book zip
  and `captions.md` with SendUserFile: the container is temporary, so
  nothing in `work/` survives the session.
- **Never commit** books, extracted text, cards, PNGs, captions or zips. The repo
  is public and the books are copyrighted.
- The zsh notes in `lessons.md` don't apply; the shell here is bash.

## Step 1: Get the text, with page numbers

Put every input in its own scratch folder and run Python with `-I`.

| Input | How |
|---|---|
| PDF with text | `python3 -m venv $S/venv && $S/venv/bin/pip install -q pypdf`, then `$S/venv/bin/python -I $SK/scripts/extract_pdf.py BOOK.pdf $S/book.txt` |
| Scanned PDF (0 words above) | `cd $S && swiftc -O $SK/scripts/ocr_pdf.swift -o $S/ocr_pdf`, test on pages 1–12, then run all pages in the background: `$S/ocr_pdf BOOK.pdf $S/book.txt` |
| Epub | `mkdir $S/book_epub && unzip -q BOOK.epub -d $S/book_epub`, then `python3 -I $SK/scripts/extract_epub.py $S/book_epub $S/book.txt` |
| Pasted transcript | Save it to `$S/book.txt`; convert any Tamil script to Tanglish |

Then map the structure: find the contents page and chapter starts with
`find_text.py --pages`, and note the page range of each chapter (you need it
for captions).

## Step 2: Read the book properly

Cards are only as good as the reading. Skim each chapter with
`python3 -I $SK/scripts/read_pages.py $S/book.txt FIRST LAST 1200`, and pull
exact stories, names and numbers with
`python3 -I $SK/scripts/find_text.py $S/book.txt --pages 20-300 "regex" ...`
(add `--loose` for OCR text with glued words). Keep a short list of
"card idea → page". Anything that comes back NOT FOUND does not go on a card.

## Step 3: Write the cards

- How many: if he gives a number, use it. Otherwise 44–96, always a multiple of
  4 so the 2×2 collages come out even (see lessons.md).
- Follow the book's order, one idea per card, in his voice
  (`references/voice.md`). Rework ideas with everyday Tamil examples and
  electronics metaphors; never copy the book's sentences.
- Leave out graphic or harmful details, slurs and party politics; his students
  follow the page (see lessons.md, "Sensitive books").
- Write to `QUOTE CARDS/cards/<book-slug>.txt` with a heredoc, then run
  `python3 -I $SK/scripts/validate_cards.py "<path>"` and fix every flagged card.

## Step 4: Render

```bash
mkdir -p $S/site $S/out_<slug>; find $S/out_<slug> -name '*.png' -delete
cp "$Q/index.html" $S/site/index.html && cp "$Q/cards/<slug>.txt" $S/site/cards.txt
cp $SK/assets/render_job.js $S/site/render_job.js
cd $SK/scripts && exec python3 upload_server.py $S/out_<slug> $S/site     # run_in_background
```
Then in the built-in browser: navigate to `http://127.0.0.1:8766/index.html`
and run this with the javascript tool (it loads and starts the render job):
```js
(0, eval)(await fetch('render_job.js?' + Date.now()).then(r => r.text())); 'launched'
```
Wait ~8 seconds, then check `JSON.stringify(window.__p)`: `done` should equal
2 × (cards + collages), `err` null, `stage` "finished". (The tool's own
settings are applied by the script: Poppins "geo" = sans, EB Garamond = serif,
black background, white highlight, signature "MCkarthiKN".)

Copy into the project and check:
```bash
D="$Q/output/<slug>"; mkdir -p "$D/sans" "$D/serif"
cp $S/out_<slug>/sans-*.png "$D/sans/" && cp $S/out_<slug>/serif-*.png "$D/serif/" && cp "$Q/cards/<slug>.txt" "$D/cards.txt"
python3 -I $SK/scripts/check_images.py "$D"
pkill -f "upload_server.py"
```
Look at one sans card and one serif collage with the Read tool before moving on.
If you fix a card later, re-render the whole set (it takes seconds) so the
collages match.

## Step 5: Captions

Follow `references/captions.md`. Write a small config JSON (see
`$SK/assets/captions_config_example.json`) with the title, author, year,
page note, hashtags and chapter names, write the caption source in batches,
then:
```bash
python3 -I $SK/scripts/make_captions.py $S/<slug>_caps.txt $S/<slug>_caps.json "$Q/cards/<slug>-captions.md"
cp "$Q/cards/<slug>-captions.md" "$Q/output/<slug>/captions.md"
```

## Step 6: Zips

```bash
cd "$Q/output" && rm -f ../zips/<slug>.zip && zip -qr ../zips/<slug>.zip <slug> -x '*.DS_Store'
cd "$Q/output" && zip -qr ../zips/all-quote-cards.zip <slug> -x '*.DS_Store'   # run_in_background (large)
```
If captions are added after the first zip: `zip -q ../zips/<slug>.zip <slug>/captions.md` (same for the all-books zip).

## Step 7: Report back

Tell him, briefly: how many cards, collages and captions; anything you left out
on purpose and why; which captions contain opinion lines in his voice (so he can
check them); any of your own examples; any numbers to double-check (OCR); and
the file paths. Suggest which few cards are safest or strongest to post first.

## Extras he has asked for before (only when asked)
- **English versions** of selected cards: write `cards/<name>-english.txt`,
  render with the same job, and for carousels add the "English version →" line
  with `$SK/assets/swipe_line.js` (Tanglish first, English second).
- **Posting plan**: one card a day with a story sticker and a story-style
  question; see lessons.md for what has worked.
- **Line-art concept visuals**: `QUOTE CARDS/visuals.html`; no AI photo prompts.
