# Reels captions

The user has asked for captions after every book, so make them in the same
turn as the cards. One caption per card, numbered to match `sans-card-NN`.

## Shape of one caption (≈60–130 words)
1. Hook: a line people say or feel, or a striking number.
2. The book's story or fact behind the card, in his words, with names and
   numbers checked against the text (find_text.py).
3. An everyday tie-in (Tamil life, school, office, phone).
4. The card's punchline, repeated as the last two lines.
5. `📖 Title, Author` + chapter + page (added by make_captions.py).
6. A question for the comments. Prefer questions people answer with a short
   story ("…eppo nadandhuchu?") over yes/no, to start real conversations.
7. A save/share line and hashtags (added by make_captions.py).

Mark your own examples honestly inside the caption, e.g.
"(Cricket example ennoda. Karuthu book-oda.)"

## Source file format (for make_captions.py)
```
07|Kai solra anbu|-|7–8|#love #parents
Claudia-ku irumal, kaaichal. ...
...
Sila anbu kurai sollum kural-la varum.
Kai-la dhaan theriyum. 🤍
Q: Thittitte anba kaattuna oruthar unga life-la yaaru?
===
```
Header fields: number | short title | chapter key from config ("-" for none) |
pages | 2 topic hashtags | optional `H` for heavy topics (adds the
"talk to a trusted adult or counsellor" line).

Write the source in batches of ~12 with heredocs (`cat >> file <<'EOF'`), then
run make_captions.py once at the end.

## Page numbers
- Text PDF or OCR'd PDF: PDF page numbers ("PDF p." label), and say so in
  `source_note`. If the printed number differs by a fixed offset, say that too
  (e.g. "add 9 to find the PDF page").
- Epub with page anchors: print page numbers ("p." label).
- No page info: chapter only.

## Care
Use `H` on captions about childhood harm, depression, suicide-adjacent
feelings, grief and similar. Leave graphic details out entirely.
