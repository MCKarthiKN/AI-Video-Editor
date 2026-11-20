"""Extract text from a PDF that has a text layer.

Usage (run with the scratch venv that has pypdf):
  <venv>/bin/python -I extract_pdf.py BOOK.pdf OUT.txt

Writes every page as "[[P<n>]]" followed by its text, so the search tools can
report PDF page numbers. If it reports ~0 words, the PDF is a scan: use
ocr_pdf.swift instead.
"""
import sys
from pypdf import PdfReader

src, out = sys.argv[1], sys.argv[2]
reader = PdfReader(src)
total = 0
with open(out, "w", encoding="utf-8") as fh:
    for i, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        total += len(text.split())
        fh.write("\n[[P%d]]\n%s\n" % (i, text))
print(len(reader.pages), "pages", total, "words")
if total < 50 * max(1, len(reader.pages)) // 10:
    print("Very little text: this is probably a scanned PDF. Use ocr_pdf.swift.")
