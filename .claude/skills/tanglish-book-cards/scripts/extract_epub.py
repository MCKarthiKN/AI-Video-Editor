"""Extract text from an unzipped epub, in reading (spine) order.

Usage:
  mkdir <scratch>/book_epub && unzip -q BOOK.epub -d <scratch>/book_epub
  python3 -I extract_epub.py <scratch>/book_epub OUT.txt

Print page numbers: if the epub has page anchors (id="page_12", "pg12",
epub:type="pagebreak" ...), each one becomes a "[[P12]]" marker, so search
results give real book pages. Each spine file also starts with a
"[[FILE name]]" line so you can map chapters.
"""
import html, os, re, sys
from xml.etree import ElementTree as ET

root, out = sys.argv[1], sys.argv[2]
container = ET.parse(os.path.join(root, "META-INF", "container.xml")).getroot()
opf_rel = container.find(".//{*}rootfile").get("full-path")
opf_path = os.path.join(root, opf_rel)
opf_dir = os.path.dirname(opf_path)
opf = ET.parse(opf_path).getroot()
manifest = {it.get("id"): it.get("href") for it in opf.find("{*}manifest")}
spine = [manifest[i.get("idref")] for i in opf.find("{*}spine") if i.get("idref") in manifest]

PAGE = re.compile(r'<[^>]+(?:id|name)="(?:page|pg|p)[_-]?(\d+)"[^>]*>|<[^>]+epub:type="pagebreak"[^>]*(?:title|aria-label)="(\d+)"[^>]*>', re.I)
pages = 0
words = 0
with open(out, "w", encoding="utf-8") as fh:
    for href in spine:
        path = os.path.join(opf_dir, href.split("#")[0])
        if not os.path.exists(path):
            continue
        t = open(path, encoding="utf-8", errors="replace").read()
        t = re.sub(r"(?is)<(script|style|head).*?</\1>", "", t)
        t = PAGE.sub(lambda m: " [[P%s]] " % (m.group(1) or m.group(2)), t)
        t = re.sub(r"(?i)</(p|h\d|div|li|blockquote|tr)>|<br\s*/?>", "\n", t)
        t = re.sub(r"<[^>]+>", "", t)
        t = html.unescape(t)
        t = re.sub(r"[ \t\xa0]+", " ", t)
        t = re.sub(r"\n\s*\n+", "\n", t).strip()
        pages += len(re.findall(r"\[\[P\d+\]\]", t))
        words += len(t.split())
        fh.write("\n[[FILE %s]]\n%s\n" % (os.path.basename(href), t))
print(len(spine), "spine files,", words, "words,", pages, "page markers")
if pages == 0:
    print("No page anchors found: cite chapters instead of page numbers in captions.")
