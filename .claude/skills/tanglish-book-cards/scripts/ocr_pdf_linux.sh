#!/usr/bin/env bash
# OCR a scanned PDF on Linux (cloud sessions), the counterpart of ocr_pdf.swift.
# Needs: pdftoppm (poppler, preinstalled) and tesseract:
#   apt-get install -y tesseract-ocr
# Usage:  bash ocr_pdf_linux.sh BOOK.pdf OUT.txt [firstPage] [lastPage]
# Output uses the same "[[P<n>]]" page markers as extract_pdf.py.
# Runs pages in parallel (one per CPU); run long books in the background.
set -euo pipefail
pdf=$1; out=$2
total=$(pdfinfo "$pdf" 2>/dev/null | awk '/^Pages:/{print $2}')
first=${3:-1}; last=${4:-$total}
[ "$last" -gt "$total" ] && last=$total
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
export pdf tmp
ocr_page() {
  local i=$1
  pdftoppm -r 300 -gray -f "$i" -l "$i" -png "$pdf" "$tmp/pg$i" >/dev/null 2>&1
  local img; img=$(ls "$tmp"/pg"$i"*.png 2>/dev/null | head -1)
  if [ -n "$img" ]; then
    tesseract "$img" "$tmp/t$i" -l eng --psm 3 >/dev/null 2>&1 || : > "$tmp/t$i.txt"
    rm -f "$img"
  else
    : > "$tmp/t$i.txt"
  fi
  [ $((i % 20)) -eq 0 ] && echo "page $i" >&2 || true
}
export -f ocr_page
seq "$first" "$last" | xargs -P "$(nproc)" -I{} bash -c 'ocr_page {}'
: > "$out"
for i in $(seq "$first" "$last"); do
  printf '\n[[P%d]]\n' "$i" >> "$out"
  cat "$tmp/t$i.txt" >> "$out"
done
echo "done $first $last of $total, $(wc -w < "$out") words"
