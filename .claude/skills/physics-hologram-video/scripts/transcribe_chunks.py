"""Transcribe speech in silence-bounded chunks (<= 24 s) so no words are lost at 30 s window edges.
usage: python3 transcribe_chunks.py <wav16k> <silences.txt> <clipname> <out.json>"""
import sys, os, json, subprocess
import numpy as np
from faster_whisper import WhisperModel
wav, sil, name, out = sys.argv[1:5]
raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', wav, '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'], capture_output=True, check=True).stdout
audio = np.frombuffer(raw, np.float32); dur = len(audio) / 16000
gaps = []
for line in open(sil):
    c, r, _ = line.split(' ', 2)
    if c != name: continue
    a, b = map(float, r.split('-')); gaps.append((a, b))
# speech islands between silences
isl, t = [], 0.0
for a, b in gaps:
    if a - t > 0.25: isl.append([t, a])
    t = b
if dur - t > 0.25: isl.append([t, dur])
# merge islands into chunks <= 24 s
chunks = []
for a, b in isl:
    if chunks and b - chunks[-1][0] <= 24: chunks[-1][1] = b
    else: chunks.append([a, b])
m = WhisperModel(os.environ['WHISPER_MODEL'], device='cpu', compute_type='int8', cpu_threads=4)
res = []
for a, b in chunks:
    a0, b0 = max(0, a - 0.25), min(dur, b + 0.25)
    seg = audio[int(a0 * 16000):int(b0 * 16000)]
    segs, _ = m.transcribe(seg, language='ta', word_timestamps=True, beam_size=5, temperature=0.0, condition_on_previous_text=False)
    for s in segs:
        res.append({'start': s.start + a0, 'end': s.end + a0, 'text': s.text,
                    'words': [{'w': w.word, 's': w.start + a0, 'e': w.end + a0} for w in s.words]})
        print(f'[{s.start + a0:6.1f}-{s.end + a0:6.1f}] {s.text}', flush=True)
json.dump({'segments': res}, open(out, 'w'), ensure_ascii=False, indent=1)
