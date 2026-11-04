#!/usr/bin/env bash
# Full pipeline for a talking-head clip: track -> Robust Video Matting -> render (4 parallel chunks) -> mix -> final mp4.
# usage: ./make_video.sh input.mp4 work_dir
# Effect timings in fx/render_v2.py and fx/audio_v2.py are set per video: edit them to match the new script and gestures.
set -euo pipefail
cd "$(dirname "$0")"
IN=$1; WD=$2; mkdir -p "$WD/out"
python3 fx/track.py "$IN" "$WD"
python3 - "$WD" <<'PY'
import json, sys, numpy as np
D = sys.argv[1]; fr = json.load(open(f'{D}/track.json'))['frames']
def palms(f):
    p = f['pose']; hs = [np.mean(np.array(h['pts'])[[0, 5, 9, 13, 17]], 0) for h in f['hands']]
    wr = {'A': np.array(p['r_wrist'][:2]), 'B': np.array(p['l_wrist'][:2])}; out = {}; used = set()
    for k in ('A', 'B'):
        best = None
        for i, h in enumerate(hs):
            if i in used: continue
            d = np.linalg.norm(h - wr[k])
            if d < 120 and (best is None or d < best[0]): best = (d, i)
        if best: used.add(best[1]); out[k] = hs[best[1]]
        else: out[k] = wr[k]
    return out
np.save(f'{D}/palms.npy', np.array([[palms(f)['A'], palms(f)['B']] for f in fr]))
PY
N=$(ls "$WD/frames" | wc -l)
python3 fx/matte_rvm.py "$WD" 0 "$N" 0
Q=$(( (N + 3) / 4 ))
for i in 0 1 2 3; do a=$((i*Q)); b=$(( (i+1)*Q < N ? (i+1)*Q : N )); MATTE=rvm python3 fx/render_v2.py "$WD" --range $a $b "$WD/out/p_$i.mp4" & done; wait
printf "file 'p_0.mp4'\nfile 'p_1.mp4'\nfile 'p_2.mp4'\nfile 'p_3.mp4'\n" > "$WD/out/parts.txt"
ffmpeg -v error -y -f concat -safe 0 -i "$WD/out/parts.txt" -c copy "$WD/out/video_only.mp4"
ffmpeg -v error -y -i "$IN" -vn -ac 1 -ar 48000 "$WD/out/voice.wav"
python3 fx/audio_v2.py "$WD/out/voice.wav" "$WD/out/mix_raw.wav"
ffmpeg -v error -y -i "$WD/out/mix_raw.wav" -af "loudnorm=I=-14:TP=-1.5:LRA=11" -ar 48000 "$WD/out/mix.wav"
ffmpeg -v error -y -i "$WD/out/video_only.mp4" -i "$WD/out/mix.wav" -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart "$WD/out/final.mp4"
echo "done: $WD/out/final.mp4"
