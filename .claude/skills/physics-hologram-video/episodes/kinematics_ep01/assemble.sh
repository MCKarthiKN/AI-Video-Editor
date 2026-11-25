#!/bin/bash
# Join the beat renders in timeline order, add the mixed sound, loudness-normalise to -14 LUFS.
set -e
cd "$(dirname "$0")"
LIST=(out/opening.mp4 out/guess.mp4 r3d/train.mp4 out/toy.mp4 out/idea.mp4 out/frame.mp4 out/book.mp4 out/position.mp4 out/origins.mp4 out/ending_a.mp4 out/ending_b.mp4)
args=(); filt=""
for i in "${!LIST[@]}"; do args+=(-i "${LIST[$i]}"); filt+="[$i:v]fps=25,format=yuv420p,setsar=1[v$i];"; done
for i in "${!LIST[@]}"; do filt+="[v$i]"; done
filt+="concat=n=${#LIST[@]}:v=1:a=0[v]"
ffmpeg -v error -y "${args[@]}" -i mix.wav -filter_complex "$filt" -map "[v]" -map ${#LIST[@]}:a \
  -af "loudnorm=I=-14:TP=-1.5:LRA=11" -c:v libx264 -preset medium -crf 16 -pix_fmt yuv420p -r 25 \
  -c:a aac -b:a 256k -ar 48000 -shortest -movflags +faststart E01_X_final.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 E01_X_final.mp4
