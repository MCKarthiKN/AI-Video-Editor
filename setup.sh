#!/usr/bin/env bash
# One-time setup: Python packages, system libraries and AI models.
set -euo pipefail
cd "$(dirname "$0")"
pip install -q -r requirements.txt
if command -v apt-get >/dev/null; then
  (apt-get install -y -q libegl1 libgles2 libgl1 ffmpeg >/dev/null 2>&1 || (apt-get update -q >/dev/null && apt-get install -y -q libegl1 libgles2 libgl1 ffmpeg >/dev/null)) || echo "apt install skipped"
fi
mkdir -p models
get() { [ -s "models/$2" ] || curl -sSL -o "models/$2" "$1"; echo "ok  models/$2"; }
MP=https://storage.googleapis.com/mediapipe-models
get $MP/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task hand_landmarker.task
get $MP/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task pose_landmarker_heavy.task
# Robust Video Matting (default cutout for every edit)
get https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx rvm_resnet50_fp32.onnx
echo "setup complete"
