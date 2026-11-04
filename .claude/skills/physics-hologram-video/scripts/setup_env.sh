#!/usr/bin/env bash
# One-time setup for a fresh cloud session (~3-5 min; downloads ~1.3 GB of models).
# Needs these in the environment's Allowed domains (besides the defaults):
#   huggingface.co  *.huggingface.co  *.hf.co          (Tamil Whisper model)
#   drive.google.com  drive.usercontent.google.com  *.googleusercontent.com   (videos shared by Drive link)
set -euo pipefail
M=${FX_MODELS:-$HOME/fx_models}; mkdir -p "$M"
pip install -q mediapipe opencv-python-headless onnxruntime scipy pillow faster-whisper huggingface_hub
(apt-get install -y -q libegl1 libgles2 libgl1 >/dev/null 2>&1 || (apt-get update -q >/dev/null && apt-get install -y -q libegl1 libgles2 libgl1 >/dev/null)) || true
get() { [ -s "$M/$2" ] || curl -sSL -o "$M/$2" "$1"; echo "ok $2"; }
MP=https://storage.googleapis.com/mediapipe-models
get $MP/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task hand_landmarker.task
get $MP/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task pose_landmarker_heavy.task
get https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx rvm_resnet50_fp32.onnx
# Tamil speech-to-text (standard large-v3 hallucinates on Tamil/Tanglish; this Tamil fine-tune is accurate)
HF_HOME="$M/hf" python3 -c "from huggingface_hub import snapshot_download as s; print('ok whisper-tamil', s('mohankumars26/whisper-tamil-medium-ct2'))"
# Tamil font (only needed for Tamil text on screen / storyboards)
mkdir -p "$M/fonts" && curl -sSL https://registry.npmjs.org/@expo-google-fonts/noto-sans-tamil/-/noto-sans-tamil-0.4.3.tgz | tar -xz -C "$M/fonts" && echo "ok fonts"
echo "export FX_MODELS=$M   # set this before running the scripts"
