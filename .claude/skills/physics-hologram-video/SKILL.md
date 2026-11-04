---
name: physics-hologram-video
description: Edit a Tamil physics-explainer clip into a Brian-Greene-style "presenter in a dark void with interactive glowing holograms" video - Robust Video Matting cutout, hand-tracked hologram assets (tree, apple, clock, knob...), physics animations, virtual camera with motion blur, subtle wide sound design. Use whenever the user (MCKarthiKN) uploads/links a video to edit, or asks for "the same kind of output".
---

# Physics hologram video (MCKarthiKN's house style)

Reference look: Brian Greene, "What is String Theory?" (World Science Festival). Presenter alone in a dark
void, face lit, glowing objects that live in the presenter's hands, horizontal lens streaks, push-ins on
"magnify" moments, and a soundtrack that is *only* there when a graphic is (no music bed).

Finished example of this skill: 19 s apple/time video - "pluck an apple from a hologram tree, turn a knob to
slow time, drop it (slow free fall with a dotted distance scale), stop time, ask: what is its velocity?"

## The user's standing preferences (do not re-ask)
- **Always cut the speaker out with Robust Video Matting** (ResNet-50 ONNX), never the MediaPipe mask.
- **No subtitles** unless asked. Text on screen only when it is part of the physics (d1, d2..., `v = ?`).
- **Sound: subtle, wide, event-driven** (see Sound). No loud chimes/hits, no constant drone.
- **Always storyboard first** and wait for approval before the long final render. Show cheap tests
  (stills / short clips) for any new effect before committing.
- Speaks **Tamil (Tanglish)**. Read their words; never invent what they said - confirm unclear words.
- Prefers short, plain-language replies; give exact text to paste when settings are needed.
- They supply their own AI-generated wireframe assets; use them, and keep them in the hologram style.

## Environment (fresh cloud session)
1. Allowed domains needed (tell the user the exact lines to paste if missing):
   `huggingface.co`, `*.huggingface.co`, `*.hf.co`, `drive.google.com`, `drive.usercontent.google.com`,
   `*.googleusercontent.com`. (`storage.googleapis.com`, GitHub release downloads, PyPI and npm work by default.)
2. `bash .claude/skills/physics-hologram-video/scripts/setup_env.sh` then `export FX_MODELS=~/fx_models`.
3. Chromium + Playwright for Node are preinstalled (`require('playwright')`) - used for storyboards / SVG icons.
4. YouTube downloads are blocked by bot checks from cloud IPs - do not try to bypass; ask for an upload.

Getting the user's video: chat upload (lands in `/root/.claude/uploads/...`) or Google Drive share link:
`curl -L -o input.mov "https://drive.usercontent.google.com/download?id=<ID>&export=download&confirm=t"`.
Their own computer is not reachable unless Remote Control folders show up in `list_computer_folders`.

## Workflow
Work dir `W` (scratchpad). Scripts live in `S=.claude/skills/physics-hologram-video/scripts`.

1. **Inspect**: `ffprobe` (iPhone 4K HEVC 50 fps is typical), frame contact sheet, `silencedetect` for pauses.
2. **Transcribe Tamil with word timings** (Tamil fine-tuned Whisper; large-v3 produces garbage on this speaker):
   `WHISPER_MODEL=$(ls -d $FX_MODELS/hf/hub/models--mohankumars26--whisper-tamil-medium-ct2/snapshots/*) python3 $S/transcribe.py voice.wav transcript.json ta`
   If it truncates near the end, transcribe the tail separately (`ffmpeg -ss`). Show the user the Tamil +
   English meaning and ask them to confirm words that look wrong (e.g. "டைம் ___ பண்ணிட்டு" was "ஸ்லோ").
3. **Working copy**: 1080p25 for tracking/matting, plus 2880x1620 frames from the 4K original for the final
   (zoom-ins stay sharp): `ffmpeg -i input.mov -vf "scale=1920:1080,fps=25" work1080.mp4` and
   `ffmpeg -i input.mov -vf "fps=25,scale=2880:1620:flags=lanczos" -q:v 2 frames_hi/%05d.jpg`.
4. **Track + matte** (background): `python3 $S/track.py work1080.mp4 $W` -> frames/, masks/, track.json;
   `python3 $S/palms.py $W` -> palms.npy (+ printed hand path every 0.4 s);
   `python3 $S/matte_rvm.py $W 0 <nframes> 0` -> masks_rvm/, fgr_rvm/ (RVM is recurrent: always run from frame 0).
5. **Gesture <-> word map**: read the printed palm path next to the word timings. Place effects ON gestures
   (raised hand = pluck point, free hand dropping = knob, hand tracing down = fall path, pointing = freeze spot).
6. **Assets**: `cut_assets.py alpha` for PNG/WebP with transparency (check `IMREAD_UNCHANGED` first - an image
   that *looks* white may already be transparent); `cut_assets.py black` for glow-on-black art (+crop).
   Send the user an asset sheet + one preview frame with everything placed.
7. **Tests**: short clips for new behaviour (`tree_scene.py` = interactive tree test, `holo_test.py`-style).
8. **Storyboard**: `sb_mock2.py` builds 8 mock panels from the plate; fill `templates/storyboard_final.html`
   (cards: time, Tamil line, meaning, visual, sound) and screenshot it with `node templates/shot.js in.html out.png`.
   Wait for "approved".
9. **Final**: edit the timing/camera block at the top of `scene_final.py` for the new video, check stills
   (`--stills t1,t2,...`), then render in 4 parallel chunks
   (`python3 $S/scene_final.py $W out/p_<a>.mp4 <a> <b>`), concat, mix (`score_final.py`), `loudnorm I=-14`.
   Fix single-moment glitches by re-rendering a few frames and splicing (trim/concat filter), not the whole video.

## Visual rules (what made it look premium)
- Dark stage: background -> navy-black, key light centred on the face (`KEY` falloff), slight cool grade,
  lights-out wipe from the edges inward over the first ~0.6 s.
- Holograms are light: additive/screen blend, scanlines, rising scan band, tiny flicker, cyan edge glow,
  projector build-up from the base with a scan line. Flat images get a pseudo-depth turn (distance-transform
  depth + small yaw) - it reads as 3D. True 3D needs `.glb` models (Meshy/Tripo3D from their images).
- **Immersion**: split the tree into back (behind speaker) and front leaf clusters (in front of the arm,
  never over the face); hand-driven spring field (velocity of each palm pushes nearby leaves, damped spring
  brings them back); pluck = inpaint that exact fruit out of the tree + branch spring-back impulse.
- Living layers everywhere: hologram floor (rings + dotted spokes), fireflies, slowly falling leaves.
  All of them run on *simulated time* (slowed by the knob, frozen at time-stop).
- Physics honesty: slow-motion free fall = positions follow y = y0 + 1/2 g tau^2 in slowed time; ghost copies at
  equal tau steps (growing gaps). Ghosts must differ from scene objects (white-cyan wireframe, not red apples).
- Measurement: thin dotted vertical scale, dashed horizontal lines per ghost, small glowing end dots, small
  amber `d1, d2...` labels (DejaVu via PIL). Dim the background behind it ("focus lane").
- Time stop: short white ripple from the object, desaturate graphics, stop all motion; only the speaker moves.
- Camera: virtual camera on the 2880x1620 composite -> 1920x1080. Slow push-ins on build-ups, punch-ins on
  actions (pluck), swing to props (knob), fast pull-out on release, snap zoom on the freeze; motion blur by
  averaging 7 camera samples over half a frame, only while the camera moves. Keep scales/labels shots blur-free.

## Sound rules (`sfx_subtle.py`, `score_final.py`)
Learned from the reference by cancelling the centred voice (L-R): sound exists only while graphics are on screen,
glassy 1.7 kHz tone with ~11 Hz flutter, airy 3-12 kHz shimmer, low swells, everything **wide stereo** and
~20 dB under the voice. Palette: `shimmer`, `swish`, `vibration`, `low_swell`, `sparkle`, `bloom`, `reverb`.
Drive sounds from the data: leaf rustle amplitude = hand speed inside the canopy; stopwatch ticks spaced by the
simulated-time rate; hush (cut highs) after time-stop with one sustained frozen tone. Voice normalised, effects
dip slightly under speech, final `loudnorm=I=-14:TP=-1.5`.

## Pitfalls already hit (avoid)
- OpenCV 5 `putText` needs uint8 images; draw text into a uint8 mask, then colour it.
- `faster-whisper` + new PyAV: pass a numpy float32 array (decode with ffmpeg) instead of a path.
- mediapipe >= 1.x has no `mp.solutions`; use the Tasks API with downloaded `.task` models; needs `libegl1`.
- Never `pgrep -f` / `pkill -f` a pattern that appears in your own command line (it matches the shell itself).
- `np.convolve(..., 'same')` with long windows on audio is O(N*M) - use cumulative-sum moving averages.
- Keep zoom transforms mapping centre -> centre (`M = [[z,0,c-z*c],...]`), or the frame slides.
- Flashes/ripples: short (tau ~0.08 s) and small amplitude; a 0.3 s full-frame lift turns the frame grey.
- Effects in front of the face are never OK (orb radius limited to below the chin, front leaves masked off the face).
- Repo is public: commit code only, never the user's footage or images.

## Templates
`templates/render_v3.py`, `score_v3.py`, `subs_v3.py`: the earlier vertical talking-head version (icons from
@tabler/icons via npm rasterised with `scripts/raster_icons.js`, Tamil key-word pop-ups / subtitles with libass +
Noto Sans Tamil). Use when the user asks for subtitles or a portrait (Shorts/Reels) talking-head edit.
