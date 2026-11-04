# physics-video-fx

Glowing "physics explainer" effects for talking-head videos, in the style of presenter-in-a-dark-void science videos:
the room goes dark, the speaker is cut out and lit from the face, and glowing graphics (equations, hand flares,
light streaks, a dotted "idea orb", atoms, a hologram) follow the speaker's tracked hands, with a matching sound design.

## Setup (once per machine)
```bash
./setup.sh
```
Installs Python packages and downloads three models into `models/`:
- **Robust Video Matting** (ResNet-50, ONNX): the default cutout for every edit (soft hair edges, steady across frames)
- MediaPipe pose landmarker: face, shoulders, wrists
- MediaPipe hand landmarker: palm and finger positions

## Make a video
```bash
./make_video.sh my_clip.mp4 work/my_clip
```
Steps: track hands and body, cut the speaker out with Robust Video Matting, render the effects in 4 parallel chunks,
mix the voice with the sound effects, and write `work/my_clip/out/final.mp4`.

The effect timeline is written per video. Before running on a new clip, edit the timings in
`fx/render_v2.py` (equations, shatter, hand flares, orb, hologram) and `fx/audio_v2.py` (sound cues)
to match the new script and gestures.

## Files
| File | What it does |
|---|---|
| `fx/track.py` | Extracts frames; tracks pose, hands and a body mask on every frame |
| `fx/matte_rvm.py` | Robust Video Matting: alpha matte + clean foreground colour per frame |
| `fx/render_v2.py` | Dark stage, face key light, all graphics, camera push-ins; `--stills` for previews, `--range` for rendering |
| `fx/fxlib.py` | Shared glow, lens streak, ray, dotted sphere, shaded ball and atom drawing |
| `fx/audio_v2.py` | Ambient bed that ducks under the voice, plus synced sound effects |
| `fx/detect.py` | Pose, hands and mask on a single image |
| `examples/` | The first test: effects animated over a single still frame |

## Filming tips
Dark plain background (or any room: the cutout removes it), dark clothes, one soft light on the face,
phone on a tripod at 1080p or 4K, hands kept in frame, a short pause after each big gesture.
