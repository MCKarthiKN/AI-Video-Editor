# Kinematics Ep 1 - "Is anything truly at rest?" (8:51, multi-clip episode)

Pipeline used for a long, multi-clip episode (7 source clips, to-camera + overhead + toy demo):

1. `transcribe_chunks.py` (scripts/) - transcribe in silence-bounded chunks <= 24 s. faster-whisper's 30 s windows
   silently drop words at the edges; chunking recovers them. Re-transcribe the *edited* voice track as a cut check.
2. `edl.py` - keep-ranges per clip (retakes, false starts, stammers removed by hand from word timings) + automatic
   pause tightening (>0.55 s, 0.9 s for overhead drawing shots). Writes edl.json; `fx.T(beat, clip_t)` maps words to timeline.
3. 1080p25 proxies of overhead clips (decode 4K HEVC once); RVM only on the to-camera spans, in chunks that each fit
   the 2 h background limit (RVM is recurrent: warm up ~50 frames before each chunk start).
4. three.js beats (templates/three: earth.html, train.html, cosmos.html) rendered with rec.js; train at 50 fps + tmix=2.
5. `beats_over.py` - overhead/toy beats: dark grade + hologram overlays fitted to the drawn board (fit the number line to
   the actual ticks; hand-drawn lines are not straight in perspective). `beats_pres.py` - presenter beats: RVM cutout on
   the void, alternating zoom per kept piece to hide jump cuts, cutaways blended in. Never put cards over the face:
   check where the presenter stands in *each* clip.
6. `score_ep01.py` - voice cut with per-piece frame-exact lengths (keeps A/V sync), levelling, event cues from word
   timings; `assemble.sh` - concat + loudnorm -14 LUFS.
Lemon Milk (user font, capitals only, donationware) lives in the work dir, not in the repo.
