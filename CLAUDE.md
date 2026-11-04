# AI-Video-Editor

Toolkit for turning Tamil physics-explainer clips into dark-stage hologram videos.

**For any video-editing request, use the `physics-hologram-video` skill** (`.claude/skills/physics-hologram-video/SKILL.md`).
It holds the full workflow, the user's standing preferences (Robust Video Matting cutout always, no subtitles
unless asked, subtle wide sound design, storyboard approval before the final render) and all scripts.

First step in a new session: `bash .claude/skills/physics-hologram-video/scripts/setup_env.sh`.
Never commit the user's videos or images (public repo).
