# AI-Video-Editor

Toolkit for turning Tamil physics-explainer clips into dark-stage hologram videos.

**For any video-editing request, use the `physics-hologram-video` skill** (`.claude/skills/physics-hologram-video/SKILL.md`).
It holds the full workflow, the user's standing preferences (Robust Video Matting cutout always, no subtitles
unless asked, subtle wide sound design, storyboard approval before the final render) and all scripts.

**For any book → Instagram quote cards request** (book PDF/epub/transcript, "create cards for this book",
captions, English versions), use the `tanglish-book-cards` skill (`.claude/skills/tanglish-book-cards/SKILL.md`);
follow its "Cloud session" section here.

First step in a new video-editing session: `bash .claude/skills/physics-hologram-video/scripts/setup_env.sh`.
Never commit the user's videos or images (public repo).
