"""usage: python3 transcribe.py <audio_or_video> <out.json> [lang]"""
import sys, os, json, time, subprocess
import numpy as np
from faster_whisper import WhisperModel
t0 = time.time()
m = WhisperModel(os.environ.get('WHISPER_MODEL', 'large-v3'), device='cpu', compute_type='int8', cpu_threads=4)
raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', sys.argv[1], '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'], capture_output=True, check=True).stdout
audio = np.frombuffer(raw, np.float32)
segs, info = m.transcribe(audio, language=(sys.argv[3] if len(sys.argv) > 3 else None),
                          word_timestamps=True, vad_filter=False, beam_size=5, temperature=0.0,
                          condition_on_previous_text=False,
                          initial_prompt=os.environ.get('WHISPER_PROMPT'))
out = {'language': info.language, 'prob': info.language_probability, 'segments': []}
for s in segs:
    out['segments'].append({'start': s.start, 'end': s.end, 'text': s.text,
                            'words': [{'w': w.word, 's': w.start, 'e': w.end} for w in s.words]})
    print(f'[{s.start:5.2f}-{s.end:5.2f}] {s.text}', flush=True)
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
print(f'language={info.language} ({info.language_probability:.2f})  {time.time()-t0:.0f}s')
