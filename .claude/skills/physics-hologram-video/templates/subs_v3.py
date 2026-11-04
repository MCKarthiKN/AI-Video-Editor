"""Tamil subtitles + keyword pop-ups as an ASS file (rendered by ffmpeg/libass with HarfBuzz shaping).
usage: python3 subs_v3.py out.ass"""
import sys

def ts(t):
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f'{h}:{m:02d}:{s:05.2f}'

HL = r'{\c&H4FC8FF&}'   # amber highlight (BGR in ASS)
RS = r'{\r}'
lines = [
    (0.00, 2.35, 'இப்ப யாராவது ஒரு ஹியூமன்'),
    (2.35, 3.95, f'அவருடைய {HL}எமோஷன{RS} வந்து'),
    (3.95, 5.35, 'வெளிய கொண்டு வரும்போது'),
    (5.40, 7.45, f'அவரு மண்டை உள்ள ஒரு {HL}பிக்சர்{RS} வந்திருக்கும்'),
    (7.50, 10.40, f'அந்த பிக்சர்ல அவரு {HL}மட்டும்தான்{RS} வாழ்ந்திருப்பாரு'),
    (10.50, 14.10, f'சோ அந்த பிக்சர்ல இருந்து நம்மகிட்ட வந்து {HL}பேசுவாரு{RS}'),
    (14.15, 16.45, 'ஆனா நம்மளுக்கு வந்து அந்த பிக்சர் வர்றதுக்கு'),
    (16.45, 18.10, f'கொஞ்சம் {HL}டிலே{RS} ஆகும்'),
    (18.25, 20.35, 'அதாவது நம்மளுக்கு ஒரு குறிப்பிட்ட டைம்ல'),
    (20.35, 22.15, f'அந்த பிக்சர் வர்றதுக்கு {HL}ரொம்ப டிலே{RS} ஆகும்'),
    (22.15, 22.90, 'அதனாலதான்…'),
]
pop = r'{\fad(120,250)\fscx60\fscy60\t(0,180,\fscx112\fscy112)\t(180,320,\fscx100\fscy100)}'
pops = [
    (2.90, 4.70, 255, 420, 'Pop', pop + 'எமோஷன்'),
    (6.50, 8.00, 540, 68, 'PopSmall', pop + 'பிக்சர்'),
    (16.90, 18.30, 872, 470, 'Pop', pop + 'டிலே'),
    (21.20, 22.60, 540, 1380, 'PopBig', r'{\fad(80,200)\fscx60\fscy60\t(0,150,\fscx115\fscy115)\t(150,300,\fscx100\fscy100)'
     r'\t(300,420,\frz-3)\t(420,540,\frz3)\t(540,660,\frz-2)\t(660,780,\frz0)}ரொம்ப டிலே!'),
]
head = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,Noto Sans Tamil,62,&H00FFFFFF,&H00FFFFFF,&H00100804,&H90000000,-1,0,0,0,100,100,0,0,1,4,2,2,70,70,250,1
Style: Pop,Noto Sans Tamil,78,&H00FFF0D8,&H00FFFFFF,&H00FF9A3C,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,5,10,10,10,1
Style: PopSmall,Noto Sans Tamil,56,&H00FFF0D8,&H00FFFFFF,&H00FF9A3C,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,5,10,10,10,1
Style: PopBig,Noto Sans Tamil,104,&H004FC8FF,&H00FFFFFF,&H00142CFF,&H00000000,-1,0,0,0,100,100,0,0,1,4,0,5,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
ev = []
for a, b, txt in lines:
    ev.append(f'Dialogue: 0,{ts(a)},{ts(b)},Sub,,0,0,0,,{{\\fad(90,90)}}{txt}')
for a, b, x, y, st, txt in pops:
    # soft glow layer underneath + crisp layer on top
    ev.append(f'Dialogue: 1,{ts(a)},{ts(b)},{st},,0,0,0,,{{\\pos({x},{y})\\blur8\\bord10\\1a&HB0&}}' + txt)
    ev.append(f'Dialogue: 2,{ts(a)},{ts(b)},{st},,0,0,0,,{{\\pos({x},{y})\\blur1}}' + txt)
open(sys.argv[1], 'w', encoding='utf-8').write(head + '\n'.join(ev) + '\n')
print('wrote', sys.argv[1], len(ev), 'events')
