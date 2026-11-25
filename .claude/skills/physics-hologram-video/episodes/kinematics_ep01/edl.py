"""Edit decision list for Kinematics Ep 1. Writes edl.json: ordered pieces with clip, in, out, beat, and timeline start.
Manual cuts remove retakes / false starts / stammers; long pauses inside kept ranges are tightened automatically."""
import json, os
D = os.path.dirname(os.path.abspath(__file__))
OP, GU, TC, FR, PO, WB, EN = ('E01_A_0000_opening', 'E01_A_0100_guess', 'E01_B_0200_toy_close', 'E01_A_0400_frame',
                              'E01_A_0600_position', 'E01_B_0830_whiteboard', 'E01_A_1030_ending')
# (clip, beat, [(in, out), ...])  - clip seconds
PLAN = [
    (OP, 'opening', [(10.0, 33.2), (35.25, 48.25), (52.25, 61.35)]),      # false start + "10 minutes" repeat cut
    (GU, 'guess',   [(11.0, 31.4)]),
    (TC, 'train',   [(7.8, 40.8)]),
    (TC, 'toy',     [(41.7, 49.6), (60.25, 64.95), (65.8, 88.6)]),                     # first "doesn't move" take + aside cut
    (TC, 'idea',    [(89.1, 89.95), (93.8, 110.9)]),                                     # "இதுதாங்க நம்ம வந்து… தாங்க" stammer
    (FR, 'frame',   [(1.4, 8.4), (10.9, 20.4), (31.0, 50.2), (53.7, 54.1)]),            # "இது வந்து ஒரு மாரி" x2, 10 s gap, repeat
    (FR, 'book',    [(54.1, 56.85), (77.7, 92.2)]),                                      # textbook retake
    (PO, 'position',[(2.3, 103.4), (104.9, 118.9), (120.6, 151.45)]),                    # "ஓ ஓ", "இந்த யூனிட்" x2
    (WB, 'origins', [(2.9, 87.05), (87.45, 101.0)]),                                     # "இந்த இந்த", "டிரெண்ட் டிரெண்ட்"
    (EN, 'ending',  [(5.5, 43.4), (54.3, 72.8), (74.75, 129.1)]),                        # first "at rest?" attempt, "இல்லையும் இல்லை"
]
TIGHT = {'position': 0.9, 'origins': 0.9}          # overhead drawing: only tighten long pauses (keeps strokes continuous)
sil = {}
for line in open(f'{D}/silences.txt'):
    c, r, _ = line.split(' ', 2); a, b = map(float, r.split('-')); sil.setdefault(c, []).append((a, b))
pieces, T = [], 0.0
for clip, beat, ranges in PLAN:
    thr = TIGHT.get(beat, 0.55)
    for a, b in ranges:
        cuts = [(s0 + .15, s1 - .15) for s0, s1 in sil.get(clip, []) if s1 - s0 > thr and s0 > a + .2 and s1 < b - .2]
        cur = a
        for c0, c1 in cuts + [(b, b)]:
            if c0 - cur > 0.04:
                pieces.append(dict(clip=clip, beat=beat, i=round(cur, 3), o=round(c0, 3), t=round(T, 3))); T += c0 - cur
            cur = c1
json.dump(dict(pieces=pieces, duration=round(T, 3)), open(f'{D}/edl.json', 'w'), indent=1)
by = {}
for p in pieces: by.setdefault(p['beat'], [0, 0, p['t']]); by[p['beat']][0] += p['o'] - p['i']; by[p['beat']][1] += 1
for k, (d, n, t0) in by.items(): print(f'{k:9s} starts {int(t0 // 60)}:{t0 % 60:04.1f}  {d:6.1f}s  {n} pieces')
print(f'total {int(T // 60)}:{T % 60:04.1f}  ({len(pieces)} pieces)')
