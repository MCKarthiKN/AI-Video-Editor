"""Per-frame palm centres for both hands, identity taken from the pose wrists (stable even when hands cross).
usage: python3 palms.py <work_dir>   -> work_dir/palms.npy  shape (frames, 2 [person's right, left], 2 [x, y])"""
import sys, json
import numpy as np
D = sys.argv[1]
fr = json.load(open(D + '/track.json'))['frames']
def palms(f):
    p = f['pose']; hs = [np.mean(np.array(h['pts'])[[0, 5, 9, 13, 17]], 0) for h in f['hands']]
    out = {}
    for k, w in (('R', p['r_wrist']), ('L', p['l_wrist'])):
        w = np.array(w[:2]); best = None
        for h in hs:
            d = np.linalg.norm(h - w)
            if d < 160 and (best is None or d < best[0]): best = (d, h)
        out[k] = best[1] if best else w
    return out
P = np.array([[palms(f)['R'], palms(f)['L']] for f in fr])
np.save(D + '/palms.npy', P)
for t in np.arange(0, len(P) / 25, 0.4):
    n = int(t * 25); print(f'{t:5.1f}s  R {P[n,0].round()}  L {P[n,1].round()}')
