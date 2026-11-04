import sys, os, subprocess, math
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from holo import Hologram
V = sys.argv[1]; out_p = sys.argv[2]; N = int(sys.argv[3])
W, H, FPS = 1920, 1080, 25
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
NAVY = (np.array([0.05,0.02,0.006], np.float32) * np.exp(-((xx-900)**2+(yy-560)**2)/(2*700.0**2))[...,None]).astype(np.float32)
RN = np.sqrt(((xx-1050)/W)**2 + ((yy-480)/H)**2); RN = (RN/RN.max()).astype(np.float32)
tree = Hologram(cv2.imread(V+'/assets/tree.png'), 1000)
TX, TY = 440, 560
def sstep(a, b, x):
    u = min(max((x-a)/(b-a), 0), 1); return u*u*(3-2*u)
ff = subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
                       '-c:v','libx264','-crf','17','-pix_fmt','yuv420p',out_p], stdin=subprocess.PIPE)
for n in range(N):
    t = n / FPS
    A = cv2.imread(V+'/frames/%05d.jpg' % (n+1)).astype(np.float32)/255
    a = cv2.imread(V+'/masks_rvm/%05d.png' % (n+1), 0).astype(np.float32)/255
    fg = cv2.imread(V+'/fgr_rvm/%05d.png' % (n+1)).astype(np.float32)/255
    key = 0.3 + 0.85*np.exp(-((xx-1050)**2 + ((yy-470)*1.05)**2)/(2*330.0**2))
    PG = (1-np.exp(-2*fg*0.95))/(1-np.exp(-2)) * key[...,None] * np.array([1.06,0.98,0.92], np.float32)
    p = sstep(0.1, 0.8, t)
    k = np.clip(1.6*p - (1-RN)*0.6, 0, 1)[...,None]
    bg = A*(1-k) + NAVY*k
    person = A*(1-p) + PG*p
    # tree materialises from the projector up (0.5 - 1.3 s)
    rv = sstep(0.5, 1.3, t)
    if rv > 0:
        img = tree.frame(t)
        reveal = np.clip((tree.yy - tree.h*(1-rv*1.15)) / 40, 0, 1)[..., None]
        img = img * reveal
        ln = np.exp(-((tree.yy - tree.h*(1-rv*1.15))/4)**2)[..., None] * np.array([1.0,0.9,0.6], np.float32) * (rv < 1)
        img = img + ln * (tree.alpha[..., None] > 0.05)
        x0, y0 = TX - tree.w//2, TY - tree.h//2
        sx0, sy0 = max(0,-x0), max(0,-y0); dx0, dy0 = max(0,x0), max(0,y0)
        dx1, dy1 = min(W, x0+tree.w), min(H, y0+tree.h)
        sub = np.clip(img[sy0:sy0+dy1-dy0, sx0:sx0+dx1-dx0], 0, 1)
        bg[dy0:dy1, dx0:dx1] = 1 - (1-bg[dy0:dy1, dx0:dx1])*(1-sub)
    out = person*a[...,None] + bg*(1-a[...,None])
    ff.stdin.write(np.clip(out*255+0.5, 0, 255).astype(np.uint8).tobytes())
ff.stdin.close(); ff.wait(); print('done')
