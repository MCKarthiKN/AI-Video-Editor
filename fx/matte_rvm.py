"""Robust Video Matting (ONNX) over a frame range.
usage: python3 matte_rvm.py <work_dir> <first> <last_exclusive> [warmup_frames]
reads work_dir/frames/NNNNN.jpg (1-indexed files, 0-indexed frame numbers)
writes work_dir/masks_rvm/NNNNN.png (alpha) and work_dir/fgr_rvm/NNNNN.png (foreground colour)
"""
import sys, os, time
import numpy as np, cv2, onnxruntime as ort

D, f0, f1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
warm = int(sys.argv[4]) if len(sys.argv) > 4 else 45
M = os.path.join(os.environ.get('FX_MODELS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')), 'rvm_resnet50_fp32.onnx')
so = ort.SessionOptions(); so.intra_op_num_threads = 4
sess = ort.InferenceSession(M, so, providers=['CPUExecutionProvider'])
os.makedirs(f'{D}/masks_rvm', exist_ok=True); os.makedirs(f'{D}/fgr_rvm', exist_ok=True)
rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4
ds = np.array([0.4], np.float32)
t0 = time.time()
for n in range(max(0, f0 - warm), f1):
    bgr = cv2.imread(f'{D}/frames/{n + 1:05d}.jpg')
    src = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255
    fgr, pha, *rec = sess.run(None, {'src': src, 'r1i': rec[0], 'r2i': rec[1], 'r3i': rec[2], 'r4i': rec[3], 'downsample_ratio': ds})
    if n >= f0:
        a = (pha[0, 0] * 255 + 0.5).clip(0, 255).astype(np.uint8)
        f = cv2.cvtColor((fgr[0].transpose(1, 2, 0) * 255 + 0.5).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{D}/masks_rvm/{n + 1:05d}.png', a)
        cv2.imwrite(f'{D}/fgr_rvm/{n + 1:05d}.png', f)
    if n % 30 == 0:
        print(f'frame {n}  {time.time() - t0:.0f}s', flush=True)
print('done', flush=True)
