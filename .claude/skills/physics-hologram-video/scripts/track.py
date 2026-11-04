"""Track body mask, pose and hands on every frame of a video.

usage: python3 track.py <video> <out_dir>
writes: out_dir/frames/NNNNN.jpg, out_dir/masks/NNNNN.png, out_dir/track.json
"""
import sys, os, json, subprocess
import numpy as np, cv2, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

M = os.environ.get('FX_MODELS', os.path.expanduser('~/fx_models')) + '/'
video, out = sys.argv[1], sys.argv[2]
os.makedirs(f'{out}/frames', exist_ok=True)
os.makedirs(f'{out}/masks', exist_ok=True)
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', video, '-q:v', '2', f'{out}/frames/%05d.jpg'], check=True)
files = sorted(os.listdir(f'{out}/frames'))
fps = 30.0
cpu = BaseOptions.Delegate.CPU
pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=M + 'pose_landmarker_heavy.task', delegate=cpu),
    running_mode=vision.RunningMode.VIDEO, output_segmentation_masks=True))
hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=M + 'hand_landmarker.task', delegate=cpu),
    running_mode=vision.RunningMode.VIDEO, num_hands=2,
    min_hand_detection_confidence=0.4, min_tracking_confidence=0.4))
POSE_IDS = {0: 'nose', 11: 'l_shoulder', 12: 'r_shoulder', 13: 'l_elbow', 14: 'r_elbow', 15: 'l_wrist', 16: 'r_wrist',
            19: 'l_index', 20: 'r_index', 23: 'l_hip', 24: 'r_hip'}
track = []
for n, f in enumerate(files):
    bgr = cv2.imread(f'{out}/frames/{f}')
    h, w = bgr.shape[:2]
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ts = int(n * 1000 / fps)
    rec = {'n': n, 'pose': None, 'hands': []}
    r = pose.detect_for_video(img, ts)
    if r.pose_landmarks:
        lm = r.pose_landmarks[0]
        rec['pose'] = {k: [lm[i].x * w, lm[i].y * h, lm[i].visibility] for i, k in POSE_IDS.items()}
        m = r.segmentation_masks[0].numpy_view()
        cv2.imwrite(f'{out}/masks/{f[:-4]}.png', (m * 255).clip(0, 255).astype(np.uint8))
    else:
        cv2.imwrite(f'{out}/masks/{f[:-4]}.png', np.zeros((h, w), np.uint8))
    rh = hands.detect_for_video(img, ts)
    for lms, hd in zip(rh.hand_landmarks, rh.handedness):
        rec['hands'].append({'side': hd[0].category_name, 'score': hd[0].score,
                             'pts': [[p.x * w, p.y * h] for p in lms]})
    track.append(rec)
    if n % 60 == 0:
        print('tracked', n, '/', len(files), flush=True)
json.dump({'w': w, 'h': h, 'fps': fps, 'frames': track}, open(f'{out}/track.json', 'w'))
nh = sum(len(t['hands']) for t in track)
print(f'done: {len(files)} frames, pose found in {sum(t["pose"] is not None for t in track)}, hand detections {nh}', flush=True)
