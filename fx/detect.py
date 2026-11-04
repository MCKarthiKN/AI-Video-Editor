import sys, os, json, cv2, numpy as np, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
M = os.environ.get('FX_MODELS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')) + '/'
src, out = sys.argv[1], sys.argv[2]
bgr = cv2.imread(src); h, w = bgr.shape[:2]
img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
cpu = BaseOptions.Delegate.CPU
res = {}
with vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=M+'pose_landmarker_heavy.task', delegate=cpu),
        output_segmentation_masks=True)) as pl:
    r = pl.detect(img)
    if r.pose_landmarks:
        lm = r.pose_landmarks[0]
        names = {0:'nose',11:'l_shoulder',12:'r_shoulder',15:'l_wrist',16:'r_wrist',19:'l_index',20:'r_index',23:'l_hip',24:'r_hip'}
        res['pose'] = {n: [lm[i].x*w, lm[i].y*h, lm[i].visibility] for i, n in names.items()}
        cv2.imwrite(out+'/mask_pose.png', (r.segmentation_masks[0].numpy_view()*255).astype(np.uint8))
with vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=M+'hand_landmarker.task', delegate=cpu), num_hands=2,
        min_hand_detection_confidence=0.3)) as hl:
    r = hl.detect(img)
    res['hands'] = [[[p.x*w, p.y*h] for p in hand] for hand in r.hand_landmarks]
with vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=M+'selfie_multiclass_256x256.tflite', delegate=cpu),
        output_confidence_masks=True)) as sg:
    r = sg.segment(img)
    bg = r.confidence_masks[0].numpy_view()
    cv2.imwrite(out+'/mask_selfie.png', ((1-bg)*255).clip(0,255).astype(np.uint8))
json.dump(res, open(out+'/landmarks.json','w'), indent=1)
print(json.dumps({k:(v if k!='hands' else f'{len(v)} hands') for k,v in res.items()}, indent=1))
