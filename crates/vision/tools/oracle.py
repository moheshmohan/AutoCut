"""Test-only oracle for filmcraft-vision: run the pinned ONNX detectors with ONNX Runtime on a
picture and write their raw outputs (float32, little endian) and OpenCV's final detections, for
the ignored live tests to compare against.

usage: oracle.py <face.onnx> <plate.onnx> <picture.png> <out dir>
"""
import json, os, sys
import numpy as np
import onnx
import onnxruntime as ort
import cv2

face_path, plate_path, pic, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
img = cv2.imread(pic, cv2.IMREAD_COLOR)  # BGR
h, w = img.shape[:2]


def dynamic(path):
    m = onnx.load(path)
    d = m.graph.input[0].type.tensor_type.shape.dim
    d[2].dim_param = 'h'
    d[3].dim_param = 'w'
    return ort.InferenceSession(m.SerializeToString(), providers=['CPUExecutionProvider'])


# YuNet: padded with black to multiples of 32, BGR 0..255
ph, pw = (h + 31) // 32 * 32, (w + 31) // 32 * 32
blob = np.zeros((1, 3, ph, pw), np.float32)
blob[0, :, :h, :w] = img.transpose(2, 0, 1)
s = dynamic(face_path)
names = [o.name for o in s.get_outputs()]
res = s.run(names, {'input': blob})
for n, r in zip(names, res):
    np.asarray(r, np.float32).tofile(os.path.join(out, f'yunet_{n}.f32'))
det = cv2.FaceDetectorYN.create(face_path, '', (w, h), 0.6, 0.3, 5000)
_, faces = det.detect(img)
faces = [] if faces is None else faces.tolist()

# LPD-YuNet: the picture as it is
blob = img.transpose(2, 0, 1)[None].astype(np.float32)
s = dynamic(plate_path)
loc, conf, iou = s.run(['loc', 'conf', 'iou'], {'input': blob})
np.asarray(loc, np.float32).tofile(os.path.join(out, 'lpd_loc.f32'))
np.asarray(conf, np.float32).tofile(os.path.join(out, 'lpd_conf.f32'))
np.asarray(iou, np.float32).tofile(os.path.join(out, 'lpd_iou.f32'))

json.dump({'width': w, 'height': h, 'faces': faces, 'outputs': names, 'priors': int(loc.shape[0])}, open(os.path.join(out, 'oracle.json'), 'w'), indent=1)
print(f'{w}x{h}: {len(faces)} face(s); {loc.shape[0]} plate priors')
