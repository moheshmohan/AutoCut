"""Reference outputs of the profile pass's networks (ONNX Runtime), for the ignored tests
`dfine_matches_onnx_runtime` and `rtmpose_matches_onnx_runtime`. Test-only: never shipped.

    python -I profile_oracle.py <dfine.onnx> <rtmpose end2end.onnx> <picture> <out dir>

D-FINE's ONNX export (for instance onnx-community/dfine-nano-coco-ONNX, `onnx/model.onnx`) is
used only as the oracle; FilmCraft reads the safetensors weights. Writes little-endian float32
files: `dfine_input.bin` ([1, 3, 640, 640], RGB 0..1), `dfine_logits.bin`, `dfine_boxes.bin`,
`pose_input.bin` ([2, 3, 256, 192], normalised), `pose_x.bin`, `pose_y.bin`.
Needs `pip install onnxruntime numpy pillow` in a virtualenv.
"""

import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image


def main():
    dfine_path, pose_path, picture, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    im = Image.open(picture).convert("RGB")
    x = np.asarray(im.resize((640, 640), Image.BILINEAR), dtype=np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None].copy()
    s = ort.InferenceSession(dfine_path, providers=["CPUExecutionProvider"])
    logits, boxes = s.run(None, {s.get_inputs()[0].name: x})
    x.tofile(os.path.join(out, "dfine_input.bin"))
    logits.astype(np.float32).tofile(os.path.join(out, "dfine_logits.bin"))
    boxes.astype(np.float32).tofile(os.path.join(out, "dfine_boxes.bin"))
    p = 1 / (1 + np.exp(-logits[0, :, 0]))
    print("people:", [(round(float(p[i]), 3), [round(float(v), 3) for v in boxes[0, i]]) for i in np.argsort(-p)[:5]])

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    crops = []
    w, h = im.size
    for box in [(0, 0, w, h), (0, 0, w // 2, h)]:
        c = np.asarray(im.crop(box).resize((192, 256), Image.BILINEAR), dtype=np.float32)
        crops.append(((c - mean) / std).transpose(2, 0, 1))
    px = np.stack(crops).astype(np.float32)
    s = ort.InferenceSession(pose_path, providers=["CPUExecutionProvider"])
    sx, sy = s.run(None, {s.get_inputs()[0].name: px})
    px.tofile(os.path.join(out, "pose_input.bin"))
    sx.astype(np.float32).tofile(os.path.join(out, "pose_x.bin"))
    sy.astype(np.float32).tofile(os.path.join(out, "pose_y.bin"))
    print("pose:", sx.shape, sy.shape)


if __name__ == "__main__":
    main()
