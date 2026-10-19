# filmcraft-vision

Privacy detection for FilmCraft's automatic blur (`clip.autoRedact`, `sequence.autoRedact` in
`crates/engine/src/redact.rs`): faces and vehicle number plates found in video frames with
small local models, linked into tracks. L2; pure Rust.

## Models

Downloaded by `redact.downloadModels` after the user confirms (4.4 MB in all), pinned to a
Hugging Face revision and SHA-256 (`catalog`), never bundled or committed:

| model | file | size | licence |
|---|---|---|---|
| YuNet face detector (OpenCV Zoo, `face_detection_yunet_2023mar`) | `face_detection_yunet_2023mar.onnx` | 232,589 B | MIT, © 2020 Shiqi Yu |
| LPD-YuNet licence-plate detector (OpenCV Zoo, `license_plate_detection_lpd_yunet_2023mar`) | `license_plate_detection_lpd_yunet_2023mar.onnx` | 4,146,213 B | Apache-2.0 (Dong Xu, watrix.ai) |

Licences checked 2026-10-19 in each model's directory (`LICENSE`, `README.md`). LPD-YuNet was
trained on Chinese plates; it finds other countries' plates less reliably.

## Modules

- `onnx`: a strict reader of ONNX files (protobuf): graph nodes with their attributes and the
  float32 / int64 initializers. Lengths are checked against the buffer; files, node and tensor
  counts and tensor sizes are capped; external data, sparse tensors and other types are refused.
  Mutation-fuzzed (truncation and bit flips under `catch_unwind`).
- `net` (feature `model`): both networks written out by hand. The code takes the file's `Conv`
  nodes in graph order and checks each one's kernel, stride, padding, groups and weight shapes
  against the layer it implements, so another file is refused, never misread. Each 1×1
  convolution is a candle matrix product, fused with the 3×3 depthwise convolution after it,
  the biases, the ReLU and the max pool into one parallel pass in plain Rust.
  Input: the RGBA frame fitted inside a longest side (area-averaged, never enlarged), BGR 0..255;
  YuNet's input is padded to multiples of 32 as OpenCV's `FaceDetectorYN` does.
- `post`: YuNet's anchor-free decoding (score √(class × objectness), centre offsets, log sizes,
  five landmarks) and LPD-YuNet's prior boxes (strides 8/16/32/64; 3, 2, 2, 3 sizes; four
  corner points; score √(softmax × IoU)), and greedy non-maximum suppression (IoU 0.3).
- `track`: detections → tracks. Greedy linking by overlap with each track's predicted box, or
  for fast motion by nearness (centres within 0.8 box sizes, sizes within 2×); gaps up to 0.7 s
  filled on straight lines; Gaussian-weighted local linear smoothing (σ 0.08 s; no lag, ends
  included). `verdict` keeps faces at least 5 % of the picture tall at some point and plates at
  least 2.5 % of its width, seen twice or once with a score ≥ 0.9, and not inside an `Exclude`
  box (at its time ± 1 s). `padded` / `padded_corners` grow boxes and plate quads.

## Accuracy (measured)

Against ONNX Runtime 1.19 on a 640 × 360 GITEX frame (`oracle.py`, ignored tests
`yunet_matches_onnx_runtime`, `lpd_matches_onnx_runtime`): YuNet's outputs within 7e-6, LPD's
within 7e-6, and the six faces equal to OpenCV 5.0 `FaceDetectorYN`'s (IoU > 0.99).

## Speed (measured, release, Apple M-series CPU, 2026-10-19)

YuNet 9.8 ms at 640 × 360, 3.6 ms at 320 × 180; LPD-YuNet 25.6 ms at 640 × 360, 16.2 ms at
480 × 270 (`detection_speed`).

## Tests

`cargo test -p filmcraft-vision --features model`. The live tests need the models and an oracle
directory: `FILMCRAFT_VISION_MODELS=<dir holding privacy-detectors/>
FILMCRAFT_VISION_ORACLE=<dir with picture.png and oracle.py's outputs> cargo test -p
filmcraft-vision --features model -- --ignored`. `tools/oracle.py` (ONNX Runtime and OpenCV in a
virtualenv, `pip install onnx onnxruntime opencv-python-headless`; test-only, never shipped)
writes the reference outputs:
`python -I tools/oracle.py <face.onnx> <plate.onnx> <picture.png> <out dir>` (the picture is a
frame at most 640 px a side; copy it into the out dir as `picture.png`).
