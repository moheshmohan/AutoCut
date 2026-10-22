# filmcraft-vision

Privacy detection for FilmCraft's automatic blur (`clip.autoRedact`, `sequence.autoRedact` in
`crates/engine/src/redact.rs`): faces and vehicle number plates found in video frames with
small local models, linked into tracks; and the profile pass, which finds the faces in profile
or half hidden that the face detector misses from people's poses. L2; pure Rust.

## Models

Downloaded by `redact.downloadModels` after the user confirms (4.4 MB, and 35.8 MB for the
profile pass), pinned to a Hugging Face revision and SHA-256 (`catalog`), never bundled or
committed:

| model | file | size | licence |
|---|---|---|---|
| YuNet face detector (OpenCV Zoo, `face_detection_yunet_2023mar`) | `face_detection_yunet_2023mar.onnx` | 232,589 B | MIT, © 2020 Shiqi Yu |
| LPD-YuNet licence-plate detector (OpenCV Zoo, `license_plate_detection_lpd_yunet_2023mar`) | `license_plate_detection_lpd_yunet_2023mar.onnx` | 4,146,213 B | Apache-2.0 (Dong Xu, watrix.ai) |
| D-FINE-N person detector (COCO; the Transformers port `ustc-community/dfine-nano-coco`, revision `066438d3`) | `dfine_n_coco.safetensors`, `dfine_n_coco.json` | 15,278,996 B, 6,557 B | Apache-2.0 (Peng et al., USTC) |
| RTMPose-s pose network (body7, 256 × 192; OpenMMLab's deployment zip, mirrored by its author `Tau-J/RTMPose`, revision `cd4d7095`, byte-identical to download.openmmlab.com's) | `rtmpose-s_simcc-body7_256x192.zip` | 20,496,303 B | Apache-2.0 (OpenMMLab) |

The last two are the profile pass's (faces in profile and half hidden, which YuNet misses; 35.8
MB). Licences checked 2026-10-22: D-FINE's model card and repository (`Peterande/D-FINE`) say
Apache-2.0; RTMPose's are MMPose's (Apache-2.0); RTMPose's body7 weights were trained on seven
public pose datasets (COCO, AI Challenger, CrowdPose, Halpe, MPII, PoseTrack18, sub-JHMDB).
Detectors not used for their licences: SCRFD / InsightFace weights (non-commercial), Ultralytics
YOLO-face (AGPL), YOLO5Face (GPL).

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
- `gpu` (feature `model`, used on macOS): the same layers as fused WGSL compute passes through
  wgpu (`net.wgsl`: the 1×1 convolution with its bias; the depthwise convolution with its bias,
  ReLU and pool; the pool; the upsampled sum), one command buffer a picture, the heads read back
  once. The graphs are written once over a `Backend` (CPU or GPU). The CPU passes are the
  reference and the fallback: a picture that fails or panics on the GPU is looked at again on the
  CPU; after `MAX_GPU_FAILURES` (3) in a row, or a device error, the detectors keep to the CPU.
  `FILMCRAFT_VISION_CPU` keeps the CPU; other platforms always use it. candle's Metal backend was
  measured first and was five times slower than the CPU (a unit as generic tensor operations).
- `post`: YuNet's anchor-free decoding (score √(class × objectness), centre offsets, log sizes,
  five landmarks) and LPD-YuNet's prior boxes (strides 8/16/32/64; 3, 2, 2, 3 sizes; four
  corner points; score √(softmax × IoU)), and greedy non-maximum suppression (IoU 0.3).
- `track`: detections → tracks. Greedy linking by overlap with each track's predicted box, or
  for fast motion by nearness (centres within 0.8 box sizes, sizes within 2×), then (`pan`,
  camera-motion compensation) the tracks left over predicted moved with the picture — the
  median shift of the tracks matched in that frame — and linked again (in a fast pan a small
  face moves two of its widths between analysed frames: each sighting started a new track, too
  brief to hide, and its mask stayed behind), the boxes
  compared grown by 15 % a side (`grow`: a head boxed as tightly as a face, moving its own width
  between two frames in a crowd, keeps its track); gaps up to 0.7 s filled on straight lines,
  up to 1.5 s after an unsure sighting (`unsure_gap`, hysteresis: `unsure` — a box within 2 % of
  the frame's edge, `EDGE_MARGIN`, or a pose head scoring under 0.5, `UNSURE_POSE`; a head cut
  by the frame is lost and found again from frame to frame, and its mask must neither start on
  one sighting nor drop on one miss; the prediction reaches at most `max_gap` ahead);
  Gaussian-weighted local linear smoothing (σ 0.08 s for places, 0.2 s for sizes, `smooth_size`;
  no lag, ends included; a sighting's centre moved at most 0.15 of its box, `SMOOTH_REACH`: a whip
  pan bends the path faster than the smoothing follows), a sighting whose size is 1.6 times off
  its neighbours' within 0.5 s taking theirs (`SIZE_SPIKE`: one frame's wrong keypoint does not
  blow a mask up). A head from a
  pose scoring under 0.35 (`WEAK_POSE`) is a weak sighting: matched after the other detections
  and only to a track still unmatched, never starting one (ByteTrack's second association), so a
  profile head the pose network is unsure of in some frames keeps its track. YuNet's faces and
  pose heads link into the same tracks (a visitor turning into profile is one track; `pose_hits`
  counts the pose's sightings); a face's landmarks are not followed (only a plate's corners).
  `verdict` keeps faces at least 5 % of the picture tall at some point (within 15 %,
  `SIZE_TOLERANCE`: detectors measure one face some 10 % differently at different picture sizes,
  and a face at the limit by one measure must be hidden by both) and plates at least 2.5 % of its
  width, seen twice or once with a score ≥ 0.9, and not inside an `Exclude` box (at its time ±
  1 s). `padded` / `padded_corners` grow boxes and plate quads; `at_edge` says whether a box meets
  the frame's edge. `Kind::Target` tracks (named logos and signs, followed by the engine from a
  vision model's boxes, `crates/engine/src/redact_targets.rs`) carry their `label`, are smoothed
  the same way (`smoothed`) and are hidden at any size, exclusions or not.
- `post::plausible_plate`: LPD-YuNet also fires on text, badges and large shapes; a plate must be
  at least 1.4 times as wide as tall, at most 40 % of the picture wide and 25 % tall.
- `safetensors`, `zip`: strict readers of the profile models' files (D-FINE's safetensors weights;
  RTMPose's `end2end.onnx` inside OpenMMLab's zip, inflated with miniz_oxide and checked against
  its CRC-32). Offsets, lengths, element counts and inflated sizes are checked and capped; both
  mutation-fuzzed.
- `head`: from a pose to a face to hide (plain Rust, no model). The face shows when the nose or
  an eye scores ≥ 0.5, or ≥ 0.25 on a pose that does not face away (`faces_away`: eyes, ears or
  shoulders mirrored, the person's left on the picture's left — how the pose network places a
  back); a head seen from behind identifies nobody and is left alone. The head's width is the
  largest of the face keypoints' distances over their shares of a head's width (ear–ear 1.05,
  eye–eye 0.41, nose–ear 0.79, eye–ear 0.56, nose–eye 0.31; calibrated against YuNet's faces of
  the same people in the OnRobot and Ameca clips), from the sure keypoints (≥ 0.5) when two are
  sure, else the middle estimate with outlying keypoints dropped; at most 3.5 times the face
  keypoints' spread, and the shoulders' distance over 2.6 only without two face keypoints. The
  box is the face as YuNet boxes it (`FACE_BOX`): 0.88 × 1.14 head widths from the front, 0.98 ×
  1.34 turned (the turn shortens the distances that measure the width), centred between the
  outermost face keypoints (ears for a frontal face, ear and nose in profile), its centre 0.13
  widths below the eyes, grown to keep the nose and eyes inside with a margin; keypoints outside
  the person's box (grown by a quarter) are ignored. Against YuNet's faces of the same people
  (808 pairs on 174 frames, 2026-10-22; ignored test `head_boxes_match_the_face_detectors`) the
  head box is a median 1.05 times as tall and 1.06 as wide (90 % under 1.40); the earlier box
  (a head wide, 1.3 tall: hair and ears) was 1.17 and 1.26. `candidates` makes the heads that no
  YuNet face matches (one face to one head, the nearest: its centre inside the head box grown by
  a tenth, heights within 3×; a neighbour's face must not hide a head, which shows once the
  neighbour is blurred) into face detections marked `pose`, scored at most 0.84
  (`HEAD_SCORE_MAX`: under the engine's sure-face 0.85 and the tracker's seen-once-surely 0.9, so
  a pose head is always seen twice and the vision check's "a robot" is believed).
- `layers` (feature `model`): the profile networks' building blocks on the CPU (candle matrix
  products and convolutions; depthwise convolutions, biases, activations, norms, attention and
  bilinear sampling in plain Rust), and the `Backend` the convolutional parts are written over.
- `dfine` (feature `model`): D-FINE (arXiv 2410.13842) by hand from its weights and
  `config.json`: HGNetv2 backbone (batch norms folded, learnable affine blocks), the hybrid
  encoder (one transformer layer on the coarsest level, RepNCSPELAN4 top-down and bottom-up with
  the RepVGG 1×1 branches folded into the 3×3 kernels), and the decoder (query selection,
  self-attention, deformable cross-attention by bilinear sampling, the sigmoid gate, FDR box
  distributions over 33 bins, LQE). Pictures keep their shape (each side a multiple of 32, 640 ×
  352 for 16:9); the sine positions of a non-square map are the ones a square map's token gets in
  the reference (which builds them with a transposing mesh grid).
- `rtmpose` (feature `model`): RTMPose by hand from its ONNX file: CSPNeXt backbone (channel
  attention, SPP), the RTMCC head (scale norms, the gated attention unit, SimCC x and y); a
  person's box grown by a quarter, made 3:4 and warped to 192 × 256 (bilinear, black outside),
  keypoints at the SimCC maxima, scored by the smaller of the two.
- `gpunn` (feature `model`, used on macOS): the whole profile pass on the face detector's
  device as WGSL compute passes — `nn.wgsl` (a convolution with bias, activation and residual; a
  tiled 1×1 convolution; pools, upsampling, copies, padding, channel attention), `tok.wgsl` (tiled
  matrix products with bias, activation, residual and clamp; layer and scale norms; softmax;
  multi-head attention; D-FINE's deformable sampling, sigmoid gate, query selection (a stable
  ranking), first boxes, FDR boxes and quality statistics; RTMPose's gated attention unit and
  SimCC maxima) and `img.wgsl` (the frame area-averaged for D-FINE, the people's crops warped for
  RTMPose). A frame goes up once (RGBA8, a u32 a pixel); a frame costs two read-backs: D-FINE's
  300 queries, then every person's 17 keypoints (index and score along x and y). The transformer
  parts mirror the CPU code operation by operation (`gpunn_tests`: each against its CPU
  reference on random data; the sampling, box and statistics code is shared as `deform_cpu`,
  `fdr_box`, `lqe_stats_cpu`, `gau_cpu`). The CPU stays the reference and the fallback, network by
  network (same rules as `gpu`). Metal compiles WGSL with fast math, so NaN is tested by its bits.
- `people` (feature `model`): `ProfileDetector` — people (score ≥ 0.25, at least 4 % of the
  picture tall, at most 24, suppression at IoU 0.7), their poses, and the faces to add; the whole
  pass on the GPU when both networks are there (`people_cpu`: the picture and crops prepared on
  the CPU, each network where it runs).

## Accuracy (measured)

Against ONNX Runtime 1.19 on a 640 × 360 GITEX frame (`oracle.py`, ignored tests
`yunet_matches_onnx_runtime`, `lpd_matches_onnx_runtime`): YuNet's outputs within 7e-6, LPD's
within 7e-6, and the six faces equal to OpenCV 5.0 `FaceDetectorYN`'s (IoU > 0.99).

The profile networks against ONNX Runtime 1.31 (`tools/profile_oracle.py`; ignored tests
`dfine_matches_onnx_runtime`, `rtmpose_matches_onnx_runtime`), on a 640 × 640 photo of two
people: D-FINE-N's 300 queries the same, the people's boxes within 1e-7 and scores within 1e-4
(the export `onnx-community/dfine-nano-coco-ONNX` as the oracle); RTMPose-s's SimCC logits within
3e-6 on two crops. The GPU against the CPU (`profile_gpu_matches_the_cpu`, 2026-10-22, on a
1080p frame of 24 visitors and 35 frames of the OnRobot and Ameca clips): D-FINE's encoder within
1.2e-5, its decoder alone (the same inputs) boxes within 1.5e-6 and logits within 4.3e-5, the
whole network boxes within 6.4e-6 and logits within 8.9e-5; RTMPose's SimCC within 5.9e-6; the
whole pass found the same 561 people (boxes within 0.007 px), 9534 of their 9537 keypoints within
a pixel (the others: two SimCC bins nearly tied).

## Speed (measured, release, Apple M-series CPU, 2026-10-19)

YuNet 9.8 ms at 640 × 360, 3.6 ms at 320 × 180; LPD-YuNet 25.6 ms at 640 × 360, 16.2 ms at
480 × 270 (`detection_speed`).

On the GPU (M3 Max, 2026-10-21, a 1920 × 1080 frame, the machine loaded by parallel builds at a
load average of about 35, fastest of three rounds): faces at a long side of 640 / 1280 / 1920 in
11 / 19 / 29 ms (CPU at that load 81 / 115 / 235 ms; 42 ms at 1920 on a quiet machine), plates
at 960 in 40 ms (CPU 281 ms; 94 ms quiet). On 16 frames of the showcase's footage the GPU finds
the same 220 faces with boxes 0.000 px apart, and the plate network's outputs are within 9.3e-7
(relative) of the CPU's (`gpu_detections_match_the_cpu`).

The profile pass (M3 Max, 14 cores, 2026-10-22, release, `profile_pass_speed`, a 1920 × 1080
frame of 24 visitors around Ameca, fastest of five rounds): all on the GPU, people and poses in
101 ms on an idle machine (load 5; 9.9 frames a second) and 105 ms (median 108; 9.2–9.5 frames a
second) with 56 busy processes at a load of 33–52; D-FINE alone 27 ms. Before (the decoder,
the transformer layer, the pose head, the picture's preparation and the crops on the CPU): 118 ms
idle, 973 ms at a load of 32–55 (1.0 frame a second; D-FINE's decoder alone 467 ms); all on the
CPU 308 ms idle, 4.4 s loaded. The CPU's share now is recording the passes and two small
read-backs, so a busy machine barely slows it.

## Tests

`cargo test -p filmcraft-vision --features model` (`gpu_layers_match_the_cpu` compares every
layer of the face detectors on the GPU with the CPU's, `gpunn_tests` every pass of the profile
networks' GPU path — matrix products with each activation, norms, softmax, attention, the
deformable sampling, selection and gathering with ties, FDR boxes, quality statistics, the gated
attention unit, SimCC maxima with ties and NaN, the frame's resize and the crops, hostile shapes
— and several threads at once; skipped where there is no GPU). The live tests need the models and an oracle
directory: `FILMCRAFT_VISION_MODELS=<dir holding privacy-detectors/>
FILMCRAFT_VISION_ORACLE=<dir with picture.png and oracle.py's outputs> cargo test -p
filmcraft-vision --features model -- --ignored`. `tools/oracle.py` (ONNX Runtime and OpenCV in a
virtualenv, `pip install onnx onnxruntime opencv-python-headless`; test-only, never shipped)
writes the reference outputs:
`python -I tools/oracle.py <face.onnx> <plate.onnx> <picture.png> <out dir>` (the picture is a
frame at most 640 px a side; copy it into the out dir as `picture.png`).

The profile networks' live tests: `FILMCRAFT_VISION_MODELS=<dir holding privacy-detectors/ with
the profile files> FILMCRAFT_PROFILE_ORACLE=<out dir of tools/profile_oracle.py> cargo test -p
filmcraft-vision --features model --release profile_ -- --ignored` (`pip install onnxruntime
numpy pillow`; `python -I tools/profile_oracle.py <D-FINE onnx> <RTMPose end2end.onnx>
<picture> <out dir>`, with the picture copied next to the out dir as `people.png` for the speed
test). `profile_on_pictures` (`FILMCRAFT_PROFILE_PICTURES=<dir of PNGs>`) draws what YuNet and
the profile pass see into `annotated/` and prints each person's keypoint scores and the head
box against YuNet's face (the calibration above); `profile_dump` writes what both find on every
picture as JSON (`FILMCRAFT_PROFILE_DUMP`) for the calibration and flicker analyses;
`head_boxes_match_the_face_detectors` measures the head boxes against YuNet's faces (the
calibration of `FACE_BOX`) and `blurred_faces_defeat_the_face_detector` how strong a blur inside
the redaction's mask keeps YuNet from finding a face again (the engine's `BLUR_PER_HEIGHT`).
`profile_gpu_matches_the_cpu` and `profile_pass_speed` take a 1080p frame from
`FILMCRAFT_PROFILE_PICTURE` (and the whole pass's parity every PNG of
`FILMCRAFT_PROFILE_PICTURES`) when there is no oracle directory. The live checks of the whole
redaction are AutoCut's examples `profile_redact` and `redact_stage` (a window of a clip redacted
and checked, the masks' area, merging and flicker measured; `crates/autocut/examples/`).
