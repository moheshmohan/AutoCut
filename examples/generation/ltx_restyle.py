#!/usr/bin/env python3
"""A generation provider step for AutoCut: make a clip again with LTX-2 through ltx-2-mlx.

AutoCut writes the stretch to change as a small video (each side a multiple of 64, a frame count
the model takes) and runs this with the Python of the environment that has `ltx_pipelines_mlx`.
What it writes has the same number of frames, so the new clip lies exactly over the old one.
Nothing here is part of the app.

`--how` says what the adapter is given:

  edges   the lines of the picture (drawn here with ffmpeg, which ltx-2-mlx needs anyway), for
          an adapter that follows a control video (Lightricks' Union-Control): the shapes and
          the movement stay, what they look like follows the description
  source  the video itself, for an adapter that changes one thing about it (Lightricks' Relight)
  given   the video is already a control video (depth or pose drawn elsewhere)

The clip is made in one pass at its full size, with the adapter applied as the model runs
rather than fused into its weights. Measured on 4-bit weights at 448x256: the tool's default
(a half-size pass, then a refine that does not see the control) did not follow the lines at all,
fused or not; this does.

usage: ltx_restyle.py --model DIR --adapter FILE --how edges|source|given --video IN.mp4
                      --out OUT.mp4 --prompt TEXT --width W --height H --frames N --fps F
                      --seed S [--strength 0..1] [--image FIRST.png] [--two-stage]
"""
import argparse
import os
import subprocess
import sys
import tempfile


def edges(video: str, out: str) -> None:
    """The lines of `video` (white on black), frame for frame."""
    from ltx_core_mlx.utils.ffmpeg import find_ffmpeg

    cmd = [find_ffmpeg(), "-v", "error", "-y", "-i", video, "-an"]
    cmd += ["-vf", "edgedetect=mode=canny:low=0.1:high=0.4,format=yuv420p"]
    cmd += ["-c:v", "libx264", "-crf", "12", out]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="folder of the MLX weights")
    ap.add_argument("--adapter", required=True, help="the adapter's .safetensors file")
    ap.add_argument("--how", required=True, choices=["edges", "source", "given"])
    ap.add_argument("--video", required=True, help="the stretch to make again")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--frames", type=int, required=True, help="8k+1, as many as the video has")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--strength", type=float, default=1.0, help="how closely the video is followed")
    ap.add_argument("--image", help="the picture the new clip starts on")
    ap.add_argument("--two-stage", action="store_true", help="the tool's default passes: faster at large sizes, follows less")
    a = ap.parse_args()
    if a.width % 64 or a.height % 64 or not (64 <= a.width <= 2048 and 64 <= a.height <= 2048):
        print("width and height must be multiples of 64", file=sys.stderr)
        return 2
    if not (9 <= a.frames <= 241 and a.frames % 8 == 1):
        print("the frames must be 8k+1, from 9 to 241", file=sys.stderr)
        return 2
    if not os.path.isfile(a.adapter):
        print(f"the adapter {a.adapter} is not on this computer", file=sys.stderr)
        return 2
    strength = min(max(a.strength, 0.0), 1.0)

    # a small adapter fused into quantized weights is mostly lost when they are quantized again
    os.environ.setdefault("LTX2_LORA_MODE", "unfused")
    from ltx_core_mlx.utils.ffmpeg import probe_video_info
    from ltx_pipelines_mlx.cli import main as ltx_main

    # never more frames than the video holds (a count the model takes: 8k+1)
    have = probe_video_info(a.video).num_frames
    frames = min(a.frames, (have - 1) // 8 * 8 + 1) if have >= 9 else 0
    if frames < 9:
        print(f"the video holds {have} frames; 9 or more are needed", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="autocut-restyle-") as tmp:
        control = a.video
        if a.how == "edges":
            control = os.path.join(tmp, "control.mp4")
            edges(a.video, control)
        sys.argv = ["ltx-2-mlx", "ic-lora", "--model", a.model, "--lora", a.adapter, "1.0"]
        sys.argv += ["--video-conditioning", control, f"{strength:g}"]
        sys.argv += ["--height", str(a.height), "--width", str(a.width), "--frames", str(frames)]
        sys.argv += ["--frame-rate", f"{a.fps:g}", "--prompt", a.prompt]
        sys.argv += ["--seed", str(a.seed % (2**31)), "--output", a.out]
        if a.image:
            sys.argv += ["--image", a.image, "0", "1.0"]
        if not a.two_stage:
            sys.argv += ["--single-stage"]
        ltx_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
