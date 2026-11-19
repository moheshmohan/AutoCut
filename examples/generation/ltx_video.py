#!/usr/bin/env python3
"""A generation provider for AutoCut: LTX-2 video on Apple Silicon through ltx-2-mlx.

AutoCut runs this with the Python of the environment that has `ltx_pipelines_mlx`
(see docs/generation.md); it writes one MP4 and exits. Nothing here is part of the app.

usage: ltx_video.py --model DIR --out OUT.mp4 --prompt TEXT --width W --height H
                    --frames N --fps F --seed S [--image START.png]
"""
import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="folder of the MLX weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", default=None, help="the picture the clip starts from")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=272)
    ap.add_argument("--frames", type=int, default=49, help="a multiple of 8, plus 1")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if not (16 <= a.width <= 2048 and 16 <= a.height <= 2048 and 9 <= a.frames <= 481 and 1 <= a.fps <= 120):
        print("size, frame count or rate out of range", file=sys.stderr)
        return 2

    from ltx_pipelines_mlx import DistilledPipeline

    pipe = DistilledPipeline(model_dir=a.model)
    pipe.generate_and_save(
        prompt=a.prompt,
        output_path=a.out,
        height=a.height,
        width=a.width,
        num_frames=a.frames,
        frame_rate=a.fps,
        seed=a.seed % (2**31),
        image=a.image or None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
