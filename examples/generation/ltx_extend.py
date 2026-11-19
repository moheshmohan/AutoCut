#!/usr/bin/env python3
"""A generation provider step for AutoCut: carry a shot on with LTX-2 through ltx-2-mlx.

AutoCut writes the last moments of the picture as a small video (33 frames, each side a
multiple of 64) and runs this with the Python of the environment that has `ltx_pipelines_mlx`.
It writes that video with the new frames after it; AutoCut places the new part.
Nothing here is part of the app.

usage: ltx_extend.py --model DIR --video CARRIED.mp4 --out OUT.mp4 --prompt TEXT
                     --frames N --seed S
"""
import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="folder of the MLX weights")
    ap.add_argument("--video", required=True, help="the moments to carry on from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--frames", type=int, default=48, help="new frames, a multiple of 8")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if not (8 <= a.frames <= 240 and a.frames % 8 == 0):
        print("the new frames must be a multiple of 8, from 8 to 240", file=sys.stderr)
        return 2

    from ltx_pipelines_mlx.cli import main as ltx_main

    sys.argv = ["ltx-2-mlx", "extend", "--distilled", "--model", a.model, "--video", a.video]
    sys.argv += ["--extend-frames", str(a.frames // 8), "--prompt", a.prompt]
    sys.argv += ["--seed", str(a.seed % (2**31)), "--output", a.out]
    ltx_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
