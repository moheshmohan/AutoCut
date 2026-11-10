#!/usr/bin/env python3
"""Render a local composition and print the AutoCut import command as JSON."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def render(args):
    source = args.source.expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in {".tsx", ".html"}:
        raise ValueError("choose a local .tsx or .html composition")

    output = args.output.expanduser().resolve()
    output.mkdir()  # Reserve this revision without overwriting an earlier render.
    with TemporaryDirectory(dir=output) as work:
        name = "frames" if args.alpha else f"{output.name}.mp4"
        staged = Path(work) / name
        render_args = [args.raea, "render", str(source), "-o", str(staged)]
        if args.composition:
            render_args.extend(["--composition", args.composition])
        render_args.extend(["--alpha"] if args.alpha else ["--quality", "master"])
        subprocess.run(render_args, cwd=source.parent, stdout=sys.stderr, check=True)

        probe_args = [args.cli, "--json", "probe"]
        if args.alpha:
            probe_args.extend([str(staged / "frame-00000.png"), "--image-sequence"])
        else:
            probe_args.append(str(staged))
        probe = json.loads(subprocess.check_output(probe_args, text=True))
        if not probe.get("video"):
            raise ValueError("AutoCut did not find a video stream in the render")

        asset = output / name
        if args.alpha:
            sequence = probe["imageSequence"]
            if not probe["video"]["has_alpha"]:
                raise ValueError("rendered PNGs lack an alpha channel")
            if sequence["missing"]:
                raise ValueError(f"missing rendered frames: {sequence['missing']}")
            command = "file.importImageSequence"
            params = {"path": str(asset / "frame-00000.png"), "fps": args.fps}
        else:
            command = "file.import"
            params = {"paths": [str(asset)]}
        staged.rename(asset)

    manifest = {"command": command, "params": params}
    report = json.dumps(manifest, indent=2)
    (output / "autocut-import.json").write_text(report + "\n", encoding="utf-8")
    print(report)


def main():
    installed = Path.home() / ".raea/versions/0.1.0/bin/raea"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, help="a new revision directory under an existing parent")
    parser.add_argument("--raea", default=str(installed) if installed.is_file() else shutil.which("raea"))
    parser.add_argument("--cli", default=shutil.which("filmcraft-cli"), help="path to the AutoCut CLI")
    parser.add_argument("--composition", help="registered composition ID")
    parser.add_argument("--alpha", action="store_true", help="render transparent PNG frames")
    parser.add_argument("--fps", type=float, help="source frame rate for alpha import, matching the composition")
    args = parser.parse_args()
    if not args.raea or not args.cli:
        parser.error("provide --raea and --cli or put both commands on PATH")
    if args.alpha and (args.fps is None or not 1 <= args.fps <= 240):
        parser.error("--alpha requires --fps from 1 to 240, matching the composition")
    if not args.alpha and args.fps is not None:
        parser.error("--fps applies only to --alpha; MP4 carries its own frame rate")
    # Resolve explicit relative binary paths before changing to the composition directory.
    args.raea = str(Path(args.raea).expanduser().resolve())
    args.cli = str(Path(args.cli).expanduser().resolve())
    try:
        render(args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Render handoff failed: {error}\n")


if __name__ == "__main__":
    main()
