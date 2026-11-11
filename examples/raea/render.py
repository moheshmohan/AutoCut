#!/usr/bin/env python3
"""Local RAEA review/master handoff, native preflight and content-addressed reuse."""
import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory


def command(args, name, params):
    call = [args.cli, "--json", "exec", name]
    call.extend(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in params.items())
    return json.loads(subprocess.check_output(call, text=True))


def dimensions(value):
    match = re.fullmatch(r"([0-9]{1,5})x([0-9]{1,5})", value)
    if not match or any(not 1 <= int(n) <= 16384 for n in match.groups()):
        raise argparse.ArgumentTypeError("dimensions must be WxH, each 1..16384")
    return tuple(map(int, match.groups()))


def render(args):
    source = args.source.expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in {".tsx", ".html"}:
        raise ValueError("choose a local .tsx or .html composition")
    if args.preview and source.suffix.lower() == ".html":
        raise ValueError("HTML previews use the HTML composition's own responsive sizing; render without --preview")
    output = args.output.expanduser().resolve()
    output.mkdir()  # A new revision; no earlier render is overwritten.
    spec = json.loads(args.spec.read_text()) if args.spec else {}
    if not isinstance(spec, dict):
        raise ValueError("preflight spec must be a JSON object")
    if "expected" in spec and not isinstance(spec["expected"], dict):
        raise ValueError("preflight expected must be a JSON object")
    width, height = args.size or args.canvas or (0, 0)
    if args.preview:
        if not width or not args.canvas:
            raise ValueError("--preview needs --canvas and --size, e.g. 960x540 and 3840x2160")
        scale = min(1.0, 960 / max(width, height))
        width, height = max(2, round(width * scale)), max(2, round(height * scale))
    dependencies = [source, *(path.expanduser().resolve(strict=True) for path in args.dependency)]
    if args.spec:
        dependencies.append(args.spec.resolve(strict=True))
    # Runtime identity is content-based too; updating a separately installed RAEA invalidates it.
    dependencies.append(Path(args.raea))
    external = {"tool": "RAEA", "composition": args.composition, "alpha": args.alpha,
                "fps": args.fps, "size": [width, height], "canvas": args.canvas,
                "quality": "draft" if args.preview else "master", "schema": 3}
    cache = command(args, "render.cachePlan", {"dependencies": list(map(str, dependencies)), "external": external}) if args.cache else None
    if cache and not cache.get("cacheAvailable", True):
        raise ValueError("native cache directory is unavailable; configure a data directory or omit --cache")
    with TemporaryDirectory(dir=output) as work:
        work = Path(work)
        name = "frames" if args.alpha else f"{output.name}.mp4"
        staged = work / name
        if cache and cache["hit"]:
            artifact = Path(cache["artifact"]["path"])
            if args.alpha:
                staged.mkdir()
                with zipfile.ZipFile(artifact) as archive:
                    members = archive.infolist()
                    if len(members) > 30000 or sum(m.file_size for m in members) > 32 * 1024**3:
                        raise ValueError("cached frame archive exceeds bounds")
                    for member in members:
                        if not re.fullmatch(r"frame-[0-9]{5,8}\.png", member.filename):
                            raise ValueError("cached archive has an invalid frame name")
                    archive.extractall(staged)
            else:
                shutil.copyfile(artifact, staged)
        else:
            render_args = [args.raea, "render", str(source), "-o", str(staged)]
            if args.composition:
                render_args.extend(["--composition", args.composition])
            render_args.extend(["--alpha"] if args.alpha else ["--quality", external["quality"]])
            if width:
                render_args.extend(["--w", str(width), "--h", str(height)])
                # v0.1.0 needs the output stage as well as --w/--h. Responsive
                # compositions read it through useVideoConfig; fixed layouts are
                # rejected by the size check rather than silently called 4K.
                render_args.extend(["--stage", f"{width}x{height}"])
            # GPU coordination remains the same shared lock as the editing pipelines. The
            # caller holds it across this helper; cache hits and native QA need no RAEA render.
            subprocess.run(render_args, cwd=source.parent, stdout=sys.stderr, check=True)
        # Once generation finishes, native probe/QA failures must not erase its work.
        asset = output / name
        staged.rename(asset)
        probe_args = [args.cli, "--json", "probe"]
        first = asset / "frame-00000.png" if args.alpha else asset
        probe_args.extend([str(first), "--image-sequence"] if args.alpha else [str(first)])
        probe = json.loads(subprocess.check_output(probe_args, text=True))
        if not probe.get("video"):
            raise ValueError("AutoCut found no video stream")
        if args.alpha and (not probe["video"]["has_alpha"] or probe["imageSequence"]["missing"]):
            raise ValueError("alpha sequence lacks alpha or has missing frames")
        expected = {**spec.get("expected", {}), **({"width": width, "height": height} if width else {}), "alpha": args.alpha}
        if args.alpha:
            expected["fps"] = args.fps
        qa_params = {**spec, "path": str(first), "expected": expected, "sheet": str(output / "contact-sheet.png")}
        if args.alpha:
            qa_params["fps"] = args.fps
        qa = command(args, "media.preflightAnimation", qa_params)
        (output / "preflight.json").write_text(json.dumps(qa, indent=2) + "\n")
        if not qa["passed"]:
            # Keep the rejected render for visual diagnosis; it is neither imported nor cached.
            raise ValueError("native animation preflight failed; inspect preflight.json and contact-sheet.png")
        if cache and not cache["hit"]:
            artifact = asset
            if args.alpha:
                artifact = work / "frames.zip"
                with zipfile.ZipFile(artifact, "w", compression=zipfile.ZIP_STORED) as archive:
                    for frame in sorted(asset.glob("frame-*.png")):
                        archive.write(frame, frame.name)
            command(args, "render.cacheCommit", {"key": cache["key"], "path": str(artifact), "manifest": cache})
    manifest = {"command": "file.importImageSequence" if args.alpha else "file.import",
                "params": {"path": str(asset / "frame-00000.png"), "fps": args.fps} if args.alpha else {"paths": [str(asset)]},
                "preflight": str(output / "preflight.json"), "sheet": str(output / "contact-sheet.png"),
                "cacheKey": cache["key"] if cache else None, "cached": bool(cache and cache["hit"]),
                "reviewOnly": args.preview, "composition": str(source), "dependencies": list(map(str, dependencies))}
    (output / "autocut-import.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    installed = Path.home() / ".raea/versions/0.1.0/bin/raea"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, help="new revision directory")
    parser.add_argument("--raea", default=str(installed) if installed.is_file() else shutil.which("raea"))
    parser.add_argument("--cli", default=shutil.which("filmcraft-cli"))
    parser.add_argument("--composition")
    parser.add_argument("--alpha", action="store_true")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--spec", type=Path, help="native QA JSON: expected duration, holds, textBounds, soundCues")
    parser.add_argument("--dependency", action="append", default=[], type=Path, help="every imported asset/module/font/sound; repeat")
    parser.add_argument("--canvas", type=dimensions, help="reference canvas dimensions (fallback output size)")
    parser.add_argument("--size", type=dimensions, help="master output dimensions")
    parser.add_argument("--preview", action="store_true", help="draft render with longest side 960")
    parser.add_argument("--cache", action="store_true", help="reuse verified native render-cache entries")
    args = parser.parse_args()
    if not args.raea or not args.cli:
        parser.error("provide --raea and --cli")
    if args.alpha and (args.fps is None or not 1 <= args.fps <= 240):
        parser.error("--alpha requires --fps in 1..240")
    if not args.alpha and args.fps is not None:
        parser.error("--fps is for alpha sequences; MP4 carries its own rate")
    try:
        args.raea = str(Path(args.raea).expanduser().resolve(strict=True))
        args.cli = str(Path(args.cli).expanduser().resolve(strict=True))
        render(args)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Render handoff failed: {error}\n")


if __name__ == "__main__":
    main()
