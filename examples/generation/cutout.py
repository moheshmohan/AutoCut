#!/usr/bin/env python3
"""A cut-out step for AutoCut's picture tools: removes a picture's background (rembg, U2-Net).

AutoCut runs this after a picture tool that cannot make a transparent picture itself
(`cutoutCommand` in the provider list, see docs/generation.md), with the Python of an
environment that has `rembg` and `Pillow`. Nothing here is part of the app.

usage: cutout.py IN.png OUT.png
"""
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    from PIL import Image
    from rembg import new_session, remove

    with Image.open(src) as im:
        cut = remove(im.convert("RGBA"), session=new_session("u2net"))
    cut.save(dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
