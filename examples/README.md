# AutoCut examples

The working files of real videos made with AutoCut, kept as worked examples for
[docs/video-pipeline.md](../docs/video-pipeline.md). No media lives here: the clips are in Google
Photos (and AutoCut's cache), the exports and `.fcproj` projects in `~/Movies/AutoCut/`.

## raea — local motion graphics

`raea/motion.tsx` defines a transparent split wipe and an animated chapter scene.
`raea/render.py` renders them and prepares their AutoCut import commands. See the
[local RAEA workflow](../docs/raea-workflow.md).

## generation — local generation tools

`generation/providers.example.json` describes three MLX tools (pictures, clips, sound) to the
`generate.*` commands; `generation/ltx_video.py` is the small runner the clip tool uses,
`generation/ltx_extend.py` the one that carries a shot on, `generation/ltx_restyle.py` the one
that makes a clip again through an adapter of the model, and `generation/cutout.py` removes a
picture's background. See
[Generation in the editor](../docs/generation.md).
