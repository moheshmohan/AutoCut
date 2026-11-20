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

## every-robot — "Every Robot I've Met" (H4hacks)

Final: Malayalam narration in the cloned voice, 4K, speech-first mix, on YouTube
(https://www.youtube.com/watch?v=7FYPOxDYWBc) with Malayalam subtitles; a 34-second Instagram Reel
trailer.

- `v5/` — the inputs of the final:
  - `report-robots.json`: the clip analysis (`analyze robot`) the builds read;
  - `plan-v5.json`, `plan-v5-ml.json`: the showcase plans (English, Malayalam);
  - `build_plan_v5.py`: wrote them from the v4 plan and the v5 review notes
    (`filmcraft-cli showcase-plan` now does this from a notes file);
  - `script-ml-v6.json`: the approved natural Malayalam lines (`malayalam-script-v6.md` reads them
    side by side with the English);
  - `album-locations.txt`, `pick-v5.json`: where each clip was shot, and the album pick;
  - `ref-transcript.json`: the voice reference's transcript (from Gemini);
  - `youtube-final.json`: title, description, tags;
  - `every-robot-ml-2.srt`: the subtitles uploaded to Studio;
  - `build-final2-ml.json`, `build-final2-ml.log`: the final build's report (the trailer reads it)
    and log.
- `final/` — the final's chapters, subtitles, QA and narration-check reports.
- `reel/` — the trailer's Instagram caption and QA.
- `history/` — the v3 and v4 plans and the researched robot facts.
