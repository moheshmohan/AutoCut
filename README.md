# AutoCut projects

The working files of videos made with AutoCut, kept apart from the product code on `main`.
No media lives here: the clips are in Google Photos (and AutoCut's cache), the exports and
`.fcproj` projects in `~/Movies/AutoCut/`.

## every-robot — "Every Robot I've Met" (H4hacks)

Final: Malayalam narration in the cloned voice, 4K, speech-first mix, uploaded private to
H4hacks on 2026-10-21 (https://www.youtube.com/watch?v=7FYPOxDYWBc), Malayalam subtitles added
in YouTube Studio.

- `v5/` — the inputs of the final:
  - `plan-v5.json`, `plan-v5-ml.json`: the showcase plans (English, Malayalam);
  - `build_plan_v5.py`: wrote them from the v4 plan and the v5 review notes
    (`filmcraft-cli showcase-plan` now does this from a notes file);
  - `script-ml-v6.json`: the approved natural Malayalam lines (`malayalam-script-v6.md` reads them
    side by side with the English);
  - `album-locations.txt`, `pick-v5.json`: where each clip was shot, and the album pick;
  - `ref-transcript.json`: the voice reference's transcript (from Gemini);
  - `youtube-final.json`: title, description, tags;
  - `every-robot-ml-2.srt`: the subtitles uploaded to Studio;
  - `build-final2-ml.log`: the final build's log.
- `final/` — the final's chapters, subtitles, QA and narration-check reports.
- `history/` — the v3 and v4 plans and the researched robot facts.

Rebuild or revise: `docs/showcase-runbook.md` on `main` (`showcase-plan` with a notes file, then
`showcase --languages ml --resolution 4k`).
