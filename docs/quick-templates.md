# Quick templates: a Short or Reel in one command

> **Last updated:** 2026-10-22 · **Change:** new (`autocut.quick`, the AI hints on template slots, six Shorts starters)

An album, a folder or the selected clips → a finished vertical Short or Reel. AutoCut indexes
the clips shot by shot, listens to what is said, picks a template and fills it: every slot's
shot, the text, the music, the captions, the look, the privacy blur. Everything runs locally and
everything works without the models; the report says what was used and what was not.

```sh
filmcraft-cli quick ~/Movies/Expo --topic "robots at the expo" --export ~/Movies/expo-short.mp4
filmcraft-cli quick talk1.mov talk2.mov broll/ --template "Talking-head Tip" --platform shorts --export tip.mp4
filmcraft-cli quick --project trip.fcproj --seconds 30 --look vhs --save
```

In the app: **Sequence ▸ Quick Short…**, or **Quick Short…** in the Templates panel: the source
(the Project panel's selection or a folder), the template (Auto or one of the starters), the
length, the platform and layout, the topic and the look. Over MCP: `quick_short` (a long job).
The engine command: `autocut.quick` ([`crates/engine/src/autocut.rs`](../crates/engine/src/autocut.rs)).

| option | default | meaning |
|---|---|---|
| sources / `items`, `paths` | the project's videos and photos | files, folders (their media, by name) or media items |
| `--template` | `auto` | a template's id or name, or `auto` (below) |
| `--aspect` | `9:16` | `9:16`, `1:1`, `4:5` or `16:9` |
| `--platform` | `reels` | `reels`, `shorts`, `tiktok` or `vertical`: whose interface the text and captions keep clear of |
| `--seconds` | 30 | the length wanted (a template cut on the beat gets music at the tempo that brings it nearest, 72–168 bpm) |
| `--topic` | — | what it is about: steers the template, the shots and the text |
| `--language` | detected | `en`, `ml`: the speech model and the language of the text |
| `--look` | the template's | `tech`, `vhs` or `clean` |
| `--music` | `generate` | a music file (the cuts follow its beats) or a generated bed |
| `--redact` | `faces,plates` | `faces`, `plates` or `off`; the speaker's own face is never hidden |
| `--export` | — | the video, plus `<out>.sheet.jpg`, `<out>.report.json`, `<out>.qa.json` |
| `--download` | off | fetch the missing scene models (fast and full), the speech models and the face and plate detectors first; nothing is ever downloaded otherwise |

## What happens

1. **Index** — `media.analyzeScenes` (fast) for every clip: cuts, sharpness, exposure, camera
   move, an aesthetic score and a Qwen3-VL 2B description of each shot. Then the items holding
   the most promising shots (12 shots at most, `fullDetail`) are described again with the 4B
   model, in full detail, so the richer descriptions are there where they matter.
2. **Speech** — `media.understandSpeech` for the clips with sound: the words, the takes to keep,
   whether it is someone talking to the camera.
3. **Template** — with `auto`, every template that has the layout is scored: the AI-ready
   starters first; enough shots for its places; someone talking to the camera when it wants one
   (and not otherwise); the shots' energy against its slots'; its length (with the tempo moved)
   against `seconds`; the topic and the subjects against its tags. The report gives the scores
   and the reasons, with the runners-up.
4. **Shots** — each picture slot's media gets the moment that suits its AI hints: the scene
   index's score for the slot's purpose (the rules of `media.bestShots`), its words
   (`media.searchScenes`), energy, speech, the least quality and the shot's length. No moment
   is used twice (a long shot gives several), a clip gives few picks, slots that ask for it keep
   the filming order (capture dates), and a talking slot takes stretches of kept speech.
5. **Trim and frame** — each pick plays from its best moment (the keyframe, or the start of the
   kept speech), framed on the subject's box for the vertical crop.
6. **Text** — the text slots are written by the local model (Ollama: the script model, else the
   classifier) from the shots' descriptions (the full ones' paragraphs, the text read on
   screen), the kept speech and the topic, each within its length; without a model, from the
   descriptions (the subject, the text read) and the topic.
7. **Apply and finish** — `template.apply`; placeholders of optional slots nothing suited are
   removed; the look; Bold animated captions of the kept speech in the platform's safe zone;
   faces and plates blurred (`clip.autoRedact`, the robot check by the local vision model); the
   safe zones checked; the export, a 12-frame contact sheet with times, the report and QA.

## AI hints on template slots

Any template can say what its slots want (`"ai"` on a slot; see
[`crates/templates/README.md`](../crates/templates/README.md#ai-hints)): a picture slot's
`purpose` (hook, broll, reveal, talking-head, establishing, outro), `query`, `energy`, `speech`
(want / avoid), `minQuality` and `chronological`; a text slot's `prompt`, `maxChars` and `about`
(the picture slot its texts describe). `template.inspect` shows them, Save as Template keeps
them. A template without hints is still filled, by the pictures' quality.

The six starters made for it (Templates ▸ Reels & Shorts, Tech, Events, Social, Vlog):

| starter | for |
|---|---|
| Tech Reveal | a hook, eight b-roll shots on the beat, the reveal and its name, an end card; tech look |
| Event Recap Short | the place with the event's name, ten moments on the beat, a closing card |
| Talking-head Tip | three kept takes of someone talking, b-roll cutaways over the joins, word captions |
| Gadget Showcase | a close-up hook, three features with spec cards, the hero shot, a question |
| Before / After Short | the earlier shot, a swipe to the later one, both stacked |
| Day in 30s | twelve moments in the order they were filmed, on tape |

## Without the models

| missing | what happens instead |
|---|---|
| scene models | cuts from the picture's changes, shots ranked by their quality alone |
| the 4B describer | the fast descriptions everywhere |
| Whisper / speech | no talking-head template, no captions |
| Ollama | the text comes from the descriptions and the topic |
| face and plate detectors | nothing is blurred (the report says so) |

Each is listed in the report's `fallbacks`. The scene index and speech understanding take the
GPU lock themselves while their models run (`filmcraft-cli locks`); Ollama queues its own work.
