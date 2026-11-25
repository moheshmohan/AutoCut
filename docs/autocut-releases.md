# AutoCut releases

Newest first. Each release is tagged `autocut-v<version>` on [moheshmohan/AutoCut](https://github.com/moheshmohan/AutoCut/releases) (the `v<version>` tags in a clone that also fetches upstream are FilmCraft's own); its downloads are on the release page. How a release is cut: [releasing.md](releasing.md) (FilmCraft's process; AutoCut's macOS build is `packaging/macos/package.sh` with `AUTOCUT_APP_NAME=AutoCut`).

## 0.15.0 — 25 November 2026

Camera footage without hardware, explainers from a script, and more tools behind the generation
list: everything that was still pending after 0.14.0.

### Download

- **macOS, Apple silicon:** `autocut-0.15.0-macos-aarch64.dmg` (the app) and
  `filmcraft-cli-0.15.0-macos-aarch64.zip` (the command line and MCP server). Ad-hoc signed, not
  notarized: the first time, right-click AutoCut.app and choose **Open**.
- Windows, Linux and Intel Macs: build from source.

### What's new

**Camera footage in software**
- **10-bit and 4:2:2 H.264** decode with AutoCut's own decoder: High 10, High 4:2:2 and their
  Intra profiles (Sony XAVC S-I, Panasonic AVC-Intra, Canon XF-AVC, the 10-bit modes of
  mirrorless cameras), bit-exact with ffmpeg on 61 streams. 8-bit H.264 got about 7 % faster.
- **HEVC 4:4:4, monochrome and 12-bit** (Main 4:4:4 / 10 / 12, Main 12, 4:2:2 12, Monochrome),
  and the range-extension coding tools (transform skip up to 32×32, RDPCM, persistent Rice
  adaptation, cross-component prediction, chroma QP offsets, SAO offset scaling): bit-exact on
  142 ffmpeg streams at 1, 3 and all threads, plus x265's own reconstructions for 12-bit. A few
  rare range-extension uses that ffmpeg decodes differently from the spec are refused rather than
  decoded unverified.
- **One setting for pictures past 8K:** the H.264, ProRes, VP8, MPEG-2 and VP9 decoders check
  frame size and picture memory against limits a workflow can raise; the defaults admit every
  file that opened before. A ProRes frame over the limit now shows the unreadable-clip slate.

**Edit like a reference, and explainers**
- `media.measureStyle` measures how a video is edited (cuts and jump cuts, shot lengths, bursts,
  stillness, an energy map, loudness and palette), `reference` on Direct and Quick Short cuts like
  it, and `sequence.checkCarry` scores how each cut carries the eye.
- **Topic to explainer:** `script.timeline` turns a marked-up script into a narrated timeline
  (shots, word-timed subtitles, chapters), `graphics.chapterBar` shows the chapters,
  `script.factLedger` checks every figure and claim on screen against its sources (a new
  `factSources` check in `delivery.preflight`), the explainer rules catch long stills, small heroes
  and cuts that do not settle, and the `builtin:topic-explainer` recipe goes from script to a
  checked explainer.

**More tools behind the generation list** (none is installed or downloaded by AutoCut)
- Each tool in `providers.json` can say what it can make (`capabilities`: sizes, aspects, lengths,
  frames, references…); `generate.*` refuses what a tool cannot do before it starts and names the
  nearest allowed value.
- A tool can be a **ComfyUI workflow** on a local server (loopback only, the GPU freed after every
  job, a model-licence allow-list per workflow), checked with `generate.checkTool`.
- `generate.render3d` renders a 3D scene with your own **Blender**, headless, frames with alpha
  imported as a sequence.

**Files**
- **Chapters:** MP4, MOV and M4A exports carry the sequence's chapter markers as chapters
  (QuickTime chapter track and Nero `chpl`).
- **EXR:** files show their main picture the way compositing apps do; tiled, mip-mapped and
  multi-part files open; one-channel depth and mask files open grey. EXR sequences export from the
  float picture in half or 32-bit float (new Bit depth row, PXR24), with chromaticities; HDR
  sequences export linear BT.2020.
- **AVIF** stills import, decoded by AutoCut's AV1 decoder, alpha included (for HEIC too).
- **DNG / CinemaDNG** decode and develop 1.9–2.8× faster with identical output; lossy-JPEG DNGs
  open; DNG opcode lists can be applied (off by default).
- **TIFF / MXF:** YCbCr TIFFs with unusable colour fields open with the TIFF defaults; CinemaDNG in
  MXF reports its real depth and levels.
- **Captions:** SCC and MCC exports count from the sequence's start timecode (and import counts
  from it); `captions.export {everyFrame: true}` writes an MCC packet on every frame for SDI and
  MXF deliveries.
- **Audio:** 5.1 FLAC exports record their speaker layout. Faster with identical output: MP2
  decoding about 2.7×, AAC encoding 2.2–2.3×, the FLAC encoder about 1.7×, and AAC, Opus and AC-3
  decoding take 7–43 % less time.
- AV1 MP4 exports list the `av01` brand; Matroska dependent frames name their real reference frame.

**Fixes**
- Opus concealment is capped at 120 ms (a huge request used to allocate without limit).
- A generation tool limit under 0.1 s no longer crashes; a generation job that panics ends with an
  error instead of staying "running"; `generate.image` with the playhead frame no longer passes a
  tool one reference too many.
- The MP4 writer counts chunk offsets from the file's start and refuses a track's 2^32nd sample
  instead of overflowing.
- Interrupted media requests no longer leave their cancel, draft or catch-up hints behind.

### Known limits

- Only the macOS Apple-silicon build is published, and it isn't notarized.
- Interlaced and 4:4:4 H.264 still need a hardware decoder.
- ComfyUI and Blender were tested with stand-ins only; Cryptomatte passes and the camera-track
  format have not run on a real Blender.
- EXR PIZ, B44 and DWA compression are not read or written.
- Built without the optional Japanese interface fonts (craft-fonts).

## 0.14.0 — 25 November 2026

Faster, wider decoding in AutoCut's own pure-Rust decoders, and the follow-ups to 0.13.0.

### Download

- **macOS, Apple silicon:** `autocut-0.14.0-macos-aarch64.dmg` (the app) and
  `filmcraft-cli-0.14.0-macos-aarch64.zip` (the command line and MCP server). Ad-hoc signed, not
  notarized: the first time, right-click AutoCut.app and choose **Open**.
- Windows, Linux and Intel Macs: build from source.

### What's new

- **4:2:2 HEVC camera footage decodes without hardware.** Main 4:2:2 10 at 8 and 10 bit (Sony
  XAVC HS, Fujifilm, Canon and others) goes through AutoCut's own decoder: intra and inter
  prediction, transforms, deblocking and SAO. It matches ffmpeg sample for sample on 30 test
  streams (intra-only and P/B, deblocking and SAO on and off, weighted prediction, transform
  skip, lossless, scaling lists, odd sizes up to 3840×2160), and a hardware decoder that fails
  mid-stream now carries on in software. 4:2:0 output is unchanged.
- **HEVC decodes 4–10 % faster** on one thread: residuals are dequantised as they are read
  (−10 % cycles on high-bitrate 1080p).
- **AV1 decodes 1.6–2.1× faster with several threads**: each frame is rebuilt row by row while
  later frames start, with pictures bit-identical to before (32 streams at 1, 2, 4 and 14
  threads, all equal to libdav1d). One thread is unchanged.
- **`insta360 import --folder F` and `--until D`** take their values (the folder was read as a
  recording name).
- **Downloads carry no builder paths**: `package.sh` keeps the home folder and checkout out of
  the binaries.
- **Docs for agents**: the agent guide, the MCP server's instructions and the README describe
  0.13.0's features (FrameAtlas and the pick by meaning, craft and motion checks, keyframes,
  music fitted to the film, review).

### Known limits

- Only the macOS Apple-silicon build is published, and it isn't notarized.
- HEVC 4:4:4 and the range-extension coding tools, and H.264 10-bit / 4:2:2, still need a hardware
  decoder. 12-bit 4:2:2 HEVC is tested only on synthetic streams.
- An AV1 decoder with many threads uses more memory (about 540 MB instead of 265 MB at 1080p with
  14 threads).

## 0.13.0 — 24 November 2026

The first tagged release of AutoCut, the agent-driven, local-first video editor built on
[FilmCraft](https://github.com/storytold/filmcraft). An AI agent plans the edit; local models on
your own computer do the media work.

### Download

- **macOS, Apple silicon:** `autocut-0.13.0-macos-aarch64.dmg` (the app) and
  `filmcraft-cli-0.13.0-macos-aarch64.zip` (the command line and MCP server).
- The app is **ad-hoc signed and not notarized**. The first time, right-click AutoCut.app and
  choose **Open**, then confirm.
- Windows, Linux and Intel Macs: build from source (README, "Run it").
- No AI model is bundled. You install the ones you want (Ollama and others), and each keeps its own licence.

### What's new since the public launch (20 November 2026)

- **Motion graphics as code.** RAEA 0.1.8 renders, checks and imports compositions onto the timeline
  (`motion.check`, `motion.stills`, `motion.render`, `composition.import`).
- **EffectCraft templates** render with alpha in the app, using its GPU compositor
  (`effectcraft.templateInfo`, `effectcraft.renderTemplate`).
- **Craft checks before delivery:**
  - motion energy, text and timeline craft, copy lint;
  - Floor / Reflex / Finish tiers in `delivery.preflight`;
  - `sequence.checkMotion`, and motion filmstrips to look at.
- **Keyframes that move well:** spring, steps, wiggle and custom curves, perceptual scale, and
  `keyframes.choreograph` for staggered entrances.
- **Music fitted to the film:**
  - `media.beats` reads downbeats, sections, drops and the ending.
  - `clip.remix` lands bursts, peaks and the landing on the music, refusing joins that jump in level.
  - Direct, Quick Short, Showcase and trailers fit your track to their film.
- **Taste memory:** each video's fingerprint is recorded, open choices are dealt against the last
  few videos so consecutive ones differ, and your dislikes are retired for good
  (`autocut.fingerprint`, `autocut.deal`, `autocut.feedback`). `filmcraft-cli eval` scores it.
- **Review pointed at frames:**
  - **Comment on Frame** in the Program monitor: click or drag a box, type a note.
  - Agents read the review inbox, look at each frame, fix it and mark it done.
  - A fresh "finish reviewer" answers ship, fix, rebuild or recapture.
  - All of it over MCP (`review_*`) and `filmcraft-cli review`.
- **Generation with your own local tools:** pictures, clips, sounds, gap fills, extensions,
  restyles and augments through the generation tools you list (`generate.*`).
- **Google Photos by meaning:** `photos pick --meaning "birthday cake with candles"` picks exactly the
  items [FrameAtlas](https://github.com/moheshmohan/FrameAtlas) finds, by the day and time each
  was taken (videos; `--photos` adds photos). It needs Chrome's View ▸ Developer ▸ Allow
  JavaScript from Apple Events.
- **From upstream FilmCraft:**
  - GPU exports read back the finished 8-bit frame, which halves CPU time per frame.
  - Responsive-design pins are resolved once per layout.
  - The whole interface is in Russian.
- **Hostile-file hardening and fixes from upstream's shared libraries:**
  - Malformed or hostile files no longer crash or exhaust memory in H.264, HEVC, AV1, VP9, APV, DNx,
    AAC, FLAC, PCM, MP4/MOV, MXF, AVI, Matroska, Ogg, Targa, EXR, DNG, OMF, FCPXML/FCP7 and fonts.
  - **AAC:** surround files with a program config now play the centre on the centre speaker
    (it used to come out on the left), and HE-AAC lines up in time (it played about 20 ms early).
    LATM 7.1 decodes, and low-bit-rate multichannel AAC exports keep their sound instead of
    coding near-silence.
  - **Damaged files:** damaged FLAC and Opus keep their timing, with silence where frames were
    lost. AIFF-C in float or µ-law is no longer imported as noise, and FLAC exports of padded
    audio are smaller.
  - **Time code:** `01;00;00;30` in a drop-frame sequence goes to `01;00;01;00`, not 108 frames
    late.
  - **AAF:** clip starts count at the media's own rate, as other editors expect for mixed-rate
    media. AAFs exported by earlier AutoCut builds still read as before.
  - **Compound files (AAF):** looping chains, repeated sectors and streams over 1 GiB are refused.
  - **Captions:** fixes for TTML (styles, entities, whitespace), WebVTT, SRT, SCC, MCC and
    EBU STL.
  - **MPEG-TS:** one frame per access unit, open-GOP seeks from recovery points, the PCR from
    the PCR_PID, and damaged, scrambled or repeated packets dropped with a warning.
  - **AVI:** files with decimal stream numbers or a damaged index keep their streams, and
    bit-field RGB only decodes with the masks it understands.
  - **Ogg:** packets across a lost page are dropped, chained links that reuse a serial number
    open, and trailing Opus packets count in the duration.
  - **AAC exports** no longer lose their first 23 ms in Apple players: MP4 `roll` groups.
  - **Pictures:**
    - Targa pixels stop at the TGA 2.0 footer.
    - DJI Mavic 3 DNGs open.
    - Odd-width QuickTime raw RGB is no longer sheared.
    - HEVC with different luma and chroma bit depths keeps the right colours.
  - **Faster, bit-identical:** the bit reader (`read_ue` about 2× faster, RBSP unescaping
    about 2.8×) and Ogg opening (a 64 MiB file opens in 38 ms instead of 179 ms).

### Known limits

- Only the macOS Apple-silicon build is published, and it isn't notarized.
- AutoCut's own HEVC decoder does not decode 4:2:2 yet; that is planned for the next release.
- Built without the optional Japanese interface fonts (craft-fonts).

### Credits

AutoCut is an independent fork of FilmCraft by the ArtCraft team, and FilmCraft's name and marks
belong to its authors. Third-party code, assets and adapted documentation are credited in
[ATTRIBUTION.md](https://github.com/moheshmohan/AutoCut/blob/main/ATTRIBUTION.md).
