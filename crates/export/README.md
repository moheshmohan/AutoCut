# filmcraft-export

Native Rust sequence rendering, encoding and muxing. Formats, parameters and
presets are described in [architecture](../../docs/architecture.md) and
[agents](../../docs/agents.md). AutoCut's render overlap, decode-ahead pipeline,
GPU compositing and explicit `maxThreads` cap remain available.

## Upstream ports, AV17

- MP4 video exports use the existing muxer's fast-start finalization: metadata
  precedes media data, with sample offsets updated using a bounded copy buffer.
  MOV and audio-only exports retain their existing layout. Disk MP4 output needs
  read/write access, including a caller-supplied `output_file`; I/O failures are
  returned as export errors. Memory sinks remain supported.
- A failed encoder factory lets later factories try the request. If none can
  encode it, the first useful error is returned.
- A nested explicit `settings.bitrateKbps` overrides an adaptive preset unless
  `adaptiveBitrate` is also supplied. Unknown settings still fail clearly.
- TikTok, Instagram Reels and YouTube Shorts presets use 1080×1920, scale to
  fill, AAC at 48 kHz, −14 LUFS and a −1 dBTP ceiling. AutoCut retains automatic
  hardware encoding for these web delivery presets and keeps the existing
  Social Vertical preset.

Regression coverage includes MP4 box order, decoded video/audio for disk,
reserved-file and memory outputs, I/O errors, encoder fallback and preset
settings. See [the integration record](../../plan/upstream-sync-2026-11-08.md)
for validation results and deliberately deferred changes.
