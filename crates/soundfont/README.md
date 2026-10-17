# filmcraft-soundfont

SoundFont instruments for FilmCraft's procedural soundtrack engine (`filmcraft_audio_dsp::music`).
Layer L2. Roadmap item D1, second milestone: real instruments, with the built-in oscillator voices
as the fallback.

- `check`: a structural check of a SoundFont 2 file, run before rustysynth reads a byte (below).
- `SoundFont`: a checked, loaded file (at most 1 GiB; GeneralUser GS loads in about 10 ms).
- `SoundFontBank` / `FontPlayer` / `generate`: the music engine's `InstrumentBank` on a font. Every
  patch plays its General MIDI program (drum kits on bank 128: Power for cinematic, Electronic,
  Standard for pop, Jazz for lo-fi, Brush for soft; the font's standard kit when it lacks one).
  Risers and impacts, patches the font has no preset for, presets that stay silent and sample rates
  outside 16–192 kHz play on the oscillators. Each font instrument is level-matched to the
  oscillator voice it replaces (a two-second phrase through both, integrated loudness), so the mix
  keeps the balance the oscillator voices were tuned to; pan and reverb send come from the same voice.
  Drums ring out (released 1.5 s, cymbals 4 s, after their last hit). The arrangement, chords and
  beat map never depend on the bank.
- `catalog`: the GeneralUser GS download pin (below). Never bundled, never committed.
- `synthetic`: a tiny SoundFont made in code (two samples, every program the engine uses, a drum
  kit) and a "runaway" font that passes the check but fails inside rustysynth, for tests.

Engine commands (`crates/engine/src/soundfonts.rs`): `music.soundfonts`, `music.downloadSoundfont`
(`consent: true` only), `music.useSoundfont` (`auto` | `oscillators` | `generaluser-gs` | a path,
kept in Preferences `music.soundfont`); `audio.generate` and `music.preview` take `soundfont` per call.

## rustysynth (MIT) and why the check exists

[rustysynth](https://github.com/sinshu/rustysynth) 1.3.7, MIT, by Nobuaki Tanaka: a pure-Rust
SoundFont synthesizer, no dependencies. Pinned exactly (`=1.3.7`) because the check below mirrors how
this version reads a file. Reviewed 2026-10-17:

- **`unsafe`, two places.** `BinaryReader::read_wave_data` allocates `size / 2` samples and then
  views them as `size` bytes: for an **odd** sample-chunk size that slice is one byte longer than the
  buffer and `read_exact` writes past it (heap overflow, undefined behaviour). The check refuses odd
  chunk sizes, so rustysynth never sees one. `SoundFontSampleData::new` reads the first 4 bytes of a
  buffer it has just checked holds at least two samples: sound.
- **Allocation by declared size.** Texts, discarded chunks and the sample data are allocated by the
  size the file declares before it is read (up to 4 GB): the check requires every chunk to lie inside
  its list and every list inside the file, and the file is at most 1 GiB.
- **Panics on hostile indices.** Zone lists are sliced and generator lists indexed with the file's
  numbers (`zones[start..end]`, `generators[index + i]`); sample offsets are added in `i32`. The check
  requires preset and instrument zone ranges to be non-empty, increasing and inside the bag lists, bag
  generator indices to be increasing and inside the generator lists, and every sample's points to lie
  inside the sample data (so offset sums cannot overflow). Sample and instrument ids in generators are
  checked by rustysynth itself (an error, not a panic).
- **Walk.** rustysynth reads `INFO`, `sdta`, `pdta` in that order by declared sizes, reads `ifil` /
  `iver` as 4 bytes whatever their size, and skips no RIFF pad bytes. The check walks the same way
  and requires the sub-chunks to tile each list exactly, `ifil`/`iver` to be 4 bytes and every size
  to be even (the SoundFont 2 rule), so both see the same chunks.
- **While playing.** A font that passes the check can still drive rustysynth out of bounds (a tiny
  loop played far above its root: `synthetic::runaway`). Every call into it runs under
  `catch_unwind`; a failure silences that instrument and is counted, and `generate` then makes the
  whole track again on the oscillators and says so.

Tests (`src/tests.rs`, `src/bank.rs`): every truncation of the tiny font, 6000 random multi-bit
flips (half of them in the structure), every 16/32-bit field set to 0, 1, 3, 0x7FFF…, 0xFFFF…, all
under `catch_unwind` and asserting rustysynth itself never panicked on a file the check let through;
an odd sample chunk refused; the oscillator fallback (bit-identical audio when no preset matches, and
below 16 kHz); the same seed giving the same plan and beat map with and without a font, stems summing
to the mix, the same render twice; a failure inside rustysynth silencing the part and the track
falling back; level matching within 0.5 LU.

## GeneralUser GS (downloaded, never bundled)

GeneralUser GS v2.0.3 by S. Christian Collins: 261 instrument presets and 13 drum kits in 32 MB.
Pinned to commit `97049183643d5fc5a9322a69c5b09efb667c6c3a` of the author's repository
`github.com/mrbumpy409/GeneralUser-GS` ("an easier source for automated packaging of GeneralUser GS"):

| File | Size | SHA-256 |
|---|---|---|
| `GeneralUser-GS.sf2` | 32 319 396 | `9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe` |
| `documentation/LICENSE.txt` → `LICENSE.txt` | 2 317 | `7b32efefdf95ce38a043799f0659853ddc00fbaa14d8c50f0aca16b9b8b405be` |

Checked 2026-10-17: the download's git blob id matches the repository's (`298b552d…`), and it passes
the check and loads (287 presets including the GS variations; its `INAM` reads "GeneralUser GS 2.0.3
BETA").

**Licence: GeneralUser GS License v2.0.** Use without restriction for music creation, private or
commercial; free to use in software projects and to modify. The author notes that some samples came
from other freely available SoundFonts and that he cannot be 100% sure where all of them originated
(none, as far as he knows, from commercial packages or sample CDs). The engine shows the size, source,
the licence in full and that note before the user agrees (`music.soundfonts` → `licenseText`,
`provenance`; the Generate Audio panel's consent box), downloads only with `consent: true` into
`<data dir>/models/generaluser-gs/`, checks both files' SHA-256 and loads the font once before
keeping it. The author asks websites not to link his download files directly; the app downloads from
his repository, which he keeps for automated packaging.

## Listening

`cargo run --release -p filmcraft-soundfont --example render_samples -- <font.sf2> <out dir>` writes
two pieces (epic cinematic, calm lo-fi) on the font and on the oscillators, and prints each stem's
level on both.
