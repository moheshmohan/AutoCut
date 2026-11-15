//! Generated audio: sound effects and music synthesized in the app
//! ([`filmcraft_audio_dsp::synth`] and the soundtrack engine [`filmcraft_audio_dsp::music`] —
//! oscillators, noise, envelopes, filters; no models; music also on a SoundFont when one is
//! chosen), written as WAV files, imported and placed.
//!
//! | command | does |
//! |---|---|
//! | `audio.generate` | synthesize a whoosh, impact, riser, drone or music `duration` seconds (or exactly `durationTicks`) long, write `<Kind> <n>.wav` (48 kHz stereo float), import it and place it at `time`/`seconds` on a free audio track — one undo step |
//! | `audio.generateKinds` | the kinds, moods, genres, stems, defaults and limits |
//!
//! **Music** is arranged to the exact length (to the sample) from section blocks and ends on a
//! bar: the final chord lands on a downbeat and rings out to the last sample, nothing padded or
//! faded to fit. `mood` (or a synonym: "suspense", "chill"…), `genre`, `bpm` (default: the
//! style's), `key`, `energy` (0..1, or a curve of up to 64 levels over the track; `intensity`
//! is the old name of a level) shape it. The result carries the track's taxonomy (mood → genre
//! → tempo → energy → use case), its sections, chords and exact beat map in ticks from its start
//! (`beats`, `downbeats`, `ending`), and so does the imported item's metadata (Description,
//! Music Mood, Music Genre, Tempo, Key, Music Energy, Use Case, Beat Map). `stems: true` also
//! writes the drums, bass, melody and pad stems (`<Kind> <n> <stem>.wav`, up to 5 minutes; they
//! sum to the mix) and places them, each on its own track, instead of the mix, so they can be
//! muted or rebalanced; `levels` ({stem: dB | "mute"}) and `mute` ([stem]) rebalance the mix.
//!
//! **Instruments.** Music plays on a SoundFont when one is chosen — `soundfont` (`auto`,
//! `oscillators`, `generaluser-gs` or an `.sf2` path), else Preferences ▸ `music.soundfont`
//! (default `auto`: GeneralUser GS once downloaded, see [`crate::soundfonts`]) — and on the
//! built-in oscillator voices otherwise; the arrangement and beat map are the same either way.
//! The result's `instruments` and the item's "Music Instruments" say what played it, `soundfont`
//! describes the font, `warning` says when a font failed and the oscillators stood in. When
//! `music.preview` just played exactly this request, its audio is placed without rendering again.
//!
//! **Where it goes.** At `time` (default: the playhead) on `track` if given (it must be free
//! there), else on the first audio track free for the cue's length, else on a new audio track.
//! `place: false` only imports. Files go to `dir`, else Scratch Disks ▸ Captured, else a
//! `Generated Audio` folder next to the project (or in the data or temporary dir).
//!
//! **Reproducible.** The same parameters and `seed` always give the same audio; without a
//! seed, each new cue of a kind gets the next one, so two whooshes differ.

use serde_json::{Value, json};

use filmcraft_audio_dsp::music;
use filmcraft_audio_dsp::synth;
use filmcraft_project::{ClipId, ItemId, TrackId, TrackKind};
use filmcraft_time::{Tick, TimeRange};

use crate::commands::{CommandSpec, always, bad, bool_p, f64_p, str_p, time_p, u64_p};
use crate::{EngineError, Result, Session};

/// Sample rate of generated audio.
pub const RATE: u32 = 48_000;
/// Metadata key marking an item as a stem of generated music (its value: the stem).
pub const STEM_KEY: &str = "Music Stem";
/// Metadata key of what played generated music ("oscillators", or a SoundFont's name).
pub const INSTRUMENTS_KEY: &str = "Music Instruments";

/// A rendered track: just made, or shared with the preview that made it.
enum Rendered {
    Own(Box<music::Track>),
    Shared(std::sync::Arc<crate::music_preview::Previewed>),
}

impl Rendered {
    fn track(&self) -> &music::Track {
        match self {
            Rendered::Own(t) => t,
            Rendered::Shared(p) => &p.track,
        }
    }
}

fn spec(
    id: &'static str,
    label: &'static str,
    params: &'static str,
    enabled: fn(&Session) -> std::result::Result<(), String>,
    run: fn(&mut Session, &Value) -> Result<Value>,
    journal: bool,
) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled, run, journal }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            "audio.generate",
            "Generate Audio",
            r#"{"kind":"whoosh|impact|riser|swish|chime|shutter|pop|thump|drone|engineRev|engineLaunch|engineFlyby|music"=music,"duration":seconds?|"durationTicks":ticks?,"seed":u64?,"intensity":0..1=0.7,"bpm":40..200?,"key":"A|C#|Eb…"="A","mood":"epic|tense|uplifting|calm|happy|dark|dreamy|energetic|sad"="epic","genre":"cinematic|electronic|pop|ambient|lofi|piano"?,"energy":0..1|[0..1 ×1..64]?,"stems":bool=false,"levels":{"drums|bass|melody|pad":dB|"mute"}?,"mute":["drums"…]?,"soundfont":"auto|oscillators|generaluser-gs"|path?,"time":ticks?|"seconds":f64?,"track":"A2"|id?,"place":bool=true,"dir":str?}"#,
            always,
            generate,
            true,
        ),
        spec("audio.generateKinds", "Generated Audio Kinds", "{}", always, kinds, false),
    ]
}

fn label(kind: synth::Kind) -> &'static str {
    match kind {
        synth::Kind::Whoosh => "Whoosh",
        synth::Kind::Impact => "Impact",
        synth::Kind::Riser => "Riser",
        synth::Kind::Swish => "Swish",
        synth::Kind::Chime => "Chime",
        synth::Kind::Shutter => "Shutter",
        synth::Kind::Pop => "Pop",
        synth::Kind::Thump => "Thump",
        synth::Kind::Drone => "Drone",
        synth::Kind::EngineRev => "Engine Rev",
        synth::Kind::EngineLaunch => "Engine Launch",
        synth::Kind::EngineFlyby => "Engine Fly-by",
        synth::Kind::Music => "Music",
    }
}

fn kinds(_s: &mut Session, _p: &Value) -> Result<Value> {
    let ks: Vec<Value> = synth::Kind::ALL
        .iter()
        .map(|k| json!({"id": k.id(), "name": label(*k), "defaultSeconds": k.default_seconds(), "maxSeconds": if matches!(k, synth::Kind::Music | synth::Kind::Drone) { synth::MAX_MUSIC_SECONDS } else { synth::MAX_SFX_SECONDS }}))
        .collect();
    let moods: Vec<&str> = synth::Mood::ALL.iter().map(|m| m.id()).collect();
    let genres: Vec<&str> = music::Genre::ALL.iter().map(|g| g.id()).collect();
    let stems: Vec<&str> = music::Stem::ALL.iter().map(|s| s.id()).collect();
    Ok(
        json!({"kinds": ks, "moods": moods, "genres": genres, "stems": stems, "maxStemsSeconds": music::MAX_STEMS_SECONDS, "sampleRate": RATE, "musicLufs": synth::MUSIC_LUFS}),
    )
}

/// The RIFF `INFO` tags a generated track carries (title, genre, comment, keywords, software).
pub type Tags = Vec<([u8; 4], String)>;

/// Interleaved stereo 32-bit float WAV.
pub fn write_wav_f32_stereo(left: &[f32], right: &[f32], rate: u32) -> Vec<u8> {
    write_wav_f32_stereo_tagged(left, right, rate, &[])
}

/// Interleaved stereo 32-bit float WAV, with a `LIST`/`INFO` chunk of `tags` after the samples
/// (readers that do not know it skip it).
pub fn write_wav_f32_stereo_tagged(left: &[f32], right: &[f32], rate: u32, tags: &[([u8; 4], String)]) -> Vec<u8> {
    let frames = left.len().min(right.len());
    let data_len = frames.saturating_mul(8);
    let mut info: Vec<u8> = Vec::new();
    for (id, text) in tags {
        // a NUL-terminated string, padded to an even length
        let mut body: Vec<u8> = text.bytes().filter(|b| *b != 0).take(4096).collect();
        body.push(0);
        info.extend_from_slice(id);
        info.extend_from_slice(&(u32::try_from(body.len()).unwrap_or(u32::MAX)).to_le_bytes());
        info.extend_from_slice(&body);
        if body.len() % 2 == 1 {
            info.push(0);
        }
    }
    let list_len = if info.is_empty() { 0 } else { 12usize.saturating_add(info.len()) };
    let mut v = Vec::with_capacity(44usize.saturating_add(data_len).saturating_add(list_len));
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(u32::try_from(36usize.saturating_add(data_len).saturating_add(list_len)).unwrap_or(u32::MAX)).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    v.extend_from_slice(&2u16.to_le_bytes()); // channels
    v.extend_from_slice(&rate.to_le_bytes());
    v.extend_from_slice(&rate.saturating_mul(8).to_le_bytes());
    v.extend_from_slice(&8u16.to_le_bytes());
    v.extend_from_slice(&32u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&(u32::try_from(data_len).unwrap_or(u32::MAX)).to_le_bytes());
    for (l, r) in left.iter().zip(right.iter()) {
        v.extend_from_slice(&l.to_le_bytes());
        v.extend_from_slice(&r.to_le_bytes());
    }
    if !info.is_empty() {
        v.extend_from_slice(b"LIST");
        v.extend_from_slice(&(u32::try_from(4usize.saturating_add(info.len())).unwrap_or(u32::MAX)).to_le_bytes());
        v.extend_from_slice(b"INFO");
        v.extend_from_slice(&info);
    }
    v
}

/// The kind a call asks for.
fn kind_p(p: &Value) -> Result<synth::Kind> {
    match str_p(p, "kind") {
        Some(k) => synth::Kind::from_id(k).ok_or_else(|| {
            let kinds: Vec<&str> = synth::Kind::ALL.iter().map(|k| k.id()).collect();
            bad("audio.generate", format!("unknown kind `{k}` ({})", kinds.join(", ")))
        }),
        None => Ok(synth::Kind::Music),
    }
}

/// The next seed for a new cue of `kind`: one more than the cues of that kind in the project
/// (stems of generated music do not count).
fn next_seed(s: &Session, kind: synth::Kind) -> u64 {
    let prefix = format!("{} ", label(kind));
    1 + s.project.items.values().filter(|i| i.name.starts_with(&prefix) && !i.metadata.contains_key(STEM_KEY)).count() as u64
}

/// The key a call asks for.
fn key_p(p: &Value) -> Result<Option<i32>> {
    match str_p(p, "key") {
        Some(k) => synth::key_from_name(k).map(Some).ok_or_else(|| bad("audio.generate", format!("unknown key `{k}` (A, A#/Bb, B, C, … G#)"))),
        None => Ok(None),
    }
}

/// The synth parameters a call asks for (validated; out-of-range numbers are clamped later).
fn params(s: &Session, p: &Value, kind: synth::Kind) -> Result<synth::Params> {
    let cmd = "audio.generate";
    let mut q = synth::Params::new(kind);
    // `duration` is the cue's length; `seconds` is where it goes, as in every other command
    if let Some(t) = crate::music::length_p(p, cmd, kind.max_seconds())? {
        q.seconds = t.seconds();
    }
    if let Some(i) = f64_p(p, "intensity") {
        q.intensity = i;
    }
    if let Some(b) = f64_p(p, "bpm") {
        q.bpm = b;
    }
    if let Some(k) = key_p(p)? {
        q.key = k;
    }
    if let Some(m) = crate::music::mood_p(p, cmd)? {
        q.mood = m;
    }
    q.seed = u64_p(p, "seed").unwrap_or_else(|| next_seed(s, kind));
    Ok(q)
}

/// The music request a call (`cmd`: `audio.generate`, `music.preview`) asks for.
pub(crate) fn music_request(s: &Session, p: &Value, cmd: &str) -> Result<music::Request> {
    let mood = crate::music::mood_p(p, cmd)?.unwrap_or(music::Mood::Epic);
    let length = crate::music::length_p(p, cmd, synth::MAX_MUSIC_SECONDS)?.unwrap_or_else(|| Tick::from_seconds_f64(synth::Kind::Music.default_seconds()));
    if length.seconds() < music::MIN_SECONDS {
        return Err(bad(cmd, format!("music must be at least {} s long", music::MIN_SECONDS)));
    }
    let energy = match crate::music::energy_p(p, cmd)? {
        Some(e) => Some(e),
        // the old name of a level: clamped, as it always was
        None => f64_p(p, "intensity").map(|i| music::Energy::Level(if i.is_finite() { i.clamp(0.0, 1.0) } else { 0.7 })),
    };
    let stems = match p.get("stems") {
        None | Some(Value::Null) => false,
        Some(v) => v.as_bool().ok_or_else(|| bad(cmd, format!("`stems` must be true or false, not {v}")))?,
    };
    Ok(music::Request {
        mood,
        genre: crate::music::genre_p(p, cmd)?,
        length,
        seed: u64_p(p, "seed").unwrap_or_else(|| next_seed(s, synth::Kind::Music)),
        bpm: f64_p(p, "bpm"),
        key: key_p(p)?,
        energy,
        stems,
        levels: crate::music::levels_p(p, cmd)?,
    })
}

fn track_is_free(s: &Session, track: TrackId, range: TimeRange) -> bool {
    s.active_sequence()
        .and_then(|q| q.audio_tracks.iter().find(|t| t.id == track))
        .is_some_and(|t| !t.locked && !t.items.iter().any(|it| it.start < range.end() && range.start < it.end()))
}

/// The audio track `p` names (`track`), checked free over `range`; `None` when it names none.
fn explicit_track(s: &Session, p: &Value, range: TimeRange) -> Result<Option<TrackId>> {
    let cmd = "audio.generate";
    let Some(v) = p.get("track") else { return Ok(None) };
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let id =
        crate::mixer::strip_ref(seq, v).filter(|id| seq.audio_tracks.iter().any(|t| t.id == *id)).ok_or_else(|| bad(cmd, format!("no audio track {v}")))?;
    if !track_is_free(s, id, range) {
        return Err(bad(cmd, format!("audio track {v} is locked or has clips there")));
    }
    Ok(Some(id))
}

/// Place `item` (`dur` long) at `at` on `track`, else the first free audio track, else a new one.
fn place_item(s: &mut Session, item: ItemId, at: Tick, dur: Tick, track: Option<TrackId>) -> Result<(ClipId, TrackId)> {
    let cmd = "audio.generate";
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let range = TimeRange::new(at, dur);
    let free = track
        .filter(|t| track_is_free(s, *t, range))
        .or_else(|| s.active_sequence().and_then(|q| q.audio_tracks.iter().map(|t| t.id).find(|id| track_is_free(s, *id, range))));
    let track = match free {
        Some(t) => t,
        None => {
            let before: Vec<TrackId> = s.active_sequence().map(|q| q.audio_tracks.iter().map(|t| t.id).collect()).unwrap_or_default();
            s.execute("sequence.addTracks", json!({"video": 0, "audio": 1, "audioAfter": "last"}))?;
            s.active_sequence()
                .and_then(|q| q.audio_tracks.iter().map(|t| t.id).find(|id| !before.contains(id)))
                .ok_or_else(|| EngineError::Other("could not add an audio track".into()))?
        }
    };
    let seq_rate = s.project.sequence(seq_id).map(|q| q.settings.frame_rate).unwrap_or_default();
    let clip: ClipId = s.edit("Generate Audio", |pr, st| {
        let mut it = pr
            .make_track_item(item, TrackKind::Audio, at, TimeRange::new(Tick::ZERO, dur), seq_rate)
            .ok_or(EngineError::Other("the generated audio is not importable".into()))?;
        it.duration = dur;
        let id = it.id;
        let seq = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        let t = seq.audio_tracks.iter_mut().find(|t| t.id == track).ok_or_else(|| bad(cmd, "the audio track was deleted"))?;
        let pos = t.items.partition_point(|x| x.start <= at);
        t.items.insert(pos, it);
        seq.check().map_err(EngineError::Other)?;
        st.selection = vec![id];
        Ok(id)
    })?;
    Ok((clip, track))
}

/// Write `bytes` to `path` and import it, with `meta` as the item's metadata.
fn write_and_import(s: &mut Session, path: &str, bytes: Vec<u8>, meta: &[(String, String)]) -> Result<ItemId> {
    s.services.write_file(path, &bytes).map_err(|e| EngineError::Other(format!("{path}: {e}")))?;
    let item: ItemId = crate::commands::import_bytes(s, path, bytes.into(), None)?;
    if !meta.is_empty() {
        s.edit("Generate Audio", |pr, _| {
            let it = pr.item_mut(item).ok_or_else(|| EngineError::Other("the generated audio was not imported".into()))?;
            it.metadata.extend(meta.iter().cloned());
            Ok(())
        })?;
    }
    Ok(item)
}

/// Undo whatever a failed generation left, and collapse a successful one into one step.
fn finish(s: &mut Session, n0: usize, result: Result<Value>) -> Result<Value> {
    crate::clip_ops::collapse_history(s, n0, "Generate Audio");
    if result.is_err() && s.history.undo.len() > n0 {
        let _ = s.undo();
        s.history.redo.clear();
    }
    result
}

fn generate(s: &mut Session, p: &Value) -> Result<Value> {
    let kind = kind_p(p)?;
    if kind == synth::Kind::Music {
        return generate_music(s, p);
    }
    let cmd = "audio.generate";
    if bool_p(p, "stems") == Some(true) {
        return Err(bad(cmd, "stems are made for music only"));
    }
    let q = params(s, p, kind)?;
    let place = bool_p(p, "place").unwrap_or(true) && s.state.active_sequence.is_some();
    let at = time_p(s, p, "").unwrap_or(s.playhead());
    if at < Tick::ZERO {
        return Err(bad(cmd, "the start time cannot be negative"));
    }
    let audio = synth::generate(&q, RATE).map_err(|e| bad(cmd, e.to_string()))?;
    let dur = Tick::from_units(audio.left.len() as i64, i64::from(RATE));
    let range = TimeRange::new(at, dur);
    // the track first, so a busy `track` fails before anything is written
    let chosen = if place { explicit_track(s, p, range)? } else { None };
    let base = label(q.kind);
    let n0 = s.history.undo.len();
    let result = (|| -> Result<Value> {
        let dir = crate::voiceover::generated_dir(s, p, "Generated Audio")?;
        let path = crate::voiceover::unique_wav_path(s, &dir, base)?;
        if !cfg!(target_arch = "wasm32") {
            std::fs::create_dir_all(&dir).map_err(|e| EngineError::Other(format!("{dir}: {e}")))?;
        }
        let item = write_and_import(s, &path, write_wav_f32_stereo(&audio.left, &audio.right, RATE), &[])?;
        let mut out = json!({
            "item": item.0, "path": path, "kind": q.kind.id(), "seconds": audio.seconds(), "seed": q.seed,
            "sampleRate": RATE, "clip": Value::Null, "track": Value::Null,
        });
        if place {
            let (clip, track) = place_item(s, item, at, dur, chosen)?;
            let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
            out["clip"] = json!(clip.0);
            out["track"] = json!(crate::mixer::strip_label(seq, track));
            out["start"] = json!(at.0);
            out["duration"] = json!(dur.0);
        }
        Ok(out)
    })();
    finish(s, n0, result)
}

/// The RIFF tags of a generated track titled `title`.
fn tags(title: &str, plan: &music::Plan) -> Tags {
    let description = crate::music::metadata(plan).into_iter().find(|(k, _)| k == "Description").map(|(_, v)| v).unwrap_or_default();
    vec![
        (*b"INAM", title.to_string()),
        (*b"IGNR", plan.genre.name().to_string()),
        (*b"ICMT", description),
        (*b"IKEY", plan.style.use_cases.join("; ")),
        (*b"ISFT", "FilmCraft procedural soundtrack".to_string()),
    ]
}

fn generate_music(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "audio.generate";
    let req = music_request(s, p, cmd)?;
    let place = bool_p(p, "place").unwrap_or(true) && s.state.active_sequence.is_some();
    let at = time_p(s, p, "").unwrap_or(s.playhead());
    if at < Tick::ZERO {
        return Err(bad(cmd, "the start time cannot be negative"));
    }
    // refuse a busy `track` (and anything the engine refuses) before rendering or writing
    let plan = music::plan(&req, RATE).map_err(|e| bad(cmd, e.to_string()))?;
    let dur = Tick::from_units(i64::try_from(plan.samples).unwrap_or(i64::MAX), i64::from(RATE));
    let chosen = if place { explicit_track(s, p, TimeRange::new(at, dur))? } else { None };
    let (inst, mut warning) = crate::soundfonts::resolve(s, p, cmd)?;
    // what `music.preview` just played, when it is exactly this; else render it now
    let rendered;
    let track: &music::Track = match s.music_preview.matching(&req, &inst.key()) {
        Some(prev) => {
            warning = warning.or_else(|| prev.warning.clone());
            rendered = Rendered::Shared(prev);
            rendered.track()
        }
        None => {
            let (t, w) = crate::soundfonts::render(&req, &inst).map_err(|e| bad(cmd, e))?;
            warning = warning.or(w);
            rendered = Rendered::Own(Box::new(t));
            rendered.track()
        }
    };
    let plan = track.plan.clone();
    let n0 = s.history.undo.len();
    let result = (|| -> Result<Value> {
        let dir = crate::voiceover::generated_dir(s, p, "Generated Audio")?;
        let path = crate::voiceover::unique_wav_path(s, &dir, "Music")?;
        if !cfg!(target_arch = "wasm32") {
            std::fs::create_dir_all(&dir).map_err(|e| EngineError::Other(format!("{dir}: {e}")))?;
        }
        let title = std::path::Path::new(&path).file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "Music".into());
        let mut meta = crate::music::metadata(&plan);
        meta.push((INSTRUMENTS_KEY.into(), track.instruments.clone()));
        let bytes = write_wav_f32_stereo_tagged(&track.mix.left, &track.mix.right, RATE, &tags(&title, &plan));
        let item = write_and_import(s, &path, bytes, &meta)?;
        let mut out = json!({
            "item": item.0, "path": path, "kind": "music", "seconds": track.mix.seconds(), "seed": req.seed,
            "sampleRate": RATE, "clip": Value::Null, "track": Value::Null, "instruments": track.instruments,
            "soundfont": inst.to_json(), "warning": warning,
        });
        if let (Some(o), Value::Object(info)) = (out.as_object_mut(), crate::music::track_json(&plan)) {
            o.extend(info);
        }
        // the stems, written beside the mix
        let mut stems: Vec<(music::Stem, ItemId, String)> = Vec::new();
        for (stem, audio) in &track.stems {
            let spath = std::path::Path::new(&dir).join(format!("{title} {}.wav", stem.id())).to_string_lossy().into_owned();
            let mut smeta = meta.clone();
            smeta.push((STEM_KEY.into(), stem.id().into()));
            let bytes = write_wav_f32_stereo_tagged(&audio.left, &audio.right, RATE, &tags(&format!("{title} {}", stem.id()), &plan));
            let sitem = write_and_import(s, &spath, bytes, &smeta)?;
            stems.push((*stem, sitem, spath));
        }
        let mut stems_json: Vec<Value> =
            stems.iter().map(|(st, it, sp)| json!({"stem": st.id(), "item": it.0, "path": sp, "clip": Value::Null, "track": Value::Null})).collect();
        if place {
            if stems.is_empty() {
                let (clip, tr) = place_item(s, item, at, dur, chosen)?;
                let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
                out["clip"] = json!(clip.0);
                out["track"] = json!(crate::mixer::strip_label(seq, tr));
            } else {
                // each stem on its own track, so it can be muted or rebalanced
                let mut want = chosen;
                let mut clips = Vec::new();
                for ((_, sitem, _), j) in stems.iter().zip(stems_json.iter_mut()) {
                    let (clip, tr) = place_item(s, *sitem, at, dur, want.take())?;
                    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
                    j["clip"] = json!(clip.0);
                    j["track"] = json!(crate::mixer::strip_label(seq, tr));
                    clips.push(clip);
                }
                s.state.selection = clips;
            }
            out["start"] = json!(at.0);
            out["duration"] = json!(dur.0);
        }
        if !stems_json.is_empty() {
            out["stems"] = json!(stems_json);
        }
        Ok(out)
    })();
    finish(s, n0, result)
}
