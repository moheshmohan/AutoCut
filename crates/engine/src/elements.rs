//! The bundled Elements library: original sound effects, stickers and animated text presets that
//! drop straight onto the timeline — raw material for edits and templates.
//!
//! | command | does |
//! |---|---|
//! | `elements.list` | the elements, optionally one `kind` (`sound`, `sticker`, `text`) or matching a `query`: id, name, category, description, length, licence and how each was made |
//! | `elements.insert` | put element `id` on the timeline at `time` (ticks; or `seconds`, `frame`, `timecode`; default: the playhead), on `track` (`"A2"`, `"V3"`, a track name or id; default: the first free track) — one undo step |
//!
//! - **Sounds** are synthesized when inserted, never stored: whooshes, impacts, risers, the swish,
//!   chime, shutter, soft pop and thump by the synth ([`filmcraft_audio_dsp::synth`]), the click, pop, swooshes, sweep and boom by
//!   [`filmcraft_audio_dsp::sfx`]. Each element has fixed parameters and a fixed seed, so it is
//!   the same sound every time. The WAV (48 kHz stereo float) goes in the `Generated Audio`
//!   folder (or `dir`); inserting an element again reuses its file and project item.
//! - **Stickers and text presets** are graphics templates from [`filmcraft_project::elements`],
//!   placed as graphic clips linked to their template, so Essential Graphics ▸ Edit shows their
//!   properties (`values` sets them on insert, as `graphics.template.apply` does). `duration`
//!   (seconds) lengthens or shortens the clip; text presets keep their intro and outro whole.
//!   The stickers' preview SVGs in `assets/elements/stickers/` are exported by [`sticker_svg`].
//!
//! Everything here is original work by FilmCraft contributors under the project licence
//! (`ATTRIBUTION.md`).

use serde_json::{Value, json};

use filmcraft_audio_dsp::sfx::{self, Sfx};
use filmcraft_audio_dsp::synth::{self, Stereo};
use filmcraft_project::elements as gfx;
use filmcraft_project::gtemplate::GraphicsTemplate;
use filmcraft_project::{ClipId, ItemId, MediaRef, TrackId, TrackKind};
use filmcraft_time::{Tick, TimeRange};

use crate::commands::{CommandSpec, always, bad, has_seq, time_p};
use crate::{EngineError, Result, Session};

const LIST: &str = "elements.list";
const INSERT: &str = "elements.insert";
/// Sample rate of the sound elements.
pub const RATE: u32 = 48_000;
/// Longest graphic element clip, seconds.
const MAX_DURATION: f64 = 3600.0;
/// Latest insertion time, seconds (a day).
const MAX_TIME: f64 = 86_400.0;
/// Where the sticker previews live in the repository.
pub const STICKER_DIR: &str = "assets/elements/stickers";

/// What an element is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Sound,
    Sticker,
    Text,
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::Sound, Kind::Sticker, Kind::Text];

    pub fn id(self) -> &'static str {
        match self {
            Kind::Sound => "sound",
            Kind::Sticker => "sticker",
            Kind::Text => "text",
        }
    }

    /// `sound`, `sticker`, `text` (or their plurals), any case.
    pub fn from_id(s: &str) -> Option<Kind> {
        let s = s.trim().to_ascii_lowercase();
        let s = s.strip_suffix('s').unwrap_or(&s);
        Kind::ALL.into_iter().find(|k| k.id() == s || (*k == Kind::Text && s == "animated-text"))
    }

    /// The folder it is listed under.
    pub fn category(self) -> &'static str {
        match self {
            Kind::Sound => "Sounds",
            Kind::Sticker => gfx::STICKERS,
            Kind::Text => gfx::ANIMATED_TEXT,
        }
    }
}

/// How a sound element is made.
#[derive(Clone, Copy, Debug)]
pub enum Recipe {
    /// One of the synth's sound effects (whoosh, impact, riser, swish, chime, shutter, pop, thump).
    Synth { kind: synth::Kind, seconds: f64, intensity: f64, seed: u64 },
    /// One of [`filmcraft_audio_dsp::sfx`]'s effects.
    Sfx { kind: Sfx, seconds: f64, seed: u64 },
}

/// A sound element.
#[derive(Clone, Copy, Debug)]
pub struct Sound {
    pub id: &'static str,
    pub name: &'static str,
    /// File name stem of its WAV.
    pub file: &'static str,
    pub description: &'static str,
    pub recipe: Recipe,
}

impl Sound {
    pub fn seconds(&self) -> f64 {
        match self.recipe {
            Recipe::Synth { seconds, .. } | Recipe::Sfx { seconds, .. } => seconds,
        }
    }

    /// How it was made, for the listing and the attribution.
    pub fn made(&self) -> String {
        match self.recipe {
            Recipe::Synth { kind, seconds, intensity, seed } => format!(
                "synthesized by FilmCraft (filmcraft_audio_dsp::synth `{}`, {seconds} s, intensity {intensity}, seed {seed}): oscillators, noise, filters and reverb, no samples",
                kind.id()
            ),
            Recipe::Sfx { kind, seconds, seed } => {
                format!(
                    "synthesized by FilmCraft (filmcraft_audio_dsp::sfx `{}`, {seconds} s, seed {seed}): oscillators, noise, filters and reverb, no samples",
                    kind.id()
                )
            }
        }
    }

    /// The audio: the same every time.
    pub fn render(&self) -> std::result::Result<Stereo, synth::SynthError> {
        match self.recipe {
            Recipe::Synth { kind, seconds, intensity, seed } => {
                let mut p = synth::Params::new(kind);
                p.seconds = seconds;
                p.intensity = intensity;
                p.seed = seed;
                synth::generate(&p, RATE)
            }
            Recipe::Sfx { kind, seconds, seed } => sfx::generate(kind, seconds, seed, RATE),
        }
    }
}

const fn synth_sound(
    id: &'static str,
    name: &'static str,
    file: &'static str,
    description: &'static str,
    kind: synth::Kind,
    seconds: f64,
    intensity: f64,
    seed: u64,
) -> Sound {
    Sound { id, name, file, description, recipe: Recipe::Synth { kind, seconds, intensity, seed } }
}

const fn sfx_sound(id: &'static str, name: &'static str, file: &'static str, description: &'static str, kind: Sfx, seconds: f64, seed: u64) -> Sound {
    Sound { id, name, file, description, recipe: Recipe::Sfx { kind, seconds, seed } }
}

/// The sound elements.
pub const SOUNDS: [Sound; 17] = [
    synth_sound("sound.whoosh-short", "Whoosh – Short", "Whoosh Short", "A quick air pass for a fast cut or move.", synth::Kind::Whoosh, 0.6, 0.6, 11),
    synth_sound("sound.whoosh-long", "Whoosh – Long", "Whoosh Long", "A slower, fuller pass for a transition.", synth::Kind::Whoosh, 1.8, 0.75, 12),
    synth_sound("sound.impact-soft", "Impact – Soft", "Impact Soft", "A gentle low hit with a short tail.", synth::Kind::Impact, 2.0, 0.25, 21),
    synth_sound(
        "sound.impact-hard",
        "Impact – Hard",
        "Impact Hard",
        "A heavy cinematic hit: sub boom, crack and a long tail.",
        synth::Kind::Impact,
        3.5,
        1.0,
        22,
    ),
    synth_sound(
        "sound.riser-2s",
        "Riser – 2 s",
        "Riser 2s",
        "Two seconds of rising tension that stops dead; end it on the cut.",
        synth::Kind::Riser,
        2.0,
        0.6,
        31,
    ),
    synth_sound("sound.riser-4s", "Riser – 4 s", "Riser 4s", "A longer build to a cut.", synth::Kind::Riser, 4.0, 0.8, 32),
    synth_sound("sound.swish", "Swish", "Swish", "A soft, short brush of air: a gentle transition.", synth::Kind::Swish, 0.5, 0.6, 33),
    synth_sound("sound.chime", "Chime", "Chime", "A soft bell ringing out, for a new chapter or a reveal.", synth::Kind::Chime, 1.8, 0.6, 34),
    synth_sound(
        "sound.shutter",
        "Shutter",
        "Shutter",
        "A camera shutter: two quick mechanical clicks, for a snap cut or a photo.",
        synth::Kind::Shutter,
        0.25,
        0.6,
        35,
    ),
    synth_sound("sound.pop-soft", "Pop – Soft", "Pop Soft", "A soft bubble pop on a cut or a reveal.", synth::Kind::Pop, 0.25, 0.5, 36),
    synth_sound("sound.thump", "Thump", "Thump", "A soft, warm low hit: a gentler, shorter impact.", synth::Kind::Thump, 0.8, 0.6, 37),
    sfx_sound("sound.swoosh-up", "Swoosh – Up", "Swoosh Up", "An airy pass rising in pitch, left to right.", Sfx::SwooshUp, 0.7, 41),
    sfx_sound("sound.swoosh-down", "Swoosh – Down", "Swoosh Down", "An airy pass falling in pitch, right to left.", Sfx::SwooshDown, 0.7, 42),
    sfx_sound("sound.click", "Click", "Click", "A dry interface click for taps and pop-ups.", Sfx::Click, 0.12, 51),
    sfx_sound("sound.pop", "Pop", "Pop", "A bubbly pop for stickers and text appearing.", Sfx::Pop, 0.3, 52),
    sfx_sound("sound.sweep", "Sweep", "Sweep", "A resonant filter sweep up the spectrum and back.", Sfx::Sweep, 1.6, 61),
    sfx_sound("sound.boom", "Boom", "Boom", "A deep sub drop with a rumble and a dark tail.", Sfx::Boom, 3.0, 71),
];

/// A sound element by id.
pub fn sound(id: &str) -> Option<&'static Sound> {
    SOUNDS.iter().find(|s| s.id == id)
}

/// A sticker or text preset by element id (`sticker.arrow`).
pub fn graphic(id: &str) -> Option<GraphicsTemplate> {
    gfx::all().into_iter().find(|t| gfx::element_id(t) == id)
}

fn graphic_kind(t: &GraphicsTemplate) -> Kind {
    if t.category == gfx::STICKERS { Kind::Sticker } else { Kind::Text }
}

/// The repository path of a sticker's preview SVG.
pub fn sticker_file(t: &GraphicsTemplate) -> String {
    let id = gfx::element_id(t);
    format!("{STICKER_DIR}/{}.svg", id.strip_prefix("sticker.").unwrap_or(id))
}

/// One element as JSON (the `elements.list` row).
fn sound_json(s: &Sound) -> Value {
    json!({
        "id": s.id, "kind": Kind::Sound.id(), "name": s.name, "category": Kind::Sound.category(), "description": s.description,
        "seconds": s.seconds(), "sampleRate": RATE, "channels": 2, "author": gfx::AUTHOR, "license": gfx::LICENSE,
        "source": "original work", "made": s.made(),
    })
}

fn graphic_json(t: &GraphicsTemplate) -> Value {
    let kind = graphic_kind(t);
    let made = match kind {
        Kind::Sticker => "vector shapes defined in code (filmcraft_project::elements); the preview SVG is exported from the same layers",
        _ => "text and shape layers animated with keyframes, defined in code (filmcraft_project::elements); the bundled Inter font",
    };
    let mut v = json!({
        "id": gfx::element_id(t), "kind": kind.id(), "name": t.name, "category": t.category, "description": t.description,
        "seconds": t.duration.seconds(), "author": t.author, "license": t.license, "source": "original work", "made": made,
        "template": t.id, "layers": t.layers.len(),
        "controls": t.controls.iter().map(|c| json!({"id": c.id, "name": c.name, "kind": c.kind.name()})).collect::<Vec<_>>(),
    });
    if kind == Kind::Sticker {
        v["file"] = json!(sticker_file(t));
    }
    v
}

/// Every element as JSON: sounds, stickers, text presets.
pub fn list(kind: Option<Kind>, query: &str) -> Vec<Value> {
    let q = query.trim().to_lowercase();
    let hit =
        |v: &Value| q.is_empty() || ["id", "name", "category", "description"].iter().any(|k| v[*k].as_str().is_some_and(|s| s.to_lowercase().contains(&q)));
    let mut out: Vec<Value> = Vec::new();
    if kind.is_none_or(|k| k == Kind::Sound) {
        out.extend(SOUNDS.iter().map(sound_json));
    }
    out.extend(gfx::all().iter().filter(|t| kind.is_none_or(|k| k == graphic_kind(t))).map(graphic_json));
    out.retain(hit);
    out
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: LIST,
            label: "List Elements",
            menu: &[],
            shortcut: None,
            params: r#"{"kind":"sound|sticker|text"?,"query":str?}"#,
            enabled: always,
            run: list_cmd,
            journal: false,
        },
        CommandSpec {
            id: INSERT,
            label: "Insert Element",
            menu: &[],
            shortcut: None,
            params: r#"{"id":"sound.whoosh-short|sticker.arrow|text.title-pop|…","time":ticks?|"seconds":f64?,"track":"A2"|"V3"|name|id?,"duration":seconds? (stickers, text),"values":{control: value}? (stickers, text),"dir":str? (sounds)}"#,
            enabled: has_seq,
            run: insert,
            journal: true,
        },
    ]
}

fn list_cmd(_s: &mut Session, p: &Value) -> Result<Value> {
    let kind = match p.get("kind") {
        None | Some(Value::Null) => None,
        Some(Value::String(k)) => Some(Kind::from_id(k).ok_or_else(|| bad(LIST, format!("unknown kind `{k}` (sound, sticker, text)")))?),
        Some(v) => return Err(bad(LIST, format!("`kind` must be sound, sticker or text, not {v}"))),
    };
    let query = match p.get("query") {
        None | Some(Value::Null) => "",
        Some(Value::String(q)) => q.as_str(),
        Some(v) => return Err(bad(LIST, format!("`query` must be text, not {v}"))),
    };
    Ok(Value::Array(list(kind, query)))
}

/// Where to insert: the time (validated) from `time` / `seconds` / `frame` / `timecode`.
fn insert_time(s: &Session, p: &Value) -> Result<Tick> {
    if let Some(v) = p.get("time")
        && !v.is_i64()
    {
        return Err(bad(INSERT, format!("`time` must be a whole number of ticks, not {v}")));
    }
    if let Some(v) = p.get("seconds")
        && !v.as_f64().is_some_and(f64::is_finite)
    {
        return Err(bad(INSERT, format!("`seconds` must be a number, not {v}")));
    }
    let at = time_p(s, p, "").unwrap_or_else(|| s.playhead());
    if at < Tick::ZERO {
        return Err(bad(INSERT, "the time cannot be negative"));
    }
    if at > Tick::from_seconds_f64(MAX_TIME) {
        return Err(bad(INSERT, "the time is more than a day into the sequence"));
    }
    Ok(at)
}

fn insert(s: &mut Session, p: &Value) -> Result<Value> {
    let id = match p.get("id") {
        Some(Value::String(id)) => id.trim().to_string(),
        Some(v) => return Err(bad(INSERT, format!("`id` must be an element id, not {v}"))),
        None => return Err(bad(INSERT, "need `id` (see elements.list)")),
    };
    if s.active_sequence().is_none() {
        return Err(EngineError::NoSequence);
    }
    let at = insert_time(s, p)?;
    if let Some(snd) = sound(&id) {
        for k in ["duration", "values"] {
            if p.get(k).is_some_and(|v| !v.is_null()) {
                return Err(bad(INSERT, format!("`{k}` is for stickers and text; a sound element has its own length and no properties")));
            }
        }
        return insert_sound(s, p, snd, at);
    }
    if let Some(t) = graphic(&id) {
        return insert_graphic(s, p, t, at);
    }
    let shown: String = id.chars().take(80).collect();
    Err(bad(INSERT, format!("no element `{shown}` (elements.list shows them)")))
}

// ------------------------------------------------------------------------------------------------
// sounds
// ------------------------------------------------------------------------------------------------

fn audio_free(s: &Session, track: TrackId, range: TimeRange) -> bool {
    s.active_sequence()
        .and_then(|q| q.audio_tracks.iter().find(|t| t.id == track))
        .is_some_and(|t| !t.locked && !t.items.iter().any(|it| it.start < range.end() && range.start < it.end()))
}

/// The audio track `track` names, if it is free for `range`; None without `track`.
fn chosen_audio_track(s: &Session, p: &Value, range: TimeRange) -> Result<Option<TrackId>> {
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let Some(v) = p.get("track").filter(|v| !v.is_null()) else { return Ok(None) };
    if !(v.is_string() || v.is_u64()) {
        return Err(bad(INSERT, format!("`track` must be a track like \"A2\", a track name or an id, not {v}")));
    }
    let id =
        crate::mixer::strip_ref(seq, v).filter(|id| seq.audio_tracks.iter().any(|t| t.id == *id)).ok_or_else(|| bad(INSERT, format!("no audio track {v}")))?;
    if !audio_free(s, id, range) {
        return Err(bad(INSERT, format!("audio track {v} is locked or has clips there")));
    }
    Ok(Some(id))
}

/// The project item holding the element's WAV, and its path: the item already imported from
/// `<dir>/<file>.wav` while that file still holds the element's audio (a deleted file is written
/// back), else that file (written, or found with the same audio) imported. A different file of
/// that name is never touched: the WAV goes beside it as `<file> <n>.wav`. Returns (item, path,
/// reused).
pub(crate) fn sound_item(s: &mut Session, p: &Value, snd: &Sound, bytes: Vec<u8>) -> Result<(ItemId, String, bool)> {
    let dir = crate::voiceover::generated_dir(s, p, "Generated Audio");
    let mut path = std::path::Path::new(&dir).join(format!("{}.wav", snd.file)).to_string_lossy().into_owned();
    let write = |s: &Session, path: &str| -> Result<()> {
        if !cfg!(target_arch = "wasm32") {
            std::fs::create_dir_all(&dir).map_err(|e| EngineError::Other(format!("{dir}: {e}")))?;
        }
        s.services.write_file(path, &bytes).map_err(|e| EngineError::Other(format!("{path}: {e}")))
    };
    let on_disk = s.services.read_file(&path).ok();
    let imported = s.project.items.values().find(|i| i.as_media().is_some_and(|m| matches!(&m.media, MediaRef::File { path: f } if *f == path))).map(|i| i.id);
    match (imported, &on_disk) {
        (Some(item), Some(b)) if *b == bytes => return Ok((item, path, true)),
        (Some(item), None) => {
            write(s, &path)?;
            return Ok((item, path, true));
        }
        _ => {}
    }
    if on_disk.is_some_and(|b| b != bytes) {
        // someone else's file has the name: keep it, write ours beside it
        path = crate::voiceover::unique_wav_path(s, &dir, snd.file)?;
        write(s, &path)?;
    } else if s.services.file_size(&path).is_err() {
        write(s, &path)?;
    }
    let item = crate::commands::import_bytes(s, &path, bytes.clone().into(), None)?;
    Ok((item, path, false))
}

fn insert_sound(s: &mut Session, p: &Value, snd: &Sound, at: Tick) -> Result<Value> {
    let audio = snd.render().map_err(|e| bad(INSERT, e.to_string()))?;
    let dur = Tick::from_units(audio.left.len() as i64, i64::from(RATE));
    let range = TimeRange::new(at, dur);
    // the track first, so a busy `track` fails before anything is written
    let chosen = chosen_audio_track(s, p, range)?;
    let n0 = s.history.undo.len();
    let result = (|| -> Result<Value> {
        let bytes = crate::audio_generate::write_wav_f32_stereo(&audio.left, &audio.right, RATE);
        let (item, path, reused) = sound_item(s, p, snd, bytes)?;
        let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
        let track = match chosen.or_else(|| s.active_sequence().and_then(|q| q.audio_tracks.iter().map(|t| t.id).find(|id| audio_free(s, *id, range)))) {
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
        let name = snd.name.to_string();
        let clip: ClipId = s.edit("Insert Element", |pr, st| {
            let mut it = pr
                .make_track_item(item, TrackKind::Audio, at, TimeRange::new(Tick::ZERO, dur), seq_rate)
                .ok_or(EngineError::Other("the element's audio is not importable".into()))?;
            it.duration = dur;
            it.name = name;
            let id = it.id;
            let seq = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
            let t = seq.audio_tracks.iter_mut().find(|t| t.id == track).ok_or_else(|| bad(INSERT, "the audio track was deleted"))?;
            let pos = t.items.partition_point(|x| x.start <= at);
            t.items.insert(pos, it);
            seq.check().map_err(EngineError::Other)?;
            st.selection = vec![id];
            Ok(id)
        })?;
        let seq = s.project.sequence(seq_id).ok_or(EngineError::NoSequence)?;
        Ok(json!({
            "id": snd.id, "kind": Kind::Sound.id(), "name": snd.name, "clip": clip.0, "track": crate::mixer::strip_label(seq, track),
            "start": at.0, "duration": dur.0, "seconds": audio.seconds(), "item": item.0, "path": path, "reused": reused, "sampleRate": RATE,
        }))
    })();
    crate::clip_ops::collapse_history(s, n0, "Insert Element");
    if result.is_err() && s.history.undo.len() > n0 {
        let _ = s.undo();
        s.history.redo.clear();
    }
    result
}

// ------------------------------------------------------------------------------------------------
// stickers and text
// ------------------------------------------------------------------------------------------------

/// The video track index `track` names (`"V2"`, a track name or id), checked free for `range`.
fn chosen_video_track(s: &Session, p: &Value, range: TimeRange) -> Result<Option<usize>> {
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let Some(v) = p.get("track").filter(|v| !v.is_null()) else { return Ok(None) };
    let idx = match v {
        Value::Number(n) => n.as_u64().and_then(|id| seq.video_tracks.iter().position(|t| t.id == TrackId(id))),
        Value::String(name) => {
            let name = name.trim();
            name.strip_prefix(['V', 'v'])
                .and_then(|n| n.parse::<usize>().ok())
                .and_then(|n| n.checked_sub(1))
                .filter(|i| *i < seq.video_tracks.len())
                .or_else(|| seq.video_tracks.iter().position(|t| t.name.eq_ignore_ascii_case(name)))
        }
        _ => return Err(bad(INSERT, format!("`track` must be a track like \"V2\", a track name or an id, not {v}"))),
    };
    let idx = idx.ok_or_else(|| bad(INSERT, format!("no video track {v}")))?;
    let t = seq.video_tracks.get(idx).ok_or_else(|| bad(INSERT, format!("no video track {v}")))?;
    if t.locked || t.items.iter().any(|i| i.range().overlaps(&range)) {
        return Err(bad(INSERT, format!("video track {v} is locked or has clips there")));
    }
    Ok(Some(idx))
}

fn insert_graphic(s: &mut Session, p: &Value, t: GraphicsTemplate, at: Tick) -> Result<Value> {
    let dur = match p.get("duration").filter(|v| !v.is_null()) {
        None => t.duration,
        Some(v) => {
            let sec =
                v.as_f64().filter(|x| x.is_finite() && *x > 0.0).ok_or_else(|| bad(INSERT, format!("`duration` must be more than zero seconds, not {v}")))?;
            if sec > MAX_DURATION {
                return Err(bad(INSERT, format!("`duration` is at most {MAX_DURATION} seconds")));
            }
            Tick::from_seconds_f64(sec)
        }
    };
    let values = match p.get("values").filter(|v| !v.is_null()) {
        None => None,
        Some(Value::Object(o)) => Some(o.clone()),
        Some(v) => return Err(bad(INSERT, format!("`values` must be {{property: value}}, not {v}"))),
    };
    let (w, h, rate) = s.active_sequence().map(|q| (q.settings.width, q.settings.height, q.settings.frame_rate)).ok_or(EngineError::NoSequence)?;
    let dur = rate.snap_nearest(dur).max(rate.frame_duration());
    let track = chosen_video_track(s, p, TimeRange::new(rate.snap(at), dur))?;
    let (layers, meta) = t.instantiate((w, h));
    let mut place = json!({"time": at.0});
    if let Some(i) = track {
        place["track"] = json!(i);
    }
    let n0 = s.history.undo.len();
    let result = (|| -> Result<Value> {
        let clip = crate::graphics::place_video_clip(s, INSERT, &t.name, &place, "Insert Element", layers, move |pr, (w, h, rate)| {
            (crate::graphics::graphic_source(pr, w, h, rate), dur)
        })?;
        s.edit_sequence("Insert Element", move |q, _, _| {
            let (_, it) = q.find_item_mut(clip).ok_or(filmcraft_edit::EditError::NoItem(clip))?;
            it.graphic = Some(Box::new(meta));
            Ok(())
        })?;
        if let Some(vals) = values {
            s.execute("graphics.template.set", json!({"clip": clip.0, "values": vals}))?;
        }
        let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
        let (track, it) = q
            .video_tracks
            .iter()
            .enumerate()
            .find_map(|(i, tr)| tr.items.iter().find(|x| x.id == clip).map(|x| (i, x)))
            .ok_or_else(|| bad(INSERT, "the inserted clip is gone"))?;
        Ok(json!({
            "id": gfx::element_id(&t), "kind": graphic_kind(&t).id(), "name": t.name, "clip": clip.0, "track": format!("V{}", track + 1),
            "start": it.start.0, "duration": it.duration.0, "seconds": it.duration.seconds(), "template": t.id,
            "controls": t.controls.iter().map(|c| json!({"id": c.id, "name": c.name, "kind": c.kind.name()})).collect::<Vec<_>>(),
        }))
    })();
    crate::clip_ops::collapse_history(s, n0, "Insert Element");
    if result.is_err() && s.history.undo.len() > n0 {
        let _ = s.undo();
        s.history.redo.clear();
    }
    s.state.graphic_layers.clear();
    result
}

// ------------------------------------------------------------------------------------------------
// sticker previews (SVG)
// ------------------------------------------------------------------------------------------------

/// A number for SVG: two decimals at most, no trailing zeros, no `-0`.
fn num(x: f64) -> String {
    let x = if x.is_finite() { x } else { 0.0 };
    let s = format!("{x:.2}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" || s.is_empty() { "0".into() } else { s.to_string() }
}

fn hex(c: [f32; 4]) -> String {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]))
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// Paint attributes: fill, the first stroke (outer strokes are drawn under the fill at twice the
/// width, which is what an outer stroke looks like), opacity of the colours.
fn paint(a: &filmcraft_project::graphic::Appearance) -> String {
    let mut out = match a.fill.or(a.gradient.as_ref().map(|g| g.start)) {
        Some(c) if c[3] < 1.0 => format!(" fill=\"{}\" fill-opacity=\"{}\"", hex(c), num(f64::from(c[3]))),
        Some(c) => format!(" fill=\"{}\"", hex(c)),
        None => " fill=\"none\"".into(),
    };
    if let Some((c, w, kind)) = a.strokes.first() {
        let w = if *kind == 1 { f64::from(*w) } else { f64::from(*w) * 2.0 };
        out.push_str(&format!(" stroke=\"{}\" stroke-width=\"{}\" stroke-linejoin=\"round\"", hex(*c), num(w)));
        if *kind != 1 {
            out.push_str(" paint-order=\"stroke\"");
        }
    }
    out
}

/// The preview SVG of a sticker (or any graphic template) at rest: its layers with responsive pins
/// resolved, cropped to what they cover. Text is set in Inter (a viewer without it falls back to
/// a sans-serif face). Deterministic, so the bundled files can be checked against the vectors.
pub fn sticker_svg(t: &GraphicsTemplate) -> String {
    use filmcraft_project::graphic::LayerContent;
    use filmcraft_render::graphic_clip::{graphic_specs, layer_local_bounds, layer_matrix, text_layout};
    let canvas = (t.canvas[0].max(1), t.canvas[1].max(1));
    let specs = graphic_specs(&t.layers, Tick::ZERO, canvas);
    let mut bounds = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    let mut body = String::new();
    let mut defs = String::new();
    for (n, spec) in specs.iter().enumerate().filter(|(_, s)| s.enabled) {
        let m = layer_matrix(spec);
        let a = &spec.appearance;
        // what the layer covers: content, half the stroke, the text box, the shadow
        let lb = layer_local_bounds(spec);
        let mut pad = a.strokes.iter().map(|(_, w, k)| if *k == 1 { f64::from(*w) / 2.0 } else { f64::from(*w) }).fold(0.0, f64::max);
        if let Some((_, p, _)) = a.background {
            pad = pad.max(f64::from(p));
        }
        let mut boxes = vec![(0.0, 0.0, pad)];
        if let Some(sh) = &a.shadow {
            boxes.push((f64::from(sh.offset.0), f64::from(sh.offset.1), pad + f64::from(sh.blur) + f64::from(sh.size)));
        }
        for (dx, dy, pad) in boxes {
            let (x0, y0, x1, y1) = (f64::from(lb[0]) + dx - pad, f64::from(lb[1]) + dy - pad, f64::from(lb[2]) + dx + pad, f64::from(lb[3]) + dy + pad);
            for (x, y) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)] {
                let q = m.apply(filmcraft_geom::Vec2::new(x, y));
                bounds = [bounds[0].min(q.x), bounds[1].min(q.y), bounds[2].max(q.x), bounds[3].max(q.y)];
            }
        }
        let mut attrs = format!(" transform=\"matrix({} {} {} {} {} {})\"", num(m.a), num(m.b), num(m.c), num(m.d), num(m.e), num(m.f));
        if spec.transform.opacity < 1.0 {
            attrs.push_str(&format!(" opacity=\"{}\"", num(f64::from(spec.transform.opacity))));
        }
        if let Some(sh) = &a.shadow {
            let id = format!("shadow{}", n + 1);
            defs.push_str(&format!(
                "    <filter id=\"{id}\" x=\"-50%\" y=\"-50%\" width=\"200%\" height=\"200%\"><feDropShadow dx=\"{}\" dy=\"{}\" stdDeviation=\"{}\" flood-color=\"{}\" flood-opacity=\"{}\"/></filter>\n",
                num(f64::from(sh.offset.0)),
                num(f64::from(sh.offset.1)),
                num(f64::from(sh.blur) / 2.0),
                hex(sh.color),
                num(f64::from(sh.color[3]))
            ));
            attrs.push_str(&format!(" filter=\"url(#{id})\""));
        }
        let name = xml_escape(&spec.name);
        match &spec.content {
            LayerContent::Shape(sh) => {
                let (w, h) = (f64::from(sh.size.0), f64::from(sh.size.1));
                let el = match sh.shape {
                    1 => format!("<ellipse cx=\"0\" cy=\"0\" rx=\"{}\" ry=\"{}\"", num(w / 2.0), num(h / 2.0)),
                    // the same outline the renderer fills
                    2 | 3 => format!("<polygon points=\"{}\"", polygon_points(sh)),
                    _ => {
                        let r = f64::from(sh.corner_radius).min(w / 2.0).min(h / 2.0).max(0.0);
                        let mut e = format!("<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"", num(-w / 2.0), num(-h / 2.0), num(w), num(h));
                        if r > 0.0 {
                            e.push_str(&format!(" rx=\"{}\"", num(r)));
                        }
                        e
                    }
                };
                body.push_str(&format!("  <g id=\"{}\"{attrs}>\n    {el}{}/>\n  </g>\n", slug_id(&name, n), paint(a)));
            }
            LayerContent::Text(tx) => {
                let mut inner = String::new();
                if let Some((c, p, r)) = a.background {
                    let b = text_layout(tx).bounds;
                    let (p, r) = (f64::from(p), f64::from(r));
                    let (x0, y0, x1, y1) = (f64::from(b[0]) - p, f64::from(b[1]) - p, f64::from(b[2]) + p, f64::from(b[3]) + p);
                    let r = r.min((x1 - x0) / 2.0).min((y1 - y0) / 2.0).max(0.0);
                    let op = if c[3] < 1.0 { format!(" fill-opacity=\"{}\"", num(f64::from(c[3]))) } else { String::new() };
                    inner.push_str(&format!(
                        "    <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" rx=\"{}\" fill=\"{}\"{op}/>\n",
                        num(x0),
                        num(y0),
                        num(x1 - x0),
                        num(y1 - y0),
                        num(r),
                        hex(c)
                    ));
                }
                let anchor = match tx.align {
                    1 => "middle",
                    2 => "end",
                    _ => "start",
                };
                let style = tx.style.to_ascii_lowercase();
                let weight = if style.contains("bold") && style.contains("semi") {
                    600
                } else if style.contains("bold") {
                    700
                } else if style.contains("medium") {
                    500
                } else {
                    400
                };
                let mut font = format!(
                    " font-family=\"{}, sans-serif\" font-size=\"{}\" font-weight=\"{weight}\" text-anchor=\"{anchor}\"",
                    xml_escape(&tx.font),
                    num(f64::from(tx.size))
                );
                if style.contains("italic") {
                    font.push_str(" font-style=\"italic\"");
                }
                if tx.tracking != 0.0 {
                    font.push_str(&format!(" letter-spacing=\"{}\"", num(f64::from(tx.tracking) / 1000.0 * f64::from(tx.size))));
                }
                let mut fill_only = a.clone();
                fill_only.background = None;
                inner.push_str(&format!("    <text x=\"0\" y=\"0\"{font}{}>{}</text>\n", paint(&fill_only), xml_escape(&tx.text)));
                body.push_str(&format!("  <g id=\"{}\"{attrs}>\n{inner}  </g>\n", slug_id(&name, n)));
            }
        }
    }
    if bounds[0] > bounds[2] {
        bounds = [0.0, 0.0, f64::from(canvas.0), f64::from(canvas.1)];
    }
    let (x0, y0) = ((bounds[0] - 8.0).floor(), (bounds[1] - 8.0).floor());
    let (x1, y1) = ((bounds[2] + 8.0).ceil(), (bounds[3] + 8.0).ceil());
    let (w, h) = (x1 - x0, y1 - y0);
    let defs = if defs.is_empty() { String::new() } else { format!("  <defs>\n{defs}  </defs>\n") };
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!-- {name}: a FilmCraft Elements sticker (`{id}`). Original work by {author}, {license}.\n     Exported from the vectors in crates/project/src/elements.rs by crates/engine/src/elements.rs;\n     regenerate with FILMCRAFT_BLESS=1 cargo test -p filmcraft-engine bundled_sticker_svgs -->\n<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"{} {} {} {}\">\n{defs}{body}</svg>\n",
        num(w),
        num(h),
        num(x0),
        num(y0),
        num(w),
        num(h),
        name = xml_escape(&t.name),
        id = gfx::element_id(t),
        author = xml_escape(&t.author),
        license = xml_escape(&t.license),
    )
}

/// `points` of an SVG polygon for a polygon (shape 2) or path (shape 3) layer.
fn polygon_points(sh: &filmcraft_project::graphic::ShapeProps) -> String {
    let pts: Vec<(f64, f64)> = if sh.shape == 3 && sh.points.len() >= 3 {
        sh.points.iter().map(|p| (f64::from(p[0]), f64::from(p[1]))).collect()
    } else {
        let n = sh.sides.max(3);
        let (w, h) = (f64::from(sh.size.0) / 2.0, f64::from(sh.size.1) / 2.0);
        (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * f64::from(i) / f64::from(n) - std::f64::consts::FRAC_PI_2;
                (a.cos() * w, a.sin() * h)
            })
            .collect()
    };
    pts.iter().map(|(x, y)| format!("{},{}", num(*x), num(*y))).collect::<Vec<_>>().join(" ")
}

/// An XML id for layer `n` from its name (`Bubble` → `bubble`; `layer3` for a name without
/// letters or digits).
fn slug_id(name: &str, n: usize) -> String {
    let s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
    let s = s.trim_matches('-');
    if s.is_empty() { format!("layer{}", n + 1) } else { s.to_string() }
}

#[cfg(test)]
#[path = "elements_tests.rs"]
mod tests;
