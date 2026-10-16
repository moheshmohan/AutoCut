//! Generated audio: sound effects and music beds synthesized in the app
//! ([`filmcraft_audio_dsp::synth`] — oscillators, noise, envelopes, filters; no samples or
//! models), written as WAV files, imported and placed.
//!
//! | command | does |
//! |---|---|
//! | `audio.generate` | synthesize a whoosh, impact, riser, drone or music bed `duration` seconds long, write `<Kind> <n>.wav` (48 kHz stereo float), import it and place it at `time`/`seconds` on a free audio track — one undo step |
//! | `audio.generateKinds` | the kinds, moods, defaults and limits |
//!
//! **Where it goes.** At `time` (default: the playhead) on `track` if given (it must be free
//! there), else on the first audio track free for the cue's length, else on a new audio track.
//! `place: false` only imports. Files go to `dir`, else Scratch Disks ▸ Captured, else a
//! `Generated Audio` folder next to the project (or in the data or temporary dir).
//!
//! **Reproducible.** The same parameters and `seed` always give the same audio; without a
//! seed, each new cue of a kind gets the next one, so two whooshes differ.

use serde_json::{Value, json};

use filmcraft_audio_dsp::synth;
use filmcraft_project::{ClipId, ItemId, TrackId, TrackKind};
use filmcraft_time::{Tick, TimeRange};

use crate::commands::{CommandSpec, always, bad, bool_p, f64_p, str_p, time_p, u64_p};
use crate::{EngineError, Result, Session};

/// Sample rate of generated audio.
pub const RATE: u32 = 48_000;

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
            r#"{"kind":"whoosh|impact|riser|drone|music"=music,"duration":seconds?,"seed":u64?,"intensity":0..1=0.7,"bpm":40..200=96,"key":"A|C#|Eb…"="A","mood":"epic|tense|uplifting|calm"="epic","time":ticks?|"seconds":f64?,"track":"A2"|id?,"place":bool=true,"dir":str?}"#,
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
        synth::Kind::Drone => "Drone",
        synth::Kind::Music => "Music",
    }
}

fn kinds(_s: &mut Session, _p: &Value) -> Result<Value> {
    let ks: Vec<Value> = synth::Kind::ALL
        .iter()
        .map(|k| json!({"id": k.id(), "name": label(*k), "defaultSeconds": k.default_seconds(), "maxSeconds": if matches!(k, synth::Kind::Music | synth::Kind::Drone) { synth::MAX_MUSIC_SECONDS } else { synth::MAX_SFX_SECONDS }}))
        .collect();
    let moods: Vec<&str> = synth::Mood::ALL.iter().map(|m| m.id()).collect();
    Ok(json!({"kinds": ks, "moods": moods, "sampleRate": RATE, "musicLufs": synth::MUSIC_LUFS}))
}

/// Interleaved stereo 32-bit float WAV.
pub fn write_wav_f32_stereo(left: &[f32], right: &[f32], rate: u32) -> Vec<u8> {
    let frames = left.len().min(right.len());
    let data_len = frames.saturating_mul(8);
    let mut v = Vec::with_capacity(44usize.saturating_add(data_len));
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(u32::try_from(36usize.saturating_add(data_len)).unwrap_or(u32::MAX)).to_le_bytes());
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
    v
}

/// The synth parameters a call asks for (validated; out-of-range numbers are clamped later).
fn params(s: &Session, p: &Value) -> Result<synth::Params> {
    let cmd = "audio.generate";
    let kind = match str_p(p, "kind") {
        Some(k) => synth::Kind::from_id(k).ok_or_else(|| bad(cmd, format!("unknown kind `{k}` (whoosh, impact, riser, drone, music)")))?,
        None => synth::Kind::Music,
    };
    let mut q = synth::Params::new(kind);
    // `duration` is the cue's length; `seconds` is where it goes, as in every other command
    if let Some(sec) = f64_p(p, "duration") {
        if !sec.is_finite() || sec <= 0.0 {
            return Err(bad(cmd, "`duration` must be more than zero seconds"));
        }
        q.seconds = sec;
    }
    if let Some(i) = f64_p(p, "intensity") {
        q.intensity = i;
    }
    if let Some(b) = f64_p(p, "bpm") {
        q.bpm = b;
    }
    if let Some(k) = str_p(p, "key") {
        q.key = synth::key_from_name(k).ok_or_else(|| bad(cmd, format!("unknown key `{k}` (A, A#/Bb, B, C, … G#)")))?;
    }
    if let Some(m) = str_p(p, "mood") {
        q.mood = synth::Mood::from_id(m).ok_or_else(|| bad(cmd, format!("unknown mood `{m}` (epic, tense, uplifting, calm)")))?;
    }
    q.seed = match u64_p(p, "seed") {
        Some(seed) => seed,
        // each new cue of a kind gets the next seed, so two whooshes differ
        None => {
            let prefix = format!("{} ", label(kind));
            1 + s.project.items.values().filter(|i| i.name.starts_with(&prefix)).count() as u64
        }
    };
    Ok(q)
}

fn track_is_free(s: &Session, track: TrackId, range: TimeRange) -> bool {
    s.active_sequence()
        .and_then(|q| q.audio_tracks.iter().find(|t| t.id == track))
        .is_some_and(|t| !t.locked && !t.items.iter().any(|it| it.start < range.end() && range.start < it.end()))
}

fn generate(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "audio.generate";
    let q = params(s, p)?;
    let place = bool_p(p, "place").unwrap_or(true) && s.state.active_sequence.is_some();
    let at = time_p(s, p, "").unwrap_or(s.playhead());
    if at < Tick::ZERO {
        return Err(bad(cmd, "the start time cannot be negative"));
    }
    let audio = synth::generate(&q, RATE).map_err(|e| bad(cmd, e.to_string()))?;
    let dur = Tick::from_units(audio.left.len() as i64, i64::from(RATE));
    let range = TimeRange::new(at, dur);
    // the track first, so a busy `track` fails before anything is written
    let chosen = if place {
        let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
        match p.get("track") {
            Some(v) => {
                let id = crate::mixer::strip_ref(seq, v)
                    .filter(|id| seq.audio_tracks.iter().any(|t| t.id == *id))
                    .ok_or_else(|| bad(cmd, format!("no audio track {v}")))?;
                if !track_is_free(s, id, range) {
                    return Err(bad(cmd, format!("audio track {v} is locked or has clips there")));
                }
                Some(id)
            }
            None => seq.audio_tracks.iter().map(|t| t.id).find(|id| track_is_free(s, *id, range)),
        }
    } else {
        None
    };
    let base = label(q.kind);
    let n0 = s.history.undo.len();
    let result = (|| -> Result<Value> {
        let dir = crate::voiceover::generated_dir(s, p, "Generated Audio");
        let path = crate::voiceover::unique_wav_path(s, &dir, base)?;
        if !cfg!(target_arch = "wasm32") {
            std::fs::create_dir_all(&dir).map_err(|e| EngineError::Other(format!("{dir}: {e}")))?;
        }
        let bytes = write_wav_f32_stereo(&audio.left, &audio.right, RATE);
        s.services.write_file(&path, &bytes).map_err(|e| EngineError::Other(format!("{path}: {e}")))?;
        let item: ItemId = crate::commands::import_bytes(s, &path, bytes.into(), None)?;
        let mut out = json!({
            "item": item.0, "path": path, "kind": q.kind.id(), "seconds": audio.seconds(), "seed": q.seed,
            "sampleRate": RATE, "clip": Value::Null, "track": Value::Null,
        });
        if place {
            let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
            let track = match chosen {
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
            let seq = s.project.sequence(seq_id).ok_or(EngineError::NoSequence)?;
            out["clip"] = json!(clip.0);
            out["track"] = json!(crate::mixer::strip_label(seq, track));
            out["start"] = json!(at.0);
            out["duration"] = json!(dur.0);
        }
        Ok(out)
    })();
    crate::clip_ops::collapse_history(s, n0, "Generate Audio");
    if result.is_err() && s.history.undo.len() > n0 {
        let _ = s.undo();
        s.history.redo.clear();
    }
    result
}
