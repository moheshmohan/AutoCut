//! Cloned narration voices: make a voice from someone's recorded speech, then narrate with it
//! (`tts.create {"voice": "clone:<name>"}`) like any other voice.
//!
//! | command | does |
//! |---|---|
//! | `tts.cloneVoice` | take speech from project items / clips (whole, or `segments`), measure the speaker's tone colour, pick the closest Kokoro voice, save the profile `clone:<name>` — a background job; `consent: true` is required |
//! | `tts.clonedVoices` | the saved cloned voices |
//! | `tts.deleteClonedVoice` | remove one |
//! | `tts.downloadCloneModel` | download the OpenVoice V2 converter (131 MB, MIT, SHA-256 checked) as a background job |
//!
//! Profiles live in `<data>/voices/<name>.json`, next to the models: a cloned voice is the
//! user's, usable in every project. See `filmcraft_tts::clone` for how speech is re-voiced.

use serde_json::{Value, json};

use filmcraft_project::{ClipId, ItemId};
use filmcraft_time::Tick;

use crate::commands::{CommandSpec, always, bad, bool_p, str_p, u64_p};
use crate::{EngineError, Result, Session};

/// Rate reference speech is decoded at (the converter's).
const REF_RATE: u32 = 22_050;
/// Most reference audio decoded, seconds (speech is trimmed and capped again in the tts crate).
const MAX_REFERENCE_SECONDS: f64 = 600.0;

fn spec(id: &'static str, label: &'static str, params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>, journal: bool) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: always, run, journal }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            "tts.cloneVoice",
            "Clone a Voice",
            r#"{"name":str,"consent":bool,"items":[id]?,"clips":[id]?,"segments":[{"item":id,"start":seconds,"end":seconds}]?,"baseVoice":"kokoro-…"?,"wait":bool?}"#,
            clone_voice,
            false,
        ),
        spec("tts.clonedVoices", "Cloned Voices", "{}", cloned_voices, false),
        spec("tts.deleteClonedVoice", "Delete Cloned Voice", r#"{"voice":"clone:…"}"#, delete_cloned, false),
        spec("tts.downloadCloneModel", "Download Voice Cloning Model", r#"{"wait":bool?}"#, download_clone_model, false),
    ]
}

/// The saved cloned voices as JSON (empty without the feature or a data directory).
pub fn cloned_json() -> Value {
    #[cfg(feature = "neural-voices")]
    {
        let Some(dir) = crate::transcript::models_dir() else { return json!([]) };
        let list = filmcraft_tts::clone::list(&filmcraft_tts::clone::voices_dir(&dir));
        json!(
            list.iter()
                .map(|p| json!({
                    "id": p.id, "name": p.name, "baseVoice": p.base_voice, "baseSimilarity": p.base_similarity,
                    "referenceSeconds": p.reference_seconds, "source": p.source, "created": p.created,
                    "gender": filmcraft_tts::clone::gender(p),
                }))
                .collect::<Vec<_>>()
        )
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        json!([])
    }
}

fn cloned_voices(_s: &mut Session, _p: &Value) -> Result<Value> {
    let dir = crate::transcript::models_dir();
    Ok(json!({
        "voices": cloned_json(),
        "model": {
            "id": filmcraft_tts::catalog::CLONE_PACKAGE_ID,
            "name": filmcraft_tts::catalog::CLONE_PACKAGE_NAME,
            "installed": dir.as_deref().is_some_and(filmcraft_tts::catalog::clone_installed),
            "missingBytes": dir.as_deref().map_or(0, filmcraft_tts::catalog::clone_missing_bytes),
            "license": filmcraft_tts::catalog::CLONE_LICENSE,
            "source": filmcraft_tts::catalog::CLONE_SOURCE,
            "available": cfg!(feature = "neural-voices"),
        },
    }))
}

fn delete_cloned(_s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "tts.deleteClonedVoice";
    let voice = str_p(p, "voice").ok_or_else(|| bad(cmd, "`voice` is required"))?;
    #[cfg(feature = "neural-voices")]
    {
        let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory"))?;
        filmcraft_tts::clone::delete(&filmcraft_tts::clone::voices_dir(&dir), voice).map_err(|e| bad(cmd, e.to_string()))?;
        Ok(json!({"deleted": voice}))
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        let _ = voice;
        Err(bad(cmd, "voice cloning is not available in this build"))
    }
}

/// Mono audio of `item` from `start` for `len` (seconds) at [`REF_RATE`].
fn decode(s: &Session, item: ItemId, start: f64, len: f64) -> Option<Vec<f32>> {
    let src = s.source(item)?;
    if !src.info().has_audio() || !start.is_finite() || !len.is_finite() || len <= 0.0 {
        return None;
    }
    let rate = i64::from(REF_RATE);
    let from = Tick::from_seconds_f64(start.max(0.0)).to_units_floor(rate).max(0);
    let n = Tick::from_seconds_f64(len.min(MAX_REFERENCE_SECONDS)).to_units_floor(rate).max(0) as usize;
    let buf = src.audio(from, n, REF_RATE).ok()?;
    let mono = filmcraft_speech::downmix(&buf.channels);
    (!mono.is_empty()).then_some(mono)
}

fn item_seconds(s: &Session, item: ItemId) -> Option<f64> {
    match &s.project.item(item)?.kind {
        filmcraft_project::ItemKind::Media(m) => Some(m.duration().seconds()),
        _ => None,
    }
}

/// The reference speech the params point at, and a description of where it came from.
fn reference(s: &Session, p: &Value) -> Result<(Vec<f32>, String)> {
    let cmd = "tts.cloneVoice";
    let mut parts: Vec<(ItemId, f64, f64)> = Vec::new();
    if let Some(list) = p.get("items") {
        let ids = list.as_array().ok_or_else(|| bad(cmd, "`items` must be a list of item ids"))?;
        for v in ids {
            let item = ItemId(v.as_u64().ok_or_else(|| bad(cmd, "`items` must be a list of item ids"))?);
            let secs = item_seconds(s, item).ok_or_else(|| bad(cmd, format!("item {} is not media", item.0)))?;
            parts.push((item, 0.0, secs));
        }
    }
    if let Some(list) = p.get("clips") {
        let ids = list.as_array().ok_or_else(|| bad(cmd, "`clips` must be a list of clip ids"))?;
        let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
        for v in ids {
            let id = ClipId(v.as_u64().ok_or_else(|| bad(cmd, "`clips` must be a list of clip ids"))?);
            let (_, it) = seq.find_item(id).ok_or_else(|| bad(cmd, format!("no clip {}", id.0)))?;
            parts.push((it.item, it.source_in.seconds(), it.duration.seconds() * it.speed.abs().max(1e-6)));
        }
    }
    if let Some(list) = p.get("segments") {
        let segs = list.as_array().ok_or_else(|| bad(cmd, "`segments` must be a list of {item, start, end}"))?;
        for seg in segs {
            let item = ItemId(u64_p(seg, "item").ok_or_else(|| bad(cmd, "a segment needs `item`"))?);
            let (Some(a), Some(b)) = (seg.get("start").and_then(Value::as_f64), seg.get("end").and_then(Value::as_f64)) else {
                return Err(bad(cmd, "a segment needs `start` and `end` seconds"));
            };
            if !(a.is_finite() && b.is_finite() && a >= 0.0 && b > a) {
                return Err(bad(cmd, format!("segment {a}..{b} s makes no sense")));
            }
            parts.push((item, a, b - a));
        }
    }
    if parts.is_empty() {
        return Err(bad(cmd, "say where the voice is: `items`, `clips` or `segments`"));
    }
    // a video clip and its linked audio clip are the same speech: use it once
    let mut seen: Vec<(ItemId, u64, u64)> = Vec::new();
    parts.retain(|(item, a, len)| {
        let key = (*item, a.to_bits(), len.to_bits());
        let new = !seen.contains(&key);
        seen.push(key);
        new
    });
    let mut audio = Vec::new();
    let mut names = Vec::new();
    let mut total = 0.0;
    for (item, start, len) in parts {
        if total >= MAX_REFERENCE_SECONDS {
            break;
        }
        let len = len.min(MAX_REFERENCE_SECONDS - total);
        let Some(a) = decode(s, item, start, len) else {
            return Err(bad(cmd, format!("item {} has no audio there", item.0)));
        };
        total += a.len() as f64 / f64::from(REF_RATE);
        audio.extend_from_slice(&a);
        let name = s.project.item(item).map(|i| i.name.clone()).unwrap_or_default();
        names.push(format!("{name} [{start:.1}–{:.1} s]", start + len));
    }
    Ok((audio, names.join(", ")))
}

fn clone_voice(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "tts.cloneVoice";
    let name = str_p(p, "name").ok_or_else(|| bad(cmd, "`name` is required"))?.to_string();
    if !bool_p(p, "consent").unwrap_or(false) {
        return Err(bad(cmd, "cloning a voice needs consent: pass `consent: true` to confirm it is your own voice or that the speaker agreed"));
    }
    let base = str_p(p, "baseVoice").filter(|b| !b.is_empty() && *b != "auto").map(str::to_string);
    // the reference first: a bad request reads the same with or without the feature
    let (audio, source) = reference(s, p)?;
    #[cfg(feature = "neural-voices")]
    {
        let models = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for voice models"))?;
        if !filmcraft_tts::catalog::installed(&models) {
            return Err(bad(cmd, "the natural voices are not downloaded yet (tts.downloadVoices)"));
        }
        if !filmcraft_tts::catalog::clone_installed(&models) {
            return Err(bad(cmd, "the voice cloning model is not downloaded yet (tts.downloadCloneModel)"));
        }
        let id = format!("{}{}", filmcraft_tts::CLONE_PREFIX, filmcraft_tts::clone::slug(&name));
        if id == filmcraft_tts::CLONE_PREFIX {
            return Err(bad(cmd, "give the voice a name with letters or digits"));
        }
        let wait = bool_p(p, "wait").unwrap_or(false);
        let job = crate::narration::new_job(s, format!("Cloning voice {name}"));
        let (prog, res) = (job.progress.clone(), job.result.clone());
        let job_id = crate::narration::spawn(s, job, wait, "filmcraft-voice-clone", move || {
            let t0 = web_time::Instant::now();
            let mut step = |line: &str| {
                *prog.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = line.to_string();
            };
            let r = filmcraft_tts::clone::create(&name, &audio, REF_RATE, base.as_deref(), &source, true, &models, &mut step)
                .map(|p| format!("Cloned {} through {} (similarity {:.2})", p.id, p.base_voice, p.base_similarity))
                .map_err(|e| e.to_string());
            crate::narration::finish(&prog, &res, r, t0);
        })?;
        Ok(json!({"job": job_id, "voice": id}))
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        let _ = (name, base, audio, source);
        Err(bad(cmd, "voice cloning is not available in this build (built without `neural-voices`)"))
    }
}

fn download_clone_model(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "tts.downloadCloneModel";
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for voice models"))?;
    if filmcraft_tts::catalog::clone_installed(&dir) {
        return Ok(json!({"job": null, "installed": true}));
    }
    if !cfg!(feature = "neural-voices") {
        return Err(bad(cmd, "voice cloning is not available in this build (built without `neural-voices`)"));
    }
    let wait = bool_p(p, "wait").unwrap_or(false);
    let job = crate::narration::new_job(s, format!("Downloading {}", filmcraft_tts::catalog::CLONE_PACKAGE_NAME));
    job.progress.total.store(filmcraft_tts::catalog::clone_missing_bytes(&dir), std::sync::atomic::Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let id = crate::narration::spawn(s, job, wait, "filmcraft-clone-download", move || {
        let t0 = web_time::Instant::now();
        let r = download(&dir, &prog);
        crate::narration::finish(&prog, &res, r.map(|()| "Voice cloning model installed".to_string()), t0);
    })?;
    Ok(json!({"job": id, "installed": false}))
}

#[cfg(feature = "neural-voices")]
fn download(dir: &std::path::Path, prog: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    use std::sync::atomic::Ordering;
    let files: Vec<filmcraft_speech::models::ModelFile> = filmcraft_tts::catalog::CLONE_FILES
        .iter()
        .map(|f| filmcraft_speech::models::ModelFile { name: f.name, url: f.url, sha256: f.sha256, size: f.size })
        .collect();
    let mut progress = |done: u64, total: u64, file: &str| {
        prog.done.store(done, Ordering::Relaxed);
        prog.total.store(total, Ordering::Relaxed);
        *prog.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = format!("Downloading {file}");
        !prog.cancel.load(Ordering::Relaxed)
    };
    filmcraft_speech::models::download_files(&filmcraft_tts::catalog::clone_package_dir(dir), &files, &mut progress).map_err(|e| e.to_string())
}

#[cfg(not(feature = "neural-voices"))]
fn download(_: &std::path::Path, _: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    Err("voice cloning is not available in this build".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo() -> Session {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        s
    }

    #[test]
    fn cloning_needs_consent_a_name_and_a_source() {
        let mut s = demo();
        let e = s.execute("tts.cloneVoice", json!({"name": "Me", "items": [5]})).unwrap_err();
        assert!(e.to_string().contains("consent"), "{e}");
        assert!(s.execute("tts.cloneVoice", json!({"consent": true, "items": [5]})).is_err(), "a name is required");
        let e = s.execute("tts.cloneVoice", json!({"name": "Me", "consent": true})).unwrap_err();
        // without the models installed this build stops earlier, which is also an error
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn reference_validates_segments() {
        let s = demo();
        for p in [
            json!({"segments": [{"item": 5, "start": 2.0, "end": 1.0}]}),
            json!({"segments": [{"item": 5, "start": -1.0, "end": 1.0}]}),
            json!({"segments": [{"item": 5}]}),
            json!({"items": "5"}),
            json!({"clips": [999_999]}),
            json!({}),
        ] {
            assert!(reference(&s, &p).is_err(), "{p}");
        }
    }

    #[test]
    fn listing_works_without_profiles() {
        let mut s = demo();
        let r = s.execute("tts.clonedVoices", json!({})).unwrap();
        assert!(r["voices"].is_array());
        assert_eq!(r["model"]["id"], "openvoice-v2");
        assert!(s.execute("tts.deleteClonedVoice", json!({})).is_err());
    }
}
