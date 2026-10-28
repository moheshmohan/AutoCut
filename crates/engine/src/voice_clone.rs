//! Cloned narration voices: make a voice from someone's recorded speech, then narrate with it
//! (`tts.create {"voice": "clone:<name>"}`) like any other voice.
//!
//! | command | does |
//! |---|---|
//! | `tts.cloneVoice` | take speech from project items / clips (whole, or `segments`) and save the profile `clone:<name>` — a background job; `consent: true` is required. `engine`: `voxcpm` (natural: the model speaks in the speaker's own voice, accent and rhythm included; the default when a VoxCPM release is installed) or `openvoice` (timbre only: the closest Kokoro voice re-voiced). Natural clones keep the best 10–25 s of speech as a voice sample, transcribed with the speech model when the mode needs it (`transcript` gives it instead); `model`, `mode`, `cfg` and `steps` tune them |
//! | `tts.clonedVoices` | the saved cloned voices, and the cloning models with their download state |
//! | `tts.deleteClonedVoice` | remove one (and its voice sample) |
//! | `tts.downloadCloneModel` | download a cloning model as a background job, every file SHA-256 checked: `openvoice-v2` (131 MB, MIT; the default), `voxcpm-1.5` (1.95 GB) or `voxcpm-2` (4.96 GB), both Apache-2.0, or `indicf5` (1.40 GB, MIT: cloned voices speaking Malayalam) |
//! | `tts.voiceReference` | a natural clone's Malayalam reference: the 10–15 s of its voice sample IndicF5 speaks from and their words (heard by the speech model once, or given in `transcript`); `redo` makes it again; `start` / `end` / `trimToSentence` cut the clip already made — `trimToSentence` ends it at the quietest point near the last sentence end its words imply (at least 8 s; a clip that stops mid-sentence garbles the first words of every line), the words cut to match unless `transcript` gives them; the clip before the first trim is kept as `<name>.indicf5.untrimmed.wav` |
//!
//! Profiles live in `<data>/voices/<name>.json` (natural ones with `<name>.wav`), next to the
//! models: a cloned voice is the user's, usable in every project. See `filmcraft_tts::clone`.

use serde_json::{Value, json};

use filmcraft_project::{ClipId, ItemId};
use filmcraft_time::Tick;

use crate::commands::{CommandSpec, always, bad, bool_p, str_p, u64_p};
use crate::{EngineError, Result, Session};

/// Rate reference speech is decoded at for the OpenVoice converter.
const REF_RATE: u32 = 22_050;
/// Rate reference speech is decoded at for VoxCPM (VoxCPM1.5 encodes at 44.1 kHz: the voice
/// sample keeps the recording's top octave).
const NATURAL_REF_RATE: u32 = 44_100;
/// Most reference audio decoded, seconds (speech is trimmed and capped again in the tts crate).
const MAX_REFERENCE_SECONDS: f64 = 600.0;
/// The timbre-only cloning model.
const OPENVOICE: &str = "openvoice-v2";

fn spec(id: &'static str, label: &'static str, params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>, journal: bool) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: always, run, journal }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            "tts.cloneVoice",
            "Clone a Voice",
            r#"{"name":str,"consent":bool,"items":[id]?,"clips":[id]?,"segments":[{"item":id,"start":seconds,"end":seconds}]?,"engine":"voxcpm"|"openvoice"?,"model":"voxcpm-1.5"|"voxcpm-2"?,"mode":"reference"|"continuation"?,"transcript":str?,"speechModel":"whisper-…"?,"cfg":1..5?,"steps":2..50?,"baseVoice":"kokoro-…"?,"wait":bool?}"#,
            clone_voice,
            false,
        ),
        spec("tts.clonedVoices", "Cloned Voices", r#"{"voice":"clone:…"? (one voice, with its sample's pitchHz)}"#, cloned_voices, false),
        spec("tts.deleteClonedVoice", "Delete Cloned Voice", r#"{"voice":"clone:…"}"#, delete_cloned, false),
        spec(
            "tts.downloadCloneModel",
            "Download Voice Cloning Model",
            r#"{"model":"openvoice-v2"|"voxcpm-1.5"|"voxcpm-2"|"indicf5"?,"wait":bool?}"#,
            download_clone_model,
            false,
        ),
        spec(
            "tts.voiceReference",
            "Voice Reference for Malayalam",
            r#"{"voice":"clone:…","items":[id]?,"clips":[id]?,"segments":[{"item":id,"start":seconds,"end":seconds}]? (a recording of the speaker: its best 10–15 s),"transcript":str? (the clip's words),"redo":bool?,"speechModel":"whisper-…"?,"start":seconds?,"end":seconds? (cut the clip already made, seconds within it),"trimToSentence":bool? (end it at the last sentence end its words imply, at least 8 s)}"#,
            voice_reference,
            false,
        ),
    ]
}

/// The Malayalam reference of a natural cloned voice: 10–15 s of the speaker IndicF5 speaks
/// from, and their words. Shown when made. Made (or made again with `redo`) from a recording
/// given as `items` / `clips` / `segments` (full band: the best source), else from the voice's
/// own recording; the words heard by the speech model (English) unless given. `transcript`
/// alone corrects the words of the clip already made.
fn voice_reference(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "tts.voiceReference";
    let voice = str_p(p, "voice").ok_or_else(|| bad(cmd, "`voice` is required (clone:<name>)"))?.to_string();
    if !voice.starts_with(filmcraft_tts::CLONE_PREFIX) {
        return Err(bad(cmd, "`voice` must be a cloned voice (clone:<name>)"));
    }
    if p.get("transcript").is_some_and(|t| !t.is_string()) {
        return Err(bad(cmd, "`transcript` must be a string"));
    }
    // a trim of the clip already made: `start` / `end` (seconds within it), `trimToSentence`
    let mut trim = ReferenceTrim::default();
    for (key, slot) in [("start", &mut trim.start), ("end", &mut trim.end)] {
        match p.get(key) {
            None | Some(Value::Null) => {}
            Some(v) => match v.as_f64().filter(|t| t.is_finite() && *t >= 0.0 && *t <= 3600.0) {
                Some(t) => *slot = Some(t),
                None => return Err(bad(cmd, format!("`{key}` must be seconds within the reference clip"))),
            },
        }
    }
    match p.get("trimToSentence") {
        None | Some(Value::Null) => {}
        Some(Value::Bool(b)) => trim.to_sentence = *b,
        Some(_) => return Err(bad(cmd, "`trimToSentence` must be true or false")),
    }
    let trimming = trim.start.is_some() || trim.end.is_some() || trim.to_sentence;
    if trimming && (["items", "clips", "segments"].iter().any(|k| p.get(*k).is_some()) || bool_p(p, "redo").unwrap_or(false)) {
        return Err(bad(cmd, "a trim cuts the reference already made: give `start`/`end`/`trimToSentence` without a recording or `redo`"));
    }
    #[cfg(feature = "neural-voices")]
    {
        let models = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for voice models"))?;
        let transcript = str_p(p, "transcript").map(str::to_string);
        let check_words = match p.get("checkTranscript") {
            None | Some(Value::Null) => true,
            Some(Value::Bool(b)) => *b,
            Some(_) => return Err(bad(cmd, "`checkTranscript` must be true or false")),
        };
        if trimming {
            let t = filmcraft_tts::clone::Trim { start: trim.start, end: trim.end, to_sentence: trim.to_sentence, transcript };
            let r = filmcraft_tts::clone::trim_cross_reference(&models, &voice, &t).map_err(|e| bad(cmd, e.to_string()))?;
            // readings kept in memory were read after the clip as it was
            crate::narration::forget_readings(s);
            let checked = if check_words { transcript_check(s, &models, &r) } else { Value::Null };
            return Ok(
                json!({"voice": voice, "source": r.source, "from": r.from, "to": r.to, "seconds": r.to - r.from, "transcript": r.transcript, "given": r.given, "clip": r.clip, "trimmed": true, "transcriptCheck": checked}),
            );
        }
        let sourced = ["items", "clips", "segments"].iter().any(|k| p.get(*k).is_some());
        // a recording given: decoded at the natural clones' rate (full band)
        let source = if sourced { Some(reference(s, p, NATURAL_REF_RATE)?) } else { None };
        let redo = bool_p(p, "redo").unwrap_or(false) || sourced;
        let current = filmcraft_tts::clone::cross_reference(&models, &voice).map_err(|e| bad(cmd, e.to_string()))?;
        let r = match (current, transcript.as_deref()) {
            // the words of the clip already made, corrected
            (Some(_), Some(t)) if !redo => filmcraft_tts::clone::set_cross_transcript(&models, &voice, t).map_err(|e| bad(cmd, e.to_string()))?,
            (Some(r), None) if !redo => r,
            _ => {
                let transcriber = if transcript.is_none() {
                    let speech = match str_p(p, "speechModel") {
                        Some(m) => json!({"model": m}),
                        None => json!({}),
                    };
                    Some(crate::transcript::transcriber(s, &speech).map_err(|e| bad(cmd, format!("{e}; or give the clip's words in `transcript`")))?)
                } else {
                    None
                };
                let mut hear = |x16: &[f32]| -> std::result::Result<String, String> {
                    let t = transcriber.as_ref().ok_or_else(|| "no speech model".to_string())?;
                    let opts = filmcraft_speech::Options { language: Some("en".into()), diarize: false, ..Default::default() };
                    t.transcribe(x16, &opts, &mut |_, _| true).map(|tr| tr.text()).map_err(|e| e.to_string())
                };
                let src = source.as_ref().map(|(audio, what)| filmcraft_tts::clone::CrossSource { audio, rate: NATURAL_REF_RATE, source: what });
                filmcraft_tts::clone::prepare_cross_lingual(&models, &voice, src, transcript.as_deref(), &mut hear).map_err(|e| bad(cmd, e.to_string()))?
            }
        };
        crate::narration::forget_readings(s);
        let checked = if check_words { transcript_check(s, &models, &r) } else { Value::Null };
        Ok(
            json!({"voice": voice, "source": r.source, "from": r.from, "to": r.to, "transcript": r.transcript, "given": r.given, "clip": r.clip, "transcriptCheck": checked}),
        )
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        let _ = (s, trimming);
        Err(bad(cmd, "voice cloning is not available in this build (built without `neural-voices`)"))
    }
}

/// Highest character error rate between a Malayalam reference's words and what the Malayalam
/// speech model hears in its clip that still counts as the clip's words. The author's own clip
/// with its hand-typed words scores 0.14 (a filler, "ഏഹ്", is not heard, and loanwords are
/// spelled a little differently); words that run past the cut, or belong to another stretch,
/// score far higher.
pub const TRANSCRIPT_MATCH_CER: f64 = 0.3;

/// Whether the words of `r` are what its clip says (IndicF5 aligns its reading to them: words
/// that do not match the clip garble the start of every line): the clip heard by the Malayalam
/// speech model and scored against them. Only words written in Malayalam are checked.
#[cfg(feature = "neural-voices")]
pub(crate) fn transcript_check(s: &Session, models: &std::path::Path, r: &filmcraft_tts::clone::CrossReference) -> Value {
    if !filmcraft_tts::malayalam::detect(&r.transcript) {
        return json!({"status": "skipped", "reason": "the words are not written in Malayalam"});
    }
    let audio = match filmcraft_tts::clone::cross_audio(models, r) {
        Ok(a) => a,
        Err(e) => return json!({"status": "skipped", "reason": e.to_string()}),
    };
    match crate::narration::hear_malayalam(s, Some(models), &audio, filmcraft_tts::clone::CROSS_REFERENCE_RATE) {
        Ok((heard, model)) => {
            let sc = filmcraft_tts::heard::score(&r.transcript, &heard);
            let matches = sc.cer <= TRANSCRIPT_MATCH_CER;
            let mut v = json!({"status": if matches { "matches" } else { "mismatch" }, "model": model, "heard": heard, "cer": sc.cer});
            if !matches {
                v["warning"] = json!(format!(
                    "the words do not match what the clip says (character error rate {:.2}, at most {TRANSCRIPT_MATCH_CER} expected): IndicF5 aligns its reading to them, so give the exact words of the clip (`transcript`) or trim the clip to them",
                    sc.cer
                ));
            }
            v
        }
        Err(why) => json!({"status": "skipped", "reason": why}),
    }
}

/// `tts.voiceReference`'s trim parameters, checked before anything is touched.
#[derive(Default)]
struct ReferenceTrim {
    start: Option<f64>,
    end: Option<f64>,
    to_sentence: bool,
}

/// The saved cloned voices as JSON (empty without the feature or a data directory).
pub fn cloned_json() -> Value {
    #[cfg(feature = "neural-voices")]
    {
        let Some(dir) = crate::transcript::models_dir() else { return json!([]) };
        let voices = filmcraft_tts::clone::voices_dir(&dir);
        let list = filmcraft_tts::clone::list(&voices);
        json!(
            list.iter()
                .map(|p| {
                    let mut v = json!({
                        "id": p.id, "name": p.name, "engine": p.engine.id(), "referenceSeconds": p.reference_seconds,
                        "source": p.source, "created": p.created, "gender": filmcraft_tts::clone::gender(p),
                    });
                    match &p.natural {
                        Some(n) => {
                            v["model"] = json!(n.model);
                            v["modelName"] = json!(filmcraft_tts::catalog::voxcpm(&n.model).map_or("", |m| m.name));
                            v["mode"] = json!(n.mode);
                            v["cfg"] = json!(n.cfg);
                            v["steps"] = json!(n.timesteps);
                            v["transcript"] = json!(n.transcript);
                            v["sample"] = json!({"from": n.from, "to": n.to});
                            // the Malayalam voice styles it keeps references for (tts.voiceStyles)
                            v["styles"] = json!(filmcraft_tts::clone::styles_on_disk(&voices, p));
                        }
                        None => {
                            v["model"] = json!(OPENVOICE);
                            v["baseVoice"] = json!(p.base_voice);
                            v["baseSimilarity"] = json!(p.base_similarity);
                        }
                    }
                    v
                })
                .collect::<Vec<_>>()
        )
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        json!([])
    }
}

/// The cloning models: id, engine, name, download state, size, licence and source.
pub fn models_json() -> Value {
    let dir = crate::transcript::models_dir();
    let mut out = vec![json!({
        "id": OPENVOICE,
        "engine": "openvoice",
        "name": filmcraft_tts::catalog::CLONE_PACKAGE_NAME,
        "description": "Timbre only: a Kokoro voice re-voiced into the speaker's tone colour. Fast; the accent and rhythm stay Kokoro's.",
        "installed": dir.as_deref().is_some_and(filmcraft_tts::catalog::clone_installed),
        "size": filmcraft_tts::catalog::CLONE_FILES.iter().map(|f| f.size).sum::<u64>(),
        "missingBytes": dir.as_deref().map_or(0, filmcraft_tts::catalog::clone_missing_bytes),
        "license": filmcraft_tts::catalog::CLONE_LICENSE,
        "source": filmcraft_tts::catalog::CLONE_SOURCE,
        "available": cfg!(feature = "neural-voices"),
    })];
    for (m, engine) in filmcraft_tts::catalog::VOXCPM.iter().map(|m| (m, "voxcpm")).chain(std::iter::once((&filmcraft_tts::catalog::INDICF5, "indicf5"))) {
        out.push(json!({
            "id": m.id,
            "engine": engine,
            "name": m.name,
            "description": m.description,
            "installed": dir.as_deref().is_some_and(|d| filmcraft_tts::catalog::voxcpm_installed(d, m)),
            "size": filmcraft_tts::catalog::voxcpm_size(m),
            "missingBytes": dir.as_deref().map_or(filmcraft_tts::catalog::voxcpm_size(m), |d| filmcraft_tts::catalog::voxcpm_missing_bytes(d, m)),
            "license": m.license,
            "source": m.source,
            "available": cfg!(feature = "neural-voices"),
        }));
    }
    json!(out)
}

/// The VoxCPM release a natural clone uses when none is named: the default if installed, else
/// any installed one, else the default.
pub fn default_natural_model() -> &'static str {
    let dir = crate::transcript::models_dir();
    let installed =
        |id: &str| dir.as_deref().is_some_and(|d| filmcraft_tts::catalog::voxcpm(id).is_some_and(|m| filmcraft_tts::catalog::voxcpm_installed(d, m)));
    if installed(filmcraft_tts::catalog::DEFAULT_VOXCPM) {
        return filmcraft_tts::catalog::DEFAULT_VOXCPM;
    }
    filmcraft_tts::catalog::VOXCPM.iter().map(|m| m.id).find(|id| installed(id)).unwrap_or(filmcraft_tts::catalog::DEFAULT_VOXCPM)
}

/// The engine new clones use: natural when this build has it and a VoxCPM release is installed.
pub fn default_engine() -> &'static str {
    let dir = crate::transcript::models_dir();
    let natural = cfg!(feature = "neural-voices")
        && dir.as_deref().is_some_and(|d| filmcraft_tts::catalog::VOXCPM.iter().any(|m| filmcraft_tts::catalog::voxcpm_installed(d, m)));
    if natural { "voxcpm" } else { "openvoice" }
}

fn cloned_voices(_s: &mut Session, p: &Value) -> Result<Value> {
    let dir = crate::transcript::models_dir();
    // one voice, with the pitch of its voice sample (measured: not for every listing)
    if let Some(id) = str_p(p, "voice") {
        let mut v = cloned_json()
            .as_array()
            .and_then(|a| a.iter().find(|v| v["id"] == id).cloned())
            .ok_or_else(|| bad("tts.clonedVoices", format!("no cloned voice `{id}`")))?;
        v["pitchHz"] = json!(sample_pitch(id));
        return Ok(v);
    }
    Ok(json!({
        "voices": cloned_json(),
        "models": models_json(),
        "defaultEngine": default_engine(),
        "defaultModel": default_natural_model(),
        // the timbre-only model, as before natural cloning existed
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

/// The median pitch of a natural cloned voice's sample, Hz.
fn sample_pitch(id: &str) -> Option<f64> {
    #[cfg(feature = "neural-voices")]
    {
        let dir = crate::transcript::models_dir()?;
        let voices = filmcraft_tts::clone::voices_dir(&dir);
        let profile = filmcraft_tts::clone::list(&voices).into_iter().find(|p| p.id == id)?;
        let clip = profile.natural.as_ref()?.clip.clone();
        let (x, rate) = filmcraft_tts::wav::read(&voices.join(clip)).ok()?;
        filmcraft_tts::clone::median_pitch(&x, rate)
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        let _ = id;
        None
    }
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

/// Mono audio of `item` from `start` for `len` (seconds) at `rate`.
fn decode(s: &Session, item: ItemId, start: f64, len: f64, rate: u32) -> Option<Vec<f32>> {
    let src = s.source(item)?;
    if !src.info().has_audio() || !start.is_finite() || !len.is_finite() || len <= 0.0 {
        return None;
    }
    let r = i64::from(rate);
    let from = Tick::from_seconds_f64(start.max(0.0)).to_units_floor(r).max(0);
    let n = Tick::from_seconds_f64(len.min(MAX_REFERENCE_SECONDS)).to_units_floor(r).max(0) as usize;
    let buf = src.audio(from, n, rate).ok()?;
    let mono = filmcraft_speech::downmix(&buf.channels);
    (!mono.is_empty()).then_some(mono)
}

fn item_seconds(s: &Session, item: ItemId) -> Option<f64> {
    match &s.project.item(item)?.kind {
        filmcraft_project::ItemKind::Media(m) => Some(m.duration().seconds()),
        _ => None,
    }
}

/// The reference speech the params point at (mono at `rate`), and a description of where it
/// came from.
fn reference(s: &Session, p: &Value, rate: u32) -> Result<(Vec<f32>, String)> {
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
        let Some(a) = decode(s, item, start, len, rate) else {
            return Err(bad(cmd, format!("item {} has no audio there", item.0)));
        };
        total += a.len() as f64 / f64::from(rate);
        audio.extend_from_slice(&a);
        let name = s.project.item(item).map(|i| i.name.clone()).unwrap_or_default();
        names.push(format!("{name} [{start:.1}–{:.1} s]", start + len));
    }
    Ok((audio, names.join(", ")))
}

/// What a natural clone request asks for (checked before anything is decoded).
#[cfg_attr(not(feature = "neural-voices"), allow(dead_code))]
struct NaturalParams {
    model: String,
    mode: Option<&'static str>,
    transcript: Option<String>,
    cfg: f32,
    steps: usize,
}

fn natural_params(p: &Value) -> Result<NaturalParams> {
    let cmd = "tts.cloneVoice";
    let model = match str_p(p, "model") {
        Some(m) if filmcraft_tts::catalog::voxcpm(m).is_some() => m.to_string(),
        Some(m) => return Err(bad(cmd, format!("unknown model `{m}` (voxcpm-1.5, voxcpm-2)"))),
        None => default_natural_model().to_string(),
    };
    let mode = match str_p(p, "mode") {
        None => None,
        Some("reference") => Some("reference"),
        Some("continuation") => Some("continuation"),
        Some(m) => return Err(bad(cmd, format!("unknown mode `{m}` (reference, continuation)"))),
    };
    if p.get("transcript").is_some_and(|t| !t.is_string()) {
        return Err(bad(cmd, "`transcript` must be a string"));
    }
    let transcript = str_p(p, "transcript").map(str::trim).filter(|t| !t.is_empty()).map(str::to_string);
    if transcript.as_ref().is_some_and(|t| t.len() > 16 * 1024) {
        return Err(bad(cmd, "`transcript` is too long"));
    }
    let cfg = match p.get("cfg") {
        None => 2.0,
        Some(v) => v.as_f64().filter(|c| c.is_finite() && (1.0..=5.0).contains(c)).ok_or_else(|| bad(cmd, "`cfg` must be a number from 1 to 5"))?,
    };
    let steps = match p.get("steps") {
        None => 10,
        Some(v) => v.as_u64().filter(|n| (2..=50).contains(n)).ok_or_else(|| bad(cmd, "`steps` must be a whole number from 2 to 50"))?,
    };
    Ok(NaturalParams { model, mode, transcript, cfg: cfg as f32, steps: steps as usize })
}

fn clone_voice(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "tts.cloneVoice";
    let name = str_p(p, "name").ok_or_else(|| bad(cmd, "`name` is required"))?.to_string();
    if !bool_p(p, "consent").unwrap_or(false) {
        return Err(bad(cmd, "cloning a voice needs consent: pass `consent: true` to confirm it is your own voice or that the speaker agreed"));
    }
    let engine = match str_p(p, "engine") {
        Some(e @ ("voxcpm" | "openvoice")) => e,
        Some(e) => return Err(bad(cmd, format!("unknown engine `{e}` (voxcpm, openvoice)"))),
        None => default_engine(),
    };
    let natural = if engine == "voxcpm" { Some(natural_params(p)?) } else { None };
    let base = str_p(p, "baseVoice").filter(|b| !b.is_empty() && *b != "auto").map(str::to_string);
    // the reference first: a bad request reads the same with or without the feature
    let rate = if natural.is_some() { NATURAL_REF_RATE } else { REF_RATE };
    let (audio, source) = reference(s, p, rate)?;
    #[cfg(feature = "neural-voices")]
    {
        let models = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for voice models"))?;
        let id = format!("{}{}", filmcraft_tts::CLONE_PREFIX, filmcraft_tts::clone::slug(&name));
        if id == filmcraft_tts::CLONE_PREFIX {
            return Err(bad(cmd, "give the voice a name with letters or digits"));
        }
        let wait = bool_p(p, "wait").unwrap_or(false);
        match natural {
            Some(np) => {
                let pkg = filmcraft_tts::catalog::voxcpm(&np.model).ok_or_else(|| bad(cmd, "unknown model"))?;
                if !filmcraft_tts::catalog::voxcpm_installed(&models, pkg) {
                    return Err(bad(
                        cmd,
                        format!("the cloning model {} is not downloaded yet (tts.downloadCloneModel {{\"model\": \"{}\"}})", pkg.name, pkg.id),
                    ));
                }
                let mode = np
                    .mode
                    .map(|m| if m == "continuation" { filmcraft_tts::clone::PromptMode::Continuation } else { filmcraft_tts::clone::PromptMode::Reference });
                let needs_transcript = pkg.id == "voxcpm-1.5" || mode == Some(filmcraft_tts::clone::PromptMode::Continuation);
                // the speech model, when the sample must be transcribed (found now: a request
                // that cannot be served fails before the job starts)
                let transcriber = if needs_transcript && np.transcript.is_none() {
                    // `model` here names the cloning model; the speech model is `speechModel`
                    // (else the one chosen in Settings)
                    let speech = match str_p(p, "speechModel") {
                        Some(m) => json!({"model": m}),
                        None => json!({}),
                    };
                    Some(crate::transcript::transcriber(s, &speech).map_err(|e| bad(cmd, format!("{e}; or pass the voice sample's `transcript`")))?)
                } else {
                    None
                };
                let req =
                    filmcraft_tts::clone::NaturalRequest { model: np.model.clone(), mode, transcript: np.transcript.clone(), cfg: np.cfg, timesteps: np.steps };
                let job = crate::narration::new_job(s, format!("Cloning voice {name}"));
                let (prog, res) = (job.progress.clone(), job.result.clone());
                let job_id = crate::narration::spawn(s, job, wait, "filmcraft-voice-clone", move || {
                    let t0 = web_time::Instant::now();
                    let mut step = |line: &str| {
                        *prog.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = line.to_string();
                    };
                    let mut transcribe = |x16: &[f32]| -> std::result::Result<String, String> {
                        let t = transcriber.as_ref().ok_or_else(|| "no speech model".to_string())?;
                        let opts = filmcraft_speech::Options { language: Some("en".into()), diarize: false, ..Default::default() };
                        t.transcribe(x16, &opts, &mut |_, _| true).map(|tr| tr.text()).map_err(|e| e.to_string())
                    };
                    let r = filmcraft_tts::clone::create_natural(&name, &audio, NATURAL_REF_RATE, &req, &source, true, &models, &mut transcribe, &mut step)
                        .map(|p| format!("Cloned {} with {} ({:.1} s voice sample)", p.id, p.model().unwrap_or(""), p.reference_seconds))
                        .map_err(|e| e.to_string());
                    crate::narration::finish(&prog, &res, r, t0);
                })?;
                Ok(json!({"job": job_id, "voice": id, "engine": "voxcpm", "model": np.model}))
            }
            None => {
                if !filmcraft_tts::catalog::installed(&models) {
                    return Err(bad(cmd, "the natural voices are not downloaded yet (tts.downloadVoices)"));
                }
                if !filmcraft_tts::catalog::clone_installed(&models) {
                    return Err(bad(cmd, "the voice cloning model is not downloaded yet (tts.downloadCloneModel)"));
                }
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
                Ok(json!({"job": job_id, "voice": id, "engine": "openvoice"}))
            }
        }
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        let _ = (name, base, audio, source, natural);
        Err(bad(cmd, "voice cloning is not available in this build (built without `neural-voices`)"))
    }
}

/// A cloning model's files, pinned.
fn model_files(model: &str) -> Option<(&'static str, &'static [filmcraft_tts::catalog::ModelFile])> {
    if model == OPENVOICE {
        return Some((filmcraft_tts::catalog::CLONE_PACKAGE_NAME, filmcraft_tts::catalog::CLONE_FILES));
    }
    filmcraft_tts::catalog::cloning_model(model).map(|m| (m.name, m.files))
}

fn download_clone_model(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "tts.downloadCloneModel";
    let model = str_p(p, "model").unwrap_or(OPENVOICE).to_string();
    let (name, files) = model_files(&model).ok_or_else(|| bad(cmd, format!("unknown model `{model}` (openvoice-v2, voxcpm-1.5, voxcpm-2, indicf5)")))?;
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for voice models"))?;
    let package = if model == OPENVOICE { filmcraft_tts::catalog::clone_package_dir(&dir) } else { dir.join(&model) };
    let missing: u64 = files.iter().filter(|f| !std::fs::metadata(package.join(f.name)).is_ok_and(|m| m.len() == f.size)).map(|f| f.size).sum();
    if missing == 0 {
        return Ok(json!({"job": null, "installed": true, "model": model}));
    }
    if !cfg!(feature = "neural-voices") {
        return Err(bad(cmd, "voice cloning is not available in this build (built without `neural-voices`)"));
    }
    let wait = bool_p(p, "wait").unwrap_or(false);
    let job = crate::narration::new_job(s, format!("Downloading {name}"));
    job.progress.total.store(missing, std::sync::atomic::Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let done = format!("{name} installed");
    let id = crate::narration::spawn(s, job, wait, "filmcraft-clone-download", move || {
        let t0 = web_time::Instant::now();
        let r = download(&package, files, &prog);
        crate::narration::finish(&prog, &res, r.map(|()| done), t0);
    })?;
    Ok(json!({"job": id, "installed": false, "model": model}))
}

#[cfg(feature = "neural-voices")]
fn download(
    package: &std::path::Path,
    files: &'static [filmcraft_tts::catalog::ModelFile],
    prog: &filmcraft_export::Progress,
) -> std::result::Result<(), String> {
    use std::sync::atomic::Ordering;
    let files: Vec<filmcraft_speech::models::ModelFile> =
        files.iter().map(|f| filmcraft_speech::models::ModelFile { name: f.name, url: f.url, sha256: f.sha256, size: f.size }).collect();
    let mut progress = |done: u64, total: u64, file: &str| {
        prog.done.store(done, Ordering::Relaxed);
        prog.total.store(total, Ordering::Relaxed);
        *prog.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = format!("Downloading {file}");
        !prog.cancel.load(Ordering::Relaxed)
    };
    filmcraft_speech::models::download_files(package, &files, &mut progress).map_err(|e| e.to_string())
}

#[cfg(not(feature = "neural-voices"))]
fn download(_: &std::path::Path, _: &'static [filmcraft_tts::catalog::ModelFile], _: &filmcraft_export::Progress) -> std::result::Result<(), String> {
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
    fn hostile_clone_parameters_are_refused() {
        let mut s = demo();
        for p in [
            json!({"name": "Me", "consent": true, "items": [5], "engine": "nope"}),
            json!({"name": "Me", "consent": true, "items": [5], "engine": "voxcpm", "model": "voxcpm-9"}),
            json!({"name": "Me", "consent": true, "items": [5], "engine": "voxcpm", "mode": "sideways"}),
            json!({"name": "Me", "consent": true, "items": [5], "engine": "voxcpm", "cfg": 50}),
            json!({"name": "Me", "consent": true, "items": [5], "engine": "voxcpm", "cfg": "high"}),
            json!({"name": "Me", "consent": true, "items": [5], "engine": "voxcpm", "steps": 0}),
            json!({"name": "Me", "consent": true, "items": [5], "engine": "voxcpm", "steps": 1e9}),
            json!({"name": "Me", "consent": true, "items": [5], "engine": "voxcpm", "transcript": 5}),
            json!({"name": "Me", "consent": true, "items": [5], "engine": "voxcpm", "transcript": "x".repeat(20_000)}),
        ] {
            let e = s.execute("tts.cloneVoice", p.clone()).unwrap_err();
            assert!(!e.to_string().is_empty(), "{p}");
        }
        assert!(s.execute("tts.downloadCloneModel", json!({"model": "voxcpm-9"})).is_err());
    }

    #[test]
    fn natural_params_have_safe_defaults() {
        let n = natural_params(&json!({})).unwrap();
        assert_eq!((n.cfg, n.steps, n.mode, n.transcript), (2.0, 10, None, None));
        assert!(filmcraft_tts::catalog::voxcpm(&n.model).is_some());
        let n = natural_params(&json!({"model": "voxcpm-2", "mode": "continuation", "transcript": "  hi  ", "cfg": 3, "steps": 6})).unwrap();
        assert_eq!((n.model.as_str(), n.mode, n.transcript.as_deref(), n.cfg, n.steps), ("voxcpm-2", Some("continuation"), Some("hi"), 3.0, 6));
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
            assert!(reference(&s, &p, REF_RATE).is_err(), "{p}");
        }
    }

    #[test]
    fn listing_works_without_profiles() {
        let mut s = demo();
        let r = s.execute("tts.clonedVoices", json!({})).unwrap();
        assert!(r["voices"].is_array());
        assert_eq!(r["model"]["id"], "openvoice-v2");
        let models = r["models"].as_array().unwrap();
        // OpenVoice, the VoxCPM releases and IndicF5 (Malayalam)
        assert_eq!(models.len(), 1 + filmcraft_tts::catalog::VOXCPM.len() + 1);
        assert!(models.iter().any(|m| m["id"] == "indicf5"));
        for m in models {
            assert!(m["license"].as_str().is_some_and(|l| !l.contains("NC")), "{m}");
            assert!(m["size"].as_u64().is_some_and(|b| b > 0), "{m}");
        }
        assert!(matches!(r["defaultEngine"].as_str(), Some("voxcpm" | "openvoice")));
        assert!(s.execute("tts.deleteClonedVoice", json!({})).is_err());
    }

    /// End to end with real models (gated): VoxCPM speaks a test sentence, zero-shot and — with
    /// `FILMCRAFT_VOXCPM_PROMPT` (a WAV of the speaker; `FILMCRAFT_VOXCPM_PROMPT_TEXT` its exact
    /// transcript, needed by VoxCPM1.5) — in a cloned voice; Whisper must hear the sentence
    /// (WER ≤ 20 %), and the clone's OpenVoice tone colour must be close to the speaker's
    /// (cosine ≥ 0.6, with `FILMCRAFT_OPENVOICE_DIR`).
    ///
    /// `FILMCRAFT_VOXCPM_DIR=<release> FILMCRAFT_WHISPER_DIR=<whisper-small> cargo test --release -p
    /// filmcraft-engine --features neural-voices,whisper live_natural_clone -- --nocapture`
    #[cfg(all(feature = "neural-voices", feature = "whisper"))]
    #[test]
    fn live_natural_clone_is_intelligible_and_similar() {
        use filmcraft_speech::Transcriber;
        use filmcraft_tts::voxcpm::{Settings, VoxCpm, text};
        let (Some(vox), Some(wh)) = (std::env::var_os("FILMCRAFT_VOXCPM_DIR"), std::env::var_os("FILMCRAFT_WHISPER_DIR")) else {
            eprintln!("skipped (FILMCRAFT_VOXCPM_DIR or FILMCRAFT_WHISPER_DIR unset)");
            return;
        };
        let m = VoxCpm::load(std::path::Path::new(&vox)).unwrap();
        let wh = std::path::PathBuf::from(wh);
        let id = wh.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let whisper = filmcraft_speech::whisper::Whisper::load(&wh, &id).unwrap();
        let sentence = "In October 2022 I met Spot, the yellow robot dog from Boston Dynamics, walking through the crowd at GITEX in Dubai.";
        let speak = |prompt: Option<&filmcraft_tts::voxcpm::Prompt>| -> Vec<f32> {
            let mut out = Vec::new();
            for p in text::pieces(sentence).unwrap() {
                if let text::Piece::Say(u) = p {
                    out.extend(m.speak(&u, prompt, &Settings::default(), &mut |_, _| true).unwrap());
                }
            }
            out
        };
        let hear = |x: &[f32]| -> f64 {
            let x16 = filmcraft_audio_dsp::resample::resample_sinc(x, m.output_rate(), 16_000);
            let opts = filmcraft_speech::Options { language: Some("en".into()), diarize: false, ..Default::default() };
            let heard = whisper.transcribe(&x16, &opts, &mut |_, _| true).unwrap().text();
            let wer = filmcraft_speech::word_error_rate(sentence, &heard);
            eprintln!("heard ({:.0} % WER): {heard}", wer * 100.0);
            wer
        };
        let t0 = web_time::Instant::now();
        let zero = speak(None);
        eprintln!("zero-shot: {:.1} s of audio in {:.1?}", zero.len() as f64 / f64::from(m.output_rate()), t0.elapsed());
        assert!(hear(&zero) <= 0.2);
        let Some(wav) = std::env::var_os("FILMCRAFT_VOXCPM_PROMPT") else { return };
        let (x, rate) = filmcraft_tts::wav::read(std::path::Path::new(&wav)).unwrap();
        let transcript = std::env::var("FILMCRAFT_VOXCPM_PROMPT_TEXT").ok().map(|t| text::sentences(&t).unwrap().join(" "));
        let xr = filmcraft_audio_dsp::resample::resample_sinc(&x, rate, m.encode_rate());
        let prompt = m.prompt(&xr, transcript.as_deref()).unwrap();
        let t0 = web_time::Instant::now();
        let cloned = speak(Some(&prompt));
        eprintln!("cloned: {:.1} s of audio in {:.1?}", cloned.len() as f64 / f64::from(m.output_rate()), t0.elapsed());
        assert!(hear(&cloned) <= 0.2);
        let Some(ov) = std::env::var_os("FILMCRAFT_OPENVOICE_DIR") else { return };
        let conv = filmcraft_tts::openvoice::Converter::load(std::path::Path::new(&ov)).unwrap();
        let colour = |x: &[f32], r: u32| {
            conv.tone_colour(&filmcraft_audio_dsp::resample::resample_sinc(&filmcraft_tts::clone::speech_only(x, r), r, filmcraft_tts::openvoice::RATE))
                .unwrap()
        };
        let (a, b) = (colour(&cloned, m.output_rate()), colour(&x, rate));
        let dot: f64 = a.iter().zip(&b).map(|(p, q)| f64::from(*p) * f64::from(*q)).sum();
        let norm = |v: &[f32]| v.iter().map(|p| f64::from(*p) * f64::from(*p)).sum::<f64>().sqrt();
        let cos = dot / (norm(&a) * norm(&b)).max(1e-12);
        eprintln!("tone-colour similarity to the speaker: {cos:.3}");
        assert!(cos >= 0.6, "{cos}");
    }
}
