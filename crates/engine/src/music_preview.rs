//! Hearing generated music before it goes on the timeline ("find music by feel").
//!
//! | command | does |
//! |---|---|
//! | `music.preview` | render the music `audio.generate` would make with the same parameters (`mood`, `genre`, `energy`, `duration`, `bpm`, `key`, `seed`, `levels`/`mute`, `soundfont`) as a background job, without touching the project; the host plays it ([`MusicPreview::latest`]); `path` also writes it as a WAV; `wait: true` returns when it is ready |
//!
//! The answer carries the seed it used (without `seed`, the one `audio.generate` would pick next),
//! the instruments, and the track's taxonomy, sections and beat map. `audio.generate` with the same
//! parameters and seed places exactly what was heard, reusing the preview instead of rendering it
//! again.

use std::sync::{Arc, Mutex, PoisonError};

use serde_json::{Value, json};

use filmcraft_audio_dsp::music;

use crate::commands::{CommandSpec, always, bad, bool_p, str_p};
use crate::soundfonts::Instruments;
use crate::{EngineError, Result, Session};

const CMD: &str = "music.preview";

/// A rendered preview.
pub struct Previewed {
    pub request: music::Request,
    /// [`Instruments::key`] of what played it.
    pub instruments: String,
    pub track: music::Track,
    /// Set when the chosen SoundFont failed and the oscillators played it instead.
    pub warning: Option<String>,
}

type Slot = Arc<Mutex<Option<(u64, std::result::Result<Arc<Previewed>, String>)>>>;

/// The latest preview (session state; never saved).
#[derive(Default)]
pub struct MusicPreview {
    slot: Slot,
    last: Option<Arc<Previewed>>,
}

impl MusicPreview {
    /// The latest finished preview.
    pub fn latest(&self) -> Option<Arc<Previewed>> {
        if let Some((_, Ok(p))) = &*self.slot.lock().unwrap_or_else(PoisonError::into_inner) {
            return Some(p.clone());
        }
        self.last.clone()
    }

    /// The outcome of preview job `job`, once it finished (`None` while running or when a later
    /// preview replaced it).
    pub fn outcome(&self, job: u64) -> Option<std::result::Result<Arc<Previewed>, String>> {
        match &*self.slot.lock().unwrap_or_else(PoisonError::into_inner) {
            Some((j, r)) if *j == job => Some(r.clone()),
            _ => None,
        }
    }

    /// The preview of exactly `req` on `instruments`, when there is one.
    pub fn matching(&self, req: &music::Request, instruments: &str) -> Option<Arc<Previewed>> {
        self.latest().filter(|p| p.request == *req && p.instruments == instruments)
    }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Preview Music",
        menu: &[],
        shortcut: None,
        params: r#"{"mood":"epic|tense|uplifting|calm|happy|dark|dreamy|energetic|sad"|synonym="epic","genre":"cinematic|electronic|pop|ambient|lofi|piano"?,"energy":0..1|[0..1 ×1..64]?,"duration":seconds|"durationTicks":ticks?,"bpm":40..200?,"key":"A|C#|Eb…"?,"seed":u64?,"levels":{"drums|bass|melody|pad":dB|"mute"}?,"mute":["drums"…]?,"soundfont":"auto|oscillators|generaluser-gs"|path?,"path":str?,"wait":bool?}"#,
        enabled: always,
        run: preview,
        journal: false,
    }]
}

fn preview(s: &mut Session, p: &Value) -> Result<Value> {
    if let Some(k) = str_p(p, "kind").filter(|k| *k != "music") {
        return Err(bad(CMD, format!("only music is previewed here, not `{k}`")));
    }
    if bool_p(p, "stems") == Some(true) {
        return Err(bad(CMD, "a preview is the mix; place stems with audio.generate"));
    }
    let mut req = crate::audio_generate::music_request(s, p, CMD)?;
    req.stems = false;
    let plan = music::plan(&req, crate::audio_generate::RATE).map_err(|e| bad(CMD, e.to_string()))?;
    let (inst, warning) = crate::soundfonts::resolve(s, p, CMD)?;
    let path = match p.get("path") {
        None | Some(Value::Null) => None,
        Some(Value::String(t)) if !t.trim().is_empty() => Some(t.clone()),
        Some(v) => return Err(bad(CMD, format!("`path` must be the WAV file to write, not {v}"))),
    };
    let wait = bool_p(p, "wait").unwrap_or(false);
    let key = inst.key();
    let mut out = json!({"seed": req.seed, "instruments": inst.to_json(), "warning": warning});
    if let (Some(o), Value::Object(info)) = (out.as_object_mut(), crate::music::track_json(&plan)) {
        o.extend(info);
    }
    // already heard: no new job
    if path.is_none()
        && let Some(prev) = s.music_preview.matching(&req, &key)
    {
        out["job"] = Value::Null;
        out["ready"] = json!(true);
        out["warning"] = json!(prev.warning.clone().or(warning));
        return Ok(out);
    }
    let job = crate::narration::new_job(s, format!("Previewing music: {} {}", plan.mood.name(), plan.genre.name()));
    job.progress.total.store(1, std::sync::atomic::Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let slot = s.music_preview.slot.clone();
    // the previous preview stays playable until this one is ready
    s.music_preview.last = s.music_preview.latest();
    let job_id = job.id;
    let services = s.services.clone();
    let r2 = req.clone();
    let inst2: Instruments = inst.clone();
    let path2 = path.clone();
    let id = crate::narration::spawn(s, job, wait, "filmcraft-music-preview", move || {
        let t0 = web_time::Instant::now();
        let r = crate::soundfonts::render(&r2, &inst2).and_then(|(track, warning)| {
            if let Some(dest) = &path2 {
                let bytes = crate::audio_generate::write_wav_f32_stereo(&track.mix.left, &track.mix.right, crate::audio_generate::RATE);
                services.write_file(dest, &bytes).map_err(|e| format!("{dest}: {e}"))?;
            }
            Ok(Arc::new(Previewed { request: r2.clone(), instruments: inst2.key(), track, warning }))
        });
        let status = r.as_ref().map(|p| format!("{:.1} s of music ready to play", p.track.mix.seconds())).map_err(Clone::clone);
        *slot.lock().unwrap_or_else(PoisonError::into_inner) = Some((job_id, r));
        prog.done.store(1, std::sync::atomic::Ordering::Relaxed);
        crate::narration::finish(&prog, &res, status, t0);
    })?;
    out["job"] = json!(id);
    out["path"] = json!(path);
    if wait {
        match s.music_preview.outcome(id) {
            Some(Ok(p)) => {
                out["ready"] = json!(true);
                out["seconds"] = json!(p.track.mix.seconds());
                out["warning"] = json!(p.warning.clone());
            }
            Some(Err(e)) => return Err(EngineError::Other(format!("music preview: {e}"))),
            None => out["ready"] = json!(false),
        }
    } else {
        out["ready"] = json!(false);
    }
    Ok(out)
}
