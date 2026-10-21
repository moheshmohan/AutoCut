//! The voice lab: better references for a natural cloned voice, found in the speaker's own clips,
//! auditioned, and put in place with the old one kept (`voice.*`; `filmcraft-cli voice …`).
//!
//! | command | does |
//! |---|---|
//! | `voice.findReferences` | scan clips (files, folders, `.fcproj` projects' media) for speech by the voice's speaker (tone-colour similarity to its reference) and rank stretches as references per engine: VoxCPM (English, 10–30 s) and IndicF5 (Malayalam, 6–12 s), on similarity, signal quality, whole sentences, length and language; writes `refs.json` and the candidates' WAVs. A background job (`wait: true` runs it now) |
//! | `voice.audition` | the current reference and the best candidates of a `refs.json` (and two variants: two references joined, one lightly cleaned) speak a fixed test set; every reading is scored on speaker similarity against held-out genuine speech, on what a speech model hears (CER, sentence-start CER) and on artefacts; writes `audition.json`, the readings (WAV and M4A) and `README.txt` |
//! | `voice.promote` | make an auditioned (or found) reference the voice's: its sample (`voxcpm`) or its Malayalam reference (`indicf5`, with the clip's words: the speech model's, or `transcript`), from the original recording at full band; the old one is kept as a version; consent carries over (same speaker, checked) |
//! | `voice.versions` / `voice.rollback` | the references a voice had, and going back to one |
//! | `voice.mixCheck` | how far a mix chain (the showcase's narration chain) moves a voice: speaker similarity of readings and of genuine speech before and after it, and the voice's balance for small speakers |
//!
//! Nothing here changes a voice except `voice.promote` and `voice.rollback`. The models are the
//! voice's: OpenVoice's tone colour, Whisper (`whisper-base` for English and language id,
//! `whisper-medium-ml` for Malayalam), VoxCPM2 and IndicF5; the big ones run under the GPU lock.

use serde_json::Value;

use crate::commands::{CommandSpec, always, bad, bool_p, str_p};
use crate::{Result, Session};

fn spec(id: &'static str, label: &'static str, params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: always, run, journal: false }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            "voice.findReferences",
            "Find Voice References",
            r#"{"voice":"clone:…","paths":[file|folder|.fcproj],"out":dir?,"maxClipSeconds":600?,"maxMinutes":240?,"maxFiles":2000?,"minSimilarity":0.85?,"top":6?,"perClip":2?,"heldOutSeconds":120?,"wait":bool?}"#,
            find_references,
        ),
        spec(
            "voice.audition",
            "Audition Voice References",
            r#"{"voice":"clone:…","report":"<dir>/refs.json","out":dir?,"top":3?,"engines":["voxcpm","indicf5"]?,"variants":bool?,"m4a":bool?,"wait":bool?}"#,
            audition,
        ),
        spec(
            "voice.promote",
            "Use Voice Reference",
            r#"{"voice":"clone:…","report":"<dir>/audition.json"|"<dir>/refs.json","candidate":id,"transcript":str? (the clip's exact words, for indicf5),"minSimilarity":0.85?}"#,
            promote,
        ),
        spec("voice.versions", "Voice Reference Versions", r#"{"voice":"clone:…"}"#, versions),
        spec("voice.rollback", "Restore Voice Reference", r#"{"voice":"clone:…","version":"r0"}"#, rollback),
        spec(
            "voice.mixCheck",
            "Voice Mix Check",
            r#"{"takes":[wav],"genuine":[wav]?,"report":"<dir>/refs.json"?,"chains":{"name":[{"effect":id,"params":{…}}]},"lufs":-16?}"#,
            mix_check,
        ),
    ]
}

/// Limits of the finder's work.
const MAX_CLIP_SECONDS: f64 = 600.0;
const MAX_MINUTES: f64 = 240.0;
const MAX_FILES: usize = 2000;
/// Hard caps on what a caller may ask for.
const CAP_CLIP_SECONDS: f64 = 3.0 * 3600.0;
const CAP_MINUTES: f64 = 24.0 * 60.0;
const CAP_FILES: usize = 20_000;

fn voice_p(cmd: &str, p: &Value) -> Result<String> {
    let v = str_p(p, "voice").ok_or_else(|| bad(cmd, "`voice` is required (clone:<name>)"))?;
    if !v.starts_with(filmcraft_tts::CLONE_PREFIX) || v.len() > 200 {
        return Err(bad(cmd, "`voice` must be a cloned voice (clone:<name>)"));
    }
    Ok(v.to_string())
}

fn num_p(cmd: &str, p: &Value, key: &str, default: f64, range: (f64, f64)) -> Result<f64> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_f64()
            .filter(|x| x.is_finite() && *x >= range.0 && *x <= range.1)
            .ok_or_else(|| bad(cmd, format!("`{key}` must be a number from {} to {}", range.0, range.1))),
    }
}

fn path_list(cmd: &str, p: &Value, key: &str, required: bool) -> Result<Vec<String>> {
    match p.get(key) {
        None | Some(Value::Null) if !required => Ok(Vec::new()),
        None | Some(Value::Null) => Err(bad(cmd, format!("`{key}` is required (a list of paths)"))),
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(a)) => {
            if a.len() > CAP_FILES {
                return Err(bad(cmd, format!("`{key}`: at most {CAP_FILES} paths")));
            }
            a.iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_string)
                        .filter(|s| !s.is_empty() && s.len() < 4096)
                        .ok_or_else(|| bad(cmd, format!("`{key}` must be a list of paths")))
                })
                .collect()
        }
        Some(_) => Err(bad(cmd, format!("`{key}` must be a list of paths"))),
    }
}

/// What `voice.findReferences` was asked.
#[cfg_attr(not(feature = "neural-voices"), allow(dead_code))]
#[derive(Clone, Debug)]
struct FindParams {
    voice: String,
    paths: Vec<String>,
    out: Option<String>,
    max_clip_seconds: f64,
    max_minutes: f64,
    max_files: usize,
    min_similarity: f64,
    top: usize,
    per_clip: usize,
    held_out_seconds: f64,
}

fn find_params(p: &Value) -> Result<FindParams> {
    let cmd = "voice.findReferences";
    let voice = voice_p(cmd, p)?;
    let paths = path_list(cmd, p, "paths", true)?;
    if paths.is_empty() {
        return Err(bad(cmd, "`paths`: give the clips (files, folders or .fcproj projects) to look in"));
    }
    Ok(FindParams {
        voice,
        paths,
        out: str_p(p, "out").map(str::to_string),
        max_clip_seconds: num_p(cmd, p, "maxClipSeconds", MAX_CLIP_SECONDS, (5.0, CAP_CLIP_SECONDS))?,
        max_minutes: num_p(cmd, p, "maxMinutes", MAX_MINUTES, (0.1, CAP_MINUTES))?,
        max_files: num_p(cmd, p, "maxFiles", MAX_FILES as f64, (1.0, CAP_FILES as f64))? as usize,
        min_similarity: num_p(cmd, p, "minSimilarity", 0.85, (0.0, 0.999))?,
        top: num_p(cmd, p, "top", 6.0, (1.0, 50.0))? as usize,
        per_clip: num_p(cmd, p, "perClip", 2.0, (1.0, 20.0))? as usize,
        held_out_seconds: num_p(cmd, p, "heldOutSeconds", 120.0, (10.0, 1200.0))?,
    })
}

fn find_references(s: &mut Session, p: &Value) -> Result<Value> {
    let fp = find_params(p)?;
    let wait = bool_p(p, "wait").unwrap_or(false);
    #[cfg(feature = "neural-voices")]
    {
        let label = format!("Finding references for {}", fp.voice);
        let (media, services) = (s.media.clone(), s.services.clone());
        run_job(s, label, wait, "filmcraft-voice-refs", move |prog| lab::find(&fp, &media, &*services, prog))
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        let _ = (s, fp, wait);
        Err(bad("voice.findReferences", UNAVAILABLE))
    }
}

/// What `voice.audition` was asked.
#[cfg_attr(not(feature = "neural-voices"), allow(dead_code))]
#[derive(Clone, Debug)]
struct AuditionParams {
    voice: String,
    report: String,
    out: Option<String>,
    top: usize,
    engines: Vec<String>,
    variants: bool,
    m4a: bool,
}

fn audition_params(p: &Value) -> Result<AuditionParams> {
    let cmd = "voice.audition";
    let voice = voice_p(cmd, p)?;
    let report = str_p(p, "report").ok_or_else(|| bad(cmd, "`report` is required: the refs.json voice.findReferences wrote"))?.to_string();
    let engines = match p.get("engines") {
        None | Some(Value::Null) => vec!["voxcpm".to_string(), "indicf5".to_string()],
        Some(Value::Array(a)) if !a.is_empty() && a.len() <= 2 => a
            .iter()
            .map(|e| match e.as_str() {
                Some(x @ ("voxcpm" | "indicf5")) => Ok(x.to_string()),
                _ => Err(bad(cmd, "`engines`: voxcpm and/or indicf5")),
            })
            .collect::<Result<Vec<_>>>()?,
        Some(_) => return Err(bad(cmd, "`engines`: voxcpm and/or indicf5")),
    };
    let flag = |k: &str, d: bool| match p.get(k) {
        None | Some(Value::Null) => Ok(d),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(bad(cmd, format!("`{k}` must be true or false"))),
    };
    Ok(AuditionParams {
        voice,
        report,
        out: str_p(p, "out").map(str::to_string),
        top: num_p(cmd, p, "top", 3.0, (1.0, 10.0))? as usize,
        engines,
        variants: flag("variants", true)?,
        m4a: flag("m4a", true)?,
    })
}

fn audition(s: &mut Session, p: &Value) -> Result<Value> {
    let ap = audition_params(p)?;
    let wait = bool_p(p, "wait").unwrap_or(false);
    #[cfg(feature = "neural-voices")]
    {
        let label = format!("Auditioning references for {}", ap.voice);
        let (media, services) = (s.media.clone(), s.services.clone());
        run_job(s, label, wait, "filmcraft-voice-audition", move |prog| lab::audition(&ap, &media, &*services, prog))
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        let _ = (s, ap, wait);
        Err(bad("voice.audition", UNAVAILABLE))
    }
}

fn promote(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "voice.promote";
    let voice = voice_p(cmd, p)?;
    let report = str_p(p, "report").ok_or_else(|| bad(cmd, "`report` is required: the audition.json (or refs.json) the candidate is in"))?.to_string();
    let candidate = str_p(p, "candidate")
        .filter(|c| !c.is_empty() && c.len() < 200)
        .ok_or_else(|| bad(cmd, "`candidate` is required (an id from the report)"))?
        .to_string();
    let min = num_p(cmd, p, "minSimilarity", 0.85, (0.0, 0.999))?;
    if p.get("transcript").is_some_and(|t| !t.is_string() && !t.is_null()) {
        return Err(bad(cmd, "`transcript` must be a string"));
    }
    let words = str_p(p, "transcript").map(str::trim).filter(|t| !t.is_empty()).map(str::to_string);
    if words.as_ref().is_some_and(|w| w.len() > 16 * 1024) {
        return Err(bad(cmd, "`transcript` is too long"));
    }
    #[cfg(feature = "neural-voices")]
    {
        lab::promote(&voice, &report, &candidate, words.as_deref(), min, &s.media, &*s.services).map_err(|e| bad(cmd, e))
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        let _ = (s, voice, report, candidate, min, words);
        Err(bad(cmd, UNAVAILABLE))
    }
}

fn versions(_s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "voice.versions";
    let voice = voice_p(cmd, p)?;
    #[cfg(feature = "neural-voices")]
    {
        let models = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory"))?;
        let p = filmcraft_tts::clone::load_profile(&filmcraft_tts::clone::voices_dir(&models), &voice).map_err(|e| bad(cmd, e.to_string()))?;
        let v = filmcraft_tts::clone::versions(&models, &voice).map_err(|e| bad(cmd, e.to_string()))?;
        Ok(serde_json::json!({"voice": voice, "revision": p.revision, "source": p.source, "versions": v}))
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        let _ = voice;
        Err(bad(cmd, UNAVAILABLE))
    }
}

fn rollback(_s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "voice.rollback";
    let voice = voice_p(cmd, p)?;
    let version = str_p(p, "version")
        .filter(|v| !v.is_empty() && v.len() < 64)
        .ok_or_else(|| bad(cmd, "`version` is required (voice.versions lists them: r0, r1, …)"))?
        .to_string();
    #[cfg(feature = "neural-voices")]
    {
        let models = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory"))?;
        let v = filmcraft_tts::clone::rollback(&models, &voice, &version).map_err(|e| bad(cmd, e.to_string()))?;
        let p = filmcraft_tts::clone::load_profile(&filmcraft_tts::clone::voices_dir(&models), &voice).map_err(|e| bad(cmd, e.to_string()))?;
        Ok(serde_json::json!({"voice": voice, "restored": v, "revision": p.revision, "source": p.source}))
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        let _ = (voice, version);
        Err(bad(cmd, UNAVAILABLE))
    }
}

/// One effect of a chain: a project audio effect id and its parameters.
#[cfg_attr(not(feature = "neural-voices"), allow(dead_code))]
#[derive(Clone, Debug)]
struct ChainStep {
    effect: String,
    params: serde_json::Map<String, Value>,
}

/// Chains at most, effects per chain.
const MAX_CHAINS: usize = 12;
const MAX_STEPS: usize = 16;

fn chains_p(cmd: &str, p: &Value) -> Result<Vec<(String, Vec<ChainStep>)>> {
    let obj = p.get("chains").and_then(Value::as_object).ok_or_else(|| bad(cmd, "`chains` is required: {\"name\": [{\"effect\": id, \"params\": {…}}, …]}"))?;
    if obj.is_empty() || obj.len() > MAX_CHAINS {
        return Err(bad(cmd, format!("`chains`: 1 to {MAX_CHAINS} of them")));
    }
    let mut out = Vec::new();
    for (name, steps) in obj {
        let list = steps.as_array().ok_or_else(|| bad(cmd, format!("chain `{name}` must be a list of effects")))?;
        if list.len() > MAX_STEPS {
            return Err(bad(cmd, format!("chain `{name}`: at most {MAX_STEPS} effects")));
        }
        let mut chain = Vec::new();
        for st in list {
            let effect = str_p(st, "effect").ok_or_else(|| bad(cmd, format!("chain `{name}`: every step needs `effect`")))?;
            let def = filmcraft_project::find_effect(effect).filter(|_| filmcraft_render::audio_fx::supported(effect));
            if def.is_none() {
                return Err(bad(cmd, format!("chain `{name}`: `{effect}` is not an audio effect")));
            }
            let params = match st.get("params") {
                None | Some(Value::Null) => serde_json::Map::new(),
                Some(Value::Object(o)) => o.clone(),
                Some(_) => return Err(bad(cmd, format!("chain `{name}`: `params` must be an object"))),
            };
            chain.push(ChainStep { effect: effect.to_string(), params });
        }
        out.push((name.chars().take(80).collect(), chain));
    }
    Ok(out)
}

fn mix_check(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "voice.mixCheck";
    let takes = path_list(cmd, p, "takes", true)?;
    if takes.is_empty() || takes.len() > 200 {
        return Err(bad(cmd, "`takes`: 1 to 200 WAV files of the voice"));
    }
    let genuine = path_list(cmd, p, "genuine", false)?;
    let report = str_p(p, "report").map(str::to_string);
    if genuine.is_empty() && report.is_none() {
        return Err(bad(cmd, "give genuine speech of the speaker: `genuine` (WAV files) or `report` (a refs.json: its held-out speech)"));
    }
    let chains = chains_p(cmd, p)?;
    let lufs = num_p(cmd, p, "lufs", -16.0, (-40.0, -6.0))?;
    #[cfg(feature = "neural-voices")]
    {
        lab::mix_check(&takes, &genuine, report.as_deref(), &chains, lufs, &s.media, &*s.services).map_err(|e| bad(cmd, e))
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        let _ = (s, takes, genuine, report, chains, lufs);
        Err(bad(cmd, UNAVAILABLE))
    }
}

#[cfg(not(feature = "neural-voices"))]
const UNAVAILABLE: &str = "the voice lab is not available in this build (built without `neural-voices`)";

/// Run `work` as a background job (now, with `wait`): its JSON is the command's result when it
/// ran now, else the job's status names where it was written.
#[cfg(feature = "neural-voices")]
fn run_job(
    s: &mut Session,
    label: String,
    wait: bool,
    thread: &str,
    work: impl FnOnce(&filmcraft_export::Progress) -> std::result::Result<Value, String> + Send + 'static,
) -> Result<Value> {
    use std::sync::{Arc, Mutex, PoisonError};
    let job = crate::narration::new_job(s, label);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let out: Arc<Mutex<Option<Value>>> = Arc::default();
    let slot = out.clone();
    let id = crate::narration::spawn(s, job, wait, thread, move || {
        let t0 = web_time::Instant::now();
        let r = work(&prog);
        let status = r.as_ref().map(|v| v["summary"].as_str().unwrap_or("done").to_string()).map_err(String::clone);
        if let Ok(v) = r {
            *slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(v);
        }
        crate::narration::finish(&prog, &res, status, t0);
    })?;
    if wait {
        let found = out.lock().unwrap_or_else(PoisonError::into_inner).take();
        return match found {
            Some(mut v) => {
                v["job"] = serde_json::json!(id);
                Ok(v)
            }
            None => {
                let err = s.jobs.iter().find(|j| j.id == id).and_then(|j| j.progress.error.lock().unwrap_or_else(PoisonError::into_inner).clone());
                Err(crate::EngineError::Other(err.unwrap_or_else(|| "the job failed".into())))
            }
        };
    }
    Ok(serde_json::json!({"job": id}))
}

#[cfg(feature = "neural-voices")]
#[path = "voice_lab_run.rs"]
mod lab;

#[cfg(test)]
#[path = "voice_lab_tests.rs"]
mod tests;
