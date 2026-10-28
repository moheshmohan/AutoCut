//! The voice lab: better references for a natural cloned voice, found in the speaker's own clips,
//! auditioned, and put in place with the old one kept (`voice.*`; `filmcraft-cli voice …`).
//!
//! | command | does |
//! |---|---|
//! | `voice.findReferences` | scan clips (files, folders, `.fcproj` projects' media) for speech by the voice's speaker (tone-colour similarity to its reference) and rank stretches as references per engine: VoxCPM (English, 10–30 s) and IndicF5 (Malayalam, 6–12 s), on similarity, signal quality, whole sentences, length and language; writes `refs.json` and the candidates' WAVs. A background job (`wait: true` runs it now) |
//! | `voice.audition` | the current reference and the best candidates of a `refs.json` (and two variants: two references joined, one lightly cleaned) speak a fixed test set, `takes` readings of each line (the same seeds for every reference); every reading is scored on speaker similarity against held-out genuine speech, on what a speech model hears (CER, sentence-start CER) and on artefacts; writes `audition.json`, the readings (WAV and M4A) and `README.txt` |
//! | `voice.promote` | make an auditioned (or found) reference the voice's: its sample (`voxcpm`) or its Malayalam reference (`indicf5`, with the clip's words: the speech model's, or `transcript`), from the original recording at full band; the old one is kept as a version; consent carries over (same speaker, checked) |
//! | `voice.versions` / `voice.rollback` | the references a voice had, and going back to one |
//! | `voice.mixCheck` | how far a mix chain (the showcase's narration chain) moves a voice: speaker similarity of readings and of genuine speech before and after it, and the voice's balance for small speakers |
//! | `voice.findStyles` | the style finder: IndicF5 candidates (6–12 s of whole phrases by the voice's speaker, ending at a sentence end, Malayalam unless `language: "any"`) in clips (`paths`: files, folders, `.fcproj` projects; `items`: the project's media) measured for prosody (`filmcraft_audio_dsp::prosody`), placed on the speaker's own scale and sorted into delivery styles (calm, plain, conversational, warm, excited, amazed, curious; `filmcraft_tts::voice_lab::styles`); the best `top` per style heard by both Malayalam speech models (their words and their agreement); writes `styles.json` and each pick's WAV. Each pick carries what a style reference needs: `path` (and `item` when it came from one), `start`, `end`, `transcript`, `style`. `speakerRefs`: more recordings of the speaker (other rooms, other microphones) whose tone colour is averaged with the voice's reference for the speaker check. A background job (`wait: true` runs it now) |
//! | `voice.prosody` | prosody of recordings (`paths`, or `segments` of them): pitch median and range, movement, final contour, energy variation, syllable rate, voiced fraction; with `shapes` (range, shift, question rise, emphasis) each reshaped by TD-PSOLA and written to `out`; with `hear` both Malayalam speech models hear the original and each reshaped version (the character error rate between them) |
//!
//! Nothing here changes a voice except `voice.promote` and `voice.rollback`. The models are the
//! voice's: OpenVoice's tone colour, Whisper (`whisper-base` for English and language id,
//! `whisper-medium-ml` for Malayalam), VoxCPM2 and IndicF5; the big ones run under the GPU lock.

use serde_json::Value;

use crate::commands::{CommandSpec, always, bad, bool_p, str_p};
use crate::{Result, Session};
use filmcraft_project::{ItemId, MediaRef};

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
            r#"{"voice":"clone:…","report":"<dir>/refs.json","out":dir?,"top":3?,"takes":1..5?,"engines":["voxcpm","indicf5"]?,"variants":bool?,"m4a":bool?,"wait":bool?}"#,
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
        spec(
            "voice.findStyles",
            "Find Voice Styles",
            r#"{"voice":"clone:…","paths":[file|folder|.fcproj]?,"items":[id]?,"top":3?,"out":dir?,"language":"ml"|"any"?,"hear":true?,"speakerRefs":[file|{"path","start","end"}]? (more recordings of the speaker for the speaker check),"maxClipSeconds":600?,"maxMinutes":240?,"maxFiles":2000?,"minSimilarity":0.85?,"wait":bool?}"#,
            find_styles,
        ),
        spec(
            "voice.prosody",
            "Voice Prosody",
            r#"{"paths":[file]?,"segments":[{"path":file,"start":s,"end":s}]?,"shapes":[{"range":0.5..2?,"shift":-12..12?,"questionRise":-12..12?,"questionEnd":s?,"emphasisDb":0..6?}]?,"out":dir? (needed with shapes),"hear":bool?,"wait":bool?}"#,
            prosody,
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
    takes: u32,
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
        takes: num_p(cmd, p, "takes", 2.0, (1.0, 5.0))? as u32,
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

/// What `voice.findStyles` was asked.
#[cfg_attr(not(feature = "neural-voices"), allow(dead_code))]
#[derive(Clone, Debug)]
struct StylesParams {
    voice: String,
    /// Clips: files, folders and projects.
    paths: Vec<String>,
    /// The project's media items asked for, and their files.
    items: Vec<(u64, String)>,
    out: Option<String>,
    top: usize,
    /// Malayalam candidates only (else any language).
    malayalam: bool,
    hear: bool,
    max_clip_seconds: f64,
    max_minutes: f64,
    max_files: usize,
    min_similarity: f64,
    /// More recordings of the speaker (other rooms and microphones): their tone colour joins the
    /// voice's reference in the speaker check.
    speaker_refs: Vec<Segment>,
}

/// Most picks a style may ask for, and extra speaker references.
const MAX_SPEAKER_REFS: usize = 20;
const MAX_TOP: usize = 20;

/// The file of each project item in `p["items"]`: media on disk, nothing else.
fn items_p(cmd: &str, s: &Session, p: &Value) -> Result<Vec<(u64, String)>> {
    let list = match p.get("items") {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Array(a)) if a.len() <= CAP_FILES => a,
        Some(_) => return Err(bad(cmd, format!("`items` must be a list of at most {CAP_FILES} item ids"))),
    };
    let mut out = Vec::new();
    for v in list {
        let id = v.as_u64().ok_or_else(|| bad(cmd, "`items` must be a list of item ids"))?;
        let item = s.project.item(ItemId(id)).ok_or_else(|| bad(cmd, format!("no item {id}")))?;
        match item.as_media().map(|m| &m.media) {
            Some(MediaRef::File { path }) => out.push((id, path.clone())),
            _ => return Err(bad(cmd, format!("item {id} ({}) is not a media file", item.name))),
        }
    }
    Ok(out)
}

fn styles_params(s: &Session, p: &Value) -> Result<StylesParams> {
    let cmd = "voice.findStyles";
    let voice = voice_p(cmd, p)?;
    let paths = path_list(cmd, p, "paths", false)?;
    let items = items_p(cmd, s, p)?;
    if paths.is_empty() && items.is_empty() {
        return Err(bad(cmd, "say where the speaker's clips are: `paths` (files, folders or .fcproj projects) and/or `items` (the project's media)"));
    }
    let malayalam = match p.get("language") {
        None | Some(Value::Null) => true,
        Some(Value::String(l)) if l == "ml" => true,
        Some(Value::String(l)) if l == "any" => false,
        Some(_) => return Err(bad(cmd, "`language` is \"ml\" (the default: Malayalam candidates only) or \"any\"")),
    };
    let hear = match p.get("hear") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(bad(cmd, "`hear` must be true or false")),
    };
    Ok(StylesParams {
        voice,
        paths,
        items,
        out: str_p(p, "out").map(str::to_string),
        top: num_p(cmd, p, "top", 3.0, (1.0, MAX_TOP as f64))? as usize,
        malayalam,
        hear,
        max_clip_seconds: num_p(cmd, p, "maxClipSeconds", MAX_CLIP_SECONDS, (5.0, CAP_CLIP_SECONDS))?,
        max_minutes: num_p(cmd, p, "maxMinutes", MAX_MINUTES, (0.1, CAP_MINUTES))?,
        max_files: num_p(cmd, p, "maxFiles", MAX_FILES as f64, (1.0, CAP_FILES as f64))? as usize,
        min_similarity: num_p(cmd, p, "minSimilarity", 0.85, (0.0, 0.999))?,
        speaker_refs: segments_p(cmd, p, "speakerRefs", MAX_SPEAKER_REFS)?,
    })
}

fn find_styles(s: &mut Session, p: &Value) -> Result<Value> {
    let sp = styles_params(s, p)?;
    let wait = bool_p(p, "wait").unwrap_or(false);
    #[cfg(feature = "neural-voices")]
    {
        let label = format!("Finding delivery styles for {}", sp.voice);
        let (media, services) = (s.media.clone(), s.services.clone());
        run_job(s, label, wait, "filmcraft-voice-styles", move |prog| lab::styles::find_styles(&sp, &media, &*services, prog))
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        let _ = (s, sp, wait);
        Err(bad("voice.findStyles", UNAVAILABLE))
    }
}

/// A recording, or a stretch of one (seconds).
#[cfg_attr(not(feature = "neural-voices"), allow(dead_code))]
#[derive(Clone, Debug)]
struct Segment {
    path: String,
    start: f64,
    end: Option<f64>,
}

/// What `voice.prosody` was asked.
#[cfg_attr(not(feature = "neural-voices"), allow(dead_code))]
#[derive(Clone, Debug)]
struct ProsodyParams {
    inputs: Vec<Segment>,
    shapes: Vec<filmcraft_audio_dsp::prosody::Shape>,
    out: Option<String>,
    hear: bool,
}

/// Recordings or stretches of them in `p[key]`: each a path, or `{"path", "start"?, "end"?}`
/// (seconds; at most [`MAX_PROSODY_SECONDS`] long), at most `max` of them.
fn segments_p(cmd: &str, p: &Value, key: &str, max: usize) -> Result<Vec<Segment>> {
    let list = match p.get(key) {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Array(a)) if a.len() <= max => a,
        Some(_) => return Err(bad(cmd, format!("`{key}` must be a list of at most {max} paths or {{path, start, end}}"))),
    };
    let mut out = Vec::new();
    for seg in list {
        let path = match seg {
            Value::String(s) => Some(s.as_str()),
            _ => str_p(seg, "path"),
        };
        let path = path.filter(|s| !s.is_empty() && s.len() < 4096).ok_or_else(|| bad(cmd, format!("each of `{key}` needs a path")))?;
        let start = seg.get("start").and_then(Value::as_f64).unwrap_or(0.0);
        let end = seg.get("end").and_then(Value::as_f64);
        if seg.get("start").is_some_and(|v| !v.is_number() && !v.is_null()) || seg.get("end").is_some_and(|v| !v.is_number() && !v.is_null()) {
            return Err(bad(cmd, format!("`{key}`: `start` and `end` are seconds")));
        }
        if !(start.is_finite() && start >= 0.0) || end.is_some_and(|e| !(e.is_finite() && e > start && e - start <= MAX_PROSODY_SECONDS)) {
            return Err(bad(cmd, format!("`{key}`: `start` and `end` must be seconds, the end after the start and at most {MAX_PROSODY_SECONDS} s on")));
        }
        out.push(Segment { path: path.to_string(), start, end });
    }
    Ok(out)
}

/// Most recordings and shapes `voice.prosody` takes at once, and the longest recording (s).
const MAX_PROSODY_INPUTS: usize = 100;
const MAX_SHAPES: usize = 8;
const MAX_PROSODY_SECONDS: f64 = 600.0;

fn shape_p(cmd: &str, v: &Value) -> Result<filmcraft_audio_dsp::prosody::Shape> {
    let o = v.as_object().ok_or_else(|| bad(cmd, "each shape is {\"range\", \"shift\", \"questionRise\", \"questionEnd\", \"emphasisDb\"}"))?;
    let mut shape = filmcraft_audio_dsp::prosody::Shape::default();
    for (k, x) in o {
        let n = || x.as_f64().filter(|f| f.is_finite()).ok_or_else(|| bad(cmd, format!("shape `{}` must be a number", k.chars().take(40).collect::<String>())));
        match k.as_str() {
            "range" => shape.range = n()?,
            "shift" => shape.shift = n()?,
            "questionRise" => shape.question_rise = n()?,
            "questionEnd" => shape.question_end = if x.is_null() { None } else { Some(n()?) },
            "emphasisDb" => shape.emphasis_db = n()?,
            other => return Err(bad(cmd, format!("unknown shape field `{}`", other.chars().take(40).collect::<String>()))),
        }
    }
    shape.check().map_err(|e| bad(cmd, format!("shape: {e}")))?;
    Ok(shape)
}

fn prosody_params(p: &Value) -> Result<ProsodyParams> {
    let cmd = "voice.prosody";
    let mut inputs: Vec<Segment> = path_list(cmd, p, "paths", false)?.into_iter().map(|path| Segment { path, start: 0.0, end: None }).collect();
    inputs.extend(segments_p(cmd, p, "segments", MAX_PROSODY_INPUTS)?);
    if inputs.is_empty() || inputs.len() > MAX_PROSODY_INPUTS {
        return Err(bad(cmd, format!("give 1 to {MAX_PROSODY_INPUTS} recordings: `paths` and/or `segments`")));
    }
    let shapes = match p.get("shapes") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) if !a.is_empty() && a.len() <= MAX_SHAPES => a.iter().map(|v| shape_p(cmd, v)).collect::<Result<Vec<_>>>()?,
        Some(_) => return Err(bad(cmd, format!("`shapes` must be a list of 1 to {MAX_SHAPES} shapes"))),
    };
    let out = str_p(p, "out").map(str::to_string);
    if !shapes.is_empty() && out.is_none() {
        return Err(bad(cmd, "`out` is required with `shapes`: the folder the reshaped recordings are written to"));
    }
    let hear = match p.get("hear") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(bad(cmd, "`hear` must be true or false")),
    };
    Ok(ProsodyParams { inputs, shapes, out, hear })
}

fn prosody(s: &mut Session, p: &Value) -> Result<Value> {
    let pp = prosody_params(p)?;
    let wait = bool_p(p, "wait").unwrap_or(false);
    #[cfg(feature = "neural-voices")]
    {
        let label = format!("Measuring the prosody of {} recordings", pp.inputs.len());
        let (media, services) = (s.media.clone(), s.services.clone());
        run_job(s, label, wait, "filmcraft-voice-prosody", move |prog| lab::styles::prosody(&pp, &media, &*services, prog))
    }
    #[cfg(not(feature = "neural-voices"))]
    {
        let _ = (s, pp, wait);
        Err(bad("voice.prosody", UNAVAILABLE))
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
