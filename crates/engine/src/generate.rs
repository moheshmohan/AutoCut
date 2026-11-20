//! Generation in the editor: pictures, video clips, music and sound effects made by the local
//! generation tools the user installed ([`crate::generate_run`]: providers, each a program on
//! this computer), imported as ordinary media and placed on the timeline.
//!
//! | command | does |
//! |---|---|
//! | `generate.providers` | the generation tools of this computer (kind, installed, licence, what must be credited), where their list is kept, and the generators built into the app |
//! | `generate.image` | a picture from a description, optionally after reference pictures (project items, files, the frame at a time): text to image, an edit, a character put into a scene; `transparent` for a cut-out; imported, and placed as a still on request |
//! | `generate.video` | a short clip from a description, optionally starting from a picture (an item, a file, the frame under the playhead); imported and placed |
//! | `generate.sound` | a sound effect or music from a description; imported and placed on an audio track |
//! | `generate.fillGap` | fills an empty stretch of a video track with a generated clip that carries on from the frame before it (or leads into the one after); `all` fills every gap of the track |
//! | `generate.augment` | changes what the frame at a time shows by an instruction (add a character, change the weather…) and lays the result over the clip, as a still or animated |
//! | `generate.extend` | carries the picture on from a time: the tool is given the last moments before it as a small video and makes what comes next, so the motion continues |
//! | `generate.restyle` | makes a clip again from a description: the tool is given the clip as a small video and an adapter of its model keeps its shapes and movement (or changes one thing about it, as the adapter does); with `look` the picture tool changes the first frame first (a new style, another character) and the clip starts on it |
//! | `generate.assets` | makes the files a scene's asset list names and leaves the ones that are there (a RAEA composition's pictures, clips and sounds, rebuilt from their descriptions and seeds) |
//! | `generate.pick` | puts another variation of a generated item in the place of the one on the timeline |
//! | `generate.history` | what was generated into this project: item, kind, description, tool, seed |
//!
//! **One at a time.** A generation holds the GPU ([`filmcraft_locks`]), so a second one is
//! refused while one runs; `jobs.cancel` stops it (its program is killed, nothing is changed).
//!
//! **When it lands.** The job writes files only. They are imported and placed when the session
//! next looks at its jobs (each UI frame, `jobs.list`, any `generate.*` command), as one undo
//! step; a track that filled up meanwhile is not overwritten (the next free one is used).
//!
//! **Reproducible where the tool is.** The seed, tool and size are kept in the item's metadata
//! (Description, Log Note, Generator); without a seed each generation takes the next one.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use serde_json::{Value, json};

use filmcraft_project::{ClipId, ItemId, ItemKind, MediaRef, TrackItem};
use filmcraft_time::{Tick, TimeRange};

use crate::commands::{CommandSpec, always, bad, bool_p, checked_u32_p, f64_p, str_p, time_p, u64_p};
use crate::generate_run::{self as run, Kind, Provider, Vars};
use crate::{EngineError, Result, Session};

const FOLDER: &str = "Generated";
/// The metadata field that ties the variations of one request together.
const VARIATION: &str = "Variation Of";
const MAX_PROMPT: usize = 4000;
/// The most gaps one `generate.fillGap` fills.
const MAX_GAPS: usize = 8;
const CONTINUE: &str = "Continue this shot naturally: the same place, light, subjects and camera movement.";
const LEAD_IN: &str = "A shot that leads naturally into this picture: the same place, light and subjects.";
const BETWEEN: &str = "A shot that moves naturally from the first picture to the last: the same place, light and subjects.";
/// The most variations one request makes.
const MAX_VARIATIONS: u64 = 4;
/// The most files one asset list names.
const MAX_ASSETS: usize = 32;
const MOVES: [&str; 8] = ["auto", "in", "out", "left", "right", "up", "down", "circle"];

type Outcome = Arc<Mutex<Option<std::result::Result<(), String>>>>;

/// One run of a provider.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
struct Step {
    provider: Provider,
    args: Vec<String>,
    out: String,
}

/// Where a result goes on the timeline.
struct Place {
    at: Tick,
    /// The wanted track (`V2`, `A3`); the next free one when it is taken.
    track: Option<String>,
    /// The lowest video track to use (0 = V1): a result laid over a clip goes above it.
    above: usize,
    /// A picture's length on the timeline.
    still: Option<Tick>,
    /// The stretch to fill: a longer clip is cut to it, a shorter one slowed towards it.
    fit: Option<Tick>,
    /// A video's own sound comes along.
    audio: bool,
    /// Where in the file the placed part starts (a carried-on clip: after the frames it was given).
    source_in: Option<Tick>,
    /// A placed picture moves in depth (`clip.photo3d`'s `move`).
    photo3d: Option<String>,
}

impl Place {
    fn at(at: Tick) -> Place {
        Place { at, track: None, above: 0, still: None, fit: None, audio: false, source_in: None, photo3d: None }
    }
}

/// One result: a file to import and, on request, place.
struct Land {
    path: String,
    kind: Kind,
    description: String,
    note: String,
    generator: String,
    attribution: String,
    seed: u64,
    import: bool,
    place: Option<Place>,
    /// The file name of the first of the variations this is one of.
    variation_of: Option<String>,
}

/// A generation in flight.
pub struct Pending {
    job: u64,
    cmd: &'static str,
    out: Outcome,
    lands: Vec<Land>,
}

/// Generations in flight, and what finished ones made (job → items, clips), reported in
/// `jobs.list` as `generated`.
#[derive(Default)]
pub struct Generation {
    pending: Vec<Pending>,
    outcomes: HashMap<u64, Value>,
    made: u64,
}

fn spec(id: &'static str, label: &'static str, params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>, journal: bool) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: always, run, journal }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec("generate.providers", "Generation Tools", "{}", providers, false),
        spec(
            "generate.image",
            "Generate Image",
            r#"{"prompt":str,"negative":str?,"references":[item id|path]? (at most 16),"frameTime":ticks?|"frameSeconds":f64?|"fromPlayhead":bool? (the program frame there as the first reference),"width":px?,"height":px?,"seed":u64?,"steps":n?,"strength":0..1?,"transparent":bool=false,"variations":1..4=1 (the first is placed; generate.pick swaps another in),"move":"auto"|"in"|"out"|"left"|"right"|"up"|"down"|"circle"? (a placed picture moves in depth),"provider":id?,"dir":str?,"import":bool=true,"place":bool=false,"track":"V2"?,"atTime":ticks?|"atSeconds":f64?,"seconds":0.1..600=5,"wait":bool=false}"#,
            image,
            true,
        ),
        spec(
            "generate.video",
            "Generate Video",
            r#"{"prompt":str,"image":item id|path|"playhead"?,"lastImage":item id|path? (the picture it ends on, where the tool can),"seconds":0.1..60=4,"width":px?,"height":px?,"seed":u64?,"variations":1..4=1,"audio":bool=false,"provider":id?,"dir":str?,"place":bool=true,"track":"V2"?,"atTime":ticks?|"atSeconds":f64?,"wait":bool=false}"#,
            video,
            true,
        ),
        spec(
            "generate.sound",
            "Generate Sound",
            r#"{"prompt":str,"kind":"sfx"|"music"="sfx","seconds":0.1..600?,"clip":id? (as long as this clip, starting where it starts),"seed":u64?,"variations":1..4=1,"provider":id?,"dir":str?,"place":bool=true,"track":"A2"?,"atTime":ticks?|"atSeconds":f64?,"wait":bool=false}"#,
            sound,
            true,
        ),
        spec(
            "generate.fillGap",
            "Fill Gap with Generated Video",
            r#"{"track":"V1"?,"atTime":ticks?|"atSeconds":f64? (a time inside the gap; default the playhead),"all":bool=false,"from":"before"|"after"|"both"|"motion"|"prompt"="before" (both: from the frame before to the frame after; motion: carries on the movement of what comes before),"prompt":str?,"seed":u64?,"provider":id?,"dir":str?,"wait":bool=false}"#,
            fill_gap,
            true,
        ),
        spec(
            "generate.extend",
            "Carry a Shot On",
            r#"{"atTime":ticks?|"atSeconds":f64? (where the new clip starts; default the end of the clip under the playhead),"seconds":0.3..10=2,"prompt":str?,"seed":u64?,"provider":id?,"track":"V2"?,"dir":str?,"place":bool=true,"wait":bool=false}"#,
            extend,
            true,
        ),
        spec(
            "generate.restyle",
            "Make a Clip Again",
            r#"{"prompt":str (what the new clip shows),"clip":id? (default the selected clip, else the one under the playhead),"how":name? (a way of the tool, generate.providers lists them; default the first that is installed),"image":item id|path? (the picture the new clip starts on),"look":str? (the picture tool changes the clip's first frame by this instruction, and the new clip starts on it),"references":[item id|path]? (pictures for `look`: a character to put in),"strength":0..1=1 (how closely the clip is followed),"seed":u64?,"provider":id?,"imageProvider":id?,"track":"V2"?,"dir":str?,"place":bool=true,"wait":bool=false}"#,
            restyle,
            true,
        ),
        spec(
            "generate.assets",
            "Generate a Scene's Assets",
            r#"{"manifest":str (an absolute path to assets.json),"only":[name]?,"force":bool=false,"import":bool=false,"dryRun":bool=false,"wait":bool=false}"#,
            assets,
            true,
        ),
        spec("generate.pick", "Use This Variation", r#"{"item":id,"clip":id?}"#, pick, true),
        spec(
            "generate.augment",
            "Augment Scene",
            r#"{"instruction":str,"atTime":ticks?|"atSeconds":f64? (default the playhead),"references":[item id|path]?,"animate":bool=false,"motion":str? (what moves, when animated),"seconds":0.1..60?,"seed":u64?,"provider":id?,"videoProvider":id?,"dir":str?,"place":bool=true,"wait":bool=false}"#,
            augment,
            true,
        ),
        spec("generate.history", "Generated Media", "{}", history, false),
    ]
}

fn provider_json(p: &Provider) -> Value {
    json!({
        "id": p.id, "kind": p.kind.name(), "name": p.name, "installed": run::installed(p),
        "license": p.license, "attribution": p.attribution, "source": p.source,
        "estimateSeconds": p.estimate_seconds, "maxSeconds": p.max_seconds, "maxSide": p.max_side,
        "maxReferences": p.max_references, "references": !p.per_reference.is_empty(),
        "bothEnds": p.optional.contains_key("lastImage"), "carriesOn": !p.extend_command.is_empty(),
        "restyles": p.restyles.iter().map(|(name, w)| json!({
            "name": name, "about": w.about, "license": w.license, "installed": run::restyle_installed(p, name),
        })).collect::<Vec<_>>(),
    })
}

/// The generation tools as JSON (also for the Generate panel): `providers`, `manifest`, `error`
/// when the list cannot be read, `builtIn`, and the running job.
pub fn providers_json(s: &Session) -> Value {
    let (list, error) = match run::load() {
        Ok(l) => (l, Value::Null),
        Err(e) => (Vec::new(), json!(e)),
    };
    json!({
        "providers": list.iter().map(provider_json).collect::<Vec<_>>(),
        "manifest": run::manifest_path().map(|p| p.to_string_lossy().into_owned()),
        "error": error,
        "running": s.generation.pending.first().map(|q| q.job),
        "builtIn": [
            {"command": "music.generateNeural", "kind": "music", "name": "ACE-Step 1.5 (in the app)"},
            {"command": "audio.generate", "kind": "music", "name": "Procedural music and sounds (in the app)"},
            {"command": "tts.create", "kind": "speech", "name": "Narration (in the app)"},
        ],
    })
}

fn providers(s: &mut Session, _p: &Value) -> Result<Value> {
    poll(s);
    Ok(providers_json(s))
}

fn history(s: &mut Session, _p: &Value) -> Result<Value> {
    poll(s);
    let mut rows: Vec<Value> = s
        .project
        .items
        .values()
        .filter_map(|it| {
            let generator = it.metadata.get("Generator")?;
            Some(json!({
                "item": it.id.0, "name": it.name, "path": item_path(s, it.id), "generator": generator,
                "description": it.metadata.get("Description"), "note": it.metadata.get("Log Note"),
                "variationOf": it.metadata.get(VARIATION),
            }))
        })
        .collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r["item"].as_u64().unwrap_or(0)));
    Ok(json!({"generated": rows, "running": s.generation.pending.first().map(|q| q.job)}))
}

/// The provider of a request: the named one, else the first installed one of the kind.
fn provider_for(cmd: &str, p: &Value, kind: Kind, key: &str) -> Result<Provider> {
    let list = run::load().map_err(|e| bad(cmd, e))?;
    let where_ = run::manifest_path().map(|m| m.to_string_lossy().into_owned()).unwrap_or_default();
    if let Some(id) = str_p(p, key) {
        let found =
            list.into_iter().find(|x| x.id == id).ok_or_else(|| bad(cmd, format!("no generation tool `{id}` in {where_} (generate.providers lists them)")))?;
        if found.kind != kind {
            return Err(bad(cmd, format!("`{id}` makes {}, not {}", found.kind.name(), kind.name())));
        }
        return Ok(found);
    }
    let of_kind: Vec<Provider> = list.into_iter().filter(|x| x.kind == kind).collect();
    if of_kind.is_empty() {
        return Err(bad(
            cmd,
            format!(
                "no {} generation tool is set up on this computer: describe one in {where_} (docs/generation.md shows how), then run this again",
                kind.name()
            ),
        ));
    }
    let first = of_kind.iter().find(|x| run::installed(x)).or(of_kind.first()).cloned();
    first.ok_or_else(|| bad(cmd, "no generation tool"))
}

fn text_p(cmd: &str, p: &Value, key: &str) -> Result<Option<String>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(t)) => {
            let t = t.trim();
            if t.chars().count() > MAX_PROMPT || t.contains('\0') {
                return Err(bad(cmd, format!("`{key}` is longer than {MAX_PROMPT} characters")));
            }
            Ok((!t.is_empty()).then(|| t.to_string()))
        }
        Some(_) => Err(bad(cmd, format!("`{key}` must be text"))),
    }
}

fn prompt_p(cmd: &str, p: &Value, key: &str) -> Result<String> {
    text_p(cmd, p, key)?.ok_or_else(|| bad(cmd, format!("`{key}` is needed: say what to make")))
}

fn seconds_p(cmd: &str, p: &Value, most: f64) -> Result<Option<f64>> {
    match p.get("seconds") {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_f64() {
            Some(x) if x.is_finite() && x >= 0.1 && x <= most => Ok(Some(x)),
            _ => Err(bad(cmd, format!("`seconds` must be 0.1 to {most}"))),
        },
    }
}

fn seed_p(s: &mut Session, p: &Value) -> u64 {
    s.generation.made = s.generation.made.saturating_add(1);
    u64_p(p, "seed").unwrap_or(s.generation.made)
}

fn variations_p(cmd: &str, p: &Value) -> Result<u64> {
    match p.get("variations") {
        None | Some(Value::Null) => Ok(1),
        Some(v) => match v.as_u64() {
            Some(n) if (1..=MAX_VARIATIONS).contains(&n) => Ok(n),
            _ => Err(bad(cmd, format!("`variations` must be 1 to {MAX_VARIATIONS}"))),
        },
    }
}

fn file_name(path: &str) -> String {
    std::path::Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// The file of a media item.
fn item_path(s: &Session, id: ItemId) -> Option<String> {
    match &s.project.item(id)?.kind {
        ItemKind::Media(m) => match &m.media {
            MediaRef::File { path } => Some(path.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// One reference: a project item (by id) or a file (an absolute path).
fn reference(s: &Session, cmd: &str, v: &Value) -> Result<String> {
    let path = match v {
        Value::Number(_) => {
            let id = v.as_u64().ok_or_else(|| bad(cmd, "a reference item id must be a whole number"))?;
            item_path(s, ItemId(id)).ok_or_else(|| bad(cmd, format!("item {id} is not a media file")))?
        }
        Value::String(t) if std::path::Path::new(t).is_absolute() && !t.contains('\0') => t.clone(),
        _ => return Err(bad(cmd, "a reference is a project item id or an absolute file path")),
    };
    if !cfg!(target_arch = "wasm32") && !std::path::Path::new(&path).is_file() {
        return Err(bad(cmd, format!("the reference {path} is not a file")));
    }
    Ok(path)
}

fn references_p(s: &Session, cmd: &str, p: &Value, provider: &Provider) -> Result<Vec<String>> {
    let Some(v) = p.get("references").filter(|v| !v.is_null()) else { return Ok(Vec::new()) };
    let list = v.as_array().ok_or_else(|| bad(cmd, "`references` must be a list"))?;
    let most = provider.max_references.map_or(run::MAX_REFERENCES, |m| (m as usize).min(run::MAX_REFERENCES));
    if list.len() > most {
        return Err(bad(cmd, format!("{} takes at most {most} references", provider.name)));
    }
    if !list.is_empty() && provider.per_reference.is_empty() {
        return Err(bad(cmd, format!("{} takes no reference pictures", provider.name)));
    }
    list.iter().map(|x| reference(s, cmd, x)).collect()
}

/// `<dir>/<base> <n>.<ext>` with the first `n` used by neither a project item nor a file.
fn unique_path(s: &Session, dir: &str, base: &str, ext: &str, taken: &[String]) -> Result<String> {
    let names: std::collections::HashSet<&str> = s.project.items.values().map(|i| i.name.as_str()).collect();
    for k in 1..=99_999u32 {
        let file = format!("{base} {k}.{ext}");
        let path = std::path::Path::new(dir).join(&file).to_string_lossy().into_owned();
        if !names.contains(file.as_str()) && !taken.contains(&path) && s.services.file_size(&path).is_err() {
            return Ok(path);
        }
    }
    Err(EngineError::Other(format!("{dir}: no free file name for \"{base} <n>.{ext}\"")))
}

fn ext(p: &Provider) -> String {
    p.extension.clone().unwrap_or_else(|| p.kind.extension().to_string())
}

/// The program frame at `time` as a PNG in `dir` (a reference picture).
fn frame_png(s: &mut Session, cmd: &str, time: Tick, dir: &str, taken: &[String]) -> Result<String> {
    if s.state.active_sequence.is_none() {
        return Err(bad(cmd, "no sequence is open to take a frame from"));
    }
    let path = unique_path(s, dir, "Reference Frame", "png", taken)?;
    if let Some(parent) = std::path::Path::new(&path).parent()
        && !cfg!(target_arch = "wasm32")
    {
        std::fs::create_dir_all(parent).map_err(|e| EngineError::Other(format!("{}: {e}", parent.display())))?;
    }
    s.execute("file.exportFrame", json!({"target": "program", "time": time.max(Tick::ZERO).0, "path": path, "format": "png"}))?;
    if s.services.file_size(&path).is_err() {
        return Err(bad(cmd, "the frame could not be written"));
    }
    Ok(path)
}

/// The frame's shape: the open sequence's, else square.
fn aspect(s: &Session) -> (u32, u32) {
    s.active_sequence().map_or((1, 1), |q| (q.settings.width.max(1), q.settings.height.max(1)))
}

fn note(p: &Provider, seed: u64, extra: &str) -> String {
    format!("{} · seed {seed}{}{}", p.name, if extra.is_empty() { "" } else { " · " }, extra)
}

fn land_of(p: &Provider, path: &str, description: &str, seed: u64, extra: &str, import: bool, place: Option<Place>) -> Land {
    Land {
        path: path.to_string(),
        kind: p.kind,
        description: description.chars().take(MAX_PROMPT).collect(),
        note: note(p, seed, extra),
        generator: p.id.clone(),
        attribution: p.attribution.clone(),
        seed,
        import,
        place,
        variation_of: None,
    }
}

/// Starts the steps as one job; their results land when it finishes.
fn start(s: &mut Session, cmd: &'static str, label: String, steps: Vec<Step>, lands: Vec<Land>, wait: bool) -> Result<Value> {
    if cfg!(target_arch = "wasm32") {
        return Err(bad(cmd, "generation tools run on the desktop app, not in the browser"));
    }
    if let Some(q) = s.generation.pending.first() {
        return Err(bad(cmd, format!("something is already being generated (job {}); wait for it or stop it (jobs.cancel)", q.job)));
    }
    for st in &steps {
        if !run::installed(&st.provider) {
            return Err(bad(
                cmd,
                format!(
                    "{} is not installed: its program {} is not on this computer",
                    st.provider.name,
                    st.provider.command.first().cloned().unwrap_or_default()
                ),
            ));
        }
    }
    let job = crate::narration::new_job(s, label);
    job.progress.total.store(1000, std::sync::atomic::Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let out: Outcome = Default::default();
    let job_id = job.id;
    let files: Vec<String> = lands.iter().map(|l| l.path.clone()).collect();
    let about: Vec<Value> = steps.iter().map(|st| json!({"provider": st.provider.id, "out": st.out})).collect();
    s.generation.pending.push(Pending { job: job_id, cmd, out: out.clone(), lands });
    let spawned = crate::narration::spawn(s, job, wait, "filmcraft-generate", move || {
        let t0 = web_time::Instant::now();
        let r = run_steps(&steps, &prog);
        let status = r.clone().map(|()| format!("{} generated", if steps.len() == 1 { "1 file".to_string() } else { format!("{} files", steps.len()) }));
        *out.lock().unwrap_or_else(PoisonError::into_inner) = Some(r);
        crate::narration::finish(&prog, &res, status, t0);
    });
    if let Err(e) = spawned {
        s.generation.pending.retain(|q| q.job != job_id);
        return Err(e);
    }
    let mut v = json!({"job": job_id, "files": files, "steps": about});
    if wait {
        poll(s);
        let o = s.generation.outcomes.get(&job_id).cloned().unwrap_or(Value::Null);
        if let Some(e) = o.get("error").and_then(Value::as_str) {
            return Err(bad(cmd, e.to_string()));
        }
        if let (Some(dst), Some(src)) = (v.as_object_mut(), o.as_object()) {
            for (k, x) in src {
                dst.insert(k.clone(), x.clone());
            }
        }
    }
    Ok(v)
}

#[cfg(not(target_arch = "wasm32"))]
fn run_steps(steps: &[Step], prog: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    use std::sync::atomic::Ordering;
    let status = |t: String| *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = t;
    status("Waiting for the GPU".into());
    // another session's model may be on the GPU: wait for it (without a place for locks, unlocked)
    let _gpu = match filmcraft_locks::acquire(filmcraft_locks::Resource::Gpu, "generation", std::time::Duration::from_secs(30 * 60), &mut |_| {}) {
        Ok(g) => Some(g),
        Err(e @ filmcraft_locks::LockError::Busy(..)) => return Err(format!("{e}; try again when it is free (`filmcraft-cli locks`)")),
        Err(filmcraft_locks::LockError::Io(_)) => None,
    };
    let total: f64 = steps.iter().map(|st| st.provider.estimate_seconds.unwrap_or(60.0).clamp(1.0, 86_400.0)).sum::<f64>().max(1.0);
    let mut before = 0.0;
    for (i, st) in steps.iter().enumerate() {
        if prog.cancel.load(Ordering::Relaxed) {
            return Err("stopped: nothing was changed".into());
        }
        let est = st.provider.estimate_seconds.unwrap_or(60.0).clamp(1.0, 86_400.0);
        status(if steps.len() > 1 { format!("{} ({} of {})", st.provider.name, i + 1, steps.len()) } else { st.provider.name.clone() });
        let mut tick = |gone: f64, said: Option<f64>| {
            // a tool that says how far it is (`progress 0.4`) is believed; the others say nothing
            // while they work, so the bar follows their usual time and never fills early
            let part = said.map_or((gone / est).min(0.97), |f| f.min(0.99));
            let done = ((before + part * est) / total * 1000.0).clamp(0.0, 990.0) as u64;
            prog.done.store(done, Ordering::Relaxed);
        };
        run::run(&st.provider, &st.args, std::path::Path::new(&st.out), &prog.cancel, &mut tick)?;
        before += est;
    }
    prog.done.store(1000, Ordering::Relaxed);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn run_steps(_: &[Step], _: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    Err("generation tools run on the desktop app".into())
}

/// What generation job `job` made, when it is one (`generated` in `jobs.list`).
pub fn outcome_json(s: &Session, job: u64) -> Option<Value> {
    s.generation.outcomes.get(&job).cloned()
}

/// Whether something is being generated (the Generate panel's busy state).
pub fn running(s: &Session) -> Option<u64> {
    s.generation.pending.first().map(|q| q.job)
}

/// Import and place the results of finished jobs. Called each UI frame (through
/// [`Session::poll_persistence`]), by `jobs.list` and by the `generate.*` commands.
pub fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.generation.pending.len() {
        let done = s.generation.pending.get(i).is_some_and(|q| q.out.lock().unwrap_or_else(PoisonError::into_inner).is_some());
        if !done {
            i += 1;
            continue;
        }
        let q = s.generation.pending.remove(i);
        let r = q.out.lock().unwrap_or_else(PoisonError::into_inner).take();
        let outcome = match r {
            Some(Ok(())) => match land(s, &q) {
                Ok(v) => v,
                Err(e) => {
                    s.error_toast(q.cmd, format!("Generation: {e}"));
                    json!({"error": e.to_string()})
                }
            },
            Some(Err(e)) => json!({"error": e}),
            None => continue,
        };
        s.generation.outcomes.insert(q.job, outcome);
        // bounded: the oldest outcomes go
        if s.generation.outcomes.len() > 256
            && let Some(old) = s.generation.outcomes.keys().min().copied()
        {
            s.generation.outcomes.remove(&old);
        }
    }
}

fn video_index(label: &str) -> Option<usize> {
    let n: usize = label.strip_prefix('V').or_else(|| label.strip_prefix('v'))?.parse().ok()?;
    n.checked_sub(1)
}

fn audio_index(label: &str) -> Option<usize> {
    let n: usize = label.strip_prefix('A').or_else(|| label.strip_prefix('a'))?.parse().ok()?;
    n.checked_sub(1)
}

fn overlaps(items: &[TrackItem], range: TimeRange) -> bool {
    items.iter().any(|it| it.start < range.end() && range.start < it.end())
}

fn video_free(s: &Session, i: usize, range: TimeRange) -> bool {
    s.active_sequence().and_then(|q| q.video_tracks.get(i)).is_some_and(|t| !t.locked && !overlaps(&t.items, range))
}

fn audio_free(s: &Session, i: usize, range: TimeRange) -> bool {
    s.active_sequence().and_then(|q| q.audio_tracks.get(i)).is_some_and(|t| !t.locked && !overlaps(&t.items, range))
}

/// The video track a result goes on: the wanted one when it is free there, else the lowest free
/// one from `above` up, else a new one.
fn video_track(s: &Session, want: Option<&str>, above: usize, range: TimeRange) -> String {
    if let Some(i) = want.and_then(video_index).filter(|i| video_free(s, *i, range)) {
        return format!("V{}", i + 1);
    }
    let n = s.active_sequence().map_or(0, |q| q.video_tracks.len());
    (above..n).find(|i| video_free(s, *i, range)).map_or_else(|| "new".to_string(), |i| format!("V{}", i + 1))
}

/// The audio track a result goes on: the wanted one when free, else the first free one, else a
/// new one.
fn audio_track(s: &mut Session, want: Option<&str>, range: TimeRange) -> Result<String> {
    if let Some(i) = want.and_then(audio_index).filter(|i| audio_free(s, *i, range)) {
        return Ok(format!("A{}", i + 1));
    }
    let n = s.active_sequence().map_or(0, |q| q.audio_tracks.len());
    if let Some(i) = (0..n).find(|i| audio_free(s, *i, range)) {
        return Ok(format!("A{}", i + 1));
    }
    s.execute("sequence.addTracks", json!({"video": 0, "audio": 1, "audioAfter": "last"}))?;
    let n = s.active_sequence().map_or(0, |q| q.audio_tracks.len());
    Ok(format!("A{}", n.max(1)))
}

fn clip_span(s: &Session, clip: u64) -> Option<(Tick, Tick)> {
    let (_, it) = s.active_sequence()?.find_item(ClipId(clip))?;
    Some((it.start, it.duration))
}

fn first_clip(placed: &Value) -> Option<u64> {
    placed.get("clips").and_then(Value::as_array).and_then(|a| a.first()).and_then(Value::as_u64)
}

/// Import one result, note what made it, and place it.
fn land_one(s: &mut Session, l: &Land) -> Result<Value> {
    let mut out = json!({"path": l.path, "kind": l.kind.name(), "seed": l.seed, "generator": l.generator, "item": Value::Null, "clip": Value::Null});
    if !l.import {
        return Ok(out);
    }
    let imported = s.execute("file.import", json!({"paths": [l.path]}))?;
    let item = imported
        .get("items")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_u64)
        .ok_or_else(|| EngineError::Other(format!("{} could not be imported", l.path)))?;
    let (description, log, generator, attribution) = (l.description.clone(), l.note.clone(), l.generator.clone(), l.attribution.clone());
    let variation = l.variation_of.clone();
    s.edit("Generate", |pr, _| {
        if let Some(it) = pr.item_mut(ItemId(item)) {
            it.metadata.insert("Description".into(), description);
            it.metadata.insert("Log Note".into(), log);
            it.metadata.insert("Generator".into(), generator);
            if !attribution.is_empty() {
                it.metadata.insert("Credit".into(), attribution);
            }
            if let Some(v) = variation {
                it.metadata.insert(VARIATION.into(), v);
            }
        }
        Ok(())
    })?;
    out["item"] = json!(item);
    let Some(pl) = l.place.as_ref().filter(|_| s.state.active_sequence.is_some()) else { return Ok(out) };
    let at = pl.at.max(Tick::ZERO);
    let rate = s.sequence_rate().sane();
    // the stretch looked at when picking a track: the still's or the gap's length, else a second
    let want = pl.still.or(pl.fit).unwrap_or(Tick::from_seconds_f64(1.0)).max(rate.frame_duration());
    let range = TimeRange::new(at, want);
    let placed = if l.kind.is_audio() {
        let track = audio_track(s, pl.track.as_deref(), range)?;
        s.execute("timeline.place", json!({"item": item, "track": track, "time": at.0}))?
    } else {
        let track = video_track(s, pl.track.as_deref(), pl.above, range);
        let mut p = json!({"item": item, "track": track, "time": at.0, "audio": false});
        if let Some(d) = pl.still.or(pl.fit) {
            p["duration"] = json!(d.max(rate.frame_duration()).0);
        }
        if let Some(skip) = pl.source_in.filter(|t| *t > Tick::ZERO) {
            p["sourceIn"] = json!(skip.0);
        }
        if pl.audio && l.kind == Kind::Video {
            p["audio"] = json!(true);
            p["audioTrack"] = json!(audio_track(s, None, range)?);
        }
        s.execute("timeline.place", p)?
    };
    let clip = first_clip(&placed).ok_or_else(|| EngineError::Other("the generated media could not be placed".into()))?;
    out["clip"] = json!(clip);
    // a tool makes its own size (a clip is often small): the picture fills the frame it was made for
    if !l.kind.is_audio() {
        s.execute("clip.fillFrame", json!({"clips": [clip]}))?;
    }
    // a picture that moves in depth; without the depth model the picture stays, and says why
    if let Some(mv) = pl.photo3d.as_deref().filter(|_| l.kind == Kind::Image) {
        match s.execute("clip.photo3d", json!({"clips": [clip], "move": mv})) {
            Ok(_) => out["move"] = json!(mv),
            Err(e) => out["moveError"] = json!(e.to_string()),
        }
    }
    // a clip shorter than the gap it fills is slowed towards it (never below 40 %: slower looks wrong)
    if let (Some(fit), Some((_, got))) = (pl.fit, clip_span(s, clip))
        && got > Tick::ZERO
        && got < fit
        && l.kind == Kind::Video
    {
        let pct = ((got.0 as f64 / fit.0 as f64) * 10_000.0).ceil() / 100.0;
        if (40.0..100.0).contains(&pct) {
            let _ = s.execute("clip.speedDuration", json!({"clips": [clip], "speed": pct}));
        }
    }
    if let Some((start, duration)) = clip_span(s, clip) {
        out["start"] = json!(start.0);
        out["duration"] = json!(duration.0);
        if let Some(fit) = pl.fit {
            out["uncovered"] = json!((fit - duration).max(Tick::ZERO).0);
        }
    }
    Ok(out)
}

/// Import and place everything a finished job made (one undo step).
fn land(s: &mut Session, q: &Pending) -> Result<Value> {
    let n0 = s.history.undo.len();
    let result = (|| -> Result<Value> {
        let mut made = Vec::new();
        for l in &q.lands {
            made.push(land_one(s, l)?);
        }
        Ok(json!({"generated": made}))
    })();
    crate::clip_ops::collapse_history(s, n0, "Generate");
    if result.is_err() && s.history.undo.len() > n0 {
        let _ = s.undo();
        s.history.redo.clear();
    }
    result
}

fn short(t: &str) -> String {
    t.chars().take(40).collect()
}

/// What one picture is made from.
struct Picture<'a> {
    prov: &'a Provider,
    prompt: &'a str,
    negative: Option<String>,
    refs: Vec<String>,
    size: (u32, u32),
    seed: u64,
    steps: Option<u32>,
    strength: Option<f64>,
    transparent: bool,
}

/// Whether a transparent picture of this tool takes its cut-out program as a second step.
fn cut_after(cmd: &str, prov: &Provider, transparent: bool) -> Result<Option<Provider>> {
    if !transparent || prov.optional.contains_key("transparent") {
        return Ok(None);
    }
    run::cutout(prov)
        .map(Some)
        .ok_or_else(|| bad(cmd, format!("{} cannot make a transparent picture (it has no `transparent` option and no `cutoutCommand`)", prov.name)))
}

/// The steps that make one picture at `out`: the tool, then its cut-out program on the picture
/// it wrote to `source` (given when [`cut_after`] says so).
fn picture_steps(cmd: &str, q: &Picture, out: &str, cutter: Option<(Provider, String)>) -> Result<Vec<Step>> {
    let first = cutter.as_ref().map_or(out, |(_, source)| source.as_str()).to_string();
    let asked = match (&cutter, &q.prov.cutout_prompt_suffix) {
        (Some(_), Some(suffix)) => format!("{}{suffix}", q.prompt),
        _ => q.prompt.to_string(),
    };
    let vars = Vars {
        out: first.clone(),
        prompt: run::prompt_for(q.prov, &asked, q.refs.len()),
        negative: q.negative.clone(),
        width: Some(q.size.0),
        height: Some(q.size.1),
        seed: q.seed,
        references: q.refs.clone(),
        steps: q.steps,
        strength: q.strength,
        transparent: q.transparent && cutter.is_none(),
        ..Default::default()
    };
    let args = run::argv(q.prov, &vars).map_err(|e| bad(cmd, e))?;
    let mut steps = vec![Step { provider: q.prov.clone(), args, out: first.clone() }];
    if let Some((cut, _)) = cutter {
        let cut_vars = Vars { out: out.to_string(), image: Some(first), ..Default::default() };
        let args = run::argv(&cut, &cut_vars).map_err(|e| bad(cmd, e))?;
        steps.push(Step { provider: cut, args, out: out.to_string() });
    }
    Ok(steps)
}

fn image(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "generate.image";
    poll(s);
    let prov = provider_for(cmd, p, Kind::Image, "provider")?;
    let prompt = prompt_p(cmd, p, "prompt")?;
    let negative = text_p(cmd, p, "negative")?;
    let mut refs = references_p(s, cmd, p, &prov)?;
    let want = (checked_u32_p(p, "width", cmd)?, checked_u32_p(p, "height", cmd)?);
    let still = seconds_p(cmd, p, 600.0)?.unwrap_or(5.0);
    let strength = match f64_p(p, "strength") {
        Some(x) if !(0.0..=1.0).contains(&x) => return Err(bad(cmd, "`strength` must be 0 to 1")),
        x => x,
    };
    let steps = checked_u32_p(p, "steps", cmd)?.filter(|n| (1..=200).contains(n));
    let count = variations_p(cmd, p)?;
    let photo3d = match str_p(p, "move") {
        None => None,
        Some(m) if MOVES.contains(&m) => Some(m.to_string()),
        Some(m) => return Err(bad(cmd, format!("unknown `move` `{m}` ({})", MOVES.join(", ")))),
    };
    let transparent = bool_p(p, "transparent").unwrap_or(false);
    let cutter = cut_after(cmd, &prov, transparent)?;
    let dir = crate::voiceover::generated_dir(s, p, FOLDER)?;
    let frame_at = time_p(s, p, "frame").or_else(|| bool_p(p, "fromPlayhead").unwrap_or(false).then(|| s.playhead()));
    let mut taken = Vec::new();
    if let Some(t) = frame_at {
        if prov.per_reference.is_empty() {
            return Err(bad(cmd, format!("{} takes no reference pictures", prov.name)));
        }
        let frame = frame_png(s, cmd, t, &dir, &taken)?;
        taken.push(frame.clone());
        refs.insert(0, frame);
    }
    let (w, h) = run::size(&prov, want, aspect(s));
    let seed = seed_p(s, p);
    let import = bool_p(p, "import").unwrap_or(true);
    let placing = import && bool_p(p, "place").unwrap_or(false) && s.state.active_sequence.is_some();
    let at = time_p(s, p, "at").unwrap_or(s.playhead());
    let (mut run_steps, mut lands, mut first) = (Vec::new(), Vec::new(), None);
    for k in 0..count {
        let out = unique_path(s, &dir, "Generated Image", &ext(&prov), &taken)?;
        taken.push(out.clone());
        let cut = match &cutter {
            Some(c) => {
                let source = unique_path(s, &dir, "Generated Image source", &ext(&prov), &taken)?;
                taken.push(source.clone());
                Some((c.clone(), source))
            }
            None => None,
        };
        let this = seed.wrapping_add(k);
        let q =
            Picture { prov: &prov, prompt: &prompt, negative: negative.clone(), refs: refs.clone(), size: (w, h), seed: this, steps, strength, transparent };
        run_steps.extend(picture_steps(cmd, &q, &out, cut)?);
        // the first variation is the one placed; the others wait in the bin (generate.pick)
        let place = (placing && k == 0).then(|| Place {
            track: str_p(p, "track").map(str::to_string),
            still: Some(Tick::from_seconds_f64(still)),
            photo3d: photo3d.clone(),
            ..Place::at(at)
        });
        let mut land = land_of(&prov, &out, &prompt, this, &format!("{w}×{h}"), import, place);
        let group = first.get_or_insert_with(|| file_name(&out)).clone();
        land.variation_of = (count > 1).then_some(group);
        lands.push(land);
    }
    let label = format!("Generating a picture: {}", short(&prompt));
    let mut v = start(s, cmd, label, run_steps, lands, bool_p(p, "wait").unwrap_or(false))?;
    v["provider"] = json!(prov.id);
    v["seed"] = json!(seed);
    v["width"] = json!(w);
    v["height"] = json!(h);
    Ok(v)
}

/// The picture a video starts from: an item, a file, or the frame under the playhead.
fn start_image(s: &mut Session, cmd: &str, p: &Value, dir: &str) -> Result<Option<String>> {
    match p.get("image") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(t)) if t == "playhead" => {
            let at = s.playhead();
            frame_png(s, cmd, at, dir, &[]).map(Some)
        }
        Some(v) => reference(s, cmd, v).map(Some),
    }
}

/// A video step and what it makes: `seconds` wanted, from `image` when given.
#[allow(clippy::too_many_arguments)]
fn video_step(
    s: &Session,
    cmd: &str,
    prov: &Provider,
    prompt: &str,
    (image, last_image): (Option<String>, Option<String>),
    seconds: f64,
    want: (Option<u32>, Option<u32>),
    seed: u64,
    dir: &str,
    taken: &[String],
) -> Result<(Step, f64, (u32, u32))> {
    if last_image.is_some() && !prov.optional.contains_key("lastImage") {
        return Err(bad(cmd, format!("{} cannot end a clip on a given picture (it has no `lastImage` option)", prov.name)));
    }
    let (frames, fps) = run::frames(prov, run::seconds(prov, seconds, 4.0));
    let (w, h) = run::size(prov, want, aspect(s));
    let out = unique_path(s, dir, "Generated Video", &ext(prov), taken)?;
    let made = f64::from(frames) / fps;
    let vars = Vars {
        out: out.clone(),
        prompt: prompt.to_string(),
        width: Some(w),
        height: Some(h),
        seed,
        seconds: Some(made),
        frames: Some(frames),
        fps: Some(fps),
        image,
        last_image,
        ..Default::default()
    };
    let args = run::argv(prov, &vars).map_err(|e| bad(cmd, e))?;
    Ok((Step { provider: prov.clone(), args, out }, made, (w, h)))
}

fn video(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "generate.video";
    poll(s);
    let prov = provider_for(cmd, p, Kind::Video, "provider")?;
    let prompt = prompt_p(cmd, p, "prompt")?;
    let seconds = seconds_p(cmd, p, 60.0)?.unwrap_or(4.0);
    let want = (checked_u32_p(p, "width", cmd)?, checked_u32_p(p, "height", cmd)?);
    let count = variations_p(cmd, p)?;
    let dir = crate::voiceover::generated_dir(s, p, FOLDER)?;
    let image = start_image(s, cmd, p, &dir)?;
    let last = match p.get("lastImage") {
        None | Some(Value::Null) => None,
        Some(v) => Some(reference(s, cmd, v)?),
    };
    let seed = seed_p(s, p);
    let placing = bool_p(p, "place").unwrap_or(true) && s.state.active_sequence.is_some();
    let at = time_p(s, p, "at").unwrap_or(s.playhead());
    let (mut steps, mut lands, mut taken, mut first, mut length) = (Vec::new(), Vec::new(), Vec::new(), None, 0.0);
    for k in 0..count {
        let this = seed.wrapping_add(k);
        let (step, made, (w, h)) = video_step(s, cmd, &prov, &prompt, (image.clone(), last.clone()), seconds, want, this, &dir, &taken)?;
        taken.push(step.out.clone());
        length = made;
        let place =
            (placing && k == 0).then(|| Place { track: str_p(p, "track").map(str::to_string), audio: bool_p(p, "audio").unwrap_or(false), ..Place::at(at) });
        let mut land = land_of(&prov, &step.out, &prompt, this, &format!("{made:.1} s · {w}×{h}"), true, place);
        let group = first.get_or_insert_with(|| file_name(&step.out)).clone();
        land.variation_of = (count > 1).then_some(group);
        lands.push(land);
        steps.push(step);
    }
    let mut v = start(s, cmd, format!("Generating a clip: {}", short(&prompt)), steps, lands, bool_p(p, "wait").unwrap_or(false))?;
    v["provider"] = json!(prov.id);
    v["seed"] = json!(seed);
    v["seconds"] = json!(length);
    Ok(v)
}

fn sound(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "generate.sound";
    poll(s);
    let kind = match str_p(p, "kind") {
        None | Some("sfx") => Kind::Sfx,
        Some("music") => Kind::Music,
        Some(k) => return Err(bad(cmd, format!("unknown kind `{k}` (sfx, music)"))),
    };
    let prov = provider_for(cmd, p, kind, "provider")?;
    let prompt = prompt_p(cmd, p, "prompt")?;
    let count = variations_p(cmd, p)?;
    // a sound for a clip: as long as it (as far as the tool goes), starting where it starts
    let under = match u64_p(p, "clip") {
        Some(id) => Some(clip_span(s, id).ok_or_else(|| bad(cmd, format!("there is no clip {id} in the open sequence")))?),
        None => None,
    };
    let asked = seconds_p(cmd, p, 600.0)?.or(under.map(|(_, d)| d.seconds())).unwrap_or(if kind == Kind::Sfx { 5.0 } else { 20.0 });
    let seconds = run::seconds(&prov, asked, 5.0);
    let dir = crate::voiceover::generated_dir(s, p, FOLDER)?;
    let seed = seed_p(s, p);
    let placing = bool_p(p, "place").unwrap_or(true) && s.state.active_sequence.is_some();
    let at = under.map(|(start, _)| start).or_else(|| time_p(s, p, "at")).unwrap_or(s.playhead());
    let (mut steps, mut lands, mut taken, mut first) = (Vec::new(), Vec::new(), Vec::new(), None);
    for k in 0..count {
        let out = unique_path(s, &dir, if kind == Kind::Sfx { "Generated Sound" } else { "Generated Music" }, &ext(&prov), &taken)?;
        taken.push(out.clone());
        let this = seed.wrapping_add(k);
        let vars = Vars { out: out.clone(), prompt: prompt.clone(), seed: this, seconds: Some(seconds), ..Default::default() };
        let args = run::argv(&prov, &vars).map_err(|e| bad(cmd, e))?;
        let place =
            (placing && k == 0).then(|| Place { track: str_p(p, "track").map(str::to_string), fit: Some(Tick::from_seconds_f64(seconds)), ..Place::at(at) });
        let mut land = land_of(&prov, &out, &prompt, this, &format!("{seconds:.1} s"), true, place);
        let group = first.get_or_insert_with(|| file_name(&out)).clone();
        land.variation_of = (count > 1).then_some(group);
        lands.push(land);
        steps.push(Step { provider: prov.clone(), args, out });
    }
    let mut v = start(s, cmd, format!("Generating sound: {}", short(&prompt)), steps, lands, bool_p(p, "wait").unwrap_or(false))?;
    v["provider"] = json!(prov.id);
    v["seed"] = json!(seed);
    v["seconds"] = json!(seconds);
    Ok(v)
}

/// The step that carries the picture on from `at`: the `carry` frames before it go to the tool's
/// extend program as a small video, and what it writes holds them and then the new frames, so
/// the new clip is placed from `skip` into its file.
struct CarryOn {
    step: Step,
    /// The new part, seconds.
    made: f64,
    size: (u32, u32),
    skip: Tick,
    /// The small video of what came before, written for the tool.
    source: String,
}

#[allow(clippy::too_many_arguments)]
fn carry_on(s: &mut Session, cmd: &str, prov: &Provider, prompt: &str, at: Tick, seconds: f64, seed: u64, dir: &str, taken: &[String]) -> Result<CarryOn> {
    let ext_prov = run::extender(prov).ok_or_else(|| bad(cmd, format!("{} cannot carry a shot on (it has no `extendCommand`)", prov.name)))?;
    let fps = prov.fps.filter(|f| f.is_finite() && *f >= 1.0 && *f <= 240.0).unwrap_or(24.0);
    let carry = run::carry_frames(prov);
    let carried = f64::from(carry) / fps;
    let from = at.seconds() - carried;
    if from < -1.0e-6 {
        return Err(bad(cmd, format!("carrying a shot on needs {carried:.1} s of picture before it")));
    }
    let (w, h) = run::grid_size(aspect(s), prov.default_side.unwrap_or(480), prov.extend_multiple_of.unwrap_or(64));
    let source = unique_path(s, dir, "Carried Frames", "mp4", taken)?;
    if let Some(parent) = std::path::Path::new(&source).parent()
        && !cfg!(target_arch = "wasm32")
    {
        std::fs::create_dir_all(parent).map_err(|e| EngineError::Other(format!("{}: {e}", parent.display())))?;
    }
    s.execute(
        "file.exportMedia",
        json!({
            "path": source, "format": "h264", "width": w, "height": h, "fps": fps, "audio": false,
            "range": "custom", "startSeconds": from.max(0.0), "endSeconds": at.seconds(), "wait": true,
        }),
    )?;
    if s.services.file_size(&source).is_err() {
        return Err(bad(cmd, "the picture before it could not be written for the tool"));
    }
    // new frames in the counts the tool takes (LTX: eights)
    let step = prov.frames_step.unwrap_or(1).clamp(1, 1024);
    let most = prov.max_seconds.filter(|m| m.is_finite() && *m > 0.0).unwrap_or(10.0);
    let want = (seconds.clamp(0.1, most) * fps).round().clamp(1.0, 100_000.0) as u32;
    let frames = want.div_ceil(step).max(1).saturating_mul(step);
    let made = f64::from(frames) / fps;
    let mut taken2 = taken.to_vec();
    taken2.push(source.clone());
    let out = unique_path(s, dir, "Generated Video", &ext(prov), &taken2)?;
    let vars = Vars {
        out: out.clone(),
        prompt: prompt.to_string(),
        width: Some(w),
        height: Some(h),
        seed,
        seconds: Some(made),
        frames: Some(frames),
        fps: Some(fps),
        video: Some(source.clone()),
        ..Default::default()
    };
    let args = run::argv(&ext_prov, &vars).map_err(|e| bad(cmd, e))?;
    // whole frames in, so the cut falls on a frame of the file
    let skip = if fps.fract() == 0.0 { Tick::from_units(i64::from(carry), fps as i64) } else { Tick::from_seconds_f64(carried) };
    Ok(CarryOn { step: Step { provider: ext_prov, args, out }, made, size: (w, h), skip, source })
}

fn extend(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "generate.extend";
    poll(s);
    let prov = provider_for(cmd, p, Kind::Video, "provider")?;
    if s.state.active_sequence.is_none() {
        return Err(EngineError::NoSequence);
    }
    let seconds = match p.get("seconds") {
        None | Some(Value::Null) => 2.0,
        Some(v) => match v.as_f64() {
            Some(x) if x.is_finite() && (0.3..=10.0).contains(&x) => x,
            _ => return Err(bad(cmd, "`seconds` must be 0.3 to 10")),
        },
    };
    let prompt = text_p(cmd, p, "prompt")?.unwrap_or_else(|| CONTINUE.to_string());
    // where the new clip starts: asked, else the end of the clip under the playhead
    let head = s.playhead().max(Tick::ZERO);
    let under = clip_under(s, head);
    let at = time_p(s, p, "at").unwrap_or_else(|| under.map_or(head, |(_, left)| head + left)).max(Tick::ZERO);
    let above = clip_under(s, (at - s.sequence_rate().sane().frame_duration()).max(Tick::ZERO)).map_or(0, |(i, _)| i);
    let dir = crate::voiceover::generated_dir(s, p, FOLDER)?;
    let seed = seed_p(s, p);
    let c = carry_on(s, cmd, &prov, &prompt, at, seconds, seed, &dir, &[])?;
    let place = bool_p(p, "place").unwrap_or(true).then(|| Place {
        track: str_p(p, "track").map(str::to_string).or_else(|| Some(format!("V{}", above + 1))),
        above,
        source_in: Some(c.skip),
        ..Place::at(at)
    });
    let (w, h) = c.size;
    let lands = vec![land_of(&prov, &c.step.out, &prompt, seed, &format!("{:.1} s · {w}×{h} · carries a shot on", c.made), true, place)];
    let mut v = start(s, cmd, "Carrying the shot on".to_string(), vec![c.step], lands, bool_p(p, "wait").unwrap_or(false))?;
    v["provider"] = json!(prov.id);
    v["seed"] = json!(seed);
    v["seconds"] = json!(c.made);
    v["at"] = json!(at.0);
    Ok(v)
}

/// A video clip of the open sequence: its track (0 = V1), start and length.
fn video_clip(s: &Session, id: u64) -> Option<(usize, Tick, Tick)> {
    let seq = s.active_sequence()?;
    seq.video_tracks.iter().enumerate().find_map(|(i, t)| t.item(ClipId(id)).map(|it| (i, it.start, it.duration)))
}

/// The clip a command is about: the named one, else the first selected video clip, else the top
/// one under the playhead.
fn clip_for(s: &Session, cmd: &str, p: &Value) -> Result<(u64, usize, Tick, Tick)> {
    if let Some(v) = p.get("clip").filter(|v| !v.is_null()) {
        let id = v.as_u64().ok_or_else(|| bad(cmd, "`clip` must be a clip id"))?;
        let (i, start, len) = video_clip(s, id).ok_or_else(|| bad(cmd, format!("there is no video clip {id} in the open sequence")))?;
        return Ok((id, i, start, len));
    }
    if let Some(found) = s.state.selection.iter().find_map(|c| video_clip(s, c.0).map(|(i, start, len)| (c.0, i, start, len))) {
        return Ok(found);
    }
    let head = s.playhead().max(Tick::ZERO);
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    seq.video_tracks
        .iter()
        .enumerate()
        .rev()
        .find_map(|(i, t)| t.items.iter().find(|it| it.start <= head && head < it.end()).map(|it| (it.id.0, i, it.start, it.duration)))
        .ok_or_else(|| bad(cmd, "no clip is selected and none is under the playhead: name one with `clip`"))
}

fn restyle(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "generate.restyle";
    poll(s);
    let prov = provider_for(cmd, p, Kind::Video, "provider")?;
    if s.state.active_sequence.is_none() {
        return Err(EngineError::NoSequence);
    }
    let prompt = prompt_p(cmd, p, "prompt")?;
    if prov.restyle_command.is_empty() || prov.restyles.is_empty() {
        return Err(bad(cmd, format!("{} cannot make a clip again (it has no `restyleCommand` and `restyles`)", prov.name)));
    }
    let ways = || prov.restyles.keys().map(String::as_str).collect::<Vec<_>>().join(", ");
    let how = match p.get("how") {
        None | Some(Value::Null) => {
            prov.restyles.keys().find(|k| run::restyle_installed(&prov, k)).or(prov.restyles.keys().next()).cloned().unwrap_or_default()
        }
        Some(Value::String(h)) if prov.restyles.contains_key(h.as_str()) => h.clone(),
        Some(v) => return Err(bad(cmd, format!("unknown `how` {} ({} can: {})", short(&v.to_string()), prov.name, ways()))),
    };
    let way = prov.restyles.get(&how).ok_or_else(|| bad(cmd, format!("{} has no way `{how}`", prov.name)))?;
    if !cfg!(target_arch = "wasm32") && !run::restyle_installed(&prov, &how) {
        return Err(bad(cmd, format!("the adapter of `{how}` is not on this computer: {} (docs/generation.md says where it comes from)", way.adapter)));
    }
    let adapter = way.adapter.clone();
    let rprov = run::restyler(&prov, &how).ok_or_else(|| bad(cmd, format!("{} has no way `{how}`", prov.name)))?;
    let strength = match p.get("strength") {
        None | Some(Value::Null) => 1.0,
        Some(v) => match v.as_f64() {
            Some(x) if (0.0..=1.0).contains(&x) => x,
            _ => return Err(bad(cmd, "`strength` must be 0 to 1")),
        },
    };
    let look = text_p(cmd, p, "look")?;
    let given = p.get("image").filter(|v| !v.is_null());
    let has_refs = p.get("references").is_some_and(|v| v.as_array().is_some_and(|a| !a.is_empty()));
    if given.is_some() && look.is_some() {
        return Err(bad(cmd, "give `image` (the first picture) or `look` (how to change the first frame), not both"));
    }
    if has_refs && look.is_none() {
        return Err(bad(cmd, "`references` go with `look` (they are the pictures the first frame is changed with)"));
    }
    if (given.is_some() || look.is_some()) && !rprov.optional.contains_key("image") {
        return Err(bad(cmd, format!("{} cannot start the new clip on a given picture (its `restyleOptional` has no `image`)", prov.name)));
    }
    // (checked before anything is written)
    let given = match given {
        Some(v) => Some(reference(s, cmd, v)?),
        None => None,
    };
    let (clip, track, at, length) = clip_for(s, cmd, p)?;
    let st = run::restyle_stretch(&prov, length.seconds()).ok_or_else(|| {
        let least = f64::from(prov.frames_step.unwrap_or(1).saturating_add(prov.frames_offset.unwrap_or(0))) / prov.fps.unwrap_or(24.0).max(1.0);
        bad(cmd, format!("the clip is too short to make again ({} makes {least:.1} s or more)", prov.name))
    })?;
    let (w, h) = run::grid_size(aspect(s), prov.default_side.unwrap_or(480), prov.extend_multiple_of.unwrap_or(64));
    let dir = crate::voiceover::generated_dir(s, p, FOLDER)?;
    let mut taken = Vec::new();
    // the stretch as the tool takes it: small, and holding exactly the frames it will make
    let source = unique_path(s, &dir, "Restyle Source", "mp4", &taken)?;
    taken.push(source.clone());
    if let Some(parent) = std::path::Path::new(&source).parent()
        && !cfg!(target_arch = "wasm32")
    {
        std::fs::create_dir_all(parent).map_err(|e| EngineError::Other(format!("{}: {e}", parent.display())))?;
    }
    s.execute(
        "file.exportMedia",
        json!({
            "path": source, "format": "h264", "width": w, "height": h, "fps": st.source_fps, "audio": false,
            "range": "custom", "startSeconds": at.seconds(), "endSeconds": at.seconds() + st.covered, "wait": true,
        }),
    )?;
    if s.services.file_size(&source).is_err() {
        return Err(bad(cmd, "the clip could not be written for the tool"));
    }
    let seed = seed_p(s, p);
    let (mut steps, mut lands) = (Vec::new(), Vec::new());
    let image = match (given, look) {
        (Some(path), _) => Some(path),
        (None, Some(look)) => {
            // the picture tool changes the first frame, and the new clip starts on its picture
            let iprov = provider_for(cmd, p, Kind::Image, "imageProvider")?;
            if iprov.per_reference.is_empty() {
                return Err(bad(cmd, format!("{} cannot change a picture (it takes no references)", iprov.name)));
            }
            let frame = frame_png(s, cmd, at, &dir, &taken)?;
            taken.push(frame.clone());
            let mut refs = vec![frame];
            let extra = references_p(s, cmd, p, &iprov)?;
            let most = iprov.max_references.map_or(run::MAX_REFERENCES, |m| (m as usize).min(run::MAX_REFERENCES));
            if extra.len() + 1 > most {
                return Err(bad(cmd, format!("{} takes at most {most} references, the frame included", iprov.name)));
            }
            refs.extend(extra);
            let size = run::size(&iprov, (None, None), aspect(s));
            let first = unique_path(s, &dir, "Restyled Frame", &ext(&iprov), &taken)?;
            taken.push(first.clone());
            let q = Picture { prov: &iprov, prompt: &look, negative: None, refs, size, seed, steps: None, strength: None, transparent: false };
            steps.extend(picture_steps(cmd, &q, &first, None)?);
            lands.push(land_of(&iprov, &first, &look, seed, &format!("{}×{} · the first frame of a clip made again", size.0, size.1), true, None));
            Some(first)
        }
        (None, None) => None,
    };
    let out = unique_path(s, &dir, "Generated Video", &ext(&prov), &taken)?;
    let made = f64::from(st.frames) / st.fps;
    let vars = Vars {
        out: out.clone(),
        prompt: prompt.clone(),
        width: Some(w),
        height: Some(h),
        seed,
        seconds: Some(made),
        frames: Some(st.frames),
        fps: Some(st.fps),
        image,
        video: Some(source.clone()),
        adapter: Some(adapter),
        strength: Some(strength),
        ..Default::default()
    };
    let args = run::argv(&rprov, &vars).map_err(|e| bad(cmd, e))?;
    steps.push(Step { provider: rprov, args, out: out.clone() });
    // over the clip it was made from, as long as the stretch it covers
    let place = bool_p(p, "place").unwrap_or(true).then(|| Place {
        track: str_p(p, "track").map(str::to_string),
        above: track.saturating_add(1),
        fit: Some(Tick::from_seconds_f64(st.covered)),
        ..Place::at(at)
    });
    lands.push(land_of(&prov, &out, &prompt, seed, &format!("{:.1} s · {w}×{h} · a clip made again ({how})", st.covered), true, place));
    let mut v = start(s, cmd, format!("Making a clip again: {}", short(&prompt)), steps, lands, bool_p(p, "wait").unwrap_or(false))?;
    v["provider"] = json!(prov.id);
    v["how"] = json!(how);
    v["seed"] = json!(seed);
    v["clip"] = json!(clip);
    v["at"] = json!(at.0);
    v["seconds"] = json!(st.covered);
    v["left"] = json!((length.seconds() - st.covered).max(0.0));
    v["source"] = json!(source);
    Ok(v)
}

/// A scene's asset list (`assets.json`): the files a composition needs, each from a description.
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AssetList {
    /// The size of the pictures and clips that name none.
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    assets: Vec<Asset>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Asset {
    /// The file's name without its extension.
    name: String,
    /// `image` (default), `video`, `sfx` or `music`.
    #[serde(default)]
    kind: Option<String>,
    prompt: String,
    #[serde(default)]
    negative: Option<String>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    transparent: bool,
    #[serde(default)]
    provider: Option<String>,
    /// Reference pictures: other assets of the list by name (made before this one) or files.
    #[serde(default)]
    references: Vec<String>,
    /// A clip's first picture: an asset by name, or a file.
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    seconds: Option<f64>,
}

/// A seed that stays the same for a name (FNV-1a), so an asset without one is made the same again.
fn seed_of(name: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h % 1_000_000_000
}

fn asset_name_ok(name: &str) -> bool {
    let n = name.chars().count();
    (1..=64).contains(&n) && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '-' | '_')) && name.trim() == name
}

#[cfg(not(target_arch = "wasm32"))]
fn read_asset_list(cmd: &str, path: &str) -> Result<AssetList> {
    use std::io::Read as _;
    let f = std::fs::File::open(path).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
    let mut text = String::new();
    f.take(run::MANIFEST_MAX + 1).read_to_string(&mut text).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
    if text.len() as u64 > run::MANIFEST_MAX {
        return Err(bad(cmd, format!("{path} is larger than 256 KiB")));
    }
    serde_json::from_str(&text).map_err(|e| bad(cmd, format!("{path}: {e}")))
}

#[cfg(target_arch = "wasm32")]
fn read_asset_list(cmd: &str, _: &str) -> Result<AssetList> {
    Err(bad(cmd, "generation tools run on the desktop app, not in the browser"))
}

fn assets(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "generate.assets";
    poll(s);
    let manifest = str_p(p, "manifest").ok_or_else(|| bad(cmd, "`manifest` is needed: the absolute path of an assets.json"))?;
    let mpath = std::path::Path::new(manifest);
    if !mpath.is_absolute() || manifest.contains('\0') || mpath.extension().and_then(|e| e.to_str()) != Some("json") {
        return Err(bad(cmd, "`manifest` must be the absolute path of a .json file"));
    }
    let dir = mpath.parent().map(|d| d.to_string_lossy().into_owned()).filter(|d| !d.is_empty()).ok_or_else(|| bad(cmd, "`manifest` has no folder"))?;
    let list = read_asset_list(cmd, manifest)?;
    if list.assets.is_empty() || list.assets.len() > MAX_ASSETS {
        return Err(bad(cmd, format!("an asset list names 1 to {MAX_ASSETS} files")));
    }
    let only: Vec<String> = match p.get("only").filter(|v| !v.is_null()) {
        None => Vec::new(),
        Some(v) => v
            .as_array()
            .ok_or_else(|| bad(cmd, "`only` must be a list of names"))?
            .iter()
            .map(|x| x.as_str().map(str::to_string).ok_or_else(|| bad(cmd, "`only` must be a list of names")))
            .collect::<Result<_>>()?,
    };
    if let Some(unknown) = only.iter().find(|n| !list.assets.iter().any(|a| &a.name == *n)) {
        return Err(bad(cmd, format!("`only` names `{}`, which the list does not have", unknown.chars().take(80).collect::<String>())));
    }
    let (force, import, dry) = (bool_p(p, "force").unwrap_or(false), bool_p(p, "import").unwrap_or(false), bool_p(p, "dryRun").unwrap_or(false));
    let shape = match (list.width, list.height) {
        (Some(w), Some(h)) if w > 0 && h > 0 => (w, h),
        _ => aspect(s),
    };
    // name → (file, whether it is there or will be by the time a later asset needs it)
    let mut known: HashMap<String, (String, bool)> = HashMap::new();
    let (mut steps, mut lands, mut rows) = (Vec::new(), Vec::new(), Vec::new());
    for a in &list.assets {
        if !asset_name_ok(&a.name) {
            return Err(bad(cmd, "an asset's name is 1 to 64 letters, digits, spaces, - or _"));
        }
        if known.contains_key(&a.name.to_lowercase()) {
            return Err(bad(cmd, format!("the name `{}` is used twice", a.name)));
        }
        let kind = match a.kind.as_deref() {
            None | Some("image") => Kind::Image,
            Some("video") => Kind::Video,
            Some("sfx") => Kind::Sfx,
            Some("music") => Kind::Music,
            Some(k) => return Err(bad(cmd, format!("{}: unknown kind `{}` (image, video, sfx, music)", a.name, k.chars().take(40).collect::<String>()))),
        };
        let chosen = a.provider.as_ref().map_or(json!({}), |id| json!({"provider": id}));
        let prov = provider_for(cmd, &chosen, kind, "provider")?;
        let prompt = prompt_p(cmd, &json!({"prompt": a.prompt}), "prompt").map_err(|_| bad(cmd, format!("{}: its `prompt` is empty or too long", a.name)))?;
        let out = std::path::Path::new(&dir).join(format!("{}.{}", a.name, ext(&prov))).to_string_lossy().into_owned();
        let there = s.services.file_size(&out).is_ok();
        let wanted = (only.is_empty() || only.contains(&a.name)) && (force || !there);
        // another asset by name (made before this one), or a file
        let resolve = |r: &str, known: &HashMap<String, (String, bool)>| -> Result<String> {
            if let Some((path, ready)) = known.get(&r.to_lowercase()) {
                return if *ready { Ok(path.clone()) } else { Err(bad(cmd, format!("{}: it needs `{r}`, which is not made yet and not asked for", a.name))) };
            }
            if std::path::Path::new(r).is_absolute() && !r.contains('\0') && s.services.file_size(r).is_ok() {
                return Ok(r.to_string());
            }
            Err(bad(cmd, format!("{}: `{}` is neither an asset listed before it nor a file", a.name, r.chars().take(120).collect::<String>())))
        };
        let seed = a.seed.unwrap_or_else(|| seed_of(&a.name));
        let mut row = json!({"name": a.name, "path": out, "kind": kind.name(), "provider": prov.id, "seed": seed, "state": if wanted { "make" } else if there { "kept" } else { "skipped" }});
        if wanted {
            let want = (a.width.or(list.width), a.height.or(list.height));
            let note = match kind {
                Kind::Image => {
                    let most = prov.max_references.map_or(run::MAX_REFERENCES, |m| (m as usize).min(run::MAX_REFERENCES));
                    if a.references.len() > most || (!a.references.is_empty() && prov.per_reference.is_empty()) {
                        return Err(bad(
                            cmd,
                            format!("{}: {} takes at most {} references", a.name, prov.name, if prov.per_reference.is_empty() { 0 } else { most }),
                        ));
                    }
                    let refs = a.references.iter().map(|r| resolve(r, &known)).collect::<Result<Vec<_>>>()?;
                    let size = run::size(&prov, want, shape);
                    let cutter = cut_after(cmd, &prov, a.transparent)?
                        .map(|c| (c, std::path::Path::new(&dir).join(format!("{} source.{}", a.name, ext(&prov))).to_string_lossy().into_owned()));
                    let negative = a.negative.clone().filter(|n| n.chars().count() <= MAX_PROMPT && !n.contains('\0'));
                    let q = Picture { prov: &prov, prompt: &prompt, negative, refs, size, seed, steps: None, strength: None, transparent: a.transparent };
                    steps.extend(picture_steps(cmd, &q, &out, cutter)?);
                    format!("{}×{}", size.0, size.1)
                }
                Kind::Video => {
                    let image = a.image.as_deref().map(|r| resolve(r, &known)).transpose()?;
                    let secs = a.seconds.filter(|x| x.is_finite() && *x > 0.0).unwrap_or(4.0);
                    let (frames, fps) = run::frames(&prov, run::seconds(&prov, secs, 4.0));
                    let (w, h) = run::size(&prov, want, shape);
                    let made = f64::from(frames) / fps;
                    let vars = Vars {
                        out: out.clone(),
                        prompt: prompt.clone(),
                        width: Some(w),
                        height: Some(h),
                        seed,
                        seconds: Some(made),
                        frames: Some(frames),
                        fps: Some(fps),
                        image,
                        ..Default::default()
                    };
                    let args = run::argv(&prov, &vars).map_err(|e| bad(cmd, e))?;
                    steps.push(Step { provider: prov.clone(), args, out: out.clone() });
                    format!("{made:.1} s · {w}×{h}")
                }
                Kind::Sfx | Kind::Music => {
                    let secs = run::seconds(&prov, a.seconds.filter(|x| x.is_finite() && *x > 0.0).unwrap_or(if kind == Kind::Sfx { 5.0 } else { 20.0 }), 5.0);
                    let vars = Vars { out: out.clone(), prompt: prompt.clone(), seed, seconds: Some(secs), ..Default::default() };
                    let args = run::argv(&prov, &vars).map_err(|e| bad(cmd, e))?;
                    steps.push(Step { provider: prov.clone(), args, out: out.clone() });
                    format!("{secs:.1} s")
                }
            };
            row["note"] = json!(note);
            lands.push(land_of(&prov, &out, &prompt, seed, &format!("{note} · asset {}", a.name), import, None));
        }
        known.insert(a.name.to_lowercase(), (out, there || wanted));
        rows.push(row);
    }
    let making = lands.len();
    if dry || making == 0 {
        // (`job: null`: nothing was started, which a caller waiting for a job can tell from a failure)
        return Ok(json!({"assets": rows, "making": making, "dir": dir, "dryRun": dry, "job": null}));
    }
    let label = format!("Generating {making} asset{} for {}", if making == 1 { "" } else { "s" }, file_name(&dir));
    let mut v = start(s, cmd, label, steps, lands, bool_p(p, "wait").unwrap_or(false))?;
    v["assets"] = json!(rows);
    v["making"] = json!(making);
    v["dir"] = json!(dir);
    Ok(v)
}

/// The clip on the timeline that shows another variation of `item`'s request.
fn sibling_clip(s: &Session, item: ItemId, group: &str) -> Option<u64> {
    let seq = s.active_sequence()?;
    seq.video_tracks.iter().flat_map(|t| t.items.iter()).chain(seq.audio_tracks.iter().flat_map(|t| t.items.iter())).find_map(|c| {
        let other = s.project.item(c.item)?;
        (c.item != item && other.metadata.get(VARIATION).map(String::as_str) == Some(group)).then_some(c.id.0)
    })
}

fn pick(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "generate.pick";
    poll(s);
    let id = u64_p(p, "item").ok_or_else(|| bad(cmd, "`item` is needed: the generated item to use"))?;
    let group = {
        let it = s.project.item(ItemId(id)).ok_or_else(|| bad(cmd, format!("there is no item {id}")))?;
        if !it.metadata.contains_key("Generator") {
            return Err(bad(cmd, format!("item {id} was not generated (generate.history lists what was)")));
        }
        it.metadata.get(VARIATION).cloned()
    };
    let clip = match u64_p(p, "clip") {
        Some(c) => {
            clip_span(s, c).ok_or_else(|| bad(cmd, format!("there is no clip {c} in the open sequence")))?;
            c
        }
        None => group
            .as_deref()
            .and_then(|g| sibling_clip(s, ItemId(id), g))
            .ok_or_else(|| bad(cmd, "none of its variations is on the timeline: name the `clip` to replace, or place it (timeline.place)"))?,
    };
    let picture = s.active_sequence().is_some_and(|q| q.video_tracks.iter().any(|t| t.items.iter().any(|c| c.id.0 == clip)));
    let n0 = s.history.undo.len();
    let result = (|| -> Result<()> {
        s.execute("clip.replaceFromBin", json!({"clips": [clip], "item": id}))?;
        if picture {
            s.execute("clip.fillFrame", json!({"clips": [clip]}))?;
        }
        Ok(())
    })();
    crate::clip_ops::collapse_history(s, n0, "Use Variation");
    if result.is_err() && s.history.undo.len() > n0 {
        let _ = s.undo();
        s.history.redo.clear();
    }
    result?;
    Ok(json!({"clip": clip, "item": id}))
}

/// The empty stretches of a video track between its clips (and before the first), at least
/// `least` long.
pub fn gaps(items: &[TrackItem], least: Tick) -> Vec<TimeRange> {
    let mut spans: Vec<(Tick, Tick)> = items.iter().map(|it| (it.start, it.end())).collect();
    spans.sort();
    let mut out = Vec::new();
    let mut cursor = Tick::ZERO;
    for (a, b) in spans {
        if a - cursor >= least {
            out.push(TimeRange::new(cursor, a - cursor));
        }
        cursor = cursor.max(b);
    }
    out
}

fn fill_gap(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "generate.fillGap";
    poll(s);
    let prov = provider_for(cmd, p, Kind::Video, "provider")?;
    let from = str_p(p, "from").unwrap_or("before");
    if !matches!(from, "before" | "after" | "both" | "motion" | "prompt") {
        return Err(bad(cmd, format!("unknown `from` `{from}` (before, after, both, motion, prompt)")));
    }
    let ends = prov.optional.contains_key("lastImage");
    if from == "both" && !ends {
        return Err(bad(cmd, format!("{} cannot end a clip on a given picture (it has no `lastImage` option): use `from: \"before\"`", prov.name)));
    }
    if from == "motion" && prov.extend_command.is_empty() {
        return Err(bad(cmd, format!("{} cannot carry a shot on (it has no `extendCommand`): use `from: \"before\"`", prov.name)));
    }
    let label = str_p(p, "track").unwrap_or("V1");
    let index = video_index(label).ok_or_else(|| bad(cmd, format!("`{label}` is not a video track (V1, V2…)")))?;
    let rate = s.sequence_rate().sane();
    let frame = rate.frame_duration();
    let (found, locked) = {
        let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
        let t = seq.video_tracks.get(index).ok_or_else(|| bad(cmd, format!("the sequence has no track {label}")))?;
        (gaps(&t.items, Tick(frame.0.saturating_mul(2))), t.locked)
    };
    if locked {
        return Err(bad(cmd, format!("track {label} is locked")));
    }
    let chosen: Vec<TimeRange> = if bool_p(p, "all").unwrap_or(false) {
        found.into_iter().take(MAX_GAPS).collect()
    } else {
        let at = time_p(s, p, "at").unwrap_or(s.playhead());
        found.into_iter().filter(|g| g.start <= at && at < g.end()).collect()
    };
    if chosen.is_empty() {
        return Err(bad(cmd, format!("there is no gap on {label} there (a gap is an empty stretch before a clip)")));
    }
    let own = text_p(cmd, p, "prompt")?;
    if from == "prompt" && own.is_none() {
        return Err(bad(cmd, "`from: \"prompt\"` needs a `prompt`"));
    }
    let dir = crate::voiceover::generated_dir(s, p, FOLDER)?;
    let (mut steps, mut lands, mut taken, mut about) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for g in &chosen {
        // a gap at the very start has nothing before it: it leads into what follows
        let opens = g.start <= Tick::ZERO;
        let seed = seed_p(s, p);
        let place = Place { track: Some(format!("V{}", index + 1)), above: index, fit: Some(g.duration), ..Place::at(g.start) };
        if from == "motion" && !opens {
            let prompt = own.clone().unwrap_or_else(|| CONTINUE.to_string());
            let c = carry_on(s, cmd, &prov, &prompt, g.start, g.duration.seconds(), seed, &dir, &taken)?;
            taken.push(c.step.out.clone());
            taken.push(c.source.clone());
            let (w, h) = c.size;
            let place = Place { source_in: Some(c.skip), ..place };
            lands.push(land_of(&prov, &c.step.out, &prompt, seed, &format!("{:.1} s · {w}×{h} · fills a gap, carrying on", c.made), true, Some(place)));
            about.push(json!({"start": g.start.0, "duration": g.duration.0, "seconds": c.made, "seed": seed}));
            steps.push(c.step);
            continue;
        }
        let frame_at = |s: &mut Session, t: Tick, taken: &mut Vec<String>| -> Result<String> {
            let f = frame_png(s, cmd, t, &dir, taken)?;
            taken.push(f.clone());
            Ok(f)
        };
        let lead_in = from == "after" || (opens && from != "prompt");
        let (image, last) = match from {
            "prompt" => (None, None),
            "both" if !opens => (Some(frame_at(s, g.start - frame, &mut taken)?), Some(frame_at(s, g.end(), &mut taken)?)),
            // a tool that can end on a picture ends on the one that follows; another starts from it
            _ if lead_in && ends => (None, Some(frame_at(s, g.end(), &mut taken)?)),
            _ if lead_in => (Some(frame_at(s, g.end(), &mut taken)?), None),
            _ => (Some(frame_at(s, g.start - frame, &mut taken)?), None),
        };
        let prompt = own.clone().unwrap_or_else(|| match (&image, &last) {
            (Some(_), Some(_)) => BETWEEN.to_string(),
            _ if lead_in => LEAD_IN.to_string(),
            _ => CONTINUE.to_string(),
        });
        let (step, made, (w, h)) = video_step(s, cmd, &prov, &prompt, (image, last), g.duration.seconds(), (None, None), seed, &dir, &taken)?;
        taken.push(step.out.clone());
        lands.push(land_of(&prov, &step.out, &prompt, seed, &format!("{made:.1} s · {w}×{h} · fills a gap"), true, Some(place)));
        about.push(json!({"start": g.start.0, "duration": g.duration.0, "seconds": made, "seed": seed}));
        steps.push(step);
    }
    let n = steps.len();
    let mut v = start(s, cmd, format!("Filling {n} gap{} on {label}", if n == 1 { "" } else { "s" }), steps, lands, bool_p(p, "wait").unwrap_or(false))?;
    v["gaps"] = json!(about);
    v["provider"] = json!(prov.id);
    Ok(v)
}

/// The topmost video clip at `at`: its track index and the time left in it.
fn clip_under(s: &Session, at: Tick) -> Option<(usize, Tick)> {
    let seq = s.active_sequence()?;
    seq.video_tracks.iter().enumerate().rev().find_map(|(i, t)| t.items.iter().find(|it| it.start <= at && at < it.end()).map(|it| (i, it.end() - at)))
}

fn augment(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "generate.augment";
    poll(s);
    let prov = provider_for(cmd, p, Kind::Image, "provider")?;
    if prov.per_reference.is_empty() {
        return Err(bad(cmd, format!("{} cannot change a picture (it takes no references)", prov.name)));
    }
    let instruction = prompt_p(cmd, p, "instruction")?;
    let animate = bool_p(p, "animate").unwrap_or(false);
    let vprov = if animate { Some(provider_for(cmd, p, Kind::Video, "videoProvider")?) } else { None };
    let at = time_p(s, p, "at").unwrap_or(s.playhead()).max(Tick::ZERO);
    let (track, left) = clip_under(s, at).ok_or_else(|| bad(cmd, "there is no picture on the timeline at that time"))?;
    let wanted = seconds_p(cmd, p, 60.0)?;
    let dir = crate::voiceover::generated_dir(s, p, FOLDER)?;
    let mut taken = Vec::new();
    let frame = frame_png(s, cmd, at, &dir, &taken)?;
    taken.push(frame.clone());
    let mut refs = vec![frame];
    let extra = references_p(s, cmd, p, &prov)?;
    let most = prov.max_references.map_or(run::MAX_REFERENCES, |m| (m as usize).min(run::MAX_REFERENCES));
    if extra.len() + 1 > most {
        return Err(bad(cmd, format!("{} takes at most {most} references, the frame included", prov.name)));
    }
    refs.extend(extra);
    let (w, h) = run::size(&prov, (None, None), aspect(s));
    let out = unique_path(s, &dir, "Augmented Frame", &ext(&prov), &taken)?;
    taken.push(out.clone());
    let seed = seed_p(s, p);
    let vars = Vars {
        out: out.clone(),
        prompt: run::prompt_for(&prov, &instruction, refs.len()),
        width: Some(w),
        height: Some(h),
        seed,
        references: refs,
        ..Default::default()
    };
    let args = run::argv(&prov, &vars).map_err(|e| bad(cmd, e))?;
    let place = bool_p(p, "place").unwrap_or(true);
    let mut steps = vec![Step { provider: prov.clone(), args, out: out.clone() }];
    let mut lands = Vec::new();
    let over = |still: Option<Tick>, fit: Option<Tick>| Place { above: track + 1, still, fit, ..Place::at(at) };
    match vprov {
        None => {
            // a still over the clip: as long as asked, else to the clip's end (five seconds at most)
            let secs = wanted.unwrap_or_else(|| left.seconds().clamp(0.2, 5.0));
            lands.push(land_of(
                &prov,
                &out,
                &instruction,
                seed,
                &format!("{w}×{h} · augments a frame"),
                true,
                place.then(|| over(Some(Tick::from_seconds_f64(secs)), None)),
            ));
        }
        Some(vp) => {
            let motion = text_p(cmd, p, "motion")?.unwrap_or_else(|| instruction.clone());
            let secs = wanted.unwrap_or_else(|| left.seconds().clamp(1.0, 4.0));
            let (step, made, _) = video_step(s, cmd, &vp, &motion, (Some(out.clone()), None), secs, (None, None), seed, &dir, &taken)?;
            lands.push(land_of(&prov, &out, &instruction, seed, &format!("{w}×{h} · augments a frame"), true, None));
            lands.push(land_of(&vp, &step.out, &motion, seed, &format!("{made:.1} s · from the augmented frame"), true, place.then(|| over(None, None))));
            steps.push(step);
        }
    }
    let mut v = start(s, cmd, format!("Augmenting the scene: {}", short(&instruction)), steps, lands, bool_p(p, "wait").unwrap_or(false))?;
    v["provider"] = json!(prov.id);
    v["seed"] = json!(seed);
    v["at"] = json!(at.0);
    Ok(v)
}

#[cfg(test)]
#[path = "generate_tests.rs"]
mod tests;
