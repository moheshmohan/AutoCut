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
const MAX_PROMPT: usize = 4000;
/// The most gaps one `generate.fillGap` fills.
const MAX_GAPS: usize = 8;
const CONTINUE: &str = "Continue this shot naturally: the same place, light, subjects and camera movement.";
const LEAD_IN: &str = "A shot that leads naturally into this picture: the same place, light and subjects.";

type Outcome = Arc<Mutex<Option<std::result::Result<(), String>>>>;

/// One run of a provider.
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
            r#"{"prompt":str,"negative":str?,"references":[item id|path]? (at most 16),"frameTime":ticks?|"frameSeconds":f64?|"fromPlayhead":bool? (the program frame there as the first reference),"width":px?,"height":px?,"seed":u64?,"steps":n?,"strength":0..1?,"transparent":bool=false,"provider":id?,"dir":str?,"import":bool=true,"place":bool=false,"track":"V2"?,"atTime":ticks?|"atSeconds":f64?,"seconds":0.1..600=5,"wait":bool=false}"#,
            image,
            true,
        ),
        spec(
            "generate.video",
            "Generate Video",
            r#"{"prompt":str,"image":item id|path|"playhead"?,"seconds":0.1..60=4,"width":px?,"height":px?,"seed":u64?,"audio":bool=false,"provider":id?,"dir":str?,"place":bool=true,"track":"V2"?,"atTime":ticks?|"atSeconds":f64?,"wait":bool=false}"#,
            video,
            true,
        ),
        spec(
            "generate.sound",
            "Generate Sound",
            r#"{"prompt":str,"kind":"sfx"|"music"="sfx","seconds":0.1..600?,"seed":u64?,"provider":id?,"dir":str?,"place":bool=true,"track":"A2"?,"atTime":ticks?|"atSeconds":f64?,"wait":bool=false}"#,
            sound,
            true,
        ),
        spec(
            "generate.fillGap",
            "Fill Gap with Generated Video",
            r#"{"track":"V1"?,"atTime":ticks?|"atSeconds":f64? (a time inside the gap; default the playhead),"all":bool=false,"from":"before"|"after"|"prompt"="before","prompt":str?,"seed":u64?,"provider":id?,"dir":str?,"wait":bool=false}"#,
            fill_gap,
            true,
        ),
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
        let mut tick = |gone: f64| {
            // the tool says nothing while it works: the bar follows its usual time and never fills early
            let part = (gone / est).min(0.97);
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
    s.edit("Generate", |pr, _| {
        if let Some(it) = pr.item_mut(ItemId(item)) {
            it.metadata.insert("Description".into(), description);
            it.metadata.insert("Log Note".into(), log);
            it.metadata.insert("Generator".into(), generator);
            if !attribution.is_empty() {
                it.metadata.insert("Credit".into(), attribution);
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
    let out = unique_path(s, &dir, "Generated Image", &ext(&prov), &taken)?;
    taken.push(out.clone());
    let seed = seed_p(s, p);
    // a cut-out: the tool's own, else its cut-out program run on the picture it made
    let transparent = bool_p(p, "transparent").unwrap_or(false);
    let cutter = if transparent && !prov.optional.contains_key("transparent") {
        Some(
            run::cutout(&prov)
                .ok_or_else(|| bad(cmd, format!("{} cannot make a transparent picture (it has no `transparent` option and no `cutoutCommand`)", prov.name)))?,
        )
    } else {
        None
    };
    let source = match &cutter {
        Some(_) => unique_path(s, &dir, "Generated Image source", &ext(&prov), &taken)?,
        None => out.clone(),
    };
    let asked = match (&cutter, &prov.cutout_prompt_suffix) {
        (Some(_), Some(suffix)) => format!("{prompt}{suffix}"),
        _ => prompt.clone(),
    };
    let vars = Vars {
        out: source.clone(),
        prompt: run::prompt_for(&prov, &asked, refs.len()),
        negative,
        width: Some(w),
        height: Some(h),
        seed,
        references: refs,
        steps,
        strength,
        transparent: transparent && cutter.is_none(),
        ..Default::default()
    };
    let args = run::argv(&prov, &vars).map_err(|e| bad(cmd, e))?;
    let mut run_steps = vec![Step { provider: prov.clone(), args, out: source.clone() }];
    if let Some(cut) = cutter {
        let cut_vars = Vars { out: out.clone(), image: Some(source), ..Default::default() };
        let args = run::argv(&cut, &cut_vars).map_err(|e| bad(cmd, e))?;
        run_steps.push(Step { provider: cut, args, out: out.clone() });
    }
    let import = bool_p(p, "import").unwrap_or(true);
    let place = (import && bool_p(p, "place").unwrap_or(false) && s.state.active_sequence.is_some()).then(|| Place {
        at: time_p(s, p, "at").unwrap_or(s.playhead()),
        track: str_p(p, "track").map(str::to_string),
        above: 0,
        still: Some(Tick::from_seconds_f64(still)),
        fit: None,
        audio: false,
    });
    let lands = vec![land_of(&prov, &out, &prompt, seed, &format!("{w}×{h}"), import, place)];
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
    image: Option<String>,
    seconds: f64,
    want: (Option<u32>, Option<u32>),
    seed: u64,
    dir: &str,
    taken: &[String],
) -> Result<(Step, f64, (u32, u32))> {
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
    let dir = crate::voiceover::generated_dir(s, p, FOLDER)?;
    let image = start_image(s, cmd, p, &dir)?;
    let seed = seed_p(s, p);
    let (step, made, (w, h)) = video_step(s, cmd, &prov, &prompt, image, seconds, want, seed, &dir, &[])?;
    let place = (bool_p(p, "place").unwrap_or(true) && s.state.active_sequence.is_some()).then(|| Place {
        at: time_p(s, p, "at").unwrap_or(s.playhead()),
        track: str_p(p, "track").map(str::to_string),
        above: 0,
        still: None,
        fit: None,
        audio: bool_p(p, "audio").unwrap_or(false),
    });
    let lands = vec![land_of(&prov, &step.out, &prompt, seed, &format!("{made:.1} s · {w}×{h}"), true, place)];
    let mut v = start(s, cmd, format!("Generating a clip: {}", short(&prompt)), vec![step], lands, bool_p(p, "wait").unwrap_or(false))?;
    v["provider"] = json!(prov.id);
    v["seed"] = json!(seed);
    v["seconds"] = json!(made);
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
    let seconds = run::seconds(&prov, seconds_p(cmd, p, 600.0)?.unwrap_or(if kind == Kind::Sfx { 5.0 } else { 20.0 }), 5.0);
    let dir = crate::voiceover::generated_dir(s, p, FOLDER)?;
    let out = unique_path(s, &dir, if kind == Kind::Sfx { "Generated Sound" } else { "Generated Music" }, &ext(&prov), &[])?;
    let seed = seed_p(s, p);
    let vars = Vars { out: out.clone(), prompt: prompt.clone(), seed, seconds: Some(seconds), ..Default::default() };
    let args = run::argv(&prov, &vars).map_err(|e| bad(cmd, e))?;
    let place = (bool_p(p, "place").unwrap_or(true) && s.state.active_sequence.is_some()).then(|| Place {
        at: time_p(s, p, "at").unwrap_or(s.playhead()),
        track: str_p(p, "track").map(str::to_string),
        above: 0,
        still: None,
        fit: Some(Tick::from_seconds_f64(seconds)),
        audio: false,
    });
    let lands = vec![land_of(&prov, &out, &prompt, seed, &format!("{seconds:.1} s"), true, place)];
    let mut v = start(
        s,
        cmd,
        format!("Generating sound: {}", short(&prompt)),
        vec![Step { provider: prov.clone(), args, out }],
        lands,
        bool_p(p, "wait").unwrap_or(false),
    )?;
    v["provider"] = json!(prov.id);
    v["seed"] = json!(seed);
    v["seconds"] = json!(seconds);
    Ok(v)
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
    if !matches!(from, "before" | "after" | "prompt") {
        return Err(bad(cmd, format!("unknown `from` `{from}` (before, after, prompt)")));
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
        // carry on from the last frame before the gap; a gap at the very start leads into what follows
        let lead_in = from == "after" || (from == "before" && g.start <= Tick::ZERO);
        let image = match from {
            "prompt" => None,
            _ => {
                let t = if lead_in { g.end() } else { g.start - frame };
                let f = frame_png(s, cmd, t, &dir, &taken)?;
                taken.push(f.clone());
                Some(f)
            }
        };
        let prompt = own.clone().unwrap_or_else(|| if lead_in { LEAD_IN.to_string() } else { CONTINUE.to_string() });
        let seed = seed_p(s, p);
        let (step, made, (w, h)) = video_step(s, cmd, &prov, &prompt, image, g.duration.seconds(), (None, None), seed, &dir, &taken)?;
        taken.push(step.out.clone());
        let place = Place { at: g.start, track: Some(format!("V{}", index + 1)), above: index, still: None, fit: Some(g.duration), audio: false };
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
    let over = |still: Option<Tick>, fit: Option<Tick>| Place { at, track: None, above: track + 1, still, fit, audio: false };
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
            let (step, made, _) = video_step(s, cmd, &vp, &motion, Some(out.clone()), secs, (None, None), seed, &dir, &taken)?;
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
