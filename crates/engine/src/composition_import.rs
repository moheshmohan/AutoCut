//! `composition.import` (PX3.3): a RAEA composition comes in as the edit it is, under one rule
//! (from Remotion's OTIO export): every visible or audible element has exactly one owner.
//!
//! RAEA (0.1.8 and later) splits a composition with `raea export layers`: what it draws becomes
//! movies of their own (the graphics and captions transparent, what lies under the footage
//! opaque, and a backing layer behind a picture in picture), while the footage and sounds the
//! composition uses stay the files they are, with their times, trims, speeds, gains, opacity and
//! boxes in `layers.json`. This command runs that export under the GPU lock (a job, like
//! `motion.render`), checks every file and number in `layers.json`, and lays it out in AutoCut:
//!
//! - footage and sounds are imported as the real clips (an item already in the project is
//!   reused), at their times, from their trims, at their speed, gain and opacity, footage scaled
//!   into its box;
//! - each layer RAEA drew is one clip on a track of its own, in RAEA's order (the lowest layer on
//!   the lowest track);
//! - a footage file's own sound comes along on an audio track of its own; RAEA's drawn layers
//!   carry no sound, so nothing is heard twice;
//! - into a new sequence of the composition's size and rate, or with `place` into the open
//!   sequence at a time, on tracks added above and below what is there (nothing is overwritten).
//!
//! What AutoCut cannot do with its own effects is said in `approximated` (a footage box that
//! RAEA crops is shown uncropped). RAEA's `reference.mp4`, the composition as one render, stays
//! in the folder to compare with.
#![cfg_attr(target_arch = "wasm32", allow(dead_code, unused_imports, unused_variables))]

use std::path::{Component, Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};

use serde_json::{Value, json};

use filmcraft_project::{ClipId, ItemId};
use filmcraft_time::{FrameRate, Tick};

use crate::commands::{CommandSpec, always, bad};
use crate::motion_graphics::{self as mg, Outcome, Pending, PlaceAt};
use crate::{EngineError, Result, Session};

pub(crate) const CMD: &str = "composition.import";
/// RAEA's description of the layers.
pub const LAYERS_JSON: &str = "layers.json";
const LAYERS_MAX: u64 = 4 << 20;
const MAX_FILES: usize = 256;
const MAX_TRACKS: usize = 64;
const MAX_CLIPS: usize = 4096;
/// The longest composition taken: a day.
const MAX_SECONDS: f64 = 86_400.0;
const EXPORT_SECONDS: f64 = 3600.0;
const STAGING: &str = ".raea-layers";

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Import Composition as Layers",
        menu: &[],
        shortcut: None,
        params: r#"{"source":path (an absolute .tsx composition),"composition":id?,"output":path (a new or empty folder for RAEA's layers),"alphaCodec":"prores"|"png"="prores" (how RAEA writes its transparent layers),"place":{"time":ticks?|"seconds":f64?}? (into the open sequence, on tracks of their own; default a new sequence of the composition's size and rate),"wait":bool=false}"#,
        enabled: always,
        run: import,
        journal: true,
    }]
}

/// One file `layers.json` names.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PlanFile {
    pub path: PathBuf,
    /// Drawn by RAEA (in the export folder), not the user's own media.
    pub made: bool,
    pub kind: FileKind,
    pub has_audio: bool,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FileKind {
    Video,
    Image,
    Audio,
}

/// One clip of a layer, in the composition's frames.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PlanClip {
    pub file: usize,
    pub start: u64,
    pub frames: u64,
    /// Seconds into the file.
    pub source_in: f64,
    pub speed: f64,
    pub opacity: f64,
    pub gain_db: Option<f64>,
    pub muted: bool,
    /// `[x, y, w, h]` in the composition's pixels, and how the picture fills it.
    pub frame_box: Option<[f64; 4]>,
    pub fit: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PlanTrack {
    pub layer: String,
    pub name: String,
    /// Sounds only (an audio track); else pictures (a video track).
    pub sound: bool,
    pub clips: Vec<PlanClip>,
}

/// `layers.json`, checked: the composition's size, rate and length, its files and its tracks
/// (pictures lowest first, then sounds).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Plan {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub frames: u64,
    pub files: Vec<PlanFile>,
    pub tracks: Vec<PlanTrack>,
}

fn num(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
}

/// A relative path that stays inside the folder it is relative to.
fn inside(rel: &str) -> bool {
    let p = Path::new(rel);
    !rel.is_empty() && p.is_relative() && p.components().all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
}

/// Read and check RAEA's `layers.json` in `dir`: every index, number and path.
pub(crate) fn parse_layers(v: &Value, dir: &Path) -> std::result::Result<Plan, String> {
    if v.get("version").and_then(Value::as_u64) != Some(1) {
        return Err(format!("layers.json version {} is not one AutoCut reads (1)", v.get("version").cloned().unwrap_or(Value::Null)));
    }
    let side = |k: &str| v.get(k).and_then(Value::as_u64).filter(|n| (1..=16_384).contains(n)).and_then(|n| u32::try_from(n).ok());
    let (width, height) = (side("width").ok_or("layers.json: `width` is 1 to 16384")?, side("height").ok_or("layers.json: `height` is 1 to 16384")?);
    let fps = num(v, "fps").filter(|f| (1.0..=240.0).contains(f)).ok_or("layers.json: `fps` is 1 to 240")?;
    let most_frames = (fps * MAX_SECONDS) as u64;
    let frames = v.get("frames").and_then(Value::as_u64).filter(|n| (1..=most_frames).contains(n)).ok_or("layers.json: `frames` is 1 to a day of frames")?;
    let name: String = v.get("name").and_then(Value::as_str).unwrap_or("Composition").chars().filter(|c| !c.is_control()).take(120).collect();
    let list = v
        .get("files")
        .and_then(Value::as_array)
        .filter(|a| a.len() <= MAX_FILES)
        .ok_or_else(|| format!("layers.json: `files` is a list of at most {MAX_FILES}"))?;
    let mut files = Vec::with_capacity(list.len());
    for (i, f) in list.iter().enumerate() {
        let raw = f.get("path").and_then(Value::as_str).filter(|p| p.len() <= 4096 && !p.contains('\0')).ok_or(format!("layers.json: file {i} has no path"))?;
        let (path, made) = if Path::new(raw).is_absolute() {
            (PathBuf::from(raw), false)
        } else if inside(raw) {
            (dir.join(raw), true)
        } else {
            return Err(format!("layers.json: file {i} ({raw}) leaves the export folder"));
        };
        if !path.is_file() {
            return Err(format!("the composition uses {}, which is not there", path.display()));
        }
        let kind = match f.get("kind").and_then(Value::as_str) {
            Some("video") => FileKind::Video,
            Some("image") => FileKind::Image,
            Some("audio") => FileKind::Audio,
            other => return Err(format!("layers.json: file {i} is of an unknown kind {}", other.unwrap_or("(none)").chars().take(20).collect::<String>())),
        };
        let dim = |k: &str| f.get(k).and_then(Value::as_u64).filter(|n| *n <= 65_536).and_then(|n| u32::try_from(n).ok()).unwrap_or(0);
        files.push(PlanFile {
            path,
            made,
            kind,
            has_audio: f.get("has_audio").and_then(Value::as_bool).unwrap_or(false),
            width: dim("width"),
            height: dim("height"),
        });
    }
    let raw_tracks = v
        .get("tracks")
        .and_then(Value::as_array)
        .filter(|a| a.len() <= MAX_TRACKS)
        .ok_or_else(|| format!("layers.json: `tracks` is a list of at most {MAX_TRACKS}"))?;
    let mut tracks = Vec::with_capacity(raw_tracks.len());
    let mut count = 0usize;
    for (t, tr) in raw_tracks.iter().enumerate() {
        let text = |k: &str| tr.get(k).and_then(Value::as_str).unwrap_or("").chars().filter(|c| !c.is_control()).take(64).collect::<String>();
        let clips_raw = tr.get("clips").and_then(Value::as_array).ok_or(format!("layers.json: track {t} has no clips list"))?;
        count = count.saturating_add(clips_raw.len());
        if count > MAX_CLIPS {
            return Err(format!("layers.json names more than {MAX_CLIPS} clips"));
        }
        let mut clips = Vec::with_capacity(clips_raw.len());
        for (c, cl) in clips_raw.iter().enumerate() {
            let at = |m: &str| format!("layers.json: track {t} clip {c}: {m}");
            let file = cl
                .get("file")
                .and_then(Value::as_u64)
                .and_then(|i| usize::try_from(i).ok())
                .filter(|i| *i < files.len())
                .ok_or_else(|| at("`file` names no file"))?;
            let start = cl.get("start").and_then(Value::as_u64).filter(|s| *s <= most_frames).ok_or_else(|| at("`start` is a frame within a day"))?;
            let length =
                cl.get("frames").and_then(Value::as_u64).filter(|n| (1..=most_frames).contains(n)).ok_or_else(|| at("`frames` is 1 to a day of frames"))?;
            let source_in = match cl.get("source_in") {
                None | Some(Value::Null) => 0.0,
                Some(x) => x.as_f64().filter(|s| s.is_finite() && (0.0..=MAX_SECONDS).contains(s)).ok_or_else(|| at("`source_in` is 0 to a day of seconds"))?,
            };
            let speed = match cl.get("speed") {
                None | Some(Value::Null) => 1.0,
                Some(x) => x.as_f64().filter(|s| s.is_finite() && (0.01..=100.0).contains(s)).ok_or_else(|| at("`speed` is 0.01 to 100"))?,
            };
            let opacity = match cl.get("opacity") {
                None | Some(Value::Null) => 1.0,
                Some(x) => x.as_f64().filter(|o| o.is_finite() && (0.0..=1.0).contains(o)).ok_or_else(|| at("`opacity` is 0 to 1"))?,
            };
            let gain_db = match cl.get("gain_db") {
                None | Some(Value::Null) => None,
                Some(x) => Some(x.as_f64().filter(|g| g.is_finite() && (-300.0..=48.0).contains(g)).ok_or_else(|| at("`gain_db` is -300 to 48"))?),
            };
            let frame_box = match cl.get("box") {
                None | Some(Value::Null) => None,
                Some(b) => {
                    let r: [f64; 4] = serde_json::from_value(b.clone()).map_err(|_| at("`box` is [x, y, w, h]"))?;
                    if !r.iter().all(|x| x.is_finite() && x.abs() <= 1.0e6) || r[2] <= 0.0 || r[3] <= 0.0 {
                        return Err(at("`box` is [x, y, w, h] with a positive size"));
                    }
                    Some(r)
                }
            };
            let fit = cl.get("fit").and_then(Value::as_str).map(|f| f.chars().take(16).collect());
            let muted = cl.get("muted").and_then(Value::as_bool).unwrap_or(false) || gain_db.is_some_and(|g| g <= -300.0);
            clips.push(PlanClip { file, start, frames: length, source_in, speed, opacity, gain_db, muted, frame_box, fit });
        }
        let audio = clips.iter().filter(|c| files.get(c.file).is_some_and(|f| f.kind == FileKind::Audio)).count();
        if audio != 0 && audio != clips.len() {
            return Err(format!("layers.json: track {t} mixes sounds and pictures"));
        }
        tracks.push(PlanTrack { layer: text("layer"), name: text("name"), sound: audio != 0, clips });
    }
    // RAEA lists its layers from the top down: the lowest goes on the lowest track
    let (mut pictures, sounds): (Vec<PlanTrack>, Vec<PlanTrack>) = tracks.into_iter().partition(|t| !t.sound);
    pictures.reverse();
    pictures.extend(sounds);
    Ok(Plan { name, width, height, fps, frames, files, tracks: pictures })
}

fn import(s: &mut Session, p: &Value) -> Result<Value> {
    mg::poll(s);
    let (source, html) = mg::source_p(CMD, p)?;
    if html {
        return Err(bad(CMD, "RAEA exports the layers of a .tsx composition; render an HTML page with motion.render"));
    }
    let composition = mg::composition_p(CMD, p)?;
    let output = mg::path_p(CMD, p, "output")?.ok_or_else(|| bad(CMD, "`output` is needed: a new folder for the layers"))?;
    let codec = match p.get("alphaCodec") {
        None | Some(Value::Null) => "prores".to_string(),
        Some(Value::String(c)) if c == "prores" || c == "png" => c.clone(),
        Some(_) => return Err(bad(CMD, "`alphaCodec` is prores or png")),
    };
    let place = mg::place_p(s, CMD, p)?;
    let wait = mg::strict_bool(CMD, p, "wait", false)?;
    if cfg!(target_arch = "wasm32") {
        return Err(bad(CMD, "RAEA runs on the desktop app, not in the browser"));
    }
    mg::refuse_while_rendering(s, CMD)?;
    let tool = mg::tool_for(CMD)?;
    let created = mg::revision_dir(CMD, &output)?;
    let file = source.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let what = match &composition {
        Some(id) => format!("{file} ({id})"),
        None => file,
    };
    let job = crate::narration::new_job(s, format!("Importing a composition: {what}"));
    job.progress.total.store(1000, Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let out: Outcome = Default::default();
    let job_id = job.id;
    let plan_slot: Arc<Mutex<Option<Plan>>> = Default::default();
    s.motion_renders.push(Pending::layers(job_id, out.clone(), place, plan_slot.clone(), format!("From RAEA's layers of {what}")));
    let media = (s.services.clone(), s.media.openers.read().unwrap_or_else(PoisonError::into_inner).clone());
    let work = (tool, source, composition, output.clone(), created, codec);
    let spawned = crate::narration::spawn(s, job, wait, "filmcraft-layers", move || {
        let t0 = web_time::Instant::now();
        let (tool, source, composition, output, created, codec) = &work;
        let r = export(tool, source, composition.as_deref(), output, *created, codec, (media.0.as_ref(), &media.1), &prog);
        let status = match &r {
            Ok((_, plan)) => {
                *plan_slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(plan.clone());
                Ok(format!("{} layers exported", plan.tracks.len()))
            }
            Err(v) => Err(v.get("error").and_then(Value::as_str).unwrap_or("the export failed").to_string()),
        };
        *out.lock().unwrap_or_else(PoisonError::into_inner) = Some(r.map(|(v, _)| v));
        crate::narration::finish(&prog, &res, status, t0);
    });
    if let Err(e) = spawned {
        s.motion_renders.forget(job_id);
        mg::free_revision(&output, STAGING, created);
        return Err(e);
    }
    let mut v = json!({"job": job_id, "output": mg::str_of(&output)});
    if wait {
        mg::poll(s);
        let o = mg::outcome_json(s, job_id).unwrap_or(Value::Null);
        if let Some(e) = o.get("error").and_then(Value::as_str) {
            return Err(bad(CMD, e.to_string()));
        }
        if let (Some(dst), Some(src)) = (v.as_object_mut(), o.as_object()) {
            for (k, x) in src {
                dst.insert(k.clone(), x.clone());
            }
        }
    }
    Ok(v)
}

/// The export, in the job: RAEA writes the layers into a staging folder, they move into the
/// revision, and `layers.json` and every file it names are checked.
#[cfg(not(target_arch = "wasm32"))]
fn export(
    tool: &mg::Tool,
    source: &Path,
    composition: Option<&str>,
    output: &Path,
    created: bool,
    codec: &str,
    (services, openers): (&dyn crate::Services, &[filmcraft_media::Opener]),
    prog: &filmcraft_export::Progress,
) -> std::result::Result<(Value, Plan), Value> {
    let status = |t: &str| *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = t.to_string();
    let failed = |e: String| {
        mg::free_revision(output, STAGING, created);
        json!({"error": e, "output": mg::str_of(output), "kept": Value::Null})
    };
    status("Waiting for the GPU");
    let _gpu = mg::gpu("composition layers (RAEA)", std::time::Duration::from_secs(30 * 60)).map_err(failed)?;
    let staging = output.join(STAGING);
    std::fs::create_dir(&staging).map_err(|e| failed(format!("{}: {e}", staging.display())))?;
    let made = staging.join("layers");
    let log = staging.join("raea.log");
    let mut args = vec!["export".to_string(), "layers".into(), mg::str_of(source), "-o".into(), mg::str_of(&made), "--alpha-codec".into(), codec.to_string()];
    if let Some(id) = composition {
        args.extend(["--composition".to_string(), id.to_string()]);
    }
    status("RAEA: exporting the layers");
    let est = tool.estimate.unwrap_or(30.0).clamp(1.0, 86_400.0);
    let mut tick = |gone: f64| prog.done.store(((gone / est).min(0.97) * 800.0) as u64, Ordering::Relaxed);
    let cwd = source.parent().unwrap_or(Path::new("/"));
    let ok = match mg::raea(tool, &args, cwd, &log, &prog.cancel, tool.timeout.unwrap_or(EXPORT_SECONDS), &mut tick) {
        Ok(ok) => ok,
        Err(e) if e == "stopped" => return Err(failed("stopped: nothing was kept".into())),
        Err(e) => return Err(failed(e)),
    };
    if !ok || !made.join(LAYERS_JSON).is_file() {
        let why = if ok { "RAEA finished but wrote no layers.json" } else { "RAEA could not export the layers" };
        return Err(failed(format!("{why}: {}", mg::tail(&log))));
    }
    // the export is finished: it moves into the revision and stays from here on
    let entries: Vec<std::fs::DirEntry> = std::fs::read_dir(&made).map_err(|e| failed(format!("{}: {e}", made.display())))?.flatten().take(10_000).collect();
    for e in entries {
        let to = output.join(e.file_name());
        if let Err(err) = std::fs::rename(e.path(), &to) {
            return Err(
                json!({"error": format!("{}: {err}; the layers are in {}", to.display(), staging.display()), "kept": mg::str_of(&staging), "output": mg::str_of(output)}),
            );
        }
    }
    let _ = std::fs::rename(&log, output.join("raea.log"));
    let _ = std::fs::remove_dir_all(&staging);
    let keep = |e: String| json!({"error": format!("{e}; RAEA's layers are kept in {} and nothing was imported", output.display()), "kept": mg::str_of(output), "output": mg::str_of(output)});
    status("Checking the layers");
    prog.done.store(850, Ordering::Relaxed);
    let layers = mg::read_json(&output.join(LAYERS_JSON), LAYERS_MAX).map_err(keep)?;
    let plan = parse_layers(&layers, output).map_err(keep)?;
    // every file RAEA drew opens, at the composition's size
    let mut warnings = Vec::new();
    for f in plan.files.iter().filter(|f| f.made) {
        let pool = crate::media_pool::MediaPool::default();
        *pool.openers.write().unwrap_or_else(PoisonError::into_inner) = openers.to_vec();
        let src = pool.open_file(&mg::str_of(&f.path), services).map_err(|e| keep(format!("AutoCut cannot read {}: {e}", f.path.display())))?;
        let info = src.info();
        if let Some(video) = &info.video {
            if (video.width, video.height) != (plan.width, plan.height) {
                return Err(keep(format!("{} is {}x{}, not the composition's {}x{}", f.path.display(), video.width, video.height, plan.width, plan.height)));
            }
            let layer = plan.tracks.iter().find(|t| t.clips.iter().any(|c| plan.files.get(c.file) == Some(f))).map(|t| t.layer.as_str()).unwrap_or("");
            if !video.has_alpha && layer != "under" {
                warnings.push(format!("{} ({layer}) has no alpha channel: it covers what is under it", f.path.display()));
            }
        }
    }
    prog.done.store(1000, Ordering::Relaxed);
    let v = json!({
        "output": mg::str_of(output), "layers": mg::str_of(&output.join(LAYERS_JSON)),
        "reference": layers.get("reference").and_then(Value::as_str).filter(|r| inside(r)).map(|r| mg::str_of(&output.join(r))),
        "name": plan.name, "width": plan.width, "height": plan.height, "fps": plan.fps, "frames": plan.frames,
        "notes": layers.get("notes").and_then(Value::as_array).map(|a| a.iter().take(100).cloned().collect::<Vec<_>>()),
        "warnings": warnings,
    });
    Ok((v, plan))
}

#[cfg(target_arch = "wasm32")]
fn export(
    _: &mg::Tool,
    _: &Path,
    _: Option<&str>,
    _: &Path,
    _: bool,
    _: &str,
    _: (&dyn crate::Services, &[filmcraft_media::Opener]),
    _: &filmcraft_export::Progress,
) -> std::result::Result<(Value, Plan), Value> {
    Err(json!({"error": "RAEA runs on the desktop app"}))
}

/// Lays a checked plan out on the timeline (called when the job has finished, as one undo step
/// with the imports).
pub(crate) fn land(s: &mut Session, plan: &Plan, place: Option<&PlaceAt>, note: &str, v: &Value) -> Result<Value> {
    let mut out = v.clone();
    let rate = FrameRate::from_f64(plan.fps).sane();
    let pictures: Vec<&PlanTrack> = plan.tracks.iter().filter(|t| !t.sound).collect();
    let sounds: Vec<&PlanTrack> = plan.tracks.iter().filter(|t| t.sound).collect();
    // a picture track whose footage has sound of its own gets an audio track for it
    let carries_sound =
        |t: &PlanTrack| t.clips.iter().any(|c| !c.muted && plan.files.get(c.file).is_some_and(|f| !f.made && f.kind == FileKind::Video && f.has_audio));
    let own_sound: Vec<bool> = pictures.iter().map(|t| carries_sound(t)).collect();
    let audio_needed = sounds.len() + own_sound.iter().filter(|x| **x).count();
    let (base_v, base_a, at) = match place {
        None => {
            let made = s.execute(
                "file.newSequence",
                json!({"name": plan.name, "width": plan.width, "height": plan.height, "fps": plan.fps, "video": pictures.len().max(1), "audio": audio_needed.max(1)}),
            )?;
            out["sequence"] = made.get("sequence").cloned().unwrap_or(Value::Null);
            (0, 0, Tick::ZERO)
        }
        Some(pl) => {
            let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
            let (v0, a0) = (q.video_tracks.len(), q.audio_tracks.len());
            s.execute("sequence.addTracks", json!({"video": pictures.len(), "audio": audio_needed}))?;
            out["sequence"] = json!(s.state.active_sequence.map(|i| i.0));
            (v0, a0, pl.at)
        }
    };
    let (seq_w, seq_h) = s.active_sequence().map_or((plan.width, plan.height), |q| (q.settings.width.max(1), q.settings.height.max(1)));
    let (kx, ky) = (f64::from(seq_w) / f64::from(plan.width), f64::from(seq_h) / f64::from(plan.height));
    let mut items: Vec<Option<u64>> = vec![None; plan.files.len()];
    let mut item_of = |s: &mut Session, i: usize| -> Result<u64> {
        if let Some(Some(id)) = items.get(i) {
            return Ok(*id);
        }
        let f = plan.files.get(i).ok_or_else(|| EngineError::Other("layers.json names no such file".into()))?;
        let r = s.execute("file.import", json!({"paths": [mg::str_of(&f.path)], "reuse": !f.made}))?;
        let id = r
            .get("items")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_u64)
            .ok_or_else(|| EngineError::Other(format!("{} could not be imported", f.path.display())))?;
        if f.made {
            let note = note.to_string();
            s.edit("Composition", |pr, _| {
                if let Some(it) = pr.item_mut(ItemId(id)) {
                    it.metadata.insert("Log Note".into(), note);
                }
                Ok(())
            })?;
        }
        if let Some(slot) = items.get_mut(i) {
            *slot = Some(id);
        }
        Ok(id)
    };
    let mut placed = Vec::new();
    let mut approximated = Vec::new();
    // (picture track, its footage's own sound track, the layer) for each layer, then the sounds
    let mut audio_slot = base_a;
    let mut jobs: Vec<(String, Option<String>, &PlanTrack)> = Vec::with_capacity(plan.tracks.len());
    for (k, t) in pictures.iter().enumerate() {
        let audio = if own_sound.get(k).copied().unwrap_or(false) {
            audio_slot += 1;
            Some(format!("A{audio_slot}"))
        } else {
            None
        };
        jobs.push((format!("V{}", base_v + k + 1), audio, *t));
    }
    for t in &sounds {
        audio_slot += 1;
        jobs.push((format!("A{audio_slot}"), None, *t));
    }
    for (track, audio_track, t) in jobs {
        for c in &t.clips {
            let Some(f) = plan.files.get(c.file) else { continue };
            let item = item_of(s, c.file)?;
            let time = Tick(at.0.saturating_add(rate.tick_of(i64::try_from(c.start).unwrap_or(i64::MAX)).0));
            let length = rate.tick_of(i64::try_from(c.frames).unwrap_or(i64::MAX));
            // the source it takes: as long as its frames at its speed
            let span = if (c.speed - 1.0).abs() > 1e-9 { Tick::from_seconds_f64(length.seconds() * c.speed) } else { length };
            let mut p = json!({"item": item, "track": track, "time": time.0, "duration": span.0});
            if f.kind != FileKind::Image {
                p["sourceIn"] = json!(Tick::from_seconds_f64(c.source_in).0);
            }
            if !t.sound {
                match &audio_track {
                    Some(a) if f.has_audio && !f.made && !c.muted => p["audioTrack"] = json!(a),
                    _ => p["audio"] = json!(false),
                }
            }
            let r = s.execute("timeline.place", p)?;
            let ids: Vec<u64> = r.get("clips").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
            let Some(&first) = ids.first() else { return Err(EngineError::Other(format!("{} could not be placed", f.path.display()))) };
            if (c.speed - 1.0).abs() > 1e-9 {
                s.execute("clip.speedDuration", json!({"clips": ids, "speed": c.speed * 100.0}))?;
            }
            let (video_clip, audio_clip) = if t.sound { (None, Some(first)) } else { (Some(first), ids.get(1).copied()) };
            if let Some(vc) = video_clip {
                frame(s, vc, f, c, plan, (kx, ky), &mut approximated)?;
                if c.opacity < 1.0 {
                    s.execute("effects.setParam", json!({"clip": vc, "effect": "opacity", "param": "opacity", "value": c.opacity * 100.0}))?;
                }
            }
            if let (Some(ac), Some(g)) = (audio_clip, c.gain_db) {
                s.execute("effects.setParam", json!({"clip": ac, "effect": "volume", "param": "level", "value": g.clamp(-287.5, 15.0)}))?;
                if !(-287.5..=15.0).contains(&g) {
                    approximated.push(json!({"clip": ac, "what": "gain", "says": format!("{g} dB is past AutoCut's clip volume; set to the nearest")}));
                }
            }
            let span_now = s.active_sequence().and_then(|q| q.find_item(ClipId(first)).map(|(_, it)| (it.start.0, it.duration.0)));
            placed.push(json!({
                "layer": t.layer, "name": t.name, "track": track, "audioTrack": if t.sound { Value::Null } else { json!(audio_track) },
                "clip": first, "clips": ids, "item": item, "file": mg::str_of(&f.path), "drawn": f.made,
                "start": span_now.map(|x| x.0), "duration": span_now.map(|x| x.1), "speed": c.speed,
            }));
        }
    }
    out["placed"] = json!(placed);
    out["items"] = json!(items.iter().flatten().collect::<Vec<_>>());
    out["approximated"] = json!(approximated);
    Ok(out)
}

/// A picture's size and place: RAEA's layers fill the frame; footage goes into its box.
fn frame(s: &mut Session, clip: u64, f: &PlanFile, c: &PlanClip, plan: &Plan, (kx, ky): (f64, f64), approximated: &mut Vec<Value>) -> Result<()> {
    let full = |b: &[f64; 4]| b[0].abs() < 0.5 && b[1].abs() < 0.5 && (b[2] - f64::from(plan.width)).abs() < 0.5 && (b[3] - f64::from(plan.height)).abs() < 0.5;
    let contain = c.fit.as_deref() == Some("contain");
    match &c.frame_box {
        None if f.made => {
            if (kx - 1.0).abs() > 1e-9 || (ky - 1.0).abs() > 1e-9 {
                s.execute("clip.fillFrame", json!({"clips": [clip]}))?;
            }
        }
        None => {}
        Some(b) if full(b) => {
            s.execute(if contain { "clip.fitToFrame" } else { "clip.fillFrame" }, json!({"clips": [clip]}))?;
        }
        Some(b) => {
            let (fw, fh) = (f64::from(f.width.max(1)), f64::from(f.height.max(1)));
            let (bw, bh) = (b[2] * kx, b[3] * ky);
            let scale = if contain { (bw / fw).min(bh / fh) } else { (bw / fw).max(bh / fh) };
            // (fit/fill first: it sets the anchor to the picture's centre and drops Scale to Frame Size)
            s.execute("clip.fitToFrame", json!({"clips": [clip]}))?;
            s.execute("effects.setParam", json!({"clip": clip, "effect": "motion", "param": "scale", "value": (scale * 100.0).clamp(0.0, 10_000.0)}))?;
            s.execute(
                "effects.setParam",
                json!({"clip": clip, "effect": "motion", "param": "position", "value": [(b[0] + b[2] / 2.0) * kx, (b[1] + b[3] / 2.0) * ky]}),
            )?;
            if !contain && (fw * scale > bw + 0.5 || fh * scale > bh + 0.5) {
                approximated.push(json!({"clip": clip, "what": "box", "says": "RAEA crops this footage to its box; AutoCut shows it whole, centred on the box (add a Crop to match)"}));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "composition_import_tests.rs"]
mod tests;
