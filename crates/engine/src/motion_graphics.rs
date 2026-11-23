//! Motion graphics as commands (PX3.2): compositions written as code (a TSX file or an HTML page)
//! rendered by RAEA, a renderer the user installs separately (FSL-1.1-ALv2). AutoCut only runs it
//! as an external program and never bundles it ([`docs/raea-workflow.md`]). These commands do what
//! `examples/raea/render.py` does:
//!
//! | command | does |
//! |---|---|
//! | `motion.check` | `raea check --quick` on a composition: its findings (`check`, `level`, `says`) and what passed |
//! | `motion.stills` | stills at the frames `at` and RAEA's render manifest: the stills' paths and the text boxes of each frame |
//! | `motion.render` | a job under the machine's GPU lock: the check, the render into a new revision folder, AutoCut's own probe (the size and rate asked for, no gaps in a PNG sequence), the animation preflight (`media.preflightAnimation`, with `qa.motionEnergy` as warnings), `autocut-import.json`, then the import (`file.import`, or `file.importImageSequence` with `fps` for a transparent sequence) and, with `place`, the clip on the timeline |
//!
//! `composition.import` ([`crate::composition_import`]) runs as one of these jobs too: RAEA's
//! layers laid out as the edit they are.
//!
//! **Where RAEA is.** A `kind: "composition"` entry with the id `raea` in `providers.json`
//! ([`crate::generate_run`]) names its program; without one (or with an empty `command`) the
//! newest `~/.raea/versions/<version>/bin/raea` (HOME read when the command runs), else `raea`
//! on PATH. No command parameter chooses the program.
//!
//! **A finished render is kept.** The render goes into a staging folder inside the revision;
//! a refused or failed render takes that away again (and the revision folder when this command
//! made it), so the name is free for the next try. Once RAEA has finished, its frames stay even
//! when the probe or the preflight refuses them (`kept` in the result says where), and
//! `autocut-import.json` is written only after every check passed.
//!
//! **Revision folders.** `output` is a folder that does not exist yet or is empty; one with
//! anything in it is refused, so an earlier render is never overwritten.
//!
//! [`docs/raea-workflow.md`]: https://github.com/storytold/filmcraft/blob/main/docs/raea-workflow.md
#![cfg_attr(target_arch = "wasm32", allow(dead_code, unused_imports, unused_variables))]

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use serde_json::{Value, json};

use filmcraft_project::ItemId;
use filmcraft_time::{FrameRate, Tick, TimeRange};

use crate::commands::{CommandSpec, always, bad};
use crate::generate_run::{self as run, Kind, Provider};
use crate::{EngineError, Result, Services, Session};

const RENDER: &str = "motion.render";
const STILLS: &str = "motion.stills";
const CHECK: &str = "motion.check";
/// What a successful handoff writes into the revision folder (the shape `render.py` writes).
pub const IMPORT_MANIFEST: &str = "autocut-import.json";
/// The id of the composition tool these commands drive.
pub const RAEA_ID: &str = "raea";
const PROGRAM: &str = if cfg!(windows) { "raea.exe" } else { "raea" };
const INSTALL: &str =
    "install it as docs/raea-workflow.md (Setup) describes, or name its program in providers.json as a `composition` entry (docs/generation.md)";
const MAX_PATH: usize = 4096;
const MAX_SIDE: u32 = 16_384;
const MAX_STILLS: usize = 64;
const MAX_STILL_FRAME: u64 = 10_000_000;
/// The most RAEA's render manifest, its check report and a preflight spec may weigh.
const RENDER_JSON_MAX: u64 = 16 << 20;
const CHECK_JSON_MAX: u64 = 4 << 20;
const SPEC_MAX: u64 = 256 << 10;
/// The most findings and text boxes a result lists (the counts are always whole).
const MAX_FINDINGS: usize = 500;
const MAX_TEXTS: usize = 20_000;
/// Version folders looked at under `~/.raea/versions`, and PATH entries.
const MAX_VERSIONS: usize = 1000;
const MAX_PATH_DIRS: usize = 256;
const MAX_OUTCOMES: usize = 256;
/// How long RAEA may take: `--version`, a check, stills, a render (unless its entry says).
const VERSION_SECONDS: f64 = 20.0;
const CHECK_SECONDS: f64 = 300.0;
const STILLS_SECONDS: f64 = 600.0;
const RENDER_SECONDS: f64 = 3600.0;
/// How long `motion.check` and `motion.stills` wait for the GPU; a render job waits longer.
const SHORT_GPU_WAIT: std::time::Duration = std::time::Duration::from_secs(15);
const RENDER_GPU_WAIT: std::time::Duration = std::time::Duration::from_secs(30 * 60);
/// The staging folders inside a revision.
const RENDER_STAGING: &str = ".raea-render";
const STILLS_STAGING: &str = ".raea-stills";

fn spec(id: &'static str, label: &'static str, params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>, journal: bool) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: always, run, journal }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            RENDER,
            "Render Motion Graphics",
            r#"{"source":path (an absolute .tsx or .html composition),"composition":id?,"output":path (a new or empty revision folder),"canvas":"WxH"? (the reference canvas, the output size without `size`),"size":"WxH"? (the output size),"preview":bool=false (a draft whose long side is 960; needs canvas),"fps":1..240? (a transparent sequence's import rate, default RAEA's render rate; an MP4's rate must match it),"alpha":bool=false (a transparent PNG sequence instead of an MP4),"quality":"draft"|"high"|"master"? (MP4; default master, draft with preview),"spec":path? (a QA JSON: expected, holds, textBounds, soundCues),"expected":{"duration":ticks?,"width":u32?,"height":u32?,"fps":f64?}?,"holds":[{"start":ticks,"end":ticks,"reason":str}]?,"textBounds":[..]?,"soundCues":[..]?,"import":bool=true,"place":{"track":"V2"?,"time":ticks?|"seconds":f64?}? (on the timeline; default the playhead, a transparent render above V1),"wait":bool=false}"#,
            render,
            true,
        ),
        spec(
            STILLS,
            "Render Motion Graphics Stills",
            r#"{"source":path,"composition":id?,"at":[frame] (1 to 64 frame numbers),"output":path (a new or empty folder)}"#,
            stills,
            false,
        ),
        spec(CHECK, "Check Motion Graphics", r#"{"source":path,"composition":id?}"#, check, false),
    ]
}

// ---------------------------------------------------------------------------------------------
// Finding RAEA

/// RAEA as found: its program, where it was found, and its `providers.json` entry's settings.
#[derive(Clone, Debug)]
pub struct Tool {
    pub program: PathBuf,
    /// `providers.json`, `~/.raea/versions` or `PATH`.
    pub from: &'static str,
    pub env: BTreeMap<String, String>,
    pub timeout: Option<f64>,
    pub estimate: Option<f64>,
}

impl Tool {
    fn json(&self) -> Value {
        json!({"name": "RAEA", "program": self.program.to_string_lossy(), "from": self.from})
    }
}

/// RAEA for `entry` (the `raea` composition entry of `providers.json`, when there is one), with
/// HOME and PATH as this process has them.
pub(crate) fn locate(entry: Option<&Provider>) -> std::result::Result<Tool, String> {
    let home = std::env::var_os("HOME").filter(|h| !h.is_empty()).map(PathBuf::from);
    let path = std::env::var_os("PATH");
    locate_in(entry, home.as_deref(), path.as_deref())
}

/// [`locate`] with HOME and PATH given: the entry's program, else the newest version under
/// `<home>/.raea/versions`, else `raea` in an absolute PATH folder.
pub(crate) fn locate_in(entry: Option<&Provider>, home: Option<&Path>, path: Option<&OsStr>) -> std::result::Result<Tool, String> {
    let tool = |program: PathBuf, from: &'static str| Tool {
        program,
        from,
        env: entry.map(|e| e.env.clone()).unwrap_or_default(),
        timeout: entry.and_then(|e| e.timeout_seconds),
        estimate: entry.and_then(|e| e.estimate_seconds),
    };
    if let Some(program) = entry.and_then(|e| e.command.first()) {
        let p = PathBuf::from(program);
        return if p.is_file() {
            Ok(tool(p, "providers.json"))
        } else {
            Err(format!("RAEA is set in providers.json as {program}, which is not on this computer: {INSTALL}"))
        };
    }
    if let Some(p) = home.and_then(newest_installed) {
        return Ok(tool(p, "~/.raea/versions"));
    }
    if let Some(p) = path.and_then(on_path) {
        return Ok(tool(p, "PATH"));
    }
    Err(format!("RAEA is not installed on this computer: {INSTALL}"))
}

/// `name` as a version (`0.1.8`): one to four whole numbers.
fn parse_version(name: &str) -> Option<Vec<u32>> {
    let parts = name
        .split('.')
        .map(|p| if p.is_empty() || p.len() > 9 || !p.bytes().all(|b| b.is_ascii_digit()) { None } else { p.parse::<u32>().ok() })
        .collect::<Option<Vec<u32>>>()?;
    (1..=4).contains(&parts.len()).then_some(parts)
}

/// The newest `<home>/.raea/versions/<version>/bin/raea`.
fn newest_installed(home: &Path) -> Option<PathBuf> {
    let root = home.join(".raea").join("versions");
    let mut best: Option<(Vec<u32>, PathBuf)> = None;
    for e in std::fs::read_dir(root).ok()?.flatten().take(MAX_VERSIONS) {
        let name = e.file_name();
        let Some(version) = name.to_str().and_then(parse_version) else { continue };
        let program = e.path().join("bin").join(PROGRAM);
        if program.is_file() && best.as_ref().is_none_or(|(v, _)| version > *v) {
            best = Some((version, program));
        }
    }
    best.map(|(_, p)| p)
}

#[cfg(not(target_arch = "wasm32"))]
fn on_path(path: &OsStr) -> Option<PathBuf> {
    std::env::split_paths(path).take(MAX_PATH_DIRS).filter(|d| d.is_absolute()).map(|d| d.join(PROGRAM)).find(|p| p.is_file())
}

#[cfg(target_arch = "wasm32")]
fn on_path(_: &OsStr) -> Option<PathBuf> {
    None
}

/// The `raea` composition entry of `providers.json`, when there is one.
fn composition_entry() -> std::result::Result<Option<Provider>, String> {
    Ok(run::load()?.into_iter().find(|p| p.kind == Kind::Composition && p.id == RAEA_ID))
}

pub(crate) fn tool_for(cmd: &str) -> Result<Tool> {
    let entry = composition_entry().map_err(|e| bad(cmd, e))?;
    locate(entry.as_ref()).map_err(|e| bad(cmd, e))
}

// ---------------------------------------------------------------------------------------------
// Parameters

pub(crate) fn path_p(cmd: &str, p: &Value, key: &str) -> Result<Option<PathBuf>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(t)) => {
            if t.is_empty() || t.len() > MAX_PATH || t.contains('\0') || !Path::new(t).is_absolute() {
                return Err(bad(cmd, format!("`{key}` must be an absolute path")));
            }
            Ok(Some(PathBuf::from(t)))
        }
        Some(_) => Err(bad(cmd, format!("`{key}` must be an absolute path"))),
    }
}

/// The composition file and whether it is an HTML page.
pub(crate) fn source_p(cmd: &str, p: &Value) -> Result<(PathBuf, bool)> {
    let source = path_p(cmd, p, "source")?.ok_or_else(|| bad(cmd, "`source` is needed: the composition's .tsx or .html file"))?;
    let ext = source.extension().and_then(OsStr::to_str).map(str::to_ascii_lowercase).unwrap_or_default();
    if ext != "tsx" && ext != "html" {
        return Err(bad(cmd, "`source` must be a .tsx or .html composition"));
    }
    if !cfg!(target_arch = "wasm32") && !source.is_file() {
        return Err(bad(cmd, format!("the composition {} is not a file", source.display())));
    }
    Ok((source, ext == "html"))
}

/// A composition id: letters, digits, `-`, `_` and `.`, starting with a letter or digit (so it
/// can never read as one of RAEA's options).
pub(crate) fn composition_p(cmd: &str, p: &Value) -> Result<Option<String>> {
    match p.get("composition") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(id))
            if (1..=128).contains(&id.len())
                && id.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
                && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')) =>
        {
            Ok(Some(id.clone()))
        }
        Some(_) => Err(bad(cmd, "`composition` is a composition id: 1 to 128 letters, digits, - _ or ., starting with a letter or digit")),
    }
}

/// `"WxH"`, each side 1 to 16384.
pub(crate) fn size_p(cmd: &str, p: &Value, key: &str) -> Result<Option<(u32, u32)>> {
    let Some(v) = p.get(key).filter(|v| !v.is_null()) else { return Ok(None) };
    let err = || bad(cmd, format!("`{key}` must be \"WxH\", each side 1 to {MAX_SIDE}"));
    let t = v.as_str().ok_or_else(err)?;
    let (w, h) = t.split_once(['x', 'X']).ok_or_else(err)?;
    let side = |s: &str| -> Option<u32> {
        if s.is_empty() || s.len() > 5 || !s.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        s.parse::<u32>().ok().filter(|n| (1..=MAX_SIDE).contains(n))
    };
    Ok(Some((side(w).ok_or_else(err)?, side(h).ok_or_else(err)?)))
}

pub(crate) fn strict_bool(cmd: &str, p: &Value, key: &str, default: bool) -> Result<bool> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(bad(cmd, format!("`{key}` must be true or false"))),
    }
}

pub(crate) fn fps_value(cmd: &str, v: &Value, key: &str) -> Result<f64> {
    v.as_f64().filter(|f| f.is_finite() && (1.0..=240.0).contains(f)).ok_or_else(|| bad(cmd, format!("`{key}` must be a frame rate from 1 to 240")))
}

/// Read a JSON file of at most `max` bytes.
pub(crate) fn read_json(path: &Path, max: u64) -> std::result::Result<Value, String> {
    use std::io::Read as _;
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut text = String::new();
    f.take(max.saturating_add(1)).read_to_string(&mut text).map_err(|e| format!("{}: {e}", path.display()))?;
    if text.len() as u64 > max {
        return Err(format!("{} is larger than {} KiB", path.display(), max / 1024));
    }
    serde_json::from_str(&text).map_err(|e| format!("{}: not valid JSON: {e}", path.display()))
}

/// The preflight spec: the `spec` file's keys, then the parameters' (each `expected` key the
/// parameters name wins), checked before anything renders.
fn qa_spec(cmd: &str, p: &Value) -> Result<(Value, Option<PathBuf>)> {
    let spec_path = path_p(cmd, p, "spec")?;
    let mut spec = match &spec_path {
        Some(path) => read_json(path, SPEC_MAX).map_err(|e| bad(cmd, e))?,
        None => json!({}),
    };
    if !spec.is_object() {
        return Err(bad(cmd, "the preflight spec must be a JSON object"));
    }
    for key in ["holds", "textBounds", "soundCues"] {
        if let Some(v) = p.get(key).filter(|v| !v.is_null()) {
            spec[key] = v.clone();
        }
    }
    match (spec.get("expected").cloned(), p.get("expected").filter(|v| !v.is_null())) {
        (Some(Value::Object(mut base)), Some(Value::Object(more))) => {
            base.extend(more.clone());
            spec["expected"] = Value::Object(base);
        }
        (None | Some(Value::Null), Some(Value::Object(more))) => spec["expected"] = Value::Object(more.clone()),
        (Some(Value::Object(_)), None) => {}
        (None | Some(Value::Null), None) => spec["expected"] = json!({}),
        _ => return Err(bad(cmd, "preflight `expected` must be a JSON object")),
    }
    let expected = &spec["expected"];
    if let Some(d) = expected.get("duration").filter(|v| !v.is_null())
        && !d.as_i64().is_some_and(|t| t > 0)
    {
        return Err(bad(cmd, "`expected.duration` is ticks, more than 0"));
    }
    for key in ["width", "height"] {
        if let Some(v) = expected.get(key).filter(|v| !v.is_null())
            && !v.as_u64().is_some_and(|n| (1..=u64::from(MAX_SIDE)).contains(&n))
        {
            return Err(bad(cmd, format!("`expected.{key}` must be 1 to {MAX_SIDE}")));
        }
    }
    if let Some(v) = expected.get("fps").filter(|v| !v.is_null()) {
        fps_value(cmd, v, "expected.fps")?;
    }
    if expected.get("alpha").is_some_and(|v| !v.is_null() && !v.is_boolean()) {
        return Err(bad(cmd, "`expected.alpha` must be true or false"));
    }
    // the holds' shapes and reasons now; that each lies inside the film once it is rendered
    crate::motion_energy::holds_p(cmd, &spec, "holds", Tick(i64::MAX))?;
    for key in ["textBounds", "soundCues"] {
        if let Some(v) = spec.get(key).filter(|v| !v.is_null())
            && !v.as_array().is_some_and(|a| a.len() <= 2000)
        {
            return Err(bad(cmd, format!("`{key}` is a list of at most 2000")));
        }
    }
    Ok((spec, spec_path))
}

/// Where a finished render goes on the timeline.
#[derive(Clone, Debug)]
pub(crate) struct PlaceAt {
    pub(crate) track: Option<String>,
    pub(crate) at: Tick,
}

pub(crate) fn place_p(s: &Session, cmd: &str, p: &Value) -> Result<Option<PlaceAt>> {
    let v = match p.get("place") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => return Ok(None),
        Some(Value::Bool(true)) => json!({}),
        Some(v @ Value::Object(_)) => v.clone(),
        Some(_) => return Err(bad(cmd, "`place` is {\"track\": \"V2\"?, \"time\": ticks?} or true")),
    };
    if s.state.active_sequence.is_none() {
        return Err(bad(cmd, "`place` needs an open sequence"));
    }
    let track = match v.get("track") {
        None | Some(Value::Null) => None,
        Some(Value::String(t)) if t.len() <= 5 && t.strip_prefix(['V', 'v']).is_some_and(|n| n.parse::<u32>().is_ok_and(|n| (1..=999).contains(&n))) => {
            Some(t.to_ascii_uppercase())
        }
        Some(_) => return Err(bad(cmd, "`place.track` is a video track: V1, V2…")),
    };
    let at = crate::commands::time_p(s, &v, "").unwrap_or(s.playhead());
    if at < Tick::ZERO || at.seconds() > 24.0 * 3600.0 {
        return Err(bad(cmd, "`place.time` must be from 0 to 24 hours"));
    }
    Ok(Some(PlaceAt { track, at }))
}

/// A new revision folder: made (with its parents) when it is not there, else it must be an
/// empty folder. Whether this call made it.
pub(crate) fn revision_dir(cmd: &str, output: &Path) -> Result<bool> {
    match std::fs::symlink_metadata(output) {
        Ok(m) if m.is_dir() => {
            let mut entries = std::fs::read_dir(output).map_err(|e| bad(cmd, format!("{}: {e}", output.display())))?;
            if entries.next().is_some() {
                return Err(bad(
                    cmd,
                    format!("{} already holds files: render each revision into a new folder, so no earlier render is overwritten", output.display()),
                ));
            }
            Ok(false)
        }
        Ok(_) => Err(bad(cmd, format!("{} is not a folder", output.display()))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir_all(output).map_err(|e| bad(cmd, format!("{}: {e}", output.display())))?;
            Ok(true)
        }
        Err(e) => Err(bad(cmd, format!("{}: {e}", output.display()))),
    }
}

/// Frees a revision that kept nothing: its staging folder, and the folder itself when this call
/// made it and it is empty again.
pub(crate) fn free_revision(output: &Path, staging: &str, created: bool) {
    let _ = std::fs::remove_dir_all(output.join(staging));
    if created {
        let _ = std::fs::remove_dir(output);
    }
}

pub(crate) fn refuse_while_rendering(s: &Session, cmd: &str) -> Result<()> {
    match s.motion_renders.pending.first() {
        Some(q) => Err(bad(cmd, format!("a motion graphics render is running (job {}); wait for it or stop it (jobs.cancel)", q.job))),
        None => Ok(()),
    }
}

// ---------------------------------------------------------------------------------------------
// Running RAEA

/// The end of a log, without terminal colour codes: for an error message.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn tail(log: &Path) -> String {
    let mut out = String::new();
    let mut escape = false;
    for c in run::log_tail(log).chars() {
        if escape {
            escape = !c.is_ascii_alphabetic();
        } else if c == '\u{1b}' {
            escape = true;
        } else if !c.is_control() {
            out.push(c);
        }
    }
    out
}

/// Runs RAEA with `args` in `cwd`, its output appended to `log`, until it exits (whether it
/// succeeded), is stopped through `cancel`, or runs past `limit` seconds (then it is killed).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn raea(
    tool: &Tool,
    args: &[String],
    cwd: &Path,
    log: &Path,
    cancel: &AtomicBool,
    limit: f64,
    tick: &mut dyn FnMut(f64),
) -> std::result::Result<bool, String> {
    use std::io::Write as _;
    use std::process::{Command, Stdio};
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(log).map_err(|e| format!("{}: {e}", log.display()))?;
    let _ = writeln!(f, "$ raea {}", args.join(" "));
    let f2 = f.try_clone().map_err(|e| format!("{}: {e}", log.display()))?;
    let mut child = Command::new(&tool.program)
        .args(args)
        .envs(&tool.env)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::from(f))
        .stderr(Stdio::from(f2))
        .spawn()
        .map_err(|e| format!("could not start RAEA ({}): {e}", tool.program.display()))?;
    let limit = limit.clamp(1.0, 86_400.0);
    let t0 = web_time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status.success()),
            Ok(None) => {}
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("RAEA: {e}"));
            }
        }
        let gone = t0.elapsed().as_secs_f64();
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            return Err("stopped".into());
        }
        if gone > limit {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("RAEA did not finish in {limit:.0} s and was stopped"));
        }
        tick(gone);
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// The machine's GPU lock (a test binary's own), waiting up to `wait`; unlocked when there is no
/// place for locks.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn gpu(what: &str, wait: std::time::Duration) -> std::result::Result<Option<filmcraft_locks::Guard>, String> {
    match filmcraft_locks::acquire(filmcraft_locks::Resource::Gpu, what, wait, &mut |_| {}) {
        Ok(g) => Ok(Some(g)),
        Err(e @ filmcraft_locks::LockError::Busy(..)) => Err(format!("{e}; try again when it is free (`filmcraft-cli locks`)")),
        Err(filmcraft_locks::LockError::Io(_)) => Ok(None),
    }
}

fn nonce() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn str_of(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// What RAEA's `check` wrote: its findings (at most [`MAX_FINDINGS`] listed), the counts of each
/// level and what passed.
fn check_summary(path: &Path) -> std::result::Result<Value, String> {
    let v = read_json(path, CHECK_JSON_MAX)?;
    let all = v.get("findings").and_then(Value::as_array).ok_or_else(|| "RAEA's check report lists no findings".to_string())?;
    let level = |f: &Value, l: &str| f.get("level").and_then(Value::as_str) == Some(l);
    let count = |l: &str| all.iter().filter(|f| level(f, l)).count();
    let findings: Vec<Value> = all
        .iter()
        .take(MAX_FINDINGS)
        .map(|f| json!({"check": f.get("check"), "level": f.get("level"), "says": f.get("says"), "frames": f.get("frames"), "fix": f.get("fix")}))
        .collect();
    let passed: Vec<Value> =
        v.get("passed").and_then(Value::as_array).map(|a| a.iter().filter(|x| x.is_string()).take(100).cloned().collect()).unwrap_or_default();
    let errors = count("error");
    Ok(json!({
        "findings": findings, "truncated": all.len() > MAX_FINDINGS, "errors": errors, "warns": count("warn"), "notes": count("note"),
        "passed": passed, "ok": errors == 0, "path": str_of(path),
    }))
}

/// RAEA's render manifest (`<name>.render.json`) in `dir`, when there is one: the first by name.
fn render_manifest_in(dir: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .take(100_000)
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.file_name().and_then(OsStr::to_str).is_some_and(|n| n.ends_with(".render.json")))
        .collect();
    found.sort();
    found.into_iter().next()
}

/// Width, height, frame rate and length (frames) a render manifest states.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Stated {
    pub(crate) width: Option<u32>,
    pub(crate) height: Option<u32>,
    pub(crate) fps: Option<f64>,
    pub(crate) length: Option<u64>,
}

fn stated(v: &Value) -> Stated {
    let side = |k: &str| v.get(k).and_then(Value::as_u64).filter(|n| (1..=u64::from(MAX_SIDE)).contains(n)).and_then(|n| u32::try_from(n).ok());
    Stated {
        width: side("width"),
        height: side("height"),
        fps: v.get("fps").and_then(Value::as_f64).filter(|f| f.is_finite() && (1.0..=240.0).contains(f)),
        length: v.get("length").and_then(Value::as_u64).filter(|n| *n <= 100_000_000),
    }
}

/// The text boxes of each frame of a render manifest (at most [`MAX_TEXTS`] in all), and whether
/// some were left out.
fn text_boxes(v: &Value) -> (Vec<Value>, bool) {
    let frames = v.get("frames").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]);
    let total: usize = frames.iter().map(|f| f.get("text").and_then(Value::as_array).map_or(0, Vec::len)).fold(0, usize::saturating_add);
    let mut out = Vec::new();
    let mut room = MAX_TEXTS;
    for f in frames {
        if room == 0 {
            break;
        }
        let kept: Vec<Value> = f.get("text").and_then(Value::as_array).map(|a| a.iter().take(room).cloned().collect()).unwrap_or_default();
        room = room.saturating_sub(kept.len());
        out.push(json!({"frame": f.get("frame"), "text": kept}));
    }
    (out, total > MAX_TEXTS)
}

// ---------------------------------------------------------------------------------------------
// motion.check

fn check(s: &mut Session, p: &Value) -> Result<Value> {
    poll(s);
    let (source, _html) = source_p(CHECK, p)?;
    let composition = composition_p(CHECK, p)?;
    if cfg!(target_arch = "wasm32") {
        return Err(bad(CHECK, "RAEA runs on the desktop app, not in the browser"));
    }
    refuse_while_rendering(s, CHECK)?;
    let tool = tool_for(CHECK)?;
    let dir = std::env::temp_dir().join(format!("filmcraft-motion-check-{}-{}", std::process::id(), nonce()));
    std::fs::create_dir_all(&dir).map_err(|e| EngineError::Other(format!("{}: {e}", dir.display())))?;
    let r = run_check(&tool, &source, composition.as_deref(), &dir.join("check.json"), &dir.join("raea.log"), &AtomicBool::new(false), true);
    let _ = std::fs::remove_dir_all(&dir);
    let mut v = r.map_err(|e| bad(CHECK, e))?;
    v["path"] = Value::Null;
    v["source"] = json!(str_of(&source));
    v["composition"] = json!(composition);
    v["tool"] = tool.json();
    Ok(v)
}

/// `raea check SOURCE --quick [--composition ID] -o OUT`, holding the GPU when `lock` says.
#[cfg(not(target_arch = "wasm32"))]
fn run_check(
    tool: &Tool,
    source: &Path,
    composition: Option<&str>,
    out: &Path,
    log: &Path,
    cancel: &AtomicBool,
    lock: bool,
) -> std::result::Result<Value, String> {
    let _gpu = if lock { gpu("motion graphics check (RAEA)", SHORT_GPU_WAIT)? } else { None };
    let mut args = vec!["check".to_string(), str_of(source), "--quick".into()];
    // (RAEA 0.1.8's quick check takes the option and looks at the first composition still)
    if let Some(id) = composition {
        args.extend(["--composition".to_string(), id.to_string()]);
    }
    args.extend(["-o".to_string(), str_of(out)]);
    let cwd = source.parent().unwrap_or(Path::new("/"));
    let ok = raea(tool, &args, cwd, log, cancel, tool.timeout.unwrap_or(CHECK_SECONDS).min(CHECK_SECONDS), &mut |_| {})?;
    match check_summary(out) {
        Ok(mut v) => {
            v["exitedOk"] = json!(ok);
            Ok(v)
        }
        Err(e) if out.is_file() => Err(e),
        Err(_) => Err(format!("RAEA's check {}: {}", if ok { "wrote no report" } else { "failed" }, tail(log))),
    }
}

#[cfg(target_arch = "wasm32")]
fn run_check(_: &Tool, _: &Path, _: Option<&str>, _: &Path, _: &Path, _: &AtomicBool, _: bool) -> std::result::Result<Value, String> {
    Err("RAEA runs on the desktop app".into())
}

// ---------------------------------------------------------------------------------------------
// motion.stills

fn at_p(p: &Value) -> Result<Vec<u64>> {
    let list = p.get("at").and_then(Value::as_array).ok_or_else(|| bad(STILLS, "`at` is needed: a list of frame numbers"))?;
    if list.is_empty() || list.len() > MAX_STILLS {
        return Err(bad(STILLS, format!("`at` names 1 to {MAX_STILLS} frames")));
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::with_capacity(list.len());
    for v in list {
        let n = v.as_u64().filter(|n| *n <= MAX_STILL_FRAME).ok_or_else(|| bad(STILLS, format!("`at` holds frame numbers from 0 to {MAX_STILL_FRAME}")))?;
        if !seen.insert(n) {
            return Err(bad(STILLS, format!("`at` names frame {n} twice")));
        }
        out.push(n);
    }
    Ok(out)
}

fn stills(s: &mut Session, p: &Value) -> Result<Value> {
    poll(s);
    let (source, html) = source_p(STILLS, p)?;
    let composition = composition_p(STILLS, p)?;
    let at = at_p(p)?;
    let output = path_p(STILLS, p, "output")?.ok_or_else(|| bad(STILLS, "`output` is needed: a new folder for the stills"))?;
    if cfg!(target_arch = "wasm32") {
        return Err(bad(STILLS, "RAEA runs on the desktop app, not in the browser"));
    }
    refuse_while_rendering(s, STILLS)?;
    let tool = tool_for(STILLS)?;
    let created = revision_dir(STILLS, &output)?;
    match stills_into(&tool, &source, html, composition.as_deref(), &at, &output) {
        Ok(mut v) => {
            let _ = std::fs::remove_dir_all(output.join(STILLS_STAGING));
            v["composition"] = json!(composition);
            v["tool"] = tool.json();
            Ok(v)
        }
        Err(e) => {
            free_revision(&output, STILLS_STAGING, created);
            Err(bad(STILLS, e))
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn stills_into(tool: &Tool, source: &Path, html: bool, composition: Option<&str>, at: &[u64], output: &Path) -> std::result::Result<Value, String> {
    let staging = output.join(STILLS_STAGING);
    std::fs::create_dir(&staging).map_err(|e| format!("{}: {e}", staging.display()))?;
    let made = staging.join("out");
    let log = staging.join("raea.log");
    let list = at.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
    let mut args = vec!["render".to_string(), str_of(source), "-o".into(), str_of(&made), "--at".into(), list];
    if let Some(id) = composition {
        args.extend(["--composition".to_string(), id.to_string()]);
    }
    let cwd = source.parent().unwrap_or(Path::new("/"));
    {
        let _gpu = gpu("motion graphics stills (RAEA)", SHORT_GPU_WAIT)?;
        let ok = raea(tool, &args, cwd, &log, &AtomicBool::new(false), tool.timeout.unwrap_or(STILLS_SECONDS).min(STILLS_SECONDS), &mut |_| {})?;
        if !ok {
            return Err(format!("RAEA could not render the stills: {}", tail(&log)));
        }
    }
    // an HTML page's stills are named by frame number, a TSX composition's by position
    let names: Vec<String> = at.iter().enumerate().map(|(i, f)| format!("frame-{:05}.png", if html { *f } else { i as u64 })).collect();
    if let Some((f, name)) = at.iter().zip(&names).find(|(_, n)| !made.join(n).is_file()) {
        return Err(format!("RAEA wrote no still for frame {f} ({name}): {}", tail(&log)));
    }
    let entries: Vec<std::fs::DirEntry> = std::fs::read_dir(&made).map_err(|e| format!("{}: {e}", made.display()))?.flatten().take(10_000).collect();
    for e in entries {
        let to = output.join(e.file_name());
        std::fs::rename(e.path(), &to).map_err(|err| format!("{}: {err}", to.display()))?;
    }
    let stills: Vec<Value> = at.iter().zip(&names).map(|(f, n)| json!({"frame": f, "path": str_of(&output.join(n))})).collect();
    let manifest = render_manifest_in(output);
    let mut v = json!({"output": str_of(output), "stills": stills, "manifest": manifest.as_deref().map(str_of)});
    if let Some(m) = &manifest {
        let data = read_json(m, RENDER_JSON_MAX)?;
        let st = stated(&data);
        let (text, cut) = text_boxes(&data);
        v["width"] = json!(st.width);
        v["height"] = json!(st.height);
        v["fps"] = json!(st.fps);
        v["length"] = json!(st.length);
        v["text"] = json!(text);
        v["textTruncated"] = json!(cut);
    } else {
        v["text"] = json!([]);
    }
    Ok(v)
}

#[cfg(target_arch = "wasm32")]
fn stills_into(_: &Tool, _: &Path, _: bool, _: Option<&str>, _: &[u64], _: &Path) -> std::result::Result<Value, String> {
    Err("RAEA runs on the desktop app".into())
}

// ---------------------------------------------------------------------------------------------
// motion.render

/// Renders in flight and what finished ones made (job → outcome, `motion` in `jobs.list`).
#[derive(Default)]
pub struct MotionRenders {
    pending: Vec<Pending>,
    outcomes: BTreeMap<u64, Value>,
}

pub(crate) type Outcome = Arc<Mutex<Option<std::result::Result<Value, Value>>>>;

/// A job whose result lands in the project when it finishes.
pub(crate) struct Pending {
    job: u64,
    out: Outcome,
    place: Option<PlaceAt>,
    note: String,
    landing: Landing,
}

/// What a finished job brings in.
enum Landing {
    /// `motion.render`: the render, unless `import` is off (a transparent one goes above V1).
    Render { import: bool, alpha: bool },
    /// `composition.import`: the layers the job checked, laid out.
    Layers(Arc<Mutex<Option<crate::composition_import::Plan>>>),
}

impl Pending {
    /// A render's job: imported when it finishes (unless `import` is off) and placed on request,
    /// above V1 when it is transparent (`alpha`).
    pub(crate) fn render(job: u64, out: Outcome, place: Option<PlaceAt>, note: String, import: bool, alpha: bool) -> Pending {
        Pending { job, out, place, note, landing: Landing::Render { import, alpha } }
    }

    pub(crate) fn layers(job: u64, out: Outcome, place: Option<PlaceAt>, plan: Arc<Mutex<Option<crate::composition_import::Plan>>>, note: String) -> Pending {
        Pending { job, out, place, note, landing: Landing::Layers(plan) }
    }
}

impl MotionRenders {
    pub(crate) fn push(&mut self, p: Pending) {
        self.pending.push(p);
    }

    pub(crate) fn forget(&mut self, job: u64) {
        self.pending.retain(|q| q.job != job);
    }
}

/// What one render job does (everything checked before it starts).
#[derive(Clone)]
struct Work {
    tool: Tool,
    source: PathBuf,
    composition: Option<String>,
    output: PathBuf,
    created: bool,
    alpha: bool,
    /// The rate asked for (a sequence's import rate; an MP4's must match).
    fps: Option<f64>,
    /// The output size RAEA is asked for (`size`, else `canvas`, scaled for a preview).
    stage: Option<(u32, u32)>,
    canvas: Option<(u32, u32)>,
    quality: Option<String>,
    preview: bool,
    spec: Value,
    spec_path: Option<PathBuf>,
    services: Arc<dyn Services>,
    openers: Vec<filmcraft_media::Opener>,
}

impl Work {
    fn asset_name(&self) -> String {
        if self.alpha {
            return "frames".into();
        }
        let stem = self.output.file_name().and_then(OsStr::to_str).filter(|n| !n.is_empty() && !n.starts_with('.')).unwrap_or("motion");
        format!("{stem}.mp4")
    }
}

/// The render's stage: `size`, else `canvas`; a preview's long side is 960 (as `render.py`).
fn stage_of(cmd: &str, canvas: Option<(u32, u32)>, size: Option<(u32, u32)>, preview: bool) -> Result<Option<(u32, u32)>> {
    let stage = size.or(canvas);
    if !preview {
        return Ok(stage);
    }
    let (Some((w, h)), Some(_)) = (stage, canvas) else {
        return Err(bad(cmd, "`preview` needs `canvas` (and `size` for a master), e.g. 960x540 and 3840x2160"));
    };
    let scale = (960.0 / f64::from(w.max(h))).min(1.0);
    let side = |x: u32| ((f64::from(x) * scale).round() as u32).max(2);
    Ok(Some((side(w), side(h))))
}

fn render(s: &mut Session, p: &Value) -> Result<Value> {
    poll(s);
    let (source, html) = source_p(RENDER, p)?;
    let composition = composition_p(RENDER, p)?;
    let output = path_p(RENDER, p, "output")?.ok_or_else(|| bad(RENDER, "`output` is needed: a new revision folder"))?;
    let canvas = size_p(RENDER, p, "canvas")?;
    let size = size_p(RENDER, p, "size")?;
    let preview = strict_bool(RENDER, p, "preview", false)?;
    if preview && html {
        return Err(bad(RENDER, "an HTML page sizes itself: render it without `preview`"));
    }
    let stage = stage_of(RENDER, canvas, size, preview)?;
    let alpha = strict_bool(RENDER, p, "alpha", false)?;
    let fps = p.get("fps").filter(|v| !v.is_null()).map(|v| fps_value(RENDER, v, "fps")).transpose()?;
    let quality = match p.get("quality") {
        None | Some(Value::Null) => None,
        Some(Value::String(q)) if alpha => return Err(bad(RENDER, format!("`quality` {q} is for an MP4: a transparent PNG sequence is lossless"))),
        Some(Value::String(q)) if matches!(q.as_str(), "draft" | "high" | "master") => Some(q.clone()),
        Some(_) => return Err(bad(RENDER, "`quality` is draft, high or master")),
    };
    let (spec, spec_path) = qa_spec(RENDER, p)?;
    let import = strict_bool(RENDER, p, "import", true)?;
    let place = place_p(s, RENDER, p)?;
    if place.is_some() && !import {
        return Err(bad(RENDER, "`place` needs the render imported (leave `import` on)"));
    }
    let wait = strict_bool(RENDER, p, "wait", false)?;
    if cfg!(target_arch = "wasm32") {
        return Err(bad(RENDER, "RAEA runs on the desktop app, not in the browser"));
    }
    refuse_while_rendering(s, RENDER)?;
    let tool = tool_for(RENDER)?;
    let created = revision_dir(RENDER, &output)?;
    let openers = s.media.openers.read().unwrap_or_else(PoisonError::into_inner).clone();
    let work = Work {
        tool,
        source,
        composition,
        output,
        created,
        alpha,
        fps,
        stage,
        canvas,
        quality,
        preview,
        spec,
        spec_path,
        services: s.services.clone(),
        openers,
    };
    let file = work.source.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let what = match &work.composition {
        Some(id) => format!("{file} ({id})"),
        None => file,
    };
    let job = crate::narration::new_job(s, format!("Rendering motion graphics: {what}"));
    job.progress.total.store(1000, Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let out: Outcome = Default::default();
    let job_id = job.id;
    let planned = work.output.join(work.asset_name());
    s.motion_renders.pending.push(Pending::render(job_id, out.clone(), place, format!("Rendered by RAEA from {what}"), import, alpha));
    let (output, staging_created) = (work.output.clone(), work.created);
    let spawned = crate::narration::spawn(s, job, wait, "filmcraft-motion", move || {
        let t0 = web_time::Instant::now();
        let r = run_render(&work, &prog);
        let status = match &r {
            Ok(_) => Ok("Motion graphics rendered and checked".to_string()),
            Err(v) => Err(v.get("error").and_then(Value::as_str).unwrap_or("the render failed").to_string()),
        };
        *out.lock().unwrap_or_else(PoisonError::into_inner) = Some(r);
        crate::narration::finish(&prog, &res, status, t0);
    });
    if let Err(e) = spawned {
        s.motion_renders.pending.retain(|q| q.job != job_id);
        free_revision(&output, RENDER_STAGING, staging_created);
        return Err(e);
    }
    let mut v = json!({"job": job_id, "output": str_of(&output), "asset": str_of(&planned), "kind": if alpha { "pngSequence" } else { "mp4" }});
    if wait {
        poll(s);
        let o = s.motion_renders.outcomes.get(&job_id).cloned().unwrap_or(Value::Null);
        if let Some(e) = o.get("error").and_then(Value::as_str) {
            return Err(bad(RENDER, e.to_string()));
        }
        if let (Some(dst), Some(src)) = (v.as_object_mut(), o.as_object()) {
            for (k, x) in src {
                dst.insert(k.clone(), x.clone());
            }
        }
    }
    Ok(v)
}

/// A failure once the render is on disk: it stays, and the result says where.
fn kept(w: &Work, error: String, detail: Value) -> Value {
    kept_at(&w.output, error, detail)
}

/// [`kept`] for a render in the revision folder `output`.
pub(crate) fn kept_at(output: &Path, error: String, detail: Value) -> Value {
    let mut v = json!({
        "error": format!("{error}; the finished render is kept in {} (no {IMPORT_MANIFEST} was written)", output.display()),
        "kept": str_of(output), "output": str_of(output),
    });
    if let (Some(dst), Some(src)) = (v.as_object_mut(), detail.as_object()) {
        for (k, x) in src {
            dst.entry(k.clone()).or_insert_with(|| x.clone());
        }
    }
    v
}

#[cfg(not(target_arch = "wasm32"))]
fn run_render(w: &Work, prog: &filmcraft_export::Progress) -> std::result::Result<Value, Value> {
    let status = |t: &str| *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = t.to_string();
    let failed = |e: String| {
        free_revision(&w.output, RENDER_STAGING, w.created);
        json!({"error": e, "output": str_of(&w.output), "kept": Value::Null})
    };
    status("Waiting for the GPU");
    // the GPU is held for the whole job: the check, the render and its QA
    let _gpu = gpu("motion graphics (RAEA)", RENDER_GPU_WAIT).map_err(failed)?;
    let staging = w.output.join(RENDER_STAGING);
    std::fs::create_dir(&staging).map_err(|e| failed(format!("{}: {e}", staging.display())))?;
    let log = staging.join("raea.log");
    let cwd = w.source.parent().unwrap_or(Path::new("/")).to_path_buf();
    let never = AtomicBool::new(false);
    // the version, for the record (a tool that does not say is still run)
    let version_out = staging.join("version.log");
    let version = match raea(&w.tool, &["--version".to_string()], &cwd, &version_out, &never, VERSION_SECONDS, &mut |_| {}) {
        Ok(true) => std::fs::read_to_string(&version_out)
            .ok()
            .and_then(|t| t.lines().filter(|l| !l.starts_with("$ ")).map(str::trim).find(|l| !l.is_empty()).map(|l| l.chars().take(64).collect::<String>())),
        _ => None,
    };
    let _ = std::fs::remove_file(&version_out);
    let mut tool = w.tool.json();
    tool["version"] = json!(version);
    if prog.cancel.load(Ordering::Relaxed) {
        return Err(failed("stopped: nothing was rendered".into()));
    }
    // 1. RAEA's own check, reported (its render refuses what its floor refuses)
    status("RAEA: checking the composition");
    let check = match run_check(&w.tool, &w.source, w.composition.as_deref(), &staging.join("check.json"), &log, &prog.cancel, false) {
        Ok(v) => v,
        Err(e) if e == "stopped" => return Err(failed("stopped: nothing was rendered".into())),
        Err(e) => json!({"error": e}),
    };
    prog.done.store(50, Ordering::Relaxed);
    // 2. the render
    status("RAEA: rendering");
    let name = w.asset_name();
    let made = staging.join(&name);
    let mut args = vec!["render".to_string(), str_of(&w.source), "-o".into(), str_of(&made)];
    if let Some(id) = &w.composition {
        args.extend(["--composition".to_string(), id.clone()]);
    }
    if w.alpha {
        args.push("--alpha".into());
    } else {
        let q = w.quality.clone().unwrap_or_else(|| if w.preview { "draft" } else { "master" }.to_string());
        args.extend(["--quality".to_string(), q]);
    }
    if let Some((width, height)) = w.stage {
        // RAEA 0.1.0 needs the output stage as well as --w/--h (0.1.8 renders the same with it);
        // a fixed-size layout is refused by the size check rather than called 4K
        args.extend(["--w".to_string(), width.to_string(), "--h".into(), height.to_string(), "--stage".into(), format!("{width}x{height}")]);
    }
    let est = w.tool.estimate.unwrap_or(30.0).clamp(1.0, 86_400.0);
    let mut tick = |gone: f64| prog.done.store((50.0 + (gone / est).min(0.97) * 750.0) as u64, Ordering::Relaxed);
    let ok = match raea(&w.tool, &args, &cwd, &log, &prog.cancel, w.tool.timeout.unwrap_or(RENDER_SECONDS), &mut tick) {
        Ok(ok) => ok,
        Err(e) if e == "stopped" => return Err(failed("stopped: nothing was kept".into())),
        Err(e) => return Err(failed(e)),
    };
    let there = if w.alpha { made.is_dir() } else { made.is_file() && std::fs::metadata(&made).is_ok_and(|m| m.len() > 0) };
    if !ok || !there {
        let why = if ok { "RAEA finished but wrote no render" } else { "RAEA could not render the composition" };
        return Err(failed(format!("{why}: {}", tail(&log))));
    }
    // 3. the render is finished: it moves into the revision and stays from here on
    let asset = w.output.join(&name);
    let mut moves: Vec<(PathBuf, PathBuf)> = vec![(made.clone(), asset.clone())];
    if !w.alpha
        && let Some(m) = render_manifest_in(&staging)
        && let Some(n) = m.file_name()
    {
        moves.push((m.clone(), w.output.join(n)));
    }
    for extra in ["check.json", "raea.log"] {
        if staging.join(extra).is_file() {
            moves.push((staging.join(extra), w.output.join(extra)));
        }
    }
    for (from, to) in &moves {
        if let Err(e) = std::fs::rename(from, to) {
            // (what did not move stays in the staging folder, which is kept then)
            return Err(
                json!({"error": format!("{}: {e}; the render is in {}", to.display(), staging.display()), "kept": str_of(&staging), "output": str_of(&w.output)}),
            );
        }
    }
    let _ = std::fs::remove_dir_all(&staging);
    let mut check = check;
    if check.get("path").is_some_and(|p| !p.is_null()) {
        check["path"] = json!(str_of(&w.output.join("check.json")));
    }
    let manifest_path = if w.alpha { render_manifest_in(&asset) } else { render_manifest_in(&w.output) };
    let stated_by = manifest_path.as_deref().and_then(|m| read_json(m, RENDER_JSON_MAX).ok()).map(|v| stated(&v)).unwrap_or_default();
    let detail = json!({"check": check, "tool": tool, "renderManifest": manifest_path.as_deref().map(str_of)});
    if prog.cancel.load(Ordering::Relaxed) {
        return Err(kept(w, "stopped after the render".into(), detail));
    }
    // 4. AutoCut's own probe: the size and rate asked for, every frame of a sequence there
    status("Probing the render");
    prog.done.store(820, Ordering::Relaxed);
    let probed = probe(w, &asset, stated_by).map_err(|e| kept(w, e, detail.clone()))?;
    let mut detail = detail;
    detail["probe"] = probed.json.clone();
    // 5. the animation preflight (with qa.motionEnergy as warnings)
    status("Checking the animation");
    prog.done.store(850, Ordering::Relaxed);
    let qa = preflight(w, &probed).map_err(|e| kept(w, e, detail.clone()))?;
    let preflight_path = w.output.join("preflight.json");
    let pretty = serde_json::to_string_pretty(&qa).unwrap_or_default();
    std::fs::write(&preflight_path, format!("{pretty}\n")).map_err(|e| kept(w, format!("{}: {e}", preflight_path.display()), detail.clone()))?;
    let summary = json!({
        "passed": qa.get("passed"), "findings": qa.get("findings"), "warnings": qa.get("warnings"),
        "skipped": qa.get("skipped"), "path": str_of(&preflight_path),
    });
    detail["preflight"] = summary.clone();
    if qa.get("passed") != Some(&Value::Bool(true)) {
        let codes: Vec<&str> =
            qa.get("findings").and_then(Value::as_array).map(|a| a.iter().filter_map(|f| f.get("code").and_then(Value::as_str)).collect()).unwrap_or_default();
        return Err(kept(w, format!("the animation preflight failed ({}): inspect preflight.json and contact-sheet.png", codes.join(", ")), detail));
    }
    // 6. the handoff: autocut-import.json, only now
    let sheet = w.output.join("contact-sheet.png");
    let mut dependencies = vec![str_of(&w.source)];
    dependencies.extend(w.spec_path.as_deref().map(str_of));
    dependencies.push(str_of(&w.tool.program));
    let (command, params) = match probed.first.as_deref() {
        Some(first) if w.alpha => ("file.importImageSequence", json!({"path": str_of(first), "fps": probed.fps})),
        _ => ("file.import", json!({"paths": [str_of(&asset)]})),
    };
    let manifest = json!({
        "command": command, "params": params,
        "preflight": str_of(&preflight_path), "sheet": str_of(&sheet),
        "cacheKey": Value::Null, "cached": false, "reviewOnly": w.preview,
        "composition": str_of(&w.source), "dependencies": dependencies,
        "compositionId": w.composition, "check": detail["check"].get("path").cloned().unwrap_or(Value::Null),
        "renderManifest": detail["renderManifest"].clone(), "tool": tool,
    });
    let manifest_file = w.output.join(IMPORT_MANIFEST);
    let partial = w.output.join(format!("{IMPORT_MANIFEST}.partial"));
    let text = serde_json::to_string_pretty(&manifest).unwrap_or_default();
    std::fs::write(&partial, format!("{text}\n"))
        .and_then(|()| std::fs::rename(&partial, &manifest_file))
        .map_err(|e| kept(w, format!("{}: {e}", manifest_file.display()), detail.clone()))?;
    prog.done.store(1000, Ordering::Relaxed);
    let mut v = detail;
    v["output"] = json!(str_of(&w.output));
    v["asset"] = json!(str_of(&asset));
    v["kind"] = json!(if w.alpha { "pngSequence" } else { "mp4" });
    v["manifest"] = json!(str_of(&manifest_file));
    v["import"] = manifest;
    v["sheet"] = json!(str_of(&sheet));
    v["reviewOnly"] = json!(w.preview);
    v["canvas"] = json!(w.canvas.map(|(a, b)| format!("{a}x{b}")));
    Ok(v)
}

#[cfg(target_arch = "wasm32")]
fn run_render(_: &Work, _: &filmcraft_export::Progress) -> std::result::Result<Value, Value> {
    Err(json!({"error": "RAEA runs on the desktop app"}))
}

/// What the native probe found.
pub(crate) struct Probed {
    pub(crate) json: Value,
    /// A sequence's first frame.
    pub(crate) first: Option<PathBuf>,
    /// The render's rate.
    pub(crate) fps: f64,
}

/// What a render is probed against.
pub(crate) struct ProbeSpec<'a> {
    /// The tool that rendered it, for the messages.
    pub(crate) tool: &'a str,
    /// How the messages name what its own record of the render states.
    pub(crate) stated_as: &'a str,
    /// Said after a size that is not the one asked for.
    pub(crate) size_hint: &'a str,
    /// The asset is a folder of numbered PNG frames (`frame-00000.png`…), else one movie.
    pub(crate) sequence: bool,
    /// It must have an alpha channel.
    pub(crate) alpha: bool,
    /// The rate asked for (a sequence's import rate; a movie's must match).
    pub(crate) fps: Option<f64>,
    /// The size asked for.
    pub(crate) stage: Option<(u32, u32)>,
    pub(crate) services: &'a dyn Services,
    pub(crate) openers: &'a [filmcraft_media::Opener],
}

const RAEA_SIZE_HINT: &str = ": a composition of a fixed size ignores --w/--h; lay it out from useVideoConfig() (docs/raea-workflow.md)";

/// The lowest-numbered `frame-N.png` of a frames folder.
fn first_frame(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .take(1_000_000)
        .filter_map(|e| {
            let name = e.file_name().to_str()?.to_string();
            let digits = name.strip_prefix("frame-")?.strip_suffix(".png")?;
            (!digits.is_empty() && digits.len() <= 9 && digits.bytes().all(|b| b.is_ascii_digit())).then(|| (digits.parse::<u64>().ok(), e.path()))
        })
        .filter_map(|(n, p)| Some((n?, p)))
        .min_by_key(|(n, _)| *n)
        .map(|(_, p)| p)
}

fn probe(w: &Work, asset: &Path, stated_by: Stated) -> std::result::Result<Probed, String> {
    let spec = ProbeSpec {
        tool: "RAEA",
        stated_as: "RAEA's render manifest says",
        size_hint: RAEA_SIZE_HINT,
        sequence: w.alpha,
        alpha: w.alpha,
        fps: w.fps,
        stage: w.stage,
        services: w.services.as_ref(),
        openers: &w.openers,
    };
    probe_asset(&spec, asset, stated_by)
}

/// AutoCut's own probe of a render: it opens, at the size and rate asked for (and as its record
/// states them), a sequence with every frame, with alpha when asked.
pub(crate) fn probe_asset(w: &ProbeSpec, asset: &Path, stated_by: Stated) -> std::result::Result<Probed, String> {
    let tool = w.tool;
    let pool = crate::media_pool::MediaPool::default();
    *pool.openers.write().unwrap_or_else(PoisonError::into_inner) = w.openers.to_vec();
    let services = w.services;
    let rate_wanted = w.fps.or(stated_by.fps);
    let (info, first, missing, frames) = if w.sequence {
        let first = first_frame(asset).ok_or_else(|| format!("{tool} wrote no numbered frames (frame-00000.png) into {}", asset.display()))?;
        let first_s = str_of(&first);
        let list = crate::media_pool::image_sequence_frames(&first_s, services).map_err(|e| format!("the sequence cannot be read: {e}"))?;
        let missing: Vec<usize> = list.iter().enumerate().filter(|(_, f)| f.is_none()).map(|(i, _)| i).take(1000).collect();
        if !missing.is_empty() {
            let shown: Vec<String> = missing.iter().take(20).map(|i| format!("frame-{i:05}.png")).collect();
            return Err(format!("the PNG sequence has gaps: {} missing ({})", missing.len(), shown.join(", ")));
        }
        if let Some(length) = stated_by.length
            && length != list.len() as u64
        {
            return Err(format!("the PNG sequence has {} frames; {} {length}", list.len(), w.stated_as));
        }
        let fps = rate_wanted.ok_or_else(|| "give `fps`: the render manifest names no frame rate for the sequence".to_string())?;
        let src = crate::media_pool::open_image_sequence(&first_s, services, FrameRate::from_f64(fps), "Motion graphics")
            .map_err(|e| format!("the sequence cannot be read: {e}"))?;
        (src.info().clone(), Some(first), missing, Some(list.len()))
    } else {
        let src = pool.open_file(&str_of(asset), services).map_err(|e| format!("AutoCut cannot read the render: {e}"))?;
        (src.info().clone(), None, Vec::new(), None)
    };
    let video = info.video.as_ref().ok_or_else(|| "AutoCut found no picture in the render".to_string())?;
    let rate = video.frame_rate.sane();
    let fps = rate.as_f64();
    let (width, height) = (video.width, video.height);
    let wanted = w.stage.or(match (stated_by.width, stated_by.height) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    });
    if let Some((ww, wh)) = wanted
        && (ww, wh) != (width, height)
    {
        return Err(format!("{tool} rendered {width}x{height}, not the {ww}x{wh} asked for{}", w.size_hint));
    }
    if let Some(want) = rate_wanted
        && (want - fps).abs() > 0.001
    {
        return Err(format!("the render runs at {fps:.3} fps, not the {want} asked for"));
    }
    if let (Some(a), Some(b)) = (w.fps, stated_by.fps)
        && (a - b).abs() > 0.001
    {
        return Err(format!("`fps` {a} differs from the {b} fps {tool} rendered at"));
    }
    if w.alpha && !video.has_alpha {
        return Err(format!("the {} has no alpha channel", if w.sequence { "PNG sequence" } else { "render" }));
    }
    let json = json!({
        "width": width, "height": height, "fps": fps, "duration": info.duration.0, "seconds": info.duration.seconds(),
        "hasAlpha": video.has_alpha, "codec": video.codec, "frames": frames, "missing": missing,
        "first": first.as_deref().map(str_of),
    });
    Ok(Probed { json, first, fps })
}

fn preflight(w: &Work, probed: &Probed) -> std::result::Result<Value, String> {
    let mut expected = w.spec.get("expected").cloned().filter(Value::is_object).unwrap_or_else(|| json!({}));
    if let Some((a, b)) = w.stage {
        expected["width"] = json!(a);
        expected["height"] = json!(b);
    }
    expected["alpha"] = json!(w.alpha);
    if w.alpha || w.fps.is_some() {
        expected["fps"] = json!(probed.fps);
    }
    let asset = match &probed.first {
        Some(first) => str_of(first),
        None => str_of(&w.output.join(w.asset_name())),
    };
    let mut q = json!({"path": asset, "expected": expected, "sheet": str_of(&w.output.join("contact-sheet.png"))});
    for key in ["holds", "textBounds", "soundCues"] {
        if let Some(v) = w.spec.get(key).filter(|v| !v.is_null()) {
            q[key] = v.clone();
        }
    }
    if w.alpha {
        q["fps"] = json!(probed.fps);
    }
    let mut isolated = Session::new(w.services.clone());
    *isolated.media.openers.write().unwrap_or_else(PoisonError::into_inner) = w.openers.clone();
    isolated.execute("media.preflightAnimation", q).map_err(|e| format!("the animation preflight could not run: {e}"))
}

/// What render job `job` made, when it is one (`motion` in `jobs.list`).
pub fn outcome_json(s: &Session, job: u64) -> Option<Value> {
    s.motion_renders.outcomes.get(&job).cloned()
}

/// Import and place the results of finished renders. Called each UI frame (through
/// [`Session::poll_persistence`]), by `jobs.list` and by the `motion.*` commands.
pub fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.motion_renders.pending.len() {
        let done = s.motion_renders.pending.get(i).is_some_and(|q| q.out.lock().unwrap_or_else(PoisonError::into_inner).is_some());
        if !done {
            i += 1;
            continue;
        }
        let q = s.motion_renders.pending.remove(i);
        let r = q.out.lock().unwrap_or_else(PoisonError::into_inner).take();
        let outcome = match r {
            Some(Ok(v)) if !matches!(q.landing, Landing::Render { import: false, .. }) => match land(s, &q, &v) {
                Ok(landed) => landed,
                Err(e) => {
                    s.error_toast(RENDER, format!("Motion graphics: {e}"));
                    let mut v = v;
                    v["error"] = json!(format!(
                        "it was made and passed its checks but could not be brought into the project: {e}; it is in {}",
                        v["output"].as_str().unwrap_or_default()
                    ));
                    v["kept"] = v["output"].clone();
                    v
                }
            },
            Some(Ok(v)) => v,
            Some(Err(v)) => v,
            None => continue,
        };
        s.motion_renders.outcomes.insert(q.job, outcome);
        if s.motion_renders.outcomes.len() > MAX_OUTCOMES
            && let Some(old) = s.motion_renders.outcomes.keys().next().copied()
        {
            s.motion_renders.outcomes.remove(&old);
        }
    }
}

/// Bring a finished job's result in (one undo step): a render as its manifest says, placed on
/// request; a composition's layers laid out.
fn land(s: &mut Session, q: &Pending, v: &Value) -> Result<Value> {
    let n0 = s.history.undo.len();
    let (result, label) = match &q.landing {
        Landing::Render { alpha, .. } => (land_render(s, q, *alpha, v), "Motion Graphics"),
        Landing::Layers(plan) => {
            let plan = plan.lock().unwrap_or_else(PoisonError::into_inner).take();
            let r = match plan {
                Some(plan) => crate::composition_import::land(s, &plan, q.place.as_ref(), &q.note, v),
                None => Err(EngineError::Other("the job left no layers to lay out".into())),
            };
            (r, "Composition")
        }
    };
    crate::clip_ops::collapse_history(s, n0, label);
    if result.is_err() && s.history.undo.len() > n0 {
        let _ = s.undo();
        s.history.redo.clear();
    }
    result
}

fn land_render(s: &mut Session, q: &Pending, alpha: bool, v: &Value) -> Result<Value> {
    let mut out = v.clone();
    let command = v["import"]["command"].as_str().unwrap_or_default();
    if command != "file.import" && command != "file.importImageSequence" {
        return Err(EngineError::Other("the import manifest names no import".into()));
    }
    let imported = s.execute(command, v["import"]["params"].clone())?;
    let item = imported
        .get("items")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_u64)
        .ok_or_else(|| EngineError::Other("the render could not be imported".into()))?;
    let note = q.note.clone();
    s.edit("Motion Graphics", |pr, _| {
        if let Some(it) = pr.item_mut(ItemId(item)) {
            it.metadata.insert("Log Note".into(), note);
        }
        Ok(())
    })?;
    out["item"] = json!(item);
    out["clip"] = Value::Null;
    let Some(pl) = q.place.as_ref() else { return Ok(out) };
    if s.state.active_sequence.is_none() {
        out["placed"] = json!("no sequence is open");
        return Ok(out);
    }
    let rate = s.sequence_rate().sane();
    let dur = s.project.item(ItemId(item)).map(|pi| crate::commands::full_duration(s, pi)).unwrap_or(Tick::from_seconds_f64(1.0)).max(rate.frame_duration());
    let range = TimeRange::new(pl.at, dur);
    // a transparent render goes over the picture (from V2 up), a full scene on the picture track
    let track = crate::generate::video_track(s, pl.track.as_deref(), usize::from(alpha), range);
    let (_, has_audio) = s.project.source_streams(ItemId(item));
    let mut params = json!({"item": item, "track": track, "time": pl.at.0});
    if has_audio {
        params["audioTrack"] = json!(crate::generate::audio_track(s, None, range)?);
    } else {
        params["audio"] = json!(false);
    }
    let placed = s.execute("timeline.place", params)?;
    let clip = placed
        .get("clips")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_u64)
        .ok_or_else(|| EngineError::Other("the render could not be placed".into()))?;
    out["clip"] = json!(clip);
    out["clips"] = placed.get("clips").cloned().unwrap_or(Value::Null);
    // a review-size render in a larger sequence fills its frame
    let render_size = (v["probe"]["width"].as_u64(), v["probe"]["height"].as_u64());
    let seq_size = s.active_sequence().map(|q| (u64::from(q.settings.width), u64::from(q.settings.height)));
    if let (Some(w), Some(h), Some(seq)) = (render_size.0, render_size.1, seq_size)
        && (w, h) != seq
    {
        s.execute("clip.fillFrame", json!({"clips": [clip]}))?;
        out["filledFrame"] = json!(true);
    }
    if let Some(q) = s.active_sequence()
        && let Some((track, it)) = q.find_item(filmcraft_project::ClipId(clip))
    {
        out["start"] = json!(it.start.0);
        out["duration"] = json!(it.duration.0);
        out["track"] = json!(q.video_tracks.iter().position(|t| t.id == track).map(|i| format!("V{}", i + 1)));
    }
    Ok(out)
}

#[cfg(test)]
#[path = "motion_graphics_tests.rs"]
mod tests;
