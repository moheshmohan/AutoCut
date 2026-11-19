//! Local generation tools (image, video, music and sound-effect models the user installed
//! themselves): the list of them, how a request becomes a command line, and running one.
//!
//! Nothing here is a model. A **provider** is a program on this computer, described in
//! `<data dir>/generation/providers.json`, a file the user keeps (see `docs/generation.md`).
//! No command sets it: what may be run is decided by whoever can write that file, never by a
//! parameter that arrived over the control channel. A request fills placeholders in the
//! provider's argument list; the arguments go to the program as they are (no shell), so a
//! prompt cannot become a command.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Deserialize;

/// The most a provider list may weigh.
pub const MANIFEST_MAX: u64 = 256 * 1024;
const MAX_PROVIDERS: usize = 32;
const MAX_ARGS: usize = 96;
const MAX_ARG_LEN: usize = 8192;
/// The most references a request may carry (the most any provider here takes).
pub const MAX_REFERENCES: usize = 16;

/// What a provider makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Image,
    Video,
    Music,
    Sfx,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Image => "image",
            Kind::Video => "video",
            Kind::Music => "music",
            Kind::Sfx => "sfx",
        }
    }

    /// The file extension a provider of this kind writes unless it says otherwise.
    pub fn extension(self) -> &'static str {
        match self {
            Kind::Image => "png",
            Kind::Video => "mp4",
            Kind::Music | Kind::Sfx => "wav",
        }
    }

    pub fn is_audio(self) -> bool {
        matches!(self, Kind::Music | Kind::Sfx)
    }
}

/// One generation program.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Provider {
    pub id: String,
    pub kind: Kind,
    pub name: String,
    /// The program (an absolute path) and its arguments, with placeholders: `{out}`, `{prompt}`,
    /// `{negative}`, `{width}`, `{height}`, `{seed}`, `{seconds}`, `{frames}`, `{fps}`, `{image}`,
    /// `{steps}`, `{strength}`; an argument that is exactly `{references}` becomes
    /// [`Provider::per_reference`] once for each reference, and one that is exactly `{optional}`
    /// becomes each group of [`Provider::optional`] whose value the request gave.
    pub command: Vec<String>,
    /// The arguments for one reference picture (`{path}` is its file).
    #[serde(default)]
    pub per_reference: Vec<String>,
    /// Arguments added only when the request gives the named value (`negative`, `steps`,
    /// `strength`, `transparent`, `image`, `seconds`…).
    #[serde(default)]
    pub optional: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub cwd: Option<String>,
    /// The output file's extension (default: by kind).
    #[serde(default)]
    pub extension: Option<String>,
    /// About how long one run takes (for the progress bar).
    #[serde(default)]
    pub estimate_seconds: Option<f64>,
    /// When a run is given up on (default an hour).
    #[serde(default)]
    pub timeout_seconds: Option<f64>,
    /// Width and height are multiples of this (default 16).
    #[serde(default)]
    pub multiple_of: Option<u32>,
    /// The long side when the request names no size (default 1024 for pictures, 480 for video).
    #[serde(default)]
    pub default_side: Option<u32>,
    #[serde(default)]
    pub max_side: Option<u32>,
    #[serde(default)]
    pub max_seconds: Option<f64>,
    /// The frame rate the program makes video at (default 24).
    #[serde(default)]
    pub fps: Option<f64>,
    /// Frame counts the program takes: `framesOffset` plus a multiple of `framesStep`.
    #[serde(default)]
    pub frames_step: Option<u32>,
    #[serde(default)]
    pub frames_offset: Option<u32>,
    #[serde(default)]
    pub max_references: Option<u32>,
    /// Put before the prompt when references are given and the prompt does not mention them
    /// itself (`referenceMarker` is how a prompt mentions one).
    #[serde(default)]
    pub reference_prompt_prefix: Option<String>,
    #[serde(default)]
    pub reference_marker: Option<String>,
    /// A picture tool that cannot make a transparent picture itself: the program that cuts the
    /// subject out of one (`{image}` in, `{out}` out), run after it when `transparent` is asked.
    #[serde(default)]
    pub cutout_command: Vec<String>,
    /// Added to the prompt of a picture that will be cut out (a plain background cuts cleanly).
    #[serde(default)]
    pub cutout_prompt_suffix: Option<String>,
    #[serde(default)]
    pub license: String,
    /// What the model's licence asks to be shown where it is used.
    #[serde(default)]
    pub attribution: String,
    #[serde(default)]
    pub source: String,
}

#[derive(Deserialize)]
struct Manifest {
    providers: Vec<Provider>,
}

/// What a request gives a provider's command line.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Vars {
    pub out: String,
    pub prompt: String,
    pub negative: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub seed: u64,
    pub seconds: Option<f64>,
    pub frames: Option<u32>,
    pub fps: Option<f64>,
    pub image: Option<String>,
    pub references: Vec<String>,
    pub steps: Option<u32>,
    pub strength: Option<f64>,
    pub transparent: bool,
}

const NAMES: [&str; 14] =
    ["out", "prompt", "negative", "width", "height", "seed", "seconds", "frames", "fps", "image", "steps", "strength", "transparent", "path"];

fn number(v: f64) -> String {
    let s = format!("{v:.3}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() { "0".into() } else { s.into() }
}

impl Vars {
    /// The value of a placeholder: `None` when the request did not give it.
    fn value(&self, name: &str) -> Option<String> {
        match name {
            "out" => Some(self.out.clone()),
            "prompt" => Some(self.prompt.clone()),
            "negative" => self.negative.clone().filter(|s| !s.is_empty()),
            "width" => self.width.map(|v| v.to_string()),
            "height" => self.height.map(|v| v.to_string()),
            "seed" => Some(self.seed.to_string()),
            "seconds" => self.seconds.map(number),
            "frames" => self.frames.map(|v| v.to_string()),
            "fps" => self.fps.map(number),
            "image" => self.image.clone().filter(|s| !s.is_empty()),
            "steps" => self.steps.map(|v| v.to_string()),
            "strength" => self.strength.map(number),
            "transparent" => self.transparent.then(|| "true".to_string()),
            _ => None,
        }
    }
}

/// The placeholders in one argument; an unknown name is an error.
fn placeholders(arg: &str) -> Result<Vec<&str>, String> {
    let mut out = Vec::new();
    let mut rest = arg;
    while let Some(i) = rest.find('{') {
        let after = rest.get(i + 1..).unwrap_or("");
        let Some(j) = after.find('}') else { break };
        let name = after.get(..j).unwrap_or("");
        if name.chars().all(|c| c.is_ascii_alphabetic()) && !name.is_empty() {
            if !NAMES.contains(&name) && name != "references" && name != "optional" {
                return Err(format!("unknown placeholder {{{name}}}"));
            }
            out.push(name);
        }
        rest = after.get(j + 1..).unwrap_or("");
    }
    Ok(out)
}

/// `arg` with its placeholders filled (`path` is the reference of a per-reference group; a value
/// the request did not give is empty).
fn fill(arg: &str, v: &Vars, path: Option<&str>) -> String {
    let mut out = String::with_capacity(arg.len());
    let mut rest = arg;
    while let Some(i) = rest.find('{') {
        out.push_str(rest.get(..i).unwrap_or(""));
        let after = rest.get(i + 1..).unwrap_or("");
        let Some(j) = after.find('}') else {
            out.push('{');
            rest = after;
            continue;
        };
        let name = after.get(..j).unwrap_or("");
        if NAMES.contains(&name) {
            let value = if name == "path" { path.map(str::to_string) } else { v.value(name) };
            out.push_str(&value.unwrap_or_default());
        } else {
            out.push('{');
            out.push_str(name);
            out.push('}');
        }
        rest = after.get(j + 1..).unwrap_or("");
    }
    out.push_str(rest);
    out
}

/// The command line of a request: the program first, then its arguments.
pub fn argv(p: &Provider, v: &Vars) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for arg in &p.command {
        match arg.as_str() {
            "{references}" => {
                for r in &v.references {
                    out.extend(p.per_reference.iter().map(|a| fill(a, v, Some(r))));
                }
            }
            "{optional}" => {
                for (key, group) in &p.optional {
                    if v.value(key).is_some() {
                        out.extend(group.iter().map(|a| fill(a, v, None)));
                    }
                }
            }
            _ => out.push(fill(arg, v, None)),
        }
    }
    if out.len() > 4 * MAX_ARGS || out.iter().any(|a| a.len() > 4 * MAX_ARG_LEN || a.contains('\0')) {
        return Err(format!("the request makes too long a command line for {}", p.id));
    }
    Ok(out)
}

/// The prompt a provider is given: its reference prefix first when references are given and the
/// prompt does not mention one itself.
pub fn prompt_for(p: &Provider, prompt: &str, references: usize) -> String {
    match (&p.reference_prompt_prefix, references) {
        (Some(prefix), n) if n > 0 && !p.reference_marker.as_deref().is_some_and(|m| !m.is_empty() && prompt.contains(m)) => format!("{prefix}{prompt}"),
        _ => prompt.to_string(),
    }
}

/// The size a provider is asked for: the wanted one, else its default long side in the shape
/// `aspect`; never past its largest side, each side a multiple of what it takes.
pub fn size(p: &Provider, want: (Option<u32>, Option<u32>), aspect: (u32, u32)) -> (u32, u32) {
    let step = p.multiple_of.unwrap_or(16).clamp(1, 256);
    let (aw, ah) = (f64::from(aspect.0.max(1)), f64::from(aspect.1.max(1)));
    let long = f64::from(p.default_side.unwrap_or(if p.kind == Kind::Video { 480 } else { 1024 }).clamp(64, 16_384));
    let (mut w, mut h) = match want {
        (Some(w), Some(h)) => (f64::from(w), f64::from(h)),
        (Some(w), None) => (f64::from(w), f64::from(w) * ah / aw),
        (None, Some(h)) => (f64::from(h) * aw / ah, f64::from(h)),
        (None, None) if aw >= ah => (long, long * ah / aw),
        (None, None) => (long * aw / ah, long),
    };
    let most = f64::from(p.max_side.unwrap_or(4096).clamp(64, 16_384));
    let big = w.max(h);
    if big > most {
        w *= most / big;
        h *= most / big;
    }
    let snap = |x: f64| {
        let s = f64::from(step);
        // (bounded above by `most`, so the cast cannot wrap)
        let n = ((x / s).round().clamp(1.0, 16_384.0)) as u32;
        n.saturating_mul(step).max(step.max(64))
    };
    (snap(w), snap(h))
}

/// The frame count and rate for `seconds` of video: a count the provider takes, at least the
/// wanted length where its longest allows.
pub fn frames(p: &Provider, seconds: f64) -> (u32, f64) {
    let fps = p.fps.filter(|f| f.is_finite() && *f >= 1.0 && *f <= 240.0).unwrap_or(24.0);
    let most = p.max_seconds.filter(|m| m.is_finite() && *m > 0.0).unwrap_or(10.0);
    let secs = if seconds.is_finite() { seconds.clamp(0.1, most) } else { most.min(4.0) };
    let want = (secs * fps).round().clamp(1.0, 100_000.0) as u32;
    let step = p.frames_step.unwrap_or(1).clamp(1, 1024);
    let off = p.frames_offset.unwrap_or(0) % step;
    let k = want.saturating_sub(off).div_ceil(step).max(1);
    (k.saturating_mul(step).saturating_add(off), fps)
}

/// The length a provider can make for a wanted one (seconds).
pub fn seconds(p: &Provider, want: f64, default: f64) -> f64 {
    let most = p.max_seconds.filter(|m| m.is_finite() && *m > 0.0).unwrap_or(if p.kind == Kind::Video { 10.0 } else { 180.0 });
    let w = if want.is_finite() && want > 0.0 { want } else { default };
    w.clamp(0.1, most)
}

fn check(p: &Provider) -> Result<(), String> {
    let id_ok = !p.id.is_empty() && p.id.len() <= 64 && p.id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if !id_ok {
        return Err(format!("provider id `{}` must be 1 to 64 letters, digits, - or _", p.id.chars().take(80).collect::<String>()));
    }
    let err = |m: String| Err(format!("provider {}: {m}", p.id));
    if p.command.is_empty() || p.command.len() > MAX_ARGS || p.per_reference.len() > MAX_ARGS || p.optional.len() > MAX_ARGS {
        return err(format!("`command` needs 1 to {MAX_ARGS} arguments"));
    }
    let groups = p.optional.values().flat_map(|g| g.iter());
    for arg in p
        .command
        .iter()
        .chain(&p.per_reference)
        .chain(&p.cutout_command)
        .chain(groups)
        .chain(p.env.values())
        .chain(p.cwd.iter())
        .chain(p.cutout_prompt_suffix.iter())
    {
        if arg.len() > MAX_ARG_LEN || arg.contains('\0') {
            return err("an argument is too long or holds a NUL".into());
        }
    }
    for arg in p.command.iter().chain(&p.per_reference).chain(&p.cutout_command).chain(p.optional.values().flat_map(|g| g.iter())) {
        if let Err(e) = placeholders(arg) {
            return err(e);
        }
    }
    for key in p.optional.keys() {
        if !NAMES.contains(&key.as_str()) || key == "path" {
            return err(format!("`optional` names an unknown value `{key}`"));
        }
    }
    let program = p.command.first().map(String::as_str).unwrap_or("");
    if !std::path::Path::new(program).is_absolute() || program.contains('{') {
        return err("the program (the first of `command`) must be an absolute path".into());
    }
    if !p.command.iter().any(|a| a.contains("{out}")) {
        return err("`command` must say where the result goes with {out}".into());
    }
    if let Some(cut) = p.cutout_command.first() {
        let io = |name: &str| p.cutout_command.iter().any(|a| a.contains(name));
        if p.cutout_command.len() > MAX_ARGS || !std::path::Path::new(cut).is_absolute() || cut.contains('{') || !io("{image}") || !io("{out}") {
            return err("`cutoutCommand` needs an absolute program, {image} and {out}".into());
        }
    }
    if p.env.len() > 64 || p.env.keys().any(|k| k.is_empty() || k.len() > 128 || k.contains('=') || k.contains('\0')) {
        return err("`env` has too many or malformed names".into());
    }
    if p.extension.as_deref().is_some_and(|e| e.is_empty() || e.len() > 8 || !e.chars().all(|c| c.is_ascii_alphanumeric())) {
        return err("`extension` must be letters and digits".into());
    }
    for (name, v) in [("estimateSeconds", p.estimate_seconds), ("timeoutSeconds", p.timeout_seconds), ("maxSeconds", p.max_seconds), ("fps", p.fps)] {
        if v.is_some_and(|x| !x.is_finite() || x <= 0.0 || x > 1.0e6) {
            return err(format!("`{name}` must be a positive number"));
        }
    }
    Ok(())
}

/// The providers a list names, each checked; duplicates and anything malformed are errors.
pub fn parse(text: &str) -> Result<Vec<Provider>, String> {
    if text.len() as u64 > MANIFEST_MAX {
        return Err("the provider list is larger than 256 KiB".into());
    }
    let m: Manifest = serde_json::from_str(text).map_err(|e| format!("the provider list is not valid: {e}"))?;
    if m.providers.len() > MAX_PROVIDERS {
        return Err(format!("the provider list names more than {MAX_PROVIDERS} providers"));
    }
    let mut seen = std::collections::BTreeSet::new();
    for p in &m.providers {
        check(p)?;
        if !seen.insert(p.id.as_str()) {
            return Err(format!("provider id `{}` is used twice", p.id));
        }
    }
    Ok(m.providers)
}

/// Where the provider list is kept: `<data dir>/generation/providers.json`.
pub fn manifest_path() -> Option<PathBuf> {
    crate::autosave::default_data_dir().map(|d| d.join("generation").join("providers.json"))
}

/// The providers of this computer; none when there is no list yet.
#[cfg(not(target_arch = "wasm32"))]
pub fn load() -> Result<Vec<Provider>, String> {
    use std::io::Read as _;
    let Some(path) = manifest_path() else { return Ok(Vec::new()) };
    let f = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let mut text = String::new();
    f.take(MANIFEST_MAX + 1).read_to_string(&mut text).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&text).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(target_arch = "wasm32")]
pub fn load() -> Result<Vec<Provider>, String> {
    Ok(Vec::new())
}

/// The provider's cut-out step as a provider of its own (the same environment and folder), when
/// it has one.
pub fn cutout(p: &Provider) -> Option<Provider> {
    if p.cutout_command.is_empty() {
        return None;
    }
    let mut c = p.clone();
    c.command = p.cutout_command.clone();
    c.name = format!("{} (cut-out)", p.name);
    c.per_reference.clear();
    c.optional.clear();
    c.estimate_seconds = Some(20.0);
    Some(c)
}

/// Whether the provider's program is there to run.
pub fn installed(p: &Provider) -> bool {
    p.command.first().is_some_and(|c| std::path::Path::new(c).is_file())
}

/// The end of a run's log, for an error message.
#[cfg(not(target_arch = "wasm32"))]
fn log_tail(path: &std::path::Path) -> String {
    use std::io::{Read as _, Seek as _};
    let Ok(mut f) = std::fs::File::open(path) else { return String::new() };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let _ = f.seek(std::io::SeekFrom::Start(len.saturating_sub(1500)));
    let mut bytes = Vec::new();
    let _ = f.take(1500).read_to_end(&mut bytes);
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let from = lines.len().saturating_sub(6);
    lines.get(from..).unwrap_or(&[]).join(" | ")
}

/// Runs a provider's command line until it has written `out`: `tick` hears the seconds gone by,
/// `cancel` stops it (the program is killed), and so does its time limit.
#[cfg(not(target_arch = "wasm32"))]
pub fn run(p: &Provider, args: &[String], out: &std::path::Path, cancel: &std::sync::atomic::AtomicBool, tick: &mut dyn FnMut(f64)) -> Result<(), String> {
    use std::process::{Command, Stdio};
    use std::sync::atomic::Ordering;
    let Some((program, rest)) = args.split_first() else { return Err(format!("{} has no program", p.id)) };
    if !std::path::Path::new(program).is_file() {
        return Err(format!("{}: its program {program} is not on this computer (see docs/generation.md)", p.name));
    }
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let log_path = out.with_extension("generation.log");
    let log = std::fs::File::create(&log_path).map_err(|e| format!("{}: {e}", log_path.display()))?;
    let log2 = log.try_clone().map_err(|e| format!("{}: {e}", log_path.display()))?;
    let mut cmd = Command::new(program);
    cmd.args(rest).envs(&p.env).stdin(Stdio::null()).stdout(Stdio::from(log)).stderr(Stdio::from(log2));
    if let Some(dir) = p.cwd.as_deref().filter(|d| !d.is_empty()) {
        cmd.current_dir(dir);
    }
    let mut child = cmd.spawn().map_err(|e| format!("{}: could not start {program}: {e}", p.name))?;
    let limit = p.timeout_seconds.unwrap_or(3600.0).clamp(1.0, 86_400.0);
    let t0 = web_time::Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("{}: {e}", p.name));
            }
        }
        let gone = t0.elapsed().as_secs_f64();
        let stopped = cancel.load(Ordering::Relaxed);
        if stopped || gone > limit {
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_file(out);
            let _ = std::fs::remove_file(&log_path);
            return Err(if stopped { "stopped: nothing was changed".into() } else { format!("{} did not finish in {limit:.0} s and was stopped", p.name) });
        }
        tick(gone);
        std::thread::sleep(std::time::Duration::from_millis(150));
    };
    let made = std::fs::metadata(out).map(|m| m.is_file() && m.len() > 0).unwrap_or(false);
    if status.success() && made {
        let _ = std::fs::remove_file(&log_path);
        return Ok(());
    }
    let tail = log_tail(&log_path);
    let _ = std::fs::remove_file(&log_path);
    let why = if status.success() { "wrote no file".to_string() } else { format!("failed ({status})") };
    Err(format!("{} {why}{}{}", p.name, if tail.is_empty() { "" } else { ": " }, tail))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(kind: &str, command: &str, extra: &str) -> String {
        format!(r#"{{"id":"p","kind":"{kind}","name":"P","command":{command}{extra}}}"#)
    }

    fn one(kind: &str, command: &str, extra: &str) -> Provider {
        parse(&format!(r#"{{"providers":[{}]}}"#, provider(kind, command, extra))).unwrap().remove(0)
    }

    #[test]
    fn a_request_fills_the_command_line_and_a_prompt_stays_one_argument() {
        let p = one(
            "image",
            r#"["/bin/gen","--out","{out}","--prompt","{prompt}","{references}","--size","{width}x{height}","{optional}","--seed","{seed}"]"#,
            r#","perReference":["--reference","{path}"],"optional":{"negative":["--negative","{negative}"],"transparent":["--transparent"],"steps":["--steps","{steps}"]}"#,
        );
        let v = Vars {
            out: "/tmp/a b.png".into(),
            prompt: "a cat; rm -rf / $(x) `y` --seed 9".into(),
            width: Some(1024),
            height: Some(576),
            seed: 7,
            references: vec!["/r/1.png".into(), "/r/2.png".into()],
            transparent: true,
            ..Default::default()
        };
        let a = argv(&p, &v).unwrap();
        assert_eq!(
            a,
            [
                "/bin/gen",
                "--out",
                "/tmp/a b.png",
                "--prompt",
                "a cat; rm -rf / $(x) `y` --seed 9",
                "--reference",
                "/r/1.png",
                "--reference",
                "/r/2.png",
                "--size",
                "1024x576",
                "--transparent",
                "--seed",
                "7"
            ]
        );
        // a value the request did not give leaves its group out, and an empty placeholder empty
        let p2 = one("video", r#"["/bin/v","{out}","{prompt}","{image}","{frames}"]"#, "");
        assert_eq!(
            argv(&p2, &Vars { out: "o.mp4".into(), prompt: "x".into(), frames: Some(49), ..Default::default() }).unwrap(),
            ["/bin/v", "o.mp4", "x", "", "49"]
        );
        assert_eq!(number(4.0), "4");
        assert_eq!(number(2.0416), "2.042");
    }

    #[test]
    fn hostile_provider_lists_are_refused() {
        let wrap = |body: &str| format!(r#"{{"providers":[{body}]}}"#);
        for (bad, why) in [
            (provider("image", r#"["gen","{out}"]"#, ""), "absolute"),
            (provider("image", r#"["/bin/gen","{prompt}"]"#, ""), "{out}"),
            (provider("image", r#"["/bin/gen","{out}","{nope}"]"#, ""), "unknown placeholder"),
            (provider("image", r#"[]"#, ""), "arguments"),
            (provider("image", r#"["/bin/gen","{out}"]"#, r#","optional":{"shell":["x"]}"#), "unknown value"),
            (provider("image", r#"["/bin/gen","{out}"]"#, r#","timeoutSeconds":-1"#), "positive"),
            (provider("image", r#"["/bin/gen","{out}"]"#, r#","extension":"../x""#), "extension"),
            (provider("sound", r#"["/bin/gen","{out}"]"#, ""), "not valid"),
            (provider("image", r#"["/bin/gen","{out}"]"#, r#","cutoutCommand":["cut","{image}","{out}"]"#), "cutoutCommand"),
            (provider("image", r#"["/bin/gen","{out}"]"#, r#","cutoutCommand":["/bin/cut","{out}"]"#), "cutoutCommand"),
        ] {
            let e = parse(&wrap(&bad)).unwrap_err();
            assert!(e.contains(why), "{bad}: {e}");
        }
        let two = format!("{0},{0}", provider("image", r#"["/bin/gen","{out}"]"#, ""));
        assert!(parse(&wrap(&two)).unwrap_err().contains("twice"));
        assert!(parse(&"x".repeat(MANIFEST_MAX as usize + 1)).unwrap_err().contains("larger"));
        let many = vec![r#"{"id":"a","kind":"image","name":"A","command":["/bin/gen","{out}"]}"#; MAX_PROVIDERS + 1].join(",");
        assert!(parse(&wrap(&many)).is_err());
        let long = format!(r#"["/bin/gen","{{out}}","{}"]"#, "a".repeat(MAX_ARG_LEN + 1));
        assert!(parse(&wrap(&provider("image", &long, ""))).unwrap_err().contains("too long"));
        assert!(parse(r#"{"providers":[]}"#).unwrap().is_empty());
    }

    #[test]
    fn sizes_and_frame_counts_are_ones_the_provider_takes() {
        let img = one("image", r#"["/bin/gen","{out}"]"#, r#","multipleOf":32,"maxSide":2048"#);
        assert_eq!(size(&img, (None, None), (16, 9)), (1024, 576));
        assert_eq!(size(&img, (None, None), (9, 16)), (576, 1024));
        assert_eq!(size(&img, (Some(4000), Some(4000)), (1, 1)), (2048, 2048));
        assert_eq!(size(&img, (Some(1000), None), (4, 3)), (992, 736));
        assert_eq!(size(&img, (Some(0), Some(0)), (0, 0)), (64, 64));
        assert_eq!(size(&img, (Some(u32::MAX), Some(1)), (1, 1)), (2048, 64));
        let vid = one("video", r#"["/bin/gen","{out}"]"#, r#","framesStep":8,"framesOffset":1,"fps":24,"maxSeconds":10"#);
        assert_eq!(size(&vid, (None, None), (16, 9)), (480, 272));
        assert_eq!(frames(&vid, 2.0), (49, 24.0));
        assert_eq!(frames(&vid, 4.0), (97, 24.0));
        assert_eq!(frames(&vid, 0.0), (9, 24.0));
        assert_eq!(frames(&vid, 1.0e9), (241, 24.0));
        assert_eq!(frames(&vid, f64::NAN).1, 24.0);
        assert_eq!(seconds(&vid, 60.0, 4.0), 10.0);
        assert_eq!(seconds(&vid, f64::NAN, 4.0), 4.0);
    }

    #[test]
    fn a_reference_prefix_goes_before_a_prompt_that_names_no_picture() {
        let p = one("image", r#"["/bin/gen","{out}"]"#, r#","referencePromptPrefix":"Edit <Picture 1>. ","referenceMarker":"<Picture""#);
        assert_eq!(prompt_for(&p, "add a red kite", 1), "Edit <Picture 1>. add a red kite");
        assert_eq!(prompt_for(&p, "put <Picture 2> beside <Picture 1>", 2), "put <Picture 2> beside <Picture 1>");
        assert_eq!(prompt_for(&p, "a red kite", 0), "a red kite");
    }

    #[cfg(unix)]
    #[test]
    fn a_run_writes_its_file_reports_a_failure_and_can_be_stopped() {
        use std::sync::atomic::AtomicBool;
        let dir = std::env::temp_dir().join(format!("fc-generate-run-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = one("image", r#"["/bin/sh","-c","printf '%s' \"$2\" > \"$1\"","sh","{out}","{prompt}"]"#, "");
        let out = dir.join("ok.png");
        let v = Vars { out: out.to_string_lossy().into_owned(), prompt: "hello; touch /tmp/never \"quoted\"".into(), ..Default::default() };
        run(&p, &argv(&p, &v).unwrap(), &out, &AtomicBool::new(false), &mut |_| {}).unwrap();
        assert_eq!(std::fs::read_to_string(&out).unwrap(), "hello; touch /tmp/never \"quoted\"");
        assert!(!out.with_extension("generation.log").exists());
        // a failing program: its last words are in the error, and no file is claimed
        let bad = one("image", r#"["/bin/sh","-c","echo out of memory >&2; exit 3","sh","{out}"]"#, "");
        let out2 = dir.join("bad.png");
        let v2 = Vars { out: out2.to_string_lossy().into_owned(), ..Default::default() };
        let e = run(&bad, &argv(&bad, &v2).unwrap(), &out2, &AtomicBool::new(false), &mut |_| {}).unwrap_err();
        assert!(e.contains("out of memory") && e.contains("failed"), "{e}");
        // success without a file is a failure too
        let none = one("image", r#"["/bin/sh","-c","true","sh","{out}"]"#, "");
        assert!(run(&none, &argv(&none, &v2).unwrap(), &out2, &AtomicBool::new(false), &mut |_| {}).unwrap_err().contains("wrote no file"));
        // stopped, and past its time limit
        let slow = one("image", r#"["/bin/sh","-c","sleep 30","sh","{out}"]"#, r#","timeoutSeconds":1"#);
        let t0 = web_time::Instant::now();
        assert!(run(&slow, &argv(&slow, &v2).unwrap(), &out2, &AtomicBool::new(true), &mut |_| {}).unwrap_err().contains("stopped"));
        assert!(run(&slow, &argv(&slow, &v2).unwrap(), &out2, &AtomicBool::new(false), &mut |_| {}).unwrap_err().contains("did not finish"));
        assert!(t0.elapsed().as_secs() < 10);
        // a program that is not there
        let gone = one("image", r#"["/nonexistent/gen","{out}"]"#, "");
        assert!(run(&gone, &argv(&gone, &v2).unwrap(), &out2, &AtomicBool::new(false), &mut |_| {}).unwrap_err().contains("not on this computer"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
