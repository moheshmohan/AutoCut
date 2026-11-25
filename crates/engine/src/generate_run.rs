//! Local generation tools (image, video, music and sound-effect models the user installed
//! themselves): the list of them, how a request becomes a command line, and running one.
//!
//! Nothing here is a model. A **provider** is a program on this computer, described in
//! `<data dir>/generation/providers.json`, a file the user keeps (see `docs/generation.md`).
//! No command sets it: what may be run is decided by whoever can write that file, never by a
//! parameter that arrived over the control channel. A request fills placeholders in the
//! provider's argument list; the arguments go to the program as they are (no shell), so a
//! prompt cannot become a command.
//!
//! A `kind: "composition"` entry names the program that renders code-authored motion graphics
//! (`motion.render`, [`crate::motion_graphics`]): `{"id": "raea", "kind": "composition", "name":
//! "RAEA", "command": ["/absolute/path/to/raea"]}`. Its `command` is the program alone (no
//! placeholders: the motion commands build RAEA's command line), or empty to find RAEA where it
//! installs itself.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Deserialize;

use crate::generate_caps::{self as caps, Ask, Capabilities};

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
    /// A renderer of code-authored compositions (RAEA): driven by the `motion.*` commands, never
    /// by `generate.*`.
    Composition,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Image => "image",
            Kind::Video => "video",
            Kind::Music => "music",
            Kind::Sfx => "sfx",
            Kind::Composition => "composition",
        }
    }

    /// The file extension a provider of this kind writes unless it says otherwise.
    pub fn extension(self) -> &'static str {
        match self {
            Kind::Image => "png",
            Kind::Video | Kind::Composition => "mp4",
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
    /// becomes each group of [`Provider::optional`] whose value the request gave. A composition
    /// tool's is its program alone, or empty (found where it installs itself).
    #[serde(default)]
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
    /// A video tool that can carry a clip on: the program that is given the last moments of the
    /// picture as a small video (`{video}`) and writes that video with `{frames}` new frames after
    /// it (`{out}`). The new part is what is placed.
    #[serde(default)]
    pub extend_command: Vec<String>,
    /// The frames of picture the extend program is given (default 33).
    #[serde(default)]
    pub carry_frames: Option<u32>,
    /// Width and height of the video the extend program takes are multiples of this (default 64).
    #[serde(default)]
    pub extend_multiple_of: Option<u32>,
    /// A video tool that can make a clip again: the program that is given a stretch of picture
    /// as a small video (`{video}`, `{frames}` frames) and writes one as long (`{out}`) through
    /// the adapter of the way asked for (`{adapter}`). An argument that is exactly `{optional}`
    /// takes the groups of [`Provider::restyle_optional`].
    #[serde(default)]
    pub restyle_command: Vec<String>,
    #[serde(default)]
    pub restyle_optional: BTreeMap<String, Vec<String>>,
    /// The ways it can make a clip again, by name (`edges`, `relight`…).
    #[serde(default)]
    pub restyles: BTreeMap<String, Restyle>,
    /// About how long that takes (default: twice `estimateSeconds`).
    #[serde(default)]
    pub restyle_estimate_seconds: Option<f64>,
    /// The longest stretch made again in one go (default 5 s, never more than `maxSeconds`).
    #[serde(default)]
    pub restyle_max_seconds: Option<f64>,
    /// What the tool can do (sizes, lengths, frame counts, inputs): a request it cannot do is
    /// refused before it runs ([`crate::generate_caps`]). Without it nothing is checked there.
    #[serde(default)]
    pub capabilities: Option<Capabilities>,
    #[serde(default)]
    pub license: String,
    /// What the model's licence asks to be shown where it is used.
    #[serde(default)]
    pub attribution: String,
    #[serde(default)]
    pub source: String,
}

/// One way a video tool makes a clip again: an adapter of its model.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Restyle {
    /// The adapter's file (an absolute path): the way is there when the file is.
    pub adapter: String,
    /// Arguments added to `restyleCommand` for this way.
    #[serde(default)]
    pub args: Vec<String>,
    /// One line on what it does.
    #[serde(default)]
    pub about: String,
    /// What the adapter's licence asks to be shown, when that differs from the tool's.
    #[serde(default)]
    pub license: String,
}

/// The most ways one tool names.
const MAX_RESTYLES: usize = 16;

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
    /// The picture a clip ends on.
    pub last_image: Option<String>,
    /// The video an extend program carries on from, or a restyle program makes again.
    pub video: Option<String>,
    /// The adapter of the way a clip is made again.
    pub adapter: Option<String>,
    pub references: Vec<String>,
    pub steps: Option<u32>,
    pub strength: Option<f64>,
    pub transparent: bool,
}

const NAMES: [&str; 18] = [
    "out",
    "prompt",
    "negative",
    "width",
    "height",
    "seed",
    "seconds",
    "frames",
    "fps",
    "image",
    "lastImage",
    "lastFrame",
    "video",
    "adapter",
    "steps",
    "strength",
    "transparent",
    "path",
];

fn number(v: f64) -> String {
    let s = format!("{v:.3}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() { "0".into() } else { s.into() }
}

/// A number as a placeholder writes it (at most three decimals, no trailing zeros).
pub(crate) fn number_text(v: f64) -> String {
    number(v)
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
            "lastImage" => self.last_image.clone().filter(|s| !s.is_empty()),
            // the index of a clip's last frame (where its last picture is pinned)
            "lastFrame" => self.frames.map(|v| v.saturating_sub(1).to_string()),
            "video" => self.video.clone().filter(|s| !s.is_empty()),
            "adapter" => self.adapter.clone().filter(|s| !s.is_empty()),
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

/// The size a request gets: [`size`], then what the tool's capabilities allow. A size the
/// request named that the tool cannot make is refused (saying what it makes and the nearest it
/// allows); a size it left open becomes the nearest allowed one.
pub fn size_for(p: &Provider, want: (Option<u32>, Option<u32>), aspect: (u32, u32)) -> Result<(u32, u32), String> {
    let computed = size(p, want, aspect);
    match &p.capabilities {
        Some(c) => {
            let grid = (p.multiple_of.unwrap_or(16).clamp(1, 256), p.max_side.unwrap_or(4096).clamp(64, 16_384));
            caps::fit_size(&p.name, c, computed, want, grid)
        }
        None => Ok(computed),
    }
}

/// The frame rate a video tool makes: its `fps`, else the first its capabilities list, else 24.
pub fn rate(p: &Provider) -> f64 {
    let ok = |f: &f64| f.is_finite() && *f >= 1.0 && *f <= 240.0;
    p.fps.filter(ok).or_else(|| p.capabilities.as_ref().and_then(|c| c.fps.as_ref()).and_then(|r| r.first().copied()).filter(ok)).unwrap_or(24.0)
}

/// `x` within `lo..=hi` (`hi` wins when they cross: never a panic).
fn within(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo.min(hi)).min(hi)
}

/// The longest a tool makes in one go: its `maxSeconds`, its capabilities' length, the shorter of
/// the two, else `default`.
pub fn longest(p: &Provider, default: f64) -> f64 {
    let own = p.max_seconds.filter(|m| m.is_finite() && *m > 0.0);
    let cap = p.capabilities.as_ref().and_then(|c| caps::length_range(c, rate(p))).map(|(_, hi)| hi);
    match (own, cap) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => default,
    }
}

/// The shortest a tool makes: a tenth of a second, or more when its capabilities say so.
fn shortest(p: &Provider) -> f64 {
    p.capabilities.as_ref().and_then(|c| caps::length_range(c, rate(p))).map_or(0.1, |(lo, _)| lo.max(0.1))
}

/// The frame count and rate for `seconds` of video: a count the provider takes, at least the
/// wanted length where its longest allows.
pub fn frames(p: &Provider, seconds: f64) -> (u32, f64) {
    let fps = rate(p);
    let most = longest(p, 10.0);
    let secs = if seconds.is_finite() { within(seconds, 0.1, most) } else { most.min(4.0) };
    let want = (secs * fps).round().clamp(1.0, 100_000.0) as u32;
    let counts = p.capabilities.as_ref().and_then(|c| c.frames);
    let (step, off) = match counts {
        Some(c) => (c.step(), c.offset()),
        None => {
            let step = p.frames_step.unwrap_or(1).clamp(1, 1024);
            (step, p.frames_offset.unwrap_or(0) % step)
        }
    };
    let k = want.saturating_sub(off).div_ceil(step).max(1);
    let n = k.saturating_mul(step).saturating_add(off);
    (counts.map_or(n, |c| c.fit(n)), fps)
}

/// The length a provider can make for a wanted one (seconds).
pub fn seconds(p: &Provider, want: f64, default: f64) -> f64 {
    let most = longest(p, if p.kind == Kind::Video { 10.0 } else { 180.0 });
    let w = if want.is_finite() && want > 0.0 { want } else { default };
    within(w, shortest(p), most)
}

/// [`seconds`] for a length the request named (`asked`) or one taken from the timeline (a gap,
/// a clip): a named length the tool's capabilities do not allow is refused, saying what it makes.
pub fn seconds_for(p: &Provider, want: f64, asked: bool, default: f64) -> Result<f64, String> {
    if asked && let Some(c) = &p.capabilities {
        caps::check_length(&p.name, c, want, rate(p))?;
    }
    Ok(seconds(p, want, default))
}

/// Whether a request's other wishes (a first or last picture, references, transparency, a
/// negative prompt, steps, sound) are ones the tool's capabilities allow.
pub fn ask(p: &Provider, a: &Ask) -> Result<(), String> {
    match &p.capabilities {
        Some(c) => caps::check(&p.name, c, a),
        None => Ok(()),
    }
}

/// Whether the tool takes reference pictures at all.
pub fn takes_references(p: &Provider) -> bool {
    !p.per_reference.is_empty()
}

/// The most reference pictures a request may give the tool.
pub fn max_references(p: &Provider) -> usize {
    let own = p.max_references.map_or(MAX_REFERENCES, |m| (m as usize).min(MAX_REFERENCES));
    p.capabilities.as_ref().and_then(|c| c.references).map_or(own, |n| own.min(n as usize))
}

/// Whether a video tool can start a clip on a given picture (unless its capabilities say not).
pub fn starts_on_picture(p: &Provider) -> bool {
    p.capabilities.as_ref().and_then(|c| c.first_frame) != Some(false)
}

/// Whether a video tool can end a clip on a given picture.
pub fn ends_on_picture(p: &Provider) -> bool {
    p.optional.contains_key("lastImage") && p.capabilities.as_ref().and_then(|c| c.last_frame) != Some(false)
}

/// Whether any argument of the tool's command (its optional groups and per-reference arguments
/// included) takes the placeholder `name`, or an optional group is keyed by it.
fn uses(p: &Provider, name: &str) -> bool {
    let tag = format!("{{{name}}}");
    p.optional.contains_key(name) || p.command.iter().chain(&p.per_reference).chain(p.optional.values().flatten()).any(|a| a.contains(&tag))
}

/// A tool's capabilities checked on their own and against what its entry can do.
fn check_capabilities(p: &Provider) -> Result<(), String> {
    let Some(c) = &p.capabilities else { return Ok(()) };
    c.validate(p.kind)?;
    let claim = |what: &str, needs: &str| Err(format!("`capabilities.{what}` is true, but {needs}"));
    if c.first_frame == Some(true) && !uses(p, "image") {
        return claim("firstFrame", "nothing in `command` or `optional` takes {image}");
    }
    if c.last_frame == Some(true) && !uses(p, "lastImage") {
        return claim("lastFrame", "nothing in `command` or `optional` takes {lastImage}");
    }
    if c.references.is_some_and(|n| n > 0) && !takes_references(p) {
        return Err("`capabilities.references` is more than 0, but there is no `perReference`".into());
    }
    if c.transparent == Some(true) && !p.optional.contains_key("transparent") && p.cutout_command.is_empty() {
        return claim("transparent", "there is no `transparent` option and no `cutoutCommand`");
    }
    if c.negative == Some(true) && !uses(p, "negative") {
        return claim("negative", "nothing in `command` or `optional` takes {negative}");
    }
    if let (Some(own), Some(list)) = (p.fps, &c.fps)
        && !list.iter().any(|f| (f - own).abs() < 1.0e-6)
    {
        return Err(format!("`fps` {} is not one of `capabilities.fps`", number(own)));
    }
    if let Some(f) = c.frames {
        let step = p.frames_step.map(|s| s.clamp(1, 1024));
        if step.is_some_and(|s| s != f.step()) || p.frames_offset.is_some_and(|o| o % f.step() != f.offset()) {
            return Err("`framesStep` / `framesOffset` disagree with `capabilities.frames`".into());
        }
    }
    if let (Some(own), Some(lo)) = (p.max_seconds, c.seconds.and_then(|s| s.min))
        && own < lo
    {
        return Err("`maxSeconds` is shorter than `capabilities.seconds.min`".into());
    }
    Ok(())
}

fn check(p: &Provider) -> Result<(), String> {
    let id_ok = !p.id.is_empty() && p.id.len() <= 64 && p.id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if !id_ok {
        return Err(format!("provider id `{}` must be 1 to 64 letters, digits, - or _", p.id.chars().take(80).collect::<String>()));
    }
    let err = |m: String| Err(format!("provider {}: {m}", p.id));
    if p.kind == Kind::Composition {
        return check_composition(p);
    }
    if p.command.is_empty() || p.command.len() > MAX_ARGS || p.per_reference.len() > MAX_ARGS || p.optional.len() > MAX_ARGS {
        return err(format!("`command` needs 1 to {MAX_ARGS} arguments"));
    }
    let groups = p.optional.values().flat_map(|g| g.iter());
    for arg in p
        .command
        .iter()
        .chain(&p.per_reference)
        .chain(&p.cutout_command)
        .chain(&p.extend_command)
        .chain(&p.restyle_command)
        .chain(p.restyle_optional.values().flat_map(|g| g.iter()))
        .chain(p.restyles.values().flat_map(|w| w.args.iter().chain([&w.adapter, &w.about, &w.license])))
        .chain(groups)
        .chain(p.env.values())
        .chain(p.cwd.iter())
        .chain(p.cutout_prompt_suffix.iter())
    {
        if arg.len() > MAX_ARG_LEN || arg.contains('\0') {
            return err("an argument is too long or holds a NUL".into());
        }
    }
    for arg in p
        .command
        .iter()
        .chain(&p.per_reference)
        .chain(&p.cutout_command)
        .chain(&p.extend_command)
        .chain(&p.restyle_command)
        .chain(p.optional.values().flat_map(|g| g.iter()))
        .chain(p.restyle_optional.values().flat_map(|g| g.iter()))
    {
        if let Err(e) = placeholders(arg) {
            return err(e);
        }
    }
    for key in p.optional.keys().chain(p.restyle_optional.keys()) {
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
    if let Some(ext) = p.extend_command.first() {
        let io = |name: &str| p.extend_command.iter().any(|a| a.contains(name));
        if p.extend_command.len() > MAX_ARGS || !std::path::Path::new(ext).is_absolute() || ext.contains('{') || !io("{video}") || !io("{out}") {
            return err("`extendCommand` needs an absolute program, {video} and {out}".into());
        }
    }
    if let Some(prog) = p.restyle_command.first() {
        let io = |name: &str| p.restyle_command.iter().any(|a| a.contains(name));
        if p.restyle_command.len() > MAX_ARGS || !std::path::Path::new(prog).is_absolute() || prog.contains('{') || !io("{video}") || !io("{out}") {
            return err("`restyleCommand` needs an absolute program, {video} and {out}".into());
        }
    }
    if p.restyles.len() > MAX_RESTYLES || p.restyle_optional.len() > MAX_ARGS || (!p.restyles.is_empty() && p.restyle_command.is_empty()) {
        return err(format!("`restyles` names at most {MAX_RESTYLES} ways and needs a `restyleCommand`"));
    }
    for (name, w) in &p.restyles {
        let name_ok = !name.is_empty() && name.len() <= 32 && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        // (a way's own arguments are taken as they are: no placeholders)
        if !name_ok || !std::path::Path::new(&w.adapter).is_absolute() || w.args.len() > 16 || w.args.iter().any(|a| a.contains('{')) {
            return err(format!(
                "the way `{}` needs a plain name, an absolute `adapter` and at most 16 plain `args`",
                name.chars().take(40).collect::<String>()
            ));
        }
    }
    if p.env.len() > 64 || p.env.keys().any(|k| k.is_empty() || k.len() > 128 || k.contains('=') || k.contains('\0')) {
        return err("`env` has too many or malformed names".into());
    }
    if p.extension.as_deref().is_some_and(|e| e.is_empty() || e.len() > 8 || !e.chars().all(|c| c.is_ascii_alphanumeric())) {
        return err("`extension` must be letters and digits".into());
    }
    for (name, v) in [
        ("estimateSeconds", p.estimate_seconds),
        ("timeoutSeconds", p.timeout_seconds),
        ("maxSeconds", p.max_seconds),
        ("fps", p.fps),
        ("restyleEstimateSeconds", p.restyle_estimate_seconds),
        ("restyleMaxSeconds", p.restyle_max_seconds),
    ] {
        if v.is_some_and(|x| !x.is_finite() || x <= 0.0 || x > 1.0e6) {
            return err(format!("`{name}` must be a positive number"));
        }
    }
    check_capabilities(p).or_else(err)
}

/// A composition tool: at most its program (an absolute path, no placeholders), nothing a
/// generation tool takes, and sane limits.
fn check_composition(p: &Provider) -> Result<(), String> {
    let err = |m: &str| Err(format!("composition tool {}: {m}", p.id));
    match p.command.as_slice() {
        [] => {}
        [program] if std::path::Path::new(program).is_absolute() && !program.contains(['{', '\0']) && program.len() <= MAX_ARG_LEN => {}
        _ => return err("`command` is the program alone, an absolute path (or empty, to find RAEA where it installs itself)"),
    }
    let generation_only = !p.per_reference.is_empty()
        || !p.optional.is_empty()
        || !p.cutout_command.is_empty()
        || !p.extend_command.is_empty()
        || !p.restyle_command.is_empty()
        || !p.restyles.is_empty()
        || p.capabilities.is_some();
    if generation_only {
        return err("it takes no generation options (perReference, optional, cutoutCommand, extendCommand, restyleCommand, restyles, capabilities)");
    }
    if p.env.len() > 64 || p.env.iter().any(|(k, v)| k.is_empty() || k.len() > 128 || k.contains(['=', '\0']) || v.len() > MAX_ARG_LEN || v.contains('\0')) {
        return err("`env` has too many or malformed names");
    }
    if p.cwd.is_some() {
        return err("`cwd` is not used: RAEA runs in the composition's folder");
    }
    if p.timeout_seconds.is_some_and(|x| !x.is_finite() || x <= 0.0 || x > 1.0e6) || p.estimate_seconds.is_some_and(|x| !x.is_finite() || x <= 0.0 || x > 1.0e6)
    {
        return err("`timeoutSeconds` and `estimateSeconds` must be positive numbers");
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

/// The provider's extend step as a provider of its own (the same environment and folder), when
/// it has one.
pub fn extender(p: &Provider) -> Option<Provider> {
    if p.extend_command.is_empty() {
        return None;
    }
    let mut c = p.clone();
    c.command = p.extend_command.clone();
    c.name = format!("{} (carrying on)", p.name);
    c.per_reference.clear();
    c.optional.clear();
    Some(c)
}

/// The provider's restyle step for one of its ways as a provider of its own (the same
/// environment and folder), when it has that way.
pub fn restyler(p: &Provider, way: &str) -> Option<Provider> {
    let w = p.restyles.get(way)?;
    if p.restyle_command.is_empty() {
        return None;
    }
    let mut c = p.clone();
    c.command = p.restyle_command.iter().chain(&w.args).cloned().collect();
    c.name = format!("{} ({way})", p.name);
    c.per_reference.clear();
    c.optional = p.restyle_optional.clone();
    c.estimate_seconds = Some(p.restyle_estimate_seconds.unwrap_or(p.estimate_seconds.unwrap_or(60.0) * 2.0).clamp(1.0, 86_400.0));
    Some(c)
}

/// Whether a way's adapter is on this computer.
pub fn restyle_installed(p: &Provider, way: &str) -> bool {
    !p.restyle_command.is_empty() && p.restyles.get(way).is_some_and(|w| std::path::Path::new(&w.adapter).is_file())
}

/// How a stretch of `seconds` is made again.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stretch {
    /// The frames the tool makes: the most its counts allow inside the stretch.
    pub frames: u32,
    /// The rate the tool makes them at.
    pub fps: f64,
    /// The seconds of the timeline they cover.
    pub covered: f64,
    /// The rate the stretch is written at for the tool, so that it holds exactly `frames`.
    pub source_fps: f64,
}

/// The frames a stretch of `seconds` is made again as. A stretch the tool can do whole is
/// written at a slightly lower rate, so that a frame count the tool takes spans all of it and
/// every new frame shows the moment of the old one; a longer stretch is covered from its start
/// as far as the tool goes. `None` when it is shorter than the fewest frames the tool makes.
pub fn restyle_stretch(p: &Provider, seconds: f64) -> Option<Stretch> {
    let fps = rate(p);
    let cap = p.max_seconds.filter(|m| m.is_finite() && *m > 0.0).unwrap_or(10.0);
    let most = p.restyle_max_seconds.filter(|m| m.is_finite() && *m > 0.0).unwrap_or(5.0).min(cap);
    if !seconds.is_finite() || seconds <= 0.0 {
        return None;
    }
    let whole = seconds <= most + 1.0e-9;
    let have = (seconds.min(most) * fps + 1.0e-6).floor().clamp(0.0, 100_000.0) as u32;
    let step = p.frames_step.unwrap_or(1).clamp(1, 1024);
    let off = p.frames_offset.unwrap_or(0) % step;
    let k = have.checked_sub(off)? / step;
    if k == 0 {
        return None;
    }
    let frames = k.saturating_mul(step).saturating_add(off);
    let made = f64::from(frames) / fps;
    Some(if whole {
        Stretch { frames, fps, covered: seconds, source_fps: f64::from(frames) / seconds }
    } else {
        Stretch { frames, fps, covered: made, source_fps: fps }
    })
}

/// The frames of picture an extend program is given.
pub fn carry_frames(p: &Provider) -> u32 {
    p.carry_frames.unwrap_or(33).clamp(9, 241)
}

/// A size on a coarse grid (each side a multiple of `step`) closest to the shape `aspect`, its
/// long side near `long`: snapping each side on its own would bend the picture.
pub fn grid_size(aspect: (u32, u32), long: u32, step: u32) -> (u32, u32) {
    let step = step.clamp(1, 512);
    let (a, b) = (f64::from(aspect.0.max(1)), f64::from(aspect.1.max(1)));
    let ratio = (a.max(b) / a.min(b)).clamp(1.0, 64.0);
    let long = long.clamp(step, 8192);
    let (lo, hi) = (f64::from(long) * 0.75, f64::from(long) * 1.34);
    let mut best: Option<(f64, u32, u32)> = None;
    let mut l = step;
    while l <= 8192 {
        if f64::from(l) >= lo && f64::from(l) <= hi {
            let s = ((f64::from(l) / ratio / f64::from(step)).round().clamp(1.0, 8192.0) as u32).saturating_mul(step).min(l);
            let off = (f64::from(l) / f64::from(s) / ratio - 1.0).abs() + (f64::from(l) - f64::from(long)).abs() / f64::from(long) * 0.01;
            if best.is_none_or(|(e, _, _)| off < e) {
                best = Some((off, l, s));
            }
        }
        l = l.saturating_add(step);
    }
    let (_, l, s) = best.unwrap_or((0.0, long.div_ceil(step).saturating_mul(step), long.div_ceil(step).saturating_mul(step)));
    if aspect.0 >= aspect.1 { (l, s) } else { (s, l) }
}

/// What a tool's output says about how far it is: its last line `progress 0.42`, `progress 42%`
/// or `progress 3/8`, as a fraction.
pub fn progress_in(text: &str) -> Option<f64> {
    let line = text.split(['\n', '\r']).rev().map(str::trim).find(|l| l.len() > 9 && l.get(..9).is_some_and(|h| h.eq_ignore_ascii_case("progress ")))?;
    let v = line.get(9..)?.trim();
    let f = if let Some(pct) = v.strip_suffix('%') {
        pct.trim().parse::<f64>().ok()? / 100.0
    } else if let Some((n, m)) = v.split_once('/') {
        let (n, m) = (n.trim().parse::<f64>().ok()?, m.trim().parse::<f64>().ok()?);
        if m <= 0.0 {
            return None;
        }
        n / m
    } else {
        v.parse::<f64>().ok()?
    };
    f.is_finite().then(|| f.clamp(0.0, 1.0))
}

/// Whether the provider's program is there to run (a composition tool without a program: whether
/// RAEA is found where it installs itself).
pub fn installed(p: &Provider) -> bool {
    match p.command.first() {
        Some(c) => std::path::Path::new(c).is_file(),
        None => p.kind == Kind::Composition && crate::motion_graphics::locate(Some(p)).is_ok(),
    }
}

/// The last 1500 bytes of a run's log.
#[cfg(not(target_arch = "wasm32"))]
fn log_end(path: &std::path::Path) -> String {
    use std::io::{Read as _, Seek as _};
    let Ok(mut f) = std::fs::File::open(path) else { return String::new() };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let _ = f.seek(std::io::SeekFrom::Start(len.saturating_sub(1500)));
    let mut bytes = Vec::new();
    let _ = f.take(1500).read_to_end(&mut bytes);
    String::from_utf8_lossy(&bytes).into_owned()
}

/// The end of a run's log, for an error message.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn log_tail(path: &std::path::Path) -> String {
    let text = log_end(path);
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let from = lines.len().saturating_sub(6);
    lines.get(from..).unwrap_or(&[]).join(" | ")
}

/// Runs a provider's command line until it has written `out`: `tick` hears the seconds gone by
/// and how far the tool says it is ([`progress_in`], when it says), `cancel` stops it (the
/// program is killed), and so does its time limit.
#[cfg(not(target_arch = "wasm32"))]
pub fn run(
    p: &Provider,
    args: &[String],
    out: &std::path::Path,
    cancel: &std::sync::atomic::AtomicBool,
    tick: &mut dyn FnMut(f64, Option<f64>),
) -> Result<(), String> {
    use std::process::{Command, Stdio};
    use std::sync::atomic::Ordering;
    let Some((program, rest)) = args.split_first() else { return Err(format!("{} has no program", p.id)) };
    if !std::path::Path::new(program).is_file() {
        return Err(format!("{}: its program {program} is not on this computer (see docs/generation.md)", p.name));
    }
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    // a file that is made again goes first: what is there afterwards is what this run wrote
    let _ = std::fs::remove_file(out);
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
    let (mut turns, mut said) = (0u32, None);
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
        // the log is looked at about once a second
        if turns % 7 == 0 {
            said = progress_in(&log_end(&log_path)).or(said);
        }
        turns = turns.wrapping_add(1);
        tick(gone, said);
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

/// Tests that write the provider list (one file per test binary) take turns on this.
#[cfg(test)]
pub(crate) static LIST_TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
    fn a_composition_entry_is_its_program_alone() {
        let wrap = |body: &str| format!(r#"{{"providers":[{body}]}}"#);
        let ok = parse(&wrap(r#"{"id":"raea","kind":"composition","name":"RAEA","command":["/opt/raea/bin/raea"],"timeoutSeconds":600}"#)).unwrap();
        assert_eq!((ok[0].kind, ok[0].kind.name()), (Kind::Composition, "composition"));
        // without a program it is found where RAEA installs itself
        assert!(parse(&wrap(r#"{"id":"raea","kind":"composition","name":"RAEA"}"#)).unwrap()[0].command.is_empty());
        for (bad, why) in [
            (r#"{"id":"raea","kind":"composition","name":"RAEA","command":["raea"]}"#, "absolute"),
            (r#"{"id":"raea","kind":"composition","name":"RAEA","command":["/bin/raea","render","{out}"]}"#, "program alone"),
            (r#"{"id":"raea","kind":"composition","name":"RAEA","command":["/bin/{x}"]}"#, "absolute"),
            (r#"{"id":"raea","kind":"composition","name":"RAEA","optional":{"seed":["--seed"]}}"#, "generation options"),
            (r#"{"id":"raea","kind":"composition","name":"RAEA","cwd":"/tmp"}"#, "cwd"),
            (r#"{"id":"raea","kind":"composition","name":"RAEA","timeoutSeconds":0}"#, "positive"),
            (r#"{"id":"raea","kind":"composition","name":"RAEA","env":{"A=B":"1"}}"#, "env"),
            (r#"{"id":"../raea","kind":"composition","name":"RAEA"}"#, "letters"),
        ] {
            let e = parse(&wrap(bad)).unwrap_err();
            assert!(e.contains(why), "{bad}: {e}");
        }
        // a generation tool still needs its program and {out}
        assert!(parse(&wrap(r#"{"id":"img","kind":"image","name":"I"}"#)).unwrap_err().contains("arguments"));
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
    fn a_tool_that_says_how_far_it_is_is_heard() {
        assert_eq!(progress_in("loading\nprogress 0.25\nstep\n"), Some(0.25));
        assert_eq!(progress_in("progress 0.1\r\nPROGRESS 40%\n"), Some(0.4));
        assert_eq!(progress_in("progress 3/8"), Some(0.375));
        assert_eq!(progress_in("progress 9"), Some(1.0), "never past the end");
        assert_eq!(progress_in("progress -4"), Some(0.0));
        for silent in ["", "progress", "progress  ", "progress x", "progress 1/0", "progress NaN", "in progress 0.5", "progression 0.5", "progress 5/"] {
            assert_eq!(progress_in(silent), None, "{silent:?}");
        }
    }

    /// A tool's log is read while it runs: the bar hears the last thing it said.
    #[cfg(unix)]
    #[test]
    fn the_progress_a_running_tool_prints_reaches_the_bar() {
        let dir = std::env::temp_dir().join(format!("fc-generate-progress-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("a.png");
        let p = one("image", r#"["/bin/sh","-c","echo progress 0.5; sleep 1.6; echo made > \"$1\"","sh","{out}"]"#, "");
        let v = Vars { out: out.to_string_lossy().into_owned(), ..Default::default() };
        let mut heard = Vec::new();
        run(&p, &argv(&p, &v).unwrap(), &out, &std::sync::atomic::AtomicBool::new(false), &mut |_, said| heard.push(said)).unwrap();
        assert!(heard.contains(&Some(0.5)), "{heard:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_coarse_grid_keeps_the_shape_of_the_picture() {
        assert_eq!(grid_size((1920, 1080), 480, 64), (576, 320));
        assert_eq!(grid_size((1080, 1920), 480, 64), (320, 576));
        assert_eq!(grid_size((1, 1), 480, 64), (448, 448));
        // hostile shapes and steps still give a size on the grid
        for (aspect, long, step) in [((0, 0), 0, 0), ((u32::MAX, 1), u32::MAX, u32::MAX), ((1, u32::MAX), 7, 3), ((16, 9), 480, 7)] {
            let (w, h) = grid_size(aspect, long, step);
            let s = step.clamp(1, 512);
            assert!(w >= 1 && h >= 1 && w % s == 0 && h % s == 0 && w <= 8192 && h <= 8192, "{aspect:?} {long} {step}: {w}×{h}");
        }
    }

    #[test]
    fn a_clip_can_end_on_a_picture_and_an_extend_program_needs_its_video() {
        let p = one(
            "video",
            r#"["/bin/gen","{out}","{frames}","{optional}"]"#,
            r#","optional":{"image":["--image","{image}","0"],"lastImage":["--image","{lastImage}","{lastFrame}"]},"extendCommand":["/bin/ext","{video}","{frames}","{out}"]"#,
        );
        let v = Vars { out: "/o.mp4".into(), frames: Some(49), image: Some("/a.png".into()), last_image: Some("/b.png".into()), ..Default::default() };
        assert_eq!(argv(&p, &v).unwrap(), ["/bin/gen", "/o.mp4", "49", "--image", "/a.png", "0", "--image", "/b.png", "48"]);
        let e = extender(&p).unwrap();
        let v = Vars { out: "/o.mp4".into(), frames: Some(48), video: Some("/carried.mp4".into()), ..Default::default() };
        assert_eq!(argv(&e, &v).unwrap(), ["/bin/ext", "/carried.mp4", "48", "/o.mp4"]);
        assert_eq!(carry_frames(&p), 33);
        assert!(extender(&one("video", r#"["/bin/gen","{out}"]"#, "")).is_none());
        for (extra, why) in [
            (r#","extendCommand":["ext","{video}","{out}"]"#, "extendCommand"),
            (r#","extendCommand":["/bin/ext","{out}"]"#, "extendCommand"),
            (r#","extendCommand":["/bin/ext","{video}"]"#, "extendCommand"),
            (r#","extendCommand":["/bin/ext","{video}","{out}","{nope}"]"#, "unknown placeholder"),
        ] {
            let e = parse(&format!(r#"{{"providers":[{}]}}"#, provider("video", r#"["/bin/gen","{out}"]"#, extra))).unwrap_err();
            assert!(e.contains(why), "{extra}: {e}");
        }
    }

    #[test]
    fn a_clip_is_made_again_through_an_adapter_in_frames_the_tool_takes() {
        let p = one(
            "video",
            r#"["/bin/gen","{out}"]"#,
            r#","framesStep":8,"framesOffset":1,"maxSeconds":10,"estimateSeconds":40,"restyleCommand":["/bin/re","{video}","{adapter}","{frames}","{strength}","{optional}","{out}"],"restyleOptional":{"image":["--image","{image}"]},"restyles":{"edges":{"adapter":"/a/union.safetensors","args":["--how","edges"],"about":"keeps the shapes"},"relight":{"adapter":"/a/relight.safetensors"}}"#,
        );
        let r = restyler(&p, "edges").unwrap();
        assert_eq!(r.estimate_seconds, Some(80.0), "twice a clip's time unless the list says");
        let v = Vars {
            out: "/o.mp4".into(),
            frames: Some(65),
            video: Some("/src.mp4".into()),
            adapter: Some("/a/union.safetensors".into()),
            strength: Some(1.0),
            ..Default::default()
        };
        assert_eq!(argv(&r, &v).unwrap(), ["/bin/re", "/src.mp4", "/a/union.safetensors", "65", "1", "/o.mp4", "--how", "edges"]);
        let with = Vars { image: Some("/first.png".into()), ..v };
        assert_eq!(argv(&r, &with).unwrap(), ["/bin/re", "/src.mp4", "/a/union.safetensors", "65", "1", "--image", "/first.png", "/o.mp4", "--how", "edges"]);
        assert!(restyler(&p, "pose").is_none() && !restyle_installed(&p, "edges") && !restyle_installed(&p, "pose"));
        // three seconds: 65 frames spread over all of it (written at 65/3 frames a second)
        let s = restyle_stretch(&p, 3.0).unwrap();
        assert_eq!((s.frames, s.fps, s.covered), (65, 24.0, 3.0));
        assert!((s.source_fps - 65.0 / 3.0).abs() < 1e-9, "{s:?}");
        // a stretch that is a count the tool takes is written as it is
        let s = restyle_stretch(&p, 73.0 / 24.0).unwrap();
        assert_eq!((s.frames, s.source_fps), (73, 24.0));
        // longer than one go: from its start as far as the tool goes (five seconds unless the list says)
        let s = restyle_stretch(&p, 12.0).unwrap();
        assert_eq!((s.frames, s.source_fps, s.covered), (113, 24.0, 113.0 / 24.0));
        let long = one("video", r#"["/bin/gen","{out}"]"#, r#","framesStep":8,"framesOffset":1,"maxSeconds":6,"restyleMaxSeconds":60"#);
        assert_eq!(restyle_stretch(&long, 12.0).unwrap().frames, 137, "never past maxSeconds");
        for bad in [0.2, 0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(restyle_stretch(&p, bad).is_none(), "{bad}");
        }
        for (extra, why) in [
            (r#","restyleCommand":["re","{video}","{out}"]"#, "restyleCommand"),
            (r#","restyleCommand":["/bin/re","{out}"]"#, "restyleCommand"),
            (r#","restyleCommand":["/bin/re","{video}","{out}","{nope}"]"#, "unknown placeholder"),
            (r#","restyles":{"edges":{"adapter":"/a.safetensors"}}"#, "needs a `restyleCommand`"),
            (r#","restyleCommand":["/bin/re","{video}","{out}"],"restyles":{"edges":{"adapter":"a.safetensors"}}"#, "absolute `adapter`"),
            (r#","restyleCommand":["/bin/re","{video}","{out}"],"restyles":{"bad name":{"adapter":"/a.safetensors"}}"#, "plain name"),
            (r#","restyleCommand":["/bin/re","{video}","{out}"],"restyles":{"edges":{"adapter":"/a.safetensors","args":["{prompt}"]}}"#, "plain `args`"),
            (r#","restyleCommand":["/bin/re","{video}","{out}"],"restyleOptional":{"nope":["--x"]}"#, "unknown value"),
            (r#","restyleMaxSeconds":-1"#, "restyleMaxSeconds"),
        ] {
            let e = parse(&format!(r#"{{"providers":[{}]}}"#, provider("video", r#"["/bin/gen","{out}"]"#, extra))).unwrap_err();
            assert!(e.contains(why), "{extra}: {e}");
        }
    }

    #[test]
    fn a_run_writes_its_file_reports_a_failure_and_can_be_stopped() {
        use std::sync::atomic::AtomicBool;
        let dir = std::env::temp_dir().join(format!("fc-generate-run-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = one("image", r#"["/bin/sh","-c","printf '%s' \"$2\" > \"$1\"","sh","{out}","{prompt}"]"#, "");
        let out = dir.join("ok.png");
        let v = Vars { out: out.to_string_lossy().into_owned(), prompt: "hello; touch /tmp/never \"quoted\"".into(), ..Default::default() };
        run(&p, &argv(&p, &v).unwrap(), &out, &AtomicBool::new(false), &mut |_, _| {}).unwrap();
        assert_eq!(std::fs::read_to_string(&out).unwrap(), "hello; touch /tmp/never \"quoted\"");
        assert!(!out.with_extension("generation.log").exists());
        // a failing program: its last words are in the error, and no file is claimed
        let bad = one("image", r#"["/bin/sh","-c","echo out of memory >&2; exit 3","sh","{out}"]"#, "");
        let out2 = dir.join("bad.png");
        let v2 = Vars { out: out2.to_string_lossy().into_owned(), ..Default::default() };
        let e = run(&bad, &argv(&bad, &v2).unwrap(), &out2, &AtomicBool::new(false), &mut |_, _| {}).unwrap_err();
        assert!(e.contains("out of memory") && e.contains("failed"), "{e}");
        // success without a file is a failure too
        let none = one("image", r#"["/bin/sh","-c","true","sh","{out}"]"#, "");
        assert!(run(&none, &argv(&none, &v2).unwrap(), &out2, &AtomicBool::new(false), &mut |_, _| {}).unwrap_err().contains("wrote no file"));
        // stopped, and past its time limit
        let slow = one("image", r#"["/bin/sh","-c","sleep 30","sh","{out}"]"#, r#","timeoutSeconds":1"#);
        let t0 = web_time::Instant::now();
        assert!(run(&slow, &argv(&slow, &v2).unwrap(), &out2, &AtomicBool::new(true), &mut |_, _| {}).unwrap_err().contains("stopped"));
        assert!(run(&slow, &argv(&slow, &v2).unwrap(), &out2, &AtomicBool::new(false), &mut |_, _| {}).unwrap_err().contains("did not finish"));
        assert!(t0.elapsed().as_secs() < 10);
        // a program that is not there
        let gone = one("image", r#"["/nonexistent/gen","{out}"]"#, "");
        assert!(run(&gone, &argv(&gone, &v2).unwrap(), &out2, &AtomicBool::new(false), &mut |_, _| {}).unwrap_err().contains("not on this computer"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `maxSeconds` under a tenth of a second made `frames` and `seconds` clamp with crossing
    /// bounds, which panics.
    #[test]
    fn a_tool_that_makes_less_than_a_tenth_of_a_second_does_not_crash() {
        let p = one("video", r#"["/bin/gen","{out}"]"#, r#","maxSeconds":0.05"#);
        assert_eq!(frames(&p, 2.0), (1, 24.0));
        assert_eq!(frames(&p, f64::NAN), (1, 24.0));
        assert!((seconds(&p, 2.0, 4.0) - 0.05).abs() < 1e-12);
        assert!((seconds(&p, -1.0, f64::NAN) - 0.05).abs() < 1e-12);
    }

    #[test]
    fn capabilities_must_match_what_the_entry_can_do() {
        let wrap = |body: &str| format!(r#"{{"providers":[{body}]}}"#);
        for (kind, command, extra, why) in [
            ("video", r#"["/bin/gen","{out}"]"#, r#","capabilities":{"firstFrame":true}"#, "takes {image}"),
            ("video", r#"["/bin/gen","{out}"]"#, r#","capabilities":{"lastFrame":true}"#, "takes {lastImage}"),
            ("image", r#"["/bin/gen","{out}"]"#, r#","capabilities":{"references":2}"#, "perReference"),
            ("image", r#"["/bin/gen","{out}"]"#, r#","capabilities":{"transparent":true}"#, "cutoutCommand"),
            ("image", r#"["/bin/gen","{out}"]"#, r#","capabilities":{"negative":true}"#, "takes {negative}"),
            ("video", r#"["/bin/gen","{out}"]"#, r#","fps":25,"capabilities":{"fps":[24,30]}"#, "not one of"),
            ("video", r#"["/bin/gen","{out}"]"#, r#","framesStep":4,"capabilities":{"frames":{"step":8,"offset":1}}"#, "disagree"),
            ("video", r#"["/bin/gen","{out}"]"#, r#","maxSeconds":2,"capabilities":{"seconds":{"min":3}}"#, "shorter than"),
            ("video", r#"["/bin/gen","{out}"]"#, r#","capabilities":{"references":1}"#, "not something a video tool"),
            ("image", r#"["/bin/gen","{out}"]"#, r#","capabilities":{"width":{"step":0}}"#, "step"),
            ("image", r#"["/bin/gen","{out}"]"#, r#","capabilities":{"colour":true}"#, "unknown field"),
        ] {
            let e = parse(&wrap(&provider(kind, command, extra))).unwrap_err();
            assert!(e.contains(why), "{extra}: {e}");
        }
        let e = parse(&wrap(r#"{"id":"raea","kind":"composition","name":"RAEA","capabilities":{}}"#)).unwrap_err();
        assert!(e.contains("capabilities"), "{e}");
        // what an entry can do, it may say; frame counts and rates then come from the block
        let p = one(
            "video",
            r#"["/bin/gen","{out}","{optional}"]"#,
            r#","optional":{"image":["--image","{image}"],"lastImage":["--last","{lastImage}"]},"capabilities":{"firstFrame":true,"lastFrame":true,"frames":{"min":9,"max":121,"step":8,"offset":1},"fps":[25]}"#,
        );
        assert_eq!(rate(&p), 25.0);
        assert_eq!(frames(&p, 2.0), (57, 25.0));
        assert_eq!(frames(&p, 30.0), (121, 25.0), "never past its most frames");
        assert!((longest(&p, 10.0) - 121.0 / 25.0).abs() < 1e-9);
        assert!(starts_on_picture(&p) && ends_on_picture(&p));
        // without the block the tool is as it always was
        let plain = one("video", r#"["/bin/gen","{out}"]"#, r#","framesStep":8,"framesOffset":1"#);
        assert_eq!(size_for(&plain, (Some(1000), Some(1000)), (1, 1)), Ok(size(&plain, (Some(1000), Some(1000)), (1, 1))));
        assert_eq!(seconds_for(&plain, 60.0, true, 4.0), Ok(10.0));
        assert_eq!(frames(&plain, 4.0), (97, 24.0));
        assert!(
            ask(&plain, &Ask { first_frame: true, last_frame: true, references: 99, transparent: true, negative: true, steps: Some(9999), audio: true })
                .is_ok()
        );
    }
}
