//! A branded channel outro for a showcase ([`crate::showcase::build`]), from the plan's `outro`
//! field (or the CLI's `--outro-*` flags): it replaces the plain "N ROBOTS and counting" card.
//!
//! ```json
//! "outro": {"logo": "/path/logo.png", "channel": "H4HACKS", "handle": "@H4hacks",
//!           "tagline": "Hacking, gadgets and robots in English and Malayalam",
//!           "cta": "SUBSCRIBE", "ctaDone": "SUBSCRIBED", "terminal": "> h4hacks.com",
//!           "background": [{"item": 4830, "from": 12, "to": 30}], "seconds": 12}
//! ```
//!
//! Every word on screen comes from these fields (a Malayalam cut gives Malayalam ones); a field
//! left out leaves its element out. The spoken outro line (`script.outro`) plays over it.
//!
//! **Motion** (all engine graphics, keyframes and effects; [`build`]):
//! 1. *Reveal* (the first [`reveal_end`] seconds, about 4.5): the `background` footage (else a
//!    near-black gradient) blurred and darkened under a near-black purple wash; the logo, on a
//!    light badge (as it was drawn) glowing purple, glitches in at the centre (an opacity flicker
//!    and a jitter, held frame to frame) and scales up past its size and back, a cyan ring
//!    pulsing out, and two rings of
//!    rainbow halftone dots — the logo's own — bursting outward and fading; then the `cta` button
//!    pops in under it, a cursor glides to it and clicks (a ripple; the button turns to `ctaDone`).
//! 2. *End screen* (the rest, at least [`END_SCREEN_MIN`] and at most [`END_SCREEN_MAX`] s): the
//!    button goes, the logo glides up into a lockup with the `channel` name (the title font, in
//!    the logo's purple-to-blue) and the `handle` (the number font, in the theme's accent), the
//!    `tagline` comes in along the bottom and the `terminal` line types itself out under it with
//!    a blinking cursor. The middle of the frame stays clear for YouTube's end-screen elements
//!    ([`end_screen`]): a video tile left of centre and the subscribe button right of it.
//!    The last half second fades to black.
//!
//! **End-screen layout** (YouTube: elements in the last 5–20 s of a video at least 25 s long, up
//! to four, placed by the creator in YouTube Studio; a video element is about a third of the
//! frame wide, 16:9, the subscribe element a circle about 300 px across on a 1920×1080 frame).
//! On 1920×1080: the video tile region is x 180–900, y 380–785 (720×405, a little larger than
//! the default tile so it can grow), the subscribe region a 350 px square centred at (1350, 582);
//! the lockup stays above y 360, the tagline and terminal line below y 820. Other sizes scale.

use serde::Serialize;
use serde_json::{Value, json};

use crate::look::{self, Bounds, Role, Theme};
use crate::showcase::{on_frame, run, ticks};
use crate::{AutoCutError, Executor};

fn invalid(msg: impl Into<String>) -> AutoCutError {
    AutoCutError::Invalid(msg.into())
}

/// Shortest, longest and usual outro, seconds.
pub const MIN_SECONDS: f64 = 8.0;
pub const MAX_SECONDS: f64 = 30.0;
pub const DEFAULT_SECONDS: f64 = 12.0;
/// The end screen's shortest and longest stretch (YouTube's 5–20 s), and the reveal before it.
pub const END_SCREEN_MIN: f64 = 5.0;
pub const END_SCREEN_MAX: f64 = 20.0;
const REVEAL: f64 = 4.5;
const MIN_REVEAL: f64 = 3.0;
/// Longest name, handle and button text; longest tagline and terminal line (characters).
const MAX_SHORT: usize = 40;
const MAX_LONG: usize = 90;
/// Longest logo path, most background clips, longest background window (seconds).
const MAX_PATH: usize = 4096;
const MAX_BACKGROUND: usize = 8;
const MAX_FOOTAGE: f64 = 600.0;

/// A stretch of footage under the outro: `item` from `from` to `to` seconds (either end left out:
/// the clip's start, or as long as the outro needs).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Footage {
    pub item: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<f64>,
}

/// The plan's `outro` ([`Outro::from_plan`]).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Outro {
    /// An image file (PNG with transparency, square): imported when the show is built.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub handle: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tagline: Option<String>,
    /// The subscribe button's words, and what it says once clicked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cta: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cta_done: Option<String>,
    /// A line typed out terminal-style ("> h4hacks.com").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terminal: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub background: Vec<Footage>,
    /// How long it lasts at least (longer when the spoken outro needs it).
    pub seconds: f64,
}

/// The fields an `outro` may hold.
const FIELDS: &[&str] = &["logo", "channel", "handle", "tagline", "cta", "ctaDone", "terminal", "background", "seconds"];

fn short(v: &Value) -> String {
    v.to_string().chars().take(60).collect()
}

/// A field's text: one line of 1 to `max` characters (trimmed), or `None` when not given.
fn text(o: &serde_json::Map<String, Value>, k: &str, max: usize) -> Result<Option<String>, AutoCutError> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => {
            let t = s.trim();
            if t.is_empty() {
                return Ok(None);
            }
            if t.chars().count() > max || t.chars().any(char::is_control) {
                return Err(invalid(format!("outro.{k}: one line of at most {max} characters")));
            }
            Ok(Some(t.to_string()))
        }
        Some(other) => Err(invalid(format!("outro.{k}: text, not {}", short(other)))),
    }
}

/// A time of a background window: seconds, from 0.
fn seconds_at(at: &str, k: &str, v: Option<&Value>) -> Result<Option<f64>, AutoCutError> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(x) => match x.as_f64() {
            Some(s) if s.is_finite() && (0.0..=24.0 * 3600.0).contains(&s) => Ok(Some(s)),
            _ => Err(invalid(format!("{at}.{k}: seconds into the clip, from 0, not {}", short(x)))),
        },
    }
}

impl Outro {
    /// The plan's `outro`: an object of the fields above (`null` or `false`: none; text: none
    /// either — a script file read as a plan holds the spoken outro line there). Everything
    /// is checked: unknown fields, text that is not one short line, a logo path that is not a
    /// path, background windows that end before they start or run too long, a length outside
    /// [`MIN_SECONDS`]–[`MAX_SECONDS`]; and it needs a logo or a channel name.
    pub fn from_plan(v: &Value) -> Result<Option<Self>, AutoCutError> {
        let o = match v {
            // (a script alone: its spoken `outro` line, not a branded outro)
            Value::Null | Value::Bool(false) | Value::String(_) => return Ok(None),
            Value::Object(o) => o,
            other => return Err(invalid(format!("outro: an object {{logo, channel, handle, tagline, cta, …}}, not {}", short(other)))),
        };
        if let Some(k) = o.keys().find(|k| !FIELDS.contains(&k.as_str())) {
            return Err(invalid(format!("outro: unknown field `{}` ({})", k.chars().take(40).collect::<String>(), FIELDS.join(", "))));
        }
        let logo = match o.get("logo") {
            None | Some(Value::Null) => None,
            Some(Value::String(p)) if !p.trim().is_empty() && p.chars().count() <= MAX_PATH && !p.contains('\0') => Some(p.trim().to_string()),
            Some(other) => return Err(invalid(format!("outro.logo: the path of an image file, not {}", short(other)))),
        };
        let mut background = Vec::new();
        match o.get("background") {
            None | Some(Value::Null) => {}
            Some(Value::Array(list)) if list.len() <= MAX_BACKGROUND => {
                for (i, f) in list.iter().enumerate() {
                    let at = format!("outro.background[{i}]");
                    let fo = f.as_object().ok_or_else(|| invalid(format!("{at}: {{item, from, to}}, not {}", short(f))))?;
                    if let Some(k) = fo.keys().find(|k| !["item", "from", "to"].contains(&k.as_str())) {
                        return Err(invalid(format!("{at}: unknown field `{}` (item, from, to)", k.chars().take(40).collect::<String>())));
                    }
                    let item = fo.get("item").and_then(Value::as_u64).ok_or_else(|| invalid(format!("{at}.item: a media item's id")))?;
                    let from = seconds_at(&at, "from", fo.get("from"))?;
                    let to = seconds_at(&at, "to", fo.get("to"))?;
                    if let Some(b) = to {
                        let a = from.unwrap_or(0.0);
                        if b - a < 0.5 {
                            return Err(invalid(format!("{at}: `to` ({b}) is not at least half a second after `from` ({a})")));
                        }
                        if b - a > MAX_FOOTAGE {
                            return Err(invalid(format!("{at}: at most {MAX_FOOTAGE:.0} s of footage")));
                        }
                    }
                    background.push(Footage { item, from, to });
                }
            }
            Some(Value::Array(list)) => return Err(invalid(format!("outro.background: at most {MAX_BACKGROUND} clips, not {}", list.len()))),
            Some(other) => return Err(invalid(format!("outro.background: a list of {{item, from, to}}, not {}", short(other)))),
        }
        let seconds = match o.get("seconds") {
            None | Some(Value::Null) => DEFAULT_SECONDS,
            Some(x) => match x.as_f64() {
                Some(s) if s.is_finite() && (MIN_SECONDS..=MAX_SECONDS).contains(&s) => s,
                _ => return Err(invalid(format!("outro.seconds: {MIN_SECONDS} to {MAX_SECONDS} (the end screen needs 5 s), not {}", short(x)))),
            },
        };
        let out = Self {
            logo,
            channel: text(o, "channel", MAX_SHORT)?,
            handle: text(o, "handle", MAX_SHORT)?,
            tagline: text(o, "tagline", MAX_LONG)?,
            cta: text(o, "cta", MAX_SHORT)?,
            cta_done: text(o, "ctaDone", MAX_SHORT)?,
            terminal: text(o, "terminal", MAX_LONG)?,
            background,
            seconds,
        };
        if out.logo.is_none() && out.channel.is_none() {
            return Err(invalid("outro: give it a `logo` or a `channel` name (or both)"));
        }
        Ok(Some(out))
    }
}

// ---------------------------------------------------------------------------------------------
// Layout and timing

/// Where the reveal ends and the end screen starts, seconds into an outro `len` long: about
/// 4.5 s in, leaving at least [`END_SCREEN_MIN`] and at most [`END_SCREEN_MAX`] seconds after it
/// (and at least 3 s for the reveal).
pub fn reveal_end(len: f64) -> f64 {
    let len = if len.is_finite() { len.max(MIN_REVEAL + END_SCREEN_MIN) } else { DEFAULT_SECONDS };
    REVEAL.min(len - END_SCREEN_MIN).max(len - END_SCREEN_MAX).max(MIN_REVEAL)
}

/// Where YouTube's end-screen elements go, left clear for the end screen's whole stretch.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct EndScreen {
    /// A video tile, left of centre (16:9).
    pub video: Bounds,
    /// The subscribe button (a circle), right of centre.
    pub subscribe: Bounds,
}

/// The end-screen regions on a `frame` (as on 1920×1080, scaled).
pub fn end_screen(frame: (f64, f64)) -> EndScreen {
    let (sx, sy) = scales(frame);
    let b = |l: f64, t: f64, r: f64, bo: f64| Bounds { left: l * sx, top: t * sy, right: r * sx, bottom: bo * sy };
    EndScreen { video: b(180.0, 380.0, 900.0, 785.0), subscribe: b(1175.0, 407.0, 1525.0, 757.0) }
}

/// Horizontal and vertical scale of a frame against 1920×1080.
fn scales((w, h): (f64, f64)) -> (f64, f64) {
    let ok = |v: f64, d: f64| if v.is_finite() && v > 0.0 { v / d } else { 1.0 };
    (ok(w, 1920.0), ok(h, 1080.0))
}

/// Where the elements go on a 1920×1080 frame (scaled to others).
struct Stage {
    w: f64,
    h: f64,
    /// Sizes against 1080 lines.
    u: f64,
    /// The logo's centre and size in the reveal, and its size in the lockup.
    centre: [f64; 2],
    big: f64,
    small: f64,
    /// The lockup's middle line, the button's centre, the tagline's and terminal line's baselines.
    lockup_y: f64,
    button: [f64; 2],
    tagline_y: f64,
    terminal_y: f64,
}

impl Stage {
    fn new(frame: (f64, f64)) -> Self {
        let (sx, sy) = scales(frame);
        let (w, h) = (1920.0 * sx, 1080.0 * sy);
        let u = look::unit((w, h));
        Self {
            w,
            h,
            u,
            centre: [w / 2.0, 432.0 * sy],
            big: 420.0 * u,
            small: 216.0 * u,
            lockup_y: 184.0 * sy,
            button: [w / 2.0, 822.0 * sy],
            tagline_y: 878.0 * sy,
            terminal_y: 948.0 * sy,
        }
    }
}

/// An element of the outro as built: its clip, when it is on screen (seconds of the timeline)
/// and where it sits once the end screen starts (`None`: gone by then).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Element {
    pub name: &'static str,
    pub clip: u64,
    pub from: f64,
    pub to: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub settled: Option<Bounds>,
}

/// The outro as built ([`build`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct BuiltOutro {
    pub start: f64,
    pub end: f64,
    /// From when to the end YouTube's end screen may show: the regions stay clear.
    pub end_screen: [f64; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regions: Option<EndScreen>,
    /// `footage` (the plan's background clips) or `gradient`.
    pub background: String,
    pub background_clips: Vec<u64>,
    pub elements: Vec<Element>,
    /// Sound effects to place: `(kind, at, seconds, intensity)`.
    #[serde(skip)]
    pub cues: Vec<(&'static str, f64, f64, f64)>,
}

/// The tracks the outro draws on: the picture track (`V2`), and video track indices for the
/// wash over the footage, the logo's badge, the outer burst (tracks the show is done with by then), and
/// eight of its own, bottom to top: the inner burst, rings, the logo, the lockup, the tagline,
/// the terminal line, the button and the cursor (with the last fade to black).
#[derive(Clone, Debug)]
pub struct Tracks {
    pub picture: String,
    pub picture_index: usize,
    pub wash: usize,
    pub glow: usize,
    pub burst: usize,
    pub own: [usize; 8],
}

/// How many tracks of its own the outro needs ([`Tracks::own`]).
pub const OWN_TRACKS: usize = 8;

/// The logo's palette: rainbow halftone dots.
const DOTS: [&str; 9] = ["#ff3b47", "#ff8a1f", "#ffd21f", "#3ddc6b", "#18d4c6", "#2fa8ff", "#5a5cff", "#b04dff", "#ff3fa4"];
/// The logo's lettering: purple into blue.
const LETTERING: (&str, &str) = ("#c04bff", "#3f7dff");
const GLOW: &str = "#7a2cff";
/// The logo's badge: the light it was drawn on.
const BADGE: &str = "#f7f4ff";
const BUTTON: &str = "#e3262f";
const BUTTON_DONE: &str = "#2a2f38";

/// A keyframe: time (seconds of the timeline), value, the interpolation out of it.
type Key = (f64, Value, &'static str);

/// Keyframe `effect`.`param` of `clip` at each of `keys` (on frames of `fps`), with their
/// interpolations.
async fn animate(ex: &dyn Executor, clip: u64, effect: &str, param: &str, keys: &[Key], fps: f64) -> Result<(), AutoCutError> {
    for (at, value, _) in keys {
        run(
            ex,
            "effects.setParam",
            json!({"clip": clip, "effect": effect, "param": param, "value": value, "time": ticks(on_frame(*at, fps)), "keyframe": true}),
        )
        .await?;
    }
    for (at, _, how) in keys.iter().filter(|k| k.2 != "linear") {
        run(ex, "effects.setInterpolation", json!({"clip": clip, "effect": effect, "param": param, "time": ticks(on_frame(*at, fps)), "interpolation": how}))
            .await?;
    }
    Ok(())
}

/// Motion's anchor and position of `clip` at `p` (sequence pixels): it then scales about `p`.
async fn pivot(ex: &dyn Executor, clip: u64, p: [f64; 2]) -> Result<(), AutoCutError> {
    run(ex, "effects.setParam", json!({"clip": clip, "effect": "motion", "param": "anchor", "value": p})).await?;
    run(ex, "effects.setParam", json!({"clip": clip, "effect": "motion", "param": "position", "value": p})).await?;
    Ok(())
}

/// A ring of halftone dots around `c` (radius `r`, `n` dots, largest `dot` across) added to
/// graphic `clip`, or a new graphic clip when `clip` is `None`: the dots swell and shrink around
/// the ring like the logo's, in its rainbow.
#[allow(clippy::too_many_arguments)]
async fn dot_ring(
    ex: &dyn Executor,
    track: usize,
    fps: f64,
    c: [f64; 2],
    r: f64,
    n: usize,
    dot: f64,
    turn: f64,
    t: f64,
    dur: f64,
) -> Result<u64, AutoCutError> {
    let mut clip: Option<u64> = None;
    for k in 0..n {
        let a = std::f64::consts::TAU * k as f64 / n.max(1) as f64 + turn;
        let d = dot * (0.45 + 0.55 * (3.0 * a).sin().abs());
        let p = [c[0] + r * a.cos(), c[1] + r * a.sin()];
        let color = DOTS.get(k % DOTS.len()).copied().unwrap_or("#ffffff");
        let props = json!({"fill_color": color, "stroke": false});
        match clip {
            None => clip = Some(look::shape_clip(ex, track, fps, "ellipse", p, [d, d], t, dur, props).await?),
            Some(cl) => {
                look::add_shape(ex, cl, "ellipse", p, [d, d], props).await?;
            }
        }
    }
    clip.ok_or_else(|| invalid("a ring of no dots"))
}

/// The installed monospaced family for the terminal line, if any.
async fn monospace(ex: &dyn Executor) -> Option<String> {
    let installed = look::installed_fonts(ex).await?;
    ["Menlo", "SF Mono", "Monaco", "JetBrains Mono", "DejaVu Sans Mono", "Consolas", "Courier New"]
        .iter()
        .find_map(|want| installed.iter().find(|(f, _)| f.eq_ignore_ascii_case(want)).map(|(f, _)| f.clone()))
}

/// Seconds an item's clip runs from `from` (read back from the picture track after placing it
/// whole), or `None`.
async fn placed_span(ex: &dyn Executor, track_index: usize, clip: u64) -> Option<(f64, f64)> {
    let seq = ex.execute("sequence.inspect", json!({})).await.ok()?;
    let items = seq["video"].get(track_index)?["items"].as_array()?;
    let c = items.iter().find(|c| c["clip"].as_u64() == Some(clip))?;
    let tps = filmcraft_time::TICKS_PER_SECOND as f64;
    Some((c["start"].as_f64()? / tps, c["end"].as_f64()? / tps))
}

/// The footage under the outro, from `t` for `len` seconds on the picture track: the plan's
/// clips in turn (round again when they are shorter than the outro), each blurred and darkened.
/// Its clips; empty when none could be placed (each failure noted).
async fn footage(ex: &dyn Executor, tr: &Tracks, fps: f64, o: &Outro, t: f64, len: f64, notes: &mut Vec<String>) -> Vec<u64> {
    let mut clips = Vec::new();
    let end = on_frame(t + len, fps);
    let mut at = on_frame(t, fps);
    let mut bad: Vec<u64> = Vec::new();
    let mut turn = 0usize;
    while at < end - 0.5 / fps && turn < 4 * MAX_BACKGROUND.max(o.background.len()) && !o.background.is_empty() {
        let Some(f) = o.background.get(turn % o.background.len()) else { break };
        turn += 1;
        if bad.contains(&f.item) {
            if bad.len() >= o.background.len() {
                break;
            }
            continue;
        }
        let from = f.from.unwrap_or(0.0);
        let mut p = json!({"item": f.item, "track": tr.picture, "time": ticks(at), "sourceIn": ticks(from), "audio": false});
        if let Some(to) = f.to {
            p["duration"] = json!(ticks((to - from).min(end - at)));
        }
        let clip = match run(ex, "timeline.place", p).await {
            Ok(r) => r["clips"].as_array().and_then(|a| a.first()).and_then(Value::as_u64),
            Err(e) => {
                notes.push(format!("outro: background item {}: {e}", f.item));
                None
            }
        };
        let Some(clip) = clip else {
            bad.push(f.item);
            continue;
        };
        // where it ends: cut where the outro ends
        let Some((_, stop)) = placed_span(ex, tr.picture_index, clip).await else {
            clips.push(clip);
            break;
        };
        if stop > end + 0.5 / fps {
            let delta = ticks(end) - ticks(stop);
            if let Err(e) = run(ex, "timeline.trim", json!({"clip": clip, "edge": "out", "mode": "regular", "delta": delta})).await {
                notes.push(format!("outro: background: {e}"));
            }
        }
        // (a still or a tiny clip is still footage; it just ends sooner)
        clips.push(clip);
        at = on_frame(stop.min(end), fps).max(at + 1.0 / fps);
    }
    for (i, c) in clips.iter().enumerate() {
        let steps = [
            ("effects.apply", json!({"clips": [c], "effect": "gaussian_blur"})),
            ("effects.setParam", json!({"clip": c, "effect": "gaussian_blur", "param": "blurriness", "value": 28.0})),
            ("effects.setParam", json!({"clip": c, "effect": "gaussian_blur", "param": "repeat_edge", "value": true})),
            ("effects.apply", json!({"clips": [c], "effect": "brightness_contrast"})),
            ("effects.setParam", json!({"clip": c, "effect": "brightness_contrast", "param": "brightness", "value": -30.0})),
            ("clip.fillFrame", json!({"clips": [c]})),
        ];
        for (id, p) in steps {
            if let Err(e) = run(ex, id, p).await {
                notes.push(format!("outro: background: {e}"));
            }
        }
        // the first comes out of black, the cuts between them dissolve
        let (effect, frames) = if i == 0 { ("dip_to_black", 12) } else { ("cross_dissolve", 12) };
        run(ex, "sequence.applyVideoTransition", json!({"clip": c, "edge": "in", "effect": effect, "frames": frames})).await.ok();
    }
    clips
}

/// Build the outro from `t` for `len` seconds ([`MIN_SECONDS`] at least) on `tracks` in
/// `theme` on a `frame` at `fps`: what it made. A logo that cannot be imported, or background
/// clips that cannot be placed, are noted and left out (a gradient stands in for the footage).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn build(
    ex: &dyn Executor,
    theme: &Theme,
    tracks: &Tracks,
    fps: f64,
    frame: (f64, f64),
    o: &Outro,
    t: f64,
    len: f64,
    notes: &mut Vec<String>,
) -> Result<BuiltOutro, AutoCutError> {
    let fps = if fps.is_finite() && fps > 0.0 { fps } else { 30.0 };
    let len = if len.is_finite() { len.max(MIN_SECONDS) } else { DEFAULT_SECONDS };
    let st = Stage::new(frame);
    let (w, h, u) = (st.w, st.h, st.u);
    let t = on_frame(t, fps);
    let end = on_frame(t + len, fps);
    let p = on_frame(t + reveal_end(len), fps);
    let f = 1.0 / fps;
    let regions = end_screen(frame);
    let mut out = BuiltOutro { start: t, end, end_screen: [p, end], regions: Some(regions), ..BuiltOutro::default() };
    let [burst_in, rings, logo_track, lockup_track, tagline_track, terminal_track, button_track, cursor_track] = tracks.own;
    let mut elements: Vec<Element> = Vec::new();
    let mut el = |name: &'static str, clip: u64, from: f64, to: f64, settled: Option<Bounds>| elements.push(Element { name, clip, from, to, settled });

    // ---- the background: the plan's footage, darkened; else a near-black gradient — and a wash
    // over it either way
    let clips = footage(ex, tracks, fps, o, t, len, notes).await;
    let full = Bounds { left: 0.0, top: 0.0, right: w, bottom: h };
    let gradient = |a: &str, b: &str| json!({"fill_kind": "Linear Gradient", "gradient_start": a, "gradient_end": b, "gradient_angle": 90.0, "stroke": false});
    if clips.is_empty() {
        if !o.background.is_empty() {
            notes.push("outro: no background clip could be placed; a gradient stands in".into());
        }
        let g = look::shape_clip(ex, tracks.picture_index, fps, "rectangle", full.centre(), [w, h], t, len, gradient("#0d0618", "#02121a")).await?;
        run(ex, "sequence.applyVideoTransition", json!({"clip": g, "edge": "in", "effect": "dip_to_black", "frames": 12})).await.ok();
        out.background = "gradient".into();
        out.background_clips = vec![g];
    } else {
        out.background = "footage".into();
        out.background_clips = clips;
    }
    let wash = look::shape_clip(ex, tracks.wash, fps, "rectangle", full.centre(), [w, h], t, len, gradient("#0b0414d0", "#010b10e0")).await?;
    animate(ex, wash, "opacity", "opacity", &[(t, json!(0.0), "linear"), (t + 0.5, json!(100.0), "linear")], fps).await.ok();

    // ---- the rings and the bursts around the logo
    let c = st.centre;
    let lockup_scale = st.small / st.big * 100.0;
    let ring = |color: &str, width: f64| json!({"fill": false, "stroke": true, "stroke_color": color, "stroke_width": width * u, "stroke_type": "Center"});
    let r1 = look::shape_clip(ex, rings, fps, "ellipse", c, [st.big, st.big], t + 0.45, 1.15, ring(&theme.accent, 6.0)).await?;
    pivot(ex, r1, c).await?;
    animate(ex, r1, "motion", "scale", &[(t + 0.45, json!(85.0), "easeOut"), (t + 1.55, json!(175.0), "linear")], fps).await?;
    animate(ex, r1, "opacity", "opacity", &[(t + 0.45, json!(90.0), "linear"), (t + 1.55, json!(0.0), "linear")], fps).await?;
    el("ring", r1, t + 0.45, t + 1.6, None);
    let outer = dot_ring(ex, tracks.burst, fps, c, 330.0 * u, 30, 22.0 * u, 0.0, t + 0.4, 1.6).await?;
    let inner = dot_ring(ex, burst_in, fps, c, 262.0 * u, 22, 15.0 * u, 0.14, t + 0.5, 1.5).await?;
    for (clip, a, from, to) in [(outer, t + 0.4, 60.0, 138.0), (inner, t + 0.5, 72.0, 124.0)] {
        pivot(ex, clip, c).await?;
        animate(ex, clip, "motion", "scale", &[(a, json!(from), "easeOut"), (a + 1.3, json!(to), "linear")], fps).await?;
        animate(
            ex,
            clip,
            "opacity",
            "opacity",
            &[(a, json!(0.0), "linear"), (a + 0.2, json!(100.0), "linear"), (a + 0.7, json!(100.0), "linear"), (a + 1.45, json!(0.0), "linear")],
            fps,
        )
        .await?;
    }
    el("burst", outer, t + 0.4, t + 2.0, None);
    el("burst", inner, t + 0.5, t + 2.0, None);

    // ---- the logo (placed now: the lockup leaves room for it only when it is there)
    let a = t + 0.25;
    let logo = match &o.logo {
        Some(path) => match place_logo(ex, logo_track, fps, path, a, end, st.big).await {
            Ok(x) => Some(x),
            Err(e) => {
                notes.push(format!("outro: the logo {path}: {e}; left out"));
                None
            }
        },
        None => None,
    };

    // ---- the lockup: the channel name and handle, measured to place the logo beside them
    let gap = 34.0 * u;
    let (name_size, handle_size) = (104.0 * u, 42.0 * u);
    let mut lockup: Option<u64> = None;
    let mut text_w: f64 = 0.0;
    let mut layers: Vec<(usize, f64)> = Vec::new();
    if let Some(name) = &o.channel {
        let mut props = theme.text_props(Role::Title, LETTERING.0, "left");
        props["caps"] = json!("All Caps");
        props["tracking"] = json!(20.0);
        props["fill_kind"] = json!("Linear Gradient");
        props["gradient_start"] = json!(LETTERING.0);
        props["gradient_end"] = json!(LETTERING.1);
        props["gradient_angle"] = json!(0.0);
        let y = st.lockup_y + if o.handle.is_some() { name_size * 0.12 } else { name_size * 0.36 };
        let clip = look::text_clip(ex, lockup_track, fps, name, [w / 2.0, y], name_size, p - 0.35, end - (p - 0.35), props).await?;
        let b = look::measure(ex, clip, Some(0)).await.unwrap_or_else(|| look::estimate(&name.to_uppercase(), name_size, [w / 2.0, y], "left"));
        text_w = text_w.max(b.right - b.left);
        layers.push((0, y));
        lockup = Some(clip);
    }
    if let Some(handle) = &o.handle {
        let mut props = theme.text_props(Role::Number, &theme.accent, "left");
        props["tracking"] = json!(40.0);
        let y = st.lockup_y + if o.channel.is_some() { name_size * 0.12 + handle_size * 1.45 } else { handle_size * 0.36 };
        let (clip, layer) = match lockup {
            Some(c) => (c, look::add_text(ex, c, handle, [w / 2.0, y], handle_size, props).await?),
            None => (look::text_clip(ex, lockup_track, fps, handle, [w / 2.0, y], handle_size, p - 0.35, end - (p - 0.35), props).await?, 0),
        };
        let b = look::measure(ex, clip, Some(layer)).await.unwrap_or_else(|| look::estimate(handle, handle_size, [w / 2.0, y], "left"));
        text_w = text_w.max(b.right - b.left);
        layers.push((layer, y));
        lockup = Some(clip);
    }
    let has_logo_room = logo.is_some();
    let group = if has_logo_room { st.small + if lockup.is_some() { gap + text_w } else { 0.0 } } else { text_w };
    let left = (w - group) / 2.0;
    let logo_at = [left + st.small / 2.0, st.lockup_y];
    let text_x = if has_logo_room { left + st.small + gap } else { left };
    if let Some(clip) = lockup {
        for (layer, y) in &layers {
            look::set(ex, clip, *layer, json!({"position": [text_x, y]})).await?;
        }
        let b = look::measure(ex, clip, None).await.unwrap_or(Bounds {
            left: text_x,
            top: st.lockup_y - name_size,
            right: text_x + text_w,
            bottom: st.lockup_y + name_size,
        });
        // it slides in from the logo's side as the logo lands, fading in
        let (home, away) = ([w / 2.0, h / 2.0], [w / 2.0 - 46.0 * u, h / 2.0]);
        animate(ex, clip, "motion", "position", &[(p - 0.35, json!(away), "easeOut"), (p + 0.2, json!(home), "linear")], fps).await?;
        animate(ex, clip, "opacity", "opacity", &[(p - 0.35, json!(0.0), "linear"), (p + 0.1, json!(100.0), "linear")], fps).await?;
        el("lockup", clip, p - 0.35, end, Some(b));
    }

    // ---- the logo on its badge — a light disc, as it was drawn on, glowing purple (its shadow) —
    // glitching in (held flickers and jitters), scaling up past its size and back, gliding into
    // the lockup
    let glide = p - 0.6;
    if let Some((clip, base)) = logo {
        let jitter = |dx: f64, dy: f64| json!([c[0] + dx * u, c[1] + dy * u]);
        let moves: Vec<Key> = vec![
            (a, jitter(16.0, 0.0), "hold"),
            (a + f, jitter(-12.0, 3.0), "hold"),
            (a + 3.0 * f, jitter(7.0, -4.0), "hold"),
            (a + 4.0 * f, jitter(-4.0, 0.0), "hold"),
            (a + 6.0 * f, json!(c), "linear"),
            (glide, json!(c), "easeOut"),
            (p, json!(logo_at), "linear"),
        ];
        let grow = [(a, 0.72, "easeOut"), (a + 0.45, 1.06, "easeIn"), (a + 0.75, 1.0, "linear"), (glide, 1.0, "easeOut"), (p, lockup_scale / 100.0, "linear")];
        let flicker: Vec<Key> = vec![
            (a, json!(0.0), "hold"),
            (a + f, json!(100.0), "hold"),
            (a + 3.0 * f, json!(12.0), "hold"),
            (a + 4.0 * f, json!(100.0), "hold"),
            (a + 6.0 * f, json!(35.0), "hold"),
            (a + 7.0 * f, json!(100.0), "linear"),
        ];
        let (spread, blur) = (16.0 * u, 64.0 * u);
        let badge = look::shape_clip(
            ex,
            tracks.glow,
            fps,
            "ellipse",
            c,
            [st.big, st.big],
            a,
            end - a,
            json!({"fill_color": BADGE, "stroke": false, "shadow": true, "shadow_color": GLOW, "shadow_opacity": 90.0, "shadow_distance": 0.0, "shadow_size": spread, "shadow_blur": blur}),
        )
        .await?;
        pivot(ex, badge, c).await?;
        for (cl, k) in [(clip, base), (badge, 100.0)] {
            animate(ex, cl, "motion", "position", &moves, fps).await?;
            let scale: Vec<Key> = grow.iter().map(|(at, x, how)| (*at, json!(k * x), *how)).collect();
            animate(ex, cl, "motion", "scale", &scale, fps).await?;
            animate(ex, cl, "opacity", "opacity", &flicker, fps).await?;
        }
        let half = st.small / 2.0;
        let around = |r: f64| Bounds { left: logo_at[0] - r, top: logo_at[1] - r, right: logo_at[0] + r, bottom: logo_at[1] + r };
        el("logo", clip, a, end, Some(around(half)));
        el("badge", badge, a, end, Some(around(half + (spread + blur) * lockup_scale / 100.0)));
        // a ring pulses out as it lands
        let land = look::shape_clip(ex, rings, fps, "ellipse", logo_at, [st.small, st.small], p, 0.9, ring(&theme.accent, 4.0)).await?;
        pivot(ex, land, logo_at).await?;
        animate(ex, land, "motion", "scale", &[(p, json!(100.0), "easeOut"), (p + 0.85, json!(150.0), "linear")], fps).await?;
        animate(ex, land, "opacity", "opacity", &[(p, json!(80.0), "linear"), (p + 0.85, json!(0.0), "linear")], fps).await?;
        el("ring", land, p, p + 0.9, Some(around(st.small * 0.75 + 4.0 * u)));
    }
    let cues = vec![("impact", t + 0.25, 2.5, 0.65), ("swish", (glide - 0.25).max(t), 0.5, 0.45)];

    // ---- the button: pops in under the logo, a cursor clicks it, it turns to `ctaDone`
    let click = t + 2.3;
    if let Some(cta) = &o.cta {
        let done = o.cta_done.as_deref();
        let first_end = if done.is_some() { click + 0.05 } else { p - 0.05 };
        let (b1, bw) = pill(ex, theme, button_track, fps, st.button, u, cta, BUTTON, (t + 1.1, first_end)).await?;
        let pop = t + 1.1;
        let mut scale =
            vec![(pop, json!(55.0), "easeOut"), (pop + 0.25, json!(108.0), "easeIn"), (pop + 0.4, json!(100.0), "linear"), (click, json!(100.0), "linear")];
        scale.push((click + 0.04, json!(92.0), "linear"));
        if done.is_none() {
            scale.push((click + 0.16, json!(100.0), "linear"));
        }
        animate(ex, b1, "motion", "scale", &scale, fps).await?;
        let mut fade = vec![(pop, json!(0.0), "linear"), (pop + 0.15, json!(100.0), "linear")];
        if done.is_none() {
            fade.extend([(p - 0.5, json!(100.0), "linear"), (p - 0.1, json!(0.0), "linear")]);
        }
        animate(ex, b1, "opacity", "opacity", &fade, fps).await?;
        el("button", b1, pop, first_end, None);
        if let Some(done) = done {
            let (b2, _) = pill(ex, theme, button_track, fps, st.button, u, done, BUTTON_DONE, (click + 0.05, p - 0.05)).await?;
            animate(ex, b2, "motion", "scale", &[(click + 0.05, json!(92.0), "easeOut"), (click + 0.2, json!(100.0), "linear")], fps).await?;
            animate(ex, b2, "opacity", "opacity", &[(p - 0.5, json!(100.0), "linear"), (p - 0.1, json!(0.0), "linear")], fps).await?;
            el("button", b2, click + 0.05, p - 0.05, None);
        }
        // the cursor: an arrow gliding in from the lower right, its tip on the button's right half
        let tip = [st.button[0] + bw * 0.28, st.button[1] + 10.0 * u];
        let arrow: Vec<[f64; 2]> =
            [[0.0, 0.0], [0.0, 46.0], [11.0, 35.0], [19.0, 53.0], [27.0, 49.0], [19.0, 32.0], [33.0, 32.0]].iter().map(|q| [q[0] * u, q[1] * u]).collect();
        let r = run(
            ex,
            "graphics.newShape",
            json!({"shape": "path", "position": tip, "points": arrow, "seconds": click + 0.5 - (t + 1.45), "time": ticks(on_frame(t + 1.45, fps)), "track": cursor_track}),
        )
        .await?;
        let cur = r["clip"].as_u64().ok_or_else(|| AutoCutError::Executor("graphics.newShape made no clip".into()))?;
        look::set(ex, cur, 0, json!({"fill_color": "#ffffff", "stroke": true, "stroke_color": "#05070a", "stroke_width": 2.5 * u, "shadow": true, "shadow_opacity": 50.0, "shadow_blur": 8.0}))
            .await?;
        pivot(ex, cur, tip).await?;
        animate(
            ex,
            cur,
            "motion",
            "position",
            &[(t + 1.45, json!([tip[0] + 260.0 * u, tip[1] + 190.0 * u]), "easeOut"), (click - 0.1, json!(tip), "linear")],
            fps,
        )
        .await?;
        animate(
            ex,
            cur,
            "motion",
            "scale",
            &[(click - 0.02, json!(100.0), "linear"), (click + 0.06, json!(82.0), "linear"), (click + 0.18, json!(100.0), "linear")],
            fps,
        )
        .await?;
        animate(
            ex,
            cur,
            "opacity",
            "opacity",
            &[
                (t + 1.45, json!(0.0), "linear"),
                (t + 1.65, json!(100.0), "linear"),
                (click + 0.3, json!(100.0), "linear"),
                (click + 0.45, json!(0.0), "linear"),
            ],
            fps,
        )
        .await?;
        el("cursor", cur, t + 1.45, click + 0.5, None);
        // the click's ripple
        let ripple = look::shape_clip(ex, rings, fps, "ellipse", tip, [40.0 * u, 40.0 * u], click, 0.55, ring("#ffffff", 3.0)).await?;
        pivot(ex, ripple, tip).await?;
        animate(ex, ripple, "motion", "scale", &[(click, json!(100.0), "easeOut"), (click + 0.5, json!(420.0), "linear")], fps).await?;
        animate(ex, ripple, "opacity", "opacity", &[(click, json!(85.0), "linear"), (click + 0.5, json!(0.0), "linear")], fps).await?;
        el("ripple", ripple, click, click + 0.55, None);
    }

    // ---- the tagline and the terminal line along the bottom
    if let Some(tag) = &o.tagline {
        let size = 40.0 * u;
        let props = theme.text_props(Role::Body, &theme.text, "center");
        let clip = look::text_clip(ex, tagline_track, fps, tag, [w / 2.0, st.tagline_y], size, p, end - p, props).await?;
        let b = look::measure(ex, clip, Some(0)).await.unwrap_or_else(|| look::estimate(tag, size, [w / 2.0, st.tagline_y], "center"));
        let b = fit_across(ex, clip, size, b, w * 0.86).await;
        animate(ex, clip, "motion", "position", &[(p, json!([w / 2.0, h / 2.0 + 18.0 * u]), "easeOut"), (p + 0.5, json!([w / 2.0, h / 2.0]), "linear")], fps)
            .await?;
        animate(ex, clip, "opacity", "opacity", &[(p, json!(0.0), "linear"), (p + 0.4, json!(100.0), "linear")], fps).await?;
        el("tagline", clip, p, end, Some(b));
    }
    if let Some(line) = &o.terminal {
        let size = 34.0 * u;
        let mut props = theme.text_props(Role::Number, &theme.accent, "left");
        if let Some(mono) = monospace(ex).await {
            props["font"] = json!(mono);
            props["font_style"] = json!("Regular");
        }
        props["shadow"] = json!(false);
        let at = p + 0.55;
        // drawn whole to measure it, then placed so the whole line is centred, then typed out
        let clip = look::text_clip(ex, terminal_track, fps, line, [w / 2.0, st.terminal_y], size, at, end - at, props).await?;
        let b = look::measure(ex, clip, Some(0)).await.unwrap_or_else(|| look::estimate(line, size, [w / 2.0, st.terminal_y], "left"));
        let x = w / 2.0 - (b.right - b.left) / 2.0;
        look::set(ex, clip, 0, json!({"position": [x, st.terminal_y]})).await?;
        let chars: Vec<char> = line.chars().collect();
        let step = (1.4 / chars.len().max(1) as f64).min(0.07).max(f);
        let mut keys: Vec<Key> = Vec::new();
        for k in 0..=chars.len() {
            let shown: String = chars.iter().take(k).collect();
            keys.push((at + k as f64 * step, json!(format!("{shown}_")), "hold"));
        }
        // then the cursor blinks
        let mut blink = at + chars.len() as f64 * step + 0.45;
        let mut on = false;
        while blink < end - 0.1 && keys.len() < 400 {
            keys.push((blink, json!(if on { format!("{line}_") } else { format!("{line} ") }), "hold"));
            on = !on;
            blink += 0.45;
        }
        keys.dedup_by(|a, b| (on_frame(a.0, fps) - on_frame(b.0, fps)).abs() < 1e-9);
        // (text keyframes hold until the next: no interpolation to set)
        for k in &mut keys {
            k.2 = "linear";
        }
        animate(ex, clip, "graphic_text", "text", &keys, fps).await?;
        let width = b.right - b.left;
        el("terminal", clip, at, end, Some(Bounds { left: x, top: b.top, right: x + width + size * 0.7, bottom: b.bottom }));
    }

    // ---- the last half second fades to black
    let fade_at = end - 0.5;
    let black =
        look::shape_clip(ex, cursor_track, fps, "rectangle", full.centre(), [w, h], fade_at, end - fade_at, json!({"fill_color": "#000000", "stroke": false}))
            .await;
    match black {
        Ok(b) => {
            animate(ex, b, "opacity", "opacity", &[(fade_at, json!(0.0), "linear"), (end - f, json!(100.0), "linear")], fps).await.ok();
            el("fade", b, fade_at, end, None);
        }
        Err(e) => notes.push(format!("outro: the fade to black: {e}")),
    }
    out.elements = elements;
    out.cues = cues;
    Ok(out)
}

/// The subscribe button: `text` in capitals on a rounded `color` pill centred at `centre`, on
/// video track `track` for `(from, to)` seconds, Motion pivoting on its centre; its clip and width.
#[allow(clippy::too_many_arguments)]
async fn pill(
    ex: &dyn Executor,
    theme: &Theme,
    track: usize,
    fps: f64,
    centre: [f64; 2],
    u: f64,
    text: &str,
    color: &str,
    (from, to): (f64, f64),
) -> Result<(u64, f64), AutoCutError> {
    let size = 40.0 * u;
    let mut props = theme.text_props(Role::Title, "#ffffff", "center");
    props["caps"] = json!("All Caps");
    props["tracking"] = json!(60.0);
    props["shadow"] = json!(false);
    let base = [centre[0], centre[1] + size * 0.36];
    let clip = look::text_clip(ex, track, fps, text, base, size, from, to - from, props).await?;
    let b = look::measure(ex, clip, Some(0)).await.unwrap_or_else(|| look::estimate(&text.to_uppercase(), size, base, "center"));
    let (bw, bh) = ((b.right - b.left) + 72.0 * u, 92.0 * u);
    let back = look::add_shape(ex, clip, "rectangle", centre, [bw, bh], json!({"fill_color": color, "stroke": false, "corner_radius": bh / 2.0})).await?;
    run(ex, "graphics.sendToBack", json!({"clip": clip, "layer": back})).await?;
    pivot(ex, clip, centre).await?;
    Ok((clip, bw))
}

/// Shrink text layer 0 of `clip` (set at `size`, measured `b`) to `room` pixels across when it is
/// wider; its bounds as they are now.
async fn fit_across(ex: &dyn Executor, clip: u64, size: f64, b: Bounds, room: f64) -> Bounds {
    let wide = b.right - b.left;
    if !(wide.is_finite() && wide > room && room > 0.0) {
        return b;
    }
    let k = room / wide;
    if look::set(ex, clip, 0, json!({"size": size * k})).await.is_err() {
        return b;
    }
    look::measure(ex, clip, Some(0)).await.unwrap_or(Bounds { left: (b.left + b.right) / 2.0 - room / 2.0, right: (b.left + b.right) / 2.0 + room / 2.0, ..b })
}

/// Import the logo image at `path` and place it on video track `track` from `t0` to `t1`, sized
/// `px` across its longer side (from the fitted scale, so a "scale to frame size" preference
/// changes nothing): its clip and that Motion scale.
async fn place_logo(ex: &dyn Executor, track: usize, fps: f64, path: &str, t0: f64, t1: f64, px: f64) -> Result<(u64, f64), AutoCutError> {
    let r = run(ex, "file.import", json!({"paths": [path]})).await?;
    let item = r["items"].as_array().and_then(|a| a.first()).and_then(Value::as_u64).ok_or_else(|| {
        let why = r["errors"].as_array().and_then(|a| a.first()).map_or_else(|| "nothing imported".to_string(), Value::to_string);
        invalid(why)
    })?;
    let (t0, t1) = (on_frame(t0, fps), on_frame(t1, fps));
    let r = run(ex, "timeline.place", json!({"item": item, "track": format!("V{}", track + 1), "time": ticks(t0), "duration": ticks(t1 - t0), "audio": false}))
        .await?;
    let clip =
        r["clips"].as_array().and_then(|a| a.first()).and_then(Value::as_u64).ok_or_else(|| AutoCutError::Executor("timeline.place placed no logo".into()))?;
    let fit = run(ex, "clip.fitToFrame", json!({"clips": [clip]})).await?;
    let fitted = fit["clips"].as_array().and_then(|a| a.iter().find(|c| c["clip"].as_u64() == Some(clip))).and_then(|c| c["scale"].as_f64());
    // the fitted picture's longer side: the frame's height for a square or tall logo
    let size = ex.execute("media.frame", json!({"item": item, "seconds": 0.0, "maxSide": 64})).await.ok();
    let (iw, ih) = size.and_then(|s| Some((s["width"].as_f64()?, s["height"].as_f64()?))).filter(|(a, b)| *a > 0.0 && *b > 0.0).unwrap_or((1.0, 1.0));
    let seq = ex.execute("sequence.inspect", json!({})).await.ok();
    let (fw, fh) = seq.and_then(|s| Some((s["settings"]["width"].as_f64()?, s["settings"]["height"].as_f64()?))).unwrap_or((1920.0, 1080.0));
    let k = (fw / iw).min(fh / ih);
    let longer = iw.max(ih) * k;
    let base = match fitted {
        Some(s) if s.is_finite() && s > 0.0 && longer > 0.0 => s * px / longer,
        _ => 100.0,
    };
    Ok((clip, base))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outros_come_from_the_plan_and_refuse_nonsense() {
        let o = Outro::from_plan(&json!({
            "logo": " /x/logo.png ", "channel": "H4HACKS", "handle": "@H4hacks", "tagline": "Hacking, gadgets and robots",
            "cta": "SUBSCRIBE", "ctaDone": "SUBSCRIBED", "terminal": "> h4hacks.com",
            "background": [{"item": 4830, "from": 12, "to": 30}, {"item": 4831}], "seconds": 11
        }))
        .unwrap()
        .unwrap();
        assert_eq!(o.logo.as_deref(), Some("/x/logo.png"));
        assert_eq!((o.channel.as_deref(), o.cta_done.as_deref(), o.seconds), (Some("H4HACKS"), Some("SUBSCRIBED"), 11.0));
        assert_eq!(o.background, vec![Footage { item: 4830, from: Some(12.0), to: Some(30.0) }, Footage { item: 4831, from: None, to: None }]);
        assert_eq!(Outro::from_plan(&json!({"channel": "x"})).unwrap().unwrap().seconds, DEFAULT_SECONDS);
        assert_eq!(Outro::from_plan(&Value::Null).unwrap(), None);
        assert_eq!(Outro::from_plan(&json!(false)).unwrap(), None);
        // a script file read as a plan: its `outro` is the spoken line
        assert_eq!(Outro::from_plan(&json!("Thanks for watching.")).unwrap(), None);
        // no English is made up: a field left out is left out
        let ml = Outro::from_plan(&json!({"channel": "എച്ച്4ഹാക്സ്", "tagline": "ഹാക്കിംഗ്"})).unwrap().unwrap();
        assert!(ml.cta.is_none() && ml.terminal.is_none() && ml.handle.is_none());
        for bad in [
            json!(5),
            json!({}),
            json!({"tagline": "no name or logo"}),
            json!({"channel": "x", "colour": "red"}),
            json!({"channel": ""}),
            json!({"channel": "x".repeat(41)}),
            json!({"channel": "two\nlines"}),
            json!({"channel": 5}),
            json!({"logo": 5, "channel": "x"}),
            json!({"logo": "", "channel": "x"}),
            json!({"channel": "x", "seconds": 3}),
            json!({"channel": "x", "seconds": 400}),
            json!({"channel": "x", "seconds": "long"}),
            json!({"channel": "x", "background": [{"item": -1}]}),
            json!({"channel": "x", "background": [{"item": 3, "from": 10, "to": 5}]}),
            json!({"channel": "x", "background": [{"item": 3, "from": -1}]}),
            json!({"channel": "x", "background": [{"item": 3, "to": 1e9}]}),
            json!({"channel": "x", "background": [{"item": 3, "speed": 2}]}),
            json!({"channel": "x", "background": vec![json!({"item": 1}); 9]}),
            json!({"channel": "x", "background": {"item": 3}}),
        ] {
            let r = std::panic::catch_unwind(|| Outro::from_plan(&bad)).expect("no panic");
            let e = r.expect_err(&bad.to_string()).to_string();
            assert!(e.contains("outro"), "{bad}: {e}");
        }
    }

    #[test]
    fn the_end_screen_gets_its_last_five_to_twenty_seconds() {
        for len in [8.0, 10.0, 12.0, 20.0, 30.0] {
            let p = reveal_end(len);
            assert!(p >= MIN_REVEAL - 1e-9, "{len}: {p}");
            assert!(len - p >= END_SCREEN_MIN - 1e-9 && len - p <= END_SCREEN_MAX + 1e-9, "{len}: {p}");
        }
        assert_eq!(reveal_end(12.0), 4.5);
        assert!(reveal_end(f64::NAN).is_finite() && reveal_end(-5.0).is_finite());
        // the regions: inside the frame, apart, a 16:9 tile left of centre and the button right of it
        for frame in [(1920.0, 1080.0), (1280.0, 720.0), (640.0, 360.0), (3840.0, 2160.0)] {
            let r = end_screen(frame);
            let (w, h) = frame;
            for b in [r.video, r.subscribe] {
                assert!(b.left >= 0.0 && b.top >= 0.0 && b.right <= w && b.bottom <= h, "{frame:?}: {b:?}");
            }
            assert!(r.video.right < r.subscribe.left && r.video.right <= w / 2.0 && r.subscribe.left >= w / 2.0);
            let (vw, vh) = (r.video.right - r.video.left, r.video.bottom - r.video.top);
            assert!((vw / vh - 16.0 / 9.0).abs() < 0.01 && vw >= w * 0.32, "{vw}x{vh}");
            let st = Stage::new(frame);
            // the lockup's band above them, the bottom lines below
            assert!(st.lockup_y + st.small / 2.0 + 60.0 * st.u < r.video.top.min(r.subscribe.top), "{frame:?}");
            assert!(st.tagline_y - 40.0 * st.u > r.video.bottom.max(r.subscribe.bottom), "{frame:?}");
            assert!(st.terminal_y + 12.0 * st.u < h * 0.95, "inside the title-safe area");
        }
    }
}
