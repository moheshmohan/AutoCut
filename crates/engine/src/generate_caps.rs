//! What a generation tool can do: the `capabilities` block of its entry in the provider list
//! ([`crate::generate_run`]). It names the sizes, shapes, lengths, frame counts and rates the tool
//! makes and the inputs it takes, so `generate.*` refuses a request the tool cannot do before
//! anything runs, saying what it can do (and the nearest value it allows), and picks an allowed
//! value where the request left one open (a size from the sequence's shape, a gap's length).
//!
//! Every field is optional. A tool without the block, or a field it leaves out, is not checked
//! there: that is how every tool behaved before the block existed.
//!
//! ```json
//! "capabilities": {
//!   "width": {"min": 256, "max": 2048, "step": 64}, "height": {"min": 256, "max": 2048, "step": 64},
//!   "aspects": ["16:9", "9:16", "1:1"],
//!   "seconds": {"min": 1, "max": 10}, "frames": {"min": 9, "max": 257, "step": 8, "offset": 1}, "fps": [24],
//!   "firstFrame": true, "lastFrame": false, "references": 4,
//!   "transparent": false, "negative": true, "steps": {"min": 1, "max": 50}, "audio": false
//! }
//! ```
//!
//! The block is hostile input like the rest of the list: counts are bounded, steps are never
//! zero, ranges never cross, and every size and frame grid named must hold at least one value.

use serde::{Deserialize, Serialize};

use crate::generate_run::Kind;

/// The largest side a size may name.
const MAX_SIDE: u32 = 16_384;
const MAX_SIZES: usize = 64;
const MAX_ASPECTS: usize = 32;
const MAX_RATES: usize = 16;
/// The most frames a count may name.
const MAX_FRAMES: u32 = 100_000;
/// The most reference pictures a tool may take (as `generate_run::MAX_REFERENCES`).
const MAX_REFS: u32 = 16;
/// The longest a tool may say it makes (a day).
const MAX_SECONDS: f64 = 86_400.0;
/// How far a size's shape may be from a listed one: 2 % (on a grid of 64, 16:9 is rarely exact).
const ASPECT_TOLERANCE: f64 = 0.02;
/// The most sides one search for the nearest allowed size looks at.
const MAX_CANDIDATES: u32 = 20_000;

/// What a tool can do. See the module documentation.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Capabilities {
    /// The sizes it makes, each `[width, height]` (instead of `width` and `height`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sizes: Option<Vec<[u32; 2]>>,
    /// The widths it makes: multiples of `step` from `min` to `max`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<Grid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<Grid>,
    /// The shapes it makes, `"W:H"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspects: Option<Vec<String>>,
    /// The lengths it makes (a clip, music, a sound effect).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seconds: Option<Span>,
    /// The frame counts a video tool makes: `offset` plus a multiple of `step`, `min` to `max`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<Counts>,
    /// The rates a video tool makes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<Vec<f64>>,
    /// Whether a clip can start on a given picture.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_frame: Option<bool>,
    /// Whether a clip can end on a given picture.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_frame: Option<bool>,
    /// The most reference pictures a picture tool takes (0: none).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub references: Option<u32>,
    /// Whether it makes transparent pictures (itself or with its cut-out program).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transparent: Option<bool>,
    /// Whether it takes a negative prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative: Option<bool>,
    /// The sampler steps it takes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<Grid>,
    /// Whether a clip comes with its own sound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<bool>,
}

/// Whole numbers from `min` to `max` that are multiples of `step`.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Grid {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<u32>,
}

/// Seconds from `min` to `max`.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Span {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
}

/// Frame counts `offset + k·step` from `min` to `max`.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Counts {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<u32>,
}

impl Grid {
    fn step(&self) -> u32 {
        self.step.unwrap_or(1).max(1)
    }
    fn lo(&self, default_hi: u32) -> u32 {
        self.min.unwrap_or(1).min(self.hi(default_hi))
    }
    fn hi(&self, default_hi: u32) -> u32 {
        self.max.unwrap_or(default_hi)
    }
    pub fn allows(&self, v: u32, default_hi: u32) -> bool {
        v >= self.lo(default_hi) && v <= self.hi(default_hi) && v.is_multiple_of(self.step())
    }
    /// The first and last allowed value (`None`: none is).
    fn ends(&self, default_hi: u32) -> Option<(u32, u32)> {
        let s = self.step();
        let first = self.lo(default_hi).div_ceil(s).checked_mul(s)?;
        let last = (self.hi(default_hi) / s).checked_mul(s)?;
        (first <= last && first > 0).then_some((first, last))
    }
    /// The allowed value nearest `v`.
    pub fn nearest(&self, v: u32, default_hi: u32) -> Option<u32> {
        let (first, last) = self.ends(default_hi)?;
        let s = self.step();
        let below = (v.clamp(first, last) / s).saturating_mul(s).max(first);
        let above = below.saturating_add(s).min(last);
        Some(if v.abs_diff(above) < v.abs_diff(below) { above } else { below })
    }
    fn describe(&self, unit: &str, default_hi: u32) -> String {
        let (lo, hi) = (self.lo(default_hi), self.hi(default_hi));
        let range = if lo == hi { format!("{lo}{unit}") } else { format!("{lo} to {hi}{unit}") };
        match self.step() {
            1 => range,
            s => format!("{range} in steps of {s}"),
        }
    }
}

impl Span {
    pub fn lo(&self) -> f64 {
        self.min.unwrap_or(0.0)
    }
    pub fn hi(&self) -> f64 {
        self.max.unwrap_or(MAX_SECONDS)
    }
}

impl Counts {
    pub fn step(&self) -> u32 {
        self.step.unwrap_or(1).max(1)
    }
    pub fn offset(&self) -> u32 {
        self.offset.unwrap_or(0) % self.step()
    }
    fn lo(&self) -> u32 {
        self.min.unwrap_or(1)
    }
    fn hi(&self) -> u32 {
        self.max.unwrap_or(MAX_FRAMES)
    }
    pub fn allows(&self, n: u32) -> bool {
        n >= self.lo() && n <= self.hi() && n % self.step() == self.offset()
    }
    /// The fewest and most frames it makes (`None`: no count is allowed).
    pub fn ends(&self) -> Option<(u32, u32)> {
        let (s, off, lo, hi) = (self.step(), self.offset(), self.lo(), self.hi());
        // the first `off + k·s` at or above `lo`, the last at or below `hi`
        let first = off.checked_add(lo.saturating_sub(off).div_ceil(s).checked_mul(s)?)?;
        let last = off.checked_add(hi.checked_sub(off)? / s * s)?;
        (first <= last && first >= 1).then_some((first, last))
    }
    /// `n` moved onto the counts: the nearest allowed one at or above it, else the most.
    pub fn fit(&self, n: u32) -> u32 {
        let Some((first, last)) = self.ends() else { return n };
        if n <= first {
            return first;
        }
        if n >= last {
            return last;
        }
        let (s, off) = (self.step(), self.offset());
        let k = n.saturating_sub(off).div_ceil(s);
        off.saturating_add(k.saturating_mul(s)).min(last)
    }
    fn describe(&self) -> String {
        let (lo, hi) = self.ends().unwrap_or((self.lo(), self.hi()));
        let grid = match (self.step(), self.offset()) {
            (1, _) => String::new(),
            (s, 0) => format!(" (multiples of {s})"),
            (s, o) => format!(" ({s}k + {o})"),
        };
        if lo == hi { format!("{lo} frames") } else { format!("{lo} to {hi} frames{grid}") }
    }
}

/// `"16:9"` as (16, 9).
fn ratio(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.split_once(':')?;
    let (a, b) = (a.trim().parse::<u32>().ok()?, b.trim().parse::<u32>().ok()?);
    ((1..=10_000).contains(&a) && (1..=10_000).contains(&b)).then_some((a, b))
}

/// How far apart two shapes are (0: the same), as the log of their ratio.
fn shape_gap(w: u32, h: u32, aw: u32, ah: u32) -> f64 {
    let r = f64::from(w.max(1)) / f64::from(h.max(1));
    let a = f64::from(aw.max(1)) / f64::from(ah.max(1));
    (r / a).ln().abs()
}

fn seconds_text(x: f64) -> String {
    let s = format!("{x:.2}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    format!("{s} s")
}

/// The fields a kind of tool may state, for the message that refuses another.
fn fields_for(kind: Kind) -> &'static [&'static str] {
    match kind {
        Kind::Image => &["sizes", "width", "height", "aspects", "references", "transparent", "negative", "steps"],
        Kind::Video => &["sizes", "width", "height", "aspects", "seconds", "frames", "fps", "firstFrame", "lastFrame", "audio"],
        Kind::Music | Kind::Sfx => &["seconds"],
        Kind::Composition => &[],
    }
}

impl Capabilities {
    /// The fields this block states, by their names in the list.
    fn stated(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        for (name, on) in [
            ("sizes", self.sizes.is_some()),
            ("width", self.width.is_some()),
            ("height", self.height.is_some()),
            ("aspects", self.aspects.is_some()),
            ("seconds", self.seconds.is_some()),
            ("frames", self.frames.is_some()),
            ("fps", self.fps.is_some()),
            ("firstFrame", self.first_frame.is_some()),
            ("lastFrame", self.last_frame.is_some()),
            ("references", self.references.is_some()),
            ("transparent", self.transparent.is_some()),
            ("negative", self.negative.is_some()),
            ("steps", self.steps.is_some()),
            ("audio", self.audio.is_some()),
        ] {
            if on {
                out.push(name);
            }
        }
        out
    }

    /// The block checked on its own: bounded, no zero steps, no crossing ranges, and every grid
    /// holds a value; and only the fields a tool of `kind` is asked for.
    pub fn validate(&self, kind: Kind) -> Result<(), String> {
        let allowed = fields_for(kind);
        if let Some(f) = self.stated().into_iter().find(|f| !allowed.contains(f)) {
            return Err(format!("`capabilities.{f}` is not something a {} tool is asked for", kind.name()));
        }
        if let Some(sizes) = &self.sizes {
            if sizes.is_empty() || sizes.len() > MAX_SIZES {
                return Err(format!("`capabilities.sizes` lists 1 to {MAX_SIZES} sizes"));
            }
            if sizes.iter().any(|[w, h]| *w == 0 || *h == 0 || *w > MAX_SIDE || *h > MAX_SIDE) {
                return Err(format!("`capabilities.sizes`: each side is 1 to {MAX_SIDE}"));
            }
            if self.width.is_some() || self.height.is_some() {
                return Err("`capabilities` gives `sizes` or `width` and `height`, not both".into());
            }
        }
        for (name, g, most) in [("width", self.width, MAX_SIDE), ("height", self.height, MAX_SIDE), ("steps", self.steps, 10_000)] {
            let Some(g) = g else { continue };
            if g.step == Some(0) || g.step.is_some_and(|s| s > 4096) {
                return Err(format!("`capabilities.{name}.step` is 1 to 4096"));
            }
            if g.min == Some(0) || g.min.is_some_and(|m| m > most) || g.max == Some(0) || g.max.is_some_and(|m| m > most) {
                return Err(format!("`capabilities.{name}` is 1 to {most}"));
            }
            if let (Some(a), Some(b)) = (g.min, g.max)
                && a > b
            {
                return Err(format!("`capabilities.{name}.min` is more than its `max`"));
            }
            if g.ends(most).is_none() {
                return Err(format!("`capabilities.{name}` allows no value"));
            }
        }
        if let Some(list) = &self.aspects {
            if list.is_empty() || list.len() > MAX_ASPECTS {
                return Err(format!("`capabilities.aspects` lists 1 to {MAX_ASPECTS} shapes"));
            }
            if let Some(bad) = list.iter().find(|a| ratio(a).is_none()) {
                return Err(format!("`capabilities.aspects`: `{}` is not a shape such as \"16:9\"", bad.chars().take(40).collect::<String>()));
            }
        }
        if let Some(sp) = self.seconds {
            let ok = |x: Option<f64>| x.is_none_or(|v| v.is_finite() && v > 0.0 && v <= MAX_SECONDS);
            if !ok(sp.min) || !ok(sp.max) || (sp.min.is_none() && sp.max.is_none()) {
                return Err("`capabilities.seconds` gives `min` and/or `max`, each more than 0 and at most a day".into());
            }
            if sp.lo() > sp.hi() {
                return Err("`capabilities.seconds.min` is more than its `max`".into());
            }
        }
        if let Some(c) = self.frames {
            if c.step == Some(0) || c.step.is_some_and(|s| s > 1024) {
                return Err("`capabilities.frames.step` is 1 to 1024".into());
            }
            if c.offset.is_some_and(|o| o >= c.step()) {
                return Err("`capabilities.frames.offset` is less than its `step`".into());
            }
            if c.min == Some(0) || c.min.is_some_and(|m| m > MAX_FRAMES) || c.max == Some(0) || c.max.is_some_and(|m| m > MAX_FRAMES) {
                return Err(format!("`capabilities.frames` is 1 to {MAX_FRAMES}"));
            }
            if c.lo() > c.hi() {
                return Err("`capabilities.frames.min` is more than its `max`".into());
            }
            if c.ends().is_none() {
                return Err("`capabilities.frames` allows no frame count".into());
            }
        }
        if let Some(r) = &self.fps
            && (r.is_empty() || r.len() > MAX_RATES || r.iter().any(|f| !f.is_finite() || *f < 1.0 || *f > 240.0))
        {
            return Err(format!("`capabilities.fps` lists 1 to {MAX_RATES} rates, each 1 to 240"));
        }
        if self.references.is_some_and(|n| n > MAX_REFS) {
            return Err(format!("`capabilities.references` is 0 to {MAX_REFS}"));
        }
        Ok(())
    }

    /// Whether the block says anything about sizes.
    pub fn says_size(&self) -> bool {
        self.sizes.is_some() || self.width.is_some() || self.height.is_some() || self.aspects.is_some()
    }

    fn shapes(&self) -> Vec<(u32, u32)> {
        self.aspects.iter().flatten().filter_map(|a| ratio(a)).collect()
    }

    /// Whether the tool makes `w`×`h`.
    pub fn size_ok(&self, (w, h): (u32, u32)) -> bool {
        if let Some(sizes) = &self.sizes
            && !sizes.contains(&[w, h])
        {
            return false;
        }
        if self.width.is_some_and(|g| !g.allows(w, MAX_SIDE)) || self.height.is_some_and(|g| !g.allows(h, MAX_SIDE)) {
            return false;
        }
        let shapes = self.shapes();
        shapes.is_empty() || shapes.iter().any(|(a, b)| shape_gap(w, h, *a, *b) <= ASPECT_TOLERANCE)
    }

    /// The size the tool makes nearest `target` (its shape first, then its area). `grid` is the
    /// step and largest side for a side the block leaves open (the tool's own `multipleOf` and
    /// `maxSide`).
    pub fn nearest_size(&self, target: (u32, u32), grid: (u32, u32)) -> Option<(u32, u32)> {
        let (tw, th) = (target.0.max(1), target.1.max(1));
        let area = f64::from(tw) * f64::from(th);
        let shapes = self.shapes();
        let aims: Vec<(u32, u32)> = if shapes.is_empty() { vec![(tw, th)] } else { shapes.clone() };
        let score = |w: u32, h: u32| -> f64 {
            // the nearest listed shape counts (else the target's own)
            let aim = aims.iter().map(|(a, b)| shape_gap(w, h, *a, *b)).fold(f64::INFINITY, f64::min);
            let to_target = shape_gap(w, h, tw, th);
            let size_gap = ((f64::from(w) * f64::from(h)) / area).ln().abs();
            aim * 8.0 + to_target * 2.0 + size_gap
        };
        let mut best: Option<(f64, u32, u32)> = None;
        let mut consider = |w: u32, h: u32| {
            if !self.size_ok((w, h)) {
                return;
            }
            let sc = score(w, h);
            if best.is_none_or(|(b, _, _)| sc < b) {
                best = Some((sc, w, h));
            }
        };
        if let Some(sizes) = &self.sizes {
            for [w, h] in sizes {
                consider(*w, *h);
            }
            return best.map(|(_, w, h)| (w, h));
        }
        let open = Grid { min: Some(grid.0.clamp(1, 4096)), max: Some(grid.1.clamp(1, MAX_SIDE)), step: Some(grid.0.clamp(1, 4096)) };
        let wg = self.width.unwrap_or(open);
        let hg = self.height.unwrap_or(open);
        let (w_first, w_last) = wg.ends(MAX_SIDE).or(open.ends(MAX_SIDE))?;
        let ws = wg.step();
        let mut looked = 0u32;
        let mut w = w_first;
        while w <= w_last && looked < MAX_CANDIDATES {
            for (a, b) in &aims {
                // the heights on the grid either side of this width's height in that shape
                let exact = (f64::from(w) * f64::from(*b) / f64::from(*a)).round().clamp(1.0, f64::from(MAX_SIDE)) as u32;
                if let Some(near) = hg.nearest(exact, MAX_SIDE) {
                    consider(w, near);
                    let s = hg.step();
                    if let Some(up) = near.checked_add(s) {
                        consider(w, up);
                    }
                    if let Some(down) = near.checked_sub(s) {
                        consider(w, down);
                    }
                }
            }
            looked = looked.saturating_add(1);
            w = match w.checked_add(ws) {
                Some(n) => n,
                None => break,
            };
        }
        best.map(|(_, w, h)| (w, h))
    }

    /// What sizes the tool makes, in words.
    pub fn describe_sizes(&self) -> String {
        let mut parts = Vec::new();
        if let Some(sizes) = &self.sizes {
            let list: Vec<String> = sizes.iter().take(8).map(|[w, h]| format!("{w}×{h}")).collect();
            let more = if sizes.len() > 8 { format!(" and {} more", sizes.len() - 8) } else { String::new() };
            parts.push(format!("the sizes {}{more}", list.join(", ")));
        }
        if let Some(g) = self.width {
            parts.push(format!("widths {}", g.describe(" px", MAX_SIDE)));
        }
        if let Some(g) = self.height {
            parts.push(format!("heights {}", g.describe(" px", MAX_SIDE)));
        }
        if let Some(a) = &self.aspects {
            parts.push(format!("the shapes {}", a.join(", ")));
        }
        parts.join(", ")
    }

    /// One line on everything the block says (for `generate.providers` and the Generate panel).
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.says_size() {
            parts.push(self.describe_sizes());
        }
        if let Some(sp) = self.seconds {
            parts.push(match (sp.min, sp.max) {
                (Some(a), Some(b)) => format!("{} to {}", seconds_text(a), seconds_text(b)),
                (None, Some(b)) => format!("up to {}", seconds_text(b)),
                (Some(a), None) => format!("at least {}", seconds_text(a)),
                (None, None) => String::new(),
            });
        }
        if let Some(c) = self.frames {
            parts.push(c.describe());
        }
        if let Some(r) = &self.fps {
            let list: Vec<String> = r.iter().map(|f| crate::generate_run::number_text(*f)).collect();
            parts.push(format!("{} fps", list.join(" or ")));
        }
        for (on, yes, no) in [
            (self.first_frame, "starts on a picture", "no start picture"),
            (self.last_frame, "ends on a picture", "no end picture"),
            (self.transparent, "transparent pictures", "no transparency"),
            (self.negative, "a negative prompt", "no negative prompt"),
            (self.audio, "clips with sound", "clips without sound"),
        ] {
            if let Some(v) = on {
                parts.push(if v { yes } else { no }.to_string());
            }
        }
        if let Some(n) = self.references {
            parts.push(match n {
                0 => "no reference pictures".to_string(),
                1 => "1 reference picture".to_string(),
                n => format!("up to {n} reference pictures"),
            });
        }
        if let Some(g) = self.steps {
            parts.push(format!("{} steps", g.describe("", 10_000)));
        }
        parts.retain(|p| !p.is_empty());
        parts.join("; ")
    }
}

/// What a request asks of a tool beyond its size and length.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ask {
    /// A clip that starts on a given picture.
    pub first_frame: bool,
    /// A clip that ends on a given picture.
    pub last_frame: bool,
    /// Reference pictures (a frame taken from the timeline included).
    pub references: usize,
    pub transparent: bool,
    pub negative: bool,
    pub steps: Option<u32>,
    /// A clip with its own sound.
    pub audio: bool,
}

/// Whether `name`'s capabilities allow `ask`; the refusal says what the tool can do.
pub fn check(name: &str, c: &Capabilities, ask: &Ask) -> Result<(), String> {
    let no = |what: &str| Err(format!("{name} cannot {what}"));
    if ask.first_frame && c.first_frame == Some(false) {
        return no("start a clip on a given picture (its capabilities say firstFrame: false)");
    }
    if ask.last_frame && c.last_frame == Some(false) {
        return no("end a clip on a given picture (its capabilities say lastFrame: false)");
    }
    if let Some(most) = c.references
        && ask.references > most as usize
    {
        return Err(match most {
            0 => format!("{name} takes no reference pictures"),
            n => format!("{name} takes at most {n} reference picture{}, and this asks for {}", if n == 1 { "" } else { "s" }, ask.references),
        });
    }
    if ask.transparent && c.transparent == Some(false) {
        return no("make a transparent picture (its capabilities say transparent: false)");
    }
    if ask.negative && c.negative == Some(false) {
        return no("take a negative prompt (its capabilities say negative: false)");
    }
    if let (Some(n), Some(g)) = (ask.steps, c.steps)
        && !g.allows(n, 10_000)
    {
        let near = g.nearest(n, 10_000).map(|v| format!(" (nearest: {v})")).unwrap_or_default();
        return Err(format!("{name} takes {} steps, not {n}{near}", g.describe("", 10_000)));
    }
    if ask.audio && c.audio == Some(false) {
        return no("make a clip with its own sound (its capabilities say audio: false)");
    }
    Ok(())
}

/// The size a request gets: `computed` (from the sides `asked`, the sequence's shape and the
/// tool's own grid) when the tool makes it; else, when the request named no side (or the nearest
/// allowed size keeps the one side it named), the nearest allowed size; else a refusal that says
/// what the tool makes and the nearest size it allows. `grid` is the tool's own step and largest
/// side, for a side the block leaves open.
pub fn fit_size(name: &str, c: &Capabilities, computed: (u32, u32), asked: (Option<u32>, Option<u32>), grid: (u32, u32)) -> Result<(u32, u32), String> {
    if !c.says_size() || c.size_ok(computed) {
        return Ok(computed);
    }
    let near = c.nearest_size(computed, grid);
    let keeps = |n: (u32, u32)| match asked {
        (None, None) => true,
        (Some(_), None) => n.0 == computed.0,
        (None, Some(_)) => n.1 == computed.1,
        (Some(_), Some(_)) => false,
    };
    // the size as the request put it (a side it named, else the one worked out)
    let (w, h) = (asked.0.unwrap_or(computed.0), asked.1.unwrap_or(computed.1));
    match near {
        Some(n) if keeps(n) => Ok(n),
        Some(n) => Err(format!("{name} cannot make {w}×{h}: it makes {} (nearest: {}×{})", c.describe_sizes(), n.0, n.1)),
        None => Err(format!("{name} cannot make {w}×{h}: it makes {}", c.describe_sizes())),
    }
}

/// The lengths a tool makes, seconds, from its `seconds` and (at `fps`) its `frames`; `None`
/// when the block says nothing about length.
pub fn length_range(c: &Capabilities, fps: f64) -> Option<(f64, f64)> {
    if c.seconds.is_none() && c.frames.is_none() {
        return None;
    }
    let (mut lo, mut hi) = c.seconds.map_or((0.0, MAX_SECONDS), |s| (s.lo(), s.hi()));
    if let Some((a, b)) = c.frames.and_then(|f| f.ends())
        && fps.is_finite()
        && fps >= 1.0
    {
        lo = lo.max(f64::from(a) / fps);
        hi = hi.min(f64::from(b) / fps);
    }
    Some((lo.min(hi), hi))
}

/// Whether a wanted length is one the tool makes (half a frame either way at `fps`); the
/// refusal says what it makes and the nearest length it allows.
pub fn check_length(name: &str, c: &Capabilities, want: f64, fps: f64) -> Result<(), String> {
    let Some((lo, hi)) = length_range(c, fps) else { return Ok(()) };
    let slack = if fps.is_finite() && fps >= 1.0 { 0.5 / fps } else { 0.02 };
    if want.is_finite() && want >= lo - slack && want <= hi + slack {
        return Ok(());
    }
    let near = if want.is_finite() { want.clamp(lo, hi) } else { hi };
    let frames = c.frames.map(|f| format!(", {}", f.describe())).unwrap_or_default();
    let rate = if c.frames.is_some() { format!(" at {} fps", crate::generate_run::number_text(fps)) } else { String::new() };
    Err(format!(
        "{name} cannot make {}: it makes {} to {}{frames}{rate} (nearest: {})",
        seconds_text(want),
        seconds_text(lo),
        seconds_text(hi),
        seconds_text(near)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(json: &str) -> Capabilities {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn a_hostile_block_is_refused_with_what_is_wrong() {
        for (kind, bad, why) in [
            (Kind::Image, r#"{"width": {"step": 0}}"#, "step"),
            (Kind::Image, r#"{"width": {"min": 900, "max": 100}}"#, "more than"),
            (Kind::Image, r#"{"width": {"min": 70, "max": 100, "step": 64}}"#, "allows no value"),
            (Kind::Image, r#"{"width": {"max": 99999}}"#, "1 to"),
            (Kind::Image, r#"{"sizes": []}"#, "1 to"),
            (Kind::Image, r#"{"sizes": [[0, 10]]}"#, "each side"),
            (Kind::Image, r#"{"sizes": [[10, 10]], "width": {"max": 10}}"#, "not both"),
            (Kind::Image, r#"{"aspects": ["16x9"]}"#, "not a shape"),
            (Kind::Image, r#"{"aspects": ["0:9"]}"#, "not a shape"),
            (Kind::Image, r#"{"references": 99}"#, "0 to"),
            (Kind::Image, r#"{"seconds": {"max": 4}}"#, "not something"),
            (Kind::Video, r#"{"seconds": {}}"#, "min"),
            (Kind::Video, r#"{"seconds": {"max": -1}}"#, "more than 0"),
            (Kind::Video, r#"{"seconds": {"min": 9, "max": 2}}"#, "more than its"),
            (Kind::Video, r#"{"frames": {"step": 0}}"#, "step"),
            (Kind::Video, r#"{"frames": {"step": 8, "offset": 8}}"#, "offset"),
            (Kind::Video, r#"{"frames": {"min": 2, "max": 5, "step": 8, "offset": 1}}"#, "allows no frame"),
            (Kind::Video, r#"{"frames": {"max": 0}}"#, "1 to"),
            (Kind::Video, r#"{"fps": []}"#, "rates"),
            (Kind::Video, r#"{"fps": [0.5]}"#, "rates"),
            (Kind::Video, r#"{"references": 2}"#, "not something"),
            (Kind::Sfx, r#"{"width": {"max": 512}}"#, "not something"),
            (Kind::Composition, r#"{"negative": true}"#, "not something"),
        ] {
            let e = caps(bad).validate(kind).unwrap_err();
            assert!(e.contains(why), "{bad}: {e}");
        }
        // unknown fields, wrong types and out-of-range numbers do not even parse
        for bad in [r#"{"widths": {}}"#, r#"{"width": {"min": -1}}"#, r#"{"references": 1.5}"#, r#"{"frames": {"max": 1e12}}"#, r#"{"sizes": [[1]]}"#] {
            assert!(serde_json::from_str::<Capabilities>(bad).is_err(), "{bad}");
        }
        assert!(caps("{}").validate(Kind::Video).is_ok());
    }

    #[test]
    fn sizes_are_checked_and_the_nearest_is_named() {
        let c = caps(r#"{"width": {"min": 256, "max": 2048, "step": 64}, "height": {"min": 256, "max": 2048, "step": 64}, "aspects": ["16:9", "1:1"]}"#);
        c.validate(Kind::Image).unwrap();
        assert!(c.size_ok((1024, 576)) && c.size_ok((1024, 1024)));
        assert!(!c.size_ok((1008, 1008)), "off the grid");
        assert!(!c.size_ok((1024, 768)), "a shape it does not make");
        // nothing asked: the nearest allowed size of the sequence's shape
        assert_eq!(fit_size("T", &c, (1008, 560), (None, None), (16, 4096)), Ok((1024, 576)));
        // a size asked that it cannot make: refused with what it can and the nearest
        let e = fit_size("T", &c, (1008, 1008), (Some(1000), Some(1000)), (16, 4096)).unwrap_err();
        assert!(e.contains("cannot make 1000×1000") && e.contains("steps of 64") && e.contains("nearest: 1024×1024"), "{e}");
        // one side asked: the other is fitted when the asked one stays
        assert_eq!(fit_size("T", &c, (1024, 580), (Some(1024), None), (16, 4096)), Ok((1024, 576)));
        assert!(fit_size("T", &c, (1008, 580), (Some(1000), None), (16, 4096)).is_err());
        // fixed sizes
        let fixed = caps(r#"{"sizes": [[1024, 1024], [1344, 768], [768, 1344]]}"#);
        assert_eq!(fixed.nearest_size((1920, 1080), (16, 4096)), Some((1344, 768)));
        assert_eq!(fixed.nearest_size((1080, 1920), (16, 4096)), Some((768, 1344)));
        assert_eq!(fixed.nearest_size((0, 0), (16, 4096)), Some((1024, 1024)));
        assert!(fixed.describe_sizes().contains("1344×768"));
        // only shapes: the tool's own grid for the sides
        let shapes = caps(r#"{"aspects": ["9:16"]}"#);
        let (w, h) = shapes.nearest_size((1024, 576), (32, 2048)).unwrap();
        assert!(w % 32 == 0 && h % 32 == 0 && shapes.size_ok((w, h)) && h > w, "{w}×{h}");
        // hostile targets and grids still give an answer or none, never a panic
        for t in [(0, 0), (u32::MAX, 1), (1, u32::MAX)] {
            let _ = c.nearest_size(t, (0, 0));
            let _ = c.nearest_size(t, (u32::MAX, u32::MAX));
        }
    }

    #[test]
    fn lengths_frame_counts_and_inputs_are_checked() {
        let c = caps(
            r#"{"seconds": {"min": 1, "max": 10}, "frames": {"min": 9, "max": 121, "step": 8, "offset": 1}, "fps": [24], "firstFrame": true, "lastFrame": false, "audio": false}"#,
        );
        c.validate(Kind::Video).unwrap();
        let (lo, hi) = length_range(&c, 24.0).unwrap();
        assert!((lo - 1.0).abs() < 1e-9 && (hi - 121.0 / 24.0).abs() < 1e-9, "{lo} {hi}");
        assert!(check_length("T", &c, 3.0, 24.0).is_ok());
        let e = check_length("T", &c, 8.0, 24.0).unwrap_err();
        assert!(e.contains("cannot make 8 s") && e.contains("121 frames") && e.contains("nearest: 5.04 s"), "{e}");
        assert!(check_length("T", &c, 0.5, 24.0).unwrap_err().contains("nearest: 1 s"));
        assert!(check_length("T", &c, f64::NAN, 24.0).is_err());
        let f = c.frames.unwrap();
        assert_eq!((f.fit(1), f.fit(49), f.fit(50), f.fit(10_000)), (9, 49, 57, 121));
        assert!(f.allows(57) && !f.allows(56) && !f.allows(129));
        assert!(check("T", &c, &Ask { first_frame: true, ..Default::default() }).is_ok());
        assert!(check("T", &c, &Ask { last_frame: true, ..Default::default() }).unwrap_err().contains("end a clip"));
        assert!(check("T", &c, &Ask { audio: true, ..Default::default() }).unwrap_err().contains("own sound"));
        let img = caps(r#"{"references": 2, "transparent": false, "negative": false, "steps": {"min": 1, "max": 50}}"#);
        img.validate(Kind::Image).unwrap();
        assert!(check("T", &img, &Ask { references: 3, ..Default::default() }).unwrap_err().contains("at most 2"));
        assert!(check("T", &img, &Ask { transparent: true, ..Default::default() }).unwrap_err().contains("transparent"));
        assert!(check("T", &img, &Ask { negative: true, ..Default::default() }).unwrap_err().contains("negative"));
        assert!(check("T", &img, &Ask { steps: Some(80), ..Default::default() }).unwrap_err().contains("nearest: 50"));
        assert!(check("T", &img, &Ask { references: 2, steps: Some(20), ..Default::default() }).is_ok());
        // a block without length says nothing about it
        assert!(length_range(&img, 24.0).is_none() && check_length("T", &img, 1.0e9, 24.0).is_ok());
        let s = c.summary();
        assert!(s.contains("1 s to 10 s") && s.contains("9 to 121 frames (8k + 1)") && s.contains("24 fps") && s.contains("no end picture"), "{s}");
    }
}
