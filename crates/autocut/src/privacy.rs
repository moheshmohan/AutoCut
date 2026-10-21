//! Privacy for AutoCut's videos: the vision check behind the engine's automatic blur
//! (`clip.autoRedact`), and the showcase's `redact` option that runs it over a built show.
//!
//! - [`OllamaJudge`]: the engine's [`VisionJudge`]. The faces the detectors found are tiled, up to
//!   six at a time, into one numbered grid picture, and the local vision model (Qwen2.5-VL
//!   through Ollama, [`crate::config::Config::analysis_model`]) says for each tile whether it is a
//!   real person, a humanoid robot, a mannequin, a statue, a poster or a face on a screen. A grid
//!   costs about as much as one picture (≈ 5 s for six faces on an M-series Mac, against ≈ 3 s
//!   for each picture alone, measured 2026-10-19), and kept its answers right on GITEX footage
//!   where several pictures in one request did not.
//! - [`Redact`] / [`redact_show`]: the showcase's privacy pass (`--redact faces,plates`, the
//!   plan's `redact`): every footage clip of the show, each shot's robot (its subject box at its
//!   largest moment) excluded.

use std::sync::Arc;

use filmcraft_engine::redact::{Crop, VisionJudge};
use filmcraft_vision::Kind;
use serde::Serialize;
use serde_json::{Value, json};

use crate::showcase::{Entry, PlacedEntry};
use crate::{AutoCutError, Executor};

/// Side of a tile in the grid, pixels.
const TILE: u32 = 224;
/// Tiles in a row.
const COLUMNS: u32 = 3;
/// Most tiles in one grid (one request).
pub const MAX_TILES: usize = 6;
const GAP: u32 = 8;

const PROMPT: &str = "This picture is a grid of {n} numbered tiles. Each tile is cropped around something a face detector found in a video filmed at a technology exhibition. For each tile say whose face it is: 'person' (a real, living human being who is there in person), 'robot' (a humanoid robot or any machine with a face), 'mannequin', 'statue', 'poster' (a printed picture), 'screen' (a face shown on a phone, tablet or display), or 'none' (there is no face). Answer JSON: {\"tiles\": [{\"n\": 1, \"what\": \"...\"}, ...]} with one entry per tile.";

/// 5 × 7 bitmaps of the digits 0–9, one row per byte (bit 4 is the left column). Drawn for
/// FilmCraft (original work).
const DIGITS: [[u8; 7]; 10] = [
    [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
    [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
    [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
    [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
    [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
    [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
    [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
    [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
    [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
    [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
];

/// Draw number `n` (1–9) in white on a black square at `(x, y)`.
fn label(img: &mut image::RgbImage, x: u32, y: u32, n: usize) {
    const SCALE: u32 = 4;
    let side = 7 * SCALE + 2 * 4;
    for dy in 0..side {
        for dx in 0..side {
            if let Some(p) = img.get_pixel_mut_checked(x + dx, y + dy) {
                *p = image::Rgb([0, 0, 0]);
            }
        }
    }
    let Some(bits) = DIGITS.get(n % 10) else { return };
    for (row, b) in bits.iter().enumerate() {
        for col in 0..5u32 {
            if b & (0x10 >> col) == 0 {
                continue;
            }
            for sy in 0..SCALE {
                for sx in 0..SCALE {
                    let (px, py) = (x + 4 + 2 * SCALE / 2 + col * SCALE + sx, y + 4 + row as u32 * SCALE + sy);
                    if let Some(p) = img.get_pixel_mut_checked(px, py) {
                        *p = image::Rgb([255, 255, 255]);
                    }
                }
            }
        }
    }
}

/// Up to [`MAX_TILES`] crops tiled into one numbered grid (rows of three on white), as PNG.
pub fn grid_png(crops: &[Crop]) -> Option<Vec<u8>> {
    let n = crops.len().min(MAX_TILES) as u32;
    if n == 0 {
        return None;
    }
    let cols = n.min(COLUMNS);
    let rows = n.div_ceil(COLUMNS);
    let mut img = image::RgbImage::from_pixel(cols * (TILE + GAP) - GAP, rows * (TILE + GAP) - GAP, image::Rgb([255, 255, 255]));
    for (i, c) in crops.iter().take(MAX_TILES).enumerate() {
        let (r, col) = (i as u32 / COLUMNS, i as u32 % COLUMNS);
        let (x0, y0) = (col * (TILE + GAP), r * (TILE + GAP));
        let Some(tile) = image::RgbaImage::from_raw(c.width, c.height, c.rgba.clone()) else { continue };
        let k = TILE as f32 / c.width.max(c.height).max(1) as f32;
        let tile = image::imageops::resize(
            &tile,
            ((c.width as f32 * k) as u32).clamp(1, TILE),
            ((c.height as f32 * k) as u32).clamp(1, TILE),
            image::imageops::FilterType::Triangle,
        );
        for (x, y, p) in tile.enumerate_pixels() {
            if let Some(o) = img.get_pixel_mut_checked(x0 + x, y0 + y) {
                *o = image::Rgb([p[0], p[1], p[2]]);
            }
        }
        label(&mut img, x0, y0, i + 1);
    }
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).ok()?;
    Some(out.into_inner())
}

const PLATE_PROMPT: &str = "This picture is a grid of {n} numbered tiles. Each tile is cropped from a video filmed at a technology exhibition; a red rectangle marks something a number-plate detector found. For each tile say what is inside the red rectangle: 'plate' (the licence or number plate of a car, motorbike or other vehicle), 'label' (a name, logo or text on a robot or machine), 'sign' (a sign, banner or poster), 'screen' (a display), or 'other'. Answer JSON: {\"tiles\": [{\"n\": 1, \"what\": \"...\"}, ...]} with one entry per tile.";

/// The model's answer for `n` tiles of `kind`: real (`Some(true)`: a person, a vehicle's plate),
/// something else (`Some(false)`: a robot, mannequin, statue, poster or screen; a label or
/// sign), or not said / not clear (`None`: hidden to be safe).
pub fn parse_answer(kind: Kind, v: &Value, n: usize) -> Vec<Option<bool>> {
    let mut out = vec![None; n];
    let list = v.get("tiles").or_else(|| v.get("pictures")).and_then(Value::as_array).cloned().unwrap_or_default();
    for (i, t) in list.iter().enumerate() {
        let k = t.get("n").and_then(Value::as_u64).map_or(i, |k| (k as usize).saturating_sub(1));
        let what = t.get("what").and_then(Value::as_str).unwrap_or_default().trim().to_ascii_lowercase();
        let a = match (kind, what.as_str()) {
            (Kind::Face, "person" | "human" | "people") => Some(true),
            (Kind::Face, "robot" | "humanoid robot" | "mannequin" | "statue" | "poster" | "screen" | "drawing" | "painting" | "photo") => Some(false),
            (Kind::Plate, "plate" | "license plate" | "licence plate" | "number plate") => Some(true),
            (Kind::Plate, "label" | "sign" | "screen" | "logo" | "text" | "poster" | "banner" | "other" | "none") => Some(false),
            _ => None,
        };
        if let Some(slot) = out.get_mut(k) {
            *slot = a;
        }
    }
    out
}

/// The second look at a face the grid called not a person ([`VisionJudge::confirm`]): the crop
/// alone, the face outlined in red. Measured 2026-10-21 on nine faces from the Ameca clips of
/// Every Robot I've Met (five visitors pressed round the robot, four views of its head): the
/// grid, outlined or not, called a visitor half hidden behind the robot's head a robot (and in
/// the build's own grid, a young man beside it); alone and outlined, all nine were right.
const CONFIRM_PROMPT: &str = "This picture is cropped from a video filmed at a technology exhibition. A red rectangle marks something a face detector found; other people, robots or things may be around it. Say what is inside the red rectangle only: 'person' (the face or head of a real, living human being who is there in person), 'robot' (a humanoid robot's face or head, or any machine), 'mannequin', 'statue', 'poster' (a printed picture), 'screen' (a face shown on a phone, tablet or display), or 'none' (no face). Answer JSON: {\"what\": \"...\"}.";

/// The second look's answer ([`CONFIRM_PROMPT`]): as [`parse_answer`] for one face.
pub fn parse_confirm(v: &Value) -> Option<bool> {
    let one = match v.get("what") {
        Some(w) => json!({"tiles": [{"n": 1, "what": w}]}),
        None => v.clone(),
    };
    parse_answer(Kind::Face, &one, 1).into_iter().next().flatten()
}

/// The vision check over a local Ollama.
pub struct OllamaJudge {
    ollama: crate::ollama::Ollama,
}

impl OllamaJudge {
    pub fn new(host: &str, model: &str) -> Self {
        Self { ollama: crate::ollama::Ollama::new(host, model, std::time::Duration::from_secs(180)) }
    }
}

impl VisionJudge for OllamaJudge {
    fn name(&self) -> String {
        format!("ollama:{}", self.ollama.model)
    }

    fn judge(&self, kind: Kind, crops: &[Crop]) -> Result<Vec<Option<bool>>, String> {
        match self.ollama.available() {
            Ok(true) => {}
            Ok(false) => return Err(format!("the vision model {} is not installed in Ollama (ollama pull {})", self.ollama.model, self.ollama.model)),
            Err(e) => return Err(e.to_string()),
        }
        let mut out = Vec::with_capacity(crops.len());
        for part in crops.chunks(MAX_TILES) {
            let png = grid_png(part).ok_or("could not draw the pictures for the vision check")?;
            let prompt = if kind == Kind::Face { PROMPT } else { PLATE_PROMPT }.replace("{n}", &part.len().to_string());
            let v = self.ollama.chat_json("You answer questions about pictures with JSON only.", &prompt, &[png]).map_err(|e| e.to_string())?;
            out.extend(parse_answer(kind, &v, part.len()));
        }
        Ok(out)
    }

    fn confirm(&self, kind: Kind, crop: &Crop) -> Result<Option<bool>, String> {
        if kind != Kind::Face {
            return Ok(self.judge(kind, std::slice::from_ref(crop))?.into_iter().next().flatten());
        }
        let img = image::RgbaImage::from_raw(crop.width, crop.height, crop.rgba.clone()).ok_or("the crop's pixels do not fit its size")?;
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(img).to_rgb8().write_to(&mut png, image::ImageFormat::Png).map_err(|e| e.to_string())?;
        let v = self.ollama.chat_json("You answer questions about pictures with JSON only.", CONFIRM_PROMPT, &[png.into_inner()]).map_err(|e| e.to_string())?;
        Ok(parse_confirm(&v))
    }

    fn find(&self, frame: &Crop, targets: &[String]) -> Result<Vec<(usize, [f64; 4])>, String> {
        if targets.is_empty() {
            return Ok(Vec::new());
        }
        let img = image::RgbaImage::from_raw(frame.width, frame.height, frame.rgba.clone()).ok_or("the frame's pixels do not fit its size")?;
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(img).to_rgb8().write_to(&mut png, image::ImageFormat::Png).map_err(|e| e.to_string())?;
        let v = self.ollama.chat_json(crate::analyze::GROUND_SYSTEM, &find_prompt(targets), &[png.into_inner()]).map_err(|e| e.to_string())?;
        Ok(parse_found(&v, targets, frame.width, frame.height))
    }
}

/// The question that asks the vision model where the user's named targets are.
pub fn find_prompt(targets: &[String]) -> String {
    let listing: Vec<String> = targets.iter().enumerate().map(|(i, t)| format!("{}. {}", i + 1, t.replace(['\n', '\r'], " "))).collect();
    format!(
        "Find these in the image: {}. Wherever one of them appears (on screens, signs, clothing, machines, the floor; large or small), give its box, each place separately; a box covers the whole of it (a logo with its emblem and its words). Answer JSON only: {{\"found\": [{{\"target\": <number from the list>, \"bbox_2d\": [x1, y1, x2, y2]}}, ...]}} in absolute pixels; an empty list when none is visible.",
        listing.join("; ")
    )
}

/// The places in the model's answer for `targets` in a `w` × `h` picture: (target index, box
/// normalised `[x, y, w, h]`). The model's boxes are in the pixels of the frame it resized the
/// picture to ([`crate::analyze::model_frame`]); hostile numbers and empty boxes are dropped.
/// Which target a place is comes from its 1-based `target` number (0 read as the first), else its
/// `label`; a place the model did not name is still kept (as the first target): Qwen2.5-VL
/// sometimes writes the box's corner into `target`, and a place it found is a place to hide.
pub fn parse_found(v: &Value, targets: &[String], w: u32, h: u32) -> Vec<(usize, [f64; 4])> {
    let n = targets.len();
    if n == 0 {
        return Vec::new();
    }
    let (fw, fh) = crate::analyze::model_frame(w, h);
    let mut out = Vec::new();
    for f in v.get("found").and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default().iter().take(64) {
        let by_label = || {
            let l = f.get("label").and_then(Value::as_str)?.trim().to_ascii_lowercase();
            targets.iter().position(|t| {
                let t = t.trim().to_ascii_lowercase();
                !l.is_empty() && (t == l || t.contains(&l) || l.contains(&t))
            })
        };
        let target = match f.get("target").and_then(Value::as_u64) {
            Some(0) => 0,
            Some(k) if (k as usize) <= n => k as usize - 1,
            _ => by_label().unwrap_or(0),
        };
        let Some(b) = f.get("bbox_2d").or_else(|| f.get("box_2d")).and_then(Value::as_array) else { continue };
        let c: Vec<f64> = b.iter().filter_map(Value::as_f64).collect();
        let [x1, y1, x2, y2] = c.as_slice() else { continue };
        if ![x1, y1, x2, y2].iter().all(|v| v.is_finite()) || fw <= 0.0 || fh <= 0.0 {
            continue;
        }
        let (x1, x2) = ((x1 / fw).clamp(0.0, 1.0), (x2 / fw).clamp(0.0, 1.0));
        let (y1, y2) = ((y1 / fh).clamp(0.0, 1.0), (y2 / fh).clamp(0.0, 1.0));
        if x2 - x1 < 0.002 || y2 - y1 < 0.002 {
            continue;
        }
        // a whole screen, the scene, or the robot wearing the logo is not the target
        // ([`filmcraft_engine::redact_targets::plausible_target`]): blurring it hid the shot
        let b = [x1, y1, x2 - x1, y2 - y1];
        if !filmcraft_engine::redact_targets::plausible_target(b, w, h) {
            continue;
        }
        out.push((target, b));
    }
    out
}

/// The vision check hosts install into the engine (`Session::redacts.judge`): AutoCut's analysis
/// model over the Ollama of the `AUTOCUT_*` environment.
pub fn engine_judge() -> Arc<dyn VisionJudge> {
    let cfg = crate::config::Config::from_env();
    Arc::new(OllamaJudge::new(&cfg.ollama_host, &cfg.analysis_model))
}

// ---------------------------------------------------------------------------------------------
// the showcase's privacy pass

/// What a showcase hides ([`crate::showcase::Options::redact`]).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Redact {
    pub faces: bool,
    pub plates: bool,
    /// `"blur"` or `"mosaic"`; none: the command's default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strength: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_face_height: Option<f64>,
    /// Ask the vision model which faces are people (default on).
    pub verify: bool,
    /// Things named in words ("DP World logo"), found by the vision model in every clip.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<String>,
}

impl Default for Redact {
    fn default() -> Self {
        Self { faces: true, plates: true, style: None, strength: None, min_face_height: None, verify: true, targets: Vec::new() }
    }
}

/// Most named targets a plan, an entry or a shot gives, and the longest.
const MAX_TARGETS: usize = 16;
const MAX_TARGET_CHARS: usize = 120;

/// A plan's `targets`: descriptions on one line.
fn targets_of(v: Option<&Value>) -> Result<Vec<String>, AutoCutError> {
    let list = match v {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::String(s)) => vec![Value::String(s.clone())],
        Some(Value::Array(a)) if a.len() <= MAX_TARGETS => a.clone(),
        Some(_) => return Err(invalid(format!("targets: up to {MAX_TARGETS} descriptions (\"DP World logo\")"))),
    };
    list.iter()
        .map(|t| {
            let t = t.as_str().map(str::trim).unwrap_or_default();
            if t.is_empty() || t.chars().count() > MAX_TARGET_CHARS || t.chars().any(char::is_control) {
                Err(invalid(format!("a target is a description on one line, 1 to {MAX_TARGET_CHARS} characters")))
            } else {
                Ok(t.to_string())
            }
        })
        .collect()
}

/// What an entry or a shot of a plan says about privacy (its `redact`): `false` leaves its
/// footage alone; `{"targets": [...]}` (or the list itself) adds things to hide in it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LocalRedact {
    pub skip: bool,
    pub targets: Vec<String>,
}

impl LocalRedact {
    pub fn from_plan(v: Option<&Value>) -> Result<Self, AutoCutError> {
        match v {
            None | Some(Value::Null) | Some(Value::Bool(true)) => Ok(Self::default()),
            Some(Value::Bool(false)) => Ok(Self { skip: true, targets: Vec::new() }),
            Some(Value::Object(o)) => Ok(Self { skip: false, targets: targets_of(o.get("targets"))? }),
            Some(t @ (Value::Array(_) | Value::String(_))) => Ok(Self { skip: false, targets: targets_of(Some(t))? }),
            Some(_) => Err(invalid("an entry's or shot's redact is false, a list of targets or {targets}")),
        }
    }
}

fn invalid(m: impl Into<String>) -> AutoCutError {
    AutoCutError::Invalid(format!("redact: {}", m.into()))
}

fn kinds(list: &[&str]) -> Result<(bool, bool), AutoCutError> {
    let (mut faces, mut plates) = (false, false);
    for k in list.iter().map(|k| k.trim()).filter(|k| !k.is_empty()) {
        match filmcraft_vision::Kind::from_name(k) {
            Some(filmcraft_vision::Kind::Face) => faces = true,
            Some(filmcraft_vision::Kind::Plate) => plates = true,
            Some(filmcraft_vision::Kind::Target) | None => {
                return Err(invalid(format!("\"{}\" is neither faces nor plates", k.chars().take(40).collect::<String>())));
            }
        }
    }
    Ok((faces, plates))
}

impl Redact {
    /// `faces,plates`, `faces`, `plates` (the CLI's `--redact`).
    pub fn parse(s: &str) -> Result<Self, AutoCutError> {
        let (faces, plates) = kinds(&s.split(',').collect::<Vec<_>>())?;
        if !faces && !plates {
            return Err(invalid("name faces, plates or both"));
        }
        Ok(Self { faces, plates, ..Self::default() })
    }

    /// The plan's `redact`: `true`, `"faces,plates"`, `["faces", "plates"]`, or
    /// `{"kinds": …, "style": "blur"|"mosaic", "strength": 1..100, "minFaceHeight": 0..1,
    /// "verify": bool, "targets": ["DP World logo", …]}` (`kinds: []` with targets hides only
    /// them); absent, `null` or `false`: nothing is hidden.
    pub fn from_plan(v: Option<&Value>) -> Result<Option<Self>, AutoCutError> {
        let list = |a: &[Value]| -> Result<(bool, bool), AutoCutError> {
            let names: Vec<&str> = a.iter().map(|x| x.as_str().ok_or_else(|| invalid("kinds are names (faces, plates)"))).collect::<Result<_, _>>()?;
            kinds(&names)
        };
        match v {
            None | Some(Value::Null) | Some(Value::Bool(false)) => Ok(None),
            Some(Value::Bool(true)) => Ok(Some(Self::default())),
            Some(Value::String(s)) => Self::parse(s).map(Some),
            Some(Value::Array(a)) => {
                let (faces, plates) = list(a)?;
                if !faces && !plates {
                    return Err(invalid("name faces, plates or both"));
                }
                Ok(Some(Self { faces, plates, ..Self::default() }))
            }
            Some(Value::Object(o)) => {
                let mut r = Self::default();
                match o.get("kinds") {
                    None | Some(Value::Null) => {}
                    Some(Value::String(s)) => (r.faces, r.plates) = kinds(&s.split(',').collect::<Vec<_>>())?,
                    Some(Value::Array(a)) => (r.faces, r.plates) = list(a)?,
                    Some(_) => return Err(invalid("kinds: \"faces,plates\" or a list")),
                }
                match o.get("style") {
                    None | Some(Value::Null) => {}
                    Some(Value::String(s)) if matches!(s.as_str(), "blur" | "mosaic") => r.style = Some(s.clone()),
                    Some(_) => return Err(invalid("style: \"blur\" or \"mosaic\"")),
                }
                let num = |k: &str, lo: f64, hi: f64| -> Result<Option<f64>, AutoCutError> {
                    match o.get(k) {
                        None | Some(Value::Null) => Ok(None),
                        Some(v) => v
                            .as_f64()
                            .filter(|f| f.is_finite() && (lo..=hi).contains(f))
                            .map(Some)
                            .ok_or_else(|| invalid(format!("{k}: a number from {lo} to {hi}"))),
                    }
                };
                r.strength = num("strength", 1.0, 100.0)?;
                r.min_face_height = num("minFaceHeight", 0.0, 1.0)?;
                match o.get("verify") {
                    None | Some(Value::Null) => {}
                    Some(Value::Bool(b)) => r.verify = *b,
                    Some(_) => return Err(invalid("verify: true or false")),
                }
                r.targets = targets_of(o.get("targets"))?;
                if !r.faces && !r.plates && r.targets.is_empty() {
                    return Err(invalid("nothing to hide: kinds faces and/or plates, or targets"));
                }
                Ok(Some(r))
            }
            Some(_) => Err(invalid("true, \"faces,plates\", a list or an object")),
        }
    }

    /// As a plan's `redact` field ([`Redact::from_plan`] reads it back).
    pub fn to_plan(&self) -> Value {
        let mut v = json!({"kinds": self.kinds_json(), "verify": self.verify});
        if let Some(s) = &self.style {
            v["style"] = json!(s);
        }
        if let Some(s) = self.strength {
            v["strength"] = json!(s);
        }
        if let Some(h) = self.min_face_height {
            v["minFaceHeight"] = json!(h);
        }
        if !self.targets.is_empty() {
            v["targets"] = json!(self.targets);
        }
        v
    }

    fn kinds_json(&self) -> Value {
        let mut k = Vec::new();
        if self.faces {
            k.push("faces");
        }
        if self.plates {
            k.push("plates");
        }
        json!(k)
    }
}

/// A shot's robot as an exclusion box: its subject box at its largest sampled moment (centre
/// and share of the frame; the box taken square in pixels), at that moment.
fn subject_box(shot: &crate::showcase::Shot, clip: u64, aspect: f64) -> Option<Value> {
    let m = shot.largest?;
    if !(m.area.is_finite() && m.area > 0.0 && aspect.is_finite() && aspect > 0.0) {
        return None;
    }
    // w·h = area with w / h = 1 / aspect (a square in pixels on a frame `aspect` wide)
    let w = (m.area / aspect).sqrt().min(1.0);
    let h = (m.area * aspect).sqrt().min(1.0);
    let [cx, cy] = m.center;
    Some(json!({"x": cx - w / 2.0, "y": cy - h / 2.0, "w": w, "h": h, "clip": clip, "seconds": m.seconds.max(0.0)}))
}

/// The clips a showcase hides things in, and what: from the plan's `redact` (`plan`: faces and
/// plates and its targets, in every footage clip) and each entry's and shot's own `redact`
/// (their targets in their clips; `false` leaves them alone). A shot's clip comes with the
/// blurred background copy behind it when it is shown whole (`fit: full`). Returns the
/// `clip.autoRedact` parameters, or `None` when there is nothing to do; problems with an entry's
/// or shot's `redact` go to `notes`.
pub fn show_params(plan: Option<&Redact>, layout: &[PlacedEntry], entries: &[Entry], aspects: &[(u64, f64)], notes: &mut Vec<String>) -> Option<Value> {
    let mut clips: Vec<u64> = Vec::new();
    let mut exclude: Vec<Value> = Vec::new();
    // each named target with the clips it is looked for in
    let mut local: Vec<(String, Vec<u64>)> = Vec::new();
    for p in layout {
        let e = entries.iter().find(|e| e.number == p.number);
        let er = match e.map(|e| LocalRedact::from_plan(e.redact.as_ref())).transpose() {
            Ok(r) => r.unwrap_or_default(),
            Err(err) => {
                notes.push(format!("privacy: entry {}: {err}", p.number));
                LocalRedact::default()
            }
        };
        for placed in &p.shots {
            let shot = e.and_then(|e| e.shots.iter().find(|s| s.item == placed.item));
            let sr = match shot.map(|s| LocalRedact::from_plan(s.redact.as_ref())).transpose() {
                Ok(r) => r.unwrap_or_default(),
                Err(err) => {
                    notes.push(format!("privacy: entry {}, a shot: {err}", p.number));
                    LocalRedact::default()
                }
            };
            if er.skip || sr.skip {
                continue;
            }
            let here: Vec<&String> = er.targets.iter().chain(&sr.targets).collect();
            if plan.is_none() && here.is_empty() {
                continue;
            }
            let ids: Vec<u64> = std::iter::once(placed.clip).chain(placed.background).collect();
            clips.extend(&ids);
            for t in here {
                match local.iter_mut().find(|(x, _)| x == t) {
                    Some((_, c)) => c.extend(&ids),
                    None => local.push((t.clone(), ids.clone())),
                }
            }
            // the robot itself keeps its face (named targets are hidden even inside it)
            if !p.scene
                && let Some(shot) = shot
            {
                let aspect = aspects.iter().find(|(i, _)| *i == placed.item).map_or(16.0 / 9.0, |a| a.1);
                for clip in &ids {
                    exclude.extend(subject_box(shot, *clip, aspect));
                }
            }
        }
    }
    if clips.is_empty() {
        return None;
    }
    let mut targets: Vec<Value> = plan.map(|r| r.targets.iter().map(|t| json!(t)).collect()).unwrap_or_default();
    targets.extend(local.into_iter().map(|(t, c)| json!({"text": t, "clips": c})));
    let mut p = json!({
        "clips": clips,
        "kinds": plan.map_or(json!([]), Redact::kinds_json),
        "exclude": exclude,
        "verify": plan.is_none_or(|r| r.verify),
        "wait": true,
    });
    if !targets.is_empty() {
        p["targets"] = Value::Array(targets);
    }
    if let Some(r) = plan {
        if let Some(s) = &r.style {
            p["style"] = json!(s);
        }
        if let Some(s) = r.strength {
            p["strength"] = json!(s);
        }
        if let Some(h) = r.min_face_height {
            p["minFaceHeight"] = json!(h);
        }
    }
    Some(p)
}

/// What the privacy pass hid in one clip and its copies (a shot's blurred background), from
/// `clip.autoRedact`'s outcome: for the showcase's QA ([`crate::qa`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Hidden {
    pub clip: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub also: Vec<u64>,
    pub name: String,
    pub faces: usize,
    pub plates: usize,
    /// Places of named targets (logos, signs).
    pub named: usize,
    pub masks: usize,
    /// The largest share of the clip's picture one mask covers, that mask, and when (sequence
    /// seconds).
    pub largest_mask: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub largest_mask_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub largest_at: Option<f64>,
    /// The most of the picture hidden at once, and when (sequence seconds).
    pub most_hidden: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub most_hidden_at: Option<f64>,
    /// The robot's boxes in the clip (`[x, y, w, h]`, shares of its picture): where a face may be
    /// the subject's, for the check of the finished video ([`crate::qa`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub robots: Vec<[f64; 4]>,
}

/// Most clips read from an outcome.
const MAX_HIDDEN: usize = 100_000;

/// The clips of a `clip.autoRedact` outcome (its `redact` object), hostile values dropped.
pub fn hidden_from(outcome: &Value) -> Vec<Hidden> {
    let share = |v: &Value| v.as_f64().filter(|x| x.is_finite()).map_or(0.0, |x| x.clamp(0.0, 1.0));
    let when = |v: &Value| v["sequence"].as_f64().filter(|x| x.is_finite() && *x >= 0.0);
    let count = |v: &Value| v.as_u64().map_or(0, |n| usize::try_from(n).unwrap_or(usize::MAX));
    outcome["clips"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .take(MAX_HIDDEN)
        .filter_map(|c| {
            let cov = &c["coverage"];
            Some(Hidden {
                clip: c["clip"].as_u64()?,
                also: c["also"].as_array().map(|a| a.iter().filter_map(Value::as_u64).take(64).collect()).unwrap_or_default(),
                name: c["name"].as_str().unwrap_or_default().chars().take(200).collect(),
                faces: count(&c["faces"]),
                plates: count(&c["plates"]),
                named: count(&c["targets"]),
                masks: count(&c["masks"]),
                largest_mask: share(&cov["largestMask"]),
                largest_mask_name: cov["largestMaskName"].as_str().map(|s| s.chars().take(80).collect()),
                largest_at: when(&cov["largestAt"]),
                most_hidden: share(&cov["mostHidden"]),
                most_hidden_at: when(&cov["mostHiddenAt"]),
                robots: c["robots"]
                    .as_array()
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .take(64)
                    .filter_map(|b| {
                        let v = [b["x"].as_f64()?, b["y"].as_f64()?, b["w"].as_f64()?, b["h"].as_f64()?];
                        (v.iter().all(|x| x.is_finite() && x.abs() <= 4.0) && v[2] > 0.0 && v[3] > 0.0).then_some(v)
                    })
                    .collect(),
            })
        })
        .collect()
}

/// The showcase's privacy pass ([`show_params`]): `clip.autoRedact` over the show's footage,
/// each robot's own box excluded from the face and plate rules; what happened goes into
/// `notes`, and what was hidden where is returned. A copy of a clip shows the same frames, so the
/// engine analyses it once with its clip and gives it the same masks.
pub async fn redact_show(
    ex: &dyn Executor,
    plan: Option<&Redact>,
    layout: &[PlacedEntry],
    entries: &[Entry],
    notes: &mut Vec<String>,
    progress: &mut dyn FnMut(&str),
) -> Vec<Hidden> {
    let wanted = plan.is_some() || entries.iter().any(|e| e.redact.is_some() || e.shots.iter().any(|s| s.redact.is_some()));
    if !wanted {
        return Vec::new();
    }
    // each footage item's shape, for the robots' boxes
    let mut aspects: Vec<(u64, f64)> = Vec::new();
    for item in layout.iter().flat_map(|p| p.shots.iter().map(|s| s.item)) {
        if aspects.iter().any(|(i, _)| *i == item) {
            continue;
        }
        let a = match ex.execute("media.frame", json!({"item": item, "seconds": 0.0, "maxSide": 64})).await {
            Ok(v) => v["width"].as_f64().zip(v["height"].as_f64()).map_or(16.0 / 9.0, |(w, h)| w / h.max(1.0)),
            Err(_) => 16.0 / 9.0,
        };
        aspects.push((item, a));
    }
    let Some(p) = show_params(plan, layout, entries, &aspects, notes) else { return Vec::new() };
    let n = p["clips"].as_array().map_or(0, Vec::len);
    let named = p["targets"].as_array().map_or(0, Vec::len);
    progress(&format!("privacy: looking for faces, plates{} in {n} clip(s)", if named > 0 { format!(" and {named} named target(s)") } else { String::new() }));
    match crate::showcase::run(ex, "clip.autoRedact", p).await {
        Ok(v) => {
            let o = &v["redact"];
            let clips = o["clips"].as_array().map(Vec::as_slice).unwrap_or_default();
            let sum = |k: &str| clips.iter().filter_map(|c| c[k].as_u64()).sum::<u64>();
            let skipped = |k: &str| clips.iter().filter_map(|c| c["skipped"][k].as_u64()).sum::<u64>();
            progress(&format!(
                "privacy: {} face(s), {} plate(s) and {} named place(s) hidden in {} clip(s); left alone: {} robot(s) or picture(s), {} small face(s), {} excluded",
                sum("faces"),
                sum("plates"),
                sum("targets"),
                o["clipsChanged"],
                skipped("notAPerson"),
                skipped("tooSmall"),
                skipped("excluded")
            ));
            let cache = &o["cache"];
            if cache["enabled"].as_bool() == Some(true) {
                progress(&format!(
                    "privacy: {} frame(s) and {} check(s) from the redaction cache, {} frame(s) decoded, {:.0} s",
                    cache["framesFromCache"],
                    cache["verdictsFromCache"],
                    cache["framesDecoded"],
                    o["seconds"]["analysis"].as_f64().unwrap_or(0.0) + o["seconds"]["check"].as_f64().unwrap_or(0.0)
                ));
            }
            match o["verify"]["state"].as_str() {
                Some("skipped") | Some("failed") => {
                    notes.push(format!("privacy: the robot check did not run ({}); every face found is hidden", o["verify"]["reason"].as_str().unwrap_or("")))
                }
                _ => {}
            }
            for c in clips {
                for e in c["looks"]["errors"].as_array().map(Vec::as_slice).unwrap_or_default().iter().take(1) {
                    notes.push(format!("privacy: {}: looking for named targets: {}", c["name"].as_str().unwrap_or(""), e.as_str().unwrap_or("")));
                }
            }
            hidden_from(o)
        }
        Err(e) => {
            notes.push(format!("privacy: nothing was hidden: {e}"));
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answers_map_to_people_and_things() {
        let v = json!({"tiles": [{"n": 1, "what": "person"}, {"n": 3, "what": "Robot"}, {"n": 2, "what": "screen"}, {"n": 9, "what": "person"}, {"n": 4, "what": "dunno"}]});
        assert_eq!(parse_answer(Kind::Face, &v, 5), vec![Some(true), Some(false), Some(false), None, None]);
        let p = json!({"tiles": [{"n": 1, "what": "plate"}, {"n": 2, "what": "label"}, {"n": 3, "what": "person"}]});
        assert_eq!(parse_answer(Kind::Plate, &p, 3), vec![Some(true), Some(false), None]);
        assert_eq!(parse_answer(Kind::Face, &json!("garbage"), 2), vec![None, None]);
        assert_eq!(parse_answer(Kind::Face, &json!({"tiles": [{"what": "mannequin"}]}), 1), vec![Some(false)]);
        // the second look's answer: one face
        assert_eq!(parse_confirm(&json!({"what": "Person"})), Some(true));
        assert_eq!(parse_confirm(&json!({"what": "robot"})), Some(false));
        assert_eq!(parse_confirm(&json!({"tiles": [{"n": 1, "what": "statue"}]})), Some(false));
        for unclear in [json!({"what": "maybe"}), json!({"what": 3}), json!("person"), json!({})] {
            assert_eq!(parse_confirm(&unclear), None, "{unclear}");
        }
        assert!(CONFIRM_PROMPT.contains("red rectangle"));
    }

    #[test]
    fn found_places_are_read_in_the_models_frame() {
        // a 1280 × 720 picture: Qwen2.5-VL sees it as 1288 × 728
        assert_eq!(crate::analyze::model_frame(1280, 720), (1288.0, 728.0));
        let v = json!({"found": [
            {"target": 1, "bbox_2d": [644, 364, 966, 546]},
            {"target": 2, "bbox_2d": [0, 0, 100, 100]},
            {"target": 0, "bbox_2d": [10, 10, 20, 20]},
            {"target": 9, "bbox_2d": [0, 0, 100, 100]},
            {"target": 1, "bbox_2d": [500, 500, 400, 400]},
            {"target": 1, "bbox_2d": "x"},
            {"bbox_2d": [0, 0, 100, 100]}
        ]});
        let two = vec!["DP World logo".to_string(), "dpworld.com text".to_string()];
        let f = parse_found(&v, &two, 1280, 720);
        // every well-formed box is kept: the unknown target number and the unnamed place too
        assert_eq!(f.len(), 5, "{f:?}");
        assert_eq!(f[0].0, 0);
        assert!((f[0].1[0] - 0.5).abs() < 1e-9 && (f[0].1[2] - 0.25).abs() < 1e-9);
        assert_eq!(f.iter().map(|x| x.0).collect::<Vec<_>>(), vec![0, 1, 0, 0, 0]);
        // the model's slip seen on GITEX footage: the box's corner in `target`, the label beside
        let slip = json!({"found": [{"target": [1085, 515], "bbox_2d": [1085, 515, 1182, 665], "label": "dpworld.com text"}]});
        assert_eq!(parse_found(&slip, &two, 1280, 720)[0].0, 1);
        let one = vec!["the logo".to_string()];
        assert_eq!(parse_found(&json!({"found": [{"bbox_2d": [0, 0, 100, 100]}]}), &one, 1280, 720).len(), 1);
        // a box over a quarter of the picture (the whole tablet in a close-up) is not a logo
        let screen = json!({"found": [{"target": 1, "bbox_2d": [464, 184, 1176, 640]}, {"target": 1, "bbox_2d": [0, 0, 1288, 728]}]});
        assert!(parse_found(&screen, &one, 1280, 720).is_empty());
        // nor a tall box: the robot that carries the logo (the Promobot was blurred whole)
        let robot = json!({"found": [{"target": 1, "bbox_2d": [600, 100, 700, 600]}]});
        assert!(parse_found(&robot, &one, 1280, 720).is_empty());
        assert!(parse_found(&json!({"found": "none"}), &one, 1280, 720).is_empty());
        assert!(parse_found(&json!({}), &one, 0, 0).is_empty());
        assert!(parse_found(&slip, &[], 1280, 720).is_empty());
        assert!(find_prompt(&["DP World logo".into(), "a\nsign".into()]).contains("1. DP World logo; 2. a sign"));
    }

    #[test]
    fn grids_hold_up_to_six_numbered_tiles() {
        let crop = |w: u32, h: u32| Crop { rgba: vec![128; (w * h * 4) as usize], width: w, height: h };
        let png = grid_png(&[crop(100, 120), crop(224, 224), crop(30, 10), crop(1, 1)]).unwrap();
        let img = image::load_from_memory(&png).unwrap();
        assert_eq!((img.width(), img.height()), (3 * TILE + 2 * GAP, 2 * TILE + GAP));
        // the label of tile 1 is drawn (white on black)
        let rgb = img.to_rgb8();
        assert_eq!(rgb.get_pixel(1, 1).0, [0, 0, 0]);
        assert!(rgb.pixels().any(|p| p.0 == [255, 255, 255]));
        assert!(grid_png(&[]).is_none());
        // a crop whose buffer is the wrong size is left blank, never a panic
        assert!(grid_png(&[Crop { rgba: vec![1, 2, 3], width: 50, height: 50 }]).is_some());
    }

    #[test]
    fn plans_and_flags_say_what_to_hide() {
        assert_eq!(Redact::from_plan(None).unwrap(), None);
        assert_eq!(Redact::from_plan(Some(&json!(false))).unwrap(), None);
        assert_eq!(Redact::from_plan(Some(&json!(true))).unwrap(), Some(Redact::default()));
        let r = Redact::from_plan(Some(&json!("faces"))).unwrap().unwrap();
        assert!(r.faces && !r.plates);
        let r = Redact::from_plan(Some(&json!(["plates"]))).unwrap().unwrap();
        assert!(!r.faces && r.plates);
        let r = Redact::from_plan(Some(&json!({"kinds": "faces,plates", "style": "mosaic", "strength": 80, "verify": false}))).unwrap().unwrap();
        assert_eq!((r.style.as_deref(), r.strength, r.verify), (Some("mosaic"), Some(80.0), false));
        for bad in [json!("elephants"), json!([]), json!(3), json!({"style": "smear"}), json!({"strength": 1e9}), json!({"kinds": 4}), json!([1])] {
            assert!(Redact::from_plan(Some(&bad)).is_err(), "{bad}");
        }
        assert!(Redact::parse("faces, plates").unwrap().plates);
        assert!(Redact::parse("").is_err());
        // a plan written out reads back the same
        let r = Redact { faces: false, style: Some("mosaic".into()), strength: Some(70.0), min_face_height: Some(0.1), verify: false, ..Redact::default() };
        assert_eq!(Redact::from_plan(Some(&r.to_plan())).unwrap(), Some(r));
    }

    #[test]
    fn entries_and_shots_add_their_own_targets() {
        use crate::showcase::{PlacedShot, Shot};
        let shot = |item: u64, redact: Option<Value>| Shot { item, redact, ..Shot::default() };
        let entries = vec![
            Entry {
                number: 1,
                shots: vec![shot(70, Some(json!({"targets": ["dpworld.com text"]}))), shot(68, None)],
                redact: Some(json!({"targets": ["DP World logo"]})),
                ..Entry::default()
            },
            Entry { number: 2, shots: vec![shot(5, None)], ..Entry::default() },
            Entry { number: 3, shots: vec![shot(6, None)], redact: Some(json!(false)), ..Entry::default() },
        ];
        let placed = |n: usize, shots: Vec<(u64, u64, Option<u64>)>| PlacedEntry {
            number: n,
            shots: shots.into_iter().map(|(clip, item, background)| PlacedShot { clip, item, background, ..PlacedShot::default() }).collect(),
            ..PlacedEntry::default()
        };
        let layout = vec![placed(1, vec![(100, 70, Some(101)), (102, 68, None)]), placed(2, vec![(103, 5, None)]), placed(3, vec![(104, 6, None)])];
        let mut notes = Vec::new();
        // no plan-level redaction: only the clips with targets, faces and plates off
        let p = show_params(None, &layout, &entries, &[], &mut notes).unwrap();
        assert_eq!(p["clips"], json!([100, 101, 102]));
        assert_eq!(p["kinds"], json!([]));
        let t = p["targets"].as_array().unwrap();
        assert_eq!(t[0], json!({"text": "DP World logo", "clips": [100, 101, 102]}));
        assert_eq!(t[1], json!({"text": "dpworld.com text", "clips": [100, 101]}));
        // with the plan's: every clip but entry 3's (false), its targets everywhere
        let plan = Redact { targets: vec!["Acme banner".into()], ..Redact::default() };
        let p = show_params(Some(&plan), &layout, &entries, &[], &mut notes).unwrap();
        assert_eq!(p["clips"], json!([100, 101, 102, 103]));
        assert_eq!(p["targets"][0], json!("Acme banner"));
        assert_eq!(p["kinds"], json!(["faces", "plates"]));
        assert!(notes.is_empty(), "{notes:?}");
        // a bad entry redact is noted, not fatal
        let bad = vec![Entry { number: 1, shots: vec![shot(70, None)], redact: Some(json!(7)), ..Entry::default() }];
        assert!(show_params(Some(&plan), &layout[..1], &bad, &[], &mut notes).is_some());
        assert_eq!(notes.len(), 1);
        // nothing asked: nothing to do
        assert!(show_params(None, &layout[1..], &entries, &[], &mut notes).is_none());
    }

    #[test]
    fn what_was_hidden_is_read_from_the_outcome() {
        let o = json!({"clips": [
            {"clip": 7, "also": [9], "name": "a.mov", "faces": 2, "plates": 0, "targets": 1, "masks": 3,
             "coverage": {"largestMask": 0.34, "largestMaskName": "Redact logo 1", "largestAt": {"media": 3.0, "sequence": 12.5},
                          "mostHidden": 0.4, "mostHiddenAt": {"media": 3.0, "sequence": 12.5}}},
            {"clip": 8, "faces": -1, "coverage": {"largestMask": 7.0, "mostHidden": "x", "largestAt": {"sequence": -3}},
             "robots": [{"x": 0.1, "y": 0.0, "w": 0.5, "h": 1.0, "seconds": 2.0}, {"x": "a"}, {"x": 0.0, "y": 0.0, "w": -1.0, "h": 1.0}, {"x": 1e300, "y": 0.0, "w": 1.0, "h": 1.0}]},
            {"name": "no id"}
        ]});
        let h = hidden_from(&o);
        assert_eq!(h.len(), 2);
        assert_eq!((h[0].clip, &h[0].also, h[0].faces, h[0].named, h[0].masks), (7, &vec![9], 2, 1, 3));
        assert_eq!((h[0].largest_mask, h[0].largest_mask_name.as_deref(), h[0].largest_at), (0.34, Some("Redact logo 1"), Some(12.5)));
        // hostile numbers: shares clamped, counts and times dropped
        assert_eq!((h[1].largest_mask, h[1].most_hidden, h[1].faces, h[1].largest_at), (1.0, 0.0, 0, None));
        // the robot's boxes: hostile ones dropped
        assert_eq!(h[1].robots, vec![[0.1, 0.0, 0.5, 1.0]]);
        assert!(h[0].robots.is_empty());
        assert!(hidden_from(&json!("nothing")).is_empty());
    }

    #[test]
    fn a_robot_box_is_square_in_pixels() {
        let shot =
            crate::showcase::Shot { item: 1, largest: Some(crate::showcase::Moment { seconds: 2.5, area: 0.09, center: [0.5, 0.4] }), ..Default::default() };
        let b = subject_box(&shot, 7, 16.0 / 9.0).unwrap();
        let (w, h) = (b["w"].as_f64().unwrap(), b["h"].as_f64().unwrap());
        assert!((w * h - 0.09).abs() < 1e-9);
        assert!((w * 16.0 - h * 9.0).abs() < 1e-9, "square on a 16:9 frame");
        assert_eq!((b["clip"].as_u64(), b["seconds"].as_f64()), (Some(7), Some(2.5)));
        let none = crate::showcase::Shot { largest: None, ..shot };
        assert!(subject_box(&none, 7, 1.7).is_none());
    }
}
