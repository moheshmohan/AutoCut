//! Privacy for AutoCut's videos: the vision check behind the engine's automatic blur
//! (`clip.autoRedact`), and the showcase's `redact` option that runs it over a built show.
//!
//! - [`OllamaJudge`]: the engine's [`FaceJudge`]. The faces the detectors found are tiled, up to
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

use filmcraft_engine::redact::{Crop, FaceJudge};
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

/// The vision check over a local Ollama.
pub struct OllamaJudge {
    ollama: crate::ollama::Ollama,
}

impl OllamaJudge {
    pub fn new(host: &str, model: &str) -> Self {
        Self { ollama: crate::ollama::Ollama::new(host, model, std::time::Duration::from_secs(180)) }
    }
}

impl FaceJudge for OllamaJudge {
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
}

/// The vision check hosts install into the engine (`Session::redacts.judge`): AutoCut's analysis
/// model over the Ollama of the `AUTOCUT_*` environment.
pub fn engine_judge() -> Arc<dyn FaceJudge> {
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
}

impl Default for Redact {
    fn default() -> Self {
        Self { faces: true, plates: true, style: None, strength: None, min_face_height: None, verify: true }
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
            None => return Err(invalid(format!("\"{}\" is neither faces nor plates", k.chars().take(40).collect::<String>()))),
        }
    }
    if faces || plates { Ok((faces, plates)) } else { Err(invalid("name faces, plates or both")) }
}

impl Redact {
    /// `faces,plates`, `faces`, `plates` (the CLI's `--redact`).
    pub fn parse(s: &str) -> Result<Self, AutoCutError> {
        let (faces, plates) = kinds(&s.split(',').collect::<Vec<_>>())?;
        Ok(Self { faces, plates, ..Self::default() })
    }

    /// The plan's `redact`: `true`, `"faces,plates"`, `["faces", "plates"]`, or
    /// `{"kinds": …, "style": "blur"|"mosaic", "strength": 1..100, "minFaceHeight": 0..1,
    /// "verify": bool}`; absent, `null` or `false`: nothing is hidden.
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

/// The showcase's privacy pass: `clip.autoRedact` over every footage clip of the built show
/// (`layout`), each robot's own box excluded; what happened goes into `notes`.
pub async fn redact_show(ex: &dyn Executor, r: &Redact, layout: &[PlacedEntry], entries: &[Entry], notes: &mut Vec<String>, progress: &mut dyn FnMut(&str)) {
    let clips: Vec<u64> = layout.iter().flat_map(|p| p.shots.iter().map(|s| s.clip)).collect();
    if clips.is_empty() {
        return;
    }
    let mut exclude = Vec::new();
    for p in layout.iter().filter(|p| !p.scene) {
        let Some(e) = entries.iter().find(|e| e.number == p.number) else { continue };
        for placed in &p.shots {
            let Some(shot) = e.shots.iter().find(|s| s.item == placed.item) else { continue };
            let aspect = match ex.execute("media.frame", json!({"item": placed.item, "seconds": 0.0, "maxSide": 64})).await {
                Ok(v) => v["width"].as_f64().zip(v["height"].as_f64()).map_or(16.0 / 9.0, |(w, h)| w / h.max(1.0)),
                Err(_) => 16.0 / 9.0,
            };
            exclude.extend(subject_box(shot, placed.clip, aspect));
        }
    }
    let mut p = json!({"clips": clips, "kinds": r.kinds_json(), "exclude": exclude, "verify": r.verify, "wait": true});
    if let Some(s) = &r.style {
        p["style"] = json!(s);
    }
    if let Some(s) = r.strength {
        p["strength"] = json!(s);
    }
    if let Some(h) = r.min_face_height {
        p["minFaceHeight"] = json!(h);
    }
    progress(&format!("privacy: looking for faces and plates in {} clip(s)", clips.len()));
    match crate::showcase::run(ex, "clip.autoRedact", p).await {
        Ok(v) => {
            let o = &v["redact"];
            let clips = o["clips"].as_array().map(Vec::as_slice).unwrap_or_default();
            let sum = |k: &str| clips.iter().filter_map(|c| c[k].as_u64()).sum::<u64>();
            let skipped = |k: &str| clips.iter().filter_map(|c| c["skipped"][k].as_u64()).sum::<u64>();
            progress(&format!(
                "privacy: {} face(s) and {} plate(s) hidden in {} clip(s); left alone: {} robot(s) or picture(s), {} small face(s), {} excluded",
                sum("faces"),
                sum("plates"),
                o["clipsChanged"],
                skipped("notAPerson"),
                skipped("tooSmall"),
                skipped("excluded")
            ));
            match o["verify"]["state"].as_str() {
                Some("skipped") | Some("failed") => {
                    notes.push(format!("privacy: the robot check did not run ({}); every face found is hidden", o["verify"]["reason"].as_str().unwrap_or("")))
                }
                _ => {}
            }
        }
        Err(e) => notes.push(format!("privacy: faces and plates were not hidden: {e}")),
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
