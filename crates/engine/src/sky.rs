//! A clip's sky found and swapped (the idea of Insta360's Sky Swap). New skies are the user's own
//! photos or made here — never stock pictures.
//!
//! | command | does |
//! |---|---|
//! | `clip.findSky` | job: the sky of a clip outlined every `every` seconds (0.2): where it can be — bright, smooth, plain or blue, reaching the top of the picture — refined by MobileSAM (given points in it and below it) when it is downloaded; kept as the clip's Opacity mask `Sky`, switched off |
//! | `clip.skySwap` | the new sky above the clip through its `Sky` mask (`feather` 6 px): `sky` an item of the project (a photo: cut to the clip's shape), or one made here — `clear`, `sunset`, `dusk`, `night`, `clouds` (`seed`) — a whole sphere for a 360° clip, seen through its virtual camera; the clip relit a little towards the new sky's colour and light (`relight` 30 %) |
//!
//! **Where the sky is.** A levelled 360° picture and an ordinary one alike have the sky at the
//! top: each analysed frame (a 256-wide copy) is looked at cell by cell for sky — bright, with
//! little detail, grey or blue — and what of it joins the top edge is the sky's first guess. With
//! MobileSAM, points well inside that guess and points in the detail below it go to the segmenter,
//! whose cut is added to the guess when they agree (overlap by at least 40 %): its edges, and the
//! guess's pieces it left out. Every piece of it of at least 0.4 % of the picture (sky between branches, over a roof; eight
//! at most) is outlined, the pieces joined into one outline by slits of no width, 256 points, as a
//! keyframe of the `Sky` mask. Sky seen through the gaps inside a tree's crown stays as it was.

use std::sync::Arc;

use filmcraft_project::{ClipId, ItemId, MaskMode, ParamValue, TrackItem};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, f64_p, has_seq, str_p, u64_p};
use crate::cutout::{FindOutline, Outlining};
use crate::{EngineError, Result, Session};

/// The name of the sky's mask on a clip's Opacity effect.
pub const SKY: &str = "Sky";
const FIND_CMD: &str = "clip.findSky";
const SWAP_CMD: &str = "clip.skySwap";
/// Points of a sky's outline.
const POINTS: usize = 256;
/// The skies made here.
pub const KINDS: [&str; 5] = ["clear", "sunset", "dusk", "night", "clouds"];
/// The width of a sky made for a 360° clip (a whole sphere, 2:1).
const SPHERE_WIDTH: usize = 4096;
/// Width of the copy a frame is looked at for sky on.
const LOOK_WIDTH: usize = 256;
/// The share of a picture the sky's first guess must cover.
const MIN_SKY: f32 = 0.03;
/// Pieces of sky outlined besides the largest: each at least this share of the picture, at most
/// this many in all.
const PIECE_SHARE: f32 = 0.004;
const MAX_PIECES: usize = 8;

/// Finds the sky in pictures: MobileSAM over the first guess, or a stand-in.
pub trait SkyFinder: Send + Sync {
    /// The sky's outline in an RGBA picture (`points` points, its pixels); `None` when there is none.
    fn find(&self, rgba: &[u8], w: usize, h: usize, points: usize) -> std::result::Result<Option<Vec<(f32, f32)>>, String>;
}

pub fn commands() -> Vec<CommandSpec> {
    let spec = |id: &'static str, label: &'static str, params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>| CommandSpec {
        id,
        label,
        menu: &["Clip", "Cut-outs"],
        shortcut: None,
        params,
        enabled: has_seq,
        run,
        journal: true,
    };
    vec![
        spec(FIND_CMD, "Find the Sky", r#"{"clip":id?,"every":seconds=0.2,"points":16..256=256,"wait":bool=false}"#, find_sky),
        spec(
            SWAP_CMD,
            "Sky Swap",
            r#"{"clip":id?,"sky":id|"clear"|"sunset"|"dusk"|"night"|"clouds"="sunset","seed":u64=1,"feather":px=6,"relight":0..100=30,"folder":str?}"#,
            sky_swap,
        ),
    ]
}

// ---------------------------------------------------------------------------------------------
// where the sky is
// ---------------------------------------------------------------------------------------------

/// The sky's first guess in an RGBA picture: on a grid `gw` × `gh` (cells of `cell` pixels), the
/// cells that look like sky and join the top edge.
pub struct Guess {
    pub gw: usize,
    pub gh: usize,
    pub cell: usize,
    pub sky: Vec<bool>,
}

impl Guess {
    pub fn share(&self) -> f32 {
        self.sky.iter().filter(|s| **s).count() as f32 / self.sky.len().max(1) as f32
    }
    fn at(&self, x: i64, y: i64) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.gw && (y as usize) < self.gh && self.sky.get(y as usize * self.gw + x as usize).copied().unwrap_or(false)
    }
}

/// [`Guess`] of a picture.
pub fn guess(rgba: &[u8], w: usize, h: usize) -> Option<Guess> {
    if w < 8 || h < 8 || rgba.len() < w.checked_mul(h)?.checked_mul(4)? {
        return None;
    }
    let cell = (w / LOOK_WIDTH).max(1);
    let (gw, gh) = (w / cell, h / cell);
    // each cell's mean colour
    let mut c = vec![[0f32; 3]; gw * gh];
    for gy in 0..gh {
        for gx in 0..gw {
            let mut acc = [0f32; 3];
            for y in gy * cell..(gy + 1) * cell {
                for x in gx * cell..(gx + 1) * cell {
                    let i = (y * w + x) * 4;
                    if let Some(p) = rgba.get(i..i + 3) {
                        for k in 0..3 {
                            acc[k] += f32::from(p[k]) / 255.0;
                        }
                    }
                }
            }
            let n = (cell * cell) as f32;
            c[gy * gw + gx] = [acc[0] / n, acc[1] / n, acc[2] / n];
        }
    }
    let luma = |p: [f32; 3]| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    let detail = |x: usize, y: usize| -> f32 {
        let here = luma(c[y * gw + x]);
        let mut d = 0f32;
        if x + 1 < gw {
            d = d.max((luma(c[y * gw + x + 1]) - here).abs());
        }
        if y + 1 < gh {
            d = d.max((luma(c[(y + 1) * gw + x]) - here).abs());
        }
        d
    };
    let skyish = |x: usize, y: usize, loose: bool| -> bool {
        let p = c[y * gw + x];
        let (mx, mn) = (p[0].max(p[1]).max(p[2]), p[0].min(p[1]).min(p[2]));
        let sat = if mx > 1e-3 { (mx - mn) / mx } else { 0.0 };
        let blue = p[2] - (p[0] + p[1]) / 2.0;
        let bright = luma(p) >= if loose { 0.35 } else { 0.42 };
        let smooth = detail(x, y) < if loose { 0.07 } else { 0.045 };
        bright && smooth && (sat < 0.25 || blue > 0.03)
    };
    // from the top edge, through what looks like sky (a little looser once inside)
    let mut sky = vec![false; gw * gh];
    let mut stack: Vec<(usize, usize)> = (0..gw).filter(|&x| skyish(x, 0, false)).map(|x| (x, 0)).collect();
    for &(x, y) in &stack {
        sky[y * gw + x] = true;
    }
    while let Some((x, y)) = stack.pop() {
        for (nx, ny) in [(x.wrapping_add(1), y), (x.wrapping_sub(1), y), (x, y + 1), (x, y.wrapping_sub(1))] {
            if nx < gw && ny < gh && !sky[ny * gw + nx] && skyish(nx, ny, true) {
                sky[ny * gw + nx] = true;
                stack.push((nx, ny));
            }
        }
    }
    Some(Guess { gw, gh, cell, sky })
}

/// The guess as a matte of the picture's own size.
fn guess_matte(g: &Guess, w: usize, h: usize) -> filmcraft_vision::matte::Matte {
    let alpha = (0..w * h).map(|i| if g.at(((i % w) / g.cell) as i64, ((i / w) / g.cell) as i64) { 255 } else { 0 }).collect();
    filmcraft_vision::matte::Matte { w, h, alpha, score: 1.0 }
}

/// Points well inside the guess (all eight neighbours sky too) and points in the detail below it,
/// picture pixels: at most `n` of each, spread out.
pub fn prompts(g: &Guess, rgba_detail: impl Fn(usize, usize) -> bool, n: usize) -> (Vec<(f32, f32)>, Vec<(f32, f32)>) {
    let centre = |x: usize, y: usize| ((x as f32 + 0.5) * g.cell as f32, (y as f32 + 0.5) * g.cell as f32);
    let inner: Vec<(usize, usize)> = (0..g.gh)
        .flat_map(|y| (0..g.gw).map(move |x| (x, y)))
        .filter(|&(x, y)| (-1..=1).all(|dy| (-1..=1).all(|dx| g.at(x as i64 + dx, y as i64 + dy))))
        .collect();
    let spread = |v: &[(usize, usize)]| -> Vec<(f32, f32)> {
        if v.is_empty() {
            return Vec::new();
        }
        (0..n.min(v.len())).filter_map(|k| v.get(k * v.len() / n.min(v.len()).max(1))).map(|&(x, y)| centre(x, y)).collect()
    };
    let lowest = (0..g.gh).rev().find(|&y| (0..g.gw).any(|x| g.at(x as i64, y as i64))).unwrap_or(0);
    let below: Vec<(usize, usize)> =
        (lowest.min(g.gh)..g.gh).flat_map(|y| (0..g.gw).map(move |x| (x, y))).filter(|&(x, y)| !g.at(x as i64, y as i64) && rgba_detail(x, y)).collect();
    (spread(&inner), spread(&below))
}

/// The sky's outline in a picture: the guess, refined by `seg` when given and when it agrees.
fn sky_outline(
    rgba: &[u8],
    w: usize,
    h: usize,
    points: usize,
    seg: Option<&dyn Fn(&[u8], usize, usize, &[(f32, f32)], &[(f32, f32)]) -> Option<filmcraft_vision::matte::Matte>>,
) -> Option<Vec<(f32, f32)>> {
    let g = guess(rgba, w, h)?;
    if g.share() < MIN_SKY {
        return None;
    }
    let mut m = guess_matte(&g, w, h);
    if let Some(seg) = seg {
        let not_sky = |x: usize, y: usize| !g.at(x as i64, y as i64);
        let (inside, outside) = prompts(&g, not_sky, 6);
        if let Some(cut) = seg(rgba, w, h, &inside, &outside).filter(|c| c.w == w && c.h == h) {
            let (mut both, mut either) = (0usize, 0usize);
            for (a, b) in cut.alpha.iter().zip(&m.alpha) {
                let (a, b) = (*a >= 128, *b >= 128);
                both += usize::from(a && b);
                either += usize::from(a || b);
            }
            // it agrees: its edges, and what of the guess it left out (pieces cut off by a branch)
            if either > 0 && both as f32 / either as f32 >= 0.4 {
                for (a, b) in m.alpha.iter_mut().zip(&cut.alpha) {
                    *a = (*a).max(*b);
                }
            }
        }
    }
    // every sizeable piece (sky between branches, over a building): one outline round them all
    filmcraft_vision::matte::outline_pieces(&m, points, PIECE_SHARE, MAX_PIECES)
}

/// The finder a job uses: the session's stand-in, else the guess refined by MobileSAM when it is
/// downloaded (and whether it is).
fn finder(s: &Session) -> (Arc<dyn SkyFinder>, bool) {
    if let Some(f) = &s.cutouts.sky {
        return (f.clone(), true);
    }
    models()
}

struct GuessOnly;

impl SkyFinder for GuessOnly {
    fn find(&self, rgba: &[u8], w: usize, h: usize, points: usize) -> std::result::Result<Option<Vec<(f32, f32)>>, String> {
        Ok(sky_outline(rgba, w, h, points, None))
    }
}

#[cfg(feature = "privacy-models")]
struct Refined {
    seg: filmcraft_vision::segment::Segmenter,
}

#[cfg(feature = "privacy-models")]
impl SkyFinder for Refined {
    fn find(&self, rgba: &[u8], w: usize, h: usize, points: usize) -> std::result::Result<Option<Vec<(f32, f32)>>, String> {
        let seg = |rgba: &[u8], w: usize, h: usize, inside: &[(f32, f32)], outside: &[(f32, f32)]| -> Option<filmcraft_vision::matte::Matte> {
            let e = self.seg.embed(rgba, w, h).ok()?;
            let m = self.seg.cut(&e, inside, outside).ok()?;
            // the cut at the picture's size
            let (mw, mh) = (m.w, m.h);
            let alpha =
                (0..w * h).map(|i| m.at(((i % w) * mw / w.max(1)).min(mw.saturating_sub(1)), ((i / w) * mh / h.max(1)).min(mh.saturating_sub(1)))).collect();
            Some(filmcraft_vision::matte::Matte { w, h, alpha, score: m.score })
        };
        Ok(sky_outline(rgba, w, h, points, Some(&seg)))
    }
}

#[cfg(feature = "privacy-models")]
fn models() -> (Arc<dyn SkyFinder>, bool) {
    match crate::transcript::models_dir().and_then(|d| filmcraft_vision::segment::Segmenter::load(&d, true).ok()) {
        Some(seg) => (Arc::new(Refined { seg }), true),
        None => (Arc::new(GuessOnly), false),
    }
}

#[cfg(not(feature = "privacy-models"))]
fn models() -> (Arc<dyn SkyFinder>, bool) {
    (Arc::new(GuessOnly), false)
}

fn find_sky(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = FIND_CMD;
    let (_, it) = crate::cutout::video_clip(s, p, cmd)?;
    let (every, points) = crate::cutout::outline_params(p, cmd, POINTS)?;
    let (f, refined) = finder(s);
    let find: FindOutline = Box::new(move |rgba, w, h, n| f.find(rgba, w, h, n));
    let o = Outlining { mask: SKY, undo: "Find the Sky", what: "the sky", not_found: "no sky was found in the clip", lost_ends: false };
    let mut r = crate::cutout::outline_job(s, cmd, &it, o, every, points, find, crate::commands::bool_p(p, "wait").unwrap_or(false))?;
    r["refined"] = json!(refined);
    Ok(r)
}

/// A clip's sky mask, when it has been found.
pub fn sky_mask(it: &TrackItem) -> Option<&filmcraft_project::Mask> {
    it.effect("opacity")?.masks.iter().find(|m| m.name == SKY && !m.path.keyframes.is_empty())
}

// ---------------------------------------------------------------------------------------------
// the skies made here
// ---------------------------------------------------------------------------------------------

fn hash(i: i64, j: i64, seed: u64) -> f64 {
    let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 31;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 29;
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// Smooth value noise at (x, y).
fn noise(x: f64, y: f64, seed: u64) -> f64 {
    let (i, j) = (x.floor() as i64, y.floor() as i64);
    let (fx, fy) = (x - x.floor(), y - y.floor());
    let s = |t: f64| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(fx), s(fy));
    let top = hash(i, j, seed) * (1.0 - sx) + hash(i + 1, j, seed) * sx;
    let bottom = hash(i, j + 1, seed) * (1.0 - sx) + hash(i + 1, j + 1, seed) * sx;
    top * (1.0 - sy) + bottom * sy
}

fn fbm(x: f64, y: f64, seed: u64) -> f64 {
    (0..5).map(|o| noise(x * f64::from(1u32 << o), y * f64::from(1u32 << o), seed.wrapping_add(o)) / f64::from(1u32 << o)).sum::<f64>() / 1.9375
}

fn mix3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    let t = t.clamp(0.0, 1.0);
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

/// The colour of sky `kind` looking `az` round (radians) and `el` up from the horizon.
fn sky_color(kind: &str, az: f64, el: f64, seed: u64) -> [f64; 3] {
    let up = (el / std::f64::consts::FRAC_PI_2).clamp(-1.0, 1.0);
    let rise = up.max(0.0).powf(0.45);
    // the sun's direction (sunset and dusk): low, ahead
    let sun = (el.cos() * az.cos() * 0.087f64.cos() + el.sin() * 0.087f64.sin()).clamp(-1.0, 1.0);
    let ground = [0.32, 0.30, 0.28];
    if up < 0.0 {
        return mix3([0.55, 0.55, 0.55], ground, (-up * 6.0).min(1.0));
    }
    match kind {
        "sunset" => {
            let base = mix3(mix3([1.0, 0.62, 0.32], [0.86, 0.42, 0.45], rise * 1.6), [0.22, 0.24, 0.52], ((rise - 0.35) * 1.6).max(0.0));
            mix3(base, [1.0, 0.86, 0.6], sun.max(0.0).powi(60) + 0.35 * sun.max(0.0).powi(6))
        }
        "dusk" => {
            let base = mix3([0.72, 0.45, 0.55], [0.08, 0.09, 0.27], rise);
            mix3(base, [0.95, 0.65, 0.55], 0.4 * sun.max(0.0).powi(8))
        }
        "night" => {
            let base = mix3([0.07, 0.09, 0.17], [0.01, 0.015, 0.05], rise);
            // stars: a few bright points, more higher up
            let (u, v) = (az * 400.0, el * 400.0);
            let star = hash(u.floor() as i64, v.floor() as i64, seed);
            let bright = if star > 0.9985 - 0.001 * up { 0.6 + 0.4 * hash(v.floor() as i64, u.floor() as i64, seed) } else { 0.0 };
            mix3(base, [1.0, 1.0, 0.95], bright)
        }
        "clouds" => {
            let base = mix3([0.72, 0.83, 0.95], [0.2, 0.43, 0.84], rise);
            // a cloud layer above: its texture on a plane, thinning towards the horizon
            let d = 1.0 / el.sin().max(0.03);
            let (x, y) = (az.cos() * d * 0.9, az.sin() * d * 0.9);
            let c = ((fbm(x, y, seed) - 0.48) * 3.2).clamp(0.0, 1.0) * (1.0 - (1.0 - up).powi(8));
            mix3(base, [0.97, 0.97, 0.99], c * 0.92)
        }
        _ => mix3([0.72, 0.84, 0.95], [0.18, 0.42, 0.85], rise),
    }
}

/// Sky `kind` as an RGBA8 picture: a whole sphere (2:1, `w` wide) when `sphere`, else a flat
/// picture `w` × `h` looking ahead, 50° up at its top to the horizon at 60% down.
pub fn made_sky(kind: &str, w: usize, h: usize, sphere: bool, seed: u64) -> Vec<u8> {
    let (w, h) = (w.clamp(16, 8192), if sphere { (w / 2).clamp(8, 4096) } else { h.clamp(8, 8192) });
    let mut out = vec![255u8; w * h * 4];
    let enc = |v: f64| {
        let v = v.clamp(0.0, 1.0);
        let s = if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        (s * 255.0).round() as u8
    };
    for y in 0..h {
        for x in 0..w {
            let (az, el) = if sphere {
                (
                    (x as f64 + 0.5) / w as f64 * std::f64::consts::TAU - std::f64::consts::PI,
                    std::f64::consts::FRAC_PI_2 - (y as f64 + 0.5) / h as f64 * std::f64::consts::PI,
                )
            } else {
                let t = (y as f64 + 0.5) / h as f64;
                ((x as f64 / w as f64 - 0.5) * 1.2, (50.0 - 50.0 / 0.6 * t).to_radians())
            };
            // (the colour in linear light, written sRGB)
            let c = sky_color(kind, az, el, seed);
            let lin = c.map(|v| v.powf(2.2));
            let i = (y * w + x) * 4;
            for k in 0..3 {
                if let Some(o) = out.get_mut(i + k) {
                    *o = enc(lin[k]);
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// clip.skySwap
// ---------------------------------------------------------------------------------------------

fn sky_swap(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = SWAP_CMD;
    let (track, it) = crate::cutout::video_clip(s, p, cmd)?;
    let mask = sky_mask(&it).cloned().ok_or_else(|| bad(cmd, format!("find the clip's sky first ({FIND_CMD})")))?;
    if (it.speed - 1.0).abs() > 1e-6 || it.reverse || it.frame_hold.is_some() {
        return Err(bad(cmd, "the clip plays at another speed, backwards or held: put the sky in before changing its time"));
    }
    let feather = f64_p(p, "feather").filter(|v| v.is_finite() && (0.0..=200.0).contains(v)).unwrap_or(6.0);
    let relight = f64_p(p, "relight").filter(|v| v.is_finite() && (0.0..=100.0).contains(v)).unwrap_or(30.0) / 100.0;
    let seed = u64_p(p, "seed").unwrap_or(1);
    let size = filmcraft_render::clip_size(&s.project, &it).ok_or_else(|| bad(cmd, "the clip has no picture"))?;
    let sphere = it.effect(filmcraft_project::vr::REFRAME).is_some();
    let dir = match str_p(p, "folder") {
        Some(d) => std::path::PathBuf::from(crate::export_tools::expand_home(d)),
        None => crate::autosave::default_data_dir().map(|d| d.join("skies")).ok_or_else(|| bad(cmd, "no data directory for the sky"))?,
    };
    // the sky's picture: made here, or the user's photo cut to the clip's shape
    let (rgba, w, h, name) = match p.get("sky") {
        None | Some(Value::Null) => made(cmd, "sunset", sphere, size, seed)?,
        Some(Value::String(k)) => made(cmd, k, sphere, size, seed)?,
        Some(Value::Number(n)) => {
            let item = n.as_u64().map(ItemId).ok_or_else(|| bad(cmd, "`sky` is an item id or a kind of sky"))?;
            photo(s, cmd, item, size)?
        }
        Some(_) => return Err(bad(cmd, "`sky` is an item id or one of clear, sunset, dusk, night, clouds")),
    };
    let new_light = mean_color(&rgba, w, h, true);
    std::fs::create_dir_all(&dir).map_err(|e| bad(cmd, format!("{}: {e}", dir.display())))?;
    let path = dir.join(format!("{name}-{}x{}-{seed}.png", w, h));
    let png = filmcraft_export::encode_png(rgba, w as u32, h as u32).map_err(|e| bad(cmd, e.to_string()))?;
    let path_s = path.to_string_lossy().into_owned();
    s.services.write_file(&path_s, &png).map_err(|e| bad(cmd, format!("{path_s}: {e}")))?;
    // the old sky's light, from a frame in the middle
    let old_light = old_sky_light(s, &it, &mask, size);
    let n0 = s.history.undo.len();
    let r = s.execute("file.import", json!({"paths": [path_s], "reuse": true}))?;
    let sky_item = r["items"][0].as_u64().map(ItemId).ok_or_else(|| bad(cmd, "the sky did not import"))?;
    if sphere {
        s.execute("media.vrProperties", json!({"items": [sky_item.0], "projection": "equirectangular"}))?;
    }
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let (temperature, exposure) = match (old_light, new_light) {
        (Some(a), Some(b)) => relit(a, b, relight),
        _ => (0.0, 0.0),
    };
    let base = it.clone();
    let clip = s.edit("Sky Swap", |pr, st| {
        let mut c = base.clone();
        c.id = ClipId(pr.alloc_id());
        c.item = sky_item;
        c.name = format!("{} (sky)", base.name);
        c.link = None;
        c.group = None;
        c.markers.clear();
        c.source_in = Tick::ZERO;
        // its picture: the sphere through the clip's own camera, or the flat sky as the clip's
        c.effects.retain(|e| e.effect == "motion" || e.effect == "opacity" || e.effect == filmcraft_project::vr::REFRAME);
        if let Some(op) = c.effect_mut("opacity") {
            let mut m = mask.clone();
            m.mode = MaskMode::Add;
            m.feather = filmcraft_project::Param::new(ParamValue::Float(feather));
            op.masks = vec![m];
            for (id, v) in [("opacity", ParamValue::Float(100.0)), ("blend", ParamValue::Choice(0))] {
                if let Some(q) = op.param_mut(id) {
                    q.keyframes.clear();
                    q.value = v;
                }
            }
        }
        // (the sky's own time starts at 0 where the clip's media is at its in point: its camera's
        // and its mask's keyframes moved to match)
        crate::cutout::shift_keys(&mut c, base.source_in);
        let id = c.id;
        let mut tracks: Vec<u64> = vec![pr.alloc_id()];
        let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        crate::cutout::place_above(q, track, vec![c], &mut || tracks.pop().unwrap_or(0));
        // the clip relit towards the new sky
        if (temperature.abs() > 0.05 || exposure.abs() > 0.005)
            && let Some((_, b)) = q.find_item_mut(base.id)
            && let Some(mut lum) = filmcraft_project::find_effect("lumetri").map(|d| d.instance())
        {
            for (k, v) in [("temperature", temperature), ("exposure", exposure)] {
                if let Some(q) = lum.param_mut(k) {
                    q.value = ParamValue::Float(v);
                }
            }
            b.effects.push(lum);
        }
        q.check().map_err(EngineError::Other)?;
        st.selection = vec![id];
        Ok(id)
    })?;
    crate::clip_ops::collapse_history(s, n0, "Sky Swap");
    let r2 = |x: f64| (x * 100.0).round() / 100.0;
    Ok(json!({"clip": it.id.0, "sky": clip.0, "item": sky_item.0, "picture": path_s, "relight": {"temperature": r2(temperature), "exposure": r2(exposure)}}))
}

/// A sky made here, at the clip's size (a whole sphere for a 360° clip).
fn made(cmd: &str, kind: &str, sphere: bool, size: (u32, u32), seed: u64) -> Result<(Vec<u8>, usize, usize, String)> {
    if !KINDS.contains(&kind) {
        return Err(bad(cmd, format!("`sky` is an item id or one of {}, not `{}`", KINDS.join(", "), kind.chars().take(20).collect::<String>())));
    }
    let (w, h) = if sphere { (SPHERE_WIDTH, SPHERE_WIDTH / 2) } else { (size.0 as usize, size.1 as usize) };
    Ok((made_sky(kind, w, h, sphere, seed), w, h, format!("sky-{kind}{}", if sphere { "-sphere" } else { "" })))
}

/// The user's photo (a still of the project) filling the clip's shape, centred.
fn photo(s: &Session, cmd: &str, item: ItemId, size: (u32, u32)) -> Result<(Vec<u8>, usize, usize, String)> {
    let src = s.source(item).ok_or_else(|| bad(cmd, format!("no picture for item {}", item.0)))?;
    let f = src.video_frame(filmcraft_media::FrameRequest { time: Tick::ZERO, scale: 1.0 }).map_err(|e| bad(cmd, format!("the photo does not decode: {e}")))?;
    let (rgba, fw, fh) = (f.to_rgba8(), f.width as usize, f.height as usize);
    let (tw, th) = (size.0 as usize, size.1 as usize);
    // cut to the clip's aspect, then scaled to its size
    let k = (fw as f64 / tw.max(1) as f64).min(fh as f64 / th.max(1) as f64);
    let (cw, ch) = (((tw as f64 * k) as usize).clamp(1, fw.max(1)), ((th as f64 * k) as usize).clamp(1, fh.max(1)));
    let (x0, y0) = ((fw - cw) / 2, (fh - ch) / 2);
    let mut crop = vec![0u8; cw * ch * 4];
    for y in 0..ch {
        let from = ((y0 + y) * fw + x0) * 4;
        if let (Some(src), Some(dst)) = (rgba.get(from..from + cw * 4), crop.get_mut(y * cw * 4..(y + 1) * cw * 4)) {
            dst.copy_from_slice(src);
        }
    }
    let (out, w, h) = crate::cutout::resize_rgba(&crop, cw, ch, tw, th);
    Ok((out, w, h, format!("sky-photo-{}", item.0)))
}

/// The mean linear colour of the upper part of a picture (`upper`: its top half), RGBA8 in sRGB.
fn mean_color(rgba: &[u8], w: usize, h: usize, upper: bool) -> Option<[f64; 3]> {
    let rows = if upper { h / 2 } else { h };
    let lin = |v: u8| {
        let s = f64::from(v) / 255.0;
        if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    };
    let (mut acc, mut n) = ([0f64; 3], 0usize);
    for y in (0..rows).step_by(4) {
        for x in (0..w).step_by(4) {
            if let Some(p) = rgba.get((y * w + x) * 4..(y * w + x) * 4 + 3) {
                for k in 0..3 {
                    acc[k] += lin(p[k]);
                }
                n += 1;
            }
        }
    }
    (n > 0).then(|| acc.map(|v| v / n as f64))
}

/// The light of the clip's old sky: the mean linear colour inside its mask on its middle frame.
fn old_sky_light(s: &Session, it: &TrackItem, mask: &filmcraft_project::Mask, size: (u32, u32)) -> Option<[f64; 3]> {
    let src = crate::vr360::clip_source(s, it)?;
    let t = it.source_time_at(it.start + Tick(it.duration.0 / 2));
    let scale = (512.0 / f64::from(size.0.max(size.1).max(1))).min(1.0) as f32;
    let f = src.video_frame(filmcraft_media::FrameRequest { time: t, scale }).ok()?;
    let (rgba, w, h) = crate::scenes::small_rgba(&f, 512);
    let (w, h) = (w as usize, h as usize);
    let mut m = mask.clone();
    m.mode = MaskMode::Add;
    let cover = filmcraft_render::mask::coverage(&filmcraft_render::mask::prepare(&[m], t, w as f32 / size.0.max(1) as f32), w, h)?;
    let lin = |v: u8| {
        let s = f64::from(v) / 255.0;
        if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    };
    let (mut acc, mut n) = ([0f64; 3], 0f64);
    for (i, c) in cover.iter().enumerate() {
        if *c > 0.5
            && let Some(p) = rgba.get(i * 4..i * 4 + 3)
        {
            for k in 0..3 {
                acc[k] += lin(p[k]);
            }
            n += 1.0;
        }
    }
    (n > 0.0).then(|| acc.map(|v| v / n))
}

/// Lumetri's temperature and exposure moving a scene lit by sky `old` towards sky `new` (linear
/// RGB means) by `amount` (0..1): warmer or cooler by the skies' warmth, brighter or darker by a
/// part of their light.
pub fn relit(old: [f64; 3], new: [f64; 3], amount: f64) -> (f64, f64) {
    let warmth = |c: [f64; 3]| (c[0] - c[2]) / (c[0] + c[1] + c[2]).max(1e-6);
    let light = |c: [f64; 3]| (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]).max(1e-4);
    let a = amount.clamp(0.0, 1.0);
    let temperature = ((warmth(new) - warmth(old)) * 120.0 * a).clamp(-40.0, 40.0);
    let exposure = ((light(new) / light(old)).log2() * 0.35 * a).clamp(-1.0, 1.0);
    (if temperature.is_finite() { temperature } else { 0.0 }, if exposure.is_finite() { exposure } else { 0.0 })
}

#[cfg(test)]
#[path = "sky_tests.rs"]
mod tests;
