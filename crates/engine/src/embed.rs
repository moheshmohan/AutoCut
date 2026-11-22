//! Vectors for search by meaning, for catalogues outside a project (FrameAtlas): SigLIP 2 image
//! embeddings of picture files and text embeddings of queries, optionally the LAION aesthetic score
//! of each picture, and the nearest rows of an index file of embeddings to a query.
//!
//! | command | does |
//! |---|---|
//! | `media.embed {paths?, texts?, aesthetic?, maxSide?, precision?}` | for each picture file of `paths` (up to 256) its unit-length SigLIP 2 image embedding (`embedding`: base64 of little-endian `f16` or `f32` by `precision`, default `f16`), and with `aesthetic` its aesthetic score (about 1–10); for each of `texts` (up to 64) its unit-length text embedding; `calibration` turns a cosine into SigLIP's match probability (`sigmoid(scale · cos + bias)`) |
//! | `media.nearest {index, query? \| vector?, dims?, precision?, limit?, min?}` | the rows of `index` (a file of `n × dims` little-endian `f16`, or `f32` by `precision`, as `media.embed` gives them, one after another) nearest a text `query` (embedded here) or a `vector` (base64 `f32`), best first: `row` (0-based), `cosine`, and for a text query `match` |
//!
//! Pictures are decoded and scaled to at most `maxSide` (160–1024, default 640) on their long side
//! before the model sees them. The models run here, on the GPU when there is one, under the GPU
//! lock; nothing leaves the machine. Without the models (or in a build without them) the commands
//! say so; nothing is downloaded (`filmcraft-cli scenes download siglip2-base-224`, and
//! `clip-vit-l14-aesthetic` for the score).

use std::path::PathBuf;
use std::sync::Arc;

use filmcraft_project::scenes::{f16_to_f32, f32_to_f16};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p, f64_p, str_p};
use crate::scene_models::{Pic, SceneModels};
use crate::{Result, Session};

pub const EMBED_CMD: &str = "media.embed";
pub const NEAREST_CMD: &str = "media.nearest";
/// Most pictures and texts one `media.embed` call takes.
pub const MAX_PATHS: usize = 256;
pub const MAX_TEXTS: usize = 64;
/// The longest text, characters.
const MAX_TEXT_CHARS: usize = 300;
/// Largest picture file read.
const MAX_FILE: u64 = 64 << 20;
/// Largest index file read (about 1.4 million 768-dimension f16 rows).
const MAX_INDEX: u64 = 2 << 30;
const MAX_DIMS: usize = 4096;
const MAX_LIMIT: usize = 1000;
/// Pictures the model sees at once.
const BATCH: usize = 32;

pub fn commands() -> Vec<CommandSpec> {
    let spec = |id: &'static str, label: &'static str, params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>| CommandSpec {
        id,
        label,
        menu: &[],
        shortcut: None,
        params,
        enabled: always,
        run,
        journal: false,
    };
    vec![
        spec(
            EMBED_CMD,
            "Embed Pictures and Texts",
            r#"{"paths":[str]? (picture files, up to 256),"texts":[str]? (up to 64),"aesthetic":bool=false,"maxSide":160..1024=640,"precision":"f16"|"f32"="f16"}"#,
            embed,
        ),
        spec(
            NEAREST_CMD,
            "Find the Nearest Embeddings",
            r#"{"index":str (n × dims little-endian floats),"query":str?,"vector":str? (base64 f32),"dims":1..4096=768,"precision":"f16"|"f32"="f16","limit":1..1000=50,"min":-1..1?}"#,
            nearest,
        ),
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    F16,
    F32,
}

impl Precision {
    fn bytes(self) -> usize {
        match self {
            Precision::F16 => 2,
            Precision::F32 => 4,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Precision::F16 => "f16",
            Precision::F32 => "f32",
        }
    }
}

fn precision_p(cmd: &str, p: &Value) -> Result<Precision> {
    match p.get("precision") {
        None | Some(Value::Null) => Ok(Precision::F16),
        Some(v) => match v.as_str() {
            Some("f16") => Ok(Precision::F16),
            Some("f32") => Ok(Precision::F32),
            _ => Err(bad(cmd, "`precision`: \"f16\" or \"f32\"")),
        },
    }
}

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 (with padding).
pub fn b64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n =
            (u32::from(c.first().copied().unwrap_or(0)) << 16) | (u32::from(c.get(1).copied().unwrap_or(0)) << 8) | u32::from(c.get(2).copied().unwrap_or(0));
        for (i, shift) in [18u32, 12, 6, 0].into_iter().enumerate() {
            if i <= c.len() {
                out.push(char::from(ALPHABET.get(((n >> shift) & 63) as usize).copied().unwrap_or(b'A')));
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// Standard base64 back to bytes; `None` for anything that is not.
pub fn b64_decode(text: &str) -> Option<Vec<u8>> {
    let text = text.trim_end_matches('=');
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in text.bytes() {
        let v = ALPHABET.iter().position(|a| *a == c)? as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((acc >> bits) & 0xff) as u8);
        }
    }
    Some(out)
}

/// A vector as base64 of little-endian floats.
pub fn encode(v: &[f32], precision: Precision) -> String {
    let bytes: Vec<u8> = match precision {
        Precision::F16 => v.iter().flat_map(|x| f32_to_f16(*x).to_le_bytes()).collect(),
        Precision::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    b64_encode(&bytes)
}

fn models(cmd: &str, s: &mut Session) -> Result<Arc<dyn SceneModels>> {
    s.scenes.check_models().ok_or_else(|| bad(cmd, "this build has no scene models (built without the `scene-models` feature)"))
}

const NO_EMBEDDER: &str = "the embedding model (SigLIP 2) is not downloaded: run `filmcraft-cli scenes download siglip2-base-224`";

/// A number parameter: absent (or null) is `None`; anything but a finite number is refused.
fn number(cmd: &str, p: &Value, key: &str, what: &str) -> Result<Option<f64>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(_) => match f64_p(p, key) {
            Some(v) if v.is_finite() => Ok(Some(v)),
            _ => Err(bad(cmd, format!("`{key}`: {what}"))),
        },
    }
}

fn strings(cmd: &str, p: &Value, key: &str, max: usize, max_chars: usize) -> Result<Vec<String>> {
    let Some(v) = p.get(key) else { return Ok(Vec::new()) };
    let list = v.as_array().ok_or_else(|| bad(cmd, format!("`{key}`: a list")))?;
    if list.len() > max {
        return Err(bad(cmd, format!("`{key}`: at most {max}")));
    }
    list.iter()
        .map(|x| {
            x.as_str()
                .map(str::trim)
                .filter(|s| !s.is_empty() && s.chars().count() <= max_chars)
                .map(str::to_string)
                .ok_or_else(|| bad(cmd, format!("`{key}`: non-empty text of up to {max_chars} characters")))
        })
        .collect()
}

/// A picture file decoded and scaled for the model, or why not.
fn picture(path: &std::path::Path, max_side: u32) -> std::result::Result<Pic, String> {
    let meta = std::fs::metadata(path).map_err(|e| e.to_string())?;
    if !meta.is_file() {
        return Err("not a file".into());
    }
    if meta.len() > MAX_FILE {
        return Err("the file is too large".into());
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("picture");
    let r = filmcraft_media::still::decode_raster(name, &bytes).map_err(|e| e.to_string())?;
    if r.width == 0 || r.height == 0 {
        return Err("the picture is empty".into());
    }
    let (rgba, w, h) = crate::media_frame::downscale_rgba(&r.rgba, r.width, r.height, max_side);
    Ok(Pic { rgba, w: w as usize, h: h as usize })
}

fn embed(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = EMBED_CMD;
    let paths: Vec<PathBuf> = strings(cmd, p, "paths", MAX_PATHS, 4095)?.into_iter().map(PathBuf::from).collect();
    let texts = strings(cmd, p, "texts", MAX_TEXTS, MAX_TEXT_CHARS)?;
    if paths.is_empty() && texts.is_empty() {
        return Err(bad(cmd, "give `paths` (picture files) or `texts`"));
    }
    let with_aesthetic = bool_p(p, "aesthetic").unwrap_or(false);
    let max_side = match number(cmd, p, "maxSide", "160–1024 pixels")? {
        None => 640,
        Some(v) => v.clamp(160.0, 1024.0) as u32,
    };
    let precision = precision_p(cmd, p)?;
    let m = models(cmd, s)?;
    let embedder = m.embedder_id().ok_or_else(|| bad(cmd, NO_EMBEDDER))?;
    let aesthetic_id = if with_aesthetic && !paths.is_empty() {
        Some(m.aesthetic_id().ok_or_else(|| bad(cmd, "the aesthetic model is not downloaded: run `filmcraft-cli scenes download clip-vit-l14-aesthetic`"))?)
    } else {
        None
    };
    let _gpu = m.gpu().map_err(|e| bad(cmd, format!("the GPU: {e}")))?;
    let t0 = web_time::Instant::now();
    let mut pictures: Vec<Value> = Vec::with_capacity(paths.len());
    let mut pics: Vec<Pic> = Vec::new();
    let mut slots: Vec<usize> = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        match picture(path, max_side) {
            Ok(pic) => {
                pictures.push(json!({"path": path, "width": pic.w, "height": pic.h}));
                pics.push(pic);
                slots.push(i);
            }
            Err(e) => pictures.push(json!({"path": path, "error": e})),
        }
    }
    let mut dims = 0usize;
    for (chunk, at) in pics.chunks(BATCH).zip(slots.chunks(BATCH)) {
        let vectors = m.embed_images(chunk).map_err(|e| bad(cmd, format!("embeddings: {e}")))?;
        if vectors.len() != chunk.len() {
            return Err(bad(cmd, "the embedding model gave a different number of embeddings than pictures"));
        }
        let scores = match &aesthetic_id {
            Some(_) => {
                let s = m.aesthetic(chunk).map_err(|e| bad(cmd, format!("aesthetic scores: {e}")))?;
                if s.len() != chunk.len() {
                    return Err(bad(cmd, "the aesthetic model gave a different number of scores than pictures"));
                }
                Some(s)
            }
            None => None,
        };
        for (k, (v, slot)) in vectors.iter().zip(at).enumerate() {
            dims = v.len();
            if let Some(entry) = pictures.get_mut(*slot) {
                entry["embedding"] = json!(encode(v, precision));
                if let Some(score) = scores.as_ref().and_then(|s| s.get(k)).filter(|x| x.is_finite()) {
                    entry["aesthetic"] = json!((f64::from(*score) * 1e4).round() / 1e4);
                }
            }
        }
    }
    let mut calibration = None;
    let mut out_texts = Vec::with_capacity(texts.len());
    for text in &texts {
        let (v, cal) = m.embed_text(text).map_err(|e| bad(cmd, format!("text embedding: {e}")))?;
        dims = v.len();
        if calibration.is_none() {
            calibration = cal;
        }
        out_texts.push(json!({"text": text, "embedding": encode(&v, precision)}));
    }
    m.release();
    Ok(json!({
        "pictures": pictures, "texts": out_texts, "model": embedder, "aestheticModel": aesthetic_id,
        "dimensions": dims, "precision": precision.name(),
        "calibration": calibration.map(|c| json!({"scale": c.scale, "bias": c.bias})),
        "seconds": (t0.elapsed().as_secs_f64() * 1e3).round() / 1e3,
    }))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The rows of a packed index nearest a unit query, best first.
pub fn rank(index: &[u8], query: &[f32], precision: Precision, limit: usize, min: Option<f32>) -> Vec<(usize, f32)> {
    let dims = query.len();
    let row_bytes = dims.saturating_mul(precision.bytes());
    if dims == 0 || row_bytes == 0 {
        return Vec::new();
    }
    let mut scored: Vec<(usize, f32)> = index
        .chunks_exact(row_bytes)
        .enumerate()
        .filter_map(|(row, bytes)| {
            let mut dot = 0f32;
            let mut norm = 0f32;
            match precision {
                Precision::F16 => {
                    for (pair, q) in bytes.as_chunks::<2>().0.iter().zip(query) {
                        let v = f16_to_f32(u16::from_le_bytes(*pair));
                        dot += v * q;
                        norm += v * v;
                    }
                }
                Precision::F32 => {
                    for (word, q) in bytes.as_chunks::<4>().0.iter().zip(query) {
                        let v = f32::from_le_bytes(*word);
                        dot += v * q;
                        norm += v * v;
                    }
                }
            }
            // rows are stored unit-length, but rounding (f16) and hand-made indexes are not exact
            let cos = if norm > 0.0 { dot / norm.sqrt() } else { return None };
            (cos.is_finite() && min.is_none_or(|m| cos >= m)).then_some((row, cos))
        })
        .collect();
    let by_score = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if scored.len() > limit {
        scored.select_nth_unstable_by(limit, by_score);
        scored.truncate(limit);
    }
    scored.sort_by(by_score);
    scored
}

fn nearest(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = NEAREST_CMD;
    let index = str_p(p, "index").filter(|x| !x.is_empty() && x.len() < 4096).ok_or_else(|| bad(cmd, "`index`: a file of embeddings"))?;
    let precision = precision_p(cmd, p)?;
    let dims = match number(cmd, p, "dims", "a whole number")? {
        None => 768,
        Some(d) if d.is_finite() && d.fract() == 0.0 && (1.0..=MAX_DIMS as f64).contains(&d) => d as usize,
        Some(_) => return Err(bad(cmd, format!("`dims`: a whole number, 1–{MAX_DIMS}"))),
    };
    let limit = match number(cmd, p, "limit", "a whole number")? {
        None => 50,
        Some(l) if l.is_finite() && l.fract() == 0.0 && (1.0..=MAX_LIMIT as f64).contains(&l) => l as usize,
        Some(_) => return Err(bad(cmd, format!("`limit`: 1–{MAX_LIMIT}"))),
    };
    let min = match number(cmd, p, "min", "a cosine")? {
        None => None,
        Some(m) if m.is_finite() && (-1.0..=1.0).contains(&m) => Some(m as f32),
        Some(_) => return Err(bad(cmd, "`min`: a cosine, -1–1")),
    };
    let text = str_p(p, "query").map(str::trim).filter(|q| !q.is_empty());
    let given = str_p(p, "vector");
    if text.is_some() == given.is_some() {
        return Err(bad(cmd, "give a text `query` or a `vector`, not both"));
    }
    if text.is_some_and(|q| q.chars().count() > MAX_TEXT_CHARS) {
        return Err(bad(cmd, format!("`query`: up to {MAX_TEXT_CHARS} characters")));
    }
    let meta = std::fs::metadata(index).map_err(|e| bad(cmd, format!("`index`: {e}")))?;
    let row_bytes = (dims * precision.bytes()) as u64;
    if !meta.is_file() || meta.len() > MAX_INDEX || meta.len() % row_bytes != 0 {
        return Err(bad(cmd, format!("`index` must hold whole rows of {dims} {} values, at most {MAX_INDEX} bytes", precision.name())));
    }
    let (mut query, calibration) = match (text, given) {
        (Some(q), _) => {
            let m = models(cmd, s)?;
            if m.embedder_id().is_none() {
                return Err(bad(cmd, NO_EMBEDDER));
            }
            let _gpu = m.gpu().map_err(|e| bad(cmd, format!("the GPU: {e}")))?;
            m.embed_text(q).map_err(|e| bad(cmd, format!("text embedding: {e}")))?
        }
        (None, Some(v)) => {
            let bytes = b64_decode(v).ok_or_else(|| bad(cmd, "`vector`: base64 of little-endian f32"))?;
            let floats: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            (floats, None)
        }
        (None, None) => return Err(bad(cmd, "give a text `query` or a `vector`")),
    };
    if query.len() != dims {
        return Err(bad(cmd, format!("the query has {} dimensions, the index {dims}", query.len())));
    }
    let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(bad(cmd, "the query vector is empty"));
    }
    query.iter_mut().for_each(|x| *x /= norm);
    let t0 = web_time::Instant::now();
    let bytes = std::fs::read(index).map_err(|e| bad(cmd, format!("`index`: {e}")))?;
    let rows = bytes.len() / row_bytes as usize;
    let hits: Vec<Value> = rank(&bytes, &query, precision, limit, min)
        .into_iter()
        .map(|(row, cos)| {
            let mut h = json!({"row": row, "cosine": (f64::from(cos) * 1e4).round() / 1e4});
            if let Some(c) = calibration {
                h["match"] = json!((f64::from(sigmoid(c.scale * cos + c.bias)) * 1e4).round() / 1e4);
            }
            h
        })
        .collect();
    Ok(json!({"hits": hits, "rows": rows, "dimensions": dims, "seconds": (t0.elapsed().as_secs_f64() * 1e3).round() / 1e3}))
}

#[cfg(test)]
#[path = "embed_tests.rs"]
mod tests;
