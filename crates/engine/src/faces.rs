//! Who is in a picture: faces found (YuNet), aligned by their landmarks and fingerprinted
//! (AuraFace-v1, an IResNet-100), so a catalogue can tell one person from another and group a
//! library's faces into people. The fingerprint network downloads on the user's consent.
//!
//! | command | does |
//! |---|---|
//! | `faces.status` | whether the fingerprint network (and the face detector) are downloaded: names, sizes, licences, sources |
//! | `faces.downloadModel` | job: download the fingerprint network (260.7 MB, Apache-2.0), pinned and checked |
//! | `media.faces` | the faces in picture files (`paths`, up to 256): each one's box and landmarks (shares of the picture), detector score, side in pixels, how frontal it is, and its fingerprint (`embedding`: 512 little-endian f32, base64); with `cropsDir`, a 128 px JPEG of each face written there |
//! | `faces.group` | fingerprints grouped into people: `path` a file of `n × 512` little-endian f32 (as `media.faces` gives them, one after another), `threshold` the cosine two faces of one person share (0.45), `k` neighbours a face looks at (20); a label per fingerprint, 0 the largest group |
//!
//! Nothing leaves the machine: the networks run here, on the GPU when there is one.

#[cfg(feature = "privacy-models")]
use std::path::PathBuf;
#[cfg(feature = "privacy-models")]
use std::sync::Arc;
use std::sync::{Mutex, PoisonError};

use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, bool_p};
#[cfg(feature = "privacy-models")]
use crate::commands::{f64_p, str_p};
use crate::{EngineError, Result, Session};

pub const STATUS_CMD: &str = "faces.status";
pub const DOWNLOAD_CMD: &str = "faces.downloadModel";
pub const FACES_CMD: &str = "media.faces";
pub const GROUP_CMD: &str = "faces.group";
/// Most pictures one `media.faces` call reads.
pub const MAX_PATHS: usize = 256;
/// Largest picture file read.
#[cfg(feature = "privacy-models")]
const MAX_FILE: u64 = 256 << 20;
/// The side of a face crop.
#[cfg(feature = "privacy-models")]
const CROP_SIDE: u32 = 128;

pub fn commands() -> Vec<CommandSpec> {
    let spec = |id: &'static str, label: &'static str, params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>| CommandSpec {
        id,
        label,
        menu: &[],
        shortcut: None,
        params,
        enabled: |_| Ok(()),
        run,
        journal: false,
    };
    vec![
        spec(STATUS_CMD, "Face Fingerprint Model", "{}", status),
        spec(DOWNLOAD_CMD, "Download the Face Fingerprint Model", r#"{"wait":bool=false}"#, download),
        spec(
            FACES_CMD,
            "Find and Fingerprint Faces",
            r#"{"paths":[str] (picture files, 1..256),"minSide":px=24,"threshold":0..1=0.7,"maxSide":160..2048=640,"cropsDir":str?}"#,
            faces,
        ),
        spec(GROUP_CMD, "Group Faces into People", r#"{"path":str (n × 512 f32, little-endian),"threshold":-1..1=0.45,"k":1..200=20}"#, group),
    ]
}

fn installed() -> bool {
    cfg!(feature = "privacy-models")
        && crate::transcript::models_dir()
            .as_deref()
            .is_some_and(|d| filmcraft_vision::catalog::face_id_installed(d) && filmcraft_vision::catalog::installed(d))
}

fn status(_: &mut Session, _: &Value) -> Result<Value> {
    use filmcraft_vision::catalog as c;
    let dir = crate::transcript::models_dir();
    Ok(json!({
        "installed": installed(),
        "fingerprints": {"name": c::FACE_ID_NAME, "license": c::FACE_ID_LICENSE, "source": c::FACE_ID_SOURCE, "bytes": c::FACE_ID.size,
                         "downloaded": dir.as_deref().is_some_and(c::face_id_installed), "id": c::face_id_id()},
        "detector": {"name": c::PACKAGE_NAME, "license": c::LICENSE, "downloaded": dir.as_deref().is_some_and(c::installed),
                     "download": "redact.downloadModels"},
        "available": cfg!(feature = "privacy-models"),
    }))
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn download(s: &mut Session, p: &Value) -> Result<Value> {
    use std::sync::atomic::Ordering;
    let cmd = DOWNLOAD_CMD;
    let wait = bool_p(p, "wait").unwrap_or(false);
    let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for models"))?;
    if filmcraft_vision::catalog::face_id_installed(&dir) {
        return Ok(json!({"job": null, "installed": true}));
    }
    if !cfg!(feature = "privacy-models") {
        return Err(bad(cmd, "face fingerprints are not available in this build (built without the `privacy-models` feature)"));
    }
    let job = crate::narration::new_job(s, format!("Downloading {}", filmcraft_vision::catalog::FACE_ID_NAME));
    job.progress.total.store(filmcraft_vision::catalog::FACE_ID.size, Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let id = crate::narration::spawn(s, job, wait, "filmcraft-face-id-download", move || {
        let t0 = web_time::Instant::now();
        let r = download_files(&dir, &prog);
        crate::narration::finish(&prog, &res, r.map(|()| "AuraFace-v1 installed".to_string()), t0);
    })?;
    if wait && let Some(e) = s.jobs.iter().find(|j| j.id == id).and_then(|j| lock(&j.progress.error).clone()) {
        return Err(EngineError::Other(format!("{cmd}: {e}")));
    }
    Ok(json!({"job": id, "installed": false, "missingBytes": filmcraft_vision::catalog::FACE_ID.size}))
}

#[cfg(feature = "privacy-models")]
fn download_files(dir: &std::path::Path, prog: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    use std::sync::atomic::Ordering;
    let files: Vec<filmcraft_speech::models::ModelFile> = filmcraft_vision::catalog::FACE_ID_FILES
        .iter()
        .map(|f| filmcraft_speech::models::ModelFile { name: f.name, url: f.url, sha256: f.sha256, size: f.size })
        .collect();
    let mut progress = |done: u64, total: u64, file: &str| {
        prog.done.store(done, Ordering::Relaxed);
        prog.total.store(total, Ordering::Relaxed);
        *lock(&prog.status) = format!("Downloading {file}");
        !prog.cancel.load(Ordering::Relaxed)
    };
    filmcraft_speech::models::download_files(&filmcraft_vision::catalog::face_id_dir(dir), &files, &mut progress).map_err(|e| e.to_string())
}

#[cfg(not(feature = "privacy-models"))]
fn download_files(_: &std::path::Path, _: &filmcraft_export::Progress) -> std::result::Result<(), String> {
    Err("face fingerprints are not available in this build".into())
}

/// The base64 of little-endian f32s.
#[cfg(any(feature = "privacy-models", test))]
fn b64(v: &[f32]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n =
            (u32::from(c.first().copied().unwrap_or(0)) << 16) | (u32::from(c.get(1).copied().unwrap_or(0)) << 8) | u32::from(c.get(2).copied().unwrap_or(0));
        for (i, shift) in [18u32, 12, 6, 0].into_iter().enumerate() {
            if i <= c.len() {
                out.push(char::from(A.get(((n >> shift) & 63) as usize).copied().unwrap_or(b'A')));
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(feature = "privacy-models")]
fn paths_p(cmd: &str, p: &Value) -> Result<Vec<PathBuf>> {
    let list = p.get("paths").and_then(Value::as_array).ok_or_else(|| bad(cmd, "`paths`: picture files"))?;
    if list.is_empty() || list.len() > MAX_PATHS {
        return Err(bad(cmd, format!("`paths`: 1–{MAX_PATHS} picture files")));
    }
    list.iter().map(|v| v.as_str().filter(|s| !s.is_empty() && s.len() < 4096).map(PathBuf::from).ok_or_else(|| bad(cmd, "`paths`: file paths"))).collect()
}

#[cfg(feature = "privacy-models")]
mod nets {
    use super::*;
    use filmcraft_vision::FaceDetector;
    use filmcraft_vision::face_id::FaceEmbedder;

    pub(super) struct Nets {
        pub det: FaceDetector,
        pub emb: FaceEmbedder,
    }

    static NETS: Mutex<Option<Arc<Nets>>> = Mutex::new(None);

    /// The detector and the fingerprint network, loaded once (on the GPU when there is one).
    pub(super) fn get(cmd: &str) -> Result<Arc<Nets>> {
        let mut slot = lock(&NETS);
        if let Some(n) = slot.as_ref() {
            return Ok(n.clone());
        }
        let dir = crate::transcript::models_dir().ok_or_else(|| bad(cmd, "no data directory for models"))?;
        use filmcraft_vision::catalog as c;
        if !c::installed(&dir) {
            return Err(bad(cmd, "the face detector is not downloaded: run redact.downloadModels first"));
        }
        if !c::face_id_installed(&dir) {
            return Err(bad(cmd, "the face fingerprint network is not downloaded: run faces.downloadModel first"));
        }
        let mut det = FaceDetector::load(&c::package_dir(&dir).join(c::FACE.name)).map_err(|e| bad(cmd, e.to_string()))?;
        let mut emb = FaceEmbedder::load(&c::face_id_dir(&dir).join(c::FACE_ID.name)).map_err(|e| bad(cmd, e.to_string()))?;
        let (dg, eg) = (det.use_gpu(), emb.use_gpu());
        log::info!("face networks loaded (detector on the {}, fingerprints on the {})", if dg { "GPU" } else { "CPU" }, if eg { "GPU" } else { "CPU" });
        let n = Arc::new(Nets { det, emb });
        *slot = Some(n.clone());
        Ok(n)
    }
}

/// A square crop round a face (its box grown by half), `CROP_SIDE` px, as JPEG.
#[cfg(feature = "privacy-models")]
fn write_crop(rgba: &[u8], w: usize, h: usize, bbox: [f32; 4], path: &std::path::Path) -> std::result::Result<(), String> {
    let (bx, by, bw, bh) = (bbox[0] * w as f32, bbox[1] * h as f32, bbox[2] * w as f32, bbox[3] * h as f32);
    let side = (bw.max(bh) * 1.5).max(4.0);
    let (cx, cy) = (bx + bw / 2.0, by + bh / 2.0);
    let (x0, y0) = (cx - side / 2.0, cy - side / 2.0);
    let n = CROP_SIDE as usize;
    let mut out = vec![0u8; n * n * 4];
    for oy in 0..n {
        for ox in 0..n {
            let sx = (x0 + (ox as f32 + 0.5) * side / n as f32).floor();
            let sy = (y0 + (oy as f32 + 0.5) * side / n as f32).floor();
            if sx < 0.0 || sy < 0.0 || sx as usize >= w || sy as usize >= h {
                continue;
            }
            let at = (sy as usize * w + sx as usize) * 4;
            for c in 0..4 {
                if let (Some(o), Some(v)) = (out.get_mut((oy * n + ox) * 4 + c), rgba.get(at + c)) {
                    *o = if c == 3 { 255 } else { *v };
                }
            }
        }
    }
    let jpeg =
        filmcraft_export::encode_still_with_quality(filmcraft_export::Format::JpegSequence, out, CROP_SIDE, CROP_SIDE, false, 85).map_err(|e| e.to_string())?;
    std::fs::write(path, jpeg).map_err(|e| e.to_string())
}

#[cfg(feature = "privacy-models")]
fn faces(_: &mut Session, p: &Value) -> Result<Value> {
    use filmcraft_vision::face_id;
    let cmd = FACES_CMD;
    let paths = paths_p(cmd, p)?;
    let min_side = f64_p(p, "minSide").unwrap_or(24.0).clamp(8.0, 4096.0) as f32;
    let threshold = f64_p(p, "threshold").unwrap_or(0.7).clamp(0.05, 1.0) as f32;
    let max_side = f64_p(p, "maxSide").unwrap_or(640.0).clamp(160.0, 2048.0) as usize;
    let crops = str_p(p, "cropsDir").map(PathBuf::from);
    if let Some(d) = &crops
        && !d.is_dir()
    {
        return Err(bad(cmd, "`cropsDir` must be an existing folder"));
    }
    let nets = nets::get(cmd)?;
    let mut results = Vec::with_capacity(paths.len());
    let mut pending: Vec<(usize, Value, Vec<f32>)> = Vec::new();
    for (pi, path) in paths.iter().enumerate() {
        let read = std::fs::metadata(path)
            .map_err(|e| e.to_string())
            .and_then(|m| if m.len() > MAX_FILE { Err("the file is too large".to_string()) } else { std::fs::read(path).map_err(|e| e.to_string()) });
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("picture");
        let raster = read.and_then(|b| filmcraft_media::still::decode_raster(name, &b).map_err(|e| e.to_string()));
        let r = match raster {
            Ok(r) => r,
            Err(e) => {
                results.push(json!({"path": path, "error": e, "faces": []}));
                continue;
            }
        };
        let (w, h) = (r.width as usize, r.height as usize);
        let found = match nets.det.detect(&r.rgba, w, h, max_side, threshold) {
            Ok(f) => f,
            Err(e) => {
                results.push(json!({"path": path, "error": e.to_string(), "faces": []}));
                continue;
            }
        };
        let mut kept = 0usize;
        for (fi, d) in found.iter().enumerate() {
            let side = (d.bbox[2] * w as f32).min(d.bbox[3] * h as f32);
            if side < min_side || d.points.len() < 5 {
                continue;
            }
            let lm: [[f32; 2]; 5] = std::array::from_fn(|k| d.points.get(k).map_or([0.0, 0.0], |q| [q[0] * w as f32, q[1] * h as f32]));
            let Some(aligned) = face_id::align(&r.rgba, w, h, &lm) else { continue };
            // how frontal: the nose between the eyes (0 turned away … 1 straight on)
            let eye = (lm[1][0] - lm[0][0]).abs().max(1.0);
            let off = (lm[2][0] - (lm[0][0] + lm[1][0]) / 2.0).abs() / eye;
            let frontal = (1.0 - 2.0 * off).clamp(0.0, 1.0);
            let mut face = json!({
                "bbox": d.bbox, "landmarks": d.points, "score": d.score, "side": side, "frontal": frontal,
            });
            if let Some(dir) = &crops {
                let file = dir.join(format!("face-{pi}-{fi}.jpg"));
                match write_crop(&r.rgba, w, h, d.bbox, &file) {
                    Ok(()) => face["crop"] = json!(file),
                    Err(e) => face["cropError"] = json!(e),
                }
            }
            pending.push((pi, face, aligned));
            kept += 1;
        }
        results.push(json!({"path": path, "width": w, "height": h, "found": found.len(), "kept": kept, "faces": []}));
    }
    // fingerprints for every kept face, in batches on the GPU
    let aligned: Vec<Vec<f32>> = pending.iter().map(|(_, _, a)| a.clone()).collect();
    let prints = nets.emb.embed(&aligned).map_err(|e| bad(cmd, e.to_string()))?;
    for ((pi, mut face, _), e) in pending.into_iter().zip(prints) {
        face["embedding"] = json!(b64(&e));
        // `results` holds one entry a path, in order: the path's own entry is at its index
        if let Some(r) = results.get_mut(pi).and_then(|r| r.get_mut("faces")).and_then(Value::as_array_mut) {
            r.push(face);
        }
    }
    Ok(json!({"pictures": results, "model": filmcraft_vision::catalog::face_id_id(), "gpu": nets.emb.on_gpu()}))
}

#[cfg(not(feature = "privacy-models"))]
fn faces(_: &mut Session, _: &Value) -> Result<Value> {
    Err(bad(FACES_CMD, "face fingerprints are not available in this build (built without the `privacy-models` feature)"))
}

#[cfg(feature = "privacy-models")]
fn group(_: &mut Session, p: &Value) -> Result<Value> {
    use filmcraft_vision::face_id::{self, DIM};
    let cmd = GROUP_CMD;
    let path = str_p(p, "path").ok_or_else(|| bad(cmd, "`path`: a file of fingerprints"))?;
    let threshold = f64_p(p, "threshold").unwrap_or(0.45) as f32;
    let k = f64_p(p, "k").unwrap_or(20.0);
    if !(1.0..=200.0).contains(&k) || k.fract() != 0.0 {
        return Err(bad(cmd, "`k`: 1–200 neighbours"));
    }
    let meta = std::fs::metadata(path).map_err(|e| bad(cmd, e.to_string()))?;
    let max = (face_id::MAX_GROUPED * DIM * 4) as u64;
    if meta.len() > max || meta.len() % (DIM as u64 * 4) != 0 {
        return Err(bad(cmd, format!("`path` must hold whole fingerprints ({} bytes each), at most {}", DIM * 4, face_id::MAX_GROUPED)));
    }
    let bytes = std::fs::read(path).map_err(|e| bad(cmd, e.to_string()))?;
    let prints: Vec<[f32; DIM]> = bytes
        .as_chunks::<{ DIM * 4 }>()
        .0
        .iter()
        .map(|c| {
            let words = c.as_chunks::<4>().0;
            std::array::from_fn(|i| words.get(i).map_or(0.0, |b| f32::from_le_bytes(*b)))
        })
        .collect();
    let t0 = web_time::Instant::now();
    let labels = face_id::group(&prints, threshold, k as usize).map_err(|e| bad(cmd, e.to_string()))?;
    let groups = labels.iter().copied().max().map_or(0, |m| m + 1);
    Ok(json!({"labels": labels, "groups": groups, "faces": prints.len(), "seconds": t0.elapsed().as_secs_f64()}))
}

#[cfg(not(feature = "privacy-models"))]
fn group(_: &mut Session, _: &Value) -> Result<Value> {
    Err(bad(GROUP_CMD, "face grouping is not available in this build (built without the `privacy-models` feature)"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_the_standard_alphabet() {
        assert_eq!(b64(&[]), "");
        assert_eq!(b64(&[1.0]), "AACAPw==");
        assert_eq!(b64(&[1.0, -2.0]), "AACAPwAAAMA=");
    }

    #[test]
    fn hostile_parameters_are_refused() {
        let mut s = Session::default();
        for p in [json!({}), json!({"paths": []}), json!({"paths": "x"}), json!({"paths": [5]}), json!({"paths": vec!["a"; MAX_PATHS + 1]})] {
            assert!(s.execute(FACES_CMD, p).is_err());
        }
        for p in [json!({}), json!({"path": "/nonexistent/prints.f32"}), json!({"path": "x", "k": 0}), json!({"path": "x", "k": 2.5})] {
            assert!(s.execute(GROUP_CMD, p).is_err());
        }
        let st = s.execute(STATUS_CMD, json!({})).expect("status");
        assert!(st["fingerprints"]["bytes"].as_u64() == Some(260_694_151));
    }
}
