use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};

use super::*;
use crate::Session;
use crate::media_test_util::tmp_dir;
use crate::scene_models::{Calibration, Pic, SceneModels};

/// Stub models: a picture's embedding is its mean colour, a text's the colour it names; the
/// aesthetic score is how red a picture is.
#[derive(Default)]
struct Fake {
    pictures_seen: AtomicUsize,
    biggest_side: AtomicUsize,
    no_aesthetic: bool,
}

fn mean_rgb(p: &Pic) -> Vec<f32> {
    let mut acc = [0f32; 3];
    let n = (p.rgba.len() / 4).max(1) as f32;
    for px in p.rgba.chunks(4) {
        for c in 0..3 {
            acc[c] += f32::from(px[c]) / 255.0;
        }
    }
    let norm = acc.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
    acc.iter().map(|v| v / n / (norm / n)).collect()
}

impl SceneModels for Fake {
    fn embedder_id(&self) -> Option<String> {
        Some("fake-siglip".into())
    }
    fn aesthetic_id(&self) -> Option<String> {
        (!self.no_aesthetic).then(|| "fake-aesthetic".into())
    }
    fn embed_images(&self, pics: &[Pic]) -> std::result::Result<Vec<Vec<f32>>, String> {
        self.pictures_seen.fetch_add(pics.len(), Ordering::Relaxed);
        for p in pics {
            self.biggest_side.fetch_max(p.w.max(p.h), Ordering::Relaxed);
        }
        Ok(pics.iter().map(mean_rgb).collect())
    }
    fn embed_text(&self, q: &str) -> std::result::Result<(Vec<f32>, Option<Calibration>), String> {
        let v = if q.contains("red") { vec![1.0, 0.0, 0.0] } else { vec![0.0, 0.0, 1.0] };
        Ok((v, Some(Calibration { scale: 10.0, bias: -5.0 })))
    }
    fn aesthetic(&self, pics: &[Pic]) -> std::result::Result<Vec<f32>, String> {
        Ok(pics.iter().map(|p| 1.0 + 9.0 * mean_rgb(p)[0]).collect())
    }
}

fn session(fake: Fake) -> (Session, Arc<Fake>) {
    let fake = Arc::new(fake);
    let mut s = Session::default();
    s.scenes.models = Some(fake.clone());
    (s, fake)
}

fn picture(dir: &std::path::Path, name: &str, rgb: [u8; 3], w: u32, h: u32) -> String {
    let path = dir.join(name);
    image::RgbaImage::from_pixel(w, h, image::Rgba([rgb[0], rgb[1], rgb[2], 255])).save(&path).unwrap();
    path.to_string_lossy().into_owned()
}

fn floats(b64: &str, precision: Precision) -> Vec<f32> {
    let bytes = b64_decode(b64).unwrap();
    match precision {
        Precision::F16 => bytes.as_chunks::<2>().0.iter().map(|c| f16_to_f32(u16::from_le_bytes(*c))).collect(),
        Precision::F32 => bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
    }
}

#[test]
fn base64_round_trips() {
    for bytes in [vec![], vec![0u8], vec![1, 2], vec![255, 0, 7], (0..=255u8).collect::<Vec<_>>()] {
        assert_eq!(b64_decode(&b64_encode(&bytes)).unwrap(), bytes);
    }
    assert_eq!(b64_encode(&1.0f32.to_le_bytes()), "AACAPw==");
    assert!(b64_decode("not base64!").is_none());
}

#[test]
fn pictures_and_texts_are_embedded_in_order() {
    let dir = tmp_dir("embed-pictures");
    let red = picture(&dir, "red.png", [230, 20, 20], 2000, 500);
    let blue = picture(&dir, "blue.png", [20, 20, 230], 64, 64);
    let (mut s, fake) = session(Fake::default());
    let r = s.execute(EMBED_CMD, json!({"paths": [red, "/nonexistent/x.png", blue], "texts": ["a red kite"], "aesthetic": true, "maxSide": 320})).unwrap();
    let pics = r["pictures"].as_array().unwrap();
    assert_eq!(pics.len(), 3);
    assert!(pics[1]["error"].is_string() && pics[1].get("embedding").is_none());
    let (r0, b0) = (floats(pics[0]["embedding"].as_str().unwrap(), Precision::F16), floats(pics[2]["embedding"].as_str().unwrap(), Precision::F16));
    assert!(r0[0] > 0.9 && b0[2] > 0.9, "{r0:?} {b0:?}");
    assert!(pics[0]["aesthetic"].as_f64().unwrap() > pics[2]["aesthetic"].as_f64().unwrap());
    assert_eq!((pics[0]["width"].as_u64(), pics[0]["height"].as_u64()), (Some(320), Some(80)), "scaled to maxSide, aspect kept");
    assert_eq!(fake.biggest_side.load(Ordering::Relaxed), 320);
    assert_eq!(fake.pictures_seen.load(Ordering::Relaxed), 2);
    let t = floats(r["texts"][0]["embedding"].as_str().unwrap(), Precision::F16);
    assert_eq!(t, vec![1.0, 0.0, 0.0]);
    assert_eq!((r["dimensions"].as_u64(), r["precision"].as_str()), (Some(3), Some("f16")));
    assert_eq!(r["calibration"]["scale"].as_f64(), Some(10.0));
    assert_eq!(r["model"], "fake-siglip");
    let full = s.execute(EMBED_CMD, json!({"texts": ["blue sea"], "precision": "f32"})).unwrap();
    assert_eq!(floats(full["texts"][0]["embedding"].as_str().unwrap(), Precision::F32), vec![0.0, 0.0, 1.0]);
    assert!(full["pictures"].as_array().unwrap().is_empty());
}

#[test]
fn many_pictures_go_in_batches() {
    let dir = tmp_dir("embed-batches");
    let paths: Vec<String> = (0..70).map(|i| picture(&dir, &format!("p{i}.png"), [i as u8, 100, 200], 8, 8)).collect();
    let (mut s, fake) = session(Fake::default());
    let r = s.execute(EMBED_CMD, json!({"paths": paths})).unwrap();
    assert_eq!(r["pictures"].as_array().unwrap().iter().filter(|p| p["embedding"].is_string()).count(), 70);
    assert_eq!(fake.pictures_seen.load(Ordering::Relaxed), 70);
}

#[test]
fn hostile_parameters_are_refused() {
    let (mut s, _) = session(Fake { no_aesthetic: true, ..Fake::default() });
    let long = "x".repeat(MAX_TEXT_CHARS + 1);
    for p in [
        json!({}),
        json!({"paths": []}),
        json!({"paths": "x"}),
        json!({"paths": [5]}),
        json!({"paths": [""]}),
        json!({"paths": vec!["a"; MAX_PATHS + 1]}),
        json!({"texts": [long]}),
        json!({"texts": vec!["a"; MAX_TEXTS + 1]}),
        json!({"texts": ["  "]}),
        json!({"texts": ["a"], "precision": "f64"}),
        json!({"texts": ["a"], "maxSide": "big"}),
        json!({"paths": ["/nonexistent/a.png"], "aesthetic": true}),
    ] {
        assert!(s.execute(EMBED_CMD, p.clone()).is_err(), "{p}");
    }
    let dir = tmp_dir("embed-nearest-hostile");
    let index = dir.join("index.f16");
    std::fs::write(&index, [0u8; 12]).unwrap();
    let i = index.to_string_lossy();
    let v = b64_encode(&[1.0f32, 0.0, 0.0].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    for p in [
        json!({}),
        json!({"index": i, "dims": 3}),
        json!({"index": i, "dims": 3, "query": "red", "vector": v}),
        json!({"index": "/nonexistent/index", "dims": 3, "vector": v}),
        json!({"index": i, "dims": 5, "vector": v}),
        json!({"index": i, "dims": 0, "vector": v}),
        json!({"index": i, "dims": 2.5, "vector": v}),
        json!({"index": i, "dims": 3, "vector": v, "limit": 0}),
        json!({"index": i, "dims": 3, "vector": v, "limit": 1e9}),
        json!({"index": i, "dims": 3, "vector": v, "min": 2}),
        json!({"index": i, "dims": "3", "vector": v}),
        json!({"index": i, "dims": 3, "vector": v, "limit": "ten"}),
        json!({"index": i, "dims": 3, "vector": "!!"}),
        json!({"index": i, "dims": 2, "vector": v, "precision": "f16"}),
        json!({"index": i, "dims": 3, "vector": b64_encode(&[0u8; 12])}),
        json!({"index": dir.to_string_lossy(), "dims": 3, "vector": v}),
    ] {
        assert!(s.execute(NEAREST_CMD, p.clone()).is_err(), "{p}");
    }
}

fn write_index(dir: &std::path::Path, rows: &[[f32; 3]], precision: Precision) -> String {
    let path = dir.join(format!("index.{}", precision.name()));
    let bytes: Vec<u8> = rows
        .iter()
        .flat_map(|r| r.iter().flat_map(move |x| if precision == Precision::F16 { f32_to_f16(*x).to_le_bytes().to_vec() } else { x.to_le_bytes().to_vec() }))
        .collect();
    std::fs::write(&path, bytes).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn nearest_rows_best_first() {
    let dir = tmp_dir("embed-nearest");
    let rows = [[0.0, 0.0, 1.0], [0.9, 0.1, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.6, 0.0, 0.8]];
    let (mut s, _) = session(Fake::default());
    for precision in [Precision::F16, Precision::F32] {
        let index = write_index(&dir, &rows, precision);
        let r = s.execute(NEAREST_CMD, json!({"index": index, "dims": 3, "query": "red", "limit": 3, "precision": precision.name()})).unwrap();
        let hits = r["hits"].as_array().unwrap();
        assert_eq!(hits.iter().map(|h| h["row"].as_u64().unwrap()).collect::<Vec<_>>(), vec![2, 1, 4], "{precision:?}");
        assert!((hits[0]["cosine"].as_f64().unwrap() - 1.0).abs() < 1e-3);
        assert!(hits[0]["match"].as_f64().unwrap() > hits[2]["match"].as_f64().unwrap());
        assert_eq!(r["rows"].as_u64(), Some(5));
    }
    let index = write_index(&dir, &rows, Precision::F16);
    let v = b64_encode(&[0.0f32, 0.0, 2.0].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    let r = s.execute(NEAREST_CMD, json!({"index": index, "dims": 3, "vector": v, "min": 0.5})).unwrap();
    let rows_found: Vec<u64> = r["hits"].as_array().unwrap().iter().map(|h| h["row"].as_u64().unwrap()).collect();
    assert_eq!(rows_found, vec![0, 4], "a zero row is skipped and min drops the rest; the vector need not be unit length");
    assert!(r["hits"][0].get("match").is_none(), "a given vector has no calibration");
}

#[test]
fn rank_handles_empty_and_short_input() {
    assert!(rank(&[], &[1.0], Precision::F16, 10, None).is_empty());
    assert!(rank(&[1, 2, 3], &[], Precision::F32, 10, None).is_empty());
    assert!(rank(&[0, 60, 0], &[1.0], Precision::F16, 10, None).len() == 1, "a trailing partial row is ignored");
}

#[test]
fn without_models_the_commands_say_so() {
    let mut s = Session::default();
    struct NoEmbedder;
    impl SceneModels for NoEmbedder {}
    s.scenes.models = Some(Arc::new(NoEmbedder));
    let e = s.execute(EMBED_CMD, json!({"texts": ["red"]})).unwrap_err().to_string();
    assert!(e.contains("siglip2-base-224"), "{e}");
    let dir = tmp_dir("embed-no-models");
    let index = write_index(&dir, &[[1.0, 0.0, 0.0]], Precision::F16);
    let e = s.execute(NEAREST_CMD, json!({"index": index, "dims": 3, "query": "red"})).unwrap_err().to_string();
    assert!(e.contains("not downloaded"), "{e}");
    let _: Value = json!(null);
}
