use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use filmcraft_project::ItemId;
use serde_json::{Value, json};

use super::*;
use crate::Session;
use crate::media_test_util::tmp_dir;
use crate::scene_models::{Calibration, DescribeProgress, DescribeStats, Detail, Pic, SceneModels};

const W: u32 = 160;
const H: u32 = 90;
/// Three shots of 20 frames: red, green, blue, each with its own moving pattern.
const SHOT: u32 = 20;

fn frame(k: u32) -> image::RgbaImage {
    let shot = k / SHOT;
    let t = (k % SHOT) as f32;
    image::RgbaImage::from_fn(W, H, |x, y| {
        let xx = x as f32 + 2.0 * t;
        let v = ((xx * 0.2).sin() * 0.5 + 0.5) * 120.0 + (y as f32 * 0.4);
        let v = v.min(255.0) as u8;
        match shot {
            0 => image::Rgba([200, v / 3, v / 4, 255]),
            1 => image::Rgba([v / 4, 190, if ((x / 8 + y / 8) % 2) == 0 { 40 } else { 90 }, 255]),
            _ => image::Rgba([30, 30 + v / 3, 210, 255]),
        }
    })
}

/// A session with one image-sequence clip of three shots (60 frames), and its item.
fn session() -> (Session, ItemId) {
    let dir = tmp_dir("scenes-clip");
    for k in 0..3 * SHOT {
        frame(k).save(dir.join(format!("clip{k:04}.png"))).unwrap();
    }
    let mut s = Session::default();
    let first = dir.join("clip0000.png");
    let r = s.execute("file.importImageSequence", json!({"path": first.to_string_lossy()})).unwrap();
    let item = ItemId(r["items"][0].as_u64().unwrap_or_else(|| panic!("{r}")));
    (s, item)
}

fn mean_rgb(p: &Pic) -> [f32; 3] {
    let mut acc = [0f32; 3];
    let n = (p.rgba.len() / 4).max(1) as f32;
    for px in p.rgba.chunks(4) {
        for c in 0..3 {
            acc[c] += f32::from(px[c]);
        }
    }
    acc.map(|v| v / n / 255.0)
}

/// Stub models: shots are described and embedded by their mean colour.
#[derive(Default)]
struct Fake {
    described: AtomicUsize,
    embedded: AtomicUsize,
    released: AtomicUsize,
}

impl SceneModels for Fake {
    fn describer_id(&self, d: Detail) -> Option<String> {
        Some(format!("fake-vlm-{}", d.name()))
    }
    fn embedder_id(&self) -> Option<String> {
        Some("fake-emb".into())
    }
    fn aesthetic_id(&self) -> Option<String> {
        Some("fake-aesthetic".into())
    }
    fn describe(&self, shots: &[Vec<Pic>], _: Detail, progress: DescribeProgress<'_>) -> std::result::Result<(Vec<String>, DescribeStats), String> {
        self.described.fetch_add(shots.len(), Ordering::Relaxed);
        progress(0, shots.len());
        let out = shots
            .iter()
            .map(|pics| {
                let [r, g, b] = mean_rgb(&pics[0]);
                let colour = if r > g && r > b { "red" } else if g > b { "green" } else { "blue" };
                // the answers are what a small model writes: fenced, sometimes cut off
                format!("```json\n{{\"shot_type\": \"wide\", \"subjects\": [{{\"name\": \"{colour} pattern\", \"box\": [0, 0, 1000, 1000]}}], \"caption\": \"A {colour} pattern moves.\", \"uses\": [\"b-roll\"], \"energy\": \"low\"")
            })
            .collect();
        Ok((out, DescribeStats { batches: 1, ..Default::default() }))
    }
    fn embed_images(&self, pics: &[Pic]) -> std::result::Result<Vec<Vec<f32>>, String> {
        self.embedded.fetch_add(pics.len(), Ordering::Relaxed);
        Ok(pics.iter().map(|p| mean_rgb(p).to_vec()).collect())
    }
    fn embed_text(&self, q: &str) -> std::result::Result<(Vec<f32>, Option<Calibration>), String> {
        let v = if q.contains("crimson") {
            vec![1.0, 0.0, 0.0]
        } else if q.contains("emerald") {
            vec![0.0, 1.0, 0.0]
        } else {
            vec![0.0, 0.0, 1.0]
        };
        Ok((v, None))
    }
    fn aesthetic(&self, pics: &[Pic]) -> std::result::Result<Vec<f32>, String> {
        Ok(pics.iter().map(|_| 5.0).collect())
    }
    fn release(&self) {
        self.released.fetch_add(1, Ordering::Relaxed);
    }
}

fn analyze(s: &mut Session, p: Value) -> Value {
    s.execute("media.analyzeScenes", p).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn finds_the_cuts_of_a_synthetic_clip_without_models() {
    let (mut s, item) = session();
    s.scenes.models = Some(Arc::new(crate::scene_models::NoModels));
    let r = analyze(&mut s, json!({"items": [item.0], "wait": true}));
    assert_eq!(r["scenes"]["items"][0]["boundaries"], "signature", "{r}");
    let idx = s.project.scenes.get(&item).expect("stored with the project");
    let rate = s.project.item(item).unwrap().as_media().unwrap().frame_rate();
    let cut = |k: i64| filmcraft_time::Tick(rate.frame_duration().0 * k);
    assert_eq!(idx.shots.len(), 3, "{:?}", idx.shots.iter().map(|x| (x.start, x.end)).collect::<Vec<_>>());
    assert_eq!(idx.shots[1].start, cut(SHOT as i64));
    assert_eq!(idx.shots[2].start, cut(2 * SHOT as i64));
    assert_eq!(idx.shots[2].end, idx.duration);
    // every shot was measured: sharp patterns, exposed, a camera that moves left
    for sh in &idx.shots {
        assert!(sh.quality.exposure > 0.2 && sh.quality.laplacian > 0.0, "{:?}", sh.quality);
        assert!(sh.quality.score > 0.0);
        assert_eq!(sh.keyframes.len(), 1);
        assert!(sh.description.is_none() && sh.embedding.is_empty());
    }
    assert!(matches!(idx.shots[0].camera, filmcraft_project::scenes::CameraMove::PanRight), "{:?}", idx.shots[0].camera);
    // one undo step
    assert_eq!(s.history.undo.last().map(|h| h.0.as_str()), Some(LABEL));
    s.execute("edit.undo", json!({})).unwrap();
    assert!(s.project.scenes.is_empty());
}

#[test]
fn models_describe_embed_and_score_and_search_ranks_by_them() {
    let (mut s, item) = session();
    let fake = Arc::new(Fake::default());
    s.scenes.models = Some(fake.clone());
    let r = analyze(&mut s, json!({"wait": true}));
    assert_eq!(r["scenes"]["described"], 3, "{r}");
    assert_eq!(fake.described.load(Ordering::Relaxed), 3);
    assert_eq!(fake.released.load(Ordering::Relaxed), 1, "the big models are let go after the job");
    let idx = s.project.scenes.get(&item).unwrap();
    assert_eq!(idx.describer, "fake-vlm-fast");
    let caps: Vec<String> = idx.shots.iter().map(|x| x.description.as_ref().unwrap().caption.clone()).collect();
    assert_eq!(caps, ["A red pattern moves.", "A green pattern moves.", "A blue pattern moves."], "the cut-off answers were read");
    assert!(idx.shots.iter().all(|x| x.quality.aesthetic == Some(5.0) && x.embedding.0.len() == 3));
    // media.scenes: seconds, no embeddings unless asked
    let v = s.execute("media.scenes", json!({"item": item.0})).unwrap();
    assert_eq!(v["shots"].as_array().unwrap().len(), 3);
    assert!(v["shots"][0].get("embedding").is_none());
    assert_eq!(v["shots"][1]["description"]["shotType"], "wide");
    let v = s.execute("media.scenes", json!({"item": item.0, "embeddings": true})).unwrap();
    assert_eq!(v["shots"][0]["embedding"].as_array().unwrap().len(), 3);
    // search: by the picture (stub embeddings) and by the description's words
    let mut hit = |q: &str| -> u64 {
        let v = s.execute("media.searchScenes", json!({"query": q})).unwrap();
        assert_eq!(v["visual"], true, "{v}");
        v["hits"][0]["shot"].as_u64().unwrap()
    };
    assert_eq!(hit("crimson"), 0);
    assert_eq!(hit("emerald"), 1);
    assert_eq!(hit("navy"), 2);
    let v = s.execute("media.searchScenes", json!({"query": "green pattern", "limit": 1})).unwrap();
    assert_eq!(v["hits"].as_array().unwrap().len(), 1);
    // best shots: every shot is described as b-roll
    let v = s.execute("media.bestShots", json!({"purpose": "broll", "count": 2})).unwrap();
    assert_eq!(v["shots"].as_array().unwrap().len(), 2, "{v}");
    assert!(v["shots"][0]["why"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("b-roll")));
}

#[test]
fn the_index_survives_save_and_open() {
    let (mut s, item) = session();
    s.scenes.models = Some(Arc::new(Fake::default()));
    analyze(&mut s, json!({"wait": true}));
    let before = s.project.scenes.get(&item).unwrap().clone();
    let path = tmp_dir("scenes-save").join("p.fcproj");
    s.execute("file.saveAs", json!({"path": path.to_string_lossy()})).unwrap();
    let mut t = Session::default();
    t.execute("file.open", json!({"path": path.to_string_lossy()})).unwrap();
    let after = t.project.scenes.get(&item).expect("the index is in the file");
    assert_eq!(*after, before, "embeddings, descriptions and times round-trip");
}

#[test]
fn the_cache_skips_footage_seen_before() {
    let (mut s, item) = session();
    let dir = tmp_dir("scenes-cache");
    s.use_scene_cache(&dir);
    let fake = Arc::new(Fake::default());
    s.scenes.models = Some(fake.clone());
    let r = analyze(&mut s, json!({"wait": true}));
    assert_eq!(r["scenes"]["items"][0]["fromCache"], false);
    // image sequences are not files the cache can key; a file-backed clip is (below)
    let _ = item;
    let (mut s2, item2) = session_from_movie();
    s2.use_scene_cache(&dir);
    let fake2 = Arc::new(Fake::default());
    s2.scenes.models = Some(fake2.clone());
    let r = analyze(&mut s2, json!({"items": [item2.0], "wait": true}));
    assert_eq!(r["scenes"]["items"][0]["fromCache"], false, "{r}");
    let first = fake2.described.load(Ordering::Relaxed);
    assert!(first > 0);
    let r = analyze(&mut s2, json!({"items": [item2.0], "wait": true}));
    assert_eq!(r["scenes"]["items"][0]["fromCache"], true, "{r}");
    assert_eq!(fake2.described.load(Ordering::Relaxed), first, "nothing described again");
    // fresh: analysed again
    let r = analyze(&mut s2, json!({"items": [item2.0], "wait": true, "fresh": true}));
    assert_eq!(r["scenes"]["items"][0]["fromCache"], false);
    // other models (another describer): another key
    let r = analyze(&mut s2, json!({"items": [item2.0], "wait": true, "detail": "full"}));
    assert_eq!(r["scenes"]["items"][0]["fromCache"], false);
    assert_eq!(s2.project.scenes.get(&item2).unwrap().detail, "full");
}

/// A session holding a short H.264 movie of a demo scene (a file the cache can key).
fn session_from_movie() -> (Session, ItemId) {
    let path = tmp_dir("scenes-movie").join("demo.mp4");
    crate::media_test_util::make_movie(&path, filmcraft_media::DemoScene::Plasma, 160, 90, 24);
    let mut s = Session::default();
    let r = s.execute("file.import", json!({"paths": [path.to_string_lossy()]})).unwrap();
    (s, ItemId(r["items"][0].as_u64().unwrap()))
}

#[test]
fn a_photo_is_one_shot() {
    let path = tmp_dir("scenes-photo").join("p.png");
    frame(45).save(&path).unwrap();
    let mut s = Session::default();
    let r = s.execute("file.import", json!({"paths": [path.to_string_lossy()]})).unwrap();
    let item = ItemId(r["items"][0].as_u64().unwrap());
    s.scenes.models = Some(Arc::new(Fake::default()));
    analyze(&mut s, json!({"items": [item.0], "wait": true}));
    let idx = s.project.scenes.get(&item).unwrap();
    assert_eq!(idx.boundaries, "still");
    assert_eq!(idx.shots.len(), 1);
    assert_eq!(idx.shots[0].description.as_ref().unwrap().caption, "A blue pattern moves.");
}

#[test]
fn hostile_parameters_are_refused() {
    let (mut s, item) = session();
    s.scenes.models = Some(Arc::new(Fake::default()));
    for p in [
        json!({"items": "all"}),
        json!({"items": [-1]}),
        json!({"items": [999_999]}),
        json!({"detail": "ultra"}),
        json!({"items": (0..3000).collect::<Vec<u64>>()}),
    ] {
        assert!(s.execute("media.analyzeScenes", p.clone()).is_err(), "{p}");
    }
    assert!(s.execute("media.searchScenes", json!({})).is_err());
    assert!(s.execute("media.searchScenes", json!({"query": "   "})).is_err());
    assert!(s.execute("media.bestShots", json!({"purpose": "cake"})).is_err());
    assert!(s.execute("media.scenes", json!({})).is_err());
    assert!(s.execute("scenes.downloadModel", json!({"model": "gpt-5"})).is_err());
    // before any analysis: empty answers, not errors
    let v = s.execute("media.searchScenes", json!({"query": "robot", "limit": 1e12})).unwrap();
    assert_eq!(v["hits"], json!([]));
    let v = s.execute("media.bestShots", json!({"count": -5})).unwrap();
    assert_eq!(v["shots"], json!([]));
    let v = s.execute("media.scenes", json!({"item": item.0})).unwrap();
    assert_eq!(v["analyzed"], false);
    // a damaged index from a project file is brought into range before use
    let mut p = (*s.project).clone();
    let bad = filmcraft_project::scenes::SceneIndex {
        duration: filmcraft_time::Tick::from_seconds_f64(2.0),
        shots: vec![filmcraft_project::scenes::Shot {
            start: filmcraft_time::Tick::from_seconds_f64(5.0),
            end: filmcraft_time::Tick::from_seconds_f64(1.0),
            ..Default::default()
        }],
        ..Default::default()
    };
    p.scenes.insert(item, Arc::new(bad));
    s.project = Arc::new(p);
    let v = s.execute("media.scenes", json!({"item": item.0})).unwrap();
    assert_eq!(v["shots"], json!([]));
    assert!(s.execute("media.bestShots", json!({})).is_ok());
}

#[test]
fn status_lists_the_models_and_their_licences() {
    let mut s = Session::default();
    let v = s.execute("scenes.status", json!({})).unwrap();
    let models = v["models"].as_array().unwrap();
    assert!(models.iter().any(|m| m["id"] == "transnetv2" && m["license"] == "MIT"));
    assert!(models.iter().any(|m| m["id"] == "qwen3-vl-2b" && m["license"] == "Apache-2.0"));
    assert!(v["fast"]["missingBytes"].as_u64().is_some());
}

#[test]
fn a_cancelled_job_changes_nothing() {
    let (mut s, item) = session();
    s.scenes.models = Some(Arc::new(Fake::default()));
    let r = s.execute("media.analyzeScenes", json!({"items": [item.0]})).unwrap();
    let job = r["job"].as_u64().unwrap();
    s.execute("jobs.cancel", json!({"job": job})).unwrap();
    for _ in 0..500 {
        let l = s.execute("jobs.list", json!({})).unwrap();
        if l.as_array().unwrap().iter().any(|j| j["id"] == job && j["finished"] == true) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let l = s.execute("jobs.list", json!({})).unwrap();
    let j = l.as_array().unwrap().iter().find(|j| j["id"] == job).cloned().unwrap();
    assert_eq!(j["finished"], true, "{j}");
    // it may have finished before the cancel landed; a cancelled one stores nothing
    if j["scenes"]["error"] == "stopped" {
        assert!(s.project.scenes.is_empty());
    }
}
