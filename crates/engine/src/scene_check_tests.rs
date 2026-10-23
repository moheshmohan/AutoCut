use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use filmcraft_project::ItemId;
use serde_json::{Value, json};

use super::*;
use crate::Session;
use crate::media_test_util::tmp_dir;
use crate::scene_models::{Calibration, Pic, SceneModels};

/// A clip of two halves: red for 20 frames, then blue.
fn session() -> (Session, ItemId) {
    let dir = tmp_dir("check-text-clip");
    for k in 0..40u32 {
        let c = if k < 20 { image::Rgba([210, 30, 20, 255]) } else { image::Rgba([20, 40, 220, 255]) };
        image::RgbaImage::from_pixel(96, 64, c).save(dir.join(format!("clip{k:04}.png"))).unwrap();
    }
    let mut s = Session::default();
    let r = s.execute("file.importImageSequence", json!({"path": dir.join("clip0000.png").to_string_lossy()})).unwrap();
    (s, ItemId(r["items"][0].as_u64().unwrap()))
}

fn mean_rgb(p: &Pic) -> Vec<f32> {
    let mut acc = [0f32; 3];
    let n = (p.rgba.len() / 4).max(1) as f32;
    for px in p.rgba.chunks(4) {
        for c in 0..3 {
            acc[c] += f32::from(px[c]) / 255.0;
        }
    }
    acc.iter().map(|v| v / n).collect()
}

/// Stub models: pictures embedded by their mean colour, words by the colour they name; the
/// "describer" says yes when the text names the frames' colour.
#[derive(Default)]
struct Fake {
    asked: AtomicUsize,
    released: AtomicUsize,
}

impl SceneModels for Fake {
    fn embedder_id(&self) -> Option<String> {
        Some("fake-emb".into())
    }
    fn asker_id(&self) -> Option<String> {
        Some("fake-vlm".into())
    }
    fn embed_images(&self, pics: &[Pic]) -> std::result::Result<Vec<Vec<f32>>, String> {
        Ok(pics.iter().map(mean_rgb).collect())
    }
    fn embed_text(&self, q: &str) -> std::result::Result<(Vec<f32>, Option<Calibration>), String> {
        let q = q.to_lowercase();
        let v = if q.contains("red") {
            vec![1.0, 0.1, 0.1]
        } else if q.contains("blue") {
            vec![0.1, 0.1, 1.0]
        } else {
            vec![0.3, 1.0, 0.3]
        };
        Ok((v, Some(Calibration { scale: 10.0, bias: -5.0 })))
    }
    fn ask(&self, frames: &[Pic], system: &str, question: &str, _: usize) -> std::result::Result<String, String> {
        self.asked.fetch_add(1, Ordering::Relaxed);
        assert!(system.contains("yes or no") && question.contains("will be shown over it"), "{question}");
        let [r, _, b] = [mean_rgb(&frames[0])[0], 0.0, mean_rgb(&frames[0])[2]];
        let colour = if r > b { "red" } else { "blue" };
        // a question about one thing of the text: that thing; else the whole text
        let about = question.split("is the \"").nth(1).and_then(|x| x.split('"').next()).unwrap_or(question);
        Ok(if about.to_lowercase().contains(colour) { "Yes.".into() } else { "No".into() })
    }
    fn release(&self) {
        self.released.fetch_add(1, Ordering::Relaxed);
    }
}

fn seconds_of(s: &Session, item: ItemId) -> f64 {
    s.project.item(item).unwrap().as_media().unwrap().duration().seconds()
}

#[test]
fn words_are_matched_to_the_frames_under_them_and_asked_about() {
    let (mut s, item) = session();
    let fake = Arc::new(Fake::default());
    s.scenes.models = Some(fake.clone());
    let d = seconds_of(&s, item);
    let undo = s.history.undo.len();
    let r = s
        .execute(
            "media.checkText",
            json!({
                "windows": [{"item": item.0, "from": 0.0, "to": d * 0.4}, {"item": item.0, "from": d * 0.6, "to": d}],
                "texts": ["A RED ROBOT", "BLUE ROBOT"],
                "asks": [{"window": 0, "text": 0}, {"window": 0, "text": 1}, {"window": 1, "text": 1}],
                "wait": true,
            }),
        )
        .unwrap();
    let v = &r["scenes"];
    assert_eq!(v["embedder"], "fake-emb", "{r}");
    assert_eq!(v["windows"][0]["frames"], 2);
    let cos = |t: usize, w: usize| v["cosine"][t][w].as_f64().unwrap();
    assert!(cos(0, 0) > cos(0, 1) && cos(1, 1) > cos(1, 0), "each text is closest to its own frames: {}", v["cosine"]);
    let m = v["match"][0][0].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&m) && m > v["match"][0][1].as_f64().unwrap());
    // the two halves look nothing alike; each window is itself
    assert!(v["similar"][0][1].as_f64().unwrap() < 0.5 && v["similar"][1][1].as_f64().unwrap() > 0.99, "{}", v["similar"]);
    let answers: Vec<&str> = v["answers"].as_array().unwrap().iter().map(|a| a["answer"].as_str().unwrap()).collect();
    assert_eq!(answers, ["yes", "no", "yes"]);
    assert_eq!(fake.asked.load(Ordering::Relaxed), 3);
    assert_eq!(fake.released.load(Ordering::Relaxed), 1, "the describer is let go after the asks");
    // nothing in the project changed
    assert_eq!(s.history.undo.len(), undo);
}

#[test]
fn without_models_it_says_so() {
    let (mut s, item) = session();
    s.scenes.models = Some(Arc::new(crate::scene_models::NoModels));
    let r = s.execute("media.checkText", json!({"windows": [{"item": item.0}], "texts": ["x"], "asks": [{"window": 0, "text": 0}], "wait": true})).unwrap();
    let v = &r["scenes"];
    assert!(v["cosine"].is_null() && v["answers"].as_array().unwrap().is_empty() && v["asker"].is_null(), "{v}");
    let notes: Vec<&str> = v["notes"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    assert!(notes.iter().any(|n| n.contains("SigLIP")) && notes.iter().any(|n| n.contains("nothing was asked")), "{notes:?}");
}

/// The frames are what the edit shows: a 16:9 picture in a 9:16 frame is its middle third, around
/// the focus; an ask about one thing names it.
#[test]
fn frames_are_cut_to_the_edit_and_asks_name_one_thing() {
    assert_eq!(crop_box(1920, 1080, 9.0 / 16.0, [0.5, 0.5]), [656, 0, 608, 1080]);
    assert_eq!(crop_box(1920, 1080, 9.0 / 16.0, [0.0, 0.5]), [0, 0, 608, 1080], "kept inside the picture");
    assert_eq!(crop_box(1920, 1080, 9.0 / 16.0, [1.0, 0.5]), [1312, 0, 608, 1080]);
    assert_eq!(crop_box(1080, 1920, 9.0 / 16.0, [0.2, 0.2]), [0, 0, 1080, 1920], "the same shape: whole");
    assert_eq!(crop_box(1080, 1920, 16.0 / 9.0, [0.5, 0.0]), [0, 0, 1080, 608]);
    for (w, h, a, f) in
        [(0, 0, 0.5, [0.5, 0.5]), (10, 10, f64::NAN, [0.5, 0.5]), (10, 10, -1.0, [0.5, 0.5]), (7, 3, 0.5, [f64::INFINITY, f64::NAN]), (1, 1, 1e9, [2.0, -3.0])]
    {
        let [x, y, cw, ch] = crop_box(w, h, a, f);
        assert!(x + cw <= w && y + ch <= h, "{w}x{h} {a} {f:?} → {x} {y} {cw} {ch}");
    }
    // the ask names the thing and the text it is in
    let q = ask_question("BOY MEETS ROBOT", Some("boy"));
    assert!(q.contains("\"boy\"") && q.contains("BOY MEETS ROBOT") && q.contains("yes or no"), "{q}");
    assert!(ask_question("A \"quoted\" title", Some(&"x".repeat(500))).chars().count() < 600);
    let (mut s, item) = session();
    s.scenes.models = Some(Arc::new(Fake::default()));
    let d = seconds_of(&s, item);
    let r = s
        .execute(
            "media.checkText",
            json!({"windows": [{"item": item.0, "from": 0.0, "to": d * 0.4, "focus": [0.3, 0.5]}], "aspect": 0.5625, "texts": ["RED ROBOT"], "asks": [{"window": 0, "text": 0, "about": "red"}, {"window": 0, "text": 0, "about": "blue"}], "wait": true}),
        )
        .unwrap();
    let a = r["scenes"]["answers"].as_array().unwrap();
    assert_eq!((a[0]["about"].as_str(), a[0]["answer"].as_str()), (Some("red"), Some("yes")));
    assert_eq!(a[1]["answer"], "no");
    for bad in [json!({"aspect": 0.0}), json!({"aspect": "tall"}), json!({"aspect": 1e9})] {
        let mut p = json!({"windows": [{"item": item.0}]});
        p["aspect"] = bad["aspect"].clone();
        assert!(s.execute("media.checkText", p).is_err(), "{bad}");
    }
    for focus in [json!([0.5]), json!("middle"), json!([0.5, "x"])] {
        assert!(s.execute("media.checkText", json!({"windows": [{"item": item.0, "focus": focus}]})).is_err(), "{focus}");
    }
    assert!(s.execute("media.checkText", json!({"windows": [{"item": item.0}], "texts": ["a"], "asks": [{"window": 0, "text": 0, "about": 3}]})).is_err());
}

#[test]
fn answers_are_read_defensively() {
    assert_eq!(read_answer(" Yes, it shows a robot."), "yes");
    assert_eq!(read_answer("NO."), "no");
    assert_eq!(read_answer(""), "unclear");
    assert_eq!(read_answer("maybe"), "unclear");
    assert_eq!(read_answer("\u{0}\u{1}yes"), "unclear");
    assert!(ask_question(&"\"".repeat(1000), None).chars().count() < 600, "quotes are not passed through and the text is capped");
}

/// Hostile parameters are refused with a message, never a panic.
#[test]
fn hostile_parameters_are_refused() {
    let (mut s, item) = session();
    s.scenes.models = Some(Arc::new(Fake::default()));
    let i = item.0;
    let many: Vec<Value> = (0..MAX_WINDOWS + 1).map(|_| json!({"item": i})).collect();
    let texts: Vec<String> = (0..MAX_TEXTS + 1).map(|k| format!("t{k}")).collect();
    for bad in [
        json!({}),
        json!({"windows": []}),
        json!({"windows": "x"}),
        json!({"windows": many}),
        json!({"windows": [{"from": 0}]}),
        json!({"windows": [{"item": 999_999}]}),
        json!({"windows": [{"item": i, "from": f64::MAX}]}),
        json!({"windows": [{"item": i, "from": -1.0}]}),
        json!({"windows": [{"item": i, "to": "soon"}]}),
        json!({"windows": [{"item": i}], "texts": texts}),
        json!({"windows": [{"item": i}], "texts": [""]}),
        json!({"windows": [{"item": i}], "texts": [7]}),
        json!({"windows": [{"item": i}], "texts": ["a"], "asks": [{"window": 1, "text": 0}]}),
        json!({"windows": [{"item": i}], "texts": ["a"], "asks": [{"window": 0, "text": -1}]}),
        json!({"windows": [{"item": i}], "texts": ["a"], "asks": [{"window": 0}]}),
        json!({"windows": [{"item": i}], "frames": 0}),
        json!({"windows": [{"item": i}], "frames": 9}),
    ] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute("media.checkText", bad.clone())));
        assert!(matches!(r, Ok(Err(_))), "{bad} → {r:?}");
    }
    // a window past the end is clamped into the media, a reversed one turned round
    let r = s.execute("media.checkText", json!({"windows": [{"item": i, "from": 3600.0, "to": 0.5}], "texts": ["red"], "wait": true})).unwrap();
    let w = &r["scenes"]["windows"][0];
    assert!(w["from"].as_f64().unwrap() <= w["to"].as_f64().unwrap(), "{w}");
    assert!(w["to"].as_f64().unwrap() <= seconds_of(&s, item) + 1e-6);
    // a very long text is cut, not refused
    let r = s.execute("media.checkText", json!({"windows": [{"item": i}], "texts": ["x".repeat(100_000)], "wait": true})).unwrap();
    assert_eq!(r["scenes"]["texts"][0].as_str().unwrap().chars().count(), MAX_TEXT_CHARS);
}
