//! `graphics.chapterBar` on an explainer laid out by `script.timeline`, on chapters given, and
//! hostile parameters refused.
use filmcraft_project::graphic::SHAPE_LAYER;
use filmcraft_project::{ClipId, ParamValue};
use serde_json::{Value, json};

use super::*;
use crate::Session;

const SCRIPT: &str = "# One\nThe first chapter speaks for a while, long enough to be seen.\n\nIt says a second thing too.\n\n\
# Two\nThe second is short.\n\n# Three\nThe third chapter is the longest of them all, by a good margin, with many words in it to say.\n";

fn explainer() -> (Session, Value) {
    let mut s = Session::default();
    let r = s.execute("script.timeline", json!({"text": SCRIPT, "estimate": true, "place": {"width": 640, "height": 360, "fps": 30}})).unwrap();
    (s, r)
}

fn color(e: &filmcraft_project::EffectInstance) -> [f32; 4] {
    match e.params.get("fill_color").map(|p| &p.value) {
        Some(ParamValue::Color(c)) => *c,
        other => panic!("{other:?}"),
    }
}

fn vec2(e: &filmcraft_project::EffectInstance, k: &str) -> (f64, f64) {
    match e.params.get(k).map(|p| &p.value) {
        Some(ParamValue::Vec2(v)) => (v.x, v.y),
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_bar_matches_the_chapters_of_the_timeline() {
    let (mut s, timeline) = explainer();
    let undo0 = s.history.undo.len();
    let r = s.execute(CMD, json!({})).unwrap();
    let chapters = timeline["chapters"].as_array().unwrap();
    let clips = r["clips"].as_array().unwrap();
    // one clip a chapter, from its start to the next chapter's, the last to the sequence's end
    assert_eq!(clips.len(), chapters.len());
    let q = s.active_sequence().unwrap();
    let end = q.duration();
    for (i, (c, id)) in chapters.iter().zip(clips).enumerate() {
        let (_, it) = q.find_item(ClipId(id.as_u64().unwrap())).unwrap();
        assert_eq!(it.start.0, c["start"].as_i64().unwrap(), "chapter {i} starts with its clip");
        let want_end = chapters.get(i + 1).map_or(end.0, |n| n["start"].as_i64().unwrap());
        assert_eq!(it.end().0, want_end);
        // a segment a chapter, as wide as the chapter is long; the playing one in the accent colour
        let segs: Vec<&filmcraft_project::EffectInstance> = it.effects.iter().filter(|e| e.effect == SHAPE_LAYER).collect();
        assert_eq!(segs.len(), chapters.len());
        for (j, seg) in segs.iter().enumerate() {
            let want = match j.cmp(&i) {
                std::cmp::Ordering::Less => DONE,
                std::cmp::Ordering::Equal => CURRENT,
                std::cmp::Ordering::Greater => TRACK,
            };
            assert_eq!(color(seg), want, "clip {i} segment {j}");
        }
        let title = it.effects.iter().find(|e| e.effect == filmcraft_project::graphic::TEXT_LAYER).unwrap();
        assert_eq!(title.params["text"].value, ParamValue::Text(c["title"].as_str().unwrap().into()));
    }
    // widths in proportion to the chapters' lengths
    let segs = r["chapters"].as_array().unwrap();
    let width = |x: &Value| x["segment"][1].as_f64().unwrap() - x["segment"][0].as_f64().unwrap();
    let length = |x: &Value| (x["end"].as_i64().unwrap() - x["start"].as_i64().unwrap()) as f64;
    for pair in segs.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        assert!((width(a) / width(b) - length(a) / length(b)).abs() < 0.02, "{a} {b}");
        assert!(a["segment"][1].as_f64().unwrap() < b["segment"][0].as_f64().unwrap(), "segments do not overlap");
    }
    // on a new top track, landscape so at the bottom; one undo step
    assert_eq!(r["track"], format!("V{}", q.video_tracks.len()));
    assert_eq!(r["bar"]["position"], "bottom");
    assert_eq!(s.history.undo.len(), undo0 + 1);
    s.undo().unwrap();
    assert!(s.active_sequence().unwrap().video_tracks.iter().all(|t| t.items.is_empty()));
}

#[test]
fn chapters_given_progress_and_a_portrait_frame() {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"width": 360, "height": 640, "fps": 30})).unwrap();
    let secs = |x: f64| Tick::from_seconds_f64(x).0;
    let r = s
        .execute(CMD, json!({"chapters": [{"seconds": 0.0, "title": "Start"}, {"start": secs(4.0)}], "end": secs(10.0), "progress": true, "colors": {"current": "#00ff00"}}))
        .unwrap();
    assert_eq!(r["bar"]["position"], "top", "a portrait frame puts the bar at the top");
    let q = s.active_sequence().unwrap();
    let (_, first) = q.find_item(ClipId(r["clips"][0].as_u64().unwrap())).unwrap();
    // the playing segment is dim under a fill that grows over the chapter
    let fill = first.effects.iter().find(|e| matches!(e.params.get("name").map(|p| &p.value), Some(ParamValue::Text(t)) if t.ends_with("progress"))).unwrap();
    assert_eq!(color(fill), [0.0, 1.0, 0.0, 1.0]);
    let size = &fill.params["size"];
    assert_eq!(size.keyframes.len(), 2);
    let (ParamValue::Vec2(a), ParamValue::Vec2(b)) = (&size.keyframes[0].value, &size.keyframes[1].value) else { panic!() };
    assert!(a.x < 2.0 && b.x > 100.0, "{a:?} {b:?}");
    assert!(size.keyframes[1].time.0 < first.duration.0);
    // the untitled chapter gets no label; the segments sit inside the frame's margins
    let (_, second) = q.find_item(ClipId(r["clips"][1].as_u64().unwrap())).unwrap();
    assert!(second.effects.iter().all(|e| e.effect != filmcraft_project::graphic::TEXT_LAYER));
    let seg = second.effects.iter().find(|e| e.effect == SHAPE_LAYER).unwrap();
    let (x, y) = vec2(seg, "position");
    assert!(x > 0.0 && x < 360.0 && y < 640.0 * 0.1, "{x} {y}");
    // it renders: the accent fill part way along the first segment at 2 s
    let dir = std::env::temp_dir().join(format!("fc-chapterbar-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let png = dir.join("bar.png");
    s.execute("file.exportFrame", json!({"path": png.to_string_lossy(), "format": "png", "target": "program", "time": secs(2.0)})).unwrap();
    let img = image::open(&png).unwrap().to_rgba8();
    let row = (img.height() as f64 * 0.05 + 3.0) as u32;
    let green = (0..img.width()).filter(|x| matches!(img.get_pixel(*x, row).0, [r, g, b, _] if g > 200 && r < 80 && b < 80)).count();
    assert!(green > 20 && green < img.width() as usize / 2, "{green} green pixels on row {row}");
    let _ = std::fs::remove_dir_all(&dir);
    // a dry run plans without editing
    let undo = s.history.undo.len();
    let d = s.execute(CMD, json!({"chapters": [{"seconds": 0.0}], "end": secs(3.0), "dryRun": true})).unwrap();
    assert_eq!(d["dryRun"], true);
    assert_eq!(s.history.undo.len(), undo);
}

#[test]
fn hostile_parameters_are_refused_without_a_panic() {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"width": 640, "height": 360, "fps": 30})).unwrap();
    let secs = |x: f64| Tick::from_seconds_f64(x).0;
    let many: Vec<Value> = (0..70).map(|i| json!({"seconds": i as f64})).collect();
    let cases: Vec<(Value, &str)> = vec![
        (json!({}), "empty"),
        (json!({"end": secs(9.0)}), "no chapters"),
        (json!({"chapters": [], "end": secs(9.0)}), "no chapters"),
        (json!({"chapters": [{"seconds": 0.0}]}), "empty"),
        (json!({"chapters": [{"seconds": 5.0}, {"seconds": 2.0}], "end": secs(9.0)}), "in order"),
        (json!({"chapters": [{"seconds": 9.0}], "end": secs(9.0)}), "at or after the end"),
        (json!({"chapters": many, "end": secs(100.0)}), "at most 64"),
        (json!({"chapters": [{"start": -1}], "end": secs(9.0)}), "start"),
        (json!({"chapters": [{"seconds": f64::MAX}], "end": secs(9.0)}), "seconds"),
        (json!({"chapters": [{}], "end": secs(9.0)}), "give `start`"),
        (json!({"chapters": [{"seconds": 0.0, "title": 7}], "end": secs(9.0)}), "title"),
        (json!({"chapters": "all", "end": secs(9.0)}), "list"),
        (json!({"chapters": [{"seconds": 0.0}], "end": -4}), "end"),
        (json!({"chapters": [{"seconds": 0.0}], "end": secs(9.0), "position": "middle"}), "position"),
        (json!({"chapters": [{"seconds": 0.0}], "end": secs(9.0), "height": 1e9}), "height"),
        (json!({"chapters": [{"seconds": 0.0}], "end": secs(9.0), "colors": {"current": "orange"}}), "#rrggbb"),
        (json!({"chapters": [{"seconds": 0.0}], "end": secs(9.0), "colors": {"sky": "#000000"}}), "not a colour"),
        (json!({"chapters": [{"seconds": 0.0}], "end": secs(9.0), "track": "V99"}), "track"),
        (json!({"chapters": [{"seconds": 0.0}], "end": secs(9.0), "labels": 1}), "true or false"),
        (json!({"chapters": [{"seconds": 0.0}], "end": secs(9.0), "shiny": true}), "unknown parameter"),
    ];
    for (p, want) in cases {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute(CMD, p.clone())));
        let e = r.unwrap_or_else(|_| panic!("panicked on {p}")).err().unwrap_or_else(|| panic!("accepted {p}")).to_string();
        assert!(e.contains(want), "{p}: {e}");
    }
    // a track with a clip under the bar is refused
    s.execute("graphics.newText", json!({"text": "Busy", "seconds": 4.0, "time": 0, "track": "V1"})).unwrap();
    let e = s.execute(CMD, json!({"chapters": [{"seconds": 0.0}], "end": secs(9.0), "track": "V1"})).unwrap_err().to_string();
    assert!(e.contains("has clips under the bar"), "{e}");
}
