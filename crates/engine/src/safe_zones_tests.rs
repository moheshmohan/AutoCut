use serde_json::{Value, json};

use super::Platform;
use crate::Session;

/// A 1080×1920 sequence with a caption track (`C1`) holding one caption.
fn vertical() -> Session {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"name": "Short", "width": 1080, "height": 1920, "fps": 30})).unwrap();
    s.execute("captions.newTrack", json!({"format": "Subtitle"})).unwrap();
    s.execute("captions.add", json!({"track": "C1", "text": "the words go here", "seconds": 0.5, "durationSeconds": 2.0})).unwrap();
    s
}

fn zones_of(v: &Value, kind: &str) -> Vec<String> {
    v["overlaps"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|o| o["kind"] == kind)
        .flat_map(|o| o["zones"].as_array().unwrap().iter().map(|z| z.as_str().unwrap().to_string()))
        .collect()
}

#[test]
fn the_view_command_turns_the_overlay_on_and_off() {
    let mut s = vertical();
    assert_eq!(s.state.safe_zones, None);
    let r = s.execute("view.safeZones", json!({"platform": "TikTok"})).unwrap();
    assert_eq!(s.state.safe_zones, Some(Platform::TikTok));
    assert_eq!((r["platform"].as_str(), r["label"].as_str(), r["monitor"].as_str()), (Some("tiktok"), Some("TikTok"), Some("program")));
    assert_eq!(r["frame"], json!([1080, 1920]));
    let zones: Vec<&str> = r["zones"].as_array().unwrap().iter().map(|z| z["name"].as_str().unwrap()).collect();
    assert_eq!(zones, ["top", "bottom", "left", "right", "buttons"]);
    // the bottom zone of a 1080×1920 frame: its lower 35 %
    let bottom = &r["zones"][1]["rect"];
    assert_eq!(bottom, &json!([0.0, 1248.0, 1080.0, 1920.0]));
    // without a platform it says which is on; the Source monitor's is the frontend's
    assert_eq!(s.execute("view.safeZones", json!({})).unwrap()["platform"], "tiktok");
    let src = s.execute("view.safeZones", json!({"platform": "reels", "monitor": "source"})).unwrap();
    assert_eq!((src["platform"].as_str(), s.state.safe_zones), (Some("reels"), Some(Platform::TikTok)));
    for off in [json!(null), json!("off"), json!(false)] {
        s.state.safe_zones = Some(Platform::Youtube);
        let r = s.execute("view.safeZones", json!({"platform": off})).unwrap();
        assert_eq!((r["platform"].clone(), s.state.safe_zones), (Value::Null, None), "{off}");
    }
    // the state is saved with the session's editor state
    s.state.safe_zones = Some(Platform::Vertical);
    let state = serde_json::to_value(&s.state).unwrap();
    assert_eq!(state["safe_zones"], "vertical");
    let back: crate::EditorState = serde_json::from_value(state).unwrap();
    assert_eq!(back.safe_zones, Some(Platform::Vertical));
}

#[test]
fn a_caption_in_the_bottom_zone_is_flagged_and_a_safe_one_passes() {
    let mut s = vertical();
    // the big outlined style shorts had before: its bottom 24 % up the frame
    s.execute("captions.setStyle", json!({"track": "C1", "size": 76, "background": false, "outline": 7, "anchor": "bottom", "margin": 0.24})).unwrap();
    for p in ["youtube", "reels", "tiktok", "vertical"] {
        let r = s.execute("sequence.checkSafeZones", json!({"platform": p})).unwrap();
        assert_eq!(r["ok"], false, "{p}: {r}");
        assert!(zones_of(&r, "caption").contains(&"bottom".to_string()), "{p}: {r}");
        let o = &r["overlaps"][0];
        assert_eq!((o["track"].as_str(), o["text"].as_str(), o["seconds"].as_f64()), (Some("C1"), Some("the words go here"), Some(0.5)));
        assert!(o["share"].as_f64().unwrap() > 0.9, "{o}");
    }
    // raised above the bottom zone: safe on YouTube, but too wide for every platform at once
    // (TikTok's side margins and button column)
    s.execute("captions.setStyle", json!({"track": "C1", "margin": 0.38})).unwrap();
    let r = s.execute("sequence.checkSafeZones", json!({"platform": "youtube"})).unwrap();
    assert_eq!((r["ok"].as_bool(), r["checked"].as_u64()), (Some(true), Some(1)), "{r}");
    assert_eq!(r["vertical"], true);
    let r = s.execute("sequence.checkSafeZones", json!({"platform": "vertical"})).unwrap();
    assert!(zones_of(&r, "caption").iter().all(|z| z == "right" || z == "buttons") && r["ok"] == false, "{r}");
    s.execute("captions.setStyle", json!({"track": "C1", "size": 60})).unwrap();
    let r = s.execute("sequence.checkSafeZones", json!({"platform": "vertical"})).unwrap();
    assert_eq!(r["ok"], true, "{r}");
    // a hidden caption track is not burned in, so not checked
    s.execute("captions.setStyle", json!({"track": "C1", "margin": 0.0})).unwrap();
    let mut p = (*s.project).clone();
    let id = s.state.active_sequence.unwrap();
    p.sequence_mut(id).unwrap().caption_tracks[0].enabled = false;
    s.project = std::sync::Arc::new(p);
    assert_eq!(s.execute("sequence.checkSafeZones", json!({})).unwrap()["checked"], 0);
}

#[test]
fn titles_in_the_top_zone_are_flagged_where_motion_puts_them() {
    let mut s = vertical();
    s.execute("captions.deleteTrack", json!({"track": "C1"})).ok();
    let top = s.execute("graphics.newText", json!({"text": "Title", "position": [400, 120], "size": 80, "seconds": 2})).unwrap()["clip"].as_u64().unwrap();
    let r = s.execute("sequence.checkSafeZones", json!({"platform": "youtube"})).unwrap();
    let hit = r["overlaps"].as_array().unwrap().iter().find(|o| o["clip"] == top).cloned().unwrap_or_else(|| panic!("{r}"));
    assert_eq!(hit["kind"], "graphic");
    assert!(hit["zones"].as_array().unwrap().contains(&json!("top")), "{hit}");
    // moved down into the middle with its Motion: safe
    s.execute("effects.setParam", json!({"clip": top, "effect": "motion", "param": "position", "value": [540.0, 1700.0]})).unwrap();
    let r = s.execute("sequence.checkSafeZones", json!({"platform": "youtube"})).unwrap();
    assert!(r["overlaps"].as_array().unwrap().iter().all(|o| o["clip"] != top), "{r}");
}

#[test]
fn hostile_parameters_are_refused() {
    let mut s = vertical();
    for p in [
        json!({"platform": 5}),
        json!({"platform": "snapchat"}),
        json!({"platform": "x".repeat(10_000)}),
        json!({"platform": ["youtube"]}),
        json!({"monitor": "left"}),
        json!({"monitor": 2}),
        json!([1, 2]),
        json!("youtube"),
    ] {
        assert!(s.execute("view.safeZones", p.clone()).is_err(), "view.safeZones {p}");
        if p.get("monitor").is_none() {
            assert!(s.execute("sequence.checkSafeZones", p.clone()).is_err(), "sequence.checkSafeZones {p}");
        }
    }
    for p in [json!({"platform": null}), json!({"sequence": "main"}), json!({"sequence": 987_654}), json!({"sequence": -1})] {
        assert!(s.execute("sequence.checkSafeZones", p.clone()).is_err(), "{p}");
    }
    // nothing open: the view still answers; the check has nothing to check
    let mut empty = Session::default();
    assert_eq!(empty.execute("view.safeZones", json!({"platform": "reels"})).unwrap()["frame"], json!([1080, 1920]));
    assert!(empty.execute("sequence.checkSafeZones", json!({})).is_err());
    // zero-sized frames, captions at hostile sizes
    let mut q = s.active_sequence().unwrap().clone();
    if let Some(t) = q.caption_tracks.first_mut() {
        t.style.size = f32::MAX;
        t.style.margin = f32::NAN;
        t.style.outline = f32::INFINITY;
        t.style.line_spacing = -3.0;
    }
    let v = super::check(&s.project, &q, Platform::Vertical);
    assert_eq!(v["checked"], 1, "{v}");
    q.settings.width = 0;
    q.settings.height = 1;
    let v = super::check(&s.project, &q, Platform::Vertical);
    assert!(v["checked"].as_u64().is_some());
    assert_eq!(super::zones_json(None, (0, 0)), json!({"platform": null}));
}
