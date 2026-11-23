//! `autocut.fingerprint` on the demo film and sequences built here (nothing is rendered; each
//! test keeps its history in a data directory of its own).
use filmcraft_time::Tick;
use serde_json::{Value, json};

use super::*;
use crate::Session;

/// The demo project in a session whose data directory is its own (named by `tag`).
fn demo(tag: &str) -> (Session, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("filmcraft-fingerprint-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut s = Session { prefs_path: Some(dir.join("preferences.json")), ..Session::default() };
    s.execute("file.openDemoProject", json!({})).unwrap();
    (s, dir)
}

fn axes(r: &Value) -> Vec<&str> {
    AXES.iter().map(|a| r["axes"][a].as_str().unwrap()).collect()
}

fn end_seconds(s: &Session) -> f64 {
    s.active_sequence().unwrap().duration().seconds()
}

fn title(s: &mut Session, text: &str, at: f64, seconds: f64) -> u64 {
    let r = s.execute("graphics.newText", json!({"text": text, "size": 60, "seconds": seconds, "time": Tick::from_seconds_f64(at).0, "track": "V3"})).unwrap();
    r["clip"].as_u64().unwrap()
}

#[test]
fn the_demo_film_reads_on_six_axes_with_evidence() {
    let (mut s, dir) = demo("axes");
    let r = s.execute(CMD, json!({})).unwrap();
    // footage from the first frame, no titles, dissolves between pictures (two dissolves and a
    // push), no look (the demo's Lumetri is a plain correction), no captions, a plain last cut
    assert_eq!(axes(&r), ["coldOpen", "none", "dissolve", "none", "none", "cut"], "{r:#}");
    assert!(r["evidence"]["transitionFamily"].as_str().unwrap().starts_with("2 of 3"), "{r:#}");
    assert_eq!((r["recorded"].as_bool(), r["matches"].as_array().map(Vec::len)), (Some(false), Some(0)));

    // a title in the first second that pops in; the colour pass's warm look; clean captions
    let open = title(&mut s, "Hello", 0.0, 3.0);
    s.execute("graphics.animateText", json!({"clip": open, "in": "pop"})).unwrap();
    let pictures: Vec<u64> = s.active_sequence().unwrap().video_tracks[0].items.iter().map(|i| i.id.0).collect();
    s.execute("clip.autoColor", json!({"clips": pictures, "look": "warm", "frames": 1})).unwrap();
    s.execute("captions.newTrack", json!({})).unwrap();
    s.execute("playhead.set", json!({"seconds": 2.0})).unwrap();
    s.execute("captions.add", json!({"text": "words"})).unwrap();
    s.execute("captions.setAnimatedStyle", json!({"preset": "clean"})).unwrap();
    let r = s.execute(CMD, json!({})).unwrap();
    assert_eq!(axes(&r)[..5], ["titleCard", "pop", "dissolve", "warm", "clean"], "{r:#}");
    assert!(r["evidence"]["opening"].as_str().unwrap().contains("over footage"), "{r:#}");

    // a closing title over the footage, then over the footage blurred: a title, then an end card
    let end = end_seconds(&s);
    let card = title(&mut s, "The end", end - 2.0, 2.0);
    assert_eq!(s.execute(CMD, json!({})).unwrap()["axes"]["close"], "title");
    let last = s.active_sequence().unwrap().video_tracks[0].items.last().unwrap().id.0;
    s.execute("effects.apply", json!({"clips": [last], "effect": "gaussian_blur"})).unwrap();
    let r = s.execute(CMD, json!({})).unwrap();
    assert_eq!(r["axes"]["close"], "endCard", "{r:#}");
    // without the title, the last picture fading to nothing
    s.execute("timeline.select", json!({"clips": [card]})).unwrap();
    s.execute("edit.clear", json!({})).unwrap();
    s.execute("sequence.applyVideoTransition", json!({"clip": last, "effect": "dip_to_black", "edge": "out"})).unwrap();
    assert_eq!(s.execute(CMD, json!({})).unwrap()["axes"]["close"], "fadeOut");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn openings_by_what_comes_first() {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"width": 320, "height": 180, "fps": 30, "video": 2, "audio": 1})).unwrap();
    let pool = s.media.clone();
    let mut ids = Vec::new();
    for k in 0..4 {
        let src = filmcraft_media::generators::GeneratorSource::new(
            filmcraft_media::generators::Generator::Demo(filmcraft_media::generators::DemoScene::Forest),
            64,
            36,
            filmcraft_time::FrameRate::FPS_30,
            Tick::from_seconds_f64(4.0),
        );
        ids.push(crate::demo::add_generator(
            std::sync::Arc::make_mut(&mut s.project),
            &pool,
            src,
            &format!("shot{k}.mov"),
            filmcraft_project::Label::Iris,
            None,
        ));
    }
    // four shots of a second each: a montage
    for (k, id) in ids.iter().enumerate() {
        s.execute(
            "timeline.place",
            json!({"item": id.0, "time": Tick::from_seconds_f64(k as f64).0, "track": "V1", "duration": Tick::from_seconds_f64(1.0).0}),
        )
        .unwrap();
    }
    let r = s.execute(CMD, json!({})).unwrap();
    assert_eq!(r["axes"]["opening"], "montage", "{r:#}");
    assert_eq!(r["axes"]["transitionFamily"], "cut");
    // the first shot's media heard speaking in its first second: a talking opening
    let tr = filmcraft_project::transcript::Transcript {
        language: "en".into(),
        source: "manual".into(),
        speakers: Vec::new(),
        words: vec![filmcraft_project::transcript::Word {
            text: "Hello".into(),
            start: Tick::from_seconds_f64(0.2),
            end: Tick::from_seconds_f64(0.6),
            speaker: None,
            confidence: 1.0,
        }],
    };
    std::sync::Arc::make_mut(&mut s.project).transcripts.insert(ids[0], std::sync::Arc::new(tr));
    assert_eq!(s.execute(CMD, json!({})).unwrap()["axes"]["opening"], "talking");
}

#[test]
fn recording_appends_and_four_shared_axes_warn() {
    let (mut s, dir) = demo("record");
    let r = s.execute(CMD, json!({"record": true, "pipeline": "direct", "deals": {"look": "warm", "endCard": "cut"}})).unwrap();
    assert_eq!(r["recorded"], true);
    assert!(r["note"].as_str().unwrap().contains("not an undo step"), "{r:#}");
    assert_eq!(r["record"]["pipeline"], "direct");
    assert_eq!(r["record"]["deals"], json!({"endCard": "cut", "look": "warm"}));
    assert!(r["record"]["when"].as_str().unwrap().ends_with('Z'));
    let lines = std::fs::read_to_string(dir.join(crate::taste_history::FILE)).unwrap();
    assert_eq!(lines.lines().count(), 1);
    // the same film again: all six axes match the last video, a warning (never an error)
    let r = s.execute(CMD, json!({})).unwrap();
    assert_eq!(r["matches"][0]["record"], 1);
    assert_eq!(r["matches"][0]["count"], 6);
    assert_eq!(r["warnings"][0]["code"], "sameAsRecent");
    assert_eq!(r["history"]["records"], 1);
    // only the three newest are compared
    for _ in 0..3 {
        s.execute(CMD, json!({"record": true})).unwrap();
    }
    let r = s.execute(CMD, json!({})).unwrap();
    assert_eq!((r["matches"].as_array().unwrap().len(), r["history"]["records"].as_u64()), (3, Some(4)));
    // a project file's name only, never its folders
    s.path = Some(dir.join("private").join("My Film.fcproj").display().to_string());
    let r = s.execute(CMD, json!({"record": true})).unwrap();
    assert_eq!(r["record"]["project"], "My Film.fcproj");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hostile_parameters_are_refused() {
    let (mut s, dir) = demo("bad");
    for p in [
        json!("x"),
        json!({"record": "yes"}),
        json!({"sequence": 999_999}),
        json!({"sequence": "main"}),
        json!({"pipeline": "a b"}),
        json!({"pipeline": "x".repeat(100)}),
        json!({"deals": {"mood": "calm"}}),
        json!({"deals": 3}),
        json!({"deals": {"look": ""}}),
        json!({"deals": {"look": 7}}),
        json!({"colour": true}),
    ] {
        assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
    }
    // a damaged history is read around, never an error
    std::fs::write(dir.join(crate::taste_history::FILE), b"{oops\n\x00\xff\n").unwrap();
    let r = s.execute(CMD, json!({})).unwrap();
    assert_eq!(r["history"]["records"], 0);
    assert!(!r["history"]["notes"].as_array().unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn transitions_fall_into_their_catalog_families() {
    for (id, family) in [
        ("cross_dissolve", "dissolve"),
        ("dip_to_black", "dissolve"),
        ("crt_mix", "dissolve"),
        ("cam_whip_pan", "whip"),
        ("crt_swipe_left", "whip"),
        ("cam_punch_in", "zoom"),
        ("crt_pull_in", "zoom"),
        ("gl_crosszoom", "zoom"),
        ("cross_zoom", "zoom"),
        ("zoom_blur", "zoom"),
        ("flash", "light"),
        ("whip", "whip"),
        ("lf_light_leak", "light"),
        ("crt_glitch", "glitch"),
        ("gl_cube", "spatial"),
        ("push", "slide"),
        ("gl_ripple", "distortion"),
        ("no_such_thing", "other"),
    ] {
        assert_eq!(family_of(id), family, "{id}");
    }
    assert_eq!(slug("Teal & Orange"), "tealOrange");
    assert_eq!(slug("Look: VHS"), "lookVhs");
}
