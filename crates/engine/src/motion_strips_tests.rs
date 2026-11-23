//! `sequence.motionStrips` on sequences built here from generated stills and titles, rendered
//! on the CPU reference renderer (no GPU, no media committed, no ffmpeg).
use filmcraft_time::Tick;
use serde_json::{Value, json};

use super::*;
use crate::Session;

fn secs(s: f64) -> i64 {
    Tick::from_seconds_f64(s).0
}

fn with_code<'a>(r: &'a Value, code: &str) -> Vec<&'a Value> {
    r["findings"].as_array().unwrap().iter().filter(|f| f["code"] == code).collect()
}

/// A session with a 320×180 30 fps sequence (two video tracks) and a folder for its files.
fn session(name: &str) -> (Session, std::path::PathBuf) {
    let dir = crate::media_test_util::tmp_dir(name);
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"width": 320, "height": 180, "fps": 30, "video": 2, "audio": 1})).unwrap();
    (s, dir)
}

/// A `w`×`h` still of one colour (`alpha` below 255 makes it an overlay), imported and placed on
/// `track` from `from` to `to` seconds: its clip id.
#[allow(clippy::too_many_arguments)]
fn still(s: &mut Session, dir: &std::path::Path, name: &str, (w, h): (u32, u32), rgba: [u8; 4], track: &str, from: f64, to: f64) -> u64 {
    let path = dir.join(name);
    if rgba[3] == 255 {
        image::save_buffer(&path, &rgba[..3].repeat((w * h) as usize), w, h, image::ColorType::Rgb8).unwrap();
    } else {
        image::save_buffer(&path, &rgba.repeat((w * h) as usize), w, h, image::ColorType::Rgba8).unwrap();
    }
    let item = crate::commands::import_streamed(s, path.to_str().unwrap(), None).unwrap();
    let r =
        s.execute("timeline.place", json!({"item": item.0, "track": track, "time": secs(from), "duration": secs(to) - secs(from), "audio": false})).unwrap();
    r["clips"][0].as_u64().or_else(|| r["clip"].as_u64()).unwrap()
}

fn title(s: &mut Session, text: &str, at: f64, seconds: f64) -> u64 {
    s.execute("graphics.newText", json!({"text": text, "size": 60, "seconds": seconds, "time": secs(at), "track": "V2"})).unwrap()["clip"].as_u64().unwrap()
}

fn strips(s: &mut Session, dir: &std::path::Path, folder: &str, extra: Value) -> Value {
    let mut p = json!({"out": dir.join(folder).to_str().unwrap(), "wait": true});
    for (k, v) in extra.as_object().cloned().unwrap_or_default() {
        p[k] = v;
    }
    s.execute(CMD, p).unwrap()
}

#[test]
fn every_animated_item_gets_a_seven_frame_sheet() {
    let (mut s, dir) = session("strips-sheets");
    let a = still(&mut s, &dir, "a.png", (320, 180), [60, 60, 60, 255], "V1", 0.0, 2.0);
    still(&mut s, &dir, "b.png", (320, 180), [200, 200, 200, 255], "V1", 2.0, 4.0);
    s.execute("sequence.applyVideoTransition", json!({"clip": a, "edge": "out", "effect": "cross_dissolve", "frames": 12})).unwrap();
    let t = title(&mut s, "Hi", 0.5, 1.2);
    s.execute("graphics.animateText", json!({"clip": t, "in": "pop", "out": {"kind": "fade", "duration": 0.3}})).unwrap();
    let r = strips(&mut s, &dir, "sheets", json!({}));
    let sheets = r["sheets"].as_array().unwrap();
    let kinds: Vec<(&str, Option<&str>)> = sheets.iter().map(|x| (x["item"]["kind"].as_str().unwrap(), x["item"]["edge"].as_str())).collect();
    assert_eq!(kinds, [("title", Some("in")), ("title", Some("out")), ("transition", None)], "{r:#}");
    for sheet in sheets {
        let path = sheet["path"].as_str().unwrap();
        let img = image::open(path).unwrap();
        assert_eq!((img.width(), img.height()), (sheet["width"].as_u64().unwrap() as u32, sheet["height"].as_u64().unwrap() as u32));
        assert!(img.width() > 7 * 320, "seven tiles of the whole frame side by side: {}", img.width());
        let roles: Vec<&str> = sheet["frames"].as_array().unwrap().iter().map(|f| f["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["before", "first", "quarter", "middle", "threeQuarters", "last", "settled"]);
    }
    // the entrance's frames: one frame before the clip, its first, … and settled after the move
    let f = &sheets[0]["frames"];
    let frame = Tick::from_seconds_f64(1.0 / 30.0).0;
    assert_eq!((f[0]["time"].as_i64().unwrap(), f[1]["time"].as_i64().unwrap()), (secs(0.5) - frame, secs(0.5)));
    assert!(f[6]["time"].as_i64().unwrap() > f[5]["time"].as_i64().unwrap());
    assert_eq!(r["checked"]["gpuFrames"], 0, "tests render on the CPU");
    assert!(r["checked"]["cpuFrames"].as_u64().unwrap() >= 21, "{r:#}");
    assert_eq!(r["ok"], true, "a quiet pop, a short fade and a dissolve: {r:#}");
    assert_eq!(r["blocking"], false);
    // the job is listed with its result
    let jobs = s.execute("jobs.list", json!({})).unwrap();
    let job = jobs.as_array().unwrap().iter().find(|j| j["id"] == r["job"]).unwrap();
    assert_eq!((job["finished"].as_bool(), job["motionStrips"]["sheets"].as_array().map(Vec::len)), (Some(true), Some(3)), "{job:#}");
    // `items` picks some; a folder with files in it is refused
    let r = strips(&mut s, &dir, "one", json!({"items": [{"clip": t, "edge": "out"}]}));
    assert_eq!(r["sheets"].as_array().unwrap().len(), 1);
    assert_eq!(r["sheets"][0]["item"]["edge"], "out");
    let e = s.execute(CMD, json!({"out": dir.join("one").to_str().unwrap(), "wait": true})).unwrap_err().to_string();
    assert!(e.contains("not empty"), "{e}");
}

#[test]
fn a_move_that_starts_with_a_jump_or_hitches_is_found() {
    let (mut s, dir) = session("strips-jumps");
    still(&mut s, &dir, "bg.png", (320, 180), [40, 40, 40, 255], "V1", 0.0, 4.0);
    // a title fully there at its first frame, then growing slowly: a jump, not an ease
    let t = title(&mut s, "WWW", 0.5, 2.0);
    s.execute(
        "keyframes.choreograph",
        json!({"target": {"clip": t, "effect": "motion"}, "tweens": [{"prop": "scale", "from": 100, "to": 104, "duration": 0.6, "ease": "none", "position": 0}]}),
    )
    .unwrap();
    // an overlay sliding slowly, with one frame that leaps half the frame
    let o = still(&mut s, &dir, "panel.png", (160, 90), [255, 255, 255, 220], "V2", 2.6, 4.0);
    s.execute(
        "keyframes.choreograph",
        json!({"target": {"clip": o, "effect": "motion"}, "tweens": [
            {"prop": "position", "from": [100, 90], "to": [110, 90], "duration": 0.6, "ease": "none", "position": 0},
            {"prop": "position", "to": [220, 90], "duration": 1.0 / 30.0, "ease": "none"},
            {"prop": "position", "to": [230, 90], "duration": 0.6, "ease": "none"},
        ]}),
    )
    .unwrap();
    let r = strips(&mut s, &dir, "jumps", json!({}));
    let jumps = with_code(&r, "jump");
    assert!(jumps.iter().any(|f| f["clip"] == t && f["edge"] == "start" && f["kind"] == "title"), "{r:#}");
    let steps = with_code(&r, "midStep");
    assert!(steps.iter().any(|f| f["clip"] == o && f["kind"] == "overlay"), "{r:#}");
    assert!(steps[0]["message"].as_str().unwrap().contains("smooth"), "says what to change");
    // eased from nothing over 12 frames, and the overlay without its leap: quiet
    let (mut s, dir) = session("strips-smooth");
    still(&mut s, &dir, "bg.png", (320, 180), [40, 40, 40, 255], "V1", 0.0, 4.0);
    let t = title(&mut s, "WWW", 0.5, 2.0);
    s.execute(
        "keyframes.choreograph",
        json!({"target": {"clip": t, "effect": "opacity"}, "tweens": [{"prop": "opacity", "from": 0, "to": 100, "duration": 0.4, "ease": "power2.out", "position": 0}]}),
    )
    .unwrap();
    let o = still(&mut s, &dir, "panel.png", (160, 90), [255, 255, 255, 220], "V2", 2.6, 4.0);
    s.execute(
        "keyframes.choreograph",
        json!({"target": {"clip": o, "effect": "motion"}, "tweens": [{"prop": "position", "from": [100, 90], "to": [230, 90], "duration": 1.3, "ease": "sine.inOut", "position": 0}]}),
    )
    .unwrap();
    // it fades in, so it does not pop in either
    s.execute(
        "keyframes.choreograph",
        json!({"target": {"clip": o, "effect": "opacity"}, "tweens": [{"prop": "opacity", "from": 0, "to": 100, "duration": 0.4, "ease": "power2.out", "position": 0}]}),
    )
    .unwrap();
    let r = strips(&mut s, &dir, "smooth", json!({}));
    assert!(with_code(&r, "jump").is_empty() && with_code(&r, "midStep").is_empty(), "{r:#}");
}

#[test]
fn an_exit_still_fading_over_the_next_shot_is_found() {
    let (mut s, dir) = session("strips-fade");
    still(&mut s, &dir, "a.png", (320, 180), [40, 40, 40, 255], "V1", 0.0, 2.0);
    still(&mut s, &dir, "b.png", (320, 180), [70, 70, 70, 255], "V1", 2.0, 4.0);
    // a 1 s fade out from 1.5 s, over the cut at 2 s
    let t = title(&mut s, "WWW", 0.5, 2.0);
    s.execute("graphics.animateText", json!({"clip": t, "out": {"kind": "fade", "unit": "line", "duration": 1.0}})).unwrap();
    let r = strips(&mut s, &dir, "late", json!({}));
    let f = with_code(&r, "fadeAfterShot");
    assert_eq!((f.len(), f.first().and_then(|f| f["cut"].as_i64())), (1, Some(secs(2.0))), "{r:#}");
    assert!(f[0]["message"].as_str().unwrap().contains("end the exit by the cut"));
    // ended before the cut: quiet
    s.execute("graphics.animateText", json!({"clip": t, "out": {"kind": "fade", "unit": "line", "duration": 0.3}})).unwrap();
    let r = strips(&mut s, &dir, "early", json!({}));
    assert!(with_code(&r, "fadeAfterShot").is_empty(), "the exit runs 2.2 to 2.5 s, after the cut: {r:#}");
}

#[test]
fn an_entrance_replayed_after_a_cut_is_found() {
    let (mut s, dir) = session("strips-replay");
    still(&mut s, &dir, "bg.png", (320, 180), [40, 40, 40, 255], "V1", 0.0, 4.0);
    // the same title cut in two, each half popping in
    for at in [0.0, 2.0] {
        let t = title(&mut s, "WWW", at, 2.0);
        s.execute("graphics.animateText", json!({"clip": t, "in": {"kind": "fade", "unit": "line", "duration": 0.4}})).unwrap();
    }
    let r = strips(&mut s, &dir, "replay", json!({}));
    let f = with_code(&r, "entranceReplay");
    assert_eq!((f.len(), f.first().and_then(|f| f["time"].as_i64())), (1, Some(secs(2.0))), "{r:#}");
    // after a cut in the footage, with nothing there before it: a first entrance, quiet
    let (mut s, dir) = session("strips-first");
    still(&mut s, &dir, "a.png", (320, 180), [40, 40, 40, 255], "V1", 0.0, 2.0);
    still(&mut s, &dir, "b.png", (320, 180), [45, 45, 45, 255], "V1", 2.0, 4.0);
    let t = title(&mut s, "WWW", 2.0, 2.0);
    s.execute("graphics.animateText", json!({"clip": t, "in": {"kind": "fade", "unit": "line", "duration": 0.4}})).unwrap();
    let r = strips(&mut s, &dir, "first", json!({}));
    assert!(with_code(&r, "entranceReplay").is_empty(), "{r:#}");
}

#[test]
fn the_detectors_read_grey_steps() {
    // a dissolve whose middle is darker than both ends; one that is not
    assert!(dip(&[0.5, 0.45, 0.4, 0.42, 0.55]).is_some_and(|(i, d)| i == 2 && (d - 0.1).abs() < 1e-9));
    assert!(dip(&[0.5, 0.5, 0.52, 0.55, 0.6]).is_none());
    assert!(dip(&[0.5, 0.5]).is_none(), "no middle");
    assert!(dip(&[]).is_none());
    assert_eq!(median(vec![f32::NAN, 3.0, 1.0, 2.0]), 2.0);
    assert_eq!(median(Vec::new()), 0.0);
}

#[test]
fn a_job_reports_progress_and_its_result_in_jobs_list() {
    let (mut s, dir) = session("strips-job");
    let t = title(&mut s, "Hi", 0.0, 2.0);
    s.execute("graphics.animateText", json!({"clip": t, "in": "fade"})).unwrap();
    let r = s.execute(CMD, json!({"out": dir.join("job").to_str().unwrap()})).unwrap();
    let id = r["job"].as_u64().unwrap();
    assert_eq!(r["poll"], "jobs.list");
    let mut job = Value::Null;
    for _ in 0..600 {
        let jobs = s.execute("jobs.list", json!({})).unwrap();
        job = jobs.as_array().unwrap().iter().find(|j| j["id"] == id).cloned().unwrap();
        if job["finished"] == true {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert_eq!(job["finished"], true, "{job:#}");
    assert!(job["total"].as_u64().unwrap() >= 7 && job["done"].as_u64().unwrap() >= 7, "{job:#}");
    assert_eq!(job["motionStrips"]["sheets"].as_array().map(Vec::len), Some(1), "{job:#}");
}

#[test]
fn hostile_parameters_are_refused() {
    let (mut s, dir) = session("strips-hostile");
    let out = dir.join("x").to_str().unwrap().to_string();
    // nothing moves yet
    let e = s.execute(CMD, json!({"out": out, "wait": true})).unwrap_err().to_string();
    assert!(e.contains("empty") || e.contains("moves"), "{e}");
    let t = title(&mut s, "Hi", 0.0, 2.0);
    s.execute("graphics.animateText", json!({"clip": t, "in": "fade"})).unwrap();
    std::fs::write(dir.join("file.txt"), b"x").unwrap();
    for p in [
        json!(3),
        json!({}),
        json!({"out": 5}),
        json!({"out": ""}),
        json!({"out": "a\u{0}b"}),
        json!({"out": out, "sequence": "main"}),
        json!({"out": out, "sequence": 999_999}),
        json!({"out": out, "items": "all"}),
        json!({"out": out, "items": []}),
        json!({"out": out, "items": [999_999]}),
        json!({"out": out, "items": [-1]}),
        json!({"out": out, "items": [{"transition": 999_999}]}),
        json!({"out": out, "items": [{"clip": t, "edge": "sideways"}]}),
        json!({"out": out, "items": [{"clip": t, "frame": 3}]}),
        json!({"out": out, "items": [{"clip": t, "transition": 1}]}),
        json!({"out": out, "items": vec![t; MAX_SELECTORS + 1]}),
        json!({"out": out, "scale": 0}),
        json!({"out": out, "scale": 2}),
        json!({"out": out, "scale": "big"}),
        json!({"out": out, "wait": "yes"}),
        json!({"out": out, "fps": 30}),
        json!({"out": dir.join("file.txt").to_str().unwrap(), "wait": true}),
    ] {
        assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
    }
    // the clip as a bare id, a long sequence of many items: capped, said so
    let r = strips(&mut s, &dir, "ok", json!({"items": [t], "scale": 0.05}));
    assert_eq!(r["sheets"].as_array().unwrap().len(), 1, "{r:#}");
}

#[test]
fn many_items_are_capped_and_said_so() {
    let (mut s, dir) = session("strips-many");
    for i in 0..(MAX_ITEMS + 3) {
        let t = title(&mut s, "x", i as f64 * 0.5, 0.4);
        s.execute("graphics.animateText", json!({"clip": t, "in": {"kind": "fade", "duration": 0.1}})).unwrap();
    }
    let r = strips(&mut s, &dir, "many", json!({"scale": 0.05}));
    assert_eq!(r["sheets"].as_array().unwrap().len(), MAX_ITEMS);
    assert!(r["skipped"].as_array().unwrap().iter().any(|k| k["reason"].as_str().unwrap().contains("3 more items")), "{r:#}");
}

#[test]
fn a_dip_through_black_is_not_a_hitch() {
    let (mut s, dir) = session("strips-dip");
    // from a dark shot into a bright one: the first step out of the black reads large in sRGB
    let a = still(&mut s, &dir, "a.png", (320, 180), [30, 30, 30, 255], "V1", 0.0, 2.0);
    still(&mut s, &dir, "b.png", (320, 180), [230, 230, 230, 255], "V1", 2.0, 4.0);
    s.execute("sequence.applyVideoTransition", json!({"clip": a, "edge": "out", "effect": "dip_to_black", "frames": 30})).unwrap();
    let r = strips(&mut s, &dir, "dip", json!({}));
    assert_eq!(r["sheets"].as_array().unwrap().len(), 1, "{r:#}");
    assert!(with_code(&r, "midStep").is_empty() && with_code(&r, "dissolveDip").is_empty(), "a dip is meant to go dark: {r:#}");
}
