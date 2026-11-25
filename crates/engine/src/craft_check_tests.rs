//! `sequence.checkCraft` on sequences built here from generated stills, an animated GIF and a WAV
//! (no media is committed and no ffmpeg is needed).
use filmcraft_project::{Keyframe, Param, ParamValue};
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

fn codes(r: &Value) -> Vec<String> {
    r["findings"].as_array().unwrap().iter().filter_map(|f| f["code"].as_str().map(str::to_string)).collect()
}

/// A 640×360 still of one grey level, imported.
fn photo(s: &mut Session, dir: &std::path::Path, name: &str, grey: u8) -> filmcraft_project::ItemId {
    marked(s, dir, name, grey, None)
}

/// A 640×360 still of one grey level with a 40×40 square of `mark` in its top left corner (far
/// from where titles and captions go), imported.
fn marked(s: &mut Session, dir: &std::path::Path, name: &str, grey: u8, mark: Option<u8>) -> filmcraft_project::ItemId {
    let path = dir.join(name);
    let mut px = vec![grey; 640 * 360 * 3];
    if let Some(m) = mark {
        for y in 0..40 {
            for x in 0..40 {
                px[(y * 640 + x) * 3..(y * 640 + x) * 3 + 3].fill(m);
            }
        }
    }
    image::save_buffer(&path, &px, 640, 360, image::ColorType::Rgb8).unwrap();
    crate::commands::import_streamed(s, path.to_str().unwrap(), None).unwrap()
}

fn place(s: &mut Session, item: filmcraft_project::ItemId, track: &str, from: f64, to: f64) -> u64 {
    let r =
        s.execute("timeline.place", json!({"item": item.0, "track": track, "time": secs(from), "duration": secs(to) - secs(from), "audio": false})).unwrap();
    r["clips"][0].as_u64().or_else(|| r["clip"].as_u64()).unwrap_or(0)
}

/// A session with a 640×360 30 fps sequence (two video tracks, one audio track).
fn session(name: &str) -> (Session, std::path::PathBuf) {
    let dir = crate::media_test_util::tmp_dir(name);
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"width": 640, "height": 360, "fps": 30, "video": 2, "audio": 1})).unwrap();
    (s, dir)
}

/// A mono 48 kHz WAV of `seconds` of a quiet tone, imported.
fn sound(s: &mut Session, dir: &std::path::Path, seconds: f64) -> filmcraft_project::ItemId {
    let n = (seconds * 48_000.0) as u32;
    let mut b = Vec::with_capacity(44 + 2 * n as usize);
    b.extend(b"RIFF");
    b.extend((36 + 2 * n).to_le_bytes());
    b.extend(b"WAVEfmt ");
    b.extend(16u32.to_le_bytes());
    b.extend(1u16.to_le_bytes());
    b.extend(1u16.to_le_bytes());
    b.extend(48_000u32.to_le_bytes());
    b.extend(96_000u32.to_le_bytes());
    b.extend(2u16.to_le_bytes());
    b.extend(16u16.to_le_bytes());
    b.extend(b"data");
    b.extend((2 * n).to_le_bytes());
    for i in 0..n {
        let v = ((i as f64 * 440.0 * std::f64::consts::TAU / 48_000.0).sin() * 3000.0) as i16;
        b.extend(v.to_le_bytes());
    }
    let path = dir.join("tone.wav");
    std::fs::write(&path, b).unwrap();
    crate::commands::import_streamed(s, path.to_str().unwrap(), None).unwrap()
}

/// A 64×36 animated GIF at 10 fps whose picture moves all through `seconds`, imported.
fn movie(s: &mut Session, dir: &std::path::Path, seconds: f64) -> filmcraft_project::ItemId {
    let frames = (seconds * 10.0) as u32;
    let mut out = Vec::new();
    {
        let mut e = image::codecs::gif::GifEncoder::new(&mut out);
        let frames = (0..frames).map(|i| {
            let img = image::RgbaImage::from_fn(64, 36, |x, _| image::Rgba([((x + i * 7) % 64 * 4) as u8, 90, 40, 255]));
            image::Frame::from_parts(img, 0, 0, image::Delay::from_numer_denom_ms(100, 1))
        });
        e.encode_frames(frames).unwrap();
    }
    let path = dir.join("moving.gif");
    std::fs::write(&path, out).unwrap();
    crate::commands::import_streamed(s, path.to_str().unwrap(), None).unwrap()
}

fn check(s: &mut Session, p: Value) -> Value {
    s.execute(CMD, p).unwrap()
}

#[test]
fn contrast_ratios_follow_wcag() {
    assert!((ratio(1.0, 0.0) - 21.0).abs() < 1e-9);
    assert!((ratio(0.3, 0.3) - 1.0).abs() < 1e-9);
    assert!((luminance([1.0, 1.0, 1.0]) - 1.0).abs() < 1e-9);
    assert!(luminance([0.5, 0.5, 0.5]) > 0.2 && luminance([0.5, 0.5, 0.5]) < 0.22, "sRGB 50 % grey is about 0.214");
    let white = Look { rect: [0.0; 4], em: 40.0, text: vec![1.0], alpha: 1.0, plate: None, stroke: None, bold: false };
    assert!((worst_contrast(&white, 0.0, 0.0) - 21.0).abs() < 1e-9);
    // a backdrop as bright as the letters somewhere in the box
    assert_eq!(worst_contrast(&Look { text: vec![0.3], ..white.clone() }, 0.1, 0.6), 1.0);
    // a 75 % black box over white footage: 1.05 / 0.30
    let boxed = Look { plate: Some((0.0, 0.75)), ..white.clone() };
    assert!((worst_contrast(&boxed, 1.0, 1.0) - 3.5).abs() < 1e-9);
    // half-transparent white letters over black show at 0.5: 0.55 / 0.05
    assert!((worst_contrast(&Look { alpha: 0.5, ..white.clone() }, 0.0, 0.0) - 11.0).abs() < 1e-9);
    // a wide black outline is a backdrop of its own
    assert!((worst_contrast(&Look { stroke: Some(0.0), ..white.clone() }, 1.0, 1.0) - 21.0).abs() < 1e-9);
    // large text: 3.9 % of the short side, or 3.03 % when bold
    assert!(Look { em: 14.1, ..white.clone() }.large(360.0));
    assert!(!Look { em: 13.9, ..white.clone() }.large(360.0));
    assert!(Look { em: 11.0, bold: true, ..white }.large(360.0));
}

#[test]
fn backdrop_percentiles_read_only_the_box() {
    // left half black, right half white
    let mut img = filmcraft_render::Image::new(20, 10);
    for y in 0..10 {
        for x in 10..20 {
            let i = (y * 20 + x) * 4;
            img.px[i..i + 4].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        }
    }
    assert_eq!(backdrop(&img, [0.0, 0.0, 10.0, 10.0]), Some((0.0, 0.0)));
    assert_eq!(backdrop(&img, [10.0, 0.0, 20.0, 10.0]), Some((1.0, 1.0)));
    assert_eq!(backdrop(&img, [5.0, 0.0, 15.0, 10.0]), Some((0.0, 1.0)));
    // outside the picture, inverted, empty or not finite
    assert_eq!(backdrop(&img, [30.0, 0.0, 40.0, 10.0]), None);
    assert_eq!(backdrop(&img, [10.0, 10.0, 5.0, 5.0]), None);
    assert_eq!(backdrop(&img, [f64::NAN, 0.0, 5.0, 5.0]), None);
    assert_eq!(backdrop(&filmcraft_render::Image::new(0, 0), [0.0, 0.0, 5.0, 5.0]), None);
}

#[test]
fn a_short_small_title_without_contrast_is_flagged_and_a_good_one_passes() {
    let (mut s, dir) = session("craft-titles");
    let white = marked(&mut s, &dir, "white.png", 255, Some(0));
    let black = photo(&mut s, &dir, "black.png", 0);
    place(&mut s, white, "V1", 0.0, 2.0);
    place(&mut s, black, "V1", 2.0, 6.0);
    // white letters on white, 8 px on a 360 px short side, five words for one second
    let bad = s
        .execute("graphics.newText", json!({"text": "Hello there my good friends", "size": 8, "seconds": 1.0, "time": secs(0.5), "track": "V2"}))
        .unwrap()["clip"]
        .as_u64()
        .unwrap();
    // white on black, 40 px, one word for three seconds
    let good =
        s.execute("graphics.newText", json!({"text": "Hi", "size": 40, "seconds": 3.0, "time": secs(2.5), "track": "V2"})).unwrap()["clip"].as_u64().unwrap();
    let r = check(&mut s, json!({}));
    for code in ["readingTime", "smallText", "lowContrast"] {
        let hits = with_code(&r, code);
        assert_eq!(hits.len(), 1, "{code}: {r}");
        assert_eq!(hits[0]["clip"], bad, "{code}: {r}");
        assert_eq!(hits[0]["severity"], "warn");
    }
    let reading = with_code(&r, "readingTime")[0];
    assert_eq!(reading["words"], 5);
    assert!((reading["needed"].as_f64().unwrap() - (5.0 / 3.0 + 0.5)).abs() < 0.001, "{reading}");
    assert_eq!(with_code(&r, "lowContrast")[0]["ratio"], 1.0, "{r}");
    assert!(r["findings"].as_array().unwrap().iter().all(|f| f["clip"] != good), "{r}");
    let contrast = &r["checked"]["contrast"];
    assert_eq!(contrast["texts"], 2, "{r}");
    assert_eq!(contrast["checked"], 2);
    assert_eq!(
        contrast["gpuFrames"].as_u64().unwrap() + contrast["cpuFrames"].as_u64().unwrap(),
        contrast["renders"].as_u64().unwrap() + 1,
        "the backdrops and the first frame"
    );
    // white on black: 21:1 at every sample
    let ratio_of = |clip: u64| contrast["ratios"].as_array().unwrap().iter().find(|v| v["clip"] == clip).map(|v| v["ratio"].as_f64().unwrap());
    assert_eq!((ratio_of(good), ratio_of(bad)), (Some(21.0), Some(1.0)), "{r}");
    // the same letters on a dark background box read again
    s.edit_sequence("Box", |q, _, _| {
        let it = q.video_tracks[1].items.iter_mut().find(|i| i.id.0 == bad).unwrap();
        let e = it.effects.iter_mut().find(|e| e.effect == filmcraft_project::graphic::TEXT_LAYER).unwrap();
        e.params.insert("background".into(), Param::new(ParamValue::Bool(true)));
        e.params.insert("background_color".into(), Param::new(ParamValue::Color([0.0, 0.0, 0.0, 1.0])));
        e.params.insert("background_opacity".into(), Param::new(ParamValue::Float(100.0)));
        Ok(())
    })
    .unwrap();
    let r = check(&mut s, json!({}));
    assert!(with_code(&r, "lowContrast").is_empty(), "{r}");
}

#[test]
fn captions_are_checked_for_reading_time_size_and_contrast() {
    let (mut s, dir) = session("craft-captions");
    let white = marked(&mut s, &dir, "white.png", 255, Some(0));
    place(&mut s, white, "V1", 0.0, 3.0);
    s.execute("captions.newTrack", json!({"format": "Subtitle"})).unwrap();
    s.execute("captions.add", json!({"track": "C1", "text": "one two three four five six seven eight nine", "seconds": 0.5, "durationSeconds": 2.0})).unwrap();
    // the default style: 5 % of the short side, white on a 75 % black box (3.5:1, large text)
    let r = check(&mut s, json!({}));
    assert_eq!(codes(&r), ["readingTime"], "{r}");
    assert_eq!(with_code(&r, "readingTime")[0]["track"], "C1");
    // no box: white on white; and small
    s.execute("captions.setStyle", json!({"track": "C1", "background": false, "size": 20})).unwrap();
    let r = check(&mut s, json!({}));
    for code in ["readingTime", "smallText", "lowContrast"] {
        assert_eq!(with_code(&r, code).len(), 1, "{code}: {r}");
    }
    assert_eq!(with_code(&r, "smallText")[0]["kind"], "caption");
    // a wide black outline carries the letters
    s.execute("captions.setStyle", json!({"track": "C1", "size": 54, "outline": 6, "outlineColor": "#000000"})).unwrap();
    let r = check(&mut s, json!({}));
    assert!(with_code(&r, "lowContrast").is_empty(), "{r}");
    // a hidden track is not burned in: its size and contrast are not judged, its reading time is
    s.execute("captions.setStyle", json!({"track": "C1", "outline": 0, "size": 20})).unwrap();
    let id = s.state.active_sequence.unwrap();
    let mut p = (*s.project).clone();
    p.sequence_mut(id).unwrap().caption_tracks[0].enabled = false;
    s.project = std::sync::Arc::new(p);
    let r = check(&mut s, json!({}));
    assert_eq!(codes(&r), ["readingTime"], "{r}");
}

#[test]
fn a_static_timeline_warns_unless_held_and_moving_pictures_or_keyframes_count_as_change() {
    let (mut s, dir) = session("craft-static");
    let grey = photo(&mut s, &dir, "grey.png", 128);
    let tone = sound(&mut s, &dir, 10.0);
    place(&mut s, grey, "V1", 0.0, 5.0);
    let moving = movie(&mut s, &dir, 5.0);
    place(&mut s, moving, "V1", 5.0, 10.0);
    s.execute("timeline.place", json!({"item": tone.0, "track": "A1", "time": 0, "duration": secs(10.0)})).unwrap();
    let r = check(&mut s, json!({}));
    let stat = with_code(&r, "staticTimeline");
    assert_eq!(stat.len(), 1, "{r}");
    assert_eq!((stat[0]["time"].as_i64(), stat[0]["end"].as_i64()), (Some(0), Some(secs(5.0))), "{r}");
    assert!(with_code(&r, "silence").is_empty(), "{r}");
    // declared as a hold
    let held = check(&mut s, json!({"holds": [{"start": 0, "end": secs(5.0), "reason": "title card"}]}));
    assert!(with_code(&held, "staticTimeline").is_empty(), "{held}");
    // half of it held: 2.5 s left, under 3 s
    let half = check(&mut s, json!({"holds": [{"start": 0, "end": secs(2.5), "reason": "title card"}]}));
    assert!(with_code(&half, "staticTimeline").is_empty(), "{half}");
    // a fade in on the photo over 4.5 s: it changes, and the first frame is empty
    s.edit_sequence("Fade in", |q, _, _| {
        let it = &mut q.video_tracks[0].items[0];
        let e = it.effects.iter_mut().find(|e| e.effect == "opacity").unwrap();
        let mut p = Param::new(ParamValue::Float(100.0));
        p.keyframes = vec![Keyframe::new(Tick::ZERO, ParamValue::Float(0.0)), Keyframe::new(Tick::from_seconds_f64(4.5), ParamValue::Float(100.0))];
        e.params.insert("opacity".into(), p);
        Ok(())
    })
    .unwrap();
    let r = check(&mut s, json!({}));
    assert!(with_code(&r, "staticTimeline").is_empty(), "{r}");
    let empty = with_code(&r, "emptyFirstFrame");
    assert_eq!(empty.len(), 1, "{r}");
    assert!(empty[0]["message"].as_str().unwrap().contains("opacity 0"), "{r}");
}

#[test]
fn long_silence_and_an_empty_or_black_opening_are_found() {
    let (mut s, dir) = session("craft-silence");
    let black = photo(&mut s, &dir, "black.png", 0);
    let moving = movie(&mut s, &dir, 20.0);
    let tone = sound(&mut s, &dir, 3.0);
    place(&mut s, moving, "V1", 1.0, 21.0);
    s.execute("timeline.place", json!({"item": tone.0, "track": "A1", "time": 0, "duration": secs(3.0)})).unwrap();
    let r = check(&mut s, json!({}));
    let quiet = with_code(&r, "silence");
    assert_eq!(quiet.len(), 1, "{r}");
    assert_eq!((quiet[0]["time"].as_i64(), quiet[0]["end"].as_i64()), (Some(secs(3.0)), Some(secs(21.0))));
    let empty = with_code(&r, "emptyFirstFrame");
    assert!(empty.len() == 1 && empty[0]["message"].as_str().unwrap().contains("nothing"), "{r}");
    // a muted track is silent too
    let id = s.state.active_sequence.unwrap();
    let mut p = (*s.project).clone();
    p.sequence_mut(id).unwrap().audio_tracks[0].muted = true;
    s.project = std::sync::Arc::new(p);
    let r = check(&mut s, json!({}));
    assert_eq!(with_code(&r, "silence")[0]["time"], 0);
    // a black photo at the first frame
    place(&mut s, black, "V1", 0.0, 1.0);
    let r = check(&mut s, json!({}));
    let empty = with_code(&r, "emptyFirstFrame");
    assert!(empty.len() == 1 && empty[0]["message"].as_str().unwrap().contains("black"), "{r}");
    // a plain grey photo: one flat colour
    let (mut t, dir) = session("craft-flat");
    let flat = photo(&mut t, &dir, "flat.png", 128);
    place(&mut t, flat, "V1", 0.0, 2.0);
    let r = check(&mut t, json!({}));
    assert!(with_code(&r, "emptyFirstFrame")[0]["message"].as_str().unwrap().contains("flat"), "{r}");
}

#[test]
fn hostile_parameters_are_refused() {
    let (mut s, dir) = session("craft-hostile");
    let grey = photo(&mut s, &dir, "grey.png", 128);
    place(&mut s, grey, "V1", 0.0, 2.0);
    let end = s.active_sequence().unwrap().duration().0;
    let mut many = Vec::new();
    for i in 0..2001i64 {
        many.push(json!({"start": i, "end": i + 1, "reason": "x"}));
    }
    for p in [
        json!([1]),
        json!("sequence"),
        json!({"sequence": "main"}),
        json!({"sequence": -1}),
        json!({"sequence": 1.5}),
        json!({"sequence": 987_654}),
        json!({"holds": "all"}),
        json!({"holds": Value::Array(many)}),
        json!({"holds": [{"start": 0, "end": end}]}),
        json!({"holds": [{"start": 0, "end": end, "reason": "  "}]}),
        json!({"holds": [{"start": -1, "end": end, "reason": "x"}]}),
        json!({"holds": [{"start": 0, "end": end + 1, "reason": "x"}]}),
        json!({"holds": [{"start": 5, "end": 5, "reason": "x"}]}),
    ] {
        assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
    }
    // a named sequence that is not the active one
    let seq = s.state.active_sequence.unwrap();
    s.execute("file.newSequence", json!({"width": 640, "height": 360, "fps": 30})).unwrap();
    assert!(s.execute(CMD, json!({})).is_err(), "the new active sequence is empty");
    assert_eq!(check(&mut s, json!({"sequence": seq.0}))["sequence"], seq.0);
    // no sequence at all
    assert!(Session::default().execute(CMD, json!({})).is_err());
    // the explainer switch is a boolean
    let (mut s, dir) = session("craft-hostile-explainer");
    let grey = photo(&mut s, &dir, "grey.png", 128);
    place(&mut s, grey, "V1", 0.0, 2.0);
    for v in [json!("yes"), json!(1), json!([true])] {
        assert!(s.execute(CMD, json!({"explainer": v})).unwrap_err().to_string().contains("true or false"), "{v}");
    }
}

/// A 640×360 card: a full-frame backdrop shape and a text of `size` px on V1 from `from` to `to`.
fn card(s: &mut Session, text: &str, size: f64, from: f64, to: f64) -> u64 {
    let c = s
        .execute(
            "graphics.newShape",
            json!({"shape": "rectangle", "position": [320, 180], "size": [640, 360], "seconds": to - from, "time": secs(from), "track": "V1"}),
        )
        .unwrap()["clip"]
        .as_u64()
        .unwrap();
    s.execute("graphics.set", json!({"clip": c, "layer": 0, "props": {"fill_color": "#1a2a3a"}})).unwrap();
    s.execute("graphics.newText", json!({"text": text, "size": size, "clip": c, "position": [60, 230]})).unwrap();
    c
}

#[test]
fn an_explainer_holds_no_still_over_three_seconds_even_under_changing_captions() {
    let (mut s, _dir) = session("craft-explainer-still");
    card(&mut s, "1903", 160.0, 0.0, 6.0);
    // captions change every 2 s: the timeline is never still, but the picture is
    s.execute("captions.newTrack", json!({})).unwrap();
    for (i, t) in ["The first flight.", "It lasted twelve seconds.", "Nobody believed it."].iter().enumerate() {
        s.execute("captions.add", json!({"text": t, "time": secs(i as f64 * 2.0), "durationSeconds": 2.0})).unwrap();
    }
    let plain = check(&mut s, json!({}));
    assert!(with_code(&plain, "staticTimeline").is_empty() && with_code(&plain, "longStill").is_empty(), "{plain:#}");
    let r = check(&mut s, json!({"explainer": true}));
    let f = with_code(&r, "longStill");
    assert_eq!(f.len(), 1, "{r:#}");
    assert_eq!((f[0]["time"].as_i64(), f[0]["end"].as_i64()), (Some(0), Some(secs(6.0))));
    assert_eq!(r["explainer"], true);
    assert!(r["skipped"].as_array().unwrap().iter().any(|x| x["check"] == "staticTimeline"));
    // a hold declared over it, or a slow push on the text, and it is no longer still
    let held = check(&mut s, json!({"explainer": true, "holds": [{"start": 0, "end": secs(6.0), "reason": "the year sits"}]}));
    assert!(with_code(&held, "longStill").is_empty(), "{held:#}");
}

#[test]
fn an_explainers_hero_takes_a_quarter_of_the_frame_height() {
    let (mut s, dir) = session("craft-explainer-hero");
    // a big year (hero), then a small label alone, then a photo filling the frame with a small title over it
    card(&mut s, "1903", 160.0, 0.0, 2.0);
    let small = card(&mut s, "a small label", 24.0, 2.0, 4.0);
    let photo_item = photo(&mut s, &dir, "plate.png", 90);
    place(&mut s, photo_item, "V1", 4.0, 6.0);
    s.execute("graphics.newText", json!({"text": "Kitty Hawk", "size": 24, "seconds": 2.0, "time": secs(4.0), "track": "V2"})).unwrap();
    let r = check(&mut s, json!({"explainer": true}));
    let f = with_code(&r, "smallHero");
    assert_eq!(f.len(), 1, "only the small label's shot: {r:#}");
    assert_eq!((f[0]["clip"].as_u64(), f[0]["time"].as_i64(), f[0]["end"].as_i64()), (Some(small), Some(secs(2.0)), Some(secs(4.0))));
    assert!(f[0]["share"].as_f64().unwrap() < HERO_SHARE);
    assert!(f[0]["message"].as_str().unwrap().contains("a small label"), "{}", f[0]["message"]);
    let heroes = &r["checked"]["heroes"];
    assert_eq!((heroes["measured"].as_u64(), heroes["pictures"].as_u64()), (Some(2), Some(1)), "{heroes}");
    // without the switch nothing is said about heroes
    assert!(with_code(&check(&mut s, json!({})), "smallHero").is_empty());
}
