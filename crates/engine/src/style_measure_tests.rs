//! `media.measureStyle` on footage made in code: sequences of the procedural demo scenes with
//! known cut points, stills with a jump cut, an animated GIF written here (no media is committed
//! and no ffmpeg is needed), and hostile parameters and files.
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{DemoScene, Generator};
use filmcraft_project::Label;
use filmcraft_time::{FrameRate, Tick};
use serde_json::json;

use super::*;
use crate::Session;

/// A session with the demo scenes as small generated clips (192×108, 30 fps, 12 s each).
fn demo_items(s: &mut Session) -> Vec<ItemId> {
    DemoScene::ALL
        .iter()
        .map(|scene| {
            let src = GeneratorSource::new(Generator::Demo(*scene), 192, 108, FrameRate::FPS_30, Tick::from_seconds_f64(12.0));
            let pool = s.media.clone();
            crate::demo::add_generator(std::sync::Arc::make_mut(&mut s.project), &pool, src, scene.file_name(), Label::Iris, None)
        })
        .collect()
}

/// A 192×108 30 fps sequence of `shots` (item index, seconds), each from 1 s into its clip, with
/// its sound when `sound`; its sequence id and the cut times (ticks).
fn edit(s: &mut Session, items: &[ItemId], shots: &[(usize, f64)], sound: bool) -> (ItemId, Vec<i64>) {
    s.execute("file.newSequence", json!({"width": 192, "height": 108, "fps": 30, "video": 1, "audio": 1})).unwrap();
    let rate = FrameRate::FPS_30;
    let mut t = 0i64;
    let mut cuts = Vec::new();
    for (k, (i, secs)) in shots.iter().enumerate() {
        let frames = (secs * 30.0).round() as i64;
        let d = rate.tick_of(frames).0;
        if k > 0 {
            cuts.push(t);
        }
        s.execute("timeline.place", json!({"item": items[*i].0, "track": "V1", "time": t, "duration": d, "sourceIn": rate.tick_of(30).0, "audio": sound}))
            .unwrap();
        t += d;
    }
    (s.state.active_sequence.unwrap(), cuts)
}

/// Nine shots: 3 s, a burst of four half-second shots, then 3, 2.5, 2 and 3 s (no two night
/// scenes side by side: aurora and city at night are too alike for the cut measure).
const SHOTS: [(usize, f64); 9] = [(0, 3.0), (1, 0.5), (3, 0.5), (2, 0.5), (4, 0.5), (5, 3.0), (0, 2.5), (1, 2.0), (3, 3.0)];

#[test]
fn measured_cuts_shots_and_bursts_match_what_was_built() {
    let mut s = Session::default();
    let items = demo_items(&mut s);
    let (seq, built) = edit(&mut s, &items, &SHOTS, true);
    let r = s.execute(CMD, json!({"item": seq.0})).unwrap();
    assert_eq!(r["source"], "sequence", "a sequence given as an item is rendered");
    let frame = Tick::from_seconds_f64(0.1).0;
    let cuts: Vec<i64> = r["cuts"]["times"].as_array().unwrap().iter().map(|c| c["time"].as_i64().unwrap()).collect();
    assert_eq!(cuts.len(), built.len(), "measured {cuts:?}, built {built:?}: {:#}", r["cuts"]);
    for (m, b) in cuts.iter().zip(&built) {
        assert!((m - b).abs() <= frame, "a cut measured at {:.2} s, built at {:.2} s", Tick(*m).seconds(), Tick(*b).seconds());
    }
    assert_eq!(r["cuts"]["jump"], 0, "different scenes are hard cuts: {:#}", r["cuts"]);
    // the shot lengths
    let shots = &r["shots"];
    assert_eq!(shots["count"], 9, "{shots:#}");
    let lengths: Vec<f64> = shots["list"].as_array().unwrap().iter().map(|s| s["seconds"].as_f64().unwrap()).collect();
    for (m, (_, b)) in lengths.iter().zip(SHOTS.iter()) {
        assert!((m - b).abs() <= 0.11, "a shot of {b} s measured {m} s: {lengths:?}");
    }
    assert!((shots["median"].as_f64().unwrap() - 2.0).abs() <= 0.11, "{shots:#}");
    assert!(shots["cv"].as_f64().unwrap() > 0.4, "the lengths spread: {shots:#}");
    // the four half-second shots are a burst
    let bursts = r["bursts"]["list"].as_array().unwrap();
    assert_eq!(bursts.len(), 1, "{:#}", r["bursts"]);
    assert_eq!(bursts[0]["shots"], 4);
    assert!((Tick(bursts[0]["start"].as_i64().unwrap()).seconds() - 3.0).abs() <= 0.11);
    assert!((Tick(bursts[0]["end"].as_i64().unwrap()).seconds() - 5.0).abs() <= 0.11);
    // the demo scenes move; the energy map is busiest at the burst
    assert!(r["stillness"]["ratio"].as_f64().unwrap() < 0.2, "{:#}", r["stillness"]);
    let energy: Vec<f64> = r["energy"]["values"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let at = |secs: f64| energy[(secs / r["energy"]["bucketSeconds"].as_f64().unwrap()) as usize];
    assert!(at(4.0) > at(1.0) && at(4.0) > at(13.5), "{energy:?}");
    // the sound, the palette and the look
    assert_eq!(r["loudness"]["measured"], true, "{:#}", r["loudness"]);
    assert!(r["loudness"]["integratedLufs"].as_f64().is_some(), "{:#}", r["loudness"]);
    let colours = r["palette"]["colors"].as_array().unwrap();
    assert!(!colours.is_empty() && colours.iter().all(|c| c["hex"].as_str().unwrap().len() == 7), "{:#}", r["palette"]);
    assert!(r["palette"]["brightness"].as_f64().is_some());
    let renderer = &r["renderer"];
    assert_eq!(renderer["gpuFrames"].as_u64().unwrap() + renderer["cpuFrames"].as_u64().unwrap(), r["analysed"]["frames"].as_u64().unwrap());
}

/// A 192×108 still of a room with a bright square (the subject) at `x` of the width. Room 0 has a
/// dark floor under a lit wall with vertical stripes; room 1 is lit from below, with horizontal
/// bands and a bluer light.
fn still(dir: &std::path::Path, name: &str, x: f64, room: u32) -> String {
    let path = dir.join(name);
    let img = image::RgbImage::from_fn(192, 108, |px, py| {
        let (fx, fy) = (f64::from(px) / 192.0, f64::from(py) / 108.0);
        if (fx - x).abs() < 0.06 && (0.35..0.65).contains(&fy) {
            return image::Rgb([240, 200, 170]);
        }
        if room == 0 {
            let stripe = if (px / 12) % 2 == 0 { 30 } else { 0 };
            let base = if fy > 0.6 { 40 } else { 150 };
            image::Rgb([base + stripe, base + stripe, base / 2])
        } else {
            let band = if (py / 9) % 2 == 0 { 25 } else { 0 };
            let base = if fy < 0.4 { 30 } else { 120 };
            image::Rgb([base / 2 + band, base / 2 + 20 + band, base + 60 + band])
        }
    });
    img.save(&path).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn a_jump_cut_keeps_its_framing_and_a_hard_cut_does_not() {
    let dir = crate::media_test_util::tmp_dir("style-jump");
    let mut s = Session::default();
    let a = crate::commands::import_streamed(&mut s, &still(&dir, "a.png", 0.3, 0), None).unwrap();
    let a2 = crate::commands::import_streamed(&mut s, &still(&dir, "a2.png", 0.42, 0), None).unwrap();
    let b = crate::commands::import_streamed(&mut s, &still(&dir, "b.png", 0.7, 1), None).unwrap();
    s.execute("file.newSequence", json!({"width": 192, "height": 108, "fps": 30, "video": 1, "audio": 1})).unwrap();
    let rate = FrameRate::FPS_30;
    for (k, item) in [a, a2, b].iter().enumerate() {
        s.execute(
            "timeline.place",
            json!({"item": item.0, "track": "V1", "time": rate.tick_of(60 * k as i64).0, "duration": rate.tick_of(60).0, "audio": false}),
        )
        .unwrap();
    }
    let seq = s.state.active_sequence.unwrap();
    let r = s.execute(CMD, json!({"sequence": seq.0})).unwrap();
    let cuts: Vec<(f64, String)> =
        r["cuts"]["times"].as_array().unwrap().iter().map(|c| (c["seconds"].as_f64().unwrap(), c["kind"].as_str().unwrap().to_string())).collect();
    assert_eq!(cuts.len(), 2, "{:#}", r["cuts"]);
    assert!((cuts[0].0 - 2.0).abs() < 0.11 && cuts[0].1 == "jump", "the subject moved in the same room: {cuts:?}");
    assert!((cuts[1].0 - 4.0).abs() < 0.11 && cuts[1].1 == "cut", "another room: {cuts:?}");
    assert_eq!(r["cuts"]["jump"].as_u64(), Some(1));
    assert!(r["stillness"]["ratio"].as_f64().unwrap() > 0.9, "stills hold still: {:#}", r["stillness"]);
    assert!(r["loudness"]["measured"] == false, "no sound on the timeline: {:#}", r["loudness"]);
}

/// A `w`×`h` animated GIF at 10 fps of `frames` (opaque RGBA8).
fn write_gif(path: &std::path::Path, w: u32, h: u32, frames: &[Vec<u8>]) {
    let mut out = Vec::new();
    {
        let mut e = image::codecs::gif::GifEncoder::new(&mut out);
        let frames = frames.iter().map(|f| {
            let img = image::RgbaImage::from_raw(w, h, f.clone()).unwrap();
            image::Frame::from_parts(img, 0, 0, image::Delay::from_numer_denom_ms(100, 1))
        });
        e.encode_frames(frames).unwrap();
    }
    std::fs::write(path, out).unwrap();
}

/// A textured grey picture moved `shift` pixels, its level and pattern set.
fn texture(w: usize, h: usize, level: f64, seed: f64, shift: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let g = (level + 50.0 * ((x as f64 + shift) * 0.3 + seed).sin() * (y as f64 * 0.4 + seed).cos()).clamp(0.0, 255.0) as u8;
            v.extend([g, (f64::from(g) * 0.8) as u8, g / 2, 255]);
        }
    }
    v
}

#[test]
fn a_file_is_measured_at_its_own_rate_and_its_settles_told() {
    let dir = crate::media_test_util::tmp_dir("style-gif");
    // 6 s at 10 fps: a shot that pans fast then stops (it settles), a cut, and a still shot that
    // starts to pan just before its end (it cuts on the move)
    let mut shift = 0.0;
    let frames: Vec<Vec<u8>> = (0..60)
        .map(|i| {
            shift += if (5..12).contains(&i) || i >= 54 { 6.0 } else { 0.0 };
            if i < 30 { texture(96, 54, 80.0, 0.0, shift) } else { texture(96, 54, 190.0, 3.0, shift) }
        })
        .collect();
    let path = dir.join("film.gif");
    write_gif(&path, 96, 54, &frames);
    let mut s = Session::default();
    let r = s.execute(CMD, json!({"path": path})).unwrap();
    assert_eq!((r["source"].as_str(), r["analysed"]["fps"].as_f64(), r["analysed"]["sampled"].as_bool()), (Some("file"), Some(10.0), Some(false)), "{r}");
    assert_eq!(r["cuts"]["count"], 1, "{:#}", r["cuts"]);
    assert!((r["cuts"]["times"][0]["seconds"].as_f64().unwrap() - 3.0).abs() < 0.11);
    let settle = &r["settle"];
    assert!(settle["settled"].as_u64().unwrap() >= 1, "the pan stops: {settle:#}");
    assert!(settle["cutOnMove"].as_f64().unwrap() > 0.0, "the second shot cuts while moving: {settle:#}");
    assert!(r["pops"].as_u64().is_some(), "at its own rate pops are looked for");
    assert_eq!(r["loudness"]["measured"], false);
    // a stretch of it
    let part = s.execute(CMD, json!({"path": path, "from": 3.5, "to": 5.5})).unwrap();
    assert_eq!(part["cuts"]["count"], 0, "{part}");
    assert_eq!(part["analysed"]["frames"], 20);
}

#[test]
fn a_long_video_is_sampled_in_windows_within_the_frame_cap() {
    let rate = FrameRate::new(10, 1);
    let hours = Tick::from_seconds_f64(2.0 * 3600.0);
    let w = plan_windows(Tick::ZERO, hours, rate);
    assert!(w.len() > 1, "{w:?}");
    let frames: i64 = w.iter().map(|(a, b)| rate.frame_at(Tick(b.0 - a.0)) + 1).sum();
    assert!(frames <= MAX_FRAMES as i64, "{frames}");
    assert_eq!(w.first().unwrap().0, Tick::ZERO);
    assert_eq!(w.last().unwrap().1, hours);
    assert!(w.windows(2).all(|p| p[0].1 <= p[1].0), "the windows do not overlap");
    // short ones are measured whole
    assert_eq!(plan_windows(Tick::ZERO, Tick::from_seconds_f64(60.0), rate), vec![(Tick::ZERO, Tick::from_seconds_f64(60.0))]);
    // a damaged rate or an odd span still gives sane windows
    assert!(!plan_windows(Tick::ZERO, Tick::from_seconds_f64(43_200.0), FrameRate { num: 0, den: 0 }).is_empty());
    assert_eq!(plan_windows(Tick(5), Tick(5), rate).len(), 1);
    // a 20-minute demo item: 3600 frames at 10 fps would not fit, so it is sampled
    let mut s = Session::default();
    let src = GeneratorSource::new(Generator::Demo(DemoScene::Plasma), 64, 36, FrameRate::FPS_30, Tick::from_seconds_f64(1200.0));
    let pool = s.media.clone();
    let item = crate::demo::add_generator(std::sync::Arc::make_mut(&mut s.project), &pool, src, "long.mov", Label::Iris, None);
    let r = s.execute(CMD, json!({"item": item.0, "fps": 2})).unwrap();
    assert_eq!(r["analysed"]["sampled"], false, "at 2 fps twenty minutes fit: {:#}", r["analysed"]);
    let r = s.execute(CMD, json!({"item": item.0, "from": 0, "to": 900})).unwrap();
    assert_eq!(r["analysed"]["sampled"], true, "{:#}", r["analysed"]);
    assert!(r["analysed"]["frames"].as_u64().unwrap() <= MAX_FRAMES as u64);
    assert!(r["analysed"]["seconds"].as_f64().unwrap() < 900.0);
}

#[test]
fn hostile_parameters_and_files_are_refused_without_a_panic() {
    let dir = crate::media_test_util::tmp_dir("style-hostile");
    let mut s = Session::default();
    let items = demo_items(&mut s);
    let (seq, _) = edit(&mut s, &items, &[(0, 1.0), (1, 1.0)], false);
    // a file that is no video, a truncated GIF and an audio-only item
    let junk = dir.join("junk.mp4");
    std::fs::write(&junk, (0..4096u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect::<Vec<u8>>()).unwrap();
    let gif = dir.join("cut.gif");
    write_gif(&gif, 16, 16, &[texture(16, 16, 90.0, 0.0, 0.0), texture(16, 16, 200.0, 1.0, 0.0)]);
    let bytes = std::fs::read(&gif).unwrap();
    let truncated = dir.join("truncated.gif");
    std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();
    let tone = GeneratorSource::new(Generator::Tone { hz: 440.0, db: -20.0 }, 16, 16, FrameRate::FPS_30, Tick::from_seconds_f64(3.0));
    let pool = s.media.clone();
    let audio_only = crate::demo::add_generator(std::sync::Arc::make_mut(&mut s.project), &pool, tone, "tone.wav", Label::Iris, None);
    s.execute("file.newSequence", json!({"width": 64, "height": 36, "fps": 30})).unwrap();
    let empty = s.state.active_sequence.unwrap();
    for p in [
        json!({}),
        json!(null),
        json!([1, 2]),
        json!("film.mp4"),
        json!({"path": ""}),
        json!({"path": "   "}),
        json!({"path": 5}),
        json!({"path": "x".repeat(5000)}),
        json!({"path": "/no/such/folder/film.mp4"}),
        json!({"path": junk}),
        json!({"path": gif, "item": seq.0}),
        json!({"item": "main"}),
        json!({"item": -1}),
        json!({"item": 1.5}),
        json!({"item": 987_654}),
        json!({"item": audio_only.0}),
        json!({"sequence": 987_654}),
        json!({"sequence": empty.0}),
        json!({"sequence": seq.0, "fps": 0}),
        json!({"sequence": seq.0, "fps": 1}),
        json!({"sequence": seq.0, "fps": 100}),
        json!({"sequence": seq.0, "fps": "fast"}),
        json!({"sequence": seq.0, "from": -1}),
        json!({"sequence": seq.0, "from": 1e300}),
        json!({"sequence": seq.0, "from": "start"}),
        json!({"sequence": seq.0, "from": 1.5, "to": 0.5}),
        json!({"sequence": seq.0, "from": 30}),
        json!({"sequence": seq.0, "colour": true}),
    ] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute(CMD, p.clone())));
        assert!(matches!(r, Ok(Err(_))), "{p}: {r:?}");
    }
    // a truncated file measures what it has or says why; it never panics
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute(CMD, json!({"path": truncated}))));
    assert!(r.is_ok(), "the truncated GIF panicked");
    // the frame helpers refuse what does not fit
    assert_eq!(framing(&[1, 2], &[1]), 0.0);
    assert_eq!(framing(&[], &[]), 0.0);
    assert_eq!(framing(&[9; 144], &[9; 144]), 0.0, "flat pictures have no framing");
    assert_eq!(sharpness(&[0; 4], 1, 1), 0.0);
    assert_eq!(sharpness(&[0; 8], 4, 4), 0.0, "a short buffer");
    assert!(quantile(&[], 0.5).is_none());
}
