//! `qa.motionEnergy` on synthetic frames, an animated GIF written here and sequences of generated
//! stills (no media is committed and no ffmpeg is needed).
use filmcraft_time::{FrameRate, Tick};
use serde_json::{Value, json};

use super::*;
use crate::Session;

/// A grey texture on a `w`×`h` grid (opaque RGBA8): `level` ± `amp`, its pattern chosen by
/// `seed`, moved `shift` pixels to the left.
fn texture(w: usize, h: usize, level: f64, amp: f64, seed: f64, shift: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let g = (level + amp * ((x as f64 + shift) * 0.3 + seed).sin() * (y as f64 * 0.4 + seed).cos()).clamp(0.0, 255.0) as u8;
            v.extend([g, g, g, 255]);
        }
    }
    v
}

fn flat(w: usize, h: usize, g: u8) -> Vec<u8> {
    [g, g, g, 255].repeat(w * h)
}

fn options() -> Options {
    Options { expect_motion: true, ..Default::default() }
}

/// The meter over `frames` at `rate` (one analysed frame per frame of the rate when `native`).
fn meter(frames: &[Vec<u8>], rate: FrameRate, native: bool, o: &Options) -> (Vec<Value>, Value, Vec<Value>) {
    let mut m = Meter::new(rate.as_f64(), native);
    for (i, f) in frames.iter().enumerate() {
        m.push(rate.tick_of(i as i64), f, GRID);
    }
    m.finish(rate.tick_of(frames.len() as i64), o)
}

fn codes(findings: &[Value]) -> Vec<&str> {
    findings.iter().filter_map(|f| f["code"].as_str()).collect()
}

fn with_code<'a>(findings: &'a [Value], code: &str) -> Vec<&'a Value> {
    findings.iter().filter(|f| f["code"] == code).collect()
}

/// Ten seconds at 30 fps: a dark moving shot with one white frame at 50, a cut at 100 to a bright
/// moving shot, which freezes on its frame 199 until 260, then a cut to a third moving shot.
fn edit() -> Vec<Vec<u8>> {
    let (w, h) = GRID;
    (0..300)
        .map(|i| match i {
            50 => flat(w, h, 255),
            0..100 => texture(w, h, 70.0, 50.0, 0.0, i as f64),
            100..200 => texture(w, h, 180.0, 50.0, 2.0, i as f64),
            200..260 => texture(w, h, 180.0, 50.0, 2.0, 199.0),
            _ => texture(w, h, 120.0, 60.0, 4.0, 2.0 * i as f64),
        })
        .collect()
}

#[test]
fn pops_cuts_and_stills_are_found_and_timed() {
    let rate = FrameRate::FPS_30;
    let (findings, measures, skipped) = meter(&edit(), rate, true, &options());
    assert!(skipped.is_empty(), "{skipped:?}");
    let pops = with_code(&findings, "pop");
    assert_eq!(pops.len(), 1, "{findings:?}");
    assert_eq!(pops[0]["time"], rate.tick_of(50).0);
    assert_eq!(pops[0]["severity"], "warn");
    // the pop is not a cut, and the freeze is not one either
    assert_eq!(measures["cuts"], json!([rate.tick_of(100).0, rate.tick_of(260).0]), "{measures}");
    assert!((measures["cutsPerMinute"].as_f64().unwrap() - 12.0).abs() < 1e-6);
    let stills = with_code(&findings, "still");
    assert_eq!(stills.len(), 1, "{findings:?}");
    assert_eq!((stills[0]["time"].as_i64(), stills[0]["end"].as_i64()), (Some(rate.tick_of(199).0), Some(rate.tick_of(260).0)));
    assert!(stills[0]["message"].as_str().unwrap().contains("2.0 s"));
    // a moving film: no flat opening, not barely moving, a normal first frame
    for code in ["flatOpening", "barelyMoves", "blackFirstFrame", "emptyFirstFrame", "loopSeam"] {
        assert!(!codes(&findings).contains(&code), "{code}: {findings:?}");
    }
    let motion = &measures["motion"];
    assert!(motion["mean"].as_f64().unwrap() > FLAT_MOTION * 5.0, "{motion}");
    assert!(motion["opening"].as_f64().unwrap() > FLAT_MOTION, "{motion}");
    assert_eq!(motion["perSecond"].as_array().unwrap().len(), 10);
    // seconds 7 and 8 are the freeze
    assert!(motion["perSecond"][7].as_f64().unwrap() < 0.001, "{motion}");
}

#[test]
fn declared_holds_exempt_a_still_and_only_the_part_they_cover() {
    let rate = FrameRate::FPS_30;
    let held = Options { holds: vec![(rate.tick_of(199).0, rate.tick_of(260).0)], ..options() };
    let (findings, _, _) = meter(&edit(), rate, true, &held);
    assert!(with_code(&findings, "still").is_empty(), "{findings:?}");
    // a hold to frame 235 leaves 25 frames (0.83 s): under a second
    let part = Options { holds: vec![(rate.tick_of(199).0, rate.tick_of(235).0)], ..options() };
    assert!(with_code(&meter(&edit(), rate, true, &part).0, "still").is_empty());
    // a hold to frame 230 leaves exactly a second
    let part = Options { holds: vec![(rate.tick_of(199).0, rate.tick_of(230).0)], ..options() };
    let (findings, _, _) = meter(&edit(), rate, true, &part);
    let stills = with_code(&findings, "still");
    assert_eq!(stills.len(), 1, "{findings:?}");
    assert_eq!(stills[0]["time"], rate.tick_of(230).0);
}

#[test]
fn a_film_that_does_not_move_is_flat_still_and_empty() {
    let rate = FrameRate::FPS_30;
    let frames = vec![flat(GRID.0, GRID.1, 128); 300];
    let (findings, measures, _) = meter(&frames, rate, true, &options());
    let found = codes(&findings);
    for code in ["still", "flatOpening", "barelyMoves", "emptyFirstFrame"] {
        assert!(found.contains(&code), "{code}: {findings:?}");
    }
    assert!(!found.contains(&"evenMotion"), "a film that does not move is not judged for even motion");
    assert!(!found.contains(&"blackFirstFrame"));
    let still = with_code(&findings, "still")[0];
    assert_eq!((still["time"].as_i64(), still["end"].as_i64()), (Some(0), Some(rate.tick_of(300).0)));
    assert_eq!(measures["cutCount"], 0);
    // held all through: the still is meant; the flat opening is still reported
    let held = Options { holds: vec![(0, rate.tick_of(300).0)], ..options() };
    let found_held = meter(&frames, rate, true, &held).0;
    assert!(with_code(&found_held, "still").is_empty());
    assert!(codes(&found_held).contains(&"flatOpening"));
    // declared not to move: only the first frame is judged
    let still_ok = Options { expect_motion: false, ..options() };
    let (findings, _, skipped) = meter(&frames, rate, true, &still_ok);
    assert_eq!(codes(&findings), ["emptyFirstFrame"]);
    assert_eq!(skipped.len(), 1);
}

#[test]
fn a_black_first_frame_and_a_loop_seam_are_reported() {
    let rate = FrameRate::new(10, 1);
    let (w, h) = GRID;
    let mut frames: Vec<Vec<u8>> = (0..30).map(|i| texture(w, h, 120.0, 60.0, 1.0, i as f64)).collect();
    frames[0] = flat(w, h, 0);
    let looped = Options { looped: true, ..options() };
    let (findings, measures, _) = meter(&frames, rate, true, &looped);
    assert!(codes(&findings).contains(&"blackFirstFrame"), "{findings:?}");
    let seam = with_code(&findings, "loopSeam");
    assert_eq!(seam.len(), 1, "{findings:?}");
    assert_eq!(seam[0]["time"], rate.tick_of(29).0);
    // the seam is measured either way, reported only for a loop
    let (findings, unlooped, _) = meter(&frames, rate, true, &options());
    assert!(!codes(&findings).contains(&"loopSeam"));
    assert_eq!(unlooped["loopSeam"], measures["loopSeam"]);
    // a pattern of period 60 px moving 1 px a frame for 60 frames comes back to its start
    let periodic = |shift: f64| {
        let mut v = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let a = std::f64::consts::TAU * (x as f64 + shift) / 60.0;
                let g = (120.0 + 60.0 * a.sin() * (y as f64 * 0.3).cos()) as u8;
                v.extend([g, g, g, 255]);
            }
        }
        v
    };
    let frames: Vec<Vec<u8>> = (0..60).map(|i| periodic(i as f64)).collect();
    let (findings, _, _) = meter(&frames, rate, true, &looped);
    assert!(!codes(&findings).contains(&"loopSeam"), "{findings:?}");
}

#[test]
fn motion_that_never_changes_pace_is_even() {
    let rate = FrameRate::new(10, 1);
    let (w, h) = GRID;
    let even: Vec<Vec<u8>> = (0..200).map(|i| texture(w, h, 120.0, 60.0, 1.0, i as f64)).collect();
    let (findings, measures, _) = meter(&even, rate, true, &options());
    assert!(codes(&findings).contains(&"evenMotion"), "{findings:?} {measures}");
    // two seconds fast, two seconds slow
    let mut shift = 0.0;
    let varied: Vec<Vec<u8>> = (0..200)
        .map(|i| {
            shift += if (i / 20) % 2 == 0 { 3.0 } else { 0.2 };
            texture(w, h, 120.0, 60.0, 1.0, shift)
        })
        .collect();
    let (findings, measures, _) = meter(&varied, rate, true, &options());
    assert!(!codes(&findings).contains(&"evenMotion"), "{findings:?}");
    assert!(measures["motion"]["contrast"].as_f64().unwrap() > EVEN_CONTRAST, "{measures}");
}

#[test]
fn below_the_source_rate_pops_are_not_looked_for() {
    let (findings, _, skipped) = meter(&edit(), FrameRate::FPS_30, false, &options());
    assert!(with_code(&findings, "pop").is_empty());
    assert!(skipped.iter().any(|s| s["check"] == "pop"), "{skipped:?}");
}

#[test]
fn reductions_are_bounded_and_refuse_hostile_pictures() {
    // a decoder that ignored the scale: a 4K frame still reduces to the grid
    let big = vec![200u8; 3840 * 2160 * 4];
    let small = box_reduce(3840, 2160, &big, 96, 54).unwrap();
    assert_eq!(small.len(), 96 * 54 * 4);
    assert!(small.as_chunks::<4>().0.iter().all(|p| p[0] == 156 && p[3] == 255), "over black: 200 at alpha 200");
    // smaller than the grid: every output pixel is filled
    let tiny = box_reduce(2, 2, &[255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 0, 255, 255, 255, 255], 96, 54).unwrap();
    assert_eq!(tiny.len(), 96 * 54 * 4);
    for (w, h, len) in [(0, 0, 0), (10, 10, 3), (u32::MAX, u32::MAX, 0), (5000, 5000, 16)] {
        assert!(box_reduce(w, h, &vec![0; len], 96, 54).is_err(), "{w}x{h}");
    }
    assert!(box_reduce(4, 4, &[0; 64], 0, 54).is_err());
    assert_eq!(mean_diff(&[1, 2], &[1]), 1.0);
    assert_eq!(mean_diff(&[], &[]), 1.0);
    assert_eq!(grid_for(1080, 1920), (54, 96));
}

#[test]
fn long_inputs_are_sampled_down_to_the_frame_cap() {
    let hour = Tick::from_seconds_f64(3600.0);
    let (k, count) = plan(hour, FrameRate::FPS_30, None);
    assert_eq!(k, 15);
    assert!(count <= MAX_FRAMES, "{count}");
    assert_eq!(plan(Tick::from_seconds_f64(10.0), FrameRate::FPS_30, None), (1, 300));
    assert_eq!(plan(Tick::from_seconds_f64(10.0), FrameRate::FPS_30, Some(10.0)), (3, 100));
    assert_eq!(plan(Tick::from_seconds_f64(10.0), FrameRate::FPS_30, Some(240.0)), (1, 300));
    // a damaged rate falls back to a sane one; twelve hours at 0.1 fps stays small
    assert!(plan(Tick::from_seconds_f64(43_200.0), FrameRate { num: 0, den: 0 }, Some(0.1)).1 <= MAX_FRAMES);
    assert_eq!(uncovered(0, 10, &[(2, 4), (3, 6), (9, 20)]), vec![(0, 2), (6, 9)]);
    assert_eq!(uncovered(0, 10, &[]), vec![(0, 10)]);
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

/// Four seconds at 10 fps: a moving shot with a white frame at 8, frozen on frame 14 until 30,
/// then a cut to a brighter moving shot.
fn gif_film() -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = crate::media_test_util::tmp_dir("motion-energy-gif");
    let (w, h) = GRID;
    let frames: Vec<Vec<u8>> = (0..40)
        .map(|i| match i {
            8 => flat(w, h, 255),
            0..15 => texture(w, h, 70.0, 50.0, 0.0, 2.0 * i as f64),
            15..30 => texture(w, h, 70.0, 50.0, 0.0, 28.0),
            _ => texture(w, h, 190.0, 50.0, 3.0, 2.0 * i as f64),
        })
        .collect();
    let path = dir.join("film.gif");
    write_gif(&path, w as u32, h as u32, &frames);
    (dir, path)
}

#[test]
fn a_rendered_file_is_measured_and_its_contact_sheet_written() {
    let (dir, path) = gif_film();
    let mut s = Session::default();
    let sheet = dir.join("sheet.png");
    let r = s.execute(CMD, json!({"path": path, "sheet": sheet})).unwrap();
    assert_eq!((r["source"].as_str(), r["fps"].as_f64(), r["sampledDown"].as_bool()), (Some("file"), Some(10.0), Some(false)), "{r}");
    assert_eq!(r["frames"], 40);
    let findings = r["findings"].as_array().unwrap();
    let rate = FrameRate::new(10, 1);
    let pop = with_code(findings, "pop");
    assert_eq!(pop.len(), 1, "{r}");
    assert_eq!(pop[0]["time"], rate.tick_of(8).0);
    let still = with_code(findings, "still");
    assert_eq!(still.len(), 1, "{r}");
    assert_eq!((still[0]["time"].as_i64(), still[0]["end"].as_i64()), (Some(rate.tick_of(14).0), Some(rate.tick_of(30).0)));
    assert_eq!(r["measures"]["cuts"], json!([rate.tick_of(30).0]), "{r}");
    assert_eq!(r["ok"], false);
    assert_eq!(r["blocking"], false);
    // the sheet: four tiles a second, sixteen to a row
    let written = &r["sheet"];
    let tiles = written["tiles"].as_u64().unwrap();
    assert!((15..=17).contains(&tiles), "{written}");
    let img = image::open(&sheet).unwrap();
    assert_eq!((u64::from(img.width()), u64::from(img.height())), (written["width"].as_u64().unwrap(), written["height"].as_u64().unwrap()));
    // the freeze declared as a hold
    let held = s.execute(CMD, json!({"path": path, "holds": [{"start": rate.tick_of(14).0, "end": rate.tick_of(30).0, "reason": "the title holds"}]})).unwrap();
    assert!(with_code(held["findings"].as_array().unwrap(), "still").is_empty(), "{held}");
    // analysed at 5 fps: sampled down, no pop search
    let slow = s.execute(CMD, json!({"path": path, "fps": 5})).unwrap();
    assert_eq!((slow["frames"].as_u64(), slow["sampledDown"].as_bool()), (Some(20), Some(true)));
    assert!(slow["skipped"].as_array().unwrap().iter().any(|v| v["check"] == "pop"));
}

#[test]
fn the_animation_preflight_reports_motion_as_warnings_only() {
    let (_, path) = gif_film();
    let rate = FrameRate::new(10, 1);
    let mut s = Session::default();
    let hold = json!([{"start": rate.tick_of(14).0, "end": rate.tick_of(30).0, "reason": "the title holds"}]);
    let r = s.execute("media.preflightAnimation", json!({"path": path, "holds": hold})).unwrap();
    assert_eq!(r["passed"], true, "a motion warning never fails the preflight: {r}");
    assert!(with_code(r["warnings"].as_array().unwrap(), "pop").len() == 1, "{r}");
    assert_eq!(r["motionEnergy"]["source"], "file");
    let off = s.execute("media.preflightAnimation", json!({"path": path, "holds": hold, "motionEnergy": false})).unwrap();
    assert!(off["warnings"].as_array().unwrap().is_empty());
    assert!(s.execute("media.preflightAnimation", json!({"path": path, "motionEnergy": "yes"})).is_err());
}

/// A 160×90 30 fps sequence: photo A for a second, photo B for one frame, photo A to 3 s.
fn photo_sequence() -> (Session, filmcraft_project::ItemId) {
    let dir = crate::media_test_util::tmp_dir("motion-energy-seq");
    let (a, b) = (dir.join("a.png"), dir.join("b.png"));
    let rgb =
        |level: f64, seed: f64| -> Vec<u8> { texture(160, 90, level, 50.0, seed, 0.0).as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect() };
    image::save_buffer(&a, &rgb(70.0, 0.0), 160, 90, image::ColorType::Rgb8).unwrap();
    image::save_buffer(&b, &rgb(200.0, 2.0), 160, 90, image::ColorType::Rgb8).unwrap();
    let mut s = Session::default();
    let ia = crate::commands::import_streamed(&mut s, a.to_str().unwrap(), None).unwrap();
    let ib = crate::commands::import_streamed(&mut s, b.to_str().unwrap(), None).unwrap();
    s.execute("file.newSequence", json!({"width": 160, "height": 90, "fps": 30, "video": 1, "audio": 1})).unwrap();
    let rate = FrameRate::FPS_30;
    for (item, from, to) in [(ia, 0, 30), (ib, 30, 31), (ia, 31, 90)] {
        let (t, d) = (rate.tick_of(from), rate.tick_of(to).0 - rate.tick_of(from).0);
        s.execute("timeline.place", json!({"item": item.0, "track": "V1", "time": t.0, "duration": d, "audio": false})).unwrap();
    }
    let seq = s.state.active_sequence.unwrap();
    (s, seq)
}

#[test]
fn a_sequence_is_rendered_and_measured() {
    let (mut s, seq) = photo_sequence();
    let rate = FrameRate::FPS_30;
    let r = s.execute(CMD, json!({"sequence": seq.0})).unwrap();
    assert_eq!(r["source"], "sequence");
    assert_eq!(r["frames"], 90, "{r}");
    let renderer = &r["renderer"];
    assert_eq!(renderer["gpuFrames"].as_u64().unwrap() + renderer["cpuFrames"].as_u64().unwrap(), 90);
    let findings = r["findings"].as_array().unwrap();
    let pop = with_code(findings, "pop");
    assert_eq!(pop.len(), 1, "{r}");
    assert_eq!(pop[0]["time"], rate.tick_of(30).0);
    assert_eq!(r["measures"]["cutCount"], 0, "the one-frame photo is a pop, not two cuts: {r}");
    let stills: Vec<(i64, i64)> = with_code(findings, "still").iter().map(|f| (f["time"].as_i64().unwrap(), f["end"].as_i64().unwrap())).collect();
    assert_eq!(stills, vec![(0, rate.tick_of(30).0), (rate.tick_of(31).0, rate.tick_of(90).0)], "{r}");
    for code in ["flatOpening", "barelyMoves"] {
        assert!(codes(findings).contains(&code), "{code}: {r}");
    }
    let held = s.execute(CMD, json!({"sequence": seq.0, "holds": [{"start": 0, "end": rate.tick_of(90).0, "reason": "photo story"}]})).unwrap();
    let held = held["findings"].as_array().unwrap();
    assert!(with_code(held, "still").is_empty());
    assert_eq!(with_code(held, "pop").len(), 1, "a hold never hides a pop");
}

#[test]
fn the_delivery_gate_carries_motion_energy_as_a_warning() {
    let (mut s, _) = photo_sequence();
    let r = s.execute("delivery.preflight", json!({"privacy": false, "narration": false, "mix": false})).unwrap();
    let leg = r["checks"].as_array().unwrap().iter().find(|c| c["check"] == "motionEnergy").cloned().unwrap_or_else(|| panic!("{r}"));
    assert_eq!(leg["state"], "warning", "{leg}");
    assert_eq!(with_code(leg["detail"]["findings"].as_array().unwrap(), "pop").len(), 1, "{leg}");
    // still photos on the timeline count as holds in the gate, as in its motion check
    assert!(with_code(leg["detail"]["findings"].as_array().unwrap(), "still").len() < 2, "{leg}");
    assert_eq!(r["passed"], true, "a motion warning never blocks delivery: {r}");
    let off = s.execute("delivery.preflight", json!({"privacy": false, "narration": false, "mix": false, "motionEnergy": false})).unwrap();
    let leg = off["checks"].as_array().unwrap().iter().find(|c| c["check"] == "motionEnergy").cloned().unwrap();
    assert_eq!(leg["state"], "skipped");
    assert!(s.execute("delivery.preflight", json!({"privacy": false, "narration": false, "mix": false, "motionEnergy": 1})).is_err());
}

#[test]
fn hostile_parameters_are_refused() {
    let (mut s, seq) = photo_sequence();
    let (_, gif) = gif_film();
    let end = s.project.sequence(seq).unwrap().duration().0;
    let hold = |start: Value, end: Value, reason: Value| json!([{"start": start, "end": end, "reason": reason}]);
    let mut many = Vec::new();
    for i in 0..2001i64 {
        many.push(json!({"start": i, "end": i + 1, "reason": "x"}));
    }
    for p in [
        json!({}),
        json!(null),
        json!([1, 2]),
        json!("film.mp4"),
        json!({"path": gif, "sequence": seq.0}),
        json!({"sequence": 987_654}),
        json!({"sequence": -1}),
        json!({"sequence": "main"}),
        json!({"sequence": 1.5}),
        json!({"path": ""}),
        json!({"path": "  "}),
        json!({"path": 5}),
        json!({"path": "x".repeat(5000)}),
        json!({"path": "/no/such/folder/film.mp4"}),
        json!({"sequence": seq.0, "fps": -1}),
        json!({"sequence": seq.0, "fps": 0}),
        json!({"sequence": seq.0, "fps": 0.05}),
        json!({"sequence": seq.0, "fps": 1e9}),
        json!({"sequence": seq.0, "fps": "fast"}),
        json!({"sequence": seq.0, "loop": "yes"}),
        json!({"sequence": seq.0, "holds": "all"}),
        json!({"sequence": seq.0, "holds": Value::Array(many)}),
        json!({"sequence": seq.0, "holds": hold(json!(0), json!(end), json!(""))}),
        json!({"sequence": seq.0, "holds": hold(json!(0), json!(end), Value::Null)}),
        json!({"sequence": seq.0, "holds": hold(json!(-5), json!(end), json!("x"))}),
        json!({"sequence": seq.0, "holds": hold(json!(0), json!(end + 1), json!("x"))}),
        json!({"sequence": seq.0, "holds": hold(json!(10), json!(5), json!("x"))}),
        json!({"sequence": seq.0, "holds": hold(json!(0), json!(end), json!("x".repeat(600)))}),
        json!({"sequence": seq.0, "sheet": ""}),
        json!({"sequence": seq.0, "sheet": 3}),
        json!({"path": gif, "sheet": gif}),
    ] {
        assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
    }
    // an empty sequence has nothing to measure
    s.execute("file.newSequence", json!({"width": 160, "height": 90, "fps": 30})).unwrap();
    let empty = s.state.active_sequence.unwrap();
    assert!(s.execute(CMD, json!({"sequence": empty.0})).is_err());
    // the highest rate is the sequence's own
    let r = s.execute(CMD, json!({"sequence": seq.0, "fps": 240})).unwrap();
    assert_eq!((r["frames"].as_u64(), r["sampledDown"].as_bool()), (Some(90), Some(false)));
}
