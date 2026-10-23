//! The privacy QA of a saved project, without building it again: AutoCut's unblurred-face check
//! (`sequence.checkPrivacy` every half second and at each shot's start, the profile pass and the
//! local vision judge, then [`filmcraft_autocut::qa::face_severity`]) on the project's active
//! sequence, and the frames it looks at as PNGs (for the vision crate's ignored `profile_dump`):
//!
//! ```text
//! FILMCRAFT_DATA_DIR=<dir holding models/privacy-detectors with the profile models> \
//!   cargo run --release -p filmcraft-autocut --features filmcraft-engine/privacy-models \
//!   --example privacy_probe -- qa <project.fcproj> [--no-judge] [--times a,b,…] [--redact]
//! … --example privacy_probe -- frames <project.fcproj> <out dir> <max side> <seconds>… [--plain]
//! … --example privacy_probe -- redact <project.fcproj> [clip id…]
//! … --example privacy_probe -- tracks <video> <from s> <to s>
//! … --example privacy_probe -- crop <out.jpg> <x0,y0,x1,y1> <x,y,w,h | -> <picture>…
//! ```
//!
//! `qa --redact` runs `sequence.autoRedact` (faces, the profile pass, the judge) on the sequence
//! first, as the Quick Short's privacy stage does. `frames --plain` renders without the automatic
//! redaction (the original picture). `redact` prints what the redaction of the project's clips
//! finds and decides (a dry run with the data directory's redaction cache and the judge);
//! `tracks`, the redaction's tracks of a window of a clip alone (a dry run, with their paths).
//! `crop` puts the same region of several pictures side by side (original | before | after), a
//! box outlined in red. Nothing outside `<out dir>` (`<out.jpg>`) is written but what `redact`
//! adds to the data directory's redaction cache; the project and its media are only read.

use std::path::Path;

use filmcraft_engine::Session;
use filmcraft_time::Tick;
use serde_json::{Value, json};

fn fail(m: impl std::fmt::Display) -> ! {
    eprintln!("privacy_probe: {m}");
    std::process::exit(1)
}

fn open(project: &str) -> Session {
    let mut s = Session::default();
    s.execute("file.open", json!({"path": project})).unwrap_or_else(|e| fail(e));
    if s.state.active_sequence.is_none() {
        let first = s.project.sequences().next().map(|i| i.id);
        s.state.active_sequence = first;
    }
    if s.state.active_sequence.is_none() {
        fail("the project has no sequence");
    }
    s
}

/// The times AutoCut's QA looks at: every half second, and each video clip's first frame and a
/// quarter second in.
fn qa_times(s: &Session) -> Vec<f64> {
    let Some(q) = s.active_sequence() else { return Vec::new() };
    let end = q.duration().seconds();
    let fps = q.settings.frame_rate.sane().as_f64();
    let mut t: Vec<f64> = (0..).map(|k| f64::from(k) * 0.5).take_while(|t| *t < end).collect();
    for it in q.video_tracks.iter().filter(|tr| tr.enabled).flat_map(|tr| tr.items.iter().filter(|it| it.enabled && it.end() > it.start)) {
        let first = (it.start.seconds() * fps - 1e-6).ceil() / fps;
        t.push(first);
        if first + 0.25 < it.end().seconds() {
            t.push(first + 0.25);
        }
    }
    let half = 0.5 / fps.max(1.0);
    t.retain(|x| x.is_finite() && *x >= 0.0 && *x < end);
    t.sort_by(f64::total_cmp);
    t.dedup_by(|a, b| (*a - *b).abs() < half);
    t
}

fn qa(s: &mut Session, times: &[f64]) -> Value {
    let mut faces = Vec::new();
    let (mut frames, mut masked) = (0u64, 0u64);
    let mut raw = Vec::new();
    for chunk in times.chunks(30) {
        let v = s.execute("sequence.checkPrivacy", json!({"seconds": chunk, "maxSide": 1280, "minFaceHeight": 0.05})).unwrap_or_else(|e| fail(e));
        frames += v["frames"].as_u64().unwrap_or(0);
        masked += v["masked"].as_u64().unwrap_or(0);
        faces.extend(filmcraft_autocut::qa::faces_of(&v));
        raw.extend(v["unmasked"].as_array().cloned().unwrap_or_default());
    }
    let mut faces = filmcraft_autocut::qa::merge_faces(faces, 1.25);
    let (mut errors, mut warnings) = (0usize, 0usize);
    for f in &mut faces {
        f.severity = filmcraft_autocut::qa::face_severity(f);
        match f.severity {
            Some(filmcraft_autocut::qa::Severity::Error) => errors += 1,
            Some(filmcraft_autocut::qa::Severity::Warning) => warnings += 1,
            _ => {}
        }
    }
    json!({"frames": frames, "masked": masked, "errors": errors, "warnings": warnings, "faces": faces, "unmasked": raw})
}

fn frames(s: &Session, out: &Path, side: f64, times: &[f64]) {
    std::fs::create_dir_all(out).unwrap_or_else(|e| fail(e));
    let Some((w, h)) = s.active_sequence().map(|q| (q.settings.width, q.settings.height)) else { fail("no sequence") };
    let scale = (side / f64::from(w.max(h).max(1))).min(1.0) as f32;
    for t in times {
        let img = s.try_render_program_at(scale, Tick::from_seconds_f64(*t)).unwrap_or_else(|e| fail(e));
        let Some(buf) = image::RgbaImage::from_raw(img.w as u32, img.h as u32, img.over_black_rgba8()) else { fail("a frame of the wrong size") };
        let p = out.join(format!("{t:08.3}.png"));
        buf.save(&p).unwrap_or_else(|e| fail(format!("{}: {e}", p.display())));
        println!("{}", p.display());
    }
}

/// The same region (`x0,y0,x1,y1`, shares of each picture) of several pictures side by side,
/// each scaled to 400 px tall, with box `x,y,w,h` (shares; `-` for none) outlined in red.
fn crop(out: &Path, region: &str, outline: &str, pictures: &[String]) {
    let nums = |v: &str| -> Vec<f64> { v.split(',').filter_map(|x| x.parse().ok()).collect() };
    let r = nums(region);
    let [x0, y0, x1, y1] = r.as_slice() else { fail("region x0,y0,x1,y1") };
    let b = nums(outline);
    let side = 400u32;
    let mut tiles: Vec<image::RgbImage> = Vec::new();
    for p in pictures {
        let img = image::open(p).unwrap_or_else(|e| fail(format!("{p}: {e}"))).to_rgb8();
        let (w, h) = (f64::from(img.width()), f64::from(img.height()));
        let (cx, cy) = ((x0 * w).max(0.0) as u32, (y0 * h).max(0.0) as u32);
        let (cw, ch) = ((((x1 - x0) * w) as u32).max(1), (((y1 - y0) * h) as u32).max(1));
        let mut c = image::imageops::crop_imm(&img, cx, cy, cw, ch).to_image();
        if let [bx, by, bw, bh] = b.as_slice() {
            let (l, t) = ((bx * w) as i64 - i64::from(cx), (by * h) as i64 - i64::from(cy));
            let (rr, bb) = (l + (bw * w) as i64, t + (bh * h) as i64);
            for (x, y, px) in c.enumerate_pixels_mut() {
                let (x, y) = (i64::from(x), i64::from(y));
                let on_v = (x - l).abs() <= 1 || (x - rr).abs() <= 1;
                let on_h = (y - t).abs() <= 1 || (y - bb).abs() <= 1;
                if (on_v && y >= t && y <= bb) || (on_h && x >= l && x <= rr) {
                    *px = image::Rgb([255, 0, 0]);
                }
            }
        }
        let tw = (f64::from(side) * f64::from(c.width()) / f64::from(c.height().max(1))).round().max(1.0) as u32;
        tiles.push(image::imageops::resize(&c, tw, side, image::imageops::FilterType::Triangle));
    }
    let gap = 8u32;
    let total = tiles.iter().map(|t| t.width()).sum::<u32>() + gap * (tiles.len().max(1) as u32 - 1);
    let mut sheet = image::RgbImage::from_pixel(total.max(1), side, image::Rgb([255, 255, 255]));
    let mut x = 0i64;
    for t in &tiles {
        image::imageops::replace(&mut sheet, t, x, 0);
        x += i64::from(t.width() + gap);
    }
    sheet.save(out).unwrap_or_else(|e| fail(format!("{}: {e}", out.display())));
}

/// The redaction's tracks (a dry run, with their paths) of `video` from `from` to `to` (media
/// seconds) placed alone in a sequence of its size.
fn tracks(video: &str, from: f64, to: f64) -> Value {
    let mut s = Session::default();
    let r = s.execute("file.import", json!({"paths": [video]})).unwrap_or_else(|e| fail(e));
    let item = r["items"][0].as_u64().map(filmcraft_project::ItemId).unwrap_or_else(|| fail(format!("import: {r}")));
    let info = s.project.item(item).and_then(|i| i.as_media()).and_then(|m| m.info.video.clone()).unwrap_or_else(|| fail("not a video"));
    let mut p = (*s.project).clone();
    let settings = filmcraft_project::SequenceSettings { width: info.width, height: info.height, frame_rate: info.frame_rate, ..Default::default() };
    let seq = p.new_sequence("Tracks", settings, 1, 1, None);
    let src = filmcraft_time::TimeRange::new(Tick::from_seconds_f64(from), Tick::from_seconds_f64(to - from));
    let Some(mut v) = p.make_track_item(item, filmcraft_project::TrackKind::Video, Tick::ZERO, src, info.frame_rate) else { fail("cannot place the clip") };
    for e in &mut v.effects {
        filmcraft_project::resolve_auto_points(e, (info.width, info.height), (info.width, info.height));
    }
    let clip = v.id;
    match p.sequence_mut(seq).and_then(|q| q.video_tracks.first_mut()) {
        Some(t) => t.items.push(v),
        None => fail("no video track"),
    }
    s.project = std::sync::Arc::new(p);
    s.state.active_sequence = Some(seq);
    s.state.selection = vec![clip];
    let r =
        s.execute("clip.autoRedact", json!({"wait": true, "profiles": true, "dryRun": true, "verify": false, "kinds": "faces"})).unwrap_or_else(|e| fail(e));
    r["redact"].clone()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let opt = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1));
    match args.get(1).map(String::as_str) {
        Some("qa") => {
            let Some(project) = args.get(2) else { fail("qa <project.fcproj> [--no-judge] [--times a,b,…] [--redact]") };
            let mut s = open(project);
            if !args.iter().any(|a| a == "--no-judge") {
                s.redacts.judge = Some(filmcraft_autocut::privacy::engine_judge());
            }
            if args.iter().any(|a| a == "--redact") {
                let r = s.execute("sequence.autoRedact", json!({"wait": true, "profiles": true, "kinds": "faces"})).unwrap_or_else(|e| fail(e));
                eprintln!("redacted: {}", r["redact"]["clips"].as_array().map_or(0, Vec::len));
            }
            let times: Vec<f64> = match opt("--times") {
                Some(v) => v.split(',').filter_map(|x| x.parse().ok()).collect(),
                None => qa_times(&s),
            };
            let v = qa(&mut s, &times);
            println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
        }
        Some("frames") => {
            let (Some(project), Some(out), Some(side)) = (args.get(2), args.get(3), args.get(4).and_then(|v| v.parse::<f64>().ok())) else {
                fail("frames <project.fcproj> <out dir> <max side> <seconds>… [--plain]")
            };
            let times: Vec<f64> = args.get(5..).unwrap_or_default().iter().filter_map(|x| x.parse().ok()).collect();
            let mut s = open(project);
            if args.iter().any(|a| a == "--plain") {
                // the original picture: the automatic redaction's effects off
                let mut p = (*s.project).clone();
                if let Some(q) = s.state.active_sequence.and_then(|id| p.sequence_mut(id)) {
                    for e in q.video_tracks.iter_mut().flat_map(|t| t.items.iter_mut()).flat_map(|it| it.effects.iter_mut()) {
                        if filmcraft_engine::redact::is_auto_redaction(e) {
                            e.enabled = false;
                        }
                    }
                }
                s.project = std::sync::Arc::new(p);
            }
            frames(&s, Path::new(out), side, &times);
        }
        Some("redact") => {
            // the redaction of the project's clips as a dry run (`clips`: ids; default all), with
            // the data directory's redaction cache and the judge: what the build hid, and why not
            let Some(project) = args.get(2) else { fail("redact <project.fcproj> [clip id…]") };
            let mut s = open(project);
            if let Some(d) = filmcraft_engine::autosave::default_data_dir() {
                s.use_redaction_cache(&d);
            }
            s.redacts.judge = Some(filmcraft_autocut::privacy::engine_judge());
            let ids: Vec<u64> = args.get(3..).unwrap_or_default().iter().filter_map(|x| x.parse().ok()).collect();
            let mut p = json!({"wait": true, "dryRun": true, "kinds": "faces"});
            if !ids.is_empty() {
                p["clips"] = json!(ids);
            }
            let cmd = if ids.is_empty() { "sequence.autoRedact" } else { "clip.autoRedact" };
            let r = s.execute(cmd, p).unwrap_or_else(|e| fail(e));
            println!("{}", serde_json::to_string_pretty(&r["redact"]).unwrap_or_default());
        }
        Some("tracks") => {
            let (Some(video), Some(from), Some(to)) = (args.get(2), args.get(3).and_then(|v| v.parse().ok()), args.get(4).and_then(|v| v.parse().ok())) else {
                fail("tracks <video> <from s> <to s>")
            };
            println!("{}", serde_json::to_string_pretty(&tracks(video, from, to)).unwrap_or_default());
        }
        Some("crop") => {
            let (Some(out), Some(region), Some(outline), Some(pictures)) = (args.get(2), args.get(3), args.get(4), args.get(5..)) else {
                fail("crop <out.jpg> <x0,y0,x1,y1> <x,y,w,h | -> <picture>…")
            };
            crop(Path::new(out), region, outline, pictures);
        }
        _ => fail("qa <project.fcproj> … | frames <project.fcproj> <out dir> <max side> <seconds>… | crop <out.jpg> <region> <box> <picture>…"),
    }
}
