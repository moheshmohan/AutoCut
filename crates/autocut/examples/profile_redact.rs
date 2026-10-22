//! A live check of the redaction's profile pass (faces in profile and half hidden, found from
//! people's poses): a window of a clip redacted without it and with it, the privacy QA
//! (`sequence.checkPrivacy`, with the profile pass) run on both, and frames and crops written:
//!
//! - `frames/<t>_{original,before,after}.png`: the moments of the faces the profile pass found,
//!   of the faces the vision check left sharp, and of the people the QA still saw with the face
//!   detector alone, as the sequence exports them;
//! - `crops/<n>_<what>.png`: each such face cropped from those three frames, side by side
//!   (`profile`: found only by the profile pass; `faceAndProfile`: by both; `leftSharp`: not a
//!   person; `qaBefore`: seen by the QA without the profile pass);
//! - `summary.json`: both runs' outcomes and QA answers.
//!
//! ```text
//! FILMCRAFT_DATA_DIR=<dir holding models/privacy-detectors with the profile models> \
//!   cargo run --release -p filmcraft-autocut --features filmcraft-engine/privacy-models \
//!   --example profile_redact -- <video> <from s> <to s> <out dir> [--no-judge] [--exclude x,y,w,h]
//! ```
//!
//! The vision check is AutoCut's (Ollama, `AUTOCUT_*` environment). Nothing outside `<out dir>` is
//! written; the video is only read.

use std::path::{Path, PathBuf};

use filmcraft_engine::Session;
use filmcraft_project::{ItemId, SequenceSettings, TrackKind};
use filmcraft_time::{Tick, TimeRange};
use serde_json::{Value, json};

fn fail(m: impl std::fmt::Display) -> ! {
    eprintln!("profile_redact: {m}");
    std::process::exit(1)
}

/// A session with `video` from `from` to `to` (seconds) on V1 of a sequence of its size.
fn session(video: &Path, from: f64, to: f64) -> (Session, ItemId) {
    let mut s = Session::default();
    let r = s.execute("file.import", json!({"paths": [video.to_string_lossy()]})).unwrap_or_else(|e| fail(e));
    let item = r["items"][0].as_u64().map(ItemId).unwrap_or_else(|| fail(format!("import: {r}")));
    let info = s.project.item(item).and_then(|i| i.as_media()).and_then(|m| m.info.video.clone()).unwrap_or_else(|| fail("not a video"));
    let rate = info.frame_rate;
    let mut p = (*s.project).clone();
    let seq = p.new_sequence("Profile check", SequenceSettings { width: info.width, height: info.height, frame_rate: rate, ..Default::default() }, 1, 1, None);
    let src = TimeRange::new(Tick::from_seconds_f64(from), Tick::from_seconds_f64(to - from));
    let mut v = p.make_track_item(item, TrackKind::Video, Tick::ZERO, src, rate).unwrap_or_else(|| fail("cannot place the clip"));
    for e in &mut v.effects {
        filmcraft_project::resolve_auto_points(e, (info.width, info.height), (info.width, info.height));
    }
    let clip = v.id;
    match p.sequence_mut(seq).and_then(|q| q.video_tracks.first_mut()) {
        Some(t) => t.items.push(v),
        None => fail("the sequence has no video track"),
    }
    s.project = std::sync::Arc::new(p);
    s.state.active_sequence = Some(seq);
    s.state.open_sequences = vec![seq];
    s.state.selection = vec![clip];
    (s, seq)
}

fn export(s: &mut Session, seconds: f64, path: &Path) {
    s.set_playhead(Tick::from_seconds_f64(seconds));
    if let Err(e) = s.execute("file.exportFrame", json!({"target": "program", "path": path.to_string_lossy()})) {
        eprintln!("export {}: {e}", path.display());
    }
}

/// The QA's answer on the whole window, every quarter second.
fn qa(s: &mut Session, len: f64, exclude: &Option<Value>) -> Value {
    let seconds: Vec<f64> = (0..).map(|i| f64::from(i) * 0.25).take_while(|t| *t < len).collect();
    let mut p = json!({"seconds": seconds, "profiles": true});
    if let Some(e) = exclude {
        p["exclude"] = e.clone();
    }
    s.execute("sequence.checkPrivacy", p).unwrap_or_else(|e| fail(e))
}

fn redact(s: &mut Session, profiles: bool, dry: bool, exclude: &Option<Value>) -> Value {
    let t = std::time::Instant::now();
    let mut p = json!({"wait": true, "profiles": profiles, "dryRun": dry, "kinds": "faces"});
    if let Some(e) = exclude {
        p["exclude"] = e.clone();
    }
    let r = s.execute("clip.autoRedact", p).unwrap_or_else(|e| fail(e));
    eprintln!("autoRedact profiles={profiles} dryRun={dry}: {:.1} s", t.elapsed().as_secs_f64());
    r["redact"].clone()
}

/// A crop `side` × `side` around a normalised box in each of `frames`, side by side.
fn triptych(frames: &[PathBuf], b: [f64; 4], out: &Path) {
    let imgs: Vec<image::RgbaImage> = frames.iter().filter_map(|p| image::open(p).ok().map(|i| i.to_rgba8())).collect();
    let Some(first) = imgs.first() else { return };
    let (w, h) = (f64::from(first.width()), f64::from(first.height()));
    let side = (b[2] * w).max(b[3] * h) * 2.4;
    let (cx, cy) = ((b[0] + b[2] / 2.0) * w, (b[1] + b[3] / 2.0) * h);
    let (x0, y0) = ((cx - side / 2.0).clamp(0.0, (w - side).max(0.0)), (cy - side / 2.0).clamp(0.0, (h - side).max(0.0)));
    let s = side.min(w).min(h) as u32;
    let tile = 320u32;
    let mut sheet = image::RgbaImage::from_pixel(tile * imgs.len() as u32 + 8 * (imgs.len() as u32 - 1), tile, image::Rgba([255, 255, 255, 255]));
    for (i, img) in imgs.iter().enumerate() {
        let c = image::imageops::crop_imm(img, x0 as u32, y0 as u32, s, s).to_image();
        let c = image::imageops::resize(&c, tile, tile, image::imageops::FilterType::Triangle);
        image::imageops::overlay(&mut sheet, &c, i64::from(i as u32 * (tile + 8)), 0);
    }
    let _ = sheet.save(out);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, video, from, to, out, ..] = args.as_slice() else {
        fail("usage: profile_redact <video> <from s> <to s> <out dir> [--no-judge] [--exclude x,y,w,h] [--dump step]");
    };
    let video = PathBuf::from(video);
    let (from, to): (f64, f64) = (from.parse().unwrap_or_else(|_| fail("from")), to.parse().unwrap_or_else(|_| fail("to")));
    if !(from.is_finite() && to.is_finite() && from >= 0.0 && to > from) {
        fail("from and to: 0 ≤ from < to");
    }
    let out = PathBuf::from(out);
    let judge = !args.iter().any(|a| a == "--no-judge");
    let (frames_dir, crops_dir) = (out.join("frames"), out.join("crops"));
    for d in [&out, &frames_dir, &crops_dir] {
        std::fs::create_dir_all(d).unwrap_or_else(|e| fail(e));
    }
    let (mut s, _) = session(&video, from, to);
    let exclude = args.iter().position(|a| a == "--exclude").and_then(|i| args.get(i + 1)).map(|v| {
        let n: Vec<f64> = v.split(',').filter_map(|x| x.parse().ok()).collect();
        let clip = s.state.selection.first().map(|c| c.0);
        json!({"clip": clip, "x": n.first(), "y": n.get(1), "w": n.get(2), "h": n.get(3)})
    });
    let len = to - from;
    // `--dump <step>`: only the window's frames every `step` seconds, into frames/all
    if let Some(step) = args.iter().position(|a| a == "--dump").and_then(|i| args.get(i + 1)).and_then(|v| v.parse::<f64>().ok()).filter(|s| *s > 0.01) {
        let all = frames_dir.join("all");
        std::fs::create_dir_all(&all).unwrap_or_else(|e| fail(e));
        let mut t = 0.0;
        while t < len {
            export(&mut s, t, &all.join(format!("{:07.3}.png", t + from)));
            t += step;
        }
        return;
    }
    if judge {
        s.redacts.judge = Some(filmcraft_autocut::privacy::engine_judge());
    }
    // what the profile pass finds, and which moments to show
    let dry = redact(&mut s, true, true, &exclude);
    let mut picks: Vec<(String, [f64; 4], f64)> = Vec::new();
    for c in dry["clips"].as_array().into_iter().flatten() {
        for t in c["tracks"].as_array().into_iter().flatten() {
            let only_pose = t["fromPose"].as_u64().unwrap_or(0) > 0 && t["fromPose"].as_u64() == t["hits"].as_u64();
            let what = match (t["hidden"].as_bool(), t["reason"].as_str(), only_pose) {
                (Some(true), _, true) => "profile",
                (Some(true), _, false) if t["fromPose"].as_u64().unwrap_or(0) > 0 => "faceAndProfile",
                (Some(false), Some("notAPerson"), _) => "leftSharp",
                _ => continue,
            };
            let b = &t["box"];
            let (Some(sec), Some(x), Some(y), Some(w), Some(h)) = (b["seconds"].as_f64(), b["x"].as_f64(), b["y"].as_f64(), b["w"].as_f64(), b["h"].as_f64())
            else {
                continue;
            };
            picks.push((what.to_string(), [x, y, w, h], sec - from));
        }
    }
    let times: Vec<f64> = {
        let mut t: Vec<f64> = picks.iter().map(|p| p.2).collect();
        t.sort_by(f64::total_cmp);
        t.dedup_by(|a, b| (*a - *b).abs() < 1e-3);
        t
    };
    let name = |t: f64, stage: &str| frames_dir.join(format!("{:07.3}_{stage}.png", t + from));
    for &t in &times {
        export(&mut s, t, &name(t, "original"));
    }
    let qa_original = qa(&mut s, len, &exclude);
    // before: the face detector alone
    let before = redact(&mut s, false, false, &exclude);
    for &t in &times {
        export(&mut s, t, &name(t, "before"));
    }
    let qa_before = qa(&mut s, len, &exclude);
    // the people the QA still saw with the face detector alone: are they hidden after?
    let mut caught: Vec<(String, [f64; 4], f64)> = Vec::new();
    for u in qa_before["unmasked"].as_array().into_iter().flatten().filter(|u| u["allowed"].as_bool() == Some(false)) {
        let b = &u["box"];
        let (Some(t), Some(x), Some(y), Some(w), Some(h)) = (u["seconds"].as_f64(), b["x"].as_f64(), b["y"].as_f64(), b["w"].as_f64(), b["h"].as_f64()) else {
            continue;
        };
        caught.push(("qaBefore".to_string(), [x, y, w, h], t));
    }
    let more: Vec<f64> = {
        let mut t: Vec<f64> = caught.iter().map(|p| p.2).filter(|t| !times.iter().any(|k| (k - t).abs() < 1e-3)).collect();
        t.sort_by(f64::total_cmp);
        t.dedup_by(|a, b| (*a - *b).abs() < 1e-3);
        t
    };
    for &t in &more {
        export(&mut s, t, &name(t, "before"));
    }
    if !more.is_empty() {
        // the original of those moments, from a session without the redaction
        let (mut o, _) = session(&video, from, to);
        for &t in &more {
            export(&mut o, t, &name(t, "original"));
        }
    }
    picks.extend(caught);
    // after: with the profile pass
    let after = redact(&mut s, true, false, &exclude);
    for &t in times.iter().chain(&more) {
        export(&mut s, t, &name(t, "after"));
    }
    let qa_after = qa(&mut s, len, &exclude);
    for (i, (what, b, t)) in picks.iter().enumerate() {
        let frames: Vec<PathBuf> = ["original", "before", "after"].iter().map(|st| name(*t, st)).collect();
        triptych(&frames, *b, &crops_dir.join(format!("{:02}_{what}_{:.2}s.png", i + 1, t + from)));
    }
    let count = |r: &Value, k: &str| r["clips"].as_array().map_or(0, |c| c.iter().map(|c| c[k].as_u64().unwrap_or(0)).sum::<u64>());
    let skipped = |r: &Value, k: &str| r["clips"].as_array().map_or(0, |c| c.iter().map(|c| c["skipped"][k].as_u64().unwrap_or(0)).sum::<u64>());
    let summary = json!({
        "video": video.to_string_lossy(),
        "window": [from, to],
        "judge": judge,
        "before": {"facesHidden": count(&before, "faces"), "leftSharpNotAPerson": skipped(&before, "notAPerson"), "qaShown": qa_before["shown"], "qaUnmasked": qa_before["unmasked"].as_array().map_or(0, Vec::len)},
        "after": {"facesHidden": count(&after, "faces"), "fromPose": count(&after, "facesFromPose"), "onlyFromPose": count(&after, "facesOnlyFromPose"), "leftSharpNotAPerson": skipped(&after, "notAPerson"), "qaShown": qa_after["shown"], "qaUnmasked": qa_after["unmasked"].as_array().map_or(0, Vec::len)},
        "original": {"qaShown": qa_original["shown"], "qaUnmasked": qa_original["unmasked"].as_array().map_or(0, Vec::len)},
        "seconds": {"before": before["seconds"], "after": after["seconds"]},
        "outcomes": {"dry": dry, "before": before, "after": after},
        "qa": {"original": qa_original, "before": qa_before, "after": qa_after},
    });
    let _ = std::fs::write(out.join("summary.json"), serde_json::to_vec_pretty(&summary).unwrap_or_default());
    println!("{}", serde_json::to_string_pretty(&json!({"before": summary["before"], "after": summary["after"], "original": summary["original"], "seconds": summary["seconds"], "crops": picks.len()})).unwrap_or_default());
}
