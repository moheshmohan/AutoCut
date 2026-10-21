//! The blurred fill and the platform safe zones on a vertical short, rendered by hand:
//!
//! ```text
//! FILMCRAFT_DEMO_DIR=target/blur-fill-demo cargo test --release -p filmcraft-autocut --test blur_fill_demo -- --ignored --nocapture
//! ```
//!
//! 1. A 1920×1080 "talk" made with FilmCraft's own export: synthetic footage (a cool, textured
//!    backdrop and a warm, textured disc drifting across it that is gone for a stretch in the
//!    middle; numbered stills imported as an image sequence) and a narrated script (the built-in
//!    formant voice) → `source-16x9.mp4`.
//! 2. That file imported as a user's would be, its transcript set from the script, and
//!    `autocut.shorts`' recipe on it for one 1080×1920 short: reframed with the `blur` fallback
//!    (the subject followed while it is there; the whole picture over its blurred copy, eased in
//!    and out, while it is not), captions and a title inside YouTube Shorts' safe zone →
//!    `short.mp4`.
//! 3. Stills of the short: `still-followed.png` (in the first line), `still-easing.png` (the crop
//!    easing out), `still-blur-fill.png` (the middle of the gap) and `still-safe-zones.png` (that
//!    frame with YouTube Shorts' zones drawn over it), the project (`demo.fcproj`, for the UI's
//!    screenshot: `safe_zones_ui`'s ignored test) and `summary.json`.

use std::path::PathBuf;

use filmcraft_autocut::{Executor, SessionExecutor};
use filmcraft_engine::Session;
use filmcraft_engine::safe_zones::Platform;
use serde_json::{Value, json};

const TPS: f64 = filmcraft_time::TICKS_PER_SECOND as f64;
/// The stills' size; the talk is 1920×1080 (the stills at 200 %).
const FW: u32 = 960;
const FH: u32 = 540;
/// Stills, and the share of them where the subject is gone.
const FRAMES: u32 = 300;
const GONE: (f64, f64) = (0.3, 0.62);

fn ticks(s: f64) -> i64 {
    (s * TPS).round() as i64
}

const SCRIPT: [&str; 3] = ["Watch the little racer cross the field.", "Now it is gone for a moment.", "And here it comes back again."];

fn hash(x: i64, y: i64, seed: u64) -> f32 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h >> 40) as f32 / (1u64 << 24) as f32
}

fn noise(u: f64, v: f64, seed: u64) -> f32 {
    let (x0, y0) = (u.floor(), v.floor());
    let (fx, fy) = ((u - x0) as f32, (v - y0) as f32);
    let (x0, y0) = (x0 as i64, y0 as i64);
    let (a, b, c, d) = (hash(x0, y0, seed), hash(x0 + 1, y0, seed), hash(x0, y0 + 1, seed), hash(x0 + 1, y0 + 1, seed));
    let top = a + (b - a) * fx;
    top + (c + (d - c) * fx - top) * fy
}

fn texture(u: f64, v: f64, seed: u64) -> f32 {
    0.5 * noise(u / 24.0, v / 24.0, seed) + 0.3 * noise(u / 7.0, v / 7.0, seed + 1) + 0.2 * noise(u / 2.5, v / 2.5, seed + 2)
}

/// Where the disc is after `t` (0..1) of its time on screen, in still pixels.
fn disc_at(t: f64) -> (f64, f64) {
    (FW as f64 * (0.3 + 0.3 * t), FH as f64 * (0.52 + 0.06 * (t * 9.0).sin()))
}

/// Still `k`: the backdrop (`bg`, RGB floats) with the disc where it is, unless it is gone.
fn still(bg: &[[f32; 3]], k: u32) -> image::RgbImage {
    let u = k as f64 / FRAMES as f64;
    let shown = 1.0 - (GONE.1 - GONE.0);
    let on = if u < GONE.0 {
        Some(u / shown)
    } else if u < GONE.1 {
        None
    } else {
        Some((u - (GONE.1 - GONE.0)) / shown)
    };
    image::RgbImage::from_fn(FW, FH, |x, y| {
        let mut c = bg[(y * FW + x) as usize];
        if let Some(t) = on {
            let (cx, cy) = disc_at(t);
            let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
            if ((px - cx).powi(2) + (py - cy).powi(2)).sqrt() < 70.0 {
                let w = texture(px - cx + 900.0, py - cy + 100.0, 77);
                c = [0.65 + 0.35 * w, 0.32 + 0.35 * w, 0.1 + 0.15 * w];
            }
        }
        image::Rgb(c.map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8))
    })
}

async fn run(ex: &dyn Executor, id: &str, p: Value) -> Value {
    ex.execute(id, p.clone()).await.unwrap_or_else(|e| panic!("{id} {p}: {e}"))
}

/// `png` with `platform`'s zones drawn over it: translucent red where the interface covers the
/// frame, the safe area outlined in white.
fn with_zones(png: &[u8], platform: Platform) -> Vec<u8> {
    let mut img = image::load_from_memory(png).unwrap().to_rgba8();
    let (w, h) = (img.width(), img.height());
    for z in platform.zones() {
        let [x0, y0, x1, y1] = z.area.in_pixels((w as f64, h as f64)).map(|v| v.round() as u32);
        for y in y0..y1.min(h) {
            for x in x0..x1.min(w) {
                let p = img.get_pixel_mut(x, y);
                let mix = |c: u8, to: f32| (c as f32 * 0.6 + to * 0.4) as u8;
                p.0 = [mix(p.0[0], 255.0), mix(p.0[1], 40.0), mix(p.0[2], 40.0), 255];
            }
        }
    }
    let [x0, y0, x1, y1] = platform.safe_area().in_pixels((w as f64, h as f64)).map(|v| v.round() as u32);
    for k in 0..3 {
        for x in x0..x1.min(w) {
            for y in [y0 + k, y1.saturating_sub(1 + k)] {
                img.put_pixel(x, y.min(h - 1), image::Rgba([255, 255, 255, 255]));
            }
        }
        for y in y0..y1.min(h) {
            for x in [x0 + k, x1.saturating_sub(1 + k)] {
                img.put_pixel(x.min(w - 1), y, image::Rgba([255, 255, 255, 255]));
            }
        }
    }
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "renders the demo short (a minute or two): run it by hand"]
async fn blur_fill_demo() {
    let dir = std::env::var_os("FILMCRAFT_DEMO_DIR").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("target/blur-fill-demo"));
    let dir = std::path::absolute(&dir).unwrap();
    let work = dir.join("work");
    let frames = work.join("frames");
    std::fs::create_dir_all(&frames).unwrap();
    let t0 = std::time::Instant::now();

    // 1. the talk: the footage as numbered stills
    let bg: Vec<[f32; 3]> = (0..FW * FH)
        .map(|i| {
            let (x, y) = ((i % FW) as f64 + 0.5, (i / FW) as f64 + 0.5);
            let b = texture(x + 400.0, y + 200.0, 5);
            [0.12 + 0.25 * b, 0.22 + 0.3 * b, 0.4 + 0.35 * b]
        })
        .collect();
    for k in 0..FRAMES {
        still(&bg, k).save(frames.join(format!("shot{k:04}.png"))).unwrap();
    }
    let ex = SessionExecutor::from(Session::default());
    let r = run(&ex, "file.importImageSequence", json!({"path": frames.join("shot0000.png").to_string_lossy()})).await;
    let footage = r["items"][0].as_u64().unwrap_or_else(|| panic!("{r}"));
    let length = {
        let g = ex.0.lock().unwrap();
        g.project.items.values().find(|i| i.id.0 == footage).and_then(|i| i.as_media()).map(|m| m.info.duration.seconds()).unwrap()
    };
    let (gone, back) = (length * GONE.0, length * GONE.1);
    run(&ex, "file.newSequence", json!({"name": "Talk", "width": 1920, "height": 1080, "fps": 30, "video": 1, "audio": 2})).await;
    let r = run(&ex, "timeline.place", json!({"item": footage, "track": "V1", "time": 0, "audio": false})).await;
    let clip = r["clips"][0].as_u64().unwrap();
    run(&ex, "effects.setParam", json!({"clip": clip, "effect": "motion", "param": "scale", "value": 200.0})).await;
    // the narration: a line while the subject is there, one while it is gone, one when it is back
    let mut lines: Vec<(f64, f64, &str)> = Vec::new();
    for (text, start) in SCRIPT.iter().zip([0.3, gone + 0.3, back + 0.3]) {
        let r = run(&ex, "tts.create", json!({"text": text, "time": ticks(start), "dir": work.to_string_lossy()})).await;
        let item = r["item"].as_u64().unwrap();
        let seq = run(&ex, "sequence.inspect", json!({})).await;
        let dur = seq["audio"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|t| t["items"].as_array().cloned().unwrap_or_default())
            .find(|c| c["item"] == item)
            .and_then(|c| c["duration"].as_f64())
            .unwrap()
            / TPS;
        lines.push((start, start + dur, text));
    }
    let source = dir.join("source-16x9.mp4");
    run(&ex, "file.exportMedia", json!({"path": source.to_string_lossy(), "format": "h264", "wait": true})).await;
    eprintln!("demo: {} ({:.0}s), the subject gone {gone:.2}–{back:.2} s of {length:.2}: {lines:?}", source.display(), t0.elapsed().as_secs_f64());

    // 2. the short
    let ex = SessionExecutor::from(Session::default());
    let item = run(&ex, "file.import", json!({"paths": [source.to_string_lossy()]})).await["items"][0].as_u64().unwrap();
    let mut words = Vec::new();
    for (a, b, text) in &lines {
        let ws: Vec<&str> = text.split_whitespace().collect();
        let step = (b - a) / ws.len() as f64;
        for (k, w) in ws.iter().enumerate() {
            let s0 = a + step * k as f64;
            words.push(json!({"text": w, "start": ticks(s0), "end": ticks(s0 + step * 0.85)}));
        }
    }
    run(&ex, "transcript.set", json!({"item": item, "transcript": {"language": "en", "speakers": [{"name": "Narrator"}], "words": words}})).await;
    let shorts_dir = dir.join("shorts");
    let p = json!({"item": item, "count": 1, "minSeconds": (length - 3.0).max(3.0), "maxSeconds": length + 1.0, "aspect": "9:16", "captions": true, "title": true, "platform": "youtube", "export": shorts_dir.to_string_lossy()});
    let shorts = filmcraft_autocut::shorts::make(&ex, &p, None, &mut |l: &str| eprintln!("shorts: {l}")).await.unwrap();
    let short = shorts["shorts"][0].clone();
    eprintln!("demo: short ({:.0}s): {short}", t0.elapsed().as_secs_f64());
    let file = PathBuf::from(short["export"].as_str().unwrap_or_else(|| panic!("{short}")));
    std::fs::copy(&file, dir.join("short.mp4")).unwrap();
    let seq = short["sequence"].as_u64().unwrap();
    let start = short["start"].as_f64().unwrap_or(0.0);
    let sequence = {
        let g = ex.0.lock().unwrap();
        let q = g.project.items.values().find(|i| i.id.0 == seq).and_then(|i| i.as_sequence()).unwrap();
        json!({
            "size": [q.settings.width, q.settings.height],
            "tracks": q.video_tracks.iter().map(|t| json!({"name": t.name, "clips": t.items.iter().map(|i| json!({"name": i.name, "start": i.start.seconds(), "end": i.end().seconds()})).collect::<Vec<_>>()})).collect::<Vec<_>>(),
            "safeZones": filmcraft_engine::safe_zones::check(&g.project, q, Platform::Youtube),
        })
    };

    // 3. stills, in the short's time
    // half way through the eased change: a little after the blurred copy starts under it
    let fill_start = sequence["tracks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "Blur Fill")
        .and_then(|t| t["clips"][0]["start"].as_f64())
        .map_or(gone + 1.2, |s| s + start);
    for (name, t) in [("followed", lines[0].0 + 1.0), ("easing", fill_start + 0.3), ("blur-fill", (gone + back) / 2.0 + 0.4)] {
        let png = ex.render(t - start, 1920).await.unwrap();
        std::fs::write(dir.join(format!("still-{name}.png")), &png).unwrap();
        if name == "blur-fill" {
            std::fs::write(dir.join("still-safe-zones.png"), with_zones(&png, Platform::Youtube)).unwrap();
        }
    }
    run(&ex, "file.saveAs", json!({"path": dir.join("demo.fcproj").to_string_lossy()})).await;
    let summary = json!({"source": source, "short": dir.join("short.mp4"), "gone": [gone, back], "made": short, "sequence": sequence, "seconds": t0.elapsed().as_secs_f64()});
    std::fs::write(dir.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    eprintln!("demo: done in {:.0}s → {}", t0.elapsed().as_secs_f64(), dir.display());
    assert!(short["blurFill"].as_u64().is_some_and(|n| n > 0), "the blurred fill under the lost stretch: {short}");
    assert_eq!(short["safeZones"]["ok"], true, "{short}");
}
