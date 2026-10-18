//! The vertical-factory demo (roadmap E3), rendered by hand:
//!
//! ```text
//! FILMCRAFT_DEMO_DIR=target/vertical-demo cargo test --release -p filmcraft-autocut --test vertical_demo -- --ignored --nocapture
//! ```
//!
//! 1. A 16:9 "long video" made with FilmCraft's own export from the procedural demo footage: a neon
//!    "subject" moving around a sunset, a cut to an aurora sky with the same subject elsewhere,
//!    and a narrated script (the built-in formant voice) → `source-16x9.mp4`.
//! 2. That file imported as a user's would be, its transcript set from the script, and
//!    `sequence.autoReframe` to 9:16 following the subject → `reframed-9x16.mp4`.
//! 3. `autocut.shorts`' recipe on it: the best passages on whole sentences, reframed, captioned,
//!    titled and exported → `shorts/NN-<title>.mp4`, and `summary.json` with what was made.

use std::path::PathBuf;

use filmcraft_autocut::{Executor, SessionExecutor};
use filmcraft_engine::Session;
use serde_json::{Value, json};

const TPS: f64 = filmcraft_time::TICKS_PER_SECOND as f64;
const W: f64 = 1280.0;
const H: f64 = 720.0;
const LENGTH: f64 = 30.0;
const CUT: f64 = 15.0;
/// The subject's size in the long video (the 1920×1080 neon loop at 20 %).
const SUBJECT_W: f64 = 384.0;
const SUBJECT_H: f64 = 216.0;

fn ticks(s: f64) -> i64 {
    (s * TPS).round() as i64
}

const SCRIPT: [&str; 6] = [
    "Why do tiny robots win the big races?",
    "Because a light frame turns faster in every single corner.",
    "Our little racer weighs just 300 grams and never stops moving.",
    "Now watch what happens after the cut.",
    "The same racer races under the northern lights at night.",
    "It never loses the line, not even in the dark.",
];

/// The item named `name` anywhere in the project's bins.
fn find(tree: &Value, name: &str) -> Option<u64> {
    if tree["name"] == name && tree["item"].is_u64() {
        return tree["item"].as_u64();
    }
    tree["children"].as_array()?.iter().find_map(|c| find(c, name))
}

/// Where the subject is at `t` (sequence pixels): around the sunset, then under the aurora.
fn path(t: f64) -> (f64, f64) {
    let tau = std::f64::consts::TAU;
    if t < CUT {
        (W / 2.0 + 430.0 * (tau * t / 9.0).sin(), H / 2.0 + 110.0 * (tau * t / 5.0).sin())
    } else {
        let u = t - CUT;
        (W / 2.0 - 400.0 * (tau * u / 11.0).cos(), H * 0.62 - 60.0 * (tau * u / 6.0).sin())
    }
}

async fn run(ex: &dyn Executor, id: &str, p: Value) -> Value {
    ex.execute(id, p.clone()).await.unwrap_or_else(|e| panic!("{id} {p}: {e}"))
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "renders the demo videos (minutes): run it by hand"]
async fn vertical_demo() {
    let dir = std::env::var_os("FILMCRAFT_DEMO_DIR").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("target/vertical-demo"));
    let dir = std::path::absolute(&dir).unwrap();
    let work = dir.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let t0 = std::time::Instant::now();

    // 1. the long video
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    let ex = SessionExecutor::from(s);
    let tree = run(&ex, "project.inspect", json!({})).await["root"].clone();
    let (ocean, sky, neon) = (find(&tree, "Ocean_Sunset.mp4").unwrap(), find(&tree, "Aurora_Timelapse.mp4").unwrap(), find(&tree, "Neon_Loop.mov").unwrap());
    run(&ex, "file.newSequence", json!({"name": "Long video", "width": W, "height": H, "fps": 24, "video": 2, "audio": 2})).await;
    run(&ex, "timeline.place", json!({"item": ocean, "track": "V1", "time": 0, "sourceIn": ticks(2.0), "duration": ticks(CUT), "audio": false})).await;
    run(&ex, "timeline.place", json!({"item": sky, "track": "V1", "time": ticks(CUT), "sourceIn": 0, "duration": ticks(LENGTH - CUT), "audio": false})).await;
    // the subject: the neon loop at 20 % (384×216, narrower than a 9:16 crop), in pieces as long
    // as the loop
    let mut pieces: Vec<(u64, f64, f64)> = Vec::new();
    let mut at = 0.0;
    while at < LENGTH - 1e-6 {
        let len = (LENGTH - at).min(if at < CUT { CUT - at } else { 10.0 }).min(10.0);
        let r =
            run(&ex, "timeline.place", json!({"item": neon, "track": "V2", "time": ticks(at), "sourceIn": 0, "duration": ticks(len), "audio": false})).await;
        pieces.push((r["clips"][0].as_u64().unwrap(), at, at + len));
        at += len;
    }
    for (clip, a, b) in &pieces {
        run(&ex, "effects.setParam", json!({"clip": clip, "effect": "motion", "param": "scale", "value": 20.0})).await;
        let mut t = *a;
        while t <= *b - 1.0 / 24.0 + 1e-9 {
            let (x, y) = path(t);
            run(&ex, "effects.setParam", json!({"clip": clip, "effect": "motion", "param": "position", "value": [x, y], "time": ticks(t), "keyframe": true}))
                .await;
            t += 0.25;
        }
    }
    // the narration, one line after another (the second half after the cut)
    let mut lines: Vec<(f64, f64, &str)> = Vec::new();
    for (k, text) in SCRIPT.iter().enumerate() {
        let after = lines.last().map_or(0.4, |l| l.1 + 0.6);
        let start = if k == 3 { after.max(CUT + 0.3) } else { after };
        let r = run(&ex, "tts.create", json!({"text": text, "time": ticks(start), "dir": work.to_string_lossy()})).await;
        let item = r["item"].as_u64().unwrap();
        let dur = {
            let seq = run(&ex, "sequence.inspect", json!({})).await;
            seq["audio"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|t| t["items"].as_array().cloned().unwrap_or_default())
                .find(|c| c["item"] == item)
                .and_then(|c| c["duration"].as_f64())
                .unwrap()
                / TPS
        };
        lines.push((start, start + dur, text));
    }
    let source = dir.join("source-16x9.mp4");
    run(&ex, "file.exportMedia", json!({"path": source.to_string_lossy(), "format": "h264", "wait": true})).await;
    eprintln!("demo: {} ({:.0}s)", source.display(), t0.elapsed().as_secs_f64());

    // 2. the file as a user brings it: imported, transcribed (here: the script, word by word
    // across each spoken line), reframed to 9:16
    let ex = SessionExecutor::from(Session::default());
    let item = run(&ex, "file.import", json!({"paths": [source.to_string_lossy()]})).await["items"][0].as_u64().unwrap();
    let seq = run(&ex, "file.newSequenceFromClip", json!({"items": [item]})).await["sequence"].as_u64().unwrap();
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
    // the subject's box every 2.5 s, as the vision model's sampled frames give it (AutoCut's
    // analysis); the tracker follows it in between
    let (bw, bh) = (SUBJECT_W / W, SUBJECT_H / H);
    let seeds: Vec<Value> = (0..12)
        .map(|k| {
            let t = 2.5 * k as f64;
            let (x, y) = path(t);
            json!({"x": x / W - bw / 2.0, "y": y / H - bh / 2.0, "w": bw, "h": bh, "seconds": t})
        })
        .collect();
    let r = run(
        &ex,
        "sequence.autoReframe",
        json!({"sequence": seq, "aspect": "9:16", "motion": "smooth", "subject": seeds, "name": "Long video (9:16)", "wait": true}),
    )
    .await;
    let reframe = r["reframe"].clone();
    let vertical = reframe["sequence"].as_u64().unwrap();
    let reframed = dir.join("reframed-9x16.mp4");
    run(&ex, "file.exportMedia", json!({"sequence": vertical, "path": reframed.to_string_lossy(), "format": "h264", "wait": true})).await;
    eprintln!("demo: {} ({:.0}s): {reframe}", reframed.display(), t0.elapsed().as_secs_f64());
    // stills of the vertical cut, to look at without a player
    let stills = dir.join("reframed-stills");
    std::fs::create_dir_all(&stills).unwrap();
    for t in [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 20.0, 24.0, 28.0] {
        if let Some(png) = ex.render(t, 640).await {
            std::fs::write(stills.join(format!("t{:02}.png", t as u32)), png).unwrap();
        }
    }

    // 3. shorts from the long video
    let p = json!({"sequence": seq, "count": 2, "minSeconds": 6.0, "maxSeconds": 13.0, "aspect": "9:16", "captions": true, "title": true, "export": dir.join("shorts").to_string_lossy()});
    let shorts = filmcraft_autocut::shorts::make(&ex, &p, None, &mut |l: &str| eprintln!("shorts: {l}")).await.unwrap();
    eprintln!("demo: shorts ({:.0}s): {}", t0.elapsed().as_secs_f64(), shorts["status"]);
    let summary = json!({"source": source, "reframed": reframed, "reframe": reframe, "shorts": shorts, "seconds": t0.elapsed().as_secs_f64()});
    std::fs::write(dir.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    for s in shorts["shorts"].as_array().unwrap() {
        assert!(s["export"].is_string(), "{s}");
    }
}
