//! The creator tools' demo reel, built with ordinary engine commands from the demo project's
//! synthetic footage: 20 s at 1920×1080, eight shots cut on the strong beats of generated music, each
//! with one of the Creator effects (and its name on screen), two speed ramps, Creator
//! transitions on the strong beats with their paired sounds, and zoom pulses on the beats.
//!
//! `FILMCRAFT_FX_DEMO=<dir> cargo test -p filmcraft-cli creator_demo_reel -- --ignored` writes
//! `<dir>/creator_reel.mp4`, `<dir>/creator_reel.sheet.jpg` (a frame of every shot and of every
//! transition, by time) and `<dir>/creator_reel.json` (what is where).

#[cfg(test)]
mod tests {
    use filmcraft_engine::Session;
    use filmcraft_time::TICKS_PER_SECOND;
    use serde_json::{Value, json};

    fn ex(s: &mut Session, id: &str, p: Value) -> Value {
        s.execute(id, p.clone()).unwrap_or_else(|e| panic!("{id} {p}: {e}"))
    }

    fn secs(t: i64) -> f64 {
        t as f64 / TICKS_PER_SECOND as f64
    }

    /// (label, effects with their settings, speed ramp, beat pulse).
    type Shot = (&'static str, Vec<(&'static str, Vec<(&'static str, Value)>)>, Option<&'static str>, bool);

    fn shots() -> Vec<Shot> {
        vec![
            ("Tech HUD + Film Grain", vec![("cr_tech_hud", vec![]), ("cr_film_grain", vec![("amount", json!(25.0))])], None, true),
            ("VHS", vec![("cr_vhs", vec![("amount", json!(85.0))])], None, true),
            ("Speed ramp: Montage + Directional Motion Blur", vec![("cr_motion_blur", vec![("length", json!(24.0))])], Some("montage"), false),
            ("Glitch", vec![("cr_glitch", vec![("amount", json!(70.0)), ("rhythm", json!(0))])], None, true),
            (
                "Kaleidoscope + Zoom Blur",
                vec![("cr_kaleidoscope", vec![("segments", json!(8.0))]), ("cr_zoom_blur", vec![("amount", json!(12.0))])],
                None,
                false,
            ),
            ("Speed ramp: Hero + Chromatic Aberration", vec![("cr_chromatic_aberration", vec![("amount", json!(10.0))])], Some("hero"), false),
            ("Duotone + Light Sweep", vec![("cr_duotone", vec![]), ("cr_light_sweep", vec![("cycle", json!(1.2)), ("intensity", json!(120.0))])], None, true),
            ("Halftone", vec![("cr_halftone", vec![("size", json!(14.0))])], None, true),
        ]
    }

    #[test]
    #[ignore]
    fn creator_demo_reel() {
        let Some(dir) = std::env::var_os("FILMCRAFT_FX_DEMO").map(std::path::PathBuf::from) else { return };
        std::fs::create_dir_all(&dir).unwrap();
        let mut s = Session::default();
        ex(&mut s, "file.openDemoProject", json!({}));
        // the demo's synthetic scenes (pictures), longest first
        let mut scenes: Vec<(u64, f64)> = s
            .project
            .items
            .values()
            .filter(|i| i.has_video() && (i.name.ends_with(".mp4") || i.name.ends_with(".mov")))
            .map(|i| (i.id.0, secs(i.duration().0)))
            .collect();
        scenes.sort_by(|a, b| b.1.total_cmp(&a.1));
        assert!(scenes.len() >= 4, "{scenes:?}");
        ex(&mut s, "file.newSequence", json!({"name": "Creator Reel", "width": 1920, "height": 1080, "fps": 30, "video": 3, "audio": 3}));
        // the music: 120 BPM, its beats are the grid
        let music =
            ex(&mut s, "audio.generate", json!({"kind": "music", "duration": 20.5, "mood": "energetic", "bpm": 120, "seed": 11, "time": 0, "track": "A1"}));
        let beats: Vec<i64> = music["beats"].as_array().unwrap().iter().filter_map(Value::as_i64).collect();
        let beat_s: Vec<f64> = beats.iter().map(|b| secs(*b)).collect();
        // shots of four and six beats (2 s and 3 s at 120 BPM): every cut on a strong beat
        let beat_no = [0usize, 4, 10, 14, 20, 24, 30, 34, 40];
        let cut_at: Vec<f64> = beat_no.iter().map(|n| if *n == 0 { 0.0 } else { beat_s.get(*n).copied().unwrap_or(*n as f64 * 0.5) }).collect();
        let mut placed = Vec::new();
        for (k, (label, effects, ramp, pulse)) in shots().into_iter().enumerate() {
            let (a, b) = (cut_at[k], cut_at[k + 1]);
            let (item, len) = scenes[k % scenes.len()];
            let src_in = (1.0 + 1.7 * k as f64).min((len - 8.0).max(0.0));
            // a ramped shot plays more media: as much as its curve's mean speed takes in its length
            let mut media = b - a;
            if let Some(r) = ramp {
                let probe = ex(
                    &mut s,
                    "timeline.place",
                    json!({"item": item, "track": "V1", "time": (a * TICKS_PER_SECOND as f64) as i64, "sourceIn": (src_in * TICKS_PER_SECOND as f64) as i64, "duration": ((b - a) * TICKS_PER_SECOND as f64) as i64, "audio": false}),
                );
                let c = probe["clips"][0].as_u64().unwrap();
                let m = ex(&mut s, "clip.speedRamp", json!({"clips": [c], "preset": r, "ripple": false}));
                media *= m["clips"][0]["meanSpeed"].as_f64().unwrap();
                ex(&mut s, "edit.undo", json!({}));
                ex(&mut s, "edit.undo", json!({}));
            }
            let r = ex(
                &mut s,
                "timeline.place",
                json!({"item": item, "track": "V1", "time": (a * TICKS_PER_SECOND as f64) as i64, "sourceIn": (src_in * TICKS_PER_SECOND as f64) as i64, "duration": (media * TICKS_PER_SECOND as f64).round() as i64, "audio": false}),
            );
            let clip = r["clips"][0].as_u64().unwrap();
            if let Some(r) = ramp {
                ex(&mut s, "clip.speedRamp", json!({"clips": [clip], "preset": r, "ripple": false}));
            }
            for (effect, params) in &effects {
                ex(&mut s, "effects.apply", json!({"clips": [clip], "effect": effect}));
                for (param, value) in params {
                    ex(&mut s, "effects.setParam", json!({"clip": clip, "effect": effect, "param": param, "value": value}));
                }
            }
            // its name on screen, low in the frame
            ex(
                &mut s,
                "graphics.newText",
                json!({"text": label, "seconds": b - a, "time": (a * TICKS_PER_SECOND as f64) as i64, "size": 54, "position": [80, 930], "box": [1760, 90]}),
            );
            placed.push((k, label, clip, a, b, ramp, pulse, effects.iter().map(|e| e.0).collect::<Vec<_>>()));
        }
        // Creator transitions on the strong beats (every other beat), their whooshes and hits
        let tr = ex(
            &mut s,
            "transitions.randomOnBeats",
            json!({"folder": "Creator", "seed": 4, "everyNthBeat": 2, "beats": beat_s, "bpm": 120, "track": "V1", "sound": true}),
        );
        // zoom pulses on every other beat of the pulsed shots
        let pulsed: Vec<u64> = placed.iter().filter(|p| p.6).map(|p| p.2).collect();
        let pulse =
            ex(&mut s, "sequence.beatPulse", json!({"clips": pulsed, "effect": "zoomPulse", "strength": 0.5, "everyNthBeat": 2, "beats": beat_s, "bpm": 120}));
        // the file
        let out = dir.join("creator_reel.mp4");
        let exported = ex(&mut s, "file.exportMedia", json!({"path": out.to_string_lossy(), "format": "h264", "wait": true}));
        // a frame of every shot and every transition, from the file
        let imp = ex(&mut s, "file.import", json!({"paths": [out.to_string_lossy()]}));
        let file_item = imp["items"][0].as_u64().unwrap();
        let mut at: Vec<f64> = placed.iter().map(|p| (p.3 + p.4) / 2.0).collect();
        let transitions: Vec<Value> = tr["transitions"].as_array().cloned().unwrap_or_default();
        at.extend(transitions.iter().filter_map(|t| t["cut"].as_i64()).map(secs));
        at.sort_by(f64::total_cmp);
        let sheet = dir.join("creator_reel.sheet.jpg");
        ex(
            &mut s,
            "media.contactSheet",
            json!({"items": [{"item": file_item, "at": at}], "side": 480, "columns": 4, "labels": "time", "path": sheet.to_string_lossy()}),
        );
        let manifest = json!({
            "file": out, "sheet": sheet, "export": exported,
            "shots": placed.iter().map(|(k, label, clip, a, b, ramp, pulse, effects)| json!({"shot": k + 1, "label": label, "clip": clip, "start": a, "end": b, "effects": effects, "speedRamp": ramp, "beatPulse": pulse})).collect::<Vec<_>>(),
            "transitions": transitions.iter().map(|t| json!({"effect": t["effect"], "name": filmcraft_engine::project::find_effect(t["effect"].as_str().unwrap_or("")).map(|d| d.name), "cut": t["cut"].as_i64().map(secs), "beat": t["beat"].as_i64().map(secs)})).collect::<Vec<_>>(),
            "beatPulses": pulse["clips"],
            "beats": beat_s,
        });
        std::fs::write(dir.join("creator_reel.json"), serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
        assert!(out.exists() && sheet.exists());
        assert!(!transitions.is_empty(), "transitions on the beats");
    }
}
