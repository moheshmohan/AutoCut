//! Headless UI test of the Generate Audio panel: pick a sound, set it up through the control
//! channel, and Add at playhead puts a generated cue on the timeline as one undo step.

use std::sync::mpsc::{Sender, channel};

use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
}

impl Driver {
    fn demo(scratch: &str) -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).expect("demo project");
        session.execute("file.projectSettings.scratchDisks", json!({"captured": scratch})).expect("scratch disk");
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(session).with_control(rx);
        let harness = Harness::builder().with_size(egui::vec2(1600.0, 980.0)).with_max_steps(10_000).build_eframe(move |_cc| app);
        let mut d = Driver { harness, tx };
        d.frames(4);
        d
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            let ctx = self.harness.ctx.clone();
            let mut raw = std::mem::take(self.harness.input_mut());
            eframe::App::raw_input_hook(self.harness.state_mut(), &ctx, &mut raw);
            *self.harness.input_mut() = raw;
            self.harness.step();
        }
    }

    fn ok(&mut self, method: &str, params: Value) -> Value {
        let (req, reply) = ControlRequest::new(method, params.clone());
        self.tx.send(req).unwrap();
        for _ in 0..600 {
            self.frames(1);
            if let Ok(v) = reply.try_recv() {
                assert_eq!(v["ok"], json!(true), "{method} {params} failed: {v}");
                return v["result"].clone();
            }
        }
        panic!("no reply to {method} {params}");
    }

    fn exec(&mut self, command: &str, params: Value) -> Value {
        self.ok("engine.execute", json!({"command": command, "params": params}))
    }

    fn audio_clips(&mut self) -> Vec<Value> {
        let q = self.exec("sequence.inspect", json!({}));
        q["audio"].as_array().unwrap().iter().flat_map(|t| t["items"].as_array().cloned().unwrap_or_default()).collect()
    }
}

#[test]
fn add_at_playhead_places_a_generated_impact() {
    let scratch = std::env::temp_dir().join(format!("filmcraft-genaudio-ui-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    let mut d = Driver::demo(&scratch.to_string_lossy());
    d.ok("ui.panel.show", json!({"panel": "Generate Audio"}));
    d.frames(3);
    let ids: Vec<String> =
        d.ok("ui.elements", json!({"prefix": "generateAudio."})).as_array().unwrap().iter().filter_map(|e| e["id"].as_str().map(str::to_string)).collect();
    for id in ["generateAudio.kind", "generateAudio.length", "generateAudio.intensity", "generateAudio.variation", "generateAudio.add"] {
        assert!(ids.iter().any(|i| i == id), "{id} in {ids:?}");
    }
    // music settings show only for music
    assert!(!ids.iter().any(|i| i == "generateAudio.mood"), "{ids:?}");
    d.ok("ui.set", json!({"generateAudio": {"kind": "music", "seconds": 4.0}}));
    d.frames(3);
    let ids = d.ok("ui.elements", json!({"prefix": "generateAudio."})).to_string();
    assert!(ids.contains("generateAudio.mood") && ids.contains("generateAudio.tempo") && ids.contains("generateAudio.key"), "{ids}");
    // an impact at 2 s
    d.ok("ui.set", json!({"generateAudio": {"kind": "impact", "seconds": 1.5, "seed": 3}}));
    d.exec("playhead.set", json!({"seconds": 2.0}));
    let before = d.audio_clips().len();
    d.ok("ui.click", json!({"id": "generateAudio.add"}));
    d.frames(3);
    let after = d.audio_clips();
    assert_eq!(after.len(), before + 1, "one new audio clip");
    let impact = after.iter().find(|c| c["name"].as_str().is_some_and(|n| n.starts_with("Impact"))).expect("an Impact clip");
    // at the playhead (on the demo sequence's frame grid)
    let start = impact["start"].as_i64().unwrap() as f64 / 254_016_000_000.0;
    assert!((start - 2.0).abs() < 0.05, "{start}");
    let status = d.ok("ui.inspect", json!({}))["ui"]["status"].as_str().unwrap_or_default().to_string();
    assert!(status.starts_with("Added Impact"), "{status}");
    // written beside the project's captured media
    assert!(std::fs::read_dir(&scratch).unwrap().any(|e| e.unwrap().file_name().to_string_lossy().starts_with("Impact")));
    let _ = std::fs::remove_dir_all(&scratch);
}
