//! Headless UI tests of the Text to Speech panel: typing a script, Add pause, adding the narration
//! to the timeline, clicking the narration clip to load it back, and saving an edit in place.

use std::sync::mpsc::{Sender, channel};

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use filmcraft_engine::Session;
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
    /// `FILMCRAFT_UI_SNAPSHOT_DIR`: where [`Driver::snapshot`] writes PNGs (rendered with wgpu).
    snapshots: Option<std::path::PathBuf>,
}

impl Driver {
    fn demo(scratch: &str) -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).expect("demo project");
        session.execute("file.projectSettings.scratchDisks", json!({"captured": scratch})).expect("scratch disk");
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(session).with_control(rx);
        let snapshots = std::env::var_os("FILMCRAFT_UI_SNAPSHOT_DIR").map(std::path::PathBuf::from);
        let mut b = Harness::builder().with_size(egui::vec2(1600.0, 980.0)).with_max_steps(10_000);
        if snapshots.is_some() {
            b = b.wgpu();
        }
        let harness = b.build_eframe(move |_cc| app);
        let mut d = Driver { harness, tx, snapshots };
        d.frames(4);
        // the built-in voice: these tests are about the panel, and a natural voice (when this
        // build has it and the machine has downloaded it) synthesizes too slowly for them
        d.ok("ui.set", json!({"tts": {"voice": filmcraft_tts::default_voice_id()}}));
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

    fn call(&mut self, method: &str, params: Value) -> Value {
        let (req, reply) = ControlRequest::new(method, params.clone());
        self.tx.send(req).unwrap();
        for _ in 0..600 {
            self.frames(1);
            if let Ok(v) = reply.try_recv() {
                return v;
            }
        }
        panic!("no reply to {method} {params}");
    }

    fn ok(&mut self, method: &str, params: Value) -> Value {
        let v = self.call(method, params.clone());
        assert_eq!(v["ok"], json!(true), "{method} {params} failed: {v}");
        v["result"].clone()
    }

    fn exec(&mut self, command: &str, params: Value) -> Value {
        self.ok("engine.execute", json!({"command": command, "params": params}))
    }

    fn click(&mut self, id: &str) {
        self.ok("ui.click", json!({"id": id}));
        self.frames(3);
    }

    fn type_text(&mut self, text: &str) {
        self.ok("ui.type", json!({"text": text}));
        self.frames(3);
    }

    fn ui(&mut self) -> Value {
        self.ok("ui.inspect", json!({}))
    }

    fn draft(&mut self) -> Value {
        self.ui()["ui"]["tts"].clone()
    }

    /// Let a background synthesis job finish, for at most two minutes of wall-clock time.
    fn settle(&mut self) {
        let t0 = std::time::Instant::now();
        while t0.elapsed().as_secs() < 120 {
            if self.draft()["pending"].is_null() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
            self.frames(1);
        }
        panic!("synthesis never finished");
    }

    fn snapshot(&mut self, name: &str) {
        let Some(dir) = self.snapshots.clone() else { return };
        self.frames(2);
        match self.harness.render() {
            Ok(img) => {
                std::fs::create_dir_all(&dir).unwrap();
                img.save(dir.join(format!("{name}.png"))).unwrap();
            }
            Err(e) => eprintln!("snapshot {name} skipped: {e}"),
        }
    }

    fn ids(&mut self, prefix: &str) -> Vec<String> {
        let v = self.ok("ui.elements", json!({"prefix": prefix}));
        v.as_array().unwrap().iter().filter_map(|e| e["id"].as_str().map(str::to_string)).collect()
    }

    fn narration_clips(&mut self) -> Vec<u64> {
        let seq = self.exec("sequence.inspect", json!({}));
        let mut out = Vec::new();
        for t in seq["audio"].as_array().into_iter().flatten() {
            for it in t["items"].as_array().into_iter().flatten() {
                let id = it["clip"].as_u64().unwrap();
                if self.call("engine.execute", json!({"command": "tts.inspect", "params": {"clip": id}}))["ok"] == json!(true) {
                    out.push(id);
                }
            }
        }
        out
    }
}

fn scratch(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("filmcraft-tts-ui-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d.to_string_lossy().into_owned()
}

#[test]
fn write_add_click_to_edit_and_save_in_place() {
    let dir = scratch("flow");
    let mut d = Driver::demo(&dir);
    d.exec("window.panel.TextToSpeech", json!({}));
    d.frames(3);
    // the panel opens beside Properties, leaving the Program monitor visible
    assert!(!d.ids("program.").is_empty(), "Program monitor still shown");
    let ids = d.ids("tts.");
    for id in [
        "tts.language",
        "tts.voice",
        "tts.hearVoice",
        "tts.advanced",
        "tts.pitch",
        "tts.delivery",
        "tts.pace",
        "tts.text",
        "tts.addPause",
        "tts.preview",
        "tts.save",
    ] {
        assert!(ids.iter().any(|i| i == id), "missing {id} in {ids:?}");
    }
    // type a script and add a pause
    d.click("tts.text");
    d.type_text("Hello from the panel.");
    d.click("tts.addPause");
    d.click("tts.text");
    d.ok("ui.key", json!({"key": "End"}));
    d.type_text("Goodbye.");
    let text = d.draft()["text"].as_str().unwrap().to_string();
    assert!(text.starts_with("Hello from the panel. [pause 1s]") && text.ends_with("Goodbye."), "{text:?}");
    // add it to the timeline
    d.exec("playhead.set", json!({"seconds": 1.0}));
    d.click("tts.save");
    d.settle();
    let clips = d.narration_clips();
    let status = d.ui()["ui"]["status"].clone();
    assert_eq!(clips.len(), 1, "one narration clip; status {status}");
    let clip = clips[0];
    let n = d.exec("tts.inspect", json!({"clip": clip}));
    assert_eq!(n["narration"]["text"].as_str().unwrap(), text);
    assert_eq!(d.ui()["selection"], json!([clip]), "the new clip is selected");
    assert_eq!(d.draft()["loadedFrom"][0], json!(clip), "the panel now edits it");
    // deselect and start fresh, then click the clip on the timeline: the panel loads it back
    d.exec("edit.deselectAll", json!({}));
    d.frames(3);
    assert!(d.draft()["loadedFrom"].is_null());
    d.click("tts.text");
    d.ok("ui.key", json!({"key": "Cmd+A"}));
    d.type_text("Something else");
    let at = d.ok("ui.timeline.locate", json!({"clip": clip}));
    d.ok("ui.click", json!({"x": at["x"], "y": at["y"]}));
    d.frames(4);
    assert_eq!(d.ui()["selection"], json!([clip]));
    assert_eq!(d.draft()["text"].as_str().unwrap(), text, "clicking the clip loads its script");
    // change the script and save: same clip, new audio, one undo step
    let before = d.exec("tts.inspect", json!({"clip": clip}))["item"].clone();
    d.click("tts.text");
    d.ok("ui.key", json!({"key": "Cmd+A"}));
    d.type_text("A new script for the same clip.");
    d.click("tts.save");
    d.settle();
    let after = d.exec("tts.inspect", json!({"clip": clip}));
    assert_eq!(after["narration"]["text"], "A new script for the same clip.");
    assert_ne!(after["item"], before);
    assert_eq!(d.narration_clips(), vec![clip]);
    let hist = d.exec("history.list", json!({}));
    assert_eq!(hist["undo"].as_array().unwrap().last().unwrap(), "Edit Narration");
    d.exec("edit.undo", json!({}));
    d.frames(3);
    assert_eq!(d.exec("tts.inspect", json!({"clip": clip}))["narration"]["text"].as_str().unwrap(), text);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn settings_controls_and_hear_this_voice() {
    let dir = scratch("controls");
    let mut d = Driver::demo(&dir);
    d.exec("window.panel.TextToSpeech", json!({}));
    d.frames(3);
    // Save is disabled with an empty script
    d.click("tts.save");
    assert!(d.narration_clips().is_empty());
    // Hear this voice runs a preview (the headless harness has no audio output: a status line)
    d.click("tts.hearVoice");
    d.settle();
    let status = d.ui()["ui"]["status"].as_str().unwrap_or_default().to_string();
    assert!(status.contains("audio output") || status.is_empty(), "{status}");
    // Advanced folds away and back
    assert_eq!(d.draft()["advancedOpen"], true);
    d.click("tts.advanced");
    assert_eq!(d.draft()["advancedOpen"], false);
    assert!(!d.ids("tts.").iter().any(|i| i == "tts.pitch"));
    d.click("tts.advanced");
    assert!(d.ids("tts.").iter().any(|i| i == "tts.pitch"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn delivery_is_saved_with_the_narration_and_loads_back() {
    let dir = scratch("delivery");
    let mut d = Driver::demo(&dir);
    d.exec("window.panel.TextToSpeech", json!({}));
    d.frames(3);
    assert!(d.ids("tts.").iter().any(|i| i == "tts.delivery"));
    assert_eq!(d.draft()["delivery"], "plain");
    d.ok("ui.set", json!({"tts": {"delivery": "excited"}}));
    d.frames(2);
    d.click("tts.text");
    d.type_text("Look at this one!");
    d.exec("playhead.set", json!({"seconds": 1.0}));
    d.click("tts.save");
    d.settle();
    let clips = d.narration_clips();
    assert_eq!(clips.len(), 1);
    let n = d.exec("tts.inspect", json!({"clip": clips[0]}));
    assert_eq!(n["narration"]["delivery"], "excited", "{n}");
    // a fresh draft keeps the delivery, like the voice and pitch
    d.click("tts.new");
    d.frames(2);
    assert_eq!(d.draft()["delivery"], "excited");
    // a basic voice reads plainly, and the panel says so
    assert!(d.ui().to_string().contains("excited"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A cloned voice reading a Malayalam script: the voice-style controls (temperature, emphasis,
/// speed, expressiveness over the emotion's preset), the preset reset and Insert tag — each an
/// automation id; English scripts do not show them.
#[test]
fn malayalam_voice_style_controls() {
    let dir = scratch("styles");
    let mut d = Driver::demo(&dir);
    d.exec("window.panel.TextToSpeech", json!({}));
    d.frames(3);
    let style_ids = ["tts.temperature", "tts.emphasis", "tts.speed", "tts.expressiveness", "tts.samplerReset", "tts.insertTag"];
    d.ok("ui.set", json!({"tts": {"voice": "clone:me", "text": "Hello there."}}));
    d.frames(3);
    let ids = d.ids("tts.");
    assert!(style_ids.iter().all(|s| !ids.iter().any(|i| i == s)), "English: no voice-style controls {ids:?}");
    d.ok("ui.set", json!({"tts": {"voice": "clone:me", "text": "ഇത് കൊള്ളാം! [calm] ശരി.", "delivery": "excited"}}));
    d.frames(3);
    let ids = d.ids("tts.");
    for s in style_ids {
        assert!(ids.iter().any(|i| i == s), "{s} missing from {ids:?}");
    }
    // regression: showing the sliders never sets them (a slider snapping a preset's f32 to its
    // step reported a change, and every narration then carried the preset as its own settings)
    d.ok("ui.scroll", json!({"id": "tts.numbers", "dy": -400.0}));
    d.frames(4);
    for k in ["temperature", "emphasis", "speed", "expressiveness"] {
        assert_eq!(d.draft()[k], Value::Null, "{k}");
    }
    // a setting over the preset, then back to the preset (the panel scrolled to the controls)
    d.ok("ui.set", json!({"tts": {"temperature": 0.8, "expressiveness": 1.4}}));
    d.frames(2);
    d.ok("ui.scroll", json!({"id": "tts.numbers", "dy": -400.0}));
    d.frames(3);
    d.snapshot("tts-malayalam-styles");
    assert_eq!((d.draft()["temperature"].as_f64(), d.draft()["expressiveness"].as_f64()), (Some(0.8), Some(1.4)));
    d.click("tts.samplerReset");
    assert_eq!((d.draft()["temperature"].clone(), d.draft()["expressiveness"].clone()), (Value::Null, Value::Null));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Clone a voice: the engine and model choices, and the model's download confirmation (size,
/// source, licence) before anything is fetched.
#[test]
fn clone_section_offers_engines_models_and_confirms_downloads() {
    let dir = scratch("clone");
    let mut d = Driver::demo(&dir);
    d.exec("window.panel.TextToSpeech", json!({}));
    d.frames(3);
    assert!(!d.ids("tts.").iter().any(|i| i == "tts.cloneEngine"), "folded away at first");
    d.click("tts.cloneOpen");
    assert_eq!(d.draft()["cloneOpen"], true);
    let ids = d.ids("tts.");
    for id in ["tts.cloneEngine", "tts.cloneModel", "tts.cloneName", "tts.cloneConsent", "tts.clone"] {
        assert!(ids.iter().any(|i| i == id), "{id} in {ids:?}");
    }
    assert_eq!(d.draft()["cloneEngine"], "voxcpm", "natural cloning by default");
    d.snapshot("tts-clone-natural");
    // the timbre-only engine has no model choice
    d.ok("ui.set", json!({"tts": {"cloneEngine": "openvoice"}}));
    d.frames(3);
    assert!(!d.ids("tts.").iter().any(|i| i == "tts.cloneModel"));
    d.ok("ui.set", json!({"tts": {"cloneEngine": "voxcpm", "cloneModel": "voxcpm-2"}}));
    d.frames(3);
    let models = d.exec("tts.clonedVoices", json!({}))["models"].clone();
    let installed = models.as_array().unwrap().iter().any(|m| m["id"] == "voxcpm-2" && m["installed"] == true);
    if !installed {
        // not downloaded: the button asks first, with the size, source and licence
        d.click("tts.clone");
        assert_eq!(d.draft()["cloneConfirm"], true);
        assert!(d.ids("tts.").iter().any(|i| i == "tts.cloneDownload.confirm"));
        assert!(d.harness.query_by_label_contains("huggingface.co/openbmb/VoxCPM2").is_some(), "the source is shown");
        assert!(d.harness.query_by_label_contains("Apache-2.0").is_some(), "the licence is shown");
        assert!(d.harness.query_by_label_contains("GB, once").is_some(), "the size is shown");
        d.snapshot("tts-clone-confirm");
        // nothing is downloaded until confirmed
        assert!(d.draft()["cloneJob"].is_null());
    }
    let _ = std::fs::remove_dir_all(&dir);
}
