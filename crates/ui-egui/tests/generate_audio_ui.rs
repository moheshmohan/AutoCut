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

impl Driver {
    fn draft(&mut self) -> Value {
        self.ok("ui.inspect", json!({}))["ui"]["generate_audio"].clone()
    }

    fn status(&mut self) -> String {
        self.ok("ui.inspect", json!({}))["ui"]["status"].as_str().unwrap_or_default().to_string()
    }

    fn ids(&mut self) -> Vec<String> {
        self.ok("ui.elements", json!({"prefix": "generateAudio."})).as_array().unwrap().iter().filter_map(|e| e["id"].as_str().map(str::to_string)).collect()
    }

    /// Scroll the panel until `id` is in view (the music section is taller than the panel).
    fn reveal(&mut self, id: &str) {
        let tall = |e: &Value| e["rect"][2].as_f64().unwrap_or(0.0) >= 6.0 && e["rect"][3].as_f64().unwrap_or(0.0) >= 6.0;
        for _ in 0..60 {
            let els = self.ok("ui.elements", json!({"prefix": "generateAudio."}));
            let all = els.as_array().cloned().unwrap_or_default();
            let target = all.iter().find(|e| e["id"] == id).cloned();
            if target.as_ref().is_some_and(tall) {
                return;
            }
            let anchor = all.iter().find(|e| tall(e) && e["id"] != "generateAudio.add").cloned().expect("something of the panel in view");
            let up = target.is_some_and(|t| t["rect"][1].as_f64() < anchor["rect"][1].as_f64());
            self.ok("ui.scroll", json!({"id": anchor["id"], "dy": if up { 120.0 } else { -120.0 }}));
            self.frames(2);
        }
        panic!("{id} never came into view");
    }

    fn click(&mut self, id: &str) {
        self.reveal(id);
        self.ok("ui.click", json!({"id": id}));
    }

    /// Frames until the panel's preview job is done.
    fn until_previewed(&mut self) {
        for _ in 0..600 {
            self.frames(2);
            if self.draft()["previewJob"].is_null() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("the preview never finished");
    }
}

/// Find music by feel through the control channel: the browser's widgets, a user's SoundFont
/// chosen in the panel, stems muted, Preview (rendered in the background), Regenerate, Add at
/// playhead placing what was heard, and the GeneralUser GS consent box (never confirmed here).
#[test]
fn find_music_by_feel_previews_and_places() {
    let scratch = std::env::temp_dir().join(format!("filmcraft-genmusic-ui-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let font = scratch.join("tiny.sf2");
    std::fs::write(&font, filmcraft_soundfont::synthetic::tiny()).unwrap();
    let mut d = Driver::demo(&scratch.to_string_lossy());
    d.ok("ui.panel.show", json!({"panel": "Generate Audio"}));
    d.ok("ui.set", json!({"generateAudio": {"kind": "music", "mood": "calm", "genre": "piano", "seconds": 4.0, "seed": 9}}));
    d.frames(4);
    let ids = d.ids();
    for id in [
        "generateAudio.mood",
        "generateAudio.genre",
        "generateAudio.style.1",
        "generateAudio.intensity",
        "generateAudio.length",
        "generateAudio.tempo",
        "generateAudio.key",
        "generateAudio.preview",
        "generateAudio.regenerate",
        "generateAudio.stem.drums.mute",
        "generateAudio.stem.pad.level",
        "generateAudio.stems",
        "generateAudio.instruments",
        "generateAudio.soundfont.path",
        "generateAudio.soundfont.use",
        "generateAudio.add",
    ] {
        assert!(ids.iter().any(|i| i == id), "{id} in {ids:?}");
    }
    let gugs = d.exec("music.soundfonts", json!({}))["soundfonts"][0]["installed"] == json!(true);
    // screenshots of the panel (kept in FILMCRAFT_UI_SNAPSHOT_DIR when set)
    let keep = std::env::var_os("FILMCRAFT_UI_SNAPSHOT_DIR").map(std::path::PathBuf::from);
    let shot = |d: &mut Driver, name: &str| {
        if let Some(dir) = &keep {
            std::fs::create_dir_all(dir).unwrap();
            d.ok("ui.screenshot", json!({"path": dir.join(format!("music-by-feel-{name}.png")).to_string_lossy(), "panel": "Generate Audio"}));
        }
    };
    shot(&mut d, "top");

    // the user's own SoundFont, chosen in the panel
    d.ok("ui.set", json!({"generateAudio": {"soundfontPath": font.to_string_lossy()}}));
    d.frames(2);
    d.click("generateAudio.soundfont.use");
    d.frames(3);
    assert!(d.status().contains("FilmCraft Test Font"), "{}", d.status());
    assert_eq!(d.exec("music.soundfonts", json!({}))["active"]["name"], "FilmCraft Test Font");

    // drums off, then hear it
    d.click("generateAudio.stem.drums.mute");
    d.frames(2);
    assert_eq!(d.draft()["muted"][0], true);
    d.click("generateAudio.preview");
    d.frames(1);
    d.until_previewed();
    let status = d.status();
    assert!(status.contains("Calm Piano") && status.contains("seed 9") && status.contains("FilmCraft Test Font"), "{status}");
    shot(&mut d, "stems");
    // the same request is already rendered: the engine answers without a job
    let draft = d.draft();
    let again = d.exec(
        "music.preview",
        json!({"mood": "calm", "genre": "piano", "duration": 4.0, "energy": draft["intensity"], "key": "A", "bpm": draft["bpm"], "mute": ["drums"], "seed": 9}),
    );
    assert_eq!((again["job"].clone(), again["ready"].clone()), (json!(null), json!(true)), "{again}");

    // Add at playhead places what was heard
    d.exec("playhead.set", json!({"seconds": 1.0}));
    let before = d.audio_clips().len();
    d.ok("ui.click", json!({"id": "generateAudio.add"}));
    d.frames(3);
    let clips = d.audio_clips();
    assert_eq!(clips.len(), before + 1, "{}", d.status());
    let music = clips.iter().find(|c| c["name"].as_str().is_some_and(|n| n.starts_with("Music"))).expect("a Music clip");
    let dur = music["duration"].as_i64().unwrap() as f64 / 254_016_000_000.0;
    assert!((dur - 4.0).abs() < 0.05, "{dur}");
    assert!(d.status().starts_with("Added Music"), "{}", d.status());

    // Regenerate: another variation, heard at once
    d.click("generateAudio.regenerate");
    d.frames(1);
    assert_ne!(d.draft()["seed"], json!(9));
    d.until_previewed();
    assert!(d.status().contains("Calm Piano"), "{}", d.status());

    // back to the oscillators from the Instruments menu's choice (as an agent would)
    d.exec("music.useSoundfont", json!({"soundfont": "oscillators"}));
    d.ok("ui.set", json!({"generateAudio": {"seed": 3}}));
    d.frames(2);
    d.click("generateAudio.preview");
    d.frames(1);
    d.until_previewed();
    assert!(d.status().contains("oscillators"), "{}", d.status());

    // GeneralUser GS: the consent box shows the licence; cancelled, nothing downloads
    if !gugs {
        let jobs = d.exec("jobs.list", json!({})).as_array().map_or(0, Vec::len);
        d.click("generateAudio.soundfont.get");
        d.frames(3);
        assert_eq!(d.draft()["confirmSoundfont"], true);
        let ids = d.ids();
        assert!(ids.iter().any(|i| i == "generateAudio.soundfont.confirm"), "{ids:?}");
        d.reveal("generateAudio.soundfont.confirm");
        shot(&mut d, "consent");
        d.click("generateAudio.soundfont.cancel");
        d.frames(2);
        assert_eq!(d.draft()["confirmSoundfont"], false);
        assert_eq!(d.exec("jobs.list", json!({})).as_array().map_or(0, Vec::len), jobs, "no download started");
    }
    let _ = std::fs::remove_dir_all(&scratch);
}
