//! Headless UI tests of the Showcase panel on the demo project, with AutoCut's showcase host and
//! scripted models: Sequence ▸ Make a Showcase… opens the Analyze form; Analyze finds the clips
//! and writes the plan as a background job whose commands the UI runs between frames; the review
//! (leave out, merge, relabel) is made by clicks and `ui.set`; Write script writes the words for
//! the entries as reviewed; Build makes the sequence; a shot plays in the Source monitor. A job
//! started elsewhere brings the panel forward and Cancel stops it; hostile drafts are refused.

use std::sync::Arc;
use std::sync::mpsc::{Sender, channel};
use std::time::Duration;

use egui_kittest::Harness;
use filmcraft_autocut::host::{Models, ShowcaseHost};
use filmcraft_engine::Session;
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
}

impl Driver {
    fn demo(models: Option<Models>) -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).expect("demo project");
        if let Some(m) = models {
            session.autocut.host = Some(Arc::new(ShowcaseHost::with_models(m)));
        }
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(session).with_control(rx);
        // tall enough that the first entries show without scrolling
        let harness = Harness::builder().with_size(egui::vec2(1600.0, 1500.0)).with_max_steps(10_000).build_eframe(move |_cc| app);
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

    fn err(&mut self, method: &str, params: Value) -> String {
        let v = self.call(method, params.clone());
        assert_eq!(v["ok"], json!(false), "{method} {params} should fail: {v}");
        v["error"].as_str().unwrap_or_default().to_string()
    }

    fn exec(&mut self, command: &str, params: Value) -> Value {
        self.ok("engine.execute", json!({"command": command, "params": params}))
    }

    fn ids(&mut self) -> Vec<String> {
        self.ok("ui.elements", json!({"prefix": "showcase."})).as_array().unwrap().iter().filter_map(|e| e["id"].as_str().map(str::to_string)).collect()
    }

    fn draft(&mut self) -> Value {
        self.ok("ui.inspect", json!({}))["ui"]["showcase"].clone()
    }

    fn status(&mut self) -> String {
        self.ok("ui.inspect", json!({}))["ui"]["status"].as_str().unwrap_or_default().to_string()
    }

    /// Run frames (the UI runs the job's commands) until the latest showcase job finishes; its
    /// `jobs.list` entry.
    fn finish_job(&mut self) -> Value {
        let t0 = std::time::Instant::now();
        loop {
            self.frames(2);
            let job = self.exec("autocut.showcaseInspect", json!({}))["job"].clone();
            if job["finished"] == true {
                self.frames(3);
                return job;
            }
            assert!(t0.elapsed() < Duration::from_secs(120), "the job never finished: {job}");
        }
    }
}

#[test]
fn analyze_review_rewrite_and_build_from_the_panel() {
    let mut d = Driver::demo(Some(Models::scripted()));
    // Sequence ▸ Make a Showcase… opens the panel with its Analyze form
    let menu = d.ok("ui.menu.list", json!({}));
    let item = menu.as_array().unwrap().iter().find(|i| i["id"] == "autocut.makeShowcase").cloned().unwrap();
    assert_eq!((item["path"].clone(), item["label"].clone(), item["enabled"].clone()), (json!(["Sequence"]), json!("Make a Showcase…"), json!(true)));
    assert_eq!(d.ok("ui.menu.invoke", json!({"id": "autocut.makeShowcase"})), json!({"panel": "Showcase"}));
    d.frames(3);
    let ids = d.ids();
    for id in [
        "showcase.analyzeOpen",
        "showcase.subject",
        "showcase.since",
        "showcase.prompt",
        "showcase.album",
        "showcase.analyze",
        "showcase.title",
        "showcase.voice",
        "showcase.mood",
        "showcase.intro",
        "showcase.outro",
        "showcase.exportPath",
        "showcase.upload",
        "showcase.writeScript",
        "showcase.build",
        "showcase.buildExport",
    ] {
        assert!(ids.iter().any(|i| i == id), "{id} in {ids:?}");
    }

    // Analyze: every demo clip is a (scripted) robot, one entry each, a line each
    d.ok("ui.set", json!({"showcase": {"subject": "robot", "title": "Robots", "voice": ""}}));
    d.ok("ui.click", json!({"id": "showcase.analyze"}));
    let job = d.finish_job();
    assert_eq!(job["result"]["error"], Value::Null, "{job}");
    assert_eq!(job["command"], "autocut.makeShowcase");
    let draft = d.draft();
    let entries = draft["entries"].as_array().cloned().unwrap();
    assert_eq!(entries.len(), 6, "{draft}");
    assert_eq!(draft["title"], "Robots");
    assert!(entries.iter().all(|e| e["include"] == true && e["line"].as_str().is_some_and(|l| !l.is_empty())), "{draft}");
    assert_eq!(draft["intro"], "This is every robot I have met, in order.");
    assert!(d.status().starts_with("Showcase: 6 entries planned"), "{}", d.status());
    let ids = d.ids();
    for id in [
        "showcase.entry.0.include",
        "showcase.entry.0.thumb",
        "showcase.entry.0.label",
        "showcase.entry.0.description",
        "showcase.entry.0.line",
        "showcase.entry.0.merge",
    ] {
        assert!(ids.iter().any(|i| i == id), "{id} in {ids:?}");
    }
    // screenshots of the panel (kept in FILMCRAFT_UI_SNAPSHOT_DIR when set)
    let keep = std::env::var_os("FILMCRAFT_UI_SNAPSHOT_DIR").map(std::path::PathBuf::from);
    let dir = keep.clone().unwrap_or_else(std::env::temp_dir);
    std::fs::create_dir_all(&dir).unwrap();
    let shot = |name: &str| dir.join(format!("showcase-{name}-{}.png", std::process::id()));
    d.ok("ui.screenshot", json!({"path": shot("planned").to_string_lossy(), "panel": "Showcase"}));
    assert!(std::fs::metadata(shot("planned")).is_ok_and(|m| m.len() > 1000), "a screenshot of the panel");

    // the review: a click leaves the first out; the fourth is the third again; the second is named
    d.ok("ui.click", json!({"id": "showcase.entry.0.include"}));
    d.frames(2);
    assert_eq!(d.draft()["entries"][0]["include"], false);
    d.ok("ui.set", json!({"showcase": {"entries": {"3": {"mergePrevious": true}, "1": {"label": "Ruby the red robot"}}}}));
    d.ok("ui.click", json!({"id": "showcase.writeScript"}));
    let job = d.finish_job();
    assert_eq!((job["command"].as_str(), job["result"]["error"].clone()), (Some("autocut.showcasePlan"), Value::Null), "{job}");
    let draft = d.draft();
    let lines: Vec<bool> = draft["entries"].as_array().unwrap().iter().map(|e| e["line"].as_str().is_some_and(|l| !l.is_empty())).collect();
    assert_eq!(lines, [false, true, true, false, true, true], "words for the entries shown: {draft}");
    assert_eq!(draft["entries"][1]["label"], "Ruby the red robot");
    assert_eq!((draft["entries"][0]["include"].clone(), draft["entries"][3]["mergePrevious"].clone()), (json!(false), json!(true)));

    // Build (small): four entries in a new sequence, the edits ordinary commands
    d.ok("ui.set", json!({"showcase": {"extra": {"width": 320, "height": 180}}}));
    let undo = d.exec("history.list", json!({}))["undo"].as_array().map_or(0, Vec::len);
    d.ok("ui.click", json!({"id": "showcase.build"}));
    let job = d.finish_job();
    assert_eq!((job["command"].as_str(), job["result"]["error"].clone()), (Some("autocut.showcaseBuild"), Value::Null), "{job}");
    assert_eq!(job["showcase"]["built"]["entries"], 4, "{job}");
    assert!(d.status().starts_with("Showcase: Built 4 entries"), "{}", d.status());
    let seq = d.exec("sequence.inspect", json!({}));
    assert_eq!(seq["settings"]["width"], 320, "the new sequence is open");
    assert!(d.exec("history.list", json!({}))["undo"].as_array().map_or(0, Vec::len) > undo, "undoable");
    d.ok("ui.screenshot", json!({"path": shot("built").to_string_lossy(), "panel": "Showcase"}));

    // a shot plays in the Source monitor
    d.ok("ui.set", json!({"showcase": {"entries": {"1": {"open": true}}}}));
    d.frames(2);
    let item = d.draft()["entries"][1]["entry"]["shots"][0]["item"].as_u64().unwrap();
    d.ok("ui.click", json!({"id": "showcase.entry.1.shot.0"}));
    d.frames(2);
    assert_eq!(d.exec("state.inspect", json!({}))["source_item"], item, "the shot's clip is in the Source monitor");
    d.ok("ui.playback", json!({"action": "stop"}));
    if keep.is_none() {
        for name in ["planned", "built"] {
            let _ = std::fs::remove_file(shot(name));
        }
    }
}

/// A job started elsewhere (an agent, the AutoCut prompt) brings the panel forward with its
/// progress, and Cancel stops it.
#[test]
fn a_job_started_elsewhere_shows_the_panel_and_cancel_stops_it() {
    let slow = filmcraft_autocut::llm::FnModel::new("slow", |_, _, _| {
        std::thread::sleep(Duration::from_millis(40));
        Ok(json!({"present": true, "bbox_2d": [10, 10, 300, 300]}))
    });
    let models = Models { analysis: Arc::new(slow), script: Models::scripted().script };
    let mut d = Driver::demo(Some(models));
    assert!(!d.ids().iter().any(|i| i == "showcase.title"), "the panel is not open yet");
    d.exec("autocut.makeShowcase", json!({"prompt": "make a video of every robot"}));
    d.frames(4);
    let ids = d.ids();
    assert!(ids.iter().any(|i| i == "showcase.cancel"), "the panel shows the running job: {ids:?}");
    if let Some(dir) = std::env::var_os("FILMCRAFT_UI_SNAPSHOT_DIR").map(std::path::PathBuf::from) {
        std::fs::create_dir_all(&dir).unwrap();
        d.frames(10);
        d.ok("ui.screenshot", json!({"path": dir.join("showcase-running.png").to_string_lossy(), "panel": "Showcase"}));
    }
    d.ok("ui.click", json!({"id": "showcase.cancel"}));
    let job = d.finish_job();
    assert_eq!(job["result"]["error"], "cancelled", "{job}");
    assert_eq!(d.status(), "Showcase: cancelled");
    assert_eq!(d.draft()["entries"], json!([]), "nothing was planned");
}

/// Without a host the menu item is disabled and the panel says why; hostile drafts are refused
/// or fail in the engine, never in the UI.
#[test]
fn hostile_drafts_and_no_host() {
    let mut d = Driver::demo(None);
    let menu = d.ok("ui.menu.list", json!({}));
    assert_eq!(menu.as_array().unwrap().iter().find(|i| i["id"] == "autocut.makeShowcase").map(|i| i["enabled"].clone()), Some(json!(false)));
    assert!(d.err("ui.menu.invoke", json!({"id": "autocut.makeShowcase"})).contains("not available"));
    d.ok("ui.panel.show", json!({"panel": "Showcase"}));
    d.frames(3);
    for bad in [
        json!({"showcase": "robots"}),
        json!({"showcase": {"entries": {"0": {"include": false}}}}),
        json!({"showcase": {"entries": {"first": {}}}}),
        json!({"showcase": {"entries": [{"label": 3}]}}),
        json!({"showcase": {"title": ["Robots"]}}),
    ] {
        d.err("ui.set", bad);
    }
    // a draft whose entry is no object: the panel draws it, the engine refuses to build it
    let mut d = Driver::demo(Some(Models::scripted()));
    d.ok("ui.panel.show", json!({"panel": "Showcase"}));
    d.ok("ui.set", json!({"showcase": {"entries": [{"include": true, "label": "x", "entry": 5}, {"include": true, "entry": {"shots": "none"}}]}}));
    d.frames(3);
    d.ok("ui.click", json!({"id": "showcase.build"}));
    d.frames(3);
    assert!(d.status().contains("each entry must be an object"), "{}", d.status());
    d.ok("ui.set", json!({"showcase": {"entries": [{"include": true, "label": "x", "entry": {"shots": [{"item": 99999, "best": 1e300}]}}]}}));
    d.frames(3);
    d.ok("ui.click", json!({"id": "showcase.build"}));
    let job = d.finish_job();
    assert!(job["result"]["error"].as_str().is_some_and(|e| e.contains("not a showcase plan")), "{job}");
}

/// The look and the channel outro from the panel: its controls (automation ids), filled by clicks
/// and `ui.set`, go into the plan the build takes, and the show ends with the branded outro.
#[test]
fn the_look_and_a_channel_outro_from_the_panel() {
    let mut d = Driver::demo(Some(Models::scripted()));
    d.ok("ui.menu.invoke", json!({"id": "autocut.makeShowcase"}));
    d.frames(3);
    d.ok("ui.set", json!({"showcase": {"subject": "robot", "title": "Robots", "voice": ""}}));
    d.ok("ui.click", json!({"id": "showcase.analyze"}));
    let job = d.finish_job();
    assert_eq!(job["result"]["error"], Value::Null, "{job}");
    // the section opens with a click; the outro's fields show once it is on
    d.ok("ui.set", json!({"showcase": {"extra": {"width": 320, "height": 180}}}));
    d.ok("ui.click", json!({"id": "showcase.lookOpen"}));
    d.frames(3);
    let ids = d.ids();
    for id in ["showcase.style", "showcase.timeline", "showcase.loudness", "showcase.embedCaptions", "showcase.outroOn"] {
        assert!(ids.iter().any(|i| i == id), "{id} in {ids:?}");
    }
    d.ok("ui.click", json!({"id": "showcase.outroOn"}));
    d.frames(3);
    assert_eq!(d.draft()["look"]["outro"]["on"], true);
    let ids = d.ids();
    for id in [
        "showcase.outroLogo",
        "showcase.outroLogoBrowse",
        "showcase.outroChannel",
        "showcase.outroHandle",
        "showcase.outroTagline",
        "showcase.outroCta",
        "showcase.outroCtaDone",
        "showcase.outroTerminal",
        "showcase.outroSeconds",
        "showcase.outroBackground",
    ] {
        assert!(ids.iter().any(|i| i == id), "{id} in {ids:?}");
    }
    let outro = json!({"channel": "H4hacks", "handle": "@H4hacks", "cta": "Subscribe", "ctaDone": "Subscribed", "terminal": "> h4hacks.com", "background": "7:1:9", "seconds": 9.0});
    d.ok("ui.set", json!({"showcase": {"look": {"style": "tech", "outro": outro}}}));
    d.frames(3);
    let keep = std::env::var_os("FILMCRAFT_UI_SNAPSHOT_DIR").map(std::path::PathBuf::from);
    let dir = keep.unwrap_or_else(std::env::temp_dir);
    std::fs::create_dir_all(&dir).unwrap();
    let shot = dir.join(format!("showcase-outro-{}.png", std::process::id()));
    d.ok("ui.screenshot", json!({"path": shot.to_string_lossy(), "panel": "Showcase"}));
    assert!(std::fs::metadata(&shot).is_ok_and(|m| m.len() > 1000), "a screenshot of the panel");
    // Build: the plan carries the look and the outro; the show ends with it
    d.ok("ui.click", json!({"id": "showcase.build"}));
    let job = d.finish_job();
    assert_eq!((job["command"].as_str(), job["result"]["error"].clone()), (Some("autocut.showcaseBuild"), Value::Null), "{job}");
    let built = &job["showcase"]["built"];
    assert_eq!(built["theme"]["name"], "tech", "{built}");
    assert_eq!(built["outro"]["background"], "footage", "{built}");
    assert!(built["outro"]["elements"].as_array().is_some_and(|e| e.iter().any(|x| x["name"] == "lockup")), "{built}");
}
