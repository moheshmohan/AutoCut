//! The Program monitor's Comment tool (PX8), driven headless through the control channel: turn it
//! on from the transport, drag a box on the picture, type the comment and press Enter; the comment
//! is in the review inbox beside the saved project, marked on the scrub bar and drawn on its
//! frame; Previous goes back to it; Escape closes the tool. A wait for comments over the control
//! channel runs as a job, so the app keeps answering.
//!
//! With `FILMCRAFT_UI_SHOTS=<dir>` the test also renders the UI with wgpu and writes PNGs there.

use std::sync::mpsc::{Sender, channel};

use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
    shots: Option<std::path::PathBuf>,
}

/// A folder of its own for one test.
fn folder(name: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir().join(format!("filmcraft-review-ui-{name}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

impl Driver {
    /// The demo project, saved (by path) in `dir` when given.
    fn demo(dir: Option<&std::path::Path>) -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).expect("demo project");
        session.path = dir.map(|d| d.join("demo.fcproj").display().to_string());
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(session).with_control(rx);
        let shots = std::env::var_os("FILMCRAFT_UI_SHOTS").map(std::path::PathBuf::from);
        let mut b = Harness::builder().with_size(egui::vec2(1600.0, 980.0)).with_max_steps(10_000);
        if shots.is_some() {
            b = b.wgpu().with_pixels_per_point(1.0);
        }
        let harness = b.build_eframe(move |_cc| app);
        let mut d = Driver { harness, tx, shots };
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
        let r = self.call(method, params.clone());
        assert_eq!(r["ok"], true, "{method} {params}: {r}");
        r["result"].clone()
    }
    fn exec(&mut self, command: &str, params: Value) -> Value {
        self.ok("engine.execute", json!({"command": command, "params": params}))
    }
    fn element(&mut self, id: &str) -> Option<Value> {
        let v = self.ok("ui.elements", json!({"prefix": id}));
        v.as_array().and_then(|a| a.iter().find(|e| e["id"] == id).cloned())
    }
    fn rect(&mut self, id: &str) -> [f64; 4] {
        let e = self.element(id).unwrap_or_else(|| panic!("no element {id}"));
        let r: Vec<f64> = e["rect"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        [r[0], r[1], r[2], r[3]]
    }
    fn has(&mut self, id: &str) -> bool {
        self.element(id).is_some()
    }
    fn click(&mut self, id: &str) {
        self.ok("ui.click", json!({"id": id}));
        self.frames(3);
    }
    fn app(&mut self) -> &mut FilmcraftApp {
        self.harness.state_mut()
    }
    fn shot(&mut self, name: &str) {
        let Some(dir) = self.shots.clone() else { return };
        for _ in 0..40 {
            self.frames(1);
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        let img = self.harness.render().expect("wgpu render");
        std::fs::create_dir_all(&dir).unwrap();
        img.save(dir.join(format!("review-tool-{name}.png"))).unwrap();
    }
}

#[test]
fn the_comment_tool_pins_a_comment_to_the_frame() {
    let dir = folder("pin");
    let mut d = Driver::demo(Some(&dir));
    let seq = d.exec("sequence.inspect", json!({}));
    let clip = &seq["video"][0]["items"][0];
    let (id, start, dur) = (clip["clip"].as_u64().unwrap(), clip["start"].as_i64().unwrap(), clip["duration"].as_i64().unwrap());
    let at = d.exec("playhead.set", json!({"time": start + dur / 3}));
    let _ = at;
    let ph = d.app().session.playhead().0;
    d.frames(3);
    assert!(d.has("program.transport.review.tool"), "the transport has the button");
    assert!(!d.has("program.review.picture"));
    d.click("program.transport.review.tool");
    assert!(d.app().ui.review_tool.is_some(), "the tool is on");
    assert!(d.has("program.review.picture") && d.has("program.review.text") && d.has("program.review.done"));
    assert_eq!(d.element("program.review.save").unwrap()["enabled"], false, "nothing to save yet");
    // drag a box in the middle of the picture
    let pic = d.rect("program.review.picture");
    let (x0, y0) = (pic[0] + pic[2] * 0.25, pic[1] + pic[3] * 0.3);
    let (x1, y1) = (pic[0] + pic[2] * 0.5, pic[1] + pic[3] * 0.6);
    d.ok("ui.drag", json!({"from": {"x": x0, "y": y0}, "to": {"x": x1, "y": y1}, "steps": 8}));
    d.frames(3);
    assert!(d.has("program.review.box"), "the box is drawn");
    let st = d.app().ui.review_tool.clone().unwrap();
    let (w, h) = {
        let q = d.app().session.active_sequence().unwrap();
        (f64::from(q.settings.width), f64::from(q.settings.height))
    };
    let b = st.area.expect("a box");
    assert!((b[0] / w - 0.25).abs() < 0.02 && (b[2] / w - 0.25).abs() < 0.02 && (b[3] / h - 0.3).abs() < 0.02, "{b:?}");
    assert_eq!((st.clip, st.time), (Some(id), Some(ph)), "the clip under the box, at the playhead");
    // the text field takes the keyboard after the box: type and press Enter
    d.ok("ui.type", json!({"text": "Logo is cut off here"}));
    d.frames(2);
    assert_eq!(d.app().ui.review_tool.as_ref().unwrap().text, "Logo is cut off here");
    assert_eq!(d.element("program.review.save").unwrap()["enabled"], true);
    d.shot("box");
    d.ok("ui.key", json!({"key": "Enter"}));
    d.frames(4);
    let inbox = d.exec("review.inbox", json!({}));
    let c = &inbox["comments"][0];
    assert_eq!((c["id"].as_str(), c["author"].as_str(), c["text"].as_str()), (Some("c1"), Some("user"), Some("Logo is cut off here")), "{inbox:#}");
    assert_eq!((c["clip"].as_u64(), c["time"].as_i64()), (Some(id), Some(ph)), "{inbox:#}");
    assert!(dir.join("build").join("review").join("inbox.jsonl").is_file());
    assert!(d.app().ui.status.contains("c1"), "{}", d.app().ui.status);
    let st = d.app().ui.review_tool.clone().unwrap();
    assert!(st.area.is_none() && st.text.is_empty(), "the tool is ready for the next one: {st:?}");
    // drawn on its frame and marked on the scrub bar
    d.frames(2);
    assert!(d.has("program.review.comment.c1"), "the comment on its frame");
    assert!(d.has("program.review.mark.c1"), "its mark on the scrub bar");
    d.shot("saved");
    // Previous goes back to it from later on
    d.exec("playhead.set", json!({"time": start + dur / 2}));
    d.frames(3);
    assert!(!d.has("program.review.comment.c1"), "only on its own frame");
    d.click("program.review.prev");
    assert_eq!(d.app().session.playhead().0, ph);
    // a click is a point; Clear drops it; Escape with nothing drawn closes the tool
    let pic = d.rect("program.review.picture");
    d.ok("ui.click", json!({"x": pic[0] + pic[2] * 0.7, "y": pic[1] + pic[3] * 0.7}));
    d.frames(3);
    let b = d.app().ui.review_tool.as_ref().and_then(|t| t.area).expect("a point");
    assert_eq!((b[2], b[3]), (0.0, 0.0), "{b:?}");
    d.click("program.review.clear");
    assert!(d.app().ui.review_tool.as_ref().is_some_and(|t| t.area.is_none()));
    d.ok("ui.key", json!({"key": "Escape"}));
    d.frames(3);
    assert!(d.app().ui.review_tool.is_none(), "Escape closed the tool");
    // the mark stays with the tool off
    assert!(d.has("program.review.mark.c1"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_unsaved_project_cannot_take_comments() {
    let mut d = Driver::demo(None);
    d.click("program.transport.review.tool");
    assert!(d.app().ui.review_tool.is_none());
    assert!(d.app().ui.status.contains("Save the project"), "{}", d.app().ui.status);
}

#[test]
fn a_wait_over_the_control_channel_is_a_job() {
    let dir = folder("wait");
    let mut d = Driver::demo(Some(&dir));
    // without the job the app would stop answering for the whole timeout
    let r = d.exec("review.inbox", json!({"wait": true, "timeout": 30}));
    let job = r["job"].as_u64().unwrap_or_else(|| panic!("a job: {r}"));
    d.exec("review.comment", json!({"seconds": 0.5, "text": "from an agent", "author": "test"}));
    let mut done = Value::Null;
    for _ in 0..400 {
        let jobs = d.exec("jobs.list", json!({}));
        if let Some(j) = jobs.as_array().unwrap().iter().find(|j| j["id"].as_u64() == Some(job) && j["finished"] == true) {
            done = j.clone();
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert_eq!(done["review"]["new"], json!(["c1"]), "{done:#}");
    let _ = std::fs::remove_dir_all(&dir);
}
