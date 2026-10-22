//! The Program monitor's Track tool, driven headless through the control channel: open it from
//! the transport, draw a box on the picture, pick the method, track both ways (a background job),
//! see the path and the object's box, attach a text callout and a blur from the Attach menu, and
//! close it.
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

impl Driver {
    fn demo() -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).expect("demo project");
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
    fn rect(&mut self, id: &str) -> [f64; 4] {
        let v = self.ok("ui.elements", json!({"prefix": id}));
        let e = v.as_array().unwrap().iter().find(|e| e["id"] == id).unwrap_or_else(|| panic!("no element {id}: {v}")).clone();
        let r: Vec<f64> = e["rect"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        [r[0], r[1], r[2], r[3]]
    }
    fn has(&mut self, id: &str) -> bool {
        let v = self.ok("ui.elements", json!({"prefix": id}));
        v.as_array().is_some_and(|a| a.iter().any(|e| e["id"] == id))
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
        img.save(dir.join(format!("track-tool-{name}.png"))).unwrap();
    }
}

#[test]
fn the_track_tool_draws_tracks_and_attaches() {
    let mut d = Driver::demo();
    let seq = d.exec("sequence.inspect", json!({}));
    let clip = &seq["video"][0]["items"][0];
    let (id, start, dur) = (clip["clip"].as_u64().unwrap(), clip["start"].as_i64().unwrap(), clip["duration"].as_i64().unwrap());
    d.exec("playhead.set", json!({"time": start + dur / 3}));
    d.exec("timeline.select", json!({"clips": [id]}));
    d.frames(3);
    // nothing tracked yet; the transport button opens the tool on the selected clip
    assert!(!d.has("program.track.picture"));
    d.click("program.transport.track.tool");
    assert!(d.app().ui.track_tool.as_ref().is_some_and(|t| t.clip == id), "the tool is on the clip");
    assert!(d.has("program.track.picture") && d.has("program.track.done"));
    // the track buttons wait for a box
    d.click("program.track.forward");
    assert!(d.app().session.project.motion_tracks.is_empty());
    // draw the box in the middle of the picture
    let pic = d.rect("program.track.picture");
    let (x0, y0) = (pic[0] + pic[2] * 0.4, pic[1] + pic[3] * 0.4);
    let (x1, y1) = (pic[0] + pic[2] * 0.6, pic[1] + pic[3] * 0.62);
    d.ok("ui.drag", json!({"from": {"x": x0, "y": y0}, "to": {"x": x1, "y": y1}, "steps": 8}));
    d.frames(2);
    assert!(d.has("program.track.box"), "the box is drawn");
    let r = d.app().ui.track_tool.as_ref().and_then(|t| t.rect).expect("a box");
    assert!((r[0] - 0.4).abs() < 0.03 && (r[2] - 0.2).abs() < 0.03, "{r:?}");
    d.click("program.track.method.points");
    assert_eq!(d.app().ui.track_tool.as_ref().unwrap().method, "points");
    d.shot("box");
    d.click("program.track.both");
    // the job runs in the background; its progress shows on the bar and the clip
    let mut tracked = false;
    for _ in 0..600 {
        d.frames(1);
        if !d.app().session.project.motion_tracks.is_empty() && d.app().ui.track_tool.as_ref().is_some_and(|t| t.job.is_none()) {
            tracked = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(tracked, "the track was made");
    d.frames(3);
    let t = d.app().session.project.motion_tracks.values().next().unwrap().clone();
    assert_eq!(t.clip.0, id);
    assert_eq!(t.method, filmcraft_project::TrackingMethod::Points);
    assert_eq!(d.app().ui.track_tool.as_ref().unwrap().track, Some(t.id.0), "the new track is shown");
    assert!(d.has("program.track.path") && d.has("program.track.object"), "the path and the object are drawn");
    assert!(d.has("program.track.select"));
    d.shot("path");
    // Attach ▸ Text callout and Attach ▸ Blur
    if std::env::var_os("TRACK_UI_DEBUG").is_some() {
        let v = d.ok("ui.elements", json!({"prefix": "program."}));
        for e in v.as_array().unwrap() {
            eprintln!("{} {}", e["id"], e["rect"]);
        }
    }
    d.click("program.track.attach");
    assert!(d.has("program.track.attach.text") && d.has("program.track.attach.blur") && d.has("program.track.attach.sticker"));
    d.click("program.track.attach.text");
    let t = d.app().session.project.motion_tracks.get(&t.id).unwrap().clone();
    assert_eq!(t.attachments.len(), 1, "a callout follows the track");
    d.click("program.track.attach");
    d.click("program.track.attach.blur");
    let t = d.app().session.project.motion_tracks.get(&t.id).unwrap().clone();
    assert_eq!(t.attachments.len(), 2);
    let q = d.app().session.active_sequence().unwrap().clone();
    assert!(q.find_item(filmcraft_project::ClipId(id)).unwrap().1.effects.iter().any(|e| e.effect == "gaussian_blur" && !e.masks.is_empty()));
    d.shot("attached");
    // undo takes the blur back; Done closes the tool
    d.exec("edit.undo", json!({}));
    assert_eq!(d.app().session.project.motion_tracks.get(&t.id).unwrap().attachments.len(), 1);
    d.click("program.track.done");
    assert!(d.app().ui.track_tool.is_none());
    assert!(!d.has("program.track.picture"));
    // the transport button opens it again, on the clip's track
    d.click("program.transport.track.tool");
    assert_eq!(d.app().ui.track_tool.as_ref().unwrap().track, Some(t.id.0));
    // Escape closes it
    d.ok("ui.key", json!({"key": "Escape"}));
    d.frames(3);
    assert!(d.app().ui.track_tool.is_none());
}
