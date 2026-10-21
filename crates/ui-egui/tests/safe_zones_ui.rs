//! Platform Safe Zones on the monitors, driven headless through the control channel: the wrench
//! menu's Platform Safe Zones submenu (automation ids `program.settings.safeZones.<platform>`), the
//! View menu, the engine command `view.safeZones` (in the app and straight on the session, as an
//! agent or a recipe runs it), and `ui.set {"program": {"safe_zones": …}}`.
//!
//! With `FILMCRAFT_UI_SHOTS=<dir>` the test also renders the UI with wgpu and writes
//! `safe-zones-*.png`: a vertical copy of the demo sequence (its blurred fill where the picture is
//! fitted) under the zones.

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
    /// The demo project, its first seconds reframed to 9:16 with the blurred fill.
    fn vertical() -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).expect("demo project");
        let r = session
            .execute("sequence.autoReframe", json!({"fallback": "blur", "range": {"startSeconds": 0.0, "endSeconds": 3.0}, "analysisFps": 2, "wait": true}))
            .expect("reframed");
        assert!(r["reframe"]["sequence"].as_u64().is_some(), "{r}");
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
    fn view(&mut self, monitor: &str) -> Value {
        self.ok("ui.inspect", json!({}))["ui"][monitor].clone()
    }
    fn has(&mut self, id: &str) -> bool {
        let v = self.ok("ui.elements", json!({"prefix": id}));
        v.as_array().unwrap().iter().any(|e| e["id"] == id)
    }
    fn click(&mut self, id: &str) {
        self.ok("ui.click", json!({"id": id}));
        self.frames(2);
    }
    fn shot(&mut self, name: &str) {
        let Some(dir) = self.shots.clone() else { return };
        // give the frame workers time to deliver full-resolution frames
        for _ in 0..8 {
            self.frames(3);
            std::thread::sleep(std::time::Duration::from_millis(150));
        }
        let img = self.harness.render().expect("wgpu render");
        std::fs::create_dir_all(&dir).unwrap();
        img.save(dir.join(format!("safe-zones-{name}.png"))).unwrap();
    }
}

#[test]
fn safe_zones_toggle_from_the_wrench_menu_the_view_menu_the_engine_and_ui_set() {
    let mut d = Driver::vertical();
    assert_eq!(d.view("program")["safe_zones"], Value::Null, "off by default");
    assert!(!d.has("program.safeZones"));
    d.shot("off");

    // the wrench menu: Platform Safe Zones ▸ TikTok
    d.click("program.settings");
    assert!(d.has("program.settings.safeZones"), "the wrench menu lists Platform Safe Zones");
    d.click("program.settings.safeZones");
    assert!(d.has("program.settings.safeZones.youtube") && d.has("program.settings.safeZones.off"), "its platforms");
    d.click("program.settings.safeZones.tiktok");
    assert_eq!(d.view("program")["safe_zones"], "tiktok");
    d.frames(2);
    assert!(d.has("program.safeZones"), "the zones are drawn over the picture");
    // the engine's state follows the Program monitor
    assert_eq!(d.exec("view.safeZones", json!({}))["platform"], "tiktok");
    d.shot("tiktok");

    // the engine command, as an agent sends it through the app
    let r = d.exec("view.safeZones", json!({"platform": "youtube"}));
    assert_eq!((r["platform"].as_str(), r["monitor"].as_str(), r["frame"].clone()), (Some("youtube"), Some("program"), json!([1080, 1920])), "{r}");
    assert_eq!(d.view("program")["safe_zones"], "youtube");
    d.shot("youtube");
    // … and straight on the session (a recipe, AutoCut): shown on the next frame
    d.harness.state_mut().session.execute("view.safeZones", json!({"platform": "vertical"})).unwrap();
    d.frames(2);
    assert_eq!(d.view("program")["safe_zones"], "vertical");
    // ui.set reaches the view state of either monitor
    d.ok("ui.set", json!({"program": {"safe_zones": "reels"}, "source": {"safe_zones": "tiktok"}}));
    assert_eq!((d.view("program")["safe_zones"].clone(), d.view("source")["safe_zones"].clone()), (json!("reels"), json!("tiktok")));
    // the View menu: checked items, Off
    let menu = d.ok("ui.menu.list", json!({}));
    let item = |id: &str| menu.as_array().unwrap().iter().find(|m| m["id"] == id).cloned().unwrap_or_else(|| panic!("no menu item {id}"));
    assert_eq!(item("view.safeZones.reels")["path"], json!(["View", "Platform Safe Zones"]));
    assert_eq!(item("view.safeZones.reels")["checked"], true);
    d.ok("ui.menu.invoke", json!({"id": "view.safeZones.off"}));
    d.frames(2);
    assert_eq!(d.view("program")["safe_zones"], Value::Null);
    assert!(!d.has("program.safeZones"));
    assert_eq!(d.harness.state().session.state.safe_zones, None, "the engine's is off too");

    // hostile values are refused and change nothing
    for p in [json!({"platform": 7}), json!({"platform": "snapchat"}), json!({"monitor": "left", "platform": "reels"})] {
        let r = d.call("engine.execute", json!({"command": "view.safeZones", "params": p}));
        assert_eq!(r["ok"], false, "{p}: {r}");
    }
    let r = d.call("ui.set", json!({"program": {"safe_zones": "snapchat"}}));
    assert_eq!(r["ok"], false, "{r}");
    assert_eq!(d.view("program")["safe_zones"], Value::Null);
}
