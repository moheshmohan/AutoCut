//! Keyboard stepping in a numeric editor through the control channel.

use std::sync::mpsc::{Sender, channel};

use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_project::{ClipId, ParamValue};
use filmcraft_time::Tick;
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
    /// The window is in the background: egui sees it focused only on frames carrying control input.
    background: bool,
}

impl Driver {
    fn demo() -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).expect("demo project");
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(session).with_control(rx);
        let harness = Harness::builder().with_size(egui::vec2(1600.0, 980.0)).with_max_steps(10_000).build_eframe(move |_cc| app);
        let mut d = Driver { harness, tx, background: false };
        d.frames(4);
        d
    }
    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            let ctx = self.harness.ctx.clone();
            let mut raw = std::mem::take(self.harness.input_mut());
            raw.focused = !self.background;
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
            if let Ok(r) = reply.try_recv() {
                assert_eq!(r["ok"], true, "{method} {params}: {r}");
                self.frames(2);
                return r["result"].clone();
            }
        }
        panic!("no reply to {method} {params}");
    }
    fn exec(&mut self, command: &str, params: Value) -> Value {
        self.ok("engine.execute", json!({"command": command, "params": params}))
    }
    /// A new text layer, selected and showing in Properties; its clip.
    fn text_layer(&mut self) -> u64 {
        let clip = self.exec("graphics.newText", json!({"text": "Hi"}))["clip"].as_u64().expect("graphic clip");
        self.exec("timeline.select", json!({"clips": [clip]}));
        self.ok("ui.panel.show", json!({"panel": "Properties"}));
        self.frames(4);
        clip
    }
    fn rotation(&self, clip: u64) -> f64 {
        let s = &self.harness.state().session;
        let (_, it) = s.active_sequence().unwrap().find_item(ClipId(clip)).unwrap();
        let e = it.effects.iter().find(|e| e.effect == "graphic_text").unwrap();
        match e.param("rotation").unwrap().value_at(Tick::ZERO) {
            ParamValue::Float(v) => v,
            v => panic!("rotation is {v:?}"),
        }
    }
    fn playhead(&self) -> Tick {
        self.harness.state().session.playhead()
    }
}

#[test]
fn up_and_down_step_the_value_and_not_the_playhead() {
    let mut d = Driver::demo();
    let clip = d.text_layer();
    d.exec("graphics.set", json!({"clip": clip, "layer": 0, "props": {"rotation": 12.5}}));
    let playhead = d.playhead();
    d.ok("ui.click", json!({"id": "graphics.prop.rotation"}));
    d.ok("ui.key", json!({"key": "Up"}));
    assert_eq!(d.rotation(clip), 13.5, "Up adds one");
    d.ok("ui.key", json!({"key": "Shift+Down"}));
    assert_eq!(d.rotation(clip), 3.5, "Shift+Down subtracts ten");
    d.ok("ui.type", json!({"text": "40"}));
    d.ok("ui.key", json!({"key": "Enter"}));
    assert_eq!(d.rotation(clip), 40.0, "the stepped value was still selected, so typing replaced it");
    assert_eq!(d.playhead(), playhead, "the arrows do not move the playhead");
}
