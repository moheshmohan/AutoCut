//! Headless UI tests of the creator tools by automation id: Clip ▸ Speed Ramp… and Clip ▸
//! Stabilize….

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
    fn new() -> Self {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        s.execute("sequence.linkedSelection", json!({"on": false})).unwrap();
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(s).with_control(rx);
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

    fn menu(&mut self, id: &str) -> Value {
        let r = self.ok("ui.menu.invoke", json!({"id": id}));
        self.frames(3);
        r
    }

    fn click(&mut self, id: &str) {
        self.ok("ui.click", json!({"id": id}));
        self.frames(3);
    }

    fn has(&mut self, id: &str) -> bool {
        let v = self.ok("ui.elements", json!({"prefix": id}));
        v.as_array().unwrap().iter().any(|e| e["id"] == json!(id))
    }

    fn app(&mut self) -> &mut FilmcraftApp {
        self.harness.state_mut()
    }

    fn last_v1(&mut self) -> filmcraft_engine::project::TrackItem {
        self.app().session.active_sequence().unwrap().video_tracks[0].items.last().cloned().unwrap()
    }
}

#[test]
fn speed_ramp_dialog_from_the_clip_menu() {
    let mut d = Driver::new();
    let c = d.last_v1();
    d.exec("timeline.select", json!({"clips": [c.id.0]}));
    // Clip ▸ Speed Ramp… opens the dialog; Cancel leaves the clip
    d.menu("clip.speedRamp");
    let dlg = d.app().ui.clip_dialog.clone().expect("the dialog opened");
    assert_eq!(dlg.command, "clip.speedRamp");
    for id in ["montage", "hero", "bullet", "jumpCut", "flashIn", "flashOut", "smoothIn", "smoothOut", "none"] {
        assert!(d.has(&format!("speedRamp.preset.{id}")), "{id}");
    }
    for id in ["speedRamp.intensity", "speedRamp.opticalFlow", "speedRamp.maintainPitch", "speedRamp.ripple", "speedRamp.ok"] {
        assert!(d.has(id), "{id}");
    }
    d.click("speedRamp.cancel");
    assert!(d.app().ui.clip_dialog.is_none());
    assert_eq!(d.last_v1(), c, "Cancel leaves it");
    // Clip ▸ Speed Ramp…: Hero at intensity 1.5, OK
    d.menu("clip.speedRamp");
    d.click("speedRamp.preset.hero");
    if let Some(dlg) = d.app().ui.clip_dialog.as_mut() {
        dlg.params["intensity"] = json!(1.5);
    }
    d.click("speedRamp.ok");
    assert!(d.app().ui.clip_dialog.is_none(), "OK closes it");
    let got = d.last_v1();
    assert!(got.has_time_remap(), "the curve is on the clip");
    assert_ne!(got.duration, c.duration, "Hero's mean speed changes the length");
    assert_eq!(got.time_interpolation, filmcraft_project::TimeInterpolation::OpticalFlow);
    // one undo step
    d.exec("edit.undo", json!({}));
    assert_eq!(d.last_v1(), c);
}

#[test]
fn stabilize_dialog_adds_warp_stabilizer_at_a_level() {
    let mut d = Driver::new();
    d.ok("ui.set", json!({"workspace": "Effects"}));
    let c = d.last_v1();
    d.exec("playhead.set", json!({"time": c.start.0}));
    d.exec("timeline.select", json!({"clips": [c.id.0]}));
    d.menu("clip.stabilize");
    assert_eq!(d.app().ui.clip_dialog.as_ref().map(|g| g.command.clone()).as_deref(), Some("clip.stabilize"));
    for id in ["recommended", "minimumCrop", "mostStable"] {
        assert!(d.has(&format!("stabilize.level.{id}")), "{id}");
    }
    d.click("stabilize.level.mostStable");
    d.click("stabilize.ok");
    let got = d.last_v1();
    let e = got.effects.iter().find(|e| e.effect == "warp_stabilizer").expect("Warp Stabilizer added");
    assert_eq!(e.param("level").map(|p| p.value.clone()), Some(filmcraft_project::ParamValue::Choice(3)));
    assert_eq!(e.param("smoothness").and_then(|p| p.value.as_f64()), Some(120.0));
    // Effect Controls lists it
    d.frames(4);
    assert!(d.has("effectControls.effect.warp_stabilizer"), "Effect Controls shows the stabilizer");
    d.exec("edit.undo", json!({}));
    assert!(!d.last_v1().effects.iter().any(|e| e.effect == "warp_stabilizer"));
}

#[test]
fn the_effects_panel_finds_the_creator_effects_and_transitions() {
    let mut d = Driver::new();
    d.ok("ui.panel.show", json!({"panel": "Effects"}));
    for (query, id) in
        [("vhs", "cr_vhs"), ("film grain", "cr_film_grain"), ("halftone", "cr_halftone"), ("pull in", "crt_pull_in"), ("swipe left", "crt_swipe_left")]
    {
        d.ok("ui.set", json!({"effectsSearch": query}));
        d.frames(3);
        assert!(d.has(&format!("effects.item.{id}")), "{query}: effects.item.{id}");
    }
    // a transition found by its keywords
    d.ok("ui.set", json!({"effectsSearch": "whip"}));
    d.frames(3);
    assert!(d.has("effects.item.crt_swipe_up"), "Swipe Up is a whip pan");
    // applying a Creator effect from the panel's id lands on the selected clip
    let c = d.last_v1();
    d.exec("timeline.select", json!({"clips": [c.id.0]}));
    d.exec("effects.apply", json!({"effect": "cr_glitch"}));
    assert!(d.last_v1().effects.iter().any(|e| e.effect == "cr_glitch"));
}
