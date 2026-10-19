//! Headless UI tests of adjustment clips and per-track auto-duck: Project panel ▸ New Item ▸
//! Adjustment Clip, the clip menu's New Adjustment Clip, and the audio track header's Auto-Duck
//! toggle (`timeline.track.A<n>.autoDuck`).

use std::sync::mpsc::{Sender, channel};

use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_project::ItemKind;
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
}

impl Driver {
    fn demo() -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).expect("demo project");
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

    fn element(&mut self, id: &str) -> Option<Value> {
        let v = self.ok("ui.elements", json!({"prefix": id}));
        v.as_array().unwrap().iter().find(|e| e["id"] == json!(id)).cloned()
    }

    fn wait_for(&mut self, id: &str) {
        for _ in 0..60 {
            if self.element(id).is_some() {
                return;
            }
            self.frames(2);
        }
        panic!("{id} never appeared");
    }

    fn app(&mut self) -> &mut FilmcraftApp {
        self.harness.state_mut()
    }

    /// Clips that show an adjustment layer item in the active sequence.
    fn adjustment_clips(&mut self) -> Vec<u64> {
        let s = &self.app().session;
        let q = s.active_sequence().unwrap();
        q.video_tracks
            .iter()
            .flat_map(|t| t.items.iter())
            .filter(|i| s.project.item(i.item).is_some_and(|it| matches!(it.kind, ItemKind::AdjustmentLayer { .. })))
            .map(|i| i.id.0)
            .collect()
    }
}

#[test]
fn new_item_menu_and_clip_menu_make_adjustment_clips() {
    let mut d = Driver::demo();
    assert!(d.adjustment_clips().is_empty());
    d.click("project.button.new-item");
    d.wait_for("project.newItem.timeline.newAdjustmentClip");
    d.click("project.newItem.timeline.newAdjustmentClip");
    let made = d.adjustment_clips();
    assert_eq!(made.len(), 1, "New Item ▸ Adjustment Clip places one");
    assert_eq!(d.app().session.state.selection.iter().map(|c| c.0).collect::<Vec<_>>(), made);
    // the clip menu over a selected clip: one more, above it
    let clip = {
        let q = d.app().session.active_sequence().unwrap();
        q.video_tracks[0].items[0].id.0
    };
    d.exec("timeline.select", json!({"clips": [clip]}));
    d.frames(3);
    d.ok("ui.click", json!({"id": format!("timeline.clip.{clip}"), "button": "right"}));
    d.frames(3);
    d.wait_for("timeline.clipMenu.timeline.newAdjustmentClip");
    d.click("timeline.clipMenu.timeline.newAdjustmentClip");
    assert_eq!(d.adjustment_clips().len(), 2);
    // both are one undo step each
    d.exec("edit.undo", json!({}));
    d.exec("edit.undo", json!({}));
    assert!(d.adjustment_clips().is_empty());
}

#[test]
fn audio_track_headers_toggle_auto_duck() {
    let mut d = Driver::demo();
    let n = d.app().session.active_sequence().unwrap().audio_tracks.len();
    assert!(n >= 2, "the demo has audio tracks");
    d.wait_for("timeline.track.A2.autoDuck");
    assert_eq!(d.element("timeline.track.A2.autoDuck").unwrap()["label"], "Auto-Duck Off");
    d.click("timeline.track.A2.autoDuck");
    let ad = d.app().session.active_sequence().unwrap().audio_tracks[1].auto_duck.clone();
    let ad = ad.expect("A2 ducks");
    let a1 = d.app().session.active_sequence().unwrap().audio_tracks[0].id;
    assert!(ad.under.contains(&a1), "under the other tracks: {ad:?}");
    assert_eq!(d.element("timeline.track.A2.autoDuck").unwrap()["label"], "Auto-Duck On");
    d.click("timeline.track.A2.autoDuck");
    assert!(d.app().session.active_sequence().unwrap().audio_tracks[1].auto_duck.is_none());
    assert!(d.element("timeline.track.V1.autoDuck").is_none(), "video tracks have no toggle");
}
