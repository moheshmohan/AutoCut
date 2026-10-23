//! Headless UI tests of editing with nested sequences: the nest toggle button, Reveal in Project
//! and the clip menu's Multi-Camera submenu.

use std::sync::mpsc::{Sender, channel};

use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use filmcraft_ui_egui::dock::PanelKind;
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
}

impl Driver {
    fn new() -> Self {
        Self::with_height(980.0)
    }

    fn with_height(height: f32) -> Self {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(s).with_control(rx);
        let harness = Harness::builder().with_size(egui::vec2(1600.0, height)).with_max_steps(10_000).build_eframe(move |_cc| app);
        let mut d = Driver { harness, tx };
        d.frames(4);
        d
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            // the control channel's clicks and keys enter through the input hook
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

    fn has(&mut self, id: &str) -> bool {
        let v = self.ok("ui.elements", json!({"prefix": id}));
        v.as_array().unwrap().iter().any(|e| e["id"] == json!(id))
    }

    fn app(&mut self) -> &mut FilmcraftApp {
        self.harness.state_mut()
    }

    /// [x, y, width, height]
    fn rect(&mut self, id: &str) -> [f64; 4] {
        let v = self.ok("ui.elements", json!({"prefix": id}));
        let e = v.as_array().unwrap().iter().find(|e| e["id"] == json!(id)).unwrap_or_else(|| panic!("no element {id}"));
        let r: Vec<f64> = e["rect"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        [r[0], r[1], r[2], r[3]]
    }

    /// With `FILMCRAFT_UI_SNAPSHOT_DIR=<dir>`, render the window offscreen (wgpu) to `<dir>/<name>.png`
    /// to look at; without it no GPU is needed.
    fn snapshot(&mut self, name: &str) {
        let Some(dir) = std::env::var_os("FILMCRAFT_UI_SNAPSHOT_DIR").map(std::path::PathBuf::from) else { return };
        self.frames(2);
        let img = match self.harness.render() {
            Ok(i) => i,
            Err(e) => {
                eprintln!("snapshot {name} skipped: {e}");
                return;
            }
        };
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.png"));
        img.save(&path).unwrap();
        eprintln!("snapshot: {}", path.display());
    }
}

#[test]
fn the_nest_button_toggles_how_sequences_are_edited_in() {
    let mut d = Driver::new();
    assert!(!d.app().session.state.sequences_as_clips, "on by default: sequences nest");
    d.click("timeline.toggle.nest");
    assert!(d.app().session.state.sequences_as_clips);
    d.click("timeline.toggle.nest");
    assert!(!d.app().session.state.sequences_as_clips);
}

#[test]
fn reveal_in_project_shows_the_clips_item_in_the_project_panel() {
    use filmcraft_engine::project_panel::ViewMode;
    for mode in [ViewMode::List, ViewMode::Icon] {
        let mut d = Driver::new();
        d.app().session.prefs.project_panel.view.mode = mode;
        let (clip, item) = {
            let it = &d.app().session.active_sequence().unwrap().video_tracks[0].items[2];
            (it.id.0, it.item)
        };
        let bin = d.app().session.project.root.parent_of(item).expect("the demo keeps its footage in a bin").0;
        // the Project panel is closed, its bins are collapsed and a search hides everything
        d.app().ui.dock.close(PanelKind::Project);
        d.app().ui.expanded_bins.clear();
        d.app().ui.project_search = "no such thing".into();
        d.frames(2);
        assert!(!d.has(&format!("project.item.{}", item.0)));
        d.exec("clip.revealInProject", json!({"clip": clip}));
        d.frames(3);
        assert!(d.app().ui.dock.is_visible(PanelKind::Project), "{mode:?}");
        assert!(d.app().ui.project_search.is_empty());
        assert_eq!(d.app().session.state.project_selection, vec![item]);
        if mode == ViewMode::List {
            // the list shows the tree with the item's bin opened
            assert!(d.app().ui.expanded_bins.contains(&bin));
            assert_eq!(d.app().ui.project_panel.bin, None);
        } else {
            // icons show one bin at a time: the panel is inside the item's bin
            assert_eq!(d.app().ui.project_panel.bin, Some(bin));
        }
        assert!(d.has(&format!("project.item.{}", item.0)), "{mode:?}: the item is on show");
    }
}

#[test]
fn the_clip_menu_offers_multi_camera_and_reveal_in_project() {
    let mut d = Driver::new();
    let (clip, item) = {
        let it = &d.app().session.active_sequence().unwrap().video_tracks[0].items[0];
        (it.id.0, it.item)
    };
    d.ok("ui.click", json!({"id": format!("timeline.clip.{clip}"), "button": "right"}));
    d.frames(3);
    assert!(d.has("timeline.clipMenu.clip.multicam"));
    assert!(d.has("timeline.clipMenu.clip.revealInProject"));
    d.click("timeline.clipMenu.clip.revealInProject");
    assert_eq!(d.app().session.state.project_selection, vec![item]);
}

/// The clip menu fits a 980-pt laptop window: with egui's own spacing it ran past the bottom,
/// cutting off Join Through Edits and Template Slot. Speed Ramp… and Stabilize… are in it and
/// open their dialogs.
#[test]
fn the_clip_menu_fits_the_window_and_offers_speed_ramp_and_stabilize() {
    let mut d = Driver::new();
    let clip = d.app().session.active_sequence().unwrap().video_tracks[0].items[0].id.0;
    d.ok("ui.click", json!({"id": format!("timeline.clip.{clip}"), "button": "right"}));
    d.frames(3);
    let entries = d.ok("ui.elements", json!({"prefix": "timeline.clipMenu."}));
    let entries = entries.as_array().unwrap();
    assert!(entries.len() > 30, "{entries:?}");
    for e in entries {
        let r = &e["rect"];
        let (top, bottom) = (r[1].as_f64().unwrap(), r[1].as_f64().unwrap() + r[3].as_f64().unwrap());
        assert!(top >= 0.0 && bottom <= 980.0, "{} is off the window: {r}", e["id"]);
    }
    assert!(d.has("timeline.clipMenu.clip.stabilize"));
    d.snapshot("clip-menu-980");
    d.click("timeline.clipMenu.clip.speedRamp");
    assert!(d.has("speedRamp.preset.montage"), "the Speed Ramp dialog");
}

/// #2852 with AutoCut's entries: on a screen shorter than the menu it scrolls (in the menu bar's
/// style), and its last entry, Template Slot, can be scrolled into view.
#[test]
fn the_clip_menu_scrolls_to_template_slot_on_a_short_screen() {
    let height = 600.0;
    let mut d = Driver::with_height(height as f32);
    let clip = d.app().session.active_sequence().unwrap().video_tracks[0].items[0].id.0;
    d.ok("ui.click", json!({"id": format!("timeline.clip.{clip}"), "button": "right"}));
    d.frames(3);
    let first = d.rect("timeline.clipMenu.edit.cut");
    assert!(first[1] >= 0.0, "the menu starts on the screen: {first:?}");
    let last = "timeline.clipMenu.template.markSlot";
    assert!(d.rect(last)[1] > height, "the menu is taller than the screen");
    d.snapshot("clip-menu-600-top");
    d.ok("ui.scroll", json!({"id": "timeline.clipMenu.edit.cut", "dy": -2000.0}));
    d.frames(10);
    let r = d.rect(last);
    assert!(r[1] >= 0.0 && r[1] + r[3] <= height, "Template Slot scrolled into view: {r:?}");
    assert!(d.has("timeline.clipMenu.clip.speedRamp") && d.has("timeline.clipMenu.clip.stabilize"));
    d.snapshot("clip-menu-600-bottom");
}
