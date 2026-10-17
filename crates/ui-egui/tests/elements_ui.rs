//! Headless UI test of Effects panel ▸ Elements: the library's folders, searching it, and putting a
//! sound and a sticker on the timeline (right-click ▸ Insert at Playhead).

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

    fn app(&mut self) -> &mut FilmcraftApp {
        self.harness.state_mut()
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

    fn click(&mut self, id: &str) {
        self.ok("ui.click", json!({"id": id}));
        self.frames(2);
    }

    fn ids(&mut self, prefix: &str) -> Vec<String> {
        let v = self.ok("ui.elements", json!({"prefix": prefix}));
        v.as_array().unwrap().iter().filter_map(|e| e["id"].as_str().map(str::to_string)).collect()
    }

    fn clips(&mut self, kind: &str) -> Vec<Value> {
        let q = self.ok("engine.execute", json!({"command": "sequence.inspect", "params": {}}));
        q[kind].as_array().unwrap().iter().flat_map(|t| t["items"].as_array().cloned().unwrap_or_default()).collect()
    }

    fn search(&mut self, q: &str) {
        self.app().ui.effects_search = q.into();
        self.frames(3);
    }
}

#[test]
fn the_elements_folder_lists_searches_and_inserts() {
    let scratch = std::env::temp_dir().join(format!("filmcraft-elements-ui-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    let mut d = Driver::demo(&scratch.to_string_lossy());
    d.ok("ui.panel.show", json!({"panel": "Effects"}));
    d.app().ui.expanded_fx.clear();
    d.frames(3);
    d.click("effects.folder.Elements");
    let folders = d.ids("effects.folder.Elements/");
    for f in ["Sounds", "Stickers", "Animated Text"] {
        assert!(folders.iter().any(|x| *x == format!("effects.folder.Elements/{f}")), "{f} in {folders:?}");
    }
    d.click("effects.folder.Elements/Stickers");
    let rows = d.ids("effects.element.");
    assert_eq!(rows.len(), 10, "the stickers: {rows:?}");
    assert!(rows.iter().all(|r| r.starts_with("effects.element.sticker.")), "{rows:?}");
    // a search opens every folder and keeps the matches
    d.search("pop");
    assert_eq!(d.ids("effects.element."), ["effects.element.sound.pop", "effects.element.text.title-pop"]);
    // right-click ▸ Insert at Playhead: the sound goes on a free audio track at the playhead
    let audio = d.clips("audio").len();
    d.ok("ui.click", json!({"id": "effects.element.sound.pop", "button": "right"}));
    d.frames(3);
    d.click("effects.elementMenu.insert");
    d.frames(2);
    let after = d.clips("audio");
    assert_eq!(after.len(), audio + 1, "a sound was added: {}", d.app().ui.status);
    assert!(after.iter().any(|c| c["name"] == "Pop"), "{after:?}");
    assert!(d.app().ui.status.starts_with("Added Pop on A"), "{}", d.app().ui.status);
    // and a sticker on a video track
    d.search("star");
    let video = d.clips("video").len();
    d.ok("ui.click", json!({"id": "effects.element.sticker.star", "button": "right"}));
    d.frames(3);
    d.click("effects.elementMenu.insert");
    d.frames(2);
    let after = d.clips("video");
    assert_eq!(after.len(), video + 1, "a sticker was added");
    assert!(after.iter().any(|c| c["name"] == "Star"), "{after:?}");
    let _ = std::fs::remove_dir_all(&scratch);
}
