//! Headless UI test of the Templates panel through the control channel, on the demo project: the
//! grid of starters with thumbnails, the layout / category / search filters, picking a template,
//! Auto-fill from the bin, text typed in, Preview (frames, the project untouched) and Build (a new
//! sequence in one undo step, shown in the Timeline); hostile drafts are refused.

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
        let harness = Harness::builder().with_size(egui::vec2(1600.0, 1200.0)).with_max_steps(10_000).build_eframe(move |_cc| app);
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

    fn click(&mut self, id: &str) {
        self.ok("ui.click", json!({"id": id}));
        self.frames(2);
    }

    fn ids(&mut self, prefix: &str) -> Vec<String> {
        let v = self.ok("ui.elements", json!({"prefix": prefix}));
        v.as_array().unwrap().iter().filter_map(|e| e["id"].as_str().map(str::to_string)).collect()
    }

    /// Scroll the template's detail view (`dy` < 0 down, > 0 up) until widget `id` shows.
    fn reveal_by(&mut self, id: &str, dy: f64) {
        for _ in 0..20 {
            if self.ids(id).iter().any(|x| x == id) {
                return;
            }
            // just above the footer's Preview button: inside the scrolling slots
            let v = self.ok("ui.elements", json!({"prefix": "templates.preview"}));
            let r = v.as_array().unwrap().iter().find(|e| e["id"] == "templates.preview").unwrap()["rect"].clone();
            let (x, y) = (r[0].as_f64().unwrap() + r[2].as_f64().unwrap() / 2.0, r[1].as_f64().unwrap() - 60.0);
            self.ok("ui.scroll", json!({"x": x, "y": y, "dx": 0.0, "dy": dy}));
            self.frames(3);
        }
        panic!("{id} never showed: {:?}", self.ids("templates."));
    }

    /// Frames until widget `id` is registered.
    fn wait_for(&mut self, id: &str) {
        for _ in 0..200 {
            if self.ids(id).iter().any(|x| x == id) {
                return;
            }
            self.frames(2);
        }
        panic!("{id} never showed: {:?}", self.ids("templates."));
    }
}

#[test]
fn browse_filter_fill_preview_and_build_a_template() {
    let scratch = std::env::temp_dir().join(format!("filmcraft-templates-ui-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let mut d = Driver::demo(&scratch.to_string_lossy());
    d.ok("ui.panel.show", json!({"panel": "Templates"}));
    d.frames(6);
    // the grid: the first starter's card, with its thumbnail drawn in the background of frames
    d.wait_for("templates.card.builtin:photo-beat-slideshow");
    for id in ["templates.search", "templates.aspect", "templates.category", "templates.slotCount", "templates.refresh"] {
        assert!(d.ids(id).iter().any(|x| x == id), "{id}");
    }
    // the layout filter, by clicks: only templates with a square layout are left
    d.click("templates.aspect");
    d.wait_for("templates.aspect.1:1");
    d.click("templates.aspect.1:1");
    assert_eq!(d.app().ui.templates.aspect, "1:1");
    d.frames(4);
    let cards = d.ids("templates.card.");
    assert!(!cards.is_empty());
    assert!(!cards.iter().any(|c| c.ends_with("builtin:tutorial-pip")), "a 16:9 and 9:16 template is filtered out: {cards:?}");
    // search, and pick the collage
    d.ok("ui.set", json!({"templates": {"query": "collage"}}));
    d.frames(4);
    assert_eq!(d.ids("templates.card."), ["templates.card.builtin:photo-collage"]);
    d.click("templates.card.builtin:photo-collage");
    assert_eq!(d.app().ui.templates.selected.as_deref(), Some("builtin:photo-collage"));
    assert_eq!(d.app().ui.templates.variant, "1:1", "the layout the browser was filtered to");
    d.wait_for("templates.build");
    for id in ["templates.back", "templates.variant", "templates.googlePhotos", "templates.autofill", "templates.slot.photos.add", "templates.preview"] {
        d.wait_for(id);
    }
    // the title starts as the template's own text
    assert_eq!(d.app().ui.templates.fills.get("title").cloned().unwrap_or_default(), [json!("Our weekend")]);
    // Auto-fill from the bin: the demo's clips go into the four cells
    d.click("templates.autofill");
    let photos = d.app().ui.templates.fills.get("photos").cloned().unwrap_or_default();
    assert_eq!(photos.len(), 4, "{:?}", d.app().ui.templates.fills);
    assert!(d.app().ui.status.starts_with("Filled"), "{}", d.app().ui.status);
    d.frames(2);
    d.reveal_by("templates.slot.photos.3.remove", -160.0);
    d.reveal_by("templates.slot.title.text.0", -160.0);
    // the title, as an agent would type it
    d.ok("ui.set", json!({"templates": {"fills": {"title": ["Our weekend away"]}, "name": "Weekend Collage"}}));
    d.frames(2);
    // Preview: frames of the filled template, the project unchanged
    let (items, undo) = {
        let s = &d.app().session;
        (s.project.items.len(), s.history.undo.len())
    };
    d.click("templates.preview");
    d.frames(3);
    assert!(d.app().ui.status.starts_with("Preview"), "{}", d.app().ui.status);
    d.reveal_by("templates.previewFrame.0", -160.0);
    assert_eq!(d.ids("templates.previewFrame.").len(), 6);
    {
        let s = &d.app().session;
        assert_eq!((s.project.items.len(), s.history.undo.len()), (items, undo), "a preview changes nothing");
    }
    // Build: one undo step, the sequence named, square, open in the Timeline
    d.click("templates.build");
    d.frames(4);
    let status = d.app().ui.status.clone();
    assert!(status.starts_with("Built “Weekend Collage”"), "{status}");
    let built = d.app().ui.templates.built.expect("a sequence");
    {
        let s = &d.app().session;
        assert_eq!(s.history.undo.len(), undo + 1);
        let it = s.project.item(filmcraft_project::ItemId(built)).unwrap();
        assert_eq!(it.name, "Weekend Collage");
        assert_eq!(it.metadata.get("Template").map(String::as_str), Some("builtin:photo-collage"));
        let q = s.project.sequence(filmcraft_project::ItemId(built)).unwrap();
        assert_eq!((q.settings.width, q.settings.height), (1080, 1080));
        assert_eq!(s.state.active_sequence, Some(filmcraft_project::ItemId(built)));
    }
    d.reveal_by("templates.openTimeline", -160.0);
    // back to the grid
    d.reveal_by("templates.back", 400.0);
    d.click("templates.back");
    assert!(d.app().ui.templates.selected.is_none());
    let _ = std::fs::remove_dir_all(&scratch);
}

#[test]
fn hostile_template_drafts_are_refused() {
    let scratch = std::env::temp_dir().join(format!("filmcraft-templates-ui-hostile-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&scratch);
    let mut d = Driver::demo(&scratch.to_string_lossy());
    d.ok("ui.panel.show", json!({"panel": "Templates"}));
    d.frames(4);
    for bad in [json!({"templates": {"fills": "everything"}}), json!({"templates": {"selected": 7}}), json!({"templates": {"fills": {"photos": 5}}})] {
        let v = d.call("ui.set", bad.clone());
        assert_eq!(v["ok"], json!(false), "{bad}: {v}");
    }
    // a template that is not there: the panel goes back to the grid
    d.ok("ui.set", json!({"templates": {"selected": "builtin:no-such-template"}}));
    d.frames(4);
    assert!(d.app().ui.templates.selected.is_none());
    // long text is cut, huge fills are capped, nothing panics
    let long = "x".repeat(50_000);
    d.ok("ui.set", json!({"templates": {"selected": "builtin:quote-card", "fills": {"quote": [long], "background": (0..1000).collect::<Vec<u64>>()}}}));
    d.frames(4);
    let f = d.app().ui.templates.fills.clone();
    assert!(f["quote"][0].as_str().unwrap().chars().count() <= 1000);
    assert!(f["background"].len() <= 256);
    // building with those refuses with a message, not a crash
    d.wait_for("templates.build");
    d.click("templates.build");
    d.frames(2);
    assert!(d.app().ui.status.starts_with("Build:"), "{}", d.app().ui.status);
    let _ = std::fs::remove_dir_all(&scratch);
}
