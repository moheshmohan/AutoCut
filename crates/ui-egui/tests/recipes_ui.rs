//! Headless UI test of recipes through the control channel, on the demo project: save a recipe,
//! run it from the Recipes panel (a background job the app's frames serve), see each step; a
//! failing step stops the run and is shown; a run is cancelled; an edit is recorded and saved as
//! a recipe; a planned AutoCut intent runs through the real planner; hostile drafts are refused.

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
    fn demo(dir: &std::path::Path) -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).expect("demo project");
        session.execute("file.projectSettings.scratchDisks", json!({"captured": dir.to_string_lossy()})).expect("scratch disk");
        session.prefs_path = Some(dir.join("prefs.json"));
        session.recipes.planner = Some(filmcraft_autocut::recipes::planner());
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

    fn exec(&mut self, id: &str, params: Value) -> Value {
        self.ok("engine.execute", json!({"command": id, "params": params}))
    }

    fn click(&mut self, id: &str) {
        self.wait_for(id);
        self.ok("ui.click", json!({"id": id}));
        self.frames(2);
    }

    fn ids(&mut self, prefix: &str) -> Vec<String> {
        let v = self.ok("ui.elements", json!({"prefix": prefix}));
        v.as_array().unwrap().iter().filter_map(|e| e["id"].as_str().map(str::to_string)).collect()
    }

    /// Frames (scrolling the panel down, then back up, when it is not in view) until widget `id`
    /// is registered.
    fn wait_for(&mut self, id: &str) {
        for k in 0..200 {
            if self.ids(id).iter().any(|x| x == id) {
                return;
            }
            // scroll the panel at its Refresh button (always near the list)
            let v = self.ok("ui.elements", json!({"prefix": "recipes."}));
            if let Some(r) = v.as_array().unwrap().iter().find(|e| e["id"].as_str().is_some_and(|i| i.starts_with("recipes."))).map(|e| e["rect"].clone()) {
                let (x, y) = (r[0].as_f64().unwrap() + 4.0, r[1].as_f64().unwrap() + 4.0);
                let dy = if (k / 10) % 2 == 0 { -120.0 } else { 120.0 };
                self.ok("ui.scroll", json!({"x": x, "y": y, "dx": 0.0, "dy": dy}));
            }
            self.frames(2);
        }
        panic!("{id} never showed: {:?}", self.ids("recipes."));
    }

    /// Frames until the panel's run job finishes; its `jobs.list` entry.
    fn finish_run(&mut self) -> Value {
        let job = self.app().ui.recipes.job.expect("a run job");
        for _ in 0..3000 {
            self.frames(1);
            let jobs = self.exec("jobs.list", json!({}));
            let j = jobs.as_array().unwrap().iter().find(|j| j["id"] == job).cloned().unwrap();
            if j["finished"] == true {
                self.frames(2);
                return j;
            }
        }
        panic!("the recipe never finished");
    }

    fn markers(&mut self) -> Vec<String> {
        self.app().session.active_sequence().map(|q| q.markers.iter().map(|m| m.name.clone()).collect()).unwrap_or_default()
    }
}

#[test]
fn save_run_fail_cancel_and_record_recipes_from_the_panel() {
    let dir = std::env::temp_dir().join(format!("filmcraft-recipes-ui-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut d = Driver::demo(&dir);
    d.ok("ui.panel.show", json!({"panel": "Recipes"}));
    d.frames(4);
    for id in [
        "recipes.item.builtin:talking-head-clean-up",
        "recipes.item.builtin:vertical-short",
        "recipes.item.builtin:template-publish-youtube",
        "recipes.record.start",
    ] {
        d.wait_for(id);
    }
    // a recipe saved over the control channel shows in the panel
    let steps = json!([
        {"label": "Mark the start", "command": "markers.add", "params": {"name": "${label}"}},
        {"label": "Select every clip", "command": "edit.selectAll"},
        {"label": "Fill the frame", "command": "clip.fillFrame"},
        {"label": "Clear the selection", "command": "edit.deselectAll"},
    ]);
    d.exec(
        "recipe.save",
        json!({"name": "UI Routine", "steps": steps, "params": [{"id": "label", "name": "Marker", "kind": "text", "default": "from a recipe"}]}),
    );
    d.wait_for("recipes.item.user:ui-routine");
    d.click("recipes.item.user:ui-routine");
    d.wait_for("recipes.param.label");
    d.ok("ui.set", json!({"recipes": {"params": {"label": "made by the panel"}}}));
    d.frames(2);
    let before = d.markers().len();
    d.click("recipes.run");
    let j = d.finish_run();
    assert!(j["result"]["error"].is_null(), "{j}");
    assert_eq!(j["recipe"]["ok"], true);
    let m = d.markers();
    assert_eq!(m.len(), before + 1);
    assert!(m.contains(&"made by the panel".to_string()), "{m:?}");
    for n in 1..=4 {
        d.wait_for(&format!("recipes.step.{n}"));
    }
    d.wait_for("recipes.progress");
    // a failing step stops the run, the panel says which
    d.exec(
        "recipe.save",
        json!({"name": "Broken", "steps": [
            {"label": "Mark", "command": "markers.add", "params": {"name": "broken-before"}},
            {"label": "A template that is not there", "command": "template.apply", "params": {"template": "builtin:no-such-template"}},
            {"label": "Never", "command": "markers.add", "params": {"name": "broken-after"}},
        ]}),
    );
    d.wait_for("recipes.item.user:broken");
    d.click("recipes.item.user:broken");
    d.click("recipes.run");
    let j = d.finish_run();
    assert!(j["result"]["error"].as_str().unwrap().starts_with("step 2 of 3"), "{j}");
    let m = d.markers();
    assert!(m.contains(&"broken-before".to_string()) && !m.contains(&"broken-after".to_string()));
    d.wait_for("recipes.step.2");
    assert_eq!(j["recipe"]["steps"].as_array().unwrap().len(), 2, "the third step never ran");
    assert!(d.ids("recipes.step.3").is_empty());
    // a long export, cancelled from the panel
    let out = dir.join("long.mov");
    d.exec(
        "recipe.save",
        json!({"name": "Long Export", "steps": [
            {"label": "Export", "command": "file.exportMedia", "params": {"path": out.to_string_lossy(), "preset": "Apple ProRes 422 HQ"}},
            {"label": "Never", "command": "markers.add", "params": {"name": "after-export"}},
        ]}),
    );
    d.wait_for("recipes.item.user:long-export");
    d.click("recipes.item.user:long-export");
    d.click("recipes.run");
    d.wait_for("recipes.cancel");
    d.click("recipes.cancel");
    let j = d.finish_run();
    assert_eq!(j["result"]["error"], "cancelled", "{j}");
    assert!(!d.markers().contains(&"after-export".to_string()));
    // an AutoCut intent through the real planner
    d.exec(
        "recipe.save",
        json!({"name": "Voice", "steps": [{"label": "Clean the voice", "intent": "enhance_voice", "params": {"preset": "podcast", "loudness": false}}]}),
    );
    d.wait_for("recipes.item.user:voice");
    d.click("recipes.item.user:voice");
    d.click("recipes.run");
    let j = d.finish_run();
    assert!(j["result"]["error"].is_null(), "{j}");
    let cmds: Vec<String> = j["recipe"]["steps"][0]["commands"].as_array().unwrap().iter().map(|c| c["command"].as_str().unwrap().to_string()).collect();
    assert!(cmds.iter().any(|c| c == "essentialSound.applyPreset"), "{cmds:?}");
    // record this edit as a recipe
    d.click("recipes.record.start");
    assert!(d.app().session.recipes.recording.is_some());
    d.exec("markers.add", json!({"name": "recorded"}));
    d.frames(2);
    d.wait_for("recipes.record.name");
    d.ok("ui.set", json!({"recipes": {"recordName": "Recorded Mark"}}));
    d.frames(2);
    d.click("recipes.record.stop");
    assert!(d.app().ui.status.contains("Recorded Mark"), "{}", d.app().ui.status);
    assert_eq!(d.app().ui.recipes.selected.as_deref(), Some("user:recorded-mark"));
    d.click("recipes.run");
    let j = d.finish_run();
    assert!(j["result"]["error"].is_null(), "{j}");
    assert_eq!(d.markers().iter().filter(|m| *m == "recorded").count(), 2);
    // edit its steps as JSON
    d.click("recipes.edit");
    d.wait_for("recipes.edit.steps");
    d.ok("ui.set", json!({"recipes": {"stepsJson": "[{\"label\": \"Edited\", \"command\": \"markers.add\", \"params\": {\"name\": \"edited\"}}]"}}));
    d.frames(2);
    d.click("recipes.edit.save");
    let e = filmcraft_engine::recipes::find(&d.app().session, "user:recorded-mark", "t").unwrap();
    assert_eq!(e.recipe.steps[0].label, "Edited");
    // delete it
    d.click("recipes.delete");
    assert!(filmcraft_engine::recipes::find(&d.app().session, "user:recorded-mark", "t").is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hostile_recipe_drafts_are_refused() {
    let dir = std::env::temp_dir().join(format!("filmcraft-recipes-ui-hostile-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let mut d = Driver::demo(&dir);
    d.ok("ui.panel.show", json!({"panel": "Recipes"}));
    d.frames(4);
    for bad in [json!({"recipes": {"params": "all"}}), json!({"recipes": {"selected": 7}}), json!({"recipes": {"stepsJson": "x".repeat(300 * 1024)}})] {
        let v = d.call("ui.set", bad.clone());
        assert_eq!(v["ok"], json!(false), "{bad}: {v}");
    }
    // a recipe that is not there; long text cut; bad values refused at Run with a message
    d.ok("ui.set", json!({"recipes": {"selected": "user:no-such-recipe"}}));
    d.frames(3);
    d.ok("ui.set", json!({"recipes": {"selected": "builtin:vertical-short", "params": {"seconds": "many", "output": "y".repeat(10_000)}}}));
    d.frames(3);
    assert!(d.app().ui.recipes.params["output"].as_str().unwrap().chars().count() <= 4096);
    let _ = std::fs::remove_dir_all(&dir);
}
