//! Headless UI test of the Generate panel with a stand-in tool (a shell one-liner that copies a
//! prepared picture): the panel lists the tool and what must be credited, Generate runs the
//! same `generate.image` command agents use, and the result lands on the timeline.
#![cfg(unix)]

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

    fn exec(&mut self, command: &str, params: Value) -> Value {
        self.ok("engine.execute", json!({"command": command, "params": params}))
    }

    fn ids(&mut self) -> Vec<String> {
        self.ok("ui.elements", json!({"prefix": "generate."})).as_array().unwrap().iter().filter_map(|e| e["id"].as_str().map(str::to_string)).collect()
    }

    fn draft(&mut self) -> Value {
        self.ok("ui.inspect", json!({}))["ui"]["generate"].clone()
    }

    fn status(&mut self) -> String {
        self.ok("ui.inspect", json!({}))["ui"]["status"].as_str().unwrap_or_default().to_string()
    }

    fn video_clips(&mut self) -> usize {
        let q = self.exec("sequence.inspect", json!({}));
        q["video"].as_array().unwrap().iter().map(|t| t["items"].as_array().map_or(0, Vec::len)).sum()
    }
}

#[test]
fn generate_lists_the_tool_runs_it_and_places_the_picture() {
    let scratch = std::env::temp_dir().join(format!("filmcraft-generate-ui-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let fixture = scratch.join("fixture.png");
    std::fs::write(&fixture, filmcraft_export::encode_png(vec![180; 64 * 36 * 4], 64, 36).unwrap()).unwrap();
    let manifest = filmcraft_engine::generate_run::manifest_path().unwrap();
    std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    std::fs::write(
        &manifest,
        format!(
            r#"{{"providers":[{{"id":"stand-in","kind":"image","name":"Stand-in pictures","command":["/bin/sh","-c","cp \"$1\" \"$2\"","sh","{}","{{out}}","{{prompt}}","{{seed}}"],"attribution":"Stand-in model","license":"A licence line"}}]}}"#,
            fixture.to_string_lossy()
        ),
    )
    .unwrap();
    let mut d = Driver::demo(&scratch.to_string_lossy());
    d.ok("ui.panel.show", json!({"panel": "Generate"}));
    d.frames(3);
    let ids = d.ids();
    for id in [
        "generate.mode.image",
        "generate.mode.video",
        "generate.mode.sound",
        "generate.mode.fill",
        "generate.mode.augment",
        "generate.mode.restyle",
        "generate.provider",
        "generate.prompt",
        "generate.useFrame",
        "generate.transparent",
        "generate.place",
        "generate.length",
        "generate.variations",
        "generate.depthMove",
        "generate.variation",
        "generate.run",
    ] {
        assert!(ids.iter().any(|i| i == id), "{id} in {ids:?}");
    }
    // nothing to describe yet: Generate does nothing
    d.ok("ui.click", json!({"id": "generate.run"}));
    d.frames(2);
    assert!(d.draft()["job"].is_null());
    // a picture, two seconds long, at the playhead
    d.ok("ui.set", json!({"generate": {"prompt": "a lighthouse at dusk", "seed": 5, "seconds": 2.0}}));
    d.exec("playhead.set", json!({"seconds": 1.0}));
    let before = d.video_clips();
    d.ok("ui.click", json!({"id": "generate.run"}));
    let mut landed = false;
    for _ in 0..400 {
        d.frames(1);
        if d.draft()["job"].is_null() {
            landed = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(landed, "the generation never finished");
    assert_eq!(d.status(), "Generated 1 item");
    assert_eq!(d.video_clips(), before + 1, "the picture is on the timeline");
    let made = d.exec("generate.history", json!({}));
    assert_eq!(made["generated"][0]["description"], "a lighthouse at dusk");
    // it is listed in the panel, ready to be placed again
    d.frames(240);
    d.ok("ui.set", json!({"generate": {"prompt": ""}}));
    d.frames(3);
    assert!(d.ids().iter().any(|i| i == "generate.history.1.place"), "{:?}", d.ids());
    if let Ok(dir) = std::env::var("FILMCRAFT_UI_SHOTS") {
        d.ok("ui.screenshot", json!({"path": format!("{dir}/generate-panel.png"), "panel": "Generate"}));
    }
    // two to choose from: the first is placed, and Use puts the other in its place
    d.ok("ui.set", json!({"generate": {"prompt": "a kite over the dunes", "variations": 2, "seconds": 2.0}}));
    d.exec("playhead.set", json!({"seconds": 4.0}));
    let before = d.video_clips();
    d.ok("ui.click", json!({"id": "generate.run"}));
    let mut landed = false;
    for _ in 0..400 {
        d.frames(1);
        if d.draft()["job"].is_null() {
            landed = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(landed, "the two variations never finished");
    assert_eq!(d.status(), "Generated 2 items");
    assert_eq!(d.video_clips(), before + 1, "one of the two is on the timeline");
    d.frames(3);
    let made = d.exec("generate.history", json!({}));
    let (second, first) = (made["generated"][0]["item"].as_u64().unwrap(), made["generated"][1]["item"].as_u64().unwrap());
    assert!(made["generated"][0]["variationOf"].as_str().is_some(), "{made}");
    let shown = |d: &mut Driver, item: u64| {
        let q = d.exec("sequence.inspect", json!({}));
        q["video"].as_array().unwrap().iter().flat_map(|t| t["items"].as_array().cloned().unwrap_or_default()).any(|c| c["item"].as_u64() == Some(item))
    };
    assert!(shown(&mut d, first) && !shown(&mut d, second));
    assert!(d.ids().iter().any(|i| i == "generate.history.1.use"), "{:?}", d.ids());
    // (a mode with few controls, so the list is in view to be clicked)
    d.ok("ui.set", json!({"generate": {"mode": "fill"}}));
    d.frames(3);
    d.ok("ui.click", json!({"id": "generate.history.1.use"}));
    d.frames(3);
    assert_eq!(d.status(), "This variation is on the timeline now");
    assert!(shown(&mut d, second) && !shown(&mut d, first));
    assert_eq!(d.video_clips(), before + 1);
    // the other modes show their own controls; a kind with no tool says so and cannot run
    d.ok("ui.set", json!({"generate": {"mode": "fill", "variations": 1}}));
    d.frames(3);
    let ids = d.ids();
    assert!(
        ids.iter().any(|i| i == "generate.track")
            && ids.iter().any(|i| i == "generate.allGaps")
            && ids.iter().any(|i| i == "generate.fillFrom")
            && !ids.iter().any(|i| i == "generate.provider"),
        "{ids:?}"
    );
    d.ok("ui.set", json!({"generate": {"mode": "video"}}));
    d.frames(3);
    assert!(d.ids().iter().any(|i| i == "generate.carryOn"));
    d.ok("ui.set", json!({"generate": {"mode": "augment", "prompt": "add snow"}}));
    d.frames(3);
    assert!(d.ids().iter().any(|i| i == "generate.animate"));
    d.ok("ui.set", json!({"generate": {"mode": "restyle", "prompt": "as a watercolour", "restyleLook": "paint it", "restyleStrength": 0.6}}));
    d.frames(3);
    let ids = d.ids();
    assert!(ids.iter().any(|i| i == "generate.restyleLook") && ids.iter().any(|i| i == "generate.restyleStrength"), "{ids:?}");
    assert_eq!((d.draft()["restyleLook"].as_str(), d.draft()["restyleStrength"].as_f64()), (Some("paint it"), Some(0.6)));
    d.ok("ui.click", json!({"id": "generate.run"}));
    d.frames(2);
    assert!(d.draft()["job"].is_null(), "no clip tool: nothing runs");
    d.ok("ui.set", json!({"generate": {"mode": "sound", "prompt": "rain"}}));
    d.frames(3);
    assert!(!d.ids().iter().any(|i| i == "generate.provider") && d.ids().iter().any(|i| i == "generate.forClip"));
    d.ok("ui.click", json!({"id": "generate.run"}));
    d.frames(2);
    assert!(d.draft()["job"].is_null(), "no sound tool: nothing runs");
    // a hostile draft from an agent is refused whole
    let (req, reply) = ControlRequest::new("ui.set", json!({"generate": {"seconds": "long"}}));
    d.tx.send(req).unwrap();
    d.frames(3);
    assert_eq!(reply.try_recv().unwrap()["ok"], json!(false));
    let _ = std::fs::remove_dir_all(&scratch);
}
