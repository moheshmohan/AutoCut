//! Headless UI test of Clip ▸ Auto Redact Faces and Plates…: the dialog's controls by automation
//! id, and OK running the job that puts a masked blur on the clip. The detector and the vision
//! check are test stand-ins (the real ones need downloaded models and Ollama). With
//! `FILMCRAFT_UI_SNAPSHOT_DIR` set, the dialog and the result are saved as PNGs to look at.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Sender, channel};

use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_engine::redact::{Crop, Detection, Kind, VisionJudge};
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
    snapshots: Option<PathBuf>,
}

/// Says every face is a person's and finds nothing named.
struct People;

impl VisionJudge for People {
    fn name(&self) -> String {
        "test".into()
    }
    fn judge(&self, _: Kind, crops: &[Crop]) -> Result<Vec<Option<bool>>, String> {
        Ok(vec![Some(true); crops.len()])
    }
}

impl Driver {
    fn new() -> Self {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        // a "face" in the middle of every frame
        s.redacts.detector = Some(Arc::new(|_: &[u8], _: usize, _: usize, _: usize| {
            vec![Detection { kind: Kind::Face, score: 0.95, bbox: [0.45, 0.35, 0.1, 0.2], points: Vec::new(), pose: false }]
        }));
        s.redacts.judge = Some(Arc::new(People));
        let (tx, rx) = channel();
        let mut app = FilmcraftApp::new(s).with_control(rx);
        app.hooks.open_path = Some(Box::new(|_, _| Ok(())));
        let snapshots = std::env::var_os("FILMCRAFT_UI_SNAPSHOT_DIR").map(PathBuf::from);
        let mut b = Harness::builder().with_size(egui::vec2(1600.0, 980.0)).with_max_steps(10_000);
        if snapshots.is_some() {
            b = b.wgpu();
        }
        let harness = b.build_eframe(move |_cc| app);
        let mut d = Driver { harness, tx, snapshots };
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

    fn has(&mut self, id: &str) -> bool {
        let v = self.ok("ui.elements", json!({"prefix": id}));
        v.as_array().unwrap().iter().any(|e| e["id"] == json!(id))
    }

    fn click(&mut self, id: &str) {
        self.ok("ui.click", json!({"id": id}));
        self.frames(3);
    }

    fn app(&mut self) -> &mut FilmcraftApp {
        self.harness.state_mut()
    }

    fn snapshot(&mut self, name: &str) {
        let Some(dir) = self.snapshots.clone() else { return };
        self.frames(2);
        match self.harness.render() {
            Ok(img) => {
                std::fs::create_dir_all(&dir).unwrap();
                img.save(dir.join(format!("auto-redact-{name}.png"))).unwrap();
            }
            Err(e) => eprintln!("snapshot {name} skipped: {e}"),
        }
    }
}

#[test]
fn the_dialog_runs_the_redaction() {
    let mut d = Driver::new();
    // it needs a selected video clip
    assert_eq!(d.call("ui.menu.invoke", json!({"id": "clip.autoRedact"}))["ok"], false);
    let clip = d.app().session.active_sequence().unwrap().video_tracks[0].items[0].id;
    d.ok("engine.execute", json!({"command": "timeline.select", "params": {"clips": [clip.0]}}));
    let r = d.ok("ui.menu.invoke", json!({"id": "clip.autoRedact"}));
    d.frames(3);
    assert_eq!(r["dialog"], "autoRedact");
    for id in [
        "autoRedact.faces",
        "autoRedact.plates",
        "autoRedact.targets",
        "autoRedact.minFaceHeight",
        "autoRedact.style.blur",
        "autoRedact.style.mosaic",
        "autoRedact.strength",
        "autoRedact.padding",
        "autoRedact.verify",
        "autoRedact.ok",
        "autoRedact.cancel",
    ] {
        assert!(d.has(id), "{id}");
    }
    d.click("autoRedact.style.mosaic");
    d.click("autoRedact.plates");
    d.snapshot("dialog");
    d.click("autoRedact.ok");
    assert!(d.app().ui.extras.dialog.is_none(), "{:?}", d.app().ui.extras.dialog);
    // the job runs in the background; the masks land when it finishes (one undo step)
    let mut done = false;
    for _ in 0..3000 {
        d.frames(1);
        let q = d.app().session.active_sequence().unwrap();
        if q.find_item(clip).is_some_and(|(_, it)| it.effects.iter().any(filmcraft_engine::redact::is_auto_redaction)) {
            done = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(done, "the clip got no redaction");
    let q = d.app().session.active_sequence().unwrap().clone();
    let it = q.find_item(clip).unwrap().1;
    let fx = it.effects.iter().find(|e| filmcraft_engine::redact::is_auto_redaction(e)).unwrap();
    assert_eq!(fx.effect, "mosaic");
    assert_eq!(fx.masks[0].name, "Redact face 1");
    assert_eq!(d.app().session.history.undo.last().unwrap().0, filmcraft_engine::redact::LABEL);
    d.snapshot("done");
}
