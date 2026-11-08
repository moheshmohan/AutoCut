//! Headless UI tests of Export mode's Destinations (B5), driven through the control channel:
//! the four rows and their connect state, a platform's form (metadata, the file, the pre-flight
//! checklist with its fixes, Publish disabled with the reason until the checks pass), and a
//! publish through `publish.run` to a stand-in Instagram publisher.
//!
//! Set `FILMCRAFT_UI_SNAPSHOT_DIR=<dir>` to also render the window offscreen with wgpu and write
//! `destinations-*.png` there; without it no GPU is needed.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};

use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_engine::instagram::{InstagramPublisher, ReelRequest};
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
    dir: PathBuf,
    snapshots: Option<PathBuf>,
}

impl Driver {
    fn new(name: &str, session: Session) -> Self {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/export-tests").join(format!("ui-dest-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(session).with_control(rx);
        let snapshots = std::env::var_os("FILMCRAFT_UI_SNAPSHOT_DIR").map(PathBuf::from);
        let mut b = Harness::builder().with_size(egui::vec2(1600.0, 980.0)).with_max_steps(10_000);
        if snapshots.is_some() {
            b = b.wgpu();
        }
        let harness = b.build_eframe(move |_cc| app);
        let mut d = Driver { harness, tx, dir, snapshots };
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

    fn element(&mut self, id: &str) -> Option<Value> {
        self.ok("ui.elements", json!({"prefix": id})).as_array().unwrap().iter().find(|e| e["id"] == json!(id)).cloned()
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
                img.save(dir.join(format!("destinations-{name}.png"))).unwrap();
            }
            Err(e) => eprintln!("snapshot {name} skipped: {e}"),
        }
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[derive(Default)]
struct FakeIg {
    reels: Mutex<Vec<ReelRequest>>,
}

impl InstagramPublisher for FakeIg {
    fn connect_ready(&self) -> Result<(), String> {
        Ok(())
    }
    fn ready(&self) -> Result<(), String> {
        Ok(())
    }
    fn account(&self) -> Result<Value, String> {
        Ok(json!({"account": {"username": "moheshmohan"}}))
    }
    fn connect(&self, _: &filmcraft_engine::export::Progress) -> Result<Value, String> {
        Ok(json!({}))
    }
    fn disconnect(&self) -> Result<Value, String> {
        Ok(json!({}))
    }
    fn limits(&self) -> Result<Value, String> {
        Ok(json!({}))
    }
    fn check(&self, _: &ReelRequest) -> Result<Value, String> {
        Ok(json!({"account": {"username": "moheshmohan"}}))
    }
    fn publish(&self, req: &ReelRequest, p: &filmcraft_engine::export::Progress) -> Result<Value, String> {
        self.reels.lock().unwrap().push(req.clone());
        p.done.store(p.total.load(Ordering::Relaxed), Ordering::Relaxed);
        Ok(json!({"published": true, "mediaId": "18000000000001", "permalink": "https://www.instagram.com/reel/Cx/"}))
    }
}

/// A short vertical H.264 export of the demo project, `secs` long.
fn export(s: &mut Session, path: &str, secs: f64) {
    s.execute("file.exportMedia", json!({"path": path, "format": "h264", "width": 180, "height": 320, "bitrateKbps": 800, "range": "custom", "startSeconds": 0.0, "endSeconds": secs, "audio": false, "wait": true}))
        .unwrap();
}

#[test]
fn destination_rows_forms_preflight_and_publish() {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    let ig = Arc::new(FakeIg::default());
    s.instagram.publisher = Some(ig.clone());
    let mut d = Driver::new("rows", s);
    let (short, good) = (d.dir.join("short.mp4").to_string_lossy().into_owned(), d.dir.join("good.mp4").to_string_lossy().into_owned());
    export(&mut d.app().session, &short, 1.0);
    export(&mut d.app().session, &good, 3.5);
    d.ok("ui.set", json!({"mode": "export"}));
    d.frames(3);
    // the four rows, with their state
    for (id, label) in [
        ("export.destination.mediaFile", "Media File"),
        ("export.destination.youtube", "YouTube: Not available"),
        ("export.destination.youtubeShort", "YouTube Short: Not available"),
        ("export.destination.instagramReel", "Instagram Reel: Connected"),
    ] {
        let e = d.element(id).unwrap_or_else(|| panic!("{id} missing"));
        assert_eq!(e["label"], label, "{e}");
    }
    assert!(d.element("export.fileName").is_some(), "the Media File settings by default");
    // the Instagram form replaces the Media File settings
    d.click("export.destination.instagramReel");
    assert!(d.element("export.fileName").is_none());
    for id in ["export.publish.caption", "export.publish.shareToFeed", "export.publish.coverSeconds", "export.publish.path", "export.publish.check"] {
        assert!(d.element(id).is_some(), "{id} missing");
    }
    let run = d.element("export.publish.run").unwrap();
    assert_eq!(run["enabled"], false);
    // a 1 s file: the check says what fails and how to fix it, Publish stays off
    d.ok("ui.set", json!({"export": {"publish": {"path": short, "caption": "Robots #ai"}}}));
    d.frames(2);
    d.click("export.publish.check");
    let dur = d.element("export.publish.check.duration").expect("the duration check");
    assert!(dur["label"].as_str().unwrap().starts_with("fail: Duration"), "{dur}");
    assert!(d.element("export.publish.fix").is_none(), "no copy mends a length");
    assert!(d.element("export.publish.connect").is_none(), "the account is connected");
    let summary = d.element("export.publish.summary").unwrap();
    assert!(summary["label"].as_str().unwrap().contains("problem"), "{summary}");
    let run = d.element("export.publish.run").unwrap();
    assert_eq!((run["enabled"].as_bool(), run["label"].as_str()), (Some(false), Some("Fix what the check found first")), "{run}");
    d.snapshot("instagram-failing");
    let count = d.element("export.publish.captionCount").unwrap();
    assert!(count["label"].as_str().unwrap().starts_with("10/2200 · #1/30"), "{count}");
    // a good file: checked, then published through publish.run
    d.ok("ui.set", json!({"export": {"publish": {"path": good}}}));
    d.frames(2);
    assert_eq!(d.element("export.publish.run").unwrap()["label"], "Check the file first", "a new file needs its own check");
    d.click("export.publish.check");
    assert_eq!(d.element("export.publish.check.duration").unwrap()["label"].as_str().map(|l| l.starts_with("pass")), Some(true));
    assert!(d.element("export.publish.fix").is_none(), "native MP4 exports already have fast-start metadata; no copy is needed");
    let run = d.element("export.publish.run").unwrap();
    assert_eq!(run["enabled"], true, "{run}");
    d.click("export.publish.run");
    for _ in 0..200 {
        if d.element("export.publish.open").is_some() {
            break;
        }
        d.frames(2);
    }
    d.snapshot("instagram-published");
    let open = d.element("export.publish.open").expect("the link to the Reel");
    assert_eq!(open["label"], "https://www.instagram.com/reel/Cx/");
    let r = ig.reels.lock().unwrap()[0].clone();
    assert_eq!((r.caption.as_deref(), r.path.as_deref(), r.share_to_feed), (Some("Robots #ai"), Some(good.as_str()), true));
    // a YouTube Short: its own form; without a publisher Publish says why
    d.click("export.destination.youtubeShort");
    for id in ["export.publish.title", "export.publish.description", "export.publish.tags", "export.publish.privacy"] {
        assert!(d.element(id).is_some(), "{id} missing");
    }
    let run = d.element("export.publish.run").unwrap();
    assert_eq!((run["enabled"].as_bool(), run["label"].as_str()), (Some(false), Some("YouTube is not available in this app")), "{run}");
    d.snapshot("youtube-short");
    // back to the Media File
    d.click("export.destination.mediaFile");
    assert!(d.element("export.fileName").is_some());
}
