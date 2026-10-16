//! Headless UI tests of the crash-recovery prompt: a project saved after the auto-save (by
//! FilmCraft or another program, such as the headless CLI) is never replaced by Recover as the
//! primary choice, and an untouched project keeps Recover as before.
//!
//! `FILMCRAFT_UI_SNAPSHOT_DIR=<dir>` also renders the prompt to `<dir>/recovery-*.png`.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Sender, channel};

use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_engine::autosave::AutosaveConfig;
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
    snapshots: Option<PathBuf>,
}

impl Driver {
    fn new(session: Session) -> Self {
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(session).with_control(rx);
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

    fn menu(&mut self, id: &str) -> Value {
        let r = self.ok("ui.menu.invoke", json!({"id": id}));
        self.frames(3);
        r
    }

    fn click(&mut self, id: &str) {
        self.ok("ui.click", json!({"id": id}));
        self.frames(3);
    }

    fn element(&mut self, id: &str) -> Option<Value> {
        let v = self.ok("ui.elements", json!({"prefix": id}));
        v.as_array().unwrap().iter().find(|e| e["id"] == json!(id)).cloned()
    }

    fn app(&mut self) -> &mut FilmcraftApp {
        self.harness.state_mut()
    }

    /// Offscreen render of the recovery prompt (only with `FILMCRAFT_UI_SNAPSHOT_DIR`).
    fn snapshot(&mut self, name: &str) {
        let Some(dir) = self.snapshots.clone() else { return };
        self.frames(2);
        match self.harness.render() {
            Ok(img) => {
                std::fs::create_dir_all(&dir).unwrap();
                let path = dir.join(format!("{name}.png"));
                img.save(&path).unwrap();
                eprintln!("snapshot: {}", path.display());
            }
            Err(e) => eprintln!("snapshot {name} skipped: {e}"),
        }
    }
}

/// Left edge of an element's rect.
fn x(e: &Value) -> f64 {
    e["rect"][0].as_f64().unwrap()
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fc-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn start(s: &mut Session, data: &Path) {
    let mut cfg = AutosaveConfig::new(data);
    cfg.journal_debounce = std::time::Duration::from_millis(20);
    s.start_autosave(cfg).unwrap();
}

/// A session that saved `proj`, made an unsaved edit (journaled) and was killed.
fn crash_with_unsaved_changes(data: &Path, proj: &str) {
    let mut a = Session::default();
    a.execute("file.openDemoProject", json!({})).unwrap();
    start(&mut a, data);
    a.execute("file.saveAs", json!({"path": proj})).unwrap();
    a.execute("sequence.addEdit", json!({"seconds": 1.5})).unwrap();
    let (rev, t0) = (a.revision, std::time::Instant::now());
    loop {
        a.poll_persistence();
        if a.persistence.as_ref().unwrap().status.journaled_revision == rev {
            break;
        }
        assert!(t0.elapsed() < std::time::Duration::from_secs(10), "timed out waiting for the journal");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    a.persistence.take().unwrap().simulate_crash();
}

/// 2026-10-16: the app was killed with unsaved changes, then the headless CLI rebuilt and saved
/// the same project. The next launch's prompt made Recover primary, over the newer file.
#[test]
fn a_project_saved_after_the_auto_save_is_kept_by_default() {
    let d = temp_dir("ui-recover-newer");
    let proj = d.join("Robot Showcase.fcproj").to_string_lossy().into_owned();
    crash_with_unsaved_changes(&d, &proj);
    // the CLI (`--project <p> --save`): open, change, save
    let mut cli = Session::default();
    cli.execute("file.open", json!({"path": proj})).unwrap();
    cli.execute("sequence.addEdit", json!({"seconds": 3.0})).unwrap();
    cli.execute("file.save", json!({})).unwrap();
    let saved = std::fs::read(&proj).unwrap();

    let mut s = Session::default();
    start(&mut s, &d);
    let mut drv = Driver::new(s);
    // asked at launch: a warning, and Open Saved Project as the primary (rightmost) button
    let warning = drv.element("recovery.warning").expect("the prompt warns");
    assert!(warning["label"].as_str().unwrap().contains("newer than this auto-save"), "{warning}");
    let open = drv.element("recovery.open").expect("Open Saved Project");
    let recover = drv.element("recovery.recover").expect("Recover Anyway stays available");
    let later = drv.element("recovery.later").expect("Not Now");
    assert_eq!(recover["label"], "Recover Anyway");
    assert!(x(&open) > x(&recover) && x(&recover) > x(&later), "the primary button is the rightmost: {open} {recover} {later}");
    drv.snapshot("recovery-newer");

    // Open Saved Project: the newer file, untouched; the auto-save kept for later
    drv.click("recovery.open");
    assert!(drv.element("recovery.open").is_none(), "the prompt closed");
    let s = &drv.app().session;
    assert_eq!(s.path.as_deref(), Some(proj.as_str()));
    assert_eq!(*s.project, *cli.project);
    assert!(!s.is_dirty());
    assert_eq!(s.recovery_candidates().len(), 1, "recovery data is never deleted automatically");
    assert_eq!(std::fs::read(&proj).unwrap(), saved);

    // File ▸ Recover Unsaved Changes… asks again, still warning; Recover Anyway recovers
    assert_eq!(drv.menu("file.recover")["dialog"], "recovery");
    assert!(drv.element("recovery.warning").is_some());
    drv.click("recovery.recover");
    assert!(drv.element("recovery.recover").is_none(), "the prompt closed");
    let s = &drv.app().session;
    assert!(s.is_dirty() && s.recovery_candidates().is_empty());
    assert_eq!(s.path.as_deref(), Some(proj.as_str()));
    assert_eq!(std::fs::read(&proj).unwrap(), saved, "nothing is written over the file until the project is saved");
    drop(drv);
    let _ = std::fs::remove_dir_all(&d);
}

/// Nothing saved over the project since the auto-save: the prompt is as it always was.
#[test]
fn an_untouched_project_keeps_recover_as_the_primary_choice() {
    let d = temp_dir("ui-recover-same");
    let proj = d.join("film.fcproj").to_string_lossy().into_owned();
    crash_with_unsaved_changes(&d, &proj);
    let mut s = Session::default();
    start(&mut s, &d);
    let mut drv = Driver::new(s);
    assert!(drv.element("recovery.warning").is_none());
    assert!(drv.element("recovery.open").is_none());
    let recover = drv.element("recovery.recover").expect("Recover");
    let later = drv.element("recovery.later").expect("Not Now");
    assert_eq!(recover["label"], "Recover");
    assert!(x(&recover) > x(&later), "Recover is the primary (rightmost) button");
    drv.snapshot("recovery-unchanged");
    drv.click("recovery.recover");
    let s = &drv.app().session;
    assert!(s.is_dirty() && s.recovery_candidates().is_empty());
    assert_eq!(s.path.as_deref(), Some(proj.as_str()));
    drop(drv);
    let _ = std::fs::remove_dir_all(&d);
}

/// The project file is gone (moved, renamed, deleted): warn, and decide later by default.
#[test]
fn a_missing_project_file_is_not_recovered_by_default() {
    let d = temp_dir("ui-recover-missing");
    let proj = d.join("moved.fcproj").to_string_lossy().into_owned();
    crash_with_unsaved_changes(&d, &proj);
    std::fs::remove_file(&proj).unwrap();
    let mut s = Session::default();
    start(&mut s, &d);
    let mut drv = Driver::new(s);
    let warning = drv.element("recovery.warning").expect("the prompt warns");
    assert!(warning["label"].as_str().unwrap().contains("can't be found"), "{warning}");
    assert!(drv.element("recovery.open").is_none(), "there is no saved project to open");
    let later = drv.element("recovery.later").expect("Not Now");
    let recover = drv.element("recovery.recover").expect("Recover Anyway");
    assert_eq!(recover["label"], "Recover Anyway");
    assert!(x(&later) > x(&recover), "Not Now is the primary (rightmost) button");
    drv.snapshot("recovery-missing");
    drv.click("recovery.later");
    assert!(drv.element("recovery.later").is_none(), "the prompt closed");
    assert_eq!(drv.app().session.recovery_candidates().len(), 1);
    drop(drv);
    let _ = std::fs::remove_dir_all(&d);
}
