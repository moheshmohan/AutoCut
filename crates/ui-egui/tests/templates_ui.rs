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
    /// The last status lines, oldest first: the bar clears a line after 8 s, which is 32 frames
    /// here, so by the time a wait gives up the line itself is empty.
    statuses: Vec<String>,
}

impl Driver {
    fn demo(scratch: &str) -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).expect("demo project");
        session.execute("file.projectSettings.scratchDisks", json!({"captured": scratch})).expect("scratch disk");
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(session).with_control(rx);
        // FILMCRAFT_UI_SNAPSHOT_DIR: render with wgpu so `snapshot` can save what the test sees
        let mut b = Harness::builder().with_size(egui::vec2(1600.0, 1200.0)).with_max_steps(10_000);
        if std::env::var_os("FILMCRAFT_UI_SNAPSHOT_DIR").is_some() {
            b = b.wgpu();
        }
        let harness = b.build_eframe(move |_cc| app);
        let mut d = Driver { harness, tx, statuses: Vec::new() };
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
            let status = &self.harness.state().ui.status;
            if !status.is_empty() && self.statuses.last() != Some(status) {
                let status = status.clone();
                if self.statuses.len() >= 8 {
                    self.statuses.remove(0);
                }
                self.statuses.push(status);
            }
        }
    }

    fn app(&mut self) -> &mut FilmcraftApp {
        self.harness.state_mut()
    }

    /// Save the window as `<FILMCRAFT_UI_SNAPSHOT_DIR>/<name>.png` (nothing without the variable).
    fn snapshot(&mut self, name: &str) {
        let Some(dir) = std::env::var_os("FILMCRAFT_UI_SNAPSHOT_DIR").map(std::path::PathBuf::from) else { return };
        self.frames(2);
        match self.harness.render() {
            Ok(img) => {
                std::fs::create_dir_all(&dir).unwrap();
                img.save(dir.join(format!("{name}.png"))).unwrap();
            }
            Err(e) => eprintln!("snapshot {name} skipped: {e}"),
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
        // a deadline rather than a number of tries, for a slow machine (what is waited for is drawn
        // in the frames themselves: the preview renders in the frame Preview is clicked in)
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while std::time::Instant::now() < deadline {
            if self.ids(id).iter().any(|x| x == id) {
                return;
            }
            // just above the footer's Preview button: inside the scrolling slots
            let v = self.ok("ui.elements", json!({"prefix": "templates.preview"}));
            let r = v.as_array().unwrap().iter().find(|e| e["id"] == "templates.preview").unwrap()["rect"].clone();
            let (x, y) = (r[0].as_f64().unwrap() + r[2].as_f64().unwrap() / 2.0, r[1].as_f64().unwrap() - 60.0);
            self.ok("ui.scroll", json!({"x": x, "y": y, "dx": 0.0, "dy": dy}));
            self.frames(3);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let shown = self.ids("templates.");
        panic!("{id} never showed (status lines: {:?}): {shown:?}", self.statuses);
    }

    /// Frames until widget `id` is registered.
    fn wait_for(&mut self, id: &str) {
        for _ in 0..200 {
            if self.ids(id).iter().any(|x| x == id) {
                return;
            }
            self.frames(2);
        }
        let shown = self.ids("templates.");
        panic!("{id} never showed (status lines: {:?}): {shown:?}", self.statuses);
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
    assert_eq!(d.app().ui.status, "Preview: 6 frames");
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

/// A preview stays while its template reads the same, when a template is saved or installed
/// elsewhere (another window, an agent; the library's revision is the process's), and goes when its
/// own template changes. It went on any install: `browse_filter_fill_preview_and_build_a_template`
/// failed whenever `save_share_and_install_from_the_panel`, beside it in this binary, installed a
/// template between its Preview click and its look at the frames (8 runs in 20 on a loaded machine).
#[test]
fn a_preview_stays_while_its_template_does() {
    let scratch = std::env::temp_dir().join(format!("filmcraft-templates-ui-preview-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    // the quote card as a package of its own
    let package = |id: &str, description: &str| {
        let mut pkg = filmcraft_templates::builtin::find("builtin:quote-card").unwrap().clone();
        pkg.manifest.id = id.into();
        pkg.manifest.name = id.into();
        pkg.manifest.description = description.into();
        let path = scratch.join(format!("{}.zip", id.replace(':', "-")));
        std::fs::write(&path, pkg.to_zip().unwrap()).unwrap();
        path.to_string_lossy().into_owned()
    };
    // another window installs a template into its own folder
    let install_elsewhere = |id: &str| {
        let before = filmcraft_engine::video_templates::library_revision();
        let mut other = Session::default();
        other.prefs_path = Some(scratch.join("elsewhere").join("prefs.json"));
        other.execute("template.install", json!({"path": package(id, "")})).unwrap();
        assert!(filmcraft_engine::video_templates::library_revision() > before);
    };
    let mut d = Driver::demo(&scratch.to_string_lossy());
    d.app().session.prefs_path = Some(scratch.join("prefs.json"));
    d.ok("ui.panel.show", json!({"panel": "Templates"}));
    d.frames(4);
    // a built-in template's preview
    d.ok("ui.set", json!({"templates": {"selected": "builtin:quote-card"}}));
    d.wait_for("templates.preview");
    d.click("templates.preview");
    assert_eq!(d.app().ui.status, "Preview: 6 frames");
    d.reveal_by("templates.previewFrame.0", -160.0);
    install_elsewhere("user:elsewhere-one");
    d.frames(4);
    assert_eq!(d.ids("templates.previewFrame.").len(), 6, "the preview stays when another template is installed");
    // a template of the user's own: its preview stays while it reads the same
    d.app().session.execute("template.install", json!({"path": package("user:my-quote", "Mine")})).unwrap();
    d.ok("ui.set", json!({"templates": {"selected": "user:my-quote"}}));
    d.frames(4);
    d.click("templates.preview");
    assert_eq!(d.app().ui.status, "Preview: 6 frames");
    d.reveal_by("templates.previewFrame.0", -160.0);
    install_elsewhere("user:elsewhere-two");
    d.frames(4);
    assert_eq!(d.ids("templates.previewFrame.").len(), 6, "the preview of an unchanged user template stays");
    // … and goes when the template is replaced: those frames show it as it was
    let changed = package("user:my-quote", "Mine, changed");
    d.app().session.execute("template.install", json!({"path": changed, "overwrite": true})).unwrap();
    d.frames(4);
    assert_eq!(d.app().ui.templates.selected.as_deref(), Some("user:my-quote"));
    assert!(d.ids("templates.previewFrame.").is_empty(), "a stale preview is not shown");
    d.wait_for("templates.preview");
    let _ = std::fs::remove_dir_all(&scratch);
}

/// A3 from the panel: mark the demo's clips as slots (the save form and the Timeline's clip menu),
/// save the sequence as a template into the user folder (it shows in the grid), export it as a
/// .zip, install that .zip again under a new id, and show the QR code of its link.
#[test]
fn save_share_and_install_from_the_panel() {
    let scratch = std::env::temp_dir().join(format!("filmcraft-templates-ui-author-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let mut d = Driver::demo(&scratch.to_string_lossy());
    d.app().session.prefs_path = Some(scratch.join("prefs.json"));
    d.ok("ui.panel.show", json!({"panel": "Templates"}));
    d.frames(4);
    // Sequence ▸ Save as Video Template… opens the form (the menu's form: `engine.execute` never
    // opens one and runs the engine command, which needs a path)
    d.ok("ui.menu.invoke", json!({"id": "template.save"}));
    d.frames(3);
    assert!(d.app().ui.templates.save.is_some());
    d.wait_for("templates.save.go");
    for id in [
        "templates.save.name",
        "templates.save.author",
        "templates.save.license",
        "templates.save.mark.video",
        "templates.save.captions",
        "templates.save.variant.1:1",
    ] {
        d.wait_for(id);
    }
    // select two clips and mark them as the `shots` slot
    let clips: Vec<u64> = d.app().session.active_sequence().unwrap().video_tracks[0].items.iter().take(2).map(|c| c.id.0).collect();
    d.ok("engine.execute", json!({"id": "timeline.select", "params": {"clips": clips}}));
    d.ok("ui.set", json!({"templates": {"save": {"slot": "shots", "name": "Demo Kit", "author": "Tester", "variants": ["9:16", "1:1"]}}}));
    d.frames(2);
    d.click("templates.save.mark.video");
    assert!(d.app().ui.status.contains("shots"), "{}", d.app().ui.status);
    d.wait_for("templates.save.slot.shots.name");
    d.ok("ui.set", json!({"templates": {"save": {"slotNames": {"shots": "Your shots"}}}}));
    d.frames(2);
    d.click("templates.save.go");
    d.frames(4);
    let status = d.app().ui.status.clone();
    assert!(status.contains("Demo Kit"), "{status}");
    assert!(d.app().ui.templates.save.is_none());
    assert_eq!(d.app().ui.templates.selected.as_deref(), Some("user:demo-kit"), "the saved template opens");
    {
        let e = filmcraft_engine::video_templates::find_template(&d.app().session, "user:demo-kit", "t").unwrap();
        assert_eq!(e.source, "user");
        let m = &e.package.manifest;
        assert_eq!(m.slot("shots").unwrap().name, "Your shots");
        assert_eq!(m.variants.iter().map(|v| v.id.as_str()).collect::<Vec<_>>(), ["16:9", "9:16", "1:1"]);
        assert_eq!(m.author, "Tester");
    }
    // the Share section: export a .zip, show the QR code of a link
    d.reveal_by("templates.share.exportPath", -160.0);
    let zip = scratch.join("demo-kit-shared.zip");
    d.ok("ui.set", json!({"templates": {"exportPath": zip.to_string_lossy(), "link": "https://example.org/templates/demo-kit.zip"}}));
    d.frames(2);
    d.reveal_by("templates.share.export", -160.0);
    d.click("templates.share.export");
    assert!(zip.exists(), "{}", d.app().ui.status);
    d.reveal_by("templates.share.qr", -160.0);
    // back in the grid: the saved template is a card; install the shared copy under another id
    d.reveal_by("templates.back", 400.0);
    d.click("templates.back");
    // (the user's templates come after the starters: search brings them into view)
    d.ok("ui.set", json!({"templates": {"query": "kit"}}));
    d.frames(3);
    d.wait_for("templates.card.user:demo-kit");
    let mut pkg = filmcraft_templates::Package::from_zip(&std::fs::read(&zip).unwrap()).unwrap();
    pkg.manifest.id = "user:friends-kit".into();
    pkg.manifest.name = "Friend's Kit".into();
    let theirs = scratch.join("friends-kit.zip");
    std::fs::write(&theirs, pkg.to_zip().unwrap()).unwrap();
    d.ok("ui.set", json!({"templates": {"install": theirs.to_string_lossy()}}));
    d.frames(2);
    d.click("templates.install.go");
    assert!(d.app().ui.status.contains("Friend's Kit"), "{}", d.app().ui.status);
    d.wait_for("templates.card.user:friends-kit");
    // a hostile file is refused with a message
    let junk = scratch.join("junk.zip");
    std::fs::write(&junk, b"PK\x03\x04 nope").unwrap();
    d.ok("ui.set", json!({"templates": {"install": junk.to_string_lossy()}}));
    d.frames(2);
    d.click("templates.install.go");
    assert!(d.app().ui.status.starts_with("Install"), "{}", d.app().ui.status);
    // the Timeline's clip menu flags clips too
    d.ok("engine.execute", json!({"id": "template.markSlot", "params": {"clips": [clips[0]], "clear": true}}));
    let q = d.app().session.active_sequence().unwrap().clone();
    assert!(q.video_tracks[0].items[0].markers.iter().all(|m| !m.name.starts_with("slot:")));
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

/// Records the jobs it is asked to run (instead of AutoCut's host).
#[derive(Default)]
struct Recorder(std::sync::Mutex<Vec<Value>>);

impl filmcraft_engine::autocut::AutoCutHost for Recorder {
    fn run(
        &self,
        task: filmcraft_engine::autocut::Task,
        _port: &filmcraft_engine::autocut::CommandPort,
        _progress: &std::sync::Arc<filmcraft_engine::export::Progress>,
    ) -> Result<Value, String> {
        self.0.lock().unwrap().push(json!({"command": task.kind.command(), "params": task.params}));
        Ok(json!({"status": "Quick short made"}))
    }
}

/// Quick Short… from the Templates panel: every widget of the dialog is agent-drivable, its
/// draft is UI state, Make Short starts the `autocut.quick` job with the choices, and a folder
/// source without a folder is refused in the dialog.
#[test]
fn quick_short_dialog_starts_the_job() {
    let scratch = std::env::temp_dir().join(format!("filmcraft-quick-ui-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let mut d = Driver::demo(&scratch.to_string_lossy());
    let host = std::sync::Arc::new(Recorder::default());
    d.app().session.autocut.host = Some(host.clone());
    d.ok("ui.panel.show", json!({"panel": "Templates"}));
    d.frames(6);
    d.wait_for("templates.quickShort");
    d.click("templates.quickShort");
    d.wait_for("quickShort.go");
    for id in [
        "quickShort.source.selection",
        "quickShort.source.folder",
        "quickShort.template",
        "quickShort.seconds",
        "quickShort.platform",
        "quickShort.aspect",
        "quickShort.topic",
        "quickShort.look",
        "quickShort.cancel",
    ] {
        assert!(d.ids(id).iter().any(|x| x == id), "{id}: {:?}", d.ids("quickShort."));
    }
    // the template and the look by clicks, the rest as UI state
    d.click("quickShort.template");
    d.wait_for("quickShort.template.builtin:tech-reveal");
    d.click("quickShort.template.builtin:tech-reveal");
    d.click("quickShort.look");
    d.wait_for("quickShort.look.vhs");
    d.click("quickShort.look.vhs");
    d.ok("ui.set", json!({"quickShort": {"source": "selection", "topic": "robots", "seconds": 20, "platform": "shorts"}}));
    d.frames(2);
    d.snapshot("quick-short-dialog");
    let draft = d.app().ui.quick_short.clone().unwrap();
    assert_eq!((draft.template.as_str(), draft.look.as_str(), draft.topic.as_str()), ("builtin:tech-reveal", "vhs", "robots"));
    d.click("quickShort.go");
    for _ in 0..200 {
        if !host.0.lock().unwrap().is_empty() {
            break;
        }
        d.frames(2);
    }
    let jobs = host.0.lock().unwrap().clone();
    assert_eq!(jobs.len(), 1, "{jobs:?}");
    let p = &jobs[0]["params"];
    assert_eq!(jobs[0]["command"], "autocut.quick");
    assert_eq!((p["template"].as_str(), p["look"].as_str(), p["topic"].as_str()), (Some("builtin:tech-reveal"), Some("vhs"), Some("robots")));
    assert_eq!((p["seconds"].as_f64(), p["platform"].as_str(), p["aspect"].as_str()), (Some(20.0), Some("youtube"), Some("9:16")));
    assert!(d.app().ui.quick_short.is_none(), "the dialog closes when the job starts");
    // a folder source needs a folder: refused in the dialog, nothing started
    d.ok("ui.set", json!({"quickShort": {"source": "folder", "folder": ""}}));
    d.frames(2);
    d.wait_for("quickShort.go");
    d.click("quickShort.go");
    d.frames(4);
    assert!(d.app().ui.quick_short.as_ref().is_some_and(|q| !q.error.is_empty()), "{:?}", d.app().ui.quick_short);
    assert_eq!(host.0.lock().unwrap().len(), 1);
    d.ok("ui.set", json!({"quickShort": null}));
    d.frames(2);
    assert!(d.app().ui.quick_short.is_none());
    assert!(d.call("ui.set", json!({"quickShort": 5}))["ok"] != json!(true), "hostile UI state is refused");
    let _ = std::fs::remove_dir_all(&scratch);
}
