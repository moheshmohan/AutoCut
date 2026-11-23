//! `motion.*` with a stand-in RAEA: a shell script the test writes, which copies prepared files
//! (an MP4 made with our own encoder, PNG frames, a check report, a render manifest) to where
//! RAEA would write them and notes how it was called. RAEA itself never runs in the tests.
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, MutexGuard, PoisonError};

use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{DemoScene, Generator};
use serde_json::{Value, json};

use super::*;

struct Bench {
    _turn: MutexGuard<'static, ()>,
    dir: PathBuf,
    s: Session,
}

const CHECK_REPORT: &str = r#"{"file":"motion.tsx","fps":30.0,"findings":[
 {"check":"exposure","level":"warn","about":[],"frames":null,"says":"frame-00000.png: -3.6 stops under mid grey","measure":null,"fix":null,"by":"raea render"},
 {"check":"clipped","level":"error","about":[],"frames":[4],"says":"NEXT STOP runs past the right edge","measure":null,"fix":"wrap it","by":"raea check"}],
 "passed":["looked at frames 4, 9 and 17 of 18","safe areas: no text in the UI zones"]}"#;

/// The stand-in: `check` copies the report, `render` copies the MP4, the frames or a still per
/// `--at` frame (named by position, or by frame number for an HTML page), with the manifest.
fn fake_raea(dir: &Path) -> PathBuf {
    let f = dir.join("fixture");
    let script = format!(
        r#"#!/bin/sh
F="{f}"
printf '%s\n' "$*" >> "{calls}"
cmd="$1"; shift
src="$1"
out=""; alpha=""; at=""; prev=""
for a in "$@"; do
  case "$prev" in -o) out="$a";; --at) at="$a";; esac
  [ "$a" = "--alpha" ] && alpha=1
  prev="$a"
done
case "$cmd" in
  --version) echo "9.9.9-test"; exit 0;;
  check)
    [ -f "$F/check-fails" ] && {{ echo "REFUSED  the check broke" >&2; exit 3; }}
    cp "$F/check.json" "$out"; exit 0;;
  render)
    [ -f "$F/render-fails" ] && {{ echo "REFUSED  the composition does not compile" >&2; exit 2; }}
    if [ -n "$at" ]; then
      mkdir -p "$out"; i=0
      for n in $(echo "$at" | tr ',' ' '); do
        case "$src" in *.html) k=$n;; *) k=$i;; esac
        cp "$F/still.png" "$out/frame-$(printf %05d "$k").png"; i=$((i+1))
      done
      cp "$F/render.json" "$out/motion.render.json"; exit 0
    fi
    if [ -n "$alpha" ]; then
      mkdir -p "$out"; cp "$F"/frames/* "$out"/; cp "$F/render.json" "$out/motion.render.json"; exit 0
    fi
    cp "$F/scene.mp4" "$out"; cp "$F/render.json" "${{out%.mp4}}.render.json"; exit 0;;
esac
echo "unknown verb $cmd" >&2; exit 64
"#,
        f = f.display(),
        calls = dir.join("calls.log").display()
    );
    let path = dir.join("raea");
    std::fs::write(&path, script).unwrap();
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn png(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            px.extend_from_slice(&f(x, y));
        }
    }
    filmcraft_export::encode_png(px, w, h).unwrap()
}

/// The manifest RAEA writes beside a render: its size, rate, length and a text box.
fn render_json(w: u32, h: u32, length: u32) -> String {
    json!({"width": w, "height": h, "fps": 30.0, "length": length, "frames": [
        {"frame": 0, "videos": [], "text": []},
        {"frame": 30, "videos": [], "text": [{"text": "NEXT STOP", "box": [8.0, 40.0, 60.0, 10.0], "size": 10.0, "opacity": 1.0}]}
    ], "sounds": {"clips": []}})
    .to_string()
}

/// Prepared files: a 1 s 30 fps 128×72 MP4 of a moving demo scene, 18 transparent frames with a
/// travelling bar, a still, a check report and a manifest.
fn fixtures(dir: &Path) {
    static MOVIE: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let f = dir.join("fixture");
    std::fs::create_dir_all(f.join("frames")).unwrap();
    let movie = MOVIE.get_or_init(|| {
        let tmp = crate::media_test_util::tmp_dir("motion-fixture");
        let path = tmp.join("scene.mp4");
        let rate = FrameRate::FPS_30;
        let src = GeneratorSource::new(Generator::Demo(DemoScene::Plasma), 128, 72, rate, rate.tick_of(30));
        crate::media_test_util::make_movie_from(&path, Arc::new(src));
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_dir_all(&tmp);
        bytes
    });
    std::fs::write(f.join("scene.mp4"), movie).unwrap();
    for i in 0..18u32 {
        let bytes = png(32, 18, |x, _| if x.abs_diff(i * 2) < 4 { [16, 52, 60, 255] } else { [0, 0, 0, 0] });
        std::fs::write(f.join("frames").join(format!("frame-{i:05}.png")), bytes).unwrap();
    }
    std::fs::write(f.join("still.png"), png(32, 18, |_, _| [200, 100, 50, 255])).unwrap();
    std::fs::write(f.join("check.json"), CHECK_REPORT).unwrap();
    std::fs::write(f.join("render.json"), render_json(128, 72, 30)).unwrap();
}

fn write_providers(body: &str) {
    let manifest = run::manifest_path().unwrap();
    std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    std::fs::write(&manifest, format!(r#"{{"providers":[{body}]}}"#)).unwrap();
}

/// A session with the demo project open, a stand-in RAEA named in providers.json, the fixtures
/// and a composition file.
fn bench(name: &str) -> Bench {
    let turn = run::LIST_TURN.lock().unwrap_or_else(PoisonError::into_inner);
    let dir = std::env::temp_dir().join(format!("fc-motion-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("comp")).unwrap();
    fixtures(&dir);
    let raea = fake_raea(&dir);
    write_providers(&format!(r#"{{"id":"raea","kind":"composition","name":"RAEA","command":["{}"]}}"#, raea.display()));
    std::fs::write(dir.join("comp").join("motion.tsx"), "// a composition; the stand-in never reads it\n").unwrap();
    std::fs::write(dir.join("comp").join("page.html"), "<!-- a page; the stand-in never reads it -->\n").unwrap();
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    Bench { _turn: turn, dir, s }
}

impl Bench {
    fn source(&self) -> String {
        self.dir.join("comp").join("motion.tsx").to_string_lossy().into_owned()
    }
    fn out(&self, name: &str) -> PathBuf {
        self.dir.join("fx").join(name)
    }
    fn calls(&self) -> Vec<String> {
        std::fs::read_to_string(self.dir.join("calls.log")).unwrap_or_default().lines().map(str::to_string).collect()
    }
    fn fixture(&self, name: &str) -> PathBuf {
        self.dir.join("fixture").join(name)
    }
    fn render(&mut self, name: &str, extra: Value) -> Result<Value> {
        let mut p = json!({"source": self.source(), "composition": "scene", "output": self.out(name), "wait": true});
        for (k, v) in extra.as_object().unwrap() {
            p[k] = v.clone();
        }
        self.s.execute("motion.render", p)
    }
}

/// A time on the demo timeline with nothing under it on any track.
fn after_everything(s: &Session) -> Tick {
    let q = s.active_sequence().unwrap();
    let end = q.video_tracks.iter().chain(&q.audio_tracks).flat_map(|t| t.items.iter().map(|i| i.end())).max().unwrap_or(Tick::ZERO);
    let rate = s.sequence_rate().sane();
    rate.tick_of(rate.frame_at(end) + 48)
}

fn read(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn an_mp4_render_is_checked_probed_preflighted_imported_and_placed() {
    let mut b = bench("mp4");
    let (undo0, items0) = (b.s.history.undo.len(), b.s.project.items.len());
    let at = after_everything(&b.s);
    let v = b.render("Next Stop v1", json!({"size": "128x72", "place": {"time": at.0}})).unwrap();
    let out = b.out("Next Stop v1");
    // RAEA's check ran first and its findings came through
    let calls = b.calls();
    let check = calls.iter().position(|c| c.starts_with("check ")).unwrap();
    let render = calls.iter().position(|c| c.starts_with("render ")).unwrap();
    assert!(check < render, "{calls:?}");
    assert!(calls[check].contains("--quick") && calls[check].contains("--composition scene"), "{calls:?}");
    assert!(calls[render].contains("--quality master") && calls[render].contains("--w 128 --h 72 --stage 128x72"), "{calls:?}");
    assert_eq!((v["check"]["errors"].as_u64(), v["check"]["warns"].as_u64()), (Some(1), Some(1)), "{v}");
    assert_eq!(v["check"]["findings"][1]["says"], "NEXT STOP runs past the right edge");
    // the files of the revision, and nothing left of the staging folder
    let asset = out.join("Next Stop v1.mp4");
    for f in ["Next Stop v1.mp4", "Next Stop v1.render.json", "check.json", "preflight.json", "contact-sheet.png", "raea.log", IMPORT_MANIFEST] {
        assert!(out.join(f).is_file(), "{f} in {:?}", std::fs::read_dir(&out).unwrap().flatten().map(|e| e.file_name()).collect::<Vec<_>>());
    }
    assert!(!out.join(".raea-render").exists());
    assert_eq!(v["probe"]["width"], 128);
    assert_eq!(v["probe"]["fps"], 30.0);
    assert_eq!(v["preflight"]["passed"], true, "{v}");
    assert_eq!(v["tool"]["version"], "9.9.9-test");
    // the import manifest has the helper's shape
    let m = read(&out.join(IMPORT_MANIFEST));
    assert_eq!(m["command"], "file.import");
    assert_eq!(m["params"]["paths"][0], asset.to_string_lossy().as_ref());
    assert_eq!(m["reviewOnly"], false);
    assert_eq!(m["cached"], false);
    assert!(m["cacheKey"].is_null());
    assert_eq!(m["composition"], b.source());
    assert_eq!(m["preflight"], out.join("preflight.json").to_string_lossy().as_ref());
    assert!(m["dependencies"].as_array().unwrap().iter().any(|d| d.as_str().unwrap().ends_with("/raea")));
    // imported and placed as one undo step, where it was asked for
    assert_eq!(b.s.project.items.len(), items0 + 1);
    assert_eq!(b.s.history.undo.len(), undo0 + 1, "one undo step");
    let item = ItemId(v["item"].as_u64().unwrap());
    assert!(b.s.project.item(item).unwrap().metadata.get("Log Note").is_some_and(|n| n.contains("motion.tsx (scene)")));
    assert_eq!(v["start"].as_i64(), Some(at.0));
    assert!(v["clip"].as_u64().is_some());
    // the job is in jobs.list with what it made
    let jobs = b.s.execute("jobs.list", json!({})).unwrap();
    let job = jobs.as_array().unwrap().iter().find(|j| j["id"] == v["job"]).unwrap();
    assert_eq!(job["motion"]["item"], v["item"]);
    // the same revision again is refused: an earlier render is never overwritten
    let e = b.render("Next Stop v1", json!({"size": "128x72"})).unwrap_err().to_string();
    assert!(e.contains("already holds files"), "{e}");
    assert_eq!(b.s.project.items.len(), items0 + 1);
    b.s.undo().unwrap();
    assert_eq!(b.s.project.items.len(), items0);
}

#[test]
fn a_transparent_sequence_is_a_job_imported_at_its_rate_above_the_picture() {
    let mut b = bench("alpha");
    std::fs::write(b.fixture("render.json"), render_json(32, 18, 18)).unwrap();
    let at = after_everything(&b.s);
    let p = json!({"source": b.source(), "composition": "wipe", "alpha": true, "output": b.out("Split Wipe v1"), "place": {"time": at.0}});
    let started = b.s.execute("motion.render", p).unwrap();
    assert_eq!(started["kind"], "pngSequence");
    let job = started["job"].clone();
    let t0 = std::time::Instant::now();
    let entry = loop {
        let jobs = b.s.execute("jobs.list", json!({})).unwrap();
        if let Some(j) = jobs.as_array().unwrap().iter().find(|j| j["id"] == job && j.get("motion").is_some()) {
            break j.clone();
        }
        assert!(t0.elapsed().as_secs() < 120, "the job did not finish");
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    let v = &entry["motion"];
    assert!(v.get("error").is_none(), "{v}");
    let out = b.out("Split Wipe v1");
    let first = out.join("frames").join("frame-00000.png");
    let m = read(&out.join(IMPORT_MANIFEST));
    assert_eq!(m["command"], "file.importImageSequence");
    assert_eq!(m["params"], json!({"path": first.to_string_lossy(), "fps": 30.0}), "the rate RAEA rendered at");
    assert_eq!(v["probe"]["frames"], 18);
    assert_eq!(v["probe"]["hasAlpha"], true);
    assert!(b.calls().iter().any(|c| c.starts_with("render ") && c.contains("--alpha") && !c.contains("--quality")), "{:?}", b.calls());
    // over the picture: not on V1
    assert_ne!(v["track"], "V1", "{v}");
    assert!(v["clip"].as_u64().is_some(), "{v}");
}

#[test]
fn the_checks_findings_pass_through() {
    let mut b = bench("check");
    let v = b.s.execute("motion.check", json!({"source": b.source(), "composition": "scene"})).unwrap();
    assert_eq!(v["errors"], 1);
    assert_eq!(v["warns"], 1);
    assert_eq!(v["ok"], false);
    assert_eq!(v["findings"][0]["check"], "exposure");
    assert_eq!(v["findings"][0]["level"], "warn");
    assert_eq!(v["findings"][1]["fix"], "wrap it");
    assert_eq!(v["passed"][0], "looked at frames 4, 9 and 17 of 18");
    assert_eq!(v["tool"]["from"], "providers.json");
    assert!(b.calls().iter().all(|c| !c.starts_with("render")), "a check renders nothing of its own");
    // a check RAEA cannot run says what RAEA said
    std::fs::write(b.fixture("check-fails"), "").unwrap();
    let e = b.s.execute("motion.check", json!({"source": b.source()})).unwrap_err().to_string();
    assert!(e.contains("the check broke"), "{e}");
}

#[test]
fn stills_come_back_with_their_paths_and_text_boxes() {
    let mut b = bench("stills");
    let out = b.out("stills");
    let v = b.s.execute("motion.stills", json!({"source": b.source(), "composition": "scene", "at": [0, 30, 59], "output": out})).unwrap();
    let stills = v["stills"].as_array().unwrap();
    assert_eq!(stills.len(), 3);
    for (i, st) in stills.iter().enumerate() {
        assert_eq!(st["frame"], [0, 30, 59][i]);
        assert!(Path::new(st["path"].as_str().unwrap()).is_file());
    }
    assert!(stills[2]["path"].as_str().unwrap().ends_with("frame-00002.png"), "a TSX composition's stills go by position");
    assert!(v["manifest"].as_str().unwrap().ends_with("motion.render.json"));
    assert_eq!(v["width"], 128);
    assert_eq!(v["text"][1]["frame"], 30);
    assert_eq!(v["text"][1]["text"][0]["text"], "NEXT STOP");
    assert!(!out.join(".raea-stills").exists());
    // an HTML page's stills go by frame number
    let html = b.dir.join("comp").join("page.html");
    let v = b.s.execute("motion.stills", json!({"source": html, "at": [30], "output": b.out("page stills")})).unwrap();
    assert!(v["stills"][0]["path"].as_str().unwrap().ends_with("frame-00030.png"), "{v}");
    // a failed render frees the folder name
    std::fs::write(b.fixture("render-fails"), "").unwrap();
    let e = b.s.execute("motion.stills", json!({"source": b.source(), "at": [0], "output": b.out("failed stills")})).unwrap_err().to_string();
    assert!(e.contains("does not compile"), "{e}");
    assert!(!b.out("failed stills").exists());
}

#[test]
fn a_render_of_the_wrong_size_is_refused_and_kept() {
    let mut b = bench("size");
    let items0 = b.s.project.items.len();
    let e = b.render("Big v1", json!({"size": "192x108"})).unwrap_err().to_string();
    assert!(e.contains("rendered 128x72, not the 192x108") && e.contains("kept"), "{e}");
    let out = b.out("Big v1");
    assert!(out.join("Big v1.mp4").is_file(), "the finished render stays");
    assert!(!out.join(IMPORT_MANIFEST).exists(), "no import manifest without a passed handoff");
    assert_eq!(b.s.project.items.len(), items0, "nothing imported");
    // a rate other than the one asked for is refused the same way
    let e = b.render("Rate v1", json!({"fps": 25})).unwrap_err().to_string();
    assert!(e.contains("not the 25"), "{e}");
    assert!(!b.out("Rate v1").join(IMPORT_MANIFEST).exists());
}

#[test]
fn a_sequence_with_a_gap_is_refused() {
    let mut b = bench("gap");
    std::fs::remove_file(b.fixture("frames").join("frame-00005.png")).unwrap();
    std::fs::write(b.fixture("render.json"), render_json(32, 18, 18)).unwrap();
    let e = b.render("Wipe v1", json!({"composition": "wipe", "alpha": true, "fps": 30})).unwrap_err().to_string();
    assert!(e.contains("gaps") && e.contains("frame-00005.png") && e.contains("kept"), "{e}");
    let out = b.out("Wipe v1");
    assert!(out.join("frames").join("frame-00017.png").is_file());
    assert!(!out.join(IMPORT_MANIFEST).exists());
    // a sequence shorter than RAEA's manifest says is refused too
    std::fs::write(b.fixture("frames").join("frame-00005.png"), png(32, 18, |_, _| [0, 0, 0, 0])).unwrap();
    std::fs::write(b.fixture("render.json"), render_json(32, 18, 20)).unwrap();
    let e = b.render("Wipe v2", json!({"composition": "wipe", "alpha": true})).unwrap_err().to_string();
    assert!(e.contains("18 frames") && e.contains("says 20"), "{e}");
    // and a rate that differs from the one RAEA rendered at
    std::fs::write(b.fixture("render.json"), render_json(32, 18, 18)).unwrap();
    let e = b.render("Wipe v3", json!({"composition": "wipe", "alpha": true, "fps": 24})).unwrap_err().to_string();
    assert!(e.contains("24") && e.contains("30"), "{e}");
    let v = b.render("Wipe v4", json!({"composition": "wipe", "alpha": true, "import": false})).unwrap();
    assert!(v["item"].is_null() && b.out("Wipe v4").join(IMPORT_MANIFEST).is_file(), "{v}");
}

#[test]
fn a_failed_preflight_keeps_the_render_and_writes_no_manifest() {
    let mut b = bench("preflight");
    let e = b
        .render("Long v1", json!({"expected": {"duration": Tick::from_seconds_f64(10.0).0}, "holds": [{"start": 0, "end": Tick::from_seconds_f64(0.5).0, "reason": "the title settles"}]}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("preflight failed (duration") && e.contains("kept"), "{e}");
    let out = b.out("Long v1");
    let qa = read(&out.join("preflight.json"));
    assert_eq!(qa["passed"], false);
    assert!(qa["holds"].is_array(), "the declared holds reached the preflight: {qa}");
    assert!(out.join("Long v1.mp4").is_file());
    assert!(!out.join(IMPORT_MANIFEST).exists());
    // the same QA from a spec file (the helper's --spec), with a preview: a draft, review only
    let spec = b.dir.join("scene-qa.json");
    std::fs::write(
        &spec,
        json!({"expected": {"duration": Tick::from_seconds_f64(1.0).0}, "holds": [{"start": 0, "end": Tick::from_seconds_f64(0.5).0, "reason": "settled"}]})
            .to_string(),
    )
    .unwrap();
    let v = b.render("Review v1", json!({"spec": spec, "canvas": "128x72", "preview": true, "import": false})).unwrap();
    assert_eq!(v["preflight"]["passed"], true, "{v}");
    let m = read(&b.out("Review v1").join(IMPORT_MANIFEST));
    assert_eq!(m["reviewOnly"], true);
    assert!(m["dependencies"].as_array().unwrap().iter().any(|d| d == &json!(spec.to_string_lossy())));
    assert!(b.calls().iter().any(|c| c.contains("--quality draft") && c.contains("--stage 128x72")), "{:?}", b.calls());
}

#[test]
fn a_failed_render_frees_its_revision_name() {
    let mut b = bench("failed");
    std::fs::write(b.fixture("render-fails"), "").unwrap();
    let e = b.render("Broken v1", json!({})).unwrap_err().to_string();
    assert!(e.contains("does not compile") && !e.contains("kept"), "{e}");
    assert!(!b.out("Broken v1").exists(), "an empty revision folder would block the same name next time");
    // a folder that was there (empty) stays, empty
    std::fs::create_dir_all(b.out("Broken v2")).unwrap();
    assert!(b.render("Broken v2", json!({})).is_err());
    assert!(b.out("Broken v2").is_dir() && std::fs::read_dir(b.out("Broken v2")).unwrap().next().is_none());
    // and with RAEA working again the same name renders
    std::fs::remove_file(b.fixture("render-fails")).unwrap();
    b.render("Broken v1", json!({"import": false})).unwrap();
}

#[test]
fn a_missing_raea_says_how_to_install_it() {
    let mut b = bench("missing");
    write_providers(&format!(r#"{{"id":"raea","kind":"composition","name":"RAEA","command":["{}"]}}"#, b.dir.join("nowhere").join("raea").display()));
    for (cmd, p) in [
        ("motion.check", json!({"source": b.source()})),
        ("motion.render", json!({"source": b.source(), "output": b.out("x")})),
        ("motion.stills", json!({"source": b.source(), "at": [0], "output": b.out("y")})),
    ] {
        let e = b.s.execute(cmd, p).unwrap_err().to_string();
        assert!(e.contains("not on this computer") && e.contains("docs/raea-workflow.md"), "{cmd}: {e}");
    }
    assert!(!b.out("x").exists() && !b.out("y").exists());
    let listed = b.s.execute("generate.providers", json!({})).unwrap();
    let entry = listed["providers"].as_array().unwrap().iter().find(|p| p["id"] == "raea").unwrap().clone();
    assert_eq!((entry["kind"].as_str(), entry["installed"].as_bool()), (Some("composition"), Some(false)));
    // a broken list is said, not ignored
    write_providers(r#"{"id":"raea","kind":"composition","name":"RAEA","command":["raea"]}"#);
    let e = b.s.execute("motion.check", json!({"source": b.source()})).unwrap_err().to_string();
    assert!(e.contains("absolute"), "{e}");
}

#[test]
fn raea_is_found_in_its_versions_folder_then_on_path() {
    let root = crate::media_test_util::tmp_dir("motion-locate");
    let home = root.join("home");
    let none = locate_in(None, Some(&home), Some(OsStr::new(""))).unwrap_err();
    assert!(none.contains("RAEA is not installed") && none.contains("docs/raea-workflow.md"), "{none}");
    let put = |p: &Path| {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, "#!/bin/sh\n").unwrap();
    };
    // the newest version folder wins (by number, not by name), odd folders are passed over
    for v in ["0.1.0", "0.1.8", "0.1.10", "latest", "0.2..1"] {
        put(&home.join(".raea/versions").join(v).join("bin/raea"));
    }
    std::fs::create_dir_all(home.join(".raea/versions/9.0.0/bin")).unwrap(); // no program in it
    let t = locate_in(None, Some(&home), None).unwrap();
    assert!(t.program.ends_with("0.1.10/bin/raea"), "{:?}", t.program);
    assert_eq!(t.from, "~/.raea/versions");
    // else PATH (absolute folders only)
    let bin = root.join("bin");
    put(&bin.join("raea"));
    let path = std::env::join_paths([PathBuf::from("relative/bin"), bin.clone()]).unwrap();
    let t = locate_in(None, Some(&root.join("empty-home")), Some(&path)).unwrap();
    assert_eq!((t.program, t.from), (bin.join("raea"), "PATH"));
    // an entry with a program wins, one without leaves the finding to the folders
    let entry = run::parse(&format!(
        r#"{{"providers":[{{"id":"raea","kind":"composition","name":"RAEA","command":["{}"],"timeoutSeconds":60,"env":{{"RAEA_QUIET":"1"}}}}]}}"#,
        bin.join("raea").display()
    ))
    .unwrap()
    .remove(0);
    let t = locate_in(Some(&entry), Some(&home), None).unwrap();
    assert_eq!((t.from, t.timeout, t.env.get("RAEA_QUIET").map(String::as_str)), ("providers.json", Some(60.0), Some("1")));
    let bare = run::parse(r#"{"providers":[{"id":"raea","kind":"composition","name":"RAEA"}]}"#).unwrap().remove(0);
    assert!(locate_in(Some(&bare), Some(&home), None).unwrap().program.ends_with("0.1.10/bin/raea"));
    assert_eq!(parse_version("0.1.8"), Some(vec![0, 1, 8]));
    assert_eq!(parse_version("1.2.3.4.5"), None);
    assert_eq!(parse_version("99999999999"), None);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn hostile_parameters_are_refused_before_anything_runs() {
    let mut b = bench("hostile");
    let src = b.source();
    let out = b.out("h").to_string_lossy().into_owned();
    let long = format!("/{}", "a".repeat(5000));
    let missing = b.dir.join("comp").join("gone.tsx").to_string_lossy().into_owned();
    let text = b.dir.join("comp").join("notes.txt");
    std::fs::write(&text, "x").unwrap();
    let cases: Vec<(&str, Value, &str)> = vec![
        ("motion.render", json!({"output": out}), "`source` is needed"),
        ("motion.render", json!({"source": "relative.tsx", "output": out}), "absolute"),
        ("motion.render", json!({"source": long, "output": out}), "absolute"),
        ("motion.render", json!({"source": "/a\u{0}b.tsx", "output": out}), "absolute"),
        ("motion.render", json!({"source": 7, "output": out}), "absolute"),
        ("motion.render", json!({"source": text, "output": out}), ".tsx or .html"),
        ("motion.render", json!({"source": missing, "output": out}), "not a file"),
        ("motion.render", json!({"source": src}), "`output` is needed"),
        ("motion.render", json!({"source": src, "output": "fx/v1"}), "absolute"),
        ("motion.render", json!({"source": src, "output": out, "composition": "--alpha"}), "composition id"),
        ("motion.render", json!({"source": src, "output": out, "composition": "a b"}), "composition id"),
        ("motion.render", json!({"source": src, "output": out, "composition": "x".repeat(200)}), "composition id"),
        ("motion.render", json!({"source": src, "output": out, "canvas": "0x540"}), "WxH"),
        ("motion.render", json!({"source": src, "output": out, "size": "99999x2"}), "WxH"),
        ("motion.render", json!({"source": src, "output": out, "size": "axb"}), "WxH"),
        ("motion.render", json!({"source": src, "output": out, "size": "-1x5"}), "WxH"),
        ("motion.render", json!({"source": src, "output": out, "size": [1920, 1080]}), "WxH"),
        ("motion.render", json!({"source": src, "output": out, "preview": true}), "needs `canvas`"),
        ("motion.render", json!({"source": src, "output": out, "preview": "yes"}), "true or false"),
        ("motion.render", json!({"source": b.dir.join("comp/page.html"), "output": out, "preview": true, "canvas": "960x540"}), "HTML"),
        ("motion.render", json!({"source": src, "output": out, "fps": 0}), "1 to 240"),
        ("motion.render", json!({"source": src, "output": out, "fps": 1e300}), "1 to 240"),
        ("motion.render", json!({"source": src, "output": out, "fps": "30"}), "1 to 240"),
        ("motion.render", json!({"source": src, "output": out, "alpha": 1}), "true or false"),
        ("motion.render", json!({"source": src, "output": out, "quality": "ultra"}), "draft, high or master"),
        ("motion.render", json!({"source": src, "output": out, "alpha": true, "quality": "master"}), "lossless"),
        ("motion.render", json!({"source": src, "output": out, "expected": []}), "JSON object"),
        ("motion.render", json!({"source": src, "output": out, "expected": {"duration": -5}}), "expected.duration"),
        ("motion.render", json!({"source": src, "output": out, "expected": {"width": 0}}), "expected.width"),
        ("motion.render", json!({"source": src, "output": out, "expected": {"fps": -1}}), "expected.fps"),
        ("motion.render", json!({"source": src, "output": out, "holds": "all"}), "intervals"),
        ("motion.render", json!({"source": src, "output": out, "holds": [{"start": 0, "end": 10}]}), "reason"),
        ("motion.render", json!({"source": src, "output": out, "holds": [{"start": 10, "end": 5, "reason": "x"}]}), "interval end"),
        ("motion.render", json!({"source": src, "output": out, "textBounds": {"a": 1}}), "at most 2000"),
        ("motion.render", json!({"source": src, "output": out, "spec": "spec.json"}), "absolute"),
        ("motion.render", json!({"source": src, "output": out, "spec": b.dir.join("no-spec.json")}), "no-spec.json"),
        ("motion.render", json!({"source": src, "output": out, "place": {"track": "A1"}}), "video track"),
        ("motion.render", json!({"source": src, "output": out, "place": {"track": "V99999999"}}), "video track"),
        ("motion.render", json!({"source": src, "output": out, "place": {"seconds": -3}}), "0 to 24 hours"),
        ("motion.render", json!({"source": src, "output": out, "place": "V2"}), "`place` is"),
        ("motion.render", json!({"source": src, "output": out, "place": true, "import": false}), "imported"),
        ("motion.render", json!({"source": src, "output": out, "wait": "soon"}), "true or false"),
        ("motion.stills", json!({"source": src, "output": out}), "`at` is needed"),
        ("motion.stills", json!({"source": src, "output": out, "at": []}), "1 to 64"),
        ("motion.stills", json!({"source": src, "output": out, "at": (0..65).collect::<Vec<u32>>()}), "1 to 64"),
        ("motion.stills", json!({"source": src, "output": out, "at": [-1]}), "frame numbers"),
        ("motion.stills", json!({"source": src, "output": out, "at": [1.5]}), "frame numbers"),
        ("motion.stills", json!({"source": src, "output": out, "at": [u64::MAX]}), "frame numbers"),
        ("motion.stills", json!({"source": src, "output": out, "at": [3, 3]}), "twice"),
        ("motion.stills", json!({"source": src, "at": [0]}), "`output` is needed"),
        ("motion.check", json!({}), "`source` is needed"),
        ("motion.check", json!({"source": src, "composition": 5}), "composition id"),
    ];
    for (cmd, p, why) in cases {
        let e = b.s.execute(cmd, p.clone()).unwrap_err().to_string();
        assert!(e.contains(why), "{cmd} {p}: {e}");
    }
    // a file where the revision folder would go, and a spec that is not an object
    std::fs::create_dir_all(b.dir.join("fx")).unwrap();
    std::fs::write(b.out("file"), "x").unwrap();
    let e = b.render("file", json!({})).unwrap_err().to_string();
    assert!(e.contains("not a folder"), "{e}");
    let spec = b.dir.join("list.json");
    std::fs::write(&spec, "[1,2]").unwrap();
    let e = b.render("s", json!({"spec": spec})).unwrap_err().to_string();
    assert!(e.contains("JSON object"), "{e}");
    let without = {
        let mut s = Session::default();
        s.execute("motion.render", json!({"source": src, "output": out, "place": {}})).unwrap_err().to_string()
    };
    assert!(without.contains("open sequence"), "{without}");
    assert!(b.calls().is_empty(), "nothing ran: {:?}", b.calls());
    assert!(!b.out("h").exists(), "no folder was made");
}

#[test]
fn the_preview_stage_keeps_the_shape_with_a_long_side_of_960() {
    assert_eq!(stage_of("t", Some((960, 540)), Some((3840, 2160)), true).unwrap(), Some((960, 540)));
    assert_eq!(stage_of("t", Some((1080, 1920)), None, true).unwrap(), Some((540, 960)));
    assert_eq!(stage_of("t", Some((640, 360)), None, true).unwrap(), Some((640, 360)), "never larger");
    assert_eq!(stage_of("t", Some((960, 540)), Some((3840, 2160)), false).unwrap(), Some((3840, 2160)));
    assert_eq!(stage_of("t", None, None, false).unwrap(), None);
    assert!(stage_of("t", None, Some((3840, 2160)), true).is_err());
    assert_eq!(stage_of("t", Some((16384, 1)), None, true).unwrap(), Some((960, 2)));
}

#[test]
fn reports_and_manifests_from_raea_are_bounded() {
    let dir = crate::media_test_util::tmp_dir("motion-bounds");
    let many: Vec<Value> = (0..2000).map(|i| json!({"check": "clipped", "level": if i % 2 == 0 { "error" } else { "note" }, "says": "x"})).collect();
    std::fs::write(dir.join("check.json"), json!({"findings": many}).to_string()).unwrap();
    let v = check_summary(&dir.join("check.json")).unwrap();
    assert_eq!((v["errors"].as_u64(), v["notes"].as_u64(), v["truncated"].as_bool()), (Some(1000), Some(1000), Some(true)));
    assert_eq!(v["findings"].as_array().unwrap().len(), MAX_FINDINGS);
    std::fs::write(dir.join("bad.json"), "{\"findings\": 3}").unwrap();
    assert!(check_summary(&dir.join("bad.json")).is_err());
    std::fs::write(dir.join("big.json"), " ".repeat(CHECK_JSON_MAX as usize + 10)).unwrap();
    assert!(check_summary(&dir.join("big.json")).unwrap_err().contains("larger"));
    let frames: Vec<Value> = (0..30).map(|f| json!({"frame": f, "text": vec![json!({"text": "t"}); 1000]})).collect();
    let (text, cut) = text_boxes(&json!({"frames": frames}));
    assert!(cut);
    assert_eq!(text.iter().map(|f| f["text"].as_array().unwrap().len()).sum::<usize>(), MAX_TEXTS);
    let st = stated(&json!({"width": 1e12, "height": -4, "fps": f64::MAX, "length": "60"}));
    assert!(st.width.is_none() && st.height.is_none() && st.fps.is_none() && st.length.is_none());
    let _ = std::fs::remove_dir_all(&dir);
}
