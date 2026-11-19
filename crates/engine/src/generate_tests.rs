//! `generate.*` with stand-in tools: shell one-liners that copy a prepared file to where the
//! result should go and note the arguments they were given. No model runs in the tests.
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use serde_json::{Value, json};

use super::*;

/// The provider list is one file per test binary: tests that write it take turns.
static TURN: Mutex<()> = Mutex::new(());

struct Bench {
    _turn: MutexGuard<'static, ()>,
    dir: PathBuf,
    s: Session,
}

fn png(w: u32, h: u32, shade: u8) -> Vec<u8> {
    filmcraft_export::encode_png(vec![shade; (w * h * 4) as usize], w, h).unwrap()
}

/// Half a second of a 440 Hz tone, 16-bit mono at 48 kHz.
fn wav() -> Vec<u8> {
    let n = 24_000u32;
    let mut b = Vec::new();
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + n * 2).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&48_000u32.to_le_bytes());
    b.extend_from_slice(&96_000u32.to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&(n * 2).to_le_bytes());
    for i in 0..n {
        let v = ((i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 8000.0) as i16;
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

/// A stand-in tool: copies `fixture` to `{out}` and writes its arguments beside it.
fn tool(id: &str, kind: &str, fixture: &Path, more_args: &str, extra: &str) -> String {
    let fixture = fixture.to_string_lossy();
    format!(
        r#"{{"id":"{id}","kind":"{kind}","name":"Stand-in {id}","command":["/bin/sh","-c","out=\"$2\"; printf '%s\\n' \"$@\" > \"$out.args\"; cp \"$1\" \"$out\"","sh","{fixture}","{{out}}","--prompt","{{prompt}}","--seed","{{seed}}"{more_args}],"attribution":"Stand-in model","license":"test"{extra}}}"#
    )
}

fn bench(name: &str, providers: &[String]) -> Bench {
    let turn = TURN.lock().unwrap_or_else(PoisonError::into_inner);
    let dir = std::env::temp_dir().join(format!("fc-generate-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = run::manifest_path().unwrap();
    std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    std::fs::write(&manifest, format!(r#"{{"providers":[{}]}}"#, providers.join(","))).unwrap();
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    Bench { _turn: turn, dir, s }
}

impl Bench {
    fn out(&self) -> String {
        self.dir.join("made").to_string_lossy().into_owned()
    }
    fn args_of(&self, path: &str) -> Vec<String> {
        std::fs::read_to_string(format!("{path}.args")).unwrap().lines().map(str::to_string).collect()
    }
}

fn made(v: &Value, i: usize) -> &Value {
    &v["generated"][i]
}

/// `secs` as whole frames of the demo sequence (the timeline places on frames).
fn sec(s: &Session, secs: f64) -> Tick {
    let rate = s.sequence_rate().sane();
    rate.tick_of((secs * rate.as_f64()).round() as i64)
}

/// A time on the demo timeline with nothing under it on any track.
fn after_everything(s: &Session) -> Tick {
    let q = s.active_sequence().unwrap();
    let end = q.video_tracks.iter().chain(&q.audio_tracks).flat_map(|t| t.items.iter().map(|i| i.end())).max().unwrap_or(Tick::ZERO);
    let rate = s.sequence_rate().sane();
    rate.tick_of(rate.frame_at(end) + 48)
}

#[test]
fn a_picture_is_generated_imported_and_placed_as_one_undo_step() {
    let dir0 = std::env::temp_dir().join(format!("fc-generate-fixture-a-{}", std::process::id()));
    std::fs::create_dir_all(&dir0).unwrap();
    let fixture = dir0.join("still.png");
    std::fs::write(&fixture, png(64, 36, 200)).unwrap();
    let image = tool(
        "img",
        "image",
        &fixture,
        r#","--size","{width}x{height}","{references}","{optional}""#,
        r#","perReference":["--reference","{path}"],"optional":{"transparent":["--transparent"],"negative":["--negative","{negative}"]},"multipleOf":32,"referencePromptPrefix":"Edit <Picture 1>. ","referenceMarker":"<Picture""#,
    );
    let mut b = bench("picture", &[image]);
    let listed = b.s.execute("generate.providers", json!({})).unwrap();
    assert_eq!(listed["providers"][0]["id"], "img");
    assert_eq!(listed["providers"][0]["installed"], true);
    assert_eq!(listed["providers"][0]["attribution"], "Stand-in model");
    assert!(listed["manifest"].as_str().unwrap().ends_with("generation/providers.json"));
    let (undo0, items0) = (b.s.history.undo.len(), b.s.project.items.len());
    let at = after_everything(&b.s);
    let prompt = "a lighthouse; rm -rf ~ \"quoted\" $(nope)";
    let v =
        b.s.execute(
            "generate.image",
            json!({"prompt": prompt, "transparent": true, "seed": 41, "dir": b.out(), "place": true, "atTime": at.0, "seconds": 2, "wait": true}),
        )
        .unwrap();
    let g = made(&v, 0);
    assert_eq!(b.s.history.undo.len(), undo0 + 1, "one undo step: {v}");
    assert_eq!(b.s.project.items.len(), items0 + 1);
    let item = ItemId(g["item"].as_u64().unwrap());
    let it = b.s.project.item(item).unwrap();
    assert_eq!(it.metadata.get("Description").map(String::as_str), Some(prompt));
    assert_eq!(it.metadata.get("Generator").map(String::as_str), Some("img"));
    assert_eq!(it.metadata.get("Credit").map(String::as_str), Some("Stand-in model"));
    assert!(it.metadata.get("Log Note").is_some_and(|n| n.contains("seed 41") && n.contains("1024×576")), "{:?}", it.metadata);
    assert_eq!(g["start"].as_i64(), Some(at.0));
    assert_eq!(g["duration"].as_i64(), Some(sec(&b.s, 2.0).0));
    // the tool's small picture fills the frame, not a corner of it
    let scale = {
        let q = b.s.active_sequence().unwrap();
        let (_, clip) = q.find_item(ClipId(g["clip"].as_u64().unwrap())).unwrap();
        format!("{:?}", clip.effects.iter().find(|e| e.effect == "motion").unwrap().params["scale"].value)
    };
    assert!(!scale.contains("100.0"), "scaled up to the frame: {scale}");
    // the tool got the prompt as one argument, the size in its multiples, and only the groups asked for
    let path = g["path"].as_str().unwrap();
    let args = b.args_of(path);
    assert!(args.contains(&prompt.to_string()), "{args:?}");
    assert!(args.contains(&"1024x576".to_string()) && args.contains(&"--transparent".to_string()) && args.contains(&"41".to_string()), "{args:?}");
    assert!(!args.iter().any(|a| a == "--negative" || a == "--reference"), "{args:?}");
    assert!(!Path::new(&format!("{}/never", b.out())).exists());
    // it is in the history list, and undo takes the clip and the item away again
    let h = b.s.execute("generate.history", json!({})).unwrap();
    assert_eq!(h["generated"][0]["item"].as_u64(), Some(item.0));
    b.s.undo().unwrap();
    assert_eq!(b.s.project.items.len(), items0);

    // a reference by item id and by the frame under the playhead: both reach the tool, the frame first
    let v = b.s.execute("generate.image", json!({"prompt": "the same, at night", "dir": b.out(), "wait": true})).unwrap();
    let first = made(&v, 0)["item"].as_u64().unwrap();
    assert!(made(&v, 0)["clip"].is_null(), "not placed unless asked: {v}");
    b.s.execute("playhead.set", json!({"seconds": 1.0})).unwrap();
    let v =
        b.s.execute("generate.image", json!({"prompt": "add a red kite", "references": [first], "fromPlayhead": true, "dir": b.out(), "wait": true})).unwrap();
    let args = b.args_of(made(&v, 0)["path"].as_str().unwrap());
    let refs: Vec<&String> = args.iter().zip(args.iter().skip(1)).filter(|(a, _)| *a == "--reference").map(|(_, r)| r).collect();
    assert_eq!(refs.len(), 2, "{args:?}");
    assert!(refs[0].contains("Reference Frame") && Path::new(refs[0]).is_file(), "{refs:?}");
    assert!(args.contains(&"Edit <Picture 1>. add a red kite".to_string()), "{args:?}");
    // a file that is only wanted on disk (an asset for a motion-graphics scene) is not imported
    let n = b.s.project.items.len();
    let v = b.s.execute("generate.image", json!({"prompt": "a paper plane", "import": false, "dir": b.out(), "wait": true})).unwrap();
    assert!(made(&v, 0)["item"].is_null() && Path::new(made(&v, 0)["path"].as_str().unwrap()).is_file());
    assert_eq!(b.s.project.items.len(), n);
    let _ = std::fs::remove_dir_all(&dir0);
}

#[test]
fn a_cut_out_is_the_tools_own_or_a_second_step_on_its_picture() {
    let dir0 = std::env::temp_dir().join(format!("fc-generate-fixture-c-{}", std::process::id()));
    std::fs::create_dir_all(&dir0).unwrap();
    let fixture = dir0.join("still.png");
    std::fs::write(&fixture, png(32, 32, 120)).unwrap();
    // a tool with no transparent option of its own, and a cut-out program that notes what it was given
    let cutting = tool(
        "cutting",
        "image",
        &fixture,
        "",
        r#","cutoutCommand":["/bin/sh","-c","printf '%s' \"$1\" > \"$2.from\"; cp \"$1\" \"$2\"","sh","{image}","{out}"],"cutoutPromptSuffix":", on a plain white background""#,
    );
    let plain = tool("plain", "image", &fixture, "", "");
    let mut b = bench("cutout", &[cutting, plain]);
    let v = b.s.execute("generate.image", json!({"prompt": "a paper kite", "transparent": true, "import": false, "dir": b.out(), "wait": true})).unwrap();
    assert_eq!(v["steps"].as_array().unwrap().len(), 2, "{v}");
    let out = made(&v, 0)["path"].as_str().unwrap();
    assert!(out.ends_with("Generated Image 1.png"), "{out}");
    // the cut-out program worked on the picture the tool made, and the tool was asked for a plain background
    let from = std::fs::read_to_string(format!("{out}.from")).unwrap();
    assert!(from.ends_with("Generated Image source 1.png") && Path::new(&from).is_file(), "{from}");
    assert!(b.args_of(&from).contains(&"a paper kite, on a plain white background".to_string()));
    // without the wish for a cut-out it is one step and the prompt is the user's
    let v = b.s.execute("generate.image", json!({"prompt": "a paper kite", "import": false, "dir": b.out(), "wait": true})).unwrap();
    assert_eq!(v["steps"].as_array().unwrap().len(), 1);
    assert!(b.args_of(made(&v, 0)["path"].as_str().unwrap()).contains(&"a paper kite".to_string()));
    // a tool that can do neither says so
    let e = b.s.execute("generate.image", json!({"prompt": "x", "transparent": true, "provider": "plain"})).unwrap_err().to_string();
    assert!(e.contains("cannot make a transparent picture"), "{e}");
    let _ = std::fs::remove_dir_all(&dir0);
}

#[test]
fn hostile_requests_are_refused_before_any_tool_runs() {
    let dir0 = std::env::temp_dir().join(format!("fc-generate-fixture-h-{}", std::process::id()));
    std::fs::create_dir_all(&dir0).unwrap();
    let fixture = dir0.join("still.png");
    std::fs::write(&fixture, png(8, 8, 9)).unwrap();
    let image = tool("img", "image", &fixture, r#","{references}""#, r#","perReference":["--reference","{path}"],"maxReferences":2"#);
    let plain = tool("plain", "image", &fixture, "", "");
    let gone = r#"{"id":"gone","kind":"video","name":"Not installed","command":["/nonexistent/tool","{out}"]}"#.to_string();
    let mut b = bench("hostile", &[image, plain, gone]);
    let items0 = b.s.project.items.len();
    let long = "x".repeat(MAX_PROMPT + 1);
    for (cmd, p, why) in [
        ("generate.image", json!({}), "is needed"),
        ("generate.image", json!({"prompt": "   "}), "is needed"),
        ("generate.image", json!({"prompt": long}), "longer than"),
        ("generate.image", json!({"prompt": 7}), "must be text"),
        ("generate.image", json!({"prompt": "x", "provider": "nope"}), "no generation tool"),
        ("generate.image", json!({"prompt": "x", "provider": "gone"}), "makes video"),
        ("generate.image", json!({"prompt": "x", "references": "a"}), "must be a list"),
        ("generate.image", json!({"prompt": "x", "references": ["relative.png"]}), "absolute file path"),
        ("generate.image", json!({"prompt": "x", "references": ["/nonexistent/a.png"]}), "is not a file"),
        ("generate.image", json!({"prompt": "x", "references": [999_999]}), "not a media file"),
        ("generate.image", json!({"prompt": "x", "references": [1, 1, 1]}), "at most 2"),
        ("generate.image", json!({"prompt": "x", "provider": "plain", "fromPlayhead": true}), "takes no reference"),
        ("generate.image", json!({"prompt": "x", "seconds": 0}), "0.1 to"),
        ("generate.image", json!({"prompt": "x", "seconds": f64::MAX}), "0.1 to"),
        ("generate.image", json!({"prompt": "x", "strength": 3}), "0 to 1"),
        ("generate.image", json!({"prompt": "x", "width": -4}), "width"),
        ("generate.video", json!({"prompt": "x", "wait": true}), "not installed"),
        ("generate.video", json!({"prompt": "x", "image": "/nonexistent/a.png"}), "is not a file"),
        ("generate.sound", json!({"prompt": "x"}), "no sfx generation tool"),
        ("generate.sound", json!({"prompt": "x", "kind": "noise"}), "unknown kind"),
        ("generate.fillGap", json!({"track": "A1"}), "not a video track"),
        ("generate.fillGap", json!({"track": "V99"}), "no track"),
        ("generate.fillGap", json!({"from": "sideways", "all": true}), "unknown `from`"),
        ("generate.augment", json!({}), "is needed"),
        ("generate.augment", json!({"instruction": "x", "provider": "plain"}), "cannot change a picture"),
        ("generate.augment", json!({"instruction": "x", "atTime": i64::MAX}), "no picture on the timeline"),
    ] {
        let e = b.s.execute(cmd, p.clone()).unwrap_err().to_string();
        assert!(e.contains(why), "{cmd} {p}: {e}");
    }
    assert_eq!(b.s.project.items.len(), items0);
    assert!(running(&b.s).is_none());
    // a damaged list is said, not ignored; no list at all says how to make one
    let manifest = run::manifest_path().unwrap();
    std::fs::write(&manifest, "{not json").unwrap();
    assert!(b.s.execute("generate.image", json!({"prompt": "x"})).unwrap_err().to_string().contains("not valid"));
    assert!(b.s.execute("generate.providers", json!({})).unwrap()["error"].as_str().unwrap().contains("not valid"));
    std::fs::remove_file(&manifest).unwrap();
    let e = b.s.execute("generate.image", json!({"prompt": "x"})).unwrap_err().to_string();
    assert!(e.contains("no image generation tool is set up") && e.contains("providers.json"), "{e}");
    let _ = std::fs::remove_dir_all(&dir0);
}

#[test]
fn a_failed_tool_changes_nothing_and_only_one_runs_at_a_time() {
    let dir0 = std::env::temp_dir().join(format!("fc-generate-fixture-f-{}", std::process::id()));
    std::fs::create_dir_all(&dir0).unwrap();
    let fixture = dir0.join("still.png");
    std::fs::write(&fixture, png(8, 8, 9)).unwrap();
    let failing =
        r#"{"id":"bad","kind":"image","name":"Failing","command":["/bin/sh","-c","echo the model ran out of memory >&2; exit 2","sh","{out}"]}"#.to_string();
    let slow = r#"{"id":"slow","kind":"sfx","name":"Slow","command":["/bin/sh","-c","sleep 20","sh","{out}"]}"#.to_string();
    let good = tool("good", "image", &fixture, "", "");
    let mut b = bench("failed", &[failing, slow, good]);
    let (undo0, items0) = (b.s.history.undo.len(), b.s.project.items.len());
    let e = b.s.execute("generate.image", json!({"prompt": "x", "provider": "bad", "dir": b.out(), "wait": true})).unwrap_err().to_string();
    assert!(e.contains("ran out of memory"), "{e}");
    assert_eq!((b.s.history.undo.len(), b.s.project.items.len()), (undo0, items0));
    // a running generation refuses a second one, and stopping it leaves nothing behind
    let v = b.s.execute("generate.sound", json!({"prompt": "rain", "dir": b.out()})).unwrap();
    let job = v["job"].as_u64().unwrap();
    assert_eq!(running(&b.s), Some(job));
    let e = b.s.execute("generate.image", json!({"prompt": "x", "provider": "good", "dir": b.out()})).unwrap_err().to_string();
    assert!(e.contains("already being generated"), "{e}");
    b.s.execute("jobs.cancel", json!({"job": job})).unwrap();
    let t0 = std::time::Instant::now();
    while running(&b.s).is_some() && t0.elapsed().as_secs() < 15 {
        std::thread::sleep(std::time::Duration::from_millis(50));
        b.s.execute("jobs.list", json!({})).unwrap();
    }
    assert!(running(&b.s).is_none(), "the stopped job must end");
    assert!(outcome_json(&b.s, job).unwrap()["error"].as_str().unwrap().contains("stopped"));
    assert_eq!((b.s.history.undo.len(), b.s.project.items.len()), (undo0, items0));
    assert!(!Path::new(v["files"][0].as_str().unwrap()).exists());
    let _ = std::fs::remove_dir_all(&dir0);
}

#[test]
fn a_gap_is_filled_from_its_start_to_its_end() {
    let dir0 = std::env::temp_dir().join(format!("fc-generate-fixture-g-{}", std::process::id()));
    std::fs::create_dir_all(&dir0).unwrap();
    let fixture = dir0.join("clip.png");
    std::fs::write(&fixture, png(64, 36, 90)).unwrap();
    // (a still stands in for the clip: the engine needs no encoder to test where it lands)
    let vid =
        tool("vid", "video", &fixture, r#","{image}","{frames}","{width}x{height}""#, r#","extension":"png","framesStep":8,"framesOffset":1,"maxSeconds":10"#);
    let img = tool("img", "image", &fixture, "", "");
    let mut b = bench("gap", &[vid, img]);
    // two stills on a track of their own, two seconds apart, well after the demo's clips
    let v = b.s.execute("generate.image", json!({"prompt": "a", "dir": b.out(), "wait": true})).unwrap();
    let item = made(&v, 0)["item"].as_u64().unwrap();
    let t0 = after_everything(&b.s);
    let unit = sec(&b.s, 1.0);
    let sec = |x: i64| Tick(unit.0 * x);
    b.s.execute("timeline.place", json!({"item": item, "track": "new", "time": t0.0, "duration": sec(1).0})).unwrap();
    let track = b.s.active_sequence().unwrap().video_tracks.len();
    let label = format!("V{track}");
    b.s.execute("timeline.place", json!({"item": item, "track": label, "time": (t0 + sec(3)).0, "duration": sec(1).0})).unwrap();
    let on_track = |s: &Session| s.active_sequence().unwrap().video_tracks[track - 1].items.len();
    assert_eq!(on_track(&b.s), 2);
    // the track's gaps: from the start to the first still, and between the two
    let found = gaps(&b.s.active_sequence().unwrap().video_tracks[track - 1].items, Tick(unit.0 / 10));
    assert_eq!(found.len(), 2);
    assert_eq!((found[1].start, found[1].duration), (t0 + sec(1), sec(2)));
    // no gap under the playhead: said; inside the gap: filled, carrying on from the frame before
    assert!(b.s.execute("generate.fillGap", json!({"track": label, "atTime": (t0 + Tick(unit.0 / 2)).0})).unwrap_err().to_string().contains("no gap"));
    let undo0 = b.s.history.undo.len();
    let v = b.s.execute("generate.fillGap", json!({"track": label, "atTime": (t0 + sec(2)).0, "dir": b.out(), "wait": true})).unwrap();
    let g = made(&v, 0);
    assert_eq!(on_track(&b.s), 3, "{v}");
    assert_eq!(g["start"].as_i64(), Some((t0 + sec(1)).0));
    assert_eq!(g["duration"].as_i64(), Some(sec(2).0));
    assert_eq!(g["uncovered"].as_i64(), Some(0));
    assert_eq!(b.s.history.undo.len(), undo0 + 1);
    let args = b.args_of(g["path"].as_str().unwrap());
    assert!(args.iter().any(|a| a.contains("Reference Frame")), "the frame before the gap starts the clip: {args:?}");
    assert!(args.contains(&"49".to_string()), "two seconds at 24 frames a second, a count the tool takes: {args:?}");
    assert!(args.contains(&CONTINUE.to_string()), "{args:?}");
    // nothing left to fill there; every gap at once fills the one before the first still
    assert!(b.s.execute("generate.fillGap", json!({"track": label, "atTime": (t0 + sec(2)).0})).unwrap_err().to_string().contains("no gap"));
    let v =
        b.s.execute("generate.fillGap", json!({"track": label, "all": true, "from": "prompt", "prompt": "clouds drifting", "dir": b.out(), "wait": true}))
            .unwrap();
    assert_eq!(v["gaps"].as_array().unwrap().len(), 1);
    assert_eq!(made(&v, 0)["start"].as_i64(), Some(0));
    assert_eq!(on_track(&b.s), 4);
    let _ = std::fs::remove_dir_all(&dir0);
}

#[test]
fn sounds_and_augmented_frames_never_overwrite_what_is_on_the_timeline() {
    let dir0 = std::env::temp_dir().join(format!("fc-generate-fixture-s-{}", std::process::id()));
    std::fs::create_dir_all(&dir0).unwrap();
    let still = dir0.join("still.png");
    std::fs::write(&still, png(64, 36, 30)).unwrap();
    let tone = dir0.join("tone.wav");
    std::fs::write(&tone, wav()).unwrap();
    let sfx = tool("sfx", "sfx", &tone, r#","--seconds","{seconds}""#, "");
    let img = tool("img", "image", &still, r#","{references}""#, r#","perReference":["--reference","{path}"]"#);
    let vid = tool("vid", "video", &still, r#","{image}""#, r#","extension":"png""#);
    let mut b = bench("sound", &[sfx, img, vid]);
    let count = |s: &Session| -> (usize, usize) {
        let q = s.active_sequence().unwrap();
        (q.video_tracks.iter().map(|t| t.items.len()).sum(), q.audio_tracks.iter().map(|t| t.items.len()).sum())
    };
    let (v0, a0) = count(&b.s);
    // a sound at the very start, where the demo already has sound: it goes on a free track
    let v =
        b.s.execute("generate.sound", json!({"prompt": "a door closing", "seconds": 0.5, "track": "A1", "atTime": 0, "dir": b.out(), "wait": true})).unwrap();
    let g = made(&v, 0);
    assert!(g["clip"].as_u64().is_some(), "{v}");
    assert_eq!(count(&b.s), (v0, a0 + 1), "nothing was replaced: {v}");
    assert!(b.args_of(g["path"].as_str().unwrap()).contains(&"0.5".to_string()));
    // an instruction changes the frame under the playhead and lies over the clip, above it
    let (at, under) = {
        let q = b.s.active_sequence().unwrap();
        let it = q.video_tracks.iter().flat_map(|t| t.items.iter()).min_by_key(|i| i.start).unwrap();
        let at = it.start + sec(&b.s, 0.25);
        (at, clip_under(&b.s, at).unwrap().0)
    };
    let one = sec(&b.s, 1.0);
    let v = b.s.execute("generate.augment", json!({"instruction": "add falling snow", "atTime": at.0, "seconds": 1, "dir": b.out(), "wait": true})).unwrap();
    let g = made(&v, 0);
    let clip = ClipId(g["clip"].as_u64().unwrap());
    let q = b.s.active_sequence().unwrap();
    let (track_id, it) = q.find_item(clip).unwrap();
    let index = q.video_tracks.iter().position(|t| t.id == track_id).unwrap();
    assert!(index > under, "over the clip, not in its place");
    assert_eq!((it.start, it.duration), (at, one));
    assert_eq!(count(&b.s), (v0 + 1, a0 + 1));
    let args = b.args_of(g["path"].as_str().unwrap());
    assert!(args.iter().any(|a| a.contains("Reference Frame")), "the frame is what gets changed: {args:?}");
    // animated: the changed frame is made first, then a clip from it; both are kept, the clip is placed
    let v = b.s.execute("generate.augment", json!({"instruction": "a flock of birds crosses the sky", "motion": "the birds fly left to right", "animate": true, "atTime": at.0, "dir": b.out(), "wait": true})).unwrap();
    assert_eq!(v["generated"].as_array().unwrap().len(), 2, "{v}");
    assert!(made(&v, 0)["clip"].is_null() && made(&v, 1)["clip"].as_u64().is_some(), "{v}");
    let args = b.args_of(made(&v, 1)["path"].as_str().unwrap());
    assert!(args.contains(&made(&v, 0)["path"].as_str().unwrap().to_string()), "the clip starts from the changed frame: {args:?}");
    assert!(args.contains(&"the birds fly left to right".to_string()));
    let _ = std::fs::remove_dir_all(&dir0);
}
