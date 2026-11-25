//! `generate.*` with stand-in tools: shell one-liners that copy a prepared file to where the
//! result should go and note the arguments they were given. No model runs in the tests.
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::sync::{MutexGuard, PoisonError};

use serde_json::{Value, json};

use super::*;

/// The provider list is one file per test binary: tests that write it take turns (with the
/// `motion.*` tests, which write it too).
use crate::generate_run::LIST_TURN as TURN;

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

#[test]
fn variations_wait_in_the_bin_and_one_can_be_swapped_in() {
    let dir0 = std::env::temp_dir().join(format!("fc-generate-fixture-v-{}", std::process::id()));
    std::fs::create_dir_all(&dir0).unwrap();
    let still = dir0.join("still.png");
    std::fs::write(&still, png(64, 36, 60)).unwrap();
    let tone = dir0.join("tone.wav");
    std::fs::write(&tone, wav()).unwrap();
    let img = tool("img", "image", &still, "", "");
    let sfx = tool("sfx", "sfx", &tone, r#","--seconds","{seconds}""#, r#","maxSeconds":30"#);
    let mut b = bench("variations", &[img, sfx]);
    let (undo0, items0) = (b.s.history.undo.len(), b.s.project.items.len());
    let at = after_everything(&b.s);
    let v =
        b.s.execute(
            "generate.image",
            json!({"prompt": "a kite", "variations": 3, "seed": 41, "dir": b.out(), "place": true, "track": "new", "atTime": at.0, "seconds": 2, "wait": true}),
        )
        .unwrap();
    assert_eq!(v["generated"].as_array().unwrap().len(), 3, "{v}");
    assert_eq!((b.s.history.undo.len(), b.s.project.items.len()), (undo0 + 1, items0 + 3), "three items, one undo step");
    // the first is on the timeline, the others wait in the bin; each got its own seed
    let clip = made(&v, 0)["clip"].as_u64().unwrap();
    assert!(made(&v, 1)["clip"].is_null() && made(&v, 2)["clip"].is_null(), "{v}");
    for (k, seed) in ["41", "42", "43"].iter().enumerate() {
        assert!(b.args_of(made(&v, k)["path"].as_str().unwrap()).contains(&seed.to_string()));
    }
    let items: Vec<u64> = (0..3).map(|k| made(&v, k)["item"].as_u64().unwrap()).collect();
    let group = |s: &Session, id: u64| s.project.item(ItemId(id)).unwrap().metadata.get(VARIATION).cloned();
    assert!(group(&b.s, items[0]).is_some_and(|g| g.ends_with("Generated Image 1.png")));
    assert_eq!(group(&b.s, items[0]), group(&b.s, items[2]));
    let h = b.s.execute("generate.history", json!({})).unwrap();
    assert!(h["generated"][0]["variationOf"].as_str().is_some(), "{h}");
    // another variation takes the place of the one on the timeline: same clip, same stretch, one undo step
    let on_clip = |s: &Session| {
        let (_, it) = s.active_sequence().unwrap().find_item(ClipId(clip)).unwrap();
        (it.item.0, it.start, it.duration)
    };
    let before = on_clip(&b.s);
    assert_eq!(before.0, items[0]);
    let undo1 = b.s.history.undo.len();
    let r = b.s.execute("generate.pick", json!({"item": items[2]})).unwrap();
    assert_eq!(r["clip"].as_u64(), Some(clip));
    assert_eq!(on_clip(&b.s), (items[2], before.1, before.2));
    assert_eq!(b.s.history.undo.len(), undo1 + 1);
    b.s.undo().unwrap();
    assert_eq!(on_clip(&b.s).0, items[0]);
    // one picture alone has no variations to swap with; a clip named outright is replaced
    let lone = b.s.execute("generate.image", json!({"prompt": "a lone gull", "dir": b.out(), "wait": true})).unwrap();
    let lone = made(&lone, 0)["item"].as_u64().unwrap();
    assert!(group(&b.s, lone).is_none());
    assert!(b.s.execute("generate.pick", json!({"item": lone})).unwrap_err().to_string().contains("none of its variations"));
    b.s.execute("generate.pick", json!({"item": lone, "clip": clip})).unwrap();
    assert_eq!(on_clip(&b.s).0, lone);
    let filmed = b.s.project.items.values().find(|i| !i.metadata.contains_key("Generator")).unwrap().id.0;
    for (p, why) in [
        (json!({}), "is needed"),
        (json!({"item": 999_999}), "no item"),
        (json!({"item": filmed}), "was not generated"),
        (json!({"item": lone, "clip": 999_999}), "no clip"),
    ] {
        let e = b.s.execute("generate.pick", p.clone()).unwrap_err().to_string();
        assert!(e.contains(why), "{p}: {e}");
    }
    for (cmd, p, why) in [
        ("generate.image", json!({"prompt": "x", "variations": 0}), "1 to 4"),
        ("generate.image", json!({"prompt": "x", "variations": 99}), "1 to 4"),
        ("generate.image", json!({"prompt": "x", "variations": "many"}), "1 to 4"),
        ("generate.image", json!({"prompt": "x", "move": "sideways"}), "unknown `move`"),
        ("generate.sound", json!({"prompt": "x", "clip": 999_999}), "no clip"),
    ] {
        let e = b.s.execute(cmd, p.clone()).unwrap_err().to_string();
        assert!(e.contains(why), "{cmd} {p}: {e}");
    }
    // a sound made for a clip is as long as it and starts where it starts
    let v = b.s.execute("generate.sound", json!({"prompt": "wind in a kite's tail", "clip": clip, "variations": 2, "dir": b.out(), "wait": true})).unwrap();
    assert_eq!(v["generated"].as_array().unwrap().len(), 2);
    assert_eq!(made(&v, 0)["start"].as_i64(), Some(before.1.0));
    let args = b.args_of(made(&v, 0)["path"].as_str().unwrap());
    let asked: f64 = args.iter().position(|a| a == "--seconds").and_then(|i| args.get(i + 1)).and_then(|x| x.parse().ok()).unwrap();
    assert!((asked - before.2.seconds()).abs() < 0.002, "the clip's length: {args:?}");
    // a placed picture asked to move in depth is placed either way; without the depth model it says why
    let v =
        b.s.execute("generate.image", json!({"prompt": "a cliff", "move": "in", "place": true, "track": "new", "atTime": at.0, "dir": b.out(), "wait": true}))
            .unwrap();
    let g = made(&v, 0);
    assert!(g["clip"].as_u64().is_some() && (g["move"] == "in" || g["moveError"].as_str().is_some()), "{v}");
    let _ = std::fs::remove_dir_all(&dir0);
}

#[test]
fn an_asset_list_makes_what_is_missing_and_keeps_the_rest() {
    let dir0 = std::env::temp_dir().join(format!("fc-generate-fixture-l-{}", std::process::id()));
    std::fs::create_dir_all(&dir0).unwrap();
    let still = dir0.join("still.png");
    std::fs::write(&still, png(32, 32, 80)).unwrap();
    let tone = dir0.join("tone.wav");
    std::fs::write(&tone, wav()).unwrap();
    let img = tool(
        "img",
        "image",
        &still,
        r#","--size","{width}x{height}","{references}""#,
        r#","perReference":["--reference","{path}"],"multipleOf":32,"cutoutCommand":["/bin/sh","-c","cp \"$1\" \"$2\"","sh","{image}","{out}"]"#,
    );
    let vid = tool("vid", "video", &still, r#","{optional}""#, r#","extension":"png","optional":{"image":["--image","{image}"]}"#);
    let sfx = tool("sfx", "sfx", &tone, r#","--seconds","{seconds}""#, "");
    let mut b = bench("assets", &[img, vid, sfx]);
    let scene = b.dir.join("scene");
    std::fs::create_dir_all(&scene).unwrap();
    let manifest = scene.join("assets.json");
    let list = json!({"width": 768, "height": 768, "assets": [
        {"name": "mascot", "prompt": "a robot mascot, full body", "seed": 11},
        {"name": "mascot wave", "prompt": "the robot waves", "references": ["mascot"], "transparent": true},
        {"name": "plate moving", "kind": "video", "prompt": "clouds drift", "image": "mascot", "seconds": 2},
        {"name": "swish", "kind": "sfx", "prompt": "a soft swish", "seconds": 1},
    ]});
    std::fs::write(&manifest, list.to_string()).unwrap();
    let m = manifest.to_string_lossy().into_owned();
    let file = |name: &str| scene.join(name).to_string_lossy().into_owned();
    let (undo0, items0) = (b.s.history.undo.len(), b.s.project.items.len());
    // a dry run says what would be made and makes nothing
    let plan = b.s.execute("generate.assets", json!({"manifest": m, "dryRun": true})).unwrap();
    assert_eq!((plan["making"].as_u64(), plan["assets"][1]["state"].as_str()), (Some(4), Some("make")), "{plan}");
    assert!(plan["job"].is_null() && !Path::new(&file("mascot.png")).exists());
    let v = b.s.execute("generate.assets", json!({"manifest": m, "wait": true})).unwrap();
    assert_eq!(v["making"].as_u64(), Some(4), "{v}");
    for name in ["mascot.png", "mascot wave.png", "mascot wave source.png", "plate moving.png", "swish.wav"] {
        assert!(Path::new(&file(name)).is_file(), "{name}");
    }
    // files for a scene: nothing was imported, nothing to undo
    assert_eq!((b.s.history.undo.len(), b.s.project.items.len()), (undo0, items0));
    let args = b.args_of(&file("mascot.png"));
    assert!(args.contains(&"11".to_string()) && args.contains(&"768x768".to_string()), "{args:?}");
    // a later asset is made from an earlier one by name; a seed that was not given follows the name
    let wave = b.args_of(&file("mascot wave source.png"));
    assert!(wave.windows(2).any(|w| w[0] == "--reference" && w[1] == file("mascot.png")), "{wave:?}");
    assert!(wave.contains(&seed_of("mascot wave").to_string()), "{wave:?}");
    assert!(b.args_of(&file("plate moving.png")).contains(&file("mascot.png")));
    // again: everything is there, so nothing runs
    let again = b.s.execute("generate.assets", json!({"manifest": m, "wait": true})).unwrap();
    assert_eq!(again["making"].as_u64(), Some(0));
    assert!(again["job"].is_null() && again["assets"].as_array().unwrap().iter().all(|a| a["state"] == "kept"), "{again}");
    // one file deleted: exactly that one is made again, from the same seed
    std::fs::remove_file(file("mascot wave.png")).unwrap();
    std::fs::remove_file(format!("{}.args", file("mascot wave source.png"))).unwrap();
    let stamp = std::fs::metadata(file("mascot.png")).unwrap().modified().unwrap();
    let v = b.s.execute("generate.assets", json!({"manifest": m, "wait": true})).unwrap();
    assert_eq!(v["making"].as_u64(), Some(1), "{v}");
    assert!(Path::new(&file("mascot wave.png")).is_file());
    assert_eq!(b.args_of(&file("mascot wave source.png")), wave);
    assert_eq!(std::fs::metadata(file("mascot.png")).unwrap().modified().unwrap(), stamp, "the others were left alone");
    // one named file made again on request, and brought into the project when asked
    let v = b.s.execute("generate.assets", json!({"manifest": m, "only": ["swish"], "force": true, "import": true, "wait": true})).unwrap();
    assert_eq!(v["making"].as_u64(), Some(1));
    assert_eq!(b.s.project.items.len(), items0 + 1);
    // hostile lists and requests are refused before anything runs
    let write = |name: &str, v: Value| {
        let p = scene.join(name);
        std::fs::write(&p, v.to_string()).unwrap();
        p.to_string_lossy().into_owned()
    };
    let many: Vec<Value> = (0..33).map(|i| json!({"name": format!("a{i}"), "prompt": "x"})).collect();
    for (p, why) in [
        (json!({}), "is needed"),
        (json!({"manifest": "assets.json"}), "absolute path"),
        (json!({"manifest": file("assets.txt")}), "absolute path"),
        (json!({"manifest": file("missing.json")}), "missing.json"),
        (json!({"manifest": m, "only": ["nope"]}), "does not have"),
        (json!({"manifest": m, "only": "swish"}), "list of names"),
        (json!({"manifest": write("bad1.json", json!({"assets": []}))}), "1 to 32"),
        (json!({"manifest": write("bad2.json", json!({"assets": many}))}), "1 to 32"),
        (json!({"manifest": write("bad3.json", json!({"assets": [{"name": "../up", "prompt": "x"}]}))}), "letters, digits"),
        (json!({"manifest": write("bad4.json", json!({"assets": [{"name": "a", "prompt": "x"}, {"name": "A", "prompt": "y"}]}))}), "used twice"),
        (
            json!({"manifest": write("bad5.json", json!({"assets": [{"name": "a", "prompt": "x", "references": ["later"]}, {"name": "later", "prompt": "y"}]}))}),
            "neither an asset",
        ),
        (json!({"manifest": write("bad6.json", json!({"assets": [{"name": "a", "prompt": "x", "shell": "rm -rf"}]}))}), "unknown field"),
        (json!({"manifest": write("bad7.json", json!({"assets": [{"name": "a", "prompt": "x", "kind": "hologram"}]}))}), "unknown kind"),
        (json!({"manifest": write("bad8.json", json!({"assets": [{"name": "a", "prompt": "  "}]}))}), "empty or too long"),
        (json!({"manifest": write("bad9.json", json!({"assets": [{"name": "a", "prompt": "x", "provider": "nope"}]}))}), "no generation tool"),
    ] {
        let e = b.s.execute("generate.assets", p.clone()).unwrap_err().to_string();
        assert!(e.contains(why), "{p}: {e}");
    }
    std::fs::write(scene.join("bad10.json"), "{not json").unwrap();
    assert!(b.s.execute("generate.assets", json!({"manifest": file("bad10.json")})).is_err());
    assert!(running(&b.s).is_none());
    let _ = std::fs::remove_dir_all(&dir0);
}

#[test]
fn a_gap_is_filled_between_two_pictures_or_by_carrying_the_shot_on() {
    let dir0 = std::env::temp_dir().join(format!("fc-generate-fixture-m-{}", std::process::id()));
    std::fs::create_dir_all(&dir0).unwrap();
    let fixture = dir0.join("clip.png");
    std::fs::write(&fixture, png(64, 36, 90)).unwrap();
    let f = fixture.to_string_lossy();
    // (a still stands in for the clip; the extend program notes what it was given)
    let vid = tool(
        "vid",
        "video",
        &fixture,
        r#","{frames}","{optional}""#,
        &format!(
            r#","extension":"png","framesStep":8,"framesOffset":1,"maxSeconds":10,"optional":{{"image":["--image","{{image}}"],"lastImage":["--last","{{lastImage}}","{{lastFrame}}"]}},"extendCommand":["/bin/sh","-c","out=\"$2\"; printf '%s\\n' \"$@\" > \"$out.args\"; cp \"$1\" \"$out\"","sh","{f}","{{out}}","--video","{{video}}","--frames","{{frames}}","--prompt","{{prompt}}"]"#
        ),
    );
    let plain = tool("plain", "video", &fixture, "", r#","extension":"png""#);
    let img = tool("img", "image", &fixture, "", "");
    let mut b = bench("motion", &[vid, plain, img]);
    let listed = b.s.execute("generate.providers", json!({})).unwrap();
    assert_eq!((listed["providers"][0]["bothEnds"].as_bool(), listed["providers"][0]["carriesOn"].as_bool()), (Some(true), Some(true)));
    assert_eq!((listed["providers"][1]["bothEnds"].as_bool(), listed["providers"][1]["carriesOn"].as_bool()), (Some(false), Some(false)));
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
    let inside = (t0 + sec(2)).0;
    // from the frame before to the frame after: the clip starts on one and ends on the other
    let v = b.s.execute("generate.fillGap", json!({"track": label, "atTime": inside, "from": "both", "dir": b.out(), "wait": true})).unwrap();
    let g = made(&v, 0);
    assert_eq!((g["start"].as_i64(), g["duration"].as_i64(), on_track(&b.s)), (Some((t0 + sec(1)).0), Some(sec(2).0), 3), "{v}");
    let args = b.args_of(g["path"].as_str().unwrap());
    let after = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned().unwrap_or_default();
    assert!(after("--image").contains("Reference Frame 1") && after("--last").contains("Reference Frame 2"), "{args:?}");
    assert!(args.contains(&"48".to_string()), "the last picture is pinned on the last of 49 frames: {args:?}");
    assert!(args.contains(&BETWEEN.to_string()), "{args:?}");
    b.s.undo().unwrap();
    assert_eq!(on_track(&b.s), 2);
    // carrying the shot on: the tool is given the moments before the gap as a video, and the new part is placed
    let v = b.s.execute("generate.fillGap", json!({"track": label, "atTime": inside, "from": "motion", "dir": b.out(), "wait": true})).unwrap();
    let g = made(&v, 0);
    assert_eq!((g["start"].as_i64(), on_track(&b.s)), (Some((t0 + sec(1)).0), 3), "{v}");
    let args = b.args_of(g["path"].as_str().unwrap());
    let after = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned().unwrap_or_default();
    let carried = after("--video");
    assert!(carried.contains("Carried Frames") && std::fs::metadata(&carried).is_ok_and(|m| m.len() > 0), "{args:?}");
    assert_eq!(after("--frames"), "48", "two seconds of new frames, in eights");
    assert!(args.contains(&CONTINUE.to_string()), "{args:?}");
    // a shot carried on from where a clip ends: it starts there, on the same track when that is free
    let end = t0 + sec(4);
    let v = b.s.execute("generate.extend", json!({"atTime": end.0, "seconds": 1, "prompt": "the kite climbs", "dir": b.out(), "wait": true})).unwrap();
    let g = made(&v, 0);
    assert_eq!(g["start"].as_i64(), Some(end.0), "{v}");
    assert_eq!(on_track(&b.s), 4, "on the track of the clip it carries on: {v}");
    let args = b.args_of(g["path"].as_str().unwrap());
    assert!(args.contains(&"24".to_string()) && args.contains(&"the kite climbs".to_string()), "{args:?}");
    // a tool that can do neither says so, and so do hostile requests
    for (cmd, p, why) in [
        ("generate.fillGap", json!({"track": label, "all": true, "from": "both", "provider": "plain"}), "cannot end a clip"),
        ("generate.fillGap", json!({"track": label, "all": true, "from": "motion", "provider": "plain"}), "cannot carry a shot on"),
        ("generate.extend", json!({"provider": "plain"}), "cannot carry a shot on"),
        ("generate.extend", json!({"seconds": 99}), "0.3 to 10"),
        ("generate.extend", json!({"seconds": "long"}), "0.3 to 10"),
        ("generate.extend", json!({"atTime": 0}), "of picture before it"),
        ("generate.video", json!({"prompt": "x", "provider": "plain", "lastImage": fixture.to_string_lossy()}), "cannot end a clip"),
        ("generate.video", json!({"prompt": "x", "lastImage": "/nonexistent/z.png"}), "is not a file"),
    ] {
        let e = b.s.execute(cmd, p.clone()).unwrap_err().to_string();
        assert!(e.contains(why), "{cmd} {p}: {e}");
    }
    let _ = std::fs::remove_dir_all(&dir0);
}

#[test]
fn a_clip_is_made_again_over_itself() {
    let dir0 = std::env::temp_dir().join(format!("fc-generate-fixture-r-{}", std::process::id()));
    std::fs::create_dir_all(&dir0).unwrap();
    let fixture = dir0.join("clip.png");
    std::fs::write(&fixture, png(64, 36, 90)).unwrap();
    let adapter = dir0.join("union.safetensors");
    std::fs::write(&adapter, b"weights").unwrap();
    let (f, ad, missing) = (fixture.to_string_lossy(), adapter.to_string_lossy(), dir0.join("relight.safetensors").to_string_lossy().into_owned());
    // (a still stands in for the clip; the restyle program notes what it was given)
    let vid = tool(
        "vid",
        "video",
        &fixture,
        "",
        &format!(
            r#","extension":"png","framesStep":8,"framesOffset":1,"maxSeconds":10,"restyleCommand":["/bin/sh","-c","out=\"$2\"; printf '%s\\n' \"$@\" > \"$out.args\"; cp \"$1\" \"$out\"","sh","{f}","{{out}}","--video","{{video}}","--adapter","{{adapter}}","--frames","{{frames}}","--strength","{{strength}}","--prompt","{{prompt}}","{{optional}}"],"restyleOptional":{{"image":["--image","{{image}}"]}},"restyles":{{"edges":{{"adapter":"{ad}","args":["--how","edges"],"about":"keeps the shapes"}},"relight":{{"adapter":"{missing}","args":["--how","source"]}}}}"#
        ),
    );
    let plain = tool("plain", "video", &fixture, "", r#","extension":"png""#);
    let img = tool("img", "image", &fixture, r#","{references}""#, r#","perReference":["--reference","{path}"]"#);
    let mut b = bench("restyle", &[vid, plain, img]);
    let listed = b.s.execute("generate.providers", json!({})).unwrap();
    let ways = &listed["providers"][0]["restyles"];
    assert_eq!(
        (ways[0]["name"].as_str(), ways[0]["installed"].as_bool(), ways[0]["about"].as_str()),
        (Some("edges"), Some(true), Some("keeps the shapes")),
        "{listed}"
    );
    assert_eq!((ways[1]["name"].as_str(), ways[1]["installed"].as_bool()), (Some("relight"), Some(false)), "{listed}");
    assert_eq!(listed["providers"][1]["restyles"].as_array().map(Vec::len), Some(0));
    let v = b.s.execute("generate.image", json!({"prompt": "a", "dir": b.out(), "wait": true})).unwrap();
    let item = made(&v, 0)["item"].as_u64().unwrap();
    let t0 = after_everything(&b.s);
    let unit = sec(&b.s, 1.0);
    let sec = |x: i64| Tick(unit.0 * x);
    let placed = b.s.execute("timeline.place", json!({"item": item, "track": "new", "time": t0.0, "duration": sec(2).0})).unwrap();
    let clip = placed["clips"][0].as_u64().unwrap();
    let track = b.s.active_sequence().unwrap().video_tracks.len();
    let clips_at = |s: &Session| s.active_sequence().unwrap().video_tracks.iter().filter(|t| t.items.iter().any(|i| i.start == t0)).count();
    assert_eq!(clips_at(&b.s), 1);
    // the clip is given to the tool as a small video, and the new one lies over it for its whole length
    let v = b.s.execute("generate.restyle", json!({"prompt": "as a watercolour", "clip": clip, "dir": b.out(), "wait": true})).unwrap();
    let g = made(&v, 0);
    assert_eq!((g["start"].as_i64(), g["duration"].as_i64(), clips_at(&b.s)), (Some(t0.0), Some(sec(2).0), 2), "{v}");
    assert_eq!((v["how"].as_str(), v["clip"].as_u64(), v["left"].as_f64()), (Some("edges"), Some(clip), Some(0.0)), "{v}");
    assert!(b.s.active_sequence().unwrap().video_tracks.len() > track, "on a track above the clip: {v}");
    let args = b.args_of(g["path"].as_str().unwrap());
    let after = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned().unwrap_or_default();
    let source = after("--video");
    assert!(source.contains("Restyle Source") && std::fs::metadata(&source).is_ok_and(|m| m.len() > 0), "{args:?}");
    assert_eq!(
        (after("--adapter"), after("--frames"), after("--strength"), after("--how")),
        (ad.to_string(), "41".to_string(), "1".to_string(), "edges".to_string()),
        "{args:?}"
    );
    assert!(!args.contains(&"--image".to_string()) && args.contains(&"as a watercolour".to_string()), "{args:?}");
    // one undo step
    b.s.undo().unwrap();
    assert_eq!(clips_at(&b.s), 1);
    // the clip under the playhead when none is named; the first frame changed by the picture tool first
    b.s.execute("playhead.set", json!({"time": (t0 + sec(1)).0})).unwrap();
    let v = b
        .s
        .execute(
            "generate.restyle",
            json!({"prompt": "a robot walks", "look": "Put the robot in the person's place", "references": [item], "strength": 0.5, "dir": b.out(), "wait": true}),
        )
        .unwrap();
    assert_eq!(v["clip"].as_u64(), Some(clip), "{v}");
    let (first, g) = (made(&v, 0), made(&v, 1));
    assert!(first["path"].as_str().is_some_and(|p| p.contains("Restyled Frame")) && first["clip"].is_null() && first["item"].is_u64(), "{v}");
    assert_eq!((g["start"].as_i64(), g["duration"].as_i64()), (Some(t0.0), Some(sec(2).0)), "{v}");
    let args = b.args_of(g["path"].as_str().unwrap());
    let after = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned().unwrap_or_default();
    assert!(after("--image").contains("Restyled Frame") && after("--strength") == "0.5", "{args:?}");
    let pic = b.args_of(first["path"].as_str().unwrap());
    let refs: Vec<&String> =
        pic.iter().enumerate().filter(|(i, _)| i.checked_sub(1).and_then(|j| pic.get(j)).is_some_and(|a| a == "--reference")).map(|(_, a)| a).collect();
    assert!(refs.len() == 2 && refs[0].contains("Reference Frame"), "the frame first, then the character: {pic:?}");
    b.s.undo().unwrap();
    assert_eq!(clips_at(&b.s), 1);
    // a given first picture
    let v = b.s.execute("generate.restyle", json!({"prompt": "x", "clip": clip, "image": f, "dir": b.out(), "wait": true})).unwrap();
    assert_eq!(
        b.args_of(made(&v, 0)["path"].as_str().unwrap()).iter().filter(|a| **a == f).count(),
        2,
        "the fixture as the tool's file and as the first picture"
    );
    b.s.undo().unwrap();
    // what cannot be done says so, before any tool runs
    let short = b.s.execute("timeline.place", json!({"item": item, "track": "new", "time": (t0 + sec(4)).0, "duration": (unit.0 / 5)})).unwrap()["clips"][0]
        .as_u64()
        .unwrap();
    for (p, why) in [
        (json!({"prompt": "x", "clip": clip, "how": "relight"}), "is not on this computer"),
        (json!({"prompt": "x", "clip": clip, "how": "nope"}), "unknown `how`"),
        (json!({"prompt": "x", "clip": clip, "how": 7}), "unknown `how`"),
        (json!({"prompt": "x", "clip": clip, "provider": "plain"}), "cannot make a clip again"),
        (json!({"prompt": "x", "clip": 999_999}), "no video clip"),
        (json!({"prompt": "x", "clip": "first"}), "must be a clip id"),
        (json!({"prompt": "x", "clip": clip, "strength": 2}), "0 to 1"),
        (json!({"prompt": "x", "clip": clip, "strength": "high"}), "0 to 1"),
        (json!({"prompt": "x", "clip": clip, "image": f, "look": "y"}), "not both"),
        (json!({"prompt": "x", "clip": clip, "references": [f]}), "go with `look`"),
        (json!({"prompt": "x", "clip": clip, "image": "/nonexistent/z.png"}), "is not a file"),
        (json!({"clip": clip}), "`prompt` is needed"),
        (json!({"prompt": "x", "clip": short}), "too short"),
    ] {
        let e = b.s.execute("generate.restyle", p.clone()).unwrap_err().to_string();
        assert!(e.contains(why), "{p}: {e}");
    }
    assert!(
        !std::fs::read_dir(b.out()).unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with("Restyle Source 4")),
        "nothing was written for a refused request"
    );
    let _ = std::fs::remove_dir_all(&dir0);
}

/// The `.args` files the stand-in tools wrote under `dir` (one a run).
fn runs_in(dir: &str) -> usize {
    std::fs::read_dir(dir).map(|d| d.flatten().filter(|e| e.file_name().to_string_lossy().ends_with(".args")).count()).unwrap_or(0)
}

#[test]
fn capabilities_refuse_what_a_tool_cannot_do_before_it_runs() {
    let dir0 = std::env::temp_dir().join(format!("fc-generate-fixture-k-{}", std::process::id()));
    std::fs::create_dir_all(&dir0).unwrap();
    let still = dir0.join("still.png");
    std::fs::write(&still, png(64, 36, 70)).unwrap();
    let tone = dir0.join("tone.wav");
    std::fs::write(&tone, wav()).unwrap();
    let img = tool(
        "capimg",
        "image",
        &still,
        r#","--size","{width}x{height}","{references}","{optional}""#,
        r#","perReference":["--reference","{path}"],"optional":{"negative":["--negative","{negative}"]},"maxReferences":4,"capabilities":{"sizes":[[1344,768],[1024,1024],[768,1344]],"references":1,"negative":false,"transparent":false}"#,
    );
    let vid = tool(
        "capvid",
        "video",
        &still,
        r#","--frames","{frames}","--size","{width}x{height}","{optional}""#,
        r#","extension":"png","optional":{"image":["--image","{image}"]},"framesStep":8,"framesOffset":1,"capabilities":{"seconds":{"max":5},"frames":{"min":9,"max":121,"step":8,"offset":1},"fps":[24],"firstFrame":false,"audio":false,"width":{"min":256,"max":1024,"step":64},"height":{"min":256,"max":1024,"step":64}}"#,
    );
    let sfx = tool("capsfx", "sfx", &tone, r#","--seconds","{seconds}""#, r#","capabilities":{"seconds":{"min":1,"max":30}}"#);
    let mut b = bench("capabilities", &[img, vid, sfx]);
    // the listing says what each tool can do, as data and in words
    let listed = b.s.execute("generate.providers", json!({})).unwrap();
    let can = listed["providers"][0]["can"].as_str().unwrap();
    assert!(can.contains("1344×768") && can.contains("1 reference picture") && can.contains("no negative prompt"), "{can}");
    assert_eq!(listed["providers"][0]["capabilities"]["references"], 1);
    assert!(listed["providers"][1]["can"].as_str().unwrap().contains("9 to 121 frames (8k + 1)"), "{listed}");
    // a size left open becomes the nearest one the tool makes
    let v = b.s.execute("generate.image", json!({"prompt": "a harbour", "dir": b.out(), "wait": true})).unwrap();
    let (w, h) = (v["width"].as_u64().unwrap(), v["height"].as_u64().unwrap());
    assert!([(1344, 768), (1024, 1024), (768, 1344)].contains(&(w, h)), "{v}");
    assert!(b.args_of(made(&v, 0)["path"].as_str().unwrap()).contains(&format!("{w}x{h}")));
    let ref_file = made(&v, 0)["path"].as_str().unwrap().to_string();
    // what it cannot do is refused before anything runs, saying what it can
    let runs0 = runs_in(&b.out());
    for (cmd, p, why) in [
        ("generate.image", json!({"prompt": "x", "width": 1000, "height": 1000}), "nearest: 1024×1024"),
        ("generate.image", json!({"prompt": "x", "negative": "blur"}), "negative prompt"),
        ("generate.image", json!({"prompt": "x", "transparent": true}), "transparent"),
        ("generate.image", json!({"prompt": "x", "references": [ref_file, ref_file]}), "at most 1"),
        ("generate.image", json!({"prompt": "x", "references": [ref_file], "fromPlayhead": true}), "this asks for 2"),
        ("generate.video", json!({"prompt": "x", "seconds": 8}), "nearest: 5 s"),
        ("generate.video", json!({"prompt": "x", "image": ref_file}), "start a clip on a given picture"),
        ("generate.video", json!({"prompt": "x", "audio": true}), "own sound"),
        ("generate.video", json!({"prompt": "x", "width": 500, "height": 300}), "steps of 64"),
        ("generate.fillGap", json!({"provider": "capvid"}), "firstFrame: false"),
        ("generate.sound", json!({"prompt": "x", "seconds": 45}), "nearest: 30 s"),
        ("generate.sound", json!({"prompt": "x", "seconds": 0.5}), "nearest: 1 s"),
    ] {
        let e = b.s.execute(cmd, p.clone()).unwrap_err().to_string();
        assert!(e.contains(why), "{cmd} {p}: {e}");
    }
    assert_eq!(runs_in(&b.out()), runs0, "no tool ran for a refused request");
    assert!(running(&b.s).is_none());
    // what it can do runs: the length in its frame counts, the size on its grid
    let v = b.s.execute("generate.video", json!({"prompt": "waves", "dir": b.out(), "wait": true})).unwrap();
    let args = b.args_of(made(&v, 0)["path"].as_str().unwrap());
    assert!(args.contains(&"97".to_string()), "four seconds, 8k + 1: {args:?}");
    let size = args.iter().skip_while(|a| *a != "--size").nth(1).unwrap();
    let (sw, sh) = size.split_once('x').map(|(a, b)| (a.parse::<u32>().unwrap(), b.parse::<u32>().unwrap())).unwrap();
    assert!(sw % 64 == 0 && sh % 64 == 0 && (256..=1024).contains(&sw) && (256..=1024).contains(&sh), "{size}");
    let v = b.s.execute("generate.sound", json!({"prompt": "rain", "seconds": 2, "dir": b.out(), "wait": true})).unwrap();
    assert!(b.args_of(made(&v, 0)["path"].as_str().unwrap()).contains(&"2".to_string()));
    let _ = std::fs::remove_dir_all(&dir0);
}
