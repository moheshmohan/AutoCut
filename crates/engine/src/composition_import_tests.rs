//! `composition.import` with a stand-in RAEA whose `export layers` copies prepared movies and a
//! `layers.json` into place. RAEA itself never runs in the tests.
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, MutexGuard, PoisonError};

use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{DemoScene, Generator};
use serde_json::{Value, json};

use super::*;
use crate::generate_run as run;

struct Bench {
    _turn: MutexGuard<'static, ()>,
    dir: PathBuf,
    s: Session,
}

/// A 1 s 30 fps 128×72 movie with sound, made once.
fn movie() -> &'static [u8] {
    static MOVIE: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    MOVIE.get_or_init(|| {
        let tmp = crate::media_test_util::tmp_dir("layers-fixture");
        let path = tmp.join("clip.mp4");
        let rate = FrameRate::FPS_30;
        let src = GeneratorSource::new(Generator::Demo(DemoScene::Aurora), 128, 72, rate, rate.tick_of(30));
        crate::media_test_util::make_movie_from(&path, Arc::new(src));
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_dir_all(&tmp);
        bytes
    })
}

/// Half a second of a tone, 16-bit mono at 48 kHz.
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

/// What RAEA wrote for a composition of footage (a box, half opacity; then full frame at double
/// speed), a sound at -12 dB, its graphics and what lies under the footage.
fn layers_json(dir: &Path) -> Value {
    let media = dir.join("comp").join("assets");
    json!({
        "version": 1, "name": "edit", "width": 128, "height": 72, "fps": 30.0, "frames": 60, "reference": "reference.mp4",
        "files": [
            {"path": media.join("clip.mp4"), "kind": "video", "seconds": 1.0, "width": 128, "height": 72, "has_audio": true},
            {"path": "graphics.prores.mov", "kind": "video", "seconds": 2.0, "width": 128, "height": 72, "has_audio": false},
            {"path": "under.prores.mov", "kind": "video", "seconds": 2.0, "width": 128, "height": 72, "has_audio": false},
            {"path": media.join("tone.wav"), "kind": "audio", "seconds": 0.5, "width": 0, "height": 0, "has_audio": true}
        ],
        "tracks": [
            {"layer": "graphics", "name": "graphics", "clips": [{"id": "g1", "file": 1, "start": 0, "frames": 60, "source_in": 0.0, "speed": 1.0, "opacity": 1.0}]},
            {"layer": "footage", "name": "footage", "clips": [
                {"id": "v1", "file": 0, "start": 10, "frames": 20, "source_in": 0.2, "speed": 1.0, "box": [64.0, 0.0, 64.0, 36.0], "fit": "cover", "opacity": 0.5},
                {"id": "v2", "file": 0, "start": 40, "frames": 10, "source_in": 0.2, "speed": 2.0, "box": [0.0, 0.0, 128.0, 72.0], "fit": "cover", "opacity": 1.0}
            ]},
            {"layer": "under", "name": "under", "clips": [{"id": "u1", "file": 2, "start": 0, "frames": 60, "source_in": 0.0, "speed": 1.0, "opacity": 1.0}]},
            {"layer": "sound", "name": "sound", "clips": [{"id": "a1", "file": 3, "start": 5, "frames": 12, "source_in": 0.1, "speed": 1.0, "opacity": 1.0, "gain_db": -12.0}]}
        ],
        "notes": []
    })
}

fn fake_raea(dir: &Path) -> PathBuf {
    let f = dir.join("fixture");
    let script = format!(
        r#"#!/bin/sh
F="{f}"
printf '%s\n' "$*" >> "{calls}"
case "$1" in
  --version) echo "9.9.9-test"; exit 0;;
  export)
    [ "$2" = "layers" ] || exit 64
    out=""; prev=""
    for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
    [ -f "$F/export-fails" ] && {{ echo "REFUSED  no layers for this one" >&2; exit 2; }}
    mkdir -p "$out"
    cp "$F/layer.mp4" "$out/graphics.prores.mov"; cp "$F/layer.mp4" "$out/under.prores.mov"; cp "$F/layer.mp4" "$out/reference.mp4"
    cp "$F/layers.json" "$out/layers.json"; exit 0;;
esac
echo "unknown verb $1" >&2; exit 64
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

fn bench(name: &str) -> Bench {
    let turn = run::LIST_TURN.lock().unwrap_or_else(PoisonError::into_inner);
    let dir = std::env::temp_dir().join(format!("fc-layers-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("comp").join("assets")).unwrap();
    std::fs::create_dir_all(dir.join("fixture")).unwrap();
    std::fs::write(dir.join("comp").join("assets").join("clip.mp4"), movie()).unwrap();
    std::fs::write(dir.join("comp").join("assets").join("tone.wav"), wav()).unwrap();
    std::fs::write(dir.join("comp").join("edit.tsx"), "// a composition; the stand-in never reads it\n").unwrap();
    std::fs::write(dir.join("fixture").join("layer.mp4"), movie()).unwrap();
    std::fs::write(dir.join("fixture").join("layers.json"), layers_json(&dir).to_string()).unwrap();
    let raea = fake_raea(&dir);
    let manifest = run::manifest_path().unwrap();
    std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    std::fs::write(&manifest, format!(r#"{{"providers":[{{"id":"raea","kind":"composition","name":"RAEA","command":["{}"]}}]}}"#, raea.display())).unwrap();
    Bench { _turn: turn, dir, s: Session::default() }
}

impl Bench {
    fn source(&self) -> String {
        self.dir.join("comp").join("edit.tsx").to_string_lossy().into_owned()
    }
    fn out(&self, name: &str) -> PathBuf {
        self.dir.join("fx").join(name)
    }
    fn import(&mut self, name: &str, extra: Value) -> crate::Result<Value> {
        let mut p = json!({"source": self.source(), "composition": "edit", "output": self.out(name), "wait": true});
        for (k, v) in extra.as_object().unwrap() {
            p[k] = v.clone();
        }
        self.s.execute("composition.import", p)
    }
    fn set_layers(&self, v: &Value) {
        std::fs::write(self.dir.join("fixture").join("layers.json"), v.to_string()).unwrap();
    }
}

fn effect_param(s: &Session, clip: u64, effect: &str, param: &str) -> String {
    let q = s.active_sequence().unwrap();
    let (_, it) = q.find_item(ClipId(clip)).unwrap();
    format!("{:?}", it.effects.iter().find(|e| e.effect == effect).unwrap().params[param].value)
}

fn frames(n: i64) -> i64 {
    FrameRate::FPS_30.tick_of(n).0
}

#[test]
fn each_element_has_one_owner_in_a_new_sequence() {
    let mut b = bench("new");
    let undo0 = b.s.history.undo.len();
    let v = b.import("Edit layers v1", json!({})).unwrap();
    let calls = std::fs::read_to_string(b.dir.join("calls.log")).unwrap();
    assert!(calls.contains("export layers") && calls.contains("--alpha-codec prores") && calls.contains("--composition edit"), "{calls}");
    // a sequence of the composition's size and rate
    let q = b.s.active_sequence().unwrap();
    assert_eq!((q.settings.width, q.settings.height), (128, 72));
    assert_eq!(q.settings.frame_rate, FrameRate::FPS_30);
    assert_eq!(b.s.history.undo.len(), undo0 + 1, "one undo step");
    let placed = v["placed"].as_array().unwrap();
    let by = |id: &str| placed.iter().filter(|p| p["layer"] == id).cloned().collect::<Vec<_>>();
    // lowest layer lowest: under on V1, footage on V2, graphics on V3, sounds on audio tracks
    assert_eq!(by("under")[0]["track"], "V1");
    assert_eq!(by("footage")[0]["track"], "V2");
    assert_eq!(by("graphics")[0]["track"], "V3");
    assert_eq!(by("sound")[0]["track"], "A2");
    // the drawn layers carry no sound; the footage brings its own on A1
    assert!(by("under")[0]["clips"].as_array().unwrap().len() == 1 && by("graphics")[0]["clips"].as_array().unwrap().len() == 1);
    let footage = &by("footage")[0];
    assert_eq!(footage["audioTrack"], "A1");
    assert_eq!(footage["clips"].as_array().unwrap().len(), 2, "the footage's own sound: {footage}");
    assert_eq!(footage["drawn"], false);
    // at its time and length, from its trim
    assert_eq!(footage["start"].as_i64(), Some(frames(10)));
    assert_eq!(footage["duration"].as_i64(), Some(frames(20)));
    let clip = footage["clip"].as_u64().unwrap();
    {
        let q = b.s.active_sequence().unwrap();
        let (_, it) = q.find_item(ClipId(clip)).unwrap();
        assert_eq!(it.source_in, Tick::from_seconds_f64(0.2));
    }
    // half opacity, scaled into its box (half size, centred on the box)
    assert!(effect_param(&b.s, clip, "opacity", "opacity").contains("50.0"), "{}", effect_param(&b.s, clip, "opacity", "opacity"));
    assert!(effect_param(&b.s, clip, "motion", "scale").contains("50.0"));
    let pos = effect_param(&b.s, clip, "motion", "position");
    assert!(pos.contains("96.0") && pos.contains("18.0"), "{pos}");
    // at double speed it still covers its frames
    let fast = &by("footage")[1];
    assert_eq!(fast["start"].as_i64(), Some(frames(40)));
    assert_eq!(fast["duration"].as_i64(), Some(frames(10)), "{fast}");
    // the sound at its gain and time
    let sound = &by("sound")[0];
    assert_eq!(sound["start"].as_i64(), Some(frames(5)));
    assert!(effect_param(&b.s, sound["clip"].as_u64().unwrap(), "volume", "level").contains("-12.0"));
    // what RAEA wrote stays in the folder: its layers, its reference render, its log
    let out = b.out("Edit layers v1");
    for f in ["layers.json", "graphics.prores.mov", "under.prores.mov", "reference.mp4", "raea.log"] {
        assert!(out.join(f).is_file(), "{f}");
    }
    assert!(!out.join(".raea-layers").exists());
    assert_eq!(v["reference"], out.join("reference.mp4").to_string_lossy().as_ref());
    assert!(v["approximated"].as_array().unwrap().is_empty(), "{v}");
    // RAEA's movies here have no alpha: said, not refused
    assert!(v["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("graphics")), "{v}");
    assert!(!v["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("(under)")), "under is opaque by design");
    b.s.undo().unwrap();
}

#[test]
fn into_the_open_sequence_on_tracks_of_their_own() {
    let mut b = bench("place");
    b.s.execute("file.openDemoProject", json!({})).unwrap();
    // the footage is in the project already: it is reused, not imported twice
    let had = b.s.execute("file.import", json!({"paths": [b.dir.join("comp/assets/clip.mp4")]})).unwrap()["items"][0].as_u64().unwrap();
    let (v0, a0, clips0) = {
        let q = b.s.active_sequence().unwrap();
        (q.video_tracks.len(), q.audio_tracks.len(), q.video_tracks.iter().chain(&q.audio_tracks).map(|t| t.items.len()).sum::<usize>())
    };
    let v = b.import("Edit layers v1", json!({"place": {"seconds": 2.0}})).unwrap();
    let q = b.s.active_sequence().unwrap();
    assert_eq!((q.video_tracks.len(), q.audio_tracks.len()), (v0 + 3, a0 + 2));
    let old: usize = q.video_tracks.iter().take(v0).chain(q.audio_tracks.iter().take(a0)).map(|t| t.items.len()).sum();
    assert_eq!(old, clips0, "nothing that was there moved or went");
    let placed = v["placed"].as_array().unwrap();
    let footage = placed.iter().find(|p| p["layer"] == "footage").unwrap();
    assert_eq!(footage["item"].as_u64(), Some(had));
    assert_eq!(footage["track"], format!("V{}", v0 + 2));
    let rate = b.s.sequence_rate();
    let at = Tick::from_seconds_f64(2.0) + FrameRate::FPS_30.tick_of(10);
    assert_eq!(footage["start"].as_i64(), Some(rate.snap_nearest(at).0));
    // a sequence of another size: the drawn layers fill it, the box scales with it
    let graphics = placed.iter().find(|p| p["layer"] == "graphics").unwrap();
    let (w, _) = (q.settings.width, q.settings.height);
    let scale = effect_param(&b.s, graphics["clip"].as_u64().unwrap(), "motion", "scale");
    assert!(!scale.contains("100.0"), "filled to {w} wide: {scale}");
}

#[test]
fn a_bad_layers_file_keeps_the_export_and_imports_nothing() {
    let mut b = bench("bad");
    let mut l = layers_json(&b.dir);
    l["files"][1]["path"] = json!("../escape.mov");
    b.set_layers(&l);
    let items0 = b.s.project.items.len();
    let e = b.import("Bad v1", json!({})).unwrap_err().to_string();
    assert!(e.contains("leaves the export folder") && e.contains("kept"), "{e}");
    assert!(b.out("Bad v1").join("layers.json").is_file());
    assert_eq!(b.s.project.items.len(), items0);
    assert!(b.s.active_sequence().is_none(), "no sequence was made");
    // a failed export frees the folder name
    std::fs::write(b.dir.join("fixture").join("export-fails"), "").unwrap();
    let e = b.import("Bad v2", json!({})).unwrap_err().to_string();
    assert!(e.contains("no layers for this one"), "{e}");
    assert!(!b.out("Bad v2").exists());
}

#[test]
fn every_index_number_and_path_in_layers_json_is_checked() {
    let b = bench("parse");
    let dir = b.dir.join("export");
    std::fs::create_dir_all(&dir).unwrap();
    for f in ["graphics.prores.mov", "under.prores.mov"] {
        std::fs::write(dir.join(f), b"x").unwrap();
    }
    let good = layers_json(&b.dir);
    let plan = parse_layers(&good, &dir).unwrap();
    assert_eq!(plan.tracks.iter().map(|t| t.layer.as_str()).collect::<Vec<_>>(), ["under", "footage", "graphics", "sound"]);
    assert!(plan.files[1].made && !plan.files[0].made);
    let cases: Vec<(&str, Value, &str)> = vec![
        ("/version", json!(2), "version"),
        ("/fps", json!(0), "`fps`"),
        ("/fps", json!(1e300), "`fps`"),
        ("/width", json!(0), "`width`"),
        ("/frames", json!(u64::MAX), "`frames`"),
        ("/files/1/path", json!("../x.mov"), "leaves the export folder"),
        ("/files/1/path", json!("/nowhere/x.mov"), "not there"),
        ("/files/1/path", json!(""), "leaves the export folder"),
        ("/files/1/kind", json!("hologram"), "unknown kind"),
        ("/tracks/0/clips/0/file", json!(99), "names no file"),
        ("/tracks/0/clips/0/file", json!(-1), "names no file"),
        ("/tracks/0/clips/0/start", json!(-5), "`start`"),
        ("/tracks/0/clips/0/frames", json!(0), "`frames`"),
        ("/tracks/0/clips/0/speed", json!(0), "`speed`"),
        ("/tracks/0/clips/0/speed", json!(1e9), "`speed`"),
        ("/tracks/0/clips/0/opacity", json!(2), "`opacity`"),
        ("/tracks/0/clips/0/source_in", json!(-1), "`source_in`"),
        ("/tracks/0/clips/0/gain_db", json!(1e6), "`gain_db`"),
        ("/tracks/1/clips/0/box", json!([0, 0, -1, 5]), "`box`"),
        ("/tracks/1/clips/0/box", json!([0, 0]), "`box`"),
        ("/tracks/1/clips/1/file", json!(3), "mixes"),
    ];
    for (at, value, why) in cases {
        let mut v = good.clone();
        let (parent, key) = at.rsplit_once('/').unwrap();
        match v.pointer_mut(parent).unwrap() {
            Value::Object(o) => {
                o.insert(key.to_string(), value.clone());
            }
            Value::Array(a) => a[key.parse::<usize>().unwrap()] = value.clone(),
            other => panic!("{other}"),
        }
        let e = parse_layers(&v, &dir).unwrap_err();
        assert!(e.contains(why), "{at} = {value}: {e}");
    }
    let mut many = good.clone();
    many["tracks"] = json!(vec![good["tracks"][0].clone(); MAX_TRACKS + 1]);
    assert!(parse_layers(&many, &dir).unwrap_err().contains("at most"));
    let mut clips = good.clone();
    clips["tracks"][0]["clips"] = json!(vec![good["tracks"][0]["clips"][0].clone(); MAX_CLIPS + 1]);
    assert!(parse_layers(&clips, &dir).unwrap_err().contains("more than"));
}

#[test]
fn hostile_parameters_are_refused_before_raea_runs() {
    let mut b = bench("hostile");
    let src = b.source();
    let out = b.out("h");
    let html = b.dir.join("comp").join("page.html");
    std::fs::write(&html, "<!-- page -->").unwrap();
    for (p, why) in [
        (json!({"source": html, "output": out}), "HTML"),
        (json!({"source": src}), "`output` is needed"),
        (json!({"source": src, "output": out, "alphaCodec": "h264"}), "prores or png"),
        (json!({"source": src, "output": out, "composition": "-o"}), "composition id"),
        (json!({"source": src, "output": out, "place": {"time": 5}}), "open sequence"),
        (json!({"source": src, "output": "relative"}), "absolute"),
    ] {
        let e = b.s.execute("composition.import", p.clone()).unwrap_err().to_string();
        assert!(e.contains(why), "{p}: {e}");
    }
    assert!(!b.dir.join("calls.log").exists(), "RAEA never ran");
    assert!(!out.exists());
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join("old.mov"), "x").unwrap();
    let e = b.import("h", json!({})).unwrap_err().to_string();
    assert!(e.contains("already holds files"), "{e}");
}

#[test]
fn inside_paths_stay_inside() {
    for (p, ok) in
        [("graphics.mov", true), ("sub/graphics.mov", true), ("./g.mov", true), ("../g.mov", false), ("a/../../g.mov", false), ("/abs.mov", false), ("", false)]
    {
        assert_eq!(inside(p), ok, "{p}");
    }
}
