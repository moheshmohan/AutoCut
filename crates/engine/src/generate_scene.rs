//! A 3D scene renderer: a `kind: "scene"` entry of the provider list with a `blender` block,
//! driven by `generate.render3d`. Blender runs as a separate program on this computer, without
//! its window:
//!
//! ```text
//! blender --background --factory-startup --disable-autoexec --python-exit-code 3 \
//!     [scene.blend] --python <AutoCut's script> -- '<request as JSON>'
//! ```
//!
//! The script is AutoCut's own ([`SCRIPT`], `scene_render.py` beside this file, MIT OR
//! Apache-2.0), written to the data folder before a render. It sets the size, rate, frame range,
//! transparency and format, sets named text objects and objects as the request says, keys a
//! camera track onto the scene's camera (a match-move), builds a plain stage for a title when no
//! scene file is given, and prints `progress k/n` after each frame. AutoCut imports the frames
//! (PNG or EXR with alpha) as an image sequence; optional Cryptomatte passes go to a multilayer
//! EXR a frame beside them. Nothing of Blender is linked, shipped or read.
//!
//! ```json
//! {"id": "blender", "kind": "scene", "name": "Blender (headless)",
//!  "blender": {"program": "/Applications/Blender.app/Contents/MacOS/Blender", "engine": "eevee"},
//!  "estimateSeconds": 120, "timeoutSeconds": 3600}
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// AutoCut's script for Blender.
pub const SCRIPT: &str = include_str!("scene_render.py");
/// Its file name in `<data dir>/generation/`.
pub const SCRIPT_NAME: &str = "scene_render.py";
/// The highest frame number rendered (frames are named with four digits).
pub const MAX_FRAME: u32 = 9999;
/// The most a request to the script may weigh (it is one argument).
const MAX_REQUEST: usize = 16 * 1024;
const MAX_TEXTS: usize = 32;
/// The longest text a text object is given (the script takes no more).
pub const MAX_TEXT: usize = 1000;
const MAX_OBJECTS: usize = 64;
/// The most a camera track file may weigh (read only where files are), and its most frames.
#[cfg(not(target_arch = "wasm32"))]
const TRACK_MAX: u64 = 16 * 1024 * 1024;
const TRACK_FRAMES: usize = 100_000;
/// The largest coordinate, angle or scale taken.
const LIMIT: f64 = 1.0e6;

/// The `blender` block of a scene entry.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Config {
    /// Blender's program (an absolute path).
    pub program: String,
    /// The scene rendered when the request names none (an absolute `.blend` path); without
    /// one, a plain stage for a title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scene: Option<String>,
    /// A script of the user's own instead of AutoCut's (an absolute `.py` path); it is given
    /// the same request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<String>,
    /// `eevee`, `cycles` or `workbench` (default: the scene's own; EEVEE for the plain stage).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples: Option<u32>,
    /// `png` (default) or `exr`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// `gpu` (default) or `cpu`, for Cycles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// Start Blender with its factory settings (default true): no add-ons or preferences of the
    /// user's own. A scene that needs an add-on sets this false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factory_startup: Option<bool>,
}

fn short(t: &str, n: usize) -> String {
    t.chars().take(n).collect()
}

/// An absolute path with the given ending (lower case), bounded.
fn abs_with(path: &str, ending: &str) -> bool {
    let p = std::path::Path::new(path);
    p.is_absolute() && !path.contains('\0') && path.len() <= 4096 && path.to_ascii_lowercase().ends_with(ending)
}

/// The block checked when the list is read: an absolute program, optional absolute `.blend`
/// and `.py` files, known engine, format and device.
pub fn check_config(c: &Config) -> Result<(), String> {
    let program = std::path::Path::new(&c.program);
    if !program.is_absolute() || c.program.contains(['\0', '{']) || c.program.len() > 4096 {
        return Err("`blender.program` must be the absolute path of Blender's program".into());
    }
    if c.scene.as_deref().is_some_and(|s| !abs_with(s, ".blend")) {
        return Err("`blender.scene` must be the absolute path of a .blend file".into());
    }
    if c.script.as_deref().is_some_and(|s| !abs_with(s, ".py")) {
        return Err("`blender.script` must be the absolute path of a .py file".into());
    }
    if c.engine.as_deref().is_some_and(|e| !matches!(e, "eevee" | "cycles" | "workbench")) {
        return Err("`blender.engine` is eevee, cycles or workbench".into());
    }
    if c.samples.is_some_and(|n| n == 0 || n > 65_536) {
        return Err("`blender.samples` is 1 to 65536".into());
    }
    if c.format.as_deref().is_some_and(|f| !matches!(f, "png" | "exr")) {
        return Err("`blender.format` is png or exr".into());
    }
    if c.device.as_deref().is_some_and(|d| !matches!(d, "gpu" | "cpu")) {
        return Err("`blender.device` is gpu or cpu".into());
    }
    Ok(())
}

/// A scene file a request names: an absolute `.blend` path that is a file.
pub fn scene_file(path: &str) -> Result<String, String> {
    if !abs_with(path, ".blend") {
        return Err(format!("`scene` must be the absolute path of a .blend file, not {}", short(path, 160)));
    }
    if !cfg!(target_arch = "wasm32") && !std::path::Path::new(path).is_file() {
        return Err(format!("the scene {} is not on this computer", short(path, 160)));
    }
    Ok(path.to_string())
}

/// A finite number within ±[`LIMIT`].
fn finite(v: &Value, what: &str) -> Result<f64, String> {
    match v.as_f64() {
        Some(x) if x.is_finite() && x.abs() <= LIMIT => Ok(x),
        _ => Err(format!("{what} must be a number from -{LIMIT:e} to {LIMIT:e}")),
    }
}

fn vector(v: &Value, what: &str, n: usize) -> Result<Vec<f64>, String> {
    let list = v.as_array().filter(|a| a.len() == n).ok_or_else(|| format!("{what} must be a list of {n} numbers"))?;
    list.iter().map(|x| finite(x, what)).collect()
}

fn name_ok(name: &str) -> bool {
    let n = name.chars().count();
    (1..=64).contains(&n) && !name.chars().any(char::is_control)
}

/// `texts`: `{object name: text}`, bounded.
pub fn texts_p(p: &Value) -> Result<serde_json::Map<String, Value>, String> {
    let Some(v) = p.get("texts").filter(|v| !v.is_null()) else { return Ok(Default::default()) };
    let m = v.as_object().ok_or("`texts` must be {object name: text}")?;
    if m.len() > MAX_TEXTS {
        return Err(format!("`texts` names at most {MAX_TEXTS} text objects"));
    }
    let mut out = serde_json::Map::new();
    for (k, t) in m {
        let t = t.as_str().filter(|t| t.chars().count() <= MAX_TEXT && !t.contains('\0'));
        let Some(t) = t.filter(|_| name_ok(k)) else {
            return Err(format!("`texts`: a name is 1 to 64 characters and a text at most {MAX_TEXT} ({})", short(k, 40)));
        };
        out.insert(k.clone(), json!(t));
    }
    Ok(out)
}

/// `objects`: `{name: {location?, rotation? (degrees), scale? (a number or three), hide?}}`.
pub fn objects_p(p: &Value) -> Result<serde_json::Map<String, Value>, String> {
    let Some(v) = p.get("objects").filter(|v| !v.is_null()) else { return Ok(Default::default()) };
    let m = v.as_object().ok_or("`objects` must be {object name: {location, rotation, scale, hide}}")?;
    if m.len() > MAX_OBJECTS {
        return Err(format!("`objects` names at most {MAX_OBJECTS} objects"));
    }
    let mut out = serde_json::Map::new();
    for (name, o) in m {
        let what = short(name, 40);
        let o = o.as_object().filter(|_| name_ok(name)).ok_or_else(|| format!("`objects`: {what} must be {{location, rotation, scale, hide}}"))?;
        let mut clean = serde_json::Map::new();
        for (k, v) in o {
            let value = match k.as_str() {
                "location" | "rotation" => json!(vector(v, &format!("{what}.{k}"), 3)?),
                "scale" if v.is_number() => {
                    let s = finite(v, &format!("{what}.scale"))?;
                    json!([s, s, s])
                }
                "scale" => json!(vector(v, &format!("{what}.scale"), 3)?),
                "hide" => json!(v.as_bool().ok_or_else(|| format!("{what}.hide must be true or false"))?),
                other => return Err(format!("`objects`: {what} takes location, rotation, scale and hide, not {}", short(other, 40))),
            };
            clean.insert(k.clone(), value);
        }
        out.insert(name.clone(), Value::Object(clean));
    }
    Ok(out)
}

/// A camera track checked before anything runs (the script checks it again): `{"sensorWidth"?,
/// "frames": [{"frame", "position": [x, y, z], "rotation": [x, y, z] (degrees) | "quaternion":
/// [w, x, y, z], "focalLength"?}]}`, metres on Blender's axes (Z up). The number of its frames.
#[cfg(not(target_arch = "wasm32"))]
pub fn check_track(path: &str) -> Result<usize, String> {
    use std::io::Read as _;
    if !abs_with(path, ".json") {
        return Err(format!("`camera` must be the absolute path of a camera track (.json), not {}", short(path, 160)));
    }
    let f = std::fs::File::open(path).map_err(|e| format!("the camera track {}: {e}", short(path, 160)))?;
    let mut text = String::new();
    f.take(TRACK_MAX + 1).read_to_string(&mut text).map_err(|e| format!("the camera track: {e}"))?;
    if text.len() as u64 > TRACK_MAX {
        return Err("the camera track is larger than 16 MiB".into());
    }
    track_of(&text)
}

#[cfg(target_arch = "wasm32")]
pub fn check_track(_: &str) -> Result<usize, String> {
    Err("scenes render on the desktop app".into())
}

/// The frames of a camera track's text, each checked.
pub fn track_of(text: &str) -> Result<usize, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("the camera track is not JSON: {e}"))?;
    let frames = v.get("frames").and_then(Value::as_array).ok_or("the camera track has no `frames` list")?;
    if frames.is_empty() || frames.len() > TRACK_FRAMES {
        return Err(format!("the camera track has 1 to {TRACK_FRAMES} frames"));
    }
    if let Some(s) = v.get("sensorWidth")
        && !s.as_f64().is_some_and(|x| (1.0..=200.0).contains(&x))
    {
        return Err("the camera track's sensorWidth is 1 to 200 mm".into());
    }
    for (k, fr) in frames.iter().enumerate() {
        let at = |w: &str| format!("camera frame {k}: {w}");
        fr.get("frame").and_then(Value::as_i64).filter(|f| f.abs() <= 1_000_000).ok_or_else(|| at("`frame` must be a whole number"))?;
        vector(fr.get("position").unwrap_or(&Value::Null), &at("position"), 3)?;
        match (fr.get("quaternion"), fr.get("rotation")) {
            (Some(q), _) => {
                let q = vector(q, &at("quaternion"), 4)?;
                if q.iter().map(|x| x * x).sum::<f64>() < 1.0e-12 {
                    return Err(at("the quaternion is zero"));
                }
            }
            (None, Some(r)) => {
                vector(r, &at("rotation"), 3)?;
            }
            (None, None) => return Err(at("it has no rotation or quaternion")),
        }
        if let Some(f) = fr.get("focalLength")
            && !f.as_f64().is_some_and(|x| (1.0..=5000.0).contains(&x))
        {
            return Err(at("focalLength is 1 to 5000 mm"));
        }
    }
    Ok(frames.len())
}

/// What the script is asked to render.
pub struct Request<'a> {
    /// The frames' folder and name: `<folder>/frame_`.
    pub output: String,
    pub format: &'a str,
    pub size: (u32, u32),
    pub fps: f64,
    pub frames: (u32, u32),
    pub transparent: bool,
    pub cryptomatte: bool,
    pub seed: u64,
    pub text: Option<String>,
    pub texts: serde_json::Map<String, Value>,
    pub objects: serde_json::Map<String, Value>,
    pub camera: Option<String>,
    pub scene: bool,
}

/// The request as the script's one argument (bounded).
pub fn request_json(c: &Config, r: &Request) -> Result<String, String> {
    let v = json!({
        "version": 1, "output": r.output, "format": r.format, "width": r.size.0, "height": r.size.1,
        "fps": r.fps, "frameStart": r.frames.0, "frameEnd": r.frames.1, "transparent": r.transparent,
        "cryptomatte": r.cryptomatte, "engine": c.engine, "samples": c.samples, "device": c.device.as_deref().unwrap_or("gpu"),
        // (Cycles takes a 31-bit seed)
        "seed": r.seed % 2_147_483_647, "text": r.text, "texts": r.texts, "objects": r.objects, "camera": r.camera, "scene": r.scene,
    });
    let text = v.to_string();
    if text.len() > MAX_REQUEST {
        return Err("the request is too large for the renderer (fewer or shorter texts and objects)".into());
    }
    Ok(text)
}

/// Blender's command line for a render.
pub fn argv(c: &Config, scene: Option<&str>, script: &str, request: &str) -> Vec<String> {
    let mut a = vec![c.program.clone(), "--background".to_string()];
    if c.factory_startup != Some(false) {
        a.push("--factory-startup".into());
    }
    // a scene's own scripts never run; a failing script ends Blender with status 3
    a.extend(["--disable-autoexec", "--python-exit-code", "3"].map(String::from));
    if let Some(s) = scene {
        a.push(s.to_string());
    }
    a.extend(["--python".to_string(), script.to_string(), "--".to_string(), request.to_string()]);
    a
}

/// AutoCut's script, written to `<data dir>/generation/scene_render.py` (again when it differs:
/// the file there is always this version's); its path.
#[cfg(not(target_arch = "wasm32"))]
pub fn ensure_script() -> Result<String, String> {
    let dir = crate::autosave::default_data_dir().ok_or("there is no data folder to put the renderer's script in")?.join("generation");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(SCRIPT_NAME);
    let same = std::fs::metadata(&path).is_ok_and(|m| m.len() == SCRIPT.len() as u64) && std::fs::read(&path).is_ok_and(|b| b == SCRIPT.as_bytes());
    if !same {
        // written beside and moved, so a render that starts meanwhile never reads half a file
        let tmp = dir.join(format!("{SCRIPT_NAME}.{}.tmp", std::process::id()));
        std::fs::write(&tmp, SCRIPT).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            format!("{}: {e}", path.display())
        })?;
    }
    Ok(path.to_string_lossy().into_owned())
}

#[cfg(target_arch = "wasm32")]
pub fn ensure_script() -> Result<String, String> {
    Err("scenes render on the desktop app".into())
}

/// Blender's version line (`Blender 4.2.1 LTS`), asked without rendering anything (at most 20 s).
#[cfg(not(target_arch = "wasm32"))]
pub fn version(c: &Config) -> Result<String, String> {
    use std::io::Read as _;
    use std::process::{Command, Stdio};
    let mut child = Command::new(&c.program)
        .args(["--background", "--factory-startup", "--version"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("{} does not start: {e}", c.program))?;
    let t0 = web_time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if t0.elapsed().as_secs() < 20 => std::thread::sleep(std::time::Duration::from_millis(50)),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("{} did not say its version in 20 s", c.program));
            }
            Err(e) => return Err(e.to_string()),
        }
    }
    let mut text = String::new();
    if let Some(out) = child.stdout.take() {
        let _ = out.take(64 * 1024).read_to_string(&mut text);
    }
    text.lines().map(str::trim).find(|l| l.starts_with("Blender")).map(|l| short(l, 80)).ok_or_else(|| format!("{} did not say it is Blender", c.program))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(json: &str) -> Config {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn a_hostile_block_is_refused() {
        assert!(check_config(&cfg(r#"{"program": "/opt/blender/blender"}"#)).is_ok());
        for (bad, why) in [
            (r#"{"program": "blender"}"#, "absolute"),
            (r#"{"program": "/opt/{x}"}"#, "absolute"),
            (r#"{"program": "/b", "scene": "scene.blend"}"#, ".blend"),
            (r#"{"program": "/b", "scene": "/s/scene.txt"}"#, ".blend"),
            (r#"{"program": "/b", "script": "s.py"}"#, ".py"),
            (r#"{"program": "/b", "engine": "octane"}"#, "eevee"),
            (r#"{"program": "/b", "samples": 0}"#, "samples"),
            (r#"{"program": "/b", "format": "tiff"}"#, "png or exr"),
            (r#"{"program": "/b", "device": "tpu"}"#, "gpu or cpu"),
        ] {
            let e = check_config(&cfg(bad)).unwrap_err();
            assert!(e.contains(why), "{bad}: {e}");
        }
        assert!(serde_json::from_str::<Config>(r#"{"program": "/b", "python": "x"}"#).is_err(), "unknown fields");
    }

    #[test]
    fn the_command_line_keeps_a_scene_s_scripts_off() {
        let c = cfg(r#"{"program": "/opt/blender"}"#);
        let a = argv(&c, Some("/s/a.blend"), "/d/scene_render.py", "{}");
        assert_eq!(
            a,
            [
                "/opt/blender",
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--python-exit-code",
                "3",
                "/s/a.blend",
                "--python",
                "/d/scene_render.py",
                "--",
                "{}"
            ]
        );
        let own = cfg(r#"{"program": "/opt/blender", "factoryStartup": false}"#);
        assert!(!argv(&own, None, "/d/s.py", "{}").contains(&"--factory-startup".to_string()));
    }

    #[test]
    fn texts_objects_and_camera_tracks_are_checked() {
        let p = json!({"texts": {"Title": "Hello"}, "objects": {"Cube": {"location": [0, 0, 1], "scale": 2, "hide": false}}});
        assert_eq!(texts_p(&p).unwrap()["Title"], "Hello");
        assert_eq!(objects_p(&p).unwrap()["Cube"]["scale"], json!([2.0, 2.0, 2.0]));
        for bad in [
            json!({"texts": "x"}),
            json!({"texts": {"": "x"}}),
            json!({"texts": {"T": 5}}),
            json!({"texts": {"T": "x".repeat(MAX_TEXT + 1)}}),
            json!({"texts": (0..40).map(|k| (format!("t{k}"), json!("x"))).collect::<serde_json::Map<_, _>>()}),
        ] {
            assert!(texts_p(&bad).is_err(), "{bad}");
        }
        for bad in [
            json!({"objects": {"Cube": {"location": [0, 0]}}}),
            json!({"objects": {"Cube": {"location": [0, 0, 1e9]}}}),
            json!({"objects": {"Cube": {"rotation": ["a", 0, 0]}}}),
            json!({"objects": {"Cube": {"hide": 1}}}),
            json!({"objects": {"Cube": {"delete": true}}}),
            json!({"objects": {"Cube": 3}}),
        ] {
            assert!(objects_p(&bad).is_err(), "{bad}");
        }
        assert_eq!(track_of(r#"{"frames": [{"frame": 1, "position": [0, -8, 1], "rotation": [90, 0, 0], "focalLength": 35}]}"#), Ok(1));
        assert_eq!(track_of(r#"{"sensorWidth": 36, "frames": [{"frame": 1, "position": [0, 0, 0], "quaternion": [1, 0, 0, 0]}]}"#), Ok(1));
        for bad in [
            "not json",
            r#"{"frames": []}"#,
            r#"{"frames": [{"frame": 1, "position": [0, 0, 0]}]}"#,
            r#"{"frames": [{"frame": 1.5, "position": [0, 0, 0], "rotation": [0, 0, 0]}]}"#,
            r#"{"frames": [{"frame": 1, "position": [0, 0], "rotation": [0, 0, 0]}]}"#,
            r#"{"frames": [{"frame": 1, "position": [0, 0, 0], "quaternion": [0, 0, 0, 0]}]}"#,
            r#"{"frames": [{"frame": 1, "position": [0, 0, 0], "rotation": [0, 0, 0], "focalLength": 0}]}"#,
            r#"{"sensorWidth": 900, "frames": [{"frame": 1, "position": [0, 0, 0], "rotation": [0, 0, 0]}]}"#,
        ] {
            assert!(track_of(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_scene_entry_is_its_blender_block_alone() {
        use crate::generate_run::{Kind, parse};
        let wrap = |body: &str| format!(r#"{{"providers":[{body}]}}"#);
        let ok = parse(&wrap(r#"{"id":"b","kind":"scene","name":"B","blender":{"program":"/opt/blender"},"capabilities":{"fps":[24],"transparent":true}}"#))
            .unwrap();
        assert_eq!((ok[0].kind, ok[0].kind.name(), ok[0].kind.extension()), (Kind::Scene, "scene", "png"));
        for (bad, why) in [
            (r#"{"id":"b","kind":"scene","name":"B"}"#, "needs one"),
            (r#"{"id":"b","kind":"image","name":"B","command":["/bin/x","{out}"],"blender":{"program":"/opt/blender"}}"#, "goes with"),
            (r#"{"id":"b","kind":"scene","name":"B","blender":{"program":"blender"}}"#, "absolute"),
            (r#"{"id":"b","kind":"scene","name":"B","command":["/bin/x","{out}"],"blender":{"program":"/opt/blender"}}"#, "alone"),
            (r#"{"id":"b","kind":"scene","name":"B","blender":{"program":"/opt/blender"},"extension":"jpg"}"#, "extension"),
            (r#"{"id":"b","kind":"scene","name":"B","blender":{"program":"/opt/blender"},"timeoutSeconds":0}"#, "positive"),
            (r#"{"id":"b","kind":"scene","name":"B","blender":{"program":"/opt/blender"},"capabilities":{"references":2}}"#, "not something a scene tool"),
            (r#"{"id":"r","kind":"composition","name":"R","blender":{"program":"/opt/blender"}}"#, "blender"),
        ] {
            let e = parse(&wrap(bad)).unwrap_err();
            assert!(e.contains(why), "{bad}: {e}");
        }
    }

    /// A render is a run of Blender like any tool's: stopped, or given up on at its time limit.
    #[cfg(unix)]
    #[test]
    fn a_render_can_be_stopped_and_has_a_time_limit() {
        use std::os::unix::fs::PermissionsExt as _;
        use std::sync::atomic::AtomicBool;
        let dir = std::env::temp_dir().join(format!("fc-scene-slow-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let program = dir.join("blender");
        std::fs::write(&program, "#!/bin/sh\necho 'Fra:1 Mem:12M | Rendering'\nsleep 30\n").unwrap();
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        let list = format!(r#"{{"providers":[{{"id":"b","kind":"scene","name":"Slow","blender":{{"program":"{}"}},"timeoutSeconds":1}}]}}"#, program.display());
        let p = crate::generate_run::parse(&list).unwrap().remove(0);
        let c = p.blender.clone().unwrap();
        let args = argv(&c, None, "/d/scene_render.py", "{}");
        let out = dir.join("frames").join("frame_0001.png");
        let t0 = web_time::Instant::now();
        let stopped = crate::generate_run::run(&p, &args, &out, &AtomicBool::new(true), &mut |_, _| {}).unwrap_err();
        assert!(stopped.contains("stopped"), "{stopped}");
        let late = crate::generate_run::run(&p, &args, &out, &AtomicBool::new(false), &mut |_, _| {}).unwrap_err();
        assert!(late.contains("did not finish"), "{late}");
        assert!(t0.elapsed().as_secs() < 10);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The script's own checking, run with a plain Python (no Blender) when one is on this
    /// computer; skipped otherwise.
    #[cfg(unix)]
    #[test]
    fn the_script_checks_its_request_without_blender() {
        let Ok(found) = std::process::Command::new("python3").arg("--version").output() else {
            eprintln!("no python3: the script's request handling is not run");
            return;
        };
        if !found.status.success() {
            eprintln!("no working python3: the script's request handling is not run");
            return;
        }
        let dir = std::env::temp_dir().join(format!("fc-scene-script-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join(SCRIPT_NAME);
        std::fs::write(&script, SCRIPT).unwrap();
        let track = dir.join("track.json");
        std::fs::write(
            &track,
            r#"{"frames": [{"frame": 1, "position": [0, -8, 1], "rotation": [90, 0, 0]}, {"frame": 24, "position": [1, -8, 1], "rotation": [90, 0, 5]}]}"#,
        )
        .unwrap();
        let check = |request: &str| {
            let out = std::process::Command::new("python3").arg("-I").arg(&script).args(["--", "--check", request]).current_dir(&dir).output().unwrap();
            (out.status.code(), String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned())
        };
        // what AutoCut sends is what the script reads
        let c = cfg(r#"{"program": "/opt/blender", "engine": "cycles", "samples": 32}"#);
        let r = Request {
            output: dir.join("frames").join("frame_").to_string_lossy().into_owned(),
            format: "png",
            size: (1920, 1080),
            fps: 29.97,
            frames: (1, 48),
            transparent: true,
            cryptomatte: false,
            seed: u64::MAX,
            text: Some("Hello \"there\"".into()),
            texts: Default::default(),
            objects: objects_p(&json!({"objects": {"Cube": {"scale": 2}}})).unwrap(),
            camera: Some(track.to_string_lossy().into_owned()),
            scene: false,
        };
        let (code, out, err) = check(&request_json(&c, &r).unwrap());
        assert_eq!(code, Some(0), "{err}");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["request"]["width"], 1920);
        assert_eq!(v["request"]["engine"], "cycles");
        assert_eq!(v["request"]["text"], "Hello \"there\"");
        assert_eq!(v["request"]["objects"]["Cube"]["scale"], json!([2.0, 2.0, 2.0]));
        assert_eq!(v["cameraFrames"], 2);
        assert_eq!(v["rate"][0], 30, "29.97 is Blender's 30 / 1.001");
        // what it refuses, it says, with status 2
        for (bad, why) in [
            ("not json", "not JSON"),
            (r#"{"output": "rel", "width": 64, "height": 64, "fps": 24}"#, "absolute"),
            (r#"{"output": "/o/frame_", "width": 64, "height": 64, "fps": 0}"#, "fps"),
            (r#"{"output": "/o/frame_", "width": 64, "height": 64, "fps": 24, "frameStart": 9, "frameEnd": 2}"#, "before"),
            (r#"{"output": "/o/frame_", "width": 64, "height": 64, "fps": 24, "cryptomatte": true}"#, "exr"),
            (r#"{"output": "/o/frame_", "width": 64, "height": 64, "fps": 24, "shell": "x"}"#, "unknown"),
            (r#"{"output": "/o/frame_", "width": 64, "height": 64, "fps": 24, "camera": "/nowhere/t.json"}"#, "camera track"),
            ("[1, 2]", "object"),
        ] {
            let (code, _, err) = check(bad);
            assert_eq!(code, Some(2), "{bad}");
            assert!(err.contains("autocut-error") && err.contains(why), "{bad}: {err}");
        }
        // run outside Blender, it says so
        let out = std::process::Command::new("python3")
            .arg("-I")
            .arg(&script)
            .args(["--", r#"{"output": "/o/frame_", "width": 64, "height": 64, "fps": 24}"#])
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&out.stderr).contains("inside Blender"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
