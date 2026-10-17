//! `cargo xtask live [--only a,b] [--google] [--picker] [--voice clone:NAME] [--clip FILE]`: the
//! opt-in live checks in one command, against this machine's real models, sign-ins and browser.
//! The offline gates (`ci`) never need any of them; this is what to run before trusting a build
//! with a real video.
//!
//! Stages, in order (`--only` picks some):
//! - `models`: the Ollama prompt classifier and frame describer (`AUTOCUT_LIVE`).
//! - `screen`: the screen agent's grounding model finds a drawn button (`AUTOCUT_SCREEN_LIVE`).
//! - `whisper`: AutoCut removes the silences of natural speech, transcribed by Whisper.
//! - `voice`: one line in a cloned voice (`--voice`, else the first natural clone), heard back by
//!   Whisper: at least half its words must come back.
//! - `analysis`: a clip (`--clip`, else one the analysis cache already knows shows a robot) is
//!   analysed twice; the second run must come from the analysis cache.
//! - `showcase`: a short narrated showcase of that clip, exported (and checked by its QA).
//! - `keychain` (with `--google`): the macOS Keychain keeps, moves in (from a `tokens.json`)
//!   and forgets a throwaway sign-in (`FILMCRAFT_LIVE_KEYCHAIN`; passes elsewhere).
//! - `album`, `upload` (with `--google`): an album in the signed-in Google Photos library
//!   ("FilmCraft live check", made once, reused after) and the showcase uploaded into it.
//! - `picker` (with `--picker`): the picker agent drives Chrome — two searches in one session —
//!   then `photos tidy`. It takes the browser lock itself; keep hands off the mouse meanwhile.
//!
//! Everything a stage makes (projects, exports, reports) goes to `<target>/live/<time>/`, with
//! each stage's output in `logs/<stage>.log` and the results in `report.json`; nothing goes to
//! `~/Movies`. The stages take the machine's locks themselves (`filmcraft-cli locks`), so the
//! check waits for, rather than collides with, another session's browser or GPU work.

use std::path::PathBuf;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

/// The album the Google stages use (made once, reused after).
const ALBUM: &str = "FilmCraft live check";
/// The subject the analysis and showcase stages look for.
const SUBJECT: &str = "robot";
/// Frames sampled per clip by the analysis stage (the analysis cache is keyed by it).
const FRAMES: &str = "3";
/// The share of a cloned line's words Whisper must hear back (as the showcase's own check).
const HEARD_ENOUGH: f64 = 0.5;

const STAGES: &[(&str, &str)] = &[
    ("models", "Ollama classifier and frame describer"),
    ("screen", "screen agent grounding"),
    ("whisper", "Whisper transcription + AutoCut silence removal"),
    ("voice", "a cloned line heard back"),
    ("analysis", "analysis cache hit"),
    ("showcase", "a short showcase built and exported"),
    ("keychain", "a throwaway sign-in kept in and moved into the Keychain"),
    ("album", "Google Photos album made or found"),
    ("upload", "the showcase uploaded into the album"),
    ("picker", "picker agent: two searches, then tidy"),
];

#[derive(Debug, Default, PartialEq)]
pub(crate) struct Options {
    pub only: Option<Vec<String>>,
    pub google: bool,
    pub picker: bool,
    pub voice: Option<String>,
    pub clip: Option<PathBuf>,
}

pub(crate) fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options::default();
    let mut it = args.iter();
    let usage = "live [--only a,b] [--google] [--picker] [--voice clone:NAME] [--clip FILE]";
    while let Some(a) = it.next() {
        match a.as_str() {
            "--only" => {
                let list: Vec<String> = it.next().ok_or(usage)?.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
                if let Some(bad) = list.iter().find(|s| !STAGES.iter().any(|(id, _)| id == s)) {
                    return Err(format!("live: no stage `{bad}` (stages: {})", STAGES.iter().map(|(id, _)| *id).collect::<Vec<_>>().join(", ")));
                }
                o.only = Some(list);
            }
            "--google" => o.google = true,
            "--picker" => o.picker = true,
            "--voice" => o.voice = Some(it.next().ok_or(usage)?.clone()),
            "--clip" => o.clip = Some(PathBuf::from(it.next().ok_or(usage)?)),
            other => return Err(format!("live: unknown `{other}` ({usage})")),
        }
    }
    Ok(o)
}

/// Whether stage `id` runs: named by `--only`, else the default ones and those its flag opts into.
pub(crate) fn wanted(o: &Options, id: &str) -> bool {
    match &o.only {
        Some(list) => list.iter().any(|s| s == id),
        None => match id {
            "keychain" | "album" | "upload" => o.google,
            "picker" => o.picker,
            _ => true,
        },
    }
}

/// How a stage went.
#[derive(Debug, Clone, PartialEq)]
enum Outcome {
    Pass(String),
    Fail(String),
    Skip(String),
}

impl Outcome {
    fn word(&self) -> &'static str {
        match self {
            Outcome::Pass(_) => "pass",
            Outcome::Fail(_) => "FAIL",
            Outcome::Skip(_) => "skip",
        }
    }

    fn detail(&self) -> &str {
        match self {
            Outcome::Pass(d) | Outcome::Fail(d) | Outcome::Skip(d) => d,
        }
    }
}

/// One live run: where it writes, and what earlier stages left for later ones.
struct Run {
    dir: PathBuf,
    cli: PathBuf,
    /// The analysis stage's project and report (for the showcase).
    analysed: Option<(PathBuf, PathBuf)>,
    /// The showcase's export (for the upload).
    export: Option<PathBuf>,
}

impl Run {
    fn log(&self, stage: &str) -> PathBuf {
        self.dir.join("logs").join(format!("{stage}.log"))
    }

    /// Run `cmd`, its output appended to the stage's log; its stdout, and whether it succeeded.
    fn exec(&self, stage: &str, cmd: &mut Command) -> Result<(String, bool), String> {
        let out = cmd.output().map_err(|e| format!("{cmd:?}: {e}"))?;
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        let text = format!("$ {cmd:?}\n--- stdout\n{stdout}\n--- stderr\n{}\n--- {}\n\n", String::from_utf8_lossy(&out.stderr), out.status);
        let path = self.log(stage);
        let mut old = std::fs::read_to_string(&path).unwrap_or_default();
        old.push_str(&text);
        std::fs::write(&path, old).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok((stdout, out.status.success()))
    }

    fn cli(&self) -> Command {
        Command::new(&self.cli)
    }

    /// `filmcraft-cli run` of `lines` (one command per line), with `extra` arguments; the
    /// results by line, or what failed.
    fn script(&self, stage: &str, name: &str, lines: &[Value], extra: &[&str]) -> Result<Vec<Value>, String> {
        let path = self.dir.join(format!("{name}.jsonl"));
        let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
        let (stdout, ok) = self.exec(stage, self.cli().arg("run").arg(&path).args(extra))?;
        let results: Vec<Value> = stdout.lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
        if !ok {
            let failed = results.iter().find(|r| r["ok"] == false).map(|r| format!("{}: {}", r["id"], r["error"])).unwrap_or_else(|| "see the log".into());
            return Err(format!("{name}: {failed}"));
        }
        Ok(results.into_iter().map(|r| r["result"].clone()).collect())
    }
}

/// The share of `said`'s words (letters only, three or more) that `heard` holds, a word
/// counting when heard as written or with its first four letters (as the showcase's check).
pub(crate) fn heard_share(said: &str, heard: &str) -> f64 {
    let words = |t: &str| -> Vec<String> { t.split(|c: char| !c.is_alphabetic()).filter(|w| w.chars().count() >= 3).map(str::to_lowercase).collect() };
    let said = words(said);
    if said.is_empty() {
        return 1.0;
    }
    let heard = words(heard);
    let stem = |w: &str| w.chars().take(4).collect::<String>();
    let found = said.iter().filter(|w| heard.iter().any(|h| h == *w || (h.chars().count() >= 4 && w.chars().count() >= 4 && stem(h) == stem(w)))).count();
    found as f64 / said.len() as f64
}

/// A `cargo test` of one live test, its gate variable set.
fn cargo_test(run: &Run, stage: &str, env: (&str, &str), args: &[&str]) -> Outcome {
    let mut c = Command::new(env!("CARGO"));
    c.current_dir(crate::root()).arg("test").arg("--release").args(args).env(env.0, env.1);
    match run.exec(stage, c.args(["--", "--nocapture"])) {
        Ok((_, true)) => Outcome::Pass(format!("{} {}", args.join(" "), env.0)),
        Ok((_, false)) => Outcome::Fail(format!("the test failed: {}", run.log(stage).display())),
        Err(e) => Outcome::Fail(e),
    }
}

/// The voice to read with: `--voice`, else the first natural (VoxCPM) clone.
fn pick_voice(run: &Run, o: &Options) -> Result<Option<String>, String> {
    if let Some(v) = &o.voice {
        return Ok(Some(v.clone()));
    }
    let (stdout, ok) = run.exec("voice", run.cli().args(["exec", "tts.clonedVoices"]))?;
    if !ok {
        return Err("tts.clonedVoices failed".into());
    }
    let v: Value = serde_json::from_str(&stdout).unwrap_or(Value::Null);
    let list = v.get("voices").and_then(Value::as_array).or_else(|| v.as_array()).cloned().unwrap_or_default();
    Ok(list
        .iter()
        .find(|p| p["engine"] == "voxcpm")
        .and_then(|p| p["id"].as_str())
        .map(|id| if id.starts_with("clone:") { id.to_string() } else { format!("clone:{id}") }))
}

fn voice(run: &Run, o: &Options) -> Outcome {
    let voice = match pick_voice(run, o) {
        Ok(Some(v)) => v,
        Ok(None) => return Outcome::Skip("no natural cloned voice (clone one, or --voice clone:NAME)".into()),
        Err(e) => return Outcome::Fail(e),
    };
    // a new line each minute, so the voice really reads (not the narration cache)
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let said = format!("This is the FilmCraft live check, read at minute {} of the day. The robots are ready for their close up.", (secs / 60) % 1440);
    let project = run.dir.join("voice.fcproj");
    let project_arg = project.to_string_lossy().into_owned();
    let made = run.script(
        "voice",
        "voice-read",
        &[
            json!({"id": "file.newProject"}),
            json!({"id": "file.newSequence", "params": {"width": 1280, "height": 720}}),
            json!({"id": "tts.create", "params": {"text": said, "voice": voice, "delivery": "warm", "time": 0}}),
        ],
        &["--save-as", &project_arg],
    );
    let created = match made {
        Ok(r) => r.get(2).cloned().unwrap_or(Value::Null),
        Err(e) => return Outcome::Fail(e),
    };
    let Some(item) = created["item"].as_u64() else { return Outcome::Fail(format!("tts.create gave no item: {created}")) };
    let heard = run.script(
        "voice",
        "voice-hear",
        &[json!({"id": "transcript.generate", "params": {"items": [item], "language": "en"}}), json!({"id": "transcript.inspect", "params": {"item": item}})],
        &["--project", &project_arg],
    );
    let text = match heard {
        Ok(r) => r.get(1).and_then(|v| v["text"].as_str()).unwrap_or("").to_string(),
        Err(e) => return Outcome::Fail(e),
    };
    let share = heard_share(&said, &text);
    let pitch = created["pitchHz"].as_f64().map(|p| format!(", {p:.0} Hz")).unwrap_or_default();
    let d = format!("{voice}: {:.0}% heard back{pitch} — \"{}\"", share * 100.0, text.trim());
    if share >= HEARD_ENOUGH { Outcome::Pass(d) } else { Outcome::Fail(d) }
}

/// A clip the analysis cache says shows the subject, from the Google Photos download cache.
fn known_clip() -> Option<PathBuf> {
    let data = data_dir()?;
    let cache = data.join("autocut/analysis-cache");
    let downloads = data.join("google-photos/cache");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&cache).ok()?.flatten().map(|e| e.path()).collect();
    entries.sort();
    for e in entries {
        let Ok(v) = std::fs::read(&e).map_err(|_| ()).and_then(|b| serde_json::from_slice::<Value>(&b).map_err(|_| ())) else { continue };
        let key = v["key"].as_str().unwrap_or("");
        if v["facts"]["present"] != true || !key.contains(&format!("subject={SUBJECT}|")) || !key.ends_with(&format!("frames={FRAMES}|screen=0")) {
            continue;
        }
        let Some(name) = v["facts"]["name"].as_str() else { continue };
        // downloads are kept as <cache>/<media item id>/<file name>
        let id = key.split('|').find_map(|p| p.strip_prefix("google:"));
        if let Some(found) = id.map(|id| downloads.join(id).join(name)).filter(|p| p.is_file()) {
            return Some(found);
        }
    }
    None
}

/// The FilmCraft data directory (as the app's).
fn data_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").filter(|h| !h.is_empty()).map(PathBuf::from);
    if cfg!(target_os = "macos") {
        home.map(|h| h.join("Library/Application Support/FilmCraft"))
    } else if cfg!(windows) {
        std::env::var_os("APPDATA").map(|a| PathBuf::from(a).join("FilmCraft"))
    } else {
        std::env::var_os("XDG_DATA_HOME").map(|x| PathBuf::from(x).join("filmcraft")).or_else(|| home.map(|h| h.join(".local/share/filmcraft")))
    }
}

fn analysis(run: &mut Run, o: &Options) -> Outcome {
    let Some(clip) = o.clip.clone().or_else(known_clip) else {
        return Outcome::Skip(format!("no clip: pass --clip FILE (a video showing a {SUBJECT})"));
    };
    let project = run.dir.join("show.fcproj");
    let project_arg = project.to_string_lossy().into_owned();
    let imported = run.script(
        "analysis",
        "import",
        &[json!({"id": "file.newProject"}), json!({"id": "file.import", "params": {"paths": [clip.to_string_lossy()]}})],
        &["--save-as", &project_arg],
    );
    let item = match imported {
        Ok(r) => r.get(1).and_then(|v| v["items"].get(0)).and_then(Value::as_u64),
        Err(e) => return Outcome::Fail(e),
    };
    let Some(item) = item else { return Outcome::Fail("the clip did not import".into()) };
    let report = run.dir.join("analysis.json");
    let mut hits = Vec::new();
    for _ in 0..2 {
        let r = run.exec(
            "analysis",
            run.cli().args(["analyze", SUBJECT, "--items", &item.to_string(), "--frames", FRAMES, "--project", &project_arg, "--out"]).arg(&report),
        );
        match r {
            Ok((stdout, true)) => hits.push(serde_json::from_str::<Value>(&stdout).ok().and_then(|v| v["cached"].as_u64()).unwrap_or(0)),
            Ok((_, false)) => return Outcome::Fail(format!("analyze failed: {}", run.log("analysis").display())),
            Err(e) => return Outcome::Fail(e),
        }
    }
    run.analysed = Some((project, report));
    let name = clip.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    match hits.as_slice() {
        [first, second] if *second >= 1 => {
            Outcome::Pass(format!("{name}: second analysis from the cache (first: {})", if *first >= 1 { "cached too" } else { "analysed" }))
        }
        _ => Outcome::Fail(format!("{name}: the second analysis was not a cache hit ({hits:?})")),
    }
}

fn showcase(run: &mut Run, o: &Options) -> Outcome {
    let Some((project, report)) = run.analysed.clone() else { return Outcome::Skip("needs the analysis stage".into()) };
    let export = run.dir.join("show.mp4");
    let mut c = run.cli();
    c.arg("showcase").arg("--report").arg(&report).arg("--project").arg(&project).args(["--title", "Live Check", "--export"]).arg(&export);
    if let Ok(Some(v)) = pick_voice(run, o) {
        c.args(["--voice", &v]);
    }
    match run.exec("showcase", &mut c) {
        Ok((stdout, true)) => {
            let v: Value = serde_json::from_str(&stdout).unwrap_or(Value::Null);
            run.export = Some(export.clone());
            // the show's own QA (on with an export): errors fail the stage, the rest is reported
            let (errors, warnings) = (v["qa"]["errors"].as_u64().unwrap_or(0), v["qa"]["warnings"].as_u64().unwrap_or(0));
            let lufs = v["qa"]["loudness"]["integratedLufs"].as_f64().map(|l| format!(", {l:.1} LUFS")).unwrap_or_default();
            let d = format!("{:.0} s show, QA {errors} error(s) {warnings} warning(s){lufs}: {}", v["seconds"].as_f64().unwrap_or(0.0), export.display());
            if errors == 0 { Outcome::Pass(d) } else { Outcome::Fail(d) }
        }
        Ok((_, false)) => Outcome::Fail(format!("showcase failed: {}", run.log("showcase").display())),
        Err(e) => Outcome::Fail(e),
    }
}

fn upload(run: &Run) -> Outcome {
    let Some(export) = &run.export else { return Outcome::Skip("needs the showcase stage".into()) };
    match run.exec("upload", run.cli().args(["photos", "upload"]).arg(export).args(["--album", ALBUM, "--description", "FilmCraft live check"])) {
        Ok((_, true)) => Outcome::Pass(format!("uploaded into \"{ALBUM}\"")),
        Ok((_, false)) => Outcome::Fail(format!("upload failed: {}", run.log("upload").display())),
        Err(e) => Outcome::Fail(e),
    }
}

fn picker(run: &Run) -> Outcome {
    match run.exec("picker", run.cli().args(["photos", "pick", "--auto", "robot", "--auto", "GITEX", "--max-items", "2", "--timeout", "900"])) {
        Ok((_, true)) => {}
        Ok((_, false)) => return Outcome::Fail(format!("photos pick failed: {}", run.log("picker").display())),
        Err(e) => return Outcome::Fail(e),
    }
    match run.exec("picker", run.cli().args(["photos", "tidy"])) {
        Ok((_, true)) => Outcome::Pass("two searches picked in one session, tabs tidied".into()),
        Ok((_, false)) => Outcome::Fail(format!("photos tidy failed: {}", run.log("picker").display())),
        Err(e) => Outcome::Fail(e),
    }
}

pub(crate) fn run_live(args: &[String]) -> Result<(), String> {
    let o = parse(args)?;
    let md = crate::metadata()?;
    let target = md["target_directory"].as_str().map(PathBuf::from).ok_or("cargo metadata names no target directory")?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let dir = target.join("live").join(stamp.to_string());
    std::fs::create_dir_all(dir.join("logs")).map_err(|e| format!("{}: {e}", dir.display()))?;
    eprintln!("live: building filmcraft-cli (release)");
    crate::run(Command::new(env!("CARGO")).current_dir(crate::root()).args(["build", "--release", "-p", "filmcraft-cli"]))?;
    let mut run = Run { cli: target.join("release").join(format!("filmcraft-cli{}", std::env::consts::EXE_SUFFIX)), dir, analysed: None, export: None };
    if let Ok((locks, _)) = run.exec("locks", run.cli().arg("locks")) {
        eprint!("live: locks now:\n{locks}");
    }
    let mut results = Vec::new();
    for (id, about) in STAGES {
        if !wanted(&o, id) {
            continue;
        }
        eprintln!("live: {id} — {about} …");
        let start = Instant::now();
        let out = match *id {
            "models" => cargo_test(&run, id, ("AUTOCUT_LIVE", "1"), &["-p", "filmcraft-autocut", "--test", "live_models"]),
            "screen" => cargo_test(&run, id, ("AUTOCUT_SCREEN_LIVE", "1"), &["-p", "filmcraft-screenagent", "--lib", "live_ground"]),
            "whisper" => cargo_test(&run, id, ("AUTOCUT_WHISPER_LIVE", "1"), &["-p", "filmcraft-cli", "--test", "autocut_whisper"]),
            "voice" => voice(&run, &o),
            "analysis" => analysis(&mut run, &o),
            "showcase" => showcase(&mut run, &o),
            "keychain" => cargo_test(&run, id, ("FILMCRAFT_LIVE_KEYCHAIN", "1"), &["-p", "filmcraft-oauth", "--test", "live_keychain"]),
            "album" => cargo_test(&run, id, ("FILMCRAFT_GPHOTOS_LIVE_ALBUM", ALBUM), &["-p", "filmcraft-gphotos", "--test", "live_albums"]),
            "upload" => upload(&run),
            "picker" => picker(&run),
            _ => Outcome::Skip("unknown stage".into()),
        };
        let secs = start.elapsed().as_secs_f64();
        eprintln!("live: {id} {} ({secs:.0} s): {}", out.word(), out.detail());
        results
            .push(json!({"stage": id, "about": about, "outcome": out.word().to_lowercase(), "detail": out.detail(), "seconds": (secs * 10.0).round() / 10.0}));
    }
    let failed = results.iter().filter(|r| r["outcome"] == "fail").count();
    let report = json!({"started": stamp, "dir": run.dir, "stages": results});
    let path = run.dir.join("report.json");
    std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap_or_default()).map_err(|e| format!("{}: {e}", path.display()))?;
    eprintln!("live: report in {}", path.display());
    if failed == 0 { Ok(()) } else { Err(format!("live: {failed} stage(s) failed")) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &[&str]) -> Vec<String> {
        s.iter().map(|a| a.to_string()).collect()
    }

    #[test]
    fn stages_run_by_default_unless_they_reach_out() {
        let o = parse(&[]).unwrap();
        let on: Vec<&str> = STAGES.iter().map(|(id, _)| *id).filter(|id| wanted(&o, id)).collect();
        assert_eq!(on, ["models", "screen", "whisper", "voice", "analysis", "showcase"]);
        let o = parse(&args(&["--google", "--picker"])).unwrap();
        assert!(wanted(&o, "keychain") && wanted(&o, "album") && wanted(&o, "upload") && wanted(&o, "picker"));
        let o = parse(&args(&["--only", "voice,album"])).unwrap();
        let on: Vec<&str> = STAGES.iter().map(|(id, _)| *id).filter(|id| wanted(&o, id)).collect();
        assert_eq!(on, ["voice", "album"]);
        assert!(parse(&args(&["--only", "voice,nope"])).unwrap_err().contains("no stage `nope`"));
        assert!(parse(&args(&["--voice"])).is_err() && parse(&args(&["--loud"])).is_err());
        assert_eq!(parse(&args(&["--clip", "a.mp4", "--voice", "clone:me"])).unwrap().clip, Some(PathBuf::from("a.mp4")));
    }

    #[test]
    fn heard_back_as_the_showcase_hears_it() {
        let said = "This is the FilmCraft live check. The robots are ready.";
        assert_eq!(heard_share(said, "This is the FilmCraft live check. The robots are ready."), 1.0);
        assert!(heard_share(said, "this is the film craft live check the robot is ready") >= 0.5);
        assert!(heard_share(said, "blah mumble") < 0.5);
        assert_eq!(heard_share("a b", ""), 1.0);
    }
}
