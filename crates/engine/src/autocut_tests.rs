//! Tests of [`crate::autocut`]: the showcase jobs run the host's work on a thread that edits only
//! through the command port (run by whoever owns the session: a `wait: true` call, or
//! [`Session::pump_port`] as the UI does each frame), keep the report, plan and build in the
//! session, report their outcome in `jobs.list`, run one at a time, stop on `jobs.cancel`, survive
//! a panicking host, and refuse hostile parameters before any job starts.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;
use crate::autocut::{AutoCutHost, CommandPort, Task, TaskKind};
use serde_json::{Value, json};

/// A host that does a little of each job through the port, or waits to be cancelled, or panics.
#[derive(Default)]
struct Fake {
    hang: bool,
    panic: bool,
    not_ready: Option<String>,
    /// It can pick Google Photos albums.
    albums: bool,
    tasks: Mutex<Vec<(TaskKind, Value)>>,
    /// The session answered a request while the job ran.
    answered: AtomicBool,
}

impl AutoCutHost for Fake {
    fn ready(&self) -> std::result::Result<(), String> {
        self.not_ready.clone().map_or(Ok(()), Err)
    }

    fn albums_ready(&self) -> std::result::Result<(), String> {
        if self.albums { Ok(()) } else { Err(crate::autocut::NO_ALBUMS.into()) }
    }

    fn run(&self, task: Task, port: &CommandPort, progress: &Arc<filmcraft_export::Progress>) -> std::result::Result<Value, String> {
        self.tasks.lock().unwrap().push((task.kind, task.params.clone()));
        if self.panic {
            panic!("a bug in the host");
        }
        if self.hang {
            for _ in 0..2000 {
                // a request at a time, as a real job: it stops once cancelled
                port.call("project.inspect", json!({}))?;
                std::thread::sleep(Duration::from_millis(2));
            }
            return Err("never cancelled".into());
        }
        *progress.status.lock().unwrap() = format!("working on {:?}", task.kind);
        match task.kind {
            TaskKind::Analyze | TaskKind::Make => {
                let project = port.call("project.inspect", json!({}))?;
                self.answered.store(project.is_object(), Ordering::Relaxed);
                let subject = task.params["subject"].as_str().unwrap_or("thing");
                let report = json!({"subject": subject, "clips": [{"item": 5, "present": true}, {"item": 6, "present": false}]});
                let mut out = json!({"report": report, "summary": {"clips": 2}, "status": "Found 1 clip"});
                if task.kind == TaskKind::Make {
                    out["plan"] = json!({"subject": subject, "entries": [{"number": 1, "label": "red robot"}], "script": {"intro": "Hi.", "lines": ["One."], "outro": "Bye."}});
                }
                Ok(out)
            }
            TaskKind::Plan => {
                let n = task.report.as_ref().and_then(|r| r["clips"].as_array().map(Vec::len)).unwrap_or(0);
                Ok(
                    json!({"plan": {"title": task.params["title"], "entries": vec![json!({"label": "x"}); n], "script": {"intro": "", "lines": [], "outro": ""}}}),
                )
            }
            TaskKind::Build => {
                let plan = task.params.get("plan").cloned().or(task.plan).unwrap_or_default();
                let r = port.call("file.newSequence", json!({"name": plan["title"].as_str().unwrap_or("Showcase"), "width": 640, "height": 360, "fps": 30}))?;
                port.call("graphics.newText", json!({"text": "ROBOTS", "seconds": 2.0, "time": 0}))?;
                let frame = port.render(1.0, 64);
                Ok(json!({"built": {"sequence": r["sequence"], "frame": frame.map(|(w, h, px)| json!([w, h, px.len()]))}, "status": "Built"}))
            }
            TaskKind::Shorts => Ok(json!({"shorts": [], "params": task.params, "status": "Made 0 shorts"})),
            TaskKind::Trailer => Ok(json!({"shots": [], "params": task.params, "status": "Made a trailer"})),
            TaskKind::RoughCut => Ok(json!({"params": task.params, "status": "Rough cut"})),
            TaskKind::MalayalamScript => {
                let plan = task.params.get("plan").cloned().or(task.plan).unwrap_or_default();
                let n = plan["entries"].as_array().map_or(0, Vec::len);
                Ok(
                    json!({"malayalam": {"intro": "ഹായ്.", "outro": "ബൈ.", "lines": {}}, "summary": {"parts": n + 2}, "params": task.params, "status": "Translated"}),
                )
            }
        }
    }
}

fn session(fake: Fake) -> (Session, Arc<Fake>) {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    let fake = Arc::new(fake);
    s.autocut.host = Some(fake.clone());
    (s, fake)
}

fn job_entry(s: &mut Session, id: u64) -> Value {
    let jobs = s.execute("jobs.list", json!({})).unwrap();
    jobs.as_array().unwrap().iter().find(|j| j["id"] == id).cloned().unwrap_or_default()
}

/// Pump the port as the UI does each frame until job `id` finishes.
fn pump_until_done(s: &mut Session, id: u64) -> Value {
    let t0 = std::time::Instant::now();
    loop {
        s.pump_port(Duration::from_millis(8));
        let j = job_entry(s, id);
        if j["finished"] == true {
            return j;
        }
        assert!(t0.elapsed() < Duration::from_secs(30), "the job never finished: {j}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn without_a_host_the_jobs_are_disabled() {
    let mut s = Session::default();
    for id in ["autocut.analyze", "autocut.showcasePlan", "autocut.makeShowcase", "autocut.showcaseBuild"] {
        let e = s.execute(id, json!({"subject": "robot"})).unwrap_err().to_string();
        assert!(e.contains("not available"), "{id}: {e}");
    }
    // reading and setting the kept plan works anywhere
    assert_eq!(s.execute("autocut.showcaseInspect", json!({})).unwrap()["available"], false);
    let (mut s, _) = session(Fake { not_ready: Some("Ollama is not running".into()), ..Fake::default() });
    let e = s.execute("autocut.analyze", json!({"subject": "robot"})).unwrap_err().to_string();
    assert!(e.contains("Ollama is not running"), "{e}");
}

#[test]
fn analyze_with_wait_runs_through_the_port_and_keeps_the_report() {
    let (mut s, fake) = session(Fake::default());
    let r = s.execute("autocut.analyze", json!({"subject": "  robot ", "since": "2020-06", "items": [5, 6], "frames": 4, "wait": true})).unwrap();
    assert!(fake.answered.load(Ordering::Relaxed), "the job's project.inspect was answered by the session");
    let (kind, params) = fake.tasks.lock().unwrap()[0].clone();
    assert_eq!(kind, TaskKind::Analyze);
    // checked and normalized for the host: the subject trimmed, the date as Unix seconds
    assert_eq!(params["subject"], "robot");
    assert_eq!(params["sinceUnix"], 1_590_969_600, "2020-06-01T00:00:00Z: {params}");
    assert_eq!((params["items"].clone(), params["frames"].clone()), (json!([5, 6]), json!(4)));
    // the answer and jobs.list carry the outcome without the whole report
    assert_eq!(r["showcase"]["summary"], json!({"clips": 2}), "{r}");
    assert!(r["showcase"].get("report").is_none(), "{r}");
    let job = r["job"].as_u64().unwrap();
    let j = job_entry(&mut s, job);
    assert_eq!((j["finished"].clone(), j["status"].clone()), (json!(true), json!("Found 1 clip")), "{j}");
    assert_eq!(j["showcase"]["summary"]["clips"], 2, "{j}");
    // the session keeps the report: a summary by default, all of it on request
    let i = s.execute("autocut.showcaseInspect", json!({})).unwrap();
    assert_eq!(i["report"], json!({"subject": "robot", "since": null, "clips": 2, "showing": 1}), "{i}");
    assert_eq!((i["reportRevision"].as_u64(), i["plan"].clone()), (Some(1), Value::Null));
    assert_eq!(i["job"]["command"], "autocut.analyze", "{i}");
    let whole = s.execute("autocut.showcaseInspect", json!({"report": true})).unwrap();
    assert_eq!(whole["report"]["clips"][0]["item"], 5);
}

#[test]
fn plan_needs_a_report_and_build_needs_a_plan() {
    let (mut s, _) = session(Fake::default());
    let e = s.execute("autocut.showcasePlan", json!({"title": "Robots"})).unwrap_err().to_string();
    assert!(e.contains("autocut.analyze"), "{e}");
    let e = s.execute("autocut.showcaseBuild", json!({})).unwrap_err().to_string();
    assert!(e.contains("autocut.showcasePlan"), "{e}");
    s.execute("autocut.analyze", json!({"subject": "robot", "wait": true})).unwrap();
    s.execute("autocut.showcasePlan", json!({"title": "Robots", "wait": true})).unwrap();
    let i = s.execute("autocut.showcaseInspect", json!({})).unwrap();
    assert_eq!((i["plan"]["title"].as_str(), i["plan"]["entries"].as_array().map(Vec::len)), (Some("Robots"), Some(2)), "{i}");
    assert_eq!(i["planRevision"], 1);
    // reviewed entries are enough without a report
    let (mut s, fake) = session(Fake::default());
    s.execute("autocut.showcasePlan", json!({"entries": [{"label": "a", "include": false}], "minCoverage": 0.5, "wait": true})).unwrap();
    let (_, params) = fake.tasks.lock().unwrap()[0].clone();
    assert_eq!((params["entries"][0]["include"].clone(), params["minCoverage"].clone()), (json!(false), json!(0.5)));
}

#[test]
fn make_keeps_both_report_and_plan() {
    let (mut s, _) = session(Fake::default());
    let r = s.execute("autocut.makeShowcase", json!({"prompt": "make a video of every robot since 2020", "wait": true})).unwrap();
    assert_eq!(r["showcase"]["plan"]["entries"][0]["label"], "red robot", "{r}");
    let i = s.execute("autocut.showcaseInspect", json!({})).unwrap();
    assert_eq!((i["reportRevision"].as_u64(), i["planRevision"].as_u64()), (Some(1), Some(1)), "{i}");
}

/// The UI's way: start without waiting, then run the job's commands frame by frame. The build's
/// edits are ordinary commands (in History), and the frame it asked for rendered.
#[test]
fn a_build_pumped_like_the_ui_edits_through_commands() {
    let (mut s, _) = session(Fake::default());
    let undo_before = s.history.undo.len();
    let plan = json!({"title": "Every Robot", "entries": [{"label": "red robot"}], "script": {"intro": "", "lines": [""], "outro": ""}});
    let r = s.execute("autocut.showcaseBuild", json!({"plan": plan})).unwrap();
    let id = r["job"].as_u64().unwrap();
    assert!(r.get("result").is_none(), "no waiting: {r}");
    assert!(crate::autocut::running(&s).is_some(), "the job waits for the session to run its commands");
    let j = pump_until_done(&mut s, id);
    assert_eq!(j["result"]["error"], Value::Null, "{j}");
    assert!(crate::autocut::running(&s).is_none());
    let seq = s.active_sequence().unwrap();
    assert_eq!((seq.settings.width, seq.settings.height), (640, 360), "the new sequence is open");
    assert!(s.history.undo.len() > undo_before, "the build's edits are undoable");
    let built = j["showcase"]["built"].clone();
    assert_eq!(built["frame"][0], 64, "a 64-px frame of the new sequence: {built}");
    assert_eq!(s.execute("autocut.showcaseInspect", json!({})).unwrap()["built"], built);
}

#[test]
fn one_showcase_job_at_a_time_and_cancel_stops_it() {
    let (mut s, _) = session(Fake { hang: true, ..Fake::default() });
    let id = s.execute("autocut.analyze", json!({"subject": "robot"})).unwrap()["job"].as_u64().unwrap();
    s.pump_port(Duration::from_millis(20));
    let e = s.execute("autocut.makeShowcase", json!({"subject": "robot"})).unwrap_err().to_string();
    assert!(e.contains("already running") && e.contains(&format!("job {id}")), "{e}");
    let e = s.execute("autocut.showcaseSet", json!({"plan": null})).unwrap_err().to_string();
    assert!(e.contains("running"), "{e}");
    s.execute("jobs.cancel", json!({"job": id})).unwrap();
    let j = pump_until_done(&mut s, id);
    assert_eq!(j["result"]["error"], "cancelled", "{j}");
    assert!(crate::autocut::running(&s).is_none());
    let i = s.execute("autocut.showcaseInspect", json!({})).unwrap();
    assert_eq!(i["report"], Value::Null, "a cancelled analysis keeps nothing: {i}");
    // and the next job may start
    assert!(s.execute("autocut.analyze", json!({"subject": "robot"})).is_ok());
}

/// A bug in the host becomes the job's error; the session and later jobs carry on.
#[test]
fn a_panicking_host_fails_its_job_only() {
    let (mut s, _) = session(Fake { panic: true, ..Fake::default() });
    let e = s.execute("autocut.analyze", json!({"subject": "robot", "wait": true})).unwrap_err().to_string();
    assert!(e.contains("internal error"), "{e}");
    assert!(crate::autocut::running(&s).is_none());
    assert!(s.execute("sequence.inspect", json!({})).is_ok());
}

#[test]
fn hostile_parameters_are_refused_before_a_job_starts() {
    let (mut s, fake) = session(Fake::default());
    let long = "x".repeat(crate::autocut::MAX_SUBJECT_CHARS + 1);
    let cases = [
        ("autocut.analyze", json!({})),
        ("autocut.analyze", json!({"subject": ""})),
        ("autocut.analyze", json!({"subject": 7})),
        ("autocut.analyze", json!({"subject": long})),
        ("autocut.analyze", json!({"subject": "robot", "items": "all"})),
        ("autocut.analyze", json!({"subject": "robot", "items": []})),
        ("autocut.analyze", json!({"subject": "robot", "items": [-1]})),
        ("autocut.analyze", json!({"subject": "robot", "items": [1.5]})),
        ("autocut.analyze", json!({"subject": "robot", "items": vec![1; crate::autocut::MAX_ITEMS + 1]})),
        ("autocut.analyze", json!({"subject": "robot", "since": "last year"})),
        ("autocut.analyze", json!({"subject": "robot", "since": "2020-13"})),
        ("autocut.analyze", json!({"subject": "robot", "since": "20-01-01"})),
        ("autocut.analyze", json!({"subject": "robot", "since": "2020-01-01-01"})),
        ("autocut.analyze", json!({"subject": "robot", "since": 99999})),
        ("autocut.analyze", json!({"subject": "robot", "since": {"y": 2020}})),
        ("autocut.analyze", json!({"subject": "robot", "frames": 0})),
        ("autocut.analyze", json!({"subject": "robot", "frames": 9})),
        ("autocut.analyze", json!({"subject": "robot", "frames": -2})),
        ("autocut.analyze", json!({"subject": "robot", "frames": 2.5})),
        ("autocut.analyze", json!({"subject": "robot", "compare": 1e12})),
        ("autocut.analyze", json!([1, 2])),
        ("autocut.analyze", json!({"prompt": "  "})),
        ("autocut.analyze", json!({"subject": "robot", "album": ""})),
        ("autocut.analyze", json!({"subject": "robot", "album": "  "})),
        ("autocut.analyze", json!({"subject": "robot", "album": 7})),
        ("autocut.analyze", json!({"subject": "robot", "album": "Robots\nand more"})),
        ("autocut.analyze", json!({"subject": "robot", "album": "a".repeat(crate::photos::MAX_ALBUM_TITLE_CHARS + 1)})),
        ("autocut.analyze", json!({"subject": "robot", "album": "Robots", "items": [5]})),
        ("autocut.makeShowcase", json!({"title": "no subject"})),
        ("autocut.makeShowcase", json!({"prompt": ""})),
        ("autocut.makeShowcase", json!({"album": "Robots"})),
        ("autocut.makeShowcase", json!({"subject": "robot", "title": "t".repeat(crate::autocut::MAX_TEXT_CHARS + 1)})),
        ("autocut.makeShowcase", json!({"subject": "robot", "voice": ["clone:me"]})),
        ("autocut.showcasePlan", json!({"entries": "all"})),
        ("autocut.showcasePlan", json!({"entries": [1, 2]})),
        ("autocut.showcasePlan", json!({"entries": vec![json!({}); crate::autocut::MAX_ENTRIES + 1]})),
        ("autocut.showcasePlan", json!({"entries": [], "minCoverage": 2})),
        ("autocut.showcasePlan", json!({"entries": [], "minCoverage": "most"})),
        ("autocut.showcaseBuild", json!({"plan": "robots.json"})),
        ("autocut.showcaseBuild", json!({"plan": {"script": {}}})),
        ("autocut.showcaseBuild", json!({"plan": {"entries": [], "script": "Hi."}})),
        ("autocut.showcaseBuild", json!({"plan": {"entries": []}, "upload": true})),
        ("autocut.showcaseBuild", json!({"plan": {"entries": []}, "export": 3})),
        ("autocut.showcaseBuild", json!({"plan": {"entries": []}, "qa": "yes"})),
        ("autocut.showcaseBuild", json!({"plan": {"entries": []}, "export": "/tmp/x.mp4", "uploadAlbum": "Robots"})),
        ("autocut.showcaseBuild", json!({"plan": {"entries": []}, "export": "/tmp/x.mp4", "upload": true, "uploadAlbum": 3})),
    ];
    for (id, p) in cases {
        assert!(s.execute(id, p.clone()).is_err(), "{id} {p} should be refused");
    }
    assert!(fake.tasks.lock().unwrap().is_empty(), "no job started");
    assert!(s.jobs.is_empty());
    // an upload needs the Google Photos uploader before anything is built
    let e = s.execute("autocut.showcaseBuild", json!({"plan": {"entries": []}, "export": "/tmp/x.mp4", "upload": true})).unwrap_err().to_string();
    assert!(e.contains("upload") && e.contains("not available"), "{e}");
    // an album needs a host that can pick albums, before anything starts
    let e = s.execute("autocut.analyze", json!({"subject": "robot", "album": "Robots"})).unwrap_err().to_string();
    assert!(e.contains("album") && e.contains("not available"), "{e}");
    assert!(s.jobs.is_empty());
}

/// `album`: the analysis's clips come from a Google Photos album the host picks; the host gets
/// the title (trimmed), with a subject or a prompt, and no `items`.
#[test]
fn an_album_seeds_the_analysis() {
    let (mut s, fake) = session(Fake { albums: true, ..Fake::default() });
    s.execute("autocut.analyze", json!({"prompt": "every robot since 2020", "album": "  Robot showcase ", "wait": true})).unwrap();
    s.execute("autocut.makeShowcase", json!({"subject": "robot", "album": "Robot showcase", "since": "2021", "wait": true})).unwrap();
    let tasks = fake.tasks.lock().unwrap().clone();
    assert_eq!(tasks.len(), 2);
    let (kind, p) = &tasks[0];
    assert_eq!(*kind, TaskKind::Analyze);
    assert_eq!(
        (p["album"].as_str(), p["prompt"].as_str(), p.get("subject"), p.get("items")),
        (Some("Robot showcase"), Some("every robot since 2020"), None, None),
        "{p}"
    );
    let (kind, p) = &tasks[1];
    assert_eq!((*kind, p["album"].as_str(), p["sinceUnix"].as_i64()), (TaskKind::Make, Some("Robot showcase"), Some(1_609_459_200)), "{p}");
}

#[test]
fn fresh_reaches_the_host_only_when_asked() {
    let (mut s, fake) = session(Fake::default());
    s.execute("autocut.analyze", json!({"subject": "robot", "fresh": true, "wait": true})).unwrap();
    s.execute("autocut.makeShowcase", json!({"subject": "robot", "fresh": false, "wait": true})).unwrap();
    s.execute("autocut.analyze", json!({"subject": "robot", "fresh": "yes", "wait": true})).unwrap();
    let tasks = fake.tasks.lock().unwrap().clone();
    let fresh: Vec<Option<&Value>> = tasks.iter().map(|(_, p)| p.get("fresh")).collect();
    assert_eq!(fresh, [Some(&json!(true)), None, None], "only a true `fresh` is passed on");
}

#[test]
fn since_reads_years_months_and_days() {
    use crate::autocut::parse_since;
    assert_eq!(parse_since(&json!("2020")), Ok(1_577_836_800));
    assert_eq!(parse_since(&json!(2020)), Ok(1_577_836_800));
    assert_eq!(parse_since(&json!(" 2020-06 ")), Ok(1_590_969_600));
    assert_eq!(parse_since(&json!("2020-06-15")), Ok(1_592_179_200));
    for bad in [json!("1800"), json!("2020-00"), json!("2020-01-32"), json!("2020/01"), json!(""), json!(null), json!(2020.5), json!("２０２０")] {
        assert!(parse_since(&bad).is_err(), "{bad}");
    }
}

/// The CLI's files load as the kept report and plan; `null` clears; bad shapes are refused.
#[test]
fn set_loads_cli_files_and_clears() {
    let (mut s, _) = session(Fake::default());
    let dir = std::env::temp_dir().join(format!("filmcraft-showcase-set-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let report = dir.join("robots.json");
    let plan = dir.join("plan.json");
    std::fs::write(&report, json!({"subject": "robot", "clips": [{"item": 5, "present": true}]}).to_string()).unwrap();
    std::fs::write(&plan, json!({"entries": [{"label": "red robot"}], "script": {"intro": "a", "lines": ["b"], "outro": "c"}}).to_string()).unwrap();
    let r = s.execute("autocut.showcaseSet", json!({"reportPath": report.to_string_lossy(), "planPath": plan.to_string_lossy()})).unwrap();
    assert_eq!((r["report"]["showing"].as_u64(), r["plan"].as_bool()), (Some(1), Some(true)), "{r}");
    let i = s.execute("autocut.showcaseInspect", json!({})).unwrap();
    assert_eq!(i["plan"]["entries"][0]["label"], "red robot");
    s.execute("autocut.showcaseSet", json!({"plan": null})).unwrap();
    assert_eq!(s.execute("autocut.showcaseInspect", json!({})).unwrap()["plan"], Value::Null);
    std::fs::write(dir.join("junk.json"), b"not json").unwrap();
    for bad in [
        json!({}),
        json!({"report": {"subject": "robot"}}),
        json!({"plan": {"script": {}}}),
        json!({"planPath": dir.join("junk.json").to_string_lossy()}),
        json!({"reportPath": dir.join("missing.json").to_string_lossy()}),
    ] {
        assert!(s.execute("autocut.showcaseSet", bad.clone()).is_err(), "{bad}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A job whose session is gone gets an error, not a hang.
#[test]
fn a_port_without_its_session_fails() {
    let s = Session::default();
    let port = s.autocut.port.handle(None);
    drop(s);
    assert_eq!(port.call("project.inspect", json!({})).unwrap_err(), "the session closed");
    assert!(port.render(0.0, 64).is_none());
}

/// `autocut.malayalamScript` translates the kept plan, a given one or one from a file, checks
/// its parameters first, and keeps the session's plan as it was.
#[test]
fn a_plan_is_translated_into_malayalam() {
    let (mut s, fake) = session(Fake::default());
    let e = s.execute("autocut.malayalamScript", json!({})).unwrap_err().to_string();
    assert!(e.contains("nothing to translate"), "{e}");
    let plan = json!({"entries": [{"label": "Pepper"}], "script": {"intro": "Hi.", "lines": ["Pepper greets people."], "outro": "Bye."}});
    let r = s.execute("autocut.malayalamScript", json!({"plan": plan, "model": "qwen3:8b", "revise": false, "maxWords": 12, "wait": true})).unwrap();
    assert_eq!(r["showcase"]["malayalam"]["intro"], "ഹായ്.", "{r}");
    assert_eq!(r["showcase"]["summary"]["parts"], 3);
    let (kind, p) = fake.tasks.lock().unwrap()[0].clone();
    assert_eq!(
        (kind, p["model"].as_str(), p["revise"].as_bool(), p["maxWords"].as_u64()),
        (TaskKind::MalayalamScript, Some("qwen3:8b"), Some(false), Some(12))
    );
    // the session's plan is not touched by a translation
    assert!(s.execute("autocut.showcaseInspect", json!({})).unwrap()["plan"].is_null());
    // a plan from a file
    let dir = std::env::temp_dir().join(format!("fc-ml-job-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("plan.json");
    std::fs::write(&path, serde_json::to_vec(&plan).unwrap()).unwrap();
    let r = s.execute("autocut.malayalamScript", json!({"planPath": path.to_string_lossy(), "wait": true})).unwrap();
    assert_eq!(r["showcase"]["params"]["plan"]["entries"][0]["label"], "Pepper", "{r}");
    std::fs::remove_dir_all(dir).ok();
    for bad in [
        json!({"plan": 5}),
        json!({"plan": {"script": {}}}),
        json!({"plan": plan, "planPath": "/x.json"}),
        json!({"planPath": "/no/such/plan.json"}),
        json!({"plan": plan, "model": "qwen3 8b"}),
        json!({"plan": plan, "model": "x".repeat(500)}),
        json!({"plan": plan, "revise": "no"}),
        json!({"plan": plan, "maxWords": 2}),
        json!({"plan": plan, "maxWords": 61}),
        json!({"plan": plan, "out": 5}),
        json!([1]),
    ] {
        assert!(s.execute("autocut.malayalamScript", bad.clone()).is_err(), "{bad} was accepted");
    }
}

/// `autocut.shorts` checks its parameters before a job starts, and hands the host the checked
/// ones (defaults filled in).
#[test]
fn shorts_parameters_are_checked_and_defaulted() {
    let (mut s, _) = session(Fake::default());
    let r = s.execute("autocut.shorts", json!({"count": 2, "aspect": "1:1", "youtube": {"privacy": "unlisted", "tags": ["a"]}, "wait": true})).unwrap();
    let p = &r["showcase"]["params"];
    assert_eq!(
        (p["count"].as_u64(), p["minSeconds"].as_f64(), p["maxSeconds"].as_f64(), p["aspect"].as_str(), p["captions"].as_bool(), p["title"].as_bool()),
        (Some(2), Some(15.0), Some(60.0), Some("1:1"), Some(true), Some(false)),
        "{r}"
    );
    assert_eq!(p["youtube"]["privacy"], "unlisted");
    assert_eq!(p["fallback"], "blur", "nobody followed: a blurred fill by default");
    assert_eq!(p["platform"], "youtube", "captions and titles clear of YouTube Shorts' interface by default");
    let r = s.execute("autocut.shorts", json!({"fallback": "Letterbox", "platform": "TikTok", "wait": true})).unwrap();
    assert_eq!((r["showcase"]["params"]["fallback"].as_str(), r["showcase"]["params"]["platform"].as_str()), (Some("letterbox"), Some("tiktok")));
    for bad in [
        json!({"fallback": "stretch"}),
        json!({"fallback": 2}),
        json!({"platform": "snapchat"}),
        json!({"platform": null}),
        json!({"platform": 1}),
        json!({"count": 0}),
        json!({"count": 21}),
        json!({"count": "three"}),
        json!({"item": 3, "sequence": 4}),
        json!({"item": -1}),
        json!({"minSeconds": 1}),
        json!({"maxSeconds": 1000}),
        json!({"minSeconds": 30, "maxSeconds": 31}),
        json!({"minSeconds": f64::MAX}),
        json!({"aspect": "9:0"}),
        json!({"aspect": 9}),
        json!({"motion": "wobbly"}),
        json!({"captions": "yes"}),
        json!({"uploadYoutube": true}),
        json!({"export": "out\u{0}dir"}),
        json!({"youtube": "private"}),
        json!({"youtube": {"privacy": "everyone"}}),
        json!({"youtube": {"password": "x"}}),
        json!({"topic": "x".repeat(10_000)}),
        json!([1]),
    ] {
        assert!(s.execute("autocut.shorts", bad.clone()).is_err(), "{bad} was accepted");
    }
    // uploads need a YouTube publisher: refused before the job starts here (there is none)
    let e = s.execute("autocut.shorts", json!({"export": "/tmp/x", "uploadYoutube": true})).unwrap_err().to_string();
    assert!(e.contains("uploadYoutube"), "{e}");
}

/// `autocut.roughCut` checks its parameters before a job starts; hostile ones are refused.
#[test]
fn rough_cut_parameters_are_checked() {
    let (mut s, _) = session(Fake::default());
    let r = s
        .execute(
            "autocut.roughCut",
            json!({"paths": ["~/clip.mov"], "language": "ml", "removeRetakes": false, "minPause": 0.8, "export": "~/out.mp4", "wait": true}),
        )
        .unwrap();
    let p = &r["showcase"]["params"];
    assert!(p["paths"][0].as_str().is_some_and(|x| !x.starts_with('~')), "home expanded: {p}");
    assert_eq!((p["language"].as_str(), p["removeRetakes"].as_bool(), p["minPause"].as_f64()), (Some("ml"), Some(false), Some(0.8)), "{p}");
    for bad in [
        json!({}),
        json!({"items": []}),
        json!({"items": "1"}),
        json!({"items": [-1]}),
        json!({"paths": [""]}),
        json!({"paths": [3]}),
        json!({"items": [1], "language": "m l"}),
        json!({"items": [1], "removePauses": "yes"}),
        json!({"items": [1], "minPause": 0}),
        json!({"items": [1], "keepPause": f64::MAX}),
        json!({"items": [1], "name": "x".repeat(500)}),
        json!({"items": (0..300).collect::<Vec<u64>>()}),
        json!([1]),
    ] {
        assert!(s.execute("autocut.roughCut", bad.clone()).is_err(), "{bad} was accepted");
    }
}

/// `autocut.trailer` checks its parameters before a job starts and hands the host the checked
/// ones (defaults filled in, the last build as its report); hostile ones are refused, never a
/// panic.
#[test]
fn trailer_parameters_are_checked_and_defaulted() {
    let (mut s, _) = session(Fake::default());
    let r = s
        .execute(
            "autocut.trailer",
            json!({"sequence": 9, "seconds": 30, "robots": 12, "hook": " Ameca ", "exclude": ["Promobot"], "redact": "Faces, Plates", "platform": "Instagram", "at": {"My first robot": {"item": 1347, "seconds": 9}, " NAO ": 3}, "wait": true}),
        )
        .unwrap();
    let p = &r["showcase"]["params"];
    assert_eq!(
        (p["sequence"].as_u64(), p["seconds"].as_f64(), p["robots"].as_u64(), p["hook"].as_str(), p["redact"].as_str(), p["platform"].as_str()),
        (Some(9), Some(30.0), Some(12), Some("Ameca"), Some("faces,plates"), Some("reels")),
        "{r}"
    );
    assert_eq!((p["aspect"].as_str(), p["fallback"].as_str(), p["bpm"].as_f64()), (Some("9:16"), Some("blur"), Some(124.0)));
    assert_eq!(p["exclude"], json!(["Promobot"]));
    assert_eq!(p["at"], json!({"My first robot": {"item": 1347, "seconds": 9.0}, "NAO": 3.0}));
    assert!(p.get("report").is_none(), "nothing built yet: no report to hand over");
    // the last build is the report when none is given
    s.autocut.state.lock().unwrap().built = Some(json!({"layout": [], "sequence": 9}));
    let r = s.execute("autocut.trailer", json!({"wait": true})).unwrap();
    assert_eq!(r["showcase"]["params"]["report"]["sequence"], 9, "{r}");
    assert_eq!(r["showcase"]["params"]["platform"], "reels", "Reels by default");
    assert!(r["showcase"]["params"].get("look").is_none() && r["showcase"]["params"].get("beatPulse").is_none(), "creator tools off by default");
    // the opt-in creator look and beat pulses
    let r = s.execute("autocut.trailer", json!({"look": "TECH", "beatPulse": true, "wait": true})).unwrap();
    assert_eq!((r["showcase"]["params"]["look"].as_str(), r["showcase"]["params"]["beatPulse"].as_bool()), (Some("tech"), Some(true)), "{r}");
    for bad in [
        json!({"look": "sparkly"}),
        json!({"look": 3}),
        json!({"beatPulse": "yes"}),
        json!({"seconds": 5}),
        json!({"seconds": 91}),
        json!({"seconds": f64::MAX}),
        json!({"seconds": "long"}),
        json!({"robots": 2}),
        json!({"robots": 31}),
        json!({"robots": -4}),
        json!({"bpm": 0}),
        json!({"bpm": 1e9}),
        json!({"seed": -1}),
        json!({"aspect": "16:9"}),
        json!({"aspect": 9}),
        json!({"platform": "snapchat"}),
        json!({"platform": null}),
        json!({"fallback": "stretch"}),
        json!({"redact": "elephants"}),
        json!({"redact": true}),
        json!({"redact": "faces,faces,plates"}),
        json!({"hookVoice": "yes"}),
        json!({"hookAt": -1}),
        json!({"hookAt": "late"}),
        json!({"at": [1]}),
        json!({"at": {"Ameca": -2}}),
        json!({"at": {"Ameca": {"item": "x", "seconds": 2}}}),
        json!({"at": {"Ameca": {"item": 3}}}),
        json!({"at": {"two\nlines": 3}}),
        json!({"qa": 1}),
        json!({"exclude": "Promobot"}),
        json!({"exclude": [1]}),
        json!({"exclude": ["two\nlines"]}),
        json!({"include": vec!["x"; 101]}),
        json!({"hook": "x".repeat(10_000)}),
        json!({"title": "a\u{0}b"}),
        json!({"export": "out\nfile.mp4"}),
        json!({"sequence": "main"}),
        json!({"plan": {"entries": 3}}),
        json!({"plan": {}, "planPath": "/x.json"}),
        json!({"report": [1]}),
        json!({"report": {}, "reportPath": "/x.json"}),
        json!({"reportPath": "/no/such/report.json"}),
        json!([1]),
    ] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute("autocut.trailer", bad.clone())));
        assert!(r.expect("no panic").is_err(), "{bad} was accepted");
    }
}
