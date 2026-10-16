//! The engine's showcase jobs with this crate as their host, on the demo project and scripted
//! models: make a showcase (analysis and plan), review it (drop, merge, relabel), write the
//! script again for the reviewed entries, build it, export it with its QA report and start the
//! Google Photos upload — all through engine commands, as the Showcase panel, the CLI and MCP
//! agents drive them. And the AutoCut prompt "make a video of every robot" starts the same job.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use filmcraft_autocut::host::{Models, SCRIPTED_LABELS, ShowcaseHost};
use filmcraft_autocut::llm::{FixedModel, FnModel};
use filmcraft_autocut::runner::{Options, Runners, run_with};
use filmcraft_autocut::{SessionExecutor, config::Config};
use filmcraft_engine::Session;
use serde_json::{Value, json};

fn demo_with(models: Models) -> Session {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s.autocut.host = Some(Arc::new(ShowcaseHost::with_models(models)));
    s
}

fn demo() -> Session {
    demo_with(Models::scripted())
}

fn entries(plan: &Value) -> Vec<Value> {
    plan["entries"].as_array().cloned().unwrap_or_default()
}

fn lines(plan: &Value) -> Vec<String> {
    plan["script"]["lines"].as_array().map(|a| a.iter().map(|l| l.as_str().unwrap_or_default().to_string()).collect()).unwrap_or_default()
}

#[test]
fn make_review_rewrite_and_build() {
    let mut s = demo();
    let r = s.execute("autocut.makeShowcase", json!({"subject": "robot", "title": "Robots", "wait": true})).unwrap();
    let out = &r["showcase"];
    assert_eq!(out["summary"]["showing"], 6, "every demo clip shows a (scripted) robot: {out}");
    let plan = out["plan"].clone();
    let es = entries(&plan);
    assert_eq!(es.len(), 6, "one entry per robot: {plan}");
    let labels: Vec<&str> = es.iter().filter_map(|e| e["label"].as_str()).collect();
    for l in SCRIPTED_LABELS {
        assert!(labels.contains(&l), "{l} in {labels:?}");
    }
    assert!(es.iter().all(|e| e["include"] == true && e["mergePrevious"] == false), "{plan}");
    assert_eq!(plan["title"], "Robots");
    assert_eq!(plan["script"]["intro"], "This is every robot I have met, in order.");
    assert!(lines(&plan).iter().all(|l| l.starts_with("Here is robot number")), "{plan}");
    // the report stays in the session
    let i = s.execute("autocut.showcaseInspect", json!({"report": true})).unwrap();
    assert_eq!(i["report"]["clips"].as_array().map(Vec::len), Some(6), "{i}");

    // the review: the second is not a robot, the fourth is the third again, the first is named
    let mut es = es;
    es[1]["include"] = json!(false);
    es[3]["mergePrevious"] = json!(true);
    es[0]["label"] = json!("Ruby the red robot");
    let r = s.execute("autocut.showcasePlan", json!({"entries": es, "voice": "", "wait": true})).unwrap();
    let plan = r["showcase"]["plan"].clone();
    assert_eq!(plan["title"], "Robots", "kept from the plan before");
    assert!(plan.get("voice").is_none(), "no narration: {plan}");
    let ls = lines(&plan);
    assert_eq!(ls.iter().map(|l| !l.is_empty()).collect::<Vec<_>>(), vec![true, false, true, false, true, true], "lines for the entries shown: {ls:?}");
    assert_eq!(entries(&plan)[0]["label"], "Ruby the red robot");
    assert_eq!((entries(&plan)[1]["include"].clone(), entries(&plan)[3]["mergePrevious"].clone()), (json!(false), json!(true)));

    // build it small: four entries, the third with the merged robot's shot too
    let mut plan = plan;
    plan["width"] = json!(320);
    plan["height"] = json!(180);
    let undo = s.history.undo.len();
    let r = s.execute("autocut.showcaseBuild", json!({"plan": plan, "wait": true})).unwrap();
    let built = &r["showcase"]["built"];
    assert_eq!(built["entries"], 4, "{built}");
    let layout = built["layout"].as_array().unwrap();
    assert_eq!(layout.iter().map(|e| e["label"].as_str().unwrap_or_default()).next(), Some("Ruby the red robot"));
    assert!(r["showcase"]["status"].as_str().unwrap_or_default().starts_with("Built 4 entries"), "{r}");
    assert_eq!(s.state.active_sequence.map(|i| i.0), built["sequence"].as_u64(), "the new sequence is open");
    assert_eq!(s.active_sequence().map(|q| q.settings.width), Some(320));
    assert!(s.history.undo.len() > undo + 10, "the build is ordinary, undoable edits");
    assert!(built.get("qa").is_none(), "no QA without an export unless asked");
    assert_eq!(s.execute("autocut.showcaseInspect", json!({})).unwrap()["built"]["sequence"], built["sequence"]);
}

/// Export, the QA report beside it and the upload, which runs as its own job.
#[test]
fn build_and_export_with_qa_and_upload() {
    struct Uploader;
    impl filmcraft_engine::photos::PhotosUploader for Uploader {
        fn ready(&self) -> Result<(), String> {
            Ok(())
        }
        fn upload(&self, _: &str, d: Option<&str>, _: &filmcraft_engine::export::Progress) -> Result<Value, String> {
            Ok(json!({"mediaItemId": "M1", "description": d}))
        }
    }
    let dir = std::env::temp_dir().join(format!("filmcraft-showcase-jobs-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("robots.mp4");
    let mut s = demo();
    s.photos = Some(Arc::new(Uploader));
    // one robot: the shortest show (the export is YouTube 1080p)
    s.execute("autocut.makeShowcase", json!({"subject": "robot", "items": [5], "title": "One Robot", "voice": "", "wait": true})).unwrap();
    let mut plan = s.execute("autocut.showcaseInspect", json!({})).unwrap()["plan"].clone();
    assert_eq!(entries(&plan).len(), 1, "{plan}");
    // few frames to encode
    plan["fps"] = json!(6);
    let r = s.execute("autocut.showcaseBuild", json!({"plan": plan, "export": out.to_string_lossy(), "upload": true, "wait": true})).unwrap();
    let built = &r["showcase"]["built"];
    assert!(out.exists(), "exported: {built}");
    assert!(dir.join("robots.qa.json").exists(), "the QA report beside it: {built}");
    assert!(built["qaSummary"].is_string(), "{built}");
    let upload = built["uploadJob"].as_u64().unwrap_or_else(|| panic!("an upload job: {built}"));
    let t0 = std::time::Instant::now();
    let job = loop {
        let jobs = s.execute("jobs.list", json!({})).unwrap();
        let j = jobs.as_array().unwrap().iter().find(|j| j["id"] == upload).cloned().unwrap();
        if j["finished"] == true {
            break j;
        }
        assert!(t0.elapsed() < Duration::from_secs(30), "{j}");
        std::thread::sleep(Duration::from_millis(10));
    };
    assert_eq!(job["upload"], json!({"mediaItemId": "M1", "description": "One Robot"}), "{job}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// "Make a video of every robot" needs no model to understand, and starts the showcase job.
#[tokio::test(flavor = "multi_thread")]
async fn a_showcase_prompt_starts_the_job() {
    let mut s = demo();
    s.execute("sequence.close", json!({})).unwrap();
    assert!(s.active_sequence().is_none(), "a showcase needs clips, not an open sequence");
    let ex = SessionExecutor::from(s);
    // the classifier has no answers: asking it would fail the run
    let runners = Runners { local: Box::new(FixedModel::new(vec![])), vlm: None };
    let r = run_with(&runners, "make a video of every robot", &ex, &Config::default(), Options::execute()).await.unwrap();
    assert_eq!((r.intent.as_deref(), r.needs_supervisor), (Some("showcase"), false), "{r:?}");
    assert_eq!(r.steps.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), ["autocut.makeShowcase"]);
    let res = r.results.as_ref().and_then(|v| v.first()).cloned().unwrap();
    assert!(res.ok, "{res:?}");
    assert_eq!(entries(&res.result.unwrap()["showcase"]["plan"]).len(), 6);
    assert_eq!(r.undo, Some(0), "nothing was edited yet");

    // a date the clips cannot match: the job says why
    let r = run_with(&runners, "make a video of every robot since 2020", &ex, &Config::default(), Options::execute()).await.unwrap();
    let res = r.results.as_ref().and_then(|v| v.first()).cloned().unwrap();
    let e = res.error.unwrap_or_default();
    assert!(!res.ok && e.contains("since 2020") && e.contains("no capture date"), "{e}");
}

/// Cancelling stops a job between model requests; nothing is kept.
#[test]
fn cancel_stops_an_analysis() {
    let slow = FnModel::new("slow", |_, _, _| {
        std::thread::sleep(Duration::from_millis(30));
        Ok(json!({"present": true, "bbox_2d": [1, 1, 400, 400]}))
    });
    let models = Models { analysis: Arc::new(slow), script: Models::scripted().script };
    let mut s = demo_with(models);
    let job = s.execute("autocut.analyze", json!({"subject": "robot"})).unwrap()["job"].as_u64().unwrap();
    let t0 = std::time::Instant::now();
    let mut cancelled = false;
    let entry = loop {
        s.pump_port(Duration::from_millis(8));
        let j = s.jobs.iter().find(|j| j.id == job).unwrap();
        if !cancelled && j.progress.done.load(Ordering::Relaxed) >= 1 {
            s.execute("jobs.cancel", json!({"job": job})).unwrap();
            cancelled = true;
        }
        let v = s.execute("jobs.list", json!({})).unwrap();
        let e = v.as_array().unwrap().iter().find(|j| j["id"] == job).cloned().unwrap();
        if e["finished"] == true {
            break e;
        }
        assert!(t0.elapsed() < Duration::from_secs(60), "{e}");
        std::thread::sleep(Duration::from_millis(2));
    };
    assert!(cancelled, "{entry}");
    assert_eq!(entry["result"]["error"], "cancelled", "{entry}");
    assert!(t0.elapsed() < Duration::from_secs(20), "stopped soon after the cancel");
    assert_eq!(s.execute("autocut.showcaseInspect", json!({})).unwrap()["report"], Value::Null);
}

/// A plan that does not hold together fails its job with the reason, before anything is built.
#[test]
fn a_bad_plan_fails_its_job() {
    let mut s = demo();
    let undo = s.history.undo.len();
    for (plan, why) in [
        (json!({"entries": [{"label": "no number"}]}), "not a showcase plan"),
        (json!({"entries": []}), "no entry is included"),
        (json!({"entries": [{"number": 1, "label": "x", "kind": "", "description": "", "unix": null, "shots": [], "include": false}]}), "no entry is included"),
    ] {
        let e = s.execute("autocut.showcaseBuild", json!({"plan": plan, "wait": true})).unwrap_err().to_string();
        assert!(e.contains(why), "{e}");
    }
    assert_eq!(s.history.undo.len(), undo);
}
