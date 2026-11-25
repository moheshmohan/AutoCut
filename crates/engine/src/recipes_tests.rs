//! A4 recipes: the format and its checks, the built-ins, save / list / inspect / delete in the
//! recipes folder, running as a job (commands and planned intents, jobs waited for), a failing step
//! stopping the run, cancelling, recording an edit as a recipe; hostile recipes refused.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};

use crate::Session;
use crate::autocut::CommandPort;
use crate::media_test_util::tmp_dir;
use crate::recipes::{Planned, PlannedCommand, Recipe, RecipePlanner, builtin, builtin_json};

fn session(dir: &Path) -> Session {
    let mut s = Session { prefs_path: Some(dir.join("prefs.json")), ..Default::default() };
    s.execute("file.openDemoProject", json!({})).unwrap();
    s
}

/// A planner for tests: `add_marker` plans a marker at the playhead; `wait_for_cancel` blocks
/// until the job is cancelled; `two_rounds` asks for a second round once.
struct Fake {
    calls: AtomicUsize,
    rounds: AtomicUsize,
}

impl RecipePlanner for Fake {
    fn intents(&self) -> Vec<String> {
        ["add_marker", "wait_for_cancel", "two_rounds", "plans_a_forbidden_command"].map(String::from).to_vec()
    }
    fn plan(&self, intent: &str, params: &Value, port: &CommandPort) -> std::result::Result<Planned, String> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        // planners read the session through the port
        port.call("sequence.inspect", json!({}))?;
        let marker = |name: &str| PlannedCommand { id: "markers.add".into(), params: json!({"name": name}), why: "a marker".into() };
        match intent {
            "add_marker" => Ok(Planned { commands: vec![marker(params["name"].as_str().unwrap_or("planned"))], notes: vec!["planned".into()], replan: false }),
            "two_rounds" => {
                let r = self.rounds.fetch_add(1, Ordering::Relaxed);
                Ok(Planned { commands: vec![marker(&format!("round {r}"))], notes: Vec::new(), replan: r == 0 })
            }
            "plans_a_forbidden_command" => {
                Ok(Planned { commands: vec![PlannedCommand { id: "prefs.set".into(), params: json!({}), why: String::new() }], ..Default::default() })
            }
            _ => {
                while !port.cancelled() {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err("cancelled".into())
            }
        }
    }
}

fn with_planner(s: &mut Session) -> Arc<Fake> {
    let f = Arc::new(Fake { calls: AtomicUsize::new(0), rounds: AtomicUsize::new(0) });
    s.recipes.planner = Some(f.clone());
    f
}

fn recipe(steps: Value) -> Value {
    json!({"format": "filmcraft.recipe", "formatVersion": 1, "id": "user:t", "name": "T", "steps": steps})
}

fn markers(s: &Session) -> Vec<String> {
    s.active_sequence().map(|q| q.markers.iter().map(|m| m.name.clone()).collect()).unwrap_or_default()
}

/// The job's `jobs.list` entry, pumping the port (as the app does each frame) until it finishes.
fn finish(s: &mut Session, job: u64) -> Value {
    for _ in 0..5000 {
        s.pump_port(std::time::Duration::from_millis(5));
        let j = s.execute("jobs.list", json!({})).unwrap().as_array().unwrap().iter().find(|j| j["id"] == job).cloned().unwrap();
        if j["finished"] == true {
            return j;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    panic!("job {job} never finished");
}

#[test]
fn the_built_in_recipes_read_and_check() {
    let all = builtin();
    assert_eq!(all.len(), builtin_json().len());
    assert_eq!(
        all.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        [
            "builtin:talking-head-clean-up",
            "builtin:vertical-short",
            "builtin:template-publish-youtube",
            "builtin:delivery-quality-gate",
            "builtin:topic-explainer"
        ]
    );
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    let f = Arc::new(Fake { calls: AtomicUsize::new(0), rounds: AtomicUsize::new(0) });
    // with the real intents named (the AutoCut planner's), every command step exists and is allowed
    struct Names;
    impl RecipePlanner for Names {
        fn intents(&self) -> Vec<String> {
            ["remove_silence", "remove_fillers", "add_captions", "enhance_voice", "highlight_reel"].map(String::from).to_vec()
        }
        fn plan(&self, _: &str, _: &Value, _: &CommandPort) -> std::result::Result<Planned, String> {
            Ok(Planned::default())
        }
    }
    s.recipes.planner = Some(Arc::new(Names));
    for r in &all {
        let plan = s
            .execute(
                "recipe.run",
                json!({"name": r.id, "dryRun": true, "params": if r.id.ends_with("youtube") || r.id=="builtin:delivery-quality-gate" { json!({"output": "/tmp/x.mp4"}) } else { json!({}) }}),
            )
            .unwrap();
        assert!(!plan["plan"].as_array().unwrap().is_empty(), "{}", r.id);
    }
    // publishing is off by default: the publish step is skipped unless asked for
    let p = s.execute("recipe.run", json!({"name": "builtin:template-publish-youtube", "dryRun": true})).unwrap();
    let steps = p["plan"].as_array().unwrap();
    assert_eq!(steps.iter().find(|x| x["command"] == "publish.run").unwrap()["skipped"], true);
    assert_eq!(steps.iter().find(|x| x["command"] == "file.exportMedia").unwrap()["skipped"], true, "no output given: no export");
    let on = s
        .execute("recipe.run", json!({"name": "builtin:template-publish-youtube", "dryRun": true, "params": {"publish": true, "output": "/tmp/x.mp4"}}))
        .unwrap();
    let publish = on["plan"].as_array().unwrap().iter().find(|x| x["command"] == "publish.run").unwrap().clone();
    assert_eq!(
        (publish["skipped"].as_bool(), publish["params"]["privacy"].as_str(), publish["params"]["path"].as_str()),
        (Some(false), Some("private"), Some("/tmp/x.mp4"))
    );
    // publishing without a file to publish is refused before anything runs
    let e = s.execute("recipe.run", json!({"name": "builtin:template-publish-youtube", "params": {"publish": true}})).unwrap_err().to_string();
    assert!(e.contains("path"), "{e}");
    // without a planner, intent steps are refused up front
    s.recipes.planner = None;
    assert!(s.execute("recipe.run", json!({"name": "builtin:talking-head-clean-up", "dryRun": true})).is_err());
    let _ = f;
}

#[test]
fn save_list_inspect_and_delete_in_the_recipes_folder() {
    let dir = tmp_dir("recipes-folder");
    let mut s = session(&dir);
    with_planner(&mut s);
    let steps = json!([
        {"label": "Mark it", "intent": "add_marker", "params": {"name": "${label}"}},
        {"label": "Select all", "command": "edit.selectAll"},
        {"label": "Fill the frame", "command": "clip.fillFrame", "when": "fill"},
    ]);
    let params = json!([
        {"id": "label", "name": "Marker name", "kind": "text", "default": "start"},
        {"id": "fill", "name": "Fill the frame", "kind": "bool", "default": false},
    ]);
    let r = s.execute("recipe.save", json!({"name": "My Routine", "description": "mine", "steps": steps, "params": params})).unwrap();
    let path = r["path"].as_str().unwrap().to_string();
    assert!(path.ends_with("Recipes/my-routine.json"), "{path}");
    let on_disk: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(on_disk["format"], "filmcraft.recipe");
    assert_eq!(on_disk["id"], "user:my-routine");
    // listed with the built-ins
    let l = s.execute("recipe.list", json!({})).unwrap();
    let ids: Vec<&str> = l["recipes"].as_array().unwrap().iter().map(|r| r["id"].as_str().unwrap()).collect();
    assert_eq!(ids.len(), builtin_json().len() + 1, "the built-ins and the one saved: {ids:?}");
    assert!(ids.contains(&"user:my-routine"));
    assert!(l["userFolder"].as_str().unwrap().ends_with("Recipes"));
    // inspected by name, with parameters: the steps as they would run
    let i = s.execute("recipe.inspect", json!({"name": "my routine", "params": {"label": "hello", "fill": true}})).unwrap();
    assert_eq!(i["ready"], true, "{i}");
    let plan = i["plan"].as_array().unwrap();
    assert_eq!(plan[0]["params"]["name"], "hello");
    assert_eq!(plan[2]["skipped"], false);
    // saving again asks first
    assert!(s.execute("recipe.save", json!({"name": "My Routine", "steps": [{"command": "edit.selectAll"}]})).is_err());
    s.execute("recipe.save", json!({"name": "My Routine", "steps": [{"command": "edit.selectAll"}], "overwrite": true})).unwrap();
    assert_eq!(s.execute("recipe.inspect", json!({"name": "user:my-routine"})).unwrap()["steps"], 1);
    // a broken file in the folder is listed as invalid, not fatal
    std::fs::write(dir.join("Recipes").join("broken.json"), b"{\"format\": \"filmcraft.recipe\", \"steps\": 3}").unwrap();
    let l = s.execute("recipe.list", json!({})).unwrap();
    assert!(l["invalid"].as_array().unwrap().iter().any(|x| x["path"].as_str().unwrap().ends_with("broken.json")), "{l}");
    // built-ins can't be deleted; the user's can
    assert!(s.execute("recipe.delete", json!({"name": "builtin:vertical-short"})).is_err());
    s.execute("recipe.delete", json!({"name": "user:my-routine"})).unwrap();
    assert!(!Path::new(&path).exists());
    assert!(s.execute("recipe.inspect", json!({"name": "user:my-routine"})).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_run_plans_each_intent_runs_each_command_and_reports() {
    let dir = tmp_dir("recipes-run");
    let mut s = session(&dir);
    let f = with_planner(&mut s);
    let r = recipe(json!([
        {"label": "First marker", "intent": "add_marker", "params": {"name": "${first}"}},
        {"label": "Select all", "command": "edit.selectAll"},
        {"label": "Skipped", "intent": "add_marker", "params": {"name": "never"}, "when": "extra"},
        {"label": "Two rounds", "intent": "two_rounds"},
        {"label": "Deselect", "command": "edit.deselectAll"},
    ]));
    let mut r = r;
    r["params"] = json!([{"id": "first", "kind": "text", "default": "one"}, {"id": "extra", "kind": "bool", "default": false}]);
    let before = markers(&s).len();
    let out = s.execute("recipe.run", json!({"recipe": r, "params": {"first": "hello"}, "wait": true})).unwrap();
    let run = &out["recipeRun"];
    assert_eq!(run["ok"], true, "{out}");
    let names = markers(&s);
    assert_eq!(names.len(), before + 3, "{names:?}");
    assert!(names.contains(&"hello".to_string()) && !names.contains(&"never".to_string()));
    let steps = run["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 5);
    assert_eq!(steps[2]["skipped"], true);
    assert_eq!(steps[3]["commands"].as_array().unwrap().len(), 2, "a round, then the second");
    assert_eq!(steps[0]["notes"][0], "planned");
    assert!(f.calls.load(Ordering::Relaxed) >= 3);
    // the job: done, progress full
    let j = s.execute("jobs.list", json!({})).unwrap().as_array().unwrap().iter().find(|j| j["id"] == out["job"]).cloned().unwrap();
    assert_eq!((j["finished"].as_bool(), j["progress"].as_f64()), (Some(true), Some(1.0)));
    assert_eq!(j["recipe"]["ok"], true, "the outcome rides in jobs.list: {j}");
    // in the background: started, then pumped like the app does
    let job = s.execute("recipe.run", json!({"recipe": recipe(json!([{"intent": "add_marker"}]))})).unwrap()["job"].as_u64().unwrap();
    let j = finish(&mut s, job);
    assert!(j["result"]["error"].is_null(), "{j}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_failing_step_stops_the_run_and_says_which() {
    let dir = tmp_dir("recipes-fail");
    let mut s = session(&dir);
    with_planner(&mut s);
    let before = markers(&s).len();
    let r = recipe(json!([
        {"label": "Mark", "intent": "add_marker", "params": {"name": "before"}},
        {"label": "Apply a template that is not there", "command": "template.apply", "params": {"template": "builtin:no-such-template"}},
        {"label": "Never", "intent": "add_marker", "params": {"name": "after"}},
    ]));
    let job = s.execute("recipe.run", json!({"recipe": r})).unwrap()["job"].as_u64().unwrap();
    let j = finish(&mut s, job);
    let e = j["result"]["error"].as_str().unwrap();
    assert!(e.starts_with("step 2 of 3 (Apply a template that is not there) failed"), "{e}");
    let out = &j["recipe"];
    assert_eq!(out["ok"], false);
    assert_eq!(out["steps"].as_array().unwrap().len(), 2, "the third step never ran: {out}");
    assert!(out["steps"][1]["error"].as_str().unwrap().contains("no-such-template"));
    let names = markers(&s);
    assert_eq!(names.len(), before + 1);
    assert!(!names.contains(&"after".to_string()));
    // a planned command outside the allowlist stops the step before anything of it runs
    let r = recipe(json!([{"intent": "plans_a_forbidden_command"}]));
    let job = s.execute("recipe.run", json!({"recipe": r})).unwrap()["job"].as_u64().unwrap();
    let j = finish(&mut s, job);
    assert!(j["result"]["error"].as_str().unwrap().contains("not a command a recipe may run"), "{j}");
    // and the session is free for the next run
    let r = recipe(json!([{"intent": "add_marker"}]));
    let job = s.execute("recipe.run", json!({"recipe": r})).unwrap()["job"].as_u64().unwrap();
    assert!(finish(&mut s, job)["result"]["error"].is_null());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_run_is_cancelled_cleanly() {
    let dir = tmp_dir("recipes-cancel");
    let mut s = session(&dir);
    with_planner(&mut s);
    let before = markers(&s).len();
    let r = recipe(json!([{"label": "Wait", "intent": "wait_for_cancel"}, {"label": "Never", "intent": "add_marker"}]));
    let job = s.execute("recipe.run", json!({"recipe": r})).unwrap()["job"].as_u64().unwrap();
    // one at a time
    assert!(s.execute("recipe.run", json!({"recipe": recipe(json!([{"intent": "add_marker"}]))})).is_err());
    for _ in 0..20 {
        s.pump_port(std::time::Duration::from_millis(5));
    }
    s.execute("jobs.cancel", json!({"job": job})).unwrap();
    let j = finish(&mut s, job);
    assert_eq!(j["result"]["error"], "cancelled", "{j}");
    assert_eq!(markers(&s).len(), before, "nothing after the cancel ran");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn record_this_edit_as_a_recipe() {
    let dir = tmp_dir("recipes-record");
    let mut s = session(&dir);
    s.execute("recipe.record", json!({"action": "start"})).unwrap();
    assert!(s.execute("recipe.run", json!({"name": "builtin:vertical-short", "dryRun": false})).is_err(), "no runs while recording");
    s.execute("markers.add", json!({"name": "intro"})).unwrap();
    s.execute("edit.selectAll", json!({})).unwrap();
    s.execute("clip.fillFrame", json!({})).unwrap();
    // preferences and the like are not part of an edit
    let _ = s.execute("prefs.set", json!({"key": "general.showTooltips", "value": true}));
    let st = s.execute("recipe.record", json!({"action": "status"})).unwrap();
    let recorded: Vec<&str> = st["steps"].as_array().unwrap().iter().map(|x| x["command"].as_str().unwrap()).collect();
    assert!(recorded.contains(&"markers.add") && recorded.contains(&"clip.fillFrame"), "{recorded:?}");
    assert!(!recorded.iter().any(|c| c.starts_with("prefs.")));
    let r = s.execute("recipe.record", json!({"action": "stop", "name": "My Fill"})).unwrap();
    assert_eq!(r["recipe"]["id"], "user:my-fill", "{r}");
    assert!(s.recipes.recording.is_none());
    // it runs again on the same project
    let before = markers(&s).len();
    let out = s.execute("recipe.run", json!({"name": "My Fill", "wait": true})).unwrap();
    assert_eq!(out["recipeRun"]["ok"], true, "{out}");
    assert_eq!(markers(&s).len(), before + 1);
    // stopping when nothing records says so; cancel discards
    assert!(s.execute("recipe.record", json!({"action": "stop"})).is_err());
    s.execute("recipe.record", json!({"action": "start"})).unwrap();
    assert_eq!(s.execute("recipe.record", json!({"action": "cancel"})).unwrap()["recording"], false);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn hostile_recipes_and_parameters_are_refused() {
    let dir = tmp_dir("recipes-hostile");
    let mut s = session(&dir);
    with_planner(&mut s);
    let deep = (0..40).fold(json!(1), |v, _| json!({"x": v}));
    let many: Vec<Value> = (0..65).map(|_| json!({"command": "edit.selectAll"})).collect();
    let bad_recipes: Vec<Value> = vec![
        json!([]),
        json!({"format": "something.else", "formatVersion": 1, "id": "user:x", "name": "X", "steps": [{"command": "edit.selectAll"}]}),
        json!({"format": "filmcraft.recipe", "formatVersion": 99, "id": "user:x", "name": "X", "steps": [{"command": "edit.selectAll"}]}),
        recipe(json!([])),
        recipe(json!(many)),
        recipe(json!([{"command": "prefs.set", "params": {}}])),
        recipe(json!([{"command": "recipe.run", "params": {"name": "x"}}])),
        recipe(json!([{"command": "app.quit"}])),
        recipe(json!([{"command": "jobs.cancel", "params": {"job": 1}}])),
        recipe(json!([{"command": "file.newProject"}])),
        recipe(json!([{"command": "no.such.command"}])),
        recipe(json!([{"command": "edit.selectAll", "intent": "add_marker"}])),
        recipe(json!([{"command": "edit.selectAll", "params": [1, 2]}])),
        recipe(json!([{"command": "edit.selectAll", "params": {"deep": deep}}])),
        recipe(json!([{"command": "edit.selectAll", "params": {"x": "${nope}"}}])),
        recipe(json!([{"command": "edit.selectAll", "when": "nope"}])),
        recipe(json!([{"command": "edit.selectAll", "colour": "red"}])),
        recipe(json!([{"intent": "teleport"}])),
        json!({"format": "filmcraft.recipe", "formatVersion": 1, "id": "Bad Id!", "name": "X", "steps": [{"command": "edit.selectAll"}]}),
        json!({"format": "filmcraft.recipe", "formatVersion": 1, "id": "user:x", "name": "X", "params": [{"id": "n", "kind": "number", "default": "five"}], "steps": [{"command": "edit.selectAll"}]}),
        json!({"format": "filmcraft.recipe", "formatVersion": 1, "id": "user:x", "name": "X", "params": [{"id": "c", "kind": "choice", "choices": []}], "steps": [{"command": "edit.selectAll"}]}),
        json!({"format": "filmcraft.recipe", "formatVersion": 1, "id": "user:x", "name": "X", "params": [{"id": "k", "kind": "spell"}], "steps": [{"command": "edit.selectAll"}]}),
        json!({"format": "filmcraft.recipe", "formatVersion": 1, "id": "user:x", "name": "X", "params": [{"id": "n", "kind": "number", "min": 5, "max": 1}], "steps": [{"command": "edit.selectAll"}]}),
    ];
    for r in bad_recipes {
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute("recipe.run", json!({"recipe": r.clone(), "dryRun": true}))));
        assert!(out.is_ok(), "panicked on {r}");
        assert!(out.unwrap().is_err(), "accepted {r}");
    }
    // parameters at run time: type, range, unknown names
    let r = json!({"format": "filmcraft.recipe", "formatVersion": 1, "id": "user:p", "name": "P",
        "params": [{"id": "n", "kind": "number", "default": 1, "min": 0, "max": 10}, {"id": "c", "kind": "choice", "choices": ["a", "b"], "default": "a"}],
        "steps": [{"intent": "add_marker", "params": {"name": "${c}-${n}"}}]});
    for p in [json!({"n": "1"}), json!({"n": 11}), json!({"c": "z"}), json!({"zz": 1}), json!([1]), json!({"n": f64::MAX})] {
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            s.execute("recipe.run", json!({"recipe": r.clone(), "params": p.clone(), "dryRun": true}))
        }));
        assert!(out.is_ok_and(|o| o.is_err()), "{p}");
    }
    let ok = s.execute("recipe.run", json!({"recipe": r, "params": {"n": 3, "c": "b"}, "dryRun": true})).unwrap();
    assert_eq!(ok["plan"][0]["params"]["name"], "b-3");
    // files: too big, not JSON, junk in the folder
    assert!(Recipe::parse(&vec![b' '; 300 * 1024]).is_err());
    assert!(Recipe::parse(b"\xff\xfe").is_err());
    assert!(Recipe::parse(&b"[".repeat(10_000)).is_err());
    // commands' parameters
    for p in [
        json!({}),
        json!({"name": 3}),
        json!({"name": "", "steps": [{"command": "edit.selectAll"}]}),
        json!({"name": "X", "steps": "all"}),
        json!({"name": "X", "id": "builtin:x", "steps": [{"command": "edit.selectAll"}]}),
    ] {
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute("recipe.save", p.clone())));
        assert!(out.is_ok_and(|o| o.is_err()), "{p}");
    }
    for (cmd, p) in [
        ("recipe.run", json!({})),
        ("recipe.run", json!({"name": "nope"})),
        ("recipe.delete", json!({})),
        ("recipe.inspect", json!({"name": 7})),
        ("recipe.record", json!({"action": "dance"})),
    ] {
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute(cmd, p.clone())));
        assert!(out.is_ok_and(|o| o.is_err()), "{cmd} {p}");
    }
    std::fs::remove_dir_all(&dir).ok();
}
