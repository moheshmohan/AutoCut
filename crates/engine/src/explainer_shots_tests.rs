//! `script.shots` and the `builtin:topic-explainer` recipe on explainers laid out by
//! `script.timeline` with the built-in voice: cards that keep to the explainer rules, renders
//! through a stand-in RAEA, the recipe's dry run and its run, hostile parameters.
use filmcraft_project::ClipId;
use serde_json::{Value, json};

use super::*;
use crate::Session;

const SCRIPT: &str = "# Why the sky is blue\n\
[shot: sunlight and a prism]\n\
Sunlight looks white. It holds every colour of the rainbow, from red to violet.\n\
\n\
[shot: 1871]\n\
Air scatters the short blue waves the most, so blue comes at us from every side of the sky.\n\
\n\
# Sunsets\n\
At sunset the light crosses far more air. The blue is scattered away on the way, and red is left.\n";

fn explainer(w: u32, h: u32, estimate: bool) -> (Session, Value) {
    let mut s = Session::default();
    let r = s
        .execute("script.timeline", json!({"text": SCRIPT, "voice": "basic-female", "estimate": estimate, "place": {"width": w, "height": h, "fps": 30}}))
        .unwrap();
    (s, r)
}

fn codes(r: &Value) -> Vec<String> {
    r["findings"].as_array().unwrap().iter().map(|f| f["code"].as_str().unwrap().to_string()).collect()
}

#[test]
fn cards_follow_the_shots_and_keep_to_the_explainer_rules() {
    let (mut s, timeline) = explainer(640, 360, false);
    let undo0 = s.history.undo.len();
    let plan = s.execute(CMD, json!({"dryRun": true})).unwrap();
    assert_eq!(plan["via"], "cards");
    assert!(plan["why"].as_str().unwrap().contains("no `source`"), "{plan:#}");
    assert_eq!(s.history.undo.len(), undo0);
    let r = s.execute(CMD, json!({})).unwrap();
    let shots = timeline["shots"].as_array().unwrap();
    let clips = r["clips"].as_array().unwrap();
    assert_eq!(clips.len(), shots.len());
    let q = s.active_sequence().unwrap();
    for (sh, c) in shots.iter().zip(clips) {
        let (track, it) = q.find_item(ClipId(c.as_u64().unwrap())).unwrap();
        assert_eq!(track, q.video_tracks[0].id, "cards go on V1");
        assert_eq!((it.start.0, it.end().0), (sh["start"].as_i64().unwrap(), sh["end"].as_i64().unwrap()), "a card spans its shot");
    }
    // the hero is the hint (else the chapter's title), at least a quarter of the frame's height
    assert_eq!(r["shots"][0]["hero"], "sunlight and a prism");
    assert_eq!(r["shots"][2]["hero"], "Sunsets");
    assert!(r["heroHeights"].as_array().unwrap().iter().all(|h| h.as_f64().unwrap() >= 0.25), "{:#}", r["heroHeights"]);
    assert!(r.get("warnings").is_none());
    // one undo step
    assert_eq!(s.history.undo.len(), undo0 + 1);
    // the explainer rules pass: no still over 3 s, a big hero, a settle before every cut, and no
    // two neighbours entering alike
    let craft = s.execute("sequence.checkCraft", json!({"explainer": true})).unwrap();
    let found = codes(&craft);
    assert!(!found.iter().any(|c| c == "longStill" || c == "smallHero"), "{found:?}: {craft:#}");
    assert_eq!(craft["checked"]["heroes"]["measured"].as_u64(), Some(shots.len() as u64));
    let motion = s.execute("sequence.checkMotion", json!({"explainer": true})).unwrap();
    let found = codes(&motion);
    assert!(!found.iter().any(|c| matches!(c.as_str(), "noSettle" | "sameEntrance" | "entranceDuration" | "overshoot" | "edgePop")), "{found:?}: {motion:#}");
    // it renders: the backdrop's colour fills the frame mid-shot
    let dir = std::env::temp_dir().join(format!("fc-shots-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let png = dir.join("card.png");
    let mid = (shots[0]["start"].as_i64().unwrap() + shots[0]["end"].as_i64().unwrap()) / 2;
    s.execute("file.exportFrame", json!({"path": png.to_string_lossy(), "format": "png", "target": "program", "time": mid})).unwrap();
    let img = image::open(&png).unwrap().to_rgba8();
    assert_eq!(img.get_pixel(4, 4).0[..3], [20, 33, 61], "the backdrop");
    let _ = std::fs::remove_dir_all(&dir);
    // the picture track is taken now: a second run is refused
    let e = s.execute(CMD, json!({})).unwrap_err().to_string();
    assert!(e.contains("has clips under the shots"), "{e}");
}

#[test]
fn hostile_parameters_are_refused_without_a_panic() {
    let (mut s, _) = explainer(640, 360, true);
    let cases: Vec<(Value, &str)> = vec![
        (json!({"via": "film"}), "via"),
        (json!({"via": "motion"}), "cannot render"),
        (json!({"source": "relative/comp.tsx"}), "absolute path"),
        (json!({"output": 7}), "absolute path"),
        (json!({"track": "V99"}), "video track"),
        (json!({"track": 2}), "V<n>"),
        (json!({"style": {"background": "navy"}}), "#rrggbb"),
        (json!({"style": {"sparkle": true}}), "not a card setting"),
        (json!({"style": {"font": ""}}), "font"),
        (json!({"exact": "yes"}), "true or false"),
        (json!({"zoom": 2}), "unknown parameter"),
        (json!([1]), "object"),
    ];
    for (p, want) in cases {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute(CMD, p.clone())));
        let e = r.unwrap_or_else(|_| panic!("panicked on {p}")).err().unwrap_or_else(|| panic!("accepted {p}")).to_string();
        assert!(e.contains(want), "{p}: {e}");
    }
    // a sequence without shot markers
    let mut t = Session::default();
    t.execute("file.newSequence", json!({"width": 640, "height": 360, "fps": 30})).unwrap();
    assert!(t.execute(CMD, json!({})).unwrap_err().to_string().contains("no shot markers"));
    // a long hint still makes a card, smaller, and says so
    let mut u = Session::default();
    let long = format!("# A\n[shot: {}]\nWords to say here.\n", "a very long hint about many things ".repeat(8));
    u.execute("script.timeline", json!({"text": long, "estimate": true, "place": {"width": 640, "height": 360, "fps": 30}})).unwrap();
    let r = u.execute(CMD, json!({})).unwrap();
    assert_eq!(r["clips"].as_array().unwrap().len(), 1);
}

#[cfg(unix)]
#[test]
fn each_shot_renders_through_raea_in_turn() {
    let mut b = crate::motion_graphics::tests::bench("explainer-shots");
    // the stand-in renders a 1 s 128×72 movie whatever it is asked: a sequence of that size
    b.s.execute("script.timeline", json!({"text": SCRIPT, "estimate": true, "place": {"width": 128, "height": 72, "fps": 30}})).unwrap();
    let out = b.dir.join("renders");
    let plan = b.s.execute(CMD, json!({"source": b.source(), "output": out.to_string_lossy(), "dryRun": true})).unwrap();
    assert_eq!(plan["via"], "motion", "a source and RAEA: compositions ({plan:#})");
    assert_eq!(plan["shots"][1]["composition"], "shot-02");
    // exact: a render must last its shot, and the 1 s stand-in does not
    let e = b.s.execute(CMD, json!({"source": b.source(), "output": b.dir.join("exact").to_string_lossy(), "wait": true})).unwrap_err().to_string();
    assert!(e.contains("shot-01"), "the first shot stops the run: {e}");
    let r = b.s.execute(CMD, json!({"source": b.source(), "output": out.to_string_lossy(), "exact": false, "wait": true})).unwrap();
    assert!(r["job"].as_u64().is_some(), "{r:#}");
    // every shot's composition rendered, in order, each into its own revision folder, and placed at its shot
    let renders: Vec<String> = b.calls().into_iter().filter(|c| c.starts_with("render ")).collect();
    let mine: Vec<&String> = renders.iter().filter(|c| c.contains(&out.to_string_lossy().to_string())).collect();
    assert_eq!(mine.len(), 3, "{renders:#?}");
    for (i, c) in mine.iter().enumerate() {
        assert!(c.contains(&format!("--composition shot-0{}", i + 1)), "{c}");
    }
    for id in ["shot-01", "shot-02", "shot-03"] {
        assert!(out.join(id).join(crate::motion_graphics::IMPORT_MANIFEST).is_file(), "{id}");
    }
    let q = b.s.active_sequence().unwrap();
    let starts: Vec<i64> = q.markers.iter().filter(|m| crate::explainer::shot_index(&m.name).is_some()).map(|m| m.start.0).collect();
    let placed: Vec<i64> = q.video_tracks.iter().flat_map(|t| t.items.iter().map(|i| i.start.0)).collect();
    for s0 in starts {
        assert!(placed.contains(&s0), "a render at {s0}: {placed:?}");
    }
}

#[test]
fn the_recipe_plans_its_steps_then_builds_a_checked_explainer() {
    let mut s = Session::default();
    let facts = json!([{"text": "Lord Rayleigh explained the blue sky in 1871", "source": "https://example.org/rayleigh"}]);
    // the dry run lists the steps as they will run, the ledger steps on with facts
    let plan = s.execute("recipe.run", json!({"name": "builtin:topic-explainer", "dryRun": true, "params": {"script": SCRIPT, "facts": facts}})).unwrap();
    let steps = plan["plan"].as_array().unwrap();
    let labels: Vec<(&str, bool)> = steps.iter().map(|x| (x["command"].as_str().unwrap(), x["skipped"].as_bool().unwrap())).collect();
    assert_eq!(
        labels,
        [
            ("script.timeline", false),
            ("script.factLedger", false),
            ("script.timeline", false),
            ("script.shots", false),
            ("graphics.chapterBar", false),
            ("sequence.checkCraft", false),
            ("sequence.checkMotion", false),
            ("script.factLedger", false),
            ("review.packet", false),
            ("delivery.preflight", false),
        ]
    );
    assert_eq!(steps[2]["params"]["text"], SCRIPT, "the script reaches the step whole");
    assert_eq!(steps[2]["params"]["place"]["width"], 1920);
    // parameters are checked before anything runs
    let e = s.execute("recipe.run", json!({"name": "builtin:topic-explainer", "dryRun": true, "params": {"language": "fr"}})).unwrap_err().to_string();
    assert!(e.contains("language"), "{e}");
    let e = s.execute("recipe.run", json!({"name": "builtin:topic-explainer", "dryRun": true, "params": {"fps": 1000}})).unwrap_err().to_string();
    assert!(e.contains("out of range"), "{e}");
    let e =
        s.execute("recipe.run", json!({"name": "builtin:topic-explainer", "dryRun": true, "params": {"script": "x".repeat(70_000)}})).unwrap_err().to_string();
    assert!(e.contains("script"), "{e}");
    // the run builds the explainer (no review packet: the project is not saved)
    let r = s
        .execute(
            "recipe.run",
            json!({"name": "builtin:topic-explainer", "wait": true, "params": {"script": SCRIPT, "facts": facts, "title": "Blue sky", "width": 640, "height": 360, "review": false}}),
        )
        .unwrap();
    let run = &r["recipeRun"];
    assert_eq!(run["ok"], true, "{run:#}");
    let q = s.active_sequence().unwrap();
    let name = s.project.items.values().find(|i| matches!(i.kind, filmcraft_project::ItemKind::Sequence(_))).map(|i| i.name.clone());
    assert_eq!(name.as_deref(), Some("Blue sky"));
    assert_eq!((q.settings.width, q.settings.height), (640, 360));
    assert_eq!(q.audio_tracks[0].items.len(), 3, "a narration clip a shot");
    assert_eq!(q.video_tracks[0].items.len(), 3, "a card a shot");
    assert!(q.video_tracks.iter().any(|t| t.name == "Chapter Bar" && t.items.len() == 2), "a chapter bar clip a chapter");
    assert_eq!(q.markers.iter().filter(|m| m.kind == filmcraft_project::MarkerKind::Chapter).count(), 2);
    assert!(!q.caption_tracks.is_empty());
    // the facts were checked on the script and on screen, and the gate ran with the explainer rules
    let ran: Vec<&str> = run["steps"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|x| x["skipped"] != true)
        .flat_map(|x| x["commands"].as_array().unwrap().iter())
        .map(|c| c["command"].as_str().unwrap())
        .collect();
    assert_eq!(ran.iter().filter(|c| **c == "script.factLedger").count(), 2, "{ran:?}");
    assert!(ran.contains(&"delivery.preflight"));
    let gate = s.execute("delivery.preflight", json!({"explainer": true, "facts": facts, "privacy": false, "mix": false, "motionEnergy": false})).unwrap();
    let fact = gate["checks"].as_array().unwrap().iter().find(|c| c["check"] == "factSources").unwrap();
    assert_eq!(fact["tier"], "floor");
}
