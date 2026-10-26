//! The director's rules, the model's plan checked, the commands, and a whole edit of the demo
//! project (one clip taken as 360°) through a session.

use super::*;
use crate::llm::FixedModel;

fn opts(p: Value) -> Options {
    Options::from_params(&p).unwrap()
}

fn look(id: &str, yaw: f64, what: &str, named: bool) -> Look {
    Look { id: id.into(), yaw, pitch: 0.0, what: what.into(), named }
}

/// A candidate: `kind`, its shot from `start` for `secs` of a clip `media` seconds long.
fn cand(id: &str, item: u64, kind: Kind, start: f64, secs: f64, media: f64, order: usize) -> Candidate {
    Candidate {
        id: id.into(),
        item,
        name: format!("clip{item}"),
        kind,
        start,
        end: start + secs,
        key: start + secs / 2.0,
        media_seconds: media,
        caption: format!("shot {id}"),
        energy: 0.6,
        quality: 0.7,
        hook: 0.5,
        wide: false,
        indoor: false,
        moving: true,
        handheld: false,
        talking: false,
        said: None,
        looks: if kind == Kind::Vr { vec![look(&format!("{id}a"), 0.0, "a person", true), look(&format!("{id}b"), 120.0, "a dog", true)] } else { Vec::new() },
        subject: None,
        aspect: 16.0 / 9.0,
        order,
    }
}

fn library() -> Vec<Candidate> {
    vec![
        cand("c1", 1, Kind::Vr, 0.0, 12.0, 40.0, 0),
        cand("c2", 2, Kind::Flat, 0.0, 6.0, 6.0, 1),
        cand("c3", 3, Kind::Vr, 5.0, 10.0, 30.0, 2),
        cand("c4", 4, Kind::Flat, 2.0, 4.0, 8.0, 3),
        cand("c5", 5, Kind::Flat, 0.0, 5.0, 5.0, 4),
        cand("c6", 6, Kind::Vr, 0.0, 15.0, 15.0, 5),
    ]
}

#[test]
fn briefs_say_the_layout_length_title_and_mood() {
    assert_eq!(aspect_from_brief("a 60 s reel of our trip"), Some("9:16"));
    assert_eq!(aspect_from_brief("a 16:9 recap"), Some("16:9"));
    assert_eq!(aspect_from_brief("something nice"), None);
    assert_eq!(seconds_from_brief("a 60 s reel"), Some(60.0));
    assert_eq!(seconds_from_brief("a 45-second recap"), Some(45.0));
    assert_eq!(seconds_from_brief("1 minute please"), Some(60.0));
    assert_eq!(seconds_from_brief("90s"), Some(90.0));
    assert_eq!(seconds_from_brief("3 cats"), None);
    assert_eq!(seconds_from_brief("a 9000 s epic"), None, "out of range");
    assert_eq!(title_from_brief("a 60 s reel of our trip to Dubai").as_deref(), Some("Our Trip To Dubai"));
    assert_eq!(title_from_brief("make a catchy reel").as_deref(), None);
    assert!(english("Our Trip") && !english("ഞങ്ങളുടെ യാത്ര"));
    assert!(indoors("Three men inside an elevator") && indoors("a luxurious hotel lobby") && !indoors("a car drives down a busy street"));
    assert_eq!(mood_from_brief("a calm sunset recap"), "calm");
    let o = opts(json!({"brief": "a 16:9 recap of the weekend, 45 seconds"}));
    assert_eq!((o.aspect.as_str(), o.seconds, o.vertical()), ("16:9", 45.0, false));
    assert_eq!(o.frame(), (1920, 1080));
    assert!(o.redact.is_none(), "privacy is asked for, not assumed");
    for bad in [json!({"aspect": "3:2"}), json!({"mood": "spooky"}), json!({"bpm": 400}), json!({"title": "ഹായ്"})] {
        assert!(Options::from_params(&bad).is_err(), "{bad}");
    }
}

#[test]
fn a_long_shot_gives_several_moments() {
    let m = crate::quick::MediaItem { item: 1, name: "long.insv".into(), still: false, seconds: 31.0, order: 0 };
    let shots = crate::quick::parse_shots(&m, None);
    let p = pieces(&shots[0]);
    assert_eq!(p.len(), 4, "{p:?}");
    assert_eq!((p[0].0, p[3].1), (0.0, 31.0));
    assert!(p.iter().all(|(a, b, k)| a < b && *k >= *a && *k <= *b));
    assert!(p.iter().any(|(a, b, _)| shots[0].keyframe >= *a && shots[0].keyframe < *b && p.iter().any(|x| x.2 == shots[0].keyframe)), "{p:?}");
    let short = crate::quick::MediaItem { seconds: 9.0, ..m.clone() };
    assert_eq!(pieces(&crate::quick::parse_shots(&short, None)[0]).len(), 1);
    // each piece a candidate of its own, its id unique
    let vr: BTreeSet<u64> = [1].into_iter().collect();
    let c = candidates(&shots, &[m], &vr, &HashMap::new(), &HashMap::new(), &[]);
    assert_eq!(c.len(), 4);
    let ids: BTreeSet<&str> = c.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(ids.len(), 4);
    assert!(c.iter().all(|c| c.kind == Kind::Vr && !c.looks.is_empty() && c.caption.contains("moment")));
}

#[test]
fn every_camera_reads_back_and_knows_its_kinds() {
    for c in Camera::ALL {
        assert_eq!(Camera::from_id(c.id()), Some(c));
        assert!(c.for_kind(Kind::Vr));
        assert!(c.min_seconds() > 0.0);
    }
    assert!(!Camera::PlanetIn.for_kind(Kind::Flat) && Camera::PushIn.for_kind(Kind::Photo));
    assert_eq!(Camera::from_id("tiny planet"), Some(Camera::PlanetIn));
    assert_eq!(Camera::from_id("loop"), None);
}

#[test]
fn rules_open_on_a_planet_alternate_and_close_on_one() {
    let o = opts(json!({"brief": "a 30 s reel of our trip", "seconds": 30}));
    let beat = beat_of(120.0);
    let plan = rules_plan(&library(), &o, beat, 120.0);
    assert_eq!(plan.by, "rules");
    let first = &plan.shots[0];
    assert_eq!((first.camera, first.text.as_deref()), (Camera::PlanetIn, Some("Our Trip")));
    let last = plan.shots.last().unwrap();
    assert_eq!((last.camera, last.enter), (Camera::PlanetOut, Enter::Dissolve));
    assert_eq!(plan.beats(), 60, "30 s at 120 bpm: {:?}", plan.shots);
    let kinds: Vec<Kind> = plan.shots.iter().map(|s| library().iter().find(|c| c.id == s.clip).unwrap().kind).collect();
    assert!(kinds.windows(3).all(|w| !(w[0] == w[1] && w[1] == w[2])), "the kinds take turns: {kinds:?}");
    // each camera on its kind, long enough, inside its clip
    for s in &plan.shots {
        let c = library().into_iter().find(|c| c.id == s.clip).unwrap();
        assert!(s.camera.for_kind(c.kind), "{s:?}");
        let secs = f64::from(s.beats) * beat;
        assert!(secs + 1e-9 >= s.camera.min_seconds(), "{s:?}");
        assert!(s.from >= 0.0 && s.from + secs * s.camera.speed() <= c.media_seconds + 1e-6, "{s:?}");
    }
    assert!(plan.shots.iter().filter(|s| s.camera == Camera::BarrelRoll).count() <= 1);
    // flat clips only: no 360 moves, still a plan
    let flat: Vec<Candidate> = library().into_iter().filter(|c| c.kind == Kind::Flat).collect();
    let p = rules_plan(&flat, &o, beat, 120.0);
    assert!(!p.shots.is_empty() && p.shots.iter().all(|s| s.camera.for_kind(Kind::Flat)));
    // nothing to plan with
    assert!(rules_plan(&[], &o, beat, 120.0).shots.is_empty());
}

#[test]
fn the_models_plan_is_checked_and_repaired() {
    let o = opts(json!({"brief": "a 20 s reel", "seconds": 20}));
    let beat = beat_of(120.0);
    let lib = library();
    let answer = json!({
        "title": "Weekend Away",
        "mood": "happy",
        "shots": [
            {"clip": "c1", "beats": 2, "camera": "planetIn", "look": "c1a", "enter": "whip"},
            {"clip": "c9", "beats": 4, "camera": "hold"},
            {"clip": "c2", "beats": 4, "camera": "orbit"},
            {"clip": "c3", "beats": 6, "camera": "whipPan", "look": "c3a", "to": "c3a"},
            {"clip": "c3", "beats": 6, "camera": "barrelRoll", "look": "c3b"},
            {"clip": "c6", "beats": 6, "camera": "barrelRoll", "text": "ഹായ്"},
            {"clip": "c5", "beats": 99, "camera": "pushIn", "enter": "dissolve"}
        ]
    });
    let plan = validate(&answer, &lib, &o, beat, 120.0).unwrap();
    assert_eq!(plan.title, "Weekend Away");
    assert_eq!(plan.beats(), 40, "20 s at 120 bpm: {:?}", plan.shots);
    assert!(plan.shots.iter().all(|s| s.clip != "c9"), "an unknown clip is left out");
    let first = &plan.shots[0];
    assert!(first.beats >= beats_for(Camera::PlanetIn.min_seconds(), beat), "a planet needs its beats");
    assert_eq!(first.enter, Enter::Cut, "nothing to whip from before the first shot");
    let c2 = plan.shots.iter().find(|s| s.clip == "c2").unwrap();
    assert!(c2.camera.for_kind(Kind::Flat), "no orbit on a flat clip: {c2:?}");
    let whip = plan.shots.iter().find(|s| s.camera == Camera::WhipPan).unwrap();
    assert_ne!(whip.look, whip.to, "a whip turns between two things");
    assert_eq!(plan.shots.iter().filter(|s| s.camera == Camera::BarrelRoll).count(), 1, "one barrel roll");
    assert!(plan.shots.iter().all(|s| s.text.as_deref().is_none_or(english)), "English text only");
    assert!(plan.notes.iter().any(|n| n.contains("c9")), "{:?}", plan.notes);
    // indoors, no reveal from the sky
    let mut inside = lib.clone();
    inside[2].indoor = true;
    let p = validate(&json!({"shots": [{"clip": "c3", "beats": 40, "camera": "reveal"}]}), &inside, &o, beat, 120.0).unwrap();
    assert_eq!(p.shots[0].camera, Camera::Drift, "{:?}", p.notes);
    // the plan wrapped as the example was, or under another name
    for wrapped in [json!({"answer": {"shots": [{"clip": "c1", "beats": 40, "camera": "orbit"}]}}), json!({"shot_list": [{"clip": "c1", "beats": 40}]})] {
        assert_eq!(validate(&wrapped, &lib, &o, beat, 120.0).unwrap().shots[0].clip, "c1", "{wrapped}");
    }
    // nothing usable
    let e = validate(&json!({"title": "x", "cuts": 3}), &lib, &o, beat, 120.0).unwrap_err();
    assert!(e.contains("title") && e.contains("cuts"), "{e}");
    assert!(validate(&json!({"shots": []}), &lib, &o, beat, 120.0).is_err());
    assert!(validate(&json!({"shots": [{"clip": "nope"}]}), &lib, &o, beat, 120.0).is_err());
    assert!(validate(&json!("garbage"), &lib, &o, beat, 120.0).is_err());
}

#[test]
fn a_plan_compiles_to_commands_on_its_beats() {
    let o = opts(json!({"brief": "a 30 s reel of our trip", "seconds": 30}));
    let beat = beat_of(120.0);
    let lib = library();
    let mut plan = rules_plan(&lib, &o, beat, 120.0);
    // a whip into the second 360 shot
    if let Some(i) = plan.shots.iter().position(|s| lib.iter().any(|c| c.id == s.clip && c.kind == Kind::Vr) && s.enter == Enter::Cut).filter(|i| *i > 0) {
        plan.shots[i].enter = Enter::Whip;
    }
    let wins = windows(&plan, &Beats::grid(120.0));
    assert_eq!(wins.len(), plan.shots.len());
    assert_eq!(wins[0].0, 0.0);
    for w in wins.windows(2) {
        assert!((w[0].1 - w[1].0).abs() < 1e-9, "back to back: {w:?}");
    }
    for (s, w) in plan.shots.iter().zip(&wins) {
        assert!(((w.1 - w.0) - f64::from(s.beats) * beat).abs() < 1.0 / FPS + 1e-9, "{s:?} {w:?}");
    }
    let steps = compile(&plan, &lib, &wins, &o, (1080.0, 1920.0), beat);
    let places = steps.iter().filter(|s| s.on == On::Place).count();
    assert_eq!(places, plan.shots.len());
    assert!(steps.iter().any(|s| s.id == "clip.vrMove" && s.params["move"] == "planetIn"));
    assert!(steps.iter().any(|s| s.id == "graphics.newText" && s.params["text"] == "Our Trip"));
    let sounds: Vec<&str> = steps.iter().filter(|s| s.id == "audio.generate").filter_map(|s| s.params["kind"].as_str()).collect();
    assert!(sounds.iter().all(|k| matches!(*k, "swish" | "chime")), "never a whoosh or a riser: {sounds:?}");
    assert!(sounds.contains(&"chime"));
    // every shot's own step comes after its placement
    let mut placed = BTreeSet::new();
    for s in &steps {
        if s.on == On::Place {
            placed.insert(s.shot);
        } else if s.shot.is_some() {
            assert!(placed.contains(&s.shot), "{s:?}");
        }
    }
    // the steps are engine commands
    let ids: BTreeSet<&str> = steps.iter().map(|s| s.id.as_str()).collect();
    for id in ids {
        assert!(filmcraft_engine::find_command(id).is_some(), "{id} is not an engine command");
    }
}

/// The demo project with its first clip's media taken as 360°.
fn demo_session() -> crate::SessionExecutor {
    let mut s = filmcraft_engine::Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    let item = s.active_sequence().unwrap().video_tracks[0].items[0].item;
    s.execute("media.vrProperties", json!({"items": [item.0], "projection": "equirectangular"})).unwrap();
    crate::SessionExecutor(std::sync::Arc::new(std::sync::Mutex::new(s)))
}

#[tokio::test]
async fn a_dry_run_plans_and_lists_the_commands() {
    let ex = demo_session();
    let mut lines = |_: &str| {};
    let r = make(&ex, &json!({"brief": "a 12 s reel of the demo", "seconds": 12, "dryRun": true, "captions": false}), None, &mut lines).await.unwrap();
    assert_eq!(r["director"], "rules", "{r}");
    assert!(r["fallbacks"].as_array().unwrap().iter().any(|f| f.as_str().unwrap().contains("no local model")), "{r}");
    assert!(r["vrMedia"].as_u64().unwrap() >= 1, "{r}");
    let steps = r["steps"].as_array().unwrap();
    assert!(steps.iter().any(|s| s["id"] == "timeline.place"), "{r}");
    assert!(r["sequence"].is_null(), "nothing built");
}

#[tokio::test]
async fn the_model_directs_a_whole_edit() {
    let ex = demo_session();
    // the model is shown the shortlist; it answers with clip ids it was given
    let short_ids: Vec<String> = {
        let mut lines = |_: &str| {};
        let r = make(&ex, &json!({"seconds": 10, "dryRun": true}), None, &mut lines).await.unwrap();
        r["plan"]["shots"].as_array().unwrap().iter().filter_map(|s| s["clip"].as_str().map(str::to_string)).collect()
    };
    let shots: Vec<Value> = short_ids.iter().map(|c| json!({"clip": c, "beats": 4, "camera": "pushIn"})).collect();
    let model = FixedModel::new(vec![json!({"title": "Demo Day", "mood": "happy", "shots": shots})]);
    let mut lines = |_: &str| {};
    let r = make(&ex, &json!({"brief": "a 10 s reel", "seconds": 10, "bpm": 120, "captions": false}), Some(&model), &mut lines).await.unwrap();
    assert_eq!(r["director"], "model:fixed", "{r}");
    assert_eq!(r["plan"]["title"], "Demo Day");
    let seq = r["sequence"].as_u64().expect("a sequence");
    let s = ex.0.lock().unwrap();
    let q = s.project.sequence(filmcraft_engine::project::ItemId(seq)).unwrap();
    assert_eq!((q.settings.width, q.settings.height), (1080, 1920));
    let pictures = &q.video_tracks[0].items;
    assert!(!pictures.is_empty());
    // back to back from 0, the music under them, the title over the first
    assert_eq!(pictures[0].start, filmcraft_time::Tick::ZERO);
    assert!(q.audio_tracks.iter().flat_map(|t| &t.items).any(|c| s.project.item(c.item).is_some_and(|i| i.name.starts_with("Music"))), "generated music");
    assert!(q.video_tracks.get(1).is_some_and(|t| !t.items.is_empty()), "the title on V2");
    // the 360° shots look through a camera
    let vr: Vec<_> = pictures.iter().filter(|c| s.project.vr_of(c.item).is_some()).collect();
    assert!(vr.iter().all(|c| filmcraft_engine::project::vr::reframe_of(c).is_some()));
}
