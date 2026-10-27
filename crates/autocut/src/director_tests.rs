//! The director's rules, the model's plan checked, the commands, and a whole edit of the demo
//! project (one clip taken as 360°) through a session.

use super::*;

fn opts(p: Value) -> Options {
    Options::from_params(&p).unwrap()
}

fn look(id: &str, yaw: f64, what: &str, named: bool) -> Look {
    Look { id: id.into(), yaw, pitch: 0.0, what: what.into(), named, size: named.then_some((20.0, 50.0)) }
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
        talking: false,
        said: None,
        looks: if kind == Kind::Vr { vec![look(&format!("{id}a"), 0.0, "a person", true), look(&format!("{id}b"), 120.0, "a dog", true)] } else { Vec::new() },
        subject: None,
        aspect: 16.0 / 9.0,
        order,
        shot: 0,
        pan: CameraMove::Static,
        captured: None,
        scores: rank::Scores { sharp: 0.8, exposure: 0.8, steady: 0.8, energy: 0.6, brightness: 0.45, ..rank::Scores::default() },
        rank: 0.6,
        embedding: Vec::new(),
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
    assert_eq!((title_case("weekend in motion").as_str(), title_case("iPhone Days").as_str()), ("Weekend In Motion", "iPhone Days"));
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
fn the_rhythm_adds_up_and_opens_and_closes() {
    let beat = beat_of(120.0);
    for (total, available) in [(40u32, 6usize), (60, 16), (12, 3), (8, 1), (5, 10)] {
        let r = rhythm(total, beat, available);
        assert_eq!(r.iter().map(|s| s.beats).sum::<u32>(), total, "{total} {available}: {r:?}");
        assert_eq!(r[0].role, Role::Hook);
        assert!(r.len() <= available.max(2), "{r:?}");
        if r.len() > 1 {
            assert_eq!(r.last().unwrap().role, Role::Ending);
        }
    }
}

#[test]
fn the_models_picks_fill_the_rhythm_and_are_repaired() {
    let o = opts(json!({"brief": "a 20 s reel", "seconds": 20}));
    let beat = beat_of(120.0);
    let lib = library();
    let slots = rhythm(40, beat, lib.len());
    let answer = json!({
        "title": "Weekend Away",
        "mood": "happy",
        "shots": [
            {"slot": 1, "clip": "c1", "camera": "planetIn", "look": "c1a"},
            {"slot": 2, "clip": "c9", "camera": "hold"},
            {"slot": 3, "clip": "c2", "camera": "orbit"},
            {"slot": 4, "clip": "c3", "camera": "whipPan", "look": "c3a", "to": "c3a"},
            {"slot": 5, "clip": "c3", "camera": "barrelRoll"},
            {"slot": 6, "clip": "c6", "camera": "planetIn", "text": "ഹായ്"}
        ]
    });
    let plan = validate(&answer, &lib, &o, beat, 120.0, &slots).unwrap();
    assert_eq!(plan.title, "Weekend Away");
    assert_eq!(plan.shots.len(), slots.len(), "{:?}", plan.notes);
    assert_eq!(plan.beats(), 40);
    assert_eq!((plan.shots[0].camera, plan.shots[0].text.as_deref()), (Camera::PlanetIn, Some("Weekend Away")));
    for (k, s) in plan.shots.iter().enumerate() {
        let c = lib.iter().find(|c| c.id == s.clip).unwrap();
        assert!(s.camera.for_kind(c.kind), "{s:?}");
        assert!(s.camera != Camera::PlanetIn || k == 0, "{s:?}");
        assert!(s.camera != Camera::PlanetOut || k + 1 == plan.shots.len(), "{s:?}");
        assert!(f64::from(s.beats) * beat + 1e-9 >= s.camera.min_seconds(), "{s:?}");
        if s.camera == Camera::WhipPan {
            assert!(s.look.is_some() && s.look != s.to, "{s:?}");
        }
        assert!(s.text.as_deref().is_none_or(english), "{s:?}");
    }
    let ids: BTreeSet<&str> = plan.shots.iter().map(|s| s.clip.as_str()).collect();
    assert_eq!(ids.len(), plan.shots.len(), "each moment once");
    assert!(plan.notes.iter().any(|n| n.contains("c9")), "{:?}", plan.notes);
    // the plan wrapped as the example was, or under another name
    for wrapped in [json!({"answer": {"shots": [{"clip": "c1", "camera": "orbit"}]}}), json!({"shot_list": [{"clip": "c1"}]})] {
        assert_eq!(validate(&wrapped, &lib, &o, beat, 120.0, &slots).unwrap().shots[0].clip, "c1", "{wrapped}");
    }
    // three moments of one file in a row: the third is another clip
    let mut same = lib.clone();
    for c in same.iter_mut().take(3) {
        c.item = 1;
    }
    let p = validate(&json!({"shots": [{"clip": "c1"}, {"clip": "c2"}, {"clip": "c3"}, {"clip": "c4"}]}), &same, &o, beat, 120.0, &slots).unwrap();
    let items: Vec<u64> = p.shots.iter().map(|s| same.iter().find(|c| c.id == s.clip).unwrap().item).collect();
    assert!(items.windows(3).all(|w| !(w[0] == w[1] && w[1] == w[2])), "{items:?} {:?}", p.notes);
    assert!(p.notes.iter().any(|n| n.contains("third shot")), "{:?}", p.notes);
    // the moments run out (the recap that came out 3 s short): a third moment of one file in a
    // row, then the shot before playing on — the edit keeps its length
    let few = vec![
        cand("c1", 1, Kind::Vr, 0.0, 12.0, 40.0, 0),
        cand("c2", 2, Kind::Flat, 0.0, 6.0, 6.0, 1),
        cand("c3", 2, Kind::Flat, 6.0, 6.0, 20.0, 2),
        cand("c4", 2, Kind::Flat, 12.0, 6.0, 20.0, 3),
    ];
    let six = rhythm(40, beat, 6);
    let p = validate(&json!({"shots": [{"clip": "c1"}, {"clip": "c2"}, {"clip": "c3"}]}), &few, &o, beat, 120.0, &six).unwrap();
    assert_eq!(p.beats(), 40, "{:?} {:?}", p.shots, p.notes);
    assert!(p.notes.iter().any(|n| n.contains("a third of it in a row")), "{:?}", p.notes);
    assert!(p.notes.iter().any(|n| n.contains("plays on")), "{:?}", p.notes);
    // indoors, no reveal from the sky
    let mut inside = lib.clone();
    inside[2].indoor = true;
    let p = validate(&json!({"shots": [{"slot": 2, "clip": "c3", "camera": "reveal"}]}), &inside, &o, beat, 120.0, &slots).unwrap();
    assert!(p.shots.iter().all(|s| s.camera != Camera::Reveal), "{:?}", p.notes);
    // nothing usable
    let e = validate(&json!({"title": "x", "cuts": 3}), &lib, &o, beat, 120.0, &slots).unwrap_err();
    assert!(e.contains("title") && e.contains("cuts"), "{e}");
    assert!(validate(&json!({"shots": []}), &lib, &o, beat, 120.0, &slots).is_err());
    assert!(validate(&json!({"shots": [{"clip": "nope"}]}), &lib, &o, beat, 120.0, &slots).is_err());
    assert!(validate(&json!("garbage"), &lib, &o, beat, 120.0, &slots).is_err());
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
    // the title fits its safe room with a margin
    let (_, size, room) = title_layout(&o, (1080.0, 1920.0), "Weekend Moments Together");
    assert!(size * 0.62 * 24.0 <= 0.9 * room + 1e-6, "{size} in {room}");
    // every shot's own step comes after its placement
    let mut placed = BTreeSet::new();
    for s in &steps {
        if s.on == On::Place {
            placed.insert(s.shot);
        } else if s.shot.is_some() {
            assert!(placed.contains(&s.shot), "{s:?}");
        }
    }
    // every 360° shot framed for the frame's shape: the subject's box (50° tall) in a 9:16 frame
    let (fov, _) = filmcraft_engine::project::vr::subject_framing(Some((20.0, 50.0)), 1080.0 / 1920.0);
    let aims: Vec<&Step> = steps.iter().filter(|s| s.id == "clip.reframe360").collect();
    assert!(!aims.is_empty() && aims.iter().all(|s| s.params["fov"].as_f64() == Some(fov.round())), "{aims:?}");
    // every 360° shot steadied from its pictures, walked or not, but a planet or a hyperlapse
    for (k, s) in plan.shots.iter().enumerate() {
        let c = lib.iter().find(|c| c.id == s.clip).unwrap();
        let steadied = steps.iter().any(|x| x.shot == Some(k) && x.id == "clip.stabilize360");
        let own = matches!(s.camera, Camera::PlanetIn | Camera::PlanetOut | Camera::Hyperlapse);
        assert_eq!(steadied, c.kind == Kind::Vr && !own, "shot {k}: {s:?}");
    }
    // a named subject under a still or pushing camera is tracked
    for (k, s) in plan.shots.iter().enumerate() {
        let c = lib.iter().find(|c| c.id == s.clip).unwrap();
        if c.kind == Kind::Vr && matches!(s.camera, Camera::Hold | Camera::PushIn | Camera::SpeedRamp | Camera::Follow) {
            assert!(steps.iter().any(|x| x.shot == Some(k) && x.id == "clip.follow360"), "shot {k}: {s:?}");
        }
    }
    // the same plan in 16:9: wider views for the same subject
    let landscape = opts(json!({"brief": "a 30 s recap of our trip", "seconds": 30, "aspect": "16:9"}));
    let wide = compile(&plan, &lib, &wins, &landscape, (1920.0, 1080.0), beat);
    let fov_of = |steps: &[Step]| steps.iter().find(|s| s.id == "clip.reframe360").and_then(|s| s.params["fov"].as_f64()).unwrap();
    assert!(fov_of(&wide) > fov_of(&steps), "16:9 {} vs 9:16 {}", fov_of(&wide), fov_of(&steps));
    // a 16:9 clip in the 9:16 frame is tracked as it is cropped
    let flat_k = plan.shots.iter().position(|s| lib.iter().any(|c| c.id == s.clip && c.kind == Kind::Flat)).unwrap();
    assert!(steps.iter().any(|x| x.shot == Some(flat_k) && x.id == "clip.autoReframe"), "{:?}", plan.shots[flat_k]);
    // the steps are engine commands
    let ids: BTreeSet<&str> = steps.iter().map(|s| s.id.as_str()).collect();
    for id in ids {
        assert!(filmcraft_engine::find_command(id).is_some(), "{id} is not an engine command");
    }
}

/// The demo project with its first clip's media taken as 360°.
fn demo_session() -> crate::SessionExecutor {
    let mut s = filmcraft_engine::Session::default();
    // (a data directory of its own: the user's default taste profile is not read)
    let dir = std::env::temp_dir().join(format!("filmcraft-director-tests-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    s.prefs_path = Some(dir.join("preferences.json"));
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
    // (the first answer fails, as a reply cut short would: the model is asked again)
    let model = crate::llm::FnModel::new("fixed", {
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let answer = json!({"title": "Demo Day", "mood": "happy", "shots": shots});
        move |_, _, _| {
            if calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 { Err(AutoCutError::Llm("not JSON".into())) } else { Ok(answer.clone()) }
        }
    });
    let mut lines = |_: &str| {};
    let r = make(&ex, &json!({"brief": "a 10 s reel", "seconds": 10, "bpm": 120, "captions": false}), Some(&model), &mut lines).await.unwrap();
    assert_eq!(r["director"], "model:fixed", "{r}");
    assert_eq!(r["plan"]["title"], "Demo Day");
    // every moment measured on its own frames, and judged with a reason
    let moments = r["moments"].as_array().unwrap();
    assert!(!moments.is_empty() && r["measured"].as_u64().unwrap() as usize == moments.len(), "{r}");
    assert!(
        moments.iter().all(|m| m["why"].as_str().is_some_and(|w| w.starts_with("kept") || w.starts_with("dropped")) && m["scores"]["measured"] == true),
        "{r}"
    );
    assert!(r["plan"]["notes"][0].as_str().unwrap().contains("asked again"), "{r}");
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
    // the colour pass measured every shot; its corrections are Lumetri of their own, marked
    let colour = r["color"]["clips"].as_array().expect("a colour pass");
    assert_eq!(colour.len(), pictures.len(), "{r}");
    let passes = pictures.iter().filter(|c| c.effects.iter().any(filmcraft_engine::auto_color::is_pass)).count();
    let changed = colour
        .iter()
        .filter(|c| {
            c["applied"]["temperature"] != 0.0 || c["applied"]["exposure"] != 0.0 || c["applied"]["saturation"] != 100.0 || c["applied"]["contrast"] != 0.0
        })
        .count();
    assert_eq!(passes, changed, "{r}");
}

#[test]
fn moments_are_ranked_the_unusable_and_copies_left_out_with_reasons() {
    let mut lib = library();
    // c2: out of focus; c4: a copy of c5 (another shot, the same picture) that ranks lower
    lib[1].scores.sharp = 0.05;
    lib[1].scores.motion = 0.0;
    lib[3].embedding = vec![1.0, 0.0, 0.2];
    lib[4].embedding = vec![1.0, 0.01, 0.2];
    lib[3].rank = 0.4;
    lib[4].rank = 0.7;
    for c in lib.iter_mut() {
        c.shot = c.item as usize;
    }
    let (kept, verdicts) = rank::choose(lib.clone(), 3);
    let ids: Vec<&str> = kept.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(ids, ["c1", "c3", "c5", "c6"], "{verdicts:?}");
    let why = |id: &str| verdicts.iter().find(|v| v.clip == id).unwrap().why.clone();
    assert!(why("c2").starts_with("dropped: out of focus"), "{}", why("c2"));
    assert!(why("c4").contains("near-duplicate of c5"), "{}", why("c4"));
    assert!(why("c1").starts_with("kept: rank"), "{}", why("c1"));
    assert_eq!(verdicts.len(), lib.len());
    // too few moments to drop them: kept, and the verdict says so
    let (kept, verdicts) = rank::choose(lib.clone(), 6);
    assert_eq!(kept.len(), 6);
    assert!(verdicts.iter().any(|v| v.why.starts_with("kept though out of focus")), "{verdicts:?}");
    // moments of one shot are not copies of each other
    let mut one = lib.clone();
    for c in one.iter_mut() {
        c.item = 9;
        c.shot = 0;
        c.embedding = vec![1.0, 0.0];
        c.scores.sharp = 0.8;
    }
    assert_eq!(rank::choose(one, 2).0.len(), 6);
}

#[test]
fn the_colour_pass_is_on_unless_switched_off_and_a_look_is_asked_for() {
    let o = opts(json!({}));
    assert!(o.color && o.look == "none" && o.color_strength == 0.7 && o.look_strength == 25.0);
    let o = opts(json!({"color": false, "look": "Punchy", "lookStrength": 400}));
    assert!(!o.color && o.look == "punchy" && o.look_strength == 100.0);
    assert!(Options::from_params(&json!({"look": "sepia"})).is_err());
}

#[test]
fn speech_faces_and_smiles_raise_a_moment() {
    assert_eq!(rank::speech_score(None, 0.0), 0.0);
    assert_eq!(rank::speech_score(None, 0.5), 0.35);
    assert_eq!(rank::speech_score(Some("we are on the train"), 0.5), 0.5);
    assert_eq!(rank::speech_score(Some("Wow, look at that!"), 0.5), 1.0);
    let plain = rank::Scores { sharp: 0.8, exposure: 0.8, steady: 0.8, energy: 0.5, ..rank::Scores::default() };
    let lively = rank::Scores { people: 0.9, expression: 1.0, speech: 1.0, ..plain.clone() };
    assert!(lively.rank(Kind::Flat) > plain.rank(Kind::Flat) + 0.2);
    assert!(plain.rank(Kind::Vr) > plain.rank(Kind::Flat));
    // a moment measured on its own frames: its picture scores and faces count
    let mut c = library().remove(1);
    rank::apply_measure(
        &mut c,
        &json!({"measured": true, "sharpness": 0.1, "exposure": 0.9, "shake": 0.5, "dark": 0.0, "blown": 0.0, "brightness": 0.5, "motion": 0.0, "faces": 2, "largestFace": 0.25}),
    );
    assert!(c.scores.measured && c.scores.sharp == 0.1 && c.scores.steady == 0.5 && c.scores.people >= 0.9, "{:?}", c.scores);
    assert_eq!(c.rank, c.scores.rank(c.kind));
    assert!(c.scores.flaw(Kind::Flat).is_some_and(|f| f.contains("out of focus")));
    // a measurement that failed changes nothing
    let mut d = library().remove(1);
    let before = d.scores.clone();
    rank::apply_measure(&mut d, &json!({"measured": false}));
    assert_eq!(d.scores, before);
    // hostile numbers stay in range
    rank::apply_measure(&mut d, &json!({"measured": true, "sharpness": 1e9, "exposure": -5, "shake": "x", "faces": 1, "largestFace": f64::MAX}));
    assert!([d.scores.sharp, d.scores.exposure, d.scores.steady, d.scores.people, d.rank].iter().all(|v| (0.0..=1.0).contains(v)), "{:?}", d.scores);
}

/// A shot of the plan: candidate `id` from `from` for `beats`, with `camera`.
fn planned(id: &str, from: f64, beats: u32, camera: Camera) -> PlannedShot {
    PlannedShot {
        clip: id.into(),
        file: String::new(),
        grade: None,
        from,
        beats,
        camera,
        look: None,
        to: None,
        enter: Enter::Cut,
        align: None,
        lead: 0.0,
        tail: 0.0,
        cut: String::new(),
        text: None,
        why: String::new(),
    }
}

fn speech_of_item(item: u64, segments: &[(f64, f64)]) -> crate::quick::ItemSpeech {
    crate::quick::ItemSpeech {
        item,
        language: "en".into(),
        kept: segments.iter().map(|(a, b)| crate::quick::Said { start: *a, end: *b, text: "and there it is".into(), fluency: 1.0 }).collect(),
        ..Default::default()
    }
}

/// A trip: a planet hook, an orbit, a talking phone clip, a fast pan, a clip five hours later, a
/// 360° ending; c2 starts talking as its shot begins, c4's sentence runs past its end.
fn trip() -> (Vec<Candidate>, Plan, Vec<crate::quick::ItemSpeech>) {
    let mut lib = library();
    let day = 1_760_000_000i64;
    for (c, t) in lib.iter_mut().zip([0, 60, 120, 180, 5 * 3600 + 200, 5 * 3600 + 300]) {
        c.captured = Some(day + t);
    }
    lib[3].pan = CameraMove::PanRight;
    lib[3].scores.motion = 0.4;
    let plan = Plan {
        title: "Trip".into(),
        mood: "happy".into(),
        bpm: 120.0,
        shots: vec![
            planned("c1", 4.0, 6, Camera::PlanetIn),
            planned("c3", 6.0, 6, Camera::Orbit),
            planned("c2", 1.0, 4, Camera::PushIn),
            planned("c4", 2.0, 4, Camera::PushIn),
            planned("c5", 0.0, 4, Camera::Hold),
            planned("c6", 2.0, 6, Camera::PlanetOut),
        ],
        by: "test".into(),
        notes: Vec::new(),
        style: None,
    };
    let speech = vec![speech_of_item(2, &[(1.2, 3.0)]), speech_of_item(4, &[(3.5, 4.3)])];
    (lib, plan, speech)
}

#[test]
fn each_cut_is_picked_from_what_the_shots_show_and_say() {
    let beat = beat_of(120.0);
    let (lib, mut plan, speech) = trip();
    cuts::joins(&mut plan, &lib, &speech, beat, &cuts::Style::default());
    let enters: Vec<Enter> = plan.shots.iter().map(|s| s.enter).collect();
    assert_eq!(enters, [Enter::Cut, Enter::Whip, Enter::Cut, Enter::Swipe, Enter::Dissolve, Enter::Dissolve], "{:#?}", plan.shots);
    assert!(plan.shots[4].cut.contains("jump in time"), "{}", plan.shots[4].cut);
    assert!(plan.shots[5].cut.contains("ending"));
    // a J cut into c2's voice, an L cut out of c4's sentence
    assert_eq!(plan.shots[2].lead, 0.25);
    assert!(plan.shots[2].cut.contains("J cut"));
    assert!((plan.shots[3].tail - 0.45).abs() < 1e-9, "{}", plan.shots[3].tail);
    assert!(plan.shots.iter().all(|s| !s.cut.is_empty()));
    // a 360° shot after a framed subject lines its subject up with it
    let mut lib2 = lib.clone();
    lib2[1].subject = Some([0.2, 0.3, 0.2, 0.5]);
    let mut p2 = Plan { shots: vec![planned("c2", 1.0, 4, Camera::Hold), planned("c3", 6.0, 6, Camera::PushIn)], ..plan.clone() };
    cuts::joins(&mut p2, &lib2, &[], beat, &cuts::Style::default());
    assert_eq!((p2.shots[1].enter, p2.shots[1].align), (Enter::Match, Some(0.3)), "{:?}", p2.shots[1]);
    // the energetic style punches across time (a light leak when light effects are allowed);
    // the cinematic dips to black across days and whips at most once
    let style = |p: Value| cuts::Style::from_params(&p, &cuts::Style::default()).unwrap();
    let mut p3 = plan.clone();
    cuts::joins(&mut p3, &lib, &speech, beat, &style(json!({"style": "energetic"})));
    assert_eq!(p3.shots[4].enter, Enter::Punch);
    cuts::joins(&mut p3, &lib, &speech, beat, &style(json!({"style": "energetic", "lightEffects": true})));
    assert_eq!(p3.shots[4].enter, Enter::LightLeak);
    let mut lib4 = lib.clone();
    lib4[4].captured = lib4[4].captured.map(|t| t + 2 * 86_400);
    let mut p4 = plan.clone();
    cuts::joins(&mut p4, &lib4, &speech, beat, &style(json!({"style": "cinematic"})));
    assert_eq!(p4.shots[4].enter, Enter::DipToBlack);
    assert!(p4.shots.iter().filter(|s| matches!(s.enter, Enter::Whip | Enter::Swipe)).count() <= 1);
    for bad in [json!({"style": "loud"}), json!({"sounds": ["laser"]}), json!({"soundLevel": 2})] {
        assert!(cuts::Style::from_params(&bad, &cuts::Style::default()).is_err(), "{bad}");
    }
}

#[test]
fn the_cuts_compile_to_transitions_split_edits_and_the_styles_sounds() {
    let beat = beat_of(120.0);
    let (lib, mut plan, speech) = trip();
    cuts::joins(&mut plan, &lib, &speech, beat, &cuts::Style::default());
    plan.shots[0].text = Some("Trip".into());
    let wins = windows(&plan, &Beats::grid(120.0));
    let o = opts(json!({"seconds": 15}));
    let steps = compile(&plan, &lib, &wins, &o, (1080.0, 1920.0), beat);
    let trans: Vec<(&str, Option<usize>)> =
        steps.iter().filter(|s| s.id == "sequence.applyVideoTransition").map(|s| (s.params["effect"].as_str().unwrap(), s.shot)).collect();
    assert_eq!(trans, [("cam_whip_pan", Some(3)), ("cross_dissolve", Some(4)), ("cross_dissolve", Some(5))], "{trans:?}");
    let whip = steps.iter().find(|s| s.params["effect"] == "cam_whip_pan").unwrap();
    assert_eq!(whip.params["params"]["direction"], 1, "panning right: the next picture comes from the east");
    // the J cut on c2 after it is placed; the L cut on c4 once c5 is placed
    let at = |pred: &dyn Fn(&Step) -> bool| steps.iter().position(pred).unwrap();
    let j = at(&|s| s.id == "clip.splitEdit" && s.params["lead"].is_number());
    let l = at(&|s| s.id == "clip.splitEdit" && s.params["tail"].is_number());
    assert_eq!((steps[j].shot, steps[l].shot), (Some(2), Some(3)));
    assert!(j > at(&|s| s.id == "timeline.place" && s.shot == Some(2)));
    assert!(l > at(&|s| s.id == "timeline.place" && s.shot == Some(4)));
    // the subtle style: swishes and the chime, quiet
    let kinds = |st: &[Step]| st.iter().filter(|s| s.id == "audio.generate").map(|s| s.params["kind"].as_str().unwrap().to_string()).collect::<BTreeSet<_>>();
    assert_eq!(kinds(&steps), BTreeSet::from(["chime".to_string(), "swish".to_string()]));
    assert!(steps.iter().filter(|s| s.id == "audio.generate").all(|s| s.params["intensity"].as_f64().unwrap() <= 0.4));
    // an energetic style asking for whooshes and an impact gets them; a style with no sounds none
    let loud = opts(json!({"seconds": 15, "style": "energetic", "soundLevel": 0.7}));
    assert_eq!(kinds(&compile(&plan, &lib, &wins, &loud, (1080.0, 1920.0), beat)), BTreeSet::from(["impact".to_string(), "whoosh".to_string()]));
    let quiet = opts(json!({"seconds": 15, "sounds": []}));
    assert!(kinds(&compile(&plan, &lib, &wins, &quiet, (1080.0, 1920.0), beat)).is_empty());
}

#[test]
fn the_director_chooses_the_style_from_the_shots_unless_one_is_named() {
    let (mut lib, plan, _) = trip();
    let pace = |lib: &[Candidate], brief: &str| {
        let (st, why) = cuts::style_from_shots(&plan, lib, brief);
        (st.pace, st.light_effects, why)
    };
    // the trip: middling energy, one shot of six moving fast, filmed over five hours
    let (p, light, why) = pace(&lib, "");
    assert_eq!((p, light), (cuts::Pace::Subtle, false), "{why}");
    // lively: most shots move fast; light effects by day, none at night
    for c in lib.iter_mut().take(4) {
        c.scores.motion = 0.4;
        c.scores.measured = true;
    }
    let (p, light, why) = pace(&lib, "");
    assert_eq!((p, light), (cuts::Pace::Energetic, true), "{why}");
    assert!(why.contains("4 of 6 shots move fast"), "{why}");
    for c in lib.iter_mut() {
        c.scores.brightness = 0.1;
        c.scores.measured = true;
    }
    assert!(!pace(&lib, "").1, "at night: no light leaks");
    // people talking in many of them: plain cuts
    for c in lib.iter_mut().take(3) {
        c.talking = true;
    }
    assert_eq!(pace(&lib, "").0, cuts::Pace::Subtle);
    // calm shots filmed over days: cinematic; a brief that names a style comes first
    let (mut calm, plan2, _) = trip();
    for (k, c) in calm.iter_mut().enumerate() {
        c.energy = 0.3;
        c.scores.motion = 0.01;
        c.captured = Some(1_760_000_000 + k as i64 * 86_400);
    }
    let (st, why) = cuts::style_from_shots(&plan2, &calm, "");
    assert_eq!(st.pace, cuts::Pace::Cinematic, "{why}");
    assert_eq!(st.sounds, ["riser", "chime"]);
    assert_eq!(cuts::style_from_shots(&plan2, &calm, "a calm recap").0.pace, cuts::Pace::Subtle);
    // the options: no style (or auto) leaves it to the director, the video's sounds kept over it
    let o = opts(json!({"sounds": ["chime"], "soundLevel": 0.2}));
    assert!(o.auto_style);
    assert_eq!(o.style_given, json!({"sounds": ["chime"], "soundLevel": 0.2}));
    let chosen = cuts::Style::from_params(&o.style_given, &cuts::Style::of_pace(cuts::Pace::Energetic, true)).unwrap();
    assert_eq!((chosen.pace, chosen.sounds.clone(), chosen.sound_level), (cuts::Pace::Energetic, vec!["chime".to_string()], 0.2));
    assert!(opts(json!({"style": "auto"})).auto_style);
    let named = opts(json!({"style": "cinematic"}));
    assert!(!named.auto_style && named.style.pace == cuts::Pace::Cinematic);
    // a hostile options object never panics
    assert!(Options::from_params(&json!({"style": "loud"})).is_err());
    assert!(Options::from_params(&json!([1, 2])).is_ok_and(|o| o.auto_style));
}

#[test]
fn a_travel_video_shows_the_places_and_its_person_now_and_then() {
    let (mut lib, mut plan, _) = trip();
    // every 360° shot: the camera's holder first, a mosque or the street behind them
    for c in lib.iter_mut().filter(|c| c.kind == Kind::Vr) {
        c.looks = vec![
            Look { id: format!("{}a", c.id), yaw: 10.0, pitch: -10.0, what: "a man in a patterned shirt".into(), named: true, size: Some((60.0, 80.0)) },
            Look { id: format!("{}b", c.id), yaw: 170.0, pitch: 0.0, what: "a mosque".into(), named: true, size: Some((30.0, 30.0)) },
        ];
        c.scores.people = 1.0;
        c.indoor = false;
    }
    lib[0].looks.truncate(1);
    let (about, why) = focus::focus_of(None, "a reel of our trip", &lib);
    assert_eq!(about, focus::Focus::Places, "{why}");
    assert_eq!(focus::focus_of(None, "", &lib).0, focus::Focus::Places, "outdoors, nobody talking");
    assert_eq!(focus::focus_of(Some(focus::Focus::People), "a trip", &lib).0, focus::Focus::People);
    let before: Vec<f64> = lib.iter().map(|c| c.rank).collect();
    focus::for_places(&mut lib);
    // places first, ids kept; c1 had only its holder: the view the other way round
    assert_eq!(lib[2].looks[0].id, "c3b");
    assert_eq!((lib[0].looks[0].id.as_str(), lib[0].looks[0].yaw.round()), ("c1b", -170.0));
    assert!(lib[0].looks[0].what.contains("away from the camera's holder"));
    assert!(lib.iter().zip(&before).all(|(c, b)| c.rank.is_finite() && (c.rank - b).abs() < 1.0));
    // the plan looking at the holder in its three 360° shots (of six): each turned to its place
    // (the holder is never the subject), the orbit drifting; a whip with no second place pushes in
    for s in plan.shots.iter_mut() {
        if lib.iter().any(|c| c.id == s.clip && c.kind == Kind::Vr) {
            s.look = Some(format!("{}a", s.clip));
        }
    }
    plan.shots[5].camera = Camera::WhipPan;
    plan.shots[5].to = Some("c6b".into());
    let changed = focus::limit_people(&mut plan, &lib);
    assert_eq!(changed.len(), 4, "{changed:?}");
    let looks: Vec<Option<&str>> = plan.shots.iter().map(|s| s.look.as_deref()).collect();
    assert_eq!((looks[0], looks[1], looks[5]), (Some("c1b"), Some("c3b"), Some("c6b")));
    assert_eq!(plan.shots[1].camera, Camera::Drift, "an orbit sweeps past the holder");
    assert_eq!((plan.shots[5].camera, plan.shots[5].to.as_deref()), (Camera::PushIn, None));
    assert!(changed[3].starts_with("shot 6: looks at c6b"), "{changed:?}");
    // the holder's voice over a street is not heard over the music; on screen it is
    let mut talker = lib[2].clone();
    talker.talking = true;
    assert!(!focus::voiced(Some(focus::Focus::Places), &talker, &plan.shots[1]));
    let mut on = plan.shots[1].clone();
    on.look = Some("c3a".into());
    assert!(focus::voiced(Some(focus::Focus::Places), &talker, &on) && focus::voiced(None, &talker, &plan.shots[1]));
    // a whip ending on the holder counts too
    let mut p2 = plan.clone();
    p2.shots[1].look = Some("c3b".into());
    p2.shots[1].to = Some("c3a".into());
    p2.shots[1].camera = Camera::WhipPan;
    focus::limit_people(&mut p2, &lib);
    assert_ne!(p2.shots[1].to.as_deref(), Some("c3a"));
    // the model is told, the person looks marked
    let mut o = opts(json!({"brief": "our trip"}));
    o.focus = Some(focus::Focus::Places);
    let refs: Vec<&Candidate> = lib.iter().collect();
    let prompt = director_prompt(&refs, &o, 0.5, &rhythm(20, 0.5, lib.len()));
    assert!(prompt.contains("This is a travel video") && prompt.contains("a man in a patterned shirt (a person)"), "{prompt}");
    assert!(Options::from_params(&json!({"focus": "dogs"})).is_err());
}

#[test]
fn two_reveals_are_enough() {
    let c = cand("c1", 1, Kind::Vr, 0.0, 12.0, 40.0, 0);
    let mut notes = Vec::new();
    let mut used = HashMap::new();
    assert_eq!(fit_camera(Some(Camera::Reveal), &c, 8, 0.5, 3, false, &used, 9.0 / 16.0, &mut notes), Camera::Reveal);
    used.insert(Camera::Reveal, 2);
    assert_ne!(fit_camera(Some(Camera::Reveal), &c, 8, 0.5, 3, false, &used, 9.0 / 16.0, &mut notes), Camera::Reveal);
    assert!(notes.iter().any(|n| n.contains("reveals are enough")), "{notes:?}");
}

#[test]
fn a_travel_video_covers_every_part_of_the_trip_and_stays_outdoors() {
    let (mut lib, mut plan, _) = trip();
    let day = 1_760_000_000i64;
    // a third part, the next day, that the plan left out
    let mut next = cand("c7", 7, Kind::Vr, 0.0, 10.0, 20.0, 6);
    next.captured = Some(day + 30 * 3600);
    next.rank = 0.7;
    lib.push(next);
    let changed = focus::cover_parts(&mut plan, &lib, 0.5);
    assert_eq!(changed.len(), 1, "{changed:?}");
    assert!(changed[0].starts_with("shot 4: c4 → c7"), "{changed:?}");
    assert_eq!((plan.shots[3].clip.as_str(), plan.shots[3].camera), ("c7", Camera::PushIn));
    assert!(plan.shots[3].look.is_some());
    // every part already in: nothing changes
    assert!(focus::cover_parts(&mut plan, &lib, 0.5).is_empty());
    // told in filming order (c2 was filmed before c3); the next day's only moment, c7, is in the
    // body already, so the ending stays
    plan.shots.swap(1, 2);
    let changed = focus::in_time_order(&mut plan, &lib, 0.5);
    let order: Vec<&str> = plan.shots.iter().map(|s| s.clip.as_str()).collect();
    assert_eq!(order, ["c1", "c2", "c3", "c5", "c7", "c6"], "{changed:?}");
    assert_eq!(changed.len(), 1, "{changed:?}");
    // at most two shots of one recording: c2, c4 and c5 are one walk; the third goes to c8
    let (mut walk, mut p3, _) = trip();
    for c in walk.iter_mut().filter(|c| ["c2", "c4", "c5"].contains(&c.id.as_str())) {
        c.item = 9;
        c.captured = Some(day + 100);
    }
    let mut other = cand("c8", 8, Kind::Vr, 0.0, 10.0, 20.0, 7);
    other.captured = Some(day + 150);
    walk.push(other);
    let changed = focus::cover_parts(&mut p3, &walk, 0.5);
    let nine = p3.shots.iter().filter(|s| walk.iter().any(|c| c.id == s.clip && c.item == 9)).count();
    assert_eq!(nine, 2, "{changed:?}");
    assert!(p3.shots.iter().any(|s| s.clip == "c8") && changed.iter().all(|c| c.ends_with("at most two shots of one recording")), "{changed:?}");
    // indoor moments: only as many as the outdoors cannot fill
    for (k, c) in lib.iter_mut().enumerate() {
        c.indoor = k < 3;
        c.rank = 0.5 + k as f64 / 100.0;
    }
    let mut pool = lib.clone();
    assert_eq!(focus::places_pool(&mut pool, 4, 6), 2);
    assert_eq!(pool.iter().filter(|c| c.indoor).map(|c| c.id.as_str()).collect::<Vec<_>>(), ["c3"], "the best indoor one kept");
    let mut short = lib.clone();
    assert_eq!(focus::places_pool(&mut short, 6, 6), 1, "two kept: the outdoors fill only four of six");
}

#[test]
fn notes_are_read_into_changes_by_rules_and_from_a_model() {
    use revise::{Op, Where};
    let (ops, unread) =
        revise::rules_ops("1:10 is too long; warmer at the end and fewer barrel rolls.\nNo whip pans. Title: \"Weekend Away\". Make it sparkle", 100.0);
    assert!(ops.contains(&Op::Shorter(Where::At(70.0), 2)), "{ops:?}");
    assert!(ops.iter().any(|o| matches!(o, Op::Grade(Some(f), None, g) if (*f - 75.0).abs() < 1e-9 && g.warmth > 0.0)), "{ops:?}");
    assert!(ops.contains(&Op::Fewer(Camera::BarrelRoll, None)) && ops.contains(&Op::Fewer(Camera::WhipPan, Some(0))), "{ops:?}");
    assert!(ops.contains(&Op::Title("Weekend Away".into())), "{ops:?}");
    assert_eq!(unread, ["Make it sparkle"]);
    // a model's answer: the usable kept, the rest said
    let (ops, bad) = revise::ops_from(&json!({"ops": [
        {"op": "remove", "shot": 3},
        {"op": "camera", "at": "0:04", "camera": "orbit"},
        {"op": "grade", "from": "0:20", "warmth": 99},
        {"op": "fly"},
        {"op": "camera", "shot": 2, "camera": "teleport"}
    ]}));
    assert_eq!(ops.len(), 3, "{ops:?}");
    assert!(matches!(ops[2], Op::Grade(Some(f), None, g) if f == 20.0 && g.warmth == revise::MAX_WARMTH));
    assert_eq!(bad.len(), 2);
    assert_eq!(revise::time_of(&json!("1:02:03")), Some(3723.0));
    assert_eq!(revise::time_of(&json!("x")), None);
}

#[test]
fn changes_apply_to_the_shots_of_the_last_cut() {
    use revise::{Grade, Op, Where};
    let beat = beat_of(120.0);
    let (lib, plan, _) = trip();
    let wins = windows(&plan, &Beats::grid(120.0));
    let mut o = opts(json!({}));
    let mut p = plan.clone();
    let ops = vec![
        Op::Remove(Where::Shot(2)),
        Op::Shorter(Where::Shot(3), 2),
        Op::Camera(Where::Shot(4), Camera::Orbit),
        Op::Camera(Where::Shot(4), Camera::Hold),
        Op::Grade(Some(wins[4].0), None, Grade { warmth: 8.0, ..Grade::default() }),
        Op::Title("Our Weekend".into()),
        Op::Style("cinematic".into()),
        Op::Look("warm".into()),
        Op::Swap(Where::Shot(1), "c5".into()),
        Op::Remove(Where::At(999.0)),
    ];
    let said = revise::apply(&mut p, &ops, &lib, &wins, beat, &mut o);
    assert_eq!(p.shots.len(), plan.shots.len() - 1, "{said:#?}");
    // (shot 3 of the last cut is c2, now the second)
    assert_eq!((p.shots[1].clip.as_str(), p.shots[1].beats), ("c2", 2));
    assert!(said.iter().any(|s| s.contains("orbit does not fit")), "an orbit on a normal clip: {said:#?}");
    assert_eq!(p.shots[2].camera, Camera::Hold);
    assert!(p.shots[3..].iter().all(|s| s.grade.is_some_and(|g| g.warmth == 8.0)) && p.shots[2].grade.is_none(), "{:?}", p.shots);
    assert_eq!((p.title.as_str(), p.shots[0].text.as_deref()), ("Our Weekend", Some("Our Weekend")));
    assert_eq!((o.style.pace, o.look.as_str()), (cuts::Pace::Cinematic, "warm"));
    // every moment is in the edit already: the swap is refused; a time past the end names no shot
    assert!(p.shots[0].clip == "c1" && said.iter().any(|s| s.contains("already in the edit")), "{said:#?}");
    assert!(said.iter().any(|s| s.contains("no shot there")), "{said:#?}");
    // fewer: half of them, or none
    let mut q = plan.clone();
    for s in q.shots.iter_mut() {
        s.camera = Camera::PushIn;
    }
    revise::apply(&mut q, &[Op::Fewer(Camera::PushIn, Some(0))], &lib, &wins, beat, &mut o);
    assert!(q.shots.iter().all(|s| s.camera != Camera::PushIn), "{:?}", q.shots);
}

#[test]
fn a_saved_edit_is_found_again_and_its_options_kept() {
    let (lib, mut plan, _) = trip();
    for s in plan.shots.iter_mut() {
        s.file = lib.iter().find(|c| c.id == s.clip).unwrap().name.clone();
    }
    let report = json!({"plan": plan, "options": {"brief": "a reel", "seconds": 15, "export": "/tmp/reel.mp4", "style": "energetic"}});
    let pr = revise::load(&report).unwrap();
    let merged = revise::merge(&pr.options, &json!({"revise": "warmer", "seconds": 20, "previous": "x"}), 120.0);
    assert_eq!((merged["seconds"].as_f64(), merged["style"].as_str(), merged["bpm"].as_f64()), (Some(20.0), Some("energetic"), Some(120.0)));
    assert_eq!(merged["export"], "/tmp/reel-revised.mp4");
    assert!(revise::saved_options(&merged).get("revise").is_none());
    // the moments' ids changed with the footage: the shots are found by file and time
    let mut renumbered = lib.clone();
    for (k, c) in renumbered.iter_mut().enumerate() {
        c.id = format!("m{}", 10 + k);
    }
    let (again, lost) = revise::remap(&pr.plan, &renumbered);
    assert!(lost.is_empty(), "{lost:?}");
    assert_eq!(again.shots[0].clip, "m10");
    let (_, lost) = revise::remap(&pr.plan, &renumbered[1..]);
    assert_eq!(lost.len(), 1);
    assert!(revise::load(&json!({"plan": {"title": "x", "bpm": 120, "shots": []}})).is_err());
    assert!(revise::load(&json!("/no/such/report.json")).is_err());
}

#[tokio::test]
async fn the_taste_profile_fills_what_a_video_leaves_out() {
    let ex = demo_session();
    {
        let mut s = ex.0.lock().unwrap();
        s.execute(
            "autocut.taste",
            json!({"set": {"style": "cinematic", "redact": "off", "color": false, "words": [{"say": "Our Days", "not": "Highlights"}]}}),
        )
        .unwrap();
    }
    let mut lines = |_: &str| {};
    let r = make(&ex, &json!({"brief": "a 10 s reel", "seconds": 10, "bpm": 120, "captions": false, "dryRun": true}), None, &mut lines).await.unwrap();
    assert_eq!(r["taste"]["style"], "cinematic", "{r}");
    // the profile's wording in the title; its style under the video's own options
    assert_eq!(r["plan"]["title"], "Our Days", "{r}");
    assert!(r["options"].get("style").is_none(), "the report keeps the video's own options: {r}");
    let own = make(&ex, &json!({"seconds": 10, "bpm": 120, "captions": false, "dryRun": true, "style": "energetic"}), None, &mut lines).await.unwrap();
    assert_eq!(own["options"]["style"], "energetic");
}

#[tokio::test]
async fn an_edit_is_revised_from_notes_not_made_again() {
    let ex = demo_session();
    let mut lines = |_: &str| {};
    let first =
        make(&ex, &json!({"brief": "a 10 s reel", "seconds": 10, "bpm": 120, "captions": false, "style": "energetic"}), None, &mut lines).await.unwrap();
    let before = first["plan"]["shots"].as_array().unwrap().len();
    assert!(first["options"]["style"] == "energetic" && first["plan"]["shots"][0]["file"].is_string(), "{first}");
    let r = make(&ex, &json!({"revise": "warmer at the end; no push ins", "previous": first, "captions": false}), None, &mut lines).await.unwrap();
    assert!(r["director"].as_str().unwrap().starts_with("revised"), "{r}");
    let changes: Vec<&str> = r["changes"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    assert!(changes.iter().any(|c| c.contains("warmer")), "{changes:?}");
    assert_eq!(r["plan"]["shots"].as_array().unwrap().len(), before, "the same shots: {r}");
    assert!(r["plan"]["shots"].as_array().unwrap().iter().all(|s| s["camera"] != "pushIn"), "{r}");
    // the options came with it, and the graded shots were warmed over the colour pass
    assert_eq!(r["options"]["style"], "energetic");
    let warm = r["color"]["clips"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["did"].as_array().is_some_and(|d| d.iter().any(|x| x.as_str().is_some_and(|x| x.starts_with("asked for")))))
        .count();
    assert!(warm >= 1, "{r}");
}

#[tokio::test]
async fn notes_the_rules_read_are_taken_literally_and_the_model_reads_the_rest() {
    let ex = demo_session();
    let mut lines = |_: &str| {};
    let first = make(&ex, &json!({"brief": "a 10 s reel", "seconds": 10, "bpm": 120, "captions": false, "color": false}), None, &mut lines).await.unwrap();
    // a model that would turn "warmer" into warmer, brighter and more vivid
    let model = crate::llm::FixedModel::new(vec![json!({"ops": [{"op": "title", "text": "Sea And Sky"}]})]);
    let r = make(&ex, &json!({"revise": "warmer at the end. call it something about the sea", "previous": first, "captions": false}), Some(&model), &mut lines)
        .await
        .unwrap();
    let changes: Vec<&str> = r["changes"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    assert!(changes.iter().any(|c| c.ends_with(": warmer")), "only warmer, as written: {changes:?}");
    assert!(changes.iter().any(|c| c.contains("Sea And Sky")), "{changes:?}");
    // the model was asked only about what the rules could not read
    let asked = model.asked.lock().unwrap();
    assert_eq!(asked.len(), 1);
    assert!(asked[0].1.contains("call it something about the sea") && !asked[0].1.contains("Notes: \"warmer"), "{}", asked[0].1);
    assert!(r["director"].as_str().unwrap().contains("rules and fixed"), "{r}");
}
