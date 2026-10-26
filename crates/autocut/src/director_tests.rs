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
        handheld: false,
        talking: false,
        said: None,
        looks: if kind == Kind::Vr { vec![look(&format!("{id}a"), 0.0, "a person", true), look(&format!("{id}b"), 120.0, "a dog", true)] } else { Vec::new() },
        subject: None,
        aspect: 16.0 / 9.0,
        order,
        shot: 0,
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
