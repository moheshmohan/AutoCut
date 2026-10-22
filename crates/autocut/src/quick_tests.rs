use super::*;
use crate::SessionExecutor;
use crate::llm::FnModel;
use filmcraft_engine::Session;
use filmcraft_engine::project::Label;
use filmcraft_engine::project::scenes::{SceneIndex, ShotType, Subject};
use filmcraft_engine::project::speech::{Retake, ShotHint, Span, SpeechAnalysis, SpeechSegment};
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{DemoScene, Generator};
use filmcraft_time::FrameRate;

fn t(s: f64) -> Tick {
    Tick::from_seconds_f64(s)
}

/// A described shot `[a, b]` of a synthetic index.
fn shot(a: f64, b: f64, quality: f32, d: Option<ShotDescription>, camera: CameraMove) -> Shot {
    Shot {
        start: t(a),
        end: t(b),
        keyframes: vec![t((a + b) / 2.0)],
        quality: ShotQuality { score: quality, motion: 0.2, shake: 0.05, ..Default::default() },
        camera,
        description: d,
        ..Shot::default()
    }
}

fn desc(caption: &str, ty: ShotType, energy: Energy, uses: &[ShotUse], subject: Option<(&str, [f32; 4])>, faces: bool) -> ShotDescription {
    ShotDescription {
        shot_type: ty,
        caption: caption.into(),
        energy,
        uses: uses.to_vec(),
        subjects: subject.map(|(n, b)| vec![Subject { name: n.into(), bbox: Some(b) }]).unwrap_or_default(),
        faces_prominent: faces,
        people: u32::from(faces),
        ..Default::default()
    }
}

/// `media.scenes`' answer for a synthetic index (through the engine's own shot JSON).
fn scenes_json(item: u64, idx: &SceneIndex) -> Value {
    json!({
        "item": item, "analyzed": true, "detail": idx.detail,
        "shots": idx.shots.iter().enumerate().map(|(i, s)| filmcraft_engine::scenes::shot_json(i, s, false)).collect::<Vec<_>>(),
    })
}

fn media(item: u64, name: &str, seconds: f64, still: bool, order: usize) -> MediaItem {
    MediaItem { item, name: name.into(), still, seconds, order }
}

/// A library: a striking close-up hook, a wide establishing shot, a reveal, six b-roll shots.
fn library() -> (Vec<MediaItem>, Vec<ShotInfo>) {
    let mut media_list = Vec::new();
    let mut shots = Vec::new();
    let hook =
        desc("A humanoid robot waves at the camera", ShotType::CloseUp, Energy::High, &[ShotUse::Hook], Some(("humanoid robot", [0.6, 0.1, 0.3, 0.7])), true);
    let wide = desc("A wide view of an exhibition hall", ShotType::Wide, Energy::Low, &[ShotUse::Establishing, ShotUse::BRoll], None, false);
    let reveal =
        desc("A robot dog walks across the stage", ShotType::Wide, Energy::Medium, &[ShotUse::Reveal], Some(("robot dog", [0.2, 0.3, 0.4, 0.5])), false);
    let items: Vec<(u64, &str, Shot)> = vec![
        (1, "hook.mov", shot(0.0, 6.0, 0.9, Some(hook), CameraMove::Static)),
        (2, "hall.mov", shot(0.0, 8.0, 0.8, Some(wide), CameraMove::PanLeft)),
        (3, "dog.mov", shot(0.0, 9.0, 0.85, Some(reveal), CameraMove::PanRight)),
    ];
    for (k, (item, name, s)) in items.into_iter().enumerate() {
        let m = media(item, name, 10.0, false, k);
        let idx = SceneIndex { detail: "fast".into(), shots: vec![s], ..Default::default() };
        shots.extend(parse_shots(&m, Some(&scenes_json(item, &idx))));
        media_list.push(m);
    }
    for k in 0..6u64 {
        let d = desc(&format!("Visitors look at robot number {k}"), ShotType::Medium, Energy::High, &[ShotUse::BRoll], None, false);
        let m = media(10 + k, &format!("broll{k}.mov"), 4.0, false, 3 + k as usize);
        let idx = SceneIndex { detail: "fast".into(), shots: vec![shot(0.0, 4.0, 0.7, Some(d), CameraMove::Static)], ..Default::default() };
        shots.extend(parse_shots(&m, Some(&scenes_json(10 + k, &idx))));
        media_list.push(m);
    }
    (media_list, shots)
}

/// The templates as `template.list` lists them.
fn templates() -> Vec<TemplateInfo> {
    let mut s = Session::default();
    let v = s.execute("template.list", json!({})).unwrap();
    v["templates"].as_array().unwrap().iter().filter_map(parse_template).collect()
}

fn inspected(id: &str) -> TemplateInfo {
    let mut s = Session::default();
    parse_template(&s.execute("template.inspect", json!({"template": id})).unwrap()).unwrap()
}

/// A talking head: kept speech with a retake dropped.
fn talking_speech(item: u64) -> ItemSpeech {
    let seg =
        |a: f64, b: f64, text: &str, fluency: f32| SpeechSegment { start: t(a), end: t(b), text: text.into(), fluency, confidence: 0.9, ..Default::default() };
    let a = SpeechAnalysis {
        language: "en".into(),
        source: "whisper-large-v3-turbo".into(),
        vad: "silero".into(),
        duration: t(40.0),
        speech: vec![Span::new(t(0.5), t(38.0))],
        segments: vec![
            seg(0.5, 4.0, "Here is the one tip nobody tells you.", 0.9),
            seg(4.5, 8.0, "Charge it, uh, charge it before you go.", 0.3),
            seg(8.5, 12.0, "Charge it before you leave the house.", 0.95),
            seg(12.5, 19.0, "Then the robot finds its way back on its own.", 0.85),
            seg(19.5, 27.0, "It maps the room the first time it drives around.", 0.8),
            seg(27.5, 37.0, "And that is why it never gets lost again.", 0.9),
        ],
        retakes: vec![Retake { takes: vec![1, 2], keep: 2, similarity: 0.8, reason: "repeat".into() }],
        shot: ShotHint::TalkingHead,
        face: Some(true),
        ..Default::default()
    };
    let v = json!({"items": [filmcraft_engine::speech_understand::analysis_json(filmcraft_engine::project::ItemId(item), &Default::default(), &a)]});
    parse_speech(&v).remove(0)
}

#[test]
fn shots_read_from_the_scene_index_and_without_one() {
    let (_, shots) = library();
    assert_eq!(shots.len(), 9);
    let hook = &shots[0];
    assert_eq!((hook.item, hook.index, hook.start, hook.end, hook.keyframe), (1, 0, 0.0, 6.0, 3.0));
    assert_eq!(hook.description().map(|d| d.energy), Some(Energy::High));
    assert_eq!(focus_of(hook), Some([0.75, 0.45]), "the subject's middle");
    // no index: the whole clip is one shot; a still has no length
    let m = media(7, "x.mov", 12.5, false, 0);
    let one = parse_shots(&m, None);
    assert_eq!((one.len(), one[0].end, one[0].keyframe), (1, 12.5, 6.25));
    assert_eq!(parse_shots(&media(8, "p.jpg", 0.0, true, 0), None)[0].end, 0.0);
    // hostile spans are left out, never a panic
    let bad = json!({"shots": [{"start": -1.0, "end": 2.0}, {"start": 3.0, "end": 1.0}, {"start": "x"}, {"start": 1.0, "end": 1e300}, {"start": 0.0, "end": 2.0, "keyframes": [99.0], "quality": "junk", "description": 4}]});
    let got = parse_shots(&m, Some(&bad));
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].keyframe, 1.0, "a keyframe outside the shot is its middle");
}

#[test]
fn speech_keeps_the_kept_takes() {
    let sp = talking_speech(4);
    assert_eq!(sp.kept.len(), 5, "the dropped take is left out");
    assert!(sp.kept.iter().all(|k| !k.text.contains("uh")));
    assert!(sp.talking() && sp.kept_seconds() > 30.0);
    let mut shots = parse_shots(&media(4, "talk.mov", 40.0, false, 0), None);
    mark_speech(&mut shots, &[sp]);
    assert!(shots[0].talking && shots[0].speech > 0.7);
    assert!(parse_speech(&json!({"items": [{"item": "x"}, {"item": 3, "segments": [{"start": 5.0, "end": 1.0}]}]}))[0].kept.is_empty());
}

#[test]
fn auto_choice_follows_the_media() {
    let all = templates();
    assert!(all.iter().filter(|t| t.ai_ready).count() >= 6);
    // lively described shots about robots: a tech starter
    let (m, shots) = library();
    let f = facts(&m, &shots, &[], Some("robots at the expo"));
    let c = choose_template(&all, &f, "9:16", 20.0).unwrap();
    assert!(["builtin:tech-reveal", "builtin:gadget-showcase", "builtin:event-recap-short"].contains(&c.chosen.id.as_str()), "{c:?}");
    assert!(c.chosen.reasons.iter().any(|r| r.contains("fill itself")), "{:?}", c.chosen.reasons);
    assert!(!c.others.is_empty());
    // someone talking to the camera: the talking-head tip
    let mut talk_shots = parse_shots(&media(4, "talk.mov", 40.0, false, 0), None);
    let sp = vec![talking_speech(4)];
    mark_speech(&mut talk_shots, &sp);
    talk_shots.extend(shots.iter().take(3).cloned());
    let f = facts(&m, &talk_shots, &sp, None);
    assert!(f.talking);
    let c = choose_template(&all, &f, "9:16", 25.0).unwrap();
    assert_eq!(c.chosen.id, "builtin:talking-head-tip", "{c:?}");
    // two clips: the before and after
    let two: Vec<ShotInfo> = shots.iter().skip(1).take(2).cloned().collect();
    let f = facts(&m[1..3], &two, &[], None);
    let c = choose_template(&all, &f, "9:16", 12.0).unwrap();
    assert_eq!(c.chosen.id, "builtin:before-after-short", "{c:?}");
    // nobody talks: a template that wants speech is ranked down
    let tip = all.iter().find(|t| t.id == "builtin:talking-head-tip").unwrap();
    let f = facts(&m, &shots, &[], None);
    assert!(score_template(tip, &f, 30.0).score < score_template(all.iter().find(|t| t.id == "builtin:tech-reveal").unwrap(), &f, 30.0).score);
    // a layout nobody has is an error, not a panic
    assert!(choose_template(&all, &f, "3:1", 30.0).is_err());
    assert!(choose_template(&[], &f, "9:16", 30.0).is_err());
}

#[test]
fn slots_get_the_shots_that_suit_them() {
    let (_, shots) = library();
    let t = inspected("builtin:tech-reveal");
    let specs = pick_specs(&t);
    assert_eq!(specs.len(), 10, "hook, eight b-roll, reveal");
    let picks = assign(&specs, &shots, &[], &HashMap::new(), Some("robots"));
    assert_eq!(picks.len(), 10);
    let of = |slot: &str| picks.iter().filter(|p| p.slot == slot).collect::<Vec<_>>();
    assert_eq!(of("hook")[0].item, 1, "the striking close-up opens: {:?}", of("hook"));
    assert!([2, 3].contains(&of("reveal")[0].item), "a wide moving shot reveals: {:?}", of("reveal"));
    // no moment twice: picks of one clip never overlap
    for (i, a) in picks.iter().enumerate() {
        for b in picks.iter().skip(i + 1) {
            assert!(a.item != b.item || b.out_seconds.min(a.out_seconds) - b.in_seconds.max(a.in_seconds) <= OVERLAP + 1e-9, "{a:?} {b:?}");
        }
    }
    // trimmed to the place around the best moment, inside the shot
    for p in &picks {
        let need = specs.iter().find(|s| s.slot == p.slot && s.media == p.media).unwrap().need;
        assert!((p.out_seconds - p.in_seconds - need).abs() < 0.01 || p.in_seconds == 0.0, "{p:?} {need}");
        assert!(!p.why.is_empty(), "{p:?}");
    }
    assert_eq!(fill_value(of("hook")[0])["focus"], json!([0.75, 0.45]));
    // more places than shots: long shots give more moments, nothing is reused
    let few: Vec<ShotInfo> = shots.iter().take(3).cloned().collect();
    let picks = assign(&specs, &few, &[], &HashMap::new(), None);
    assert!(picks.len() >= 6, "{} picks from three long shots", picks.len());
}

#[test]
fn chronological_slots_keep_the_filming_order() {
    let (_, shots) = library();
    let t = inspected("builtin:day-in-30s");
    let specs = pick_specs(&t);
    assert!(specs.iter().all(|s| s.hints.chronological));
    let picks = assign(&specs, &shots, &[], &HashMap::new(), None);
    let order = |item: u64| shots.iter().find(|s| s.item == item).unwrap().order;
    let seq: Vec<(usize, f64)> = picks.iter().map(|p| (order(p.item), p.in_seconds)).collect();
    assert!(seq.windows(2).all(|w| w[0] <= w[1]), "{seq:?}");
    // before and after: the earlier clip is the before
    let t = inspected("builtin:before-after-short");
    let two: Vec<ShotInfo> = shots.iter().skip(1).take(2).cloned().collect();
    let picks = assign(&pick_specs(&t), &two, &[], &HashMap::new(), None);
    assert_eq!(picks.iter().map(|p| (p.slot.as_str(), p.item)).collect::<Vec<_>>(), vec![("before", 2), ("after", 3)]);
}

#[test]
fn talking_slots_take_kept_speech() {
    let (_, broll) = library();
    let mut shots = parse_shots(&media(4, "talk.mov", 40.0, false, 0), None);
    let sp = vec![talking_speech(4)];
    mark_speech(&mut shots, &sp);
    shots.extend(broll.into_iter().skip(1));
    let t = inspected("builtin:talking-head-tip");
    let picks = assign(&pick_specs(&t), &shots, &sp, &HashMap::new(), None);
    let takes: Vec<&Pick> = picks.iter().filter(|p| p.slot == "takes").collect();
    assert_eq!(takes.len(), 3);
    assert!(takes.iter().all(|p| p.item == 4 && p.said.is_some()), "{takes:?}");
    assert!(takes.windows(2).all(|w| w[0].in_seconds < w[1].in_seconds), "in the order they were said");
    // the dropped take is never the start of one
    assert!(takes.iter().all(|p| !(4.0..8.0).contains(&(p.in_seconds + 0.12))), "{takes:?}");
    let cut: Vec<&Pick> = picks.iter().filter(|p| p.slot == "cutaways").collect();
    assert!(cut.iter().all(|p| p.item != 4), "cutaways are not the speaker: {cut:?}");
}

#[test]
fn text_is_tidy_and_falls_back() {
    assert_eq!(tidy_text("  \"Meet the robots!\"  ", 40, false), "Meet the robots!");
    assert_eq!(tidy_text("a very long line about robots and the future of everything", 20, false), "a very long line");
    assert_eq!(tidy_text("meet\nthem.", 20, true), "MEET THEM");
    assert_eq!(tidy_text(&"x".repeat(10_000), 5, false).chars().count(), 5);
    assert_eq!(tidy_text("ab\u{0}cd", 10, false), "ab cd");
    let (_, shots) = library();
    let t = inspected("builtin:tech-reveal");
    let picks = assign(&pick_specs(&t), &shots, &[], &HashMap::new(), None);
    let specs = text_specs(&t, &picks);
    let title = specs.iter().find(|s| s.slot == "title").unwrap();
    assert!(title.capitals && title.about.is_some());
    let text = fallback_text(title, &picks, &shots, None).unwrap();
    assert!(text.chars().count() <= title.max_chars && text == text.to_uppercase(), "{text}");
    let cta = specs.iter().find(|s| s.slot == "outro").unwrap();
    assert_eq!(fallback_text(cta, &picks, &shots, None).as_deref(), Some("Follow for more"));
    let hook = specs.iter().find(|s| s.slot == "hook_line").unwrap();
    assert!(fallback_text(hook, &picks, &shots, Some("robots")).is_some());
}

#[test]
fn the_model_writes_and_bad_answers_fall_back() {
    let (_, shots) = library();
    let t = inspected("builtin:tech-reveal");
    let picks = assign(&pick_specs(&t), &shots, &[], &HashMap::new(), None);
    let specs = text_specs(&t, &picks);
    let model = FnModel::new("writer", |system, user, _| {
        assert!(system.contains("on-screen text") && user.contains("humanoid robot"), "{user}");
        Ok(json!({"texts": [
            {"slot": "hook_line", "n": 1, "text": "\"meet ameca\""},
            {"slot": "title", "n": 1, "text": "THE ROBOT DOG THAT WALKS ON ITS OWN ALL DAY"},
            {"slot": "nope", "n": 1, "text": "x"},
            {"slot": "outro", "n": 0, "text": "x"},
            {"slot": "outro", "n": 1, "text": 42},
        ]}))
    });
    let mut notes = Vec::new();
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let got = rt.block_on(write_texts(Some(&model), &specs, &picks, &shots, &[], Some("robots"), None, &mut notes));
    let of = |slot: &str| got.iter().find(|(s, _, _)| s.slot == slot).cloned().unwrap();
    assert_eq!((of("hook_line").1.as_deref(), of("hook_line").2), (Some("MEET AMECA"), "model"));
    assert!(of("title").1.unwrap().chars().count() <= 16);
    assert_eq!(of("outro").2, "fallback");
    // a model that fails: every text falls back, with a note
    let broken = FnModel::new("broken", |_, _, _| Err(AutoCutError::Llm("down".into())));
    let got = rt.block_on(write_texts(Some(&broken), &specs, &picks, &shots, &[], None, None, &mut notes));
    assert!(got.iter().all(|(_, _, by)| *by == "fallback"));
    assert!(notes.iter().any(|n| n.contains("did not answer")));
}

#[test]
fn the_tempo_moves_the_length_within_reason() {
    let (bpm, secs) = stretch(120.0, 17.0, 20.0);
    assert!((secs - 20.0).abs() < 0.2 && (bpm - 102.0).abs() < 0.1, "{bpm} {secs}");
    let (bpm, secs) = stretch(120.0, 17.0, 60.0);
    assert_eq!(bpm, BPM_RANGE.0);
    assert!(secs < 30.0);
    assert_eq!(stretch(f64::NAN, 17.0, 20.0).1, 17.0);
    assert_eq!(stretch(120.0, 17.0, 0.0), (120.0, 17.0));
}

#[test]
fn options_default_and_refuse() {
    let o = Options::from_params(&json!({})).unwrap();
    assert_eq!((o.aspect.as_str(), o.platform, o.seconds, o.template.is_none()), ("9:16", Platform::Reels, DEFAULT_SECONDS, true));
    assert!(o.redact.as_ref().is_some_and(|r| r.faces && r.plates), "faces and plates by default");
    let o = Options::from_params(&json!({"template": "auto", "redact": false, "music": "generate", "seconds": 1e9})).unwrap();
    assert!(o.template.is_none() && o.redact.is_none() && o.music.is_none());
    assert_eq!(o.seconds, 180.0);
    assert!(Options::from_params(&json!({"aspect": "7:3"})).is_err());
    assert!(Options::from_params(&json!({"look": "sparkly"})).is_err());
}

#[test]
fn folders_give_their_media_files() {
    let dir = std::env::temp_dir().join(format!("quick-folder-{}-{}", std::process::id(), filmcraft_engine::autosave::unix_now()));
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    for n in ["b.mov", "a.JPG", "notes.txt", ".hidden.mp4"] {
        std::fs::write(dir.join(n), b"x").unwrap();
    }
    let got = expand_paths(&[dir.to_string_lossy().to_string(), "/elsewhere/c.mp4".into()]).unwrap();
    let names: Vec<String> = got.iter().map(|p| Path::new(p).file_name().unwrap().to_string_lossy().to_string()).collect();
    assert_eq!(names, ["a.JPG", "b.mov", "c.mp4"]);
    std::fs::create_dir_all(dir.join("empty")).unwrap();
    assert!(expand_paths(&[dir.join("empty").to_string_lossy().to_string()]).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

/// The whole job on generated media with no model installed (no scene models, no Whisper, no
/// detectors, no Ollama in this build): every step falls back and says so, and the short is
/// built.
#[test]
fn a_quick_short_without_any_model() {
    let dir = std::env::temp_dir().join(format!("quick-job-{}-{}", std::process::id(), filmcraft_engine::autosave::unix_now()));
    std::fs::create_dir_all(&dir).unwrap();
    for k in 0..3u8 {
        image::RgbImage::from_fn(320, 240, |x, y| image::Rgb([k * 60, (x % 256) as u8, (y % 256) as u8])).save(dir.join(format!("photo{k}.png"))).unwrap();
    }
    // three small generated clips (the procedural demo scenes) and the photos
    let mut s = Session::default();
    let mut movies = Vec::new();
    for (k, scene) in [DemoScene::Forest, DemoScene::CityNight, DemoScene::Dunes].into_iter().enumerate() {
        let src = GeneratorSource::new(Generator::Demo(scene), 192, 108, FrameRate::FPS_30, Tick::from_seconds_f64(3.0 + k as f64));
        let pool = s.media.clone();
        movies.push(filmcraft_engine::demo::add_generator(std::sync::Arc::make_mut(&mut s.project), &pool, src, &format!("clip{k}.mov"), Label::Iris, None).0);
    }
    let ex = SessionExecutor::from(s);
    let p = json!({"items": movies, "paths": [dir.to_string_lossy()], "seconds": 20, "topic": "a sunny day", "platform": "reels"});
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let mut said = Vec::new();
    let out = rt.block_on(make(&ex, &p, None, &mut |l| said.push(l.to_string()))).unwrap();
    assert!(out["sequence"].as_u64().is_some(), "{out}");
    assert!(out["template"]["chosen"]["id"].as_str().is_some_and(|id| id.starts_with("builtin:")), "{out}");
    assert!(!out["picks"].as_array().unwrap().is_empty());
    assert_eq!(out["media"], 3 + movies.len() as u64);
    let fallbacks: Vec<&str> = out["fallbacks"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    assert!(fallbacks.iter().any(|f| f.starts_with("text:")), "{fallbacks:?}");
    assert!(out["texts"].as_array().unwrap().iter().all(|t| t["by"] == "fallback"));
    assert!(out["status"].as_str().unwrap().starts_with("Quick Short"));
    assert!(said.iter().any(|l| l.starts_with("[4/8]")), "{said:?}");
    // the sequence is there, vertical
    let g = ex.0.lock().unwrap();
    let q = g.project.sequence(filmcraft_engine::project::ItemId(out["sequence"].as_u64().unwrap())).unwrap();
    assert_eq!((q.settings.width, q.settings.height), (1080, 1920));
    drop(g);
    std::fs::remove_dir_all(&dir).ok();
}
