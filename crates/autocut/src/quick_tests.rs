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

/// What a recogniser writes for a crowd or music (the expo clips of the first live run: "Thank
/// you.", "mga mga mga mga mga", a lone "The") is not someone talking to the camera.
#[test]
fn hallucinated_speech_is_not_talking() {
    let seg = |a: f64, b: f64, text: &str| SpeechSegment { start: t(a), end: t(b), text: text.into(), fluency: 0.6, confidence: 0.6, ..Default::default() };
    let a = SpeechAnalysis {
        duration: t(24.0),
        speech: vec![Span::new(t(0.3), t(20.0))],
        segments: vec![seg(0.3, 6.0, "Thank you."), seg(6.5, 12.0, "mga mga mga mga mga"), seg(12.5, 20.0, "The")],
        shot: ShotHint::TalkingHead,
        ..Default::default()
    };
    let v = json!({"items": [filmcraft_engine::speech_understand::analysis_json(filmcraft_engine::project::ItemId(9), &Default::default(), &a)]});
    let sp = parse_speech(&v).remove(0);
    assert!(sp.kept.is_empty() && !sp.talking(), "{sp:?}");
    assert!(believable("Charge it before you leave the house.", 0.9));
    assert!(!believable("Charge it before you leave the house.", 0.1), "not confident");
    assert!(!believable("पार पार पार पार", 0.9));
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
    // nine shots for ten places: the long ones' second moments would come within 3 s of their first
    assert_eq!(picks.len(), 9);
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
    // more places than shots: a long moving shot gives several moments, 3 s apart; nothing is
    // reused (shots of 6–9 s give one each: a second would come within 3 s of the first)
    let mut long = shots[2].clone();
    (long.end, long.keyframe) = (30.0, 15.0);
    long.shot.end = Tick::from_seconds_f64(30.0);
    let picks = assign(&specs, &[long], &[], &HashMap::new(), None);
    assert!(picks.len() >= 4, "{} picks from one 30 s pan", picks.len());
    let few: Vec<ShotInfo> = shots.iter().take(3).cloned().collect();
    assert_eq!(assign(&specs, &few, &[], &HashMap::new(), None).len(), 3);
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
    // a subject's name too long for its place keeps its noun
    let mut long = library().1;
    if let Some(d) = long.iter_mut().find(|s| s.item == 1).and_then(|s| s.shot.description.as_mut()) {
        d.subjects[0].name = "bright orange humanoid robot".into();
    }
    let picks = assign(&pick_specs(&t), &long, &[], &HashMap::new(), None);
    let hook = text_specs(&t, &picks).into_iter().find(|s| s.slot == "hook_line").unwrap();
    assert_eq!(fallback_options(&hook, &picks, &long, None).first().map(String::as_str), Some("HUMANOID ROBOT"));
}

/// Each text knows what plays under it: the picture placements its own placement overlaps in the
/// template.
#[test]
fn texts_know_what_is_shown_under_them() {
    let (_, shots) = library();
    let t = inspected("builtin:tech-reveal");
    let picks = assign(&pick_specs(&t), &shots, &[], &HashMap::new(), None);
    let specs = text_specs(&t, &picks);
    let under = |slot: &str| specs.iter().find(|s| s.slot == slot).unwrap().under.iter().map(|i| picks[*i].slot.as_str()).collect::<Vec<_>>();
    assert_eq!(under("hook_line"), ["hook"]);
    assert_eq!(under("title"), ["reveal"]);
    assert_eq!(under("outro"), ["reveal"], "the end card is the reveal, blurred");
    // the event recap's closing card is over its opening shot, again
    let t = inspected("builtin:event-recap-short");
    let picks = assign(&pick_specs(&t), &shots, &[], &HashMap::new(), None);
    let specs = text_specs(&t, &picks);
    let closing = specs.iter().find(|s| s.slot == "closing").unwrap();
    assert_eq!(closing.under.iter().map(|i| picks[*i].slot.as_str()).collect::<Vec<_>>(), ["establishing"]);
    // a template without timing: the pick it is about
    let mut t2 = inspected("builtin:tech-reveal");
    for s in &mut t2.slots {
        for p in &mut s.placements {
            p.start = None;
        }
    }
    let picks = assign(&pick_specs(&t2), &shots, &[], &HashMap::new(), None);
    let title = text_specs(&t2, &picks).into_iter().find(|s| s.slot == "title").unwrap();
    assert_eq!(title.under, title.about.into_iter().collect::<Vec<_>>());
}

#[test]
fn words_must_come_from_what_is_shown_the_topic_or_the_speech() {
    assert!(content_words("DON'T MISS IT — FOLLOW FOR MORE").is_empty(), "{:?}", content_words("DON'T MISS IT — FOLLOW FOR MORE"));
    assert_eq!(content_words("Boy meets robot"), ["boy", "robot"]);
    let g = Grounds { told: vec!["robot".into(), "expo".into()], shown: vec!["serving".into(), "tray".into(), "bottle".into(), "delivery".into()] };
    let w = check_words("DRONE DELIVERY", &g);
    assert_eq!((w.unknown.as_slice(), w.shown.as_slice()), (&["drone".to_string()][..], &["delivery".to_string()][..]));
    assert_eq!(check_words("Robotic expo", &g), WordCheck::default(), "robotic is the topic's robot");
    assert_eq!(check_words("TOP 5 ROBOTS", &g).unknown, ["5"], "no invented numbers");
    assert!(check_words("", &g).unknown.is_empty());
    // the shots under a text, the topic and the speech
    let (_, shots) = library();
    let t = inspected("builtin:tech-reveal");
    let picks = assign(&pick_specs(&t), &shots, &[], &HashMap::new(), None);
    let specs = text_specs(&t, &picks);
    let title = specs.iter().find(|s| s.slot == "title").unwrap();
    let g = Grounds::of(title, &picks, &shots, &[], Some("robots at the expo"));
    assert!(g.told.contains(&"expo".to_string()) && g.shown.iter().any(|w| w == "dog"), "{g:?}");
    assert!(!g.shown.iter().any(|w| w == "humanoid"), "the hook's subject is not under the title: {g:?}");
}

/// A fake `media.checkText`: each window's item shows `truth[item]`; the describer says no when a
/// text names anything else; SigLIP's cosine is higher the more of a text a window shows.
struct Checker {
    truth: HashMap<u64, Vec<&'static str>>,
    asker: bool,
    calls: std::sync::Mutex<Vec<Value>>,
}

impl Checker {
    fn new(truth: &[(u64, &[&'static str])], asker: bool) -> Self {
        Checker { truth: truth.iter().map(|(i, w)| (*i, w.to_vec())).collect(), asker, calls: Default::default() }
    }
    fn sees(&self, item: u64, text: &str) -> (usize, usize) {
        let words = content_words(text);
        let shown = self.truth.get(&item).cloned().unwrap_or_default();
        (words.iter().filter(|w| shown.contains(&w.as_str())).count(), words.len())
    }
}

impl crate::Executor for Checker {
    fn execute(&self, id: &str, p: Value) -> crate::ExecFuture {
        assert_eq!(id, "media.checkText");
        self.calls.lock().unwrap().push(p.clone());
        let items: Vec<u64> = p["windows"].as_array().unwrap().iter().map(|w| w["item"].as_u64().unwrap()).collect();
        let texts: Vec<String> = p["texts"].as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect();
        let cosine: Vec<Vec<f64>> =
            texts.iter().map(|t| items.iter().map(|i| self.sees(*i, t)).map(|(hit, n)| 0.05 + 0.1 * hit as f64 / n.max(1) as f64).collect()).collect();
        let answers: Vec<Value> = if self.asker {
            p["asks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| {
                    let (w, t) = (a["window"].as_u64().unwrap() as usize, a["text"].as_u64().unwrap() as usize);
                    // one thing of the text at a time
                    let about = a["about"].as_str().unwrap();
                    assert!(texts[t].to_lowercase().contains(about), "{about} is a word of {}", texts[t]);
                    let (hit, n) = self.sees(items[w], about);
                    json!({"window": w, "text": t, "about": about, "answer": if hit == n { "yes" } else { "no" }})
                })
                .collect()
        } else {
            Vec::new()
        };
        // like the engine: a describer is named only when it was asked something
        let asked = p["asks"].as_array().is_some_and(|a| !a.is_empty());
        let v = json!({"job": 1, "scenes": {"cosine": cosine, "answers": answers, "embedder": "siglip-fake", "asker": if self.asker && asked { json!("vlm-fake") } else { Value::Null }}});
        Box::pin(async move { Ok(v) })
    }
}

/// The library with a boy in the hook shot's description (he walks out before the moment used)
/// and a serving robot as the reveal.
fn expo_library() -> Vec<ShotInfo> {
    let (_, mut shots) = library();
    for s in &mut shots {
        let Some(d) = s.shot.description.as_mut() else { continue };
        match s.item {
            1 => d.caption = "A humanoid robot waves as a young boy approaches it".into(),
            3 => {
                d.caption = "A white serving robot carries bottles on its trays across the stage".into();
                d.subjects[0].name = "serving robot".into();
            }
            _ => {}
        }
    }
    shots
}

type Asked = std::sync::Arc<std::sync::Mutex<Vec<String>>>;

/// A writer that answers `answers` in turn (the last one again), keeping what it was asked.
fn expo_writer(answers: Vec<Value>) -> (FnModel, Asked) {
    let asked: Asked = Default::default();
    let seen = asked.clone();
    let answers = std::sync::Mutex::new(answers);
    let m = FnModel::new("writer", move |system, user, _| {
        assert!(system.contains("only what those shots show"), "{system}");
        seen.lock().unwrap().push(user.to_string());
        let mut a = answers.lock().unwrap();
        if a.len() > 1 { Ok(a.remove(0)) } else { Ok(a[0].clone()) }
    });
    (m, asked)
}

#[test]
fn the_writer_sees_only_the_shots_under_each_text() {
    let shots = expo_library();
    let t = inspected("builtin:tech-reveal");
    let picks = assign(&pick_specs(&t), &shots, &[], &HashMap::new(), None);
    let specs = text_specs(&t, &picks);
    let (model, asked) = expo_writer(vec![json!({"texts": [{"slot": "title", "n": 1, "options": ["SERVING ROBOT"]}]})]);
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let mut notes = Vec::new();
    rt.block_on(write_texts(None, None, Some(&model), &specs, &picks, &shots, &[], Some("robots"), None, &mut notes));
    let user = asked.lock().unwrap()[0].clone();
    let block = |slot: &str| {
        let at = user.find(&format!("slot \"{slot}\"")).unwrap();
        let rest = &user[at..];
        rest[..rest[1..].find("slot \"").map_or(rest.len(), |k| k + 1)].to_string()
    };
    assert!(block("hook_line").contains("humanoid robot") && !block("hook_line").contains("serving robot"), "{}", block("hook_line"));
    assert!(block("title").contains("serving robot") && !block("title").contains("humanoid"), "{}", block("title"));
    assert!(user.contains("options"), "three options are asked for");
}

/// "DRONE DELIVERY" over a serving robot: a word from nowhere. "BOY MEETS ROBOT" over the hook
/// once the boy has left: the describer does not see him. Both are rejected, with why, and the
/// next option is used.
#[test]
fn texts_are_checked_against_their_frames() {
    let shots = expo_library();
    let t = inspected("builtin:tech-reveal");
    let picks = assign(&pick_specs(&t), &shots, &[], &HashMap::new(), None);
    let specs = text_specs(&t, &picks);
    let checker = Checker::new(&[(1, &["humanoid", "robot", "wave"]), (3, &["serving", "robot", "bottle", "tray", "stage", "white"])], true);
    let (model, _) = expo_writer(vec![json!({"texts": [
        {"slot": "hook_line", "n": 1, "options": ["BOY MEETS ROBOT", "HUMANOID SAYS HI", "ROBOTS!"]},
        {"slot": "title", "n": 1, "options": ["DRONE DELIVERY", "SERVING ROBOT"]},
        {"slot": "outro", "n": 1, "options": ["FOLLOW FOR PART 2"]},
        {"slot": "outro_sub", "n": 1, "text": "More robots every week"},
    ]})]);
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let mut notes = Vec::new();
    let (got, summary) =
        rt.block_on(write_texts(Some(&checker), Some(9.0 / 16.0), Some(&model), &specs, &picks, &shots, &[], Some("robots at the expo"), None, &mut notes));
    let of = |slot: &str| got.iter().find(|w| w.spec.slot == slot).unwrap();
    let hook = of("hook_line");
    assert_eq!((hook.text.as_deref(), hook.by.as_str()), (Some("HUMANOID SAYS HI"), "model"), "{:?}", hook.tried);
    assert_eq!(hook.tried[0].answer.as_deref(), Some("no"));
    assert!(hook.tried[0].why[0].contains("boy"), "{:?}", hook.tried[0]);
    assert_eq!(hook.tried[1].answer.as_deref(), Some("yes"));
    let title = of("title");
    assert_eq!(title.text.as_deref(), Some("SERVING ROBOT"));
    assert!(!title.tried[0].ok && title.tried[0].words.unknown.contains(&"drone".to_string()), "{:?}", title.tried[0]);
    assert_eq!(title.tried[0].answer, None, "a word from nowhere is not even asked about");
    // "2" is said nowhere: the call to action falls back to its own line
    assert_eq!(of("outro").text.as_deref(), Some("Follow for more"), "{:?}", of("outro").tried);
    assert_eq!(of("outro_sub").text.as_deref(), Some("More robots every week"));
    assert_eq!(summary["how"], "words, the describer's answers and SigLIP", "the describer answered in the first round: {summary}");
    assert_eq!(summary["asker"], "vlm-fake");
    assert!(summary["asked"].as_u64().unwrap() >= 2, "{summary}");
    assert!(summary["rejected"].as_u64().unwrap() >= 3, "{summary}");
    // the report's entry: what was under it and every candidate
    let j = hook.to_json(&picks);
    assert_eq!(j["under"][0]["slot"], "hook");
    assert_eq!(j["candidates"].as_array().unwrap().len(), 2);
    // the describer was asked only about what the descriptions alone name, over the text's own shot
    let calls = checker.calls.lock().unwrap();
    let asks = calls[0]["asks"].as_array().unwrap();
    let texts: Vec<&str> = asks.iter().map(|a| calls[0]["texts"][a["text"].as_u64().unwrap() as usize].as_str().unwrap()).collect();
    assert!(texts.contains(&"BOY MEETS ROBOT") && !texts.contains(&"DRONE DELIVERY") && !texts.contains(&"ROBOTS!"), "{texts:?}");
}

/// Texts that name only the topic's words need no question: the report says so rather than that
/// the describer is missing.
#[test]
fn nothing_to_ask_is_not_a_missing_describer() {
    let shots = expo_library();
    let t = inspected("builtin:tech-reveal");
    let picks = assign(&pick_specs(&t), &shots, &[], &HashMap::new(), None);
    let specs = text_specs(&t, &picks);
    let checker = Checker::new(&[(1, &["robot"])], true);
    let (model, _) = expo_writer(vec![json!({"texts": [
        {"slot": "hook_line", "n": 1, "options": ["ROBOTS!"]},
        {"slot": "title", "n": 1, "options": ["THE ROBOT"]},
        {"slot": "outro", "n": 1, "options": ["Follow for more"]},
        {"slot": "outro_sub", "n": 1, "options": ["More robots soon"]},
    ]})]);
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let mut notes = Vec::new();
    let (_, summary) = rt.block_on(write_texts(Some(&checker), Some(9.0 / 16.0), Some(&model), &specs, &picks, &shots, &[], Some("robots"), None, &mut notes));
    assert_eq!((summary["how"].as_str(), summary["asked"].as_u64()), (Some("words and SigLIP (nothing to ask the describer)"), Some(0)), "{summary}");
}

/// The same words twice on screen read as a mistake: the second text takes its next option.
#[test]
fn no_two_texts_say_the_same() {
    let shots = expo_library();
    let t = inspected("builtin:tech-reveal");
    let picks = assign(&pick_specs(&t), &shots, &[], &HashMap::new(), None);
    let specs = text_specs(&t, &picks);
    let (model, _) = expo_writer(vec![json!({"texts": [
        {"slot": "outro", "n": 1, "options": ["Follow for more"]},
        {"slot": "outro_sub", "n": 1, "options": ["FOLLOW FOR MORE", "More robots every week"]},
    ]})]);
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let mut notes = Vec::new();
    let (got, _) = rt.block_on(write_texts(None, None, Some(&model), &specs, &picks, &shots, &[], Some("robots"), None, &mut notes));
    let sub = got.iter().find(|w| w.spec.slot == "outro_sub").unwrap();
    assert_eq!(sub.text.as_deref(), Some("More robots every week"), "{:?}", sub.tried);
    assert!(sub.tried[0].why.iter().any(|w| w.contains("the same as another text")), "{:?}", sub.tried[0]);
    let mut all: Vec<String> = got.iter().filter_map(|w| w.text.clone()).map(|x| x.to_lowercase()).collect();
    let n = all.len();
    all.sort();
    all.dedup();
    assert_eq!(all.len(), n, "{all:?}");
}

/// Without the describer, SigLIP decides: a text that matches another shot's frames better than
/// its own is rejected.
#[test]
fn siglip_alone_rejects_a_text_about_another_shot() {
    let shots = expo_library();
    let t = inspected("builtin:tech-reveal");
    let picks = assign(&pick_specs(&t), &shots, &[], &HashMap::new(), None);
    let specs = text_specs(&t, &picks);
    // the reveal's description says "white", but only the hall shows it
    let checker = Checker::new(&[(1, &["humanoid", "robot"]), (3, &["serving", "robot"]), (2, &["white", "stage", "hall"])], false);
    let (model, _) = expo_writer(vec![json!({"texts": [{"slot": "title", "n": 1, "options": ["WHITE STAGE", "SERVING ROBOT"]}]})]);
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let mut notes = Vec::new();
    let (got, summary) =
        rt.block_on(write_texts(Some(&checker), Some(9.0 / 16.0), Some(&model), &specs, &picks, &shots, &[], Some("robots"), None, &mut notes));
    let title = got.iter().find(|w| w.spec.slot == "title").unwrap();
    assert_eq!(title.text.as_deref(), Some("SERVING ROBOT"), "{:?}", title.tried);
    assert!(title.tried[0].why[0].contains("another shot"), "{:?}", title.tried[0]);
    assert!(title.tried[0].own.unwrap() < title.tried[0].elsewhere.unwrap());
    assert_eq!(summary["how"], "words and SigLIP");
}

/// Options that all fail: the writer is asked once more with what was rejected and why; when
/// that fails too, the deterministic text from the descriptions.
#[test]
fn rejected_texts_are_asked_again_then_written_by_rule() {
    let shots = expo_library();
    let t = inspected("builtin:tech-reveal");
    let picks = assign(&pick_specs(&t), &shots, &[], &HashMap::new(), None);
    let specs = text_specs(&t, &picks);
    let checker = Checker::new(&[(1, &["humanoid", "robot"]), (3, &["serving", "robot", "bottle"])], true);
    let (model, asked) = expo_writer(vec![
        json!({"texts": [{"slot": "title", "n": 1, "options": ["DRONE DELIVERY", "CAR WASH"]}, {"slot": "hook_line", "n": 1, "options": ["BOY MEETS ROBOT"]}]}),
        json!({"texts": [{"slot": "title", "n": 1, "options": ["BOTTLE ROBOT"]}, {"slot": "hook_line", "n": 1, "options": ["A BOY AND A ROBOT"]}]}),
    ]);
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let mut notes = Vec::new();
    let (got, _) = rt.block_on(write_texts(Some(&checker), Some(9.0 / 16.0), Some(&model), &specs, &picks, &shots, &[], Some("robots"), None, &mut notes));
    let asked = asked.lock().unwrap();
    assert_eq!(asked.len(), 2, "asked once more");
    assert!(asked[1].contains("Rejected, do not repeat") && asked[1].contains("DRONE DELIVERY") && asked[1].contains("drone"), "{}", asked[1]);
    assert!(!asked[1].contains("slot \"outro\""), "only the texts that failed are asked again");
    let of = |slot: &str| got.iter().find(|w| w.spec.slot == slot).unwrap();
    assert_eq!((of("title").text.as_deref(), of("title").by.as_str()), (Some("BOTTLE ROBOT"), "model, asked again"));
    // the hook failed twice: the subject's name, which the describer sees
    assert_eq!((of("hook_line").text.as_deref(), of("hook_line").by.as_str()), (Some("HUMANOID ROBOT"), "fallback"), "{:?}", of("hook_line").tried);
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
            {"slot": "HOOK_LINE", "n": "1", "text": "\"meet the humanoid\""},
            {"slot": "title", "n": 1, "text": "THE ROBOT DOG THAT WALKS ON ITS OWN ALL DAY"},
            {"slot": "nope", "n": 1, "text": "x"},
            {"slot": "outro", "n": 0, "text": "x"},
            {"slot": "outro", "n": 1, "text": 42},
            {"slot": "outro_sub", "n": 1, "options": "junk"},
        ]}))
    });
    let mut notes = Vec::new();
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let (got, summary) = rt.block_on(write_texts(None, None, Some(&model), &specs, &picks, &shots, &[], Some("robots"), None, &mut notes));
    let of = |slot: &str| got.iter().find(|w| w.spec.slot == slot).unwrap();
    assert_eq!((of("hook_line").text.as_deref(), of("hook_line").by.as_str()), (Some("MEET THE HUMANOID"), "model"));
    assert!(of("title").text.as_ref().unwrap().chars().count() <= 16);
    assert_eq!(of("outro").by, "fallback");
    assert_eq!(summary["how"], "words only", "no checker: the words alone");
    // a model that fails: every text falls back, with a note
    let broken = FnModel::new("broken", |_, _, _| Err(AutoCutError::Llm("down".into())));
    let (got, _) = rt.block_on(write_texts(None, None, Some(&broken), &specs, &picks, &shots, &[], None, None, &mut notes));
    assert!(got.iter().all(|w| w.by == "fallback"));
    assert!(notes.iter().any(|n| n.contains("did not answer")));
    // no model, no checker: the descriptions
    let (got, _) = rt.block_on(write_texts(None, None, None, &specs, &picks, &shots, &[], None, None, &mut notes));
    assert!(got.iter().all(|w| w.by == "fallback" && w.text.is_some()), "{got:?}");
}

/// What a checker answers is read defensively: junk is no answer, never a panic.
#[test]
fn hostile_check_answers_are_read_defensively() {
    for v in [
        json!(null),
        json!({"scenes": 3}),
        json!({"cosine": [[1e300, "x"], 7], "answers": [{"window": -1, "text": 0, "answer": "yes"}, {"window": 0}]}),
        json!([1, 2]),
    ] {
        let l = Looked::parse(&v);
        assert!(l.answers.is_empty(), "{v}");
    }
    let l = Looked::parse(&json!({"cosine": [[0.1]], "answers": [{"window": 0, "text": 0, "answer": "no"}]}));
    let g = Grounds { told: vec![], shown: vec!["dog".into()] };
    // rows and windows out of range are no scores
    let v = judge("DOG", "model", &g, Some(&l), &Spot { row: Some(99), under: &[5, 7], others: &[usize::MAX], alternative: Some(1000) });
    assert!(v.ok && v.own.is_none(), "{v:?}");
    let v = judge("DOG", "model", &g, Some(&l), &Spot { row: Some(0), under: &[0], others: &[], alternative: None });
    assert!(!v.ok && v.answer.as_deref() == Some("no"));
    assert!(!judge("", "model", &g, None, &Spot::default()).ok);
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

/// Three generated clips (no sound of their own) in a session, and their items.
fn clips_session(n: usize) -> (Session, Vec<u64>) {
    // (a data directory of its own, one per session: the user's taste and taste history are not
    // read, and one test's history is not another's)
    static SESSIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let k = SESSIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("filmcraft-quick-tests-{}-{k}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::create_dir_all(&dir);
    let mut s = Session::default();
    s.prefs_path = Some(dir.join("preferences.json"));
    let mut movies = Vec::new();
    let scenes = [DemoScene::Forest, DemoScene::CityNight, DemoScene::Dunes, DemoScene::Plasma];
    for k in 0..n {
        let scene = scenes[k % scenes.len()];
        let src = GeneratorSource::new(Generator::Demo(scene), 192, 108, FrameRate::FPS_30, Tick::from_seconds_f64(6.0 + k as f64));
        let pool = s.media.clone();
        movies.push(filmcraft_engine::demo::add_generator(std::sync::Arc::make_mut(&mut s.project), &pool, src, &format!("clip{k}.mov"), Label::Iris, None).0);
    }
    (s, movies)
}

/// The music bed carries a quick short nobody speaks in: with the stems levelled and filtered
/// (`quick_mix`) and the mix at −14 LUFS, the Mix's limiter has (almost) nothing to do and little
/// of the energy is below 120 Hz — the QA's two mix warnings of the first real run — while the
/// bed keeps its low end (not thin) and the whooshes and impacts sit under it.
#[test]
fn the_bed_and_the_effects_are_mixed_for_small_speakers() {
    let (s, movies) = clips_session(4);
    let ex = SessionExecutor::from(s);
    let p = json!({"items": movies, "template": "builtin:tech-reveal", "seconds": 28, "topic": "robots", "redact": "off"});
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let out = rt.block_on(make(&ex, &p, None, &mut |_| {})).unwrap();
    let seq = out["sequence"].as_u64().unwrap();
    let mix = &out["mix"];
    assert_eq!(mix["profile"], "music", "{mix}");
    assert!(!mix["stems"]["music"].as_array().unwrap().is_empty() && !mix["stems"]["effects"].as_array().unwrap().is_empty(), "{mix}");
    let a = rt.block_on(ex.execute("audio.analyzeMix", json!({"sequence": seq}))).unwrap();
    let findings = crate::qa::mix_findings(&a, &crate::qa::Options::default());
    let sub = a["mix"]["bands"]["sub"].as_f64().unwrap();
    let limiter = a["limiter"]["maxGainReductionDb"].as_f64().unwrap_or(0.0);
    let lufs = a["mix"]["integratedLufs"].as_f64().unwrap();
    eprintln!("mix: {lufs:.2} LUFS, sub {sub:.3}, limiter {limiter:.2} dB, bands {}, octaves {}", a["mix"]["bands"], a["mix"]["octavesDb"]);
    for st in a["stems"].as_array().unwrap() {
        eprintln!(
            "  {} {}: {} LUFS, peak {} dBTP, sub {}, limiter {}",
            st["track"], st["role"], st["integratedLufs"], st["truePeakDbtp"], st["bands"]["sub"], st["limiter"]
        );
    }
    assert!(findings.is_empty(), "{findings:?}");
    assert!((lufs + 14.0).abs() < 0.6, "at the social loudness: {lufs}");
    assert!(sub <= crate::qa::MAX_LOW_SHARE && limiter <= crate::qa::MAX_LIMITER_DB, "sub {sub}, limiter {limiter}");
    // not thin: the bed keeps its low end — its 63 and 125 Hz octaves within a few dB of its mids
    let music = a["stems"].as_array().unwrap().iter().find(|s| s["track"] == "A1").unwrap();
    let oct = |hz: &str| music["octavesDb"][hz].as_f64().unwrap();
    assert!(oct("63").max(oct("125")) > oct("1000") - 6.0, "the bed's low end: {}", music["octavesDb"]);
    assert!(music["bands"]["sub"].as_f64().unwrap() + music["bands"]["lowMid"].as_f64().unwrap() > 0.2, "{}", music["bands"]);
    // the effects sit under the music
    let louder = a["stems"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["track"] != "A1" && s["track"] != "transitions")
        .filter_map(|s| s["integratedLufs"].as_f64())
        .fold(f64::MIN, f64::max);
    assert!(louder < music["integratedLufs"].as_f64().unwrap() - 4.0, "effects {louder} vs music {}", music["integratedLufs"]);
}

/// The roles of an applied template's sound: the music slot's clip, the talking slots' own sound,
/// and the rest (the Elements sounds) as effects; hostile answers give no roles, never a panic.
#[test]
fn stems_are_typed_by_their_slots() {
    let t = inspected("builtin:tech-reveal");
    let applied = json!({"slots": [{"slot": "music", "clips": [40]}, {"slot": "hook", "clips": [10]}]});
    let seq = json!({
        "video": [{"items": [{"clip": 10, "link": 7}]}],
        "audio": [{"items": [{"clip": 40}]}, {"items": [{"clip": 41}, {"clip": 42, "link": 7}]}, {"items": [{"clip": 43}]}],
    });
    let st = Stems::of(&t, &applied, &seq, &[]);
    assert_eq!((st.music.as_slice(), st.sound.as_slice(), st.effects.as_slice()), (&[40][..], &[42][..], &[41, 43][..]));
    assert!(st.talk.is_empty());
    for junk in [json!(null), json!({"audio": 3}), json!({"audio": [{"items": [{"clip": "x"}, 5]}], "video": [[]]})] {
        let st = Stems::of(&t, &json!({"slots": "x"}), &junk, &[]);
        assert!(st.music.is_empty() && st.effects.is_empty());
    }
    // the bed's chain keeps its ceiling under the loudness target's peak, whatever level it is given
    for lufs in [-16.0, f64::NAN, -1e9, 0.0] {
        let chain = crate::mix::bed_dynamics(lufs);
        let max = chain.iter().find(|(e, _)| *e == "hard_limiter").unwrap().1["max"].as_f64().unwrap();
        assert!(max.is_finite() && max <= -1.0, "{lufs} → {max}");
    }
}

/// The tempo is never slowed so far that one shot holds longer than its cap: the Tech Reveal's
/// reveal (8 beats) would hold 6.7 s at the 72 bpm that makes it 28 s, so it stops at 96 bpm
/// (5 s) and the video is 21 s; a slot asking for a longer hold (`maxHold`) gets it.
#[test]
fn holds_are_capped_when_the_tempo_is_moved() {
    let t = inspected("builtin:tech-reveal");
    let tp = tempo(&t, 30.0).unwrap();
    let held = tp.held.as_ref().expect("held");
    assert_eq!(held.slot, "reveal");
    assert!(held.would > 6.5 && (held.cap - MAX_HOLD).abs() < 1e-9, "{held:?}");
    assert!((tp.bpm - 96.0).abs() < 0.05 && (tp.seconds - 21.25).abs() < 0.1, "{tp:?}");
    // every shot within its cap at that tempo (the end card's reprise is the template's own)
    for s in t.slots.iter().filter(|s| s.picture()) {
        for p in s.placements.iter().take(1) {
            assert!(p.seconds * t.bpm.unwrap() / tp.bpm <= hold_cap(s, p).unwrap() + 1e-6, "{} {p:?}", s.id);
        }
    }
    // shorter than the template: faster, nothing held
    let fast = tempo(&t, 12.0).unwrap();
    assert!(fast.held.is_none() && fast.bpm > 120.0);
    // the template asks for a longer reveal: it may hold 8 s
    let mut long = t.clone();
    for s in long.slots.iter_mut().filter(|s| s.id == "reveal") {
        s.ai.get_or_insert_with(Hints::default).max_hold = Some(8.0);
    }
    let tp = tempo(&long, 30.0).unwrap();
    assert!(tp.held.is_none() && (tp.bpm - 72.0).abs() < 0.05, "{tp:?}");
    // the choice knows: the Tech Reveal says it stops short of the length asked
    let (m, shots) = library();
    let f = facts(&m, &shots, &[], Some("robots"));
    let s = score_template(&t, &f, 30.0);
    assert!(s.reasons.iter().any(|r| r.contains("no slower")) && (s.seconds - 21.3).abs() < 0.2, "{s:?}");
    // talking slots are not capped; no beat grid, no tempo; hostile numbers are no tempo
    let tip = inspected("builtin:talking-head-tip");
    assert!(tip.slots.iter().filter(|s| s.ai.as_ref().is_some_and(|h| h.speech.as_deref() == Some("want"))).all(|s| hold_cap(s, &s.placements[0]).is_none()));
    assert!(tempo(&tip, 30.0).is_none());
    let mut bad = t.clone();
    bad.bpm = Some(f64::NAN);
    assert!(tempo(&bad, 30.0).is_none());
    bad.bpm = Some(120.0);
    for s in &mut bad.slots {
        for p in &mut s.placements {
            p.seconds = 0.0;
        }
    }
    assert!(tempo(&bad, f64::INFINITY).is_some_and(|x| x.seconds.is_finite()));
}

/// Two picks of one clip are at least [`MIN_GAP`] apart; a measured shot whose camera stands
/// still gives one moment when frames can't be compared; a moment that looked like another is
/// not used again.
#[test]
fn picks_of_one_clip_are_kept_apart() {
    let (_, shots) = library();
    let t = inspected("builtin:tech-reveal");
    let specs = pick_specs(&t);
    let few: Vec<ShotInfo> = shots.iter().take(3).cloned().collect();
    let gap = |a: &Pick, b: &Pick| a.in_seconds.max(b.in_seconds) - a.out_seconds.min(b.out_seconds);
    let picks = assign(&specs, &few, &[], &HashMap::new(), None);
    for (i, a) in picks.iter().enumerate() {
        for b in picks.iter().skip(i + 1).filter(|b| b.item == a.item) {
            assert!(gap(a, b) >= MIN_GAP - 1e-9, "{a:?} {b:?}");
        }
    }
    assert!(picks.len() < specs.len(), "three shots do not give ten distinct moments");
    assert!(facts(&[], &few, &[], None).moments >= picks.len(), "the moments the choice counted");
    // a still camera: one moment each when the frames can't tell
    let mut still = few.clone();
    for s in &mut still {
        s.shot.camera = CameraMove::Static;
        s.shot.quality.motion = 0.0;
    }
    let once = assign_spread(&specs, &still, &[], &HashMap::new(), None, &Spread { still_once: true, ..Spread::default() });
    assert_eq!(once.len(), 3, "{once:?}");
    // a moment to avoid is not used, nor anything near it
    let first = picks[0].clone();
    let spread = Spread { avoid: vec![(first.item, first.in_seconds, first.out_seconds)], ..Spread::default() };
    let again = assign_spread(&specs, &few, &[], &HashMap::new(), None, &spread);
    assert!(again.iter().all(|p| p.item != first.item || gap(p, &first) >= MIN_GAP - 1e-9), "{again:?}");
    // someone talking: takes of one clip may follow each other (their words differ)
    let (_, broll) = library();
    let mut talk = parse_shots(&media(4, "talk.mov", 40.0, false, 0), None);
    let sp = vec![talking_speech(4)];
    mark_speech(&mut talk, &sp);
    talk.extend(broll.into_iter().skip(1));
    let tip = inspected("builtin:talking-head-tip");
    assert_eq!(assign(&pick_specs(&tip), &talk, &sp, &HashMap::new(), None).iter().filter(|p| p.slot == "takes").count(), 3);
}

/// A fake `media.checkText` for the moments' frames: each window looks like `looks[item]`.
struct Frames(HashMap<u64, Vec<f32>>);

impl crate::Executor for Frames {
    fn execute(&self, id: &str, p: Value) -> crate::ExecFuture {
        assert_eq!(id, "media.checkText");
        let items: Vec<u64> = p["windows"].as_array().unwrap().iter().map(|w| w["item"].as_u64().unwrap()).collect();
        let v = |i: &u64| self.0.get(i).cloned().unwrap_or_else(|| vec![0.0, 0.0, 1.0]);
        let cos = |a: &[f32], b: &[f32]| {
            let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let n = |x: &[f32]| x.iter().map(|v| v * v).sum::<f32>().sqrt();
            f64::from(d / (n(a) * n(b)))
        };
        let similar: Vec<Vec<f64>> = items.iter().map(|a| items.iter().map(|b| cos(&v(a), &v(b))).collect()).collect();
        let out = json!({"job": 1, "scenes": {"similar": similar, "embedder": "fake"}});
        Box::pin(async move { Ok(out) })
    }
}

/// Picks that look alike: the cutaway is replaced, not the hook; without an embedding model the
/// frames can't be compared (`None`).
#[test]
fn moments_that_look_alike_are_found() {
    let (_, shots) = library();
    let t = inspected("builtin:tech-reveal");
    let specs = pick_specs(&t);
    let picks = assign(&specs, &shots, &[], &HashMap::new(), None);
    let hook = picks.iter().position(|p| p.slot == "hook").unwrap();
    let broll = picks.iter().position(|p| p.slot == "broll").unwrap();
    // the hook's clip and the first cutaway's look the same; everything else differs
    let mut looks: HashMap<u64, Vec<f32>> = HashMap::new();
    for (k, p) in picks.iter().enumerate() {
        let mut v = vec![0.0; picks.len()];
        v[k] = 1.0;
        looks.insert(p.item, v);
    }
    looks.insert(picks[broll].item, looks[&picks[hook].item].clone());
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let mut notes = Vec::new();
    let pairs = rt.block_on(look_alike(&Frames(looks), &picks, &specs, Some(0.5625), &mut notes)).unwrap();
    assert_eq!(pairs.len(), 1, "{pairs:?}");
    assert_eq!((pairs[0].0, pairs[0].1), (broll, hook), "the cutaway goes, the hook stays");
    // no embedding model: the frames can't be compared
    struct NoSim;
    impl crate::Executor for NoSim {
        fn execute(&self, _: &str, _: Value) -> crate::ExecFuture {
            Box::pin(async { Ok(json!({"scenes": {"similar": null}})) })
        }
    }
    assert!(rt.block_on(look_alike(&NoSim, &picks, &specs, None, &mut notes)).is_none());
}

/// Fewer distinct moments than the b-roll's places: the places that would show a moment again
/// are taken out and the music made again for the new length; no clip is shown twice near the
/// same moment, and no shot holds past its cap.
#[test]
fn too_few_moments_shorten_the_video_instead_of_repeating() {
    let (s, movies) = clips_session(3);
    let ex = SessionExecutor::from(s);
    let p = json!({"items": movies, "template": "builtin:tech-reveal", "seconds": 28, "topic": "robots", "redact": "off"});
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let out = rt.block_on(make(&ex, &p, None, &mut |_| {})).unwrap();
    let spread = &out["shotsSpread"];
    assert!(spread["filled"].as_u64().unwrap() < spread["places"].as_u64().unwrap(), "{spread}");
    let short = &spread["shortened"];
    assert!(short["to"].as_f64().unwrap() < short["from"].as_f64().unwrap(), "{spread}");
    assert_eq!(short["musicAgain"], true, "{short}");
    assert!(spread["longestHold"]["seconds"].as_f64().unwrap() <= MAX_HOLD + 0.05, "{spread}");
    assert!(out["warnings"].as_array().unwrap().iter().any(|w| w.as_str().is_some_and(|w| w.contains("shortened"))), "{}", out["warnings"]);
    // the picture: each moment once (the end card's blurred reprise of the reveal is the template's)
    let seq = rt.block_on(ex.execute("sequence.inspect", json!({"item": out["sequence"]}))).unwrap();
    let v1: Vec<&Value> = seq["video"][0]["items"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| movies.contains(&c["item"].as_u64().unwrap()))
        .filter(|c| !c["effects"].as_array().unwrap().iter().any(|e| e["effect"] == "gaussian_blur"))
        .collect();
    let tps = TICKS_PER_SECOND as f64;
    for (i, a) in v1.iter().enumerate() {
        for b in v1.iter().skip(i + 1).filter(|b| b["item"] == a["item"]) {
            let (a0, a1) = (a["sourceIn"].as_i64().unwrap() as f64 / tps, a["sourceOut"].as_i64().unwrap() as f64 / tps);
            let (b0, b1) = (b["sourceIn"].as_i64().unwrap() as f64 / tps, b["sourceOut"].as_i64().unwrap() as f64 / tps);
            assert!(a0.max(b0) - a1.min(b1) >= MIN_GAP - 0.6, "one moment twice: {a} {b}");
        }
    }
    // the new bed is the music, and it is as long as the video
    let music: Vec<&Value> = seq["audio"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|t| t["items"].as_array().unwrap())
        .filter(|c| out["mix"]["stems"]["music"].as_array().unwrap().contains(&c["clip"]))
        .collect();
    assert_eq!(music.len(), 1, "{}", out["mix"]);
    assert!((music[0]["end"].as_i64().unwrap() - seq["duration"].as_i64().unwrap()).abs() <= TICKS_PER_SECOND / 10);
}

/// Every frame of one clip looks alike; the others differ (a fake `media.checkText` over a real
/// session).
struct AlikeFrames {
    inner: SessionExecutor,
    same: u64,
}

impl crate::Executor for AlikeFrames {
    fn execute(&self, id: &str, p: Value) -> crate::ExecFuture {
        if id != "media.checkText" {
            return self.inner.execute(id, p);
        }
        let items: Vec<u64> = p["windows"].as_array().unwrap().iter().map(|w| w["item"].as_u64().unwrap()).collect();
        let sim: Vec<Vec<f64>> = (0..items.len())
            .map(|a| (0..items.len()).map(|b| if a == b || (items[a] == self.same && items[b] == self.same) { 1.0 } else { 0.3 }).collect())
            .collect();
        let v = json!({"job": 1, "scenes": {"similar": sim, "embedder": "fake"}});
        Box::pin(async move { Ok(v) })
    }
}

/// Two moments of one clip that look alike: the clip gives one moment, the others go elsewhere.
#[test]
fn a_clip_whose_moments_look_alike_gives_one() {
    let mut s = Session::default();
    let mut movies = Vec::new();
    for (k, (scene, secs)) in
        [(DemoScene::Forest, 40.0), (DemoScene::CityNight, 14.0), (DemoScene::Dunes, 14.0), (DemoScene::Plasma, 14.0)].into_iter().enumerate()
    {
        let src = GeneratorSource::new(Generator::Demo(scene), 192, 108, FrameRate::FPS_30, Tick::from_seconds_f64(secs));
        let pool = s.media.clone();
        movies.push(filmcraft_engine::demo::add_generator(std::sync::Arc::make_mut(&mut s.project), &pool, src, &format!("clip{k}.mov"), Label::Iris, None).0);
    }
    let ex = AlikeFrames { inner: SessionExecutor::from(s), same: movies[0] };
    let p = json!({"items": movies, "template": "builtin:tech-reveal", "seconds": 20, "redact": "off"});
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let out = rt.block_on(make(&ex, &p, None, &mut |_| {})).unwrap();
    let of_same = out["picks"].as_array().unwrap().iter().filter(|p| p["item"] == movies[0]).count();
    assert_eq!(of_same, 1, "{}", out["picks"]);
    let alike = out["shotsSpread"]["lookAlike"].as_array().unwrap();
    assert!(!alike.is_empty() && alike.iter().all(|a| a["replaced"]["name"] == "clip0.mov"), "{alike:?}");
}

/// The film events a Quick Short fits the user's track to: a burst on the music's first beat,
/// peaks on the title hits and the landing on the end card (the last title, late in the short).
#[test]
fn the_film_events_mark_the_title_hits_and_the_end_card() {
    let t = TemplateInfo {
        id: "t".into(),
        name: "T".into(),
        description: String::new(),
        category: String::new(),
        tags: Vec::new(),
        seconds: 20.0,
        aspects: Vec::new(),
        ai_ready: true,
        captions: false,
        bpm: None,
        slots: ["hook", "title", "line", "end"]
            .iter()
            .map(|id| SlotInfo {
                id: id.to_string(),
                kind: if *id == "hook" { "video".into() } else { "text".into() },
                name: id.to_string(),
                prompt: String::new(),
                count: 1,
                min_count: 1,
                max_chars: None,
                ai: None,
                placements: Vec::new(),
            })
            .collect(),
    };
    let tk = |s: f64| (s * TICKS_PER_SECOND as f64) as i64;
    let applied = json!({"seconds": 20.0, "slots": [
        {"slot": "hook", "start": 0, "duration": tk(4.0)},
        {"slot": "title", "start": tk(0.2), "duration": tk(3.0)},
        {"slot": "line", "start": tk(6.0), "duration": tk(3.0)},
        {"slot": "end", "start": tk(17.0), "duration": tk(3.0)},
    ]});
    let events = film_events(&t, &applied, 0.25);
    let kinds: Vec<(&str, f64)> = events.iter().map(|e| (e["kind"].as_str().unwrap(), e["seconds"].as_f64().unwrap())).collect();
    assert_eq!(kinds, [("burst", 0.25), ("peak", 6.0), ("landing", 17.0)]);
    // hostile slots: no times, strings, past the end: left out
    let applied = json!({"seconds": 20.0, "slots": [{"slot": "line", "start": "x"}, {"slot": "line", "start": tk(500.0)}, {"slot": "end"}]});
    assert_eq!(film_events(&t, &applied, f64::NAN).len(), 0);
}

/// A Quick Short on the user's track: the template laid it, then it is fitted to the short
/// (`clip.remix`: a burst on its first beat, peaks on the title hits, its own ending on the end
/// card);
/// the report says how (`musicFit`).
#[test]
fn a_quick_short_fits_the_users_track() {
    // (enough clips that no place repeats a moment: the short keeps its length)
    let (s, movies) = clips_session(12);
    let ex = SessionExecutor::from(s);
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let dir = std::env::temp_dir().join(format!("filmcraft-quick-music-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let g = rt
        .block_on(ex.execute(
            "audio.generate",
            json!({"kind": "music", "duration": 45, "bpm": 124, "seed": 2, "soundfont": "oscillators", "place": false, "dir": dir.to_string_lossy()}),
        ))
        .unwrap();
    let path = g["path"].as_str().unwrap().to_string();
    let p = json!({"items": movies, "template": "builtin:tech-reveal", "seconds": 28, "topic": "robots", "redact": "off", "music": path});
    let out = rt.block_on(make(&ex, &p, None, &mut |_| {})).unwrap();
    let fit = &out["musicFit"];
    assert_eq!(fit["ending"], "own", "{out}");
    let landed = fit["eventsLanded"].as_array().cloned().unwrap_or_default();
    assert!(landed.first().is_some_and(|e| e["kind"] == "burst" && e["landed"] == true), "{fit}");
    assert!(fit["seconds"].as_f64().is_some_and(|x| x > 5.0 && x <= out["seconds"].as_f64().unwrap() + 0.6), "{fit} {}", out["seconds"]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Taste memory: a short takes its template's own look while the last videos did not use it;
/// the next one, whose history did, is dealt another creator look over the template. Both are
/// recorded; without guidance nothing is dealt or recorded.
#[test]
fn consecutive_shorts_are_dealt_different_looks() {
    let (s, movies) = clips_session(3);
    let ex = SessionExecutor::from(s);
    let p = json!({"items": movies, "seconds": 12, "topic": "a sunny day", "platform": "reels", "redact": false});
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let first = rt.block_on(make(&ex, &p, None, &mut |_| {})).unwrap();
    let own = first["deals"]["look"]["choice"].as_str().unwrap().to_string();
    assert_eq!(first["deals"]["look"]["from"][0], own.as_str(), "the template's own look first: {:#}", first["deals"]);
    assert_eq!(first["fingerprint"]["recorded"], true, "{:#}", first["fingerprint"]);
    let second = rt.block_on(make(&ex, &p, None, &mut |_| {})).unwrap();
    let next = second["deals"]["look"]["choice"].as_str().unwrap();
    assert_ne!(next, own, "{:#}", second["deals"]);
    assert_eq!(second["deals"]["look"]["skipped"][0]["why"], "used in the last 3");
    if next != "clean" {
        assert_eq!(second["fingerprint"]["axes"]["colorLook"], next, "the dealt look is the one seen: {:#}", second["fingerprint"]);
    }
    let plain = rt.block_on(make(&ex, &json!({"items": movies, "seconds": 12, "topic": "a sunny day", "guidance": false}), None, &mut |_| {})).unwrap();
    assert!(plain["deals"].is_null() && plain["fingerprint"].is_null(), "{plain}");
}

/// PX6.2: a short made like a reference (a fingerprint kept from an earlier measure) takes its
/// length, picks its template by the reference's shots, is measured as built and reports how
/// close it came; a length or template asked for wins.
#[test]
fn a_short_is_made_like_a_reference() {
    let (s, movies) = clips_session(4);
    let ex = SessionExecutor::from(s);
    let fingerprint = json!({
        "version": 1, "seconds": 14.0, "cutsPerMinute": 90.0,
        "shot": {"median": 0.7, "p25": 0.5, "p75": 1.0, "cv": 0.4},
        "bursts": {"perMinute": 0.0, "share": 0.0, "at": []},
        "pace": vec![0.7; 16], "energy": vec![0.5; 16], "loudness": null,
    });
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let p = json!({"items": movies, "topic": "robots", "redact": false, "guidance": false, "reference": {"fingerprint": fingerprint}});
    let out = rt.block_on(make(&ex, &p, None, &mut |_| {})).unwrap();
    let r = &out["reference"];
    assert!(r["applied"].as_array().unwrap().iter().any(|a| a.as_str().unwrap().starts_with("seconds: 14.0")), "{r:#}");
    assert!(out["template"]["chosen"]["reasons"].as_array().unwrap().iter().any(|x| x.as_str().unwrap().contains("the reference")), "{:#}", out["template"]);
    assert_eq!(r["achieved"]["from"], "render", "{:#}", r["achieved"]);
    assert!(r["distance"]["overall"].as_f64().is_some_and(|d| (0.0..=1.0).contains(&d)), "{:#}", r["distance"]);
    // asked for wins
    let asked = json!({"items": movies, "topic": "robots", "redact": false, "guidance": false, "seconds": 20, "template": "builtin:tech-reveal", "reference": {"fingerprint": fingerprint}});
    let out = rt.block_on(make(&ex, &asked, None, &mut |_| {})).unwrap();
    assert_eq!(out["template"]["chosen"]["id"], "builtin:tech-reveal");
    let kept: Vec<&str> = out["reference"]["kept"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    assert!(kept.iter().any(|k| k.starts_with("seconds: 20")) && kept.iter().any(|k| k.starts_with("template")), "{kept:?}");
}
