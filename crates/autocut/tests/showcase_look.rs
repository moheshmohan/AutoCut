//! The showcase's look ([`filmcraft_autocut::look`]) against a real headless session: a theme
//! (and its fallback when fonts are missing), the timeline and its marker, spec chips, scene
//! shots, transitions from the catalogue with their sounds on the beat, and the finishing
//! (loudness, embedded captions, the SoundFont) — and hostile plans refused before anything is
//! built.

use filmcraft_autocut::analyze::{ClipFacts, FrameFact};
use filmcraft_autocut::look::{self, Look, Style, TransitionPlan};
use filmcraft_autocut::showcase::{self, Built, Entry, Options, Script, Shot};
use filmcraft_autocut::{Executor as _, SessionExecutor};
use filmcraft_engine::Session;
use filmcraft_engine::project::{ClipId, ItemId, ParamValue, Sequence, TrackItem};
use filmcraft_engine::time::Tick;
use serde_json::{Value, json};

const TPS: f64 = 254_016_000_000.0;
const FPS: f64 = 30.0;

fn demo() -> SessionExecutor {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    SessionExecutor::from(s)
}

fn fact(item: u64, label: &str, unix: i64, seconds: f64) -> ClipFacts {
    ClipFacts {
        item,
        name: format!("{item}.mp4"),
        seconds,
        created: None,
        unix: Some(unix),
        present: true,
        coverage: 1.0,
        label: label.into(),
        kind: "humanoid".into(),
        description: format!("a {label}"),
        best: seconds / 2.0,
        frames: vec![FrameFact { seconds: seconds / 2.0, present: true, area: 0.4, center: [0.5, 0.5] }],
        ..ClipFacts::default()
    }
}

/// GITEX 2022 opened by a scene of the hall: Pepper (2020), the scene, Spot and a toy robot.
fn gitex() -> Vec<Entry> {
    let facts = vec![
        fact(5, "Pepper humanoid robot", 1_600_000_000, 18.0),
        fact(6, "Boston Dynamics Spot", 1_665_741_668, 16.0),
        fact(7, "red toy robot", 1_665_841_668, 14.0),
    ];
    let robots = showcase::group(&facts, "robot", 0.3);
    let hall = Entry {
        label: "Dubai World Trade Centre".into(),
        kind: "scene".into(),
        caption: Some("GITEX 2022 · Dubai World Trade Centre".into()),
        unix: Some(1_665_741_668 - 3600),
        event: Some("GITEX 2022".into()),
        shots: vec![Shot { item: 7, name: "hall.mp4".into(), clip_seconds: 14.0, best: 7.0, ..Shot::default() }],
        ..Entry::default()
    };
    let mut out = vec![robots[0].clone(), hall];
    for mut e in robots.into_iter().skip(1) {
        e.event = Some("GITEX 2022".into());
        out.push(e);
    }
    for (i, e) in out.iter_mut().enumerate() {
        e.number = i + 1;
    }
    out
}

fn gitex_script() -> Script {
    Script {
        intro: "Every robot I have met, in order.".into(),
        lines: vec!["Pepper waved at me.".into(), String::new(), "Spot walked right up to me.".into(), "A small red toy robot.".into()],
        outro: "Thanks for watching.".into(),
        chips: [("3".to_string(), vec!["32.5 kg".to_string(), "1.6 m/s".to_string(), "14 kg payload".to_string()])].into_iter().collect(),
        ..Script::default()
    }
}

fn tech(look: Look) -> Options {
    Options { title: "Every Robot I've Met".into(), voice: Some("basic-male".into()), bpm: 120.0, width: 640, height: 360, look, ..Options::default() }
}

/// Run `f` on the built show's sequence.
fn with_seq<T>(ex: &SessionExecutor, built: &Built, f: impl FnOnce(&Sequence) -> T) -> T {
    let g = ex.0.lock().unwrap();
    let seq = g.project.sequence(ItemId(built.sequence.unwrap())).unwrap();
    f(seq)
}

fn item(seq: &Sequence, clip: u64) -> TrackItem {
    seq.find_item(ClipId(clip)).unwrap().1.clone()
}

/// The text layers' fonts of a graphic clip, in layer order.
fn fonts(it: &TrackItem) -> Vec<String> {
    it.effects
        .iter()
        .filter(|e| e.effect == "graphic_text")
        .filter_map(|e| match &e.param("font")?.value {
            ParamValue::Text(f) => Some(f.clone()),
            _ => None,
        })
        .collect()
}

fn shapes(it: &TrackItem) -> usize {
    it.effects.iter().filter(|e| e.effect == "graphic_shape").count()
}

fn on_frame(t: f64) -> bool {
    (t * FPS - (t * FPS).round()).abs() < 1e-6
}

#[tokio::test(flavor = "multi_thread")]
async fn a_tech_show_with_its_timeline_chips_scene_and_catalogue_transitions() {
    let ex = demo();
    let entries = gitex();
    let look = Look { style: Some(Style::tech()), timeline: true, transitions: Some(TransitionPlan::catalog()), ..Look::default() };
    let mut lines = Vec::new();
    let built = showcase::build(&ex, &tech(look), &entries, &gitex_script(), &mut |l| lines.push(l.to_string())).await.expect("the showcase builds");
    let notes: Vec<&String> = built.notes.iter().filter(|n| !n.starts_with("style: the font")).collect();
    assert!(notes.is_empty(), "{notes:?}\n{lines:#?}");
    assert_eq!((built.entries, built.scenes), (4, 1));

    // ---- the theme: its fonts as installed (Inter where one is missing) on every text
    let th = built.theme.clone().expect("a theme");
    let installed = look::installed_fonts(&ex).await.expect("fonts.list");
    let has = |f: &str| installed.iter().any(|(x, _)| x.eq_ignore_ascii_case(f));
    for (family, asked) in [(&th.title.family, "DIN Condensed"), (&th.body.family, "Futura"), (&th.number.family, "DIN Alternate")] {
        assert_eq!(family.as_str(), if has(asked) { asked } else { look::FALLBACK_FAMILY }, "{th:?}");
    }
    let (spot, hall) = (&built.layout[2], &built.layout[1]);
    with_seq(&ex, &built, |seq| {
        // the lower third: the name in the title font and the place in the body font, over a
        // panel with an accent bar
        let third = item(seq, spot.lower_third.unwrap());
        assert_eq!(fonts(&third), vec![th.title.family.clone(), th.body.family.clone()]);
        assert_eq!(shapes(&third), 2, "a panel and its bar");
        // the counter in the number font; it counts the three robots, not the scene
        let counter = seq.video_tracks[4].items.iter().find(|c| c.name == "Counter").expect("the counter").clone();
        assert_eq!(fonts(&counter), vec![th.number.family.clone()]);
        let text = counter.effects.iter().find(|e| e.effect == "graphic_text").and_then(|e| e.param("text")).unwrap();
        assert_eq!(text.keyframes.len(), 3, "three robots counted");
        // the second robot's number from its first frame, before its name comes in
        let says = |t: f64| text.value_at(counter.source_time_at(Tick::from_seconds_f64(t)));
        assert_eq!(says(spot.start + 0.1), ParamValue::Text("ROBOT #2".into()), "over the second robot's footage");
        // …and steps aside while the scene plays
        let op = counter.effect("opacity").and_then(|e| e.param("opacity")).expect("faded");
        assert_eq!(op.keyframes.len(), 4);
        let at = |t: f64| op.value_at(counter.source_time_at(Tick::from_seconds_f64(t))).as_f64().unwrap();
        assert_eq!((at(hall.start), at((hall.start + hall.end) / 2.0), at(hall.end)), (100.0, 0.0, 100.0));
        // the title card and the outro in the title font
        let titles: Vec<TrackItem> = seq.video_tracks[2].items.iter().filter(|c| fonts(c).first() == Some(&th.title.family)).cloned().collect();
        assert!(titles.len() >= 4, "title, three lower thirds, outro: {}", titles.len());
    });

    // ---- the scene: full frame, a caption and no name lower third, no chips, not counted
    assert!(hall.scene && hall.lower_third.is_none() && hall.caption.is_some() && hall.chips.is_empty(), "{hall:?}");
    assert!(built.layout.iter().filter(|e| !e.scene).all(|e| e.lower_third.is_some() && e.caption.is_none()));
    with_seq(&ex, &built, |seq| {
        let caption = item(seq, hall.caption.unwrap());
        assert_eq!(fonts(&caption), vec![th.body.family.clone()], "one line in the body font");
        let text = caption.effects.iter().find(|e| e.effect == "graphic_text").and_then(|e| e.param("text")).map(|p| p.value.clone());
        assert_eq!(text, Some(ParamValue::Text("GITEX 2022 · Dubai World Trade Centre".into())));
    });
    // no words over it, so nothing ducks the music there
    assert!(built.narration.iter().all(|n| n.end <= hall.start || n.start >= hall.end), "{:?}", built.narration);
    with_seq(&ex, &built, |seq| {
        let bed = seq.audio_tracks[2].items.first().expect("the music bed").clone();
        let level = bed.effect("volume").and_then(|e| e.param("level")).expect("ducked: volume keyframes");
        assert!(level.keyframes.len() >= 2, "the bed is ducked under the voice");
        let top = level.keyframes.iter().filter_map(|k| k.value.as_f64()).fold(f64::MIN, f64::max);
        let low = level.keyframes.iter().filter_map(|k| k.value.as_f64()).fold(f64::MAX, f64::min);
        assert!(top - low > 3.0, "{top} {low}");
        for k in 1..10 {
            let at = hall.start + 0.15 + (hall.end - hall.start - 0.3) * f64::from(k) / 10.0;
            let v = level.value_at(bed.source_time_at(Tick::from_seconds_f64(at))).as_f64().unwrap();
            assert!((v - top).abs() < 0.01, "the music is ducked at {at:.2}s in the scene ({hall:?}): {v} dB, not {top}");
        }
    });

    // ---- the chips: three, beside Spot's lower third only, one after the other, all leaving together
    for e in &built.layout {
        assert_eq!(e.chips.len(), if e.number == 3 { 3 } else { 0 }, "entry {}", e.number);
    }
    with_seq(&ex, &built, |seq| {
        let third = item(seq, spot.lower_third.unwrap());
        let chips: Vec<TrackItem> = spot.chips.iter().map(|c| item(seq, *c)).collect();
        for (k, c) in chips.iter().enumerate() {
            assert_eq!(fonts(c), vec![th.number.family.clone()]);
            assert_eq!(c.effect("opacity").and_then(|e| e.param("opacity")).map(|p| p.keyframes.len()), Some(4), "fades in and out");
            let track = seq.video_tracks.iter().position(|t| t.items.iter().any(|i| i.id == c.id)).unwrap();
            assert_eq!(track, 8 + k, "a track for each chip");
            if k > 0 {
                let gap = (c.start - chips[k - 1].start).seconds();
                assert!((gap - look::CHIP_STAGGER).abs() < 1.0 / FPS, "staggered: {gap}");
            }
            assert_eq!(c.end(), chips[0].end());
            assert!(c.start >= third.start && c.start.seconds() <= third.end().seconds(), "while the lower third is up");
        }
    });

    // ---- the timeline: from the first entry to the outro, the marker gliding to each date
    let tl = built.timeline.clone().expect("a timeline");
    assert_eq!((tl.scale.first_year, tl.scale.last_year), (2020, 2022));
    assert_eq!(tl.from, built.layout[0].start);
    assert_eq!(tl.to, built.chapters.last().unwrap().seconds, "hidden for the outro");
    assert_eq!(tl.bands.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(), vec!["GITEX 2022"]);
    // a stop at each new date: Pepper, the hall (Spot was filmed there the same day), the toy
    assert_eq!(tl.stops.iter().map(|s| s.entry).collect::<Vec<_>>(), vec![1, 2, 4]);
    for s in &tl.stops[1..] {
        let e = built.layout.iter().find(|e| e.number == s.entry).unwrap();
        assert_eq!(s.start, e.start, "it leaves as its entry starts");
        assert!(on_frame(s.start) && on_frame(s.arrive) && (s.arrive - s.start - look::GLIDE).abs() < 1.0 / FPS, "{s:?}");
    }
    with_seq(&ex, &built, |seq| {
        let marker = item(seq, tl.marker.unwrap());
        assert!(marker.start.seconds() - tl.from < 1e-6 && (marker.end().seconds() - tl.to).abs() < 1.0 / FPS);
        let pos = marker.effect("motion").and_then(|e| e.param("position")).expect("motion position");
        assert_eq!(pos.keyframes.len(), 2 * (tl.stops.len() - 1), "{:?}", pos.keyframes);
        let x_at = |t: f64| match pos.value_at(marker.source_time_at(Tick::from_seconds_f64(t))) {
            ParamValue::Vec2(v) => v.x,
            other => panic!("{other:?}"),
        };
        for w in tl.stops.windows(2) {
            let (before, s) = (w[0], w[1]);
            // where the previous date is as the entry starts, at the new one GLIDE later, and on its
            // way (eased) between
            assert!((x_at(s.start) - before.x).abs() < 1e-6, "{s:?}");
            assert!((x_at(s.arrive) - s.x).abs() < 1e-6, "{s:?}");
            let mid = x_at((s.start + s.arrive) / 2.0);
            assert!(mid > before.x.min(s.x) && mid < before.x.max(s.x), "gliding: {mid} between {} and {}", before.x, s.x);
        }
        // the years of the show picked out run by run, over the dim line, ticks, years and band
        assert_eq!(tl.now.len(), tl.runs.len());
        let base = item(seq, tl.base.unwrap());
        assert!(fonts(&base).len() >= 3 && shapes(&base) >= 5, "{} texts, {} shapes", fonts(&base).len(), shapes(&base));
    });

    // ---- transitions from the catalogue, each with its sound, snapped to the beat
    let at: Vec<(&str, &str, Option<&str>)> = built.transitions.iter().map(|t| (t.effect.as_str(), t.at, t.sound)).collect();
    assert_eq!(
        at,
        vec![
            ("lf_light_leak", "chapter", Some("riser")),
            ("gl_filmburn", "chapter", Some("riser")),
            ("cam_punch_in", "between", Some("whoosh")),
            ("flow_morph", "between", Some("whoosh")),
        ],
        "{:#?}",
        built.transitions
    );
    let snap = built.beat_snap.clone().expect("snapped");
    assert_eq!(snap["snapped"], 4, "{snap}");
    let beat = 0.5;
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    let placed: Vec<&Value> = seq["video"].as_array().unwrap().iter().flat_map(|t| t["transitions"].as_array().unwrap()).collect();
    for t in &built.transitions {
        let x = placed.iter().find(|x| x["id"].as_u64() == Some(t.id)).expect("on the timeline");
        assert_eq!(x["effect"], t.effect.as_str());
        assert_eq!(x["sound"]["kind"], t.sound.unwrap());
        let (start, dur) = (x["start"].as_f64().unwrap() / TPS, x["duration"].as_f64().unwrap() / TPS);
        let mid = start + dur / 2.0;
        let off = ((mid / beat) - (mid / beat).round()).abs() * beat;
        assert!(off <= 1.0 / FPS + 1e-6, "{} at {mid:.3}s is {off:.3}s off the beat", t.effect);
        let beats = dur / beat;
        assert!([0.25, 0.5, 1.0, 2.0, 4.0].iter().any(|b| (beats - b).abs() * beat <= 1.0 / FPS + 1e-6), "{} lasts {beats} beats", t.effect);
    }
    // no whoosh clips of its own: the transitions carry their sounds
    let audio: Vec<String> = seq["audio"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|t| t["items"].as_array().cloned().unwrap_or_default())
        .filter_map(|c| c["name"].as_str().map(str::to_string))
        .collect();
    assert!(!audio.iter().any(|n| n.starts_with("Whoosh")), "{audio:?}");
    // within entries: cuts (no transitions by default), on the beat too
    for e in &built.layout {
        for s in e.shots.iter().skip(1) {
            let off = ((s.start / beat) - (s.start / beat).round()).abs() * beat;
            assert!(off <= 1.0 / FPS + 1e-6, "a cut in entry {} at {:.3}s", e.number, s.start);
        }
    }
    // ---- and it renders, the timeline and a chip on screen
    let chip_time = with_seq(&ex, &built, |seq| item(seq, spot.chips[2]).start.seconds() + 1.0);
    assert!(ex.render(chip_time, 640).await.is_some_and(|p| p.starts_with(b"\x89PNG")));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_short_scene_keeps_its_caption_inside_it() {
    // a 2.5-second scene with no words between two robots: its caption used to be held for at
    // least 1.5 s after a lead-in, spilling into the next entry's lower third on the same track
    // ("V3 is not free here") and failing the whole build
    let ex = demo();
    let mut entries = gitex();
    entries[1].shots[0].clip_seconds = 2.5;
    entries[1].shots[0].best = 1.25;
    let look = Look { style: Some(Style::tech()), timeline: true, ..Look::default() };
    let built = showcase::build(&ex, &tech(look), &entries, &gitex_script(), &mut |_| {}).await.expect("a short scene still builds");
    let hall = built.layout.iter().find(|e| e.scene).expect("the scene");
    let next = built.layout.iter().skip_while(|e| !e.scene).nth(1).expect("the robot after it");
    with_seq(&ex, &built, |seq| {
        let caption = item(seq, hall.caption.expect("a caption"));
        let lower = item(seq, next.lower_third.expect("a lower third"));
        assert!(caption.end() <= lower.start, "the caption ends at {:?}, after the next lower third starts at {:?}", caption.end(), lower.start);
    });
}

#[tokio::test(flavor = "multi_thread")]
async fn a_theme_falls_back_to_inter_when_its_fonts_are_missing() {
    let ex = demo();
    let style = Style { title_font: "No Such Family Anywhere".into(), body_font: "Neither This One".into(), ..Style::tech() };
    let entries = showcase::group(&[fact(5, "Pepper humanoid robot", 1_600_000_000, 12.0), fact(6, "Boston Dynamics Spot", 1_665_741_668, 12.0)], "robot", 0.3);
    let script = Script { intro: "Hello.".into(), lines: vec!["One.".into(), "Two.".into()], outro: "Bye.".into(), ..Script::default() };
    let opts = Options { look: Look { style: Some(style), ..Look::default() }, ..tech(Look::default()) };
    let built = showcase::build(&ex, &opts, &entries, &script, &mut |_| {}).await.expect("the showcase builds");
    let th = built.theme.clone().unwrap();
    assert_eq!((th.title.family.as_str(), th.body.family.as_str()), ("Inter", "Inter"));
    assert!(th.missing.contains(&"No Such Family Anywhere".to_string()) && th.missing.contains(&"Neither This One".to_string()), "{th:?}");
    assert!(built.notes.iter().any(|n| n.contains("No Such Family Anywhere") && n.contains("Inter")), "{:?}", built.notes);
    with_seq(&ex, &built, |seq| {
        let third = item(seq, built.layout[0].lower_third.unwrap());
        assert_eq!(fonts(&third), vec!["Inter".to_string(), "Inter".to_string()]);
    });
    // and it still renders, the lower third over its panel
    let t = built.layout[0].start + 2.0;
    assert!(ex.render(t, 320).await.is_some_and(|p| p.starts_with(b"\x89PNG")));
}

#[tokio::test(flavor = "multi_thread")]
async fn without_a_look_the_show_is_as_before() {
    let ex = demo();
    let entries = showcase::group(&[fact(5, "Pepper humanoid robot", 1_600_000_000, 12.0), fact(6, "Boston Dynamics Spot", 1_665_741_668, 12.0)], "robot", 0.3);
    let script = Script { intro: "Hello.".into(), lines: vec!["One.".into(), "Two.".into()], outro: "Bye.".into(), ..Script::default() };
    let opts = Options { title: "Robots".into(), width: 640, height: 360, ..Options::default() };
    let built = showcase::build(&ex, &opts, &entries, &script, &mut |_| {}).await.expect("the showcase builds");
    assert!(built.notes.is_empty(), "{:?}", built.notes);
    assert!(built.theme.is_none() && built.timeline.is_none() && built.transitions.is_empty() && built.beat_snap.is_none() && built.loudness.is_none());
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    assert_eq!(seq["video"].as_array().map(Vec::len), Some(5), "no overlay tracks");
    let names: Vec<String> = seq["audio"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|t| t["items"].as_array().cloned().unwrap_or_default())
        .filter_map(|c| c["name"].as_str().map(str::to_string))
        .collect();
    assert!(names.iter().filter(|n| n.starts_with("Whoosh")).count() >= 2, "the whooshes as before: {names:?}");
    // the rotating set of transitions, a dip to black into each new year
    let effects: Vec<String> = seq["video"][1]["transitions"].as_array().unwrap().iter().filter_map(|x| x["effect"].as_str().map(str::to_string)).collect();
    assert!(effects.iter().filter(|e| *e == "dip_to_black").count() >= 2, "{effects:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn loudness_embedded_captions_and_the_soundfont_on_export() {
    let ex = demo();
    let dir = std::env::temp_dir().join(format!("filmcraft-look-export-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("look.mp4");
    let entries = showcase::group(&[fact(5, "Pepper humanoid robot", 1_600_000_000, 12.0), fact(6, "Boston Dynamics Spot", 1_665_741_668, 12.0)], "robot", 0.3);
    let script = Script {
        intro: "Every robot, in order.".into(),
        lines: vec!["The first one.".into(), "The second one.".into()],
        outro: "Bye for now.".into(),
        ..Script::default()
    };
    let look = Look { loudness: Some(look::Loudness::Named("social".into())), embed_captions: true, soundfont: Some("oscillators".into()), ..Look::default() };
    let opts = Options { export: Some(out.to_string_lossy().into_owned()), ..tech(look) };
    let built = showcase::build(&ex, &opts, &entries, &script, &mut |_| {}).await.expect("the showcase builds");
    assert!(built.notes.is_empty(), "{:?}", built.notes);
    // the mix at −14 LUFS under a −1 dBTP ceiling
    let l = built.loudness.clone().expect("normalised");
    assert_eq!(l["target"], "social", "{l}");
    assert!((l["loudnessLufs"].as_f64().unwrap() + 14.0).abs() < 1.0, "{l}");
    assert!(l["truePeakDbtp"].as_f64().unwrap() <= -0.9, "{l}");
    // the bed on the chosen instruments
    assert_eq!(built.music.as_ref().and_then(|m| m.instruments.as_deref()), Some("Built-in oscillator voices"), "{:?}", built.music);
    // the captions in the video (CEA-608/708 in ATSC A/53 user data: "GA94"), and beside it
    let written = built.export.as_ref().and_then(|r| r["path"].as_str()).map(std::path::PathBuf::from).unwrap_or(out.clone());
    let bytes = std::fs::read(&written).unwrap();
    assert!(bytes.windows(4).any(|w| w == b"GA94"), "no caption data in {}", written.display());
    assert_eq!(built.subtitle_files.len(), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn hostile_plans_are_refused_before_anything_is_built() {
    let ex = demo();
    let sequences = async || ex.execute("project.inspect", json!({})).await.unwrap().to_string().matches("\"type\":\"Sequence\"").count();
    let before = sequences().await;
    let entries = gitex();
    // chips for an entry that is not there, too many, too long
    for chips in [json!({"9": ["x"]}), json!({"3": ["a", "b", "c", "d"]}), json!({"3": ["x".repeat(200)]}), json!({"three": ["x"]})] {
        let script = Script { chips: serde_json::from_value(chips.clone()).unwrap(), ..gitex_script() };
        let err = showcase::build(&ex, &tech(Look::default()), &entries, &script, &mut |_| {}).await.unwrap_err().to_string();
        assert!(err.contains("chips"), "{chips}: {err}");
    }
    // a timeline over dates no calendar holds
    for unix in [i64::MAX, i64::MIN, 1_000_000_000_000_000, -10_000_000_000_000] {
        let mut hostile = entries.clone();
        hostile[3].unix = Some(unix);
        let opts = tech(Look { timeline: true, ..Look::default() });
        let err = showcase::build(&ex, &opts, &hostile, &gitex_script(), &mut |_| {}).await.unwrap_err().to_string();
        assert!(err.contains("timeline"), "{unix}: {err}");
    }
    // scenes alone are no showcase
    let scenes: Vec<Entry> = entries.iter().filter(|e| e.is_scene()).cloned().collect();
    let script = Script { lines: vec![String::new()], chips: Default::default(), ..gitex_script() };
    assert!(showcase::build(&ex, &tech(Look::default()), &scenes, &script, &mut |_| {}).await.is_err());
    assert_eq!(sequences().await, before, "nothing was built");
    // the plan's own fields: each refused with what is wrong, never a panic
    for plan in [
        json!({"style": "neon"}),
        json!({"style": {"accent": "electric"}}),
        json!({"style": {"titleFont": ""}}),
        json!({"style": {"panel": {"color": "#000", "alpha": 9}}}),
        json!({"style": [1, 2]}),
        json!({"timeline": 1}),
        json!({"events": [{"name": "GITEX", "from": "2022-13-01", "to": "2022-10-14"}]}),
        json!({"events": [{"name": "GITEX", "from": "2022-10-14", "to": "2021-10-10"}]}),
        json!({"events": {"name": "GITEX"}}),
        json!({"transitions": {"between": ["not_a_transition"]}}),
        json!({"transitions": {"between": [null]}}),
        json!({"transitions": {"sideways": []}}),
        json!({"transitions": 7}),
        json!({"loudness": "deafening"}),
        json!({"loudness": 12}),
        json!({"embedCaptions": "yes"}),
        json!({"soundfont": 5}),
    ] {
        let r = std::panic::catch_unwind(|| Look::from_plan(&plan));
        let err = r.expect("no panic").expect_err(&plan.to_string()).to_string();
        assert!(err.starts_with("invalid plan:"), "{plan}: {err}");
    }
    // …and through the app's reviewed plan too
    let doc: filmcraft_autocut::host::PlanDoc = serde_json::from_value(json!({
        "entries": [{"number": 1, "label": "x", "kind": "toy", "description": "", "unix": null, "shots": []}],
        "transitions": {"between": ["nope"]}
    }))
    .unwrap();
    assert!(doc.resolve().unwrap_err().to_string().contains("nope"));
    let doc: filmcraft_autocut::host::PlanDoc = serde_json::from_value(json!({
        "entries": [{"number": 1, "label": "x", "kind": "toy", "description": "", "unix": null, "shots": []}],
        "style": "tech", "timeline": true, "script": {"intro": "", "lines": ["a"], "outro": "", "chips": {"1": ["9 kg"]}}
    }))
    .unwrap();
    let (opts, _, script) = doc.resolve().unwrap();
    assert!(opts.look.timeline && opts.look.style == Some(Style::tech()));
    assert_eq!(script.chips.get("1").map(Vec::len), Some(1));
    assert_eq!(serde_json::to_value(&doc).unwrap()["style"], "tech", "the look is kept as written");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_bigger_frame_draws_every_title_and_the_counter_bigger() {
    // the counter, the intro and closing titles were sized in pixels: a 4K show drew them at
    // half their 1080p size. Built at twice the frame they now come out twice as big
    async fn sizes(w: u32, h: u32) -> (f64, Vec<f64>) {
        let ex = demo();
        let opts = Options { width: w, height: h, ..tech(Look::default()) };
        let built = showcase::build(&ex, &opts, &gitex(), &gitex_script(), &mut |_| {}).await.expect("the showcase builds");
        with_seq(&ex, &built, |seq| {
            let size = |c: &TrackItem| {
                c.effects.iter().find(|e| e.effect == "graphic_text").and_then(|e| e.param("size")).and_then(|p| p.value.as_f64()).unwrap_or(f64::NAN)
            };
            let all: Vec<&TrackItem> = seq.video_tracks.iter().flat_map(|t| t.items.iter()).collect();
            let counter = all.iter().find(|c| c.name == "Counter").map(|c| size(c)).expect("the counter");
            let mut titles: Vec<f64> = all.iter().filter(|c| c.name != "Counter").map(|c| size(c)).filter(|s| s.is_finite()).collect();
            titles.sort_by(|a, b| b.total_cmp(a));
            titles.truncate(3);
            (counter, titles)
        })
    }
    let (c1, t1) = sizes(640, 360).await;
    let (c2, t2) = sizes(1280, 720).await;
    assert!((c2 / c1 - 2.0).abs() < 1e-6, "counter {c1} → {c2}");
    assert_eq!(t1.len(), t2.len());
    for (a, b) in t1.iter().zip(&t2) {
        assert!((b / a - 2.0).abs() < 1e-6, "title {a} → {b}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_plans_counter_counts_versions_over_a_subject_qa_still_looks_for() {
    // the Akku Tesla show counted "VERSION #1/#2/#3" over a toy car: the counter word came from
    // the subject, so QA looked for a "version" in every shot
    let ex = demo();
    let opts = Options { subject: "toy car".into(), counter: Some("version".into()), ..tech(Look::default()) };
    let built = showcase::build(&ex, &opts, &gitex(), &gitex_script(), &mut |_| {}).await.expect("the showcase builds");
    let spot = built.layout[2].clone();
    with_seq(&ex, &built, |seq| {
        let all: Vec<&TrackItem> = seq.video_tracks.iter().flat_map(|t| t.items.iter()).collect();
        let counter = all.iter().find(|c| c.name == "Counter").expect("the counter");
        let text = counter.effects.iter().find(|e| e.effect == "graphic_text").and_then(|e| e.param("text")).unwrap();
        assert_eq!(text.value_at(counter.source_time_at(Tick::from_seconds_f64(spot.start + 0.1))), ParamValue::Text("VERSION #2".into()));
        let texts: Vec<String> = all
            .iter()
            .flat_map(|c| c.effects.iter())
            .filter(|e| e.effect == "graphic_text")
            .filter_map(|e| match &e.param("text")?.value {
                ParamValue::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert!(texts.iter().any(|t| t == "3 VERSIONS"), "the closing card counts versions: {texts:?}");
        assert!(!texts.iter().any(|t| t.contains("TOY CAR")), "{texts:?}");
    });
    // QA looks for the toy car
    let q = filmcraft_autocut::qa::Options::for_showcase(&opts, &built);
    assert_eq!((q.subject.as_str(), q.subject_check), ("toy car", true));
}
