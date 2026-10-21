use super::*;

#[test]
fn a_robot_too_wide_for_the_crop_is_shown_whole_in_a_band() {
    let frame = (1080.0, 1920.0);
    let land = 16.0 / 9.0;
    // portrait footage, and a robot narrow enough for the crop: followed by the crop
    assert_eq!(band(9.0 / 16.0, frame, Some([0.2, 0.2, 0.6, 0.6])), None);
    assert_eq!(band(land, frame, Some([0.45, 0.2, 0.2, 0.6])), None, "a fifth of the picture fits the crop's third");
    // a wide robot: the picture as wide as shows all of it, covering the frame's width
    let b = band(land, frame, Some([0.1, 0.2, 0.6, 0.6])).unwrap();
    assert!((b.width - 0.9 * 1080.0 / 0.6).abs() < 1e-6, "{b:?}");
    let (half_w, half_h) = (b.width / 2.0, b.width / land / 2.0);
    assert!(b.centre[0] - half_w <= 1e-6 && b.centre[0] + half_w >= 1080.0 - 1e-6, "{b:?}");
    assert!(b.centre[1] - half_h >= -1e-6 && b.centre[1] + half_h <= 1920.0 + 1e-6, "{b:?}");
    // the robot in the middle as far as the picture allows
    let robot_x = b.centre[0] - half_w + 0.4 * b.width;
    assert!((robot_x - 540.0).abs() < 1.0, "{robot_x}");
    // a robot filling the picture: the whole picture across the frame; not known: half again wider
    assert_eq!(band(land, frame, Some([0.0, 0.0, 1.0, 1.0])).map(|b| b.width), Some(1080.0));
    assert_eq!(band(land, frame, None).map(|b| b.width), Some(1620.0));
    // hostile numbers: no band, or a sane one
    for (a, f, s) in
        [(f64::NAN, frame, None), (land, (0.0, 1920.0), None), (land, frame, Some([f64::NAN, 0.0, 0.5, 0.5])), (1e9, frame, Some([0.0, 0.0, 1e-9, 1.0]))]
    {
        if let Some(b) = band(a, f, s) {
            assert!(b.width.is_finite() && b.centre.iter().all(|v| v.is_finite()), "{b:?}");
        }
    }
}

#[test]
fn shots_start_where_asked() {
    let a = at_of(&json!({"My first robot": {"item": 1347, "seconds": 9.5}, " NAO ": 3, "bad": -1, "worse": "x", "": 2, "no item": {"seconds": 1}}));
    for want in [
        At { label: "My first robot".into(), item: Some(1347), seconds: 9.5 },
        At { label: "NAO".into(), item: None, seconds: 3.0 },
        At { label: "no item".into(), item: None, seconds: 1.0 },
    ] {
        assert!(a.contains(&want), "{want:?} in {a:?}");
    }
    assert_eq!(a.len(), 3, "the bad ones left out: {a:?}");
    assert!(at_of(&json!([1, 2])).is_empty() && at_of(&json!(null)).is_empty());
    // another clip of the entry: the window the show played of it
    let e = Entry {
        number: 1,
        label: "My first robot".into(),
        shots: vec![Shot { item: 1346, clip_seconds: 42.0, ..Shot::default() }, Shot { item: 1347, clip_seconds: 29.7, ..Shot::default() }],
        ..Entry::default()
    };
    let layout = vec![PlacedEntry {
        number: 1,
        shots: vec![PlacedShot { clip: 5, item: 1347, source: Some([6.7, 14.5]), ..PlacedShot::default() }],
        ..PlacedEntry::default()
    }];
    let p = pick_for(&e, &layout, 1347).unwrap();
    assert_eq!((p.item, p.window, p.show_clip), (1347, [6.7, 14.5], Some(5)));
    assert_eq!(pick_for(&e, &layout, 1346).map(|p| p.window), Some([0.2, 41.8]), "not played: the whole clip");
    assert!(pick_for(&e, &layout, 9).is_none());
}

fn cand(number: usize, label: &str, kind: &str, year: i64, score: f64) -> Candidate {
    Candidate {
        number,
        label: label.into(),
        kind: kind.into(),
        unix: crate::look::unix_of(year, 6, 1),
        score,
        branded: false,
        shot: ShotPick {
            item: number as u64,
            name: format!("{number}.mp4"),
            clip_seconds: 20.0,
            window: [2.0, 12.0],
            moment: Some(Moment { seconds: 6.0, area: 0.5, center: [0.5, 0.5] }),
            photo: false,
            show_clip: Some(100 + number as u64),
            show_fit: None,
        },
    }
}

/// A show's robots over the years, a few visited twice.
fn show() -> Vec<Candidate> {
    vec![
        cand(1, "My first robot", "wheeled", 2013, 1.0),
        cand(2, "DIY tracked robot", "wheeled", 2020, 0.9),
        cand(3, "Promobot Robo-C", "humanoid", 2021, 1.4),
        cand(4, "Boston Dynamics Spot", "quadruped", 2022, 1.7),
        cand(5, "Pepper", "humanoid", 2022, 1.5),
        cand(6, "Ameca", "humanoid", 2022, 1.6),
        cand(7, "Unitree Go1", "quadruped", 2022, 1.2),
        cand(8, "Dilly S serving robot", "service robot", 2023, 0.8),
        cand(9, "Pepper, again", "humanoid", 2024, 1.3),
        cand(10, "Unitree Go2", "quadruped", 2024, 1.3),
        cand(11, "Spot, again", "quadruped", 2025, 1.8),
        cand(12, "Ameca, again", "humanoid", 2025, 2.0),
        cand(13, "Robotic basketball hoop", "arm", 2025, 0.7),
    ]
}

#[test]
fn every_cut_lands_on_a_beat_and_the_lengths_hold() {
    for bpm in [80.0, 96.0, 110.0, 120.0, 124.0, 128.0, 140.0, 160.0, 180.0] {
        let beats = Beats::grid(bpm);
        for seconds in [10.0, 15.0, 20.0, 30.0, 34.0, 40.0, 60.0, 90.0] {
            let n = shot_count(beats.period(), seconds, None, 40);
            let t = timing(&beats, seconds, n).unwrap_or_else(|e| panic!("{bpm} bpm, {seconds} s, {n} shots: {e}"));
            assert_eq!(t.shots.len(), n);
            // every cut on a beat, to the frame
            for c in t.cuts() {
                assert!(beats.distance(c) <= 1.0 / FPS, "{bpm} bpm {seconds} s: cut at {c} is {} off the beat", beats.distance(c));
            }
            // the shots follow each other with no gap, from the hook to the end card
            assert_eq!(t.hook[0], 0.0);
            assert_eq!(t.shots.first().map(|s| s[0]), Some(t.hook[1]));
            for w in t.shots.windows(2) {
                assert_eq!(w[0][1], w[1][0]);
            }
            assert_eq!(t.shots.last().map(|s| s[1]), Some(t.end_card[0]));
            assert_eq!(t.end_card[1], t.total);
            // 1.3–2.4 s a shot (to a frame; longer only when 30 of those cannot fill a long one),
            // the longer ones first
            let most = max_shot(beats.period(), seconds - HOOK_SECONDS - END_CARD_SECONDS);
            assert!(seconds > 70.0 || most == MAX_SHOT);
            for s in &t.shots {
                let len = s[1] - s[0];
                assert!(len >= MIN_SHOT - 1.5 / FPS && len <= most + 1.5 / FPS, "{bpm} bpm: a {len} s shot");
            }
            assert!(t.shot_beats.windows(2).all(|w| w[0] >= w[1]), "the cutting quickens: {:?}", t.shot_beats);
            // the hook about three seconds, the end card about three from the hit, on a bar
            assert!((t.hook[1] - 3.0).abs() <= t.beat * 2.5, "{bpm}: hook {:?}", t.hook);
            let end = t.end_card[1] - t.end_card[0];
            assert!((1.2..=3.4).contains(&end), "{bpm}: end card {end}");
            let hit_beats = t.hook_beats + t.shot_beats.iter().sum::<usize>();
            if n >= 4 {
                assert_eq!(hit_beats % 4, 0, "{bpm} bpm {seconds} s: the final hit on a bar");
                // and the length near the one asked for (within a bar and a half)
                assert!((t.total - seconds).abs() <= 6.0 * t.beat + 0.1, "{bpm} bpm: {} s for {seconds}", t.total);
            }
        }
    }
}

#[test]
fn a_reel_at_124_bpm_is_about_34_seconds_of_16_robots() {
    let beats = Beats::grid(124.0);
    let n = shot_count(beats.period(), 34.0, None, 47);
    assert!((12..=18).contains(&n), "{n}");
    let t = timing(&beats, 34.0, n).unwrap();
    assert!((30.0..=40.0).contains(&t.total), "{}", t.total);
    assert_eq!(t.hook_beats, 6);
    // a generated bed of the trailer's length puts its final hit where the end card starts:
    // it rings out at least two beats (1.2 s) and less than a bar more
    let tail = t.total - t.end_card[0];
    assert!(tail >= 1.2 && tail < 4.0 * t.beat + 1.2, "{tail}");
    // the review sheet: 12–16 frames, the hook first, the end card last, the first and the last
    // shot in it
    let sheet = sheet_times(&t);
    assert!((12..=SHEET_FRAMES).contains(&sheet.len()), "{}", sheet.len());
    assert!(sheet.windows(2).all(|w| w[0] < w[1]));
    assert!(sheet[0] < t.hook[1] && *sheet.last().unwrap() > t.end_card[0]);
    let mid = |s: [f64; 2]| (s[0] + s[1]) / 2.0;
    assert!(sheet.contains(&mid(t.shots[0])) && sheet.contains(&mid(*t.shots.last().unwrap())));
    // a count asked for is kept
    assert_eq!(shot_count(beats.period(), 34.0, Some(14), 47), 14);
    assert_eq!(shot_count(beats.period(), 34.0, Some(30), 47), 30, "the length follows the count asked for");
    let long = timing(&beats, 34.0, 30).unwrap();
    assert!(long.total > 40.0 && long.shots.iter().all(|s| s[1] - s[0] <= MAX_SHOT + 0.05), "{long:?}");
    assert_eq!(shot_count(beats.period(), 34.0, None, 5), 5, "never more than there are");
    // hostile numbers give something sane, never a panic
    for (beat, secs) in [(f64::NAN, 30.0), (0.0, 30.0), (-1.0, f64::INFINITY), (1e9, 1e9)] {
        let n = shot_count(beat, secs, None, 10);
        assert!((1..=MAX_ROBOTS).contains(&n));
    }
    assert!(timing(&beats, 34.0, 0).is_err());
    assert!(timing(&beats, f64::NAN, 3).is_ok(), "a NaN length falls back to the default");
}

#[test]
fn the_hook_is_the_most_striking_and_the_montage_runs_through_the_years() {
    let c = show();
    let ch = choose(&c, 8, None, &[], &[]).unwrap();
    assert_eq!(c[ch.hook].label, "Ameca, again", "the most striking");
    let labels: Vec<&str> = ch.montage.iter().map(|&i| c[i].label.as_str()).collect();
    assert_eq!(labels.len(), 8, "{labels:?}");
    // chronological
    let years: Vec<i64> = ch.montage.iter().map(|&i| crate::showcase::ymd(c[i].unix.unwrap()).0).collect();
    assert!(years.windows(2).all(|w| w[0] <= w[1]), "{years:?}");
    // the first robot and the last
    assert_eq!(labels.first(), Some(&"My first robot"));
    assert_eq!(labels.last(), Some(&"Robotic basketball hoop"));
    // no robot twice, and not the hook's robot again
    assert!(!labels.contains(&"Ameca"), "the hook's robot again: {labels:?}");
    let spot = labels.iter().filter(|l| l.contains("Spot")).count();
    let pepper = labels.iter().filter(|l| l.contains("Pepper")).count();
    assert!(spot <= 1 && pepper <= 1, "{labels:?}");
    // asked for: a hook, robots left out and ones kept in
    let ch = choose(&c, 6, Some("boston dynamics spot"), &["Dilly S serving robot".into()], &["Pepper".into(), "#3".into()]).unwrap();
    assert_eq!(c[ch.hook].label, "Boston Dynamics Spot");
    let labels: Vec<&str> = ch.montage.iter().map(|&i| c[i].label.as_str()).collect();
    assert!(labels.contains(&"Dilly S serving robot"), "{labels:?}");
    assert!(!labels.iter().any(|l| l.contains("Pepper") || l.contains("Spot") || *l == "Promobot Robo-C"), "{labels:?}");
    let e = choose(&c, 6, Some("R2-D2"), &[], &[]).unwrap_err().to_string();
    assert!(e.contains("R2-D2") && e.contains("Ameca"), "{e}");
    // a robot whose entry hides a sponsor's logo is left for last
    let mut c2 = show();
    for x in c2.iter_mut().filter(|x| x.label == "Promobot Robo-C") {
        x.branded = true;
        x.score = striking(&x.kind, 0.5, false, false, true);
    }
    let ch = choose(&c2, 6, None, &[], &[]).unwrap();
    assert!(!ch.montage.iter().any(|&i| c2[i].branded), "branded only when nothing else is left");
    // a clip too short for the hook is not the hook
    let mut c3 = show();
    for x in c3.iter_mut().filter(|x| x.label == "Ameca, again") {
        x.shot.clip_seconds = 3.0;
    }
    assert_ne!(c3[choose(&c3, 4, None, &[], &[]).unwrap().hook].label, "Ameca, again");
    assert!(choose(&c3, 4, Some("Ameca, again"), &[], &[]).is_err());
    // nothing to pick from
    assert!(choose(&[], 4, None, &[], &[]).is_err());
}

#[test]
fn visits_of_one_robot_are_one_robot() {
    assert!(same_robot("Pepper", "Pepper, again"));
    assert!(same_robot("Boston Dynamics Spot", "Spot, again"));
    assert!(same_robot("Ameca", "ameca again"));
    assert!(!same_robot("Unitree Go1", "Unitree Go2"));
    assert!(!same_robot("Serving robot", "Reception robot"));
    assert!(!same_robot("", ""));
}

#[test]
fn shots_play_the_best_moment_inside_the_shows_window() {
    let c = cand(1, "x", "humanoid", 2022, 1.0);
    // the moment at 6 s of a 2–12 s window: a little before it, inside
    let (s, inside) = window_start(&c.shot, 2.0).unwrap();
    assert!(inside && (s - (6.0 - 0.9)).abs() < 1e-9, "{s}");
    // near the window's end: kept inside it
    let mut late = c.shot.clone();
    late.moment = Some(Moment { seconds: 11.9, area: 0.4, center: [0.5, 0.5] });
    let (s, inside) = window_start(&late, 2.0).unwrap();
    assert!(inside && s + 2.0 <= 12.0 + 1e-9, "{s}");
    // a window shorter than the shot: centred on it, reaching past it
    let mut short = c.shot.clone();
    short.window = [5.0, 6.0];
    let (s, inside) = window_start(&short, 2.0).unwrap();
    assert!(!inside && (s - 4.5).abs() < 1e-9, "{s}");
    // a clip too short, a photo, nonsense
    let mut tiny = c.shot.clone();
    tiny.clip_seconds = 1.0;
    tiny.window = [0.1, 0.9];
    assert!(window_start(&tiny, 2.0).is_none());
    let photo = ShotPick { photo: true, ..tiny };
    assert_eq!(window_start(&photo, 2.0), Some((0.0, true)));
    assert!(window_start(&c.shot, f64::NAN).is_none() && window_start(&c.shot, -1.0).is_none());
}

#[test]
fn faces_the_show_hid_are_hidden_at_the_size_it_hid_them() {
    let (show, reel) = ((1920.0, 1080.0), (1080.0, 1920.0));
    // landscape footage fills both frames to their height: about the show's 5 %
    let land = face_threshold(None, 16.0 / 9.0, show, reel);
    assert!((0.04..0.05).contains(&land), "{land}");
    // a portrait clip filled the 16:9 show three times larger: its faces count from a third
    let tall = face_threshold(Some(Fit::Fill), 9.0 / 16.0, show, reel);
    assert!(tall < land / 3.0 + 0.002 && tall >= 0.005, "{tall}");
    // shown whole in the show (smaller than here): the show's 5 %, give or take the margin
    let whole = face_threshold(Some(Fit::Full), 9.0 / 16.0, show, reel);
    assert!((0.04..0.05).contains(&whole), "{whole}");
    // never above the show's, never zero, for anything
    for a in [f64::NAN, 0.0, -1.0, 0.1, 1.0, 4.0, f64::INFINITY] {
        for fit in [None, Some(Fit::Fill), Some(Fit::Full)] {
            let t = face_threshold(fit, a, (f64::NAN, 0.0), reel);
            assert!((0.005..=SHOW_MIN_FACE).contains(&t), "{a} {fit:?}: {t}");
        }
    }
}

#[test]
fn redaction_carries_over_with_the_entries_targets_and_robots_left_sharp() {
    use crate::showcase::Shot;
    // the show's Promobot entry hides the DP World logo; its robot is excluded from the faces
    let promobot = Entry {
        number: 53,
        label: "Promobot".into(),
        kind: "humanoid".into(),
        shots: vec![Shot { item: 70, largest: Some(Moment { seconds: 6.0, area: 0.2, center: [0.5, 0.4] }), ..Shot::default() }],
        redact: Some(json!({"targets": ["DP World logo"]})),
        ..Entry::default()
    };
    let ameca =
        Entry { number: 52, label: "Ameca, again".into(), kind: "humanoid".into(), shots: vec![Shot { item: 67, ..Shot::default() }], ..Entry::default() };
    let entries = vec![ameca, promobot];
    // the trailer's clips of them (one entry a shot, as the trailer lays them out)
    let layout = vec![
        PlacedEntry { number: 52, shots: vec![PlacedShot { clip: 900, item: 67, ..PlacedShot::default() }], ..PlacedEntry::default() },
        PlacedEntry { number: 53, shots: vec![PlacedShot { clip: 901, item: 70, ..PlacedShot::default() }], ..PlacedEntry::default() },
    ];
    let th = face_threshold(None, 16.0 / 9.0, (1920.0, 1080.0), (1080.0, 1920.0));
    let plan = Redact { min_face_height: Some(th), ..Redact::default() };
    let mut notes = Vec::new();
    let p = crate::privacy::show_params(Some(&plan), &layout, &entries, &[(70, 16.0 / 9.0)], &mut notes).unwrap();
    assert_eq!(p["clips"], json!([900, 901]));
    assert_eq!(p["kinds"], json!(["faces", "plates"]));
    assert_eq!(p["minFaceHeight"].as_f64(), Some(th));
    assert_eq!(p["targets"], json!([{"text": "DP World logo", "clips": [901]}]), "the logo stays hidden in the trailer's Promobot");
    let ex = p["exclude"].as_array().unwrap();
    assert!(ex.iter().any(|b| b["clip"] == 901 && b["seconds"] == 6.0), "the robot itself stays sharp: {p}");
    // the show hid faces in item 67's clip; the trailer's shot of it hides none: QA says so
    let show_hidden = vec![Hidden { clip: 500, name: "IMG_8187.MOV".into(), faces: 5, ..Hidden::default() }];
    let show_layout = vec![PlacedEntry { number: 52, shots: vec![PlacedShot { clip: 500, item: 67, ..PlacedShot::default() }], ..PlacedEntry::default() }];
    let shot = |clip: u64, item: u64| TrailerShot {
        number: 1,
        entry: 52,
        label: "Ameca, again".into(),
        year: Some(2025),
        clip,
        item,
        start: 2.9,
        end: 4.8,
        from: 30.0,
        to: 31.9,
        in_show_window: true,
        band: None,
        background: None,
        followed: Some(true),
        label_clip: None,
    };
    let f = carried_over(&show_hidden, &show_layout, &[], &[shot(900, 67)], &[]);
    assert_eq!(f.len(), 1);
    assert!(f[0].message.contains("5 face(s)") && f[0].severity == Severity::Warning, "{:?}", f[0]);
    // hidden in the trailer too: nothing to say
    let here = vec![Hidden { clip: 900, faces: 2, ..Hidden::default() }];
    assert!(carried_over(&show_hidden, &show_layout, &here, &[shot(900, 67)], &[]).is_empty());
    // footage the show hid nothing in
    assert!(carried_over(&show_hidden, &show_layout, &[], &[shot(901, 70)], &[]).is_empty());
    // the show's own blur carried onto the shot: for the record only
    let f = carried_over(&show_hidden, &show_layout, &[], &[shot(900, 67)], &[900]);
    assert_eq!((f.len(), f[0].severity), (1, Severity::Info), "{f:?}");
}

#[test]
fn text_and_the_logo_sit_inside_the_safe_zone() {
    for platform in Platform::ALL {
        for frame in [(1080.0, 1920.0), (1080.0, 1080.0), (1080.0, 1350.0), (720.0, 1280.0)] {
            let b = boxes(platform, frame);
            for (what, r) in [("title", b.title), ("label", b.label), ("logo", b.logo), ("end text", b.end_text)] {
                assert!(inside(r, b.safe), "{platform:?} {frame:?}: {what} {r:?} outside {:?}", b.safe);
                assert!(r[2] > r[0] && r[3] > r[1], "{platform:?} {frame:?}: {what} is empty: {r:?}");
            }
            // the title above the label, the logo above its lines
            assert!(b.title[3] < b.label[1], "{platform:?} {frame:?}");
            assert!(b.logo[3] < b.end_text[1], "{platform:?} {frame:?}");
            // Reels' safe area: 6 % at the sides, 14 % at the top, 35 % at the bottom
            if platform == Platform::Reels && frame == (1080.0, 1920.0) {
                assert_eq!(b.safe.map(f64::round), [65.0, 269.0, 1015.0, 1248.0]);
            }
        }
    }
    // a nonsense frame gives a sane one
    let b = boxes(Platform::Reels, (f64::NAN, -3.0));
    assert!(b.safe.iter().all(|v| v.is_finite()));
}

#[test]
fn titles_labels_and_the_caption() {
    assert_eq!(hook_title(None, 48, Some((2013, 2025)), "robot"), ("48 robots".to_string(), Some("2013 → 2025".to_string())));
    assert_eq!(hook_title(None, 1, Some((2020, 2020)), "robot"), ("1 robot".to_string(), Some("2020".to_string())));
    assert_eq!(hook_title(Some("Every robot · since 2013"), 9, None, "robot"), ("Every robot".to_string(), Some("since 2013".to_string())));
    assert_eq!(hook_title(Some("  "), 3, None, "robot").0, "3 robots");
    let c = show();
    assert_eq!(label_lines(&c[5]), ("Ameca".to_string(), Some("2022".to_string())));
    let picked: Vec<&Candidate> = c.iter().take(5).collect();
    let (text, tags) = caption(
        &("48 robots".into(), Some("2013 → 2025".into())),
        &picked,
        &["Promobot Robo-C".into()],
        &["GITEX Global 2022, Dubai".into(), "Expo 2020 Dubai".into()],
        Some("@H4hacks"),
        Some("ml"),
        DEFAULT_END_TEXT,
    );
    assert!(text.starts_with("48 robots, 2013 → 2025."), "{text}");
    // by name (not "My first robot", not "DIY tracked robot"): the one asked for, then another kind
    assert!(text.contains("\nPromobot Robo-C, Boston Dynamics Spot, Pepper and more: every robot"), "{text}");
    assert!(text.contains("Full video on YouTube (in Malayalam): @H4hacks, link in bio."), "{text}");
    assert!(!text.contains("http"), "a Reel's caption holds no link: {text}");
    for t in ["#robots", "#humanoidrobot", "#robotdog", "#GITEX", "#Expo2020", "#Dubai", "#Malayalam", "#H4hacks"] {
        assert!(tags.iter().any(|x| x == t), "{t} in {tags:?}");
    }
    assert!(tags.len() <= 15);
}

#[test]
fn hostile_plans_reports_and_parameters_never_panic() {
    let bad = [
        json!(null),
        json!(5),
        json!({"entries": "x"}),
        json!({"entries": [1, "a", null, {"number": -3, "label": 7}, {"label": "ok", "shots": [{"item": "x"}]}]}),
        json!({"entries": [{"number": 1, "label": "R", "kind": "humanoid", "unix": i64::MAX, "shots": [{"item": 1, "name": "a", "clip_seconds": f64::MAX, "best": -1e300}]}]}),
        json!({"layout": [{"number": 1, "label": "R", "start": 0, "end": 1, "shots": [{"clip": 1, "item": 1, "start": 0, "end": 1, "source": [5, 2]}]}], "privacy": [{"clip": "x"}], "narration": "y"}),
    ];
    for plan in &bad {
        for report in &bad {
            let s = std::panic::catch_unwind(|| Show::from_docs(Some(plan), Some(report))).expect("no panic");
            let cands = candidates(&s.entries, &s.layout);
            for c in &cands {
                let _ = window_start(&c.shot, 2.0);
            }
            let _ = choose(&cands, 4, None, &[], &[]);
            let _ = s.years();
        }
    }
    let (e, l) = entries_from_sequence(
        &json!({"video": [{"items": [{"clip": 1}]}, {"items": [{"clip": 2, "item": 3, "start": 9, "end": 2}, "x", {"clip": 4, "item": 5, "start": 0, "end": 254016000000_i64, "sourceIn": 0, "sourceOut": -1}]}]}),
    );
    assert!(e.is_empty() && l.is_empty());
    for p in [
        json!({"seconds": f64::MAX}),
        json!({"seconds": -1}),
        json!({"aspect": "16:9"}),
        json!({"platform": "snapchat"}),
        json!({"robots": 1000}),
        json!({"bpm": 1e12}),
        json!({"redact": "elephants"}),
        json!({"redact": 3}),
    ] {
        assert!(Options::from_params(&p).is_err(), "{p}");
    }
    let o = Options::from_params(&json!({"title": "a\nb", "hook": "  ", "exclude": [1, "x", " "], "seed": -3})).unwrap();
    assert_eq!((o.title, o.hook, o.exclude, o.seed), (None, None, vec!["x".to_string()], 7));
}

#[test]
fn entries_come_from_the_sequence_when_there_is_no_plan() {
    const T: i64 = 254_016_000_000;
    let seq = json!({"video": [
        {"items": []},
        {"items": [
            {"clip": 10, "item": 1, "name": "a.mp4", "start": 0, "end": 5 * T, "sourceIn": 2 * T, "sourceOut": 7 * T},
            {"clip": 11, "item": 2, "name": "b.mp4", "start": 5 * T, "end": 9 * T, "sourceIn": 0, "sourceOut": 4 * T},
            {"clip": 12, "item": 3, "name": "c.mov", "start": 9 * T, "end": 12 * T, "sourceIn": T, "sourceOut": 4 * T}
        ]},
        {"items": [
            {"clip": 20, "item": 9, "name": "Pepper", "start": T / 2, "end": 8 * T},
            {"clip": 21, "item": 9, "name": "NAO", "start": 9 * T + T / 3, "end": 11 * T}
        ]}
    ]});
    let (entries, layout) = entries_from_sequence(&seq);
    let labels: Vec<&str> = entries.iter().map(|e| e.label.as_str()).collect();
    assert_eq!(labels, ["Pepper", "NAO"], "two clips under one lower third are one entry");
    assert_eq!(layout[0].shots.len(), 2);
    assert_eq!(layout[1].shots[0].source, Some([1.0, 4.0]));
    let c = candidates(&entries, &layout);
    assert_eq!(c.len(), 2);
}
