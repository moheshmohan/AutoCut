use super::*;

/// Where `a` and `b` first differ (numbers to the bit: plans keep every number exactly), as a
/// path; `None` when they are the same.
fn diff(a: &Value, b: &Value, path: &str) -> Option<String> {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for k in x.keys().chain(y.keys()) {
                let p = format!("{path}.{k}");
                match (x.get(k), y.get(k)) {
                    (Some(u), Some(v)) => {
                        if let Some(d) = diff(u, v, &p) {
                            return Some(d);
                        }
                    }
                    (u, v) => return Some(format!("{p}: {u:?} vs {v:?}")),
                }
            }
            None
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return Some(format!("{path}: {} vs {} items", x.len(), y.len()));
            }
            x.iter().zip(y).enumerate().find_map(|(i, (u, v))| diff(u, v, &format!("{path}[{i}]")))
        }
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            (x.to_bits() != y.to_bits()).then(|| format!("{path}: {x} vs {y}"))
        }
        _ => (a != b).then(|| format!("{path}: {a} vs {b}")),
    }
}

fn data(name: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/plan_patch").join(name);
    serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap()
}

/// The trimmed fixture: seven entries of the real v4 plan and the v5 round's notes for them,
/// against what the round's script made of them (renumbered).
#[test]
fn a_real_round_of_notes_reproduces_its_plans() {
    let clips = clips_from_inspect(&data("clips.json"));
    assert_eq!(clips.len(), 6);
    let p = apply(&data("base.json"), &[data("notes.json")], &clips).unwrap();
    if let Some(d) = diff(&p.plan, &data("expected.json"), "plan") {
        panic!("English plan differs at {d}");
    }
    assert_eq!((p.entries, p.robots, p.silent_scenes), (10, 9, 0));
    assert!(p.unlined.is_empty(), "{:?}", p.unlined);
    // "Six robots…" in the base became "Nine robots…"
    assert!(p.plan["script"]["outro"].as_str().unwrap().starts_with("Nine robots"));
    assert_eq!(p.notes.len(), 1, "{:?}", p.notes);
    let (ml, notes) = with_lines(&p.plan, &data("lines-ml.json"), "ml").unwrap();
    assert!(notes.is_empty(), "{notes:?}");
    if let Some(d) = diff(&ml, &data("expected-ml.json"), "plan") {
        panic!("Malayalam plan differs at {d}");
    }
}

/// The whole real round, when its files are at hand (`FILMCRAFT_SHOWCASE_FIXTURES`: a folder with
/// plan-v4.json, notes-v5.json, project-items.json, script-ml.json, plan-v5.json and
/// plan-v5-ml.json); skipped otherwise.
#[test]
fn the_whole_v5_round_when_its_files_are_here() {
    let Some(dir) = std::env::var_os("FILMCRAFT_SHOWCASE_FIXTURES").map(std::path::PathBuf::from) else { return };
    let read = |n: &str| -> Value { serde_json::from_slice(&std::fs::read(dir.join(n)).unwrap()).unwrap() };
    let clips = clips_from_inspect(&read("project-items.json"));
    let p = apply(&read("plan-v4.json"), &[read("notes-v5.json")], &clips).unwrap();
    if let Some(d) = diff(&p.plan, &read("plan-v5.json"), "plan") {
        panic!("plan-v5.json differs at {d}");
    }
    assert_eq!((p.entries, p.robots), (54, 48));
    let (ml, _) = with_lines(&p.plan, &read("script-ml.json"), "ml").unwrap();
    if let Some(d) = diff(&ml, &read("plan-v5-ml.json"), "plan") {
        panic!("plan-v5-ml.json differs at {d}");
    }
}

fn shot(item: u64, best: f64) -> Value {
    json!({"item": item, "name": format!("c{item}.mp4"), "clip_seconds": 20.0, "best": best, "seen": [best], "unseen": [], "largest": null})
}

fn base() -> Value {
    json!({
        "style": "tech",
        "entries": [
            {"number": 1, "label": "Hall", "kind": "scene", "description": "", "unix": 100, "shots": [shot(1, 2.0)]},
            {"number": 2, "label": "Arm", "kind": "arm", "description": "an arm", "unix": 200, "shots": [shot(2, 3.0), shot(3, 4.0)]},
            {"number": 3, "label": "Dog", "kind": "quadruped", "description": "a dog", "unix": 300, "shots": [shot(4, 5.0)], "place": "Dubai"},
        ],
        "script": {
            "intro": "Hi.", "outro": "Two robots, the end.", "lines": ["", "The arm.", "The dog."], "templated": 2,
            "chips": {"1": ["x"], "2": ["a", "b", "c", "d"]}, "delivery": {"intro": "excited", "3": "calm"},
            "pronounce": {"Dog": "Dawg"}, "say": {"3": "The dawg."}
        }
    })
}

#[test]
fn entries_change_split_go_and_come_in_time_order() {
    let clips: Clips = [(9, ClipInfo { name: "new.mov".into(), seconds: 30.0, still: false })].into();
    let patch = json!({
        "timezone": "+04:00",
        "entries": {
            "Arm": {"label": "Big arm", "line": "A big arm.", "shotEdits": {"3": {"speed": 2.0, "mute": true}},
                    "split": [{"label": "Small arm", "unix": "1970-01-01 04:04:00", "shots": [{"item": 3}], "chips": ["s"]}]},
            "dog": {"place": null, "event": "Expo", "delivery": "amazed", "say": null},
            "Hall": {"remove": true}
        },
        "add": [{"label": "New", "unix": 250, "shots": [{"item": 9, "from": 2, "to": 10}], "line": "New one.", "chips": ["n1", "n2", "n3", "n4"]}],
        "script": {"pronounce": {"Dog": null, "Arm": "Ahm"}, "delivery": {"outro": "calm"}},
        "transitions": true,
        "style": null
    });
    let p = apply(&base(), &[patch], &clips).unwrap();
    let e = p.plan["entries"].as_array().unwrap();
    let labels: Vec<&str> = e.iter().map(|e| e["label"].as_str().unwrap()).collect();
    // the split at 04:04 in +04:00 is 240 s after the epoch: between the arm (200) and the new one
    assert_eq!(labels, ["Big arm", "Small arm", "New", "Dog"]);
    assert_eq!(e.iter().map(|e| e["number"].as_u64().unwrap()).collect::<Vec<_>>(), [1, 2, 3, 4]);
    assert_eq!(e[1]["unix"], json!(240));
    // the edit: on the arm's shot of item 3, before the split copied it…
    assert_eq!(e[0]["shots"][1]["speed"], json!(2.0));
    // …but the split's shot is the plan's analysed shot of item 3 as it was
    assert_eq!(e[1]["shots"][0], shot(3, 4.0));
    assert_eq!(e[1]["description"], json!("an arm"), "a split copies the entry");
    // a plain shot of a new item: name and length from the project, seen across its window
    let s = &e[2]["shots"][0];
    assert_eq!((s["name"].as_str(), s["clip_seconds"].as_f64(), s["from"].as_f64()), (Some("new.mov"), Some(30.0), Some(2.0)));
    assert_eq!(s["seen"], json!([2.0, 4.0, 6.0, 8.0, 10.0]));
    assert_eq!((s["best"].as_f64(), &s["largest"]), (Some(6.0), &Value::Null));
    assert_eq!((e[3].get("place"), &e[3]["event"]), (None, &json!("Expo")));
    let sc = &p.plan["script"];
    assert_eq!(sc["lines"], json!(["A big arm.", "", "New one.", "The dog."]));
    // chips cut to three, by the new numbers; the dog's delivery and say moved with it
    assert_eq!(sc["chips"], json!({"1": ["a", "b", "c"], "2": ["s"], "3": ["n1", "n2", "n3"]}));
    assert_eq!(sc["delivery"], json!({"intro": "excited", "outro": "calm", "1": "warm", "2": "warm", "3": "warm", "4": "amazed"}));
    assert_eq!(sc.get("say"), None, "the dog's say was removed");
    assert_eq!(sc["pronounce"], json!({"Arm": "Ahm"}));
    assert_eq!(sc["templated"], json!(0));
    // two robots then, four now
    assert_eq!(sc["outro"], json!("Four robots, the end."));
    assert_eq!((p.plan["transitions"].as_bool(), p.plan.get("style")), (Some(true), None));
    assert_eq!(p.unlined, ["Small arm"]);
    assert_eq!((p.robots, p.lines, p.silent_scenes), (4, 3, 0));
}

#[test]
fn voice_settings_follow_their_entries() {
    let mut b = base();
    b["script"]["voice"] = json!({"intro": {"temperature": 1.2}, "2": {"speed": 1.1}, "3": {"emphasis": 2.4}});
    b["script"]["voiceDefault"] = json!({"temperature": 0.9});
    // the hall goes, the dog gets its own settings, the arm keeps its own
    let patch = json!({"entries": {"Hall": {"remove": true}, "Dog": {"voice": {"temperature": 1.1, "speed": 1.05}}}});
    let p = apply(&b, &[patch], &Clips::new()).unwrap();
    let sc = &p.plan["script"];
    assert_eq!(sc["voice"], json!({"intro": {"temperature": 1.2}, "1": {"speed": 1.1}, "2": {"temperature": 1.1, "speed": 1.05}}));
    assert_eq!(sc["voiceDefault"], json!({"temperature": 0.9}), "the plan's default stays");
    // null removes an entry's; a misspelt or out-of-range setting is refused, saying where
    let p = apply(&b, &[json!({"entries": {"Arm": {"voice": null}}})], &Clips::new()).unwrap();
    assert_eq!(p.plan["script"]["voice"], json!({"intro": {"temperature": 1.2}, "3": {"emphasis": 2.4}}));
    for (bad, want) in [
        (json!({"entries": {"Arm": {"voice": {"temprature": 1.0}}}}), "temprature"),
        (json!({"entries": {"Arm": {"voice": {"speed": 3}}}}), "speed"),
        (json!({"entries": {"Arm": {"voice": "loud"}}}), "voice settings"),
    ] {
        let e = apply(&b, std::slice::from_ref(&bad), &Clips::new()).unwrap_err().to_string();
        assert!(e.contains(want) && e.contains("Arm"), "{bad}: {e}");
    }
    // a plan without any keeps none
    let p = apply(&base(), &[], &Clips::new()).unwrap();
    assert!(p.plan["script"].get("voice").is_none());
}

#[test]
fn bad_notes_say_what_and_where() {
    let none = Clips::new();
    let cases = [
        (json!({"entries": {"Cat": {"line": "x"}}}), "no entry is labelled \"Cat\""),
        (json!({"entries": {"Arm": {"shots": [{"item": 77}]}}}), "its length is unknown"),
        (json!({"entries": {"Arm": {"shots": [{"item": 2, "plain": [5, 1]}]}}}), "stretch in view"),
        (json!({"entries": {"Arm": {"shotEdits": {"9": {"speed": 2}}}}}), "no shot of item 9"),
        (json!({"entries": {"Arm": {"shotEdits": {"2": {"speed": 99}}}}}), "speed"),
        (json!({"entries": {"Arm": {"unix": "yesterday"}}}), "not a date"),
        (json!({"entries": {"Arm": {"number": 4}}}), "numbers are given"),
        (json!({"entries": {"Arm": {"chips": ["this chip is far too long to fit beside a lower third"]}}}), "chips"),
        (json!({"add": [{"label": "X"}]}), "needs shots"),
        (json!({"add": [{"shots": [{"item": 2}]}]}), "needs a label"),
        (json!({"timezone": "Mars"}), "timezone"),
        (json!([1, 2]), "a JSON object"),
    ];
    for (patch, want) in cases {
        let e = apply(&base(), std::slice::from_ref(&patch), &none).unwrap_err().to_string();
        assert!(e.contains(want), "{patch}: {e}");
    }
    // a base plan that is not one
    assert!(apply(&json!({"entries": 3}), &[], &none).is_err());
    let mut lopsided = base();
    lopsided["script"]["lines"] = json!(["one"]);
    assert!(apply(&lopsided, &[], &none).unwrap_err().to_string().contains("3 entries"));
}

#[test]
fn several_patches_apply_in_turn_and_twins_are_numbered() {
    let mut b = base();
    b["entries"][2]["label"] = json!("Arm");
    let first = json!({"entries": {"Arm #2": {"label": "Arm, again"}}});
    let second = json!({"entries": {"Arm, again": {"line": "Again."}, "Arm": {"line": "Once."}}});
    let p = apply(&b, &[first, second], &Clips::new()).unwrap();
    assert_eq!(p.plan["script"]["lines"], json!(["", "Once.", "Again."]));
    // the twins without a number are ambiguous
    let e = apply(&b, &[json!({"entries": {"Arm": {"line": "?"}}})], &Clips::new()).unwrap_err().to_string();
    assert!(e.contains("2 entries are labelled"), "{e}");
}

#[test]
fn counts_are_rewritten_in_the_same_style() {
    assert_eq!(count_words("Forty-three robots.", 43, 48), "Forty-eight robots.");
    assert_eq!(count_words("all forty three of them", 43, 48), "all forty eight of them");
    assert_eq!(count_words("43 robots, 143 photos", 43, 48), "48 robots, 143 photos");
    assert_eq!(count_words("{Count} robots, {count} in all", 0, 21), "Twenty-one robots, twenty-one in all");
    assert_eq!(count_words("Seven robots", 7, 7), "Seven robots");
    assert_eq!(count_words("Seventy robots", 7, 9), "Seventy robots", "not inside a longer word");
    assert_eq!(words(100, true), "One hundred");
    assert_eq!(words(118, false), "one hundred and eighteen");
    assert_eq!(words(40, false), "forty");
    assert_eq!(words(12_345, false), "12345");
}

#[test]
fn dates_read_in_the_patch_timezone_or_their_own() {
    assert_eq!(date("2022-01-23 13:40:35", 4 * 3600), Some(1_642_930_835));
    assert_eq!(date("2022-01-23T09:40:35Z", 4 * 3600), Some(1_642_930_835));
    assert_eq!(date("2022-01-23 13:40:35+04:00", 0), Some(1_642_930_835));
    assert_eq!(date("2022-01-23 05:40:35-04:00", 0), Some(1_642_930_835));
    assert_eq!(date("2024-02-29", 0), Some(1_709_164_800));
    assert_eq!(date("1969-12-31 23:59:59", 0), Some(-1));
    for bad in ["", "2022-13-01", "2022-01-23 25:00", "2022/01/23", "2022-01-23 10:00:00+99:00", "x-y-z", "2022-01-23 10:00:00:00"] {
        assert_eq!(date(bad, 0), None, "{bad}");
    }
    assert_eq!(offset("+0530"), Some(19_800));
    assert_eq!(offset("UTC"), Some(0));
    assert_eq!(offset("4"), None);
}

/// The dates chrono computes are the ones the hand-rolled conversion gave: the ends of the years a
/// patch may use, a leap second, and a day past the month's end running on into the next month.
#[test]
fn dates_through_chrono_keep_their_old_values() {
    assert_eq!(days_from_civil(1970, 1, 1), Some(0));
    assert_eq!(days_from_civil(2000, 3, 1), Some(11_017));
    assert_eq!(date("0001-01-01", 0), Some(-62_135_596_800));
    assert_eq!(date("9999-12-31 23:59:59", 0), Some(253_402_300_799));
    assert_eq!(date("2016-12-31 23:59:60", 0), date("2017-01-01", 0));
    assert_eq!(date("2022-02-31", 0), date("2022-03-03", 0));
    assert_eq!(date("2024-02-30 12:00", 0), date("2024-03-01 12:00", 0));
    assert_eq!(date("0000-01-01", 0), None);
    assert_eq!(date("10000-01-01", 0), None);
}

#[test]
fn places_from_a_locations_file_fill_entries_without_one() {
    let mut plan = apply(&base(), &[], &Clips::new()).unwrap().plan;
    // the arm's second shot and the dog's shot were read by `photos locations`
    let file = json!([
        {"fileName": "C3.MP4", "place": "Abu Dhabi", "estimated": false},
        {"fileName": "c4.mov", "place": "Sydney, Australia", "estimated": true},
        {"fileName": "c9.mp4"}
    ]);
    let places = crate::places::Places::from_json(&file).unwrap();
    let filled = fill_places(&mut plan, &places);
    // the dog keeps the place its entry gives; the hall's clip has none
    assert_eq!(filled, ["Arm"]);
    assert_eq!(plan["entries"][1]["place"], json!("Abu Dhabi"));
    assert_eq!(plan["entries"][2]["place"], json!("Dubai"));
    assert_eq!(plan["entries"][0].get("place"), None);
    assert!(fill_places(&mut json!({"entries": 3}), &places).is_empty());
}

#[test]
fn another_language_needs_every_spoken_line() {
    let p = apply(&base(), &[], &Clips::new()).unwrap().plan;
    let lines = json!({"intro": "ഹായ്", "outro": "നന്ദി", "lines": {"Arm": "കൈ", "Dog": "നായ", "Ghost": "?"}});
    let (ml, notes) = with_lines(&p, &lines, "ml").unwrap();
    assert_eq!(ml["script"]["lines"], json!(["", "കൈ", "നായ"]));
    assert_eq!((ml["language"].as_str(), &ml["script"]["pronounce"], ml["script"].get("say")), (Some("ml"), &json!({}), None));
    assert_eq!(ml["script"]["chips"], p["script"]["chips"], "chips stay as written");
    assert_eq!(notes.len(), 1, "the line for an unknown entry is noted");
    let short = json!({"intro": "a", "outro": "b", "lines": {"Arm": "കൈ"}});
    assert!(with_lines(&p, &short, "ml").unwrap_err().to_string().contains("Dog"));
    let no_intro = json!({"outro": "b", "lines": {"Arm": "a", "Dog": "b"}});
    assert!(with_lines(&p, &no_intro, "ml").is_err());
    assert!(with_lines(&p, &lines, "m l").is_err());
    assert!(with_lines(&p, &json!({"lines": 3}), "ml").is_err());
}

#[test]
fn notes_add_photo_shots_and_entries_of_photos() {
    // item 20: a photo the project holds (a still); 21: one the plan analysed already
    let t = 254_016_000_000i64;
    let clips = clips_from_inspect(&json!({"root": {"children": [
        {"item": 20, "name": "IMG_4512.HEIC", "type": "Still Image", "duration": 5 * t},
        {"item": 9, "name": "new.mov", "type": "Movie", "duration": 30 * t}
    ]}}));
    assert_eq!(clips.get(&20).map(|c| c.still), Some(true));
    let mut plan = base();
    plan["entries"][2]["shots"]
        .as_array_mut()
        .unwrap()
        .push(json!({"item": 21, "name": "dog.jpg", "clip_seconds": 20.0, "best": 0.0, "seen": [], "unseen": [], "largest": null, "photo": true, "subject": [0.1, 0.2, 0.3, 0.4]}));
    let patch = json!({
        "entries": {
            // the arm: its video, then a photo of it held 4 s moving in
            "Arm": {"shots": [{"item": 2}, {"item": 20, "hold": 4.0, "move": "in"}]},
            // the dog's analysed photo: a pan across it, held 3 s
            "Dog": {"shotEdits": {"21": {"hold": 3.0, "move": "pan"}}}
        },
        // a robot only photographed
        "add": [{"label": "Toy robot", "unix": 250, "shots": [{"item": 20, "photo": true, "fit": "full"}], "line": "A toy."}]
    });
    let p = apply(&plan, &[patch], &clips).unwrap();
    let e = p.plan["entries"].as_array().unwrap();
    let labels: Vec<&str> = e.iter().map(|e| e["label"].as_str().unwrap()).collect();
    assert_eq!(labels, ["Hall", "Arm", "Toy robot", "Dog"]);
    let arm = e[1]["shots"].as_array().unwrap();
    assert_eq!(arm[0], shot(2, 3.0), "its video as analysed");
    assert_eq!(
        arm[1],
        json!({"item": 20, "name": "IMG_4512.HEIC", "clip_seconds": crate::showcase::PHOTO_REACH, "best": 0.0, "seen": [], "unseen": [], "largest": null, "photo": true, "hold": 4.0, "move": "in"})
    );
    assert_eq!(e[2]["shots"][0]["fit"], "full");
    assert_eq!(e[2]["shots"][0]["photo"], true);
    let dog = &e[3]["shots"][1];
    assert_eq!((dog["hold"].as_f64(), dog["move"].as_str(), dog["subject"].clone()), (Some(3.0), Some("pan"), json!([0.1, 0.2, 0.3, 0.4])), "{dog}");
    // the plan as patched reads back into entries, and the photo controls check out
    let entries: Vec<Entry> = serde_json::from_value(p.plan["entries"].clone()).unwrap();
    let edits = crate::shot_edit::check(&entries).unwrap();
    assert_eq!(edits[1][1].fixed(), Some((0.0, 4.0)));
    // a photo has no window; a video is not a photo
    for (spec, says) in [
        (json!({"item": 20, "from": 1.0}), "no `from`"),
        (json!({"item": 20, "photo": true, "plain": true}), "no `plain`"),
        (json!({"item": 9, "photo": true}), "a video"),
    ] {
        let err = apply(&plan, &[json!({"entries": {"Arm": {"shots": [spec]}}})], &clips).unwrap_err().to_string();
        assert!(err.contains(says), "{says}: {err}");
    }
    // without the project, a photo said to be one is taken as one
    let p = apply(&plan, &[json!({"entries": {"Arm": {"shots": [{"item": 77, "photo": true, "name": "x.jpg"}]}}})], &Clips::new()).unwrap();
    assert_eq!(p.plan["entries"][1]["shots"][0]["clip_seconds"], json!(crate::showcase::PHOTO_REACH));
}

#[test]
fn project_clips_and_the_outro_logo() {
    let inv = json!({"bins": [{"items": [{"item": 5, "name": "a.mov", "duration": 254_016_000_000i64 * 3}, {"item": 6, "duration": -1}]}], "item": 7});
    let c = clips_from_inspect(&inv);
    assert_eq!(c.get(&5), Some(&ClipInfo { name: "a.mov".into(), seconds: 3.0, still: false }));
    assert_eq!(c.len(), 1);
    // nesting deeper than the walk goes is left alone, not a stack overflow
    let mut deep = json!({"item": 1, "duration": 1});
    for _ in 0..10_000 {
        deep = json!([deep]);
    }
    assert!(clips_from_inspect(&deep).is_empty());
    if let Some(home) = std::env::var_os("HOME").filter(|h| !h.is_empty()) {
        let p = apply(&base(), &[json!({"outro": {"logo": "~/logo.png", "channel": "C"}})], &Clips::new()).unwrap();
        assert_eq!(p.plan["outro"]["logo"].as_str(), Some(std::path::Path::new(&home).join("logo.png").to_string_lossy().as_ref()));
    }
}
