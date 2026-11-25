//! `script.timeline` with the built-in voice (fast and deterministic) and with estimates: shots,
//! chapter starts and subtitles from the markup, laid out to add up, placed in one undo step,
//! and hostile scripts and parameters refused.
use serde_json::{Value, json};

use super::*;
use crate::Session;

const SCRIPT: &str = "# Why the sky is blue\n\
[shot: sunlight and a prism]\n\
Sunlight looks white. It holds every colour of the rainbow.\n\
\n\
Air scatters the short blue waves the most. [shot: scattered blue light] So blue comes at us from every side.\n\
\n\
# Sunsets\n\
At sunset the light crosses far more air. [pause 1s] The blue is scattered away, and red is left.\n";

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("fc-explainer-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn ticks(v: &Value) -> i64 {
    v.as_i64().unwrap()
}

#[test]
fn the_markup_is_read_into_chapters_paragraphs_and_hints() {
    let c = parse(SCRIPT, false).unwrap();
    assert_eq!(c.len(), 2);
    assert_eq!(c[0].title, "Why the sky is blue");
    assert_eq!(c[0].paragraphs.len(), 2);
    // a hint on its own line before its paragraph, and one inline
    assert_eq!(c[0].paragraphs[0].hints, vec!["sunlight and a prism"]);
    assert_eq!(c[0].paragraphs[1].hints, vec!["scattered blue light"]);
    assert!(!c[0].paragraphs[1].text.contains("shot"), "{}", c[0].paragraphs[1].text);
    // the voice's own pause marker stays for the voice
    assert!(c[1].paragraphs[0].text.contains("[pause 1s]"));
    // text before the first heading is an untitled chapter; `#tag` is text
    let c = parse("Welcome to the #explainer.\n\n# One\nFirst words here.\n", false).unwrap();
    assert_eq!((c.len(), c[0].title.as_str()), (2, ""));
    assert!(c[0].paragraphs[0].text.contains("#explainer"));
    // a hint line inside a paragraph belongs to it
    let c = parse("# A\nOne line.\n[shot: a map]\nTwo line.\n", false).unwrap();
    assert_eq!(c[0].paragraphs.len(), 1);
    assert_eq!(c[0].paragraphs[0].hints, vec!["a map"]);
    assert_eq!(c[0].paragraphs[0].text, "One line. Two line.");
    // delivery tags are read in Malayalam scripts only
    assert!(parse("# A\n[warm] Hello there.\n", false).is_err());
    assert!(parse("# A\n[warm] Hello there.\n", true).is_ok());
}

#[test]
fn sentences_split_at_their_ends_but_not_at_abbreviations() {
    assert_eq!(sentences("It flew. Then it fell! Why? Nobody knows…"), vec!["It flew.", "Then it fell!", "Why?", "Nobody knows…"]);
    assert_eq!(sentences("Dr. Wright flew in 1903. It took 12 s."), vec!["Dr. Wright flew in 1903.", "It took 12 s."]);
    assert_eq!(sentences("Water, e.g. rain, falls. Then 3.5 cm of it."), vec!["Water, e.g. rain, falls.", "Then 3.5 cm of it."]);
    assert_eq!(sentences("He said \"stop.\" And it did."), vec!["He said \"stop.\"", "And it did."]);
    // a lone pause marker stays with a sentence
    assert_eq!(sentences("One. [pause 1s] Two."), vec!["One.", "[pause 1s] Two."]);
    assert_eq!(sentences("no end here"), vec!["no end here"]);
}

#[test]
fn subtitle_blocks_fit_and_break_into_two_lines() {
    let b = blocks("Air scatters the short blue waves the most, so blue comes at us from every side of the sky above us all day long.", 40);
    assert!(b.len() >= 3, "{b:?}");
    assert!(b.iter().all(|x| x.chars().count() <= 40), "{b:?}");
    assert_eq!(b.join(" "), "Air scatters the short blue waves the most, so blue comes at us from every side of the sky above us all day long.");
    let two = lines("Air scatters the short blue waves the most", 40);
    assert_eq!(two.lines().count(), 2, "{two}");
}

#[test]
fn a_two_chapter_script_becomes_shots_chapters_and_subtitles_that_add_up() {
    let dir = tmp("two");
    let mut s = Session::default();
    let undo0 = s.history.undo.len();
    let r = s
        .execute(
            CMD,
            json!({"text": SCRIPT, "voice": "basic-female",
                "place": {"sequence": "new", "name": "Sky", "width": 640, "height": 360, "fps": 30, "dir": dir.to_string_lossy()}}),
        )
        .unwrap();
    // three paragraphs, three shots; two chapters
    assert_eq!(r["counts"]["chapters"], 2, "{r:#}");
    assert_eq!(r["counts"]["shots"], 3);
    assert_eq!(r["counts"]["sentences"], 6);
    let shots = r["shots"].as_array().unwrap();
    let chapters = r["chapters"].as_array().unwrap();
    assert_eq!(shots[0]["id"], "shot-01");
    assert_eq!(shots[0]["hints"][0], "sunlight and a prism");
    assert_eq!(shots[1]["hints"][0], "scattered blue light");
    assert_eq!((shots[0]["chapter"].as_u64(), shots[2]["chapter"].as_u64()), (Some(0), Some(1)));
    // chapter starts: the first shot of each chapter
    assert_eq!(ticks(&chapters[0]["start"]), 0);
    assert_eq!(chapters[1]["start"], shots[2]["start"]);
    assert_eq!(chapters[1]["title"], "Sunsets");
    // the shots tile the timeline, on the frame grid, and add up to the narration with its gaps
    let total = ticks(&r["duration"]);
    let frame = FrameRate::FPS_30.frame_duration().0;
    let mut sum = 0;
    for (i, sh) in shots.iter().enumerate() {
        let (a, e) = (ticks(&sh["start"]), ticks(&sh["end"]));
        assert_eq!(a % frame, 0, "shot {i} starts off the frame grid");
        assert_eq!(e - a, ticks(&sh["duration"]));
        assert!(ticks(&sh["speech"]) < e - a, "speech fits in its shot");
        if let Some(next) = shots.get(i + 1) {
            assert_eq!(e, ticks(&next["start"]), "no gap or overlap between shots {i} and {}", i + 1);
        }
        sum += e - a;
    }
    assert_eq!(sum, total);
    assert_eq!(ticks(&shots.last().unwrap()["end"]), total);
    assert_eq!(ticks(&r["narration"]["speech"]) + ticks(&r["narration"]["gaps"]), total);
    // subtitles: in order, never overlapping, inside the timeline, every word of the script shown
    let subs = r["subtitles"].as_array().unwrap();
    assert!(subs.len() >= 6, "{}", subs.len());
    let mut prev = 0;
    for b in subs {
        let (a, e) = (ticks(&b["start"]), ticks(&b["end"]));
        assert!(a >= prev && e > a && e <= total, "{b}");
        assert!(!b["words"].as_array().unwrap().is_empty());
        prev = e;
    }
    let shown: Vec<String> = subs.iter().flat_map(|b| b["text"].as_str().unwrap().split_whitespace().map(str::to_string).collect::<Vec<_>>()).collect();
    let script_words: Vec<String> =
        shots.iter().flat_map(|sh| sh["text"].as_str().unwrap().split_whitespace().map(str::to_string).collect::<Vec<_>>()).collect();
    assert_eq!(shown, script_words);
    assert!(!shown.iter().any(|w| w.contains("pause") || w.contains('[')), "markers are not shown");
    // markers: a chapter marker at each chapter start, a shot marker at each shot start
    let m = r["markers"].as_array().unwrap();
    assert_eq!(m.iter().filter(|x| x["kind"] == "chapter").count(), 2);
    assert_eq!(m.iter().filter(|x| x["kind"] == "shot").count(), 3);
    // placed: a new sequence with one narration clip a shot, the captions and the markers
    let q = s.active_sequence().unwrap();
    assert_eq!((q.settings.width, q.settings.height), (640, 360));
    let narr = &q.audio_tracks[0].items;
    assert_eq!(narr.len(), 3);
    for (clip, sh) in narr.iter().zip(shots) {
        assert_eq!(clip.start.0, ticks(&sh["start"]));
        assert_eq!(clip.duration.0, ticks(&sh["speech"]), "the clip is the shot's speech");
        let n = s.project.narrations.get(&clip.item).expect("a narration item tts.edit can edit");
        assert_eq!(n.voice, "basic-female");
    }
    assert_eq!(q.caption_tracks.len(), 1);
    assert_eq!(q.caption_tracks[0].captions.len(), subs.len());
    assert!(q.caption_tracks[0].captions.iter().all(|c| !c.words.is_empty()));
    let chapter_markers: Vec<&filmcraft_project::Marker> = q.markers.iter().filter(|m| m.kind == MarkerKind::Chapter).collect();
    assert_eq!(chapter_markers.len(), 2);
    assert_eq!(chapter_markers[1].name, "Sunsets");
    assert_eq!(q.markers.iter().filter(|m| shot_index(&m.name).is_some()).count(), 3);
    assert_eq!(q.markers.iter().find(|m| m.name == "shot-02").unwrap().comment, "scattered blue light");
    // one undo step takes it all back
    assert_eq!(s.history.undo.len(), undo0 + 1);
    s.undo().unwrap();
    assert!(s.project.sequences().next().is_none(), "the new sequence went with the undo");
    // the voice speaks the same again: the same timeline
    let again = s.execute(CMD, json!({"text": SCRIPT, "voice": "basic-female", "fps": 30})).unwrap();
    assert_eq!(again["duration"], r["duration"]);
    assert_eq!(again["placed"], Value::Null);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn out_writes_one_json_a_shot_for_its_composition() {
    let dir = tmp("out");
    let mut s = Session::default();
    let r = s.execute(CMD, json!({"text": SCRIPT, "estimate": true, "fps": 25, "out": dir.to_string_lossy()})).unwrap();
    assert_eq!(r["estimated"], true);
    assert_eq!(r["voice"], Value::Null);
    let files = r["files"]["shots"].as_array().unwrap();
    assert_eq!(files.len(), 3);
    let shot: Value = serde_json::from_slice(&std::fs::read(files[1].as_str().unwrap()).unwrap()).unwrap();
    assert_eq!(shot["format"], SHOT_FORMAT);
    assert_eq!(shot["id"], "shot-02");
    assert_eq!(shot["chapter"]["title"], "Why the sky is blue");
    assert_eq!(shot["hints"][0], "scattered blue light");
    assert_eq!(shot["fps"], 25.0);
    assert_eq!(shot["frames"], r["shots"][1]["frames"]);
    // times are relative to the shot: its first sentence starts at frame 0
    assert_eq!(shot["sentences"][0]["from"], 0);
    assert!(shot["subtitles"].as_array().unwrap().iter().all(|b| b["to"].as_i64().unwrap() <= shot["frames"].as_i64().unwrap()));
    let whole: Value = serde_json::from_slice(&std::fs::read(r["files"]["timeline"].as_str().unwrap()).unwrap()).unwrap();
    assert_eq!(whole["format"], TIMELINE_FORMAT);
    assert_eq!(whole["duration"], r["duration"]);
    // a folder with files in it is refused: nothing is overwritten
    let e = s.execute(CMD, json!({"text": SCRIPT, "estimate": true, "out": dir.to_string_lossy()})).unwrap_err().to_string();
    assert!(e.contains("not empty"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_long_paragraph_is_split_into_sentence_groups() {
    let mut s = Session::default();
    let para = "One two three four five six seven eight nine ten. ".repeat(6);
    let r = s.execute(CMD, json!({"text": format!("# Long\n{para}\n"), "estimate": true, "maxShotSeconds": 9})).unwrap();
    // six sentences of 4 s each: two to a shot
    assert_eq!(r["counts"]["shots"], 3, "{r:#}");
    for sh in r["shots"].as_array().unwrap() {
        assert_eq!(sh["parts"], 3);
        assert_eq!(sh["sentences"].as_array().unwrap().len(), 2);
    }
    assert_eq!(r["chapters"][0]["shots"], json!([0, 2]));
}

#[test]
fn an_estimate_places_markers_and_captions_but_no_narration() {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"width": 640, "height": 360, "fps": 30})).unwrap();
    let r = s.execute(CMD, json!({"text": SCRIPT, "estimate": true, "place": {"sequence": "active", "seconds": 2.0}})).unwrap();
    assert!(r["placed"]["narration"]["skipped"].is_string(), "{r:#}");
    let q = s.active_sequence().unwrap();
    assert!(q.audio_tracks.iter().all(|t| t.items.is_empty()));
    // everything moved by `at`
    let first = q.markers.iter().map(|m| m.start).min().unwrap();
    assert_eq!(first, Tick::from_seconds_f64(2.0));
    assert!(!q.caption_tracks[0].captions.is_empty());
}

#[test]
fn hostile_scripts_and_parameters_are_refused_without_a_panic() {
    let mut s = Session::default();
    let huge = format!("# A\n{}\n", "word ".repeat(20_000));
    let many_chapters: String = (0..70).map(|i| format!("# C{i}\nWords here.\n\n")).collect();
    let many_sentences = format!("# A\n{}\n", "Go. ".repeat(700));
    let cases: Vec<(Value, &str)> = vec![
        (json!({"text": huge}), "bytes"),
        (json!({"text": "# Empty\n\n# Full\nWords.\n"}), "no paragraphs"),
        (json!({"text": "# Last\n"}), "no paragraphs"),
        (json!({"text": "#\nWords.\n"}), "needs a title"),
        (json!({"text": "# A\nWords [shot: never closed\n"}), "not closed"),
        (json!({"text": "# A\nWords [shot:   ] here.\n"}), "empty"),
        (json!({"text": "# A\nWords [aside] here.\n"}), "unknown mark"),
        (json!({"text": "# A\n[warm] Words here.\n"}), "not read here"),
        (json!({"text": "# A\nWords ] here.\n"}), "unknown mark"),
        (json!({"text": "# A\nWords.\n\n[shot: a map]\n# B\nMore.\n"}), "no paragraph after it"),
        (json!({"text": "# A\n[pause 1s]\n"}), "no words"),
        (json!({"text": "# A [shot: x]\nWords.\n"}), "no marks"),
        (json!({"text": "Bad\u{0}text"}), "NUL"),
        (json!({"text": many_chapters}), "chapters"),
        (json!({"text": many_sentences, "estimate": true}), "sentences"),
        (json!({"text": ""}), "says nothing"),
        (json!({"text": "   \n\n  "}), "says nothing"),
        (json!({"text": 42}), "string"),
        (json!({}), "need the script"),
        (json!({"text": "# A\nWords.", "path": "/x"}), "not both"),
        (json!({"text": "# A\nWords.", "colour": "red"}), "unknown parameter"),
        (json!({"text": "# A\nWords.", "estimate": "yes"}), "true or false"),
        (json!({"text": "# A\nWords.", "gaps": {"shot": -1}}), "gaps.shot"),
        (json!({"text": "# A\nWords.", "gaps": {"lunch": 1}}), "not a gap"),
        (json!({"text": "# A\nWords.", "gaps": [1]}), "gaps"),
        (json!({"text": "# A\nWords.", "maxShotSeconds": 1e300}), "maxShotSeconds"),
        (json!({"text": "# A\nWords.", "captionChars": 3}), "captionChars"),
        (json!({"text": "# A\nWords.", "fps": 0}), "fps"),
        (json!({"text": "# A\nWords.", "out": ""}), "folder"),
        (json!({"text": "# A\nWords.", "place": "yes"}), "place"),
        (json!({"text": "# A\nWords.", "place": {"sequence": "other"}}), "new"),
        (json!({"text": "# A\nWords.", "place": {"sequence": "active"}}), "open sequence"),
        (json!({"text": "# A\nWords.", "place": {"width": 1}}), "pixels"),
        (json!({"text": "# A\nWords.", "place": {"at": -5}}), "place.at"),
        (json!({"text": "# A\nWords.", "place": {"track": "A1"}}), "active sequence"),
        (json!({"text": "# A\nWords.", "place": {"stuff": 1}}), "placement setting"),
        (json!({"text": "# A\nWords.", "voice": "nobody"}), "unknown voice"),
        (json!({"text": "# A\nWords.", "language": "xx"}), "language"),
        (json!({"path": "/no/such/script.md"}), "script"),
        (json!([1, 2]), "object"),
    ];
    for (p, want) in cases {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute(CMD, p.clone())));
        let e = r.unwrap_or_else(|_| panic!("panicked on {p}")).err().unwrap_or_else(|| panic!("accepted {p}")).to_string();
        assert!(e.contains(want), "{p}: {e}");
    }
    // a script made of the limits still works
    let ok = format!("# A\n{}\n", "Word word word. ".repeat(40));
    assert!(s.execute(CMD, json!({"text": ok, "estimate": true})).is_ok());
}
