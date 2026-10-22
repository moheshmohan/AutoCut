use std::sync::Arc;

use serde_json::json;

use crate::Session;
use filmcraft_project::{ItemId, ItemKind, Transcript, Word};
use filmcraft_speech::FixedTranscriber;
use filmcraft_time::{Tick, TimeRange};

/// The demo project with a fake recogniser for the first A1 clip's media: a line said twice (the
/// first take with a filler), a pause, and a closing line by a second speaker. Word times are
/// media time, from the clip's source In.
fn session() -> (Session, ItemId, Tick) {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    let q = s.active_sequence().unwrap();
    let a = &q.audio_tracks[0].items[0];
    let (item, sin) = (a.item, a.source_in);
    let sec = |x: f64| sin + Tick::from_seconds_f64(x);
    let mut t = Transcript { language: "en".into(), ..Default::default() };
    for (text, a, b, sp) in [
        ("Today", 0.20, 0.50, 0),
        ("we", 0.55, 0.70, 0),
        ("um", 0.80, 1.00, 0),
        ("test", 1.10, 1.40, 0),
        ("it.", 1.45, 1.70, 0),
        ("Today", 2.60, 2.90, 0),
        ("we", 2.95, 3.10, 0),
        ("test", 3.15, 3.45, 0),
        ("it.", 3.50, 3.80, 0),
        ("It", 5.00, 5.20, 1),
        ("works.", 5.25, 5.70, 1),
    ] {
        let mut w = Word::new(text, sec(a), sec(b));
        w.speaker = Some(sp);
        w.confidence = 0.9;
        t.words.push(w);
    }
    t.normalize();
    s.transcriber = Some(Arc::new(FixedTranscriber { transcript: t, id: "fixed".into() }));
    (s, item, sin)
}

fn understand(s: &mut Session, item: ItemId) -> serde_json::Value {
    s.execute("media.understandSpeech", json!({"items": [item.0], "wait": true, "faces": false, "diarize": false})).unwrap()
}

#[test]
fn understanding_finds_the_retake_and_is_one_undo_step() {
    let (mut s, item, _) = session();
    let undo0 = s.history.undo.len();
    let r = understand(&mut s, item);
    let a = &r["speech"]["items"][0];
    assert_eq!(a["item"], item.0);
    assert_eq!(a["retakes"].as_array().unwrap().len(), 1, "{a}");
    assert_eq!(a["retakes"][0]["takes"], json!([0, 1]));
    assert_eq!(a["retakes"][0]["keep"], 1, "the take without the filler");
    assert_eq!(a["droppedTakes"], json!([0]));
    assert!(a["fillers"].as_array().unwrap().iter().any(|f| f["text"] == "um"));
    assert_eq!(a["segments"][0]["words"].as_array().unwrap().len(), 5);
    assert_eq!(a["source"], "fixed");
    assert!(s.project.speech.contains_key(&item) && s.project.transcripts.contains_key(&item));
    // stored, inspectable, and the job reports it
    let i = s.execute("media.inspectSpeech", json!({"items": [item.0]})).unwrap();
    assert_eq!(i["items"][0]["droppedTakes"], json!([0]));
    let jobs = s.execute("jobs.list", json!({})).unwrap();
    assert!(jobs.as_array().unwrap().iter().any(|j| j["speech"]["items"][0]["item"] == item.0));
    // one undo step
    assert_eq!(s.history.undo.len(), undo0 + 1);
    s.execute("edit.undo", json!({})).unwrap();
    assert!(s.project.speech.is_empty() && s.project.transcripts.is_empty());
}

#[test]
fn rough_cut_keeps_whole_words_without_gaps_and_undoes() {
    let (mut s, item, _) = session();
    understand(&mut s, item);
    let seqs0 = s.project.items.values().filter(|i| matches!(i.kind, ItemKind::Sequence(_))).count();
    let r = s.execute("sequence.roughCut", json!({"items": [item.0]})).unwrap();
    assert_eq!(r["removed"]["retakes"], 1, "{r}");
    let q = s.active_sequence().unwrap();
    assert_eq!(s.state.active_sequence.map(|i| i.0), r["sequence"].as_u64());
    let t = s.project.transcripts[&item].clone();
    // A1: back to back from zero, every cut between words, the dropped take and filler gone
    let clips = &q.audio_tracks[0].items;
    assert!(!clips.is_empty());
    let mut at = Tick::ZERO;
    for c in clips {
        assert_eq!(c.start, at, "no gaps");
        at = c.end();
        let media = TimeRange::from_bounds(c.source_in, c.source_in + c.duration);
        for w in &t.words {
            for b in [media.start, media.end()] {
                assert!(!(w.start < b && b < w.end), "cut at {b:?} inside {:?}", w.text);
            }
        }
    }
    // no media is heard twice
    assert!(clips.windows(2).all(|w| w[0].source_in + w[0].duration <= w[1].source_in), "overlapping source ranges");
    let heard = |w: &Word| clips.iter().any(|c| c.source_in <= w.start && w.end <= c.source_in + c.duration);
    let kept: Vec<&str> = t.words.iter().filter(|w| heard(w)).map(|w| w.text.as_str()).collect();
    assert_eq!(kept, ["Today", "we", "test", "it.", "It", "works."]);
    // video follows the audio, linked
    assert_eq!(q.video_tracks[0].items.len(), clips.len());
    assert!(q.video_tracks[0].items.iter().zip(clips).all(|(v, a)| v.start == a.start && v.duration == a.duration && v.link.is_some() && v.link == a.link));
    // one undo step takes the whole sequence away
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(s.project.items.values().filter(|i| matches!(i.kind, ItemKind::Sequence(_))).count(), seqs0);
    // keeping everything but the pauses: the retake stays
    let r = s.execute("sequence.roughCut", json!({"items": [item.0], "removeRetakes": false, "removeFillers": false})).unwrap();
    assert_eq!(r["removed"]["retakes"], 0);
    assert!(r["removed"]["fillers"] == 0 && r["removed"]["words"] == 0, "{r}");
}

#[test]
fn remove_retakes_cuts_the_dropped_take_from_the_sequence() {
    let (mut s, item, _) = session();
    understand(&mut s, item);
    let words = |s: &mut Session| -> Vec<String> {
        let r = s.execute("transcript.inspect", json!({})).unwrap();
        r["words"].as_array().unwrap().iter().filter(|w| w["item"] == item.0).map(|w| w["text"].as_str().unwrap().to_string()).collect()
    };
    let before = words(&mut s);
    assert_eq!(before.iter().filter(|w| *w == "Today").count(), 2, "{before:?}");
    let r = s.execute("transcript.removeRetakes", json!({})).unwrap();
    assert_eq!(r["takes"], 1, "{r}");
    let after = words(&mut s);
    assert_eq!(after.iter().filter(|w| *w == "Today").count(), 1, "{after:?}");
    assert!(!after.contains(&"um".to_string()), "the filler was in the dropped take");
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(words(&mut s), before);
}

#[test]
fn speakers_label_clips() {
    let (mut s, item, _) = session();
    // two speakers in the transcript (the fake recogniser's labels)
    s.execute("transcript.generate", json!({"items": [item.0], "diarize": false})).unwrap();
    let r = s.execute("sequence.speakerTrack", json!({})).unwrap();
    assert!(!r["clips"].as_array().unwrap().is_empty(), "{r}");
    assert_eq!(r["clips"][0]["speaker"], "Speaker 1");
}

#[test]
fn hostile_parameters_are_errors() {
    let (mut s, item, _) = session();
    for p in [
        json!({"items": [item.0], "language": 5}),
        json!({"items": [item.0], "language": "not a language at all"}),
        json!({"items": [item.0], "maxSpeakers": 0}),
        json!({"items": [item.0], "maxSpeakers": 1e9}),
        json!({"items": [item.0], "minPause": -1}),
        json!({"items": [item.0], "minPause": "x"}),
        json!({"items": [item.0], "diarize": "yes"}),
        json!({"items": [item.0], "fillers": "um"}),
        json!({"items": [999_999]}),
    ] {
        assert!(s.execute("media.understandSpeech", p.clone()).is_err(), "{p}");
    }
    assert!(s.execute("sequence.roughCut", json!({"items": [item.0]})).is_err(), "not understood yet");
    understand(&mut s, item);
    for p in [
        json!({"items": [item.0], "minPause": f64::MAX}),
        json!({"items": [item.0], "keepPause": -3}),
        json!({"items": [item.0], "removePauses": 1}),
        json!({"items": []}),
    ] {
        let _ = s.execute("sequence.roughCut", p.clone());
    }
    assert!(s.execute("sequence.roughCut", json!({"items": [item.0], "keepPause": -3})).is_err());
    // a clip claiming to last for years is refused before any audio is read
    if let Some(ItemKind::Media(m)) = Arc::make_mut(&mut s.project).items.get_mut(&item).map(|i| &mut i.kind) {
        m.info.duration = Tick::from_seconds_f64(1e9);
    }
    let e = s.execute("media.understandSpeech", json!({"items": [item.0], "wait": true})).unwrap_err().to_string();
    assert!(e.contains("at most"), "{e}");
}

#[test]
fn empty_audio_and_no_words() {
    let (mut s, item, _) = session();
    // a recogniser that hears nothing
    s.transcriber = Some(Arc::new(FixedTranscriber { transcript: Transcript::default(), id: "silent".into() }));
    let r = understand(&mut s, item);
    let a = &r["speech"]["items"][0];
    assert!(a["segments"].as_array().unwrap().is_empty());
    assert!(a["retakes"].as_array().unwrap().is_empty());
    assert!(s.execute("sequence.roughCut", json!({"items": [item.0]})).is_err(), "nothing to keep");
}
