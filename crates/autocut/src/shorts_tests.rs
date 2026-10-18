use super::*;
use crate::SessionExecutor;
use crate::highlight::Word;
use crate::llm::FnModel;
use filmcraft_engine::Session;
use filmcraft_time::TICKS_PER_SECOND;

/// Speech: each sentence's words 0.4 s apart (0.3 s spoken), sentences 0.5 s apart, starting at
/// `t0`.
fn speech(t0: f64, sentences: &[&str]) -> Vec<Word> {
    let mut out = Vec::new();
    let mut t = t0;
    for s in sentences {
        for w in s.split_whitespace() {
            out.push(Word { i: out.len() as u64, start: t, end: t + 0.3, text: w.to_string() });
            t += 0.4;
        }
        t += 0.5;
    }
    out
}

const FILLER: &str = "and so yeah we kind of went there and it was fine I guess.";
const HOOK: &str = "Why do 90 percent of rockets fail on their first launch?";
const BODY: &str = "The engines run at 3000 degrees and a single valve sticking for 40 milliseconds destroys the vehicle.";
const PAYOFF: &str = "Engineers now test every valve 500 times before flight, and failures dropped by half.";

/// A long talk: filler, a strong passage, more filler, another good passage, filler.
fn talk() -> Vec<&'static str> {
    let mut v = vec![FILLER; 6];
    v.extend([HOOK, BODY, PAYOFF]);
    v.extend(vec![FILLER; 8]);
    v.extend([
        "You can build a weather station for 30 dollars this weekend.",
        "It needs 3 sensors, a 5 dollar board and 20 lines of code.",
        "Mine has logged 2 years of rain without a single reboot.",
    ]);
    v.extend(vec![FILLER; 5]);
    v
}

#[test]
fn cuts_sentences_at_punctuation_pauses_and_length() {
    let mut words = speech(0.0, &["One two three.", "Four five six!", "Seven eight nine?"]);
    // a long pause without punctuation, and a run-on of 100 words
    words.push(Word { i: 99, start: 30.0, end: 30.3, text: "after".into() });
    let run_on: Vec<String> = (0..100).map(|k| format!("w{k}")).collect();
    words.extend(speech(31.0, &[&run_on.join(" ")]));
    let s = sentences(&words);
    assert_eq!(s[0].text, "One two three.");
    assert_eq!(s[2].text, "Seven eight nine?");
    assert!(s.iter().all(|x| x.last - x.first < MAX_SENTENCE_WORDS), "run-on speech is cut");
    assert!(s.iter().any(|x| x.text.starts_with("after")), "a long pause starts a sentence");
    for x in &s {
        assert!(x.start <= x.end);
    }
}

#[test]
fn the_strong_passages_win_and_do_not_overlap() {
    let words = speech(0.0, &talk());
    let s = sentences(&words);
    let c = candidates(&s, 8.0, 30.0, None);
    assert!(!c.is_empty());
    for x in &c {
        let d = x.seconds();
        assert!((8.0..=30.0).contains(&d), "{d}");
        // whole sentences: it starts on a sentence's first word and ends on one's last
        assert!(s.iter().any(|y| y.start == x.start) && s.iter().any(|y| y.end == x.end));
    }
    let chosen = pick(&c, 2);
    assert_eq!(chosen.len(), 2);
    let hooks: Vec<&str> = chosen.iter().map(|x| x.hook.as_str()).collect();
    assert!(hooks.contains(&HOOK), "{hooks:?}");
    assert!(hooks.iter().any(|h| h.starts_with("You can build")), "{hooks:?}");
    let (a, b) = (&chosen[0], &chosen[1]);
    assert!(a.end + MIN_GAP <= b.start || b.end + MIN_GAP <= a.start, "no overlap");
    // more than there is room for: as many as fit
    assert!(pick(&c, 50).len() < 50);
    // a topic pulls its passage up
    let topic = candidates(&s, 8.0, 30.0, Some("weather"));
    assert!(pick(&topic, 1)[0].hook.starts_with("You can build"));
}

#[tokio::test]
async fn hook_ratings_lift_a_passage_and_bad_answers_are_ignored() {
    let words = speech(0.0, &talk());
    let s = sentences(&words);
    let mut c = candidates(&s, 8.0, 30.0, None);
    // the model loves the weather station's opening and hates the rocket's
    let m = FnModel::new("rater", |_, user, _| {
        let n = user.lines().filter(|l| l.contains(". opening:")).count();
        let ratings: Vec<Value> = user
            .lines()
            .filter(|l| l.contains(". opening:"))
            .enumerate()
            .map(|(k, l)| json!({"n": k + 1, "hook": if l.contains("weather station") { 10 } else { 0 }, "title": "Weather \"station\"\n on a budget"}))
            .chain([json!({"n": 0, "hook": 5}), json!({"n": n + 7, "hook": 5}), json!({"n": 1, "hook": "lots"})])
            .collect();
        Ok(json!({"ratings": ratings}))
    });
    let mut notes = Vec::new();
    rate_hooks(&m, &mut c, &mut notes).await;
    assert!(notes.is_empty(), "{notes:?}");
    let best = &pick(&c, 1)[0];
    assert!(best.hook.starts_with("You can build"), "{}", best.hook);
    assert_eq!(best.rating, Some(10.0));
    // the model's title, on one line
    assert_eq!(title_of(best), "Weather \"station\" on a budget");
    // a model that fails leaves everything unrated
    let broken = FnModel::new("broken", |_, _, _| Err(AutoCutError::Llm("down".into())));
    let mut c2 = candidates(&s, 8.0, 30.0, None);
    rate_hooks(&broken, &mut c2, &mut notes).await;
    assert!(c2.iter().all(|x| x.rating.is_none()) && !notes.is_empty());
}

#[test]
fn titles_and_slugs() {
    let c = Candidate { from: 0, to: 0, start: 0.0, end: 1.0, hook: HOOK.into(), text: HOOK.into(), score: 0.0, rating: None, title: None };
    assert_eq!(title_of(&c), "Why do 90 percent of rockets fail");
    assert_eq!(slug("Why do 90% of rockets fail?"), "why-do-90-of-rockets-fail");
    assert_eq!(slug("¿¡!!"), "short");
    assert!(slug(&"long ".repeat(50)).len() <= 40);
    assert_eq!(tidy_title("  \"Quoted\u{7}\"  "), "Quoted");
}

#[test]
fn hostile_transcripts_never_panic() {
    let nan = vec![Word { i: 0, start: f64::NAN, end: 1.0, text: "a.".into() }, Word { i: 1, start: 5.0, end: 2.0, text: "b.".into() }];
    let inf = vec![Word { i: 0, start: 0.0, end: f64::INFINITY, text: "x".into() }];
    let empty: Vec<Word> = Vec::new();
    let blank = vec![Word { i: 0, start: 0.0, end: 0.5, text: String::new() }; 3];
    for w in [nan, inf, empty, blank] {
        let s = sentences(&w);
        for (min, max) in [(15.0, 60.0), (f64::NAN, f64::NAN), (-5.0, 1e300), (100.0, 1.0)] {
            let c = candidates(&s, min, max, Some(""));
            let _ = pick(&c, 3);
        }
    }
    // a long talk with a short range makes many candidates, within the cap
    let many: Vec<&str> = vec!["One two three four five six seven eight nine ten."; 3000];
    let words = speech(0.0, &many);
    let s = sentences(&words);
    let c = candidates(&s, 3.0, 180.0, None);
    assert!(c.len() <= MAX_CANDIDATES);
}

fn ticks(s: f64) -> i64 {
    (s * TICKS_PER_SECOND as f64).round() as i64
}

/// End to end on a headless session: a 160-second clip with a transcript becomes two vertical
/// shorts on whole sentences, each its own sequence with styled captions; one is exported.
#[tokio::test(flavor = "multi_thread")]
async fn makes_vertical_shorts_with_captions_end_to_end() {
    let mut s = Session::default();
    let item = s.execute("file.newBarsAndTone", json!({"seconds": 160})).unwrap()["item"].as_u64().unwrap();
    let words: Vec<Value> = speech(1.0, &talk()).iter().map(|w| json!({"text": w.text, "start": ticks(w.start), "end": ticks(w.end)})).collect();
    s.execute("transcript.set", json!({"item": item, "transcript": {"language": "en", "speakers": [{"name": "Host"}], "words": words}})).unwrap();
    let ex = SessionExecutor::from(s);
    let dir = std::env::temp_dir().join(format!("filmcraft-shorts-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let p = json!({"item": item, "count": 2, "minSeconds": 8.0, "maxSeconds": 30.0, "aspect": "9:16", "captions": true, "title": true, "analysisFps": 2});
    let mut lines = Vec::new();
    let out = make(&ex, &p, None, &mut |l: &str| lines.push(l.to_string())).await.unwrap();
    let shorts = out["shorts"].as_array().unwrap();
    assert_eq!(shorts.len(), 2, "{out}");
    {
        let g = ex.0.lock().unwrap();
        for sh in shorts {
            let id = sh["sequence"].as_u64().unwrap();
            let q = g.project.items.values().find(|i| i.id.0 == id).and_then(|i| i.as_sequence()).unwrap();
            assert_eq!((q.settings.width, q.settings.height), (1080, 1920), "vertical");
            let secs = q.duration().seconds();
            assert!((8.0..=31.0).contains(&secs), "{secs}");
            let cap = q.caption_tracks.first().expect("captions");
            assert!(cap.captions.len() >= 4, "{} captions", cap.captions.len());
            assert!(!cap.style.background && cap.style.outline > 0.0 && cap.style.size > 60.0, "the social style");
            assert!(cap.captions.iter().all(|c| c.text.chars().count() <= 24), "a few words at a time");
            assert!(q.video_tracks.iter().flat_map(|t| &t.items).count() >= 2, "the clip and the title card");
        }
    }
    assert!(lines.iter().any(|l| l.contains("reframing")), "{lines:?}");
    // the same with an export folder: one short, exported with its captions burned in
    let p = json!({"item": item, "count": 1, "minSeconds": 8.0, "maxSeconds": 12.0, "captions": true, "analysisFps": 1, "export": dir.to_string_lossy()});
    let out = make(&ex, &p, None, &mut |_| {}).await.unwrap();
    let file = out["shorts"][0]["export"].as_str().unwrap_or_else(|| panic!("{out}")).to_string();
    assert!(std::fs::metadata(&file).map(|m| m.len() > 1000).unwrap_or(false), "{file}");
    let _ = std::fs::remove_dir_all(&dir);
    // no transcript, no shorts (and a clear error)
    let mut bare = Session::default();
    let item = bare.execute("file.newBarsAndTone", json!({"seconds": 20})).unwrap()["item"].as_u64().unwrap();
    let ex = SessionExecutor::from(bare);
    let e = make(&ex, &json!({"item": item}), None, &mut |_| {}).await.unwrap_err().to_string();
    assert!(e.contains("transcri"), "{e}");
}
