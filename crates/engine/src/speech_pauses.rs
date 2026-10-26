//! Pauses in the middle of a phrase (`media.checkPauses`): in an item's transcript (made by
//! `media.understandSpeech` or `media.transcribe`), every silence between two words of at least
//! `min` seconds (0.45 by default) where the first word does not end a phrase — no comma, full
//! stop, question or exclamation mark, colon, semicolon, dash or danda after it. A narration
//! read with a breath in the wrong place sounds broken there; checked on the export, it is
//! found the way a listener finds it. `from` and `to` (seconds) limit the check to a stretch.

use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, f64_p, item_p};
use crate::{Result, Session};

pub const PAUSES_CMD: &str = "media.checkPauses";
/// Most pauses listed.
const MAX_LISTED: usize = 200;
/// Words shown either side of a pause.
const AROUND: usize = 3;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: PAUSES_CMD,
        label: "Check Pauses",
        menu: &[],
        shortcut: None,
        params: r#"{"item":id,"min":0.2..3=0.45,"from":seconds?,"to":seconds?}"#,
        enabled: always,
        run: check,
        journal: false,
    }]
}

/// Whether a word, as written, ends a phrase (its last character is punctuation that ends one).
pub fn ends_phrase(word: &str) -> bool {
    let w = word.trim_end_matches(['"', '\'', '”', '’', ')', ']', '»']);
    w.ends_with([',', '.', '!', '?', ':', ';', '—', '–', '-', '…', '।', '॥', '،', '。', '，', '、'])
}

fn check(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = PAUSES_CMD;
    let item = item_p(p, "item").ok_or_else(|| bad(cmd, "need `item`"))?;
    let min = match f64_p(p, "min") {
        None => 0.45,
        Some(m) if m.is_finite() && (0.2..=3.0).contains(&m) => m,
        Some(_) => return Err(bad(cmd, "`min` is 0.2 to 3 seconds")),
    };
    let secs = |k: &str| -> Result<Option<f64>> {
        match f64_p(p, k) {
            None => Ok(None),
            Some(x) if x.is_finite() && x >= 0.0 => Ok(Some(x)),
            Some(_) => Err(bad(cmd, format!("`{k}` is seconds, zero or more"))),
        }
    };
    let (from, to) = (secs("from")?.unwrap_or(0.0), secs("to")?.unwrap_or(f64::MAX));
    let t = s.project.transcripts.get(&item).cloned().ok_or_else(|| bad(cmd, "the item has no transcript yet (media.understandSpeech first)"))?;
    let words: Vec<&filmcraft_project::transcript::Word> = t.words.iter().filter(|w| w.start.seconds() >= from && w.end.seconds() <= to).collect();
    let text = |a: usize, b: usize| words.get(a..b).unwrap_or_default().iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
    let mut pauses = Vec::new();
    let mut longest = 0.0f64;
    for (k, pair) in words.windows(2).enumerate() {
        let (Some(a), Some(b)) = (pair.first(), pair.get(1)) else { continue };
        let gap = (b.start - a.end).max(Tick::ZERO).seconds();
        if gap < min || ends_phrase(&a.text) {
            continue;
        }
        longest = longest.max(gap);
        if pauses.len() < MAX_LISTED {
            pauses.push(json!({
                "at": (a.end.seconds() * 100.0).round() / 100.0,
                "gap": (gap * 100.0).round() / 100.0,
                "before": text((k + 1).saturating_sub(AROUND), k + 1),
                "after": text(k + 1, (k + 1 + AROUND).min(words.len())),
            }));
        }
    }
    Ok(json!({"item": item.0, "language": t.language, "words": words.len(), "min": min, "pauses": pauses, "longest": (longest * 100.0).round() / 100.0}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_project::transcript::{Transcript, Word};

    fn word(text: &str, a: f64, b: f64) -> Word {
        Word { text: text.into(), start: Tick::from_seconds_f64(a), end: Tick::from_seconds_f64(b), speaker: None, confidence: 1.0 }
    }

    #[test]
    fn a_breath_inside_a_phrase_is_found_one_after_a_comma_is_not() {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let item = s.active_sequence().unwrap().video_tracks[0].items[0].item;
        let t = Transcript {
            language: "en".into(),
            source: "manual".into(),
            speakers: Vec::new(),
            words: vec![
                word("This", 0.0, 0.2),
                word("has", 0.25, 0.4),
                word("been", 0.45, 0.6),
                word("my", 1.4, 1.5),
                word("journey,", 1.55, 1.9),
                word("and", 2.8, 2.9),
                word("it", 2.95, 3.0),
                word("continues.", 3.05, 3.5),
            ],
        };
        s.edit("transcript", |pr, _| {
            pr.transcripts.insert(item, std::sync::Arc::new(t));
            Ok(())
        })
        .unwrap();
        let r = s.execute(PAUSES_CMD, json!({"item": item.0})).unwrap();
        let pauses = r["pauses"].as_array().unwrap();
        assert_eq!(pauses.len(), 1, "{r}");
        assert_eq!((pauses[0]["at"].as_f64(), pauses[0]["gap"].as_f64()), (Some(0.6), Some(0.8)));
        assert_eq!((pauses[0]["before"].as_str(), pauses[0]["after"].as_str()), (Some("This has been"), Some("my journey, and")));
        // a stretch: only its words
        let r = s.execute(PAUSES_CMD, json!({"item": item.0, "from": 1.0})).unwrap();
        assert!(r["pauses"].as_array().unwrap().is_empty());
        for bad in [json!({}), json!({"item": item.0, "min": 9}), json!({"item": item.0, "from": -1})] {
            assert!(s.execute(PAUSES_CMD, bad.clone()).is_err(), "{bad}");
        }
        assert!(ends_phrase("ഉണ്ട്,") && ends_phrase("काम।") && ends_phrase("\"done.\"") && !ends_phrase("my"));
    }
}
