//! Long, narrated videos (a showcase like a series about one's 3D printers): the steps once done
//! by hand, now part of writing the plan and building it.
//!
//! - **Not every entry narrated** (`narrate`: the share of entries that get words; all by
//!   default): the first and the last keep theirs, then the ones with checked facts, then others
//!   spread through the show ([`pick_narrated`]); the rest play under the music with their labels
//!   on screen (an empty line), as a person telling the story would leave many shots alone.
//! - **The person's own voice** (`bites`): clips where the person talks over their own footage —
//!   speech understanding's off-screen voice or talking head, said cleanly — give short bites, a
//!   whole sentence of 1.5 to 8 s ([`find_bites`]). The entry's shot of that clip is set to the
//!   bite and heard, captioned in English (the script model translates other languages), and the
//!   entry needs no narration of its own ([`apply_bites`]).
//! - **B-roll for each line**: a narrated entry's shots are ordered by how well each shows what
//!   its line says ([`order_by_line`]: `media.searchScenes` with the line, its items only).
//! - **Pauses**: after the export, `media.checkPauses` on its transcript lists the breaths in
//!   the middle of a phrase.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::llm::ChatModel;
use crate::showcase::Entry;
use crate::{AutoCutError, Executor};

/// A bite's length, seconds, and how cleanly it must be said (speech understanding's fluency).
pub const BITE_SECONDS: (f64, f64) = (1.5, 8.0);
pub const BITE_FLUENCY: f64 = 0.6;
/// Room around a bite's words, seconds (before, after).
const BITE_ROOM: (f64, f64) = (0.2, 0.3);
/// How much louder a bite is heard than the shots' own sound, dB.
pub const BITE_GAIN: f64 = 6.0;
/// Most bites in one show.
const MAX_BITES: usize = 40;

/// The entries (indices into `entries`) that keep their words: about `share` of the entries that
/// are not scenes — the first and the last always, then those with facts, then others spread
/// evenly through the show.
pub fn pick_narrated(entries: &[Entry], share: f64) -> BTreeSet<usize> {
    let robots: Vec<usize> = entries.iter().enumerate().filter(|(_, e)| !e.is_scene()).map(|(i, _)| i).collect();
    let share = if share.is_finite() { share.clamp(0.0, 1.0) } else { 1.0 };
    let want = ((robots.len() as f64 * share).round() as usize).clamp(robots.len().min(1), robots.len());
    let mut keep: BTreeSet<usize> = BTreeSet::new();
    keep.extend(robots.first());
    if want >= 2 {
        keep.extend(robots.last());
    }
    for &i in &robots {
        if keep.len() >= want {
            break;
        }
        if entries.get(i).is_some_and(|e| !e.facts.is_empty()) {
            keep.insert(i);
        }
    }
    // the rest spread: every n-th of those left
    let left: Vec<usize> = robots.iter().copied().filter(|i| !keep.contains(i)).collect();
    let need = want.saturating_sub(keep.len());
    if need > 0 && !left.is_empty() {
        let step = left.len() as f64 / need as f64;
        for k in 0..need {
            if let Some(i) = left.get(((k as f64 + 0.5) * step) as usize) {
                keep.insert(*i);
            }
        }
    }
    keep
}

/// A bite of the person's own voice for an entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Bite {
    /// The entry's number.
    pub entry: usize,
    pub item: u64,
    /// Seconds into the clip.
    pub from: f64,
    pub to: f64,
    /// What is said, as heard, and in English for the caption.
    pub text: String,
    pub english: String,
    pub language: String,
}

/// One kept stretch of speech in an item: (start, end, text, fluency).
type Said = (f64, f64, String, f64);

/// The best bite in `said` (an item's kept speech, its shot hint): a whole, clean sentence of the
/// right length, the most fluent first, then the longest.
pub fn best_bite(said: &[Said], hint: &str) -> Option<(f64, f64, String)> {
    if !matches!(hint, "voice" | "talkingHead") {
        return None;
    }
    said.iter()
        .filter(|(a, b, t, f)| {
            let len = b - a;
            len.is_finite() && (BITE_SECONDS.0..=BITE_SECONDS.1).contains(&len) && *f >= BITE_FLUENCY && t.split_whitespace().count() >= 3
        })
        .max_by(|x, y| x.3.total_cmp(&y.3).then((x.1 - x.0).total_cmp(&(y.1 - y.0))))
        .map(|(a, b, t, _)| (*a, *b, t.trim().to_string()))
}

const TRANSLATE_SYSTEM: &str =
    "You translate short spoken lines into plain English for on-screen captions: the meaning and tone, short, no additions. Answer with JSON only.";

/// The person's own voice in the entries' clips: understood (`media.understandSpeech`), and the
/// best bite of each entry's clips ([`best_bite`]), translated to English by `model` where it
/// is in another language (left as heard without a model).
pub async fn find_bites(ex: &dyn Executor, model: Option<&dyn ChatModel>, entries: &[Entry], lines: &mut dyn FnMut(&str)) -> Vec<Bite> {
    let mut items: Vec<u64> = entries.iter().filter(|e| !e.is_scene()).flat_map(|e| e.shots.iter().filter(|s| !s.photo).map(|s| s.item)).collect();
    items.sort_unstable();
    items.dedup();
    if items.is_empty() {
        return Vec::new();
    }
    lines(&format!("listening for your own voice in {} clip(s)", items.len()));
    if ex.execute("media.understandSpeech", json!({"items": items, "wait": true})).await.is_err() {
        return Vec::new();
    }
    let Ok(v) = ex.execute("media.inspectSpeech", json!({"items": items})).await else { return Vec::new() };
    let speech = crate::quick::parse_speech(&v);
    let mut bites = Vec::new();
    for e in entries.iter().filter(|e| !e.is_scene()) {
        let found = e.shots.iter().filter(|s| !s.photo).find_map(|s| {
            let sp = speech.iter().find(|x| x.item == s.item)?;
            let said: Vec<Said> = sp.kept.iter().map(|k| (k.start, k.end, k.text.clone(), k.fluency)).collect();
            best_bite(&said, &sp.shot).map(|(a, b, t)| (s.item, a, b, t, sp.language.clone()))
        });
        if let Some((item, from, to, text, language)) = found {
            let english = if language.starts_with("en") { text.clone() } else { String::new() };
            bites.push(Bite { entry: e.number, item, from, to, text, english, language });
        }
        if bites.len() >= MAX_BITES {
            break;
        }
    }
    // the others, into English in one go
    let foreign: Vec<usize> = (0..bites.len()).filter(|&i| bites.get(i).is_some_and(|b| b.english.is_empty())).collect();
    if let Some(m) = model.filter(|_| !foreign.is_empty()) {
        let list: Vec<&str> = foreign.iter().filter_map(|&i| bites.get(i)).map(|b| b.text.as_str()).collect();
        let user = format!("Translate each line into English, in order.\n{}\nJSON: {{\"english\": [\"...\"]}}", json!(list));
        if let Ok(v) = m.complete_json(TRANSLATE_SYSTEM, &user, &[]).await {
            let out = v["english"].as_array().cloned().unwrap_or_default();
            for (k, &i) in foreign.iter().enumerate() {
                if let (Some(b), Some(t)) = (bites.get_mut(i), out.get(k).and_then(Value::as_str).map(str::trim).filter(|t| !t.is_empty())) {
                    b.english = if crate::director::english(t) { t.to_string() } else { String::new() };
                }
            }
        }
    }
    for b in bites.iter_mut().filter(|b| b.english.is_empty()) {
        b.english = b.text.clone();
    }
    bites
}

/// The bites into the plan: each entry's shot of its clip set to the bite (heard, louder than
/// the shots' sound, its caption to show), moved first; the entry's own line left empty (its own
/// words tell it). `lines` are the script's lines, by entry index.
pub fn apply_bites(entries: &mut [Entry], lines: &mut [String], bites: &[Bite]) -> usize {
    let mut n = 0;
    for (k, e) in entries.iter_mut().enumerate() {
        let Some(b) = bites.iter().find(|b| b.entry == e.number) else { continue };
        let Some(i) = e.shots.iter().position(|s| s.item == b.item && !s.photo) else { continue };
        let mut shot = e.shots.remove(i);
        shot.from = Some(((b.from - BITE_ROOM.0).max(0.0) * 100.0).round() / 100.0);
        shot.to = Some((((b.to + BITE_ROOM.1).min(shot.clip_seconds.max(b.to))) * 100.0).round() / 100.0);
        shot.speed = None;
        shot.mute = Some(false);
        shot.volume = Some(BITE_GAIN);
        shot.says = Some(b.english.clone());
        e.shots.insert(0, shot);
        if let Some(l) = lines.get_mut(k) {
            l.clear();
        }
        n += 1;
    }
    n
}

/// A narrated entry's shots, the one that best shows what `line` says first
/// (`media.searchScenes` with the line over the entry's clips); as they were without a match.
pub async fn order_by_line(ex: &dyn Executor, e: &mut Entry, line: &str) {
    let items: Vec<u64> = e.shots.iter().filter(|s| !s.photo).map(|s| s.item).collect();
    if line.split_whitespace().count() < 3 || items.len() < 2 || e.shots.first().is_some_and(|s| s.says.is_some()) {
        return;
    }
    let Ok(v) = ex.execute("media.searchScenes", json!({"query": line, "items": items, "limit": 20})).await else { return };
    let rank: Vec<u64> = v["hits"].as_array().map(Vec::as_slice).unwrap_or_default().iter().filter_map(|h| h["item"].as_u64()).collect();
    if rank.is_empty() {
        return;
    }
    let pos = |item: u64| rank.iter().position(|r| *r == item).unwrap_or(usize::MAX);
    e.shots.sort_by_key(|s| if s.photo { usize::MAX } else { pos(s.item) });
}

/// Breaths in the middle of a phrase in an exported video's speech: the file transcribed in a
/// session of its own (the project is left as it is) in `language`, and checked
/// (`media.checkPauses`).
pub async fn check_pauses(path: &str, language: &str) -> Result<Value, String> {
    let (path, language) = (path.to_string(), language.to_string());
    tokio::task::spawn_blocking(move || {
        let mut s = filmcraft_engine::Session::default();
        let r = s.execute("file.import", json!({"paths": [path]})).map_err(|e| e.to_string())?;
        let item = r["items"].as_array().and_then(|a| a.first()).and_then(Value::as_u64).ok_or("the export could not be read back")?;
        match s.execute("media.understandSpeech", json!({"items": [item], "language": language, "wait": true})) {
            Ok(_) => s.execute("media.checkPauses", json!({"item": item})).map_err(|e| e.to_string()),
            // (a build without speech-to-text cannot listen: said, not an error)
            Err(e) if e.to_string().contains("not available in this build") => Ok(json!({"skipped": "speech-to-text is not in this build", "pauses": []})),
            Err(e) => Err(e.to_string()),
        }
    })
    .await
    .map_err(|e| format!("internal error: {e}"))?
}

/// Shape a written script for a long, narrated show (`p`'s `narrate` and `bites`): the bites of
/// the person's own voice into their entries, the words left to about `narrate` of the entries,
/// each narrated entry's shots ordered by its line; what was done, in words.
pub async fn shape(
    ex: &dyn Executor,
    model: Option<&dyn ChatModel>,
    p: &Value,
    entries: &mut [Entry],
    lines: &mut [String],
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<String>, AutoCutError> {
    let (narrate, bites) = options(p)?;
    let mut said = Vec::new();
    if bites {
        let found = find_bites(ex, model, entries, progress).await;
        let n = apply_bites(entries, lines, &found);
        said.push(format!("{n} entr{} told in the person's own voice", if n == 1 { "y" } else { "ies" }));
    }
    if narrate < 1.0 {
        let keep = pick_narrated(entries, narrate);
        let mut quiet = 0;
        for (k, l) in lines.iter_mut().enumerate() {
            if !keep.contains(&k) && !l.is_empty() && entries.get(k).is_some_and(|e| !e.is_scene()) {
                l.clear();
                quiet += 1;
            }
        }
        said.push(format!("{quiet} entr{} left to the music and their labels", if quiet == 1 { "y" } else { "ies" }));
    }
    for (e, l) in entries.iter_mut().zip(lines.iter()) {
        if !l.trim().is_empty() {
            order_by_line(ex, e, l).await;
        }
    }
    Ok(said)
}

/// The plan options a narrated show reads: `narrate` (0..1) and `bites` (bool).
pub fn options(p: &Value) -> Result<(f64, bool), AutoCutError> {
    let narrate = match p.get("narrate").filter(|v| !v.is_null()) {
        None => 1.0,
        Some(v) => v
            .as_f64()
            .filter(|x| x.is_finite() && (0.0..=1.0).contains(x))
            .ok_or_else(|| AutoCutError::Invalid("narrate: the share of entries narrated, 0 to 1".into()))?,
    };
    let bites = match p.get("bites").filter(|v| !v.is_null()) {
        None => false,
        Some(v) => v.as_bool().ok_or_else(|| AutoCutError::Invalid("bites: true or false".into()))?,
    };
    Ok((narrate, bites))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::showcase::Shot;

    fn entry(number: usize, facts: bool, scene: bool) -> Entry {
        Entry {
            number,
            label: format!("printer {number}"),
            kind: if scene { crate::showcase::SCENE.into() } else { "printer".into() },
            facts: if facts { vec!["fact".into()] } else { Vec::new() },
            shots: vec![
                Shot { item: 100 + number as u64, clip_seconds: 30.0, ..Shot::default() },
                Shot { item: 200 + number as u64, clip_seconds: 20.0, ..Shot::default() },
            ],
            ..Entry::default()
        }
    }

    #[test]
    fn half_the_entries_keep_their_words_the_ends_and_the_facts_first() {
        let entries: Vec<Entry> = (1..=10).map(|n| entry(n, n == 4, n == 6)).collect();
        let keep = pick_narrated(&entries, 0.5);
        // nine entries that are not scenes: five keep their words
        assert_eq!(keep.len(), 5, "{keep:?}");
        assert!(keep.contains(&0) && keep.contains(&9) && keep.contains(&3), "{keep:?}");
        assert!(!keep.contains(&5), "a scene is not narrated");
        assert_eq!(pick_narrated(&entries, 1.0).len(), 9);
        assert_eq!(pick_narrated(&entries, 0.0).len(), 1, "the first always");
        assert_eq!(pick_narrated(&entries, f64::NAN).len(), 9);
    }

    #[test]
    fn a_clean_sentence_of_the_persons_own_voice_is_a_bite() {
        let said = vec![
            (1.0, 1.6, "uh so".to_string(), 0.9),
            (4.0, 9.0, "I printed this in eighteen minutes".to_string(), 0.8),
            (12.0, 15.0, "and then the cooling fan".to_string(), 0.65),
            (20.0, 40.0, "a very long ramble that goes on".to_string(), 0.9),
        ];
        assert_eq!(best_bite(&said, "voice"), Some((4.0, 9.0, "I printed this in eighteen minutes".to_string())));
        assert_eq!(best_bite(&said, "bRoll"), None, "someone in the background is not the narrator");
        let mut entries = vec![entry(1, false, false), entry(2, false, false)];
        let mut lines = vec!["one".to_string(), "two".to_string()];
        let bites = vec![Bite {
            entry: 2,
            item: 202,
            from: 4.0,
            to: 9.0,
            text: "x".into(),
            english: "I printed this in eighteen minutes".into(),
            language: "ml".into(),
        }];
        assert_eq!(apply_bites(&mut entries, &mut lines, &bites), 1);
        let s = &entries[1].shots[0];
        assert_eq!(
            (s.item, s.from, s.to, s.volume, s.says.as_deref()),
            (202, Some(3.8), Some(9.3), Some(BITE_GAIN), Some("I printed this in eighteen minutes"))
        );
        assert_eq!(lines, ["one", ""]);
        assert!(options(&json!({"narrate": 2})).is_err() && options(&json!({"bites": "yes"})).is_err());
        assert_eq!(options(&json!({"narrate": 0.5, "bites": true})).unwrap(), (0.5, true));
    }
}
