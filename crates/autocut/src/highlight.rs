//! Highlight selection: score transcript segments, keep the best until the target duration,
//! and cut the complement with ripple text-extracts (last range first, so word indices hold).

use serde_json::Value;

use filmcraft_time::TICKS_PER_SECOND;

/// One transcript word, in seconds.
#[derive(Debug, Clone)]
pub struct Word {
    pub i: u64,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// The words of a `transcript.inspect` result (ticks become seconds); `None` when there are none.
pub fn parse_words(t: &Value) -> Option<Vec<Word>> {
    let arr = t["words"].as_array()?;
    if arr.is_empty() {
        return None;
    }
    let w = |v: &Value| v.as_i64().map(|t| t as f64 / TICKS_PER_SECOND as f64);
    let words: Vec<Word> = arr
        .iter()
        .filter_map(|v| {
            Some(Word {
                i: v["i"].as_u64()?,
                start: w(v.get("start").unwrap_or(&Value::Null))?,
                end: w(v.get("end").unwrap_or(&Value::Null))?,
                text: v["text"].as_str().unwrap_or_default().to_string(),
            })
        })
        .collect();
    (words.len() == arr.len()).then_some(words)
}

/// A stretch of speech bounded by word indices `first..=last`.
#[derive(Debug, Clone)]
struct Segment {
    first: u64,
    /// Positions of the first and last word in the word list.
    first_pos: usize,
    last_pos: usize,
    start: f64,
    end: f64,
    score: f64,
    text: String,
}

/// Words spoken without a break longer than `gap` seconds, or across sentence punctuation.
fn segments(words: &[Word]) -> Vec<Segment> {
    const GAP: f64 = 1.2;
    let mut segs: Vec<Segment> = Vec::new();
    let mut start = 0usize;
    for w in 0..words.len() {
        let last_of_sentence = words[w].text.ends_with(['.', '!', '?']);
        let big_gap = words.get(w + 1).is_some_and(|n| n.start - words[w].end > GAP);
        let too_long = w - start >= 40;
        if (last_of_sentence || big_gap || too_long) && w >= start {
            segs.push(segment(words, start, w));
            start = w + 1;
        }
    }
    if start < words.len() {
        segs.push(segment(words, start, words.len() - 1));
    }
    segs
}

fn segment(words: &[Word], first: usize, last: usize) -> Segment {
    let text = words[first..=last].iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
    Segment { first: words[first].i, first_pos: first, last_pos: last, start: words[first].start, end: words[last].end, score: 0.0, text }
}

/// Short glue words that carry no interest.
fn is_stopword(s: &str) -> bool {
    const STOP: &[&str] = &[
        "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "is", "are", "was", "were", "be", "been", "it", "this", "that", "these", "those",
        "i", "you", "we", "they", "he", "she", "as", "for", "with", "so", "just", "really", "very", "um", "uh", "like", "okay", "ok", "yeah",
    ];
    let lower: String = s.chars().filter(char::is_ascii_alphabetic).collect::<String>().to_ascii_lowercase();
    STOP.contains(&lower.as_str()) || lower.len() <= 1
}

fn score(seg: &Segment, topic: Option<&str>) -> f64 {
    let dur = seg.end - seg.start;
    // sweet spot around 6..15s; shorter or much longer is less highlight-worthy
    let length = if (6.0..=15.0).contains(&dur) {
        1.0
    } else if (2.0..=30.0).contains(&dur) {
        0.5
    } else {
        0.1
    };
    let words: Vec<&str> = seg.text.split_whitespace().collect();
    let content = words.iter().filter(|w| !is_stopword(w)).count() as f64 / words.len().max(1) as f64;
    let mut topic_hits = 0.0;
    if let Some(topic) = topic {
        let terms: Vec<String> = topic.split_whitespace().map(|t| t.to_ascii_lowercase()).collect();
        for w in &words {
            let lower = w.to_ascii_lowercase();
            if terms.iter().any(|t| !t.is_empty() && lower.contains(t)) {
                topic_hits += 1.0;
            }
        }
    }
    let numbers = words.iter().filter(|w| w.chars().any(|c| c.is_ascii_digit())).count() as f64 * 0.3;
    length + content + topic_hits * 2.0 + numbers
}

/// Lead-in kept before a passage's first word, seconds (taken only from the silence before it).
const PRE_ROLL: f64 = 0.25;
/// Tail kept after its last word, so a sentence lands instead of being clipped.
const POST_ROLL: f64 = 0.45;
/// Kept passages closer than this are joined: a cut that short reads as a glitch.
const MERGE_GAP: f64 = 1.0;
/// Removals shorter than this are left in for the same reason.
const MIN_CUT: f64 = 0.4;

/// Pick the passages to keep and the time ranges to cut. Returns (kept count, removal ranges
/// `(start, end)` in sequence seconds, **last first** so ripple extraction does not shift the
/// earlier ones, notes). Kept passages get a little lead-in and tail from the surrounding
/// silence, close passages are joined, the silent head and tail of the sequence go too, and no
/// sliver shorter than [`MIN_CUT`] is cut — the cuts land between thoughts, not on words.
pub fn plan(words: &[Word], target_seconds: f64, topic: Option<&str>, total_seconds: f64) -> (usize, Vec<(f64, f64)>, Vec<String>) {
    let mut notes = Vec::new();
    let mut segs = segments(words);
    if segs.is_empty() {
        return (0, Vec::new(), vec!["the transcript has no words".into()]);
    }
    for s in &mut segs {
        s.score = score(s, topic);
    }
    let speech: f64 = segs.iter().map(|s| s.end - s.start).sum();
    if speech <= target_seconds {
        let s = format!("the speech is only {speech:.0}s; keeping all of it");
        notes.push(s);
        return (segs.len(), Vec::new(), notes);
    }
    // best first until the target is filled, then back to chronological order
    let mut order: Vec<usize> = (0..segs.len()).collect();
    order.sort_by(|a, b| segs[*b].score.total_cmp(&segs[*a].score).then(segs[*a].start.total_cmp(&segs[*b].start)));
    let mut kept: Vec<usize> = Vec::new();
    let mut total = 0.0;
    for i in order {
        if total >= target_seconds {
            break;
        }
        let d = segs[i].end - segs[i].start;
        if total + d > target_seconds * 1.25 && !kept.is_empty() {
            continue; // would overshoot badly; try a shorter one
        }
        total += d;
        kept.push(i);
    }
    kept.sort_by_key(|i| segs[*i].first);
    let kept_count = kept.len();
    notes.push(format!("kept: {}", kept.iter().map(|i| format!("\"{}\"", truncate(&segs[*i].text, 60))).collect::<Vec<_>>().join(" / ")));
    let total_seconds = if total_seconds.is_finite() && total_seconds > 0.0 { total_seconds } else { words.last().map_or(0.0, |w| w.end) };
    // padded keep ranges: lead-in and tail come only from the silence around the passage
    let mut keeps: Vec<(f64, f64)> = Vec::new();
    for i in &kept {
        let s = &segs[*i];
        let before = s.first_pos.checked_sub(1).and_then(|p| words.get(p)).map(|w| (w.end + s.start) / 2.0).unwrap_or(0.0);
        let after = words.get(s.last_pos + 1).map(|w| (s.end + w.start) / 2.0).unwrap_or(total_seconds);
        let a = (s.start - PRE_ROLL).max(before).max(0.0);
        let b = (s.end + POST_ROLL).min(after).min(total_seconds);
        match keeps.last_mut() {
            Some(last) if a - last.1 < MERGE_GAP => last.1 = last.1.max(b),
            _ => keeps.push((a, b.max(a))),
        }
    }
    // the complement over the whole sequence, without slivers
    let mut cuts: Vec<(f64, f64)> = Vec::new();
    let mut cursor = 0.0;
    for (a, b) in &keeps {
        if a - cursor >= MIN_CUT {
            cuts.push((cursor, *a));
        }
        cursor = cursor.max(*b);
    }
    if total_seconds - cursor >= MIN_CUT {
        cuts.push((cursor, total_seconds));
    }
    cuts.reverse();
    (kept_count, cuts, notes)
}

fn truncate(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn word(i: u64, text: &str, start: f64, end: f64) -> Value {
        let t = |s: f64| (s * TICKS_PER_SECOND as f64) as i64;
        json!({"i": i, "text": text, "start": t(start), "end": t(end), "speaker": 0})
    }

    fn words(v: &[Value]) -> Value {
        json!({"words": v})
    }

    #[test]
    fn parses_words_and_none() {
        let w = parse_words(&words(&[word(0, "a", 0.0, 0.5), word(1, "b", 0.6, 1.0)])).unwrap();
        assert_eq!((w.len(), w[1].start), (2, 0.6));
        assert!(parse_words(&json!({"words": []})).is_none());
        assert!(parse_words(&json!({})).is_none());
        // a word missing its index makes the whole transcript unusable, not silently shorter
        let broken = json!({"words": [json!({"text": "x", "start": 0, "end": 1})]});
        assert!(parse_words(&broken).is_none());
    }

    #[test]
    fn keeps_topic_segments_and_cuts_the_rest_last_first() {
        // 10 stretches of ~2.5s speech separated by pauses; "launch" is the topic of #2 and #7
        let mut ws = Vec::new();
        let mut i = 0u64;
        for s in 0..10 {
            let base = s as f64 * 12.0;
            let text = if s == 2 || s == 7 { "the launch sequence begins now" } else { "and then we waited" };
            for w in text.split_whitespace() {
                ws.push(word(i, w, base + i as f64 * 0.6, base + i as f64 * 0.6 + 0.5));
                i += 1;
            }
        }
        let words = parse_words(&words(&ws)).unwrap();
        let total = 125.0;
        let (kept, cuts, notes) = plan(&words, 6.0, Some("launch"), total);
        assert_eq!(kept, 2, "{notes:?}");
        assert_eq!(cuts.len(), 3, "{cuts:?}"); // head + between + tail
        // last first, and the tail goes to the end of the sequence
        assert_eq!(cuts[0].1, total);
        assert!(cuts[0].0 > cuts[1].0, "ranges go last-first: {cuts:?}");
        // the kept passages keep a lead-in and a tail: nothing is cut on a word
        let first_kept = words.iter().find(|w| w.text == "launch").unwrap();
        let head_end = cuts[2].1;
        assert!(head_end < first_kept.start - 0.6 + 0.01, "lead-in kept: cut ends {head_end}, passage starts {}", first_kept.start);
        for (a, b) in &cuts {
            assert!(b - a >= MIN_CUT, "no slivers: {a}..{b}");
            assert!(!words.iter().any(|w| w.text == "launch" && w.start < *b && *a < w.end), "a kept word was cut");
        }
    }

    #[test]
    fn close_passages_join_and_slivers_stay() {
        // two kept sentences 0.5 s apart become one keep: no 0.5 s jump cut between them
        let mut ws = Vec::new();
        let mut t = 0.0;
        for (k, text) in ["one two three.", "four five six.", "padding padding padding padding padding padding padding."].iter().enumerate() {
            for w in text.split_whitespace() {
                ws.push(word(ws.len() as u64, w, t, t + 0.4));
                t += 0.5;
            }
            t += if k == 0 { 0.5 } else { 20.0 };
        }
        let words = parse_words(&words(&ws)).unwrap();
        let (_, cuts, _) = plan(&words, 3.0, Some("two five"), t);
        // nothing is cut between the first two sentences
        assert!(!cuts.iter().any(|(a, b)| *a > 1.0 && *b < 3.6), "{cuts:?}");
    }

    #[test]
    fn short_speech_keeps_everything() {
        let ws = [word(0, "hi", 0.0, 0.5), word(1, "there", 0.6, 1.0)];
        let words = parse_words(&words(&ws)).unwrap();
        let (kept, cuts, _) = plan(&words, 60.0, None, 1.0);
        assert_eq!((kept, cuts.len()), (1, 0));
    }
}
