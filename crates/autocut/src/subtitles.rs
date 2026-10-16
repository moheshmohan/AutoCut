//! Subtitles from narration: each spoken line, as the viewer should read it (the shown text, not
//! the respelled one the voice reads), split into readable cues and timed against the line's
//! clip — the usual reading rules (YouTube's, and most broadcasters'): at most two lines of 42
//! characters, each cue on screen for 1–7 s, cut at sentence ends first, then at clauses, and
//! timed in proportion to its characters.
//!
//! [`cues`] does the splitting and timing; the showcase puts the cues on a caption track and
//! exports it as `<out>.srt` / `<out>.vtt` (`captions.export`).

use serde::{Deserialize, Serialize};

/// Most characters on one subtitle line.
pub const MAX_LINE: usize = 42;
/// Shortest and longest time a cue is on screen, seconds.
pub const MIN_CUE: f64 = 1.0;
pub const MAX_CUE: f64 = 7.0;
/// Words of one line looked at, at most (a hostile plan's line can be any length).
const MAX_WORDS: usize = 4000;

/// Text on screen from `start` to `end` (seconds); the lines of a cue are joined by `\n`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Cue {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

fn chars(s: &str) -> usize {
    s.chars().count()
}

/// Words a line should not end on: articles, prepositions, conjunctions, "I" — breaking after
/// them separates them from what they belong to.
const CLINGING: &[&str] = &[
    "a", "an", "the", "i", "to", "of", "in", "on", "at", "by", "for", "with", "from", "into", "and", "or", "but", "this", "that", "my", "our", "your", "its",
    "his", "her", "their", "very", "so",
];

/// What a line break between `before` and `after` costs, in characters of imbalance: less after
/// a sentence or clause, more after a clinging word or inside a name ("Zaku II", "DP World":
/// a capitalised word followed by another, or by a number).
fn line_break_cost(before: &str, after: &str) -> i64 {
    let capital = |w: &str| w.chars().next().is_some_and(char::is_uppercase);
    let lower = before.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
    if before.ends_with(['.', '!', '?', '…', ',', ';', ':', '—', '–']) {
        -4
    } else if capital(before) && (capital(after) || after.starts_with(|c: char| c.is_ascii_digit())) {
        8
    } else if CLINGING.contains(&lower.as_str()) {
        5
    } else {
        0
    }
}

/// `text` (one line of words) on at most two lines of at most [`MAX_LINE`] characters, broken
/// where the lines balance best and the break reads well ([`line_break_cost`]); `None` when it
/// does not fit. A single word longer than a line stays whole (there is nowhere to break it).
pub fn wrap(text: &str) -> Option<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let text = words.join(" ");
    if chars(&text) <= MAX_LINE || words.len() < 2 {
        return Some(text);
    }
    let mut best: Option<(usize, i64)> = None;
    let mut a = 0usize;
    let total = chars(&text);
    for k in 1..words.len() {
        a += words.get(k - 1).map_or(0, |w| chars(w)) + usize::from(k > 1);
        let b = total.saturating_sub(a + 1);
        if a > MAX_LINE || b > MAX_LINE {
            continue;
        }
        let cost = a.max(b) as i64 + line_break_cost(words.get(k - 1).copied().unwrap_or(""), words.get(k).copied().unwrap_or(""));
        if best.is_none_or(|(_, c)| cost < c) {
            best = Some((k, cost));
        }
    }
    let (k, _) = best?;
    Some(format!("{}\n{}", words.get(..k)?.join(" "), words.get(k..)?.join(" ")))
}

/// How good a place the end of `word` is to end a cue: a sentence ends there (best), a clause,
/// or neither.
fn break_cost(word: &str, next: Option<&str>) -> f64 {
    let core = word.trim_end_matches(['"', '\'', '”', '’', ')', ']']);
    if core.ends_with(['.', '!', '?', '…']) {
        0.0
    } else if core.ends_with([',', ';', ':', '—', '–']) || next.is_some_and(|n| n.starts_with(['—', '–', '-'])) {
        0.6
    } else {
        3.0
    }
}

/// The cues of one narration line spoken from `start` to `end` (seconds), before frame
/// alignment: the words split where reading is easiest — fewest cues, at sentence ends, then
/// clauses — with every cue fitting two lines and no cue on screen longer than [`MAX_CUE`]
/// (shorter than [`MIN_CUE`] only when there is no better split), each timed in proportion to
/// its characters.
fn split_line(text: &str, start: f64, end: f64) -> Vec<Cue> {
    let words: Vec<&str> = text.split_whitespace().take(MAX_WORDS).collect();
    let n = words.len();
    let span = end - start;
    if n == 0 || !span.is_finite() || span <= 0.0 {
        return Vec::new();
    }
    // characters before each word (a space between words counts as one)
    let mut at = vec![0usize; n + 1];
    for (i, w) in words.iter().enumerate() {
        let prev = at.get(i).copied().unwrap_or(0);
        if let Some(slot) = at.get_mut(i + 1) {
            *slot = prev + chars(w) + usize::from(i + 1 < n);
        }
    }
    let total = at.get(n).copied().unwrap_or(0).max(1) as f64;
    let len = |a: usize, b: usize| at.get(b).copied().unwrap_or(0).saturating_sub(at.get(a).copied().unwrap_or(0));
    // best[i]: the cheapest way to cut words[..i] into cues, and where its last cue starts
    let mut best: Vec<Option<(f64, usize)>> = vec![None; n + 1];
    if let Some(b) = best.get_mut(0) {
        *b = Some((0.0, 0));
    }
    for i in 1..=n {
        let mut pick: Option<(f64, usize)> = None;
        for j in (0..i).rev() {
            let Some((before, _)) = best.get(j).copied().flatten() else { continue };
            let cue = words.get(j..i).map(|w| w.join(" ")).unwrap_or_default();
            let single = j + 1 == i;
            if wrap(&cue).is_none() && !single {
                break; // longer cues (smaller j) only get longer
            }
            let seconds = span * len(j, i) as f64 / total;
            if seconds > MAX_CUE && !single {
                break;
            }
            let ends = if i == n { 0.0 } else { words.get(i - 1).map_or(3.0, |w| break_cost(w, words.get(i).copied())) };
            let short = if seconds < MIN_CUE { 5.0 * (MIN_CUE - seconds) } else { 0.0 };
            let cost = before + 1.0 + ends + short;
            if pick.is_none_or(|(c, _)| cost < c - 1e-9) {
                pick = Some((cost, j));
            }
        }
        if let Some(b) = best.get_mut(i) {
            *b = pick;
        }
    }
    let mut cuts = vec![n];
    let mut i = n;
    while i > 0 {
        let Some((_, j)) = best.get(i).copied().flatten() else { return Vec::new() };
        cuts.push(j);
        i = j;
    }
    cuts.reverse();
    cuts.windows(2)
        .filter_map(|w| {
            let (a, b) = (*w.first()?, *w.get(1)?);
            let text = words.get(a..b)?.join(" ");
            let text = wrap(&text).unwrap_or(text);
            Some(Cue { start: start + span * at.get(a).copied()? as f64 / total, end: start + span * at.get(b).copied()? as f64 / total, text })
        })
        .collect()
}

/// Subtitle cues for narration `lines` (each its shown text, spoken from `start` to `end`
/// seconds), in time order: every line split into readable cues ([`split_line`]), all times on
/// frames of `fps`. A cue that would be on screen for less than [`MIN_CUE`] stays up longer —
/// into the pause after its line, or by starting its neighbour in the line a little later —
/// but never past the start of the next line, and no cue overlaps another.
pub fn cues(lines: &[Cue], fps: f64) -> Vec<Cue> {
    let fps = if fps.is_finite() && fps > 0.0 { fps } else { 30.0 };
    let frame = |x: f64| (x.max(0.0) * fps).round() / fps;
    let mut lines: Vec<&Cue> = lines.iter().filter(|l| l.start.is_finite() && l.end.is_finite() && l.end > l.start && !l.text.trim().is_empty()).collect();
    lines.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut out: Vec<Cue> = Vec::new();
    for (k, l) in lines.iter().enumerate() {
        let next_line = lines.get(k + 1).map_or(f64::INFINITY, |n| frame(n.start));
        let start = frame(l.start).max(out.last().map_or(0.0, |c| c.end));
        let end = frame(l.end).min(next_line).max(start);
        let mut cs: Vec<Cue> = split_line(&l.text, start, end).into_iter().map(|c| Cue { start: frame(c.start), end: frame(c.end), ..c }).collect();
        // too short to read: longer, at the cost of the next cue of the line (which keeps at
        // least MIN_CUE), or into the pause after the line
        for i in 0..cs.len() {
            let Some(c) = cs.get(i) else { break };
            if c.end - c.start >= MIN_CUE - 1e-9 {
                continue;
            }
            let limit = match cs.get(i + 1) {
                Some(n) => (n.end - MIN_CUE).max(c.end),
                None => next_line,
            };
            let new_end = frame((c.start + MIN_CUE).min(limit)).max(c.end);
            if let Some(c) = cs.get_mut(i) {
                c.end = new_end;
            }
            if let Some(n) = cs.get_mut(i + 1) {
                n.start = n.start.max(new_end);
            }
        }
        out.extend(cs.into_iter().filter(|c| c.end > c.start));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, start: f64, end: f64) -> Cue {
        Cue { start, end, text: text.into() }
    }

    #[test]
    fn wrapping_balances_two_lines() {
        assert_eq!(wrap("Short one."), Some("Short one.".into()));
        let w = wrap("I met Spot at an expo in Dubai and it danced for the crowd.").unwrap();
        let ls: Vec<&str> = w.lines().collect();
        assert_eq!(ls.len(), 2);
        assert!(ls.iter().all(|l| l.chars().count() <= MAX_LINE), "{w}");
        assert!((ls[0].chars().count() as i64 - ls[1].chars().count() as i64).abs() <= 8, "balanced: {w}");
        assert_eq!(wrap(&"word ".repeat(30)), None, "more than two lines");
        // a word longer than a line stays whole
        let long = "x".repeat(60);
        assert_eq!(wrap(&long), Some(long.clone()));
        // a break keeps names together and prefers a clause end to perfect balance
        assert_eq!(
            wrap("In September 2020 I built this Zaku II model kit, one tiny part at a time.").as_deref(),
            Some("In September 2020 I built this Zaku II\nmodel kit, one tiny part at a time.")
        );
        assert_eq!(
            wrap("This is every robot I have met, in the order I met them.").as_deref(),
            Some("This is every robot I have met,\nin the order I met them.")
        );
        assert_eq!(wrap("Then at the stand of DP World in Dubai I met this one.").as_deref().map(|w| w.contains("DP\nWorld")), Some(false));
        // multi-byte characters count as one
        assert!(wrap("Zürich, München, Genève — und Köln! Ünd noch ein paar Wörter.").is_some_and(|w| w.lines().all(|l| l.chars().count() <= MAX_LINE)));
    }

    #[test]
    fn lines_split_at_sentences_then_clauses() {
        let text = "This is the Zaku II, a model kit I built in March. It took me three weekends, and the paint took one more.";
        let cs = cues(&[line(text, 10.0, 17.0)], 30.0);
        assert!(cs.len() >= 2, "{cs:#?}");
        assert!(cs[0].text.replace('\n', " ").ends_with("March."), "a sentence per cue: {cs:#?}");
        for c in &cs {
            assert!(c.text.lines().count() <= 2 && c.text.lines().all(|l| l.chars().count() <= MAX_LINE), "{c:?}");
            assert!(c.end - c.start >= MIN_CUE - 1e-9 && c.end - c.start <= MAX_CUE + 1e-9, "{c:?}");
            assert!(((c.start * 30.0).round() - c.start * 30.0).abs() < 1e-6, "on frames: {c:?}");
        }
        assert!((cs[0].start - 10.0).abs() < 1e-9 && (cs.last().unwrap().end - 17.0).abs() < 1e-9);
        for w in cs.windows(2) {
            assert!(w[0].end <= w[1].start + 1e-9, "{w:?}");
        }
        // the words come out as they went in
        let back: Vec<String> = cs.iter().map(|c| c.text.replace('\n', " ")).collect();
        assert_eq!(back.join(" "), text);
        // timed by characters: the longer second sentence gets more time
        let first_len = cs[0].end - cs[0].start;
        assert!(first_len < 3.5, "{first_len}");
    }

    #[test]
    fn a_long_sentence_breaks_at_a_clause_and_no_cue_outstays_seven_seconds() {
        let text = "When I walked into the hall that morning, the first thing I saw was a robot arm painting portraits of the visitors, one stroke at a time, while a crowd of kids watched it in total silence.";
        let cs = cues(&[line(text, 0.0, 14.0)], 25.0);
        assert!(cs.iter().all(|c| c.end - c.start <= MAX_CUE + 1e-9), "{cs:#?}");
        assert!(cs[0].text.replace('\n', " ").ends_with(','), "cut at a clause: {cs:#?}");
        // a slow line of few words: still at most 7 s a cue
        let cs = cues(&[line("Spot. Atlas. Pepper. Digit.", 0.0, 20.0)], 30.0);
        assert!(cs.len() >= 3 && cs.iter().all(|c| c.end - c.start <= MAX_CUE + 1e-9), "{cs:#?}");
    }

    #[test]
    fn short_lines_stay_up_a_second_without_running_into_the_next() {
        // "One." spoken in 0.4 s, the next line 2 s later: up for a second
        let cs = cues(&[line("One.", 5.0, 5.4), line("Two.", 7.0, 7.3)], 30.0);
        assert_eq!(cs.len(), 2);
        assert!((cs[0].end - 6.0).abs() < 1e-9 && (cs[1].end - 8.0).abs() < 1e-9, "{cs:?}");
        // the next line follows at once: the cue ends where it starts
        let cs = cues(&[line("Hi.", 5.0, 5.3), line("And here is the next one.", 5.5, 7.0)], 30.0);
        assert!((cs[0].end - 5.5).abs() < 1e-9 && (cs[1].start - 5.5).abs() < 1e-9, "{cs:?}");
    }

    #[test]
    fn hostile_lines_give_nothing_or_something_sane() {
        assert!(cues(&[], 30.0).is_empty());
        let bad = [line("", 0.0, 2.0), line("x", f64::NAN, 2.0), line("y", 3.0, 1.0), line("z", 0.0, f64::INFINITY)];
        assert!(cues(&bad, 30.0).is_empty());
        // overlapping lines (out of order) come out in order without overlaps
        let cs = cues(&[line("Second line here.", 2.0, 4.0), line("First line, longer than it should be.", 0.0, 3.0)], f64::NAN);
        for w in cs.windows(2) {
            assert!(w[0].end <= w[1].start + 1e-9, "{cs:?}");
        }
        // one enormous word and a very long line
        let cs = cues(&[line(&"a".repeat(500), 0.0, 3.0)], 30.0);
        assert_eq!(cs.len(), 1);
        let long = "word ".repeat(10_000);
        let cs = cues(&[line(&long, 0.0, 3600.0)], 30.0);
        assert!(!cs.is_empty() && cs.iter().all(|c| c.text.lines().count() <= 2));
    }
}
