//! Understanding a raw talking clip from its transcript and audio ([`analyse`]): speech segments,
//! pauses, filler words, hesitations and retakes, with the take to keep. No model here: the
//! transcript (Whisper), the voice activity (Silero) and the speaker labels come in.
//!
//! - **Segments**: words grouped into sentences: a new segment starts after sentence-ending
//!   punctuation, at a pause of [`Params::segment_gap`], at a change of speaker, or after
//!   [`Params::max_segment`].
//! - **Fillers** ([`fillers_for`]): English hesitation words (um, uh, erm, hmm…) and, for
//!   Malayalam, the hesitation sounds and the discourse fillers its speakers lean on ("ആ", "ഉം",
//!   "എന്ന് പറഞ്ഞാൽ", "അതായത്", and their Manglish spellings "aa", "ennu paranjaal",
//!   "athayath"). The short sounds that are also words ("ആ" is also "that") count only when a
//!   pause sets them apart. **Hesitations**: speech that the voice activity detector heard between
//!   words but the recogniser wrote nothing for (Whisper leaves most "uh"s out) — reported, and
//!   removed only on request.
//! - **Pauses**: silences between words of at least [`Params::min_pause`] that the voice activity
//!   detector agrees are silent.
//! - **Retakes**: a segment that says (nearly) what one of the few before it said (character
//!   similarity of the texts without fillers at least [`Params::repeat_similarity`]), or a short
//!   segment whose words are how the next one starts (a false start, prefix similarity at least
//!   [`Params::false_start_similarity`]). Chains of takes are one retake; the take kept is the
//!   most fluent ([`fluency`]: confident words, few fillers and restarts, no clipping; an
//!   abandoned start never wins), the later take winning a near tie (people redo a line until it
//!   comes out right).

use std::ops::Range;

use filmcraft_project::speech::{Filler, Retake, ShotHint, Span, SpeechAnalysis, SpeechSegment};
use filmcraft_project::{Transcript, Word};
use filmcraft_time::Tick;

use crate::{SAMPLE_RATE, TICKS_PER_SAMPLE, seconds_tick};

/// Tunables.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Shortest pause reported, seconds.
    pub min_pause: f64,
    /// A pause this long ends a segment, seconds.
    pub segment_gap: f64,
    /// Longest segment, seconds.
    pub max_segment: f64,
    /// Similarity at which a segment repeats an earlier one.
    pub repeat_similarity: f32,
    /// Prefix similarity at which a short segment is a false start of the next.
    pub false_start_similarity: f32,
    /// Segments looked back at for a retake.
    pub look_back: usize,
    /// Longest gap between two takes of one line, seconds.
    pub max_retake_gap: f64,
    /// Filler words and phrases; empty: [`fillers_for`] the transcript's language.
    pub fillers: Vec<String>,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            min_pause: 0.5,
            segment_gap: 0.6,
            max_segment: 20.0,
            repeat_similarity: 0.6,
            false_start_similarity: 0.75,
            look_back: 3,
            max_retake_gap: 30.0,
            fillers: Vec::new(),
        }
    }
}

/// English filler words (as `transcript.removeFillers` removes by default).
pub const ENGLISH_FILLERS: &[&str] = &["um", "uh", "umm", "uhm", "erm", "er", "ah", "hmm", "mm", "mhm"];

/// Malayalam fillers, in Malayalam script and in Manglish: hesitation sounds and the discourse
/// fillers "എന്ന് പറഞ്ഞാൽ" ("that is to say") and "അതായത്" ("that is").
pub const MALAYALAM_FILLERS: &[&str] = &[
    "ആ",
    "ആആ",
    "ആഹ്",
    "ഉം",
    "ഉംം",
    "ഹ്മ്മ്",
    "ഹ്മ്",
    "എന്ന് പറഞ്ഞാൽ",
    "എന്നു പറഞ്ഞാൽ",
    "അതായത്",
    "aa",
    "aah",
    "hmm",
    "umm",
    "ennu paranjaal",
    "ennu paranjal",
    "athayath",
    "athaayath",
    "athayathu",
];

/// Fillers that are also ordinary words ("ആ" is "that"): they count only when a pause of at
/// least [`ISOLATED_GAP`] seconds sets them apart from the words before or after.
const AMBIGUOUS: &[&str] = &["ആ", "aa", "ah", "er", "mm"];
const ISOLATED_GAP: f64 = 0.25;

/// The filler list for `language` (ISO 639-1): English's, plus Malayalam's for `ml`.
pub fn fillers_for(language: &str) -> Vec<String> {
    let mut v: Vec<String> = ENGLISH_FILLERS.iter().map(|s| s.to_string()).collect();
    if language.eq_ignore_ascii_case("ml") {
        v.extend(MALAYALAM_FILLERS.iter().map(|s| s.to_string()));
    }
    v.sort();
    v.dedup();
    v
}

/// A word as compared: lower case, punctuation trimmed (Malayalam vowel signs and viramas kept),
/// zero-width joiners dropped.
fn norm(s: &str) -> String {
    s.trim_matches(|c: char| !is_letter(c)).to_lowercase().chars().filter(|c| !matches!(c, '\u{200C}' | '\u{200D}')).collect()
}

/// A letter or an Indic vowel sign, virama or anusvara (kept when texts are compared).
fn is_letter(c: char) -> bool {
    let cp = c as u32;
    c.is_alphanumeric() || ((0x0900..0x0E00).contains(&cp) && !matches!(c, '\u{0964}' | '\u{0965}'))
}

/// Longest text compared, characters (the edit distance is quadratic).
const MAX_COMPARED: usize = 600;

fn letters(words: &[String]) -> Vec<char> {
    let mut v: Vec<char> = words.iter().flat_map(|w| w.chars()).filter(|c| is_letter(*c)).collect();
    v.truncate(MAX_COMPARED);
    v
}

fn edit_distance(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(ca != cb)).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Similarity of two texts' letters, 0..1 (1 − edit distance / the longer length).
pub fn similarity(a: &[char], b: &[char]) -> f32 {
    let n = a.len().max(b.len());
    if n == 0 {
        return 0.0;
    }
    1.0 - edit_distance(a, b) as f32 / n as f32
}

/// How well the shorter `a` matches the start of `b`, 0..1.
pub fn prefix_similarity(a: &[char], b: &[char]) -> f32 {
    if a.is_empty() || a.len() > b.len() {
        return 0.0;
    }
    // the best end point of the prefix within ±20 % of `a`'s length
    let slack = a.len() / 5 + 1;
    let lo = a.len().saturating_sub(slack);
    let hi = (a.len() + slack).min(b.len());
    (lo.max(1)..=hi).map(|k| similarity(a, b.get(..k).unwrap_or(b))).fold(0.0, f32::max)
}

/// Where the filler phrases of `fillers` are in `words`: word index ranges. Ambiguous short
/// fillers count only when a pause sets them apart.
pub fn find_fillers(words: &[Word], fillers: &[String]) -> Vec<Range<usize>> {
    let normed: Vec<String> = words.iter().map(|w| norm(&w.text)).collect();
    let mut phrases: Vec<Vec<String>> =
        fillers.iter().map(|f| f.split_whitespace().map(norm).filter(|s| !s.is_empty()).collect::<Vec<_>>()).filter(|p| !p.is_empty()).collect();
    phrases.sort_by_key(|p| std::cmp::Reverse(p.len()));
    let gap = seconds_tick(ISOLATED_GAP);
    let isolated = |i: usize, j: usize| -> bool {
        let before = i == 0 || words.get(i).zip(words.get(i - 1)).is_some_and(|(w, p)| w.start - p.end >= gap);
        let after = j >= words.len() || words.get(j).zip(words.get(j - 1)).is_some_and(|(n, w)| n.start - w.end >= gap);
        before || after
    };
    let mut out = Vec::new();
    let mut i = 0;
    while i < normed.len() {
        let hit = phrases.iter().find(|p| i + p.len() <= normed.len() && p.iter().enumerate().all(|(k, w)| normed[i + k] == *w));
        match hit {
            Some(p) if !(p.len() == 1 && AMBIGUOUS.contains(&p[0].as_str())) || isolated(i, i + 1) => {
                out.push(i..i + p.len());
                i += p.len();
            }
            _ => i += 1,
        }
    }
    out
}

/// Word index ranges of the segments of `words` (see the module docs).
pub fn segment_words(words: &[Word], p: &Params) -> Vec<Range<usize>> {
    let gap = seconds_tick(p.segment_gap.clamp(0.05, 60.0));
    let max = seconds_tick(p.max_segment.clamp(1.0, 600.0));
    let mut out = Vec::new();
    let mut a = 0;
    for i in 1..=words.len() {
        let split = i == words.len() || {
            let (prev, w) = (&words[i - 1], &words[i]);
            prev.text.ends_with(['.', '?', '!', '।']) || w.start - prev.end >= gap || w.speaker != prev.speaker || w.end - words[a].start > max
        };
        if split {
            if a < i {
                out.push(a..i);
            }
            a = i;
        }
    }
    out
}

/// Words said twice in a row ("the the").
fn restarts(words: &[Word]) -> u32 {
    let n: Vec<String> = words.iter().map(|w| norm(&w.text)).collect();
    n.windows(2).filter(|w| !w[0].is_empty() && w[0] == w[1]).count() as u32
}

/// Whether `audio` (mono 16 kHz) clips between samples `a` and `b`: three full-scale samples in a
/// row.
fn clipped(audio: &[f32], a: usize, b: usize) -> bool {
    let s = audio.get(a.min(audio.len())..b.min(audio.len())).unwrap_or(&[]);
    let mut run = 0;
    for v in s {
        run = if v.abs() >= 0.999 { run + 1 } else { 0 };
        if run >= 3 {
            return true;
        }
    }
    false
}

/// How cleanly a segment was said, 0..1: its words' mean confidence, less 0.08 a filler, 0.12 a
/// restart and 0.25 for clipping.
pub fn fluency(confidence: f32, fillers: u32, restarts: u32, clipped: bool) -> f32 {
    let c = if confidence.is_finite() { confidence.clamp(0.0, 1.0) } else { 0.0 };
    (c - 0.08 * fillers as f32 - 0.12 * restarts as f32 - if clipped { 0.25 } else { 0.0 }).clamp(0.0, 1.0)
}

fn sample(t: Tick) -> usize {
    (t.0.max(0) / TICKS_PER_SAMPLE) as usize
}

/// Understand a clip: `t` (normalised, word times in the clip's media time), its mono 16 kHz
/// `audio`, the voice activity `speech` (sample ranges) and how it was found (`vad`:
/// `silero`/`energy`), and how speakers were labelled (`diarization`).
pub fn analyse(t: &Transcript, audio: &[f32], speech: &[Range<usize>], vad: &str, diarization: &str, p: &Params) -> SpeechAnalysis {
    let words = &t.words;
    let fillers_list = if p.fillers.is_empty() { fillers_for(&t.language) } else { p.fillers.clone() };
    let filler_hits = find_fillers(words, &fillers_list);
    let mut is_filler = vec![false; words.len()];
    for h in &filler_hits {
        for k in h.clone() {
            if let Some(f) = is_filler.get_mut(k) {
                *f = true;
            }
        }
    }
    let speech_spans: Vec<Span> = speech.iter().map(|r| Span::new(crate::sample_tick(r.start as i64), crate::sample_tick(r.end as i64))).collect();
    // speech (by the detector) inside a media range, in ticks
    let voiced = |a: Tick, b: Tick| -> Tick { speech_spans.iter().map(|s| (s.end.min(b) - s.start.max(a)).max(Tick::ZERO)).fold(Tick::ZERO, |x, y| x + y) };
    // segments
    let ranges = segment_words(words, p);
    let mut segments: Vec<SpeechSegment> = ranges
        .iter()
        .map(|r| {
            let ws = &words[r.clone()];
            let (first, last) = (&ws[0], &ws[ws.len() - 1]);
            let conf = ws.iter().map(|w| w.confidence).sum::<f32>() / ws.len().max(1) as f32;
            let nf = r.clone().filter(|&k| is_filler[k]).count() as u32;
            let rs = restarts(ws);
            let cl = clipped(audio, sample(first.start), sample(last.end));
            SpeechSegment {
                start: first.start,
                end: last.end,
                speaker: first.speaker,
                text: ws.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" "),
                from: r.start,
                to: r.end - 1,
                confidence: conf,
                fluency: fluency(conf, nf, rs, cl),
                fillers: nf,
                restarts: rs,
                clipped: cl,
            }
        })
        .collect();
    // fillers of the transcript, then hesitations (speech between words with no word)
    let mut fillers: Vec<Filler> = filler_hits
        .iter()
        .filter_map(|h| {
            let (a, b) = (words.get(h.start)?, words.get(h.end - 1)?);
            Some(Filler {
                start: a.start,
                end: b.end,
                text: words[h.clone()].iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" "),
                from: Some(h.start),
                to: Some(h.end - 1),
            })
        })
        .collect();
    if vad == "silero" {
        let margin = seconds_tick(0.08);
        for pair in words.windows(2) {
            let (a, b) = (pair[0].end + margin, pair[1].start - margin);
            if b <= a {
                continue;
            }
            for s in &speech_spans {
                let (x, y) = (s.start.max(a), s.end.min(b));
                let d = y - x;
                if d >= seconds_tick(0.15) && d <= seconds_tick(1.2) {
                    fillers.push(Filler { start: x, end: y, text: "(hesitation)".into(), from: None, to: None });
                }
            }
        }
    }
    fillers.sort_by_key(|f| f.start);
    // pauses: gaps between words the detector agrees are (mostly) silent
    let min_pause = seconds_tick(p.min_pause.clamp(0.05, 60.0));
    let pauses: Vec<Span> = words
        .windows(2)
        .filter_map(|w| {
            let (a, b) = (w[0].end, w[1].start);
            if b - a < min_pause {
                return None;
            }
            let v = voiced(a, b);
            // at most a fifth of it heard as speech
            (v.0.saturating_mul(5) <= (b - a).0).then(|| Span::new(a, b))
        })
        .collect();
    // retakes
    let texts: Vec<Vec<char>> = ranges
        .iter()
        .map(|r| {
            let ws: Vec<String> = r.clone().filter(|&k| !is_filler[k]).filter_map(|k| words.get(k)).map(|w| norm(&w.text)).collect();
            letters(&ws)
        })
        .collect();
    let max_gap = seconds_tick(p.max_retake_gap.clamp(0.0, 600.0));
    // links: (earlier, later, similarity, false start)
    let mut links: Vec<(usize, usize, f32, bool)> = Vec::new();
    for i in 1..segments.len() {
        for j in i.saturating_sub(p.look_back.clamp(1, 10))..i {
            let (a, b) = (&segments[j], &segments[i]);
            if b.start - a.end > max_gap || (a.speaker.is_some() && b.speaker.is_some() && a.speaker != b.speaker) {
                continue;
            }
            let (ta, tb) = (&texts[j], &texts[i]);
            if ta.len() < 4 || tb.len() < 4 {
                continue;
            }
            let ratio = ta.len().min(tb.len()) as f32 / ta.len().max(tb.len()) as f32;
            let full = similarity(ta, tb);
            if ratio >= 0.6 && full >= p.repeat_similarity {
                links.push((j, i, full, false));
            } else if ta.len() >= 8 && ta.len() < tb.len() {
                let pre = prefix_similarity(ta, tb);
                if pre >= p.false_start_similarity {
                    links.push((j, i, pre, true));
                }
            }
        }
    }
    // chains of linked takes
    let mut group: Vec<usize> = (0..segments.len()).collect();
    fn root(g: &[usize], mut x: usize) -> usize {
        let mut guard = 0;
        while g[x] != x && guard < g.len() {
            x = g[x];
            guard += 1;
        }
        x
    }
    for &(a, b, _, _) in &links {
        let (ra, rb) = (root(&group, a), root(&group, b));
        if ra != rb {
            group[rb.max(ra)] = ra.min(rb);
        }
    }
    let mut retakes: Vec<Retake> = Vec::new();
    let mut seen = vec![false; segments.len()];
    for i in 0..segments.len() {
        let r = root(&group, i);
        if seen[r] {
            continue;
        }
        let takes: Vec<usize> = (0..segments.len()).filter(|&k| root(&group, k) == r).collect();
        for &k in &takes {
            seen[k] = true;
        }
        seen[r] = true;
        if takes.len() < 2 {
            continue;
        }
        let in_group: Vec<&(usize, usize, f32, bool)> = links.iter().filter(|l| takes.contains(&l.0)).collect();
        // an abandoned start never wins
        let false_starts: Vec<usize> = in_group.iter().filter(|l| l.3).map(|l| l.0).collect();
        let score = |pos: usize, k: usize| -> f32 {
            let s = &segments[k];
            s.fluency + 0.04 * pos as f32 - if false_starts.contains(&k) { 1.0 } else { 0.0 }
        };
        let keep = takes.iter().enumerate().max_by(|(pa, a), (pb, b)| score(*pa, **a).total_cmp(&score(*pb, **b))).map_or(takes[takes.len() - 1], |(_, k)| *k);
        let similarity = in_group.iter().map(|l| l.2).fold(1.0f32, f32::min);
        let reason = if in_group.iter().all(|l| l.3) { "falseStart" } else { "repeat" };
        retakes.push(Retake { takes, keep, similarity, reason: reason.into() });
    }
    // segments carry their speaker; the clip's length
    let duration = crate::sample_tick(audio.len() as i64);
    for s in &mut segments {
        s.text = s.text.trim().to_string();
    }
    let share = {
        let total: i64 = speech_spans.iter().map(|s| s.duration().0).sum();
        if duration.0 > 0 { total as f64 / duration.0 as f64 } else { 0.0 }
    };
    let shot = if share < 0.15 && words.len() < 5 { ShotHint::BRoll } else { ShotHint::Unknown };
    SpeechAnalysis {
        language: t.language.clone(),
        source: t.source.clone(),
        vad: vad.to_string(),
        diarization: diarization.to_string(),
        duration,
        speech: speech_spans,
        segments,
        pauses,
        fillers,
        retakes,
        shot,
        face: None,
    }
}

/// Samples per second (for callers converting spans).
pub const RATE: u32 = SAMPLE_RATE;

#[cfg(test)]
mod tests {
    use super::*;

    fn words(spec: &[(&str, f64, f64)]) -> Vec<Word> {
        spec.iter().map(|(t, a, b)| Word { confidence: 0.9, ..Word::new(*t, seconds_tick(*a), seconds_tick(*b)) }).collect()
    }

    fn transcript(spec: &[(&str, f64, f64)], lang: &str) -> Transcript {
        let mut t = Transcript { language: lang.into(), source: "test".into(), words: words(spec), ..Default::default() };
        t.normalize();
        t
    }

    #[test]
    fn a_repeated_line_keeps_the_last_clean_take() {
        let t = transcript(
            &[
                ("Today", 0.0, 0.3),
                ("we", 0.3, 0.45),
                ("look", 0.45, 0.7),
                ("at", 0.7, 0.8),
                ("the", 0.8, 0.9),
                ("um", 0.95, 1.2),
                ("phone.", 1.3, 1.7),
                ("Today", 2.6, 2.9),
                ("we", 2.9, 3.05),
                ("look", 3.05, 3.3),
                ("at", 3.3, 3.4),
                ("the", 3.4, 3.5),
                ("phone.", 3.5, 3.9),
                ("It", 4.8, 5.0),
                ("is", 5.0, 5.2),
                ("fast.", 5.2, 5.6),
            ],
            "en",
        );
        let a = analyse(&t, &[], &[], "energy", "off", &Params::default());
        assert_eq!(a.segments.len(), 3);
        assert_eq!(a.retakes.len(), 1, "{:?}", a.retakes);
        let r = &a.retakes[0];
        assert_eq!(r.takes, vec![0, 1]);
        assert_eq!(r.keep, 1, "the take without the filler");
        assert_eq!(r.reason, "repeat");
        assert_eq!(a.dropped_takes(), vec![0]);
        assert_eq!(a.fillers.len(), 1);
        assert_eq!(a.fillers[0].from, Some(5));
        // pauses between the takes and before the last line
        assert_eq!(a.pauses.len(), 2);
        a.check(t.words.len()).unwrap();
    }

    #[test]
    fn a_false_start_is_dropped() {
        let t = transcript(
            &[
                ("So", 0.0, 0.2),
                ("the", 0.2, 0.3),
                ("new", 0.3, 0.5),
                ("camera", 0.5, 0.9),
                ("So", 2.0, 2.2),
                ("the", 2.2, 2.3),
                ("new", 2.3, 2.5),
                ("camera", 2.5, 2.9),
                ("shoots", 2.9, 3.2),
                ("in", 3.2, 3.3),
                ("eight", 3.3, 3.6),
                ("K.", 3.6, 3.9),
            ],
            "en",
        );
        let a = analyse(&t, &[], &[], "energy", "off", &Params::default());
        assert_eq!(a.retakes.len(), 1, "{:?}", a.segments);
        assert_eq!(a.retakes[0].reason, "falseStart");
        assert_eq!(a.retakes[0].keep, 1);
    }

    #[test]
    fn different_lines_are_not_retakes() {
        let t = transcript(
            &[
                ("The", 0.0, 0.2),
                ("battery", 0.2, 0.6),
                ("lasts", 0.6, 0.9),
                ("two", 0.9, 1.1),
                ("days.", 1.1, 1.5),
                ("The", 2.5, 2.7),
                ("screen", 2.7, 3.0),
                ("is", 3.0, 3.1),
                ("very", 3.1, 3.3),
                ("bright.", 3.3, 3.7),
            ],
            "en",
        );
        assert!(analyse(&t, &[], &[], "energy", "off", &Params::default()).retakes.is_empty());
    }

    #[test]
    fn malayalam_fillers_and_retakes() {
        // "ആ" isolated by a pause is a filler; "ആ വീട്" (that house) is not
        let t = transcript(
            &[
                ("ആ", 0.0, 0.3),
                ("ഇന്ന്", 0.8, 1.2),
                ("നമ്മൾ", 1.2, 1.6),
                ("ആ", 1.6, 1.7),
                ("വീട്", 1.7, 2.0),
                ("കാണും.", 2.0, 2.5),
                ("അതായത്", 3.5, 4.0),
                ("ഇന്ന്", 4.6, 5.0),
                ("നമ്മൾ", 5.0, 5.4),
                ("ആ", 5.4, 5.5),
                ("വീട്", 5.5, 5.8),
                ("കാണും.", 5.8, 6.3),
            ],
            "ml",
        );
        let a = analyse(&t, &[], &[], "energy", "off", &Params::default());
        let texts: Vec<&str> = a.fillers.iter().map(|f| f.text.as_str()).collect();
        assert_eq!(texts, vec!["ആ", "അതായത്"], "{texts:?}");
        assert_eq!(a.retakes.len(), 1, "{:?}", a.segments.iter().map(|s| &s.text).collect::<Vec<_>>());
        // Manglish spellings too
        let m = transcript(&[("athayath", 0.0, 0.4), ("ennu", 1.0, 1.2), ("paranjaal", 1.2, 1.6), ("phone", 1.7, 2.0)], "ml");
        let a = analyse(&m, &[], &[], "energy", "off", &Params::default());
        assert_eq!(a.fillers.len(), 2);
        assert_eq!((a.fillers[1].from, a.fillers[1].to), (Some(1), Some(2)));
    }

    #[test]
    fn hesitations_come_from_unworded_speech() {
        let t = transcript(&[("one", 0.0, 0.4), ("two", 2.0, 2.4)], "en");
        let sr = SAMPLE_RATE as usize;
        // speech 0–0.4, a sound at 1.0–1.4 nobody wrote down, speech 2.0–2.4
        let speech = vec![0..sr * 4 / 10, sr..sr * 14 / 10, 2 * sr..sr * 24 / 10];
        let a = analyse(&t, &vec![0.0; 3 * sr], &speech, "silero", "off", &Params::default());
        assert_eq!(a.fillers.len(), 1);
        assert!(a.fillers[0].is_hesitation());
        // the gap holds speech: not a pause
        assert!(a.pauses.is_empty());
        // without Silero no hesitations are guessed
        assert!(analyse(&t, &vec![0.0; 3 * sr], &speech, "energy", "off", &Params::default()).fillers.is_empty());
    }

    #[test]
    fn hostile_inputs() {
        let empty = Transcript::default();
        let a = analyse(&empty, &[], &[], "energy", "off", &Params::default());
        assert!(a.segments.is_empty() && a.retakes.is_empty());
        assert_eq!(a.shot, ShotHint::BRoll);
        // NaN confidence, absurd params, huge times
        let mut t = transcript(&[("a", 0.0, 1.0), ("a", 1.0, 2.0)], "en");
        t.words[0].confidence = f32::NAN;
        let weird = Params {
            min_pause: f64::NAN,
            segment_gap: -1.0,
            max_segment: f64::INFINITY,
            look_back: usize::MAX,
            max_retake_gap: f64::NAN,
            ..Default::default()
        };
        let all = 0..usize::MAX;
        let a = analyse(&t, &[f32::NAN; 10], std::slice::from_ref(&all), "silero", "off", &weird);
        assert!(a.segments.iter().all(|s| s.fluency.is_finite()));
        let far = transcript(&[("x", 1e7, 1e7 + 1.0)], "en");
        let _ = analyse(&far, &[], &[], "energy", "off", &Params::default());
        assert_eq!(similarity(&[], &[]), 0.0);
        assert_eq!(prefix_similarity(&['a'], &[]), 0.0);
    }

    #[test]
    fn fluency_prefers_clean_takes() {
        assert!(fluency(0.9, 0, 0, false) > fluency(0.9, 1, 0, false));
        assert!(fluency(0.9, 0, 0, false) > fluency(0.9, 0, 1, false));
        assert!(fluency(0.9, 0, 0, false) > fluency(0.9, 0, 0, true));
        assert_eq!(fluency(f32::NAN, 0, 0, false), 0.0);
        assert!(clipped(&[0.0, 1.0, 1.0, 1.0, 0.0], 0, 5));
        assert!(!clipped(&[0.0, 1.0, 0.5, 1.0, 0.0], 0, 5));
        assert!(!clipped(&[1.0; 5], 10, 2));
    }
}
