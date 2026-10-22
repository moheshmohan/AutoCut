//! A rough cut of a talking clip: the media ranges to keep once pauses, filler words and
//! abandoned takes are cut out (`sequence.roughCut`, `autocut.roughCut`).
//!
//! The clip's transcript gives the words; its speech analysis
//! ([`filmcraft_project::SpeechAnalysis`]) the fillers and the retakes. Words are removed when
//! they are fillers (`remove_fillers`) or belong to a take that a retake drops
//! (`remove_retakes`). The kept words form **runs**: a run ends at a removed word and, with
//! `remove_pauses`, at a silence of at least `min_pause` between kept words. Each run becomes one
//! media range:
//!
//! - it reaches `keep` into the silence on both sides, but never into a neighbouring word (kept
//!   or not) or a hesitation being removed, so **every cut lands between words**;
//! - its start is snapped down to a frame (up when that would reach into the word before) and its
//!   length is a whole number of frames that does not reach into the next word, so the clips
//!   placed back to back leave no gaps.

use std::ops::Range;

use filmcraft_project::{SpeechAnalysis, Transcript};
use filmcraft_time::{FrameRate, Tick, TimeRange};

/// What a rough cut takes out.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub remove_pauses: bool,
    pub remove_fillers: bool,
    pub remove_retakes: bool,
    /// Also cut hesitations (speech the recogniser wrote no word for) out of the silences kept.
    pub remove_hesitations: bool,
    /// Silences at least this long between kept words are cut.
    pub min_pause: Tick,
    /// Silence kept next to the words at each cut.
    pub keep: Tick,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            remove_pauses: true,
            remove_fillers: true,
            remove_retakes: true,
            remove_hesitations: false,
            min_pause: Tick::from_seconds_f64(0.6),
            keep: Tick::from_seconds_f64(0.12),
        }
    }
}

/// A media range kept and the words (indices into the transcript) it holds.
#[derive(Clone, Debug, PartialEq)]
pub struct Kept {
    pub range: TimeRange,
    pub words: Range<usize>,
}

/// What was taken out.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Removed {
    pub pauses: usize,
    pub fillers: usize,
    pub retakes: usize,
    pub words: usize,
}

/// The ranges of `t`'s media to keep (see the module docs), in order. `media_end`: the clip's
/// length (nothing is kept past it).
pub fn keep_ranges(t: &Transcript, a: Option<&SpeechAnalysis>, o: &Options, rate: FrameRate, media_end: Tick) -> (Vec<Kept>, Removed) {
    let words = &t.words;
    let n = words.len();
    let mut removed = vec![false; n];
    let mut stats = Removed::default();
    if let Some(a) = a {
        if o.remove_fillers {
            for f in &a.fillers {
                if let (Some(from), Some(to)) = (f.from, f.to)
                    && from <= to
                    && to < n
                {
                    removed[from..=to].iter_mut().for_each(|r| *r = true);
                    stats.fillers += 1;
                }
            }
        }
        if o.remove_retakes {
            for k in a.dropped_takes() {
                if let Some(s) = a.segments.get(k)
                    && s.from <= s.to
                    && s.to < n
                {
                    removed[s.from..=s.to].iter_mut().for_each(|r| *r = true);
                    stats.retakes += 1;
                }
            }
        }
    }
    stats.words = removed.iter().filter(|r| **r).count();
    let hesitations: Vec<TimeRange> = match a {
        Some(a) if o.remove_hesitations => {
            a.fillers.iter().filter(|f| f.is_hesitation()).map(|f| TimeRange::from_bounds(f.start, f.end.max(f.start))).collect()
        }
        _ => Vec::new(),
    };
    // runs of kept words
    let mut runs: Vec<Range<usize>> = Vec::new();
    let mut cur: Option<Range<usize>> = None;
    for i in 0..n {
        if removed[i] {
            if let Some(r) = cur.take() {
                runs.push(r);
            }
            continue;
        }
        match cur.as_mut() {
            Some(r) if r.end == i && !(o.remove_pauses && words[i].start - words[i - 1].end >= o.min_pause) => r.end = i + 1,
            Some(r) => {
                if o.remove_pauses && r.end == i {
                    stats.pauses += 1;
                }
                runs.push(r.clone());
                cur = Some(i..i + 1);
            }
            None => cur = Some(i..i + 1),
        }
    }
    if let Some(r) = cur {
        runs.push(r);
    }
    let fd = rate.frame_duration().max(Tick(1));
    let keep = o.keep.max(Tick::ZERO);
    let mut out = Vec::new();
    for r in runs {
        let (first, last) = (&words[r.start], &words[r.end - 1]);
        // the silence around the run, up to the neighbouring words (and removed hesitations)
        let mut lo = r.start.checked_sub(1).and_then(|i| words.get(i)).map_or(Tick::ZERO, |w| w.end);
        let mut hi = words.get(r.end).map_or(media_end, |w| w.start).min(media_end);
        for h in &hesitations {
            if h.end() <= first.start {
                lo = lo.max(h.end());
            }
            if h.start >= last.end {
                hi = hi.min(h.start);
            }
        }
        let Some((s, frames)) = fit(first.start, last.end, lo, hi.max(lo), keep, rate) else { continue };
        out.push(Kept { range: TimeRange::new(s, Tick(frames.saturating_mul(fd.0))), words: r });
    }
    (out, stats)
}

/// The start and length in whole frames of a range holding the words `a..b`, reaching `keep`
/// into the silence on both sides and staying within `lo..hi` (the neighbouring words): the
/// longest length that fits, its start nearest `a − keep` (on the frame grid when it can be).
/// Words with less than whole frames of room around them (touching their neighbours) overrun
/// into a neighbour by less than a frame.
fn fit(a: Tick, b: Tick, lo: Tick, hi: Tick, keep: Tick, rate: FrameRate) -> Option<(Tick, i64)> {
    let fd = rate.frame_duration().max(Tick(1)).0;
    if b < a {
        return None;
    }
    let words = (b - a).0;
    let need = ((words + fd - 1) / fd).max(1);
    let want_s = (a - keep).max(lo);
    let want_e = (b + keep).min(hi);
    let pref = ((want_e - want_s).0 / fd).max(need);
    // at most a few frames of padding to try
    for k in (need..=pref.min(need + 64)).rev() {
        let len = k.saturating_mul(fd);
        let lo_s = lo.max(Tick(b.0.saturating_sub(len)));
        let hi_s = a.min(Tick(hi.0.saturating_sub(len)));
        if lo_s > hi_s {
            continue;
        }
        let mut s = want_s.max(lo_s).min(hi_s);
        let g = rate.snap(s);
        if g >= lo_s && g <= hi_s {
            s = g;
        } else if g + Tick(fd) >= lo_s && g + Tick(fd) <= hi_s {
            s = g + Tick(fd);
        }
        return Some((s, k));
    }
    // no room for whole frames between the neighbours: hold the words and end at the next one
    // (or the media's end), reaching less than a frame back into the word before
    let len = need.saturating_mul(fd);
    Some((a.min(Tick(hi.0.saturating_sub(len)).max(Tick::ZERO)), need))
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_project::Word;
    use filmcraft_project::speech::{Filler, Retake, SpeechSegment};

    fn s(x: f64) -> Tick {
        Tick::from_seconds_f64(x)
    }

    fn transcript() -> Transcript {
        let w = |t: &str, a: f64, b: f64| Word::new(t, s(a), s(b));
        Transcript {
            language: "en".into(),
            words: vec![
                w("Today", 0.50, 0.80),
                w("um", 0.90, 1.10),
                w("we", 1.20, 1.35),
                w("test.", 1.35, 1.70),
                w("Today", 3.00, 3.30),
                w("we", 3.30, 3.45),
                w("test.", 3.45, 3.80),
                w("It", 5.20, 5.35),
                w("works.", 5.35, 5.90),
            ],
            ..Default::default()
        }
    }

    fn analysis() -> SpeechAnalysis {
        let seg = |from, to, a: f64, b: f64| SpeechSegment { from, to, start: s(a), end: s(b), ..Default::default() };
        SpeechAnalysis {
            segments: vec![seg(0, 3, 0.5, 1.7), seg(4, 6, 3.0, 3.8), seg(7, 8, 5.2, 5.9)],
            fillers: vec![Filler { start: s(0.9), end: s(1.1), text: "um".into(), from: Some(1), to: Some(1) }],
            retakes: vec![Retake { takes: vec![0, 1], keep: 1, similarity: 0.9, reason: "repeat".into() }],
            ..Default::default()
        }
    }

    /// No cut is inside a word; kept words are inside a kept range, removed ones outside all.
    fn check(t: &Transcript, kept: &[Kept], removed_words: &[usize]) {
        for k in kept {
            for b in [k.range.start, k.range.end()] {
                for w in &t.words {
                    assert!(!(w.start < b && b < w.end), "cut at {b:?} inside {:?}", w.text);
                }
            }
        }
        for (i, w) in t.words.iter().enumerate() {
            let inside = kept.iter().any(|k| k.range.start <= w.start && w.end <= k.range.end());
            assert_eq!(inside, !removed_words.contains(&i), "word {i} {:?}", w.text);
        }
    }

    #[test]
    fn retakes_fillers_and_pauses_are_cut_between_words() {
        let t = transcript();
        let a = analysis();
        let rate = FrameRate::new(25, 1);
        let (kept, st) = keep_ranges(&t, Some(&a), &Options::default(), rate, s(7.0));
        // the first take (with its filler) goes; the pause before "It works." is cut
        assert_eq!(kept.iter().map(|k| k.words.clone()).collect::<Vec<_>>(), vec![4..7, 7..9]);
        assert_eq!((st.retakes, st.fillers, st.pauses), (1, 1, 1));
        check(&t, &kept, &[0, 1, 2, 3]);
        for k in &kept {
            assert_eq!(k.range.duration.0 % rate.frame_duration().0, 0, "whole frames");
        }
        // keeping the retakes: the filler alone is cut, the cuts stay between words
        let o = Options { remove_retakes: false, remove_pauses: false, ..Default::default() };
        let (kept, _) = keep_ranges(&t, Some(&a), &o, rate, s(7.0));
        assert_eq!(kept.iter().map(|k| k.words.clone()).collect::<Vec<_>>(), vec![0..1, 2..9]);
        check(&t, &kept, &[1]);
        // nothing removed: one range
        let none = Options { remove_retakes: false, remove_pauses: false, remove_fillers: false, ..Default::default() };
        let (kept, _) = keep_ranges(&t, Some(&a), &none, rate, s(7.0));
        assert_eq!(kept.len(), 1);
        check(&t, &kept, &[]);
    }

    #[test]
    fn tight_words_and_odd_rates() {
        // words touching each other and the media's end, at 29.97
        let w = |a: f64, b: f64| Word::new("x", s(a), s(b));
        let t = Transcript { words: vec![w(0.0, 0.5), w(0.5, 1.0), w(1.0, 1.013)], ..Default::default() };
        let a = SpeechAnalysis { fillers: vec![Filler { start: s(0.5), end: s(1.0), text: "x".into(), from: Some(1), to: Some(1) }], ..Default::default() };
        let rate = FrameRate::new(30000, 1001);
        let (kept, _) = keep_ranges(&t, Some(&a), &Options::default(), rate, s(1.013));
        // touching words leave no room for whole frames: the kept words stay whole and a cut
        // reaches less than a frame into the removed word
        assert_eq!(kept.len(), 2);
        for k in &kept {
            assert!(t.words[k.words.clone()].iter().all(|w| k.range.start <= w.start && w.end <= k.range.end()));
        }
        assert!(kept[0].range.end() - s(0.5) < rate.frame_duration());
        // with a frame of silence on each side the cuts are exact
        let w2 = |a: f64, b: f64| Word::new("x", s(a), s(b));
        let t2 = Transcript { words: vec![w2(0.0, 0.5), w2(0.6, 0.9), w2(1.0, 1.5)], ..Default::default() };
        let a2 = SpeechAnalysis { fillers: vec![Filler { start: s(0.6), end: s(0.9), text: "x".into(), from: Some(1), to: Some(1) }], ..Default::default() };
        let (kept, _) = keep_ranges(&t2, Some(&a2), &Options::default(), rate, s(1.6));
        check(&t2, &kept, &[1]);
        // hostile: empty, out-of-range analysis indices
        let (k, _) = keep_ranges(&Transcript::default(), None, &Options::default(), rate, Tick::ZERO);
        assert!(k.is_empty());
        let bad = SpeechAnalysis {
            fillers: vec![Filler { from: Some(9), to: Some(99), ..Default::default() }],
            retakes: vec![Retake { takes: vec![0, 5], keep: 0, ..Default::default() }],
            ..Default::default()
        };
        let (k, _) = keep_ranges(&t, Some(&bad), &Options::default(), rate, s(2.0));
        assert!(!k.is_empty());
    }
}
