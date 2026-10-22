//! Neural speaker labelling ("who spoke when") from speaker embeddings.
//!
//! 1. **Where speech is**: voice activity segments ([`crate::vad::speech_segments`]: Silero, else
//!    the energy detector).
//! 2. **Windows**: each segment is cut into windows of [`Params::window`] every [`Params::hop`]
//!    (a shorter segment is one window; segments under [`Params::min_window`] get no window and
//!    take their neighbours' speaker).
//! 3. **Embeddings**: one per window ([`Embedder`]: ECAPA-TDNN, [`crate::ecapa`]).
//! 4. **Clustering** with an automatic speaker count ([`cluster`]): average-linkage
//!    agglomerative clustering on cosine similarity, merging while the closest clusters are more
//!    similar than [`Params::threshold`] (or while there are more than `max_speakers`); clusters
//!    holding less than [`Params::min_share`] of the speech join their nearest; then every window
//!    goes to the nearest cluster centre (one refinement pass).
//! 5. **Turns**: every 10 ms of speech takes the speaker of the window centred nearest to it, the
//!    labels are smoothed with a majority vote over [`Params::smooth`], and turns shorter than
//!    [`Params::min_turn`] join a neighbour.
//! 6. **Words** take the speaker covering most of their span (or the nearest turn).
//!
//! The threshold was tuned on mixes of distinct voices (`examples/diarize_lab.rs`, crate README:
//! "Speaker labels"). The MFCC clustering ([`crate::diarize::diarize`]) is the fallback when the
//! embedding model is not downloaded.

use std::ops::Range;

use filmcraft_project::{Speaker, Transcript};

use crate::{SAMPLE_RATE, SpeechError, TICKS_PER_SAMPLE};

/// Turns speech into speaker embeddings.
pub trait Embedder {
    /// One embedding per clip (mono 16 kHz), all of the same length.
    fn embed_many(&self, clips: &[&[f32]]) -> Result<Vec<Vec<f32>>, SpeechError>;
}

#[cfg(feature = "whisper")]
impl Embedder for crate::ecapa::Ecapa {
    fn embed_many(&self, clips: &[&[f32]]) -> Result<Vec<Vec<f32>>, SpeechError> {
        crate::ecapa::Ecapa::embed_many(self, clips)
    }
}

/// Tunables.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Window length, seconds.
    pub window: f64,
    /// Window step, seconds.
    pub hop: f64,
    /// Shortest segment embedded, seconds.
    pub min_window: f64,
    /// Cosine similarity above which two clusters are one speaker.
    pub threshold: f32,
    pub max_speakers: usize,
    /// Clusters with less of the speech join their nearest neighbour.
    pub min_share: f32,
    /// Majority-vote span, seconds.
    pub smooth: f64,
    /// Shortest turn, seconds.
    pub min_turn: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self { window: 3.0, hop: 1.5, min_window: 0.4, threshold: 0.33, max_speakers: 6, min_share: 0.04, smooth: 0.75, min_turn: 0.6 }
    }
}

/// Most windows embedded and clustered (about 37 minutes of speech at the default step; the
/// clustering is cubic in it): longer audio uses a wider step.
const MAX_WINDOWS: usize = 1500;
/// Labelling frame, samples (10 ms).
const FRAME: usize = (SAMPLE_RATE / 100) as usize;

/// A speaker turn: a sample range and the speaker (0-based, numbered by first appearance).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Turn {
    pub start: usize,
    pub end: usize,
    pub speaker: usize,
}

fn samples(secs: f64) -> usize {
    if secs.is_finite() && secs > 0.0 { (secs.min(3600.0) * f64::from(SAMPLE_RATE)) as usize } else { 0 }
}

/// The windows of `segments`: sample ranges, in order.
pub fn windows(segments: &[Range<usize>], p: &Params) -> Vec<Range<usize>> {
    let total: usize = segments.iter().map(|r| r.len()).sum();
    let win = samples(p.window).max(FRAME);
    let mut hop = samples(p.hop).max(FRAME);
    // cap the work: widen the step on very long audio
    if total / hop > MAX_WINDOWS {
        hop = total / MAX_WINDOWS + 1;
    }
    let min = samples(p.min_window);
    let mut out = Vec::new();
    for s in segments {
        if s.len() < min.max(1) {
            continue;
        }
        if s.len() <= win {
            out.push(s.clone());
            continue;
        }
        let mut a = s.start;
        loop {
            let b = (a + win).min(s.end);
            out.push(a..b);
            if b >= s.end {
                break;
            }
            a += hop;
            // the last window ends at the segment's end
            if a + win > s.end {
                out.push(s.end - win..s.end);
                break;
            }
        }
    }
    out
}

/// Cluster embeddings (see the module docs): a label per embedding, clusters numbered in order of
/// first appearance. `weights`: how much speech each embedding stands for.
pub fn cluster(embs: &[Vec<f32>], weights: &[f32], p: &Params) -> Vec<usize> {
    let n = embs.len();
    if n == 0 {
        return Vec::new();
    }
    // unit vectors (an unusable embedding becomes zeros: similar to nothing)
    let unit: Vec<Vec<f32>> = embs
        .iter()
        .map(|e| {
            let norm = e.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm.is_finite() && norm > 0.0 { e.iter().map(|x| x / norm).collect() } else { vec![0.0; e.len()] }
        })
        .collect();
    let dot = |a: &[f32], b: &[f32]| -> f32 { if a.len() == b.len() { a.iter().zip(b).map(|(x, y)| x * y).sum() } else { 0.0 } };
    // pairwise similarities (n is capped by the window count)
    let mut sim = vec![0f32; n * n];
    for i in 0..n {
        for j in i + 1..n {
            let v = dot(&unit[i], &unit[j]);
            sim[i * n + j] = v;
            sim[j * n + i] = v;
        }
    }
    let max_speakers = p.max_speakers.clamp(1, 64);
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    // average linkage, kept as a matrix of sums for speed
    let mut link: Vec<Vec<f32>> = (0..n).map(|i| (0..n).map(|j| sim[i * n + j]).collect()).collect();
    let avg = |c: &[Vec<usize>], l: &[Vec<f32>], a: usize, b: usize| l[a][b] / (c[a].len() * c[b].len()) as f32;
    while clusters.len() > 1 {
        let mut best = (f32::NEG_INFINITY, 0, 0);
        for a in 0..clusters.len() {
            for b in a + 1..clusters.len() {
                let v = avg(&clusters, &link, a, b);
                if v > best.0 {
                    best = (v, a, b);
                }
            }
        }
        if best.0 < p.threshold && clusters.len() <= max_speakers {
            break;
        }
        // merge b into a (a < b): the link sums add, then b's row and column go
        let (_, a, b) = best;
        let m = clusters.len();
        for k in 0..m {
            let v = link[b][k];
            link[a][k] += v;
        }
        for k in 0..m {
            link[k][a] = link[a][k];
        }
        link.remove(b);
        for r in &mut link {
            r.remove(b);
        }
        let moved = clusters.remove(b);
        clusters[a].extend(moved);
    }
    let w = |i: usize| weights.get(i).copied().filter(|v| v.is_finite() && *v > 0.0).unwrap_or(1.0);
    let total: f32 = (0..n).map(w).sum::<f32>().max(1e-6);
    let centre = |c: &[usize]| -> Vec<f32> {
        let d = unit.first().map_or(0, Vec::len);
        let mut m = vec![0f32; d];
        for &i in c {
            for (k, v) in unit[i].iter().enumerate() {
                if let Some(x) = m.get_mut(k) {
                    *x += v * w(i);
                }
            }
        }
        let norm = m.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 { m.iter().map(|x| x / norm).collect() } else { m }
    };
    // fold small clusters into their nearest
    while clusters.len() > 1 {
        let small = clusters.iter().position(|c| c.iter().map(|&i| w(i)).sum::<f32>() < p.min_share * total);
        let Some(si) = small else { break };
        let s = clusters.remove(si);
        let cs = centre(&s);
        let near = (0..clusters.len()).max_by(|&a, &b| dot(&cs, &centre(&clusters[a])).total_cmp(&dot(&cs, &centre(&clusters[b])))).unwrap_or(0);
        if let Some(c) = clusters.get_mut(near) {
            c.extend(s);
        }
    }
    // one refinement pass: every window to its nearest centre
    let centres: Vec<Vec<f32>> = clusters.iter().map(|c| centre(c)).collect();
    let mut label: Vec<usize> =
        (0..n).map(|i| (0..centres.len()).max_by(|&a, &b| dot(&unit[i], &centres[a]).total_cmp(&dot(&unit[i], &centres[b]))).unwrap_or(0)).collect();
    // a zero embedding keeps its cluster
    for (ci, c) in clusters.iter().enumerate() {
        for &i in c {
            if unit[i].iter().all(|x| *x == 0.0) {
                label[i] = ci;
            }
        }
    }
    renumber(&mut label);
    label
}

/// Number labels in order of first appearance.
fn renumber(label: &mut [usize]) {
    let mut order: Vec<usize> = Vec::new();
    for l in label.iter_mut() {
        let k = match order.iter().position(|o| o == l) {
            Some(k) => k,
            None => {
                order.push(*l);
                order.len() - 1
            }
        };
        *l = k;
    }
}

/// Speaker turns of `audio` (mono 16 kHz) from its speech `segments`, embedded by `embedder`.
pub fn turns(audio: &[f32], segments: &[Range<usize>], embedder: &dyn Embedder, p: &Params) -> Result<Vec<Turn>, SpeechError> {
    let mut segments: Vec<Range<usize>> = segments.iter().map(|r| r.start.min(audio.len())..r.end.min(audio.len())).filter(|r| !r.is_empty()).collect();
    segments.sort_by_key(|r| r.start);
    let wins = windows(&segments, p);
    if wins.is_empty() {
        return Ok(segments.iter().map(|r| Turn { start: r.start, end: r.end, speaker: 0 }).collect());
    }
    // embed windows of equal length together
    let mut embs: Vec<Vec<f32>> = vec![Vec::new(); wins.len()];
    let mut order: Vec<usize> = (0..wins.len()).collect();
    order.sort_by_key(|&i| wins[i].len());
    for group in order.chunk_by(|a, b| wins[*a].len() == wins[*b].len()) {
        let clips: Vec<&[f32]> = group.iter().map(|&i| audio.get(wins[i].clone()).unwrap_or(&[])).collect();
        let e = embedder.embed_many(&clips)?;
        for (k, &i) in group.iter().enumerate() {
            if let (Some(slot), Some(v)) = (embs.get_mut(i), e.get(k)) {
                *slot = v.clone();
            }
        }
    }
    let weights: Vec<f32> = wins.iter().map(|w| w.len() as f32).collect();
    let labels = cluster(&embs, &weights, p);
    Ok(label_frames(&segments, &wins, &labels, p))
}

/// Turns from labelled windows (steps 5 of the module docs).
fn label_frames(segments: &[Range<usize>], wins: &[Range<usize>], labels: &[usize], p: &Params) -> Vec<Turn> {
    let centres: Vec<usize> = wins.iter().map(|w| w.start + w.len() / 2).collect();
    let nearest = |x: usize| -> usize {
        let i = centres.partition_point(|&c| c < x);
        let cand = [i.checked_sub(1), Some(i).filter(|&i| i < centres.len())];
        cand.into_iter().flatten().min_by_key(|&k| centres[k].abs_diff(x)).and_then(|k| labels.get(k).copied()).unwrap_or(0)
    };
    // window centres are sorted only within segments: sort a copy with labels
    let mut out: Vec<Turn> = Vec::new();
    let half = samples(p.smooth) / FRAME / 2;
    for s in segments {
        let frames: Vec<usize> = (s.start..s.end).step_by(FRAME).map(|f| nearest(f + FRAME / 2)).collect();
        // majority vote over the smoothing span
        let n_lab = frames.iter().copied().max().map_or(1, |m| m + 1);
        let smoothed: Vec<usize> = (0..frames.len())
            .map(|i| {
                let a = i.saturating_sub(half);
                let b = (i + half + 1).min(frames.len());
                let mut count = vec![0usize; n_lab];
                for &l in frames.get(a..b).unwrap_or(&[]) {
                    if let Some(c) = count.get_mut(l) {
                        *c += 1;
                    }
                }
                count.iter().enumerate().max_by_key(|(k, c)| (**c, usize::MAX - k)).map_or(0, |(k, _)| k)
            })
            .collect();
        let mut a = 0;
        for i in 1..=smoothed.len() {
            if i == smoothed.len() || smoothed[i] != smoothed[a] {
                let start = s.start + a * FRAME;
                let end = (s.start + i * FRAME).min(s.end);
                if end > start {
                    out.push(Turn { start, end, speaker: smoothed[a] });
                }
                a = i;
            }
        }
    }
    // short turns join the longer neighbour (in the same segment or touching)
    let min = samples(p.min_turn);
    for _ in 0..4 {
        let mut changed = false;
        for i in 0..out.len() {
            if out[i].end - out[i].start >= min {
                continue;
            }
            let touch = |j: usize| out.get(j).is_some_and(|t: &Turn| t.start <= out[i].end + FRAME && out[i].start <= t.end + FRAME);
            let prev = i.checked_sub(1).filter(|&j| touch(j));
            let next = Some(i + 1).filter(|&j| touch(j));
            let pick = match (prev, next) {
                (Some(a), Some(b)) => Some(if out[a].end - out[a].start >= out[b].end - out[b].start { a } else { b }),
                (a, b) => a.or(b),
            };
            if let Some(j) = pick
                && out[j].speaker != out[i].speaker
            {
                out[i].speaker = out[j].speaker;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // merge touching turns of one speaker
    let mut merged: Vec<Turn> = Vec::new();
    for t in out {
        match merged.last_mut() {
            Some(l) if l.speaker == t.speaker && t.start <= l.end + FRAME => l.end = l.end.max(t.end),
            _ => merged.push(t),
        }
    }
    let mut labels: Vec<usize> = merged.iter().map(|t| t.speaker).collect();
    renumber(&mut labels);
    for (t, l) in merged.iter_mut().zip(labels) {
        t.speaker = l;
    }
    merged
}

/// The speaker of each word of `t` from `turns` (the turn covering most of the word, else the
/// nearest): replaces `t.speakers` and every word's speaker. Returns the number of speakers.
pub fn label_words(t: &mut Transcript, turns: &[Turn]) -> usize {
    if t.words.is_empty() || turns.is_empty() {
        return 0;
    }
    let to_sample = |tick: i64| (tick.max(0) / TICKS_PER_SAMPLE) as usize;
    let mut used: Vec<usize> = Vec::new();
    for w in &mut t.words {
        let (a, b) = (to_sample(w.start.0), to_sample(w.end.0).max(to_sample(w.start.0) + 1));
        let best = turns
            .iter()
            .map(|tr| (tr.end.min(b).saturating_sub(tr.start.max(a)), tr))
            .filter(|(o, _)| *o > 0)
            .max_by_key(|(o, _)| *o)
            .map(|(_, tr)| tr.speaker)
            .or_else(|| turns.iter().min_by_key(|tr| if a >= tr.end { a - tr.end } else { tr.start.saturating_sub(b) }).map(|tr| tr.speaker))
            .unwrap_or(0);
        w.speaker = Some(best as u32);
        if !used.contains(&best) {
            used.push(best);
        }
    }
    // renumber by first appearance in the words
    let mut map: Vec<usize> = Vec::new();
    for w in &t.words {
        let s = w.speaker.unwrap_or(0) as usize;
        if !map.contains(&s) {
            map.push(s);
        }
    }
    for w in &mut t.words {
        let s = w.speaker.unwrap_or(0) as usize;
        w.speaker = map.iter().position(|&m| m == s).map(|k| k as u32);
    }
    t.speakers = (0..map.len()).map(|i| Speaker { name: format!("Speaker {}", i + 1) }).collect();
    map.len()
}

/// The diarization error rate of `hyp` against `truth` (both turns), frame by frame (10 ms),
/// with the best one-to-one mapping of speakers: (missed + false alarm + confusion) / truth
/// speech. `collar`: samples around every truth boundary not scored.
pub fn error_rate(truth: &[Turn], hyp: &[Turn], len: usize, collar: usize) -> f64 {
    let frames = len / FRAME + 1;
    let lab = |turns: &[Turn]| -> Vec<Option<usize>> {
        let mut v = vec![None; frames];
        for t in turns {
            for f in t.start / FRAME..(t.end / FRAME).min(frames) {
                v[f] = Some(t.speaker);
            }
        }
        v
    };
    let (tv, hv) = (lab(truth), lab(hyp));
    let mut skip = vec![false; frames];
    for t in truth {
        for edge in [t.start, t.end] {
            let a = edge.saturating_sub(collar) / FRAME;
            let b = ((edge + collar) / FRAME).min(frames);
            for s in skip.iter_mut().take(b).skip(a) {
                *s = true;
            }
        }
    }
    let nt = truth.iter().map(|t| t.speaker + 1).max().unwrap_or(0);
    let nh = hyp.iter().map(|t| t.speaker + 1).max().unwrap_or(0);
    // co-occurrence counts, then the best mapping (greedy on the largest overlaps: exact for the
    // few speakers here when clusters are clean, an upper bound on the error otherwise)
    let mut co = vec![vec![0usize; nh.max(1)]; nt.max(1)];
    let (mut speech, mut missed, mut fa) = (0usize, 0usize, 0usize);
    for f in 0..frames {
        if skip[f] {
            continue;
        }
        match (tv[f], hv[f]) {
            (Some(a), Some(b)) => {
                speech += 1;
                co[a][b] += 1;
            }
            (Some(_), None) => {
                speech += 1;
                missed += 1;
            }
            (None, Some(_)) => fa += 1,
            (None, None) => {}
        }
    }
    let mut pairs: Vec<(usize, usize, usize)> = (0..nt).flat_map(|a| (0..nh).map(move |b| (a, b))).map(|(a, b)| (co[a][b], a, b)).collect();
    pairs.sort_by_key(|p| std::cmp::Reverse(p.0));
    let (mut ta, mut hb) = (vec![false; nt.max(1)], vec![false; nh.max(1)]);
    let mut matched = 0usize;
    for (c, a, b) in pairs {
        if !ta[a] && !hb[b] {
            ta[a] = true;
            hb[b] = true;
            matched += c;
        }
    }
    let both: usize = co.iter().flatten().sum();
    let confusion = both - matched;
    if speech == 0 {
        return if fa == 0 { 0.0 } else { 1.0 };
    }
    (missed + fa + confusion) as f64 / speech as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seconds_tick;
    use filmcraft_project::Word;

    /// Embeddings of `k` speakers around orthogonal directions with noise.
    fn fake(speakers: &[usize], noise: f32, seed: u32) -> Vec<Vec<f32>> {
        let mut rng = seed;
        speakers
            .iter()
            .map(|&s| {
                (0..16)
                    .map(|d| {
                        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let r = ((rng >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0 * noise;
                        (if d == s { 1.0 } else { 0.0 }) + r
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn clustering_finds_the_speaker_count() {
        for k in 1..=4usize {
            let truth: Vec<usize> = (0..60).map(|i| (i / 5) % k).collect();
            let e = fake(&truth, 0.25, k as u32);
            let got = cluster(&e, &[], &Params::default());
            assert_eq!(got, truth, "{k} speakers");
        }
        // the cap on speakers merges the closest
        let truth: Vec<usize> = (0..40).map(|i| (i / 5) % 4).collect();
        let got = cluster(&fake(&truth, 0.2, 9), &[], &Params { max_speakers: 2, ..Default::default() });
        assert_eq!(got.iter().max(), Some(&1));
        // hostile embeddings: empty, NaN, ragged
        assert!(cluster(&[], &[], &Params::default()).is_empty());
        let weird = vec![vec![f32::NAN; 4], vec![], vec![1.0, 0.0, 0.0, 0.0], vec![0.0; 4]];
        assert_eq!(cluster(&weird, &[f32::NAN, -1.0], &Params::default()).len(), 4);
    }

    #[test]
    fn a_rare_speaker_folds_into_its_nearest() {
        let mut truth: Vec<usize> = (0..100).map(|i| (i / 10) % 2).collect();
        truth[50] = 2;
        let got = cluster(&fake(&truth, 0.1, 3), &[], &Params::default());
        assert_eq!(got.iter().max(), Some(&1), "a 1 % speaker is folded");
    }

    struct Stub;
    impl Embedder for Stub {
        /// "Speaker" = whether the clip's mean is above zero.
        fn embed_many(&self, clips: &[&[f32]]) -> Result<Vec<Vec<f32>>, SpeechError> {
            Ok(clips.iter().map(|c| if c.iter().sum::<f32>() > 0.0 { vec![1.0, 0.1] } else { vec![0.1, 1.0] }).collect())
        }
    }

    #[test]
    fn turns_follow_the_speakers_and_label_words() {
        let sr = 16_000usize;
        // A 0–3 s, B 3.5–6 s, A 6.5–8 s (sign marks the speaker for the stub)
        let mut audio = vec![0f32; 8 * sr];
        for (r, v) in [(0..3 * sr, 0.1), (3 * sr + sr / 2..6 * sr, -0.1), (6 * sr + sr / 2..8 * sr, 0.1)] {
            for x in &mut audio[r] {
                *x = v;
            }
        }
        let segs = vec![0..3 * sr, 3 * sr + sr / 2..6 * sr, 6 * sr + sr / 2..8 * sr];
        let t = turns(&audio, &segs, &Stub, &Params::default()).unwrap();
        assert_eq!(t.iter().map(|t| t.speaker).collect::<Vec<_>>(), vec![0, 1, 0], "{t:?}");
        assert_eq!(
            error_rate(
                &[
                    Turn { start: 0, end: 3 * sr, speaker: 0 },
                    Turn { start: 3 * sr + sr / 2, end: 6 * sr, speaker: 1 },
                    Turn { start: 6 * sr + sr / 2, end: 8 * sr, speaker: 0 }
                ],
                &t,
                audio.len(),
                0
            ),
            0.0
        );
        let mut tr = Transcript {
            words: vec![
                Word::new("a", seconds_tick(1.0), seconds_tick(1.5)),
                Word::new("b", seconds_tick(4.0), seconds_tick(4.4)),
                Word::new("c", seconds_tick(7.0), seconds_tick(7.3)),
                Word::new("gap", seconds_tick(3.1), seconds_tick(3.2)),
            ],
            ..Default::default()
        };
        tr.normalize();
        assert_eq!(label_words(&mut tr, &t), 2);
        assert_eq!(tr.words.iter().map(|w| w.speaker).collect::<Vec<_>>(), vec![Some(0), Some(0), Some(1), Some(0)]);
        tr.check().unwrap();
        // nothing to embed: one speaker per segment, no error
        assert!(turns(&[], &[], &Stub, &Params::default()).unwrap().is_empty());
        let far = 7 * sr..100 * sr;
        let out_of_range = turns(&audio, std::slice::from_ref(&far), &Stub, &Params::default()).unwrap();
        assert!(out_of_range.iter().all(|t| t.end <= audio.len()));
    }

    #[test]
    fn error_rate_counts_misses_and_confusion() {
        let t = |a, b, s| Turn { start: a, end: b, speaker: s };
        let truth = [t(0, 16_000, 0), t(16_000, 32_000, 1)];
        assert_eq!(error_rate(&truth, &[t(0, 16_000, 1), t(16_000, 32_000, 0)], 32_000, 0), 0.0, "labels are matched");
        let all_one = error_rate(&truth, &[t(0, 32_000, 0)], 32_000, 0);
        assert!((all_one - 0.5).abs() < 0.01, "{all_one}");
        assert!((error_rate(&truth, &[], 32_000, 0) - 1.0).abs() < 0.01);
        assert_eq!(error_rate(&[], &[], 0, 0), 0.0);
    }

    #[test]
    fn windows_cover_segments() {
        let sr = 16_000usize;
        let w = windows(&[0..sr / 10, sr..sr * 2, 3 * sr..6 * sr + 100], &Params::default());
        assert_eq!(w[0], sr..2 * sr, "a short segment is one window; a tiny one none");
        assert!(w.iter().skip(1).all(|r| r.len() == 48_000));
        assert_eq!(w.last().map(|r| r.end), Some(6 * sr + 100));
        // an hour of speech stays within the cap
        let hour = 0..sr * 7200;
        assert!(windows(std::slice::from_ref(&hour), &Params::default()).len() <= MAX_WINDOWS + 2);
    }
}
