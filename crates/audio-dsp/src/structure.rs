//! Music structure: what fitting a track to a film needs beyond its beats (`media.beats`'
//! downbeats, sections, drops, end and ending; `clip.remix`'s lifts, stable stretches and
//! endings). Pure functions of the mono signal, the tracked beats and the beat features of
//! [`crate::remix`].
//!
//! ```text
//! level:     mean square of 10 ms blocks, as prefix sums (the RMS of any window in O(1));
//!            the level of a moment is its 100 ms window, in dB
//! extent:    the music starts with the first 100 ms window within 45 dB of the loudest one
//!            (and above −70 dBFS), and ends with the last 10 ms block over that
//! per beat:  the RMS (dB) over the beat, the same in its low band (two one-pole low-passes at
//!            150 Hz) and the chroma change from the beat before
//! downbeats: beat numbers count the tracker's gaps; for meters of 4 and 3 beats and each phase,
//!            the mean of z(chroma change) + ½·z(low-band rise) + ¼·z(level rise) over the beats
//!            of that phase wins; 3 only when clearly better than 4
//! sections:  at downbeats, the checkerboard novelty of the beat self-similarity (J. Foote,
//!            "Automatic audio segmentation using a measure of audio novelty", ICME 2000; four
//!            bars each side) plus the level change per 6 dB; peaks over a threshold, at least
//!            four bars apart (two in a short track). A section's energy is its loudness against
//!            the loudest section's, halving every 10 dB: 2^((dB − loudest) / 10)
//! drops:     a downbeat where the two bars after are ≥ 6 dB louder than the two before (or the
//!            low band ≥ 10 dB), into the loud part (within 4 dB of the loudest beats); one per
//!            four bars
//! stops:     where the level falls 20 dB or more within 50 ms from within 20 dB of the loudest
//!            beats and stays down for 400 ms
//! ending:    cut (attacks go on to within a beat and a half of the end, where the level falls
//!            25 dB within 50 ms from within 24 dB of the loudest beats), fade (a decline of at least 2.5 s from its top, the first moment
//!            walking back from the end as loud as the 4 s before it, with real attacks going on
//!            into it), otherwise cadence (a final hit, then a ring-out without attacks)
//! confidence: beats (how even their intervals are, and how much of the music they cover),
//!            downbeats (how far the winning phase is ahead of the next), sections (how far the
//!            boundaries stand out); overall ½·beats + 0.3·downbeats + 0.2·sections
//! ```
//!
//! Positions are samples of the analysed signal. Beat positions passed in are the boundaries
//! [`crate::remix::Analysis::beats`] (each `lead` samples before its attack); times reported
//! for beats, downbeats and drops are attacks. Deterministic; non-finite samples are silence and
//! the work is bounded by the input length and the beat count.

/// Length of a level block, seconds.
pub const BLOCK_S: f64 = 0.010;
/// Blocks in the window a level is measured over (100 ms).
const WINDOW_BLOCKS: usize = 10;
/// The music's extent: windows within this much of the loudest one.
const EXTENT_DB: f64 = 45.0;
/// ... and above this level (dBFS).
const SILENCE_DBFS: f64 = -70.0;
/// Level of silence, dB.
pub const FLOOR_DB: f64 = -120.0;
/// A drop: the two bars after it this much louder than the two before (or the low band so much).
const DROP_DB: f64 = 6.0;
const DROP_LOW_DB: f64 = 10.0;
/// ... into the loud part: within this much of the loudest beats.
const DROP_NEAR_LOUD_DB: f64 = 4.0;
/// A stop: the level falls this much ...
const STOP_FALL_DB: f64 = 20.0;
/// ... from within this much of the loudest beats.
const STOP_NEAR_LOUD_DB: f64 = 20.0;
/// A cut ending: the level falls this much where the music ends ...
const CUT_FALL_DB: f64 = 25.0;
/// ... from within this much of the loudest beats.
const CUT_NEAR_LOUD_DB: f64 = 24.0;
/// A fade ending: a decline at least this long, seconds.
const FADE_MIN_S: f64 = 2.5;
/// The low band's corner, Hz.
const LOW_HZ: f64 = 150.0;
/// Section novelty: the threshold of a boundary (checkerboard novelty plus level change).
const SECTION_NOVELTY: f64 = 0.25;
/// Most sections reported (a hostile or very long input could peak anywhere).
const MAX_SECTIONS: usize = 512;
/// Most stops reported.
const MAX_STOPS: usize = 512;

fn db(ms: f64) -> f64 {
    if ms.is_finite() && ms > 1e-12 { 10.0 * ms.log10() } else { FLOOR_DB }
}

/// How the music ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndingKind {
    /// A final hit that rings out, with no beats after it.
    Cadence,
    /// The level declines over seconds while the beat goes on.
    Fade,
    /// The music stops while still at level.
    Cut,
}

impl EndingKind {
    pub fn id(self) -> &'static str {
        match self {
            EndingKind::Cadence => "cadence",
            EndingKind::Fade => "fade",
            EndingKind::Cut => "cut",
        }
    }
}

/// The ending of the music.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ending {
    pub kind: EndingKind,
    /// The final hit (cadence), where the fade starts (fade) or where it stops (cut), samples.
    pub at: i64,
    /// Where the sound ends ([`Structure::music_end`]).
    pub end: i64,
}

/// A section: a stretch of like music between two boundaries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Section {
    /// Start and end, samples (the first starts with the music, the last ends with it).
    pub start: i64,
    pub end: i64,
    /// Its first beat (index into the beats), the downbeat it starts on.
    pub first_beat: usize,
    /// Its level (RMS over its beats), dB.
    pub db: f64,
    /// Loudness against the loudest section (1), halving every 10 dB.
    pub energy: f64,
}

/// A drop: the music lifts into its loud part on a downbeat.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drop {
    /// The downbeat (index into the beats).
    pub beat: usize,
    /// How much louder the two bars after are than the two before, dB.
    pub rise_db: f64,
}

/// How sure the analysis is, 0 … 1.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Confidence {
    pub beats: f64,
    pub downbeats: f64,
    pub sections: f64,
    pub overall: f64,
}

/// The structure of a track (see the module docs).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Structure {
    /// Level of each beat (RMS over the beat), dB.
    pub beat_db: Vec<f32>,
    /// Beat numbers: consecutive beats differ by one, a gap the tracker left by more.
    pub beat_num: Vec<i64>,
    /// Beats in a bar (4 or 3).
    pub meter: usize,
    /// The phase of the downbeats: beat `i` is a downbeat when `beat_num[i] % meter == phase`.
    pub phase: usize,
    /// Downbeats (indices into the beats).
    pub downbeats: Vec<usize>,
    pub sections: Vec<Section>,
    pub drops: Vec<Drop>,
    /// Where the sound stops abruptly (samples), the music going quiet or silent.
    pub stops: Vec<i64>,
    /// Where the music's sound starts and ends, samples.
    pub music_start: i64,
    pub music_end: i64,
    pub ending: Option<Ending>,
    /// The level of the loud part: the 90th percentile of the beats' levels, dB.
    pub loud_db: f64,
    pub confidence: Confidence,
}

impl Structure {
    /// Whether beat `i` starts a bar.
    pub fn is_downbeat(&self, i: usize) -> bool {
        let m = self.meter.max(1) as i64;
        self.beat_num.get(i).is_some_and(|n| n.rem_euclid(m) == self.phase as i64)
    }

    /// Whether a jump from beat `i` to beat `j` keeps the bars whole (the same place in the bar).
    pub fn same_phase(&self, i: usize, j: usize) -> bool {
        let m = self.meter.max(1) as i64;
        match (self.beat_num.get(i), self.beat_num.get(j)) {
            (Some(a), Some(b)) => (a - b).rem_euclid(m) == 0,
            _ => false,
        }
    }

    /// The section boundaries after the first (indices of their first beats).
    pub fn boundaries(&self) -> impl Iterator<Item = usize> + '_ {
        self.sections.iter().skip(1).map(|s| s.first_beat)
    }

    /// The section a beat lies in.
    pub fn section_of(&self, beat: usize) -> Option<&Section> {
        self.sections.iter().rev().find(|s| s.first_beat <= beat)
    }
}

/// Prefix sums of 10 ms block mean squares: the level of any window.
struct Levels {
    block: usize,
    cum: Vec<f64>,
}

impl Levels {
    fn new(mono: &[f32], block: usize) -> Levels {
        let block = block.max(1);
        let mut cum = Vec::with_capacity(mono.len() / block + 2);
        cum.push(0.0);
        let mut acc = 0f64;
        for c in mono.chunks(block) {
            let ms = c.iter().map(|v| if v.is_finite() { f64::from(*v) * f64::from(*v) } else { 0.0 }).sum::<f64>() / block as f64;
            acc += ms;
            cum.push(acc);
        }
        Levels { block, cum }
    }

    fn blocks(&self) -> usize {
        self.cum.len().saturating_sub(1)
    }

    /// Mean square of blocks `[a, b)` (clamped to the signal).
    fn ms(&self, a: usize, b: usize) -> f64 {
        let n = self.blocks();
        let (a, b) = (a.min(n), b.min(n));
        if b <= a {
            return 0.0;
        }
        (self.cum[b] - self.cum[a]).max(0.0) / (b - a) as f64
    }

    fn block_of(&self, s: i64) -> usize {
        usize::try_from(s.max(0)).unwrap_or(usize::MAX) / self.block
    }

    /// Level of samples `[s0, s1)`, dB (block resolution).
    fn db_of(&self, s0: i64, s1: i64) -> f64 {
        let (a, b) = (self.block_of(s0), self.block_of(s1));
        db(self.ms(a, b.max(a + 1)))
    }

    /// Level of the 100 ms window starting at block `k`, dB.
    fn window_db(&self, k: usize) -> f64 {
        db(self.ms(k, k + WINDOW_BLOCKS))
    }
}

fn percentile(v: &[f64], q: f64) -> Option<f64> {
    let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return None;
    }
    s.sort_by(f64::total_cmp);
    let k = ((s.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    s.get(k).copied()
}

fn zscore(v: &[f64]) -> Vec<f64> {
    let n = v.len().max(1) as f64;
    let mean = v.iter().sum::<f64>() / n;
    let sd = (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n).sqrt();
    if sd > 1e-9 { v.iter().map(|x| (x - mean) / sd).collect() } else { vec![0.0; v.len()] }
}

fn cos12(a: &[f32; 12], b: &[f32; 12]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let na = a.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
    let nb = b.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
    if na > 0.0 && nb > 0.0 { d / (na * nb) } else { 1.0 }
}

/// Energy-mean of beat levels `[a, b)` (dB), `None` when empty.
fn mean_db(v: &[f32], a: usize, b: usize) -> Option<f64> {
    let s = v.get(a.min(v.len())..b.min(v.len()))?;
    if s.is_empty() {
        return None;
    }
    Some(db(s.iter().map(|d| 10f64.powf(f64::from(*d) / 10.0)).sum::<f64>() / s.len() as f64))
}

/// Analyse the structure of `mono` at `sr` given its `beats` (boundaries, `lead` samples before
/// each attack), the mean beat `period` (samples), the per-beat `chroma` and the beat
/// self-similarity `sim` (`beats.len()²`, row-major). Mismatched inputs give an empty structure,
/// never a panic.
pub fn analyse(mono: &[f32], sr: u32, beats: &[i64], lead: i64, period: f64, chroma: &[[f32; 12]], sim: &[f32]) -> Structure {
    let sr_f = f64::from(sr.max(1));
    let block = ((BLOCK_S * sr_f).round() as usize).max(1);
    let lv = Levels::new(mono, block);
    let len = mono.len() as i64;
    let mut st = Structure { meter: 4, loud_db: FLOOR_DB, ..Structure::default() };
    // ---- the music's extent
    let nb = lv.blocks();
    let windows = nb.saturating_sub(WINDOW_BLOCKS - 1).max(usize::from(nb > 0));
    let peak = (0..windows).map(|k| lv.window_db(k)).fold(FLOOR_DB, f64::max);
    let thr = (peak - EXTENT_DB).max(SILENCE_DBFS);
    let (Some(first), Some(_)) = ((0..windows).find(|&k| lv.window_db(k) >= thr), (0..windows).rev().find(|&k| lv.window_db(k) >= thr)) else {
        return st;
    };
    // the end to the block (10 ms): the last block over the threshold
    let last = (0..nb).rev().find(|&k| db(lv.ms(k, k + 1)) >= thr).unwrap_or(first);
    st.music_start = (first * block) as i64;
    st.music_end = (((last + 1) * block) as i64).min(len);
    let m = beats.len();
    if m < 2 || !(period.is_finite() && period > 0.0) {
        return st;
    }
    // ---- per beat: level, low band, beat numbers
    let end_of = |i: usize| -> i64 { beats.get(i + 1).copied().unwrap_or_else(|| beats[i].saturating_add(period as i64)) };
    st.beat_db = (0..m).map(|i| lv.db_of(beats[i], end_of(i)) as f32).collect();
    let low = low_band(mono, sr_f);
    let lowlv = Levels::new(&low, block);
    let low_db: Vec<f64> = (0..m).map(|i| lowlv.db_of(beats[i], end_of(i))).collect();
    let mut num = Vec::with_capacity(m);
    num.push(0i64);
    for i in 1..m {
        let step = ((beats[i] - beats[i - 1]) as f64 / period).round().clamp(1.0, 8.0) as i64;
        num.push(num[i - 1] + step);
    }
    st.beat_num = num;
    let levels: Vec<f64> = st.beat_db.iter().map(|d| f64::from(*d)).filter(|d| *d > FLOOR_DB + 1.0).collect();
    st.loud_db = percentile(&levels, 0.9).unwrap_or(FLOOR_DB);
    // ---- downbeats
    let downbeat_confidence = downbeats(&mut st, chroma, &low_db);
    // ---- sections
    let (bounds, section_confidence) = boundaries(&st, sim, m);
    let starts: Vec<usize> = std::iter::once(0).chain(bounds.iter().copied()).collect();
    for (k, &b) in starts.iter().enumerate() {
        let e = starts.get(k + 1).copied().unwrap_or(m);
        let start = if k == 0 { st.music_start.min(beats[b]) } else { beats[b] + lead };
        let end = match starts.get(k + 1) {
            Some(&n) => beats[n] + lead,
            None => st.music_end,
        };
        let d = mean_db(&st.beat_db, b, e).unwrap_or(FLOOR_DB);
        st.sections.push(Section { start, end: end.max(start), first_beat: b, db: d, energy: 0.0 });
    }
    let loudest = st.sections.iter().map(|s| s.db).fold(FLOOR_DB, f64::max);
    for s in &mut st.sections {
        s.energy = if s.db > FLOOR_DB + 1.0 { ((s.db - loudest) / 10.0).exp2().clamp(0.0, 1.0) } else { 0.0 };
        s.energy = (s.energy * 1000.0).round() / 1000.0;
    }
    // ---- drops
    let bar = st.meter;
    let mut drops: Vec<Drop> = Vec::new();
    for &k in &st.downbeats {
        if k < bar || beats[k] + lead >= st.music_end {
            continue;
        }
        let (Some(pre), Some(post)) = (mean_db(&st.beat_db, k.saturating_sub(2 * bar), k), mean_db(&st.beat_db, k, k + 2 * bar)) else { continue };
        let lpre = energy_mean(&low_db, k.saturating_sub(2 * bar), k);
        let lpost = energy_mean(&low_db, k, k + 2 * bar);
        let rise = post - pre;
        let low_rise = lpost - lpre;
        if (rise >= DROP_DB || low_rise >= DROP_LOW_DB) && post >= st.loud_db - DROP_NEAR_LOUD_DB {
            let strength = rise.max(0.6 * low_rise);
            // one per four bars: the strongest
            match drops.last_mut() {
                Some(d) if st.beat_num[k] - st.beat_num[d.beat] < 4 * bar as i64 => {
                    if strength > d.rise_db {
                        *d = Drop { beat: k, rise_db: strength };
                    }
                }
                _ => drops.push(Drop { beat: k, rise_db: strength }),
            }
        }
    }
    for d in &mut drops {
        d.rise_db = (d.rise_db * 10.0).round() / 10.0;
    }
    st.drops = drops;
    // ---- stops
    let near = st.loud_db - STOP_NEAR_LOUD_DB;
    let mut k = 30usize;
    while k + 45 < nb && st.stops.len() < MAX_STOPS {
        let before = db(lv.ms(k - 30, k));
        let after = db(lv.ms(k + 5, k + 45));
        if before >= near && after <= before - STOP_FALL_DB {
            // the block where it falls: the first under the fall
            let fall = 10f64.powf((before - STOP_FALL_DB) / 10.0);
            let at = (k.saturating_sub(5)..k + 5).find(|&b| lv.ms(b, b + 1) < fall).unwrap_or(k);
            st.stops.push((at * block) as i64);
            k = at + 100;
        } else {
            k += 1;
        }
    }
    // ---- the ending
    st.ending = ending(&st, &lv, beats, lead, period, sr_f, block);
    // ---- confidence
    let beats_conf = beat_confidence(beats, &st.beat_num, period, st.music_start, st.music_end);
    let c = Confidence { beats: beats_conf, downbeats: downbeat_confidence, sections: section_confidence, overall: 0.0 };
    let overall = 0.5 * c.beats + 0.3 * c.downbeats + 0.2 * c.sections;
    let r = |x: f64| (x.clamp(0.0, 1.0) * 100.0).round() / 100.0;
    st.confidence = Confidence { beats: r(c.beats), downbeats: r(c.downbeats), sections: r(c.sections), overall: r(overall) };
    st
}

fn energy_mean(v: &[f64], a: usize, b: usize) -> f64 {
    let s = v.get(a.min(v.len())..b.min(v.len())).unwrap_or(&[]);
    if s.is_empty() {
        return FLOOR_DB;
    }
    db(s.iter().map(|d| 10f64.powf(d / 10.0)).sum::<f64>() / s.len() as f64)
}

/// The low band of `x` (two one-pole low-passes at [`LOW_HZ`]).
fn low_band(x: &[f32], sr: f64) -> Vec<f32> {
    let a = (1.0 - (-2.0 * std::f64::consts::PI * LOW_HZ / sr.max(1.0)).exp()) as f32;
    let (mut y1, mut y2) = (0f32, 0f32);
    x.iter()
        .map(|v| {
            let v = if v.is_finite() { v.clamp(-16.0, 16.0) } else { 0.0 };
            y1 += a * (v - y1);
            y2 += a * (y1 - y2);
            y2
        })
        .collect()
}

/// Sets the meter, phase and downbeats; returns the downbeat confidence.
fn downbeats(st: &mut Structure, chroma: &[[f32; 12]], low_db: &[f64]) -> f64 {
    let m = st.beat_num.len();
    if m < 4 {
        st.downbeats = (0..m).collect();
        return 0.0;
    }
    let change: Vec<f64> = (0..m).map(|i| if i == 0 { 0.0 } else { chroma.get(i).zip(chroma.get(i - 1)).map_or(0.0, |(a, b)| 1.0 - cos12(a, b)) }).collect();
    let low_rise: Vec<f64> = (0..m).map(|i| if i == 0 { 0.0 } else { (low_db[i] - low_db[i - 1]).max(0.0) }).collect();
    let rise: Vec<f64> = (0..m).map(|i| if i == 0 { 0.0 } else { f64::from(st.beat_db[i] - st.beat_db[i - 1]).max(0.0) }).collect();
    let (zc, zl, zr) = (zscore(&change), zscore(&low_rise), zscore(&rise));
    let f: Vec<f64> = (0..m).map(|i| zc[i] + 0.5 * zl[i] + 0.25 * zr[i]).collect();
    // (score of the best phase, of the next best, the phase) for a meter
    let score = |meter: usize| -> (f64, f64, usize) {
        let mut s: Vec<(f64, usize)> = (0..meter)
            .map(|p| {
                let v: Vec<f64> = (1..m).filter(|&i| st.beat_num[i].rem_euclid(meter as i64) == p as i64).map(|i| f[i]).collect();
                (if v.is_empty() { f64::NEG_INFINITY } else { v.iter().sum::<f64>() / v.len() as f64 }, p)
            })
            .collect();
        s.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        (s[0].0, s.get(1).map_or(f64::NEG_INFINITY, |x| x.0), s[0].1)
    };
    let (b4, n4, p4) = score(4);
    let (b3, n3, p3) = score(3);
    let (best, next, meter, phase) = if b3 - n3 > 1.25 * (b4 - n4) + 0.1 && b3 > b4 { (b3, n3, 3, p3) } else { (b4, n4, 4, p4) };
    st.meter = meter;
    st.phase = phase;
    st.downbeats = (0..m).filter(|&i| st.beat_num[i].rem_euclid(meter as i64) == phase as i64).collect();
    if best.is_finite() && next.is_finite() { ((best - next) / 0.6).clamp(0.0, 1.0) } else { 0.0 }
}

/// Section boundaries (downbeat indices, increasing) and the section confidence.
fn boundaries(st: &Structure, sim: &[f32], m: usize) -> (Vec<usize>, f64) {
    let bar = st.meter.max(1);
    let total_bars = st.beat_num.last().copied().unwrap_or(0) / bar as i64;
    let half = if total_bars >= 16 { 4 * bar } else { 2 * bar };
    let min_gap = if total_bars >= 24 { 4 * bar as i64 } else { 2 * bar as i64 };
    let have_sim = sim.len() == m.saturating_mul(m) && m > 0;
    let s = |i: usize, j: usize| -> f64 { if have_sim { f64::from(sim[i * m + j]) } else { 0.0 } };
    let mut scored: Vec<(f64, usize)> = Vec::new();
    for &k in &st.downbeats {
        if k < bar || k + bar > m {
            continue;
        }
        let w = half.min(k).min(m - k);
        if w < bar {
            continue;
        }
        let mut nov = 0f64;
        for a in 1..=w {
            for b in 1..=w {
                nov += s(k - a, k - b) + s(k + a - 1, k + b - 1) - s(k - a, k + b - 1) - s(k + a - 1, k - b);
            }
        }
        nov /= (w * w) as f64 * 2.0;
        // the level change over a bar each side (sharp: a wider window blurs where it happens)
        let before = mean_db(&st.beat_db, k - bar, k).unwrap_or(FLOOR_DB);
        let after = mean_db(&st.beat_db, k, k + bar).unwrap_or(FLOOR_DB);
        let level = ((after - before).abs() / 6.0).min(3.0);
        scored.push((nov.max(0.0) + level, k));
    }
    let mean = scored.iter().map(|x| x.0).sum::<f64>() / scored.len().max(1) as f64;
    let sd = (scored.iter().map(|x| (x.0 - mean).powi(2)).sum::<f64>() / scored.len().max(1) as f64).sqrt();
    let thr = SECTION_NOVELTY.max(mean + 0.5 * sd);
    // local maxima over ±1 bar (of equal neighbours, the earlier)
    let peak = |k: usize, v: f64| scored.iter().filter(|(_, j)| j.abs_diff(k) <= bar && *j != k).all(|(u, j)| *u < v || (*u == v && k < *j));
    let mut order: Vec<(f64, usize)> = scored.iter().copied().filter(|(v, k)| *v >= thr && peak(*k, *v)).collect();
    order.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut out: Vec<usize> = Vec::new();
    let mut strengths = Vec::new();
    for (v, k) in order {
        if out.len() >= MAX_SECTIONS {
            break;
        }
        if out.iter().any(|&j| (st.beat_num[j] - st.beat_num[k]).abs() < min_gap) {
            continue;
        }
        out.push(k);
        strengths.push(v);
    }
    out.sort_unstable();
    let conf = if strengths.is_empty() { 0.5 } else { (strengths.iter().sum::<f64>() / strengths.len() as f64 / (2.0 * thr)).clamp(0.0, 1.0) };
    (out, conf)
}

fn ending(st: &Structure, lv: &Levels, beats: &[i64], lead: i64, period: f64, sr: f64, block: usize) -> Option<Ending> {
    let end = st.music_end;
    if end <= st.music_start || beats.is_empty() {
        return None;
    }
    let e = lv.block_of(end);
    let attack = |i: usize| beats[i].saturating_add(lead);
    // a real attack: the 30 ms after it at least 3 dB over the 30 ms before (a level rise);
    // ghost beats in a ring-out have none
    let rises = |i: usize| -> bool {
        let a = lv.block_of(attack(i));
        db(lv.ms(a, a + 3)) >= db(lv.ms(a.saturating_sub(4), a.saturating_sub(1))) + 3.0
    };
    let last_attack = (0..beats.len()).rev().filter(|&i| attack(i) < end).find(|&i| rises(i)).map_or(st.music_start, attack);
    // it stops: the attacks go on to within a beat and a half of the end, and from the 150 ms
    // before its last block to the 100 ms after the level falls by CUT_FALL_DB or more (a
    // ring-out or a fade dies away over seconds; a file that ends mid-music falls to nothing)
    let before = db(lv.ms(e.saturating_sub(16), e.saturating_sub(1)));
    let after = db(lv.ms(e + 1, e + 11));
    let still_playing = ((end - last_attack) as f64) < (1.5 * period).max(0.4 * sr);
    if still_playing && before - after >= CUT_FALL_DB && before >= st.loud_db - CUT_NEAR_LOUD_DB {
        return Some(Ending { kind: EndingKind::Cut, at: end, end });
    }
    // where the decline starts: walking back from the end (in 100 ms steps, the level of 1 s
    // around each), the first moment as loud as the 4 s before it (within 2 dB): the top of the
    // decline
    let sec = (1.0 / BLOCK_S).round() as usize;
    let step = sec / 10;
    let smooth = |k: usize| db(lv.ms(k.saturating_sub(sec / 2), k + sec / 2));
    let mut k = e;
    let mut decline = None;
    while k > step {
        let here = smooth(k);
        let before = (1..=40usize).filter_map(|q| k.checked_sub(q * step)).map(smooth).fold(FLOOR_DB, f64::max);
        if here >= before - 2.0 && here > FLOOR_DB + 1.0 {
            decline = Some(k);
            break;
        }
        k -= step;
    }
    let decline = decline.unwrap_or(lv.block_of(st.music_start));
    let at = (decline * block) as i64;
    let in_decline = (0..beats.len()).filter(|&i| attack(i) as f64 >= at as f64 + 0.5 * sr && attack(i) < end && rises(i)).count();
    if (end - at) as f64 / sr >= FADE_MIN_S && in_decline >= 3 {
        return Some(Ending { kind: EndingKind::Fade, at, end });
    }
    // a cadence: the final hit is the last beat at least 0.3 s before the end that is no
    // quieter than the beat before it (less 1 dB) and within 4 dB of the eight before it
    let hit = (1..beats.len())
        .rev()
        .filter(|&i| (end - attack(i)) as f64 / sr >= 0.3)
        .find(|&i| {
            let here = f64::from(st.beat_db.get(i).copied().unwrap_or(FLOOR_DB as f32));
            let prev = f64::from(st.beat_db.get(i - 1).copied().unwrap_or(FLOOR_DB as f32));
            let before = mean_db(&st.beat_db, i.saturating_sub(8), i).unwrap_or(FLOOR_DB);
            here >= prev - 1.0 && here >= before - 4.0
        })
        .map_or(at, attack);
    Some(Ending { kind: EndingKind::Cadence, at: hit.min(end), end })
}

fn beat_confidence(beats: &[i64], num: &[i64], period: f64, start: i64, end: i64) -> f64 {
    let mut dev: Vec<f64> = Vec::new();
    for i in 1..beats.len().min(num.len()) {
        if num[i] - num[i - 1] == 1 {
            dev.push(((beats[i] - beats[i - 1]) as f64 / period - 1.0).abs());
        }
    }
    let Some(mad) = percentile(&dev, 0.5) else { return 0.0 };
    let regular = (1.0 - 5.0 * mad).clamp(0.0, 1.0);
    let span = (end - start).max(1) as f64;
    let covered = match (beats.first(), beats.last()) {
        (Some(a), Some(z)) => ((z - a) as f64 + period) / span,
        _ => 0.0,
    };
    regular * (0.5 + 0.5 * covered.clamp(0.0, 1.0))
}

// ------------------------------------------------------------------------------------- fixtures

/// One part of [`fixture`] music.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Part {
    pub bars: usize,
    /// Its level, dB (0: full).
    pub db: f64,
    /// Drums and bass play (otherwise only the chords).
    pub drums: bool,
    /// A build: a snare roll and a noise riser rising over the part, no kick or bass.
    pub build: bool,
}

/// How [`fixture`] music ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixtureEnd {
    /// A final hit on the downbeat after the last bar, ringing out for 2.5 s.
    Cadence,
    /// The last four bars fade out (linearly in dB to −60 dB).
    Fade,
    /// The music stops at the end of the last bar.
    Cut,
}

/// Generated music with known structure (for tests here and in the engine).
#[derive(Clone, Debug, PartialEq)]
pub struct Fixture {
    /// Stereo planar audio, 1 s of silence after the music.
    pub audio: Vec<Vec<f32>>,
    /// Attack of every beat, samples (a cadence's final hit included).
    pub beats: Vec<i64>,
    /// Attack of every downbeat (a cadence's final hit included).
    pub downbeats: Vec<i64>,
    /// Where each part starts (its first downbeat).
    pub parts: Vec<i64>,
    /// The final hit (cadence), fade start (fade) or the stop (cut).
    pub ending_at: i64,
    /// Where the music's sound ends, about.
    pub music_end: i64,
}

/// Music of `parts` at `bpm` in 4/4 ending as `end`: chords that change every bar (a different
/// progression in odd parts), kick on 1 and 3, snare on 2 and 4, hats on the eighths, a bass note
/// per beat and a crash on each part's first downbeat when the drums play.
pub fn fixture(sr: u32, bpm: f64, parts: &[Part], end: FixtureEnd, seed: u32) -> Fixture {
    let sr_f = f64::from(sr.max(1));
    let beat = sr_f * 60.0 / bpm.clamp(40.0, 240.0);
    let bars: usize = parts.iter().map(|p| p.bars).sum::<usize>().min(1000);
    let music = (bars as f64 * 4.0 * beat).round() as usize;
    let tail = match end {
        FixtureEnd::Cadence => (2.5 * sr_f) as usize + beat as usize,
        _ => 0,
    };
    let n = music + tail + sr as usize;
    let mut l = vec![0f32; n];
    let mut rng = seed.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let mut noise = move || {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (rng >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    };
    let tau = std::f64::consts::TAU;
    let prog_a = [3, 0, 8, 10];
    let prog_b = [5, 10, 7, 0];
    let minor = |r: i32| matches!(r, 0 | 5 | 7);
    let mut out = Fixture { audio: Vec::new(), beats: Vec::new(), downbeats: Vec::new(), parts: Vec::new(), ending_at: 0, music_end: 0 };
    let mut bar = 0usize;
    let chord = |l: &mut [f32], root: i32, from: usize, to: usize, gain: f64, decay: f64| {
        let third = if minor(root) { 3 } else { 4 };
        for i in from..to.min(l.len()) {
            let t = (i - from) as f64 / sr_f;
            let env = (1.0 - (-t * 30.0).exp()) * (-t * decay).exp();
            let mut v = 0.0;
            for nt in [root, root + third, root + 7] {
                let f = 220.0 * 2f64.powf(f64::from(nt) / 12.0);
                let x = tau * f * i as f64 / sr_f;
                v += x.sin() + 0.3 * (2.0 * x).sin();
            }
            l[i] += (gain * 0.06 * env * v) as f32;
        }
    };
    for (pi, p) in parts.iter().enumerate() {
        let g = 10f64.powf(p.db / 20.0);
        let prog = if pi % 2 == 0 { prog_a } else { prog_b };
        out.parts.push((bar as f64 * 4.0 * beat).round() as i64);
        for pb in 0..p.bars {
            if bar >= bars {
                break;
            }
            let root = prog[pb % 4];
            let b0 = (bar as f64 * 4.0 * beat).round() as usize;
            let b1 = ((bar + 1) as f64 * 4.0 * beat).round() as usize;
            chord(&mut l, root, b0, b1, g, 0.1);
            out.downbeats.push(b0 as i64);
            if p.drums && pb == 0 {
                for k in 0..(sr as usize).min(n.saturating_sub(b0)) {
                    let t = k as f32 / sr as f32;
                    l[b0 + k] += (g as f32) * 0.25 * (-t * 4.0).exp() * noise();
                }
            }
            for q in 0..4 {
                let start = ((bar * 4 + q) as f64 * beat).round() as usize;
                out.beats.push(start as i64);
                let dur = (beat * 0.9) as usize;
                let bass_f = 110.0 * 2f64.powf(f64::from(root) / 12.0);
                if p.build {
                    // a snare roll in sixteenths, louder through the part, and a noise riser
                    let x = (pb * 4 + q) as f64 / (p.bars * 4).max(1) as f64;
                    for s16 in 0..4 {
                        let hs = start + (s16 as f64 * beat / 4.0) as usize;
                        for k in 0..(sr as usize / 20).min(n.saturating_sub(hs)) {
                            let t = k as f32 / sr as f32;
                            l[hs + k] += (g * (0.1 + 0.3 * x)) as f32 * (-t * 40.0).exp() * noise();
                        }
                    }
                    for k in 0..(beat as usize).min(n.saturating_sub(start)) {
                        l[start + k] += (g * 0.05 * (0.2 + x)) as f32 * noise();
                    }
                    continue;
                }
                if !p.drums {
                    continue;
                }
                for k in 0..dur.min(n.saturating_sub(start)) {
                    let t = k as f64 / sr_f;
                    let i = start + k;
                    if q % 2 == 0 {
                        let f = 50.0 + 90.0 * (-t * 25.0).exp();
                        l[i] += (g * 0.8 * (-t * 9.0).exp() * (tau * f * t).sin()) as f32;
                    } else if t < 0.2 {
                        l[i] += (g as f32) * 0.35 * (-t as f32 * 22.0).exp() * noise();
                    }
                    l[i] += (g * 0.15 * (-t * 3.0).exp() * (tau * bass_f * t).sin()) as f32;
                }
                for h in 0..2 {
                    let hs = start + (f64::from(h) * beat / 2.0) as usize;
                    for k in 0..(sr as usize / 25).min(n.saturating_sub(hs)) {
                        let t = k as f32 / sr as f32;
                        l[hs + k] += (g as f32) * 0.08 * (-t * 120.0).exp() * noise();
                    }
                }
            }
            bar += 1;
        }
    }
    match end {
        FixtureEnd::Cadence => {
            // the final hit: the tonic chord, a kick and a crash on the next downbeat, ringing out
            let h = music;
            let decay = 60.0 / 20.0 * std::f64::consts::LN_10 / 2.5;
            chord(&mut l, 3, h, h + tail, 2.0, decay);
            for k in 0..tail.min(n.saturating_sub(h)) {
                let t = k as f64 / sr_f;
                let f = 50.0 + 90.0 * (-t * 25.0).exp();
                l[h + k] += (0.8 * (-t * 9.0).exp() * (tau * f * t).sin()) as f32;
                l[h + k] += (0.25 * (-t * decay).exp()) as f32 * noise();
            }
            out.ending_at = h as i64;
            out.beats.push(h as i64);
            out.downbeats.push(h as i64);
            out.music_end = (h as f64 + 2.5 * sr_f * 45.0 / 60.0) as i64;
        }
        FixtureEnd::Fade => {
            let from = ((bars.saturating_sub(4)) as f64 * 4.0 * beat).round() as usize;
            for (i, v) in l.iter_mut().enumerate().take(music).skip(from) {
                let x = (i - from) as f64 / (music - from).max(1) as f64;
                *v *= 10f64.powf(-60.0 * x / 20.0) as f32;
            }
            for v in l.iter_mut().skip(music) {
                *v = 0.0;
            }
            out.ending_at = from as i64;
            out.music_end = (from as f64 + (music - from) as f64 * 0.75) as i64;
        }
        FixtureEnd::Cut => {
            for v in l.iter_mut().skip(music) {
                *v = 0.0;
            }
            out.ending_at = music as i64;
            out.music_end = music as i64;
        }
    }
    let peak = l.iter().fold(0f32, |m, v| m.max(v.abs()));
    if peak > 0.9 {
        let k = 0.9 / peak;
        l.iter_mut().for_each(|v| *v *= k);
    }
    out.audio = vec![l.clone(), l];
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remix;

    const SR: u32 = 22_050;

    fn part(bars: usize, db: f64, drums: bool) -> Part {
        Part { bars, db, drums, build: false }
    }

    fn analysed(f: &Fixture) -> remix::Analysis {
        let refs: Vec<&[f32]> = f.audio.iter().map(Vec::as_slice).collect();
        remix::analyze(&refs, SR).expect("analysis")
    }

    fn secs(x: i64) -> f64 {
        x as f64 / f64::from(SR)
    }

    /// Downbeats found where the bars start, the meter is four, and the sections and their
    /// energy follow the parts: a quiet intro, a verse, a build, a drop (the loudest), a
    /// breakdown and a second drop, then a cadence that rings out.
    #[test]
    fn downbeats_sections_drops_and_a_cadence() {
        let parts = [
            part(4, -16.0, false),
            part(8, -7.0, true),
            Part { bars: 4, db: -9.0, drums: false, build: true },
            part(8, 0.0, true),
            part(4, -14.0, false),
            part(8, 0.0, true),
        ];
        let f = fixture(SR, 120.0, &parts, FixtureEnd::Cadence, 3);
        let a = analysed(&f);
        let st = &a.structure;
        let lead = remix::beat_lead(SR);
        assert_eq!(st.meter, 4);
        // every downbeat found is a bar start (within 25 ms), and most bar starts are found
        let tol = (0.025 * f64::from(SR)) as i64;
        let found: Vec<i64> = st.downbeats.iter().map(|&i| a.beats[i] + lead).collect();
        let on = found.iter().filter(|d| f.downbeats.iter().any(|t| (t - *d).abs() <= tol)).count();
        assert!(on * 10 >= found.len() * 9, "{on} of {} downbeats on bar starts", found.len());
        assert!(found.len() * 10 >= f.downbeats.len() * 8, "{} of {} bars", found.len(), f.downbeats.len());
        // the section boundaries are part boundaries (within a bar)
        let bar = (4.0 * 0.5 * f64::from(SR)) as i64;
        let starts: Vec<i64> = st.sections.iter().skip(1).map(|s| s.start).collect();
        eprintln!("sections: {:?}", st.sections.iter().map(|s| (secs(s.start), secs(s.end), s.energy)).collect::<Vec<_>>());
        eprintln!("drops: {:?}", st.drops.iter().map(|d| (secs(a.beats[d.beat] + lead), d.rise_db)).collect::<Vec<_>>());
        for s in &starts {
            assert!(f.parts.iter().any(|p| (p - s).abs() <= bar / 4), "a section at {:.2} s is not a part boundary", secs(*s));
        }
        for p in f.parts.iter().skip(1) {
            assert!(starts.iter().any(|s| (p - s).abs() <= bar / 4), "the part at {:.2} s is not a section", secs(*p));
        }
        // energy: the drops are the loudest sections, the intro and breakdown quiet
        let energy_at = |t: i64| st.sections.iter().find(|s| s.start <= t + bar / 4 && t + bar / 4 < s.end).map_or(-1.0, |s| s.energy);
        assert!((energy_at(f.parts[3]) - 1.0).abs() < 0.15, "{}", energy_at(f.parts[3]));
        assert!(energy_at(f.parts[0]) < 0.4 && energy_at(f.parts[4]) < 0.5, "{} {}", energy_at(f.parts[0]), energy_at(f.parts[4]));
        // the two drops, nothing else
        let drops: Vec<i64> = st.drops.iter().map(|d| a.beats[d.beat] + lead).collect();
        assert_eq!(drops.len(), 2, "{drops:?}");
        for (d, p) in drops.iter().zip([f.parts[3], f.parts[5]]) {
            assert!((d - p).abs() <= tol, "drop at {:.2} s, part at {:.2} s", secs(*d), secs(p));
        }
        // the cadence: its final hit, then the music ends as the ring-out dies away
        let e = st.ending.expect("an ending");
        assert_eq!(e.kind, EndingKind::Cadence, "{e:?}");
        assert!((e.at - f.ending_at).abs() <= tol, "hit at {:.2} s, made at {:.2} s", secs(e.at), secs(f.ending_at));
        assert!((st.music_end - f.music_end).abs() as f64 / f64::from(SR) < 0.6, "ends at {:.2} s, about {:.2} s", secs(st.music_end), secs(f.music_end));
        assert!(st.confidence.overall > 0.6, "{:?}", st.confidence);
    }

    #[test]
    fn a_fade_and_a_cut_are_told_apart_from_a_cadence() {
        let parts = [part(4, -8.0, true), part(8, 0.0, true), part(8, -4.0, true)];
        for (end, kind) in [(FixtureEnd::Fade, EndingKind::Fade), (FixtureEnd::Cut, EndingKind::Cut), (FixtureEnd::Cadence, EndingKind::Cadence)] {
            let f = fixture(SR, 124.0, &parts, end, 5);
            let a = analysed(&f);
            let e = a.structure.ending.expect("an ending");
            eprintln!("{end:?}: {e:?} (made at {:.2} s), music end {:.2} s", secs(f.ending_at), secs(a.structure.music_end));
            assert_eq!(e.kind, kind, "{end:?}");
            let tol = match end {
                FixtureEnd::Fade => 1.0,
                _ => 0.05,
            };
            assert!((secs(e.at) - secs(f.ending_at)).abs() < tol, "{end:?}: at {:.2} s, made at {:.2} s", secs(e.at), secs(f.ending_at));
            if end == FixtureEnd::Cut {
                assert!((a.structure.music_end - f.music_end).abs() as f64 / f64::from(SR) < 0.06);
                assert!(a.structure.stops.iter().any(|s| (s - f.ending_at).abs() as f64 / f64::from(SR) < 0.06), "{:?}", a.structure.stops);
            }
        }
    }

    /// The music engine's own tracks: its bars, sections and final hit are known exactly.
    #[test]
    fn the_music_engines_bars_and_final_hit() {
        use crate::music::{Genre, Mood, Request, generate};
        for (bpm, mood, genre, seconds, seed) in [(120.0, Mood::Energetic, Genre::Electronic, 40.0, 1), (100.0, Mood::Happy, Genre::Pop, 36.0, 4)] {
            let req = Request { bpm: Some(bpm), genre: Some(genre), seed, ..Request::new(mood, filmcraft_time::Tick::from_seconds_f64(seconds)) };
            let t = generate(&req, SR).expect("music");
            let a = remix::analyze(&[&t.mix.left, &t.mix.right], SR).expect("analysis");
            let st = &a.structure;
            let lead = remix::beat_lead(SR);
            let bars: Vec<f64> = t.plan.bars().iter().map(|b| b.seconds()).collect();
            let found: Vec<f64> = st.downbeats.iter().map(|&i| secs(a.beats[i] + lead)).collect();
            let on = found.iter().filter(|d| bars.iter().any(|b| (b - *d).abs() < 0.035)).count();
            eprintln!("{genre:?}: {on} of {} downbeats on bars ({} bars), meter {}, {:?}", found.len(), bars.len(), st.meter, st.confidence);
            assert!(on * 10 >= found.len() * 8, "{genre:?}: {on} of {} downbeats on the bars", found.len());
            let e = st.ending.expect("an ending");
            assert_eq!(e.kind, EndingKind::Cadence, "{genre:?}: {e:?}");
            assert!((secs(e.at) - t.plan.ending.seconds()).abs() < 0.05, "{genre:?}: hit at {:.2} s, made at {:.2} s", secs(e.at), t.plan.ending.seconds());
        }
    }

    #[test]
    fn deterministic_and_hostile_inputs_never_panic() {
        let f = fixture(SR, 120.0, &[part(8, 0.0, true)], FixtureEnd::Cut, 1);
        assert_eq!(analysed(&f).structure, analysed(&f).structure);
        let mono = &f.audio[0];
        let beats: Vec<i64> = f.beats.clone();
        // empty, mismatched, NaN, silent, absurd periods and rates
        let _ = analyse(&[], SR, &[], 0, 1000.0, &[], &[]);
        let _ = analyse(mono, SR, &beats, 0, f64::NAN, &[], &[]);
        let _ = analyse(mono, SR, &beats, 0, 0.0, &[], &[1.0; 3]);
        let _ = analyse(mono, 0, &beats, 0, 11_025.0, &[[0.0; 12]; 2], &[]);
        let _ = analyse(mono, u32::MAX, &beats, i64::MAX / 4, 11_025.0, &[], &[]);
        let nan = vec![f32::NAN; 50_000];
        let s = analyse(&nan, SR, &[0, 10_000, 20_000], 0, 10_000.0, &[], &[]);
        assert_eq!(s.music_end, 0, "NaN is silence");
        let quiet = vec![0f32; 50_000];
        assert!(analyse(&quiet, SR, &[0, 10_000, 20_000], 0, 10_000.0, &[], &[]).ending.is_none());
        let backwards: Vec<i64> = beats.iter().rev().copied().collect();
        let _ = analyse(mono, SR, &backwards, 0, 11_025.0, &[], &[]);
    }
}
