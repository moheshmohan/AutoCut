//! Beat grids: snapping a template's cuts to its beats, and re-quantizing them when the music
//! slot gets a different track (its beats from `media.beats`).
//!
//! The cuts are the edges of the beat-synced placements. Each cut keeps its beat number
//! ([`QuantizeMode::Beats`]: a slower song gives a longer video) or moves to the beat nearest it
//! ([`QuantizeMode::Nearest`]: the length stays), within every placement's min / max duration.
//! The old → new cut times then form a [`TimeMap`] that re-times the whole sequence, so titles,
//! transitions, markers and captions stay attached to the placements they belong to.

use filmcraft_time::{FrameRate, Tick};

use crate::manifest::QuantizeMode;

/// A monotonic piecewise-linear map from template time to applied time. Before the first point
/// and after the last, times shift with the nearest point.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TimeMap {
    points: Vec<(Tick, Tick)>,
}

impl TimeMap {
    /// From (old, new) points; points that would not keep both sides strictly increasing are
    /// dropped.
    pub fn new(mut pts: Vec<(Tick, Tick)>) -> TimeMap {
        pts.sort_by_key(|p| p.0);
        let mut points: Vec<(Tick, Tick)> = Vec::with_capacity(pts.len());
        for p in pts {
            if points.last().is_none_or(|l| p.0 > l.0 && p.1 > l.1) {
                points.push(p);
            }
        }
        TimeMap { points }
    }
    pub fn is_identity(&self) -> bool {
        self.points.iter().all(|(a, b)| a == b)
    }
    pub fn points(&self) -> &[(Tick, Tick)] {
        &self.points
    }
    pub fn map(&self, t: Tick) -> Tick {
        let (Some(first), Some(last)) = (self.points.first(), self.points.last()) else { return t };
        if t <= first.0 {
            return t + (first.1 - first.0);
        }
        if t >= last.0 {
            return t + (last.1 - last.0);
        }
        let i = self.points.partition_point(|p| p.0 <= t);
        let (Some(a), Some(b)) = (self.points.get(i.wrapping_sub(1)), self.points.get(i)) else { return t };
        let span_old = i128::from((b.0 - a.0).0).max(1);
        let span_new = i128::from((b.1 - a.1).0);
        let off = i128::from((t - a.0).0);
        let v = i128::from(a.1.0) + off * span_new / span_old;
        Tick(i64::try_from(v).unwrap_or(if v < 0 { Tick::MIN.0 } else { Tick::MAX.0 }))
    }
}

/// One beat-synced placement for the duration limits: the indexes of its first and last cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub from: usize,
    pub to: usize,
    pub min: Tick,
    pub max: Tick,
}

/// The beat of `grid` nearest `t` (its index).
pub fn nearest(grid: &[Tick], t: Tick) -> Option<usize> {
    if grid.is_empty() {
        return None;
    }
    let i = grid.partition_point(|b| *b < t);
    let after = grid.get(i).map(|b| (*b - t).abs());
    let before = i.checked_sub(1).and_then(|j| grid.get(j)).map(|b| (t - *b).abs());
    match (before, after) {
        (Some(b), Some(a)) => Some(if b <= a { i - 1 } else { i }),
        (Some(_), None) => Some(i - 1),
        (None, Some(_)) => Some(i),
        (None, None) => None,
    }
}

/// The `k`th beat of `grid`, continuing past its end at `period`.
pub fn beat(grid: &[Tick], k: usize, period: Tick) -> Option<Tick> {
    if let Some(b) = grid.get(k) {
        return Some(*b);
    }
    let last = *grid.last()?;
    let extra = i64::try_from(k.checked_sub(grid.len() - 1)?).ok()?;
    Some(Tick(last.0.saturating_add(extra.saturating_mul(period.0.max(1)))))
}

/// The typical spacing of a beat list (its median gap), or `None` with fewer than two beats.
pub fn period(grid: &[Tick]) -> Option<Tick> {
    let mut gaps: Vec<i64> = grid.windows(2).filter_map(|w| Some((*w.get(1)? - *w.first()?).0)).filter(|g| *g > 0).collect();
    if gaps.is_empty() {
        return None;
    }
    gaps.sort_unstable();
    gaps.get(gaps.len() / 2).map(|g| Tick(*g))
}

/// New times for `cuts` (strictly increasing template times at or after `anchor`, which stays
/// put): each cut's beat in `old_grid` becomes the same beat of `new_grid` (`Beats`), or the
/// beat of `new_grid` nearest the cut (`Nearest`), adjusted so every span keeps its limits and
/// at least a frame. Returns the new times (on frames of `rate`) and warnings.
pub fn quantize(
    cuts: &[Tick],
    anchor: Tick,
    old_grid: &[Tick],
    new_grid: &[Tick],
    mode: QuantizeMode,
    spans: &[Span],
    rate: FrameRate,
) -> (Vec<Tick>, Vec<String>) {
    let frame = rate.frame_duration().max(Tick(1));
    let old: Vec<Tick> = old_grid.iter().copied().filter(|b| *b >= anchor).collect();
    let new: Vec<Tick> = new_grid.iter().copied().filter(|b| *b >= anchor).collect();
    let per = period(&new).unwrap_or(frame);
    let mut out: Vec<Tick> = Vec::with_capacity(cuts.len());
    let mut warnings = Vec::new();
    for (i, &c) in cuts.iter().enumerate() {
        let want = if c == anchor || new.is_empty() {
            c
        } else {
            match mode {
                QuantizeMode::Beats => match nearest(&old, c).and_then(|k| beat(&new, k, per)) {
                    Some(t) => t,
                    None => nearest(&new, c).and_then(|k| new.get(k).copied()).unwrap_or(c),
                },
                QuantizeMode::Nearest => nearest(&new, c).and_then(|k| new.get(k).copied()).unwrap_or(c),
            }
        };
        // the limits of the spans ending at this cut
        let prev = out.last().copied();
        let mut lo = prev.map_or(Tick::MIN, |p| p + frame);
        let mut hi = Tick::MAX;
        for s in spans.iter().filter(|s| s.to == i) {
            if let Some(start) = out.get(s.from).copied() {
                lo = lo.max(start + s.min);
                hi = hi.min(start + s.max);
            }
        }
        let ok = |t: Tick| t >= lo && t <= hi;
        let mut t = rate.snap_nearest(want);
        if !ok(t) {
            // the beat nearest the wanted time that keeps the limits
            let fits = new.iter().copied().map(|b| rate.snap_nearest(b)).filter(|b| ok(*b)).min_by_key(|b| (*b - want).abs());
            t = match fits {
                Some(b) => b,
                None => {
                    if lo <= hi {
                        warnings.push(format!("cut {} could not land on a beat inside its slot's duration limits", i + 1));
                        rate.snap_nearest(want.clamp(lo, hi)).clamp(lo, hi)
                    } else {
                        warnings.push(format!("cut {} cannot keep its slot's duration limits", i + 1));
                        lo
                    }
                }
            };
        }
        if prev.is_some_and(|p| t <= p) {
            t = prev.map_or(t, |p| p + frame);
        }
        out.push(t);
    }
    (out, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_time::TICKS_PER_SECOND;

    const S: i64 = TICKS_PER_SECOND;
    const R: FrameRate = FrameRate::FPS_30;
    fn grid(bpm: f64, n: usize) -> Vec<Tick> {
        (0..n).map(|k| Tick((k as f64 * 60.0 / bpm * S as f64).round() as i64)).collect()
    }

    #[test]
    fn time_map_interpolates_and_shifts() {
        let m = TimeMap::new(vec![(Tick(0), Tick(0)), (Tick(10), Tick(20)), (Tick(20), Tick(25))]);
        assert_eq!(m.map(Tick(5)), Tick(10));
        assert_eq!(m.map(Tick(15)), Tick(22));
        assert_eq!(m.map(Tick(30)), Tick(35), "after the last point: shifted");
        assert_eq!(m.map(Tick(-4)), Tick(-4));
        assert!(TimeMap::new(vec![(Tick(3), Tick(3))]).is_identity());
        // points that would fold the map are dropped
        let f = TimeMap::new(vec![(Tick(0), Tick(0)), (Tick(10), Tick(5)), (Tick(20), Tick(4))]);
        assert_eq!(f.points().len(), 2);
    }

    #[test]
    fn cuts_keep_their_beat_numbers_on_a_slower_song() {
        // cuts every 2 beats at 120 bpm (1 s) re-quantized to 90 bpm: every 1.333 s
        let cuts: Vec<Tick> = (0..=4).map(|k| Tick(k * S)).collect();
        let (new, w) = quantize(&cuts, Tick::ZERO, &grid(120.0, 20), &grid(90.0, 20), QuantizeMode::Beats, &[], R);
        assert!(w.is_empty());
        let want: Vec<Tick> = (0..=4).map(|k| R.snap_nearest(Tick((k as f64 * 2.0 * 60.0 / 90.0 * S as f64).round() as i64))).collect();
        assert_eq!(new, want);
        // past the end of the new grid the beats continue at its tempo
        let (long, _) = quantize(&[Tick(0), Tick(9 * S)], Tick::ZERO, &grid(120.0, 20), &grid(90.0, 4), QuantizeMode::Beats, &[], R);
        assert_eq!(long[1], R.snap_nearest(Tick((18.0 * 60.0 / 90.0 * S as f64).round() as i64)));
    }

    #[test]
    fn nearest_mode_keeps_the_length() {
        let cuts = vec![Tick(0), Tick(S), Tick(2 * S)];
        let (new, _) = quantize(&cuts, Tick::ZERO, &grid(120.0, 10), &grid(100.0, 10), QuantizeMode::Nearest, &[], R);
        assert_eq!(new, vec![Tick(0), Tick(S * 6 / 5), Tick(S * 18 / 10)]);
    }

    #[test]
    fn spans_keep_their_limits() {
        let cuts = vec![Tick(0), Tick(S), Tick(2 * S)];
        let spans = [Span { from: 0, to: 1, min: Tick(S / 2), max: Tick(S) }, Span { from: 1, to: 2, min: Tick(S / 2), max: Tick(S) }];
        // a very slow song would make each span 2 s: the limits pull them back onto nearer beats
        let (new, _) = quantize(&cuts, Tick::ZERO, &grid(120.0, 10), &grid(60.0, 10), QuantizeMode::Beats, &spans, R);
        assert_eq!(new, vec![Tick(0), Tick(S), Tick(2 * S)]);
        // limits no beat can meet: clamped, with a warning
        let spans = [Span { from: 0, to: 1, min: Tick(S / 4), max: Tick(S / 3) }];
        let (new, w) = quantize(&cuts[..2], Tick::ZERO, &grid(120.0, 10), &grid(20.0, 10), QuantizeMode::Beats, &spans, R);
        assert_eq!(new[1], Tick(S / 3));
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn hostile_inputs_never_panic() {
        let huge = [Tick(i64::MAX / 4), Tick(i64::MIN / 4), Tick(0)];
        let _ = quantize(&huge, Tick::ZERO, &huge, &huge, QuantizeMode::Beats, &[Span { from: 9, to: 0, min: Tick::MAX, max: Tick::MIN }], R);
        let _ = quantize(&[Tick(5)], Tick::ZERO, &[], &[], QuantizeMode::Nearest, &[], R);
        let m = TimeMap::new(vec![(Tick::MIN, Tick::MAX), (Tick::MAX, Tick::MIN)]);
        let _ = m.map(Tick(0));
        assert_eq!(beat(&[], 3, Tick(1)), None);
        assert_eq!(beat(&[Tick(0)], 1 << 40, Tick(i64::MAX)).map(|t| t.0 > 0), Some(true));
        let _ = beat(&[Tick(0)], usize::MAX, Tick(i64::MAX));
    }
}
