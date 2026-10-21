//! Applied transitions: where one may sit over its cut, its alignment, and moving / resizing /
//! removing one (Effect Controls' Duration and Alignment, dragging its edges or its middle in the
//! Timeline, Delete), and where one goes when it is snapped to the music's beat
//! ([`snap_to_beat`]: its midpoint on a beat, its length a musical fraction of a beat).
//!
//! Premiere (Align transitions; Change transition duration): a transition between two clips stays
//! over their cut, "Center at Cut", "Start at Cut", "End at Cut" or, once dragged off-centre,
//! "Custom Start". Changing its duration moves both ends equally for Center at Cut / Custom Start,
//! only the end for Start at Cut and only the beginning for End at Cut. A one-sided transition (a
//! fade from or to nothing) starts or ends at its clip's edge.
//!
//! Premiere limits a transition by the media its clips have beyond the cut (handles). FilmCraft
//! does not check handles when applying one, so here a transition may cover at most the clips it
//! joins: from the outgoing clip's start to the incoming clip's end.

use filmcraft_project::{Sequence, Track, Transition, TransitionAlign, TransitionId};
use filmcraft_time::Tick;

use crate::{Edge, EditError, Result};

/// How a transition sits on its cut, as Effect Controls' Alignment menu shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alignment {
    Center,
    Start,
    End,
    /// Premiere's "Custom Start": dragged to a place over the cut that is none of the others.
    Custom,
}

impl Alignment {
    pub const MENU: [Alignment; 3] = [Alignment::Center, Alignment::Start, Alignment::End];

    pub fn id(self) -> &'static str {
        match self {
            Alignment::Center => "center",
            Alignment::Start => "start",
            Alignment::End => "end",
            Alignment::Custom => "custom",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Alignment::Center => "Center at Cut",
            Alignment::Start => "Start at Cut",
            Alignment::End => "End at Cut",
            Alignment::Custom => "Custom Start",
        }
    }

    /// `center`, `start` or `end` (Custom Start is reached by moving a transition, not chosen).
    pub fn parse(s: &str) -> Option<Alignment> {
        Alignment::MENU.into_iter().find(|a| a.id().eq_ignore_ascii_case(s) || a.label().eq_ignore_ascii_case(s))
    }
}

/// Where a transition may be: `(earliest start, cut, latest end)`. Between two clips that is the
/// outgoing clip's start, the cut and the incoming clip's end; for a fade to or from nothing the cut
/// is its clip's edge. `None` when its clips are gone.
pub fn bounds(track: &Track, tr: &Transition) -> Option<(Tick, Tick, Tick)> {
    let from = tr.from.and_then(|c| track.item(c));
    let to = tr.to.and_then(|c| track.item(c));
    match (from, to) {
        (Some(f), Some(t)) => Some((f.start, t.start, t.end())),
        (None, Some(t)) => Some((t.start, t.start, t.end())),
        (Some(f), None) => Some((f.start, f.end(), f.end())),
        (None, None) => None,
    }
}

/// A transition's alignment on `cut`. Centred means within half a `frame` of the cut, as
/// centring an odd number of frames lands on a frame boundary.
pub fn alignment(tr: &Transition, cut: Tick, frame: Tick) -> Alignment {
    match (tr.from, tr.to) {
        (None, Some(_)) => return Alignment::Start,
        (Some(_), None) => return Alignment::End,
        _ => {}
    }
    if tr.start == cut {
        Alignment::Start
    } else if tr.end() == cut {
        Alignment::End
    } else if (tr.start.0.saturating_mul(2).saturating_add(tr.duration.0) - cut.0.saturating_mul(2)).abs() <= frame.0.max(0) {
        Alignment::Center
    } else {
        Alignment::Custom
    }
}

fn find(seq: &Sequence, id: TransitionId) -> Option<(&Track, &Transition)> {
    seq.all_tracks().find_map(|t| t.transitions.iter().find(|x| x.id == id).map(|x| (t, x)))
}

/// Where a transition would start for `duration` and `align`, keeping its cut (and, for Center
/// at Cut / Custom Start, its centre; `snap` puts a centred start on a frame).
pub fn start_for(seq: &Sequence, id: TransitionId, duration: Tick, align: Alignment, frame: Tick, snap: impl Fn(Tick) -> Tick) -> Option<Tick> {
    let (track, tr) = find(seq, id)?;
    let (_, cut, _) = bounds(track, tr)?;
    let one_sided = alignment(tr, cut, frame);
    let align = match one_sided {
        Alignment::Start | Alignment::End if tr.from.is_none() || tr.to.is_none() => one_sided,
        _ => align,
    };
    Some(match align {
        Alignment::Start => cut,
        Alignment::End => cut - duration,
        Alignment::Center => snap(cut - duration.mul_ratio(1, 2)),
        // Custom Start: the ends move equally, around where it is now
        Alignment::Custom => snap(tr.start + tr.duration.mul_ratio(1, 2) - duration.mul_ratio(1, 2)),
    })
}

/// Move / resize a transition to cover `start .. start + duration`. Refused (leaving the sequence
/// as it was) when its track is locked, it would be shorter than `min`, it would leave its clips,
/// or it would no longer touch its cut (a one-sided one must keep its edge).
pub fn set_span(seq: &mut Sequence, id: TransitionId, start: Tick, duration: Tick, min: Tick, frame: Tick) -> Result<()> {
    let (track, tr) = find(seq, id).ok_or_else(|| EditError::Other(format!("no transition {}", id.0)))?;
    if track.locked {
        return Err(EditError::Locked);
    }
    let (lo, cut, hi) = bounds(track, tr).ok_or_else(|| EditError::Other("the transition's clips are gone".into()))?;
    if duration < min.max(Tick(1)) {
        return Err(EditError::Other("a transition lasts at least one frame".into()));
    }
    let end = start.0.checked_add(duration.0).map(Tick).ok_or_else(|| EditError::Other("transition out of range".into()))?;
    if start < lo || end > hi {
        return Err(EditError::Other("a transition can't extend past the clips it joins".into()));
    }
    let on_cut = match (tr.from, tr.to) {
        (None, Some(_)) => start == cut,
        (Some(_), None) => end == cut,
        _ => start <= cut && cut <= end,
    };
    if !on_cut {
        return Err(EditError::Other(if tr.from.is_some() && tr.to.is_some() {
            "a transition between two clips must stay over their cut".into()
        } else {
            "a one-sided transition stays on its clip's edge".into()
        }));
    }
    let tid = track.id;
    let Some(x) = seq.track_mut(tid).and_then(|t| t.transitions.iter_mut().find(|x| x.id == id)) else {
        return Err(EditError::Other(format!("no transition {}", id.0)));
    };
    x.start = start;
    x.duration = duration;
    // the stored alignment (interchange): Custom Start has no variant, its span says it all
    x.align = match alignment(x, cut, frame) {
        Alignment::Start => TransitionAlign::StartAtCut,
        Alignment::End => TransitionAlign::EndAtCut,
        Alignment::Center | Alignment::Custom => TransitionAlign::CenterAtCut,
    };
    Ok(())
}

/// Where dragging a transition by `delta` in the Timeline puts it, as `(start, duration)`. Its
/// middle (`edge` None) slides it over the cut. An end trims that end only (Premiere shows the
/// Trim-In / Trim-Out icon there): the other end stays, whatever the alignment, so a Center at
/// Cut transition becomes Custom Start. (A typed duration, in Effect Controls or Set Transition
/// Duration, follows the alignment instead: see [`start_for`].) A fade to or from nothing keeps
/// its clip-edge end, and does not slide. Always clamped to where it may be ([`bounds`], over the
/// cut, at least one `frame`). `None` when its clips are gone.
pub fn drag_span(track: &Track, tr: &Transition, edge: Option<Edge>, delta: Tick, frame: Tick) -> Option<(Tick, Tick)> {
    let (lo, cut, hi) = bounds(track, tr)?;
    let frame = frame.max(Tick(1));
    let (s0, e0, d0) = (tr.start, tr.end(), tr.duration);
    let one_sided = tr.from.is_none() || tr.to.is_none();
    let unchanged = Some((s0, d0));
    match edge {
        None if one_sided => unchanged,
        None => {
            let (min_s, max_s) = (lo.max(cut - d0), (hi - d0).min(cut));
            if min_s > max_s { unchanged } else { Some(((s0 + delta).clamp(min_s, max_s), d0)) }
        }
        Some(Edge::In) if tr.from.is_none() => unchanged,
        Some(Edge::In) => {
            let max_s = cut.min(e0 - frame).max(lo);
            let st = (s0 + delta).clamp(lo, max_s);
            Some((st, e0 - st))
        }
        Some(Edge::Out) if tr.to.is_none() => unchanged,
        Some(Edge::Out) => {
            let min_e = cut.max(s0 + frame).min(hi);
            let en = (e0 + delta).clamp(min_e, hi);
            Some((s0, en - s0))
        }
    }
}

/// Remove transitions (on unlocked tracks). Returns how many went.
pub fn remove(seq: &mut Sequence, ids: &[TransitionId]) -> Result<usize> {
    if seq.all_tracks().any(|t| t.locked && t.transitions.iter().any(|x| ids.contains(&x.id))) {
        return Err(EditError::Locked);
    }
    let mut n = 0;
    for t in seq.all_tracks_mut() {
        let before = t.transitions.len();
        t.transitions.retain(|x| !ids.contains(&x.id));
        n += before - t.transitions.len();
    }
    if n == 0 { Err(EditError::Nothing) } else { Ok(n) }
}

/// Lengths a transition snapped to the beat may take, as fractions of a beat (numerator,
/// denominator): a quarter, a half, one, two or four beats.
pub const BEAT_FRACTIONS: [(i64, i64); 5] = [(1, 4), (1, 2), (1, 1), (2, 1), (4, 1)];

/// How many of the beats nearest a transition's midpoint [`snap_to_beat`] tries.
const BEATS_TRIED: usize = 6;

/// Where [`snap_to_beat`] puts a transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BeatSnap {
    pub start: Tick,
    pub duration: Tick,
    /// The beat its midpoint went to (None for a fade to or from nothing, which keeps its clip
    /// edge and only takes a musical length).
    pub beat: Option<Tick>,
    /// Its length in beats, one of [`BEAT_FRACTIONS`].
    pub beats: (i64, i64),
}

/// Snap transition `tr` on `track` to a beat grid: `beats` are the music's beats in timeline
/// ticks (sorted, distinct) and `period` the length of one beat. Its midpoint goes to the nearest
/// beat it can reach and its duration becomes the musical fraction of a beat
/// ([`BEAT_FRACTIONS`]) nearest its own length, both on whole frames (`snap` rounds a time to the
/// nearest frame boundary, `frame` is one frame), so the midpoint ends within half a frame of the
/// beat. It stays where a transition may be ([`bounds`]: over its cut, inside the clips it joins)
/// and clear of `others` (the spans of the other transitions on its track). A fade to or from
/// nothing keeps its clip edge and only takes a musical length. None when no beat in reach works
/// (or the clips are gone, or `period` is not positive).
pub fn snap_to_beat(
    track: &Track,
    tr: &Transition,
    beats: &[Tick],
    period: Tick,
    frame: Tick,
    snap: impl Fn(Tick) -> Tick,
    others: &[(Tick, Tick)],
) -> Option<BeatSnap> {
    let (lo, cut, hi) = bounds(track, tr)?;
    if period <= Tick::ZERO {
        return None;
    }
    let frame = frame.max(Tick(1));
    let d0 = tr.duration.max(frame);
    // the musical lengths on whole frames, the one nearest the current length (by ratio) first
    let mut lengths: Vec<((i64, i64), Tick)> = BEAT_FRACTIONS.iter().map(|&(n, d)| ((n, d), snap(period.mul_ratio(n, d)).max(frame))).collect();
    lengths.sort_by(|a, b| ratio_distance(a.1, d0).total_cmp(&ratio_distance(b.1, d0)));
    let clear = |s: Tick, e: Tick| others.iter().all(|&(os, oe)| e <= os || oe <= s);
    let fits = |s: Tick, d: Tick| {
        let e = s + d;
        lo <= s && e <= hi && clear(s, e)
    };
    match (tr.from, tr.to) {
        (None, Some(_)) => lengths.iter().find(|(_, d)| fits(cut, *d)).map(|&(beats, d)| BeatSnap { start: cut, duration: d, beat: None, beats }),
        (Some(_), None) => lengths.iter().find(|(_, d)| fits(cut - *d, *d)).map(|&(beats, d)| BeatSnap { start: cut - d, duration: d, beat: None, beats }),
        _ => {
            let mid = tr.start + tr.duration.mul_ratio(1, 2);
            for b in nearest(beats, mid, BEATS_TRIED) {
                for &(f, d) in &lengths {
                    let s = snap(b - d.mul_ratio(1, 2));
                    if s <= cut && cut <= s + d && fits(s, d) {
                        return Some(BeatSnap { start: s, duration: d, beat: Some(b), beats: f });
                    }
                }
            }
            None
        }
    }
}

/// How far apart two lengths are, as the size of their ratio.
fn ratio_distance(a: Tick, b: Tick) -> f64 {
    let (a, b) = (a.0.max(1) as f64, b.0.max(1) as f64);
    (a / b).ln().abs()
}

/// Up to `n` of `beats` (sorted) nearest `t`, nearest first (the earlier one on a tie).
fn nearest(beats: &[Tick], t: Tick, n: usize) -> Vec<Tick> {
    let i = beats.partition_point(|b| *b < t);
    let (mut left, mut right) = (i, i);
    let mut out = Vec::with_capacity(n.min(beats.len()));
    while out.len() < n {
        let l = left.checked_sub(1).and_then(|k| beats.get(k).map(|b| (k, *b)));
        let r = beats.get(right).copied();
        match (l, r) {
            (Some((k, a)), Some(b)) => {
                if (t - a) <= (b - t) {
                    out.push(a);
                    left = k;
                } else {
                    out.push(b);
                    right += 1;
                }
            }
            (Some((k, a)), None) => {
                out.push(a);
                left = k;
            }
            (None, Some(b)) => {
                out.push(b);
                right += 1;
            }
            (None, None) => break,
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_project::{ClipId, ItemId, Label, Project, SequenceSettings, TrackItem, find_effect};
    use filmcraft_time::FrameRate;

    const R: FrameRate = FrameRate::FPS_24;

    fn f(n: i64) -> Tick {
        R.tick_of(n)
    }

    fn clip(id: u64, start: i64, dur: i64) -> TrackItem {
        TrackItem {
            id: ClipId(id),
            item: ItemId(1),
            name: format!("c{id}"),
            label: Label::Iris,
            start: f(start),
            duration: f(dur),
            source_in: Tick::ZERO,
            speed: 1.0,
            reverse: false,
            enabled: true,
            link: None,
            group: None,
            effects: vec![],
            markers: vec![],
            gain_db: 0.0,
            frame_hold: None,
            scale_to_frame: false,
            essential: None,
            multicam: None,
            time_interpolation: Default::default(),
            maintain_pitch: false,
            hold_filters: false,
            field_options: None,
            source_channels: Vec::new(),
            audio_stream: 0,
            graphic: None,
        }
    }

    fn transition(id: u64, start: i64, dur: i64, from: Option<u64>, to: Option<u64>) -> Transition {
        // any effect: these tests are about where the transition sits
        let effect = find_effect("opacity").unwrap().instance();
        Transition {
            id: TransitionId(id),
            effect,
            start: f(start),
            duration: f(dur),
            from: from.map(ClipId),
            to: to.map(ClipId),
            align: TransitionAlign::CenterAtCut,
            reverse: false,
            sound: None,
        }
    }

    /// V1: c1 0..48, c2 48..96, c3 96..144; a centred 24-frame transition on 48 (id 1), a fade in
    /// at 0 (id 2) and a fade out at 144 (id 3).
    fn seq() -> Sequence {
        let mut p = Project::new("t");
        let s = p.new_sequence("s", SequenceSettings { frame_rate: R, ..Default::default() }, 1, 0, None);
        let mut q = p.sequence(s).unwrap().clone();
        let t = &mut q.video_tracks[0];
        t.items = vec![clip(1, 0, 48), clip(2, 48, 48), clip(3, 96, 48)];
        t.transitions = vec![transition(1, 36, 24, Some(1), Some(2)), transition(2, 0, 12, None, Some(1)), transition(3, 132, 12, Some(3), None)];
        q
    }

    fn span(q: &Sequence, id: u64) -> (Tick, Tick) {
        let x = q.video_tracks[0].transitions.iter().find(|x| x.id.0 == id).unwrap();
        (x.start, x.duration)
    }

    fn align(q: &Sequence, id: u64) -> Alignment {
        let (t, x) = find(q, TransitionId(id)).unwrap();
        let (_, cut, _) = bounds(t, x).unwrap();
        alignment(x, cut, f(1))
    }

    #[test]
    fn alignment_reads_the_span() {
        let mut q = seq();
        assert_eq!((align(&q, 1), align(&q, 2), align(&q, 3)), (Alignment::Center, Alignment::Start, Alignment::End));
        set_span(&mut q, TransitionId(1), f(48), f(24), f(1), f(1)).unwrap();
        assert_eq!(align(&q, 1), Alignment::Start);
        set_span(&mut q, TransitionId(1), f(24), f(24), f(1), f(1)).unwrap();
        assert_eq!(align(&q, 1), Alignment::End);
        set_span(&mut q, TransitionId(1), f(40), f(24), f(1), f(1)).unwrap();
        assert_eq!(align(&q, 1), Alignment::Custom);
        // an odd centred duration sits half a frame off the cut and is still Center at Cut
        set_span(&mut q, TransitionId(1), f(42), f(13), f(1), f(1)).unwrap();
        assert_eq!(align(&q, 1), Alignment::Center);
    }

    #[test]
    fn duration_changes_follow_the_alignment() {
        let q = seq();
        let snap = |t: Tick| R.snap(t);
        // Center at Cut: both ends move
        assert_eq!(start_for(&q, TransitionId(1), f(36), Alignment::Center, f(1), snap), Some(f(30)));
        // Start at Cut: only the end moves; End at Cut: only the beginning
        assert_eq!(start_for(&q, TransitionId(1), f(36), Alignment::Start, f(1), snap), Some(f(48)));
        assert_eq!(start_for(&q, TransitionId(1), f(36), Alignment::End, f(1), snap), Some(f(12)));
        // a one-sided transition keeps its edge whatever is asked
        assert_eq!(start_for(&q, TransitionId(2), f(20), Alignment::Center, f(1), snap), Some(f(0)));
        assert_eq!(start_for(&q, TransitionId(3), f(20), Alignment::Center, f(1), snap), Some(f(124)));
    }

    #[test]
    fn custom_start_resizes_around_its_middle() {
        let mut q = seq();
        set_span(&mut q, TransitionId(1), f(40), f(20), f(1), f(1)).unwrap();
        assert_eq!(start_for(&q, TransitionId(1), f(10), Alignment::Custom, f(1), |t| R.snap(t)), Some(f(45)));
    }

    #[test]
    fn spans_stay_on_the_cut_and_inside_the_clips() {
        let mut q = seq();
        let before = q.clone();
        for (id, start, dur) in [
            (1, -1, 24),  // before the outgoing clip
            (1, 40, 60),  // past the incoming clip's end
            (1, 10, 20),  // entirely before the cut
            (1, 50, 10),  // entirely after it
            (1, 40, 0),   // empty
            (2, 1, 12),   // a fade in must start at its clip's start
            (3, 120, 12), // a fade out must end at its clip's end
            (9, 0, 12),   // no such transition
        ] {
            assert!(set_span(&mut q, TransitionId(id), f(start), f(dur), f(1), f(1)).is_err(), "{id} {start} {dur}");
            assert_eq!(q, before, "refused changes leave the sequence as it was");
        }
        // the whole of both clips is allowed
        set_span(&mut q, TransitionId(1), f(0), f(96), f(1), f(1)).unwrap();
        assert_eq!(span(&q, 1), (f(0), f(96)));
        // Tick::MAX never overflows
        assert!(set_span(&mut q, TransitionId(1), Tick(i64::MAX - 1), Tick(i64::MAX), f(1), f(1)).is_err());
    }

    fn drag(q: &Sequence, id: u64, edge: Option<Edge>, frames: i64) -> (i64, i64) {
        let (t, x) = find(q, TransitionId(id)).unwrap();
        let (st, d) = drag_span(t, x, edge, f(frames), f(1)).unwrap();
        (R.frame_at(st), R.frame_at(d))
    }

    #[test]
    fn dragging_a_transition() {
        let mut q = seq();
        // centred (36..60 over the cut at 48): an end trims that end only, the other stays
        assert_eq!(drag(&q, 1, Some(Edge::In), -4), (32, 28));
        assert_eq!(drag(&q, 1, Some(Edge::Out), -10), (36, 14));
        assert_eq!(drag(&q, 1, Some(Edge::Out), -100), (36, 12), "the end stops at the cut");
        assert_eq!(drag(&q, 1, Some(Edge::Out), 100), (36, 60), "and at the incoming clip's end");
        assert_eq!(drag(&q, 1, Some(Edge::In), -100), (0, 60), "the start at the outgoing clip's start");
        // the middle slides it, staying over the cut and inside the clips
        assert_eq!(drag(&q, 1, None, 5), (41, 24));
        assert_eq!(drag(&q, 1, None, 100), (48, 24));
        assert_eq!(drag(&q, 1, None, -100), (24, 24));
        // not centred: the other end stays
        set_span(&mut q, TransitionId(1), f(40), f(20), f(1), f(1)).unwrap();
        assert_eq!(drag(&q, 1, Some(Edge::In), -4), (36, 24));
        assert_eq!(drag(&q, 1, Some(Edge::In), 100), (48, 12), "the start stops at the cut");
        assert_eq!(drag(&q, 1, Some(Edge::Out), 6), (40, 26));
        assert_eq!(drag(&q, 1, Some(Edge::Out), -100), (40, 8), "the end stops at the cut");
        // a fade in keeps its start, a fade out its end; neither slides
        assert_eq!(drag(&q, 2, Some(Edge::In), 3), (0, 12));
        assert_eq!(drag(&q, 2, Some(Edge::Out), 3), (0, 15));
        assert_eq!(drag(&q, 2, None, 3), (0, 12));
        assert_eq!(drag(&q, 3, Some(Edge::Out), 3), (132, 12));
        assert_eq!(drag(&q, 3, Some(Edge::In), -3), (129, 15));
        // whatever comes out is a span set_span accepts
        for (id, edge, n) in [(1, Some(Edge::In), -70), (1, None, 33), (2, Some(Edge::Out), 99), (3, Some(Edge::In), -99)] {
            let (t, x) = find(&q, TransitionId(id)).unwrap();
            let (st, d) = drag_span(t, x, edge, f(n), f(1)).unwrap();
            let mut c = q.clone();
            set_span(&mut c, TransitionId(id), st, d, f(1), f(1)).unwrap();
        }
    }

    #[test]
    fn locked_tracks_refuse() {
        let mut q = seq();
        q.video_tracks[0].locked = true;
        assert_eq!(set_span(&mut q, TransitionId(1), f(40), f(20), f(1), f(1)), Err(EditError::Locked));
        assert_eq!(remove(&mut q, &[TransitionId(1)]), Err(EditError::Locked));
    }

    #[test]
    fn remove_takes_only_the_named_transitions() {
        let mut q = seq();
        assert_eq!(remove(&mut q, &[TransitionId(1), TransitionId(3)]), Ok(2));
        let left: Vec<u64> = q.video_tracks[0].transitions.iter().map(|x| x.id.0).collect();
        assert_eq!(left, vec![2]);
        assert_eq!(q.video_tracks[0].items.len(), 3, "the clips stay");
        assert_eq!(remove(&mut q, &[TransitionId(1)]), Err(EditError::Nothing));
    }

    // ---- snapping to the beat (roadmap C4)

    /// A beat grid: `n` beats of `period` from `first`.
    fn grid(first: Tick, period: Tick, n: i64) -> Vec<Tick> {
        (0..n).map(|k| first + Tick(period.0 * k)).collect()
    }

    fn snap_nearest(t: Tick) -> Tick {
        R.snap_nearest(t)
    }

    #[test]
    fn a_transition_moves_its_midpoint_to_the_nearest_beat() {
        let q = seq();
        let t = &q.video_tracks[0];
        let tr = t.transitions.iter().find(|x| x.id.0 == 1).unwrap();
        // 120 BPM (12 frames a beat) with the beats 5 frames off the frame grid's cut at 48
        let period = f(12);
        let beats = grid(f(5), period, 12);
        let s = snap_to_beat(t, tr, &beats, period, f(1), snap_nearest, &[]).unwrap();
        // the nearest beat to the midpoint 48 is 53; 24 frames (2 beats) is the length nearest 24
        assert_eq!(s.beat, Some(f(53)));
        assert_eq!((s.start, s.duration, s.beats), (f(41), f(24), (2, 1)));
        assert!(s.start <= f(48) && f(48) <= s.start + s.duration, "still over its cut");
        // a short transition (half a beat) whose nearest beat is 5 frames past its cut: half a
        // beat on that beat would leave the cut, so it takes the next nearest length, one beat
        let mut short = tr.clone();
        short.start = f(45);
        short.duration = f(6);
        let s = snap_to_beat(t, &short, &beats, period, f(1), snap_nearest, &[]).unwrap();
        assert_eq!((s.start, s.duration, s.beat, s.beats), (f(47), f(12), Some(f(53)), (1, 1)));
        // ...and keeps half a beat when the beat is close enough to its cut
        let near = grid(f(2), period, 12);
        let s = snap_to_beat(t, &short, &near, period, f(1), snap_nearest, &[]).unwrap();
        assert_eq!((s.duration, s.beats, s.beat), (f(6), (1, 2), Some(f(50))));
        let mid = s.start + s.duration.mul_ratio(1, 2);
        assert!((mid - f(50)).abs() <= f(1).mul_ratio(1, 2), "midpoint {mid:?}");
        assert!(s.start <= f(48) && f(48) <= s.start + s.duration);
    }

    #[test]
    fn odd_tempos_land_within_half_a_frame() {
        for (rate, bpm) in [(FrameRate::FPS_29_97, 77.7), (FrameRate::FPS_23_976, 203.0), (FrameRate::FPS_25, 41.3), (FrameRate::FPS_59_94, 133.33)] {
            let frame = rate.frame_duration();
            let fr = |n: i64| rate.tick_of(n);
            let mut q = seq();
            let t = &mut q.video_tracks[0];
            t.items = vec![clip(1, 0, 0), clip(2, 0, 0)];
            t.items[0].start = Tick::ZERO;
            t.items[0].duration = fr(300);
            t.items[1].start = fr(300);
            t.items[1].duration = fr(300);
            t.transitions = vec![transition(1, 0, 0, Some(1), Some(2))];
            t.transitions[0].start = fr(290);
            t.transitions[0].duration = fr(20);
            let period = Tick::from_seconds_f64(60.0 / bpm);
            let beats = grid(Tick::from_seconds_f64(0.137), period, 2000);
            let s = snap_to_beat(t, &t.transitions[0], &beats, period, frame, |x| rate.snap_nearest(x), &[]).unwrap();
            let b = s.beat.unwrap();
            let mid = s.start + s.duration.mul_ratio(1, 2);
            assert!((mid - b).abs() <= frame.mul_ratio(1, 2), "{rate:?} {bpm}: midpoint off the beat by {:?}", mid - b);
            // the length is a musical fraction of the beat, rounded to whole frames
            let (n, d) = s.beats;
            assert!((s.duration - period.mul_ratio(n, d)).abs() <= frame.mul_ratio(1, 2) + Tick(1), "{rate:?} {bpm}: {:?}", s.duration);
            assert_eq!(s.duration, rate.snap_nearest(s.duration), "whole frames");
            assert_eq!(s.start, rate.snap_nearest(s.start), "on a frame");
            assert!(s.start <= fr(300) && fr(300) <= s.start + s.duration);
        }
    }

    #[test]
    fn fades_keep_their_edge_and_neighbours_are_respected() {
        let q = seq();
        let t = &q.video_tracks[0];
        let period = f(16);
        let beats = grid(f(3), period, 12);
        // the fade in at 0 keeps its start and takes the nearest musical length (16 frames)
        let fade_in = t.transitions.iter().find(|x| x.id.0 == 2).unwrap();
        let s = snap_to_beat(t, fade_in, &beats, period, f(1), snap_nearest, &[]).unwrap();
        assert_eq!((s.start, s.duration, s.beat), (f(0), f(16), None));
        let fade_out = t.transitions.iter().find(|x| x.id.0 == 3).unwrap();
        let s = snap_to_beat(t, fade_out, &beats, period, f(1), snap_nearest, &[]).unwrap();
        assert_eq!((s.start + s.duration, s.beat), (f(144), None));
        // a neighbour in the way: the span must not overlap it
        let tr = t.transitions.iter().find(|x| x.id.0 == 1).unwrap();
        let s = snap_to_beat(t, tr, &beats, period, f(1), snap_nearest, &[(f(30), f(46))]).unwrap();
        assert!(s.start >= f(46), "{s:?}");
        // nothing fits: every beat out of reach of the cut
        assert!(snap_to_beat(t, tr, &[f(1000), f(1016)], period, f(1), snap_nearest, &[]).is_none());
    }

    #[test]
    fn hostile_grids_never_panic() {
        let q = seq();
        let t = &q.video_tracks[0];
        let tr = t.transitions.iter().find(|x| x.id.0 == 1).unwrap();
        let huge = [Tick(i64::MIN), Tick(-1), Tick(0), Tick(i64::MAX / 2), Tick(i64::MAX)];
        for period in [Tick(i64::MIN), Tick(-5), Tick(0), Tick(1), f(1), Tick(i64::MAX)] {
            for beats in [&[][..], &huge[..], &[f(48)][..], &[f(48), f(48), f(48)][..]] {
                for frame in [Tick(0), Tick(-1), f(1), Tick(i64::MAX)] {
                    if let Some(s) = snap_to_beat(t, tr, beats, period, frame, snap_nearest, &[]) {
                        assert!(s.duration > Tick::ZERO && s.start >= Tick::ZERO && s.start <= f(48) && f(48) <= s.start + s.duration, "{s:?}");
                    }
                }
            }
        }
        // nearest beats: in order of distance, never more than asked
        let b = [f(0), f(10), f(20), f(30)];
        assert_eq!(nearest(&b, f(19), 3), vec![f(20), f(10), f(30)]);
        assert_eq!(nearest(&b, f(-100), 2), vec![f(0), f(10)]);
        assert_eq!(nearest(&b, f(100), 9), vec![f(30), f(20), f(10), f(0)]);
        assert!(nearest(&[], f(1), 4).is_empty());
    }
}
