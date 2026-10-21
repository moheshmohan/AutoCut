//! The blurred copies behind pictures shown whole — the pillars beside a portrait picture in a
//! wide frame, the bars above and below a wide one in a vertical frame — kept under the pictures
//! through their transitions. Auto Reframe's Blur Fill track ([`crate::reframe`]) and AutoCut's
//! showcase both put such a copy on the track below its picture, for the picture's own span; both
//! plan what the transitions need with [`pillar_edits`] (the reframe makes the plan on the
//! sequence itself, the showcase through commands).
//!
//! Why: a transition on the pictures' track runs past the cut on both sides (the clips' handles).
//! Out of a picture shown whole, its second half had nothing under the outgoing picture's empty
//! sides — black there until the next picture covered them; into one, the first half had nothing
//! under the incoming picture's sides. And a dip to black renders its colour opaque over the whole
//! frame (as in Premiere), so on the pictures' track it blacked the sides out at its first frame
//! while the picture was still fading. So each copy reaches on to the end of the transition out
//! of its picture and back to the start of the one into it, as far as its media and the copies
//! beside it allow; two copies that meet at the cut dissolve into each other over the pictures'
//! transition; and a dip next to a picture shown whole goes down to the copies' track, the
//! pictures fading out and in over its halves (Opacity keyframes), so the fill dips with the
//! picture. Copies inside a picture (a reframe's blur-filled stretch between followed ones) and
//! the eased changes between them are left as they are.

use std::collections::HashSet;

use filmcraft_edit::{Edge, move_edge};
use filmcraft_project::vtransition::find_transition;
use filmcraft_project::{ClipId, EffectKind, Param, ParamValue, Sequence, Track, TrackItem, Transition, TransitionAlign, TransitionId};
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick};

/// A blurred copy on its track: its clip id and span, and how far its media lets it reach
/// (earliest start, latest end), ticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub clip: u64,
    pub start: i64,
    pub end: i64,
    pub earliest: i64,
    pub latest: i64,
}

impl Span {
    /// A copy whose media reaches as far as it needs (a still, or one the caller's edits stop).
    pub fn new(clip: u64, start: i64, end: i64) -> Self {
        Self { clip, start, end, earliest: i64::MIN, latest: i64::MAX }
    }
}

/// A transition on the pictures' track: its id and span (ticks), the pictures it goes from and
/// to, and — a dip to black or white to take down to the copies — the dip ([`dip_of`]). A dip
/// already on the copies' track (moved there by an earlier run) is a join too, with its own id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Join {
    pub id: u64,
    pub start: i64,
    pub end: i64,
    pub from: Option<u64>,
    pub to: Option<u64>,
    pub dip: Option<&'static str>,
}

/// A picture and a blurred copy of it on the track below: whether the copy is under the
/// picture's first frame (`head`: it may reach back under the transition into the picture) and
/// under its last (`tail`: on under the transition out of it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Backdrop {
    pub picture: u64,
    pub copy: u64,
    pub head: bool,
    pub tail: bool,
}

impl Backdrop {
    /// A copy under the whole of its picture (the showcase's).
    pub fn whole(picture: u64, copy: u64) -> Self {
        Self { picture, copy, head: true, tail: true }
    }
}

/// The dips taken down to the copies' track, by the transition's effect id; a dip's colour is
/// opaque, over the whole frame.
pub fn dip_of(effect: &str) -> Option<&'static str> {
    match effect {
        "dip_to_black" | "dip_to_black_legacy" => Some("dip_to_black"),
        "dip_to_white" => Some("dip_to_white"),
        _ => None,
    }
}

/// An edit that keeps a picture's blurred copy under its transitions ([`pillar_edits`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PillarEdit {
    /// Move the copy's Out edge `by` ticks later: on until the transition out of its picture ends.
    Out { clip: u64, by: i64 },
    /// Move its In edge `by` ticks earlier: there when the transition into its picture begins.
    In { clip: u64, by: i64 },
    /// `effect` on the copies' track over `[start, end)`, for join `join`: from copy `from` into
    /// copy `to` where both are given (they meet at the cut), else out of or into the one given,
    /// at its own edge (the span then ends or starts there).
    Copies { join: u64, effect: &'static str, from: Option<u64>, to: Option<u64>, start: i64, end: i64 },
    /// The pictures' dip `transition` taken off: picture `from` fades out over its first half and
    /// `to` in over its second (Opacity keyframes), over the dip on the copies' track (always
    /// followed by that dip's [`PillarEdit::Copies`]).
    Fades { transition: u64, from: Option<u64>, to: Option<u64>, start: i64, end: i64 },
}

/// Plan the copies through the pictures' transitions (see the module docs): `copies`, the
/// copies' track; `joins`, the transitions in the order they are met; `backdrops`, which copy is
/// under which picture's edges. Each copy reaches on to the end of the transition out of its
/// picture and back to the start of the one into it, as far as the copies beside it leave room
/// and its media lets it; two copies that meet at the cut inside a transition dissolve into each
/// other over it; a dip next to a picture shown whole goes to the copies' track (each copy dipping
/// at its own edge, or the two together), the pictures fading over its halves — only where it
/// does go there: a dip whose copies have nowhere to take it stays as it is. Hostile spans are
/// skipped.
pub fn pillar_edits(copies: &[Span], joins: &[Join], backdrops: &[Backdrop]) -> Vec<PillarEdit> {
    let mut track: Vec<Span> = copies.iter().copied().filter(|s| s.end > s.start).collect();
    track.sort_by_key(|s| (s.start, s.end));
    // the copy under picture `p`'s last frame (`tail`) or first
    let copy_of = |p: Option<u64>, tail: bool| p.and_then(|p| backdrops.iter().find(|b| b.picture == p && if tail { b.tail } else { b.head }).map(|b| b.copy));
    let at = |track: &[Span], copy: Option<u64>| copy.and_then(|c| track.iter().position(|s| s.clip == c));
    let mut edits = Vec::new();
    for j in joins.iter().filter(|j| j.end > j.start) {
        let (a, b) = (copy_of(j.from, true), copy_of(j.to, false));
        // the outgoing picture's copy, on to the transition's end
        if let Some(i) = at(&track, a) {
            let room = track.get(i + 1).map_or(i64::MAX, |s| s.start);
            if let Some(s) = track.get_mut(i) {
                let to = j.end.min(room).min(s.latest);
                if to > s.end {
                    edits.push(PillarEdit::Out { clip: s.clip, by: to.saturating_sub(s.end) });
                    s.end = to;
                }
            }
        }
        // the incoming picture's copy, back to the transition's start
        if let Some(i) = at(&track, b) {
            let room = i.checked_sub(1).and_then(|k| track.get(k)).map_or(i64::MIN, |s| s.end);
            if let Some(s) = track.get_mut(i) {
                let from = j.start.max(room).max(s.earliest);
                if from < s.start {
                    edits.push(PillarEdit::In { clip: s.clip, by: s.start.saturating_sub(from) });
                    s.start = from;
                }
            }
        }
        let (ia, ib) = (at(&track, a), at(&track, b));
        let (sa, sb) = (ia.and_then(|i| track.get(i)).copied(), ib.and_then(|i| track.get(i)).copied());
        // both shown whole, their copies meeting at the cut inside the transition
        let meet = matches!((ia, ib, sa, sb), (Some(i), Some(k), Some(x), Some(y)) if k == i + 1 && x.end == y.start && j.start < x.end && y.start < j.end);
        let effect = match j.dip {
            Some(dip) => dip,
            None if meet => "cross_dissolve",
            None => continue,
        };
        let mut over = Vec::new();
        if meet {
            over.push(PillarEdit::Copies { join: j.id, effect, from: sa.map(|s| s.clip), to: sb.map(|s| s.clip), start: j.start, end: j.end });
        } else {
            // a copy on its own dips at its own edge, unless another copy touches it there (the
            // transition would join the two)
            if let (Some(i), Some(s)) = (ia, sa)
                && track.get(i + 1).is_none_or(|n| n.start > s.end)
                && s.end > j.start
            {
                over.push(PillarEdit::Copies { join: j.id, effect, from: Some(s.clip), to: None, start: j.start.max(s.start), end: s.end });
            }
            if let (Some(k), Some(s)) = (ib, sb)
                && k.checked_sub(1).and_then(|p| track.get(p)).is_none_or(|p| p.end < s.start)
                && s.start < j.end
            {
                over.push(PillarEdit::Copies { join: j.id, effect, from: None, to: Some(s.clip), start: s.start, end: j.end.min(s.end) });
            }
        }
        if over.is_empty() {
            continue;
        }
        if j.dip.is_some() {
            edits.push(PillarEdit::Fades { transition: j.id, from: j.from, to: j.to, start: j.start, end: j.end });
        }
        edits.extend(over);
    }
    edits
}

// ---------------------------------------------------------------------------------------------
// on a sequence (the reframe's Blur Fill)
// ---------------------------------------------------------------------------------------------

/// How far blurred copy `it` may reach (earliest start, latest end; ticks, on frames of `rate`)
/// with its media starting at `media.0` and ending at `media.1` (None: a still, without end): a
/// copy is lengthened at either edge only into media it has, played as it plays it (forwards or
/// reversed, at its speed). A frame hold shows one frame wherever it is.
pub(crate) fn reach(it: &TrackItem, media: (Tick, Option<Tick>), rate: FrameRate) -> (i64, i64) {
    if it.frame_hold.is_some() {
        return (i64::MIN, i64::MAX);
    }
    // (a NaN speed is the least, as the trims take it)
    let speed = it.speed.abs().max(1e-9);
    let head = (it.source_in - media.0).max(Tick::ZERO);
    let tail = media.1.map(|m| (m - it.source_out()).max(Tick::ZERO));
    // a reversed clip's In edge consumes the media after its source Out, its Out edge the media
    // before its source In
    let (before, after) = if it.reverse { (tail, Some(head)) } else { (Some(head), tail) };
    let timeline = |m: Tick| (m.0 as f64 / speed).floor() as i64;
    let earliest = before.map_or(i64::MIN, |m| {
        let t = Tick(it.start.0.saturating_sub(timeline(m)));
        let s = rate.snap(t);
        // on the first frame wholly inside the media
        if s < t { rate.tick_of(rate.frame_at(t).saturating_add(1)).0 } else { s.0 }
    });
    let latest = after.map_or(i64::MAX, |m| rate.snap(Tick(it.end().0.saturating_add(timeline(m)))).0);
    (earliest, latest)
}

/// The picture on the track above whose blurred copy `copy` is: a clip of the same media, played
/// the same way (speed, direction, frame hold), showing at a moment they share what the copy
/// shows; of several (a cut through one shot), the one it overlaps most.
pub(crate) fn owner_of(pictures: &[TrackItem], copy: &TrackItem) -> Option<ClipId> {
    let near = Tick(TICKS_PER_SECOND / 1000);
    pictures
        .iter()
        .filter(|p| p.item == copy.item && p.reverse == copy.reverse && p.frame_hold == copy.frame_hold && (p.speed - copy.speed).abs() < 1e-9)
        .filter_map(|p| {
            let (a, e) = (p.start.max(copy.start), p.end().min(copy.end()));
            (e > a && (p.source_time_at(a) - copy.source_time_at(a)).abs() <= near).then_some((p.id, e - a))
        })
        .max_by_key(|(_, overlap)| *overlap)
        .map(|(id, _)| id)
}

/// What [`clear_under`] left on the copies' track: the dips there that lost or changed a copy
/// (to be taken over by the new copies: [`through_transitions`]), and whose copy each copy left
/// is (copy, picture).
#[derive(Debug, Default)]
pub(crate) struct Cleared {
    pub moved: Vec<Join>,
    pub owners: Vec<(ClipId, ClipId)>,
}

/// Make room on copies' track `fills` for new copies under the clips `refilled` of `pictures`
/// (the track above): their old copies go (and any copy whose picture is unknown that is under
/// them), and the copies of other pictures that reached under them under a transition are
/// trimmed back off them (they reach again where they may: [`through_transitions`]). Cross
/// dissolves on those copies go with them; dips stay, to be taken over ([`Cleared::moved`]).
pub(crate) fn clear_under(fills: &mut Track, pictures: &[TrackItem], refilled: &[ClipId]) -> Cleared {
    let under: Vec<(Tick, Tick)> = pictures.iter().filter(|p| refilled.contains(&p.id)).map(|p| (p.start, p.end())).collect();
    let owners: Vec<(ClipId, Option<ClipId>)> = fills.items.iter().map(|c| (c.id, owner_of(pictures, c))).collect();
    let owner = |c: ClipId| owners.iter().find(|o| o.0 == c).and_then(|o| o.1);
    let overlaps = |a: Tick, e: Tick| under.iter().any(|(u0, u1)| a < *u1 && *u0 < e);
    let mut changed: HashSet<ClipId> = HashSet::new();
    let mut kept = Vec::with_capacity(fills.items.len());
    for mut c in std::mem::take(&mut fills.items) {
        if !overlaps(c.start, c.end()) {
            kept.push(c);
            continue;
        }
        changed.insert(c.id);
        match owner(c.id) {
            Some(p) if !refilled.contains(&p) => {
                // another picture's copy that reached under a refilled one: back off it
                for &(u0, u1) in &under {
                    if c.start < u0 && c.end() > u0 {
                        let d = u0 - c.end();
                        move_edge(&mut c, Edge::Out, d, true);
                    } else if c.start < u1 && c.end() > u1 {
                        let d = u1 - c.start;
                        move_edge(&mut c, Edge::In, d, true);
                    }
                }
                if c.duration > Tick::ZERO && !overlaps(c.start, c.end()) {
                    kept.push(c);
                }
            }
            _ => {}
        }
    }
    fills.items = kept;
    // the pictures either side of a moved dip: its copies' pictures, else the picture at the cut
    let at_cut = |p: Option<ClipId>, after: bool| -> Option<ClipId> {
        let p = pictures.iter().find(|x| Some(x.id) == p)?;
        pictures.iter().find(|x| if after { x.start == p.end() } else { x.end() == p.start }).map(|x| x.id)
    };
    let mut moved = Vec::new();
    fills.transitions.retain(|x| {
        let touches = [x.from, x.to].into_iter().flatten().any(|c| changed.contains(&c));
        if !touches {
            return true;
        }
        let Some(dip) = dip_of(&x.effect.effect) else { return false };
        let (mut from, mut to) = (x.from.and_then(owner), x.to.and_then(owner));
        if from.is_none() {
            from = at_cut(to, false);
        }
        if to.is_none() {
            to = at_cut(from, true);
        }
        moved.push(Join { id: x.id.0, start: x.start.0, end: x.end().0, from: from.map(|c| c.0), to: to.map(|c| c.0), dip: Some(dip) });
        true
    });
    let left: HashSet<ClipId> = fills.items.iter().map(|c| c.id).collect();
    let owners = owners.into_iter().filter_map(|(c, p)| Some((c, p?))).filter(|(c, _)| left.contains(c)).collect();
    Cleared { moved, owners }
}

/// Opacity keyframes on picture `it` at sequence times `keys`: the value given, or (None) the
/// one it has there now.
fn fade(it: &mut TrackItem, keys: [(Tick, Option<f64>); 2]) {
    let at = keys.map(|(t, v)| (it.effect_time_at(t), v));
    if it.effect("opacity").is_none()
        && let Some(def) = filmcraft_project::effect::find_effect("opacity")
    {
        it.effects.push(def.instance());
    }
    let Some(e) = it.effect_mut("opacity") else { return };
    e.enabled = true;
    let values = at.map(|(mt, v)| (mt, v.unwrap_or_else(|| e.f64_at("opacity", mt))));
    if e.param("opacity").is_none() {
        e.params.insert("opacity".into(), Param::new(ParamValue::Float(100.0)));
    }
    let Some(p) = e.param_mut("opacity") else { return };
    for (mt, v) in values {
        p.put_keyframe(mt, ParamValue::Float(v));
    }
}

/// Keep the blurred copies on video track `fills` of `q` under the pictures on track `pictures`
/// through the pictures' transitions ([`pillar_edits`]) and make the plan: `owners` says which
/// copy is under which picture (copy, picture); `moved`, the dips already on the copies' track
/// to take over ([`clear_under`]); `reach`, how far a copy's media lets it go ([`reach`]);
/// `alloc`, new transition ids.
pub(crate) fn through_transitions(
    q: &mut Sequence,
    (pictures, fills): (usize, usize),
    owners: &[(ClipId, ClipId)],
    moved: &[Join],
    reach: &dyn Fn(&TrackItem) -> (i64, i64),
    alloc: &mut dyn FnMut() -> u64,
) {
    let (Some(pt), Some(ft)) = (q.video_tracks.get(pictures), q.video_tracks.get(fills)) else { return };
    let spans: Vec<Span> = ft
        .items
        .iter()
        .map(|c| {
            let (earliest, latest) = reach(c);
            Span { clip: c.id.0, start: c.start.0, end: c.end().0, earliest, latest }
        })
        .collect();
    let backdrops: Vec<Backdrop> = owners
        .iter()
        .filter_map(|&(copy, picture)| {
            let (c, p) = (ft.item(copy)?, pt.item(picture)?);
            Some(Backdrop { picture: picture.0, copy: copy.0, head: c.start <= p.start, tail: c.end() >= p.end() })
        })
        .collect();
    let mut joins: Vec<Join> = pt
        .transitions
        .iter()
        .map(|x| Join { id: x.id.0, start: x.start.0, end: x.end().0, from: x.from.map(|c| c.0), to: x.to.map(|c| c.0), dip: dip_of(&x.effect.effect) })
        .chain(moved.iter().copied())
        .collect();
    joins.sort_by_key(|j| (j.start, j.end));
    // the pictures' dips taken down, and the joins whose transition is on the copies' track now
    let mut taken: Vec<Transition> = Vec::new();
    let mut placed: HashSet<u64> = HashSet::new();
    for edit in pillar_edits(&spans, &joins, &backdrops) {
        match edit {
            PillarEdit::Out { clip, by } => {
                if let Some(it) = q.video_tracks.get_mut(fills).and_then(|t| t.item_mut(ClipId(clip))) {
                    move_edge(it, Edge::Out, Tick(by), true);
                }
            }
            PillarEdit::In { clip, by } => {
                if let Some(it) = q.video_tracks.get_mut(fills).and_then(|t| t.item_mut(ClipId(clip))) {
                    move_edge(it, Edge::In, Tick(by.saturating_neg()), true);
                }
            }
            PillarEdit::Fades { transition, from, to, start, end } => {
                let Some(pt) = q.video_tracks.get_mut(pictures) else { continue };
                // (a dip an earlier run took down: the pictures fade over it already)
                let Some(i) = pt.transitions.iter().position(|x| x.id.0 == transition) else { continue };
                taken.push(pt.transitions.remove(i));
                let middle = Tick(start.saturating_add(end.saturating_sub(start) / 2));
                if let Some(a) = from.and_then(|c| pt.item_mut(ClipId(c))) {
                    fade(a, [(Tick(start), None), (middle, Some(0.0))]);
                }
                if let Some(b) = to.and_then(|c| pt.item_mut(ClipId(c))) {
                    fade(b, [(middle, Some(0.0)), (Tick(end), None)]);
                }
            }
            PillarEdit::Copies { join, effect, from, to, start, end } => {
                let Some(ft) = q.video_tracks.get_mut(fills) else { continue };
                let first = placed.insert(join);
                // the pictures' own dip (its settings and sound with the first copy it goes to),
                // a dip already down here, or a new dissolve
                let own = taken.iter().find(|x| x.id.0 == join).cloned();
                let here = ft.transitions.iter().position(|x| x.id.0 == join);
                let mut x = match (own, here) {
                    (Some(x), _) if first => x,
                    (None, Some(i)) if first => ft.transitions.remove(i),
                    (Some(x), _) => Transition { id: TransitionId(alloc()), sound: None, ..x },
                    (None, Some(i)) => match ft.transitions.get(i) {
                        Some(x) => Transition { id: TransitionId(alloc()), sound: None, ..x.clone() },
                        None => continue,
                    },
                    (None, None) => {
                        let Some(def) = find_transition(effect, EffectKind::VideoTransition) else { continue };
                        Transition {
                            id: TransitionId(alloc()),
                            effect: def.instance(),
                            start: Tick::ZERO,
                            duration: Tick::ZERO,
                            from: None,
                            to: None,
                            align: TransitionAlign::default(),
                            reverse: false,
                            sound: None,
                        }
                    }
                };
                (x.from, x.to, x.start, x.duration) = (from.map(ClipId), to.map(ClipId), Tick(start), Tick(end.saturating_sub(start)));
                x.align = match (from, to) {
                    (Some(_), None) => TransitionAlign::EndAtCut,
                    (None, Some(_)) => TransitionAlign::StartAtCut,
                    _ => TransitionAlign::CenterAtCut,
                };
                // the one an earlier run left at this place gives way
                ft.transitions.retain(|y| !(y.range().overlaps(&x.range()) && (y.from == x.from || y.to == x.to)));
                ft.transitions.push(x);
            }
        }
    }
    let Some(ft) = q.video_tracks.get_mut(fills) else { return };
    // a dip taken down before whose copies are gone goes (its pictures fade to nothing)
    ft.transitions.retain(|x| placed.contains(&x.id.0) || !moved.iter().any(|j| j.id == x.id.0));
    let ids: HashSet<ClipId> = ft.items.iter().map(|i| i.id).collect();
    ft.transitions.retain(|x| (x.from.is_some() || x.to.is_some()) && x.from.is_none_or(|c| ids.contains(&c)) && x.to.is_none_or(|c| ids.contains(&c)));
    ft.sort();
}

#[cfg(test)]
#[path = "pillars_tests.rs"]
mod tests;
