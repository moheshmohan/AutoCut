//! Two angles of one moment: where two cameras filmed at the same time (`media.sameMoments`: a
//! phone and a 360° camera, lined up by their sound), a shot is cut half-way to the other camera's
//! picture of the same instants, so the action carries on across the cut.

use serde_json::Value;

use super::{Camera, Candidate, Enter, Kind, Plan, PlannedShot};

/// One pair of takes of the same moment: `b`'s media time `t` is `a`'s `t + offset` (seconds).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pair {
    pub a: u64,
    pub b: u64,
    pub offset: f64,
}

/// Most two-angle cuts in one edit.
pub const MAX_SPLITS: usize = 2;
/// The fewest beats a shot needs to be cut in two.
const MIN_BEATS: u32 = 4;

/// The pairs of a `media.sameMoments` answer.
pub fn pairs(v: &Value) -> Vec<Pair> {
    v["pairs"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .filter_map(|p| Some(Pair { a: p["a"].as_u64()?, b: p["b"].as_u64()?, offset: p["offset"].as_f64().filter(|o| o.is_finite())? }))
        .collect()
}

/// The other camera's item and media time for item `item` at media time `t`, if it filmed then.
fn other(pairs: &[Pair], item: u64, t: f64) -> Option<(u64, f64)> {
    pairs.iter().find_map(|p| {
        if p.a == item {
            Some((p.b, t - p.offset))
        } else if p.b == item {
            Some((p.a, t + p.offset))
        } else {
            None
        }
    })
}

/// Cut up to [`MAX_SPLITS`] shots half-way to the other camera's picture of the same instants:
/// the best-ranked moments another camera filmed, held or pushing in (a move needs its whole
/// length), not the first or last shot, the other camera's moment covering the second half. The
/// changes made, for the report.
pub fn split(plan: &mut Plan, cands: &[Candidate], pairs: &[Pair], beat: f64) -> Vec<String> {
    if pairs.is_empty() || !(beat.is_finite() && beat > 0.0) {
        return Vec::new();
    }
    let n = plan.shots.len();
    // (shot index, its candidate's rank) of the shots that can be cut in two, best first
    let mut order: Vec<(usize, f64)> = plan
        .shots
        .iter()
        .enumerate()
        .filter(|(k, s)| *k > 0 && *k + 1 < n && s.beats >= MIN_BEATS && matches!(s.camera, Camera::Hold | Camera::PushIn | Camera::Drift))
        .filter_map(|(k, s)| cands.iter().find(|c| c.id == s.clip).map(|c| (k, c.rank)))
        .collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut done: Vec<(usize, PlannedShot, String)> = Vec::new();
    for (k, _) in order {
        if done.len() >= MAX_SPLITS {
            break;
        }
        let Some(s) = plan.shots.get(k) else { continue };
        let Some(c) = cands.iter().find(|c| c.id == s.clip) else { continue };
        let first = s.beats / 2;
        let second = s.beats - first;
        let mid = s.from + f64::from(first) * beat;
        let Some((item, t)) = other(pairs, c.item, mid) else { continue };
        let len = f64::from(second) * beat;
        // the other camera's moment that holds the second half
        let Some(o) = cands.iter().filter(|o| o.item == item && o.start <= t + 1e-6 && t + len <= o.end + 1e-6).max_by(|a, b| a.rank.total_cmp(&b.rank)) else {
            continue;
        };
        let camera = Camera::Hold;
        if f64::from(first) * beat < s.camera.min_seconds() || len < camera.min_seconds() {
            continue;
        }
        let shot = PlannedShot {
            clip: o.id.clone(),
            file: String::new(),
            grade: None,
            from: t,
            beats: second,
            camera,
            look: if o.kind == Kind::Vr { o.looks.first().map(|l| l.id.clone()) } else { None },
            to: None,
            enter: Enter::Cut,
            align: None,
            lead: 0.0,
            tail: 0.0,
            cut: "two angles of one moment".into(),
            text: None,
            why: format!("the same moment from the other camera ({})", o.caption),
        };
        let change = format!("shot {} ({}) cut half-way to the other camera's {} at the same moment", k + 1, c.id, o.id);
        done.push((k, shot, change));
    }
    // (from the last: the indices before stay right)
    done.sort_by_key(|d| std::cmp::Reverse(d.0));
    let mut changes = Vec::new();
    for (k, shot, change) in done {
        if let Some(s) = plan.shots.get_mut(k) {
            s.beats -= shot.beats;
            plan.shots.insert(k + 1, shot);
            changes.push(change);
        }
    }
    changes.reverse();
    changes
}
