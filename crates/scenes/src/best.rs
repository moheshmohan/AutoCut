//! The best shots for a purpose: a ranked list from the picture's quality (sharpness, exposure,
//! steadiness, aesthetic score), what the describer said the shot is good for, and what the
//! purpose needs (a hook wants energy and a face or close framing, b-roll steady cutaways, a
//! reveal a moving camera on the subject…), picked so the list is not five takes of one moment.

use filmcraft_project::scenes::{CameraMove, Energy, Shot, ShotType, ShotUse};
use serde::Serialize;

/// One shot to choose from.
pub struct Entry<'a> {
    pub item: u64,
    /// Index of the shot in its item's index.
    pub shot: usize,
    pub data: &'a Shot,
}

/// A chosen shot.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Pick {
    pub item: u64,
    pub shot: usize,
    /// 0..1.
    pub score: f32,
    /// Why, in a few words each.
    pub why: Vec<String>,
}

/// Shots shorter than this are rarely usable.
const MIN_SECONDS: f64 = 0.8;
/// At most this many picks from one item before the others get a turn.
const PER_ITEM: usize = 2;
/// Two shots this alike (embedding cosine) are the same moment.
const SAME_MOMENT: f32 = 0.92;

/// How well `s` serves `purpose`, 0..1, and why.
pub fn score(s: &Shot, purpose: ShotUse) -> (f32, Vec<String>) {
    let mut why = Vec::new();
    let q = s.quality.score.clamp(0.0, 1.0);
    let mut v = 0.45 * q;
    if q >= 0.7 {
        why.push("sharp and well exposed".to_string());
    }
    let d = s.description.as_ref();
    match d {
        Some(d) if d.uses.contains(&purpose) => {
            v += 0.3;
            why.push(format!("described as {}", purpose.name()));
        }
        Some(_) => {}
        // nobody looked: neither for nor against
        None => v += 0.1,
    }
    let energy = d.map_or(Energy::Unknown, |d| d.energy);
    let ty = d.map_or(ShotType::Unknown, |d| d.shot_type);
    let faces = d.is_some_and(|d| d.faces_prominent);
    let people = d.map_or(0, |d| d.people);
    let has_subject = d.is_some_and(|d| !d.subjects.is_empty());
    let steady = s.quality.shake < 0.3;
    let moving = matches!(s.camera, CameraMove::PanLeft | CameraMove::PanRight | CameraMove::TiltUp | CameraMove::TiltDown | CameraMove::Moving);
    let secs = s.seconds();
    let mut add = |ok: bool, by: f32, what: &str| {
        if ok {
            v += by;
            why.push(what.to_string());
        }
    };
    match purpose {
        ShotUse::Hook => {
            add(energy == Energy::High, 0.1, "high energy");
            add(energy == Energy::Medium, 0.05, "lively");
            add(faces || matches!(ty, ShotType::CloseUp | ShotType::ExtremeCloseUp), 0.08, "close and personal");
            add((0.05..0.6).contains(&s.quality.motion), 0.07, "movement in frame");
        }
        ShotUse::BRoll => {
            add(!faces && people <= 2, 0.08, "no one talking to camera");
            add(steady, 0.07, "steady");
            add(matches!(ty, ShotType::Wide | ShotType::Insert | ShotType::Medium), 0.05, "a cutaway framing");
            add(secs >= 3.0, 0.05, "long enough to cut into");
        }
        ShotUse::Reveal => {
            add(moving, 0.1, "the camera moves");
            add(matches!(ty, ShotType::Wide | ShotType::Medium), 0.08, "shows the whole subject");
            add(has_subject, 0.07, "a clear subject");
        }
        ShotUse::TalkingHead => {
            add(faces, 0.15, "a face to camera");
            add((1..=2).contains(&people), 0.05, "one or two people");
            add(s.camera == CameraMove::Static, 0.05, "locked off");
        }
        ShotUse::Establishing => {
            add(ty == ShotType::Wide, 0.15, "wide");
            add(matches!(s.camera, CameraMove::Static | CameraMove::PanLeft | CameraMove::PanRight), 0.05, "static or a slow pan");
            add(people <= 3, 0.05, "the place, not the people");
        }
        ShotUse::Unknown => {}
    }
    if secs < MIN_SECONDS {
        v *= 0.5;
        why.push("very short".to_string());
    }
    (if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }, why)
}

/// The best `count` shots of `entries` for `purpose`, best first: each scored by [`score`], then
/// picked greedily with a penalty for a third pick from one item and for a shot that looks like
/// one already picked.
pub fn best_shots(entries: &[Entry<'_>], purpose: ShotUse, count: usize) -> Vec<Pick> {
    let mut scored: Vec<(usize, f32, Vec<String>)> =
        entries.iter().enumerate().map(|(i, e)| (i, score(e.data, purpose))).map(|(i, (s, w))| (i, s, w)).collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut picks: Vec<Pick> = Vec::new();
    let mut taken: Vec<usize> = Vec::new();
    while picks.len() < count && taken.len() < scored.len() {
        let mut best: Option<(usize, f32)> = None;
        for (k, (i, s, _)) in scored.iter().enumerate() {
            if taken.contains(&k) {
                continue;
            }
            let Some(e) = entries.get(*i) else { continue };
            let mut adj = *s;
            if picks.iter().filter(|p| p.item == e.item).count() >= PER_ITEM {
                adj *= 0.7;
            }
            let alike = taken
                .iter()
                .filter_map(|t| scored.get(*t))
                .filter_map(|(j, _, _)| entries.get(*j))
                .any(|o| !o.data.embedding.is_empty() && o.data.embedding.cosine(&e.data.embedding.0) > SAME_MOMENT);
            if alike {
                adj *= 0.6;
            }
            if best.is_none_or(|(_, b)| adj > b) {
                best = Some((k, adj));
            }
        }
        let Some((k, adj)) = best else { break };
        taken.push(k);
        if let Some((i, _, why)) = scored.get(k)
            && let Some(e) = entries.get(*i)
        {
            picks.push(Pick { item: e.item, shot: e.shot, score: adj, why: why.clone() });
        }
    }
    picks
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_project::scenes::{Embedding, ShotDescription, ShotQuality, Subject};
    use filmcraft_time::Tick;

    fn shot(secs: f64, quality: f32, d: Option<ShotDescription>, camera: CameraMove) -> Shot {
        Shot {
            start: Tick::ZERO,
            end: Tick::from_seconds_f64(secs),
            quality: ShotQuality { score: quality, motion: 0.2, ..Default::default() },
            camera,
            description: d,
            ..Default::default()
        }
    }

    #[test]
    fn purposes_pick_different_shots() {
        let talking =
            ShotDescription { shot_type: ShotType::CloseUp, faces_prominent: true, people: 1, uses: vec![ShotUse::TalkingHead], ..Default::default() };
        let wide = ShotDescription { shot_type: ShotType::Wide, uses: vec![ShotUse::Establishing, ShotUse::BRoll], ..Default::default() };
        let action = ShotDescription {
            shot_type: ShotType::Medium,
            energy: Energy::High,
            subjects: vec![Subject { name: "robot".into(), bbox: None }],
            uses: vec![ShotUse::Hook, ShotUse::Reveal],
            ..Default::default()
        };
        let shots = [
            shot(6.0, 0.8, Some(talking), CameraMove::Static),
            shot(8.0, 0.8, Some(wide), CameraMove::Static),
            shot(4.0, 0.8, Some(action), CameraMove::PanLeft),
        ];
        let e: Vec<Entry> = shots.iter().enumerate().map(|(i, s)| Entry { item: i as u64, shot: 0, data: s }).collect();
        assert_eq!(best_shots(&e, ShotUse::Hook, 1)[0].item, 2);
        assert_eq!(best_shots(&e, ShotUse::Reveal, 1)[0].item, 2);
        assert_eq!(best_shots(&e, ShotUse::BRoll, 1)[0].item, 1);
        assert_eq!(best_shots(&e, ShotUse::Establishing, 1)[0].item, 1);
        assert_eq!(best_shots(&e, ShotUse::TalkingHead, 1)[0].item, 0);
        let all = best_shots(&e, ShotUse::Hook, 10);
        assert_eq!(all.len(), 3);
        assert!(all.windows(2).all(|w| w[0].score >= w[1].score));
        assert!(all[0].why.iter().any(|w| w.contains("hook")));
        assert!(best_shots(&e, ShotUse::Hook, 0).is_empty());
        assert!(best_shots(&[], ShotUse::Hook, 3).is_empty());
    }

    #[test]
    fn quality_and_length_count() {
        let shots = [shot(5.0, 0.2, None, CameraMove::Static), shot(5.0, 0.9, None, CameraMove::Static), shot(0.3, 1.0, None, CameraMove::Static)];
        let e: Vec<Entry> = shots.iter().enumerate().map(|(i, s)| Entry { item: 1, shot: i, data: s }).collect();
        let picks = best_shots(&e, ShotUse::BRoll, 3);
        assert_eq!(picks[0].shot, 1);
        assert_eq!(picks.last().map(|p| p.shot), Some(2), "a very short shot comes last");
    }

    #[test]
    fn picks_are_diverse() {
        // three near-identical takes of one moment in item 1, and a weaker shot in item 2
        let mut shots: Vec<Shot> = (0..3).map(|_| shot(5.0, 0.9, None, CameraMove::Static)).collect();
        for s in &mut shots {
            s.embedding = Embedding(vec![1.0, 0.0]);
        }
        let mut other = shot(5.0, 0.7, None, CameraMove::Static);
        other.embedding = Embedding(vec![0.0, 1.0]);
        shots.push(other);
        let e: Vec<Entry> = shots.iter().enumerate().map(|(i, s)| Entry { item: if i == 3 { 2 } else { 1 }, shot: i, data: s }).collect();
        let picks = best_shots(&e, ShotUse::BRoll, 2);
        assert_eq!(picks.iter().map(|p| p.item).collect::<Vec<_>>(), vec![1, 2], "the second pick is not another take of the first");
    }

    #[test]
    fn hostile_scores_stay_in_range() {
        let mut s = shot(5.0, f32::NAN, None, CameraMove::Unknown);
        s.quality.motion = f32::INFINITY;
        let (v, _) = score(&s, ShotUse::Hook);
        assert!((0.0..=1.0).contains(&v));
        let (v, _) = score(&shot(5.0, 9.0, None, CameraMove::Static), ShotUse::Unknown);
        assert!((0.0..=1.0).contains(&v));
    }
}
