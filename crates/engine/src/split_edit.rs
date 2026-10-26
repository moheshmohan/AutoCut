//! Split edits (`clip.splitEdit`): a **J cut** — a clip's sound starts `lead` seconds before its
//! picture, under the end of the shot before (someone heard before they are seen) — and an
//! **L cut** — its sound runs `tail` seconds on under the start of the shot after (a sentence
//! finished over the next picture). The neighbouring sound on the same track gives way, its edge
//! trimmed back; picture and sound stay linked, so moving or trimming them later keeps them in
//! sync, and the edit is an ordinary one to change by hand. Only the media there is is used (a
//! clip that starts at its media's first frame cannot lead). One undo step.

use filmcraft_edit::{Edge, TrimMode};
use filmcraft_project::{ClipId, TrackKind};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, clip_p, f64_p, has_seq};
use crate::{Result, Session};

pub const SPLIT_CMD: &str = "clip.splitEdit";
/// Longest lead or tail, seconds.
pub const MAX_SPLIT: f64 = 5.0;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: SPLIT_CMD,
        label: "Split Edit",
        menu: &["Clip"],
        shortcut: None,
        params: r#"{"clip":id (a picture clip with linked sound),"lead":0..5 (seconds: J cut)?,"tail":0..5 (seconds: L cut)?}"#,
        enabled: has_seq,
        run: split,
        journal: true,
    }]
}

fn seconds(p: &Value, k: &str) -> Result<f64> {
    match f64_p(p, k) {
        None => Ok(0.0),
        Some(v) if v.is_finite() && (0.0..=MAX_SPLIT).contains(&v) => Ok(v),
        Some(_) => Err(bad(SPLIT_CMD, format!("`{k}` is 0 to {MAX_SPLIT} seconds"))),
    }
}

fn split(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = SPLIT_CMD;
    let clip = clip_p(p, "clip").ok_or_else(|| bad(cmd, "need `clip`, a picture clip with linked sound"))?;
    let (lead, tail) = (seconds(p, "lead")?, seconds(p, "tail")?);
    if lead <= 0.0 && tail <= 0.0 {
        return Err(bad(cmd, "give a `lead` (J cut) or a `tail` (L cut)"));
    }
    let seq = s.active_sequence().ok_or(crate::EngineError::NoSequence)?;
    let (track, it) = seq.find_item(clip).ok_or_else(|| bad(cmd, format!("no clip {} in the active sequence", clip.0)))?;
    if seq.track(track).is_none_or(|t| t.kind != TrackKind::Video) {
        return Err(bad(cmd, "`clip` is a picture clip (its linked sound is what moves)"));
    }
    let link = it.link.ok_or_else(|| bad(cmd, "the clip has no linked sound"))?;
    let sounds: Vec<ClipId> = seq.audio_tracks.iter().flat_map(|t| &t.items).filter(|i| i.link == Some(link) && i.id != clip).map(|i| i.id).collect();
    if sounds.is_empty() {
        return Err(bad(cmd, "the clip has no linked sound"));
    }
    let (lead_t, tail_t) = (Tick::from_seconds_f64(lead), Tick::from_seconds_f64(tail));
    let done = s.edit_sequence("Split Edit", |q, ctx, _| {
        let mut out = Vec::new();
        for a in &sounds {
            let mut got = (Tick::ZERO, Tick::ZERO);
            if lead_t > Tick::ZERO {
                // the sound before on the track gives way to the lead
                let Some((tid, me)) = q.find_item(*a) else { continue };
                let start = me.start;
                let before = q.track(tid).and_then(|t| {
                    t.items
                        .iter()
                        .filter(|i| i.id != *a && i.end() <= start && i.end() > start - lead_t)
                        .max_by_key(|i| i.end())
                        .map(|i| (i.id, i.end() - (start - lead_t), i.duration))
                });
                if let Some((b, over, len)) = before {
                    // (never trimmed away entirely: a frame of it stays)
                    let back = over.min(len - ctx_frame(q));
                    if back > Tick::ZERO {
                        filmcraft_edit::trim(q, b, Edge::Out, TrimMode::Regular, -back, ctx)?;
                    }
                }
                let d = filmcraft_edit::trim(q, *a, Edge::In, TrimMode::Regular, -lead_t, ctx)?;
                got.0 = -d;
            }
            if tail_t > Tick::ZERO {
                // the sound after gives way to the tail
                let Some((tid, me)) = q.find_item(*a) else { continue };
                let end = me.end();
                let after = q.track(tid).and_then(|t| {
                    t.items
                        .iter()
                        .filter(|i| i.id != *a && i.start >= end && i.start < end + tail_t)
                        .min_by_key(|i| i.start)
                        .map(|i| (i.id, end + tail_t - i.start, i.duration))
                });
                if let Some((n, over, len)) = after {
                    let fwd = over.min(len - ctx_frame(q));
                    if fwd > Tick::ZERO {
                        filmcraft_edit::trim(q, n, Edge::In, TrimMode::Regular, fwd, ctx)?;
                    }
                }
                let d = filmcraft_edit::trim(q, *a, Edge::Out, TrimMode::Regular, tail_t, ctx)?;
                got.1 = d;
            }
            out.push(json!({"clip": a.0, "lead": round3(got.0.seconds()), "tail": round3(got.1.seconds())}));
        }
        Ok(out)
    })?;
    Ok(json!({"clip": clip.0, "sounds": done}))
}

/// A frame of the sequence (what a neighbour keeps at least).
fn ctx_frame(q: &filmcraft_project::Sequence) -> Tick {
    q.settings.frame_rate.sane().frame_duration().max(Tick(1))
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_project::ItemId;

    /// A sequence of two clips of the demo's first media, back to back from 10 s with linked
    /// sound, each from 2 s into its media (room to lead and to run on): their picture ids.
    fn two_shots() -> (Session, ClipId, ClipId) {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let item: ItemId = s.active_sequence().unwrap().video_tracks[0].items[0].item;
        s.execute("file.newSequence", json!({"name": "Split", "width": 640, "height": 360, "fps": 30})).unwrap();
        let sec = |x: f64| Tick::from_seconds_f64(x).0;
        let a = s
            .execute(
                "timeline.place",
                json!({"item": item.0, "track": "V1", "audioTrack": "A1", "time": sec(10.0), "sourceIn": sec(2.0), "duration": sec(2.0)}),
            )
            .unwrap();
        let b = s
            .execute(
                "timeline.place",
                json!({"item": item.0, "track": "V1", "audioTrack": "A1", "time": sec(12.0), "sourceIn": sec(2.0), "duration": sec(2.0)}),
            )
            .unwrap();
        let first = |r: &Value| ClipId(r["clips"][0].as_u64().unwrap());
        (s, first(&a), first(&b))
    }

    fn range(s: &Session, c: ClipId) -> (f64, f64) {
        let (_, i) = s.active_sequence().unwrap().find_item(c).unwrap();
        ((i.start.seconds() * 100.0).round() / 100.0, (i.end().seconds() * 100.0).round() / 100.0)
    }

    fn sound(s: &Session, picture: ClipId) -> ClipId {
        let q = s.active_sequence().unwrap();
        let link = q.find_item(picture).unwrap().1.link.unwrap();
        q.audio_tracks.iter().flat_map(|t| &t.items).find(|i| i.link == Some(link)).unwrap().id
    }

    #[test]
    fn a_j_cut_leads_with_the_sound_and_an_l_cut_runs_it_on() {
        let (mut s, a, b) = two_shots();
        let r = s.execute(SPLIT_CMD, json!({"clip": b.0, "lead": 0.5})).unwrap();
        assert_eq!(r["sounds"][0]["lead"], 0.5, "{r}");
        // the pictures cut where they did; the sound before gave way
        assert_eq!((range(&s, a), range(&s, b)), ((10.0, 12.0), (12.0, 14.0)));
        assert_eq!((range(&s, sound(&s, a)), range(&s, sound(&s, b))), ((10.0, 11.5), (11.5, 14.0)));
        s.undo().unwrap();
        assert_eq!(range(&s, sound(&s, b)), (12.0, 14.0));
        // an L cut: the first shot's sound runs on under the second
        let r = s.execute(SPLIT_CMD, json!({"clip": a.0, "tail": 0.4})).unwrap();
        assert_eq!(r["sounds"][0]["tail"], 0.4, "{r}");
        assert_eq!((range(&s, sound(&s, a)), range(&s, sound(&s, b))), ((10.0, 12.4), (12.4, 14.0)));
        // still linked: picture and sound move together
        let q = s.active_sequence().unwrap();
        assert_eq!(q.find_item(a).unwrap().1.link, q.find_item(sound(&s, a)).unwrap().1.link);
    }

    #[test]
    fn split_edits_refuse_what_they_cannot_do() {
        let (mut s, a, _) = two_shots();
        for bad in [
            json!({}),
            json!({"clip": a.0}),
            json!({"clip": a.0, "lead": -1}),
            json!({"clip": a.0, "lead": 9}),
            json!({"clip": a.0, "tail": f64::NAN}),
            json!({"clip": 999_999, "lead": 0.5}),
        ] {
            assert!(s.execute(SPLIT_CMD, bad.clone()).is_err(), "{bad}");
        }
        let snd = sound(&s, a);
        assert!(s.execute(SPLIT_CMD, json!({"clip": snd.0, "lead": 0.5})).is_err(), "a sound clip is not a picture");
        // a lead longer than the media before it: only what there is (2 s)
        let r = s.execute(SPLIT_CMD, json!({"clip": a.0, "lead": 3.0})).unwrap();
        assert_eq!(r["sounds"][0]["lead"], 2.0, "{r}");
    }
}
