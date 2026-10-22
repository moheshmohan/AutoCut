//! The privacy check of a finished edit: faces the picture still shows.
//!
//! | command | does |
//! |---|---|
//! | `sequence.checkPrivacy` | renders the active sequence at each of `seconds` (as it exports: effects, masks, titles), runs the face detector on each frame, and says which faces no redaction mask hides: each followed across the frames it is seen in, with when, where (shares of the frame), the clip under it and where in that clip, whether it is on the subject (inside one of the clip's `exclude` boxes, at any moment), and — with `verify` and the host's vision model — whether it is a person |
//!
//! It checks what [`crate::redact`] made, on the picture as a viewer sees it: a face the
//! redaction never found, a mask that does not cover the face it was made for, a face left sharp
//! for the robot's (the bug of 2026-10-21: visitors next to Ameca were left sharp because they
//! stood inside the robot's box). A face the frame shows is *masked* when the top-most media clip
//! there has an automatic redaction mask (`Redact …`) over that point at that moment.
//!
//! A face may stay sharp (`allowed`) only when the vision model says it is not a person, twice
//! (the grid of crops, then alone and outlined: [`crate::redact::VisionJudge::confirm`]), and it
//! is on the subject or the detector is not sure of it (score under `sureFace`) — the same rule
//! `clip.autoRedact` hides by. A face on the subject the grid calls a person is looked at alone
//! too, and is a person only if that agrees (`verdict` `unsure` otherwise: the robot's own face
//! is sometimes called a person in a grid, [`both_looks`]). Without a vision model nothing
//! unmasked is allowed. AutoCut's QA
//! (`crates/autocut/src/qa.rs`) samples a showcase every half second and at every shot's start
//! with this, and stops its upload when a face is shown that should not be.
//!
//! **Faces in profile.** With `profiles` (on by default once the person and pose models are
//! downloaded) the check also runs the redaction's profile pass: the people in the frame, their
//! poses, and the head of each one whose face shows (nose or an eye) that the face detector
//! missed is a face too (`fromPose` in the answer), under the same rules and the same vision
//! check. They have their own thresholds, so `faceScore` does not apply to them. (2026-10-22: two
//! visitors in profile behind a robot arm stayed sharp in a reel, and this check, with the face
//! detector alone, did not see them either.)

use filmcraft_geom::Vec2;
use filmcraft_project::{ClipId, ItemId, ItemKind, Sequence, TrackItem};
use filmcraft_time::Tick;
use filmcraft_vision::track::{Exclude, Frame, LinkParams, Track};
use filmcraft_vision::{Detection, Kind};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, has_seq};
use crate::redact::{Crop, DetectFn, JUDGE_BATCH, VisionJudge};
use crate::{EngineError, Result, Session};

const CMD: &str = "sequence.checkPrivacy";
/// Most frames one call looks at.
pub const MAX_TIMES: usize = 2000;
/// Most faces a frame reports, and unmasked faces asked about in one call.
const MAX_FACES: usize = 64;
const MAX_ASKED: usize = 600;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Check for Unblurred Faces",
        menu: &[],
        shortcut: None,
        params: r#"{"seconds":[f64] (sequence seconds, 1..2000),"maxSide":160..3840=1280,"faceScore":0.05..1=0.6,"minFaceHeight":0..1=0.05,"sureFace":0.05..1=0.85,"exclude":{"clip":id,"x":0..1,"y":0..1,"w":0..1,"h":0..1}|[box]?,"verify":bool=true,"profiles":bool=installed}"#,
        enabled: has_seq,
        run: check,
        journal: false,
    }]
}

/// Where a face seen in the sequence's frame is in the clip under it.
#[derive(Clone, Debug, PartialEq)]
struct Under {
    clip: ClipId,
    item: ItemId,
    name: String,
    /// Media time, seconds.
    media: f64,
    /// The face's centre in the clip's picture (shares).
    at: [f64; 2],
    /// An automatic redaction mask of the clip covers it.
    masked: bool,
}

/// One face seen in one frame.
#[derive(Clone, Debug)]
struct Seen {
    t: f64,
    det: Detection,
    under: Option<Under>,
    subject: bool,
}

/// The parameters' times: finite, not negative, in order, without repeats.
fn times_p(p: &Value) -> Result<Vec<f64>> {
    let list = p.get("seconds").and_then(Value::as_array).ok_or_else(|| bad(CMD, "`seconds` is a list of sequence times (seconds)"))?;
    if list.is_empty() || list.len() > MAX_TIMES {
        return Err(bad(CMD, format!("`seconds` lists 1 to {MAX_TIMES} times")));
    }
    let mut out: Vec<f64> = list
        .iter()
        .map(|v| v.as_f64().filter(|t| t.is_finite() && (0.0..=1e7).contains(t)))
        .collect::<Option<_>>()
        .ok_or_else(|| bad(CMD, "each time is seconds from 0"))?;
    out.sort_by(f64::total_cmp);
    out.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
    Ok(out)
}

/// The face detector: the host's own, else the downloaded one (loaded once for each `profiles`
/// setting: with the profile pass, the heads in profile the face detector misses are faces too).
fn detector(s: &mut Session, profiles: Option<bool>) -> Result<DetectFn> {
    face_detector(s, CMD, profiles)
}

/// The face detector for command `cmd` (its errors name it): the host's own, else the
/// downloaded one (loaded once for each `profiles` setting, shared with `sequence.checkPrivacy`).
pub(crate) fn face_detector(s: &mut Session, cmd: &str, profiles: Option<bool>) -> Result<DetectFn> {
    if let Some(d) = &s.redacts.detector {
        return Ok(d.clone());
    }
    // a low floor: the score asked for is applied here, so one detector serves every call
    let o = crate::redact::Options { faces: true, plates: false, face_score: 0.3, profiles, ..crate::redact::Options::default() };
    let on = crate::redact::profiles_on(&o);
    if let Some((_, d)) = s.redacts.faces_only.iter().find(|(p, _)| *p == on) {
        return Ok(d.clone());
    }
    crate::redact::ready(cmd)?;
    let d = crate::redact::load_detectors(&crate::redact::Options { profiles: Some(on), ..o }).map_err(|e| bad(cmd, e))?;
    s.redacts.faces_only.push((on, d.clone()));
    Ok(d)
}

/// Whether point `p` (clip pixels) is inside mask path `poly` (flattened), by the even-odd rule.
fn inside(poly: &[Vec2], p: Vec2) -> bool {
    let mut odd = false;
    let n = poly.len();
    for i in 0..n {
        let (Some(a), Some(b)) = (poly.get(i), poly.get((i + n - 1) % n)) else { continue };
        if (a.y > p.y) != (b.y > p.y) {
            let x = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if p.x < x {
                odd = !odd;
            }
        }
    }
    odd
}

/// The top-most media clip of `q` at `t` whose picture is at sequence pixel `c`, and where. In a
/// transition both its clips are drawn: a face hidden in either (the outgoing clip's visitor
/// seen through a whip into the next shot) counts as masked.
fn under(s: &Session, q: &Sequence, t: Tick, c: Vec2) -> Option<Under> {
    let mut found: Option<Under> = None;
    for track in q.video_tracks.iter().rev().filter(|tr| tr.enabled) {
        let mut ids: Vec<ClipId> = track.items.iter().filter(|it| it.enabled && it.start <= t && t < it.end()).map(|it| it.id).collect();
        for tr in track.transitions.iter().filter(|tr| tr.start <= t && t < tr.end()) {
            let both: Vec<ClipId> = tr.from.into_iter().chain(tr.to).filter(|id| !ids.contains(id)).collect();
            ids.extend(both);
        }
        for id in ids {
            let Some(it) = track.items.iter().find(|it| it.id == id && it.enabled) else { continue };
            if !is_footage(s, it) {
                continue;
            }
            let Some(size) = filmcraft_render::source_size(&s.project, it.item).filter(|(w, h)| *w > 0 && *h > 0) else { continue };
            let mt = it.effect_time_at(t);
            let m = filmcraft_render::motion_matrix(q, it, size, filmcraft_render::source_par(&s.project, it.item), mt);
            let Some(inv) = m.inverse() else { continue };
            let sp = inv.apply(c);
            let (w, h) = (f64::from(size.0), f64::from(size.1));
            if !(sp.x.is_finite() && sp.y.is_finite() && (0.0..w).contains(&sp.x) && (0.0..h).contains(&sp.y)) {
                continue;
            }
            let masked = it
                .effects
                .iter()
                .filter(|e| e.enabled && crate::redact::is_auto_redaction(e))
                .any(|e| e.masks.iter().any(|mk| mk.opacity.f64_at(mt) > 50.0 && inside(&mk.path_at(mt).flatten(1.0), sp) != mk.inverted));
            let u = Under { clip: it.id, item: it.item, name: it.name.clone(), media: mt.seconds(), at: [sp.x / w, sp.y / h], masked };
            if masked {
                return Some(u);
            }
            found.get_or_insert(u);
        }
        if found.is_some() {
            return found;
        }
    }
    found
}

/// A clip of video footage (not a title, a colour matte, a nested sequence).
fn is_footage(s: &Session, it: &TrackItem) -> bool {
    match s.project.item(it.item).map(|i| &i.kind) {
        Some(ItemKind::Media(m)) => m.info.has_video(),
        Some(ItemKind::Subclip { .. }) => s.project.resolve_media(it.item).is_some_and(|(_, m, _)| m.info.has_video()),
        _ => false,
    }
}

/// The sequence's frame at `t` (seconds) at `scale` as it exports (captions are not burned in):
/// RGBA over black, width, height.
fn frame(s: &Session, seq: ItemId, scale: f32, t: f64) -> Option<(Vec<u8>, usize, usize)> {
    let q = s.project.sequence(seq)?;
    let t = q.settings.frame_rate.sane().snap(Tick::from_seconds_f64(t).max(Tick::ZERO));
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    let opts = filmcraft_render::RenderOptions { scale, captions: false, ..Default::default() };
    let img = filmcraft_render::render_sequence(&s.project, seq, t, opts, &provider);
    (img.w > 0 && img.h > 0).then(|| (img.over_black_rgba8(), img.w, img.h))
}

/// An unmasked face followed across the frames it is seen in, and what the check made of it.
struct Unmasked {
    track: Track,
    best: Seen,
    subject: bool,
    person: Option<bool>,
    checked: bool,
}

fn check(s: &mut Session, p: &Value) -> Result<Value> {
    if !p.is_object() {
        return Err(bad(CMD, "parameters must be an object"));
    }
    let seq = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let times = times_p(p)?;
    let max_side = crate::redact::unit_p(CMD, p, "maxSide", 160.0, 3840.0, 1280.0)?;
    let score = crate::redact::unit_p(CMD, p, "faceScore", 0.05, 1.0, 0.6)? as f32;
    let min_h = crate::redact::unit_p(CMD, p, "minFaceHeight", 0.0, 1.0, 0.05)?;
    let sure = crate::redact::unit_p(CMD, p, "sureFace", 0.05, 1.0, f64::from(crate::redact::SURE_FACE))? as f32;
    let verify = match p.get("verify").filter(|v| !v.is_null()) {
        None => true,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(bad(CMD, "`verify` is true or false")),
    };
    let exclude = crate::redact::exclude_p(CMD, p)?;
    let profiles = crate::redact::profiles_p(CMD, p)?;
    let det = detector(s, profiles)?;
    // the profile pass is heavy GPU work: the machine's GPU lock when it is free (the check runs
    // anyway when another user has it; it does not wait)
    let heavy = s.redacts.detector.is_none() && crate::redact::profiles_on(&crate::redact::Options { profiles, ..crate::redact::Options::default() });
    let _gpu = heavy.then(|| filmcraft_locks::try_acquire(filmcraft_locks::Resource::Gpu, "privacy check (people and poses)").ok()).flatten();
    let q = s.project.sequence(seq).ok_or(EngineError::NoSequence)?.clone();
    let (sw, sh) = (f64::from(q.settings.width.max(1)), f64::from(q.settings.height.max(1)));
    let scale = (max_side / sw.max(sh)).min(1.0) as f32;
    let end = q.duration().seconds();
    let mut seen: Vec<Seen> = Vec::new();
    let (mut rendered, mut missed, mut masked, mut small) = (0usize, 0usize, 0usize, 0usize);
    for &t in times.iter().filter(|t| **t < end) {
        let Some((rgba, w, h)) = frame(s, seq, scale, t) else {
            missed += 1;
            continue;
        };
        rendered += 1;
        let faces: Vec<Detection> = det(&rgba, w, h, w.max(h))
            .into_iter()
            // a head found from a pose has passed its own thresholds
            .filter(|d| d.kind == Kind::Face && (d.pose || d.score >= score) && d.bbox.iter().all(|v| v.is_finite()))
            .take(MAX_FACES)
            .collect();
        for d in faces {
            if f64::from(d.bbox[3]) < min_h {
                small += 1;
                continue;
            }
            let c = Vec2::new(f64::from(d.bbox[0] + d.bbox[2] / 2.0) * sw, f64::from(d.bbox[1] + d.bbox[3] / 2.0) * sh);
            let u = under(s, &q, Tick::from_seconds_f64(t), c);
            if u.as_ref().is_some_and(|u| u.masked) {
                masked += 1;
                continue;
            }
            let subject = u
                .as_ref()
                .is_some_and(|u| exclude.iter().filter(|e| e.clip.is_none_or(|c| c == u.clip)).any(|e| Exclude { t: None, ..e.ex }.contains(u.at[0], u.at[1])));
            seen.push(Seen { t, det: d, under: u, subject });
        }
    }
    // each unmasked face followed across the frames, clip by clip
    let mut clips: Vec<Option<ClipId>> = seen.iter().map(|x| x.under.as_ref().map(|u| u.clip)).collect();
    clips.sort_by_key(|c| c.map(|c| c.0));
    clips.dedup();
    let step = times.windows(2).filter_map(|w| w.get(1).zip(w.first()).map(|(b, a)| b - a)).fold(f64::INFINITY, f64::min);
    let link = LinkParams {
        max_gap: if step.is_finite() { (step * 2.5).clamp(0.2, 5.0) } else { 1.0 },
        min_iou: 0.1,
        max_jump: 1.5,
        smooth: 0.0,
        // the faces as seen: boxes compared as they are
        grow: 0.0,
        ..LinkParams::default()
    };
    let mut found: Vec<Unmasked> = Vec::new();
    for c in clips {
        let mine: Vec<&Seen> = seen.iter().filter(|x| x.under.as_ref().map(|u| u.clip) == c).collect();
        let mut frames: Vec<Frame> = Vec::new();
        for x in &mine {
            match frames.last_mut() {
                Some(f) if (f.t - x.t).abs() < 1e-9 => f.detections.push(x.det.clone()),
                _ => frames.push(Frame { t: x.t, detections: vec![x.det.clone()] }),
            }
        }
        for track in filmcraft_vision::track::link(&frames, &link) {
            let Some(bp) = track.points.iter().filter(|p| p.seen).max_by(|a, b| a.bbox[3].total_cmp(&b.bbox[3])) else { continue };
            // the sighting at the best moment, and whether any sighting was on the subject
            let near = |x: &&&Seen| (x.t - bp.t).abs() < 1e-9;
            let at_best = mine.iter().filter(near).min_by(|a, b| {
                let d = |x: &Seen| (f64::from(x.det.bbox[0]) - bp.bbox[0]).abs() + (f64::from(x.det.bbox[1]) - bp.bbox[1]).abs();
                d(a).total_cmp(&d(b))
            });
            let Some(best) = at_best.map(|x| (*x).clone()) else { continue };
            let on_subject = track
                .points
                .iter()
                .filter(|p| p.seen)
                .any(|p| mine.iter().any(|x| (x.t - p.t).abs() < 1e-9 && x.subject && (f64::from(x.det.bbox[0]) - p.bbox[0]).abs() < 1e-3));
            found.push(Unmasked { track, best, subject: on_subject, person: None, checked: false });
        }
    }
    found.sort_by(|a, b| a.track.start().total_cmp(&b.track.start()));
    // what the vision model makes of them
    let judge = if verify { s.redacts.judge.clone() } else { None };
    let verified = match &judge {
        None if !verify => json!({"state": "off"}),
        None => json!({"state": "skipped", "reason": "no local vision model to tell people from robots: every unmasked face counts"}),
        Some(j) => ask(s, seq, scale, j.as_ref(), &mut found, sure),
    };
    let r = |v: f64| (v * 1000.0).round() / 1000.0;
    let list: Vec<Value> = found
        .iter()
        .map(|f| {
            let b = &f.best.det.bbox;
            let allowed = f.person == Some(false) && (f.subject || f.track.best_score < sure);
            let verdict = match (f.checked, f.person) {
                (false, _) => "unchecked",
                (true, Some(true)) => "person",
                (true, Some(false)) => "notPerson",
                (true, None) => "unsure",
            };
            let at = |p: Option<&filmcraft_vision::track::TrackPoint>| {
                p.map(|p| json!({"x": r(p.bbox[0]), "y": r(p.bbox[1]), "w": r(p.bbox[2]), "h": r(p.bbox[3])}))
            };
            json!({
                "start": r(f.track.start()),
                "end": r(f.track.end()),
                "seconds": r(f.best.t),
                "box": {"x": r(f64::from(b[0])), "y": r(f64::from(b[1])), "w": r(f64::from(b[2])), "h": r(f64::from(b[3]))},
                // where it was first and last seen (a face followed on in the next call)
                "first": at(f.track.points.iter().find(|p| p.seen)),
                "last": at(f.track.points.iter().rev().find(|p| p.seen)),
                "score": r(f64::from(f.track.best_score)),
                "hits": f.track.hits,
                "clip": f.best.under.as_ref().map(|u| u.clip.0),
                "item": f.best.under.as_ref().map(|u| u.item.0),
                "name": f.best.under.as_ref().map(|u| u.name.clone()),
                "media": f.best.under.as_ref().map(|u| r(u.media)),
                "at": f.best.under.as_ref().map(|u| [r(u.at[0]), r(u.at[1])]),
                "onSubject": f.subject,
                // found (also) by the profile pass: a head in profile or half hidden
                "fromPose": f.track.pose_hits > 0,
                "person": f.person,
                "verdict": verdict,
                "allowed": allowed,
            })
        })
        .collect();
    Ok(json!({
        "sequence": seq.0,
        "frames": rendered,
        "missedFrames": missed,
        "masked": masked,
        "tooSmall": small,
        "unmasked": list,
        "shown": list.iter().filter(|v| v["allowed"] != json!(true)).count(),
        "profiles": heavy,
        "verify": verified,
    }))
}

/// What two looks say together: not a person only when both say so; a person when the second
/// says so, or (away from the robot) the first did; on the robot, a first "person" the second
/// look does not confirm is unclear.
pub fn both_looks(first: Option<bool>, second: Option<bool>, subject: bool) -> Option<bool> {
    match (first, second) {
        (Some(false), Some(false)) => Some(false),
        (_, Some(true)) => Some(true),
        (Some(true), _) if !subject => Some(true),
        _ => None,
    }
}

/// Ask `judge` about each unmasked face (at its best moment, rendered again): the grid first, then
/// alone and outlined for a face it calls not a person where that would count.
fn ask(s: &Session, seq: ItemId, scale: f32, judge: &dyn VisionJudge, found: &mut [Unmasked], sure: f32) -> Value {
    let crop = |f: &Unmasked, mark: bool| -> Option<Crop> {
        let (rgba, w, h) = frame(s, seq, scale, f.best.t)?;
        let b = f.best.det.bbox.map(f64::from);
        crate::redact::crop_around(&rgba, w, h, &b, mark)
    };
    let picked: Vec<usize> = (0..found.len()).take(MAX_ASKED).collect();
    let mut crops: Vec<(usize, Crop)> = Vec::new();
    for i in picked {
        if let Some(c) = found.get(i).and_then(|f| crop(f, false)) {
            crops.push((i, c));
        }
    }
    let (mut asked, mut again) = (0usize, 0usize);
    for batch in crops.chunks(JUDGE_BATCH) {
        let pics: Vec<Crop> = batch.iter().map(|(_, c)| c.clone()).collect();
        let answers = match judge.judge(Kind::Face, &pics) {
            Ok(a) => a,
            Err(e) => return json!({"state": "failed", "reason": e, "judge": judge.name(), "asked": asked}),
        };
        asked += batch.len();
        for ((i, _), a) in batch.iter().zip(answers.into_iter().chain(std::iter::repeat(None))) {
            if let Some(f) = found.get_mut(*i) {
                f.person = a;
                f.checked = true;
            }
        }
    }
    // the second look: at a face called not a person where that would let it stay sharp, and at
    // any face on the robot (its own face is often called a person in a grid: two looks that
    // disagree make it one to check by eye, not a stop)
    let mut failed: Option<String> = None;
    let second = |f: &Unmasked| f.checked && (f.subject || (f.person == Some(false) && f.track.best_score < sure));
    for f in found.iter_mut().filter(|f| second(f)) {
        if failed.is_some() {
            // not looked at again: not allowed to stay sharp
            f.person = f.person.filter(|p| *p);
            continue;
        }
        let Some(c) = crop(f, true) else {
            f.person = f.person.filter(|p| *p);
            continue;
        };
        match judge.confirm(Kind::Face, &c) {
            Ok(a) => {
                again += 1;
                f.person = both_looks(f.person, a, f.subject);
            }
            Err(e) => {
                f.person = f.person.filter(|p| *p);
                failed = Some(e);
            }
        }
    }
    match failed {
        Some(e) => json!({"state": "failed", "reason": e, "judge": judge.name(), "asked": asked, "lookedAgain": again}),
        None => json!({"state": "done", "judge": judge.name(), "asked": asked, "lookedAgain": again}),
    }
}

#[cfg(test)]
#[path = "privacy_check_tests.rs"]
mod tests;
