//! The privacy check of a finished edit: faces the picture still shows.
//!
//! | command | does |
//! |---|---|
//! | `sequence.checkPrivacy` | renders the active sequence at each of `seconds` (as it exports: effects, masks, titles), runs the face detector on each frame, and says which faces no redaction mask hides: each followed across the frames it is seen in, with when, where (shares of the frame), the clip under it and where in that clip, how much blur covers it there (`blur`), whether it is on the subject (inside one of the clip's `exclude` boxes, at any moment), whether it is an unsure head seen in one frame alone (`brief`: not shown), and — with `verify` and the host's vision model — whether it is a person |
//!
//! It checks what [`crate::redact`] made, on the picture as a viewer sees it: a face the
//! redaction never found, a mask that does not cover the face it was made for, a face left sharp
//! for the robot's (the bug of 2026-10-21: visitors next to Ameca were left sharp because they
//! stood inside the robot's box). A face the frame shows is *masked* when the top-most media clip
//! there has an automatic redaction blur (`Redact …` masks) over it at that moment, as the mask
//! is drawn — its path, soft edge and expansion: covered at least [`HIDDEN`] (`blur_at`) — at
//! its centre (for a face cut by the frame whose centre lies outside the picture, the visible
//! point of its box nearest the centre, `visible_centre`), or at every one of its features (eyes,
//! nose and mouth corners; a head found from a pose: its nose and eyes), two at least: a face is
//! who it is by those (`hidden_by`; 2026-10-23, a head in profile whose face was blurred,
//! flagged for the back of its head beside the blur).
//!
//! **Seen once.** An unsure head found from a pose (under
//! `filmcraft_vision::track::UNSURE_POSE`) that the check sees in one of its frames alone is
//! looked for again a tenth, two and three tenths of a second before and after (`persists`):
//! unmasked and linked to it as the redaction links its sightings, it is shown; else it is
//! `brief` — the pose network's guess at a blur, a hand or a scarf, which the redaction (it hides
//! a face it saw twice) leaves too — reported, not shown and not asked about (2026-10-23: OnRobot
//! at 87 s, QA's only error there).
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
use filmcraft_vision::track::{Exclude, Frame, LinkParams, Track, UNSURE_POSE};
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
    /// The clip's automatic redaction hides it ([`hidden_by`]).
    masked: bool,
    /// How much the blur covers its point, 0..1 ([`blur_at`]).
    blur: f64,
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

/// A point of a face is hidden when the automatic redaction's blur covers it at least this much
/// ([`blur_at`]): inside the drawn mask's half-way line, where the blurred picture is mixed in
/// at half or more (a mask's padded shape is wholly covered; its soft edge falls off beyond).
pub const HIDDEN: f64 = 0.5;

/// How much the automatic redaction's blur covers clip pixel `p` of `it` at media time `mt`,
/// 0..1: each redaction effect's masks as the renderer draws them (their path, feather,
/// expansion, opacity and modes: `filmcraft_render::mask`), the effects over each other. The
/// check used to test the mask's path alone, a point just past it counting as sharp although
/// the blur's soft edge covered it (2026-10-23: a blurred head, its box a little lower than its
/// mask, flagged).
fn blur_at(it: &TrackItem, mt: Tick, p: Vec2) -> f64 {
    if !(p.x.is_finite() && p.y.is_finite()) {
        return 0.0;
    }
    let mut sharp = 1.0f64;
    for e in it.effects.iter().filter(|e| e.enabled && crate::redact::is_auto_redaction(e)) {
        let flat = filmcraft_render::mask::prepare(&e.masks, mt, 1.0);
        if let Some(c) = filmcraft_render::mask::coverage_at(&flat, p.x as f32, p.y as f32).filter(|c| c.is_finite()) {
            sharp *= 1.0 - f64::from(c).clamp(0.0, 1.0);
        }
    }
    (1.0 - sharp).clamp(0.0, 1.0)
}

/// Whether a face is hidden by the blur of a clip: its point `c` (clip pixels) covered at least
/// [`HIDDEN`], or every one of its features that the clip shows (`features`, clip pixels: a face
/// detector's eyes, nose and mouth corners, or the nose and eyes of a head found from a pose),
/// two at least. A face is who it is by its eyes, nose and mouth: a head in profile whose face is
/// under the blur and the back of its head beside it is hidden, though the middle of its box (between the
/// ear and the nose) is not (2026-10-23, IMG_9625 in the Quick Short: the face detector's face,
/// the front of a turned head, masked; the head found from the pose on the blurred frame flagged).
fn hidden_by(it: &TrackItem, mt: Tick, c: Vec2, features: &[Vec2]) -> (bool, f64) {
    let at = blur_at(it, mt, c);
    if at >= HIDDEN {
        return (true, at);
    }
    let all = features.len() >= 2 && features.iter().all(|f| blur_at(it, mt, *f) >= HIDDEN);
    (all, at)
}

/// The top-most media clip of `q` at `t` whose picture is at sequence pixel `c`, and where; its
/// face is masked when the clip's automatic redaction hides it there ([`hidden_by`], the face's
/// `features` in sequence pixels). In a transition both its clips are drawn: a face hidden in
/// either (the outgoing clip's visitor seen through a whip into the next shot) counts as masked.
fn under(s: &Session, q: &Sequence, t: Tick, c: Vec2, features: &[Vec2]) -> Option<Under> {
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
            let Some(size) = filmcraft_render::clip_size(&s.project, it).filter(|(w, h)| *w > 0 && *h > 0) else { continue };
            let mt = it.effect_time_at(t);
            let m = filmcraft_render::motion_matrix(q, it, size, filmcraft_render::clip_par(&s.project, it), mt);
            let Some(inv) = m.inverse() else { continue };
            let sp = inv.apply(c);
            let (w, h) = (f64::from(size.0), f64::from(size.1));
            if !(sp.x.is_finite() && sp.y.is_finite() && (0.0..w).contains(&sp.x) && (0.0..h).contains(&sp.y)) {
                continue;
            }
            // the face's features this clip shows, in its pixels
            let shown: Vec<Vec2> = features
                .iter()
                .map(|f| inv.apply(*f))
                .filter(|f| f.x.is_finite() && f.y.is_finite() && (0.0..w).contains(&f.x) && (0.0..h).contains(&f.y))
                .collect();
            let (masked, blur) = hidden_by(it, mt, sp, &shown);
            let u = Under { clip: it.id, item: it.item, name: it.name.clone(), media: mt.seconds(), at: [sp.x / w, sp.y / h], masked, blur };
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

/// The point of face box `b` (normalised) the check looks under, sequence pixels (`sw` × `sh`):
/// its centre, or for a face cut by the frame whose centre lies outside it, the visible point of
/// its box nearest the centre (half a pixel inside the frame). A head half outside the frame (a
/// pose head's box runs past the edge) is masked when the part the frame shows is, and flagged
/// when it is not; its centre outside the picture was under no clip, so it was always flagged,
/// masked or not (2026-10-22).
fn visible_centre(b: &[f32; 4], sw: f64, sh: f64) -> Vec2 {
    let b = b.map(f64::from);
    let (cx, cy) = ((b[0] + b[2] / 2.0) * sw, (b[1] + b[3] / 2.0) * sh);
    let inside = |v: f64, lo: f64, hi: f64, size: f64| {
        let (lo, hi) = ((lo * size).max(0.5), (hi * size).min(size - 0.5));
        if lo <= hi { v.clamp(lo, hi) } else { v }
    };
    Vec2::new(inside(cx, b[0], b[0] + b[2], sw), inside(cy, b[1], b[1] + b[3], sh))
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

/// The faces the detector finds in a frame (RGBA, `w` × `h`): the face detector's scoring at
/// least `score`, and the heads found from poses (they have passed their own thresholds).
fn faces_in(det: &DetectFn, rgba: &[u8], w: usize, h: usize, score: f32) -> Vec<Detection> {
    det(rgba, w, h, w.max(h))
        .into_iter()
        .filter(|d| d.kind == Kind::Face && (d.pose || d.score >= score) && d.bbox.iter().all(|v| v.is_finite()))
        .take(MAX_FACES)
        .collect()
}

/// The features of face `d` (its landmarks, normalised) in sequence pixels (`sw` × `sh`), those
/// inside the frame.
fn features(d: &Detection, sw: f64, sh: f64) -> Vec<Vec2> {
    d.points
        .iter()
        .map(|p| Vec2::new(f64::from(p[0]) * sw, f64::from(p[1]) * sh))
        .filter(|p| p.x.is_finite() && p.y.is_finite() && (0.0..sw).contains(&p.x) && (0.0..sh).contains(&p.y))
        .collect()
}

/// The step, seconds, of the frames looked at again around a head found from a pose seen once
/// ([`persists`]: the redaction's analysis step), and how many steps each way (within the reach
/// the redaction links sightings over, `filmcraft_vision::track::LinkParams::max_gap`).
const AGAIN_STEP: f64 = 0.1;
const AGAIN_STEPS: usize = 3;

/// Whether head `x`, found from a pose and seen once, is seen again unmasked in the same clip:
/// the frames [`AGAIN_STEP`] apart before and after it, out to [`AGAIN_STEPS`] steps each way
/// (nearest first), rendered and looked at as `x`'s was, their unmasked faces linked with it as
/// the redaction links its sightings (`LinkParams::default()`), until its track has the
/// sightings the redaction hides a face on (`KeepParams::min_hits`; a head from a pose is never
/// sure enough for one). The redaction leaves a head it saw once, and the check agrees with it:
/// an unsure head seen in one frame alone is the pose network's guess at the blur (2026-10-23,
/// OnRobot at 87 s: a head of 0.37 on a blurred face and a dark scarf, QA's only error there),
/// not a face shown.
fn persists(s: &Session, seq: ItemId, q: &Sequence, scale: f32, det: &DetectFn, x: &Seen, score: f32, (sw, sh): (f64, f64)) -> bool {
    let Some(clip) = x.under.as_ref().map(|u| u.clip) else { return true };
    let end = q.duration().seconds();
    let keep = filmcraft_vision::track::KeepParams::default();
    let mut frames: Vec<Frame> = vec![Frame { t: x.t, detections: vec![x.det.clone()] }];
    for k in 1..=AGAIN_STEPS {
        for dir in [-1.0, 1.0] {
            let t = x.t + dir * k as f64 * AGAIN_STEP;
            let at = Tick::from_seconds_f64(t);
            // the same clip there (not the next shot)
            if !(t >= 0.0 && t < end) || under(s, q, at, visible_centre(&x.det.bbox, sw, sh), &[]).is_none_or(|u| u.clip != clip) {
                continue;
            }
            let Some((rgba, w, h)) = frame(s, seq, scale, t) else { continue };
            let unmasked: Vec<Detection> = faces_in(det, &rgba, w, h, score)
                .into_iter()
                .filter(|d| under(s, q, at, visible_centre(&d.bbox, sw, sh), &features(d, sw, sh)).is_some_and(|u| u.clip == clip && !u.masked))
                .collect();
            frames.push(Frame { t, detections: unmasked });
        }
        frames.sort_by(|a, b| a.t.total_cmp(&b.t));
        // `x` is the only sighting at its moment: its track is the one seen then
        let tracks = filmcraft_vision::track::link(&frames, &LinkParams::default());
        let mine = tracks.iter().find(|tr| tr.points.iter().any(|p| p.seen && (p.t - x.t).abs() < 1e-9));
        if mine.is_some_and(|tr| tr.hits >= keep.min_hits.max(2) || tr.best_score >= keep.sure_score) {
            return true;
        }
    }
    false
}

/// An unmasked face followed across the frames it is seen in, and what the check made of it.
struct Unmasked {
    track: Track,
    best: Seen,
    subject: bool,
    person: Option<bool>,
    checked: bool,
    /// An unsure head found from a pose seen in one frame alone ([`persists`]): not shown.
    brief: bool,
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
        for d in faces_in(&det, &rgba, w, h, score) {
            if f64::from(d.bbox[3]) < min_h {
                small += 1;
                continue;
            }
            let c = visible_centre(&d.bbox, sw, sh);
            let u = under(s, &q, Tick::from_seconds_f64(t), c, &features(&d, sw, sh));
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
        // the faces as seen, each sighting reported: boxes compared as they are, no size
        // smoothing or spike removal, no longer gaps after an unsure sighting and no camera
        // motion compensation (a face the frame shows is reported where it shows, as before)
        grow: 0.0,
        smooth_size: 0.0,
        unsure_gap: 0.0,
        pan: false,
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
            found.push(Unmasked { track, best, subject: on_subject, person: None, checked: false, brief: false });
        }
    }
    found.sort_by(|a, b| a.track.start().total_cmp(&b.track.start()));
    // an unsure head found from a pose and seen once: shown only if it is seen again
    for f in found.iter_mut().filter(|f| f.track.hits < 2 && f.track.pose_hits == f.track.hits && f.track.best_score < UNSURE_POSE) {
        f.brief = !persists(s, seq, &q, scale, &det, &f.best, score, (sw, sh));
    }
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
            let blur = f.best.under.as_ref().map_or(0.0, |u| u.blur);
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
                // an unsure head from a pose seen in one frame alone: not shown (`persists`)
                "brief": f.brief,
                // how much the blur covers its point there (under HIDDEN: not hidden)
                "blur": r(blur),
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
        "shown": list.iter().filter(|v| v["allowed"] != json!(true) && v["brief"] != json!(true)).count(),
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
    // a head seen in one frame alone is not shown: not asked about
    let picked: Vec<usize> = (0..found.len()).filter(|i| found.get(*i).is_some_and(|f| !f.brief)).take(MAX_ASKED).collect();
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
