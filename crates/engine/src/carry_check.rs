//! `sequence.checkCarry` (PX6.3): how well each cut carries the eye from one shot into the next.
//! The idea of a continuity check across cuts comes from onetake's continuity oracle
//! (feitangyuan/onetake, PolyForm Noncommercial): only the idea is taken, from its README; the
//! measures, thresholds and words here are written fresh.
//!
//! Every cut of the sequence — two pictures butting on an enabled video track, titles and other
//! graphics left out — is rendered on its two sides (the last frame before it and its first
//! frame, as the viewer sees them; the GPU frame renderer where the host has one, else the CPU,
//! at 192 pixels on the long side, without captions) and measured:
//!
//! | measure | how |
//! |---|---|
//! | subject | the salient region of each side ([`filmcraft_render::subject::salient_box`]: frequency-tuned saliency with a centre and a skin-tone prior) and how far its centre moves, as a share of the frame (0 in the same place, 1 corner to corner) |
//! | palette | how far the two sides' colours are apart: half the L1 distance of their 4×4×4 colour histograms (0 the same colours, 1 none shared) |
//! | carry | `1 − (0.6·min(1, move / 0.5) + 0.4·min(1, palette / 0.8))`; with no subject on a side, the palette alone |
//!
//! | code | warns when | starting value |
//! |---|---|---|
//! | `subjectJump` | the subject's centre moves across the cut by at least this share of the frame: the eye has to search for it | [`SUBJECT_JUMP`] |
//! | `paletteJump` | the colours change this much at a plain cut | [`PALETTE_JUMP`] |
//!
//! A cut under a transition (a dissolve, a whip) is scored but not flagged: the transition is the
//! bridge. Every finding is a warning (decision D7), with what to change. A montage of different
//! places cuts on contrast on purpose, so this check reads the cuts meant to flow (a scene, a
//! match cut, a one-take style) and is not part of `delivery.preflight`. At most [`MAX_CUTS`]
//! cuts are read (two renders each); the rest are listed as skipped.
use std::collections::BTreeSet;

use filmcraft_project::{ItemId, ItemKind, Project};
use filmcraft_render::subject::{SubjectBox, salient_box};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, has_seq};
use crate::motion_energy::{bounded, finding, r3};
use crate::{EngineError, Result, Session};

const CMD: &str = "sequence.checkCarry";
/// A subject moving this share of the frame across a cut has jumped.
pub const SUBJECT_JUMP: f64 = 0.3;
/// Colours this far apart at a plain cut have jumped.
pub const PALETTE_JUMP: f64 = 0.6;
/// The carry score's scales: a move of half the frame, and a palette distance of 0.8, count fully.
const MOVE_SCALE: f64 = 0.5;
const PALETTE_SCALE: f64 = 0.8;
/// Most cuts read, by default and at most.
pub const DEFAULT_CUTS: usize = 200;
pub const MAX_CUTS: usize = 400;
/// The rendered picture's long side, pixels.
const LONG_SIDE: f64 = 192.0;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Check Continuity Across Cuts",
        menu: &[],
        shortcut: None,
        params: r#"{"sequence":id? (default the active one),"maxCuts":1..400=200 (cuts read, from the start)}"#,
        enabled: has_seq,
        run,
        journal: false,
    }]
}

/// A cut: where, on which track, the clips either side and the transition over it.
#[derive(Clone, Debug)]
struct Cut {
    at: Tick,
    track: usize,
    out_clip: u64,
    in_clip: u64,
    transition: Option<String>,
}

/// Whether a clip of `item` is a picture (media with video, or a nested sequence), not a title.
fn picture(project: &Project, item: ItemId) -> bool {
    match project.item(item).map(|i| &i.kind) {
        Some(ItemKind::Media(m)) => m.info.video.is_some(),
        Some(ItemKind::Sequence(_)) | Some(ItemKind::Subclip { .. }) => project.item(item).is_some_and(|i| i.has_video()),
        _ => false,
    }
}

/// The cuts of `q`: pictures that butt (within a frame) on an enabled video track, in time order.
fn cuts_of(project: &Project, q: &filmcraft_project::Sequence) -> Vec<Cut> {
    let frame = q.settings.frame_rate.sane().frame_duration().0.max(1);
    let mut out = Vec::new();
    for (ti, t) in q.video_tracks.iter().enumerate().filter(|(_, t)| t.enabled) {
        let mut clips: Vec<&filmcraft_project::TrackItem> = t.items.iter().filter(|c| c.enabled && picture(project, c.item)).collect();
        clips.sort_by_key(|c| c.start);
        for pair in clips.windows(2) {
            let [a, b] = pair else { continue };
            let end = a.end();
            if (b.start.0 - end.0).abs() > frame || b.start <= Tick::ZERO {
                continue;
            }
            let transition = t.transitions.iter().find(|x| x.start <= b.start && x.end() >= b.start && x.duration.0 > 0).map(|x| x.effect.effect.clone());
            out.push(Cut { at: b.start, track: ti, out_clip: a.id.0, in_clip: b.id.0, transition });
        }
    }
    out.sort_by_key(|c| (c.at, c.track));
    let mut seen = BTreeSet::new();
    out.retain(|c| seen.insert(c.at.0 / frame));
    out
}

/// A rendered side of a cut: its salient box and colour histogram.
struct Side {
    subject: Option<SubjectBox>,
    hist: Vec<f64>,
}

fn side_of(w: usize, h: usize, rgba: &[u8]) -> Side {
    let mut hist = vec![0.0f64; 64];
    let px = rgba.as_chunks::<4>().0;
    for p in px {
        if let Some(b) = hist.get_mut(usize::from(p[0] >> 6) * 16 + usize::from(p[1] >> 6) * 4 + usize::from(p[2] >> 6)) {
            *b += 1.0;
        }
    }
    let n = px.len().max(1) as f64;
    for b in hist.iter_mut() {
        *b /= n;
    }
    Side { subject: salient_box(rgba, w, h), hist }
}

/// Half the L1 distance of two colour histograms (0 the same … 1 nothing shared).
fn palette_distance(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / 2.0).clamp(0.0, 1.0)
}

/// How far a subject's centre moves, as a share of the frame (0..1, corner to corner 1).
fn move_of(a: &SubjectBox, b: &SubjectBox) -> f64 {
    (((a.cx - b.cx).powi(2) + (a.cy - b.cy).powi(2)).sqrt() / std::f64::consts::SQRT_2).clamp(0.0, 1.0)
}

/// The carry score of a cut (1 carries the eye, 0 throws it).
fn carry(moved: Option<f64>, palette: f64) -> f64 {
    let p = (palette / PALETTE_SCALE).min(1.0);
    let c = match moved {
        Some(m) => 1.0 - (0.6 * (m / MOVE_SCALE).min(1.0) + 0.4 * p),
        None => 1.0 - p,
    };
    c.clamp(0.0, 1.0)
}

fn box_json(b: &SubjectBox) -> Value {
    json!({"cx": r3(b.cx), "cy": r3(b.cy), "w": r3(b.w), "h": r3(b.h)})
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    if !p.is_object() && !p.is_null() {
        return Err(bad(CMD, "parameters must be an object"));
    }
    if let Some(k) = p.as_object().and_then(|o| o.keys().find(|k| !matches!(k.as_str(), "sequence" | "maxCuts"))) {
        return Err(bad(CMD, format!("unknown parameter `{}`: sequence or maxCuts", k.chars().take(64).collect::<String>())));
    }
    let seq = match p.get("sequence").filter(|v| !v.is_null()) {
        Some(v) => {
            let id = ItemId(v.as_u64().ok_or_else(|| bad(CMD, "`sequence` is a sequence id"))?);
            s.project.sequence(id).ok_or_else(|| bad(CMD, "no such sequence"))?;
            id
        }
        None => s.state.active_sequence.ok_or(EngineError::NoSequence)?,
    };
    let max_cuts = match p.get("maxCuts").filter(|v| !v.is_null()) {
        None => DEFAULT_CUTS,
        Some(v) => v.as_u64().filter(|n| (1..=MAX_CUTS as u64).contains(n)).ok_or_else(|| bad(CMD, format!("`maxCuts` is 1 to {MAX_CUTS}")))? as usize,
    };
    let project = s.project.clone();
    let q = project.sequence(seq).ok_or(EngineError::NoSequence)?;
    q.check_bounds().map_err(|e| bad(CMD, e))?;
    if q.duration().0 <= 0 {
        return Err(bad(CMD, "the sequence is empty"));
    }
    let rate = q.settings.frame_rate.sane();
    let frame = rate.frame_duration();
    let (w, h) = (q.settings.width.max(1), q.settings.height.max(1));
    let all = cuts_of(&project, q);
    let skipped_cuts = all.len().saturating_sub(max_cuts);
    let provider = s.media.provider(project.clone(), s.services.clone());
    let mut gpu = filmcraft_export::frame_renderer();
    let (mut on_gpu, mut on_cpu) = (0usize, 0usize);
    let scale = (LONG_SIDE / f64::from(w.max(h))).min(1.0) as f32;
    let opts = filmcraft_render::RenderOptions { scale, ..Default::default() };
    let mut render = |t: Tick| -> Side {
        let img = match gpu.as_mut().and_then(|r| r.render(&project, seq, t, opts, &provider)) {
            Some(img) => {
                on_gpu += 1;
                img
            }
            None => {
                on_cpu += 1;
                filmcraft_render::render_sequence(&project, seq, t, opts, &provider)
            }
        };
        side_of(img.w, img.h, &img.over_black_rgba8())
    };
    let mut findings = Vec::new();
    let mut scored = Vec::new();
    for c in all.iter().take(max_cuts) {
        let before = render(Tick(c.at.0.saturating_sub(frame.0).max(0)));
        let after = render(c.at);
        let palette = palette_distance(&before.hist, &after.hist);
        let moved = match (&before.subject, &after.subject) {
            (Some(a), Some(b)) => Some(move_of(a, b)),
            _ => None,
        };
        let score = carry(moved, palette);
        let at = c.at;
        if c.transition.is_none() {
            if let (Some(m), Some(a), Some(b)) = (moved, &before.subject, &after.subject)
                && m >= SUBJECT_JUMP
            {
                findings.push(finding(
                    "subjectJump",
                    at,
                    None,
                    format!(
                        "at {:.2} s the subject jumps {:.0} % of the frame across the cut (from {:.0} %, {:.0} % to {:.0} %, {:.0} %): the eye has to search for it; start the incoming shot with its subject near where the eye already is (reframe it, or move its Position), cut on a movement, or bridge the cut with a transition",
                        at.seconds(),
                        m * 100.0,
                        a.cx * 100.0,
                        a.cy * 100.0,
                        b.cx * 100.0,
                        b.cy * 100.0
                    ),
                    json!({"move": r3(m), "from": box_json(a), "to": box_json(b), "carry": r3(score), "clips": [c.out_clip, c.in_clip]}),
                ));
            }
            if palette >= PALETTE_JUMP {
                findings.push(finding(
                    "paletteJump",
                    at,
                    None,
                    format!(
                        "at {:.2} s the colours change sharply across the cut ({:.0} % of the palette): if the two shots belong together, match them (clip.autoColor with match) or bridge the cut with a dissolve",
                        at.seconds(),
                        palette * 100.0
                    ),
                    json!({"palette": r3(palette), "carry": r3(score), "clips": [c.out_clip, c.in_clip]}),
                ));
            }
        }
        scored.push(json!({
            "time": at.0,
            "seconds": r3(at.seconds()),
            "carry": r3(score),
            "subject": match (&before.subject, &after.subject) {
                (Some(a), Some(b)) => json!({"from": box_json(a), "to": box_json(b), "move": moved.map(r3)}),
                (a, b) => json!({"from": a.as_ref().map(box_json), "to": b.as_ref().map(box_json), "move": null, "note": "no salient subject on one side: scored by colour alone"}),
            },
            "palette": r3(palette),
            "transition": c.transition,
            "track": format!("V{}", c.track + 1),
            "clips": [c.out_clip, c.in_clip],
        }));
    }
    let (findings, counts) = bounded(findings);
    let carries: Vec<f64> = scored.iter().filter_map(|c| c["carry"].as_f64()).collect();
    let mean = (!carries.is_empty()).then(|| carries.iter().sum::<f64>() / carries.len() as f64);
    let mut lowest: Vec<&Value> = scored.iter().filter(|c| c["transition"].is_null()).collect();
    lowest.sort_by(|a, b| a["carry"].as_f64().unwrap_or(1.0).total_cmp(&b["carry"].as_f64().unwrap_or(1.0)));
    let lowest: Vec<Value> = lowest.iter().take(5).map(|c| json!({"time": c["time"], "seconds": c["seconds"], "carry": c["carry"]})).collect();
    let mut skipped = Vec::new();
    if skipped_cuts > 0 {
        skipped.push(json!({"check": "all", "reason": format!("only the first {max_cuts} cuts were read; {skipped_cuts} more were not")}));
    }
    Ok(json!({
        "sequence": seq.0,
        "ok": findings.is_empty(),
        "blocking": false,
        "findings": findings,
        "counts": counts,
        "skipped": skipped,
        "cuts": scored,
        "summary": {"cuts": all.len(), "read": carries.len(), "meanCarry": mean.map(r3), "lowest": lowest},
        "renderer": {"gpuFrames": on_gpu, "cpuFrames": on_cpu, "captions": false},
        "thresholds": {
            "subjectJump": SUBJECT_JUMP, "paletteJump": PALETTE_JUMP, "moveScale": MOVE_SCALE, "paletteScale": PALETTE_SCALE,
            "note": "starting values, to be tuned on real projects",
        },
        "method": {
            "cuts": "pictures that butt within a frame on an enabled video track (titles and graphics left out); a cut under a transition is scored, not flagged",
            "sides": "the frame before the cut and the cut's own frame, rendered as the viewer sees them, 192 pixels on the long side",
            "subject": "frequency-tuned saliency with a centre and a skin-tone prior on each side; its centre's move as a share of the frame",
            "palette": "half the L1 distance of 4×4×4 colour histograms",
            "delivery": "not a leg of delivery.preflight: a montage cuts on contrast on purpose, so this reads the cuts meant to flow",
        },
    }))
}

#[cfg(test)]
#[path = "carry_check_tests.rs"]
mod tests;
