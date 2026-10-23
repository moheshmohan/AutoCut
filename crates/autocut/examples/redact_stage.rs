//! One stage of the privacy blur's live check, so that two builds are measured the same way: a
//! window of a clip redacted with the profile pass (`clip.autoRedact`; stage `original` leaves it
//! as it is), the privacy QA on it (`sequence.checkPrivacy` every quarter second, with the profile
//! pass), the masks measured, and its frames and video written:
//!
//! - `<stage>.json`: the redaction's outcome, the QA's answer, and the masks' measures — the share
//!   of the frame they cover (inside their 50 % edge, and fully), neighbouring masks whose soft
//!   edges join although their shapes do not touch, and flicker: dropouts (a part of the frame
//!   blurred, sharp for under half a second and blurred again) in the whole frame and at its
//!   edges, the gaps where one face's blur passes from a mask to the next, the masks' on/off
//!   switches and those shown under a second, and their size and position jitter ([`measure`]);
//! - `<stage>-masks.json`: at each sample (media seconds), the bounds of each mask that shows
//!   (shares of the picture), to measure against faces found elsewhere;
//! - `<stage>/<media ms>.jpg`: the window's frames every `--every` seconds (0.1) as the sequence
//!   exports them;
//! - `<stage>.mp4`: the first `--video` seconds (10) of the window.
//!
//! `--time-only` measures the analysis alone (a dry run without the vision check), for timing
//! under load. `--compose <out.mp4> <a.mp4> <b.mp4> <c.mp4>` puts three videos side by side.
//! `--crops <out dir> <clip dir> <stage>...` cuts the same places from several stages' frames,
//! side by side: each face the last stage hides and each face its QA still shows.
//!
//! ```text
//! FILMCRAFT_DATA_DIR=<dir holding models/privacy-detectors with the profile models> \
//!   cargo run --release -p filmcraft-autocut --features filmcraft-engine/privacy-models \
//!   --example redact_stage -- <video> <from s> <to s> <out dir> <stage> [--no-judge] [--exclude x,y,w,h]
//! ```
//!
//! The vision check is AutoCut's (Ollama, `AUTOCUT_*` environment). Nothing outside `<out dir>` is
//! written; the video is only read.

use std::path::{Path, PathBuf};

use filmcraft_engine::Session;
use filmcraft_project::{ItemId, ParamValue, SequenceSettings, TrackKind};
use filmcraft_time::{Tick, TimeRange};
use serde_json::{Value, json};

fn fail(m: impl std::fmt::Display) -> ! {
    eprintln!("redact_stage: {m}");
    std::process::exit(1)
}

/// A session with `video` from `from` to `to` (seconds) on V1 of a sequence of its size.
fn session(video: &Path, from: f64, to: f64) -> (Session, ItemId) {
    let mut s = Session::default();
    let r = s.execute("file.import", json!({"paths": [video.to_string_lossy()]})).unwrap_or_else(|e| fail(e));
    let item = r["items"][0].as_u64().map(ItemId).unwrap_or_else(|| fail(format!("import: {r}")));
    let info = s.project.item(item).and_then(|i| i.as_media()).and_then(|m| m.info.video.clone()).unwrap_or_else(|| fail("not a video"));
    let rate = info.frame_rate;
    let mut p = (*s.project).clone();
    let seq = p.new_sequence("Privacy check", SequenceSettings { width: info.width, height: info.height, frame_rate: rate, ..Default::default() }, 1, 1, None);
    let src = TimeRange::new(Tick::from_seconds_f64(from), Tick::from_seconds_f64(to - from));
    let mut v = p.make_track_item(item, TrackKind::Video, Tick::ZERO, src, rate).unwrap_or_else(|| fail("cannot place the clip"));
    for e in &mut v.effects {
        filmcraft_project::resolve_auto_points(e, (info.width, info.height), (info.width, info.height));
    }
    let clip = v.id;
    match p.sequence_mut(seq).and_then(|q| q.video_tracks.first_mut()) {
        Some(t) => t.items.push(v),
        None => fail("the sequence has no video track"),
    }
    s.project = std::sync::Arc::new(p);
    s.state.active_sequence = Some(seq);
    s.state.open_sequences = vec![seq];
    s.state.selection = vec![clip];
    (s, seq)
}

/// Signed distance from `(px, py)` to the polygon `poly` (positive inside, even-odd).
fn signed_distance(poly: &[(f64, f64)], (px, py): (f64, f64)) -> f64 {
    let n = poly.len();
    let (mut inside, mut best) = (false, f64::INFINITY);
    for i in 0..n {
        let (Some(&(ax, ay)), Some(&(bx, by))) = (poly.get(i), poly.get((i + n - 1) % n)) else { continue };
        if (ay > py) != (by > py) {
            let x = ax + (py - ay) * (bx - ax) / (by - ay);
            if px < x {
                inside = !inside;
            }
        }
        let (dx, dy) = (bx - ax, by - ay);
        let len = dx * dx + dy * dy;
        let t = if len > 0.0 { (((px - ax) * dx + (py - ay) * dy) / len).clamp(0.0, 1.0) } else { 0.0 };
        let (cx, cy) = (ax + t * dx - px, ay + t * dy - py);
        best = best.min((cx * cx + cy * cy).sqrt());
    }
    if inside { best } else { -best }
}

/// One mask at one moment: its polygon (clip pixels), expansion and feather, whether it shows.
struct MaskAt {
    poly: Vec<(f64, f64)>,
    expansion: f64,
    feather: f64,
    on: bool,
    /// The cells it may cover (its shape grown by its soft edge), and its shape's bounds.
    bounds: [f64; 4],
    shape: [f64; 4],
}

/// A part of the frame blurred, sharp for less than this (seconds) and blurred again is a
/// dropout: the blur flickering off (a mask switching off and on, one face's mask handed to
/// another with a gap, a mask's edge jittering). (Blurred for less than this between two sharp
/// stretches is a flash; a small mask moving fast flashes over the cells it passes too.)
const BLINK: f64 = 0.5;
/// A mask switching off and another (or itself) switching on within this (seconds), centred
/// within the first one's size, is the blur of one face handed over with a gap.
const HANDOVER: f64 = 1.0;
/// The frame's edge band, a share of its width and height on each side: where the faces cut by
/// the frame are.
const EDGE_BAND: f64 = 0.1;

/// One cell's history: its state, the samples since it last switched, and whether that run began
/// with a switch (not at the window's start).
#[derive(Clone, Copy, Default)]
struct Cell {
    on: bool,
    run: u32,
    switched: bool,
    toggles: u32,
    dropouts: u32,
    flashes: u32,
    ever: bool,
}

/// The value at share `p` of sorted `v` (0 when empty).
fn quantile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(((v.len() as f64 * p) as usize).min(v.len().saturating_sub(1))).copied().unwrap_or(0.0)
}

/// The masks' measures over the window, sampled every `step` seconds on a grid of cells (a cell
/// is covered when the blur there is at least half on: inside a showing mask's 50 % edge):
///
/// - area: the share of the frame covered (mean, largest), fully hidden, ever covered, and each
///   mask's own share while it shows (median, 90th percentile);
/// - merging: pairs of neighbouring masks whose soft edges join although their shapes do not
///   touch (pairs × samples);
/// - flicker: dropouts (a covered cell uncovered for less than [`BLINK`] and covered again) per
///   covered cell per second, in the whole frame and in its edge band ([`EDGE_BAND`]), and the
///   share of the frame that drops out a second; flashes (the reverse); the gaps where one
///   face's blur is handed from a mask to the next ([`HANDOVER`]); the masks' own on/off
///   switches, and those shown under a second; their jitter between samples while they show —
///   the change of their size, and the second difference of their centre (acceleration), as
///   shares of their size.
fn measure(s: &Session, seq: ItemId, len: f64, step: f64) -> Value {
    let Some(q) = s.project.sequence(seq) else { return Value::Null };
    let Some(it) = q.video_tracks.first().and_then(|t| t.items.first()) else { return Value::Null };
    let Some((w, h)) = filmcraft_engine::render::source_size(&s.project, it.item) else { return Value::Null };
    let (w, h) = (f64::from(w), f64::from(h));
    let masks: Vec<&filmcraft_project::Mask> =
        it.effects.iter().filter(|e| filmcraft_engine::redact::is_auto_redaction(e)).flat_map(|e| e.masks.iter()).collect();
    let (gw, gh) = (192usize, 108usize);
    let blink_samples = (BLINK / step).round().max(1.0) as u32;
    let handover_samples = (HANDOVER / step).round().max(1.0) as usize;
    // the masks switched off: (sample, centre, size), and the handover gaps (samples)
    let mut offs: Vec<(usize, f64, f64, f64)> = Vec::new();
    let mut gaps: Vec<usize> = Vec::new();
    // each sample's masks that show: media seconds, then each one's shape bounds (shares of the
    // picture) and opacity, for measuring against faces found elsewhere
    let mut shown: Vec<Value> = Vec::new();
    let times: Vec<f64> = (0..).map(|i| f64::from(i) * step).take_while(|t| *t < len - 1e-9).collect();
    let mut cells = vec![Cell::default(); gw * gh];
    let (mut area, mut core, mut area_max) = (Vec::new(), Vec::new(), 0f64);
    let mut on_prev: Vec<Option<bool>> = vec![None; masks.len()];
    let mut mask_toggles = 0usize;
    let mut on_time = vec![0f64; masks.len()];
    let mut joined = 0usize;
    // each mask's own share of the frame at each sample it shows
    let mut each: Vec<f64> = Vec::new();
    // each mask's last two (centre, size) while it shows, and the jitters
    let mut last: Vec<Vec<(f64, f64, f64)>> = vec![Vec::new(); masks.len()];
    let (mut size_jitter, mut centre_jitter): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    for (k, &t) in times.iter().enumerate() {
        let mt = it.effect_time_at(Tick::from_seconds_f64(t));
        let mut here: Vec<Value> = Vec::new();
        let now: Vec<MaskAt> = masks
            .iter()
            .map(|m| {
                let poly: Vec<(f64, f64)> = m.path_at(mt).flatten(1.0).iter().map(|p| (p.x, p.y)).collect();
                let (expansion, feather) = (m.expansion.f64_at(mt), m.feather.f64_at(mt).max(0.0));
                let (x0, y0) = poly.iter().fold((f64::INFINITY, f64::INFINITY), |(a, b), p| (a.min(p.0), b.min(p.1)));
                let (x1, y1) = poly.iter().fold((f64::NEG_INFINITY, f64::NEG_INFINITY), |(a, b), p| (a.max(p.0), b.max(p.1)));
                let g = expansion.abs() + feather;
                MaskAt { poly, expansion, feather, on: m.opacity.f64_at(mt) >= 50.0, bounds: [x0 - g, y0 - g, x1 + g, y1 + g], shape: [x0, y0, x1, y1] }
            })
            .collect();
        for m in now.iter().filter(|m| m.on && m.poly.len() > 2) {
            let b = m.shape;
            let r = |v: f64| (v * 10000.0).round() / 10000.0;
            here.push(json!([r(b[0] / w), r(b[1] / h), r(b[2] / w), r(b[3] / h)]));
        }
        shown.push(json!([mt.seconds(), here]));
        for (i, m) in now.iter().enumerate() {
            let b = m.shape;
            let (cx, cy, size) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, ((b[2] - b[0]) * (b[3] - b[1])).max(0.0).sqrt());
            if let Some(slot) = on_prev.get_mut(i) {
                if slot.is_some_and(|p| p != m.on) {
                    mask_toggles += 1;
                    if m.on {
                        // switched on: near where a mask switched off a moment ago?
                        if let Some(at) =
                            offs.iter().filter(|(j, x, y, sz)| k - j <= handover_samples && (cx - x).hypot(cy - y) <= sz.max(size)).map(|(j, ..)| *j).max()
                        {
                            gaps.push(k - at);
                        }
                    } else if size.is_finite() {
                        offs.push((k, cx, cy, size));
                    }
                }
                *slot = Some(m.on);
            }
            if m.on
                && let Some(o) = on_time.get_mut(i)
            {
                *o += step;
            }
            let Some(hist) = last.get_mut(i) else { continue };
            if !m.on || m.poly.len() < 3 {
                hist.clear();
                continue;
            }
            if let Some(&(_, _, s0)) = hist.last()
                && s0 > 0.0
            {
                size_jitter.push((size - s0).abs() / s0);
            }
            if let [(ax, ay, _), (bx, by, sb)] = hist.as_slice()
                && *sb > 0.0
            {
                centre_jitter.push((cx - 2.0 * bx + ax).hypot(cy - 2.0 * by + ay) / sb);
            }
            hist.push((cx, cy, size));
            if hist.len() > 2 {
                hist.remove(0);
            }
        }
        let mut cov = vec![false; gw * gh];
        let mut full = vec![false; gw * gh];
        // which mask's 50 % edge and which shape covers each cell
        let mut owner: Vec<Option<usize>> = vec![None; gw * gh];
        let mut shape_owner: Vec<Option<usize>> = vec![None; gw * gh];
        let mut pairs: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        let mut shape_pairs: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        for (mi, m) in now.iter().enumerate().filter(|(_, m)| m.on && m.poly.len() > 2) {
            let mut own = 0usize;
            let cell = |v: f64, n: usize| (v.max(0.0) as usize).min(n);
            let (cx0, cx1) = (cell((m.bounds[0] / w * gw as f64).floor(), gw), cell((m.bounds[2] / w * gw as f64).ceil(), gw));
            let (cy0, cy1) = (cell((m.bounds[1] / h * gh as f64).floor(), gh), cell((m.bounds[3] / h * gh as f64).ceil(), gh));
            for cy in cy0..cy1 {
                for cx in cx0..cx1 {
                    let p = ((cx as f64 + 0.5) * w / gw as f64, (cy as f64 + 0.5) * h / gh as f64);
                    let sd = signed_distance(&m.poly, p);
                    let i = cy * gw + cx;
                    if sd + m.expansion >= 0.0 {
                        own += 1;
                        if let Some(c) = cov.get_mut(i) {
                            *c = true;
                        }
                        if let Some(o) = owner.get_mut(i) {
                            if let Some(prev) = o.filter(|o| *o != mi) {
                                pairs.insert((prev.min(mi), prev.max(mi)));
                            }
                            *o = Some(mi);
                        }
                    }
                    if sd + m.expansion - m.feather / 2.0 >= 0.0
                        && let Some(f) = full.get_mut(i)
                    {
                        *f = true;
                    }
                    if sd >= 0.0
                        && let Some(o) = shape_owner.get_mut(i)
                    {
                        if let Some(prev) = o.filter(|o| *o != mi) {
                            shape_pairs.insert((prev.min(mi), prev.max(mi)));
                        }
                        *o = Some(mi);
                    }
                }
            }
            if own > 0 {
                each.push(own as f64 / (gw * gh) as f64);
            }
        }
        joined += pairs.difference(&shape_pairs).count();
        let a = cov.iter().filter(|c| **c).count() as f64 / (gw * gh) as f64;
        area.push(a);
        area_max = area_max.max(a);
        core.push(full.iter().filter(|c| **c).count() as f64 / (gw * gh) as f64);
        let first = t < step / 2.0;
        for (c, on) in cells.iter_mut().zip(&cov) {
            c.ever |= *on;
            if first {
                c.on = *on;
                c.run = 1;
                continue;
            }
            if c.on != *on {
                c.toggles += 1;
                if c.switched && c.run < blink_samples {
                    // a short run between two switches: sharp (the blur dropped out) or blurred
                    if *on {
                        c.dropouts += 1;
                    } else {
                        c.flashes += 1;
                    }
                }
                (c.on, c.run, c.switched) = (*on, 1, true);
            } else {
                c.run = c.run.saturating_add(1);
            }
        }
    }
    let n = times.len().max(1) as f64;
    let secs = len.max(1e-6);
    let edge = |i: usize| {
        let (cx, cy) = ((i % gw) as f64 + 0.5, (i / gw) as f64 + 0.5);
        cx < gw as f64 * EDGE_BAND || cx > gw as f64 * (1.0 - EDGE_BAND) || cy < gh as f64 * EDGE_BAND || cy > gh as f64 * (1.0 - EDGE_BAND)
    };
    let covered = cells.iter().filter(|c| c.ever).count();
    let covered_edge = cells.iter().enumerate().filter(|(i, c)| c.ever && edge(*i)).count();
    let dropouts: u32 = cells.iter().map(|c| c.dropouts).sum();
    let edge_dropouts: u32 = cells.iter().enumerate().filter(|(i, _)| edge(*i)).map(|(_, c)| c.dropouts).sum();
    let flashes: u32 = cells.iter().map(|c| c.flashes).sum();
    let total_toggles: u32 = cells.iter().map(|c| c.toggles).sum();
    let r = |v: f64| (v * 10000.0).round() / 10000.0;
    json!({
        "masks": masks.len(),
        // one mask's share of the frame while it shows (inside its 50 % edge), in percent
        "maskAreaPercentMedian": r(100.0 * quantile(&mut each, 0.5)),
        "maskAreaPercentP90": r(100.0 * quantile(&mut each, 0.9)),
        "samples": times.len(),
        "step": step,
        // shares of the frame, in percent
        "areaPercentMean": r(100.0 * area.iter().sum::<f64>() / n),
        "areaPercentMax": r(100.0 * area_max),
        "fullyHiddenPercentMean": r(100.0 * core.iter().sum::<f64>() / n),
        "everCoveredPercent": r(100.0 * covered as f64 / (gw * gh) as f64),
        // flicker
        "blinkSeconds": BLINK,
        "dropoutsPerCoveredCellPerSecond": r(f64::from(dropouts) / covered.max(1) as f64 / secs),
        "edgeBand": EDGE_BAND,
        "edgeDropoutsPerCoveredCellPerSecond": r(f64::from(edge_dropouts) / covered_edge.max(1) as f64 / secs),
        "dropoutAreaPercentPerSecond": r(100.0 * f64::from(dropouts) / (gw * gh) as f64 / secs),
        "flashesPerCoveredCellPerSecond": r(f64::from(flashes) / covered.max(1) as f64 / secs),
        "handoverSeconds": HANDOVER,
        "handoverGaps": gaps.len(),
        "handoverGapSecondsTotal": r(gaps.iter().sum::<usize>() as f64 * step),
        "cellSwitchesPerCoveredCellPerSecond": r(f64::from(total_toggles) / covered.max(1) as f64 / secs),
        "maskSwitches": mask_toggles,
        "maskSwitchesPerMaskPerSecond": r(mask_toggles as f64 / masks.len().max(1) as f64 / secs),
        "masksShownUnderASecond": on_time.iter().filter(|t| **t > 0.0 && **t < 1.0 - 1e-9).count(),
        "sizeJitterPercentMedian": r(100.0 * quantile(&mut size_jitter, 0.5)),
        "sizeJitterPercentP90": r(100.0 * quantile(&mut size_jitter, 0.9)),
        "centreJitterPercentMedian": r(100.0 * quantile(&mut centre_jitter, 0.5)),
        "centreJitterPercentP90": r(100.0 * quantile(&mut centre_jitter, 0.9)),
        // neighbours' masks whose soft edges join while their shapes do not touch (mask pairs × samples)
        "joinedNeighbours": joined,
        "shown": shown,
    })
}

/// The QA's answer with the faces it shows split by whether they touch the frame's edge.
fn qa(s: &mut Session, len: f64, exclude: &Option<Value>) -> Value {
    let seconds: Vec<f64> = (0..).map(|i| f64::from(i) * 0.25).take_while(|t| *t < len).collect();
    let mut p = json!({"seconds": seconds, "profiles": true});
    if let Some(e) = exclude {
        p["exclude"] = e.clone();
    }
    let mut r = s.execute("sequence.checkPrivacy", p).unwrap_or_else(|e| fail(e));
    let shown: Vec<&Value> = r["unmasked"].as_array().into_iter().flatten().filter(|u| u["allowed"].as_bool() != Some(true)).collect();
    let edge = |u: &&&Value| {
        let b = &u["box"];
        let (x, y, w, h) = (b["x"].as_f64().unwrap_or(0.5), b["y"].as_f64().unwrap_or(0.5), b["w"].as_f64().unwrap_or(0.0), b["h"].as_f64().unwrap_or(0.0));
        x < 0.02 || y < 0.02 || x + w > 0.98 || y + h > 0.98
    };
    let at_edge = shown.iter().filter(edge).count();
    let from_pose = shown.iter().filter(|u| u["fromPose"].as_bool() == Some(true)).count();
    r["shownAtEdge"] = json!(at_edge);
    r["shownFromPose"] = json!(from_pose);
    r
}

fn export_frames(s: &mut Session, from: f64, len: f64, every: f64, dir: &Path) {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| fail(e));
    let mut k = 0u32;
    loop {
        let t = f64::from(k) * every;
        if t >= len - 1e-9 {
            break;
        }
        s.set_playhead(Tick::from_seconds_f64(t));
        let path = dir.join(format!("{:07}.jpg", ((t + from) * 1000.0).round() as i64));
        if let Err(e) = s.execute("file.exportFrame", json!({"target": "program", "path": path.to_string_lossy(), "format": "jpeg"})) {
            eprintln!("export {}: {e}", path.display());
        }
        k += 1;
    }
}

fn export_video(s: &mut Session, seconds: f64, path: &Path) {
    let r = s.execute("file.exportMedia", json!({"path": path.to_string_lossy(), "wait": true, "startSeconds": 0.0, "endSeconds": seconds}));
    if let Err(e) = r {
        // a range the command does not take: the whole sequence
        eprintln!("export with a range: {e}; exporting the whole window");
        if let Err(e) = s.execute("file.exportMedia", json!({"path": path.to_string_lossy(), "wait": true})) {
            eprintln!("export {}: {e}", path.display());
        }
    }
}

/// Three videos side by side, each a third of the width.
fn compose(out: &Path, inputs: &[String]) {
    let mut s = Session::default();
    let mut items = Vec::new();
    for i in inputs {
        let r = s.execute("file.import", json!({"paths": [i]})).unwrap_or_else(|e| fail(e));
        items.push(r["items"][0].as_u64().map(ItemId).unwrap_or_else(|| fail(format!("import {i}: {r}"))));
    }
    let Some(info) = items.first().and_then(|i| s.project.item(*i)).and_then(|i| i.as_media()).and_then(|m| m.info.video.clone()) else {
        fail("the first video has no picture")
    };
    let n = items.len() as u32;
    let (pw, ph) = (info.width / 2, info.height / 2);
    let mut p = (*s.project).clone();
    let seq =
        p.new_sequence("Side by side", SequenceSettings { width: pw * n, height: ph, frame_rate: info.frame_rate, ..Default::default() }, n as usize, 0, None);
    let dur = items.iter().filter_map(|i| p.item(*i)).filter_map(|i| i.as_media()).map(|m| m.duration()).min().unwrap_or(Tick::ZERO);
    let mut clips = Vec::new();
    for (k, item) in items.iter().enumerate() {
        let Some(mut v) = p.make_track_item(*item, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, dur), info.frame_rate) else { continue };
        v.scale_to_frame = false;
        if let Some(sc) = v.effects.iter_mut().find(|e| e.effect == "motion").and_then(|m| m.param_mut("scale")) {
            sc.value = ParamValue::Float(50.0);
        }
        clips.push((v.id, k));
        if let Some(t) = p.sequence_mut(seq).and_then(|q| q.video_tracks.get_mut(k)) {
            t.items.push(v);
        }
    }
    s.project = std::sync::Arc::new(p);
    s.state.active_sequence = Some(seq);
    s.state.open_sequences = vec![seq];
    for (clip, k) in clips {
        let at = [f64::from(pw) * (k as f64 + 0.5), f64::from(ph) / 2.0];
        if let Err(e) = s.execute("effects.setParam", json!({"clip": clip.0, "effect": "motion", "param": "position", "value": at})) {
            fail(format!("placing video {k}: {e}"));
        }
    }
    if let Err(e) = s.execute("file.exportMedia", json!({"path": out.to_string_lossy(), "wait": true})) {
        fail(format!("export: {e}"));
    }
}

/// The exported frame of `dir` nearest media time `s` (frames are named by media milliseconds).
fn frame_near(dir: &Path, s: f64) -> Option<PathBuf> {
    let want = (s * 1000.0).round() as i64;
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter_map(|p| {
            let ms: i64 = p.file_stem()?.to_str()?.parse().ok()?;
            (p.extension().is_some_and(|x| x == "jpg")).then_some((ms, p))
        })
        .min_by_key(|(ms, _)| (ms - want).abs())
        .map(|(_, p)| p)
}

/// Side-by-side crops of the same places in several stages' frames: each face the last stage
/// hides (at its best moment) and each face its QA still shows, cut from every stage's frame
/// around the face (three faces wide, square), scaled to 360 px and put side by side
/// (`<out>/<n>_<what>_<media ms>.jpg`, the stages left to right in the order given).
fn crops(out: &Path, clip_dir: &Path, stages: &[String], max: usize) {
    let Some(last) = stages.last() else { fail("--crops needs stages") };
    let v: Value = std::fs::read(clip_dir.join(format!("{last}.json")))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_else(|| fail(format!("cannot read {last}.json")));
    // (what, media seconds, box)
    let mut places: Vec<(String, f64, [f64; 4])> = Vec::new();
    let bx = |b: &Value| [b["x"].as_f64().unwrap_or(0.0), b["y"].as_f64().unwrap_or(0.0), b["w"].as_f64().unwrap_or(0.0), b["h"].as_f64().unwrap_or(0.0)];
    for t in v["outcome"]["clips"].as_array().into_iter().flatten().flat_map(|c| c["tracks"].as_array().into_iter().flatten()) {
        if t["hidden"].as_bool() == Some(true) && t["kind"] == "face" {
            let what = if t["fromPose"].as_u64().unwrap_or(0) > 0 { "hidden-pose" } else { "hidden" };
            places.push((what.into(), t["box"]["seconds"].as_f64().unwrap_or(0.0), bx(&t["box"])));
        }
    }
    for u in v["qa"]["unmasked"].as_array().into_iter().flatten().filter(|u| u["allowed"].as_bool() != Some(true)) {
        places.push(("qa-shown".into(), u["media"].as_f64().unwrap_or(0.0), bx(&u["box"])));
    }
    std::fs::create_dir_all(out).unwrap_or_else(|e| fail(e));
    let side = 360u32;
    let gap = 8u32;
    let mut written = 0usize;
    for (k, (what, s, b)) in places.iter().enumerate().take(max) {
        let mut tiles: Vec<image::RgbImage> = Vec::new();
        for st in stages {
            let Some(p) = frame_near(&clip_dir.join(st), *s) else { continue };
            let Ok(img) = image::open(&p) else { continue };
            let img = img.to_rgb8();
            let (w, h) = (f64::from(img.width()), f64::from(img.height()));
            let r = (b[2] * w).max(b[3] * h) * 1.5;
            let (cx, cy) = ((b[0] + b[2] / 2.0) * w, (b[1] + b[3] / 2.0) * h);
            let r = r.clamp(16.0, w.min(h) / 2.0);
            let x0 = (cx - r).clamp(0.0, (w - 2.0 * r).max(0.0));
            let y0 = (cy - r).clamp(0.0, (h - 2.0 * r).max(0.0));
            let c = image::imageops::crop_imm(&img, x0 as u32, y0 as u32, (2.0 * r) as u32, (2.0 * r) as u32).to_image();
            tiles.push(image::imageops::resize(&c, side, side, image::imageops::FilterType::Triangle));
        }
        if tiles.len() != stages.len() {
            continue;
        }
        let n = tiles.len() as u32;
        let mut sheet = image::RgbImage::from_pixel(n * side + (n - 1) * gap, side, image::Rgb([255, 255, 255]));
        for (i, t) in tiles.iter().enumerate() {
            image::imageops::replace(&mut sheet, t, i64::from(i as u32 * (side + gap)), 0);
        }
        let name = format!("{k:03}_{what}_{:07}.jpg", (s * 1000.0).round() as i64);
        if let Err(e) = sheet.save(out.join(&name)) {
            eprintln!("{name}: {e}");
        } else {
            written += 1;
        }
    }
    println!("{written} crops in {}", out.display());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--compose") {
        let (Some(out), Some(inputs)) = (args.get(2), args.get(3..)) else { fail("--compose <out.mp4> <a.mp4> <b.mp4> <c.mp4>") };
        compose(Path::new(out), inputs);
        return;
    }
    if args.get(1).map(String::as_str) == Some("--crops") {
        let (Some(out), Some(dir), Some(stages)) = (args.get(2), args.get(3), args.get(4..)) else {
            fail("--crops <out dir> <clip dir> <stage>... (the last stage's hidden faces and QA)")
        };
        crops(Path::new(out), Path::new(dir), stages, 200);
        return;
    }
    let [_, video, from, to, out, stage, ..] = args.as_slice() else {
        fail("usage: redact_stage <video> <from s> <to s> <out dir> <stage> [--no-judge] [--exclude x,y,w,h] [--every s] [--video s] [--time-only]");
    };
    let video = PathBuf::from(video);
    let (from, to): (f64, f64) = (from.parse().unwrap_or_else(|_| fail("from")), to.parse().unwrap_or_else(|_| fail("to")));
    if !(from.is_finite() && to.is_finite() && from >= 0.0 && to > from) {
        fail("from and to: 0 ≤ from < to");
    }
    let opt = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1));
    let every: f64 = opt("--every").and_then(|v| v.parse().ok()).filter(|v: &f64| *v >= 0.02).unwrap_or(0.1);
    let video_s: f64 = opt("--video").and_then(|v| v.parse().ok()).filter(|v: &f64| *v > 0.0).unwrap_or(10.0);
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap_or_else(|e| fail(e));
    let len = to - from;
    let (mut s, seq) = session(&video, from, to);
    let exclude = opt("--exclude").map(|v| {
        let n: Vec<f64> = v.split(',').filter_map(|x| x.parse().ok()).collect();
        let clip = s.state.selection.first().map(|c| c.0);
        json!({"clip": clip, "x": n.first(), "y": n.get(1), "w": n.get(2), "h": n.get(3)})
    });
    if args.iter().any(|a| a == "--time-only") {
        let t = std::time::Instant::now();
        let mut p = json!({"wait": true, "profiles": true, "dryRun": true, "verify": false, "kinds": "faces"});
        if let Some(e) = &exclude {
            p["exclude"] = e.clone();
        }
        let r = s.execute("clip.autoRedact", p).unwrap_or_else(|e| fail(e));
        let frames = r["frames"].as_u64().unwrap_or(0);
        let analysis = r["redact"]["seconds"]["analysis"].as_f64().unwrap_or(0.0);
        let load = std::process::Command::new("sysctl")
            .args(["-n", "vm.loadavg"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        let v = json!({"stage": stage, "frames": frames, "analysisSeconds": analysis, "msPerAnalysedFrame": (analysis * 1000.0 / frames.max(1) as f64).round(), "wall": t.elapsed().as_secs_f64(), "load": load});
        println!("{v}");
        let _ =
            std::fs::write(out.join(format!("{stage}-time.json")), serde_json::to_vec_pretty(&json!({"time": v, "outcome": r["redact"]})).unwrap_or_default());
        return;
    }
    if !args.iter().any(|a| a == "--no-judge") {
        s.redacts.judge = Some(filmcraft_autocut::privacy::engine_judge());
    }
    let t = std::time::Instant::now();
    let outcome = if stage == "original" {
        Value::Null
    } else {
        let mut p = json!({"wait": true, "profiles": true, "kinds": "faces"});
        if let Some(e) = &exclude {
            p["exclude"] = e.clone();
        }
        s.execute("clip.autoRedact", p).unwrap_or_else(|e| fail(e))["redact"].clone()
    };
    let redact_s = t.elapsed().as_secs_f64();
    let mut metrics = measure(&s, seq, len, every);
    // the masks shown at each sample, in a file of their own
    if let Some(shown) = metrics.get_mut("shown").map(Value::take) {
        let _ = std::fs::write(out.join(format!("{stage}-masks.json")), serde_json::to_vec(&shown).unwrap_or_default());
    }
    if let Some(m) = metrics.as_object_mut() {
        m.remove("shown");
    }
    let qa = qa(&mut s, len, &exclude);
    export_frames(&mut s, from, len, every, &out.join(stage));
    export_video(&mut s, video_s.min(len), &out.join(format!("{stage}.mp4")));
    let count = |k: &str| outcome["clips"].as_array().map_or(0, |c| c.iter().map(|c| c[k].as_u64().unwrap_or(0)).sum::<u64>());
    let skipped = |k: &str| outcome["clips"].as_array().map_or(0, |c| c.iter().map(|c| c["skipped"][k].as_u64().unwrap_or(0)).sum::<u64>());
    let summary = json!({
        "stage": stage,
        "video": video.to_string_lossy(),
        "window": [from, to],
        "facesHidden": count("faces"),
        "fromPose": count("facesFromPose"),
        "onlyFromPose": count("facesOnlyFromPose"),
        "robotsLeftSharp": skipped("notAPerson"),
        "qaShown": qa["shown"],
        "qaUnmasked": qa["unmasked"].as_array().map_or(0, Vec::len),
        "qaShownAtEdge": qa["shownAtEdge"],
        "qaShownFromPose": qa["shownFromPose"],
        "masks": metrics,
        "seconds": {"redaction": redact_s, "analysis": outcome["seconds"]["analysis"], "check": outcome["seconds"]["check"]},
    });
    let full = json!({"summary": summary, "outcome": outcome, "qa": qa});
    let _ = std::fs::write(out.join(format!("{stage}.json")), serde_json::to_vec_pretty(&full).unwrap_or_default());
    println!("{}", serde_json::to_string_pretty(&summary).unwrap_or_default());
}
