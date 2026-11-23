//! `sequence.motionStrips` (PX1.6): motion filmstrips. For each animated title (its entrance and
//! its exit), each transition and each motion-graphics overlay (a clip of a picture with alpha,
//! as `motion.render` and `composition.import` bring in), seven frames of the sequence as it
//! renders, in one labelled sheet per item, after the capture points of impeccable's motion
//! reviewer (Apache-2.0; read on GitHub, written fresh): just before, the first frame, three
//! between, the last frame and settled after. `filmcraft-cli frames --motion` runs it.
//!
//! Beside the sheets every frame of the move (at most [`MAX_DENSE`], evenly spread when it is
//! longer) is rendered small, reduced to the grey grid of `qa.motionEnergy` (96×54) and read by
//! detectors:
//!
//! | code | warns when | starting value |
//! |---|---|---|
//! | `jump` | the step into the move's first frame, or out of its last, is far larger than the move's own steps (a pop instead of an ease) | at least 0.02 of full scale and 3 × the median step; not where the picture under it cuts |
//! | `midStep` | one frame-to-frame change inside the move is far larger than its neighbours (a hitch) | at least 0.01 and 3 × the median of the two steps on each side; only when every frame was read, not from or to a frame of one flat colour (a dip's black) |
//! | `dissolveDip` | a dissolve's picture gets darker in the middle than both its ends | mean grey 0.04 under the darker end |
//! | `fadeAfterShot` | a title's or overlay's exit is still visible when another clip cuts under it | what the layer adds to the frame at the cut is at least a tenth of what it adds when its exit starts |
//! | `entranceReplay` | an entrance starts at an edit and brings in what was on screen just before it (the same title entering again after a cut) | the frame before the edit within half the layer's own share of the settled frame, a frame of the move at least a quarter of it away |
//!
//! What a layer adds to a frame is the frame against the same frame rendered without the
//! layer's clip, so a title's own pixels are told from the footage under it.
//!
//! Sequences render on the GPU frame renderer where the host has one, on the CPU reference
//! renderer otherwise (tests and headless hosts without a GPU), without captions. At most
//! [`MAX_ITEMS`] items are read; the rest are listed in `skipped`. It is a job (`jobs.list` shows
//! its progress and, once finished, its result under `motionStrips`); `wait: true` runs it in the
//! call. Every finding is a warning (decision D7); the numbers are starting values to tune.
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_project::{ClipId, ItemId, Project};
use filmcraft_time::{FrameRate, Tick};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, has_seq};
use crate::craft_check::Renderer;
use crate::motion_energy::{bounded, box_reduce, finding, grey_of, grid_for, mean_diff, r3};
use crate::{EngineError, Result, Session};

const CMD: &str = "sequence.motionStrips";
/// Most items one call reads, and most `items` it takes.
pub const MAX_ITEMS: usize = 32;
const MAX_SELECTORS: usize = 200;
/// Most frames of one move analysed (a longer move is read every n-th frame).
pub const MAX_DENSE: usize = 64;
/// How long after a move its settled frame is, seconds.
const SETTLE_SECONDS: f64 = 0.25;
/// The sheet tiles' long side, and the analysis render's, pixels.
const SHEET_SIDE: f64 = 320.0;
const ANALYSIS_SIDE: f64 = 192.0;
/// `jump`: the step at an end is at least this, and this many times the move's median step.
pub const JUMP_MIN: f32 = 0.02;
pub const JUMP_FACTOR: f32 = 3.0;
/// `midStep`: a step at least this, and this many times its neighbours' median.
pub const STEP_MIN: f32 = 0.01;
pub const STEP_FACTOR: f32 = 3.0;
/// `dissolveDip`: mean grey (share of full scale) under the darker end.
pub const DIP_MIN: f64 = 0.04;
/// A layer that adds less than this to a frame (mean grey change) is not seen.
pub const LAYER_MIN: f32 = 0.002;
/// `fadeAfterShot`: the share of the layer's look at its exit's start still there at the cut.
pub const FADE_SHARE: f32 = 0.1;
/// `entranceReplay`: the frame before the edit is within this share of the layer's look from the
/// settled frame, and a frame of the move (first to three quarters) at least this share away.
pub const REPLAY_SAME: f32 = 0.5;
pub const REPLAY_MOVE: f32 = 0.25;
/// The transitions that mix two pictures without meaning to darken (a dip to black does).
const DISSOLVES: &[&str] = &["cross_dissolve", "film_dissolve", "blur_dissolve", "blend_dissolve", "morph_cut"];
/// Most finished outcomes kept for `jobs.list`.
const MAX_OUTCOMES: usize = 64;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Motion Filmstrips",
        menu: &[],
        shortcut: None,
        params: r#"{"sequence":id? (default the active one),"items":[clipId|{"clip":id,"edge":"in"|"out"?}|{"transition":id}]? (default every animated title, transition and overlay, at most 32),"out":path (a new or empty folder for the sheets),"scale":0.05..1? (the tiles' share of the frame; default a long side of 320 px),"wait":bool=false}"#,
        enabled: has_seq,
        run,
        journal: false,
    }]
}

/// The finished strips' results, by job, for `jobs.list`.
#[derive(Default)]
pub struct StripJobs {
    outcomes: HashMap<u64, Arc<Mutex<Option<Value>>>>,
}

/// A finished `sequence.motionStrips` job's result (`null` while it runs or after a failure).
pub fn outcome_json(s: &Session, job: u64) -> Option<Value> {
    let o = s.motion_strips.outcomes.get(&job)?;
    Some(o.lock().unwrap_or_else(PoisonError::into_inner).clone().unwrap_or(Value::Null))
}

/// One thing to filmstrip.
#[derive(Clone, Debug)]
struct Item {
    /// `title`, `transition` or `overlay`.
    kind: &'static str,
    clip: Option<u64>,
    transition: Option<u64>,
    /// `in` or `out` for a title's entrance or exit; a transition into or out of a title or
    /// overlay says which here too.
    edge: Option<&'static str>,
    /// The transition's effect (its Creator preset's base).
    effect: Option<String>,
    what: String,
    /// The move, timeline ticks: `[start, end)`.
    start: Tick,
    end: Tick,
}

impl Item {
    fn json(&self) -> Value {
        json!({"kind": self.kind, "clip": self.clip, "transition": self.transition, "edge": self.edge, "effect": self.effect, "what": self.what, "moveStart": self.start.0, "moveEnd": self.end.0})
    }
    fn file(&self, n: usize) -> String {
        let id = self.clip.or(self.transition).unwrap_or(0);
        match self.edge {
            Some(e) => format!("{n:02}-{}-{id}-{e}.png", self.kind),
            None => format!("{n:02}-{}-{id}.png", self.kind),
        }
    }
}

/// Every animated title (entrance and exit), transition and overlay of `q`, in time order.
fn candidates(project: &Project, q: &filmcraft_project::Sequence) -> Vec<Item> {
    let mut out: Vec<Item> = crate::motion_check::windows(project, q)
        .into_iter()
        .map(|w| Item { kind: w.role, clip: Some(w.clip), transition: None, edge: w.edge, effect: None, what: w.what, start: w.start, end: w.end })
        .collect();
    let roles: HashMap<u64, &'static str> = out.iter().filter_map(|i| i.clip.map(|c| (c, i.kind))).collect();
    for track in q.video_tracks.iter().filter(|t| t.enabled) {
        for tr in &track.transitions {
            let base = filmcraft_project::creator_transitions::base_id(&tr.effect.effect).to_string();
            let label = filmcraft_project::find_effect(&tr.effect.effect).map_or_else(|| tr.effect.effect.clone(), |d| d.name.to_string());
            let layer = |c: Option<ClipId>| c.and_then(|c| roles.get(&c.0)).is_some();
            let edge = match (tr.from, tr.to) {
                (None, Some(_)) if layer(tr.to) => Some("in"),
                (Some(_), None) if layer(tr.from) => Some("out"),
                _ => None,
            };
            out.push(Item {
                kind: "transition",
                clip: if edge == Some("out") { tr.from.map(|c| c.0) } else { tr.to.map(|c| c.0) },
                transition: Some(tr.id.0),
                edge,
                effect: Some(base),
                what: format!("{label} transition"),
                start: tr.start,
                end: tr.end(),
            });
        }
    }
    out.sort_by_key(|i| (i.start, i.end));
    out
}

/// The items `items` names, from `all`.
fn chosen(all: &[Item], v: &Value) -> Result<Vec<Item>> {
    let list = v
        .as_array()
        .filter(|a| !a.is_empty() && a.len() <= MAX_SELECTORS)
        .ok_or_else(|| bad(CMD, format!("`items` is a list of 1 to {MAX_SELECTORS} clip ids, {{clip, edge?}} or {{transition}}")))?;
    let mut out: Vec<Item> = Vec::new();
    for sel in list {
        let (clip, edge, transition) = match sel {
            Value::Number(n) => (Some(n.as_u64().ok_or_else(|| bad(CMD, "a clip id is a whole number"))?), None, None),
            Value::Object(o) => {
                if let Some(k) = o.keys().find(|k| !matches!(k.as_str(), "clip" | "edge" | "transition")) {
                    return Err(bad(CMD, format!("an item has no `{}`: clip, edge or transition", k.chars().take(64).collect::<String>())));
                }
                let id = |k: &str| match o.get(k) {
                    None | Some(Value::Null) => Ok(None),
                    Some(v) => v.as_u64().map(Some).ok_or_else(|| bad(CMD, format!("`{k}` is an id"))),
                };
                let edge = match o.get("edge") {
                    None | Some(Value::Null) => None,
                    Some(Value::String(e)) if e == "in" || e == "out" => Some(if e == "in" { "in" } else { "out" }),
                    Some(_) => return Err(bad(CMD, "`edge` is \"in\" or \"out\"")),
                };
                (id("clip")?, edge, id("transition")?)
            }
            _ => return Err(bad(CMD, "an item is a clip id, {clip, edge?} or {transition}")),
        };
        let found: Vec<&Item> = match (clip, transition) {
            (Some(c), None) => all.iter().filter(|i| i.transition.is_none() && i.clip == Some(c) && edge.is_none_or(|e| i.edge == Some(e))).collect(),
            (None, Some(t)) => all.iter().filter(|i| i.transition == Some(t)).collect(),
            _ => return Err(bad(CMD, "an item names a clip or a transition")),
        };
        if found.is_empty() {
            return Err(bad(
                CMD,
                match (clip, transition) {
                    (Some(c), _) => format!("clip {c} has no animated entrance, exit or overlay motion on a visible track"),
                    (_, t) => format!("no transition {} on a visible video track", t.unwrap_or(0)),
                },
            ));
        }
        for i in found {
            if !out.iter().any(|o| o.clip == i.clip && o.transition == i.transition && o.edge == i.edge) {
                out.push(i.clone());
            }
        }
    }
    out.sort_by_key(|i| (i.start, i.end));
    Ok(out)
}

/// The seven frames of an item: (role, time).
fn seven(item: &Item, rate: FrameRate, end: Tick) -> Vec<(&'static str, Tick)> {
    let f = rate.frame_duration().0.max(1);
    let last_frame = end.0.saturating_sub(1).max(0);
    let clamp = |t: i64| Tick(rate.snap(Tick(t.clamp(0, last_frame))).0.clamp(0, last_frame));
    let (a, b) = (item.start.0, item.end.0.max(item.start.0.saturating_add(f)));
    let span = b.saturating_sub(a);
    let at = |k: i64| clamp(a.saturating_add(span / 4 * k));
    let settled = if item.edge == Some("out") { b } else { b.saturating_add(Tick::from_seconds_f64(SETTLE_SECONDS).0) };
    vec![
        ("before", clamp(a.saturating_sub(f))),
        ("first", clamp(a)),
        ("quarter", at(1)),
        ("middle", at(2)),
        ("threeQuarters", at(3)),
        ("last", clamp(b.saturating_sub(f))),
        ("settled", clamp(settled)),
    ]
}

/// The analysis frames of an item.
struct Dense {
    /// From the frame before the move (when there is one) to the first frame after it (when the
    /// sequence goes on), every `every`-th frame in between so at most [`MAX_DENSE`] are read.
    times: Vec<Tick>,
    every: i64,
    before: bool,
    after: bool,
}

fn dense(item: &Item, rate: FrameRate, end: Tick) -> Dense {
    let first = rate.frame_at(item.start).max(0);
    let final_frame = rate.frame_at(Tick(end.0.saturating_sub(1))).max(0);
    let last = rate.frame_at(Tick(item.end.0.saturating_sub(1))).clamp(first, final_frame.max(first));
    let frames = last.saturating_sub(first).saturating_add(1);
    let every = (frames.saturating_add(MAX_DENSE as i64 - 1) / MAX_DENSE as i64).max(1);
    let (before, after) = (first > 0, last < final_frame);
    let mut times = Vec::new();
    let mut n = if before { first - 1 } else { first };
    let stop = if after { last + 1 } else { last };
    while n <= stop && times.len() < MAX_DENSE + 4 {
        times.push(rate.tick_of(n));
        n = if n < first || n >= last { n.saturating_add(1) } else { n.saturating_add(every).min(last) };
    }
    Dense { times, every, before, after }
}

/// A tile's time: hundredths of a second (neighbouring frames differ), minutes past a minute.
fn label(seconds: f64) -> String {
    if seconds.is_finite() && (0.0..60.0).contains(&seconds) { format!("{seconds:.2}s") } else { crate::contact_sheet::stamp(seconds) }
}

fn median(mut v: Vec<f32>) -> f32 {
    v.retain(|x| x.is_finite());
    v.sort_by(f32::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

/// The deepest point of a dissolve's mean grey `lum` (its ends first and last) under its darker
/// end, when it is more than [`DIP_MIN`] under it: (index, depth).
fn dip(lum: &[f64]) -> Option<(usize, f64)> {
    let (first, last) = (lum.first()?, lum.last()?);
    let ends = first.min(*last);
    let (i, low) = lum.iter().enumerate().skip(1).take(lum.len().saturating_sub(2)).fold((0, f64::INFINITY), |m, (i, v)| if *v < m.1 { (i, *v) } else { m });
    let depth = ends - low;
    (low.is_finite() && depth > DIP_MIN).then_some((i, depth))
}

/// A frame of one flat colour (black, white, a dip's colour): its grey spread is under
/// `qa.motionEnergy`'s.
fn flat_frame(g: &[u8]) -> bool {
    let mu = mean_grey(g) * 255.0;
    let n = g.len().max(1) as f64;
    let spread = (g.iter().map(|v| (f64::from(*v) - mu).powi(2)).sum::<f64>() / n).sqrt() / 255.0;
    spread < crate::motion_energy::EMPTY_SPREAD
}

fn mean_grey(g: &[u8]) -> f64 {
    if g.is_empty() {
        return 0.0;
    }
    g.iter().map(|v| f64::from(*v)).sum::<f64>() / g.len() as f64 / 255.0
}

/// What the background run needs.
struct Work {
    project: Arc<Project>,
    seq: ItemId,
    items: Vec<Item>,
    skipped_items: usize,
    out: String,
    scale: f32,
    rate: FrameRate,
    end: Tick,
    size: (u32, u32),
    /// Times where clips on visible video tracks start or end, and which.
    cuts: BTreeMap<i64, Vec<u64>>,
    services: Arc<dyn crate::Services>,
    media: Arc<crate::media_pool::MediaPool>,
}

/// Renders frames of one project and counts them.
struct Frames<'a> {
    renderer: &'a mut Renderer,
    project: &'a Project,
    seq: ItemId,
    provider: &'a crate::media_pool::PoolProvider,
    progress: &'a filmcraft_export::Progress,
}

impl Frames<'_> {
    fn image(&mut self, t: Tick, scale: f32) -> filmcraft_render::Image {
        let opts = filmcraft_render::RenderOptions { scale, ..Default::default() };
        let img = self.renderer.render(self.project, self.seq, t, opts, self.provider);
        self.progress.done.fetch_add(1, Ordering::Relaxed);
        img
    }
}

fn grey(img: &filmcraft_render::Image, grid: (usize, usize)) -> Option<Vec<u8>> {
    box_reduce(img.w as u32, img.h as u32, &img.over_black_rgba8(), grid.0, grid.1).ok().map(|small| grey_of(&small))
}

/// `project` with clip `clip` of sequence `seq` turned off.
fn without(project: &Project, seq: ItemId, clip: u64) -> Project {
    let mut p = project.clone();
    if let Some(it) = p.sequence_mut(seq).and_then(|q| q.find_item_mut(ClipId(clip)).map(|(_, it)| it)) {
        it.enabled = false;
    }
    p
}

/// Draw and read every item: the result.
fn strips(w: Work, progress: &filmcraft_export::Progress) -> std::result::Result<Value, String> {
    let provider = w.media.provider(w.project.clone(), w.services.clone());
    let mut renderer = Renderer::new();
    let long = f64::from(w.size.0.max(w.size.1).max(1));
    let small = (ANALYSIS_SIDE / long).min(1.0) as f32;
    let grid = grid_for(w.size.0, w.size.1);
    let f = w.rate.frame_duration().0.max(1);
    let fps = w.rate.as_f64();
    let mut findings = Vec::new();
    let mut sheets = Vec::new();
    let mut sparse = 0usize;
    for (n, item) in w.items.iter().enumerate() {
        if progress.cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        *progress.status.lock().unwrap_or_else(PoisonError::into_inner) = format!("Item {} of {}: {}", n + 1, w.items.len(), item.what);
        let mut fr = Frames { renderer: &mut renderer, project: &w.project, seq: w.seq, provider: &provider, progress };
        // ---- the sheet
        let frames = seven(item, w.rate, w.end);
        let images: Vec<(filmcraft_render::Image, String)> =
            frames.iter().enumerate().map(|(i, (_, t))| (fr.image(*t, w.scale), format!("{} {}", i + 1, label(t.seconds())))).collect();
        let greys: Vec<Option<Vec<u8>>> = images.iter().map(|(img, _)| grey(img, grid)).collect();
        let path = std::path::Path::new(&w.out).join(item.file(n + 1)).display().to_string();
        let (bytes, sw, sh) = crate::contact_sheet::encode_rows(CMD, &[images], false).map_err(|e| e.to_string())?;
        w.services.write_file_new(&path, &bytes).map_err(|e| format!("{path}: {e}"))?;
        // ---- every frame of the move, small
        let dn = dense(item, w.rate, w.end);
        let k = dn.every;
        if k > 1 {
            sparse += 1;
        }
        let g: Vec<(Tick, Vec<u8>)> = dn.times.iter().filter_map(|t| grey(&fr.image(*t, small), grid).map(|g| (*t, g))).collect();
        let whole = g.len() == dn.times.len();
        let steps: Vec<(Tick, f32)> = g
            .windows(2)
            .filter_map(|p| match p {
                [(_, a), (t, b)] => Some((*t, mean_diff(a, b))),
                _ => None,
            })
            .collect();
        let base = item.json();
        let with = |extra: Value| {
            let mut v = base.clone();
            if let (Some(o), Value::Object(x)) = (v.as_object_mut(), extra) {
                o.extend(x);
            }
            v
        };
        // another clip starts or ends there: the picture under the item cuts
        let cut_at = |t: Tick| w.cuts.get(&t.0).is_some_and(|clips| clips.iter().any(|c| Some(*c) != item.clip));
        // the move's own steps: between its first frame and its last
        let (s0, s1) = (usize::from(dn.before), steps.len().saturating_sub(usize::from(dn.after)));
        let inner_steps: Vec<(Tick, f32)> = steps.get(s0..s1.max(s0)).map(<[_]>::to_vec).unwrap_or_default();
        let inner: Vec<f32> = inner_steps.iter().map(|s| s.1).collect();
        let typical = median(inner.clone());
        if k == 1 && whole && !inner.is_empty() {
            let ends = [("start", dn.before.then(|| steps.first()).flatten()), ("end", dn.after.then(|| steps.last()).flatten())];
            for (edge, step) in ends {
                let Some(&(t, d)) = step else { continue };
                let at_cut = if edge == "start" { cut_at(item.start) } else { cut_at(item.end) };
                if !at_cut && d >= JUMP_MIN && d >= JUMP_FACTOR * typical.max(1e-4) {
                    findings.push(finding(
                        "jump",
                        Tick(t.0.saturating_sub(f)),
                        Some(t),
                        format!(
                            "the {} {} with a jump ({d:.3} of full scale in one frame, {:.1} × its usual step); {}",
                            item.what,
                            if edge == "start" { "starts" } else { "ends" },
                            d / typical.max(1e-4),
                            if edge == "start" {
                                "start it from nothing (opacity 0 or off frame) and ease it in"
                            } else {
                                "ease it into its last frame (Quart Out or a spring at 0.9 or more)"
                            }
                        ),
                        with(json!({"edge": edge, "step": r3(f64::from(d)), "median": r3(f64::from(typical))})),
                    ));
                }
            }
            // a hitch inside the move
            for i in 1..inner.len().saturating_sub(1) {
                let Some(&d) = inner.get(i) else { continue };
                let around: Vec<f32> = (i.saturating_sub(2)..(i + 3).min(inner.len())).filter(|j| *j != i).filter_map(|j| inner.get(j).copied()).collect();
                let Some(&(t, _)) = inner_steps.get(i) else { continue };
                // a step from or to a flat frame (the black in the middle of a dip) is the fade's own
                // curve, not a hitch
                let flat = [s0 + i, s0 + i + 1].iter().any(|k| g.get(*k).is_none_or(|(_, x)| flat_frame(x)));
                if around.len() < 2 || cut_at(t) || flat {
                    continue;
                }
                let local = median(around);
                if d >= STEP_MIN && d >= STEP_FACTOR * local.max(1e-4) {
                    findings.push(finding(
                        "midStep",
                        Tick(t.0.saturating_sub(f)),
                        Some(t),
                        format!(
                            "the {} hitches at {:.2} s: one frame changes {d:.3}, {:.1} × the frames around it; look for a keyframe there, a hold, or two moves overlapping, and smooth the curve",
                            item.what,
                            t.seconds(),
                            d / local.max(1e-4)
                        ),
                        with(json!({"step": r3(f64::from(d)), "around": r3(f64::from(local))})),
                    ));
                }
            }
        }
        // ---- a dissolve that darkens
        if item.kind == "transition" && item.effect.as_deref().is_some_and(|e| DISSOLVES.contains(&e)) {
            let lum: Vec<f64> = g.iter().map(|(_, g)| mean_grey(g)).collect();
            if let Some((low_i, depth)) = dip(&lum) {
                let ends = lum.first().copied().unwrap_or(0.0).min(lum.last().copied().unwrap_or(0.0));
                let low = ends - depth;
                let t = g.get(low_i).map_or(item.start, |x| x.0);
                findings.push(finding(
                    "dissolveDip",
                    t,
                    Some(item.end),
                    format!(
                        "the {} darkens in the middle ({:.0} % of full scale under its darker end at {:.2} s); use a dissolve that mixes in linear light, shorten it, or a dip to black if a dip is meant",
                        item.what,
                        depth * 100.0,
                        t.seconds()
                    ),
                    with(json!({"dip": r3(depth), "ends": r3(ends), "lowest": r3(low)})),
                ));
            }
        }
        // what the layer adds to the frame at `t`: the frame against it without the layer's clip
        let adds = |fr: &mut Frames, clip: u64, t: Tick| -> Option<f32> {
            let bare = Arc::new(without(&w.project, w.seq, clip));
            let bare_provider = w.media.provider(bare.clone(), w.services.clone());
            let shown = grey(&fr.image(t, small), grid)?;
            let opts = filmcraft_render::RenderOptions { scale: small, ..Default::default() };
            let gone = grey(&fr.renderer.render(&bare, w.seq, t, opts, &bare_provider), grid);
            fr.progress.done.fetch_add(1, Ordering::Relaxed);
            Some(mean_diff(&shown, &gone?))
        };
        // ---- an exit still on screen when the picture under it cuts
        if item.edge == Some("out")
            && let Some(clip) = item.clip
            && let Some(c) = w
                .cuts
                .range(item.start.0.saturating_add(f)..item.end.0.saturating_sub(f).max(item.start.0.saturating_add(f)))
                .find(|(_, clips)| clips.iter().any(|x| *x != clip))
                .map(|(t, _)| *t)
            && let (Some(full), Some(left)) = (adds(&mut fr, clip, item.start), adds(&mut fr, clip, Tick(c)))
            && full >= LAYER_MIN
            && left >= LAYER_MIN.max(FADE_SHARE * full)
        {
            let more = Tick(item.end.0.saturating_sub(c)).seconds();
            findings.push(finding(
                "fadeAfterShot",
                Tick(c),
                Some(item.end),
                format!(
                    "the {} is still on screen when the picture under it cuts at {:.2} s, and goes on fading for {more:.2} s over the next shot; end the exit by the cut, or move the cut",
                    item.what,
                    Tick(c).seconds()
                ),
                with(json!({"cut": c, "atCut": r3(f64::from(left)), "atExitStart": r3(f64::from(full)), "after": r3(more)})),
            ));
        }
        // ---- an entrance of what was already there
        if item.edge == Some("in")
            && item.start.0 > 0
            && cut_at(item.start)
            && let Some(clip) = item.clip
            && let (Some(b), Some(st)) = (greys.first().cloned().flatten(), greys.get(6).cloned().flatten())
            && let Some(look) = frames.get(6).and_then(|(_, t)| adds(&mut fr, clip, *t))
            && look >= LAYER_MIN
        {
            // how far the move's frames (first to three quarters) are from where it settles
            let moved = greys.get(1..5).into_iter().flatten().flatten().map(|m| mean_diff(m, &st)).fold(0.0f32, f32::max);
            let same = mean_diff(&b, &st);
            if same < REPLAY_SAME * look && moved >= REPLAY_MOVE * look {
                findings.push(finding(
                    "entranceReplay",
                    item.start,
                    Some(item.end),
                    format!(
                        "the {} plays again after the edit at {:.2} s: the frame before the edit already shows what it brings in; take the entrance off this clip (or join the two clips) so it enters once",
                        item.what,
                        item.start.seconds()
                    ),
                    with(json!({"beforeToSettled": r3(f64::from(same)), "moveToSettled": r3(f64::from(moved)), "layer": r3(f64::from(look))})),
                ));
            }
        }
        sheets.push(json!({
            "item": base,
            "path": path,
            "width": sw,
            "height": sh,
            "frames": frames.iter().map(|(role, t)| json!({"role": role, "time": t.0, "seconds": r3(t.seconds())})).collect::<Vec<_>>(),
            "analysed": {"frames": g.len(), "every": k, "medianStep": r3(f64::from(typical))},
        }));
    }
    let (findings, counts) = bounded(findings);
    let mut skipped = Vec::new();
    if w.skipped_items > 0 {
        skipped.push(json!({"check": "all", "reason": format!("{} more items than the {MAX_ITEMS} read: name them with `items`", w.skipped_items)}));
    }
    if sparse > 0 {
        skipped.push(json!({"check": "jump, midStep", "reason": format!("{sparse} moves longer than {MAX_DENSE} frames were read every few frames, too sparse to tell one frame's step")}));
    }
    Ok(json!({
        "sequence": w.seq.0,
        "ok": findings.is_empty(),
        "blocking": false,
        "folder": w.out,
        "sheets": sheets,
        "findings": findings,
        "counts": counts,
        "skipped": skipped,
        "checked": {"items": w.items.len(), "renders": progress.done.load(Ordering::Relaxed), "gpuFrames": renderer.on_gpu, "cpuFrames": renderer.on_cpu, "fps": r3(fps)},
        "thresholds": {
            "jumpMin": r3(f64::from(JUMP_MIN)), "jumpFactor": r3(f64::from(JUMP_FACTOR)),
            "stepMin": r3(f64::from(STEP_MIN)), "stepFactor": r3(f64::from(STEP_FACTOR)),
            "dipMin": DIP_MIN, "layerMin": r3(f64::from(LAYER_MIN)), "fadeShare": r3(f64::from(FADE_SHARE)),
            "replaySame": r3(f64::from(REPLAY_SAME)), "replayMove": r3(f64::from(REPLAY_MOVE)),
            "maxItems": MAX_ITEMS, "maxFrames": MAX_DENSE, "grid": [grid.0, grid.1],
            "note": "starting values, to be tuned on real projects",
        },
        "method": {
            "frames": "before, first, quarter, middle, three quarters, last and settled (0.25 s after an entrance or a transition, the first frame after an exit)",
            "detectors": "every frame of the move rendered small and reduced to a 96×54 grey grid; steps are the mean grey change between neighbouring frames",
            "renderer": "the GPU frame renderer where the host has one, else the CPU reference renderer; no captions",
        },
    }))
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    if !p.is_object() {
        return Err(bad(CMD, "parameters must be an object with `out`"));
    }
    if let Some(k) = p.as_object().and_then(|o| o.keys().find(|k| !matches!(k.as_str(), "sequence" | "items" | "out" | "scale" | "wait"))) {
        return Err(bad(CMD, format!("unknown parameter `{}`: sequence, items, out, scale or wait", k.chars().take(64).collect::<String>())));
    }
    let seq = match p.get("sequence").filter(|v| !v.is_null()) {
        Some(v) => {
            let id = ItemId(v.as_u64().ok_or_else(|| bad(CMD, "`sequence` is a sequence id"))?);
            s.project.sequence(id).ok_or_else(|| bad(CMD, "no such sequence"))?;
            id
        }
        None => s.state.active_sequence.ok_or(EngineError::NoSequence)?,
    };
    let out = p
        .get("out")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|o| !o.is_empty() && o.len() <= 4096 && !o.contains('\0'))
        .ok_or_else(|| bad(CMD, "`out` is the folder for the sheets"))?;
    let out = crate::export_tools::expand_home(out);
    let wait = match p.get("wait") {
        None | Some(Value::Null) => false,
        Some(v) => v.as_bool().ok_or_else(|| bad(CMD, "`wait` is true or false"))?,
    };
    let project = s.project.clone();
    let q = project.sequence(seq).ok_or(EngineError::NoSequence)?;
    q.check_bounds().map_err(|e| bad(CMD, e))?;
    let end = q.duration();
    if end.0 <= 0 {
        return Err(bad(CMD, "the sequence is empty"));
    }
    let (w, h) = (q.settings.width.max(1), q.settings.height.max(1));
    let scale = match p.get("scale").filter(|v| !v.is_null()) {
        Some(v) => {
            v.as_f64().filter(|x| x.is_finite() && (0.05..=1.0).contains(x)).ok_or_else(|| bad(CMD, "`scale` is the tiles' share of the frame, 0.05 to 1"))?
                as f32
        }
        None => (SHEET_SIDE / f64::from(w.max(h))).min(1.0) as f32,
    };
    let all = candidates(&project, q);
    let mut items = match p.get("items").filter(|v| !v.is_null()) {
        Some(v) => chosen(&all, v)?,
        None => all,
    };
    if items.is_empty() {
        return Err(bad(CMD, "nothing in the sequence moves: no animated title, transition or overlay on a visible video track"));
    }
    let skipped_items = items.len().saturating_sub(MAX_ITEMS);
    items.truncate(MAX_ITEMS);
    // a new or empty folder: no sheet replaces a file that is there
    match s.services.list_dir(&out) {
        Some(Ok(names)) if !names.is_empty() => return Err(bad(CMD, format!("{out} is not empty: choose a new or empty folder for the sheets"))),
        Some(Err(e)) if e.kind() != std::io::ErrorKind::NotFound => return Err(bad(CMD, format!("{out}: {e}"))),
        _ => {}
    }
    for (n, item) in items.iter().enumerate() {
        let path = std::path::Path::new(&out).join(item.file(n + 1)).display().to_string();
        crate::output_safety::protect(s, &s.project, &path, CMD)?;
    }
    let mut cuts: BTreeMap<i64, Vec<u64>> = BTreeMap::new();
    for it in q.video_tracks.iter().filter(|t| t.enabled).flat_map(|t| t.items.iter().filter(|i| i.enabled)) {
        for t in [it.start.0, it.end().0].into_iter().filter(|t| *t > 0 && *t < end.0) {
            cuts.entry(t).or_default().push(it.id.0);
        }
    }
    // the sheet, the move and (for an exit or an entrance at an edit) two frames with and without the layer
    let renders: usize = items.iter().map(|i| 7 + dense(i, q.settings.frame_rate.sane(), end).times.len() + if i.edge.is_some() { 4 } else { 0 }).sum();
    let work = Work {
        project: project.clone(),
        seq,
        items,
        skipped_items,
        out: out.clone(),
        scale,
        rate: q.settings.frame_rate.sane(),
        end,
        size: (w, h),
        cuts,
        services: s.services.clone(),
        media: s.media.clone(),
    };
    let job = crate::narration::new_job(s, "Motion filmstrips".into());
    job.progress.total.store(renders as u64, Ordering::Relaxed);
    *job.progress.status.lock().unwrap_or_else(PoisonError::into_inner) = "Rendering the filmstrips".into();
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let outcome: Arc<Mutex<Option<Value>>> = Arc::default();
    s.motion_strips.outcomes.insert(job.id, outcome.clone());
    if s.motion_strips.outcomes.len() > MAX_OUTCOMES
        && let Some(old) = s.motion_strips.outcomes.keys().min().copied()
    {
        s.motion_strips.outcomes.remove(&old);
    }
    let id = crate::narration::spawn(s, job, wait, "filmcraft-motion-strips", move || {
        let t0 = web_time::Instant::now();
        let r = strips(work, &prog).map(|v| {
            let status = format!("{} sheets, {} findings", v["sheets"].as_array().map_or(0, Vec::len), v["findings"].as_array().map_or(0, Vec::len));
            *outcome.lock().unwrap_or_else(PoisonError::into_inner) = Some(v);
            status
        });
        crate::narration::finish(&prog, &res, r, t0);
    })?;
    if !wait {
        return Ok(json!({"job": id, "poll": "jobs.list", "cancel": "jobs.cancel", "folder": out}));
    }
    if let Some(e) = s.jobs.iter().find(|j| j.id == id).and_then(|j| j.to_json()["result"]["error"].as_str().map(str::to_string)) {
        return Err(EngineError::Other(format!("{CMD}: {e}")));
    }
    let mut v = outcome_json(s, id).unwrap_or(Value::Null);
    if let Some(o) = v.as_object_mut() {
        o.insert("job".into(), json!(id));
    }
    Ok(v)
}

#[cfg(test)]
#[path = "motion_strips_tests.rs"]
mod tests;
