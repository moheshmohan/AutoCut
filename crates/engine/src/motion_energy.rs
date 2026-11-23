//! `qa.motionEnergy`: how a finished film or a sequence moves (PX1, after the motion QA idea in
//! huashu-motion; written fresh). Every frame (or, for a long input, every n-th frame) is reduced
//! to a 96×54 grey picture (54×96 for a portrait one) and measured:
//!
//! | code | warns when | starting value |
//! |---|---|---|
//! | `pop` | one frame differs from both neighbours (a flash frame or a stray frame between shots) | both scene steps ≥ 0.2 and ≥ 2.5 × the local median + 0.05; only at the source's own frame rate |
//! | `still` | the picture stays the same for 1 s or more outside the declared holds | mean grey difference to the stretch's first frame < 0.003 of full scale |
//! | `flatOpening` | the first 5 s neither move nor cut | mean motion < 0.05 a second and no cut |
//! | `barelyMoves` | the whole film hardly moves and hardly cuts | mean motion < 0.05 a second and fewer than 2 cuts a minute |
//! | `evenMotion` | motion is the same every second (no build, no rest), films of 10 s or more | per-second motion varies by less than 0.2 of its mean (coefficient of variation) |
//! | `blackFirstFrame` / `emptyFirstFrame` | the first frame is black (or transparent), or one flat colour | mean grey < 0.04; grey spread (standard deviation) < 0.01 |
//! | `loopSeam` | with `loop: true`, the jump from the last frame back to the first stands out | > 4 × the median step between frames, and > 0.02 |
//! | `decode` | frames that could not be read or rendered (skipped) | any |
//!
//! Motion is the mean grey change between frames about a tenth of a second apart, scaled to a
//! second, as a share of full scale (0 is still); steps across a cut or a pop are left out of it.
//! Cuts come from the Scene Edit Detection measure ([`filmcraft_render::scene`]) at sensitivity
//! 50, with a shortest shot of 0.2 s. All of these are starting values, to be tuned on real
//! projects; every finding is a warning (decision D7) and never blocks a delivery.
//!
//! Files are read through the media pool (its hardware decoders where the host registered them,
//! our decoders otherwise); sequences are rendered on the GPU frame renderer when the host has one,
//! on the CPU reference renderer otherwise, without captions (as a horizontal delivery is
//! exported). At most [`MAX_FRAMES`] frames are analysed: a longer input is sampled at a lower
//! rate, which the result reports (`sampledDown`), and pops are then not looked for.
use std::collections::VecDeque;

use filmcraft_media::SharedSource;
use filmcraft_project::ItemId;
use filmcraft_render::scene::{SceneOptions, Signature, detect_cuts, difference, signature};
use filmcraft_time::{FrameRate, Tick};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad};
use crate::{Result, Session};

const CMD: &str = "qa.motionEnergy";
/// The analysis picture of a landscape frame; a portrait frame is analysed at 54×96.
pub const GRID: (usize, usize) = (96, 54);
/// Most frames one measurement analyses; a longer input is sampled at a lower rate.
pub const MAX_FRAMES: i64 = 7200;
/// Longest input measured, seconds.
const MAX_SECONDS: f64 = 12.0 * 3600.0;
/// Two pictures closer than this (mean grey difference, share of full scale) are the same.
pub const STILL_DIFF: f32 = 0.003;
/// Shortest still stretch reported, seconds.
pub const STILL_SECONDS: f64 = 1.0;
/// The opening, seconds.
pub const OPENING_SECONDS: f64 = 5.0;
/// Motion a second below which an opening (or a film) is flat.
pub const FLAT_MOTION: f64 = 0.05;
/// A film that barely moves also cuts fewer times a minute than this.
pub const BARELY_CUTS_PER_MINUTE: f64 = 2.0;
/// Per-second motion varying less than this share of its mean is even.
pub const EVEN_CONTRAST: f64 = 0.2;
/// Films shorter than this are not judged for even motion, seconds.
pub const EVEN_MIN_SECONDS: f64 = 10.0;
/// A first frame darker than this mean grey is black.
pub const BLACK_MEAN: f64 = 0.04;
/// A first frame with less grey spread than this is one flat colour.
pub const EMPTY_SPREAD: f64 = 0.01;
/// A pop's scene steps into and out of its frame are at least this.
pub const POP_STEP: f32 = 0.2;
/// The loop seam stands out at this many median steps, and at least [`SEAM_MIN`].
pub const SEAM_FACTOR: f32 = 4.0;
pub const SEAM_MIN: f32 = 0.02;
/// The motion comparison spans about this long, seconds.
const MOTION_SPAN: f64 = 0.1;
/// Contact-sheet frames a second, tiles a row and the tiles' long side.
const SHEET_FPS: f64 = 4.0;
const SHEET_COLUMNS: usize = 16;
const SHEET_SIDE: u32 = 128;
/// Most findings of one code listed (their count is always given).
const MAX_PER_CODE: usize = 200;
/// Most cut times listed.
const MAX_CUTS_LISTED: usize = 2000;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Measure Motion Energy",
        menu: &[],
        shortcut: None,
        params: r#"{"path":path? (a rendered film; exactly one of path or sequence),"sequence":id?,"fps":0.1..240? (frames analysed a second; default the source's own, lowered to keep 7200 frames),"holds":[{"start":ticks,"end":ticks,"reason":str}]?,"loop":bool=false,"sheet":path? (contact sheet, 4 frames a second)}"#,
        enabled: always,
        run,
        journal: false,
    }]
}

/// What to measure and how.
#[derive(Clone, Debug, Default)]
pub(crate) struct Options {
    /// Frames analysed a second (None: the source's own rate, lowered for long inputs).
    pub fps: Option<f64>,
    /// Declared holds, ticks: a still inside one is intended.
    pub holds: Vec<(i64, i64)>,
    /// Report stills and flat or even motion (an animation that is meant to be still says no).
    pub expect_motion: bool,
    /// The film loops: its last frame should lead back into its first.
    pub looped: bool,
    /// Where to write the contact sheet.
    pub sheet: Option<String>,
    /// A transparent overlay: its first frame shows what is under it, so it is not judged.
    pub overlay: bool,
}

/// A finding: its code, a warning (D7), where it is, what it means and its own measures.
pub(crate) fn finding(code: &str, start: Tick, end: Option<Tick>, message: String, detail: Value) -> Value {
    let mut v = json!({"code": code, "severity": "warn", "time": start.0, "seconds": r3(start.seconds()), "message": message});
    if let Some(obj) = v.as_object_mut() {
        if let Some(e) = end {
            obj.insert("end".into(), json!(e.0));
            obj.insert("until".into(), json!(r3(e.seconds())));
        }
        if let Value::Object(extra) = detail {
            obj.extend(extra);
        }
    }
    v
}

pub(crate) fn r3(x: f64) -> f64 {
    if x.is_finite() { (x * 1000.0).round() / 1000.0 } else { 0.0 }
}

fn r4(x: f64) -> f64 {
    if x.is_finite() { (x * 10_000.0).round() / 10_000.0 } else { 0.0 }
}

/// The parts of `[a, b)` no hold covers.
pub(crate) fn uncovered(a: i64, b: i64, holds: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let mut pieces = vec![(a, b)];
    for &(s, e) in holds {
        let mut next = Vec::with_capacity(pieces.len() + 1);
        for (x, y) in pieces {
            if e <= x || s >= y {
                next.push((x, y));
                continue;
            }
            if s > x {
                next.push((x, s));
            }
            if e < y {
                next.push((e, y));
            }
        }
        pieces = next;
    }
    pieces
}

/// Findings in time order, at most [`MAX_PER_CODE`] of each code, and how many each code had.
pub(crate) fn bounded(mut findings: Vec<Value>) -> (Vec<Value>, Value) {
    findings.sort_by_key(|f| f["time"].as_i64().unwrap_or(0));
    let mut counts = serde_json::Map::new();
    let mut kept = Vec::new();
    for f in findings {
        let code = f["code"].as_str().unwrap_or("").to_string();
        let n = counts.get(&code).and_then(Value::as_u64).unwrap_or(0) + 1;
        counts.insert(code, json!(n));
        if n as usize <= MAX_PER_CODE {
            kept.push(f);
        }
    }
    (kept, Value::Object(counts))
}

/// `rgba` (straight sRGB RGBA8, `w`×`h`) over black, box-averaged to `ow`×`oh` opaque RGBA8. A
/// large picture is read on a lattice of at most four points per output pixel each way, so a
/// decoder that ignores the requested scale costs a bounded amount per frame.
pub(crate) fn box_reduce(w: u32, h: u32, rgba: &[u8], ow: usize, oh: usize) -> Result<Vec<u8>> {
    let pixels = u64::from(w) * u64::from(h);
    if w == 0 || h == 0 || pixels > 16_777_216 || pixels * 4 != rgba.len() as u64 || ow == 0 || oh == 0 || ow > 4096 || oh > 4096 {
        return Err(bad(CMD, "invalid or oversized picture"));
    }
    let (wu, hu) = (w as usize, h as usize);
    let px = rgba.as_chunks::<4>().0;
    let (sx, sy) = ((wu / (ow * 4)).max(1), (hu / (oh * 4)).max(1));
    let mut sum = vec![[0u64; 3]; ow * oh];
    let mut count = vec![0u64; ow * oh];
    for y in (0..hu).step_by(sy) {
        let cy = y * oh / hu;
        for x in (0..wu).step_by(sx) {
            let (Some(p), Some(acc), Some(n)) = (px.get(y * wu + x), sum.get_mut(cy * ow + x * ow / wu), count.get_mut(cy * ow + x * ow / wu)) else {
                continue;
            };
            let a = u64::from(p[3]);
            for (s, c) in acc.iter_mut().zip(p) {
                *s += u64::from(*c) * a;
            }
            *n += 1;
        }
    }
    let mut out = Vec::with_capacity(ow * oh * 4);
    for cy in 0..oh {
        for cx in 0..ow {
            let i = cy * ow + cx;
            let (s, n) = match (sum.get(i), count.get(i)) {
                (Some(s), Some(n)) if *n > 0 => (*s, *n),
                // an output pixel no source pixel fell in (a picture smaller than the grid): its nearest
                _ => {
                    let p = px.get((cy * hu / oh) * wu + cx * wu / ow).copied().unwrap_or([0; 4]);
                    let a = u64::from(p[3]);
                    ([u64::from(p[0]) * a, u64::from(p[1]) * a, u64::from(p[2]) * a], 1)
                }
            };
            out.extend(s.map(|v| (v / (n * 255)).min(255) as u8));
            out.push(255);
        }
    }
    Ok(out)
}

/// Grey (sRGB-encoded luma) of opaque RGBA8.
pub(crate) fn grey_of(rgba: &[u8]) -> Vec<u8> {
    rgba.as_chunks::<4>().0.iter().map(|p| ((54 * u32::from(p[0]) + 183 * u32::from(p[1]) + 18 * u32::from(p[2]) + 127) / 255) as u8).collect()
}

/// Mean absolute difference of two grey pictures, a share of full scale (1 when they don't match).
pub(crate) fn mean_diff(a: &[u8], b: &[u8]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 1.0;
    }
    let s: u64 = a.iter().zip(b).map(|(x, y)| u64::from(x.abs_diff(*y))).sum();
    (s as f64 / (a.len() as f64 * 255.0)) as f32
}

fn median(mut v: Vec<f32>) -> f32 {
    v.retain(|x| x.is_finite());
    v.sort_by(f32::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

/// The analysis grid for a `w`×`h` picture.
pub(crate) fn grid_for(w: u32, h: u32) -> (usize, usize) {
    if w >= h { GRID } else { (GRID.1, GRID.0) }
}

/// The measurements, fed one analysed frame at a time (bounded state: a tenth of a second of
/// pictures, two signatures and a few numbers a frame).
pub(crate) struct Meter {
    /// Frames analysed a second.
    rate: f64,
    /// Analysed at the source's own rate (pops can be told).
    native: bool,
    /// Frames back for the motion comparison.
    back: usize,
    times: Vec<Tick>,
    /// Grey difference to the previous frame (`step[0]` is 0).
    step: Vec<f32>,
    /// Scene-signature difference to the previous frame (`scene[0]` is 0).
    scene: Vec<f32>,
    /// Scene-signature difference to the frame two back.
    skip: Vec<f32>,
    /// Grey difference to the frame `back` earlier.
    span: Vec<Option<f32>>,
    recent: VecDeque<Vec<u8>>,
    sigs: VecDeque<Signature>,
    first: Option<Vec<u8>>,
    last: Option<Vec<u8>>,
    /// The still stretch being followed: its first frame and picture.
    anchor: Option<(usize, Vec<u8>)>,
    /// Still stretches: (first frame, first frame that changed).
    runs: Vec<(usize, usize)>,
}

impl Meter {
    pub(crate) fn new(rate: f64, native: bool) -> Self {
        let rate = if rate.is_finite() && rate > 0.0 { rate } else { 1.0 };
        let back = ((rate * MOTION_SPAN).round() as usize).clamp(1, 32);
        Meter {
            rate,
            native,
            back,
            times: Vec::new(),
            step: Vec::new(),
            scene: Vec::new(),
            skip: Vec::new(),
            span: Vec::new(),
            recent: VecDeque::new(),
            sigs: VecDeque::new(),
            first: None,
            last: None,
            anchor: None,
            runs: Vec::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.times.len()
    }

    /// The next frame at `t`: `small` is its opaque RGBA on the analysis grid (`grid`).
    pub(crate) fn push(&mut self, t: Tick, small: &[u8], grid: (usize, usize)) {
        let grey = grey_of(small);
        let idx = self.times.len();
        self.step.push(self.recent.back().map_or(0.0, |p| mean_diff(p, &grey)));
        let sig = signature(small, grid.0, grid.1);
        self.scene.push(self.sigs.back().map_or(0.0, |p| difference(p, &sig)));
        self.skip.push(if self.sigs.len() == 2 { self.sigs.front().map_or(0.0, |p| difference(p, &sig)) } else { 0.0 });
        self.sigs.push_back(sig);
        if self.sigs.len() > 2 {
            self.sigs.pop_front();
        }
        self.span.push(if self.recent.len() == self.back { self.recent.front().map(|p| mean_diff(p, &grey)) } else { None });
        match &self.anchor {
            Some((_, a)) if mean_diff(a, &grey) < STILL_DIFF => {}
            Some((start, _)) => {
                if idx.saturating_sub(*start) >= 2 {
                    self.runs.push((*start, idx));
                }
                self.anchor = Some((idx, grey.clone()));
            }
            None => self.anchor = Some((idx, grey.clone())),
        }
        if self.first.is_none() {
            self.first = Some(grey.clone());
        }
        self.recent.push_back(grey.clone());
        if self.recent.len() > self.back {
            self.recent.pop_front();
        }
        self.last = Some(grey);
        self.times.push(t);
    }

    /// The findings and measures, the film being `duration` long.
    pub(crate) fn finish(mut self, duration: Tick, o: &Options) -> (Vec<Value>, Value, Vec<Value>) {
        let n = self.times.len();
        let mut findings = Vec::new();
        let mut skipped = Vec::new();
        let time = |i: usize| self.times.get(i).copied().unwrap_or(duration);
        // ---- pops: a one-frame shot
        let mut pops = Vec::new();
        if self.native {
            for i in 1..n.saturating_sub(1) {
                let (Some(&a), Some(&b)) = (self.scene.get(i), self.scene.get(i + 1)) else { continue };
                if a < POP_STEP || b < POP_STEP {
                    continue;
                }
                let around: Vec<f32> =
                    (i.saturating_sub(5).max(1)..(i + 7).min(n)).filter(|j| *j != i && *j != i + 1).filter_map(|j| self.scene.get(j).copied()).collect();
                if a.min(b) >= 2.5 * median(around) + 0.05 {
                    pops.push(i);
                }
            }
        } else {
            skipped.push(json!({"check": "pop", "reason": "frames were analysed below the source's own rate; a single frame cannot be told"}));
        }
        // ---- cuts: Scene Edit Detection's measure, with each pop taken out
        let mut diffs = self.scene.clone();
        for &i in &pops {
            if let Some(d) = diffs.get_mut(i) {
                *d = 0.0;
            }
            let across = self.skip.get(i + 1).copied().unwrap_or(0.0);
            if let Some(d) = diffs.get_mut(i + 1) {
                *d = across;
            }
        }
        let min_shot = ((self.rate * 0.2).round() as usize).max(1);
        let cuts = detect_cuts(&diffs, &SceneOptions { sensitivity: 50.0, min_shot_frames: min_shot });
        // steps that are not motion: a cut, a pop and the step out of it
        let mut jump = vec![false; n];
        for &c in &cuts {
            if let Some(j) = jump.get_mut(c) {
                *j = true;
            }
        }
        for &p in &pops {
            for k in [p, p + 1] {
                if let Some(j) = jump.get_mut(k) {
                    *j = true;
                }
            }
        }
        // `before[i]`: how many of the frames before `i` are jumps
        let mut seen = 0usize;
        let before: Vec<usize> = std::iter::once(0)
            .chain(jump.iter().map(|j| {
                seen += usize::from(*j);
                seen
            }))
            .collect();
        let jumps_in = |a: usize, b: usize| before.get(b + 1).copied().unwrap_or(0).saturating_sub(before.get(a + 1).copied().unwrap_or(0)) > 0;
        // ---- motion a second
        let scale = self.rate / self.back as f64;
        let motion: Vec<(usize, f64)> = (0..n)
            .filter_map(|j| {
                let d = self.span.get(j).copied().flatten()?;
                (!jumps_in(j.saturating_sub(self.back), j)).then_some((j, f64::from(d) * scale))
            })
            .collect();
        let seconds = duration.seconds().max(0.0);
        let bucket = (seconds / 3600.0).ceil().max(1.0);
        let buckets = ((seconds / bucket).ceil() as usize).clamp(1, 3600);
        let mut per = vec![(0.0f64, 0usize); buckets];
        for &(j, m) in &motion {
            let b = ((time(j).seconds() / bucket).floor().max(0.0) as usize).min(buckets - 1);
            if let Some(slot) = per.get_mut(b) {
                slot.0 += m;
                slot.1 += 1;
            }
        }
        let per_second: Vec<Option<f64>> = per.iter().map(|(s, c)| (*c > 0).then(|| s / *c as f64)).collect();
        let mean = (!motion.is_empty()).then(|| motion.iter().map(|m| m.1).sum::<f64>() / motion.len() as f64);
        let filled: Vec<f64> = per_second.iter().flatten().copied().collect();
        let contrast = match mean {
            Some(m) if filled.len() >= 3 && m > 1e-6 => {
                let mu = filled.iter().sum::<f64>() / filled.len() as f64;
                let var = filled.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / filled.len() as f64;
                (mu > 1e-9).then(|| var.sqrt() / mu)
            }
            _ => None,
        };
        let opening_end = Tick::from_seconds_f64(OPENING_SECONDS.min(seconds));
        let opening: Vec<f64> = motion.iter().filter(|(j, _)| time(*j) < opening_end).map(|m| m.1).collect();
        let opening_motion = (!opening.is_empty()).then(|| opening.iter().sum::<f64>() / opening.len() as f64);
        let opening_cuts = cuts.iter().filter(|c| time(**c) < opening_end).count();
        let per_minute = if seconds > 0.0 { cuts.len() as f64 * 60.0 / seconds } else { 0.0 };
        // ---- findings
        for &i in &pops {
            findings.push(finding(
                "pop",
                time(i),
                Some(time(i + 1)),
                format!("the frame at {:.2} s differs from the frames on both sides: a single-frame pop", time(i).seconds()),
                json!({"into": r4(f64::from(self.scene.get(i).copied().unwrap_or(0.0))), "outOf": r4(f64::from(self.scene.get(i + 1).copied().unwrap_or(0.0)))}),
            ));
        }
        if o.expect_motion {
            // the last stretch still open runs to the end of the film
            if let Some((start, _)) = self.anchor.take()
                && n.saturating_sub(start) >= 2
            {
                self.runs.push((start, n));
            }
            let minimum = Tick::from_seconds_f64(STILL_SECONDS).0;
            for &(a, b) in &self.runs {
                for (x, y) in uncovered(time(a).0, time(b).0, &o.holds) {
                    if y.saturating_sub(x) >= minimum {
                        let secs = Tick(y.saturating_sub(x)).seconds();
                        findings.push(finding(
                            "still",
                            Tick(x),
                            Some(Tick(y)),
                            format!("the picture holds still for {secs:.1} s; declare a hold if it is meant"),
                            json!({"length": r3(secs)}),
                        ));
                    }
                }
            }
            if let Some(m) = opening_motion
                && m < FLAT_MOTION
                && opening_cuts == 0
            {
                findings.push(finding(
                    "flatOpening",
                    Tick::ZERO,
                    Some(opening_end),
                    format!("the first {:.1} s neither move ({m:.3} a second) nor cut", opening_end.seconds()),
                    json!({"motion": r4(m)}),
                ));
            }
            if let Some(m) = mean
                && m < FLAT_MOTION
                && per_minute < BARELY_CUTS_PER_MINUTE
            {
                findings.push(finding(
                    "barelyMoves",
                    Tick::ZERO,
                    Some(duration),
                    format!("the film barely moves ({m:.3} a second on average) and cuts {per_minute:.1} times a minute"),
                    json!({"motion": r4(m), "cutsPerMinute": r3(per_minute)}),
                ));
            }
            if let Some(c) = contrast
                && c < EVEN_CONTRAST
                && seconds >= EVEN_MIN_SECONDS
                && mean.is_some_and(|m| m >= FLAT_MOTION)
            {
                findings.push(finding(
                    "evenMotion",
                    Tick::ZERO,
                    Some(duration),
                    format!("motion is the same every second (it varies by {:.0} % of its mean): nothing builds or rests", c * 100.0),
                    json!({"contrast": r4(c)}),
                ));
            }
        } else {
            skipped.push(
                json!({"check": "still, flatOpening, barelyMoves, evenMotion", "reason": "the composition is declared not to move (expectMotion: false)"}),
            );
        }
        let (first_mean, first_spread) = self.first.as_deref().map_or((0.0, 0.0), |g| {
            let len = g.len().max(1) as f64;
            let mu = g.iter().map(|v| f64::from(*v)).sum::<f64>() / len;
            let var = g.iter().map(|v| (f64::from(*v) - mu).powi(2)).sum::<f64>() / len;
            (mu / 255.0, var.sqrt() / 255.0)
        });
        if o.overlay {
            skipped.push(json!({"check": "blackFirstFrame, emptyFirstFrame", "reason": "a transparent overlay starts on whatever is under it"}));
        } else if self.first.is_some() {
            if first_mean < BLACK_MEAN {
                findings.push(finding(
                    "blackFirstFrame",
                    Tick::ZERO,
                    None,
                    "the first frame is black (or transparent)".into(),
                    json!({"mean": r4(first_mean)}),
                ));
            } else if first_spread < EMPTY_SPREAD {
                findings.push(finding(
                    "emptyFirstFrame",
                    Tick::ZERO,
                    None,
                    "the first frame is one flat colour".into(),
                    json!({"mean": r4(first_mean), "spread": r4(first_spread)}),
                ));
            }
        }
        let steps: Vec<f32> = (1..n).filter(|i| !jump.get(*i).copied().unwrap_or(false)).filter_map(|i| self.step.get(i).copied()).collect();
        let median_step = median(steps);
        let seam = match (&self.first, &self.last) {
            (Some(a), Some(b)) if n >= 2 => Some(mean_diff(b, a)),
            _ => None,
        };
        if o.looped
            && let Some(seam) = seam
            && seam > (SEAM_FACTOR * median_step).max(SEAM_MIN)
        {
            let last = time(n.saturating_sub(1));
            findings.push(finding(
                "loopSeam",
                last,
                Some(duration),
                format!("going from the last frame back to the first jumps {seam:.3}, {:.1} times the usual step between frames", seam / median_step.max(1e-4)),
                json!({"seam": r4(f64::from(seam)), "medianStep": r4(f64::from(median_step))}),
            ));
        }
        let measures = json!({
            "cutsPerMinute": r3(per_minute),
            "cutCount": cuts.len(),
            "cuts": cuts.iter().take(MAX_CUTS_LISTED).map(|c| time(*c).0).collect::<Vec<_>>(),
            "pops": pops.len(),
            "motion": {
                "unit": "mean grey change a second, share of full scale (0 = still)",
                "mean": mean.map(r4),
                "contrast": contrast.map(r4),
                "opening": opening_motion.map(r4),
                "bucketSeconds": bucket,
                "perSecond": per_second.iter().map(|v| v.map(r4)).collect::<Vec<_>>(),
            },
            "firstFrame": {"mean": r4(first_mean), "spread": r4(first_spread)},
            "loopSeam": {"seam": seam.map(|v| r4(f64::from(v))), "medianStep": r4(f64::from(median_step))},
        });
        (findings, measures, skipped)
    }
}

/// The starting values, as the result reports them.
pub(crate) fn thresholds() -> Value {
    json!({
        "grid": [GRID.0, GRID.1],
        "stillDiff": r4(f64::from(STILL_DIFF)), "stillSeconds": STILL_SECONDS,
        "openingSeconds": OPENING_SECONDS, "flatMotion": FLAT_MOTION,
        "barelyCutsPerMinute": BARELY_CUTS_PER_MINUTE,
        "evenContrast": EVEN_CONTRAST, "evenMinSeconds": EVEN_MIN_SECONDS,
        "blackMean": BLACK_MEAN, "emptySpread": EMPTY_SPREAD,
        "popStep": r4(f64::from(POP_STEP)), "seamFactor": r4(f64::from(SEAM_FACTOR)), "seamMin": r4(f64::from(SEAM_MIN)),
        "cutSensitivity": 50, "maxFrames": MAX_FRAMES,
    })
}

/// Holds from `p[key]`: up to 2000 `{start, end, reason}` intervals inside `[0, end]`.
pub(crate) fn holds_p(cmd: &str, p: &Value, key: &str, end: Tick) -> Result<Vec<(i64, i64)>> {
    let Some(values) = p.get(key).filter(|v| !v.is_null()) else { return Ok(Vec::new()) };
    let values = values.as_array().filter(|a| a.len() <= 2000).ok_or_else(|| bad(cmd, format!("{key} needs up to 2000 intervals")))?;
    values
        .iter()
        .map(|v| {
            let a = v["start"].as_i64().filter(|a| *a >= 0).ok_or_else(|| bad(cmd, "interval start is ticks"))?;
            let b = v["end"].as_i64().filter(|b| *b > a && *b <= end.0).ok_or_else(|| bad(cmd, "interval end must follow start within the film"))?;
            if v["reason"].as_str().is_none_or(|s| s.trim().is_empty() || s.len() > 512) {
                return Err(bad(cmd, "a hold needs a reason"));
            }
            Ok((a, b))
        })
        .collect()
}

/// Every `k`-th frame of `rate` over `duration`, with `k` keeping at most [`MAX_FRAMES`] frames
/// and the asked `fps`: (`k`, frames).
fn plan(duration: Tick, rate: FrameRate, fps: Option<f64>) -> (i64, i64) {
    let rate = rate.sane();
    let native = rate.frame_at(Tick(duration.0.saturating_sub(1))).saturating_add(1).max(1);
    let capped = native.saturating_add(MAX_FRAMES - 1) / MAX_FRAMES;
    let asked = fps.map_or(1, |f| (rate.as_f64() / f).round().clamp(1.0, 1e12) as i64);
    let k = capped.max(asked).max(1);
    (k, native.saturating_add(k - 1) / k)
}

/// Measure `duration` of pictures `frame` gives (straight RGBA8 at a time), at `rate`.
fn measure(
    s: &Session,
    duration: Tick,
    rate: FrameRate,
    size: (u32, u32),
    o: &Options,
    mut frame: impl FnMut(Tick) -> std::result::Result<(u32, u32, Vec<u8>), String>,
) -> Result<Value> {
    if duration.0 <= 0 || duration.seconds() > MAX_SECONDS {
        return Err(bad(CMD, "the film must be longer than zero and at most 12 hours"));
    }
    let rate = rate.sane();
    let (k, count) = plan(duration, rate, o.fps);
    let analysed = rate.as_f64() / k as f64;
    let grid = grid_for(size.0, size.1);
    let mut meter = Meter::new(analysed, k == 1);
    let interval = (1.0 / SHEET_FPS).max(duration.seconds() / crate::contact_sheet::MAX_TILES as f64);
    let mut next_tile = 0.0f64;
    let mut tiles: Vec<(Vec<u8>, u32, u32, String)> = Vec::new();
    let (mut failures, mut errors) = (0usize, Vec::new());
    for j in 0..count {
        let t = rate.tick_of(j.saturating_mul(k));
        if t >= duration {
            break;
        }
        let picture = frame(t).and_then(|(w, h, rgba)| {
            let small = box_reduce(w, h, &rgba, grid.0, grid.1).map_err(|e| e.to_string())?;
            let tile = if o.sheet.is_some() && t.seconds() + 1e-9 >= next_tile {
                let side = f64::from(SHEET_SIDE);
                let (tw, th) = if w >= h { (side, side * f64::from(h) / f64::from(w)) } else { (side * f64::from(w) / f64::from(h), side) };
                let (tw, th) = ((tw.round() as u32).max(1), (th.round() as u32).max(1));
                Some((box_reduce(w, h, &rgba, tw as usize, th as usize).map_err(|e| e.to_string())?, tw, th))
            } else {
                None
            };
            Ok((small, tile))
        });
        match picture {
            Ok((small, tile)) => {
                meter.push(t, &small, grid);
                if let Some((rgba, tw, th)) = tile {
                    tiles.push((rgba, tw, th, crate::contact_sheet::stamp(t.seconds())));
                    while next_tile <= t.seconds() + 1e-9 {
                        next_tile += interval;
                    }
                }
            }
            Err(e) => {
                failures += 1;
                if errors.len() < 20 {
                    errors.push(json!({"time": t.0, "error": e}));
                }
            }
        }
    }
    if meter.len() == 0 {
        let why = errors.first().and_then(|e| e["error"].as_str()).unwrap_or("no frame").to_string();
        return Err(bad(CMD, format!("no frame could be read: {why}")));
    }
    let frames = meter.len();
    let (mut findings, measures, skipped) = meter.finish(duration, o);
    if failures > 0 {
        findings.push(finding(
            "decode",
            errors.first().and_then(|e| e["time"].as_i64()).map_or(Tick::ZERO, Tick),
            None,
            format!("{failures} frames could not be read and were left out"),
            json!({"failed": failures, "errors": errors}),
        ));
    }
    let (findings, counts) = bounded(findings);
    let sheet = match &o.sheet {
        Some(path) => {
            let count = tiles.len();
            let mut rows = Vec::with_capacity(count.div_ceil(SHEET_COLUMNS));
            let mut rest = tiles.into_iter().peekable();
            while rest.peek().is_some() {
                rows.push(rest.by_ref().take(SHEET_COLUMNS).collect());
            }
            let written = crate::contact_sheet::write_tiles(s, CMD, rows, path)?;
            Some(
                json!({"path": written["path"], "width": written["width"], "height": written["height"], "fps": r3(1.0 / interval.max(1e-9)), "columns": SHEET_COLUMNS, "tiles": count}),
            )
        }
        None => None,
    };
    Ok(json!({
        "ok": findings.is_empty(),
        "blocking": false,
        "findings": findings,
        "counts": counts,
        "measures": measures,
        "skipped": skipped,
        "duration": duration.0,
        "frames": frames,
        "fps": r3(analysed),
        "nativeFps": r3(rate.as_f64()),
        "sampledDown": k > 1,
        "grid": [grid.0, grid.1],
        "holds": o.holds.iter().map(|(a, b)| json!({"start": a, "end": b})).collect::<Vec<_>>(),
        "thresholds": thresholds(),
        "sheet": sheet,
    }))
}

/// Measure an opened file (or image sequence).
pub(crate) fn measure_source(s: &Session, source: &SharedSource, o: &Options) -> Result<Value> {
    let info = source.info();
    let video = info.video.as_ref().ok_or_else(|| bad(CMD, "the file has no picture"))?;
    let (w, h) = (video.width.max(1), video.height.max(1));
    let scale = (192.0 / f64::from(w.max(h))).min(1.0) as f32;
    let mut v = measure(s, info.duration, video.frame_rate, (w, h), o, |time| {
        let f = source.video_frame(filmcraft_media::FrameRequest { time, scale }).map_err(|e| e.to_string())?;
        Ok((f.width, f.height, f.to_rgba8()))
    })?;
    v["source"] = json!("file");
    v["reader"] = json!("media pool decoders (hardware where the host registered them)");
    Ok(v)
}

/// Measure sequence `seq` as rendered: on the GPU frame renderer where the host has one, else on
/// the CPU reference renderer. `full_res` renders from the originals, never proxies.
pub(crate) fn measure_sequence(s: &Session, seq: ItemId, o: &Options, full_res: bool) -> Result<Value> {
    let project = s.project.clone();
    let q = project.sequence(seq).ok_or_else(|| bad(CMD, "no such sequence"))?;
    q.check_bounds().map_err(|e| bad(CMD, e))?;
    let duration = q.duration();
    if duration.0 <= 0 {
        return Err(bad(CMD, "the sequence is empty"));
    }
    let (w, h) = (q.settings.width.max(1), q.settings.height.max(1));
    let rate = q.settings.frame_rate;
    let provider =
        if full_res { s.media.full_res_provider(project.clone(), s.services.clone()) } else { s.media.provider(project.clone(), s.services.clone()) };
    let mut gpu = filmcraft_export::frame_renderer();
    let (mut on_gpu, mut on_cpu) = (0usize, 0usize);
    let scale = (192.0 / f64::from(w.max(h))).min(1.0) as f32;
    let opts = filmcraft_render::RenderOptions { scale, ..Default::default() };
    let mut v = measure(s, duration, rate, (w, h), o, |t| {
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
        Ok((img.w as u32, img.h as u32, img.over_black_rgba8()))
    })?;
    v["source"] = json!("sequence");
    v["sequence"] = json!(seq.0);
    v["renderer"] = json!({"gpuFrames": on_gpu, "cpuFrames": on_cpu, "captions": false});
    Ok(v)
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    if !p.is_object() && !p.is_null() {
        return Err(bad(CMD, "parameters must be an object"));
    }
    let path = match p.get("path").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(v.as_str().filter(|s| !s.trim().is_empty() && s.len() <= 4096).ok_or_else(|| bad(CMD, "`path` is a file path"))?.to_string()),
    };
    let seq = match p.get("sequence").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(ItemId(v.as_u64().ok_or_else(|| bad(CMD, "`sequence` is a sequence id"))?)),
    };
    let fps = match p.get("fps").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(v.as_f64().filter(|f| f.is_finite() && (0.1..=240.0).contains(f)).ok_or_else(|| bad(CMD, "`fps` is frames a second, 0.1 to 240"))?),
    };
    let looped = match p.get("loop").filter(|v| !v.is_null()) {
        None => false,
        Some(v) => v.as_bool().ok_or_else(|| bad(CMD, "`loop` is true or false"))?,
    };
    let sheet = match p.get("sheet").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(v.as_str().map(str::trim).filter(|s| !s.is_empty() && s.len() <= 4096).ok_or_else(|| bad(CMD, "`sheet` is a file path"))?.to_string()),
    };
    if let Some(out) = &sheet {
        let out = crate::export_tools::expand_home(out);
        crate::output_safety::protect(s, &s.project, &out, CMD)?;
        if path.as_deref().is_some_and(|input| crate::output_safety::aliases(&out, &crate::export_tools::expand_home(input))) {
            return Err(bad(CMD, "the sheet would replace the film being measured; choose another path"));
        }
    }
    let mut o = Options { fps, holds: Vec::new(), expect_motion: true, looped, sheet, overlay: false };
    match (path, seq) {
        (Some(path), None) => {
            let path = crate::export_tools::expand_home(&path);
            let source = s.media.open_file(&path, s.services.as_ref()).map_err(|e| bad(CMD, e.to_string()))?;
            let end = source.info().duration;
            o.holds = holds_p(CMD, p, "holds", end)?;
            let mut v = measure_source(s, &source, &o)?;
            v["path"] = json!(path);
            Ok(v)
        }
        (None, Some(seq)) => {
            let q = s.project.sequence(seq).ok_or_else(|| bad(CMD, "no such sequence"))?;
            o.holds = holds_p(CMD, p, "holds", q.duration())?;
            measure_sequence(s, seq, &o, false)
        }
        _ => Err(bad(CMD, "give exactly one of `path` (a rendered film) or `sequence`")),
    }
}

#[cfg(test)]
#[path = "motion_energy_tests.rs"]
mod tests;
