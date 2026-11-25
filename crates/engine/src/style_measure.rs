//! `media.measureStyle` (PX6.1, PX6.4): how a video is edited, measured so that another edit can
//! be cut like it ("edit like a reference"). The idea of measuring a reference film's rhythm comes
//! from onetake (feitangyuan/onetake, PolyForm Noncommercial): only the idea is taken, from its
//! README; every measure, threshold and word here is written fresh.
//!
//! A video file (`path`), a media item or a sequence (`item`, or `sequence`), between `from` and
//! `to` (seconds), is sampled at `fps` frames a second (10 by default, never above its own rate)
//! and each frame reduced as `qa.motionEnergy` reduces it (96×54 grey for its meter, 192×108 for
//! sharpness, 16×9 for framing). What is measured:
//!
//! | measure | how |
//! |---|---|
//! | cuts | `qa.motionEnergy`'s meter: the Scene Edit Detection measure ([`filmcraft_render::scene`]) at sensitivity 50, a shortest shot of 0.2 s, one-frame pops taken out |
//! | jump cuts | a cut whose two sides share their framing and colours (16×9 layouts correlating at [`CUT_SAME_FRAMING`] or more, mean colours within [`CUT_SAME_COLOUR`]), and a sudden lasting jump in the picture the cut measure lets through (a step at least [`JUMP_MIN`] and [`JUMP_FACTOR`] times the steps either side, the next frame not going back, the framing correlating at [`SAME_FRAMING`]) |
//! | shot lengths | between cuts (jump cuts included); median, 10th, 25th, 75th and 90th percentiles, mean and spread (coefficient of variation); shots cut off by a sampled window are left out |
//! | bursts | [`BURST_MIN_SHOTS`] or more shots in a row each at most [`BURST_SHOT`] s and [`BURST_SHARE`] of the median |
//! | stillness | the share of the time in still stretches of 1 s or more (`qa.motionEnergy`'s `still`) |
//! | energy map | per stretch (at most 60): `0.6·m/(m+0.25) + 0.4·c/(c+0.5)`, `m` the motion a second (mean grey change, cuts left out) and `c` the cuts a second; 0 still … 1 frantic |
//! | settle | after each move inside a shot (motion at least [`MOVE_MIN`] a second and twice the shot's median), how long until the motion falls to [`SETTLE_SHARE`] of its peak; a shot that cuts first counts as a cut on the move |
//! | motion blur | per shot, how much sharpness (mean grey gradient) its most moving third of frames loses against its calmest third, when they differ in motion; 0 crisp … 1 smeared |
//! | loudness | ITU-R BS.1770 (`filmcraft_audio_dsp::LoudnessMeter`): integrated, range, true peak and the momentary loudness per stretch |
//! | palette | the commonest colours (4×4×4 bins of every analysed frame), and brightness, contrast, saturation and warmth from the colour pass's measure ([`filmcraft_render::color_balance`]) on up to 12 frames |
//!
//! **Bounded.** At most [`MAX_FRAMES`] frames are analysed; a longer stretch is sampled as evenly
//! spread windows of [`WINDOW_SECONDS`] s (the answer's `analysed.windows`), and a stretch longer
//! than [`MAX_SECONDS`] is cut there. Sound is read for the analysed windows only, in 0.1 s
//! chunks. Files are read through the media pool (its hardware decoders where the host
//! registered them, ours otherwise); sequences render on the GPU frame renderer where the host
//! has one, else on the CPU, without captions. Frames that cannot be read are skipped and
//! counted.
use filmcraft_project::ItemId;
use filmcraft_render::color_balance;
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad};
use crate::motion_energy::{Meter, Options as MeterOptions, box_reduce, grey_of, grid_for, mean_diff, r3};
use crate::{Result, Session};

const CMD: &str = "media.measureStyle";
/// Frames analysed a second by default, and the range a caller may ask.
pub const DEFAULT_FPS: f64 = 10.0;
pub const FPS_RANGE: (f64, f64) = (2.0, 30.0);
/// Most frames one measurement analyses.
pub const MAX_FRAMES: usize = 3600;
/// A stretch with more frames than that is sampled in windows this long, seconds.
pub const WINDOW_SECONDS: f64 = 30.0;
/// Longest stretch measured, seconds (a longer one is cut here).
pub const MAX_SECONDS: f64 = 12.0 * 3600.0;
/// The decoded picture's long side, pixels.
const LONG_SIDE: f64 = 192.0;
/// The sharpness picture (a portrait frame is measured at 108×192).
const SHARP_GRID: (usize, usize) = (192, 108);
/// The framing picture.
const LAYOUT: (usize, usize) = (16, 9);
/// A sudden lasting jump the cut measure lets through: its grey step is at least this …
pub const JUMP_MIN: f32 = 0.015;
/// … and this many times the steps either side …
pub const JUMP_FACTOR: f32 = 4.0;
/// … and the framing either side correlates at least this much.
pub const SAME_FRAMING: f64 = 0.6;
/// A cut whose two sides' framing correlates this much, and whose colours stay this close (mean
/// RGB, a share of full scale), is a jump cut.
pub const CUT_SAME_FRAMING: f64 = 0.85;
pub const CUT_SAME_COLOUR: f64 = 0.08;
/// A burst: this many shots in a row or more …
pub const BURST_MIN_SHOTS: usize = 3;
/// … each at most this long (seconds) and this share of the median.
pub const BURST_SHOT: f64 = 1.0;
pub const BURST_SHARE: f64 = 0.6;
/// A move: motion a second at least this (and twice its shot's median).
pub const MOVE_MIN: f64 = 0.15;
/// A move has settled when its motion falls to this share of its peak.
pub const SETTLE_SHARE: f64 = 0.25;
/// The energy map's halves: motion a second, and cuts a second, at which each part is half way.
const MOTION_HALF: f64 = 0.25;
const CUTS_HALF: f64 = 0.5;
/// Stretches of the energy map and the loudness curve, at most.
const MAX_BUCKETS: usize = 60;
/// Sound is measured at this rate, in chunks this long (0.1 s).
const AUDIO_RATE: u32 = 48_000;
const AUDIO_CHUNK: usize = 4_800;
/// Most channels read.
const MAX_CHANNELS: usize = 8;
/// Palette colours listed (each at least 2 % of the picture) and frames the colour pass measures.
const PALETTE_COLOURS: usize = 6;
const LOOK_FRAMES: usize = 12;
/// Most shots, cuts and bursts listed (their counts are always given).
const MAX_SHOTS_LISTED: usize = 500;
const MAX_CUTS_LISTED: usize = 2000;
const MAX_BURSTS_LISTED: usize = 100;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Measure Editing Style",
        menu: &[],
        shortcut: None,
        params: r#"{"path":file? | "item":id? (a media item or a sequence) | "sequence":id? (exactly one),"from":seconds?,"to":seconds? (default the whole video),"fps":2..30=10 (frames analysed a second, never above the video's own rate)}"#,
        enabled: always,
        run,
        journal: false,
    }]
}

/// What is measured.
enum Input {
    File(String),
    Item(ItemId),
    Sequence(ItemId),
}

/// One analysed frame.
struct Sample {
    /// Its time, ticks (media time for a file or item, sequence time for a sequence).
    t: i64,
    window: usize,
    /// Grey difference to the window's frame before (0 for its first) and to the one two back.
    step: f32,
    skip: f32,
    /// Mean grey gradient of the sharpness picture.
    sharp: f32,
    /// The 16×9 grey framing picture.
    layout: Vec<u8>,
    /// Mean colour (sRGB, 0..1).
    colour: [f32; 3],
}

/// A shot between two cuts.
#[derive(Clone, Debug)]
struct Shot {
    start: i64,
    end: i64,
    /// Both its ends are cuts or the ends of the measured stretch (not a sampled window's edge).
    complete: bool,
    window: usize,
    /// How it begins: `start`, `cut` or `jump`.
    kind: &'static str,
}

impl Shot {
    fn seconds(&self) -> f64 {
        Tick(self.end.saturating_sub(self.start)).seconds()
    }
}

/// Everything measured, before it is reported.
struct Analysis {
    from: Tick,
    to: Tick,
    capped: bool,
    rate: FrameRate,
    native_fps: f64,
    windows: Vec<(Tick, Tick)>,
    samples: Vec<Sample>,
    failed: usize,
    errors: Vec<Value>,
    cuts: Vec<(i64, &'static str)>,
    pops: Option<usize>,
    stills: Vec<(i64, i64)>,
    motion_mean: Option<f64>,
    palette: Vec<(u64, [u64; 3])>,
    palette_pixels: u64,
    looks: Vec<color_balance::Stats>,
    loudness: Value,
}

/// The sound of the measured stretch: channels at [`AUDIO_RATE`] from a time, or why not.
type AudioFn<'a> = Box<dyn FnMut(Tick, usize) -> std::result::Result<Vec<Vec<f32>>, String> + 'a>;

pub(crate) fn r2(x: f64) -> f64 {
    if x.is_finite() { (x * 100.0).round() / 100.0 } else { 0.0 }
}

fn r1(x: f64) -> f64 {
    if x.is_finite() { (x * 10.0).round() / 10.0 } else { 0.0 }
}

/// The value at share `q` (0..1) of sorted `v` (nearest rank).
fn quantile(v: &[f64], q: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let i = ((v.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    v.get(i.min(v.len() - 1)).copied()
}

fn median_of(mut v: Vec<f64>) -> Option<f64> {
    v.retain(|x| x.is_finite());
    v.sort_by(f64::total_cmp);
    quantile(&v, 0.5)
}

/// The windows a stretch is analysed in: the whole of it when its frames fit [`MAX_FRAMES`], else
/// evenly spread windows of [`WINDOW_SECONDS`] s, the first at its start and the last at its end.
fn plan_windows(from: Tick, to: Tick, rate: FrameRate) -> Vec<(Tick, Tick)> {
    let span = to.0.saturating_sub(from.0).max(0);
    let frames = rate.frame_at(Tick(span)).saturating_add(1).max(1);
    if frames <= MAX_FRAMES as i64 {
        return vec![(from, to)];
    }
    let w = Tick::from_seconds_f64(WINDOW_SECONDS).0.min(span);
    let per = rate.frame_at(Tick(w)).saturating_add(1).max(1);
    let n = (MAX_FRAMES as i64 / per).clamp(1, 1000);
    if n == 1 {
        return vec![(from, Tick(from.0.saturating_add(w)))];
    }
    let gap = (i128::from(span) - i128::from(w)).max(0) / i128::from(n - 1);
    (0..n)
        .map(|k| {
            let a = i128::from(from.0) + gap * i128::from(k);
            let a = i64::try_from(a).unwrap_or(from.0);
            (Tick(a), Tick(a.saturating_add(w).min(to.0)))
        })
        .collect()
}

/// Block-average grey `g` (`gw`×`gh`) to the 16×9 framing picture (9×16 for a portrait one).
fn layout_of(g: &[u8], gw: usize, gh: usize) -> Vec<u8> {
    let (lw, lh) = if gw >= gh { LAYOUT } else { (LAYOUT.1, LAYOUT.0) };
    let mut sum = vec![0u32; lw * lh];
    let mut n = vec![0u32; lw * lh];
    for y in 0..gh {
        for x in 0..gw {
            let c = (y * lh / gh.max(1)).min(lh - 1) * lw + (x * lw / gw.max(1)).min(lw - 1);
            if let (Some(v), Some(s), Some(k)) = (g.get(y * gw + x), sum.get_mut(c), n.get_mut(c)) {
                *s += u32::from(*v);
                *k += 1;
            }
        }
    }
    sum.iter().zip(&n).map(|(s, k)| (s / (*k).max(1)).min(255) as u8).collect()
}

/// Zero-mean normalised correlation of two framing pictures (0 when either is flat or they do not
/// match in size).
pub(crate) fn framing(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f64;
    let ma = a.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
    let mb = b.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (dx, dy) = (f64::from(*x) - ma, f64::from(*y) - mb);
        ab += dx * dy;
        aa += dx * dx;
        bb += dy * dy;
    }
    // (a flat picture has no framing to compare)
    if aa / n < 4.0 || bb / n < 4.0 {
        return 0.0;
    }
    (ab / (aa.sqrt() * bb.sqrt())).clamp(-1.0, 1.0)
}

/// The mean colour of opaque RGBA8 (sRGB, 0..1).
fn mean_colour(rgba: &[u8]) -> [f32; 3] {
    let px = rgba.as_chunks::<4>().0;
    let n = px.len().max(1) as f64;
    let mut s = [0u64; 3];
    for p in px {
        for (a, c) in s.iter_mut().zip(p) {
            *a += u64::from(*c);
        }
    }
    s.map(|v| (v as f64 / n / 255.0) as f32)
}

/// Mean absolute difference of two mean colours (0..1).
pub(crate) fn colour_distance(a: [f32; 3], b: [f32; 3]) -> f64 {
    a.iter().zip(&b).map(|(x, y)| f64::from((x - y).abs())).sum::<f64>() / 3.0
}

/// Mean absolute grey gradient (right and down neighbours) of opaque RGBA8 `w`×`h`, a share of
/// full scale.
fn sharpness(rgba: &[u8], w: usize, h: usize) -> f32 {
    let g = grey_of(rgba);
    if w < 2 || h < 2 || g.len() < w * h {
        return 0.0;
    }
    let mut s = 0u64;
    let mut n = 0u64;
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let (Some(c), Some(r), Some(d)) = (g.get(y * w + x), g.get(y * w + x + 1), g.get((y + 1) * w + x)) else { continue };
            s += u64::from(c.abs_diff(*r)) + u64::from(c.abs_diff(*d));
            n += 2;
        }
    }
    (s as f64 / (n.max(1) as f64 * 255.0)) as f32
}

/// Opaque sRGB RGBA8 as the colour pass's picture (premultiplied linear RGBA).
fn picture_of(rgba: &[u8], w: usize, h: usize) -> filmcraft_render::Image {
    let mut img = filmcraft_render::Image::new(w, h);
    for (d, s) in img.px.as_chunks_mut::<4>().0.iter_mut().zip(rgba.as_chunks::<4>().0) {
        *d = [
            filmcraft_color::srgb_to_linear(f32::from(s[0]) / 255.0),
            filmcraft_color::srgb_to_linear(f32::from(s[1]) / 255.0),
            filmcraft_color::srgb_to_linear(f32::from(s[2]) / 255.0),
            1.0,
        ];
    }
    img
}

/// The palette bin of an sRGB colour: 4 levels a channel.
fn bin_of(p: &[u8; 4]) -> usize {
    usize::from(p[0] >> 6) * 16 + usize::from(p[1] >> 6) * 4 + usize::from(p[2] >> 6)
}

/// Analyse `[from, to)` at `rate`: `frame` gives straight RGBA8 at a time, `audio` the sound.
fn analyse(
    from: Tick,
    to: Tick,
    capped: bool,
    rate: FrameRate,
    native_fps: f64,
    mut frame: impl FnMut(Tick) -> std::result::Result<(u32, u32, Vec<u8>), String>,
    audio: Option<AudioFn<'_>>,
) -> Result<Analysis> {
    let rate = rate.sane();
    let fps = rate.as_f64();
    let native = (fps - native_fps).abs() < 1e-3;
    let windows = plan_windows(from, to, rate);
    let total_frames: usize = windows.iter().map(|(a, b)| rate.frame_at(Tick(b.0.saturating_sub(a.0))).max(0) as usize + 1).sum::<usize>().min(MAX_FRAMES);
    let look_every = total_frames.div_ceil(LOOK_FRAMES).max(1);
    let mut samples: Vec<Sample> = Vec::with_capacity(total_frames.min(MAX_FRAMES));
    let (mut failed, mut errors) = (0usize, Vec::new());
    let mut cuts: Vec<(i64, &'static str)> = Vec::new();
    let mut pops: Option<usize> = native.then_some(0);
    let mut stills = Vec::new();
    let (mut motion_sum, mut motion_n) = (0.0f64, 0usize);
    let mut palette = vec![(0u64, [0u64; 3]); 64];
    let mut palette_pixels = 0u64;
    let mut looks = Vec::new();
    let mut seen = 0usize;
    for (wi, &(start, end)) in windows.iter().enumerate() {
        let mut meter = Meter::new(fps, native);
        let first = samples.len();
        let (mut prev, mut prev2): (Option<Vec<u8>>, Option<Vec<u8>>) = (None, None);
        let mut j = 0i64;
        loop {
            if samples.len() >= MAX_FRAMES {
                break;
            }
            let t = Tick(start.0.saturating_add(rate.tick_of(j).0));
            if t >= end {
                break;
            }
            j += 1;
            let picture = frame(t).and_then(|(w, h, rgba)| {
                let grid = grid_for(w, h);
                let small = box_reduce(w, h, &rgba, grid.0, grid.1).map_err(|e| e.to_string())?;
                let sg = if w >= h { SHARP_GRID } else { (SHARP_GRID.1, SHARP_GRID.0) };
                let sharp_px = box_reduce(w, h, &rgba, sg.0, sg.1).map_err(|e| e.to_string())?;
                Ok((grid, small, sharpness(&sharp_px, sg.0, sg.1)))
            });
            let (grid, small, sharp) = match picture {
                Ok(v) => v,
                Err(e) => {
                    failed += 1;
                    if errors.len() < 20 {
                        errors.push(json!({"time": t.0, "error": e}));
                    }
                    continue;
                }
            };
            meter.push(Tick(t.0 - start.0), &small, grid);
            let grey = grey_of(&small);
            let step = prev.as_ref().map_or(0.0, |p| mean_diff(p, &grey));
            let skip = prev2.as_ref().map_or(0.0, |p| mean_diff(p, &grey));
            for p in small.as_chunks::<4>().0 {
                if let Some((n, s)) = palette.get_mut(bin_of(p)) {
                    *n += 1;
                    s[0] += u64::from(p[0]);
                    s[1] += u64::from(p[1]);
                    s[2] += u64::from(p[2]);
                }
                palette_pixels += 1;
            }
            if seen.is_multiple_of(look_every) && looks.len() < LOOK_FRAMES {
                looks.push(color_balance::measure(&picture_of(&small, grid.0, grid.1)));
            }
            seen += 1;
            samples.push(Sample { t: t.0, window: wi, step, skip, sharp, layout: layout_of(&grey, grid.0, grid.1), colour: mean_colour(&small) });
            prev2 = prev.replace(grey);
        }
        if samples.len() == first {
            continue;
        }
        // qa.motionEnergy's meter: its cuts (pops taken out), pops and still stretches
        let len = Tick(end.0.saturating_sub(start.0));
        let (findings, measures, _) = meter.finish(len, &MeterOptions { expect_motion: true, ..Default::default() });
        for c in measures["cuts"].as_array().map(Vec::as_slice).unwrap_or_default().iter().filter_map(Value::as_i64) {
            cuts.push((start.0.saturating_add(c), "cut"));
        }
        if let Some(p) = pops.as_mut() {
            *p += measures["pops"].as_u64().unwrap_or(0) as usize;
        }
        for f in findings.iter().filter(|f| f["code"] == "still") {
            if let (Some(a), Some(b)) = (f["time"].as_i64(), f["end"].as_i64()) {
                stills.push((start.0.saturating_add(a), start.0.saturating_add(b)));
            }
        }
        if let Some(m) = measures["motion"]["mean"].as_f64() {
            let n = samples.len() - first;
            motion_sum += m * n as f64;
            motion_n += n;
        }
    }
    if samples.is_empty() {
        let why = errors.first().and_then(|e| e["error"].as_str()).unwrap_or("no frame").to_string();
        return Err(bad(CMD, format!("no frame could be read: {why}")));
    }
    let frame_ticks = rate.frame_duration().0.max(1);
    // ---- jump cuts: hard cuts that keep their framing, and lasting jumps the cut measure let by
    let index_at = |t: i64| samples.binary_search_by_key(&t, |s| s.t).ok();
    for c in cuts.iter_mut() {
        if let Some(i) = index_at(c.0)
            && i > 0
            && let (Some(a), Some(b)) = (samples.get(i - 1), samples.get(i))
            && a.window == b.window
            && framing(&a.layout, &b.layout) >= CUT_SAME_FRAMING
            && colour_distance(a.colour, b.colour) <= CUT_SAME_COLOUR
        {
            c.1 = "jump";
        }
    }
    let mut jumps = Vec::new();
    for i in 1..samples.len() {
        let (Some(a), Some(b)) = (samples.get(i - 1), samples.get(i)) else { continue };
        if a.window != b.window || b.step < JUMP_MIN {
            continue;
        }
        let before = if samples.get(i.wrapping_sub(2)).is_some_and(|p| p.window == b.window) { a.step } else { 0.0 };
        let next = samples.get(i + 1).filter(|n| n.window == b.window);
        let after = next.map_or(0.0, |n| n.step);
        if b.step < JUMP_FACTOR * before.max(after).max(0.002) {
            continue;
        }
        if next.is_some_and(|n| n.skip < 0.6 * b.step) {
            continue;
        }
        if framing(&a.layout, &b.layout) < SAME_FRAMING {
            continue;
        }
        if cuts.iter().any(|c| (c.0 - b.t).abs() <= 2 * frame_ticks) {
            continue;
        }
        jumps.push((b.t, "jump"));
    }
    cuts.extend(jumps);
    cuts.sort_by_key(|c| c.0);
    cuts.dedup_by_key(|c| c.0);
    let motion_mean = (motion_n > 0).then(|| motion_sum / motion_n as f64);
    // ---- the sound, for the analysed windows only
    let span = to.0.saturating_sub(from.0).max(1);
    let bucket = (span / MAX_BUCKETS as i64).max(TICKS_PER_SECOND);
    let buckets = (span.saturating_add(bucket - 1) / bucket).clamp(1, MAX_BUCKETS as i64 + 1) as usize;
    let loudness = match audio {
        None => json!({"measured": false, "reason": "the video has no sound"}),
        Some(mut read) => measure_sound(&windows, from, bucket, buckets, &mut read).0,
    };
    Ok(Analysis {
        from,
        to,
        capped,
        rate,
        native_fps,
        windows,
        samples,
        failed,
        errors,
        cuts,
        pops,
        stills,
        motion_mean,
        palette,
        palette_pixels,
        looks,
        loudness,
    })
}

/// The loudness of the windows' sound: the summary, the momentary loudness per stretch (LUFS)
/// and the integrated loudness.
fn measure_sound(windows: &[(Tick, Tick)], from: Tick, bucket: i64, buckets: usize, read: &mut AudioFn<'_>) -> (Value, Vec<Option<f64>>, Option<f64>) {
    let mut meter: Option<filmcraft_audio_dsp::LoudnessMeter> = None;
    let mut sum = vec![(0.0f64, 0usize); buckets];
    let mut chunks = 0usize;
    let mut error = None;
    'windows: for &(a, b) in windows {
        let total = Tick(b.0.saturating_sub(a.0)).to_units_floor(i64::from(AUDIO_RATE)).max(0);
        let mut pos = 0i64;
        while pos < total {
            let n = (total - pos).min(AUDIO_CHUNK as i64) as usize;
            let at = Tick(a.0.saturating_add(Tick::from_units(pos, i64::from(AUDIO_RATE)).0));
            let channels = match read(at, n) {
                Ok(c) => c,
                Err(e) => {
                    error = Some(e);
                    break 'windows;
                }
            };
            pos += n as i64;
            let channels: Vec<Vec<f32>> = channels.into_iter().take(MAX_CHANNELS).collect();
            if channels.is_empty() || channels.iter().any(|c| c.len() != n) {
                continue;
            }
            let m = meter.get_or_insert_with(|| filmcraft_audio_dsp::LoudnessMeter::new(f64::from(AUDIO_RATE), channels.len()));
            if m.channels() != channels.len() {
                continue;
            }
            let refs: Vec<&[f32]> = channels.iter().map(Vec::as_slice).collect();
            m.process(&refs);
            chunks += 1;
            let lufs = m.momentary();
            let e = if lufs.is_finite() { 10f64.powf(lufs.max(-70.0) / 10.0) } else { 1e-7 };
            let k = (at.0.saturating_sub(from.0).max(0) / bucket.max(1)) as usize;
            if let Some(slot) = sum.get_mut(k.min(buckets - 1)) {
                slot.0 += e;
                slot.1 += 1;
            }
        }
    }
    let Some(m) = meter.filter(|_| chunks > 0) else {
        let why = error.map_or_else(|| "no sound could be read".to_string(), |e| format!("the sound could not be read: {e}"));
        return (json!({"measured": false, "reason": why}), Vec::new(), None);
    };
    let curve: Vec<Option<f64>> = sum.iter().map(|(e, n)| (*n > 0).then(|| 10.0 * (e / *n as f64).max(1e-7).log10())).collect();
    let integrated = m.integrated();
    let integrated = (integrated.is_finite() && integrated > -70.0).then_some(integrated);
    let range = m.loudness_range();
    let peak = m.true_peak_dbtp();
    (
        json!({
            "measured": true,
            "integratedLufs": integrated.map(r1),
            "rangeLu": range.is_finite().then(|| r1(range)),
            "truePeakDbtp": peak.is_finite().then(|| r1(peak)),
            "bucketSeconds": r3(Tick(bucket).seconds()),
            "momentaryLufs": curve.iter().map(|v| v.map(r1)).collect::<Vec<_>>(),
            "silent": integrated.is_none(),
        }),
        curve,
        integrated,
    )
}

/// The shots between `a.cuts`, window by window.
fn shots_of(a: &Analysis) -> Vec<Shot> {
    let mut out = Vec::new();
    for (wi, &(start, end)) in a.windows.iter().enumerate() {
        let inside: Vec<&(i64, &'static str)> = a.cuts.iter().filter(|c| c.0 > start.0 && c.0 < end.0).collect();
        let mut from = (start.0, if start == a.from { "start" } else { "window" });
        for c in inside.iter().copied().chain(std::iter::once(&(end.0, "end"))) {
            let (s, e) = (from.0, c.0);
            if e > s {
                let begins_open = from.1 == "window";
                let ends_open = c.1 == "end" && end != a.to;
                out.push(Shot { start: s, end: e, complete: !begins_open && !ends_open, window: wi, kind: if from.1 == "window" { "start" } else { from.1 } });
            }
            from = (c.0, c.1);
        }
    }
    out
}

/// Per-frame motion a second (grey change, cut steps left out) of each sample, `None` at a shot's
/// first frame.
fn motions(a: &Analysis) -> Vec<Option<f64>> {
    let fps = a.rate.as_f64();
    let mut cut_at = a.cuts.iter().map(|c| c.0).peekable();
    let mut out = Vec::with_capacity(a.samples.len());
    let mut last_window = usize::MAX;
    for s in &a.samples {
        while cut_at.peek().is_some_and(|c| *c < s.t) {
            cut_at.next();
        }
        let at_cut = cut_at.peek() == Some(&s.t);
        let first = s.window != last_window;
        last_window = s.window;
        out.push((!first && !at_cut).then(|| f64::from(s.step) * fps));
    }
    out
}

/// The bursts of `shots`: runs of at least [`BURST_MIN_SHOTS`] quick shots inside one window.
fn bursts_of(shots: &[Shot], quick: f64) -> Vec<(i64, i64, usize)> {
    let mut out = Vec::new();
    let mut run: Vec<&Shot> = Vec::new();
    let flush = |run: &mut Vec<&Shot>, out: &mut Vec<(i64, i64, usize)>| {
        if run.len() >= BURST_MIN_SHOTS
            && let (Some(f), Some(l)) = (run.first(), run.last())
        {
            out.push((f.start, l.end, run.len()));
        }
        run.clear();
    };
    for s in shots {
        if run.last().is_some_and(|l| l.window != s.window) {
            flush(&mut run, &mut out);
        }
        if s.seconds() <= quick {
            run.push(s);
        } else {
            flush(&mut run, &mut out);
        }
    }
    flush(&mut run, &mut out);
    out
}

/// The samples of shot `sh` (they are in time order): `lo..hi`.
fn span_of(a: &Analysis, sh: &Shot) -> (usize, usize) {
    (a.samples.partition_point(|s| s.t < sh.start), a.samples.partition_point(|s| s.t < sh.end))
}

/// Settle times after moves: (settle seconds of each settled move, moves cut before settling).
fn settles(a: &Analysis, shots: &[Shot], motion: &[Option<f64>]) -> (Vec<f64>, usize) {
    let mut settled = Vec::new();
    let mut on_move = 0usize;
    for sh in shots {
        let (lo, hi) = span_of(a, sh);
        let series: Vec<(i64, f64)> = (lo..hi).filter_map(|i| Some((a.samples.get(i)?.t, (*motion.get(i)?)?))).collect();
        if series.len() < 3 {
            continue;
        }
        let rest = median_of(series.iter().map(|s| s.1).collect()).unwrap_or(0.0);
        let level = MOVE_MIN.max(2.0 * rest);
        let mut k = 0usize;
        while k < series.len() {
            let Some(&(_, m)) = series.get(k) else { break };
            if m < level {
                k += 1;
                continue;
            }
            // the move's peak, then where it falls to a share of it
            let mut peak = k;
            let mut j = k;
            while let Some(&(_, v)) = series.get(j) {
                if v < level {
                    break;
                }
                if series.get(peak).is_some_and(|p| v > p.1) {
                    peak = j;
                }
                j += 1;
            }
            let (pt, pm) = series.get(peak).copied().unwrap_or((0, 0.0));
            match series.iter().skip(peak + 1).find(|s| s.1 <= SETTLE_SHARE * pm) {
                Some(&(t, _)) => settled.push(Tick(t - pt).seconds()),
                None => on_move += 1,
            }
            k = j.max(k + 1);
        }
    }
    (settled, on_move)
}

/// The motion blur estimate: (seconds-weighted mean, shots measured).
fn blur_of(a: &Analysis, shots: &[Shot], motion: &[Option<f64>]) -> (Option<f64>, usize) {
    let (mut sum, mut weight, mut n) = (0.0, 0.0, 0usize);
    for sh in shots {
        let (lo, hi) = span_of(a, sh);
        let mut frames: Vec<(f64, f64)> = (lo..hi).filter_map(|i| Some(((*motion.get(i)?)?, f64::from(a.samples.get(i)?.sharp)))).collect();
        if frames.len() < 6 {
            continue;
        }
        frames.sort_by(|x, y| x.0.total_cmp(&y.0));
        let third = frames.len() / 3;
        let calm = frames.get(..third).unwrap_or_default();
        let moving = frames.get(frames.len() - third..).unwrap_or_default();
        let mean = |v: &[(f64, f64)], k: usize| v.iter().map(|f| if k == 0 { f.0 } else { f.1 }).sum::<f64>() / v.len().max(1) as f64;
        let (cm, mm) = (mean(calm, 0), mean(moving, 0));
        let (cs, ms) = (mean(calm, 1), mean(moving, 1));
        if mm < MOVE_MIN.max(2.0 * cm) || cs <= 1e-4 {
            continue;
        }
        let b = (1.0 - ms / cs).clamp(0.0, 1.0);
        let w = sh.seconds().max(0.01);
        sum += b * w;
        weight += w;
        n += 1;
    }
    ((weight > 0.0).then(|| sum / weight), n)
}

/// Shot lengths, rates and bursts.
fn rhythm_json(shots: &[Shot], lengths: &[f64], cut_count: usize, analysed: f64) -> Value {
    let mut sorted = lengths.to_vec();
    sorted.retain(|x| x.is_finite());
    sorted.sort_by(f64::total_cmp);
    let q = |x: f64| quantile(&sorted, x).map(r2);
    let mean = (!sorted.is_empty()).then(|| sorted.iter().sum::<f64>() / sorted.len() as f64);
    let cv = mean.filter(|m| *m > 1e-9 && sorted.len() >= 2).map(|m| {
        let var = sorted.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / sorted.len() as f64;
        var.sqrt() / m
    });
    let median = quantile(&sorted, 0.5);
    let quick = median.map_or(BURST_SHOT, |m| BURST_SHOT.min(BURST_SHARE * m));
    let bursts = bursts_of(shots, quick);
    let burst_time: i64 = bursts.iter().map(|b| b.1.saturating_sub(b.0)).sum();
    let minutes = (analysed / 60.0).max(1e-9);
    json!({
        "cutsPerMinute": r2(cut_count as f64 / minutes),
        "shot": {"median": q(0.5), "p10": q(0.1), "p25": q(0.25), "p75": q(0.75), "p90": q(0.9), "mean": mean.map(r2), "cv": cv.map(r3), "count": sorted.len()},
        "bursts": {"count": bursts.len(), "perMinute": r2(bursts.len() as f64 / minutes), "share": r3(Tick(burst_time).seconds() / analysed.max(1e-9)), "quickShot": r2(quick)},
    })
}

/// The report of an analysis.
fn report(a: &Analysis) -> Value {
    let span = a.to.0.saturating_sub(a.from.0).max(1);
    let analysed: f64 = a.windows.iter().map(|(x, y)| Tick(y.0.saturating_sub(x.0)).seconds()).sum::<f64>().max(1e-9);
    let fps = a.rate.as_f64();
    let shots = shots_of(a);
    let complete: Vec<&Shot> = shots.iter().filter(|s| s.complete).collect();
    let lengths: Vec<f64> = if complete.is_empty() { shots.iter().map(Shot::seconds).collect() } else { complete.iter().map(|s| s.seconds()).collect() };
    let jump_count = a.cuts.iter().filter(|c| c.1 == "jump").count();
    let rhythm = rhythm_json(&shots, &lengths, a.cuts.len(), analysed);
    let motion = motions(a);
    // ---- the energy map
    let bucket = (span / MAX_BUCKETS as i64).max(TICKS_PER_SECOND);
    let buckets = (span.saturating_add(bucket - 1) / bucket).clamp(1, MAX_BUCKETS as i64 + 1) as usize;
    let mut acc = vec![(0.0f64, 0usize, 0usize, 0usize); buckets];
    for (s, m) in a.samples.iter().zip(&motion) {
        let k = ((s.t.saturating_sub(a.from.0)).max(0) / bucket) as usize;
        if let Some(slot) = acc.get_mut(k.min(buckets - 1)) {
            slot.2 += 1;
            if let Some(m) = m {
                slot.0 += m;
                slot.1 += 1;
            }
        }
    }
    for c in &a.cuts {
        let k = ((c.0.saturating_sub(a.from.0)).max(0) / bucket) as usize;
        if let Some(slot) = acc.get_mut(k.min(buckets - 1)) {
            slot.3 += 1;
        }
    }
    let energy: Vec<Option<f64>> = acc
        .iter()
        .map(|(ms, mn, frames, cuts)| {
            if *frames == 0 {
                return None;
            }
            let m = if *mn > 0 { ms / *mn as f64 } else { 0.0 };
            let cr = *cuts as f64 / (*frames as f64 / fps).max(1e-9);
            Some(0.6 * m / (m + MOTION_HALF) + 0.4 * cr / (cr + CUTS_HALF))
        })
        .collect();
    // ---- stillness, settle, blur
    let still_time: i64 = a.stills.iter().map(|(x, y)| y.saturating_sub(*x)).sum();
    let stillness = Tick(still_time).seconds() / analysed;
    let longest_still = a.stills.iter().map(|(x, y)| Tick(y.saturating_sub(*x)).seconds()).fold(0.0, f64::max);
    let (settled, on_move) = settles(a, &shots, &motion);
    let moves = settled.len() + on_move;
    let settle_median = median_of(settled.clone());
    let mut settled_sorted = settled.clone();
    settled_sorted.sort_by(f64::total_cmp);
    let settle_p75 = quantile(&settled_sorted, 0.75);
    let cut_on_move = (moves > 0).then(|| on_move as f64 / moves as f64);
    let (blur, blur_shots) = blur_of(a, &shots, &motion);
    // ---- the palette and the look
    let mut bins: Vec<&(u64, [u64; 3])> = a.palette.iter().filter(|b| b.0 > 0).collect();
    bins.sort_by_key(|b| std::cmp::Reverse(b.0));
    let colours: Vec<Value> = bins
        .iter()
        .filter(|b| b.0 as f64 / a.palette_pixels.max(1) as f64 >= 0.02)
        .take(PALETTE_COLOURS)
        .map(|b| {
            let c = b.1.map(|v| (v / b.0.max(1)).min(255));
            json!({"hex": format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2]), "share": r3(b.0 as f64 / a.palette_pixels.max(1) as f64)})
        })
        .collect();
    let mean_of = |f: &dyn Fn(&color_balance::Stats) -> Option<f64>| -> Option<f64> {
        let v: Vec<f64> = a.looks.iter().filter_map(f).filter(|x| x.is_finite()).collect();
        (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
    };
    let brightness = mean_of(&|s| Some(s.median));
    let contrast = mean_of(&|s| Some(s.contrast));
    let saturation = mean_of(&|s| Some(s.saturation));
    let warmth = mean_of(&|s| s.grey.map(|g| (g[0] - g[2]) / (g[0] + g[1] + g[2]).max(1e-6) * 3.0));
    let mut out = json!({
        "range": {"from": a.from.0, "to": a.to.0, "seconds": r3(Tick(span).seconds()), "capped": a.capped},
        "analysed": {
            "seconds": r3(analysed),
            "frames": a.samples.len(),
            "failed": a.failed,
            "fps": r3(fps),
            "nativeFps": r3(a.native_fps),
            "sampled": a.windows.len() > 1,
            "windows": a.windows.iter().map(|(x, y)| json!({"from": x.0, "to": y.0})).collect::<Vec<_>>(),
        },
    });
    if a.failed > 0
        && let Some(o) = out["analysed"].as_object_mut()
    {
        o.insert("errors".into(), json!(a.errors));
    }
    let extra = json!({
        "cuts": {
            "count": a.cuts.len(),
            "perMinute": rhythm["cutsPerMinute"],
            "jump": jump_count,
            "hard": a.cuts.len() - jump_count,
            "times": a.cuts.iter().take(MAX_CUTS_LISTED).map(|c| json!({"time": c.0, "seconds": r3(Tick(c.0).seconds()), "kind": c.1})).collect::<Vec<_>>(),
        },
        "shots": {
            "count": shots.len(),
            "complete": complete.len(),
            "median": rhythm["shot"]["median"], "p10": rhythm["shot"]["p10"], "p25": rhythm["shot"]["p25"],
            "p75": rhythm["shot"]["p75"], "p90": rhythm["shot"]["p90"], "mean": rhythm["shot"]["mean"], "cv": rhythm["shot"]["cv"],
            "list": shots.iter().take(MAX_SHOTS_LISTED).map(|s| json!({"start": s.start, "end": s.end, "seconds": r3(s.seconds()), "begins": s.kind, "complete": s.complete})).collect::<Vec<_>>(),
        },
        "bursts": {
            "count": rhythm["bursts"]["count"], "perMinute": rhythm["bursts"]["perMinute"], "share": rhythm["bursts"]["share"], "quickShot": rhythm["bursts"]["quickShot"],
            "list": bursts_of(&shots, rhythm["bursts"]["quickShot"].as_f64().unwrap_or(BURST_SHOT)).iter().take(MAX_BURSTS_LISTED)
                .map(|b| json!({"start": b.0, "end": b.1, "shots": b.2, "meanShot": r3(Tick(b.1.saturating_sub(b.0)).seconds() / b.2.max(1) as f64)})).collect::<Vec<_>>(),
        },
        "stillness": {"ratio": r3(stillness), "stretches": a.stills.len(), "longest": r3(longest_still)},
        "motion": {"unit": "mean grey change a second, share of full scale (0 = still)", "mean": a.motion_mean.map(r3)},
        "pops": a.pops,
        "energy": {"unit": "0 still … 1 frantic: motion and cuts a second", "bucketSeconds": r3(Tick(bucket).seconds()), "values": energy.iter().map(|v| v.map(r3)).collect::<Vec<_>>()},
        "settle": {"moves": moves, "settled": settled.len(), "medianSeconds": settle_median.map(r3), "p75Seconds": settle_p75.map(r3), "cutOnMove": cut_on_move.map(r3)},
        "motionBlur": {"estimate": blur.map(r3), "shots": blur_shots, "unit": "sharpness the most moving frames of a shot lose against its calmest, 0 crisp … 1 smeared"},
        "loudness": a.loudness,
        "palette": {"colors": colours, "brightness": brightness.map(r3), "contrast": contrast.map(r3), "saturation": saturation.map(r3), "warmth": warmth.map(r3), "framesMeasured": a.looks.len()},
        "thresholds": {
            "cutSensitivity": 50, "jumpMin": r3(f64::from(JUMP_MIN)), "jumpFactor": r3(f64::from(JUMP_FACTOR)), "sameFraming": SAME_FRAMING, "cutSameFraming": CUT_SAME_FRAMING, "cutSameColour": CUT_SAME_COLOUR,
            "burstMinShots": BURST_MIN_SHOTS, "burstShot": BURST_SHOT, "burstShare": BURST_SHARE, "moveMin": MOVE_MIN, "settleShare": SETTLE_SHARE,
            "maxFrames": MAX_FRAMES, "windowSeconds": WINDOW_SECONDS, "note": "starting values, to be tuned on real videos",
        },
    });
    if let (Some(o), Value::Object(e)) = (out.as_object_mut(), extra) {
        o.extend(e);
    }
    out
}

/// The frames a second analysed: the asked rate, never above the video's own.
fn analysis_rate(fps: f64, native: FrameRate) -> (FrameRate, f64) {
    let native = native.sane();
    let n = native.as_f64();
    if n.is_finite() && n > 0.0 && n <= fps + 1e-6 { (native, n) } else { (FrameRate::from_f64(fps).sane(), if n.is_finite() { n } else { fps }) }
}

/// `[from, to)` of a video `duration` long, checked; and whether it was cut at [`MAX_SECONDS`].
fn range_of(from: Option<f64>, to: Option<f64>, duration: Tick) -> Result<(Tick, Tick, bool)> {
    if duration.0 <= 0 {
        return Err(bad(CMD, "the video is empty: nothing to measure"));
    }
    let a = Tick::from_seconds_f64(from.unwrap_or(0.0));
    let b = to.map_or(duration, |t| Tick::from_seconds_f64(t).min(duration));
    if a >= b {
        return Err(bad(CMD, "`from` must come before `to`, inside the video"));
    }
    let cap = Tick(a.0.saturating_add(Tick::from_seconds_f64(MAX_SECONDS).0));
    Ok((a, b.min(cap), b > cap))
}

fn seconds_p(p: &Value, key: &str) -> Result<Option<f64>> {
    match p.get(key).filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(v) => v
            .as_f64()
            .filter(|x| x.is_finite() && (0.0..=MAX_SECONDS * 4.0).contains(x))
            .map(Some)
            .ok_or_else(|| bad(CMD, format!("`{key}` is seconds, 0 or more"))),
    }
}

/// Measure what `p` names (see the module docs).
pub fn measure(s: &Session, p: &Value) -> Result<Value> {
    if !p.is_object() {
        return Err(bad(CMD, "parameters must be an object: {path} or {item}"));
    }
    if let Some(k) = p.as_object().and_then(|o| o.keys().find(|k| !matches!(k.as_str(), "path" | "item" | "sequence" | "from" | "to" | "fps"))) {
        return Err(bad(CMD, format!("unknown parameter `{}`: path, item, sequence, from, to or fps", k.chars().take(64).collect::<String>())));
    }
    let path = match p.get("path").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(v.as_str().map(str::trim).filter(|s| !s.is_empty() && s.len() <= 4096).ok_or_else(|| bad(CMD, "`path` is a video file"))?.to_string()),
    };
    let id = |k: &str| -> Result<Option<ItemId>> {
        match p.get(k).filter(|v| !v.is_null()) {
            None => Ok(None),
            Some(v) => Ok(Some(ItemId(v.as_u64().ok_or_else(|| bad(CMD, format!("`{k}` is an item id")))?))),
        }
    };
    let input = match (path, id("item")?, id("sequence")?) {
        (Some(path), None, None) => Input::File(crate::export_tools::expand_home(&path)),
        (None, Some(item), None) => {
            if s.project.sequence(item).is_some() {
                Input::Sequence(item)
            } else {
                Input::Item(item)
            }
        }
        (None, None, Some(seq)) => Input::Sequence(seq),
        _ => return Err(bad(CMD, "give exactly one of `path` (a video file), `item` (a media item or a sequence) or `sequence`")),
    };
    let (from, to) = (seconds_p(p, "from")?, seconds_p(p, "to")?);
    let fps = match p.get("fps").filter(|v| !v.is_null()) {
        None => DEFAULT_FPS,
        Some(v) => v
            .as_f64()
            .filter(|f| f.is_finite() && (FPS_RANGE.0..=FPS_RANGE.1).contains(f))
            .ok_or_else(|| bad(CMD, format!("`fps` is frames a second, {} to {}", FPS_RANGE.0, FPS_RANGE.1)))?,
    };
    match input {
        Input::File(path) => {
            let src = s.media.open_file(&path, s.services.as_ref()).map_err(|e| bad(CMD, e.to_string()))?;
            let mut v = measure_source(&src, from, to, fps)?;
            v["source"] = json!("file");
            v["path"] = json!(path);
            Ok(v)
        }
        Input::Item(item) => {
            let it = s.project.item(item).ok_or_else(|| bad(CMD, format!("no item {}", item.0)))?;
            if !it.has_video() {
                return Err(bad(CMD, format!("item {} has no picture", item.0)));
            }
            let src = crate::vr360::analysis_view(s, item).ok_or_else(|| bad(CMD, "the item's media is offline or unreadable"))?;
            let mut v = measure_source(&src, from, to, fps)?;
            v["source"] = json!("item");
            v["item"] = json!(item.0);
            Ok(v)
        }
        Input::Sequence(seq) => {
            let mut v = measure_sequence(s, seq, from, to, fps)?;
            v["source"] = json!("sequence");
            v["sequence"] = json!(seq.0);
            Ok(v)
        }
    }
}

/// Measure an opened file or item.
fn measure_source(src: &filmcraft_media::SharedSource, from: Option<f64>, to: Option<f64>, fps: f64) -> Result<Value> {
    let info = src.info();
    let video = info.video.as_ref().ok_or_else(|| bad(CMD, "the video has no picture"))?;
    let (w, h) = (video.width.max(1), video.height.max(1));
    let (a, b, capped) = range_of(from, to, info.duration)?;
    let (rate, native) = analysis_rate(fps, video.frame_rate);
    let scale = (LONG_SIDE / f64::from(w.max(h))).min(1.0) as f32;
    let audio: Option<AudioFn<'_>> = (!info.audio_streams.is_empty()).then(|| {
        let read: AudioFn<'_> = Box::new(move |t: Tick, n: usize| {
            let start = t.to_units_floor(i64::from(AUDIO_RATE));
            src.audio(start, n, AUDIO_RATE).map(|b| b.channels).map_err(|e| e.to_string())
        });
        read
    });
    let analysis = analyse(
        a,
        b,
        capped,
        rate,
        native,
        |time| {
            let f = src.video_frame(filmcraft_media::FrameRequest { time, scale }).map_err(|e| e.to_string())?;
            Ok((f.width, f.height, f.to_rgba8()))
        },
        audio,
    )?;
    let mut v = report(&analysis);
    v["reader"] = json!("media pool decoders (hardware where the host registered them)");
    Ok(v)
}

/// Measure sequence `seq` as rendered (GPU first) and mixed.
fn measure_sequence(s: &Session, seq: ItemId, from: Option<f64>, to: Option<f64>, fps: f64) -> Result<Value> {
    let project = s.project.clone();
    let q = project.sequence(seq).ok_or_else(|| bad(CMD, "no such sequence"))?;
    q.check_bounds().map_err(|e| bad(CMD, e))?;
    let (a, b, capped) = range_of(from, to, q.duration())?;
    let (w, h) = (q.settings.width.max(1), q.settings.height.max(1));
    let (rate, native) = analysis_rate(fps, q.settings.frame_rate);
    let provider = s.media.provider(project.clone(), s.services.clone());
    let mut gpu = filmcraft_export::frame_renderer();
    let (mut on_gpu, mut on_cpu) = (0usize, 0usize);
    let scale = (LONG_SIDE / f64::from(w.max(h))).min(1.0) as f32;
    let opts = filmcraft_render::RenderOptions { scale, ..Default::default() };
    let has_sound = q.audio_tracks.iter().any(|t| !t.items.is_empty());
    let layout = filmcraft_render::audio::master_layout(q);
    let stream = filmcraft_render::MixStream::new();
    let sr = i64::from(q.settings.sample_rate.clamp(8_000, 384_000));
    let audio: Option<AudioFn<'_>> = has_sound.then(|| {
        let project = project.clone();
        let provider = &provider;
        let stream = &stream;
        let read: AudioFn<'_> = Box::new(move |t: Tick, n: usize| {
            let Some(q) = project.sequence(seq) else { return Err("no such sequence".to_string()) };
            // the mix at the sequence's own rate, resampled by nearest sample to the meter's
            let frames = ((n as i64).saturating_mul(sr) / i64::from(AUDIO_RATE)).clamp(1, 1 << 20) as usize;
            let b = filmcraft_render::audio::mix_sequence_layout(
                &project,
                q,
                t.to_units_floor(sr),
                frames,
                provider,
                layout,
                filmcraft_audio_dsp::channels::Mixdown::FrontRear,
                stream,
            );
            Ok(b.channels.iter().map(|c| (0..n).map(|i| c.get(i * frames / n.max(1)).copied().unwrap_or(0.0)).collect()).collect())
        });
        read
    });
    let analysis = analyse(
        a,
        b,
        capped,
        rate,
        native,
        |t| {
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
        },
        audio,
    )?;
    let mut v = report(&analysis);
    v["renderer"] = json!({"gpuFrames": on_gpu, "cpuFrames": on_cpu, "captions": false});
    Ok(v)
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    measure(s, p)
}

#[cfg(test)]
#[path = "style_measure_tests.rs"]
mod tests;
