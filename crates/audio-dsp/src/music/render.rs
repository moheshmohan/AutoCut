//! Rendering a score: every part through its instrument (blocks split at each note event, so
//! timing is sample-exact) into four stems, each with its own reverb; the stems summed into the
//! mix; the mix loudness-normalised and soft-limited by one per-sample gain that is applied to
//! the stems as well, so the stems still sum to the mix.
//!
//! The stems of each stretch are rendered side by side on scoped threads where the platform has
//! them (one thread otherwise, as on the web); the result does not depend on it. Memory is the
//! output (the mix, plus the stems when asked for) and a few blocks per stem.

use crate::AudioEffect;
use crate::loudness::{LoudnessMeter, normalize_gain_db};
use crate::synth::{MUSIC_LUFS, Stereo, SynthError, soft_limit};

use super::Plan;
use super::compose::{Score, Stem};
use super::instruments::{Bus, Context, Instrument, InstrumentBank};
use super::taxonomy::Genre;
use super::tempo::sample_of;

/// Samples rendered at a time (a stretch: each stem renders a stretch, then they are mixed).
const BLOCK: usize = 16_384;
/// The last moment of the track fades over this long at most (no click at the very end).
const END_FADE_SECONDS: f64 = 0.03;

/// How the stems are rendered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Options {
    /// Keep the four stems (else only the mix).
    pub stems: bool,
    /// Linear gain of each stem ([`Stem::index`] order); 0 mutes it.
    pub gains: [f32; 4],
}

/// The rendered audio.
pub(crate) struct Rendered {
    pub mix: Stereo,
    pub stems: Vec<(Stem, Stereo)>,
}

#[derive(Clone, Copy, Debug)]
struct Event {
    at: u64,
    on: bool,
    key: u8,
    velocity: f32,
}

struct Player {
    inst: Box<dyn Instrument>,
    events: Vec<Event>,
    next: usize,
    pump: Vec<f32>,
}

/// A zeroed buffer of `n` samples, or an error instead of an abort when memory runs out.
fn zeros(n: usize) -> Result<Vec<f32>, SynthError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|e| SynthError(format!("cannot allocate {n} samples of audio: {e}")))?;
    v.resize(n, 0.0);
    Ok(v)
}

/// Reverb decay (s) and room size (%) of a genre.
fn room(genre: Genre) -> (f32, f32) {
    match genre {
        Genre::Cinematic => (3.0, 85.0),
        Genre::Electronic => (2.2, 70.0),
        Genre::Pop => (1.6, 55.0),
        Genre::Ambient => (4.5, 95.0),
        Genre::Lofi => (1.2, 40.0),
        Genre::Piano => (2.6, 75.0),
    }
}

/// The genre's mix: each stem's level (dB, [`Stem::index`] order) next to the others.
fn trim(genre: Genre) -> [f64; 4] {
    match genre {
        Genre::Electronic => [0.0, 3.0, 0.0, 0.0],
        Genre::Piano => [0.0, 0.0, 2.0, -2.0],
        Genre::Pop | Genre::Ambient => [0.0, 0.0, 2.0, 0.0],
        Genre::Cinematic | Genre::Lofi => [0.0; 4],
    }
}

fn new_reverb(sr: f32, genre: Genre) -> Result<Box<dyn AudioEffect>, SynthError> {
    let (decay, size) = room(genre);
    let mut fx = crate::effects::create_effect("reverb", sr, 2).ok_or_else(|| SynthError("the reverb is missing".into()))?;
    fx.set_param("decay", decay);
    fx.set_param("size", size);
    fx.set_param("damping", 55.0);
    fx.set_param("predelay", 15.0);
    fx.set_param("mix", 100.0);
    fx.reset();
    Ok(fx)
}

/// Four buffers of a stretch.
struct Scratch {
    dry_l: Vec<f32>,
    dry_r: Vec<f32>,
    send_l: Vec<f32>,
    send_r: Vec<f32>,
}

impl Scratch {
    fn new() -> Self {
        Scratch { dry_l: vec![0.0; BLOCK], dry_r: vec![0.0; BLOCK], send_l: vec![0.0; BLOCK], send_r: vec![0.0; BLOCK] }
    }

    fn clear(&mut self, len: usize) {
        for b in [&mut self.dry_l, &mut self.dry_r, &mut self.send_l, &mut self.send_r] {
            b.iter_mut().take(len).for_each(|x| *x = 0.0);
        }
    }

    /// The bus over samples `from..to` of the stretch.
    fn bus(&mut self, from: usize, to: usize) -> Option<Bus<'_>> {
        Some(Bus {
            dry_l: self.dry_l.get_mut(from..to)?,
            dry_r: self.dry_r.get_mut(from..to)?,
            send_l: self.send_l.get_mut(from..to)?,
            send_r: self.send_r.get_mut(from..to)?,
        })
    }
}

/// One stem's players, its stretch of audio and (with stems kept) its own reverb.
struct Group {
    players: Vec<Player>,
    out: Scratch,
    part: Scratch,
    reverb: Option<Box<dyn AudioEffect>>,
}

/// The beat grid the side-chain pumps on, in samples.
#[derive(Clone, Copy)]
struct Grid {
    beat: f64,
    bar: f64,
}

/// Play `p`'s events and audio for samples `b0..b0 + len` into `out`.
fn play(p: &mut Player, b0: u64, len: usize, out: &mut Scratch) {
    let end = b0.saturating_add(len as u64);
    let mut at = b0;
    // every pass consumes an event or reaches the end; bounded anyway
    for _ in 0..=p.events.len().saturating_add(1) {
        while let Some(ev) = p.events.get(p.next).copied().filter(|e| e.at <= at) {
            if ev.on {
                p.inst.note_on(ev.key, ev.velocity);
            } else {
                p.inst.note_off(ev.key);
            }
            p.next += 1;
        }
        let next_at = p.events.get(p.next).map_or(end, |e| e.at.min(end)).max(at);
        if next_at > at
            && let Some(mut bus) = out.bus((at - b0) as usize, (next_at - b0) as usize)
        {
            p.inst.render(&mut bus);
        }
        at = next_at;
        if at >= end {
            break;
        }
    }
}

/// Render `g`'s stretch `b0..b0 + len`: its players (pumped ones ducked on the beat), then its
/// reverb into the send (which then holds the wet sound).
fn render_group(g: &mut Group, b0: usize, len: usize, grid: Grid) {
    g.out.clear(len);
    for p in &mut g.players {
        if p.pump.iter().all(|d| *d <= 0.0) {
            play(p, b0 as u64, len, &mut g.out);
            continue;
        }
        g.part.clear(len);
        play(p, b0 as u64, len, &mut g.part);
        // side-chain: ducked on every beat, back by half a beat
        let ps = &g.part;
        let src = ps.dry_l.iter().zip(&ps.dry_r).zip(ps.send_l.iter().zip(&ps.send_r));
        let o = &mut g.out;
        let dst = o.dry_l.iter_mut().zip(o.dry_r.iter_mut()).zip(o.send_l.iter_mut().zip(o.send_r.iter_mut()));
        for (i, (((dl, dr), (sl, sr_)), ((xl, xr), (yl, yr)))) in dst.zip(src).take(len).enumerate() {
            let at = (b0 + i) as f64;
            let depth = p.pump.get((at / grid.bar) as usize).copied().unwrap_or(0.0);
            let since = (at % grid.beat) / grid.beat;
            let d = (1.0 - f64::from(depth) * (1.0 - (since * 2.0).min(1.0)).powi(2)) as f32;
            *dl += xl * d;
            *dr += xr * d;
            *sl += yl * d;
            *sr_ += yr * d;
        }
    }
    if let (Some(r), Some(sl), Some(sr_)) = (g.reverb.as_mut(), g.out.send_l.get_mut(..len), g.out.send_r.get_mut(..len)) {
        r.process(&mut [sl, sr_]);
    }
}

/// Whether stems can be rendered on threads here (not on the web).
fn threads_available() -> bool {
    std::thread::available_parallelism().is_ok_and(|n| n.get() >= 2)
}

/// Render the stretch of every group, side by side when `parallel`.
fn render_groups(groups: &mut [Group], b0: usize, len: usize, grid: Grid, parallel: bool) -> Result<(), SynthError> {
    if !parallel {
        groups.iter_mut().for_each(|g| render_group(g, b0, len, grid));
        return Ok(());
    }
    std::thread::scope(|scope| {
        let mut rest = groups.iter_mut();
        let first = rest.next();
        let mut handles = Vec::new();
        for g in rest {
            let h = std::thread::Builder::new()
                .name("music-stem".into())
                .spawn_scoped(scope, move || render_group(g, b0, len, grid))
                .map_err(|e| SynthError(format!("cannot start a render thread: {e}")))?;
            handles.push(h);
        }
        if let Some(g) = first {
            render_group(g, b0, len, grid);
        }
        let mut failed = false;
        for h in handles {
            failed |= h.join().is_err();
        }
        if failed { Err(SynthError("a render thread failed".into())) } else { Ok(()) }
    })
}

/// Render `score` (planned by `plan`) on `bank`.
pub(crate) fn render(plan: &Plan, score: &Score, bank: &dyn InstrumentBank, opts: &Options) -> Result<Rendered, SynthError> {
    let n = usize::try_from(plan.samples).map_err(|_| SynthError("the track is too long".into()))?;
    let sr = plan.sample_rate;
    let srf = f64::from(sr);
    let ctx = Context { sample_rate: sr, tempo: plan.tempo, genre: plan.genre, seed: plan.seed };
    let trims = trim(plan.genre).map(|db| 10f64.powf(db / 20.0) as f32);
    let gain = |s: usize| opts.gains.get(s).copied().filter(|g| g.is_finite() && *g > 0.0).unwrap_or(0.0) * trims.get(s).copied().unwrap_or(1.0);

    let mut mix = [zeros(n)?, zeros(n)?];
    let mut stems: Vec<[Vec<f32>; 2]> = Vec::new();
    if opts.stems {
        for _ in Stem::ALL {
            stems.push([zeros(n)?, zeros(n)?]);
        }
    }

    // the stems that sound, each with its players
    let mut groups: Vec<Group> = Vec::new();
    let mut group_stem: Vec<usize> = Vec::new();
    for stem in Stem::ALL {
        let s = stem.index();
        if gain(s) == 0.0 {
            continue;
        }
        let mut players = Vec::new();
        for part in score.parts.iter().filter(|p| p.stem == stem) {
            let mut events = Vec::with_capacity(part.notes.len().saturating_mul(2));
            for note in &part.notes {
                events.push(Event { at: note.start, on: true, key: note.key, velocity: note.velocity });
                events.push(Event { at: note.start.saturating_add(note.len.max(1)), on: false, key: note.key, velocity: 0.0 });
            }
            // at the same sample a release comes before a new note
            events.sort_by_key(|e| (e.at, e.on));
            players.push(Player { inst: bank.instrument(part.patch, &ctx)?, events, next: 0, pump: part.pump.clone() });
        }
        // each stem has its own room; the mix alone shares one (a reverb is linear, so the sum
        // is the same music, to rounding, for a quarter of the work)
        let reverb = if opts.stems { Some(new_reverb(srf as f32, plan.genre)?) } else { None };
        groups.push(Group { players, out: Scratch::new(), part: Scratch::new(), reverb });
        group_stem.push(s);
    }
    let mut shared = if opts.stems { None } else { Some(new_reverb(srf as f32, plan.genre)?) };
    let mut room_send = Scratch::new();

    // the final chord's room dies with it
    let hit = usize::try_from(sample_of(plan.ending, sr).min(plan.samples)).unwrap_or(n);
    let tail_seconds = n.saturating_sub(hit) as f64 / srf;
    let end_decay = ((tail_seconds * 0.6) as f32).clamp(0.3, room(plan.genre).0);
    let beat = (srf * plan.tempo.beat_seconds()).max(1.0);
    let grid = Grid { beat, bar: beat * 4.0 };
    let parallel = groups.len() > 1 && threads_available();

    let mut b0 = 0usize;
    let mut ending = false;
    while b0 < n {
        let mut b1 = b0.saturating_add(BLOCK).min(n);
        if b0 < hit && hit < b1 {
            b1 = hit;
        }
        if !ending && b0 >= hit {
            ending = true;
            for r in groups.iter_mut().filter_map(|g| g.reverb.as_mut()).chain(shared.as_mut()) {
                r.set_param("decay", end_decay);
            }
        }
        let len = b1 - b0;
        render_groups(&mut groups, b0, len, grid, parallel)?;
        let [all_l, all_r] = &mut mix;
        let (Some(mix_l), Some(mix_r)) = (all_l.get_mut(b0..b1), all_r.get_mut(b0..b1)) else { break };
        if shared.is_some() {
            room_send.clear(len);
        }
        // mixed in stem order, so the sum never depends on the threads
        for (g, s) in groups.iter().zip(&group_stem) {
            let gn = gain(*s);
            let o = &g.out;
            let lanes = o.dry_l.iter().zip(&o.dry_r).zip(o.send_l.iter().zip(&o.send_r)).take(len);
            if shared.is_some() {
                // the dry sound into the mix now, the send into the shared room
                let rs = room_send.send_l.iter_mut().zip(room_send.send_r.iter_mut());
                for ((((dl, dr), (sl, sr_)), (ml, mr)), (rl, rr)) in lanes.zip(mix_l.iter_mut().zip(mix_r.iter_mut())).zip(rs) {
                    *ml += dl * gn;
                    *mr += dr * gn;
                    *rl += sl * gn;
                    *rr += sr_ * gn;
                }
                continue;
            }
            let Some([st_l, st_r]) = stems.get_mut(*s) else { continue };
            let (Some(st_l), Some(st_r)) = (st_l.get_mut(b0..b1), st_r.get_mut(b0..b1)) else { continue };
            for ((((dl, dr), (wl, wr)), (ml, mr)), (xl, xr)) in lanes.zip(mix_l.iter_mut().zip(mix_r.iter_mut())).zip(st_l.iter_mut().zip(st_r.iter_mut())) {
                let l = (dl + wl) * gn;
                let r = (dr + wr) * gn;
                *xl = l;
                *xr = r;
                *ml += l;
                *mr += r;
            }
        }
        if let (Some(r), Some(sl), Some(sr_)) = (shared.as_mut(), room_send.send_l.get_mut(..len), room_send.send_r.get_mut(..len)) {
            r.process(&mut [&mut *sl, &mut *sr_]);
            for ((ml, mr), (wl, wr)) in mix_l.iter_mut().zip(mix_r.iter_mut()).zip(sl.iter().zip(sr_.iter())) {
                *ml += wl;
                *mr += wr;
            }
        }
        b0 = b1;
    }

    // one gain for the loudness, the limiter per sample, the last few milliseconds to silence
    let mut meter = LoudnessMeter::new(srf, 2);
    meter.process(&[&mix[0], &mix[1]]);
    let gain_db = normalize_gain_db(meter.integrated(), MUSIC_LUFS).clamp(-40.0, 40.0);
    let g = 10f64.powf(gain_db / 20.0) as f32;
    let fade = ((END_FADE_SECONDS * srf) as usize).min(n / 4).max(1);
    let ring = n.saturating_sub(hit);
    for (c, channel) in mix.iter_mut().enumerate() {
        for (i, m) in channel.iter_mut().enumerate() {
            let x = *m;
            let gx = g * x;
            let mut f = if gx.abs() > 1e-20 { soft_limit(gx) / x } else { g };
            // the final chord dies away over the ring-out (−26 dB by the end), then the last
            // few milliseconds close to silence
            if i > hit && ring > 0 {
                let x = (i - hit) as f64 / ring as f64;
                f *= (-3.0 * x * x).exp() as f32;
            }
            let left = n - i;
            if left < fade {
                f *= super::arrange::smoothstep(left as f64 / fade as f64) as f32;
            }
            if !f.is_finite() {
                f = 0.0;
            }
            if opts.stems {
                let mut sum = 0.0f32;
                for st in &mut stems {
                    if let Some(y) = st.get_mut(c).and_then(|ch| ch.get_mut(i)) {
                        *y *= f;
                        if !y.is_finite() {
                            *y = 0.0;
                        }
                        sum += *y;
                    }
                }
                *m = sum;
            } else {
                *m = x * f;
            }
            if !m.is_finite() {
                *m = 0.0;
            }
        }
    }
    let [left, right] = mix;
    let out_stems = stems.into_iter().zip(Stem::ALL).map(|([l, r], s)| (s, Stereo { left: l, right: r, sample_rate: sr })).collect();
    Ok(Rendered { mix: Stereo { left, right, sample_rate: sr }, stems: out_stems })
}
