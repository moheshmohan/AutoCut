//! Music fitted to a film (`clip.remix` with film events, an ending and a stretch): the track's
//! real lifts land on the film's events, it ends the way asked, every join sits in a stable
//! stretch with a short equal-power crossfade, and a join that changes the level more than the
//! music itself does is refused after rendering. The fitting rules follow the ideas of
//! huashu-motion's music fitting (MIT), written fresh here.
//!
//! ```text
//! events:   burst, peak, landing, stop at film times (samples from the clip's start); each is
//!           offered real lifts of the track (crate::structure): drops, section starts that rise,
//!           section boundaries, downbeats, stops and the track's own ending, weighted by kind
//!           (a burst or a peak wants a drop, a landing a boundary or the ending, a stop a stop);
//!           nothing is invented: an event no lift reaches is reported, not forced
//! anchors:  a lift ℓ landing on an event at film time t means source position ℓ plays at t.
//!           Between two anchors (t_a, x_a) → (t_b, x_b) the film has D = t_b − t_a and the track
//!           x_b − x_a: within half a beat they need no join (the difference is how far the event
//!           is off); otherwise joins leave the track at beat i and continue at beat j, so the
//!           jumps sum to the difference (one to skip, one per stretch of track to repeat),
//!           placed evenly (or, when none fits there, up to 64 beats away) and chosen by the beat
//!           self-similarity of their contexts (Analysis::jump_score), the last landing the anchor
//!           within half a beat
//! joins:    first in stable stretches that keep the bars whole (both beats inside one section,
//!           the four pairs of beats around them within 4 dB of each other, the same place in the bar,
//!           away from the ending); then in stable stretches across the bar; then anywhere, each
//!           step costing more
//! ending:   source (as before: the remix ends with the source), own (the track's own ending
//!           ends the film: its music end at the film's end, or its final hit on a landing),
//!           button (a strong downbeat at the end, or on a landing, then a short tail faded out:
//!           a beat, 0.25–0.8 s), fade (wherever the music is at the end, faded out over two bars,
//!           2–6 s, or from a landing to the end); a stop as the last event cuts the music there
//! search:   a shortest path over the events in film order (start → each event's candidate lifts
//!           or a miss → the end), costs in joins: a join 0.8 plus 1.2·(1 − similarity), 0.6 more
//!           across the bar, 1.5 more outside a stable stretch; a lift 3·(1 − its weight); a miss
//!           2–3; starting the track mid-way for an opening event 0.8 (so a drop or rising
//!           section lands on it exactly: the film's beat grid is kept when the event is on it;
//!           a 10 ms fade-in); an anchor off
//!           by r, (r / tolerance)²; ½ per 3 dB of the level jump a join is predicted to make
//!           from the beats' levels (a stable stretch does not try one over 6 dB)
//! stretch:  up to maxStretch the whole remix is time-stretched (pitch kept, at playback) so its
//!           end lands on the film's end exactly, or, with events, so the anchors fit best
//!           (least squares)
//! check:    each join is rendered; the RMS of the half second after it against the half second
//!           before is the join's level change, set against the change the music makes at each of
//!           its two cut points; more than 6 dB away from both refuses the join and the planner
//!           tries again without it and its neighbours (8 beats either side; six rounds); a plan
//!           that still fails is kept with a warning
//! ```
//!
//! Without events, ending or stretch the plan is [`remix::plan`]'s (the Remix of before), with the
//! join check on top. Positions are samples of the analysed signal; everything is deterministic.

use crate::remix::{self, Analysis, MIN_PIECE_BEATS, Params, Piece, Plan, RemixError, XFADE_S};
use crate::structure::{EndingKind, FLOOR_DB, Structure};

/// Most film events a fit takes.
pub const MAX_EVENTS: usize = 32;
/// The most a fit may time-stretch the music, percent.
pub const MAX_STRETCH_PCT: f64 = 25.0;
/// A join's level is compared over this much before and after it, seconds.
pub const JOIN_WINDOW_S: f64 = 0.5;
/// A join may change the level this much more than the music does at its cut points, dB.
pub const JOIN_MAX_DB: f64 = 6.0;
/// Rounds of planning, checking and planning again without the joins that failed.
const ROUNDS: usize = 6;
/// Lifts offered to each event.
const CANDIDATES: usize = 6;
/// A stable stretch: the beats either side within this many dB of each other.
const STABLE_DB: f32 = 4.0;
/// Most events skipped between two anchors the search considers.
const MAX_SKIP: usize = 4;
/// Most joins between two anchors.
const MAX_SPAN_JOINS: usize = 256;
/// When no join is found near its even place, the search looks this many beats either side.
const WIDE_BEATS: usize = 64;
/// A button's tail: a beat, within these seconds.
const BUTTON_TAIL_S: (f64, f64) = (0.25, 0.8);
/// A fade ending: two bars, within these seconds (a landing's fade: within the wider range).
const FADE_S: (f64, f64) = (2.0, 6.0);
const LANDING_FADE_S: (f64, f64) = (1.0, 10.0);
/// A track started mid-way fades in over this long, seconds.
const COLD_FADE_S: f64 = 0.010;
/// A cut-short own ending fades over this long, seconds.
const CUT_SHORT_FADE_S: f64 = 0.3;
// costs, in joins
const JOIN_COST: f64 = 0.8;
const SCORE_COST: f64 = 1.2;
const PHASE_COST: f64 = 0.6;
const UNSTABLE_COST: f64 = 1.5;
const LIFT_COST: f64 = 3.0;
const COLD_COST: f64 = 0.8;
const RESIDUAL_COST: f64 = 1.0;
/// Per 3 dB of predicted level jump at a join (beyond 2 dB).
const LEVEL_COST: f64 = 0.5;
/// A join refused by the level check keeps the planner this many beats away (each side).
const AVOID_BEATS: usize = 8;

/// Reads planar source samples `[src, src + len)` (silence outside the source).
pub type Reader<'a> = dyn FnMut(i64, usize) -> Vec<Vec<f32>> + 'a;

/// What happens in the film at an event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EventKind {
    /// The film bursts into life (an opening, a reveal): a drop or a rising section start.
    Burst,
    /// A high point (a hero shot, a title hit): the strongest lift.
    Peak,
    /// The film lands (an end card): a section boundary, or the ending itself.
    Landing,
    /// A hard cut to silence: a stop in the track, or the music cut there.
    Stop,
}

impl EventKind {
    pub fn id(self) -> &'static str {
        match self {
            EventKind::Burst => "burst",
            EventKind::Peak => "peak",
            EventKind::Landing => "landing",
            EventKind::Stop => "stop",
        }
    }
    pub fn parse(s: &str) -> Option<EventKind> {
        Some(match s {
            "burst" => EventKind::Burst,
            "peak" => EventKind::Peak,
            "landing" => EventKind::Landing,
            "stop" => EventKind::Stop,
            _ => return None,
        })
    }
    fn miss_cost(self) -> f64 {
        match self {
            EventKind::Burst | EventKind::Peak => 2.5,
            EventKind::Landing => 3.0,
            EventKind::Stop => 2.0,
        }
    }
}

/// A film event: `kind` at output sample `at` (from the clip's start).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Event {
    pub at: i64,
    pub kind: EventKind,
}

/// How the fitted music ends.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Ending {
    /// As the plain remix: the last piece runs to the end of the source.
    #[default]
    Source,
    /// The track's own ending ends the film.
    Own,
    /// A strong downbeat, then a short tail faded out.
    Button,
    /// Faded out at the end.
    Fade,
}

impl Ending {
    pub fn id(self) -> &'static str {
        match self {
            Ending::Source => "source",
            Ending::Own => "own",
            Ending::Button => "button",
            Ending::Fade => "fade",
        }
    }
    pub fn parse(s: &str) -> Option<Ending> {
        Some(match s {
            "source" => Ending::Source,
            "own" => Ending::Own,
            "button" => Ending::Button,
            "fade" => Ending::Fade,
            _ => return None,
        })
    }
}

/// What to fit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Request {
    /// The film's length, output samples.
    pub target: i64,
    /// The Remix sliders (Segments applies to the plain remix only; Variations is how far a join
    /// may move from its even place).
    pub params: Params,
    pub events: Vec<Event>,
    pub ending: Ending,
    /// The most the music may be time-stretched, a fraction (0.05: 5 %; 0: never).
    pub max_stretch: f64,
}

/// What kind of lift landed on an event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiftKind {
    Drop,
    Section,
    Downbeat,
    /// The track's own ending (its final hit, fade start or stop).
    Ending,
    /// A stop in the track.
    Stop,
    /// The music cut by the remix (a stop event on a downbeat).
    Cut,
}

impl LiftKind {
    pub fn id(self) -> &'static str {
        match self {
            LiftKind::Drop => "drop",
            LiftKind::Section => "section",
            LiftKind::Downbeat => "downbeat",
            LiftKind::Ending => "ending",
            LiftKind::Stop => "stop",
            LiftKind::Cut => "cut",
        }
    }
}

/// How an event came out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Landed {
    pub kind: EventKind,
    /// The event's film time, output samples.
    pub at: i64,
    /// The lift that plays nearest it (within a bar), and where (output samples, stretched).
    pub lift: Option<LiftKind>,
    pub plays_at: Option<i64>,
    /// `plays_at − at` (positive: late).
    pub off_by: Option<i64>,
    /// Within half a beat.
    pub landed: bool,
}

/// A join of the plan, as checked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Join {
    /// Output position (unstretched samples).
    pub at: i64,
    /// Where the track is left and where it continues (source samples).
    pub out: i64,
    pub into: i64,
    /// Similarity of the contexts, 0 … 1 (`None`: not at beats).
    pub score: Option<f32>,
    /// Both sides in stable stretches; the bars kept whole.
    pub stable: bool,
    pub keeps_bars: bool,
    /// Level of the half second after against the half second before, dB.
    pub change_db: f64,
    /// How far that is from what the music does at the nearer of its two cut points, dB.
    pub excess_db: f64,
    pub ok: bool,
}

/// A film length that needs no join.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Alternative {
    /// Output samples.
    pub len: i64,
    pub joins: usize,
    pub ending: Ending,
    /// With a stretch: the shortest and longest film this length plays to without a join.
    pub min: i64,
    pub max: i64,
}

/// A fitted remix.
#[derive(Clone, Debug, PartialEq)]
pub struct Fit {
    pub plan: Plan,
    /// Plan samples per output sample (1: not stretched).
    pub stretch: f64,
    /// Output length, samples (after the stretch).
    pub len: i64,
    pub ending: Ending,
    /// How it ends: `source`, `cadence`, `fade` or `cut` (own), `button`, `fade`, `stop`.
    pub ended: &'static str,
    pub events: Vec<Landed>,
    pub joins: Vec<Join>,
    pub alternatives: Vec<Alternative>,
    pub warnings: Vec<String>,
}

// ------------------------------------------------------------------------------------- planning

/// A planned remix before its joins are checked.
struct Planned {
    plan: Plan,
    /// The beats of each join (in plan order).
    beats: Vec<(usize, usize)>,
    stretch: f64,
    ended: &'static str,
    events: Vec<Landed>,
    warnings: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Lift {
    kind: LiftKind,
    /// The attack, source samples.
    src: i64,
    /// Strength within its kind, 0 … 1.
    strength: f64,
}

/// How the output ends after a node.
#[derive(Clone, Copy, Debug, PartialEq)]
enum End {
    /// Not an end.
    No,
    /// The output ends where the node's source position plays.
    At,
    /// It plays this many more samples, then ends.
    Ext(i64),
    /// It plays on to the film's end wherever the track is.
    Free,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Node {
    f: i64,
    x: i64,
    cost: f64,
    lift: Option<LiftKind>,
    event: Option<usize>,
    end: End,
    /// The tolerance its arrival may be off by.
    tol: i64,
}

#[derive(Clone, Debug, PartialEq)]
struct Span {
    joins: Vec<(usize, usize)>,
    /// Where the anchor plays against where it should (output samples).
    residual: i64,
    cost: f64,
}

#[derive(Clone, Copy)]
struct Level {
    stable: bool,
    phase: bool,
}

struct Ctx<'a> {
    a: &'a Analysis,
    st: &'a Structure,
    b: &'a [i64],
    m: usize,
    p: f64,
    lead: i64,
    sr: f64,
    tol: i64,
    h: i64,
    min_piece: i64,
    window: usize,
    xfade: i64,
    stable: Vec<bool>,
    /// The level of each beat and the next together, dB (a beat pattern's loud and soft beats
    /// evened out).
    pair_db: Vec<f64>,
    /// The track before its own ending (joins and fades stay before it).
    usable_end: i64,
    bar: f64,
}

fn sq(x: f64) -> f64 {
    x * x
}

impl<'a> Ctx<'a> {
    fn new(a: &'a Analysis, params: Params) -> Ctx<'a> {
        let st = &a.structure;
        let b = a.beats.as_slice();
        let m = b.len();
        let p = a.period;
        let sr = f64::from(a.sample_rate.max(1));
        let lead = remix::beat_lead(a.sample_rate);
        let music_end = if st.music_end > 0 { st.music_end } else { a.len };
        let usable_end = match st.ending {
            Some(e) if e.kind == EndingKind::Cut => e.at.min(music_end),
            Some(e) => e.at,
            None => music_end,
        };
        let bounds: Vec<usize> = st.boundaries().collect();
        // the level of two beats (a kick and a snare are far apart, two beats together are not)
        let pair = |a: usize| -> f64 {
            let e = st.beat_db.get(a..a + 2).map_or(0.0, |w| w.iter().map(|d| 10f64.powf(f64::from(*d) / 10.0)).sum::<f64>() / 2.0);
            if e > 1e-12 { 10.0 * e.log10() } else { FLOOR_DB }
        };
        let pair_db: Vec<f64> = (0..m).map(pair).collect();
        let stable = (0..m)
            .map(|k| {
                if k < 4 || k + 4 > m || st.beat_db.len() != m {
                    return false;
                }
                let w = [pair_db[k - 4], pair_db[k - 2], pair_db[k], pair_db[k + 2]];
                let (lo, hi) = w.iter().fold((f64::MAX, f64::MIN), |(l, h), v| (l.min(*v), h.max(*v)));
                hi - lo <= f64::from(STABLE_DB)
                    && f64::from(st.beat_db[k]) >= st.loud_db - 30.0
                    && !bounds.iter().any(|&s| s + 2 > k && s < k + 2)
                    && b[k] + lead + (2.0 * p) as i64 <= usable_end
                    && b[k] >= st.music_start
            })
            .collect();
        let window = 1 + (7.0 * params.variations.clamp(0.0, 100.0) / 100.0).round() as usize;
        Ctx {
            a,
            st,
            b,
            m,
            p,
            lead,
            sr,
            tol: ((0.5 * p) as i64).max(1),
            h: ((MIN_PIECE_BEATS as f64 / 2.0) * p) as i64,
            min_piece: (MIN_PIECE_BEATS as f64 * p) as i64,
            window,
            xfade: (XFADE_S * sr).round() as i64,
            stable,
            pair_db,
            usable_end,
            bar: p * st.meter.max(1) as f64,
        }
    }

    fn closest(&self, x: i64) -> usize {
        let i = self.b.partition_point(|&v| v < x).min(self.m - 1);
        if i > 0 && (x - self.b[i - 1]).abs() <= (self.b[i] - x).abs() { i - 1 } else { i }
    }

    fn attack(&self, i: usize) -> i64 {
        self.b[i] + self.lead
    }

    /// The level jump a join from beat `i` to beat `j` should make, predicted from the beats:
    /// against what the music does at the nearer of its two cut points (as [`check_joins`]
    /// measures it after rendering).
    fn predicted_jump(&self, i: usize, j: usize) -> f64 {
        let at = |k: usize| self.pair_db.get(k).copied();
        let after = match (at(i), at(j)) {
            (Some(a), Some(b)) => (a - b).abs(),
            _ => 0.0,
        };
        let before = match (i.checked_sub(2).and_then(at), j.checked_sub(2).and_then(at)) {
            (Some(a), Some(b)) => (a - b).abs(),
            _ => 0.0,
        };
        after.min(before)
    }

    fn join_cost(&self, i: usize, j: usize) -> f64 {
        let score = f64::from(self.a.jump_score(i, j)).clamp(0.0, 1.0);
        let mut c = JOIN_COST + SCORE_COST * (1.0 - score);
        c += LEVEL_COST * (self.predicted_jump(i, j) - 2.0).max(0.0) / 3.0;
        if !self.st.same_phase(i, j) {
            c += PHASE_COST;
        }
        if !(self.stable[i] && self.stable[j]) {
            c += UNSTABLE_COST;
        }
        c
    }

    fn tail(&self) -> i64 {
        ((self.p / self.sr).clamp(BUTTON_TAIL_S.0, BUTTON_TAIL_S.1) * self.sr) as i64
    }

    /// Every lift of the track.
    fn lifts(&self) -> Vec<Lift> {
        let st = self.st;
        let mut out = Vec::new();
        let drop_beats: Vec<usize> = st.drops.iter().map(|d| d.beat).collect();
        for d in &st.drops {
            if d.beat < self.m {
                out.push(Lift { kind: LiftKind::Drop, src: self.attack(d.beat), strength: 0.8 + 0.2 * (d.rise_db / 12.0).clamp(0.0, 1.0) });
            }
        }
        for w in st.sections.windows(2) {
            let (prev, s) = (&w[0], &w[1]);
            if s.first_beat >= self.m || drop_beats.contains(&s.first_beat) {
                continue;
            }
            let rise = s.db - prev.db;
            // a rising section start is a lift; a falling one is still a boundary a landing can use
            let strength = if rise >= 2.0 { 0.6 + 0.4 * (rise / 10.0).clamp(0.0, 1.0) } else { 0.3 };
            out.push(Lift { kind: LiftKind::Section, src: self.attack(s.first_beat), strength: strength * (0.5 + 0.5 * s.energy) });
        }
        for &k in &st.downbeats {
            if k >= self.m || self.attack(k) >= self.usable_end {
                continue;
            }
            let level = f64::from(st.beat_db.get(k).copied().unwrap_or(FLOOR_DB as f32));
            out.push(Lift { kind: LiftKind::Downbeat, src: self.attack(k), strength: 0.6 + 0.4 * ((level - (st.loud_db - 12.0)) / 12.0).clamp(0.0, 1.0) });
        }
        for &s in &st.stops {
            out.push(Lift { kind: LiftKind::Stop, src: s, strength: 1.0 });
        }
        if let Some(e) = st.ending {
            out.push(Lift { kind: LiftKind::Ending, src: e.at, strength: 1.0 });
        }
        out
    }

    /// How much an event of `kind` wants a lift of `lift` (0: not at all). `end`: the event ends
    /// the film with `ending`.
    fn accept(kind: EventKind, lift: LiftKind, end: Option<Ending>) -> f64 {
        use LiftKind as L;
        match (kind, end, lift) {
            (EventKind::Burst, _, L::Drop) | (EventKind::Peak, _, L::Drop) => 1.0,
            (EventKind::Burst, _, L::Section) => 0.8,
            (EventKind::Burst, _, L::Downbeat) => 0.35,
            (EventKind::Peak, _, L::Section) => 0.7,
            (EventKind::Peak, _, L::Downbeat) => 0.25,
            (EventKind::Landing, Some(Ending::Own), L::Ending) => 1.0,
            (EventKind::Landing, Some(Ending::Own), _) => 0.0,
            (EventKind::Landing, Some(Ending::Button), L::Section | L::Drop) => 0.9,
            (EventKind::Landing, Some(Ending::Button), L::Downbeat) => 0.6,
            (EventKind::Landing, _, L::Section) => 0.8,
            (EventKind::Landing, _, L::Drop) => 0.6,
            (EventKind::Landing, _, L::Downbeat) => 0.45,
            (EventKind::Stop, _, L::Stop) => 1.0,
            (EventKind::Stop, Some(_), L::Downbeat) => 0.5,
            _ => 0.0,
        }
    }

    /// The lifts offered to an event at `at`: the best by weight, nearer ones first, and the
    /// acceptable downbeat nearest it (the join-free option).
    fn candidates(&self, lifts: &[Lift], kind: EventKind, at: i64, end: Option<Ending>) -> Vec<(Lift, f64)> {
        let far = (16.0 * self.bar).max(1.0);
        let mut v: Vec<(Lift, f64, f64)> = lifts
            .iter()
            .filter_map(|l| {
                let w = Self::accept(kind, l.kind, end) * l.strength;
                (w > 0.0 && l.src >= 0 && l.src < self.a.len).then(|| (*l, w, w - 0.4 * ((l.src - at).abs() as f64 / far).min(1.0)))
            })
            .collect();
        v.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.src.cmp(&b.0.src)));
        let mut out: Vec<(Lift, f64)> = v.iter().take(CANDIDATES).map(|x| (x.0, x.1)).collect();
        if let Some(n) = v.iter().filter(|x| x.0.kind == LiftKind::Downbeat).min_by_key(|x| ((x.0.src - at).abs(), x.0.src))
            && !out.iter().any(|o| o.0.src == n.0.src)
        {
            out.push((n.0, n.1));
        }
        out
    }

    /// Joins from source `x_a` (playing at the start of a `d`-sample stretch of film) so that
    /// source `x_b` plays at its end, within `tol`.
    fn span(&self, x_a: i64, x_b: i64, d: i64, tol: i64, avoid: &[(usize, usize)]) -> Option<Span> {
        if d < 0 {
            return None;
        }
        let delta = d.checked_sub(x_b.checked_sub(x_a)?)?;
        let tol = tol.max(1);
        if delta.abs() <= tol {
            return Some(Span { joins: Vec::new(), residual: -delta, cost: RESIDUAL_COST * sq(delta as f64 / tol as f64) });
        }
        let music = (self.usable_end.min(self.a.len) - self.st.music_start).max(0);
        let max_shift = ((music - 2 * self.min_piece) as f64 * 0.9).max(self.p);
        let k0 = if delta > 0 { (delta as f64 / max_shift).ceil().max(1.0) } else { 1.0 };
        if !(k0.is_finite() && k0 <= MAX_SPAN_JOINS as f64) {
            return None;
        }
        let k0 = k0 as usize;
        let mut best: Option<Span> = None;
        for level in [Level { stable: true, phase: true }, Level { stable: true, phase: false }, Level { stable: false, phase: false }] {
            for k in k0..=(k0 + 1).min(MAX_SPAN_JOINS) {
                if let Some(s) = self.greedy(x_a, x_b, d, k, tol, level, self.window, avoid)
                    && best.as_ref().is_none_or(|b| s.cost < b.cost)
                {
                    best = Some(s);
                }
            }
            // none near the even places: wider, before a looser level
            if best.is_none() {
                best = self.greedy(x_a, x_b, d, k0, tol, level, self.window.max(WIDE_BEATS), avoid);
            }
            // a stricter level that works is not given up for a looser one
            if best.is_some() {
                break;
            }
        }
        best
    }

    #[allow(clippy::too_many_arguments)]
    fn greedy(&self, x_a: i64, x_b: i64, d: i64, k: usize, tol: i64, level: Level, wide: usize, avoid: &[(usize, usize)]) -> Option<Span> {
        let b = self.b;
        let m = self.m;
        let w = self.window;
        let dir = (d - (x_b - x_a)).signum();
        let mut src = x_a;
        let mut out = 0i64;
        let mut joins = Vec::with_capacity(k);
        let mut cost = 0.0;
        let mut residual = 0i64;
        for q in 0..k {
            let left = (k - q) as i64;
            let out_left = d - out;
            let piece = out_left / (left + 1);
            let shift = ((x_b - src) - out_left) / left;
            let ni = self.closest(src + piece);
            let last = q + 1 == k;
            let lead_in = if q == 0 { self.h } else { self.min_piece };
            let mut pick: Option<(i64, f64, usize, usize, i64)> = None;
            for i in ni.saturating_sub(wide)..=(ni + wide).min(m - 1) {
                let bi = b[i];
                if bi - src < lead_in || out + (bi - src) > d - self.h || (level.stable && !self.stable[i]) {
                    continue;
                }
                let nj = self.closest(bi + shift);
                for j in nj.saturating_sub(w)..=(nj + w).min(m - 1) {
                    let bj = b[j];
                    if j == i || (bj - bi).signum() != -dir || avoid.contains(&(i, j)) {
                        continue;
                    }
                    if (level.stable && !self.stable[j]) || (level.phase && !self.st.same_phase(i, j)) {
                        continue;
                    }
                    // a join the level check would refuse is not tried in a stable stretch
                    if level.stable && self.predicted_jump(i, j) > JOIN_MAX_DB {
                        continue;
                    }
                    let new_out = out + (bi - src);
                    let rest_src = x_b - bj;
                    if rest_src < self.h {
                        continue;
                    }
                    let err = (d - new_out) - rest_src;
                    if last {
                        if err.abs() > tol {
                            continue;
                        }
                    } else {
                        let rem_out = d - new_out;
                        if rem_out < (left - 1) * self.min_piece + self.h || err.signum() != dir || (err.abs() as f64) < (left - 1) as f64 * self.p * 0.75 {
                            continue;
                        }
                    }
                    let c = self.join_cost(i, j);
                    let key = if last { err.abs() * 4 / tol } else { 0 };
                    let better = match pick {
                        None => true,
                        Some((k0, c0, i0, j0, _)) => key < k0 || (key == k0 && (c < c0 || (c == c0 && (i, j) < (i0, j0)))),
                    };
                    if better {
                        pick = Some((key, c, i, j, err));
                    }
                }
            }
            let (_, c, i, j, err) = pick?;
            joins.push((i, j));
            cost += c;
            out += b[i] - src;
            src = b[j];
            if last {
                residual = -err;
            }
        }
        cost += RESIDUAL_COST * sq(residual as f64 / tol as f64);
        Some(Span { joins, residual, cost })
    }

    /// The plain remix (no events, ending or stretch), as before.
    fn plain(&self, target: i64, params: Params, avoid: &[(usize, usize)]) -> Result<Planned, RemixError> {
        let plan = remix::plan_avoiding(self.a, target, params, avoid)?;
        let beats = plan.cuts().iter().filter_map(|(o, i)| Some((self.b.binary_search(o).ok()?, self.b.binary_search(i).ok()?))).collect();
        Ok(Planned { plan, beats, stretch: 1.0, ended: "source", events: Vec::new(), warnings: Vec::new() })
    }

    /// The fit (see the module docs).
    fn fitted(&self, events: &[Event], req: &Request, max_stretch: f64, avoid: &[(usize, usize)]) -> Result<Planned, RemixError> {
        let target = req.target;
        let lifts = self.lifts();
        let n = events.len();
        let mut warnings = Vec::new();
        // the event that ends the film: a last stop, or a last landing with an own or button ending
        let last = events.last().copied();
        let end_event = match last {
            Some(e) if e.kind == EventKind::Stop => true,
            Some(e) if e.kind == EventKind::Landing && matches!(req.ending, Ending::Own | Ending::Button) => true,
            _ => false,
        };
        let mut layers: Vec<Vec<Node>> = Vec::with_capacity(n + 2);
        // ---- starts: the intro, or (for an opening burst or peak) the track from a section or drop
        let mut starts = vec![Node { f: 0, x: 0, cost: 0.0, lift: None, event: None, end: End::No, tol: self.tol }];
        if let Some(e0) = events.first().filter(|e| matches!(e.kind, EventKind::Burst | EventKind::Peak) && (e.at as f64) <= 2.0 * self.bar) {
            for (l, _) in self.candidates(&lifts, e0.kind, e0.at, None) {
                if !matches!(l.kind, LiftKind::Drop | LiftKind::Section) {
                    continue;
                }
                // the track starts where the lift plays exactly at the event (on the film's beat
                // when the event is on it)
                let want = l.src - e0.at;
                if want > 0 && !starts.iter().any(|s| s.x == want) {
                    starts.push(Node { f: 0, x: want, cost: COLD_COST, lift: None, event: None, end: End::No, tol: self.tol });
                }
            }
        }
        layers.push(starts);
        // ---- the events' candidate lifts
        let tail = self.tail();
        let own_ext = |hit: i64| (self.st.music_end.max(hit) - hit).max(0);
        for (ei, e) in events.iter().enumerate() {
            let is_end = end_event && ei + 1 == n;
            let end_mode = if is_end { Some(if e.kind == EventKind::Stop { Ending::Source } else { req.ending }) } else { None };
            let mut v = Vec::new();
            for (l, w) in self.candidates(&lifts, e.kind, e.at, end_mode) {
                let (end, lift) = match (is_end, e.kind, req.ending) {
                    (false, _, _) => (End::No, l.kind),
                    (true, EventKind::Stop, _) => (End::At, if l.kind == LiftKind::Downbeat { LiftKind::Cut } else { l.kind }),
                    (true, _, Ending::Own) => (End::Ext(own_ext(l.src)), l.kind),
                    (true, _, _) => (End::Ext(tail), l.kind),
                };
                v.push(Node { f: e.at, x: l.src, cost: LIFT_COST * (1.0 - w), lift: Some(lift), event: Some(ei), end, tol: self.tol });
            }
            layers.push(v);
        }
        // ---- the end (also when the ending event is missed)
        let tol_end = if n == 0 { (max_stretch * target as f64).max(self.p) as i64 } else { self.p as i64 };
        let music_end = if self.st.music_end > 0 { self.st.music_end } else { self.a.len };
        let ends: Vec<Node> = match req.ending {
            Ending::Source => vec![Node { f: target, x: self.a.len, cost: 0.0, lift: None, event: None, end: End::At, tol: tol_end }],
            Ending::Own => vec![Node { f: target, x: music_end, cost: 0.0, lift: None, event: None, end: End::At, tol: tol_end }],
            Ending::Fade => vec![Node { f: target, x: -1, cost: 0.0, lift: None, event: None, end: End::Free, tol: tol_end }],
            Ending::Button => {
                let mut v: Vec<(Lift, f64)> = self.candidates(&lifts, EventKind::Landing, target - tail, Some(Ending::Button));
                v.retain(|(l, _)| l.src + tail <= self.usable_end.max(l.src));
                v.into_iter()
                    .map(|(l, w)| Node {
                        f: target - tail,
                        x: l.src,
                        cost: LIFT_COST * (1.0 - w),
                        lift: Some(l.kind),
                        event: None,
                        end: End::Ext(tail),
                        tol: tol_end,
                    })
                    .collect()
            }
        };
        layers.push(ends);
        // ---- the shortest path
        let nl = layers.len();
        type Back = Option<(f64, usize, usize, Span)>;
        let mut best: Vec<Vec<Back>> = layers.iter().map(|l| vec![None; l.len()]).collect();
        for (pi, p) in layers[0].iter().enumerate() {
            best[0][pi] = Some((p.cost, usize::MAX, usize::MAX, Span { joins: Vec::new(), residual: 0, cost: 0.0 }));
        }
        let free_room = self.usable_end.min(self.a.len);
        for lc in 1..nl {
            let is_end_layer = lc == nl - 1;
            for ci in 0..layers[lc].len() {
                let c = layers[lc][ci];
                let lo = if is_end_layer { 0 } else { lc.saturating_sub(MAX_SKIP + 1) };
                for lp in lo..lc {
                    // events skipped between the two (layer k is event k − 1)
                    let missed: f64 = (lp + 1..lc).filter_map(|l| events.get(l - 1)).map(|e| e.kind.miss_cost()).sum();
                    for pi in 0..layers[lp].len() {
                        let p = layers[lp][pi];
                        let Some(pc) = best[lp][pi].as_ref().map(|b| b.0) else { continue };
                        if p.end != End::No {
                            continue;
                        }
                        let span = match c.end {
                            End::Free => {
                                let room = target - p.f;
                                if room < 0 {
                                    None
                                } else if p.x + room <= free_room {
                                    Some(Span { joins: Vec::new(), residual: 0, cost: 0.0 })
                                } else {
                                    self.span(p.x, free_room, room, c.tol, avoid)
                                }
                            }
                            _ => self.span(p.x, c.x, c.f - p.f, c.tol, avoid),
                        };
                        let Some(span) = span else { continue };
                        let total = pc + missed + span.cost + c.cost;
                        if best[lc][ci].as_ref().is_none_or(|b| total < b.0) {
                            best[lc][ci] = Some((total, lp, pi, span));
                        }
                    }
                }
            }
        }
        // the best way to end: an end node, or an ending event's node
        let mut fin: Option<(f64, usize, usize)> = None;
        for (l, layer) in layers.iter().enumerate() {
            for (i, node) in layer.iter().enumerate() {
                if node.end == End::No {
                    continue;
                }
                if let Some((c, ..)) = &best[l][i]
                    && fin.is_none_or(|f| *c < f.0)
                {
                    fin = Some((*c, l, i));
                }
            }
        }
        let (_, mut l, mut i) = fin.ok_or(RemixError::TargetTooShort)?;
        // ---- back along the path
        let mut path: Vec<(Node, Span)> = Vec::new();
        while l != usize::MAX && path.len() <= nl {
            let Some((_, pl, pi, span)) = best[l][i].clone() else { break };
            path.push((layers[l][i], span));
            l = pl;
            i = pi;
        }
        path.reverse();
        let Some(((start, _), rest)) = path.split_first() else { return Err(RemixError::TargetTooShort) };
        // ---- the pieces
        let b = self.b;
        let mut pieces = Vec::new();
        let mut beats = Vec::new();
        let mut src = start.x;
        let mut out = 0i64;
        let mut played: Vec<(usize, i64, LiftKind)> = Vec::new();
        let mut ended = req.ending.id();
        let mut fade_out = 0i64;
        for (c, span) in rest {
            for &(i, j) in &span.joins {
                pieces.push(Piece { src, len: b[i] - src });
                out += b[i] - src;
                src = b[j];
                beats.push((i, j));
            }
            if let (Some(e), Some(lift)) = (c.event, c.lift) {
                played.push((e, out + (c.x - src), lift));
            }
            match c.end {
                End::No => {}
                End::At => {
                    pieces.push(Piece { src, len: c.x - src });
                    out += c.x - src;
                }
                End::Ext(ext) => {
                    pieces.push(Piece { src, len: c.x + ext - src });
                    out += c.x + ext - src;
                }
                End::Free => {
                    let len = target - out;
                    pieces.push(Piece { src, len });
                    out += len;
                }
            }
        }
        if pieces.iter().any(|p| p.len <= 0 || p.src < 0) || pieces.is_empty() {
            return Err(RemixError::TargetTooShort);
        }
        let total: i64 = pieces.iter().map(|p| p.len).sum();
        let last_node = rest.last().map(|x| x.0);
        match last_node.map(|n| (n.end, n.event)) {
            Some((End::At, Some(_))) => ended = "stop",
            Some((End::At, None)) if req.ending == Ending::Own => ended = self.st.ending.map_or("own", |e| e.kind.id()),
            Some((End::Ext(ext), _)) if req.ending == Ending::Button => {
                ended = "button";
                fade_out = ext;
            }
            Some((End::Ext(_), _)) => {
                ended = self.st.ending.map_or("own", |e| e.kind.id());
                // the ring-out runs past the film's end: cut there, faded
                if total > target
                    && let Some(lp) = pieces.last_mut()
                {
                    let cut = (total - target).min(lp.len - self.xfade.max(1));
                    if cut > 0 {
                        lp.len -= cut;
                    }
                    fade_out = ((CUT_SHORT_FADE_S * self.sr) as i64).min(lp.len);
                }
            }
            Some((End::Free, _)) => {
                ended = "fade";
                // a landing before the end fades from there, else two bars
                let landing = events.iter().rev().find(|e| e.kind == EventKind::Landing).map(|e| target - e.at);
                let (lo, hi) = if landing.is_some() { LANDING_FADE_S } else { FADE_S };
                let want = landing.unwrap_or((2.0 * self.bar) as i64) as f64 / self.sr;
                fade_out = (want.clamp(lo, hi) * self.sr) as i64;
            }
            _ => {}
        }
        let total: i64 = pieces.iter().map(|p| p.len).sum();
        let fade_out = fade_out.clamp(0, total / 2);
        // a track started mid-way fades in over 10 ms (its first sample would click)
        let fade_in = if start.x > 0 { ((COLD_FADE_S * self.sr) as i64).min(total / 2) } else { 0 };
        let plan = Plan { pieces, xfade: self.xfade, fade_out, fade_in };
        if !plan.is_playable() {
            return Err(RemixError::TargetTooShort);
        }
        // ---- the stretch: the end on the film's end, or the anchors fitted best
        let mut stretch = 1.0;
        if max_stretch > 0.0 {
            let pairs: Vec<(f64, f64)> =
                played.iter().filter_map(|(e, at, _)| events.get(*e).map(|ev| (*at as f64, ev.at as f64))).filter(|(p, _)| *p > 0.0).collect();
            let s = if pairs.is_empty() || matches!(last_node.map(|n| n.end), Some(End::At | End::Free)) && n == 0 {
                total as f64 / target.max(1) as f64
            } else {
                let mut pairs = pairs;
                if !matches!(last_node.map(|n| n.end), Some(End::Ext(_))) {
                    pairs.push((total as f64, target as f64));
                }
                let (pp, pf) = pairs.iter().fold((0.0, 0.0), |(a, c), (p, f)| (a + p * p, c + p * f));
                if pf > 0.0 { pp / pf } else { 1.0 }
            };
            if s.is_finite() && s > 0.0 {
                stretch = s.clamp(1.0 - max_stretch, 1.0 + max_stretch);
            }
            if (stretch - 1.0).abs() < 1e-5 {
                stretch = 1.0;
            }
        }
        let events = self.landings(&plan, events, &played, stretch, &lifts);
        for e in events.iter().filter(|e| !e.landed) {
            warnings.push(match e.off_by {
                Some(o) => format!("the {} at {:.2} s: the nearest lift plays {:+.2} s off it", e.kind.id(), e.at as f64 / self.sr, o as f64 / self.sr),
                None => format!("the {} at {:.2} s: no lift of the track reaches it", e.kind.id(), e.at as f64 / self.sr),
            });
        }
        Ok(Planned { plan, beats, stretch, ended, events, warnings })
    }

    /// How each event came out: the lift its path chose, or the nearest one as played (within a bar).
    fn landings(&self, plan: &Plan, events: &[Event], played: &[(usize, i64, LiftKind)], stretch: f64, lifts: &[Lift]) -> Vec<Landed> {
        let s = if stretch.is_finite() && stretch > 0.0 { stretch } else { 1.0 };
        let out_of = |x: i64| (x as f64 / s).round() as i64;
        // every lift as it plays (output samples)
        let mut playing: Vec<(i64, LiftKind)> = Vec::new();
        let mut d = 0i64;
        for p in &plan.pieces {
            for l in lifts {
                if l.src >= p.src && l.src < p.src + p.len {
                    playing.push((d + (l.src - p.src), l.kind));
                }
            }
            d += p.len;
        }
        events
            .iter()
            .enumerate()
            .map(|(k, e)| {
                if let Some((_, at, lift)) = played.iter().find(|p| p.0 == k) {
                    let at = out_of(*at);
                    let off = at - e.at;
                    return Landed { kind: e.kind, at: e.at, lift: Some(*lift), plays_at: Some(at), off_by: Some(off), landed: off.abs() <= self.tol };
                }
                let near = playing
                    .iter()
                    .filter(|(_, kind)| Self::accept(e.kind, *kind, None) > 0.0 || (e.kind == EventKind::Landing && *kind == LiftKind::Ending))
                    .map(|(at, kind)| (out_of(*at), *kind))
                    .min_by_key(|(at, _)| ((at - e.at).abs(), *at))
                    .filter(|(at, _)| ((at - e.at).abs() as f64) <= self.bar);
                match near {
                    Some((at, kind)) => {
                        let off = at - e.at;
                        Landed { kind: e.kind, at: e.at, lift: Some(kind), plays_at: Some(at), off_by: Some(off), landed: off.abs() <= self.tol }
                    }
                    None => Landed { kind: e.kind, at: e.at, lift: None, plays_at: None, off_by: None, landed: false },
                }
            })
            .collect()
    }

    /// Film lengths that need no join, nearest the target first.
    fn alternatives(&self, target: i64, max_stretch: f64) -> Vec<Alternative> {
        let mut v: Vec<(i64, Ending)> = vec![(self.a.len, Ending::Source)];
        let music_end = self.st.music_end;
        if music_end > 0 {
            v.push((music_end, Ending::Own));
        }
        let tail = self.tail();
        let strong: Vec<i64> = self
            .st
            .downbeats
            .iter()
            .copied()
            .filter(|&k| k < self.m && f64::from(self.st.beat_db.get(k).copied().unwrap_or(FLOOR_DB as f32)) >= self.st.loud_db - 6.0)
            .map(|k| self.attack(k))
            .filter(|&a| a > 0 && a + tail <= self.usable_end.max(1))
            .collect();
        let bars: Vec<i64> =
            self.st.downbeats.iter().copied().filter(|&k| k < self.m).map(|k| self.attack(k)).filter(|&a| a > 0 && a <= self.usable_end).collect();
        let around = |list: &[i64], want: i64| -> Vec<i64> {
            let below = list.iter().copied().filter(|x| *x <= want).max();
            let above = list.iter().copied().filter(|x| *x > want).min();
            below.into_iter().chain(above).collect()
        };
        for d in around(&strong, target - tail) {
            v.push((d + tail, Ending::Button));
        }
        for d in around(&bars, target) {
            v.push((d, Ending::Fade));
        }
        v.sort_by_key(|(len, e)| ((len - target).abs(), *len, e.id()));
        v.dedup();
        let ms = max_stretch.clamp(0.0, MAX_STRETCH_PCT / 100.0);
        v.into_iter()
            .take(8)
            .map(|(len, ending)| Alternative {
                len,
                joins: 0,
                ending,
                min: (len as f64 / (1.0 + ms)).round() as i64,
                max: (len as f64 / (1.0 - ms).max(0.5)).round() as i64,
            })
            .collect()
    }
}

/// The level of planar audio, dB (mean square over every channel and sample).
fn level_db(x: &[Vec<f32>]) -> f64 {
    let (mut s, mut n) = (0f64, 0usize);
    for c in x {
        for v in c {
            if v.is_finite() {
                s += f64::from(*v) * f64::from(*v);
            }
            n += 1;
        }
    }
    let ms = s / n.max(1) as f64;
    if ms > 1e-12 { 10.0 * ms.log10() } else { FLOOR_DB }
}

/// Each join of `plan` rendered and measured (see the module docs). `beats` are the joins' beat
/// pairs, in order, when known.
pub fn check_joins(a: &Analysis, plan: &Plan, beats: &[(usize, usize)], read: &mut Reader<'_>) -> Vec<Join> {
    let sr = f64::from(a.sample_rate.max(1));
    let w = ((JOIN_WINDOW_S * sr) as usize).max(1);
    let wi = w as i64;
    let st = &a.structure;
    let quiet = -70.0;
    let mut out = Vec::new();
    for (k, (at, (o, i))) in plan.joints().into_iter().zip(plan.cuts()).enumerate() {
        let before = level_db(&remix::render(plan, at - wi, w, 2, &mut *read));
        let after = level_db(&remix::render(plan, at, w, 2, &mut *read));
        let change = if before < quiet && after < quiet { 0.0 } else { after - before };
        let natural = |x: i64, read: &mut Reader<'_>| -> f64 {
            let (pre, post) = (level_db(&read(x - wi, w)), level_db(&read(x, w)));
            if pre < quiet && post < quiet { 0.0 } else { post - pre }
        };
        let (n_out, n_in) = (natural(o, read), natural(i, read));
        let excess = (change - n_out).abs().min((change - n_in).abs());
        let pair = beats.get(k).copied();
        let score = pair.map(|(bi, bj)| a.jump_score(bi, bj));
        let r1 = |x: f64| (x * 10.0).round() / 10.0;
        out.push(Join {
            at,
            out: o,
            into: i,
            score,
            stable: false,
            keeps_bars: pair.is_some_and(|(bi, bj)| st.same_phase(bi, bj)),
            change_db: r1(change),
            excess_db: r1(excess),
            ok: excess <= JOIN_MAX_DB,
        });
    }
    out
}

/// Fit the analysed music to `req` (see the module docs), reading the source through `read` to
/// check each join after rendering.
pub fn fit(a: &Analysis, req: &Request, read: &mut Reader<'_>) -> Result<Fit, RemixError> {
    if a.beats.len() < 2 || !a.period.is_finite() || a.period <= 0.0 || a.structure.beat_db.len() != a.beats.len() && !a.structure.beat_db.is_empty() {
        return Err(RemixError::TooShort);
    }
    if req.target <= 0 {
        return Err(RemixError::TargetTooShort);
    }
    let sr = f64::from(a.sample_rate.max(1));
    let max_stretch = if req.max_stretch.is_finite() { req.max_stretch.clamp(0.0, MAX_STRETCH_PCT / 100.0) } else { 0.0 };
    let mut warnings = Vec::new();
    let mut events: Vec<Event> = Vec::new();
    for e in &req.events {
        if e.at < 0 || e.at > req.target {
            warnings.push(format!("the {} at {:.2} s is outside the clip: left out", e.kind.id(), e.at as f64 / sr));
        } else if events.len() >= MAX_EVENTS {
            warnings.push(format!("more than {MAX_EVENTS} events: the rest are left out"));
            break;
        } else {
            events.push(*e);
        }
    }
    events.sort();
    events.dedup();
    // a landing that ends the film (own or button) is the last thing the music does
    if matches!(req.ending, Ending::Own | Ending::Button)
        && let Some(k) = events.iter().rposition(|e| e.kind == EventKind::Landing)
        && k + 1 < events.len()
    {
        warnings.push(format!("{} event(s) after the landing are left out: the music ends there", events.len() - k - 1));
        events.truncate(k + 1);
    }
    let ctx = Ctx::new(a, req.params);
    let plain = events.is_empty() && req.ending == Ending::Source && max_stretch == 0.0;
    let mut avoid: Vec<(usize, usize)> = Vec::new();
    let mut best: Option<(usize, Planned, Vec<Join>)> = None;
    let mut fell_back = false;
    for round in 0..ROUNDS {
        let planned = if plain {
            ctx.plain(req.target, req.params, &avoid)
        } else {
            match ctx.fitted(&events, req, max_stretch, &avoid) {
                Ok(p) => Ok(p),
                Err(_) if round == 0 => {
                    // nothing fits the events and ending: the plain remix, the events reported as they fall
                    fell_back = true;
                    ctx.plain(req.target, req.params, &avoid).map(|mut p| {
                        let lifts = ctx.lifts();
                        p.events = ctx.landings(&p.plan, &events, &[], 1.0, &lifts);
                        p
                    })
                }
                Err(e) => Err(e),
            }
        };
        let p = match planned {
            Ok(p) => p,
            Err(e) if best.is_none() => return Err(e),
            Err(_) => break,
        };
        let mut joins = check_joins(a, &p.plan, &p.beats, read);
        for (j, pair) in joins.iter_mut().zip(&p.beats) {
            j.stable = ctx.stable.get(pair.0).copied().unwrap_or(false) && ctx.stable.get(pair.1).copied().unwrap_or(false);
        }
        let failed: Vec<(usize, usize)> = joins.iter().zip(&p.beats).filter(|(j, _)| !j.ok).map(|(_, b)| *b).collect();
        let fails = joins.iter().filter(|j| !j.ok).count();
        let new = failed.iter().filter(|b| !avoid.contains(b)).count();
        if best.as_ref().is_none_or(|(n, ..)| fails < *n) {
            best = Some((fails, p, joins));
        }
        if fails == 0 || new == 0 || fell_back {
            break;
        }
        // the refused joins and their neighbours (the same passages, a beat or two away)
        let r = AVOID_BEATS;
        for (i, j) in failed {
            for a in i.saturating_sub(r)..=i + r {
                for b in j.saturating_sub(r)..=j + r {
                    if !avoid.contains(&(a, b)) {
                        avoid.push((a, b));
                    }
                }
            }
        }
    }
    let (_, p, joins) = best.ok_or(RemixError::TargetTooShort)?;
    if fell_back {
        warnings.push("the events and ending could not be fitted: the plain remix is used".into());
    }
    for j in joins.iter().filter(|j| !j.ok) {
        warnings.push(format!(
            "the join at {:.2} s (from {:.2} s to {:.2} s of the track) changes the level by {:+.1} dB, {:.1} dB more than the music does there",
            j.at as f64 / sr,
            j.out as f64 / sr,
            j.into as f64 / sr,
            j.change_db,
            j.excess_db
        ));
    }
    if req.ending != Ending::Source && p.ended == "stop" {
        warnings.push(format!("the music stops at the last stop event: the {} ending is not used", req.ending.id()));
    }
    warnings.extend(p.warnings);
    let plan_len = p.plan.len();
    let len = (plan_len as f64 / p.stretch).round() as i64;
    Ok(Fit {
        plan: p.plan,
        stretch: p.stretch,
        len,
        ending: req.ending,
        ended: if fell_back { "source" } else { p.ended },
        events: p.events,
        joins,
        alternatives: ctx.alternatives(req.target, max_stretch),
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::{FixtureEnd, Part, fixture};

    const SR: u32 = 22_050;

    fn part(bars: usize, db: f64, drums: bool, build: bool) -> Part {
        Part { bars, db, drums, build }
    }

    /// Intro (8 s), verse (16 s), build (8 s), drop (16 s), breakdown (8 s), drop (16 s), a
    /// cadence: 72 s of music at 120 BPM.
    fn track(end: FixtureEnd) -> (Vec<Vec<f32>>, Analysis, crate::structure::Fixture) {
        let parts = [
            part(4, -16.0, false, false),
            part(8, -7.0, true, false),
            part(4, -9.0, false, true),
            part(8, 0.0, true, false),
            part(4, -14.0, false, false),
            part(8, 0.0, true, false),
        ];
        let f = fixture(SR, 120.0, &parts, end, 3);
        let refs: Vec<&[f32]> = f.audio.iter().map(Vec::as_slice).collect();
        let a = remix::analyze(&refs, SR).expect("analysis");
        (f.audio.clone(), a, f)
    }

    fn reader(m: &[Vec<f32>]) -> impl FnMut(i64, usize) -> Vec<Vec<f32>> + '_ {
        move |s: i64, n: usize| {
            m.iter().map(|c| (0..n as i64).map(|k| usize::try_from(s + k).ok().and_then(|i| c.get(i)).copied().unwrap_or(0.0)).collect()).collect()
        }
    }

    fn secs(x: i64) -> f64 {
        x as f64 / f64::from(SR)
    }

    fn at(s: f64) -> i64 {
        (s * f64::from(SR)) as i64
    }

    fn req(target: f64) -> Request {
        Request { target: at(target), ..Request::default() }
    }

    /// Without events, ending or stretch, the fit is the plain remix of before.
    #[test]
    fn the_plain_remix_is_unchanged() {
        let (m, a, _) = track(FixtureEnd::Cadence);
        for t in [40.0, 60.0, 100.0] {
            let f = fit(&a, &req(t), &mut reader(&m)).expect("fit");
            let plain = remix::plan(&a, at(t), Params::default()).expect("plan");
            if f.joins.iter().all(|j| j.ok) {
                assert_eq!(f.plan, plain, "{t} s");
            }
            assert_eq!(f.stretch, 1.0);
            assert_eq!(f.ended, "source");
            assert!(f.events.is_empty());
        }
    }

    /// A peak at 20 s gets the track's drop (32 s into the track) on it: the intro and verse are
    /// cut short, the drop lands within half a beat, and every join is in a stable stretch.
    #[test]
    fn a_peak_lands_on_the_drop() {
        let (m, a, f) = track(FixtureEnd::Cadence);
        let mut r = req(50.0);
        r.events = vec![Event { kind: EventKind::Peak, at: at(20.0) }];
        r.ending = Ending::Own;
        let fit = fit(&a, &r, &mut reader(&m)).expect("fit");
        eprintln!("{:?}\n{:?}\n{:?}", fit.events, fit.joins, fit.warnings);
        let e = fit.events[0];
        assert!(e.landed, "{e:?}");
        assert_eq!(e.lift, Some(LiftKind::Drop));
        assert!(secs(e.off_by.unwrap()).abs() <= 0.25, "{e:?}");
        // the drop played is one of the track's
        let drops: Vec<i64> = f.parts.iter().copied().filter(|p| [f.parts[3], f.parts[5]].contains(p)).collect();
        assert!(!drops.is_empty());
        assert!(fit.joins.iter().all(|j| j.ok && j.stable), "{:?}", fit.joins);
        // own ending: the cadence ends the film
        assert_eq!(fit.ended, "cadence");
        assert!((secs(fit.len) - 50.0).abs() <= 0.5, "{}", secs(fit.len));
        let last = fit.plan.pieces.last().unwrap();
        assert_eq!(last.src + last.len, a.structure.music_end);
    }

    /// The burst at the opening lands the first drop with a cold open; a landing with a button
    /// ending ends on a strong downbeat with a short faded tail.
    #[test]
    fn a_burst_and_a_button_landing() {
        let (m, a, _) = track(FixtureEnd::Cadence);
        let mut r = req(30.0);
        r.events = vec![Event { kind: EventKind::Burst, at: 0 }, Event { kind: EventKind::Landing, at: at(27.0) }];
        r.ending = Ending::Button;
        let fit = fit(&a, &r, &mut reader(&m)).expect("fit");
        eprintln!("{:?}\n{:?}\n{:?}\n{:?}", fit.plan, fit.events, fit.joins, fit.warnings);
        assert!(fit.events[0].landed && fit.events[0].lift == Some(LiftKind::Drop), "{:?}", fit.events);
        // a cold open: the drop's attack is the first thing heard, exactly, after a short fade-in
        assert_eq!(fit.events[0].off_by, Some(0), "{:?}", fit.events);
        assert!(fit.plan.pieces[0].src > 0 && fit.plan.fade_in > 0 && fit.plan.fade_in <= 221, "{:?}", fit.plan);
        assert!(fit.events[1].landed, "{:?}", fit.events);
        assert_eq!(fit.ended, "button");
        let tail = secs(fit.plan.fade_out);
        assert!((0.25..=0.8).contains(&tail), "{tail}");
        assert!((secs(fit.len) - (27.0 + tail)).abs() < 0.3, "{}", secs(fit.len));
    }

    /// A fade ending with no events plays the intro on and fades out over two bars at the film's
    /// end: no join at all when the track is long enough.
    #[test]
    fn a_fade_needs_no_join_when_the_track_is_long_enough() {
        let (m, a, _) = track(FixtureEnd::Cadence);
        let mut r = req(40.0);
        r.ending = Ending::Fade;
        let fit = fit(&a, &r, &mut reader(&m)).expect("fit");
        assert_eq!(fit.plan.pieces.len(), 1, "{:?}", fit.plan);
        assert_eq!(fit.len, at(40.0));
        assert_eq!(fit.ended, "fade");
        assert!((secs(fit.plan.fade_out) - 4.0).abs() < 0.1, "{}", secs(fit.plan.fade_out));
        // the alternatives: lengths with no join, the nearest first
        assert!(!fit.alternatives.is_empty() && fit.alternatives.iter().all(|x| x.joins == 0));
        assert!(fit.alternatives.iter().any(|x| x.ending == Ending::Own && (x.len - a.structure.music_end).abs() < 2));
    }

    /// A stretch of up to 5 % brings a film 2 % longer than the track's own length to its end
    /// without a join, exactly.
    #[test]
    fn a_small_stretch_avoids_a_join() {
        let (m, a, _) = track(FixtureEnd::Cadence);
        let natural = a.structure.music_end;
        let mut r = Request { target: (natural as f64 * 1.02) as i64, ending: Ending::Own, max_stretch: 0.05, ..Request::default() };
        let f = fit(&a, &r, &mut reader(&m)).expect("fit");
        assert_eq!(f.plan.pieces.len(), 1, "{:?}", f.plan);
        assert!((f.stretch - 1.0 / 1.02).abs() < 1e-3, "{}", f.stretch);
        assert!((f.len - r.target).abs() <= 1, "{} vs {}", f.len, r.target);
        // without the stretch it takes a join
        r.max_stretch = 0.0;
        let f = fit(&a, &r, &mut reader(&m)).expect("fit");
        assert!(f.plan.pieces.len() > 1);
        assert_eq!(f.stretch, 1.0);
    }

    /// A join from a loud part into a quiet one is refused after rendering: the planner takes
    /// another, or warns. Here: the quiet breakdown and loud drop are 14 dB apart.
    #[test]
    fn a_join_that_jumps_is_refused() {
        let (m, a, _) = track(FixtureEnd::Cadence);
        // a plan that jumps from the drop (40 s) straight into the breakdown (50 s)
        let b = &a.beats;
        let i = b.partition_point(|&x| x < at(40.0));
        let j = b.partition_point(|&x| x < at(50.0));
        let plan = Plan { pieces: vec![Piece { src: 0, len: b[i] }, Piece { src: b[j], len: a.len - b[j] }], xfade: 441, fade_out: 0, fade_in: 0 };
        let joins = check_joins(&a, &plan, &[(i, j)], &mut reader(&m));
        assert_eq!(joins.len(), 1);
        assert!(!joins[0].ok && joins[0].excess_db > 6.0, "{:?}", joins[0]);
        // and a join inside one part passes
        let j2 = b.partition_point(|&x| x < at(44.0));
        let plan = Plan { pieces: vec![Piece { src: 0, len: b[i] }, Piece { src: b[j2], len: a.len - b[j2] }], xfade: 441, fade_out: 0, fade_in: 0 };
        let joins = check_joins(&a, &plan, &[(i, j2)], &mut reader(&m));
        assert!(joins[0].ok, "{:?}", joins[0]);
        // every fitted plan's joins pass (the planner keeps to stable stretches)
        for (t, ending) in [(30.0, Ending::Own), (90.0, Ending::Own), (45.0, Ending::Button), (100.0, Ending::Fade)] {
            let f = fit(&a, &Request { target: at(t), ending, ..Request::default() }, &mut reader(&m)).expect("fit");
            assert!(f.joins.iter().all(|j| j.ok), "{t} s {ending:?}: {:?} {:?}", f.joins, f.warnings);
        }
    }

    /// Hostile requests: NaN and huge stretches, events past the clip and before it, absurd
    /// targets, a track shorter than the film, a degenerate analysis: a fit, an error or a
    /// warning, never a panic.
    #[test]
    fn hostile_requests_never_panic() {
        let (m, a, _) = track(FixtureEnd::Fade);
        let many: Vec<Event> = (0..10_000).map(|k| Event { kind: EventKind::Peak, at: k * 100 }).collect();
        for r in [
            Request { target: at(30.0), max_stretch: f64::NAN, ..Request::default() },
            Request { target: at(30.0), max_stretch: 1e300, ending: Ending::Own, ..Request::default() },
            Request { target: at(30.0), max_stretch: -3.0, ending: Ending::Button, ..Request::default() },
            Request {
                target: at(30.0),
                events: vec![Event { kind: EventKind::Burst, at: -5 }, Event { kind: EventKind::Peak, at: i64::MAX }],
                ..Request::default()
            },
            Request { target: at(30.0), events: many.clone(), ending: Ending::Fade, ..Request::default() },
            Request { target: at(400.0), ending: Ending::Own, events: vec![Event { kind: EventKind::Stop, at: at(200.0) }], ..Request::default() },
            Request { target: 1, ending: Ending::Own, ..Request::default() },
            Request { target: i64::MAX / 4, ending: Ending::Fade, ..Request::default() },
            Request { target: -5, ..Request::default() },
        ] {
            let res = fit(&a, &r, &mut reader(&m));
            if let Ok(f) = &res {
                assert!(f.plan.is_playable(), "{:?}", f.plan);
                assert!(f.len > 0);
            }
        }
        let f = fit(&a, &Request { target: at(30.0), events: many, ..Request::default() }, &mut reader(&m)).expect("fit");
        assert!(f.warnings.iter().any(|w| w.contains("left out")), "{:?}", f.warnings);
        // a track shorter than the film is repeated, at its beats, and still ends its own way
        let f = fit(&a, &Request { target: at(150.0), ending: Ending::Own, ..Request::default() }, &mut reader(&m)).expect("fit");
        assert!((secs(f.len) - 150.0).abs() < 0.6, "{}", secs(f.len));
        // a degenerate analysis is refused
        let empty = Analysis { beats: Vec::new(), sim: Vec::new(), ..a.clone() };
        assert!(fit(&empty, &req(10.0), &mut reader(&m)).is_err());
        let nan = Analysis { period: f64::NAN, ..a };
        assert!(fit(&nan, &req(10.0), &mut reader(&m)).is_err());
    }

    /// A stop as the last event cuts the music there, on a downbeat (the track has no stop of its
    /// own before its end): the film is silent after it, and the ending asked for is not used.
    #[test]
    fn a_last_stop_cuts_the_music() {
        let (m, a, _) = track(FixtureEnd::Cadence);
        let mut r = req(40.0);
        r.ending = Ending::Own;
        r.events = vec![Event { kind: EventKind::Peak, at: at(10.0) }, Event { kind: EventKind::Stop, at: at(30.0) }];
        let f = fit(&a, &r, &mut reader(&m)).expect("fit");
        assert_eq!(f.ended, "stop", "{:?}", f.events);
        let stop = f.events[1];
        assert!(stop.landed && matches!(stop.lift, Some(LiftKind::Cut | LiftKind::Stop)), "{stop:?}");
        assert!((secs(f.len) - 30.0).abs() <= 0.25, "{}", secs(f.len));
        assert_eq!(f.plan.fade_out, 0, "a hard cut");
        assert!(f.warnings.iter().any(|w| w.contains("not used")), "{:?}", f.warnings);
    }

    #[test]
    fn deterministic() {
        let (m, a, _) = track(FixtureEnd::Cadence);
        let r = Request {
            target: at(45.0),
            events: vec![Event { kind: EventKind::Peak, at: at(15.0) }, Event { kind: EventKind::Landing, at: at(40.0) }],
            ending: Ending::Own,
            ..Request::default()
        };
        assert_eq!(fit(&a, &r, &mut reader(&m)), fit(&a, &r, &mut reader(&m)));
    }
}
