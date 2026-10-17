//! The arranger: a track is a path through a graph of section blocks (intro, verse, build,
//! chorus, bridge, outro — each a few bars long), ending on the final hit. Given the number of
//! whole bars before the hit, a dynamic programme over (bar, last section) finds the path that
//! fills them **exactly**, follows the requested energy curve, keeps a song's shape (a build
//! resolves into a chorus, the intro comes first, the outro last) and keeps the intro and outro
//! in proportion. It is deterministic; the seed only nudges ties so variations differ.

/// Most points an energy curve may have.
pub const MAX_CURVE_POINTS: usize = 64;
/// Most bars arranged (15 minutes at 200 BPM is 750).
pub const MAX_BARS: u32 = 1024;

/// How busy and loud the track is over its length.
#[derive(Clone, Debug, PartialEq)]
pub enum Energy {
    /// One level (0..1), shaped into an arc: a gentler start, the peak in the middle and late
    /// part, a softer close.
    Level(f64),
    /// Levels (0..1) spread evenly from the start to the end, joined by straight lines.
    Curve(Vec<f64>),
}

impl Default for Energy {
    fn default() -> Self {
        Energy::Level(0.7)
    }
}

impl Energy {
    /// A curve from `points`: 1..=[`MAX_CURVE_POINTS`] finite numbers in 0..1.
    pub fn curve(points: &[f64]) -> Result<Energy, String> {
        if points.is_empty() || points.len() > MAX_CURVE_POINTS {
            return Err(format!("an energy curve has 1 to {MAX_CURVE_POINTS} points, not {}", points.len()));
        }
        if let Some(bad) = points.iter().find(|p| !p.is_finite() || !(0.0..=1.0).contains(*p)) {
            return Err(format!("energy {bad} is outside 0..1"));
        }
        Ok(if let [one] = points { Energy::Level(*one) } else { Energy::Curve(points.to_vec()) })
    }

    /// The energy at `x` (0 = start, 1 = end), in 0..1.
    pub fn at(&self, x: f64) -> f64 {
        let x = if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 };
        let v = match self {
            Energy::Level(l) => {
                let l = if l.is_finite() { l.clamp(0.0, 1.0) } else { 0.7 };
                l * arc(x)
            }
            Energy::Curve(p) => match p.len() {
                0 => 0.7,
                1 => p.first().copied().unwrap_or(0.7),
                n => {
                    let f = x * (n - 1) as f64;
                    let i = (f.floor() as usize).min(n - 2);
                    let (a, b) = (p.get(i).copied().unwrap_or(0.7), p.get(i + 1).copied().unwrap_or(0.7));
                    a + (b - a) * (f - i as f64)
                }
            },
        };
        if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.7 }
    }

    /// The energy asked for, as one number: the level, or a curve's average (the vocabulary's
    /// "energy" of a track).
    pub fn nominal(&self) -> f64 {
        match self {
            Energy::Level(l) if l.is_finite() => l.clamp(0.0, 1.0),
            _ => self.mean(),
        }
    }

    /// The average energy over the track, as played.
    pub fn mean(&self) -> f64 {
        (0..64).map(|i| self.at((f64::from(i) + 0.5) / 64.0)).sum::<f64>() / 64.0
    }
}

/// The shape a single energy level is played with: 0.55 of it at the start, all of it from
/// a third of the way to 85 %, 0.7 of it at the end.
fn arc(x: f64) -> f64 {
    if x < 0.35 {
        0.55 + 0.45 * smoothstep(x / 0.35)
    } else if x < 0.85 {
        1.0
    } else {
        1.0 - 0.3 * smoothstep((x - 0.85) / 0.15)
    }
}

pub(crate) fn smoothstep(x: f64) -> f64 {
    let x = if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 };
    x * x * (3.0 - 2.0 * x)
}

/// A part of a song.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SectionKind {
    Intro,
    Verse,
    Build,
    Chorus,
    Bridge,
    Outro,
}

impl SectionKind {
    pub const ALL: [SectionKind; 6] =
        [SectionKind::Intro, SectionKind::Verse, SectionKind::Build, SectionKind::Chorus, SectionKind::Bridge, SectionKind::Outro];

    pub fn id(self) -> &'static str {
        match self {
            SectionKind::Intro => "intro",
            SectionKind::Verse => "verse",
            SectionKind::Build => "build",
            SectionKind::Chorus => "chorus",
            SectionKind::Bridge => "bridge",
            SectionKind::Outro => "outro",
        }
    }

    /// The energy the section is written at, `i` bars into a block of `len` (a build rises).
    pub fn nominal(self, i: u32, len: u32) -> f64 {
        match self {
            SectionKind::Intro => 0.3,
            SectionKind::Verse => 0.55,
            SectionKind::Build => 0.55 + 0.3 * (f64::from(i) + 0.5) / f64::from(len.max(1)),
            SectionKind::Chorus => 0.9,
            SectionKind::Bridge => 0.4,
            SectionKind::Outro => 0.35,
        }
    }

    /// Block lengths in bars, with how much less usual each is.
    fn lengths(self) -> &'static [(u32, f64)] {
        match self {
            SectionKind::Intro | SectionKind::Outro => &[(1, 0.3), (2, 0.0), (4, 0.0), (8, 0.8)],
            SectionKind::Verse | SectionKind::Chorus => &[(2, 1.0), (4, 0.1), (8, 0.0)],
            SectionKind::Build => &[(1, 0.8), (2, 0.0), (4, 0.0)],
            SectionKind::Bridge => &[(2, 0.8), (4, 0.0), (8, 0.2)],
        }
    }
}

/// The cost of going from `from` (`None`: the start) into `to`; `None` when a song never does.
fn transition(from: Option<SectionKind>, to: SectionKind) -> Option<f64> {
    use SectionKind::*;
    match (from, to) {
        (None, Intro) => Some(0.0),
        (None, Verse | Chorus | Build) => Some(2.5),
        (None, _) => None,
        (Some(_), Intro) => None,
        (Some(Outro), _) => None,
        (Some(Build), Chorus) => Some(0.0),
        (Some(Build), _) => None,
        (Some(Intro), Verse) => Some(0.0),
        (Some(Intro), Build) => Some(0.3),
        (Some(Intro), Chorus) => Some(0.8),
        (Some(Intro), Bridge) => Some(1.0),
        (Some(Intro), Outro) => Some(1.0),
        (Some(Verse), Build) => Some(0.0),
        (Some(Verse), Chorus) => Some(0.2),
        (Some(Verse), Bridge) => Some(0.6),
        (Some(Verse), Verse) => Some(1.5),
        (Some(Verse), Outro) => Some(0.8),
        (Some(Chorus), Verse) => Some(0.1),
        (Some(Chorus), Bridge) => Some(0.1),
        (Some(Chorus), Outro) => Some(0.0),
        (Some(Chorus), Chorus) => Some(1.5),
        (Some(Chorus), Build) => Some(0.8),
        (Some(Bridge), Build) => Some(0.0),
        (Some(Bridge), Chorus) => Some(0.3),
        (Some(Bridge), Verse) => Some(0.4),
        (Some(Bridge), Outro) => Some(0.6),
        (Some(Bridge), Bridge) => Some(2.0),
    }
}

/// The cost of ending (the final hit) after `last`.
fn end_cost(last: SectionKind) -> f64 {
    match last {
        SectionKind::Outro => 0.0,
        SectionKind::Chorus => 0.6,
        SectionKind::Build => 0.9,
        SectionKind::Bridge => 1.5,
        SectionKind::Verse => 2.0,
        SectionKind::Intro => 2.5,
    }
}

/// How much a bar off the energy curve costs (squared difference, per bar).
const ENERGY_WEIGHT: f64 = 1.5;
/// How much each bar of intro or outro off its proportion costs.
const PROPORTION_WEIGHT: f64 = 0.25;
/// Most seeded nudge to a block's cost.
const VARIATION: f64 = 0.4;
/// How far a section pulls its bars' energy from the curve towards its own.
const CONTRAST: f64 = 0.35;

/// A block of the arrangement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    pub kind: SectionKind,
    /// First bar.
    pub start: u32,
    pub bars: u32,
}

/// The sections filling the bars before the final hit, and the energy each bar is played at.
#[derive(Clone, Debug, PartialEq)]
pub struct Arrangement {
    pub blocks: Vec<Block>,
    /// Whole bars before the final hit (the hit is on the downbeat of bar `bars`).
    pub bars: u32,
    /// Energy (0..1) of each bar, `bars + 1` of them: the last is the final hit's.
    pub energy: Vec<f64>,
}

impl Arrangement {
    /// The section bar `bar` is in (`None` for the final hit and past it).
    pub fn kind_at(&self, bar: u32) -> Option<SectionKind> {
        self.blocks.iter().find(|b| bar >= b.start && bar < b.start.saturating_add(b.bars)).map(|b| b.kind)
    }
}

/// A deterministic hash of the seed and a choice, in [0, 1).
fn nudge(seed: u64, a: u64, b: u64, c: u64) -> f64 {
    let mut x = seed ^ a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ c.wrapping_mul(0x1656_67B1_9E37_79F9);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    x = x.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// What the sections are chosen against: the energy curve, partly stretched over the sections'
/// own range, so a quiet track still has verses and a chorus (played quietly) and a loud one
/// still breathes.
struct Shape<'a> {
    energy: &'a Energy,
    lo: f64,
    hi: f64,
}

/// How much of the section target is the stretched curve (the rest is the curve itself).
const STRETCH: f64 = 0.65;

impl<'a> Shape<'a> {
    fn new(energy: &'a Energy) -> Self {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..64 {
            let e = energy.at((f64::from(i) + 0.5) / 64.0);
            lo = lo.min(e);
            hi = hi.max(e);
        }
        Shape { energy, lo, hi }
    }

    fn target(&self, x: f64) -> f64 {
        let e = self.energy.at(x);
        let stretched = if self.hi - self.lo < 0.05 { 0.6 } else { 0.3 + 0.6 * (e - self.lo) / (self.hi - self.lo) };
        (1.0 - STRETCH) * e + STRETCH * stretched
    }
}

/// The cost of playing a `kind` block of `len` bars from bar `pos` of `n`.
fn block_cost(kind: SectionKind, pos: usize, len: u32, n: usize, shape: &Shape<'_>) -> f64 {
    let mut c = 0.0;
    for i in 0..len {
        let x = (pos as f64 + f64::from(i) + 0.5) / n.max(1) as f64;
        let d = kind.nominal(i, len) - shape.target(x);
        c += d * d;
    }
    c *= ENERGY_WEIGHT;
    if matches!(kind, SectionKind::Intro | SectionKind::Outro) {
        let ideal = (n as f64 * 0.08).round().clamp(1.0, 8.0);
        c += PROPORTION_WEIGHT * (f64::from(len) - ideal).abs();
    }
    c
}

/// Arrange `bars` whole bars (at most [`MAX_BARS`]) for `energy`; `seed` picks among near-equal
/// arrangements.
pub fn arrange(bars: u32, energy: &Energy, seed: u64) -> Arrangement {
    let bars = bars.min(MAX_BARS);
    let n = bars as usize;
    let blocks =
        if n == 0 { Vec::new() } else { search(n, &Shape::new(energy), seed).unwrap_or_else(|| vec![Block { kind: SectionKind::Verse, start: 0, bars }]) };
    let mut arr = Arrangement { blocks, bars, energy: Vec::new() };
    let mut e: Vec<f64> = Vec::with_capacity(n + 1);
    for b in &arr.blocks {
        for i in 0..b.bars {
            let bar = f64::from(b.start) + f64::from(i);
            let want = energy.at((bar + 0.5) / n.max(1) as f64);
            e.push((want + CONTRAST * (b.kind.nominal(i, b.bars) - 0.6)).clamp(0.02, 1.0));
        }
    }
    // the final hit: where the curve ends, a little stronger
    e.push((energy.at(1.0) + 0.15).clamp(0.1, 1.0));
    arr.energy = e;
    arr
}

/// The cheapest block path filling exactly `n` bars (`None` if there is none, which the graph
/// never allows for `n` ≥ 1: an intro of 1, 2, 4 or 8 bars, then outros, reach every length).
fn search(n: usize, shape: &Shape<'_>, seed: u64) -> Option<Vec<Block>> {
    const STATES: usize = 1 + SectionKind::ALL.len();
    let kind_of = |st: usize| st.checked_sub(1).and_then(|k| SectionKind::ALL.get(k)).copied();
    let mut best = vec![[f64::INFINITY; STATES]; n + 1];
    // (previous bar, previous state, block length) that reached this state most cheaply
    let mut back: Vec<[Option<(usize, usize, u32)>; STATES]> = vec![[None; STATES]; n + 1];
    if let Some(b) = best.first_mut() {
        b[0] = 0.0;
    }
    for pos in 0..n {
        for st in 0..STATES {
            let c0 = best.get(pos).map_or(f64::INFINITY, |b| b[st]);
            if !c0.is_finite() {
                continue;
            }
            let from = kind_of(st);
            if st > 0 && from.is_none() {
                continue;
            }
            for (ki, kind) in SectionKind::ALL.iter().enumerate() {
                let Some(tc) = transition(from, *kind) else { continue };
                for &(len, lc) in kind.lengths() {
                    let end = pos + len as usize;
                    if end > n {
                        continue;
                    }
                    let cost = c0 + tc + lc + block_cost(*kind, pos, len, n, shape) + VARIATION * nudge(seed, pos as u64, ki as u64, u64::from(len));
                    if let (Some(b), Some(k)) = (best.get_mut(end), back.get_mut(end))
                        && cost < b[ki + 1]
                    {
                        b[ki + 1] = cost;
                        k[ki + 1] = Some((pos, st, len));
                    }
                }
            }
        }
    }
    let last = best.get(n)?;
    let (mut st, _) = (1..STATES).filter_map(|st| Some((st, last[st] + end_cost(kind_of(st)?)))).filter(|(_, c)| c.is_finite()).fold(
        None,
        |acc: Option<(usize, f64)>, (st, c)| match acc {
            Some((_, bc)) if bc <= c => acc,
            _ => Some((st, c)),
        },
    )?;
    let mut pos = n;
    let mut out = Vec::new();
    // at most n blocks: every block is at least a bar long
    for _ in 0..=n {
        if pos == 0 {
            break;
        }
        let (prev, prev_st, len) = back.get(pos)?[st]?;
        out.push(Block { kind: kind_of(st)?, start: u32::try_from(prev).ok()?, bars: len });
        pos = prev;
        st = prev_st;
    }
    if pos != 0 {
        return None;
    }
    out.reverse();
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(a: &Arrangement) {
        let mut at = 0;
        for b in &a.blocks {
            assert_eq!(b.start, at, "{:?}", a.blocks);
            assert!(b.bars >= 1);
            at += b.bars;
        }
        assert_eq!(at, a.bars, "{:?}", a.blocks);
        assert_eq!(a.energy.len(), a.bars as usize + 1);
        assert!(a.energy.iter().all(|e| (0.0..=1.0).contains(e)));
        // the graph's rules hold
        for w in a.blocks.windows(2) {
            assert!(transition(Some(w[0].kind), w[1].kind).is_some(), "{:?} → {:?}", w[0].kind, w[1].kind);
        }
        if let Some(first) = a.blocks.first() {
            assert!(transition(None, first.kind).is_some());
        }
    }

    #[test]
    fn every_length_is_filled_exactly() {
        for bars in 0..=200 {
            for seed in [1, 7, 12345] {
                let a = arrange(bars, &Energy::Level(0.7), seed);
                check(&a);
            }
        }
        check(&arrange(MAX_BARS + 50, &Energy::Level(0.9), 3));
        assert_eq!(arrange(u32::MAX, &Energy::Level(0.9), 3).bars, MAX_BARS);
    }

    #[test]
    fn songs_have_a_shape() {
        let a = arrange(48, &Energy::Level(0.8), 1);
        let kinds: Vec<SectionKind> = a.blocks.iter().map(|b| b.kind).collect();
        assert_eq!(kinds.first(), Some(&SectionKind::Intro), "{kinds:?}");
        assert!(matches!(kinds.last(), Some(SectionKind::Outro | SectionKind::Chorus)), "{kinds:?}");
        assert!(kinds.contains(&SectionKind::Chorus), "an energetic track has a chorus: {kinds:?}");
        assert!(kinds.contains(&SectionKind::Verse), "{kinds:?}");
        // a calm one still has a shape, played quietly
        let calm = arrange(48, &Energy::Level(0.2), 1);
        assert!(calm.blocks.iter().any(|b| matches!(b.kind, SectionKind::Verse | SectionKind::Chorus)), "{:?}", calm.blocks);
        assert!(calm.energy.iter().all(|e| *e < 0.45), "{:?}", calm.energy);
        // seeds vary the arrangement, the same seed repeats it
        let variants: std::collections::HashSet<Vec<(u32, u32)>> =
            (0..12).map(|s| arrange(48, &Energy::Level(0.8), s).blocks.iter().map(|b| (b.kind as u32, b.bars)).collect()).collect();
        assert!(variants.len() > 1, "seeds give variations");
        assert_eq!(arrange(48, &Energy::Level(0.8), 5), arrange(48, &Energy::Level(0.8), 5));
    }

    #[test]
    fn energy_follows_the_curve() {
        let rising = arrange(40, &Energy::Curve(vec![0.1, 0.95]), 2);
        let falling = arrange(40, &Energy::Curve(vec![0.95, 0.1]), 2);
        let half = |a: &Arrangement, first: bool| {
            let e = &a.energy[..a.bars as usize];
            let (x, y) = e.split_at(e.len() / 2);
            let v = if first { x } else { y };
            v.iter().sum::<f64>() / v.len() as f64
        };
        assert!(half(&rising, false) > half(&rising, true) + 0.3, "{:?}", rising.energy);
        assert!(half(&falling, true) > half(&falling, false) + 0.3, "{:?}", falling.energy);
        // the high end gets the chorus
        let last_chorus = rising.blocks.iter().filter(|b| b.kind == SectionKind::Chorus).map(|b| b.start).max();
        assert!(last_chorus.is_some_and(|s| s >= 20), "{:?}", rising.blocks);
    }

    #[test]
    fn curves_are_validated_and_read() {
        assert!(Energy::curve(&[]).is_err());
        assert!(Energy::curve(&[0.5; MAX_CURVE_POINTS + 1]).is_err());
        assert!(Energy::curve(&[0.5, f64::NAN]).is_err());
        assert!(Energy::curve(&[0.5, 1.5]).is_err());
        assert_eq!(Energy::curve(&[0.4]), Ok(Energy::Level(0.4)));
        let c = Energy::curve(&[0.0, 1.0, 0.0]).unwrap();
        assert!((c.at(0.25) - 0.5).abs() < 1e-9 && (c.at(0.5) - 1.0).abs() < 1e-9 && c.at(1.0).abs() < 1e-9);
        assert!((c.at(f64::NAN)).abs() < 1e-9 && c.at(7.0).abs() < 1e-9);
        let l = Energy::Level(0.8);
        assert!(l.at(0.0) < l.at(0.5) && l.at(1.0) < l.at(0.5) && (l.at(0.5) - 0.8).abs() < 1e-9);
        assert!((Energy::Level(f64::NAN).at(0.5) - 0.7).abs() < 1e-9);
    }
}
