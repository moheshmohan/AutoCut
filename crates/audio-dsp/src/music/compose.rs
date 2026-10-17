//! The composer: harmony for every bar of an arrangement, and the notes of every part — drums,
//! bass, melody (lead and arpeggio) and pad (pad and keys) — written as patterns over the chord
//! and energy of each bar, on exact sample positions from the tempo's ticks.

use super::Plan;
use super::arrange::{Arrangement, SectionKind};
use super::instruments::{Drum, Fx, Kit, Patch, Rng};
use super::taxonomy::{Genre, Mood};
use super::tempo::{MILLI, STEPS_PER_BAR, sample_of};

/// The four layers a track is rendered as; they sum to the mix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Stem {
    Drums,
    Bass,
    Melody,
    Pad,
}

impl Stem {
    pub const ALL: [Stem; 4] = [Stem::Drums, Stem::Bass, Stem::Melody, Stem::Pad];

    pub fn id(self) -> &'static str {
        match self {
            Stem::Drums => "drums",
            Stem::Bass => "bass",
            Stem::Melody => "melody",
            Stem::Pad => "pad",
        }
    }

    pub fn from_id(s: &str) -> Option<Stem> {
        Stem::ALL.into_iter().find(|x| x.id().eq_ignore_ascii_case(s.trim()))
    }

    pub fn index(self) -> usize {
        match self {
            Stem::Drums => 0,
            Stem::Bass => 1,
            Stem::Melody => 2,
            Stem::Pad => 3,
        }
    }
}

/// A chord's quality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    Major,
    Minor,
    Major7,
    Minor7,
    Dominant7,
}

/// A chord: its root in semitones above the key's tonic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chord {
    pub root: i32,
    pub quality: Quality,
}

const NOTE_NAMES: [&str; 12] = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];

/// The pitch class (C = 0) of the tonic of `key` (semitones above A).
pub fn tonic_pc(key: i32) -> i32 {
    (9 + key).rem_euclid(12)
}

/// The name of pitch class `pc` (C = 0).
pub fn pc_name(pc: i32) -> &'static str {
    NOTE_NAMES.get(pc.rem_euclid(12) as usize).copied().unwrap_or("C")
}

impl Chord {
    const fn new(root: i32, quality: Quality) -> Self {
        Chord { root, quality }
    }

    /// Semitones above the root.
    pub fn intervals(self) -> &'static [i32] {
        match self.quality {
            Quality::Major => &[0, 4, 7],
            Quality::Minor => &[0, 3, 7],
            Quality::Major7 => &[0, 4, 7, 11],
            Quality::Minor7 => &[0, 3, 7, 10],
            Quality::Dominant7 => &[0, 4, 7, 10],
        }
    }

    /// Its root's pitch class (C = 0) in `key`.
    pub fn root_pc(self, key: i32) -> i32 {
        (tonic_pc(key) + self.root).rem_euclid(12)
    }

    /// Its pitch classes (C = 0) in `key`.
    fn pcs(self, key: i32) -> impl Iterator<Item = i32> {
        let r = self.root_pc(key);
        self.intervals().iter().map(move |i| (r + i).rem_euclid(12))
    }

    fn third(self) -> i32 {
        if matches!(self.quality, Quality::Minor | Quality::Minor7) { 3 } else { 4 }
    }

    /// Its name in `key` ("Am", "Fmaj7", "G7").
    pub fn name(self, key: i32) -> String {
        let suffix = match self.quality {
            Quality::Major => "",
            Quality::Minor => "m",
            Quality::Major7 => "maj7",
            Quality::Minor7 => "m7",
            Quality::Dominant7 => "7",
        };
        format!("{}{suffix}", pc_name(self.root_pc(key)))
    }
}

use Quality::{Dominant7 as D7, Major as Ma, Major7 as M7, Minor as Mi, Minor7 as m7};

/// Four-chord progressions for the verse (and intro), chorus (and outro) and bridge.
struct Progressions {
    verse: [Chord; 4],
    chorus: [Chord; 4],
    bridge: [Chord; 4],
}

const fn c(root: i32, q: Quality) -> Chord {
    Chord::new(root, q)
}

fn progressions(mood: Mood) -> Progressions {
    match mood {
        // i–VI–III–VII / VI–III–VII–i / iv–VI–i–V
        Mood::Epic => Progressions {
            verse: [c(0, Mi), c(8, Ma), c(3, Ma), c(10, Ma)],
            chorus: [c(8, Ma), c(3, Ma), c(10, Ma), c(0, Mi)],
            bridge: [c(5, Mi), c(8, Ma), c(0, Mi), c(7, Ma)],
        },
        Mood::Tense => Progressions {
            verse: [c(0, Mi), c(0, Mi), c(8, Ma), c(7, Ma)],
            chorus: [c(0, Mi), c(5, Mi), c(8, Ma), c(7, Ma)],
            bridge: [c(8, Ma), c(5, Mi), c(0, Mi), c(7, Ma)],
        },
        // vi–IV–I–V / I–V–vi–IV / ii–IV–I–V
        Mood::Uplifting => Progressions {
            verse: [c(9, Mi), c(5, Ma), c(0, Ma), c(7, Ma)],
            chorus: [c(0, Ma), c(7, Ma), c(9, Mi), c(5, Ma)],
            bridge: [c(2, Mi), c(5, Ma), c(0, Ma), c(7, Ma)],
        },
        Mood::Calm => Progressions {
            verse: [c(0, M7), c(9, m7), c(5, M7), c(7, Ma)],
            chorus: [c(5, M7), c(0, Ma), c(7, Ma), c(9, m7)],
            bridge: [c(2, m7), c(7, Ma), c(0, M7), c(5, M7)],
        },
        Mood::Happy => Progressions {
            verse: [c(0, Ma), c(5, Ma), c(7, Ma), c(5, Ma)],
            chorus: [c(0, Ma), c(7, Ma), c(9, Mi), c(5, Ma)],
            bridge: [c(5, Ma), c(7, Ma), c(4, Mi), c(9, Mi)],
        },
        // i–VI–iv–V / i–VII–VI–V (Andalusian) / iv–i–VI–V
        Mood::Dark => Progressions {
            verse: [c(0, Mi), c(8, Ma), c(5, Mi), c(7, Ma)],
            chorus: [c(0, Mi), c(10, Ma), c(8, Ma), c(7, Ma)],
            bridge: [c(5, Mi), c(0, Mi), c(8, Ma), c(7, Ma)],
        },
        Mood::Dreamy => Progressions {
            verse: [c(0, M7), c(9, m7), c(2, m7), c(7, D7)],
            chorus: [c(5, M7), c(4, m7), c(9, m7), c(7, D7)],
            bridge: [c(2, m7), c(7, D7), c(0, M7), c(5, M7)],
        },
        Mood::Energetic => Progressions {
            verse: [c(0, Mi), c(0, Mi), c(8, Ma), c(10, Ma)],
            chorus: [c(0, Mi), c(10, Ma), c(8, Ma), c(10, Ma)],
            bridge: [c(5, Mi), c(8, Ma), c(3, Ma), c(10, Ma)],
        },
        Mood::Sad => Progressions {
            verse: [c(0, Mi), c(5, Mi), c(10, Ma), c(3, Ma)],
            chorus: [c(8, Ma), c(3, Ma), c(5, Mi), c(7, Ma)],
            bridge: [c(5, Mi), c(0, Mi), c(8, Ma), c(7, Ma)],
        },
    }
}

/// The chord before the final hit (the cadence): V, or V7 in the jazzier styles.
fn dominant(mood: Mood, genre: Genre) -> Chord {
    if mood == Mood::Dreamy || genre == Genre::Lofi { c(7, D7) } else { c(7, Ma) }
}

/// The chord of the final hit.
fn tonic(mood: Mood) -> Chord {
    if mood.minor() {
        c(0, Mi)
    } else if mood == Mood::Dreamy {
        c(0, M7)
    } else {
        c(0, Ma)
    }
}

/// The chord of every bar of `arr`, and the tonic for the final hit (`bars + 1` chords): each
/// section plays its progression, a build turns to the dominant, the last bar before the hit
/// is the dominant so the track ends on a cadence.
pub fn harmony(mood: Mood, genre: Genre, arr: &Arrangement) -> Vec<Chord> {
    let p = progressions(mood);
    let dom = dominant(mood, genre);
    let at = |a: &[Chord; 4], i: u32| a.get((i % 4) as usize).copied().unwrap_or(c(0, Mi));
    let mut out = Vec::with_capacity(arr.bars as usize + 1);
    for b in &arr.blocks {
        for i in 0..b.bars {
            out.push(match b.kind {
                SectionKind::Intro | SectionKind::Verse => at(&p.verse, i),
                SectionKind::Chorus | SectionKind::Outro => at(&p.chorus, i),
                SectionKind::Bridge => at(&p.bridge, i),
                SectionKind::Build if i + 1 == b.bars => dom,
                SectionKind::Build => at(&p.bridge, i),
            });
        }
    }
    out.truncate(arr.bars as usize);
    if let Some(last) = out.last_mut() {
        *last = dom;
    }
    out.push(tonic(mood));
    out
}

// ---------------------------------------------------------------------------------------------
// the score

/// A note: `start` and `len` in samples; `key` a MIDI note (or a drum / effect key).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Note {
    pub start: u64,
    pub len: u64,
    pub key: u8,
    pub velocity: f32,
}

/// One instrument's notes on one stem.
#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    pub stem: Stem,
    pub patch: Patch,
    pub notes: Vec<Note>,
    /// Side-chain pumping depth (0..1) per bar, empty for none.
    pub pump: Vec<f32>,
}

/// Every part of a track.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Score {
    pub parts: Vec<Part>,
}

/// The instruments of a style.
struct Band {
    kit: Kit,
    bass: Option<Patch>,
    pad: Option<Patch>,
    keys: Option<Patch>,
    lead: Patch,
    arp: Option<Patch>,
    /// Thousandths of a sixteenth the off-beat sixteenths are late by.
    swing: u64,
    pump: bool,
    /// Lowest and highest lead note.
    lead_range: (i32, i32),
}

fn band(mood: Mood, genre: Genre) -> Band {
    match genre {
        Genre::Cinematic => Band {
            kit: Kit::Cinematic,
            bass: Some(Patch::SynthBass),
            pad: Some(Patch::Strings),
            keys: None,
            lead: match mood {
                Mood::Epic | Mood::Uplifting | Mood::Energetic => Patch::Brass,
                Mood::Tense | Mood::Dark => Patch::SawLead,
                _ => Patch::Flute,
            },
            arp: Some(Patch::Pluck),
            swing: 0,
            pump: false,
            lead_range: (60, 79),
        },
        Genre::Electronic => Band {
            kit: Kit::Electronic,
            bass: Some(if mood == Mood::Happy { Patch::SynthBass } else { Patch::SubBass }),
            pad: Some(Patch::SuperSaw),
            keys: None,
            lead: if mood == Mood::Happy { Patch::Pluck } else { Patch::SawLead },
            arp: Some(Patch::Pluck),
            swing: 0,
            pump: true,
            lead_range: (67, 86),
        },
        Genre::Pop => Band {
            kit: Kit::Pop,
            bass: Some(if mood == Mood::Happy { Patch::RoundBass } else { Patch::SynthBass }),
            pad: Some(Patch::WarmPad),
            keys: Some(Patch::Piano),
            lead: if mood == Mood::Happy { Patch::Flute } else { Patch::Pluck },
            arp: if matches!(mood, Mood::Uplifting | Mood::Energetic) { Some(Patch::Pluck) } else { None },
            swing: if mood == Mood::Happy { 60 } else { 0 },
            pump: false,
            lead_range: (67, 84),
        },
        Genre::Ambient => Band {
            kit: Kit::Soft,
            bass: Some(Patch::SubBass),
            pad: Some(Patch::WarmPad),
            keys: None,
            lead: Patch::Bell,
            arp: Some(Patch::Bell),
            swing: 0,
            pump: false,
            lead_range: (72, 91),
        },
        Genre::Lofi => Band {
            kit: Kit::Lofi,
            bass: Some(Patch::RoundBass),
            pad: None,
            keys: Some(Patch::EPiano),
            lead: Patch::Bell,
            arp: None,
            swing: 280,
            pump: false,
            lead_range: (67, 84),
        },
        Genre::Piano => Band {
            kit: Kit::Soft,
            bass: None,
            pad: Some(Patch::Strings),
            keys: Some(Patch::Piano),
            lead: Patch::Piano,
            arp: None,
            swing: 0,
            pump: false,
            lead_range: (64, 84),
        },
    }
}

/// How busy a bar of energy `e` is: 0 (barely anything) .. 4 (everything).
fn tier(e: f64) -> usize {
    if e < 0.15 {
        0
    } else if e < 0.35 {
        1
    } else if e < 0.55 {
        2
    } else if e < 0.75 {
        3
    } else {
        4
    }
}

type Lines = &'static [(Drum, &'static str)];

/// The drum lines of a bar: one sixteenth per character, `x` a hit, `o` a soft one.
fn drum_lines(genre: Genre, tier: usize) -> Lines {
    use Drum::*;
    match (genre, tier) {
        (Genre::Cinematic, 0) => &[],
        (Genre::Cinematic, 1) => &[(Taiko, "x...o...x...o...")],
        (Genre::Cinematic, 2) => &[(Taiko, "x.....x.x......."), (HatClosed, "o...o...o...o...")],
        (Genre::Cinematic, 3) => &[(Taiko, "x.....x.x...o..."), (Snare, "....x.......x..."), (HatClosed, "x.o.x.o.x.o.x.o.")],
        (Genre::Cinematic, _) => &[
            (Taiko, "x.o...x.x.o...o."),
            (Kick, "x.......x......."),
            (Snare, "....x.......x..."),
            (HatClosed, "xoxoxoxoxoxoxoxo"),
            (TomLow, "..............o."),
        ],
        (Genre::Electronic, 0) => &[],
        (Genre::Electronic, 1) => &[(Kick, "x.......x......."), (HatClosed, "..o...o...o...o.")],
        (Genre::Electronic, 2) => &[(Kick, "x...x...x...x..."), (HatOpen, "..o...o...o...o.")],
        (Genre::Electronic, 3) => &[(Kick, "x...x...x...x..."), (Clap, "....x.......x..."), (HatOpen, "..x...x...x...x."), (HatClosed, "o.o.o.o.o.o.o.o.")],
        (Genre::Electronic, _) => &[
            (Kick, "x...x...x...x..."),
            (Clap, "....x.......x..."),
            (HatOpen, "..x...x...x...x."),
            (HatClosed, "oxoxoxoxoxoxoxox"),
            (Shaker, "o.o.o.o.o.o.o.o."),
        ],
        (Genre::Pop, 0) => &[],
        (Genre::Pop, 1) => &[(Kick, "x.......x......."), (HatClosed, "o.o.o.o.o.o.o.o.")],
        (Genre::Pop, 2) => &[(Kick, "x.......x.x....."), (Snare, "....x.......x..."), (HatClosed, "x.o.x.o.x.o.x.o.")],
        (Genre::Pop, 3) => &[
            (Kick, "x..x....x.x....."),
            (Snare, "....x.......x..."),
            (Clap, "....o.......o..."),
            (HatClosed, "x.o.x.o.x.o.x.o."),
            (HatOpen, "..............o."),
        ],
        (Genre::Pop, _) => &[
            (Kick, "x..x..x.x.x....."),
            (Snare, "....x.......x..o"),
            (Clap, "....x.......x..."),
            (HatClosed, "xoxoxoxoxoxoxoxo"),
            (HatOpen, "......o.......o."),
        ],
        (Genre::Lofi, 0) => &[(HatClosed, "o...o...o...o...")],
        (Genre::Lofi, 1) => &[(Kick, "x.........x....."), (Rim, "....o.......o..."), (HatClosed, "o.o.o.o.o.o.o.o.")],
        (Genre::Lofi, 2) => &[(Kick, "x......x..x....."), (Snare, "....x.......x..."), (HatClosed, "x.o.x.o.x.o.x.o.")],
        (Genre::Lofi, _) => &[(Kick, "x......x..x..x.."), (Snare, "....x.......x..o"), (HatClosed, "x.oox.o.x.oox.o."), (Shaker, "..o...o...o...o.")],
        (Genre::Ambient, 0 | 1) => &[],
        (Genre::Ambient, 2) => &[(Shaker, "..o...o...o...o.")],
        (Genre::Ambient, 3) => &[(Kick, "x.......x......."), (Shaker, "o.o.o.o.o.o.o.o.")],
        (Genre::Ambient, _) => &[(Kick, "x.......x......."), (Rim, "....o.......o..."), (Shaker, "o.o.o.o.o.o.o.o.")],
        (Genre::Piano, 0..=2) => &[],
        (Genre::Piano, 3) => &[(Taiko, "x.......o.......")],
        (Genre::Piano, _) => &[(Taiko, "x.....x.x......."), (Snare, "....o.......o...")],
    }
}

/// The second half of a bar leading into a new section (eight sixteenths).
fn fill_lines(genre: Genre) -> Lines {
    use Drum::*;
    match genre {
        Genre::Cinematic => &[(Taiko, "x.x.x.xx"), (TomLow, "....x.x.")],
        Genre::Electronic => &[(Snare, "oooxxxxx")],
        Genre::Pop => &[(TomHigh, "x.x....."), (TomMid, "....x.x."), (TomLow, "......xx"), (Kick, "x...x...")],
        Genre::Lofi => &[(Snare, "..o..x.x"), (Kick, "x......."), (HatClosed, "o.o.o.o.")],
        Genre::Ambient => &[],
        Genre::Piano => &[(Taiko, "x...x.x.")],
    }
}

/// Lines a fill replaces in the second half of the bar.
fn filled(d: Drum) -> bool {
    matches!(d, Drum::Snare | Drum::Clap | Drum::TomLow | Drum::TomMid | Drum::TomHigh | Drum::HatOpen | Drum::Taiko | Drum::Rim)
}

/// The bass of a bar: `R` the root, `F` the fifth, `O` the octave, `-` holds, `.` rests.
fn bass_pattern(genre: Genre, tier: usize) -> &'static str {
    match (genre, tier) {
        (Genre::Cinematic, 0) => "................",
        (Genre::Cinematic, 1) => "R---------------",
        (Genre::Cinematic, 2) => "R-------R-------",
        (Genre::Cinematic, 3) => "R.R.R.R.R.R.R.R.",
        (Genre::Cinematic, _) => "R.RRR.RRR.RRR.RR",
        (Genre::Electronic, 0) => "................",
        (Genre::Electronic, 1) => "R---------------",
        (Genre::Electronic, 2) => "..R...R...R...R.",
        (Genre::Electronic, 3) => "..R.R.R...R.R.RO",
        (Genre::Electronic, _) => "..RR..RR..RR..RO",
        (Genre::Pop, 0) => "................",
        (Genre::Pop, 1) => "R-------R-------",
        (Genre::Pop, 2) => "R-----R-R-------",
        (Genre::Pop, 3) => "R.R.R.R.R.R.R.F.",
        (Genre::Pop, _) => "R.RRR.R.O.R.R.F.",
        (Genre::Lofi, 0) => "R---------------",
        (Genre::Lofi, 1) => "R-------F-------",
        (Genre::Lofi, 2) => "R-----R...R-----",
        (Genre::Lofi, _) => "R-----RF..R--O--",
        (Genre::Ambient, 0..=3) => "R-------R-------",
        (Genre::Ambient, _) => "R...R...R...R...",
        (Genre::Piano, _) => "................",
    }
}

/// Keys comping: `x` the chord, `-` holds, `.` rests.
fn comp_pattern(genre: Genre, tier: usize) -> &'static str {
    match (genre, tier) {
        (Genre::Lofi, 0) => "x---------------",
        (Genre::Lofi, 1) => "x-------x-------",
        (Genre::Lofi, 2) => "x-----x---x-----",
        (Genre::Lofi, _) => "x-----x-x---x---",
        (_, 0) => "x---------------",
        (_, 1) => "x-------x-------",
        (_, 2) => "x-----x-x-------",
        (_, 3) => "x..x..x...x..x..",
        (_, _) => "x.xx.x.xx.xx.x.x",
    }
}

/// 2-bar melody rhythms: (sixteenth, length) pairs.
const RHYTHMS: [&[(u64, u64)]; 6] = [
    &[(0, 4), (4, 2), (6, 2), (8, 4), (12, 4), (16, 6), (22, 2), (24, 8)],
    &[(0, 3), (3, 3), (6, 2), (8, 8), (16, 3), (19, 3), (22, 2), (24, 8)],
    &[(0, 2), (2, 2), (4, 2), (6, 2), (8, 4), (12, 4), (16, 2), (18, 2), (20, 4), (24, 8)],
    &[(0, 8), (8, 8), (16, 12), (28, 4)],
    &[(0, 3), (3, 5), (8, 2), (10, 6), (16, 3), (19, 5), (24, 8)],
    &[(2, 2), (4, 4), (8, 2), (10, 2), (12, 4), (18, 2), (20, 4), (24, 6)],
];

/// Orders of chord tones arpeggios walk.
const ARPS: [[usize; 8]; 3] = [[0, 1, 2, 3, 2, 1, 2, 3], [0, 2, 1, 3, 0, 2, 1, 3], [0, 1, 2, 1, 3, 2, 1, 2]];

/// A 2-bar melodic idea: a rhythm and how far each note moves along the scale.
#[derive(Clone, Debug)]
struct Motif {
    notes: Vec<(u64, u64, i32)>,
}

fn motif(rng: &mut Rng, sparse: bool) -> Motif {
    let pool: &[usize] = if sparse { &[1, 3, 4, 5] } else { &[0, 1, 2, 4] };
    let pick = pool.get(rng.below(pool.len())).copied().unwrap_or(0);
    let rhythm = RHYTHMS.get(pick).copied().unwrap_or(&[(0, 16)]);
    let mut dir: i32 = if rng.unit() < 0.5 { 1 } else { -1 };
    let mut at: i32 = 0;
    let notes = rhythm
        .iter()
        .enumerate()
        .map(|(i, &(s, l))| {
            if i == 0 {
                return (s, l, 0);
            }
            if rng.unit() < 0.35 || at.abs() >= 4 {
                dir = if at.abs() >= 4 { -at.signum() } else { -dir };
            }
            let step = if rng.unit() < 0.1 {
                0
            } else if rng.unit() < 0.7 {
                1
            } else {
                2
            };
            at += dir * step;
            (s, l, dir * step)
        })
        .collect();
    Motif { notes }
}

/// Scale (semitones above the tonic) for a bar over `chord`: the harmonic minor's leading
/// tone over a major V in a minor key.
fn scale(minor: bool, chord: Chord) -> [i32; 7] {
    if !minor {
        [0, 2, 4, 5, 7, 9, 11]
    } else if chord.root == 7 && !matches!(chord.quality, Quality::Minor | Quality::Minor7) {
        [0, 2, 3, 5, 7, 8, 11]
    } else {
        [0, 2, 3, 5, 7, 8, 10]
    }
}

/// `note` moved `delta` steps along `sc` (in `key`).
fn scale_step(note: i32, delta: i32, key: i32, sc: &[i32; 7]) -> i32 {
    let tonic = tonic_pc(key);
    let rel = note - tonic;
    let oct = rel.div_euclid(12);
    let pc = rel.rem_euclid(12);
    let idx = sc.iter().rposition(|s| *s <= pc).unwrap_or(0) as i32;
    let total = oct * 7 + idx + delta.clamp(-14, 14);
    let o = total.div_euclid(7);
    let i = total.rem_euclid(7) as usize;
    tonic + o * 12 + sc.get(i).copied().unwrap_or(0)
}

/// The note of pitch class `pcs` (C = 0) nearest `note` (the lower one on a tie).
fn nearest_pc(note: i32, pcs: impl IntoIterator<Item = i32>) -> i32 {
    let mut best = note;
    let mut dist = i32::MAX;
    for pc in pcs {
        let base = note - (note - pc).rem_euclid(12);
        for cand in [base - 12, base, base + 12] {
            let d = (cand - note).abs();
            if d < dist || (d == dist && cand < best) {
                dist = d;
                best = cand;
            }
        }
    }
    best
}

/// The tone of `chord` (in `key`) nearest `note`.
fn nearest_chord_tone(note: i32, chord: Chord, key: i32) -> i32 {
    nearest_pc(note, chord.pcs(key))
}

/// `note` folded by octaves into `lo..=hi` (`hi - lo` at least an octave).
fn fold(note: i32, lo: i32, hi: i32) -> i32 {
    let hi = hi.max(lo + 12);
    if note > hi {
        note - ((note - hi + 11) / 12) * 12
    } else if note < lo {
        note + ((lo - note + 11) / 12) * 12
    } else {
        note
    }
}

/// A chord voiced closely in the octave from `lo`.
fn voicing(chord: Chord, key: i32, lo: i32) -> Vec<i32> {
    let mut v: Vec<i32> = chord.pcs(key).map(|pc| lo + (pc - lo).rem_euclid(12)).collect();
    v.sort_unstable();
    v
}

/// The root of `chord` in the bass register (E1..D#2).
fn bass_root(chord: Chord, key: i32) -> i32 {
    28 + (chord.root_pc(key) - 4).rem_euclid(12)
}

fn midi(n: i32) -> u8 {
    n.clamp(0, 127) as u8
}

/// Places notes at exact samples.
struct Writer<'a> {
    plan: &'a Plan,
    swing: u64,
}

impl Writer<'_> {
    /// The sample of sixteenth `step` of bar `bar` (off-beat sixteenths swung).
    fn at(&self, bar: u32, step: u64) -> u64 {
        let s = u64::from(bar).saturating_mul(STEPS_PER_BAR).saturating_add(step);
        let swing = if s % 2 == 1 { self.swing } else { 0 };
        let t = self.plan.tempo.tick_at(s.saturating_mul(MILLI).saturating_add(swing));
        sample_of(t, self.plan.sample_rate).min(self.plan.samples)
    }

    /// A note from sixteenth `step` of `bar`, `len` sixteenths long (`gate` of it sounding).
    fn note(&self, bar: u32, step: u64, len: u64, key: i32, vel: f64, gate: f64) -> Option<Note> {
        let start = self.at(bar, step);
        let end = self.at(bar, step.saturating_add(len.max(1)));
        let len = ((end.saturating_sub(start)) as f64 * gate.clamp(0.05, 1.0)) as u64;
        (start < self.plan.samples && len > 0).then_some(Note { start, len: len.max(1), key: midi(key), velocity: vel.clamp(0.0, 1.0) as f32 })
    }
}

/// Velocity for a hit marked `ch` at energy `e`, humanized.
fn hit_velocity(ch: char, e: f64, rng: &mut Rng) -> Option<f64> {
    let base = match ch {
        'X' => 1.0,
        'x' => 0.9,
        'o' => 0.5,
        _ => return None,
    };
    Some((base * (0.55 + 0.45 * e) * (0.95 + 0.1 * rng.unit())).clamp(0.0, 1.0))
}

/// The score of `plan`.
pub fn compose(plan: &Plan) -> Score {
    let arr = &plan.arrangement;
    let genre = plan.genre;
    let mood = plan.mood;
    let key = plan.key;
    let minor = mood.minor();
    let b = band(mood, genre);
    let w = Writer { plan, swing: b.swing };
    let mut rng = Rng::new(plan.seed ^ 0x6D75_7369_6321);
    let bars = arr.bars;
    let energy = |bar: u32| arr.energy.get(bar as usize).copied().unwrap_or(0.5);
    let chord = |bar: u32| plan.chords.get(bar as usize).copied().unwrap_or(c(0, Mi));
    let hit = sample_of(plan.ending, plan.sample_rate).min(plan.samples);
    let tail = plan.samples.saturating_sub(hit);
    let sr = f64::from(plan.sample_rate);
    let e_hit = energy(bars);
    // the final chord rings until shortly before the end: released `release` seconds early
    let ring = |release: f64| -> u64 {
        let r = (release * sr) as u64;
        let len = tail.saturating_sub(r.saturating_add((0.05 * sr) as u64));
        len.max(tail / 3).max(1)
    };

    let mut drums = Part { stem: Stem::Drums, patch: Patch::Kit(b.kit), notes: Vec::new(), pump: Vec::new() };
    let mut fx = Part { stem: Stem::Drums, patch: Patch::Fx, notes: Vec::new(), pump: Vec::new() };
    let mut bass = b.bass.map(|p| Part { stem: Stem::Bass, patch: p, notes: Vec::new(), pump: Vec::new() });
    let mut pad = b.pad.map(|p| Part { stem: Stem::Pad, patch: p, notes: Vec::new(), pump: Vec::new() });
    let mut keys = b.keys.map(|p| Part { stem: Stem::Pad, patch: p, notes: Vec::new(), pump: Vec::new() });
    let mut lead = Part { stem: Stem::Melody, patch: b.lead, notes: Vec::new(), pump: Vec::new() };
    let mut arp = b.arp.map(|p| Part { stem: Stem::Melody, patch: p, notes: Vec::new(), pump: Vec::new() });

    // ---- drums
    for blk in &arr.blocks {
        let next = arr.blocks.iter().find(|n| n.start == blk.start.saturating_add(blk.bars)).map(|n| n.kind);
        for i in 0..blk.bars {
            let bar = blk.start.saturating_add(i);
            let e = energy(bar);
            let t = tier(e);
            let last_of_block = i + 1 == blk.bars;
            let fill = t >= 2 && last_of_block && (matches!(next, Some(SectionKind::Chorus | SectionKind::Build)) || bar + 1 == bars);
            for (d, line) in drum_lines(genre, t) {
                for (s, ch) in line.chars().enumerate() {
                    let s = s as u64;
                    if fill && s >= 8 && filled(*d) {
                        continue;
                    }
                    if let Some(v) = hit_velocity(ch, e, &mut rng)
                        && let Some(n) = w.note(bar, s, 1, i32::from(d.gm_key()), v, 1.0)
                    {
                        drums.notes.push(n);
                    }
                }
            }
            if fill {
                for (d, line) in fill_lines(genre) {
                    let n_ch = line.chars().count().max(1) as f64;
                    for (s, ch) in line.chars().enumerate() {
                        // rolls swell into the next section
                        let swell = 0.6 + 0.4 * s as f64 / n_ch;
                        if let Some(v) = hit_velocity(ch, e, &mut rng)
                            && let Some(n) = w.note(bar, 8 + s as u64, 1, i32::from(d.gm_key()), v * swell, 1.0)
                        {
                            drums.notes.push(n);
                        }
                    }
                }
            }
            // a crash opens a chorus
            let opens_chorus = i == 0 && blk.kind == SectionKind::Chorus;
            if opens_chorus && t >= 2 && !matches!(genre, Genre::Ambient | Genre::Lofi | Genre::Piano) {
                // held two bars: it rings out on its own before it is damped
                drums.notes.extend(w.note(bar, 0, 2 * STEPS_PER_BAR, i32::from(Drum::Crash.gm_key()), 0.6 + 0.3 * e, 1.0));
            }
        }
        // a riser over the end of a build
        if blk.kind == SectionKind::Build && matches!(genre, Genre::Cinematic | Genre::Electronic | Genre::Pop) {
            let e = energy(blk.start);
            if e >= 0.3 {
                let from = blk.start.saturating_add(blk.bars.saturating_sub(2));
                let len = u64::from(blk.start.saturating_add(blk.bars).saturating_sub(from)) * STEPS_PER_BAR;
                fx.notes.extend(w.note(from, 0, len, i32::from(Fx::Riser.key()), 0.5 + 0.5 * e, 1.0));
            }
        }
    }
    // the final hit
    let ht = tier(e_hit);
    if hit < plan.samples {
        let hit_note = |part: &mut Part, k: u8, v: f64, len: u64| {
            part.notes.push(Note { start: hit, len: len.max(1), key: k, velocity: v.clamp(0.0, 1.0) as f32 });
        };
        // a soft piece ends on its chord alone
        if ht >= 1 && !(matches!(genre, Genre::Ambient | Genre::Piano) && ht < 3) {
            let k = if matches!(genre, Genre::Cinematic | Genre::Piano) { Drum::Taiko } else { Drum::Kick };
            hit_note(&mut drums, k.gm_key(), 0.6 + 0.4 * e_hit, 1);
            // the cymbal and the boom are damped just before the end, so the track ends in silence
            if ht >= 2 && !matches!(genre, Genre::Piano | Genre::Lofi) {
                hit_note(&mut drums, Drum::Crash.gm_key(), 0.5 + 0.4 * e_hit, ring(super::instruments::CYMBAL_DAMP));
            }
            if ht >= 3 && matches!(genre, Genre::Cinematic | Genre::Electronic) {
                hit_note(&mut fx, Fx::Impact.key(), 0.4 + 0.4 * e_hit, ring(super::instruments::CYMBAL_DAMP));
            }
        }
    }

    // ---- bass
    if let Some(bp) = bass.as_mut() {
        let gate = if matches!(genre, Genre::Electronic | Genre::Pop) { 0.85 } else { 1.0 };
        for bar in 0..bars {
            let e = energy(bar);
            let ch = chord(bar);
            let root = bass_root(ch, key);
            let pat: Vec<char> = bass_pattern(genre, tier(e)).chars().collect();
            for (s, c_) in pat.iter().enumerate() {
                let k = match c_ {
                    'R' => root,
                    'F' => root + 7,
                    'O' => root + 12,
                    _ => continue,
                };
                let hold = pat.iter().skip(s + 1).take_while(|x| **x == '-').count() as u64;
                let accent = if s.is_multiple_of(4) { 1.0 } else { 0.85 };
                let v = (0.55 + 0.4 * e) * accent * (0.96 + 0.08 * rng.unit());
                bp.notes.extend(w.note(bar, s as u64, 1 + hold, k, v, gate));
            }
        }
        if hit < plan.samples {
            let k = bass_root(chord(bars), key);
            bp.notes.push(Note { start: hit, len: ring(0.15), key: midi(k), velocity: (0.6 + 0.3 * e_hit) as f32 });
        }
        if b.pump {
            bp.pump = (0..=bars).map(|bar| if tier(energy(bar)) >= 2 { 0.45 } else { 0.0 }).collect();
        }
    }

    // ---- pad: the chord held through each bar, common tones tied over the bar line
    if let Some(pp) = pad.as_mut() {
        let lo = if pp.patch == Patch::SuperSaw { 55 } else { 52 };
        let mut open: Vec<(i32, usize)> = Vec::new();
        for bar in 0..=bars {
            let e = energy(bar);
            let ch = chord(bar);
            let mut v = voicing(ch, key, lo);
            if pp.patch == Patch::Strings && tier(e) >= 2 {
                v.insert(0, lo - 12 + (ch.root_pc(key) - (lo - 12)).rem_euclid(12));
            }
            let start = w.at(bar, 0);
            let end = if bar == bars { hit.saturating_add(ring(1.1)).min(plan.samples) } else { w.at(bar + 1, 0) };
            let vel = (0.5 + 0.4 * e) as f32;
            let mut next_open = Vec::new();
            for n in v {
                // the final chord is struck again; inside the track a common tone is held over
                let tied = if bar < bars { open.iter().find(|(k, _)| *k == n).map(|(_, i)| *i) } else { None };
                if let Some(note) = tied.and_then(|i| pp.notes.get_mut(i)) {
                    note.len = end.saturating_sub(note.start).max(1);
                    next_open.extend(tied.map(|i| (n, i)));
                } else if start < plan.samples && end > start {
                    next_open.push((n, pp.notes.len()));
                    pp.notes.push(Note { start, len: end - start, key: midi(n), velocity: vel });
                }
            }
            open = next_open;
        }
        if b.pump {
            pp.pump = (0..=bars).map(|bar| if tier(energy(bar)) >= 2 { 0.55 } else { 0.0 }).collect();
        }
    }

    // ---- keys: comping chords, or broken chords for the piano
    if let Some(kp) = keys.as_mut() {
        for bar in 0..bars {
            let e = energy(bar);
            let ch = chord(bar);
            let t = tier(e);
            if genre == Genre::Piano {
                let r = bass_root(ch, key) + 12;
                let third = ch.third();
                let tones = [r, r + 7, r + 12, r + 12 + third, r + 19, r + 12 + third, r + 12, r + 7];
                if t == 0 {
                    for n in [r, r + 7, r + 12 + third] {
                        kp.notes.extend(w.note(bar, 0, 16, n, 0.45 + 0.3 * e, 1.0));
                    }
                } else {
                    let every: u64 = if t >= 4 { 1 } else { 2 };
                    for (i, s) in (0..STEPS_PER_BAR).step_by(every as usize).enumerate() {
                        let n = tones.get(i % tones.len()).copied().unwrap_or(r);
                        let v = (0.4 + 0.35 * e) * if s.is_multiple_of(4) { 1.0 } else { 0.8 };
                        // held like a sustain pedal: each note rings for half a bar
                        kp.notes.extend(w.note(bar, s, 8, n, v, 1.0));
                    }
                }
            } else {
                let pat: Vec<char> = comp_pattern(genre, t).chars().collect();
                let mut v = voicing(ch, key, 55);
                if t >= 2 {
                    v.insert(0, bass_root(ch, key) + 12);
                }
                for (s, c_) in pat.iter().enumerate() {
                    if *c_ != 'x' {
                        continue;
                    }
                    let hold = pat.iter().skip(s + 1).take_while(|x| **x == '-').count() as u64;
                    let vel = (0.45 + 0.35 * e) * if s.is_multiple_of(4) { 1.0 } else { 0.8 } * (0.95 + 0.1 * rng.unit());
                    for n in &v {
                        kp.notes.extend(w.note(bar, s as u64, 1 + hold, *n, vel, 0.9));
                    }
                }
            }
        }
        if hit < plan.samples {
            let ch = chord(bars);
            let mut v = voicing(ch, key, 55);
            v.insert(0, bass_root(ch, key) + 12);
            for n in v {
                kp.notes.push(Note { start: hit, len: ring(0.4), key: midi(n), velocity: (0.45 + 0.35 * e_hit) as f32 });
            }
        }
    }

    // ---- melody: motifs over the verses and choruses, long tones in the bridge
    let (lo, hi) = b.lead_range;
    let verse_motif = motif(&mut rng, true);
    let chorus_motif = motif(&mut rng, false);
    let mut prev = (lo + hi) / 2;
    let quiet_ok = matches!(genre, Genre::Ambient | Genre::Piano);
    let mut play = |m: &Motif, bar0: u32, avail: u32, range: (i32, i32), vel: f64, phrase_end: bool, lead: &mut Part, prev: &mut i32| {
        let count = m.notes.len();
        for (i, &(s, l, delta)) in m.notes.iter().enumerate() {
            if s >= u64::from(avail) * STEPS_PER_BAR {
                break;
            }
            let bar = bar0.saturating_add((s / STEPS_PER_BAR) as u32);
            let ch = chord(bar);
            let sc = scale(minor, ch);
            let mut n = if i == 0 { nearest_chord_tone(*prev, ch, key) } else { scale_step(*prev, delta, key, &sc) };
            let strong = s.is_multiple_of(8) || l >= 6;
            if strong {
                n = nearest_chord_tone(n, ch, key);
            }
            if phrase_end && i + 1 == count {
                // the phrase comes to rest on the root or the third
                let r = ch.root_pc(key);
                n = nearest_pc(n, [r, (r + ch.third()).rem_euclid(12)]);
            }
            n = fold(n, range.0, range.1);
            let v = vel * if strong { 1.0 } else { 0.85 } * (0.95 + 0.1 * rng.unit());
            let gate = if lead.patch == Patch::Pluck || lead.patch == Patch::Piano { 1.0 } else { 0.92 };
            if let Some(note) = w.note(bar0, s, l, n, v, gate) {
                lead.notes.push(note);
            }
            *prev = n;
        }
    };
    for blk in &arr.blocks {
        let e0 = energy(blk.start);
        if e0 < 0.12 && !quiet_ok {
            continue;
        }
        let vel = 0.5 + 0.45 * e0;
        match blk.kind {
            SectionKind::Verse | SectionKind::Chorus => {
                let (m, range) = if blk.kind == SectionKind::Chorus { (&chorus_motif, (lo + 5, hi)) } else { (&verse_motif, (lo, hi - 5)) };
                let mut wbar = 0;
                while wbar < blk.bars {
                    let avail = (blk.bars - wbar).min(2);
                    let phrase_end = (wbar / 2) % 2 == 1 || wbar + 2 >= blk.bars;
                    if wbar == 4 {
                        // the second phrase of a long section starts a step higher
                        prev = scale_step(prev, 2, key, &scale(minor, chord(blk.start + wbar)));
                    }
                    play(m, blk.start + wbar, avail, range, vel, phrase_end, &mut lead, &mut prev);
                    wbar += 2;
                }
            }
            SectionKind::Intro if blk.bars >= 4 => {
                play(&verse_motif, blk.start + blk.bars - 2, 2, (lo, hi - 5), vel * 0.6, true, &mut lead, &mut prev);
            }
            SectionKind::Outro => {
                play(&chorus_motif, blk.start, 1, (lo, hi), vel * 0.7, true, &mut lead, &mut prev);
            }
            SectionKind::Bridge if tier(e0) >= 1 || quiet_ok => {
                for i in 0..blk.bars {
                    let bar = blk.start + i;
                    let ch = chord(bar);
                    // the third or the fifth, whichever is nearer
                    let r = ch.root_pc(key);
                    let n = fold(nearest_pc(prev, [(r + ch.third()).rem_euclid(12), (r + 7).rem_euclid(12)]), lo, hi);
                    lead.notes.extend(w.note(bar, 0, 14, n, vel * 0.75, 0.95));
                    prev = n;
                }
            }
            _ => {}
        }
    }
    if hit < plan.samples && (e_hit >= 0.12 || quiet_ok) {
        // the tonic itself, nearest where the melody was
        let n = fold(nearest_pc(prev, [chord(bars).root_pc(key)]), lo, hi);
        lead.notes.push(Note { start: hit, len: ring(0.6), key: midi(n), velocity: (0.5 + 0.4 * e_hit) as f32 });
    }

    // ---- arpeggio / ostinato
    if let Some(ap) = arp.as_mut() {
        let order = ARPS.get(rng.below(ARPS.len())).copied().unwrap_or([0, 1, 2, 3, 2, 1, 2, 3]);
        for blk in &arr.blocks {
            for i in 0..blk.bars {
                let bar = blk.start + i;
                let e = energy(bar);
                let plays = match (genre, blk.kind) {
                    (Genre::Ambient, _) => true,
                    (_, SectionKind::Build) => e >= 0.3,
                    (_, SectionKind::Chorus) => e >= 0.6,
                    (Genre::Electronic, SectionKind::Verse) => e >= 0.5,
                    (Genre::Cinematic, SectionKind::Verse) => e >= 0.65,
                    _ => false,
                };
                if !plays {
                    continue;
                }
                let ch = chord(bar);
                let mut tones = voicing(ch, key, 60);
                if tones.len() < 4 {
                    tones.push(tones.first().copied().unwrap_or(60) + 12);
                }
                // a build climbs an octave per two bars
                let lift = if blk.kind == SectionKind::Build { (i / 2) as i32 * 12 } else { 0 };
                if genre == Genre::Ambient {
                    // bells: a few chord tones on random eighths
                    for s in (0..STEPS_PER_BAR).step_by(2) {
                        if rng.unit() < 0.15 + 0.4 * e {
                            let n = tones.get(rng.below(tones.len())).copied().unwrap_or(72) + 12;
                            ap.notes.extend(w.note(bar, s, 4, n, 0.3 + 0.3 * e, 1.0));
                        }
                    }
                    continue;
                }
                let every: u64 = if e >= 0.6 || blk.kind == SectionKind::Build { 1 } else { 2 };
                for (j, s) in (0..STEPS_PER_BAR).step_by(every as usize).enumerate() {
                    let idx = order.get(j % order.len()).copied().unwrap_or(0);
                    let n = tones.get(idx % tones.len()).copied().unwrap_or(60) + lift;
                    let swell = if blk.kind == SectionKind::Build { 0.7 + 0.3 * (s as f64 / STEPS_PER_BAR as f64) } else { 1.0 };
                    let v = (0.35 + 0.4 * e) * swell * if s.is_multiple_of(4) { 1.0 } else { 0.8 };
                    ap.notes.extend(w.note(bar, s, every, fold(n, 55, 88), v, 0.9));
                }
            }
        }
    }

    let mut parts = vec![drums, fx, lead];
    parts.extend(bass);
    parts.extend(arp);
    parts.extend(pad);
    parts.extend(keys);
    parts.retain(|p| !p.notes.is_empty());
    for p in &mut parts {
        p.notes.sort_by_key(|n| n.start);
        if !matches!(p.patch, Patch::Kit(_) | Patch::Fx) {
            untangle(&mut p.notes);
        }
    }
    Score { parts }
}

/// A note struck again before it ends is cut where it is struck again (and one struck twice at
/// once is struck once), so every note-off releases its own note (`notes` sorted by start).
fn untangle(notes: &mut Vec<Note>) {
    let mut next: [Option<u64>; 128] = [None; 128];
    for n in notes.iter_mut().rev() {
        if let Some(slot) = next.get_mut(usize::from(n.key)) {
            if let Some(again) = *slot
                && n.start.saturating_add(n.len) > again
            {
                n.len = again.saturating_sub(n.start);
            }
            *slot = Some(n.start);
        }
    }
    notes.retain(|n| n.len > 0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::music::arrange::Energy;

    #[test]
    fn harmony_ends_on_a_cadence() {
        let arr = crate::music::arrange::arrange(20, &Energy::Level(0.7), 1);
        for mood in Mood::ALL {
            let h = harmony(mood, Genre::Pop, &arr);
            assert_eq!(h.len(), 21);
            assert_eq!(h[19].root, 7, "{mood:?}: the dominant before the hit");
            assert_eq!(h[20].root, 0, "{mood:?}: the tonic on the hit");
            assert_eq!(matches!(h[20].quality, Quality::Minor), mood.minor(), "{mood:?}");
        }
        let none = crate::music::arrange::arrange(0, &Energy::Level(0.7), 1);
        assert_eq!(harmony(Mood::Epic, Genre::Cinematic, &none).len(), 1);
    }

    #[test]
    fn chords_and_scales_are_named_and_walked() {
        assert_eq!(c(0, Mi).name(0), "Am");
        assert_eq!(c(8, Ma).name(0), "F");
        assert_eq!(c(7, D7).name(3), "G7", "C major's V7");
        assert_eq!(c(0, M7).name(3), "Cmaj7");
        let maj = scale(false, c(0, Ma));
        // C major (key 3): E up two steps is G, down one is D
        assert_eq!(scale_step(64, 2, 3, &maj), 67);
        assert_eq!(scale_step(64, -1, 3, &maj), 62);
        assert_eq!(scale_step(61, 1, 3, &maj), 62, "off the scale: from the tone below");
        assert_eq!(nearest_chord_tone(65, c(0, Ma), 3), 64);
        assert_eq!(nearest_chord_tone(66, c(0, Ma), 3), 67);
        assert_eq!(fold(100, 60, 79), 76);
        assert_eq!(fold(20, 60, 79), 68);
        assert_eq!(voicing(c(0, Ma), 3, 52), vec![52, 55, 60]);
        assert_eq!(bass_root(c(0, Mi), 0), 33, "A1");
        // hostile steps stay in range
        assert!((0..=127).contains(&i32::from(midi(scale_step(i32::MAX / 2, i32::MAX, 0, &maj)))));
    }
}
