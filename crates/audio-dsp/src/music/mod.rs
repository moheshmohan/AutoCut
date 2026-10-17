//! The procedural soundtrack engine: music of an **exact length** that ends on a bar.
//!
//! A [`Request`] (mood, optional genre, length in [`Tick`]s, seed, tempo, key, energy) becomes a
//! [`Plan`] and then a [`Track`]:
//!
//! 1. **Taxonomy** ([`taxonomy`]): the mood and genre pick a [`Style`] — tempo range, energy,
//!    use cases — the vocabulary of a music library.
//! 2. **Arrangement** ([`arrange`]): the length fixes how many whole bars fit before the final
//!    hit, leaving a ring-out of at least two beats (1.2–3 s); the arranger fills exactly those
//!    bars with section blocks (intro, verse, build, chorus, bridge, outro) following the energy
//!    curve. Nothing is padded, looped or faded to fit: the final chord lands on the downbeat of
//!    the last bar and rings until the last sample.
//! 3. **Composition** ([`compose`]): a chord per bar (each section its progression, a dominant
//!    before the hit), and per-instrument patterns over chord and energy — drums, bass, a seeded
//!    melody (motifs over verses and choruses), arpeggios, pads and keys.
//! 4. **Rendering** ([`render`]): on the instruments of an [`InstrumentBank`] (built in: the
//!    oscillator voices), into four stems — drums, bass, melody, pad — that sum to the mix, at
//!    [`MUSIC_LUFS`](crate::synth::MUSIC_LUFS).
//!
//! Every beat and bar is an exact tick ([`tempo::Tempo`]); [`Plan::beats`] is the beat map an
//! editor cuts on. The same request always gives the same audio.

pub mod arrange;
pub mod compose;
pub mod instruments;
mod render;
pub mod taxonomy;
pub mod tempo;

use filmcraft_time::{TICKS_PER_SECOND, Tick};

pub use arrange::{Energy, SectionKind};
pub use compose::{Chord, Stem};
pub use instruments::{Instrument, InstrumentBank, OscillatorBank, Patch};
pub use taxonomy::{Genre, Mood, Style};
pub use tempo::Tempo;

use crate::synth::{MAX_MUSIC_SECONDS, Stereo, SynthError};

/// Shortest track, seconds.
pub const MIN_SECONDS: f64 = 0.05;
/// Longest track whose stems are rendered too, seconds (five stereo float buffers of it are held).
pub const MAX_STEMS_SECONDS: f64 = 300.0;
/// Quietest stem level, dB (anything lower is muted).
pub const MIN_LEVEL_DB: f64 = -60.0;
/// Loudest stem level, dB.
pub const MAX_LEVEL_DB: f64 = 12.0;

/// What to make.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub mood: Mood,
    /// `None`: the mood's usual genre.
    pub genre: Option<Genre>,
    /// The exact length.
    pub length: Tick,
    pub seed: u64,
    /// `None`: the style's usual tempo.
    pub bpm: Option<f64>,
    /// Tonic as semitones above A; `None`: A.
    pub key: Option<i32>,
    /// `None`: the style's energy, as an arc.
    pub energy: Option<Energy>,
    /// Render the four stems as well as the mix.
    pub stems: bool,
    /// Level of each stem in dB ([`Stem::index`] order); `None` mutes it.
    pub levels: [Option<f64>; 4],
}

impl Request {
    pub fn new(mood: Mood, length: Tick) -> Self {
        Request { mood, genre: None, length, seed: 1, bpm: None, key: None, energy: None, stems: false, levels: [Some(0.0); 4] }
    }
}

/// A section of a planned track.
#[derive(Clone, Debug, PartialEq)]
pub struct Section {
    pub kind: SectionKind,
    /// Where it starts, from the start of the track.
    pub start: Tick,
    pub bars: u32,
    /// Its average energy, 0..1.
    pub energy: f64,
}

/// A track before it is rendered: everything about it but the audio (cheap to make).
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub sample_rate: u32,
    /// Exactly `round(length · sample_rate)` samples.
    pub samples: u64,
    pub length: Tick,
    pub tempo: Tempo,
    /// Tonic as semitones above A.
    pub key: i32,
    pub mood: Mood,
    pub genre: Genre,
    pub style: Style,
    pub energy: Energy,
    pub seed: u64,
    pub arrangement: arrange::Arrangement,
    /// The chord of each bar, and the final hit's (`bars + 1`).
    pub chords: Vec<Chord>,
    /// The final hit: the downbeat of bar [`arrange::Arrangement::bars`].
    pub ending: Tick,
    /// From the final hit to the end (the ring-out).
    pub tail: Tick,
}

impl Plan {
    /// Every beat from the start to the final hit, inclusive (the ring-out has none).
    pub fn beats(&self) -> Vec<Tick> {
        let n = u64::from(self.arrangement.bars) * tempo::BEATS_PER_BAR;
        (0..=n).map(|k| self.tempo.beat(k)).collect()
    }

    /// Every downbeat from the start to the final hit, inclusive.
    pub fn bars(&self) -> Vec<Tick> {
        (0..=u64::from(self.arrangement.bars)).map(|b| self.tempo.bar(b)).collect()
    }

    /// The sections, in order.
    pub fn sections(&self) -> Vec<Section> {
        self.arrangement
            .blocks
            .iter()
            .map(|b| {
                let e = &self.arrangement.energy;
                let from = b.start as usize;
                let to = from.saturating_add(b.bars as usize).min(e.len());
                let slice = e.get(from..to).unwrap_or(&[]);
                let energy = if slice.is_empty() { 0.0 } else { slice.iter().sum::<f64>() / slice.len() as f64 };
                Section { kind: b.kind, start: self.tempo.bar(u64::from(b.start)), bars: b.bars, energy }
            })
            .collect()
    }

    /// Whether the tonic is minor.
    pub fn minor(&self) -> bool {
        self.mood.minor()
    }

    /// The key's name ("A minor", "C major").
    pub fn key_name(&self) -> String {
        format!("{} {}", compose::pc_name(compose::tonic_pc(self.key)), if self.minor() { "minor" } else { "major" })
    }
}

/// A rendered track.
pub struct Track {
    pub plan: Plan,
    pub mix: Stereo,
    /// The four stems (drums, bass, melody, pad) when asked for; they sum to `mix`.
    pub stems: Vec<(Stem, Stereo)>,
    /// What played it ([`InstrumentBank::name`]).
    pub instruments: String,
}

/// The least ring-out after the final hit: two beats, within 1.2–3 s.
fn tail_min(tempo: Tempo) -> Tick {
    tempo.beat(2).max(Tick(TICKS_PER_SECOND * 6 / 5)).min(Tick(TICKS_PER_SECOND * 3))
}

/// The most whole bars before a final hit that leaves at least [`tail_min`] of `length`.
fn bars_before_hit(tempo: Tempo, length: Tick) -> u32 {
    let room = length.0.saturating_sub(tail_min(tempo).0);
    if room <= 0 {
        return 0;
    }
    let bar = tempo.bar(1).0.max(1);
    let mut b = u64::try_from(room / bar).unwrap_or(0).min(u64::from(arrange::MAX_BARS));
    // the estimate is within a bar of the truth; exact tick arithmetic settles it
    while b > 0 && tempo.bar(b).0 > room {
        b -= 1;
    }
    while b < u64::from(arrange::MAX_BARS) && tempo.bar(b + 1).0 <= room {
        b += 1;
    }
    u32::try_from(b).unwrap_or(arrange::MAX_BARS)
}

/// Plan `req` at `sample_rate` (8–192 kHz). Lengths outside [`MIN_SECONDS`]..[`MAX_MUSIC_SECONDS`]
/// and stems of tracks over [`MAX_STEMS_SECONDS`] are refused.
pub fn plan(req: &Request, sample_rate: u32) -> Result<Plan, SynthError> {
    if !(8_000..=192_000).contains(&sample_rate) {
        return Err(SynthError(format!("sample rate {sample_rate} Hz is outside 8–192 kHz")));
    }
    let seconds = req.length.seconds();
    if !seconds.is_finite() || seconds < MIN_SECONDS {
        return Err(SynthError(format!("music must be at least {MIN_SECONDS} s long")));
    }
    if seconds > MAX_MUSIC_SECONDS + 1e-9 {
        return Err(SynthError(format!("music can be at most {} minutes long", MAX_MUSIC_SECONDS / 60.0)));
    }
    if req.stems && seconds > MAX_STEMS_SECONDS + 1e-9 {
        return Err(SynthError(format!("stems are rendered for tracks up to {} minutes", MAX_STEMS_SECONDS / 60.0)));
    }
    let style = Style::of(req.mood, req.genre);
    let tempo = Tempo::from_bpm(req.bpm.unwrap_or(style.bpm.1));
    let key = req.key.unwrap_or(0).rem_euclid(12);
    let energy = match &req.energy {
        Some(Energy::Curve(p)) => Energy::curve(p).map_err(SynthError)?,
        Some(Energy::Level(l)) => Energy::Level(if l.is_finite() { l.clamp(0.0, 1.0) } else { style.energy }),
        None => Energy::Level(style.energy),
    };
    let samples = tempo::sample_of(req.length, sample_rate);
    if samples == 0 {
        return Err(SynthError("the track is shorter than a sample".into()));
    }
    let bars = bars_before_hit(tempo, req.length);
    let arrangement = arrange::arrange(bars, &energy, req.seed);
    let chords = compose::harmony(req.mood, style.genre, &arrangement);
    let ending = tempo.bar(u64::from(arrangement.bars));
    Ok(Plan {
        sample_rate,
        samples,
        length: req.length,
        tempo,
        key,
        mood: req.mood,
        genre: style.genre,
        style,
        energy,
        seed: req.seed,
        arrangement,
        chords,
        ending,
        tail: Tick(req.length.0.saturating_sub(ending.0)),
    })
}

/// Linear stem gains from dB levels (`None` or below [`MIN_LEVEL_DB`] mutes).
fn gains(levels: &[Option<f64>; 4]) -> Result<[f32; 4], SynthError> {
    let mut g = [0.0f32; 4];
    for (out, (l, s)) in g.iter_mut().zip(levels.iter().zip(Stem::ALL)) {
        *out = match l {
            None => 0.0,
            Some(db) if !db.is_finite() || *db > MAX_LEVEL_DB => {
                return Err(SynthError(format!("the {} level must be a number up to {MAX_LEVEL_DB} dB", s.id())));
            }
            Some(db) if *db < MIN_LEVEL_DB => 0.0,
            Some(db) => 10f64.powf(db / 20.0) as f32,
        };
    }
    if g.iter().all(|x| *x == 0.0) {
        return Err(SynthError("every stem is muted".into()));
    }
    Ok(g)
}

/// Make the track `req` asks for, on the built-in instruments.
pub fn generate(req: &Request, sample_rate: u32) -> Result<Track, SynthError> {
    generate_with(req, sample_rate, &OscillatorBank)
}

/// Make the track `req` asks for, on `bank`'s instruments.
pub fn generate_with(req: &Request, sample_rate: u32, bank: &dyn InstrumentBank) -> Result<Track, SynthError> {
    let g = gains(&req.levels)?;
    let plan = plan(req, sample_rate)?;
    let score = compose::compose(&plan);
    let out = render::render(&plan, &score, bank, &render::Options { stems: req.stems, gains: g })?;
    Ok(Track { plan, mix: out.mix, stems: out.stems, instruments: bank.name().to_string() })
}

#[cfg(test)]
mod tests;
