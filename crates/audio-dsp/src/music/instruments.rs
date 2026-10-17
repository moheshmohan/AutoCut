//! The instruments a score is played on.
//!
//! A score names each part's [`Patch`]; an [`InstrumentBank`] turns a patch into an
//! [`Instrument`] that takes note-on / note-off events and renders blocks of stereo audio
//! (dry, plus a reverb send). Every patch maps to a General MIDI program and every drum to a
//! General MIDI percussion key, so a SoundFont bank — for example one built on `rustysynth`
//! (MIT, pure Rust) loading a user-supplied `.sf2` — can play the same score through the same
//! two traits. The built-in [`OscillatorBank`] synthesizes everything from band-limited
//! oscillators, FM, additive partials, noise and filters: no samples, deterministic.

use std::f64::consts::{PI, TAU};

use crate::biquad::{Biquad, Coeffs, FilterType};
use crate::synth::SynthError;

use super::taxonomy::Genre;
use super::tempo::Tempo;

/// A drum kit's character.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kit {
    /// Taiko, toms, big snare, long room.
    Cinematic,
    /// Punchy kick, claps, crisp hats.
    Electronic,
    /// A rounded acoustic-style kit.
    Pop,
    /// Soft, dark and dusty.
    Lofi,
    /// Quiet kick, shaker and rim (ambient, piano).
    Soft,
}

/// What a part is played on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Patch {
    /// Drums (General MIDI channel 10; notes are [`Drum`] keys).
    Kit(Kit),
    /// Risers and impacts ([`Fx`] keys); always synthesized.
    Fx,
    /// A slow-attack string ensemble.
    Strings,
    /// A soft, breathing synth pad.
    WarmPad,
    /// Detuned saw chords.
    SuperSaw,
    /// An acoustic-style piano.
    Piano,
    /// An FM electric piano.
    EPiano,
    /// A short plucked synth.
    Pluck,
    /// A brass-like lead that opens up as it holds.
    Brass,
    /// A bell / vibraphone.
    Bell,
    /// A soft, breathy lead.
    Flute,
    /// A bright saw lead.
    SawLead,
    /// A filtered saw-and-sine bass.
    SynthBass,
    /// A pure, heavy sub bass.
    SubBass,
    /// A round plucked bass.
    RoundBass,
}

impl Patch {
    /// The General MIDI program (0-based) a SoundFont bank plays this patch with; `None` for
    /// drums (channel 10) and effects.
    pub fn gm_program(self) -> Option<u8> {
        Some(match self {
            Patch::Kit(_) | Patch::Fx => return None,
            Patch::Strings => 48,
            Patch::WarmPad => 89,
            Patch::SuperSaw => 90,
            Patch::Piano => 0,
            Patch::EPiano => 4,
            Patch::Pluck => 45,
            Patch::Brass => 61,
            Patch::Bell => 11,
            Patch::Flute => 73,
            Patch::SawLead => 81,
            Patch::SynthBass => 38,
            Patch::SubBass => 39,
            Patch::RoundBass => 33,
        })
    }

    pub fn id(self) -> &'static str {
        match self {
            Patch::Kit(Kit::Cinematic) => "kit-cinematic",
            Patch::Kit(Kit::Electronic) => "kit-electronic",
            Patch::Kit(Kit::Pop) => "kit-pop",
            Patch::Kit(Kit::Lofi) => "kit-lofi",
            Patch::Kit(Kit::Soft) => "kit-soft",
            Patch::Fx => "fx",
            Patch::Strings => "strings",
            Patch::WarmPad => "warm-pad",
            Patch::SuperSaw => "supersaw",
            Patch::Piano => "piano",
            Patch::EPiano => "electric-piano",
            Patch::Pluck => "pluck",
            Patch::Brass => "brass",
            Patch::Bell => "bell",
            Patch::Flute => "flute",
            Patch::SawLead => "saw-lead",
            Patch::SynthBass => "synth-bass",
            Patch::SubBass => "sub-bass",
            Patch::RoundBass => "round-bass",
        }
    }
}

/// Drum sounds, numbered as General MIDI percussion keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Drum {
    Kick = 36,
    Rim = 37,
    Snare = 38,
    Clap = 39,
    /// A big low drum (GM low floor tom).
    Taiko = 41,
    HatClosed = 42,
    TomLow = 45,
    HatOpen = 46,
    TomMid = 47,
    Crash = 49,
    TomHigh = 50,
    Ride = 51,
    /// GM maracas.
    Shaker = 70,
}

impl Drum {
    pub const ALL: [Drum; 13] = [
        Drum::Kick,
        Drum::Rim,
        Drum::Snare,
        Drum::Clap,
        Drum::Taiko,
        Drum::HatClosed,
        Drum::TomLow,
        Drum::HatOpen,
        Drum::TomMid,
        Drum::Crash,
        Drum::TomHigh,
        Drum::Ride,
        Drum::Shaker,
    ];

    /// The General MIDI percussion key.
    pub fn gm_key(self) -> u8 {
        self as u8
    }

    pub fn from_key(key: u8) -> Option<Drum> {
        Drum::ALL.into_iter().find(|d| d.gm_key() == key)
    }
}

/// Effect sounds played by [`Patch::Fx`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fx {
    /// Noise and pitch rising until the note ends.
    Riser = 1,
    /// A sub boom with a crack.
    Impact = 2,
}

impl Fx {
    pub fn key(self) -> u8 {
        self as u8
    }
}

/// Where an instrument is played.
#[derive(Clone, Copy, Debug)]
pub struct Context {
    pub sample_rate: u32,
    pub tempo: Tempo,
    pub genre: Genre,
    pub seed: u64,
}

/// A block of stereo output: the dry signal and the reverb send, all the same length.
pub struct Bus<'a> {
    pub dry_l: &'a mut [f32],
    pub dry_r: &'a mut [f32],
    pub send_l: &'a mut [f32],
    pub send_r: &'a mut [f32],
}

impl Bus<'_> {
    pub fn len(&self) -> usize {
        self.dry_l.len().min(self.dry_r.len()).min(self.send_l.len()).min(self.send_r.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Something that plays notes. Events take effect at the start of the next [`render`]
/// (the renderer splits blocks at event times, so they are sample-accurate).
///
/// [`render`]: Instrument::render
pub trait Instrument: Send {
    /// Start `key` (MIDI note, or a [`Drum`] / [`Fx`] key) at `velocity` (0..1).
    fn note_on(&mut self, key: u8, velocity: f32);
    /// Release `key` (one-shot drums ignore it).
    fn note_off(&mut self, key: u8);
    /// Add the next `bus.len()` samples into `bus`.
    fn render(&mut self, bus: &mut Bus<'_>);
    /// About how long a released note keeps sounding, seconds (the score releases the final
    /// chord this long before the end).
    fn release_seconds(&self) -> f64;
}

/// Makes instruments for patches.
pub trait InstrumentBank {
    /// Shown in a track's metadata ("oscillators", or a SoundFont's name).
    fn name(&self) -> &str;
    fn instrument(&self, patch: Patch, ctx: &Context) -> Result<Box<dyn Instrument>, SynthError>;
}

/// The built-in bank: every patch synthesized.
#[derive(Clone, Copy, Debug, Default)]
pub struct OscillatorBank;

impl InstrumentBank for OscillatorBank {
    fn name(&self) -> &str {
        "oscillators"
    }

    fn instrument(&self, patch: Patch, ctx: &Context) -> Result<Box<dyn Instrument>, SynthError> {
        if !(8_000..=192_000).contains(&ctx.sample_rate) {
            return Err(SynthError(format!("sample rate {} Hz is outside 8–192 kHz", ctx.sample_rate)));
        }
        let sr = f64::from(ctx.sample_rate);
        Ok(match patch {
            Patch::Kit(kit) => Box::new(DrumKit { kit, sr, voices: Vec::new(), seed: ctx.seed, count: 0 }),
            Patch::Fx => Box::new(FxRack { sr, rise: (ctx.tempo.bar_seconds() * 2.0).clamp(0.5, 12.0), voices: Vec::new(), seed: ctx.seed, count: 0 }),
            p => Box::new(Tonal { tone: tone(p, ctx.genre), sr, voices: Vec::new() }),
        })
    }
}

// ---------------------------------------------------------------------------------------------
// building blocks

/// xorshift64* — small, fast, deterministic.
#[derive(Clone, Debug)]
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [-1, 1).
    pub(crate) fn white(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
    /// Uniform in [0, 1).
    pub(crate) fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in 0..n (0 when n is 0).
    pub(crate) fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next_u64() % n as u64) as usize }
    }
}

fn wrap(phase: f64) -> f64 {
    phase - phase.floor()
}

fn poly_blep(t: f64, dt: f64) -> f64 {
    if dt <= 0.0 {
        0.0
    } else if t < dt {
        let x = t / dt;
        x + x - x * x - 1.0
    } else if t > 1.0 - dt {
        let x = (t - 1.0) / dt;
        x * x + x + x + 1.0
    } else {
        0.0
    }
}

/// Band-limited sawtooth (PolyBLEP).
fn saw(phase: f64, dt: f64) -> f64 {
    2.0 * phase - 1.0 - poly_blep(phase, dt)
}

fn triangle(phase: f64) -> f64 {
    1.0 - 4.0 * (phase - 0.5).abs()
}

/// Equal-power pan gains for `pan` in -1 (left) .. 1 (right).
fn pan_gains(pan: f64) -> (f64, f64) {
    let a = (pan.clamp(-1.0, 1.0) + 1.0) * PI / 4.0;
    (a.cos(), a.sin())
}

/// MIDI note → Hz (A4 = 69 = 440 Hz).
pub(crate) fn hz(note: f64) -> f64 {
    440.0 * 2f64.powf((note - 69.0) / 12.0)
}

/// Most voices an instrument holds (a new note past it takes the oldest voice's place).
const MAX_VOICES: usize = 48;

// ---------------------------------------------------------------------------------------------
// pitched instruments

#[derive(Clone, Copy, Debug)]
enum Osc {
    Saw,
    Triangle,
    /// Sine with this much saw mixed in.
    SineSaw(f64),
    /// Two-operator FM: modulator at `ratio` × the pitch, its `index` decaying over `decay` s.
    Fm {
        ratio: f64,
        index: f64,
        decay: f64,
    },
    /// Harmonics 1..=`partials` at 1/h, the higher ones dying faster (`decay` s for the second).
    Additive {
        partials: usize,
        decay: f64,
    },
}

#[derive(Clone, Copy, Debug)]
struct Tone {
    osc: Osc,
    /// Oscillators per note (1..=7), spread over `detune` cents and `width` of the stereo field.
    unison: usize,
    detune: f64,
    width: f64,
    attack: f64,
    decay: f64,
    sustain: f64,
    release: f64,
    /// Percussive: after the attack the level dies with this time constant (seconds at A4,
    /// shorter up high); `decay`/`sustain` are then unused.
    ring: Option<f64>,
    /// Low-pass cutoff at full velocity for A4, Hz.
    cutoff: f64,
    /// 0..1: how much the cutoff follows the pitch.
    key_track: f64,
    /// Filter envelope depth, octaves, dying over `fdecay` seconds (negative: opening up).
    fenv: f64,
    fdecay: f64,
    q: f64,
    /// Vibrato rate (Hz), depth (cents), delay (s).
    vibrato: (f64, f64, f64),
    gain: f64,
    pan: f64,
    send: f64,
}

const MAX_UNISON: usize = 7;

fn base_tone() -> Tone {
    Tone {
        osc: Osc::Saw,
        unison: 1,
        detune: 0.0,
        width: 0.0,
        attack: 0.005,
        decay: 0.2,
        sustain: 1.0,
        release: 0.3,
        ring: None,
        cutoff: 4000.0,
        key_track: 0.5,
        fenv: 0.0,
        fdecay: 0.3,
        q: 0.7,
        vibrato: (0.0, 0.0, 0.0),
        gain: 0.3,
        pan: 0.0,
        send: 0.2,
    }
}

/// The sound of a pitched patch (some vary with the genre).
fn tone(p: Patch, genre: Genre) -> Tone {
    let b = base_tone();
    match p {
        Patch::Strings => Tone {
            unison: 3,
            detune: 14.0,
            width: 0.7,
            attack: if genre == Genre::Cinematic { 0.35 } else { 0.5 },
            decay: 0.5,
            sustain: 0.9,
            release: 0.7,
            cutoff: 2600.0,
            key_track: 0.6,
            vibrato: (5.2, 6.0, 0.4),
            gain: 0.164,
            send: 0.45,
            ..b
        },
        Patch::WarmPad => Tone {
            osc: Osc::SineSaw(0.45),
            unison: 3,
            detune: 10.0,
            width: 0.8,
            attack: 0.9,
            decay: 1.0,
            sustain: 0.85,
            release: 1.4,
            cutoff: 1500.0,
            key_track: 0.4,
            vibrato: (0.3, 4.0, 0.0),
            gain: 0.43,
            send: 0.55,
            ..b
        },
        Patch::SuperSaw => Tone {
            unison: 7,
            detune: 24.0,
            width: 0.9,
            attack: 0.03,
            decay: 0.4,
            sustain: 0.8,
            release: 0.35,
            cutoff: 3200.0,
            key_track: 0.4,
            gain: 0.19,
            send: 0.3,
            ..b
        },
        Patch::Piano => Tone {
            osc: Osc::Additive { partials: 8, decay: 0.9 },
            attack: 0.002,
            release: 0.25,
            ring: Some(2.2),
            cutoff: 5000.0,
            key_track: 0.3,
            fenv: 1.2,
            fdecay: 0.25,
            gain: 0.18,
            send: 0.3,
            ..b
        },
        Patch::EPiano => Tone {
            osc: Osc::Fm { ratio: 1.0, index: 1.6, decay: 0.35 },
            unison: 2,
            detune: 4.0,
            width: 0.4,
            attack: 0.003,
            release: 0.3,
            ring: Some(1.6),
            cutoff: if genre == Genre::Lofi { 2400.0 } else { 4500.0 },
            key_track: 0.3,
            vibrato: (4.5, 5.0, 0.0),
            gain: 0.2,
            send: 0.25,
            ..b
        },
        Patch::Pluck => Tone {
            osc: if genre == Genre::Cinematic { Osc::Triangle } else { Osc::Saw },
            unison: 2,
            detune: 8.0,
            width: 0.5,
            attack: 0.002,
            release: 0.15,
            ring: Some(0.28),
            cutoff: 900.0,
            key_track: 0.5,
            fenv: 2.4,
            fdecay: 0.09,
            q: 1.1,
            gain: 0.4,
            pan: 0.15,
            send: 0.35,
            ..b
        },
        Patch::Brass => Tone {
            unison: 2,
            detune: 6.0,
            width: 0.3,
            attack: 0.08,
            decay: 0.3,
            sustain: 0.85,
            release: 0.3,
            cutoff: 2600.0,
            key_track: 0.6,
            fenv: -1.6,
            fdecay: 0.25,
            vibrato: (5.0, 9.0, 0.35),
            gain: 0.53,
            send: 0.4,
            ..b
        },
        Patch::Bell => Tone {
            osc: Osc::Fm { ratio: 3.5, index: 2.2, decay: 0.6 },
            attack: 0.002,
            release: 0.6,
            ring: Some(2.4),
            cutoff: 9000.0,
            key_track: 0.0,
            gain: 0.17,
            pan: -0.2,
            send: 0.6,
            ..b
        },
        Patch::Flute => Tone {
            osc: Osc::SineSaw(0.12),
            attack: 0.06,
            decay: 0.2,
            sustain: 0.9,
            release: 0.18,
            cutoff: 3500.0,
            key_track: 0.5,
            vibrato: (5.4, 12.0, 0.25),
            gain: 0.35,
            send: 0.45,
            ..b
        },
        Patch::SawLead => Tone {
            unison: 3,
            detune: 12.0,
            width: 0.4,
            attack: 0.01,
            decay: 0.3,
            sustain: 0.75,
            release: 0.2,
            cutoff: 3000.0,
            key_track: 0.5,
            fenv: 1.4,
            fdecay: 0.2,
            vibrato: (5.5, 7.0, 0.3),
            gain: 0.4,
            send: 0.35,
            ..b
        },
        Patch::SynthBass => Tone {
            osc: Osc::SineSaw(0.4),
            attack: 0.004,
            decay: 0.25,
            sustain: 0.8,
            release: 0.08,
            cutoff: 420.0,
            key_track: 0.4,
            fenv: 1.4,
            fdecay: 0.12,
            q: 0.9,
            gain: 0.93,
            send: 0.02,
            ..b
        },
        Patch::SubBass => {
            Tone { osc: Osc::SineSaw(0.08), attack: 0.006, decay: 0.3, sustain: 0.95, release: 0.1, cutoff: 300.0, key_track: 0.2, gain: 0.36, send: 0.0, ..b }
        }
        Patch::RoundBass => Tone {
            osc: Osc::Triangle,
            attack: 0.004,
            release: 0.1,
            ring: Some(0.9),
            cutoff: 700.0,
            key_track: 0.3,
            fenv: 1.0,
            fdecay: 0.08,
            gain: 0.47,
            send: 0.02,
            ..b
        },
        Patch::Kit(_) | Patch::Fx => b,
    }
}

/// A level falling by a constant factor each sample (an exponential decay without an `exp` per
/// sample).
#[derive(Clone, Copy, Debug)]
struct Decay {
    v: f64,
    m: f64,
}

impl Decay {
    /// From `v`, falling by e every `tau` seconds.
    fn new(v: f64, tau: f64, sr: f64) -> Self {
        Decay { v, m: (-1.0 / (tau.max(1e-5) * sr.max(1.0))).exp() }
    }

    /// Holding at `v`.
    fn hold(v: f64) -> Self {
        Decay { v, m: 1.0 }
    }

    fn next(&mut self) -> f64 {
        let v = self.v;
        self.v *= self.m;
        v
    }
}

/// Most partials of an additive tone.
const MAX_PARTIALS: usize = 12;

#[derive(Clone, Debug)]
struct Voice {
    key: u8,
    freq: f64,
    vel: f64,
    /// Samples since note-on.
    t: u64,
    /// Released: samples since note-on when it was, and the falling level.
    rel: Option<(u64, Decay)>,
    /// The level now.
    level: f64,
    /// The ring of a struck note.
    ring: Decay,
    phases: [f64; MAX_UNISON],
    /// Each oscillator's pitch ratio and stereo gains.
    ratio: [f64; MAX_UNISON],
    gains: [(f64, f64); MAX_UNISON],
    /// FM modulator phase and index.
    mphase: f64,
    index: Decay,
    /// Additive partials' levels.
    partials: [Decay; MAX_PARTIALS],
    /// The filter envelope (octaves), and whether the cutoff has stopped moving.
    fenv: Decay,
    settled: bool,
    lp: [Biquad; 2],
    coeffs: Coeffs,
    vib_phase: f64,
    /// The vibrato's pitch factor now (updated every [`VIBRATO_EVERY`] samples).
    fmul: f64,
    done: bool,
}

/// Samples between vibrato updates.
const VIBRATO_EVERY: u64 = 16;

/// A pitched, polyphonic instrument.
struct Tonal {
    tone: Tone,
    sr: f64,
    voices: Vec<Voice>,
}

/// Samples between filter-coefficient updates while the cutoff moves.
const FILTER_EVERY: u64 = 32;

impl Tonal {
    fn start(&self, key: u8, vel: f64) -> Voice {
        let tone = &self.tone;
        let sr = self.sr;
        let freq = hz(f64::from(key));
        let n = tone.unison.clamp(1, MAX_UNISON);
        let norm = 1.0 / (n as f64).sqrt();
        let spread = |u: usize| if n == 1 { 0.0 } else { u as f64 / (n - 1) as f64 * 2.0 - 1.0 };
        let ratio = std::array::from_fn(|u| 2f64.powf(spread(u) * tone.detune * 0.5 / 1200.0));
        let gains = std::array::from_fn(|u| {
            let (l, r) = pan_gains(spread(u) * tone.width);
            (l * norm, r * norm)
        });
        let ring = match tone.ring {
            Some(ring) => Decay::new(1.0, (ring * (440.0 / freq.max(20.0)).powf(0.3)).max(0.02), sr),
            None => Decay::hold(1.0),
        };
        let index = match tone.osc {
            Osc::Fm { index, decay, .. } => Decay::new(index * (0.4 + 0.6 * vel), decay.max(0.01), sr),
            _ => Decay::hold(0.0),
        };
        let partials = std::array::from_fn(|i| match tone.osc {
            Osc::Additive { decay, .. } if i > 0 => Decay::new(1.0 / (i + 1) as f64, decay.max(0.01) / i as f64, sr),
            _ => Decay::hold(1.0 / (i + 1) as f64),
        });
        Voice {
            key,
            freq,
            vel,
            t: 0,
            rel: None,
            level: 0.0,
            ring,
            phases: std::array::from_fn(|u| (u as f64 * 0.137 + f64::from(key) * 0.071).fract()),
            ratio,
            gains,
            mphase: 0.0,
            index,
            partials,
            fenv: Decay::new(tone.fenv, tone.fdecay.max(0.005), sr),
            settled: false,
            lp: [Biquad::default(), Biquad::default()],
            coeffs: Coeffs::IDENTITY,
            vib_phase: 0.0,
            fmul: 1.0,
            done: false,
        }
    }

    fn voice_sample(tone: &Tone, sr: f64, v: &mut Voice) -> (f64, f64) {
        let ts = v.t as f64 / sr;
        let attack = tone.attack.max(1e-4);
        // amplitude envelope
        let env = match v.rel.as_mut() {
            Some((t0, d)) => {
                if v.t.saturating_sub(*t0) as f64 / sr > tone.release.max(0.005) * 1.5 {
                    v.done = true;
                }
                d.next()
            }
            None if ts < attack => ts / attack,
            None if tone.ring.is_some() => {
                let e = v.ring.next();
                if e < 1e-5 {
                    v.done = true;
                }
                e
            }
            None => {
                let d = tone.decay.max(1e-4);
                let s = tone.sustain.clamp(0.0, 1.0);
                if ts < attack + d { 1.0 - (1.0 - s) * (ts - attack) / d } else { s }
            }
        };
        v.level = env;
        // pitch, with a delayed vibrato fading in
        let (rate, depth, delay) = tone.vibrato;
        if depth > 0.0 && ts > delay && v.t.is_multiple_of(VIBRATO_EVERY) {
            v.vib_phase = wrap(v.vib_phase + rate * VIBRATO_EVERY as f64 / sr);
            let fade = if ts - delay < 0.3 { smooth_in(ts - delay, 0.3) } else { 1.0 };
            v.fmul = 1.0 + (TAU * v.vib_phase).sin() * depth * fade * std::f64::consts::LN_2 / 1200.0;
        }
        let fmul = v.fmul;
        let n = tone.unison.clamp(1, MAX_UNISON);
        let idx = v.index.next();
        let (mut l, mut r) = (0.0, 0.0);
        for u in 0..n {
            let (Some(ph), Some(ratio), Some((gl, gr))) = (v.phases.get_mut(u), v.ratio.get(u), v.gains.get(u)) else { break };
            let fu = v.freq * fmul * ratio;
            let dt = (fu / sr).min(0.49);
            *ph = wrap(*ph + dt);
            let s = match tone.osc {
                Osc::Saw => saw(*ph, dt),
                Osc::Triangle => triangle(*ph),
                Osc::SineSaw(mix) => (TAU * *ph).sin() * (1.0 - mix) + saw(*ph, dt) * mix,
                Osc::Fm { ratio, .. } => {
                    if u == 0 {
                        v.mphase = wrap(v.mphase + fu * ratio / sr);
                    }
                    (TAU * *ph + idx * (TAU * v.mphase).sin()).sin()
                }
                Osc::Additive { partials, .. } => {
                    // sin(hθ) by the Chebyshev recurrence: one sine and cosine per sample
                    let th = TAU * *ph;
                    let (s1, c1) = th.sin_cos();
                    let (mut prev, mut cur) = (0.0, s1);
                    let mut acc = 0.0;
                    let most = partials.clamp(1, MAX_PARTIALS);
                    for (h, p) in v.partials.iter().take(most).enumerate() {
                        if fu * (h + 1) as f64 > sr * 0.45 {
                            break;
                        }
                        acc += cur * p.v;
                        let next = 2.0 * c1 * cur - prev;
                        prev = cur;
                        cur = next;
                    }
                    acc
                }
            };
            l += s * gl;
            r += s * gr;
        }
        if let Osc::Additive { partials, .. } = tone.osc {
            for p in v.partials.iter_mut().take(partials.clamp(1, MAX_PARTIALS)) {
                p.next();
            }
        }
        // filter: redesigned while its envelope moves
        let fe = v.fenv.next();
        if !v.settled && v.t.is_multiple_of(FILTER_EVERY) {
            let track = (v.freq / 440.0).powf(tone.key_track.clamp(0.0, 1.0));
            let cut = (tone.cutoff * (0.35 + 0.65 * v.vel) * track * 2f64.powf(fe)).clamp(30.0, sr * 0.45);
            v.coeffs = Coeffs::design(FilterType::LowPass, sr, cut, tone.q, 0.0);
            v.settled = fe.abs() < 0.01;
        }
        let fl = v.lp[0].tick(&v.coeffs, l);
        let fr = v.lp[1].tick(&v.coeffs, r);
        v.t = v.t.saturating_add(1);
        let g = env * v.vel * tone.gain;
        (fl * g, fr * g)
    }
}

/// 0 → 1 over `secs` seconds (vibrato fading in).
fn smooth_in(x: f64, secs: f64) -> f64 {
    super::arrange::smoothstep(x / secs.max(1e-3))
}

impl Instrument for Tonal {
    fn note_on(&mut self, key: u8, velocity: f32) {
        let vel = if velocity.is_finite() { f64::from(velocity).clamp(0.0, 1.0) } else { 0.0 };
        if self.voices.len() >= MAX_VOICES {
            // the quietest released voice, else the oldest, makes room
            let victim = self
                .voices
                .iter()
                .enumerate()
                .filter(|(_, v)| v.rel.is_some())
                .min_by(|a, b| a.1.level.total_cmp(&b.1.level))
                .or_else(|| self.voices.iter().enumerate().max_by_key(|(_, v)| v.t))
                .map(|(i, _)| i);
            if let Some(i) = victim {
                self.voices.swap_remove(i);
            }
        }
        // the same key held again is released first
        self.note_off(key);
        let v = self.start(key, vel);
        self.voices.push(v);
    }

    fn note_off(&mut self, key: u8) {
        let (sr, release) = (self.sr, self.tone.release.max(0.005));
        for v in self.voices.iter_mut().filter(|v| v.key == key && v.rel.is_none()) {
            v.rel = Some((v.t, Decay::new(v.level, release / 5.0, sr)));
        }
    }

    fn render(&mut self, bus: &mut Bus<'_>) {
        let n = bus.len();
        let tone = self.tone;
        let (pl, pr) = pan_gains(tone.pan);
        let (pl, pr) = (pl * std::f64::consts::SQRT_2, pr * std::f64::consts::SQRT_2);
        let send = tone.send as f32;
        for v in &mut self.voices {
            let lanes = bus.dry_l.iter_mut().zip(bus.dry_r.iter_mut()).zip(bus.send_l.iter_mut().zip(bus.send_r.iter_mut())).take(n);
            for ((dl, dr), (sl, sr_)) in lanes {
                if v.done {
                    break;
                }
                let (l, r) = Self::voice_sample(&tone, self.sr, v);
                let (l, r) = ((l * pl) as f32, (r * pr) as f32);
                *dl += l;
                *dr += r;
                *sl += l * send;
                *sr_ += r * send;
            }
        }
        self.voices.retain(|v| !v.done);
    }

    fn release_seconds(&self) -> f64 {
        self.tone.release * 1.5
    }
}

// ---------------------------------------------------------------------------------------------
// drums

#[derive(Clone, Debug)]
struct DrumVoice {
    drum: Drum,
    vel: f64,
    t: u64,
    /// Samples it lasts at most.
    longest: u64,
    rng: Rng,
    /// A second noise source for the right channel of wide cymbals.
    rng2: Rng,
    phase: f64,
    phase2: f64,
    /// Lowest pitch, and how far above it the pitch starts (falling).
    base: f64,
    sweep: Decay,
    /// Amplitude envelopes of the drum's components.
    d: [Decay; 3],
    /// How hard the kick is driven (1 = clean).
    drive: f64,
    /// The lo-fi kit: everything through a dark low-pass.
    dark: bool,
    metal: [f64; 6],
    f: [Biquad; 4],
    c: [Coeffs; 4],
    /// Choked (a closed hat cuts an open one, a released cymbal is damped): samples left of
    /// how many.
    choke: Option<(u64, u64)>,
    done: bool,
}

struct DrumKit {
    kit: Kit,
    sr: f64,
    voices: Vec<DrumVoice>,
    seed: u64,
    count: u64,
}

/// Seconds a released cymbal takes to be damped.
pub const CYMBAL_DAMP: f64 = 0.3;

/// Six square oscillators at inharmonic ratios: the metallic body of hats and cymbals.
const METAL: [f64; 6] = [205.3, 304.4, 369.6, 522.7, 540.0, 800.0];

fn drum_mix(kit: Kit, d: Drum) -> (f64, f64, f64) {
    // (gain, pan, send)
    let soft = if kit == Kit::Lofi { 0.6 } else { 1.0 };
    match d {
        Drum::Kick => (0.32, 0.0, 0.04),
        Drum::Taiko => (0.32, 0.0, if kit == Kit::Cinematic { 0.55 } else { 0.4 }),
        Drum::TomLow => (0.3, -0.3, 0.3),
        Drum::TomMid => (0.3, 0.0, 0.3),
        Drum::TomHigh => (0.3, 0.3, 0.3),
        Drum::Snare => (0.47, 0.0, 0.3 * soft),
        Drum::Clap => (0.98, 0.05, 0.35 * soft),
        Drum::Rim => (0.7, 0.1, 0.2 * soft),
        Drum::HatClosed => (0.35, 0.25, 0.08),
        Drum::HatOpen => (0.18, 0.3, 0.1),
        Drum::Crash => (0.19, 0.0, 0.25),
        Drum::Ride => (0.25, 0.35, 0.15),
        Drum::Shaker => (0.2, -0.35, 0.15),
    }
}

impl DrumKit {
    fn design(sr: f64, kind: FilterType, f: f64, q: f64) -> Coeffs {
        Coeffs::design(kind, sr, f.clamp(20.0, sr * 0.45), q, 0.0)
    }

    fn voice(&mut self, drum: Drum, vel: f64) -> DrumVoice {
        self.count = self.count.wrapping_add(1);
        let s = self.seed ^ self.count.wrapping_mul(0xD6E8_FEB8_6659_FD93) ^ u64::from(drum.gm_key());
        let sr = self.sr;
        let kit = self.kit;
        let dark = kit == Kit::Lofi;
        let mut c = [Coeffs::IDENTITY; 4];
        let dk = |v: f64, tau: f64| Decay::new(v, tau, sr);
        let none = Decay::hold(0.0);
        // (base pitch, pitch sweep, envelopes, drive, longest seconds)
        let (base, sweep, d, drive, longest) = match drum {
            Drum::Kick => {
                let (base, amount, drop, tau, click) = match kit {
                    Kit::Electronic => (50.0, 110.0, 0.035, 0.32, 0.4),
                    Kit::Pop => (55.0, 90.0, 0.03, 0.25, 0.5),
                    Kit::Lofi => (52.0, 80.0, 0.04, 0.22, 0.2),
                    Kit::Cinematic => (45.0, 70.0, 0.05, 0.6, 0.3),
                    Kit::Soft => (55.0, 60.0, 0.04, 0.3, 0.1),
                };
                c[0] = Self::design(sr, FilterType::HighPass, 3000.0, 0.7);
                c[2] = Self::design(sr, FilterType::LowPass, if dark { 2500.0 } else { 9000.0 }, 0.7);
                (base, dk(amount, drop), [dk(1.0, tau), dk(click, 0.003), none], 1.6, tau * 8.0)
            }
            Drum::Taiko | Drum::TomLow | Drum::TomMid | Drum::TomHigh => {
                let (base, tau) = match drum {
                    Drum::Taiko => (if kit == Kit::Cinematic { 52.0 } else { 62.0 }, 0.45),
                    Drum::TomLow => (85.0, 0.3),
                    Drum::TomMid => (115.0, 0.28),
                    _ => (150.0, 0.25),
                };
                c[1] = Self::design(sr, FilterType::LowPass, if dark { 1800.0 } else { 900.0 }, 0.7);
                c[2] = Self::design(sr, FilterType::LowPass, if dark { 2500.0 } else { 9000.0 }, 0.7);
                (base, dk(base * 0.5, 0.04), [dk(1.0, tau), dk(0.5, 0.03), none], 1.0, tau * 8.0)
            }
            Drum::Snare => {
                let (tone_tau, noise_tau) = match kit {
                    Kit::Lofi => (0.06, 0.12),
                    Kit::Electronic => (0.06, 0.14),
                    _ => (0.07, 0.18),
                };
                c[0] = Self::design(sr, FilterType::BandPass, 2000.0, 0.7);
                c[1] = Self::design(sr, FilterType::HighPass, 5000.0, 0.7);
                c[2] = Self::design(sr, FilterType::LowPass, if dark { 4500.0 } else { 12000.0 }, 0.7);
                (190.0, none, [dk(0.45, tone_tau), dk(1.4, noise_tau), dk(0.6, noise_tau * 0.5)], 1.0, noise_tau * 8.0)
            }
            Drum::Clap => {
                c[0] = Self::design(sr, FilterType::BandPass, 1300.0, 1.0);
                c[2] = Self::design(sr, FilterType::LowPass, if dark { 4500.0 } else { 12000.0 }, 0.7);
                (0.0, none, [dk(0.6, 0.11), none, none], 1.0, 1.0)
            }
            Drum::Rim => {
                c[0] = Self::design(sr, FilterType::BandPass, 1800.0, 1.5);
                (820.0, none, [dk(1.5, 0.008), dk(0.6, 0.015), none], 1.0, 0.2)
            }
            Drum::HatClosed | Drum::HatOpen => {
                let tau = if drum == Drum::HatClosed { 0.032 } else { 0.22 };
                c[0] = Self::design(sr, FilterType::HighPass, 7500.0, 0.7);
                c[1] = Self::design(sr, FilterType::HighPass, 6500.0, 0.7);
                c[2] = Self::design(sr, FilterType::LowPass, if dark { 8000.0 } else { 16000.0 }, 0.7);
                (0.0, none, [dk(1.0, tau), none, none], 1.0, tau * 8.0)
            }
            Drum::Crash => {
                c[0] = Self::design(sr, FilterType::HighPass, 3000.0, 0.7);
                c[1] = Self::design(sr, FilterType::HighPass, 3000.0, 0.7);
                c[3] = Self::design(sr, FilterType::HighPass, 5000.0, 0.7);
                (0.0, none, [dk(0.4, 0.9), dk(0.7, 1.1), none], 1.0, 8.0)
            }
            Drum::Ride => {
                c[0] = Self::design(sr, FilterType::BandPass, 5000.0, 0.9);
                c[3] = Self::design(sr, FilterType::HighPass, 4000.0, 0.7);
                (3200.0, none, [dk(1.0, 0.7), dk(0.3, 0.35), dk(0.15, 0.3)], 1.0, 5.0)
            }
            Drum::Shaker => {
                c[0] = Self::design(sr, FilterType::BandPass, 5500.0, 1.2);
                (0.0, none, [dk(1.4, 0.045), none, none], 1.0, 0.5)
            }
        };
        DrumVoice {
            drum,
            vel,
            t: 0,
            longest: (longest * sr) as u64,
            rng: Rng::new(s),
            rng2: Rng::new(s ^ 0x5555_5555),
            phase: 0.0,
            phase2: 0.0,
            base,
            sweep,
            d,
            drive,
            dark,
            metal: [0.0; 6],
            f: [Biquad::default(), Biquad::default(), Biquad::default(), Biquad::default()],
            c,
            choke: None,
            done: false,
        }
    }

    /// One sample of `v` (left, right).
    fn sample(sr: f64, v: &mut DrumVoice) -> (f64, f64) {
        let t = v.t as f64 / sr;
        v.t = v.t.saturating_add(1);
        let [d0, d1, d2] = &mut v.d;
        let (l, r) = match v.drum {
            // components that have died away are no longer computed
            Drum::Kick => {
                let f = v.base + v.sweep.next();
                v.phase = wrap(v.phase + f / sr);
                let drive = v.drive.max(1.0);
                let mut s = soft_clip((TAU * v.phase).sin() * drive) / soft_clip(drive) * d0.next();
                if d1.v > QUIET {
                    s += v.f[0].tick(&v.c[0], v.rng.white()) * d1.next();
                }
                if v.dark {
                    s = v.f[2].tick(&v.c[2], s);
                }
                (s, s)
            }
            Drum::Taiko | Drum::TomLow | Drum::TomMid | Drum::TomHigh => {
                let f = v.base + v.sweep.next();
                v.phase = wrap(v.phase + f / sr);
                let mut s = (TAU * v.phase).sin() * d0.next();
                if d1.v > QUIET {
                    s += v.f[1].tick(&v.c[1], v.rng.white()) * d1.next();
                }
                if v.dark {
                    s = v.f[2].tick(&v.c[2], s);
                }
                (s, s)
            }
            Drum::Snare => {
                let mut s = 0.0;
                if d0.v > QUIET {
                    v.phase = wrap(v.phase + v.base / sr);
                    v.phase2 = wrap(v.phase2 + v.base * 1.74 / sr);
                    s += ((TAU * v.phase).sin() + 0.6 * (TAU * v.phase2).sin()) * d0.next();
                }
                let w = v.rng.white();
                s += v.f[0].tick(&v.c[0], w) * d1.next();
                if d2.v > QUIET {
                    s += v.f[1].tick(&v.c[1], w) * d2.next();
                }
                if v.dark {
                    s = v.f[2].tick(&v.c[2], s);
                }
                (s, s)
            }
            Drum::Clap => {
                // three quick bursts, then the tail
                let mut env = 0.0;
                if t < 0.06 {
                    for k in 0..3 {
                        let t0 = f64::from(k) * 0.011;
                        if t >= t0 {
                            env += (-(t - t0) / 0.0045).exp();
                        }
                    }
                }
                if t >= 0.022 {
                    env += d0.next();
                }
                let s = v.f[0].tick(&v.c[0], v.rng.white()) * env * 1.6;
                let s = v.f[2].tick(&v.c[2], s);
                (s, s * 0.95)
            }
            Drum::Rim => {
                v.phase = wrap(v.phase + v.base / sr);
                let s = v.f[0].tick(&v.c[0], v.rng.white()) * d0.next() + (TAU * v.phase).sin() * d1.next();
                (s, s)
            }
            Drum::HatClosed | Drum::HatOpen | Drum::Ride | Drum::Crash => {
                let mut metal = 0.0;
                let scale = if v.drum == Drum::Ride { 1.4 } else { 1.0 };
                for (ph, f) in v.metal.iter_mut().zip(METAL) {
                    *ph = wrap(*ph + f * scale / sr);
                    metal += if *ph < 0.5 { 1.0 } else { -1.0 };
                }
                metal /= 6.0;
                match v.drum {
                    Drum::HatClosed | Drum::HatOpen => {
                        let n = v.f[0].tick(&v.c[0], v.rng.white());
                        let m = v.f[1].tick(&v.c[1], metal);
                        let s = v.f[2].tick(&v.c[2], (n * 0.8 + m * 0.5) * d0.next());
                        (s, s)
                    }
                    Drum::Ride => {
                        v.phase = wrap(v.phase + v.base / sr);
                        let m = v.f[3].tick(&v.c[3], metal) * d0.next();
                        let n = v.f[0].tick(&v.c[0], v.rng.white()) * d1.next();
                        let s = m + n + (TAU * v.phase).sin() * d2.next();
                        (s, s)
                    }
                    _ => {
                        let m = v.f[3].tick(&v.c[3], metal) * d0.next();
                        let e = d1.next();
                        let nl = v.f[0].tick(&v.c[0], v.rng.white()) * e;
                        let nr = v.f[1].tick(&v.c[1], v.rng2.white()) * e;
                        (nl + m, nr + m)
                    }
                }
            }
            Drum::Shaker => {
                let a = (t / 0.006).min(1.0);
                let e = if t < 0.006 { d0.v } else { d0.next() };
                let s = v.f[0].tick(&v.c[0], v.rng.white()) * a * e;
                (s, s)
            }
        };
        let mut g = v.vel;
        if let Some((left, total)) = v.choke.as_mut() {
            *left = left.saturating_sub(1);
            g *= *left as f64 / (*total).max(1) as f64;
            if *left == 0 {
                v.done = true;
            }
        }
        if v.t > v.longest {
            v.done = true;
        }
        // inaudible: faded out over a few milliseconds and dropped
        if v.choke.is_none() && t > 0.07 && v.d.iter().all(|d| d.v.abs() < DRUM_SILENT) {
            let n = ((0.005 * sr) as u64).max(1);
            v.choke = Some((n, n));
        }
        (l * g, r * g)
    }
}

/// Envelope level under which a drum's component is no longer computed.
const QUIET: f64 = 1e-5;

/// Envelope level under which a drum is no longer heard (−50 dB).
const DRUM_SILENT: f64 = 3e-3;

/// tanh, closely enough for |x| ≤ 3 (a Padé approximant), cheaply.
fn soft_clip(x: f64) -> f64 {
    let x = x.clamp(-3.0, 3.0);
    let x2 = x * x;
    x * (27.0 + x2) / (27.0 + 9.0 * x2)
}

impl Instrument for DrumKit {
    fn note_on(&mut self, key: u8, velocity: f32) {
        let Some(drum) = Drum::from_key(key) else { return };
        let vel = if velocity.is_finite() { f64::from(velocity).clamp(0.0, 1.0) } else { 0.0 };
        if drum == Drum::HatClosed {
            let n = ((0.01 * self.sr) as u64).max(1);
            for v in self.voices.iter_mut().filter(|v| v.drum == Drum::HatOpen && v.choke.is_none()) {
                v.choke = Some((n, n));
            }
        }
        if self.voices.len() >= MAX_VOICES
            && let Some(i) = self.voices.iter().enumerate().max_by_key(|(_, v)| v.t).map(|(i, _)| i)
        {
            self.voices.swap_remove(i);
        }
        let v = self.voice(drum, vel);
        self.voices.push(v);
    }

    /// Cymbals are damped by hand when released (over [`CYMBAL_DAMP`]); the other drums ring
    /// out on their own.
    fn note_off(&mut self, key: u8) {
        let Some(drum) = Drum::from_key(key).filter(|d| matches!(d, Drum::Crash | Drum::Ride)) else { return };
        let n = ((CYMBAL_DAMP * self.sr) as u64).max(1);
        // the oldest one still ringing: each release damps its own strike
        if let Some(v) = self.voices.iter_mut().filter(|v| v.drum == drum && v.choke.is_none()).max_by_key(|v| v.t) {
            v.choke = Some((n, n));
        }
    }

    fn render(&mut self, bus: &mut Bus<'_>) {
        let n = bus.len();
        for v in &mut self.voices {
            let (gain, pan, send) = drum_mix(self.kit, v.drum);
            let (pl, pr) = pan_gains(pan);
            let (pl, pr) = (pl * std::f64::consts::SQRT_2 * gain, pr * std::f64::consts::SQRT_2 * gain);
            let send = send as f32;
            let lanes = bus.dry_l.iter_mut().zip(bus.dry_r.iter_mut()).zip(bus.send_l.iter_mut().zip(bus.send_r.iter_mut())).take(n);
            for ((dl, dr), (sl, sr_)) in lanes {
                if v.done {
                    break;
                }
                let (l, r) = Self::sample(self.sr, v);
                let (l, r) = ((l * pl) as f32, (r * pr) as f32);
                *dl += l;
                *dr += r;
                *sl += l * send;
                *sr_ += r * send;
            }
        }
        self.voices.retain(|v| !v.done);
    }

    fn release_seconds(&self) -> f64 {
        0.0
    }
}

// ---------------------------------------------------------------------------------------------
// effects

#[derive(Clone, Debug)]
struct FxVoice {
    fx: Fx,
    vel: f64,
    t: u64,
    off: Option<u64>,
    rng: Rng,
    phase: f64,
    /// The riser's pitch now (updated with its filter).
    pitch: f64,
    sub: Decay,
    crack: Decay,
    bp: Biquad,
    c: Coeffs,
    lp: Biquad,
    lc: Coeffs,
    done: bool,
}

struct FxRack {
    sr: f64,
    /// Seconds a riser takes to rise fully.
    rise: f64,
    voices: Vec<FxVoice>,
    seed: u64,
    count: u64,
}

impl FxRack {
    fn sample(sr: f64, rise: f64, v: &mut FxVoice) -> (f64, f64) {
        let t = v.t as f64 / sr;
        v.t = v.t.saturating_add(1);
        let since_off = v.off.map(|t0| v.t.saturating_sub(t0) as f64 / sr);
        let out = match v.fx {
            Fx::Riser => {
                let x = (t / rise).min(1.0);
                if v.t % 16 == 1 {
                    v.c = Coeffs::design(FilterType::BandPass, sr, (300.0 * 24f64.powf(x)).min(sr * 0.45), 1.3, 0.0);
                    v.pitch = 110.0 * 6f64.powf(x);
                }
                let air = v.bp.tick(&v.c, v.rng.white()) * 1.8;
                v.phase = wrap(v.phase + v.pitch / sr);
                let tone = saw(v.phase, v.pitch / sr) * 0.12;
                let mut g = x * x;
                if let Some(since) = since_off {
                    g *= (1.0 - since / 0.012).max(0.0);
                    if since >= 0.012 {
                        v.done = true;
                    }
                }
                (air + tone) * g
            }
            Fx::Impact => {
                let f = 38.0 + 72.0 * v.crack.v.powf(4.0 / 3.0);
                v.phase = wrap(v.phase + f / sr);
                let sub = ((TAU * v.phase).sin() * 2.0).tanh() / 2f64.tanh() * v.sub.next();
                let crack = v.lp.tick(&v.lc, v.rng.white()) * v.crack.next() * 0.9;
                let mut g = 1.0;
                if let Some(since) = since_off {
                    // released: dies away over a third of a second
                    g = (1.0 - since / CYMBAL_DAMP).max(0.0);
                    if since >= CYMBAL_DAMP {
                        v.done = true;
                    }
                }
                (sub + crack) * g
            }
        };
        if t > 30.0 {
            v.done = true;
        }
        let g = out * v.vel;
        (g, g)
    }
}

impl Instrument for FxRack {
    fn note_on(&mut self, key: u8, velocity: f32) {
        let fx = match key {
            1 => Fx::Riser,
            2 => Fx::Impact,
            _ => return,
        };
        if self.voices.len() >= MAX_VOICES {
            return;
        }
        self.count = self.count.wrapping_add(1);
        let vel = if velocity.is_finite() { f64::from(velocity).clamp(0.0, 1.0) } else { 0.0 };
        let sr = self.sr;
        self.voices.push(FxVoice {
            fx,
            vel,
            t: 0,
            off: None,
            rng: Rng::new(self.seed ^ self.count.wrapping_mul(0xA24B_AED4_963E_E407)),
            phase: 0.0,
            pitch: 110.0,
            sub: Decay::new(1.0, 1.1, sr),
            // the crack's decay (0.08 s) also drives the boom's pitch drop (0.06 s)
            crack: Decay::new(1.0, 0.08, sr),
            bp: Biquad::default(),
            c: Coeffs::IDENTITY,
            lp: Biquad::default(),
            lc: Coeffs::design(FilterType::LowPass, sr, 1500.0f64.min(sr * 0.45), 0.8, 0.0),
            done: false,
        });
    }

    fn note_off(&mut self, key: u8) {
        let fx = if key == Fx::Riser.key() {
            Fx::Riser
        } else if key == Fx::Impact.key() {
            Fx::Impact
        } else {
            return;
        };
        for v in self.voices.iter_mut().filter(|v| v.fx == fx && v.off.is_none()) {
            v.off = Some(v.t);
        }
    }

    fn render(&mut self, bus: &mut Bus<'_>) {
        let n = bus.len();
        for v in &mut self.voices {
            let (gain, send) = match v.fx {
                Fx::Riser => (0.18, 0.4f32),
                Fx::Impact => (0.8, 0.5f32),
            };
            let lanes = bus.dry_l.iter_mut().zip(bus.dry_r.iter_mut()).zip(bus.send_l.iter_mut().zip(bus.send_r.iter_mut())).take(n);
            for ((dl, dr), (sl, sr_)) in lanes {
                if v.done {
                    break;
                }
                let (l, r) = Self::sample(self.sr, self.rise, v);
                let (l, r) = ((l * gain) as f32, (r * gain) as f32);
                *dl += l;
                *dr += r;
                *sl += l * send;
                *sr_ += r * send;
            }
        }
        self.voices.retain(|v| !v.done);
    }

    fn release_seconds(&self) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> Context {
        Context { sample_rate: 48_000, tempo: Tempo::from_bpm(120.0), genre: Genre::Pop, seed: 1 }
    }

    fn play(inst: &mut dyn Instrument, key: u8, on: usize, total: usize) -> (Vec<f32>, Vec<f32>) {
        let mut l = vec![0.0f32; total];
        let mut r = vec![0.0f32; total];
        let mut sl = vec![0.0f32; total];
        let mut sr = vec![0.0f32; total];
        inst.note_on(key, 0.9);
        let (a, b) = l.split_at_mut(on.min(total));
        let (c, d) = r.split_at_mut(on.min(total));
        let (e, f) = sl.split_at_mut(on.min(total));
        let (g, h) = sr.split_at_mut(on.min(total));
        inst.render(&mut Bus { dry_l: a, dry_r: c, send_l: e, send_r: g });
        inst.note_off(key);
        inst.render(&mut Bus { dry_l: b, dry_r: d, send_l: f, send_r: h });
        (l, r)
    }

    fn rms(x: &[f32]) -> f64 {
        (x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / x.len().max(1) as f64).sqrt()
    }

    #[test]
    fn every_patch_sounds_finite_and_stops_after_its_release() {
        let bank = OscillatorBank;
        let patches = [
            Patch::Strings,
            Patch::WarmPad,
            Patch::SuperSaw,
            Patch::Piano,
            Patch::EPiano,
            Patch::Pluck,
            Patch::Brass,
            Patch::Bell,
            Patch::Flute,
            Patch::SawLead,
            Patch::SynthBass,
            Patch::SubBass,
            Patch::RoundBass,
        ];
        for p in patches {
            let mut inst = bank.instrument(p, &ctx()).unwrap();
            let (l, r) = play(inst.as_mut(), 60, 24_000, 48_000 * 4);
            assert!(l.iter().chain(&r).all(|x| x.is_finite() && x.abs() < 2.0), "{p:?}");
            assert!(rms(&l[..24_000]) > 1e-3, "{p:?} sounds");
            assert!(rms(&l[48_000 * 3..]) < 1e-4, "{p:?} has stopped 2.5 s after release");
            assert!(p.gm_program().is_some());
        }
        assert_eq!(Patch::Kit(Kit::Pop).gm_program(), None);
    }

    #[test]
    fn every_drum_of_every_kit_hits_and_dies() {
        let bank = OscillatorBank;
        for kit in [Kit::Cinematic, Kit::Electronic, Kit::Pop, Kit::Lofi, Kit::Soft] {
            for d in Drum::ALL {
                let mut inst = bank.instrument(Patch::Kit(kit), &ctx()).unwrap();
                let (l, r) = play(inst.as_mut(), d.gm_key(), 100, 48_000 * 10);
                assert!(l.iter().chain(&r).all(|x| x.is_finite() && x.abs() < 3.0), "{kit:?} {d:?}");
                assert!(rms(&l[..4_800]) > 1e-3, "{kit:?} {d:?} sounds");
                assert!(rms(&l[48_000 * 9..]) < 1e-5, "{kit:?} {d:?} dies");
                assert_eq!(Drum::from_key(d.gm_key()), Some(d));
            }
        }
        // a key that is not a drum is ignored
        let mut inst = bank.instrument(Patch::Kit(Kit::Pop), &ctx()).unwrap();
        let (l, _) = play(inst.as_mut(), 12, 100, 4_800);
        assert!(l.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn effects_rise_and_hit() {
        let bank = OscillatorBank;
        let mut fx = bank.instrument(Patch::Fx, &ctx()).unwrap();
        let (l, _) = play(fx.as_mut(), Fx::Riser.key(), 96_000, 100_000);
        assert!(rms(&l[80_000..96_000]) > 3.0 * rms(&l[..16_000]), "rises");
        assert!(rms(&l[97_000..]) < 1e-6, "stops at its end");
        let mut fx = bank.instrument(Patch::Fx, &ctx()).unwrap();
        let (l, _) = play(fx.as_mut(), Fx::Impact.key(), 100, 48_000 * 9);
        assert!(rms(&l[..12_000]) > 10.0 * rms(&l[48_000 * 4..]));
    }

    #[test]
    fn polyphony_is_capped_and_hostile_notes_are_harmless() {
        let bank = OscillatorBank;
        let mut inst = bank.instrument(Patch::Strings, &ctx()).unwrap();
        for k in 0..=255u8 {
            inst.note_on(k, f32::NAN);
            inst.note_on(k, 1e30);
        }
        let mut l = vec![0.0f32; 512];
        let mut r = vec![0.0f32; 512];
        let mut a = vec![0.0f32; 512];
        let mut b = vec![0.0f32; 512];
        inst.render(&mut Bus { dry_l: &mut l, dry_r: &mut r, send_l: &mut a, send_r: &mut b });
        assert!(l.iter().all(|x| x.is_finite()));
        // mismatched buffers render only what they share
        let mut short = vec![0.0f32; 3];
        inst.render(&mut Bus { dry_l: &mut l, dry_r: &mut short, send_l: &mut a, send_r: &mut b });
        assert!(bank.instrument(Patch::Piano, &Context { sample_rate: 0, ..ctx() }).is_err());
    }

    /// Loudness of `patch` playing `notes` (start s, length s, key, velocity) for `secs`.
    fn lufs(patch: Patch, notes: &[(f64, f64, u8, f32)], secs: f64) -> f64 {
        let mut inst = OscillatorBank.instrument(patch, &ctx()).unwrap();
        let n = (secs * 48_000.0) as usize;
        let mut ev: Vec<(usize, bool, u8, f32)> = Vec::new();
        for &(s, l, k, v) in notes {
            ev.push(((s * 48_000.0) as usize, true, k, v));
            ev.push((((s + l) * 48_000.0) as usize, false, k, v));
        }
        ev.sort_by_key(|e| (e.0, e.1));
        let (mut l, mut r, mut sl, mut sr) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
        let (mut at, mut i) = (0, 0);
        while at < n {
            while i < ev.len() && ev[i].0 <= at {
                if ev[i].1 {
                    inst.note_on(ev[i].2, ev[i].3)
                } else {
                    inst.note_off(ev[i].2)
                }
                i += 1;
            }
            let next = if i < ev.len() { ev[i].0 } else { n }.max(at + 1).min(n);
            inst.render(&mut Bus { dry_l: &mut l[at..next], dry_r: &mut r[at..next], send_l: &mut sl[at..next], send_r: &mut sr[at..next] });
            at = next;
        }
        let mut m = crate::loudness::LoudnessMeter::new(48_000.0, 2);
        m.process(&[&l, &r]);
        m.integrated()
    }

    /// Calibration aid: `cargo test -p filmcraft-audio-dsp --release patch_loudness -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn patch_loudness() {
        let chord: Vec<(f64, f64, u8, f32)> = (0..4).flat_map(|b| [60u8, 64, 67].map(|k| (f64::from(b) * 2.0, 2.0, k, 0.7))).collect();
        let line: Vec<(f64, f64, u8, f32)> = (0..32).map(|i| (i as f64 * 0.25, 0.22, [72u8, 74, 76, 79, 76, 74, 72, 71][i % 8], 0.7)).collect();
        let bass: Vec<(f64, f64, u8, f32)> = (0..16).map(|i| (i as f64 * 0.5, 0.45, 33u8, 0.7)).collect();
        for p in [Patch::Strings, Patch::WarmPad, Patch::SuperSaw, Patch::Piano, Patch::EPiano] {
            println!("chord {:16} {:6.1}", p.id(), lufs(p, &chord, 8.0));
        }
        for p in [Patch::Pluck, Patch::Brass, Patch::Bell, Patch::Flute, Patch::SawLead, Patch::Piano, Patch::EPiano] {
            println!("line  {:16} {:6.1}", p.id(), lufs(p, &line, 8.0));
        }
        for p in [Patch::SynthBass, Patch::SubBass, Patch::RoundBass] {
            println!("bass  {:16} {:6.1}", p.id(), lufs(p, &bass, 8.0));
        }
        for kit in [Kit::Cinematic, Kit::Electronic, Kit::Pop, Kit::Lofi, Kit::Soft] {
            let mut row = String::new();
            for d in Drum::ALL {
                let hits: Vec<(f64, f64, u8, f32)> = (0..16).map(|i| (f64::from(i) * 0.5, 0.1, d.gm_key(), 0.7)).collect();
                row.push_str(&format!("{d:?} {:.0}  ", lufs(Patch::Kit(kit), &hits, 8.0)));
            }
            println!("{kit:?}: {row}");
        }
    }
}
