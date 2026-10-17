//! The SoundFont loader against hostile files (truncations, bit flips, corrupt sizes, all under
//! `catch_unwind`), the oscillator fallback, and the same arrangement with and without a font.

use std::panic::{AssertUnwindSafe, catch_unwind};

use filmcraft_audio_dsp::music::{self, Energy, Genre, Mood, Request};
use filmcraft_time::Tick;

use super::*;

/// The message [`SoundFont::from_bytes`] gives when rustysynth itself panicked: the check must
/// have let something through.
const READER_FAILED: &str = "its reader failed";

/// Load `bytes`, proving neither the check nor rustysynth panics on them.
fn load(bytes: &[u8]) -> Result<SoundFont, Error> {
    let r = catch_unwind(AssertUnwindSafe(|| SoundFont::from_bytes(bytes)));
    let r = r.unwrap_or_else(|_| panic!("the loader panicked on a {}-byte file", bytes.len()));
    if let Err(e) = &r {
        assert!(!e.0.contains(READER_FAILED), "the check let a file through that rustysynth panicked on: {e}");
    }
    r
}

/// xorshift, for reproducible mutations.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// Where the sample data (after the `smpl` header) lies.
fn sample_data(b: &[u8]) -> (usize, usize) {
    let at = b.windows(4).position(|w| w == b"smpl").unwrap();
    let size = u32::from_le_bytes(b[at + 4..at + 8].try_into().unwrap()) as usize;
    (at + 8, at + 8 + size)
}

fn req(mood: Mood, genre: Genre, seconds: f64, seed: u64) -> Request {
    Request { genre: Some(genre), seed, ..Request::new(mood, Tick::from_seconds_f64(seconds)) }
}

/// Play a few notes on every patch of a loaded font (under `catch_unwind`: nothing may escape).
fn play_everything(font: &SoundFont) {
    let bank = SoundFontBank::new(font.clone());
    let r = catch_unwind(AssertUnwindSafe(|| generate(&req(Mood::Happy, Genre::Pop, 1.5, 3), 48_000, &bank)));
    let r = r.unwrap_or_else(|_| panic!("playing a loaded font panicked"));
    let (track, _) = r.unwrap();
    assert!(track.mix.left.iter().chain(&track.mix.right).all(|x| x.is_finite()));
}

#[test]
fn the_tiny_font_loads() {
    let f = load(&synthetic::tiny()).unwrap();
    assert_eq!(f.name(), "FilmCraft Test Font");
    assert!(f.has(0, 0) && f.has(0, 48) && f.has(DRUM_BANK, 0) && f.has_drums());
    assert!(!f.has(0, 1));
    assert_eq!(f.presets(), synthetic::MUSIC_PROGRAMS.len() + 1);
    assert_eq!(f.summary().version, (2, 1));
    assert_eq!(f.summary().instruments, 2);
    assert_eq!(f.summary().sample_headers, 2);
    assert!(f.summary().presets.iter().any(|(bank, program, name)| (*bank, *program, name.as_str()) == (128, 0, "Noise Kit")));
}

#[test]
fn every_truncation_is_refused_without_a_panic() {
    let full = synthetic::tiny();
    for n in 0..full.len() {
        assert!(load(&full[..n]).is_err(), "a {n}-byte prefix loaded");
    }
    assert!(load(&full).is_ok());
}

#[test]
fn bit_flips_never_panic() {
    let full = synthetic::tiny();
    let (s0, s1) = sample_data(&full);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut loaded = 0;
    for round in 0..6000 {
        let mut b = full.clone();
        for _ in 0..1 + rng.below(4) {
            // half the rounds flip only the structure (everything but the sample data)
            let mut i = rng.below(b.len());
            if round % 2 == 0 && (s0..s1).contains(&i) {
                i = if rng.below(2) == 0 { rng.below(s0) } else { s1 + rng.below(b.len() - s1) };
            }
            b[i] ^= 1 << rng.below(8);
        }
        if let Ok(f) = load(&b) {
            loaded += 1;
            // a font that loads plays without anything escaping
            if loaded % 40 == 1 {
                play_everything(&f);
            }
        }
    }
    assert!(loaded > 0, "some flips (in sample data) keep a valid font");
}

#[test]
fn corrupt_sizes_and_indices_never_panic() {
    let full = synthetic::tiny();
    let (s0, s1) = sample_data(&full);
    for at in (0..full.len().saturating_sub(4)).filter(|i| !(s0 + 8..s1).contains(i)) {
        for v in [0u32, 1, 3, 0x7FFF_FFFF, 0xFFFF_FFFF, 0x8000_0000, 0xFFFF, 0x1_0000] {
            let mut b = full.clone();
            b[at..at + 4].copy_from_slice(&v.to_le_bytes());
            let _ = load(&b);
        }
        for v in [0u16, 1, 0x7FFF, 0xFFFF, 0x8000] {
            let mut b = full.clone();
            b[at..at + 2].copy_from_slice(&v.to_le_bytes());
            let _ = load(&b);
        }
    }
}

#[test]
fn junk_and_odd_samples_are_refused() {
    for b in [&b""[..], b"RIFF", b"RIFF\0\0\0\0sfbk", &[0xFFu8; 64][..], b"RIFF\x04\0\0\0WAVE"] {
        assert!(load(b).is_err());
    }
    // an odd sample-chunk size (rustysynth would write past its buffer): refused by the check
    let full = synthetic::tiny();
    let at = full.windows(4).position(|w| w == b"smpl").unwrap();
    let size = u32::from_le_bytes(full[at + 4..at + 8].try_into().unwrap());
    let mut b = full.clone();
    b[at + 4..at + 8].copy_from_slice(&(size - 1).to_le_bytes());
    let e = load(&b).unwrap_err();
    assert!(e.0.contains("odd size"), "{e}");
    // compressed (SF3) samples
    let mut b = full.clone();
    b[at + 8..at + 12].copy_from_slice(b"OggS");
    assert!(load(&b).unwrap_err().0.contains("SF3"));
    // a missing file, a directory
    assert!(SoundFont::open(std::path::Path::new("/definitely/not/here.sf2")).is_err());
    assert!(SoundFont::open(&std::env::temp_dir()).is_err());
}

#[test]
fn the_same_seed_gives_the_same_arrangement_with_and_without_a_font() {
    let font = load(&synthetic::tiny()).unwrap();
    let bank = SoundFontBank::new(font);
    for (mood, genre) in [(Mood::Epic, Genre::Cinematic), (Mood::Calm, Genre::Piano), (Mood::Energetic, Genre::Electronic), (Mood::Sad, Genre::Lofi)] {
        let mut r = req(mood, genre, 7.3, 11);
        r.energy = Some(Energy::Level(0.8));
        r.stems = true;
        let osc = music::generate(&r, 48_000).unwrap();
        let (sf, note) = generate(&r, 48_000, &bank).unwrap();
        assert!(note.is_none());
        assert_eq!(osc.plan, sf.plan, "{mood:?} {genre:?}");
        assert_eq!(osc.plan.beats(), sf.plan.beats());
        assert_eq!(sf.mix.left.len(), osc.mix.left.len());
        assert_eq!(sf.mix.left.len() as u64, sf.plan.samples);
        assert_ne!(sf.mix.left, osc.mix.left, "the font played ({mood:?})");
        assert!(sf.instruments.starts_with("FilmCraft Test Font (SoundFont)"), "{}", sf.instruments);
        assert_eq!(osc.instruments, "oscillators");
        // the stems still sum to the mix
        for i in (0..sf.mix.left.len()).step_by(997) {
            let sum: f32 = sf.stems.iter().map(|(_, s)| s.left[i]).sum();
            assert!((sum - sf.mix.left[i]).abs() < 1e-4, "{i}");
        }
        // and the same request plays the same again
        let (again, _) = generate(&r, 48_000, &bank).unwrap();
        assert_eq!(again.mix.left, sf.mix.left);
        assert_eq!(again.mix.right, sf.mix.right);
    }
}

#[test]
fn without_a_matching_preset_the_oscillators_play() {
    // a font whose only preset is a program the engine never plays, and no drums
    let font = load(&synthetic::build("Odd Font", &[127], false)).unwrap();
    let bank = SoundFontBank::new(font);
    let r = req(Mood::Uplifting, Genre::Pop, 6.0, 5);
    let osc = music::generate(&r, 48_000).unwrap();
    let (sf, note) = generate(&r, 48_000, &bank).unwrap();
    assert!(note.is_none());
    assert_eq!(sf.mix.left, osc.mix.left, "every part fell back to its oscillator voice");
    assert!(sf.instruments.contains("+ oscillators"), "{}", sf.instruments);
    // below rustysynth's sample rates too
    let low = music::generate(&r, 8_000).unwrap();
    let (sf, _) = generate(&r, 8_000, &bank).unwrap();
    assert_eq!(sf.mix.left, low.mix.left);
}

#[test]
fn a_font_that_fails_while_playing_falls_back_to_the_oscillators() {
    let font = load(&synthetic::runaway()).unwrap();
    let bank = SoundFontBank::new(font);
    let r = req(Mood::Happy, Genre::Pop, 4.0, 2);
    let osc = music::generate(&r, 48_000).unwrap();
    let r2 = r.clone();
    let out = catch_unwind(AssertUnwindSafe(|| generate(&r2, 48_000, &bank))).expect("nothing escapes");
    let (sf, _) = out.unwrap();
    // its level check already heard it fail: every instrument is on the oscillators
    assert_eq!(sf.mix.left, osc.mix.left);
    // past the level check (levels known), the failure happens mid-track: the whole track is
    // made again on the oscillators and the note says so
    bank.assume_levels(&r, 48_000);
    let (sf, note) = generate(&r, 48_000, &bank).unwrap();
    assert!(note.as_deref().is_some_and(|n| n.contains("failed while playing")), "{note:?}");
    assert_eq!(sf.mix.left, osc.mix.left);
    assert_eq!(sf.instruments, "oscillators");
}
