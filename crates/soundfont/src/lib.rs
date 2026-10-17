//! SoundFont instruments for FilmCraft's soundtrack engine (layer L2).
//!
//! The procedural music engine ([`filmcraft_audio_dsp::music`]) plays every part on an
//! [`InstrumentBank`](filmcraft_audio_dsp::music::InstrumentBank); every patch names a General
//! MIDI program and every drum a General MIDI percussion key. This crate plays them on a
//! SoundFont 2 file through **rustysynth** (MIT, pure Rust):
//!
//! - [`check`]: a structural check of an `.sf2` file — every chunk inside its parent, every size
//!   even, every record list whole, every zone and sample range in bounds — run before rustysynth
//!   reads a byte. A SoundFont is hostile input; rustysynth trusts it (its sample loader is only
//!   memory-safe for an even sample-chunk size, and its parser indexes with the file's numbers).
//! - [`SoundFont`]: a checked, loaded file (at most [`check::MAX_FILE_BYTES`]).
//! - [`SoundFontBank`]: the instrument bank. A patch the font has no preset for, risers and
//!   impacts, and sample rates rustysynth does not run at play on the built-in oscillators. Each
//!   SoundFont instrument is level-matched to the oscillator voice it replaces, so the mix keeps
//!   its balance. Every call into rustysynth runs under `catch_unwind`; [`generate`] makes the
//!   track again on the oscillators if the font failed while playing.
//! - [`catalog`]: the GeneralUser GS download (S. Christian Collins, GeneralUser GS License
//!   v2.0), pinned to a commit of the author's repository and its SHA-256. Never bundled.
//! - [`synthetic`]: a tiny generated SoundFont for tests.
//!
//! The arrangement, harmony and beat map do not depend on the bank: the same request gives the
//! same [`Plan`](filmcraft_audio_dsp::music::Plan) on the oscillators and on any SoundFont.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod catalog;
pub mod check;
pub mod synthetic;

mod bank;

pub use bank::{FontPlayer, SoundFontBank, generate};

use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Arc;

/// Why a SoundFont could not be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// General MIDI bank of the percussion kits.
pub const DRUM_BANK: i32 = 128;

/// A checked, loaded SoundFont. Cheap to clone (the samples are shared).
#[derive(Clone)]
pub struct SoundFont {
    font: Arc<rustysynth::SoundFont>,
    name: String,
    presets: HashSet<(i32, i32)>,
    bytes: u64,
    summary: check::Summary,
}

impl std::fmt::Debug for SoundFont {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SoundFont").field("name", &self.name).field("presets", &self.presets.len()).field("bytes", &self.bytes).finish()
    }
}

impl SoundFont {
    /// Check and load a SoundFont from its bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() as u64 > check::MAX_FILE_BYTES {
            return Err(Error(format!("the SoundFont is larger than {} MB", check::MAX_FILE_BYTES >> 20)));
        }
        let summary = check::check(bytes)?;
        let parsed = catch_unwind(AssertUnwindSafe(|| {
            let mut cursor = std::io::Cursor::new(bytes);
            rustysynth::SoundFont::new(&mut cursor)
        }));
        let font = match parsed {
            Ok(Ok(f)) => f,
            Ok(Err(e)) => return Err(Error(format!("not a usable SoundFont: {e}"))),
            Err(_) => return Err(Error("not a usable SoundFont (its reader failed on it)".into())),
        };
        let presets: HashSet<(i32, i32)> = font.get_presets().iter().map(|p| (p.get_bank_number(), p.get_patch_number())).collect();
        if presets.is_empty() {
            return Err(Error("the SoundFont has no presets".into()));
        }
        let name = if summary.name.trim().is_empty() { "SoundFont".to_string() } else { summary.name.trim().to_string() };
        Ok(SoundFont { font: Arc::new(font), name, presets, bytes: bytes.len() as u64, summary })
    }

    /// Check and load the SoundFont at `path` (files over [`check::MAX_FILE_BYTES`] are refused
    /// before they are read).
    pub fn open(path: &Path) -> Result<Self, Error> {
        let md = std::fs::metadata(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        if !md.is_file() {
            return Err(Error(format!("{} is not a file", path.display())));
        }
        if md.len() > check::MAX_FILE_BYTES {
            return Err(Error(format!("{} is larger than {} MB", path.display(), check::MAX_FILE_BYTES >> 20)));
        }
        let bytes = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        Self::from_bytes(&bytes).map_err(|e| Error(format!("{}: {e}", path.display())))
    }

    /// Its name (the file's `INAM`), e.g. "GeneralUser GS v2.0.3".
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Size of the file, bytes.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Number of presets.
    pub fn presets(&self) -> usize {
        self.presets.len()
    }

    /// Whether it has the preset `program` in `bank` ([`DRUM_BANK`] for drum kits).
    pub fn has(&self, bank: i32, program: i32) -> bool {
        self.presets.contains(&(bank, program))
    }

    /// Whether it has any drum kit.
    pub fn has_drums(&self) -> bool {
        self.presets.iter().any(|(b, _)| *b == DRUM_BANK)
    }

    /// What [`check::check`] read from the file.
    pub fn summary(&self) -> &check::Summary {
        &self.summary
    }

    pub(crate) fn inner(&self) -> &Arc<rustysynth::SoundFont> {
        &self.font
    }
}

#[cfg(test)]
mod tests;
