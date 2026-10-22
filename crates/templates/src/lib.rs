//! FilmCraft video templates: whole-sequence starters with media slots (`.fcvideo-template`).
//!
//! A template is an ordinary project fragment plus slot metadata, not a new timeline model. See
//! the crate README for the file format; in short, a package is a folder (or the same files in a
//! ZIP archive) holding
//!
//! - `template.json`, the [`Manifest`]: name, description, version, author, licence, aspect
//!   [`Variant`]s, typed [`Slot`]s (video, photo, text, logo, music, captions) with their placements,
//!   constraints and auto-fit policy, references to a look, a caption style and an export
//!   preset, and the music's [`BeatGrid`] in exact ticks;
//! - `sequence.json`, a FilmCraft project file (the `.fcproj` schema) with the template's
//!   sequence and its generated placeholder media;
//! - optionally `assets/…`, openly licensed files the sequence uses, each listed in the manifest
//!   with its SHA-256, author, source and licence.
//!
//! Modules:
//!
//! - [`manifest`]: the manifest types and the strict reader (every error names its field);
//! - [`package`]: packages from a file map, a folder or a ZIP ([`zip`]), cross-checked against
//!   the sequence;
//! - [`fit`]: how a slot's media conforms to its placement (trim, loop, stretch, hold, remix);
//! - [`beats`]: snapping cuts to the beat grid and re-quantizing them to a new track's beats;
//! - [`apply`]: instantiating a template into a project — fresh ids, filled slots, re-timed
//!   cuts, the variant's layout, the look and caption style;
//! - [`save`]: the minimal "save sequence as template" (placeholder-flagged clips become slots);
//! - [`builtin`]: the built-in starters, original designs made in code.
//!
//! Nothing here touches the file system: the engine reads and writes files through its host
//! services and calls these functions on bytes and projects.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod apply;
pub mod beats;
pub mod builtin;
pub mod fit;
pub mod manifest;
pub mod package;
pub mod qr;
pub mod save;
pub mod zip;

pub use apply::{Applied, ApplyOptions, Cue, MediaFill, MusicBeats, SlotFill};
pub use manifest::{
    AiHints, AssetEntry, BeatGrid, CaptionSpec, Constraints, EnergyHint, Fit, Framing, Hint, LayerHint, Look, Manifest, Orientation, Placement, Portion,
    Purpose, QuantizeMode, ShortFit, Slot, SlotDefault, SlotKind, SpeechHint, Variant,
};
pub use package::Package;

/// `format` of a template manifest.
pub const FORMAT: &str = "filmcraft.videoTemplate";
/// The manifest format version this build reads and writes; newer packages are refused.
pub const FORMAT_VERSION: u32 = 1;
/// Folder extension of a template package (`My Template.fcvideo-template/`).
pub const EXTENSION: &str = "fcvideo-template";
/// The manifest's file name in a package.
pub const MANIFEST_FILE: &str = "template.json";
/// The default name of the sequence file in a package.
pub const SEQUENCE_FILE: &str = "sequence.json";
/// The folder of bundled assets in a package.
pub const ASSETS_DIR: &str = "assets/";
/// Item metadata naming the Elements sound (`sound.whoosh-short`) an item stands in for: in a
/// package the item is generated silence of the sound's length; the engine swaps in the
/// synthesized sound when the template is applied (and `save` turns it back into a stand-in).
pub const ELEMENT_METADATA: &str = "Element";

/// Whether `id` is a well-formed sound element id: `sound.` and 1–48 of `[a-z0-9-]`.
pub fn is_sound_element_id(id: &str) -> bool {
    id.strip_prefix("sound.").is_some_and(|r| !r.is_empty() && r.len() <= 48 && r.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'))
}

/// One problem found in a template, with the field it is about (`slots[2].fit.short`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Issue {
    pub field: String,
    pub message: String,
}

impl std::fmt::Display for Issue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.field.is_empty() { write!(f, "{}", self.message) } else { write!(f, "{}: {}", self.field, self.message) }
    }
}

/// Most issues one validation reports (a hostile file could otherwise produce millions).
pub const MAX_ISSUES: usize = 100;

/// Collected validation problems.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Issues(pub Vec<Issue>);

impl Issues {
    pub fn push(&mut self, field: impl Into<String>, message: impl Into<String>) {
        if self.0.len() < MAX_ISSUES {
            self.0.push(Issue { field: field.into(), message: message.into() });
        }
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn into_result(self) -> Result<(), TemplateError> {
        if self.0.is_empty() { Ok(()) } else { Err(TemplateError::Invalid(self.0)) }
    }
}

fn join(issues: &[Issue]) -> String {
    let mut s = String::new();
    for (i, x) in issues.iter().enumerate() {
        if i > 0 {
            s.push_str("; ");
        }
        s.push_str(&x.to_string());
    }
    s
}

/// Why a template cannot be read, validated or applied.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum TemplateError {
    #[error("not a FilmCraft video template: {0}")]
    NotTemplate(String),
    #[error("this video template was made by a newer FilmCraft (format version {0}; this build reads up to {FORMAT_VERSION})")]
    TooNew(u32),
    #[error("invalid video template: {}", join(.0))]
    Invalid(Vec<Issue>),
    #[error("template package: {0}")]
    Package(String),
    #[error("{0}")]
    Apply(String),
}

impl TemplateError {
    /// The problems with their fields (one for errors that are not about a field).
    pub fn issues(&self) -> Vec<Issue> {
        match self {
            TemplateError::Invalid(v) => v.clone(),
            other => vec![Issue { field: String::new(), message: other.to_string() }],
        }
    }
}

impl From<zip::ZipError> for TemplateError {
    fn from(e: zip::ZipError) -> Self {
        TemplateError::Package(e.to_string())
    }
}

/// Lower-case slug of a name (`Photo Beat Slideshow` → `photo-beat-slideshow`).
pub fn slug(name: &str) -> String {
    let mut s = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            s.push(c.to_ascii_lowercase());
        } else if !s.ends_with('-') && !s.is_empty() {
            s.push('-');
        }
        if s.len() >= 48 {
            break;
        }
    }
    let s = s.trim_end_matches('-').to_string();
    if s.is_empty() { "template".into() } else { s }
}

/// Lower-case hex SHA-256 of `data`.
pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::Digest;
    let d = sha2::Sha256::digest(data);
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod save_tests;
#[cfg(test)]
mod starter_tests;
#[cfg(test)]
mod tests;
