//! The downloadable voice model. The file is pinned to a commit of the DeepFilterNet repository
//! and checked against its SHA-256 when downloaded (by `filmcraft_speech::models::download_files`,
//! run by the engine's `audio.downloadVoiceModel` after the user confirms). Nothing here is
//! bundled with FilmCraft or committed to this repository.
//!
//! Licence: DeepFilterNet (code and the released models in its `models/` directory) is dual
//! licensed MIT or Apache-2.0, © 2021 Hendrik Schröter
//! (<https://github.com/Rikorose/DeepFilterNet/blob/main/LICENSE>). Pin checked 2026-10-22 with
//! `shasum -a 256` against the file at the pinned commit.

use std::path::{Path, PathBuf};

use serde::Serialize;

/// One model file.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelFile {
    /// Local file name inside the package directory.
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

/// Directory name of the package under the models directory.
pub const PACKAGE_ID: &str = "deepfilternet3";
pub const PACKAGE_NAME: &str = "Voice isolation model (DeepFilterNet 3)";
pub const LICENSE: &str = "MIT OR Apache-2.0 (DeepFilterNet, © 2021 Hendrik Schröter)";
pub const SOURCE: &str = "https://github.com/Rikorose/DeepFilterNet (models/DeepFilterNet3_onnx.tar.gz)";
/// The commit the file is pinned to.
pub const REVISION: &str = "d375b2d8309e0935d165700c91da9de862a99c31";

/// The model archive: the encoder and both decoders as ONNX, and the training config.
pub const ARCHIVE: ModelFile = ModelFile {
    name: "DeepFilterNet3_onnx.tar.gz",
    url: "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/d375b2d8309e0935d165700c91da9de862a99c31/models/DeepFilterNet3_onnx.tar.gz",
    sha256: "c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616",
    size: 7_983_136,
};

pub static FILES: &[ModelFile] = &[ARCHIVE];

pub fn package_dir(models_dir: &Path) -> PathBuf {
    models_dir.join(PACKAGE_ID)
}

/// Where the archive lives once downloaded.
pub fn archive_path(models_dir: &Path) -> PathBuf {
    package_dir(models_dir).join(ARCHIVE.name)
}

fn present(dir: &Path, f: &ModelFile) -> bool {
    std::fs::metadata(dir.join(f.name)).is_ok_and(|m| m.len() == f.size)
}

/// The archive is present with its pinned size (its contents were verified when downloaded).
pub fn installed(models_dir: &Path) -> bool {
    let d = package_dir(models_dir);
    FILES.iter().all(|f| present(&d, f))
}

/// Bytes still to download.
pub fn missing_bytes(models_dir: &Path) -> u64 {
    let d = package_dir(models_dir);
    FILES.iter().filter(|f| !present(&d, f)).map(|f| f.size).sum()
}

/// Total size of the package.
pub fn total_bytes() -> u64 {
    FILES.iter().map(|f| f.size).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_archive_is_pinned_and_checksummed() {
        for f in FILES {
            let prefix = format!("https://raw.githubusercontent.com/Rikorose/DeepFilterNet/{REVISION}/models/");
            assert!(f.url.starts_with(&prefix) && f.url.ends_with(f.name), "{}", f.url);
            // a 40-hex commit, not a branch name
            assert_eq!(REVISION.len(), 40);
            assert!(REVISION.bytes().all(|b| b.is_ascii_hexdigit()));
            assert_eq!(f.sha256.len(), 64);
            assert!(f.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
        }
        assert!(total_bytes() < 10_000_000);
        assert!(LICENSE.contains("MIT") && LICENSE.contains("Apache-2.0"));
    }

    #[test]
    fn install_state_follows_the_file() {
        let dir = std::env::temp_dir().join(format!("filmcraft-voice-catalog-{}", std::process::id()));
        assert!(!installed(&dir));
        assert_eq!(missing_bytes(&dir), total_bytes());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
