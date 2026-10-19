//! The downloadable detector models. Every file is pinned to a revision and checked against its
//! SHA-256 when downloaded (by `filmcraft_speech::models::download_files`, run by the engine's
//! `redact.downloadModels` after the user confirms). Nothing here is bundled with FilmCraft or
//! committed to the repository.
//!
//! Licences (read from each model's directory in OpenCV Zoo, mirrored on Hugging Face by OpenCV):
//! - YuNet face detector (`face_detection_yunet_2023mar.onnx`): MIT, © 2020 Shiqi Yu
//!   (libfacedetection.train).
//! - LPD-YuNet licence-plate detector (`license_plate_detection_lpd_yunet_2023mar.onnx`):
//!   Apache-2.0, contributed to OpenCV Zoo by Dong Xu (watrix.ai). Trained on Chinese plates, so
//!   it finds other countries' plates less reliably.
//!
//! Pins verified 2026-10-19 against the Hugging Face API (LFS `oid`) and `shasum -a 256`.

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
pub const PACKAGE_ID: &str = "privacy-detectors";
pub const PACKAGE_NAME: &str = "Face and plate detectors (YuNet, LPD-YuNet)";
pub const LICENSE: &str = "MIT (YuNet face detector, Shiqi Yu); Apache-2.0 (LPD-YuNet plate detector, OpenCV Zoo)";
pub const SOURCE: &str = "https://github.com/opencv/opencv_zoo (models/face_detection_yunet, models/license_plate_detection_yunet)";

/// The face detector.
pub const FACE: ModelFile = ModelFile {
    name: "face_detection_yunet_2023mar.onnx",
    url: "https://huggingface.co/opencv/face_detection_yunet/resolve/3cc26e7f1014a5ee5d74a42acee58bafc9d0a310/face_detection_yunet_2023mar.onnx",
    sha256: "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    size: 232_589,
};

/// The plate detector.
pub const PLATE: ModelFile = ModelFile {
    name: "license_plate_detection_lpd_yunet_2023mar.onnx",
    url: "https://huggingface.co/opencv/license_plate_detection_yunet/resolve/962e8d8849878106cb84039dab61cf97bb683b31/license_plate_detection_lpd_yunet_2023mar.onnx",
    sha256: "6d4978a7b6d25514d5e24811b82bfb511d166bdd8ca3b03aa63c1623d4d039c7",
    size: 4_146_213,
};

pub static FILES: &[ModelFile] = &[FACE, PLATE];

pub fn package_dir(models_dir: &Path) -> PathBuf {
    models_dir.join(PACKAGE_ID)
}

fn present(dir: &Path, f: &ModelFile) -> bool {
    std::fs::metadata(dir.join(f.name)).is_ok_and(|m| m.len() == f.size)
}

/// Both files are present with their pinned sizes (contents were verified when downloaded).
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
    fn files_are_pinned_and_checksummed() {
        for f in FILES {
            assert!(f.url.starts_with("https://huggingface.co/opencv/") && f.url.contains("/resolve/") && f.url.ends_with(f.name), "{}", f.url);
            // a 40-hex revision, not a branch name
            let rev = f.url.split("/resolve/").nth(1).and_then(|r| r.split('/').next()).unwrap();
            assert_eq!(rev.len(), 40);
            assert!(rev.bytes().all(|b| b.is_ascii_hexdigit()));
            assert_eq!(f.sha256.len(), 64);
            assert!(f.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
        }
        assert!(total_bytes() < 5_000_000);
    }

    #[test]
    fn install_state_follows_the_files() {
        let dir = std::env::temp_dir().join(format!("filmcraft-vision-catalog-{}", std::process::id()));
        assert!(!installed(&dir));
        assert_eq!(missing_bytes(&dir), total_bytes());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
