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

/// The profile pass's models (faces in profile and half hidden, which YuNet misses): a person
/// detector and a pose network, in the same package directory, downloaded with the detectors.
pub const PROFILE_NAME: &str = "Person and pose models for faces in profile (D-FINE-N, RTMPose-s)";
pub const PROFILE_LICENSE: &str = "Apache-2.0 (D-FINE, Peng et al., USTC; RTMPose, OpenMMLab)";
pub const PROFILE_SOURCE: &str = "https://huggingface.co/ustc-community/dfine-nano-coco (D-FINE's Transformers port), https://huggingface.co/Tau-J/RTMPose (RTMPose's author's mirror of download.openmmlab.com)";

/// D-FINE-N (COCO): the person detector's weights (Apache-2.0).
pub const PERSON_WEIGHTS: ModelFile = ModelFile {
    name: "dfine_n_coco.safetensors",
    url: "https://huggingface.co/ustc-community/dfine-nano-coco/resolve/066438d3d8f0da137a37b38fdf3368fd4afceced/model.safetensors",
    sha256: "19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b",
    size: 15_278_996,
};

/// D-FINE-N's configuration (the network's sizes).
pub const PERSON_CONFIG: ModelFile = ModelFile {
    name: "dfine_n_coco.json",
    url: "https://huggingface.co/ustc-community/dfine-nano-coco/resolve/066438d3d8f0da137a37b38fdf3368fd4afceced/config.json",
    sha256: "7edf2d96d3deb10aa4ec920be0e8e444b617454545bb58093e32bd8e2b96fa2f",
    size: 6_557,
};

/// RTMPose-s (body7, 256 × 192): OpenMMLab's deployment zip holding `end2end.onnx` (Apache-2.0);
/// the same bytes as on download.openmmlab.com.
pub const POSE: ModelFile = ModelFile {
    name: "rtmpose-s_simcc-body7_256x192.zip",
    url: "https://huggingface.co/Tau-J/RTMPose/resolve/cd4d7095f5cfc9cfc4f46289bee91ea4a1e1d9fd/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
    sha256: "7673922e531014906ca4f0f239b7e233b740146a10b632deaa2a28d45470d802",
    size: 20_496_303,
};

pub static PROFILE_FILES: &[ModelFile] = &[PERSON_WEIGHTS, PERSON_CONFIG, POSE];

pub fn package_dir(models_dir: &Path) -> PathBuf {
    models_dir.join(PACKAGE_ID)
}

fn present(dir: &Path, f: &ModelFile) -> bool {
    std::fs::metadata(dir.join(f.name)).is_ok_and(|m| m.len() == f.size)
}

/// The profile pass's files are present with their pinned sizes.
pub fn profiles_installed(models_dir: &Path) -> bool {
    let d = package_dir(models_dir);
    PROFILE_FILES.iter().all(|f| present(&d, f))
}

/// Bytes of the profile pass's files still to download.
pub fn profile_missing_bytes(models_dir: &Path) -> u64 {
    let d = package_dir(models_dir);
    PROFILE_FILES.iter().filter(|f| !present(&d, f)).map(|f| f.size).sum()
}

/// Total size of the profile pass's files.
pub fn profile_total_bytes() -> u64 {
    PROFILE_FILES.iter().map(|f| f.size).sum()
}

/// The profile models' identity (their names and pinned SHA-256s), for cache keys: another model
/// is another key.
pub fn profile_model_ids() -> String {
    PROFILE_FILES.iter().map(|f| format!("{}={}", f.name, f.sha256.get(..16).unwrap_or(f.sha256))).collect::<Vec<_>>().join(",")
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
        for f in PROFILE_FILES {
            assert!(f.url.starts_with("https://huggingface.co/") && f.url.contains("/resolve/"), "{}", f.url);
            let rev = f.url.split("/resolve/").nth(1).and_then(|r| r.split('/').next()).unwrap();
            assert_eq!(rev.len(), 40);
            assert!(rev.bytes().all(|b| b.is_ascii_hexdigit()));
            assert_eq!(f.sha256.len(), 64);
            assert!(f.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
        }
        assert!(profile_total_bytes() < 64_000_000);
        // names distinct from the detectors' (one directory)
        let mut names: Vec<&str> = FILES.iter().chain(PROFILE_FILES).map(|f| f.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), FILES.len() + PROFILE_FILES.len());
        assert!(profile_model_ids().contains("dfine_n_coco.safetensors=19e06bdc873da819"));
    }

    #[test]
    fn install_state_follows_the_files() {
        let dir = std::env::temp_dir().join(format!("filmcraft-vision-catalog-{}", std::process::id()));
        assert!(!installed(&dir));
        assert_eq!(missing_bytes(&dir), total_bytes());
        assert!(!profiles_installed(&dir));
        assert_eq!(profile_missing_bytes(&dir), profile_total_bytes());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
