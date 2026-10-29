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

/// The segmenter's package (its own directory under the models directory): MobileSAM, which cuts a
/// person out of a picture given points on them (the cut-outs, `crate::segment`).
pub const SEGMENT_PACKAGE_ID: &str = "mobile-sam";
pub const SEGMENT_NAME: &str = "MobileSAM (people and objects cut out of a picture)";
pub const SEGMENT_LICENSE: &str = "Apache-2.0 (MobileSAM, Chaoning Zhang et al., Kyung Hee University; distilled from Meta's Segment Anything on 1% of SA-1B)";
pub const SEGMENT_SOURCE: &str = "https://github.com/ChaoningZhang/MobileSAM (weights/mobile_sam.pt)";

/// MobileSAM's weights: a TinyViT image encoder with Segment Anything's prompt encoder and mask
/// decoder, as the authors' repository holds them (a PyTorch state dict, read by candle's pickle
/// reader, which runs nothing). Pinned to commit f706ad9c; the SHA-256 checked 2026-10-28 by
/// `shasum -a 256` and against an independent mirror's LFS record.
pub const SEGMENTER: ModelFile = ModelFile {
    name: "mobile_sam.pt",
    url: "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/f706ad9c4eb7f219c00d9050e46328518ffb65d2/weights/mobile_sam.pt",
    sha256: "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
    size: 40_728_226,
};

pub static SEGMENT_FILES: &[ModelFile] = &[SEGMENTER];

pub fn segment_dir(models_dir: &Path) -> PathBuf {
    models_dir.join(SEGMENT_PACKAGE_ID)
}

/// The segmenter's file is present with its pinned size.
pub fn segmenter_installed(models_dir: &Path) -> bool {
    let d = segment_dir(models_dir);
    SEGMENT_FILES.iter().all(|f| present(&d, f))
}

/// The segmenter's identity, for cache keys.
pub fn segmenter_id() -> String {
    format!("{}={}", SEGMENTER.name, SEGMENTER.sha256.get(..16).unwrap_or(SEGMENTER.sha256))
}

/// The depth network's package (its own directory under the models directory): Depth Anything V2
/// Small, how near each part of a picture is (`crate::depth`: the 3D photo and depth blur).
pub const DEPTH_PACKAGE_ID: &str = "depth-anything-v2-small";
pub const DEPTH_NAME: &str = "Depth Anything V2 Small (how near each part of a picture is)";
pub const DEPTH_LICENSE: &str =
    "Apache-2.0 (Depth Anything V2 Small, Lihe Yang et al., HKU and TikTok; its Base, Large and Giant models are non-commercial and not used)";
pub const DEPTH_SOURCE: &str = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small (depth_anything_v2_vits.pth)";

/// Depth Anything V2 Small's weights: a DINOv2 ViT-S/14 encoder and a DPT head, as the authors
/// publish them (a PyTorch state dict, read by candle's pickle reader, which runs nothing). Trained
/// on 595K synthetic pictures with exact depth (Hypersim, Virtual KITTI 2, BlendedMVS, IRS,
/// TartanAir), then on 62M real pictures of public datasets labelled by a larger teacher model.
/// Pinned to revision 03876f86; the SHA-256 checked 2026-10-28 by `shasum -a 256` and against the
/// Hugging Face API's LFS record.
pub const DEPTH: ModelFile = ModelFile {
    name: "depth_anything_v2_vits.pth",
    url: "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/03876f8651c73a60fe4c2c48294e09fcb6838fcf/depth_anything_v2_vits.pth",
    sha256: "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378",
    size: 99_218_434,
};

pub static DEPTH_FILES: &[ModelFile] = &[DEPTH];

pub fn depth_dir(models_dir: &Path) -> PathBuf {
    models_dir.join(DEPTH_PACKAGE_ID)
}

/// The depth network's file is present with its pinned size.
pub fn depth_installed(models_dir: &Path) -> bool {
    let d = depth_dir(models_dir);
    DEPTH_FILES.iter().all(|f| present(&d, f))
}

/// The depth network's identity, for cache keys.
pub fn depth_id() -> String {
    format!("{}={}", DEPTH.name, DEPTH.sha256.get(..16).unwrap_or(DEPTH.sha256))
}

/// The upscaler's package (its own directory under the models directory): Real-ESRGAN's compact
/// general model, a picture made four times as large with its detail restored (`crate::upscale`).
pub const UPSCALE_PACKAGE_ID: &str = "real-esrgan";
pub const UPSCALE_NAME: &str = "Real-ESRGAN general x4v3 (pictures made larger and cleaner)";
pub const UPSCALE_LICENSE: &str = "BSD-3-Clause (Real-ESRGAN, Xintao Wang et al., Tencent ARC Lab)";
pub const UPSCALE_SOURCE: &str = "https://github.com/xinntao/Real-ESRGAN (release v0.2.5.0, realesr-general-x4v3.pth)";

/// Real-ESRGAN's `realesr-general-x4v3`: SRVGGNetCompact, 64 features, 32 convolutions with
/// PReLUs, ×4 (a PyTorch state dict under `params`, read by candle's pickle reader, which runs
/// nothing). Trained, as the project's training notes say, on DF2K (DIV2K, whose site gives it for
/// academic research, and Flickr2K) and OST with synthetic degradations. A release asset (GitHub
/// gives no digest for it): the size and the SHA-256 checked 2026-10-28 by `shasum -a 256`.
pub const UPSCALER: ModelFile = ModelFile {
    name: "realesr-general-x4v3.pth",
    url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    sha256: "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
    size: 4_885_111,
};

pub static UPSCALE_FILES: &[ModelFile] = &[UPSCALER];

pub fn upscale_dir(models_dir: &Path) -> PathBuf {
    models_dir.join(UPSCALE_PACKAGE_ID)
}

/// The upscaler's file is present with its pinned size.
pub fn upscaler_installed(models_dir: &Path) -> bool {
    let d = upscale_dir(models_dir);
    UPSCALE_FILES.iter().all(|f| present(&d, f))
}

/// The upscaler's identity, for cache keys and file names.
pub fn upscaler_id() -> String {
    format!("{}={}", UPSCALER.name, UPSCALER.sha256.get(..16).unwrap_or(UPSCALER.sha256))
}

/// The interpolator's package (its own directory): RIFE 4.26, the frames between two frames
/// (`crate::interp`).
pub const INTERP_PACKAGE_ID: &str = "rife-4.26";
pub const INTERP_NAME: &str = "RIFE 4.26 (frames made between frames)";
pub const INTERP_LICENSE: &str = "MIT (Practical-RIFE, Zhewei Huang et al.)";
pub const INTERP_SOURCE: &str = "https://huggingface.co/hzwer/RIFE (RIFEv4.26_0921.zip, the author's upload)";

/// RIFE 4.26 as its author publishes it: a zip of the network's PyTorch state dict
/// (`flownet.pkl`, read by candle's pickle reader, which runs nothing) and its Python files
/// (never run). The training data of the 4.x models is not published (the paper's model was
/// trained on Vimeo-90K). Pinned to revision 01fdc7e9; the SHA-256 checked 2026-10-28 by
/// `shasum -a 256` and against the Hugging Face API's LFS record.
pub const INTERPOLATOR: ModelFile = ModelFile {
    name: "RIFEv4.26_0921.zip",
    url: "https://huggingface.co/hzwer/RIFE/resolve/01fdc7e97404120c243c3ea7b427046e5dc7643e/RIFEv4.26_0921.zip",
    sha256: "1fa9b9cda3d9b8c3e301359e2595960902f97bf926c08598b0e9957a3f3f760e",
    size: 22_869_906,
};

pub static INTERP_FILES: &[ModelFile] = &[INTERPOLATOR];

/// The weights inside the zip, taken out beside it, and their size.
pub const INTERP_WEIGHTS: &str = "flownet.pkl";
pub const INTERP_WEIGHTS_SIZE: u64 = 24_636_301;

pub fn interp_dir(models_dir: &Path) -> PathBuf {
    models_dir.join(INTERP_PACKAGE_ID)
}

/// The interpolator's zip is present with its pinned size.
pub fn interpolator_installed(models_dir: &Path) -> bool {
    let d = interp_dir(models_dir);
    INTERP_FILES.iter().all(|f| present(&d, f))
}

/// The interpolator's identity, for cache keys and file names.
pub fn interpolator_id() -> String {
    format!("{}={}", INTERPOLATOR.name, INTERPOLATOR.sha256.get(..16).unwrap_or(INTERPOLATOR.sha256))
}

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
        for f in SEGMENT_FILES.iter().chain(DEPTH_FILES).chain(UPSCALE_FILES).chain(INTERP_FILES) {
            assert_eq!(f.sha256.len(), 64);
            assert!(f.sha256.bytes().all(|b| b.is_ascii_hexdigit()) && f.url.ends_with(f.name), "{}", f.url);
        }
        assert!(DEPTH.url.contains("/resolve/03876f8651c73a60fe4c2c48294e09fcb6838fcf/") && depth_id().starts_with("depth_anything_v2_vits.pth=715fade1"));
    }

    #[test]
    fn install_state_follows_the_files() {
        let dir = std::env::temp_dir().join(format!("filmcraft-vision-catalog-{}", std::process::id()));
        assert!(!installed(&dir));
        assert_eq!(missing_bytes(&dir), total_bytes());
        assert!(!profiles_installed(&dir));
        assert_eq!(profile_missing_bytes(&dir), profile_total_bytes());
        assert!(!depth_installed(&dir));
        assert!(!upscaler_installed(&dir));
        assert!(!interpolator_installed(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
