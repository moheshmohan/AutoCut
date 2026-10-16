//! The downloadable neural voices: Kokoro-82M, its voice packs and the CMUdict pronunciation
//! dictionary. Every file is pinned to a revision and checked against its SHA-256 when downloaded
//! (the downloader is `filmcraft_speech::models::download_files`, run by the engine after the
//! user confirms). Nothing here is bundled with FilmCraft or committed to the repository.
//!
//! Licences: Kokoro-82M weights and voices by hexgrad, Apache-2.0 (declared on the model card; the
//! repository has no separate LICENSE file). CMUdict, Carnegie Mellon University, BSD-style.
//! Pins verified 2026-10-07 against the Hugging Face API and by `sha256sum`.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::{Gender, VoiceInfo};

/// One file of the neural voice package.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelFile {
    /// Local file name inside the package directory.
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

/// Directory name of the package under the models directory.
pub const PACKAGE_ID: &str = "kokoro-82m";
pub const PACKAGE_NAME: &str = "Natural voices (Kokoro-82M)";
pub const LICENSE: &str = "Apache-2.0 (Kokoro-82M by hexgrad); CMUdict: BSD-style (Carnegie Mellon University)";
pub const SOURCE: &str = "https://huggingface.co/hexgrad/Kokoro-82M";

macro_rules! hf {
    ($name:literal, $path:literal, $sha:literal, $size:expr) => {
        ModelFile {
            name: $name,
            url: concat!("https://huggingface.co/hexgrad/Kokoro-82M/resolve/f3ff3571791e39611d31c381e3a41a3af07b4987/", $path),
            sha256: $sha,
            size: $size,
        }
    };
}

pub static FILES: &[ModelFile] = &[
    hf!("config.json", "config.json", "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f", 2351),
    hf!("kokoro-v1_0.pth", "kokoro-v1_0.pth", "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4", 327_212_226),
    ModelFile {
        name: "cmudict.dict",
        url: "https://raw.githubusercontent.com/cmusphinx/cmudict/74790861f652b15e4ac49015a90074ad62a27690/cmudict.dict",
        sha256: "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22",
        size: 3_618_488,
    },
    hf!("af_heart.pt", "voices/af_heart.pt", "0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff", 523_425),
    hf!("af_bella.pt", "voices/af_bella.pt", "8cb64e02fcc8de0327a8e13817e49c76c945ecf0052ceac97d3081480e8e48d6", 523_425),
    hf!("af_nicole.pt", "voices/af_nicole.pt", "c5561808bcf5250fe8c5f5de32caf2d94f27e57e95befdb098c5c85991d4c5da", 523_430),
    hf!("af_aoede.pt", "voices/af_aoede.pt", "c03bd1a4c3716c2d8eaa3d50022f62d5c31cfbd6e15933a00b17fefe13841cc4", 523_425),
    hf!("af_kore.pt", "voices/af_kore.pt", "8bfbc512321c3db49dff984ac675fa5ac7eaed5a96cc31104d3a9080e179d69d", 523_420),
    hf!("af_sarah.pt", "voices/af_sarah.pt", "49bd364ea3be9eb3e9685e8f9a15448c4883112a7c0ff7ab139fa4088b08cef9", 523_425),
    hf!("am_michael.pt", "voices/am_michael.pt", "9a443b79a4b22489a5b0ab7c651a0bcd1a30bef675c28333f06971abbd47bd37", 523_435),
    hf!("am_fenrir.pt", "voices/am_fenrir.pt", "98e507eca1db08230ae3b6232d59c10aec9630022d19accac4f5d12fcec3c37a", 523_430),
    hf!("am_puck.pt", "voices/am_puck.pt", "dd1d8973f4ce4b7d8ae407c77a435f485dabc052081b80ea75c4f30b84f36223", 523_420),
];

/// The voice-cloning package: the OpenVoice V2 tone-colour converter (MIT, MyShell). Pinned and
/// checked like the voices (verified 2026-10-16 against the Hugging Face API and `shasum -a 256`).
pub const CLONE_PACKAGE_ID: &str = "openvoice-v2";
pub const CLONE_PACKAGE_NAME: &str = "Voice cloning (OpenVoice V2 converter)";
pub const CLONE_LICENSE: &str = "MIT (OpenVoice V2 by MyShell)";
pub const CLONE_SOURCE: &str = "https://huggingface.co/myshell-ai/OpenVoiceV2";

pub static CLONE_FILES: &[ModelFile] = &[
    ModelFile {
        name: "config.json",
        url: "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/f36e7edfe1684461a8343844af60babc2efbb727/converter/config.json",
        sha256: "9dfff60350b8c63f2c664efd92a61b2516efb22671466960f0e5dfebd881fa47",
        size: 838,
    },
    ModelFile {
        name: "checkpoint.pth",
        url: "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/f36e7edfe1684461a8343844af60babc2efbb727/converter/checkpoint.pth",
        sha256: "9652c27e92b6b2a91632590ac9962ef7ae2b712e5c5b7f4c34ec55ee2b37ab9e",
        size: 131_320_490,
    },
];

pub fn clone_package_dir(models_dir: &Path) -> PathBuf {
    models_dir.join(CLONE_PACKAGE_ID)
}

/// The converter's files are all present with their pinned sizes.
pub fn clone_installed(models_dir: &Path) -> bool {
    let d = clone_package_dir(models_dir);
    CLONE_FILES.iter().all(|f| std::fs::metadata(d.join(f.name)).is_ok_and(|m| m.len() == f.size))
}

/// Bytes of the cloning package still to download.
pub fn clone_missing_bytes(models_dir: &Path) -> u64 {
    let d = clone_package_dir(models_dir);
    CLONE_FILES.iter().filter(|f| !std::fs::metadata(d.join(f.name)).is_ok_and(|m| m.len() == f.size)).map(|f| f.size).sum()
}

/// A VoxCPM release for natural voice cloning (feature `voxcpm`): the model speaks in the
/// cloned voice directly, accent and rhythm included. Pinned and checked like the voices
/// (verified 2026-10-16 against the Hugging Face API and `shasum -a 256`). Code and weights are
/// Apache-2.0 (OpenBMB, "free for commercial use"); nothing NC/ND.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoxcpmPackage {
    /// Model id in profiles and commands, and the package's directory name.
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub license: &'static str,
    pub source: &'static str,
    pub files: &'static [ModelFile],
}

pub const VOXCPM_LICENSE: &str = "Apache-2.0 (VoxCPM by OpenBMB)";

macro_rules! voxcpm_file {
    ($repo:literal, $rev:literal, $name:literal, $sha:literal, $size:expr) => {
        ModelFile { name: $name, url: concat!("https://huggingface.co/openbmb/", $repo, "/resolve/", $rev, "/", $name), sha256: $sha, size: $size }
    };
}

/// The VoxCPM releases, fastest first.
pub static VOXCPM: &[VoxcpmPackage] = &[
    VoxcpmPackage {
        id: "voxcpm-1.5",
        name: "Natural cloning, faster (VoxCPM1.5)",
        description: "0.8 B parameters, 44.1 kHz, about 2.5× faster than VoxCPM2. Continues the voice sample, so it needs a clean recording and the sample's exact transcript: a noisy or mis-transcribed sample derails it into babble.",
        license: VOXCPM_LICENSE,
        source: "https://huggingface.co/openbmb/VoxCPM1.5",
        files: &[
            voxcpm_file!(
                "VoxCPM1.5",
                "8cdc403854cda0e3af12252d27da038fda5982ac",
                "config.json",
                "2459071497a422bf25fb0f47ff8bd30181dbf62ef7903b293bcff59a615eda26",
                2831
            ),
            voxcpm_file!(
                "VoxCPM1.5",
                "8cdc403854cda0e3af12252d27da038fda5982ac",
                "tokenizer.json",
                "63443666e076f48a575030d4861e5bdbf64b71632b48e75d4e867c46dd980aa5",
                3_676_778
            ),
            voxcpm_file!(
                "VoxCPM1.5",
                "8cdc403854cda0e3af12252d27da038fda5982ac",
                "audiovae.pth",
                "8160987a7f10acd4c49b72a9a3e974c0b5c1c786538cd4bcaac5f2df388c0f14",
                346_023_712
            ),
            voxcpm_file!(
                "VoxCPM1.5",
                "8cdc403854cda0e3af12252d27da038fda5982ac",
                "model.safetensors",
                "7e97ee2cd2ca891662f7da368921df08368a616410afe67461253e4dcd3823cc",
                1_603_516_600
            ),
        ],
    },
    VoxcpmPackage {
        id: "voxcpm-2",
        name: "Natural cloning, best quality (VoxCPM2)",
        description: "2.3 B parameters, 48 kHz, 30 languages. Clones from the voice sample alone, so noisy phone recordings and transcription errors do not matter. About 5 GB of memory on a GPU (10 GB on the CPU).",
        license: VOXCPM_LICENSE,
        source: "https://huggingface.co/openbmb/VoxCPM2",
        files: &[
            voxcpm_file!(
                "VoxCPM2",
                "32279effe8c19989596f05d353d1447f51d9e915",
                "config.json",
                "405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c",
                4336
            ),
            voxcpm_file!(
                "VoxCPM2",
                "32279effe8c19989596f05d353d1447f51d9e915",
                "tokenizer.json",
                "f8984687e4a92a3503d521396d454b7d68e9fdaab2a0288eb3536c7c1aa4bc20",
                3_676_772
            ),
            voxcpm_file!(
                "VoxCPM2",
                "32279effe8c19989596f05d353d1447f51d9e915",
                "audiovae.pth",
                "94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1",
                376_951_122
            ),
            voxcpm_file!(
                "VoxCPM2",
                "32279effe8c19989596f05d353d1447f51d9e915",
                "model.safetensors",
                "f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d",
                4_580_080_592
            ),
        ],
    },
];

/// The VoxCPM release new natural clones use unless another is chosen: VoxCPM2, whose isolated
/// reference prompt held up on a real phone recording (Whisper WER 4.8 %, speaker similarity
/// 0.56 against 0.15 for the timbre-only clone) where VoxCPM1.5's continuation babbled (see the
/// crate README); it still runs faster than real time on an Apple GPU.
pub const DEFAULT_VOXCPM: &str = "voxcpm-2";

pub fn voxcpm(id: &str) -> Option<&'static VoxcpmPackage> {
    VOXCPM.iter().find(|p| p.id == id)
}

pub fn voxcpm_dir(models_dir: &Path, p: &VoxcpmPackage) -> PathBuf {
    models_dir.join(p.id)
}

/// Every file of the release present with its pinned size.
pub fn voxcpm_installed(models_dir: &Path, p: &VoxcpmPackage) -> bool {
    let d = voxcpm_dir(models_dir, p);
    p.files.iter().all(|f| std::fs::metadata(d.join(f.name)).is_ok_and(|m| m.len() == f.size))
}

/// Bytes of the release still to download.
pub fn voxcpm_missing_bytes(models_dir: &Path, p: &VoxcpmPackage) -> u64 {
    let d = voxcpm_dir(models_dir, p);
    p.files.iter().filter(|f| !std::fs::metadata(d.join(f.name)).is_ok_and(|m| m.len() == f.size)).map(|f| f.size).sum()
}

/// Total download size of a release.
pub fn voxcpm_size(p: &VoxcpmPackage) -> u64 {
    p.files.iter().map(|f| f.size).sum()
}

/// A neural voice: catalogue entry and its voice-pack file.
#[derive(Clone, Copy, Debug)]
pub struct NeuralVoice {
    pub info: VoiceInfo,
    pub file: &'static str,
}

const NEURAL_LICENSE: &str = "Apache-2.0 (Kokoro-82M by hexgrad)";

macro_rules! voice {
    ($id:literal, $name:literal, $gender:ident, $file:literal, $desc:literal) => {
        NeuralVoice {
            info: VoiceInfo {
                id: $id,
                name: $name,
                language: "en-US",
                gender: Gender::$gender,
                engine: "neural",
                description: $desc,
                license: NEURAL_LICENSE,
                installed: false,
            },
            file: $file,
        }
    };
}

/// US English voices the model card grades C+ or better, best first.
pub static VOICES: &[NeuralVoice] = &[
    voice!("kokoro-heart", "Heart", Female, "af_heart.pt", "Natural voice, warm (model card grade A)."),
    voice!("kokoro-bella", "Bella", Female, "af_bella.pt", "Natural voice, bright (grade A−)."),
    voice!("kokoro-nicole", "Nicole", Female, "af_nicole.pt", "Natural voice, soft (grade B−)."),
    voice!("kokoro-michael", "Michael", Male, "am_michael.pt", "Natural voice (grade C+)."),
    voice!("kokoro-fenrir", "Fenrir", Male, "am_fenrir.pt", "Natural voice, deep (grade C+)."),
    voice!("kokoro-puck", "Puck", Male, "am_puck.pt", "Natural voice, lively (grade C+)."),
    voice!("kokoro-aoede", "Aoede", Female, "af_aoede.pt", "Natural voice (grade C+)."),
    voice!("kokoro-kore", "Kore", Female, "af_kore.pt", "Natural voice (grade C+)."),
    voice!("kokoro-sarah", "Sarah", Female, "af_sarah.pt", "Natural voice (grade C+)."),
];

pub fn find(id: &str) -> Option<&'static NeuralVoice> {
    VOICES.iter().find(|v| v.info.id == id)
}

/// Total download size.
pub fn size() -> u64 {
    FILES.iter().map(|f| f.size).sum()
}

pub fn package_dir(models_dir: &Path) -> PathBuf {
    models_dir.join(PACKAGE_ID)
}

/// Every file present with its pinned size (contents were verified when downloaded).
pub fn installed(models_dir: &Path) -> bool {
    let d = package_dir(models_dir);
    FILES.iter().all(|f| std::fs::metadata(d.join(f.name)).is_ok_and(|m| m.len() == f.size))
}

/// Bytes still to download.
pub fn missing_bytes(models_dir: &Path) -> u64 {
    let d = package_dir(models_dir);
    FILES.iter().filter(|f| !std::fs::metadata(d.join(f.name)).is_ok_and(|m| m.len() == f.size)).map(|f| f.size).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_are_pinned_and_voices_resolve() {
        for f in FILES {
            assert_eq!(f.sha256.len(), 64, "{}", f.name);
            assert!(f.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
            assert!(
                f.url.starts_with("https://")
                    && (f.url.contains("/f3ff3571791e39611d31c381e3a41a3af07b4987/") || f.url.contains("/74790861f652b15e4ac49015a90074ad62a27690/")),
                "{}",
                f.url
            );
            assert!(!f.name.contains('/'), "flat package layout: {}", f.name);
        }
        for v in VOICES {
            assert!(FILES.iter().any(|f| f.name == v.file), "{} has no file", v.info.id);
            assert_eq!(find(v.info.id).map(|x| x.file), Some(v.file));
        }
        assert!(size() > 330_000_000 && size() < 340_000_000, "{}", size());
        // the VoxCPM releases: the files the loader reads, pinned to a revision, Apache-2.0
        assert!(voxcpm(DEFAULT_VOXCPM).is_some());
        for p in VOXCPM {
            let mut names: Vec<&str> = p.files.iter().map(|f| f.name).collect();
            names.sort_unstable();
            assert_eq!(names, vec!["audiovae.pth", "config.json", "model.safetensors", "tokenizer.json"], "{}", p.id);
            for f in p.files {
                assert_eq!(f.sha256.len(), 64, "{}", f.name);
                assert!(f.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
                assert!(f.url.starts_with("https://huggingface.co/openbmb/") && f.url.contains("/resolve/") && f.url.ends_with(f.name), "{}", f.url);
                let rev = f.url.split("/resolve/").nth(1).and_then(|r| r.split('/').next()).unwrap_or("");
                assert_eq!(rev.len(), 40, "pinned to a commit: {}", f.url);
            }
            assert!(p.license.starts_with("Apache-2.0") && !p.license.contains("NC"));
            assert_eq!(voxcpm(p.id).map(|x| x.id), Some(p.id));
        }
        assert!(voxcpm_size(voxcpm("voxcpm-1.5").unwrap()) > 1_900_000_000);
        assert!(voxcpm_size(voxcpm("voxcpm-2").unwrap()) > 4_900_000_000);
        let dir = std::env::temp_dir().join(format!("filmcraft-tts-cat-{}", std::process::id()));
        assert!(!installed(&dir));
        assert_eq!(missing_bytes(&dir), size());
        for p in VOXCPM {
            assert!(!voxcpm_installed(&dir, p));
            assert_eq!(voxcpm_missing_bytes(&dir, p), voxcpm_size(p));
        }
    }
}
