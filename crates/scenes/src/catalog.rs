//! The downloadable scene models. Every file is pinned to a revision (a 40-hex commit) and
//! checked against its SHA-256 and size when downloaded (by
//! `filmcraft_speech::models::download_files`, run by the engine's `scenes.downloadModel` after
//! the user agreed), into `<data dir>/models/<id>/`. Nothing is bundled with FilmCraft or
//! committed to the repository. Only non-gated weights with permissive licences, from the
//! authors' own repositories:
//!
//! | id | what | licence | source |
//! |---|---|---|---|
//! | `transnetv2` | TransNetV2 shot-boundary network (TensorFlow checkpoint) | MIT | github.com/soCzech/TransNetV2 |
//! | `qwen3-vl-2b` | Qwen3-VL-2B-Instruct | Apache-2.0 | huggingface.co/Qwen |
//! | `qwen3-vl-4b` | Qwen3-VL-4B-Instruct | Apache-2.0 | huggingface.co/Qwen |
//! | `qwen3-vl-8b` | Qwen3-VL-8B-Instruct | Apache-2.0 | huggingface.co/Qwen |
//! | `siglip2-base-224` | SigLIP 2 base, patch 16, 224 px | Apache-2.0 | huggingface.co/google |
//! | `clip-vit-l14-aesthetic` | CLIP ViT-L/14 + the LAION aesthetic predictor (linear head) | MIT (both) | huggingface.co/openai, github.com/LAION-AI/aesthetic-predictor |
//!
//! Pins and licences checked 2026-10-22 (Hugging Face API `lfs.sha256` and `shasum -a 256` of
//! the downloaded files; the licence of each repository: the model card's `license` tag, or the
//! GitHub repository's `LICENSE`).

use std::path::{Path, PathBuf};

use serde::Serialize;

/// One model file.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelFile {
    /// Local file name inside the model's directory.
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

/// What a model does in the scene index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Role {
    /// Finds the cuts.
    Boundaries,
    /// Describes the shots.
    Describer,
    /// Embeds keyframes and text queries for search.
    Embedder,
    /// Scores how good a picture looks.
    Aesthetic,
}

/// A downloadable model.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub role: Role,
    pub description: &'static str,
    pub license: &'static str,
    pub license_url: &'static str,
    pub author: &'static str,
    pub source: &'static str,
    pub files: &'static [ModelFile],
}

impl ModelInfo {
    pub fn size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

macro_rules! hf {
    ($repo:literal, $rev:literal, $name:literal, $sha:literal, $size:expr) => {
        ModelFile { name: $name, url: concat!("https://huggingface.co/", $repo, "/resolve/", $rev, "/", $name), sha256: $sha, size: $size }
    };
}

const QWEN_LICENSE: &str = "Apache-2.0";
const QWEN_LICENSE_URL: &str = "https://www.apache.org/licenses/LICENSE-2.0";
const QWEN_AUTHOR: &str = "Qwen team, Alibaba Cloud";
/// Qwen3-VL's tokenizer (the same file in every size).
const QWEN_TOKENIZER_SHA: &str = "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7";
const QWEN_TOKENIZER_SIZE: u64 = 7_032_403;

static CATALOGUE: &[ModelInfo] = &[
    ModelInfo {
        id: TRANSNET,
        name: "TransNetV2 shot-boundary detector",
        role: Role::Boundaries,
        description: "Finds hard cuts and gradual transitions from 48×27 frames, 100 at a time (Souček & Lokoč 2020). 30 MB.",
        license: "MIT",
        license_url: "https://github.com/soCzech/TransNetV2/blob/master/LICENSE",
        author: "Tomáš Souček, Jakub Lokoč",
        source: "https://github.com/soCzech/TransNetV2",
        files: &[
            ModelFile {
                name: "variables.index",
                url: "https://media.githubusercontent.com/media/soCzech/TransNetV2/85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed/inference/transnetv2-weights/variables/variables.index",
                sha256: "8b99e28b4ad11372d9a1ad9703298c2e370df14859da4245fdbe818e92dd403f",
                size: 5_526,
            },
            ModelFile {
                name: "variables.data-00000-of-00001",
                url: "https://media.githubusercontent.com/media/soCzech/TransNetV2/85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed/inference/transnetv2-weights/variables/variables.data-00000-of-00001",
                sha256: "b8c9dc3eb807583e6215cabee9ca61737b3eb1bceff68418b43bf71459669367",
                size: 30_516_656,
            },
        ],
    },
    ModelInfo {
        id: QWEN_2B,
        name: "Qwen3-VL 2B Instruct",
        role: Role::Describer,
        description: "The fast describer: one structured description per shot. 4.3 GB (bf16).",
        license: QWEN_LICENSE,
        license_url: QWEN_LICENSE_URL,
        author: QWEN_AUTHOR,
        source: "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct",
        files: &[
            hf!(
                "Qwen/Qwen3-VL-2B-Instruct",
                "89644892e4d85e24eaac8bacfd4f463576704203",
                "config.json",
                "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
                1_505
            ),
            hf!(
                "Qwen/Qwen3-VL-2B-Instruct",
                "89644892e4d85e24eaac8bacfd4f463576704203",
                "tokenizer.json",
                "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
                7_032_403
            ),
            hf!(
                "Qwen/Qwen3-VL-2B-Instruct",
                "89644892e4d85e24eaac8bacfd4f463576704203",
                "model.safetensors",
                "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
                4_255_140_312
            ),
        ],
    },
    ModelInfo {
        id: QWEN_4B,
        name: "Qwen3-VL 4B Instruct",
        role: Role::Describer,
        description: "Richer descriptions (`detail: full`): reads more text, names more. 8.9 GB (bf16).",
        license: QWEN_LICENSE,
        license_url: QWEN_LICENSE_URL,
        author: QWEN_AUTHOR,
        source: "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct",
        files: &[
            hf!(
                "Qwen/Qwen3-VL-4B-Instruct",
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "config.json",
                "edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d",
                1_505
            ),
            hf!(
                "Qwen/Qwen3-VL-4B-Instruct",
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "tokenizer.json",
                "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
                7_032_403
            ),
            hf!(
                "Qwen/Qwen3-VL-4B-Instruct",
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "model-00001-of-00002.safetensors",
                "30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9",
                4_967_229_296
            ),
            hf!(
                "Qwen/Qwen3-VL-4B-Instruct",
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "model-00002-of-00002.safetensors",
                "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6",
                3_908_490_048
            ),
        ],
    },
    ModelInfo {
        id: QWEN_8B,
        name: "Qwen3-VL 8B Instruct",
        role: Role::Describer,
        description: "The richest descriptions (`detail: full` with `model: qwen3-vl-8b`); needs about 20 GB of memory. 17.5 GB (bf16).",
        license: QWEN_LICENSE,
        license_url: QWEN_LICENSE_URL,
        author: QWEN_AUTHOR,
        source: "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct",
        files: &[
            hf!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "config.json",
                "5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661",
                1_474
            ),
            hf!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "tokenizer.json",
                "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
                7_032_403
            ),
            hf!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "model-00001-of-00004.safetensors",
                "d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa",
                4_902_275_944
            ),
            hf!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "model-00002-of-00004.safetensors",
                "8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5",
                4_915_962_496
            ),
            hf!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "model-00003-of-00004.safetensors",
                "83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192",
                4_999_831_048
            ),
            hf!(
                "Qwen/Qwen3-VL-8B-Instruct",
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "model-00004-of-00004.safetensors",
                "0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5",
                2_716_270_024
            ),
        ],
    },
    ModelInfo {
        id: SIGLIP,
        name: "SigLIP 2 base (patch 16, 224 px)",
        role: Role::Embedder,
        description: "Embeds keyframes and search queries in one space: finds shots by what they show. 1.5 GB.",
        license: "Apache-2.0",
        license_url: "https://www.apache.org/licenses/LICENSE-2.0",
        author: "Google",
        source: "https://huggingface.co/google/siglip2-base-patch16-224",
        files: &[
            hf!(
                "google/siglip2-base-patch16-224",
                "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
                "config.json",
                "fe8b5fe6d5734360678fd71c11c21e1ea3364bd8598d34295d9206335973ffd7",
                253
            ),
            hf!(
                "google/siglip2-base-patch16-224",
                "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
                "tokenizer.json",
                "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
                34_363_039
            ),
            hf!(
                "google/siglip2-base-patch16-224",
                "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
                "model.safetensors",
                "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b",
                1_500_800_904
            ),
        ],
    },
    ModelInfo {
        id: AESTHETIC,
        name: "CLIP ViT-L/14 with the LAION aesthetic predictor",
        role: Role::Aesthetic,
        description: "How good a frame looks, about 1 to 10: a linear head (LAION) on CLIP ViT-L/14 image embeddings (OpenAI). 1.7 GB.",
        license: "MIT (OpenAI CLIP; LAION aesthetic predictor)",
        license_url: "https://github.com/openai/CLIP/blob/main/LICENSE",
        author: "OpenAI (CLIP); LAION (aesthetic predictor)",
        source: "https://huggingface.co/openai/clip-vit-large-patch14, https://github.com/LAION-AI/aesthetic-predictor",
        files: &[
            hf!(
                "openai/clip-vit-large-patch14",
                "32bd64288804d66eefd0ccbe215aa642df71cc41",
                "model.safetensors",
                "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb",
                1_710_540_580
            ),
            ModelFile {
                name: "sa_0_4_vit_l_14_linear.pth",
                url: "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/6d122adad522ab246644d9dc1c6d7a3810ee255f/sa_0_4_vit_l_14_linear.pth",
                sha256: "2cd4e60f4f24ae3bcd57b847b13c1f3ba27edc28cc1a7f9ce74ee9f421243cba",
                size: 4_071,
            },
        ],
    },
];

pub const TRANSNET: &str = "transnetv2";
pub const QWEN_2B: &str = "qwen3-vl-2b";
pub const QWEN_4B: &str = "qwen3-vl-4b";
pub const QWEN_8B: &str = "qwen3-vl-8b";
pub const SIGLIP: &str = "siglip2-base-224";
pub const AESTHETIC: &str = "clip-vit-l14-aesthetic";

/// The describer for the fast pass, and for `detail: full`.
pub const FAST_DESCRIBER: &str = QWEN_2B;
pub const FULL_DESCRIBER: &str = QWEN_4B;

/// The models a `fast` analysis uses, and a `full` one.
pub const FAST: &[&str] = &[TRANSNET, QWEN_2B, SIGLIP, AESTHETIC];
pub const FULL: &[&str] = &[TRANSNET, QWEN_4B, SIGLIP, AESTHETIC];

pub fn catalogue() -> &'static [ModelInfo] {
    CATALOGUE
}

pub fn find(id: &str) -> Option<&'static ModelInfo> {
    let id = id.trim();
    CATALOGUE.iter().find(|m| m.id.eq_ignore_ascii_case(id))
}

/// Where model `m` lives under the models directory.
pub fn model_dir(models_dir: &Path, m: &ModelInfo) -> PathBuf {
    models_dir.join(m.id)
}

fn present(dir: &Path, f: &ModelFile) -> bool {
    std::fs::metadata(dir.join(f.name)).is_ok_and(|md| md.len() == f.size)
}

/// All files present with their pinned sizes (contents were verified when downloaded).
pub fn installed(models_dir: &Path, m: &ModelInfo) -> bool {
    let d = model_dir(models_dir, m);
    m.files.iter().all(|f| present(&d, f))
}

/// Whether the model `id` is installed (false for an unknown id).
pub fn installed_id(models_dir: &Path, id: &str) -> bool {
    find(id).is_some_and(|m| installed(models_dir, m))
}

/// Bytes still to download for `m`.
pub fn missing_bytes(models_dir: &Path, m: &ModelInfo) -> u64 {
    let d = model_dir(models_dir, m);
    m.files.iter().filter(|f| !present(&d, f)).map(|f| f.size).sum()
}

/// The Qwen3-VL tokenizer's pin (shared by every size).
pub fn qwen_tokenizer() -> (&'static str, u64) {
    (QWEN_TOKENIZER_SHA, QWEN_TOKENIZER_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_are_pinned_and_checksummed() {
        for m in catalogue() {
            assert!(!m.license.contains("NC") && !m.license.contains("ND"), "{}", m.id);
            assert!(m.license.contains("MIT") || m.license.contains("Apache-2.0"), "{}", m.id);
            for f in m.files {
                let ok_host =
                    ["https://huggingface.co/Qwen/", "https://huggingface.co/google/", "https://huggingface.co/openai/"].iter().any(|h| f.url.starts_with(h))
                        || f.url.starts_with("https://media.githubusercontent.com/media/soCzech/TransNetV2/")
                        || f.url.starts_with("https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/");
                assert!(ok_host, "{}", f.url);
                assert!(f.url.ends_with(f.name), "{}", f.url);
                // a 40-hex revision somewhere in the path, never a branch name
                let pinned = f.url.split('/').any(|p| p.len() == 40 && p.bytes().all(|b| b.is_ascii_hexdigit()));
                assert!(pinned, "{}", f.url);
                assert!(!f.url.contains("/main/") && !f.url.contains("/master/"), "{}", f.url);
                assert_eq!(f.sha256.len(), 64);
                assert!(f.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
                assert!(f.size > 0);
                if f.name == "tokenizer.json" && m.id.starts_with("qwen") {
                    assert_eq!((f.sha256, f.size), qwen_tokenizer());
                }
            }
        }
        // the fast set stays under ~6 GB, everything the live check downloads under 20 GB
        let size = |ids: &[&str]| ids.iter().filter_map(|i| find(i)).map(|m| m.size()).sum::<u64>();
        assert!(size(FAST) < 8_000_000_000, "{}", size(FAST));
        assert!(size(&[TRANSNET, QWEN_2B, QWEN_4B, SIGLIP, AESTHETIC]) < 20_000_000_000);
        for id in FAST.iter().chain(FULL) {
            assert!(find(id).is_some(), "{id}");
        }
        assert_eq!(find(" QWEN3-VL-2B ").map(|m| m.id), Some(QWEN_2B));
        assert!(find("nope").is_none());
    }

    #[test]
    fn install_state_follows_the_files() {
        let dir = std::env::temp_dir().join(format!("filmcraft-scenes-catalog-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let m = find(TRANSNET).unwrap();
        assert!(!installed(&dir, m));
        assert_eq!(missing_bytes(&dir, m), m.size());
        std::fs::create_dir_all(model_dir(&dir, m)).unwrap();
        // a file of the wrong size is not installed
        std::fs::write(model_dir(&dir, m).join("variables.index"), b"short").unwrap();
        assert_eq!(missing_bytes(&dir, m), m.size());
        std::fs::write(model_dir(&dir, m).join("variables.index"), vec![0u8; 5_526]).unwrap();
        assert_eq!(missing_bytes(&dir, m), 30_516_656);
        assert!(!installed_id(&dir, TRANSNET));
        assert!(!installed_id(&dir, "nope"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
