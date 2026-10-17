//! The downloadable neural music models. Every file is pinned to a revision of the official
//! Hugging Face repository and checked against its SHA-256 when downloaded (the downloader is
//! `filmcraft_speech::models::download_files`, run by the engine after the user confirms the
//! size, source and licence). Nothing here is bundled with FilmCraft or committed.
//!
//! Licences (checked 2026-10-17, see the crate README): the ACE-Step 1.5 weights (DiT, VAE,
//! silence latent) and the official code are MIT (ACE Studio and StepFun; the modeling file that
//! ships with the weights carries an Apache-2.0 header); the bundled text encoder is Qwen's
//! Qwen3-Embedding-0.6B, Apache-2.0. Pins verified against the Hugging Face API (LFS SHA-256)
//! and `shasum -a 256`.

use std::path::{Path, PathBuf};

use serde::Serialize;

/// One file of a model package.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelFile {
    /// Local file name inside the package directory (flat: no subdirectories).
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

/// A downloadable text-to-music model.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Package {
    /// Model id in commands and the package's directory name under the models directory.
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub license: &'static str,
    pub source: &'static str,
    /// Diffusion steps the model is distilled for.
    pub steps: usize,
    pub files: &'static [ModelFile],
}

/// File names the loader reads from a package directory.
pub const DIT_CONFIG: &str = "dit-config.json";
pub const DIT_WEIGHTS: &str = "dit.safetensors";
pub const SILENCE_LATENT: &str = "silence_latent.pt";
pub const VAE_CONFIG: &str = "vae-config.json";
pub const VAE_WEIGHTS: &str = "vae.safetensors";
pub const TEXT_CONFIG: &str = "text-config.json";
pub const TEXT_TOKENIZER: &str = "text-tokenizer.json";
pub const TEXT_WEIGHTS: &str = "text-encoder.safetensors";

pub const LICENSE: &str =
    "MIT (ACE-Step 1.5 by ACE Studio and StepFun: DiT, VAE and code); Apache-2.0 (Qwen3-Embedding-0.6B text encoder by the Qwen team, Alibaba)";

macro_rules! ace {
    ($name:expr, $path:literal, $sha:literal, $size:expr) => {
        ModelFile {
            name: $name,
            url: concat!("https://huggingface.co/ACE-Step/Ace-Step1.5/resolve/19671f406d603126926c1b7e2adc169acbcade22/", $path),
            sha256: $sha,
            size: $size,
        }
    };
}

/// The models, smallest download first.
pub static MODELS: &[Package] = &[Package {
    id: "acestep-v15-turbo",
    name: "ACE-Step 1.5 turbo (neural music)",
    description: "Text to music: a 2 B diffusion transformer distilled to 8 steps, a Qwen3 text encoder and a 48 kHz stereo VAE. Instrumental music from a description (genre, mood, instruments, tempo), 5 s to 4 min. About 5 GB of memory on a Mac GPU (10 GB on the CPU); seconds to a minute per clip on Apple Silicon.",
    license: LICENSE,
    source: "https://huggingface.co/ACE-Step/Ace-Step1.5",
    steps: 8,
    files: &[
        ace!(DIT_CONFIG, "acestep-v15-turbo/config.json", "74745ff704ea49164c3d2d1c99fc0670f3fc635a869f0aec2d1311e6a52d400a", 1968),
        ace!(SILENCE_LATENT, "acestep-v15-turbo/silence_latent.pt", "a778e9dd942f5e8b2c09c55370782d318834432b03dabbcdf70e6ed49ad6358b", 3_841_215),
        ace!(VAE_CONFIG, "vae/config.json", "14e019904df567f26df750317a70e2bd08f9f8f3c40ff4a24c97d1cd3f20ccd2", 425),
        ace!(TEXT_CONFIG, "Qwen3-Embedding-0.6B/config.json", "bb23c1607cfe059a58d8f0196cf1cebb52082b1056b8e358a579da80a5759420", 1359),
        ace!(TEXT_TOKENIZER, "Qwen3-Embedding-0.6B/tokenizer.json", "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a", 11_423_705),
        ace!(VAE_WEIGHTS, "vae/diffusion_pytorch_model.safetensors", "da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0", 337_431_388),
        ace!(TEXT_WEIGHTS, "Qwen3-Embedding-0.6B/model.safetensors", "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd", 1_191_586_416),
        ace!(DIT_WEIGHTS, "acestep-v15-turbo/model.safetensors", "3f6e0797fad420a39bd33979eb6e840e30989e34a3794e843d23b60ec6e422d7", 4_787_825_604),
    ],
}];

/// The model used when none is named.
pub const DEFAULT_MODEL: &str = "acestep-v15-turbo";

/// The package with this id.
pub fn model(id: &str) -> Option<&'static Package> {
    MODELS.iter().find(|m| m.id == id)
}

/// Where a package lives under the models directory.
pub fn package_dir(models_dir: &Path, m: &Package) -> PathBuf {
    models_dir.join(m.id)
}

/// Download size of a package, bytes.
pub fn size(m: &Package) -> u64 {
    m.files.iter().map(|f| f.size).sum()
}

/// Bytes of the package still to download (files missing or of the wrong size).
pub fn missing_bytes(models_dir: &Path, m: &Package) -> u64 {
    let d = package_dir(models_dir, m);
    m.files.iter().filter(|f| !std::fs::metadata(d.join(f.name)).is_ok_and(|md| md.len() == f.size)).map(|f| f.size).sum()
}

/// Every file of the package is present with its pinned size.
pub fn installed(models_dir: &Path, m: &Package) -> bool {
    missing_bytes(models_dir, m) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pins_are_well_formed_and_licences_are_open() {
        assert!(model(DEFAULT_MODEL).is_some());
        for m in MODELS {
            assert!(!m.license.contains("NC") && !m.license.contains("ND"), "{}", m.id);
            assert!(m.license.contains("MIT"));
            assert!(m.source.starts_with("https://huggingface.co/ACE-Step/"));
            let mut names = std::collections::HashSet::new();
            for f in m.files {
                assert!(names.insert(f.name), "duplicate {}", f.name);
                assert!(!f.name.contains('/') && !f.name.contains(".."), "{}", f.name);
                assert_eq!(f.sha256.len(), 64, "{}", f.name);
                assert!(f.sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
                assert!(f.url.starts_with("https://huggingface.co/ACE-Step/Ace-Step1.5/resolve/19671f406d603126926c1b7e2adc169acbcade22/"), "{}", f.url);
                assert!(f.size > 0);
            }
            for n in [DIT_CONFIG, DIT_WEIGHTS, SILENCE_LATENT, VAE_CONFIG, VAE_WEIGHTS, TEXT_CONFIG, TEXT_TOKENIZER, TEXT_WEIGHTS] {
                assert!(names.contains(n), "{n} is not downloaded");
            }
            assert!(size(m) > 6_000_000_000 && size(m) < 7_000_000_000);
        }
    }

    #[test]
    fn install_state_follows_the_files() {
        let d = std::env::temp_dir().join(format!("filmcraft-acestep-cat-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let m = &MODELS[0];
        assert!(!installed(&d, m));
        assert_eq!(missing_bytes(&d, m), size(m));
        let p = package_dir(&d, m);
        std::fs::create_dir_all(&p).unwrap();
        std::fs::write(p.join(DIT_CONFIG), vec![b' '; 1968]).unwrap();
        assert_eq!(missing_bytes(&d, m), size(m) - 1968);
        // a file of the wrong size still counts as missing
        std::fs::write(p.join(VAE_CONFIG), b"{}").unwrap();
        assert_eq!(missing_bytes(&d, m), size(m) - 1968);
        let _ = std::fs::remove_dir_all(&d);
    }
}
