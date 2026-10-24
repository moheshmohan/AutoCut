//! The speech model catalogue and the model store.
//!
//! Models are downloaded on first use, after the user confirms (the dialog shows the size, the
//! source URL and the licence), into `<data dir>/models/<id>/`. They are never bundled with
//! FilmCraft or committed to the repository. Every file is pinned to a Hugging Face revision
//! (OpenAI's own repositories, or a fine-tune's) and checked against its SHA-256 before it is
//! used.
//!
//! Licence: OpenAI released the Whisper code and model weights under the MIT licence
//! (<https://github.com/openai/whisper/blob/main/LICENSE>); the safetensors conversions OpenAI
//! publishes at `huggingface.co/openai/whisper-*` are labelled Apache-2.0. The Malayalam model
//! ([`MALAYALAM_MODEL`]) is Thennal D K's fine-tune of Whisper medium, Apache-2.0. All are
//! permissive (no NC/ND). The English option ([`TURBO_MODEL`]) is OpenAI's large-v3-turbo (MIT);
//! the Malayalam second opinion ([`INDICWHISPER_ML`]) is AI4Bharat's IndicWhisper (MIT).
//!
//! [`support`] lists the models that help understand speech without transcribing it: Silero VAD
//! (MIT, where speech is) and SpeechBrain's ECAPA-TDNN (Apache-2.0, who speaks). They download
//! and verify the same way.

use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::Digest;

/// One file of a model.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelFile {
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

/// What a model does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ModelKind {
    /// A speech recogniser (Whisper): `transcript.generate` and the narration checks hear with it.
    Recogniser,
    /// Voice activity detection: where speech is.
    VoiceActivity,
    /// Speaker embeddings: who is speaking (diarization, the narration voice gate).
    Speaker,
}

/// A downloadable speech model.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub kind: ModelKind,
    /// Recognises languages other than English (and detects the language).
    pub multilingual: bool,
    /// Fine-tuned for this one language (ISO 639-1): ask it for that language only.
    pub language: Option<&'static str>,
    pub description: &'static str,
    pub license: &'static str,
    pub license_url: &'static str,
    pub author: &'static str,
    /// Human-readable source page.
    pub source: &'static str,
    pub files: &'static [ModelFile],
}

impl ModelInfo {
    pub fn size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

const LICENSE: &str = "MIT (OpenAI Whisper weights); Hugging Face conversion: Apache-2.0";
const LICENSE_URL: &str = "https://github.com/openai/whisper/blob/main/LICENSE";
const AUTHOR: &str = "OpenAI";

macro_rules! hf {
    ($repo:literal, $rev:literal, $name:literal, $sha:literal, $size:expr) => {
        ModelFile { name: $name, url: concat!("https://huggingface.co/", $repo, "/resolve/", $rev, "/", $name), sha256: $sha, size: $size }
    };
}

const TOKENIZER_MULTI: &str = "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566";

static CATALOGUE: &[ModelInfo] = &[
    ModelInfo {
        id: "whisper-tiny",
        name: "Whisper tiny (multilingual)",
        kind: ModelKind::Recogniser,
        multilingual: true,
        language: None,
        description: "39 M parameters. Fastest; fine for clear speech, rough on noisy audio.",
        license: LICENSE,
        license_url: LICENSE_URL,
        author: AUTHOR,
        source: "https://huggingface.co/openai/whisper-tiny",
        files: &[
            hf!(
                "openai/whisper-tiny",
                "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
                "config.json",
                "ffdccec4f3211f4c63310f2b7098f309fe70f3952cedc5e4d11e43f5b2379b98",
                1983
            ),
            hf!(
                "openai/whisper-tiny",
                "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
                "generation_config.json",
                "a5d5325911f16e74001a72fa13d6e208eee51548f994646de1f4b4cc8b35b512",
                3747
            ),
            hf!(
                "openai/whisper-tiny",
                "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
                "tokenizer.json",
                "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566",
                2480466
            ),
            hf!(
                "openai/whisper-tiny",
                "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
                "model.safetensors",
                "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395",
                151061672
            ),
        ],
    },
    ModelInfo {
        id: "whisper-base",
        name: "Whisper base (multilingual)",
        kind: ModelKind::Recogniser,
        multilingual: true,
        language: None,
        description: "74 M parameters. The default: a good balance of speed and accuracy.",
        license: LICENSE,
        license_url: LICENSE_URL,
        author: AUTHOR,
        source: "https://huggingface.co/openai/whisper-base",
        files: &[
            hf!(
                "openai/whisper-base",
                "e37978b90ca9030d5170a5c07aadb050351a65bb",
                "config.json",
                "a153c53883a6799b6f056b4a8d1a515c9926d03994682ba88a7616618d7da0c1",
                1983
            ),
            hf!(
                "openai/whisper-base",
                "e37978b90ca9030d5170a5c07aadb050351a65bb",
                "generation_config.json",
                "444b3f636d2fff89dd9ecf549e2a085b61f7ff0fa0246d4628bac6a3b8cc9ba4",
                3807
            ),
            hf!(
                "openai/whisper-base",
                "e37978b90ca9030d5170a5c07aadb050351a65bb",
                "tokenizer.json",
                "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566",
                2480466
            ),
            hf!(
                "openai/whisper-base",
                "e37978b90ca9030d5170a5c07aadb050351a65bb",
                "model.safetensors",
                "07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b",
                290403936
            ),
        ],
    },
    ModelInfo {
        id: "whisper-small",
        name: "Whisper small (multilingual)",
        kind: ModelKind::Recogniser,
        multilingual: true,
        language: None,
        description: "244 M parameters. Most accurate here; about 4× slower than base.",
        license: LICENSE,
        license_url: LICENSE_URL,
        author: AUTHOR,
        source: "https://huggingface.co/openai/whisper-small",
        files: &[
            hf!(
                "openai/whisper-small",
                "973afd24965f72e36ca33b3055d56a652f456b4d",
                "config.json",
                "e6a2b489da1b5aed65a8eb8d1e7466fa867ad5643a8bc138ba708bd56b2875c4",
                1967
            ),
            hf!(
                "openai/whisper-small",
                "973afd24965f72e36ca33b3055d56a652f456b4d",
                "generation_config.json",
                "71565b8ef50d0bf7a1193ed4bbed195b94e70c18894d81bba2f1233dcec3ab53",
                3868
            ),
            hf!(
                "openai/whisper-small",
                "973afd24965f72e36ca33b3055d56a652f456b4d",
                "tokenizer.json",
                "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566",
                2480466
            ),
            hf!(
                "openai/whisper-small",
                "973afd24965f72e36ca33b3055d56a652f456b4d",
                "model.safetensors",
                "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b",
                966995080
            ),
        ],
    },
    // Whisper medium fine-tuned on Malayalam: the most accurate allowed recogniser measured for
    // Malayalam (crate README, "Malayalam"); the strict check of Malayalam narration hears with
    // it. Its weights are the safetensors conversion Hugging Face's bot made of the release
    // (pull request 2, the same tensors as `pytorch_model.bin`); its tokenizer is Whisper
    // medium's own (the fine-tune ships the same vocabulary without `tokenizer.json`).
    ModelInfo {
        id: MALAYALAM_MODEL,
        name: "Whisper medium, Malayalam (Thennal D K)",
        kind: ModelKind::Recogniser,
        multilingual: true,
        language: Some("ml"),
        description: "769 M parameters fine-tuned on Malayalam speech (Common Voice, FLEURS, IMaSC, MSC, ULCA, Indic TTS). Checks what Malayalam narration said (the strict check) and transcribes Malayalam, English loanwords in Malayalam script.",
        license: "Apache-2.0 (fine-tune by Thennal D K); base model MIT (OpenAI Whisper)",
        license_url: "https://www.apache.org/licenses/LICENSE-2.0",
        author: "Thennal D K (fine-tune of OpenAI Whisper medium)",
        source: "https://huggingface.co/thennal/whisper-medium-ml",
        files: &[
            hf!(
                "thennal/whisper-medium-ml",
                "c65085c5b75f47dc5f1585773a83bf83f2e4dbc7",
                "config.json",
                "0509bb0a122ef0793235600d447225e629fd14de222a5bc15cc1cd42dbba3616",
                1296
            ),
            hf!(
                "thennal/whisper-medium-ml",
                "c65085c5b75f47dc5f1585773a83bf83f2e4dbc7",
                "generation_config.json",
                "6281fbb87a7827a766f69db14bad2f8c049be69926bbc91386fe38b343f56713",
                3504
            ),
            hf!(
                "openai/whisper-medium",
                "abdf7c39ab9d0397620ccaea8974cc764cd0953e",
                "tokenizer.json",
                "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566",
                2480466
            ),
            hf!(
                "thennal/whisper-medium-ml",
                "c65085c5b75f47dc5f1585773a83bf83f2e4dbc7",
                "model.safetensors",
                "72042c8077a9856fed274503433a4545706b3b0b5bc1ba987627982978e08e95",
                3055544304
            ),
        ],
    },
    // Whisper large-v3-turbo: large-v3's encoder with a 4-layer decoder (809 M parameters). The
    // most accurate English recogniser here at about the speed of small; it writes Malayalam in
    // other scripts (crate README), so it is the English option.
    ModelInfo {
        id: TURBO_MODEL,
        name: "Whisper large-v3-turbo (multilingual; best for English)",
        kind: ModelKind::Recogniser,
        multilingual: true,
        language: None,
        description: "809 M parameters: large-v3's encoder with a 4-layer decoder. The most accurate English transcription here, about as fast as small on the GPU; checks English narration.",
        license: "MIT (OpenAI Whisper weights)",
        license_url: LICENSE_URL,
        author: AUTHOR,
        source: "https://huggingface.co/openai/whisper-large-v3-turbo",
        files: &[
            hf!(
                "openai/whisper-large-v3-turbo",
                "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
                "config.json",
                "c5b526b3e3cd64cd8940dabb45e8ba726629e22d8ed389c29b552f9140daf04a",
                1256
            ),
            hf!(
                "openai/whisper-large-v3-turbo",
                "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
                "generation_config.json",
                "cce11bfe3aaa6ae9e072ea2637caaec8795e68d9b67e655a5af16ee509681a4c",
                3772
            ),
            hf!(
                "openai/whisper-large-v3-turbo",
                "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
                "tokenizer.json",
                "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd",
                2710337
            ),
            hf!(
                "openai/whisper-large-v3-turbo",
                "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
                "model.safetensors",
                "542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1",
                1617824864
            ),
        ],
    },
    // IndicWhisper Malayalam (AI4Bharat's Vistaar benchmark release, MIT): the second opinion of
    // the strict check of Malayalam narration. Best on FLEURS of the candidates measured (crate
    // README) but writes spoken numbers as digits, so it never judges alone. The weights are
    // Kavya Manohar's safetensors mirror of the release (the same tensors as AI4Bharat's
    // `pytorch_model.bin`, also mirrored by VRCLC); the tokenizer is Whisper medium's own.
    ModelInfo {
        id: INDICWHISPER_ML,
        name: "IndicWhisper Malayalam (AI4Bharat)",
        kind: ModelKind::Recogniser,
        multilingual: true,
        language: Some("ml"),
        description: "Whisper medium fine-tuned on Malayalam by AI4Bharat (Vistaar). The second opinion of the strict check of Malayalam narration: a line is retaken only when both Malayalam models disagree with the script.",
        license: "MIT (AI4Bharat IndicWhisper); base model MIT (OpenAI Whisper)",
        license_url: "https://github.com/AI4Bharat/vistaar/blob/master/LICENSE",
        author: "AI4Bharat (fine-tune of OpenAI Whisper medium); mirror by Kavya Manohar",
        source: "https://huggingface.co/kavyamanohar/AI4B-Indicwhisper-ml",
        files: &[
            hf!(
                "kavyamanohar/AI4B-Indicwhisper-ml",
                "3fc46dc7338c7737e1d4580a80f1f0ff697d017e",
                "config.json",
                "8de96b551ddff9ccf8df4379a84fe7e1cd79d7312bd16673082d1f71501388b7",
                1007
            ),
            hf!(
                "kavyamanohar/AI4B-Indicwhisper-ml",
                "3fc46dc7338c7737e1d4580a80f1f0ff697d017e",
                "generation_config.json",
                "67f1d18ec3b85ef8d02cb72dada44bbdfb825cf21a64e1e82ab32fd2f3ed1525",
                3491
            ),
            hf!(
                "openai/whisper-medium",
                "abdf7c39ab9d0397620ccaea8974cc764cd0953e",
                "tokenizer.json",
                "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566",
                2480466
            ),
            hf!(
                "kavyamanohar/AI4B-Indicwhisper-ml",
                "3fc46dc7338c7737e1d4580a80f1f0ff697d017e",
                "model.safetensors",
                "4732f2bc64b2f702d086c9ab25fc8f9334d3615e16214fa53e725286d53d97d6",
                1527827760
            ),
        ],
    },
];

/// Models that help understand speech without transcribing it: where speech is and who speaks.
static SUPPORT: &[ModelInfo] = &[
    // Silero VAD v5, the 16 kHz network as the authors publish it in safetensors (MIT).
    ModelInfo {
        id: SILERO_VAD,
        name: "Silero VAD v5",
        kind: ModelKind::VoiceActivity,
        multilingual: true,
        language: None,
        description: "Voice activity detection (1.2 MB): where people speak, in 32 ms steps, robust to noise and music. The energy detector is the fallback.",
        license: "MIT (Silero Team)",
        license_url: "https://github.com/snakers4/silero-vad/blob/master/LICENSE",
        author: "Silero Team",
        source: "https://github.com/snakers4/silero-vad",
        files: &[ModelFile {
            name: "silero_vad_16k.safetensors",
            url: "https://raw.githubusercontent.com/snakers4/silero-vad/fb7d7c7f5df63235f0afd037d8397593829aeb41/src/silero_vad/data/silero_vad_16k.safetensors",
            sha256: "c59271c284ae9c8335d795d60e0bfdb71aaaceec578d9bd9ffc1b8153c319ea1",
            size: 1239748,
        }],
    },
    // SpeechBrain's ECAPA-TDNN trained on VoxCeleb 1+2 (Apache-2.0): 192-value speaker
    // embeddings for diarization and the narration voice gate.
    ModelInfo {
        id: ECAPA_MODEL,
        name: "ECAPA-TDNN speaker embeddings (SpeechBrain, VoxCeleb)",
        kind: ModelKind::Speaker,
        multilingual: true,
        language: None,
        description: "Speaker embeddings (83 MB): tells voices apart for speaker labels, and checks that a cloned narration still sounds like its speaker.",
        license: "Apache-2.0 (SpeechBrain)",
        license_url: "https://www.apache.org/licenses/LICENSE-2.0",
        author: "SpeechBrain (Ravanelli et al.); ECAPA-TDNN by Desplanques, Thienpondt and Demuynck",
        source: "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb",
        files: &[hf!(
            "speechbrain/spkrec-ecapa-voxceleb",
            "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
            "embedding_model.ckpt",
            "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2",
            83316686
        )],
    },
];

/// The default model.
pub const DEFAULT_MODEL: &str = "whisper-base";

/// The Malayalam model: what the strict check of Malayalam narration hears with.
pub const MALAYALAM_MODEL: &str = "whisper-medium-ml";

/// The second Malayalam model: the strict check's second opinion.
pub const INDICWHISPER_ML: &str = "indicwhisper-ml";

/// The English option: the strict check of English narration hears with it when installed.
pub const TURBO_MODEL: &str = "whisper-large-v3-turbo";

/// The voice activity detector.
pub const SILERO_VAD: &str = "silero-vad";

/// The speaker embedding model.
pub const ECAPA_MODEL: &str = "ecapa-voxceleb";

/// The recognisers.
pub fn catalogue() -> &'static [ModelInfo] {
    CATALOGUE
}

/// The voice activity and speaker models.
pub fn support() -> &'static [ModelInfo] {
    SUPPORT
}

/// A recogniser by id.
pub fn find(id: &str) -> Option<&'static ModelInfo> {
    CATALOGUE.iter().find(|m| m.id == id)
}

/// Any model (recogniser or support model) by id.
pub fn find_any(id: &str) -> Option<&'static ModelInfo> {
    find(id).or_else(|| SUPPORT.iter().find(|m| m.id == id))
}

/// The installed file of single-file support model `id` (`None`: unknown or not downloaded).
pub fn support_file(models_dir: &Path, id: &str) -> Option<PathBuf> {
    let m = SUPPORT.iter().find(|m| m.id == id)?;
    let f = m.files.first()?;
    installed(models_dir, m).then(|| model_dir(models_dir, m).join(f.name))
}

/// Where model `m` lives under `models_dir`.
pub fn model_dir(models_dir: &Path, m: &ModelInfo) -> PathBuf {
    models_dir.join(m.id)
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && !name.contains(['/', '\\'])
        && !matches!(name, "." | "..")
        && Path::new(name).components().count() == 1
        && Path::new(name).file_name().is_some_and(|part| part == name)
}

fn valid_file(path: &Path, file: &ModelFile) -> bool {
    if !valid_name(file.name) || !std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file() && metadata.len() == file.size) {
        return false;
    }
    let Ok(mut input) = std::fs::File::open(path) else { return false };
    let mut hasher = sha2::Sha256::new();
    if std::io::copy(&mut input, &mut hasher).is_err() {
        return false;
    }
    format!("{:x}", hasher.finalize()).eq_ignore_ascii_case(file.sha256)
}

/// All files present and matching their pinned size and SHA-256.
pub fn installed(models_dir: &Path, m: &ModelInfo) -> bool {
    let d = model_dir(models_dir, m);
    m.files.iter().all(|f| valid_file(&d.join(f.name), f))
}

/// Bytes still to download for `m` (files already present are skipped).
pub fn missing_bytes(models_dir: &Path, m: &ModelInfo) -> u64 {
    let d = model_dir(models_dir, m);
    m.files.iter().filter(|f| !valid_file(&d.join(f.name), f)).map(|f| f.size).sum()
}

/// The tokenizer shared by the multilingual models has one checksum.
pub fn shared_tokenizer_sha() -> &'static str {
    TOKENIZER_MULTI
}

/// Download progress: `(bytes done, bytes total, file)`; return false to cancel.
#[cfg(feature = "download")]
pub type DownloadProgress<'a> = &'a mut dyn FnMut(u64, u64, &str) -> bool;

/// Download the missing files of `m` into `models_dir/<id>/`: each file streams into `<name>.part`,
/// its SHA-256 is checked, then it is renamed into place. A wrong checksum deletes the part file
/// and fails. Pure Rust TLS (rustls + RustCrypto) with the operating system's certificate
/// verifier.
#[cfg(feature = "download")]
pub fn download(models_dir: &Path, m: &ModelInfo, progress: DownloadProgress) -> Result<(), crate::SpeechError> {
    download_files(&model_dir(models_dir, m), m.files, progress)
}

/// [`download`] for any pinned file list (the neural voices use it too): missing files of
/// `files` go into `dir`, each verified against its SHA-256 and size before it is renamed into
/// place.
#[cfg(feature = "download")]
pub fn download_files(dir: &Path, files: &[ModelFile], progress: DownloadProgress) -> Result<(), crate::SpeechError> {
    use crate::SpeechError;
    use std::io::{Read, Write};
    if let Some(file) = files.iter().find(|file| !valid_name(file.name)) {
        return Err(SpeechError::Download(format!("invalid model artifact name: {:?}", file.name)));
    }
    std::fs::create_dir_all(dir)?;
    let total: u64 = files.iter().filter(|f| !valid_file(&dir.join(f.name), f)).map(|f| f.size).sum();
    let agent = agent();
    let mut done = 0u64;
    for f in files {
        let dest = dir.join(f.name);
        if valid_file(&dest, f) {
            continue;
        }
        let part = dir.join(format!("{}.part", f.name));
        let resp = agent.get(f.url).call().map_err(|e| SpeechError::Download(format!("{}: {e}", f.url)))?;
        let mut body = resp.into_body();
        let mut reader = body.as_reader();
        let mut out = std::io::BufWriter::new(std::fs::File::create(&part)?);
        let mut hasher = sha2::Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut got = 0u64;
        loop {
            let n = reader.read(&mut buf).map_err(|e| SpeechError::Download(format!("{}: {e}", f.name)))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            out.write_all(&buf[..n])?;
            got += n as u64;
            if !progress(done + got, total, f.name) {
                drop(out);
                let _ = std::fs::remove_file(&part);
                return Err(SpeechError::Cancelled);
            }
        }
        out.flush()?;
        drop(out);
        let sha: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if sha != f.sha256 || got != f.size {
            let _ = std::fs::remove_file(&part);
            return Err(SpeechError::Download(format!("{}: checksum mismatch (got {sha}, {got} bytes; expected {}, {} bytes)", f.name, f.sha256, f.size)));
        }
        std::fs::rename(&part, &dest)?;
        done += f.size;
    }
    Ok(())
}

#[cfg(feature = "download")]
fn agent() -> ureq::Agent {
    let provider = std::sync::Arc::new(rustls_rustcrypto::provider());
    ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::Rustls)
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .unversioned_rustls_crypto_provider(provider)
                .build(),
        )
        .timeout_connect(Some(std::time::Duration::from_secs(30)))
        .build()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_is_pinned_and_checksummed() {
        assert!(find(DEFAULT_MODEL).is_some());
        for m in catalogue() {
            assert!(m.size() > 100_000_000, "{}", m.id);
            assert_eq!(m.kind, ModelKind::Recogniser);
            for f in m.files {
                assert!(
                    f.url.starts_with("https://huggingface.co/openai/whisper-")
                        || f.url.starts_with("https://huggingface.co/thennal/whisper-")
                        || f.url.starts_with("https://huggingface.co/kavyamanohar/AI4B-Indicwhisper-ml/"),
                    "{}",
                    f.url
                );
                // pinned to a commit, never a branch
                let rev = f.url.split("/resolve/").nth(1).and_then(|r| r.split('/').next()).unwrap_or("");
                assert!(rev.len() == 40 && rev.bytes().all(|b| b.is_ascii_hexdigit()), "{}", f.url);
                assert!(f.url.contains("/resolve/") && f.url.ends_with(f.name));
                assert_eq!(f.sha256.len(), 64);
                assert!(f.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
                // large-v3's vocabulary has one more language (51 866 tokens): its own tokenizer
                if f.name == "tokenizer.json" && m.id != TURBO_MODEL {
                    assert_eq!(f.sha256, shared_tokenizer_sha());
                }
            }
        }
    }

    #[test]
    fn support_models_are_pinned_and_permissive() {
        for id in [SILERO_VAD, ECAPA_MODEL] {
            let m = find_any(id).unwrap();
            assert!(find(id).is_none(), "{id} is not a recogniser");
            assert_ne!(m.kind, ModelKind::Recogniser);
            assert!(!m.license.contains("NC") && !m.license.contains("ND") && !m.license.contains("GPL"), "{id}");
            for f in m.files {
                assert_eq!(f.sha256.len(), 64);
                assert!(f.url.ends_with(f.name));
                // a commit, never a branch
                assert!(f.url.split('/').any(|p| p.len() == 40 && p.bytes().all(|b| b.is_ascii_hexdigit())), "{}", f.url);
            }
        }
        assert_eq!(find_any(SILERO_VAD).unwrap().size(), 1239748);
        assert!(support_file(std::path::Path::new("/nonexistent"), SILERO_VAD).is_none());
        assert!(support_file(std::path::Path::new("/nonexistent"), "whisper-base").is_none());
        // the new recognisers: turbo is the English option, IndicWhisper the Malayalam second opinion
        assert_eq!(find(INDICWHISPER_ML).unwrap().language, Some("ml"));
        assert!(find(TURBO_MODEL).unwrap().language.is_none());
    }

    #[test]
    fn the_malayalam_model_is_listed() {
        let m = find(MALAYALAM_MODEL).unwrap();
        assert_eq!(m.language, Some("ml"));
        assert!(m.license.starts_with("Apache-2.0") && !m.license.contains("NC"));
        assert_eq!(m.files.iter().map(|f| f.name).collect::<Vec<_>>(), ["config.json", "generation_config.json", "tokenizer.json", "model.safetensors"]);
        assert_eq!(m.size(), 1296 + 3504 + 2480466 + 3055544304);
        assert!(catalogue().iter().filter(|m| m.id != MALAYALAM_MODEL && m.id != INDICWHISPER_ML).all(|m| m.language.is_none()));
    }

    #[test]
    fn installed_checks_sizes() {
        let dir = std::env::temp_dir().join(format!("filmcraft-speech-models-{}", std::process::id()));
        let m = find("whisper-tiny").unwrap();
        assert!(!installed(&dir, m));
        assert_eq!(missing_bytes(&dir, m), m.size());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn artifact_names_stay_inside_the_model_directory() {
        for name in ["../escape", "/tmp/escape", r"..\escape", r"C:\escape", "", ".", ".."] {
            assert!(!valid_name(name), "accepted {name:?}");
        }
        assert!(valid_name("model.safetensors"));
    }

    #[test]
    fn same_length_corruption_is_not_installed() {
        static FILES: &[ModelFile] = &[ModelFile {
            name: "tiny.bin",
            url: "https://example.invalid/tiny.bin",
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            size: 3,
        }];
        let model = ModelInfo {
            id: "integrity-test",
            name: "integrity test",
            multilingual: false,
            description: "",
            license: "",
            license_url: "",
            author: "",
            source: "",
            files: FILES,
        };
        let root = tempfile::tempdir().unwrap();
        let directory = model_dir(root.path(), &model);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("tiny.bin"), b"bad").unwrap();
        assert!(!installed(root.path(), &model));
        assert_eq!(missing_bytes(root.path(), &model), 3);
        std::fs::write(directory.join("tiny.bin"), b"abc").unwrap();
        assert!(installed(root.path(), &model));
    }
}
