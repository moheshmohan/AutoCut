//! Speech-to-text for FilmCraft's Text panel ▸ Transcript.
//!
//! - [`Transcriber`]: the trait every speech recogniser implements. It takes mono 16 kHz samples
//!   and returns a [`Transcript`] whose words carry media-time bounds (`Tick`s; one 16 kHz sample
//!   is exactly 15 876 000 ticks).
//! - [`models`]: the catalogue of downloadable Whisper models (source URL pinned to a revision,
//!   SHA-256, size, licence), where they live in the per-user data directory, and (feature
//!   `download`) the verified downloader. Weights are **never** bundled or committed.
//! - [`whisper`] (feature `whisper`): Whisper inference in pure Rust on candle, with timestamp
//!   decoding, language detection and word-level timestamps from cross-attention alignment (DTW).
//! - [`diarize`]: speaker labelling by clustering per-chunk MFCC statistics (classical, no model).
//! - [`mel`]: the log-mel front end shared by Whisper and diarization.
//! - [`vad`]: where speech is (Silero VAD, [`silero`], with an energy fallback) and energy-based
//!   tightening of word bounds (keeps pauses out of words).
//! - [`ecapa`] (feature `whisper`): ECAPA-TDNN speaker embeddings on candle ([`fbank`] is its
//!   front end): neural speaker labels and the narration voice gate.
//!
//! See `docs/transcripts.md` for the user-facing behaviour and accuracy numbers.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod diarize;
#[cfg(feature = "whisper")]
pub mod ecapa;
pub mod fbank;
pub mod mel;
pub mod models;
pub mod silero;
pub mod speakers;
pub mod understand;
pub mod vad;
#[cfg(feature = "whisper")]
pub mod whisper;

use filmcraft_project::Transcript;
use filmcraft_time::{TICKS_PER_SECOND, Tick};

/// The sample rate every transcriber takes.
pub const SAMPLE_RATE: u32 = 16_000;

/// Ticks per 16 kHz sample (exact).
pub const TICKS_PER_SAMPLE: i64 = TICKS_PER_SECOND / SAMPLE_RATE as i64;

/// Media time of a 16 kHz sample index.
pub fn sample_tick(n: i64) -> Tick {
    Tick(n.saturating_mul(TICKS_PER_SAMPLE))
}

/// Media time of `s` seconds (rounded to the nearest 16 kHz sample).
pub fn seconds_tick(s: f64) -> Tick {
    sample_tick((s * SAMPLE_RATE as f64).round() as i64)
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SpeechError {
    #[error("stopped")]
    Cancelled,
    #[error("the speech model `{0}` is not downloaded")]
    NotInstalled(String),
    #[error("unknown speech model `{0}`")]
    UnknownModel(String),
    #[error("speech-to-text is not available in this build ({0})")]
    Unavailable(String),
    #[error("model: {0}")]
    Model(String),
    #[error("download: {0}")]
    Download(String),
    #[error("{0}")]
    Io(String),
}

impl From<std::io::Error> for SpeechError {
    fn from(e: std::io::Error) -> Self {
        SpeechError::Io(e.to_string())
    }
}

/// Transcription options.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// ISO 639-1 language code; `None` = detect from the first 30 s.
    pub language: Option<String>,
    /// Label speakers ([`diarize`]); otherwise every word gets speaker 0.
    pub diarize: bool,
    /// Upper bound for the number of speakers found.
    pub max_speakers: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { language: None, diarize: true, max_speakers: 6 }
    }
}

/// Progress callback: `(fraction 0..1, status)`; return `false` to cancel.
pub type ProgressFn<'a> = &'a mut dyn FnMut(f32, &str) -> bool;

/// A speech recogniser.
pub trait Transcriber: Send + Sync {
    /// Stable id of the model (`"whisper-base"`), stored in the transcript's `source`.
    fn id(&self) -> String;
    /// Transcribe mono 16 kHz `audio`. Word times are relative to the first sample.
    fn transcribe(&self, audio: &[f32], opts: &Options, progress: ProgressFn) -> Result<Transcript, SpeechError>;
    /// [`Transcriber::transcribe`] of the speech in `chunks` only (sample ranges of `audio`, each
    /// at most 30 s: [`vad::chunks`]); word times stay times in `audio`. The default transcribes
    /// the whole audio.
    fn transcribe_chunks(&self, audio: &[f32], chunks: &[std::ops::Range<usize>], opts: &Options, progress: ProgressFn) -> Result<Transcript, SpeechError> {
        let _ = chunks;
        self.transcribe(audio, opts, progress)
    }
    /// The words of mono 16 kHz `audio` as plain text in `language` (ISO 639-1; empty or `auto`:
    /// detected), for checking what a voice said: no word times, no speakers. Whisper decodes
    /// without timestamps, as models fine-tuned without them read best.
    fn hear(&self, audio: &[f32], language: &str) -> Result<String, SpeechError> {
        let opts = Options { language: Some(language.to_string()).filter(|l| !l.is_empty() && l != "auto"), diarize: false, ..Default::default() };
        self.transcribe(audio, &opts, &mut |_, _| true).map(|t| t.text())
    }
}

/// A transcriber that returns a fixed transcript (tests, demos and agents that bring their own
/// transcript); words beyond the audio's length are dropped.
pub struct FixedTranscriber {
    pub transcript: Transcript,
    pub id: String,
}

impl Transcriber for FixedTranscriber {
    fn id(&self) -> String {
        self.id.clone()
    }
    fn transcribe(&self, audio: &[f32], _: &Options, progress: ProgressFn) -> Result<Transcript, SpeechError> {
        if !progress(0.5, "Transcribing") {
            return Err(SpeechError::Cancelled);
        }
        let end = sample_tick(audio.len() as i64);
        let mut t = self.transcript.clone();
        t.words.retain(|w| w.start < end);
        t.source = self.id.clone();
        progress(1.0, "Done");
        Ok(t)
    }
}

/// Load the transcriber for catalogue model `id` from `models_dir` (feature `whisper`).
pub fn load(models_dir: &std::path::Path, id: &str) -> Result<std::sync::Arc<dyn Transcriber>, SpeechError> {
    load_on(models_dir, id, false)
}

/// [`load`], on the GPU when `gpu` and the machine has one (Metal on macOS): the strict check of
/// narration hears on the GPU it already holds.
pub fn load_on(models_dir: &std::path::Path, id: &str, gpu: bool) -> Result<std::sync::Arc<dyn Transcriber>, SpeechError> {
    let m = models::find(id).ok_or_else(|| SpeechError::UnknownModel(id.into()))?;
    if !models::installed(models_dir, m) {
        return Err(SpeechError::NotInstalled(id.into()));
    }
    #[cfg(feature = "whisper")]
    {
        Ok(std::sync::Arc::new(whisper::Whisper::load_on(&models::model_dir(models_dir, m), m.id, gpu)?))
    }
    #[cfg(not(feature = "whisper"))]
    {
        let _ = gpu;
        Err(SpeechError::Unavailable("built without the `whisper` feature".into()))
    }
}

/// [`load`] on the GPU where the machine has one (Metal on macOS; `FILMCRAFT_SPEECH_CPU` keeps
/// the CPU), the CPU being the fallback: a model the GPU cannot load is loaded on the CPU, and a
/// transcription that fails or panics on the GPU is run again on the CPU (which then keeps the
/// model for the rest of the session). Whisper-base transcribes about 9× faster on an M3 Max GPU
/// with the same words and word times (`docs/gpu.md`).
pub fn load_gpu_first(models_dir: &std::path::Path, id: &str) -> Result<std::sync::Arc<dyn Transcriber>, SpeechError> {
    let gpu = match load_on(models_dir, id, true) {
        Ok(t) => Some(t),
        Err(e @ (SpeechError::UnknownModel(_) | SpeechError::NotInstalled(_) | SpeechError::Unavailable(_))) => return Err(e),
        Err(_) => None,
    };
    Ok(std::sync::Arc::new(GpuFirst {
        id: id.to_string(),
        dir: models_dir.to_path_buf(),
        gpu,
        failed: std::sync::atomic::AtomicBool::new(false),
        cpu: std::sync::Mutex::new(None),
    }))
}

/// A transcriber on the GPU with the CPU as its fallback ([`load_gpu_first`]).
struct GpuFirst {
    id: String,
    dir: std::path::PathBuf,
    gpu: Option<std::sync::Arc<dyn Transcriber>>,
    /// The GPU failed once: the CPU from then on.
    failed: std::sync::atomic::AtomicBool,
    cpu: std::sync::Mutex<Option<std::sync::Arc<dyn Transcriber>>>,
}

impl GpuFirst {
    fn cpu(&self) -> Result<std::sync::Arc<dyn Transcriber>, SpeechError> {
        let mut g = self.cpu.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(t) = g.as_ref() {
            return Ok(t.clone());
        }
        let t = load_on(&self.dir, &self.id, false)?;
        *g = Some(t.clone());
        Ok(t)
    }

    /// `f` with the GPU model, or `None` for the caller to use the CPU's: no GPU model, it failed
    /// before, or it fails (an error other than being stopped) or panics now.
    fn on_gpu<T>(&self, f: impl FnOnce(&dyn Transcriber) -> Result<T, SpeechError>) -> Option<Result<T, SpeechError>> {
        use std::sync::atomic::Ordering;
        let g = self.gpu.as_ref().filter(|_| !self.failed.load(Ordering::Relaxed))?;
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(g.as_ref()))) {
            Ok(Ok(v)) => Some(Ok(v)),
            Ok(Err(SpeechError::Cancelled)) => Some(Err(SpeechError::Cancelled)),
            _ => {
                self.failed.store(true, Ordering::Relaxed);
                None
            }
        }
    }
}

impl Transcriber for GpuFirst {
    fn id(&self) -> String {
        self.id.clone()
    }
    fn transcribe(&self, audio: &[f32], opts: &Options, progress: ProgressFn) -> Result<Transcript, SpeechError> {
        if let Some(r) = self.on_gpu(|t| t.transcribe(audio, opts, &mut *progress)) {
            return r;
        }
        self.cpu()?.transcribe(audio, opts, progress)
    }
    fn hear(&self, audio: &[f32], language: &str) -> Result<String, SpeechError> {
        if let Some(r) = self.on_gpu(|t| t.hear(audio, language)) {
            return r;
        }
        self.cpu()?.hear(audio, language)
    }
    fn transcribe_chunks(&self, audio: &[f32], chunks: &[std::ops::Range<usize>], opts: &Options, progress: ProgressFn) -> Result<Transcript, SpeechError> {
        if let Some(r) = self.on_gpu(|t| t.transcribe_chunks(audio, chunks, opts, &mut *progress)) {
            return r;
        }
        self.cpu()?.transcribe_chunks(audio, chunks, opts, progress)
    }
}

/// Whether this build can run speech models.
pub const fn available() -> bool {
    cfg!(feature = "whisper")
}

/// Mix a multi-channel buffer down to mono.
pub fn downmix(channels: &[Vec<f32>]) -> Vec<f32> {
    let n = channels.iter().map(Vec::len).min().unwrap_or(0);
    if channels.len() == 1 {
        return channels[0][..n].to_vec();
    }
    let k = 1.0 / channels.len().max(1) as f32;
    (0..n).map(|i| channels.iter().map(|c| c[i]).sum::<f32>() * k).collect()
}

/// Word error rate between a reference and a hypothesis (words compared case-insensitively
/// without punctuation, hyphenated words split): (substitutions + deletions + insertions) / reference words.
pub fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let norm = |s: &str| -> Vec<String> {
        s.split(|c: char| c.is_whitespace() || c == '-').map(filmcraft_project::transcript::normalize_word).filter(|w| !w.is_empty()).collect()
    };
    let (r, h) = (norm(reference), norm(hypothesis));
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for i in 1..=r.len() {
        let mut cur = vec![i; h.len() + 1];
        for j in 1..=h.len() {
            let sub = prev[j - 1] + usize::from(r[i - 1] != h[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        prev = cur;
    }
    prev[h.len()] as f64 / r.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_project::Word;

    #[test]
    fn sample_ticks_are_exact() {
        assert_eq!(TICKS_PER_SAMPLE * SAMPLE_RATE as i64, TICKS_PER_SECOND);
        assert_eq!(seconds_tick(1.5), Tick(TICKS_PER_SECOND * 3 / 2));
    }

    #[test]
    fn wer_counts_edits() {
        assert_eq!(word_error_rate("the cat sat", "The cat, sat."), 0.0);
        assert!((word_error_rate("the cat sat on the mat", "the cat sat on mat") - 1.0 / 6.0).abs() < 1e-9);
        assert!((word_error_rate("a b", "a x b y") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fixed_transcriber_trims_and_cancels() {
        let mut t = Transcript::default();
        t.words.push(Word::new("hi", seconds_tick(0.1), seconds_tick(0.3)));
        t.words.push(Word::new("late", seconds_tick(5.0), seconds_tick(5.5)));
        let f = FixedTranscriber { transcript: t, id: "fixed".into() };
        let out = f.transcribe(&vec![0.0; 16_000], &Options::default(), &mut |_, _| true).unwrap();
        assert_eq!(out.words.len(), 1);
        assert_eq!(out.source, "fixed");
        assert_eq!(f.transcribe(&[], &Options::default(), &mut |_, _| false), Err(SpeechError::Cancelled));
    }

    #[test]
    fn downmix_averages() {
        assert_eq!(downmix(&[vec![1.0, 0.0], vec![0.0, 0.0, 9.0]]), vec![0.5, 0.0]);
    }

    /// A GPU transcriber that fails, panics or is stopped.
    struct Broken(&'static str);

    impl Transcriber for Broken {
        fn id(&self) -> String {
            "broken".into()
        }
        fn transcribe(&self, _: &[f32], _: &Options, _: ProgressFn) -> Result<Transcript, SpeechError> {
            match self.0 {
                "panic" => panic!("a GPU kernel went wrong"),
                "stop" => Err(SpeechError::Cancelled),
                _ => Err(SpeechError::Model("no kernel for this op on the GPU".into())),
            }
        }
    }

    /// The GPU-first transcriber falls back to the CPU's model when the GPU fails or panics (and
    /// keeps to it), but a stop is a stop.
    #[test]
    fn gpu_failures_fall_back_to_the_cpu() {
        let mut t = Transcript::default();
        t.words.push(Word::new("hi", seconds_tick(0.1), seconds_tick(0.3)));
        let cpu: std::sync::Arc<dyn Transcriber> = std::sync::Arc::new(FixedTranscriber { transcript: t, id: "cpu".into() });
        for how in ["error", "panic", "stop"] {
            let g = GpuFirst {
                id: "whisper-base".into(),
                dir: std::path::PathBuf::new(),
                gpu: Some(std::sync::Arc::new(Broken(how))),
                failed: std::sync::atomic::AtomicBool::new(false),
                cpu: std::sync::Mutex::new(Some(cpu.clone())),
            };
            let got = g.transcribe(&vec![0.0; 16_000], &Options::default(), &mut |_, _| true);
            if how == "stop" {
                assert_eq!(got, Err(SpeechError::Cancelled));
                assert!(!g.failed.load(std::sync::atomic::Ordering::Relaxed));
                continue;
            }
            assert_eq!(got.map(|t| t.source), Ok("cpu".to_string()), "{how}");
            assert!(g.failed.load(std::sync::atomic::Ordering::Relaxed), "{how}: the CPU from now on");
            assert_eq!(g.hear(&vec![0.0; 16_000], "en"), Ok("hi".to_string()));
        }
    }
}
