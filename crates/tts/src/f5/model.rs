//! An IndicF5 release ready to speak: the reference clip, the duration, and the flow-matching
//! sampler.
//!
//! As the reference (`f5_tts/infer/utils_infer.py`, `f5_tts/model/cfm.py`):
//!
//! - **Reference**: mono 24 kHz raised to an RMS of 0.1 when quieter (the output is scaled back
//!   by the same factor), its mel ([`super::mel`]), and its transcript ending in `". "` plus the
//!   reference's extra space after an ASCII last character.
//! - **Text**: the transcript and the text to say as one string, tokenized ([`super::vocab`]).
//! - **Duration**: the reference's frames (`samples / 256`) plus as many again as the text is
//!   longer than the transcript, divided by the speed. The reference measures the two texts in
//!   UTF-8 bytes; that is their character ratio when both are in one script, but an English
//!   sample speaking Malayalam would then speak three times too slowly (a Malayalam letter is
//!   three bytes), so characters are counted ([`spoken_len`]). At least one frame more than the
//!   reference and the text tokens; at most 4096 (the text table's ~44 s).
//! - **Sampling**: Gaussian noise (seeded), Euler steps on the sway schedule
//!   `t + s·(cos(πt/2) − 1 + t)` (`s = −1`), classifier-free guidance `v + (v − v∅)·cfg` with the
//!   unconditioned pass dropping both the reference audio and the text; the reference frames
//!   are put back over the result, the rest is decoded by Vocos.

use std::path::Path;

use candle_core::{DType, Device, Tensor};

use super::config::Config;
use super::dit::{Dit, TEXT_POSITIONS};
use super::vocab::Vocab;
use super::vocos::Vocos;
use super::{HOP, MELS, RATE, Settings, mel};
use crate::TtsError;
use crate::voxcpm::weights::SafeTensors;
use crate::voxcpm::{Loader, Noise, R, err};

/// The reference raises quieter references to this RMS.
const TARGET_RMS: f32 = 0.1;
/// Reference clip length accepted, seconds.
pub const REFERENCE_SECONDS: (f64, f64) = (1.0, 30.0);
/// The reference's chunking aims at this much audio, reference included, per generation.
pub const TOTAL_SECONDS: f64 = 25.0;
/// Longest transcript accepted.
const MAX_REFERENCE_TEXT: usize = 2000;

/// A reference clip prepared for speaking.
#[derive(Clone, Debug)]
pub struct Reference {
    /// `[frames][MELS]` log mel of the (RMS-raised) clip.
    pub(crate) mel: Vec<f32>,
    pub(crate) frames: usize,
    /// Whole hops in the clip (`samples / 256`): where the generated frames start.
    pub(crate) audio_len: usize,
    /// The clip's RMS before raising (the output is scaled back by it).
    pub(crate) rms: f32,
    /// The transcript as the reference prepares it.
    pub(crate) text: String,
}

impl Reference {
    pub fn seconds(&self) -> f64 {
        (self.audio_len * HOP) as f64 / f64::from(RATE)
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// The longest text (characters) one generation takes after this reference: the speaking
    /// rate of the transcript times what is left of [`TOTAL_SECONDS`] (the reference's
    /// `chunk_text` limit), at least 40.
    pub fn max_chars(&self) -> usize {
        let secs = self.seconds().max(0.1);
        let rate = spoken_len(&self.text) as f64 / secs;
        ((rate * (TOTAL_SECONDS - secs).max(5.0)) as usize).max(40)
    }
}

/// The length the duration is estimated from: characters (see the module docs).
pub fn spoken_len(text: &str) -> usize {
    text.chars().count()
}

/// The transcript as the reference uses it: ending in `". "` (or `"。"`), then one more space
/// when its last character is ASCII.
fn reference_text(t: &str) -> String {
    let mut s = t.trim().to_string();
    if !s.ends_with(". ") && !s.ends_with('。') {
        if s.ends_with('.') {
            s.push(' ');
        } else {
            s.push_str(". ");
        }
    }
    if s.chars().last().is_some_and(|c| c.is_ascii()) {
        s.push(' ');
    }
    s
}

/// `torch.linspace(0, 1, steps + 1)` in f32 (filled from both ends).
fn linspace(steps: usize) -> Vec<f32> {
    let n = steps + 1;
    let step = 1.0f32 / steps.max(1) as f32;
    (0..n).map(|i| if i < n / 2 { step * i as f32 } else { 1.0 - step * (n - 1 - i) as f32 }).collect()
}

/// The sampler's time grid on the sway schedule.
pub(crate) fn time_grid(steps: usize, sway: f32) -> Vec<f32> {
    linspace(steps)
        .into_iter()
        .map(|t| {
            let c = f64::from(std::f32::consts::FRAC_PI_2 * t).cos() as f32;
            t + sway * (c - 1.0 + t)
        })
        .collect()
}

pub struct F5 {
    cfg: Config,
    dit: Dit,
    vocos: Vocos,
    vocab: Vocab,
    fb: Vec<f32>,
    dev: Device,
}

impl F5 {
    /// Load the release in `dir` on the default device ([`super::device`]).
    pub fn load(dir: &Path) -> R<F5> {
        Self::load_on(dir, &super::device())
    }

    pub fn load_on(dir: &Path, dev: &Device) -> R<F5> {
        let vocab = Vocab::read(&dir.join("vocab.txt"))?;
        let mut st = SafeTensors::open(&dir.join("model.safetensors"))?;
        let cfg = Config::read(&st)?;
        if vocab.len() + 1 != cfg.text_rows {
            return Err(err(format!("vocab.txt has {} tokens; the model expects {}", vocab.len(), cfg.text_rows.saturating_sub(1))));
        }
        let wdtype = match std::env::var("F5_WDTYPE").as_deref() {
            Ok("bf16") if dev.is_metal() => DType::BF16,
            Ok("f16") if dev.is_metal() => DType::F16,
            _ => DType::F32,
        };
        let mut ld = Loader { st: &mut st, dev, wdtype };
        let dit = Dit::load(&mut ld, &cfg)?;
        let vocos = Vocos::load(&mut ld, &cfg)?;
        Ok(F5 { cfg, dit, vocos, vocab, fb: mel::filterbank(), dev: dev.clone() })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// Prepare `audio` (mono, 24 kHz) whose words are `transcript` as the voice to speak in.
    pub fn reference(&self, audio: &[f32], transcript: &str) -> R<Reference> {
        let secs = audio.len() as f64 / f64::from(RATE);
        if !(REFERENCE_SECONDS.0..=REFERENCE_SECONDS.1).contains(&secs) {
            return Err(TtsError::Invalid(format!("the reference clip is {secs:.1} s; {}–{} s is needed", REFERENCE_SECONDS.0, REFERENCE_SECONDS.1)));
        }
        if audio.iter().any(|v| !v.is_finite()) {
            return Err(TtsError::Invalid("the reference clip has non-finite samples".into()));
        }
        if !transcript.chars().any(char::is_alphanumeric) || transcript.chars().count() > MAX_REFERENCE_TEXT {
            return Err(TtsError::Invalid("the reference clip needs its transcript (at most 2000 characters)".into()));
        }
        let rms = (audio.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / audio.len().max(1) as f64).sqrt() as f32;
        if rms < 1e-5 {
            return Err(TtsError::Invalid("the reference clip is silent".into()));
        }
        let x: Vec<f32> = if rms < TARGET_RMS { audio.iter().map(|v| v * TARGET_RMS / rms).collect() } else { audio.to_vec() };
        let (mel, frames) = mel::mel(&x, &self.fb);
        Ok(Reference { mel, frames, audio_len: audio.len() / HOP, rms, text: reference_text(transcript) })
    }

    /// The frames to generate for `text` after `r` at `speed` (the reference's formula), and
    /// the tokens of transcript and text together.
    pub(crate) fn plan(&self, r: &Reference, text: &str, speed: f32) -> R<(Vec<u32>, usize, Vec<char>)> {
        let enc = self.vocab.encode(&format!("{}{text}", r.text));
        let (ref_len, gen_len) = (spoken_len(&r.text).max(1), spoken_len(text));
        let speed = if speed.is_finite() && speed > 0.0 { f64::from(speed) } else { 1.0 };
        let wanted = r.audio_len as f64 + (r.audio_len as f64 / ref_len as f64 * gen_len as f64 / speed).floor();
        let lens = enc.ids.len().max(r.frames);
        let duration = (wanted as usize).max(lens + 1);
        if duration > TEXT_POSITIONS {
            return Err(TtsError::Invalid(format!(
                "the text is too long to say in one go after a {:.0} s reference (at most {} characters)",
                r.seconds(),
                r.max_chars()
            )));
        }
        Ok((enc.ids, duration, enc.unknown))
    }

    /// Speak `text` in the voice of `r`: mono 24 kHz. `progress(step, steps)` is called after
    /// each sampling step; returning `false` stops with an error.
    pub fn speak(&self, r: &Reference, text: &str, s: &Settings, progress: &mut dyn FnMut(usize, usize) -> bool) -> R<Vec<f32>> {
        s.check()?;
        if !text.chars().any(char::is_alphanumeric) {
            return Err(TtsError::Empty);
        }
        let (tokens, duration, _) = self.plan(r, text, s.speed)?;
        let noise = Noise::new(s.seed).take(duration * MELS);
        let mel = self.sample(r, &tokens, duration, &noise, s, None, progress)?;
        self.vocode(r, &mel, duration)
    }

    /// The mel after the reference (from its last whole hop), decoded, scaled back to the
    /// reference's level.
    pub(crate) fn vocode(&self, r: &Reference, mel: &[f32], duration: usize) -> R<Vec<f32>> {
        let from = r.audio_len.min(duration);
        let out = mel.get(from * MELS..duration * MELS).ok_or_else(|| err("the generated mel is short"))?;
        let mut audio = self.vocos.decode(out, duration - from)?;
        if r.rms < TARGET_RMS {
            let g = r.rms / TARGET_RMS;
            audio.iter_mut().for_each(|v| *v *= g);
        }
        Ok(audio)
    }

    /// Flow-matching sampling of `duration` frames from `noise` (`[duration][MELS]`): the whole
    /// mel, the reference's frames put back. `trace(step, y)` sees the state after each step.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample(
        &self,
        r: &Reference,
        tokens: &[u32],
        duration: usize,
        noise: &[f32],
        s: &Settings,
        mut trace: Option<&mut dyn FnMut(usize, &[f32])>,
        progress: &mut dyn FnMut(usize, usize) -> bool,
    ) -> R<Vec<f32>> {
        if noise.len() != duration * MELS || duration > TEXT_POSITIONS || duration <= r.frames {
            return Err(err("bad sampling request"));
        }
        // the conditioning: the reference mel over the first max(text, reference) frames
        let lens = tokens.len().max(r.frames).min(duration);
        let mut cond = vec![0f32; duration * MELS];
        if let (Some(dst), Some(src)) = (cond.get_mut(..r.frames * MELS), r.mel.get(..r.frames * MELS)) {
            dst.copy_from_slice(src);
        }
        let ctx = self.dit.context(&cond, tokens, duration)?;
        let grid = time_grid(s.steps, s.sway);
        let mut y = Tensor::from_vec(noise.to_vec(), (1, duration, MELS), &self.dev).map_err(err)?;
        let steps = grid.len().saturating_sub(1);
        for (i, w) in grid.windows(2).enumerate() {
            let (Some(t0), Some(t1)) = (w.first(), w.get(1)) else { break };
            let pred = self.dit.forward(&y, *t0, &ctx)?;
            let c = pred.narrow(0, 0, 1).map_err(err)?;
            let u = pred.narrow(0, 1, 1).map_err(err)?;
            let v = if s.cfg < 1e-5 { c } else { (&c + ((&c - &u).map_err(err)? * f64::from(s.cfg)).map_err(err)?).map_err(err)? };
            y = (y + (v * f64::from(t1 - t0)).map_err(err)?).map_err(err)?;
            if !progress(i + 1, steps) {
                return Err(TtsError::Invalid("stopped".into()));
            }
            if let Some(t) = trace.as_deref_mut() {
                t(i + 1, &to_vec(&y)?);
            }
        }
        let mut out = to_vec(&y)?;
        if out.iter().any(|v| !v.is_finite()) {
            return Err(err("sampling produced non-finite values"));
        }
        if let (Some(dst), Some(src)) = (out.get_mut(..lens * MELS), cond.get(..lens * MELS)) {
            dst.copy_from_slice(src);
        }
        Ok(out)
    }
}

fn to_vec(t: &Tensor) -> R<Vec<f32>> {
    t.to_device(&Device::Cpu).map_err(err)?.flatten_all().map_err(err)?.to_vec1::<f32>().map_err(err)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The vocabulary of the fake release: a space, ASCII letters and punctuation, and the
    /// Malayalam letters (so the front end's text all tokenizes).
    pub(crate) fn fake_vocab() -> String {
        let mut v: Vec<char> = vec![' ', '!', '"', '\'', ',', '.', '?', ':', '-'];
        v.extend('a'..='z');
        v.extend('A'..='Z');
        v.extend((0x0D02u32..=0x0D7F).filter_map(char::from_u32));
        v.extend(['\u{200c}', '\u{200d}']);
        v.iter().map(|c| format!("{c}\n")).collect()
    }

    /// A tiny random IndicF5 release (2 blocks of width 128, a 2-block vocoder) in `dir`.
    pub(crate) fn fake_release(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let vocab = fake_vocab();
        std::fs::write(dir.join("vocab.txt"), &vocab).unwrap();
        let rows = vocab.lines().count() + 1;
        let mut rng = Noise::new(3);
        let mut out: Vec<(String, Tensor)> = Vec::new();
        let mut put = |name: String, shape: &[usize], scale: f32| {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = rng.take(n).into_iter().map(|x| x * scale).collect();
            out.push((name, Tensor::from_vec(v, shape, &Device::Cpu).unwrap()));
        };
        let (d, td, ff, tf) = (128usize, 32usize, 256usize, 64usize);
        let p = "ema_model._orig_mod.transformer.";
        put(format!("{p}text_embed.text_embed.weight"), &[rows, td], 0.5);
        for i in 0..2 {
            let b = format!("{p}text_embed.text_blocks.{i}");
            put(format!("{b}.dwconv.weight"), &[td, 1, 7], 0.2);
            put(format!("{b}.dwconv.bias"), &[td], 0.1);
            put(format!("{b}.norm.weight"), &[td], 0.1);
            put(format!("{b}.norm.bias"), &[td], 0.1);
            put(format!("{b}.pwconv1.weight"), &[tf, td], 0.1);
            put(format!("{b}.pwconv1.bias"), &[tf], 0.1);
            put(format!("{b}.grn.gamma"), &[1, 1, tf], 0.1);
            put(format!("{b}.grn.beta"), &[1, 1, tf], 0.1);
            put(format!("{b}.pwconv2.weight"), &[td, tf], 0.1);
            put(format!("{b}.pwconv2.bias"), &[td], 0.1);
        }
        put(format!("{p}time_embed.time_mlp.0.weight"), &[d, 256], 0.05);
        put(format!("{p}time_embed.time_mlp.0.bias"), &[d], 0.05);
        put(format!("{p}time_embed.time_mlp.2.weight"), &[d, d], 0.05);
        put(format!("{p}time_embed.time_mlp.2.bias"), &[d], 0.05);
        put(format!("{p}input_embed.proj.weight"), &[d, 2 * MELS + td], 0.05);
        put(format!("{p}input_embed.proj.bias"), &[d], 0.05);
        for c in [0, 2] {
            put(format!("{p}input_embed.conv_pos_embed.conv1d.{c}.weight"), &[d, d / 16, 31], 0.05);
            put(format!("{p}input_embed.conv_pos_embed.conv1d.{c}.bias"), &[d], 0.05);
        }
        let inv: Vec<f32> = (0..32).map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / 64.0)).collect();
        out.push((format!("{p}rotary_embed.inv_freq"), Tensor::from_vec(inv, 32, &Device::Cpu).unwrap()));
        let mut put = |name: String, shape: &[usize], scale: f32| {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = rng.take(n).into_iter().map(|x| x * scale).collect();
            out.push((name, Tensor::from_vec(v, shape, &Device::Cpu).unwrap()));
        };
        for i in 0..2 {
            let b = format!("{p}transformer_blocks.{i}");
            put(format!("{b}.attn_norm.linear.weight"), &[6 * d, d], 0.05);
            put(format!("{b}.attn_norm.linear.bias"), &[6 * d], 0.05);
            for n in ["to_q", "to_k", "to_v", "to_out.0"] {
                put(format!("{b}.attn.{n}.weight"), &[d, d], 0.08);
                put(format!("{b}.attn.{n}.bias"), &[d], 0.05);
            }
            put(format!("{b}.ff.ff.0.0.weight"), &[ff, d], 0.08);
            put(format!("{b}.ff.ff.0.0.bias"), &[ff], 0.05);
            put(format!("{b}.ff.ff.2.weight"), &[d, ff], 0.08);
            put(format!("{b}.ff.ff.2.bias"), &[d], 0.05);
        }
        put(format!("{p}norm_out.linear.weight"), &[2 * d, d], 0.05);
        put(format!("{p}norm_out.linear.bias"), &[2 * d], 0.05);
        put(format!("{p}proj_out.weight"), &[MELS, d], 0.05);
        put(format!("{p}proj_out.bias"), &[MELS], 0.05);
        let v = "vocoder._orig_mod.";
        let (vd, vf) = (32usize, 64usize);
        put(format!("{v}backbone.embed.weight"), &[vd, MELS, 7], 0.02);
        put(format!("{v}backbone.embed.bias"), &[vd], 0.02);
        for n in ["norm", "final_layer_norm"] {
            put(format!("{v}backbone.{n}.weight"), &[vd], 0.1);
            put(format!("{v}backbone.{n}.bias"), &[vd], 0.1);
        }
        for i in 0..2 {
            let b = format!("{v}backbone.convnext.{i}");
            put(format!("{b}.dwconv.weight"), &[vd, 1, 7], 0.2);
            put(format!("{b}.dwconv.bias"), &[vd], 0.1);
            put(format!("{b}.norm.weight"), &[vd], 0.1);
            put(format!("{b}.norm.bias"), &[vd], 0.1);
            put(format!("{b}.pwconv1.weight"), &[vf, vd], 0.1);
            put(format!("{b}.pwconv1.bias"), &[vf], 0.1);
            put(format!("{b}.pwconv2.weight"), &[vd, vf], 0.1);
            put(format!("{b}.pwconv2.bias"), &[vd], 0.1);
            put(format!("{b}.gamma"), &[vd], 0.1);
        }
        put(format!("{v}head.out.weight"), &[1026, vd], 0.05);
        put(format!("{v}head.out.bias"), &[1026], 0.05);
        crate::voxcpm::weights::write(&dir.join("model.safetensors"), &out);
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("filmcraft-f5-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// A clip with some "speech": a few seconds of two tones.
    pub(crate) fn tone(secs: f32) -> Vec<f32> {
        (0..(secs * RATE as f32) as usize).map(|i| ((i as f32 * 0.031).sin() + (i as f32 * 0.0071).sin()) * 0.2).collect()
    }

    #[test]
    fn a_tiny_release_speaks_deterministically() {
        let d = tmp("speak");
        fake_release(&d);
        let m = F5::load_on(&d, &Device::Cpu).unwrap();
        assert_eq!((m.config().dim, m.config().depth, m.config().heads, m.config().voc_blocks), (128, 2, 2, 2));
        let r = m.reference(&tone(2.0), "hello there").unwrap();
        assert_eq!(r.text(), "hello there.  ");
        assert_eq!(r.frames, 2 * RATE as usize / HOP + 1);
        let s = Settings { steps: 4, ..Settings::default() };
        let a = m.speak(&r, "ഇത് ശരിയാണ്.", &s, &mut |_, _| true).unwrap();
        assert!(!a.is_empty() && a.iter().all(|v| v.is_finite()));
        // the duration follows the transcript's rate: 12 characters after 14 in 2 s
        let (_, duration, unknown) = m.plan(&r, "ഇത് ശരിയാണ്.", 1.0).unwrap();
        assert!(unknown.is_empty());
        assert_eq!(duration, r.audio_len + r.audio_len * 12 / 14);
        assert_eq!(a.len(), (duration - r.audio_len - 1) * HOP);
        assert_eq!(m.speak(&r, "ഇത് ശരിയാണ്.", &s, &mut |_, _| true).unwrap(), a, "deterministic");
        let other = m.speak(&r, "ഇത് ശരിയാണ്.", &Settings { seed: 2, ..s }, &mut |_, _| true).unwrap();
        assert_ne!(other, a, "another seed reads it afresh");
        // faster speech is shorter
        let fast = m.speak(&r, "ഇത് ശരിയാണ്.", &Settings { speed: 2.0, ..s }, &mut |_, _| true).unwrap();
        assert!(fast.len() < a.len());
        // stopping, nothing to say, bad settings
        assert!(m.speak(&r, "ഇത്", &s, &mut |_, _| false).is_err());
        assert_eq!(m.speak(&r, "  ...", &s, &mut |_, _| true).err(), Some(TtsError::Empty));
        assert!(m.speak(&r, "ഇത്", &Settings { steps: 0, ..s }, &mut |_, _| true).is_err());
        // more than the model's 4096 frames
        assert!(m.speak(&r, &"ഇത് ശരിയാണ്. ".repeat(200), &s, &mut |_, _| true).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn hostile_references_are_refused() {
        let d = tmp("refs");
        fake_release(&d);
        let m = F5::load_on(&d, &Device::Cpu).unwrap();
        assert!(m.reference(&tone(0.5), "hi").is_err(), "too short");
        assert!(m.reference(&tone(31.0), "hi").is_err(), "too long");
        assert!(m.reference(&vec![0.0; RATE as usize * 2], "hi").is_err(), "silent");
        assert!(m.reference(&vec![f32::NAN; RATE as usize * 2], "hi").is_err(), "NaN");
        assert!(m.reference(&tone(2.0), "  ").is_err(), "no transcript");
        assert!(m.reference(&tone(2.0), &"a".repeat(3000)).is_err(), "transcript too long");
        // a quiet reference is raised to the reference RMS; the voice is put back at its level
        let quiet: Vec<f32> = tone(2.0).iter().map(|v| v * 0.01).collect();
        let r = m.reference(&quiet, "hello").unwrap();
        assert!(r.rms < 0.01);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn broken_releases_are_errors() {
        let d = tmp("broken");
        fake_release(&d);
        // a vocabulary that does not match the embedding
        std::fs::write(d.join("vocab.txt"), " \na\nb\n").unwrap();
        assert!(F5::load_on(&d, &Device::Cpu).is_err());
        std::fs::remove_file(d.join("vocab.txt")).unwrap();
        assert!(F5::load_on(&d, &Device::Cpu).is_err());
        fake_release(&d);
        // weights that are not IndicF5's
        crate::voxcpm::weights::write(&d.join("model.safetensors"), &[("x".into(), Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap())]);
        assert!(F5::load_on(&d, &Device::Cpu).is_err());
        std::fs::write(d.join("model.safetensors"), b"not a safetensors file").unwrap();
        assert!(F5::load_on(&d, &Device::Cpu).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The GPU path (fused attention, batched rotary tables) equals the CPU's (regression: the
    /// fused attention misread a value tensor that did not start its buffer).
    #[test]
    fn the_gpu_matches_the_cpu() {
        let Ok(gpu) = Device::new_metal(0) else { return };
        let d = tmp("gpu");
        fake_release(&d);
        let (a, b) = (F5::load_on(&d, &Device::Cpu).unwrap(), F5::load_on(&d, &gpu).unwrap());
        let r = a.reference(&tone(2.0), "hello there").unwrap();
        let (tokens, duration, _) = a.plan(&r, "ഇത് ശരിയാണ്.", 1.0).unwrap();
        let noise = Noise::new(5).take(duration * MELS);
        let s = Settings { steps: 3, ..Settings::default() };
        let x = a.sample(&r, &tokens, duration, &noise, &s, None, &mut |_, _| true).unwrap();
        let y = b.sample(&r, &tokens, duration, &noise, &s, None, &mut |_, _| true).unwrap();
        let (dmax, mag, _) = max_diff(&y, &x);
        assert!(dmax < 1e-3 * mag.max(1.0), "{dmax} (max {mag})");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn reference_text_ends_like_the_reference() {
        assert_eq!(reference_text("Hello there"), "Hello there.  ");
        assert_eq!(reference_text("Hello there."), "Hello there.  ");
        assert_eq!(reference_text(" Hello there. "), "Hello there.  ");
        assert_eq!(reference_text("ഇത് ശരിയാണ്"), "ഇത് ശരിയാണ്.  ");
        assert_eq!(reference_text("你好。"), "你好。");
    }

    fn read_f32(p: &Path) -> Vec<f32> {
        std::fs::read(p).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    }

    fn write_f32(p: &Path, x: &[f32]) {
        std::fs::write(p, x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    }

    /// Max |a − b|, max |b|, and the mean |a − b|.
    fn max_diff(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let mean = a.iter().zip(b).map(|(x, y)| f64::from((x - y).abs())).sum::<f64>() / a.len().max(1) as f64;
        (d, b.iter().map(|v| v.abs()).fold(0f32, f32::max), mean as f32)
    }

    /// Against the PyTorch reference run as an oracle with the same noise (gated). With a
    /// release in `FILMCRAFT_F5_DIR` and a work directory `F5_ORACLE` holding `ref.wav` (any
    /// rate) and `meta.json` (`refText`, `text`, `steps`): the first run writes the inputs
    /// (`ref.f32` at 24 kHz, `noise.f32`, `inputs.json`); `oracle/f5_oracle.py` then writes the
    /// reference's results (`py_*.f32`, `py_tokens.json`); the next run compares.
    #[test]
    #[ignore = "needs an IndicF5 release and the oracle's output"]
    fn live_matches_the_reference_oracle() {
        let (Some(dir), Some(work)) = (std::env::var_os("FILMCRAFT_F5_DIR"), std::env::var_os("F5_ORACLE")) else {
            eprintln!("skipped (FILMCRAFT_F5_DIR or F5_ORACLE unset)");
            return;
        };
        let work = std::path::PathBuf::from(work);
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(work.join("meta.json")).unwrap()).unwrap();
        let (ref_text, text) = (meta["refText"].as_str().unwrap(), meta["text"].as_str().unwrap());
        let steps = meta["steps"].as_u64().unwrap_or(32) as usize;
        let dev = super::super::device();
        let t0 = std::time::Instant::now();
        let m = F5::load_on(Path::new(&dir), &dev).unwrap();
        eprintln!("loaded on {:?} in {:.1} s", dev.location(), t0.elapsed().as_secs_f64());
        if !work.join("py_final.f32").exists() {
            let (x, rate) = crate::wav::read(&work.join("ref.wav")).unwrap();
            let x = filmcraft_audio_dsp::resample::resample_sinc(&x, rate, RATE);
            let r = m.reference(&x, ref_text).unwrap();
            let (tokens, duration, unknown) = m.plan(&r, text, 1.0).unwrap();
            write_f32(&work.join("ref.f32"), &x);
            write_f32(&work.join("noise.f32"), &Noise::new(1).take(duration * MELS));
            let inputs = serde_json::json!({"refText": r.text, "text": text, "duration": duration, "steps": steps, "tokens": tokens, "unknown": unknown});
            std::fs::write(work.join("inputs.json"), serde_json::to_vec_pretty(&inputs).unwrap()).unwrap();
            eprintln!("inputs written: {} frames ({} reference); now run oracle/f5_oracle.py", duration, r.frames);
            return;
        }
        let inputs: serde_json::Value = serde_json::from_slice(&std::fs::read(work.join("inputs.json")).unwrap()).unwrap();
        let x = read_f32(&work.join("ref.f32"));
        let r = m.reference(&x, ref_text).unwrap();
        let duration = inputs["duration"].as_u64().unwrap() as usize;
        let (tokens, _, _) = m.plan(&r, text, 1.0).unwrap();
        let py_tokens: Vec<u32> = serde_json::from_slice::<Vec<u32>>(&std::fs::read(work.join("py_tokens.json")).unwrap()).unwrap();
        eprintln!("tokens: {} ours, {} reference, equal: {}", tokens.len(), py_tokens.len(), tokens == py_tokens);
        let py_mel = read_f32(&work.join("py_mel.f32"));
        let (d, mag, _) = max_diff(&r.mel, &py_mel);
        // log mel near its floor magnifies rounding: also the error in linear magnitude
        let lin = r.mel.iter().zip(&py_mel).map(|(a, b)| (a.exp() - b.exp()).abs()).fold(0f32, f32::max);
        eprintln!("reference mel: max |diff| {d:.2e} (max |value| {mag:.2}), linear {lin:.2e}");
        write_f32(&work.join("rust_mel.f32"), &r.mel);
        if std::env::var_os("F5_ORACLE_MEL_ONLY").is_some() {
            return;
        }
        // the network from here on gets exactly the reference's inputs
        let mut r = r;
        r.mel = py_mel;
        let noise = read_f32(&work.join("noise.f32"));
        let s = Settings { steps, ..Settings::default() };
        let mut states: Vec<(usize, Vec<f32>)> = Vec::new();
        let t0 = std::time::Instant::now();
        let mel = m.sample(&r, &py_tokens, duration, &noise, &s, Some(&mut |i, y: &[f32]| states.push((i, y.to_vec()))), &mut |_, _| true).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        for (i, y) in &states {
            let p = work.join(format!("py_step{i}.f32"));
            if p.exists() {
                let (d, mag, mean) = max_diff(y, &read_f32(&p));
                eprintln!("after step {i}: max |diff| {d:.2e}, mean {mean:.2e} (max |value| {mag:.2})");
            }
        }
        let (d, mag, mean) = max_diff(&mel, &read_f32(&work.join("py_final.f32")));
        let gen_secs = ((duration - r.audio_len) * HOP) as f64 / f64::from(RATE);
        eprintln!("final mel: max |diff| {d:.2e}, mean {mean:.2e} (max |value| {mag:.2}); sampling {secs:.1} s for {gen_secs:.1} s of speech");
        write_f32(&work.join("rust_final.f32"), &mel);
        let py_final = read_f32(&work.join("py_final.f32"));
        let wave = m.vocode(&r, &py_final, duration).unwrap();
        let (d, mag, _) = max_diff(&wave, &read_f32(&work.join("py_wave.f32")));
        eprintln!("vocoder on the reference's mel: max |diff| {d:.2e} (max |value| {mag:.2}), {} samples", wave.len());
        let ours = m.vocode(&r, &mel, duration).unwrap();
        crate::wav::write(&work.join("rust.wav"), &ours, RATE).unwrap();
    }

    /// Where the time goes (gated): FILMCRAFT_F5_DIR=<release> cargo test --release -p
    /// filmcraft-tts --features indicf5 live_f5_parts -- --ignored --nocapture
    #[test]
    #[ignore = "needs an IndicF5 release"]
    fn live_f5_parts() {
        let Some(dir) = std::env::var_os("FILMCRAFT_F5_DIR") else { return };
        let dev = super::super::device();
        let m = F5::load_on(Path::new(&dir), &dev).unwrap();
        let n = std::env::var("F5_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(1300usize);
        let cond = vec![-5f32; n * MELS];
        let tokens: Vec<u32> = (0..150).map(|i| (i % 50) as u32 + 10).collect();
        let t = std::time::Instant::now();
        let ctx = m.dit.context(&cond, &tokens, n).unwrap();
        let sync = |t: &Tensor| t.flatten_all().unwrap().to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap().len();
        let _ = &ctx;
        eprintln!("{:?}, {n} frames: context {:.0} ms", dev.location(), t.elapsed().as_secs_f64() * 1000.0);
        let x = Tensor::from_vec(Noise::new(1).take(n * MELS), (1, n, MELS), &dev).unwrap();
        let _ = sync(&m.dit.forward(&x, 0.5, &ctx).unwrap());
        let reps = 4;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            let _ = sync(&m.dit.forward(&x, 0.5, &ctx).unwrap());
        }
        eprintln!("forward (both branches): {:.0} ms", t.elapsed().as_secs_f64() * 1000.0 / reps as f64);
        // the floor: the matrix products and the attention alone
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let a = Tensor::ones((2 * n, 1024), dt, &dev).unwrap();
            let w = Tensor::ones((3072, 1024), dt, &dev).unwrap();
            let _ = sync(&a.matmul(&w.t().unwrap()).unwrap().to_dtype(DType::F32).unwrap());
            let t = std::time::Instant::now();
            let mut last = None;
            for _ in 0..20 {
                last = Some(a.matmul(&w.t().unwrap()).unwrap());
            }
            let _ = sync(&last.unwrap().to_dtype(DType::F32).unwrap());
            let ms = t.elapsed().as_secs_f64() * 1000.0 / 20.0;
            eprintln!("{dt:?} [{}×1024]·[1024×3072]: {ms:.2} ms ({:.1} TFLOPS)", 2 * n, (2 * n * 1024 * 3072 * 2) as f64 / ms / 1e9);
        }
        let q = Tensor::ones((2, 16, n, 64), DType::F32, &dev).unwrap();
        let t = std::time::Instant::now();
        let mut last = None;
        for _ in 0..20 {
            last = Some(candle_nn::ops::sdpa(&q, &q, &q, None, false, 0.125, 1.0).unwrap());
        }
        let _ = sync(&last.unwrap());
        eprintln!("sdpa [2, 16, {n}, 64]: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0 / 20.0);
        let mel = vec![-3f32; 600 * MELS];
        let t = std::time::Instant::now();
        let w = m.vocos.decode(&mel, 600).unwrap();
        eprintln!("vocoder: {:.0} ms for {:.1} s", t.elapsed().as_secs_f64() * 1000.0, w.len() as f64 / 24_000.0);
    }

    #[test]
    fn the_time_grid_sways_towards_the_start() {
        let g = time_grid(32, -1.0);
        assert_eq!(g.len(), 33);
        assert_eq!((g[0], g[32]), (0.0, 1.0));
        // 1 − cos(πt/2): small steps first
        assert!((g[16] - (1.0 - (std::f32::consts::FRAC_PI_4).cos())).abs() < 1e-6);
        assert!(g.windows(2).all(|w| w[1] > w[0]));
        assert!(g[1] - g[0] < g[32] - g[31]);
        assert_eq!(time_grid(4, 0.0), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }
}
