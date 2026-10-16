//! The whole model: prompt layout, prefill and the autoregressive patch loop.
//!
//! **Sequence.** One position per text token or per latent patch. A text position carries its
//! token embedding, an audio position the LocEnc embedding of its patch; masks say which is which.
//! Continuation ("ultimate cloning"): `[tokens of prompt text + target text, <audio_start>,
//! prompt patches…]`, so generation continues the speaker mid-recording. VoxCPM2 can also put an
//! isolated reference in front: `[<ref_start>, reference patches…, <ref_end>, …]`.
//!
//! **Loop.** After the prefill, each step: the two LMs' last states → LocDiT samples the next
//! patch (conditioned on the previous one) → LocEnc embeds it → one step of the text-semantic LM
//! (its output scalar-quantized) and one of the residual LM. The stop head reads the LM state
//! before each step; the utterance is also capped at six patches per text token (plus ten), as
//! the reference does, and regenerated with the next seed when it runs into that cap.
//!
//! **Decode.** The last two (VoxCPM 1) or three (VoxCPM2) prompt patches go in front of the
//! generated ones so the causal decoder starts mid-speech, and their audio is cut off again.

use std::path::Path;

use candle_core::{D, Device, Tensor};

use super::config::{self, Config, Variant};
use super::local::{LocDit, LocEnc};
use super::minicpm::Stack;
use super::tokenizer::Tokenizer;
use super::vae::{Sig, Vae};
use super::{Linear, Loader, Noise, R, Settings, err, weights};
use crate::TtsError;

const AUDIO_START: u32 = 101;
const REF_START: u32 = 103;
const REF_END: u32 = 104;
/// The stop head is ignored for the first patches (the reference's `min_len`).
const MIN_PATCHES: usize = 2;
/// Patches per text token before an utterance counts as runaway (the reference's ratio).
const RATIO: usize = 6;
/// Shortest and longest prompt audio, seconds.
pub const MIN_PROMPT_SECONDS: f64 = 1.0;
pub const MAX_PROMPT_SECONDS: f64 = 40.0;

/// A speaker prompt: the clip's latents and its exact transcript.
#[derive(Clone, Debug, PartialEq)]
pub struct Prompt {
    /// The transcript, normalised, ending in a space (the target text follows it directly);
    /// empty for a reference-only prompt.
    pub text: String,
    /// Continuation latents `[patches × patch × feat]`, the clip left-padded to whole patches
    /// (none for a reference-only prompt).
    feat: Vec<f32>,
    patches: usize,
    /// VoxCPM2: the same clip as an isolated reference, right-padded.
    reference: Option<(Vec<f32>, usize)>,
    /// Length of the clip.
    pub seconds: f64,
}

impl Prompt {
    /// Continuation patches.
    pub fn patches(&self) -> usize {
        self.patches
    }

    /// The prompt carries the clip's transcript (continuation, "ultimate cloning"); otherwise it
    /// is a VoxCPM2 reference-only prompt.
    pub fn continues(&self) -> bool {
        self.patches > 0
    }
}

pub struct VoxCpm {
    cfg: Config,
    tok: Tokenizer,
    embed: Tensor,
    base: Stack,
    residual: Stack,
    enc: LocEnc,
    dit: LocDit,
    fsq_in: Linear,
    fsq_out: Linear,
    enc_to_lm: Linear,
    lm_to_dit: Linear,
    res_to_dit: Linear,
    fusion: Option<Linear>,
    stop_proj: Linear,
    stop_head: Linear,
    vae: Vae,
    dev: Device,
}

/// How the linear layers' weights are stored: VoxCPM2 on a GPU keeps its bf16 weights as they
/// are released (half the memory: about 5 GB instead of 9; activations stay f32), everything else
/// runs in f32 (on the CPU, and for VoxCPM1.5 on the GPU, the conversions cost more than they
/// save).
fn weight_dtype(variant: Variant, dev: &Device) -> candle_core::DType {
    if variant == Variant::V2 && dev.is_metal() { candle_core::DType::BF16 } else { candle_core::DType::F32 }
}

/// What [`VoxCpm::speak`] reports while it works: `(patches so far, cap)`; return `false` to stop.
pub type Progress<'a> = &'a mut dyn FnMut(usize, usize) -> bool;

impl VoxCpm {
    /// Load a release from `dir` (`config.json`, `tokenizer.json`, `model.safetensors`,
    /// `audiovae.pth`) onto [`super::device`].
    pub fn load(dir: &Path) -> R<VoxCpm> {
        Self::load_on(dir, &super::device())
    }

    pub fn load_on(dir: &Path, dev: &Device) -> R<VoxCpm> {
        // the small files first: a wrong release fails before gigabytes are read
        config::read(&dir.join("config.json"))?;
        Tokenizer::load(&dir.join("tokenizer.json"))?;
        let vae_w = weights::load_vae(&dir.join("audiovae.pth"))?;
        Self::load_parts(dir, &vae_w, dev)
    }

    /// Load with the AudioVAE weights already read.
    pub(crate) fn load_parts(dir: &Path, vae_w: &crate::kokoro::weights::Weights, dev: &Device) -> R<VoxCpm> {
        let cfg = config::read(&dir.join("config.json"))?;
        let tok = Tokenizer::load(&dir.join("tokenizer.json"))?;
        let mut st = weights::SafeTensors::open(&dir.join("model.safetensors"))?;
        let vae = Vae::load(vae_w, &cfg)?;
        let mut ld = Loader { st: &mut st, dev, wdtype: weight_dtype(cfg.variant, dev) };
        let h = cfg.base.hidden;
        let hd = cfg.dit.hidden;
        let max = cfg.max_length;
        let embed = ld.tensor("base_lm.embed_tokens.weight", &[cfg.vocab, h])?;
        let base = Stack::load(&mut ld, "base_lm", &cfg.base, &cfg.lm, max)?;
        let residual = Stack::load(&mut ld, "residual_lm", &cfg.residual, &cfg.lm, max)?;
        let enc = LocEnc::load(&mut ld, &cfg)?;
        let dit = LocDit::load(&mut ld, &cfg)?;
        let fsq_in = ld.linear("fsq_layer.in_proj", h, cfg.fsq_dim, true)?;
        let fsq_out = ld.linear("fsq_layer.out_proj", cfg.fsq_dim, h, true)?;
        let enc_to_lm = ld.linear("enc_to_lm_proj", cfg.encoder.hidden, h, true)?;
        let lm_to_dit = ld.linear("lm_to_dit_proj", h, hd, true)?;
        let res_to_dit = ld.linear("res_to_dit_proj", h, hd, true)?;
        let fusion = match cfg.variant {
            Variant::V2 => Some(ld.linear("fusion_concat_proj", 2 * h, h, true)?),
            Variant::V1 => None,
        };
        let stop_proj = ld.linear("stop_proj", h, h, true)?;
        let stop_head = ld.linear("stop_head", h, 2, false)?;
        if dit.mu_width()
            != match cfg.variant {
                Variant::V1 => hd,
                Variant::V2 => 2 * hd,
            }
        {
            return Err(err("the DiT conditioning width does not match"));
        }
        Ok(VoxCpm {
            cfg,
            tok,
            embed,
            base,
            residual,
            enc,
            dit,
            fsq_in,
            fsq_out,
            enc_to_lm,
            lm_to_dit,
            res_to_dit,
            fusion,
            stop_proj,
            stop_head,
            vae,
            dev: dev.clone(),
        })
    }

    pub fn variant(&self) -> Variant {
        self.cfg.variant
    }

    /// Rate prompt audio must be at.
    pub fn encode_rate(&self) -> u32 {
        self.cfg.encode_rate()
    }

    /// Rate of the generated audio.
    pub fn output_rate(&self) -> u32 {
        self.cfg.output_rate()
    }

    /// Output seconds per generated patch.
    pub fn patch_seconds(&self) -> f64 {
        (self.cfg.patch * self.cfg.decode_hop()) as f64 / f64::from(self.output_rate().max(1))
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// Encode audio (mono at [`Self::encode_rate`]) to patches `[n × patch × feat]`, padded with
    /// silence on the left (`left`) or the right to whole patches.
    fn patches(&self, audio: &[f32], left: bool) -> R<(Vec<f32>, usize)> {
        let (p, d) = (self.cfg.patch, self.cfg.feat);
        let plen = p * self.vae.enc_hop;
        let pad = (plen - audio.len() % plen) % plen;
        let mut a = Vec::with_capacity(audio.len() + pad);
        if left {
            a.resize(pad, 0.0);
            a.extend_from_slice(audio);
        } else {
            a.extend_from_slice(audio);
            a.resize(audio.len() + pad, 0.0);
        }
        let z = self.vae.encode(&a)?;
        let n = z.t / p;
        let mut out = vec![0f32; n * p * d];
        for i in 0..n {
            for j in 0..p {
                for c in 0..d {
                    if let (Some(o), Some(v)) = (out.get_mut((i * p + j) * d + c), z.x.get(c * z.t + i * p + j)) {
                        *o = *v;
                    }
                }
            }
        }
        Ok((out, n))
    }

    /// A prompt from the speaker's clip (mono at [`Self::encode_rate`]) and its exact transcript:
    /// a continuation prompt (VoxCPM2: plus the isolated reference). Without a transcript,
    /// VoxCPM2 makes a reference-only prompt; VoxCPM 1 needs the transcript.
    pub fn prompt(&self, audio: &[f32], transcript: Option<&str>) -> R<Prompt> {
        let rate = f64::from(self.encode_rate());
        let seconds = audio.len() as f64 / rate;
        if seconds < MIN_PROMPT_SECONDS {
            return Err(TtsError::Invalid(format!("the voice sample is {seconds:.1} s; at least {MIN_PROMPT_SECONDS} s is needed")));
        }
        if seconds > MAX_PROMPT_SECONDS {
            return Err(TtsError::Invalid(format!("the voice sample is {seconds:.0} s; at most {MAX_PROMPT_SECONDS} s is used")));
        }
        if audio.iter().any(|v| !v.is_finite()) {
            return Err(TtsError::Invalid("the voice sample has invalid samples".into()));
        }
        let text = transcript.map(|t| format!("{} ", t.split_whitespace().collect::<Vec<_>>().join(" "))).filter(|t| t.chars().any(char::is_alphanumeric));
        let reference = match self.cfg.variant {
            Variant::V2 => Some(self.patches(audio, false)?),
            Variant::V1 => None,
        };
        match text {
            Some(text) => {
                if self.tok.encode(&text)?.is_empty() {
                    return Err(TtsError::Invalid("the voice sample's transcript has no words".into()));
                }
                let (feat, patches) = self.patches(audio, true)?;
                Ok(Prompt { text, feat, patches, reference, seconds })
            }
            None if reference.is_some() => Ok(Prompt { text: String::new(), feat: Vec::new(), patches: 0, reference, seconds }),
            None => Err(TtsError::Invalid("VoxCPM1.5 needs the voice sample's exact transcript".into())),
        }
    }

    fn fsq(&self, x: &Tensor) -> R<Tensor> {
        let s = self.cfg.fsq_scale;
        let h = self.fsq_in.fwd(x)?.tanh().map_err(err)?;
        let q = ((h * s).map_err(err)?.round().map_err(err)? / s).map_err(err)?;
        self.fsq_out.fwd(&q)
    }

    /// Text tokens, patches and masks of the whole sequence.
    fn layout(&self, target: &str, prompt: Option<&Prompt>) -> R<(Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>, usize)> {
        let pd = self.cfg.patch * self.cfg.feat;
        let text = match prompt {
            Some(p) => format!("{}{target}", p.text),
            None => target.to_string(),
        };
        let mut text_ids = self.tok.encode(&text)?;
        text_ids.push(AUDIO_START);
        let mut ids: Vec<u32> = Vec::new();
        let mut feats: Vec<f32> = Vec::new();
        let mut tmask: Vec<f32> = Vec::new();
        let push = |ids: &mut Vec<u32>, feats: &mut Vec<f32>, tmask: &mut Vec<f32>, id: u32, patch: Option<&[f32]>| {
            ids.push(id);
            match patch {
                Some(p) => {
                    feats.extend_from_slice(p);
                    tmask.push(0.0);
                }
                None => {
                    feats.resize(feats.len() + pd, 0.0);
                    tmask.push(1.0);
                }
            }
        };
        if let Some((r, n)) = prompt.and_then(|p| p.reference.as_ref()) {
            push(&mut ids, &mut feats, &mut tmask, REF_START, None);
            for i in 0..*n {
                push(&mut ids, &mut feats, &mut tmask, 0, r.get(i * pd..(i + 1) * pd));
            }
            push(&mut ids, &mut feats, &mut tmask, REF_END, None);
        }
        for id in &text_ids {
            push(&mut ids, &mut feats, &mut tmask, *id, None);
        }
        let mut context = 0;
        if let Some(p) = prompt {
            for i in 0..p.patches {
                push(&mut ids, &mut feats, &mut tmask, 0, p.feat.get(i * pd..(i + 1) * pd));
            }
            let keep = match self.cfg.variant {
                Variant::V1 => 2,
                Variant::V2 => 3,
            };
            context = keep.min(p.patches);
        }
        let amask: Vec<f32> = tmask.iter().map(|t| 1.0 - t).collect();
        if feats.len() != ids.len() * pd {
            return Err(err("prompt layout is inconsistent"));
        }
        Ok((ids, feats, tmask, amask, context))
    }

    /// Speak `text` (one utterance, already normalised) in the prompt's voice, or in a voice the
    /// model makes up without one. Mono at [`Self::output_rate`].
    pub fn speak(&self, text: &str, prompt: Option<&Prompt>, s: &Settings, progress: Progress) -> R<Vec<f32>> {
        s.check()?;
        let target = self.tok.encode(text)?;
        if target.is_empty() {
            return Err(TtsError::Empty);
        }
        let (ids, feats, tmask, amask, context) = self.layout(text, prompt)?;
        let len = ids.len();
        let room = self.cfg.max_length.saturating_sub(len + 1);
        let cap = (target.len() * RATIO + 10).min(room);
        if cap <= MIN_PATCHES {
            return Err(TtsError::Invalid("the text and voice sample are too long for the model".into()));
        }
        let mut seed = s.seed;
        let mut last = None;
        for _ in 0..=s.retries {
            let patches = self.generate(&ids, &feats, &tmask, &amask, cap, s, seed, progress)?;
            let runaway = patches.len() / (self.cfg.patch * self.cfg.feat) >= target.len() * RATIO;
            last = Some(patches);
            if !runaway {
                break;
            }
            seed = seed.wrapping_add(1);
        }
        let generated = last.ok_or_else(|| err("nothing was generated"))?;
        self.decode(prompt, context, &generated)
    }

    /// The prefill and the patch loop. Returns the generated patches `[n × patch × feat]`.
    #[allow(clippy::too_many_arguments)]
    fn generate(&self, ids: &[u32], feats: &[f32], tmask: &[f32], amask: &[f32], cap: usize, s: &Settings, seed: u64, progress: Progress) -> R<Vec<f32>> {
        let dev = &self.dev;
        let (p, d, len) = (self.cfg.patch, self.cfg.feat, ids.len());
        if ids.iter().any(|i| *i as usize >= self.cfg.vocab) {
            return Err(err("a text token is outside the vocabulary"));
        }
        let feat_t = Tensor::from_slice(feats, (len, p, d), dev).map_err(err)?;
        let feat_emb = self.enc_to_lm.fwd(&self.enc.fwd(&feat_t)?)?; // [len, h]
        let id_t = Tensor::from_slice(ids, len, dev).map_err(err)?;
        let mut text_emb = self.embed.index_select(&id_t, 0).map_err(err)?;
        if self.cfg.emb_scale != 1.0 {
            text_emb = (text_emb * self.cfg.emb_scale).map_err(err)?;
        }
        let tm = Tensor::from_slice(tmask, (len, 1), dev).map_err(err)?;
        let am = Tensor::from_slice(amask, (len, 1), dev).map_err(err)?;
        let feat_emb_a = feat_emb.broadcast_mul(&am).map_err(err)?;
        let combined = (text_emb.broadcast_mul(&tm).map_err(err)? + &feat_emb_a).map_err(err)?;
        let mut base_cache = self.base.cache();
        let out = self.base.forward(&combined.unsqueeze(0).map_err(err)?, true, Some(&mut base_cache))?.squeeze(0).map_err(err)?;
        let out = (self.fsq(&out)?.broadcast_mul(&am).map_err(err)? + out.broadcast_mul(&tm).map_err(err)?).map_err(err)?;
        let mut lm_h = out.narrow(0, len - 1, 1).map_err(err)?;
        let res_in = match &self.fusion {
            Some(f) => f.fwd(&Tensor::cat(&[&out, &feat_emb_a], D::Minus1).map_err(err)?)?,
            None => (&out + &feat_emb_a).map_err(err)?,
        };
        let mut res_cache = self.residual.cache();
        let res_out = self.residual.forward(&res_in.unsqueeze(0).map_err(err)?, true, Some(&mut res_cache))?.squeeze(0).map_err(err)?;
        let mut res_h = res_out.narrow(0, len - 1, 1).map_err(err)?;
        let mut cond = feat_t.narrow(0, len - 1, 1).map_err(err)?;
        let mut noise = Noise::new(seed);
        let mut out_patches: Vec<Tensor> = Vec::with_capacity(cap.min(4096));
        for i in 0..cap {
            if !progress(i, cap) {
                return Err(TtsError::Model("stopped".into()));
            }
            let mu = match self.cfg.variant {
                Variant::V1 => (self.lm_to_dit.fwd(&lm_h)? + self.res_to_dit.fwd(&res_h)?).map_err(err)?,
                Variant::V2 => Tensor::cat(&[self.lm_to_dit.fwd(&lm_h)?, self.res_to_dit.fwd(&res_h)?], D::Minus1).map_err(err)?,
            };
            let pred = self.dit.sample(&mu, &cond, s.timesteps, s.cfg, &mut noise)?; // [1, p, d]
            out_patches.push(pred.clone());
            let curr = self.enc_to_lm.fwd(&self.enc.fwd(&pred)?)?; // [1, h]
            cond = pred;
            let logits = self.stop_head.fwd(&candle_nn::ops::silu(&self.stop_proj.fwd(&lm_h)?).map_err(err)?)?;
            let l = logits.flatten_all().map_err(err)?.to_vec1::<f32>().map_err(err)?;
            let stop = matches!((l.first(), l.get(1)), (Some(a), Some(b)) if b > a);
            if i > MIN_PATCHES && stop {
                break;
            }
            if i + 1 == cap {
                break;
            }
            lm_h = self.fsq(&self.base.step(&curr, &mut base_cache)?)?;
            let r_in = match &self.fusion {
                Some(f) => f.fwd(&Tensor::cat(&[&lm_h, &curr], D::Minus1).map_err(err)?)?,
                None => (&lm_h + &curr).map_err(err)?,
            };
            res_h = self.residual.step(&r_in, &mut res_cache)?;
        }
        if out_patches.is_empty() {
            return Ok(Vec::new());
        }
        let all = Tensor::cat(&out_patches, 0).map_err(err)?.flatten_all().map_err(err)?.to_vec1::<f32>().map_err(err)?;
        if all.iter().any(|v| !v.is_finite()) {
            return Err(err("the diffusion steps diverged"));
        }
        Ok(all)
    }

    /// Audio of `generated` patches, decoded after the prompt's last `context` patches (whose
    /// audio is then dropped).
    fn decode(&self, prompt: Option<&Prompt>, context: usize, generated: &[f32]) -> R<Vec<f32>> {
        let (p, d) = (self.cfg.patch, self.cfg.feat);
        let pd = p * d;
        let mut all: Vec<f32> = Vec::new();
        if let Some(pr) = prompt
            && context > 0
        {
            let from = pr.patches.saturating_sub(context) * pd;
            all.extend_from_slice(pr.feat.get(from..pr.patches * pd).unwrap_or(&[]));
        }
        all.extend_from_slice(generated);
        let frames = all.len() / d;
        // [n, p, d] → [d, n·p]
        let mut z = vec![0f32; d * frames];
        for f in 0..frames {
            for c in 0..d {
                if let (Some(o), Some(v)) = (z.get_mut(c * frames + f), all.get(f * d + c)) {
                    *o = *v;
                }
            }
        }
        let audio = self.vae.decode(&Sig { c: d, t: frames, x: z })?;
        let skip = context * p * self.vae.dec_hop;
        let out = audio.get(skip..).unwrap_or(&[]).to_vec();
        if out.iter().any(|v| !v.is_finite()) {
            return Err(err("the decoder produced invalid samples"));
        }
        Ok(out)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::voxcpm::{minicpm, vae};

    /// A tiny random VoxCPM (V1 or V2) written to `dir` as a release: config, tokenizer,
    /// safetensors and an AudioVAE checkpoint.
    pub(crate) fn fake_release(dir: &Path, v2: bool) {
        std::fs::create_dir_all(dir).unwrap();
        let cfg_json = config::tests::tiny(v2);
        std::fs::write(dir.join("config.json"), &cfg_json).unwrap();
        std::fs::write(dir.join("tokenizer.json"), crate::voxcpm::tokenizer::tests::tiny_json()).unwrap();
        let cfg = config::parse(cfg_json.as_bytes()).unwrap();
        let mut rng = Noise::new(21);
        let mut ts: Vec<(String, Tensor)> = Vec::new();
        let dev = Device::Cpu;
        let t = |ts: &mut Vec<(String, Tensor)>, rng: &mut Noise, name: &str, dims: &[usize], scale: f32| {
            let n: usize = dims.iter().product();
            ts.push((name.to_string(), Tensor::from_vec(rng.take(n).into_iter().map(|x| x * scale).collect::<Vec<f32>>(), dims, &dev).unwrap()));
        };
        let (h, f) = (cfg.base.hidden, cfg.feat);
        t(&mut ts, &mut rng, "base_lm.embed_tokens.weight", &[cfg.vocab, h], 0.5);
        minicpm::tests::fake("base_lm", &cfg.base, &mut rng, &mut ts);
        minicpm::tests::fake("residual_lm", &cfg.residual, &mut rng, &mut ts);
        minicpm::tests::fake("feat_encoder.encoder", &cfg.encoder, &mut rng, &mut ts);
        minicpm::tests::fake("feat_decoder.estimator.decoder", &cfg.dit, &mut rng, &mut ts);
        let lin = |ts: &mut Vec<(String, Tensor)>, rng: &mut Noise, name: &str, i: usize, o: usize, bias: bool| {
            let n = i * o;
            let s = 1.0 / (i as f32).sqrt();
            ts.push((format!("{name}.weight"), Tensor::from_vec(rng.take(n).into_iter().map(|x| x * s).collect::<Vec<f32>>(), (o, i), &Device::Cpu).unwrap()));
            if bias {
                ts.push((format!("{name}.bias"), Tensor::from_vec(rng.take(o).into_iter().map(|x| x * 0.01).collect::<Vec<f32>>(), o, &Device::Cpu).unwrap()));
            }
        };
        let he = cfg.encoder.hidden;
        let hd = cfg.dit.hidden;
        t(&mut ts, &mut rng, "feat_encoder.special_token", &[1, 1, 1, he], 1.0);
        lin(&mut ts, &mut rng, "feat_encoder.in_proj", f, he, true);
        for n in ["in_proj", "cond_proj"] {
            lin(&mut ts, &mut rng, &format!("feat_decoder.estimator.{n}"), f, hd, true);
        }
        lin(&mut ts, &mut rng, "feat_decoder.estimator.out_proj", hd, f, true);
        for n in ["time_mlp", "delta_time_mlp"] {
            lin(&mut ts, &mut rng, &format!("feat_decoder.estimator.{n}.linear_1"), hd, hd, true);
            lin(&mut ts, &mut rng, &format!("feat_decoder.estimator.{n}.linear_2"), hd, hd, true);
        }
        lin(&mut ts, &mut rng, "fsq_layer.in_proj", h, cfg.fsq_dim, true);
        lin(&mut ts, &mut rng, "fsq_layer.out_proj", cfg.fsq_dim, h, true);
        lin(&mut ts, &mut rng, "enc_to_lm_proj", he, h, true);
        lin(&mut ts, &mut rng, "lm_to_dit_proj", h, hd, true);
        lin(&mut ts, &mut rng, "res_to_dit_proj", h, hd, true);
        if v2 {
            lin(&mut ts, &mut rng, "fusion_concat_proj", 2 * h, h, true);
        }
        lin(&mut ts, &mut rng, "stop_proj", h, h, true);
        lin(&mut ts, &mut rng, "stop_head", h, 2, false);
        weights::write(&dir.join("model.safetensors"), &ts);
        // the AudioVAE as a PyTorch checkpoint is not something the tests can write; the loader is
        // exercised with real files in the live tests, and here through `load_parts`
        let vae_map = vae::tests::fake(&cfg, 5);
        let names: Vec<String> = vae_map.keys().cloned().collect();
        let st: Vec<(String, Tensor)> = names.into_iter().map(|n| (n.clone(), vae_map[&n].clone())).collect();
        weights::write(&dir.join("audiovae.safetensors"), &st);
    }

    /// Load a fake release (the VAE from its test safetensors instead of a `.pth`).
    pub(crate) fn load_fake(dir: &Path) -> VoxCpm {
        let dev = Device::Cpu;
        let cfg = config::read(&dir.join("config.json")).unwrap();
        let mut vst = weights::SafeTensors::open(&dir.join("audiovae.safetensors")).unwrap();
        let fake = vae::tests::fake(&cfg, 5);
        let mut map = std::collections::HashMap::new();
        for (k, v) in fake {
            map.insert(k.clone(), vst.load(&k, v.dims(), &dev).unwrap());
        }
        let w = crate::kokoro::weights::Weights::from_map(map);
        VoxCpm::load_parts(dir, &w, &dev).unwrap()
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("filmcraft-voxcpm-model-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn a_tiny_model_speaks_deterministically_with_and_without_a_prompt() {
        for v2 in [false, true] {
            let d = tmp(if v2 { "v2" } else { "v1" });
            fake_release(&d, v2);
            let m = load_fake(&d);
            let s = Settings { timesteps: 4, ..Default::default() };
            let mut calls = 0;
            let a = m
                .speak("hello world", None, &s, &mut |_, _| {
                    calls += 1;
                    true
                })
                .unwrap();
            assert!(calls >= 1);
            assert!(!a.is_empty() && a.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
            // patches are whole: a multiple of the samples per patch
            let per = m.cfg.patch * m.cfg.decode_hop();
            assert_eq!(a.len() % per, 0);
            // capped at 6 patches per token + 10 (2 tokens here)
            assert!(a.len() / per <= 2 * RATIO + 10);
            let b = m.speak("hello world", None, &s, &mut |_, _| true).unwrap();
            assert_eq!(a, b, "same seed, same audio");
            let c = m.speak("hello world", None, &Settings { seed: 99, ..s }, &mut |_, _| true).unwrap();
            assert_ne!(a, c, "another seed, other audio");
            // with a prompt: one second of a tone and its "transcript"
            let rate = m.encode_rate() as usize;
            let audio: Vec<f32> = (0..rate * 3 / 2).map(|i| (i as f32 * 0.07).sin() * 0.3).collect();
            let pr = m.prompt(&audio, Some("hello")).unwrap();
            assert_eq!(pr.text, "hello ");
            assert!(pr.patches() > 0);
            let e = m.speak("world", Some(&pr), &s, &mut |_, _| true).unwrap();
            assert!(!e.is_empty() && e.len().is_multiple_of(per));
            // stopping
            let r = m.speak("hello world", None, &s, &mut |_, _| false);
            assert!(r.is_err());
            let _ = std::fs::remove_dir_all(&d);
        }
    }

    #[test]
    fn layouts_follow_the_reference() {
        let d = tmp("layout");
        fake_release(&d, true);
        let m = load_fake(&d);
        let rate = m.encode_rate() as usize;
        let audio = vec![0.1f32; rate * 2];
        let pr = m.prompt(&audio, Some("hello")).unwrap();
        let (ids, feats, tm, am, ctx) = m.layout("world", Some(&pr)).unwrap();
        let n = pr.patches();
        let r = pr.reference.as_ref().unwrap().1;
        // [ref_start, ref…, ref_end, "▁hello", "▁world", audio_start, prompt…]
        assert_eq!(ids.len(), 1 + r + 1 + 2 + 1 + n);
        assert_eq!(ids[0], REF_START);
        assert_eq!(ids[r + 1], REF_END);
        assert_eq!(ids[r + 4], AUDIO_START);
        assert_eq!(tm[0], 1.0);
        assert_eq!(am[1], 1.0);
        assert_eq!(tm[r + 4], 1.0);
        assert!(am[r + 5..].iter().all(|a| *a == 1.0));
        assert_eq!(feats.len(), ids.len() * m.cfg.patch * m.cfg.feat);
        assert_eq!(ctx, 3.min(n));
        // reference only (no transcript): [ref_start, ref…, ref_end, "▁world", audio_start]
        let pr = m.prompt(&audio, None).unwrap();
        assert!(!pr.continues() && pr.text.is_empty());
        let (ids, _, tm, am, ctx) = m.layout("world", Some(&pr)).unwrap();
        assert_eq!(ids.len(), 1 + r + 1 + 1 + 1);
        assert_eq!(*ids.last().unwrap(), AUDIO_START);
        assert_eq!(tm.last(), Some(&1.0));
        assert_eq!(am.iter().filter(|a| **a == 1.0).count(), r);
        assert_eq!(ctx, 0, "nothing to continue from");
        // without a prompt: tokens and audio_start only
        let (ids, _, tm, _, ctx) = m.layout("world", None).unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(*ids.last().unwrap(), AUDIO_START);
        assert!(tm.iter().all(|t| *t == 1.0));
        assert_eq!(ctx, 0);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn hostile_prompts_and_text_are_errors() {
        let d = tmp("hostile");
        fake_release(&d, false);
        let m = load_fake(&d);
        let s = Settings::default();
        let rate = m.encode_rate() as usize;
        assert!(m.prompt(&vec![0.0; rate / 2], Some("hi")).is_err(), "too short");
        assert!(m.prompt(&vec![0.0; rate * 50], Some("hi")).is_err(), "too long");
        assert!(m.prompt(&vec![f32::NAN; rate * 2], Some("hi")).is_err(), "NaN");
        assert!(m.prompt(&vec![0.0; rate * 2], Some("  ")).is_err(), "VoxCPM 1: no transcript");
        assert!(m.prompt(&vec![0.0; rate * 2], None).is_err(), "VoxCPM 1: no transcript");
        // silence is a valid (if useless) prompt
        assert!(m.prompt(&vec![0.0; rate * 2], Some("hello")).is_ok());
        assert_eq!(m.speak("", None, &s, &mut |_, _| true).err(), Some(TtsError::Empty));
        assert!(m.speak(&"hello ".repeat(5000), None, &s, &mut |_, _| true).is_err(), "too long for the tokenizer");
        assert!(m.speak("hello", None, &Settings { cfg: 100.0, ..s }, &mut |_, _| true).is_err());
        // longer than the model's 1024 positions
        assert!(m.speak(&"hello ".repeat(1100), None, &s, &mut |_, _| true).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn broken_releases_are_errors() {
        let d = tmp("broken");
        fake_release(&d, false);
        // no audiovae.pth in the fake release: loading the real way is an error, not a panic
        assert!(VoxCpm::load_on(&d, &Device::Cpu).is_err());
        // a safetensors file missing a tensor
        std::fs::write(d.join("audiovae.pth"), b"PK\x03\x04 not a checkpoint").unwrap();
        assert!(VoxCpm::load_on(&d, &Device::Cpu).is_err());
        std::fs::write(d.join("config.json"), b"{}").unwrap();
        assert!(VoxCpm::load_on(&d, &Device::Cpu).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Against the PyTorch reference run as an oracle with the same noise (gated): the patches
    /// it generated for "Hello, this is a test of the voice model." (zero-shot, or after the
    /// prompt `VOXCPM_ORACLE_PROMPT` = raw f32 at the encoder rate + `VOXCPM_ORACLE_PROMPT_TEXT`)
    /// are in `VOXCPM_ORACLE` (f32, `[n × patch × feat]`).
    #[test]
    fn live_matches_the_reference_oracle() {
        let (Some(dir), Some(file)) = (std::env::var_os("FILMCRAFT_VOXCPM_DIR"), std::env::var_os("VOXCPM_ORACLE")) else {
            eprintln!("skipped (FILMCRAFT_VOXCPM_DIR or VOXCPM_ORACLE unset)");
            return;
        };
        let want: Vec<f32> = std::fs::read(&file).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        let m = VoxCpm::load_on(Path::new(&dir), &crate::voxcpm::device()).unwrap();
        let pd = m.cfg.patch * m.cfg.feat;
        let n = want.len() / pd;
        let prompt = std::env::var_os("VOXCPM_ORACLE_PROMPT").map(|w| {
            let x: Vec<f32> = std::fs::read(&w).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
            let t = std::env::var("VOXCPM_ORACLE_PROMPT_TEXT").unwrap();
            let mut p = m.prompt(&x, Some(&t)).unwrap();
            p.text = t; // exactly as the oracle tokenized it
            p
        });
        let text = "Hello, this is a test of the voice model.";
        let (ids, feats, tm, am, _) = m.layout(text, prompt.as_ref()).unwrap();
        eprintln!("ids {:?}", &ids[..ids.len().min(80)]);
        let got = m.generate(&ids, &feats, &tm, &am, n, &Settings::default(), 1, &mut |_, _| true).unwrap();
        eprintln!("reference {n} patches, port {} patches", got.len() / pd);
        for i in 0..n.min(got.len() / pd) {
            let d = (0..pd).map(|j| (got[i * pd + j] - want[i * pd + j]).abs()).fold(0f32, f32::max);
            let mag = (0..pd).map(|j| want[i * pd + j].abs()).fold(0f32, f32::max);
            eprintln!("patch {i}: max |diff| {d:.2e} (max |value| {mag:.2})");
        }
    }

    /// Where the time goes, per part, with a real release (gated):
    /// FILMCRAFT_VOXCPM_DIR=<release> cargo test --release -p filmcraft-tts --features voxcpm
    /// live_voxcpm_parts -- --nocapture
    #[test]
    fn live_voxcpm_parts() {
        let Some(dir) = std::env::var_os("FILMCRAFT_VOXCPM_DIR") else {
            eprintln!("skipped (FILMCRAFT_VOXCPM_DIR unset)");
            return;
        };
        let dev = crate::voxcpm::device();
        let m = VoxCpm::load_on(Path::new(&dir), &dev).unwrap();
        let sync = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (p, d, h) = (m.cfg.patch, m.cfg.feat, m.cfg.base.hidden);
        let mu = Tensor::zeros((1, m.dit.mu_width()), candle_core::DType::F32, &dev).unwrap();
        let cond = Tensor::zeros((1, p, d), candle_core::DType::F32, &dev).unwrap();
        let mut noise = Noise::new(1);
        let n = 8;
        let _ = sync(&m.dit.sample(&mu, &cond, 10, 2.0, &mut noise).unwrap());
        let t = std::time::Instant::now();
        for _ in 0..n {
            let _ = sync(&m.dit.sample(&mu, &cond, 10, 2.0, &mut noise).unwrap());
        }
        eprintln!("{:?}: LocDiT sample (10 steps): {:.1} ms", dev.location(), t.elapsed().as_secs_f64() * 1000.0 / n as f64);
        let t = std::time::Instant::now();
        for _ in 0..n {
            let _ = sync(&m.enc.fwd(&cond).unwrap());
        }
        eprintln!("LocEnc: {:.1} ms", t.elapsed().as_secs_f64() * 1000.0 / n as f64);
        let x = Tensor::zeros((1, 64, h), candle_core::DType::F32, &dev).unwrap();
        let mut c = m.base.cache();
        let mut rc = m.residual.cache();
        m.base.forward(&x, true, Some(&mut c)).unwrap();
        m.residual.forward(&x, true, Some(&mut rc)).unwrap();
        let one = Tensor::zeros((1, h), candle_core::DType::F32, &dev).unwrap();
        let t = std::time::Instant::now();
        for _ in 0..n {
            let _ = sync(&m.base.step(&one, &mut c).unwrap());
        }
        eprintln!("text-semantic LM step: {:.1} ms", t.elapsed().as_secs_f64() * 1000.0 / n as f64);
        let t = std::time::Instant::now();
        for _ in 0..n {
            let _ = sync(&m.residual.step(&one, &mut rc).unwrap());
        }
        eprintln!("residual LM step: {:.1} ms", t.elapsed().as_secs_f64() * 1000.0 / n as f64);
        let frames = 100;
        let z = Sig { c: d, t: frames, x: vec![0.0; d * frames] };
        let t = std::time::Instant::now();
        let y = m.vae.decode(&z).unwrap();
        let secs = y.len() as f64 / f64::from(m.output_rate());
        eprintln!("AudioVAE decode: {:.1} ms per second of audio", t.elapsed().as_secs_f64() * 1000.0 / secs);
        let audio = vec![0.0f32; m.encode_rate() as usize * 10];
        let t = std::time::Instant::now();
        let _ = m.vae.encode(&audio).unwrap();
        eprintln!("AudioVAE encode: {:.1} ms per second of audio", t.elapsed().as_secs_f64() * 100.0);
    }
}
