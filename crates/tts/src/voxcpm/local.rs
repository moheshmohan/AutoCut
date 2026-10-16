//! The local networks around each patch: LocEnc (a patch of latents → one LM-sized vector) and
//! LocDiT (the LMs' state → the next patch, by flow matching).
//!
//! **LocEnc**: project each frame of the patch, put a learned token in front, run a
//! bidirectional MiniCPM stack and keep that first token's output.
//!
//! **LocDiT** predicts the velocity of a straight path from noise (t = 1) to the patch (t = 0).
//! Its sequence is `[conditioning tokens, previous patch, noisy patch]`: VoxCPM 1 sums the two
//! LMs' projected states with the time embedding into one token; VoxCPM2 keeps the two states as
//! two tokens and adds a time token. Sampling ([`LocDit::sample`]) integrates from t = 1 to 0
//! with Euler steps on the "sway" schedule `t + (cos(πt/2) − 1 + t)`, skipping the first step's
//! velocity (zero init), and guides each step with CFG-Zero★: the unconditional velocity (zeroed
//! LM state) is rescaled by its optimal projection onto the conditional one before the usual
//! `uncond + w · (cond − uncond)`.

use candle_core::Tensor;

use super::config::{Config, Variant};
use super::minicpm::Stack;
use super::{Linear, Loader, Noise, R, err};

pub(crate) struct LocEnc {
    special: Tensor,
    in_proj: Linear,
    enc: Stack,
}

impl LocEnc {
    pub(crate) fn load(ld: &mut Loader, cfg: &Config) -> R<LocEnc> {
        let h = cfg.encoder.hidden;
        Ok(LocEnc {
            special: ld.tensor("feat_encoder.special_token", &[1, 1, 1, h])?.reshape((1, 1, h)).map_err(err)?,
            in_proj: ld.linear("feat_encoder.in_proj", cfg.feat, h, true)?,
            enc: Stack::load(ld, "feat_encoder.encoder", &cfg.encoder, &cfg.lm, cfg.patch + 1)?,
        })
    }

    /// Patches `[n, patch, feat]` → `[n, hidden]`.
    pub(crate) fn fwd(&self, x: &Tensor) -> R<Tensor> {
        let n = x.dim(0).map_err(err)?;
        let h = self.in_proj.fwd(x)?;
        let s = self.special.broadcast_as((n, 1, self.enc.hidden())).map_err(err)?.contiguous().map_err(err)?;
        let seq = Tensor::cat(&[&s, &h], 1).map_err(err)?;
        let out = self.enc.forward(&seq, false, None)?;
        out.narrow(1, 0, 1).map_err(err)?.squeeze(1).map_err(err)
    }
}

/// `[sin(1000 t f_i), cos(1000 t f_i)]` with `f_i = exp(−ln(10⁴) i / (half − 1))`.
pub(crate) fn time_embedding(t: f64, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let e = (10_000f64).ln() / (half.saturating_sub(1).max(1)) as f64;
    let mut out = vec![0f32; dim];
    for i in 0..half {
        let a = 1000.0 * t * (-e * i as f64).exp();
        if let Some(s) = out.get_mut(i) {
            *s = a.sin() as f32;
        }
        if let Some(c) = out.get_mut(half + i) {
            *c = a.cos() as f32;
        }
    }
    out
}

struct TimeMlp {
    a: Linear,
    b: Linear,
}

impl TimeMlp {
    fn load(ld: &mut Loader, name: &str, h: usize) -> R<TimeMlp> {
        Ok(TimeMlp { a: ld.linear(&format!("{name}.linear_1"), h, h, true)?, b: ld.linear(&format!("{name}.linear_2"), h, h, true)? })
    }
    fn fwd(&self, x: &Tensor) -> R<Tensor> {
        self.b.fwd(&candle_nn::ops::silu(&self.a.fwd(x)?).map_err(err)?)
    }
}

pub(crate) struct LocDit {
    variant: Variant,
    in_proj: Linear,
    cond_proj: Linear,
    out_proj: Linear,
    time: TimeMlp,
    /// `delta_time_mlp(emb(0))`: the mean-flow interval input, always zero at inference.
    dt0: Tensor,
    dec: Stack,
    hidden: usize,
    feat: usize,
    patch: usize,
}

impl LocDit {
    pub(crate) fn load(ld: &mut Loader, cfg: &Config) -> R<LocDit> {
        let h = cfg.dit.hidden;
        let p = "feat_decoder.estimator";
        let time = TimeMlp::load(ld, &format!("{p}.time_mlp"), h)?;
        let delta = TimeMlp::load(ld, &format!("{p}.delta_time_mlp"), h)?;
        let dt0 = delta.fwd(&Tensor::from_vec(time_embedding(0.0, h), (1, h), ld.dev).map_err(err)?)?;
        let mu_tokens = if cfg.variant == Variant::V2 { 2 } else { 1 };
        Ok(LocDit {
            variant: cfg.variant,
            in_proj: ld.linear(&format!("{p}.in_proj"), cfg.feat, h, true)?,
            cond_proj: ld.linear(&format!("{p}.cond_proj"), cfg.feat, h, true)?,
            out_proj: ld.linear(&format!("{p}.out_proj"), h, cfg.feat, true)?,
            time,
            dt0,
            dec: Stack::load(ld, &format!("{p}.decoder"), &cfg.dit, &cfg.lm, mu_tokens + 1 + 2 * cfg.patch)?,
            hidden: h,
            feat: cfg.feat,
            patch: cfg.patch,
        })
    }

    /// Width of the conditioning vector `mu` (the projected LM states).
    pub(crate) fn mu_width(&self) -> usize {
        match self.variant {
            Variant::V1 => self.hidden,
            Variant::V2 => 2 * self.hidden,
        }
    }

    /// Velocity for noisy patches `x` `[n, patch, feat]`, given the conditioning head tokens
    /// (`mu` and the time embedding, `[n, head, hidden]`) and the projected previous patch `c`
    /// `[n, patch, hidden]`.
    fn velocity(&self, x: &Tensor, head: &Tensor, c: &Tensor) -> R<Tensor> {
        let xin = self.in_proj.fwd(x)?;
        let skip = head.dim(1).map_err(err)? + self.patch;
        let seq = Tensor::cat(&[head, c, &xin], 1).map_err(err)?;
        let y = self.dec.forward(&seq, false, None)?;
        self.out_proj.fwd(&y.narrow(1, skip, self.patch).map_err(err)?)
    }

    /// The head tokens for `mu` `[n, mu_width]` at time embedding `temb` `[1, hidden]`: VoxCPM 1
    /// sums them into one token, VoxCPM2 keeps `mu`'s two halves and adds a time token.
    fn head(&self, mu: &Tensor, temb: &Tensor) -> R<Tensor> {
        let n = mu.dim(0).map_err(err)?;
        let h = self.hidden;
        match self.variant {
            Variant::V1 => mu.broadcast_add(temb).map_err(err)?.unsqueeze(1).map_err(err),
            Variant::V2 => {
                let m = mu.reshape((n, 2, h)).map_err(err)?;
                let t = temb.unsqueeze(1).map_err(err)?.broadcast_as((n, 1, h)).map_err(err)?.contiguous().map_err(err)?;
                Tensor::cat(&[&m, &t], 1).map_err(err)
            }
        }
    }

    /// The next patch `[1, patch, feat]` for `mu` `[1, mu_width]` after `cond` `[1, patch, feat]`.
    /// Everything stays on the device: no value is read back until the caller wants the patch.
    pub(crate) fn sample(&self, mu: &Tensor, cond: &Tensor, steps: usize, cfg_scale: f32, noise: &mut Noise) -> R<Tensor> {
        let dev = mu.device();
        let (p, d, h) = (self.patch, self.feat, self.hidden);
        // the reference draws [b, feat, patch]; keep that order for the same seed, same noise
        let z = Tensor::from_vec(noise.take(d * p), (1, d, p), dev).map_err(err)?.transpose(1, 2).map_err(err)?.contiguous().map_err(err)?;
        let span = schedule(steps);
        let zero_init = ((span.len() as f64 * 0.04) as usize).max(1);
        // every step's time embedding in one pass (t at step s is span[s - 1])
        let times: Vec<f32> = span.iter().flat_map(|t| time_embedding(*t, h)).collect();
        let temb = self.time.fwd(&Tensor::from_vec(times, (span.len(), h), dev).map_err(err)?)?.broadcast_add(&self.dt0).map_err(err)?;
        // conditional and unconditional (zero mu) halves of one batch; the previous patch is shared
        let mu2 = Tensor::cat(&[mu, &mu.zeros_like().map_err(err)?], 0).map_err(err)?;
        let c2 = self.cond_proj.fwd(&Tensor::cat(&[cond, cond], 0).map_err(err)?)?;
        let mut x = z;
        let mut t = span.first().copied().unwrap_or(1.0);
        let mut dt = t - span.get(1).copied().unwrap_or(0.0);
        for step in 1..span.len() {
            if step > zero_init {
                let head = self.head(&mu2, &temb.narrow(0, step - 1, 1).map_err(err)?)?;
                let x2 = Tensor::cat(&[&x, &x], 0).map_err(err)?;
                let v = self.velocity(&x2, &head, &c2)?;
                let pos = v.narrow(0, 0, 1).map_err(err)?;
                let neg = v.narrow(0, 1, 1).map_err(err)?;
                // CFG-Zero*: project the unconditional velocity onto the conditional one
                let dot = (&pos * &neg).map_err(err)?.sum_all().map_err(err)?;
                let nn = (neg.sqr().map_err(err)?.sum_all().map_err(err)? + 1e-8).map_err(err)?;
                let st = (dot / nn).map_err(err)?;
                let scaled = neg.broadcast_mul(&st).map_err(err)?;
                let guided = (&scaled + ((&pos - &scaled).map_err(err)? * f64::from(cfg_scale)).map_err(err)?).map_err(err)?;
                x = (&x - (guided * dt).map_err(err)?).map_err(err)?;
            }
            t -= dt;
            if let Some(next) = span.get(step + 1) {
                dt = t - next;
            }
        }
        Ok(x)
    }
}

/// The sway-sampling time grid from 1 to 0 in `steps` steps.
pub(crate) fn schedule(steps: usize) -> Vec<f64> {
    (0..=steps)
        .map(|i| {
            let t = 1.0 - i as f64 / steps.max(1) as f64;
            t + ((std::f64::consts::FRAC_PI_2 * t).cos() - 1.0 + t)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_runs_from_one_to_zero_and_is_front_loaded() {
        let s = schedule(10);
        assert_eq!(s.len(), 11);
        assert!((s[0] - 1.0).abs() < 1e-12 && s[10].abs() < 1e-12);
        assert!(s.windows(2).all(|w| w[1] < w[0]), "monotone");
        // 2t + cos(πt/2) − 1: small steps near the noise end, large ones near the data
        assert!((s[1] - (0.8 + (0.45 * std::f64::consts::PI).cos())).abs() < 1e-12);
        assert!(s[0] - s[1] < s[9] - s[10]);
    }

    #[test]
    fn time_embedding_matches_the_formula() {
        let e = time_embedding(0.0, 8);
        assert_eq!(e, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let e = time_embedding(0.5, 8);
        // half = 4: f_i = exp(−ln(1e4) i / 3), argument 500 f_i
        let f1 = (-(10_000f64).ln() / 3.0).exp();
        assert!((f64::from(e[1]) - (500.0 * f1).sin()).abs() < 1e-5);
        assert!((f64::from(e[5]) - (500.0 * f1).cos()).abs() < 1e-5);
    }
}
