//! The port against the official reference, run as an oracle (env-gated; `cargo test` stays
//! offline).
//!
//! `FILMCRAFT_ACESTEP_DIR` names a package directory (the catalogue's flat file names) and
//! `ACESTEP_ORACLE` the `oracle.safetensors` (with its `oracle.json`) written by running the
//! official MIT/Apache-2.0 reference on the CPU in f32 with the same prompt and noise (the
//! script is in the crate README). Every stage is fed the reference's own inputs, so each
//! comparison measures one stage: the tokenizer's ids, the text encoder, the condition encoder,
//! each of the eight DiT velocity passes (and the first pass's inner states), the sampler from the
//! same noise, and the VAE decoder.
//!
//! `FILMCRAFT_ACESTEP_DIR=… ACESTEP_ORACLE=…/oracle24.safetensors cargo test --release -p
//! filmcraft-acestep --features model live_oracle -- --nocapture --test-threads 1`
//! (CPU, f32 weights). `ACESTEP_ORACLE_DEVICE=metal` runs the same comparison on the GPU with the
//! bf16-stored weights the app uses there (widened to f32 for every product: same tolerances).

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};

use super::weights::SafeTensors;
use super::*;

struct Oracle {
    st: SafeTensors,
    meta: serde_json::Value,
}

impl Oracle {
    fn open() -> Option<(Oracle, PathBuf)> {
        let (Some(dir), Some(o)) = (std::env::var_os("FILMCRAFT_ACESTEP_DIR"), std::env::var_os("ACESTEP_ORACLE")) else {
            eprintln!("skipped (FILMCRAFT_ACESTEP_DIR or ACESTEP_ORACLE unset)");
            return None;
        };
        let p = PathBuf::from(o);
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(p.with_extension("json")).unwrap()).unwrap();
        Some((Oracle { st: SafeTensors::open(&p).unwrap(), meta }, PathBuf::from(dir)))
    }

    fn get(&mut self, name: &str, dev: &Device) -> Tensor {
        let shape = self.st.shape(name).unwrap_or_else(|| panic!("oracle has no `{name}`")).to_vec();
        self.st.load(name, &shape, dev).unwrap()
    }

    fn ids(&self, key: &str) -> Vec<u32> {
        self.meta[key].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
    }
}

/// Largest absolute difference, and the reference's largest magnitude.
fn compare(name: &str, got: &Tensor, want: &Tensor) -> (f32, f32) {
    let g = got.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let w = want.to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(g.len(), w.len(), "{name}: {:?} vs {:?}", got.dims(), want.dims());
    let diff = g.iter().zip(&w).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let peak = w.iter().fold(0f32, |m, x| m.max(x.abs()));
    let rms = (g.iter().zip(&w).map(|(a, b)| f64::from(a - b).powi(2)).sum::<f64>() / g.len().max(1) as f64).sqrt();
    eprintln!("{name:>16}: max |Δ| {diff:.3e} (rms {rms:.3e}) of peak {peak:.3e} → {:.2e} relative", diff / peak.max(1e-12));
    (diff, peak)
}

fn check(name: &str, got: &Tensor, want: &Tensor, rel: f32) {
    let (diff, peak) = compare(name, got, want);
    assert!(diff <= rel * peak.max(1e-6), "{name}: off by {diff} (peak {peak}, allowed {rel} relative)");
}

#[test]
fn live_oracle() {
    let Some((mut o, dir)) = Oracle::open() else { return };
    let metal = std::env::var("ACESTEP_ORACLE_DEVICE").is_ok_and(|d| d == "metal");
    let dev = if metal { device() } else { Device::Cpu };
    // the GPU is shared with other sessions: hold its lock while the comparison runs there
    let _gpu = metal.then(|| {
        filmcraft_locks::acquire(filmcraft_locks::Resource::Gpu, "ACE-Step oracle test (Metal)", std::time::Duration::from_secs(1800), &mut |e| {
            eprintln!("waiting for the GPU: {e}")
        })
        .unwrap()
    });
    // the arithmetic is f32 on both devices (bf16-stored weights are widened exactly), so both
    // agree with the reference to float noise
    let (tight, loose) = (2e-4, 2e-3);
    let m = AceStep::load_on(&dir, dev.clone()).unwrap();
    let seconds = o.meta["seconds"].as_f64().unwrap();
    let mut req = Request::new(o.meta["caption"].as_str().unwrap(), seconds, o.meta["seed"].as_u64().unwrap());
    req.bpm = Some(124);
    req.key = Some("A minor".into());
    req.time_signature = Some(4);

    // 1. the conditioning text and its tokens
    assert_eq!(req.text_prompt(), o.meta["text_prompt"].as_str().unwrap());
    assert_eq!(req.lyrics_prompt(), o.meta["lyrics_text"].as_str().unwrap());
    let (text_ids, lyric_ids) = m.tokens(&req).unwrap();
    assert_eq!(text_ids, o.ids("text_ids"), "text tokens");
    assert_eq!(lyric_ids, o.ids("lyric_ids"), "lyric tokens");
    for case in o.meta["tokenizer_cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let want: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        assert_eq!(m.tokenizer.encode(text, usize::MAX).unwrap(), want, "{text:?}");
    }
    let tr = &o.meta["tokenizer_truncated"];
    let want: Vec<u32> = tr["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    assert_eq!(m.tokenizer.encode(tr["text"].as_str().unwrap(), 256).unwrap(), want, "truncation");
    eprintln!(
        "tokenizer: ids equal ({} text, {} lyric tokens, {} more cases)",
        text_ids.len(),
        lyric_ids.len(),
        o.meta["tokenizer_cases"].as_array().unwrap().len()
    );

    // 2. the text encoder
    let t0 = std::time::Instant::now();
    {
        let mut te = qwen3::TextEncoder::load(&dir.join(crate::catalog::TEXT_WEIGHTS), &m.text_cfg, &dev, m.wdtype).unwrap();
        check("text layer 0", &te.first_layer(&text_ids).unwrap().squeeze(0).unwrap(), &o.get("text_layer0", &Device::Cpu), tight);
        check("text hidden", &te.encode(&text_ids).unwrap().squeeze(0).unwrap(), &o.get("text_hidden", &Device::Cpu), loose);
        check("lyric embeds", &te.embed(&lyric_ids).unwrap().squeeze(0).unwrap(), &o.get("lyric_embeds", &Device::Cpu), 1e-6);
    }
    eprintln!("text encoder: {:.1?}", t0.elapsed());

    // 3. the condition encoder, from the reference's text states
    let t0 = std::time::Instant::now();
    let dit = m.load_dit().unwrap();
    eprintln!("DiT loaded in {:.1?}", t0.elapsed());
    let text_hidden = o.get("text_hidden", &dev).unsqueeze(0).unwrap();
    let lyric_embeds = o.get("lyric_embeds", &dev).unsqueeze(0).unwrap();
    check("lyric encoded", &dit.lyrics(&lyric_embeds).unwrap().squeeze(0).unwrap(), &o.get("lyric_encoded", &Device::Cpu), tight);
    check("timbre", &dit.timbre(&m.timbre_reference().unwrap()).unwrap().squeeze(0).unwrap(), &o.get("timbre", &Device::Cpu), tight);
    let enc = dit.encode(&text_hidden, &lyric_embeds, &m.timbre_reference().unwrap()).unwrap();
    check("encoder hidden", &enc.squeeze(0).unwrap(), &o.get("encoder_hidden", &Device::Cpu), tight);
    let mask = o.get("encoder_mask", &Device::Cpu).to_vec1::<f32>().unwrap();
    assert!(mask.iter().all(|x| *x == 1.0), "a single prompt packs without padding");
    let frames = req.frames();
    check("context", &m.context(frames).unwrap().squeeze(0).unwrap(), &o.get("context", &Device::Cpu), 0.0);

    // 4. each velocity pass from the reference's x_t (and the first pass's inner states)
    let enc_ref = o.get("encoder_hidden", &dev).unsqueeze(0).unwrap();
    let cond = dit.condition(&enc_ref).unwrap();
    check("dit_cond", &dit.embed_condition(&enc_ref).unwrap().squeeze(0).unwrap(), &o.get("dit_cond", &Device::Cpu), tight);
    let ctx = m.context(frames).unwrap();
    let ts = schedule(3).unwrap();
    for (i, &t) in ts.iter().enumerate() {
        let xt = o.get(&format!("xt_{i}"), &dev).unsqueeze(0).unwrap();
        let t0 = std::time::Instant::now();
        let mut trace = dit::Trace::default();
        let v = dit.velocity_traced(&xt, t, &ctx, &cond, if i == 0 { Some(&mut trace) } else { None }).unwrap();
        eprintln!("velocity pass {i} (t = {t}): {:.1?}", t0.elapsed());
        for (name, x) in trace.items {
            check(name, &x.squeeze(0).unwrap(), &o.get(name, &Device::Cpu), tight);
        }
        check(&format!("velocity {i}"), &v.squeeze(0).unwrap(), &o.get(&format!("vt_{i}"), &Device::Cpu), loose);
    }
    drop((dit, cond, ctx));

    // 5. the whole sampler and decoder from the reference's noise and prompt
    let noise = o.get("noise", &Device::Cpu).flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let t0 = std::time::Instant::now();
    let (music, detail) = m.generate_with(&req, &noise, &mut |_| true).unwrap();
    eprintln!("generated {:.1} s in {:.1?}", music.seconds(), t0.elapsed());
    let lat = Tensor::from_vec(detail.latents.clone(), (detail.frames, 64), &Device::Cpu).unwrap();
    check("latents", &lat, &o.get("latents", &Device::Cpu), 5e-3);

    // 6. the VAE alone, from the reference's latents
    let vae = m.load_vae().unwrap();
    let z = o.get("vae_z", &Device::Cpu).t().unwrap().contiguous().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (l, r) = vae.decode(&z, 40, &mut |_, _| true).unwrap();
    let got =
        Tensor::cat(&[Tensor::from_vec(l, (1, 40 * 1920), &Device::Cpu).unwrap(), Tensor::from_vec(r, (1, 40 * 1920), &Device::Cpu).unwrap()], 0).unwrap();
    check("vae (noise)", &got, &o.get("vae_out", &Device::Cpu), 1e-3);
    let zl = o.get("latents", &Device::Cpu).flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let t0 = std::time::Instant::now();
    let (l, r) = vae.decode(&zl, frames, &mut |_, _| true).unwrap();
    eprintln!("VAE decode {:.1} s: {:.1?}", l.len() as f64 / 48_000.0, t0.elapsed());
    let n = l.len();
    let got = Tensor::cat(&[Tensor::from_vec(l, (1, n), &Device::Cpu).unwrap(), Tensor::from_vec(r, (1, n), &Device::Cpu).unwrap()], 0).unwrap();
    check("vae (music)", &got, &o.get("audio", &Device::Cpu), 1e-3);
}
