//! Run the scene models on still pictures and print what they say and how long it took:
//!
//! `cargo run --release -p filmcraft-scenes --features model --example scenes_probe -- <models dir> <picture>… [--model qwen3-vl-2b] [--detail] [--query "a robot"]`

use std::time::Instant;

use filmcraft_scenes::catalog;
use filmcraft_scenes::model::qwen3vl::{Frame, Options, Qwen3Vl};
use filmcraft_scenes::model::{aesthetic::Aesthetic, device, siglip::Siglip};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let models = std::path::PathBuf::from(args.first().ok_or("usage: scenes_probe <models dir> <picture>…")?);
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let detail = args.iter().any(|a| a == "--detail");
    let model = opt("--model").unwrap_or_else(|| catalog::QWEN_2B.to_string());
    // pictures: what is not an option or an option's value
    let mut pics: Vec<&String> = Vec::new();
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        if ["--model", "--query", "--repeat", "--max-new"].contains(&a.as_str()) {
            it.next();
        } else if !a.starts_with("--") {
            pics.push(a);
        }
    }
    let imgs: Vec<(Vec<u8>, usize, usize)> = pics
        .iter()
        .map(|p| {
            let i = image::open(p).map(|i| i.to_rgba8()).map_err(|e| format!("{p}: {e}"))?;
            let (w, h) = (i.width() as usize, i.height() as usize);
            Ok::<_, String>((i.into_raw(), w, h))
        })
        .collect::<Result<_, _>>()?;
    let dev = device();
    eprintln!("device: {:?}", dev.location());
    if args.iter().any(|a| a == "--bench") {
        // skinny matrix products, as decoding does them
        use candle_core::{DType, Tensor};
        let w = Tensor::randn(0f32, 0.02, (6144, 2048), &dev)?.to_dtype(DType::BF16)?;
        for m in [1usize, 2, 5, 8, 16, 32] {
            let x = Tensor::randn(0f32, 1.0, (m, 2048), &dev)?.to_dtype(DType::BF16)?;
            let wt = w.t()?;
            let _ = x.matmul(&wt)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
            let t = Instant::now();
            let mut acc = x.matmul(&wt)?;
            for _ in 0..99 {
                acc = x.matmul(&wt)?;
            }
            let _ = acc.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
            eprintln!("matmul {m}x2048 @ 2048x6144 bf16: {:.3} ms", t.elapsed().as_secs_f64() * 10.0);
        }
        return Ok(());
    }
    let t = Instant::now();
    let mut vlm = Qwen3Vl::load(&models.join(&model), &model, &dev)?;
    eprintln!("{model} loaded in {:.1}s", t.elapsed().as_secs_f64());
    // the pictures as one batch (each its own shot); a batch shares a size, so group by it
    let mut groups: std::collections::BTreeMap<(usize, usize), Vec<usize>> = Default::default();
    let max_new = opt("--max-new").and_then(|v| v.parse().ok()).unwrap_or(if detail { 640 } else { 320 });
    let o = Options { max_pixels: if detail { 640 * 360 } else { 448 * 448 }, max_new };
    if let Some(n) = opt("--repeat").and_then(|v| v.parse::<usize>().ok()) {
        // one picture as a batch of 1, 2, … n shots: what batching buys
        let (p, w, h) = imgs.first().ok_or("no picture")?;
        for b in (1..=n).filter(|b| b.is_power_of_two() || *b == n) {
            let shots: Vec<Vec<Frame>> = (0..b).map(|_| vec![Frame { rgba: p, w: *w, h: *h }]).collect();
            let t = Instant::now();
            let (_, stats) = vlm.generate(
                &shots,
                filmcraft_scenes::prompt::SYSTEM,
                &filmcraft_scenes::prompt::shot_prompt(1, detail),
                filmcraft_scenes::prompt::PREFILL,
                &o,
                &|| false,
            )?;
            let secs = t.elapsed().as_secs_f64();
            let steps = stats.generated / b;
            eprintln!(
                "batch {b}: {secs:.2}s (vision {:.2}s, prefill {:.2}s of {} tokens each), {steps} steps, {:.1} ms/step, {:.2} shots/s",
                stats.vision_seconds,
                stats.prefill_seconds,
                stats.prompt_tokens,
                1000.0 * stats.decode_seconds / steps.max(1) as f64,
                b as f64 / secs
            );
        }
        return Ok(());
    }
    for (i, (_, w, h)) in imgs.iter().enumerate() {
        groups.entry(vlm.frame_size(*w, *h, o.max_pixels)).or_default().push(i);
    }
    for (size, idx) in groups {
        let shots: Vec<Vec<Frame>> = idx.iter().map(|i| vec![Frame { rgba: &imgs[*i].0, w: imgs[*i].1, h: imgs[*i].2 }]).collect();
        let t = Instant::now();
        let (answers, stats) = vlm.generate(
            &shots,
            filmcraft_scenes::prompt::SYSTEM,
            &filmcraft_scenes::prompt::shot_prompt(1, detail),
            filmcraft_scenes::prompt::PREFILL,
            &o,
            &|| false,
        )?;
        let secs = t.elapsed().as_secs_f64();
        eprintln!("batch of {} at {size:?}: {secs:.2}s ({stats:?})", shots.len());
        for (i, a) in idx.iter().zip(&answers) {
            println!("== {}\n{a}", pics[*i]);
            match filmcraft_scenes::json::parse_description(a, detail) {
                Ok(d) => println!("-> {}", serde_json::to_string(&d)?),
                Err(e) => println!("-> unreadable: {e}"),
            }
        }
    }
    drop(vlm);
    let pics_ref: Vec<(&[u8], usize, usize)> = imgs.iter().map(|(p, w, h)| (p.as_slice(), *w, *h)).collect();
    if catalog::installed_id(&models, catalog::SIGLIP) {
        let t = Instant::now();
        let s = Siglip::load(&models.join(catalog::SIGLIP), &dev)?;
        let e = s.embed_images(&pics_ref)?;
        eprintln!("siglip: {} embeddings of {} in {:.2}s", e.len(), e.first().map_or(0, Vec::len), t.elapsed().as_secs_f64());
        if let Some(q) = opt("--query") {
            let qe = s.embed_text(&q)?;
            let k = s.calibration();
            for (p, e) in pics.iter().zip(&e) {
                let cos: f32 = e.iter().zip(&qe).map(|(a, b)| a * b).sum();
                println!("{p}: \"{q}\" cos {cos:.3}, p {:.3}", 1.0 / (1.0 + (-(k.scale * cos + k.bias)).exp()));
            }
        }
    }
    if catalog::installed_id(&models, catalog::AESTHETIC) {
        let t = Instant::now();
        let a = Aesthetic::load(&models.join(catalog::AESTHETIC), &dev)?;
        let s = a.score(&pics_ref)?;
        eprintln!("aesthetic in {:.2}s", t.elapsed().as_secs_f64());
        for (p, s) in pics.iter().zip(&s) {
            println!("{p}: aesthetic {s:.2}");
        }
    }
    Ok(())
}
