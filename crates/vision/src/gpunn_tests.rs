//! The profile pass's token and picture passes on the GPU (`tok.wgsl`, `img.wgsl`) against the
//! CPU code they mirror, on random data and synthetic pictures: every operation the person
//! detector's decoder and the pose network's head use. Skipped without a GPU (the CPU is then
//! the only path).

use std::sync::Arc;

use crate::dfine::{Memory, deform_cpu, fdr_box, lqe_stats_cpu};
use crate::gpunn::{GM, Nn, Run, TAct};
use crate::layers::{Attention, Backend, Feat, Linear, Norm, sigmoid, softmax_rows};

fn gpu() -> Option<Arc<Nn>> {
    let nn = Nn::shared();
    if nn.is_none() {
        eprintln!("SKIPPED (profile GPU passes): no GPU");
    }
    nn
}

fn noise(seed: u32) -> impl FnMut() -> f32 {
    let mut s = seed.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s % 20_000) as f32 / 10_000.0 - 1.0
    }
}

fn rand(next: &mut impl FnMut() -> f32, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| next() * scale).collect()
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "lengths");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn read(r: &Run<'_>, m: &GM) -> Vec<f32> {
    r.read_mats(&[m]).unwrap().remove(0)
}

fn linear(next: &mut impl FnMut() -> f32, cin: usize, cout: usize, bias: bool, nn: &Nn) -> Linear {
    let w = rand(next, cin * cout, 0.5);
    let b = bias.then(|| rand(next, cout, 0.5));
    let mut l = Linear::new(&w, b, cin, cout).unwrap();
    l.upload(nn).unwrap();
    l
}

fn gelu(x: f32) -> f32 {
    // the reference's erf (dfine.rs), through the encoder layer's activation
    let s = x.signum();
    let a = (x / std::f32::consts::SQRT_2).abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let y = 1.0 - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_74) * t + 0.254_829_6) * t * (-a * a).exp();
    0.5 * x * (1.0 + s * y)
}

#[test]
fn linear_layers_norms_and_softmax_match_the_cpu() {
    let Some(nn) = gpu() else { return };
    let mut next = noise(3);
    for (rows, cin, cout) in [(1usize, 4usize, 4usize), (17, 48, 256), (300, 128, 132), (37, 33, 7), (1100, 128, 80)] {
        for bias in [false, true] {
            let l = linear(&mut next, cin, cout, bias, &nn);
            let x = rand(&mut next, rows * cin, 1.0);
            let res = rand(&mut next, rows * cout, 1.0);
            let cpu = l.forward(&x).unwrap();
            let r = Run::new(&nn);
            let gx = r.upload_mat(&x, rows, cin).unwrap();
            let gres = r.upload_mat(&res, rows, cout).unwrap();
            for (act, f) in
                [(TAct::None, (|v| v) as fn(f32) -> f32), (TAct::Relu, |v: f32| v.max(0.0)), (TAct::Gelu, gelu), (TAct::Silu, |v: f32| v / (1.0 + (-v).exp()))]
            {
                let got = read(&r, &r.linear(&gx, &l, act, None, None).unwrap());
                let want: Vec<f32> = cpu.iter().map(|v| f(*v)).collect();
                assert!(max_err(&got, &want) < 1e-4, "{rows}×{cin}→{cout} bias {bias} {act:?}: {}", max_err(&got, &want));
            }
            // a residual after the activation, then a clamp
            let got = read(&r, &r.linear(&gx, &l, TAct::Relu, Some(&gres), Some(0.5)).unwrap());
            let want: Vec<f32> = cpu.iter().zip(&res).map(|(v, s)| (v.max(0.0) + s).clamp(-0.5, 0.5)).collect();
            assert!(max_err(&got, &want) < 1e-4);
            // hostile shapes are errors, not panics
            assert!(r.linear(&gres, &l, TAct::None, None, None).is_err() || cin == cout);
            assert!(r.linear(&gx, &l, TAct::None, Some(&gx), None).is_err() || cin == cout);
        }
    }
    // layer norm, scale norm, softmax
    let (rows, cols) = (300usize, 128usize);
    let x = rand(&mut next, rows * cols, 3.0);
    let mut norm = Norm::new(rand(&mut next, cols, 1.0), rand(&mut next, cols, 1.0), 1e-5).unwrap();
    norm.upload(&nn);
    let r = Run::new(&nn);
    let gx = r.upload_mat(&x, rows, cols).unwrap();
    let mut want = x.clone();
    norm.apply(&mut want);
    assert!(max_err(&read(&r, &r.layernorm(&gx, &norm).unwrap()), &want) < 1e-4);
    let mut want = x.clone();
    crate::rtmpose::scale_norm(&mut want, cols, 0.7);
    assert!(max_err(&read(&r, &r.scalenorm(&gx, 0.7).unwrap()), &want) < 1e-5);
    let mut want = x.clone();
    softmax_rows(&mut want, cols);
    assert!(max_err(&read(&r, &r.softmax(&gx).unwrap()), &want) < 1e-6);
    // a norm of another width
    let narrow = r.upload_mat(&x[..rows * 4], rows, 4).unwrap();
    assert!(r.layernorm(&narrow, &norm).is_err());
}

#[test]
fn attention_matches_the_cpu() {
    let Some(nn) = gpu() else { return };
    let mut next = noise(5);
    for (n, d, heads) in [(300usize, 128usize, 8usize), (220, 128, 8), (5, 16, 2), (1, 8, 1)] {
        let mut lin = || {
            let w = rand(&mut next, d * d, 0.3);
            let b = rand(&mut next, d, 0.3);
            Linear::new(&w, Some(b), d, d).unwrap()
        };
        let mut at = Attention { q: lin(), k: lin(), v: lin(), o: lin(), heads };
        at.upload(&nn).unwrap();
        let (qk, v, res) = (rand(&mut next, n * d, 1.0), rand(&mut next, n * d, 1.0), rand(&mut next, n * d, 1.0));
        let cpu: Vec<f32> = at.forward(&qk, &v).unwrap().iter().zip(&res).map(|(a, b)| a + b).collect();
        let r = Run::new(&nn);
        let (gqk, gv, gres) = (r.upload_mat(&qk, n, d).unwrap(), r.upload_mat(&v, n, d).unwrap(), r.upload_mat(&res, n, d).unwrap());
        let got = read(&r, &r.mha(&at, &gqk, &gv, Some(&gres)).unwrap());
        assert!(max_err(&got, &cpu) < 1e-4, "{n} tokens, {d}/{heads}: {}", max_err(&got, &cpu));
    }
}

#[test]
fn token_shapes_selection_and_boxes_match_the_cpu() {
    let Some(nn) = gpu() else { return };
    let mut next = noise(9);
    let r = Run::new(&nn);
    // a map's planes to token rows (at an offset) and back
    let f = Feat::new(rand(&mut next, 8 * 5 * 7, 1.0), 1, 8, 5, 7).unwrap();
    let g = r.input(&f).unwrap();
    let rows = r.mat(3 + 35, 8).unwrap();
    // the first rows: some other level's tokens
    let other = Feat::new(rand(&mut next, 8 * 3, 1.0), 1, 8, 1, 3).unwrap();
    r.planes_into_rows(&r.input(&other).unwrap(), &rows, 0).unwrap();
    r.planes_into_rows(&g, &rows, 3).unwrap();
    let got = read(&r, &rows);
    assert_eq!(&got[..24], &other.tokens(0)[..]);
    assert_eq!(&got[24..], &f.tokens(0)[..]);
    let back = r.rows_to_planes(&rows, 3, 5, 7).unwrap();
    assert_eq!(r.read(&[&back]).unwrap().remove(0), f);
    assert!(r.planes_into_rows(&g, &rows, 4).is_err(), "past the end");
    assert!(r.rows_to_planes(&rows, 4, 5, 7).is_err());
    // the best rows: ties keep their order (a stable sort's), fewer than k values give all
    let scores: Vec<f32> = (0..1100).map(|i| ((i * 37) % 101) as f32 / 7.0 - 3.0).collect();
    let classes = 5;
    let mat: Vec<f32> = scores.iter().flat_map(|s| (0..classes).map(move |c| s - c as f32 * 0.25)).collect();
    let gm = r.upload_mat(&mat, 1100, classes).unwrap();
    let best = r.rowmax(&gm).unwrap();
    assert_eq!(read(&r, &best), scores);
    let mut order: Vec<usize> = (0..1100).collect();
    order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
    for k in [300usize, 1100, 2000] {
        let top = r.top(&best, k).unwrap();
        let got: Vec<usize> = read(&r, &top).iter().map(|v| *v as usize).collect();
        assert_eq!(got, order[..k.min(1100)], "k {k}");
    }
    // gather rows, the first boxes, the first layer's boxes
    let top = r.top(&best, 300).unwrap();
    let picked = read(&r, &r.gather(&gm, &top).unwrap());
    let want: Vec<f32> = order[..300].iter().flat_map(|i| mat[i * classes..(i + 1) * classes].to_vec()).collect();
    assert_eq!(picked, want);
    let reg = rand(&mut next, 300 * 4, 2.0);
    let mut anchors = rand(&mut next, 1100 * 4, 3.0);
    anchors[order[0] * 4] = f32::MAX;
    let (greg, ganch) = (r.upload_mat(&reg, 300, 4).unwrap(), r.upload_mat(&anchors, 1100, 4).unwrap());
    let refs = r.refs_init(&greg, &ganch, &top).unwrap();
    let want: Vec<f32> = (0..300 * 4).map(|i| sigmoid(reg[i] + anchors[order[i / 4] * 4 + i % 4])).collect();
    let got = read(&r, &refs);
    assert!(max_err(&got, &want) < 1e-6, "{}", max_err(&got, &want));
    let pre = rand(&mut next, 300 * 4, 2.0);
    let gpre = r.upload_mat(&pre, 300, 4).unwrap();
    let inv = |x: f32| {
        let x = x.clamp(0.0, 1.0);
        (x.max(1e-5) / (1.0 - x).max(1e-5)).ln()
    };
    let want: Vec<f32> = pre.iter().zip(&got).map(|(p, b)| sigmoid(p + inv(*b))).collect();
    assert!(max_err(&read(&r, &r.init_box(&gpre, &refs).unwrap()), &want) < 1e-5);
    // FDR boxes and the quality statistics from 33-bin distributions
    let bins = 33;
    let project: Vec<f32> = (0..bins).map(|i| (i as f32 - 16.0) / 4.0).collect();
    let corners = rand(&mut next, 300 * 4 * bins, 4.0);
    let gc = r.upload_mat(&corners, 300, 4 * bins).unwrap();
    let init: Vec<f32> = got.clone();
    let gi = r.upload_mat(&init, 300, 4).unwrap();
    let gp = nn.upload(&project);
    let boxes = read(&r, &r.to_box(&gi, &gc, &gp, bins, 4.0).unwrap());
    let want: Vec<f32> =
        (0..300).flat_map(|q| fdr_box(&project, 4.0, bins, init[q * 4..q * 4 + 4].try_into().unwrap(), &corners[q * 4 * bins..(q + 1) * 4 * bins])).collect();
    assert!(max_err(&boxes, &want) < 1e-5, "{}", max_err(&boxes, &want));
    let stats = read(&r, &r.lqe_stats(&gc, bins, 4).unwrap());
    let want = lqe_stats_cpu(&corners, bins, 4);
    assert!(max_err(&stats, &want) < 1e-6, "{}", max_err(&stats, &want));
    // equal probabilities (a flat distribution): the statistics still match
    let flat = vec![0.5f32; 2 * 4 * bins];
    let stats = read(&r, &r.lqe_stats(&r.upload_mat(&flat, 2, 4 * bins).unwrap(), bins, 4).unwrap());
    assert!(max_err(&stats, &lqe_stats_cpu(&flat, bins, 4)) < 1e-6);
    // the gate, side by side, masked rows, a scalar a row
    let (h1, cross, g) = (rand(&mut next, 6 * 4, 1.0), rand(&mut next, 6 * 4, 1.0), rand(&mut next, 6 * 8, 2.0));
    let (gh, gx, gg) = (r.upload_mat(&h1, 6, 4).unwrap(), r.upload_mat(&cross, 6, 4).unwrap(), r.upload_mat(&g, 6, 8).unwrap());
    let want: Vec<f32> = (0..24).map(|i| sigmoid(g[(i / 4) * 8 + i % 4]) * h1[i] + sigmoid(g[(i / 4) * 8 + 4 + i % 4]) * cross[i]).collect();
    assert!(max_err(&read(&r, &r.gate(&gg, &gh, &gx).unwrap()), &want) < 1e-6);
    let both = read(&r, &r.concat2(&gh, &gx).unwrap());
    assert_eq!(&both[..8], &[h1[0], h1[1], h1[2], h1[3], cross[0], cross[1], cross[2], cross[3]]);
    let keep = r.upload_mat(&[1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 6, 1).unwrap();
    let masked = read(&r, &r.mask_rows(&gh, &keep).unwrap());
    assert_eq!(&masked[4..8], &[0.0; 4]);
    assert_eq!(&masked[..4], &h1[..4]);
    let plus = read(&r, &r.add_rowscalar(&gh, &keep).unwrap());
    assert_eq!(plus[0], h1[0] + 1.0);
    assert_eq!(plus[4], h1[4]);
    // hostile shapes
    assert!(r.gate(&gh, &gh, &gx).is_err());
    assert!(r.concat2(&gh, &gg.clone()).is_ok());
    assert!(r.add(&gh, &gg).is_err());
    assert!(r.to_box(&gh, &gc, &gp, bins, 4.0).is_err());
    assert!(r.lqe_stats(&gc, bins, 0).is_err());
}

#[test]
fn deformable_sampling_matches_the_cpu() {
    let Some(nn) = gpu() else { return };
    let mut next = noise(13);
    // two levels (8 × 10 and 4 × 5), 8 heads of 16 channels, 6 points a level
    let (d, heads) = (128usize, 8usize);
    let levels = [(0usize, 8usize, 10usize), (80, 4, 5)];
    let len = 80 + 20;
    let points = [6usize, 6];
    let total: usize = points.iter().sum();
    let level_of: Vec<usize> = points.iter().enumerate().flat_map(|(l, n)| std::iter::repeat_n(l, *n)).collect();
    let point_scale: Vec<f32> = (0..total).map(|_| 1.0 / 6.0).collect();
    for nq in [300usize, 7] {
        let tokens = rand(&mut next, len * d, 1.0);
        let offsets = rand(&mut next, nq * heads * total * 2, 3.0);
        let mut weights = rand(&mut next, nq * heads * total, 2.0);
        softmax_rows(&mut weights, total);
        // boxes: inside, at the edges, partly and wholly outside the picture
        let refs: Vec<[f32; 4]> =
            (0..nq).map(|i| [(i % 13) as f32 / 10.0 - 0.1, (i % 7) as f32 / 5.0 - 0.1, 0.05 + (i % 5) as f32 / 10.0, 0.1 + (i % 3) as f32 / 4.0]).collect();
        let mem = Memory { tokens: &tokens, levels: &levels };
        let want = deform_cpu(&mem, d, &offsets, &weights, &refs, &point_scale, &level_of, heads, total, 0.5);
        let r = Run::new(&nn);
        let gt = r.upload_mat(&tokens, len, d).unwrap();
        let go = r.upload_mat(&offsets, nq, heads * total * 2).unwrap();
        let gw = r.upload_mat(&weights, nq, heads * total).unwrap();
        let gr = r.upload_mat(&refs.iter().flatten().copied().collect::<Vec<_>>(), nq, 4).unwrap();
        let meta: Vec<f32> = point_scale.iter().copied().chain(level_of.iter().map(|l| *l as f32)).collect();
        let gp = nn.upload(&meta);
        let lv: Vec<f32> = levels.iter().flat_map(|(s, h, w)| [*s as f32, *h as f32, *w as f32]).collect();
        let gl = r.upload_mat(&lv, 2, 3).unwrap();
        let got = read(&r, &r.deform(&gt, &go, &gw, &gr, &gp, &gl, heads, total, 0.5).unwrap());
        assert!(max_err(&got, &want) < 1e-5, "{nq} queries: {}", max_err(&got, &want));
        // hostile shapes
        assert!(r.deform(&gt, &go, &gw, &gr, &gp, &gl, 7, total, 0.5).is_err());
        assert!(r.deform(&gt, &gw, &gw, &gr, &gp, &gl, heads, total, 0.5).is_err());
    }
}

#[test]
fn the_pose_heads_parts_match_the_cpu() {
    let Some(nn) = gpu() else { return };
    let mut next = noise(17);
    let (e, s, k) = (32usize, 16usize, crate::head::KEYPOINTS);
    for n in [1usize, 3] {
        let uv = rand(&mut next, n * k * (2 * e + s), 1.0);
        let (gamma, beta) = (rand(&mut next, 2 * s, 1.0), rand(&mut next, 2 * s, 0.5));
        let want = crate::rtmpose::gau_cpu(&uv, &gamma, &beta, e, s, n);
        let r = Run::new(&nn);
        let guv = r.upload_mat(&uv, n * k, 2 * e + s).unwrap();
        let got = read(&r, &r.gau(&guv, &nn.upload(&gamma), &nn.upload(&beta), e, s, k).unwrap());
        assert!(max_err(&got, &want) < 1e-4, "{n} crops: {}", max_err(&got, &want));
        assert!(r.gau(&guv, &nn.upload(&gamma), &nn.upload(&beta), e + 1, s, k).is_err());
    }
    // the residual's scale, and the first maximum of each row (ties: the first)
    let r = Run::new(&nn);
    let (a, b, sc) = (rand(&mut next, 4 * 6, 1.0), rand(&mut next, 4 * 6, 1.0), rand(&mut next, 6, 1.0));
    let got = read(&r, &r.scale_add(&r.upload_mat(&a, 4, 6).unwrap(), &r.upload_mat(&b, 4, 6).unwrap(), &nn.upload(&sc)).unwrap());
    let want: Vec<f32> = (0..24).map(|i| sc[i % 6] * a[i] + b[i]).collect();
    assert!(max_err(&got, &want) < 1e-6);
    let rows = [vec![0.1f32, 0.7, 0.3, 0.7], vec![-5.0, -6.0, -5.0, -7.0], vec![f32::NAN, 1.0, 2.0, 2.0], vec![-1.0; 4]];
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let got = read(&r, &r.argmax(&r.upload_mat(&flat, 4, 4).unwrap()).unwrap());
    for (i, row) in rows.iter().enumerate() {
        let want = crate::rtmpose::argmax(row);
        assert_eq!((got[i * 2] as usize, got[i * 2 + 1]), want, "row {i}");
    }
}

/// A picture with structure at every scale: gradients, a checkerboard and noise.
fn picture(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut next = noise(seed);
    let mut v = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            v[i] = (x * 255 / w.max(1)) as u8;
            v[i + 1] = if (x / 3 + y / 5) % 2 == 0 { 230 } else { 20 };
            v[i + 2] = ((next() + 1.0) * 127.0) as u8;
            v[i + 3] = 255;
        }
    }
    v
}

#[test]
fn pictures_are_prepared_and_cropped_as_on_the_cpu() {
    let Some(nn) = gpu() else { return };
    for (w, h, ow, oh) in [(1920usize, 1080usize, 640usize, 352usize), (333, 217, 96, 64), (64, 48, 64, 48), (50, 30, 96, 64)] {
        let rgba = picture(w, h, (w * h) as u32);
        let want = crate::dfine::prepare_exact(&rgba, w, h, ow, oh).unwrap();
        let r = Run::new(&nn);
        let img = r.upload_rgba(&rgba, w, h).unwrap();
        let got = r.read(&[&r.resize_area(&img, ow, oh).unwrap()]).unwrap().remove(0);
        assert_eq!((got.c, got.h, got.w), (3, oh, ow));
        assert!(max_err(&got.v, &want.v) < 1e-6, "{w}×{h} → {ow}×{oh}: {}", max_err(&got.v, &want.v));
        // people's boxes: inside, at an edge, partly and wholly outside the picture
        let boxes = [
            [w as f32 * 0.3, h as f32 * 0.2, w as f32 * 0.2, h as f32 * 0.6],
            [-20.0, 5.0, 40.0, 90.0],
            [w as f32 - 10.0, h as f32 - 30.0, 50.0, 60.0],
            [1e5, -1e5, 5.0, 5.0],
        ];
        let (cpu, at): (Vec<Vec<f32>>, Vec<[f32; 3]>) = boxes
            .iter()
            .map(|b| {
                let (c, a) = crate::rtmpose::crop_for_test(&rgba, w, h, b).unwrap();
                (c, a)
            })
            .unzip();
        let got = r.read(&[&r.crops(&img, &at).unwrap()]).unwrap().remove(0);
        let want: Vec<f32> = cpu.into_iter().flatten().collect();
        assert!(max_err(&got.v, &want) < 1e-4, "{w}×{h} crops: {}", max_err(&got.v, &want));
    }
    // hostile pictures and places: errors, not panics
    let r = Run::new(&nn);
    assert!(r.upload_rgba(&[0u8; 10], 4, 4).is_err());
    assert!(r.upload_rgba(&[], 0, 4).is_err());
    let img = r.upload_rgba(&[0u8; 64], 4, 4).unwrap();
    assert!(r.resize_area(&img, 0, 3).is_err());
    assert!(r.crops(&img, &[]).is_err());
    assert!(r.crops(&img, &[[f32::NAN, 0.0, 1.0]]).is_err());
}

/// The GPU's tensors survive several runs on several threads at once, as the redaction looks at
/// several frames at once.
#[test]
fn token_passes_run_on_several_threads_at_once() {
    let Some(nn) = gpu() else { return };
    let mut next = noise(21);
    let l = linear(&mut next, 64, 96, true, &nn);
    let x = rand(&mut next, 200 * 64, 1.0);
    let want = l.forward(&x).unwrap();
    std::thread::scope(|s| {
        for _ in 0..6 {
            s.spawn(|| {
                for _ in 0..8 {
                    let r = Run::new(&nn);
                    let g = r.upload_mat(&x, 200, 64).unwrap();
                    let got = read(&r, &r.linear(&g, &l, TAct::None, None, None).unwrap());
                    assert!(max_err(&got, &want) < 1e-4);
                }
            });
        }
    });
    assert!(!crate::gpu::failed(), "the GPU is still in use");
}
