//! Live tests of the profile pass's networks: against ONNX Runtime's outputs
//! (`tools/profile_oracle.py`), and on real pictures. Ignored: they need the models
//! (`FILMCRAFT_VISION_MODELS`, holding `privacy-detectors/`) and, for the oracle tests, the
//! oracle's directory (`FILMCRAFT_PROFILE_ORACLE`).

use std::path::PathBuf;

use crate::catalog;
use crate::dfine::{self, Dfine};
use crate::layers::Feat;
use crate::rtmpose::{INPUT_H, INPUT_W, RtmPose};

fn models() -> Option<PathBuf> {
    Some(catalog::package_dir(&PathBuf::from(std::env::var_os("FILMCRAFT_VISION_MODELS")?)))
}

fn oracle() -> Option<PathBuf> {
    Some(PathBuf::from(std::env::var_os("FILMCRAFT_PROFILE_ORACLE")?))
}

fn f32s(p: &std::path::Path) -> Vec<f32> {
    std::fs::read(p).unwrap().as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

fn person() -> Dfine {
    let d = models().unwrap();
    Dfine::load(&d.join(catalog::PERSON_WEIGHTS.name), &d.join(catalog::PERSON_CONFIG.name)).unwrap()
}

#[test]
#[ignore = "needs the profile models and the oracle's outputs"]
fn dfine_matches_onnx_runtime() {
    let (Some(_), Some(o)) = (models(), oracle()) else { return };
    let net = person();
    let x = Feat::new(f32s(&o.join("dfine_input.bin")), 1, 3, 640, 640).unwrap();
    let t = std::time::Instant::now();
    let (logits, boxes) = net.raw(&x).unwrap();
    eprintln!("D-FINE at 640 × 640: {:?}", t.elapsed());
    let (want_l, want_b) = (f32s(&o.join("dfine_logits.bin")), f32s(&o.join("dfine_boxes.bin")));
    assert_eq!(logits.len(), want_l.len());
    let classes = logits.len() / boxes.len();
    // the people the reference finds (score ≥ 0.3), each found here at the same place and score
    let wb: Vec<[f32; 4]> = want_b.as_chunks::<4>().0.to_vec();
    let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
    let mut checked = 0;
    for (q, b) in wb.iter().enumerate() {
        let p = sig(want_l[q * classes]);
        if p < 0.3 {
            continue;
        }
        let best =
            (0..boxes.len()).map(|k| (k, boxes[k].iter().zip(b).map(|(a, c)| (a - c).abs()).fold(0.0, f32::max))).min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        let got = sig(logits[best.0 * classes]);
        eprintln!("person {p:.4} at {b:?}: here {got:.4}, box off by {:.2e}", best.1);
        assert!(best.1 < 2e-3, "box {b:?} off by {}", best.1);
        assert!((got - p).abs() < 2e-3, "score {p} vs {got}");
        checked += 1;
    }
    assert!(checked > 0, "the oracle's picture shows nobody");
    // query by query (the same queries in the same order unless two tokens tie)
    let same = (0..boxes.len()).filter(|&k| boxes[k].iter().zip(&wb[k]).all(|(a, b)| (a - b).abs() < 1e-3)).count();
    eprintln!("{same} of {} queries agree to 1e-3", boxes.len());
    assert!(same * 10 >= boxes.len() * 9);
}

#[test]
#[ignore = "needs the profile models and the oracle's outputs"]
fn rtmpose_matches_onnx_runtime() {
    let (Some(d), Some(o)) = (models(), oracle()) else { return };
    let net = RtmPose::load(&d.join(catalog::POSE.name)).unwrap();
    let x = Feat::new(f32s(&o.join("pose_input.bin")), 2, 3, INPUT_H, INPUT_W).unwrap();
    let t = std::time::Instant::now();
    let (sx, sy) = net.raw(&x).unwrap();
    eprintln!("RTMPose, 2 crops: {:?}", t.elapsed());
    for (got, want) in [(sx, f32s(&o.join("pose_x.bin"))), (sy, f32s(&o.join("pose_y.bin")))] {
        assert_eq!(got.len(), want.len());
        let e = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        let scale = want.iter().map(|v| v.abs()).fold(0.0, f32::max);
        eprintln!("SimCC max error {e:.2e} (largest value {scale:.3})");
        assert!(e < 1e-3 * scale.max(1.0), "{e}");
    }
}

/// Speed of the profile pass on a 1920 × 1080 picture (`FILMCRAFT_PROFILE_PICTURE`, else the
/// oracle's `people.png` scaled), release build.
#[test]
#[ignore = "needs the profile models"]
fn profile_pass_speed() {
    let Some(_) = models() else { return };
    let path = std::env::var_os("FILMCRAFT_PROFILE_PICTURE").map(PathBuf::from).or_else(|| oracle().map(|o| o.join("../people.png"))).unwrap();
    let img = image::open(&path).unwrap().resize_exact(1920, 1080, image::imageops::FilterType::Triangle).to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rgba = img.into_raw();
    let dir = PathBuf::from(std::env::var_os("FILMCRAFT_VISION_MODELS").unwrap());
    for gpu in [true, false] {
        let det = crate::ProfileDetector::load_on(&dir, gpu).unwrap();
        let mut net = person();
        if gpu {
            net.use_gpu();
        }
        for side in [640usize, 800] {
            let x = dfine::prepare(&rgba, w, h, side).unwrap();
            let mut best = std::time::Duration::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                let _ = net.people(&x, 0.4).unwrap();
                best = best.min(t.elapsed());
            }
            eprintln!("{}: D-FINE at {}×{}: {best:?}", det.device_name(), x.w, x.h);
            // the parts: the encoder (GPU or CPU) and the decoder (CPU)
            let (mut enc, mut dec) = (std::time::Duration::MAX, std::time::Duration::MAX);
            for _ in 0..5 {
                let t = std::time::Instant::now();
                let outs = match crate::gpunn::GpuUse::run(&net.gpu, |r| net.encode(r, &x)) {
                    Some(o) => o,
                    None => net.encode(&crate::layers::Cpu, &x).unwrap(),
                };
                enc = enc.min(t.elapsed());
                let t = std::time::Instant::now();
                let _ = net.decode(&outs).unwrap();
                dec = dec.min(t.elapsed());
            }
            eprintln!("    encoder {enc:?}, decoder {dec:?}");
        }
        let mut best = std::time::Duration::MAX;
        let mut n = 0;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let people = det.people(&rgba, w, h).unwrap();
            best = best.min(t.elapsed());
            n = people.len();
        }
        eprintln!("{}: people and poses ({n} people): {best:?} = {:.1} frames a second", det.device_name(), 1.0 / best.as_secs_f64());
    }
}

/// What the face detector and the profile pass see in each PNG of `FILMCRAFT_PROFILE_PICTURES`:
/// printed, and drawn into `annotated/` there (people blue, face keypoints yellow, YuNet's faces
/// green, heads added red, heads YuNet found white, backs of heads grey).
#[test]
#[ignore = "needs the models and pictures (FILMCRAFT_VISION_MODELS, FILMCRAFT_PROFILE_PICTURES)"]
fn profile_on_pictures() {
    let (Some(_), Some(dir)) = (models(), std::env::var_os("FILMCRAFT_PROFILE_PICTURES").map(PathBuf::from)) else { return };
    let models_dir = PathBuf::from(std::env::var_os("FILMCRAFT_VISION_MODELS").unwrap());
    let det = crate::ProfileDetector::load_on(&models_dir, true).unwrap();
    let faces = crate::Detectors::load(&models_dir, true, false).unwrap();
    let out = dir.join("annotated");
    std::fs::create_dir_all(&out).unwrap();
    let mut paths: Vec<PathBuf> =
        std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "png")).collect();
    paths.sort();
    for p in paths {
        let img = image::open(&p).unwrap().to_rgba8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgba = img.as_raw().clone();
        let f = faces.face.as_ref().unwrap();
        let mut found = f.detect(&rgba, w, h, 640, 0.6).unwrap();
        found.extend(f.detect(&rgba, w, h, 1280.min(w.max(h)), 0.6).unwrap());
        let found = crate::post::nms(found, 0.3, 750);
        let people = det.people(&rgba, w, h).unwrap();
        let added = crate::head::candidates(&people, &found, &det.params.head, w as f32, h as f32);
        let mut canvas = image::RgbaImage::from_raw(w as u32, h as u32, rgba).unwrap();
        let rect = |c: &mut image::RgbaImage, b: [f32; 4], col: [u8; 4], t: i64| {
            for k in 0..t {
                let (x0, y0, x1, y1) = (b[0] as i64 - k, b[1] as i64 - k, (b[0] + b[2]) as i64 + k, (b[1] + b[3]) as i64 + k);
                for x in x0..=x1 {
                    for y in [y0, y1] {
                        if x >= 0 && y >= 0 && (x as u32) < c.width() && (y as u32) < c.height() {
                            c.put_pixel(x as u32, y as u32, image::Rgba(col));
                        }
                    }
                }
                for y in y0..=y1 {
                    for x in [x0, x1] {
                        if x >= 0 && y >= 0 && (x as u32) < c.width() && (y as u32) < c.height() {
                            c.put_pixel(x as u32, y as u32, image::Rgba(col));
                        }
                    }
                }
            }
        };
        let (fw, fh) = (w as f32, h as f32);
        println!("{}: {} YuNet faces, {} people, {} heads added", p.display(), found.len(), people.len(), added.len());
        for d in &found {
            rect(&mut canvas, [d.bbox[0] * fw, d.bbox[1] * fh, d.bbox[2] * fw, d.bbox[3] * fh], [0, 255, 0, 255], 3);
        }
        for person in &people {
            rect(&mut canvas, person.bbox, [60, 120, 255, 255], 2);
            let s: Vec<String> = person.keypoints.iter().take(5).map(|k| format!("{:.2}", k.score)).collect();
            let head = crate::head::head_of(person, &det.params.head);
            let kp = |i: usize| person.keypoints[i];
            // left minus right, in the picture: positive for a body facing the camera
            let side = |l: usize, r: usize| (kp(l).x - kp(r).x) / person.bbox[2].max(1.0);
            println!(
                "  person {:.2} at {:?}: nose/eyes/ears {} | shoulders {:.2}/{:.2} lr {:+.2}, ears lr {:+.2}, eyes lr {:+.2} → {:?}",
                person.score,
                person.bbox.map(|v| v.round()),
                s.join(" "),
                kp(5).score,
                kp(6).score,
                side(5, 6),
                side(3, 4),
                side(1, 2),
                head.map(|h| (h.view, h.bbox.map(|v| v.round())))
            );
            for k in person.keypoints.iter().take(5) {
                if k.score >= det.params.head.keypoint {
                    rect(&mut canvas, [k.x - 3.0, k.y - 3.0, 6.0, 6.0], [255, 230, 0, 255], 3);
                }
            }
            if let Some(hd) = head {
                let col = if !hd.shows_face() { [150, 150, 150, 255] } else { [255, 255, 255, 255] };
                rect(&mut canvas, hd.bbox, col, 2);
                // calibration: the head against the face detector's face of the same person
                let yunet: Vec<[f32; 4]> = found.iter().map(|d| [d.bbox[0] * fw, d.bbox[1] * fh, d.bbox[2] * fw, d.bbox[3] * fh]).collect();
                let inside = |y: &&[f32; 4]| {
                    let (cx, cy) = (y[0] + y[2] / 2.0, y[1] + y[3] / 2.0);
                    cx >= hd.bbox[0] && cx <= hd.bbox[0] + hd.bbox[2] && cy >= hd.bbox[1] && cy <= hd.bbox[1] + hd.bbox[3]
                };
                if let Some(y) = yunet.iter().find(inside) {
                    use crate::head::*;
                    let pairs = [
                        ("ear-ear", LEFT_EAR, RIGHT_EAR, 0.9),
                        ("eye-eye", LEFT_EYE, RIGHT_EYE, 0.43),
                        ("nose-lear", NOSE, LEFT_EAR, 0.76),
                        ("nose-rear", NOSE, RIGHT_EAR, 0.76),
                        ("leye-lear", LEFT_EYE, LEFT_EAR, 0.55),
                        ("reye-rear", RIGHT_EYE, RIGHT_EAR, 0.55),
                        ("nose-leye", NOSE, LEFT_EYE, 0.38),
                        ("nose-reye", NOSE, RIGHT_EYE, 0.38),
                        ("shoulders", LEFT_SHOULDER, RIGHT_SHOULDER, 2.6),
                    ];
                    // each pair's width estimate over the face detector's face width ×1 (a face box is about a head wide)
                    let est: Vec<String> = pairs
                        .iter()
                        .map(|(n, a, b, s)| {
                            let (ka, kb) = (kp(*a), kp(*b));
                            let d = ((ka.x - kb.x).powi(2) + (ka.y - kb.y).powi(2)).sqrt() / s;
                            format!("{n} {:.2}@{:.2}", d / (y[3] / 1.3), ka.score.min(kb.score))
                        })
                        .collect();
                    println!("    CAL head h {:.0} / face h {:.0} = {:.2} ({:?}) [{}]", hd.bbox[3], y[3], hd.bbox[3] / y[3], hd.view, est.join(", "));
                }
            }
        }
        for d in &added {
            rect(&mut canvas, [d.bbox[0] * fw, d.bbox[1] * fh, d.bbox[2] * fw, d.bbox[3] * fh], [255, 0, 0, 255], 4);
        }
        canvas.save(out.join(p.file_name().unwrap())).unwrap();
    }
}

/// The GPU passes against the CPU's on the oracle's inputs.
#[test]
#[ignore = "needs the profile models, the oracle's inputs and a GPU"]
fn profile_gpu_matches_the_cpu() {
    let (Some(d), Some(o)) = (models(), oracle()) else { return };
    let cpu = person();
    let mut gpu = person();
    if !gpu.use_gpu() {
        eprintln!("no GPU: skipped");
        return;
    }
    let x = Feat::new(f32s(&o.join("dfine_input.bin")), 1, 3, 640, 640).unwrap();
    let (lc, bc) = cpu.raw(&x).unwrap();
    for round in 0..3 {
        let t = std::time::Instant::now();
        let (lg, bg) = gpu.raw(&x).unwrap();
        eprintln!("D-FINE on the GPU, round {round}: {:?}", t.elapsed());
        assert!(gpu.on_gpu());
        let eb = bc.iter().flatten().zip(bg.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        let el = lc.iter().zip(&lg).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        eprintln!("boxes within {eb:.2e}, logits within {el:.2e}");
        assert!(eb < 1e-3 && el < 1e-2, "{eb} {el}");
    }
    let cpu = RtmPose::load(&d.join(catalog::POSE.name)).unwrap();
    let mut gpu = RtmPose::load(&d.join(catalog::POSE.name)).unwrap();
    assert!(gpu.use_gpu());
    let x = Feat::new(f32s(&o.join("pose_input.bin")), 2, 3, INPUT_H, INPUT_W).unwrap();
    let (cx, cy) = cpu.raw(&x).unwrap();
    for round in 0..3 {
        let t = std::time::Instant::now();
        let (gx, gy) = gpu.raw(&x).unwrap();
        eprintln!("RTMPose on the GPU, 2 crops, round {round}: {:?}", t.elapsed());
        assert!(gpu.on_gpu());
        let e = cx.iter().chain(&cy).zip(gx.iter().chain(&gy)).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        eprintln!("SimCC within {e:.2e}");
        assert!(e < 1e-3, "{e}");
    }
}
