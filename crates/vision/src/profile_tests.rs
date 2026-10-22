//! Live tests of the profile pass's networks: against ONNX Runtime's outputs
//! (`tools/profile_oracle.py`), and on real pictures. Ignored: they need the models
//! (`FILMCRAFT_VISION_MODELS`, holding `privacy-detectors/`) and, for the oracle tests, the
//! oracle's directory (`FILMCRAFT_PROFILE_ORACLE`).

use std::path::PathBuf;

use crate::catalog;
use crate::dfine::{self, Dfine};
use crate::layers::{Backend, Feat};
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

/// The picture the speed and parity tests look at: `FILMCRAFT_PROFILE_PICTURE`, else the
/// oracle's `people.png`, scaled to 1920 × 1080.
fn frame_1080() -> (Vec<u8>, usize, usize) {
    let path = std::env::var_os("FILMCRAFT_PROFILE_PICTURE").map(PathBuf::from).or_else(|| oracle().map(|o| o.join("../people.png"))).unwrap();
    let img = image::open(&path).unwrap().resize_exact(1920, 1080, image::imageops::FilterType::Triangle).to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    (img.into_raw(), w, h)
}

fn best_of(rounds: usize, mut f: impl FnMut()) -> std::time::Duration {
    let mut best = std::time::Duration::MAX;
    for _ in 0..rounds {
        let t = std::time::Instant::now();
        f();
        best = best.min(t.elapsed());
    }
    best
}

/// Speed of the profile pass on a 1920 × 1080 picture ([`frame_1080`]), release build: the whole
/// pass (people and their poses) on the GPU and on the CPU, and its parts. Fastest of
/// `FILMCRAFT_PROFILE_ROUNDS` (5) rounds, and their median.
#[test]
#[ignore = "needs the profile models"]
fn profile_pass_speed() {
    let Some(_) = models() else { return };
    let (rgba, w, h) = frame_1080();
    let rounds: usize = std::env::var("FILMCRAFT_PROFILE_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(5).max(1);
    let dir = PathBuf::from(std::env::var_os("FILMCRAFT_VISION_MODELS").unwrap());
    eprintln!("load average: {}", load_average());
    for gpu in [true, false] {
        let det = crate::ProfileDetector::load_on(&dir, gpu).unwrap();
        let x = dfine::prepare(&rgba, w, h, 640).unwrap();
        let d = best_of(rounds, || {
            let _ = det.person.people(&x, 0.4).unwrap();
        });
        eprintln!("{}: D-FINE at {}×{}: {d:?}", det.device_name(), x.w, x.h);
        let p = best_of(rounds, || {
            let _ = dfine::prepare(&rgba, w, h, 640).unwrap();
        });
        eprintln!("    the picture prepared on the CPU: {p:?}");
        let mut times = Vec::new();
        let mut n = 0;
        for _ in 0..rounds {
            let t = std::time::Instant::now();
            let people = det.people(&rgba, w, h).unwrap();
            times.push(t.elapsed());
            n = people.len();
        }
        times.sort();
        let (best, median) = (times[0], times[times.len() / 2]);
        eprintln!(
            "{}: people and poses ({n} people): best {best:?} = {:.1} frames a second, median {median:?} = {:.1}",
            det.device_name(),
            1.0 / best.as_secs_f64(),
            1.0 / median.as_secs_f64()
        );
        if gpu {
            let c = best_of(rounds, || {
                let _ = det.people_cpu(&rgba, w, h).unwrap();
            });
            eprintln!("gpu: the networks on the GPU, the picture and crops prepared on the CPU (the previous path's shape): {c:?}");
        }
    }
    eprintln!("load average: {}", load_average());
}

fn load_average() -> String {
    std::process::Command::new("sysctl").args(["-n", "vm.loadavg"]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
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

/// What the face detector and the profile pass find on every PNG of
/// `FILMCRAFT_PROFILE_PICTURES` (frames of a clip, named by their media time in seconds), as JSON
/// in `FILMCRAFT_PROFILE_DUMP`: per frame its size, YuNet's faces (as the redaction looks: at 640
/// and closer, suppressed together) and every person with their pose and head box. The head-box
/// calibration and the redaction's flicker analysis read it.
#[test]
#[ignore = "needs the models, pictures and an output path (FILMCRAFT_VISION_MODELS, FILMCRAFT_PROFILE_PICTURES, FILMCRAFT_PROFILE_DUMP)"]
fn profile_dump() {
    let (Some(_), Some(dir), Some(out)) =
        (models(), std::env::var_os("FILMCRAFT_PROFILE_PICTURES").map(PathBuf::from), std::env::var_os("FILMCRAFT_PROFILE_DUMP").map(PathBuf::from))
    else {
        return;
    };
    let models_dir = PathBuf::from(std::env::var_os("FILMCRAFT_VISION_MODELS").unwrap());
    let det = crate::ProfileDetector::load_on(&models_dir, true).unwrap();
    let faces = crate::Detectors::load(&models_dir, true, false).unwrap();
    let f = faces.face.as_ref().unwrap();
    let mut paths: Vec<PathBuf> =
        std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "png")).collect();
    paths.sort();
    let mut frames = Vec::new();
    for p in paths {
        let img = image::open(&p).unwrap().to_rgba8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgba = img.into_raw();
        let mut found = f.detect(&rgba, w, h, 640, 0.6).unwrap();
        if let Some(close) = Some(1280.min(w.max(h))).filter(|c| *c > 640) {
            found.extend(f.detect(&rgba, w, h, close, 0.6).unwrap());
        }
        let found = crate::post::nms(found, 0.3, 750);
        let people = det.people(&rgba, w, h).unwrap();
        let heads: Vec<serde_json::Value> = people
            .iter()
            .map(|person| {
                let hd = crate::head::head_of(person, &det.params.head);
                serde_json::json!({
                    "bbox": person.bbox,
                    "score": person.score,
                    "keypoints": person.keypoints.iter().map(|k| [k.x, k.y, k.score]).collect::<Vec<_>>(),
                    "head": hd.map(|h| serde_json::json!({"bbox": h.bbox, "view": format!("{:?}", h.view), "faceScore": h.face_score})),
                })
            })
            .collect();
        let added = crate::head::candidates(&people, &found, &det.params.head, w as f32, h as f32);
        frames.push(serde_json::json!({
            "name": p.file_name().map(|n| n.to_string_lossy().to_string()),
            "w": w,
            "h": h,
            "faces": found.iter().map(|d| serde_json::json!({"bbox": d.bbox, "score": d.score})).collect::<Vec<_>>(),
            "people": heads,
            "added": added.iter().map(|d| serde_json::json!({"bbox": d.bbox, "score": d.score})).collect::<Vec<_>>(),
        }));
    }
    std::fs::write(&out, serde_json::to_vec(&frames).unwrap()).unwrap();
    eprintln!("{} frames written to {}", frames.len(), out.display());
}

/// The GPU passes against the CPU's: the person detector (its encoder, its decoder on the same
/// inputs, the whole network) and the pose network on the oracle's inputs (else on
/// [`frame_1080`]), and the whole pass (people and poses) on every PNG of
/// `FILMCRAFT_PROFILE_PICTURES` (else [`frame_1080`]).
#[test]
#[ignore = "needs the profile models and a GPU (and the oracle's inputs or pictures)"]
fn profile_gpu_matches_the_cpu() {
    let Some(d) = models() else { return };
    let cpu = person();
    let mut gpu = person();
    if !gpu.use_gpu() {
        eprintln!("no GPU: skipped");
        return;
    }
    let x = match oracle() {
        Some(o) if o.join("dfine_input.bin").exists() => Feat::new(f32s(&o.join("dfine_input.bin")), 1, 3, 640, 640).unwrap(),
        _ => {
            let (rgba, w, h) = frame_1080();
            dfine::prepare(&rgba, w, h, 640).unwrap()
        }
    };
    let max_err = |a: &[f32], b: &[f32]| {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    };
    // the encoder alone
    let enc_cpu = cpu.encode_cpu(&x).unwrap();
    let enc_gpu = crate::gpunn::GpuUse::run(&gpu.gpu, |r| {
        let outs = gpu.encode_gpu(r, r.input(&x)?)?;
        r.read(&outs.iter().collect::<Vec<_>>())
    })
    .unwrap();
    for (a, b) in enc_cpu.iter().zip(&enc_gpu) {
        let scale = a.v.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("encoder level {}×{}: within {:.2e} (largest {scale:.2})", a.w, a.h, max_err(&a.v, &b.v));
        assert!(max_err(&a.v, &b.v) < 1e-3 * scale.max(1.0));
    }
    // the decoder alone, on the CPU encoder's outputs
    let (lc, bc) = cpu.decode(&enc_cpu).unwrap();
    let (lg, bg) = crate::gpunn::GpuUse::run(&gpu.gpu, |r| {
        let ins: Vec<_> = enc_cpu.iter().map(|f| r.input(f)).collect::<Result<_, _>>()?;
        let (l, b) = gpu.decode_gpu(r, &ins)?;
        let mut v = r.read_mats(&[&l, &b])?;
        let b = v.pop().unwrap();
        Ok((v.pop().unwrap(), b))
    })
    .unwrap();
    let eb = max_err(&bc.iter().flatten().copied().collect::<Vec<_>>(), &bg);
    let el = max_err(&lc, &lg);
    eprintln!("decoder alone: boxes within {eb:.2e}, logits within {el:.2e}");
    assert!(eb < 1e-3 && el < 1e-2, "{eb} {el}");
    // the whole network
    let (lc, bc) = cpu.raw_cpu(&x).unwrap();
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
    let cpu_pose = RtmPose::load(&d.join(catalog::POSE.name)).unwrap();
    let mut gpu_pose = RtmPose::load(&d.join(catalog::POSE.name)).unwrap();
    assert!(gpu_pose.use_gpu());
    let x = match oracle() {
        Some(o) if o.join("pose_input.bin").exists() => Feat::new(f32s(&o.join("pose_input.bin")), 2, 3, INPUT_H, INPUT_W).unwrap(),
        _ => {
            // crops of the people the CPU finds
            let (rgba, w, h) = frame_1080();
            let found = cpu.people(&dfine::prepare(&rgba, w, h, 640).unwrap(), 0.4).unwrap();
            let mut v = Vec::new();
            for f in found.iter().take(4) {
                let [cx, cy, bw, bh] = f.cxcywh;
                let b = [(cx - bw / 2.0) * w as f32, (cy - bh / 2.0) * h as f32, bw * w as f32, bh * h as f32];
                v.extend(crate::rtmpose::crop_for_test(&rgba, w, h, &b).unwrap().0);
            }
            let n = v.len() / (3 * INPUT_H * INPUT_W);
            assert!(n > 0, "nobody in the picture");
            Feat::new(v, n, 3, INPUT_H, INPUT_W).unwrap()
        }
    };
    let (cx, cy) = cpu_pose.raw_cpu(&x).unwrap();
    for round in 0..3 {
        let t = std::time::Instant::now();
        let (gx, gy) = gpu_pose.raw(&x).unwrap();
        eprintln!("RTMPose on the GPU, {} crops, round {round}: {:?}", x.n, t.elapsed());
        assert!(gpu_pose.on_gpu());
        let e = cx.iter().chain(&cy).zip(gx.iter().chain(&gy)).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        eprintln!("SimCC within {e:.2e}");
        assert!(e < 1e-3, "{e}");
    }
    // the whole pass: the same people and keypoints
    let models_dir = PathBuf::from(std::env::var_os("FILMCRAFT_VISION_MODELS").unwrap());
    let on_gpu = crate::ProfileDetector::load_on(&models_dir, true).unwrap();
    let on_cpu = crate::ProfileDetector::load_on(&models_dir, false).unwrap();
    let pictures: Vec<(String, Vec<u8>, usize, usize)> = match std::env::var_os("FILMCRAFT_PROFILE_PICTURES").map(PathBuf::from) {
        Some(dir) => {
            let mut paths: Vec<PathBuf> =
                std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "png")).collect();
            paths.sort();
            paths
                .into_iter()
                .map(|p| {
                    let img = image::open(&p).unwrap().to_rgba8();
                    let (w, h) = (img.width() as usize, img.height() as usize);
                    (p.display().to_string(), img.into_raw(), w, h)
                })
                .collect()
        }
        None => {
            let (rgba, w, h) = frame_1080();
            vec![("picture".to_string(), rgba, w, h)]
        }
    };
    let (mut people, mut kps, mut close, mut worst_box) = (0usize, 0usize, 0usize, 0f32);
    for (name, rgba, w, h) in &pictures {
        let a = on_gpu.people(rgba, *w, *h).unwrap();
        let b = on_cpu.people(rgba, *w, *h).unwrap();
        assert!(on_gpu.device_name() == "gpu");
        assert_eq!(a.len(), b.len(), "{name}: people");
        for (p, q) in a.iter().zip(&b) {
            let e = p.bbox.iter().zip(&q.bbox).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
            worst_box = worst_box.max(e);
            assert!(e < 0.5, "{name}: a person's box {:?} vs {:?}", p.bbox, q.bbox);
            assert!((p.score - q.score).abs() < 1e-3);
            for (k, l) in p.keypoints.iter().zip(&q.keypoints) {
                kps += 1;
                // a bin's width in the picture is k / 2 (≤ a pixel or two); a tie between two
                // bins may go either way
                let near = (k.x - l.x).abs() < 1.0 && (k.y - l.y).abs() < 1.0 && (k.score - l.score).abs() < 1e-2;
                close += usize::from(near);
            }
            people += 1;
        }
    }
    eprintln!(
        "whole pass on {} pictures: {people} people the same (boxes within {worst_box:.2e} px), {close} of {kps} keypoints within a pixel",
        pictures.len()
    );
    assert!(close * 100 >= kps * 99, "{close} of {kps}");
}
