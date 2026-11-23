//! What works in every build: the package reader, the value and size checks, the colour helpers
//! and (without the feature) the stub's answer. The tests with EffectCraft are in `ec/tests.rs`.

use filmcraft_frame::VideoFrame;
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick};
use serde_json::{Value, json};

use crate::limits::{MAX_FRAMES, MAX_VALUE_DEPTH, MAX_VALUES};
use crate::{Error, color, package, params};

fn zip(files: &[(&str, &[u8])]) -> Vec<u8> {
    let files: Vec<(String, Vec<u8>)> = files.iter().map(|(n, b)| (n.to_string(), b.to_vec())).collect();
    filmcraft_templates::zip::write(&files).unwrap()
}

fn manifest(extra: Value) -> Vec<u8> {
    let mut m = json!({"format": "effectcraft-template", "version": 1, "name": "Card", "comp": 3});
    if let (Some(m), Some(e)) = (m.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            m.insert(k.clone(), v.clone());
        }
    }
    serde_json::to_vec(&m).unwrap()
}

fn template_err(r: crate::Result<package::Package>) -> String {
    match r {
        Err(Error::Template(m) | Error::TooLarge(m)) => m,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_package_is_read_with_its_media_and_poster() {
    let png = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR\0\0\x01\x40\0\0\0\xb4\x08\x06\0\0\0";
    let bytes = zip(&[
        ("manifest.json", &manifest(json!({}))),
        ("project.ecproj", b"{}"),
        ("media/7/logo.png", b"not really a png"),
        ("media/x/odd.png", b"?"),
        ("poster.png", png),
        ("notes.txt", b"hello"),
    ]);
    let p = package::read(&bytes).unwrap();
    assert_eq!((p.comp, p.name.as_str(), p.project.as_str()), (3, "Card", "{}"));
    assert_eq!(p.media.keys().collect::<Vec<_>>(), ["media/7/logo.png"]);
    assert_eq!(p.ignored, ["media/x/odd.png", "notes.txt"]);
    assert_eq!(p.poster.as_deref().and_then(package::png_size), Some((320, 180)));
    assert_eq!(package::media_entry("media/7/logo.png"), Some((7, "logo.png")));
    for bad in ["media/7/", "media//a.png", "media/7/a/b.png", "media/7/..", "other/7/a.png", "media/99999999999999999999/a.png"] {
        assert_eq!(package::media_entry(bad), None, "{bad}");
    }
    let dir = std::path::Path::new("/tmp/x");
    assert_eq!(package::media_path(dir, "media/7/logo.png"), Some(dir.join("7").join("logo.png")));
}

#[test]
fn hostile_packages_are_refused_with_a_reason() {
    let good_manifest = manifest(json!({}));
    let cases: Vec<(Vec<u8>, &str)> = vec![
        (b"not a zip".to_vec(), "ZIP"),
        (Vec::new(), "ZIP"),
        (zip(&[("project.ecproj", b"{}")]), "no manifest.json"),
        (zip(&[("manifest.json", b"{nope"), ("project.ecproj", b"{}")]), "not valid JSON"),
        (zip(&[("manifest.json", b"[1, 2]"), ("project.ecproj", b"{}")]), "not a JSON object"),
        (zip(&[("manifest.json", &manifest(json!({"format": "lottie"}))), ("project.ecproj", b"{}")]), "`format`"),
        (zip(&[("manifest.json", &manifest(json!({"version": 2}))), ("project.ecproj", b"{}")]), "newer EffectCraft"),
        (zip(&[("manifest.json", &manifest(json!({"version": "one"}))), ("project.ecproj", b"{}")]), "whole number"),
        (zip(&[("manifest.json", &manifest(json!({"comp": null}))), ("project.ecproj", b"{}")]), "`comp`"),
        (zip(&[("manifest.json", &good_manifest)]), "no project.ecproj"),
        (zip(&[("manifest.json", &good_manifest), ("project.ecproj", b"\xff\xfe")]), "UTF-8"),
        (zip(&[("manifest.json", &vec![b' '; crate::limits::MAX_MANIFEST_BYTES + 1]), ("project.ecproj", b"{}")]), "manifest.json is"),
    ];
    for (bytes, says) in cases {
        let m = template_err(package::read(&bytes));
        assert!(m.contains(says), "{m:?} should say {says:?}");
    }
    // nesting past serde_json's limit is a parse error, not a stack overflow
    let deep = format!("{}{}", "[".repeat(100_000), "]".repeat(100_000));
    let m = template_err(package::read(&zip(&[("manifest.json", deep.as_bytes()), ("project.ecproj", b"{}")])));
    assert!(m.contains("not valid JSON"), "{m}");
}

#[test]
fn truncated_and_bit_flipped_packages_never_panic() {
    let bytes = zip(&[("manifest.json", &manifest(json!({}))), ("project.ecproj", b"{\"schema\": 1}"), ("media/1/a.png", &[7u8; 300])]);
    for n in 0..bytes.len() {
        let r = std::panic::catch_unwind(|| package::read(&bytes[..n]));
        assert!(r.is_ok_and(|r| r.is_err()), "a prefix of {n} bytes");
    }
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    for _ in 0..2000 {
        let mut b = bytes.clone();
        for _ in 0..3 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let i = (seed % b.len() as u64) as usize;
            b[i] ^= 1 << (seed >> 61);
        }
        assert!(std::panic::catch_unwind(|| package::read(&b)).is_ok());
    }
}

#[test]
fn values_are_an_object_of_bounded_shape() {
    assert_eq!(params::values(None).unwrap().len(), 0);
    assert_eq!(params::values(Some(&Value::Null)).unwrap().len(), 0);
    let ok = params::values(Some(&json!({"Name": "Ada", "Bar Color": [0.2, 0.4, 0.8], "Logo": {"path": "/a.png"}}))).unwrap();
    assert_eq!(ok.len(), 3);
    for (v, says) in [
        (json!([1, 2]), "an object"),
        (json!("Ada"), "an object"),
        (json!({"": 1}), "1 to 256 bytes"),
        (json!({"x".repeat(300): 1}), "1 to 256 bytes"),
        (json!({"Name": "a".repeat(crate::limits::MAX_STRING_CHARS + 1)}), "longer than"),
    ] {
        let e = params::values(Some(&v)).unwrap_err().to_string();
        assert!(e.contains(says), "{e:?} should say {says:?}");
    }
    let many: serde_json::Map<String, Value> = (0..=MAX_VALUES).map(|i| (format!("c{i}"), json!(i))).collect();
    assert!(matches!(params::values(Some(&Value::Object(many))), Err(Error::TooLarge(_))));
    let mut deep = json!(1);
    for _ in 0..=MAX_VALUE_DEPTH + 1 {
        deep = json!([deep]);
    }
    let e = params::values(Some(&json!({"Name": deep}))).unwrap_err().to_string();
    assert!(e.contains("nests deeper"), "{e}");
    let wide = json!({"Name": vec![json!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]); 20_000]});
    assert!(matches!(params::values(Some(&wide)), Err(Error::TooLarge(_))));
}

#[test]
fn the_output_size_keeps_the_shape_and_its_caps() {
    assert_eq!(params::output_size((1920, 1080), None).unwrap(), ((1920, 1080), 1.0));
    assert_eq!(params::output_size((1920, 1080), Some((960, 540))).unwrap(), ((960, 540), 0.5));
    assert_eq!(params::output_size((1920, 1080), Some((3840, 2160))).unwrap(), ((3840, 2160), 2.0));
    let e = params::output_size((1920, 1080), Some((1000, 1000))).unwrap_err().to_string();
    assert!(e.contains("1000x563 does"), "{e}");
    for (comp, want) in [((30_000, 30_000), None), ((1920, 1080), Some((8640, 4860))), ((100, 100), Some((9000, 9000)))] {
        assert!(matches!(params::output_size(comp, want), Err(Error::TooLarge(_))), "{comp:?} {want:?}");
    }
    assert!(params::output_size((0, 10), None).is_err());
    assert!(params::output_size((10, 10), Some((0, 0))).is_err());
}

#[test]
fn the_frame_count_is_exact_and_capped() {
    let s = |x: f64| Tick::from_seconds_f64(x);
    assert_eq!(params::frame_count(Tick::ZERO, s(1.0), FrameRate::FPS_30).unwrap(), 30);
    assert_eq!(params::frame_count(Tick::ZERO, s(1.0), FrameRate::FPS_29_97).unwrap(), 30);
    assert_eq!(params::frame_count(Tick::ZERO, Tick(1), FrameRate::FPS_30).unwrap(), 1);
    assert_eq!(params::frame_count(s(0.5), s(1.0), FrameRate::FPS_24).unwrap(), 12);
    // exactly at a frame boundary: that frame is not part of it
    let one = FrameRate::FPS_30.tick_of(1);
    assert_eq!(params::frame_count(Tick::ZERO, one, FrameRate::FPS_30).unwrap(), 1);
    assert_eq!(params::frame_count(Tick::ZERO, Tick(one.0 + 1), FrameRate::FPS_30).unwrap(), 2);
    assert!(params::frame_count(s(1.0), s(1.0), FrameRate::FPS_30).is_err());
    let long = Tick(TICKS_PER_SECOND.saturating_mul((MAX_FRAMES / 60 + 1) as i64));
    assert!(matches!(params::frame_count(Tick::ZERO, long, FrameRate::FPS_60), Err(Error::TooLarge(_))));
    assert!(matches!(params::frame_count(Tick::ZERO, Tick(i64::MAX), FrameRate::FPS_60), Err(Error::TooLarge(_))));
    // a broken rate is the default one, never a division by zero
    assert!(params::frame_count(Tick::ZERO, s(1.0), FrameRate { num: 0, den: 0 }).is_ok());
}

#[test]
fn the_range_lies_inside_the_comp() {
    let d = Tick::from_seconds_f64(2.0);
    assert_eq!(params::range(None, None, d).unwrap(), (Tick::ZERO, d));
    assert_eq!(params::range(Some(Tick(5)), None, d).unwrap(), (Tick(5), d));
    for (s, e) in [(Some(Tick(-1)), None), (Some(d), None), (None, Some(Tick::ZERO)), (None, Some(Tick(d.0 + 1))), (Some(Tick(9)), Some(Tick(9)))] {
        assert!(params::range(s, e, d).is_err(), "{s:?} {e:?}");
    }
    assert!(params::range(None, None, Tick::ZERO).is_err());
}

#[test]
fn colour_goes_across_both_ways() {
    // linear → encoded → linear is the identity (to float precision)
    for p in [[0.5f32, 0.25, 0.125, 1.0], [0.1, 0.2, 0.3, 0.5], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]] {
        let back = color::encoded_to_linear(color::linear_to_encoded(p));
        for c in 0..4 {
            assert!((back[c] - p[c]).abs() < 1e-5, "{p:?} → {back:?}");
        }
    }
    // linear mid grey is encoded higher (the sRGB curve), alpha stays
    let e = color::linear_to_encoded([0.214, 0.214, 0.214, 1.0]);
    assert!((e[0] - 0.5).abs() < 0.01 && e[3] == 1.0, "{e:?}");
    // straight 8-bit levels go across without a curve
    for px in [[0u8, 0, 0, 255], [51, 102, 204, 255], [200, 100, 50, 128], [255, 255, 255, 1]] {
        assert_eq!(color::encoded_to_rgba8(color::rgba8_to_encoded(px)), px);
    }
    // hostile numbers: no panic, sanitised
    for p in [[f32::NAN, 0.5, 0.5, 1.0], [f32::INFINITY, -1.0, 2.0, 0.5], [0.5, 0.5, 0.5, f32::NAN], [0.5, 0.5, 0.5, -1.0], [1e30, 1e30, 1e30, 1e30]] {
        let o = color::encoded_to_rgba8(p);
        let l = color::encoded_to_linear(p);
        assert!(l.iter().all(|x| x.is_finite()), "{p:?} → {l:?}");
        let _ = o;
    }
    assert_eq!(color::encoded_to_rgba8([0.3, 0.3, 0.3, 0.0]), [0, 0, 0, 0]);
    assert_eq!(color::frame_to_rgba8(&[[0.5, 0.0, 0.0, 0.5], [0.0; 4]]), vec![255, 0, 0, 128, 0, 0, 0, 0]);
}

#[test]
fn host_frames_become_encoded_footage() {
    // straight sRGB RGBA8: premultiplied, no curve
    let f = VideoFrame::rgba8(2, 1, vec![255, 0, 0, 255, 0, 0, 255, 128]);
    let (w, h, px) = color::host_to_encoded(&f, false, 100).unwrap();
    assert_eq!((w, h), (2, 1));
    assert_eq!(px[0], [1.0, 0.0, 0.0, 1.0]);
    assert!((px[1][2] - 128.0 / 255.0).abs() < 1e-6 && (px[1][3] - 128.0 / 255.0).abs() < 1e-6);
    // ignoring alpha makes it opaque with its colour
    let (_, _, opaque) = color::host_to_encoded(&f, true, 100).unwrap();
    assert!((opaque[1][2] - 1.0).abs() < 1e-6 && opaque[1][3] == 1.0);
    // linear float frames are encoded with the sRGB curve
    let lin = VideoFrame::rgba_f32(1, 1, vec![0.214, 0.214, 0.214, 1.0]);
    let (_, _, e) = color::host_to_encoded(&lin, false, 100).unwrap();
    assert!((e[0][0] - 0.5).abs() < 0.01, "{e:?}");
    // too large, empty or short frames give nothing
    assert!(color::host_to_encoded(&f, false, 1).is_none());
    assert!(color::host_to_encoded(&VideoFrame::rgba8(0, 0, vec![]), false, 100).is_none());
    let short = VideoFrame { width: 4, ..VideoFrame::rgba8(2, 1, vec![0; 8]) };
    assert!(color::host_to_encoded(&short, false, 100).is_none());
    // and back to AutoCut's linear frames
    let back = color::encoded_to_host(1, 1, &e).unwrap();
    assert_eq!((back.width, back.height), (1, 1));
    assert!(color::encoded_to_host(2, 2, &e).is_none());
    assert!(color::encoded_to_host(0, 0, &[]).is_none());
}

#[cfg(not(all(feature = "effectcraft", not(target_arch = "wasm32"))))]
#[test]
fn without_effectcraft_every_call_says_so() {
    assert!(!crate::available());
    let bytes = zip(&[("manifest.json", &manifest(json!({}))), ("project.ecproj", b"{}")]);
    let e = crate::template_info(&bytes).unwrap_err();
    assert_eq!(e, Error::NotBuilt);
    assert!(e.to_string().contains("`effectcraft` feature"), "{e}");
    let req = crate::RenderRequest::default();
    assert!(matches!(crate::prepare(&bytes, &req, std::path::Path::new("/nonexistent"), &crate::NoFrames), Err(Error::NotBuilt)));
    assert_eq!(crate::about()["available"], json!(false));
}
