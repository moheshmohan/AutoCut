//! `effectcraft.*` in both builds: the parameters and the answer without EffectCraft always; with
//! the feature, templates made in code ([`filmcraft_effectcraft::testing`]) read through the
//! command, and hostile files refused without a panic.

use serde_json::json;

use super::*;

#[test]
fn template_info_needs_an_absolute_template() {
    let mut s = Session::default();
    for p in [json!({}), json!({"template": "relative.ectemplate"}), json!({"template": 7}), json!({"template": ""})] {
        let e = s.execute(INFO, p.clone()).unwrap_err().to_string();
        assert!(e.contains("`template`"), "{p}: {e}");
    }
}

#[test]
fn without_effectcraft_template_info_says_so() {
    if filmcraft_effectcraft::available() {
        return;
    }
    let mut s = Session::default();
    let e = s.execute(INFO, json!({"template": "/nowhere/card.ectemplate"})).unwrap_err().to_string();
    assert!(e.contains("`effectcraft` feature"), "{e}");
}

#[test]
fn without_effectcraft_render_template_says_so_and_keeps_nothing() {
    let mut s = Session::default();
    let e = s.execute(RENDER, json!({"values": {}})).unwrap_err().to_string();
    assert!(e.contains("`template`"), "{e}");
    if filmcraft_effectcraft::available() {
        return;
    }
    let out = crate::media_test_util::tmp_dir("effectcraft-off").join("r001");
    let e = s.execute(RENDER, json!({"template": "/nowhere/card.ectemplate", "out": out.to_string_lossy(), "wait": true})).unwrap_err().to_string();
    assert!(e.contains("`effectcraft` feature"), "{e}");
    assert!(!out.exists());
}

#[cfg(feature = "effectcraft")]
mod linked {
    use std::path::PathBuf;

    use filmcraft_effectcraft::testing::{Logo, LowerThird};
    use serde_json::json;

    use super::super::{INFO, RENDER};
    use crate::Session;

    fn dir(name: &str) -> PathBuf {
        crate::media_test_util::tmp_dir(&format!("effectcraft-{name}"))
    }

    #[test]
    fn a_template_is_described() {
        let d = dir("info");
        let path = d.join("lower third.ectemplate");
        let logo =
            Logo { png: filmcraft_export::encode_png([255u8, 0, 0, 255].repeat(16), 4, 4).unwrap(), width: 4, height: 4, embedded: true, path: String::new() };
        std::fs::write(&path, LowerThird { logo: Some(logo), ..Default::default() }.build()).unwrap();
        let mut s = Session::default();
        let v = s.execute(INFO, json!({"template": path.to_string_lossy()})).unwrap();
        assert_eq!((v["name"].as_str(), v["comp"]["width"].as_u64(), v["comp"]["frames"].as_u64()), (Some("Lower Third"), Some(320), Some(30)), "{v:#}");
        let names: Vec<&str> = v["controls"].as_array().unwrap().iter().filter_map(|c| c["name"].as_str()).collect();
        assert_eq!(names, ["Name", "Bar Color", "Bar Opacity", "Logo"]);
        assert_eq!(v["media"].as_array().map(Vec::len), Some(1));
        assert_eq!(v["path"].as_str(), Some(path.to_string_lossy().as_ref()));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn hostile_files_are_refused_without_a_panic() {
        let d = dir("hostile");
        let mut s = Session::default();
        let bytes = LowerThird::default().build();
        let at = |name: &str, b: &[u8]| {
            let p = d.join(name);
            std::fs::write(&p, b).unwrap();
            json!({"template": p.to_string_lossy()})
        };
        for (name, b, says) in
            [("half.ectemplate", &bytes[..bytes.len() / 2], "ZIP"), ("empty.ectemplate", &[][..], "ZIP"), ("text.ectemplate", &b"just text"[..], "ZIP")]
        {
            let e = s.execute(INFO, at(name, b)).unwrap_err().to_string();
            assert!(e.contains(says), "{name}: {e}");
        }
        // too large: refused from its size, never read
        let big = d.join("big.ectemplate");
        std::fs::File::create(&big).unwrap().set_len(filmcraft_effectcraft::limits::MAX_TEMPLATE_BYTES + 1).unwrap();
        let e = s.execute(INFO, json!({"template": big.to_string_lossy()})).unwrap_err().to_string();
        assert!(e.contains("at most 256 MiB"), "{e}");
        // not there, and a folder
        let e = s.execute(INFO, json!({"template": d.join("missing.ectemplate").to_string_lossy()})).unwrap_err().to_string();
        assert!(e.contains("missing.ectemplate"), "{e}");
        assert!(s.execute(INFO, json!({"template": d.to_string_lossy()})).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A session with the demo project open and a template written into a folder of its own.
    fn bench(name: &str, lt: &LowerThird) -> (Session, PathBuf, PathBuf) {
        let d = dir(name);
        let t = d.join("lower third.ectemplate");
        std::fs::write(&t, lt.build()).unwrap();
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        (s, d, t)
    }

    #[test]
    fn a_lower_third_renders_to_png_frames_imports_and_lands_over_the_picture() {
        let (mut s, d, t) = bench("png", &LowerThird::default());
        let out = d.join("r001");
        let v = s
            .execute(
                RENDER,
                json!({"template": t.to_string_lossy(), "values": {"Name": "Grace Hopper", "Bar Color": "#ff0000"}, "range": {"end": 254016000000i64 / 2},
                       "out": out.to_string_lossy(), "format": "png", "place": {"seconds": 1.0}, "wait": true}),
            )
            .unwrap();
        assert_eq!((v["kind"].as_str(), v["frames"].as_u64()), (Some("pngSequence"), Some(15)), "{v:#}");
        assert_eq!(
            (v["probe"]["width"].as_u64(), v["probe"]["height"].as_u64(), v["probe"]["hasAlpha"].as_bool()),
            (Some(320), Some(180), Some(true)),
            "{v:#}"
        );
        assert_eq!(v["probe"]["frames"].as_u64(), Some(15));
        // on the GPU where there is one (under the lock), the CPU for the rest
        assert_eq!(v["backend"]["cpuFrames"].as_u64().unwrap() + v["backend"]["gpuFrames"].as_u64().unwrap(), 15, "{v:#}");
        assert_eq!(v["import"]["command"], "file.importImageSequence");
        assert!(out.join("frames").join("frame-00014.png").is_file() && !out.join("frames").join("frame-00015.png").exists());
        assert!(out.join("autocut-import.json").is_file());
        assert!(!out.join(".effectcraft-render").exists(), "the staging folder is gone");
        // imported, noted and placed above the picture at 1 s
        let item = v["item"].as_u64().unwrap();
        assert!(s.project.item(filmcraft_project::ItemId(item)).is_some_and(|it| it.metadata.get("Log Note").is_some_and(|n| n.contains("EffectCraft"))));
        assert_ne!(v["clip"], serde_json::Value::Null, "{v:#}");
        // (on the sequence's frame grid)
        let frame = s.sequence_rate().frame_duration().0;
        assert!(v["start"].as_i64().is_some_and(|t| (t - 254016000000).abs() <= frame), "{v:#}");
        assert_ne!(v["track"].as_str(), Some("V1"));
        // one undo step takes it all away
        assert!(s.undo().is_some());
        assert!(s.project.item(filmcraft_project::ItemId(item)).is_none());
        // the job says what it made
        let jobs = s.execute("jobs.list", json!({})).unwrap();
        let job = jobs.as_array().unwrap().iter().find(|j| j["id"] == v["job"]).unwrap();
        assert_eq!(job["motion"]["kind"], "pngSequence", "{job:#}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_template_renders_to_prores_4444_with_alpha() {
        let (mut s, d, t) = bench("prores", &LowerThird::default());
        let v = s
            .execute(RENDER, json!({"template": t.to_string_lossy(), "size": "160x90", "fps": 24, "range": {"end": 254016000000i64 / 4}, "gpu": false, "import": false, "wait": true}))
            .unwrap();
        assert_eq!(v["kind"], "prores4444", "{v:#}");
        let asset = PathBuf::from(v["asset"].as_str().unwrap());
        assert!(asset.extension().is_some_and(|e| e == "mov") && asset.is_file(), "{v:#}");
        // the default revision folder is beside the template
        assert!(asset.starts_with(d.join("lower third renders").join("r001")), "{}", asset.display());
        assert_eq!((v["probe"]["width"].as_u64(), v["probe"]["fps"].as_f64(), v["probe"]["hasAlpha"].as_bool()), (Some(160), Some(24.0), Some(true)), "{v:#}");
        assert_eq!(v["frames"].as_u64(), Some(6));
        assert_eq!(v["item"], serde_json::Value::Null, "not imported");
        // the movie's pixels: a bar pixel opaque in the bar's colour, a corner transparent
        let src = crate::media_pool::MediaPool::default().open_file(&asset.to_string_lossy(), s.services.as_ref()).unwrap();
        let f = src.video_frame(filmcraft_media::FrameRequest { time: filmcraft_time::Tick::ZERO, scale: 1.0 }).unwrap();
        let rgba = f.to_rgba8();
        let at = |x: usize, y: usize| &rgba[(y * 160 + x) * 4..(y * 160 + x) * 4 + 4];
        let bar = at(97, 75);
        assert!(bar[3] > 250 && bar[2] > 190 && bar[0] < 70, "{bar:?}");
        assert!(at(2, 2)[3] < 5, "{:?}", at(2, 2));
        // the next render goes into the next free folder
        let v2 = s.execute(RENDER, json!({"template": t.to_string_lossy(), "range": {"end": 1}, "gpu": false, "import": false, "wait": true})).unwrap();
        assert!(v2["output"].as_str().unwrap().ends_with("r002"), "{v2:#}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn hostile_render_requests_are_refused_before_anything_is_kept() {
        let (mut s, d, t) = bench("hostile", &LowerThird::default());
        let out = d.join("refused");
        let base = json!({"template": t.to_string_lossy(), "out": out.to_string_lossy(), "gpu": false, "wait": true});
        let with = |k: &str, v: serde_json::Value| {
            let mut p = base.clone();
            p[k] = v;
            p
        };
        for (p, says) in [
            (with("format", json!("gif")), "`format`"),
            (with("size", json!("1000x1000")), "shape"),
            (with("size", json!("big")), "`size`"),
            (with("fps", json!(0)), "frame rate"),
            (with("range", json!({"start": -5})), "range.start"),
            (with("range", json!({"end": "soon"})), "range.end"),
            (with("values", json!({"Title": "x"})), "no control `Title`"),
            (with("values", json!({"Bar Opacity": 1e300})), "finite"),
            (with("values", json!([1])), "an object"),
            (with("place", json!({"track": "A1"})), "video track"),
            (with("gpu", json!("yes")), "`gpu`"),
        ] {
            let e = s.execute(RENDER, p.clone()).unwrap_err().to_string();
            assert!(e.contains(says), "{p}: {e:?} should say {says:?}");
            assert!(!out.exists(), "{p}: the refused render left {}", out.display());
        }
        // an output folder with something in it is never written into
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(out.join("keep.txt"), "mine").unwrap();
        let e = s.execute(RENDER, base.clone()).unwrap_err().to_string();
        assert!(e.contains("already holds files"), "{e}");
        assert_eq!(std::fs::read_to_string(out.join("keep.txt")).unwrap(), "mine");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn carried_media_is_decoded_by_the_engine() {
        let logo = Logo {
            png: filmcraft_export::encode_png([0u8, 255, 0, 255].repeat(16 * 8), 16, 8).unwrap(),
            width: 16,
            height: 8,
            embedded: true,
            path: String::new(),
        };
        let lt = LowerThird { logo: Some(logo.clone()), ..Default::default() };
        let (mut s, d, t) = bench("media", &lt);
        let v = s
            .execute(RENDER, json!({"template": t.to_string_lossy(), "format": "png", "range": {"end": 1}, "gpu": false, "import": false, "wait": true}))
            .unwrap();
        let first = PathBuf::from(v["output"].as_str().unwrap()).join("frames").join("frame-00000.png");
        let img = crate::media_pool::MediaPool::default().open_file(&first.to_string_lossy(), s.services.as_ref()).unwrap();
        let rgba = img.video_frame(filmcraft_media::FrameRequest { time: filmcraft_time::Tick::ZERO, scale: 1.0 }).unwrap().to_rgba8();
        let [cx, cy] = filmcraft_effectcraft::testing::logo_centre(lt.size, &logo);
        let i = ((cy as usize) * 320 + cx as usize) * 4;
        let got = &rgba[i..i + 4];
        assert!(got.iter().zip([0u8, 255, 0, 255]).all(|(a, b)| a.abs_diff(b) <= 1), "the carried logo is drawn: {got:?}");
        // a media value naming a file that is not there is refused up front
        let e = s
            .execute(RENDER, json!({"template": t.to_string_lossy(), "values": {"Logo": d.join("nope.png").to_string_lossy()}, "gpu": false, "wait": true}))
            .unwrap_err()
            .to_string();
        assert!(e.contains("Logo"), "{e}");
        let _ = std::fs::remove_dir_all(&d);
    }
}
