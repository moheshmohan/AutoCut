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

#[cfg(feature = "effectcraft")]
mod linked {
    use std::path::PathBuf;

    use filmcraft_effectcraft::testing::{Logo, LowerThird};
    use serde_json::json;

    use super::super::INFO;
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
}
