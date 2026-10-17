//! Tests of [`crate::elements`]: the library lists every element, inserts each as one undo step
//! (sounds reproducibly, reusing their file), refuses hostile parameters without changing
//! anything, and the bundled sticker previews match the vectors and carry attribution.

use super::*;
use filmcraft_project::ClipId;
use serde_json::json;

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("filmcraft-elements-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d.to_string_lossy().into_owned()
}

fn demo() -> Session {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s
}

fn repo() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn clip_count(s: &Session) -> usize {
    s.active_sequence().unwrap().all_tracks().map(|t| t.items.len()).sum()
}

#[test]
fn lists_every_element_with_its_provenance() {
    let mut s = Session::default();
    let all = s.execute("elements.list", json!({})).unwrap();
    let all = all.as_array().unwrap();
    assert_eq!(all.len(), 12 + 10 + 6);
    let ids: std::collections::HashSet<&str> = all.iter().filter_map(|e| e["id"].as_str()).collect();
    assert_eq!(ids.len(), all.len(), "unique ids");
    for e in all {
        for k in ["id", "kind", "name", "category", "description", "author", "license", "source", "made"] {
            assert!(e[k].as_str().is_some_and(|v| !v.is_empty()), "{k} of {e}");
        }
        assert_eq!(e["license"], "MIT OR Apache-2.0");
        assert!(e["seconds"].as_f64().is_some_and(|x| x > 0.0 && x <= 10.0), "{e}");
        let id = e["id"].as_str().unwrap();
        assert!(id.starts_with(&format!("{}.", e["kind"].as_str().unwrap())), "{id}");
    }
    for (kind, n) in [("sound", 12), ("sticker", 10), ("text", 6), ("Stickers", 10), (" TEXT ", 6)] {
        assert_eq!(s.execute("elements.list", json!({"kind": kind})).unwrap().as_array().unwrap().len(), n, "{kind}");
    }
    let whooshes = s.execute("elements.list", json!({"query": "whoosh"})).unwrap();
    assert_eq!(whooshes.as_array().unwrap().len(), 2);
    let bell = s.execute("elements.list", json!({"kind": "sticker", "query": "bell"})).unwrap();
    assert_eq!(bell[0]["file"], "assets/elements/stickers/bell.svg");
    assert!(bell[0]["controls"].as_array().unwrap().iter().any(|c| c["id"] == "badge"));
    for bad in [json!({"kind": "music"}), json!({"kind": 3}), json!({"query": ["x"]})] {
        assert!(s.execute("elements.list", bad.clone()).is_err(), "{bad}");
    }
}

#[test]
fn every_sound_is_deterministic_and_well_formed() {
    for snd in SOUNDS {
        let a = snd.render().unwrap();
        assert_eq!(a.sample_rate, RATE);
        assert!((a.seconds() - snd.seconds()).abs() < 1e-3, "{}", snd.id);
        assert!(a.left.iter().chain(&a.right).all(|x| x.is_finite() && x.abs() <= 1.0), "{}", snd.id);
        assert!(a.left.iter().any(|x| x.abs() > 0.1), "{} is audible", snd.id);
        assert_eq!(a, snd.render().unwrap(), "{} is the same every time", snd.id);
    }
    // two elements of one kind are different sounds
    assert_ne!(sound("sound.swoosh-up").unwrap().render().unwrap().left, sound("sound.swoosh-down").unwrap().render().unwrap().left);
}

#[test]
fn a_sound_is_placed_as_one_undo_step_and_its_file_reused() {
    let mut s = demo();
    let dir = tmp("sound");
    let undo0 = s.history.undo.len();
    let clips0 = clip_count(&s);
    let items0 = s.project.items.len();
    let r = s.execute("elements.insert", json!({"id": "sound.whoosh-short", "seconds": 3.0, "dir": dir})).unwrap();
    assert_eq!(r["kind"], "sound");
    assert_eq!(r["reused"], false);
    let path = r["path"].as_str().unwrap().to_string();
    assert!(path.ends_with("Whoosh Short.wav"), "{path}");
    let (_, it) = s.active_sequence().unwrap().find_item(ClipId(r["clip"].as_u64().unwrap())).unwrap();
    assert_eq!(it.start, Tick::from_seconds_f64(3.0));
    assert_eq!(it.duration, Tick::from_units(28_800, 48_000));
    assert_eq!(it.name, "Whoosh – Short");
    assert_eq!(s.history.undo.len(), undo0 + 1, "one undo step");
    assert_eq!(clip_count(&s), clips0 + 1);
    // the file holds exactly the element's sound
    let want = crate::audio_generate::write_wav_f32_stereo(
        &sound("sound.whoosh-short").unwrap().render().unwrap().left,
        &sound("sound.whoosh-short").unwrap().render().unwrap().right,
        RATE,
    );
    assert_eq!(std::fs::read(&path).unwrap(), want);
    // again, at the same time: the same item, another free track
    let again = s.execute("elements.insert", json!({"id": "sound.whoosh-short", "seconds": 3.0, "dir": dir})).unwrap();
    assert_eq!((again["reused"].as_bool(), again["item"].as_u64(), again["path"].as_str()), (Some(true), r["item"].as_u64(), Some(path.as_str())));
    assert_ne!(again["track"], r["track"]);
    assert_eq!(s.project.items.len(), items0 + 1, "one project item for both");
    // a deleted file is written back for the same item
    std::fs::remove_file(&path).unwrap();
    let back = s.execute("elements.insert", json!({"id": "sound.whoosh-short", "seconds": 20.0, "dir": dir})).unwrap();
    assert_eq!((back["reused"].as_bool(), back["item"].as_u64()), (Some(true), r["item"].as_u64()));
    assert_eq!(std::fs::read(&path).unwrap(), want);
    s.undo().unwrap();
    // on a named track, and undo takes it off
    let free = s.execute("elements.insert", json!({"id": "sound.click", "seconds": 600.0, "track": "A1", "dir": dir})).unwrap();
    assert_eq!(free["track"], "A1");
    s.undo().unwrap();
    s.undo().unwrap();
    s.undo().unwrap();
    assert_eq!(clip_count(&s), clips0);
}

#[test]
fn a_file_of_the_same_name_is_never_overwritten() {
    let mut s = demo();
    let dir = tmp("foreign");
    std::fs::create_dir_all(&dir).unwrap();
    let mine = std::path::Path::new(&dir).join("Pop.wav");
    std::fs::write(&mine, b"not ours").unwrap();
    let r = s.execute("elements.insert", json!({"id": "sound.pop", "seconds": 1.0, "dir": dir})).unwrap();
    assert_eq!(std::fs::read(&mine).unwrap(), b"not ours");
    assert!(r["path"].as_str().unwrap().ends_with("Pop 1.wav"), "{}", r["path"]);
}

#[test]
fn every_sticker_and_text_preset_inserts_as_a_linked_graphic() {
    let mut s = demo();
    for (k, t) in filmcraft_project::elements::all().into_iter().enumerate() {
        let id = filmcraft_project::elements::element_id(&t).to_string();
        let undo0 = s.history.undo.len();
        let at = 900.0 + 20.0 * k as f64;
        let r = s.execute("elements.insert", json!({"id": id, "seconds": at})).unwrap();
        assert_eq!(s.history.undo.len(), undo0 + 1, "{id}: one undo step");
        let clip = ClipId(r["clip"].as_u64().unwrap());
        let (_, it) = s.active_sequence().unwrap().find_item(clip).unwrap();
        assert_eq!(it.start, s.sequence_rate().snap(Tick::from_seconds_f64(at)), "{id}");
        let link = it.graphic.as_ref().and_then(|g| g.template.as_ref()).unwrap();
        assert_eq!(link.id, t.id);
        let ctl = s.execute("graphics.template.controls", json!({"clip": clip.0})).unwrap();
        assert_eq!(ctl["controls"].as_array().unwrap().len(), t.controls.len(), "{id}");
        // it renders something at its middle
        let (_, it) = s.active_sequence().unwrap().find_item(clip).unwrap();
        let mut img = filmcraft_render::image::Image::new(320, 180);
        let mid = it.source_in + Tick(it.duration.0 / 2);
        filmcraft_render::graphic_clip::render_graphic(it, mid, (1920, 1080), &filmcraft_geom::Affine::scale(1.0 / 6.0, 1.0 / 6.0), &mut img);
        assert!(img.px.chunks(4).filter(|p| p[3] > 0.5).count() > 100, "{id}: something shows");
    }
}

#[test]
fn graphics_take_a_track_a_duration_and_property_values() {
    let mut s = demo();
    let undo0 = s.history.undo.len();
    let r = s
        .execute(
            "elements.insert",
            json!({"id": "sticker.speech-bubble", "seconds": 700.0, "duration": 2.5, "track": "V2", "values": {"text": "Hi!", "Bubble Color": "#ffcc00"}}),
        )
        .unwrap();
    assert_eq!(s.history.undo.len(), undo0 + 1, "one undo step, values included");
    assert_eq!(r["track"], "V2");
    let clip = ClipId(r["clip"].as_u64().unwrap());
    let ctl = s.execute("graphics.template.controls", json!({"clip": clip.0})).unwrap();
    let value = |id: &str| ctl["controls"].as_array().unwrap().iter().find(|c| c["id"] == id).map(|c| c["value"].clone()).unwrap();
    assert_eq!((value("text"), value("bubble_color")), (json!("Hi!"), json!("#ffcc00")));
    let (_, it) = s.active_sequence().unwrap().find_item(clip).unwrap();
    assert_eq!(it.duration, s.sequence_rate().snap_nearest(Tick::from_seconds_f64(2.5)));
    // a text preset lengthened keeps its intro: the title is still popping in after 0.1 s
    let r = s.execute("elements.insert", json!({"id": "text.title-pop", "seconds": 800.0, "duration": 20.0})).unwrap();
    let (_, it) = s.active_sequence().unwrap().find_item(ClipId(r["clip"].as_u64().unwrap())).unwrap();
    let specs = filmcraft_render::graphic_clip::item_layer_specs(it, it.source_in + Tick::from_seconds_f64(0.1), (1920, 1080));
    let title = specs.iter().find(|(_, sp)| sp.name == "Title").map(|(_, sp)| sp.transform.scale.y).unwrap();
    assert!(title > 0.2 && title < 1.0, "popping: {title}");
    let specs = filmcraft_render::graphic_clip::item_layer_specs(it, it.source_in + Tick::from_seconds_f64(10.0), (1920, 1080));
    assert!(specs.iter().any(|(_, sp)| sp.name == "Title" && (sp.transform.scale.y - 1.0).abs() < 1e-6), "settled in the middle");
    // an unknown property undoes the whole insert
    let clips = clip_count(&s);
    assert!(s.execute("elements.insert", json!({"id": "sticker.arrow", "seconds": 900.0, "values": {"nope": 1}})).is_err());
    assert_eq!(clip_count(&s), clips);
    // a clip shorter than a preset's intro and outro still renders, frame by frame
    for (k, t) in filmcraft_project::elements::text_presets().into_iter().enumerate() {
        let id = filmcraft_project::elements::element_id(&t).to_string();
        let r = s.execute("elements.insert", json!({"id": id, "seconds": 1000.0 + k as f64, "duration": 0.1})).unwrap();
        let (_, it) = s.active_sequence().unwrap().find_item(ClipId(r["clip"].as_u64().unwrap())).unwrap();
        let mut img = filmcraft_render::image::Image::new(64, 36);
        for f in 0..4 {
            let mt = it.source_in + Tick(it.duration.0 * f / 4);
            filmcraft_render::graphic_clip::render_graphic(it, mt, (1920, 1080), &filmcraft_geom::Affine::scale(1.0 / 30.0, 1.0 / 30.0), &mut img);
        }
        assert!(img.px.iter().all(|x| x.is_finite()), "{id}");
    }
}

#[test]
fn hostile_parameters_are_refused_without_changes() {
    let mut s = demo();
    let dir = tmp("hostile");
    let undo0 = s.history.undo.len();
    let clips0 = clip_count(&s);
    for p in [
        json!({}),
        json!({"id": 7}),
        json!({"id": "sound.explosion"}),
        json!({"id": ""}),
        json!({"id": "x".repeat(10_000)}),
        json!({"id": "sound.pop", "time": -1}),
        json!({"id": "sound.pop", "time": 1.5}),
        json!({"id": "sound.pop", "time": "soon"}),
        json!({"id": "sound.pop", "seconds": -3.0}),
        json!({"id": "sound.pop", "seconds": "x"}),
        json!({"id": "sound.pop", "seconds": 1e300}),
        json!({"id": "sound.pop", "time": i64::MAX}),
        json!({"id": "sound.pop", "track": "A99"}),
        json!({"id": "sound.pop", "track": "V1"}),
        json!({"id": "sound.pop", "track": [1]}),
        json!({"id": "sound.pop", "track": "A1", "seconds": 0.0}),
        json!({"id": "sound.pop", "duration": 3.0}),
        json!({"id": "sound.pop", "values": {"a": 1}}),
        json!({"id": "sticker.arrow", "track": "V99"}),
        json!({"id": "sticker.arrow", "track": "A1"}),
        json!({"id": "sticker.arrow", "track": "V1", "seconds": 0.0}),
        json!({"id": "sticker.arrow", "track": 123456789}),
        json!({"id": "sticker.arrow", "track": -1}),
        json!({"id": "sticker.arrow", "duration": 0.0}),
        json!({"id": "sticker.arrow", "duration": -2.0}),
        json!({"id": "sticker.arrow", "duration": 1e9}),
        json!({"id": "sticker.arrow", "duration": "long"}),
        json!({"id": "sticker.arrow", "values": [1, 2]}),
        json!({"id": "text.title-pop", "values": {"title": 1}}),
    ] {
        let p = {
            let mut p = p;
            if p.get("id").is_some_and(|v| v.as_str().is_some_and(|i| i.starts_with("sound."))) {
                p["dir"] = json!(dir);
            }
            p
        };
        assert!(s.execute("elements.insert", p.clone()).is_err(), "{p}");
        assert_eq!(s.history.undo.len(), undo0, "{p}: no undo step");
        assert_eq!(clip_count(&s), clips0, "{p}: no clip");
    }
    assert!(std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0) == 0, "no file was written");
    // without a sequence
    let mut empty = Session::default();
    assert!(empty.execute("elements.insert", json!({"id": "sound.pop"})).is_err());
}

/// The sticker previews in `assets/elements/stickers/` are exported from the vectors; this
/// fails when they drift (`FILMCRAFT_BLESS=1` rewrites them).
#[test]
fn bundled_sticker_svgs_match_the_vectors() {
    let dir = repo().join(STICKER_DIR);
    let bless = std::env::var_os("FILMCRAFT_BLESS").is_some();
    let stickers = filmcraft_project::elements::stickers();
    for t in &stickers {
        let svg = sticker_svg(t);
        assert!(svg.starts_with("<?xml") && svg.trim_end().ends_with("</svg>"));
        assert!(!svg.contains("NaN") && !svg.contains("inf"), "{}", t.id);
        let path = repo().join(sticker_file(t));
        if bless {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(&path, &svg).unwrap();
        }
        let on_disk = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e} (FILMCRAFT_BLESS=1 writes it)", path.display()));
        assert_eq!(on_disk, svg, "{} drifted from {} (FILMCRAFT_BLESS=1 cargo test -p filmcraft-engine bundled_sticker_svgs)", path.display(), t.id);
    }
    // no stray previews
    let svgs: Vec<String> =
        std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned())).filter(|n| n.ends_with(".svg")).collect();
    assert_eq!(svgs.len(), stickers.len(), "{svgs:?}");
}

/// AGENTS.md §1: every file of the bundled library has a complete sidecar and an index row
/// (`cargo xtask assets` checks the whole repository; this pins the library's own files).
#[test]
fn every_bundled_element_file_has_a_sidecar_and_an_index_row() {
    let root = repo();
    let index = std::fs::read_to_string(root.join("ATTRIBUTION.md")).unwrap();
    let mut files = Vec::new();
    let mut stack = vec![root.join("assets/elements")];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if !p.to_string_lossy().ends_with(".attribution") && p.extension().is_none_or(|e| e != "md") {
                files.push(p);
            }
        }
    }
    assert!(files.len() >= 10, "{files:?}");
    for f in files {
        let rel = f.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/");
        let side = std::fs::read_to_string(format!("{}.attribution", f.display())).unwrap_or_else(|e| panic!("{rel}: no sidecar: {e}"));
        for field in ["asset", "title", "author", "source", "license", "added", "notes"] {
            assert!(
                side.lines().any(|l| l.split_once(':').is_some_and(|(k, v)| k.trim() == field && !v.trim().is_empty())),
                "{rel}.attribution: `{field}` missing or empty"
            );
        }
        assert!(side.contains("MIT OR Apache-2.0") && side.contains("original work"), "{rel}.attribution");
        assert!(index.contains(&format!("`{rel}`")), "{rel} is not in ATTRIBUTION.md");
    }
}
