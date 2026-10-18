//! Crate tests: manifest round trips and validation errors, hostile packages, and applying the
//! built-in starters to a project (exact durations, dedup, variants, beat snapping).

use std::collections::BTreeMap;

use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{DemoScene, Generator, MediaKind, MediaSource};
use filmcraft_project::{ClipId, ItemId, ItemKind, Label, MediaClip, MediaRef, Project, Sequence, TrackItem};
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick};
use serde_json::{Value, json};

use crate::apply::{ApplyOptions, MediaFill, MusicBeats, SlotFill, apply, parse_slot_marker};
use crate::builtin::{BUILTIN_IDS, build_all, builtin_templates};
use crate::manifest::{self, SlotKind};
use crate::{Package, TemplateError};

const S: i64 = TICKS_PER_SECOND;

fn builtin(id: &str) -> Package {
    builtin_templates().into_iter().find(|p| p.manifest.id == id).unwrap()
}

fn manifest_json(id: &str) -> Value {
    serde_json::to_value(&builtin(id).manifest).unwrap()
}

/// The issues of a manifest edited by `f`.
fn issues_after(f: impl FnOnce(&mut Value)) -> Vec<String> {
    let mut v = manifest_json(BUILTIN_IDS[0]);
    f(&mut v);
    match manifest::from_value(&v) {
        Err(TemplateError::Invalid(is)) => is.iter().map(|i| i.to_string()).collect(),
        Err(e) => vec![e.to_string()],
        Ok(_) => Vec::new(),
    }
}

fn media(p: &mut Project, name: &str, kind: MediaKind, secs: f64, size: (u32, u32), audio: bool) -> ItemId {
    let src = GeneratorSource::new(Generator::Demo(DemoScene::Forest), size.0, size.1, FrameRate::FPS_30, Tick::from_seconds_f64(secs));
    let mut info = src.info().clone();
    info.kind = kind;
    info.name = name.into();
    if !audio {
        info.audio_streams.clear();
    }
    if kind == MediaKind::AudioOnly {
        info.video = None;
    }
    let path = format!("/media/{name}");
    let clip = MediaClip {
        media: MediaRef::File { path },
        info,
        interpret: Default::default(),
        mark_in: None,
        mark_out: None,
        markers: vec![],
        offline: false,
        proxy: None,
        identity: None,
    };
    p.add_item(name, Label::Iris, ItemKind::Media(clip), None)
}

fn photo(p: &mut Project, name: &str) -> ItemId {
    media(p, name, MediaKind::Still, 5.0, (4000, 3000), false)
}

fn seq(p: &Project, id: ItemId) -> &Sequence {
    p.sequence(id).unwrap()
}

fn clip(q: &Sequence, id: ClipId) -> &TrackItem {
    q.find_item(id).unwrap().1
}

#[test]
fn every_builtin_builds_and_validates() {
    for (name, r) in build_all() {
        let p = r.unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(p.check().is_empty(), "{name}: {:?}", p.check());
        assert!(p.manifest.variants.len() >= 2, "{name}");
        assert!(p.manifest.variant(Some("9:16")).is_some() && p.manifest.variant(Some("16:9")).is_some());
    }
    let ids: Vec<String> = builtin_templates().iter().map(|p| p.manifest.id.clone()).collect();
    assert_eq!(ids, BUILTIN_IDS);
}

#[test]
fn manifest_round_trips() {
    for p in builtin_templates() {
        let json = p.manifest.to_json();
        let back = manifest::parse(json.as_bytes()).unwrap_or_else(|e| panic!("{}: {e}", p.manifest.id));
        assert_eq!(back, p.manifest);
    }
}

#[test]
fn package_round_trips_through_files_and_zip() {
    for p in builtin_templates() {
        let files: BTreeMap<String, Vec<u8>> = p.files().into_iter().collect();
        let back = Package::from_files(files).unwrap();
        assert_eq!(back.manifest, p.manifest);
        assert_eq!(back.fragment.items.len(), p.fragment.items.len());
        let zip = p.to_zip().unwrap();
        let z = Package::from_zip(&zip).unwrap();
        assert_eq!(z.manifest, p.manifest);
        // a zipped folder (`Name.fcvideo-template/…`) reads the same
        let nested: Vec<(String, Vec<u8>)> = p.files().into_iter().map(|(k, v)| (format!("My.fcvideo-template/{k}"), v)).collect();
        let n = Package::from_zip(&crate::zip::write(&nested).unwrap()).unwrap();
        assert_eq!(n.manifest, p.manifest);
    }
}

#[test]
fn validation_errors_name_their_field() {
    let cases: Vec<(Box<dyn FnOnce(&mut Value)>, &str)> = vec![
        (Box::new(|v| v["name"] = json!(42)), "name: expected a string"),
        (Box::new(|v| v["name"] = json!("  ")), "name: is empty"),
        (Box::new(|v| v["id"] = json!("Not An Id")), "id: `Not An Id`"),
        (Box::new(|v| v["license"] = json!("CC-BY-NC-4.0")), "license:"),
        (Box::new(|v| v["formatVersion"] = json!("one")), "formatVersion"),
        (Box::new(|v| v["bogus"] = json!(1)), "bogus: unknown field"),
        (Box::new(|v| v["duration"] = json!(-5)), "duration:"),
        (Box::new(|v| v["variants"] = json!([])), "variants: a template needs at least one variant"),
        (Box::new(|v| v["variants"][0]["width"] = json!(0)), "variants[0].width"),
        (Box::new(|v| v["variants"][1]["id"] = json!("16:9")), "variants[1].id: `16:9` is used twice"),
        (Box::new(|v| v["defaultVariant"] = json!("4:3")), "defaultVariant"),
        (Box::new(|v| v["slots"][0]["kind"] = json!("sticker")), "slots[0].kind: `sticker`"),
        (Box::new(|v| v["slots"][0]["id"] = json!("Bad Id")), "slots[0].id"),
        (Box::new(|v| v["slots"][1]["id"] = json!("photos")), "slots[1].id: `photos` is used twice"),
        (Box::new(|v| v["slots"][0]["count"] = json!(0)), "slots[0].count"),
        (Box::new(|v| v["slots"][0]["minCount"] = json!(9)), "slots[0].minCount: 9 is more than count"),
        (Box::new(|v| v["slots"][0]["placements"] = json!([])), "slots[0].placements: a slot needs at least one placement"),
        (Box::new(|v| v["slots"][0]["placements"][0]["media"] = json!(8)), "slots[0].placements[0].media"),
        (Box::new(|v| v["slots"][0]["placements"][1]["clip"] = v["slots"][0]["placements"][0]["clip"].clone()), "is placed twice"),
        (Box::new(|v| v["slots"][0]["placements"][0]["layer"] = json!(1)), "slots[0].placements[0].layer: only text slots"),
        (Box::new(|v| v["slots"][0]["fit"]["short"] = json!("teleport")), "slots[0].fit.short"),
        (Box::new(|v| v["slots"][0]["fit"]["short"] = json!("remix")), "slots[0].fit.short: `remix` is for music slots"),
        (Box::new(|v| v["slots"][0]["fit"]["minSpeed"] = json!(5)), "slots[0].fit.minSpeed"),
        (Box::new(|v| v["slots"][0]["constraints"]["minDuration"] = json!(10 * S)), "slots[0].constraints.minDuration: is longer than maxDuration"),
        (Box::new(|v| v["slots"][0]["constraints"]["aspect"] = json!("wide")), "slots[0].constraints.aspect"),
        (Box::new(|v| v["slots"][0]["constraints"]["orientation"] = json!("diagonal")), "slots[0].constraints.orientation"),
        (Box::new(|v| v["slots"][0]["constraints"]["maxChars"] = json!(10)), "slots[0].constraints.maxChars: only text slots"),
        (Box::new(|v| v["slots"][1]["count"] = json!(2)), "slots[1].count: a music slot takes one"),
        (Box::new(|v| v["slots"][1]["default"] = json!({"generate": {"mood": "angry"}})), "slots[1].default.generate.mood"),
        (Box::new(|v| v["slots"][2]["default"] = json!({"generate": {}})), "slots[2].default.generate: only music slots"),
        (Box::new(|v| v["slots"][0]["default"] = json!({"text": "x"})), "slots[0].default.text: only text slots"),
        (Box::new(|v| v["slots"][1]["beatSync"] = json!(true)), "slots[1].beatSync"),
        (Box::new(|v| v["beatGrid"]["beats"] = json!([5, 3])), "beatGrid.beats[1]: beats must be in increasing order"),
        (Box::new(|v| v["beatGrid"]["bpm"] = json!(1000)), "beatGrid.bpm"),
        (Box::new(|v| v.as_object_mut().unwrap().remove("beatGrid").map(|_| ()).unwrap()), "beatGrid: slots cut on the beat"),
        (Box::new(|v| v["look"] = json!({"lumetri": "A", "preset": "B"})), "look.lumetri"),
        (Box::new(|v| v["captionStyle"]["size"] = json!(-3)), "captionStyle.size"),
        (Box::new(|v| v["captionStyle"]["glow"] = json!(true)), "captionStyle.glow: unknown field"),
        (Box::new(|v| v["sequenceFile"] = json!("../outside.json")), "sequenceFile"),
        (
            Box::new(|v| v["assets"] = json!([{"path": "../x", "sha256": "zz", "title": "t", "author": "a", "source": "s", "license": "CC-BY-ND-4.0"}])),
            "assets[0].path",
        ),
        (Box::new(|v| v["variants"][1]["hints"][0]["position"] = json!([0.5])), "variants[1].hints[0].position: expected [x, y]"),
        (Box::new(|v| v["tags"] = json!([1])), "tags[0]"),
        // boxes, the poster, the generated bed's genre
        (Box::new(|v| v["slots"][0]["placements"][0]["box"] = json!([0, 0, 0, 1])), "slots[0].placements[0].box: expected [x, y, width, height]"),
        (Box::new(|v| v["slots"][0]["placements"][0]["box"] = json!([0, 0, f64::MAX, 1])), "slots[0].placements[0].box"),
        (Box::new(|v| v["slots"][2]["placements"][0]["box"] = json!([0, 0, 1, 1])), "only video, photo and logo slots take a box"),
        (Box::new(|v| v["variants"][1]["hints"][0]["box"] = json!([0, 0, 5, 1])), "variants[1].hints[0].box"),
        (Box::new(|v| v["poster"] = json!(99 * S)), "poster: is after the end"),
        (Box::new(|v| v["poster"] = json!(-1)), "poster:"),
        (Box::new(|v| v["slots"][1]["default"] = json!({"generate": {"genre": "polka"}})), "slots[1].default.generate.genre"),
        // captions slots
        (
            Box::new(|v| v["slots"].as_array_mut().unwrap().push(json!({"id": "cc", "kind": "captions", "captions": {"source": ["nope"]}}))),
            "slots[4].captions.source[0]: the template has no slot `nope`",
        ),
        (
            Box::new(|v| v["slots"].as_array_mut().unwrap().push(json!({"id": "cc", "kind": "captions", "captions": {"source": ["title"]}}))),
            "is a text slot; captions come from video or music slots",
        ),
        (
            Box::new(|v| {
                v["slots"].as_array_mut().unwrap().push(json!({"id": "cc", "kind": "captions", "placements": [{"clip": 5}], "captions": {"source": ["music"]}}))
            }),
            "a captions slot has no placements",
        ),
        (Box::new(|v| v["slots"].as_array_mut().unwrap().push(json!({"id": "cc", "kind": "captions"}))), "a captions slot needs `captions`"),
        (
            Box::new(|v| v["slots"].as_array_mut().unwrap().push(json!({"id": "cc", "kind": "captions", "count": 2, "captions": {"source": ["music"]}}))),
            "a captions slot takes one",
        ),
        (
            Box::new(|v| {
                v["slots"].as_array_mut().unwrap().push(json!({"id": "cc", "kind": "captions", "captions": {"source": ["music"], "lines": 9, "maxChars": 1}}))
            }),
            "slots[4].captions.lines",
        ),
        (
            Box::new(|v| {
                let s = json!({"id": "c1", "kind": "captions", "captions": {"source": ["music"]}});
                let a = v["slots"].as_array_mut().unwrap();
                a.push(s.clone());
                a.push(json!({"id": "c2", "kind": "captions", "captions": {"source": ["music"]}}));
            }),
            "a template has one captions slot at most",
        ),
        (Box::new(|v| v["slots"][0]["captions"] = json!({"source": ["music"]})), "slots[0].captions: only captions slots take `captions`"),
    ];
    for (i, (f, want)) in cases.into_iter().enumerate() {
        let got = issues_after(f);
        assert!(got.iter().any(|g| g.contains(want)), "case {i}: wanted an issue containing {want:?}, got {got:?}");
    }
    // several problems are reported together
    let many = issues_after(|v| {
        v["name"] = json!(1);
        v["slots"][0]["kind"] = json!("x");
        v["variants"][0]["height"] = json!("tall");
    });
    assert!(many.len() >= 3, "{many:?}");
}

#[test]
fn format_and_version_gate() {
    let mut v = manifest_json(BUILTIN_IDS[1]);
    v["formatVersion"] = json!(99);
    assert_eq!(manifest::from_value(&v), Err(TemplateError::TooNew(99)));
    v["format"] = json!("filmcraft.graphicsTemplate");
    assert!(matches!(manifest::from_value(&v), Err(TemplateError::NotTemplate(_))));
    assert!(matches!(manifest::parse(b"PK\x03\x04rest"), Err(TemplateError::NotTemplate(_))));
    assert!(matches!(manifest::parse(b"\xff\xfe"), Err(TemplateError::NotTemplate(_))));
    assert!(matches!(manifest::parse(b"[1,2]"), Err(TemplateError::NotTemplate(_))));
}

#[test]
fn cross_checks_with_the_sequence() {
    let p = builtin("builtin:photo-beat-slideshow");
    let check = |f: &dyn Fn(&mut Package)| -> Vec<String> {
        let mut q = p.clone();
        f(&mut q);
        q.check().0.iter().map(|i| i.to_string()).collect()
    };
    let got = check(&|q| q.manifest.slots[0].placements[0].clip = 999_999);
    assert!(got.iter().any(|g| g.contains("slots[0].placements[0].clip") && g.contains("not in the template sequence")), "{got:?}");
    let got = check(&|q| q.manifest.duration = Tick(3 * S));
    assert!(got.iter().any(|g| g.starts_with("duration")), "{got:?}");
    let got = check(&|q| {
        let music = q.manifest.slots[1].placements[0].clip;
        q.manifest.slots[0].placements[0].clip = music;
    });
    assert!(got.iter().any(|g| g.contains("must be on a video track")), "{got:?}");
    let got = check(&|q| q.manifest.slots[2].placements[0].layer = Some(77));
    assert!(got.iter().any(|g| g.contains("slots[2].placements[0].layer")), "{got:?}");
    let got = check(&|q| q.manifest.slots[0].constraints.max_duration = Some(Tick(S)));
    assert!(got.iter().any(|g| g.contains("maxDuration")), "{got:?}");
    // a template's media is generated or bundled, never a path to someone's file
    let got = check(&|q| {
        if let Some(ItemKind::Media(m)) = q.fragment.items.values_mut().find(|i| i.as_media().is_some()).map(|i| &mut i.kind) {
            m.media = MediaRef::File { path: "/Users/someone/Movies/private.mov".into() };
        }
    });
    assert!(got.iter().any(|g| g.contains("is generated or a bundled asset")), "{got:?}");
    // an unlisted bundled file and a listed one with a bad hash
    let got = check(&|q| {
        q.assets.insert("assets/extra.wav".into(), vec![1, 2, 3]);
    });
    assert!(got.iter().any(|g| g.contains("assets/extra.wav")), "{got:?}");
}

fn hostile_bytes() -> Vec<Vec<u8>> {
    let deep = format!("{}{}", "[".repeat(5000), "]".repeat(5000));
    let deep_obj = format!("{{\"format\":\"filmcraft.videoTemplate\",\"formatVersion\":1,\"slots\":{}{}}}", "[".repeat(200), "]".repeat(200));
    let long = format!("{{\"format\":\"filmcraft.videoTemplate\",\"formatVersion\":1,\"name\":\"{}\"}}", "x".repeat(1 << 20));
    let many = format!(
        "{{\"format\":\"filmcraft.videoTemplate\",\"formatVersion\":1,\"slots\":[{}]}}",
        (0..5000).map(|i| format!("{{\"id\":\"s{i}\",\"kind\":\"photo\",\"placements\":[{{\"clip\":{i}}}]}}")).collect::<Vec<_>>().join(",")
    );
    let wrong_types = r#"{"format":"filmcraft.videoTemplate","formatVersion":1,"id":[],"name":{},"sequence":"seven","duration":1e308,"variants":"16:9","slots":{"a":1},"beatGrid":{"bpm":null,"beats":[1.5,"x",-1]},"look":"warm","captionStyle":7,"assets":[null]}"#;
    let huge_numbers = r#"{"format":"filmcraft.videoTemplate","formatVersion":18446744073709551615,"id":"a","name":"n","author":"a","license":"MIT","sequence":18446744073709551615,"duration":9223372036854775807,"variants":[{"id":"v","width":4294967295,"height":-1}],"slots":[]}"#;
    vec![
        deep.into_bytes(),
        deep_obj.into_bytes(),
        long.into_bytes(),
        many.into_bytes(),
        wrong_types.as_bytes().to_vec(),
        huge_numbers.as_bytes().to_vec(),
        vec![0xEF, 0xBB, 0xBF],
    ]
}

#[test]
fn hostile_manifests_never_panic() {
    for b in hostile_bytes() {
        let r = std::panic::catch_unwind(|| manifest::parse(&b));
        assert!(r.is_ok(), "panicked on {:?}", String::from_utf8_lossy(&b[..b.len().min(80)]));
        assert!(r.unwrap().is_err());
    }
}

#[test]
fn hostile_packages_never_panic() {
    let good = builtin("builtin:top-3-listicle");
    let zip = good.to_zip().unwrap();
    // byte mutations of a whole package (manifest, sequence, central directory)
    let mut n = 0;
    for i in (0..zip.len()).step_by(7) {
        for v in [0u8, 0xFF, b'9', b'"'] {
            let mut z = zip.clone();
            z[i] = v;
            let r = std::panic::catch_unwind(|| Package::from_zip(&z));
            assert!(r.is_ok(), "panicked with byte {i} = {v}");
            n += 1;
        }
    }
    assert!(n > 100);
    // path traversal, duplicate and unexpected files at the package level
    let mut files = good.files();
    files.push(("../../escape.txt".into(), b"x".to_vec()));
    assert!(crate::zip::write(&files).is_err(), "the writer refuses it too");
    let mut files: BTreeMap<String, Vec<u8>> = good.files().into_iter().collect();
    files.insert("../../escape.txt".into(), b"x".to_vec());
    assert!(matches!(Package::from_files(files), Err(TemplateError::Package(_))));
    let mut files: BTreeMap<String, Vec<u8>> = good.files().into_iter().collect();
    files.insert("run.sh".into(), b"rm -rf /".to_vec());
    assert!(matches!(Package::from_files(files), Err(TemplateError::Invalid(_))));
    // a sequence file that is not a project, or one from the future
    let mut files: BTreeMap<String, Vec<u8>> = good.files().into_iter().collect();
    files.insert("sequence.json".into(), b"{\"format\":\"filmcraft.project\",\"schema_version\":9999,\"project\":{}}".to_vec());
    assert!(matches!(Package::from_files(files), Err(TemplateError::Invalid(_))));
    let mut files: BTreeMap<String, Vec<u8>> = good.files().into_iter().collect();
    files.remove("sequence.json");
    assert!(Package::from_files(files).is_err());
    // manifests of the hostile kind inside an otherwise good package
    for b in hostile_bytes() {
        let mut files: BTreeMap<String, Vec<u8>> = good.files().into_iter().collect();
        files.insert("template.json".into(), b);
        let r = std::panic::catch_unwind(|| Package::from_files(files));
        assert!(r.is_ok_and(|r| r.is_err()));
    }
}

/// A project with media for every slot of the slideshow.
fn slideshow_fills(p: &mut Project, photos: usize) -> BTreeMap<String, SlotFill> {
    let mut fills = BTreeMap::new();
    let ph: Vec<MediaFill> = (0..photos).map(|i| MediaFill::new(photo(p, &format!("photo{i}.jpg")))).collect();
    fills.insert("photos".into(), SlotFill::Media(ph));
    fills.insert("title".into(), SlotFill::Text(vec!["Lisbon 2026".into()]));
    fills
}

#[test]
fn apply_fills_slots_with_exact_tick_durations() {
    let pkg = builtin("builtin:photo-beat-slideshow");
    let mut p = Project::new("p");
    let fills = slideshow_fills(&mut p, 8);
    let a = apply(&mut p, &pkg, &ApplyOptions { fills, ..Default::default() }).unwrap();
    assert_eq!(a.variant, "16:9");
    assert_eq!(a.duration, Tick(16 * S), "exactly the template's length");
    let q = seq(&p, a.sequence);
    assert_eq!((q.settings.width, q.settings.height), (1920, 1080));
    let photos: Vec<_> = a.placed.iter().filter(|x| x.slot == "photos").collect();
    assert_eq!(photos.len(), 8);
    for (k, pc) in photos.iter().enumerate() {
        assert_eq!(pc.clips.len(), 1);
        let c = clip(q, pc.clips[0]);
        assert_eq!((c.start, c.duration), (Tick(2 * S * k as i64), Tick(2 * S)), "photo {k}");
        assert_eq!(p.item(c.item).unwrap().name, format!("photo{k}.jpg"));
        assert!(
            c.markers
                .iter()
                .any(|m| parse_slot_marker(&m.name).is_some_and(|f| (f.kind, f.slot.as_str(), f.placement) == (SlotKind::Photo, "photos", Some(k))))
        );
        // the slot keeps its effects: the push-in and the look
        assert!(c.effect("motion").unwrap().param("scale").unwrap().is_animated());
        assert!(c.effect("lumetri").is_none(), "the look is applied by the engine (resolved preset)");
    }
    // the title text was set; the end line kept its placeholder text
    let title = a.placed.iter().find(|x| x.slot == "title").unwrap();
    let t = clip(q, title.clips[0]);
    let text = t.effects.iter().find(|e| e.effect == "graphic_text").unwrap().param("text").unwrap().value.clone();
    assert_eq!(text, filmcraft_project::ParamValue::Text("Lisbon 2026".into()));
    // transitions stay attached to the slots
    let v1 = &q.video_tracks[0];
    assert!(v1.transitions.iter().any(|tr| tr.from == Some(photos[3].clips[0]) && tr.to == Some(photos[4].clips[0])));
    assert!(v1.transitions.iter().any(|tr| tr.from == Some(photos[7].clips[0]) && tr.to.is_none()));
    // the export preset of the variant, and the template recorded on the sequence
    assert_eq!(a.export_preset.as_deref(), Some("YouTube 1080p Full HD"));
    assert_eq!(p.item(a.sequence).unwrap().metadata.get("Template").map(String::as_str), Some("builtin:photo-beat-slideshow"));
    assert!(q.caption_tracks.len() == 1 && q.caption_tracks[0].style.outline == 3.0, "the caption style travels with the template");
    q.check().unwrap();
}

#[test]
fn one_photo_fills_several_slots() {
    let pkg = builtin("builtin:photo-beat-slideshow");
    let mut p = Project::new("p");
    let fills = slideshow_fills(&mut p, 3);
    let a = apply(&mut p, &pkg, &ApplyOptions { fills, ..Default::default() }).unwrap();
    let q = seq(&p, a.sequence);
    let names: Vec<String> = a.placed.iter().filter(|x| x.slot == "photos").map(|x| p.item(clip(q, x.clips[0]).item).unwrap().name.clone()).collect();
    assert_eq!(names, ["photo0.jpg", "photo1.jpg", "photo2.jpg", "photo0.jpg", "photo1.jpg", "photo2.jpg", "photo0.jpg", "photo1.jpg"]);
    // the same item, not copies
    let items: std::collections::HashSet<ItemId> = a.placed.iter().filter(|x| x.slot == "photos").map(|x| x.item).collect();
    assert_eq!(items.len(), 3);
}

#[test]
fn explicit_media_indexes_reorder_a_countdown() {
    let pkg = builtin("builtin:top-3-listicle");
    let mut p = Project::new("p");
    let clips: Vec<MediaFill> = (1..=3).map(|k| MediaFill::new(media(&mut p, &format!("best{k}.mp4"), MediaKind::Movie, 10.0, (1920, 1080), true))).collect();
    let mut fills = BTreeMap::new();
    fills.insert("items".into(), SlotFill::Media(clips));
    fills.insert("labels".into(), SlotFill::Text(vec!["Gold".into(), "Silver".into(), "Bronze".into()]));
    let a = apply(&mut p, &pkg, &ApplyOptions { fills, ..Default::default() }).unwrap();
    let q = seq(&p, a.sequence);
    let order: Vec<String> = q.video_tracks[0].items.iter().filter_map(|c| p.item(c.item).map(|i| i.name.clone())).filter(|n| n.starts_with("best")).collect();
    assert_eq!(order, ["best3.mp4", "best2.mp4", "best1.mp4"], "#3 plays first, #1 last");
    let labels: Vec<String> = a
        .placed
        .iter()
        .filter(|x| x.slot == "labels")
        .map(|x| {
            let c = clip(q, x.clips[0]);
            match &c.effects.iter().filter(|e| e.effect == "graphic_text").nth(1).unwrap().param("text").unwrap().value {
                filmcraft_project::ParamValue::Text(t) => t.clone(),
                _ => String::new(),
            }
        })
        .collect();
    assert_eq!(labels, ["Bronze", "Silver", "Gold"]);
    // long clips are trimmed around their middle (portion: center) to exactly 4 s
    for x in a.placed.iter().filter(|x| x.slot == "items") {
        let c = clip(q, x.clips[0]);
        assert_eq!((c.duration, c.source_in, x.how.as_str()), (Tick(4 * S), Tick(3 * S), "trimmed"));
    }
    q.check().unwrap();
}

#[test]
fn short_clips_stretch_or_loop_to_the_slot() {
    let pkg = builtin("builtin:top-3-listicle");
    let mut p = Project::new("p");
    let three = media(&mut p, "three.mp4", MediaKind::Movie, 3.0, (1920, 1080), false);
    let one = media(&mut p, "one.mp4", MediaKind::Movie, 1.0, (1920, 1080), false);
    let mut fills = BTreeMap::new();
    fills.insert("items".into(), SlotFill::Media(vec![MediaFill::new(three), MediaFill::new(one)]));
    let a = apply(&mut p, &pkg, &ApplyOptions { fills, ..Default::default() }).unwrap();
    let q = seq(&p, a.sequence);
    for x in a.placed.iter().filter(|x| x.slot == "items") {
        let total: Tick = x.clips.iter().map(|c| clip(q, *c).duration).fold(Tick::ZERO, |a, b| a + b);
        assert_eq!(total, Tick(4 * S), "{x:?}");
        if x.item == three {
            assert_eq!(x.how, "stretched");
            assert!((x.speed - 0.75).abs() < 1e-9);
        } else {
            assert_eq!(x.how, "looped");
            assert_eq!(x.clips.len(), 4);
        }
    }
    assert!(a.warnings.iter().any(|w| w.contains("loops")), "{:?}", a.warnings);
    q.check().unwrap();
}

#[test]
fn variants_reframe_the_same_media() {
    let pkg = builtin("builtin:photo-beat-slideshow");
    for (variant, size) in [("16:9", (1920u32, 1080u32)), ("9:16", (1080, 1920))] {
        let mut p = Project::new("p");
        let mut fills = slideshow_fills(&mut p, 1);
        if let Some(SlotFill::Media(list)) = fills.get_mut("photos") {
            list[0].focus_by_variant.insert("9:16".into(), [0.2, 0.5]);
        }
        let a = apply(&mut p, &pkg, &ApplyOptions { fills, variant: Some(variant.into()), ..Default::default() }).unwrap();
        let q = seq(&p, a.sequence);
        assert_eq!((q.settings.width, q.settings.height), size);
        assert_eq!(a.variant, variant);
        let first = clip(q, a.placed.iter().find(|x| x.slot == "photos").unwrap().clips[0]);
        let m = first.effect("motion").unwrap();
        // a 4000×3000 photo fills the frame: scale = max(W/4000, H/3000) of the template's 100 → 108
        let k = (size.0 as f64 / 4000.0).max(size.1 as f64 / 3000.0) * 100.0;
        let keys: Vec<f64> = m.param("scale").unwrap().keyframes.iter().map(|k| k.value.as_f64().unwrap()).collect();
        assert!((keys[0] - k).abs() < 1e-6 && (keys[1] - k * 1.08).abs() < 1e-6, "{variant}: {keys:?}");
        let pos = m.param("position").unwrap().value.as_vec2().unwrap();
        if variant == "9:16" {
            // focus on the left: the picture moves right, but never uncovers the frame
            let half = 4000.0 * k / 100.0 / 2.0;
            assert!((pos.x - half).abs() < 1e-6, "{pos:?}");
        } else {
            assert!((pos.x - 960.0).abs() < 1e-6 && (pos.y - 540.0).abs() < 1e-6);
        }
        // the title follows the variant's hint
        let t = clip(q, a.placed.iter().find(|x| x.slot == "title").unwrap().clips[0]);
        let layer = t.effects.iter().find(|e| e.effect == "graphic_text").unwrap();
        let tp = layer.param("position").unwrap().value.as_vec2().unwrap();
        if variant == "9:16" {
            assert!((tp.y - 0.72 * 1920.0).abs() < 1e-6 && layer.param("size").unwrap().value.as_f64() == Some(96.0));
        } else {
            assert!((tp.y - 900.0).abs() < 1e-6);
        }
        assert_eq!(a.export_preset.as_deref(), Some(if variant == "9:16" { "Social Vertical 1080×1920" } else { "YouTube 1080p Full HD" }));
        // the template's own graphics canvas follows the frame
        assert!(matches!(p.item(t.item).unwrap().kind, ItemKind::Graphic { width, height, .. } if (width, height) == size));
    }
}

#[test]
fn cuts_follow_the_beats_of_new_music() {
    let pkg = builtin("builtin:photo-beat-slideshow");
    let mut p = Project::new("p");
    let mut fills = slideshow_fills(&mut p, 8);
    let song = media(&mut p, "song.wav", MediaKind::AudioOnly, 60.0, (0, 0), true);
    fills.insert("music".into(), SlotFill::Media(vec![MediaFill::new(song)]));
    // the song beats at 100 bpm from 0.25 s (media time)
    let period = 0.6;
    let beats: Vec<Tick> = (0..100).map(|k| Tick::from_seconds_f64(0.25 + period * k as f64)).collect();
    let a = apply(&mut p, &pkg, &ApplyOptions { fills, music_beats: Some(MusicBeats { beats: beats.clone(), mode: None }), ..Default::default() }).unwrap();
    assert!(a.retimed);
    let q = seq(&p, a.sequence);
    let rate = q.settings.frame_rate;
    // every cut between photos is on a beat of the song (to the frame), each photo still 4 beats
    let photos: Vec<&TrackItem> = a.placed.iter().filter(|x| x.slot == "photos").map(|x| clip(q, x.clips[0])).collect();
    for (k, c) in photos.iter().enumerate().skip(1) {
        let want = rate.snap_nearest(beats[4 * k]);
        assert_eq!(c.start, want, "photo {k}");
        assert_eq!(photos[k - 1].end(), c.start, "back to back");
    }
    // the music now plays for the whole re-timed show, from its start
    let music = clip(q, a.placed.iter().find(|x| x.slot == "music").unwrap().clips[0]);
    assert_eq!((music.start, music.end()), (Tick::ZERO, q.duration()));
    // and the outro text moved with the last photo
    let outro = clip(q, a.placed.iter().find(|x| x.slot == "outro").unwrap().clips[0]);
    assert_eq!((outro.start, outro.end()), (photos[7].start, photos[7].end()));
    q.check().unwrap();
}

#[test]
fn without_new_music_cuts_stay_on_the_template_grid() {
    let pkg = builtin("builtin:photo-beat-slideshow");
    let mut p = Project::new("p");
    let fills = slideshow_fills(&mut p, 8);
    let a = apply(&mut p, &pkg, &ApplyOptions { fills, ..Default::default() }).unwrap();
    assert!(!a.retimed);
    let grid = &pkg.manifest.beat_grid.as_ref().unwrap().beats;
    for c in &a.cuts {
        assert!(grid.contains(c), "{c:?} is on the grid");
    }
    assert_eq!(a.cuts.len(), 9);
}

#[test]
fn bad_fills_are_refused_with_the_slot_named() {
    let pkg = builtin("builtin:photo-beat-slideshow");
    let mut p = Project::new("p");
    let song = media(&mut p, "song.wav", MediaKind::AudioOnly, 60.0, (0, 0), true);
    let pic = photo(&mut p, "a.jpg");
    let cases: Vec<(&str, SlotFill, &str)> = vec![
        ("nope", SlotFill::Text(vec![]), "no slot `nope`"),
        ("photos", SlotFill::Text(vec!["x".into()]), "takes photo media, not text"),
        ("title", SlotFill::Media(vec![MediaFill::new(pic)]), "takes text, not media"),
        ("title", SlotFill::Text(vec!["x".repeat(41)]), "longer than 40 characters"),
        ("photos", SlotFill::Media(vec![MediaFill::new(song)]), "has none"),
        ("music", SlotFill::Media(vec![MediaFill::new(pic)]), "has no sound"),
        ("photos", SlotFill::Media((0..9).map(|_| MediaFill::new(pic)).collect()), "at most 8"),
        ("photos", SlotFill::Media(vec![MediaFill::new(ItemId(987_654))]), "not in the project"),
        ("photos", SlotFill::Media(vec![MediaFill { focus: Some([2.0, f64::NAN]), ..MediaFill::new(pic) }]), "focus"),
    ];
    for (slot, fill, want) in cases {
        let mut fills = BTreeMap::new();
        fills.insert(slot.to_string(), fill);
        let before = p.clone();
        let e = apply(&mut p, &pkg, &ApplyOptions { fills, ..Default::default() }).unwrap_err().to_string();
        assert!(e.contains(want), "{slot}: {e}");
        assert_eq!(p, before, "a refused apply changes nothing");
    }
    // unknown variant
    let e = apply(&mut p, &pkg, &ApplyOptions { variant: Some("4:3".into()), ..Default::default() }).unwrap_err().to_string();
    assert!(e.contains("no variant `4:3`") && e.contains("16:9"), "{e}");
    // unfilled required slots keep their placeholders (a preview), unless required
    let a = apply(&mut p, &pkg, &ApplyOptions::default()).unwrap();
    assert!(a.placed.iter().filter(|x| x.slot == "photos").all(|x| x.how == "placeholder"));
    assert!(a.warnings.iter().any(|w| w.contains("photos")));
    assert!(apply(&mut p, &pkg, &ApplyOptions { require_all: true, ..Default::default() }).is_err());
}

#[test]
fn every_builtin_applies_in_every_variant_unfilled() {
    for pkg in builtin_templates() {
        for v in pkg.manifest.variants.iter().map(|v| v.id.as_str()) {
            let mut p = Project::new("p");
            let a =
                apply(&mut p, &pkg, &ApplyOptions { variant: Some(v.into()), ..Default::default() }).unwrap_or_else(|e| panic!("{} {v}: {e}", pkg.manifest.id));
            let q = seq(&p, a.sequence);
            q.check().unwrap();
            assert_eq!(q.duration(), pkg.manifest.duration, "{} {v}", pkg.manifest.id);
            assert_eq!(a.placed.len(), pkg.manifest.slots.iter().map(|s| s.placements.len()).sum::<usize>());
        }
    }
}

#[test]
fn applying_twice_never_shares_ids() {
    let pkg = builtin("builtin:channel-intro");
    let mut p = Project::new("p");
    let a = apply(&mut p, &pkg, &ApplyOptions::default()).unwrap();
    let b = apply(&mut p, &pkg, &ApplyOptions { variant: Some("9:16".into()), ..Default::default() }).unwrap();
    assert_ne!(a.sequence, b.sequence);
    let ids = |s: ItemId| -> Vec<ClipId> { seq(&p, s).all_tracks().flat_map(|t| t.items.iter().map(|c| c.id)).collect() };
    let (x, y) = (ids(a.sequence), ids(b.sequence));
    assert!(x.iter().all(|c| !y.contains(c)));
    assert!(a.items.iter().all(|i| !b.items.contains(i)));
}

#[test]
fn the_intro_frames_a_logo_inside_the_frame() {
    let pkg = builtin("builtin:channel-intro");
    let mut p = Project::new("p");
    let logo = media(&mut p, "logo.png", MediaKind::Still, 5.0, (1000, 500), false);
    let mut fills = BTreeMap::new();
    fills.insert("logo".into(), SlotFill::Media(vec![MediaFill::new(logo)]));
    fills.insert("name".into(), SlotFill::Text(vec!["Robot Lab".into()]));
    fills.insert("tagline".into(), SlotFill::Text(vec!["Robots, weekly".into()]));
    let a = apply(&mut p, &pkg, &ApplyOptions { fills, ..Default::default() }).unwrap();
    let q = seq(&p, a.sequence);
    let l = clip(q, a.placed.iter().find(|x| x.slot == "logo").unwrap().clips[0]);
    // fit: a 2:1 logo fits by its width (1920/1000), at the template's 40 % after the pop
    let k = (1920.0f64 / 1000.0).min(1080.0 / 500.0);
    let keys: Vec<f64> = l.effect("motion").unwrap().param("scale").unwrap().keyframes.iter().map(|k| k.value.as_f64().unwrap()).collect();
    assert!((keys.last().unwrap() - 40.0 * k).abs() < 1e-6, "{keys:?}");
    // both text slots share one graphic clip, a layer each
    let texts: Vec<String> = a
        .placed
        .iter()
        .filter(|x| x.slot == "name" || x.slot == "tagline")
        .map(|x| x.clips[0])
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .flat_map(|c| {
            clip(q, c)
                .effects
                .iter()
                .filter(|e| e.effect == "graphic_text")
                .filter_map(|e| match &e.param("text")?.value {
                    filmcraft_project::ParamValue::Text(t) => Some(t.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(texts, ["Robot Lab", "Robots, weekly"]);
    assert_eq!(q.duration(), Tick(5 * S));
}

#[test]
fn save_turns_flagged_clips_back_into_slots() {
    // apply the intro (two text slots share one graphic) and a looped countdown, then save both
    let mut p = Project::new("p");
    let intro = apply(&mut p, &builtin("builtin:channel-intro"), &ApplyOptions::default()).unwrap();
    let opts = crate::save::SaveOptions { name: "My Intro".into(), author: "me".into(), license: "CC0-1.0".into(), ..Default::default() };
    let saved = crate::save::save(&p, intro.sequence, &opts, None).unwrap();
    let m = &saved.package.manifest;
    assert_eq!(m.id, "user:my-intro");
    let name = m.slot("name").unwrap();
    let tag = m.slot("tagline").unwrap();
    assert_eq!(name.placements[0].clip, tag.placements[0].clip, "one graphic");
    assert_ne!(name.placements[0].layer, tag.placements[0].layer, "a layer each");
    assert!(m.slot("logo").is_some() && m.slot("music").is_some() && m.slot("background").is_some());
    assert_eq!(m.variants.iter().map(|v| v.id.as_str()).collect::<Vec<_>>(), ["16:9", "9:16"]);
    // and it applies like any template
    let back = apply(&mut p, &saved.package, &ApplyOptions { variant: Some("9:16".into()), ..Default::default() }).unwrap();
    assert_eq!(back.duration, Tick(5 * S));

    let mut fills = BTreeMap::new();
    let short = media(&mut p, "short.mp4", MediaKind::Movie, 1.0, (1920, 1080), false);
    fills.insert("items".into(), SlotFill::Media(vec![MediaFill::new(short)]));
    let top = apply(&mut p, &builtin("builtin:top-3-listicle"), &ApplyOptions { fills, ..Default::default() }).unwrap();
    assert!(top.placed.iter().any(|x| x.clips.len() > 1), "looped placements");
    let so = crate::save::SaveOptions { name: "Top".into(), license: "MIT".into(), ..Default::default() };
    let saved = crate::save::save(&p, top.sequence, &so, None).unwrap();
    let items = saved.package.manifest.slot("items").unwrap();
    assert_eq!(items.placements.len(), 3, "the loop pieces merged back into one placement each");
    let q = saved.package.sequence().unwrap();
    for pl in &items.placements {
        let c = q.find_item(ClipId(pl.clip)).unwrap().1;
        assert_eq!(c.duration, Tick(4 * S));
        // the user's file is gone, a generated placeholder stands in
        assert!(matches!(saved.package.fragment.item(c.item).unwrap().as_media().unwrap().media, MediaRef::Generator(_)));
    }
    let bytes = filmcraft_format::encode(&saved.package.fragment, false);
    assert!(!bytes.windows(9).any(|w| w == b"short.mp4"));
}
