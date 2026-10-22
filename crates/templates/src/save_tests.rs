//! The authoring flow (A3): a sequence saved as a template keeps what the format carries —
//! captions slots, picture boxes, every layout, the poster, Elements sounds — strips the user's
//! media, bundles the files asked for with their attribution, and reads back the same.

use std::collections::BTreeMap;

use filmcraft_geom::Vec2;
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{DemoScene, Generator, MediaKind, MediaSource};
use filmcraft_project::{ClipId, ItemId, ItemKind, Label, MediaClip, MediaRef, Param, ParamValue, Project, Sequence, TrackItem};
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick};

use crate::apply::{ApplyOptions, MediaFill, SlotFill, apply, slot_marker_name};
use crate::builtin::{BUILTIN_IDS, builtin_templates};
use crate::manifest::{Framing, SlotKind};
use crate::save::{BundleFile, CaptionsChoice, CaptionsEdit, SaveOptions, SlotEdit, save, variant_size};
use crate::{ELEMENT_METADATA, Package};

const S: i64 = TICKS_PER_SECOND;

fn builtin(id: &str) -> Package {
    builtin_templates().into_iter().find(|p| p.manifest.id == id).unwrap()
}

fn opts(name: &str, base: Option<Package>) -> SaveOptions {
    SaveOptions { name: name.into(), author: "me".into(), license: "CC-BY-4.0".into(), base, ..Default::default() }
}

fn file_media(p: &mut Project, name: &str, size: (u32, u32), audio: bool) -> ItemId {
    let src = GeneratorSource::new(Generator::Demo(DemoScene::Forest), size.0, size.1, FrameRate::FPS_30, Tick::from_seconds_f64(20.0));
    let mut info = src.info().clone();
    info.kind = MediaKind::Movie;
    info.name = name.into();
    if !audio {
        info.audio_streams.clear();
    }
    let clip = MediaClip {
        media: MediaRef::File { path: format!("/Users/someone/Private/{name}") },
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

fn stand_ins(p: &Package) -> usize {
    p.fragment.items.values().filter(|i| i.metadata.contains_key(ELEMENT_METADATA)).count()
}

/// Every starter, applied (placeholders) and saved again with itself as the base, keeps its
/// slots, boxes, captions slot, layouts, poster, look and Elements sounds.
#[test]
fn every_starter_survives_apply_then_save() {
    for id in BUILTIN_IDS {
        let b = builtin(id);
        let mut p = Project::new("p");
        let a = apply(&mut p, &b, &ApplyOptions::default()).unwrap_or_else(|e| panic!("{id}: {e}"));
        let saved = save(&p, a.sequence, &opts("Again", Some(b.clone())), b.manifest.beat_grid.clone()).unwrap_or_else(|e| panic!("{id}: {e}"));
        let (m, bm) = (&saved.package.manifest, &b.manifest);
        let shape = |m: &crate::Manifest| -> Vec<(String, SlotKind, u32, u32, usize, Vec<Option<[f64; 4]>>)> {
            m.slots.iter().map(|s| (s.id.clone(), s.kind, s.count, s.min_count, s.placements.len(), s.placements.iter().map(|p| p.area).collect())).collect()
        };
        let mut want = shape(bm);
        let mut got = shape(m);
        want.sort_by(|a, b| a.0.cmp(&b.0));
        got.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(got, want, "{id}: slots, counts, placements and boxes");
        for s in &bm.slots {
            let g = m.slot(&s.id).unwrap();
            assert_eq!(
                (g.name.as_str(), g.prompt.as_str(), g.fit, g.beat_sync),
                (s.name.as_str(), s.prompt.as_str(), s.fit, s.beat_sync),
                "{id}: slot {}",
                s.id
            );
            assert_eq!(g.captions, s.captions, "{id}: captions of {}", s.id);
            assert_eq!(g.default, s.default, "{id}: default of {}", s.id);
            assert_eq!(g.ai, s.ai, "{id}: AI hints of {}", s.id);
        }
        let layouts = |m: &crate::Manifest| {
            let mut v = m.variants.iter().map(|v| (v.id.clone(), v.width, v.height, v.export_preset.clone())).collect::<Vec<_>>();
            v.sort_by(|a, b| a.0.cmp(&b.0));
            v
        };
        assert_eq!(layouts(m), layouts(bm), "{id}: layouts");
        // the layout it was applied in is the design now (its own, first)
        assert_eq!((m.default_variant.as_str(), m.variants[0].id.as_str()), (a.variant.as_str(), a.variant.as_str()), "{id}");
        assert!(m.variants[0].hints.is_empty(), "{id}: the design layout needs no hints");
        // the other layouts keep every hint, on the saved clips
        for v in m.variants.iter().skip(1) {
            let bv = bm.variants.iter().find(|x| x.id == v.id).unwrap();
            assert_eq!(v.hints.len(), bv.hints.len(), "{id} {}: hints", v.id);
            let q = saved.package.sequence().unwrap();
            assert!(v.hints.iter().all(|h| q.find_item(ClipId(h.clip)).is_some()), "{id} {}", v.id);
        }
        assert_eq!(m.poster, bm.poster, "{id}: poster");
        assert_eq!(m.look, bm.look, "{id}: look");
        assert_eq!(m.duration, bm.duration, "{id}: length");
        assert_eq!(m.caption_style, bm.caption_style, "{id}: caption style");
        assert_eq!(stand_ins(&saved.package), stand_ins(&b), "{id}: Elements sounds");
        // the text slots show the template's own text again
        for s in bm.slots.iter().filter(|s| s.kind == SlotKind::Text) {
            assert_eq!(saved.package.placeholder_texts(&s.id), b.placeholder_texts(&s.id), "{id}: text of {}", s.id);
        }
        // and it applies in every layout
        for v in &m.variants {
            apply(&mut p, &saved.package, &ApplyOptions { variant: Some(v.id.clone()), ..Default::default() }).unwrap_or_else(|e| panic!("{id} {}: {e}", v.id));
        }
    }
}

/// Saving, exporting as a ZIP and installing (reading the ZIP back) gives the same template.
#[test]
fn save_then_zip_round_trip_is_identical() {
    for id in ["builtin:podcast-two-speakers", "builtin:product-promo-15", "builtin:photo-collage"] {
        let b = builtin(id);
        let mut p = Project::new("p");
        let a = apply(&mut p, &b, &ApplyOptions::default()).unwrap();
        let saved = save(&p, a.sequence, &opts("Mine", Some(b.clone())), b.manifest.beat_grid.clone()).unwrap();
        let zip = saved.package.to_zip().unwrap();
        let back = Package::from_zip(&zip).unwrap();
        assert_eq!(back.manifest, saved.package.manifest, "{id}");
        assert_eq!(back.assets, saved.package.assets, "{id}");
        // (JSON floats read back within an ulp: serde_json's reader is not bit-exact)
        let as_json = |p: &Package| serde_json::from_slice::<serde_json::Value>(&filmcraft_format::encode(&p.fragment, true)).unwrap();
        assert!(same_json(&as_json(&back), &as_json(&saved.package)), "{id}: the sequence changed in the round trip");
        // a second trip reads the same again
        let again = Package::from_zip(&back.to_zip().unwrap()).unwrap();
        assert_eq!(again.manifest, back.manifest, "{id}");
        assert!(same_json(&as_json(&again), &as_json(&back)), "{id}");
    }
}

/// Equal JSON, numbers within a relative 1e-12.
fn same_json(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use serde_json::Value as V;
    match (a, b) {
        (V::Number(x), V::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap_or(f64::NAN), y.as_f64().unwrap_or(f64::NAN));
            x == y || (x - y).abs() <= 1e-12 * x.abs().max(y.abs()).max(1.0)
        }
        (V::Array(x), V::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(a, b)| same_json(a, b)),
        (V::Object(x), V::Object(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same_json(v, w))),
        _ => a == b,
    }
}

fn motion(c: &mut TrackItem) -> &mut filmcraft_project::EffectInstance {
    if !c.effects.iter().any(|e| e.effect == "motion") {
        c.effects.insert(0, filmcraft_project::find_effect("motion").unwrap().instance());
    }
    c.effects.iter_mut().find(|e| e.effect == "motion").unwrap()
}

fn scale(c: &TrackItem) -> f64 {
    c.effects.iter().find(|e| e.effect == "motion").and_then(|e| e.params.get("scale")).and_then(|p| p.value.as_f64()).unwrap_or(100.0)
}

/// A sequence of the user's own clips: a split screen made with a crop, a full-frame clip at
/// half size, a non-slot clip, a caption track.
fn authored(p: &mut Project) -> (ItemId, ClipId, ClipId, ClipId, ItemId) {
    let left = file_media(p, "left.mov", (1920, 1080), true);
    let full = file_media(p, "full.mov", (3840, 2160), false);
    let logo = file_media(p, "my-logo.mov", (400, 400), false);
    let settings = filmcraft_project::SequenceSettings { width: 1920, height: 1080, frame_rate: FrameRate::FPS_30, ..Default::default() };
    let s = p.new_sequence("Authored", settings, 1, 1, None);
    let mut ids = Vec::new();
    for (item, start, len) in [(left, 0, 4 * S), (full, 4 * S, 3 * S), (logo, 7 * S, 2 * S)] {
        let mut c = p
            .make_track_item(item, filmcraft_project::TrackKind::Video, Tick(start), filmcraft_time::TimeRange::new(Tick::ZERO, Tick(len)), FrameRate::FPS_30)
            .unwrap();
        c.duration = Tick(len);
        ids.push(c.id);
        p.sequence_mut(s).unwrap().video_tracks[0].items.push(c);
    }
    let q: &mut Sequence = p.sequence_mut(s).unwrap();
    // the left half of the frame: the picture at 100% (filling the frame), cropped to its left half
    {
        let c = &mut q.video_tracks[0].items[0];
        motion(c);
        let mut crop = filmcraft_project::find_effect("crop").unwrap().instance();
        crop.params.insert("right".into(), Param::new(ParamValue::Float(50.0)));
        c.effects.push(crop);
        c.markers.push(filmcraft_project::Marker {
            id: filmcraft_project::MarkerId(9001),
            start: Tick::ZERO,
            duration: Tick::ZERO,
            name: slot_marker_name(SlotKind::Video, "speaker", None, None),
            comment: String::new(),
            kind: filmcraft_project::MarkerKind::Comment,
            color: Label::Mango,
        });
        c.link = Some(77);
    }
    // a 4K clip at 50%: exactly filling the 1080p frame
    {
        let c = &mut q.video_tracks[0].items[1];
        let m = motion(c);
        m.params.insert("scale".into(), Param::new(ParamValue::Float(50.0)));
        m.params.insert("position".into(), Param::new(ParamValue::Vec2(Vec2::new(960.0, 540.0))));
        c.markers.push(filmcraft_project::Marker {
            id: filmcraft_project::MarkerId(9002),
            start: Tick::ZERO,
            duration: Tick::ZERO,
            name: slot_marker_name(SlotKind::Video, "broll", None, None),
            comment: String::new(),
            kind: filmcraft_project::MarkerKind::Comment,
            color: Label::Mango,
        });
    }
    // the user's words
    q.caption_tracks.push(filmcraft_project::CaptionTrack {
        id: filmcraft_project::TrackId(9100),
        name: "Captions".into(),
        format: filmcraft_project::CaptionFormat::Subtitle,
        language: "en".into(),
        style: Default::default(),
        enabled: true,
        locked: false,
        sync_lock: true,
        captions: vec![filmcraft_project::Caption {
            id: ClipId(9101),
            start: Tick(S),
            duration: Tick(2 * S),
            text: "my secret plans".into(),
            speaker: Some("Me".into()),
            cue_id: None,
            settings: String::new(),
            words: Vec::new(),
        }],
        vtt_blocks: Vec::new(),
    });
    (s, ids[0], ids[1], ids[2], logo)
}

#[test]
fn a_crop_becomes_a_box_and_motion_is_relative_to_the_slot() {
    let mut p = Project::new("p");
    let (s, _, full, logo_clip, _) = authored(&mut p);
    let mut o = opts("Split", None);
    o.captions = CaptionsChoice::Slot(CaptionsEdit { source: vec![], ..Default::default() });
    o.variants = vec!["9:16".into(), "1:1".into(), "4:5".into()];
    o.poster = Some(Tick(2 * S));
    o.slots.insert("speaker".into(), SlotEdit { name: Some("You".into()), prompt: Some("Talk to the camera".into()), ..Default::default() });
    let saved = save(&p, s, &o, None).unwrap();
    let m = &saved.package.manifest;
    let speaker = m.slot("speaker").unwrap();
    assert_eq!((speaker.name.as_str(), speaker.prompt.as_str()), ("You", "Talk to the camera"));
    assert_eq!(speaker.placements[0].area, Some([0.0, 0.0, 0.5, 1.0]), "the visible left half is the box");
    let broll = m.slot("broll").unwrap();
    assert_eq!(broll.placements[0].area, None);
    let q = saved.package.sequence().unwrap();
    // Fill in the half-frame box: the 16:9 picture covers 960×1080 at 1080/1080 = 1.0, the user
    // showed it at 1.0: 100 %
    assert!((scale(q.find_item(ClipId(speaker.placements[0].clip)).unwrap().1) - 100.0).abs() < 1e-6);
    // the crop is the apply's to set again
    let crop = q.find_item(ClipId(speaker.placements[0].clip)).unwrap().1.effects.iter().find(|e| e.effect == "crop").unwrap();
    assert_eq!(crop.params.get("right").unwrap().value.as_f64(), Some(0.0));
    // the 4K clip at 50 % filled the frame exactly: 100 % of the slot's framing
    assert!((scale(q.find_item(ClipId(broll.placements[0].clip)).unwrap().1) - 100.0).abs() < 1e-6, "{}", scale(q.find_item(full).unwrap().1));
    // a captions slot of the speaker (the only video slot with sound), with neutral preview text
    let cs = m.captions_slot().unwrap();
    assert_eq!(cs.captions.as_ref().unwrap().source, ["speaker"]);
    let caps = &q.caption_tracks[0].captions;
    assert_eq!((caps.len(), caps[0].start, caps[0].duration), (1, Tick(S), Tick(2 * S)));
    assert!(!caps[0].text.contains("secret") && caps[0].speaker.is_none());
    // layouts: its own, then those asked for, at its short side
    let ids: Vec<(&str, u32, u32)> = m.variants.iter().map(|v| (v.id.as_str(), v.width, v.height)).collect();
    assert_eq!(ids, [("16:9", 1920, 1080), ("9:16", 1080, 1920), ("1:1", 1080, 1080), ("4:5", 1080, 1350)]);
    assert_eq!(m.poster, Some(Tick(2 * S)));
    // the user's own non-slot clip and every private path are gone
    assert!(q.find_item(logo_clip).is_none());
    assert!(saved.warnings.iter().any(|w| w.contains("your own media")), "{:?}", saved.warnings);
    let bytes = filmcraft_format::encode(&saved.package.fragment, false);
    assert!(!bytes.windows(7).any(|w| w == b"Private"), "no private path in the package");
    assert!(!bytes.windows(6).any(|w| w == b"secret"), "no caption text of the user's");
    // it applies in every layout, and a filled box crops the new picture to it again
    let mut p2 = Project::new("p2");
    let wide = file_media(&mut p2, "wide.mov", (1920, 1080), true);
    let mut fills = BTreeMap::new();
    fills.insert("speaker".into(), SlotFill::Media(vec![MediaFill::new(wide)]));
    for v in &m.variants {
        let a = apply(&mut p2, &saved.package, &ApplyOptions { variant: Some(v.id.clone()), fills: fills.clone(), ..Default::default() }).unwrap();
        let q2 = p2.sequence(a.sequence).unwrap();
        let c = q2.video_tracks.iter().flat_map(|t| t.items.iter()).find(|c| c.item == wide).unwrap();
        assert!(c.effects.iter().any(|e| e.effect == "crop"), "{}", v.id);
    }
}

#[test]
fn bundled_files_carry_their_attribution() {
    let mut p = Project::new("p");
    let (s, _, _, logo_clip, logo) = authored(&mut p);
    let mut o = opts("Branded", None);
    o.bundle = vec![BundleFile {
        item: logo,
        bytes: b"my logo bytes".to_vec(),
        name: "My Logo (final).mov".into(),
        title: "My logo".into(),
        author: "me".into(),
        source: "original work".into(),
        license: "CC-BY-4.0".into(),
    }];
    let saved = save(&p, s, &o, None).unwrap();
    let m = &saved.package.manifest;
    assert_eq!(m.assets.len(), 1);
    let a = &m.assets[0];
    assert_eq!(a.path, "assets/My-Logo-final.mov");
    assert_eq!(a.sha256, crate::sha256_hex(b"my logo bytes"));
    assert_eq!((a.author.as_str(), a.source.as_str(), a.license.as_str()), ("me", "original work", "CC-BY-4.0"));
    assert_eq!(saved.package.assets.get(&a.path).map(Vec::as_slice), Some(&b"my logo bytes"[..]));
    let q = saved.package.sequence().unwrap();
    let c = q.find_item(logo_clip).expect("the bundled clip stays").1;
    let it = saved.package.fragment.item(c.item).unwrap().as_media().unwrap();
    assert_eq!(it.media, MediaRef::File { path: "assets/My-Logo-final.mov".into() });
    assert!(saved.package.check().is_empty());
    let back = Package::from_zip(&saved.package.to_zip().unwrap()).unwrap();
    assert_eq!(back.assets, saved.package.assets);
    // a bundle under a licence no template may carry is refused by the reader
    let mut bad = o.clone();
    bad.bundle[0].license = "CC-BY-NC-4.0".into();
    assert!(save(&p, s, &bad, None).is_err());
}

#[test]
fn authoring_edits_are_checked_and_layouts_sized() {
    assert_eq!(variant_size([4, 5], (1920, 1080)), (1080, 1350));
    assert_eq!(variant_size([9, 16], (1920, 1080)), (1080, 1920));
    assert_eq!(variant_size([16, 9], (1080, 1920)), (1920, 1080));
    assert_eq!(variant_size([1, 1], (720, 1280)), (720, 720));
    assert_eq!(variant_size([100, 1], (16, 16)), (1600, 16));
    let mut p = Project::new("p");
    let (s, ..) = authored(&mut p);
    // a bad layout is left out with a warning, a bad slot edit is refused by the reader
    let mut o = opts("X", None);
    o.variants = vec!["sideways".into()];
    let saved = save(&p, s, &o, None).unwrap();
    assert!(saved.warnings.iter().any(|w| w.contains("sideways")));
    let edit = serde_json::json!({"name": "Broll", "fit": {"short": "teleport"}});
    let e = crate::manifest::read_slot_edit(&edit, SlotKind::Video, "slots.broll").unwrap_err().to_string();
    assert!(e.contains("slots.broll.fit.short"), "{e}");
    assert!(crate::manifest::read_slot_edit(&serde_json::json!({"colour": 1}), SlotKind::Video, "slots.x").is_err());
    let ok = crate::manifest::read_slot_edit(&serde_json::json!({"minCount": 0, "fit": {"frame": "fit"}}), SlotKind::Video, "slots.x").unwrap();
    assert_eq!((ok.min_count, ok.fit.map(|f| f.frame)), (Some(0), Some(Framing::Fit)));
    // beat sync without a beat grid is dropped, with a warning
    let mut o = opts("Beat", None);
    o.slots.insert("broll".into(), SlotEdit { beat_sync: Some(true), ..Default::default() });
    let saved = save(&p, s, &o, None).unwrap();
    assert!(!saved.package.manifest.slot("broll").unwrap().beat_sync);
    // a built-in id is never taken
    let mut o = opts("Fake", None);
    o.id = "builtin:photo-beat-slideshow".into();
    assert_eq!(save(&p, s, &o, None).unwrap().package.manifest.id, "user:fake");
}
