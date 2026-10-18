//! The built-in starters, all of them: filled with generated media in every layout they have
//! they apply into exact tick durations, every placement is whole frames adding up to its
//! length, they survive a save and load of the project, and hostile captions, boxes and element
//! stand-ins are refused.

use std::collections::BTreeMap;

use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{DemoScene, Generator, MediaKind, MediaSource};
use filmcraft_project::{ItemId, ItemKind, Label, MarkerKind, MediaClip, MediaRef, ParamValue, Project, Sequence};
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick};

use crate::apply::{ApplyOptions, Cue, MediaFill, SlotFill, apply, set_captions, swap_item};
use crate::builtin::{SOUND_SECONDS, builtin_templates};
use crate::manifest::{Manifest, SlotKind};
use crate::{ELEMENT_METADATA, Package, TemplateError};

const S: i64 = TICKS_PER_SECOND;

fn builtin(id: &str) -> Package {
    builtin_templates().into_iter().find(|p| p.manifest.id == id).unwrap()
}

/// A generated media item: a demo scene `secs` long at `size`, with sound or not, of `kind`.
fn generated(p: &mut Project, name: &str, kind: MediaKind, secs: f64, size: (u32, u32), audio: bool) -> ItemId {
    let g = Generator::Demo(DemoScene::Forest);
    let src = GeneratorSource::new(g.clone(), size.0.max(16), size.1.max(16), FrameRate::FPS_30, Tick::from_seconds_f64(secs));
    let mut info = src.info().clone();
    info.kind = kind;
    info.name = name.into();
    if !audio {
        info.audio_streams.clear();
    }
    if kind == MediaKind::AudioOnly {
        info.video = None;
    }
    let clip = MediaClip {
        media: MediaRef::Generator(g),
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

/// Fills for every slot of `m`: generated media, texts at their longest allowed, three cues.
fn fill_all(p: &mut Project, m: &Manifest) -> BTreeMap<String, SlotFill> {
    let mut fills = BTreeMap::new();
    for s in &m.slots {
        let fill = match s.kind {
            SlotKind::Text => SlotFill::Text(
                (0..s.count)
                    .map(|i| {
                        let t = format!("{} {}", s.name, i + 1);
                        t.chars().take(s.constraints.max_chars.unwrap_or(40) as usize).collect()
                    })
                    .collect(),
            ),
            SlotKind::Captions => SlotFill::Captions(vec![
                Cue { start: Tick(S / 2), end: Tick(2 * S), text: "Hello there".into(), speaker: None },
                Cue { start: Tick(2 * S + S / 3), end: Tick(4 * S), text: "This is a caption".into(), speaker: Some("Me".into()) },
                Cue { start: Tick(4 * S), end: Tick(5 * S), text: "and another".into(), speaker: None },
            ]),
            SlotKind::Music => SlotFill::Media(vec![MediaFill::new(generated(p, &format!("{} song.wav", s.id), MediaKind::AudioOnly, 90.0, (0, 0), true))]),
            SlotKind::Logo => SlotFill::Media(vec![MediaFill::new(generated(p, &format!("{} logo.png", s.id), MediaKind::Still, 5.0, (800, 400), false))]),
            SlotKind::Photo => SlotFill::Media(
                (0..s.count).map(|i| MediaFill::new(generated(p, &format!("{} {i}.jpg", s.id), MediaKind::Still, 5.0, (4000, 3000), false))).collect(),
            ),
            SlotKind::Video => SlotFill::Media(
                (0..s.count)
                    .map(|i| {
                        // a mix of long and short clips, landscape and portrait
                        let (secs, size) = if i % 2 == 0 { (75.0, (1920, 1080)) } else { (1.3, (1080, 1920)) };
                        MediaFill::new(generated(p, &format!("{} {i}.mov", s.id), MediaKind::Movie, secs, size, true))
                    })
                    .collect(),
            ),
        };
        fills.insert(s.id.clone(), fill);
    }
    fills
}

fn on_frames(t: Tick, rate: FrameRate) -> bool {
    rate.snap_nearest(t) == t
}

#[test]
fn twenty_starters_in_every_layout_they_name() {
    let all = builtin_templates();
    assert!(all.len() >= 20, "{} starters", all.len());
    let mut categories = std::collections::BTreeSet::new();
    for pkg in &all {
        let m = &pkg.manifest;
        categories.insert(m.category.clone());
        assert!(crate::builtin::CATEGORIES.contains(&m.category.as_str()), "{}: {}", m.id, m.category);
        for v in ["16:9", "9:16"] {
            assert!(m.variant(Some(v)).is_some(), "{} has {v}", m.id);
        }
        assert!(m.poster.is_some_and(|p| p <= m.duration), "{} has a poster frame", m.id);
        assert!(!m.description.is_empty() && !m.tags.is_empty(), "{}", m.id);
        // every slot is reachable and explained
        for s in &m.slots {
            assert!(!s.name.is_empty(), "{} {}", m.id, s.id);
        }
        // the variants export at their own size
        for v in &m.variants {
            let want = match v.id.as_str() {
                "16:9" => (1920, 1080),
                "9:16" => (1080, 1920),
                "1:1" => (1080, 1080),
                "4:5" => (1080, 1350),
                other => panic!("{}: unexpected variant {other}", m.id),
            };
            assert_eq!((v.width, v.height), want, "{} {}", m.id, v.id);
            assert!(v.export_preset.is_some(), "{} {}", m.id, v.id);
        }
    }
    assert!(categories.len() >= 5, "{categories:?}");
    // the slot kinds a starter can have, all used somewhere
    for k in SlotKind::ALL {
        assert!(all.iter().any(|p| p.manifest.slots.iter().any(|s| s.kind == k)), "no starter has a {} slot", k.id());
    }
}

#[test]
fn every_starter_applies_with_generated_media_into_exact_ticks() {
    for pkg in builtin_templates() {
        let m = &pkg.manifest;
        for v in &m.variants {
            let mut p = Project::new("p");
            let fills = fill_all(&mut p, m);
            let a = apply(&mut p, &pkg, &ApplyOptions { variant: Some(v.id.clone()), fills, ..Default::default() })
                .unwrap_or_else(|e| panic!("{} {}: {e}", m.id, v.id));
            let q = p.sequence(a.sequence).unwrap();
            q.check().unwrap();
            let rate = q.settings.frame_rate;
            assert_eq!((q.settings.width, q.settings.height), (v.width, v.height), "{} {}", m.id, v.id);
            assert_eq!(a.duration, m.duration, "{} {}: exactly the template's length", m.id, v.id);
            assert_eq!(q.duration(), m.duration, "{} {}", m.id, v.id);
            for pc in &a.placed {
                let slot = m.slot(&pc.slot).unwrap();
                if slot.kind == SlotKind::Text {
                    continue;
                }
                assert!(pc.media.is_some(), "{} {}: {} placement {} is filled", m.id, v.id, pc.slot, pc.placement);
                let total = pc.clips.iter().map(|c| q.find_item(*c).unwrap().1.duration).fold(Tick::ZERO, |x, y| x + y);
                assert_eq!(total, pc.duration, "{} {}: {} placement {} adds up ({})", m.id, v.id, pc.slot, pc.placement, pc.how);
                for c in &pc.clips {
                    let it = q.find_item(*c).unwrap().1;
                    assert!(on_frames(it.start, rate) && on_frames(it.duration, rate), "{} {}: {} on frames", m.id, v.id, pc.slot);
                }
            }
            for t in q.all_tracks() {
                for c in &t.items {
                    assert!(on_frames(c.start, rate) && on_frames(c.duration, rate), "{} {}: clip “{}” on frames", m.id, v.id, c.name);
                    assert!(c.duration > Tick::ZERO);
                }
            }
            // text in its layers, cues on the caption track
            for s in m.slots.iter().filter(|s| s.kind == SlotKind::Text) {
                for (i, pl) in s.placements.iter().enumerate() {
                    let pc = a.placed.iter().find(|x| x.slot == s.id && x.placement == i).unwrap();
                    let it = q.find_item(pc.clips[0]).unwrap().1;
                    let layer = it
                        .effects
                        .iter()
                        .find(|e| e.effect == "graphic_text" && pl.layer.is_none_or(|l| filmcraft_project::gtemplate::layer_uid(e) == l))
                        .unwrap_or_else(|| panic!("{} {}: {} layer", m.id, v.id, s.id));
                    match &layer.param("text").unwrap().value {
                        ParamValue::Text(t) => assert!(t.starts_with(&s.name.chars().take(4).collect::<String>()), "{} {}: {t}", m.id, s.id),
                        other => panic!("{other:?}"),
                    }
                }
            }
            if m.captions_slot().is_some() {
                let caps = &q.caption_tracks.first().unwrap().captions;
                assert_eq!(caps.len(), 3, "{} {}", m.id, v.id);
                assert!(caps.iter().all(|c| on_frames(c.start, rate) && on_frames(c.duration, rate)));
                assert!(!a.captions_pending);
                assert!(!a.caption_clips.is_empty(), "{}: the speakers' sound is captioned", m.id);
            }
            assert_eq!(a.export_preset, v.export_preset, "{} {}", m.id, v.id);
            // pictures in a box are cropped to it, and stay inside the frame
            for s in m.slots.iter().filter(|s| s.kind.is_visual()) {
                for (i, pl) in s.placements.iter().enumerate() {
                    let area = v.hints.iter().find(|h| h.clip == pl.clip).and_then(|h| h.area).or(pl.area);
                    let Some(bx) = area else { continue };
                    let pc = a.placed.iter().find(|x| x.slot == s.id && x.placement == i).unwrap();
                    let it = q.find_item(pc.clips[0]).unwrap().1;
                    let pos = it.effect("motion").unwrap().param("position").unwrap().value.as_vec2().unwrap();
                    let (cx, cy) = ((bx[0] + bx[2] / 2.0) * v.width as f64, (bx[1] + bx[3] / 2.0) * v.height as f64);
                    assert!((pos.x - cx).abs() < 1e-6 && (pos.y - cy).abs() < 1e-6, "{} {} {}: {pos:?} vs ({cx}, {cy})", m.id, v.id, s.id);
                }
            }
        }
    }
}

/// The same JSON, numbers within a few units in the last place (the project file's decimal
/// numbers read back to the nearest double, which may be the next one).
fn close(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use serde_json::Value as V;
    match (a, b) {
        (V::Number(x), V::Number(y)) => match (x.as_f64(), y.as_f64()) {
            (Some(x), Some(y)) => x == y || (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1.0),
            _ => x == y,
        },
        (V::Array(x), V::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(a, b)| close(a, b)),
        (V::Object(x), V::Object(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| close(v, w))),
        _ => a == b,
    }
}

#[test]
fn applied_starters_survive_a_save_and_load() {
    for pkg in builtin_templates() {
        let mut p = Project::new("p");
        let fills = fill_all(&mut p, &pkg.manifest);
        let v = pkg.manifest.variants.last().map(|v| v.id.clone());
        let a = apply(&mut p, &pkg, &ApplyOptions { variant: v, fills, ..Default::default() }).unwrap();
        let bytes = filmcraft_format::encode(&p, false);
        let back = filmcraft_format::decode(&bytes).unwrap_or_else(|e| panic!("{}: {e}", pkg.manifest.id)).project;
        // (as JSON: unset Motion anchors are NaN, which never equals itself)
        let json = |q: Option<&Sequence>| serde_json::to_string(&q.cloned()).unwrap();
        assert!(back.sequence(a.sequence).is_some());
        let (x, y) = (json(back.sequence(a.sequence)), json(p.sequence(a.sequence)));
        let parse = |s: &str| serde_json::from_str::<serde_json::Value>(s).unwrap();
        if !close(&parse(&x), &parse(&y)) {
            let at = x.bytes().zip(y.bytes()).position(|(a, b)| a != b).unwrap_or(0);
            let window = |s: &str| s.get(at.saturating_sub(200)..(at + 200).min(s.len())).unwrap_or("").to_string();
            panic!("{}: loaded back differs at {at}:\n{}\n---\n{}", pkg.manifest.id, window(&x), window(&y));
        }
        assert_eq!(back.items.len(), p.items.len());
        // and the template's own package too (its manifest and sequence file)
        let files: BTreeMap<String, Vec<u8>> = pkg.files().into_iter().collect();
        let again = Package::from_files(files).unwrap_or_else(|e| panic!("{}: {e}", pkg.manifest.id));
        assert_eq!(again.manifest, pkg.manifest);
        assert!(close(&parse(&json(again.sequence())), &parse(&json(pkg.sequence()))), "{}", pkg.manifest.id);
    }
}

#[test]
fn captions_slots_keep_preview_captions_until_a_speaker_is_filled() {
    let pkg = builtin("builtin:talking-head-captions");
    let cs = pkg.manifest.captions_slot().unwrap();
    assert_eq!(cs.captions.as_ref().unwrap().source, ["speaker"]);
    // unfilled: the preview's captions stay
    let mut p = Project::new("p");
    let a = apply(&mut p, &pkg, &ApplyOptions::default()).unwrap();
    let placeholder = pkg.sequence().unwrap().caption_tracks[0].captions.len();
    assert!(placeholder > 0);
    assert_eq!(p.sequence(a.sequence).unwrap().caption_tracks[0].captions.len(), placeholder);
    assert!(!a.captions_pending);
    // the speaker filled: the placeholders go, the engine is to transcribe the speaker's sound
    let talk = generated(&mut p, "talk.mov", MediaKind::Movie, 40.0, (1080, 1920), true);
    let mut fills = BTreeMap::new();
    fills.insert("speaker".to_string(), SlotFill::Media(vec![MediaFill::new(talk)]));
    let a = apply(&mut p, &pkg, &ApplyOptions { fills, ..Default::default() }).unwrap();
    let q = p.sequence(a.sequence).unwrap();
    assert!(a.captions_pending);
    assert!(q.caption_tracks[0].captions.is_empty());
    assert_eq!(a.caption_clips.len(), 1, "the speaker's sound");
    let audio = q.find_item(a.caption_clips[0]).unwrap().1;
    assert_eq!((audio.item, audio.duration), (talk, Tick(30 * S)), "its first 30 s");
    assert_eq!(q.caption_tracks[0].style, pkg.manifest.caption_style.clone().unwrap(), "in the template's caption style");
    // the transcript's cues go on that track, snapped to frames
    let cues = vec![
        Cue { start: Tick(S / 7), end: Tick(S), text: "Hi".into(), speaker: None },
        Cue { start: Tick(S / 2), end: Tick(2 * S), text: "overlap".into(), speaker: None },
    ];
    let n = set_captions(&mut p, a.sequence, &cues, None).unwrap();
    assert_eq!(n, 2);
    let caps = &p.sequence(a.sequence).unwrap().caption_tracks[0].captions;
    let rate = FrameRate::FPS_30;
    assert_eq!(caps[0].start, rate.snap_nearest(Tick(S / 7)));
    assert!(caps[1].start >= caps[0].start + caps[0].duration, "a cue never covers the one before");
    // an empty transcript clears; captions given directly win over the placeholders
    assert_eq!(set_captions(&mut p, a.sequence, &[], None).unwrap(), 0);
    let mut fills = BTreeMap::new();
    fills.insert("captions".to_string(), SlotFill::Captions(cues.clone()));
    let a = apply(&mut p, &pkg, &ApplyOptions { fills, ..Default::default() }).unwrap();
    assert_eq!(p.sequence(a.sequence).unwrap().caption_tracks[0].captions.len(), 2);
}

#[test]
fn hostile_captions_are_refused() {
    let pkg = builtin("builtin:talking-head-captions");
    let mut p = Project::new("p");
    let cue = |a: i64, b: i64, t: &str| Cue { start: Tick(a), end: Tick(b), text: t.into(), speaker: None };
    let cases: Vec<(&str, SlotFill, &str)> = vec![
        ("captions", SlotFill::Captions(vec![cue(5 * S, 2 * S, "backwards")]), "end after it starts"),
        ("captions", SlotFill::Captions(vec![cue(-S, 2 * S, "early")]), "start at 0 or later"),
        ("captions", SlotFill::Captions(vec![cue(0, i64::MAX, "forever")]), "end after it starts"),
        ("captions", SlotFill::Captions(vec![cue(0, S, &"x".repeat(5000))]), "longer than"),
        ("captions", SlotFill::Captions((0..10_001).map(|i| cue(i, i + 1, "a")).collect()), "cues (at most"),
        ("captions", SlotFill::Text(vec!["words".into()]), "takes captions"),
        ("hook", SlotFill::Captions(vec![cue(0, S, "a")]), "not captions"),
    ];
    for (slot, fill, want) in cases {
        let mut fills = BTreeMap::new();
        fills.insert(slot.to_string(), fill);
        let before = p.clone();
        let e = apply(&mut p, &pkg, &ApplyOptions { fills, ..Default::default() }).unwrap_err().to_string();
        assert!(e.contains(want), "{slot}: {e}");
        assert_eq!(p, before);
    }
}

#[test]
fn chapter_markers_follow_the_chapter_titles() {
    let pkg = builtin("builtin:lower-third-chapter-kit");
    let mut p = Project::new("p");
    let mut fills = BTreeMap::new();
    fills.insert("chapters".to_string(), SlotFill::Text(vec!["Setup".into(), "Editing".into(), "Export".into()]));
    let a = apply(&mut p, &pkg, &ApplyOptions { fills, ..Default::default() }).unwrap();
    let q = p.sequence(a.sequence).unwrap();
    let chapters: Vec<(i64, &str)> = q.markers.iter().filter(|m| m.kind == MarkerKind::Chapter).map(|m| (m.start.0 / S, m.name.as_str())).collect();
    assert_eq!(chapters, [(0, "Intro"), (10, "Setup"), (20, "Editing"), (30, "Export")]);
    // unfilled, they keep the template's names
    let a = apply(&mut p, &pkg, &ApplyOptions::default()).unwrap();
    let names: Vec<String> = p.sequence(a.sequence).unwrap().markers.iter().map(|m| m.name.clone()).collect();
    assert_eq!(names, ["Intro", "Getting started", "The details", "Wrapping up"]);
}

#[test]
fn element_sounds_are_silent_stand_ins_the_engine_swaps() {
    let mut used = std::collections::BTreeSet::new();
    for pkg in builtin_templates() {
        let q = pkg.sequence().unwrap();
        for it in pkg.fragment.items.values() {
            let Some(e) = it.metadata.get(ELEMENT_METADATA) else { continue };
            let (_, secs) = SOUND_SECONDS.iter().find(|(id, _)| id == e).unwrap_or_else(|| panic!("{}: unknown sound {e}", pkg.manifest.id));
            used.insert(e.clone());
            let m = it.as_media().unwrap();
            assert!(matches!(m.media, MediaRef::Generator(Generator::Tone { .. })) && m.info.video.is_none(), "{}", pkg.manifest.id);
            for c in q.audio_tracks.iter().flat_map(|t| t.items.iter()).filter(|c| c.item == it.id) {
                assert!(c.duration <= Tick::from_seconds_f64(*secs), "{}: {e} fits the sound", pkg.manifest.id);
            }
        }
    }
    assert!(used.len() >= 8, "the starters use most of the Elements sounds: {used:?}");
    // swapping: every clip of the stand-in shows the sound, and the stand-in goes
    let pkg = builtin("builtin:countdown");
    let mut p = Project::new("p");
    let a = apply(&mut p, &pkg, &ApplyOptions::default()).unwrap();
    let boom = a.items.iter().copied().find(|i| p.item(*i).unwrap().metadata.get(ELEMENT_METADATA).map(String::as_str) == Some("sound.boom")).unwrap();
    let wav = generated(&mut p, "Boom.wav", MediaKind::AudioOnly, 3.0, (0, 0), true);
    assert_eq!(swap_item(&mut p, a.sequence, boom, wav).unwrap(), 1);
    assert!(p.item(boom).is_none());
    let q: &Sequence = p.sequence(a.sequence).unwrap();
    assert!(q.audio_tracks.iter().flat_map(|t| t.items.iter()).any(|c| c.item == wav && c.name == "Boom.wav"));
    assert!(swap_item(&mut p, a.sequence, boom, ItemId(9_999_999)).is_err());
    assert!(swap_item(&mut p, ItemId(9_999_999), wav, wav).is_err());
}

#[test]
fn hostile_element_stand_ins_are_refused() {
    let good = builtin("builtin:countdown");
    let stand_in = good.fragment.items.values().find(|i| i.metadata.contains_key(ELEMENT_METADATA)).unwrap().id;
    for (bad, want) in [("../../etc/passwd", "not a sound element id"), ("sound.", "not a sound element id"), ("sticker.star", "not a sound element id")] {
        let mut q = good.clone();
        q.fragment.item_mut(stand_in).unwrap().metadata.insert(ELEMENT_METADATA.into(), bad.into());
        let got: Vec<String> = q.check().0.iter().map(|i| i.to_string()).collect();
        assert!(got.iter().any(|g| g.contains(want)), "{bad}: {got:?}");
    }
    // a stand-in is generated and has no picture
    let mut q = good.clone();
    if let Some(ItemKind::Media(m)) = q.fragment.item_mut(stand_in).map(|i| &mut i.kind) {
        m.info.video =
            Some(filmcraft_media::VideoStreamInfo { width: 64, height: 64, ..GeneratorSource::demo(DemoScene::Aurora).info().video.clone().unwrap() });
    }
    assert!(q.check().0.iter().any(|i| i.message.contains("no picture")));
    // and a whole package carrying one is refused when read
    let files: BTreeMap<String, Vec<u8>> = q.files().into_iter().collect();
    assert!(matches!(Package::from_files(files), Err(TemplateError::Invalid(_))));
}

#[test]
fn a_box_crops_the_picture_to_its_cell() {
    let pkg = builtin("builtin:photo-collage");
    let mut p = Project::new("p");
    let photo = generated(&mut p, "wide.jpg", MediaKind::Still, 5.0, (4000, 1000), false);
    let mut fills = BTreeMap::new();
    let mut f = MediaFill::new(photo);
    f.focus = Some([0.1, 0.5]);
    fills.insert("photos".to_string(), SlotFill::Media(vec![f]));
    let a = apply(&mut p, &pkg, &ApplyOptions { variant: Some("1:1".into()), fills, ..Default::default() }).unwrap();
    let q = p.sequence(a.sequence).unwrap();
    let first = a.placed.iter().find(|x| x.slot == "photos" && x.placement == 0).unwrap();
    let it = q.find_item(first.clips[0]).unwrap().1;
    // the cell: 2 % in, 47 % wide, 45.5 % tall of a 1080 square
    let (bw, bh): (f64, f64) = (0.47 * 1080.0, 0.455 * 1080.0);
    let k = (bw / 4000.0).max(bh / 1000.0);
    let crop = it.effect("crop").expect("cropped to the cell");
    let pct = |n: &str| crop.param(n).unwrap().value.as_f64().unwrap();
    let shown = 4000.0 * k;
    // the focus is on the left: nothing cut there, the rest cut on the right
    assert!(pct("left") < 1e-9, "{}", pct("left"));
    assert!((pct("right") - (shown - bw) / shown * 100.0).abs() < 1e-6, "{}", pct("right"));
    assert!(pct("top") < 1e-9 && pct("bottom") < 1e-9);
    // scale: the template's pop to 100 is "exactly filling the cell"
    let keys: Vec<f64> = it.effect("motion").unwrap().param("scale").unwrap().keyframes.iter().map(|k| k.value.as_f64().unwrap()).collect();
    assert!((keys.last().unwrap() - 100.0 * k).abs() < 1e-6, "{keys:?}");
    // the anchor is the picture's point at the cell's centre
    let anchor = it.effect("motion").unwrap().param("anchor").unwrap().value.as_vec2().unwrap();
    assert!((anchor.x - bw / 2.0 / k).abs() < 1e-6, "{anchor:?}");
    // a picture that fits its box is not cropped
    let mut p = Project::new("p");
    let square = generated(&mut p, "sq.png", MediaKind::Still, 5.0, (900, 900), false);
    let mut fills = BTreeMap::new();
    fills.insert("logo".to_string(), SlotFill::Media(vec![MediaFill::new(square)]));
    let a = apply(&mut p, &builtin("builtin:event-recap"), &ApplyOptions { fills, ..Default::default() }).unwrap();
    let logo = a.placed.iter().find(|x| x.slot == "logo").unwrap();
    assert!(p.sequence(a.sequence).unwrap().find_item(logo.clips[0]).unwrap().1.effect("crop").is_none());
}

#[test]
fn hostile_boxes_and_posters_in_a_package_are_refused() {
    let good = builtin("builtin:photo-collage");
    for (path, v) in [
        ("box", serde_json::json!([0.0, 0.0, -1.0, 1.0])),
        ("box", serde_json::json!(["a", 0, 1, 1])),
        ("box", serde_json::json!([0.0, 0.0, 1.0])),
        ("box", serde_json::json!(null)),
    ] {
        let mut m = serde_json::to_value(&good.manifest).unwrap();
        m["slots"][0]["placements"][0][path] = v.clone();
        let mut files: BTreeMap<String, Vec<u8>> = good.files().into_iter().collect();
        files.insert("template.json".into(), serde_json::to_vec(&m).unwrap());
        let r = std::panic::catch_unwind(|| Package::from_files(files));
        let r = r.unwrap_or_else(|_| panic!("panicked on {v}"));
        if !v.is_null() {
            assert!(r.is_err(), "{v} refused");
        }
    }
    let mut m = serde_json::to_value(&good.manifest).unwrap();
    m["poster"] = serde_json::json!(i64::MAX);
    assert!(crate::manifest::from_value(&m).is_err());
}
