//! Regressions from the 11 November full review. All files are generated under test temp dirs.
use crate::{EngineError, Session};
use filmcraft_project::{AudioType, EssentialSound, ItemId, ItemKind, Label};
use filmcraft_time::{Tick, TimeRange};
use serde_json::json;

fn portrait() -> (Session, ItemId, ItemId, std::path::PathBuf) {
    let dir = crate::media_test_util::tmp_dir("review-portrait");
    let image = dir.join("source.png");
    image::save_buffer(&image, &[140; 80 * 120 * 3], 80, 120, image::ColorType::Rgb8).unwrap();
    let mut s = Session::default();
    let parent = crate::commands::import_streamed(&mut s, image.to_str().unwrap(), None).unwrap();
    let child = s
        .edit("Make subclip", |p, _| {
            Ok(p.add_item(
                "Portrait subclip",
                Label::Iris,
                ItemKind::Subclip { parent, range: TimeRange::new(Tick::ZERO, Tick::from_seconds_f64(4.0)), restrict_trims: true },
                None,
            ))
        })
        .unwrap();
    s.execute("file.newSequence", json!({"width":80,"height":120,"fps":30,"video":1,"audio":1})).unwrap();
    (s, parent, child, image)
}

#[test]
fn review_stem_assignment_preserves_processing_and_mutes() {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s.edit_sequence("Keep dialogue treatment", |q, _, _| {
        let mut sound = EssentialSound::new(AudioType::Dialogue);
        sound.mute = true;
        sound.volume.on = true;
        sound.volume.level_db = -12.0;
        q.audio_tracks[0].items[0].essential = Some(sound);
        Ok(())
    })
    .unwrap();
    let sound = s.active_sequence().unwrap().audio_tracks[0].items[0].essential.clone();
    s.execute("audio.configureStems", json!({"roles":{"A1":"narration"}})).unwrap();
    assert_eq!(s.active_sequence().unwrap().audio_tracks[0].items[0].essential, sound);
}

#[test]
fn review_subclip_placement_uses_the_parent_picture_size() {
    let (mut s, parent, child, _) = portrait();
    let place = |s: &mut Session, item: ItemId| {
        s.execute("timeline.place", json!({"item":item.0,"track":"V1","time":0,"duration":Tick::from_seconds_f64(1.0).0,"audio":false})).unwrap();
        s.active_sequence().unwrap().video_tracks[0].items[0].effects.iter().find(|e| e.effect == "motion").unwrap().params["anchor"].value.clone()
    };
    let expected = place(&mut s, parent);
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(place(&mut s, child), expected);
}

#[test]
fn review_screen_protection_on_subclips_uses_source_pixels() {
    let (mut s, _, child, _) = portrait();
    s.execute("timeline.place", json!({"item":child.0,"track":"V1","time":0,"duration":Tick::from_seconds_f64(1.0).0,"audio":false})).unwrap();
    let id = s.active_sequence().unwrap().video_tracks[0].items[0].id;
    s.execute("clip.autoColor", json!({"clips":[id.0],"look":"punchy","frames":1,"protectScreens":{id.0.to_string():[[0.25,0.25,0.5,0.5]]}})).unwrap();
    let clip = &s.active_sequence().unwrap().video_tracks[0].items[0];
    let pass = clip.effects.iter().find(|e| crate::auto_color::is_pass(e)).unwrap();
    let path = pass.masks[0].path_at(Tick::ZERO);
    assert_eq!(path.vertices[0].p, filmcraft_geom::Vec2::new(20.0, 30.0));
}

#[test]
fn review_failed_edit_group_preserves_saved_state() {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s.saved_revision = s.revision;
    let before = s.project.clone();
    let history = s.history.undo.len();
    let result: crate::Result<()> = s.edit_group("Rejected edit", |s| {
        s.execute("file.newSequence", json!({"name":"Must disappear"}))?;
        Err(EngineError::Other("synthetic failure".into()))
    });
    assert!(result.is_err());
    assert_eq!(&*s.project, &*before);
    assert_eq!(s.history.undo.len(), history);
    assert!(!s.is_dirty(), "a rejected atomic edit must not make a saved project dirty");
}

#[test]
fn review_invalid_cache_sequence_never_falls_back_to_active_sequence() {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    for sequence in [json!("typo"), json!(-1), json!(false)] {
        assert!(s.execute("render.cachePlan", json!({"sequence":sequence})).is_err());
    }
}

#[test]
fn review_qa_report_cannot_replace_source_media() {
    let (mut s, parent, _, path) = portrait();
    let before = std::fs::read(&path).unwrap();
    s.execute("timeline.place", json!({"item":parent.0,"track":"V1","time":0,"duration":Tick::from_seconds_f64(1.0).0,"audio":false})).unwrap();
    assert!(s.execute("delivery.preflight", json!({"privacy":false,"narration":false,"mix":false,"report":path})).is_err());
    assert_eq!(std::fs::read(path).unwrap(), before);
}

#[test]
fn review_frame_export_cannot_replace_source_media() {
    let (mut s, parent, _, path) = portrait();
    let before = std::fs::read(&path).unwrap();
    s.execute("timeline.place", json!({"item":parent.0,"track":"V1","time":0,"duration":Tick::from_seconds_f64(1.0).0,"audio":false})).unwrap();
    assert!(s.execute("file.exportFrame", json!({"path":path,"seconds":0.5})).is_err());
    assert_eq!(std::fs::read(path).unwrap(), before);
}

#[test]
fn review_delivery_checks_the_exporters_resolved_filename() {
    let (mut s, parent, _, image) = portrait();
    s.execute("timeline.place", json!({"item":parent.0,"track":"V1","time":0,"duration":Tick::from_seconds_f64(1.0).0,"audio":false})).unwrap();
    let output = image.with_file_name("existing.mp4");
    std::fs::write(&output, b"previous approved master").unwrap();
    let r = s.execute("delivery.prepare", json!({"path":image.with_file_name("existing"),"qa":{"privacy":false,"narration":false,"mix":false},"cache":false}));
    assert!(r.is_err());
    assert_eq!(std::fs::read(output).unwrap(), b"previous approved master");
}

#[test]
fn review_export_cannot_replace_the_saved_project() {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    let path = crate::media_test_util::tmp_dir("review-project").join("keep.fcproj");
    s.execute("file.saveAs", json!({"path":path})).unwrap();
    let before = std::fs::read(&path).unwrap();
    let alias = path.with_file_name("project-alias.wav");
    std::fs::hard_link(&path, &alias).unwrap();
    let r = s.execute("file.exportMedia", json!({"path":alias,"format":"wav","range":"custom","startSeconds":0,"endSeconds":0.1,"wait":true}));
    assert!(r.is_err());
    assert_eq!(std::fs::read(path).unwrap(), before);
}

#[test]
fn review_delivery_finds_missing_media_inside_a_subclip() {
    let (mut s, _, child, path) = portrait();
    s.execute("timeline.place", json!({"item":child.0,"track":"V1","time":0,"duration":Tick::from_seconds_f64(1.0).0,"audio":false})).unwrap();
    std::fs::remove_file(path).unwrap();
    let r = s.execute("delivery.preflight", json!({"privacy":false,"narration":false,"mix":false})).unwrap();
    assert_eq!(r["passed"], false);
    assert!(r["checks"].as_array().unwrap().iter().any(|v| v["check"] == "sources" && v["state"] == "fail"));
}

#[test]
fn review_corrupt_cache_copy_never_publishes_partial_output() {
    let dir = crate::media_test_util::tmp_dir("review-cache-copy");
    let source = dir.join("cache");
    std::fs::write(&source, b"complete fixture").unwrap();
    let target = dir.join("fresh/subdir/video.mp4");
    let artifact = json!({"path":source,"bytes":16,"sha256":"0".repeat(64)});
    assert!(crate::render_cache::copy_artifact(&artifact, &target).is_err());
    assert!(!target.exists());
    let (hash, bytes) = crate::render_cache::digest_file(&source).unwrap();
    crate::render_cache::copy_artifact(&json!({"path":source,"bytes":bytes,"sha256":hash}), &target).unwrap();
    assert_eq!(std::fs::read(&target).unwrap(), b"complete fixture");
    assert!(crate::render_cache::copy_artifact(&json!({"path":source,"bytes":bytes,"sha256":hash}), &target).is_err());
    assert_eq!(std::fs::read(target).unwrap(), b"complete fixture");
}

#[test]
fn review_delivery_never_overwrites_an_existing_subtitle() {
    let (mut s, parent, _, image) = portrait();
    s.execute("timeline.place", json!({"item":parent.0,"track":"V1","time":0,"duration":Tick::from_seconds_f64(1.0).0,"audio":false})).unwrap();
    s.execute("captions.add", json!({"text":"Caption","seconds":0,"durationSeconds":1.0})).unwrap();
    let video = image.with_file_name("delivery.mp4");
    let subtitle = std::path::PathBuf::from(format!("{}.C1.srt", video.display()));
    std::fs::write(&subtitle, b"keep these notes").unwrap();
    let r = s.execute("delivery.prepare", json!({"path":video,"qa":{"privacy":false,"narration":false,"mix":false},"cache":false}));
    assert!(r.is_err());
    assert_eq!(std::fs::read(subtitle).unwrap(), b"keep these notes");
    assert!(!video.exists());
}

#[test]
fn review_delivery_refreshes_sources_changed_since_preview() {
    let (mut s, parent, _, path) = portrait();
    s.execute("timeline.place", json!({"item":parent.0,"track":"V1","time":0,"duration":Tick::from_seconds_f64(1.0).0,"audio":false})).unwrap();
    s.execute("file.exportFrame", json!({"path":path.with_file_name("warm.png"),"seconds":0.5})).unwrap();
    let red: Vec<u8> = (0..80 * 120).flat_map(|_| [230u8, 20, 20]).collect();
    image::save_buffer(&path, &red, 80, 120, image::ColorType::Rgb8).unwrap();
    s.redacts.detector = Some(std::sync::Arc::new(|pixels, _, _, _| {
        if pixels.first().is_some_and(|r| *r > 200) {
            vec![filmcraft_vision::Detection {
                kind: filmcraft_vision::Kind::Face,
                score: 0.99,
                bbox: [100.0, 200.0, 160.0, 160.0],
                points: Vec::new(),
                pose: false,
            }]
        } else {
            Vec::new()
        }
    }));
    let r = s.execute("delivery.preflight", json!({"narration":false,"mix":false})).unwrap();
    assert_eq!(r["passed"], false, "changed original was approved using its stale preview: {r}");
}

#[test]
fn review_unused_proxy_is_not_required_for_original_render_cache() {
    let (mut s, parent, _, path) = portrait();
    s.execute("timeline.place", json!({"item":parent.0,"track":"V1","time":0,"duration":Tick::from_seconds_f64(1.0).0,"audio":false})).unwrap();
    let before = s.execute("render.cachePlan", json!({})).unwrap();
    s.edit("Attach absent editing proxy", |p, _| {
        p.item_mut(parent).unwrap().as_media_mut().unwrap().proxy =
            Some(filmcraft_project::MediaRef::File { path: path.with_file_name("absent-proxy.mp4").to_string_lossy().into_owned() });
        Ok(())
    })
    .unwrap();
    assert_eq!(s.execute("render.cachePlan", json!({})).unwrap()["key"], before["key"]);
}

#[test]
fn review_still_overlays_and_disabled_photos_do_not_hide_frozen_video() {
    let (mut s, photo, _, image) = portrait();
    let duration = Tick::from_seconds_f64(3.0);
    s.execute("timeline.place", json!({"item":photo.0,"track":"V1","time":0,"duration":duration.0,"audio":false})).unwrap();
    let motion = |s: &mut Session| {
        let report = s.execute("delivery.preflight", json!({"privacy":false,"narration":false,"mix":false})).unwrap();
        report["checks"].as_array().unwrap().iter().find(|v| v["check"] == "motion").unwrap()["state"].clone()
    };
    assert_eq!(motion(&mut s), "pass", "an intentional photo hold is not a stalled video");
    s.execute("sequence.addTracks", json!({"video":1,"audio":0})).unwrap();
    let movie = image.with_file_name("frozen.mov");
    let generator = filmcraft_media::generators::GeneratorSource::new(
        filmcraft_media::Generator::ColorMatte { color: [0.4, 0.4, 0.4, 1.0] },
        80,
        120,
        filmcraft_time::FrameRate::FPS_30,
        duration,
    );
    crate::media_test_util::make_movie_from(&movie, std::sync::Arc::new(generator));
    let video = crate::commands::import_streamed(&mut s, movie.to_str().unwrap(), None).unwrap();
    s.execute("timeline.place", json!({"item":video.0,"track":"V2","time":0,"duration":duration.0,"audio":false})).unwrap();
    s.edit_sequence("Disable photo", |q, _, _| {
        q.video_tracks[0].enabled = false;
        Ok(())
    })
    .unwrap();
    assert_eq!(motion(&mut s), "warning", "a disabled photo must not exempt the active video");
    s.edit_sequence("Show static background", |q, _, _| {
        q.video_tracks[0].enabled = true;
        Ok(())
    })
    .unwrap();
    assert_eq!(motion(&mut s), "warning", "a static background must not exempt the video above it");
}
