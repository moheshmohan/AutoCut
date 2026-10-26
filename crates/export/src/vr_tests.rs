//! Video is VR: 360° exports carry Spherical Video metadata (V1 and V2), as asked or as a 360°
//! sequence implies.

use super::*;
use filmcraft_isobmff::TrackKind as Kind;
use filmcraft_isobmff::spherical::{Projection, StereoMode};
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{Generator, MediaSource, Spherical};
use filmcraft_project::{ItemKind, Label, MediaClip, MediaRef, SequenceSettings, TrackKind};
use filmcraft_render::SourceMap;

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("fc-export-vr-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name).to_string_lossy().to_string()
}

/// A second of a `w` × `h` picture in a sequence of its size; the media 360° when `vr`.
fn project(w: u32, h: u32, vr: bool) -> (Arc<Project>, ItemId, SourceMap) {
    let r = FrameRate::FPS_24;
    let mut p = Project::new("x");
    let g = GeneratorSource::new(Generator::ColorMatte { color: [0.2, 0.5, 0.8, 1.0] }, w, h, r, r.tick_of(24));
    let info = g.info().clone();
    let clip = MediaClip {
        media: MediaRef::Generator(g.generator.clone()),
        info: info.clone(),
        interpret: filmcraft_project::Interpretation { vr: vr.then(Spherical::equirect), ..Default::default() },
        mark_in: None,
        mark_out: None,
        markers: vec![],
        offline: false,
        proxy: None,
        identity: None,
    };
    let item = p.add_item(&info.name, Label::Iris, ItemKind::Media(clip), None);
    let seq = p.new_sequence("s", SequenceSettings { width: w, height: h, frame_rate: r, ..Default::default() }, 1, 1, None);
    let v = p.make_track_item(item, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, r.tick_of(24)), r).unwrap();
    p.sequence_mut(seq).unwrap().video_tracks[0].items.push(v);
    let mut m = SourceMap::default();
    m.0.insert(item, Arc::new(g));
    (Arc::new(p), seq, m)
}

/// Export `project` as H.264 with `vr`; the video track's (stereo, `sv3d`, V1) metadata.
fn exported(
    name: &str,
    (p, seq, m): (Arc<Project>, ItemId, SourceMap),
    vr: Option<bool>,
) -> Result<(Option<StereoMode>, Option<filmcraft_isobmff::spherical::SphericalBox>, Option<filmcraft_isobmff::spherical::SphericalV1>)> {
    let s = ExportSettings { format: Format::H264, path: tmp(name), include_audio: false, vr, ..Default::default() };
    export(&p, seq, &s, &m, &Progress::default())?;
    let f = filmcraft_isobmff::open(std::fs::read(&s.path).unwrap()).unwrap();
    let t = &f.tracks[f.track_of_kind(Kind::Video).unwrap()];
    let v = t.entries[0].video.clone().unwrap();
    Ok((v.stereo, v.spherical, t.spherical_v1.clone()))
}

#[test]
fn a_360_export_says_so() {
    let (stereo, sv3d, v1) = exported("asked.mp4", project(320, 160, false), Some(true)).unwrap();
    assert_eq!(stereo, Some(StereoMode::Mono));
    let sv3d = sv3d.expect("sv3d");
    assert!(sv3d.source.starts_with("FilmCraft"), "{sv3d:?}");
    assert_eq!(sv3d.projection, Some(Projection::Equirect { top: 0, bottom: 0, left: 0, right: 0 }));
    let v1 = v1.expect("the V1 uuid box");
    assert!(v1.spherical && v1.stitched && v1.projection == "equirectangular", "{v1:?}");

    // unset: a sequence of 360° media at 2:1 is 360° video; a flat one is not
    assert!(exported("auto360.mp4", project(320, 160, true), None).unwrap().1.is_some());
    assert_eq!(exported("flat.mp4", project(320, 160, false), None).unwrap(), (None, None, None));
    // off when asked, even for a 360° sequence
    assert_eq!(exported("off.mp4", project(320, 160, true), Some(false)).unwrap(), (None, None, None));
    // a 360° picture is 2:1: asked of another shape, the export says why it cannot
    let err = exported("wide.mp4", project(320, 180, false), Some(true)).unwrap_err();
    assert!(err.to_string().contains("2:1"), "{err}");
    assert_eq!(exported("wide-auto.mp4", project(320, 180, true), None).unwrap(), (None, None, None));
}

#[test]
fn video_is_vr_only_at_two_to_one() {
    assert!(job::video_is_vr(Some(true), false, (3840, 1920)).unwrap());
    assert!(job::video_is_vr(Some(true), false, (3840, 1080)).is_err());
    assert!(!job::video_is_vr(Some(false), true, (3840, 1920)).unwrap());
    assert!(job::video_is_vr(None, true, (3842, 1920)).unwrap(), "an even-rounded frame is still 2:1");
    assert!(!job::video_is_vr(None, true, (1920, 1080)).unwrap());
    assert!(!job::video_is_vr(None, false, (3840, 1920)).unwrap());
}
