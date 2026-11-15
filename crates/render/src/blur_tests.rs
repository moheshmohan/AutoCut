//! A large Gaussian Blur runs on a reduced working image ([`crate::effects_scale`]): the same
//! picture as the full-size blur to within a fraction of a percent, and only where nothing else
//! depends on the working size.

use std::sync::Arc;

use filmcraft_geom::Affine;
use filmcraft_media::MediaSource;
use filmcraft_media::still::StillSource;
use filmcraft_project::{ClipId, ItemId, ItemKind, Label, MediaClip, MediaRef, ParamValue, Project, SequenceSettings, TrackKind, find_effect};
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick, TimeRange};

use crate::{Image, RenderOptions, SourceMap, effects, effects_scale};

/// A 1080×1920 portrait picture with every kind of detail a blur has to smooth: a fine checkerboard,
/// hard edges, saturated bars and gradients.
fn picture() -> Vec<u8> {
    let (w, h) = (1080usize, 1920usize);
    let mut px = vec![255u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let o = (y * w + x) * 4;
            let rgb = if y < 480 {
                if (x / 3 + y / 3) % 2 == 0 { [250, 250, 250] } else { [5, 5, 5] }
            } else if y < 960 {
                [(x * 255 / w) as u8, (y * 7 % 256) as u8, 255 - (x * 255 / w) as u8]
            } else if y < 1440 {
                match (x / 90) % 4 {
                    0 => [255, 0, 0],
                    1 => [0, 255, 0],
                    2 => [0, 0, 255],
                    _ => [255, 255, 0],
                }
            } else if (x as i64 - 540).pow(2) + (y as i64 - 1680).pow(2) < 150 * 150 {
                [255, 255, 255]
            } else {
                [20, 30, 40]
            };
            px[o..o + 3].copy_from_slice(&rgb);
        }
    }
    px
}

/// A 1920×1080 sequence with the portrait picture filling it (Motion 177.78 %, as AutoCut's
/// blurred backgrounds), blurred by `blurriness`.
fn setup(blurriness: f64) -> (Project, ItemId, ClipId, SourceMap) {
    let mut p = Project::new("blur");
    let still = StillSource::from_rgba("portrait.png", 1080, 1920, picture(), false, 0);
    let info = still.info().clone();
    let item = p.add_item(
        "portrait.png",
        Label::Iris,
        ItemKind::Media(MediaClip {
            media: MediaRef::File { path: "portrait.png".into() },
            info,
            interpret: Default::default(),
            mark_in: None,
            mark_out: None,
            markers: vec![],
            offline: false,
            proxy: None,
            identity: None,
        }),
        None,
    );
    let r = FrameRate::FPS_30;
    let seq = p.new_sequence("s", SequenceSettings { width: 1920, height: 1080, frame_rate: r, ..Default::default() }, 1, 1, None);
    let mut ti = p.make_track_item(item, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, Tick(TICKS_PER_SECOND)), r).unwrap();
    for e in ti.effects.iter_mut() {
        filmcraft_project::resolve_auto_points(e, (1080, 1920), (1920, 1080));
    }
    ti.effect_mut("motion").unwrap().param_mut("scale").unwrap().value = ParamValue::Float(1920.0 / 1080.0 * 100.0);
    let mut blur = find_effect("gaussian_blur").unwrap().instance();
    blur.param_mut("blurriness").unwrap().value = ParamValue::Float(blurriness);
    ti.effects.insert(0, blur);
    let clip = ti.id;
    p.sequence_mut(seq).unwrap().video_tracks[0].items.push(ti);
    let mut m = SourceMap::default();
    m.0.insert(item, Arc::new(still));
    (p, seq, clip, m)
}

/// The clip's layer as rendering did before: effects on the full-size working image, then Motion.
fn full_size_layer(p: &Project, seq: ItemId, clip: ClipId, m: &SourceMap) -> Image {
    let s = p.sequence(seq).unwrap();
    let (_, item) = s.find_item(clip).unwrap();
    let t = Tick::ZERO;
    let src_size = crate::source_size(p, item.item).unwrap();
    let motion = crate::motion_matrix(s, item, src_size, crate::source_par(p, item.item), t);
    let mut layer = crate::base_layer(p, s, item, t, RenderOptions::default(), m, 1.0).unwrap();
    assert_eq!((layer.w, layer.h), (1080, 1920));
    let cx = effects::FxCtx {
        t,
        px_scale: 1.0,
        mask_scale: 1.0,
        seconds: 0.0,
        timecode: "",
        clip_name: "",
        project: Some(p),
        env: None,
        frame_rate: 30.0,
        working: s.settings.color.working,
    };
    effects::apply(&mut layer, &item.effects[0], &cx);
    layer.transformed(1920, 1080, &motion.then_apply(&Affine::IDENTITY))
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut se, mut n) = (0f64, 0usize);
    for (pa, pb) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
        for k in 0..3 {
            se += (f64::from(pa[k]) - f64::from(pb[k])).powi(2);
            n += 1;
        }
    }
    let mse = se / n.max(1) as f64;
    if mse == 0.0 { 99.0 } else { 10.0 * (255.0f64 * 255.0 / mse).log10() }
}

#[test]
fn a_large_blur_on_a_reduced_working_image_is_the_same_picture() {
    for blurriness in [30.0, 60.0, 120.0, 400.0] {
        let (p, seq, clip, m) = setup(blurriness);
        let s = p.sequence(seq).unwrap();
        let (_, item) = s.find_item(clip).unwrap();
        // σ = blurriness / 2 source pixels, kept at 6 working pixels or more (at most 8× smaller)
        let scale = effects_scale(item, Tick::ZERO, 1.0, RenderOptions::default());
        assert_eq!(scale, (6.0 / (blurriness * 0.5)).max(0.125) as f32, "blurriness {blurriness}");
        let reduced = crate::render_clip(&p, seq, clip, Tick::ZERO, RenderOptions::default(), &m).unwrap();
        let reference = full_size_layer(&p, seq, clip, &m);
        let q = psnr(&reduced.over_black_rgba8(), &reference.over_black_rgba8());
        assert!(q >= 40.0, "blurriness {blurriness}: {q:.1} dB");
        eprintln!("blurriness {blurriness}: working scale {scale}, {q:.1} dB against the full-size blur");
    }
}

#[test]
fn only_a_lone_large_blur_lowers_the_working_scale() {
    let opts = RenderOptions::default();
    let at = |p: &Project, seq: ItemId, clip: ClipId, want: f32, opts: RenderOptions| {
        let (_, item) = p.sequence(seq).unwrap().find_item(clip).unwrap();
        effects_scale(item, Tick::ZERO, want, opts)
    };
    let (mut p, seq, clip, _) = setup(60.0);
    assert_eq!(at(&p, seq, clip, 1.0, opts), 0.2);
    // never above what Motion asks for
    assert_eq!(at(&p, seq, clip, 0.1, opts), 0.1);
    // effects off (fast scrubbing): no blur, nothing to lower
    assert_eq!(at(&p, seq, clip, 1.0, RenderOptions { effects: false, ..opts }), 1.0);
    // a small blur keeps the full working image
    let (p2, seq2, clip2, _) = setup(12.0);
    assert_eq!(at(&p2, seq2, clip2, 1.0, opts), 1.0);
    // hostile values keep it too
    for v in [f64::NAN, f64::INFINITY, -60.0, 0.0] {
        let (p3, seq3, clip3, _) = setup(v);
        assert_eq!(at(&p3, seq3, clip3, 1.0, opts), 1.0, "blurriness {v}");
    }
    fn clip_of(p: &mut Project, seq: ItemId, clip: ClipId) -> &mut filmcraft_project::TrackItem {
        p.sequence_mut(seq).unwrap().find_item_mut(clip).unwrap().1
    }
    // a blur along one dimension keeps the detail of the other
    clip_of(&mut p, seq, clip).effects[0].param_mut("dimensions").unwrap().value = ParamValue::Choice(1);
    assert_eq!(at(&p, seq, clip, 1.0, opts), 1.0);
    clip_of(&mut p, seq, clip).effects[0].param_mut("dimensions").unwrap().value = ParamValue::Choice(0);
    // a disabled blur is no blur
    clip_of(&mut p, seq, clip).effects[0].enabled = false;
    assert_eq!(at(&p, seq, clip, 1.0, opts), 1.0);
    clip_of(&mut p, seq, clip).effects[0].enabled = true;
    // another effect after (or before) the blur sees the working image
    clip_of(&mut p, seq, clip).effects.push(find_effect("sharpen").unwrap().instance());
    assert_eq!(at(&p, seq, clip, 1.0, opts), 1.0);
    clip_of(&mut p, seq, clip).effects.pop();
    assert_eq!(at(&p, seq, clip, 1.0, opts), 0.2);
    // a mask on the blur, or on the clip's opacity, is drawn at the working size
    let path = filmcraft_project::mask::MaskPath::ellipse(filmcraft_geom::Vec2::new(540.0, 960.0), filmcraft_geom::Vec2::new(300.0, 300.0));
    let mask = filmcraft_project::mask::Mask::new("Mask (1)", path);
    clip_of(&mut p, seq, clip).effects[0].masks.push(mask.clone());
    assert_eq!(at(&p, seq, clip, 1.0, opts), 1.0);
    clip_of(&mut p, seq, clip).effects[0].masks.clear();
    clip_of(&mut p, seq, clip).effect_mut("opacity").unwrap().masks.push(mask);
    assert_eq!(at(&p, seq, clip, 1.0, opts), 1.0);
}
