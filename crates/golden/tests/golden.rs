//! Golden-image tests: procedural projects rendered through the CPU compositor and compared with
//! committed reference PNGs (`goldens/*.png`, made by this test with `FILMCRAFT_BLESS=1`), plus
//! GPU-vs-CPU parity on the same scenes when a GPU adapter is available.
//!
//! Tolerances (8-bit sRGB levels, see `filmcraft_testkit::golden`):
//! - CPU vs golden: `Tolerance::RENDER` — PSNR ≥ 45 dB, max abs ≤ 12, 99th percentile ≤ 2.
//! - GPU vs CPU: 99th percentile of the per-pixel max channel difference ≤ 6 and mean < 1.5
//!   (antialiased edges and half-float textures differ slightly; the CPU is the reference).

use std::path::PathBuf;
use std::sync::Arc;

use filmcraft_color::{ColorInfo, ColorSpace, Gamut, Primaries, Transfer};
use filmcraft_geom::Vec2;
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{DemoScene, Generator, MediaSource, SharedSource};
use filmcraft_project::{
    ClipId, EffectInstance, ItemId, ItemKind, Label, Mask, MaskMode, MaskPath, MediaClip, MediaRef, ParamValue, Project, SequenceSettings, TrackItem,
    TrackKind, Transition, TransitionId, find_effect,
};
use filmcraft_render::{RenderOptions, SourceMap, render_sequence};
use filmcraft_testkit::golden::{Rgba8, Tolerance, assert_golden, check_golden, diff};
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick, TimeRange};

const W: u32 = 320;
const H: u32 = 180;
const RATE: FrameRate = FrameRate::FPS_24;

/// A procedurally generated project and the frame to render.
struct Scene {
    project: Project,
    seq: ItemId,
    sources: SourceMap,
    frame: i64,
}

struct Builder {
    p: Project,
    map: SourceMap,
    seq: ItemId,
    next_transition: u64,
}

impl Builder {
    fn new(video_tracks: usize) -> Self {
        let mut p = Project::new("golden");
        let seq = p.new_sequence("golden", SequenceSettings { width: W, height: H, frame_rate: RATE, ..Default::default() }, video_tracks, 0, None);
        Builder { p, map: SourceMap::default(), seq, next_transition: 1 }
    }

    /// A generated media item (640×360, 24 fps, 10 s).
    fn media(&mut self, g: Generator) -> ItemId {
        let src = GeneratorSource::new(g, 640, 360, RATE, Tick(10 * TICKS_PER_SECOND));
        let info = src.info().clone();
        let id = self.p.add_item(
            &info.name.clone(),
            Label::Iris,
            ItemKind::Media(MediaClip {
                media: MediaRef::Generator(src.generator.clone()),
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
        self.map.0.insert(id, Arc::new(src) as SharedSource);
        id
    }

    fn demo(&mut self, scene: DemoScene) -> ItemId {
        self.media(Generator::Demo(scene))
    }

    /// Demo footage re-encoded as camera log / HDR (see [`Encoded`]), with an optional Interpret
    /// Footage colour-space override.
    fn encoded(&mut self, scene: DemoScene, target: ColorSpace, gain: f32, flag: Option<ColorInfo>, interpret: Option<ColorSpace>) -> ItemId {
        let id = self.demo(scene);
        let inner = self.map.0.remove(&id).unwrap();
        self.map.0.insert(id, Arc::new(Encoded { inner, target, gain, flag }) as SharedSource);
        if let ItemKind::Media(m) = &mut self.p.item_mut(id).unwrap().kind {
            m.interpret.color_space = interpret;
        }
        id
    }

    /// Place `item` on video track `track` (frames), scaled to the frame size.
    fn place(&mut self, track: usize, item: ItemId, start: i64, dur: i64) -> ClipId {
        let mut ti = self.p.make_track_item(item, TrackKind::Video, RATE.tick_of(start), TimeRange::new(Tick::ZERO, RATE.tick_of(dur)), RATE).unwrap();
        ti.scale_to_frame = true;
        let id = ti.id;
        let s = self.p.sequence_mut(self.seq).unwrap();
        s.video_tracks[track].items.push(ti);
        s.video_tracks[track].sort();
        id
    }

    fn clip(&mut self, id: ClipId) -> &mut TrackItem {
        self.p.sequence_mut(self.seq).unwrap().find_item_mut(id).unwrap().1
    }

    /// Set intrinsic-effect parameters (`motion`, `opacity`).
    fn fixed(&mut self, id: ClipId, effect: &str, params: &[(&str, ParamValue)]) {
        let e = self.clip(id).effect_mut(effect).unwrap();
        set(e, params);
    }

    /// Append a standard effect with parameters.
    fn effect(&mut self, id: ClipId, effect: &str, params: &[(&str, ParamValue)]) {
        let mut e = find_effect(effect).unwrap_or_else(|| panic!("no effect {effect}")).instance();
        set(&mut e, params);
        self.clip(id).effects.push(e);
    }

    /// A transition of `dur` frames centred on the cut at `cut` between `a` and `b` on `track`.
    fn transition(&mut self, track: usize, effect: &str, a: ClipId, b: ClipId, cut: i64, dur: i64) {
        let id = TransitionId(self.next_transition);
        self.next_transition += 1;
        let tr = Transition {
            id,
            effect: find_effect(effect).unwrap_or_else(|| panic!("no transition {effect}")).instance(),
            start: RATE.tick_of(cut - dur / 2),
            duration: RATE.tick_of(dur),
            from: Some(a),
            to: Some(b),
            align: Default::default(),
            reverse: false,
            sound: None,
        };
        self.p.sequence_mut(self.seq).unwrap().video_tracks[track].transitions.push(tr);
    }

    /// Add a mask to the clip's effect `effect` (feather/expansion in clip pixels).
    fn mask(&mut self, id: ClipId, effect: &str, path: MaskPath, feather: f64, f: impl FnOnce(&mut Mask)) {
        let mut m = Mask::new("Mask", path);
        m.feather.value = fl(feather);
        f(&mut m);
        self.clip(id).effect_mut(effect).unwrap_or_else(|| panic!("no effect {effect}")).masks.push(m);
    }

    fn at(self, frame: i64) -> Scene {
        Scene { project: self.p, seq: self.seq, sources: self.map, frame }
    }
}

fn set(e: &mut EffectInstance, params: &[(&str, ParamValue)]) {
    let id = e.effect.clone();
    for (k, v) in params {
        e.param_mut(k).unwrap_or_else(|| panic!("{id}: no param {k}")).value = v.clone();
    }
}

fn fl(v: f64) -> ParamValue {
    ParamValue::Float(v)
}

fn pt(x: f64, y: f64) -> ParamValue {
    ParamValue::Vec2(Vec2::new(x, y))
}

fn blend(name: &str) -> ParamValue {
    ParamValue::Choice(filmcraft_project::effect::BLEND_MODES.iter().position(|b| *b == name).unwrap_or_else(|| panic!("blend {name}")) as u32)
}

/// A source that re-encodes another source's (sRGB) frames into a camera log or HDR colour space:
/// linear BT.709 × `gain` → `target` gamut → `target` curve, stored as full-range RGBA8. `flag`
/// replaces the frame's colour metadata (e.g. PQ/BT.2020, so auto-detection kicks in).
struct Encoded {
    inner: SharedSource,
    target: ColorSpace,
    gain: f32,
    flag: Option<ColorInfo>,
}

impl MediaSource for Encoded {
    fn info(&self) -> &filmcraft_media::MediaInfo {
        self.inner.info()
    }
    fn video_frame(&self, req: filmcraft_media::FrameRequest) -> filmcraft_media::Result<Arc<filmcraft_frame::VideoFrame>> {
        let f = self.inner.video_frame(req)?;
        let rgba = f.to_rgba8();
        let m = filmcraft_color::spaces::to_f32(&filmcraft_color::spaces::gamut_matrix(Gamut::Bt709, self.target.gamut()));
        let curve = self.target.curve();
        let out: Vec<u8> = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| {
                let lin = [0, 1, 2].map(|k| filmcraft_color::srgb_to_linear(p[k] as f32 / 255.0) * self.gain);
                let c = filmcraft_color::spaces::apply3(&m, lin);
                let e = c
                    .map(|v| (filmcraft_color::transform::encode_channel(v as f64, curve, filmcraft_color::Range::Full).clamp(0.0, 1.0) * 255.0).round() as u8);
                [e[0], e[1], e[2], p[3]]
            })
            .collect();
        let mut nf = filmcraft_frame::VideoFrame::rgba8(f.width, f.height, out);
        if let Some(c) = self.flag {
            nf.color = c;
        }
        Ok(Arc::new(nf))
    }
    fn audio(&self, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<filmcraft_frame::AudioBuffer> {
        self.inner.audio(start, frames, sample_rate)
    }
}

// ---- scenes ----

/// Motion (scale, rotation, position) and 70 % opacity over a full-frame background.
fn transform_opacity() -> Scene {
    let mut b = Builder::new(2);
    let bg = b.demo(DemoScene::OceanSunset);
    let fg = b.demo(DemoScene::Aurora);
    b.place(0, bg, 0, 48);
    let top = b.place(1, fg, 0, 48);
    b.fixed(top, "motion", &[("scale", fl(45.0)), ("rotation", fl(20.0)), ("position", pt(215.0, 70.0))]);
    b.fixed(top, "opacity", &[("opacity", fl(70.0))]);
    b.at(12)
}

/// Four quadrant layers over a background in Multiply, Screen, Overlay and Difference.
fn blend_modes() -> Scene {
    let mut b = Builder::new(5);
    let bg = b.demo(DemoScene::OceanSunset);
    let fg = b.demo(DemoScene::Dunes);
    b.place(0, bg, 0, 48);
    for (i, (mode, x, y)) in [("Multiply", 80.0, 45.0), ("Screen", 240.0, 45.0), ("Overlay", 80.0, 135.0), ("Difference", 240.0, 135.0)].into_iter().enumerate()
    {
        let c = b.place(i + 1, fg, 0, 48);
        b.fixed(c, "motion", &[("scale", fl(50.0)), ("position", pt(x, y))]);
        b.fixed(c, "opacity", &[("blend", blend(mode))]);
    }
    b.at(12)
}

/// Gaussian Blur on colour bars (sharp edges show the kernel).
fn gaussian_blur() -> Scene {
    let mut b = Builder::new(1);
    let bars = b.media(Generator::BarsAndTone);
    let c = b.place(0, bars, 0, 48);
    b.effect(c, "gaussian_blur", &[("blurriness", fl(12.0))]);
    b.at(12)
}

/// Lumetri Color basic correction: temperature, exposure, contrast, highlights/shadows, saturation.
fn lumetri_basic() -> Scene {
    let mut b = Builder::new(1);
    let bg = b.demo(DemoScene::OceanSunset);
    let c = b.place(0, bg, 0, 48);
    b.effect(
        c,
        "lumetri",
        &[
            ("temperature", fl(25.0)),
            ("exposure", fl(0.6)),
            ("contrast", fl(30.0)),
            ("highlights", fl(-30.0)),
            ("shadows", fl(25.0)),
            ("saturation", fl(135.0)),
        ],
    );
    b.at(12)
}

/// Crop (left/top/right/bottom) of a top layer revealing the background.
fn crop() -> Scene {
    let mut b = Builder::new(2);
    let bg = b.demo(DemoScene::OceanSunset);
    let bars = b.media(Generator::BarsAndTone);
    b.place(0, bg, 0, 48);
    let c = b.place(1, bars, 0, 48);
    b.effect(c, "crop", &[("left", fl(15.0)), ("top", fl(10.0)), ("right", fl(25.0)), ("bottom", fl(20.0))]);
    b.at(12)
}

/// Two clips with a 12-frame transition centred on the cut at frame 24, rendered at `frame`.
fn transition(effect: &str, frame: i64) -> Scene {
    let mut b = Builder::new(1);
    let a = b.demo(DemoScene::OceanSunset);
    let c = b.demo(DemoScene::Aurora);
    let ca = b.place(0, a, 0, 24);
    let cb = b.place(0, c, 24, 24);
    b.transition(0, effect, ca, cb, 24, 12);
    b.at(frame)
}

/// Timecode and Clip Name burn-ins (text engine: JetBrains Mono / Inter).
fn text_burnin() -> Scene {
    let mut b = Builder::new(1);
    let bg = b.demo(DemoScene::CityNight);
    let c = b.place(0, bg, 0, 48);
    b.effect(c, "timecode", &[("position", pt(160.0, 40.0)), ("size", fl(20.0))]);
    b.effect(c, "clip_name", &[("position", pt(160.0, 145.0)), ("size", fl(12.0))]);
    b.at(30)
}

/// A graphic clip on V2 over `bg` on V1, with `layers` (graphic layer effect instances).
fn graphic_scene(bg: DemoScene, layers: Vec<EffectInstance>) -> Scene {
    let mut b = Builder::new(2);
    let bgi = b.demo(bg);
    b.place(0, bgi, 0, 48);
    let g = b.p.add_item("Graphic", Label::Rose, ItemKind::Graphic { width: W, height: H, rate: RATE }, None);
    let c = b.place(1, g, 0, 48);
    for mut l in layers {
        filmcraft_project::resolve_auto_points(&mut l, (W, H), (W, H));
        b.clip(c).effects.push(l);
    }
    b.at(12)
}

/// A built-in graphics template (Lower Third – Slab) placed on a 320×180 sequence with its
/// editable properties overridden (name, slab and accent colours, hidden role line): checks the
/// template's scaling, pins (the slab widens with the longer name) and the overrides.
fn graphic_template() -> Scene {
    use filmcraft_project::gtemplate::{builtin_templates, layer_uid};
    let t = builtin_templates().into_iter().find(|t| t.id == "builtin:lower-third-slab").unwrap();
    let (mut layers, meta) = t.instantiate((W, H));
    let mut over = |control: &str, v: ParamValue| {
        let c = t.controls.iter().find(|c| c.id == control).unwrap();
        let l = layers.iter_mut().find(|l| layer_uid(l) == c.layer).unwrap();
        if c.param == "enabled" {
            l.enabled = v.as_bool().unwrap();
        } else {
            l.params.get_mut(&c.param).unwrap().value = v;
        }
    };
    over("name", ParamValue::Text("Golden Overrides".into()));
    over("slab_color", ParamValue::Color([0.75, 0.15, 0.18, 1.0]));
    over("accent_color", ParamValue::Color([0.2, 0.85, 0.4, 1.0]));
    over("show_role", ParamValue::Bool(false));
    let mut s = graphic_scene(DemoScene::Forest, layers);
    let seq = s.seq;
    let q = s.project.sequence_mut(seq).unwrap();
    q.video_tracks[1].items[0].graphic = Some(Box::new(meta));
    s
}

/// Per-character styles (a bold, bigger, coloured word and an underlined one) and a rounded box
/// pinned around the text on all four edges.
fn graphic_rich_text() -> Scene {
    use filmcraft_project::graphic::{new_shape_layer, new_text_layer};
    use filmcraft_project::graphic_design::{CharStyle, LayerExtra, Pin, PinTarget, StyleRun};
    let mut text = new_text_layer("Mixed STYLES in one layer", Vec2::new(28.0, 100.0), 16.0);
    let runs = vec![
        StyleRun { start: 6, end: 12, style: CharStyle { size: Some(26.0), faux_bold: Some(true), fill: Some([1.0, 0.8, 0.1, 1.0]), ..Default::default() } },
        StyleRun {
            start: 16,
            end: 19,
            style: CharStyle { underline: Some(true), font: Some("Noto Serif".into()), faux_italic: Some(true), ..Default::default() },
        },
    ];
    text.layer = Some(Box::new(LayerExtra { uid: 1, runs, ..Default::default() }));
    let mut bx = new_shape_layer(0, Vec2::new(150.0, 90.0), Vec2::new(10.0, 10.0), vec![]);
    set(&mut bx, &[("corner_radius", fl(6.0)), ("fill_color", ParamValue::Color([0.1, 0.12, 0.3, 1.0])), ("opacity", fl(80.0))]);
    let pin = Pin { to: PinTarget::Layer(1), left: true, top: true, right: true, bottom: true, offsets: [-10.0, -8.0, 10.0, 8.0] };
    bx.layer = Some(Box::new(LayerExtra { uid: 2, pin: Some(pin), ..Default::default() }));
    graphic_scene(DemoScene::Dunes, vec![bx, text])
}

/// A title: bold centred text with an outer stroke and a soft drop shadow, a lower-third bar with
/// rounded corners and a background-boxed caption line.
fn graphic_title() -> Scene {
    use filmcraft_project::graphic::{new_shape_layer, new_text_layer};
    let mut bar = new_shape_layer(0, Vec2::new(160.0, 150.0), Vec2::new(280.0, 34.0), vec![]);
    set(&mut bar, &[("corner_radius", fl(8.0)), ("fill_color", ParamValue::Color([0.12, 0.35, 0.8, 1.0])), ("opacity", fl(85.0))]);
    let mut title = new_text_layer("Night Drive", Vec2::new(160.0, 80.0), 44.0);
    set(
        &mut title,
        &[
            ("font_style", ParamValue::Text("Bold".into())),
            ("align", ParamValue::Choice(1)),
            ("stroke", ParamValue::Bool(true)),
            ("stroke_width", fl(2.5)),
            ("stroke_color", ParamValue::Color([0.05, 0.05, 0.1, 1.0])),
            ("shadow", ParamValue::Bool(true)),
            ("shadow_distance", fl(5.0)),
            ("shadow_blur", fl(8.0)),
            ("tracking", fl(40.0)),
        ],
    );
    let mut sub = new_text_layer("Directed by Nobody", Vec2::new(160.0, 156.0), 16.0);
    set(
        &mut sub,
        &[
            ("align", ParamValue::Choice(1)),
            ("font", ParamValue::Text("Noto Serif".into())),
            ("faux_italic", ParamValue::Bool(true)),
            ("caps", ParamValue::Choice(2)),
            ("fill_color", ParamValue::Color([1.0, 0.92, 0.6, 1.0])),
        ],
    );
    graphic_scene(DemoScene::CityNight, vec![bar, title, sub])
}

/// Word-timed animated captions (Bold): three words on a page, the second being spoken (yellow,
/// popped 1.1×, the first pushed aside), with a keyword in green, over footage; the same overlay
/// the preview and the export burn in.
fn captions_animated() -> Scene {
    use filmcraft_project::{AnimatedCaptions, Caption, CaptionPreset, CaptionStyle, CaptionTrack, CaptionWord, TrackId};
    let mut b = Builder::new(1);
    let bg = b.demo(DemoScene::CityNight);
    b.place(0, bg, 0, 48);
    let mut a = AnimatedCaptions::preset(CaptionPreset::Bold);
    a.keywords = vec!["count".into()];
    let mut t = CaptionTrack::new(TrackId(7001), "Captions".into(), filmcraft_project::CaptionFormat::Subtitle);
    t.style = CaptionStyle { animated: Some(a), anchor: filmcraft_project::CaptionAnchor::Middle, ..CaptionPreset::Bold.base_style() };
    let w = |text: &str, f0: i64, f1: i64| CaptionWord { text: text.into(), start: RATE.tick_of(f0), end: RATE.tick_of(f1) };
    t.captions.push(Caption {
        id: ClipId(7002),
        start: Tick::ZERO,
        duration: RATE.tick_of(48),
        text: "make it count".into(),
        speaker: None,
        cue_id: None,
        settings: String::new(),
        words: vec![w("make", 0, 8), w("it", 9, 14), w("count", 15, 30)],
    });
    b.p.sequence_mut(b.seq).unwrap().caption_tracks.push(t);
    // frame 12: "it" spoken (0.125 s in: popped and settled)
    b.at(12)
}

/// Text styling and animation: a neon outer glow, an inner shadow, text on an arc, a third
/// stroke, and a title popping in letter by letter, caught mid-animation.
fn graphic_text_effects() -> Scene {
    use filmcraft_project::graphic::new_text_layer;
    use filmcraft_project::graphic_design::LayerExtra;
    use filmcraft_project::text_anim::{AnimKind, Animator, TextAnimation};
    let mut neon = new_text_layer("NEON", Vec2::new(160.0, 62.0), 40.0);
    set(
        &mut neon,
        &[
            ("font_style", ParamValue::Text("Bold".into())),
            ("align", ParamValue::Choice(1)),
            ("fill_color", ParamValue::Color([0.85, 1.0, 1.0, 1.0])),
            ("glow", ParamValue::Bool(true)),
            ("glow_color", ParamValue::Color([0.1, 0.9, 1.0, 1.0])),
            ("glow_radius", fl(10.0)),
            ("glow_intensity", fl(140.0)),
        ],
    );
    let mut arc = new_text_layer("ON AN ARC", Vec2::new(160.0, 120.0), 26.0);
    set(
        &mut arc,
        &[
            ("font_style", ParamValue::Text("Bold".into())),
            ("align", ParamValue::Choice(1)),
            ("curve", fl(45.0)),
            ("fill_color", ParamValue::Color([1.0, 0.8, 0.2, 1.0])),
            ("inner_shadow", ParamValue::Bool(true)),
            ("inner_shadow_distance", fl(2.0)),
            ("inner_shadow_blur", fl(2.0)),
            ("stroke", ParamValue::Bool(true)),
            ("stroke_width", fl(1.5)),
            ("stroke2", ParamValue::Bool(true)),
            ("stroke2_width", fl(3.0)),
            ("stroke2_color", ParamValue::Color([1.0, 0.3, 0.4, 1.0])),
            ("stroke3", ParamValue::Bool(true)),
            ("stroke3_width", fl(5.0)),
        ],
    );
    let mut pop = new_text_layer("POP", Vec2::new(160.0, 168.0), 30.0);
    set(
        &mut pop,
        &[("font_style", ParamValue::Text("Bold".into())), ("align", ParamValue::Choice(1)), ("fill_color", ParamValue::Color([1.0, 0.85, 0.1, 1.0]))],
    );
    // Back Out by name: the scene shows the overshoot (Pop's own easing is quiet since PX5.2)
    let anim = TextAnimation {
        enter: Some(Animator { stagger: 0.1, easing: filmcraft_project::text_anim::curve::BACK_OUT, ..Animator::preset(AnimKind::Pop) }),
        ..Default::default()
    };
    pop.layer = Some(Box::new(LayerExtra { anim: Some(anim), ..Default::default() }));
    // frame 6 = 0.25 s: P settled-ish, O overshooting, the last P just starting
    let mut s = graphic_scene(DemoScene::CityNight, vec![neon, arc, pop]);
    s.frame = 6;
    s
}

/// Shapes: ellipse with centre stroke, rotated polygon with inner stroke, a pen path, rotated
/// boxed text with a background and 2 strokes.
fn graphic_shapes() -> Scene {
    use filmcraft_project::graphic::{new_shape_layer, new_text_layer};
    let mut ell = new_shape_layer(1, Vec2::new(70.0, 60.0), Vec2::new(100.0, 70.0), vec![]);
    set(
        &mut ell,
        &[
            ("stroke", ParamValue::Bool(true)),
            ("stroke_width", fl(6.0)),
            ("stroke_type", ParamValue::Choice(1)),
            ("stroke_color", ParamValue::Color([1.0, 1.0, 1.0, 1.0])),
        ],
    );
    let mut poly = new_shape_layer(2, Vec2::new(250.0, 60.0), Vec2::new(80.0, 80.0), vec![]);
    set(
        &mut poly,
        &[
            ("sides", fl(5.0)),
            ("rotation", fl(18.0)),
            ("fill_color", ParamValue::Color([0.2, 0.8, 0.4, 1.0])),
            ("stroke", ParamValue::Bool(true)),
            ("stroke_type", ParamValue::Choice(2)),
            ("stroke_width", fl(5.0)),
        ],
    );
    let path = new_shape_layer(3, Vec2::new(70.0, 140.0), Vec2::new(0.0, 0.0), vec![[-40.0, 20.0], [0.0, -25.0], [40.0, 20.0], [0.0, 5.0]]);
    let mut txt = new_text_layer("Rotated\nTwo lines", Vec2::new(215.0, 140.0), 20.0);
    set(
        &mut txt,
        &[
            ("rotation", fl(-12.0)),
            ("align", ParamValue::Choice(1)),
            ("background", ParamValue::Bool(true)),
            ("background_radius", fl(6.0)),
            ("background_size", fl(6.0)),
            ("stroke", ParamValue::Bool(true)),
            ("stroke_width", fl(1.5)),
            ("stroke2", ParamValue::Bool(true)),
            ("stroke2_width", fl(3.5)),
        ],
    );
    graphic_scene(DemoScene::Aurora, vec![ell, poly, path, txt])
}

/// Camera log → Rec. 709: Ocean Sunset encoded as S-Log3 / S-Gamut3.Cine one stop over, then
/// interpreted as such in a Rec. 709 sequence (decode, gamut compression, BT.2390 tone mapping).
fn log_to_rec709() -> Scene {
    let mut b = Builder::new(1);
    let src = b.encoded(DemoScene::OceanSunset, ColorSpace::SLog3SGamut3Cine, 2.0, None, Some(ColorSpace::SLog3SGamut3Cine));
    b.place(0, src, 0, 48);
    b.at(12)
}

/// HDR → SDR: City Night as Rec. 2100 PQ with highlights up to ~1200 cd/m² (×6 over reference
/// white), detected from the frame's metadata and tone mapped into a Rec. 709 sequence.
fn hdr_tone_map() -> Scene {
    let mut b = Builder::new(1);
    let pq = ColorInfo { transfer: Transfer::Pq, primaries: Primaries::Bt2020, ..ColorInfo::SRGB_FULL };
    let src = b.encoded(DemoScene::CityNight, ColorSpace::Rec2100Pq, 6.0, Some(pq), None);
    b.place(0, src, 0, 48);
    b.at(12)
}

/// HDR grading: Rec. 2100 PQ footage in a PQ sequence, graded by Lumetri in the PQ signal
/// (exposure, contrast over HDR White, HDR Specular pulling the speculars in), shown tone mapped.
fn lumetri_hdr_pq() -> Scene {
    let mut b = Builder::new(1);
    b.p.sequence_mut(b.seq).unwrap().settings.color =
        filmcraft_color::ColorPipeline { working: filmcraft_color::WorkingSpace::Rec2100Pq, ..filmcraft_color::ColorPipeline::REC709 };
    let pq = ColorInfo { transfer: Transfer::Pq, primaries: Primaries::Bt2020, ..ColorInfo::SRGB_FULL };
    let src = b.encoded(DemoScene::CityNight, ColorSpace::Rec2100Pq, 6.0, Some(pq), None);
    let c = b.place(0, src, 0, 48);
    b.effect(
        c,
        "lumetri",
        &[("exposure", fl(0.4)), ("contrast", fl(30.0)), ("hdr_white", fl(1000.0)), ("hdr_specular", fl(-60.0)), ("temperature", fl(-15.0))],
    );
    b.at(12)
}

/// Gaussian Blur limited to a feathered ellipse mask (media is 640×360 clip pixels).
fn mask_blur() -> Scene {
    let mut b = Builder::new(1);
    let bars = b.media(Generator::BarsAndTone);
    let c = b.place(0, bars, 0, 48);
    b.effect(c, "gaussian_blur", &[("blurriness", fl(14.0))]);
    b.mask(c, "gaussian_blur", MaskPath::ellipse(Vec2::new(320.0, 170.0), Vec2::new(210.0, 120.0)), 60.0, |_| {});
    b.at(12)
}

/// Black & White inside a hard 4-point polygon (expanded by 12 px) minus a feathered pen (Bézier)
/// mask: mask modes, expansion and smooth vertices.
fn mask_color() -> Scene {
    let mut b = Builder::new(1);
    let bg = b.demo(DemoScene::OceanSunset);
    let c = b.place(0, bg, 0, 48);
    b.effect(c, "black_white", &[]);
    let quad = MaskPath::polygon(&[Vec2::new(60.0, 40.0), Vec2::new(560.0, 70.0), Vec2::new(600.0, 320.0), Vec2::new(90.0, 300.0)]);
    b.mask(c, "black_white", quad, 0.0, |m| m.expansion.value = fl(12.0));
    let mut pen = MaskPath::polygon(&[Vec2::new(250.0, 110.0), Vec2::new(420.0, 160.0), Vec2::new(300.0, 270.0)]);
    pen.vertices[0] = filmcraft_project::MaskVertex::smooth(Vec2::new(250.0, 110.0), Vec2::new(60.0, -30.0));
    pen.vertices[2] = filmcraft_project::MaskVertex::smooth(Vec2::new(300.0, 270.0), Vec2::new(-80.0, -10.0));
    b.mask(c, "black_white", pen, 16.0, |m| m.mode = MaskMode::Subtract);
    b.at(12)
}

/// Mosaic outside an ellipse (inverted mask), 80 % mask opacity.
fn mask_inverted() -> Scene {
    let mut b = Builder::new(1);
    let bg = b.demo(DemoScene::Aurora);
    let c = b.place(0, bg, 0, 48);
    b.effect(c, "mosaic", &[("horizontal", fl(24.0)), ("vertical", fl(14.0))]);
    b.mask(c, "mosaic", MaskPath::ellipse(Vec2::new(300.0, 180.0), Vec2::new(150.0, 110.0)), 8.0, |m| {
        m.inverted = true;
        m.opacity.value = fl(80.0);
    });
    b.at(12)
}

/// Opacity mask: a feathered ellipse cut out of a rotated, scaled top layer over a background
/// (the mask is in clip space, so it follows Motion).
fn mask_opacity() -> Scene {
    let mut b = Builder::new(2);
    let bg = b.demo(DemoScene::OceanSunset);
    let fg = b.demo(DemoScene::CityNight);
    b.place(0, bg, 0, 48);
    let top = b.place(1, fg, 0, 48);
    b.fixed(top, "motion", &[("scale", fl(70.0)), ("rotation", fl(-10.0)), ("position", pt(170.0, 95.0))]);
    b.mask(top, "opacity", MaskPath::ellipse(Vec2::new(320.0, 180.0), Vec2::new(250.0, 140.0)), 90.0, |_| {});
    b.at(12)
}

/// M5.11 Distort: Corner Pin (a demo picture pinned into a quad) over Turbulent-Displaced bars.
fn vfx_distort() -> Scene {
    let mut b = Builder::new(2);
    let bars = b.media(Generator::BarsAndTone);
    let fg = b.demo(DemoScene::Dunes);
    let c = b.place(0, bars, 0, 48);
    b.effect(c, "turbulent_displace", &[("amount", fl(40.0)), ("size", fl(80.0)), ("complexity", fl(2.0))]);
    let top = b.place(1, fg, 0, 48);
    b.effect(
        top,
        "corner_pin",
        &[("upper_left", pt(120.0, 60.0)), ("upper_right", pt(560.0, 20.0)), ("lower_left", pt(60.0, 330.0)), ("lower_right", pt(520.0, 300.0))],
    );
    b.at(12)
}

/// M5.11 Keying: Ultra Key removing the bars' green over a demo background, with spill suppression.
fn vfx_ultra_key() -> Scene {
    let mut b = Builder::new(2);
    let bg = b.demo(DemoScene::OceanSunset);
    let bars = b.media(Generator::BarsAndTone);
    b.place(0, bg, 0, 48);
    let top = b.place(1, bars, 0, 48);
    b.effect(top, "ultra_key", &[("key_color", ParamValue::Color([0.0, 0.75, 0.0, 1.0]))]);
    b.at(12)
}

/// M5.11 Lights & Glows: Wonder Glow and Glint on a night city.
fn vfx_glows() -> Scene {
    let mut b = Builder::new(1);
    let city = b.demo(DemoScene::CityNight);
    let c = b.place(0, city, 0, 48);
    b.effect(c, "wonder_glow", &[("threshold", fl(55.0)), ("radius", fl(20.0))]);
    b.effect(c, "glint", &[("threshold", fl(80.0)), ("length", fl(40.0))]);
    b.at(12)
}

/// M5.11 Utility / Perspective: Spacer inset with rounded corners, an inside Stroke and a fading
/// Long Shadow, over a Gradient generator.
fn vfx_spacer_stroke_shadow() -> Scene {
    let mut b = Builder::new(2);
    let bg = b.media(Generator::BlackVideo);
    let fg = b.demo(DemoScene::Forest);
    let back = b.place(0, bg, 0, 48);
    b.effect(
        back,
        "gradient",
        &[
            ("start", pt(0.0, 0.0)),
            ("end", pt(640.0, 360.0)),
            ("start_color", ParamValue::Color([0.15, 0.2, 0.45, 1.0])),
            ("end_color", ParamValue::Color([0.9, 0.6, 0.3, 1.0])),
        ],
    );
    let top = b.place(1, fg, 0, 48);
    b.effect(top, "spacer", &[("left", fl(90.0)), ("radius", fl(30.0))]);
    b.effect(top, "stroke", &[("width", fl(6.0)), ("position", ParamValue::Choice(2))]);
    b.effect(top, "long_shadow", &[("length", fl(80.0)), ("opacity", fl(70.0))]);
    b.at(12)
}

/// M5.11 Immersive Video: VR Rotate Sphere (pan and tilt) on equirectangular-treated aurora.
fn vfx_vr_rotate() -> Scene {
    let mut b = Builder::new(1);
    let aur = b.demo(DemoScene::Aurora);
    let c = b.place(0, aur, 0, 48);
    b.effect(c, "vr_rotate_sphere", &[("pan", fl(90.0)), ("tilt", fl(25.0)), ("roll", fl(10.0))]);
    b.at(12)
}

/// M5.11 Stylize / Image Control: Rounded Crop, Color Emboss and Channel Mix on plasma.
fn vfx_stylize() -> Scene {
    let mut b = Builder::new(1);
    let pl = b.demo(DemoScene::Plasma);
    let c = b.place(0, pl, 0, 48);
    b.effect(c, "channel_mix", &[("rr", fl(0.0)), ("rb", fl(100.0)), ("bb", fl(0.0)), ("br", fl(100.0))]);
    b.effect(c, "color_emboss", &[("relief", fl(3.0)), ("contrast", fl(150.0))]);
    b.effect(c, "rounded_crop", &[("left", fl(8.0)), ("right", fl(8.0)), ("top", fl(8.0)), ("bottom", fl(8.0)), ("radius", fl(50.0)), ("border", fl(4.0))]);
    b.at(12)
}

/// (name, title, scene).
fn scenes() -> Vec<(&'static str, &'static str, fn() -> Scene)> {
    vec![
        ("transform_opacity", "Motion transform and opacity", transform_opacity),
        ("blend_modes", "Blend modes (Multiply, Screen, Overlay, Difference)", blend_modes),
        ("gaussian_blur", "Gaussian Blur effect", gaussian_blur),
        ("lumetri_basic", "Lumetri Color basic correction", lumetri_basic),
        ("crop", "Crop effect", crop),
        ("transition_cross_dissolve", "Cross Dissolve at 50%", || transition("cross_dissolve", 24)),
        ("transition_dip_to_black", "Dip to Black at 25%", || transition("dip_to_black", 21)),
        ("transition_wipe", "Wipe at 50%", || transition("wipe", 24)),
        ("transition_gl_mosaic", "GL Mosaic (GL Transitions, WGSL port) at 50%", || transition("gl_mosaic", 24)),
        ("text_burnin", "Timecode and Clip Name burn-in text", text_burnin),
        ("graphic_title", "Graphic clip: title text with stroke and shadow, lower-third bar", graphic_title),
        ("graphic_shapes", "Graphic clip: ellipse, polygon, path and rotated text with background", graphic_shapes),
        ("graphic_template", "Graphics template (Lower Third – Slab) with overridden properties", graphic_template),
        ("graphic_rich_text", "Per-character text styles and a box pinned around the text", graphic_rich_text),
        ("captions_animated", "Word-timed animated captions (Bold): the spoken word highlighted and popped, a keyword coloured", captions_animated),
        ("graphic_text_effects", "Text glow, inner shadow, arc, three strokes and a letter-by-letter pop mid-animation", graphic_text_effects),
        ("log_to_rec709", "Colour management: S-Log3/S-Gamut3.Cine footage to Rec. 709 (tone mapped)", log_to_rec709),
        ("hdr_tone_map", "Colour management: Rec. 2100 PQ footage tone mapped into a Rec. 709 sequence", hdr_tone_map),
        ("lumetri_hdr_pq", "Lumetri grading in a Rec. 2100 PQ sequence (HDR White, HDR Specular), tone mapped", lumetri_hdr_pq),
        ("mask_blur", "Masks: Gaussian Blur inside a feathered ellipse mask", mask_blur),
        ("mask_color", "Masks: Black & White inside an expanded polygon minus a feathered Bezier mask", mask_color),
        ("mask_inverted", "Masks: Mosaic outside an inverted ellipse mask at 80% opacity", mask_inverted),
        ("mask_opacity", "Masks: feathered ellipse opacity mask on a rotated top layer", mask_opacity),
        ("vfx_distort", "Corner Pin over Turbulent Displace", vfx_distort),
        ("vfx_ultra_key", "Ultra Key on colour bars over footage", vfx_ultra_key),
        ("vfx_glows", "Wonder Glow and Glint", vfx_glows),
        ("vfx_spacer_stroke_shadow", "Spacer, Stroke and Long Shadow over a Gradient", vfx_spacer_stroke_shadow),
        ("vfx_vr_rotate", "VR Rotate Sphere", vfx_vr_rotate),
        ("vfx_stylize", "Channel Mix, Color Emboss and Rounded Crop", vfx_stylize),
    ]
}

fn render_cpu(s: &Scene) -> Rgba8 {
    // captions burned in, as the export does (only the caption scenes have caption tracks)
    let img = render_sequence(&s.project, s.seq, RATE.tick_of(s.frame), RenderOptions { captions: true, ..RenderOptions::default() }, &s.sources);
    Rgba8::new(img.w as u32, img.h as u32, img.over_black_rgba8())
}

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens").join(format!("{name}.png"))
}

fn check(name: &str) {
    let (_, title, make) = scenes().into_iter().find(|(n, _, _)| *n == name).unwrap();
    let img = render_cpu(&make());
    assert_eq!((img.w, img.h), (W, H));
    let d = assert_golden(&golden_path(name), &img, Tolerance::RENDER, title, &format!("crates/golden/tests/golden.rs ({name})"));
    eprintln!("{name}: {d}");
}

macro_rules! goldens {
    ($($name:ident),* $(,)?) => {
        mod cpu {
            $(
                #[test]
                fn $name() {
                    super::check(stringify!($name));
                }
            )*
        }

        #[test]
        fn every_scene_has_a_test() {
            let tested = [$(stringify!($name)),*];
            for (n, _, _) in scenes() {
                assert!(tested.contains(&n), "scene {n} has no golden test");
            }
        }
    };
}

goldens!(
    transform_opacity,
    blend_modes,
    gaussian_blur,
    lumetri_basic,
    crop,
    transition_cross_dissolve,
    transition_dip_to_black,
    transition_wipe,
    transition_gl_mosaic,
    text_burnin,
    graphic_title,
    graphic_shapes,
    graphic_template,
    graphic_rich_text,
    captions_animated,
    graphic_text_effects,
    log_to_rec709,
    hdr_tone_map,
    lumetri_hdr_pq,
    mask_blur,
    mask_color,
    mask_inverted,
    mask_opacity,
    vfx_distort,
    vfx_ultra_key,
    vfx_glows,
    vfx_spacer_stroke_shadow,
    vfx_vr_rotate,
    vfx_stylize
);

/// Video Transitions ▸ GL Transitions: the Rust reference of every ported shader
/// (`filmcraft_render::transitions::shader`) at 30 % and 60 %, one row per transition, from two
/// rendered demo frames (OceanSunset → Aurora) at 48×27, on sheets of ten rows
/// (`gl_transitions_sheet_<n>`, numbered from 1, each under the golden size limit).
#[test]
fn gl_transitions_sheet() {
    use filmcraft_render::transitions::{apply, shader};
    const ROWS_PER_SHEET: usize = 10;
    let frame = |scene: DemoScene| {
        let mut b = Builder::new(1);
        let m = b.demo(scene);
        b.place(0, m, 0, 48);
        let s = b.at(12);
        render_sequence(&s.project, s.seq, RATE.tick_of(s.frame), RenderOptions { scale: 0.15, ..RenderOptions::default() }, &s.sources)
    };
    let (a, b) = (frame(DemoScene::OceanSunset), frame(DemoScene::Aurora));
    assert_eq!((a.w, a.h), (48, 27));
    let (cw, ch) = (a.w, a.h);
    let ps = [0.3f32, 0.6];
    let sheets: Vec<&[shader::ShaderDef]> = shader::shaders().chunks(ROWS_PER_SHEET).collect();
    let mut failures = Vec::new();
    for (i, rows) in sheets.iter().enumerate() {
        let (w, h) = (cw * ps.len(), ch * rows.len());
        let mut px = vec![0u8; w * h * 4];
        for (r, d) in rows.iter().enumerate() {
            let e = find_effect(d.id).unwrap().instance();
            for (c, &p) in ps.iter().enumerate() {
                let cell = apply(&e, &a, &b, p).over_black_rgba8();
                for y in 0..ch {
                    let dst = ((r * ch + y) * w + c * cw) * 4;
                    px[dst..dst + cw * 4].copy_from_slice(&cell[y * cw * 4..(y + 1) * cw * 4]);
                }
            }
        }
        let rgba = Rgba8::new(w as u32, h as u32, px);
        let name = format!("gl_transitions_sheet_{}", i + 1);
        let ids: Vec<&str> = rows.iter().map(|d| d.id).collect();
        let title = format!("GL Transitions (WGSL ports, Rust reference) at 30 % and 60 %: {}", ids.join(", "));
        match check_golden(&golden_path(&name), &rgba, Tolerance::RENDER, &title, &format!("crates/golden/tests/golden.rs ({name})")) {
            Ok(d) => eprintln!("{name}: {d}"),
            Err(e) => failures.push(e),
        }
    }
    // no sheet left over from a longer list
    let stale = golden_path(&format!("gl_transitions_sheet_{}", sheets.len() + 1));
    assert!(!stale.exists(), "{} is not drawn any more: remove it, its sidecar and its ATTRIBUTION.md row", stale.display());
    assert!(failures.is_empty(), "golden mismatch: {failures:#?}");
}

/// Video Transitions ▸ Luma Wipes and Blend Dissolves: every preset of the two transition engines
/// (`filmcraft_render::transitions::shader`, `luma_wipe.wgsl` and `blend_dissolve.wgsl` with their
/// Rust references) at 45 %, from the same rendered demo frames as the GL sheets (48×27), in a
/// grid of four by four presets per sheet (`transition_engines_sheet_<n>`, numbered from 1; luma
/// wipes, then blend dissolves, each in id order).
#[test]
fn transition_engines_sheet() {
    use filmcraft_project::transition_engines::{BLEND_PRESETS, LUMA_PRESETS};
    use filmcraft_render::transitions::apply;
    const PER_ROW: usize = 4;
    const PER_SHEET: usize = 16;
    let frame = |scene: DemoScene| {
        let mut b = Builder::new(1);
        let m = b.demo(scene);
        b.place(0, m, 0, 48);
        let s = b.at(12);
        render_sequence(&s.project, s.seq, RATE.tick_of(s.frame), RenderOptions { scale: 0.15, ..RenderOptions::default() }, &s.sources)
    };
    let (a, b) = (frame(DemoScene::OceanSunset), frame(DemoScene::Aurora));
    let (cw, ch) = (a.w, a.h);
    let ids: Vec<&str> = LUMA_PRESETS.iter().map(|p| p.id).chain(BLEND_PRESETS.iter().map(|p| p.id)).collect();
    let sheets: Vec<&[&str]> = ids.chunks(PER_SHEET).collect();
    let mut failures = Vec::new();
    for (i, cells) in sheets.iter().enumerate() {
        let rows = cells.len().div_ceil(PER_ROW);
        let (w, h) = (cw * PER_ROW, ch * rows);
        let mut px = vec![0u8; w * h * 4];
        for (k, id) in cells.iter().enumerate() {
            let (r, c) = (k / PER_ROW, k % PER_ROW);
            let cell = apply(&find_effect(id).unwrap().instance(), &a, &b, 0.45).over_black_rgba8();
            for y in 0..ch {
                let dst = ((r * ch + y) * w + c * cw) * 4;
                px[dst..dst + cw * 4].copy_from_slice(&cell[y * cw * 4..(y + 1) * cw * 4]);
            }
        }
        let rgba = Rgba8::new(w as u32, h as u32, px);
        let name = format!("transition_engines_sheet_{}", i + 1);
        let title = format!("Luma Wipes and Blend Dissolves (FilmCraft transition engines, Rust reference) at 45 %: {}", cells.join(", "));
        match check_golden(&golden_path(&name), &rgba, Tolerance::RENDER, &title, &format!("crates/golden/tests/golden.rs ({name})")) {
            Ok(d) => eprintln!("{name}: {d}"),
            Err(e) => failures.push(e),
        }
    }
    let stale = golden_path(&format!("transition_engines_sheet_{}", sheets.len() + 1));
    assert!(!stale.exists(), "{} is not drawn any more: remove it, its sidecar and its ATTRIBUTION.md row", stale.display());
    assert!(failures.is_empty(), "golden mismatch: {failures:#?}");
}

/// Roadmap C3 and C5: the flow transitions and the Content-Aware Wipe (their flow and seams
/// computed from the two frames), then the light & film, pattern and camera presets, at 45 %,
/// from the same rendered demo frames as the other transition sheets (48×27), four by four per
/// sheet (`transition_styles_sheet_<n>`, numbered from 1).
#[test]
fn transition_styles_sheet() {
    use filmcraft_project::flow_transitions::{CONTENT_AWARE_WIPE, FLOW_PRESETS};
    use filmcraft_project::transition_catalog::STYLE_PRESETS;
    use filmcraft_render::transitions::apply;
    const PER_ROW: usize = 4;
    const PER_SHEET: usize = 16;
    let frame = |scene: DemoScene| {
        let mut b = Builder::new(1);
        let m = b.demo(scene);
        b.place(0, m, 0, 48);
        let s = b.at(12);
        render_sequence(&s.project, s.seq, RATE.tick_of(s.frame), RenderOptions { scale: 0.15, ..RenderOptions::default() }, &s.sources)
    };
    let (a, b) = (frame(DemoScene::OceanSunset), frame(DemoScene::Aurora));
    let (cw, ch) = (a.w, a.h);
    let ids: Vec<&str> = FLOW_PRESETS.iter().map(|p| p.id).chain([CONTENT_AWARE_WIPE]).chain(STYLE_PRESETS.iter().map(|p| p.id)).collect();
    let sheets: Vec<&[&str]> = ids.chunks(PER_SHEET).collect();
    let mut failures = Vec::new();
    for (i, cells) in sheets.iter().enumerate() {
        let rows = cells.len().div_ceil(PER_ROW);
        let (w, h) = (cw * PER_ROW, ch * rows);
        let mut px = vec![0u8; w * h * 4];
        for (k, id) in cells.iter().enumerate() {
            let (r, c) = (k / PER_ROW, k % PER_ROW);
            let cell = apply(&find_effect(id).unwrap().instance(), &a, &b, 0.45).over_black_rgba8();
            for y in 0..ch {
                let dst = ((r * ch + y) * w + c * cw) * 4;
                px[dst..dst + cw * 4].copy_from_slice(&cell[y * cw * 4..(y + 1) * cw * 4]);
            }
        }
        let rgba = Rgba8::new(w as u32, h as u32, px);
        let name = format!("transition_styles_sheet_{}", i + 1);
        let title = format!(
            "Flow & Morph, Light & Film, Particles & Patterns and camera transitions (FilmCraft engines, Rust reference) at 45 %: {}",
            cells.join(", ")
        );
        match check_golden(&golden_path(&name), &rgba, Tolerance::RENDER, &title, &format!("crates/golden/tests/golden.rs ({name})")) {
            Ok(d) => eprintln!("{name}: {d}"),
            Err(e) => failures.push(e),
        }
    }
    let stale = golden_path(&format!("transition_styles_sheet_{}", sheets.len() + 1));
    assert!(!stale.exists(), "{} is not drawn any more: remove it, its sidecar and its ATTRIBUTION.md row", stale.display());
    assert!(failures.is_empty(), "golden mismatch: {failures:#?}");
}

/// Effects panel ▸ Lumetri Presets thumbnail grid: every built-in preset graded by our Lumetri on
/// the procedural preview picture (`filmcraft_render::lumetri_presets`).
#[test]
fn lumetri_presets_grid() {
    use filmcraft_render::lumetri_presets as lp;
    let img = lp::grid(&lp::presets(), 6, 96, 54);
    let rgba = Rgba8::new(img.w as u32, img.h as u32, img.over_black_rgba8());
    let name = "lumetri_presets_grid";
    let d = assert_golden(&golden_path(name), &rgba, Tolerance::RENDER, "Lumetri Presets thumbnail grid", &format!("crates/golden/tests/golden.rs ({name})"));
    eprintln!("{name}: {d}");
}

/// Sanity: the scenes are not trivially empty or identical to each other.
#[test]
fn scenes_are_distinct() {
    let imgs: Vec<(&str, Rgba8)> = scenes().into_iter().map(|(n, _, f)| (n, render_cpu(&f()))).collect();
    for (n, img) in &imgs {
        let lit = img.px.as_chunks::<4>().0.iter().filter(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 30).count();
        assert!(lit > img.px.len() / 4 / 4, "{n}: mostly black");
    }
    for i in 0..imgs.len() {
        for j in i + 1..imgs.len() {
            let d = diff(&imgs[i].1, &imgs[j].1).unwrap();
            assert!(d.psnr < 30.0, "{} and {} are near-identical ({d})", imgs[i].0, imgs[j].0);
        }
    }
}

// ---- GPU parity ----

/// A device for the GPU parity tests, or `None` (with a `SKIPPED` note) when the machine has no
/// usable adapter. `require_float_target` additionally demands a renderable `Rgba32Float`, which
/// only the effect-chain plan needs; main falls back to CPU effects on adapters without it.
fn device(context: &str, require_float_target: bool) -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let Some(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok() else {
        eprintln!("SKIPPED ({context}): no GPU adapter");
        return None;
    };
    let name = adapter.get_info().name;
    let renderable = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba32Float).allowed_usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT);
    if require_float_target && !renderable {
        eprintln!("SKIPPED ({context}): adapter {name} cannot render to Rgba32Float");
        return None;
    }
    match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())) {
        Ok(device) => Some(device),
        Err(_) => {
            eprintln!("SKIPPED ({context}): adapter {name} refused a device");
            None
        }
    }
}

#[test]
fn gpu_matches_cpu_on_golden_scenes() {
    let Some((dev, q)) = device("gpu parity", false) else {
        return;
    };
    let mut c = filmcraft_gpu::GpuCompositor::new(&dev, &q);
    let mut failures = Vec::new();
    for (name, _, make) in scenes() {
        let s = make();
        let t = RATE.tick_of(s.frame);
        let cpu = render_cpu(&s);
        let plan = filmcraft_render::plan::plan_frame(&s.project, s.seq, t, RenderOptions { captions: true, ..RenderOptions::default() }, &s.sources);
        let kind = if matches!(plan, filmcraft_render::plan::FramePlan::Layers { .. }) { "layers" } else { "cpu image" };
        c.composite(&plan);
        let (gw, gh, mut px) = c.read_output().expect("GPU readback");
        px.as_chunks_mut::<4>().0.iter_mut().for_each(|p| p[3] = 255);
        let gpu = Rgba8::new(gw, gh, px);
        let d = diff(&cpu, &gpu).unwrap();
        // mean over RGB only (alpha is equal by construction)
        let mean_rgb = d.mean_abs * 4.0 / 3.0;
        eprintln!("{name} ({kind}): GPU vs CPU {d}");
        if d.p99 > 6 || mean_rgb >= 1.5 {
            let dir = filmcraft_testkit::golden::failures_dir();
            let _ = filmcraft_testkit::golden::write_png(&dir.join(format!("{name}.gpu.png")), &gpu);
            let _ = filmcraft_testkit::golden::write_png(&dir.join(format!("{name}.cpu.png")), &cpu);
            failures.push(format!("{name} ({kind}): {d}; images in {}", dir.display()));
        }
    }
    assert!(failures.is_empty(), "GPU differs from the CPU reference:\n{}", failures.join("\n"));
}

/// Per-pixel max RGB difference (99th percentile, mean) over the pixels at least 1.5 px from every
/// plan layer's quad outline: the CPU resampler fades a layer's edge over one more pixel than the
/// GPU rasterises, and with blend modes those edge pixels are a visible share of a 320×180 frame.
fn interior_diff(plan: &filmcraft_render::plan::FramePlan, a: &Rgba8, b: &Rgba8) -> (u32, f64) {
    let filmcraft_render::plan::FramePlan::Layers { width, layers, .. } = plan else { return (u32::MAX, f64::MAX) };
    let (p99, mean, _) = layers_interior_diff(*width, &layers.iter().collect::<Vec<_>>(), a, b);
    (p99, mean)
}

/// [`interior_diff`] away from the outlines of `layers` (the layers of every step of a plan), and
/// the PSNR of the per-pixel max channel difference there (dB).
fn layers_interior_diff(width: usize, layers: &[&filmcraft_render::plan::PlanLayer], a: &Rgba8, b: &Rgba8) -> (u32, f64, f64) {
    let inv: Vec<_> = layers.iter().filter_map(|l| Some((l.matrix.inverse()?, l.size().0 as f64, l.size().1 as f64))).collect();
    let mut d: Vec<u32> = Vec::new();
    for (i, (pa, pb)) in a.px.chunks(4).zip(b.px.chunks(4)).enumerate() {
        let (x, y) = ((i % width) as f64 + 0.5, (i / width) as f64 + 0.5);
        let edge = inv.iter().any(|(m, fw, fh)| {
            let c = [(-1.5, -1.5), (1.5, -1.5), (-1.5, 1.5), (1.5, 1.5)].map(|(dx, dy)| {
                let p = m.apply(Vec2::new(x + dx, y + dy));
                p.x >= 0.0 && p.y >= 0.0 && p.x <= *fw && p.y <= *fh
            });
            c.iter().any(|v| *v != c[0])
        });
        if !edge {
            d.push((0..3).map(|k| (pa[k] as i32 - pb[k] as i32).unsigned_abs()).max().unwrap_or(0));
        }
    }
    d.sort_unstable();
    let mse = d.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / d.len() as f64;
    let psnr = if mse > 0.0 { 10.0 * (255.0f64 * 255.0 / mse).log10() } else { f64::INFINITY };
    (d[d.len() * 99 / 100], d.iter().sum::<u32>() as f64 / d.len() as f64, psnr)
}

/// Every blend mode composited by the GPU from `plan_frame` (no CPU fallback): two overlapping
/// transformed clips at partial opacity in the mode over footage, plus a clip with a standard
/// effect (rendered on the CPU as a layer image) in the same mode, match the CPU render with the
/// GPU parity tolerance (p99 ≤ 6, mean < 1.5) away from layer outlines; whole-frame numbers are
/// printed too.
#[test]
fn gpu_blend_modes_match_cpu_render() {
    let Some((dev, q)) = device("gpu blend parity", false) else {
        return;
    };
    let mut c = filmcraft_gpu::GpuCompositor::new(&dev, &q);
    let mut failures = Vec::new();
    for mode in filmcraft_project::effect::BLEND_MODES {
        let mut b = Builder::new(4);
        let bg = b.demo(DemoScene::OceanSunset);
        let (dunes, aurora, bars) = (b.demo(DemoScene::Dunes), b.demo(DemoScene::Aurora), b.media(Generator::BarsAndTone));
        b.place(0, bg, 0, 48);
        let v2 = b.place(1, dunes, 0, 48);
        b.fixed(v2, "motion", &[("scale", fl(60.0)), ("rotation", fl(12.0)), ("position", pt(130.0, 80.0))]);
        b.fixed(v2, "opacity", &[("opacity", fl(70.0)), ("blend", blend(mode))]);
        let v3 = b.place(2, aurora, 0, 48);
        b.fixed(v3, "motion", &[("scale", fl(45.0)), ("rotation", fl(-20.0)), ("position", pt(210.0, 110.0))]);
        b.fixed(v3, "opacity", &[("opacity", fl(60.0)), ("blend", blend(mode))]);
        let v4 = b.place(3, bars, 0, 48);
        b.fixed(v4, "motion", &[("scale", fl(30.0)), ("position", pt(250.0, 50.0))]);
        b.fixed(v4, "opacity", &[("opacity", fl(85.0)), ("blend", blend(mode))]);
        b.effect(v4, "crop", &[("left", fl(10.0)), ("bottom", fl(20.0))]);
        let s = b.at(12);
        let t = RATE.tick_of(s.frame);
        let cpu = render_cpu(&s);
        let plan = filmcraft_render::plan::plan_frame(&s.project, s.seq, t, RenderOptions::default(), &s.sources);
        let filmcraft_render::plan::FramePlan::Layers { layers, .. } = &plan else {
            failures.push(format!("{mode}: planned as a CPU image"));
            continue;
        };
        assert_eq!(layers.len(), 4, "{mode}");
        c.composite(&plan);
        let (gw, gh, mut px) = c.read_output().expect("GPU readback");
        px.as_chunks_mut::<4>().0.iter_mut().for_each(|p| p[3] = 255);
        let gpu = Rgba8::new(gw, gh, px);
        let d = diff(&cpu, &gpu).unwrap();
        let (p99, mean) = interior_diff(&plan, &cpu, &gpu);
        eprintln!("{mode}: GPU vs CPU interior p99 {p99}, mean {mean:.3}; whole frame {d}");
        if p99 > 6 || mean >= 1.5 {
            failures.push(format!("{mode}: interior p99 {p99}, mean {mean}; {d}"));
        }
    }
    assert!(failures.is_empty(), "GPU blend modes differ from the CPU render:\n{}", failures.join("\n"));
}

/// Inside a transition the CPU mixes the clips and composites the result Normal, ignoring their
/// blend modes: the plan does the same (Screen clips cross-dissolving on V2 over V1).
#[test]
fn gpu_transition_of_blended_clips_matches_cpu_render() {
    let Some((dev, q)) = device("gpu blend parity", false) else {
        return;
    };
    let mut b = Builder::new(2);
    let (bg, x, y) = (b.demo(DemoScene::OceanSunset), b.demo(DemoScene::Aurora), b.demo(DemoScene::Dunes));
    b.place(0, bg, 0, 48);
    let ca = b.place(1, x, 0, 24);
    let cb = b.place(1, y, 24, 24);
    for id in [ca, cb] {
        b.fixed(id, "opacity", &[("blend", blend("Screen"))]);
    }
    b.transition(1, "cross_dissolve", ca, cb, 24, 12);
    let s = b.at(22);
    let plan = filmcraft_render::plan::plan_frame(&s.project, s.seq, RATE.tick_of(s.frame), RenderOptions::default(), &s.sources);
    let filmcraft_render::plan::FramePlan::Layers { layers, .. } = &plan else { panic!("planned as a CPU image") };
    assert!(layers.iter().all(|l| l.blend == filmcraft_render::Blend::Normal));
    let mut c = filmcraft_gpu::GpuCompositor::new(&dev, &q);
    c.composite(&plan);
    let (gw, gh, mut px) = c.read_output().expect("GPU readback");
    px.as_chunks_mut::<4>().0.iter_mut().for_each(|p| p[3] = 255);
    let d = diff(&render_cpu(&s), &Rgba8::new(gw, gh, px)).unwrap();
    eprintln!("Screen clips in a cross dissolve: GPU vs CPU {d}");
    assert!(d.p99 <= 6 && d.mean_abs * 4.0 / 3.0 < 1.5, "{d}");
}

/// Standard effect chains run on the GPU: every clip below plans as a GPU layer with its effects
/// (no CPU-rendered image), including keyframed parameters, transformed clips, partial opacity
/// and blend modes, and matches the CPU render with the GPU parity tolerance (p99 ≤ 6, mean < 1.5
/// away from layer outlines) at three times. A clip with an effect the GPU stage lacks (Emboss)
/// still falls back to a CPU layer image.
#[test]
fn gpu_effect_chains_match_cpu_render() {
    let mut b = Builder::new(5);
    let (bg, dunes, aurora, bars, sunset) =
        (b.demo(DemoScene::OceanSunset), b.demo(DemoScene::Dunes), b.demo(DemoScene::Aurora), b.media(Generator::BarsAndTone), b.demo(DemoScene::OceanSunset));
    let v1 = b.place(0, bg, 0, 48);
    b.effect(v1, "brightness_contrast", &[("brightness", fl(-10.0)), ("contrast", fl(20.0))]);
    b.effect(v1, "gaussian_blur", &[("repeat_edge", ParamValue::Bool(true))]);
    // keyframed blur: evaluated at the frame's time
    let p = b.clip(v1).effects.last_mut().and_then(|e| e.params.get_mut("blurriness")).expect("blurriness");
    p.put_keyframe(Tick::ZERO, fl(2.0));
    p.put_keyframe(RATE.tick_of(24), fl(20.0));
    let v2 = b.place(1, dunes, 0, 48);
    b.fixed(v2, "motion", &[("scale", fl(60.0)), ("rotation", fl(12.0)), ("position", pt(130.0, 80.0))]);
    b.fixed(v2, "opacity", &[("opacity", fl(70.0)), ("blend", blend("Screen"))]);
    b.effect(v2, "tint", &[("amount", fl(60.0))]);
    b.effect(v2, "unsharp_mask", &[("amount", fl(120.0)), ("radius", fl(2.0))]);
    b.effect(v2, "crop", &[("left", fl(8.0)), ("bottom", fl(12.0)), ("feather", fl(6.0))]);
    let v3 = b.place(2, aurora, 0, 48);
    b.fixed(v3, "motion", &[("scale", fl(45.0)), ("rotation", fl(-20.0)), ("position", pt(210.0, 110.0))]);
    b.fixed(v3, "opacity", &[("opacity", fl(85.0))]);
    b.effect(v3, "levels", &[("in_black", fl(20.0)), ("gamma", fl(130.0))]);
    b.effect(v3, "mirror", &[("angle", fl(60.0))]);
    b.effect(v3, "directional_blur", &[("length", fl(4.0)), ("direction", fl(45.0))]);
    b.effect(v3, "transform", &[("rotation", fl(10.0)), ("scale_height", fl(110.0))]);
    let v4 = b.place(3, bars, 0, 48);
    b.fixed(v4, "motion", &[("scale", fl(30.0)), ("position", pt(250.0, 50.0))]);
    b.fixed(v4, "opacity", &[("blend", blend("Multiply"))]);
    b.effect(v4, "posterize", &[("levels", fl(5.0))]);
    b.effect(v4, "invert", &[("blend", fl(40.0))]);
    b.effect(v4, "offset", &[("shift", pt(400.0, 120.0))]);
    b.effect(v4, "horizontal_flip", &[]);
    b.effect(v4, "black_white", &[]);
    let v5 = b.place(4, sunset, 0, 48);
    b.fixed(v5, "motion", &[("scale", fl(25.0)), ("position", pt(60.0, 140.0))]);
    b.effect(v5, "emboss", &[]);
    let mut s = b.at(12);
    let plan = filmcraft_render::plan::plan_frame(&s.project, s.seq, RATE.tick_of(s.frame), RenderOptions::default(), &s.sources);
    let filmcraft_render::plan::FramePlan::Layers { layers, .. } = &plan else { panic!("planned as a CPU image") };
    assert_eq!(layers.len(), 5);
    for (i, l) in layers.iter().take(4).enumerate() {
        let fx = l.fx.as_ref().unwrap_or_else(|| panic!("V{}: not a GPU effect layer", i + 1));
        assert!(!fx.ops.is_empty());
    }
    assert!(layers[4].fx.is_none(), "Emboss is rendered on the CPU");
    let Some((dev, q)) = device("gpu effect parity", true) else {
        return;
    };
    let mut c = filmcraft_gpu::GpuCompositor::new(&dev, &q);
    let mut failures = Vec::new();
    for frame in [0, 12, 30] {
        s.frame = frame;
        let plan = filmcraft_render::plan::plan_frame(&s.project, s.seq, RATE.tick_of(frame), RenderOptions::default(), &s.sources);
        c.composite(&plan);
        let (gw, gh, mut px) = c.read_output().expect("GPU readback");
        px.as_chunks_mut::<4>().0.iter_mut().for_each(|p| p[3] = 255);
        let gpu = Rgba8::new(gw, gh, px);
        let cpu = render_cpu(&s);
        let d = diff(&cpu, &gpu).unwrap();
        let (p99, mean) = interior_diff(&plan, &cpu, &gpu);
        eprintln!("effect chains @ {frame}: GPU vs CPU interior p99 {p99}, mean {mean:.3}; whole frame {d}");
        if p99 > 6 || mean >= 1.5 {
            let dir = filmcraft_testkit::golden::failures_dir();
            let _ = filmcraft_testkit::golden::write_png(&dir.join(format!("fx_chains_{frame}.gpu.png")), &gpu);
            let _ = filmcraft_testkit::golden::write_png(&dir.join(format!("fx_chains_{frame}.cpu.png")), &cpu);
            failures.push(format!("frame {frame}: interior p99 {p99}, mean {mean}; {d}; images in {}", dir.display()));
        }
    }
    // ½ playback resolution: the working images (and the effects' pixel radii) shrink with it
    let opts = RenderOptions { scale: 0.5, ..RenderOptions::default() };
    let plan = filmcraft_render::plan::plan_frame(&s.project, s.seq, RATE.tick_of(s.frame), opts, &s.sources);
    let filmcraft_render::plan::FramePlan::Layers { layers, .. } = &plan else { panic!("½: planned as a CPU image") };
    assert!(layers.iter().take(4).all(|l| l.fx.is_some()));
    c.composite(&plan);
    let (gw, gh, mut px) = c.read_output().expect("GPU readback");
    px.as_chunks_mut::<4>().0.iter_mut().for_each(|p| p[3] = 255);
    let gpu = Rgba8::new(gw, gh, px);
    let img = render_sequence(&s.project, s.seq, RATE.tick_of(s.frame), opts, &s.sources);
    let cpu = Rgba8::new(img.w as u32, img.h as u32, img.over_black_rgba8());
    let (p99, mean) = interior_diff(&plan, &cpu, &gpu);
    eprintln!("effect chains @ ½: GPU vs CPU interior p99 {p99}, mean {mean:.3}; whole frame {}", diff(&cpu, &gpu).unwrap());
    if p99 > 6 || mean >= 1.5 {
        failures.push(format!("½ resolution: interior p99 {p99}, mean {mean}"));
    }
    assert!(failures.is_empty(), "GPU effect chains differ from the CPU render:\n{}", failures.join("\n"));
}

/// A cross dissolve is planned as layers (A, then B over it at the progress) only where that is
/// the CPU's linear mix: B fills the frame opaquely. Between scaled clips it is a two-input step,
/// and the plan run on the CPU is the CPU render.
#[test]
fn cross_dissolves_plan_as_the_cpu_mixes_them() {
    for scaled in [false, true] {
        let mut b = Builder::new(2);
        let (bg, x, y) = (b.demo(DemoScene::Forest), b.demo(DemoScene::Aurora), b.demo(DemoScene::Dunes));
        b.place(0, bg, 0, 48);
        let ca = b.place(1, x, 0, 24);
        let cb = b.place(1, y, 24, 24);
        if scaled {
            b.fixed(ca, "motion", &[("scale", fl(70.0))]);
            b.fixed(cb, "motion", &[("scale", fl(50.0))]);
        }
        b.transition(1, "cross_dissolve", ca, cb, 24, 12);
        let s = b.at(22);
        let t = RATE.tick_of(s.frame);
        let plan = filmcraft_render::plan::plan_frame(&s.project, s.seq, t, RenderOptions::default(), &s.sources);
        let two_input = matches!(&plan, filmcraft_render::plan::FramePlan::Composite { steps, .. }
            if steps.iter().any(|st| matches!(st, filmcraft_render::plan::PlanStep::Transition { .. })));
        assert_eq!(two_input, scaled, "scaled {scaled}");
        let cpu = render_cpu(&s);
        let img = filmcraft_render::plan::execute_cpu(&plan);
        let d = diff(&cpu, &Rgba8::new(img.w as u32, img.h as u32, img.over_black_rgba8())).unwrap();
        eprintln!("cross dissolve (scaled {scaled}): the plan on the CPU vs the CPU render {d}");
        assert!(d.p99 <= 1, "scaled {scaled}: {d}");
    }
}

/// A showcase-like scene through the GPU — the default since 2026-10-21 (`docs/gpu.md`: GPU
/// rendering is Auto): a blurred backdrop under scaled clips (the portrait fill), a redaction blur
/// behind a tracked (keyframed) face mask, motion and opacity keyframes, cross dissolve, dip to
/// black, Push and a whip pan, and a title fading in. Every frame plans for the GPU (the masked
/// blur as a GPU effect, not a CPU layer image) and both GPU paths — the monitors' composite and
/// the export renderer's linear read-back — match the CPU render with the GPU parity tolerance
/// (p99 ≤ 6, mean < 1.5 and PSNR ≥ 45 dB away from layer outlines, as for the blend modes; the
/// whole-frame numbers, outlines included, are printed).
#[test]
fn gpu_showcase_scene_matches_cpu() {
    use filmcraft_project::graphic::new_text_layer;
    let mut b = Builder::new(3);
    let backdrop = b.demo(DemoScene::Forest);
    let bg = b.place(0, backdrop, 0, 120);
    b.effect(bg, "gaussian_blur", &[("blurriness", fl(60.0)), ("repeat_edge", ParamValue::Bool(true))]);
    let scenes = [DemoScene::Aurora, DemoScene::CityNight, DemoScene::OceanSunset, DemoScene::Dunes, DemoScene::Plasma];
    let mut clips = Vec::new();
    for (k, s) in scenes.into_iter().enumerate() {
        let item = b.demo(s);
        let c = b.place(1, item, 24 * k as i64, 24);
        b.fixed(c, "motion", &[("scale", fl(60.0))]);
        clips.push(c);
    }
    // keyframed motion and opacity
    let p = b.clip(clips[0]).effect_mut("motion").and_then(|e| e.params.get_mut("scale")).expect("scale");
    p.put_keyframe(Tick::ZERO, fl(55.0));
    p.put_keyframe(RATE.tick_of(24), fl(66.0));
    let p = b.clip(clips[1]).effect_mut("opacity").and_then(|e| e.params.get_mut("opacity")).expect("opacity");
    p.put_keyframe(Tick::ZERO, fl(100.0));
    p.put_keyframe(RATE.tick_of(12), fl(55.0));
    p.put_keyframe(RATE.tick_of(24), fl(100.0));
    // the redaction: a blur behind a face mask that moves (a tracked path)
    b.effect(clips[2], "gaussian_blur", &[("blurriness", fl(60.0)), ("repeat_edge", ParamValue::Bool(true))]);
    b.mask(clips[2], "gaussian_blur", MaskPath::ellipse(Vec2::new(250.0, 150.0), Vec2::new(70.0, 90.0)), 12.0, |m| {
        m.path.put_keyframe(Tick::ZERO, ParamValue::Path(MaskPath::ellipse(Vec2::new(250.0, 150.0), Vec2::new(70.0, 90.0))));
        m.path.put_keyframe(RATE.tick_of(24), ParamValue::Path(MaskPath::ellipse(Vec2::new(390.0, 190.0), Vec2::new(80.0, 100.0))));
    });
    for (k, id) in ["cross_dissolve", "dip_to_black", "push", "cam_whip_pan"].into_iter().enumerate() {
        b.transition(1, id, clips[k], clips[k + 1], 24 * (k as i64 + 1), 12);
    }
    // the title
    let g = b.p.add_item("Title", Label::Rose, ItemKind::Graphic { width: W, height: H, rate: RATE }, None);
    let mut title = b.p.make_track_item(g, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, RATE.tick_of(120)), RATE).expect("title");
    title.effects.push(new_text_layer("Every Robot", Vec2::new(20.0, 40.0), 26.0));
    for e in title.effects.iter_mut() {
        filmcraft_project::resolve_auto_points(e, (W, H), (W, H));
    }
    let o = title.effect_mut("opacity").and_then(|e| e.params.get_mut("opacity")).expect("opacity");
    o.put_keyframe(Tick::ZERO, fl(0.0));
    o.put_keyframe(RATE.tick_of(18), fl(100.0));
    b.p.sequence_mut(b.seq).expect("sequence").video_tracks[2].items.push(title);
    let mut s = b.at(0);
    let Some((dev, q)) = device("gpu showcase parity", true) else {
        return;
    };
    let mut c = filmcraft_gpu::GpuCompositor::new(&dev, &q);
    let mut export = filmcraft_gpu::ExportRenderer::new();
    let mut failures = Vec::new();
    for frame in [6, 21, 24, 27, 45, 48, 50, 60, 70, 72, 75, 93, 96, 99, 110] {
        s.frame = frame;
        let t = RATE.tick_of(frame);
        let plan = filmcraft_render::plan::plan_frame(&s.project, s.seq, t, RenderOptions::default(), &s.sources);
        let layers: Vec<&filmcraft_render::plan::PlanLayer> = match &plan {
            filmcraft_render::plan::FramePlan::Layers { layers, .. } => layers.iter().collect(),
            filmcraft_render::plan::FramePlan::Composite { steps, .. } => steps
                .iter()
                .flat_map(|st| match st {
                    filmcraft_render::plan::PlanStep::Layers(l) => l.iter().collect::<Vec<_>>(),
                    filmcraft_render::plan::PlanStep::Transition { inputs, .. } => inputs.iter().flatten().collect(),
                })
                .collect(),
            filmcraft_render::plan::FramePlan::Image(_) => panic!("frame {frame}: planned as a CPU image"),
        };
        if (50..70).contains(&frame) {
            let masked =
                layers.iter().any(|l| l.fx.as_ref().is_some_and(|fx| fx.ops.iter().any(|o| matches!(o, filmcraft_render::gpufx::FxOp::Masked { .. }))));
            assert!(masked, "frame {frame}: the masked blur runs on the GPU");
        }
        let cpu = render_cpu(&s);
        c.composite(&plan);
        let (gw, gh, mut px) = c.read_output().expect("GPU readback");
        px.as_chunks_mut::<4>().0.iter_mut().for_each(|p| p[3] = 255);
        let mut outputs = vec![("monitor", Rgba8::new(gw, gh, px))];
        if let Some(r) = export.as_mut() {
            let img = r.render(&s.project, s.seq, t, RenderOptions::default(), &s.sources).expect("the export renderer draws the frame");
            outputs.push(("export", Rgba8::new(img.w as u32, img.h as u32, img.over_black_rgba8())));
        }
        for (path, gpu) in outputs {
            let d = diff(&cpu, &gpu).unwrap();
            let (p99, mean, psnr) = layers_interior_diff(W as usize, &layers, &cpu, &gpu);
            eprintln!("showcase @ {frame} ({path}): GPU vs CPU interior p99 {p99}, mean {mean:.3}, PSNR {psnr:.1} dB; whole frame {d}");
            if p99 > 6 || mean >= 1.5 || psnr < 45.0 {
                let dir = filmcraft_testkit::golden::failures_dir();
                let _ = filmcraft_testkit::golden::write_png(&dir.join(format!("showcase_{frame}_{path}.gpu.png")), &gpu);
                let _ = filmcraft_testkit::golden::write_png(&dir.join(format!("showcase_{frame}.cpu.png")), &cpu);
                let _ = filmcraft_testkit::golden::write_png(
                    &dir.join(format!("showcase_{frame}_{path}.diff.png")),
                    &filmcraft_testkit::golden::diff_image(&cpu, &gpu),
                );
                failures.push(format!("frame {frame} ({path}): {d}; images in {}", dir.display()));
            }
        }
    }
    assert!(failures.is_empty(), "the GPU differs from the CPU on the showcase scene:\n{}", failures.join("\n"));
}
