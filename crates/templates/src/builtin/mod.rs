//! The built-in starter templates: original designs made for FilmCraft in code. Their
//! placeholder media is generated (the procedural demo scenes and colour mattes of
//! `filmcraft-media`, graphics drawn with the bundled Inter font), never stock footage; their
//! sounds are Elements sound effects (synthesized when the template is applied), their titles
//! and stickers Elements graphics, their transitions from the built-in and GL Transitions
//! catalogs, and an unfilled music slot gets a bed generated in the app — so they need no asset
//! files at all.
//!
//! | id | what |
//! |---|---|
//! | `builtin:photo-beat-slideshow` | 8 photos cut on the beat with a slow push-in, title and end text |
//! | `builtin:channel-intro` | a 5 s channel intro: logo pop over a moving background, channel name and tagline |
//! | `builtin:top-3-listicle` | "top 3" countdown: title card, then #3, #2, #1 each with a number card and a clip |
//! | `builtin:talking-head-captions` | 30 s talking head with a hook title, name lower third and auto-captions |
//! | `builtin:product-promo-15` | 15 s product promo: title, four shots with feature captions on the beat, offer card |
//! | `builtin:travel-vlog-intro` | 10 s fast-cut travel intro: three punch-in words, the place name |
//! | `builtin:travel-vlog-outro` | 10 s slow outro: three clips crossfading, thank-you and next stop |
//! | `builtin:top-5-listicle` | "top 5" countdown with number cards and GL transitions |
//! | `builtin:before-after-reveal` | before, a wipe to after, then both side by side |
//! | `builtin:lower-third-chapter-kit` | a video with a name lower third and three chapter cards (chapter markers) |
//! | `builtin:end-card-subscribe` | 12 s end card: two "watch next" frames and a subscribe button with a bell |
//! | `builtin:tutorial-pip` | screen recording with the webcam picture-in-picture, title, steps and captions |
//! | `builtin:podcast-two-speakers` | two speakers side by side, name tags, show title and captions |
//! | `builtin:photo-collage` | four photos popping into a 2×2 grid, a title over them |
//! | `builtin:quote-card` | a quote and its author over a darkened photo |
//! | `builtin:countdown` | 3, 2, 1, GO! over a background, then the event title |
//! | `builtin:event-recap` | twelve clips cut on the beat, event title, logo end card |
//! | `builtin:day-in-my-life` | six moments of a day, each with its time |
//! | `builtin:unboxing` | package, opening, the reveal, details and a verdict |
//! | `builtin:qa-session` | three questions on cards, each answered on camera, with captions |
//! | `builtin:robot-showcase` | five robots (or gadgets), each with a spec card, glitch cuts on the beat |
//!
//! Each comes in 16:9 (1920×1080) and 9:16 (1080×1920); the social ones also in 1:1
//! (1080×1080) and 4:5 (1080×1350).

mod classic;
mod creator;
mod social;
mod stories;

use std::collections::BTreeMap;

use filmcraft_geom::Vec2;
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{DemoScene, Generator, MediaKind, MediaSource};
use filmcraft_project::graphic::{new_shape_layer, new_text_layer};
use filmcraft_project::graphic_design::LayerExtra;
use filmcraft_project::{
    Caption, CaptionAlign, CaptionAnchor, CaptionFormat, CaptionStyle, CaptionTrack, ClipId, EffectInstance, ItemId, ItemKind, Keyframe, Label, Marker,
    MarkerId, MarkerKind, MediaClip, MediaRef, Param, ParamValue, Project, SequenceSettings, TrackId, TrackItem, TrackKind, Transition, TransitionAlign,
    TransitionId, find_effect,
};
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick, TimeRange};

use crate::manifest::{
    BeatGrid, CaptionSpec, Constraints, Fit, Framing, Hint, LayerHint, Manifest, Placement, Portion, ShortFit, Slot, SlotDefault, SlotKind, Variant,
    default_fit,
};
use crate::{FORMAT, FORMAT_VERSION, Package, SEQUENCE_FILE, TemplateError};

pub(crate) const RATE: FrameRate = FrameRate::FPS_30;
pub(crate) const W: u32 = 1920;
pub(crate) const H: u32 = 1080;
const YOUTUBE: &str = "YouTube 1080p Full HD";
const VERTICAL: &str = "Social Vertical 1080×1920";
/// Exports at the sequence's own size (1080×1080, 1080×1350).
const MATCH_SOURCE: &str = "Match Source – Adaptive High Bitrate";
const AUTHOR: &str = "FilmCraft contributors";
const LICENSE: &str = "MIT OR Apache-2.0";

/// Category names of the library.
pub const CATEGORIES: [&str; 7] = ["Reels & Shorts", "YouTube", "Vlog", "Business", "Social", "Events", "Tech"];

/// The Elements sound effects a template may use and their length in seconds (as the engine's
/// `elements::SOUNDS` makes them; an engine test keeps the two in step).
pub const SOUND_SECONDS: [(&str, f64); 12] = [
    ("sound.whoosh-short", 0.6),
    ("sound.whoosh-long", 1.8),
    ("sound.impact-soft", 2.0),
    ("sound.impact-hard", 3.5),
    ("sound.riser-2s", 2.0),
    ("sound.riser-4s", 4.0),
    ("sound.swoosh-up", 0.7),
    ("sound.swoosh-down", 0.7),
    ("sound.click", 0.12),
    ("sound.pop", 0.3),
    ("sound.sweep", 1.6),
    ("sound.boom", 3.0),
];

pub(crate) fn frames(n: i64) -> Tick {
    RATE.tick_of(n)
}

pub(crate) fn secs(s: f64) -> Tick {
    RATE.snap_nearest(Tick::from_seconds_f64(s))
}

pub(crate) fn white() -> [f32; 4] {
    [1.0, 1.0, 1.0, 1.0]
}

/// `0xRRGGBB` as RGBA fractions (the sRGB values graphics store).
pub(crate) fn rgb(hex: u32) -> [f32; 4] {
    let c = |s: u32| ((hex >> s) & 0xff) as f32 / 255.0;
    [c(16), c(8), c(0), 1.0]
}

/// A generated stand-in for an Elements sound `id`, `len` long: silence, sound only, its item's
/// metadata naming the element (the engine swaps in the synthesized sound when applying).
pub(crate) fn sound_stand_in(p: &mut Project, id: &str, name: &str, len: Tick) -> ItemId {
    let g = Generator::Tone { hz: 0.0, db: -120.0 };
    let src = GeneratorSource::new(g.clone(), 16, 16, RATE, len.max(Tick(1)));
    let mut info = src.info().clone();
    info.name = name.to_string();
    info.video = None;
    info.kind = MediaKind::AudioOnly;
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
    let item = p.add_item(name, Label::Caribbean, ItemKind::Media(clip), None);
    if let Some(it) = p.item_mut(item) {
        it.metadata.insert(crate::ELEMENT_METADATA.into(), id.to_string());
    }
    item
}

/// Builds a template's sequence file (designed at 1920×1080, 30 fps).
pub(crate) struct Builder {
    p: Project,
    seq: ItemId,
    canvas: Option<ItemId>,
    stand_ins: BTreeMap<String, ItemId>,
}

impl Builder {
    pub(crate) fn new(name: &str, video: usize, audio: usize) -> Builder {
        let mut p = Project::new(name);
        let settings = SequenceSettings { width: W, height: H, frame_rate: RATE, preset: "Template 1080p 30".into(), ..Default::default() };
        let seq = p.new_sequence(name, settings, video, audio, None);
        Builder { p, seq, canvas: None, stand_ins: BTreeMap::new() }
    }

    pub(crate) fn generator(&mut self, name: &str, g: Generator, size: (u32, u32), label: Label) -> ItemId {
        let src = GeneratorSource::new(g.clone(), size.0, size.1, RATE, Tick(60 * TICKS_PER_SECOND));
        let mut info = src.info().clone();
        info.name = name.to_string();
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
        self.p.add_item(name, label, ItemKind::Media(clip), None)
    }

    /// A full-frame demo scene (a picture placeholder).
    pub(crate) fn scene(&mut self, name: &str, scene: DemoScene) -> ItemId {
        self.generator(name, Generator::Demo(scene), (W, H), Label::Iris)
    }

    /// The `k`-th demo scene, round the set.
    pub(crate) fn nth_scene(&mut self, name: &str, k: usize) -> ItemId {
        let scenes = DemoScene::ALL;
        let scene = scenes.get(k % scenes.len()).copied().unwrap_or(DemoScene::OceanSunset);
        self.scene(name, scene)
    }

    /// A full-frame colour matte.
    pub(crate) fn matte(&mut self, name: &str, color: [f32; 4]) -> ItemId {
        self.generator(name, Generator::ColorMatte { color }, (W, H), Label::Lavender)
    }

    /// A generated stand-in for music (sound only).
    pub(crate) fn music_placeholder(&mut self) -> ItemId {
        let id = self.generator("Placeholder Music", Generator::Demo(DemoScene::Aurora), (W, H), Label::Caribbean);
        if let Some(m) = self.p.item_mut(id).and_then(|i| i.as_media_mut()) {
            m.info.video = None;
            m.info.kind = MediaKind::AudioOnly;
        }
        id
    }

    /// The template's graphics canvas (one item; each graphic clip has its own layers).
    pub(crate) fn graphic(&mut self, name: &str) -> ItemId {
        if let Some(g) = self.canvas {
            return g;
        }
        let g = self.p.add_item(name, Label::Rose, ItemKind::Graphic { width: W, height: H, rate: RATE }, None);
        self.canvas = Some(g);
        g
    }

    pub(crate) fn clip(&mut self, kind: TrackKind, track: usize, item: ItemId, start: Tick, dur: Tick) -> ClipId {
        let Some(mut it) = self.p.make_track_item(item, kind, start, TimeRange::new(Tick::ZERO, dur), RATE) else { return ClipId(0) };
        it.duration = dur;
        let id = it.id;
        if let Some(t) = self.p.sequence_mut(self.seq).and_then(|q| q.tracks_mut(kind).get_mut(track)) {
            t.items.push(it);
            t.sort();
        }
        id
    }

    /// A picture placeholder with its sound: linked clips on video track `v` and audio track `a`
    /// (a slot's media brings its own sound there). Returns the video clip.
    pub(crate) fn av(&mut self, v: usize, a: usize, item: ItemId, start: Tick, dur: Tick) -> ClipId {
        let link = self.p.alloc_id();
        let vc = self.clip(TrackKind::Video, v, item, start, dur);
        let ac = self.clip(TrackKind::Audio, a, item, start, dur);
        for c in [vc, ac] {
            self.with_clip(c, |it| it.link = Some(link));
        }
        vc
    }

    pub(crate) fn with_clip(&mut self, clip: ClipId, f: impl FnOnce(&mut TrackItem)) {
        if let Some((_, it)) = self.p.sequence_mut(self.seq).and_then(|q| q.find_item_mut(clip)) {
            f(it);
        }
    }

    /// Animate `effect.param` through `(media time, value)` keys.
    pub(crate) fn ramp(&mut self, clip: ClipId, effect: &str, param: &str, keys: &[(Tick, f64)]) {
        self.with_clip(clip, |it| {
            if let Some(e) = it.effects.iter_mut().find(|e| e.effect == effect) {
                let prm = e.params.entry(param.to_string()).or_insert_with(|| Param::new(ParamValue::Float(0.0)));
                prm.keyframes = keys.iter().map(|(t, v)| Keyframe::new(*t, ParamValue::Float(*v))).collect();
                if let Some((_, v)) = keys.first() {
                    prm.value = ParamValue::Float(*v);
                }
            }
        });
    }

    pub(crate) fn fade_in_out(&mut self, clip: ClipId, dur: Tick, fade: Tick) {
        self.ramp(clip, "opacity", "opacity", &[(Tick::ZERO, 0.0), (fade, 100.0), (dur - fade, 100.0), (dur, 0.0)]);
    }

    /// Set a parameter of one of the clip's effects (static).
    pub(crate) fn set(&mut self, clip: ClipId, effect: &str, param: &str, v: ParamValue) {
        self.with_clip(clip, |it| {
            if let Some(e) = it.effects.iter_mut().find(|e| e.effect == effect) {
                e.params.insert(param.to_string(), Param::new(v));
            }
        });
    }

    pub(crate) fn opacity(&mut self, clip: ClipId, pct: f64) {
        self.set(clip, "opacity", "opacity", ParamValue::Float(pct));
    }

    /// An audio clip's level, dB.
    pub(crate) fn volume(&mut self, clip: ClipId, db: f64) {
        self.set(clip, "volume", "level", ParamValue::Float(db));
    }

    /// Add effect `id` (before the intrinsic ones) with `params`.
    pub(crate) fn effect(&mut self, clip: ClipId, id: &str, params: &[(&str, ParamValue)]) {
        let Some(def) = find_effect(id) else { return };
        let mut e = def.instance();
        for (k, v) in params {
            e.params.insert((*k).to_string(), Param::new(v.clone()));
        }
        self.with_clip(clip, |it| {
            let at = it.effects.iter().position(|e| e.def().is_some_and(|d| d.intrinsic)).unwrap_or(it.effects.len());
            it.effects.insert(at, e);
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn text(&mut self, clip: ClipId, uid: u64, text: &str, pos: (f64, f64), size: f64, style: &str, color: [f32; 4]) {
        self.text_ex(clip, uid, text, pos, size, style, color, 1, &[]);
    }

    /// A text layer: `align` 0 left, 1 centre, 2 right; `extra` more parameters (`box_width`,
    /// `background`…).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn text_ex(
        &mut self,
        clip: ClipId,
        uid: u64,
        text: &str,
        pos: (f64, f64),
        size: f64,
        style: &str,
        color: [f32; 4],
        align: u32,
        extra: &[(&str, ParamValue)],
    ) {
        self.with_clip(clip, |it| {
            let mut e = new_text_layer(text, Vec2::new(pos.0, pos.1), size);
            let set = |e: &mut EffectInstance, k: &str, v: ParamValue| {
                e.params.insert(k.to_string(), Param::new(v));
            };
            set(&mut e, "name", ParamValue::Text(format!("Text {uid}")));
            set(&mut e, "font", ParamValue::Text("Inter".into()));
            set(&mut e, "font_style", ParamValue::Text(style.into()));
            set(&mut e, "align", ParamValue::Choice(align));
            set(&mut e, "fill_color", ParamValue::Color(color));
            set(&mut e, "shadow", ParamValue::Bool(true));
            set(&mut e, "shadow_distance", ParamValue::Float(4.0));
            set(&mut e, "shadow_blur", ParamValue::Float(18.0));
            set(&mut e, "shadow_opacity", ParamValue::Float(55.0));
            for (k, v) in extra {
                set(&mut e, k, v.clone());
            }
            e.layer = Some(Box::new(LayerExtra { uid, ..Default::default() }));
            it.effects.push(e);
        });
    }

    /// A text tag: centred text (`centre` is the middle of its capitals) on a rounded box
    /// (`bg`, `opacity` %, `padding` and `radius` pixels) that grows with the text.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tag(&mut self, clip: ClipId, uid: u64, text: &str, centre: (f64, f64), size: f64, style: &str, color: [f32; 4], bg: [f32; 4], opacity: f64) {
        let extra = [
            ("background", ParamValue::Bool(true)),
            ("background_color", ParamValue::Color(bg)),
            ("background_opacity", ParamValue::Float(opacity)),
            ("background_size", ParamValue::Float((size * 0.35).round())),
            ("background_radius", ParamValue::Float((size * 0.3).round())),
            ("shadow", ParamValue::Bool(false)),
        ];
        self.text_ex(clip, uid, text, (centre.0, centre.1 + (0.36 * size).round()), size, style, color, 1, &extra);
    }

    /// A rectangle layer (design pixels, centred on `pos`), `radius` its corner radius.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rect(&mut self, clip: ClipId, uid: u64, pos: (f64, f64), size: (f64, f64), color: [f32; 4], opacity: f64, radius: f64) {
        self.with_clip(clip, |it| {
            let mut e = new_shape_layer(0, Vec2::new(pos.0, pos.1), Vec2::new(size.0, size.1), Vec::new());
            e.params.insert("name".into(), Param::new(ParamValue::Text(format!("Shape {uid}"))));
            e.params.insert("fill_color".into(), Param::new(ParamValue::Color(color)));
            e.params.insert("opacity".into(), Param::new(ParamValue::Float(opacity)));
            e.params.insert("corner_radius".into(), Param::new(ParamValue::Float(radius)));
            e.layer = Some(Box::new(LayerExtra { uid, ..Default::default() }));
            it.effects.push(e);
        });
    }

    /// An Elements sticker or animated text preset (`text.title-pop`) as a graphic clip.
    pub(crate) fn element(&mut self, track: usize, start: Tick, dur: Tick, id: &str) -> ClipId {
        let Some(t) = filmcraft_project::elements::all().into_iter().find(|t| filmcraft_project::elements::element_id(t) == id) else { return ClipId(0) };
        let (layers, meta) = t.instantiate((W, H));
        let g = self.graphic("Template Graphics");
        let c = self.clip(TrackKind::Video, track, g, start, dur);
        self.with_clip(c, |it| {
            it.name = t.name.clone();
            it.effects.extend(layers);
            it.graphic = Some(Box::new(meta));
        });
        c
    }

    /// Set the text of a layer.
    pub(crate) fn layer_text(&mut self, clip: ClipId, uid: u64, text: &str) {
        self.with_clip(clip, |it| {
            if let Some(e) = it.effects.iter_mut().find(|e| filmcraft_project::gtemplate::layer_uid(e) == uid) {
                e.params.insert("text".into(), Param::new(ParamValue::Text(text.into())));
            }
        });
    }

    /// Set a text layer's size; a text centred by its anchor stays centred.
    pub(crate) fn text_size(&mut self, clip: ClipId, uid: u64, size: f64) {
        self.with_clip(clip, |it| {
            let Some(e) = it.effects.iter_mut().find(|e| filmcraft_project::gtemplate::layer_uid(e) == uid) else { return };
            let old = e.params.get("size").and_then(|p| p.value.as_f64()).filter(|x| *x > 0.0);
            if let (Some(old), Some(a)) = (old, e.params.get_mut("anchor")) {
                let k = size / old;
                a.map_values(|v| match v {
                    ParamValue::Vec2(q) if !q.x.is_nan() => ParamValue::Vec2(Vec2::new(q.x * k, q.y * k)),
                    o => o,
                });
            }
            e.params.insert("size".into(), Param::new(ParamValue::Float(size)));
        });
    }

    /// Move a layer (every position keyframe) by `(dx, dy)` design pixels.
    pub(crate) fn shift_layer(&mut self, clip: ClipId, uid: u64, dx: f64, dy: f64) {
        self.with_clip(clip, |it| {
            if let Some(p) = it.effects.iter_mut().find(|e| filmcraft_project::gtemplate::layer_uid(e) == uid).and_then(|e| e.params.get_mut("position")) {
                p.map_values(|v| match v {
                    ParamValue::Vec2(q) if !q.x.is_nan() => ParamValue::Vec2(Vec2::new(q.x + dx, q.y + dy)),
                    o => o,
                });
            }
        });
    }

    /// An Elements sound effect on audio track `track` at `at` (a stand-in the engine swaps for
    /// the synthesized sound).
    pub(crate) fn sfx(&mut self, track: usize, at: Tick, id: &str) -> ClipId {
        let Some((_, s)) = SOUND_SECONDS.iter().find(|(i, _)| *i == id) else { return ClipId(0) };
        // whole frames, never longer than the sound
        let len = RATE.tick_of(RATE.frame_at(Tick::from_seconds_f64(*s)).max(1));
        let item = match self.stand_ins.get(id) {
            Some(i) => *i,
            None => {
                let i = sound_stand_in(&mut self.p, id, id, len);
                self.stand_ins.insert(id.to_string(), i);
                i
            }
        };
        self.clip(TrackKind::Audio, track, item, at, len)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transition(
        &mut self,
        kind: TrackKind,
        track: usize,
        effect: &str,
        from: Option<ClipId>,
        to: Option<ClipId>,
        dur: Tick,
        align: TransitionAlign,
    ) {
        self.transition_ex(kind, track, effect, from, to, dur, align, &[]);
    }

    /// A transition with parameters (a GL transition's `easing`, its own settings).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transition_ex(
        &mut self,
        kind: TrackKind,
        track: usize,
        effect: &str,
        from: Option<ClipId>,
        to: Option<ClipId>,
        dur: Tick,
        align: TransitionAlign,
        params: &[(&str, ParamValue)],
    ) {
        let Some(def) = find_effect(effect) else { return };
        let id = TransitionId(self.p.alloc_id());
        let Some(q) = self.p.sequence_mut(self.seq) else { return };
        let at = match (from.and_then(|c| q.find_item(c)), to.and_then(|c| q.find_item(c))) {
            (Some((_, a)), _) if align == TransitionAlign::EndAtCut || to.is_none() => a.end(),
            (_, Some((_, b))) => b.start,
            (Some((_, a)), None) => a.end(),
            _ => return,
        };
        let start = match align {
            TransitionAlign::CenterAtCut => at - Tick(dur.0 / 2),
            TransitionAlign::StartAtCut => at,
            TransitionAlign::EndAtCut => at - dur,
        };
        let mut effect = def.instance();
        for (k, v) in params {
            effect.params.insert((*k).to_string(), Param::new(v.clone()));
        }
        if let Some(t) = q.tracks_mut(kind).get_mut(track) {
            t.transitions.push(Transition { id, effect, start, duration: dur, from, to, align, reverse: false, sound: None });
            t.sort();
        }
    }

    /// A chapter marker on the sequence; with `text`, its name follows that text slot's text
    /// (`text:<slot>:<media index>` in the comment).
    pub(crate) fn chapter(&mut self, at: Tick, name: &str, text: Option<(&str, u32)>) {
        let id = MarkerId(self.p.alloc_id());
        let comment = text.map(|(s, i)| format!("{}{s}:{i}", crate::apply::TEXT_MARKER)).unwrap_or_default();
        if let Some(q) = self.p.sequence_mut(self.seq) {
            q.markers.push(Marker { id, start: at, duration: Tick::ZERO, name: name.into(), comment, kind: MarkerKind::Chapter, color: Label::Mango });
            q.markers.sort_by_key(|m| m.start);
        }
    }

    /// A caption track with placeholder captions `(start s, end s, text)` (a captions slot's
    /// preview; replaced by the transcript of the user's media).
    pub(crate) fn captions(&mut self, cues: &[(f64, f64, &str)]) {
        let track = TrackId(self.p.alloc_id());
        let ids: Vec<ClipId> = cues.iter().map(|_| ClipId(self.p.alloc_id())).collect();
        let Some(q) = self.p.sequence_mut(self.seq) else { return };
        let mut t = CaptionTrack::new(track, "Captions".into(), CaptionFormat::Subtitle);
        t.captions = cues
            .iter()
            .zip(ids)
            .map(|((a, b, text), id)| {
                let start = secs(*a);
                let end = secs(*b).max(start + frames(1));
                Caption { id, start, duration: end - start, text: (*text).into(), speaker: None, cue_id: None, settings: String::new(), words: Vec::new() }
            })
            .collect();
        q.caption_tracks.push(t);
    }

    pub(crate) fn finish(self, mut m: Manifest) -> Result<Package, TemplateError> {
        let dur = self.p.sequence(self.seq).map(|q| q.duration()).ok_or_else(|| TemplateError::Apply("the template has no sequence".into()))?;
        m.sequence = self.seq.0;
        m.duration = dur;
        Package::new(m, self.p)
    }
}

pub(crate) fn manifest(id: &str, name: &str, category: &str, description: &str, tags: &[&str]) -> Manifest {
    Manifest {
        format: FORMAT.into(),
        format_version: FORMAT_VERSION,
        id: format!("builtin:{id}"),
        name: name.into(),
        description: description.into(),
        version: "1.0.0".into(),
        author: AUTHOR.into(),
        license: LICENSE.into(),
        category: category.into(),
        tags: tags.iter().map(|t| t.to_string()).collect(),
        sequence_file: SEQUENCE_FILE.into(),
        sequence: 0,
        duration: Tick::ZERO,
        variants: Vec::new(),
        default_variant: "16:9".into(),
        slots: Vec::new(),
        poster: None,
        look: None,
        caption_style: None,
        export_preset: None,
        beat_grid: None,
        assets: Vec::new(),
    }
}

/// One aspect layout: `16:9`, `9:16`, `1:1` or `4:5`.
pub(crate) fn variant(id: &str, hints: Vec<Hint>) -> Variant {
    let (name, w, h, preset) = match id {
        "9:16" => ("Vertical (Shorts, Reels, TikTok)", 1080, 1920, VERTICAL),
        "1:1" => ("Square (feed)", 1080, 1080, MATCH_SOURCE),
        "4:5" => ("Portrait 4:5 (Instagram feed)", 1080, 1350, MATCH_SOURCE),
        _ => ("Landscape (YouTube)", W, H, YOUTUBE),
    };
    Variant { id: id.into(), name: name.into(), width: w, height: h, export_preset: Some(preset.into()), hints }
}

/// 16:9 and 9:16 (with the vertical hints).
pub(crate) fn variants(vertical_hints: Vec<Hint>) -> Vec<Variant> {
    vec![variant("16:9", Vec::new()), variant("9:16", vertical_hints)]
}

/// All four layouts.
pub(crate) fn variants4(tall: Vec<Hint>, square: Vec<Hint>, four_five: Vec<Hint>) -> Vec<Variant> {
    vec![variant("16:9", Vec::new()), variant("9:16", tall), variant("1:1", square), variant("4:5", four_five)]
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn slot(id: &str, kind: SlotKind, name: &str, prompt: &str, order: u32, count: u32, min_count: u32, placements: Vec<Placement>) -> Slot {
    Slot {
        id: id.into(),
        kind,
        name: name.into(),
        prompt: prompt.into(),
        order,
        count,
        min_count,
        placements,
        constraints: Constraints::default(),
        fit: default_fit(kind),
        beat_sync: false,
        default: None,
        captions: None,
        ai: None,
    }
}

/// A text slot of one or more placements (`(clip, layer)`, media index in turn), at most
/// `max_chars` characters.
pub(crate) fn text_slot(id: &str, name: &str, prompt: &str, order: u32, places: &[(ClipId, u64)], max_chars: u32) -> Slot {
    let count = u32::try_from(places.len()).unwrap_or(1).max(1);
    let pls = places.iter().enumerate().map(|(i, (c, l))| text_place(*c, u32::try_from(i).unwrap_or(0), *l)).collect();
    let mut s = slot(id, SlotKind::Text, name, prompt, order, count, 0, pls);
    s.constraints.max_chars = Some(max_chars);
    s
}

/// The captions slot: what `source` slots say, `max_chars` a line.
pub(crate) fn captions_slot(order: u32, source: &[&str], max_chars: u32, lines: u32) -> Slot {
    let mut s = slot(
        "captions",
        SlotKind::Captions,
        "Auto captions",
        "Captions of what is said, transcribed when the template is applied (needs a speech model: Transcribe…)",
        order,
        1,
        0,
        Vec::new(),
    );
    s.captions = Some(CaptionSpec { source: source.iter().map(|x| x.to_string()).collect(), max_chars, lines });
    s
}

/// A music slot with a generated default bed.
pub(crate) fn music_slot(order: u32, clip: ClipId, prompt: &str, mood: &str, genre: &str, bpm: f64, seed: u64) -> Slot {
    let mut s = slot("music", SlotKind::Music, "Music", prompt, order, 1, 0, vec![place(clip, 0)]);
    s.default = Some(SlotDefault::Generate { mood: mood.into(), bpm, seed, genre: Some(genre.into()) });
    s
}

pub(crate) fn place(clip: ClipId, media: u32) -> Placement {
    Placement { clip: clip.0, media, layer: None, area: None }
}

pub(crate) fn boxed(clip: ClipId, media: u32, area: [f64; 4]) -> Placement {
    Placement { clip: clip.0, media, layer: None, area: Some(area) }
}

pub(crate) fn text_place(clip: ClipId, media: u32, layer: u64) -> Placement {
    Placement { clip: clip.0, media, layer: Some(layer), area: None }
}

/// Layer positions (fractions) and text sizes (pixels; 0 = keep) of a graphic clip in a layout.
pub(crate) fn layers_hint(clip: ClipId, layers: &[(u64, [f64; 2], f64)]) -> Hint {
    Hint {
        clip: clip.0,
        position: None,
        scale: None,
        frame: None,
        area: None,
        layers: layers.iter().map(|(l, p, s)| LayerHint { layer: *l, position: Some(*p), size: (*s > 0.0).then_some(*s) }).collect(),
    }
}

/// A `text.title-pop` title at height `y` (fraction) in a layout, `size` pixels, its rule `gap`
/// (fraction of the frame height) under it.
pub(crate) fn title_hint(clip: ClipId, y: f64, size: f64, gap: f64) -> Hint {
    layers_hint(clip, &[(1, [0.5, y], size), (2, [0.5, y + gap], 0.0)])
}

/// Text sizes (pixels in the layout) of a graphic clip's layers, their places kept.
pub(crate) fn sizes_hint(clip: ClipId, layers: &[(u64, f64)]) -> Hint {
    Hint {
        clip: clip.0,
        position: None,
        scale: None,
        frame: None,
        area: None,
        layers: layers.iter().map(|(l, s)| LayerHint { layer: *l, position: None, size: Some(*s) }).collect(),
    }
}

pub(crate) fn layer_hint(clip: ClipId, layer: u64, pos: [f64; 2], size: f64) -> Hint {
    layers_hint(clip, &[(layer, pos, size)])
}

/// A picture's box in a layout.
pub(crate) fn box_hint(clip: ClipId, area: [f64; 4]) -> Hint {
    Hint { clip: clip.0, position: None, scale: None, frame: None, area: Some(area), layers: Vec::new() }
}

pub(crate) fn generate(mood: &str, bpm: f64, seed: u64) -> Option<SlotDefault> {
    Some(SlotDefault::Generate { mood: mood.into(), bpm, seed, genre: None })
}

/// A caption style for templates that caption speech (sizes for a 1080-line frame; vertical
/// layouts scale them with the frame height, so lines stay short).
pub(crate) fn speech_captions(size: f32) -> CaptionStyle {
    CaptionStyle {
        size,
        background: true,
        background_color: [12, 12, 20, 200],
        outline: 0.0,
        anchor: CaptionAnchor::Bottom,
        align: CaptionAlign::Center,
        margin: 0.1,
        ..Default::default()
    }
}

/// A picture slot cut on the beat: `fit` center portion, stretched (no slower than half speed).
pub(crate) fn beat_slot(mut s: Slot, min: Tick, max: Tick) -> Slot {
    s.beat_sync = true;
    s.fit = Fit { short: ShortFit::Stretch, portion: Portion::Center, frame: Framing::Fill, min_speed: 0.5 };
    s.constraints = Constraints { min_duration: Some(min), max_duration: Some(max), ..Default::default() };
    s
}

/// A long talking slot (a person speaking): the start of the clip, held on its last frame if it
/// is shorter than the template.
pub(crate) fn speech_slot(mut s: Slot) -> Slot {
    s.fit = Fit { short: ShortFit::Hold, portion: Portion::Start, frame: Framing::Fill, min_speed: 0.5 };
    s
}

pub(crate) fn steady_grid(bpm: f64, until: Tick) -> Option<BeatGrid> {
    Some(BeatGrid::steady(bpm, Tick::ZERO, until))
}

/// Every starter, in library order: (id without `builtin:`, builder).
type Make = fn() -> Result<Package, TemplateError>;
const STARTERS: [(&str, Make); 21] = [
    ("photo-beat-slideshow", classic::photo_beat_slideshow),
    ("channel-intro", classic::channel_intro),
    ("top-3-listicle", classic::top_3_listicle),
    ("talking-head-captions", creator::talking_head_captions),
    ("product-promo-15", social::product_promo_15),
    ("travel-vlog-intro", stories::travel_vlog_intro),
    ("travel-vlog-outro", stories::travel_vlog_outro),
    ("top-5-listicle", social::top_5_listicle),
    ("before-after-reveal", social::before_after_reveal),
    ("lower-third-chapter-kit", creator::lower_third_chapter_kit),
    ("end-card-subscribe", creator::end_card_subscribe),
    ("tutorial-pip", creator::tutorial_pip),
    ("podcast-two-speakers", creator::podcast_two_speakers),
    ("photo-collage", social::photo_collage),
    ("quote-card", social::quote_card),
    ("countdown", stories::countdown),
    ("event-recap", stories::event_recap),
    ("day-in-my-life", stories::day_in_my_life),
    ("unboxing", social::unboxing),
    ("qa-session", creator::qa_session),
    ("robot-showcase", stories::robot_showcase),
];

/// The built-in templates (each checked like a file from disk; one that would fail is left out).
pub fn builtin_templates() -> Vec<Package> {
    shared().to_vec()
}

/// The built-in templates, built once.
pub fn shared() -> &'static [Package] {
    static ALL: std::sync::OnceLock<Vec<Package>> = std::sync::OnceLock::new();
    ALL.get_or_init(|| STARTERS.iter().filter_map(|(_, make)| make().ok()).collect())
}

/// A built-in template by id (`builtin:…`) or name.
pub fn find(key: &str) -> Option<&'static Package> {
    shared().iter().find(|p| p.manifest.id == key || p.manifest.name.eq_ignore_ascii_case(key))
}

/// Ids of the built-in templates, in library order.
pub const BUILTIN_IDS: [&str; 21] = [
    "builtin:photo-beat-slideshow",
    "builtin:channel-intro",
    "builtin:top-3-listicle",
    "builtin:talking-head-captions",
    "builtin:product-promo-15",
    "builtin:travel-vlog-intro",
    "builtin:travel-vlog-outro",
    "builtin:top-5-listicle",
    "builtin:before-after-reveal",
    "builtin:lower-third-chapter-kit",
    "builtin:end-card-subscribe",
    "builtin:tutorial-pip",
    "builtin:podcast-two-speakers",
    "builtin:photo-collage",
    "builtin:quote-card",
    "builtin:countdown",
    "builtin:event-recap",
    "builtin:day-in-my-life",
    "builtin:unboxing",
    "builtin:qa-session",
    "builtin:robot-showcase",
];

/// Build every built-in template, with the errors of any that does not validate (for tests).
pub fn build_all() -> Vec<(&'static str, Result<Package, TemplateError>)> {
    STARTERS.iter().map(|(id, make)| (*id, make())).collect()
}
