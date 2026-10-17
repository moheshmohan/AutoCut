//! The built-in starter templates: original designs made for FilmCraft in code. Their
//! placeholder media is generated (the procedural demo scenes and colour mattes of
//! `filmcraft-media`, graphics drawn with the bundled Inter font), never stock footage, and an
//! unfilled music slot gets a bed generated in the app — so they need no asset files at all.
//!
//! | id | what | slots |
//! |---|---|---|
//! | `builtin:photo-beat-slideshow` | 8 photos cut on the beat with a slow push-in, title and end text | photos ×8, music, title, outro |
//! | `builtin:channel-intro` | a 5 s channel intro: logo pop over a moving background, channel name and tagline | logo, name, tagline, background, music |
//! | `builtin:top-3-listicle` | "top 3" countdown: title card, then #3, #2, #1 each with a number card and a clip | items ×3, labels ×3, title, music |
//!
//! Each comes in 16:9 (1920×1080) and 9:16 (1080×1920).

use filmcraft_geom::Vec2;
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{DemoScene, Generator, MediaKind, MediaSource};
use filmcraft_project::graphic::new_text_layer;
use filmcraft_project::graphic_design::LayerExtra;
use filmcraft_project::{
    CaptionAlign, CaptionAnchor, CaptionStyle, ClipId, EffectInstance, ItemId, ItemKind, Keyframe, Label, MediaClip, MediaRef, Param, ParamValue, Project,
    SequenceSettings, TrackKind, Transition, TransitionAlign, TransitionId, find_effect,
};
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick, TimeRange};

use crate::manifest::{
    BeatGrid, Constraints, Fit, Framing, Hint, LayerHint, Look, Manifest, Placement, Portion, ShortFit, Slot, SlotDefault, SlotKind, Variant, default_fit,
};
use crate::{FORMAT, FORMAT_VERSION, Package, SEQUENCE_FILE, TemplateError};

const RATE: FrameRate = FrameRate::FPS_30;
const W: u32 = 1920;
const H: u32 = 1080;
const YOUTUBE: &str = "YouTube 1080p Full HD";
const VERTICAL: &str = "Social Vertical 1080×1920";
const AUTHOR: &str = "FilmCraft contributors";
const LICENSE: &str = "MIT OR Apache-2.0";

fn frames(n: i64) -> Tick {
    RATE.tick_of(n)
}

/// Builds a template's sequence file.
struct Builder {
    p: Project,
    seq: ItemId,
}

impl Builder {
    fn new(name: &str, video: usize, audio: usize) -> Builder {
        let mut p = Project::new(name);
        let settings = SequenceSettings { width: W, height: H, frame_rate: RATE, preset: "Template 1080p 30".into(), ..Default::default() };
        let seq = p.new_sequence(name, settings, video, audio, None);
        Builder { p, seq }
    }

    fn generator(&mut self, name: &str, g: Generator, size: (u32, u32), label: Label) -> ItemId {
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

    /// A generated stand-in for music (sound only).
    fn music_placeholder(&mut self) -> ItemId {
        let id = self.generator("Placeholder Music", Generator::Demo(DemoScene::Aurora), (W, H), Label::Caribbean);
        if let Some(m) = self.p.item_mut(id).and_then(|i| i.as_media_mut()) {
            m.info.video = None;
            m.info.kind = MediaKind::AudioOnly;
        }
        id
    }

    fn graphic(&mut self, name: &str) -> ItemId {
        self.p.add_item(name, Label::Rose, ItemKind::Graphic { width: W, height: H, rate: RATE }, None)
    }

    fn clip(&mut self, kind: TrackKind, track: usize, item: ItemId, start: Tick, dur: Tick) -> ClipId {
        let Some(mut it) = self.p.make_track_item(item, kind, start, TimeRange::new(Tick::ZERO, dur), RATE) else { return ClipId(0) };
        it.duration = dur;
        let id = it.id;
        if let Some(t) = self.p.sequence_mut(self.seq).and_then(|q| q.tracks_mut(kind).get_mut(track)) {
            t.items.push(it);
            t.sort();
        }
        id
    }

    fn with_clip(&mut self, clip: ClipId, f: impl FnOnce(&mut filmcraft_project::TrackItem)) {
        if let Some((_, it)) = self.p.sequence_mut(self.seq).and_then(|q| q.find_item_mut(clip)) {
            f(it);
        }
    }

    /// Animate `effect.param` from `a` to `b` over the clip's media time `[t0, t1]`.
    fn ramp(&mut self, clip: ClipId, effect: &str, param: &str, keys: &[(Tick, f64)]) {
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

    fn fade_in_out(&mut self, clip: ClipId, dur: Tick, fade: Tick) {
        self.ramp(clip, "opacity", "opacity", &[(Tick::ZERO, 0.0), (fade, 100.0), (dur - fade, 100.0), (dur, 0.0)]);
    }

    fn text(&mut self, clip: ClipId, uid: u64, text: &str, pos: (f64, f64), size: f64, style: &str, color: [f32; 4]) {
        self.with_clip(clip, |it| {
            let mut e = new_text_layer(text, Vec2::new(pos.0, pos.1), size);
            let set = |e: &mut EffectInstance, k: &str, v: ParamValue| {
                e.params.insert(k.to_string(), Param::new(v));
            };
            set(&mut e, "name", ParamValue::Text(format!("Text {uid}")));
            set(&mut e, "font", ParamValue::Text("Inter".into()));
            set(&mut e, "font_style", ParamValue::Text(style.into()));
            set(&mut e, "align", ParamValue::Choice(1));
            set(&mut e, "fill_color", ParamValue::Color(color));
            set(&mut e, "shadow", ParamValue::Bool(true));
            set(&mut e, "shadow_distance", ParamValue::Float(4.0));
            set(&mut e, "shadow_blur", ParamValue::Float(18.0));
            set(&mut e, "shadow_opacity", ParamValue::Float(55.0));
            e.layer = Some(Box::new(LayerExtra { uid, ..Default::default() }));
            it.effects.push(e);
        });
    }

    fn transition(&mut self, kind: TrackKind, track: usize, effect: &str, from: Option<ClipId>, to: Option<ClipId>, dur: Tick, align: TransitionAlign) {
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
        if let Some(t) = q.tracks_mut(kind).get_mut(track) {
            t.transitions.push(Transition { id, effect: def.instance(), start, duration: dur, from, to, align, reverse: false });
            t.sort();
        }
    }

    fn finish(self, mut m: Manifest) -> Result<Package, TemplateError> {
        let dur = self.p.sequence(self.seq).map(|q| q.duration()).ok_or_else(|| TemplateError::Apply("the template has no sequence".into()))?;
        m.sequence = self.seq.0;
        m.duration = dur;
        Package::new(m, self.p)
    }
}

fn white() -> [f32; 4] {
    [1.0, 1.0, 1.0, 1.0]
}

fn manifest(id: &str, name: &str, category: &str, description: &str, tags: &[&str]) -> Manifest {
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
        look: None,
        caption_style: None,
        export_preset: None,
        beat_grid: None,
        assets: Vec::new(),
    }
}

fn variants(vertical_hints: Vec<Hint>) -> Vec<Variant> {
    vec![
        Variant { id: "16:9".into(), name: "Landscape (YouTube)".into(), width: W, height: H, export_preset: Some(YOUTUBE.into()), hints: Vec::new() },
        Variant {
            id: "9:16".into(),
            name: "Vertical (Shorts, Reels)".into(),
            width: H,
            height: W,
            export_preset: Some(VERTICAL.into()),
            hints: vertical_hints,
        },
    ]
}

fn slot(id: &str, kind: SlotKind, name: &str, prompt: &str, order: u32, count: u32, min_count: u32, placements: Vec<Placement>) -> Slot {
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
    }
}

fn place(clip: ClipId, media: u32) -> Placement {
    Placement { clip: clip.0, media, layer: None }
}

fn text_place(clip: ClipId, media: u32, layer: u64) -> Placement {
    Placement { clip: clip.0, media, layer: Some(layer) }
}

fn layer_hint(clip: ClipId, layer: u64, pos: [f64; 2], size: f64) -> Hint {
    Hint { clip: clip.0, position: None, scale: None, frame: None, layers: vec![LayerHint { layer, position: Some(pos), size: Some(size) }] }
}

fn generate(mood: &str, bpm: f64, seed: u64) -> Option<SlotDefault> {
    Some(SlotDefault::Generate { mood: mood.into(), bpm, seed })
}

/// 8 photos, each 4 beats at 120 bpm, a slow push-in alternating with a pull-out, a flash at
/// the midpoint, a title over the first photo and an end line over the last.
fn photo_beat_slideshow() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Photo Beat Slideshow", 3, 2);
    let beat = frames(15);
    let each = frames(60);
    let scenes = DemoScene::ALL;
    let mut photos = Vec::new();
    for k in 0..8u32 {
        let scene = scenes.get(k as usize % scenes.len()).copied().unwrap_or(DemoScene::OceanSunset);
        let item = b.generator(&format!("Photo {}", k + 1), Generator::Demo(scene), (W, H), Label::Iris);
        let c = b.clip(TrackKind::Video, 0, item, Tick(each.0 * i64::from(k)), each);
        let (a, z) = if k % 2 == 0 { (100.0, 108.0) } else { (108.0, 100.0) };
        b.ramp(c, "motion", "scale", &[(Tick::ZERO, a), (each, z)]);
        photos.push(c);
    }
    let end = Tick(each.0 * 8);
    if let (Some(&a), Some(&z)) = (photos.get(3), photos.get(4)) {
        b.transition(TrackKind::Video, 0, "flash", Some(a), Some(z), frames(6), TransitionAlign::CenterAtCut);
    }
    if let Some(&last) = photos.last() {
        b.transition(TrackKind::Video, 0, "dip_to_black", Some(last), None, beat, TransitionAlign::EndAtCut);
    }
    let g = b.graphic("Slideshow Text");
    let title = b.clip(TrackKind::Video, 1, g, Tick::ZERO, each);
    b.text(title, 1, "Summer Memories", (960.0, 900.0), 110.0, "Bold", white());
    b.fade_in_out(title, each, frames(10));
    let outro = b.clip(TrackKind::Video, 1, g, end - each, each);
    b.text(outro, 1, "Thanks for watching", (960.0, 900.0), 84.0, "SemiBold", white());
    b.fade_in_out(outro, each, frames(10));
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 0, music_item, Tick::ZERO, end);
    b.transition(TrackKind::Audio, 0, "constant_power", Some(music), None, frames(30), TransitionAlign::EndAtCut);

    let mut m = manifest(
        "photo-beat-slideshow",
        "Photo Beat Slideshow",
        "Reels & Shorts",
        "Eight photos cut on the beat of the music, each with a slow push-in, a title over the first and an end line over the last.",
        &["slideshow", "photos", "beat", "music"],
    );
    let mut photos_slot = slot(
        "photos",
        SlotKind::Photo,
        "Photos",
        "Up to 8 photos (or clips), best first; fewer are reused in turn",
        0,
        8,
        1,
        photos.iter().enumerate().map(|(i, c)| place(*c, i as u32)).collect(),
    );
    photos_slot.beat_sync = true;
    photos_slot.constraints = Constraints { min_duration: Some(beat), max_duration: Some(frames(150)), ..Default::default() };
    let mut music_slot = slot("music", SlotKind::Music, "Music", "A song; the photos change on its beat", 1, 1, 0, vec![place(music, 0)]);
    music_slot.default = generate("uplifting", 120.0, 7);
    let mut title_slot = slot("title", SlotKind::Text, "Title", "Shown over the first photo", 2, 1, 0, vec![text_place(title, 0, 1)]);
    title_slot.constraints.max_chars = Some(40);
    let mut outro_slot = slot("outro", SlotKind::Text, "End line", "Shown over the last photo", 3, 1, 0, vec![text_place(outro, 0, 1)]);
    outro_slot.constraints.max_chars = Some(40);
    m.slots = vec![photos_slot, music_slot, title_slot, outro_slot];
    m.variants = variants(vec![layer_hint(title, 1, [0.5, 0.72], 96.0), layer_hint(outro, 1, [0.5, 0.72], 72.0)]);
    m.look = Some(Look::Lumetri("Golden Dusk".into()));
    m.caption_style = Some(CaptionStyle {
        size: 64.0,
        background: false,
        outline: 3.0,
        anchor: CaptionAnchor::Bottom,
        align: CaptionAlign::Center,
        margin: 0.12,
        ..Default::default()
    });
    m.beat_grid = Some(BeatGrid::steady(120.0, Tick::ZERO, end));
    b.finish(m)
}

/// A 5 s channel intro: a moving background, the logo popping in on the first beat, the channel
/// name and a tagline fading up under it, a dip to black at the end.
fn channel_intro() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Channel Intro", 3, 2);
    let total = frames(150);
    let bg_item = b.generator("Background Placeholder", Generator::Demo(DemoScene::Plasma), (W, H), Label::Iris);
    let bg = b.clip(TrackKind::Video, 0, bg_item, Tick::ZERO, total);
    b.transition(TrackKind::Video, 0, "dip_to_black", Some(bg), None, frames(15), TransitionAlign::EndAtCut);
    let logo_item = b.generator("Logo Placeholder", Generator::ColorMatte { color: [0.98, 0.76, 0.18, 1.0] }, (600, 600), Label::Mango);
    let logo_start = frames(15);
    let logo_len = total - logo_start;
    let logo = b.clip(TrackKind::Video, 1, logo_item, logo_start, logo_len);
    b.with_clip(logo, |it| {
        if let Some(pos) = it.effects.iter_mut().find(|e| e.effect == "motion").and_then(|m| m.params.get_mut("position")) {
            pos.value = ParamValue::Vec2(Vec2::new(960.0, 450.0));
        }
    });
    b.ramp(logo, "motion", "scale", &[(Tick::ZERO, 24.0), (frames(9), 44.0), (frames(14), 40.0)]);
    b.ramp(logo, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(6), 100.0)]);
    let g = b.graphic("Intro Text");
    let text_start = frames(30);
    let names = b.clip(TrackKind::Video, 2, g, text_start, total - text_start);
    b.text(names, 1, "My Channel", (960.0, 800.0), 120.0, "Bold", white());
    b.text(names, 2, "New videos every week", (960.0, 900.0), 54.0, "Medium", [0.9, 0.92, 1.0, 1.0]);
    b.ramp(names, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(12), 100.0)]);
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 0, music_item, Tick::ZERO, total);
    b.transition(TrackKind::Audio, 0, "constant_power", Some(music), None, frames(15), TransitionAlign::EndAtCut);

    let mut m = manifest(
        "channel-intro",
        "Channel Intro",
        "YouTube",
        "A 5 second channel intro: your logo pops in over a moving background, your channel name and tagline fade up under it.",
        &["intro", "logo", "youtube", "channel"],
    );
    let mut logo_slot = slot("logo", SlotKind::Logo, "Logo", "Your logo (a square image works best; transparency is kept)", 0, 1, 1, vec![place(logo, 0)]);
    logo_slot.fit = Fit { short: ShortFit::Hold, frame: Framing::Fit, ..Fit::default() };
    let mut name_slot = slot("name", SlotKind::Text, "Channel name", "", 1, 1, 0, vec![text_place(names, 0, 1)]);
    name_slot.constraints.max_chars = Some(32);
    let mut tag_slot = slot("tagline", SlotKind::Text, "Tagline", "A short line under the name", 2, 1, 0, vec![text_place(names, 0, 2)]);
    tag_slot.constraints.max_chars = Some(48);
    let mut bg_slot = slot("background", SlotKind::Video, "Background", "Optional: a clip to play behind the logo", 3, 1, 0, vec![place(bg, 0)]);
    bg_slot.fit = Fit { short: ShortFit::Loop, portion: Portion::Center, ..Fit::default() };
    let mut music_slot = slot("music", SlotKind::Music, "Music", "Optional: a sting or the start of a song", 4, 1, 0, vec![place(music, 0)]);
    music_slot.fit = Fit { short: ShortFit::Loop, frame: Framing::None, ..Fit::default() };
    music_slot.default = generate("epic", 120.0, 3);
    m.slots = vec![logo_slot, name_slot, tag_slot, bg_slot, music_slot];
    m.variants = variants(vec![
        Hint { clip: logo.0, position: Some([0.5, 0.38]), scale: Some(140.0), frame: None, layers: Vec::new() },
        Hint {
            clip: names.0,
            position: None,
            scale: None,
            frame: None,
            layers: vec![
                LayerHint { layer: 1, position: Some([0.5, 0.6]), size: Some(100.0) },
                LayerHint { layer: 2, position: Some([0.5, 0.65]), size: Some(46.0) },
            ],
        },
    ]);
    b.finish(m)
}

/// A countdown: a title card, then for #3, #2 and #1 a number card with its label followed by
/// the clip, all on a 120 bpm grid; zooms into the clips and pushes out of them.
fn top_3_listicle() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Top 3 Listicle", 3, 2);
    let (title_len, card, item_len) = (frames(60), frames(45), frames(120));
    let matte = b.generator("Card Background", Generator::ColorMatte { color: [0.07, 0.09, 0.16, 1.0] }, (W, H), Label::Lavender);
    let g = b.graphic("Countdown Cards");
    let scenes = [DemoScene::CityNight, DemoScene::Dunes, DemoScene::Forest];
    let mut at = Tick::ZERO;
    let title_bg = b.clip(TrackKind::Video, 0, matte, at, title_len);
    let title = b.clip(TrackKind::Video, 1, g, at, title_len);
    b.text(title, 1, "Top 3 Moments", (960.0, 540.0), 140.0, "Bold", white());
    b.fade_in_out(title, title_len, frames(8));
    at += title_len;
    let mut prev_v1 = title_bg;
    let (mut items, mut labels) = (Vec::new(), Vec::new());
    for (k, rank) in [3u32, 2, 1].iter().enumerate() {
        let card_bg = b.clip(TrackKind::Video, 0, matte, at, card);
        b.transition(
            TrackKind::Video,
            0,
            "push",
            Some(prev_v1),
            Some(card_bg),
            frames(10),
            if k == 0 { TransitionAlign::CenterAtCut } else { TransitionAlign::EndAtCut },
        );
        let card_text = b.clip(TrackKind::Video, 1, g, at, card);
        b.text(card_text, 1, &format!("#{rank}"), (960.0, 470.0), 300.0, "Black", [0.98, 0.76, 0.18, 1.0]);
        let label = ["Number three", "Number two", "Number one"].get(k).copied().unwrap_or("");
        b.text(card_text, 2, label, (960.0, 690.0), 76.0, "SemiBold", white());
        b.fade_in_out(card_text, card, frames(6));
        at += card;
        let scene = scenes.get(k).copied().unwrap_or(DemoScene::Aurora);
        let item = b.generator(&format!("Clip #{rank}"), Generator::Demo(scene), (W, H), Label::Iris);
        let c = b.clip(TrackKind::Video, 0, item, at, item_len);
        b.transition(TrackKind::Video, 0, "cross_zoom", Some(card_bg), Some(c), frames(9), TransitionAlign::StartAtCut);
        at += item_len;
        prev_v1 = c;
        items.push((c, 2 - k as u32));
        labels.push((card_text, 2 - k as u32));
    }
    b.transition(TrackKind::Video, 0, "dip_to_black", Some(prev_v1), None, frames(15), TransitionAlign::EndAtCut);
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 0, music_item, Tick::ZERO, at);
    b.transition(TrackKind::Audio, 0, "constant_power", Some(music), None, frames(30), TransitionAlign::EndAtCut);

    let mut m = manifest(
        "top-3-listicle",
        "Top 3 Listicle",
        "Reels & Shorts",
        "A countdown: a title card, then #3, #2 and #1, each with a number card and its clip, cut on a 120 bpm grid.",
        &["listicle", "top 3", "countdown", "ranking"],
    );
    let mut items_slot =
        slot("items", SlotKind::Video, "Clips", "Three clips, the best (#1) first", 0, 3, 1, items.iter().map(|(c, mi)| place(*c, *mi)).collect());
    items_slot.beat_sync = true;
    items_slot.fit = Fit { short: ShortFit::Stretch, portion: Portion::Center, frame: Framing::Fill, min_speed: 0.5 };
    items_slot.constraints = Constraints { min_duration: Some(frames(60)), max_duration: Some(frames(240)), ..Default::default() };
    let mut labels_slot =
        slot("labels", SlotKind::Text, "Labels", "What each clip is, #1 first", 1, 3, 0, labels.iter().map(|(c, mi)| text_place(*c, *mi, 2)).collect());
    labels_slot.constraints.max_chars = Some(32);
    let mut title_slot = slot("title", SlotKind::Text, "Title", "", 2, 1, 0, vec![text_place(title, 0, 1)]);
    title_slot.constraints.max_chars = Some(32);
    let mut music_slot = slot("music", SlotKind::Music, "Music", "A song; the cuts follow its beat", 3, 1, 0, vec![place(music, 0)]);
    music_slot.default = generate("uplifting", 120.0, 11);
    m.slots = vec![items_slot, labels_slot, title_slot, music_slot];
    let mut hints = vec![layer_hint(title, 1, [0.5, 0.5], 112.0)];
    for (c, _) in &labels {
        hints.push(Hint {
            clip: c.0,
            position: None,
            scale: None,
            frame: None,
            layers: vec![
                LayerHint { layer: 1, position: Some([0.5, 0.42]), size: Some(260.0) },
                LayerHint { layer: 2, position: Some([0.5, 0.55]), size: Some(68.0) },
            ],
        });
    }
    m.variants = variants(hints);
    m.look = Some(Look::Lumetri("Boost Vibrance".into()));
    m.caption_style = Some(CaptionStyle { size: 58.0, background: true, background_color: [10, 12, 24, 200], margin: 0.1, ..Default::default() });
    m.beat_grid = Some(BeatGrid::steady(120.0, Tick::ZERO, at));
    b.finish(m)
}

/// The built-in templates (each checked like a file from disk; one that would fail is left out).
pub fn builtin_templates() -> Vec<Package> {
    static ALL: std::sync::OnceLock<Vec<Package>> = std::sync::OnceLock::new();
    ALL.get_or_init(|| [photo_beat_slideshow(), channel_intro(), top_3_listicle()].into_iter().filter_map(Result::ok).collect()).clone()
}

/// Ids of the built-in templates, in library order.
pub const BUILTIN_IDS: [&str; 3] = ["builtin:photo-beat-slideshow", "builtin:channel-intro", "builtin:top-3-listicle"];

/// Build every built-in template, with the errors of any that does not validate (for tests).
pub fn build_all() -> Vec<(&'static str, Result<Package, TemplateError>)> {
    vec![("photo-beat-slideshow", photo_beat_slideshow()), ("channel-intro", channel_intro()), ("top-3-listicle", top_3_listicle())]
}
