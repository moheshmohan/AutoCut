//! The first three starters: a photo slideshow on the beat, a channel intro and a "top 3".

use filmcraft_geom::Vec2;
use filmcraft_media::{DemoScene, Generator};
use filmcraft_project::{CaptionAlign, CaptionAnchor, CaptionStyle, Label, ParamValue, TrackKind, TransitionAlign};
use filmcraft_time::Tick;

use super::{Builder, Hint, LayerHint, beat_slot, frames, generate, layer_hint, manifest, place, slot, steady_grid, text_place, variant, variants, white};
use crate::manifest::{Fit, Framing, Look, Portion, ShortFit, SlotKind};
use crate::{Package, TemplateError};

/// 8 photos, each 4 beats at 120 bpm, a slow push-in alternating with a pull-out, a flash at
/// the midpoint, a title over the first photo and an end line over the last.
pub(super) fn photo_beat_slideshow() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Photo Beat Slideshow", 3, 2);
    let beat = frames(15);
    let each = frames(60);
    let mut photos = Vec::new();
    for k in 0..8u32 {
        let item = b.nth_scene(&format!("Photo {}", k + 1), k as usize);
        let c = b.clip(TrackKind::Video, 0, item, Tick(each.0 * i64::from(k)), each);
        let (a, z) = if k.is_multiple_of(2) { (100.0, 108.0) } else { (108.0, 100.0) };
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
    let photos_slot = beat_slot(
        slot(
            "photos",
            SlotKind::Photo,
            "Photos",
            "Up to 8 photos (or clips), best first; fewer are reused in turn",
            0,
            8,
            1,
            photos.iter().enumerate().map(|(i, c)| place(*c, i as u32)).collect(),
        ),
        beat,
        frames(150),
    );
    let photos_slot = crate::manifest::Slot { fit: Fit::default(), ..photos_slot };
    let mut music_slot = slot("music", SlotKind::Music, "Music", "A song; the photos change on its beat", 1, 1, 0, vec![place(music, 0)]);
    music_slot.default = generate("uplifting", 120.0, 7);
    let mut title_slot = slot("title", SlotKind::Text, "Title", "Shown over the first photo", 2, 1, 0, vec![text_place(title, 0, 1)]);
    title_slot.constraints.max_chars = Some(40);
    let mut outro_slot = slot("outro", SlotKind::Text, "End line", "Shown over the last photo", 3, 1, 0, vec![text_place(outro, 0, 1)]);
    outro_slot.constraints.max_chars = Some(40);
    m.slots = vec![photos_slot, music_slot, title_slot, outro_slot];
    let tall = vec![layer_hint(title, 1, [0.5, 0.72], 96.0), layer_hint(outro, 1, [0.5, 0.72], 72.0)];
    let square = vec![layer_hint(title, 1, [0.5, 0.8], 80.0), layer_hint(outro, 1, [0.5, 0.8], 60.0)];
    m.variants = vec![variant("16:9", Vec::new()), variant("9:16", tall), variant("1:1", square.clone()), variant("4:5", square)];
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
    m.beat_grid = steady_grid(120.0, end);
    m.poster = Some(frames(20));
    b.finish(m)
}

/// A 5 s channel intro: a moving background, the logo popping in on the first beat, the channel
/// name and a tagline fading up under it, a dip to black at the end.
pub(super) fn channel_intro() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Channel Intro", 3, 3);
    let total = frames(150);
    let bg_item = b.generator("Background Placeholder", Generator::Demo(DemoScene::Plasma), (super::W, super::H), Label::Iris);
    let bg = b.clip(TrackKind::Video, 0, bg_item, Tick::ZERO, total);
    b.transition(TrackKind::Video, 0, "dip_to_black", Some(bg), None, frames(15), TransitionAlign::EndAtCut);
    let logo_item = b.generator("Logo Placeholder", Generator::ColorMatte { color: [0.98, 0.76, 0.18, 1.0] }, (600, 600), Label::Mango);
    let logo_start = frames(15);
    let logo_len = total - logo_start;
    let logo = b.clip(TrackKind::Video, 1, logo_item, logo_start, logo_len);
    b.set(logo, "motion", "position", ParamValue::Vec2(Vec2::new(960.0, 450.0)));
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
    // the logo lands with a swish and a soft hit
    b.sfx(1, Tick::ZERO, "sound.swish");
    b.sfx(2, logo_start, "sound.impact-soft");

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
        Hint { clip: logo.0, position: Some([0.5, 0.38]), scale: Some(140.0), frame: None, area: None, layers: Vec::new() },
        Hint {
            clip: names.0,
            position: None,
            scale: None,
            frame: None,
            area: None,
            layers: vec![
                LayerHint { layer: 1, position: Some([0.5, 0.6]), size: Some(100.0) },
                LayerHint { layer: 2, position: Some([0.5, 0.65]), size: Some(46.0) },
            ],
        },
    ]);
    m.poster = Some(frames(75));
    b.finish(m)
}

/// A countdown: a title card, then for #3, #2 and #1 a number card with its label followed by
/// the clip, all on a 120 bpm grid; zooms into the clips and pushes out of them.
pub(super) fn top_3_listicle() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Top 3 Listicle", 3, 2);
    let (title_len, card, item_len) = (frames(60), frames(45), frames(120));
    let matte = b.generator("Card Background", Generator::ColorMatte { color: [0.07, 0.09, 0.16, 1.0] }, (super::W, super::H), Label::Lavender);
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
        let item = b.scene(&format!("Clip #{rank}"), scene);
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
    let items_slot = beat_slot(
        slot("items", SlotKind::Video, "Clips", "Three clips, the best (#1) first", 0, 3, 1, items.iter().map(|(c, mi)| place(*c, *mi)).collect()),
        frames(60),
        frames(240),
    );
    let mut labels_slot =
        slot("labels", SlotKind::Text, "Labels", "What each clip is, #1 first", 1, 3, 0, labels.iter().map(|(c, mi)| text_place(*c, *mi, 2)).collect());
    labels_slot.constraints.max_chars = Some(32);
    let mut title_slot = slot("title", SlotKind::Text, "Title", "", 2, 1, 0, vec![text_place(title, 0, 1)]);
    title_slot.constraints.max_chars = Some(32);
    let mut music_slot = slot("music", SlotKind::Music, "Music", "A song; the cuts follow its beat", 3, 1, 0, vec![place(music, 0)]);
    music_slot.default = generate("uplifting", 120.0, 11);
    m.slots = vec![items_slot, labels_slot, title_slot, music_slot];
    let mut tall = vec![layer_hint(title, 1, [0.5, 0.5], 112.0)];
    let mut square = vec![layer_hint(title, 1, [0.5, 0.5], 96.0)];
    for (c, _) in &labels {
        tall.push(super::layers_hint(*c, &[(1, [0.5, 0.42], 260.0), (2, [0.5, 0.55], 68.0)]));
        square.push(super::layers_hint(*c, &[(1, [0.5, 0.42], 220.0), (2, [0.5, 0.6], 60.0)]));
    }
    m.variants = vec![variant("16:9", Vec::new()), variant("9:16", tall), variant("1:1", square.clone()), variant("4:5", square)];
    m.look = Some(Look::Lumetri("Boost Vibrance".into()));
    m.caption_style = Some(CaptionStyle { size: 58.0, background: true, background_color: [10, 12, 24, 200], margin: 0.1, ..Default::default() });
    m.beat_grid = steady_grid(120.0, at);
    m.poster = Some(frames(70));
    b.finish(m)
}
