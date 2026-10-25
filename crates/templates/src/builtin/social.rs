//! Short social starters: a 15 s product promo, a "top 5", a before/after reveal, a photo collage,
//! a quote card and an unboxing.

use filmcraft_geom::Vec2;
use filmcraft_media::{DemoScene, Generator};
use filmcraft_project::{ClipId, Label, ParamValue, TrackKind, TransitionAlign};
use filmcraft_time::Tick;

use super::{
    Builder, Hint, beat_slot, box_hint, boxed, frames, layer_hint, layers_hint, manifest, music_slot, place, rgb, secs, sizes_hint, slot, steady_grid,
    text_slot, title_hint, variant, white,
};
use crate::manifest::{Fit, Framing, Look, Portion, ShortFit, SlotKind};
use crate::{Package, TemplateError};

const F: fn(f64) -> ParamValue = ParamValue::Float;

/// A pop: from nothing to a little past full size and back, over 10 frames.
fn pop(b: &mut Builder, c: ClipId, full: f64) {
    b.ramp(c, "motion", "scale", &[(Tick::ZERO, 0.0), (frames(6), full * 1.08), (frames(10), full)]);
}

/// 15 s on a 120 bpm grid: the product's name with the logo, four shots with a feature caption
/// each, cut on the beat with GL transitions and whooshes, then an offer card.
pub(super) fn product_promo_15() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Product Promo 15 s", 5, 3);
    let brand = b.matte("Brand Background", rgb(0x111827));
    let logo_item = b.generator("Logo Placeholder", Generator::ColorMatte { color: [0.98, 0.76, 0.18, 1.0] }, (600, 600), Label::Mango);
    let logo_box = [0.44, 0.08, 0.12, 0.213333];
    // the title card
    let title_len = secs(2.0);
    let title_bg = b.clip(TrackKind::Video, 0, brand, Tick::ZERO, title_len);
    let logo1 = b.clip(TrackKind::Video, 1, logo_item, Tick::ZERO, title_len);
    pop(&mut b, logo1, 100.0);
    let name = b.element(2, Tick::ZERO, title_len, "text.title-pop");
    b.layer_text(name, 1, "AURA ONE");
    b.shift_layer(name, 1, 0.0, 60.0);
    b.shift_layer(name, 2, 0.0, 60.0);
    b.sfx(1, Tick::ZERO, "sound.pop");
    // four shots, 2 s (4 beats) each, a feature caption over each
    let features = ["All-day battery", "Fits in a pocket", "Water resistant", "Made to last"];
    let scenes = [DemoScene::OceanSunset, DemoScene::Plasma, DemoScene::Forest, DemoScene::CityNight];
    let (mut shots, mut captions) = (Vec::new(), Vec::new());
    let mut prev = title_bg;
    for k in 0..4usize {
        let at = title_len + secs(2.0 * k as f64);
        let item = b.scene(&format!("Shot {}", k + 1), scenes.get(k).copied().unwrap_or(DemoScene::Aurora));
        let c = b.clip(TrackKind::Video, 0, item, at, secs(2.0));
        let (a, z) = if k.is_multiple_of(2) { (100.0, 110.0) } else { (110.0, 100.0) };
        b.ramp(c, "motion", "scale", &[(Tick::ZERO, a), (secs(2.0), z)]);
        let gl = ["gl_circleopen", "gl_directionalwarp", "gl_simplezoom", "gl_crosszoom"].get(k).copied().unwrap_or("gl_crosszoom");
        b.transition_ex(TrackKind::Video, 0, gl, Some(prev), Some(c), frames(8), TransitionAlign::CenterAtCut, &[("easing", ParamValue::Choice(9))]);
        if k > 0 {
            b.sfx(2, at - frames(9), "sound.swish");
        }
        let cap = b.element(2, at + frames(6), secs(1.8), "text.caption-box");
        b.layer_text(cap, 1, features.get(k).copied().unwrap_or(""));
        shots.push(c);
        captions.push(cap);
        prev = c;
    }
    // the offer card
    let offer_at = secs(10.0);
    let offer_len = secs(5.0);
    let offer_bg = b.clip(TrackKind::Video, 0, brand, offer_at, offer_len);
    b.transition(TrackKind::Video, 0, "flash", Some(prev), Some(offer_bg), frames(6), TransitionAlign::CenterAtCut);
    b.transition(TrackKind::Video, 0, "dip_to_black", Some(offer_bg), None, frames(15), TransitionAlign::EndAtCut);
    let logo2 = b.clip(TrackKind::Video, 1, logo_item, offer_at, offer_len);
    pop(&mut b, logo2, 100.0);
    let burst = b.element(3, offer_at + frames(6), offer_len - frames(6), "sticker.burst-callout");
    b.layer_text(burst, 2, "20% OFF");
    b.text_size(burst, 2, 120.0);
    b.set(burst, "motion", "position", ParamValue::Vec2(Vec2::new(960.0, 560.0)));
    pop(&mut b, burst, 100.0);
    let g = b.graphic("Template Graphics");
    let cta = b.clip(TrackKind::Video, 4, g, offer_at + frames(12), offer_len - frames(12));
    b.text(cta, 1, "Shop now at aura.example", (960.0, 900.0), 56.0, "SemiBold", white());
    b.ramp(cta, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(8), 100.0)]);
    b.sfx(1, offer_at, "sound.impact-soft");
    b.sfx(2, offer_at + frames(6), "sound.pop");
    let end = offer_at + offer_len;
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 0, music_item, Tick::ZERO, end);
    b.transition(TrackKind::Audio, 0, "constant_power", Some(music), None, frames(20), TransitionAlign::EndAtCut);

    let mut m = manifest(
        "product-promo-15",
        "Product Promo 15 s",
        "Business",
        "Fifteen seconds for a product: its name with your logo, four shots with a feature each, cut on the beat with GL transitions and whooshes, then the offer.",
        &["product", "promo", "ad", "commercial", "beat", "15 seconds"],
    );
    let shots_slot = beat_slot(
        slot(
            "shots",
            SlotKind::Video,
            "Product shots",
            "Four clips or photos of the product, in order",
            0,
            4,
            1,
            shots.iter().enumerate().map(|(i, c)| place(*c, i as u32)).collect(),
        ),
        secs(1.0),
        secs(4.0),
    );
    let mut logo_slot =
        slot("logo", SlotKind::Logo, "Logo", "Your logo (transparency is kept)", 1, 1, 1, vec![boxed(logo1, 0, logo_box), boxed(logo2, 0, logo_box)]);
    logo_slot.fit = Fit { short: ShortFit::Hold, frame: Framing::Fit, ..Fit::default() };
    let cap_places: Vec<(ClipId, u64)> = captions.iter().map(|c| (*c, 1)).collect();
    m.slots = vec![
        shots_slot,
        logo_slot,
        text_slot("product", "Product name", "", 2, &[(name, 1)], 16),
        text_slot("features", "Features", "One line for each shot", 3, &cap_places, 28),
        text_slot("offer", "Offer", "The big line on the last card", 4, &[(burst, 2)], 12),
        text_slot("cta", "Call to action", "Where to buy", 5, &[(cta, 1)], 36),
        music_slot(6, music, "A song; the shots change on its beat", "energetic", "electronic", 120.0, 71),
    ];
    let logo_tall = [0.39, 0.1, 0.22, 0.12375];
    let mut tall = vec![
        title_hint(name, 0.5, 110.0, 0.035),
        box_hint(logo1, logo_tall),
        box_hint(logo2, logo_tall),
        layer_hint(cta, 1, [0.5, 0.74], 48.0),
        sizes_hint(burst, &[(2, 96.0)]),
    ];
    let logo_square = [0.39, 0.06, 0.22, 0.22];
    let square = vec![
        title_hint(name, 0.52, 96.0, 0.06),
        box_hint(logo1, logo_square),
        box_hint(logo2, logo_square),
        layer_hint(cta, 1, [0.5, 0.86], 44.0),
        sizes_hint(burst, &[(2, 84.0)]),
    ];
    let logo_45 = [0.39, 0.06, 0.22, 0.176];
    let mut four_five = square.clone();
    four_five[1] = box_hint(logo1, logo_45);
    four_five[2] = box_hint(logo2, logo_45);
    for c in &captions {
        tall.push(sizes_hint(*c, &[(1, 44.0)]));
    }
    m.variants = vec![variant("9:16", tall), variant("16:9", Vec::new()), variant("1:1", square), variant("4:5", four_five)];
    m.default_variant = "9:16".into();
    m.look = Some(Look::Lumetri("Boost Vibrance".into()));
    m.beat_grid = steady_grid(120.0, end);
    m.poster = Some(secs(3.0));
    b.finish(m)
}

/// A "top 5": a title card, then for #5 … #1 a number card with its label and the clip; GL
/// transitions and a pop on every card, on a 120 bpm grid.
pub(super) fn top_5_listicle() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Top 5 Listicle", 3, 2);
    let (title_len, card, item_len) = (secs(2.0), secs(1.0), secs(3.0));
    let matte = b.matte("Card Background", rgb(0x1e1b4b));
    let g = b.graphic("Countdown Cards");
    let mut at = Tick::ZERO;
    let title_bg = b.clip(TrackKind::Video, 0, matte, at, title_len);
    let title = b.clip(TrackKind::Video, 1, g, at, title_len);
    b.text(title, 1, "Top 5 Gadgets", (960.0, 590.0), 150.0, "Bold", white());
    b.fade_in_out(title, title_len, frames(8));
    b.sfx(1, Tick::ZERO, "sound.whoosh-long");
    at += title_len;
    let mut prev = title_bg;
    let (mut items, mut labels) = (Vec::new(), Vec::new());
    let names = ["Number five", "Number four", "Number three", "Number two", "Number one"];
    for k in 0..5usize {
        let rank = 5 - k;
        let card_bg = b.clip(TrackKind::Video, 0, matte, at, card);
        b.transition_ex(TrackKind::Video, 0, "gl_cube", Some(prev), Some(card_bg), frames(10), TransitionAlign::EndAtCut, &[("easing", ParamValue::Choice(9))]);
        let card_text = b.clip(TrackKind::Video, 1, g, at, card);
        b.text(card_text, 1, &format!("#{rank}"), (960.0, 560.0), 300.0, "Black", rgb(0xfacc15));
        b.text(card_text, 2, names.get(k).copied().unwrap_or(""), (960.0, 760.0), 76.0, "SemiBold", white());
        b.fade_in_out(card_text, card, frames(5));
        b.sfx(1, at, "sound.pop");
        at += card;
        let item = b.nth_scene(&format!("Clip #{rank}"), k + 2);
        let c = b.clip(TrackKind::Video, 0, item, at, item_len);
        b.ramp(c, "motion", "scale", &[(Tick::ZERO, 100.0), (item_len, 106.0)]);
        b.transition(TrackKind::Video, 0, "gl_simplezoom", Some(card_bg), Some(c), frames(9), TransitionAlign::StartAtCut);
        at += item_len;
        prev = c;
        items.push((c, (4 - k) as u32));
        labels.push((card_text, (4 - k) as u32));
    }
    b.transition(TrackKind::Video, 0, "dip_to_black", Some(prev), None, frames(15), TransitionAlign::EndAtCut);
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 0, music_item, Tick::ZERO, at);
    b.transition(TrackKind::Audio, 0, "constant_power", Some(music), None, frames(30), TransitionAlign::EndAtCut);

    let mut m = manifest(
        "top-5-listicle",
        "Top 5 Listicle",
        "Reels & Shorts",
        "A countdown of five: a title card, then #5 to #1, each a number card with its label followed by the clip, cube and zoom transitions on a 120 bpm grid.",
        &["listicle", "top 5", "countdown", "ranking"],
    );
    let items_slot = beat_slot(
        slot("items", SlotKind::Video, "Clips", "Five clips, the best (#1) first", 0, 5, 1, items.iter().map(|(c, mi)| place(*c, *mi)).collect()),
        secs(2.0),
        secs(8.0),
    );
    let mut labels_slot =
        slot("labels", SlotKind::Text, "Labels", "What each clip is, #1 first", 1, 5, 0, labels.iter().map(|(c, mi)| super::text_place(*c, *mi, 2)).collect());
    labels_slot.constraints.max_chars = Some(28);
    m.slots = vec![
        items_slot,
        labels_slot,
        text_slot("title", "Title", "", 2, &[(title, 1)], 24),
        music_slot(3, music, "A song; the cuts follow its beat", "energetic", "pop", 120.0, 81),
    ];
    let mut tall = vec![layer_hint(title, 1, [0.5, 0.52], 110.0)];
    let mut square = vec![layer_hint(title, 1, [0.5, 0.55], 96.0)];
    for (c, _) in &labels {
        tall.push(layers_hint(*c, &[(1, [0.5, 0.48], 260.0), (2, [0.5, 0.58], 68.0)]));
        square.push(layers_hint(*c, &[(1, [0.5, 0.5], 220.0), (2, [0.5, 0.66], 60.0)]));
    }
    m.variants = vec![variant("16:9", Vec::new()), variant("9:16", tall), variant("1:1", square.clone()), variant("4:5", square)];
    m.look = Some(Look::Lumetri("Boost Vibrance".into()));
    m.beat_grid = steady_grid(120.0, at);
    m.poster = Some(secs(2.5));
    b.finish(m)
}

/// Before, a wipe to after, then both side by side (stacked in the vertical and square
/// layouts), labelled, with the title under them.
pub(super) fn before_after_reveal() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Before / After", 3, 2);
    let before_item = b.scene("Before Placeholder", DemoScene::Dunes);
    let after_item = b.scene("After Placeholder", DemoScene::Forest);
    let cut = secs(3.5);
    let split_at = secs(7.0);
    let total = secs(10.0);
    let before = b.clip(TrackKind::Video, 0, before_item, Tick::ZERO, cut);
    let after = b.clip(TrackKind::Video, 0, after_item, cut, split_at - cut);
    b.transition_ex(TrackKind::Video, 0, "linear_wipe", Some(before), Some(after), secs(1.0), TransitionAlign::CenterAtCut, &[]);
    let left = b.clip(TrackKind::Video, 0, before_item, split_at, total - split_at);
    let right = b.clip(TrackKind::Video, 1, after_item, split_at, total - split_at);
    b.transition(TrackKind::Video, 1, "dip_to_white", Some(right), None, frames(12), TransitionAlign::EndAtCut);
    let dark = rgb(0x111827);
    let g = b.graphic("Template Graphics");
    let tag_a = b.clip(TrackKind::Video, 2, g, Tick::ZERO, cut);
    b.tag(tag_a, 1, "BEFORE", (190.0, 95.0), 40.0, "Bold", white(), dark, 85.0);
    let tag_b = b.clip(TrackKind::Video, 2, g, cut, split_at - cut);
    b.tag(tag_b, 1, "AFTER", (190.0, 95.0), 40.0, "Bold", white(), dark, 85.0);
    let split = b.clip(TrackKind::Video, 2, g, split_at, total - split_at);
    b.tag(split, 1, "BEFORE", (190.0, 95.0), 40.0, "Bold", white(), dark, 85.0);
    b.tag(split, 2, "AFTER", (1150.0, 95.0), 40.0, "Bold", white(), dark, 85.0);
    b.tag(split, 3, "Room makeover", (960.0, 985.0), 60.0, "Bold", rgb(0x111827), white(), 92.0);
    b.ramp(split, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(8), 100.0)]);
    b.sfx(1, cut - secs(0.6), "sound.sweep");
    b.sfx(1, split_at, "sound.pop");
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 0, music_item, Tick::ZERO, total);
    b.volume(music, -6.0);
    b.transition(TrackKind::Audio, 0, "constant_power", Some(music), None, frames(30), TransitionAlign::EndAtCut);

    let mut m = manifest(
        "before-after-reveal",
        "Before / After",
        "Reels & Shorts",
        "Show the change: the before, a wipe to the after, then both side by side (stacked when vertical) with labels and a title.",
        &["before after", "transformation", "reveal", "makeover", "comparison", "split screen"],
    );
    let mut before_slot =
        slot("before", SlotKind::Photo, "Before", "The before photo or clip", 0, 1, 1, vec![place(before, 0), boxed(left, 0, [0.0, 0.0, 0.5, 1.0])]);
    before_slot.fit = Fit { short: ShortFit::Hold, portion: Portion::Start, ..Fit::default() };
    let mut after_slot = slot(
        "after",
        SlotKind::Photo,
        "After",
        "The after photo or clip (framed like the before)",
        1,
        1,
        1,
        vec![place(after, 0), boxed(right, 0, [0.5, 0.0, 0.5, 1.0])],
    );
    after_slot.fit = Fit { short: ShortFit::Hold, portion: Portion::Start, ..Fit::default() };
    m.slots = vec![
        before_slot,
        after_slot,
        text_slot("labels", "Labels", "The two labels", 2, &[(tag_a, 1), (tag_b, 1)], 16),
        text_slot("split_labels", "Side-by-side labels", "The labels over the comparison", 3, &[(split, 1), (split, 2)], 16),
        text_slot("title", "Title", "Under the comparison", 4, &[(split, 3)], 32),
        music_slot(5, music, "Optional: a song", "uplifting", "pop", 110.0, 91),
    ];
    let stacked = |title: f64| -> Vec<Hint> {
        vec![
            box_hint(left, [0.0, 0.0, 1.0, 0.5]),
            box_hint(right, [0.0, 0.5, 1.0, 0.5]),
            layers_hint(split, &[(1, [0.2, 0.05], 0.0), (2, [0.2, 0.55], 0.0), (3, [0.5, title], 0.0)]),
        ]
    };
    m.variants = vec![variant("9:16", stacked(0.93)), variant("16:9", Vec::new()), variant("1:1", stacked(0.9)), variant("4:5", stacked(0.92))];
    m.default_variant = "9:16".into();
    m.poster = Some(secs(8.0));
    b.finish(m)
}

/// Four photos pop into a 2×2 grid one after the other, then a title over them.
pub(super) fn photo_collage() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Photo Collage", 6, 2);
    let total = secs(10.0);
    let paper = b.matte("Paper", rgb(0xf5f5f4));
    b.clip(TrackKind::Video, 0, paper, Tick::ZERO, total);
    let cells = [[0.02, 0.03, 0.47, 0.455], [0.51, 0.03, 0.47, 0.455], [0.02, 0.515, 0.47, 0.455], [0.51, 0.515, 0.47, 0.455]];
    let mut photos = Vec::new();
    for k in 0..4usize {
        let start = secs(0.3 + 0.45 * k as f64);
        let len = total - start;
        let item = b.nth_scene(&format!("Photo {}", k + 1), k);
        let c = b.clip(TrackKind::Video, 1 + k, item, start, len);
        pop(&mut b, c, 100.0);
        b.ramp(c, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(4), 100.0), (len - frames(12), 100.0), (len, 0.0)]);
        b.sfx(1, start, "sound.pop");
        photos.push(c);
    }
    let title = b.element(5, secs(4.5), total - secs(4.5), "text.caption-box");
    b.layer_text(title, 1, "Our weekend");
    b.text_size(title, 1, 72.0);
    b.shift_layer(title, 1, 0.0, -390.0);
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 0, music_item, Tick::ZERO, total);
    b.transition(TrackKind::Audio, 0, "constant_power", Some(music), None, frames(30), TransitionAlign::EndAtCut);

    let mut m = manifest(
        "photo-collage",
        "Photo Collage",
        "Social",
        "Four photos pop into a 2×2 grid one after the other, then a title appears over them.",
        &["collage", "grid", "photos", "album", "square"],
    );
    let mut photos_slot = slot(
        "photos",
        SlotKind::Photo,
        "Photos",
        "Four photos (or clips); fewer are reused",
        0,
        4,
        1,
        photos.iter().zip(cells).enumerate().map(|(i, (c, cell))| boxed(*c, i as u32, cell)).collect(),
    );
    photos_slot.fit = Fit { short: ShortFit::Loop, portion: Portion::Center, ..Fit::default() };
    m.slots = vec![
        photos_slot,
        text_slot("title", "Title", "Over the grid", 1, &[(title, 1)], 28),
        music_slot(2, music, "Optional: a song", "happy", "pop", 110.0, 101),
    ];
    let tall = vec![sizes_hint(title, &[(1, 64.0)])];
    let square = vec![sizes_hint(title, &[(1, 56.0)])];
    m.variants = vec![variant("1:1", square.clone()), variant("16:9", Vec::new()), variant("9:16", tall), variant("4:5", square)];
    m.default_variant = "1:1".into();
    m.poster = Some(secs(5.5));
    b.finish(m)
}

/// A quote and its author over a darkened, slowly pushing-in photo.
pub(super) fn quote_card() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Quote Card", 3, 1);
    let total = secs(8.0);
    let item = b.scene("Background Placeholder", DemoScene::Aurora);
    let bg = b.clip(TrackKind::Video, 0, item, Tick::ZERO, total);
    b.ramp(bg, "motion", "scale", &[(Tick::ZERO, 100.0), (total, 108.0)]);
    b.effect(bg, "gaussian_blur", &[("blurriness", F(6.0)), ("repeat_edge", ParamValue::Bool(true))]);
    let shade_item = b.matte("Shade", [0.0, 0.0, 0.0, 1.0]);
    let shade = b.clip(TrackKind::Video, 1, shade_item, Tick::ZERO, total);
    b.opacity(shade, 55.0);
    let g = b.graphic("Template Graphics");
    let card = b.clip(TrackKind::Video, 2, g, Tick::ZERO, total);
    b.text(card, 1, "\u{201c}", (960.0, 400.0), 260.0, "Black", rgb(0xfacc15));
    let wrap = [("box_width", F(1300.0)), ("shadow", ParamValue::Bool(true))];
    b.text_ex(card, 2, "Make it simple, but significant.", (310.0, 430.0), 72.0, "SemiBold", white(), 1, &wrap);
    b.text(card, 3, "\u{2014} Your Name", (960.0, 820.0), 42.0, "Medium", rgb(0xd1d5db));
    b.fade_in_out(card, total, frames(15));
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 0, music_item, Tick::ZERO, total);
    b.volume(music, -10.0);
    b.transition(TrackKind::Audio, 0, "constant_power", Some(music), None, frames(30), TransitionAlign::EndAtCut);

    let mut m = manifest(
        "quote-card",
        "Quote Card",
        "Social",
        "A quote and its author over a darkened photo that slowly pushes in.",
        &["quote", "text", "inspiration", "square", "card"],
    );
    let mut bg_slot = slot("background", SlotKind::Photo, "Background", "A photo or clip behind the quote", 0, 1, 0, vec![place(bg, 0)]);
    bg_slot.fit = Fit { short: ShortFit::Loop, portion: Portion::Center, ..Fit::default() };
    m.slots = vec![
        bg_slot,
        text_slot("quote", "Quote", "", 1, &[(card, 2)], 140),
        text_slot("author", "Author", "Who said it", 2, &[(card, 3)], 40),
        music_slot(3, music, "Optional: music", "calm", "piano", 70.0, 111),
    ];
    let vertical = vec![sizes_hint(card, &[(1, 180.0), (2, 56.0), (3, 34.0)])];
    m.variants = vec![variant("1:1", vertical.clone()), variant("16:9", Vec::new()), variant("9:16", vertical.clone()), variant("4:5", vertical)];
    m.default_variant = "1:1".into();
    m.look = Some(Look::Lumetri("Faded Print".into()));
    m.poster = Some(secs(3.0));
    b.finish(m)
}

/// Package, opening, the reveal (a flash, a burst and a hit), two details and a verdict card, on
/// a 120 bpm grid with the clips' own sound.
pub(super) fn unboxing() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Unboxing", 4, 4);
    let parts = [("package", 0.0, 2.0, DemoScene::Dunes), ("opening", 2.0, 3.0, DemoScene::OceanSunset), ("reveal", 5.0, 3.0, DemoScene::Plasma)];
    let mut clips = Vec::new();
    for (name, at, len, scene) in parts {
        let item = b.scene(&format!("{name} placeholder"), scene);
        let c = b.av(0, 0, item, secs(at), secs(len));
        b.ramp(c, "motion", "scale", &[(Tick::ZERO, 100.0), (secs(len), 106.0)]);
        clips.push(c);
    }
    let details: Vec<ClipId> = (0..2)
        .map(|k| {
            let item = b.nth_scene(&format!("Detail {}", k + 1), k + 3);
            b.av(0, 0, item, secs(8.0 + 2.0 * k as f64), secs(2.0))
        })
        .collect();
    let verdict_at = secs(12.0);
    let verdict_len = secs(3.0);
    let ink = b.matte("Verdict Background", rgb(0x0f172a));
    let verdict_bg = b.clip(TrackKind::Video, 0, ink, verdict_at, verdict_len);
    let (package, opening, reveal) = (clips.first().copied(), clips.get(1).copied(), clips.get(2).copied());
    b.transition(TrackKind::Video, 0, "gl_directionalwarp", package, opening, frames(8), TransitionAlign::CenterAtCut);
    b.transition(TrackKind::Video, 0, "flash", opening, reveal, frames(6), TransitionAlign::CenterAtCut);
    b.transition(TrackKind::Video, 0, "gl_simplezoom", reveal, details.first().copied(), frames(8), TransitionAlign::CenterAtCut);
    b.transition(TrackKind::Video, 0, "gl_crosszoom", details.get(1).copied(), Some(verdict_bg), frames(10), TransitionAlign::CenterAtCut);
    b.transition(TrackKind::Video, 0, "dip_to_black", Some(verdict_bg), None, frames(12), TransitionAlign::EndAtCut);
    let hook = b.element(1, Tick::ZERO, secs(2.0), "text.title-pop");
    b.layer_text(hook, 1, "WHAT'S INSIDE?");
    b.text_size(hook, 1, 120.0);
    b.shift_layer(hook, 1, 0.0, -250.0);
    b.shift_layer(hook, 2, 0.0, -270.0);
    let burst = b.element(2, secs(5.0), secs(1.6), "sticker.burst-callout");
    b.layer_text(burst, 2, "IT'S HERE!");
    b.text_size(burst, 2, 70.0);
    b.set(burst, "motion", "position", ParamValue::Vec2(Vec2::new(1480.0, 300.0)));
    pop(&mut b, burst, 80.0);
    let star = b.element(1, verdict_at + frames(4), verdict_len - frames(4), "sticker.star");
    b.set(star, "motion", "position", ParamValue::Vec2(Vec2::new(960.0, 330.0)));
    pop(&mut b, star, 45.0);
    let g = b.graphic("Template Graphics");
    let verdict = b.clip(TrackKind::Video, 2, g, verdict_at + frames(8), verdict_len - frames(8));
    b.text(verdict, 1, "VERDICT", (960.0, 610.0), 48.0, "SemiBold", rgb(0xfacc15));
    b.text(verdict, 2, "9/10 \u{2014} worth it", (960.0, 740.0), 96.0, "Bold", white());
    b.ramp(verdict, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(8), 100.0)]);
    b.sfx(2, Tick::ZERO, "sound.pop");
    b.sfx(3, secs(5.0), "sound.impact-soft");
    b.sfx(2, secs(5.0) + frames(3), "sound.pop");
    b.sfx(2, verdict_at, "sound.pop");
    let end = verdict_at + verdict_len;
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 1, music_item, Tick::ZERO, end);
    b.volume(music, -8.0);
    b.transition(TrackKind::Audio, 1, "constant_power", Some(music), None, frames(20), TransitionAlign::EndAtCut);

    let mut m = manifest(
        "unboxing",
        "Unboxing",
        "Reels & Shorts",
        "The box, the opening, the reveal with a flash and a hit, two close-ups and your verdict, on the beat with the clips' own sound.",
        &["unboxing", "product", "review", "reveal", "haul"],
    );
    let one = |id: &str, name: &str, prompt: &str, order: u32, c: Option<ClipId>| {
        beat_slot(slot(id, SlotKind::Video, name, prompt, order, 1, 1, vec![place(c.unwrap_or(ClipId(0)), 0)]), secs(1.0), secs(6.0))
    };
    m.slots = vec![
        one("package", "The box", "The package before it is opened", 0, package),
        one("opening", "Opening", "Opening it", 1, opening),
        one("reveal", "The reveal", "The first look at what is inside", 2, reveal),
        beat_slot(
            slot("details", SlotKind::Video, "Close-ups", "Two close-ups", 3, 2, 1, details.iter().enumerate().map(|(i, c)| place(*c, i as u32)).collect()),
            secs(1.0),
            secs(4.0),
        ),
        text_slot("hook", "Hook", "The first words", 4, &[(hook, 1)], 18),
        text_slot("reveal_text", "Reveal shout", "On the burst", 5, &[(burst, 2)], 12),
        text_slot("verdict", "Verdict", "Your verdict", 6, &[(verdict, 2)], 24),
        music_slot(7, music, "A song; the cuts follow its beat", "happy", "electronic", 120.0, 121),
    ];
    let tall = vec![title_hint(hook, 0.2, 84.0, 0.03), sizes_hint(verdict, &[(1, 40.0), (2, 72.0)])];
    let square = vec![title_hint(hook, 0.2, 72.0, 0.05), sizes_hint(verdict, &[(1, 36.0), (2, 64.0)])];
    m.variants = vec![variant("9:16", tall), variant("16:9", Vec::new()), variant("1:1", square.clone()), variant("4:5", square)];
    m.default_variant = "9:16".into();
    m.beat_grid = steady_grid(120.0, end);
    m.poster = Some(secs(5.4));
    b.finish(m)
}
