//! Story starters: a travel vlog's intro and outro, a countdown, an event recap, a day in my life
//! and a robot (or tech) showcase.

use filmcraft_media::{DemoScene, Generator};
use filmcraft_project::{ClipId, Label, ParamValue, TrackKind, TransitionAlign};
use filmcraft_time::Tick;

use super::{
    Builder, Hint, beat_slot, box_hint, boxed, frames, layer_hint, layers_hint, manifest, music_slot, place, rgb, secs, sizes_hint, slot, steady_grid,
    text_slot, title_hint, variants4, white,
};
use crate::manifest::{Fit, Framing, Look, Portion, ShortFit, SlotKind};
use crate::{Package, TemplateError};

const F: fn(f64) -> ParamValue = ParamValue::Float;

/// Alternate a slow push-in and pull-out over a clip.
fn drift(b: &mut Builder, c: ClipId, k: usize, len: Tick, amount: f64) {
    let (a, z) = if k.is_multiple_of(2) { (100.0, 100.0 + amount) } else { (100.0 + amount, 100.0) };
    b.ramp(c, "motion", "scale", &[(Tick::ZERO, a), (len, z)]);
}

/// 10 s on a 120 bpm grid: three words punch in over fast cuts with GL transitions and
/// whooshes, then the place's name over the last, longest shot.
pub(super) fn travel_vlog_intro() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Travel Vlog Intro", 3, 2);
    let beat = frames(15);
    let gls = ["gl_directionalwarp", "gl_crosszoom", "gl_wind", "gl_swirl", "gl_dreamyzoom"];
    let mut at = Tick::ZERO;
    let mut clips: Vec<ClipId> = Vec::new();
    for (k, n) in [4i64, 2, 2, 2, 2, 8].into_iter().enumerate() {
        let len = Tick(beat.0 * n);
        let item = b.nth_scene(&format!("Clip {}", k + 1), k);
        let c = b.clip(TrackKind::Video, 0, item, at, len);
        drift(&mut b, c, k, len, 8.0);
        if let Some(&prev) = clips.last() {
            let gl = gls.get(k - 1).copied().unwrap_or("gl_crosszoom");
            let d = if k == 5 { frames(12) } else { frames(8) };
            b.transition_ex(TrackKind::Video, 0, gl, Some(prev), Some(c), d, TransitionAlign::CenterAtCut, &[("easing", ParamValue::Choice(9))]);
            b.sfx(1, at - frames(9), "sound.swish");
        }
        clips.push(c);
        at += len;
    }
    let total = at;
    b.transition(TrackKind::Video, 0, "dip_to_black", clips.last().copied(), None, frames(15), TransitionAlign::EndAtCut);
    let hook = b.element(1, Tick::ZERO, secs(3.2), "text.kinetic-words");
    for (uid, w) in [(1, "PACK"), (2, "YOUR"), (3, "BAGS")] {
        b.layer_text(hook, uid, w);
    }
    let place_c = b.element(1, secs(6.0), total - secs(6.0), "text.title-pop");
    b.layer_text(place_c, 1, "LISBON");
    let g = b.graphic("Template Graphics");
    let date = b.clip(TrackKind::Video, 2, g, secs(6.4), total - secs(6.4));
    b.text(date, 1, "Portugal \u{b7} Summer 2026", (960.0, 712.0), 48.0, "Medium", white());
    b.ramp(date, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(10), 100.0)]);
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 0, music_item, Tick::ZERO, total);
    b.transition(TrackKind::Audio, 0, "constant_power", Some(music), None, frames(20), TransitionAlign::EndAtCut);

    let mut m = manifest(
        "travel-vlog-intro",
        "Travel Vlog Intro",
        "Vlog",
        "A 10 second travel intro: three words punch in over fast cuts on the beat with GL transitions and whooshes, then the place's name over the last shot.",
        &["travel", "vlog", "intro", "trip", "beat", "fast cuts"],
    );
    m.slots = vec![
        beat_slot(
            slot(
                "clips",
                SlotKind::Video,
                "Clips",
                "Six clips of the trip; the last one gets the title",
                0,
                6,
                1,
                clips.iter().enumerate().map(|(i, c)| place(*c, i as u32)).collect(),
            ),
            beat,
            secs(6.0),
        ),
        text_slot("hook", "Three words", "One word each, punched in one after the other", 1, &[(hook, 1), (hook, 2), (hook, 3)], 12),
        text_slot("place", "Place", "Where you went", 2, &[(place_c, 1)], 18),
        text_slot("date", "Subtitle", "Under the place", 3, &[(date, 1)], 32),
        music_slot(4, music, "A song; the cuts follow its beat", "happy", "pop", 120.0, 131),
    ];
    let tall = vec![sizes_hint(hook, &[(1, 120.0), (2, 120.0), (3, 120.0)]), title_hint(place_c, 0.46, 110.0, 0.035), layer_hint(date, 1, [0.5, 0.535], 40.0)];
    let square = vec![sizes_hint(hook, &[(1, 100.0), (2, 100.0), (3, 100.0)]), title_hint(place_c, 0.46, 96.0, 0.06), layer_hint(date, 1, [0.5, 0.58], 36.0)];
    m.variants = variants4(tall, square.clone(), square);
    m.look = Some(Look::Lumetri("Golden Dusk".into()));
    m.beat_grid = steady_grid(120.0, total);
    m.poster = Some(secs(7.0));
    b.finish(m)
}

/// 10 s, slow: three clips crossfading, "thanks for coming along", where you go next and a heart.
pub(super) fn travel_vlog_outro() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Travel Vlog Outro", 4, 2);
    let total = secs(10.0);
    let spans = [(0.0, 3.5), (3.5, 7.0), (7.0, 10.0)];
    let mut clips: Vec<ClipId> = Vec::new();
    for (k, (a, z)) in spans.into_iter().enumerate() {
        let item = b.nth_scene(&format!("Clip {}", k + 1), k + 1);
        let len = secs(z - a);
        let c = b.clip(TrackKind::Video, 0, item, secs(a), len);
        drift(&mut b, c, k, len, 6.0);
        if let Some(&prev) = clips.last() {
            b.transition(TrackKind::Video, 0, "cross_dissolve", Some(prev), Some(c), secs(1.0), TransitionAlign::CenterAtCut);
        }
        clips.push(c);
    }
    b.transition(TrackKind::Video, 0, "dip_to_black", clips.last().copied(), None, secs(1.0), TransitionAlign::EndAtCut);
    let g = b.graphic("Template Graphics");
    let thanks = b.clip(TrackKind::Video, 1, g, secs(6.0), total - secs(6.0));
    b.text(thanks, 1, "Thanks for coming along", (960.0, 500.0), 92.0, "Bold", white());
    b.fade_in_out(thanks, total - secs(6.0), frames(15));
    let next = b.clip(TrackKind::Video, 2, g, secs(7.0), total - secs(7.0));
    b.text(next, 1, "Next stop: Tokyo", (960.0, 612.0), 50.0, "Medium", rgb(0xfde68a));
    b.fade_in_out(next, total - secs(7.0), frames(15));
    let heart = b.element(3, secs(7.5), total - secs(7.5), "sticker.heart");
    b.set(heart, "motion", "position", ParamValue::Vec2(filmcraft_geom::Vec2::new(960.0, 330.0)));
    b.ramp(heart, "motion", "scale", &[(Tick::ZERO, 0.0), (frames(6), 27.0), (frames(10), 25.0)]);
    b.sfx(1, secs(7.5), "sound.pop");
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 0, music_item, Tick::ZERO, total);
    b.transition(TrackKind::Audio, 0, "constant_power", Some(music), None, secs(2.0), TransitionAlign::EndAtCut);

    let mut m = manifest(
        "travel-vlog-outro",
        "Travel Vlog Outro",
        "Vlog",
        "A slow 10 second outro: three clips crossfading, a thank-you, where you are going next and a heart.",
        &["travel", "vlog", "outro", "ending", "calm"],
    );
    m.slots = vec![
        slot("clips", SlotKind::Video, "Clips", "Three calm clips to end on", 0, 3, 1, clips.iter().enumerate().map(|(i, c)| place(*c, i as u32)).collect()),
        text_slot("thanks", "Thank-you line", "", 1, &[(thanks, 1)], 40),
        text_slot("next", "Next stop", "Where you go next", 2, &[(next, 1)], 32),
        music_slot(3, music, "Optional: a song", "dreamy", "ambient", 80.0, 141),
    ];
    let tall = vec![sizes_hint(thanks, &[(1, 64.0)]), sizes_hint(next, &[(1, 40.0)])];
    let square = vec![sizes_hint(thanks, &[(1, 58.0)]), sizes_hint(next, &[(1, 36.0)])];
    m.variants = variants4(tall, square.clone(), square);
    m.look = Some(Look::Lumetri("Golden Dusk".into()));
    m.poster = Some(secs(8.0));
    b.finish(m)
}

/// 3, 2, 1, GO! over a blurred background with ticks, a riser and a boom, then the event's title.
pub(super) fn countdown() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Countdown", 4, 3);
    let total = secs(8.0);
    let bg_item = b.scene("Background Placeholder", DemoScene::Plasma);
    let bg = b.clip(TrackKind::Video, 0, bg_item, Tick::ZERO, total);
    b.effect(bg, "gaussian_blur", &[("blurriness", F(18.0)), ("repeat_edge", ParamValue::Bool(true))]);
    let shade_item = b.matte("Shade", [0.0, 0.0, 0.0, 1.0]);
    let shade = b.clip(TrackKind::Video, 1, shade_item, Tick::ZERO, total);
    b.opacity(shade, 40.0);
    b.element(2, Tick::ZERO, secs(4.0), "text.countdown");
    let title = b.element(3, secs(4.0), secs(4.0), "text.title-pop");
    b.layer_text(title, 1, "LAUNCH DAY");
    let g = b.graphic("Template Graphics");
    let sub = b.clip(TrackKind::Video, 2, g, secs(4.3), total - secs(4.3));
    b.text(sub, 1, "Live at 7 PM", (960.0, 712.0), 52.0, "Medium", white());
    b.ramp(sub, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(10), 100.0)]);
    for s in [0.0, 1.0, 2.0] {
        b.sfx(2, secs(s), "sound.click");
    }
    b.sfx(1, secs(1.0), "sound.riser-2s");
    b.sfx(2, secs(3.0), "sound.boom");
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 0, music_item, secs(4.0), secs(4.0));
    b.volume(music, -4.0);
    b.transition(TrackKind::Audio, 0, "constant_power", Some(music), None, frames(30), TransitionAlign::EndAtCut);

    let mut m = manifest(
        "countdown",
        "Countdown",
        "Events",
        "3, 2, 1, GO! over a blurred background, with ticks, a riser and a boom on GO, then your event's title.",
        &["countdown", "launch", "event", "timer", "reveal"],
    );
    let mut bg_slot = slot("background", SlotKind::Video, "Background", "Optional: a clip behind the countdown", 0, 1, 0, vec![place(bg, 0)]);
    bg_slot.fit = Fit { short: ShortFit::Loop, portion: Portion::Center, ..Fit::default() };
    m.slots = vec![
        bg_slot,
        text_slot("title", "Title", "Shown after GO", 1, &[(title, 1)], 16),
        text_slot("subtitle", "Subtitle", "Under the title", 2, &[(sub, 1)], 32),
        music_slot(3, music, "Optional: what plays after GO", "epic", "cinematic", 120.0, 151),
    ];
    let tall = vec![title_hint(title, 0.46, 110.0, 0.035), layer_hint(sub, 1, [0.5, 0.535], 44.0)];
    let square = vec![title_hint(title, 0.46, 96.0, 0.06), layer_hint(sub, 1, [0.5, 0.58], 40.0)];
    m.variants = variants4(tall, square.clone(), square);
    m.poster = Some(secs(1.4));
    b.finish(m)
}

/// 16 s on a 120 bpm grid: the event's name over the first clip, eleven more cut on every other
/// beat with glitch and zoom transitions, then the logo and a closing line.
pub(super) fn event_recap() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Event Recap", 4, 3);
    let mut clips: Vec<ClipId> = Vec::new();
    let mut at = Tick::ZERO;
    for k in 0..12usize {
        let len = if k == 0 { secs(2.0) } else { secs(1.0) };
        let item = b.nth_scene(&format!("Clip {}", k + 1), k);
        let c = b.clip(TrackKind::Video, 0, item, at, len);
        drift(&mut b, c, k, len, 6.0);
        if let Some(&prev) = clips.last() {
            let t = match k {
                4 => Some(("gl_glitchdisplace", frames(6))),
                7 => Some(("gl_pixelize", frames(6))),
                10 => Some(("gl_crosszoom", frames(8))),
                _ => None,
            };
            if let Some((gl, d)) = t {
                b.transition(TrackKind::Video, 0, gl, Some(prev), Some(c), d, TransitionAlign::CenterAtCut);
                b.sfx(2, at - frames(9), "sound.swish");
            }
        }
        clips.push(c);
        at += len;
    }
    let end_at = at;
    let end_len = secs(3.0);
    let total = end_at + end_len;
    let ink = b.matte("End Background", rgb(0x0b0f19));
    let end_bg = b.clip(TrackKind::Video, 0, ink, end_at, end_len);
    b.transition(TrackKind::Video, 0, "gl_filmburn", clips.last().copied(), Some(end_bg), frames(10), TransitionAlign::CenterAtCut);
    b.transition(TrackKind::Video, 0, "dip_to_black", Some(end_bg), None, frames(15), TransitionAlign::EndAtCut);
    let logo_item = b.generator("Logo Placeholder", Generator::ColorMatte { color: [0.98, 0.76, 0.18, 1.0] }, (600, 600), Label::Mango);
    let logo_box = [0.44, 0.2, 0.12, 0.213333];
    let logo = b.clip(TrackKind::Video, 1, logo_item, end_at + frames(6), end_len - frames(6));
    b.ramp(logo, "motion", "scale", &[(Tick::ZERO, 0.0), (frames(6), 108.0), (frames(10), 100.0)]);
    let event = b.element(1, Tick::ZERO, secs(2.0), "text.title-pop");
    b.layer_text(event, 1, "SUMMIT 2026");
    let g = b.graphic("Template Graphics");
    let date = b.clip(TrackKind::Video, 2, g, secs(0.3), secs(1.7));
    b.text(date, 1, "Lisbon \u{b7} 12\u{2013}14 May", (960.0, 712.0), 48.0, "Medium", white());
    b.ramp(date, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(8), 100.0)]);
    let outro = b.clip(TrackKind::Video, 2, g, end_at + frames(12), end_len - frames(12));
    b.text(outro, 1, "See you next year", (960.0, 760.0), 70.0, "Bold", white());
    b.ramp(outro, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(10), 100.0)]);
    b.sfx(1, secs(2.0), "sound.impact-soft");
    b.sfx(1, end_at, "sound.boom");
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 0, music_item, Tick::ZERO, total);
    b.transition(TrackKind::Audio, 0, "constant_power", Some(music), None, frames(30), TransitionAlign::EndAtCut);

    let mut m = manifest(
        "event-recap",
        "Event Recap",
        "Events",
        "Twelve clips of an event cut on the beat with glitch and zoom transitions, the event's name over the first, your logo and a closing line at the end.",
        &["event", "recap", "highlights", "conference", "party", "beat", "fast cuts"],
    );
    let mut logo_slot = slot("logo", SlotKind::Logo, "Logo", "The event's or your logo", 1, 1, 1, vec![boxed(logo, 0, logo_box)]);
    logo_slot.fit = Fit { short: ShortFit::Hold, frame: Framing::Fit, ..Fit::default() };
    m.slots = vec![
        beat_slot(
            slot(
                "clips",
                SlotKind::Video,
                "Clips",
                "Twelve clips (fewer are reused), the opening one first",
                0,
                12,
                1,
                clips.iter().enumerate().map(|(i, c)| place(*c, i as u32)).collect(),
            ),
            secs(0.5),
            secs(4.0),
        ),
        logo_slot,
        text_slot("event", "Event", "Its name", 2, &[(event, 1)], 16),
        text_slot("date", "Where and when", "", 3, &[(date, 1)], 32),
        text_slot("outro", "Closing line", "", 4, &[(outro, 1)], 32),
        music_slot(5, music, "A song; the cuts follow its beat", "energetic", "electronic", 120.0, 161),
    ];
    let tall = vec![
        title_hint(event, 0.46, 110.0, 0.035),
        layer_hint(date, 1, [0.5, 0.535], 40.0),
        layer_hint(outro, 1, [0.5, 0.6], 56.0),
        box_hint(logo, [0.39, 0.25, 0.22, 0.12375]),
    ];
    let square = vec![
        title_hint(event, 0.46, 96.0, 0.06),
        layer_hint(date, 1, [0.5, 0.58], 36.0),
        layer_hint(outro, 1, [0.5, 0.75], 50.0),
        box_hint(logo, [0.39, 0.12, 0.22, 0.22]),
    ];
    let mut four_five = square.clone();
    four_five[3] = box_hint(logo, [0.39, 0.15, 0.22, 0.176]);
    m.variants = variants4(tall, square, four_five);
    m.look = Some(Look::Lumetri("Hard Sun".into()));
    m.beat_grid = steady_grid(120.0, total);
    m.poster = Some(secs(1.0));
    b.finish(m)
}

/// Six moments of a day, 3 s each on a 100 bpm grid, each with the time it happened.
pub(super) fn day_in_my_life() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Day in My Life", 3, 2);
    let len = secs(3.0);
    let times = ["7:00 AM", "9:30 AM", "12:15 PM", "3:00 PM", "6:45 PM", "10:00 PM"];
    let g = b.graphic("Template Graphics");
    let (mut moments, mut tags): (Vec<ClipId>, Vec<ClipId>) = (Vec::new(), Vec::new());
    for k in 0..6usize {
        let at = Tick(len.0 * k as i64);
        let item = b.nth_scene(&format!("Moment {}", k + 1), k);
        let c = b.clip(TrackKind::Video, 0, item, at, len);
        drift(&mut b, c, k, len, 5.0);
        if let Some(&prev) = moments.last() {
            let (t, d) = if k == 3 { ("gl_wind", frames(12)) } else { ("cross_dissolve", frames(10)) };
            b.transition(TrackKind::Video, 0, t, Some(prev), Some(c), d, TransitionAlign::CenterAtCut);
        }
        let tag_len = secs(2.4);
        let tg = b.clip(TrackKind::Video, 2, g, at + secs(0.3), tag_len);
        b.tag(tg, 1, times.get(k).copied().unwrap_or(""), (220.0, 960.0), 40.0, "Bold", rgb(0x111827), white(), 92.0);
        b.fade_in_out(tg, tag_len, frames(6));
        b.sfx(1, at + secs(0.3), "sound.click");
        moments.push(c);
        tags.push(tg);
    }
    let total = Tick(len.0 * 6);
    b.transition(TrackKind::Video, 0, "dip_to_black", moments.last().copied(), None, frames(15), TransitionAlign::EndAtCut);
    let title = b.element(1, Tick::ZERO, len, "text.title-pop");
    b.layer_text(title, 1, "A DAY IN MY LIFE");
    b.text_size(title, 1, 120.0);
    b.shift_layer(title, 1, 0.0, -260.0);
    b.shift_layer(title, 2, 0.0, -280.0);
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 0, music_item, Tick::ZERO, total);
    b.transition(TrackKind::Audio, 0, "constant_power", Some(music), None, frames(30), TransitionAlign::EndAtCut);

    let mut m = manifest(
        "day-in-my-life",
        "Day in My Life",
        "Vlog",
        "Six moments of your day, 3 seconds each on the beat, each with the time it happened, crossfading from morning to night.",
        &["day in my life", "vlog", "routine", "lifestyle", "diary"],
    );
    let tag_places: Vec<(ClipId, u64)> = tags.iter().map(|c| (*c, 1)).collect();
    m.slots = vec![
        beat_slot(
            slot(
                "moments",
                SlotKind::Video,
                "Moments",
                "Six clips from morning to night",
                0,
                6,
                1,
                moments.iter().enumerate().map(|(i, c)| place(*c, i as u32)).collect(),
            ),
            secs(1.2),
            secs(6.0),
        ),
        text_slot("title", "Title", "", 1, &[(title, 1)], 24),
        text_slot("times", "Times", "When each moment happened", 2, &tag_places, 12),
        music_slot(3, music, "A song; the moments change on its beat", "happy", "lofi", 100.0, 171),
    ];
    let mut tall: Vec<Hint> = vec![layer_hint(title, 1, [0.5, 0.16], 84.0)];
    let mut square: Vec<Hint> = vec![layer_hint(title, 1, [0.5, 0.15], 72.0)];
    for c in &tags {
        tall.push(layers_hint(*c, &[(1, [0.22, 0.9], 44.0)]));
        square.push(layers_hint(*c, &[(1, [0.2, 0.92], 36.0)]));
    }
    let vs = variants4(tall, square.clone(), square);
    m.variants = vs;
    m.default_variant = "9:16".into();
    m.look = Some(Look::Lumetri("Warm Negative".into()));
    m.beat_grid = steady_grid(100.0, total);
    m.poster = Some(secs(4.0));
    b.finish(m)
}

/// In the spirit of an expo walk-through: a title with a riser and a hit, five robots (or
/// gadgets) cut on the beat with glitch transitions, each with a spec card, and a question to
/// end on.
pub(super) fn robot_showcase() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Robot Showcase", 4, 3);
    let cyan = rgb(0x22d3ee);
    let intro_len = secs(3.0);
    let intro_item = b.scene("Intro Placeholder", DemoScene::Plasma);
    let intro = b.clip(TrackKind::Video, 0, intro_item, Tick::ZERO, intro_len);
    let title = b.element(1, Tick::ZERO, intro_len, "text.title-pop");
    b.layer_text(title, 1, "MEET THE ROBOTS");
    b.text_size(title, 1, 130.0);
    let g = b.graphic("Template Graphics");
    let sub = b.clip(TrackKind::Video, 2, g, secs(0.4), intro_len - secs(0.4));
    b.text(sub, 1, "Expo highlights \u{b7} 2026", (960.0, 712.0), 44.0, "Medium", rgb(0x67e8f9));
    b.ramp(sub, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(10), 100.0)]);
    b.sfx(1, secs(1.0), "sound.riser-2s");
    b.sfx(2, intro_len, "sound.impact-hard");
    let names = ["Humanoid helper", "Robot dog", "Service robot", "Delivery bot", "Robotic arm"];
    let facts = [
        "Walks, talks and waves back",
        "Climbs stairs on four legs",
        "Greets visitors at the door",
        "Carries parcels across town",
        "Builds things to the millimetre",
    ];
    let (mut robots, mut cards): (Vec<ClipId>, Vec<ClipId>) = (Vec::new(), Vec::new());
    let each = secs(4.0);
    let mut prev = intro;
    for k in 0..5usize {
        let at = intro_len + Tick(each.0 * k as i64);
        let item = b.nth_scene(&format!("Robot {}", k + 1), k + 1);
        let c = b.clip(TrackKind::Video, 0, item, at, each);
        drift(&mut b, c, k, each, 7.0);
        let (gl, d) = if k == 0 { ("gl_crosszoom", frames(10)) } else { ("gl_glitchdisplace", frames(8)) };
        b.transition(TrackKind::Video, 0, gl, Some(prev), Some(c), d, TransitionAlign::CenterAtCut);
        if k > 0 {
            b.sfx(1, at - secs(1.0), "sound.sweep");
        }
        let card_len = each - secs(0.6);
        let card = b.clip(TrackKind::Video, 1, g, at + secs(0.4), card_len);
        b.rect(card, 1, (340.0, 862.0), (600.0, 176.0), rgb(0x0b1220), 88.0, 10.0);
        b.rect(card, 2, (44.0, 862.0), (8.0, 176.0), cyan, 100.0, 0.0);
        b.text_ex(card, 3, &format!("ROBOT {:02}", k + 1), (70.0, 815.0), 26.0, "SemiBold", cyan, 0, &[]);
        b.text_ex(card, 4, names.get(k).copied().unwrap_or(""), (70.0, 872.0), 56.0, "Bold", white(), 0, &[]);
        b.text_ex(card, 5, facts.get(k).copied().unwrap_or(""), (70.0, 922.0), 30.0, "Regular", rgb(0xcbd5e1), 0, &[]);
        b.fade_in_out(card, card_len, frames(8));
        robots.push(c);
        cards.push(card);
        prev = c;
    }
    let end_at = intro_len + Tick(each.0 * 5);
    let end_len = secs(3.0);
    let total = end_at + end_len;
    let ink = b.matte("End Background", rgb(0x05070d));
    let end_bg = b.clip(TrackKind::Video, 0, ink, end_at, end_len);
    b.transition(TrackKind::Video, 0, "gl_glitchmemories", Some(prev), Some(end_bg), frames(10), TransitionAlign::CenterAtCut);
    b.transition(TrackKind::Video, 0, "dip_to_black", Some(end_bg), None, frames(15), TransitionAlign::EndAtCut);
    let outro = b.clip(TrackKind::Video, 1, g, end_at + frames(9), end_len - frames(9));
    b.text(outro, 1, "Which one was your favourite?", (960.0, 540.0), 76.0, "Bold", white());
    b.text(outro, 2, "Tell me in the comments", (960.0, 630.0), 40.0, "Medium", rgb(0x67e8f9));
    b.ramp(outro, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(10), 100.0)]);
    b.sfx(2, end_at, "sound.boom");
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 0, music_item, Tick::ZERO, total);
    b.transition(TrackKind::Audio, 0, "constant_power", Some(music), None, frames(30), TransitionAlign::EndAtCut);

    let mut m = manifest(
        "robot-showcase",
        "Robot Showcase",
        "Tech",
        "An expo walk-through: a title with a riser and a hit, five robots (or gadgets) cut on the beat with glitch transitions, a spec card on each, and a question to end on.",
        &["robots", "tech", "showcase", "expo", "gadgets", "review"],
    );
    let mut intro_slot = slot("intro", SlotKind::Video, "Opening shot", "Optional: a wide shot of the place", 0, 1, 0, vec![place(intro, 0)]);
    intro_slot.fit = Fit { short: ShortFit::Loop, portion: Portion::Center, ..Fit::default() };
    let name_places: Vec<(ClipId, u64)> = cards.iter().map(|c| (*c, 4)).collect();
    let fact_places: Vec<(ClipId, u64)> = cards.iter().map(|c| (*c, 5)).collect();
    m.slots = vec![
        intro_slot,
        beat_slot(
            slot(
                "robots",
                SlotKind::Video,
                "Robots",
                "Five clips, one robot (or gadget) each, in order",
                1,
                5,
                1,
                robots.iter().enumerate().map(|(i, c)| place(*c, i as u32)).collect(),
            ),
            secs(2.0),
            secs(8.0),
        ),
        text_slot("title", "Title", "", 2, &[(title, 1)], 18),
        text_slot("subtitle", "Subtitle", "Where you filmed", 3, &[(sub, 1)], 40),
        text_slot("names", "Names", "What each one is", 4, &name_places, 22),
        text_slot("facts", "Facts", "One line about each", 5, &fact_places, 36),
        text_slot("outro", "Closing question", "", 6, &[(outro, 1)], 40),
        text_slot("outro_sub", "Closing line", "Under the question", 7, &[(outro, 2)], 40),
        music_slot(8, music, "A song; the robots change on its beat", "epic", "electronic", 120.0, 181),
    ];
    let mut tall: Vec<Hint> = vec![title_hint(title, 0.46, 96.0, 0.032), layer_hint(sub, 1, [0.5, 0.53], 34.0), sizes_hint(outro, &[(1, 52.0), (2, 30.0)])];
    let mut square: Vec<Hint> = vec![title_hint(title, 0.46, 84.0, 0.055), layer_hint(sub, 1, [0.5, 0.57], 30.0), sizes_hint(outro, &[(1, 46.0), (2, 28.0)])];
    for c in &cards {
        tall.push(sizes_hint(*c, &[(3, 18.0), (4, 34.0), (5, 20.0)]));
        square.push(sizes_hint(*c, &[(3, 18.0), (4, 34.0), (5, 20.0)]));
    }
    m.variants = variants4(tall, square.clone(), square);
    m.look = Some(Look::Lumetri("Steel Blue".into()));
    m.beat_grid = steady_grid(120.0, total);
    m.poster = Some(secs(4.5));
    b.finish(m)
}
