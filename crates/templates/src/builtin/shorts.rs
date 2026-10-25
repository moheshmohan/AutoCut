//! Shorts and Reels starters made to fill themselves (`autocut.quick`): vertical first, every
//! slot with AI hints ([`crate::manifest::AiHints`]) — what each picture slot's shot is for and
//! what each text slot says — creator looks on an adjustment clip (`Look: tech`, `Look: vhs`),
//! Creator transitions (`crt_*`), cuts and zoom pulses on the beat, animated captions, and text
//! kept inside every platform's safe zone in the vertical layout (the frame between 15 % and
//! 65 % of its height and 12 % and 82 % of its width, which YouTube Shorts, Reels and TikTok
//! all leave clear).
//!
//! | id | what |
//! |---|---|
//! | `builtin:tech-reveal` | a hook, eight b-roll shots on the beat, the reveal, an end card |
//! | `builtin:event-recap-short` | an establishing shot with the event's name, a fast montage, a closing card |
//! | `builtin:talking-head-tip` | three kept takes of you talking, b-roll cutaways over the joins, word captions |
//! | `builtin:gadget-showcase` | a hook, three features with spec cards, the hero shot, an end card |
//! | `builtin:before-after-short` | the before, a swipe to the after, both stacked |
//! | `builtin:day-in-30s` | twelve moments of a day in the order they were filmed |

use filmcraft_media::DemoScene;
use filmcraft_project::{AnimatedCaptions, CaptionPreset, ClipId, ItemId, ParamValue, TrackKind, TransitionAlign};
use filmcraft_time::Tick;

use super::{
    Builder, Hint, beat_slot, box_hint, boxed, captions_slot, frames, layer_hint, layers_hint, manifest, music_slot, place, rgb, secs, shot_ai, sizes_hint,
    slot, speech_slot, steady_grid, text_ai, text_slot, variant, variants4, white, with_ai,
};
use crate::manifest::{AiHints, EnergyHint, Fit, Look, Portion, Purpose, ShortFit, SlotKind, SpeechHint};
use crate::{Package, TemplateError};

const F: fn(f64) -> ParamValue = ParamValue::Float;
/// The horizontal middle of the safe area all three platforms leave (12 %–82 % of the width).
const SAFE_X: f64 = 0.47;
/// The look's name prefix on its adjustment clip (`autocut.quick` swaps it for another look).
pub const LOOK_PREFIX: &str = "Look: ";

/// A creator look on an adjustment clip over `[start, start + dur)` (top video track `track`):
/// `tech` (a HUD, scanlines, a touch of chromatic aberration and grain) or `vhs` (tape and grain).
fn creator_look(b: &mut Builder, track: usize, start: Tick, dur: Tick, look: &str) -> ClipId {
    let c = b.adjustment(&format!("{LOOK_PREFIX}{look}"), track, start, dur);
    if look == "vhs" {
        b.effect(c, "cr_vhs", &[("amount", F(60.0)), ("tracking", F(30.0)), ("noise", F(20.0))]);
        b.effect(c, "cr_film_grain", &[("amount", F(22.0)), ("size", F(2.0))]);
    } else {
        b.effect(c, "cr_chromatic_aberration", &[("amount", F(3.0))]);
        b.effect(
            c,
            "cr_tech_hud",
            &[
                ("color", ParamValue::Color(filmcraft_project::effect::HUD_COLOR)),
                ("scanlines", F(28.0)),
                ("grid", F(8.0)),
                ("brackets", F(85.0)),
                ("vignette", F(40.0)),
                ("tint", F(10.0)),
                ("flicker", F(6.0)),
            ],
        );
        b.effect(c, "cr_film_grain", &[("amount", F(14.0)), ("size", F(1.2))]);
    }
    c
}

/// A `text.title-pop` title over `[at, at + dur)` on track `track`, `size` pixels, `dy` design
/// pixels below the middle of the frame.
fn title(b: &mut Builder, track: usize, at: Tick, dur: Tick, text: &str, size: f64, dy: f64) -> ClipId {
    let c = b.element(track, at, dur, "text.title-pop");
    b.layer_text(c, 1, text);
    b.text_size(c, 1, size);
    if dy != 0.0 {
        b.shift_layer(c, 1, 0.0, dy);
        b.shift_layer(c, 2, 0.0, dy);
    }
    c
}

/// A title's place in a vertical (or square) layout: centred on the safe area at height `y`,
/// `size` pixels, its rule `gap` under it.
fn title_at(clip: ClipId, y: f64, size: f64, gap: f64) -> Hint {
    // a fraction JSON writes and reads back exactly
    let rule = ((y + gap) * 1e4).round() / 1e4;
    layers_hint(clip, &[(1, [SAFE_X, y], size), (2, [SAFE_X, rule], 0.0)])
}

/// One text layer's place in a vertical (or square) layout: centred on the safe area.
fn text_at(clip: ClipId, layer: u64, y: f64, size: f64) -> Hint {
    layer_hint(clip, layer, [SAFE_X, y], size)
}

/// Two text layers (1 and 2) of one graphic clip in a vertical (or square) layout: each centred
/// on the safe area at its `(height, size)`.
fn two_lines(clip: ClipId, first: (f64, f64), second: (f64, f64)) -> Hint {
    layers_hint(clip, &[(1, [SAFE_X, first.0], first.1), (2, [SAFE_X, second.0], second.1)])
}

/// A soft swish just before each cut of `cuts` on audio track `track`.
fn whooshes(b: &mut Builder, track: usize, cuts: &[Tick]) {
    for &c in cuts {
        if c > frames(9) {
            b.sfx(track, c - frames(9), "sound.swish");
        }
    }
}

/// The music bed: a placeholder over the whole show, faded out at its end.
fn music(b: &mut Builder, track: usize, total: Tick, db: f64) -> ClipId {
    let item = b.music_placeholder();
    let m = b.clip(TrackKind::Audio, track, item, Tick::ZERO, total);
    if db != 0.0 {
        b.volume(m, db);
    }
    b.transition(TrackKind::Audio, track, "constant_power", Some(m), None, frames(30), TransitionAlign::EndAtCut);
    m
}

/// Picture hints in filming order (`chronological`).
fn in_order(mut a: AiHints) -> AiHints {
    a.chronological = true;
    a
}

/// An end card from `at` for `len`: a blurred, darkened copy of `item` (a slot's placeholder: the
/// card's clip becomes another placement of that slot's media, so the end shows the same footage)
/// on track 0, pushing in slowly, a shade over it on track 1, dipping to black at its end. Returns
/// the card's clip.
fn end_card(b: &mut Builder, after: Option<ClipId>, at: Tick, len: Tick, transition: &str, item: ItemId) -> ClipId {
    let card = b.clip(TrackKind::Video, 0, item, at, len);
    b.effect(card, "gaussian_blur", &[("blurriness", F(40.0)), ("repeat_edge", ParamValue::Bool(true))]);
    b.ramp(card, "motion", "scale", &[(Tick::ZERO, 106.0), (len, 114.0)]);
    let ink = b.matte("Shade", rgb(0x05070d));
    let shade = b.clip(TrackKind::Video, 1, ink, at, len);
    b.opacity(shade, 55.0);
    if let Some(prev) = after {
        b.transition(TrackKind::Video, 0, transition, Some(prev), Some(card), frames(12), TransitionAlign::CenterAtCut);
    }
    b.transition(TrackKind::Video, 0, "dip_to_black", Some(card), None, frames(15), TransitionAlign::EndAtCut);
    card
}

/// 17 s on a 120 bpm grid: a hook with a big line over it, eight b-roll shots of a beat pair each
/// (glitch cuts, a zoom pulse on every beat), a white flash into the reveal with its name, an end
/// card. The tech look over the picture.
pub(super) fn tech_reveal() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Tech Reveal", 5, 3);
    let beat = frames(15);
    let hook_len = Tick(beat.0 * 4);
    let broll_len = Tick(beat.0 * 2);
    let reveal_len = Tick(beat.0 * 8);
    let end_len = Tick(beat.0 * 6);
    let hook_item = b.scene("Hook Placeholder", DemoScene::Plasma);
    let hook = b.clip(TrackKind::Video, 0, hook_item, Tick::ZERO, hook_len);
    b.pulse(hook, hook_len, Tick(beat.0 * 2), 6.0);
    let mut at = hook_len;
    let mut prev = hook;
    let mut brolls = Vec::new();
    for k in 0..8usize {
        let item = b.nth_scene(&format!("B-roll {}", k + 1), k + 2);
        let c = b.clip(TrackKind::Video, 0, item, at, broll_len);
        b.pulse(c, broll_len, beat, 7.0);
        let (t, d) = match k {
            0 => ("crt_flash_zoom", frames(10)),
            k if k % 2 == 1 => ("crt_glitch", frames(8)),
            _ => ("crt_pull_in", frames(8)),
        };
        b.transition(TrackKind::Video, 0, t, Some(prev), Some(c), d, TransitionAlign::CenterAtCut);
        brolls.push(c);
        prev = c;
        at += broll_len;
    }
    let reveal_item = b.scene("Reveal Placeholder", DemoScene::CityNight);
    let reveal = b.clip(TrackKind::Video, 0, reveal_item, at, reveal_len);
    b.ramp(reveal, "motion", "scale", &[(Tick::ZERO, 100.0), (reveal_len, 112.0)]);
    b.transition(TrackKind::Video, 0, "crt_white_flash", Some(prev), Some(reveal), frames(9), TransitionAlign::CenterAtCut);
    let reveal_at = at;
    at += reveal_len;
    let end_at = at;
    let end_bg = end_card(&mut b, Some(reveal), end_at, end_len, "crt_blur", reveal_item);
    let total = end_at + end_len;
    creator_look(&mut b, 4, Tick::ZERO, end_at, "tech");
    let hook_line = title(&mut b, 1, Tick::ZERO, hook_len, "WAIT FOR IT", 110.0, -250.0);
    let name = title(&mut b, 1, reveal_at + frames(10), reveal_len - frames(10), "THE NEW ONE", 120.0, 0.0);
    let g = b.graphic("Template Graphics");
    let outro = b.clip(TrackKind::Video, 2, g, end_at + frames(6), end_len - frames(6));
    b.text(outro, 1, "Follow for part 2", (960.0, 520.0), 84.0, "Bold", white());
    b.text(outro, 2, "More tech every week", (960.0, 610.0), 44.0, "Medium", rgb(0x67e8f9));
    b.ramp(outro, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(10), 100.0)]);
    whooshes(&mut b, 1, &[hook_len]);
    b.sfx(1, reveal_at - secs(2.0), "sound.riser-2s");
    b.sfx(2, reveal_at, "sound.impact-hard");
    b.sfx(2, end_at, "sound.boom");
    let music = music(&mut b, 0, total, 0.0);

    let mut m = manifest(
        "tech-reveal",
        "Tech Reveal",
        "Tech",
        "A vertical tech short that fills itself: a hook with a big line, eight b-roll shots cut on the beat with glitch cuts and zoom pulses, a white flash into the reveal and its name, an end card — under a tech HUD look.",
        &["tech", "reveal", "product", "gadget", "robot", "shorts", "reels", "ai", "beat"],
    );
    m.slots = vec![
        with_ai(
            beat_slot(slot("hook", SlotKind::Video, "Hook", "The most striking shot: it plays first", 0, 1, 1, vec![place(hook, 0)]), secs(1.0), secs(4.0)),
            shot_ai(Purpose::Hook, Some(EnergyHint::High), ""),
        ),
        with_ai(
            beat_slot(
                slot(
                    "broll",
                    SlotKind::Video,
                    "B-roll",
                    "Eight short shots around it (fewer are reused)",
                    1,
                    8,
                    1,
                    brolls.iter().enumerate().map(|(i, c)| place(*c, i as u32)).collect(),
                ),
                secs(0.5),
                secs(2.5),
            ),
            shot_ai(Purpose::Broll, Some(EnergyHint::High), ""),
        ),
        with_ai(
            beat_slot(
                slot(
                    "reveal",
                    SlotKind::Video,
                    "Reveal",
                    "The thing itself, shown whole (blurred again under the end card)",
                    2,
                    1,
                    1,
                    vec![place(reveal, 0), place(end_bg, 0)],
                ),
                secs(2.0),
                secs(8.0),
            ),
            shot_ai(Purpose::Reveal, None, ""),
        ),
        with_ai(
            text_slot("hook_line", "Hook line", "Over the first shot", 3, &[(hook_line, 1)], 18),
            text_ai("A punchy hook of 2 to 4 words about the topic and what the first shot shows, that makes people keep watching, in capitals", 18, "hook"),
        ),
        with_ai(
            text_slot("title", "Name", "What is revealed", 4, &[(name, 1)], 16),
            text_ai("What the reveal shows, by its name or what it is (a product, a robot, a place), 1 to 3 words, in capitals", 16, "reveal"),
        ),
        with_ai(text_slot("outro", "Call to action", "On the end card", 5, &[(outro, 1)], 22), text_ai("A short call to action, 2 to 5 words", 22, "")),
        with_ai(
            text_slot("outro_sub", "Closing line", "Under the call to action", 6, &[(outro, 2)], 32),
            text_ai("One short closing line about the channel or the topic", 32, ""),
        ),
        music_slot(7, music, "A song; the cuts follow its beat", "energetic", "electronic", 120.0, 201),
    ];
    let tall = vec![title_at(hook_line, 0.24, 78.0, 0.03), title_at(name, 0.42, 84.0, 0.03), two_lines(outro, (0.4, 60.0), (0.45, 34.0))];
    let square = vec![title_at(hook_line, 0.22, 72.0, 0.05), title_at(name, 0.45, 80.0, 0.05), two_lines(outro, (0.45, 56.0), (0.53, 32.0))];
    m.variants = variants4(tall, square.clone(), square);
    m.default_variant = "9:16".into();
    m.look = Some(Look::Lumetri("Steel Blue".into()));
    m.beat_grid = steady_grid(120.0, total);
    m.poster = Some(reveal_at + secs(1.0));
    b.finish(m)
}

/// 16 s on a 120 bpm grid: the place with the event's name and when, ten shots of a beat pair
/// each (swipes and pulls, a pulse on the beat), a closing card.
pub(super) fn event_recap_short() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Event Recap Short", 4, 3);
    let beat = frames(15);
    let open_len = Tick(beat.0 * 6);
    let shot_len = Tick(beat.0 * 2);
    let end_len = Tick(beat.0 * 6);
    let open_item = b.scene("Establishing Placeholder", DemoScene::CityNight);
    let open = b.clip(TrackKind::Video, 0, open_item, Tick::ZERO, open_len);
    b.ramp(open, "motion", "scale", &[(Tick::ZERO, 100.0), (open_len, 108.0)]);
    let mut at = open_len;
    let mut prev = open;
    let mut shots = Vec::new();
    let mut cuts = Vec::new();
    for k in 0..10usize {
        let item = b.nth_scene(&format!("Moment {}", k + 1), k);
        let c = b.clip(TrackKind::Video, 0, item, at, shot_len);
        b.pulse(c, shot_len, beat, 6.0);
        let t = match k % 3 {
            0 => Some("crt_swipe_up"),
            1 => Some("crt_pull_in"),
            _ => None,
        };
        if let Some(t) = t {
            b.transition(TrackKind::Video, 0, t, Some(prev), Some(c), frames(8), TransitionAlign::CenterAtCut);
            cuts.push(at);
        }
        shots.push(c);
        prev = c;
        at += shot_len;
    }
    let end_at = at;
    let end_bg = end_card(&mut b, Some(prev), end_at, end_len, "crt_light_leak", open_item);
    let total = end_at + end_len;
    let event = title(&mut b, 1, frames(6), open_len - frames(6), "SUMMIT 2026", 120.0, 0.0);
    let g = b.graphic("Template Graphics");
    let when = b.clip(TrackKind::Video, 2, g, secs(0.5), open_len - secs(0.5));
    b.text(when, 1, "Three days, one city", (960.0, 700.0), 46.0, "Medium", white());
    b.ramp(when, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(8), 100.0)]);
    let closing = b.clip(TrackKind::Video, 2, g, end_at + frames(8), end_len - frames(8));
    b.text(closing, 1, "See you next year", (960.0, 540.0), 80.0, "Bold", white());
    b.ramp(closing, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(10), 100.0)]);
    whooshes(&mut b, 1, &cuts);
    b.sfx(2, open_len, "sound.impact-soft");
    b.sfx(2, end_at, "sound.boom");
    let music = music(&mut b, 0, total, 0.0);

    let mut m = manifest(
        "event-recap-short",
        "Event Recap Short",
        "Events",
        "A vertical event recap that fills itself: the place with the event's name, ten moments cut on the beat with swipes and zoom pulses, a closing card.",
        &["event", "recap", "expo", "conference", "festival", "highlights", "shorts", "reels", "beat"],
    );
    m.slots = vec![
        with_ai(
            beat_slot(
                slot(
                    "establishing",
                    SlotKind::Video,
                    "Opening shot",
                    "A wide shot of the place (blurred again under the end card)",
                    0,
                    1,
                    1,
                    vec![place(open, 0), place(end_bg, 0)],
                ),
                secs(1.5),
                secs(6.0),
            ),
            shot_ai(Purpose::Establishing, None, ""),
        ),
        with_ai(
            beat_slot(
                slot(
                    "moments",
                    SlotKind::Video,
                    "Moments",
                    "Ten moments of the event (fewer are reused)",
                    1,
                    10,
                    1,
                    shots.iter().enumerate().map(|(i, c)| place(*c, i as u32)).collect(),
                ),
                secs(0.5),
                secs(2.5),
            ),
            in_order(shot_ai(Purpose::Broll, Some(EnergyHint::High), "")),
        ),
        with_ai(text_slot("event", "Event", "Its name", 2, &[(event, 1)], 16), text_ai("The event's name, 1 to 3 words, in capitals", 16, "establishing")),
        with_ai(
            text_slot("when", "Where and when", "Under the name", 3, &[(when, 1)], 28),
            text_ai("Where or when it was, or what it was about, in a few words", 28, "establishing"),
        ),
        with_ai(text_slot("closing", "Closing line", "On the end card", 4, &[(closing, 1)], 22), text_ai("A warm closing line of 2 to 5 words", 22, "moments")),
        music_slot(5, music, "A song; the cuts follow its beat", "energetic", "pop", 120.0, 211),
    ];
    let tall = vec![title_at(event, 0.3, 84.0, 0.03), text_at(when, 1, 0.37, 34.0), text_at(closing, 1, 0.42, 58.0)];
    let square = vec![title_at(event, 0.4, 80.0, 0.05), text_at(when, 1, 0.52, 32.0), text_at(closing, 1, 0.47, 54.0)];
    m.variants = variants4(tall, square.clone(), square);
    m.default_variant = "9:16".into();
    m.look = Some(Look::Lumetri("Hard Sun".into()));
    m.beat_grid = steady_grid(120.0, total);
    m.poster = Some(secs(1.5));
    b.finish(m)
}

/// 24 s of someone giving a tip: three takes of them talking (the kept takes), a b-roll cutaway
/// over each join (their voice goes on), a hook at the top, word-by-word captions, an end card.
pub(super) fn talking_head_tip() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Talking-head Tip", 4, 3);
    let take_len = secs(7.0);
    let mut takes = Vec::new();
    let mut last_item = None;
    for k in 0..3usize {
        let item = b.scene(&format!("Take {}", k + 1), DemoScene::Forest);
        let c = b.av(0, 0, item, Tick(take_len.0 * k as i64), take_len);
        takes.push(c);
        last_item = Some(item);
    }
    let talk = Tick(take_len.0 * 3);
    let end_len = secs(3.0);
    let total = talk + end_len;
    let mut cutaways = Vec::new();
    for k in 1..3i64 {
        let item = b.nth_scene(&format!("Cutaway {k}"), k as usize + 1);
        let start = Tick(take_len.0 * k) - secs(1.5);
        let c = b.clip(TrackKind::Video, 1, item, start, secs(3.0));
        b.ramp(c, "motion", "scale", &[(Tick::ZERO, 104.0), (secs(3.0), 112.0)]);
        b.transition(TrackKind::Video, 1, "crt_mix", None, Some(c), frames(8), TransitionAlign::StartAtCut);
        b.transition(TrackKind::Video, 1, "crt_mix", Some(c), None, frames(8), TransitionAlign::EndAtCut);
        cutaways.push(c);
    }
    let last = takes.last().copied();
    let end_item = last_item.ok_or_else(|| TemplateError::Apply("no take".into()))?;
    let end_bg = end_card(&mut b, last, talk, end_len, "crt_blur", end_item);
    let hook = title(&mut b, 2, Tick::ZERO, secs(3.0), "THE ONE TIP", 110.0, -300.0);
    let g = b.graphic("Template Graphics");
    let cta = b.clip(TrackKind::Video, 2, g, talk + frames(6), end_len - frames(6));
    b.text(cta, 1, "Save this for later", (960.0, 540.0), 80.0, "Bold", white());
    b.ramp(cta, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(10), 100.0)]);
    b.captions(&[(0.3, 2.0, "Here is the one tip"), (2.1, 4.2, "nobody tells you"), (7.2, 9.6, "your words appear here"), (9.8, 12.0, "as you say them")]);
    b.sfx(2, Tick::ZERO, "sound.pop");
    b.sfx(2, talk, "sound.click");
    let music = music(&mut b, 1, total, -22.0);

    let mut m = manifest(
        "talking-head-tip",
        "Talking-head Tip",
        "Reels & Shorts",
        "A vertical tip that fills itself from your talking clips: the best three takes of what you said (retakes and pauses left out), a b-roll cutaway over each join, a hook at the top, word-by-word captions, an end card.",
        &["talking head", "tip", "how to", "advice", "captions", "creator", "shorts", "reels"],
    );
    let mut takes_slot = speech_slot(slot(
        "takes",
        SlotKind::Video,
        "You, talking",
        "Three takes of you talking to the camera (one clip is fine: its best parts are used)",
        0,
        3,
        1,
        takes.iter().enumerate().map(|(i, c)| place(*c, i as u32)).chain(std::iter::once(place(end_bg, 2))).collect(),
    ));
    takes_slot.ai = Some(AiHints { purpose: Some(Purpose::TalkingHead), speech: Some(SpeechHint::Want), chronological: true, ..AiHints::default() });
    let mut cut_slot = slot(
        "cutaways",
        SlotKind::Video,
        "Cutaways",
        "Two shots of what you talk about (your voice goes on over them)",
        1,
        2,
        0,
        cutaways.iter().enumerate().map(|(i, c)| place(*c, i as u32)).collect(),
    );
    cut_slot.fit = Fit { short: ShortFit::Loop, portion: Portion::Center, ..Fit::default() };
    cut_slot.ai = Some(AiHints { purpose: Some(Purpose::Broll), speech: Some(SpeechHint::Avoid), ..AiHints::default() });
    m.slots = vec![
        takes_slot,
        cut_slot,
        with_ai(
            text_slot("hook", "Hook", "The first words on screen", 2, &[(hook, 1)], 18),
            text_ai("The tip as a hook of 2 to 4 words, in capitals, from what is said", 18, "takes"),
        ),
        with_ai(text_slot("cta", "Call to action", "On the end card", 3, &[(cta, 1)], 22), text_ai("A short call to action, 2 to 5 words", 22, "")),
        captions_slot(4, &["takes"], 18, 1),
        music_slot(5, music, "Optional: music under your voice (kept quiet)", "calm", "lofi", 90.0, 221),
    ];
    let tall = vec![title_at(hook, 0.22, 78.0, 0.03), text_at(cta, 1, 0.42, 58.0)];
    let square = vec![title_at(hook, 0.2, 72.0, 0.05), text_at(cta, 1, 0.47, 54.0)];
    m.variants = variants4(tall, square.clone(), square);
    m.default_variant = "9:16".into();
    let mut style = CaptionPreset::Bold.base_style();
    style.size = 92.0;
    style.animated =
        Some(AnimatedCaptions { platform: Some(filmcraft_project::safe_zones::Platform::Vertical), ..AnimatedCaptions::preset(CaptionPreset::Bold) });
    m.caption_style = Some(style);
    m.poster = Some(secs(1.5));
    b.finish(m)
}

/// 16.5 s on a 120 bpm grid: a close-up hook, three features of three seconds each with a spec
/// card (slices and pulls between them, a pulse every other beat), the hero shot with its name,
/// an end card; the tech look over the picture.
pub(super) fn gadget_showcase() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Gadget Showcase", 5, 3);
    let beat = frames(15);
    let hook_len = Tick(beat.0 * 4);
    let feat_len = Tick(beat.0 * 6);
    let hero_len = Tick(beat.0 * 6);
    let end_len = Tick(beat.0 * 5);
    let cyan = rgb(0x22d3ee);
    let hook_item = b.scene("Close-up Placeholder", DemoScene::Plasma);
    let hook = b.clip(TrackKind::Video, 0, hook_item, Tick::ZERO, hook_len);
    b.pulse(hook, hook_len, Tick(beat.0 * 2), 6.0);
    let g = b.graphic("Template Graphics");
    let mut at = hook_len;
    let mut prev = hook;
    let (mut feats, mut cards) = (Vec::new(), Vec::new());
    let names = ["Battery", "Camera", "Speed"];
    let facts = ["All day on one charge", "Sharp in the dark", "Ready in a second"];
    for k in 0..3usize {
        let item = b.nth_scene(&format!("Feature {}", k + 1), k + 1);
        let c = b.clip(TrackKind::Video, 0, item, at, feat_len);
        b.pulse(c, feat_len, Tick(beat.0 * 2), 5.0);
        let t = if k == 1 { "crt_slice" } else { "crt_pull_in" };
        b.transition(TrackKind::Video, 0, t, Some(prev), Some(c), frames(9), TransitionAlign::CenterAtCut);
        let card_len = feat_len - secs(0.5);
        let card = b.clip(TrackKind::Video, 1, g, at + secs(0.3), card_len);
        b.rect(card, 1, (640.0, 860.0), (1040.0, 200.0), rgb(0x0b1220), 88.0, 12.0);
        b.rect(card, 2, (124.0, 860.0), (8.0, 200.0), cyan, 100.0, 0.0);
        b.text_ex(card, 3, &format!("FEATURE {:02}", k + 1), (150.0, 815.0), 26.0, "SemiBold", cyan, 0, &[]);
        b.text_ex(card, 4, names.get(k).copied().unwrap_or(""), (150.0, 872.0), 56.0, "Bold", white(), 0, &[]);
        b.text_ex(card, 5, facts.get(k).copied().unwrap_or(""), (150.0, 922.0), 30.0, "Regular", rgb(0xcbd5e1), 0, &[]);
        b.fade_in_out(card, card_len, frames(8));
        b.sfx(1, at, "sound.swoosh-up");
        feats.push(c);
        cards.push(card);
        prev = c;
        at += feat_len;
    }
    let hero_item = b.scene("Hero Placeholder", DemoScene::CityNight);
    let hero = b.clip(TrackKind::Video, 0, hero_item, at, hero_len);
    b.ramp(hero, "motion", "scale", &[(Tick::ZERO, 112.0), (hero_len, 100.0)]);
    b.transition(TrackKind::Video, 0, "crt_white_flash", Some(prev), Some(hero), frames(9), TransitionAlign::CenterAtCut);
    let hero_at = at;
    at += hero_len;
    let end_at = at;
    let end_bg = end_card(&mut b, Some(hero), end_at, end_len, "crt_blur", hero_item);
    let total = end_at + end_len;
    creator_look(&mut b, 4, Tick::ZERO, end_at, "tech");
    let name = title(&mut b, 2, hero_at + frames(8), hero_len - frames(8), "MEET IT", 120.0, -200.0);
    let cta = b.clip(TrackKind::Video, 2, g, end_at + frames(6), end_len - frames(6));
    b.text(cta, 1, "Would you buy it?", (960.0, 520.0), 84.0, "Bold", white());
    b.text(cta, 2, "Tell me in the comments", (960.0, 610.0), 42.0, "Medium", rgb(0x67e8f9));
    b.ramp(cta, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(10), 100.0)]);
    b.sfx(2, hero_at, "sound.impact-hard");
    b.sfx(1, end_at, "sound.boom");
    let music = music(&mut b, 0, total, 0.0);

    let mut m = manifest(
        "gadget-showcase",
        "Gadget Showcase",
        "Tech",
        "A vertical gadget short that fills itself: a close-up hook, three features with spec cards cut on the beat, the hero shot with its name, a question to end on — under a tech HUD look.",
        &["gadget", "product", "review", "tech", "features", "shorts", "reels", "beat"],
    );
    let name_places: Vec<(ClipId, u64)> = cards.iter().map(|c| (*c, 4)).collect();
    let fact_places: Vec<(ClipId, u64)> = cards.iter().map(|c| (*c, 5)).collect();
    m.slots = vec![
        with_ai(
            beat_slot(slot("hook", SlotKind::Video, "Close-up", "A close detail that makes people look", 0, 1, 1, vec![place(hook, 0)]), secs(1.0), secs(4.0)),
            shot_ai(Purpose::Hook, Some(EnergyHint::High), "close-up detail"),
        ),
        with_ai(
            beat_slot(
                slot(
                    "features",
                    SlotKind::Video,
                    "Features",
                    "Three shots, one feature each",
                    1,
                    3,
                    1,
                    feats.iter().enumerate().map(|(i, c)| place(*c, i as u32)).collect(),
                ),
                secs(1.5),
                secs(6.0),
            ),
            shot_ai(Purpose::Broll, Some(EnergyHint::Medium), ""),
        ),
        with_ai(
            beat_slot(
                slot(
                    "hero",
                    SlotKind::Video,
                    "Hero shot",
                    "The whole thing at its best (blurred again under the end card)",
                    2,
                    1,
                    1,
                    vec![place(hero, 0), place(end_bg, 0)],
                ),
                secs(1.5),
                secs(6.0),
            ),
            shot_ai(Purpose::Reveal, None, ""),
        ),
        with_ai(
            text_slot("names", "Feature names", "One or two words each", 3, &name_places, 18),
            text_ai("The feature each shot shows, 1 or 2 words", 18, "features"),
        ),
        with_ai(
            text_slot("facts", "Feature lines", "One line about each", 4, &fact_places, 30),
            text_ai("One short line about the feature each shot shows", 30, "features"),
        ),
        with_ai(text_slot("name", "Name", "Over the hero shot", 5, &[(name, 1)], 16), text_ai("The gadget's name, 1 to 3 words, in capitals", 16, "hero")),
        with_ai(
            text_slot("question", "Closing question", "On the end card", 6, &[(cta, 1)], 24),
            text_ai("A short question for the viewers, 3 to 5 words", 24, ""),
        ),
        with_ai(
            text_slot("question_sub", "Closing line", "Under the question", 7, &[(cta, 2)], 32),
            text_ai("One short line asking viewers to comment or follow", 32, ""),
        ),
        music_slot(8, music, "A song; the cuts follow its beat", "uplifting", "electronic", 120.0, 231),
    ];
    let mut tall: Vec<Hint> = vec![title_at(name, 0.26, 84.0, 0.03), two_lines(cta, (0.4, 58.0), (0.45, 32.0))];
    let mut square: Vec<Hint> = vec![title_at(name, 0.25, 80.0, 0.05), two_lines(cta, (0.45, 54.0), (0.53, 30.0))];
    for c in &cards {
        // the card sits low in the safe area of the vertical layout
        tall.push(layers_hint(
            *c,
            &[(1, [SAFE_X, 0.58], 0.0), (2, [0.205, 0.58], 0.0), (3, [0.225, 0.566], 20.0), (4, [0.225, 0.588], 36.0), (5, [0.225, 0.603], 22.0)],
        ));
        square.push(sizes_hint(*c, &[(3, 20.0), (4, 38.0), (5, 22.0)]));
    }
    m.variants = variants4(tall, square.clone(), square);
    m.default_variant = "9:16".into();
    m.look = Some(Look::Lumetri("Steel Blue".into()));
    m.beat_grid = steady_grid(120.0, total);
    m.poster = Some(hero_at + secs(1.0));
    b.finish(m)
}

/// 10.5 s: the before with its label, a swipe to the after, then both (side by side; stacked
/// when vertical) with a title, in the order they were filmed.
pub(super) fn before_after_short() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Before / After Short", 4, 2);
    let before_item = b.scene("Before Placeholder", DemoScene::Dunes);
    let after_item = b.scene("After Placeholder", DemoScene::Forest);
    let cut = secs(3.5);
    let split_at = secs(7.0);
    let total = secs(10.5);
    let before = b.clip(TrackKind::Video, 0, before_item, Tick::ZERO, cut);
    let after = b.clip(TrackKind::Video, 0, after_item, cut, split_at - cut);
    b.transition(TrackKind::Video, 0, "crt_swipe_left", Some(before), Some(after), frames(12), TransitionAlign::CenterAtCut);
    let top = b.clip(TrackKind::Video, 0, before_item, split_at, total - split_at);
    let bottom = b.clip(TrackKind::Video, 1, after_item, split_at, total - split_at);
    b.transition(TrackKind::Video, 0, "crt_white_flash", Some(after), Some(top), frames(9), TransitionAlign::CenterAtCut);
    b.transition(TrackKind::Video, 1, "dip_to_black", Some(bottom), None, frames(12), TransitionAlign::EndAtCut);
    let dark = rgb(0x111827);
    let g = b.graphic("Template Graphics");
    let tag_a = b.clip(TrackKind::Video, 2, g, frames(6), cut - frames(6));
    b.tag(tag_a, 1, "BEFORE", (960.0, 200.0), 52.0, "Bold", white(), dark, 85.0);
    let tag_b = b.clip(TrackKind::Video, 2, g, cut + frames(6), split_at - cut - frames(6));
    b.tag(tag_b, 1, "AFTER", (960.0, 200.0), 52.0, "Bold", rgb(0x111827), rgb(0xfacc15), 95.0);
    let head = b.clip(TrackKind::Video, 3, g, split_at + frames(8), total - split_at - frames(8));
    b.tag(head, 1, "What a difference", (960.0, 540.0), 60.0, "Bold", rgb(0x111827), white(), 92.0);
    b.ramp(head, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(8), 100.0)]);
    b.sfx(1, cut - frames(9), "sound.swish");
    b.sfx(1, split_at, "sound.pop");
    let music = music(&mut b, 0, total, -6.0);

    let mut m = manifest(
        "before-after-short",
        "Before / After Short",
        "Social",
        "A vertical before and after that fills itself: the earlier shot, a swipe to the later one, then both stacked with a title — the two picked in the order they were filmed.",
        &["before after", "transformation", "makeover", "progress", "comparison", "shorts", "reels"],
    );
    let picture_fit = Fit { short: ShortFit::Hold, portion: Portion::Center, ..Fit::default() };
    let mut before_slot = slot("before", SlotKind::Video, "Before", "How it was", 0, 1, 1, vec![place(before, 0), boxed(top, 0, [0.0, 0.0, 0.5, 1.0])]);
    before_slot.fit = picture_fit;
    before_slot.ai = Some(in_order(shot_ai(Purpose::Establishing, None, "")));
    let mut after_slot = slot("after", SlotKind::Video, "After", "How it is now", 1, 1, 1, vec![place(after, 0), boxed(bottom, 0, [0.5, 0.0, 0.5, 1.0])]);
    after_slot.fit = picture_fit;
    after_slot.ai = Some(in_order(shot_ai(Purpose::Reveal, None, "")));
    m.slots = vec![
        before_slot,
        after_slot,
        with_ai(
            text_slot("labels", "Labels", "The two labels", 2, &[(tag_a, 1), (tag_b, 1)], 10),
            text_ai("Two one-word labels for the earlier and the later shot, in capitals (BEFORE, AFTER)", 10, ""),
        ),
        with_ai(text_slot("title", "Title", "Over the comparison", 3, &[(head, 1)], 24), text_ai("What changed, 2 to 5 words", 24, "after")),
        music_slot(4, music, "Optional: a song", "uplifting", "pop", 110.0, 241),
    ];
    let stacked = |title_y: f64| -> Vec<Hint> {
        vec![
            box_hint(top, [0.0, 0.0, 1.0, 0.5]),
            box_hint(bottom, [0.0, 0.5, 1.0, 0.5]),
            layer_hint(tag_a, 1, [SAFE_X, 0.2], 46.0),
            layer_hint(tag_b, 1, [SAFE_X, 0.2], 46.0),
            layer_hint(head, 1, [SAFE_X, title_y], 50.0),
        ]
    };
    m.variants = vec![variant("9:16", stacked(0.51)), variant("16:9", Vec::new()), variant("1:1", stacked(0.51)), variant("4:5", stacked(0.51))];
    m.default_variant = "9:16".into();
    m.poster = Some(secs(8.5));
    b.finish(m)
}

/// About 29 s on a 100 bpm grid: twelve moments of a day, four beats each, in the order they were
/// filmed (dissolves and a light leak, a soft pulse), a title over the first, a closing line over
/// the last; a tape look.
pub(super) fn day_in_30s() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Day in 30s", 4, 2);
    let beat = frames(18);
    let len = Tick(beat.0 * 4);
    let mut moments: Vec<ClipId> = Vec::new();
    for k in 0..12usize {
        let at = Tick(len.0 * k as i64);
        let item = b.nth_scene(&format!("Moment {}", k + 1), k);
        let c = b.clip(TrackKind::Video, 0, item, at, len);
        b.pulse(c, len, Tick(beat.0 * 2), 3.0);
        if let Some(&prev) = moments.last() {
            let (t, d) = if k % 4 == 0 { ("crt_light_leak", frames(14)) } else { ("crt_mix", frames(10)) };
            b.transition(TrackKind::Video, 0, t, Some(prev), Some(c), d, TransitionAlign::CenterAtCut);
        }
        moments.push(c);
    }
    let total = Tick(len.0 * 12);
    b.transition(TrackKind::Video, 0, "dip_to_black", moments.last().copied(), None, frames(15), TransitionAlign::EndAtCut);
    creator_look(&mut b, 3, Tick::ZERO, total, "vhs");
    let heading = title(&mut b, 1, Tick::ZERO, len, "MY DAY", 120.0, -260.0);
    let g = b.graphic("Template Graphics");
    let closing_at = Tick(len.0 * 11);
    let closing = b.clip(TrackKind::Video, 2, g, closing_at, len);
    b.tag(closing, 1, "Same time tomorrow?", (960.0, 860.0), 52.0, "Bold", rgb(0x111827), white(), 92.0);
    b.fade_in_out(closing, len, frames(8));
    b.sfx(1, Tick::ZERO, "sound.pop");
    let music = music(&mut b, 0, total, 0.0);

    let mut m = manifest(
        "day-in-30s",
        "Day in 30s",
        "Vlog",
        "A vertical day in thirty seconds that fills itself: twelve moments in the order they were filmed, four beats each with dissolves and a light leak, a title over the first and a closing line over the last — on tape.",
        &["day in my life", "vlog", "routine", "diary", "travel", "shorts", "reels", "30 seconds"],
    );
    m.slots = vec![
        with_ai(
            beat_slot(
                slot(
                    "moments",
                    SlotKind::Video,
                    "Moments",
                    "Twelve moments, morning to night (fewer are reused)",
                    0,
                    12,
                    1,
                    moments.iter().enumerate().map(|(i, c)| place(*c, i as u32)).collect(),
                ),
                secs(1.2),
                secs(4.8),
            ),
            in_order(shot_ai(Purpose::Broll, None, "")),
        ),
        with_ai(
            text_slot("title", "Title", "Over the first moment", 1, &[(heading, 1)], 16),
            text_ai("A title for the day, 2 to 4 words, in capitals", 16, "moments"),
        ),
        with_ai(
            text_slot("closing", "Closing line", "Over the last moment", 2, &[(closing, 1)], 24),
            text_ai("A warm closing line of 2 to 5 words", 24, "moments"),
        ),
        music_slot(3, music, "A song; the moments change on its beat", "happy", "lofi", 100.0, 251),
    ];
    let tall = vec![title_at(heading, 0.24, 84.0, 0.03), text_at(closing, 1, 0.58, 40.0)];
    let square = vec![title_at(heading, 0.2, 76.0, 0.05), text_at(closing, 1, 0.8, 40.0)];
    m.variants = variants4(tall, square.clone(), square);
    m.default_variant = "9:16".into();
    m.look = Some(Look::Lumetri("Warm Negative".into()));
    m.beat_grid = steady_grid(100.0, total);
    m.poster = Some(secs(5.0));
    b.finish(m)
}
