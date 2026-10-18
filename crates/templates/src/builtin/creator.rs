//! Starters for people talking to the camera: a talking head with auto-captions, a lower-third
//! and chapter kit, an end card, a tutorial with the webcam picture-in-picture, a two-person
//! podcast and a Q&A.

use filmcraft_geom::Vec2;
use filmcraft_media::DemoScene;
use filmcraft_project::{ClipId, ParamValue, TrackKind, TransitionAlign};
use filmcraft_time::Tick;

use super::{
    Builder, box_hint, boxed, captions_slot, frames, layers_hint, manifest, music_slot, place, rgb, secs, sizes_hint, slot, speech_captions, speech_slot,
    text_slot, variant, white,
};
use crate::manifest::{Fit, Framing, Portion, ShortFit, SlotKind};
use crate::{Package, TemplateError};

const F: fn(f64) -> ParamValue = ParamValue::Float;

/// 30 s of someone talking: a hook that pops at the top, a name lower third, a call to action at
/// the end, music kept low under the voice and captions of what is said.
pub(super) fn talking_head_captions() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Talking Head + Captions", 3, 3);
    let total = secs(30.0);
    let speaker_item = b.scene("Speaker Placeholder", DemoScene::Forest);
    let speaker = b.av(0, 0, speaker_item, Tick::ZERO, total);
    let hook = b.element(1, Tick::ZERO, secs(3.5), "text.title-pop");
    b.layer_text(hook, 1, "THE ONE THING");
    b.shift_layer(hook, 1, 0.0, -300.0);
    b.shift_layer(hook, 2, 0.0, -300.0);
    let lower = b.element(1, secs(4.0), secs(6.0), "text.lower-third-slide-in");
    // above the captions
    for uid in 1..=4 {
        b.shift_layer(lower, uid, 0.0, -190.0);
    }
    let cta = b.element(2, secs(26.0), secs(4.0), "text.caption-box");
    b.layer_text(cta, 1, "Follow for more tips");
    b.shift_layer(cta, 1, 0.0, -760.0);
    b.captions(&[
        (0.4, 2.2, "Here's the one thing"),
        (2.3, 4.4, "nobody tells you"),
        (4.6, 7.0, "about editing on a phone"),
        (7.2, 9.8, "Your words appear here"),
        (10.0, 12.6, "as you say them"),
    ]);
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 1, music_item, Tick::ZERO, total);
    b.volume(music, -20.0);
    b.transition(TrackKind::Audio, 1, "constant_power", Some(music), None, frames(30), TransitionAlign::EndAtCut);
    b.sfx(2, Tick::ZERO, "sound.pop");
    b.sfx(2, secs(4.0), "sound.swoosh-up");
    b.sfx(2, secs(26.0), "sound.click");

    let mut m = manifest(
        "talking-head-captions",
        "Talking Head + Captions",
        "Reels & Shorts",
        "Thirty seconds of you talking to the camera: a hook that pops in, your name in a lower third, a call to action at the end, quiet music under your voice and captions of every word.",
        &["talking head", "captions", "subtitles", "reels", "shorts", "creator"],
    );
    m.slots = vec![
        speech_slot(slot(
            "speaker",
            SlotKind::Video,
            "You, talking",
            "A clip of you talking to the camera (up to 30 s; what you say becomes the captions)",
            0,
            1,
            1,
            vec![place(speaker, 0)],
        )),
        text_slot("hook", "Hook", "The first words on screen", 1, &[(hook, 1)], 24),
        text_slot("name", "Your name", "", 2, &[(lower, 3)], 28),
        text_slot("role", "Who you are", "Under your name", 3, &[(lower, 4)], 36),
        text_slot("cta", "Call to action", "The last line", 4, &[(cta, 1)], 40),
        captions_slot(5, &["speaker"], 20, 1),
        music_slot(6, music, "Optional: music under your voice (kept quiet)", "calm", "lofi", 90.0, 21),
    ];
    let tall = vec![layers_hint(hook, &[(1, [0.5, 0.2], 100.0), (2, [0.5, 0.245], 0.0)]), sizes_hint(lower, &[(3, 44.0), (4, 26.0)])];
    let square = vec![layers_hint(hook, &[(1, [0.5, 0.17], 84.0), (2, [0.5, 0.235], 0.0)])];
    m.variants = vec![variant("9:16", tall), variant("16:9", Vec::new()), variant("1:1", square.clone()), variant("4:5", square)];
    m.default_variant = "9:16".into();
    m.caption_style = Some(speech_captions(46.0));
    m.poster = Some(secs(5.5));
    b.finish(m)
}

/// A video with a name lower third and three chapter cards, chapter markers named after them.
pub(super) fn lower_third_chapter_kit() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Lower Thirds + Chapters", 2, 2);
    let total = secs(40.0);
    let item = b.scene("Video Placeholder", DemoScene::CityNight);
    let video = b.av(0, 0, item, Tick::ZERO, total);
    let lower = b.element(1, secs(2.0), secs(6.0), "text.lower-third-slide-in");
    b.sfx(1, secs(2.0), "sound.swoosh-up");
    let g = b.graphic("Template Graphics");
    let titles = ["Getting started", "The details", "Wrapping up"];
    let mut cards = Vec::new();
    b.chapter(Tick::ZERO, "Intro", None);
    for (k, at) in [10.0, 20.0, 30.0].into_iter().enumerate() {
        let len = secs(3.5);
        let c = b.clip(TrackKind::Video, 1, g, secs(at), len);
        b.rect(c, 1, (330.0, 150.0), (560.0, 150.0), rgb(0x111827), 85.0, 14.0);
        b.rect(c, 2, (55.0, 150.0), (10.0, 150.0), rgb(0xfacc15), 100.0, 0.0);
        b.text_ex(c, 3, &format!("CHAPTER {}", k + 1), (80.0, 122.0), 30.0, "SemiBold", rgb(0xfacc15), 0, &[]);
        let title = titles.get(k).copied().unwrap_or("Chapter");
        b.text_ex(c, 4, title, (80.0, 186.0), 52.0, "Bold", white(), 0, &[]);
        b.fade_in_out(c, len, frames(9));
        b.sfx(1, secs(at), "sound.click");
        b.chapter(secs(at), title, Some(("chapters", k as u32)));
        cards.push(c);
    }

    let mut m = manifest(
        "lower-third-chapter-kit",
        "Lower Thirds + Chapters",
        "YouTube",
        "Your video with a name lower third and three chapter cards; the chapter markers take the chapters' names, so the export's YouTube chapters match.",
        &["lower third", "chapters", "kit", "youtube", "titles"],
    );
    let chapter_places: Vec<(ClipId, u64)> = cards.iter().map(|c| (*c, 4)).collect();
    m.slots = vec![
        speech_slot(slot(
            "video",
            SlotKind::Video,
            "Your video",
            "The video the titles go over (its first 40 s; lengthen the clip after applying)",
            0,
            1,
            1,
            vec![place(video, 0)],
        )),
        text_slot("name", "Name", "Who is speaking", 1, &[(lower, 3)], 28),
        text_slot("role", "Role", "Under the name", 2, &[(lower, 4)], 36),
        text_slot("chapters", "Chapters", "The three chapter titles, in order", 3, &chapter_places, 22),
    ];
    let mut tall = vec![sizes_hint(lower, &[(3, 40.0), (4, 24.0)])];
    for c in &cards {
        tall.push(sizes_hint(*c, &[(3, 18.0), (4, 30.0)]));
    }
    m.variants = vec![variant("16:9", Vec::new()), variant("9:16", tall)];
    m.poster = Some(secs(11.0));
    b.finish(m)
}

/// 12 s end card: a blurred background, two framed "watch next" videos, a thank-you heading and a
/// subscribe button with a ringing bell.
pub(super) fn end_card_subscribe() -> Result<Package, TemplateError> {
    let mut b = Builder::new("End Card", 7, 3);
    let total = secs(12.0);
    let bg_item = b.scene("Background Placeholder", DemoScene::Aurora);
    let bg = b.clip(TrackKind::Video, 0, bg_item, Tick::ZERO, total);
    b.effect(bg, "gaussian_blur", &[("blurriness", F(40.0)), ("repeat_edge", ParamValue::Bool(true))]);
    let dark = b.matte("Shade", [0.0, 0.0, 0.0, 1.0]);
    let shade = b.clip(TrackKind::Video, 1, dark, Tick::ZERO, total);
    b.opacity(shade, 45.0);
    // the frames of the end card element: 640×360 around (590, 640) and (1330, 640)
    let boxes = [[0.140625, 0.425926, 0.333333, 0.333333], [0.526042, 0.425926, 0.333333, 0.333333]];
    let mut next = Vec::new();
    for (k, (scene, delay)) in [(DemoScene::CityNight, 0.4), (DemoScene::Dunes, 0.55)].into_iter().enumerate() {
        let item = b.scene(&format!("Watch Next {}", k + 1), scene);
        let len = total - secs(delay);
        let c = b.clip(TrackKind::Video, 2 + k, item, secs(delay), len);
        b.ramp(c, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(15), 100.0)]);
        next.push(c);
    }
    let card = b.element(4, Tick::ZERO, total, "text.end-card");
    let g = b.graphic("Template Graphics");
    // the button is drawn at the canvas centre and placed with Motion, so it pops about itself
    let sub = b.clip(TrackKind::Video, 5, g, secs(1.0), total - secs(1.0));
    b.tag(sub, 1, "SUBSCRIBE", (960.0, 540.0), 40.0, "Bold", white(), rgb(0xdc2626), 100.0);
    b.set(sub, "motion", "position", ParamValue::Vec2(Vec2::new(930.0, 985.0)));
    b.ramp(sub, "motion", "scale", &[(Tick::ZERO, 0.0), (frames(6), 108.0), (frames(10), 100.0)]);
    let bell = b.element(6, secs(1.2), total - secs(1.2), "sticker.bell");
    b.set(bell, "motion", "position", ParamValue::Vec2(Vec2::new(1115.0, 985.0)));
    b.ramp(bell, "motion", "scale", &[(Tick::ZERO, 0.0), (frames(6), 20.0), (frames(10), 17.0)]);
    b.ramp(bell, "motion", "rotation", &[(frames(10), 0.0), (frames(14), -14.0), (frames(18), 12.0), (frames(22), -8.0), (frames(26), 4.0), (frames(30), 0.0)]);
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 0, music_item, Tick::ZERO, total);
    b.volume(music, -8.0);
    b.transition(TrackKind::Audio, 0, "constant_power", Some(music), None, frames(30), TransitionAlign::EndAtCut);
    b.sfx(1, Tick::ZERO, "sound.whoosh-long");
    b.sfx(2, secs(1.0), "sound.pop");
    b.sfx(1, secs(2.3), "sound.click");

    let mut m = manifest(
        "end-card-subscribe",
        "End Card + Subscribe",
        "YouTube",
        "A 12 second end card: two framed videos to watch next over a blurred background, a thank-you heading and a subscribe button with a ringing bell.",
        &["end card", "end screen", "subscribe", "outro", "youtube"],
    );
    let mut next_slot = slot(
        "next",
        SlotKind::Video,
        "Watch next",
        "Two videos (or thumbnails) to watch next",
        0,
        2,
        1,
        next.iter().zip(boxes).enumerate().map(|(i, (c, bx))| boxed(*c, i as u32, bx)).collect(),
    );
    next_slot.fit = Fit { short: ShortFit::Loop, portion: Portion::Center, ..Fit::default() };
    let mut bg_slot = slot("background", SlotKind::Video, "Background", "Optional: a clip to blur behind the card", 1, 1, 0, vec![place(bg, 0)]);
    bg_slot.fit = Fit { short: ShortFit::Loop, portion: Portion::Center, ..Fit::default() };
    m.slots = vec![
        next_slot,
        bg_slot,
        text_slot("heading", "Heading", "", 2, &[(card, 1)], 32),
        text_slot("subheading", "Subheading", "", 3, &[(card, 2)], 48),
        text_slot("labels", "Frame labels", "Under the two frames", 4, &[(card, 4), (card, 6)], 24),
        music_slot(5, music, "Optional: music for the end card", "uplifting", "pop", 100.0, 31),
    ];
    let tall = vec![
        box_hint(next.first().copied().unwrap_or(ClipId(0)), [0.140625, 0.539844, 0.333333, 0.105469]),
        box_hint(next.get(1).copied().unwrap_or(ClipId(0)), [0.526042, 0.539844, 0.333333, 0.105469]),
    ];
    m.variants = vec![variant("16:9", Vec::new()), variant("9:16", tall)];
    m.poster = Some(secs(3.0));
    b.finish(m)
}

/// A screen recording with the webcam in a corner (picture-in-picture), a title, three step tags
/// and captions of what is said.
pub(super) fn tutorial_pip() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Tutorial: Screen + Webcam", 4, 4);
    let total = secs(60.0);
    let screen_item = b.scene("Screen Recording Placeholder", DemoScene::CityNight);
    let screen = b.av(0, 0, screen_item, Tick::ZERO, total);
    let cam_item = b.scene("Webcam Placeholder", DemoScene::Forest);
    let cam = b.av(1, 1, cam_item, Tick::ZERO, total);
    b.effect(cam, "drop_shadow", &[("opacity", F(60.0)), ("distance", F(10.0)), ("softness", F(30.0))]);
    let title = b.element(2, Tick::ZERO, secs(4.0), "text.title-pop");
    b.layer_text(title, 1, "EDIT FASTER");
    b.shift_layer(title, 1, 0.0, -330.0);
    b.shift_layer(title, 2, 0.0, -330.0);
    let g = b.graphic("Template Graphics");
    let steps = ["Step 1 · Open your project", "Step 2 · Cut the pauses", "Step 3 · Export"];
    let mut tags = Vec::new();
    for (k, at) in [5.0, 22.0, 40.0].into_iter().enumerate() {
        let len = secs(5.0);
        let c = b.clip(TrackKind::Video, 3, g, secs(at), len);
        b.tag(c, 1, steps.get(k).copied().unwrap_or(""), (960.0, 100.0), 40.0, "SemiBold", white(), rgb(0x1d4ed8), 92.0);
        b.fade_in_out(c, len, frames(8));
        b.sfx(3, secs(at), "sound.click");
        tags.push(c);
    }
    b.captions(&[(4.2, 6.8, "First, open the project"), (7.0, 9.6, "and find the timeline"), (9.8, 12.4, "Your words appear here")]);
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 2, music_item, Tick::ZERO, total);
    b.volume(music, -24.0);
    b.transition(TrackKind::Audio, 2, "constant_power", Some(music), None, frames(45), TransitionAlign::EndAtCut);
    b.sfx(3, Tick::ZERO, "sound.pop");

    let mut m = manifest(
        "tutorial-pip",
        "Tutorial: Screen + Webcam",
        "YouTube",
        "A screen recording with your webcam in the corner (picture-in-picture), a title, three step tags and captions of what you say.",
        &["tutorial", "screen recording", "webcam", "picture in picture", "how to", "captions"],
    );
    let mut screen_slot =
        slot("screen", SlotKind::Video, "Screen recording", "The screen recording (shown whole, never cropped)", 0, 1, 1, vec![place(screen, 0)]);
    screen_slot.fit = Fit { short: ShortFit::Hold, portion: Portion::Start, frame: Framing::Fit, min_speed: 0.5 };
    let cam_slot = speech_slot(slot(
        "webcam",
        SlotKind::Video,
        "Webcam",
        "You on camera, recorded with the screen",
        1,
        1,
        1,
        vec![boxed(cam, 0, [0.735, 0.665, 0.245, 0.30])],
    ));
    let step_places: Vec<(ClipId, u64)> = tags.iter().map(|c| (*c, 1)).collect();
    m.slots = vec![
        screen_slot,
        cam_slot,
        text_slot("title", "Title", "What the tutorial teaches", 2, &[(title, 1)], 24),
        text_slot("steps", "Steps", "Three step names, in order", 3, &step_places, 36),
        captions_slot(4, &["webcam", "screen"], 36, 2),
        music_slot(5, music, "Optional: quiet music under the voice", "calm", "lofi", 85.0, 41),
    ];
    let mut tall = vec![
        box_hint(screen, [0.0, 0.06, 1.0, 0.42]),
        box_hint(cam, [0.06, 0.52, 0.88, 0.40]),
        layers_hint(title, &[(1, [0.5, 0.04], 64.0), (2, [0.5, 0.06], 0.0)]),
    ];
    for c in &tags {
        tall.push(sizes_hint(*c, &[(1, 34.0)]));
    }
    m.variants = vec![variant("16:9", Vec::new()), variant("9:16", tall)];
    m.caption_style = Some(speech_captions(40.0));
    m.poster = Some(secs(6.0));
    b.finish(m)
}

/// Two people talking, side by side (stacked when vertical): a show title, name tags and
/// captions of both.
pub(super) fn podcast_two_speakers() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Podcast: Two Speakers", 4, 4);
    let total = secs(60.0);
    let host_item = b.scene("Host Placeholder", DemoScene::Forest);
    let guest_item = b.scene("Guest Placeholder", DemoScene::OceanSunset);
    let host = b.av(0, 0, host_item, Tick::ZERO, total);
    let guest = b.av(1, 1, guest_item, Tick::ZERO, total);
    let g = b.graphic("Template Graphics");
    let bar = b.clip(TrackKind::Video, 2, g, Tick::ZERO, total);
    b.tag(bar, 1, "THE DEEP DIVE · EP 12", (960.0, 62.0), 34.0, "Bold", white(), rgb(0x0f172a), 82.0);
    let tag_len = secs(8.0);
    let tags = b.clip(TrackKind::Video, 3, g, secs(1.0), tag_len);
    b.tag(tags, 1, "Sam Rivera", (480.0, 150.0), 36.0, "SemiBold", rgb(0x111827), white(), 92.0);
    b.tag(tags, 2, "Jordan Lee", (1440.0, 150.0), 36.0, "SemiBold", rgb(0x111827), white(), 92.0);
    b.fade_in_out(tags, tag_len, frames(10));
    b.captions(&[(1.0, 3.4, "Welcome back to the show"), (3.6, 6.0, "Today we talk about"), (6.2, 8.8, "Your words appear here")]);
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 2, music_item, Tick::ZERO, secs(6.0));
    b.volume(music, -12.0);
    b.transition(TrackKind::Audio, 2, "constant_power", Some(music), None, frames(45), TransitionAlign::EndAtCut);
    b.sfx(3, secs(1.0), "sound.swoosh-up");

    let mut m = manifest(
        "podcast-two-speakers",
        "Podcast: Two Speakers",
        "YouTube",
        "A conversation: two speakers side by side (stacked in the vertical and square layouts), the show's title, name tags and captions of both voices.",
        &["podcast", "interview", "conversation", "long-form", "split screen", "captions"],
    );
    m.slots = vec![
        speech_slot(slot("host", SlotKind::Video, "Host", "The host's camera", 0, 1, 1, vec![boxed(host, 0, [0.0, 0.0, 0.5, 1.0])])),
        speech_slot(slot("guest", SlotKind::Video, "Guest", "The guest's camera", 1, 1, 1, vec![boxed(guest, 0, [0.5, 0.0, 0.5, 1.0])])),
        text_slot("show", "Show title", "The show and episode", 2, &[(bar, 1)], 32),
        text_slot("host_name", "Host's name", "", 3, &[(tags, 1)], 24),
        text_slot("guest_name", "Guest's name", "", 4, &[(tags, 2)], 24),
        captions_slot(5, &["host", "guest"], 26, 2),
        music_slot(6, music, "Optional: an intro sting (the first 6 s)", "calm", "lofi", 90.0, 51),
    ];
    let stacked = |top: f64| -> Vec<super::Hint> {
        vec![
            box_hint(host, [0.0, 0.0, 1.0, 0.5]),
            box_hint(guest, [0.0, 0.5, 1.0, 0.5]),
            layers_hint(bar, &[(1, [0.5, 0.045], 28.0)]),
            layers_hint(tags, &[(1, [0.5, top], 32.0), (2, [0.5, 0.5 + top], 32.0)]),
        ]
    };
    m.variants = vec![variant("16:9", Vec::new()), variant("9:16", stacked(0.1)), variant("1:1", stacked(0.15)), variant("4:5", stacked(0.12))];
    m.caption_style = Some(speech_captions(40.0));
    m.poster = Some(secs(3.0));
    b.finish(m)
}

/// Three questions on speech-bubble cards, each answered on camera, with captions.
pub(super) fn qa_session() -> Result<Package, TemplateError> {
    let mut b = Builder::new("Q&A", 3, 3);
    let matte = b.matte("Card Background", rgb(0x0b1020));
    let questions = ["What camera do you use?", "How long does an edit take?", "Any tips for beginners?"];
    let (card_len, answer_len) = (secs(2.0), secs(6.0));
    let mut prev: Option<ClipId> = None;
    let (mut answers, mut bubbles) = (Vec::new(), Vec::new());
    for k in 0..3usize {
        let at = secs(8.0 * k as f64);
        let card = b.clip(TrackKind::Video, 0, matte, at, card_len);
        if let Some(p) = prev {
            b.transition(TrackKind::Video, 0, "gl_directionalwarp", Some(p), Some(card), frames(10), TransitionAlign::CenterAtCut);
        }
        let q = b.element(1, at, card_len, "sticker.speech-bubble");
        b.layer_text(q, 2, questions.get(k).copied().unwrap_or(""));
        b.text_size(q, 2, 64.0);
        b.ramp(q, "motion", "scale", &[(Tick::ZERO, 60.0), (frames(6), 104.0), (frames(10), 100.0)]);
        b.ramp(q, "opacity", "opacity", &[(Tick::ZERO, 0.0), (frames(4), 100.0), (card_len - frames(5), 100.0), (card_len, 0.0)]);
        b.sfx(2, at, "sound.pop");
        let item = b.nth_scene(&format!("Answer {}", k + 1), k + 1);
        let a = b.av(0, 0, item, at + card_len, answer_len);
        b.transition(TrackKind::Video, 0, "gl_directionalwarp", Some(card), Some(a), frames(10), TransitionAlign::CenterAtCut);
        prev = Some(a);
        answers.push(a);
        bubbles.push(q);
    }
    b.captions(&[(2.3, 4.6, "I film on my phone"), (4.8, 7.4, "Your answers appear here")]);
    let total = secs(24.0);
    let music_item = b.music_placeholder();
    let music = b.clip(TrackKind::Audio, 1, music_item, Tick::ZERO, total);
    b.volume(music, -22.0);
    b.transition(TrackKind::Audio, 1, "constant_power", Some(music), None, frames(30), TransitionAlign::EndAtCut);

    let mut m = manifest(
        "qa-session",
        "Q&A",
        "Social",
        "Three questions on speech-bubble cards, each followed by your answer on camera, with captions of what you say.",
        &["q&a", "questions", "answers", "faq", "talking", "captions"],
    );
    let bubble_places: Vec<(ClipId, u64)> = bubbles.iter().map(|c| (*c, 2)).collect();
    m.slots = vec![
        speech_slot(slot(
            "answers",
            SlotKind::Video,
            "Answers",
            "Three clips of you answering, in order",
            0,
            3,
            1,
            answers.iter().enumerate().map(|(i, c)| place(*c, i as u32)).collect(),
        )),
        text_slot("questions", "Questions", "The three questions, in order", 1, &bubble_places, 48),
        captions_slot(2, &["answers"], 22, 1),
        music_slot(3, music, "Optional: quiet music under the answers", "calm", "lofi", 95.0, 61),
    ];
    let tall: Vec<super::Hint> = bubbles.iter().map(|c| sizes_hint(*c, &[(2, 52.0)])).collect();
    m.variants = vec![variant("9:16", tall.clone()), variant("16:9", Vec::new()), variant("1:1", tall.clone()), variant("4:5", tall)];
    m.default_variant = "9:16".into();
    m.caption_style = Some(speech_captions(46.0));
    m.poster = Some(secs(1.0));
    b.finish(m)
}
