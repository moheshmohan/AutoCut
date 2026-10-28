use std::sync::Arc;

use serde_json::json;

use crate::Session;
use filmcraft_project::graphic::{LayerContent, eval_layer, layer_indices};
use filmcraft_project::text_anim::{AnimKind, AnimUnit};
use filmcraft_project::{CaptionPreset, Narration, Transcript, Word};
use filmcraft_speech::FixedTranscriber;
use filmcraft_time::{TICKS_PER_SECOND, Tick};

fn demo() -> Session {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s
}

/// The demo project with a caption track of two captions over the first A1 clip, the clip's
/// media transcribed: "make it count" then "every single day".
fn captioned() -> Session {
    let mut s = demo();
    let q = s.active_sequence().unwrap();
    let a = &q.audio_tracks[0].items[0];
    let (item, sin, start) = (a.item, a.source_in, a.start);
    let mut t = Transcript { language: "en".into(), ..Default::default() };
    for (text, x, y) in [("make", 0.2, 0.5), ("it", 0.55, 0.7), ("count", 0.75, 1.2), ("every", 2.0, 2.3), ("single", 2.35, 2.7), ("day", 2.75, 3.2)] {
        t.words.push(Word::new(text, sin + Tick::from_seconds_f64(x), sin + Tick::from_seconds_f64(y)));
    }
    t.normalize();
    s.transcriber = Some(Arc::new(FixedTranscriber { transcript: t, id: "fixed".into() }));
    s.execute("transcript.generate", json!({"items": [item.0]})).unwrap();
    s.execute("captions.newTrack", json!({"name": "Captions"})).unwrap();
    for (text, at, d) in [("make it count", 0.1, 1.5), ("every single day", 1.9, 1.5)] {
        s.execute("captions.add", json!({"track": "C1", "text": text, "time": (start + Tick::from_seconds_f64(at)).0, "durationSeconds": d})).unwrap();
    }
    s
}

fn track(s: &Session) -> &filmcraft_project::CaptionTrack {
    &s.active_sequence().unwrap().caption_tracks[0]
}

#[test]
fn animated_caption_style_presets_options_and_off() {
    let mut s = captioned();
    let r = s
        .execute(
            "captions.setAnimatedStyle",
            json!({"preset": "bold", "options": {"keywords": ["count"], "emoji": {"day": "🔥"}, "platform": "youtube", "wordsPerPage": 3, "activeColor": "#ff0066"}}),
        )
        .unwrap();
    assert_eq!(r["animated"]["preset"], "bold", "{r}");
    assert_eq!(r["animated"]["platform"], "youtube");
    let st = track(&s).style.clone();
    let a = st.animated.clone().unwrap();
    assert_eq!((a.preset, a.page_words(), a.active_color), (CaptionPreset::Bold, 3, [255, 0, 102, 255]));
    assert_eq!(a.keywords, ["count"]);
    assert_eq!(a.emoji_for("day"), Some("🔥"));
    assert_eq!(st.size, CaptionPreset::Bold.base_style().size, "the preset's look");
    // another preset keeps the keywords and the platform; the same preset keeps the options
    s.execute("captions.setAnimatedStyle", json!({"preset": "karaoke"})).unwrap();
    let k = track(&s).style.animated.clone().unwrap();
    assert_eq!((k.preset, k.keywords.len(), k.platform.is_some()), (CaptionPreset::Karaoke, 1, true));
    s.execute("captions.setAnimatedStyle", json!({"preset": "karaoke", "pop": 1.3, "position": "top", "keepStyle": true})).unwrap();
    assert!((track(&s).style.animated.as_ref().unwrap().pop_scale() - 1.3).abs() < 1e-6);
    assert_eq!(track(&s).style.anchor, filmcraft_project::CaptionAnchor::Top);
    // the list shows it, off goes back to whole captions, undo brings it back
    let l = s.execute("captions.list", json!({})).unwrap();
    assert_eq!(l["tracks"][0]["style"]["animated"]["preset"], "karaoke", "{l}");
    s.execute("captions.setAnimatedStyle", json!({"preset": "off"})).unwrap();
    assert!(track(&s).style.animated.is_none());
    s.execute("edit.undo", json!({})).unwrap();
    assert!(track(&s).style.animated.is_some());
}

/// Live (the Akku Tesla captions): the "box" preset's white words over a white toy car could not
/// be read. `panel` puts a backing panel behind each line.
#[test]
fn animated_captions_take_a_backing_panel() {
    let mut s = captioned();
    let r = s.execute("captions.setAnimatedStyle", json!({"preset": "box", "options": {"panel": true, "position": "bottom"}})).unwrap();
    assert_eq!(r["panel"], "#00000099", "{r}");
    let st = track(&s).style.clone();
    assert!(st.background && st.background_color == [0, 0, 0, 153], "{st:?}");
    // the same preset again keeps it; a colour of its own; off
    s.execute("captions.setAnimatedStyle", json!({"preset": "box", "options": {"activeColor": "#29e0ff"}})).unwrap();
    assert!(track(&s).style.background);
    s.execute("captions.setAnimatedStyle", json!({"preset": "box", "panel": "#10203080"})).unwrap();
    assert_eq!(track(&s).style.background_color, [16, 32, 48, 128]);
    let r = s.execute("captions.setAnimatedStyle", json!({"preset": "box", "panel": false})).unwrap();
    assert!(!track(&s).style.background && r["panel"].is_null(), "{r}");
    // a fully clear colour is no panel; nonsense is refused
    s.execute("captions.setAnimatedStyle", json!({"preset": "box", "panel": "#00000000"})).unwrap();
    assert!(!track(&s).style.background);
    for bad in [json!("dark"), json!(3), json!({"a": 1})] {
        assert!(s.execute("captions.setAnimatedStyle", json!({"preset": "box", "panel": bad})).is_err(), "{bad}");
    }
}

#[test]
fn animated_caption_style_refuses_bad_params() {
    let mut s = captioned();
    for p in [
        json!({}),
        json!({"preset": "neon"}),
        json!({"preset": "bold", "options": 5}),
        json!({"preset": "bold", "activeColor": "red"}),
        json!({"preset": "bold", "platform": "snapchat"}),
        json!({"preset": "bold", "position": "left"}),
        json!({"preset": "bold", "wordsPerPage": "three"}),
        json!({"preset": "bold", "keywords": 7}),
        json!({"preset": "bold", "uppercase": "yes"}),
        json!({"track": 999_999, "preset": "bold"}),
    ] {
        assert!(s.execute("captions.setAnimatedStyle", p.clone()).is_err(), "{p}");
    }
    // hostile sizes are clamped; long lists capped
    let many: Vec<String> = (0..5000).map(|i| format!("w{i}{}", "x".repeat(300))).collect();
    s.execute(
        "captions.setAnimatedStyle",
        json!({"preset": "bold", "wordsPerPage": 1e300, "pop": -4, "size": 1e9, "margin": -1, "xOffset": 99, "keywords": many}),
    )
    .unwrap();
    let st = &track(&s).style;
    let a = st.animated.as_ref().unwrap();
    assert_eq!((a.page_words(), a.pop_scale(), st.size, st.margin, st.x_offset), (8, 1.0, 400.0, 0.0, 0.5));
    assert!(a.keywords.len() <= 200 && a.keywords.iter().all(|k| k.chars().count() <= 64));
    assert!(s.execute("captions.setAnimatedStyle", json!({"preset": "bold"})).is_ok());
    let mut empty = demo();
    assert!(empty.execute("captions.setAnimatedStyle", json!({"preset": "bold"})).is_err(), "no caption track");
}

#[test]
fn word_timing_from_the_transcript() {
    let mut s = captioned();
    assert!(track(&s).captions.iter().all(|c| c.words.is_empty()));
    let r = s.execute("captions.wordTiming", json!({"from": "transcript"})).unwrap();
    assert_eq!((r["captions"].as_u64(), r["words"].as_u64()), (Some(2), Some(6)), "{r}");
    let c = &track(&s).captions[0];
    let w = c.timed_words();
    assert_eq!(w.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(), ["make", "it", "count"]);
    let a1 = s.active_sequence().unwrap().audio_tracks[0].items[0].start;
    assert_eq!(w[1].start, a1 + Tick::from_seconds_f64(0.55), "the transcript's times");
    let l = s.execute("captions.list", json!({})).unwrap();
    assert_eq!(l["tracks"][0]["captions"][1]["timedWords"], 3);
    // undoable; a bad source is refused
    s.execute("edit.undo", json!({})).unwrap();
    assert!(track(&s).captions[0].words.is_empty());
    assert!(s.execute("captions.wordTiming", json!({"from": "telepathy"})).is_err());
    // the captions transcript.createCaptions makes carry word timing already
    s.execute("transcript.createCaptions", json!({"maxChars": 18, "lines": 1})).unwrap();
    let made = &s.active_sequence().unwrap().caption_tracks[0];
    assert!(made.captions.iter().all(|c| !c.words.is_empty()), "{:?}", made.captions);
}

#[test]
fn word_timing_from_a_narration() {
    let mut s = captioned();
    let q = s.active_sequence().unwrap();
    let a = q.audio_tracks[0].items[0].clone();
    let n = Narration {
        text: "make it count [pause 1s] every single day".into(),
        language: "en-US".into(),
        voice: "test".into(),
        pitch: Default::default(),
        pace: 1.0,
        delivery: Default::default(),
        take: 0,
        numbers: Default::default(),
        temperature: None,
        emphasis: None,
        speed: None,
        expressiveness: None,
        speech_duration: Tick(3 * TICKS_PER_SECOND),
    };
    let seq = s.state.active_sequence.unwrap();
    let p = Arc::make_mut(&mut s.project);
    p.narrations.insert(a.item, n);
    // a narration clip plays its speech from the start of the file
    p.sequence_mut(seq).unwrap().audio_tracks[0].items[0].source_in = Tick::ZERO;
    let words = super::narration_words(&s);
    assert_eq!(words.len(), 6, "{words:?}");
    // in order, inside the clip, the pause between "count" and "every"
    assert!(words.windows(2).all(|w| w[0].1 <= w[1].1));
    let gap = words[3].1 - words[2].2;
    assert!(gap >= Tick(TICKS_PER_SECOND), "the pause is kept: {gap:?}");
    let r = s.execute("captions.wordTiming", json!({"from": "narration"})).unwrap();
    assert!(r["captions"].as_u64().unwrap() >= 1, "{r}");
    // no narration: refused with a reason
    let mut d = captioned();
    let e = d.execute("captions.wordTiming", json!({"from": "narration"})).unwrap_err().to_string();
    assert!(e.contains("narration"), "{e}");
}

/// A graphic clip with one text layer, `text`, and its clip id.
fn titled(text: &str) -> (Session, u64) {
    let mut s = demo();
    let r = s.execute("graphics.newText", json!({"text": text, "seconds": 4})).unwrap();
    (s, r["clip"].as_u64().unwrap())
}

fn layer(s: &Session, clip: u64) -> filmcraft_project::EffectInstance {
    let (_, it) = s.active_sequence().unwrap().find_item(filmcraft_project::ClipId(clip)).unwrap();
    let i = layer_indices(&it.effects)[0];
    it.effects[i].clone()
}

#[test]
fn animate_text_sets_in_out_and_loop() {
    let (mut s, clip) = titled("Hello there");
    let r = s
        .execute(
            "graphics.animateText",
            json!({"clip": clip, "in": {"kind": "pop", "unit": "letter", "stagger": 0.05, "easing": "Back Out"}, "out": "fade", "loop": {"kind": "wave", "amount": 50}}),
        )
        .unwrap();
    assert_eq!(r["animation"]["in"]["kind"], "pop", "{r}");
    assert_eq!(r["animation"]["out"]["kind"], "fade");
    assert_eq!(r["animation"]["loop"]["amount"], 50.0);
    let a = layer(&s, clip).layer.unwrap().anim.unwrap();
    assert_eq!((a.enter.as_ref().unwrap().kind, a.enter.as_ref().unwrap().unit), (AnimKind::Pop, AnimUnit::Letter));
    assert_eq!(a.enter.as_ref().unwrap().easing, 23);
    // the layer evaluates with its animation; the out slot can be removed alone
    let spec = eval_layer(&layer(&s, clip), Tick::ZERO, (1920, 1080)).unwrap();
    let LayerContent::Text(t) = spec.content else { panic!() };
    assert!(t.anim.is_some() && t.clock.is_some());
    s.execute("graphics.animateText", json!({"clip": clip, "out": null})).unwrap();
    let a = layer(&s, clip).layer.unwrap().anim.unwrap();
    assert!(a.out.is_none() && a.enter.is_some() && a.repeat.is_some());
    // a preset by name; clear removes everything (and the empty layer extra)
    s.execute("graphics.animateText", json!({"clip": clip, "clear": true, "preset": "typewriter"})).unwrap();
    let a = layer(&s, clip).layer.unwrap().anim.unwrap();
    assert_eq!(a.enter.unwrap().kind, AnimKind::Typewriter);
    assert!(a.repeat.is_none());
    s.execute("graphics.animateText", json!({"clip": clip, "clear": true})).unwrap();
    assert!(layer(&s, clip).layer.as_ref().is_none_or(|x| x.anim.is_none()));
    // undo brings the typewriter back
    s.execute("edit.undo", json!({})).unwrap();
    assert!(layer(&s, clip).layer.unwrap().anim.is_some());
}

#[test]
fn animate_text_refuses_bad_params_and_clamps_hostile_numbers() {
    let (mut s, clip) = titled("Hostile");
    for p in [
        json!({"clip": clip}),
        json!({"clip": clip, "in": {"kind": "wave"}}),
        json!({"clip": clip, "loop": "pop"}),
        json!({"clip": clip, "in": {"kind": "teleport"}}),
        json!({"clip": clip, "in": {"unit": "letter"}}),
        json!({"clip": clip, "in": {"kind": "fade", "unit": "paragraph"}}),
        json!({"clip": clip, "in": {"kind": "fade", "easing": "Wobble"}}),
        json!({"clip": clip, "in": {"kind": "fade", "amount": "lots"}}),
        json!({"clip": clip, "in": 42}),
        json!({"clip": clip, "preset": "nope"}),
    ] {
        assert!(s.execute("graphics.animateText", p.clone()).is_err(), "{p}");
    }
    s.execute("graphics.animateText", json!({"clip": clip, "in": {"kind": "drop", "amount": 1e300, "stagger": -5, "duration": 1e9, "easing": 4000}}))
        .unwrap_err();
    s.execute("graphics.animateText", json!({"clip": clip, "in": {"kind": "drop", "amount": 1e300, "stagger": -5, "duration": 1e9, "easing": 3}})).unwrap();
    let a = layer(&s, clip).layer.unwrap().anim.unwrap().enter.unwrap();
    assert_eq!((a.amount, a.stagger, a.duration), (1000.0, 0.0, 60.0));
    // a shape layer is not text
    s.execute("graphics.newShape", json!({"clip": clip, "shape": "rectangle"})).unwrap();
    let e = s.execute("graphics.animateText", json!({"clip": clip, "layer": 1, "in": "fade"}));
    assert!(e.is_err(), "{e:?}");
}

#[test]
fn text_style_glow_inner_shadow_curve_and_stroke3() {
    let (mut s, clip) = titled("Neon");
    let r = s
        .execute(
            "graphics.textStyle",
            json!({"clip": clip, "glow": {"color": "#00e5ff", "radius": 30, "intensity": 150}, "innerShadow": {"distance": 6, "opacity": 80}, "curve": 40, "stroke3": {"width": 12, "type": "center"}}),
        )
        .unwrap();
    assert_eq!(r["set"]["glow"], true, "{r}");
    let spec = eval_layer(&layer(&s, clip), Tick::ZERO, (1920, 1080)).unwrap();
    let g = spec.appearance.glow.clone().unwrap();
    assert!((g.radius - 30.0).abs() < 1e-6 && (g.intensity - 1.5).abs() < 1e-6 && g.color[0] < 0.01);
    let is = spec.appearance.inner_shadow.clone().unwrap();
    assert!((is.color[3] - 0.8).abs() < 1e-6);
    assert_eq!(spec.appearance.strokes.len(), 1);
    assert_eq!(spec.appearance.strokes[0].2, 1, "centre");
    let LayerContent::Text(t) = &spec.content else { panic!() };
    assert!((t.curve.unwrap().bend - 0.4).abs() < 1e-6);
    // off again
    s.execute("graphics.textStyle", json!({"clip": clip, "glow": false, "curve": false, "innerShadow": false, "stroke3": false})).unwrap();
    let spec = eval_layer(&layer(&s, clip), Tick::ZERO, (1920, 1080)).unwrap();
    let LayerContent::Text(t) = &spec.content else { panic!() };
    assert!(spec.appearance.glow.is_none() && spec.appearance.inner_shadow.is_none() && t.curve.is_none() && spec.appearance.strokes.is_empty());
    for p in [
        json!({"clip": clip}),
        json!({"clip": clip, "glow": {"color": "neon"}}),
        json!({"clip": clip, "curve": "round"}),
        json!({"clip": clip, "stroke3": {"type": "dotted"}}),
        json!({"clip": clip, "glow": {"radius": "big"}}),
    ] {
        assert!(s.execute("graphics.textStyle", p.clone()).is_err(), "{p}");
    }
    // hostile numbers clamp
    s.execute("graphics.textStyle", json!({"clip": clip, "glow": {"radius": 1e300, "intensity": -5}, "curve": 1e9})).unwrap();
    let spec = eval_layer(&layer(&s, clip), Tick::ZERO, (1920, 1080)).unwrap();
    let g = spec.appearance.glow.unwrap();
    assert!(g.radius <= 1000.0 && g.intensity >= 0.0);
}

#[test]
fn animated_captions_are_checked_against_safe_zones_on_their_pages() {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"name": "Short", "width": 1080, "height": 1920, "fps": 30})).unwrap();
    s.execute("captions.newTrack", json!({})).unwrap();
    let text = "the quick brown fox jumps over the lazy dog and keeps running far away";
    s.execute("captions.add", json!({"track": "C1", "text": text, "seconds": 0.0, "durationSeconds": 4.0})).unwrap();
    // the old style, centred on the frame near the bottom, reaches into the bottom zone
    s.execute("captions.setStyle", json!({"track": "C1", "size": 90, "anchor": "bottom", "margin": 0.05})).unwrap();
    let before = s.execute("sequence.checkSafeZones", json!({"platform": "youtube"})).unwrap();
    assert!(!before["overlaps"].as_array().unwrap().is_empty(), "{before}");
    // Bold animated captions kept to YouTube's safe zone: a page at a time, every page clear
    s.execute("captions.setAnimatedStyle", json!({"preset": "bold", "platform": "youtube"})).unwrap();
    let r = s.execute("sequence.checkSafeZones", json!({"platform": "youtube"})).unwrap();
    assert!(r["checked"].as_u64().unwrap() >= 3, "a box per page: {r}");
    assert!(r["overlaps"].as_array().unwrap().is_empty(), "{r}");
}
