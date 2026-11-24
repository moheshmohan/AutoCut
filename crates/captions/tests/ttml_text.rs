//! TTML / DFXP text on export and import (craft-libs 86e59ebc08): characters XML 1.0 forbids are
//! dropped instead of making the file unreadable, and literal `<` / `>` that is not markup keeps
//! its text.

use filmcraft_captions::{Cue, Document, Format, WriteOptions, parse, write};
use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick};

/// Control characters that XML 1.0 forbids must not make the export unreadable.
#[test]
fn ttml_export_drops_xml_illegal_control_chars() {
    // every forbidden character, each between two letters that must survive
    let forbidden = (0u32..0x20).filter(|c| ![0x9, 0xA, 0xD].contains(c)).chain([0xFFFE, 0xFFFF]).filter_map(char::from_u32);
    let text: String = forbidden.flat_map(|c| ['a', c]).chain(['e']).collect();
    let want = format!("{}e", "a".repeat(31));
    for format in [Format::Ttml, Format::Dfxp] {
        let cue = Cue { start: Tick(TICKS_PER_SECOND), end: Tick(2 * TICKS_PER_SECOND), text: text.clone(), ..Default::default() };
        let doc = Document { cues: vec![cue], ..Default::default() };
        let xml = String::from_utf8(write(&doc, format, WriteOptions::default())).unwrap();
        let back = parse(xml.as_bytes(), format);
        assert!(back.is_ok(), "{format:?}: {xml}");
        assert_eq!(back.unwrap().cues[0].text, want, "{format:?}");
    }
}

fn export(text: &str, format: Format) -> String {
    let rate = FrameRate::FPS_25;
    let doc = Document { cues: vec![Cue { start: rate.tick_of(25), end: rate.tick_of(75), text: text.into(), ..Default::default() }], ..Default::default() };
    String::from_utf8(write(&doc, format, WriteOptions::default())).unwrap()
}

#[test]
fn literal_angle_brackets_keep_middle_text() {
    for format in [Format::Ttml, Format::Dfxp] {
        for (input, expected) in [("1 < 2 and 3 > 2", "1 &lt; 2 and 3 &gt; 2"), ("a<b and c>d", "a&lt;b and c&gt;d")] {
            let text = export(input, format);
            assert!(text.contains(expected), "{format:?}: {text}");
        }
    }
}

#[test]
fn real_tags_still_produce_markup() {
    for format in [Format::Ttml, Format::Dfxp] {
        let text = export("<b>text</b> x <y> z <i>it</i>", format);
        assert!(text.contains("<span tts:fontWeight=\"bold\">text</span>"), "{format:?}: {text}");
        assert!(text.contains("<span tts:fontStyle=\"italic\">it</span>"), "{format:?}: {text}");
        assert!(text.contains(" x  z "), "{format:?}: {text}");
    }
}

#[test]
fn self_closing_tags_are_still_stripped() {
    for format in [Format::Ttml, Format::Dfxp] {
        let text = export("a<br/>b <b/>c", format);
        assert!(text.contains(">ab c</p>"), "{format:?}: {text}");
    }
}

#[test]
fn quoted_gt_in_attribute_does_not_end_tag() {
    for format in [Format::Ttml, Format::Dfxp] {
        let text = export("<b title=\">\">text</b>", format);
        assert!(text.contains("<span tts:fontWeight=\"bold\">text</span>"), "{format:?}: {text}");
    }
}

#[test]
fn single_quoted_attributes_are_stripped() {
    for format in [Format::Ttml, Format::Dfxp] {
        let text = export("<b title='caption'>word</b>", format);
        assert!(text.contains("<span tts:fontWeight=\"bold\">word</span>"), "{format:?}: {text}");
        assert!(!text.contains("caption"), "{format:?}: {text}");
    }
}

#[test]
fn whitespace_after_lt_is_literal_text() {
    for format in [Format::Ttml, Format::Dfxp] {
        let text = export("1 < b > 2", format);
        assert!(text.contains("1 &lt; b &gt; 2"), "{format:?}: {text}");
        assert!(!text.contains("fontWeight"), "{format:?}: {text}");
    }
}

#[test]
fn many_unmatched_lt_is_fast() {
    let input = format!("{}>", "<".repeat(256_000));
    let start = std::time::Instant::now();
    let text = export(&input, Format::Ttml);
    assert!(start.elapsed() < std::time::Duration::from_secs(2), "took {:?}", start.elapsed());
    assert!(text.contains(&format!("{}&gt;", "&lt;".repeat(256_000))));
}
