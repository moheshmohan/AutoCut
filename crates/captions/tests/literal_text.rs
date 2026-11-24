//! Literal `<` and `&` stay text beside the cue's own style tags (craft-libs fdbb9c2471): the TTML
//! and EBU STL readers keep a literal `<` as `&lt;` (and an `&` as `&amp;`), as a WebVTT import
//! keeps its entities, so text like "&lt;b&gt;" never becomes a bold tag; the TTML and STL
//! writers decode those entities.

use filmcraft_captions::ttml::{Flavor, write};
use filmcraft_captions::{Cue, Document, Format, parse, srt, stl};
use filmcraft_project::plain_text;
use filmcraft_time::{TICKS_PER_SECOND, Tick};

fn caption(xml: &str, format: Format) -> String {
    let doc = parse(xml.as_bytes(), format).unwrap();
    assert_eq!(doc.cues.len(), 1);
    assert_eq!((doc.cues[0].start, doc.cues[0].end), (Tick(TICKS_PER_SECOND), Tick(TICKS_PER_SECOND * 2)));
    doc.cues[0].text.clone()
}

/// A literal `<` in cue text is kept as `&lt;`, the way a VTT import keeps its entities, so the
/// style tags around it stay tags and plain text (burn-in, export) decodes the bracket instead of
/// cutting at it; a `>` starts no tag and stays literal.
#[test]
fn ttml_literal_brackets_keep_their_words_and_their_style() {
    for (xml, stored, plain) in [
        (
            r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body tts:fontStyle="italic"><div><p begin="1s" end="2s">1 &lt; 2</p></div></body></tt>"#,
            "<i>1 &lt; 2</i>",
            "1 < 2",
        ),
        (
            r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body><div><p begin="1s" end="2s">a <span tts:fontWeight="bold">b &gt; c</span> d</p></div></body></tt>"#,
            "a <b>b > c</b> d",
            "a b > c d",
        ),
        (
            r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body><div><p begin="1s" end="2s">Use &lt;<span tts:fontWeight="bold">input</span>&gt;</p></div></body></tt>"#,
            "Use &lt;<b>input</b>>",
            "Use <input>",
        ),
        (
            r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body tts:fontStyle="italic"><div><p begin="1s" end="2s">2 &lt; 3 &gt; 1</p></div></body></tt>"#,
            "<i>2 &lt; 3 > 1</i>",
            "2 < 3 > 1",
        ),
        (
            // a literal "&lt;" in the source text (an escaped ampersand) stays distinct from `<`
            r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body><div><p begin="1s" end="2s">Type &lt; and &amp;lt;</p></div></body></tt>"#,
            "Type &lt; and &amp;lt;",
            "Type < and &lt;",
        ),
    ] {
        let text = caption(xml, Format::Ttml);
        assert_eq!(text, stored);
        assert_eq!(plain_text(&text), plain);
        // written back out, the XML text is the source text escaped once (not the stored entity
        // escaped again), and imported again, in both flavours, the cue reads the same
        let doc = parse(xml.as_bytes(), Format::Ttml).unwrap();
        for flavor in [Flavor::Imsc1, Flavor::Dfxp] {
            let again = write(&doc, flavor, None, "en");
            assert!(!again.contains("&amp;lt;") || stored.contains("&amp;lt;"), "{flavor:?}: {again}");
            if stored == "<i>1 &lt; 2</i>" {
                assert!(again.contains(">1 &lt; 2<"), "{flavor:?}: {again}");
            }
            assert_eq!(caption(&again, Format::Ttml), stored, "{flavor:?}: {again}");
        }
    }
}

fn document(text: &str) -> Document {
    Document {
        cues: vec![Cue { start: Tick(TICKS_PER_SECOND), end: Tick(2 * TICKS_PER_SECOND), text: text.into(), ..Default::default() }],
        ..Default::default()
    }
}

#[test]
fn srt_roundtrips_literal_brackets_and_style_tags() {
    for text in ["1 &lt; 2 & <i>x</i>", "x &lt;y> <u>z</u>", "<i>Type &lt; and &amp;lt;</i>"] {
        let expected = plain_text(text);
        let mut doc = document(text);
        for _ in 0..2 {
            let encoded = srt::write(&doc);
            doc = srt::parse(&encoded).unwrap();
            assert_eq!(doc.cues.len(), 1);
            assert_eq!(plain_text(&doc.cues[0].text), expected, "{text:?}: {encoded}");
            assert_eq!(doc.cues[0].text, text, "literal entities and genuine tags stay distinct");
        }
    }
}

#[test]
fn stl_roundtrips_literal_brackets_and_style_controls() {
    // The italic control and 110 letters put the literal '<' at the last text-field byte.
    let boundary = format!("<i>{}&lt;i></i> &amp;lt;", "x".repeat(110));
    for (text, stored) in [
        ("1 &lt; 2 & <i>x</i>", "1 &lt; 2 &amp; <i>x</i>"),
        ("x &lt;y> <u>z</u>", "x &lt;y> <u>z</u>"),
        ("<i>Type &lt; and &amp;lt;</i>", "<i>Type &lt; and &amp;lt;</i>"),
        ("<u>literal &lt;i> text</u> <i>styled</i>", "<u>literal &lt;i> text</u> <i>styled</i>"),
        (boundary.as_str(), boundary.as_str()),
    ] {
        let expected = plain_text(text);
        let mut doc = document(text);
        for _ in 0..2 {
            let encoded = stl::write(&doc, None);
            if text == boundary {
                assert!(encoded.len() > 1024 + 128, "the fixture crosses a text-field boundary");
            }
            doc = stl::parse(&encoded).unwrap();
            assert_eq!(doc.cues.len(), 1);
            assert_eq!(plain_text(&doc.cues[0].text), expected, "{text:?}: {:?}", doc.cues[0].text);
            assert_eq!(doc.cues[0].text, stored, "literal text and styled ranges remain distinct");
            assert_eq!(stl::write(&doc, None), encoded, "the next export preserves text and style-control bytes");
            for tag in ["<i>", "</i>", "<u>", "</u>"] {
                assert_eq!(doc.cues[0].text.contains(tag), text.contains(tag), "style control {tag}");
            }
        }
    }
}
