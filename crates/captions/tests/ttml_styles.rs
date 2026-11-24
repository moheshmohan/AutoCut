//! TTML import keeps inherited and reset text styles (craft-libs fdbb9c2471): italic, bold and
//! underline set on `<body>` or a timed `<span>` cue root reach the cue text; `normal`,
//! `noUnderline` and `inherit` reset or restore the parent value; styling is read from the TTML
//! styling namespaces (modern and 2006 drafts) with unqualified attributes as a fallback, and
//! foreign namespaces are ignored.

use filmcraft_captions::{Format, WriteOptions, parse, write};
use filmcraft_time::{TICKS_PER_SECOND, Tick};

fn caption(xml: &str, format: Format) -> String {
    let doc = parse(xml.as_bytes(), format).unwrap();
    assert_eq!(doc.cues.len(), 1);
    assert_eq!((doc.cues[0].start, doc.cues[0].end), (Tick(TICKS_PER_SECOND), Tick(TICKS_PER_SECOND * 2)));
    assert!(doc.warnings.is_empty(), "{:?}", doc.warnings);
    doc.cues[0].text.clone()
}

// ---- body styles ----------------------------------------------------------------------------

#[test]
fn body_inline_styles_are_inherited_by_paragraphs() {
    for format in [Format::Ttml, Format::Dfxp] {
        for (attribute, expected) in
            [(r#"tts:fontStyle="italic""#, "<i>Hello</i>"), (r#"tts:fontWeight="bold""#, "<b>Hello</b>"), (r#"tts:textDecoration="underline""#, "<u>Hello</u>")]
        {
            let xml = format!(
                r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body {attribute}><div><p begin="1s" end="2s">Hello</p></div></body></tt>"#
            );
            assert_eq!(caption(&xml, format), expected, "{format:?}, {attribute}");
        }
    }
}

#[test]
fn body_referenced_styles_are_inherited_by_paragraphs() {
    let xml = r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling">
<head><styling><style xml:id="emphasis" tts:fontStyle="italic" tts:fontWeight="bold"/></styling></head>
<body style="emphasis"><div><p begin="1s" end="2s">Hello</p></div></body></tt>"#;
    for format in [Format::Ttml, Format::Dfxp] {
        assert_eq!(caption(xml, format), "<i><b>Hello</b></i>");
    }
}

#[test]
fn child_spans_can_reset_inherited_and_paragraph_root_styles() {
    for format in [Format::Ttml, Format::Dfxp] {
        for (attribute, reset, tag) in [
            (r#"tts:fontStyle="italic""#, r#"tts:fontStyle="normal""#, "i"),
            (r#"tts:fontWeight="bold""#, r#"tts:fontWeight="normal""#, "b"),
            (r#"tts:textDecoration="underline""#, r#"tts:textDecoration="noUnderline""#, "u"),
        ] {
            for (body, paragraph) in [(attribute, ""), ("", attribute)] {
                let xml = format!(
                    r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body {body}><div><p {paragraph} begin="1s" end="2s">Before<span {reset}>plain</span>After</p></div></body></tt>"#
                );
                assert_eq!(caption(&xml, format), format!("<{tag}>Before</{tag}>plain<{tag}>After</{tag}>"), "{format:?}, {body}, {paragraph}, {reset}");
            }
        }
    }
}

#[test]
fn draft_inherit_keeps_parent_styles() {
    for (parent, inherited, expected) in [
        (r#"tts:fontStyle="italic""#, r#"tts:fontStyle="inherit""#, "<i>Hello</i>"),
        (r#"tts:fontWeight="bold""#, r#"tts:fontWeight="inherit""#, "<b>Hello</b>"),
        (r#"tts:textDecoration="underline""#, r#"tts:textDecoration="inherit""#, "<u>Hello</u>"),
    ] {
        for cue in [format!(r#"<p {inherited} begin="1s" end="2s">Hello</p>"#), format!(r#"<p><span {inherited} begin="1s" end="2s">Hello</span></p>"#)] {
            let xml = format!(
                r#"<tt xmlns="http://www.w3.org/2006/04/ttaf1" xmlns:tts="http://www.w3.org/2006/04/ttaf1#style"><body {parent}><div>{cue}</div></body></tt>"#
            );
            assert_eq!(caption(&xml, Format::Dfxp), expected, "{parent}, {cue}");
        }
    }
}

#[test]
fn referenced_draft_inherit_restores_the_parent_after_earlier_styles() {
    for (property, on, off, tag) in [
        ("fontStyle", r#"tts:fontStyle="italic""#, r#"tts:fontStyle="normal""#, "i"),
        ("fontWeight", r#"tts:fontWeight="bold""#, r#"tts:fontWeight="normal""#, "b"),
        ("textDecoration", r#"tts:textDecoration="underline""#, r#"tts:textDecoration="noUnderline""#, "u"),
    ] {
        let inherit = format!(r#"tts:{property}="inherit""#);
        for (parent, first, expected) in [(on, off, format!("<{tag}>Hello</{tag}>")), ("", on, "Hello".to_string())] {
            for cue in [
                r#"<p style="first restore" begin="1s" end="2s">Hello</p>"#.to_string(),
                format!(r#"<p style="first" {inherit} begin="1s" end="2s">Hello</p>"#),
            ] {
                let xml = format!(
                    r#"<tt xmlns="http://www.w3.org/2006/04/ttaf1" xmlns:tts="http://www.w3.org/2006/04/ttaf1#style"><head><styling><style xml:id="first" {first}/><style xml:id="restore" {inherit}/></styling></head><body {parent}><div>{cue}</div></body></tt>"#
                );
                assert_eq!(caption(&xml, Format::Dfxp), expected, "{parent}, {first}, {cue}");
            }
        }
    }
}

#[test]
fn unknown_font_values_do_not_reset_inherited_styles() {
    for (parent, unsupported, expected) in [
        (r#"tts:fontStyle="italic""#, r#"tts:fontStyle="garbage""#, "<i>Hello</i>"),
        (r#"tts:fontWeight="bold""#, r#"tts:fontWeight="bolder""#, "<b>Hello</b>"),
    ] {
        let xml = format!(
            r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body {parent}><div><p {unsupported} begin="1s" end="2s">Hello</p></div></body></tt>"#
        );
        assert_eq!(caption(&xml, Format::Ttml), expected);
    }
}

#[test]
fn referenced_normal_styles_override_inherited_values() {
    for format in [Format::Ttml, Format::Dfxp] {
        for (attribute, reset) in [
            (r#"tts:fontStyle="italic""#, r#"tts:fontStyle="normal""#),
            (r#"tts:fontWeight="bold""#, r#"tts:fontWeight="normal""#),
            (r#"tts:textDecoration="underline""#, r#"tts:textDecoration="noUnderline""#),
        ] {
            let xml = format!(
                r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><head><styling><style xml:id="plain" {reset}/></styling></head><body {attribute}><div><p style="plain" begin="1s" end="2s">Hello</p></div></body></tt>"#
            );
            assert_eq!(caption(&xml, format), "Hello", "{format:?}, {attribute}, {reset}");
        }
        let xml = r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><head><styling><style xml:id="plain" tts:fontStyle="normal"/></styling></head><body tts:fontStyle="italic" tts:fontWeight="bold"><div><p style="plain" begin="1s" end="2s">Hello</p></div></body></tt>"#;
        assert_eq!(caption(xml, format), "<b>Hello</b>", "{format:?}, unspecified bold remains inherited");
    }
}

#[test]
fn pending_spaces_keep_their_source_text_style() {
    for format in [Format::Ttml, Format::Dfxp] {
        for (body_style, text, expected) in [
            ("", r#"Before<span tts:textDecoration="underline"> world</span>After"#, "Before<u> world</u>After"),
            (r#"tts:textDecoration="underline""#, r#"Before<span tts:textDecoration="noUnderline"> world</span>After"#, "<u>Before</u> world<u>After</u>"),
            (r#"tts:textDecoration="underline""#, r#"Before <span tts:textDecoration="noUnderline">world</span>After"#, "<u>Before </u>world<u>After</u>"),
            ("", r#"Before<span tts:textDecoration="underline">world </span>After"#, "Before<u>world </u>After"),
            ("", r#"Before<span tts:textDecoration="underline"> </span><span tts:textDecoration="noUnderline"> </span>After"#, "Before<u> </u>After"),
            ("", r#"Before<span tts:textDecoration="noUnderline"> </span><span tts:textDecoration="underline"> </span>After"#, "Before After"),
            ("", r#"<span tts:textDecoration="underline"> </span><span tts:textDecoration="noUnderline"> </span>After"#, "After"),
            (r#"tts:textDecoration="underline""#, r#"Before<span style="plain"> world</span>After"#, "<u>Before</u> world<u>After</u>"),
        ] {
            for cue in [format!(r#"<p begin="1s" end="2s">{text}</p>"#), format!(r#"<p><span begin="1s" end="2s">{text}</span></p>"#)] {
                let xml = format!(
                    r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><head><styling><style xml:id="plain" tts:textDecoration="noUnderline"/></styling></head><body {body_style}><div>{cue}</div></body></tt>"#
                );
                assert_eq!(caption(&xml, format), expected, "{format:?}, {body_style}, {cue}");
            }
        }
    }
}

#[test]
fn root_styles_keep_fractional_timing_and_speaker_metadata() {
    for format in [Format::Ttml, Format::Dfxp] {
        for cue in [r#"<p begin="1.5t" end="2.5t" ttm:agent="voice">Hello</p>"#, r#"<p><span begin="1.5t" end="2.5t" ttm:agent="voice">Hello</span></p>"#] {
            let xml = format!(
                r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling" xmlns:ttp="http://www.w3.org/ns/ttml#parameter" xmlns:ttm="http://www.w3.org/ns/ttml#metadata" ttp:tickRate="10"><head><metadata><ttm:agent xml:id="voice" type="person"><ttm:name type="full">Voice_One 日本</ttm:name></ttm:agent></metadata></head><body begin="5s" tts:fontStyle="italic"><div begin="2s">{cue}</div></body></tt>"#
            );
            let doc = parse(xml.as_bytes(), format).unwrap();
            assert!(doc.warnings.is_empty(), "{format:?}, {cue}: {:?}", doc.warnings);
            assert_eq!(doc.cues.len(), 1, "{format:?}, {cue}");
            let parsed = &doc.cues[0];
            assert_eq!((parsed.start, parsed.end), (Tick(TICKS_PER_SECOND * 7 + TICKS_PER_SECOND * 3 / 20), Tick(TICKS_PER_SECOND * 7 + TICKS_PER_SECOND / 4)));
            assert_eq!(parsed.text, "<i>Hello</i>", "{format:?}, {cue}");
            assert_eq!(parsed.speaker.as_deref(), Some("Voice_One 日本"), "{format:?}, {cue}");
        }
    }
}

#[test]
fn descendant_overrides_and_unstyled_body_keep_their_behavior() {
    for xml in [
        r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body tts:fontStyle="italic"><div><p tts:fontStyle="normal" begin="1s" end="2s">Hello</p></div></body></tt>"#,
        r#"<tt xmlns="http://www.w3.org/ns/ttml"><body><div><p begin="1s" end="2s">Hello</p></div></body></tt>"#,
    ] {
        assert_eq!(caption(xml, Format::Ttml), "Hello");
    }
}

#[test]
fn non_xml_spaces_remain_visible_inside_styled_cues() {
    for space in ['\u{a0}', '\u{202f}', '\u{3000}'] {
        let visible = format!("{space}A{space}B{space}");
        let xml = format!(
            r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body tts:fontStyle="italic"><div><p begin="1s" end="2s">{visible}</p></div></body></tt>"#
        );
        assert_eq!(caption(&xml, Format::Ttml), format!("<i>{visible}</i>"));
    }
}

#[test]
fn deeply_nested_inline_styles_respect_the_xml_depth_limit() {
    for format in [Format::Ttml, Format::Dfxp] {
        for depth in [123, 124, 1_000] {
            let mut xml =
                String::from(r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body><div><p begin="1s" end="2s">"#);
            xml.push_str(&"<span>".repeat(depth));
            xml.push_str("<span tts:fontStyle=\"italic\">Deep</span>");
            xml.push_str(&"</span>".repeat(depth));
            xml.push_str("</p></div></body></tt>");
            // Four containers plus the styled span leave room for 123 enclosing spans.
            if depth == 123 {
                assert_eq!(caption(&xml, format), "<i>Deep</i>");
            } else {
                assert!(parse(xml.as_bytes(), format).is_err(), "{format:?}, depth {depth}");
            }
        }
    }
}

// ---- styling namespaces -----------------------------------------------------------------------

fn check(head: &str, body: &str, expected: &str) {
    let xml = format!(
        r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling" xmlns:app="urn:example">{head}<body>{body}</body></tt>"#
    );
    for format in [Format::Ttml, Format::Dfxp] {
        let doc = parse(xml.as_bytes(), format).unwrap();
        assert!(doc.warnings.is_empty(), "{xml}");
        assert_eq!(doc.cues.len(), 1, "{xml}");
        assert_eq!((doc.cues[0].start, doc.cues[0].end), (Tick(TICKS_PER_SECOND), Tick(2 * TICKS_PER_SECOND)));
        assert_eq!(doc.cues[0].text, expected, "{format:?}, {xml}");
    }
}

#[test]
fn recognized_styling_namespaces_have_fixed_priority() {
    let namespaces = [
        "http://www.w3.org/ns/ttml#styling",
        "http://www.w3.org/2006/10/ttaf1#style",
        "http://www.w3.org/2006/10/ttaf1#styling",
        "http://www.w3.org/2006/04/ttaf1#style",
        "http://www.w3.org/2006/04/ttaf1#styling",
    ];
    for (property, positive, reset, tag) in
        [("fontStyle", "italic", "normal", "i"), ("fontWeight", "bold", "normal", "b"), ("textDecoration", "underline", "noUnderline", "u")]
    {
        for (i, high) in namespaces.iter().enumerate() {
            for low in namespaces.iter().skip(i + 1) {
                for attrs in
                    [format!(r#"lo:{property}="{reset}" hi:{property}="{positive}""#), format!(r#"hi:{property}="{positive}" lo:{property}="{reset}""#)]
                {
                    check(
                        "",
                        &format!(r#"<div><p xmlns:hi="{high}" xmlns:lo="{low}" {attrs} begin="1s" end="2s">Hello</p></div>"#),
                        &format!("<{tag}>Hello</{tag}>"),
                    );
                }
            }
        }
    }
}

#[test]
fn genuine_styling_wins_over_foreign_and_unqualified_properties() {
    for (property, positive, reset, tag) in
        [("fontStyle", "italic", "normal", "i"), ("fontWeight", "bold", "normal", "b"), ("textDecoration", "underline", "noUnderline", "u")]
    {
        for other in ["app:", ""] {
            for reversed in [false, true] {
                let actual = format!(r#"tts:{property}="{positive}""#);
                let collision = format!(r#"{other}{property}="{reset}""#);
                let attrs = if reversed { format!("{actual} {collision}") } else { format!("{collision} {actual}") };
                for body in [
                    format!(r#"<div {attrs}><p begin="1s" end="2s">Hello</p></div>"#),
                    format!(r#"<div><p {attrs} begin="1s" end="2s">Hello</p></div>"#),
                    format!(r#"<div><p><span {attrs} begin="1s" end="2s">Hello</span></p></div>"#),
                ] {
                    check("", &body, &format!("<{tag}>Hello</{tag}>"));
                }
                let head = format!(r#"<head><styling><style xml:id="emphasis" {attrs}/></styling></head>"#);
                check(&head, r#"<div style="emphasis"><p begin="1s" end="2s">Hello</p></div>"#, &format!("<{tag}>Hello</{tag}>"));
                let actual_reset = format!(r#"tts:{property}="{reset}""#);
                let foreign_positive = format!(r#"app:{property}="{positive}""#);
                let reset_attrs = if reversed { format!("{actual_reset} {foreign_positive}") } else { format!("{foreign_positive} {actual_reset}") };
                check("", &format!(r#"<div tts:{property}="{positive}"><p {reset_attrs} begin="1s" end="2s">Hello</p></div>"#), "Hello");
            }
        }
        check("", &format!(r#"<div><p app:{property}="{positive}" begin="1s" end="2s">Hello</p></div>"#), "Hello");
        check(
            "",
            &format!(r#"<div tts:{property}="{positive}"><p app:{property}="{reset}" begin="1s" end="2s">Hello</p></div>"#),
            &format!("<{tag}>Hello</{tag}>"),
        );
        // Body styling resolves before the cue walk, so namespace priority must hold on that path too.
        let xml = format!(
            r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling" xmlns:app="urn:example"><body app:{property}="{reset}" tts:{property}="{positive}"><div><p begin="1s" end="2s">Hello</p></div></body></tt>"#
        );
        for format in [Format::Ttml, Format::Dfxp] {
            let doc = parse(xml.as_bytes(), format).unwrap();
            assert_eq!(doc.cues.len(), 1);
            assert_eq!(doc.cues[0].text, format!("<{tag}>Hello</{tag}>"));
        }
    }
}

#[test]
fn foreign_style_references_and_definitions_are_ignored() {
    let real = r#"<style xml:id="emphasis" tts:fontStyle="italic"/>"#;
    let foreign = r#"<app:style xml:id="emphasis" tts:fontStyle="normal"/>"#;
    for definitions in [format!("{real}{foreign}"), format!("{foreign}{real}")] {
        let head = format!("<head><styling>{definitions}</styling></head>");
        for attrs in [r#"app:style="missing" style="emphasis""#, r#"style="emphasis" app:style="missing""#] {
            check(&head, &format!(r#"<div><p {attrs} begin="1s" end="2s">Hello</p></div>"#), "<i>Hello</i>");
        }
        check(&head, r#"<div><p app:style="emphasis" begin="1s" end="2s">Hello</p></div>"#, "Hello");
    }
}

#[test]
fn genuine_style_ids_win_over_foreign_and_legacy_ids() {
    for attrs in [
        r#"app:id="wrong" xml:id="emphasis""#,
        r#"xml:id="emphasis" app:id="wrong""#,
        r#"id="wrong" xml:id="emphasis""#,
        r#"xml:id="emphasis" id="wrong""#,
        r#"id="emphasis""#,
    ] {
        check(
            &format!(r#"<head><styling><style {attrs} tts:fontStyle="italic"/></styling></head>"#),
            r#"<div><p style="emphasis" begin="1s" end="2s">Hello</p></div>"#,
            "<i>Hello</i>",
        );
    }
    check(
        r#"<head><styling><style app:id="emphasis" tts:fontStyle="italic"/></styling></head>"#,
        r#"<div><p style="emphasis" begin="1s" end="2s">Hello</p></div>"#,
        "Hello",
    );
}

#[test]
fn modern_draft_and_unqualified_styling_remain_supported() {
    for (tt, tts, id) in [
        ("http://www.w3.org/ns/ttml", "http://www.w3.org/ns/ttml#styling", "xml:id"),
        ("http://www.w3.org/2006/04/ttaf1", "http://www.w3.org/2006/04/ttaf1#style", "id"),
        ("http://www.w3.org/2006/10/ttaf1", "http://www.w3.org/2006/10/ttaf1#style", "xml:id"),
        ("http://www.w3.org/2006/04/ttaf1", "http://www.w3.org/2006/04/ttaf1#styling", "id"),
        ("http://www.w3.org/2006/10/ttaf1", "http://www.w3.org/2006/10/ttaf1#styling", "xml:id"),
        ("", "", "id"),
    ] {
        let ns = if tt.is_empty() { String::new() } else { format!(r#"xmlns="{tt}" xmlns:tts="{tts}""#) };
        let prefix = if tts.is_empty() { "" } else { "tts:" };
        let xml = format!(
            r#"<tt {ns}><head><styling><style {id}="emphasis" {prefix}fontStyle="italic"/></styling></head><body style="emphasis"><div><p {prefix}fontWeight="bold" begin="1s" end="2s">Hello</p></div></body></tt>"#
        );
        for format in [Format::Ttml, Format::Dfxp] {
            let doc = parse(xml.as_bytes(), format).unwrap();
            assert_eq!(doc.cues.len(), 1);
            assert!(doc.warnings.is_empty());
            assert_eq!((doc.cues[0].start, doc.cues[0].end), (Tick(TICKS_PER_SECOND), Tick(2 * TICKS_PER_SECOND)));
            assert_eq!(doc.cues[0].text, "<i><b>Hello</b></i>", "{format:?}, {xml}");
        }
    }
}

// ---- timed spans ------------------------------------------------------------------------------

#[test]
fn timed_span_root_styles_match_timed_paragraph_styles() {
    for format in [Format::Ttml, Format::Dfxp] {
        for (attribute, expected) in
            [(r#"tts:fontStyle="italic""#, "<i>Hello</i>"), (r#"tts:fontWeight="bold""#, "<b>Hello</b>"), (r#"tts:textDecoration="underline""#, "<u>Hello</u>")]
        {
            let head = r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body><div>"#;
            let paragraph = format!(r#"{head}<p {attribute} begin="1s" end="2s">Hello</p></div></body></tt>"#);
            let span = format!(r#"{head}<p><span {attribute} begin="1s" end="2s">Hello</span></p></div></body></tt>"#);
            assert_eq!(caption(&span, format), expected, "{format:?}, timed span, {attribute}");
            assert_eq!(caption(&paragraph, format), expected, "{format:?}, paragraph, {attribute}");
        }
    }
}

#[test]
fn timed_span_referenced_and_nested_styles_are_kept() {
    let xml = r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling">
<head><styling><style xml:id="emphasis" tts:fontStyle="italic"/></styling></head>
<body><div><p><span style="emphasis" begin="1s" end="2s">Hello <span tts:fontWeight="bold">world</span></span></p></div></body></tt>"#;
    for format in [Format::Ttml, Format::Dfxp] {
        assert_eq!(caption(xml, format), "<i>Hello <b>world</b></i>");
    }
}

#[test]
fn body_and_paragraph_styles_are_kept_on_timed_spans() {
    let xml = r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body tts:fontStyle="italic"><div><p tts:fontWeight="bold"><span tts:textDecoration="underline" begin="1s" end="2s">Hello</span></p></div></body></tt>"#;
    for format in [Format::Ttml, Format::Dfxp] {
        assert_eq!(caption(xml, format), "<i><b><u>Hello</u></b></i>");
    }
}

#[test]
fn empty_styled_timed_spans_remain_empty() {
    let xml = r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body><div><p><span tts:fontStyle="italic" begin="1s" end="2s"> <br/> </span></p></div></body></tt>"#;
    for format in [Format::Ttml, Format::Dfxp] {
        let doc = parse(xml.as_bytes(), format).unwrap();
        assert!(doc.cues.is_empty());
        assert!(doc.warnings.is_empty());
    }
}

#[test]
fn timed_span_children_can_reset_root_styles() {
    for format in [Format::Ttml, Format::Dfxp] {
        for (attribute, reset, tag) in [
            (r#"tts:fontStyle="italic""#, r#"tts:fontStyle="normal""#, "i"),
            (r#"tts:fontWeight="bold""#, r#"tts:fontWeight="normal""#, "b"),
            (r#"tts:textDecoration="underline""#, r#"tts:textDecoration="noUnderline""#, "u"),
        ] {
            let xml = format!(
                r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body><div><p><span {attribute} begin="1s" end="2s">Before<span {reset}>plain</span>After</span></p></div></body></tt>"#
            );
            assert_eq!(caption(&xml, format), format!("<{tag}>Before</{tag}>plain<{tag}>After</{tag}>"), "{format:?}, {attribute}, {reset}");
            let xml = format!(
                r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body><div><p><span {attribute} begin="1s" end="2s"><span {reset}>Hello</span></span></p></div></body></tt>"#
            );
            assert_eq!(caption(&xml, format), "Hello", "{format:?}, {attribute}, {reset}");
        }
    }
}

/// Line breaks are content: one at the start or end of a cue is kept (in its own style), as the
/// writer writes them, and only XML whitespace next to a break goes.
#[test]
fn styled_roots_keep_edge_and_interior_lines() {
    for format in [Format::Ttml, Format::Dfxp] {
        for (text, expected) in [
            ("<br/>Hello<br/>", "<i>\nHello\n</i>"),
            ("Hello<br/><br/>world<br/>", "<i>Hello\n\nworld\n</i>"),
            ("Hello <br/>world", "<i>Hello\nworld</i>"),
            (
                r#"<span tts:fontStyle="normal"><span tts:fontStyle="italic">Hello</span><br/><span tts:fontWeight="bold">world</span></span>"#,
                "<i>Hello</i>\n<b>world</b>",
            ),
            ("<br/> <br/>Hello<br/> <br/>", "<i>\n\nHello\n\n</i>"),
            (r#"Hello<span tts:fontWeight="bold">world<br/></span>"#, "<i>Hello<b>world\n</b></i>"),
        ] {
            for cue in [
                format!(r#"<p tts:fontStyle="italic" begin="1s" end="2s">{text}</p>"#),
                format!(r#"<p><span tts:fontStyle="italic" begin="1s" end="2s">{text}</span></p>"#),
            ] {
                let xml = format!(r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body><div>{cue}</div></body></tt>"#);
                assert_eq!(caption(&xml, format), expected, "{format:?}, {cue}");
            }
        }
    }
}

#[test]
fn styled_breaks_roundtrip_through_ttml_export() {
    let xml = r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling"><body><div><p begin="1s" end="2s"><span tts:fontStyle="italic">Hello</span><br/><span tts:fontWeight="bold">world</span></p></div></body></tt>"#;
    for format in [Format::Ttml, Format::Dfxp] {
        let imported = parse(xml.as_bytes(), format).unwrap();
        assert_eq!(imported.cues[0].text, "<i>Hello</i>\n<b>world</b>");
        let exported = write(&imported, format, WriteOptions::default());
        let reparsed = parse(&exported, format).unwrap();
        assert_eq!(reparsed.cues.len(), 1);
        assert_eq!(reparsed.cues[0].text, imported.cues[0].text, "{format:?}");
    }
}

#[test]
fn unstyled_timed_span_remains_plain() {
    let xml = r#"<tt xmlns="http://www.w3.org/ns/ttml"><body><div><p><span begin="1s" end="2s">Hello</span></p></div></body></tt>"#;
    assert_eq!(caption(xml, Format::Ttml), "Hello");
}
