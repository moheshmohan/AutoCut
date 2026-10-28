//! Script parsing: speech text and pause markers.
//!
//! A pause marker is `[pause <n>s]` or `[pause <n>ms]` (case-insensitive, `n` a decimal number,
//! spaces optional: `[pause 1.5s]`, `[Pause 500 ms]`). Pauses are capped at [`MAX_PAUSE_MS`]. A
//! bracket that is not a well-formed marker is ordinary text. The Text to Speech panel's
//! *Add pause* button inserts `[pause 1s]`.
//!
//! Malayalam narration also reads style tags ([`parse_marked`], [`crate::styles::tag`]):
//! `[excited]`, `[calm]`, … switch the voice style for what follows, `[plain]` / `[neutral]` go
//! back to the voice's own, `[breath]` is a breath-length pause. [`parse`] (every other voice)
//! leaves them as text, as before.

/// Longest single pause.
pub const MAX_PAUSE_MS: u32 = 10_000;

/// A piece of a script.
#[derive(Clone, Debug, PartialEq)]
pub enum Segment {
    Speech(String),
    Pause { millis: u32 },
}

/// The marker *Add pause* inserts.
pub const PAUSE_MARKER: &str = "[pause 1s]";

/// A piece of a script with its markers ([`parse_marked`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Mark {
    Speech(String),
    Pause {
        millis: u32,
    },
    /// A style tag (`[excited]`, `[plain]`) or `[breath]`.
    Tag(crate::styles::Tag),
}

/// Split `text` into speech and pauses, in order. Empty speech pieces are dropped.
pub fn parse(text: &str) -> Vec<Segment> {
    split(text, false)
        .into_iter()
        .filter_map(|m| match m {
            Mark::Speech(s) => Some(Segment::Speech(s)),
            Mark::Pause { millis } => Some(Segment::Pause { millis }),
            Mark::Tag(_) => None,
        })
        .collect()
}

/// Split `text` into speech, pauses and style tags, in order (see the module docs). Empty speech
/// pieces are dropped; speech that is only white space between markers is kept when it holds a
/// line break (a Malayalam reading pauses at one).
pub fn parse_marked(text: &str) -> Vec<Mark> {
    split(text, true)
}

/// `text` without its pause markers and style tags: the words a voice says (each marker a space,
/// white space made single spaces).
pub fn strip_marks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for m in parse_marked(text) {
        match m {
            Mark::Speech(s) => out.push_str(&s),
            Mark::Pause { .. } | Mark::Tag(_) => out.push(' '),
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// [`parse_marked`] (`tags`), or [`parse`] (pause markers only).
fn split(text: &str, tags: bool) -> Vec<Mark> {
    let mut out = Vec::new();
    let mut speech = String::new();
    let mut rest = text;
    while let Some(open) = rest.find('[') {
        let (before, from) = rest.split_at(open);
        speech.push_str(before);
        let close = from.find(']');
        let marker = close.and_then(|c| from.get(..=c));
        let found = marker
            .and_then(pause_millis)
            .map(|millis| Mark::Pause { millis })
            .or_else(|| marker.filter(|_| tags).and_then(|m| m.get(1..m.len().saturating_sub(1))).and_then(crate::styles::tag).map(Mark::Tag));
        match (close, found) {
            (Some(close), Some(m)) => {
                flush(&mut out, &mut speech, tags);
                out.push(m);
                rest = from.get(close + 1..).unwrap_or("");
            }
            _ => {
                speech.push('[');
                rest = from.get(1..).unwrap_or("");
            }
        }
    }
    speech.push_str(rest);
    flush(&mut out, &mut speech, tags);
    out
}

fn flush(out: &mut Vec<Mark>, speech: &mut String, keep_breaks: bool) {
    if !speech.trim().is_empty() || (keep_breaks && speech.contains('\n')) {
        out.push(Mark::Speech(std::mem::take(speech)));
    }
    speech.clear();
}

/// `[pause 1.5s]` → 1500. None if `marker` is not a pause marker.
fn pause_millis(marker: &str) -> Option<u32> {
    let inner = marker.strip_prefix('[')?.strip_suffix(']')?.trim();
    let lower = inner.to_ascii_lowercase();
    let arg = lower.strip_prefix("pause")?.trim();
    let (num, scale) = if let Some(n) = arg.strip_suffix("ms") { (n.trim(), 1.0) } else { (arg.strip_suffix('s')?.trim(), 1000.0) };
    if num.is_empty() || num.len() > 12 || !num.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return None;
    }
    let v: f64 = num.parse().ok()?;
    let ms = (v * scale).round();
    if !ms.is_finite() || ms < 0.0 {
        return None;
    }
    Some(ms.min(f64::from(MAX_PAUSE_MS)) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sp(s: &str) -> Segment {
        Segment::Speech(s.into())
    }

    #[test]
    fn splits_speech_and_pauses() {
        assert_eq!(parse("Hello. [pause 1s] World"), vec![sp("Hello. "), Segment::Pause { millis: 1000 }, sp(" World")]);
        assert_eq!(parse("[Pause 500 ms]Go"), vec![Segment::Pause { millis: 500 }, sp("Go")]);
        assert_eq!(parse("a[pause 1.5s][pause 2s]b"), vec![sp("a"), Segment::Pause { millis: 1500 }, Segment::Pause { millis: 2000 }, sp("b")]);
        assert_eq!(parse(PAUSE_MARKER), vec![Segment::Pause { millis: 1000 }]);
    }

    #[test]
    fn malformed_markers_are_text() {
        assert_eq!(parse("[pause]"), vec![sp("[pause]")]);
        assert_eq!(parse("[pause 1x] a"), vec![sp("[pause 1x] a")]);
        assert_eq!(parse("[pause 1s"), vec![sp("[pause 1s")]);
        assert_eq!(parse("[note] [pause -1s]"), vec![sp("[note] [pause -1s]")]);
        assert_eq!(parse("[[pause 1s]]"), vec![sp("["), Segment::Pause { millis: 1000 }, sp("]")]);
    }

    #[test]
    fn style_tags_are_markers_only_when_asked_for() {
        use crate::styles::Tag;
        let text = "[excited] ഇത് കൊള്ളാം! [calm]ശരി. [breath] [Plain] അതെ. [note]";
        assert_eq!(
            parse_marked(text),
            vec![
                Mark::Tag(Tag::Style("excited")),
                Mark::Speech(" ഇത് കൊള്ളാം! ".into()),
                Mark::Tag(Tag::Style("calm")),
                Mark::Speech("ശരി. ".into()),
                Mark::Tag(Tag::Breath),
                Mark::Tag(Tag::Plain),
                Mark::Speech(" അതെ. [note]".into()),
            ]
        );
        // every other voice reads them as text, as before
        assert_eq!(parse(text), vec![sp(text)]);
        assert_eq!(strip_marks(text), "ഇത് കൊള്ളാം! ശരി. അതെ. [note]");
        assert_eq!(strip_marks("a[pause 1s]b"), "a b");
        // a line break between markers is kept for the reading; plain white space is not
        assert_eq!(
            parse_marked("a.[calm]\n\n[warm]b."),
            vec![
                Mark::Speech("a.".into()),
                Mark::Tag(Tag::Style("calm")),
                Mark::Speech("\n\n".into()),
                Mark::Tag(Tag::Style("warm")),
                Mark::Speech("b.".into())
            ]
        );
        assert_eq!(parse_marked("[calm] [warm]"), vec![Mark::Tag(Tag::Style("calm")), Mark::Tag(Tag::Style("warm"))]);
        for s in ["[", "[excited", "[[calm]]", "[calm][", "]]][breath", "[ബ്രെത്ത്]", "[]", "[ ]"] {
            let _ = parse_marked(s);
            let _ = strip_marks(s);
        }
        assert_eq!(parse_marked("[[calm]]"), vec![Mark::Speech("[".into()), Mark::Tag(Tag::Style("calm")), Mark::Speech("]".into())]);
    }

    #[test]
    fn hostile_inputs_never_panic_and_pauses_are_capped() {
        assert_eq!(parse("[pause 999999999999s]"), vec![Segment::Pause { millis: MAX_PAUSE_MS }]);
        assert_eq!(parse("[pause 1e9s]"), vec![sp("[pause 1e9s]")]);
        for s in ["", "[", "]", "[[[[", "日本語[pause 1s]👍", "\u{301}[", "[pause 1.2.3s]", "\0[pause 0s]"] {
            let _ = parse(s);
        }
        assert_eq!(parse("  "), vec![]);
    }
}
