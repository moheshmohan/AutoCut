//! Advanced SubStation Alpha / SubStation Alpha (`.ass`, `.ssa`).
//!
//! ```text
//! [Script Info]
//! ScriptType: v4.00+
//! PlayResX: 1920
//! PlayResY: 1080
//!
//! [V4+ Styles]
//! Format: Name, Fontname, Fontsize, PrimaryColour, …, Alignment, MarginL, MarginR, MarginV, Encoding
//! Style: Default,Arial,54,&H00FFFFFF,…,2,96,96,86,1
//!
//! [Events]
//! Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
//! Dialogue: 0,0:00:01.00,0:00:03.50,Default,Ann,0,0,0,,{\i1}Hello{\i0}\Nthere
//! ```
//!
//! **Reading.** The `[Script Info]` play resolution (`PlayResX` / `PlayResY`, 384 × 288 when
//! absent, one derived from the other when only one is given), the styles of `[V4+ Styles]`
//! (ASS, numpad alignment) or `[V4 Styles]` (SSA, the legacy 1–3 / 5–7 / 9–11 alignment) and the
//! `Dialogue` lines of `[Events]`, each mapped field by field through its section's `Format`
//! line (the format's default field order when there is none). Other sections (`[Fonts]`,
//! `[Graphics]`, Aegisub's project data) and other event kinds (`Comment`, `Picture`, …) are
//! skipped.
//!
//! - The style most dialogue lines use becomes the track style: font, size (scaled from the
//!   script's height to 1080 lines), primary colour (ASS alpha is inverted: `00` is opaque),
//!   alignment, vertical margin, the outline width and colour, and the opaque box of
//!   `BorderStyle=3` (drawn in the outline colour) as the background box.
//! - Bold, italic and underline (of the line's style and the `\b` / `\i` / `\u` overrides) become
//!   `<b>` / `<i>` / `<u>` in the text; `\r` resets them; `\N` and `\n` are line breaks and `\h`
//!   a no-break space. The line's `Name` (actor) is the caption's speaker.
//! - Placement that differs from the track style becomes WebVTT cue settings, which burn-in and
//!   WebVTT export honour: `\an` / `\a` alignment overrides, another style's alignment or margin,
//!   or a line's own `MarginV` give `line:N%[,center|end] align:…`; `\pos(x,y)` (or the start of
//!   `\move`) gives `position:X% line:Y%,… align:…`.
//! - Other override tags (colours, fades, transforms, karaoke, clips…) and `{comments}` are
//!   removed; drawings (`\p1` … `\p0`) are dropped. An unterminated `{` is kept as text.
//! - Lines that share both times are merged into one caption (a caption track shows one caption
//!   at a time; other overlaps are cut on import, as for every format).
//!
//! Readers are forgiving and total: a BOM, CRLF, missing sections or Format lines, unknown or
//! malformed fields, hostile numbers and lines of any length never panic; a bad line is skipped
//! with a warning. Caption text is capped at [`MAX_TEXT_BYTES`].
//!
//! **Writing.** ASS v4.00+ at 1920 × 1080 with one `Default` style from the track style (so sizes
//! keep their 1080-line values), times in centiseconds, `<i>` / `<b>` / `<u>` as overrides, the
//! speaker as `Name`, and WebVTT `line` / `position` / `align` settings as `\an`, `MarginV` or
//! `\pos`. Text the format cannot hold changes: `{` `}` become `(` `)`, a backslash before `N`,
//! `n` or `h` becomes `＼`, commas in names become spaces.
//!
//! Written from the public SubStation Alpha v4.00+ script format description; no other
//! implementation's code was consulted.

use filmcraft_project::{CaptionAlign, CaptionAnchor, CaptionStyle};
use filmcraft_time::{TICKS_PER_SECOND, Tick};

use crate::{Cue, Document, Error, Result, parse_clock};

/// Longest caption text kept from one event, in bytes (the rest is dropped with a warning).
pub const MAX_TEXT_BYTES: usize = 16 * 1024;
/// Most warnings a document keeps (the rest are counted).
const MAX_WARNINGS: usize = 64;
/// Most styles kept (later ones are ignored).
const MAX_STYLES: usize = 10_000;
/// Longest font or speaker name kept, in characters.
const MAX_NAME_CHARS: usize = 256;
/// Largest play resolution taken from a file; larger (or zero, negative, NaN) is ignored.
const MAX_RES: f64 = 100_000.0;
/// The format's script resolution when a file gives none.
const DEFAULT_RES: (f64, f64) = (384.0, 288.0);
/// The writer's play resolution: 1080 lines, so the track style's sizes are written as they are.
const WRITE_RES: (f64, f64) = (1920.0, 1080.0);
/// Deepest parenthesis nesting followed inside an override tag's arguments.
const MAX_PAREN_DEPTH: usize = 64;
/// One centisecond in ticks (ASS times are `H:MM:SS.cc`).
const TICKS_PER_CS: i64 = TICKS_PER_SECOND / 100;

const STYLE_FIELDS_ASS: [&str; 23] = [
    "name",
    "fontname",
    "fontsize",
    "primarycolour",
    "secondarycolour",
    "outlinecolour",
    "backcolour",
    "bold",
    "italic",
    "underline",
    "strikeout",
    "scalex",
    "scaley",
    "spacing",
    "angle",
    "borderstyle",
    "outline",
    "shadow",
    "alignment",
    "marginl",
    "marginr",
    "marginv",
    "encoding",
];
const STYLE_FIELDS_SSA: [&str; 18] = [
    "name",
    "fontname",
    "fontsize",
    "primarycolour",
    "secondarycolour",
    "tertiarycolour",
    "backcolour",
    "bold",
    "italic",
    "borderstyle",
    "outline",
    "shadow",
    "alignment",
    "marginl",
    "marginr",
    "marginv",
    "alphalevel",
    "encoding",
];
const EVENT_FIELDS: [&str; 10] = ["layer", "start", "end", "style", "name", "marginl", "marginr", "marginv", "effect", "text"];

/// Horizontal placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum H {
    Left,
    Center,
    Right,
}

/// Vertical placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum V {
    Top,
    Middle,
    Bottom,
}

/// ASS numpad alignment (1–3 bottom, 4–6 middle, 7–9 top).
fn numpad(n: i64) -> Option<(H, V)> {
    let v = match n {
        1..=3 => V::Bottom,
        4..=6 => V::Middle,
        7..=9 => V::Top,
        _ => return None,
    };
    let h = match n % 3 {
        1 => H::Left,
        2 => H::Center,
        _ => H::Right,
    };
    Some((h, v))
}

/// SSA's legacy alignment (1–3 bottom, +4 top, +8 middle).
fn legacy(n: i64) -> Option<(H, V)> {
    let v = match n {
        1..=3 => V::Bottom,
        5..=7 => V::Top,
        9..=11 => V::Middle,
        _ => return None,
    };
    let h = match n % 4 {
        1 => H::Left,
        2 => H::Center,
        _ => H::Right,
    };
    Some((h, v))
}

fn to_numpad(h: H, v: V) -> u8 {
    let base = match v {
        V::Bottom => 1,
        V::Middle => 4,
        V::Top => 7,
    };
    base + match h {
        H::Left => 0,
        H::Center => 1,
        H::Right => 2,
    }
}

/// Bold / italic / underline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Fmt {
    b: bool,
    i: bool,
    u: bool,
}

/// One style line.
#[derive(Clone, Debug, PartialEq)]
struct Style {
    name: String,
    font: String,
    size: f64,
    primary: [u8; 4],
    outline_colour: [u8; 4],
    fmt: Fmt,
    border_style: i64,
    outline: f64,
    place: (H, V),
    margin_v: f64,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            name: "Default".into(),
            font: "Arial".into(),
            size: 20.0,
            primary: [255, 255, 255, 255],
            outline_colour: [0, 0, 0, 255],
            fmt: Fmt::default(),
            border_style: 1,
            outline: 2.0,
            place: (H::Center, V::Bottom),
            margin_v: 10.0,
        }
    }
}

/// A `Dialogue` line before its text is converted.
struct Event {
    start: Tick,
    end: Tick,
    style: String,
    name: String,
    margin_v: f64,
    text: String,
}

/// Warnings, capped.
struct Warnings {
    list: Vec<String>,
    more: usize,
}

impl Warnings {
    fn push(&mut self, msg: String) {
        if self.list.len() < MAX_WARNINGS {
            self.list.push(msg);
        } else {
            self.more = self.more.saturating_add(1);
        }
    }
    fn finish(mut self) -> Vec<String> {
        if self.more > 0 {
            self.list.push(format!("… and {} more", self.more));
        }
        self.list
    }
}

#[derive(Clone, Copy)]
enum Section {
    None,
    Info,
    Styles { legacy: bool },
    Events,
    Other,
}

fn field_names(value: &str) -> Vec<String> {
    value.split(',').map(|f| f.trim().to_ascii_lowercase()).collect()
}

/// A finite number.
fn num(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok().filter(|v| v.is_finite())
}

/// An ASS colour: `&HAABBGGRR` / `&HBBGGRR` (an optional trailing `&`), or SSA's decimal. ASS
/// alpha is transparency (`00` opaque); the result is sRGB with opacity.
fn colour(s: &str) -> Option<[u8; 4]> {
    let s = s.trim();
    let v: u32 = match s.strip_prefix("&H").or_else(|| s.strip_prefix("&h")) {
        Some(hex) => {
            let hex = hex.trim_end_matches('&');
            if hex.is_empty() || hex.len() > 8 {
                return None;
            }
            u32::from_str_radix(hex, 16).ok()?
        }
        None => {
            if s.is_empty() || s.len() > 20 {
                return None;
            }
            let n: i64 = s.parse().ok()?;
            u32::try_from(n & 0xffff_ffff).ok()?
        }
    };
    let [r, g, b, a] = v.to_le_bytes();
    Some([r, g, b, 255 - a])
}

fn ass_colour(c: [u8; 4]) -> String {
    format!("&H{:02X}{:02X}{:02X}{:02X}", 255 - c[3], c[2], c[1], c[0])
}

fn short(s: &str) -> String {
    s.trim().chars().take(MAX_NAME_CHARS).collect()
}

fn parse_style(value: &str, fields: &[String], legacy_align: bool) -> Option<Style> {
    let parts: Vec<&str> = value.splitn(fields.len().max(1), ',').collect();
    let get = |names: &[&str]| fields.iter().position(|f| names.contains(&f.as_str())).and_then(|i| parts.get(i)).map(|s| s.trim());
    let name = get(&["name"])?;
    let mut st = Style { name: name.trim_start_matches('*').to_string(), ..Default::default() };
    if let Some(f) = get(&["fontname"]).filter(|f| !f.is_empty()) {
        st.font = short(f);
    }
    if let Some(v) = get(&["fontsize"]).and_then(num).filter(|v| *v > 0.0) {
        st.size = v;
    }
    if let Some(c) = get(&["primarycolour", "primarycolor"]).and_then(colour) {
        st.primary = c;
    }
    if let Some(c) = get(&["outlinecolour", "outlinecolor", "tertiarycolour", "tertiarycolor"]).and_then(colour) {
        st.outline_colour = c;
    }
    let flag = |names: &[&str]| get(names).and_then(num).map(|v| v != 0.0);
    st.fmt = Fmt { b: flag(&["bold"]).unwrap_or(false), i: flag(&["italic"]).unwrap_or(false), u: flag(&["underline"]).unwrap_or(false) };
    if let Some(v) = get(&["borderstyle"]).and_then(num) {
        st.border_style = v as i64;
    }
    if let Some(v) = get(&["outline"]).and_then(num).filter(|v| *v >= 0.0) {
        st.outline = v;
    }
    if let Some(n) = get(&["alignment"]).and_then(num) {
        let n = n as i64;
        if let Some(p) = if legacy_align { legacy(n) } else { numpad(n) } {
            st.place = p;
        }
    }
    if let Some(v) = get(&["marginv"]).and_then(num).filter(|v| *v >= 0.0) {
        st.margin_v = v;
    }
    Some(st)
}

/// Parse an ASS or SSA script.
pub fn parse(text: &str) -> Result<Document> {
    let mut warn = Warnings { list: Vec::new(), more: 0 };
    let mut section = Section::None;
    let (mut res_x, mut res_y): (Option<f64>, Option<f64>) = (None, None);
    let mut style_fields: Option<Vec<String>> = None;
    let mut event_fields: Option<Vec<String>> = None;
    let mut styles: Vec<Style> = Vec::new();
    let mut events: Vec<Event> = Vec::new();
    let mut recognised = false;
    for (n, raw) in text.lines().enumerate() {
        let line_no = n + 1;
        let line = raw.trim_start_matches('\u{feff}').trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with("!:") {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            section = match name.trim().to_ascii_lowercase().as_str() {
                "script info" => Section::Info,
                "v4+ styles" | "v4 styles+" => Section::Styles { legacy: false },
                "v4 styles" => Section::Styles { legacy: true },
                "events" => Section::Events,
                _ => Section::Other,
            };
            recognised |= !matches!(section, Section::Other);
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            if !matches!(section, Section::Other | Section::None) {
                warn.push(format!("line {line_no}: ignored \"{}\"", line.chars().take(40).collect::<String>()));
            }
            continue;
        };
        let key = key.trim();
        let value = value.trim_start();
        match section {
            Section::Info => {
                if key.eq_ignore_ascii_case("PlayResX") {
                    res_x = num(value).filter(|v| *v >= 1.0 && *v <= MAX_RES);
                } else if key.eq_ignore_ascii_case("PlayResY") {
                    res_y = num(value).filter(|v| *v >= 1.0 && *v <= MAX_RES);
                }
            }
            Section::Styles { legacy } => {
                if key.eq_ignore_ascii_case("Format") {
                    style_fields = Some(field_names(value));
                } else if key.eq_ignore_ascii_case("Style") {
                    if styles.len() >= MAX_STYLES {
                        warn.push(format!("line {line_no}: more than {MAX_STYLES} styles; ignored"));
                        continue;
                    }
                    let defaults: Vec<String> = if legacy {
                        STYLE_FIELDS_SSA.iter().map(|s| s.to_string()).collect()
                    } else {
                        STYLE_FIELDS_ASS.iter().map(|s| s.to_string()).collect()
                    };
                    let fields = style_fields.as_deref().unwrap_or(&defaults);
                    // the section decides the alignment numbering (SSA's legacy in `[V4 Styles]`)
                    match parse_style(value, fields, legacy) {
                        Some(st) => styles.push(st),
                        None => warn.push(format!("line {line_no}: style without a name; ignored")),
                    }
                }
            }
            Section::Events | Section::None => {
                if key.eq_ignore_ascii_case("Format") && matches!(section, Section::Events) {
                    event_fields = Some(field_names(value));
                } else if key.eq_ignore_ascii_case("Dialogue") {
                    recognised = true;
                    let defaults: Vec<String> = EVENT_FIELDS.iter().map(|s| s.to_string()).collect();
                    let fields = event_fields.as_deref().unwrap_or(&defaults);
                    match parse_event(value, fields) {
                        Ok(e) => events.push(e),
                        Err(msg) => warn.push(format!("line {line_no}: {msg}; skipped")),
                    }
                }
            }
            Section::Other => {}
        }
    }
    if !recognised {
        return if text.trim_start_matches('\u{feff}').trim().is_empty() { Ok(Document::default()) } else { Err(Error::NotFormat("SubStation Alpha")) };
    }
    let (rx, ry) = match (res_x, res_y) {
        (Some(x), Some(y)) => (x, y),
        (Some(x), None) => (x, if x == 1280.0 { 1024.0 } else { x * 3.0 / 4.0 }),
        (None, Some(y)) => (if y == 1024.0 { 1280.0 } else { y * 4.0 / 3.0 }, y),
        (None, None) => DEFAULT_RES,
    };
    let find_style = |name: &str| -> Option<usize> {
        let name = name.trim().trim_start_matches('*');
        styles.iter().position(|s| s.name == name).or_else(|| styles.iter().position(|s| s.name.eq_ignore_ascii_case(name)))
    };
    let fallback = find_style("Default").or(if styles.is_empty() { None } else { Some(0) });
    // the track style: the one most lines use (the first such style on a tie)
    let mut uses = vec![0usize; styles.len()];
    let resolved: Vec<Option<usize>> = events.iter().map(|e| find_style(&e.style).or(fallback)).collect();
    for i in resolved.iter().flatten() {
        if let Some(u) = uses.get_mut(*i) {
            *u = u.saturating_add(1);
        }
    }
    let track_idx = uses.iter().enumerate().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(&a.0))).map(|(i, _)| i).filter(|_| !uses.is_empty()).or(fallback);
    let default_style = Style::default();
    let track = track_idx.and_then(|i| styles.get(i)).unwrap_or(&default_style);
    let mut doc = Document { style: (!styles.is_empty()).then(|| caption_style(track, ry)), ..Default::default() };
    let style_fmt = |name: &str| find_style(name).and_then(|i| styles.get(i)).map(|s| s.fmt);
    let mut empty = 0usize;
    for (e, si) in events.iter().zip(&resolved) {
        let st = si.and_then(|i| styles.get(i)).unwrap_or(&default_style);
        let conv = convert(&e.text, st.fmt, &style_fmt);
        if conv.truncated {
            warn.push(format!("caption at {}: text longer than {MAX_TEXT_BYTES} bytes; cut", crate::format_clock(e.start, '.')));
        }
        if !conv.visible {
            empty = empty.saturating_add(1);
            continue;
        }
        let place = conv.align.unwrap_or(st.place);
        let margin_v = if e.margin_v > 0.0 { e.margin_v } else { st.margin_v };
        let settings = if let Some((x, y)) = conv.pos {
            format!("position:{}% line:{}%,{} align:{}", pct(x / rx), pct(y / ry), line_align(place.1), text_align(place.0))
        } else if place != track.place || (place.1 != V::Middle && margin_v != track.margin_v) {
            let m = margin_v / ry;
            match place.1 {
                V::Top => format!("line:{}% align:{}", pct(m), text_align(place.0)),
                V::Middle => format!("line:50%,center align:{}", text_align(place.0)),
                V::Bottom => format!("line:{}%,end align:{}", pct(1.0 - m), text_align(place.0)),
            }
        } else {
            String::new()
        };
        let speaker = Some(short(&e.name)).filter(|n| !n.is_empty());
        doc.cues.push(Cue { start: e.start, end: e.end, text: conv.text, speaker, id: None, settings });
    }
    if empty > 0 {
        warn.push(format!("{empty} line(s) without text (drawings or empty) skipped"));
    }
    merge_simultaneous(&mut doc.cues);
    doc.warnings = warn.finish();
    Ok(doc)
}

/// Lines sharing both times become one caption, in file order.
fn merge_simultaneous(cues: &mut Vec<Cue>) {
    cues.sort_by_key(|c| (c.start, c.end));
    let mut out: Vec<Cue> = Vec::with_capacity(cues.len());
    for c in cues.drain(..) {
        match out.last_mut() {
            Some(prev) if prev.start == c.start && prev.end == c.end => {
                if prev.text.len().saturating_add(c.text.len()) < MAX_TEXT_BYTES {
                    prev.text.push('\n');
                    prev.text.push_str(&c.text);
                }
                if prev.speaker != c.speaker {
                    prev.speaker = None;
                }
            }
            _ => out.push(c),
        }
    }
    *cues = out;
}

fn parse_event(value: &str, fields: &[String]) -> std::result::Result<Event, String> {
    let parts: Vec<&str> = value.splitn(fields.len().max(1), ',').collect();
    let get = |name: &str| fields.iter().position(|f| f == name).and_then(|i| parts.get(i)).copied();
    let start = get("start").ok_or("no Start field")?;
    let end = get("end").ok_or("no End field")?;
    let text = get("text").ok_or("no Text field")?;
    let start = parse_clock(start).ok_or_else(|| format!("bad start time \"{}\"", start.trim().chars().take(20).collect::<String>()))?;
    let end = parse_clock(end).ok_or_else(|| format!("bad end time \"{}\"", end.trim().chars().take(20).collect::<String>()))?;
    if end <= start {
        return Err("end before start".into());
    }
    Ok(Event {
        start,
        end,
        style: get("style").unwrap_or("").trim().to_string(),
        name: get("name").or_else(|| get("actor")).unwrap_or("").to_string(),
        margin_v: get("marginv").and_then(num).filter(|v| *v > 0.0 && *v <= MAX_RES).unwrap_or(0.0),
        text: text.to_string(),
    })
}

/// A percentage for cue settings: 0–100, at most two decimals, no trailing zeros.
fn pct(f: f64) -> String {
    let v = if f.is_finite() { (f * 100.0).clamp(0.0, 100.0) } else { 0.0 };
    trim_num(format!("{v:.2}"))
}

fn trim_num(s: String) -> String {
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

fn line_align(v: V) -> &'static str {
    match v {
        V::Top => "start",
        V::Middle => "center",
        V::Bottom => "end",
    }
}

fn text_align(h: H) -> &'static str {
    match h {
        H::Left => "start",
        H::Center => "center",
        H::Right => "end",
    }
}

/// The track style for a script style (sizes scaled from `res_y` lines to 1080).
fn caption_style(st: &Style, res_y: f64) -> CaptionStyle {
    let scale = 1080.0 / res_y.max(1.0);
    let boxed = st.border_style == 3;
    let d = CaptionStyle::default();
    CaptionStyle {
        font: if st.font.trim().is_empty() { d.font.clone() } else { st.font.clone() },
        size: (st.size * scale).clamp(4.0, 400.0) as f32,
        color: st.primary,
        background: boxed,
        background_color: if boxed { st.outline_colour } else { d.background_color },
        align: match st.place.0 {
            H::Left => CaptionAlign::Left,
            H::Center => CaptionAlign::Center,
            H::Right => CaptionAlign::Right,
        },
        anchor: match st.place.1 {
            V::Top => CaptionAnchor::Top,
            V::Middle => CaptionAnchor::Middle,
            V::Bottom => CaptionAnchor::Bottom,
        },
        margin: (st.margin_v / res_y.max(1.0)).clamp(0.0, 0.45) as f32,
        outline: if boxed { 0.0 } else { (st.outline * scale).clamp(0.0, 40.0) as f32 },
        outline_color: st.outline_colour,
        ..d
    }
}

/// Converted event text.
#[derive(Debug, Default)]
struct Converted {
    text: String,
    /// The first `\an` / `\a`.
    align: Option<(H, V)>,
    /// The first `\pos` (or `\move` start), in script pixels.
    pos: Option<(f64, f64)>,
    /// Any character drawn (not only tags, drawings and spaces).
    visible: bool,
    truncated: bool,
}

/// Output text with `<b>` / `<i>` / `<u>` kept properly nested.
struct Out<'a> {
    c: &'a mut Converted,
    open: Vec<char>,
    want: Fmt,
    line_start: bool,
}

impl Out<'_> {
    fn full(&self) -> bool {
        self.c.text.len() >= MAX_TEXT_BYTES
    }
    /// Close the open tags that are no longer wanted; with `open`, also open the wanted ones
    /// (only right before a character, so no empty `<i></i>` pairs are written).
    fn sync_tags(&mut self, open: bool) {
        let mut desired = Vec::with_capacity(3);
        for (on, t) in [(self.want.b, 'b'), (self.want.i, 'i'), (self.want.u, 'u')] {
            if on {
                desired.push(t);
            }
        }
        if desired == self.open {
            return;
        }
        let keep = self.open.iter().zip(&desired).take_while(|(a, b)| a == b).count();
        while self.open.len() > keep {
            if let Some(t) = self.open.pop() {
                self.c.text.push_str(&format!("</{t}>"));
            }
        }
        if open {
            for t in desired.iter().skip(keep) {
                self.c.text.push_str(&format!("<{t}>"));
                self.open.push(*t);
            }
        }
    }
    fn push(&mut self, ch: char) {
        if self.full() {
            self.c.truncated = true;
            return;
        }
        if ch == '\n' {
            while self.c.text.ends_with([' ', '\t']) {
                self.c.text.pop();
            }
            // a tag switched off before the break closes on this line
            self.sync_tags(false);
            if self.c.visible && !self.line_start {
                self.c.text.push('\n');
            }
            self.line_start = true;
            return;
        }
        if self.line_start && (ch == ' ' || ch == '\t') {
            return;
        }
        if ch.is_control() {
            return;
        }
        self.sync_tags(true);
        self.c.text.push(ch);
        self.line_start = false;
        if !ch.is_whitespace() {
            self.c.visible = true;
        }
    }
    fn finish(mut self) {
        while self.c.text.ends_with(['\n', ' ', '\t']) {
            self.c.text.pop();
        }
        while let Some(t) = self.open.pop() {
            self.c.text.push_str(&format!("</{t}>"));
        }
    }
}

/// The index of the `)` closing the `(` at the start of `s`.
fn closing_paren(s: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (i, b) in s.bytes().enumerate() {
        match b {
            b'(' => {
                depth = depth.saturating_add(1);
                if depth > MAX_PAREN_DEPTH {
                    return None;
                }
            }
            b')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// `0` / `1` / a weight (bold: 1 or ≥ 600 is on), or nothing (the style's value).
fn toggle(arg: &str, base: bool, weight: bool) -> Option<bool> {
    if arg.is_empty() {
        return Some(base);
    }
    if !arg.bytes().all(|b| b.is_ascii_digit()) || arg.len() > 6 {
        return None;
    }
    let n: u32 = arg.parse().ok()?;
    Some(n == 1 || (weight && n >= 600))
}

fn numbers(args: &str) -> Vec<f64> {
    args.split(',').take(8).filter_map(num).collect()
}

/// Apply one `{…}` block's tags.
fn apply_block(block: &str, out: &mut Out, drawing: &mut bool, base: Fmt, style_fmt: &dyn Fn(&str) -> Option<Fmt>) {
    // text before the first backslash is a comment
    let Some(first) = block.find('\\') else { return };
    let mut rest = block.get(first..).unwrap_or("");
    while let Some(r) = rest.strip_prefix('\\') {
        let end = r.find(['\\', '(']).unwrap_or(r.len());
        let (name, after) = r.split_at(end);
        let (args, next) = match after.strip_prefix('(') {
            Some(_) => match closing_paren(after) {
                Some(close) => (after.get(1..close).unwrap_or(""), after.get(close + 1..).unwrap_or("")),
                None => (after.get(1..).unwrap_or(""), ""),
            },
            None => ("", after),
        };
        let name = name.trim();
        let digits = |prefix: &str| name.strip_prefix(prefix).filter(|d| d.bytes().all(|b| b.is_ascii_digit()));
        if let Some(d) = name.strip_prefix("an").filter(|d| !d.is_empty() && d.len() <= 3 && d.bytes().all(|b| b.is_ascii_digit())) {
            if out.c.align.is_none() {
                out.c.align = d.parse().ok().and_then(numpad);
            }
        } else if let Some(d) = digits("a").filter(|d| !d.is_empty() && d.len() <= 3) {
            if out.c.align.is_none() {
                out.c.align = d.parse().ok().and_then(legacy);
            }
        } else if name == "pos" || name == "move" {
            let v = numbers(args);
            let need = if name == "pos" { 2 } else { 4 };
            if out.c.pos.is_none()
                && v.len() >= need
                && let (Some(x), Some(y)) = (v.first(), v.get(1))
            {
                out.c.pos = Some((*x, *y));
            }
        } else if let Some(d) = digits("b") {
            if let Some(on) = toggle(d, base.b, true) {
                out.want.b = on;
            }
        } else if let Some(d) = digits("i") {
            if let Some(on) = toggle(d, base.i, false) {
                out.want.i = on;
            }
        } else if let Some(d) = digits("u") {
            if let Some(on) = toggle(d, base.u, false) {
                out.want.u = on;
            }
        } else if let Some(d) = digits("p") {
            *drawing = d.parse::<u32>().is_ok_and(|n| n > 0);
        } else if let Some(style) = name.strip_prefix('r') {
            out.want = if style.trim().is_empty() { base } else { style_fmt(style).unwrap_or(base) };
        }
        rest = next.find('\\').and_then(|p| next.get(p..)).unwrap_or("");
    }
}

/// Event text → caption text and placement overrides.
fn convert(raw: &str, base: Fmt, style_fmt: &dyn Fn(&str) -> Option<Fmt>) -> Converted {
    let mut conv = Converted::default();
    let last_close = raw.rfind('}');
    let mut drawing = false;
    {
        let mut out = Out { c: &mut conv, open: Vec::new(), want: base, line_start: true };
        let mut i = 0usize;
        while let Some(rest) = raw.get(i..).filter(|r| !r.is_empty()) {
            if out.full() {
                out.c.truncated = true;
                break;
            }
            if rest.starts_with('{')
                && last_close.is_some_and(|c| c > i)
                && let Some(close) = rest.find('}')
            {
                apply_block(rest.get(1..close).unwrap_or(""), &mut out, &mut drawing, base, style_fmt);
                i += close + 1;
                continue;
            }
            if let Some(esc) = rest.strip_prefix('\\')
                && let Some(c) = esc.chars().next()
                && matches!(c, 'N' | 'n' | 'h')
            {
                if !drawing {
                    out.push(if c == 'h' { '\u{a0}' } else { '\n' });
                }
                i += 2;
                continue;
            }
            let Some(c) = rest.chars().next() else { break };
            if !drawing {
                out.push(c);
            }
            i += c.len_utf8();
        }
        out.finish();
    }
    conv
}

/// Cue settings → (`\an`/`\pos` prefix, MarginV) for the writer.
fn placement(settings: &str, style: &CaptionStyle) -> (String, i64) {
    let mut line: Option<(f64, V)> = None;
    let mut position: Option<f64> = None;
    let mut align: Option<H> = None;
    for kv in settings.split_whitespace() {
        let Some((k, v)) = kv.split_once(':') else { continue };
        match k {
            "line" => {
                let mut parts = v.split(',');
                let Some(p) = parts.next().and_then(|p| p.strip_suffix('%')).and_then(num) else { continue };
                let la = match parts.next() {
                    Some("center" | "middle") => V::Middle,
                    Some("end") => V::Bottom,
                    _ => V::Top,
                };
                line = Some(((p / 100.0).clamp(0.0, 1.0), la));
            }
            "position" => {
                position = v.split(',').next().and_then(|p| p.strip_suffix('%')).and_then(num).map(|p| (p / 100.0).clamp(0.0, 1.0));
            }
            "align" => {
                align = match v {
                    "start" | "left" => Some(H::Left),
                    "end" | "right" => Some(H::Right),
                    "center" | "middle" => Some(H::Center),
                    _ => None,
                }
            }
            _ => {}
        }
    }
    let style_h = match style.align {
        CaptionAlign::Left => H::Left,
        CaptionAlign::Center => H::Center,
        CaptionAlign::Right => H::Right,
    };
    let style_v = match style.anchor {
        CaptionAnchor::Top => V::Top,
        CaptionAnchor::Middle => V::Middle,
        CaptionAnchor::Bottom => V::Bottom,
    };
    let margin = f64::from(style.margin.clamp(0.0, 0.45));
    let (w, h) = WRITE_RES;
    let pos = |x: f64, y: f64, hh: H, vv: V| (format!("{{\\an{}\\pos({},{})}}", to_numpad(hh, vv), coord(x), coord(y)), 0);
    if line.is_none() && position.is_none() && align.is_none() {
        return (String::new(), 0);
    }
    let hh = align.unwrap_or(style_h);
    let (f, vv) = line.unwrap_or(match style_v {
        V::Top => (margin, V::Top),
        V::Middle => (0.5, V::Middle),
        V::Bottom => (1.0 - margin, V::Bottom),
    });
    if let Some(x) = position {
        return pos(x * w, f * h, hh, vv);
    }
    let x = match hh {
        H::Left => 0.05 * w,
        H::Center => 0.5 * w,
        H::Right => 0.95 * w,
    };
    let an = |m: f64| {
        let px = (m * h).round();
        if px >= 1.0 { Some((format!("{{\\an{}}}", to_numpad(hh, vv)), px as i64)) } else { None }
    };
    let r = match vv {
        V::Top => an(f),
        V::Bottom => an(1.0 - f),
        V::Middle if (f - 0.5).abs() < 1e-9 => Some((format!("{{\\an{}}}", to_numpad(hh, vv)), 0)),
        V::Middle => None,
    };
    r.unwrap_or_else(|| pos(x, f * h, hh, vv))
}

fn coord(v: f64) -> String {
    trim_num(format!("{:.2}", if v.is_finite() { v } else { 0.0 }))
}

/// `H:MM:SS.cc`, rounded to the nearest centisecond.
fn clock(t: Tick) -> String {
    let cs = (i128::from(t.0.max(0)) + i128::from(TICKS_PER_CS) / 2) / i128::from(TICKS_PER_CS);
    format!("{}:{:02}:{:02}.{:02}", cs / 360_000, (cs / 6000) % 60, (cs / 100) % 60, cs % 100)
}

/// Caption text → ASS event text.
fn ass_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 16);
    let mut i = 0usize;
    while let Some(rest) = text.get(i..).filter(|r| !r.is_empty()) {
        let Some(c) = rest.chars().next() else { break };
        match c {
            '<' => {
                if let Some(close) = rest.find('>') {
                    let tag = rest.get(1..close).unwrap_or("").trim().to_ascii_lowercase();
                    let mapped = match tag.as_str() {
                        "i" => "{\\i1}",
                        "/i" => "{\\i0}",
                        "b" => "{\\b1}",
                        "/b" => "{\\b0}",
                        "u" => "{\\u1}",
                        "/u" => "{\\u0}",
                        _ => "",
                    };
                    out.push_str(mapped);
                    i += close + 1;
                    continue;
                }
                out.push('<');
            }
            '&' => {
                const ENTITIES: [(&str, &str); 7] =
                    [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&nbsp;", "\\h"), ("&quot;", "\""), ("&apos;", "'"), ("&#39;", "'")];
                if let Some((name, rep)) = ENTITIES.iter().find(|(n, _)| rest.starts_with(n)) {
                    out.push_str(rep);
                    i += name.len();
                    continue;
                }
                out.push('&');
            }
            '\n' => out.push_str("\\N"),
            '\r' => {}
            '{' => out.push('('),
            '}' => out.push(')'),
            '\\' if rest.get(1..2).is_some_and(|n| matches!(n, "N" | "n" | "h")) => out.push('＼'),
            c => out.push(c),
        }
        i += c.len_utf8();
    }
    out
}

/// Write an ASS v4.00+ script (one `Default` style from `doc.style`).
pub fn write(doc: &Document) -> String {
    let st = doc.style.clone().unwrap_or_default();
    let (w, h) = WRITE_RES;
    let mut out = format!(
        "[Script Info]\n; Written by FilmCraft\nScriptType: v4.00+\nPlayResX: {w}\nPlayResY: {h}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
    );
    let font: String = st.font.replace([',', '\n', '\r'], " ").trim().chars().take(MAX_NAME_CHARS).collect();
    let font = if font.is_empty() { "Inter".to_string() } else { font };
    let size = f64::from(if st.size.is_finite() { st.size.clamp(4.0, 400.0) } else { 54.0 });
    let (outline_colour, border, outline_w) = if st.background {
        (st.background_color, 3, (size * 0.15).round())
    } else {
        (st.outline_color, 1, f64::from(if st.outline.is_finite() { st.outline.clamp(0.0, 40.0) } else { 0.0 }))
    };
    let h_align = match st.align {
        CaptionAlign::Left => H::Left,
        CaptionAlign::Center => H::Center,
        CaptionAlign::Right => H::Right,
    };
    let v_align = match st.anchor {
        CaptionAnchor::Top => V::Top,
        CaptionAnchor::Middle => V::Middle,
        CaptionAnchor::Bottom => V::Bottom,
    };
    let margin_v = (f64::from(if st.margin.is_finite() { st.margin.clamp(0.0, 0.45) } else { 0.08 }) * h).round();
    out.push_str(&format!(
        "Style: Default,{font},{},{},&H000000FF,{},{},0,0,0,0,100,100,0,0,{border},{},0,{},{},{},{},1\n\n",
        coord(size),
        ass_colour(st.color),
        ass_colour(outline_colour),
        ass_colour(st.background_color),
        coord(outline_w),
        to_numpad(h_align, v_align),
        (0.05 * w).round(),
        (0.05 * w).round(),
        margin_v,
    ));
    out.push_str("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n");
    for c in doc.cues.iter().filter(|c| c.end > c.start) {
        let (prefix, mv) = placement(&c.settings, &st);
        let name: String = c.speaker.as_deref().unwrap_or("").replace([',', '\n', '\r'], " ").trim().chars().take(MAX_NAME_CHARS).collect();
        out.push_str(&format!("Dialogue: 0,{},{},Default,{name},0,0,{mv},,{prefix}{}\n", clock(c.start), clock(c.end), ass_text(&c.text)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TICKS_PER_MS;

    fn ms(v: i64) -> Tick {
        Tick(v * TICKS_PER_MS)
    }

    const SAMPLE: &str = "\u{feff}[Script Info]\r\n; Script generated by a subtitle editor\r\nTitle: Sample\r\nScriptType: v4.00+\r\nWrapStyle: 0\r\nPlayResX: 1280\r\nPlayResY: 720\r\nScaledBorderAndShadow: yes\r\n\r\n[Aegisub Project Garbage]\r\nVideo File: ../clip.mp4\r\n\r\n[V4+ Styles]\r\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\r\nStyle: Default,Noto Sans,48,&H0000FFFF,&H000000FF,&H00101010,&H80000000,0,0,0,0,100,100,0,0,1,2.5,1,2,20,20,40,1\r\nStyle: Sign,Arial,36,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,8,10,10,30,1\r\n\r\n[Events]\r\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\r\nComment: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,a note\r\nDialogue: 0,0:00:01.00,0:00:03.50,Default,Ann,0,0,0,,Hello, {\\i1}big{\\i0} world\\Nsecond line\r\nDialogue: 0,0:00:04.00,0:00:05.00,Sign,,0,0,0,,{\\fad(200,200)}TOP SIGN\r\nDialogue: 0,0:00:06.00,0:00:07.00,Default,,0,0,0,,{\\an8}Moved up\r\nDialogue: 0,0:00:08.00,0:00:09.00,Default,,0,0,0,,{\\pos(640,360)\\an5}Centre\r\nDialogue: 1,0:00:10.00,0:00:11.00,Default,,0,0,0,,{\\p1}m 0 0 l 100 0 100 100 0 100{\\p0}\r\nDialogue: 0,0:00:12.00,0:00:13.00,Default,,0,0,0,,{a comment}Plain {\\b1}bold{\\b0} and {\\u1}under\r\nDialogue: 0,0:00:14.00,0:00:15.00,Default,Bo,0,0,0,,- One\r\nDialogue: 0,0:00:14.00,0:00:15.00,Default,Cy,0,0,0,,- Two\r\nDialogue: 0,0:00:16.00,0:00:15.00,Default,,0,0,0,,inverted\r\n";

    #[test]
    fn real_world_ass() {
        let text = crate::decode_text(SAMPLE.as_bytes());
        let d = parse(&text).unwrap();
        let st = d.style.clone().unwrap();
        assert_eq!(st.font, "Noto Sans");
        assert!((st.size - 72.0).abs() < 1e-3, "48 at 720 lines is 72 at 1080: {}", st.size);
        assert_eq!(st.color, [255, 255, 0, 255], "&H0000FFFF is yellow");
        assert_eq!((st.align, st.anchor, st.background), (CaptionAlign::Center, CaptionAnchor::Bottom, false));
        assert!((st.margin - 40.0 / 720.0).abs() < 1e-6);
        assert!((st.outline - 3.75).abs() < 1e-3);
        assert_eq!(st.outline_color, [16, 16, 16, 255]);
        let got: Vec<(i64, &str, &str)> = d.cues.iter().map(|c| (c.start.0 / TICKS_PER_MS, c.text.as_str(), c.settings.as_str())).collect();
        assert_eq!(
            got,
            vec![
                (1000, "Hello, <i>big</i> world\nsecond line", ""),
                (4000, "<b>TOP SIGN</b>", "line:4.17% align:center"),
                (6000, "Moved up", "line:5.56% align:center"),
                (8000, "Centre", "position:50% line:50%,center align:center"),
                (12000, "Plain <b>bold</b> and <u>under</u>", ""),
                (14000, "- One\n- Two", ""),
            ]
        );
        assert_eq!(d.cues[0].end, ms(3500));
        assert_eq!(d.cues[0].speaker.as_deref(), Some("Ann"));
        assert_eq!(d.cues[5].speaker, None, "merged lines of two speakers");
        assert!(d.warnings.iter().any(|w| w.contains("end before start")), "{:?}", d.warnings);
        assert!(d.warnings.iter().any(|w| w.contains("without text")), "{:?}", d.warnings);
    }

    #[test]
    fn ssa_v4_with_legacy_alignment_and_decimal_colours() {
        let ssa = "[Script Info]\nScriptType: v4.00\nPlayResY: 480\n\n[V4 Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, TertiaryColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, AlphaLevel, Encoding\nStyle: Default,Tahoma,24,16777215,65535,0,0,0,-1,3,1,0,6,15,15,24,0,0\n\n[Events]\nFormat: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: Marked=0,0:00:00.50,0:00:02.00,Default,NTP,0000,0000,0000,,Top centre {\\a1}then\\nleft\n";
        let d = parse(ssa).unwrap();
        let st = d.style.clone().unwrap();
        assert_eq!((st.align, st.anchor), (CaptionAlign::Center, CaptionAnchor::Top), "legacy 6 is top centre");
        assert!(st.background, "BorderStyle 3 is the opaque box");
        assert_eq!(st.background_color, [0, 0, 0, 255]);
        assert_eq!(st.color, [255, 255, 255, 255]);
        assert!((st.size - 54.0).abs() < 1e-3);
        let c = &d.cues[0];
        assert_eq!((c.start, c.end), (ms(500), ms(2000)));
        assert_eq!(c.text, "<i>Top centre then\nleft</i>");
        assert_eq!(c.settings, "line:95%,end align:start", "\\a1 is bottom left, the style's 24 px margin");
        assert_eq!(c.speaker.as_deref(), Some("NTP"));
    }

    #[test]
    fn missing_sections_and_formats() {
        // no Script Info, no styles, no Format line
        let d = parse("[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Just text\n").unwrap();
        assert_eq!(d.cues.len(), 1);
        assert_eq!(d.style, None, "no styles: the track keeps its own");
        // headers only
        assert!(parse("[Script Info]\nTitle: x\n").unwrap().cues.is_empty());
        // a Format line without Text
        let d = parse("[Events]\nFormat: Start, End\nDialogue: 0:00:01.00,0:00:02.00\n").unwrap();
        assert!(d.cues.is_empty() && d.warnings.iter().any(|w| w.contains("no Text")), "{:?}", d.warnings);
        assert!(parse("hello world").is_err());
        assert!(parse("").unwrap().cues.is_empty());
        assert!(parse("\u{feff}").unwrap().cues.is_empty());
    }

    #[test]
    fn hostile_input_never_panics() {
        let huge = "9".repeat(400);
        let cases = [
            format!(
                "[Script Info]\nPlayResX: {huge}\nPlayResY: -5\n[V4+ Styles]\nStyle: Default,{huge},{huge},&H{huge},x,y,z,{huge},1,1,1,1,1,1,1,{huge},NaN,inf,{huge},1,1,{huge},1\n[Events]\nDialogue: {huge},{huge}:00:00.00,99999:99:99.99,Default,,{huge},0,{huge},,{{\\pos({huge},-{huge})}}x\n"
            ),
            "[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\pos(1e308,1e308)\\an99\\a77\\b99999999999\\p}x\n".into(),
            "[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{{{{\\i1unterminated\n".into(),
            "[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,}}}{\\t(((((((\\b1)))x\n".into(),
            "[Events]\nDialogue:\nDialogue: ,\nDialogue: 0,,,,,,,,,\nStyle: x\nFormat:\n".into(),
            "[V4+ Styles]\nFormat:\nStyle: \nStyle: ,,,,\n[Events]\nFormat: Text\nDialogue: x\n".into(),
            format!("[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{}\n", "{\\b1}".repeat(200_000)),
            format!("[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{}\n", "{".repeat(1 << 20)),
            format!("[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{}\n", "\\N".repeat(1 << 19)),
            format!("[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{{\\t({})}}x\n", "(".repeat(1 << 20)),
            format!("[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{}\n", "é".repeat(1 << 19)),
        ];
        for src in &cases {
            let r = std::panic::catch_unwind(|| parse(src));
            assert!(r.is_ok(), "panicked on {:?}", src.chars().take(80).collect::<String>());
            if let Ok(Ok(d)) = r {
                for c in &d.cues {
                    assert!(c.text.len() <= MAX_TEXT_BYTES + 32, "text {} bytes", c.text.len());
                }
                let _ = write(&d);
            }
        }
        // a 1 MB line of text is cut, with a warning
        let d = parse(&cases[10]).unwrap();
        assert!(d.warnings.iter().any(|w| w.contains("cut")), "{:?}", d.warnings);
        // an unterminated brace is text
        let d = parse("[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,a {b\n").unwrap();
        assert_eq!(d.cues[0].text, "a {b");
        // huge \pos values are clamped into the frame
        let d = parse(&cases[1]).unwrap();
        assert_eq!(d.cues[0].settings, "position:100% line:100%,end align:center");
    }

    #[test]
    fn many_warnings_are_capped() {
        let src = format!("[Events]\n{}", "Dialogue: 0,bad,bad,Default,,0,0,0,,x\n".repeat(10_000));
        let d = parse(&src).unwrap();
        assert!(d.warnings.len() <= MAX_WARNINGS + 1);
        assert!(d.warnings.last().is_some_and(|w| w.contains("more")));
    }

    #[test]
    fn colours() {
        assert_eq!(colour("&H00FF8000"), Some([0, 128, 255, 255]));
        assert_eq!(colour("&H80FFFFFF&"), Some([255, 255, 255, 127]));
        assert_eq!(colour("&HFF"), Some([255, 0, 0, 255]));
        assert_eq!(colour("255"), Some([255, 0, 0, 255]));
        assert_eq!(colour("-1"), Some([255, 255, 255, 0]));
        assert_eq!(colour("&H"), None);
        assert_eq!(colour("&H123456789"), None);
        assert_eq!(colour("zz"), None);
        assert_eq!(ass_colour([0, 128, 255, 255]), "&H00FF8000");
    }

    #[test]
    fn writes_and_reads_back() {
        let style = CaptionStyle { font: "Noto Serif".into(), size: 60.0, background: false, outline: 3.0, ..Default::default() };
        let doc = Document {
            cues: vec![
                Cue { start: ms(1000), end: ms(2500), text: "Hello <i>there</i>\nsecond {line}".into(), speaker: Some("Ann, B".into()), ..Default::default() },
                Cue { start: ms(3000), end: ms(4000), text: "Top".into(), settings: "line:10% align:start".into(), ..Default::default() },
                Cue { start: ms(5000), end: ms(6000), text: "At".into(), settings: "position:25% line:75%,end align:end".into(), ..Default::default() },
                Cue { start: ms(7000), end: ms(8000), text: "a &amp; b \\N".into(), ..Default::default() },
            ],
            style: Some(style.clone()),
            ..Default::default()
        };
        let w = write(&doc);
        assert!(w.contains("Style: Default,Noto Serif,60,&H00FFFFFF,&H000000FF,&H00000000,"), "{w}");
        assert!(w.contains("Dialogue: 0,0:00:01.00,0:00:02.50,Default,Ann  B,0,0,0,,Hello {\\i1}there{\\i0}\\Nsecond (line)\n"), "{w}");
        assert!(w.contains("Default,,0,0,108,,{\\an7}Top"), "{w}");
        assert!(w.contains("{\\an3\\pos(480,810)}At"), "{w}");
        assert!(w.contains("a & b ＼N"), "{w}");
        let back = parse(&w).unwrap();
        let bs = back.style.clone().unwrap();
        assert_eq!((bs.font.as_str(), bs.size, bs.background, bs.outline), ("Noto Serif", 60.0, false, 3.0));
        assert!((bs.margin - style.margin).abs() <= 0.5 / 1080.0, "MarginV is whole pixels: {}", bs.margin);
        let texts: Vec<(&str, &str)> = back.cues.iter().map(|c| (c.text.as_str(), c.settings.as_str())).collect();
        assert_eq!(
            texts,
            vec![("Hello <i>there</i>\nsecond (line)", ""), ("Top", "line:10% align:start"), ("At", "position:25% line:75%,end align:end"), ("a & b ＼N", ""),]
        );
        assert_eq!(back.cues[0].speaker.as_deref(), Some("Ann  B"));
        // the default (boxed) track style writes the opaque box
        let w = write(&Document::default());
        assert!(w.contains(",3,8,0,2,96,96,86,1\n"), "{w}");
        let back = parse(&w).unwrap().style.unwrap();
        assert!(back.background && back.background_color == CaptionStyle::default().background_color);
    }

    #[test]
    fn clock_rounds_to_centiseconds() {
        assert_eq!(clock(ms(3_723_456)), "1:02:03.46");
        assert!(clock(Tick(i64::MAX)).len() > 8);
        assert_eq!(clock(Tick(-5)), "0:00:00.00");
    }
}
