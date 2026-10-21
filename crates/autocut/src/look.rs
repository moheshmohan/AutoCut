//! The look of a showcase ([`crate::showcase::build`]), all of it optional, from the plan file
//! (top-level fields beside `entries` and `script`) and the CLI's flags; without any of it a
//! showcase looks as it always did.
//!
//! | plan field | CLI | what it does |
//! |---|---|---|
//! | `style`: `"tech"` or `{titleFont, bodyFont, numberFont, accent, text, panel, base}` | `--style tech` | a theme ([`Style`]) on the title card, lower thirds, chapter cards, spec chips, the timeline, the counter and the outro; a family that is not installed falls back to Inter ([`Style::resolve`]) |
//! | `timeline: true`, `events: [{name, from, to}]` | `--timeline` | a timeline along the bottom ([`plan_timeline`]): the years as ticks, event bands, a marker gliding to each entry's date as it starts, the current year and event picked out; hidden during the title and the outro |
//! | `script.chips: {"3": ["32.5 kg", "1.6 m/s"]}` | | up to three spec chips beside an entry's lower third, in one after the other while its narration plays ([`chips_by_entry`]) |
//! | entries with `"kind": "scene"` (and `caption`) | | scene shots: footage without a robot, full frame, with a caption instead of a name lower third, opening their chapter; no counter, no ducking without words |
//! | `transitions: {within, between, chapter}` (or `true`) | | transitions from the catalogue ([`TransitionPlan`]) with a sound each (`transitions.addSound`), snapped to the music's beats (`transitions.snapToBeats`) |
//! | `loudness: "social"` | `--loudness social` | `audio.normalizeLoudness` on the mix (−14 LUFS, −1 dBTP) |
//! | `mix: "speech"` (the default), `"gentle"` or `"cinematic"` | `--mix gentle`, `--mix cinematic` | how the stems are balanced and treated ([`crate::mix::MixProfile`]): speech first for small speakers, the same with a voice-preserving narration EQ, or the earlier full-range balance |
//! | `embedCaptions: true` | `--embed-captions` | the export carries the narration's captions (CEA-608/708) |
//! | `soundfont: "generaluser-gs"` | `--soundfont …` | the instruments of the generated music bed |
//! | `outro: {logo, channel, handle, tagline, cta, ctaDone, terminal, background, seconds}` | `--outro-logo …` | a branded channel outro with the end screen's regions left clear ([`crate::outro`]) |
//!
//! Every field of the plan is checked before anything is built: a field that makes no sense is
//! an error that says which and why, never a panic and never a half-built show.

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::{Value, json};

use crate::showcase::{on_frame, run, ticks, ymd};
use crate::{AutoCutError, Executor};

fn invalid(msg: impl Into<String>) -> AutoCutError {
    AutoCutError::Invalid(msg.into())
}

/// The fields of a plan this module reads.
pub const PLAN_KEYS: &[&str] = &["style", "timeline", "events", "transitions", "loudness", "mix", "embedCaptions", "soundfont", "outro"];

// ---------------------------------------------------------------------------------------------
// Colours and fonts

/// A colour written `#rrggbb` or `#rrggbbaa` (the `#` optional): its bytes, alpha 255 without one.
pub fn parse_hex(s: &str) -> Option<[u8; 4]> {
    let t = s.trim();
    let h = t.strip_prefix('#').unwrap_or(t);
    if !(h.len() == 6 || h.len() == 8) || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let b = |i: usize| h.get(i..i + 2).and_then(|x| u8::from_str_radix(x, 16).ok());
    Some([b(0)?, b(2)?, b(4)?, if h.len() == 8 { b(6)? } else { 255 }])
}

fn hex6(c: [u8; 4]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

/// `#rrggbb` with `alpha` (0..1) as `#rrggbbaa`.
fn with_alpha(rgb: &str, alpha: f64) -> String {
    let c = parse_hex(rgb).unwrap_or([0, 0, 0, 255]);
    let a = if alpha.is_finite() { (alpha.clamp(0.0, 1.0) * 255.0).round() as u8 } else { 255 };
    format!("{}{a:02x}", hex6(c))
}

/// The family that stands in for one that is not installed (bundled with the app).
pub const FALLBACK_FAMILY: &str = "Inter";
/// Longest font family name a style may give.
const MAX_FONT_CHARS: usize = 64;

/// A theme for the showcase's text and overlays, as the plan gives it ([`Style::from_plan`]).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Style {
    /// `tech`, `classic` or `custom`.
    pub name: String,
    pub title_font: String,
    pub body_font: String,
    pub number_font: String,
    /// `#rrggbb`.
    pub accent: String,
    pub text: String,
    pub panel: String,
    /// The panels' opacity, 0..1.
    pub panel_alpha: f64,
}

impl Style {
    /// DIN Condensed titles, DIN Alternate numbers, Futura text, an electric cyan accent and
    /// dark translucent panels.
    pub fn tech() -> Self {
        Self {
            name: "tech".into(),
            title_font: "DIN Condensed".into(),
            body_font: "Futura".into(),
            number_font: "DIN Alternate".into(),
            accent: "#00e5ff".into(),
            text: "#f4f8fb".into(),
            panel: "#0a1018".into(),
            panel_alpha: 0.72,
        }
    }

    /// Today's look in theme form (Inter, white, yellow, black panels): what the new overlays —
    /// timeline, chips, scene captions — wear when no theme is asked for.
    pub fn classic() -> Self {
        Self {
            name: "classic".into(),
            title_font: FALLBACK_FAMILY.into(),
            body_font: FALLBACK_FAMILY.into(),
            number_font: FALLBACK_FAMILY.into(),
            accent: "#ffcc00".into(),
            text: "#ffffff".into(),
            panel: "#000000".into(),
            panel_alpha: 0.55,
        }
    }

    /// A built-in theme by name: `tech`; `classic` (or `none`) is today's look, no theme.
    fn named(name: &str) -> Result<Option<Self>, AutoCutError> {
        match name.trim().to_ascii_lowercase().as_str() {
            "tech" => Ok(Some(Self::tech())),
            "classic" | "none" | "default" | "" => Ok(None),
            other => Err(invalid(format!("style: unknown theme `{}` (tech, classic)", clip_chars(other, 40)))),
        }
    }

    /// The plan's `style`: a theme name, or an object of `titleFont`, `bodyFont`, `numberFont`,
    /// `accent`, `text` (`#rrggbb`), `panel` (`#rrggbb`, `#rrggbbaa` or `{color, alpha}`) over a
    /// `base` theme (`tech` when not given). `None`: today's look.
    pub fn from_plan(v: &Value) -> Result<Option<Self>, AutoCutError> {
        let o = match v {
            Value::Null => return Ok(None),
            Value::String(s) => return Self::named(s),
            Value::Object(o) => o,
            other => return Err(invalid(format!("style: a theme name (\"tech\") or an object of fonts and colours, not {}", kind(other)))),
        };
        let mut st = match o.get("base") {
            None | Some(Value::Null) => Self::tech(),
            Some(Value::String(b)) => Self::named(b)?.unwrap_or_else(Self::classic),
            Some(other) => return Err(invalid(format!("style.base: a theme name, not {}", kind(other)))),
        };
        for (k, v) in o {
            match k.as_str() {
                "base" => {}
                "titleFont" => st.title_font = font(k, v)?,
                "bodyFont" => st.body_font = font(k, v)?,
                "numberFont" => st.number_font = font(k, v)?,
                "accent" => st.accent = colour(k, v)?,
                "text" => st.text = colour(k, v)?,
                "panel" => {
                    let (c, a) = panel(v)?;
                    st.panel = c;
                    if let Some(a) = a {
                        st.panel_alpha = a;
                    }
                }
                other => {
                    return Err(invalid(format!(
                        "style: unknown field `{}` (titleFont, bodyFont, numberFont, accent, text, panel, base)",
                        clip_chars(other, 40)
                    )));
                }
            }
        }
        if o.keys().any(|k| k != "base") {
            st.name = "custom".into();
        }
        Ok(Some(st))
    }

    /// The theme as drawn, its families resolved against those `installed` (`fonts.list`; `None`
    /// when the engine cannot say, and the names are used as given): a family that is not there
    /// is drawn in Inter, and listed in [`Theme::missing`].
    pub fn resolve(&self, installed: Option<&[(String, Vec<String>)]>) -> Theme {
        let mut missing: Vec<String> = Vec::new();
        let mut pick = |family: &str, role: Role| -> Font {
            let Some(list) = installed else { return Font { family: family.trim().to_string(), style: role.style().into() } };
            match list.iter().find(|(f, _)| f.eq_ignore_ascii_case(family.trim())) {
                Some((f, styles)) => Font { family: f.clone(), style: pick_style(styles, role) },
                None => {
                    if !missing.iter().any(|m| m.eq_ignore_ascii_case(family.trim())) {
                        missing.push(family.trim().to_string());
                    }
                    Font { family: FALLBACK_FAMILY.into(), style: role.style().into() }
                }
            }
        };
        let title = pick(&self.title_font, Role::Title);
        let body = pick(&self.body_font, Role::Body);
        let number = pick(&self.number_font, Role::Number);
        Theme {
            name: self.name.clone(),
            title,
            body,
            number,
            accent: self.accent.clone(),
            text: self.text.clone(),
            panel: self.panel.clone(),
            panel_alpha: self.panel_alpha,
            missing,
        }
    }
}

/// The installed style closest to what `role` wants: its own name, else one of the same
/// weight, else the first upright one.
fn pick_style(styles: &[String], role: Role) -> String {
    let want = role.style();
    let italic = |s: &str| {
        let l = s.to_ascii_lowercase();
        l.contains("italic") || l.contains("oblique")
    };
    if let Some(s) = styles.iter().find(|s| s.eq_ignore_ascii_case(want)) {
        return s.clone();
    }
    let weight = want.to_ascii_lowercase();
    styles
        .iter()
        .find(|s| s.to_ascii_lowercase().contains(&weight) && !italic(s))
        .or_else(|| styles.iter().find(|s| !italic(s)))
        .cloned()
        .unwrap_or_else(|| want.to_string())
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a true/false",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "an object",
    }
}

/// The first `n` characters of `s` (for messages that quote the plan).
fn clip_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn font(k: &str, v: &Value) -> Result<String, AutoCutError> {
    let s = v.as_str().map(str::trim).ok_or_else(|| invalid(format!("style.{k}: a font family name, not {}", kind(v))))?;
    if s.is_empty() || s.chars().count() > MAX_FONT_CHARS || s.chars().any(char::is_control) {
        return Err(invalid(format!("style.{k}: a font family name of 1 to {MAX_FONT_CHARS} characters")));
    }
    Ok(s.to_string())
}

fn colour(k: &str, v: &Value) -> Result<String, AutoCutError> {
    v.as_str()
        .and_then(|s| parse_hex(s).filter(|c| c[3] == 255 || s.trim().trim_start_matches('#').len() == 6))
        .map(hex6)
        .ok_or_else(|| invalid(format!("style.{k}: a colour as #rrggbb, not {}", short(v))))
}

/// `panel`: `#rrggbb`, `#rrggbbaa` or `{"color": "#rrggbb", "alpha": 0..1}`.
fn panel(v: &Value) -> Result<(String, Option<f64>), AutoCutError> {
    let bad = || invalid(format!("style.panel: #rrggbb, #rrggbbaa or {{\"color\": \"#rrggbb\", \"alpha\": 0..1}}, not {}", short(v)));
    match v {
        Value::String(s) => {
            let c = parse_hex(s).ok_or_else(bad)?;
            let alpha = (s.trim().trim_start_matches('#').len() == 8).then(|| f64::from(c[3]) / 255.0);
            Ok((hex6(c), alpha))
        }
        Value::Object(o) => {
            if let Some(k) = o.keys().find(|k| *k != "color" && *k != "alpha") {
                return Err(invalid(format!("style.panel: unknown field `{}` (color, alpha)", clip_chars(k, 40))));
            }
            let c = o.get("color").and_then(Value::as_str).and_then(parse_hex).ok_or_else(bad)?;
            let alpha = match o.get("alpha") {
                None | Some(Value::Null) => None,
                Some(a) => Some(a.as_f64().filter(|a| a.is_finite() && (0.0..=1.0).contains(a)).ok_or_else(bad)?),
            };
            Ok((hex6(c), alpha))
        }
        _ => Err(bad()),
    }
}

/// A plan value quoted in a message, short.
fn short(v: &Value) -> String {
    clip_chars(&v.to_string(), 60)
}

/// What a piece of text is: a title (names, cards), body text (places, captions) or a number
/// (years, counts, specs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Title,
    Body,
    Number,
}

impl Role {
    /// The style each role is set in.
    pub fn style(self) -> &'static str {
        match self {
            Role::Title | Role::Number => "Bold",
            Role::Body => "Medium",
        }
    }
}

/// A family and a style of it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Font {
    pub family: String,
    pub style: String,
}

/// A [`Style`] as drawn: its families resolved to installed ones.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Theme {
    pub name: String,
    pub title: Font,
    pub body: Font,
    pub number: Font,
    pub accent: String,
    pub text: String,
    pub panel: String,
    pub panel_alpha: f64,
    /// Families the style asked for that are not installed (Inter stands in).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub missing: Vec<String>,
}

impl Theme {
    pub fn font(&self, role: Role) -> &Font {
        match role {
            Role::Title => &self.title,
            Role::Body => &self.body,
            Role::Number => &self.number,
        }
    }

    /// `graphics.set` props for text in `role`, in `color`, aligned `align`: the theme's font, a
    /// soft shadow and no outline.
    pub fn text_props(&self, role: Role, color: &str, align: &str) -> Value {
        let f = self.font(role);
        json!({
            "font": f.family, "font_style": f.style, "fill_color": color, "align": align,
            "stroke": false, "shadow": true, "shadow_opacity": 60, "shadow_distance": 2, "shadow_blur": 16, "shadow_size": 0,
        })
    }

    /// The panels' colour with their opacity, `#rrggbbaa`.
    pub fn panel_rgba(&self) -> String {
        with_alpha(&self.panel, self.panel_alpha)
    }
}

/// The installed font families and their styles (`fonts.list`, which also makes the engine
/// look in the system's font folders); `None` when the engine cannot say.
pub async fn installed_fonts(ex: &dyn Executor) -> Option<Vec<(String, Vec<String>)>> {
    let r = run(ex, "fonts.list", json!({"system": true})).await.ok()?;
    let list = r.as_array()?;
    Some(
        list.iter()
            .take(20_000)
            .filter_map(|f| {
                let family = f["family"].as_str()?.to_string();
                let styles = f["styles"].as_array().map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).take(64).collect()).unwrap_or_default();
                Some((family, styles))
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------------------------
// Dates and events

/// Years a plan date may name.
const YEARS: std::ops::RangeInclusive<i64> = 1900..=2200;

fn leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        2 if leap(y) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Unix seconds of the start of `y-m-d` (UTC), for a date that exists in [`YEARS`].
pub fn unix_of(y: i64, m: u32, d: u32) -> Option<i64> {
    if !YEARS.contains(&y) || !(1..=12).contains(&m) || d == 0 || d > days_in_month(y, m) {
        return None;
    }
    let yy = if m <= 2 { y - 1 } else { y };
    let era = yy.div_euclid(400);
    let yoe = yy - era * 400;
    let mp = i64::from((m + 9) % 12);
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some((era * 146_097 + doe - 719_468) * 86_400)
}

/// The first and last second (UTC) a plan date covers: `YYYY`, `YYYY-MM` or `YYYY-MM-DD` (a
/// whole number is a year).
pub fn date_span(v: &Value) -> Option<(i64, i64)> {
    if let Some(y) = v.as_i64() {
        return Some((unix_of(y, 1, 1)?, unix_of(y.checked_add(1)?, 1, 1).map_or(unix_of(y, 12, 31)? + 86_399, |n| n - 1)));
    }
    let s = v.as_str()?.trim();
    let parts: Vec<&str> = s.split('-').collect();
    let num = |i: usize, len: usize| parts.get(i).filter(|p| p.len() == len && p.bytes().all(|b| b.is_ascii_digit())).and_then(|p| p.parse::<i64>().ok());
    let y = num(0, 4)?;
    match parts.len() {
        1 => date_span(&json!(y)),
        2 => {
            let m = u32::try_from(num(1, 2)?).ok()?;
            let first = unix_of(y, m, 1)?;
            Some((first, unix_of(y, m, days_in_month(y, m))? + 86_399))
        }
        3 => {
            let (m, d) = (u32::try_from(num(1, 2)?).ok()?, u32::try_from(num(2, 2)?).ok()?);
            let first = unix_of(y, m, d)?;
            Some((first, first + 86_399))
        }
        _ => None,
    }
}

/// An event of the plan's `events`: a named stretch of time ("GITEX 2022", 10–14 October 2022).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Event {
    pub name: String,
    /// Unix seconds: its first second, and its last.
    pub from: i64,
    pub to: i64,
}

/// Most events a plan may name, and the longest name.
const MAX_EVENTS: usize = 64;
const MAX_EVENT_CHARS: usize = 60;

/// The plan's `events`: `[{"name": "GITEX 2022", "from": "2022-10-10", "to": "2022-10-14"}]`.
pub fn events_from_plan(v: &Value) -> Result<Vec<Event>, AutoCutError> {
    let list = match v {
        Value::Null => return Ok(Vec::new()),
        Value::Array(a) => a,
        other => return Err(invalid(format!("events: a list of {{name, from, to}}, not {}", kind(other)))),
    };
    if list.len() > MAX_EVENTS {
        return Err(invalid(format!("events: at most {MAX_EVENTS}, not {}", list.len())));
    }
    let mut out = Vec::with_capacity(list.len());
    for (i, e) in list.iter().enumerate() {
        let at = format!("events[{i}]");
        let o = e.as_object().ok_or_else(|| invalid(format!("{at}: an object {{name, from, to}}, not {}", kind(e))))?;
        if let Some(k) = o.keys().find(|k| !["name", "from", "to"].contains(&k.as_str())) {
            return Err(invalid(format!("{at}: unknown field `{}` (name, from, to)", clip_chars(k, 40))));
        }
        let name = o.get("name").and_then(Value::as_str).map(str::trim).unwrap_or_default();
        if name.is_empty() || name.chars().count() > MAX_EVENT_CHARS || name.chars().any(char::is_control) {
            return Err(invalid(format!("{at}.name: a name of 1 to {MAX_EVENT_CHARS} characters")));
        }
        let date = |k: &str| {
            let v = o.get(k).unwrap_or(&Value::Null);
            date_span(v).ok_or_else(|| invalid(format!("{at}.{k}: a date as YYYY-MM-DD (or YYYY-MM, YYYY) from 1900 to 2200, not {}", short(v))))
        };
        let (from, _) = date("from")?;
        let (_, to) = date("to")?;
        if to < from {
            return Err(invalid(format!("{at} (\"{name}\"): `to` is before `from`")));
        }
        out.push(Event { name: name.to_string(), from, to });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Transitions

/// Between entries, when the plan names none: a curated mix of Zooms & Camera, Flow & Morph and
/// GL transitions.
pub const BETWEEN: &[&str] = &["cam_punch_in", "flow_morph", "gl_crosswarp", "cam_whip_pan", "gl_directionalwarp", "flow_doppler_warp"];
/// Into a new chapter: Light & Film's light leak and film burn.
pub const CHAPTER: &[&str] = &["lf_light_leak", "gl_filmburn"];
/// Most transitions one list may name.
const MAX_TRANSITIONS: usize = 64;

/// Transitions from the catalogue (`effects.list`): `within` an entry (between its shots;
/// none — straight cuts on the beat — by default), `between` entries and into a new `chapter`,
/// each list used in turn.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TransitionPlan {
    pub within: Vec<String>,
    pub between: Vec<String>,
    pub chapter: Vec<String>,
}

impl TransitionPlan {
    /// The defaults: cuts within entries, [`BETWEEN`] between them, [`CHAPTER`] into chapters.
    pub fn catalog() -> Self {
        let own = |l: &[&str]| l.iter().map(|s| s.to_string()).collect();
        Self { within: Vec::new(), between: own(BETWEEN), chapter: own(CHAPTER) }
    }

    /// The plan's `transitions`: `true` (the defaults) or `{within, between, chapter}`, each a
    /// list of transition ids (a list left out keeps its default; an empty one means cuts).
    pub fn from_plan(v: &Value) -> Result<Option<Self>, AutoCutError> {
        let o = match v {
            Value::Null | Value::Bool(false) => return Ok(None),
            Value::Bool(true) => return Ok(Some(Self::catalog())),
            Value::Object(o) => o,
            other => return Err(invalid(format!("transitions: true or {{within, between, chapter}}, not {}", kind(other)))),
        };
        let mut plan = Self::catalog();
        for (k, v) in o {
            let list = transition_ids(k, v)?;
            match k.as_str() {
                "within" => plan.within = list,
                "between" => plan.between = list,
                "chapter" => plan.chapter = list,
                other => return Err(invalid(format!("transitions: unknown field `{}` (within, between, chapter)", clip_chars(other, 40)))),
            }
        }
        Ok(Some(plan))
    }
}

fn transition_ids(k: &str, v: &Value) -> Result<Vec<String>, AutoCutError> {
    let items: Vec<&Value> = match v {
        Value::String(_) => vec![v],
        Value::Array(a) if a.len() <= MAX_TRANSITIONS => a.iter().collect(),
        Value::Array(a) => return Err(invalid(format!("transitions.{k}: at most {MAX_TRANSITIONS} ids, not {}", a.len()))),
        other => return Err(invalid(format!("transitions.{k}: a list of transition ids, not {}", kind(other)))),
    };
    items
        .into_iter()
        .map(|x| {
            let id = x.as_str().ok_or_else(|| invalid(format!("transitions.{k}: transition ids are strings, not {}", short(x))))?;
            filmcraft_engine::project::vtransition::find_transition(id, filmcraft_engine::project::EffectKind::VideoTransition)
                .map(|d| d.id.to_string())
                .ok_or_else(|| invalid(format!("transitions.{k}: unknown transition `{}` (see `effects.list` with kind VideoTransition)", clip_chars(id, 60))))
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Finishing

/// Where the mix is normalised to (`audio.normalizeLoudness`).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Loudness {
    /// `social` (−14 LUFS), `voice` (−16) or `broadcast` (−23).
    Named(String),
    /// Integrated loudness, LUFS (−70…0).
    Lufs(f64),
}

/// The ceiling of a normalised mix, dBTP.
pub const TRUE_PEAK: f64 = -1.0;

impl Loudness {
    /// `audio.normalizeLoudness`'s `target`.
    pub fn target(&self) -> Value {
        match self {
            Loudness::Named(n) => json!(n),
            Loudness::Lufs(l) => json!(l),
        }
    }

    /// A plan's or the CLI's `loudness`: `social`, `voice`, `broadcast`, a number of LUFS from
    /// −70 to 0 (`"-14"` too), or `off`.
    pub fn from_plan(v: &Value) -> Result<Option<Self>, AutoCutError> {
        let lufs = |x: f64| (x.is_finite() && (-70.0..=0.0).contains(&x)).then_some(Loudness::Lufs(x));
        let bad = || invalid(format!("loudness: social, voice, broadcast or LUFS from -70 to 0, not {}", short(v)));
        match v {
            Value::Null => Ok(None),
            Value::Number(n) => n.as_f64().and_then(lufs).map(Some).ok_or_else(bad),
            Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
                "off" | "none" | "" => Ok(None),
                n @ ("social" | "voice" | "broadcast") => Ok(Some(Loudness::Named(n.to_string()))),
                other => other.parse::<f64>().ok().and_then(lufs).map(Some).ok_or_else(bad),
            },
            _ => Err(bad()),
        }
    }
}

/// Longest SoundFont choice (a path).
const MAX_SOUNDFONT_CHARS: usize = 4096;

/// A plan's or the CLI's `soundfont`: `auto`, `oscillators`, `generaluser-gs` or the path of an
/// `.sf2` file (the engine checks which).
pub fn soundfont_from(v: &Value) -> Result<Option<String>, AutoCutError> {
    match v {
        Value::Null => Ok(None),
        Value::String(s) if !s.trim().is_empty() && s.chars().count() <= MAX_SOUNDFONT_CHARS && !s.contains('\0') => Ok(Some(s.trim().to_string())),
        other => Err(invalid(format!("soundfont: auto, oscillators, generaluser-gs or the path of an .sf2 file, not {}", short(other)))),
    }
}

// ---------------------------------------------------------------------------------------------
// The look as a whole

/// Everything a plan or the CLI asks of the showcase's look (all off by default).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Look {
    /// A theme; `None`: today's look.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<Style>,
    pub timeline: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<Event>,
    /// Transitions from the catalogue; `None`: today's rotating set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transitions: Option<TransitionPlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loudness: Option<Loudness>,
    /// How the stems are mixed (speech first unless the plan or the CLI says `gentle` or `cinematic`).
    pub mix: crate::mix::MixProfile,
    pub embed_captions: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub soundfont: Option<String>,
    /// A branded channel outro instead of the plain card ([`crate::outro`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outro: Option<crate::outro::Outro>,
}

impl Look {
    /// The look a plan asks for, from its top-level fields ([`PLAN_KEYS`]); a plan without
    /// them (or that is not an object) asks for nothing.
    pub fn from_plan(plan: &Value) -> Result<Self, AutoCutError> {
        let Some(o) = plan.as_object() else { return Ok(Self::default()) };
        let null = Value::Null;
        let get = |k: &str| o.get(k).unwrap_or(&null);
        let flag = |k: &str| match get(k) {
            Value::Null => Ok(false),
            Value::Bool(b) => Ok(*b),
            other => Err(invalid(format!("{k}: true or false, not {}", short(other)))),
        };
        Ok(Self {
            style: Style::from_plan(get("style"))?,
            timeline: flag("timeline")?,
            events: events_from_plan(get("events"))?,
            transitions: TransitionPlan::from_plan(get("transitions"))?,
            loudness: Loudness::from_plan(get("loudness"))?,
            mix: crate::mix::MixProfile::from_plan(get("mix"))?,
            embed_captions: flag("embedCaptions")?,
            soundfont: soundfont_from(get("soundfont"))?,
            outro: crate::outro::Outro::from_plan(get("outro"))?,
        })
    }

    /// The fields of `plan` that are about the look, as they are (to write a plan out again).
    pub fn plan_fields(plan: &Value) -> serde_json::Map<String, Value> {
        // (a script file's `outro` is its spoken line, not the look's)
        let ours = |k: &str, v: &Value| PLAN_KEYS.contains(&k) && !(k == "outro" && v.is_string());
        plan.as_object().map(|o| o.iter().filter(|(k, v)| ours(k, v)).map(|(k, v)| (k.clone(), v.clone())).collect()).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------------------------
// Spec chips

/// Chips one entry shows at most, and the longest chip.
pub const MAX_CHIPS: usize = 3;
pub const MAX_CHIP_CHARS: usize = 32;

/// A script's `chips` (by entry number, as written) checked against the entries' `numbers`:
/// each key an entry's number, each entry at most [`MAX_CHIPS`] chips of 1 to
/// [`MAX_CHIP_CHARS`] characters on one line. An entry with an empty list shows none.
pub fn chips_by_entry(chips: &BTreeMap<String, Vec<String>>, numbers: &[usize]) -> Result<BTreeMap<usize, Vec<String>>, AutoCutError> {
    let mut out = BTreeMap::new();
    for (k, list) in chips {
        let n: usize = k.trim().parse().map_err(|_| invalid(format!("script.chips: `{}` is not an entry number", clip_chars(k, 20))))?;
        if !numbers.contains(&n) {
            return Err(invalid(format!("script.chips: there is no entry {n}")));
        }
        if list.len() > MAX_CHIPS {
            return Err(invalid(format!("script.chips.{n}: at most {MAX_CHIPS} chips, not {}", list.len())));
        }
        let mut kept = Vec::with_capacity(list.len());
        for c in list {
            let c = c.trim();
            if c.is_empty() || c.chars().count() > MAX_CHIP_CHARS || c.chars().any(char::is_control) {
                return Err(invalid(format!("script.chips.{n}: a chip is 1 to {MAX_CHIP_CHARS} characters on one line, not `{}`", clip_chars(c, 40))));
            }
            kept.push(c.to_string());
        }
        if !kept.is_empty() {
            out.insert(n, kept);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Drawing: panels of text, chips

/// The size of the frame's elements relative to 1080 lines.
pub(crate) fn unit(frame: (f64, f64)) -> f64 {
    let h = frame.1;
    if h.is_finite() && h > 0.0 { h / 1080.0 } else { 1.0 }
}

/// A box on the frame, sequence pixels.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Bounds {
    pub left: f64,
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
}

impl Bounds {
    pub(crate) fn union(self, o: Bounds) -> Bounds {
        Bounds { left: self.left.min(o.left), top: self.top.min(o.top), right: self.right.max(o.right), bottom: self.bottom.max(o.bottom) }
    }

    pub(crate) fn pad(self, x: f64, y: f64) -> Bounds {
        Bounds { left: self.left - x, top: self.top - y, right: self.right + x, bottom: self.bottom + y }
    }

    pub(crate) fn centre(&self) -> [f64; 2] {
        [(self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0]
    }

    pub(crate) fn size(&self) -> [f64; 2] {
        [(self.right - self.left).max(1.0), (self.bottom - self.top).max(1.0)]
    }
}

/// About where point text of `size` at `pos` (its alignment point on the baseline) lands, when
/// the engine cannot measure it.
pub(crate) fn estimate(text: &str, size: f64, pos: [f64; 2], align: &str) -> Bounds {
    let longest = text.lines().map(|l| l.chars().count()).max().unwrap_or(0) as f64;
    let lines = text.lines().count().max(1) as f64;
    let w = longest * size * 0.56;
    let left = match align {
        "center" => pos[0] - w / 2.0,
        "right" => pos[0] - w,
        _ => pos[0],
    };
    Bounds { left, top: pos[1] - size * 0.74, right: left + w, bottom: pos[1] + size * 0.22 + (lines - 1.0) * size * 1.2 }
}

/// The bounds of the layers of graphic clip `clip` (`graphics.list`), all of them or `layer`.
pub(crate) async fn measure(ex: &dyn Executor, clip: u64, layer: Option<usize>) -> Option<Bounds> {
    let r = run(ex, "graphics.list", json!({"clip": clip})).await.ok()?;
    let mut out: Option<Bounds> = None;
    for (i, l) in r["layers"].as_array()?.iter().enumerate() {
        if layer.is_some_and(|k| k != i) {
            continue;
        }
        let pts: Vec<[f64; 2]> = l["quad"].as_array()?.iter().filter_map(|p| Some([p.get(0)?.as_f64()?, p.get(1)?.as_f64()?])).collect();
        let xs = pts.iter().map(|p| p[0]);
        let ys = pts.iter().map(|p| p[1]);
        let b = Bounds {
            left: xs.clone().fold(f64::MAX, f64::min),
            right: xs.fold(f64::MIN, f64::max),
            top: ys.clone().fold(f64::MAX, f64::min),
            bottom: ys.fold(f64::MIN, f64::max),
        };
        if !(b.left.is_finite() && b.right.is_finite() && b.top.is_finite() && b.bottom.is_finite()) || b.right < b.left || b.bottom < b.top {
            continue;
        }
        out = Some(out.map_or(b, |o| o.union(b)));
    }
    out
}

pub(crate) async fn set(ex: &dyn Executor, clip: u64, layer: usize, props: Value) -> Result<(), AutoCutError> {
    run(ex, "graphics.set", json!({"clip": clip, "layer": layer, "props": props})).await.map(|_| ())
}

/// A text layer added to graphic clip `clip`: its index.
pub(crate) async fn add_text(ex: &dyn Executor, clip: u64, text: &str, pos: [f64; 2], size: f64, props: Value) -> Result<usize, AutoCutError> {
    let r = run(ex, "graphics.newText", json!({"clip": clip, "text": text, "position": pos, "size": size})).await?;
    let layer = layer_of(&r, clip, "graphics.newText")?;
    set(ex, clip, layer, props).await?;
    Ok(layer)
}

/// A shape layer (`rectangle`, `ellipse`) added to graphic clip `clip`: its index.
pub(crate) async fn add_shape(ex: &dyn Executor, clip: u64, shape: &str, centre: [f64; 2], size: [f64; 2], props: Value) -> Result<usize, AutoCutError> {
    let r = run(ex, "graphics.newShape", json!({"clip": clip, "shape": shape, "position": centre, "size": size})).await?;
    let layer = layer_of(&r, clip, "graphics.newShape")?;
    set(ex, clip, layer, props).await?;
    Ok(layer)
}

fn layer_of(r: &Value, clip: u64, cmd: &str) -> Result<usize, AutoCutError> {
    if r["clip"].as_u64() != Some(clip) {
        return Err(AutoCutError::Executor(format!("{cmd} did not add to graphic {clip}")));
    }
    r["layer"].as_u64().and_then(|l| usize::try_from(l).ok()).ok_or_else(|| AutoCutError::Executor(format!("{cmd} added no layer")))
}

/// A new graphic clip of one shape on video track `track` (an index) from `t` for `dur`
/// seconds (both ends on frames): its id.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn shape_clip(
    ex: &dyn Executor,
    track: usize,
    fps: f64,
    shape: &str,
    centre: [f64; 2],
    size: [f64; 2],
    t: f64,
    dur: f64,
    props: Value,
) -> Result<u64, AutoCutError> {
    let (t0, t1) = (on_frame(t, fps), on_frame(t + dur, fps));
    let dur = (t1 - t0).max(1.0 / fps.max(1.0));
    let r = run(ex, "graphics.newShape", json!({"shape": shape, "position": centre, "size": size, "seconds": dur, "time": ticks(t0), "track": track})).await?;
    let clip = r["clip"].as_u64().ok_or_else(|| AutoCutError::Executor("graphics.newShape made no clip".into()))?;
    set(ex, clip, 0, props).await?;
    Ok(clip)
}

/// A new graphic clip of one text on video track `track` from `t` for `dur` seconds.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn text_clip(
    ex: &dyn Executor,
    track: usize,
    fps: f64,
    text: &str,
    pos: [f64; 2],
    size: f64,
    t: f64,
    dur: f64,
    props: Value,
) -> Result<u64, AutoCutError> {
    crate::showcase::title(ex, track, fps, text, t, dur, size, (pos[0], pos[1]), props).await
}

/// A line of a [`panel_text`].
pub(crate) struct Line<'a> {
    pub text: &'a str,
    pub role: Role,
    /// Size at 1080 lines.
    pub size: f64,
    pub color: &'a str,
    pub caps: bool,
    /// Opacity, percent.
    pub opacity: f64,
}

/// Lines of text over a translucent panel with an accent bar down its left edge — the themed
/// lower third and scene caption — as one graphic clip on track `track` from `t` for `dur`
/// seconds, the first line's baseline at `y` and its left edge `x` in from the frame's left
/// (sequence pixels): its clip and the panel's bounds. The panel is sized to the text as the
/// engine measures it (`graphics.list`; estimated when it cannot).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn panel_text(
    ex: &dyn Executor,
    theme: &Theme,
    track: usize,
    fps: f64,
    frame: (f64, f64),
    lines: &[Line<'_>],
    t: f64,
    dur: f64,
    (x, y): (f64, f64),
) -> Result<(u64, Bounds), AutoCutError> {
    let u = unit(frame);
    let (bar, pad_x, pad_y, gap) = (5.0 * u, 22.0 * u, 16.0 * u, 10.0 * u);
    let text_x = x + bar + pad_x;
    let mut clip: Option<u64> = None;
    let mut est: Option<Bounds> = None;
    let mut baseline = y;
    for (i, l) in lines.iter().enumerate() {
        let size = l.size * u;
        if i > 0 {
            baseline += size + gap;
        }
        let mut props = theme.text_props(l.role, l.color, "left");
        props["opacity"] = json!(l.opacity);
        if l.caps {
            props["caps"] = json!("All Caps");
        }
        let pos = [text_x, baseline];
        match clip {
            None => clip = Some(text_clip(ex, track, fps, l.text, pos, size, t, dur, props).await?),
            Some(c) => {
                add_text(ex, c, l.text, pos, size, props).await?;
            }
        }
        let shown = if l.caps { l.text.to_uppercase() } else { l.text.to_string() };
        let b = estimate(&shown, size, pos, "left");
        est = Some(est.map_or(b, |e| e.union(b)));
    }
    let clip = clip.ok_or_else(|| AutoCutError::Invalid("a panel without text".into()))?;
    let text = match measure(ex, clip, None).await {
        Some(b) => b,
        None => est.unwrap_or(Bounds { left: text_x, top: y - 40.0 * u, right: text_x + 200.0 * u, bottom: y + 10.0 * u }),
    };
    let panel = Bounds { left: x, top: text.top - pad_y, right: text.right + pad_x, bottom: text.bottom + pad_y };
    let back =
        add_shape(ex, clip, "rectangle", panel.centre(), panel.size(), json!({"fill_color": theme.panel_rgba(), "stroke": false, "corner_radius": 4.0 * u}))
            .await?;
    run(ex, "graphics.sendToBack", json!({"clip": clip, "layer": back})).await?;
    let bar_box = Bounds { left: x, top: panel.top, right: x + bar, bottom: panel.bottom };
    add_shape(ex, clip, "rectangle", bar_box.centre(), bar_box.size(), json!({"fill_color": theme.accent, "stroke": false})).await?;
    Ok((clip, panel))
}

/// Shrink text layer `layer` of `clip` (set at `size`, measured `b`) to fit `room` pixels
/// across, when it is wider: a long title in a wide fallback family. Its bounds as they are now.
async fn fit_width(ex: &dyn Executor, clip: u64, layer: usize, size: f64, b: Bounds, room: f64) -> Bounds {
    let wide = b.right - b.left;
    if !(wide.is_finite() && wide > room && room > 0.0) {
        return b;
    }
    let k = room / wide;
    if set(ex, clip, layer, json!({"size": size * k})).await.is_err() {
        return b;
    }
    measure(ex, clip, Some(layer)).await.unwrap_or(Bounds { left: b.left * k, top: b.top, right: b.right * k, bottom: b.bottom })
}

/// A short accent rule under `b`, centred on `x`, added to `clip`: its bounds.
async fn rule(ex: &dyn Executor, theme: &Theme, clip: u64, x: f64, b: Bounds, u: f64) -> Result<Bounds, AutoCutError> {
    let r = Bounds { left: x - 70.0 * u, top: b.bottom + 16.0 * u, right: x + 70.0 * u, bottom: b.bottom + 21.0 * u };
    add_shape(ex, clip, "rectangle", r.centre(), r.size(), json!({"fill_color": theme.accent, "stroke": false})).await?;
    Ok(r)
}

/// The themed title card: the title in the title font, in capitals, over a short accent rule,
/// and the years under it in the number font and the accent colour — the title on track
/// `titles` from `t`, the years on `over` from half a second later, both to `end`.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn title_card(
    ex: &dyn Executor,
    theme: &Theme,
    [titles, over]: [usize; 2],
    fps: f64,
    frame: (f64, f64),
    title: &str,
    subtitle: &str,
    t: f64,
    end: f64,
) -> Result<(), AutoCutError> {
    let (w, h) = frame;
    let u = unit(frame);
    let chars = title.chars().count().max(1) as f64;
    let size = (w * 0.86 / (0.5 * chars)).clamp(48.0 * u, 132.0 * u);
    let pos = [w / 2.0, h * 0.46];
    let mut props = theme.text_props(Role::Title, &theme.text, "center");
    props["caps"] = json!("All Caps");
    props["tracking"] = json!(30.0);
    props["shadow_blur"] = json!(30.0);
    props["shadow_opacity"] = json!(70.0);
    let clip = text_clip(ex, titles, fps, title, pos, size, t, end - t, props).await?;
    let b = measure(ex, clip, Some(0)).await.unwrap_or_else(|| estimate(&title.to_uppercase(), size, pos, "center"));
    let b = fit_width(ex, clip, 0, size, b, w * 0.9).await;
    let r = rule(ex, theme, clip, pos[0], b, u).await?;
    if !subtitle.trim().is_empty() && end - t > 0.6 {
        let mut props = theme.text_props(Role::Number, &theme.accent, "center");
        props["tracking"] = json!(120.0);
        let size = 46.0 * u;
        text_clip(ex, over, fps, subtitle, [pos[0], r.bottom + 22.0 * u + size * 0.74], size, t + 0.5, end - t - 0.5, props).await?;
    }
    Ok(())
}

/// The themed chapter card (track `track`, from `t` for `dur` seconds): a year in the number
/// font, an event's name in the title font in capitals, the text as [`crate::showcase`]'s card
/// lays it out at `size`, over a short accent rule.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn chapter_card(
    ex: &dyn Executor,
    theme: &Theme,
    track: usize,
    fps: f64,
    frame: (f64, f64),
    text: &str,
    size: f64,
    t: f64,
    dur: f64,
) -> Result<u64, AutoCutError> {
    let (w, h) = frame;
    let u = unit(frame);
    let year = text.trim().chars().all(|c| c.is_ascii_digit());
    let (role, size) = if year { (Role::Number, 170.0 * u) } else { (Role::Title, size * 1.1) };
    let pos = [w / 2.0, h * 0.55];
    let mut props = theme.text_props(role, &theme.text, "center");
    props["caps"] = json!("All Caps");
    props["tracking"] = json!(if year { 20.0 } else { 40.0 });
    props["shadow_blur"] = json!(30.0);
    props["shadow_opacity"] = json!(70.0);
    let clip = text_clip(ex, track, fps, text, pos, size, t, dur, props).await?;
    let b = measure(ex, clip, Some(0)).await.unwrap_or_else(|| estimate(text, size, pos, "center"));
    let b = fit_width(ex, clip, 0, size, b, w * 0.9).await;
    rule(ex, theme, clip, pos[0], b, u).await?;
    Ok(clip)
}

/// The themed outro: the count (`closing`) in the title font over an accent rule, `tagline`
/// under it in the body font and the accent colour, from `t` for `len` seconds.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn outro_card(
    ex: &dyn Executor,
    theme: &Theme,
    [titles, over]: [usize; 2],
    fps: f64,
    frame: (f64, f64),
    closing: &str,
    tagline: &str,
    t: f64,
    len: f64,
) -> Result<(), AutoCutError> {
    let (w, h) = frame;
    let u = unit(frame);
    let size = 150.0 * u;
    let pos = [w / 2.0, h * 0.48];
    let mut props = theme.text_props(Role::Title, &theme.text, "center");
    props["caps"] = json!("All Caps");
    props["tracking"] = json!(30.0);
    props["shadow_blur"] = json!(30.0);
    props["shadow_opacity"] = json!(70.0);
    let clip = text_clip(ex, titles, fps, closing, pos, size, t + 0.3, len - 0.3, props).await?;
    let b = measure(ex, clip, Some(0)).await.unwrap_or_else(|| estimate(closing, size, pos, "center"));
    let b = fit_width(ex, clip, 0, size, b, w * 0.9).await;
    let r = rule(ex, theme, clip, pos[0], b, u).await?;
    if len > 1.5 {
        let size = 50.0 * u;
        let props = theme.text_props(Role::Body, &theme.accent, "center");
        text_clip(ex, over, fps, tagline, [pos[0], r.bottom + 22.0 * u + size * 0.74], size, t + 0.9, len - 0.9, props).await?;
    }
    Ok(())
}

/// How long each chip waits after the one before, and how far a chip rises as it comes in
/// (1080 lines).
pub const CHIP_STAGGER: f64 = 0.3;
const CHIP_RISE: f64 = 14.0;
/// Chip text size (1080 lines).
const CHIP_SIZE: f64 = 30.0;

/// The spec chips of an entry: one graphic clip each on `tracks` (one track a chip), in a row
/// beside `beside` (the lower third's bounds) — above it when the row would leave the frame —
/// the first coming in at `t0`, each next [`CHIP_STAGGER`] later, rising and fading in, all
/// leaving together at `t1`. Their clips.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn chips(
    ex: &dyn Executor,
    theme: &Theme,
    tracks: &[usize],
    fps: f64,
    frame: (f64, f64),
    texts: &[String],
    beside: Bounds,
    t0: f64,
    t1: f64,
) -> Result<Vec<u64>, AutoCutError> {
    let u = unit(frame);
    let (size, pad, gap) = (CHIP_SIZE * u, 12.0 * u, 14.0 * u);
    let height = size + 2.0 * pad;
    let mut x = beside.right + gap + pad;
    let baseline = (beside.top + beside.bottom) / 2.0 + size * 0.36;
    let mut placed: Vec<(u64, f64, f64)> = Vec::new();
    for (k, (text, track)) in texts.iter().zip(tracks).enumerate() {
        let t_in = on_frame(t0 + k as f64 * CHIP_STAGGER, fps);
        if t1 - t_in < 0.5 {
            break;
        }
        let mut props = theme.text_props(Role::Number, &theme.accent, "left");
        props["background"] = json!(true);
        props["background_color"] = json!(theme.panel);
        props["background_opacity"] = json!((theme.panel_alpha * 100.0).clamp(0.0, 100.0));
        props["background_size"] = json!(pad);
        props["background_radius"] = json!(height / 2.0);
        props["shadow"] = json!(false);
        let clip = text_clip(ex, *track, fps, text, [x, baseline], size, t_in, t1 - t_in, props).await?;
        let b = measure(ex, clip, Some(0)).await.unwrap_or_else(|| estimate(text, size, [x, baseline], "left"));
        placed.push((clip, t_in, b.right - b.left));
        x = b.right + pad + gap + pad;
    }
    // the row past the frame's safe edge: above the lower third instead, from its left
    let (w, _) = frame;
    if x - gap - pad > w * 0.95 {
        let y = beside.top - gap - pad - size * 0.22;
        let mut left = beside.left + pad;
        for (clip, _, width) in &placed {
            set(ex, *clip, 0, json!({"position": [left, y]})).await?;
            left += width + 2.0 * pad + gap;
        }
    }
    for (clip, start, _) in &placed {
        // cosmetic, like the lower thirds' slide: a chip that does not move is still right
        rise_in_out(ex, *clip, *start, on_frame(t1, fps), frame, fps, CHIP_RISE * u).await.ok();
    }
    Ok(placed.into_iter().map(|p| p.0).collect())
}

/// How long a chip takes to rise in (and sink out), seconds.
const RISE: f64 = 0.35;

/// Animate graphic clip `clip` (on screen from `t0` to `t1`, on frames): up into place from
/// `by` pixels lower while fading in, decelerating; out the same way — Motion position and
/// Opacity keyframes.
async fn rise_in_out(ex: &dyn Executor, clip: u64, t0: f64, t1: f64, (w, h): (f64, f64), fps: f64, by: f64) -> Result<(), AutoCutError> {
    let last = on_frame(t1 - 1.0 / fps, fps);
    if last - t0 < 2.0 * RISE {
        return Ok(());
    }
    let (home, away) = ([w / 2.0, h / 2.0], [w / 2.0, h / 2.0 + by]);
    let (shown, gone) = (on_frame(t0 + RISE, fps), on_frame(last - RISE, fps));
    let keys = [
        ("motion", "position", t0, json!(away), "linear"),
        ("motion", "position", shown, json!(home), "easeIn"),
        ("motion", "position", gone, json!(home), "easeOut"),
        ("motion", "position", last, json!(away), "linear"),
        ("opacity", "opacity", t0, json!(0.0), "linear"),
        ("opacity", "opacity", shown, json!(100.0), "linear"),
        ("opacity", "opacity", gone, json!(100.0), "linear"),
        ("opacity", "opacity", last, json!(0.0), "linear"),
    ];
    for (effect, param, at, value, _) in &keys {
        run(ex, "effects.setParam", json!({"clip": clip, "effect": effect, "param": param, "value": value, "time": ticks(*at), "keyframe": true})).await?;
    }
    for (effect, param, at, _, ease) in keys.iter().filter(|k| k.4 != "linear") {
        run(ex, "effects.setInterpolation", json!({"clip": clip, "effect": effect, "param": param, "time": ticks(*at), "interpolation": ease})).await?;
    }
    Ok(())
}

/// Fade graphic clip `clip` in at `t0` and out at `t1` over `fade` seconds (either end left
/// alone when `None`): Opacity keyframes.
async fn fade(ex: &dyn Executor, clip: u64, t0: Option<f64>, t1: Option<f64>, fade: f64, fps: f64) -> Result<(), AutoCutError> {
    let mut keys = Vec::new();
    if let Some(a) = t0 {
        keys.push((on_frame(a, fps), 0.0));
        keys.push((on_frame(a + fade, fps), 100.0));
    }
    if let Some(b) = t1 {
        let last = on_frame(b - 1.0 / fps, fps);
        keys.push((on_frame(last - fade, fps), 100.0));
        keys.push((last, 0.0));
    }
    keys.sort_by(|a, b| a.0.total_cmp(&b.0));
    keys.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);
    for (at, v) in keys {
        run(ex, "effects.setParam", json!({"clip": clip, "effect": "opacity", "param": "opacity", "value": v, "time": ticks(at), "keyframe": true})).await?;
    }
    Ok(())
}

/// How long a graphic takes to step aside for a scene, and to come back, seconds.
const STEP_ASIDE: f64 = 0.3;

/// Fade graphic clip `clip` out at the start of each of `spans` (seconds) and back in at its
/// end, over [`STEP_ASIDE`] each way (Opacity keyframes): the counter while a scene plays. A
/// span too short to fade out and in is left alone.
pub(crate) async fn hide_during(ex: &dyn Executor, clip: u64, spans: &[(f64, f64)], fps: f64) -> Result<(), AutoCutError> {
    let mut keys: Vec<(f64, f64)> = Vec::new();
    for (a, b) in spans {
        if !(a.is_finite() && b.is_finite()) || b - a < 4.0 * STEP_ASIDE {
            continue;
        }
        keys.extend([(on_frame(*a, fps), 100.0), (on_frame(a + STEP_ASIDE, fps), 0.0), (on_frame(b - STEP_ASIDE, fps), 0.0), (on_frame(*b, fps), 100.0)]);
    }
    for (at, v) in keys {
        run(ex, "effects.setParam", json!({"clip": clip, "effect": "opacity", "param": "opacity", "value": v, "time": ticks(at), "keyframe": true})).await?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The timeline

/// How long the marker takes to glide to an entry's date, seconds.
pub const GLIDE: f64 = 0.6;
/// Most years a timeline spans.
pub const MAX_YEARS: i64 = 60;
/// How long the timeline takes to fade in and out, seconds.
const TIMELINE_FADE: f64 = 0.4;
/// Where the timeline runs, as shares of the frame: from the left, to the right, and its line.
const LEFT: f64 = 0.06;
const RIGHT: f64 = 0.94;
pub const LINE_Y: f64 = 0.925;

/// An entry as the timeline sees it: its number, when it starts on the timeline (seconds), when
/// it was filmed and the event it belongs to.
#[derive(Clone, Debug, PartialEq)]
pub struct TimelineEntry {
    pub number: usize,
    pub start: f64,
    pub unix: Option<i64>,
    pub event: Option<String>,
}

/// Dates → places along the timeline: from the first second of the first year to the last of
/// the last, `left` to `right` pixels.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Scale {
    pub first_year: i64,
    pub last_year: i64,
    pub left: f64,
    pub right: f64,
    start: i64,
    end: i64,
}

impl Scale {
    /// The scale over the years of `dates` (Unix seconds), across a frame `frame` wide; `None`
    /// without dates; an error for years outside 1900–2200 or more than [`MAX_YEARS`] of them.
    pub fn new(dates: &[i64], frame: (f64, f64)) -> Result<Option<Self>, AutoCutError> {
        let years: Vec<i64> = dates.iter().map(|u| ymd(*u).0).collect();
        let (Some(&first), Some(&last)) = (years.iter().min(), years.iter().max()) else { return Ok(None) };
        if !YEARS.contains(&first) || !YEARS.contains(&last) {
            return Err(invalid(format!("timeline: entries are dated from {first} to {last}; the timeline shows years from 1900 to 2200")));
        }
        if last - first + 1 > MAX_YEARS {
            return Err(invalid(format!("timeline: the entries span {} years; it shows at most {MAX_YEARS}", last - first + 1)));
        }
        let start = unix_of(first, 1, 1).ok_or_else(|| invalid("timeline: no first year"))?;
        let end = unix_of(last + 1, 1, 1).unwrap_or(start + 366 * 86_400);
        let w = if frame.0.is_finite() && frame.0 > 0.0 { frame.0 } else { 1920.0 };
        Ok(Some(Self { first_year: first, last_year: last, left: w * LEFT, right: w * RIGHT, start, end }))
    }

    /// Where `unix` sits, pixels (clamped to the timeline's ends).
    pub fn x(&self, unix: i64) -> f64 {
        let span = (self.end - self.start).max(1) as f64;
        let f = (unix.clamp(self.start, self.end) - self.start) as f64 / span;
        self.left + f * (self.right - self.left)
    }

    /// Where year `y` starts and ends, pixels.
    pub fn year(&self, y: i64) -> (f64, f64) {
        let a = unix_of(y, 1, 1).map_or(self.left, |u| self.x(u));
        let b = unix_of(y + 1, 1, 1).map_or(self.right, |u| self.x(u));
        (a, b)
    }
}

/// An event band on the timeline.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Band {
    pub name: String,
    pub x0: f64,
    pub x1: f64,
}

/// Where the marker is: it leaves at an entry's `start` and is at `x` from `arrive` on.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Stop {
    pub entry: usize,
    pub start: f64,
    pub arrive: f64,
    pub x: f64,
}

/// A stretch of entries with the same year and event: the year and the band picked out.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Run {
    pub start: f64,
    pub end: f64,
    pub year: Option<i64>,
    pub band: Option<usize>,
}

/// The timeline as laid out ([`plan_timeline`]), and the clips it was drawn with.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Timeline {
    pub scale: Scale,
    /// On screen from the first entry to the outro, seconds.
    pub from: f64,
    pub to: f64,
    pub bands: Vec<Band>,
    pub stops: Vec<Stop>,
    pub runs: Vec<Run>,
    /// The line, ticks, years and bands; the marker; the picked-out year and event of each run.
    pub base: Option<u64>,
    pub marker: Option<u64>,
    pub now: Vec<u64>,
}

/// Narrowest event band, pixels at 1080 lines.
const MIN_BAND: f64 = 14.0;

/// Lay out the timeline for `entries` (in time order) shown from `from` to `to` seconds: its
/// years from the first entry's to the last's; bands for the plan's `events` (clipped to those
/// years), or else for the entries' own events (from the first of each to its last); the
/// marker's stops — at the first dated entry from `from`, then gliding to each later dated
/// entry's date over [`GLIDE`] seconds from its start (on frames of `fps`); and the runs of
/// entries with the same year and event. `None` when no entry has a date.
pub fn plan_timeline(entries: &[TimelineEntry], events: &[Event], from: f64, to: f64, frame: (f64, f64), fps: f64) -> Result<Option<Timeline>, AutoCutError> {
    let dates: Vec<i64> = entries.iter().filter_map(|e| e.unix).collect();
    let Some(scale) = Scale::new(&dates, frame)? else { return Ok(None) };
    let fps = if fps.is_finite() && fps > 0.0 { fps } else { 30.0 };
    let from = if from.is_finite() { on_frame(from, fps) } else { 0.0 };
    let to = if to.is_finite() && to >= from { on_frame(to, fps) } else { from };
    let min_band = MIN_BAND * unit(frame);
    let widen = |x0: f64, x1: f64| {
        if x1 - x0 >= min_band {
            (x0, x1)
        } else {
            let mid = ((x0 + x1) / 2.0).clamp(scale.left + min_band / 2.0, scale.right - min_band / 2.0);
            (mid - min_band / 2.0, mid + min_band / 2.0)
        }
    };
    let (lo, hi) = (scale.start, scale.end);
    // the bands, and which band each entry falls in
    let (bands, of_entry): (Vec<Band>, Vec<Option<usize>>) = if events.is_empty() {
        let mut bands: Vec<(String, i64, i64)> = Vec::new();
        let mut of = Vec::new();
        for e in entries {
            let name = e.event.as_deref().map(str::trim).filter(|s| !s.is_empty());
            let idx = match (name, e.unix) {
                (Some(n), Some(u)) => Some(match bands.iter().position(|b| b.0.eq_ignore_ascii_case(n)) {
                    Some(i) => {
                        if let Some(b) = bands.get_mut(i) {
                            b.1 = b.1.min(u);
                            b.2 = b.2.max(u);
                        }
                        i
                    }
                    None => {
                        bands.push((n.to_string(), u, u));
                        bands.len() - 1
                    }
                }),
                (Some(n), None) => bands.iter().position(|b| b.0.eq_ignore_ascii_case(n)),
                _ => None,
            };
            of.push(idx);
        }
        let bands = bands
            .into_iter()
            .map(|(name, a, b)| {
                let (x0, x1) = widen(scale.x(a), scale.x(b));
                Band { name, x0, x1 }
            })
            .collect();
        (bands, of)
    } else {
        let shown: Vec<&Event> = events.iter().filter(|ev| ev.to >= lo && ev.from < hi).collect();
        let bands = shown
            .iter()
            .map(|ev| {
                let (x0, x1) = widen(scale.x(ev.from), scale.x(ev.to));
                Band { name: ev.name.clone(), x0, x1 }
            })
            .collect();
        let of = entries
            .iter()
            .map(|e| {
                let by_date = e.unix.and_then(|u| shown.iter().position(|ev| ev.from <= u && u <= ev.to));
                let by_name = || e.event.as_deref().map(str::trim).and_then(|n| shown.iter().position(|ev| ev.name.eq_ignore_ascii_case(n)));
                by_date.or_else(by_name)
            })
            .collect();
        (bands, of)
    };
    // the marker's stops
    let mut stops: Vec<Stop> = Vec::new();
    for (k, e) in entries.iter().enumerate() {
        let Some(u) = e.unix else { continue };
        let x = scale.x(u);
        let start = on_frame(e.start, fps).clamp(from, to);
        if stops.is_empty() {
            stops.push(Stop { entry: e.number, start: from, arrive: from, x });
            continue;
        }
        if stops.last().is_some_and(|s| (s.x - x).abs() < 0.5) {
            continue;
        }
        // arriving before the next entry starts (and before the timeline goes)
        let next = entries.get(k + 1).map_or(to, |n| on_frame(n.start, fps)).min(to);
        let arrive = on_frame(start + GLIDE, fps).min(next).max(on_frame(start + 1.0 / fps, fps));
        stops.push(Stop { entry: e.number, start, arrive, x });
    }
    // the runs of one year and event
    let mut runs: Vec<Run> = Vec::new();
    let (mut year, mut band): (Option<i64>, Option<usize>) = (None, None);
    for (e, b) in entries.iter().zip(&of_entry) {
        let y = e.unix.map(|u| ymd(u).0).or(year);
        let b = b.or(if e.unix.is_none() { band } else { None });
        let start = on_frame(e.start, fps).clamp(from, to);
        if runs.last().is_some_and(|r| r.year == y && r.band == b) {
            continue;
        }
        if let Some(last) = runs.last_mut() {
            last.end = start;
        }
        runs.push(Run { start: if runs.is_empty() { from } else { start }, end: to, year: y, band: b });
        (year, band) = (y, b);
    }
    runs.retain(|r| r.end > r.start);
    Ok(Some(Timeline { scale, from, to, bands, stops, runs, base: None, marker: None, now: Vec::new() }))
}

/// Draw the timeline `tl` ([`plan_timeline`]) in `theme` on three video tracks: `base` (the
/// line, the years and the bands, dim), `now` (the year and event of each run, picked out) and
/// `marker` (a dot gliding along it: Motion position keyframes, eased out of each stop's start
/// and into its arrival). It fades in and out.
pub(crate) async fn draw_timeline(
    ex: &dyn Executor,
    theme: &Theme,
    tl: &mut Timeline,
    [base, now, marker]: [usize; 3],
    fps: f64,
    frame: (f64, f64),
) -> Result<(), AutoCutError> {
    let (w, h) = frame;
    let u = unit(frame);
    let len = tl.to - tl.from;
    if len < 1.0 / fps.max(1.0) {
        return Ok(());
    }
    let s = tl.scale;
    let y = h * LINE_Y;
    let (thick, tick, band_h) = ((2.0 * u).max(1.0), 12.0 * u, 8.0 * u);
    let (year_size, event_size) = (21.0 * u, 18.0 * u);
    // (the event names clear of the marker, which is 30 px across)
    let (year_y, event_y) = (y + 30.0 * u, y - 24.0 * u);
    // the line, the ticks at each year's start, the years and the bands
    let line = Bounds { left: s.left, top: y - thick / 2.0, right: s.right, bottom: y + thick / 2.0 };
    let b =
        shape_clip(ex, base, fps, "rectangle", line.centre(), line.size(), tl.from, len, json!({"fill_color": theme.text, "stroke": false, "opacity": 40.0}))
            .await?;
    tl.base = Some(b);
    let mut labels: Vec<(f64, f64)> = Vec::new();
    let clear = |labels: &[(f64, f64)], a: f64, z: f64| labels.iter().all(|(l, r)| z + 6.0 * u < *l || a > *r + 6.0 * u);
    for yr in s.first_year..=s.last_year + 1 {
        let (x0, _) = s.year(yr);
        add_shape(ex, b, "rectangle", [x0, y], [thick, tick], json!({"fill_color": theme.text, "stroke": false, "opacity": 55.0})).await?;
    }
    for yr in s.first_year..=s.last_year {
        let (x0, x1) = s.year(yr);
        let mid = (x0 + x1) / 2.0;
        let half = estimate(&yr.to_string(), year_size, [mid, year_y], "center");
        if !clear(&labels, half.left, half.right) {
            continue;
        }
        labels.push((half.left, half.right));
        let mut props = theme.text_props(Role::Number, &theme.text, "center");
        props["opacity"] = json!(55.0);
        add_text(ex, b, &yr.to_string(), [mid, year_y], year_size, props).await?;
    }
    let mut event_labels: Vec<(f64, f64)> = Vec::new();
    for band in &tl.bands {
        let bx = Bounds { left: band.x0, top: y - band_h / 2.0, right: band.x1, bottom: y + band_h / 2.0 };
        add_shape(
            ex,
            b,
            "rectangle",
            bx.centre(),
            bx.size(),
            json!({"fill_color": theme.accent, "stroke": false, "opacity": 38.0, "corner_radius": band_h / 2.0}),
        )
        .await?;
        let mid = (band.x0 + band.x1) / 2.0;
        let e = estimate(&band.name, event_size, [mid, event_y], "center");
        if !clear(&event_labels, e.left, e.right) || e.left < 0.0 || e.right > w {
            continue;
        }
        event_labels.push((e.left, e.right));
        let mut props = theme.text_props(Role::Body, &theme.text, "center");
        props["opacity"] = json!(60.0);
        add_text(ex, b, &band.name, [mid, event_y], event_size, props).await?;
    }
    fade(ex, b, Some(tl.from), Some(tl.to), TIMELINE_FADE, fps).await.ok();
    // the year and event of each run, picked out
    let pill = |props: &mut Value| {
        props["background"] = json!(true);
        props["background_color"] = json!(theme.panel);
        props["background_opacity"] = json!(90.0);
        props["background_size"] = json!(6.0 * u);
        props["background_radius"] = json!(8.0 * u);
        props["shadow"] = json!(false);
    };
    let n_runs = tl.runs.len();
    for (i, r) in tl.runs.clone().iter().enumerate() {
        let mut clip: Option<u64> = None;
        if let Some(yr) = r.year {
            let (x0, x1) = s.year(yr);
            let mut props = theme.text_props(Role::Number, &theme.accent, "center");
            pill(&mut props);
            clip = Some(text_clip(ex, now, fps, &yr.to_string(), [(x0 + x1) / 2.0, year_y], year_size * 1.1, r.start, r.end - r.start, props).await?);
        }
        if let Some(band) = r.band.and_then(|k| tl.bands.get(k)) {
            let bx = Bounds { left: band.x0, top: y - band_h / 2.0, right: band.x1, bottom: y + band_h / 2.0 };
            let props = json!({"fill_color": theme.accent, "stroke": false, "corner_radius": band_h / 2.0});
            let c = match clip {
                Some(c) => {
                    add_shape(ex, c, "rectangle", bx.centre(), bx.size(), props).await?;
                    c
                }
                None => shape_clip(ex, now, fps, "rectangle", bx.centre(), bx.size(), r.start, r.end - r.start, props).await?,
            };
            let mid = (band.x0 + band.x1) / 2.0;
            let mut props = theme.text_props(Role::Body, &theme.text, "center");
            pill(&mut props);
            add_text(ex, c, &band.name, [mid, event_y], event_size, props).await?;
            clip = Some(c);
        }
        if let Some(c) = clip {
            let first = (i == 0).then_some(tl.from);
            let last = (i + 1 == n_runs).then_some(tl.to);
            fade(ex, c, first, last, TIMELINE_FADE, fps).await.ok();
            tl.now.push(c);
        }
    }
    // the marker: a dot and its glow, drawn mid-frame and moved along with Motion position
    let d = 16.0 * u;
    let m = shape_clip(
        ex,
        marker,
        fps,
        "ellipse",
        [w / 2.0, y],
        [d * 1.9, d * 1.9],
        tl.from,
        len,
        json!({"fill_color": theme.accent, "stroke": false, "opacity": 30.0}),
    )
    .await?;
    add_shape(
        ex,
        m,
        "ellipse",
        [w / 2.0, y],
        [d, d],
        json!({"fill_color": theme.accent, "stroke": true, "stroke_color": theme.panel, "stroke_width": 2.0 * u}),
    )
    .await?;
    tl.marker = Some(m);
    let at = |x: f64| json!([x, h / 2.0]);
    if let Some(first) = tl.stops.first() {
        run(ex, "effects.setParam", json!({"clip": m, "effect": "motion", "param": "position", "value": at(first.x)})).await?;
    }
    let mut prev: Option<f64> = tl.stops.first().map(|s| s.x);
    let mut keys: Vec<(f64, f64, &str)> = Vec::new();
    for st in tl.stops.iter().skip(1) {
        let Some(px) = prev else { break };
        keys.push((st.start, px, "easeOut"));
        keys.push((st.arrive.max(st.start + 0.5 / fps.max(1.0)), st.x, "easeIn"));
        prev = Some(st.x);
    }
    for (t, x, _) in &keys {
        run(ex, "effects.setParam", json!({"clip": m, "effect": "motion", "param": "position", "value": at(*x), "time": ticks(*t), "keyframe": true})).await?;
    }
    for (t, _, ease) in &keys {
        run(ex, "effects.setInterpolation", json!({"clip": m, "effect": "motion", "param": "position", "time": ticks(*t), "interpolation": ease})).await.ok();
    }
    fade(ex, m, Some(tl.from), Some(tl.to), TIMELINE_FADE, fps).await.ok();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colours_parse_with_and_without_alpha() {
        assert_eq!(parse_hex("#00e5ff"), Some([0, 0xe5, 0xff, 255]));
        assert_eq!(parse_hex("0a1018b8"), Some([0x0a, 0x10, 0x18, 0xb8]));
        for bad in ["", "#12345", "#gggggg", "#1234567", "red", "#ffffff00ff", "ÿÿÿÿÿÿ"] {
            assert_eq!(parse_hex(bad), None, "{bad}");
        }
        assert_eq!(with_alpha("#0a1018", 0.72), "#0a1018b8");
        assert_eq!(with_alpha("#0a1018", f64::NAN), "#0a1018ff");
    }

    #[test]
    fn dates_round_trip_through_unix_seconds() {
        for (y, m, d) in [(1970, 1, 1), (2000, 2, 29), (2022, 10, 14), (2026, 12, 31), (1900, 3, 1), (2200, 12, 31)] {
            let u = unix_of(y, m, d).unwrap();
            assert_eq!(ymd(u), (y, m, d));
        }
        assert_eq!(unix_of(2023, 2, 29), None);
        assert_eq!(unix_of(1899, 12, 31), None);
        assert_eq!(unix_of(2022, 13, 1), None);
        assert_eq!(date_span(&json!("2022-10")), Some((unix_of(2022, 10, 1).unwrap(), unix_of(2022, 10, 31).unwrap() + 86_399)));
        assert_eq!(date_span(&json!(2021)).map(|s| ymd(s.1)), Some((2021, 12, 31)));
        for bad in [json!("2022-1-5"), json!("22-01-05"), json!("2022-02-30"), json!("2022/10/01"), json!(-5), json!(true), json!("2022-10-01-02")] {
            assert_eq!(date_span(&bad), None, "{bad}");
        }
    }

    #[test]
    fn styles_come_from_a_name_or_fields_and_refuse_nonsense() {
        assert_eq!(Style::from_plan(&json!("tech")).unwrap(), Some(Style::tech()));
        assert_eq!(Style::from_plan(&json!("classic")).unwrap(), None);
        assert_eq!(Style::from_plan(&Value::Null).unwrap(), None);
        let s = Style::from_plan(&json!({"accent": "#FF0080", "panel": {"color": "#101010", "alpha": 0.5}, "titleFont": "Avenir Next"})).unwrap().unwrap();
        assert_eq!((s.name.as_str(), s.accent.as_str(), s.panel.as_str(), s.panel_alpha), ("custom", "#ff0080", "#101010", 0.5));
        assert_eq!((s.title_font.as_str(), s.body_font.as_str()), ("Avenir Next", "Futura"), "over the tech theme");
        assert_eq!(Style::from_plan(&json!({"panel": "#10101080"})).unwrap().unwrap().panel_alpha, 128.0 / 255.0);
        for bad in [
            json!("neon"),
            json!(5),
            json!({"accent": "cyan"}),
            json!({"accent": "#00e5ff80"}),
            json!({"titleFont": ""}),
            json!({"titleFont": "x".repeat(65)}),
            json!({"titleFont": 3}),
            json!({"panel": {"color": "#000000", "alpha": 2.0}}),
            json!({"panel": {"colour": "#000000"}}),
            json!({"accnet": "#00e5ff"}),
            json!({"base": "neon"}),
        ] {
            let e = Style::from_plan(&bad).unwrap_err().to_string();
            assert!(e.contains("style"), "{bad}: {e}");
        }
    }

    #[test]
    fn missing_families_fall_back_to_inter() {
        let mac = vec![
            ("DIN Condensed".to_string(), vec!["Bold".to_string()]),
            ("DIN Alternate".to_string(), vec!["Bold".to_string()]),
            ("Futura".to_string(), vec!["Medium".to_string(), "Medium Italic".to_string(), "Bold".to_string()]),
            ("Inter".to_string(), vec!["Regular".to_string(), "Bold".to_string()]),
        ];
        let t = Style::tech().resolve(Some(&mac));
        assert_eq!((t.title.family.as_str(), t.title.style.as_str()), ("DIN Condensed", "Bold"));
        assert_eq!((t.body.family.as_str(), t.body.style.as_str()), ("Futura", "Medium"));
        assert_eq!(t.number.family, "DIN Alternate");
        assert!(t.missing.is_empty());
        // elsewhere (Linux, Windows): Inter for all three, and which were missing
        let bare = vec![("Inter".to_string(), vec!["Regular".to_string(), "Medium".to_string(), "Bold".to_string()])];
        let t = Style::tech().resolve(Some(&bare));
        assert!([&t.title, &t.body, &t.number].iter().all(|f| f.family == FALLBACK_FAMILY), "{t:?}");
        assert_eq!((t.title.style.as_str(), t.body.style.as_str()), ("Bold", "Medium"));
        assert_eq!(t.missing, vec!["DIN Condensed", "Futura", "DIN Alternate"]);
        // a family with other styles: the nearest upright one; case as installed
        let odd = vec![("din condensed".to_string(), vec!["Light Italic".to_string(), "Light".to_string()])];
        assert_eq!(Style::tech().resolve(Some(&odd)).title, Font { family: "din condensed".into(), style: "Light".into() });
        // the engine cannot say: the names as given
        assert_eq!(Style::tech().resolve(None).title.family, "DIN Condensed");
        let p = t.text_props(Role::Title, "#ffffff", "left");
        assert_eq!((p["font"].as_str(), p["font_style"].as_str()), (Some("Inter"), Some("Bold")));
    }

    #[test]
    fn events_transitions_and_finishing_are_checked() {
        let ev = events_from_plan(
            &json!([{"name": "GITEX 2022", "from": "2022-10-10", "to": "2022-10-14"}, {"name": "Expo 2020", "from": "2021-10", "to": "2022-03"}]),
        )
        .unwrap();
        assert_eq!(ev.len(), 2);
        assert_eq!(ymd(ev[1].to), (2022, 3, 31));
        for bad in [
            json!({"name": "x"}),
            json!([{"name": "", "from": "2022", "to": "2022"}]),
            json!([{"name": "x", "from": "2023-01-01", "to": "2022-01-01"}]),
            json!([{"name": "x", "from": "soon", "to": "2022"}]),
            json!([{"name": "x", "from": "2022", "to": "2022", "where": "Dubai"}]),
            json!([{"name": "x", "from": "1066", "to": "2022"}]),
            json!(vec![json!({"name": "x", "from": "2022", "to": "2022"}); 65]),
        ] {
            assert!(events_from_plan(&bad).is_err(), "{bad}");
        }
        assert_eq!(TransitionPlan::from_plan(&json!(true)).unwrap(), Some(TransitionPlan::catalog()));
        let t = TransitionPlan::from_plan(&json!({"between": ["GL CrossZoom", "flow_morph"], "within": []})).unwrap().unwrap();
        assert!(t.within.is_empty() && t.chapter == TransitionPlan::catalog().chapter);
        assert_eq!(t.between.len(), 2);
        for id in TransitionPlan::catalog().between.iter().chain(&TransitionPlan::catalog().chapter) {
            assert!(transition_ids("x", &json!(id)).is_ok(), "{id} is in the catalogue");
        }
        for bad in [
            json!({"between": ["no_such_transition"]}),
            json!({"between": [3]}),
            json!({"after": []}),
            json!("all"),
            json!({"chapter": vec!["flow_morph"; 65]}),
        ] {
            assert!(TransitionPlan::from_plan(&bad).is_err(), "{bad}");
        }
        assert_eq!(Loudness::from_plan(&json!("social")).unwrap(), Some(Loudness::Named("social".into())));
        assert_eq!(Loudness::from_plan(&json!("-16")).unwrap(), Some(Loudness::Lufs(-16.0)));
        assert_eq!(Loudness::from_plan(&json!("off")).unwrap(), None);
        for bad in [json!("loud"), json!(5), json!(-100), json!("NaN"), json!("inf"), json!([])] {
            assert!(Loudness::from_plan(&bad).is_err(), "{bad}");
        }
        assert!(soundfont_from(&json!("")).is_err() && soundfont_from(&json!(1)).is_err());
        let look = Look::from_plan(&json!({"style": "tech", "timeline": true, "transitions": {}, "embedCaptions": true, "entries": []})).unwrap();
        assert!(look.style.is_some() && look.timeline && look.embed_captions && look.transitions == Some(TransitionPlan::catalog()));
        assert!(Look::from_plan(&json!({"timeline": "yes"})).is_err());
        assert_eq!(Look::from_plan(&json!([1, 2])).unwrap(), Look::default());
        assert_eq!(Look::plan_fields(&json!({"timeline": true, "entries": [], "style": "tech"})).len(), 2);
        // a script alone read as a plan (the CLI's --script-in): its spoken outro is no look
        let script = json!({"intro": "Hi.", "lines": ["One."], "outro": "Thanks for watching."});
        assert_eq!(Look::from_plan(&script).unwrap(), Look::default());
        assert!(Look::plan_fields(&script).is_empty());
    }

    #[test]
    fn chips_belong_to_entries_and_stay_short() {
        let mut c: BTreeMap<String, Vec<String>> = BTreeMap::new();
        c.insert("2".into(), vec![" 32.5 kg ".into(), "1.6 m/s".into()]);
        c.insert("3".into(), vec![]);
        let got = chips_by_entry(&c, &[1, 2, 3]).unwrap();
        assert_eq!(got.get(&2), Some(&vec!["32.5 kg".to_string(), "1.6 m/s".to_string()]));
        assert!(!got.contains_key(&3) && !got.contains_key(&1));
        for (k, v) in [("9", vec!["a"]), ("two", vec!["a"]), ("2", vec!["a", "b", "c", "d"]), ("2", vec![""]), ("2", vec!["line\nbreak"])] {
            let mut c = BTreeMap::new();
            c.insert(k.to_string(), v.into_iter().map(str::to_string).collect());
            assert!(chips_by_entry(&c, &[1, 2]).is_err(), "{k}");
        }
        let long: BTreeMap<String, Vec<String>> = [("1".to_string(), vec!["x".repeat(33)])].into_iter().collect();
        assert!(chips_by_entry(&long, &[1]).is_err());
    }

    fn te(number: usize, start: f64, date: (i64, u32, u32), event: Option<&str>) -> TimelineEntry {
        TimelineEntry { number, start, unix: unix_of(date.0, date.1, date.2), event: event.map(str::to_string) }
    }

    #[test]
    fn the_marker_glides_to_each_entry_s_date_as_it_starts() {
        let entries = vec![
            te(1, 6.0, (2020, 9, 8), None),
            te(2, 14.0, (2021, 12, 1), Some("Expo 2020")),
            te(3, 22.5, (2022, 1, 20), Some("Expo 2020")),
            te(4, 31.0, (2022, 10, 12), Some("GITEX 2022")),
            TimelineEntry { number: 5, start: 40.0, unix: None, event: None },
        ];
        let tl = plan_timeline(&entries, &[], 6.0, 48.0, (1920.0, 1080.0), 30.0).unwrap().unwrap();
        assert_eq!((tl.scale.first_year, tl.scale.last_year), (2020, 2022));
        assert_eq!((tl.scale.left, tl.scale.right), (1920.0 * LEFT, 1920.0 * RIGHT));
        // a stop per dated entry: the first from the start, each later one leaving at its entry's
        // start and arriving GLIDE later, on frames, at its date's place
        assert_eq!(tl.stops.iter().map(|s| s.entry).collect::<Vec<_>>(), vec![1, 2, 3, 4]);
        let first = tl.stops[0];
        assert_eq!((first.start, first.arrive), (6.0, 6.0));
        for (s, e) in tl.stops.iter().zip(&entries).skip(1) {
            assert!((s.start - e.start).abs() < 1e-9, "{s:?}");
            assert!((s.arrive - on_frame(e.start + GLIDE, 30.0)).abs() < 1e-9, "{s:?}");
            assert!((s.start * 30.0 - (s.start * 30.0).round()).abs() < 1e-6 && (s.arrive * 30.0 - (s.arrive * 30.0).round()).abs() < 1e-6);
            assert!((s.x - tl.scale.x(e.unix.unwrap())).abs() < 1e-9);
        }
        assert!(tl.stops.windows(2).all(|w| w[1].x > w[0].x), "later dates further right");
        // mid-2021 sits in the middle of the three years' second third
        let mid = tl.scale.x(unix_of(2021, 7, 2).unwrap());
        let third = (tl.scale.right - tl.scale.left) / 3.0;
        assert!((mid - (tl.scale.left + 1.5 * third)).abs() < 2.0, "{mid}");
        // bands from the entries' events, the year and event picked out run by run
        assert_eq!(tl.bands.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(), vec!["Expo 2020", "GITEX 2022"]);
        assert!(tl.bands.iter().all(|b| b.x1 - b.x0 >= MIN_BAND - 1e-9));
        let runs: Vec<(Option<i64>, Option<usize>, f64)> = tl.runs.iter().map(|r| (r.year, r.band, r.start)).collect();
        assert_eq!(runs, vec![(Some(2020), None, 6.0), (Some(2021), Some(0), 14.0), (Some(2022), Some(0), 22.5), (Some(2022), Some(1), 31.0)]);
        assert_eq!(tl.runs.last().unwrap().end, 48.0, "the undated entry stays in the run before it");
        // the plan's events instead: bands by date, clipped to the years shown
        let ev =
            events_from_plan(&json!([{"name": "Expo 2020", "from": "2021-10-01", "to": "2022-03-31"}, {"name": "Old", "from": "1990", "to": "1991"}])).unwrap();
        let tl = plan_timeline(&entries, &ev, 6.0, 48.0, (1920.0, 1080.0), 30.0).unwrap().unwrap();
        assert_eq!(tl.bands.len(), 1);
        assert_eq!(tl.runs.iter().map(|r| r.band).collect::<Vec<_>>(), vec![None, Some(0), Some(0), None]);
        // nothing dated: no timeline; hostile dates: an error
        let undated = vec![TimelineEntry { number: 1, start: 0.0, unix: None, event: None }];
        assert!(plan_timeline(&undated, &[], 0.0, 10.0, (1920.0, 1080.0), 30.0).unwrap().is_none());
        for u in [i64::MIN / 2, i64::MAX / 2, -5_000_000_000_000] {
            let e = vec![te(1, 0.0, (2020, 1, 1), None), TimelineEntry { number: 2, start: 5.0, unix: Some(u), event: None }];
            assert!(plan_timeline(&e, &[], 0.0, 10.0, (1920.0, 1080.0), 30.0).is_err(), "{u}");
        }
        let wide = vec![te(1, 0.0, (1950, 1, 1), None), te(2, 5.0, (2020, 1, 1), None)];
        assert!(plan_timeline(&wide, &[], 0.0, 10.0, (1920.0, 1080.0), 30.0).is_err());
        // hostile times and frames give a timeline that is still finite
        let tl = plan_timeline(&entries, &[], f64::NAN, f64::INFINITY, (f64::NAN, -3.0), f64::NAN).unwrap().unwrap();
        assert!(tl.stops.iter().all(|s| s.x.is_finite() && s.start.is_finite() && s.arrive.is_finite()));
    }
}
