//! Where each item of a Google Photos album was taken, read from the Google Photos web page.
//!
//! Google strips the location from what its API and picker hand out (0 of 752 downloaded clips
//! had a location atom): the only place left is the Info panel of the Google Photos page, where
//! the owner sees each item's date, time and time zone, file name, size and place ("Dubai World
//! Trade Centre", "Busan, South Korea", sometimes marked "Estimated location"). This module
//! reads that panel, item by item, the way a person would:
//!
//! 1. the album's page (`https://photos.google.com/album/<id>`; found once on the albums page
//!    by its exact title and remembered by the caller);
//! 2. its first item opened (or, resuming, the item after the last one read, opened by its
//!    address), the Info panel opened with its keyboard shortcut (`i`);
//! 3. for each item: the panel's text read by the local vision model ([`PanelEyes`]), parsed by
//!    [`parse_panel`] into a [`Location`], then the next item (Right arrow; the "View next
//!    photo" button when the arrow does not move on), until the address stops changing.
//!
//! **Nothing in the library is changed.** The keys pressed are the Right arrow and `i` (show
//! the Info panel), and only after the address says an item is open; nothing is typed. The only
//! clicks are an album's card on the albums page, the album's first thumbnail and the "next"
//! arrow beside an item — each checked to lie where such a thing sits (below the top bar, the
//! arrow on the right half at mid-height), so a misplaced box can never reach the toolbar's
//! favourite, delete or share buttons. Addresses come only from `photos.google.com`.
//!
//! Bounded: at most [`Request::max_items`] items, [`Request::per_item`] for each (an item that
//! cannot be read is recorded with its `error` and skipped), [`Request::budget`] in all; a run
//! can be resumed from the items it already read ([`Request::done`]).

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::ScreenError;
use crate::capture::Capture;
use crate::ground::Element;

/// Where one item of an album was taken, as its Info panel shows it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Location {
    /// The item's file name, as Google Photos shows it ("VID_20211021_111656.mp4").
    pub file_name: String,
    /// The day it was taken, `YYYY-MM-DD` (as written when it could not be read as a date).
    #[serde(default)]
    pub date: Option<String>,
    /// The time it was taken, `HH:MM` (24-hour), in its own time zone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time: Option<String>,
    /// Its time zone, `+04:00`.
    #[serde(default)]
    pub gmt_offset: Option<String>,
    /// The place ("Dubai World Trade Centre", "Busan, South Korea"); `None` when the panel
    /// shows none.
    #[serde(default)]
    pub place: Option<String>,
    /// The line under the place, when there is one ("Dubai - United Arab Emirates").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub area: Option<String>,
    /// Google marked the place "Estimated location" (from location history, not the camera).
    #[serde(default)]
    pub estimated: bool,
    /// Its size in pixels as shown ("1920 × 1080").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    /// Its id in the page's address (`…/photo/<id>`), which a resumed run skips by.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Why it could not be read (its other fields are then empty).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Location {
    /// The place, when the panel showed one.
    pub fn place(&self) -> Option<&str> {
        self.place.as_deref().map(str::trim).filter(|p| !p.is_empty())
    }
}

/// Longest line, file name and place taken from a panel (they are model output: hostile).
const MAX_LINE: usize = 300;
const MAX_FILE_NAME: usize = 255;
const MAX_PLACE: usize = 160;
/// Most lines of one panel looked at.
const MAX_LINES: usize = 200;

/// File extensions of photos and videos (a token ending in one is a file name).
const MEDIA_EXT: [&str; 24] = [
    "mp4", "mov", "m4v", "3gp", "mkv", "webm", "avi", "mts", "m2ts", "wmv", "mpg", "jpg", "jpeg", "heic", "heif", "png", "gif", "webp", "dng", "raw", "arw",
    "cr2", "nef", "tif",
];

/// Lines of the panel that are its furniture, not an item's details.
const FURNITURE: [&str; 22] = [
    "info",
    "details",
    "add a description",
    "add a caption",
    "description",
    "other",
    "people",
    "albums",
    "album",
    "done",
    "close",
    "edit",
    "add location",
    "edit location",
    "remove location",
    "location",
    "show more",
    "show less",
    "more options",
    "open info",
    "close info",
    "map",
];

/// Starts of lines that are not a place.
const NOT_PLACE_STARTS: [&str; 17] = [
    "this item",
    "learn more",
    "storage",
    "uploaded",
    "backed up",
    "not backed up",
    "map data",
    "©",
    "shared",
    "saved",
    "storage saver",
    "original quality",
    "express",
    "added by",
    "edited",
    "keyboard",
    "google photos",
];

/// Camera makers and model words: a line whose first word is one names the camera, not a place.
const CAMERAS: [&str; 36] = [
    "apple",
    "iphone",
    "ipad",
    "google",
    "pixel",
    "nexus",
    "samsung",
    "galaxy",
    "dji",
    "osmo",
    "mavic",
    "gopro",
    "canon",
    "nikon",
    "sony",
    "xperia",
    "xiaomi",
    "redmi",
    "poco",
    "oneplus",
    "huawei",
    "oppo",
    "vivo",
    "realme",
    "motorola",
    "nokia",
    "fujifilm",
    "panasonic",
    "olympus",
    "insta360",
    "leica",
    "ricoh",
    "hasselblad",
    "asus",
    "lenovo",
    "lge",
];

/// One panel line made plain: invisible characters dropped, whitespace collapsed, a list
/// bullet or markdown emphasis dropped, at most [`MAX_LINE`] characters.
fn clean_line(line: &str) -> String {
    let t: String = line.chars().filter(|c| !matches!(c, '\u{200B}'..='\u{200F}' | '\u{FEFF}') && !c.is_control()).collect();
    let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut t = t.as_str();
    for p in ["- ", "* ", "• ", "· "] {
        t = t.strip_prefix(p).unwrap_or(t);
    }
    t.trim_matches(|c| c == '*' || c == '`').trim().chars().take(MAX_LINE).collect()
}

/// The month a word names ("Oct", "October", "Sept"), 1–12.
fn month_of(word: &str) -> Option<u32> {
    let w = word.to_ascii_lowercase();
    if w.len() < 3 {
        return None;
    }
    let full = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"];
    full.iter().position(|f| f.starts_with(&w)).map(|i| i as u32 + 1)
}

/// The date a line shows ("Oct 21, 2021", "21 Oct 2021", "Thu, Oct 21, 2021") as `YYYY-MM-DD`.
pub fn parse_date(line: &str) -> Option<String> {
    let words: Vec<&str> = line.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).take(40).collect();
    let month = words.iter().find_map(|w| month_of(w))?;
    let year = words.iter().filter(|w| w.len() == 4).filter_map(|w| w.parse::<u32>().ok()).find(|y| (1900..=2100).contains(y))?;
    let day = words.iter().filter(|w| w.len() <= 2).filter_map(|w| w.parse::<u32>().ok()).find(|d| (1..=31).contains(d))?;
    Some(format!("{year:04}-{month:02}-{day:02}"))
}

/// The clock time and time zone a line shows ("Thu, 11:16 AM GMT+04:00" → `11:16`, `+04:00`;
/// "7:42 PM" → `19:42`).
pub fn parse_time(line: &str) -> Option<(String, Option<String>)> {
    let zone_at = line.find("GMT").or_else(|| line.find("UTC"));
    for (i, _) in line.match_indices(':') {
        // the minutes of a time zone ("GMT+04:00") are not the time
        if zone_at.is_some_and(|z| i > z && i <= z + 8) {
            continue;
        }
        let before = line.get(..i).unwrap_or("");
        let hours: String = before.chars().rev().take_while(char::is_ascii_digit).collect::<Vec<_>>().into_iter().rev().collect();
        let after = line.get(i + 1..).unwrap_or("");
        let minutes: String = after.chars().take_while(char::is_ascii_digit).collect();
        if hours.is_empty() || hours.len() > 2 || minutes.len() != 2 {
            continue;
        }
        let (Ok(mut h), Ok(m)) = (hours.parse::<u32>(), minutes.parse::<u32>()) else { continue };
        if h > 23 || m > 59 {
            continue;
        }
        let rest = after.get(2..).unwrap_or("").trim_start().to_ascii_uppercase();
        let pm = rest.starts_with("PM") || rest.starts_with("P.M");
        let am = rest.starts_with("AM") || rest.starts_with("A.M");
        if (am || pm) && !(1..=12).contains(&h) {
            continue;
        }
        if pm && h < 12 {
            h += 12;
        }
        if am && h == 12 {
            h = 0;
        }
        return Some((format!("{h:02}:{m:02}"), parse_offset(line)));
    }
    None
}

/// The time zone in a line: "GMT+04:00", "GMT+4", "UTC-05:30", "GMT+0530", "GMT" → `+04:00`,
/// `+04:00`, `-05:30`, `+05:30`, `+00:00`.
pub fn parse_offset(line: &str) -> Option<String> {
    let at = line.find("GMT").or_else(|| line.find("UTC"))?;
    let rest = line.get(at + 3..)?.trim_start();
    let mut chars = rest.chars();
    let sign = match chars.next() {
        Some('+') => '+',
        Some('-' | '−') => '-',
        _ => return Some("+00:00".into()),
    };
    let body: String = chars.take_while(|c| c.is_ascii_digit() || *c == ':').take(6).collect();
    let (h, m) = match body.split_once(':') {
        Some((h, m)) => (h.parse::<u32>().ok()?, m.get(..m.len().min(2))?.parse::<u32>().ok()?),
        None if body.len() == 4 => (body.get(..2)?.parse().ok()?, body.get(2..)?.parse().ok()?),
        None => (body.parse::<u32>().ok()?, 0),
    };
    (h <= 14 && m <= 59).then(|| format!("{sign}{h:02}:{m:02}"))
}

/// A token of `line` that ends in a photo or video extension.
fn media_token(line: &str) -> Option<String> {
    line.split_whitespace().map(|t| t.trim_matches(|c: char| matches!(c, ',' | ';' | '"' | '\'' | '(' | ')'))).find_map(|t| {
        let (stem, ext) = t.rsplit_once('.')?;
        (!stem.is_empty() && MEDIA_EXT.contains(&ext.to_ascii_lowercase().as_str())).then(|| t.to_string())
    })
}

/// "1920 × 1080", "4032 x 3024", "12MP 4000 × 3000": the pixel size.
fn resolution_of(line: &str) -> Option<String> {
    let t: Vec<char> = line.chars().map(|c| if matches!(c, '×' | 'X' | '*') { 'x' } else { c }).collect();
    let digits = |from: usize| -> (String, usize) {
        let s: String = t.get(from..).unwrap_or(&[]).iter().take_while(|c| c.is_ascii_digit()).take(7).collect();
        let n = s.len();
        (s, from + n)
    };
    let mut i = 0;
    while i < t.len() {
        let (a, end) = digits(i);
        if a.is_empty() {
            i += 1;
            continue;
        }
        let mut j = end;
        while t.get(j) == Some(&' ') {
            j += 1;
        }
        if t.get(j) == Some(&'x') {
            j += 1;
            while t.get(j) == Some(&' ') {
                j += 1;
            }
            let (b, _) = digits(j);
            if let (Ok(w), Ok(h)) = (a.parse::<u32>(), b.parse::<u32>())
                && (16..=100_000).contains(&w)
                && (16..=100_000).contains(&h)
            {
                return Some(format!("{w} × {h}"));
            }
        }
        i = end.max(i + 1);
    }
    None
}

/// A detail of the file or camera, not a place: "12MP", "1080p", "4K", "12.3 MB", "0:42",
/// "ƒ/1.8 1/100 4.38mm ISO100", a camera's name ("Google Pixel 6", "DJI FC3170").
fn is_detail(line: &str) -> bool {
    let l = line.to_lowercase();
    let compact: String = l.chars().filter(|c| !c.is_whitespace()).collect();
    let unit = |u: &str| compact.strip_suffix(u).is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit() || c == '.' || c == ','));
    if ["mp", "mb", "gb", "kb", "p", "k", "fps", "mbps", "s"].iter().any(|u| unit(u)) {
        return true;
    }
    // a duration
    if compact.contains(':') && compact.chars().all(|c| c.is_ascii_digit() || c == ':') {
        return true;
    }
    // exposure
    let words: Vec<&str> = l.split_whitespace().collect();
    if l.contains('ƒ')
        || l.contains("f/")
        || words.iter().any(|w| w.starts_with("iso") && w.get(3..).is_some_and(|n| n.is_empty() || n.chars().all(|c| c.is_ascii_digit())))
        || words.iter().any(|w| w.starts_with("1/") || (w.ends_with("mm") && w.len() > 2 && w.starts_with(|c: char| c.is_ascii_digit())))
    {
        return true;
    }
    // a camera: its maker or model first ("Google Pixel 6", "SM-G991B", "ILCE-7M3", "FC3170")
    let first: String = words.first().map(|w| w.chars().filter(|c| c.is_alphanumeric() || *c == '-').collect()).unwrap_or_default();
    let model = |p: &str| first.strip_prefix(p).is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit() || c == '-'));
    CAMERAS.contains(&first.as_str()) || model("sm") || model("ilce") || model("fc") || model("hero")
}

/// Furniture of the panel or Google's notes (never a place).
fn is_furniture(line: &str) -> bool {
    let l = line.to_lowercase();
    let l = l.trim_end_matches([':', '.']).trim();
    // an album the item is in: "415 items · Shared · Sep 19, 2012 – …"
    let album_summary = l
        .split_whitespace()
        .collect::<Vec<_>>()
        .windows(2)
        .any(|w| w.first().is_some_and(|n| n.chars().all(|c| c.is_ascii_digit() || c == ',')) && w.get(1).is_some_and(|i| *i == "items" || *i == "item"));
    l.is_empty() || FURNITURE.contains(&l) || NOT_PLACE_STARTS.iter().any(|s| l.starts_with(s)) || l.starts_with("```") || album_summary
}

/// Lines with a long file name wrapped onto the next ("VID_20200908_223605_HSR_120.mp" and
/// "4") joined again: a short run of letters and digits that completes a photo or video
/// extension.
fn join_wrapped(lines: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    for line in lines {
        let short = (1..=5).contains(&line.chars().count()) && line.chars().all(|c| c.is_ascii_alphanumeric());
        if short
            && let Some(prev) = out.last_mut()
            && media_token(prev).is_none()
            && !prev.contains(' ')
            && media_token(&format!("{prev}{line}")).is_some()
        {
            prev.push_str(&line);
            continue;
        }
        out.push(line);
    }
    out
}

/// Read one Info panel's text — as the page shows it, or as the vision model transcribes it (a
/// line may start "Location:") — into a [`Location`] (`id` and `error` unset). `None` when no
/// file name is in it: not an item's panel, or not open.
///
/// The panel lists, after any description: the date; the day, time and GMT offset; the file
/// name, its size and pixel size; the camera and exposure; the place (and the area under it),
/// sometimes marked "Estimated location". Lines are told apart by their shape (a date, a time,
/// a file extension, "1920 × 1080", "ƒ/1.8", a camera maker), and by their order: the line after
/// the time is the file name, the first other line after the file the place.
pub fn parse_panel(text: &str) -> Option<Location> {
    let all = join_wrapped(text.lines().take(MAX_LINES).map(clean_line).filter(|l| !l.is_empty()).collect());
    // a description (anything) comes before the Details heading: start after it when it shows
    let start = all.iter().position(|l| l.trim_end_matches(':').eq_ignore_ascii_case("details")).map_or(0, |i| i + 1);
    let mut out = Location::default();
    let mut explicit: Option<String> = None;
    let mut file: Option<String> = None;
    // 0: before the date and time, 1: after them (the file name is next), 2: after the file name
    let mut stage = 0;
    for line in all.iter().skip(start) {
        let lower = line.to_lowercase();
        if let Some(k) = ["location:", "place:"].iter().find(|k| lower.starts_with(*k)) {
            let p = line.get(k.len()..).unwrap_or("").trim().trim_matches(|c| c == '"' || c == '*' || c == '`').trim();
            if !p.is_empty() && !matches!(p.to_lowercase().as_str(), "none" | "null" | "-" | "n/a" | "unknown" | "no location") {
                explicit = Some(p.chars().take(MAX_PLACE).collect());
            }
            continue;
        }
        if lower.contains("estimated") {
            out.estimated = true;
            continue;
        }
        if let Some(k) = ["file name:", "filename:", "file:"].iter().find(|k| lower.starts_with(*k)) {
            let f = line.get(k.len()..).unwrap_or("").trim();
            if !f.is_empty() && file.is_none() {
                file = Some(f.to_string());
                stage = 2;
            }
            continue;
        }
        if is_furniture(line) {
            continue;
        }
        if stage < 2 {
            // the date nearest the file name wins (an album's dates come before it)
            if let Some(d) = parse_date(line) {
                out.date = Some(d);
                if let Some((t, o)) = parse_time(line) {
                    out.time = Some(t);
                    out.gmt_offset = o;
                }
                stage = 1;
                continue;
            }
            if let Some((t, o)) = parse_time(line) {
                // a time with its zone wins over one read before
                if out.time.is_none() || o.is_some() {
                    out.time = Some(t);
                    out.gmt_offset = o;
                }
                stage = 1;
                continue;
            }
        }
        if file.is_none() {
            if let Some(tok) = media_token(line) {
                file = Some(tok);
                stage = 2;
                continue;
            }
            // the line after the date and time is the file name, whatever its shape
            if stage == 1 && resolution_of(line).is_none() && !is_detail(line) && parse_date(line).is_none() {
                file = Some(line.clone());
                stage = 2;
            }
            continue;
        }
        if let Some(r) = resolution_of(line) {
            out.resolution.get_or_insert(r);
            continue;
        }
        if is_detail(line) || parse_date(line).is_some() || !line.chars().any(char::is_alphabetic) {
            continue;
        }
        if out.place.is_none() {
            out.place = Some(line.chars().take(MAX_PLACE).collect());
        } else if out.area.is_none() {
            out.area = Some(line.chars().take(MAX_PLACE).collect());
        }
    }
    let file: String = file?.trim().chars().take(MAX_FILE_NAME).collect();
    if file.is_empty() || file.chars().any(char::is_control) {
        return None;
    }
    out.file_name = file;
    // a place the model took from Google's notes ("Shared by saranya mohesh", "Storage saver.",
    // live) is none
    let explicit = explicit.filter(|p| {
        let lower = p.to_lowercase();
        !is_furniture(p) && !all.iter().any(|l| is_furniture(l) && l.to_lowercase().contains(&lower))
    });
    if let Some(p) = explicit {
        // the model said which line is the place: trust it over the layout
        if out.place.as_deref() != Some(p.as_str()) {
            out.area = None;
        }
        out.place = Some(p);
    }
    Some(out)
}

// ---------------------------------------------------------------------------------------------
// Addresses

const PHOTOS: &str = "https://photos.google.com/";
const MAX_URL: usize = 2048;

/// The path of a Google Photos page after `https://photos.google.com/` and an account prefix
/// (`u/1/`), with that prefix; `None` for any other address.
fn photos_path(url: &str) -> Option<(&str, &str)> {
    if url.len() > MAX_URL || url.chars().any(|c| c.is_whitespace() || c.is_control() || c == '"' || c == '\\') {
        return None;
    }
    let rest = url.strip_prefix(PHOTOS)?;
    if let Some(after) = rest.strip_prefix("u/") {
        let digits = after.chars().take_while(char::is_ascii_digit).count();
        let path = after.get(digits..)?.strip_prefix('/')?;
        if digits == 0 {
            return None;
        }
        let prefix = url.get(..url.len() - path.len())?;
        return Some((prefix, path));
    }
    Some((url.get(..PHOTOS.len())?, rest))
}

fn is_id(s: &str) -> bool {
    !s.is_empty() && s.len() <= 200 && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// The album page an address is on or under (`https://photos.google.com/album/<id>`, an
/// account's `…/u/1/album/<id>`, a shared album's `…/share/<id>`), without the item or query.
pub fn album_url(url: &str) -> Option<String> {
    let (prefix, path) = photos_path(url)?;
    let path = path.split(['?', '#']).next().unwrap_or("");
    let mut parts = path.split('/');
    let kind = parts.next()?;
    let id = parts.next()?;
    (matches!(kind, "album" | "share") && is_id(id)).then(|| format!("{prefix}{kind}/{id}"))
}

/// The id of the item an address shows (`…/album/<a>/photo/<id>`, `…/photo/<id>`).
pub fn item_id(url: &str) -> Option<String> {
    let (_, path) = photos_path(url)?;
    let path = path.split(['?', '#']).next().unwrap_or("");
    let parts: Vec<&str> = path.split('/').collect();
    let at = parts.iter().position(|p| *p == "photo")?;
    let id = parts.get(at + 1)?;
    is_id(id).then(|| id.to_string())
}

/// The address of item `id` inside the album at `album`.
pub fn item_url(album: &str, id: &str) -> Option<String> {
    let album = album_url(album)?;
    is_id(id).then(|| format!("{album}/photo/{id}"))
}

/// Google Photos' albums page.
pub const ALBUMS_PAGE: &str = "https://photos.google.com/albums";

/// Whether `url` is a Google Photos page (the only addresses the reader opens).
pub fn is_photos_page(url: &str) -> bool {
    photos_path(url).is_some()
}

// ---------------------------------------------------------------------------------------------
// The reader

/// The browser the reader drives: one tab it opened, its address, keys, clicks and captures.
/// The Chrome implementation is [`ChromeViewer`]; tests drive fakes.
pub trait Viewer {
    /// Open `url` (a Google Photos page) in the reader's own tab, opening the tab the first
    /// time, and bring it to the front.
    fn open(&mut self, url: &str) -> Result<(), ScreenError>;
    /// The reader's tab's address now (`None`: the tab is gone).
    fn url(&mut self) -> Result<Option<String>, ScreenError>;
    /// The Right arrow (next item).
    fn key_right(&mut self) -> Result<(), ScreenError>;
    /// `i`: show or hide the Info panel (only pressed while an item is open).
    fn key_info(&mut self) -> Result<(), ScreenError>;
    /// A capture of the window. `panel`: only its right part, where the Info panel is, at a
    /// resolution that keeps its text readable — for reading, never for clicking.
    fn capture(&mut self, panel: bool) -> Result<Capture, ScreenError>;
    /// Click a pixel of `cap` (a whole-window capture).
    fn click(&mut self, cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError>;
    /// Wheel-scroll the page (negative: up).
    fn scroll(&mut self, lines: i32) -> Result<(), ScreenError>;
    /// Wheel-scroll the Info panel to its end, where the place is: an item in albums and with
    /// people has them above its Details, which push the place below the window.
    fn scroll_panel(&mut self) -> Result<(), ScreenError>;
    fn sleep(&mut self, ms: u64);
    /// Close the reader's tab (only it, and only while it shows Google Photos).
    fn close(&mut self) -> Result<(), ScreenError>;
}

/// The reader's eyes (the local vision model): text read in a capture, and an element found
/// by description.
pub trait PanelEyes {
    /// What the model reads in `png` when asked `ask`.
    fn read(&self, png: &[u8], ask: &str) -> Result<String, ScreenError>;
    fn find(&self, png: &[u8], target: &str) -> Result<Option<Element>, ScreenError>;
}

/// What the vision model is asked about a panel capture. The `Location:` line marks the place
/// (the line beside the map-pin icon), which the layout alone cannot always tell from a
/// camera's name.
///
/// It offers no way out ("write NONE if there is no panel"): live, qwen2.5vl took that way out
/// for panels it read perfectly a moment before. Whether a panel is open at all is told from what
/// it reads ([`is_panel_text`]). Nor does it mention "Estimated location": asked to copy those
/// words when shown, the model wrote them under every place; [`ESTIMATED_ASK`] asks about the
/// icon that marks an estimated place instead.
pub const PANEL_ASK: &str = "This is the right part of a Google Photos page; its Info panel is on the right. Transcribe the text of the panel's Details section exactly as shown, one line per line, top to bottom: the date, the day and time with its GMT offset, the file name, the picture size, the camera, the place. Write the place's line — the words right of the map-pin icon — as `Location: <those words>`; never take names from a map; if no map-pin icon is shown, write no Location line. Write nothing else.";
/// Whether the place is estimated: Google marks an estimated place with a small circled "i"
/// after it ("Estimated location" on hover). Asked only when a place was read; live, the model
/// answered it right where the transcription had invented the words.
pub const ESTIMATED_ASK: &str = "In this Google Photos Info panel, look at the place line beside the map-pin icon near the bottom. Is there a small circled \"i\" icon right after the place name? Answer yes or no.";
/// An album page's title, read to check the album opened is the one asked for.
pub const ALBUM_TITLE_ASK: &str =
    "This is a Google Photos album page. What is the album's title, the large text above its photos? Answer with the title only, exactly as written.";
/// The "next item" arrow, asked for when the Right arrow did not move on.
pub const NEXT_ASK: &str =
    "the 'View next photo' button: a right-pointing arrow (chevron) at the right edge of the photo, vertically centred, left of the Info panel";
/// An album's thumbnail grid's first item.
pub const FIRST_ITEM_ASK: &str = "the first photo or video thumbnail of the album's grid (the top-left one, below the album's title)";

/// The Info panel's width in points (CSS pixels at 100 % zoom) and a margin: the "next"
/// arrow is never clicked inside it.
const INFO_PANEL_POINTS: f32 = 360.0;

/// What to read.
#[derive(Clone, Debug)]
pub struct Request {
    /// The album's page (`https://photos.google.com/album/<id>`).
    pub album_url: String,
    /// Read at most this many items.
    pub max_items: usize,
    /// The longest one item may take to read; past it the item is recorded with an error and
    /// skipped.
    pub per_item: Duration,
    /// The longest the whole run may take.
    pub budget: Duration,
    /// Items read before (a resumed run): skipped by id, and the run starts after the last.
    pub done: Vec<Location>,
}

/// The most items one run reads.
pub const MAX_ITEMS: usize = 5000;

impl Request {
    pub fn new(album_url: &str) -> Self {
        Self { album_url: album_url.to_string(), max_items: 1000, per_item: Duration::from_secs(90), budget: Duration::from_secs(3 * 3600), done: Vec::new() }
    }
}

/// What a run read.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    /// Every item read, in album order (a resumed run's earlier items first).
    pub locations: Vec<Location>,
    /// Items read in this run.
    pub read: usize,
    /// Items read in this run whose panel showed no place.
    pub without_place: usize,
    /// Items that could not be read in this run (their `error` says why).
    pub failed: usize,
    /// Why the run stopped before the album's end (`None`: it read to the end).
    pub stopped: Option<String>,
}

/// Errors that end a run (a lost grant, the model gone): every other one only skips an item.
fn fatal(e: &ScreenError) -> bool {
    matches!(e, ScreenError::Model(..) | ScreenError::Capture(..) | ScreenError::Input(..) | ScreenError::NotInFront(_) | ScreenError::Unsupported(_))
}

/// Wait up to `limit` for the tab's address to differ from `from`; the new address.
fn wait_moved<V: Viewer + ?Sized>(v: &mut V, from: &str, limit: Duration) -> Result<Option<String>, ScreenError> {
    // polls of [`POLL_MS`], so the wait is the viewer's own sleeping (a fake's is none)
    let polls = (limit.as_millis() / u128::from(POLL_MS)).clamp(1, 2000);
    for _ in 0..=polls {
        match v.url()? {
            Some(u) if u != from => return Ok(Some(u)),
            None => return Err(ScreenError::Other("the Google Photos tab was closed".into())),
            _ => {}
        }
        v.sleep(POLL_MS);
    }
    Ok(None)
}

/// How often the tab's address is looked at while waiting for it to change.
const POLL_MS: u64 = 300;

/// A box is below the page's top bar (where an album card or a thumbnail sits, never the
/// toolbar's buttons) and of a card's size.
fn in_grid(e: &Element, w: u32, h: u32) -> bool {
    let (_, cy) = e.center();
    let frame = u64::from(w) * u64::from(h);
    cy > h / 6 && cy < h.saturating_sub(4) && e.area() >= frame / 400 && e.area() <= frame / 3
}

/// A box is where an item's "next" arrow sits: the right half but left of the Info panel,
/// mid-height, button-sized.
fn is_next_arrow(e: &Element, cap: &Capture) -> bool {
    let (w, h) = (cap.png_w, cap.png_h);
    let (cx, cy) = e.center();
    let panel = (INFO_PANEL_POINTS * cap.px_per_point).max(0.0) as u32;
    cx > w / 2 && cx < w.saturating_sub(panel) && cy > h / 4 && cy < h * 3 / 4 && e.area() > 0 && e.area() <= u64::from(w) * u64::from(h) / 100
}

/// One look at the panel: what the model read, and the item's place when it read as one
/// (with `estimated` asked about when there is a place).
fn look<V: Viewer + ?Sized, E: PanelEyes + ?Sized>(v: &mut V, eyes: &E) -> Result<(String, Option<Location>), ScreenError> {
    v.scroll_panel()?;
    v.sleep(500);
    let cap = v.capture(true)?;
    let text = match eyes.read(&cap.png, PANEL_ASK) {
        Ok(t) => t,
        Err(e) if fatal(&e) => return Err(e),
        Err(e) => return Ok((e.to_string(), None)),
    };
    let Some(mut l) = parse_panel(&text) else { return Ok((text, None)) };
    if !l.estimated && l.place().is_some() {
        match eyes.read(&cap.png, ESTIMATED_ASK) {
            Ok(a) => l.estimated = a.trim().to_ascii_lowercase().starts_with("yes"),
            Err(e) if fatal(&e) => return Err(e),
            Err(_) => {}
        }
    }
    Ok((text, Some(l)))
}

/// Read the item open now, at most twice (the panel can lag behind the address); `prev` is
/// the file name read for the item before (a panel still showing it is read again).
fn read_item<V: Viewer + ?Sized, E: PanelEyes + ?Sized>(v: &mut V, eyes: &E, prev: Option<&str>, deadline: Instant) -> Result<Location, ScreenError> {
    let mut last_err = String::from("the Info panel showed no file name");
    for attempt in 0..2 {
        if attempt > 0 {
            if Instant::now() >= deadline {
                last_err = format!("{last_err} (out of time for this item)");
                break;
            }
            v.sleep(1500);
        }
        let (text, read) = look(v, eyes)?;
        match read {
            Some(l) if attempt == 0 && prev.is_some_and(|p| p == l.file_name) => last_err = "the panel still showed the item before".into(),
            Some(l) => return Ok(l),
            None => last_err = format!("the Info panel could not be read (\"{}\")", text.lines().next().unwrap_or("").chars().take(80).collect::<String>()),
        }
    }
    Err(ScreenError::Other(last_err))
}

/// Open the album's first item from its page: its first thumbnail, clicked where a thumbnail
/// can be. The address must then be an item of the same album.
fn open_first<V: Viewer + ?Sized, E: PanelEyes + ?Sized>(v: &mut V, eyes: &E, album: &str, progress: &mut dyn FnMut(&str)) -> Result<String, ScreenError> {
    v.open(album)?;
    v.sleep(3000);
    for attempt in 0..3 {
        let cap = v.capture(false)?;
        let Some(tile) = eyes.find(&cap.png, FIRST_ITEM_ASK)?.filter(|e| in_grid(e, cap.png_w, cap.png_h)) else {
            progress(&format!("locations: no thumbnail on the album page yet (look {})", attempt + 1));
            v.sleep(2000);
            continue;
        };
        let (px, py) = tile.center();
        let from = v.url()?.unwrap_or_default();
        v.click(&cap, px, py)?;
        if let Some(u) = wait_moved(v, &from, Duration::from_secs(8))?
            && item_id(&u).is_some()
            && album_url(&u).as_deref() == Some(album)
        {
            return Ok(u);
        }
        // it opened something else: back to the album
        v.open(album)?;
        v.sleep(2500);
    }
    Err(ScreenError::Stalled("could not open the album's first item".into()))
}

/// Whether a transcription is of an Info panel at all (a Details heading, a date or a time),
/// even one whose item could not be read.
fn is_panel_text(text: &str) -> bool {
    text.lines().take(MAX_LINES).any(|l| l.to_lowercase().contains("details") || parse_date(l).is_some() || parse_time(l).is_some())
}

/// Make sure the Info panel shows: read it; when nothing like a panel shows, press `i` once and
/// read again. The item's panel when it read (`None`: the panel shows, this item did not read).
fn ensure_info<V: Viewer + ?Sized, E: PanelEyes + ?Sized>(v: &mut V, eyes: &E, progress: &mut dyn FnMut(&str)) -> Result<Option<Location>, ScreenError> {
    v.sleep(1500);
    let mut last = String::new();
    for pressed in [false, true] {
        if pressed {
            progress("locations: opening the Info panel (i)");
            v.key_info()?;
            v.sleep(2000);
        }
        for _ in 0..2 {
            let (text, read) = look(v, eyes)?;
            if let Some(l) = read {
                return Ok(Some(l));
            }
            if is_panel_text(&text) {
                return Ok(None);
            }
            last = text.lines().next().unwrap_or("").chars().take(80).collect();
            v.sleep(1000);
        }
    }
    Err(ScreenError::Stalled(format!(
        "the Info panel did not open (the model read \"{last}\"): click the ⓘ (Open info) button in the Google Photos tab once — it stays open from item to item — then run again with --resume"
    )))
}

/// Move to the next item: the Right arrow, else the "next" arrow beside the item. The new
/// address, or `None` at the album's end.
fn next_item<V: Viewer + ?Sized, E: PanelEyes + ?Sized>(v: &mut V, eyes: &E, from: &str, wait: Duration) -> Result<Option<String>, ScreenError> {
    v.key_right()?;
    if let Some(u) = wait_moved(v, from, wait)? {
        return Ok(Some(u));
    }
    let cap = v.capture(false)?;
    if let Some(arrow) = eyes.find(&cap.png, NEXT_ASK)?.filter(|e| is_next_arrow(e, &cap)) {
        let (px, py) = arrow.center();
        v.click(&cap, px, py)?;
        return wait_moved(v, from, wait);
    }
    Ok(None)
}

/// Read the places of an album's items (see the module docs). `on_item` hears every item as it
/// is read, so a caller can keep its progress on disk, and answers whether to go on (`false`:
/// stop, e.g. cancelled). The reader's tab is closed at the end.
pub fn read_album<V: Viewer + ?Sized, E: PanelEyes + ?Sized>(
    v: &mut V,
    eyes: &E,
    req: &Request,
    progress: &mut dyn FnMut(&str),
    on_item: &mut dyn FnMut(&Location) -> bool,
) -> Result<Report, ScreenError> {
    let r = read_inner(v, eyes, req, progress, on_item);
    if let Err(e) = v.close() {
        progress(&format!("locations: the Google Photos tab stays open ({e})"));
    }
    r
}

fn read_inner<V: Viewer + ?Sized, E: PanelEyes + ?Sized>(
    v: &mut V,
    eyes: &E,
    req: &Request,
    progress: &mut dyn FnMut(&str),
    on_item: &mut dyn FnMut(&Location) -> bool,
) -> Result<Report, ScreenError> {
    let album = album_url(&req.album_url)
        .ok_or_else(|| ScreenError::Other(format!("not a Google Photos album address: {}", req.album_url.chars().take(120).collect::<String>())))?;
    let t0 = Instant::now();
    let max = req.max_items.clamp(1, MAX_ITEMS);
    let per_item = req.per_item.clamp(Duration::from_secs(5), Duration::from_secs(600));
    let mut report = Report { locations: req.done.iter().take(MAX_ITEMS).cloned().collect(), ..Report::default() };
    let done_ids: std::collections::HashSet<String> = report.locations.iter().filter(|l| l.error.is_none()).filter_map(|l| l.id.clone()).collect();
    // where to start: after the last item read well (by its address; one that failed is read
    // again), else the first
    let resume_at = report.locations.iter().rev().filter(|l| l.error.is_none()).find_map(|l| l.id.clone()).and_then(|id| item_url(&album, &id));
    let mut url = match resume_at {
        Some(u) => {
            progress("locations: resuming after the last item read");
            v.open(&u)?;
            v.sleep(3000);
            let here = v.url()?.unwrap_or_default();
            if item_id(&here).is_none() {
                return Err(ScreenError::Stalled(format!(
                    "the last item read did not open ({}); run again without --resume",
                    here.chars().take(120).collect::<String>()
                )));
            }
            here
        }
        None => open_first(v, eyes, &album, progress)?,
    };
    let mut first = ensure_info(v, eyes, progress)?;
    let mut prev: Option<String> = None;
    let mut seen = 0usize;
    loop {
        if report.read >= max {
            report.stopped = Some(format!("read {max} item(s), the most asked for"));
            break;
        }
        if t0.elapsed() >= req.budget {
            report.stopped = Some(format!("out of time ({} min)", req.budget.as_secs() / 60));
            break;
        }
        seen += 1;
        if seen > MAX_ITEMS {
            report.stopped = Some(format!("more than {MAX_ITEMS} items"));
            break;
        }
        let id = item_id(&url);
        let prefetched = first.take();
        if !id.as_ref().is_some_and(|i| done_ids.contains(i)) {
            let read = match prefetched {
                Some(l) => Ok(l),
                None => read_item(v, eyes, prev.as_deref(), Instant::now() + per_item),
            };
            let loc = match read {
                Ok(mut l) => {
                    l.id = id.clone();
                    if l.place().is_none() {
                        report.without_place += 1;
                    }
                    prev = Some(l.file_name.clone());
                    l
                }
                Err(e) if fatal(&e) => return Err(e),
                Err(e) => {
                    report.failed += 1;
                    Location { id: id.clone(), error: Some(e.to_string()), ..Location::default() }
                }
            };
            let position = report.locations.iter().position(|l| l.error.is_some() && l.id.is_some() && l.id == loc.id).unwrap_or(report.locations.len());
            progress(&format!(
                "locations: {} {}{}",
                position + 1,
                if loc.file_name.is_empty() { "(unread)" } else { loc.file_name.as_str() },
                match (loc.place(), &loc.error) {
                    (_, Some(e)) => format!(": {e}"),
                    (Some(p), _) => format!(" — {p}{}", if loc.estimated { " (estimated)" } else { "" }),
                    (None, _) => " — no place".into(),
                }
            ));
            let go_on = on_item(&loc);
            // an item that failed before and reads now takes its place
            match report.locations.iter_mut().find(|l| l.error.is_some() && l.id.is_some() && l.id == loc.id) {
                Some(slot) => *slot = loc,
                None => report.locations.push(loc),
            }
            report.read += 1;
            if !go_on {
                report.stopped = Some("stopped".into());
                break;
            }
            if report.read >= max {
                report.stopped = Some(format!("read {max} item(s), the most asked for"));
                break;
            }
        }
        match next_item(v, eyes, &url, per_item.min(Duration::from_secs(10)))? {
            Some(u) if item_id(&u).is_some() => url = u,
            Some(u) => {
                report.stopped = Some(format!("the page left the album ({})", u.chars().take(120).collect::<String>()));
                break;
            }
            None => break,
        }
    }
    Ok(report)
}

// ---------------------------------------------------------------------------------------------
// Finding an album's address

fn norm_title(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

/// Pages of the albums page looked through for an album.
const ALBUM_PAGES: u32 = 8;

/// Find the address of the album titled exactly `title` on the albums page: its card is found
/// by the vision model (a page at a time, scrolling down), clicked, and the page it opens must
/// be an album whose title reads as `title`. The reader's tab stays open (on the album).
pub fn find_album<V: Viewer + ?Sized, E: PanelEyes + ?Sized>(v: &mut V, eyes: &E, title: &str, progress: &mut dyn FnMut(&str)) -> Result<String, ScreenError> {
    let title = title.trim();
    if title.is_empty() || title.chars().count() > 500 || title.chars().any(char::is_control) {
        return Err(ScreenError::Other("give the album's exact title".into()));
    }
    let want = norm_title(title);
    let ask = format!("the album card titled exactly \"{}\" (its cover picture, above its title)", title.replace('"', "'"));
    v.open(ALBUMS_PAGE)?;
    v.sleep(3500);
    for scrolled in 0..ALBUM_PAGES {
        let cap = v.capture(false)?;
        if let Some(card) = eyes.find(&cap.png, &ask)?.filter(|e| in_grid(e, cap.png_w, cap.png_h)) {
            let from = v.url()?.unwrap_or_default();
            let (px, py) = card.center();
            v.click(&cap, px, py)?;
            if let Some(u) = wait_moved(v, &from, Duration::from_secs(8))?
                && let Some(a) = album_url(&u)
            {
                v.sleep(2500);
                let page = v.capture(false)?;
                let shown = eyes.read(&page.png, ALBUM_TITLE_ASK)?;
                let shown = shown.lines().map(clean_line).find(|l| !l.is_empty()).unwrap_or_default();
                if norm_title(&shown) == want {
                    progress(&format!("locations: the album \"{title}\" is {a}"));
                    return Ok(a);
                }
                progress(&format!("locations: that card opened \"{}\", not \"{title}\"; looking on", shown.chars().take(80).collect::<String>()));
            }
            // back to where the albums page was
            v.open(ALBUMS_PAGE)?;
            v.sleep(2500);
            for _ in 0..scrolled {
                v.scroll(10)?;
                v.sleep(600);
            }
        }
        v.scroll(10)?;
        v.sleep(1500);
    }
    Err(ScreenError::Stalled(format!(
        "the album \"{title}\" was not found on the albums page: open it in Google Photos and give its address (--album-url https://photos.google.com/album/…)"
    )))
}

// ---------------------------------------------------------------------------------------------
// Chrome on macOS

/// The reader's Chrome tab: opened by the reader, named by its window and tab id, and touched
/// only while it shows Google Photos.
pub struct ChromeViewer {
    shell: Box<dyn crate::capture::Shell>,
    hands: crate::act::MacActor,
    tab: Option<crate::tabs::Tab>,
}

impl ChromeViewer {
    pub fn new() -> Self {
        Self::with_shells(Box::new(crate::capture::OsShell), Box::new(crate::capture::OsShell))
    }

    /// A viewer whose AppleScript (`shell`) and input (`hands`) go through these shells.
    pub fn with_shells(shell: Box<dyn crate::capture::Shell>, hands: Box<dyn crate::capture::Shell>) -> Self {
        Self { shell, hands: crate::act::MacActor::new(crate::tabs::CHROME, hands), tab: None }
    }

    fn chrome(&self, body: &str) -> Result<String, ScreenError> {
        self.shell.run("osascript", &["-e", &crate::tabs::timed(crate::tabs::CHROME, body)])
    }

    /// `body` on our tab (`w` its window, `k` its index, `t` the tab); `Ok(false)`: it is gone.
    fn on_tab(&self, body: &str) -> Result<bool, ScreenError> {
        let Some(tab) = &self.tab else { return Ok(false) };
        let out = self.chrome(&format!(
            "repeat with w in windows\nif (id of w as text) is \"{win}\" then\nrepeat with k from 1 to (count of tabs of w)\nset t to tab k of w\nif (id of t as text) is \"{id}\" then\n{body}\nreturn \"ok\"\nend if\nend repeat\nend if\nend repeat\nreturn \"gone\"",
            win = tab.window,
            id = tab.id
        ))?;
        Ok(out.trim() == "ok")
    }

    /// Our tab in front, still on Google Photos, before a key, a click or a capture: keys and
    /// captures go to the front tab.
    fn front(&mut self) -> Result<(), ScreenError> {
        if !self.on_tab("set active tab index of w to k\nset index of w to 1\nactivate")? {
            return Err(ScreenError::Other("the Google Photos tab was closed".into()));
        }
        let url = self.url()?.unwrap_or_default();
        if !is_photos_page(&url) {
            return Err(ScreenError::Other(format!("the reader's tab left Google Photos ({}); stopping", url.chars().take(120).collect::<String>())));
        }
        Ok(())
    }
}

impl Default for ChromeViewer {
    fn default() -> Self {
        Self::new()
    }
}

/// Rows of colour that make the map: a row is the map's when this share of its pixels (in the
/// panel's columns) are coloured, not grey.
const MAP_ROW_COLOURED: f64 = 0.15;
/// A pixel is coloured when its channels differ by more than this.
const COLOURED: u8 = 25;
/// The map is at least this tall (pixels) before it is cut off.
const MAP_MIN_ROWS: u32 = 40;

/// The panel without the map at its bottom: the map's place names ("Business Bay Metro
/// Station") were taken for the item's place ("Home") by the model (live). The panel is grey
/// (text and icons on a dark or light grey); the map is coloured. The coloured block that ends at
/// the bottom, in the right 40 % of `img` (inside the panel whatever the window's width), is cut
/// off when it is at least [`MAP_MIN_ROWS`] tall. The rows kept.
pub fn rows_above_map(img: &image::RgbaImage) -> u32 {
    let (w, h) = (img.width(), img.height());
    if w < 10 || h < MAP_MIN_ROWS * 2 {
        return h;
    }
    let x0 = w - (w * 2 / 5).max(1);
    let coloured = |y: u32| -> bool {
        let n = (x0..w)
            .filter(|&x| {
                let p = img.get_pixel(x, y).0;
                let (hi, lo) = (p[0].max(p[1]).max(p[2]), p[0].min(p[1]).min(p[2]));
                hi - lo > COLOURED
            })
            .count();
        n as f64 / f64::from(w - x0) >= MAP_ROW_COLOURED
    };
    // up from the bottom while the rows are coloured (gaps of a few rows: a grey road)
    let mut top = h;
    let mut gap = 0;
    let mut y = h;
    while y > 0 {
        y -= 1;
        if coloured(y) {
            top = y;
            gap = 0;
        } else {
            gap += 1;
            if gap > 12 {
                break;
            }
        }
    }
    if top < h && h - top >= MAP_MIN_ROWS && top > h / 5 { top } else { h }
}

/// The right `fraction` of a capture (where the Info panel is), without the map at its bottom
/// ([`rows_above_map`]), scaled so its long side is at most `max_side`; the PNG and its size.
pub fn crop_panel(png: &[u8], fraction: f64, max_side: u32) -> Result<(Vec<u8>, u32, u32), ScreenError> {
    let img = image::load_from_memory(png).map_err(|e| ScreenError::Parse(format!("the screenshot is not a decodable image: {e}")))?;
    let (w, h) = (img.width(), img.height());
    if w < 4 || h < 4 {
        return Err(ScreenError::Parse("the screenshot has no area".into()));
    }
    let f = if fraction.is_finite() { fraction.clamp(0.1, 1.0) } else { 0.4 };
    let cw = ((f64::from(w) * f).round() as u32).clamp(1, w);
    let mut crop = img.crop_imm(w - cw, 0, cw, h);
    let keep = rows_above_map(&crop.to_rgba8());
    if keep < h {
        crop = crop.crop_imm(0, 0, cw, keep.max(1));
    }
    let h = crop.height();
    let long = cw.max(h);
    let side = max_side.max(64);
    if long > side {
        let s = f64::from(side) / f64::from(long);
        crop =
            crop.resize_exact(((f64::from(cw) * s).round() as u32).max(1), ((f64::from(h) * s).round() as u32).max(1), image::imageops::FilterType::Triangle);
    }
    let (cw, ch) = (crop.width(), crop.height());
    let mut out = std::io::Cursor::new(Vec::new());
    crop.write_to(&mut out, image::ImageFormat::Png).map_err(|e| ScreenError::Parse(format!("cannot encode the panel: {e}")))?;
    Ok((out.into_inner(), cw, ch))
}

/// The share of the window's width the Info panel capture keeps (the panel is ~360 points of
/// a wide window; a little of the photo beside it does no harm).
const PANEL_FRACTION: f64 = 0.42;
/// The panel capture's long side, pixels.
const PANEL_SIDE: u32 = 1600;
/// The largest side of the full-resolution window capture the panel is cropped from.
const FULL_SIDE: u32 = 4096;

impl Viewer for ChromeViewer {
    fn open(&mut self, url: &str) -> Result<(), ScreenError> {
        if !is_photos_page(url) {
            return Err(ScreenError::Other(format!("refusing to open an address that is not Google Photos: {}", url.chars().take(120).collect::<String>())));
        }
        let u = crate::capture::escape_applescript(url)?;
        if self.tab.is_some() {
            if self.on_tab(&format!("set URL of t to \"{u}\""))? {
                return self.front();
            }
            self.tab = None;
        }
        let out = self.chrome(&format!(
            "if (count of windows) is 0 then make new window\nset w to front window\nset t to make new tab at end of tabs of w with properties {{URL:\"{u}\"}}\nset active tab index of w to (count of tabs of w)\nset index of w to 1\nactivate\nreturn (id of w as text) & (character id 9) & (id of t as text) & (character id 9) & (URL of t)"
        ))?;
        let tab = crate::tabs::parse_tabs(&out)
            .into_iter()
            .next()
            .ok_or_else(|| ScreenError::Other(format!("Chrome did not open a tab ({})", out.trim().chars().take(80).collect::<String>())))?;
        self.tab = Some(tab);
        Ok(())
    }

    fn url(&mut self) -> Result<Option<String>, ScreenError> {
        match &self.tab {
            Some(t) => crate::tabs::tab_url(self.shell.as_ref(), t),
            None => Ok(None),
        }
    }

    fn key_right(&mut self) -> Result<(), ScreenError> {
        self.front()?;
        // 124: the Right arrow
        self.hands.key_code(124)
    }

    fn key_info(&mut self) -> Result<(), ScreenError> {
        self.front()?;
        if item_id(&self.url()?.unwrap_or_default()).is_none() {
            return Err(ScreenError::Other("not pressing `i`: no item is open".into()));
        }
        // 34: the I key, Google Photos' Info shortcut (an item page has no field to type in)
        self.hands.key_code(34)
    }

    fn capture(&mut self, panel: bool) -> Result<Capture, ScreenError> {
        self.front()?;
        if !panel {
            return crate::capture::capture_window(self.shell.as_ref(), crate::tabs::CHROME);
        }
        let full = crate::capture::capture_window_max(self.shell.as_ref(), crate::tabs::CHROME, FULL_SIDE)?;
        let (png, png_w, png_h) = crop_panel(&full.png, PANEL_FRACTION, PANEL_SIDE)?;
        Ok(Capture { png, png_w, png_h, ..full })
    }

    fn click(&mut self, cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
        use crate::act::Act as _;
        self.front()?;
        self.hands.click_png(cap, px, py)
    }

    fn scroll(&mut self, lines: i32) -> Result<(), ScreenError> {
        use crate::act::Act as _;
        self.front()?;
        self.hands.scroll(lines)
    }

    fn scroll_panel(&mut self) -> Result<(), ScreenError> {
        self.front()?;
        // one wheel event over the panel's middle (one, so the map that scrolls up under the
        // pointer never takes a second one as a zoom)
        let r = crate::capture::window_bounds(self.shell.as_ref(), crate::tabs::CHROME)?;
        let half = |v: u32| i32::try_from(v / 2).unwrap_or(i32::MAX);
        let x = r.x.saturating_add(i32::try_from(r.w).unwrap_or(i32::MAX)).saturating_sub(INFO_PANEL_POINTS as i32 / 2);
        let y = r.y.saturating_add(half(r.h));
        self.hands.scroll_at(x, y, 40)
    }

    fn sleep(&mut self, ms: u64) {
        std::thread::sleep(Duration::from_millis(ms.min(10_000)));
    }

    fn close(&mut self) -> Result<(), ScreenError> {
        let Some(url) = self.url()? else {
            self.tab = None;
            return Ok(());
        };
        // only our tab, and only while it shows Google Photos (never the picker)
        if is_photos_page(&url) && !crate::tabs::is_picker_url(&url) {
            self.on_tab("close t")?;
        }
        self.tab = None;
        Ok(())
    }
}

impl PanelEyes for crate::ground::QwenGrounder {
    fn read(&self, png: &[u8], ask: &str) -> Result<String, ScreenError> {
        let text = self.read_text(png, ask)?;
        Ok(if text.trim().trim_matches('.').eq_ignore_ascii_case("none") { String::new() } else { text })
    }

    fn find(&self, png: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
        crate::ground::Ground::find(self, png, target)
    }
}

#[cfg(test)]
#[path = "locations_tests.rs"]
mod tests;
