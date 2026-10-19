//! What the Google Photos picker looks like: pure predicates over grounded elements, so
//! the agent's decisions are unit-testable without a browser.

use crate::ground::Element;

/// Description of the picker's search field for `find` — it searches the user's whole
/// library, which is the only "search Google Photos" any app can offer since the Library
/// API retired.
pub const SEARCH_FIELD: &str = "the search field at the top of the picker (an input box, may contain a magnifier icon or placeholder text like 'Search')";

/// What to type into the picker's search. When only videos are wanted the search asks for
/// them ("<query> videos", or just "videos"): Google's search then filters the grid to
/// videos, so a tile grounded one position off is still a video. A 7B model on a dense mixed
/// grid boxed about one video in five a tile too far right — onto a photo — in live tests.
/// `None`: nothing to search (photos allowed and no query).
pub fn search_text(query: Option<&str>, videos_only: bool) -> Option<String> {
    let q = query.map(str::trim).unwrap_or("");
    if !videos_only {
        return (!q.is_empty()).then(|| q.to_string());
    }
    if q.is_empty() {
        return Some("videos".into());
    }
    Some(if q.to_lowercase().contains("video") { q.to_string() } else { format!("{q} videos") })
}

/// The confirm button, in the order to try.
pub const CONFIRM_LABELS: [&str; 4] =
    ["the Done button (top right, blue/enabled when items are selected)", "the Add button", "the Insert button", "the Confirm button"];

/// Whether a label carries a duration badge (`0:30`, `12:04`, `1:02:33`) — the visual
/// signature of a video tile.
pub fn label_has_duration(label: &str) -> bool {
    let b = label.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            // need at least one more digit group after a colon
            if i < b.len() && b[i] == b':' {
                let mut j = i + 1;
                let mut digits = 0;
                while j < b.len() && b[j].is_ascii_digit() {
                    j += 1;
                    digits += 1;
                }
                if digits >= 1 {
                    return true;
                }
                i = j;
            }
        } else {
            i += 1;
        }
    }
    false
}

/// The inventory ask for video tiles. Explicit about one box per thumbnail and about what
/// to label: a vaguer ask ("media thumbnails to select") made the model answer with one box
/// around the whole grid, labelled with the task, so nothing was ever clicked.
pub const VIDEO_TILES: &str = "Locate every video thumbnail in this photo grid. A video thumbnail shows a play icon with a duration such as 0:06 in its top-right corner. Use the duration text as each label.";

/// The inventory ask when photos are wanted too.
pub const ALL_TILES: &str =
    "Locate each thumbnail in this photo grid. Use the visible duration badge text (e.g. 0:12) as the label when the thumbnail has one, otherwise \"photo\".";

/// The inventory ask for the picker's selection counter ("12 selected").
/// No number in it: an example count ("12 selected", as it once read) is what the model
/// answered when the header showed no selection at all — live, three picks "selected 12".
pub const SELECTED_COUNTER: &str = "Locate the text near the top-left corner that says how many items are selected: a number followed by the word 'selected'. Use that exact text as the label. If the header shows no such text (it invites you to select items instead), answer with an empty list.";

/// The number in an "N selected" label (a number, white space, then "selected"); `None` for
/// anything else — a bare number, "Select up to … items" — or a count beyond the picker's
/// 2000-item ceiling.
pub fn selected_count(label: &str) -> Option<u32> {
    let l = label.to_lowercase();
    l.match_indices("selected").find_map(|(at, _)| {
        let before = l.get(..at)?;
        let number = before.trim_end();
        if number.len() == before.len() {
            return None; // "12selected": not the picker's text
        }
        let digits: Vec<char> = number.chars().rev().take_while(char::is_ascii_digit).take(7).collect();
        if digits.is_empty() || digits.len() > 6 {
            return None;
        }
        digits.iter().rev().collect::<String>().parse::<u32>().ok().filter(|n| (1..=2000).contains(n))
    })
}

/// A title as compared: lower case, letters and digits only, words separated by one space
/// ("Robots, 2020–2026" and "robots 2020-2026" are the same).
pub fn normalize_title(s: &str) -> String {
    let spaced: String = s.chars().flat_map(|c| if c.is_alphanumeric() { c.to_lowercase().collect::<Vec<_>>() } else { vec![' '] }).collect();
    spaced.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether a grounded element is inside the search `field`: its centre lies in the field's box,
/// or at least half of its own box does. The field holds what was typed — an album's title,
/// which the model offered live as "the album", as the album's heading and as a tile — so in an
/// album pick nothing there is ever a result, a tile or a heading.
pub fn in_field(e: &Element, field: &Element) -> bool {
    let (cx, cy) = e.center();
    if cx >= field.x1 && cx < field.x2 && cy >= field.y1 && cy < field.y2 {
        return true;
    }
    let w = u64::from(e.x2.min(field.x2).saturating_sub(e.x1.max(field.x1)));
    let h = u64::from(e.y2.min(field.y2).saturating_sub(e.y1.max(field.y1)));
    e.area() > 0 && w.saturating_mul(h).saturating_mul(2) >= e.area()
}

/// Whether the tile ending a range is the one that started it: the grid never moved
/// (`scrolled` false) and either box holds the other's click point. A Shift+click there would
/// toggle the only selected item back off.
pub fn same_tile(first: &Element, last: &Element, scrolled: bool) -> bool {
    if scrolled {
        return false;
    }
    let inside = |p: (u32, u32), e: &Element| p.0 >= e.x1 && p.0 < e.x2 && p.1 >= e.y1 && p.1 < e.y2;
    inside(click_point(last), first) || inside(click_point(first), last)
}

/// Appended for the one retry after [`enumeration_failed`].
pub const STRICT: &str = "Your previous answer grouped several thumbnails into one box or repeated one label: give exactly one tight box per individual thumbnail, each labelled with its own duration text.";

/// Smallest tile worth clicking, in capture pixels (icons and badges are smaller).
const MIN_TILE_SIDE: u32 = 20;
/// Furthest a click goes into a tile from its left edge, in capture pixels. Grounded boxes
/// start where the tile starts but sometimes run on into the neighbour, so the left third
/// is the safe place to click, not the centre (live: 10 of 11 tiles hit vs 7 of 11).
const MAX_CLICK_INSET: u32 = 40;

/// Whether an element looks like a selectable media tile in a `w`×`h` capture: big enough
/// to be a thumbnail, small enough not to be the whole grid.
pub fn is_tile(e: &Element, w: u32, h: u32) -> bool {
    let screen = u64::from(w) * u64::from(h);
    let side_ok = e.x2.saturating_sub(e.x1) >= MIN_TILE_SIDE && e.y2.saturating_sub(e.y1) >= MIN_TILE_SIDE;
    side_ok && e.area().saturating_mul(10) < screen.saturating_mul(6)
}

/// Whether an element looks like a video tile (duration badge or 'video' in text).
pub fn is_video_tile(e: &Element) -> bool {
    label_has_duration(&e.label) || e.kind.to_lowercase().contains("video") || e.label.to_lowercase().contains("video")
}

/// Whether an inventory reply is the model failing to enumerate rather than a real list:
/// one box covering most of the screen (the grid as a whole), or one label stamped on many
/// boxes (the model looping). An empty list is not a failure — nothing is visible.
pub fn enumeration_failed(elements: &[Element], w: u32, h: u32) -> bool {
    let screen = u64::from(w) * u64::from(h);
    if elements.iter().any(|e| e.area().saturating_mul(10) >= screen.saturating_mul(6)) {
        return true;
    }
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for e in elements {
        let l = e.label.trim().to_lowercase();
        if !l.is_empty() && l != "photo" {
            *counts.entry(l).or_default() += 1;
        }
    }
    counts.values().any(|&n| n >= 4 && n.saturating_mul(2) > elements.len())
}

/// The inventory ask for the buttons of Chrome's "Leave site?" dialog (the picker holds its
/// selection in a beforeunload handler, so closing or navigating its tab asks first), each
/// labelled with its own text. Asked to `find` "the Leave button" instead, the model boxes it
/// right but labels it with the ask's own words ("Leave button of the browser's 'Leave site?'
/// dialog", live), which says nothing about what the box reads; listed with their text, the
/// buttons come back as "Cancel" and "Leave", and a screen without the dialog as none.
pub const LEAVE_DIALOG_BUTTONS: &str = "Locate the buttons of the browser's 'Leave site?' dialog (it asks whether to leave the page). Use each button's own text, exactly as written on it, as its label.";

/// Whether a grounded element is the Leave button: labelled "Leave" (nothing else — never
/// Cancel, Stay or a page element the model offered instead) and button-sized, never a
/// thumbnail-sized box.
pub fn is_leave_button(e: &Element, w: u32, h: u32) -> bool {
    let screen = u64::from(w) * u64::from(h);
    let label = normalize_title(&e.label);
    matches!(label.as_str(), "leave" | "leave button") && e.area() > 0 && e.area().saturating_mul(40) < screen
}

/// The Leave button among a [`LEAVE_DIALOG_BUTTONS`] inventory — only with the dialog's Cancel
/// beside it: on the same row, just left of it. Live, with no dialog showing, the model once
/// offered a lone "Leave" box on the picker's grid, and clicking it opened a photo; the dialog
/// always shows both buttons together.
pub fn leave_button(elements: &[Element], w: u32, h: u32) -> Option<&Element> {
    let screen = u64::from(w) * u64::from(h);
    let is_cancel =
        |e: &Element| matches!(normalize_title(&e.label).as_str(), "cancel" | "cancel button") && e.area() > 0 && e.area().saturating_mul(40) < screen;
    elements.iter().filter(|e| is_leave_button(e, w, h)).find(|leave| {
        let (lw, lh) = (leave.x2.saturating_sub(leave.x1), leave.y2.saturating_sub(leave.y1));
        elements.iter().filter(|c| is_cancel(c)).any(|c| {
            let same_row = leave.center().1.abs_diff(c.center().1) <= lh.max(c.y2.saturating_sub(c.y1)).max(1);
            let just_left = c.x2 <= leave.x1.saturating_add(lw / 4) && leave.x1.saturating_sub(c.x2) <= lw.saturating_mul(3).max(1);
            same_row && just_left
        })
    })
}

/// The inventory ask for the date groups of a search result: Google's picker heads each day's
/// items with its date and a round check button left of it that selects that whole day — except a
/// day of one item, which in a search result's compact rows shows no check button at all, not even
/// on hover (live, Chrome 155). The headings are asked for, not the buttons: asked for "the round
/// check button left of each date heading", qwen2.5vl boxed the selection circles of tiles under
/// other headings and labelled them with those dates (live), so a click selected one tile or
/// unselected another day; asked for headings, it boxes each on its own row with its own date —
/// from the check button, or from the date's text. Nor is it asked whether a check is filled in:
/// it never said so (live), and the header's count tells instead.
pub const DAY_HEADINGS: &str = "Locate every date heading on this screen: a date such as 'Thu, Oct 13, 2022' or 'October 13, 2022' written above a group of photos, with or without a round check button beside it. Give one box per date heading, around its text, and use the date heading's text exactly as written as the label.";

const MONTH_NAMES: [&str; 12] = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"];

/// The month (1–12) a word names: its full name, or its first three letters ("oct", "Sept").
fn month_of(word: &str) -> Option<u32> {
    let w = word.to_lowercase();
    if w.chars().count() < 3 {
        return None;
    }
    MONTH_NAMES.iter().position(|m| m.starts_with(w.as_str())).and_then(|i| u32::try_from(i + 1).ok())
}

/// The words of `s`: letters and digits, lower case.
fn words(s: &str) -> Vec<String> {
    normalize_title(s).split(' ').filter(|w| !w.is_empty()).map(str::to_string).collect()
}

/// Whether a search asks for a date ("videos October 13 2022", "Oct 2022", "videos 2019"): it
/// names a month or a year. Such results come grouped by day, and the day groups' check
/// buttons select them whole.
pub fn is_date_search(q: &str) -> bool {
    words(q).iter().any(|w| month_of(w).is_some() || is_year(w))
}

fn is_year(w: &str) -> bool {
    w.len() == 4 && w.parse::<u32>().is_ok_and(|y| (1900..=2100).contains(&y))
}

/// The day a date heading names, as a key that stays the same however the heading is boxed or
/// abbreviated: `"2022-10-13"`, or `"10-13"` when it shows no year (Google leaves the current
/// year out), or `"2022-10"` for a month heading. `None` when the label names no month, or
/// names what a heading never does — videos, photos, a search: live, the model offered the search
/// box, labelled with the typed `videos October 2022`, as a date group.
pub fn date_key(label: &str) -> Option<String> {
    let ws = words(label);
    if !is_heading(&ws) {
        return None;
    }
    key_of(date_parts(&ws))
}

/// Whether words could be a date heading's: none of them names what a heading never does.
fn is_heading(ws: &[String]) -> bool {
    !ws.iter().any(|w| matches!(w.as_str(), "video" | "videos" | "photo" | "photos" | "search" | "results"))
}

/// The (year, month, day) words name; a day only beside a month.
fn date_parts(ws: &[String]) -> (Option<u32>, Option<u32>, Option<u32>) {
    let month = ws.iter().find_map(|w| month_of(w));
    let year = ws.iter().find(|w| is_year(w)).and_then(|w| w.parse::<u32>().ok());
    let day = ws.iter().filter_map(|w| w.parse::<u32>().ok()).find(|d| (1..=31).contains(d));
    (year, month, month.and(day))
}

fn key_of((year, month, day): (Option<u32>, Option<u32>, Option<u32>)) -> Option<String> {
    let month = month?;
    Some(match (year, day) {
        (Some(y), Some(d)) => format!("{y}-{month:02}-{d:02}"),
        (None, Some(d)) => format!("{month:02}-{d:02}"),
        (Some(y), None) => format!("{y}-{month:02}"),
        (None, None) => return None,
    })
}

/// The day a date heading names inside the date search `query` ([`date_key`]), with the query's
/// year when the heading shows none (Google leaves the current year out; one heading read with and
/// without its year is one day, and a second click would clear it). `None` for a heading that is
/// no date, or a date outside the searched one: the heading must name the query's month and day
/// when it names them, and its year when both show one — a neighbouring day's results are another
/// search's to pick.
pub fn day_key(label: &str, query: &str) -> Option<String> {
    let ws = words(label);
    if !is_heading(&ws) {
        return None;
    }
    let (y, m, d) = date_parts(&ws);
    let (qy, qm, qd) = date_parts(&words(query));
    let named = |q: Option<u32>, h: Option<u32>| q.is_none_or(|q| h == Some(q));
    if !(named(qm, m) && named(qd, d) && (y.is_none() || named(qy, y))) {
        return None;
    }
    key_of((y.or(qy), m, d))
}

/// The search for one day of a date search, from its key (`"2022-10-13"`): `videos October 13
/// 2022`, the form of a sweep's own day searches — "videos" in front only when `query` asks for
/// videos. `None` for a key that names no day or no year (no search would find just that day).
pub fn day_query(key: &str, query: &str) -> Option<String> {
    let mut parts = key.split('-').map(|p| p.parse::<u32>().ok());
    let (Some(Some(y)), Some(Some(m)), Some(Some(d)), None) = (parts.next(), parts.next(), parts.next(), parts.next()) else { return None };
    let name = MONTH_NAMES.get(usize::try_from(m.checked_sub(1)?).ok()?)?;
    if !(is_year(&y.to_string()) && (1..=31).contains(&d)) {
        return None;
    }
    let mut chars = name.chars();
    let name: String = chars.next().map(|c| c.to_uppercase().chain(chars).collect()).unwrap_or_default();
    let videos = words(query).iter().any(|w| w == "video" || w == "videos");
    Some(if videos { format!("videos {name} {d} {y}") } else { format!("{name} {d} {y}") })
}

/// The tile to select a day by when its heading has no check button — a day of one item, in a
/// search result's compact rows (live: not even on hover): the tile just below the heading that
/// starts about where the heading starts. Only when no other tile follows it on its row before the
/// next heading there (`headings`, the page's date headings), none before it reaches under the
/// heading's start (the box may sit right of where its day starts), and no other heading on the
/// row starts over it: a click on one tile of a bigger day would select part of it and pass for
/// the whole, and another day's tile would pass for this one. `None` then, or when no tile is there.
pub fn lone_tile(heading: &Element, headings: &[Element], tiles: &[Element]) -> Option<Element> {
    let hh = heading.y2.saturating_sub(heading.y1);
    let (_, hy) = heading.center();
    let side = |t: &Element| (t.x2.saturating_sub(t.x1), t.y2.saturating_sub(t.y1));
    let tile = tiles
        .iter()
        .filter(|t| {
            let (tw, th) = side(t);
            t.y1 >= hy && t.y1.saturating_sub(heading.y2) <= (th / 2).max(hh) && t.x1.abs_diff(heading.x1) <= tw / 2
        })
        .min_by_key(|t| (t.y1, t.x1.abs_diff(heading.x1)))?;
    let (tw, th) = side(tile);
    let (_, ty) = tile.center();
    // where the next day on the heading's row starts
    let end =
        headings.iter().filter(|o| o.center().1.abs_diff(hy) <= hh.max(1) && o.x1 > tile.x1.saturating_add(tw / 2)).map(|o| o.x1).min().unwrap_or(u32::MAX);
    let more = tiles
        .iter()
        .any(|t| t != tile && t.center().1.abs_diff(ty) < (th / 2).max(1) && t.x1 >= tile.x1.saturating_add(tw / 2) && t.x1.saturating_add(tw / 4) < end);
    // a tile just before it on its row (tiles of one day sit a few pixels apart, days further) or
    // reaching under the heading's start is of the same day
    let behind = tiles.iter().any(|t| {
        let reach = t.x2.saturating_add(hh) > tile.x1 || t.x2 > heading.x1.saturating_sub(hh);
        t != tile && t.center().1.abs_diff(ty) < (th / 2).max(1) && t.x1 < tile.x1 && reach
    });
    // another heading on the row starting over it too: two days claim the tile (live, a box for
    // Oct 7 lay over Oct 8's lone tile, and Oct 8's clip was picked for Oct 7)
    let claimed = headings.iter().any(|o| o != heading && o.center().1.abs_diff(hy) <= hh.max(1) && o.x1.abs_diff(tile.x1) <= tw / 2);
    (!more && !behind && !claimed).then(|| tile.clone())
}

/// Whether an element is the size of a heading (or its check button) in a `w`×`h` capture: never
/// a thumbnail, never a band across the grid.
pub fn heading_sized(e: &Element, w: u32, h: u32) -> bool {
    let screen = u64::from(w) * u64::from(h);
    e.area() > 0 && e.area().saturating_mul(60) < screen && e.y2.saturating_sub(e.y1).saturating_mul(14) <= h
}

/// The date-group checks worth clicking in a `w`×`h` capture, in reading order: labelled with
/// a date ([`date_key`]: the key that keeps one day from being clicked twice — a second click
/// clears it), heading-sized (never a thumbnail), one per date.
pub fn day_checks(elements: &[Element], w: u32, h: u32) -> Vec<Element> {
    let band = (h / 16).max(1);
    let mut found: Vec<Element> = elements.iter().filter(|e| date_key(&e.label).is_some()).filter(|e| heading_sized(e, w, h)).cloned().collect();
    found.sort_by_key(|e| (e.y1 / band, e.x1));
    let mut keys: Vec<String> = Vec::new();
    found.retain(|e| match date_key(&e.label) {
        Some(k) if !keys.contains(&k) => {
            keys.push(k);
            true
        }
        _ => false,
    });
    found
}

/// Where to click a day check: the centre of a squarish box (the button itself), or the left
/// end of a wide one (the model boxed the heading too; the button sits left of the date).
pub fn check_point(e: &Element) -> (u32, u32) {
    let (w, h) = (e.x2.saturating_sub(e.x1), e.y2.saturating_sub(e.y1));
    let (cx, cy) = e.center();
    if w > h.saturating_mul(2) { (e.x1.saturating_add(h / 2), cy) } else { (cx, cy) }
}

/// Where a heading's check button may be, in the order to try, both on the heading's own row: the
/// left end of its box ([`check_point`]: the model boxed the heading from the button), then just
/// left of the box (it boxed the date's text, and the button sits left of it — live, both).
pub fn check_points(e: &Element) -> [(u32, u32); 2] {
    let h = e.y2.saturating_sub(e.y1).max(1);
    let (x, y) = check_point(e);
    [(x, y), (e.x1.saturating_sub(h), y)]
}

/// A resumed date search: its search and the day it resumes from — `"videos October 2022 from
/// 2022-10-29"` is `("videos October 2022", Some("2022-10-29"))` — the days after that one (newer)
/// were picked by an earlier session. Anything else is the search as it is.
pub fn resume_point(query: &str) -> (&str, Option<&str>) {
    match query.rsplit_once(" from ") {
        Some((base, key)) if day_query(key.trim(), "").is_some() && !base.trim().is_empty() => (base.trim_end(), Some(key.trim())),
        _ => (query, None),
    }
}

/// The search that resumes `query` from the day `key` (`"2022-10-29"`) — see [`resume_point`].
pub fn resume_from(query: &str, key: &str) -> String {
    format!("{} from {key}", resume_point(query).0)
}

/// Where to click a grounded tile: its left third (at most [`MAX_CLICK_INSET`] in),
/// vertically centred.
pub fn click_point(e: &Element) -> (u32, u32) {
    let inset = (e.x2.saturating_sub(e.x1) / 3).min(MAX_CLICK_INSET);
    (e.x1.saturating_add(inset), e.center().1)
}

/// The tiles worth clicking in a `w`×`h` capture, in reading order (the picker lists newest
/// first), videos before photos when photos are allowed. A box whose click point falls
/// inside a tile already listed is the model repeating itself and is dropped.
pub fn tile_candidates(elements: &[Element], videos_only: bool, w: u32, h: u32) -> Vec<Element> {
    let band = (h / 16).max(1);
    let mut tiles: Vec<Element> = elements.iter().filter(|e| is_tile(e, w, h) && (!videos_only || is_video_tile(e))).cloned().collect();
    tiles.sort_by_key(|e| (e.y1 / band, e.x1));
    tiles.sort_by_key(|e| !is_video_tile(e)); // stable: reading order kept within each kind
    let mut kept: Vec<Element> = Vec::new();
    for t in tiles {
        let (px, py) = click_point(&t);
        if !kept.iter().any(|k| px >= k.x1 && px < k.x2 && py >= k.y1 && py < k.y2) {
            kept.push(t);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn el(label: &str, kind: &str, x1: u32, y1: u32, x2: u32, y2: u32) -> Element {
        Element { label: label.into(), kind: kind.into(), x1, y1, x2, y2 }
    }

    #[test]
    fn duration_badges_are_recognized() {
        for yes in ["0:30", "video 12:04", "1:02:33 trip", "0:06", "Duration 4:02"] {
            assert!(label_has_duration(yes), "{yes}");
        }
        for no in ["beach", "2024", "13: no", ":", "12-04", "search", "1h02"] {
            assert!(!label_has_duration(no), "{no}");
        }
    }

    #[test]
    fn selected_counts_are_read_from_the_header() {
        assert_eq!(selected_count("37 selected"), Some(37));
        assert_eq!(selected_count("1 Selected"), Some(1));
        assert_eq!(selected_count("✓ 2000  selected"), Some(2000));
        // only "<number> selected" is a count: the live failure read a bare "12" (or parroted
        // the prompt's example) off a header that showed no selection
        for no in [
            "12",
            "Selected: 12 items",
            "Select up to 300 items",
            "Add from Google Photos",
            "Done",
            "selected",
            "12selected",
            "0 selected",
            "2001 selected",
            "99999999 selected",
        ] {
            assert_eq!(selected_count(no), None, "{no}");
        }
        // and the ask itself offers no number to copy
        assert!(!SELECTED_COUNTER.chars().any(|c| c.is_ascii_digit()), "{SELECTED_COUNTER}");
    }

    #[test]
    fn video_searches_ask_google_for_videos() {
        assert_eq!(search_text(Some("beach sunset"), true).as_deref(), Some("beach sunset videos"));
        assert_eq!(search_text(Some("  dog  "), true).as_deref(), Some("dog videos"));
        assert_eq!(search_text(Some("Birthday Videos"), true).as_deref(), Some("Birthday Videos")); // already asks
        assert_eq!(search_text(Some("my video diary"), true).as_deref(), Some("my video diary"));
        assert_eq!(search_text(None, true).as_deref(), Some("videos")); // no query: still filter
        assert_eq!(search_text(Some("   "), true).as_deref(), Some("videos"));
        // photos allowed: the query as given, or no search at all
        assert_eq!(search_text(Some("beach"), false).as_deref(), Some("beach"));
        assert_eq!(search_text(Some(" "), false), None);
        assert_eq!(search_text(None, false), None);
    }

    #[test]
    fn tiles_filter_and_rank_in_reading_order() {
        let els = vec![
            el("photo", "", 0, 0, 100, 100),        // photo tile, first row
            el("0:30", "", 200, 300, 400, 500),     // video, second row
            el("0:06", "", 150, 2, 250, 98),        // video, first row (y1 jitter)
            el("Done", "button", 900, 0, 990, 12),  // too thin to be a tile
            el("granny", "text", 0, 500, 800, 512), // a text strip, too thin
            el("0:06", "", 160, 10, 260, 90),       // the model repeating the 0:06 tile
        ];
        let vids = tile_candidates(&els, true, 1000, 800);
        assert_eq!(vids.iter().map(|e| e.label.as_str()).collect::<Vec<_>>(), vec!["0:06", "0:30"]);
        assert_eq!(vids[0].x1, 150); // the first of the duplicates wins
        let all = tile_candidates(&els, false, 1000, 800);
        assert_eq!(all.iter().map(|e| e.label.as_str()).collect::<Vec<_>>(), vec!["0:06", "0:30", "photo"]);
    }

    #[test]
    fn tiny_icons_and_whole_grid_boxes_are_not_tiles() {
        assert!(tile_candidates(&[el("0:06", "", 0, 0, 15, 15)], true, 1000, 800).is_empty());
        // the failure seen live: one box around the grid, labelled with the task
        let grid = el("media thumbnails to select", "", 40, 150, 990, 790);
        assert!(tile_candidates(std::slice::from_ref(&grid), false, 1000, 800).is_empty());
        assert!(enumeration_failed(&[grid], 1000, 800));
    }

    #[test]
    fn enumeration_failures_are_recognized() {
        // one label stamped on many boxes is the model looping
        let looping: Vec<Element> = (0..12).map(|i| el("2:54", "", i * 80, 600, i * 80 + 70, 760)).collect();
        assert!(enumeration_failed(&looping, 1344, 784));
        // a real list (and an empty one) is not
        let real = vec![el("0:06", "", 60, 405, 171, 444), el("4:02", "", 445, 405, 557, 444), el("2:41", "", 742, 405, 853, 444)];
        assert!(!enumeration_failed(&real, 1344, 784));
        assert!(!enumeration_failed(&[], 1344, 784));
        // many plain photos are fine
        let photos: Vec<Element> = (0..8).map(|i| el("photo", "", i * 100, 0, i * 100 + 90, 90)).collect();
        assert!(!enumeration_failed(&photos, 1344, 784));
    }

    #[test]
    fn titles_are_compared_by_their_words() {
        assert_eq!(normalize_title("  Robots,\t2020–2026! "), "robots 2020 2026");
        assert_eq!(normalize_title("ÉTÉ à Paris"), "été à paris");
        assert_eq!(normalize_title(" !! "), "");
    }

    #[test]
    fn what_lies_in_the_search_field_is_in_it() {
        let field = el("Search", "input", 100, 10, 600, 40);
        // the typed title, boxed tightly, loosely, or as a tile-sized "photo" (live: all three)
        for typed in [el("Every Robot I've Met", "", 120, 15, 300, 35), el("Every Robot I've Met", "", 90, 5, 620, 45), el("photo", "", 110, 0, 160, 50)] {
            assert!(in_field(&typed, &field), "{typed:?}");
        }
        // the grid's first row just under the field, a heading beside it, the Done button
        for outside in [el("0:12", "", 100, 38, 200, 138), el("Thu, Oct 16", "", 10, 50, 90, 60), el("Done", "", 900, 10, 990, 40)] {
            assert!(!in_field(&outside, &field), "{outside:?}");
        }
        // degenerate boxes don't panic
        assert!(in_field(&el("", "", 200, 20, 200, 20), &field));
        assert!(!in_field(&el("", "", 5, 5, 5, 5), &field));
        assert!(!in_field(&el("", "", 5, 5, 5, 5), &el("", "", 5, 5, 5, 5)));
    }

    #[test]
    fn a_range_that_ends_where_it_started_is_one_tile() {
        let a = el("0:07", "", 10, 40, 50, 80);
        assert!(same_tile(&a, &a, false));
        assert!(same_tile(&a, &el("0:07", "", 12, 38, 52, 82), false), "boxed a little differently");
        assert!(!same_tile(&a, &el("0:12", "", 60, 40, 100, 80), false), "the next tile");
        assert!(!same_tile(&a, &a, true), "after scrolling, the same spot is another tile");
    }

    #[test]
    fn only_a_leave_labelled_button_is_the_leave_button() {
        assert!(is_leave_button(&el("Leave", "button", 700, 400, 760, 430), 1344, 784));
        assert!(is_leave_button(&el(" leave ", "", 700, 400, 760, 430), 1344, 784));
        assert!(is_leave_button(&el("Leave button", "", 700, 400, 760, 430), 1344, 784));
        // anything else is refused: the other button, the dialog's title, page elements
        for label in ["Cancel", "Stay", "Leave site?", "Leave page", "Done", "", "Don't leave"] {
            assert!(!is_leave_button(&el(label, "button", 700, 400, 760, 430), 1344, 784), "{label:?}");
        }
        // a thumbnail-sized box, or an empty one, is never clicked as a button
        assert!(!is_leave_button(&el("Leave", "", 100, 100, 400, 400), 1344, 784));
        assert!(!is_leave_button(&el("Leave", "", 5, 5, 5, 5), 1344, 784));
    }

    #[test]
    fn the_leave_button_is_read_from_the_dialogs_buttons() {
        let mut png = std::io::Cursor::new(Vec::new());
        image::GrayImage::from_pixel(1344, 784, image::Luma([20])).write_to(&mut png, image::ImageFormat::Png).unwrap();
        let png = png.into_inner();
        // qwen2.5vl's live answer to LEAVE_DIALOG_BUTTONS over Chrome 155's dialog
        let reply = r#"```json [ {"bbox_2d": [609, 296, 663, 316], "label": "Cancel"}, {"bbox_2d": [675, 296, 730, 316], "label": "Leave"} ] ```"#;
        let els = crate::ground::parse_inventory_reply(reply, &png).unwrap();
        let leave = leave_button(&els, 1344, 784).unwrap();
        assert_eq!((leave.label.as_str(), leave.x1, leave.x2), ("Leave", 675, 730));
        // its answer to a `find` of "the Leave button": the right box, labelled with the ask's
        // own words — no proof of what it reads, so not clicked
        let echo = crate::ground::parse_inventory_reply(
            r#"```json {"bbox_2d": [676, 296, 757, 318], "label": "Leave button of the browser's 'Leave site?' dialog"} ```"#,
            &png,
        )
        .unwrap();
        assert!(leave_button(&echo, 1344, 784).is_none());
        // no dialog: "There are none."
        assert!(leave_button(&crate::ground::parse_inventory_reply("There are none.", &png).unwrap(), 1344, 784).is_none());
        // only Cancel offered: nothing
        assert!(leave_button(&els[..1], 1344, 784).is_none());
        // a lone "Leave" (live, with no dialog showing: clicking it opened a photo): nothing
        assert!(leave_button(&els[1..], 1344, 784).is_none());
        // Cancel on another row, right of Leave, or far away: not the dialog's pair
        for (x1, y1) in [(609, 400), (760, 296), (100, 296)] {
            let cancel = el("Cancel", "", x1, y1, x1 + 54, y1 + 20);
            let pair = [cancel, els[1].clone()];
            assert!(leave_button(&pair, 1344, 784).is_none(), "{x1},{y1}");
        }
    }

    #[test]
    fn date_searches_and_date_headings_are_recognized() {
        for yes in ["videos October 13 2022", "Oct 2022", "videos 2019", "sept 3", "VIDEOS DECEMBER 2017"] {
            assert!(is_date_search(yes), "{yes}");
        }
        for no in ["robots", "beach sunset videos", "videos", "1:30", "123456", "ma"] {
            assert!(!is_date_search(no), "{no}");
        }
        assert_eq!(date_key("Thu, Oct 13, 2022").as_deref(), Some("2022-10-13"));
        assert_eq!(date_key("October 13, 2022").as_deref(), Some("2022-10-13"), "the same day however it is written");
        assert_eq!(date_key("Sun, Sep 4").as_deref(), Some("09-04"), "no year shown");
        assert_eq!(date_key("December 2017").as_deref(), Some("2017-12"));
        for no in ["2022", "photo", "", "0:30", "Thu 13", "October", "videos October 2022", "Search results for October 2022", "photos Oct 13, 2022"] {
            assert_eq!(date_key(no), None, "{no:?}");
        }
    }

    #[test]
    fn day_checks_are_dated_heading_sized_and_one_per_day() {
        let els = vec![
            el("Wed, Oct 12, 2022", "check", 20, 300, 44, 324), // second row
            el("Thu, Oct 13, 2022", "check", 20, 100, 44, 124), // first row
            el("October 13, 2022", "check", 22, 102, 46, 126),  // the model repeating the first
            el("0:30", "", 60, 130, 200, 260),                  // a tile: no date
            el("Fri, Oct 14, 2022", "", 0, 0, 900, 700),        // a whole-grid box
            el("Search", "", 300, 10, 700, 40),                 // no date
            el("videos October 2022", "", 300, 10, 700, 40),    // the search box, as typed (live)
        ];
        let checks = day_checks(&els, 1344, 784);
        assert_eq!(checks.iter().map(|e| e.label.as_str()).collect::<Vec<_>>(), ["Thu, Oct 13, 2022", "Wed, Oct 12, 2022"]);
        // a square box is clicked in its centre; a wide one (the heading boxed too) at its left end
        assert_eq!(check_point(&el("", "", 20, 100, 44, 124)), (32, 112));
        assert_eq!(check_point(&el("", "", 20, 100, 320, 124)), (32, 112));
        assert_eq!(check_point(&el("", "", 9, 9, 9, 9)), (9, 9));
    }

    #[test]
    fn day_keys_are_the_searched_dates_days() {
        let q = "videos October 2022";
        assert_eq!(day_key("Thu, Oct 13, 2022", q).as_deref(), Some("2022-10-13"));
        // read without its year (Google leaves the current year out): the same day
        assert_eq!(day_key("Thu, Oct 13", q).as_deref(), Some("2022-10-13"));
        assert_eq!(day_key("Thu, Oct 13", "videos 2026").as_deref(), Some("2026-10-13"));
        assert_eq!(day_key("Thu, Oct 13", "videos October").as_deref(), Some("10-13"));
        // a neighbouring day Google's search shows too is another search's
        for no in ["Fri, Sep 30, 2022", "Thu, Oct 13, 2021", "videos October 2022", "0:30", "Search", ""] {
            assert_eq!(day_key(no, q), None, "{no:?}");
        }
        // a day search takes only its day, never a month heading
        let day = "videos October 13 2022";
        assert_eq!(day_key("October 13, 2022", day).as_deref(), Some("2022-10-13"));
        assert_eq!(day_key("Wed, Oct 12, 2022", day), None);
        assert_eq!(day_key("October 2022", day), None);
        assert_eq!(day_key("October 2022", q).as_deref(), Some("2022-10"));
    }

    #[test]
    fn day_searches_are_written_as_a_sweep_writes_them() {
        assert_eq!(day_query("2022-10-13", "videos October 2022").as_deref(), Some("videos October 13 2022"));
        assert_eq!(day_query("2022-11-05", "November 2022 videos").as_deref(), Some("videos November 5 2022"));
        assert_eq!(day_query("2017-12-31", "December 2017").as_deref(), Some("December 31 2017"));
        // no day or no year: no search finds just that day
        for no in ["10-13", "2022-10", "", "2022-13-01", "2022-00-01", "2022-10-32", "1800-10-13", "2022-10-13-1", "x-y-z"] {
            assert_eq!(day_query(no, "videos October 2022"), None, "{no:?}");
        }
    }

    #[test]
    fn a_day_without_a_check_is_selected_by_its_one_tile() {
        let t = |x1: u32, y1: u32| el("0:12", "", x1, y1, x1 + 100, y1 + 100);
        // compact rows (live: a day's tiles a few pixels apart, days further): Oct 14 (two tiles),
        // Oct 13 (one, no check button: its text boxed), Oct 12 (two)
        let oct14 = el("Fri, Oct 14, 2022", "", 10, 100, 30, 120);
        let oct13 = el("Thu, Oct 13, 2022", "", 250, 100, 350, 120);
        let oct12 = el("Wed, Oct 12, 2022", "", 390, 100, 410, 120);
        let headings = [oct14.clone(), oct13.clone(), oct12.clone()];
        let tiles = [t(10, 128), t(114, 128), t(250, 128), t(390, 128), t(494, 128), t(10, 240)];
        assert_eq!(lone_tile(&oct13, &headings, &tiles), Some(t(250, 128)));
        // a day with more tiles on its row: a click on one would select part of it
        assert_eq!(lone_tile(&oct14, &headings, &tiles), None);
        assert_eq!(lone_tile(&oct12, &headings, &tiles), None);
        // without the next heading, a tile after it may be its own: not lone either
        assert_eq!(lone_tile(&oct13, std::slice::from_ref(&oct13), &tiles), None);
        // the last day of a row
        assert_eq!(lone_tile(&oct13, &headings, &tiles[..3]), Some(t(250, 128)));
        // a box right of where its day starts: the tile under it is that day's second
        let off = el("Thu, Oct 13, 2022", "", 150, 100, 250, 120);
        assert_eq!(lone_tile(&off, &[oct14.clone(), off.clone()], &tiles[..2]), None);
        // another day's heading over the same tile: the tile is not this day's for sure
        let oct12_over = el("Wed, Oct 12, 2022", "", 255, 100, 355, 120);
        assert_eq!(lone_tile(&oct13, &[oct14.clone(), oct13.clone(), oct12_over], &tiles[..3]), None);
        // no tile right below, or none starting where the heading starts
        assert_eq!(lone_tile(&oct13, &headings, &[t(250, 400)]), None);
        assert_eq!(lone_tile(&oct13, &headings, &[t(320, 128)]), None);
        assert_eq!(lone_tile(&oct13, &headings, &[]), None);
        // degenerate boxes don't panic
        assert_eq!(lone_tile(&el("", "", 5, 5, 5, 5), &[], &[el("", "", 5, 5, 5, 5)]), Some(el("", "", 5, 5, 5, 5)));
    }

    #[test]
    fn clicks_land_in_the_left_third() {
        // a box that ran on into the next tile: the click stays near where the tile starts
        assert_eq!(click_point(&el("0:06", "", 59, 399, 322, 597)), (99, 498));
        // a narrow box: a third of its width
        assert_eq!(click_point(&el("0:25", "", 887, 628, 965, 765)), (913, 696));
        // degenerate boxes don't panic
        assert_eq!(click_point(&el("", "", 10, 10, 10, 10)), (10, 10));
    }
}
