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

/// The picker's Albums view (a tab or button), if it has one. Google's picker guide says
/// albums "are not shown directly" and users find one by searching its title, so this is
/// looked for first and the search is the way when it is absent.
pub const ALBUMS_TAB: &str = "the 'Albums' tab or button in the picker's navigation (it switches the view from photos to the user's albums)";

/// The inventory ask for albums: cards of an albums view, or album entries the search suggests.
pub const ALBUM_TILES: &str = "Locate every album shown on this screen: album cards in a grid (a cover image with the album's title under it) and album entries in a list of search suggestions (an album icon or cover next to the album's title). Do not include the search box itself or individual photos. Use each album's title text, exactly as written, as its label.";

/// The inventory ask for the heading of an opened album (or the album a filtered grid shows).
pub const ALBUM_HEADING: &str = "Locate the title of the album whose photos this screen shows (the heading above the photos, or the album name in the search box or filter chip). Use that exact text as the label.";

/// A title as compared: lower case, letters and digits only, words separated by one space
/// ("Robots, 2020–2026" and "robots 2020-2026" are the same).
pub fn normalize_title(s: &str) -> String {
    let spaced: String = s.chars().flat_map(|c| if c.is_alphanumeric() { c.to_lowercase().collect::<Vec<_>>() } else { vec![' '] }).collect();
    spaced.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether a grounded label names the album `title`: the same normalized words, optionally
/// with the item count Google prints under a card ("12 items"), or "album" in front.
pub fn album_label_matches(label: &str, title: &str) -> bool {
    let t = normalize_title(title);
    if t.is_empty() {
        return false;
    }
    let mut l = normalize_title(label);
    if !t.starts_with("album ")
        && let Some(rest) = l.strip_prefix("album ")
    {
        l = rest.to_string();
    }
    if l == t {
        return true;
    }
    let Some(rest) = l.strip_prefix(&t).and_then(|r| r.strip_prefix(' ')) else { return false };
    let words: Vec<&str> = rest.split(' ').collect();
    matches!(words.as_slice(), [n, unit] if n.chars().all(|c| c.is_ascii_digit()) && matches!(*unit, "item" | "items" | "photo" | "photos" | "video" | "videos"))
}

/// Whether an element is the Albums tab [`ALBUMS_TAB`] asked for: labelled "album…" and
/// small, never a thumbnail-sized box (a misgrounded tile click would select a photo).
pub fn is_albums_tab(e: &Element, w: u32, h: u32) -> bool {
    let screen = u64::from(w) * u64::from(h);
    e.label.to_lowercase().contains("album") && e.area() > 0 && e.area().saturating_mul(40) < screen
}

/// The albums in a `w`×`h` capture whose label names `title`, in reading order: never a
/// whole-screen box, never one overlapping `avoid` (the search box, which holds the typed
/// title), and a box whose centre falls inside one already kept is the model repeating itself.
pub fn album_candidates(elements: &[Element], title: &str, avoid: Option<&Element>, w: u32, h: u32) -> Vec<Element> {
    let screen = u64::from(w) * u64::from(h);
    let band = (h / 16).max(1);
    let overlaps = |a: &Element, b: &Element| a.x1 < b.x2 && b.x1 < a.x2 && a.y1 < b.y2 && b.y1 < a.y2;
    let mut found: Vec<Element> = elements
        .iter()
        .filter(|e| album_label_matches(&e.label, title))
        .filter(|e| e.area() > 0 && e.area().saturating_mul(10) < screen.saturating_mul(6))
        .filter(|e| avoid.is_none_or(|a| !overlaps(e, a)))
        .cloned()
        .collect();
    found.sort_by_key(|e| (e.y1 / band, e.x1));
    let mut kept: Vec<Element> = Vec::new();
    for e in found {
        let (cx, cy) = e.center();
        if !kept.iter().any(|k| cx >= k.x1 && cx < k.x2 && cy >= k.y1 && cy < k.y2) {
            kept.push(e);
        }
    }
    kept
}

/// Whether a heading read off an opened album names `title` (it may say more: "Robots ·
/// 12 items", a search chip "Robots").
pub fn heading_names(label: &str, title: &str) -> bool {
    let t = normalize_title(title);
    !t.is_empty() && (album_label_matches(label, title) || format!(" {} ", normalize_title(label)).contains(&format!(" {t} ")))
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

/// The `find` ask for Chrome's "Leave site?" dialog (the picker holds its selection in a
/// beforeunload handler, so closing or navigating its tab asks first).
pub const LEAVE_BUTTON: &str =
    "the 'Leave' button of the browser's 'Leave site?' dialog (it asks whether to leave the page; its two buttons are 'Leave' and 'Cancel')";

/// Whether a grounded element is the Leave button [`LEAVE_BUTTON`] asked for: labelled
/// "Leave" (nothing else — never Cancel, Stay or a page element the model offered instead)
/// and button-sized, never a thumbnail-sized box.
pub fn is_leave_button(e: &Element, w: u32, h: u32) -> bool {
    let screen = u64::from(w) * u64::from(h);
    let label = normalize_title(&e.label);
    matches!(label.as_str(), "leave" | "leave button") && e.area() > 0 && e.area().saturating_mul(40) < screen
}

/// The inventory ask for the date groups of a search result: Google's picker heads each day's
/// items with its date and a round check button that selects that whole day.
pub const DAY_CHECKS: &str = "Locate the round check button just left of each date heading (a date such as 'Thu, Oct 13, 2022' or 'October 13, 2022' written above a group of photos). Give one box per date heading, around its round check button, and use the date heading's text exactly as written as the label. If the check button is already filled in (the day is selected), use the kind \"checked\". Do not include the check marks on individual photos.";

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
/// year out), or `"2022-10"` for a month heading. `None` when the label names no month.
pub fn date_key(label: &str) -> Option<String> {
    let ws = words(label);
    let month = ws.iter().find_map(|w| month_of(w))?;
    let year = ws.iter().find(|w| is_year(w)).cloned();
    let day = ws.iter().filter_map(|w| w.parse::<u32>().ok()).find(|d| (1..=31).contains(d));
    Some(match (year, day) {
        (Some(y), Some(d)) => format!("{y}-{month:02}-{d:02}"),
        (None, Some(d)) => format!("{month:02}-{d:02}"),
        (Some(y), None) => format!("{y}-{month:02}"),
        (None, None) => return None,
    })
}

/// Whether a grounded day check is already filled in (the day is selected).
pub fn is_checked(e: &Element) -> bool {
    let k = e.kind.to_lowercase();
    (k.contains("checked") && !k.contains("unchecked")) || k.contains("selected")
}

/// The date-group checks worth clicking in a `w`×`h` capture, in reading order: labelled with
/// a date ([`date_key`]: the key that keeps one day from being clicked twice — a second click
/// clears it), heading-sized (never a thumbnail), one per date.
pub fn day_checks(elements: &[Element], w: u32, h: u32) -> Vec<Element> {
    let screen = u64::from(w) * u64::from(h);
    let band = (h / 16).max(1);
    let mut found: Vec<Element> = elements
        .iter()
        .filter(|e| date_key(&e.label).is_some())
        .filter(|e| e.area() > 0 && e.area().saturating_mul(60) < screen && e.y2.saturating_sub(e.y1).saturating_mul(14) <= h)
        .cloned()
        .collect();
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
    fn album_titles_match_by_their_words() {
        assert_eq!(normalize_title("  Robots,\t2020–2026! "), "robots 2020 2026");
        assert_eq!(normalize_title("ÉTÉ à Paris"), "été à paris");
        for (label, title) in [
            ("Robot showcase", "Robot showcase"),
            ("robot SHOWCASE", "Robot showcase"),
            ("Robots, 2020-2026", "Robots, 2020–2026"),
            ("Robot showcase 12 items", "Robot showcase"),
            ("Robot showcase · 1 item", "Robot showcase"),
            ("Robot showcase\n3 videos", "Robot showcase"),
            ("Album: Robot showcase", "Robot showcase"),
            ("Album 2024", "Album 2024"),
        ] {
            assert!(album_label_matches(label, title), "{label:?} ~ {title:?}");
        }
        for (label, title) in [
            ("Robot showcase 2", "Robot showcase"),   // another album
            ("Robot showcase old", "Robot showcase"), // another album
            ("Robot", "Robot showcase"),              // a prefix is not the album
            ("Robots", "Robot"),                      // nor a longer word
            ("Robot showcase 12 apples", "Robot showcase"),
            ("Search", "Robot showcase"),
            ("anything", "!!!"), // nothing to match
            ("", ""),
        ] {
            assert!(!album_label_matches(label, title), "{label:?} !~ {title:?}");
        }
        assert!(heading_names("Robot showcase", "Robot showcase"));
        assert!(heading_names("Search: robot showcase", "Robot showcase"));
        assert!(!heading_names("Robot showcases", "Robot showcase"));
        assert!(!heading_names("Photos", "Robot showcase") && !heading_names("Photos", ""));
    }

    #[test]
    fn album_candidates_are_matching_cards_not_the_search_box() {
        let search = el("Robot showcase", "input", 100, 10, 600, 40);
        let els = vec![
            search.clone(),                                      // the typed title in the search box
            el("Robot showcase", "album", 100, 50, 400, 90),     // the suggestion
            el("Robot showcase 2", "album", 100, 100, 400, 140), // another album
            el("Robot showcase", "album", 110, 55, 390, 85),     // the model repeating itself
            el("Robot showcase", "", 0, 0, 1000, 800),           // a whole-screen box
            el("Robot showcase", "", 5, 5, 5, 5),                // an empty box
        ];
        let found = album_candidates(&els, "Robot showcase", Some(&search), 1000, 800);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!((found[0].x1, found[0].y1), (100, 50));
        // two different albums with the title: both kept (the agent refuses to choose)
        let two = vec![el("Robots", "", 10, 100, 200, 300), el("Robots", "", 400, 100, 600, 300)];
        assert_eq!(album_candidates(&two, "Robots", None, 1000, 800).len(), 2);
        assert!(album_candidates(&two, "Cars", None, 1000, 800).is_empty());
    }

    #[test]
    fn the_albums_tab_is_a_small_album_labelled_element() {
        assert!(is_albums_tab(&el("Albums", "tab", 200, 10, 260, 30), 1000, 800));
        assert!(!is_albums_tab(&el("Photos", "tab", 200, 10, 260, 30), 1000, 800));
        // a thumbnail-sized box is never taken for the tab (clicking it would select a photo)
        assert!(!is_albums_tab(&el("album cover", "", 100, 100, 300, 300), 1000, 800));
        assert!(!is_albums_tab(&el("Albums", "", 10, 10, 10, 10), 1000, 800));
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
        for no in ["2022", "photo", "", "0:30", "Thu 13", "October"] {
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
        ];
        let checks = day_checks(&els, 1344, 784);
        assert_eq!(checks.iter().map(|e| e.label.as_str()).collect::<Vec<_>>(), ["Thu, Oct 13, 2022", "Wed, Oct 12, 2022"]);
        assert!(!is_checked(&checks[0]));
        assert!(is_checked(&el("x", "checked", 0, 0, 1, 1)) && is_checked(&el("x", "Selected", 0, 0, 1, 1)));
        assert!(!is_checked(&el("x", "unchecked", 0, 0, 1, 1)));
        // a square box is clicked in its centre; a wide one (the heading boxed too) at its left end
        assert_eq!(check_point(&el("", "", 20, 100, 44, 124)), (32, 112));
        assert_eq!(check_point(&el("", "", 20, 100, 320, 124)), (32, 112));
        assert_eq!(check_point(&el("", "", 9, 9, 9, 9)), (9, 9));
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
