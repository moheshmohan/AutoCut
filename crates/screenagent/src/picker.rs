//! What the Google Photos picker looks like: pure predicates over grounded elements, so
//! the agent's decisions are unit-testable without a browser.

use crate::ground::Element;

/// Description of the picker's search field for `find` — it searches the user's whole
/// library, which is the only "search Google Photos" any app can offer since the Library
/// API retired.
pub const SEARCH_FIELD: &str = "the search field at the top of the picker (an input box, may contain a magnifier icon or placeholder text like 'Search')";

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
    fn clicks_land_in_the_left_third() {
        // a box that ran on into the next tile: the click stays near where the tile starts
        assert_eq!(click_point(&el("0:06", "", 59, 399, 322, 597)), (99, 498));
        // a narrow box: a third of its width
        assert_eq!(click_point(&el("0:25", "", 887, 628, 965, 765)), (913, 696));
        // degenerate boxes don't panic
        assert_eq!(click_point(&el("", "", 10, 10, 10, 10)), (10, 10));
    }
}
