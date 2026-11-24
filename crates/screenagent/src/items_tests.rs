use super::*;

fn w(captured: &str, kind: &str) -> Wanted {
    Wanted::parse(captured, kind).unwrap()
}

fn d(s: &str) -> NaiveDate {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
}

fn labels(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// A Mac four hours east of UTC.
fn east4(_: DateTime<Utc>) -> i32 {
    240
}

#[test]
fn wanted_items_parse_and_drop_fractions() {
    let a = w("2026-07-08T08:03:16.352Z", "photo");
    assert_eq!((a.rfc3339().as_str(), a.video), ("2026-07-08T08:03:16Z", false));
    let b = w(" 2023-11-26T06:50:11+04:00 ", "VIDEO");
    assert_eq!((b.rfc3339().as_str(), b.video), ("2023-11-26T02:50:11Z", true));
    for bad in ["", "yesterday", "2023-13-01T00:00:00Z", "2023-11-26 02:50:11", "9999999-01-01T00:00:00Z"] {
        assert!(Wanted::parse(bad, "video").is_none(), "{bad}");
    }
}

#[test]
fn tile_labels_give_kind_and_local_time() {
    let (video, at) = tile_time("Video - Landscape - Jul 28, 2025, 7:18:02\u{202f}PM").unwrap();
    assert!(video);
    assert_eq!(at.to_string(), "2025-07-28 19:18:02");
    let (video, at) = tile_time("Photo - Portrait - Mar 10, 2024, 12:42:31 AM").unwrap();
    assert!(!video);
    assert_eq!(at.to_string(), "2024-03-10 00:42:31");
    assert_eq!(tile_time("Photo - Square - Sept. 3, 2023, 9:00:00 PM").map(|t| t.1.to_string()).as_deref(), Some("2023-09-03 21:00:00"));
    for bad in ["", "Video", "Select all photos from Tue, Jul 29, 2025", "Video - Landscape - Jul 32, 2025, 7:18:02 PM", "Photo - x - 13:00"] {
        assert!(tile_time(bad).is_none(), "{bad}");
    }
}

#[test]
fn days_are_typed_as_the_picker_reads_them() {
    assert_eq!(day_text(d("2026-10-03")), "October 3 2026");
    assert_eq!(day_text(d("2023-11-26")), "November 26 2023");
}

#[test]
fn the_hash_is_fnv1a_of_the_utf16_units() {
    assert_eq!(label_hash(""), 0x811c_9dc5);
    assert_eq!(label_hash("a"), 0xe40c_292c);
    assert_eq!(label_hash("foobar"), 0xbf9c_f968);
    // what the page script's fnv() answers for the same label (node, 2026-11-23)
    assert_eq!(label_hash("Video - Landscape - Jul 28, 2025, 7:18:02 PM"), 0xca8b_4a88);
    assert_ne!(label_hash("Video - Landscape - Jul 28, 2025, 7:18:02 PM"), label_hash("Video - Landscape - Jul 28, 2025, 7:18:03 PM"));
}

#[test]
fn an_item_is_ticked_on_its_local_day_at_a_real_offset() {
    // taken 02:50:11 UTC, shown at 06:50:11 four hours east
    let mut m = Matcher::new(vec![w("2023-11-26T02:50:11Z", "video")], &east4);
    assert_eq!(m.next_day(), Some(d("2023-11-26")));
    let grid = labels(&[
        "Photo - Portrait - Nov 26, 2023, 6:50:11 AM",  // a photo, not the video
        "Video - Landscape - Nov 26, 2023, 6:50:12 AM", // a second off
        "Video - Landscape - Nov 26, 2023, 6:57:11 AM", // 4h07: no zone is
        "Video - Landscape - Nov 26, 2023, 6:50:11 AM",
    ]);
    assert_eq!(m.choose(d("2023-11-26"), &grid), labels(&["Video - Landscape - Nov 26, 2023, 6:50:11 AM"]));
    assert_eq!((m.found(), m.next_day()), (1, None));
    assert!(m.left().is_empty());
}

#[test]
fn an_item_past_midnight_locally_is_looked_for_on_the_next_day() {
    // 21:30 UTC is 01:30 the next day four hours east
    let mut m = Matcher::new(vec![w("2025-07-28T21:30:05Z", "photo")], &|_| 0);
    assert_eq!(m.next_day(), Some(d("2025-07-28")));
    assert!(m.choose(d("2025-07-28"), &labels(&["Photo - Portrait - Jul 28, 2025, 9:00:00 PM"])).is_empty());
    assert_eq!(m.next_day(), Some(d("2025-07-29")), "an offset east can put it on the next day");
    let picked = m.choose(d("2025-07-29"), &labels(&["Photo - Portrait - Jul 29, 2025, 1:30:05 AM"]));
    assert_eq!(picked.len(), 1);
}

#[test]
fn the_mac_zone_is_the_first_guess_for_the_day() {
    let m = Matcher::new(vec![w("2025-07-28T21:30:05Z", "photo")], &east4);
    assert_eq!(m.next_day(), Some(d("2025-07-29")));
}

#[test]
fn two_equal_tiles_tick_neither_and_the_tally_settles_later_ones() {
    // the item at 10:00:00 UTC fits 14:00:00 (+4) and 15:30:00 (+5:30) equally: neither
    let mut m = Matcher::new(vec![w("2024-01-05T10:00:00Z", "photo")], &|_| 0);
    let grid = labels(&["Photo - Square - Jan 5, 2024, 2:00:00 PM", "Photo - Square - Jan 5, 2024, 3:30:00 PM"]);
    assert!(m.choose(d("2024-01-05"), &grid).is_empty());
    assert_eq!((m.found(), m.ambiguous.len(), m.next_day()), (0, 1, None));
    assert_eq!(m.left(), ["2024-01-05T10:00:00Z"]);

    // with a clear item at +4 the same day, the doubtful one takes +4 too
    let mut m = Matcher::new(vec![w("2024-01-05T10:00:00Z", "photo"), w("2024-01-05T08:11:12Z", "video")], &|_| 0);
    let mut grid = grid.clone();
    grid.push("Video - Landscape - Jan 5, 2024, 12:11:12 PM".into());
    let picked = m.choose(d("2024-01-05"), &grid);
    assert_eq!(picked, labels(&["Video - Landscape - Jan 5, 2024, 12:11:12 PM", "Photo - Square - Jan 5, 2024, 2:00:00 PM"]));
    assert_eq!((m.found(), m.ambiguous.len()), (2, 0));
}

#[test]
fn a_tile_goes_to_one_item_only_and_an_unticked_one_is_sought_again() {
    let mut m = Matcher::new(vec![w("2024-01-05T10:00:00Z", "photo"), w("2024-01-05T10:00:00Z", "photo")], &east4);
    let grid = labels(&["Photo - Portrait - Jan 5, 2024, 2:00:00 PM"]);
    assert_eq!(m.choose(d("2024-01-05"), &grid).len(), 1);
    assert_eq!(m.found(), 1);
    m.untick("Photo - Portrait - Jan 5, 2024, 2:00:00 PM");
    assert_eq!(m.found(), 0);
    assert!(m.next_day().is_some(), "both are looked for again on another day");
}

#[test]
fn items_are_given_up_after_three_days() {
    let mut m = Matcher::new(vec![w("2024-06-15T11:00:00Z", "video")], &|_| 0);
    let mut days = 0;
    while let Some(day) = m.next_day() {
        assert!(m.choose(day, &[]).is_empty());
        days += 1;
        assert!(days <= MAX_TRIES, "a day loop must end");
    }
    assert_eq!(days, 3, "11:00 UTC can be on the day before (−12 h), the day, or the day after (+14 h)");
    assert_eq!(m.left().len(), 1);
}

#[test]
fn hostile_input_is_bounded() {
    let many: Vec<Wanted> = (0..1000).map(|_| w("2024-06-15T12:00:00Z", "video")).collect();
    assert_eq!(Matcher::new(many, &|_| 0).len(), MAX_ITEMS);
    // a zone that does not exist is not a guess
    let m = Matcher::new(vec![w("2024-06-15T23:59:59Z", "photo")], &|_| 7);
    assert_eq!(m.next_day(), Some(d("2024-06-15")));
    // the latest and earliest times chrono holds do not overflow
    let edge = Wanted { taken: DateTime::<Utc>::MAX_UTC, video: false };
    let mut m = Matcher::new(vec![edge, Wanted { taken: DateTime::<Utc>::MIN_UTC, video: true }], &|_| 840);
    while let Some(day) = m.next_day() {
        m.choose(day, &labels(&["Photo - Portrait - Jan 5, 2024, 2:00:00 PM", "garbage"]));
    }
    assert_eq!(m.left().len(), 2);
}
