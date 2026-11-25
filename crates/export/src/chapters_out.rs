//! A sequence's chapter markers as the chapter list of an MP4 / MOV / M4A export.
//!
//! The muxer writes the list both as a QuickTime chapter track and as a Nero `chpl` list
//! ([`filmcraft_isobmff::chapters`]), so QuickTime Player, Apple devices, VLC and ffprobe list
//! the chapters. Only markers of kind Chapter count; their names are the chapter titles.

use filmcraft_isobmff::Chapter;
use filmcraft_project::{Marker, MarkerKind};
use filmcraft_time::Tick;

/// The movie timescale the exports write ([`filmcraft_isobmff::WriterOptions::new`]'s default,
/// one tick a millisecond): chapter starts are in it.
const MOVIE_TICKS_PER_SECOND: i64 = 1000;

/// The chapter list of an export of `start..end` (sequence time): every chapter marker inside it,
/// its start counted from `start` in milliseconds, and the chapter still running at `start` (the
/// last chapter marker before it) at zero, unless a marker starts there. The muxer sorts the list,
/// keeps one chapter per start, drops what lies past the end of the file and keeps at most
/// [`filmcraft_isobmff::MAX_CHAPTERS`].
pub(crate) fn chapters(markers: &[Marker], start: Tick, end: Tick) -> Vec<Chapter> {
    let at = |t: Tick| u64::try_from((t - start).to_rational_round(1, MOVIE_TICKS_PER_SECOND)).unwrap_or(0);
    let chapter_markers = || markers.iter().filter(|m| m.kind == MarkerKind::Chapter);
    let mut list: Vec<Chapter> =
        chapter_markers().filter(|m| m.start >= start && m.start < end).map(|m| Chapter { start: at(m.start), title: m.name.clone() }).collect();
    if !list.iter().any(|c| c.start == 0)
        && let Some(running) = chapter_markers().filter(|m| m.start < start).max_by_key(|m| m.start)
    {
        list.push(Chapter { start: 0, title: running.name.clone() });
    }
    list
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_project::{Label, MarkerId};
    use filmcraft_time::TICKS_PER_SECOND;

    fn marker(seconds_x10: i64, kind: MarkerKind, name: &str) -> Marker {
        Marker {
            id: MarkerId(0),
            start: Tick(seconds_x10 * TICKS_PER_SECOND / 10),
            duration: Tick::ZERO,
            name: name.into(),
            comment: String::new(),
            kind,
            color: Label::Violet,
        }
    }

    fn s(seconds_x10: i64) -> Tick {
        Tick(seconds_x10 * TICKS_PER_SECOND / 10)
    }

    fn list(v: &[Chapter]) -> Vec<(u64, &str)> {
        v.iter().map(|c| (c.start, c.title.as_str())).collect()
    }

    #[test]
    fn chapter_markers_inside_the_range_count_from_its_start() {
        let m = [
            marker(0, MarkerKind::Chapter, "Intro"),
            marker(15, MarkerKind::Comment, "note"),
            marker(25, MarkerKind::Chapter, "Middle"),
            marker(40, MarkerKind::Chapter, "after the end"),
        ];
        assert_eq!(list(&chapters(&m, s(0), s(40))), [(0, "Intro"), (2500, "Middle")]);
        // no chapter markers: no chapters
        assert!(chapters(&[marker(5, MarkerKind::Comment, "x")], s(0), s(40)).is_empty());
        assert!(chapters(&[], s(0), s(40)).is_empty());
    }

    #[test]
    fn the_chapter_running_at_the_start_begins_the_list() {
        let m = [marker(0, MarkerKind::Chapter, "One"), marker(10, MarkerKind::Chapter, "Two"), marker(30, MarkerKind::Chapter, "Three")];
        // an export from 2 s: "Two" is running, "Three" starts 1 s in
        assert_eq!(list(&chapters(&m, s(20), s(40))), [(1000, "Three"), (0, "Two")]);
        // a marker exactly at the start wins over the running one
        assert_eq!(list(&chapters(&m, s(10), s(40))), [(0, "Two"), (2000, "Three")]);
    }

    #[test]
    fn hostile_times_do_not_overflow() {
        let m = [marker(0, MarkerKind::Chapter, "a"), Marker { start: Tick::MAX, ..marker(0, MarkerKind::Chapter, "far") }];
        assert_eq!(list(&chapters(&m, Tick::MIN, Tick::MAX)).len(), 1);
        let m = [Marker { start: Tick::MIN, ..marker(0, MarkerKind::Chapter, "early") }];
        assert_eq!(list(&chapters(&m, Tick::MAX, Tick::MAX)), [(0, "early")]);
    }
}
