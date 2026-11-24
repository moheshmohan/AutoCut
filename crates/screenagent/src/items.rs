//! Picking exact items — ones a catalogue found, say by what they look like — in Google's picker,
//! which has no search by name or id. What it does show (live, [`crate::dom`]): every media tile's
//! checkbox is labelled with its kind, its shape and the second it was taken, in the local time it
//! was taken in (`Video - Landscape - Jul 28, 2025, 7:18:02 PM`), and a date typed into its search
//! shows that day's items. A catalogue knows when each item was taken in UTC.
//!
//! So each wanted item is searched by its day and ticked by its tile: a tile is one of an item's
//! candidates when it has the item's kind and its local time is the item's UTC time moved by a
//! real time-zone offset ([`OFFSETS`]) to the second. Another tile of the day can match an item
//! that way by chance (about one in 2,300 per tile), so an item takes the candidate whose offset
//! the run has already seen most ([`Matcher`] keeps the tally; the Mac's own zone is each item's
//! first guess), an item with two equally likely candidates is not ticked at all, and an item
//! whose tile is not on the day searched is looked for on the day next to it.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, Utc};

/// The most items one pick asks for.
pub const MAX_ITEMS: usize = 200;
/// Days an item is looked for on before it is given up (its UTC day and the two next to it).
const MAX_TRIES: usize = 3;

/// The real time-zone offsets, in minutes east of UTC (whole hours from −12 to +14, and the half-
/// and three-quarter-hour zones in use).
pub const OFFSETS: [i32; 38] = [
    -720, -660, -600, -570, -540, -480, -420, -360, -300, -240, -210, -180, -120, -60, 0, 60, 120, 180, 210, 240, 270, 300, 330, 345, 360, 390, 420, 480, 525,
    540, 570, 600, 630, 660, 720, 765, 780, 840,
];

/// One item to pick: when it was taken (UTC) and whether it is a video.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wanted {
    pub taken: DateTime<Utc>,
    pub video: bool,
}

impl Wanted {
    /// From a catalogue's RFC 3339 time (`2024-10-02T06:54:35Z`, fractions of a second dropped:
    /// the picker shows whole seconds) and its kind (`video`, else a photo).
    pub fn parse(captured: &str, kind: &str) -> Option<Self> {
        let taken = DateTime::parse_from_rfc3339(captured.trim()).ok()?.with_timezone(&Utc);
        let taken = taken.checked_sub_signed(Duration::nanoseconds(i64::from(taken.timestamp_subsec_nanos())))?;
        Some(Self { taken, video: kind.trim().eq_ignore_ascii_case("video") })
    }

    /// The local time it shows as in a zone `offset` minutes east of UTC.
    fn local(&self, offset: i32) -> Option<NaiveDateTime> {
        self.taken.naive_utc().checked_add_signed(Duration::minutes(i64::from(offset)))
    }

    /// Its RFC 3339 time, as a catalogue gave it (whole seconds).
    pub fn rfc3339(&self) -> String {
        self.taken.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
    }
}

/// A tile's kind and local time, from its label ([`crate::dom::tile_label`]): `Video - Landscape -
/// Jul 28, 2025, 7:18:02 PM` → (video, 2025-07-28 19:18:02).
pub fn tile_time(label: &str) -> Option<(bool, NaiveDateTime)> {
    let kind = label.split(" - ").next()?;
    let when = label.rsplit(" - ").next()?;
    let when: String = when.replace(['\u{202f}', '\u{a0}'], " ").split_whitespace().collect::<Vec<_>>().join(" ");
    // the month as its first three letters (`Sept.` and `September` read as `Sep`)
    let (month, rest) = when.split_once(' ')?;
    let month: String = month.chars().take(3).collect();
    let at = NaiveDateTime::parse_from_str(&format!("{month} {rest}"), "%b %d, %Y, %I:%M:%S %p").ok()?;
    Some((kind.trim() == "Video", at))
}

/// What the picker's search is typed as for a day: `October 30 2026`.
pub fn day_text(day: NaiveDate) -> String {
    day.format("%B %-d %Y").to_string()
}

/// The 32-bit FNV-1a hash of a label's UTF-16 code units — what the page script computes for each
/// tile's label ([`crate::dom::tick_by_hash`]), so only numbers ever go into a script.
pub fn label_hash(label: &str) -> u32 {
    label.encode_utf16().fold(0x811c_9dc5u32, |h, unit| (h ^ u32::from(unit)).wrapping_mul(0x0100_0193))
}

/// The offset (minutes) at which `label`'s tile could be `want`, if any.
fn offset_of(want: &Wanted, label: &str) -> Option<i32> {
    let (video, local) = tile_time(label)?;
    if video != want.video {
        return None;
    }
    let minutes = local.signed_duration_since(want.taken.naive_utc());
    let secs = minutes.num_seconds();
    if secs % 60 != 0 {
        return None;
    }
    let offset = i32::try_from(secs / 60).ok()?;
    OFFSETS.contains(&offset).then_some(offset)
}

/// Which tile is which item, over a whole pick: the offsets matched so far, the day searches each
/// item had, the tile each one got.
#[derive(Debug, Clone)]
pub struct Matcher {
    wanted: Vec<Wanted>,
    /// Each item's own first guess: the Mac's zone when it was taken (minutes).
    home: Vec<i32>,
    found: Vec<Option<String>>,
    tried: Vec<BTreeSet<NaiveDate>>,
    tally: BTreeMap<i32, u32>,
    /// Items not ticked because two tiles fitted them equally well.
    pub ambiguous: BTreeSet<usize>,
}

impl Matcher {
    /// `home(t)`: the Mac's offset (minutes) at `t`, each item's first guess.
    pub fn new(wanted: Vec<Wanted>, home: &dyn Fn(DateTime<Utc>) -> i32) -> Self {
        let wanted: Vec<Wanted> = wanted.into_iter().take(MAX_ITEMS).collect();
        let n = wanted.len();
        let home = wanted.iter().map(|w| home(w.taken)).map(|o| if OFFSETS.contains(&o) { o } else { 0 }).collect();
        Self { wanted, home, found: vec![None; n], tried: vec![BTreeSet::new(); n], tally: BTreeMap::new(), ambiguous: BTreeSet::new() }
    }

    pub fn len(&self) -> usize {
        self.wanted.len()
    }

    pub fn is_empty(&self) -> bool {
        self.wanted.is_empty()
    }

    /// Items with a tile.
    pub fn found(&self) -> usize {
        self.found.iter().filter(|f| f.is_some()).count()
    }

    /// The offset the run has seen most, if any.
    fn best(&self) -> Option<i32> {
        self.tally.iter().max_by_key(|(o, n)| (**n, std::cmp::Reverse(o.abs()))).map(|(o, _)| *o)
    }

    /// The days item `i` may be on, most likely first: the run's best offset, its own zone, UTC,
    /// then any day an offset can put it on.
    fn days(&self, i: usize) -> Vec<NaiveDate> {
        let Some(w) = self.wanted.get(i) else { return Vec::new() };
        let mut guesses: Vec<i32> = self.best().into_iter().collect();
        guesses.extend(self.home.get(i).copied());
        guesses.push(0);
        guesses.extend(OFFSETS);
        let mut out: Vec<NaiveDate> = Vec::new();
        for o in guesses {
            if let Some(d) = w.local(o).map(|t| t.date()).filter(|d| !out.contains(d)) {
                out.push(d);
            }
        }
        out
    }

    fn pending(&self, i: usize) -> bool {
        self.found.get(i).is_some_and(Option::is_none) && !self.ambiguous.contains(&i) && self.tried.get(i).is_some_and(|t| t.len() < MAX_TRIES)
    }

    /// The next day to search: the most likely untried day of the first item still without a tile.
    pub fn next_day(&self) -> Option<NaiveDate> {
        (0..self.wanted.len()).filter(|&i| self.pending(i)).find_map(|i| self.days(i).into_iter().find(|d| !self.tried.get(i).is_some_and(|t| t.contains(d))))
    }

    /// Take the tiles of `day`'s whole grid (labels, in any order): every item still without a tile
    /// that could be on that day counts it tried, and gets its tile when one fits clearly. Answers
    /// the labels to tick.
    pub fn choose(&mut self, day: NaiveDate, labels: &[String]) -> Vec<String> {
        let taken: BTreeSet<&String> = self.found.iter().flatten().collect();
        let free: Vec<&String> = labels.iter().filter(|l| !taken.contains(l)).collect();
        let mut fits: Vec<(usize, Vec<(String, i32)>)> = Vec::new();
        for i in 0..self.wanted.len() {
            if !self.pending(i) || !self.days(i).contains(&day) {
                continue;
            }
            if let Some(t) = self.tried.get_mut(i) {
                t.insert(day);
            }
            let Some(w) = self.wanted.get(i) else { continue };
            let c: Vec<(String, i32)> =
                free.iter().filter_map(|l| offset_of(w, l).filter(|o| w.local(*o).is_some_and(|t| t.date() == day)).map(|o| ((*l).clone(), o))).collect();
            if !c.is_empty() {
                fits.push((i, c));
            }
        }
        // the clear ones first, so their offsets weigh in for the rest
        fits.sort_by_key(|(_, c)| c.len());
        let mut tick: Vec<String> = Vec::new();
        for (i, cands) in fits {
            let home = self.home.get(i).copied();
            let score = |o: i32| self.tally.get(&o).copied().unwrap_or(0).saturating_mul(2).saturating_add(u32::from(home == Some(o)));
            let cands: Vec<(String, i32)> = cands.into_iter().filter(|(l, _)| !tick.contains(l)).collect();
            let top = cands.iter().map(|(_, o)| score(*o)).max().unwrap_or(0);
            let best: Vec<&(String, i32)> = cands.iter().filter(|(_, o)| score(*o) == top).collect();
            match best.as_slice() {
                [(label, o)] if top > 0 || cands.len() == 1 => {
                    *self.tally.entry(*o).or_insert(0) += 1;
                    if let Some(f) = self.found.get_mut(i) {
                        *f = Some(label.clone());
                    }
                    tick.push(label.clone());
                }
                [] => {}
                _ => {
                    self.ambiguous.insert(i);
                }
            }
        }
        tick
    }

    /// A tile chosen for an item that then would not tick: the item is looked for again.
    pub fn untick(&mut self, label: &str) {
        for f in &mut self.found {
            if f.as_deref() == Some(label) {
                *f = None;
            }
        }
    }

    /// The items with no tile, as their RFC 3339 times.
    pub fn left(&self) -> Vec<String> {
        self.wanted.iter().zip(&self.found).filter(|(_, f)| f.is_none()).map(|(w, _)| w.rfc3339()).collect()
    }
}

#[cfg(test)]
#[path = "items_tests.rs"]
mod tests;
