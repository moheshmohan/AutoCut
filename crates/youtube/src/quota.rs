//! YouTube Data API quota, as this machine has spent it today.
//!
//! Google counts quota per Google Cloud project and per Pacific-time day (it resets at midnight
//! in California): 10,000 units for most calls, and a separate bucket of 100 `videos.insert`
//! calls (1 unit each) for uploads. FilmCraft keeps its own count in
//! `<data dir>/youtube/quota.json` so it can say how much is left and when it resets — an
//! estimate: other machines or apps on the same project spend from the same quota, and only
//! Google's `quotaExceeded` answer is authoritative.
//!
//! Costs (developers.google.com/youtube/v3/determine_quota_cost): `channels.list`,
//! `playlists.list`, `videos.list` 1 unit; `playlistItems.insert`, `thumbnails.set` 50 units;
//! `videos.insert` 1 unit in the uploads bucket. Every request costs at least one unit, invalid
//! ones too.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use serde::{Deserialize, Serialize};

/// The general daily quota of a project (Google's default).
pub const DAILY_UNITS: u64 = 10_000;
/// The daily `videos.insert` calls of a project (Google's default).
pub const DAILY_UPLOADS: u64 = 100;
/// The quota file.
pub const FILE: &str = "quota.json";
/// More than the file ever needs.
const MAX_FILE: u64 = 64 * 1024;

/// A call that costs quota.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Call {
    ChannelsList,
    PlaylistsList,
    VideosList,
    VideosInsert,
    PlaylistItemsInsert,
    ThumbnailsSet,
}

impl Call {
    pub fn name(self) -> &'static str {
        match self {
            Call::ChannelsList => "channels.list",
            Call::PlaylistsList => "playlists.list",
            Call::VideosList => "videos.list",
            Call::VideosInsert => "videos.insert",
            Call::PlaylistItemsInsert => "playlistItems.insert",
            Call::ThumbnailsSet => "thumbnails.set",
        }
    }

    /// Units from the general quota, and calls from the uploads bucket.
    pub fn cost(self) -> (u64, u64) {
        match self {
            Call::ChannelsList | Call::PlaylistsList | Call::VideosList => (1, 0),
            Call::PlaylistItemsInsert | Call::ThumbnailsSet => (50, 0),
            Call::VideosInsert => (0, 1),
        }
    }
}

/// One Pacific day's spending.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct QuotaDay {
    /// `YYYY-MM-DD`, Pacific time.
    pub day: String,
    pub units: u64,
    pub uploads: u64,
    pub calls: BTreeMap<String, u64>,
    /// Google said `quotaExceeded` (or `uploadLimitExceeded`) today: what, and when (Unix seconds).
    pub exceeded: Option<(String, u64)>,
}

impl QuotaDay {
    /// As JSON for `youtube.channels` and the CLI: what is spent and left, when it resets.
    pub fn to_json(&self, now: u64) -> serde_json::Value {
        let reset = next_reset(now);
        serde_json::json!({
            "day": self.day, "units": self.units, "unitsLeft": DAILY_UNITS.saturating_sub(self.units),
            "uploads": self.uploads, "uploadsLeft": DAILY_UPLOADS.saturating_sub(self.uploads), "calls": self.calls,
            "exceeded": self.exceeded.as_ref().map(|(what, at)| serde_json::json!({"reason": what, "at": at})),
            "resetsAt": reset, "resetsIn": describe_wait(reset.saturating_sub(now)),
            "note": "FilmCraft's own count on this machine; Google counts per Google Cloud project (other machines and apps spend from it too)",
        })
    }
}

/// The quota file of one data directory.
pub struct QuotaBook {
    path: PathBuf,
    lock: Mutex<()>,
}

impl QuotaBook {
    pub fn new(dir: &Path) -> Self {
        Self { path: dir.join(FILE), lock: Mutex::new(()) }
    }

    /// Today's spending (a file from another day, a missing or damaged file: nothing spent).
    pub fn today(&self) -> QuotaDay {
        self.today_at(crate::now())
    }

    pub fn today_at(&self, now: u64) -> QuotaDay {
        let day = pacific_day(now);
        let read = std::fs::File::open(&self.path).ok().and_then(|f| {
            use std::io::Read as _;
            let mut b = Vec::new();
            f.take(MAX_FILE).read_to_end(&mut b).ok()?;
            serde_json::from_slice::<QuotaDay>(&b).ok()
        });
        match read {
            Some(q) if q.day == day => q,
            _ => QuotaDay { day, ..Default::default() },
        }
    }

    /// Count `call` (best effort: the count is an estimate, and a full disk must not fail an
    /// upload).
    pub fn record(&self, call: Call) {
        self.update(|q| {
            let (units, uploads) = call.cost();
            q.units = q.units.saturating_add(units);
            q.uploads = q.uploads.saturating_add(uploads);
            let n = q.calls.entry(call.name().to_string()).or_insert(0);
            *n = n.saturating_add(1);
        });
    }

    /// Note that Google said the quota (or the channel's upload limit) ran out.
    pub fn exceeded(&self, reason: &str) {
        let now = crate::now();
        self.update(|q| q.exceeded = Some((reason.chars().take(60).collect(), now)));
    }

    fn update(&self, f: impl FnOnce(&mut QuotaDay)) {
        let _guard = self.lock.lock().unwrap_or_else(PoisonError::into_inner);
        let mut q = self.today();
        f(&mut q);
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let tmp = self.path.with_extension("json.tmp");
        if std::fs::write(&tmp, serde_json::to_vec_pretty(&q).unwrap_or_default()).is_ok() {
            let _ = std::fs::rename(&tmp, &self.path);
        }
    }
}

/// "9 h 58 min", "12 min", "less than a minute".
pub fn describe_wait(secs: u64) -> String {
    let (h, m) = (secs / 3600, (secs % 3600) / 60);
    match (h, m) {
        (0, 0) => "less than a minute".into(),
        (0, m) => format!("{m} min"),
        (h, 0) => format!("{h} h"),
        (h, m) => format!("{h} h {m} min"),
    }
}

// -------------------------------------------------------------------------------------------
// Pacific time (America/Los_Angeles: UTC−8, UTC−7 from the second Sunday of March at 02:00 to
// the first Sunday of November at 02:00, the US rule since 2007)
// -------------------------------------------------------------------------------------------

/// Days since 1970-01-01 of a proleptic Gregorian date (H. Hinnant's `days_from_civil`).
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = i64::from(m);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The date of a day number (the inverse of [`days_from_civil`]).
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// The day number of the `n`th Sunday of a month.
fn nth_sunday(y: i64, m: u32, n: i64) -> i64 {
    let first = days_from_civil(y, m, 1);
    // 1970-01-01 was a Thursday: weekday 4 with Sunday 0
    let weekday = (first + 4).rem_euclid(7);
    first + (7 - weekday) % 7 + 7 * (n - 1)
}

/// Pacific time's offset from UTC at Unix second `t`, in seconds.
pub fn pacific_offset(t: i64) -> i64 {
    let (y, _, _) = civil_from_days((t - 8 * 3600).div_euclid(86_400));
    // 02:00 PST = 10:00 UTC; 02:00 PDT = 09:00 UTC
    let start = nth_sunday(y, 3, 2) * 86_400 + 10 * 3600;
    let end = nth_sunday(y, 11, 1) * 86_400 + 9 * 3600;
    if (start..end).contains(&t) { -7 * 3600 } else { -8 * 3600 }
}

/// The Pacific date (`YYYY-MM-DD`) at Unix second `now`.
pub fn pacific_day(now: u64) -> String {
    let t = i64::try_from(now).unwrap_or(i64::MAX / 2);
    let (y, m, d) = civil_from_days((t + pacific_offset(t)).div_euclid(86_400));
    format!("{y:04}-{m:02}-{d:02}")
}

/// The Unix second of the next Pacific midnight after `now` (when Google's quota resets).
pub fn next_reset(now: u64) -> u64 {
    let t = i64::try_from(now).unwrap_or(i64::MAX / 2);
    let local_day = (t + pacific_offset(t)).div_euclid(86_400);
    let midnight_local = (local_day + 1) * 86_400;
    // the offset at that midnight (it differs from now's on the days the clocks change)
    let guess = midnight_local - pacific_offset(t);
    let at = midnight_local - pacific_offset(guess);
    u64::try_from(at).unwrap_or(now)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unix(y: i64, m: u32, d: u32, h: i64, min: i64) -> u64 {
        u64::try_from(days_from_civil(y, m, d) * 86_400 + h * 3600 + min * 60).unwrap()
    }

    #[test]
    fn civil_dates_round_trip() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 3, 1), 11_017);
        for z in [-1_000_000, -1, 0, 1, 11_016, 20_000, 2_000_000] {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z, "{z}");
        }
        // 2026-03-08 and 2026-11-01 are the second and first Sundays
        assert_eq!(civil_from_days(nth_sunday(2026, 3, 2)), (2026, 3, 8));
        assert_eq!(civil_from_days(nth_sunday(2026, 11, 1)), (2026, 11, 1));
        assert_eq!(civil_from_days(nth_sunday(2025, 3, 2)), (2025, 3, 9));
    }

    #[test]
    fn the_quota_day_is_pacific() {
        // 2026-10-17 06:59 UTC is 23:59 PDT on the 16th; 07:00 UTC is midnight of the 17th
        assert_eq!(pacific_day(unix(2026, 10, 17, 6, 59)), "2026-10-16");
        assert_eq!(pacific_day(unix(2026, 10, 17, 7, 0)), "2026-10-17");
        // winter: UTC−8
        assert_eq!(pacific_day(unix(2026, 1, 10, 7, 59)), "2026-01-09");
        assert_eq!(pacific_day(unix(2026, 1, 10, 8, 0)), "2026-01-10");
        // the clocks change at 02:00 local
        assert_eq!(pacific_offset(i64::try_from(unix(2026, 3, 8, 9, 59)).unwrap()), -8 * 3600);
        assert_eq!(pacific_offset(i64::try_from(unix(2026, 3, 8, 10, 0)).unwrap()), -7 * 3600);
        assert_eq!(pacific_offset(i64::try_from(unix(2026, 11, 1, 8, 59)).unwrap()), -7 * 3600);
        assert_eq!(pacific_offset(i64::try_from(unix(2026, 11, 1, 9, 0)).unwrap()), -8 * 3600);
        // resets: the next local midnight
        assert_eq!(next_reset(unix(2026, 10, 17, 12, 0)), unix(2026, 10, 18, 7, 0));
        assert_eq!(next_reset(unix(2026, 1, 10, 9, 0)), unix(2026, 1, 11, 8, 0));
        // the night the clocks go back: midnight of Nov 1 is PDT (07:00 UTC), of Nov 2 PST (08:00 UTC)
        assert_eq!(next_reset(unix(2026, 11, 1, 12, 0)), unix(2026, 11, 2, 8, 0));
        assert_eq!(next_reset(unix(2026, 3, 7, 20, 0)), unix(2026, 3, 8, 8, 0));
        assert_eq!(next_reset(unix(2026, 3, 8, 20, 0)), unix(2026, 3, 9, 7, 0));
        // extreme clocks never panic
        assert!(!pacific_day(u64::MAX).is_empty() && !pacific_day(0).is_empty());
        let _ = next_reset(u64::MAX);
        assert_eq!(describe_wait(0), "less than a minute");
        assert_eq!(describe_wait(3600 * 9 + 58 * 60), "9 h 58 min");
        assert_eq!(describe_wait(7200), "2 h");
        assert_eq!(describe_wait(600), "10 min");
    }

    #[test]
    fn the_book_counts_a_day_and_survives_odd_files() {
        let dir = std::env::temp_dir().join(format!("filmcraft-yt-quota-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let book = QuotaBook::new(&dir);
        assert_eq!(book.today().units, 0);
        book.record(Call::ChannelsList);
        book.record(Call::VideosInsert);
        book.record(Call::ThumbnailsSet);
        book.record(Call::PlaylistItemsInsert);
        let q = book.today();
        assert_eq!((q.units, q.uploads, q.calls.get("videos.insert").copied()), (101, 1, Some(1)));
        book.exceeded("quotaExceeded");
        assert_eq!(book.today().exceeded.unwrap().0, "quotaExceeded");
        let j = book.today().to_json(crate::now());
        assert_eq!((j["unitsLeft"].as_u64(), j["uploadsLeft"].as_u64()), (Some(DAILY_UNITS - 101), Some(99)), "{j}");
        // yesterday's file is a fresh day
        std::fs::write(dir.join(FILE), r#"{"day":"1999-01-01","units":9999}"#).unwrap();
        assert_eq!(book.today().units, 0);
        for odd in ["", "{", "[1]", r#"{"units":-4}"#, r#"{"units":1e99}"#] {
            std::fs::write(dir.join(FILE), odd).unwrap();
            assert_eq!(book.today().units, 0, "{odd}");
            book.record(Call::VideosList);
            assert_eq!(book.today().units, 1, "{odd}");
        }
        std::fs::write(dir.join(FILE), vec![b' '; 200_000]).unwrap();
        assert_eq!(book.today().units, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
