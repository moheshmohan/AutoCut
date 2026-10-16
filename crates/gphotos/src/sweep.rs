//! A whole stretch of a big library in a few picks: the searches that sweep it, month by month
//! ("videos October 2022") or day by day within each month ("videos October 13 2022").
//!
//! Google offers apps no listing of a library, only its picker, whose own search understands
//! dates. A sweep is the list of date searches covering `from..to`; the caller types them into
//! as few picking sessions as the item cap allows (`photos pick --sweep 2017-10..2025-12`),
//! carrying over whatever one session could not take into the next.

/// The most searches one sweep may generate (a century of months; about eight years of days).
pub const MAX_QUERIES: usize = 3000;
const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const YEARS: std::ops::RangeInclusive<i32> = 1900..=2100;

/// A calendar month.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Month {
    pub year: i32,
    /// 1–12.
    pub month: u32,
}

impl Month {
    fn next(self) -> Self {
        if self.month >= 12 { Month { year: self.year.saturating_add(1), month: 1 } } else { Month { year: self.year, month: self.month + 1 } }
    }

    fn name(self) -> &'static str {
        MONTHS.get(self.month.saturating_sub(1) as usize).copied().unwrap_or("January")
    }
}

/// One `YYYY` or `YYYY-MM`; `end` picks December for a bare year at the end of a range.
fn parse_month(s: &str, end: bool) -> Result<Month, String> {
    let s = s.trim();
    let (y, m) = match s.split_once('-') {
        Some((y, m)) => (y, Some(m)),
        None => (s, None),
    };
    let year: i32 = y.trim().parse().ok().filter(|y| YEARS.contains(y)).ok_or_else(|| format!("`{s}` is not a year (1900–2100) or a month like 2022-10"))?;
    let month = match m {
        Some(m) => m.trim().parse::<u32>().ok().filter(|m| (1..=12).contains(m)).ok_or_else(|| format!("`{s}`: the month is 01–12"))?,
        None if end => 12,
        None => 1,
    };
    Ok(Month { year, month })
}

/// A `--sweep` range: `2017-10..2025-12`, `2019..2021` (whole years), `2022` or `2022-10`
/// (just that year or month). The end may not come before the start.
pub fn parse_months(spec: &str) -> Result<(Month, Month), String> {
    let spec = spec.trim();
    let (from, to) = match spec.split_once("..") {
        Some((a, b)) => (parse_month(a, false)?, parse_month(b, true)?),
        None => (parse_month(spec, false)?, parse_month(spec, true)?),
    };
    if to < from {
        return Err(format!("`{spec}` ends before it starts"));
    }
    Ok((from, to))
}

/// A `--month-days` range: `10-21`, or one day `13` (each 1–31, the first not after the last).
pub fn parse_days(spec: &str) -> Result<(u32, u32), String> {
    let day = |s: &str| s.trim().parse::<u32>().ok().filter(|d| (1..=31).contains(d)).ok_or_else(|| format!("`{}` is not a day of the month (1–31)", s.trim()));
    let (a, b) = match spec.split_once('-') {
        Some((a, b)) => (day(a)?, day(b)?),
        None => (day(spec)?, day(spec)?),
    };
    if b < a {
        return Err(format!("`{}`: the days run from the first to the last", spec.trim()));
    }
    Ok((a, b))
}

/// Days in `month` of `year` (Gregorian).
pub fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// The search for a month's videos: `videos October 2022`.
pub fn month_query(m: Month) -> String {
    format!("videos {} {}", m.name(), m.year)
}

/// The search for a day's videos: `videos October 13 2022`.
pub fn day_query(m: Month, day: u32) -> String {
    format!("videos {} {day} {}", m.name(), m.year)
}

/// The searches sweeping `from..=to`: one per month, or with `days` one per day of that range
/// in each month (days a month does not have are skipped). More than [`MAX_QUERIES`] is an
/// error that says how to narrow it.
pub fn queries(from: Month, to: Month, days: Option<(u32, u32)>) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut m = from;
    while m <= to {
        match days {
            None => out.push(month_query(m)),
            Some((a, b)) => out.extend((a..=b.min(days_in_month(m.year, m.month))).map(|d| day_query(m, d))),
        }
        if out.len() > MAX_QUERIES {
            return Err(format!("that sweep is more than {MAX_QUERIES} searches; sweep a shorter stretch (or fewer days a month) at a time"));
        }
        m = m.next();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(year: i32, month: u32) -> Month {
        Month { year, month }
    }

    #[test]
    fn sweep_ranges_parse() {
        assert_eq!(parse_months("2017-10..2025-12"), Ok((m(2017, 10), m(2025, 12))));
        assert_eq!(parse_months(" 2019..2021 "), Ok((m(2019, 1), m(2021, 12))));
        assert_eq!(parse_months("2022"), Ok((m(2022, 1), m(2022, 12))));
        assert_eq!(parse_months("2022-03"), Ok((m(2022, 3), m(2022, 3))));
        assert_eq!(parse_months("2022-3..2022-04"), Ok((m(2022, 3), m(2022, 4))));
        for bad in ["", "..", "2025..2017", "2022-13", "2022-00", "1800", "abcd", "2022-10..", "..2022", "2022-10-01", "99999999999"] {
            assert!(parse_months(bad).is_err(), "{bad:?}");
        }
        assert_eq!(parse_days("10-21"), Ok((10, 21)));
        assert_eq!(parse_days(" 13 "), Ok((13, 13)));
        for bad in ["", "0", "32", "21-10", "x", "1-", "-5", "10-40"] {
            assert!(parse_days(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn months_and_days_become_searches() {
        let q = queries(m(2017, 11), m(2018, 2), None).unwrap();
        assert_eq!(q, ["videos November 2017", "videos December 2017", "videos January 2018", "videos February 2018"]);
        // eight years and three months of months
        assert_eq!(queries(m(2017, 10), m(2025, 12), None).unwrap().len(), 99);
        // days a month does not have are skipped; leap years are leap years
        let q = queries(m(2023, 2), m(2024, 2), Some((28, 31))).unwrap();
        assert_eq!(q.first().map(String::as_str), Some("videos February 28 2023"));
        assert_eq!(q.iter().filter(|s| s.contains("February 2") && s.ends_with("2023")).count(), 1);
        assert!(q.contains(&"videos February 29 2024".to_string()) && !q.iter().any(|s| s.contains("February 30")));
        assert!(q.contains(&"videos April 30 2023".to_string()) && !q.contains(&"videos April 31 2023".to_string()));
        assert_eq!(queries(m(2022, 10), m(2022, 10), Some((13, 13))).unwrap(), ["videos October 13 2022"]);
        assert_eq!((days_in_month(1900, 2), days_in_month(2000, 2), days_in_month(2023, 12)), (28, 29, 31));
    }

    #[test]
    fn a_sweep_too_big_is_refused_not_built() {
        let e = queries(m(1900, 1), m(2100, 12), Some((1, 31))).unwrap_err();
        assert!(e.contains("more than") && e.contains("shorter"), "{e}");
        assert!(queries(m(1900, 1), m(2100, 12), None).unwrap().len() <= MAX_QUERIES);
    }
}
