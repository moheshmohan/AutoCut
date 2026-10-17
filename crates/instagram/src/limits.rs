//! Instagram's publishing limits, as this machine spent them: posts published and containers
//! made in the rolling 24 hours (`<data dir>/instagram/limits.json`), and what Instagram's own
//! `content_publishing_limit` said last.
//!
//! Meta's content-publishing guide says an account may publish **100 posts** in a 24-hour moving
//! window (the `content_publishing_limit` reference currently says 50, so Instagram's own
//! `config.quota_total` wins whenever it was read); FilmCraft also counts the **400 containers**
//! a 24-hour window allows. Other apps and devices spend from the same allowance, so this count
//! never replaces Instagram's answer: it stops a call only when this machine alone has used it up.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::write_private;

/// The rolling window.
pub const WINDOW_SECS: u64 = 24 * 60 * 60;
/// Posts in the window (Meta's content-publishing guide), until Instagram says otherwise.
pub const DEFAULT_POSTS: u64 = 100;
/// Containers in the window.
pub const CONTAINERS: u64 = 400;
/// The most timestamps kept per list (more than a window can hold).
const MAX_KEPT: usize = 1000;

/// What Instagram's `content_publishing_limit` said.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Reported {
    pub quota_total: u64,
    pub quota_usage: u64,
    pub quota_duration: u64,
    /// Unix seconds it was read.
    pub checked_at: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Ledger {
    /// Unix seconds of each post published from here.
    pub posts: Vec<u64>,
    /// Unix seconds of each container made from here.
    pub containers: Vec<u64>,
    pub reported: Option<Reported>,
}

impl Ledger {
    fn prune(&mut self, now: u64) {
        let since = now.saturating_sub(WINDOW_SECS);
        for list in [&mut self.posts, &mut self.containers] {
            list.retain(|t| *t > since && *t <= now.saturating_add(60));
            list.sort_unstable();
            if list.len() > MAX_KEPT {
                let cut = list.len() - MAX_KEPT;
                list.drain(..cut);
            }
        }
    }

    /// The posts allowed in the window: Instagram's `quota_total` when it said one (and it is
    /// sane), else [`DEFAULT_POSTS`].
    pub fn post_quota(&self) -> u64 {
        self.reported.as_ref().map(|r| r.quota_total).filter(|q| (1..=10_000).contains(q)).unwrap_or(DEFAULT_POSTS)
    }
}

/// Seconds until the oldest of `list` leaves the window (when a slot frees up).
fn next_slot(list: &[u64], now: u64) -> u64 {
    list.first().map(|t| t.saturating_add(WINDOW_SECS).saturating_sub(now)).unwrap_or(0)
}

/// "3 h 20 min", "12 min", "under a minute".
pub fn describe_wait(secs: u64) -> String {
    let (h, m) = (secs / 3600, (secs % 3600) / 60);
    match (h, m) {
        (0, 0) => "under a minute".into(),
        (0, m) => format!("{m} min"),
        (h, 0) => format!("{h} h"),
        (h, m) => format!("{h} h {m} min"),
    }
}

/// The ledger file, with a lock so concurrent jobs in one process don't lose a count.
pub struct LimitBook {
    path: PathBuf,
    lock: Mutex<()>,
}

impl LimitBook {
    pub fn new(dir: &Path) -> Self {
        Self { path: dir.join("limits.json"), lock: Mutex::new(()) }
    }

    /// The ledger now (pruned to the window). A missing or damaged file is an empty ledger.
    pub fn load(&self, now: u64) -> Ledger {
        let mut l: Ledger = std::fs::read(&self.path).ok().filter(|b| b.len() < 1 << 20).and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        l.prune(now);
        l
    }

    fn update(&self, now: u64, f: impl FnOnce(&mut Ledger)) {
        let _g = self.lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut l = self.load(now);
        f(&mut l);
        l.prune(now);
        // a count that cannot be written is only an estimate lost, never a failed publish
        let _ = write_private(&self.path, &serde_json::to_vec_pretty(&l).unwrap_or_default());
    }

    pub fn record_container(&self, now: u64) {
        self.update(now, |l| l.containers.push(now));
    }

    pub fn record_post(&self, now: u64) {
        self.update(now, |l| l.posts.push(now));
    }

    pub fn record_reported(&self, now: u64, r: Reported) {
        self.update(now, |l| l.reported = Some(r));
    }

    /// Whether a container may be made now (this machine's count only).
    pub fn check_container(&self, now: u64) -> Result<(), String> {
        let l = self.load(now);
        if l.containers.len() as u64 >= CONTAINERS {
            return Err(format!(
                "{} Reel containers were made from this machine in the last 24 hours, the most Instagram allows (400); the next one can be made in {}",
                l.containers.len(),
                describe_wait(next_slot(&l.containers, now))
            ));
        }
        Ok(())
    }

    /// Whether a post may be published now: this machine's count, or Instagram's count just
    /// read (`fresh`: it covers every app) when there is one.
    pub fn check_post(&self, now: u64, fresh: Option<&Reported>) -> Result<(), String> {
        let l = self.load(now);
        let quota = fresh.map(|r| r.quota_total).filter(|q| (1..=10_000).contains(q)).unwrap_or_else(|| l.post_quota());
        let used = fresh.map_or(0, |r| r.quota_usage).max(l.posts.len() as u64);
        if used >= quota {
            let wait = next_slot(&l.posts, now);
            let when = if wait > 0 { format!("the next slot opens in {}", describe_wait(wait)) } else { "try again in a few hours".into() };
            return Err(format!("the account has published {used} posts through the API in the last 24 hours, the most Instagram allows ({quota}); {when}"));
        }
        Ok(())
    }

    /// The counts as JSON (`instagram.account`, the publish answer).
    pub fn to_json(&self, now: u64) -> Value {
        let l = self.load(now);
        json!({
            "window": "24 h (rolling)",
            "posts": l.posts.len(), "postQuota": l.post_quota(),
            "containers": l.containers.len(), "containerQuota": CONTAINERS,
            "nextPostSlotIn": (l.posts.len() as u64 >= l.post_quota()).then(|| describe_wait(next_slot(&l.posts, now))),
            "instagram": l.reported.as_ref().map(|r| json!({"quotaUsage": r.quota_usage, "quotaTotal": r.quota_total, "quotaDuration": r.quota_duration, "checkedAt": r.checked_at})),
            "note": "this machine's count; other apps on the account spend from the same allowance (Instagram's content_publishing_limit is authoritative)",
        })
    }
}

/// `content_publishing_limit`'s answer (`{"data":[{"quota_usage":2,"config":{"quota_total":50,"quota_duration":86400}}]}`),
/// hostile: odd values are left out.
pub fn parse_reported(body: &str, now: u64) -> Option<Reported> {
    let v: Value = serde_json::from_str(body).ok()?;
    let d = v.get("data")?.as_array()?.first()?;
    let num = |v: Option<&Value>| v.and_then(Value::as_u64).filter(|n| *n <= 1_000_000);
    let usage = num(d.get("quota_usage"))?;
    let cfg = d.get("config");
    Some(Reported {
        quota_usage: usage,
        quota_total: num(cfg.and_then(|c| c.get("quota_total"))).unwrap_or(0),
        quota_duration: num(cfg.and_then(|c| c.get("quota_duration"))).unwrap_or(0),
        checked_at: now,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("filmcraft-ig-limits-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_window_rolls_and_limits_say_when() {
        let d = dir("roll");
        let b = LimitBook::new(&d);
        let t0 = 1_000_000;
        for i in 0..100 {
            b.record_post(t0 + i);
        }
        let e = b.check_post(t0 + 200, None).unwrap_err();
        assert!(e.contains("100 posts") && e.contains("next slot opens in 23 h 56 min"), "{e}");
        assert!(b.check_post(t0 + WINDOW_SECS + 1, None).is_ok(), "the oldest left the window");
        for _ in 0..400 {
            b.record_container(t0);
        }
        assert!(b.check_container(t0 + 10).unwrap_err().contains("400"));
        assert!(b.check_container(t0 + WINDOW_SECS + 1).is_ok());
        // Instagram's own quota wins
        let later = t0 + WINDOW_SECS + 200;
        let r = Reported { quota_total: 50, quota_usage: 50, quota_duration: 86400, checked_at: later };
        assert!(b.check_post(later, Some(&r)).unwrap_err().contains("(50)"));
        b.record_reported(later, r);
        assert!(b.check_post(later, None).is_ok(), "an old answer does not block by itself");
        let j = b.to_json(later);
        assert_eq!(j["postQuota"], 50);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn damaged_files_and_answers_are_harmless() {
        let d = dir("damaged");
        std::fs::write(d.join("limits.json"), b"{not json").unwrap();
        let b = LimitBook::new(&d);
        assert_eq!(b.load(5), Ledger::default());
        std::fs::write(d.join("limits.json"), br#"{"posts":[1,2,99999999999999],"containers":"x"}"#).unwrap();
        assert_eq!(b.load(5).posts.len(), 0, "a damaged list is an empty ledger");
        assert_eq!(parse_reported(r#"{"data":[{"quota_usage":2,"config":{"quota_total":50,"quota_duration":86400}}]}"#, 9).unwrap().quota_total, 50);
        for odd in ["", "[]", r#"{"data":[]}"#, r#"{"data":[{"quota_usage":-1}]}"#, r#"{"data":[{"quota_usage":"2"}]}"#, r#"{"data":"x"}"#] {
            assert!(parse_reported(odd, 0).is_none(), "{odd}");
        }
        let r = parse_reported(r#"{"data":[{"quota_usage":3,"config":{"quota_total":1e30}}]}"#, 0).unwrap();
        assert_eq!(r.quota_total, 0);
        assert_eq!(Ledger { reported: Some(r), ..Default::default() }.post_quota(), DEFAULT_POSTS, "an odd quota is not trusted");
        assert_eq!(describe_wait(0), "under a minute");
        assert_eq!(describe_wait(7260), "2 h 1 min");
        std::fs::remove_dir_all(&d).ok();
    }
}
