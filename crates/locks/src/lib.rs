//! Advisory locks on shared machine resources, so parallel sessions and agents take turns.
//!
//! Some things on a desktop can't be shared: the **browser** an agent drives with the real mouse
//! and keyboard (the Google Photos picker agent, `photos tidy`, a Claude session driving Chrome),
//! and the **GPU** a large model speaks on (VoxCPM narration). Two users of one at once collide —
//! a tab closed under another session, a typed search that never lands, two 5 GB models in
//! memory. A [`Guard`] from [`acquire`] says "mine until dropped"; others wait or report who has
//! it ([`Holder`]).
//!
//! The lock is the operating system's file lock on `<data dir>/locks/<resource>.lock`
//! (`std::fs::File::try_lock`): it is released when the guard drops *or the process ends*, so a
//! crashed holder never leaves a stale lock. The file also names the holder (pid, since, what)
//! for messages. Where file locks are unsupported (the web build), every acquire succeeds.
//!
//! Ollama is deliberately not a resource here: it queues requests itself, so a lock would only
//! serialize what is already serialized.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// A resource one process uses at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Resource {
    /// The desktop browser driven with the real mouse and keyboard.
    Chrome,
    /// The GPU, for large on-device models.
    Gpu,
}

impl Resource {
    pub const ALL: [Resource; 2] = [Resource::Chrome, Resource::Gpu];

    /// The id used in file names and commands (`chrome`, `gpu`).
    pub fn id(self) -> &'static str {
        match self {
            Resource::Chrome => "chrome",
            Resource::Gpu => "gpu",
        }
    }

    pub fn from_id(s: &str) -> Option<Resource> {
        let s = s.trim().to_ascii_lowercase();
        Self::ALL.into_iter().find(|r| r.id() == s)
    }

    /// For people: "the browser", "the GPU".
    pub fn label(self) -> &'static str {
        match self {
            Resource::Chrome => "the browser",
            Resource::Gpu => "the GPU",
        }
    }
}

/// Who holds a resource, as its lock file says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Holder {
    pub pid: u32,
    /// Unix seconds when it was taken.
    pub since: u64,
    /// What it is doing ("photos pick", "narration", "Claude session: album work").
    pub what: String,
}

impl std::fmt::Display for Holder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ago = now().saturating_sub(self.since);
        write!(f, "{} (pid {}, for {})", self.what, self.pid, duration_words(ago))
    }
}

/// Why a resource could not be taken.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LockError {
    /// Someone else has it (and who, when the lock file says).
    Busy(Resource, Option<Holder>),
    /// The lock file could not be made or opened.
    Io(String),
}

impl std::fmt::Display for LockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LockError::Busy(r, Some(h)) => write!(f, "{} is busy: {h}", r.label()),
            LockError::Busy(r, None) => write!(f, "{} is busy (another process holds its lock)", r.label()),
            LockError::Io(e) => write!(f, "resource lock: {e}"),
        }
    }
}

impl std::error::Error for LockError {}

/// A held resource; dropping it lets the next one in.
#[derive(Debug)]
pub struct Guard {
    file: Option<File>,
    resource: Resource,
}

impl Guard {
    pub fn resource(&self) -> Resource {
        self.resource
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        // say nobody holds it (best effort), then the lock goes with the file
        if let Some(f) = self.file.as_mut() {
            let _ = f.set_len(0);
        }
    }
}

/// Longest "what" kept, characters.
const MAX_WHAT: usize = 200;
/// How often a waiting [`acquire`] tries again.
const POLL: Duration = Duration::from_millis(500);

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn duration_words(secs: u64) -> String {
    match secs {
        0..=59 => format!("{secs} s"),
        60..=3599 => format!("{} min", secs / 60),
        _ => format!("{} h {} min", secs / 3600, (secs % 3600) / 60),
    }
}

/// Where the lock files live: `FILMCRAFT_LOCK_DIR`, else the FilmCraft data directory's
/// `locks/` (macOS `~/Library/Application Support/FilmCraft`, Windows `%APPDATA%\FilmCraft`,
/// elsewhere `$XDG_DATA_HOME/filmcraft` or `~/.local/share/filmcraft`).
pub fn dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("FILMCRAFT_LOCK_DIR").filter(|d| !d.is_empty()) {
        return Some(PathBuf::from(d));
    }
    let home = || std::env::var_os("HOME").filter(|h| !h.is_empty()).map(PathBuf::from);
    let base = if cfg!(target_os = "macos") {
        home().map(|h| h.join("Library/Application Support/FilmCraft"))
    } else if cfg!(windows) {
        std::env::var_os("APPDATA").filter(|a| !a.is_empty()).map(|a| PathBuf::from(a).join("FilmCraft"))
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .filter(|x| !x.is_empty())
            .map(|x| PathBuf::from(x).join("filmcraft"))
            .or_else(|| home().map(|h| h.join(".local/share/filmcraft")))
    };
    base.map(|b| b.join("locks"))
}

fn lock_path(dir: &Path, r: Resource) -> PathBuf {
    dir.join(format!("{}.lock", r.id()))
}

fn open(dir: &Path, r: Resource) -> Result<File, LockError> {
    std::fs::create_dir_all(dir).map_err(|e| LockError::Io(format!("{}: {e}", dir.display())))?;
    let path = lock_path(dir, r);
    OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path).map_err(|e| LockError::Io(format!("{}: {e}", path.display())))
}

/// The holder line: `pid<TAB>since<TAB>what`.
fn parse_holder(text: &str) -> Option<Holder> {
    let line = text.lines().next()?;
    let mut parts = line.splitn(3, '\t');
    let pid = parts.next()?.trim().parse().ok()?;
    let since = parts.next()?.trim().parse().ok()?;
    let what = parts.next().unwrap_or("").trim().chars().take(MAX_WHAT).collect();
    Some(Holder { pid, since, what })
}

fn clean_what(what: &str) -> String {
    let w: String = what.chars().map(|c| if c.is_control() { ' ' } else { c }).take(MAX_WHAT).collect();
    let w = w.split_whitespace().collect::<Vec<_>>().join(" ");
    if w.is_empty() { "unnamed".into() } else { w }
}

/// Take `r` now, or say who has it. `what` names the holder for others' messages.
pub fn try_acquire(r: Resource, what: &str) -> Result<Guard, LockError> {
    match dir() {
        Some(d) => try_acquire_in(&d, r, what),
        // nowhere to keep locks: run unlocked rather than refuse to work
        None => Ok(Guard { file: None, resource: r }),
    }
}

/// [`try_acquire`] waiting up to `wait` for the holder to finish; `waiting` hears who is in
/// the way (once, when the wait starts).
pub fn acquire(r: Resource, what: &str, wait: Duration, waiting: &mut dyn FnMut(&LockError)) -> Result<Guard, LockError> {
    match dir() {
        Some(d) => acquire_in(&d, r, what, wait, waiting),
        None => Ok(Guard { file: None, resource: r }),
    }
}

/// Who holds `r` now (`None`: free, or no lock directory).
pub fn holder(r: Resource) -> Option<Holder> {
    holder_in(&dir()?, r)
}

pub(crate) fn try_acquire_in(dir: &Path, r: Resource, what: &str) -> Result<Guard, LockError> {
    let mut file = open(dir, r)?;
    match file.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            let mut text = String::new();
            let _ = file.read_to_string(&mut text);
            return Err(LockError::Busy(r, parse_holder(&text)));
        }
        // file locks unsupported here (the web build): unlocked
        Err(TryLockError::Error(e)) if e.kind() == std::io::ErrorKind::Unsupported => return Ok(Guard { file: None, resource: r }),
        Err(TryLockError::Error(e)) => return Err(LockError::Io(e.to_string())),
    }
    // ours: say who holds it
    let line = format!("{}\t{}\t{}\n", std::process::id(), now(), clean_what(what));
    let _ = file.set_len(0).and_then(|_| file.seek(SeekFrom::Start(0))).and_then(|_| file.write_all(line.as_bytes())).and_then(|_| file.flush());
    Ok(Guard { file: Some(file), resource: r })
}

pub(crate) fn acquire_in(dir: &Path, r: Resource, what: &str, wait: Duration, waiting: &mut dyn FnMut(&LockError)) -> Result<Guard, LockError> {
    let start = Instant::now();
    let mut told = false;
    loop {
        match try_acquire_in(dir, r, what) {
            Err(e @ LockError::Busy(..)) => {
                if start.elapsed() >= wait {
                    return Err(e);
                }
                if !told {
                    waiting(&e);
                    told = true;
                }
                std::thread::sleep(POLL.min(wait.saturating_sub(start.elapsed())).max(Duration::from_millis(10)));
            }
            other => return other,
        }
    }
}

pub(crate) fn holder_in(dir: &Path, r: Resource) -> Option<Holder> {
    let path = lock_path(dir, r);
    let mut file = OpenOptions::new().read(true).open(&path).ok()?;
    match file.try_lock() {
        // free: nobody holds it (the lock goes again as the file closes)
        Ok(()) => None,
        Err(TryLockError::WouldBlock) => {
            let mut text = String::new();
            file.read_to_string(&mut text).ok()?;
            Some(parse_holder(&text).unwrap_or(Holder { pid: 0, since: 0, what: "unknown".into() }))
        }
        Err(TryLockError::Error(_)) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("filmcraft-locks-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn one_holder_at_a_time_and_others_learn_who() {
        let d = scratch("one");
        let g = try_acquire_in(&d, Resource::Chrome, "photos pick").unwrap();
        // a second taker is told who has it
        match try_acquire_in(&d, Resource::Chrome, "photos tidy") {
            Err(LockError::Busy(Resource::Chrome, Some(h))) => {
                assert_eq!((h.pid, h.what.as_str()), (std::process::id(), "photos pick"));
                assert!(LockError::Busy(Resource::Chrome, Some(h)).to_string().starts_with("the browser is busy: photos pick (pid"));
            }
            other => panic!("expected busy, got {other:?}"),
        }
        assert_eq!(holder_in(&d, Resource::Chrome).map(|h| h.what), Some("photos pick".into()));
        // another resource is independent
        let gpu = try_acquire_in(&d, Resource::Gpu, "narration").unwrap();
        drop(gpu);
        // dropping frees it
        drop(g);
        assert_eq!(holder_in(&d, Resource::Chrome), None);
        let again = try_acquire_in(&d, Resource::Chrome, "photos tidy").unwrap();
        assert_eq!(again.resource(), Resource::Chrome);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn waiting_gets_it_when_the_holder_lets_go_and_gives_up_after_the_wait() {
        let d = scratch("wait");
        let g = try_acquire_in(&d, Resource::Gpu, "first").unwrap();
        // gives up after the wait, having said who is in the way once
        let mut heard = 0;
        let r = acquire_in(&d, Resource::Gpu, "second", Duration::from_millis(120), &mut |_| heard += 1);
        assert!(matches!(r, Err(LockError::Busy(Resource::Gpu, _))));
        assert_eq!(heard, 1);
        // released while waiting: taken
        let d2 = d.clone();
        let t = std::thread::spawn(move || acquire_in(&d2, Resource::Gpu, "third", Duration::from_secs(10), &mut |_| {}).map(|g| g.resource()));
        std::thread::sleep(Duration::from_millis(200));
        drop(g);
        assert_eq!(t.join().unwrap(), Ok(Resource::Gpu));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn hostile_lock_files_and_names_are_harmless() {
        let d = scratch("hostile");
        std::fs::create_dir_all(&d).unwrap();
        // garbage left in a lock file nobody holds: free, and taken
        std::fs::write(lock_path(&d, Resource::Chrome), [0xffu8, 0, 9, 10, 13].repeat(10_000)).unwrap();
        assert_eq!(holder_in(&d, Resource::Chrome), None);
        let g = try_acquire_in(&d, Resource::Chrome, &format!("a\tb\nc{}", "x".repeat(10_000))).unwrap();
        let h = holder_in(&d, Resource::Chrome).unwrap();
        assert!(!h.what.contains('\t') && !h.what.contains('\n') && h.what.chars().count() <= MAX_WHAT, "{h:?}");
        drop(g);
        for bad in ["", "1\tx\ty", "x\t1\tz", "\t\t\t"] {
            assert!(parse_holder(bad).is_none() || parse_holder(bad).is_some_and(|h| h.what.len() <= MAX_WHAT));
        }
        assert_eq!(Resource::from_id(" Chrome "), Some(Resource::Chrome));
        assert_eq!(Resource::from_id("ollama"), None);
        assert_eq!(clean_what("   "), "unnamed");
        assert_eq!(duration_words(7_384), "2 h 3 min");
        let _ = std::fs::remove_dir_all(&d);
    }
}
