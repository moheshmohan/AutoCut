//! `cargo xtask sweep [--days N] [--dev] [--dry-run]`: give back the disk that builds and tests
//! took. Parallel agents each build in `target/agent-<name>` (tens of GB apiece) and nothing
//! removes those when the agent is done; debug builds keep growing incremental caches.
//!
//! From the workspace's target directory the sweep removes:
//! - `agent-*` build directories untouched for `--days` (default 3);
//! - every profile's `incremental/` cache (the next build of that profile is a little slower);
//! - test outputs (`tmp/`, `golden-failures/`, `export-tests/`) untouched for `--days`;
//! - with `--dev`, the whole dev profile (`debug/`), as `cargo clean --profile dev` would.
//!
//! A profile a build is using right now (cargo holds its `.cargo-lock`) is left alone, and the
//! sweep holds that lock while it deletes, so no build starts halfway. `--dry-run` only reports.

use std::fs::{self, File, TryLockError};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

const DEFAULT_DAYS: u64 = 3;
/// What tests leave in the target directory.
const TEST_OUTPUTS: &[&str] = &["tmp", "golden-failures", "export-tests"];
/// Deepest directory nesting measured.
const MAX_DEPTH: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Options {
    pub days: u64,
    pub dev: bool,
    pub dry_run: bool,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options { days: DEFAULT_DAYS, dev: false, dry_run: false };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--days" => o.days = it.next().and_then(|d| d.parse().ok()).filter(|d| *d <= 3650).ok_or("--days wants a whole number of days")?,
            "--dev" => o.dev = true,
            "--dry-run" => o.dry_run = true,
            other => return Err(format!("sweep: unknown `{other}` (sweep [--days N] [--dev] [--dry-run])")),
        }
    }
    Ok(o)
}

/// Bytes under `p`, not following links.
fn size(p: &Path, depth: usize) -> u64 {
    let Ok(m) = fs::symlink_metadata(p) else { return 0 };
    if !m.is_dir() || depth >= MAX_DEPTH {
        return m.len();
    }
    fs::read_dir(p).into_iter().flatten().flatten().map(|e| size(&e.path(), depth + 1)).fold(0, u64::saturating_add)
}

/// When `p`, or anything up to two levels in, last changed.
fn touched(p: &Path) -> Option<SystemTime> {
    let mtime = |p: &Path| fs::symlink_metadata(p).and_then(|m| m.modified()).ok();
    let mut newest = mtime(p)?;
    for e in fs::read_dir(p).into_iter().flatten().flatten() {
        let child = e.path();
        newest = newest.max(mtime(&child).unwrap_or(newest));
        if e.file_type().is_ok_and(|t| t.is_dir()) {
            for g in fs::read_dir(&child).into_iter().flatten().flatten() {
                newest = newest.max(mtime(&g.path()).unwrap_or(newest));
            }
        }
    }
    Some(newest)
}

/// The profile directories under a target directory (`debug/`, `release/`,
/// `<triple>/<profile>/`): those cargo keeps a `.cargo-lock` in.
fn profiles(target: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in fs::read_dir(target).into_iter().flatten().flatten() {
        let p = e.path();
        if !e.file_type().is_ok_and(|t| t.is_dir()) || is_agent_dir(&p) {
            continue;
        }
        if p.join(".cargo-lock").exists() {
            out.push(p);
        } else {
            out.extend(fs::read_dir(&p).into_iter().flatten().flatten().map(|g| g.path()).filter(|g| g.join(".cargo-lock").exists()));
        }
    }
    out.sort();
    out
}

fn is_agent_dir(p: &Path) -> bool {
    p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("agent-"))
}

/// Cargo's lock on a profile, held while we delete in it; `None` when a build has it now.
fn lock_profile(profile: &Path) -> Option<Option<File>> {
    let Ok(f) = File::options().read(true).write(true).open(profile.join(".cargo-lock")) else {
        // no lock file: no build there to collide with
        return Some(None);
    };
    match f.try_lock() {
        Ok(()) => Some(Some(f)),
        Err(TryLockError::WouldBlock) => None,
        Err(TryLockError::Error(_)) => Some(None),
    }
}

/// Every profile of `target` locked, or `None` when a build is using one.
fn lock_all(target: &Path) -> Option<Vec<File>> {
    let mut held = Vec::new();
    for p in profiles(target) {
        held.extend(lock_profile(&p)?);
    }
    Some(held)
}

/// What a sweep did (or, dry, would do).
#[derive(Debug, Default)]
pub(crate) struct Report {
    pub removed: Vec<(PathBuf, u64)>,
    pub busy: Vec<PathBuf>,
    pub errors: Vec<String>,
}

impl Report {
    fn freed(&self) -> u64 {
        self.removed.iter().map(|(_, n)| *n).fold(0, u64::saturating_add)
    }
}

fn remove(r: &mut Report, p: &Path, dry_run: bool) {
    let n = size(p, 0);
    if !dry_run && let Err(e) = fs::remove_dir_all(p) {
        r.errors.push(format!("{}: {e}", p.display()));
        return;
    }
    r.removed.push((p.to_path_buf(), n));
}

fn older_than(p: &Path, days: u64, now: SystemTime) -> bool {
    let age = touched(p).and_then(|t| now.duration_since(t).ok()).unwrap_or(Duration::ZERO);
    age >= Duration::from_secs(days.saturating_mul(86_400))
}

/// The sweep of `target` as of `now`.
pub(crate) fn sweep(target: &Path, o: Options, now: SystemTime) -> Report {
    let mut r = Report::default();
    // agents' build directories, whole
    let mut agents: Vec<PathBuf> = fs::read_dir(target).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.is_dir() && is_agent_dir(p)).collect();
    agents.sort();
    for a in &agents {
        if !older_than(a, o.days, now) {
            continue;
        }
        match lock_all(a) {
            Some(_held) => remove(&mut r, a, o.dry_run),
            None => r.busy.push(a.clone()),
        }
    }
    // test outputs
    for t in TEST_OUTPUTS.iter().map(|t| target.join(t)).filter(|t| t.is_dir()) {
        if older_than(&t, o.days, now) {
            remove(&mut r, &t, o.dry_run);
        }
    }
    // each profile's incremental cache, or with --dev the dev profile whole
    for p in profiles(target) {
        let dev = p.file_name().is_some_and(|n| n == "debug");
        let what = if o.dev && dev { p.clone() } else { p.join("incremental") };
        if !what.is_dir() {
            continue;
        }
        match lock_profile(&p) {
            Some(_held) => remove(&mut r, &what, o.dry_run),
            None => r.busy.push(p),
        }
    }
    r
}

fn gib(n: u64) -> String {
    format!("{:.1} GB", n as f64 / 1e9)
}

pub(crate) fn run_sweep(args: &[String]) -> Result<(), String> {
    let o = parse(args)?;
    let md = crate::metadata()?;
    let target = md["target_directory"].as_str().map(PathBuf::from).ok_or("cargo metadata names no target directory")?;
    let r = sweep(&target, o, SystemTime::now());
    let verb = if o.dry_run { "would remove" } else { "removed" };
    for (p, n) in &r.removed {
        eprintln!("sweep: {verb} {} ({})", p.display(), gib(*n));
    }
    for p in &r.busy {
        eprintln!("sweep: kept {} (a build is using it)", p.display());
    }
    eprintln!("sweep: {} {} from {}", if o.dry_run { "would free" } else { "freed" }, gib(r.freed()), target.display());
    if r.errors.is_empty() { Ok(()) } else { Err(format!("sweep: could not remove:\n  {}", r.errors.join("\n  "))) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("xtask-sweep-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    fn file(p: &Path, bytes: usize) {
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, vec![0u8; bytes]).unwrap();
    }

    /// A target directory: the main profiles, two agents' builds, test output.
    fn target(name: &str) -> PathBuf {
        let t = scratch(name);
        file(&t.join("debug/.cargo-lock"), 0);
        file(&t.join("debug/deps/libx.rlib"), 100);
        file(&t.join("debug/incremental/x-1/s-1/dep-graph.bin"), 1000);
        file(&t.join("release/.cargo-lock"), 0);
        file(&t.join("release/deps/libx.rlib"), 100);
        file(&t.join("aarch64-apple-darwin/release/.cargo-lock"), 0);
        file(&t.join("aarch64-apple-darwin/release/incremental/y/z.bin"), 10);
        file(&t.join("agent-music/debug/.cargo-lock"), 0);
        file(&t.join("agent-music/debug/deps/liby.rlib"), 500);
        file(&t.join("agent-oauth/release/.cargo-lock"), 0);
        file(&t.join("golden-failures/a.png"), 7);
        file(&t.join("fixtures/a.mp4"), 9);
        t
    }

    fn later(days: u64) -> SystemTime {
        SystemTime::now() + Duration::from_secs(days * 86_400)
    }

    fn removed(r: &Report, t: &Path) -> Vec<String> {
        r.removed.iter().map(|(p, _)| p.strip_prefix(t).unwrap().to_string_lossy().replace('\\', "/")).collect()
    }

    #[test]
    fn fresh_agents_and_outputs_stay_and_incremental_caches_go() {
        let t = target("fresh");
        let o = Options { days: 3, dev: false, dry_run: false };
        let r = sweep(&t, o, SystemTime::now());
        assert_eq!(removed(&r, &t), ["aarch64-apple-darwin/release/incremental", "debug/incremental"]);
        assert_eq!(r.removed.iter().map(|(_, n)| n).sum::<u64>(), 1010);
        assert!(!t.join("debug/incremental").exists() && t.join("debug/deps/libx.rlib").exists());
        assert!(t.join("agent-music").exists() && t.join("golden-failures").exists() && t.join("fixtures").exists());
        // days later the agents' builds and the test output go too; fixtures always stay
        let r = sweep(&t, o, later(4));
        assert_eq!(removed(&r, &t), ["agent-music", "agent-oauth", "golden-failures"]);
        assert!(t.join("fixtures/a.mp4").exists() && t.join("release/deps/libx.rlib").exists());
        let _ = fs::remove_dir_all(&t);
    }

    #[test]
    fn a_build_in_progress_is_left_alone_and_a_dry_run_removes_nothing() {
        let t = target("busy");
        // a "build" holds the music agent's and the main debug profile's locks
        let a = File::options().read(true).write(true).open(t.join("agent-music/debug/.cargo-lock")).unwrap();
        a.try_lock().unwrap();
        let b = File::options().read(true).write(true).open(t.join("debug/.cargo-lock")).unwrap();
        b.try_lock().unwrap();
        let r = sweep(&t, Options { days: 3, dev: true, dry_run: false }, later(4));
        assert_eq!(removed(&r, &t), ["agent-oauth", "golden-failures", "aarch64-apple-darwin/release/incremental"]);
        assert_eq!(r.busy, [t.join("agent-music"), t.join("debug")]);
        assert!(t.join("debug/incremental").exists());
        drop((a, b));
        // --dev takes the whole dev profile once it is free; dry, it only reports
        let r = sweep(&t, Options { days: 3, dev: true, dry_run: true }, later(4));
        assert_eq!(removed(&r, &t), ["agent-music", "debug"]);
        assert!(t.join("agent-music").exists() && t.join("debug").exists());
        let _ = fs::remove_dir_all(&t);
    }

    #[test]
    fn options() {
        let args = |s: &[&str]| s.iter().map(|a| a.to_string()).collect::<Vec<_>>();
        assert_eq!(parse(&[]).unwrap(), Options { days: 3, dev: false, dry_run: false });
        assert_eq!(parse(&args(&["--days", "0", "--dev", "--dry-run"])).unwrap(), Options { days: 0, dev: true, dry_run: true });
        for bad in [&["--days"][..], &["--days", "-1"], &["--days", "9999"], &["--all"]] {
            assert!(parse(&args(bad)).is_err(), "{bad:?}");
        }
    }
}
