//! The desktop app's control-channel token, shared with the owner's local clients.
//!
//! The control server refuses every request that does not carry the session's token
//! (`docs/control-protocol.md`). A token in [`ENV`] (`FILMCRAFT_CONTROL_TOKEN`) is used as given,
//! and clients started from the same environment send it. Without one, the app makes a fresh
//! random token each time its control server starts and writes it to [`path`]`(port)`: a file only
//! its owner can read (0600 in a 0700 folder on Unix; the per-user profile folder on Windows),
//! replaced at every launch and removed when the app quits. A client with no token in its
//! environment ([`crate::bridge::BridgeClient`]) reads it from there, so `filmcraft-cli --bridge`,
//! the MCP bridge and scripts run by the same user need no token step, and other users get none.
//!
//! The file is keyed by port, not by data directory: the port is what a client knows, and an app
//! started with its own `--data-dir` or `FILMCRAFT_DATA_DIR` is still found.

use std::io::Read;
use std::path::{Path, PathBuf};

/// The environment variable holding a token chosen by whoever starts the app.
pub const ENV: &str = "FILMCRAFT_CONTROL_TOKEN";
/// Overrides the folder of the token files (tests, separate profiles).
pub const DIR_ENV: &str = "FILMCRAFT_CONTROL_DIR";
/// A token is this many hex characters (32 random bytes).
pub const LEN: usize = 64;
/// Bytes read from a token file at most; a token is [`LEN`] plus a newline.
const MAX_FILE: u64 = 256;

/// Whether `token` has the form the control server accepts: [`LEN`] hex characters.
pub fn is_valid(token: &str) -> bool {
    token.len() == LEN && token.bytes().all(|b| b.is_ascii_hexdigit())
}

/// The token in [`ENV`], when it is set and not empty (not checked: the server judges it).
pub fn from_env() -> Option<String> {
    std::env::var(ENV).ok().filter(|t| !t.is_empty())
}

/// The folder of the token files: [`DIR_ENV`], else `control/` in the per-user FilmCraft folder
/// (macOS `~/Library/Application Support/FilmCraft`, Windows `%APPDATA%\FilmCraft`, elsewhere
/// `$XDG_DATA_HOME/filmcraft` or `~/.local/share/filmcraft`).
pub fn dir() -> Option<PathBuf> {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(d) = var(DIR_ENV) {
        return Some(d);
    }
    let base = if cfg!(target_os = "macos") {
        var("HOME").map(|h| h.join("Library/Application Support/FilmCraft"))
    } else if cfg!(windows) {
        var("APPDATA").map(|a| a.join("FilmCraft"))
    } else {
        var("XDG_DATA_HOME").map(|x| x.join("filmcraft")).or_else(|| var("HOME").map(|h| h.join(".local/share/filmcraft")))
    };
    base.map(|b| b.join("control"))
}

/// The token file of the app listening on `port`.
pub fn path(port: u16) -> Option<PathBuf> {
    dir().map(|d| d.join(format!("{port}.token")))
}

/// The port of a `host:port` address (`127.0.0.1:9876`, `[::1]:9876`).
pub fn port_of(addr: &str) -> Option<u16> {
    addr.rsplit_once(':')?.1.parse().ok()
}

/// A fresh token: [`LEN`] hex characters from the operating system's secure random source.
pub fn generate() -> Result<String, String> {
    let mut bytes = [0u8; LEN / 2];
    getrandom::fill(&mut bytes).map_err(|e| format!("the system's random number source failed: {e}"))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// The token in `text` (a token file's contents), when it is a valid one.
pub fn parse(text: &str) -> Option<String> {
    let t = text.trim();
    is_valid(t).then(|| t.to_string())
}

/// The token in the file at `path`; `None` when there is none or it is not a valid token.
pub fn read(path: &Path) -> Option<String> {
    let mut text = String::new();
    std::fs::File::open(path).ok()?.take(MAX_FILE).read_to_string(&mut text).ok()?;
    parse(&text)
}

/// Write `token` to `path`, readable by its owner only: written to a new file beside it and
/// renamed over the old one, so a client never reads half a token.
pub fn publish(path: &Path, token: &str) -> std::io::Result<()> {
    let dir = path.parent().ok_or_else(|| std::io::Error::other("the token file has no folder"))?;
    std::fs::create_dir_all(dir)?;
    owner_only(dir, 0o700)?;
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = dir.join(format!(".{name}.{}.tmp", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    let written = write_new(&tmp, token).and_then(|()| std::fs::rename(&tmp, path));
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written
}

/// Remove the token file at `path` if it still holds `token` (a later app on the port may have
/// replaced it).
pub fn withdraw(path: &Path, token: &str) {
    if read(path).as_deref() == Some(token) {
        let _ = std::fs::remove_file(path);
    }
}

fn write_new(path: &Path, token: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let mut f = options.open(path)?;
    f.write_all(format!("{token}\n").as_bytes())?;
    f.sync_all()
}

#[cfg(unix)]
fn owner_only(path: &Path, mode: u32) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
}

#[cfg(not(unix))]
fn owner_only(_path: &Path, _mode: u32) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_tokens_are_valid_and_differ() {
        let a = generate().unwrap();
        let b = generate().unwrap();
        assert!(is_valid(&a) && is_valid(&b), "{a} {b}");
        assert_ne!(a, b);
    }

    #[test]
    fn only_well_formed_tokens_are_read() {
        let good = "0123456789abcdef".repeat(4);
        assert_eq!(parse(&format!("{good}\n")), Some(good.clone()));
        for bad in ["", "abc", &"z".repeat(64), &"a".repeat(63), &"a".repeat(65), &format!("{good} {good}")] {
            assert_eq!(parse(bad), None, "{bad}");
        }
    }

    #[test]
    fn ports_come_from_the_address() {
        assert_eq!(port_of("127.0.0.1:9876"), Some(9876));
        assert_eq!(port_of("[::1]:9877"), Some(9877));
        assert_eq!(port_of("localhost"), None);
        assert_eq!(port_of("127.0.0.1:99999"), None);
    }

    #[test]
    fn a_published_token_is_read_back_and_withdrawn_only_by_its_owner() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("control").join("9877.token");
        let token = generate().unwrap();
        publish(&path, &token).unwrap();
        assert_eq!(read(&path), Some(token.clone()));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
            assert_eq!(std::fs::metadata(path.parent().unwrap()).unwrap().permissions().mode() & 0o777, 0o700);
        }
        // a later launch replaces it; the earlier app's exit leaves the new token alone
        let next = generate().unwrap();
        publish(&path, &next).unwrap();
        assert_eq!(read(&path), Some(next.clone()));
        withdraw(&path, &token);
        assert_eq!(read(&path), Some(next.clone()));
        withdraw(&path, &next);
        assert!(!path.exists());
        // nothing else is left in the folder
        assert_eq!(std::fs::read_dir(path.parent().unwrap()).unwrap().count(), 0);
    }

    #[test]
    fn an_oversized_or_junk_file_is_no_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("1.token");
        std::fs::write(&path, "a".repeat(10_000)).unwrap();
        assert_eq!(read(&path), None);
        assert_eq!(read(&dir.path().join("missing.token")), None);
    }
}
