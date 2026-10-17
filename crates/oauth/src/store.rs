//! Secret storage behind one trait: the macOS Keychain ([`crate::keychain::KeychainStore`]), an
//! owner-only file ([`FileStore`], the fallback), and memory ([`MemoryStore`], for tests).

use std::collections::HashMap;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

/// The environment variable that chooses the store: `file` keeps secrets in owner-only files
/// even where a Keychain exists (headless machines, tests, CI). Anything else, or unset, uses
/// the Keychain where there is one.
pub const STORE_ENV: &str = "FILMCRAFT_SECRET_STORE";
/// More than any secret FilmCraft keeps (a token set is a few KB).
const MAX_SECRET: u64 = 1024 * 1024;

/// A failure of a secret store, said in words.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct StoreError(pub String);

/// Somewhere to keep secrets by account name.
pub trait SecretStore: Send + Sync {
    /// Where the secrets go, for messages ("the macOS Keychain").
    fn name(&self) -> String;
    /// The secret of `account`, `None` when there is none.
    fn get(&self, account: &str) -> Result<Option<Vec<u8>>, StoreError>;
    /// Create or replace the secret of `account`.
    fn set(&self, account: &str, secret: &[u8]) -> Result<(), StoreError>;
    /// Remove the secret of `account`; removing what is not there is not an error.
    fn delete(&self, account: &str) -> Result<(), StoreError>;
}

/// The operating system's store for `service`, when there is one and [`STORE_ENV`] does not say
/// `file`: the macOS Keychain (checked to answer, once per process). `None` elsewhere: the
/// caller keeps the secret in an owner-only file.
pub fn system_store(service: &str) -> Option<Arc<dyn SecretStore>> {
    if files_requested(std::env::var(STORE_ENV).ok().as_deref()) {
        return None;
    }
    #[cfg(target_os = "macos")]
    if crate::keychain::available() {
        return crate::keychain::KeychainStore::new(service).ok().map(|k| Arc::new(k) as Arc<dyn SecretStore>);
    }
    let _ = service;
    None
}

/// Whether [`STORE_ENV`]'s value asks for files.
fn files_requested(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim().eq_ignore_ascii_case("file"))
}

/// Secrets in memory, for tests.
#[derive(Default)]
pub struct MemoryStore {
    items: Mutex<HashMap<String, Vec<u8>>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// How many secrets it holds.
    pub fn len(&self) -> usize {
        self.items.lock().unwrap_or_else(PoisonError::into_inner).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl SecretStore for MemoryStore {
    fn name(&self) -> String {
        "memory".into()
    }

    fn get(&self, account: &str) -> Result<Option<Vec<u8>>, StoreError> {
        Ok(self.items.lock().unwrap_or_else(PoisonError::into_inner).get(account).cloned())
    }

    fn set(&self, account: &str, secret: &[u8]) -> Result<(), StoreError> {
        self.items.lock().unwrap_or_else(PoisonError::into_inner).insert(account.to_string(), secret.to_vec());
        Ok(())
    }

    fn delete(&self, account: &str) -> Result<(), StoreError> {
        self.items.lock().unwrap_or_else(PoisonError::into_inner).remove(account);
        Ok(())
    }
}

/// Secrets as files in one directory, one per account (the account is the file name), readable
/// and writable by the owner only (`0600` on Unix), each replaced atomically.
#[derive(Debug, Clone)]
pub struct FileStore {
    dir: PathBuf,
}

impl FileStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The file of `account`: a plain file name (letters, digits, `.`, `_`, `-`; not starting
    /// with `.`), so no account can name a file outside the directory.
    pub fn path(&self, account: &str) -> Result<PathBuf, StoreError> {
        let ok = !account.is_empty()
            && account.len() <= 128
            && !account.starts_with('.')
            && account.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
        if !ok {
            return Err(StoreError(format!("`{}` is not a usable secret file name", crate::truncate(account, 40))));
        }
        Ok(self.dir.join(account))
    }
}

impl SecretStore for FileStore {
    fn name(&self) -> String {
        format!("an owner-only file in {}", self.dir.display())
    }

    fn get(&self, account: &str) -> Result<Option<Vec<u8>>, StoreError> {
        let path = self.path(account)?;
        let file = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(StoreError(format!("cannot read {}: {e}", path.display()))),
        };
        let mut bytes = Vec::new();
        file.take(MAX_SECRET + 1).read_to_end(&mut bytes).map_err(|e| StoreError(format!("cannot read {}: {e}", path.display())))?;
        if bytes.len() as u64 > MAX_SECRET {
            return Err(StoreError(format!("{} is too large to be a FilmCraft secret", path.display())));
        }
        Ok(Some(bytes))
    }

    fn set(&self, account: &str, secret: &[u8]) -> Result<(), StoreError> {
        let path = self.path(account)?;
        write_private(&path, secret).map_err(|e| StoreError(format!("cannot write {}: {e}", path.display())))
    }

    fn delete(&self, account: &str) -> Result<(), StoreError> {
        let path = self.path(account)?;
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(StoreError(format!("cannot remove {}: {e}", path.display()))),
        }
    }
}

/// Write `bytes` to `path` owner-only (`0600` on Unix, from creation on): a temporary file in the
/// same directory, then renamed over `path`, so a crash never leaves half a file.
pub fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)?;
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "secret".into());
    let tmp = dir.join(format!(".{name}.{}.tmp", std::process::id()));
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        opts.mode(0o600);
    }
    let result = (|| {
        let mut f = opts.open(&tmp)?;
        #[cfg(unix)]
        {
            // a temporary file left by an older run may have other permissions
            use std::os::unix::fs::PermissionsExt as _;
            f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        }
        f.write_all(bytes)?;
        f.sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("filmcraft-oauth-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn memory_store() {
        let m = MemoryStore::new();
        assert!(m.is_empty() && m.get("a").unwrap().is_none());
        m.set("a", b"1").unwrap();
        m.set("a", b"2").unwrap();
        assert_eq!(m.get("a").unwrap().as_deref(), Some(&b"2"[..]));
        m.delete("a").unwrap();
        m.delete("a").unwrap();
        assert!(m.is_empty());
    }

    #[test]
    fn file_store_is_owner_only_and_confined() {
        let dir = temp_dir("filestore");
        let f = FileStore::new(dir.join("nested"));
        assert_eq!(f.get("tokens.json").unwrap(), None);
        f.set("tokens.json", b"{}").unwrap();
        f.set("tokens.json", b"{\"a\":1}").unwrap();
        assert_eq!(f.get("tokens.json").unwrap().as_deref(), Some(&b"{\"a\":1}"[..]));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(dir.join("nested/tokens.json")).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "owner-only");
        }
        let left: Vec<_> = std::fs::read_dir(dir.join("nested")).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(left.len(), 1, "no temporary files left: {left:?}");
        for bad in ["", "../x", "a/b", ".hidden", "a\\b", "a b", &"x".repeat(129)] {
            assert!(f.set(bad, b"x").is_err(), "{bad:?}");
            assert!(f.get(bad).is_err(), "{bad:?}");
        }
        f.delete("tokens.json").unwrap();
        f.delete("tokens.json").unwrap();
        assert_eq!(f.get("tokens.json").unwrap(), None);
        // a huge file is refused, not read whole
        std::fs::write(dir.join("nested/big"), vec![b'x'; MAX_SECRET as usize + 1]).unwrap();
        assert!(f.get("big").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_environment_can_ask_for_files() {
        for v in ["file", " FILE ", "File"] {
            assert!(files_requested(Some(v)), "{v}");
        }
        for v in [None, Some(""), Some("keychain"), Some("files")] {
            assert!(!files_requested(v), "{v:?}");
        }
    }
}
