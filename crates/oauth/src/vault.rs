//! Where one integration's tokens live: a [`SecretStore`] (the macOS Keychain) when there is one,
//! the owner-only `tokens.json` in the integration's data directory otherwise.
//!
//! The rules:
//!
//! - **Saving** goes to the Keychain and is read back; only a copy that reads back the same
//!   counts. A Keychain that cannot take it (locked and nobody answered, too large, broken) means
//!   the owner-only file instead, and the [`Stored::note`] says why.
//! - **`tokens.json` beside a Keychain is the newer copy**: written by a FilmCraft from before
//!   the Keychain, or by a save the Keychain refused. On load it is moved into the Keychain (the
//!   migration) and removed only after the Keychain copy reads back the same; if that fails it
//!   stays, and is used.
//! - **Readiness never touches the Keychain** (it is asked while drawing): [`TokenVault::summary`]
//!   reads the granted scopes from `tokens.json`, or from `signin.json`, a non-secret note of
//!   where the tokens are and what they grant, written beside them.
//! - Each data directory has its own Keychain item (the account is the directory's path,
//!   [`crate::keychain::account_for`]), so `FILMCRAFT_DATA_DIR` profiles (and tests) never share
//!   or overwrite a sign-in.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};

use crate::keychain::account_for;
use crate::store::{FileStore, STORE_ENV, SecretStore, system_store, write_private};
use crate::{OAuthError, TokenSet, now};

/// The owner-only token file (the fallback, and where FilmCraft kept tokens before the Keychain).
pub const TOKENS_FILE: &str = "tokens.json";
/// The non-secret note beside tokens kept in a secret store.
pub const SUMMARY_FILE: &str = "signin.json";

/// Where a sign-in is kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Place {
    /// In a secret store, by its name ("the macOS Keychain (…)").
    Secure(String),
    /// In the owner-only file.
    File(PathBuf),
}

impl std::fmt::Display for Place {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Place::Secure(name) => f.write_str(name),
            Place::File(p) => write!(f, "{} (readable only by you)", p.display()),
        }
    }
}

/// Where a load or save left the tokens, and anything the user should hear about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stored {
    pub place: Place,
    /// A migration done, or why the file was used instead of the Keychain.
    pub note: Option<String>,
}

/// What a sign-in grants, readable without the secret store.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Summary {
    /// The granted scopes, space-separated.
    pub scope: String,
    /// `keychain` or `file`.
    pub store: String,
    /// Unix seconds of the last save.
    pub saved_at: u64,
}

impl Summary {
    pub fn grants(&self, scope: &str) -> bool {
        self.scope.split_whitespace().any(|s| s == scope)
    }
}

enum Secure {
    /// Files only.
    None,
    /// The OS's store for this service, looked up on first use.
    System(String),
    Given(Arc<dyn SecretStore>),
}

/// The tokens of one integration (one data directory).
pub struct TokenVault {
    dir: PathBuf,
    secure: Secure,
    resolved: OnceLock<Option<Arc<dyn SecretStore>>>,
}

impl std::fmt::Debug for TokenVault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let secure = match &self.secure {
            Secure::None => "files".to_string(),
            Secure::System(s) => format!("system ({s})"),
            Secure::Given(s) => s.name(),
        };
        f.debug_struct("TokenVault").field("dir", &self.dir).field("secure", &secure).finish()
    }
}

impl TokenVault {
    /// The Keychain item named `service` on a Mac that has one (and unless
    /// `FILMCRAFT_SECRET_STORE=file`), `dir/tokens.json` otherwise.
    pub fn system(dir: &Path, service: &str) -> Self {
        Self { dir: dir.to_path_buf(), secure: Secure::System(service.to_string()), resolved: OnceLock::new() }
    }

    /// `dir/tokens.json` only.
    pub fn file_only(dir: &Path) -> Self {
        Self { dir: dir.to_path_buf(), secure: Secure::None, resolved: OnceLock::new() }
    }

    /// `store`, with `dir/tokens.json` as the fallback (tests give a fake store here).
    pub fn with_store(dir: &Path, store: Arc<dyn SecretStore>) -> Self {
        Self { dir: dir.to_path_buf(), secure: Secure::Given(store), resolved: OnceLock::new() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The owner-only token file.
    pub fn file(&self) -> PathBuf {
        self.dir.join(TOKENS_FILE)
    }

    /// The account of this directory's item in the secret store.
    pub fn account(&self) -> String {
        account_for(&self.dir)
    }

    fn store(&self) -> Option<&Arc<dyn SecretStore>> {
        self.resolved
            .get_or_init(|| match &self.secure {
                Secure::None => None,
                Secure::System(service) => system_store(service),
                Secure::Given(s) => Some(s.clone()),
            })
            .as_ref()
    }

    fn files(&self) -> FileStore {
        FileStore::new(&self.dir)
    }

    /// The saved tokens and where they are, `None` when there is no sign-in. A `tokens.json`
    /// beside a Keychain is moved into it here (see the module docs).
    pub fn load(&self) -> Result<Option<(TokenSet, Stored)>, OAuthError> {
        let file = self.file();
        let from_file = match self.files().get(TOKENS_FILE).map_err(|e| OAuthError::Store(e.0))? {
            Some(bytes) => Some(TokenSet::from_json(&bytes, &file.display().to_string())?),
            None => None,
        };
        let Some(store) = self.store() else {
            return Ok(from_file.map(|t| (t, Stored { place: Place::File(file), note: None })));
        };
        if let Some(tokens) = from_file {
            let stored = match self.put(store.as_ref(), &tokens) {
                Ok(()) => {
                    let mut note = match self.files().delete(TOKENS_FILE) {
                        Ok(()) => format!("moved the sign-in from {} into {}", file.display(), store.name()),
                        Err(e) => format!("copied the sign-in into {}, but {e}: delete that file by hand", store.name()),
                    };
                    if let Err(e) = self.write_summary(&tokens) {
                        note.push_str(&format!(" ({e})"));
                    }
                    Stored { place: Place::Secure(store.name()), note: Some(note) }
                }
                Err(e) => Stored { place: Place::File(file), note: Some(format!("kept the sign-in in the file: {e}")) },
            };
            return Ok(Some((tokens, stored)));
        }
        match store.get(&self.account()) {
            Ok(Some(bytes)) => Ok(Some((TokenSet::from_json(&bytes, "the saved sign-in")?, Stored { place: Place::Secure(store.name()), note: None }))),
            Ok(None) => Ok(None),
            Err(e) => Err(OAuthError::Store(format!(
                "{e}. Unlock the Keychain (log in at the Mac), or set {STORE_ENV}=file and sign in again to keep the sign-in in an owner-only file"
            ))),
        }
    }

    /// Keep `tokens`: in the secret store (read back to be sure), else in the owner-only file.
    pub fn save(&self, tokens: &TokenSet) -> Result<Stored, OAuthError> {
        let mut note = None;
        if let Some(store) = self.store() {
            match self.put(store.as_ref(), tokens) {
                Ok(()) => {
                    let mut said = Vec::new();
                    if let Err(e) = self.files().delete(TOKENS_FILE) {
                        said.push(format!("an older copy stays: {e}; delete it by hand"));
                    }
                    if let Err(e) = self.write_summary(tokens) {
                        said.push(e);
                    }
                    return Ok(Stored { place: Place::Secure(store.name()), note: (!said.is_empty()).then(|| said.join("; ")) });
                }
                Err(e) => note = Some(format!("{e}; kept the sign-in in the owner-only file instead")),
            }
        }
        self.files().set(TOKENS_FILE, &tokens.to_json()).map_err(|e| OAuthError::Store(e.0))?;
        // the file speaks for itself
        let _ = std::fs::remove_file(self.dir.join(SUMMARY_FILE));
        Ok(Stored { place: Place::File(self.file()), note })
    }

    /// Forget the sign-in here: the secret store's item, `tokens.json` and `signin.json`. (It does
    /// not revoke it at the provider: [`crate::OAuthClient::revoke`] does.)
    pub fn remove(&self) -> Result<(), OAuthError> {
        let mut errors = Vec::new();
        if let Some(store) = self.store()
            && let Err(e) = store.delete(&self.account())
        {
            errors.push(e.0);
        }
        if let Err(e) = self.files().delete(TOKENS_FILE) {
            errors.push(e.0);
        }
        match std::fs::remove_file(self.dir.join(SUMMARY_FILE)) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => errors.push(format!("cannot remove {}: {e}", self.dir.join(SUMMARY_FILE).display())),
            _ => {}
        }
        if errors.is_empty() { Ok(()) } else { Err(OAuthError::Store(errors.join("; "))) }
    }

    /// What the saved sign-in grants, without asking the secret store (cheap enough to ask while
    /// drawing): from `tokens.json` when it is there, else from `signin.json`. `None` when
    /// neither is.
    pub fn summary(&self) -> Result<Option<Summary>, OAuthError> {
        if let Some(bytes) = self.files().get(TOKENS_FILE).map_err(|e| OAuthError::Store(e.0))? {
            let t = TokenSet::from_json(&bytes, &self.file().display().to_string())?;
            return Ok(Some(Summary { scope: t.scope, store: "file".into(), saved_at: 0 }));
        }
        if matches!(self.secure, Secure::None) {
            return Ok(None);
        }
        let Some(bytes) = self.files().get(SUMMARY_FILE).map_err(|e| OAuthError::Store(e.0))? else { return Ok(None) };
        serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|e| OAuthError::Store(format!("{} is corrupt ({e}); sign in again", self.dir.join(SUMMARY_FILE).display())))
    }

    /// Write and read back; only an identical copy counts.
    fn put(&self, store: &dyn SecretStore, tokens: &TokenSet) -> Result<(), String> {
        let bytes = tokens.to_json();
        let account = self.account();
        store.set(&account, &bytes).map_err(|e| e.0)?;
        match store.get(&account).map_err(|e| e.0)? {
            Some(back) if back == bytes => Ok(()),
            Some(_) => Err(format!("{} gave back something else than was saved", store.name())),
            None => Err(format!("{} did not keep the sign-in", store.name())),
        }
    }

    fn write_summary(&self, tokens: &TokenSet) -> Result<(), String> {
        let s = Summary { scope: tokens.scope.clone(), store: "keychain".into(), saved_at: now() };
        let bytes = serde_json::to_vec_pretty(&s).unwrap_or_default();
        let path = self.dir.join(SUMMARY_FILE);
        write_private(&path, &bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{MemoryStore, StoreError};
    use std::sync::Mutex;

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("filmcraft-oauth-vault-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn tokens(scope: &str) -> TokenSet {
        TokenSet { access_token: "ya29.a".into(), refresh_token: Some("1//r".into()), expires_at: 123, scope: scope.into() }
    }

    /// What the pre-Keychain FilmCraft wrote.
    const LEGACY: &str = r#"{
  "access_token": "ya29.legacy",
  "expires_at": 1760000000,
  "refresh_token": "1//legacy",
  "scope": "https://www.googleapis.com/auth/photospicker.mediaitems.readonly"
}"#;

    /// A store that takes writes but gives back something else (or nothing), or refuses them.
    struct Faulty {
        mode: &'static str,
        inner: MemoryStore,
        writes: Mutex<usize>,
    }

    impl Faulty {
        fn new(mode: &'static str) -> Self {
            Self { mode, inner: MemoryStore::new(), writes: Mutex::new(0) }
        }
    }

    impl SecretStore for Faulty {
        fn name(&self) -> String {
            "the faulty store".into()
        }
        fn get(&self, account: &str) -> Result<Option<Vec<u8>>, StoreError> {
            match self.mode {
                "garbles" => Ok(self.inner.get(account)?.map(|mut b| {
                    b.push(b' ');
                    b
                })),
                "forgets" => Ok(None),
                "locked" => Err(StoreError("the macOS Keychain could not read the sign-in (locked)".into())),
                _ => self.inner.get(account),
            }
        }
        fn set(&self, account: &str, secret: &[u8]) -> Result<(), StoreError> {
            *self.writes.lock().unwrap() += 1;
            match self.mode {
                "refuses" | "locked" => Err(StoreError("the macOS Keychain could not save the sign-in (locked)".into())),
                _ => self.inner.set(account, secret),
            }
        }
        fn delete(&self, account: &str) -> Result<(), StoreError> {
            self.inner.delete(account)
        }
    }

    #[test]
    fn a_legacy_file_moves_into_the_store_on_first_use() {
        let dir = temp_dir("migrate");
        std::fs::write(dir.join(TOKENS_FILE), LEGACY).unwrap();
        let store = Arc::new(MemoryStore::new());
        let vault = TokenVault::with_store(&dir, store.clone());
        // readiness sees it before anything moves
        assert!(vault.summary().unwrap().unwrap().grants("https://www.googleapis.com/auth/photospicker.mediaitems.readonly"));
        let (t, stored) = vault.load().unwrap().unwrap();
        assert_eq!((t.access_token.as_str(), t.refresh_token.as_deref()), ("ya29.legacy", Some("1//legacy")));
        assert_eq!(stored.place, Place::Secure("memory".into()));
        assert!(stored.note.as_deref().unwrap().contains("moved the sign-in"), "{stored:?}");
        assert!(!dir.join(TOKENS_FILE).exists(), "the plaintext file is gone");
        assert_eq!(TokenSet::from_json(&store.get(&vault.account()).unwrap().unwrap(), "x").unwrap(), t);
        // the summary now comes from signin.json, which holds no token
        let note = std::fs::read_to_string(dir.join(SUMMARY_FILE)).unwrap();
        assert!(!note.contains("ya29") && !note.contains("1//") && note.contains("photospicker"), "{note}");
        assert_eq!(vault.summary().unwrap().unwrap().store, "keychain");
        // and the next load reads the store, quietly
        let (again, stored) = vault.load().unwrap().unwrap();
        assert_eq!((again, stored.note), (t, None));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_file_stays_unless_the_store_reads_back_the_same() {
        for mode in ["garbles", "forgets", "refuses"] {
            let dir = temp_dir(mode);
            std::fs::write(dir.join(TOKENS_FILE), LEGACY).unwrap();
            let vault = TokenVault::with_store(&dir, Arc::new(Faulty::new(mode)));
            let (t, stored) = vault.load().unwrap().unwrap();
            assert_eq!(t.access_token, "ya29.legacy", "{mode}");
            assert_eq!(stored.place, Place::File(dir.join(TOKENS_FILE)), "{mode}");
            assert!(stored.note.as_deref().unwrap().contains("kept the sign-in in the file"), "{mode}: {stored:?}");
            assert_eq!(std::fs::read_to_string(dir.join(TOKENS_FILE)).unwrap(), LEGACY, "{mode}: untouched");
            assert!(!dir.join(SUMMARY_FILE).exists(), "{mode}");
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    #[test]
    fn saves_go_to_the_store_and_fall_back_to_the_file() {
        let dir = temp_dir("save");
        let store = Arc::new(MemoryStore::new());
        let vault = TokenVault::with_store(&dir, store.clone());
        assert!(vault.load().unwrap().is_none() && vault.summary().unwrap().is_none());
        let stored = vault.save(&tokens("a b")).unwrap();
        assert_eq!(stored, Stored { place: Place::Secure("memory".into()), note: None });
        assert!(!dir.join(TOKENS_FILE).exists());
        assert_eq!(vault.load().unwrap().unwrap().0, tokens("a b"));
        let s = vault.summary().unwrap().unwrap();
        assert!(s.grants("a") && s.grants("b") && s.store == "keychain" && s.saved_at > 0);
        // a save replaces a stale file (from an older FilmCraft)
        std::fs::write(dir.join(TOKENS_FILE), LEGACY).unwrap();
        vault.save(&tokens("c")).unwrap();
        assert!(!dir.join(TOKENS_FILE).exists());
        assert_eq!(vault.load().unwrap().unwrap().0.scope, "c");
        std::fs::remove_dir_all(&dir).ok();

        // a store that refuses: the owner-only file, saying why
        let dir = temp_dir("save-fallback");
        let vault = TokenVault::with_store(&dir, Arc::new(Faulty::new("refuses")));
        let stored = vault.save(&tokens("a")).unwrap();
        assert_eq!(stored.place, Place::File(dir.join(TOKENS_FILE)));
        assert!(stored.note.as_deref().unwrap().contains("could not save") && stored.note.as_deref().unwrap().contains("owner-only file"), "{stored:?}");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(std::fs::metadata(dir.join(TOKENS_FILE)).unwrap().permissions().mode() & 0o777, 0o600);
        }
        assert_eq!(vault.summary().unwrap().unwrap().store, "file");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_locked_store_is_an_error_that_says_what_to_do() {
        let dir = temp_dir("locked");
        let vault = TokenVault::with_store(&dir, Arc::new(Faulty::new("locked")));
        let e = vault.load().unwrap_err().to_string();
        assert!(e.contains("locked") && e.contains("FILMCRAFT_SECRET_STORE=file"), "{e}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn files_only() {
        let dir = temp_dir("files");
        let vault = TokenVault::file_only(&dir);
        assert!(vault.load().unwrap().is_none());
        let stored = vault.save(&tokens("s")).unwrap();
        assert_eq!(stored, Stored { place: Place::File(dir.join(TOKENS_FILE)), note: None });
        assert!(stored.place.to_string().contains("readable only by you"));
        assert_eq!(vault.load().unwrap().unwrap().0, tokens("s"));
        assert_eq!(vault.summary().unwrap().unwrap().scope, "s");
        // a stray signin.json does not make a file-only vault signed in
        vault.remove().unwrap();
        std::fs::write(dir.join(SUMMARY_FILE), r#"{"scope":"s","store":"keychain"}"#).unwrap();
        assert!(vault.summary().unwrap().is_none() && vault.load().unwrap().is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn odd_files_are_errors_not_crashes() {
        let dir = temp_dir("odd");
        let vault = TokenVault::with_store(&dir, Arc::new(MemoryStore::new()));
        std::fs::write(dir.join(TOKENS_FILE), "{not json").unwrap();
        assert!(vault.load().unwrap_err().to_string().contains("corrupt"));
        assert!(vault.summary().is_err());
        std::fs::write(dir.join(TOKENS_FILE), r#"{"access_token":"a","expires_at":-5}"#).unwrap();
        assert!(vault.load().is_err());
        std::fs::remove_file(dir.join(TOKENS_FILE)).unwrap();
        std::fs::write(dir.join(SUMMARY_FILE), "[1,2").unwrap();
        assert!(vault.summary().unwrap_err().to_string().contains("signin.json is corrupt"));
        // a damaged store item
        let store = Arc::new(MemoryStore::new());
        let vault = TokenVault::with_store(&dir, store.clone());
        store.set(&vault.account(), b"\xff\xfe").unwrap();
        assert!(vault.load().unwrap_err().to_string().contains("corrupt"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn removing_forgets_everywhere_and_directories_do_not_share() {
        let store: Arc<MemoryStore> = Arc::new(MemoryStore::new());
        let a = temp_dir("remove-a");
        let b = temp_dir("remove-b");
        let va = TokenVault::with_store(&a, store.clone());
        let vb = TokenVault::with_store(&b, store.clone());
        assert_ne!(va.account(), vb.account());
        va.save(&tokens("a")).unwrap();
        vb.save(&tokens("b")).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(va.load().unwrap().unwrap().0.scope, "a");
        std::fs::write(a.join(TOKENS_FILE), LEGACY).unwrap();
        va.remove().unwrap();
        assert!(va.load().unwrap().is_none() && va.summary().unwrap().is_none());
        assert!(!a.join(TOKENS_FILE).exists() && !a.join(SUMMARY_FILE).exists());
        assert_eq!(vb.load().unwrap().unwrap().0.scope, "b", "the other directory keeps its sign-in");
        va.remove().unwrap();
        std::fs::remove_dir_all(&a).ok();
        std::fs::remove_dir_all(&b).ok();
    }
}
