//! The real macOS Keychain — opt-in, since it writes to the login keychain (a throwaway item,
//! service "FilmCraft live test", removed at the end):
//!
//! ```sh
//! FILMCRAFT_LIVE_KEYCHAIN=1 cargo test -p filmcraft-oauth --test live_keychain -- --nocapture
//! ```

#[cfg(target_os = "macos")]
#[test]
fn keychain_round_trip_and_migration() {
    use std::sync::Arc;

    use filmcraft_oauth::keychain::KeychainStore;
    use filmcraft_oauth::vault::{Place, TOKENS_FILE};
    use filmcraft_oauth::{SecretStore, TokenSet, TokenVault};

    if std::env::var("FILMCRAFT_LIVE_KEYCHAIN").is_err() {
        eprintln!("skipped: set FILMCRAFT_LIVE_KEYCHAIN=1 to use the real Keychain");
        return;
    }
    assert!(filmcraft_oauth::keychain::available(), "this Mac has no usable default keychain");
    let dir = std::env::temp_dir().join(format!("filmcraft-live-keychain-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let store = Arc::new(KeychainStore::new("FilmCraft live test").unwrap());
    let vault = TokenVault::with_store(&dir, store.clone());
    let account = vault.account();
    let _ = store.delete(&account);

    // nothing yet; a set, a read, an update
    assert_eq!(store.get(&account).unwrap(), None);
    store.set(&account, b"{\"first\":1}").unwrap();
    assert_eq!(store.get(&account).unwrap().as_deref(), Some(&b"{\"first\":1}"[..]));
    // a realistic token set, with every byte value JSON allows in strings
    let tokens = TokenSet {
        access_token: format!("ya29.{}", "aB3-_".repeat(80)),
        refresh_token: Some("1//0g-live-test-refresh".into()),
        expires_at: 1_760_000_000,
        scope: "https://www.googleapis.com/auth/photospicker.mediaitems.readonly https://www.googleapis.com/auth/photoslibrary.appendonly https://www.googleapis.com/auth/photoslibrary.readonly.appcreateddata".into(),
    };
    // the migration: a plaintext file moves into the Keychain and is removed
    std::fs::write(dir.join(TOKENS_FILE), tokens.to_json()).unwrap();
    let (loaded, stored) = vault.load().unwrap().unwrap();
    assert_eq!(loaded, tokens);
    assert!(matches!(stored.place, Place::Secure(_)), "{stored:?}");
    assert!(!dir.join(TOKENS_FILE).exists(), "the plaintext copy is gone");
    eprintln!("{}", stored.note.unwrap_or_default());
    let (again, _) = vault.load().unwrap().unwrap();
    assert_eq!(again, tokens, "read back from the Keychain");
    // a refreshed set replaces it
    let fresh = TokenSet { access_token: "ya29.second".into(), ..tokens.clone() };
    vault.save(&fresh).unwrap();
    assert_eq!(vault.load().unwrap().unwrap().0, fresh);
    // sign-out removes it
    vault.remove().unwrap();
    assert_eq!(store.get(&account).unwrap(), None);
    assert!(vault.load().unwrap().is_none());
    std::fs::remove_dir_all(&dir).ok();
}
