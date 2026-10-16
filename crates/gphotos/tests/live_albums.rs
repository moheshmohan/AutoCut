//! Real album calls against the signed-in Google Photos library, opt-in: they need the one-time
//! setup (crates/gphotos/README.md), the Photos Library API enabled in the Google Cloud project,
//! and `filmcraft-cli photos auth --albums`. It creates the album for real when it is missing
//! (an empty album; run it twice to see the second run reuse it). Run:
//!
//! ```sh
//! FILMCRAFT_GPHOTOS_LIVE_ALBUM="FilmCraft test album" cargo test -p filmcraft-gphotos --test live_albums -- --nocapture
//! ```
//!
//! Without the variable it returns at once, so `cargo test` stays offline.

use filmcraft_gphotos::GooglePhotos;

#[test]
fn creates_or_finds_an_album_and_lists_it() {
    let Some(title) = std::env::var("FILMCRAFT_GPHOTOS_LIVE_ALBUM").ok().filter(|t| !t.trim().is_empty()) else {
        eprintln!("skipped: set FILMCRAFT_GPHOTOS_LIVE_ALBUM=<title> to create (or find) that album in your Google Photos library");
        return;
    };
    let gp = GooglePhotos::configured().unwrap_or_else(|e| panic!("Google Photos is not set up: {e}"));
    let ensured = gp.albums(|a| a.ensure(&title, Some("Made by the FilmCraft live album test"))).unwrap_or_else(|e| panic!("ensure failed: {e}"));
    eprintln!("{}", ensured.to_json());
    let listed = gp.albums(|a| a.list()).unwrap_or_else(|e| panic!("list failed: {e}"));
    eprintln!("{} album(s) made by this app, complete: {}", listed.albums.len(), listed.complete);
    assert!(listed.albums.iter().any(|a| a.id == ensured.album.id), "the album is listed");
    let again = gp.albums(|a| a.ensure(&title, None)).unwrap_or_else(|e| panic!("second ensure failed: {e}"));
    assert_eq!(again.album.id, ensured.album.id, "reused, not duplicated");
    assert!(!again.created);
}
