//! A real upload to the signed-in Google Photos library, opt-in: it needs the one-time setup
//! (crates/gphotos/README.md), the Photos Library API enabled in the Google Cloud project, and
//! `filmcraft-cli photos auth --upload`. It adds the file to the library for real. Run:
//!
//! ```sh
//! FILMCRAFT_GPHOTOS_LIVE_UPLOAD=/path/to/clip.mp4 cargo test -p filmcraft-gphotos --test live_upload -- --nocapture
//! ```
//!
//! Without the variable it returns at once, so `cargo test` stays offline.

use std::path::PathBuf;

use filmcraft_gphotos::GooglePhotos;
use filmcraft_gphotos::upload::UploadOptions;

#[test]
fn uploads_a_file_to_google_photos() {
    let Some(path) = std::env::var_os("FILMCRAFT_GPHOTOS_LIVE_UPLOAD").map(PathBuf::from) else {
        eprintln!("skipped: set FILMCRAFT_GPHOTOS_LIVE_UPLOAD=<file> to upload it to your Google Photos library");
        return;
    };
    let gp = GooglePhotos::configured().unwrap_or_else(|e| panic!("Google Photos is not set up: {e}"));
    let opts = UploadOptions { description: Some("FilmCraft live upload test".into()), ..Default::default() };
    let mut last = 0;
    let up = gp
        .upload(&path, &opts, &mut |sent, total, status| {
            let pct = sent.saturating_mul(100).checked_div(total).unwrap_or(0);
            if pct >= last + 10 || sent == total {
                eprintln!("{pct:>3}% {status}");
                last = pct;
            }
            true
        })
        .unwrap_or_else(|e| panic!("upload failed: {e}"));
    eprintln!("{}", up.to_json());
    assert!(!up.media_item_id.is_empty());
    assert_eq!(up.bytes, std::fs::metadata(&path).unwrap().len());
}
