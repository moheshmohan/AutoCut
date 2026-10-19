//! `filmcraft-cli photos locations`: where an album's items were taken, read from the Google
//! Photos page by the screen agent ([`filmcraft_automation::locations`]), written to a JSON
//! file, and — with `--project`/`--bridge` — stored with the project's items
//! (`photos.applyLocations`).

use std::path::PathBuf;
use std::time::Duration;

use serde_json::{Value, json};

use crate::args::Args;
use crate::{Backend, album_opt, diag, fail, print, usage};

pub const USAGE: &str = "photos locations --album '<exact title>' | --album-url https://photos.google.com/album/… [--out locations.json] [--resume] [--max-items N] [--per-item S] [--timeout MIN] [--project p.fcproj --save | --bridge HOST:PORT]";

fn number(a: &Args, k: &str, lo: u64, hi: u64, default: u64) -> u64 {
    match a.opt(k) {
        None => default,
        Some(v) => match v.trim().parse::<u64>() {
            Ok(n) if (lo..=hi).contains(&n) => n,
            _ => usage(format!("photos locations: {k} wants a whole number from {lo} to {hi}, not `{v}`")),
        },
    }
}

/// The run `a` asks for.
fn run_of(a: &Args) -> filmcraft_automation::locations::Run {
    let album = album_opt(a, "photos locations", "--album");
    let album_url = a.opt("--album-url").map(|u| u.trim().to_string());
    if album.is_none() && album_url.is_none() {
        usage(format!("photos locations: say which album: --album '<its exact title>' or --album-url <its address>\n  {USAGE}"));
    }
    if let Some(u) = &album_url
        && filmcraft_screenagent::locations::album_url(u).is_none()
    {
        usage(format!("photos locations: --album-url wants a Google Photos album address (https://photos.google.com/album/…), not `{u}`"));
    }
    filmcraft_automation::locations::Run {
        album,
        album_url,
        max_items: number(a, "--max-items", 1, 5000, 1000) as usize,
        per_item: Duration::from_secs(number(a, "--per-item", 5, 600, 90)),
        budget: Duration::from_secs(number(a, "--timeout", 1, 24 * 60, 180) * 60),
        out: a.opt("--out").map(PathBuf::from),
        resume: a.flag("--resume"),
    }
}

/// `photos locations`.
pub async fn run(a: &Args) {
    let run = run_of(a);
    if let Err(e) = filmcraft_automation::locations::readiness() {
        fail(format!("photos locations: {e}"));
    }
    let report = tokio::task::spawn_blocking(move || filmcraft_automation::locations::read(&run, &mut |l| diag(format_args!("{l}")), &|| true))
        .await
        .unwrap_or_else(|e| fail(format!("photos locations: {e}")))
        .unwrap_or_else(|e| fail(format!("photos locations: {e}")));
    let out = report["out"].as_str().unwrap_or_default().to_string();
    let mut summary = json!({
        "album": report["album"],
        "albumUrl": report["albumUrl"],
        "out": out,
        "items": report["locations"].as_array().map_or(0, Vec::len),
        "read": report["read"],
        "withoutPlace": report["withoutPlace"],
        "failed": report["failed"],
        "stopped": report["stopped"],
    });
    // the places go with the project's items when a project is open
    if a.opt("--project").is_some() || a.opt("--bridge").is_some() {
        let mut b = Backend::open(a);
        match b.exec("photos.applyLocations", json!({"path": out})).await {
            Ok(v) => summary["applied"] = v,
            Err(e) => diag(format_args!("photos locations: the places were read but not stored with the project's items: {e}")),
        }
        b.finish(a).await;
    }
    let failed = summary["failed"].as_u64().unwrap_or(0);
    diag(format_args!(
        "photos locations: wrote {out} ({} item(s), {} without a place{}){}",
        summary["items"],
        summary["withoutPlace"],
        if failed > 0 { format!(", {failed} unread: run again with --resume") } else { String::new() },
        summary["stopped"].as_str().map(|s| format!("; stopped: {s}")).unwrap_or_default()
    ));
    print(a, &summary);
}

/// `showcase --places-from-locations <file>`: the places of a locations file, for
/// [`filmcraft_autocut::places::fill`].
pub fn places_file(path: &str) -> filmcraft_autocut::places::Places {
    let bytes = std::fs::read(path).unwrap_or_else(|e| fail(format!("showcase: --places-from-locations {path}: {e}")));
    let v: Value = serde_json::from_slice(&bytes).unwrap_or_else(|e| fail(format!("showcase: --places-from-locations {path}: not JSON ({e})")));
    filmcraft_autocut::places::Places::from_json(&v).unwrap_or_else(|e| fail(format!("showcase: --places-from-locations {path}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Args {
        Args::parse(s.split_whitespace().map(str::to_string))
    }

    #[test]
    fn a_run_is_read_from_the_flags() {
        let r =
            run_of(&args("photos locations --album-url https://photos.google.com/album/AF1 --max-items 5 --per-item 30 --timeout 10 --out l.json --resume"));
        assert_eq!(r.album_url.as_deref(), Some("https://photos.google.com/album/AF1"));
        assert_eq!((r.max_items, r.per_item, r.budget), (5, Duration::from_secs(30), Duration::from_secs(600)));
        assert_eq!(r.out.as_deref(), Some(std::path::Path::new("l.json")));
        assert!(r.resume);
        let d = run_of(&args("photos locations --album-url https://photos.google.com/album/AF1"));
        assert_eq!((d.max_items, d.per_item, d.resume), (1000, Duration::from_secs(90), false));
    }
}
