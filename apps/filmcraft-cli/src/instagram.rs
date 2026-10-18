//! `filmcraft-cli instagram …`: set up the Meta app, connect the Instagram account (browser
//! sign-in or a pasted dashboard token), see it and its limits, publish a Reel — over
//! `filmcraft-instagram` (progress on stderr, the result as JSON on stdout).
//!
//! Secrets (the app secret, a token) are read from standard input — piped, or typed at a
//! prompt that does not show them — never from the command line (other users can list it) and
//! never printed.

use std::io::{BufRead as _, IsTerminal as _, Read as _};

use serde_json::Value;

use crate::args::Args;
use crate::{diag, fail, print, usage};
use filmcraft_engine::instagram::{self as ig, ReelRequest};

const USAGE: &str = "instagram setup --app-id ID [--redirect-uri URI] | token | auth | account [--json] | refresh | limits [--json] | disconnect | publish <file> [--caption C] …";

/// The options `instagram publish` takes.
const PUBLISH_OPTS: &[&str] = &["--caption", "--cover-at", "--cover-url", "--no-feed", "--video-url", "--upload-only", "--no-preflight", "--compact", "--json"];

/// The longest secret line read.
const MAX_SECRET_LINE: u64 = 16 * 1024;

pub async fn run(a: &Args) {
    let what = a.pos(1).unwrap_or_else(|| usage(USAGE));
    if a.opt("--secret").is_some() || a.opt("--token").is_some() || a.flag("--secret") || a.flag("--token") {
        usage("instagram: secrets are never taken on the command line (other users can see it): pipe them in, or type them at the prompt");
    }
    match what {
        "setup" => setup(a),
        "token" => {
            if let Some(o) = a.unknown_opt(&["--compact"]) {
                usage(format!("instagram token: unknown option `{o}`"));
            }
            let token = read_secret(
                "Paste the access token from App Dashboard ▸ Instagram ▸ API setup with Instagram login ▸ Generate access tokens (it is not shown): ",
            )
            .unwrap_or_else(|e| fail(format!("instagram token: {e}")));
            let r = tokio::task::spawn_blocking(move || {
                filmcraft_instagram::Instagram::new().connect_token(&token, &mut |m| diag(format_args!("instagram token: {m}")))
            })
            .await
            .unwrap_or_else(|e| fail(format!("instagram token: {e}")));
            connected(a, "instagram token", r);
        }
        "auth" | "connect" => {
            if let Some(o) = a.unknown_opt(&["--compact"]) {
                usage(format!("instagram auth: unknown option `{o}`"));
            }
            let r = tokio::task::spawn_blocking(move || {
                let i = filmcraft_instagram::Instagram::new();
                let mut paste = |url: &str| -> Option<String> {
                    diag(format_args!("instagram auth: open this link, log in as the professional account and allow FilmCraft:\n{url}"));
                    diag(format_args!("instagram auth: then copy the whole address your browser shows and paste it here:"));
                    let mut line = String::new();
                    std::io::stdin().lock().take(MAX_SECRET_LINE).read_line(&mut line).ok()?;
                    Some(line.trim().to_string()).filter(|l| !l.is_empty())
                };
                i.connect(Some(&mut paste), &mut |m| diag(format_args!("instagram auth: {m}")))
            })
            .await
            .unwrap_or_else(|e| fail(format!("instagram auth: {e}")));
            connected(a, "instagram auth", r);
        }
        "account" | "status" => {
            let l = filmcraft_instagram::Instagram::new().listing().unwrap_or_else(|e| fail(format!("instagram account: {e}")));
            if a.flag("--json") {
                return print(a, &l);
            }
            describe_account(&l);
        }
        "refresh" => {
            let r = tokio::task::spawn_blocking(|| filmcraft_instagram::Instagram::new().refresh_now())
                .await
                .unwrap_or_else(|e| fail(format!("instagram refresh: {e}")));
            match r {
                Ok(acct) => outln!(
                    "refreshed the sign-in of {}: it lasts {} more days",
                    acct.label(),
                    acct.expires_at.saturating_sub(filmcraft_instagram::now()) / 86_400
                ),
                Err(e) => fail(format!("instagram refresh: {e}")),
            }
        }
        "limits" => {
            let r =
                tokio::task::spawn_blocking(|| filmcraft_instagram::Instagram::new().limits()).await.unwrap_or_else(|e| fail(format!("instagram limits: {e}")));
            let v = r.unwrap_or_else(|e| fail(format!("instagram limits: {e}")));
            if a.flag("--json") {
                return print(a, &v);
            }
            print_limits(&v);
        }
        "disconnect" => match filmcraft_instagram::Instagram::new().disconnect() {
            Ok(v) => {
                outln!("disconnected {}", v["disconnected"]["username"].as_str().map(|u| format!("@{u}")).unwrap_or_else(|| "the account".into()));
                diag(format_args!("instagram disconnect: {}", v["note"].as_str().unwrap_or_default()));
            }
            Err(e) => fail(format!("instagram disconnect: {e}")),
        },
        "publish" | "upload" => {
            if a.positionals.len() > 3 {
                usage("instagram publish: one file at a time");
            }
            if let Some(o) = a.unknown_opt(PUBLISH_OPTS) {
                usage(format!("instagram publish: unknown option `{o}`"));
            }
            let path = a.pos(2).map(crate::absolute);
            let req = publish_request(a, path.as_deref()).unwrap_or_else(|e| usage(format!("instagram publish: {e}")));
            match publish(req, a.flag("--no-preflight")).await {
                Ok(v) => print(a, &v),
                Err(e) => fail(format!("instagram publish: {e}")),
            }
        }
        other => usage(format!("instagram: unknown `{other}` ({USAGE})")),
    }
}

fn setup(a: &Args) {
    if let Some(o) = a.unknown_opt(&["--app-id", "--redirect-uri", "--forget", "--compact"]) {
        usage(format!("instagram setup: unknown option `{o}` (--app-id, --redirect-uri, --forget)"));
    }
    let i = filmcraft_instagram::Instagram::new();
    if a.flag("--forget") {
        match filmcraft_instagram::auth::AppStore::system(i.dir()).remove() {
            Ok(()) => return outln!("forgot the Instagram app (its id, redirect URI and secret)"),
            Err(e) => fail(format!("instagram setup --forget: {e}")),
        }
    }
    let app_id = a.opt("--app-id").unwrap_or_else(|| {
        usage("instagram setup --app-id <Instagram app ID> [--redirect-uri http://localhost:8714/]: the Instagram app ID from App Dashboard ▸ Instagram ▸ API setup with Instagram login (not the Meta app ID); the secret is read from standard input")
    });
    let app_id = filmcraft_instagram::auth::check_app_id(app_id).unwrap_or_else(|e| usage(format!("instagram setup: {e}")));
    let redirect = a.opt("--redirect-uri").unwrap_or(filmcraft_instagram::auth::DEFAULT_REDIRECT);
    let (redirect, _) = filmcraft_instagram::auth::check_redirect(redirect).unwrap_or_else(|e| usage(format!("instagram setup: {e}")));
    let secret = read_secret(
        "Paste the Instagram app secret (App Dashboard ▸ Instagram ▸ API setup with Instagram login ▸ Instagram app secret ▸ Show; it is not shown here): ",
    )
    .unwrap_or_else(|e| fail(format!("instagram setup: {e}")));
    match i.setup(&app_id, &redirect, &secret) {
        Ok(place) => {
            outln!("set up the Instagram app {app_id}; its secret is kept in {place}");
            diag(format_args!(
                "instagram setup: in the Meta App Dashboard ▸ Instagram ▸ API setup with Instagram login ▸ Set up Instagram business login, add exactly this OAuth redirect URI: {redirect}"
            ));
            diag(format_args!("instagram setup: then connect: `filmcraft-cli instagram auth`"));
        }
        Err(e) => fail(format!("instagram setup: {e}")),
    }
}

/// A secret from standard input: the first line when it is piped; typed at a prompt that does
/// not echo when it is a terminal.
fn read_secret(prompt: &str) -> Result<String, String> {
    let stdin = std::io::stdin();
    let mut line = String::new();
    if !stdin.is_terminal() {
        stdin.lock().take(MAX_SECRET_LINE).read_line(&mut line).map_err(|e| format!("reading standard input: {e}"))?;
        return Ok(line.trim().to_string()).and_then(|l| if l.is_empty() { Err("nothing came on standard input".into()) } else { Ok(l) });
    }
    diag(format_args!("{prompt}"));
    let hidden = set_echo(false);
    let r = stdin.lock().take(MAX_SECRET_LINE).read_line(&mut line);
    if hidden {
        set_echo(true);
        diag(format_args!(""));
    }
    r.map_err(|e| format!("reading the prompt: {e}"))?;
    let l = line.trim().to_string();
    if l.is_empty() { Err("nothing was typed".into()) } else { Ok(l) }
}

/// Turn the terminal's echo off or on with `stty` (no `unsafe`, no new dependency). `false` when
/// it could not (Windows, no `/dev/tty`): the typing then shows.
fn set_echo(on: bool) -> bool {
    #[cfg(unix)]
    {
        let Ok(tty) = std::fs::File::open("/dev/tty") else { return false };
        std::process::Command::new("stty")
            .arg(if on { "echo" } else { "-echo" })
            .stdin(tty)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }
    #[cfg(not(unix))]
    {
        let _ = on;
        false
    }
}

fn connected(a: &Args, what: &str, r: Result<filmcraft_instagram::Connected, filmcraft_instagram::IgError>) {
    match r {
        Ok(c) => {
            if let Some(n) = &c.note {
                diag(format_args!("{what}: {n}"));
            }
            if let Some(p) = &c.replaced {
                diag(format_args!("{what}: {p} was connected before; it is replaced"));
            }
            diag(format_args!(
                "{what}: connected {} ({}); the sign-in lasts {} days and is refreshed while FilmCraft is used; kept in {}",
                c.account.label(),
                c.account.account_type,
                c.account.expires_at.saturating_sub(filmcraft_instagram::now()) / 86_400,
                c.place
            ));
            print(a, &c.to_json());
        }
        Err(e) => fail(format!("{what}: {e}")),
    }
}

fn describe_account(l: &Value) {
    if let Some(s) = l["setup"].as_str() {
        diag(format_args!("instagram: {s}"));
    }
    match l["account"].as_object() {
        None => outln!("no Instagram account connected: run `filmcraft-cli instagram auth` (or `instagram token`)"),
        Some(acct) => {
            let username = acct.get("username").and_then(Value::as_str).unwrap_or_default();
            let kind = acct.get("accountType").and_then(Value::as_str).unwrap_or_default();
            let days = acct.get("expiresInDays").and_then(Value::as_u64).unwrap_or(0);
            outln!(
                "@{username} ({kind}), connected by {}; the sign-in {}",
                acct.get("via").and_then(Value::as_str).unwrap_or("login"),
                if acct.get("expired") == Some(&Value::Bool(true)) { "ran out: connect again".to_string() } else { format!("lasts {days} more days") }
            );
        }
    }
    print_limits(&l["limits"]);
}

fn print_limits(v: &Value) {
    outln!(
        "last 24 h from this machine: {} of {} posts, {} of {} containers{}",
        v["posts"],
        v["postQuota"],
        v["containers"],
        v["containerQuota"],
        v["instagram"]["quotaUsage"].as_u64().map(|u| format!("; Instagram counted {u} of {} posts", v["instagram"]["quotaTotal"])).unwrap_or_default()
    );
}

/// `instagram publish`'s options as a checked request (the same checks as the engine's).
pub fn publish_request(a: &Args, path: Option<&str>) -> Result<ReelRequest, String> {
    let video_url = a.opt("--video-url").map(|u| ig::check_public_url("--video-url", u)).transpose()?;
    if path.is_none() && video_url.is_none() {
        return Err("give the video: a file (sent from here), or --video-url https://… (a public address Instagram fetches it from)".into());
    }
    let cover_ms = match a.opt("--cover-at") {
        Some(s) => {
            let secs: f64 = s.trim().parse().map_err(|_| format!("--cover-at `{s}` is not a number of seconds"))?;
            if !secs.is_finite() || !(0.0..=ig::MAX_REEL_SECONDS).contains(&secs) {
                return Err(format!("--cover-at must be 0 to {:.0} seconds into the video", ig::MAX_REEL_SECONDS));
            }
            Some((secs * 1000.0).round() as u64)
        }
        None => None,
    };
    Ok(ReelRequest {
        path: path.map(str::to_string),
        video_url,
        caption: ig::check_caption(a.opt("--caption"))?,
        share_to_feed: !a.flag("--no-feed"),
        cover_ms,
        cover_url: a.opt("--cover-url").map(|u| ig::check_public_url("--cover-url", u)).transpose()?,
        upload_only: a.flag("--upload-only"),
    })
}

/// Publish with progress on stderr (a line per status change, at most two a second while the
/// bytes go); the Reel as JSON.
pub async fn publish(req: ReelRequest, skip_preflight: bool) -> Result<Value, String> {
    tokio::task::spawn_blocking(move || {
        let i = filmcraft_instagram::Instagram::new();
        let pre = i.preflight(&req).map_err(|e| e.to_string())?;
        // the shared pre-flight of the file (unless only a URL is published)
        if let Some(path) = req.path.as_deref().filter(|_| !skip_preflight) {
            let r = crate::publish::preflight(filmcraft_engine::publish::Destination::InstagramReel, path, false)?;
            crate::publish::print_report(&r);
            if r["ok"] != true {
                return Err(format!("{path} does not pass the Instagram Reel pre-flight; `filmcraft-cli publish fix instagramReel {path}` writes a copy that does where the checks say so (or --no-preflight)"));
            }
        }
        diag(format_args!("instagram publish: to {} — {}", pre.account.label(), req.source()));
        for n in &pre.notes {
            diag(format_args!("instagram publish: {n}"));
        }
        let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
        let mut last: Option<(u64, String, std::time::Instant)> = None;
        let mut progress = |sent: u64, total: u64, status: &str| {
            let pct = sent.saturating_mul(100).checked_div(total).unwrap_or(0);
            let show = last.as_ref().is_none_or(|(p, s, at)| s != status || (*p != pct && (pct == 100 || at.elapsed().as_millis() >= 500)));
            if show {
                diag(format_args!("instagram publish: {pct:>3}% ({:.1} of {:.1} MiB) {status}", mib(sent), mib(total)));
                last = Some((pct, status.to_string(), std::time::Instant::now()));
            }
            true
        };
        let p = i.publish(&req, &mut progress).map_err(|e| e.to_string())?;
        for n in &p.notes {
            diag(format_args!("instagram publish: {n}"));
        }
        match &p.media_id {
            Some(id) => diag(format_args!("instagram publish: published {}", p.permalink.as_deref().unwrap_or(id))),
            None => diag(format_args!(
                "instagram publish: uploaded and processed, not published (container {}); run the same command without --upload-only within 23 hours to publish it",
                p.container
            )),
        }
        Ok(p.to_json())
    })
    .await
    .map_err(|e| format!("internal error: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &[&str]) -> Args {
        Args::parse(s.iter().map(|x| x.to_string()))
    }

    #[test]
    fn publish_options_become_a_checked_request() {
        let a = args(&["instagram", "publish", "a.mp4", "--caption", "Robots #ai", "--cover-at", "1.5", "--no-feed"]);
        let r = publish_request(&a, Some("/x/a.mp4")).unwrap();
        assert_eq!((r.caption.as_deref(), r.cover_ms, r.share_to_feed, r.video_url), (Some("Robots #ai"), Some(1500), false, None));
        let r = publish_request(&args(&["instagram", "publish", "--video-url", "https://cdn.example.com/a.mp4"]), None).unwrap();
        assert_eq!(r.video_url.as_deref(), Some("https://cdn.example.com/a.mp4"));
        for (bad, want) in [
            (vec!["instagram", "publish"], "give the video"),
            (vec!["instagram", "publish", "--video-url", "http://localhost/a.mp4"], "public"),
            (vec!["instagram", "publish", "--video-url", "https://localhost/a.mp4"], "public server"),
            (vec!["instagram", "publish", "a.mp4", "--cover-at", "-1"], "--cover-at"),
            (vec!["instagram", "publish", "a.mp4", "--cover-at", "NaN"], "--cover-at"),
            (vec!["instagram", "publish", "a.mp4", "--cover-url", "ftp://x"], "public"),
        ] {
            let path = bad.get(2).filter(|p| !p.starts_with("--")).map(|_| "/x/a.mp4");
            let e = publish_request(&args(&bad), path).unwrap_err();
            assert!(e.contains(want), "{bad:?}: {e}");
        }
        let many = "#x ".repeat(31);
        assert!(publish_request(&args(&["instagram", "publish", "a.mp4", "--caption", &many]), Some("/x/a.mp4")).unwrap_err().contains("hashtags"));
    }
}
