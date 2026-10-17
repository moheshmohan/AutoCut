//! `filmcraft-cli youtube …`: connect YouTube channels, choose the default, upload a file —
//! over `filmcraft-youtube` (progress on stderr, the result as JSON on stdout). Also the upload
//! behind `showcase --upload-youtube`.

use serde_json::Value;

use crate::args::Args;
use crate::{diag, fail, print, usage};
use filmcraft_engine::youtube::{self as yt, ConnectOptions, Privacy, UploadRequest, VideoMeta};

const USAGE: &str =
    "youtube auth [--default] [--playlists] | channels [--json] | default <channel> | disconnect <channel> | quota | upload <file> --title T [--channel C] …";

/// The options `youtube upload` takes.
const UPLOAD_OPTS: &[&str] = &[
    "--title",
    "--channel",
    "--description",
    "--tags",
    "--category",
    "--privacy",
    "--made-for-kids",
    "--publish-at",
    "--playlist",
    "--thumbnail",
    "--no-sidecars",
    "--no-notify",
    "--compact",
];

pub async fn run(a: &Args) {
    let what = a.pos(1).unwrap_or_else(|| usage(USAGE));
    match what {
        "auth" => {
            if let Some(o) = a.unknown_opt(&["--default", "--playlists", "--compact"]) {
                usage(format!("youtube auth: unknown option `{o}` (--default, --playlists)"));
            }
            let opts = ConnectOptions { make_default: a.flag("--default"), playlists: a.flag("--playlists") };
            let r = tokio::task::spawn_blocking(move || {
                let y = filmcraft_youtube::YouTube::new();
                y.connect(opts, &mut |m| diag(format_args!("youtube auth: {m}")))
            })
            .await
            .unwrap_or_else(|e| fail(format!("youtube auth: {e}")));
            match r {
                Ok(c) => {
                    if let Some(n) = &c.note {
                        diag(format_args!("youtube auth: {n}"));
                    }
                    diag(format_args!(
                        "youtube auth: connected {}{}{}; its sign-in is kept in {}",
                        c.channel.label(),
                        if c.replaced { " again" } else { "" },
                        if c.is_default { " (the default channel)" } else { "" },
                        c.place
                    ));
                    print(a, &c.to_json());
                }
                Err(e) => fail(format!("youtube auth: {e}")),
            }
        }
        "channels" | "list" => {
            let l = filmcraft_youtube::YouTube::new().listing().unwrap_or_else(|e| fail(format!("youtube channels: {e}")));
            if a.flag("--json") {
                print(a, &l);
                return;
            }
            if let Some(s) = l["setup"].as_str() {
                diag(format_args!("youtube: {s}"));
            }
            let list = l["channels"].as_array().cloned().unwrap_or_default();
            if list.is_empty() {
                outln!("no YouTube channel connected: run `filmcraft-cli youtube auth` and pick one in Google's account chooser");
            }
            for c in &list {
                outln!(
                    "{} {}{} [{}]{}{}",
                    if c["default"] == true { "*" } else { " " },
                    c["title"].as_str().unwrap_or_default(),
                    c["handle"].as_str().filter(|h| !h.is_empty()).map(|h| format!(" ({h})")).unwrap_or_default(),
                    c["id"].as_str().unwrap_or_default(),
                    if c["playlists"] == true { " playlists" } else { "" },
                    if c["signedIn"] == false { " — not signed in: run `youtube auth` again" } else { "" },
                );
            }
            print_quota(&l["quota"]);
        }
        "quota" => {
            let l = filmcraft_youtube::YouTube::new().quota();
            let v = l.to_json(filmcraft_youtube::now());
            if a.flag("--json") { print(a, &v) } else { print_quota(&v) }
        }
        "default" => {
            let c = channel_arg(a, "default");
            match filmcraft_youtube::YouTube::new().set_default(&c) {
                Ok(ch) => outln!("uploads go to {} [{}] unless --channel says otherwise", ch.label(), ch.id),
                Err(e) => fail(format!("youtube default: {e}")),
            }
        }
        "disconnect" => {
            let c = channel_arg(a, "disconnect");
            let r = tokio::task::spawn_blocking(move || filmcraft_youtube::YouTube::new().disconnect(&c))
                .await
                .unwrap_or_else(|e| fail(format!("youtube disconnect: {e}")));
            match r {
                Ok(d) => {
                    if let Err(e) = &d.revoked {
                        diag(format_args!("youtube disconnect: Google did not confirm the revocation: {e}"));
                    }
                    outln!(
                        "disconnected {}{}{}",
                        d.channel.label(),
                        if d.revoked.is_ok() { " (revoked at Google)" } else { " here" },
                        d.default.as_ref().map(|c| format!("; the default is now {}", c.label())).unwrap_or_default()
                    );
                }
                Err(e) => fail(format!("youtube disconnect: {e}")),
            }
        }
        "upload" => {
            let path = a
                .pos(2)
                .unwrap_or_else(|| usage("youtube upload <file> --title T [--channel C] [--description D] [--tags a,b] [--privacy private|unlisted|public] …"));
            if a.positionals.len() > 3 {
                usage("youtube upload: one file at a time");
            }
            if let Some(o) = a.unknown_opt(UPLOAD_OPTS) {
                usage(format!("youtube upload: unknown option `{o}`"));
            }
            let req = upload_request(a, &crate::absolute(path)).unwrap_or_else(|e| usage(format!("youtube upload: {e}")));
            match upload(req).await {
                Ok(v) => print(a, &v),
                Err(e) => fail(format!("youtube upload: {e}")),
            }
        }
        other => usage(format!("youtube: unknown `{other}` ({USAGE})")),
    }
}

fn channel_arg(a: &Args, what: &str) -> String {
    let c = a.positionals.get(2..).map(|p| p.join(" ")).unwrap_or_default();
    yt::check_ref("channel", &c).unwrap_or_else(|_| usage(format!("youtube {what} <channel> (its id, @handle or title: `youtube channels` lists them)")))
}

fn print_quota(q: &Value) {
    outln!(
        "quota today ({} Pacific): {} of {} units, {} of {} uploads; resets in {} — FilmCraft's own count{}",
        q["day"].as_str().unwrap_or_default(),
        q["units"],
        filmcraft_youtube::quota::DAILY_UNITS,
        q["uploads"],
        filmcraft_youtube::quota::DAILY_UPLOADS,
        q["resetsIn"].as_str().unwrap_or_default(),
        q["exceeded"]["reason"].as_str().map(|r| format!("; Google said {r} today")).unwrap_or_default()
    );
}

/// `youtube upload`'s options as a checked request (the same checks as the engine's).
pub fn upload_request(a: &Args, path: &str) -> Result<UploadRequest, String> {
    let title = a.opt("--title").ok_or("--title is needed: the video's title (at most 100 characters)")?;
    meta_request(a, path, title, a.opt("--description"), "--channel", "--privacy", "--playlist", "--tags")
}

/// A request from the options named (the showcase spells them `--youtube-…`).
#[allow(clippy::too_many_arguments)]
pub fn meta_request(
    a: &Args,
    path: &str,
    title: &str,
    description: Option<&str>,
    channel: &str,
    privacy: &str,
    playlist: &str,
    tags: &str,
) -> Result<UploadRequest, String> {
    let title = yt::check_title(title)?;
    let description = yt::check_description(description)?;
    let tags = match a.opt(tags) {
        Some(t) => yt::check_tags(&t.split(',').map(str::to_string).collect::<Vec<_>>())?,
        None => Vec::new(),
    };
    let privacy = a.opt(privacy).map(Privacy::parse).transpose()?.unwrap_or_default();
    let publish_at = a.opt("--publish-at").map(yt::check_publish_at).transpose()?;
    if publish_at.is_some() && privacy != Privacy::Private {
        return Err("--publish-at schedules a private video to go public: leave --privacy private".into());
    }
    Ok(UploadRequest {
        path: path.to_string(),
        channel: a.opt(channel).map(|c| yt::check_ref("channel", c)).transpose()?,
        meta: VideoMeta {
            title,
            description,
            tags,
            category: yt::check_category(a.opt("--category"))?,
            privacy,
            made_for_kids: a.flag("--made-for-kids"),
            publish_at,
            notify_subscribers: !a.flag("--no-notify"),
        },
        playlist: a.opt(playlist).map(|p| yt::check_ref("playlist", p)).transpose()?,
        thumbnail: a.opt("--thumbnail").map(crate::absolute),
        sidecars: !a.flag("--no-sidecars"),
    })
}

/// Whether an upload could start, from disk (no network): the client file and the channel.
pub fn readiness(req: &UploadRequest) -> Result<(), String> {
    let y = filmcraft_youtube::YouTube::new();
    y.upload_readiness()?;
    let book = y.store().load().map_err(|e| e.to_string())?;
    book.find(req.channel.as_deref()).map(drop).map_err(|e| e.to_string())
}

/// Upload with progress on stderr (a line per status change, at most two a second while the
/// bytes go); the video as JSON.
pub async fn upload(req: UploadRequest) -> Result<Value, String> {
    tokio::task::spawn_blocking(move || {
        let y = filmcraft_youtube::YouTube::new();
        let pre = y.preflight(&req).map_err(|e| e.to_string())?;
        diag(format_args!("youtube upload: to {} — {}", pre.channel.label(), pre.shorts_note));
        for n in &pre.notes {
            diag(format_args!("youtube upload: {n}"));
        }
        if req.meta.privacy == Privacy::Private && req.meta.publish_at.is_none() {
            diag(format_args!("youtube upload: private: only you see it until you change it in YouTube Studio"));
        }
        let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
        let mut last: Option<(u64, String, std::time::Instant)> = None;
        let mut progress = |sent: u64, total: u64, status: &str| {
            let pct = sent.saturating_mul(100).checked_div(total).unwrap_or(0);
            let show = last.as_ref().is_none_or(|(p, s, at)| s != status || (*p != pct && (pct == 100 || at.elapsed().as_millis() >= 500)));
            if show {
                diag(format_args!("youtube upload: {pct:>3}% ({:.1} of {:.1} MiB) {status}", mib(sent), mib(total)));
                last = Some((pct, status.to_string(), std::time::Instant::now()));
            }
            true
        };
        let up = y.upload(&req, &mut progress).map_err(|e| e.to_string())?;
        for n in &up.notes {
            diag(format_args!("youtube upload: {n}"));
        }
        if let Some(Err(e)) = &up.thumbnail {
            diag(format_args!("youtube upload: the thumbnail was not set: {e}"));
        }
        if let Some((p, Err(e))) = &up.playlist {
            diag(format_args!("youtube upload: not added to the playlist {}: {e}", p.title));
        }
        diag(format_args!("youtube upload: {} ({})", up.url(), up.video.privacy_status));
        Ok(up.to_json())
    })
    .await
    .map_err(|e| format!("internal error: {e}"))?
}

/// `showcase --upload-youtube`: the request for the export, titled like the video, its chapters
/// and thumbnail from beside it (or the build's thumbnail).
pub fn showcase_request(a: &Args, export: &str, title: &str, thumbnail: Option<&str>) -> Result<UploadRequest, String> {
    let description = a.opt("--youtube-description").or(a.opt("--description"));
    // YouTube's title limit: the show's title, cut at 100 characters
    let title: String = if title.trim().is_empty() { "Showcase".into() } else { title.trim().chars().take(yt::MAX_TITLE_CHARS).collect() };
    let title = title.as_str();
    let mut req =
        meta_request(a, &crate::absolute(export), title, description, "--youtube-channel", "--youtube-privacy", "--youtube-playlist", "--youtube-tags")?;
    req.thumbnail = thumbnail.map(crate::absolute);
    Ok(req)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &[&str]) -> Args {
        Args::parse(s.iter().map(|x| x.to_string()))
    }

    #[test]
    fn upload_options_become_a_checked_request() {
        let a = args(&[
            "youtube",
            "upload",
            "a.mp4",
            "--title",
            "Every Robot",
            "--channel",
            "@H4hacks",
            "--tags",
            "robots, expo 2020",
            "--privacy",
            "unlisted",
            "--category",
            "Science & Technology",
            "--no-notify",
        ]);
        let r = upload_request(&a, "/x/a.mp4").unwrap();
        assert_eq!((r.meta.title.as_str(), r.channel.as_deref(), r.meta.tags.len(), r.meta.privacy), ("Every Robot", Some("@H4hacks"), 2, Privacy::Unlisted));
        assert_eq!((r.meta.category.as_str(), r.meta.notify_subscribers, r.sidecars, r.meta.made_for_kids), ("28", false, true, false));
        for (bad, want) in [
            (vec!["youtube", "upload", "a.mp4"], "--title"),
            (vec!["youtube", "upload", "a.mp4", "--title", "<x>"], "< or >"),
            (vec!["youtube", "upload", "a.mp4", "--title", "T", "--privacy", "secret"], "privacy"),
            (vec!["youtube", "upload", "a.mp4", "--title", "T", "--publish-at", "2026-11-01T10:00:00Z", "--privacy", "public"], "leave --privacy private"),
            (vec!["youtube", "upload", "a.mp4", "--title", "T", "--category", "Cooking"], "category"),
        ] {
            let e = upload_request(&args(&bad), "/x/a.mp4").unwrap_err();
            assert!(e.contains(want), "{bad:?}: {e}");
        }
        let s = showcase_request(
            &args(&["showcase", "--youtube-channel", "Mohesh Mohan", "--description", "D"]),
            "/o/show.mp4",
            "Every Robot",
            Some("/o/show.thumbnail.jpg"),
        )
        .unwrap();
        assert_eq!(
            (s.channel.as_deref(), s.meta.description.as_deref(), s.thumbnail.as_deref()),
            (Some("Mohesh Mohan"), Some("D"), Some("/o/show.thumbnail.jpg"))
        );
    }
}
