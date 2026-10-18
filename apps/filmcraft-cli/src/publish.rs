//! `filmcraft-cli publish …`: the destinations' pre-flight (`publish check`), a fixed copy
//! (`publish fix`) and a publish that checks first (`publish run`) — over the engine's
//! `publish.*` commands, then `youtube upload` / `instagram publish` (progress on stderr).

use serde_json::{Value, json};

use crate::args::Args;
use crate::{diag, fail, print, usage};
use filmcraft_engine::Session;
use filmcraft_engine::publish::Destination;

const USAGE: &str = "publish destinations | check <destination> <file> [--no-loudness] [--json] | fix <destination> <file> [--out F] | run <destination> <file> [youtube upload / instagram publish options] [--no-preflight]";

fn destination(a: &Args) -> Destination {
    let d = a.pos(2).unwrap_or_else(|| usage(format!("publish: name the destination: youtube, youtubeShort or instagramReel ({USAGE})")));
    Destination::parse(d).unwrap_or_else(|e| usage(format!("publish: {e}")))
}

fn file(a: &Args) -> String {
    let f = a.pos(3).unwrap_or_else(|| usage(format!("publish: name the file ({USAGE})")));
    if a.positionals.len() > 4 {
        usage("publish: one file at a time");
    }
    crate::absolute(f)
}

pub async fn run(a: &Args) {
    let what = a.pos(1).unwrap_or_else(|| usage(USAGE));
    match what {
        "destinations" => match backend_session().execute("publish.destinations", json!({})) {
            Ok(v) => print(a, &v),
            Err(e) => fail(format!("publish destinations: {e}")),
        },
        "check" | "preflight" => {
            if let Some(o) = a.unknown_opt(&["--no-loudness", "--json", "--compact"]) {
                usage(format!("publish check: unknown option `{o}`"));
            }
            let (d, path) = (destination(a), file(a));
            let r = preflight(d, &path, !a.flag("--no-loudness")).unwrap_or_else(|e| fail(format!("publish check: {e}")));
            if a.flag("--json") {
                print(a, &r);
            } else {
                print_report(&r);
            }
            if r["ok"] != true {
                std::process::exit(1);
            }
        }
        "fix" => {
            if let Some(o) = a.unknown_opt(&["--out", "--compact"]) {
                usage(format!("publish fix: unknown option `{o}`"));
            }
            let (d, path) = (destination(a), file(a));
            let mut p = json!({"destination": d.id(), "path": path, "wait": true});
            if let Some(o) = a.opt("--out") {
                p["out"] = json!(crate::absolute(o));
            }
            diag(format_args!("publish fix: writing a copy {} takes (this re-encodes when a check needs it)", d.label()));
            let r = tokio::task::spawn_blocking(move || Session::default().execute("publish.fixCopy", p).map_err(|e| e.to_string()))
                .await
                .unwrap_or_else(|e| fail(format!("publish fix: {e}")));
            match r {
                Ok(v) => {
                    let out = &v["publish"];
                    diag(format_args!(
                        "publish fix: {} ({})",
                        out["path"].as_str().unwrap_or_default(),
                        out["fixes"].as_array().map(|f| f.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", ")).unwrap_or_default()
                    ));
                    if out["preflight"].is_object() {
                        print_report(&out["preflight"]);
                    }
                    print(a, &v);
                }
                Err(e) => fail(format!("publish fix: {e}")),
            }
        }
        "run" => {
            let (d, path) = (destination(a), file(a));
            if !a.flag("--no-preflight") {
                let r = preflight(d, &path, false).unwrap_or_else(|e| fail(format!("publish run: {e}")));
                print_report(&r);
                if r["ok"] != true {
                    fail(format!(
                        "publish run: {} does not pass the {} pre-flight; `filmcraft-cli publish fix {} {}` writes a copy that does where the checks say so",
                        path,
                        d.label(),
                        d.id(),
                        path
                    ));
                }
            }
            let result = match d {
                Destination::MediaFile => usage("publish run: the Media File is the export itself; pick youtube, youtubeShort or instagramReel"),
                Destination::Youtube | Destination::YoutubeShort => {
                    let stem: String = std::path::Path::new(&path).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                    let title =
                        a.opt("--title").map(str::to_string).unwrap_or(stem).chars().take(filmcraft_engine::youtube::MAX_TITLE_CHARS).collect::<String>();
                    let req = crate::youtube::meta_request(a, &path, &title, a.opt("--description"), "--channel", "--privacy", "--playlist", "--tags")
                        .unwrap_or_else(|e| usage(format!("publish run: {e}")));
                    crate::youtube::upload(req).await
                }
                Destination::InstagramReel => {
                    let req = crate::instagram::publish_request(a, Some(&path)).unwrap_or_else(|e| usage(format!("publish run: {e}")));
                    // checked above (or skipped on purpose)
                    crate::instagram::publish(req, true).await
                }
            };
            match result {
                Ok(v) => print(a, &json!({"destination": d.id(), "result": v})),
                Err(e) => fail(format!("publish run: {e}")),
            }
        }
        other => usage(format!("publish: unknown `{other}` ({USAGE})")),
    }
}

/// A session with the publishers (for the connect states).
fn backend_session() -> Session {
    let mut s = Session::default();
    s.youtube.publisher = Some(filmcraft_youtube::engine_publisher());
    s.instagram.publisher = Some(filmcraft_instagram::engine_publisher());
    s
}

/// `publish.preflight` in a fresh session.
pub fn preflight(d: Destination, path: &str, loudness: bool) -> Result<Value, String> {
    Session::default().execute("publish.preflight", json!({"destination": d.id(), "path": path, "loudness": loudness})).map_err(|e| e.to_string())
}

/// The checklist on stderr: one line per check, the fix under what does not pass.
pub fn print_report(r: &Value) {
    diag(format_args!(
        "{} pre-flight of {}: {} error(s), {} warning(s)",
        r["label"].as_str().unwrap_or_default(),
        r["path"].as_str().unwrap_or_default(),
        r["errors"],
        r["warnings"]
    ));
    for c in r["checks"].as_array().into_iter().flatten() {
        let mark = match c["status"].as_str() {
            Some("pass") => "  ok  ",
            Some("warn") => "  WARN",
            Some("fail") => "  FAIL",
            _ => "  note",
        };
        diag(format_args!(
            "{mark} {}: {} (takes {})",
            c["label"].as_str().unwrap_or_default(),
            c["value"].as_str().unwrap_or_default(),
            c["need"].as_str().unwrap_or_default()
        ));
        if c["status"] != "pass"
            && let Some(f) = c["fix"].as_str()
        {
            diag(format_args!("         → {f}"));
        }
    }
}
