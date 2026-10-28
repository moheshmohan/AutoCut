//! Insta360 footage: the user's 360° recordings, from where Insta360 Studio downloads its cloud
//! files (or the camera's card) into the project.
//!
//! | command | does |
//! |---|---|
//! | `insta360.list` | the recordings in `folders` (default: Studio's download folder and every mounted camera card), newest first: stem, when it was shot, file, size, proxy, whether it has finished downloading, the camera |
//! | `insta360.import` | imports recordings (`names` — full stems like `VID_20260913_114003_00_093` or Studio's short `VID_093` — or `since`/`until` dates, or `all`) as 360° media: VR Properties from the file, the camera's `.lrv` attached as the proxy (`proxies`), the lenses aligned (`align`); a file already in the project is reused; optionally `reframe` puts them in the active sequence reframed |
//! | `insta360.download` | asks Insta360 Studio (signed in to the user's Insta360 cloud) to download `names`, driving its Cloud Media list with the local vision agent (the `chrome` lock, as every agent that moves the mouse), as a job (`jobs.list`); with `wait`, waits for the files and imports them (`import`, default on) |
//!
//! The engine only runs them: finding files and driving Studio is `filmcraft-insta360`'s, a layer
//! above, which the host installs as [`Session::insta360`]. Without one (the web build, tests)
//! the commands are disabled and say so.

use std::sync::PoisonError;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, bool_p, str_p};
use crate::{EngineError, Result, Session};

/// What the commands say without a host.
const NOT_HERE: &str = "Insta360 footage is not available in this app (no file access)";
/// Most names one command takes.
pub const MAX_NAMES: usize = 1000;

/// One recording, as the host found it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Recording {
    pub stem: String,
    /// `video` or `photo`.
    pub kind: String,
    /// When it started, camera time: `2026-09-13T11:40:03`.
    pub shot: String,
    pub path: String,
    pub bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub second_lens: Option<String>,
    pub complete: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<String>,
}

/// What Studio did with one wanted recording.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Downloaded {
    pub name: String,
    pub requested: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
}

/// Finding Insta360 files and driving Insta360 Studio, installed by the host.
pub trait Insta360Host: Send + Sync {
    /// Where recordings are looked for by default.
    fn default_folders(&self) -> Vec<String>;
    /// The recordings under `folders`, newest first.
    fn scan(&self, folders: &[String]) -> Vec<Recording>;
    /// Whether Studio can be driven now (installed and running, the vision model there, the
    /// mouse free). `Err` says why not and what fixes it. Cheap.
    fn studio_ready(&self) -> std::result::Result<(), String>;
    /// Asks Studio to download `names` and, with `wait`, waits for them (blocking; `progress`
    /// for status, cancel when it is set).
    fn download(&self, names: &[String], wait: bool, progress: &filmcraft_export::Progress) -> std::result::Result<Vec<Downloaded>, String>;
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "insta360.list",
            label: "Insta360 Recordings",
            menu: &[],
            shortcut: None,
            params: r#"{"folders":[str]?,"since":"YYYY-MM-DD"?,"until":"YYYY-MM-DD"?}"#,
            enabled: host_ready,
            run: list,
            journal: false,
        },
        CommandSpec {
            id: "insta360.import",
            label: "Import Insta360 Recordings…",
            menu: &["File"],
            shortcut: None,
            params: r#"{"names":[str]?,"folders":[str]?,"since":"YYYY-MM-DD"?,"until":"YYYY-MM-DD"?,"all":bool=false,"proxies":bool=true,"align":bool=true,"bin":binId?,"reframe":bool=false}"#,
            enabled: host_ready,
            run: import,
            journal: true,
        },
        CommandSpec {
            id: "insta360.download",
            label: "Download from Insta360 Cloud…",
            menu: &["File"],
            shortcut: None,
            params: r#"{"names":[str],"wait":bool=false,"import":bool=true,"proxies":bool=true,"align":bool=true}"#,
            enabled: host_ready,
            run: download,
            journal: true,
        },
    ]
}

fn host_ready(s: &Session) -> std::result::Result<(), String> {
    s.insta360.as_ref().map(|_| ()).ok_or_else(|| NOT_HERE.into())
}

fn host(s: &Session, cmd: &str) -> Result<std::sync::Arc<dyn Insta360Host>> {
    s.insta360.clone().ok_or_else(|| bad(cmd, NOT_HERE))
}

/// A list of strings (`key`), capped; absent is empty.
fn strings(p: &Value, key: &str, cmd: &str) -> Result<Vec<String>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(a)) if a.len() <= MAX_NAMES => a
            .iter()
            .map(|v| v.as_str().map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).ok_or_else(|| bad(cmd, format!("`{key}` must be a list of names"))))
            .collect(),
        Some(Value::Array(_)) => Err(bad(cmd, format!("at most {MAX_NAMES} `{key}`"))),
        Some(_) => Err(bad(cmd, format!("`{key}` must be a list"))),
    }
}

/// A `YYYY-MM-DD` date parameter.
fn date_p(p: &Value, key: &str, cmd: &str) -> Result<Option<chrono::NaiveDate>> {
    match str_p(p, key) {
        None => Ok(None),
        Some(d) => chrono::NaiveDate::parse_from_str(d.trim(), "%Y-%m-%d").map(Some).map_err(|_| bad(cmd, format!("`{key}` is a date, YYYY-MM-DD"))),
    }
}

/// The recordings the parameters choose (from the host's scan).
fn chosen(s: &Session, p: &Value, cmd: &str) -> Result<(Vec<Recording>, Vec<String>)> {
    let h = host(s, cmd)?;
    let folders = strings(p, "folders", cmd)?;
    let folders = if folders.is_empty() { h.default_folders() } else { folders };
    let names = strings(p, "names", cmd)?;
    let (since, until) = (date_p(p, "since", cmd)?, date_p(p, "until", cmd)?);
    let all = bool_p(p, "all").unwrap_or(false);
    let recs = h.scan(&folders);
    let in_range = |r: &Recording| {
        let d = chrono::NaiveDate::parse_from_str(r.shot.get(..10).unwrap_or(""), "%Y-%m-%d").ok();
        since.is_none_or(|s| d.is_some_and(|d| d >= s)) && until.is_none_or(|u| d.is_some_and(|d| d <= u))
    };
    if names.is_empty() {
        if !(all || since.is_some() || until.is_some()) {
            return Err(bad(cmd, "say which recordings: `names`, `since`/`until`, or `all`"));
        }
        return Ok((recs.into_iter().filter(in_range).collect(), Vec::new()));
    }
    let mut found: Vec<Recording> = Vec::new();
    let mut missing = Vec::new();
    for n in &names {
        let base = n.rsplit_once('.').map_or(n.as_str(), |(b, _)| b);
        let hits: Vec<&Recording> = recs.iter().filter(|r| r.stem.eq_ignore_ascii_case(base) || short_name(&r.stem).eq_ignore_ascii_case(base)).collect();
        if hits.is_empty() {
            missing.push(n.clone());
        }
        for r in hits {
            if in_range(r) && !found.iter().any(|f| f.stem == r.stem) {
                found.push(r.clone());
            }
        }
    }
    Ok((found, missing))
}

/// Studio's short name of a recording (`VID_093`).
pub fn short_name(stem: &str) -> String {
    let mut p = stem.split('_');
    match (p.next(), p.next_back()) {
        (Some(a), Some(n)) => format!("{a}_{n}"),
        _ => stem.to_string(),
    }
}

fn list(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "insta360.list";
    let h = host(s, cmd)?;
    let folders = strings(p, "folders", cmd)?;
    let folders = if folders.is_empty() { h.default_folders() } else { folders };
    let mut q = p.as_object().cloned().unwrap_or_default();
    if !["names", "since", "until"].iter().any(|k| q.get(*k).is_some_and(|v| !v.is_null())) {
        q.insert("all".into(), json!(true));
    }
    q.insert("folders".into(), json!(folders));
    let (recs, missing) = chosen(s, &Value::Object(q), cmd)?;
    Ok(json!({"folders": folders, "recordings": recs, "missing": missing}))
}

/// Imports `recs` (finished videos; one `file.import` each, reusing what the project holds), the
/// proxies attached and the lenses aligned when asked. Every edit folds into one undo step.
fn import_recordings(s: &mut Session, recs: &[Recording], p: &Value, cmd: &str) -> Result<Vec<Value>> {
    let proxies = bool_p(p, "proxies").unwrap_or(true);
    let align = bool_p(p, "align").unwrap_or(true);
    let bin = p.get("bin").filter(|v| !v.is_null()).cloned();
    let n0 = s.history.undo.len();
    let mut out = Vec::new();
    for r in recs {
        if !r.complete {
            out.push(json!({"stem": r.stem, "imported": false, "why": "still downloading (or not a finished recording)"}));
            continue;
        }
        if r.kind != "video" {
            out.push(json!({"stem": r.stem, "imported": false, "why": "360° photos are not imported yet"}));
            continue;
        }
        let mut params = serde_json::Map::new();
        params.insert("paths".into(), json!([r.path]));
        params.insert("reuse".into(), json!(true));
        if let Some(b) = bin.clone() {
            params.insert("bin".into(), b);
        }
        let params = Value::Object(params);
        let res = match s.execute("file.import", params) {
            Ok(r) => r,
            Err(e) => {
                out.push(json!({"stem": r.stem, "imported": false, "why": e.to_string()}));
                continue;
            }
        };
        let Some(item) = res["items"].as_array().and_then(|a| a.first()).and_then(Value::as_u64) else {
            let why = res["errors"].as_array().and_then(|e| e.first()).and_then(Value::as_str).unwrap_or("not imported").to_string();
            out.push(json!({"stem": r.stem, "imported": false, "why": why}));
            continue;
        };
        let mut row = json!({"stem": r.stem, "imported": true, "item": item, "shot": r.shot, "camera": r.camera});
        if proxies && let Some(px) = &r.proxy {
            let attached = s.execute("media.attachProxies", json!({"item": item, "path": px}));
            row["proxy"] = match attached {
                Ok(_) => json!(px),
                Err(e) => json!({"error": e.to_string()}),
            };
        }
        if align
            && crate::vr360::is_360(s, filmcraft_project::ItemId(item))
            && let Ok(a) = s.execute("media.align360", json!({"items": [item]}))
        {
            row["aligned"] = a["items"][0]["aligned"].clone();
        }
        out.push(row);
    }
    crate::clip_ops::collapse_history(s, n0, "Import Insta360 Recordings");
    if out.iter().all(|r| r["imported"] != json!(true)) && !out.is_empty() {
        return Err(bad(cmd, format!("nothing was imported: {}", out.iter().filter_map(|r| r["why"].as_str()).collect::<Vec<_>>().join("; "))));
    }
    Ok(out)
}

fn import(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "insta360.import";
    let (recs, missing) = chosen(s, p, cmd)?;
    if recs.is_empty() {
        return Err(bad(
            cmd,
            if missing.is_empty() { "no Insta360 recordings there".to_string() } else { format!("no recording named {}", missing.join(", ")) },
        ));
    }
    let rows = import_recordings(s, &recs, p, cmd)?;
    Ok(json!({"recordings": rows, "missing": missing}))
}

/// The outcome a download job leaves for `insta360.download` with `wait`.
type Outcomes = std::sync::Arc<std::sync::Mutex<Option<std::result::Result<Vec<Downloaded>, String>>>>;

fn download(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "insta360.download";
    let h = host(s, cmd)?;
    let names = strings(p, "names", cmd)?;
    if names.is_empty() {
        return Err(bad(cmd, "need `names`: the recordings to download (VID_20260913_114003_00_093, or Studio's short VID_093)"));
    }
    h.studio_ready().map_err(|e| bad(cmd, e))?;
    let wait = bool_p(p, "wait").unwrap_or(false);
    let label = format!("Downloading {} from the Insta360 cloud", if names.len() == 1 { names[0].clone() } else { format!("{} recordings", names.len()) });
    let job = crate::narration::new_job(s, label);
    job.progress.total.store(names.len() as u64, std::sync::atomic::Ordering::Relaxed);
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let outcomes: Outcomes = Default::default();
    let mine = outcomes.clone();
    let wanted = names.clone();
    let id = crate::narration::spawn(s, job, wait, "filmcraft-insta360-download", move || {
        let t0 = web_time::Instant::now();
        let r = h.download(&wanted, true, &prog);
        let report = match &r {
            Ok(d) => {
                let done = d.iter().filter(|o| o.path.is_some()).count();
                *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = format!("{done} of {} downloaded", d.len());
                prog.done.store(done as u64, std::sync::atomic::Ordering::Relaxed);
                Ok(filmcraft_export::Report {
                    path: String::new(),
                    frames: 0,
                    seconds: t0.elapsed().as_secs_f64(),
                    bytes: 0,
                    render_fps: 0.0,
                    extra_files: Vec::new(),
                })
            }
            Err(e) => {
                *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = e.clone();
                *prog.error.lock().unwrap_or_else(PoisonError::into_inner) = Some(e.clone());
                Err(e.clone())
            }
        };
        *mine.lock().unwrap_or_else(PoisonError::into_inner) = Some(r);
        prog.finished.store(true, std::sync::atomic::Ordering::Relaxed);
        *res.lock().unwrap_or_else(PoisonError::into_inner) = Some(report);
    })?;
    let mut out = json!({"job": id, "names": names});
    if !wait {
        return Ok(out);
    }
    let done = outcomes.lock().unwrap_or_else(PoisonError::into_inner).clone();
    match done {
        Some(Ok(d)) => {
            out["downloads"] = json!(d);
            if bool_p(p, "import").unwrap_or(true) {
                let got: Vec<String> = d.iter().filter(|o| o.path.is_some()).map(|o| o.name.clone()).collect();
                if !got.is_empty() {
                    let mut q = serde_json::Map::new();
                    q.insert("names".into(), json!(got));
                    for k in ["proxies", "align", "bin"] {
                        if let Some(v) = p.get(k) {
                            q.insert(k.into(), v.clone());
                        }
                    }
                    let (recs, _) = chosen(s, &Value::Object(q.clone()), cmd)?;
                    out["recordings"] = json!(import_recordings(s, &recs, &Value::Object(q), cmd)?);
                }
            }
            Ok(out)
        }
        Some(Err(e)) => Err(EngineError::Other(format!("{cmd}: {e}"))),
        None => Ok(out),
    }
}

/// The camera's own low-resolution copy of the recording at `path` — `LRV_<date>_<time>_01_<n>.lrv`
/// beside `VID_<date>_<time>_00_<n>.insv` (or `.mp4`) — when the host can list its folder.
pub(crate) fn camera_proxy(s: &Session, path: &str) -> Option<String> {
    let p = std::path::Path::new(path);
    let ext = p.extension()?.to_str()?.to_ascii_lowercase();
    if !matches!(ext.as_str(), "insv" | "mp4") {
        return None;
    }
    let stem = p.file_stem()?.to_str()?;
    let ["VID", date, time, _, number] = stem.split('_').collect::<Vec<_>>()[..] else { return None };
    let dir = p.parent()?;
    let names = s.services.list_dir(&dir.to_string_lossy())?.ok()?;
    let fits = |name: &str| {
        let q = std::path::Path::new(name);
        let ext_ok = q.extension().and_then(|e| e.to_str()).is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "lrv" | "insv" | "mp4"));
        let parts: Vec<&str> = q.file_stem().and_then(|x| x.to_str()).unwrap_or_default().split('_').collect();
        ext_ok && matches!(parts[..], ["LRV", d, t, _, n] if d == date && t == time && n == number)
    };
    let name = names.into_iter().filter(|n| fits(n)).min()?;
    Some(dir.join(name).to_string_lossy().into_owned())
}

/// Attaches the camera's `.lrv` ([`camera_proxy`]) to each freshly imported recording among
/// `items` without a proxy — what the 360° analyses read ([`crate::vr360::analysis_view`]) — when
/// it fits (its length and rate checked); the ones attached. Part of the import's undo step.
pub(crate) fn attach_camera_proxies(s: &mut Session, items: &[filmcraft_project::ItemId]) -> Vec<Value> {
    let mut pairs = Vec::new();
    for &id in items {
        let Some(m) = s.project.item(id).and_then(|it| it.as_media()) else { continue };
        let filmcraft_project::MediaRef::File { path } = &m.media else { continue };
        if m.proxy.is_some() {
            continue;
        }
        if let Some(px) = camera_proxy(s, path)
            && crate::proxies::check_proxy(s, id, &px).is_ok()
        {
            pairs.push((id, px));
        }
    }
    let n0 = s.history.undo.len();
    if pairs.is_empty() || crate::proxies::attach(s, &pairs, true).is_err() {
        return Vec::new();
    }
    // (undoing the import, the step before, takes the proxies away with the items)
    if n0 > 0 {
        s.history.undo.truncate(n0);
    }
    pairs.into_iter().map(|(i, p)| json!({"item": i.0, "path": p})).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host over fixed recordings that downloads by "finding" them.
    struct Fake {
        recs: Vec<Recording>,
    }
    impl Insta360Host for Fake {
        fn default_folders(&self) -> Vec<String> {
            vec!["/fake".into()]
        }
        fn scan(&self, _folders: &[String]) -> Vec<Recording> {
            self.recs.clone()
        }
        fn studio_ready(&self) -> std::result::Result<(), String> {
            Ok(())
        }
        fn download(&self, names: &[String], _wait: bool, _progress: &filmcraft_export::Progress) -> std::result::Result<Vec<Downloaded>, String> {
            Ok(names.iter().map(|n| Downloaded { name: n.clone(), requested: true, path: Some(format!("/fake/{n}.insv")), why: None }).collect())
        }
    }

    fn rec(stem: &str, shot: &str, complete: bool) -> Recording {
        Recording {
            stem: stem.into(),
            kind: "video".into(),
            shot: shot.into(),
            path: format!("/nowhere/{stem}.insv"),
            bytes: 1,
            proxy: None,
            second_lens: None,
            complete,
            camera: Some("Insta360 X5".into()),
        }
    }

    fn session() -> Session {
        let host = std::sync::Arc::new(Fake {
            recs: vec![rec("VID_20260913_114003_00_093", "2026-09-13T11:40:03", true), rec("VID_20260912_214043_00_087", "2026-09-12T21:40:43", false)],
        });
        Session { insta360: Some(host), ..Default::default() }
    }

    #[test]
    fn listing_and_choosing() {
        let mut s = session();
        let all = s.execute("insta360.list", json!({})).unwrap();
        assert_eq!(all["recordings"].as_array().unwrap().len(), 2);
        let one = s.execute("insta360.list", json!({"since": "2026-09-13"})).unwrap();
        assert_eq!(one["recordings"][0]["stem"], "VID_20260913_114003_00_093");
        let named = s.execute("insta360.list", json!({"names": ["VID_087", "VID_999"]})).unwrap();
        assert_eq!(named["recordings"][0]["stem"], "VID_20260912_214043_00_087");
        assert_eq!(named["missing"], json!(["VID_999"]));
        // the files are not there: nothing imports, and it says why
        let e = s.execute("insta360.import", json!({"names": ["VID_093"]})).unwrap_err().to_string();
        assert!(e.contains("nothing was imported"), "{e}");
        let e = s.execute("insta360.import", json!({"names": ["VID_087"]})).unwrap_err().to_string();
        assert!(e.contains("still downloading"), "{e}");
    }

    #[test]
    fn without_a_host_the_commands_are_disabled() {
        let mut s = Session::default();
        assert!(s.execute("insta360.list", json!({})).is_err());
        assert!(!s.is_enabled("insta360.import"));
    }

    #[test]
    fn hostile_parameters_are_refused() {
        let mut s = session();
        for bad in [
            json!({"names": "VID_093"}),
            json!({"names": [1, 2]}),
            json!({"since": "yesterday"}),
            json!({"until": "2026-13-40"}),
            json!({"folders": [null]}),
            json!({"names": vec!["x"; MAX_NAMES + 1]}),
            json!({}),
        ] {
            assert!(s.execute("insta360.import", bad.clone()).is_err(), "{bad}");
        }
        assert!(s.execute("insta360.download", json!({})).is_err());
        assert!(s.execute("insta360.download", json!({"names": []})).is_err());
        assert!(s.execute("insta360.list", json!([1, 2, 3])).is_ok_and(|v| v["recordings"].as_array().is_some_and(|a| a.len() == 2)));
    }

    #[test]
    fn a_waited_download_reports_each_recording() {
        let mut s = session();
        let r = s.execute("insta360.download", json!({"names": ["VID_093"], "wait": true, "import": false})).unwrap();
        assert_eq!(r["downloads"][0]["requested"], true);
        assert!(r["job"].as_u64().is_some());
    }
}
