//! The steps of a showcase round that used to be throwaway scripts, as subcommands:
//! `showcase-plan` (review notes applied to a plan, and the plan in another language), `sheet`
//! (contact sheets of a project's source clips) and `frames` (frames of any video file).

use serde_json::{Value, json};

use crate::args::Args;
use crate::{Backend, absolute, diag, fail, print, usage};

fn read_json(what: &str, path: &str) -> Value {
    let bytes = std::fs::read(path).unwrap_or_else(|e| fail(format!("{what}: {path}: {e}")));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| fail(format!("{what}: {path}: not JSON ({e})")))
}

fn write_json(what: &str, path: &str, v: &Value) {
    let text = serde_json::to_string_pretty(v).unwrap_or_default();
    std::fs::write(path, format!("{text}\n")).unwrap_or_else(|e| fail(format!("{what}: {path}: {e}")));
}

/// `plan.json` → `plan-ml.json`.
fn language_path(out: &str, language: &str) -> String {
    match out.strip_suffix(".json") {
        Some(stem) => format!("{stem}-{language}.json"),
        None => format!("{out}-{language}.json"),
    }
}

/// `showcase-plan --base plan.json [--patch notes.json]… [--project p.fcproj | --clips inspect.json]
/// [--out new.json] [--lines script-ml.json [--language ml] [--lines-out new-ml.json]]`.
pub async fn plan(a: &Args) {
    const USAGE: &str = "showcase-plan --base plan.json [--patch notes.json]... [--project p.fcproj | --clips items.json] [--out new.json] [--lines script-ml.json [--language ml] [--lines-out new-ml.json]]";
    let Some(base_path) = a.opt("--base") else { usage(USAGE) };
    let lines_path = a.opt("--lines");
    let language = a.opt("--language").unwrap_or("ml").trim().to_ascii_lowercase();
    let out = a.opt("--out");
    let lines_out = a.opt("--lines-out").map(str::to_string).or_else(|| lines_path.and(out).map(|o| language_path(o, &language)));
    if lines_path.is_some() && lines_out.is_none() {
        usage("showcase-plan: --lines writes another plan; give --out (it goes beside it) or --lines-out");
    }
    let base = read_json("showcase-plan: --base", base_path);
    let patches: Vec<Value> = a.all("--patch").into_iter().map(|p| read_json("showcase-plan: --patch", p)).collect();
    // the project's clips, for plain shots: their names and lengths
    let clips = match (a.opt("--clips"), a.opt("--project").is_some() || a.opt("--bridge").is_some()) {
        (Some(path), _) => filmcraft_autocut::plan_patch::clips_from_inspect(&read_json("showcase-plan: --clips", path)),
        (None, true) => {
            let mut b = Backend::open(a);
            let v = b.exec("project.inspect", json!({})).await.unwrap_or_else(|e| fail(format!("showcase-plan: the project: {e}")));
            filmcraft_autocut::plan_patch::clips_from_inspect(&v)
        }
        (None, false) => Default::default(),
    };
    let patched = filmcraft_autocut::plan_patch::apply(&base, &patches, &clips).unwrap_or_else(|e| fail(format!("showcase-plan: {e}")));
    for n in &patched.notes {
        diag(format_args!("showcase-plan: {n}"));
    }
    if !patched.unlined.is_empty() {
        diag(format_args!(
            "showcase-plan: {} entr{} without a line: {}",
            patched.unlined.len(),
            if patched.unlined.len() == 1 { "y" } else { "ies" },
            patched.unlined.join("; ")
        ));
    }
    let mut summary = patched.summary();
    if let (Some(lp), Some(lo)) = (lines_path, lines_out.as_deref()) {
        let lines = read_json("showcase-plan: --lines", lp);
        let (other, notes) =
            filmcraft_autocut::plan_patch::with_lines(&patched.plan, &lines, &language).unwrap_or_else(|e| fail(format!("showcase-plan: {e}")));
        for n in &notes {
            diag(format_args!("showcase-plan: {n}"));
        }
        write_json("showcase-plan", lo, &other);
        summary["language"] = json!({"language": language, "out": lo, "notes": notes});
    }
    match out {
        Some(path) => {
            write_json("showcase-plan", path, &patched.plan);
            summary["out"] = json!(path);
            print(a, &summary);
        }
        None => print(a, &patched.plan),
    }
}

/// `--items 2407,2408:0-30,14` as `media.contactSheet` items.
fn items(spec: &str) -> Result<Vec<Value>, String> {
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            let (item, range) = s.split_once(':').map_or((s, None), |(i, r)| (i, Some(r)));
            let item: u64 = item.trim().parse().map_err(|_| format!("`{s}`: an item id, then :from-to in seconds if you like"))?;
            let mut v = json!({"item": item});
            if let Some(r) = range {
                let (from, to) = r.split_once('-').ok_or_else(|| format!("`{s}`: the range is from-to, e.g. 0-30"))?;
                let num = |x: &str| x.trim().parse::<f64>().ok().filter(|x| x.is_finite() && *x >= 0.0);
                match (num(from), num(to)) {
                    (Some(f), Some(t)) if t > f => {
                        v["from"] = json!(f);
                        v["to"] = json!(t);
                    }
                    _ => return Err(format!("`{s}`: the range is from-to in seconds, from before to")),
                }
            }
            Ok(v)
        })
        .collect()
}

/// `--at 3.5,7`: seconds.
fn times(spec: &str) -> Result<Vec<f64>, String> {
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|t| t.parse::<f64>().ok().filter(|x| x.is_finite() && *x >= 0.0).ok_or_else(|| format!("--at: `{t}` is not a time in seconds")))
        .collect()
}

/// The shared sheet options (`--n`, `--side`, `--scale`, `--columns`, `--quality`) as parameters.
fn sheet_options(a: &Args, p: &mut Value, cmd: &str) {
    for (flag, key) in [("--n", "n"), ("--side", "side"), ("--columns", "columns"), ("--quality", "quality")] {
        if let Some(v) = a.opt(flag) {
            p[key] = json!(v.parse::<u64>().unwrap_or_else(|_| usage(format!("{cmd}: {flag} wants a whole number, not `{v}`"))));
        }
    }
    if let Some(v) = a.opt("--scale") {
        p["scale"] =
            json!(v.parse::<f64>().ok().filter(|x| x.is_finite()).unwrap_or_else(|| usage(format!("{cmd}: --scale wants a share like 0.5, not `{v}`"))));
    }
}

/// What a sheet run prints: where it went and how it came out.
fn sheet_summary(v: &Value) -> Value {
    json!({"path": v["path"], "width": v["width"], "height": v["height"], "frames": v["tiles"].as_array().map_or(0, Vec::len), "missing": v["missing"]})
}

/// `sheet --items 2407,2408:0-30,14 [--n 8] [--at 3.5,7] [--side 320 | --scale 0.25] [--columns N]
/// --out sheet.jpg` over the --project (or the running app with --bridge).
pub async fn sheet(a: &Args) {
    const USAGE: &str = "sheet --project p.fcproj --items 2407,2408:0-30,14 [--n 8] [--at 3.5,7] [--side 320 | --scale 0.25] [--columns N] --out sheet.jpg";
    let (Some(spec), Some(out)) = (a.opt("--items"), a.opt("--out")) else { usage(USAGE) };
    let mut list = items(spec).unwrap_or_else(|e| usage(format!("sheet: --items: {e}")));
    if list.is_empty() {
        usage(USAGE);
    }
    if let Some(at) = a.opt("--at") {
        let at = times(at).unwrap_or_else(|e| usage(format!("sheet: {e}")));
        for it in &mut list {
            it["at"] = json!(at);
        }
    }
    let mut p = json!({"items": list, "path": absolute(out)});
    sheet_options(a, &mut p, "sheet");
    let mut b = Backend::open(a);
    match b.exec("media.contactSheet", p).await {
        Ok(v) => print(a, &sheet_summary(&v)),
        Err(e) => fail(format!("sheet: {e}")),
    }
}

/// `frames <video> --at 601,603 [--n 6] [--scale 0.5] [--columns 4] --out sheet.jpg`: the video is
/// imported into a fresh headless session (no project is touched) and its frames laid out.
pub async fn frames(a: &Args) {
    const USAGE: &str = "frames <video> (--at 601,603,605 | --n 6) [--scale 0.5] [--columns 4] --out sheet.jpg";
    let (Some(video), Some(out)) = (a.pos(1), a.opt("--out")) else { usage(USAGE) };
    if a.opt("--bridge").is_some() {
        usage("frames reads the file in its own headless session; leave out --bridge");
    }
    let at = a.opt("--at").map(|s| times(s).unwrap_or_else(|e| usage(format!("frames: {e}"))));
    if at.as_ref().is_some_and(Vec::is_empty) || (at.is_none() && a.opt("--n").is_none()) {
        usage(USAGE);
    }
    if !std::path::Path::new(video).is_file() {
        fail(format!("frames: {video}: no such file"));
    }
    let mut s = filmcraft_engine::Session::default();
    let imported = s.execute("file.import", json!({"paths": [absolute(video)]})).unwrap_or_else(|e| fail(format!("frames: {video}: {e}")));
    let Some(item) = imported["items"].as_array().and_then(|i| i.first()).and_then(Value::as_u64) else {
        let why = imported["errors"].as_array().and_then(|e| e.first()).map(Value::to_string).unwrap_or_default();
        fail(format!("frames: {video}: could not be imported {why}"))
    };
    let mut entry = json!({"item": item});
    if let Some(at) = at {
        entry["at"] = json!(at);
    }
    let mut p = json!({"items": [entry], "path": absolute(out), "labels": "time", "scale": 0.5, "columns": 4});
    sheet_options(a, &mut p, "frames");
    match s.execute("media.contactSheet", p) {
        Ok(v) => print(a, &sheet_summary(&v)),
        Err(e) => fail(format!("frames: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn item_lists_and_times_parse_or_say_why() {
        assert_eq!(items("2407, 2408:0-30,14").unwrap(), vec![json!({"item": 2407}), json!({"item": 2408, "from": 0.0, "to": 30.0}), json!({"item": 14})]);
        for bad in ["x", "1:5", "1:9-2", "1:-3-4", "1:a-b"] {
            assert!(items(bad).is_err(), "{bad}");
        }
        assert_eq!(times("601, 603.5").unwrap(), vec![601.0, 603.5]);
        assert!(times("1,-2").is_err() && times("NaN").is_err());
        assert_eq!(language_path("/a/plan-v5.json", "ml"), "/a/plan-v5-ml.json");
        assert_eq!(language_path("plan", "ml"), "plan-ml.json");
    }
}
