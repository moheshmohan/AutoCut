//! The steps of a showcase round that used to be throwaway scripts, as subcommands:
//! `showcase-plan` (review notes applied to a plan, and the plan in another language), `sheet`
//! (contact sheets of a project's source clips) and `frames` (frames of any video file); and
//! `showcase`'s build step, once per language with `--languages en,ml`.

use filmcraft_autocut::showcase::{Entry, Options, Script};
use serde_json::{Value, json};

use crate::args::Args;
use crate::{Backend, absolute, diag, fail, print, usage};

pub(crate) fn read_json(what: &str, path: &str) -> Value {
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
/// [--places-from-locations locations.json]
/// [--out new.json] [--lines script-ml.json [--language ml] [--lines-out new-ml.json]]`.
pub async fn plan(a: &Args) {
    const USAGE: &str = "showcase-plan --base plan.json [--patch notes.json]... [--project p.fcproj | --clips items.json] [--places-from-locations locations.json] [--out new.json] [--lines script-ml.json [--language ml] [--lines-out new-ml.json]]";
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
    let mut patched = filmcraft_autocut::plan_patch::apply(&base, &patches, &clips).unwrap_or_else(|e| fail(format!("showcase-plan: {e}")));
    for n in &patched.notes {
        diag(format_args!("showcase-plan: {n}"));
    }
    // --places-from-locations: entries without a place get the one Google Photos shows for their
    // first shot (the file `photos locations` wrote)
    let filled = a.opt("--places-from-locations").map(|path| {
        let v = read_json("showcase-plan: --places-from-locations", path);
        let places = filmcraft_autocut::places::Places::from_json(&v).unwrap_or_else(|e| fail(format!("showcase-plan: {path}: {e}")));
        let filled = filmcraft_autocut::plan_patch::fill_places(&mut patched.plan, &places);
        diag(format_args!("showcase-plan: {} entr(y/ies) got their place from {path} ({} place(s) in it)", filled.len(), places.len()));
        filled
    });
    if !patched.unlined.is_empty() {
        diag(format_args!(
            "showcase-plan: {} entr{} without a line: {}",
            patched.unlined.len(),
            if patched.unlined.len() == 1 { "y" } else { "ies" },
            patched.unlined.join("; ")
        ));
    }
    let mut summary = patched.summary();
    if let Some(f) = filled {
        summary["placesFilled"] = json!(f);
    }
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

// ---------------------------------------------------------------------------------------------
// showcase: building, checking and shipping, once per language

/// The languages `--languages` names (`en,ml`): English and Malayalam, each once.
pub fn parse_languages(s: &str) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for l in s.split(',').map(|l| l.trim().to_ascii_lowercase()).filter(|l| !l.is_empty()) {
        let l = match l.as_str() {
            "en" | "en-us" | "english" => "en",
            "ml" | "ml-in" | "malayalam" => "ml",
            _ => return Err(format!("`{l}` is not a language a showcase speaks (en, ml)")),
        };
        if out.iter().any(|o| o == l) {
            return Err(format!("{l} is named twice"));
        }
        out.push(l.to_string());
    }
    if out.is_empty() {
        return Err("name the languages, e.g. en,ml".into());
    }
    Ok(out)
}

/// The language's name as a file name and a title carry it ("Malayalam").
fn language_name(l: &str) -> &'static str {
    match l {
        "ml" => "Malayalam",
        _ => "English",
    }
}

/// `out.mp4` → `out (Malayalam).mp4`, beside it.
pub fn language_export(path: &str, name: &str) -> String {
    let p = std::path::Path::new(path);
    let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let file = match p.extension() {
        Some(ext) => format!("{stem} ({name}).{}", ext.to_string_lossy()),
        None => format!("{stem} ({name})"),
    };
    p.with_file_name(file).to_string_lossy().into_owned()
}

/// One showcase to build: its options (language, export), script, and the title its uploads
/// carry.
pub struct Run {
    pub opts: Options,
    pub script: Script,
    pub title: String,
}

/// The builds of `--languages` from one (English) `plan` and its `script`: the first language
/// exports to `--export`, each next one beside it as `<out> (<Language>).<ext>`; Malayalam reads
/// `lines_ml` over the plan ([`filmcraft_autocut::plan_patch::with_lines`]). Without languages,
/// the one build as asked.
pub fn language_runs(opts: &Options, plan: &Value, script: &Script, languages: Option<&[String]>, lines_ml: Option<&Value>) -> Result<Vec<Run>, String> {
    let Some(languages) = languages else {
        return Ok(vec![Run { opts: opts.clone(), script: script.clone(), title: opts.title.clone() }]);
    };
    let mut runs = Vec::with_capacity(languages.len());
    for (i, l) in languages.iter().enumerate() {
        let mut o = opts.clone();
        let name = language_name(l);
        let script = match l.as_str() {
            "ml" => {
                let lines = lines_ml.ok_or("--languages with ml needs --lines-ml script-ml.json")?;
                let (p, notes) = filmcraft_autocut::plan_patch::with_lines(plan, lines, "ml").map_err(|e| e.to_string())?;
                for n in notes {
                    diag(format_args!("showcase: {n}"));
                }
                o.language = Some("ml".into());
                serde_json::from_value::<Script>(p["script"].clone()).map_err(|e| format!("the Malayalam script: {e}"))?
            }
            _ => {
                o.language = None;
                script.clone()
            }
        };
        let title = if i == 0 { opts.title.clone() } else { format!("{} ({name})", opts.title) };
        if i > 0 {
            o.export = opts.export.as_deref().map(|p| language_export(p, name));
        }
        runs.push(Run { opts: o, script, title });
    }
    Ok(runs)
}

/// What happens after a showcase is built: QA (with the grounding model, when reachable) and
/// the uploads asked for.
pub struct Ship<'a> {
    pub qa_on: bool,
    pub qa_model: Option<&'a filmcraft_autocut::ollama::Ollama>,
    pub upload: bool,
    pub upload_youtube: bool,
    pub album: Option<String>,
}

/// Build `run` in `b`, check it, write its chapters, QA report and sheets beside the export, and
/// upload it: the build's JSON, and what went wrong uploading (the video is built either way).
pub async fn build_and_ship(a: &Args, b: &mut Backend, run: &Run, entries: &[Entry], ship: &Ship<'_>) -> Result<(Value, Option<String>), String> {
    let opts = &run.opts;
    let lang = opts.language.as_deref().map(|l| format!(" ({})", language_name(l))).unwrap_or_default();
    let mut progress = |l: &str| diag(format_args!("showcase{lang}: {l}"));
    let (result, qa) = match b {
        Backend::Local(s) => {
            let session = std::mem::take(s);
            let ex = filmcraft_autocut::SessionExecutor::from(*session);
            let r = filmcraft_autocut::showcase::build(&ex, opts, entries, &run.script, &mut progress).await;
            let qa = match &r {
                Ok(built) if ship.qa_on => Some(crate::showcase_qa(&ex, opts, built, ship.qa_model, &mut progress).await),
                _ => None,
            };
            if let Ok(inner) = std::sync::Arc::try_unwrap(ex.0) {
                **s = inner.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            (r, qa)
        }
        Backend::Bridge(_) => {
            let addr = a.opt("--bridge").unwrap_or_default();
            let client = filmcraft_automation::BridgeClient::new(addr).map_err(|e| e.to_string())?;
            let ex = filmcraft_automation::AutocutExecutor::bridge(client);
            let r = filmcraft_autocut::showcase::build(&ex, opts, entries, &run.script, &mut progress).await;
            let qa = match &r {
                Ok(built) if ship.qa_on => Some(crate::showcase_qa(&ex, opts, built, ship.qa_model, &mut progress).await),
                _ => None,
            };
            (r, qa)
        }
    };
    let built = result.map_err(|e| e.to_string())?;
    for n in &built.notes {
        progress(n);
    }
    // chapters for the YouTube description, next to the export
    let chapters = filmcraft_autocut::showcase::youtube_chapters(&built.chapters, built.seconds);
    if let Some(path) = &opts.export
        && !chapters.is_empty()
    {
        let txt = std::path::Path::new(path).with_extension("chapters.txt");
        match std::fs::write(&txt, format!("{chapters}\n")) {
            Ok(()) => progress(&format!("chapters in {}", txt.display())),
            Err(e) => progress(&format!("{}: {e}", txt.display())),
        }
    }
    for file in &built.subtitle_files {
        progress(&format!("subtitles in {file}"));
    }
    // what the strict check heard in a Malayalam narration, line by line, next to the export
    if let (Some(path), Some(check)) = (&opts.export, &built.narration_check) {
        let file = std::path::Path::new(path).with_extension("narration-check.json");
        let (clear, of) = check.clear();
        match std::fs::write(&file, serde_json::to_vec_pretty(check).unwrap_or_default()) {
            Ok(()) => progress(&format!("narration check in {} ({clear} of {of} Malayalam lines heard clearly)", file.display())),
            Err(e) => progress(&format!("{}: {e}", file.display())),
        }
    }
    if let Some(file) = &built.thumbnail {
        progress(&format!("thumbnail in {file}"));
    }
    let mut out = serde_json::to_value(&built).unwrap_or_default();
    out["youtubeChapters"] = json!(chapters);
    if let Some(l) = &opts.language {
        out["language"] = json!(l);
    }
    if let Some(mut report) = qa {
        progress(&format!("QA: {}", report.summary()));
        for f in report.findings.iter().filter(|f| f.severity >= filmcraft_autocut::qa::Severity::Warning).take(40) {
            progress(&format!("QA {:?} at {:.2}s ({}): {}", f.severity, f.seconds, f.check, f.message));
        }
        // the report, the contact sheet and the privacy sheet next to the export
        if let Some(path) = &opts.export {
            match filmcraft_autocut::qa::write_next_to(&mut report, std::path::Path::new(path)) {
                Ok((j, png)) => {
                    progress(&format!("QA report in {}", j.display()));
                    if let Some(p) = png {
                        progress(&format!("contact sheet in {}", p.display()));
                    }
                    if let Some(p) = &report.privacy_sheet {
                        progress(&format!("privacy sheet in {p}"));
                    }
                }
                Err(e) => progress(&format!("QA: {e}")),
            }
        }
        out["qa"] = serde_json::to_value(&report).unwrap_or_default();
    }
    // --upload: the finished export to Google Photos, titled like the video
    let mut upload_failed = None;
    if ship.upload
        && let Some(requested) = &opts.export
    {
        let file = built.export.as_ref().and_then(|r| crate::written_paths(r, requested).into_iter().next()).unwrap_or_else(|| requested.clone());
        let description = a.opt("--description").map(str::to_string).or_else(|| Some(run.title.clone()).filter(|t| !t.trim().is_empty()));
        match crate::upload_to_photos(absolute(&file), description, ship.album.clone()).await {
            Ok(v) => out["upload"] = v,
            Err(e) => {
                out["uploadError"] = json!(e);
                upload_failed = Some(e);
            }
        }
    }
    // --upload-youtube: the export to YouTube, titled like the video, with its chapters and
    // thumbnail
    if ship.upload_youtube
        && let Some(requested) = &opts.export
    {
        let file = built.export.as_ref().and_then(|r| crate::written_paths(r, requested).into_iter().next()).unwrap_or_else(|| requested.clone());
        let r = match crate::youtube::showcase_request(a, &file, &run.title, built.thumbnail.as_deref()) {
            Ok(req) => crate::youtube::upload(req).await,
            Err(e) => Err(e),
        };
        match r {
            Ok(v) => out["youtube"] = v,
            Err(e) => {
                out["youtubeError"] = json!(e);
                upload_failed = Some(upload_failed.map_or(e.clone(), |p| format!("{p}; YouTube: {e}")));
            }
        }
    }
    Ok((out, upload_failed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_and_their_exports() {
        assert_eq!(parse_languages("en, ML").unwrap(), ["en", "ml"]);
        assert_eq!(parse_languages("malayalam").unwrap(), ["ml"]);
        for bad in ["", "en,en", "fr", ",,"] {
            assert!(parse_languages(bad).is_err(), "{bad}");
        }
        assert_eq!(language_export("/m/Every Robot - v6.mp4", "Malayalam"), "/m/Every Robot - v6 (Malayalam).mp4");
        assert_eq!(language_export("out", "Malayalam"), "out (Malayalam)");
        // the runs: English as planned, then Malayalam from its lines, exported beside it
        let plan = json!({
            "entries": [{"number": 1, "label": "Arm", "kind": "arm", "description": "", "unix": 1, "shots": []}],
            "script": {"intro": "Hi", "outro": "Bye", "lines": ["The arm."], "pronounce": {"Arm": "Ahm"}}
        });
        let script: Script = serde_json::from_value(plan["script"].clone()).unwrap();
        let opts = Options { title: "Robots".into(), export: Some("/m/out.mp4".into()), language: Some("en".into()), ..Options::default() };
        let langs = parse_languages("en,ml").unwrap();
        let lines = json!({"intro": "ഹായ്", "outro": "നന്ദി", "lines": {"Arm": "കൈ"}});
        let runs = language_runs(&opts, &plan, &script, Some(&langs), Some(&lines)).unwrap();
        assert_eq!(runs.len(), 2);
        assert_eq!((runs[0].opts.export.as_deref(), runs[0].opts.language.as_deref(), runs[0].title.as_str()), (Some("/m/out.mp4"), None, "Robots"));
        assert_eq!((runs[1].opts.export.as_deref(), runs[1].opts.language.as_deref()), (Some("/m/out (Malayalam).mp4"), Some("ml")));
        assert_eq!((runs[1].script.lines[0].as_str(), runs[1].script.pronounce.len(), runs[1].title.as_str()), ("കൈ", 0, "Robots (Malayalam)"));
        // Malayalam without its lines, or a spoken entry without one, is refused
        assert!(language_runs(&opts, &plan, &script, Some(&langs), None).is_err());
        assert!(language_runs(&opts, &plan, &script, Some(&langs), Some(&json!({"intro": "a", "outro": "b", "lines": {}}))).is_err());
        // no languages: the one build as asked
        let one = language_runs(&opts, &plan, &script, None, None).unwrap();
        assert_eq!((one.len(), one[0].opts.language.as_deref()), (1, Some("en")));
    }

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
