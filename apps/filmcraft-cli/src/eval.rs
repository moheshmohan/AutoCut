//! `eval`: the taste-memory eval harness (PX10.4; the idea of an eval from hallmark, MIT, written
//! fresh). Fixed briefs (`eval/briefs/*.json`) run headless on footage made here in code (the
//! procedural demo scenes: never the user's media, nothing committed), each twice — with the
//! guidance (`guidance: true`: open choices dealt against a fresh taste history, every video's
//! fingerprint recorded) and without it (`guidance: false`: each pipeline's own defaults) — and
//! scored:
//!
//! - **motion**: `sequence.checkMotion`'s findings on the built sequence (PX1);
//! - **preflight**: `delivery.preflight`'s checks, by tier, and whether it would block;
//! - **variety**: `autocut.fingerprint` of every video, how many of the six axes repeat between
//!   consecutive briefs, how many pairs share four or more (the history's warning), and how many
//!   values each axis takes across the briefs;
//! - **judge** (optional): the local vision model (Ollama on this machine, `AUTOCUT_VLM_MODEL`)
//!   scores each export's frame sheet (`--judge-model` names another local model); skipped, and
//!   said so, when it is not reachable — never a cloud call.
//!
//! A brief is `{name, pipeline: "direct"|"quick", seconds, options: {…the job's parameters},
//! footage: {scenes: [aurora|oceanSunset|cityNight|dunes|forest|plasma], seconds?, width?,
//! height?}}`. The report — `report.json` and `summary.md` in `target/eval/<when>/`, beside the
//! exports, sheets and the runs' data directories — compares the two runs and the previous
//! report there. Not part of `cargo xtask ci` (it builds and exports every brief twice): run it
//! when a pipeline's choices change (docs/testing.md).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use filmcraft_autocut::Executor as _;
use serde_json::{Value, json};

use crate::args::Args;
use crate::{diag, fail, print, usage};

pub const USAGE: &str = "\
eval [--briefs eval/briefs] [--out target/eval] [--modes guided,plain] [--only NAME]
     [--no-export] [--no-judge] [--judge-model NAME]";

/// Most briefs one run takes, the largest brief file, most clips a brief makes.
const MAX_BRIEFS: usize = 12;
const MAX_BRIEF_BYTES: u64 = 64 * 1024;
const MAX_CLIPS: usize = 8;
/// The footage's bounds: seconds a clip, pixels a side.
const CLIP_SECONDS: (f64, f64) = (2.0, 20.0);
const SIDE: (u32, u32) = (64, 640);
/// Largest frame sheet sent to the judge, bytes.
const MAX_SHEET_BYTES: u64 = 8 * 1024 * 1024;
/// Axes shared with the brief before from which a pair is too alike (the history's warning).
const ALIKE: usize = 4;

/// One fixed brief.
#[derive(Clone, Debug)]
struct Brief {
    name: String,
    pipeline: Pipeline,
    seconds: f64,
    options: Value,
    scenes: Vec<filmcraft_media::DemoScene>,
    clip_seconds: f64,
    size: (u32, u32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Pipeline {
    Direct,
    Quick,
}

impl Pipeline {
    fn name(self) -> &'static str {
        match self {
            Pipeline::Direct => "direct",
            Pipeline::Quick => "quick",
        }
    }
}

fn scene(name: &str) -> Option<filmcraft_media::DemoScene> {
    use filmcraft_media::DemoScene as D;
    Some(match name.trim().to_ascii_lowercase().as_str() {
        "aurora" => D::Aurora,
        "oceansunset" | "ocean" => D::OceanSunset,
        "citynight" | "city" => D::CityNight,
        "dunes" => D::Dunes,
        "forest" => D::Forest,
        "plasma" => D::Plasma,
        _ => return None,
    })
}

/// A brief from its file's JSON; what is wrong with it otherwise.
fn brief(v: &Value, file: &str) -> Result<Brief, String> {
    let name =
        v["name"].as_str().map(str::trim).filter(|n| !n.is_empty() && n.len() <= 64 && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
    let name = name.ok_or_else(|| format!("{file}: `name` is a short id (letters, digits, - or _)"))?.to_string();
    let pipeline = match v["pipeline"].as_str() {
        Some("direct") => Pipeline::Direct,
        Some("quick") => Pipeline::Quick,
        _ => return Err(format!("{file}: `pipeline` is direct or quick")),
    };
    let seconds = v["seconds"].as_f64().filter(|s| s.is_finite() && (5.0..=60.0).contains(s)).ok_or_else(|| format!("{file}: `seconds` is 5 to 60"))?;
    let options = if v["options"].is_null() { json!({}) } else { v["options"].clone() };
    if !options.is_object() {
        return Err(format!("{file}: `options` is an object of the job's parameters"));
    }
    let f = &v["footage"];
    let scenes: Vec<filmcraft_media::DemoScene> = f["scenes"]
        .as_array()
        .filter(|a| !a.is_empty() && a.len() <= MAX_CLIPS)
        .ok_or_else(|| format!("{file}: `footage.scenes` lists 1 to {MAX_CLIPS} demo scenes"))?
        .iter()
        .map(|s| s.as_str().and_then(scene).ok_or_else(|| format!("{file}: `footage.scenes`: aurora, oceanSunset, cityNight, dunes, forest or plasma")))
        .collect::<Result<_, _>>()?;
    let clip_seconds = f["seconds"].as_f64().unwrap_or(6.0);
    if !(clip_seconds.is_finite() && (CLIP_SECONDS.0..=CLIP_SECONDS.1).contains(&clip_seconds)) {
        return Err(format!("{file}: `footage.seconds` is {} to {}", CLIP_SECONDS.0, CLIP_SECONDS.1));
    }
    let side = |k: &str, d: u64| f[k].as_u64().unwrap_or(d);
    let (w, h) = (side("width", 192), side("height", 108));
    let ok = |x: u64| (u64::from(SIDE.0)..=u64::from(SIDE.1)).contains(&x) && x.is_multiple_of(2);
    if !ok(w) || !ok(h) {
        return Err(format!("{file}: `footage.width` and `height` are even, {} to {}", SIDE.0, SIDE.1));
    }
    let size = (u32::try_from(w).unwrap_or(192), u32::try_from(h).unwrap_or(108));
    Ok(Brief { name, pipeline, seconds, options, scenes, clip_seconds, size })
}

/// The briefs in `dir` (`*.json`, by name).
fn briefs(dir: &Path) -> Result<Vec<Brief>, String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json") && p.is_file())
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("{}: no briefs (*.json)", dir.display()));
    }
    if files.len() > MAX_BRIEFS {
        return Err(format!("{}: {} briefs, at most {MAX_BRIEFS}", dir.display(), files.len()));
    }
    let mut out = Vec::new();
    for f in files {
        let name = f.display().to_string();
        let size = std::fs::metadata(&f).map(|m| m.len()).map_err(|e| format!("{name}: {e}"))?;
        if size > MAX_BRIEF_BYTES {
            return Err(format!("{name}: over {MAX_BRIEF_BYTES} bytes"));
        }
        let text = std::fs::read_to_string(&f).map_err(|e| format!("{name}: {e}"))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{name}: not JSON ({e})"))?;
        out.push(brief(&v, &name)?);
    }
    Ok(out)
}

/// The local vision model that scores frame sheets.
struct Judge {
    model: filmcraft_autocut::ollama::Ollama,
    name: String,
}

const JUDGE_SYSTEM: &str = "You judge short edited videos from a sheet of frames taken across each one. Answer only JSON: {\"score\": 1-5, \"variety\": 1-5, \"craft\": 1-5, \"notes\": \"one sentence\"}. variety: do the shots, framings and graphics change across the sheet? craft: are titles legible, the colour pleasant and consistent, nothing broken (black frames, stretched or empty pictures)? score: how good the edit looks overall. The footage is synthetic test footage: judge the edit, not the scenery.";

const JUDGE_TEXT: &str = "You judge short edited videos from a sheet of frames taken across each one. The footage is synthetic test footage: judge the edit, not the scenery. Answer in exactly four lines: \"score: N\", \"variety: N\", \"craft: N\" (each N from 1 to 5: overall quality; whether shots, framings and graphics change across the sheet; whether titles are legible and nothing looks broken) and \"notes: one sentence\".";

/// The scores in a plain answer (`score: 4`, `variety: 3`, …) and its notes.
fn scores_in(text: &str) -> Value {
    let mut v = json!({});
    for line in text.lines().take(64) {
        let Some((k, rest)) = line.split_once(':') else { continue };
        let key = k.trim().trim_matches(|c: char| !c.is_ascii_alphabetic()).to_ascii_lowercase();
        if key == "notes" {
            v["notes"] = json!(rest.trim().chars().take(300).collect::<String>());
            continue;
        }
        if matches!(key.as_str(), "score" | "variety" | "craft")
            && let Some(d) = rest.chars().find(char::is_ascii_digit).and_then(|c| c.to_digit(10))
        {
            v[key] = json!(f64::from(d));
        }
    }
    v
}

impl Judge {
    /// The model (`name`, else `AUTOCUT_VLM_MODEL`'s), or why there is none (a host that is not
    /// this machine is never asked).
    fn find(name: Option<&str>) -> Result<Judge, String> {
        let mut cfg = filmcraft_autocut::config::Config::from_env();
        if let Some(n) = name.map(str::trim).filter(|n| !n.is_empty()) {
            cfg.vlm_model = n.chars().take(128).collect();
        }
        let host = cfg.ollama_host.trim().trim_end_matches('/').to_ascii_lowercase();
        let local = ["http://127.0.0.1", "http://localhost", "http://[::1]"].iter().any(|h| host == *h || host.starts_with(&format!("{h}:")));
        if !local {
            return Err("the vision model's host is not a loopback address: no judge (never a cloud call)".into());
        }
        let model = filmcraft_autocut::ollama::Ollama::new(&cfg.ollama_host, &cfg.vlm_model, Duration::from_secs(240));
        match model.available() {
            Ok(true) => Ok(Judge { model, name: cfg.vlm_model }),
            Ok(false) => Err(format!("Ollama answers but has no {} model: no judge", cfg.vlm_model)),
            Err(e) => Err(format!("no judge: {e}")),
        }
    }

    fn score(&self, sheet: &Path) -> Value {
        let size = std::fs::metadata(sheet).map(|m| m.len()).unwrap_or(0);
        if size == 0 || size > MAX_SHEET_BYTES {
            return json!({"skipped": "no frame sheet"});
        }
        let Ok(bytes) = std::fs::read(sheet) else { return json!({"skipped": "the frame sheet could not be read"}) };
        // (small vision models break JSON mode: a plain answer is read for its numbers instead)
        let v = match self.model.chat_json(JUDGE_SYSTEM, "Judge this edit's frame sheet.", std::slice::from_ref(&bytes)) {
            Ok(v) if v["score"].as_f64().is_some() => v,
            _ => match self.model.chat_text(JUDGE_TEXT, "Judge this edit's frame sheet.", &[bytes]) {
                Ok(t) => scores_in(&t),
                Err(e) => return json!({"skipped": format!("the judge did not answer: {e}")}),
            },
        };
        let n = |k: &str| v[k].as_f64().filter(|x| x.is_finite() && (1.0..=5.0).contains(x));
        if n("score").is_none() {
            return json!({"skipped": "the judge gave no score"});
        }
        let notes: String = v["notes"].as_str().unwrap_or_default().chars().filter(|c| !c.is_control()).take(300).collect();
        json!({"model": self.name, "score": n("score"), "variety": n("variety"), "craft": n("craft"), "notes": notes})
    }
}

/// The findings of a check by code (`findings: [{code}]`).
fn by_code(v: &Value) -> Value {
    let mut m = serde_json::Map::new();
    for f in v["findings"].as_array().map(Vec::as_slice).unwrap_or_default() {
        let code = f["code"].as_str().unwrap_or("other").to_string();
        let n = m.get(&code).and_then(Value::as_u64).unwrap_or(0);
        m.insert(code, json!(n.saturating_add(1)));
    }
    Value::Object(m)
}

/// What `delivery.preflight` said, in short: whether it would block, and the open findings by
/// tier.
fn preflight_summary(v: &Value) -> Value {
    json!({"passed": v["passed"], "blocking": v["blocking"], "summary": v["summary"], "checks": v["checks"].as_array().map_or(0, Vec::len)})
}

/// Run brief `b` once (`guided` or not) with its own session over `data` (the taste history's
/// directory): the outcome and its scores.
async fn run_brief(b: &Brief, guided: bool, data: &Path, out: &Path, export: bool, judge: Option<&Judge>) -> Value {
    let started = Instant::now();
    let mode = if guided { "guided" } else { "plain" };
    let mut s = filmcraft_engine::Session::default();
    s.prefs_path = Some(data.join("preferences.json"));
    let pool = s.media.clone();
    let mut items = Vec::new();
    for (k, sc) in b.scenes.iter().enumerate() {
        let src = filmcraft_media::generators::GeneratorSource::new(
            filmcraft_media::generators::Generator::Demo(*sc),
            b.size.0,
            b.size.1,
            filmcraft_time::FrameRate::FPS_30,
            filmcraft_time::Tick::from_seconds_f64(b.clip_seconds),
        );
        let name = format!("{}-{k}.mov", sc.file_name().split('.').next().unwrap_or("clip"));
        let id = filmcraft_engine::demo::add_generator(std::sync::Arc::make_mut(&mut s.project), &pool, src, &name, filmcraft_project::Label::Iris, None);
        items.push(id.0);
    }
    let ex = filmcraft_autocut::SessionExecutor::from(s);
    let mut p = b.options.clone();
    p["items"] = json!(items);
    p["seconds"] = json!(b.seconds);
    p["guidance"] = json!(guided);
    p["qa"] = json!(false);
    let file = out.join(format!("{}.mp4", b.name));
    if export {
        p["export"] = json!(file.display().to_string());
    }
    let mut lines = |_: &str| {};
    let made = match b.pipeline {
        Pipeline::Direct => filmcraft_autocut::director::make(&ex, &p, None, &mut lines).await,
        Pipeline::Quick => filmcraft_autocut::quick::make(&ex, &p, None, &mut lines).await,
    };
    let made = match made {
        Ok(m) => m,
        Err(e) => {
            return json!({"brief": b.name, "mode": mode, "pipeline": b.pipeline.name(), "error": e.to_string(), "seconds": started.elapsed().as_secs_f64()});
        }
    };
    let seq = made["sequence"].as_u64();
    let on = |mut p: Value| {
        if let Some(q) = seq {
            p["sequence"] = json!(q);
        }
        p
    };
    // the fingerprint: recorded by a guided run, read (not recorded) for a plain one
    let fingerprint = if guided && made["fingerprint"]["axes"].is_object() {
        made["fingerprint"].clone()
    } else {
        ex.execute("autocut.fingerprint", on(json!({}))).await.unwrap_or_else(|e| json!({"error": e}))
    };
    let motion = match ex.execute("sequence.checkMotion", on(json!({}))).await {
        Ok(v) => json!({"findings": v["findings"].as_array().map_or(0, Vec::len), "byCode": by_code(&v)}),
        Err(e) => json!({"error": e}),
    };
    if let Some(q) = seq {
        let _ = ex.execute("sequence.open", json!({"item": q})).await;
    }
    // (the footage is drawn in code: no faces or plates to hide, and no narration)
    let preflight = match ex.execute("delivery.preflight", json!({"narration": false, "privacy": false})).await {
        Ok(v) => preflight_summary(&v),
        Err(e) => json!({"error": e}),
    };
    let sheet = filmcraft_autocut::trailer::beside(&file.display().to_string(), "sheet.jpg");
    let judged = match judge {
        Some(j) if export => j.score(&sheet),
        Some(_) => json!({"skipped": "--no-export: no frame sheet"}),
        None => Value::Null,
    };
    json!({
        "brief": b.name,
        "mode": mode,
        "pipeline": b.pipeline.name(),
        "status": made["status"],
        "deals": made["deals"],
        "axes": fingerprint["axes"],
        "evidence": fingerprint["evidence"],
        "motion": motion,
        "preflight": preflight,
        "judge": judged,
        "export": export.then(|| file.display().to_string()),
        "sheet": (export && sheet.exists()).then(|| sheet.display().to_string()),
        "seconds": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0,
    })
}

const AXES: [&str; 6] = ["opening", "titleEntrance", "transitionFamily", "colorLook", "captionStyle", "close"];

/// One mode's scores across its runs.
fn summarize(runs: &[&Value]) -> Value {
    let ok: Vec<&&Value> = runs.iter().filter(|r| r["error"].is_null()).collect();
    let n = ok.len();
    // variety: values each axis takes, and the axes consecutive briefs share
    let mut distinct = serde_json::Map::new();
    let mut spread = 0.0;
    for a in AXES {
        let mut vals: Vec<&str> = ok.iter().filter_map(|r| r["axes"][a].as_str()).collect();
        vals.sort_unstable();
        vals.dedup();
        if n > 1 {
            spread += vals.len().saturating_sub(1) as f64 / n.saturating_sub(1) as f64;
        }
        distinct.insert(a.to_string(), json!(vals));
    }
    let shared: Vec<usize> = ok
        .windows(2)
        .map(|w| {
            let (Some(x), Some(y)) = (w.first(), w.get(1)) else { return 0 };
            AXES.iter().filter(|a| matches!((x["axes"][**a].as_str(), y["axes"][**a].as_str()), (Some(p), Some(q)) if p == q)).count()
        })
        .collect();
    let alike = shared.iter().filter(|s| **s >= ALIKE).count();
    let mean = |v: &[f64]| if v.is_empty() { Value::Null } else { json!((v.iter().sum::<f64>() / v.len() as f64 * 100.0).round() / 100.0) };
    let motion: u64 = ok.iter().filter_map(|r| r["motion"]["findings"].as_u64()).sum();
    let blocked = ok.iter().filter(|r| r["preflight"]["passed"].as_bool() == Some(false)).count();
    let tier = |t: &str| -> u64 { ok.iter().filter_map(|r| r["preflight"]["summary"][t].as_u64()).sum() };
    let scores: Vec<f64> = ok.iter().filter_map(|r| r["judge"]["score"].as_f64()).collect();
    json!({
        "runs": runs.len(),
        "failed": runs.len().saturating_sub(n),
        "variety": {
            "spread": if n > 1 { json!((spread / AXES.len() as f64 * 100.0).round() / 100.0) } else { Value::Null },
            "sharedWithPrevious": shared,
            "meanShared": mean(&shared.iter().map(|s| *s as f64).collect::<Vec<_>>()),
            "alikePairs": alike,
            "values": distinct,
        },
        "motionFindings": motion,
        "preflight": {"blocked": blocked, "floor": tier("floor"), "reflex": tier("reflex"), "finish": tier("finish")},
        "judge": {"scored": scores.len(), "mean": mean(&scores)},
        "seconds": ok.iter().filter_map(|r| r["seconds"].as_f64()).sum::<f64>().round(),
    })
}

/// The newest earlier report under `root` (not `now`'s), if any.
fn previous(root: &Path, now: &str) -> Option<(String, Value)> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root).ok()?.flatten().map(|e| e.path()).filter(|p| p.join("report.json").is_file()).collect();
    dirs.sort();
    let dir = dirs.into_iter().rev().find(|d| d.file_name().and_then(|n| n.to_str()) != Some(now))?;
    let f = dir.join("report.json");
    if std::fs::metadata(&f).map(|m| m.len()).unwrap_or(u64::MAX) > MAX_BRIEF_BYTES * 64 {
        return None;
    }
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&f).ok()?).ok()?;
    Some((dir.file_name()?.to_str()?.to_string(), v))
}

fn num(v: &Value) -> String {
    match v.as_f64() {
        Some(x) if x.fract() == 0.0 => format!("{x:.0}"),
        Some(x) => format!("{x:.2}"),
        None => "–".into(),
    }
}

/// The report as a short markdown summary.
fn markdown(report: &Value) -> String {
    let mut md = format!("# Taste memory eval, {}\n\n", report["when"].as_str().unwrap_or("?"));
    md.push_str(&format!("{} briefs; judge: {}.\n\n", report["briefs"].as_array().map_or(0, Vec::len), report["judge"].as_str().unwrap_or("?")));
    md.push_str("| | runs | failed | axis spread | mean axes shared with the brief before | pairs sharing 4+ axes | motion findings | preflight blocked | judge mean | seconds |\n|---|---|---|---|---|---|---|---|---|---|\n");
    for mode in ["guided", "plain"] {
        let s = &report["summary"][mode];
        if s.is_null() {
            continue;
        }
        md.push_str(&format!(
            "| {mode} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            num(&s["runs"]),
            num(&s["failed"]),
            num(&s["variety"]["spread"]),
            num(&s["variety"]["meanShared"]),
            num(&s["variety"]["alikePairs"]),
            num(&s["motionFindings"]),
            num(&s["preflight"]["blocked"]),
            num(&s["judge"]["mean"]),
            num(&s["seconds"])
        ));
    }
    md.push_str("\nAxis spread: 0 when every brief has the same value on every axis, 1 when every brief differs on every axis.\n\n## Runs\n\n| brief | mode | opening | title | transitions | look | captions | close | motion | deals |\n|---|---|---|---|---|---|---|---|---|---|\n");
    for r in report["runs"].as_array().map(Vec::as_slice).unwrap_or_default() {
        if let Some(e) = r["error"].as_str() {
            md.push_str(&format!(
                "| {} | {} | failed: {} |||||||| \n",
                r["brief"].as_str().unwrap_or("?"),
                r["mode"].as_str().unwrap_or("?"),
                e.chars().take(120).collect::<String>()
            ));
            continue;
        }
        let a = |k: &str| r["axes"][k].as_str().unwrap_or("–").to_string();
        let deals: Vec<String> =
            r["deals"].as_object().map(|o| o.iter().filter_map(|(k, v)| v["choice"].as_str().map(|c| format!("{k} {c}"))).collect()).unwrap_or_default();
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            r["brief"].as_str().unwrap_or("?"),
            r["mode"].as_str().unwrap_or("?"),
            a("opening"),
            a("titleEntrance"),
            a("transitionFamily"),
            a("colorLook"),
            a("captionStyle"),
            a("close"),
            num(&r["motion"]["findings"]),
            if deals.is_empty() { "–".into() } else { deals.join(", ") }
        ));
    }
    if let Some(p) = report["previous"]["when"].as_str() {
        md.push_str(&format!(
            "\n## Against the previous run ({p})\n\n| | axis spread then → now | motion findings then → now | judge mean then → now |\n|---|---|---|---|\n"
        ));
        for mode in ["guided", "plain"] {
            let (now, then) = (&report["summary"][mode], &report["previous"]["summary"][mode]);
            if now.is_null() {
                continue;
            }
            md.push_str(&format!(
                "| {mode} | {} → {} | {} → {} | {} → {} |\n",
                num(&then["variety"]["spread"]),
                num(&now["variety"]["spread"]),
                num(&then["motionFindings"]),
                num(&now["motionFindings"]),
                num(&then["judge"]["mean"]),
                num(&now["judge"]["mean"])
            ));
        }
    }
    md
}

pub async fn run(a: &Args) {
    if let Some(o) = a.unknown_opt(&["--briefs", "--out", "--modes", "--only", "--no-export", "--no-judge", "--judge-model", "--compact"]) {
        usage(format!("eval: unknown option {o}\n{USAGE}"));
    }
    let dir = PathBuf::from(a.opt("--briefs").unwrap_or("eval/briefs"));
    let root = PathBuf::from(a.opt("--out").unwrap_or("target/eval"));
    let modes: Vec<bool> = match a
        .opt("--modes")
        .unwrap_or("guided,plain")
        .split(',')
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .map(|m| match m {
            "guided" => Ok(true),
            "plain" => Ok(false),
            o => Err(o.to_string()),
        })
        .collect::<Result<Vec<bool>, String>>()
    {
        Ok(m) if !m.is_empty() => m,
        _ => usage(format!("eval: --modes is guided, plain or both (guided,plain)\n{USAGE}")),
    };
    let mut all = briefs(&dir).unwrap_or_else(|e| fail(format!("eval: {e}")));
    if let Some(only) = a.opt("--only") {
        all.retain(|b| b.name == only);
        if all.is_empty() {
            fail(format!("eval: no brief named {only}"));
        }
    }
    let export = !a.flag("--no-export");
    let when = filmcraft_engine::taste_history::now();
    let stamp = when.replace(':', "-");
    let out = root.join(&stamp);
    let judge = if a.flag("--no-judge") { Err("--no-judge".to_string()) } else { Judge::find(a.opt("--judge-model")) };
    let judge_said = match &judge {
        Ok(j) => format!("{} (local Ollama)", j.name),
        Err(why) => format!("skipped: {why}"),
    };
    diag(format_args!("eval: {} brief(s), {} mode(s), into {}; judge {judge_said}", all.len(), modes.len(), out.display()));
    let mut runs: Vec<Value> = Vec::new();
    for guided in &modes {
        let mode = if *guided { "guided" } else { "plain" };
        let here = out.join(mode);
        let data = here.join("data");
        if let Err(e) = std::fs::create_dir_all(&data) {
            fail(format!("eval: {}: {e}", data.display()));
        }
        for b in &all {
            diag(format_args!("eval: {mode}: {} ({}, {} s)…", b.name, b.pipeline.name(), b.seconds));
            let r = run_brief(b, *guided, &data, &here, export, judge.as_ref().ok()).await;
            diag(format_args!(
                "eval: {mode}: {}: {}",
                b.name,
                r["error"]
                    .as_str()
                    .map_or_else(|| AXES.iter().map(|k| r["axes"][k].as_str().unwrap_or("?")).collect::<Vec<_>>().join(" · "), |e| format!("failed: {e}"))
            ));
            runs.push(r);
        }
    }
    let mut summary = serde_json::Map::new();
    for guided in &modes {
        let mode = if *guided { "guided" } else { "plain" };
        let mine: Vec<&Value> = runs.iter().filter(|r| r["mode"] == mode).collect();
        summary.insert(mode.to_string(), summarize(&mine));
    }
    let previous = previous(&root, &stamp).map(|(w, v)| json!({"when": w, "summary": v["summary"]}));
    let report = json!({
        "when": when,
        "briefs": all.iter().map(|b| json!({"name": b.name, "pipeline": b.pipeline.name(), "seconds": b.seconds})).collect::<Vec<_>>(),
        "judge": judge_said,
        "runs": runs,
        "summary": summary,
        "previous": previous,
    });
    let file = out.join("report.json");
    let md = out.join("summary.md");
    let written = std::fs::create_dir_all(&out)
        .and_then(|()| std::fs::write(&file, format!("{}\n", serde_json::to_string_pretty(&report).unwrap_or_default())))
        .and_then(|()| std::fs::write(&md, markdown(&report)));
    if let Err(e) = written {
        fail(format!("eval: {}: {e}", out.display()));
    }
    diag(format_args!("eval: wrote {} and {}", file.display(), md.display()));
    print(a, &json!({"report": file.display().to_string(), "summary": md.display().to_string(), "scores": report["summary"]}));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_committed_briefs_read() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../eval/briefs");
        let all = briefs(&dir).unwrap();
        assert!(all.len() >= 3 && all.iter().any(|b| b.pipeline == Pipeline::Quick) && all.iter().any(|b| b.pipeline == Pipeline::Direct));
        // owned footage only: demo scenes made in code, small
        assert!(all.iter().all(|b| b.size.0 <= SIDE.1 && b.clip_seconds <= CLIP_SECONDS.1 && b.seconds <= 60.0));
    }

    #[test]
    fn its_options_take_values() {
        let a =
            Args::parse(["eval", "--judge-model", "gemma3:12b", "--only", "quick-weekend", "--modes", "guided", "--briefs", "b"].iter().map(|s| s.to_string()));
        assert_eq!(
            (a.opt("--judge-model"), a.opt("--only"), a.opt("--modes"), a.opt("--briefs")),
            (Some("gemma3:12b"), Some("quick-weekend"), Some("guided"), Some("b"))
        );
        assert_eq!(a.positionals, ["eval"]);
    }

    #[test]
    fn hostile_briefs_are_refused() {
        for v in [
            json!({}),
            json!({"name": "a b", "pipeline": "direct", "seconds": 10, "footage": {"scenes": ["forest"]}}),
            json!({"name": "x", "pipeline": "showcase", "seconds": 10, "footage": {"scenes": ["forest"]}}),
            json!({"name": "x", "pipeline": "direct", "seconds": 1e9, "footage": {"scenes": ["forest"]}}),
            json!({"name": "x", "pipeline": "direct", "seconds": 10, "footage": {"scenes": []}}),
            json!({"name": "x", "pipeline": "direct", "seconds": 10, "footage": {"scenes": ["my-holiday.mov"]}}),
            json!({"name": "x", "pipeline": "direct", "seconds": 10, "footage": {"scenes": ["forest"], "width": 100_000}}),
            json!({"name": "x", "pipeline": "direct", "seconds": 10, "options": [1], "footage": {"scenes": ["forest"]}}),
        ] {
            assert!(brief(&v, "t.json").is_err(), "{v}");
        }
    }

    #[test]
    fn scores_compare_variety() {
        let run = |looks: &str| json!({"mode": "guided", "axes": {"opening": "coldOpen", "titleEntrance": "fade", "transitionFamily": "dissolve", "colorLook": looks, "captionStyle": "none", "close": "cut"}, "motion": {"findings": 1}, "preflight": {"passed": true, "summary": {"floor": 0, "reflex": 2, "finish": 1}}});
        let (a, b, c) = (run("none"), run("none"), run("warm"));
        let s = summarize(&[&a, &b, &c]);
        assert_eq!(s["variety"]["sharedWithPrevious"], json!([6, 5]));
        assert_eq!(s["variety"]["alikePairs"], 2);
        assert_eq!(s["motionFindings"], 3);
        assert_eq!((s["preflight"]["blocked"].as_u64(), s["preflight"]["reflex"].as_u64()), (Some(0), Some(6)));
        assert!(s["variety"]["spread"].as_f64().unwrap() > 0.0);
        let md = markdown(&json!({"when": "now", "briefs": [], "judge": "skipped", "runs": [a], "summary": {"guided": s}}));
        let v = scores_in("Score: 4/5\nvariety: 2\ncraft - none\nnotes: titles are clear");
        assert_eq!(
            (v["score"].as_f64(), v["variety"].as_f64(), v["craft"].as_f64(), v["notes"].as_str()),
            (Some(4.0), Some(2.0), None, Some("titles are clear"))
        );
        assert!(md.contains("| guided |") && md.contains("coldOpen"));
    }
}
