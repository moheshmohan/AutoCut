//! Natural spoken Malayalam narration (`docs/malayalam-style.md`): `ml-lint` (a Malayalam script
//! checked, as `script.lintMalayalam`), `ml-script` (a plan's English voice-over translated with
//! the local model, checked and revised once; `--remember` keeps a reviewed script's lines as
//! examples) and `showcase`'s check of a Malayalam script before it is voiced.

use filmcraft_autocut::ml_translate::{self, Settings, script_parts as parts_of};
use filmcraft_autocut::showcase::{Entry, Script};
use filmcraft_engine::Session;
use filmcraft_engine::ml_script::lint::{self, Level, Options};
use filmcraft_engine::ml_script::{English, english_of_plan, lint_parts, malayalam_parts};
use serde_json::{Value, json};

use crate::args::Args;
use crate::showcase_tools::{Run, read_json};
use crate::{absolute, diag, fail, print, usage};

fn write_json(what: &str, path: &str, v: &Value) {
    let text = serde_json::to_string_pretty(v).unwrap_or_default();
    std::fs::write(path, format!("{text}\n")).unwrap_or_else(|e| fail(format!("{what}: {path}: {e}")));
}

fn max_words(a: &Args, cmd: &str) -> Option<u64> {
    a.opt("--max-words").map(|m| {
        m.parse::<u64>()
            .ok()
            .filter(|n| (lint::MIN_MAX_WORDS as u64..=lint::MAX_MAX_WORDS as u64).contains(n))
            .unwrap_or_else(|| usage(format!("{cmd}: --max-words wants {}..{}, not `{m}`", lint::MIN_MAX_WORDS, lint::MAX_MAX_WORDS)))
    })
}

/// The findings of a `script.lintMalayalam` report as lines for people (hints too with `hints`).
fn report_lines(r: &Value, hints: bool) -> Vec<String> {
    let mut out = Vec::new();
    for p in r["parts"].as_array().into_iter().flatten() {
        let part = p["part"].as_str().unwrap_or("");
        for f in p["findings"].as_array().into_iter().flatten() {
            let hint = f["level"] == "hint";
            if hint && !hints {
                continue;
            }
            let kind = f["kind"].as_str().unwrap_or("");
            let (text, suggestion) = (f["text"].as_str().unwrap_or(""), f["suggestion"].as_str().unwrap_or(""));
            let tag = if hint { format!("hint ({kind})") } else { kind.to_string() };
            match kind {
                "longSentence" | "noVerb" => out.push(format!("{part}: {tag}: {} — {suggestion}", f["message"].as_str().unwrap_or(""))),
                _ => out.push(format!("{part}: {tag}: «{text}» → {suggestion}")),
            }
        }
        let missing: Vec<&str> = p["facts"]["missing"].as_array().into_iter().flatten().filter_map(|m| m["text"].as_str()).collect();
        let added: Vec<&str> = p["facts"]["added"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        if !missing.is_empty() || !added.is_empty() {
            let mut s = Vec::new();
            if !missing.is_empty() {
                s.push(format!("missing: {}", missing.join(", ")));
            }
            if !added.is_empty() {
                s.push(format!("added: {}", added.join(", ")));
            }
            out.push(format!("{part}: facts: {}", s.join("; ")));
        }
    }
    out
}

/// `ml-lint <script> [--plan plan.json] [--max-words 14] [--fix --out fixed.json] [--hints]
/// [--report-out report.json] [--strict]`.
pub fn lint(a: &Args) {
    const USAGE: &str = "ml-lint <script-ml.json | plan-ml.json | script.txt> [--plan plan.json] [--max-words 14] [--fix --out fixed.json] [--hints] [--report-out report.json] [--strict]";
    let Some(path) = a.pos(1) else { usage(USAGE) };
    let mut p = json!({"path": absolute(path)});
    if let Some(e) = a.opt("--plan") {
        p["englishPath"] = json!(absolute(e));
    }
    if let Some(n) = max_words(a, "ml-lint") {
        p["maxWords"] = json!(n);
    }
    let fix = a.flag("--fix");
    if fix && a.opt("--out").is_none() {
        usage("ml-lint: --fix writes the fixed script: add --out fixed.json");
    }
    if fix {
        p["fix"] = json!(true);
    }
    let mut s = Session::default();
    let r = s.execute("script.lintMalayalam", p).unwrap_or_else(|e| fail(format!("ml-lint: {e}")));
    for line in report_lines(&r, a.flag("--hints")) {
        diag(format_args!("ml-lint: {line}"));
    }
    if let (true, Some(out)) = (fix, a.opt("--out")) {
        write_json("ml-lint", out, &r["fixedScript"]);
        let applied: usize = r["parts"].as_array().into_iter().flatten().map(|p| p["applied"].as_array().map_or(0, Vec::len)).sum();
        diag(format_args!("ml-lint: {applied} safe fix(es) made; the script is in {out}"));
    }
    if let Some(out) = a.opt("--report-out") {
        write_json("ml-lint", out, &r);
    }
    print(a, &r["summary"]);
    let clean = r["summary"]["clean"].as_bool().unwrap_or(false);
    if a.flag("--strict") && !clean {
        fail(format!(
            "ml-lint: {} warning(s), {} fact(s) missing, {} number(s) added",
            r["summary"]["warnings"],
            r["summary"]["factsMissing"].as_u64().unwrap_or(0),
            r["summary"]["numbersAdded"].as_u64().unwrap_or(0)
        ));
    }
}

/// `ml-script --plan plan.json --out script-ml.json [--model qwen3:8b] [--examples pairs.json]
/// [--no-revise] [--max-words 14] [--batch 3] [--report-out report.json]`, or `--remember
/// reviewed.json` to keep a reviewed script's lines as examples.
pub async fn translate(a: &Args) {
    const USAGE: &str = "ml-script --plan plan.json (--out script-ml.json [--model qwen3:8b] [--examples pairs.json] [--no-revise] [--max-words 14] [--batch 3] [--report-out report.json] | --remember script-ml.json)";
    let Some(plan_path) = a.opt("--plan") else { usage(USAGE) };
    let examples_path = a.opt("--examples").map(std::path::PathBuf::from).or_else(ml_translate::examples_path);
    let max = max_words(a, "ml-script");
    let batch = a
        .opt("--batch")
        .map(|b| b.parse::<usize>().ok().filter(|n| (1..=12).contains(n)).unwrap_or_else(|| usage(format!("ml-script: --batch wants 1..12, not `{b}`"))));
    let remember = a.opt("--remember");
    let out = a.opt("--out");
    if remember.is_none() && out.is_none() {
        usage(USAGE);
    }
    let plan = read_json("ml-script: --plan", plan_path);
    let english = english_of_plan(&plan).unwrap_or_else(|e| fail(format!("ml-script: {plan_path}: {e}")));
    if let Some(reviewed) = remember {
        let ml = read_json("ml-script: --remember", reviewed);
        let parts = malayalam_parts(&ml).unwrap_or_else(|e| fail(format!("ml-script: {reviewed}: {e}")));
        let pairs = ml_translate::pairs_of(&english, &parts);
        let path = examples_path.unwrap_or_else(|| fail("ml-script: no data directory for the examples file; give --examples pairs.json"));
        let kept = ml_translate::remember(&path, &pairs).unwrap_or_else(|e| fail(format!("ml-script: {e}")));
        print(a, &json!({"remembered": pairs.len(), "kept": kept, "examples": path.display().to_string()}));
        return;
    }
    let Some(out) = out else { usage(USAGE) };
    let cfg = filmcraft_autocut::config::Config::from_env();
    let name = a.opt("--model").map_or_else(|| cfg.script_model.clone(), str::to_string);
    let model = filmcraft_autocut::ollama::Ollama::new(&cfg.ollama_host, &name, cfg.http_timeout).with_context(ml_translate::CONTEXT_TOKENS);
    let probe = model.clone();
    let ready = tokio::task::spawn_blocking(move || probe.available()).await.unwrap_or_else(|e| Err(filmcraft_autocut::AutoCutError::Other(e.to_string())));
    match ready {
        Ok(true) => {}
        Ok(false) => fail(format!("ml-script: the model `{name}` is not in Ollama at {}: `ollama pull {name}`", cfg.ollama_host)),
        Err(e) => fail(format!("ml-script: {e}")),
    }
    let examples = match &examples_path {
        Some(p) => ml_translate::load_examples(p).unwrap_or_else(|e| fail(format!("ml-script: {e}"))),
        None => Vec::new(),
    };
    if !examples.is_empty() {
        diag(format_args!(
            "ml-script: {} approved example(s) from {}",
            examples.len(),
            examples_path.as_ref().map(|p| p.display().to_string()).unwrap_or_default()
        ));
    }
    let defaults = Settings::default();
    let settings = Settings {
        max_words: max.map_or(defaults.max_words, |n| n as usize),
        revise: !a.flag("--no-revise"),
        batch: batch.unwrap_or(defaults.batch),
        examples,
    };
    let t0 = std::time::Instant::now();
    let t = ml_translate::translate(&model, &plan, &settings, &mut |l| diag(format_args!("ml-script: {l}")))
        .await
        .unwrap_or_else(|e| fail(format!("ml-script: {e}")));
    write_json("ml-script", out, &t.script());
    for n in t.notes() {
        diag(format_args!("ml-script: {n}"));
    }
    if let Some(r) = a.opt("--report-out") {
        write_json("ml-script", r, &serde_json::to_value(&t).unwrap_or_default());
    }
    let mut summary = t.summary();
    summary["out"] = json!(out);
    summary["seconds"] = json!((t0.elapsed().as_secs_f64() * 10.0).round() / 10.0);
    print(a, &summary);
}

/// The Malayalam script of each Malayalam build, linted (and checked against `english`, the
/// plan's English script, when the build reads its lines over it) before anything is voiced:
/// the warnings are printed; with `strict` they fail the build.
pub fn check_showcase(runs: &[Run], entries: &[Entry], english: Option<&Script>, strict: bool) -> Result<(), String> {
    for run in runs.iter().filter(|r| r.opts.malayalam()) {
        let parts = parts_of(&run.script, entries);
        let en =
            english.map(|s| English { parts: parts_of(s, entries), pronounce: s.pronounce.clone(), labels: entries.iter().map(|e| e.label.clone()).collect() });
        let reports = lint_parts(&parts, en.as_ref(), &Options::default(), false);
        let warnings: usize = reports.iter().map(|r| r.warnings()).sum();
        let facts: usize = reports.iter().map(|r| r.facts_missing()).sum();
        for r in &reports {
            for line in r.lines(false) {
                diag(format_args!("showcase (Malayalam): script: {line}"));
            }
        }
        let hints: usize = reports.iter().flat_map(|r| &r.findings).filter(|f| f.level == Level::Hint).count();
        diag(format_args!(
            "showcase (Malayalam): script: {warnings} warning(s){}, {hints} hint(s) in {} part(s) (`filmcraft-cli ml-lint` shows them; docs/malayalam-style.md)",
            if english.is_some() { format!(", {facts} fact(s) lost or added") } else { String::new() },
            reports.len()
        ));
        if strict && warnings + facts > 0 {
            return Err(format!(
                "--strict-script: the Malayalam script has {warnings} warning(s){} (above); fix them, or leave out --strict-script",
                if english.is_some() { format!(" and {facts} fact(s) lost or added") } else { String::new() }
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_autocut::showcase::Options as ShowOptions;

    fn entry(label: &str) -> Entry {
        serde_json::from_value(json!({"number": 1, "label": label, "kind": "robot", "description": "", "unix": 1, "shots": []})).unwrap()
    }

    #[test]
    fn a_malayalam_build_is_checked_before_it_is_voiced() {
        let entries = vec![entry("Pepper"), entry("Scene")];
        let english = Script {
            intro: "Hi.".into(),
            lines: vec!["Pepper greets visitors; SoftBank made about twenty-seven thousand.".into(), "".into()],
            outro: "Bye.".into(),
            ..Script::default()
        };
        let ml = Script {
            intro: "ഹായ്.".into(),
            lines: vec!["പെപ്പർ സന്ദർശകരെ സ്വാഗതം ചെയ്യുന്നു. സോഫ്റ്റ്ബാങ്ക് ഇരുപത്തിയേഴായിരം എണ്ണം ഉണ്ടാക്കി.".into(), "".into()],
            outro: "ബൈ.".into(),
            ..Script::default()
        };
        let run = |language: Option<&str>, script: &Script| Run {
            opts: ShowOptions { language: language.map(str::to_string), ..ShowOptions::default() },
            script: script.clone(),
            title: "t".into(),
        };
        // warnings are printed; only --strict-script fails on them
        let runs = vec![run(None, &english), run(Some("ml"), &ml)];
        assert!(check_showcase(&runs, &entries, Some(&english), false).is_ok());
        let e = check_showcase(&runs, &entries, Some(&english), true).unwrap_err();
        assert!(e.contains("3 warning(s)") && e.contains("0 fact(s)"), "{e}");
        // an English build is not checked; a clean Malayalam one passes strict
        assert!(check_showcase(&[run(None, &ml)], &entries, None, true).is_ok());
        let good = Script {
            lines: vec!["പെപ്പർ വിസിറ്റേഴ്സിനെ വെൽക്കം ചെയ്യുന്നു. സോഫ്റ്റ്ബാങ്ക് 27,000 എണ്ണം ഉണ്ടാക്കി.".into(), "".into()], ..ml.clone()
        };
        assert!(check_showcase(&[run(Some("ml"), &good)], &entries, Some(&english), true).is_ok());
        assert_eq!(parts_of(&good, &entries).len(), 3);
    }
}
