//! `voice …`: the voice lab ([`filmcraft_engine::voice_lab`]) — better references for a natural
//! cloned voice, found in the speaker's own clips, auditioned, and put in place with the old one
//! kept; and the mix check (how far the showcase's narration chain moves a voice).

use serde_json::{Value, json};

use crate::args::Args;
use crate::{Backend, absolute, diag, fail, print, usage};

const USAGE: &str = "\
voice refs --voice clone:<name> <media|folder|project.fcproj>… [--out DIR] [--max-minutes M] [--max-clip-seconds S]
           [--max-files N] [--min-similarity X] [--top N] [--per-clip N] [--held-out-seconds S]
voice audition --voice clone:<name> --report <refs.json> [--out DIR] [--top K] [--takes N] [--engines voxcpm,indicf5] [--no-variants] [--no-m4a]
voice promote --voice clone:<name> --report <audition.json|refs.json> <candidate id> [--transcript \"exact words\"] [--min-similarity X]
voice versions --voice clone:<name>
voice rollback --voice clone:<name> <version (r0, r1, …)>
voice mix-check --takes <wav|folder>… (--genuine <wav>… | --report <refs.json>) [--chains chains.json] [--lufs -16]";

/// The chains `voice mix-check` compares by default: the narration chain as the showcase puts it
/// on the cloned voice (all of its EQ: the voice measured dark), and the voice-preserving chain.
fn default_chains() -> Value {
    let to_json = |chain: Vec<(&str, Value)>| Value::Array(chain.into_iter().map(|(e, p)| json!({"effect": e, "params": p})).collect());
    json!({
        "none": [],
        "speech": to_json(filmcraft_autocut::mix::voice_chain(1.0)),
        "speech (EQ only)": to_json(filmcraft_autocut::mix::voice_chain(1.0).into_iter().take(1).collect()),
        "gentle": to_json(filmcraft_autocut::mix::voice_chain_gentle(1.0)),
    })
}

fn wavs(paths: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for p in paths {
        let path = std::path::Path::new(p);
        if path.is_dir() {
            let mut found: Vec<String> = std::fs::read_dir(path)
                .into_iter()
                .flatten()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|q| q.extension().is_some_and(|x| x.eq_ignore_ascii_case("wav")))
                .map(|q| q.to_string_lossy().into_owned())
                .collect();
            found.sort();
            out.extend(found);
        } else {
            out.push(absolute(p));
        }
    }
    out
}

fn num(a: &Args, opt: &str) -> Option<f64> {
    a.opt(opt).map(|v| v.parse::<f64>().ok().filter(|x| x.is_finite()).unwrap_or_else(|| usage(format!("voice: {opt} wants a number, not {v}"))))
}

pub async fn run(a: &Args) {
    let what = a.pos(1).unwrap_or_else(|| usage(format!("voice: what?\n{USAGE}")));
    let allowed = [
        "--voice",
        "--out",
        "--max-minutes",
        "--max-clip-seconds",
        "--max-files",
        "--min-similarity",
        "--top",
        "--per-clip",
        "--held-out-seconds",
        "--report",
        "--engines",
        "--no-variants",
        "--no-m4a",
        "--takes",
        "--genuine",
        "--chains",
        "--lufs",
        "--transcript",
        "--json",
        "--compact",
        "--data-dir",
        "--bridge",
    ];
    if let Some(o) = a.unknown_opt(&allowed) {
        usage(format!("voice: unknown option {o}\n{USAGE}"));
    }
    let voice = || a.opt("--voice").unwrap_or_else(|| usage(format!("voice {what}: --voice clone:<name> is required\n{USAGE}"))).to_string();
    let mut b = Backend::open(a);
    // the lab's jobs run to the end here; through the app (--bridge) they run as its jobs
    let wait = matches!(b, Backend::Local(_));
    let (id, mut params) = match what {
        "refs" | "references" | "find" => {
            let paths: Vec<String> = a.positionals.iter().skip(2).map(|p| absolute(p)).collect();
            if paths.is_empty() {
                usage(format!("voice refs: give the clips to look in (files, folders or .fcproj projects)\n{USAGE}"));
            }
            let mut p = json!({"voice": voice(), "paths": paths, "wait": wait});
            for (opt, key) in [
                ("--max-minutes", "maxMinutes"),
                ("--max-clip-seconds", "maxClipSeconds"),
                ("--max-files", "maxFiles"),
                ("--min-similarity", "minSimilarity"),
                ("--top", "top"),
                ("--per-clip", "perClip"),
                ("--held-out-seconds", "heldOutSeconds"),
            ] {
                if let Some(v) = num(a, opt) {
                    p[key] = json!(v);
                }
            }
            ("voice.findReferences", p)
        }
        "audition" => {
            let report = a.opt("--report").unwrap_or_else(|| usage(format!("voice audition: --report <refs.json> is required\n{USAGE}")));
            let mut p = json!({"voice": voice(), "report": absolute(report), "wait": wait, "variants": !a.flag("--no-variants"), "m4a": !a.flag("--no-m4a")});
            if let Some(v) = num(a, "--top") {
                p["top"] = json!(v);
            }
            if let Some(v) = num(a, "--takes") {
                p["takes"] = json!(v);
            }
            if let Some(e) = a.opt("--engines") {
                p["engines"] = json!(e.split(',').map(str::trim).filter(|x| !x.is_empty()).collect::<Vec<_>>());
            }
            ("voice.audition", p)
        }
        "promote" => {
            let report = a.opt("--report").unwrap_or_else(|| usage(format!("voice promote: --report is required\n{USAGE}")));
            let candidate = a.pos(2).unwrap_or_else(|| usage(format!("voice promote: which candidate?\n{USAGE}")));
            let mut p = json!({"voice": voice(), "report": absolute(report), "candidate": candidate});
            if let Some(v) = num(a, "--min-similarity") {
                p["minSimilarity"] = json!(v);
            }
            if let Some(t) = a.opt("--transcript") {
                p["transcript"] = json!(t);
            }
            ("voice.promote", p)
        }
        "versions" => ("voice.versions", json!({"voice": voice()})),
        "rollback" => {
            let version = a.pos(2).unwrap_or_else(|| usage(format!("voice rollback: which version (voice versions lists them)?\n{USAGE}")));
            ("voice.rollback", json!({"voice": voice(), "version": version}))
        }
        "mix-check" => {
            let takes = wavs(&a.all("--takes"));
            if takes.is_empty() {
                usage(format!("voice mix-check: --takes <wav|folder> is required\n{USAGE}"));
            }
            let chains = match a.opt("--chains") {
                Some(f) => crate::showcase_tools::read_json("voice mix-check: --chains", f),
                None => default_chains(),
            };
            let mut p = json!({"takes": takes, "genuine": wavs(&a.all("--genuine")), "chains": chains});
            if let Some(r) = a.opt("--report") {
                p["report"] = json!(absolute(r));
            }
            if let Some(v) = num(a, "--lufs") {
                p["lufs"] = json!(v);
            }
            ("voice.mixCheck", p)
        }
        other => usage(format!("voice: unknown `{other}`\n{USAGE}")),
    };
    if let Some(o) = a.opt("--out") {
        params["out"] = json!(absolute(o));
    }
    match b.exec(id, params).await {
        Ok(v) => {
            if let Some(s) = v["summary"].as_str() {
                diag(format_args!("voice {what}: {s}"));
            }
            print(a, &v);
        }
        Err(e) => fail(format!("voice {what}: {e}")),
    }
    b.finish(a).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_chains_are_the_showcase_narration_chain() {
        let c = default_chains();
        let speech = c["speech"].as_array().unwrap();
        let effects: Vec<&str> = speech.iter().map(|s| s["effect"].as_str().unwrap()).collect();
        assert_eq!(effects, filmcraft_autocut::mix::MIX_INSERTS);
        assert!(c["none"].as_array().unwrap().is_empty());
    }
}
