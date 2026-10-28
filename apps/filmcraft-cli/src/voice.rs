//! `voice …`: the voice lab ([`filmcraft_engine::voice_lab`]) — better references for a natural
//! cloned voice, found in the speaker's own clips, auditioned, and put in place with the old one
//! kept; the mix check (how far the showcase's narration chain moves a voice); the style finder
//! (references sorted by delivery) and the prosody measure (and reshaping) of recordings.

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
voice mix-check --takes <wav|folder>… (--genuine <wav>… | --report <refs.json>) [--chains chains.json] [--lufs -16]
voice styles --voice clone:<name> [<media|folder|project.fcproj>…] [--items ID,ID… (with --project)] [--top 3] [--out DIR]
             [--language ml|any] [--no-hear] [--speaker-ref <file>[@start-end]]… [--max-minutes M] [--max-clip-seconds S]
             [--max-files N] [--min-similarity X]
voice prosody [<wav|media>…] [--segment <file>@<start>-<end>]… [--range 1.3,1.6] [--shift ST] [--question ST [--question-end S]]
              [--emphasis DB] [--out DIR] [--hear]";

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

/// `<file>@<start>-<end>` (seconds) as `{"path", "start", "end"}`; a bare file is all of it.
fn segment_arg(what: &str, s: &str) -> Value {
    let bad = || -> ! { usage(format!("{what} is <file>@<start>-<end> (seconds), not {s}")) };
    let Some((file, span)) = s.rsplit_once('@') else { return json!(absolute(s)) };
    let Some((start, end)) = span.split_once('-') else { bad() };
    let n = |v: &str| match v.trim().parse::<f64>() {
        Ok(x) if x.is_finite() => x,
        _ => bad(),
    };
    json!({"path": absolute(file), "start": n(start), "end": n(end)})
}

/// `voice prosody`'s parameters: the recordings (whole, or `--segment file@start-end`) and the
/// shapes (one per `--range` value, each with the shift, question rise and emphasis given).
fn prosody_params(a: &Args, wait: bool) -> Value {
    let paths: Vec<String> = a.positionals.iter().skip(2).map(|p| absolute(p)).collect();
    let segments: Vec<Value> = a.all("--segment").into_iter().map(|s| segment_arg("voice prosody: --segment", s)).collect();
    if paths.is_empty() && segments.is_empty() {
        usage(format!("voice prosody: give the recordings (files, or --segment file@start-end)\n{USAGE}"));
    }
    let mut p = json!({"wait": wait, "hear": a.flag("--hear")});
    if !paths.is_empty() {
        p["paths"] = json!(paths);
    }
    if !segments.is_empty() {
        p["segments"] = json!(segments);
    }
    let ranges: Vec<f64> = a
        .opt("--range")
        .map(|v| {
            v.split(',')
                .map(|x| {
                    x.trim().parse::<f64>().ok().filter(|r| r.is_finite()).unwrap_or_else(|| usage(format!("voice prosody: --range wants numbers, not {v}")))
                })
                .collect()
        })
        .unwrap_or_default();
    let (shift, question, end, emphasis) = (num(a, "--shift"), num(a, "--question"), num(a, "--question-end"), num(a, "--emphasis"));
    let shaped = !ranges.is_empty() || shift.is_some() || question.is_some() || emphasis.is_some();
    if shaped {
        let base = |range: f64| {
            let mut s = json!({"range": range});
            if let Some(v) = shift {
                s["shift"] = json!(v);
            }
            if let Some(v) = question {
                s["questionRise"] = json!(v);
            }
            if let Some(v) = end {
                s["questionEnd"] = json!(v);
            }
            if let Some(v) = emphasis {
                s["emphasisDb"] = json!(v);
            }
            s
        };
        let shapes: Vec<Value> = if ranges.is_empty() { vec![base(1.0)] } else { ranges.iter().map(|r| base(*r)).collect() };
        p["shapes"] = json!(shapes);
    }
    p
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
        "--items",
        "--language",
        "--no-hear",
        "--segment",
        "--speaker-ref",
        "--range",
        "--shift",
        "--question",
        "--question-end",
        "--emphasis",
        "--hear",
        "--project",
        "--json",
        "--compact",
        "--data-dir",
        "--bridge",
    ];
    if let Some(o) = a.unknown_opt(&allowed) {
        usage(format!("voice: unknown option {o}\n{USAGE}"));
    }
    let voice = || a.opt("--voice").unwrap_or_else(|| usage(format!("voice {what}: --voice clone:<name> is required\n{USAGE}"))).to_string();
    if what == "styles" && a.opt("--items").is_some() && a.opt("--project").is_none() && a.opt("--bridge").is_none() {
        usage(format!("voice styles: --items needs the project they are in (--project file.fcproj), or --bridge to the app\n{USAGE}"));
    }
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
        "styles" | "find-styles" => {
            let paths: Vec<String> = a.positionals.iter().skip(2).map(|p| absolute(p)).collect();
            let items: Vec<u64> = a
                .opt("--items")
                .map(|v| {
                    v.split(',')
                        .map(str::trim)
                        .filter(|x| !x.is_empty())
                        .map(|x| x.parse::<u64>().unwrap_or_else(|_| usage(format!("voice styles: --items wants item ids, not {x}"))))
                        .collect()
                })
                .unwrap_or_default();
            if paths.is_empty() && items.is_empty() {
                usage(format!("voice styles: give the clips to look in (files, folders or .fcproj projects) or --items\n{USAGE}"));
            }
            let mut p = json!({"voice": voice(), "wait": wait, "hear": !a.flag("--no-hear")});
            if !paths.is_empty() {
                p["paths"] = json!(paths);
            }
            if !items.is_empty() {
                p["items"] = json!(items);
            }
            if let Some(l) = a.opt("--language") {
                p["language"] = json!(l);
            }
            let refs: Vec<Value> = a.all("--speaker-ref").into_iter().map(|s| segment_arg("voice styles: --speaker-ref", s)).collect();
            if !refs.is_empty() {
                p["speakerRefs"] = json!(refs);
            }
            for (opt, key) in [
                ("--top", "top"),
                ("--max-minutes", "maxMinutes"),
                ("--max-clip-seconds", "maxClipSeconds"),
                ("--max-files", "maxFiles"),
                ("--min-similarity", "minSimilarity"),
            ] {
                if let Some(v) = num(a, opt) {
                    p[key] = json!(v);
                }
            }
            ("voice.findStyles", p)
        }
        "prosody" => ("voice.prosody", prosody_params(a, wait)),
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
    fn prosody_options_become_shapes_and_segments() {
        let a = Args::parse(["voice", "prosody", "/a.wav", "--segment", "/b c.mov@1.5-9", "--range", "1.3,1.6", "--question", "3", "--hear"].map(String::from));
        let p = prosody_params(&a, true);
        assert_eq!(p["paths"], json!(["/a.wav"]));
        assert_eq!(p["segments"], json!([{"path": "/b c.mov", "start": 1.5, "end": 9.0}]));
        assert_eq!(p["shapes"], json!([{"range": 1.3, "questionRise": 3.0}, {"range": 1.6, "questionRise": 3.0}]));
        assert_eq!((p["hear"].as_bool(), p["wait"].as_bool()), (Some(true), Some(true)));
        assert_eq!(segment_arg("x", "/r.mov"), json!("/r.mov"));
        // measuring only: no shapes
        let a = Args::parse(["voice", "prosody", "/a.wav"].map(String::from));
        assert!(prosody_params(&a, false).get("shapes").is_none());
    }

    #[test]
    fn the_default_chains_are_the_showcase_narration_chain() {
        let c = default_chains();
        let speech = c["speech"].as_array().unwrap();
        let effects: Vec<&str> = speech.iter().map(|s| s["effect"].as_str().unwrap()).collect();
        assert_eq!(effects, filmcraft_autocut::mix::MIX_INSERTS);
        assert!(c["none"].as_array().unwrap().is_empty());
    }
}
