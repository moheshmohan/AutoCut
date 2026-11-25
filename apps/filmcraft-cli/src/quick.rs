//! `quick …`: an album, a folder or clips → a finished vertical Short or Reel from a template
//! filled automatically (`autocut.quick`).

use serde_json::{Value, json};

use crate::args::Args;
use crate::{Backend, absolute, diag, fail, print, usage};

pub const USAGE: &str = "\
quick <folder | files… | project.fcproj> [--template auto|ID] [--aspect 9:16|1:1|4:5|16:9]
      [--platform reels|shorts|tiktok|vertical] [--seconds 30] [--topic T] [--language en|ml]
      [--look tech|vhs|clean] [--music file|generate] [--redact faces,plates|faces|plates|off]
      [--caption-style bold|clean|glow|box|karaoke|twoLine|plain] [--name N] [--full-detail 12]
      [--reference liked.mp4|ITEM] [--no-qa] [--export out.mp4] [--download]
      [--save | --save-as p.fcproj]";

/// `quick`'s options as `autocut.quick` parameters (with `wait: true`); a usage error says what
/// is wrong. `sources` are the positional files, folders or one project.
pub fn params(a: &Args) -> Result<Value, String> {
    let mut p = json!({"wait": true});
    for (opt, key) in [
        ("--template", "template"),
        ("--aspect", "aspect"),
        ("--platform", "platform"),
        ("--topic", "topic"),
        ("--language", "language"),
        ("--look", "look"),
        ("--redact", "redact"),
        ("--caption-style", "captionStyle"),
        ("--name", "name"),
    ] {
        if let Some(v) = a.opt(opt) {
            p[key] = json!(v);
        }
    }
    if let Some(v) = a.opt("--seconds") {
        let s = v.parse::<f64>().ok().filter(|f| f.is_finite() && (5.0..=180.0).contains(f)).ok_or_else(|| format!("--seconds wants 5 to 180, not `{v}`"))?;
        p["seconds"] = json!(s);
    }
    if let Some(v) = a.opt("--full-detail") {
        let n = v.parse::<u64>().ok().filter(|n| *n <= 64).ok_or_else(|| format!("--full-detail wants 0 to 64 shots, not `{v}`"))?;
        p["fullDetail"] = json!(n);
    }
    if let Some(m) = a.opt("--music") {
        p["music"] = json!(if m.eq_ignore_ascii_case("generate") { m.to_string() } else { absolute(m) });
    }
    if let Some(e) = a.opt("--export") {
        p["export"] = json!(absolute(e));
    }
    if let Some(r) = a.opt("--reference") {
        p["reference"] = crate::direct::reference(r)?;
    }
    if a.flag("--no-qa") {
        p["qa"] = json!(false);
    }
    if a.flag("--download") {
        p["download"] = json!(true);
    }
    if let Some(look) = a.opt("--look")
        && !matches!(look, "tech" | "vhs" | "clean")
    {
        return Err(format!("--look is tech, vhs or clean, not `{look}`"));
    }
    Ok(p)
}

pub async fn run(a: &Args) {
    let sources: Vec<String> = a.positionals.get(1..).map(<[String]>::to_vec).unwrap_or_default();
    let allowed = [
        "--template",
        "--aspect",
        "--platform",
        "--seconds",
        "--topic",
        "--language",
        "--look",
        "--music",
        "--redact",
        "--caption-style",
        "--name",
        "--full-detail",
        "--reference",
        "--no-qa",
        "--export",
        "--download",
        "--save",
        "--save-as",
        "--project",
        "--bridge",
        "--data-dir",
        "--compact",
    ];
    if let Some(o) = a.unknown_opt(&allowed) {
        usage(format!("quick: unknown option {o}\n{USAGE}"));
    }
    let mut p = params(a).unwrap_or_else(|e| usage(format!("quick: {e}\n{USAGE}")));
    let project = sources.iter().find(|s| std::path::Path::new(s).extension().is_some_and(|x| x.eq_ignore_ascii_case("fcproj"))).cloned();
    let files: Vec<String> = sources.iter().filter(|s| Some(*s) != project.as_ref()).map(|s| absolute(s)).collect();
    if sources.is_empty() && a.opt("--project").is_none() && a.opt("--bridge").is_none() {
        usage(format!("quick: what to make it of? a folder, files or a project\n{USAGE}"));
    }
    if a.opt("--bridge").is_none() {
        for f in &files {
            if !std::path::Path::new(f).exists() {
                fail(format!("quick: {f}: no such file or folder"));
            }
        }
    }
    let mut b = Backend::open(a);
    if let Some(proj) = &project {
        b.exec("file.open", json!({"path": absolute(proj)})).await.unwrap_or_else(|e| fail(format!("quick: {proj}: {e}")));
    }
    if !files.is_empty() {
        p["paths"] = json!(files);
    }
    diag(format_args!("quick: indexing, listening, choosing a template and filling it (this takes a while the first time)"));
    match b.exec("autocut.quick", p).await {
        Ok(r) => {
            let made = &r["showcase"];
            let chosen = &made["template"]["chosen"];
            diag(format_args!(
                "quick: template {} ({}) — {}",
                chosen["name"].as_str().unwrap_or("?"),
                chosen["id"].as_str().unwrap_or("?"),
                chosen["reasons"].as_array().map(|r| r.iter().filter_map(Value::as_str).collect::<Vec<_>>().join("; ")).unwrap_or_default()
            ));
            for pk in made["picks"].as_array().map(Vec::as_slice).unwrap_or_default().iter().take(64) {
                diag(format_args!(
                    "quick:   {} {}: {} {:.1}–{:.1} s ({:.2}) {}",
                    pk["slot"].as_str().unwrap_or_default(),
                    pk["media"].as_u64().unwrap_or(0) + 1,
                    pk["name"].as_str().unwrap_or_default(),
                    pk["inSeconds"].as_f64().unwrap_or(0.0),
                    pk["outSeconds"].as_f64().unwrap_or(0.0),
                    pk["score"].as_f64().unwrap_or(0.0),
                    pk["why"].as_array().map(|w| w.iter().filter_map(Value::as_str).take(3).collect::<Vec<_>>().join(", ")).unwrap_or_default()
                ));
            }
            for t in made["texts"].as_array().map(Vec::as_slice).unwrap_or_default().iter().take(64) {
                diag(format_args!(
                    "quick:   text {} {}: \"{}\" ({})",
                    t["slot"].as_str().unwrap_or_default(),
                    t["n"],
                    t["text"].as_str().unwrap_or_default(),
                    t["by"].as_str().unwrap_or_default()
                ));
            }
            for f in made["fallbacks"].as_array().map(Vec::as_slice).unwrap_or_default() {
                diag(format_args!("quick: fallback: {}", f.as_str().unwrap_or_default()));
            }
            for w in made["warnings"].as_array().map(Vec::as_slice).unwrap_or_default().iter().take(30) {
                diag(format_args!("quick: note: {}", w.as_str().unwrap_or_default()));
            }
            diag(format_args!("quick: {}", made["status"].as_str().unwrap_or("done")));
            print(a, made);
            b.finish(a).await;
        }
        Err(e) => fail(format!("quick: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Args {
        Args::parse(v.iter().map(|s| s.to_string()))
    }

    #[test]
    fn options_become_parameters() {
        let p = params(&args(&[
            "quick",
            "clips/",
            "--template",
            "builtin:tech-reveal",
            "--aspect",
            "1:1",
            "--seconds",
            "20",
            "--topic",
            "robots",
            "--look",
            "vhs",
            "--music",
            "generate",
            "--redact",
            "off",
            "--full-detail",
            "4",
            "--no-qa",
            "--download",
        ]))
        .unwrap();
        assert_eq!(p["template"], "builtin:tech-reveal");
        assert_eq!((p["aspect"].as_str(), p["seconds"].as_f64(), p["look"].as_str()), (Some("1:1"), Some(20.0), Some("vhs")));
        assert_eq!((p["music"].as_str(), p["redact"].as_str(), p["fullDetail"].as_u64()), (Some("generate"), Some("off"), Some(4)));
        assert_eq!((p["qa"].as_bool(), p["download"].as_bool(), p["wait"].as_bool()), (Some(false), Some(true), Some(true)));
        assert_eq!(params(&args(&["quick", "x", "--reference", "7"])).unwrap()["reference"], json!(7));
        for bad in [
            &["quick", "x", "--seconds", "0"][..],
            &["quick", "x", "--seconds", "NaN"],
            &["quick", "x", "--full-detail", "999"],
            &["quick", "x", "--look", "sparkly"],
            &["quick", "x", "--reference", ""],
        ] {
            assert!(params(&args(bad)).is_err(), "{bad:?}");
        }
    }
}
