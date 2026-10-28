//! `direct …`: normal and 360° clips → one catchy edit on the beat, directed by the local model
//! (`autocut.direct`).

use serde_json::{Value, json};

use crate::args::Args;
use crate::{Backend, absolute, diag, fail, print, usage};

pub const USAGE: &str = "\
direct [folder | files… | project.fcproj] [--brief \"a 30 s reel of our trip\"]
       [--aspect 9:16|16:9|1:1|4:5] [--seconds 30] [--platform reels|shorts|tiktok|vertical]
       [--insta360 VID_093,VID_096] [--album TITLE] [--mood happy] [--bpm 110]
       [--music file|generate] [--title T] [--no-captions] [--caption-style clean]
       [--redact faces,plates|faces|plates|off] [--name N] [--full-detail 8] [--seed N]
       [--style auto|subtle|energetic|cinematic] [--focus auto|places|people] [--light-effects] [--sounds swish,chime|none]
       [--sound-level 0.4] [--no-color] [--color-strength 0.7] [--no-cutouts] [--no-denoise] [--no-depth] [--no-upscale]
       [--look none|punchy|soft|warm|cool|golden|teal] [--look-strength 25]
       [--revise \"1:10 is too long; warmer at the end\" --from out.report.json]
       [--dry-run] [--no-qa] [--export out.mp4] [--download]
       [--save | --save-as p.fcproj]";

/// `direct`'s options as `autocut.direct` parameters (with `wait: true`); a usage error says
/// what is wrong.
pub fn params(a: &Args) -> Result<Value, String> {
    let mut p = json!({"wait": true});
    for (opt, key) in [
        ("--brief", "brief"),
        ("--aspect", "aspect"),
        ("--platform", "platform"),
        ("--album", "album"),
        ("--mood", "mood"),
        ("--title", "title"),
        ("--caption-style", "captionStyle"),
        ("--redact", "redact"),
        ("--name", "name"),
        ("--style", "style"),
        ("--focus", "focus"),
        ("--look", "look"),
        ("--revise", "revise"),
    ] {
        if let Some(v) = a.opt(opt) {
            p[key] = json!(v);
        }
    }
    if let Some(v) = a.opt("--seconds") {
        let s = v.parse::<f64>().ok().filter(|f| f.is_finite() && (6.0..=300.0).contains(f)).ok_or_else(|| format!("--seconds wants 6 to 300, not `{v}`"))?;
        p["seconds"] = json!(s);
    }
    if let Some(v) = a.opt("--bpm") {
        let b = v.parse::<f64>().ok().filter(|f| f.is_finite() && (70.0..=150.0).contains(f)).ok_or_else(|| format!("--bpm wants 70 to 150, not `{v}`"))?;
        p["bpm"] = json!(b);
    }
    if let Some(v) = a.opt("--full-detail") {
        let n = v.parse::<u64>().ok().filter(|n| *n <= 64).ok_or_else(|| format!("--full-detail wants 0 to 64 shots, not `{v}`"))?;
        p["fullDetail"] = json!(n);
    }
    if let Some(v) = a.opt("--seed") {
        p["seed"] = json!(v.parse::<u64>().map_err(|_| format!("--seed wants a whole number, not `{v}`"))?);
    }
    if let Some(v) = a.opt("--sounds") {
        let kinds: Vec<&str> = if v.trim().eq_ignore_ascii_case("none") { Vec::new() } else { v.split(',').map(str::trim).filter(|k| !k.is_empty()).collect() };
        p["sounds"] = json!(kinds);
    }
    for (opt, key, max) in [("--color-strength", "colorStrength", 1.0), ("--look-strength", "lookStrength", 100.0)] {
        if let Some(v) = a.opt(opt) {
            let x = v.parse::<f64>().ok().filter(|f| f.is_finite() && (0.0..=max).contains(f)).ok_or_else(|| format!("{opt} wants 0 to {max}, not `{v}`"))?;
            p[key] = json!(x);
        }
    }
    if let Some(v) = a.opt("--sound-level") {
        let l = v.parse::<f64>().ok().filter(|f| f.is_finite() && (0.0..=1.0).contains(f)).ok_or_else(|| format!("--sound-level wants 0 to 1, not `{v}`"))?;
        p["soundLevel"] = json!(l);
    }
    if let Some(v) = a.opt("--insta360") {
        let names: Vec<&str> = v.split(',').map(str::trim).filter(|n| !n.is_empty()).collect();
        if names.is_empty() {
            return Err("--insta360 names recordings: VID_093,VID_096".into());
        }
        p["insta360"] = json!(names);
    }
    if let Some(m) = a.opt("--music") {
        p["music"] = json!(if m.eq_ignore_ascii_case("generate") { m.to_string() } else { absolute(m) });
    }
    if let Some(f) = a.opt("--from") {
        p["previous"] = json!(absolute(f));
    }
    if p.get("revise").is_some() != p.get("previous").is_some() {
        return Err("--revise takes the notes and --from the report of the edit they are about".into());
    }
    if let Some(e) = a.opt("--export") {
        p["export"] = json!(absolute(e));
    }
    for (flag, key, v) in [
        ("--no-captions", "captions", false),
        ("--dry-run", "dryRun", true),
        ("--no-qa", "qa", false),
        ("--download", "download", true),
        ("--light-effects", "lightEffects", true),
        ("--no-color", "color", false),
        ("--no-cutouts", "cutouts", false),
        ("--no-denoise", "denoise", false),
        ("--no-depth", "depth", false),
        ("--no-upscale", "upscale", false),
    ] {
        if a.flag(flag) {
            p[key] = json!(v);
        }
    }
    Ok(p)
}

pub async fn run(a: &Args) {
    let sources: Vec<String> = a.positionals.get(1..).map(<[String]>::to_vec).unwrap_or_default();
    let allowed = [
        "--brief",
        "--aspect",
        "--seconds",
        "--platform",
        "--insta360",
        "--album",
        "--mood",
        "--bpm",
        "--music",
        "--title",
        "--no-captions",
        "--caption-style",
        "--redact",
        "--name",
        "--full-detail",
        "--seed",
        "--style",
        "--focus",
        "--light-effects",
        "--sounds",
        "--sound-level",
        "--no-color",
        "--color-strength",
        "--no-cutouts",
        "--no-denoise",
        "--no-depth",
        "--no-upscale",
        "--look",
        "--look-strength",
        "--revise",
        "--from",
        "--dry-run",
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
        usage(format!("direct: unknown option {o}\n{USAGE}"));
    }
    let mut p = params(a).unwrap_or_else(|e| usage(format!("direct: {e}\n{USAGE}")));
    let project = sources.iter().find(|s| std::path::Path::new(s).extension().is_some_and(|x| x.eq_ignore_ascii_case("fcproj"))).cloned();
    let files: Vec<String> = sources.iter().filter(|s| Some(*s) != project.as_ref()).map(|s| absolute(s)).collect();
    let elsewhere = a.opt("--insta360").is_some()
        || a.opt("--album").is_some()
        || a.opt("--project").is_some()
        || a.opt("--bridge").is_some()
        || a.opt("--revise").is_some();
    if sources.is_empty() && !elsewhere {
        usage(format!("direct: what to make it of? a folder, files, a project, --insta360 recordings or an --album\n{USAGE}"));
    }
    if a.opt("--bridge").is_none() {
        for f in &files {
            if !std::path::Path::new(f).exists() {
                fail(format!("direct: {f}: no such file or folder"));
            }
        }
    }
    let mut b = Backend::open(a);
    if let Some(proj) = &project {
        b.exec("file.open", json!({"path": absolute(proj)})).await.unwrap_or_else(|e| fail(format!("direct: {proj}: {e}")));
    }
    if !files.is_empty() {
        p["paths"] = json!(files);
    }
    diag(format_args!("direct: indexing, listening, looking round the 360° shots and directing (this takes a while the first time)"));
    match b.exec("autocut.direct", p).await {
        Ok(r) => {
            let made = &r["showcase"];
            let plan = &made["plan"];
            diag(format_args!(
                "direct: \"{}\" ({}, {} bpm) directed by {}",
                plan["title"].as_str().unwrap_or("?"),
                plan["mood"].as_str().unwrap_or("?"),
                plan["bpm"].as_f64().unwrap_or(0.0),
                made["director"].as_str().unwrap_or("?")
            ));
            for s in plan["shots"].as_array().map(Vec::as_slice).unwrap_or_default().iter().take(MAX_LISTED) {
                diag(format_args!(
                    "direct:   {} {} beats, {}{}{}{} — {}",
                    s["clip"].as_str().unwrap_or("?"),
                    s["beats"],
                    s["camera"].as_str().unwrap_or("?"),
                    s["look"].as_str().map(|l| format!(" at {l}")).unwrap_or_default(),
                    s["to"].as_str().map(|l| format!(" to {l}")).unwrap_or_default(),
                    s["text"].as_str().map(|t| format!(" \"{t}\"")).unwrap_or_default(),
                    s["cut"].as_str().unwrap_or("a cut")
                ));
            }
            for c in made["changes"].as_array().map(Vec::as_slice).unwrap_or_default().iter().take(MAX_LISTED) {
                diag(format_args!("direct: changed: {}", c.as_str().unwrap_or_default()));
            }
            for m in made["moments"].as_array().map(Vec::as_slice).unwrap_or_default().iter().filter(|m| m["kept"] == false).take(MAX_LISTED) {
                diag(format_args!(
                    "direct: left out {} ({}): {}",
                    m["clip"].as_str().unwrap_or("?"),
                    m["name"].as_str().unwrap_or("?"),
                    m["why"].as_str().unwrap_or_default()
                ));
            }
            for n in plan["notes"].as_array().map(Vec::as_slice).unwrap_or_default().iter().take(30) {
                diag(format_args!("direct: repaired: {}", n.as_str().unwrap_or_default()));
            }
            for c in made["color"]["clips"].as_array().map(Vec::as_slice).unwrap_or_default().iter().take(MAX_LISTED) {
                let k = &c["applied"];
                diag(format_args!(
                    "direct: colour: clip {}: temperature {}, tint {}, exposure {}, saturation {}, contrast {}{}",
                    c["clip"],
                    k["temperature"],
                    k["tint"],
                    k["exposure"],
                    k["saturation"],
                    k["contrast"],
                    c["backedOff"].as_str().map(|w| format!(" (backed off: {w})")).unwrap_or_default()
                ));
            }
            for f in made["fallbacks"].as_array().map(Vec::as_slice).unwrap_or_default() {
                diag(format_args!("direct: fallback: {}", f.as_str().unwrap_or_default()));
            }
            for w in made["warnings"].as_array().map(Vec::as_slice).unwrap_or_default().iter().take(30) {
                diag(format_args!("direct: note: {}", w.as_str().unwrap_or_default()));
            }
            diag(format_args!("direct: {}", made["status"].as_str().unwrap_or("done")));
            print(a, made);
            b.finish(a).await;
        }
        Err(e) => fail(format!("direct: {e}")),
    }
}

/// Shots listed on the terminal, at most.
const MAX_LISTED: usize = 64;

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Args {
        Args::parse(v.iter().map(|s| s.to_string()))
    }

    #[test]
    fn options_become_parameters() {
        let p = params(&args(&[
            "direct",
            "clips/",
            "--brief",
            "a 45 s recap",
            "--aspect",
            "16:9",
            "--seconds",
            "45",
            "--insta360",
            "VID_093, VID_096",
            "--bpm",
            "100",
            "--no-captions",
            "--dry-run",
            "--seed",
            "3",
        ]))
        .unwrap();
        assert_eq!((p["brief"].as_str(), p["aspect"].as_str(), p["seconds"].as_f64()), (Some("a 45 s recap"), Some("16:9"), Some(45.0)));
        assert_eq!(p["insta360"], json!(["VID_093", "VID_096"]));
        assert_eq!((p["bpm"].as_f64(), p["captions"].as_bool(), p["dryRun"].as_bool(), p["seed"].as_u64()), (Some(100.0), Some(false), Some(true), Some(3)));
        assert_eq!(p["wait"], json!(true));
        // the cutting style
        let p = params(&args(&["direct", "x", "--style", "energetic", "--sounds", "whoosh,impact", "--sound-level", "0.6", "--light-effects"])).unwrap();
        assert_eq!(
            (p["style"].as_str(), p["sounds"].clone(), p["soundLevel"].as_f64(), p["lightEffects"].as_bool()),
            (Some("energetic"), json!(["whoosh", "impact"]), Some(0.6), Some(true))
        );
        assert_eq!(params(&args(&["direct", "x", "--sounds", "none"])).unwrap()["sounds"], json!([]));
        assert!(params(&args(&["direct", "x", "--sound-level", "3"])).is_err());
        // a revision: the notes and the report they are about, both or neither
        let p = params(&args(&["direct", "--revise", "warmer at the end", "--from", "/tmp/r.report.json"])).unwrap();
        assert_eq!((p["revise"].as_str(), p["previous"].as_str()), (Some("warmer at the end"), Some("/tmp/r.report.json")));
        assert!(params(&args(&["direct", "--revise", "warmer"])).is_err());
        // the colour pass
        let p = params(&args(&["direct", "x", "--no-color", "--look", "warm", "--look-strength", "20", "--color-strength", "0.5"])).unwrap();
        assert_eq!(
            (p["color"].as_bool(), p["look"].as_str(), p["lookStrength"].as_f64(), p["colorStrength"].as_f64()),
            (Some(false), Some("warm"), Some(20.0), Some(0.5))
        );
        assert!(params(&args(&["direct", "x", "--color-strength", "1.5"])).is_err());
        // the brief is a value, not a file
        let a = args(&["direct", "clip.mp4", "--brief", "a 30 s reel", "--full-detail", "4"]);
        assert_eq!(a.positionals, vec!["direct".to_string(), "clip.mp4".to_string()]);
        assert_eq!(params(&a).unwrap()["fullDetail"], json!(4));
        for bad in [
            &["direct", "x", "--seconds", "2"][..],
            &["direct", "x", "--seconds", "NaN"],
            &["direct", "x", "--bpm", "300"],
            &["direct", "x", "--insta360", " , "],
            &["direct", "x", "--seed", "-1"],
        ] {
            assert!(params(&args(bad)).is_err(), "{bad:?}");
        }
    }
}
