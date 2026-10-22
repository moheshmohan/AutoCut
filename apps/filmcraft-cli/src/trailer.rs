//! `trailer`: a vertical trailer of a finished showcase (`autocut.trailer`, see
//! `filmcraft_autocut::trailer`), run as the engine job in the CLI's session (or the app's over
//! `--bridge`) with its progress on stderr.

use serde_json::{Value, json};

use crate::args::Args;
use crate::{Backend, absolute, diag, fail, print, usage};

const USAGE: &str = "trailer (--project p.fcproj | --bridge addr) [--sequence id] [--plan plan.json] [--report build.json] [--seconds 34] [--aspect 9:16|1:1|4:5] [--platform reels|youtube|tiktok|vertical] [--robots N] [--hook LABEL [--hook-at SECONDS]] [--at LABEL=[ITEM@]SECONDS]... [--title \"48 robots · 2013 → 2025\"] [--no-hook-voice] [--logo logo.png] [--channel @handle] [--end-text \"Full video on YouTube\"] [--link URL] [--exclude LABEL]... [--include LABEL]... [--bpm 124] [--mood energetic] [--seed 7] [--redact faces,plates|off] [--fallback blur|letterbox|center] [--export out.mp4] [--preset P] [--no-qa] [--look tech|vhs|clean] [--beat-pulse] [--save]";

const OPTIONS: &[&str] = &[
    "--project",
    "--bridge",
    "--data-dir",
    "--save",
    "--save-as",
    "--json",
    "--compact",
    "--sequence",
    "--plan",
    "--report",
    "--seconds",
    "--aspect",
    "--platform",
    "--robots",
    "--hook",
    "--hook-at",
    "--at",
    "--title",
    "--hook-voice",
    "--no-hook-voice",
    "--logo",
    "--channel",
    "--end-text",
    "--link",
    "--exclude",
    "--include",
    "--bpm",
    "--mood",
    "--seed",
    "--redact",
    "--fallback",
    "--export",
    "--preset",
    "--qa",
    "--no-qa",
    "--look",
    "--beat-pulse",
];

/// `trailer` options as `autocut.trailer` parameters (the job is followed, not waited on); a
/// usage error says what is wrong. Paths are made absolute.
pub fn trailer_params(a: &Args) -> Result<Value, String> {
    if let Some(o) = a.unknown_opt(OPTIONS) {
        return Err(format!("unknown option {o}"));
    }
    let mut p = json!({"wait": false});
    let num = |k: &str| -> Result<Option<f64>, String> {
        a.opt(k).map(|v| v.parse::<f64>().ok().filter(|f| f.is_finite()).ok_or_else(|| format!("{k} wants a number, not `{v}`"))).transpose()
    };
    let whole =
        |k: &str| -> Result<Option<u64>, String> { a.opt(k).map(|v| v.parse::<u64>().map_err(|_| format!("{k} wants a whole number, not `{v}`"))).transpose() };
    if let Some(q) = whole("--sequence")? {
        p["sequence"] = json!(q);
    }
    for (flag, key) in [("--plan", "planPath"), ("--report", "reportPath"), ("--logo", "logo"), ("--export", "export")] {
        if let Some(v) = a.opt(flag) {
            p[key] = json!(absolute(v));
        }
    }
    for (flag, key) in [("--seconds", "seconds"), ("--bpm", "bpm"), ("--hook-at", "hookAt")] {
        if let Some(v) = num(flag)? {
            p[key] = json!(v);
        }
    }
    for (flag, key) in [("--robots", "robots"), ("--seed", "seed")] {
        if let Some(v) = whole(flag)? {
            p[key] = json!(v);
        }
    }
    for (flag, key) in [
        ("--aspect", "aspect"),
        ("--hook", "hook"),
        ("--title", "title"),
        ("--channel", "channel"),
        ("--end-text", "endText"),
        ("--link", "link"),
        ("--mood", "mood"),
        ("--preset", "preset"),
    ] {
        if let Some(v) = a.opt(flag) {
            p[key] = json!(v);
        }
    }
    if let Some(pl) = a.opt("--platform") {
        let pl =
            filmcraft_engine::safe_zones::Platform::from_name(pl).ok_or_else(|| format!("--platform is reels, youtube, tiktok or vertical, not `{pl}`"))?;
        p["platform"] = json!(pl.name());
    }
    if let Some(f) = a.opt("--fallback") {
        let f = filmcraft_render::reframe::Fallback::from_name(f).ok_or_else(|| format!("--fallback is blur, letterbox or center, not `{f}`"))?;
        p["fallback"] = json!(f.name());
    }
    if let Some(r) = a.opt("--redact") {
        p["redact"] = match r.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "no" | "false" => json!(false),
            other => json!(other),
        };
    }
    match (a.flag("--hook-voice"), a.flag("--no-hook-voice")) {
        (true, true) => return Err("--hook-voice or --no-hook-voice, not both".into()),
        (true, false) => p["hookVoice"] = json!(true),
        (false, true) => p["hookVoice"] = json!(false),
        _ => {}
    }
    // opt-in creator tools: a look over the hook and the montage, zoom pulses on the beats
    if let Some(l) = a.opt("--look") {
        let l = l.trim().to_ascii_lowercase();
        if !matches!(l.as_str(), "tech" | "vhs" | "clean") {
            return Err(format!("--look is tech, vhs or clean, not `{l}`"));
        }
        p["look"] = json!(l);
    }
    if a.flag("--beat-pulse") {
        p["beatPulse"] = json!(true);
    }
    match (a.flag("--qa"), a.flag("--no-qa")) {
        (true, true) => return Err("--qa or --no-qa, not both".into()),
        (true, false) => p["qa"] = json!(true),
        (false, true) => p["qa"] = json!(false),
        _ => {}
    }
    // where robots' shots start: LABEL=SECONDS, or LABEL=ITEM@SECONDS for another clip of it
    let mut at = serde_json::Map::new();
    for v in a.all("--at") {
        let (label, spec) = v.rsplit_once('=').ok_or_else(|| format!("--at wants LABEL=SECONDS or LABEL=ITEM@SECONDS, not `{v}`"))?;
        let bad = || format!("--at wants LABEL=SECONDS or LABEL=ITEM@SECONDS, not `{v}`");
        let secs = |s: &str| s.trim().parse::<f64>().ok().filter(|f| f.is_finite() && *f >= 0.0).ok_or_else(bad);
        let value = match spec.split_once('@') {
            Some((item, s)) => json!({"item": item.trim().parse::<u64>().map_err(|_| bad())?, "seconds": secs(s)?}),
            None => json!(secs(spec)?),
        };
        at.insert(label.trim().to_string(), value);
    }
    if !at.is_empty() {
        p["at"] = Value::Object(at);
    }
    for (flag, key) in [("--exclude", "exclude"), ("--include", "include")] {
        let list: Vec<&str> = a.all(flag);
        if !list.is_empty() {
            p[key] = json!(list);
        }
    }
    Ok(p)
}

/// Run `autocut.trailer` with `params` on `b`, its progress on stderr (in the CLI's own session
/// nothing else runs the job's commands: it runs them as the app's UI does). The job's outcome.
pub async fn trailer_job(b: &mut Backend, params: Value) -> Result<Value, String> {
    let start = b.exec("autocut.trailer", params).await?;
    let job = start["job"].as_u64().ok_or_else(|| format!("autocut.trailer started no job: {start}"))?;
    let mut last = String::new();
    let entry = loop {
        if let Backend::Local(s) = b {
            s.pump_port(std::time::Duration::from_millis(50));
        }
        let jobs = b.exec("jobs.list", json!({})).await?;
        let j = jobs.as_array().and_then(|l| l.iter().find(|j| j["id"].as_u64() == Some(job))).cloned().ok_or_else(|| format!("job {job} is gone"))?;
        let failed = j["result"]["error"].is_string();
        let status = j["status"].as_str().unwrap_or_default();
        if !failed && !status.is_empty() && status != last {
            diag(format_args!("trailer: {status}"));
            status.clone_into(&mut last);
        }
        if j["finished"].as_bool() == Some(true) {
            break j;
        }
        let pause = if matches!(b, Backend::Local(_)) { 10 } else { 250 };
        tokio::time::sleep(std::time::Duration::from_millis(pause)).await;
    };
    if let Some(e) = entry["result"]["error"].as_str() {
        return Err(e.to_string());
    }
    Ok(entry["showcase"].clone())
}

/// `trailer …`: the job, then what it made.
pub async fn run(a: &Args) {
    let params = trailer_params(a).unwrap_or_else(|e| usage(format!("trailer: {e}\n{USAGE}")));
    if a.opt("--project").is_none() && a.opt("--bridge").is_none() {
        usage(format!("trailer: open the showcase's project (--project) or the running app (--bridge)\n{USAGE}"));
    }
    let mut b = Backend::open(a);
    let made = trailer_job(&mut b, params).await.unwrap_or_else(|e| fail(format!("trailer: {e}")));
    for s in made["shots"].as_array().map(Vec::as_slice).unwrap_or_default() {
        diag(format_args!(
            "trailer: {:>5.2}–{:>5.2} s  {}{}  ({:.1}–{:.1} s of item {}{})",
            s["start"].as_f64().unwrap_or(0.0),
            s["end"].as_f64().unwrap_or(0.0),
            s["label"].as_str().unwrap_or_default(),
            s["year"].as_i64().map(|y| format!(" · {y}")).unwrap_or_default(),
            s["from"].as_f64().unwrap_or(0.0),
            s["to"].as_f64().unwrap_or(0.0),
            s["item"],
            if s["followed"] == json!(false) { ", whole over its blurred copy" } else { "" },
        ));
    }
    for k in ["cover", "caption", "frames"] {
        if let Some(p) = made[k].as_str() {
            diag(format_args!("trailer: {k}: {p}"));
        }
    }
    if let Some(q) = made["qaSummary"].as_str() {
        diag(format_args!("trailer: QA: {q}"));
    }
    for n in made["notes"].as_array().map(Vec::as_slice).unwrap_or_default() {
        diag(format_args!("trailer: note: {}", n.as_str().unwrap_or_default()));
    }
    if let Some(s) = made["status"].as_str() {
        diag(format_args!("trailer: {s}"));
    }
    print(a, &made);
    b.finish(a).await;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use filmcraft_autocut::host::{Models, ShowcaseHost};
    use serde_json::json;

    use super::*;

    fn args(v: &[&str]) -> Args {
        Args::parse(v.iter().map(|s| s.to_string()))
    }

    /// `trailer` options become `autocut.trailer` parameters; bad values are usage errors.
    #[test]
    fn trailer_options_become_job_params() {
        let p = trailer_params(&args(&[
            "trailer",
            "--project",
            "p.fcproj",
            "--sequence",
            "9755",
            "--plan",
            "/v5/plan-v5-ml.json",
            "--report",
            "/v5/build.json",
            "--seconds",
            "34",
            "--robots",
            "14",
            "--hook",
            "Ameca, again",
            "--exclude",
            "Promobot",
            "--exclude",
            "Pepper, again",
            "--logo",
            "/b/logo.png",
            "--channel",
            "@H4hacks",
            "--platform",
            "Instagram",
            "--redact",
            "off",
            "--no-hook-voice",
            "--at",
            "My first robot=1347@9.5",
            "--at",
            "NAO=3",
            "--export",
            "/m/Reel.mp4",
        ]))
        .unwrap();
        assert_eq!(
            p,
            json!({"wait": false, "sequence": 9755, "planPath": "/v5/plan-v5-ml.json", "reportPath": "/v5/build.json", "seconds": 34.0, "robots": 14,
                   "hook": "Ameca, again", "exclude": ["Promobot", "Pepper, again"], "logo": "/b/logo.png", "channel": "@H4hacks", "platform": "reels",
                   "redact": false, "hookVoice": false, "export": "/m/Reel.mp4", "at": {"My first robot": {"item": 1347, "seconds": 9.5}, "NAO": 3.0}})
        );
        for bad in [
            &["trailer", "--seconds", "long"][..],
            &["trailer", "--seconds", "NaN"],
            &["trailer", "--robots", "-3"],
            &["trailer", "--sequence", "main"],
            &["trailer", "--platform", "snapchat"],
            &["trailer", "--fallback", "stretch"],
            &["trailer", "--qa", "--no-qa"],
            &["trailer", "--hook-voice", "--no-hook-voice"],
            &["trailer", "--colour", "red"],
            &["trailer", "--at", "NAO"],
            &["trailer", "--at", "NAO=x@3"],
            &["trailer", "--at", "NAO=-1"],
            &["trailer", "--look", "sparkly"],
        ] {
            assert!(trailer_params(&args(bad)).is_err(), "{bad:?}");
        }
    }

    /// The CLI runs the trailer job in its own session, running the job's commands itself: a
    /// plan of the demo footage (no build report: the shots' windows come from the plan) gives a
    /// 9:16 sequence; a plan without robots is an error, not a crash.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_trailer_job_runs_in_the_cli_session() {
        let mut s = filmcraft_engine::Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        s.autocut.host = Some(Arc::new(ShowcaseHost::with_models(Models::scripted())));
        let mut b = Backend::Local(Box::new(s));
        let entry = |n: u64, label: &str, kind: &str, year: i64, item: u64, len: f64| {
            json!({"number": n, "label": label, "kind": kind, "description": "", "unix": (year - 1970) * 31_557_600,
                   "shots": [{"item": item, "name": format!("{item}.mp4"), "clip_seconds": len, "best": len / 2.0,
                              "largest": {"seconds": len / 2.0, "area": 0.3, "center": [0.5, 0.5]}}]})
        };
        let plan = json!({"title": "Robots", "entries": [
            entry(1, "Rover", "wheeled", 2014, 5, 18.0), entry(2, "Walker", "humanoid", 2017, 6, 16.0), entry(3, "Dog", "quadruped", 2020, 7, 14.0),
            entry(4, "Arm", "arm", 2023, 8, 15.0), entry(5, "Greeter", "humanoid", 2025, 9, 12.0)
        ], "script": {"intro": "", "lines": [], "outro": ""}});
        let made = trailer_job(&mut b, json!({"plan": plan.clone(), "seconds": 14, "wait": false})).await.unwrap();
        assert_eq!(made["frame"], json!([1080, 1920]), "{made}");
        assert!(made["shots"].as_array().is_some_and(|s| s.len() >= 3), "{made}");
        assert_eq!(made["cutsOnBeat"], made["cuts"]);
        assert!(made.get("look").is_none() && made.get("beatPulses").is_none(), "no creator tools unless asked");
        // the opt-in creator look and beat pulses
        let looked = trailer_job(&mut b, json!({"plan": plan, "seconds": 14, "look": "tech", "beatPulse": true, "wait": false})).await.unwrap();
        assert_eq!(looked["look"], json!("tech"), "{looked}");
        assert!(looked["beatPulses"].as_u64().is_some_and(|n| n > 0), "{looked}");
        let Backend::Local(s) = &b else { panic!("local") };
        let q = s.active_sequence().unwrap();
        let hud = q.video_tracks.iter().flat_map(|t| t.items.iter()).find(|i| i.effect("cr_tech_hud").is_some()).expect("the look's adjustment clip");
        assert!(hud.start == filmcraft_time::Tick::ZERO && hud.duration.seconds() > 5.0, "over the hook and the montage");
        let empty = json!({"title": "Robots", "entries": [{"number": 1, "label": "Hall", "kind": "scene", "shots": []}]});
        assert!(trailer_job(&mut b, json!({"plan": empty, "wait": false})).await.unwrap_err().contains("trailer"));
    }

    #[test]
    fn creator_tools_are_opt_in() {
        let p = trailer_params(&args(&["trailer", "--project", "p.fcproj", "--look", "Tech", "--beat-pulse"])).unwrap();
        assert_eq!((p["look"].as_str(), p["beatPulse"].as_bool()), (Some("tech"), Some(true)));
        let d = trailer_params(&args(&["trailer", "--project", "p.fcproj"])).unwrap();
        assert!(d.get("look").is_none() && d.get("beatPulse").is_none(), "off unless asked: the trailer as before");
    }
}
