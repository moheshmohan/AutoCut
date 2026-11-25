//! `script.factLedger`: figures and claims on screen against a ledger with sources, from a
//! sequence's titles, a script and a ledger file; its Floor leg in `delivery.preflight`; hostile
//! parameters refused.
use filmcraft_time::Tick;
use serde_json::{Value, json};

use super::*;
use crate::Session;

fn secs(s: f64) -> i64 {
    Tick::from_seconds_f64(s).0
}

fn codes(r: &Value) -> Vec<(String, String)> {
    r["findings"].as_array().unwrap().iter().map(|f| (f["code"].as_str().unwrap().to_string(), f["text"].as_str().unwrap_or("").to_string())).collect()
}

/// A 640×360 sequence with four titles, one after the other.
fn titled() -> Session {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"width": 640, "height": 360, "fps": 30})).unwrap();
    for (i, t) in ["1903: the first powered flight", "It weighed 274 kg", "Only 5 people watched", "The fastest plane of its day"].iter().enumerate() {
        s.execute("graphics.newText", json!({"text": t, "size": 40, "seconds": 2.0, "time": secs(i as f64 * 2.0), "track": "V1"})).unwrap();
    }
    s
}

fn ledger() -> Value {
    json!([
        {"text": "The Wright brothers made the first powered flight in 1903; it lasted 12 seconds", "source": "https://example.org/first-flight"},
        274
    ])
}

#[test]
fn a_figure_not_in_the_sources_is_flagged_and_one_that_is_passes() {
    let mut s = titled();
    let r = s.execute(CMD, json!({"facts": ledger()})).unwrap();
    let found = codes(&r);
    // 1903 and the first flight are in the ledger with a source: nothing said
    assert!(!found.iter().any(|(_, t)| t.starts_with("1903")), "{found:?}");
    // 274 is in the ledger without a source; 5 is not in it; two claims nobody sourced
    assert!(found.contains(&("unsourcedFigure".into(), "It weighed 274 kg".into())), "{found:?}");
    assert!(found.contains(&("unledgeredFigure".into(), "Only 5 people watched".into())), "{found:?}");
    assert!(found.contains(&("unledgeredClaim".into(), "Only 5 people watched".into())), "{found:?}");
    assert!(found.contains(&("unledgeredClaim".into(), "The fastest plane of its day".into())), "{found:?}");
    assert_eq!(found.len(), 4, "{r:#}");
    assert_eq!(r["blocking"], false);
    assert_eq!((r["ledger"]["sourced"].as_u64(), r["ledger"]["unsourced"].as_u64()), (Some(1), Some(1)));
    // every finding says where it is
    assert!(r["findings"].as_array().unwrap().iter().all(|f| f["time"].is_i64() && f["clip"].is_u64()));
    // a source for 274 and a sourced claim for the plane, and only the 5 is left
    let more = json!([
        {"text": "The Wright brothers made the first powered flight in 1903", "source": "https://example.org/first-flight"},
        {"text": "It weighed 274 kg with its pilot", "sources": ["a museum's catalogue, page 12"]},
        {"text": "The fastest plane of its day, at 48 km/h", "source": "a note from the archive"}
    ]);
    let r = s.execute(CMD, json!({"facts": more})).unwrap();
    let found = codes(&r);
    assert_eq!(
        found,
        vec![("unledgeredFigure".to_string(), "Only 5 people watched".to_string()), ("unledgeredClaim".into(), "Only 5 people watched".into())],
        "{r:#}"
    );
    // claims: false checks the figures alone
    let r = s.execute(CMD, json!({"facts": more, "claims": false})).unwrap();
    assert_eq!(codes(&r).len(), 1);
    // copy.lint reads the same ledger, sources and all
    let c = s.execute("copy.lint", json!({"sequence": s.state.active_sequence.unwrap().0, "facts": more})).unwrap();
    assert_eq!(c["findings"].as_array().unwrap().iter().filter(|f| f["code"] == "unledgeredFigure").count(), 1, "{c:#}");
}

#[test]
fn a_script_is_checked_before_it_is_built() {
    let mut s = Session::default();
    let script = "# 1903\n[shot: 12 seconds in the air]\nThey flew for the first time.\n\n# After\nBy 1905 they flew for 39 minutes.\n";
    let facts = json!([{"text": "On 17 December 1903 they flew for 12 seconds, the first time", "source": "https://example.org/a"}]);
    let r = s.execute(CMD, json!({"text": script, "facts": facts})).unwrap();
    let found = codes(&r);
    // the chapter title and the hint pass; 1905 and 39 are not in the ledger
    let figures: Vec<&str> =
        r["findings"].as_array().unwrap().iter().filter(|f| f["code"] == "unledgeredFigure").map(|f| f["figure"].as_str().unwrap()).collect();
    assert_eq!(figures, vec!["1905", "39"], "{found:?}");
    assert!(!found.iter().any(|(c, _)| c.ends_with("Claim")), "the first time is in the ledger: {found:?}");
    assert_eq!(r["sequence"], Value::Null);
    // bad markup is refused as script.timeline refuses it
    let e = s.execute(CMD, json!({"text": "# Empty\n\n# Full\nWords.\n", "facts": facts})).unwrap_err().to_string();
    assert!(e.contains("no paragraphs"), "{e}");
}

#[test]
fn the_ledger_file_beside_the_project_carries_the_sources() {
    let dir = std::env::temp_dir().join(format!("fc-facts-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("build")).unwrap();
    let mut s = titled();
    s.execute("file.saveAs", json!({"path": dir.join("x.fcproj").to_string_lossy()})).unwrap();
    std::fs::write(dir.join("build").join("facts.json"), json!({"facts": ledger()}).to_string()).unwrap();
    let r = s.execute(CMD, json!({})).unwrap();
    assert!(r["ledger"]["from"].as_str().unwrap().ends_with("facts.json"));
    assert_eq!(codes(&r).len(), 4);
    // the delivery gate: a Floor check that warns this round and blocks with floorBlocks
    let quick = json!({"privacy": false, "narration": false, "mix": false, "motionEnergy": false, "craft": false, "motionCraft": false});
    let d = s.execute("delivery.preflight", quick.clone()).unwrap();
    let check = d["checks"].as_array().unwrap().iter().find(|c| c["check"] == "factSources").unwrap().clone();
    assert_eq!((check["state"].as_str(), check["tier"].as_str()), (Some("warning"), Some("floor")), "{check:#}");
    let leg: Vec<&str> = check["detail"]["findings"].as_array().unwrap().iter().map(|f| f["code"].as_str().unwrap()).collect();
    assert!(leg.contains(&"unsourcedFigure") && leg.contains(&"unledgeredClaim") && !leg.contains(&"unledgeredFigure"), "{leg:?}");
    let mut blocking = quick.clone();
    blocking["floorBlocks"] = json!(true);
    let d = s.execute("delivery.preflight", blocking).unwrap();
    assert!(d["blocking"].as_array().unwrap().contains(&json!("factSources")), "{:#}", d["blocking"]);
    // a ledger without sources leaves it to `figures`; factSources: false turns it off
    std::fs::write(dir.join("build").join("facts.json"), json!(["1903", 274, 5]).to_string()).unwrap();
    let d = s.execute("delivery.preflight", quick.clone()).unwrap();
    let check = d["checks"].as_array().unwrap().iter().find(|c| c["check"] == "factSources").unwrap();
    assert_eq!(check["state"], "skipped");
    let mut off = quick;
    off["factSources"] = json!(false);
    let d = s.execute("delivery.preflight", off).unwrap();
    assert_eq!(d["checks"].as_array().unwrap().iter().find(|c| c["check"] == "factSources").unwrap()["detail"], "disabled by explicit project policy");
    // a broken ledger file is an error, not a guess
    std::fs::write(dir.join("build").join("facts.json"), "{not json").unwrap();
    assert!(s.execute(CMD, json!({})).unwrap_err().to_string().contains("could not be read"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hostile_parameters_are_refused_without_a_panic() {
    let mut s = titled();
    let many_sources: Vec<String> = (0..20).map(|i| format!("s{i}")).collect();
    let cases: Vec<(Value, &str)> = vec![
        (json!({}), "no fact ledger"),
        (json!({"facts": "1903"}), "list"),
        (json!({"facts": [{"text": "x", "source": 7}]}), "source is a text"),
        (json!({"facts": [{"text": "x", "sources": many_sources}]}), "sources"),
        (json!({"facts": [{"source": "https://example.org"}]}), "needs a `text`"),
        (json!({"facts": [[1]]}), "ledger entry"),
        (json!({"facts": [1], "claims": "yes"}), "true or false"),
        (json!({"facts": [1], "text": 5}), "string"),
        (json!({"facts": [1], "text": "x".repeat(70_000)}), "bytes"),
        (json!({"facts": [1], "sequence": "one"}), "sequence id"),
        (json!({"facts": [1], "sequence": 999_999}), "no such sequence"),
        (json!({"facts": [1], "bias": 1}), "unknown parameter"),
        (json!("facts"), "object"),
    ];
    for (p, want) in cases {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute(CMD, p.clone())));
        let e = r.unwrap_or_else(|_| panic!("panicked on {p}")).err().unwrap_or_else(|| panic!("accepted {p}")).to_string();
        assert!(e.contains(want), "{p}: {e}");
    }
    // no sequence and no script
    let mut empty = Session::default();
    assert!(empty.execute(CMD, json!({"facts": [1]})).unwrap_err().to_string().contains("open a sequence"));
}

#[test]
fn claims_are_matched_in_the_words_the_screen_uses() {
    let fact = |t: &str| Fact { text: t.into(), values: Vec::new(), sources: vec!["x".into()] };
    let l = Ledger {
        from: "facts".into(),
        entries: 2,
        values: Vec::new(),
        facts: vec![fact("The first powered flights were short"), fact("Ice is the lightest solid")],
    };
    let (c, w) = claim_parts("The first flight");
    assert_eq!(c, vec!["first"]);
    assert_eq!(claim_matches(&l, &c, &w).len(), 1, "flight / flights share a stem");
    let (c, w) = claim_parts("The first rocket launch");
    assert!(claim_matches(&l, &c, &w).is_empty());
    let (c, w) = claim_parts("FIRST!");
    assert_eq!(claim_matches(&l, &c, &w).len(), 1, "a lone claim word needs only itself");
    let (c, _) = claim_parts("Flying over the hills");
    assert!(c.is_empty(), "no claim word, no claim");
}
