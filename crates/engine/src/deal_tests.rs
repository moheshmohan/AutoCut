//! `autocut.deal` and `autocut.feedback` against histories written here (each test keeps its
//! data directory of its own; the user's profile and history are never read).
use serde_json::{Value, json};

use super::*;
use crate::Session;
use crate::taste_history::{Axes, FILE, Record};

fn session(tag: &str) -> (Session, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("filmcraft-deal-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    (Session { prefs_path: Some(dir.join("preferences.json")), ..Session::default() }, dir)
}

/// A history of videos that used these looks (oldest first).
fn looks(dir: &std::path::Path, used: &[&str]) {
    let lines: Vec<String> = used
        .iter()
        .map(|l| {
            serde_json::to_string(&Record {
                when: "2026-11-01T00:00:00Z".into(),
                axes: Axes { color_look: (*l).into(), ..Axes::default() },
                ..Record::default()
            })
            .unwrap()
        })
        .collect();
    std::fs::write(dir.join(FILE), format!("{}\n", lines.join("\n"))).unwrap();
}

fn deal_p(s: &mut Session, p: Value) -> Value {
    s.execute(DEAL_CMD, p).unwrap()
}

fn why_of<'a>(r: &'a Value, id: &str) -> Option<&'a str> {
    r["skipped"].as_array().unwrap().iter().find(|x| x["id"] == id).and_then(|x| x["why"].as_str())
}

#[test]
fn consecutive_videos_get_different_looks() {
    let (mut s, dir) = session("rotate");
    // no history: the first of the options
    let r = deal_p(&mut s, json!({"for": "look", "options": ["none", "warm", "cool", "teal"]}));
    assert_eq!((r["choice"].as_str(), r["kept"].as_bool()), (Some("none"), Some(true)), "{r:#}");
    assert_eq!(r["from"], json!(["none", "warm", "cool", "teal"]));
    // the three newest videos used none, warm and cool: teal
    looks(&dir, &["teal", "none", "warm", "cool"]);
    let r = deal_p(&mut s, json!({"for": "look", "options": ["none", "warm", "cool", "teal"]}));
    assert_eq!(r["choice"], "teal", "{r:#}");
    assert_eq!(why_of(&r, "warm"), Some("used in the last 3"));
    // all used lately: the least recently used, and it says so
    looks(&dir, &["cool", "warm", "none"]);
    let r = deal_p(&mut s, json!({"for": "look", "options": ["none", "warm", "cool"], "dryRun": true}));
    assert_eq!(r["choice"], "cool");
    assert!(r["why"].as_str().unwrap().contains("least recently used"), "{r:#}");
    assert_eq!(r["kept"], false);
    // deterministic for a history and a seed; a seed picks among the fresh ones
    let a = deal_p(&mut s, json!({"for": "look", "seed": 5, "dryRun": true}));
    let b = deal_p(&mut s, json!({"for": "look", "seed": 5, "dryRun": true}));
    assert_eq!(a, b);
    assert!(!["cool", "warm", "none"].contains(&a["choice"].as_str().unwrap()), "{a:#}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_taste_profile_and_feedback_win() {
    let (mut s, dir) = session("taste");
    // no whoosh or riser: the families that carry only those are left out; dissolves stay
    s.execute("autocut.taste", json!({"scope": "default", "set": {"sounds": ["swish", "chime"], "lightEffects": false}})).unwrap();
    let r = deal_p(&mut s, json!({"for": "transitionFamily"}));
    assert_eq!(r["choice"], "dissolve");
    assert_eq!(why_of(&r, "zoom"), Some("not allowed by taste"), "{r:#}");
    assert_eq!(why_of(&r, "spatial"), Some("not allowed by taste"));
    assert_eq!(why_of(&r, "light"), Some("not allowed by taste"));
    assert_eq!(why_of(&r, "whip"), None, "a whip takes the swish");
    // a bouncing entrance only with playful motion
    assert_eq!(why_of(&deal_p(&mut s, json!({"for": "titleStyle"})), "drop"), Some("not allowed by taste"));
    // a retired choice is never dealt, even when it is the only fresh one
    let r = s.execute(FEEDBACK_CMD, json!({"about": "colorLook", "id": "Warm", "verdict": "dislike", "note": "too orange"})).unwrap();
    assert_eq!((r["about"].as_str(), r["id"].as_str(), r["scope"].as_str()), (Some("look"), Some("warm"), Some("default")));
    assert_eq!(r["retired"], json!({"look": ["warm"]}));
    assert_eq!(r["taste"]["default"]["reasons"]["look:warm"], "too orange");
    looks(&dir, &["none", "cool"]);
    let r = deal_p(&mut s, json!({"for": "look", "options": ["none", "cool", "warm"]}));
    assert_eq!(r["choice"], "none", "least recently used of the allowed, never the retired: {r:#}");
    assert_eq!(why_of(&r, "warm"), Some("retired"));
    // liked first among the fresh ones
    s.execute(FEEDBACK_CMD, json!({"about": "look", "id": "golden", "verdict": "like"})).unwrap();
    assert_eq!(deal_p(&mut s, json!({"for": "look", "dryRun": true}))["choice"], "golden");
    // a liked id that is disliked later is retired (and no longer liked)
    let r = s.execute(FEEDBACK_CMD, json!({"about": "look", "id": "golden", "verdict": "dislike"})).unwrap();
    assert_eq!(r["retired"], json!({"look": ["warm", "golden"]}));
    assert!(r["liked"].is_null(), "{r:#}");
    // the whoosh lesson: a retired sound is not heard, and families of whooshes go
    s.execute("autocut.taste", json!({"scope": "default", "set": {"sounds": null}})).unwrap();
    s.execute(FEEDBACK_CMD, json!({"about": "sound", "id": "whoosh", "verdict": "dislike"})).unwrap();
    let t = crate::taste::resolved(&s);
    assert!(!allows_sound(&t, "whoosh") && allows_sound(&t, "swish"));
    assert_eq!(why_of(&deal_p(&mut s, json!({"for": "transitionFamily"})), "spatial"), Some("not allowed by taste"));
    // nothing allowed: nothing dealt
    let r = deal_p(&mut s, json!({"for": "look", "options": ["warm", "golden"]}));
    assert!(r["choice"].is_null() && r["kept"] == false, "{r:#}");
    // shorts stay silent
    let r = deal_p(&mut s, json!({"for": "music", "pipeline": "shorts"}));
    assert!(r["choice"].is_null() && r["why"].as_str().unwrap().contains("silent"));
    // autocut.taste shows the retired choices and clears them
    let r = s.execute("autocut.taste", json!({})).unwrap();
    assert_eq!(r["resolved"]["retired"]["sound"], json!(["whoosh"]));
    s.execute("autocut.taste", json!({"scope": "default", "set": {"retired": null, "reasons": null}})).unwrap();
    assert!(crate::taste::resolved(&s).retired.is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_kept_deal_is_recorded_with_the_video() {
    let (mut s, dir) = session("record");
    s.execute("file.openDemoProject", json!({})).unwrap();
    deal_p(&mut s, json!({"for": "music", "options": ["happy", "calm"]}));
    deal_p(&mut s, json!({"for": "endCard", "options": ["cut", "fadeOut"], "dryRun": true}));
    let r = s.execute("autocut.fingerprint", json!({"record": true, "pipeline": "direct"})).unwrap();
    assert_eq!(r["record"]["deals"], json!({"music": "happy"}), "a dry run is not kept: {r:#}");
    assert!(s.taste_deals.is_empty(), "recorded once");
    // the next video's deal skips it
    assert_eq!(deal_p(&mut s, json!({"for": "music", "options": ["happy", "calm"]}))["choice"], "calm");
    // project feedback is one undo step
    s.execute(FEEDBACK_CMD, json!({"about": "endCard", "id": "fadeOut", "verdict": "dislike", "scope": "project"})).unwrap();
    assert!(s.project.taste.as_ref().is_some_and(|t| t.is_retired("endCard", "fadeOut")));
    s.undo().unwrap();
    assert!(s.project.taste.is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hostile_parameters_are_refused() {
    let (mut s, dir) = session("bad");
    for p in [
        json!({}),
        json!({"for": "colour"}),
        json!({"for": "look", "options": []}),
        json!({"for": "look", "options": ["sepia"]}),
        json!({"for": "look", "options": "warm"}),
        json!({"for": "look", "options": vec!["warm"; 100]}),
        json!({"for": "look", "seed": -1}),
        json!({"for": "look", "seed": 1.5}),
        json!({"for": "look", "dryRun": "yes"}),
        json!({"for": "look", "pipeline": "a b"}),
        json!({"for": "look", "extra": 1}),
        json!("look"),
    ] {
        assert!(s.execute(DEAL_CMD, p.clone()).is_err(), "{p}");
    }
    for p in [
        json!({"about": "look", "id": "warm"}),
        json!({"about": "look", "id": "sepia", "verdict": "dislike"}),
        json!({"about": "mood", "id": "laser", "verdict": "dislike"}),
        json!({"about": "everything", "id": "warm", "verdict": "dislike"}),
        json!({"about": "look", "id": "warm", "verdict": "hate"}),
        json!({"about": "look", "id": "warm", "verdict": "like", "scope": "world"}),
        json!({"about": "look", "id": "warm", "verdict": "like", "note": "x".repeat(1000)}),
    ] {
        assert!(s.execute(FEEDBACK_CMD, p.clone()).is_err(), "{p}");
    }
    for bad in [json!({"set": {"retired": {"look": ["sepia"]}}}), json!({"set": {"retired": {"everything": ["x"]}}}), json!({"set": {"liked": "warm"}})] {
        assert!(s.execute("autocut.taste", bad.clone()).is_err(), "{bad}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_catalog_is_real() {
    for kind in FEEDBACK_KINDS {
        let c = catalog(kind);
        assert!(!c.is_empty(), "{kind}");
        for id in &c {
            assert_eq!(canonical(kind, &id.to_ascii_uppercase()), Some(*id));
        }
    }
    assert!(catalog("look").contains(&"golden") && catalog("look").contains(&"vhs"));
    assert!(catalog("titleStyle").contains(&"slideUp"));
    assert_eq!(catalog("transitionFamily").last(), Some(&"cut"));
    assert!(catalog("register").contains(&"warm"));
    assert_eq!(kind_of("Close"), Some("endCard"));
    let (silent, sounds) = family_sounds("dissolve");
    assert!(silent && sounds.is_empty());
    let (silent, sounds) = family_sounds("zoom");
    assert!(!silent && sounds.contains(&"whoosh"));
}
