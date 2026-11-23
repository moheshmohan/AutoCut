//! `copy.lint` on texts, publish copy, script files and sequences made here: each list firing and
//! not firing, the device budgets and their levels, the fact ledger, the person's phrases beating
//! the lists (D6), the safe fixes, and hostile parameters.
use serde_json::{Value, json};

use super::*;
use crate::Session;

/// A session whose default taste profile lives in its own folder.
fn session(name: &str) -> (Session, std::path::PathBuf) {
    let dir = crate::media_test_util::tmp_dir(name);
    let s = Session { prefs_path: Some(dir.join("preferences.json")), ..Session::default() };
    (s, dir)
}

fn lint(s: &mut Session, p: Value) -> Value {
    s.execute(CMD, p).unwrap()
}

fn with_code<'a>(r: &'a Value, code: &str) -> Vec<&'a Value> {
    r["findings"].as_array().unwrap().iter().filter(|f| f["code"] == code).collect()
}

fn phrases(r: &Value, code: &str) -> Vec<String> {
    with_code(r, code).iter().map(|f| f["phrase"].as_str().unwrap().to_string()).collect()
}

fn budget<'a>(r: &'a Value, source: &str, device: &str) -> &'a Value {
    r["budgets"].as_array().unwrap().iter().find(|b| b["source"] == source && b["device"] == device).unwrap()
}

const PLAIN: &str = "We built the printer from a kit over two weekends. The frame went together first, then the bed, \
                     and the first bracket came off the plate in under an hour.";

#[test]
fn each_list_fires_on_its_phrases_and_plain_copy_is_clean() {
    let (mut s, _) = session("copy-lists");
    let r = lint(&mut s, json!({"text": PLAIN}));
    assert_eq!(r["ok"], true, "{r:#}");
    assert_eq!(r["blocking"], false);
    // stock phrases: video ones and marketing ones, whatever the apostrophe or the hyphen
    let r = lint(
        &mut s,
        json!({"text": "In this video, we\u{2019}ll build a printer. Let's dive in! It is a game-changer. Smash that like button and don't forget to subscribe."}),
    );
    let stock = phrases(&r, "stockPhrase");
    for want in ["in this video we'll", "let's dive in", "game changer", "smash that like button", "don't forget to subscribe"] {
        assert!(stock.iter().any(|p| p == want), "{want}: {stock:?}");
    }
    let categories: Vec<&str> = with_code(&r, "stockPhrase").iter().map(|f| f["category"].as_str().unwrap()).collect();
    assert!(categories.contains(&"video") && categories.contains(&"general"), "{categories:?}");
    assert!(with_code(&r, "stockPhrase").iter().all(|f| f["severity"] == "warn"));
    let dive = with_code(&r, "stockPhrase").into_iter().find(|f| f["phrase"] == "let's dive in").unwrap();
    assert_eq!(dive["match"], "Let's dive in");
    assert!(dive["context"].as_str().unwrap().contains("Let's dive in!"), "{dive}");
    // a stock phrase across a sentence break is not one
    let r = lint(&mut s, json!({"text": "We tested it in this video. We'll print more later."}));
    assert!(with_code(&r, "stockPhrase").is_empty(), "{r:#}");
    // filler and hedges
    let r = lint(&mut s, json!({"text": "Basically it kind of works, and I think it is probably fine."}));
    let cats: Vec<(String, String)> =
        with_code(&r, "filler").iter().map(|f| (f["category"].as_str().unwrap().into(), f["phrase"].as_str().unwrap().into())).collect();
    for want in [("filler", "basically"), ("filler", "kind of"), ("hedge", "i think"), ("hedge", "probably")] {
        assert!(cats.contains(&(want.0.into(), want.1.into())), "{want:?}: {cats:?}");
    }
    // "a kind of printer" names a kind
    let r = lint(&mut s, json!({"text": "It is a kind of printer, and it sort of works. What sort of bed?"}));
    assert_eq!(phrases(&r, "filler"), ["sort of"], "{r:#}");
    // narration that describes the shot: in narration, not in a description
    let shot = "As you can see, the bed is level. Here we see the first layer.";
    let r = lint(&mut s, json!({"text": shot}));
    assert_eq!(phrases(&r, "describesShot"), ["as you can see", "here we see"], "{r:#}");
    let r = lint(&mut s, json!({"publish": {"description": shot}}));
    assert!(with_code(&r, "describesShot").is_empty(), "{r:#}");
    assert_eq!(r["lists"]["videoStock"], VIDEO_STOCK.len());
}

#[test]
fn budgets_warn_over_the_budget_and_mark_the_fail_level_at_twice_plus_one() {
    let (mut s, _) = session("copy-budgets");
    // under 100 words: the budget of 100 (one exclamation mark), the fail level at 3
    for (marks, level) in [(1u64, "ok"), (2, "warn"), (3, "fail"), (4, "fail")] {
        let text = format!("{PLAIN}{}", " Done!".repeat(marks as usize));
        let r = lint(&mut s, json!({"text": text}));
        let b = budget(&r, "text", "exclamation");
        assert_eq!((b["count"].as_u64(), b["level"].as_str()), (Some(marks), Some(level)), "{b}");
        assert_eq!((b["budget"].as_f64(), b["failAt"].as_f64()), (Some(1.0), Some(3.0)));
        let over = with_code(&r, "overBudget");
        if level == "ok" {
            assert!(over.is_empty(), "{r:#}");
        } else {
            assert_eq!(over.len(), 1, "{r:#}");
            assert_eq!((over[0]["level"].as_str(), over[0]["severity"].as_str()), (Some(level), Some("warn")), "a fail level is still a warning (D7)");
        }
    }
    // over 250 words: over two and a half exclamation marks, the fail level over six
    let long = format!("{} ", "word ".repeat(250));
    for (marks, level) in [(2, "ok"), (3, "warn"), (6, "warn"), (7, "fail")] {
        let r = lint(&mut s, json!({"text": format!("{long}{}", "Yes! ".repeat(marks))}));
        assert_eq!(budget(&r, "text", "exclamation")["level"], level, "{marks}");
    }
    // each device
    let r = lint(
        &mut s,
        json!({"text": "Why? Really?? Who knows? It works \u{2014} mostly \u{2014} today \u{2014} and -- tomorrow. \u{1F525}\u{1F525}\u{1F525} \u{1F1E6}\u{1F1EA} This is AMAZING and INSANE, FREE too. The UAE team used PLA and PETG."}),
    );
    let count = |device: &str| budget(&r, "text", device)["count"].as_u64().unwrap();
    assert_eq!(count("question"), 3, "a run of question marks is one question");
    assert_eq!(count("emDash"), 4);
    assert_eq!(count("emoji"), 4, "three pictures and one flag");
    assert_eq!(count("allCaps"), 3, "AMAZING, INSANE and FREE; UAE, PLA and PETG are acronyms");
    // a title set in capitals is a style, not shouting
    let r = lint(&mut s, json!({"publish": {"title": "BUILDING THE PRINTER FRAME"}}));
    assert_eq!(budget(&r, "title", "allCaps")["count"], 0, "{r:#}");
    // devices in a web address are not counted
    let r = lint(&mut s, json!({"publish": {"description": "Plans: https://example.com/a?b=1&c=2--x!"}}));
    assert_eq!((budget(&r, "description", "question")["count"].as_u64(), budget(&r, "description", "exclamation")["count"].as_u64()), (Some(0), Some(0)));
}

#[test]
fn figures_are_checked_against_the_fact_ledger() {
    let (mut s, dir) = session("copy-facts");
    let text = "The kit cost 1,200 dirhams in 2014 and prints at 60 mm/s, 25% faster. Part 2 at 0:45 shows the 3D bed in 4K and 1080p; \
                see https://example.com/v/1234 and #2024. It weighs 10kg.";
    // without a ledger figures are not judged, and the result says so
    let r = lint(&mut s, json!({"text": text}));
    assert!(with_code(&r, "unledgeredFigure").is_empty());
    assert!(r["skipped"].as_array().unwrap().iter().any(|k| k["check"] == "unledgeredFigure"), "{r:#}");
    assert!(r["ledger"]["from"].is_null());
    // with one: by value, however it is written
    let r = lint(&mut s, json!({"text": text, "facts": ["Bought in 2014 for 1200 dirhams", 25, {"text": "weight 10 kg"}]}));
    let flagged: Vec<&str> = with_code(&r, "unledgeredFigure").iter().map(|f| f["figure"].as_str().unwrap()).collect();
    assert_eq!(flagged, ["60"], "codes, formats, clock times, numbering, addresses and hashtags are not figures: {r:#}");
    assert_eq!((r["ledger"]["from"].as_str(), r["ledger"]["entries"].as_u64()), (Some("facts"), Some(3)));
    // the ledger beside the saved project
    std::fs::create_dir_all(dir.join("build")).unwrap();
    s.path = Some(dir.join("edit.fcproj").display().to_string());
    std::fs::write(dir.join("build/facts.json"), br#"{"facts": ["60 mm/s print speed", "25 percent"]}"#).unwrap();
    let r = lint(&mut s, json!({"text": "60 mm/s, 25% faster, 300 prints"}));
    let flagged: Vec<&str> = with_code(&r, "unledgeredFigure").iter().map(|f| f["figure"].as_str().unwrap()).collect();
    assert_eq!(flagged, ["300"], "{r:#}");
    assert!(r["ledger"]["from"].as_str().unwrap().ends_with("facts.json"));
    // a broken ledger file is said, not taken as empty
    std::fs::write(dir.join("build/facts.json"), b"{not json").unwrap();
    let r = lint(&mut s, json!({"text": "300 prints"}));
    assert!(with_code(&r, "unledgeredFigure").is_empty());
    assert!(r["ledger"]["error"].is_string(), "{r:#}");
    // `facts` wins over the file
    let r = lint(&mut s, json!({"text": "300 prints", "facts": []}));
    assert_eq!(with_code(&r, "unledgeredFigure").len(), 1, "an empty ledger allows no figure");
}

#[test]
fn the_persons_own_phrases_beat_the_lists() {
    let (mut s, _) = session("copy-allow");
    let text = "Hello everyone! Let's dive in. Basically WOW!! It is amazing.";
    let r = lint(&mut s, json!({"text": text}));
    assert_eq!(phrases(&r, "stockPhrase"), ["hello everyone", "let's dive in"], "{r:#}");
    assert_eq!(budget(&r, "text", "exclamation")["count"], 3);
    // the `allow` parameter
    let r = lint(&mut s, json!({"text": text, "allow": ["let's dive in", "WOW!!"]}));
    assert_eq!(phrases(&r, "stockPhrase"), ["hello everyone"], "{r:#}");
    assert_eq!(budget(&r, "text", "exclamation")["count"], 1, "marks inside the person's phrase are theirs");
    assert_eq!(r["allowed"]["suppressed"], 1);
    // the project's taste profile, and the user's default one, together
    s.execute("autocut.taste", json!({"set": {"phrases": ["hello everyone"]}})).unwrap();
    s.execute("autocut.taste", json!({"scope": "default", "set": {"phrases": ["let's dive in"], "words": [{"say": "basically", "not": "essentially"}]}}))
        .unwrap();
    let r = lint(&mut s, json!({"text": text}));
    assert!(with_code(&r, "stockPhrase").is_empty() && with_code(&r, "filler").is_empty(), "{r:#}");
    assert_eq!((r["allowed"]["fromTaste"].as_u64(), r["allowed"]["suppressed"].as_u64()), (Some(3), Some(3)));
    // a figure inside the person's phrase is theirs too
    let r = lint(&mut s, json!({"text": "Printer 2 forever, and 7 more", "facts": [], "allow": ["printer 2 forever"]}));
    let flagged: Vec<&str> = with_code(&r, "unledgeredFigure").iter().map(|f| f["figure"].as_str().unwrap()).collect();
    assert_eq!(flagged, ["7"]);
}

#[test]
fn sequence_titles_and_caption_cues_are_read_where_they_are() {
    let (mut s, _) = session("copy-sequence");
    s.execute("file.newSequence", json!({"width": 640, "height": 360, "fps": 30, "video": 2, "audio": 1})).unwrap();
    let at = filmcraft_time::Tick::from_seconds_f64(2.0).0;
    let title = s
        .execute("graphics.newText", json!({"text": "Let's dive in!!! As you can see", "size": 40, "seconds": 3.0, "time": at, "track": "V2"}))
        .unwrap()["clip"]
        .as_u64()
        .unwrap();
    s.execute("captions.newTrack", json!({"format": "Subtitle"})).unwrap();
    s.execute("captions.add", json!({"track": "C1", "text": "As you can see the bed is level", "seconds": 1.0, "durationSeconds": 2.0})).unwrap();
    let seq = s.state.active_sequence.unwrap().0;
    let r = lint(&mut s, json!({"sequence": seq}));
    let stock = with_code(&r, "stockPhrase");
    assert_eq!(stock.len(), 1, "{r:#}");
    assert_eq!((stock[0]["kind"].as_str(), stock[0]["clip"].as_u64(), stock[0]["time"].as_i64()), (Some("title"), Some(title), Some(at)));
    assert_eq!((stock[0]["source"].as_str(), stock[0]["track"].as_str()), (Some("titles"), Some("V2")));
    // a title is not narration; a caption cue is
    let shots = with_code(&r, "describesShot");
    assert_eq!(shots.len(), 1, "{r:#}");
    assert_eq!((shots[0]["kind"].as_str(), shots[0]["source"].as_str(), shots[0]["track"].as_str()), (Some("caption"), Some("captions:C1"), Some("C1")));
    assert_eq!(budget(&r, "titles", "exclamation")["count"], 3);
    assert_eq!(r["sequence"], seq);
}

#[test]
fn script_files_are_read_part_by_part() {
    let (mut s, dir) = session("copy-script");
    let lines = dir.join("lines.json");
    std::fs::write(&lines, serde_json::to_vec(&json!({"intro": "Hey guys, welcome back to my channel.", "outro": "See you in the next one.", "lines": {"Frame": "Here we see the frame going together."}})).unwrap()).unwrap();
    let r = lint(&mut s, json!({"script": lines}));
    let parts: Vec<&str> = r["findings"].as_array().unwrap().iter().filter_map(|f| f["part"].as_str()).collect();
    for want in ["intro", "outro", "Frame"] {
        assert!(parts.contains(&want), "{want}: {r:#}");
    }
    assert_eq!(with_code(&r, "describesShot")[0]["part"], "Frame");
    let plain = dir.join("script.txt");
    std::fs::write(&plain, "Without further ado, the frame.").unwrap();
    assert_eq!(phrases(&lint(&mut s, json!({"script": plain})), "stockPhrase"), ["without further ado"]);
    let list = dir.join("list.json");
    std::fs::write(&list, b"[\"a\"]").unwrap();
    assert!(s.execute(CMD, json!({"script": list})).is_err());
}

#[test]
fn hostile_parameters_are_errors_and_hostile_text_is_read() {
    let (mut s, dir) = session("copy-hostile");
    let big = "a".repeat(MAX_TEXT_BYTES + 1);
    for (p, want) in [
        (json!([1]), "must be an object"),
        (json!({}), "at least one"),
        (json!({"text": null}), "at least one"),
        (json!({"text": 5}), "must be a string"),
        (json!({"text": big}), "at most"),
        (json!({"publish": "title"}), "`publish` is"),
        (json!({"publish": {}}), "none of"),
        (json!({"publish": {"title": 3}}), "must be a string"),
        (json!({"publish": {"tags": "a,b"}}), "`tags`"),
        (json!({"publish": {"tags": [1]}}), "each tag"),
        (json!({"publish": {"tags": vec!["x"; MAX_TAGS + 1]}}), "`tags`"),
        (json!({"publish": {"tags": ["x".repeat(MAX_TAG_CHARS + 1)]}}), "each tag"),
        (json!({"text": "x", "facts": "60"}), "`facts`"),
        (json!({"text": "x", "facts": [true]}), "ledger entry"),
        (json!({"text": "x", "facts": [{"source": "a"}]}), "`text`"),
        (json!({"text": "x", "facts": vec![1; MAX_FACTS + 1]}), "at most"),
        (json!({"text": "x", "allow": "mine"}), "`allow`"),
        (json!({"text": "x", "allow": [7]}), "`allow`"),
        (json!({"text": "x", "allow": vec!["a"; MAX_ALLOW + 1]}), "`allow`"),
        (json!({"sequence": "main"}), "sequence id"),
        (json!({"sequence": 999_999}), "no such sequence"),
        (json!({"script": 3}), "file path"),
        (json!({"script": "a\u{0}b"}), "file path"),
        (json!({"script": dir.join("missing.txt")}), "missing.txt"),
    ] {
        let e = s.execute(CMD, p.clone()).unwrap_err().to_string();
        assert!(e.contains(want), "{p}: {e}");
    }
    // text no list expects: joined emoji, marks, scripts without case, lone apostrophes
    for t in [
        "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}!!",
        "'' '' ' -- \u{2014}",
        "\u{0D28}\u{0D2E}\u{0D38}\u{0D4D}\u{0D15}\u{0D3E}\u{0D30}\u{0D02} 2014",
        "e\u{301}\u{301}\u{301}",
        "#",
        "0:0:0:0",
        "1,,2..3",
        "9".repeat(400).as_str(),
    ] {
        let r = lint(&mut s, json!({"text": t, "facts": []}));
        assert_eq!(r["blocking"], false, "{t}");
    }
    // a long text full of hedges: every one is counted, a bounded number listed
    let r = lint(&mut s, json!({"text": "maybe ".repeat(50_000)}));
    assert_eq!(r["counts"]["filler"], 50_000);
    assert_eq!(with_code(&r, "filler").len(), 200);
}

#[test]
fn safe_fixes_trim_only_whole_stock_sentences_and_collapse_exclamation_marks() {
    let none: Vec<Vec<String>> = Vec::new();
    let (t, fixes) = fix_text("Hey guys! Today we print a bracket in PETG. It took 2 hours!!! Don't forget to like and subscribe!", true, &none);
    assert_eq!(t, "Today we print a bracket in PETG. It took 2 hours!");
    let kinds: Vec<&str> = fixes.iter().map(|f| f["fix"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["trimmedStart", "trimmedEnd", "collapsedExclamations"], "{fixes:?}");
    assert_eq!(fixes[1]["removed"], "Don't forget to like and subscribe!");
    // a stock phrase with content after it is the writer's to change
    for keep in ["Without further ado, here is the build.", "Let's dive in!", "Build log. Subscribe for more. Then the bed."] {
        assert_eq!(fix_text(keep, true, &none).0, keep);
    }
    // a run of greetings and calls to action over several lines, at both ends
    let (t, _) =
        fix_text("Hello everyone!\nSo, welcome back to my channel.\n\nThe bed is level.\n\nLike and subscribe!\nSee you in the next one.", true, &none);
    assert_eq!(t, "The bed is level.");
    // the person's own greeting stays
    let mine = vec![vec!["hey".to_string(), "guys".to_string()]];
    assert_eq!(fix_text("Hey guys! The bed is level.", true, &mine).0, "Hey guys! The bed is level.");
    // a title is never trimmed, only collapsed; question marks and single marks stay
    assert_eq!(fix_text("Let's dive in!! Printing?!", false, &none).0, "Let's dive in! Printing?!");
    // the meaning: what is left is the original's words in order, nothing replaced
    let original = "Hi everyone! We rebuilt the extruder twice!! It finally works. Smash that like button!";
    let (fixed, _) = fix_text(original, true, &none);
    assert_eq!(fixed, "We rebuilt the extruder twice! It finally works.");
    let words = |t: &str| tokens(t, &[]).into_iter().map(|t| t.norm).collect::<Vec<_>>();
    let (all, kept) = (words(original), words(&fixed));
    let mut it = all.iter();
    assert!(kept.iter().all(|w| it.any(|o| o == w)), "every kept word is the original's, in order");
}
