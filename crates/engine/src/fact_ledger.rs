//! `script.factLedger` (PX7.3): every figure and claim shown on screen is in the fact ledger,
//! and every one there says where it is from. The ledger is `copy.lint`'s (`facts`, or
//! `build/facts.json` beside the saved project), whose entries may now name their sources:
//! `{"text": "The first powered flight lasted 12 seconds, in 1903", "source": "https://…"}` (or
//! `sources: [...]`; a web address or a note). The idea of checking an explainer's numbers
//! against its research comes from anything2explainer (PolyForm Noncommercial; idea only).
//!
//! | code | warns when |
//! |---|---|
//! | `unledgeredFigure` | a figure in digits is not in the ledger (as `copy.lint` finds it) |
//! | `unsourcedFigure` | a figure is in the ledger, but no entry holding it names a source |
//! | `unledgeredClaim` | a line makes a claim (a word of [`CLAIM_WORDS`]: firsts, onlys, records, superlatives, "invented", "million"…) that no ledger entry makes |
//! | `unsourcedClaim` | the claim is in the ledger, but without a source |
//!
//! **What is on screen.** The sequence's titles (every enabled text layer of a graphic clip) and
//! caption cues, as `copy.lint` reads them; or a script (`text`, in `script.timeline`'s markup):
//! its chapter titles, its `[shot: …]` hints and its paragraphs (the subtitles show them).
//!
//! **A claim matches an entry** when the entry has the claim word and shares at least two fifths
//! of the line's other words of three letters or more (at least one; words compared without
//! `s`, `es`, `ed` and `ing` endings, digits and the commonest words left out); a line with no
//! other word needs the claim word alone. So write a claim in the ledger in the words the screen
//! uses. A starting list, to tune on real projects.
//!
//! Every finding is a warning (decision D7). `delivery.preflight` runs the sources and claims as
//! the Floor check `factSources` once the ledger names a source (`factSources: true` always,
//! `false` never): a warning this round, blocking with `floorBlocks: true`.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::collections::BTreeSet;

use serde_json::{Map, Value, json};

use filmcraft_project::ItemId;

use crate::commands::{CommandSpec, always, bad};
use crate::copy_lint::{Fact, Kind, Ledger, LedgerState, Segment, figures, same_value, tokens};
use crate::{EngineError, Result, Session};

pub const CMD: &str = "script.factLedger";
/// Words that make a line a claim to source.
pub const CLAIM_WORDS: &[&str] = &[
    "first",
    "only",
    "record",
    "records",
    "never",
    "always",
    "invented",
    "discovered",
    "proved",
    "proven",
    "most",
    "least",
    "best",
    "worst",
    "biggest",
    "largest",
    "smallest",
    "fastest",
    "slowest",
    "highest",
    "lowest",
    "longest",
    "shortest",
    "oldest",
    "newest",
    "youngest",
    "tallest",
    "deepest",
    "hottest",
    "coldest",
    "heaviest",
    "lightest",
    "strongest",
    "richest",
    "percent",
    "hundred",
    "hundreds",
    "thousand",
    "thousands",
    "million",
    "millions",
    "billion",
    "billions",
    "trillion",
    "half",
    "twice",
    "double",
    "triple",
    "majority",
];
/// The commonest words, left out when a claim is compared with the ledger.
const COMMON: &[&str] = &[
    "the", "and", "for", "with", "that", "this", "from", "was", "were", "are", "its", "it's", "has", "have", "had", "but", "not", "you", "your", "our", "they",
    "their", "them", "his", "her", "she", "him", "who", "what", "when", "where", "which", "will", "would", "can", "could", "all", "any", "into", "out",
    "about", "than", "then", "there", "here", "been", "being", "also", "just", "one", "two", "ever", "over", "only", "very", "such", "these", "those", "how",
    "why",
];
/// Least share of a claim's other words an entry must share.
pub const CLAIM_SHARE: f64 = 0.4;
/// Most findings of one code listed, and the most texts read.
const MAX_PER_CODE: usize = 200;
const MAX_SEGMENTS: usize = 40_000;
const MAX_TEXT_BYTES: usize = 8 << 20;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Check Facts and Sources",
        menu: &[],
        shortcut: None,
        params: r#"{"sequence":id? (its titles and caption cues; the default when no `text` is given: the active sequence),"text":str? (a script in script.timeline's markup: chapter titles, hints and paragraphs),"facts":[str|number|{"text","source"?|"sources"?}]? (the ledger; default build/facts.json beside the saved project),"claims":bool=true (also check the lines that make a claim)}"#,
        enabled: always,
        run,
        journal: false,
    }]
}

/// A word compared without its commonest endings.
fn stem(w: &str) -> String {
    let w = w.trim_end_matches("'s");
    for end in ["ing", "ed", "es", "s"] {
        if let Some(x) = w.strip_suffix(end)
            && x.chars().count() >= 3
        {
            return x.to_string();
        }
    }
    w.to_string()
}

/// The claim words of `text` and its other words that count (stems, three letters or more, no
/// digits, not the commonest).
fn claim_parts(text: &str) -> (Vec<String>, BTreeSet<String>) {
    let mut claims = Vec::new();
    let mut words = BTreeSet::new();
    for t in tokens(text, &[]) {
        if t.tag {
            continue;
        }
        if CLAIM_WORDS.contains(&t.norm.as_str()) {
            if !claims.contains(&t.norm) {
                claims.push(t.norm.clone());
            }
            continue;
        }
        if t.norm.chars().count() >= 3 && !t.norm.chars().any(|c| c.is_ascii_digit()) && !COMMON.contains(&t.norm.as_str()) {
            words.insert(stem(&t.norm));
        }
    }
    (claims, words)
}

/// The entries of `ledger` that make the claim of `text` (its claim words and other words).
fn claim_matches<'a>(ledger: &'a Ledger, claims: &[String], words: &BTreeSet<String>) -> Vec<&'a Fact> {
    let need = if words.is_empty() { 0 } else { ((words.len() as f64 * CLAIM_SHARE).ceil() as usize).max(1) };
    ledger
        .facts
        .iter()
        .filter(|f| {
            let toks = tokens(&f.text, &[]);
            let has_claim = claims.iter().any(|c| toks.iter().any(|t| &t.norm == c));
            let theirs: BTreeSet<String> = toks.iter().map(|t| stem(&t.norm)).collect();
            has_claim && words.iter().filter(|w| theirs.contains(*w)).count() >= need
        })
        .collect()
}

fn finding(seg: &Segment, code: &str, message: String, extra: Value) -> Value {
    let mut o = Map::new();
    o.insert("code".into(), json!(code));
    o.insert("severity".into(), json!("warn"));
    o.insert("source".into(), json!(seg.unit));
    o.insert("message".into(), json!(message));
    o.extend(seg.at.clone());
    if let Value::Object(x) = extra {
        o.extend(x);
    }
    Value::Object(o)
}

fn excerpt(text: &str) -> String {
    let t: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    t.chars().take(160).collect()
}

/// The figures and claims of `segments` against `ledger`: findings (each code at most
/// [`MAX_PER_CODE`]) and counts.
pub(crate) fn report(segments: &[Segment], ledger: &Ledger, claims: bool) -> Value {
    let mut findings: Vec<Value> = Vec::new();
    let mut counts: std::collections::BTreeMap<&'static str, usize> = Default::default();
    let mut keep = |code: &'static str, f: Value, findings: &mut Vec<Value>| {
        let n = counts.entry(code).or_default();
        *n += 1;
        if *n <= MAX_PER_CODE {
            findings.push(f);
        }
    };
    let (mut read, mut bytes, mut n_figures, mut n_claims) = (0usize, 0usize, 0usize, 0usize);
    let mut skipped = Vec::new();
    for seg in segments.iter().take(MAX_SEGMENTS) {
        bytes = bytes.saturating_add(seg.text.len());
        if bytes > MAX_TEXT_BYTES {
            skipped.push(json!({"reason": format!("{read} of {} texts read: at most {MAX_TEXT_BYTES} bytes in one call", segments.len())}));
            break;
        }
        read += 1;
        for f in figures(&seg.text, &[]) {
            n_figures += 1;
            let holding: Vec<&Fact> = ledger.facts.iter().filter(|x| x.values.iter().any(|v| same_value(*v, f.value))).collect();
            let extra = json!({"figure": f.text, "value": f.value, "text": excerpt(&seg.text)});
            if holding.is_empty() {
                keep(
                    "unledgeredFigure",
                    finding(seg, "unledgeredFigure", format!("the figure {} is not in the fact ledger: add it with its source, or take it out", f.text), extra),
                    &mut findings,
                );
            } else if holding.iter().all(|x| x.sources.is_empty()) {
                let mut extra = extra;
                extra["facts"] = json!(holding.iter().take(5).map(|x| excerpt(&x.text)).collect::<Vec<_>>());
                keep(
                    "unsourcedFigure",
                    finding(
                        seg,
                        "unsourcedFigure",
                        format!("the figure {} is in the ledger, but no entry with it says where it is from: add its source", f.text),
                        extra,
                    ),
                    &mut findings,
                );
            }
        }
        if !claims {
            continue;
        }
        let (said, words) = claim_parts(&seg.text);
        if said.is_empty() {
            continue;
        }
        n_claims += 1;
        let matched = claim_matches(ledger, &said, &words);
        let extra = json!({"claim": said, "text": excerpt(&seg.text)});
        if matched.is_empty() {
            keep(
                "unledgeredClaim",
                finding(
                    seg,
                    "unledgeredClaim",
                    format!(
                        "“{}” claims {} and no ledger entry says so: add the claim with its source (in the words the screen uses), or soften it",
                        excerpt(&seg.text),
                        said.join(", ")
                    ),
                    extra,
                ),
                &mut findings,
            );
        } else if matched.iter().all(|x| x.sources.is_empty()) {
            let mut extra = extra;
            extra["facts"] = json!(matched.iter().take(5).map(|x| excerpt(&x.text)).collect::<Vec<_>>());
            keep(
                "unsourcedClaim",
                finding(seg, "unsourcedClaim", format!("“{}” is in the ledger without a source: add where it is from", excerpt(&seg.text)), extra),
                &mut findings,
            );
        }
    }
    let unsourced: Vec<String> = ledger.facts.iter().filter(|f| f.sources.is_empty()).take(50).map(|f| excerpt(&f.text)).collect();
    json!({
        "ok": findings.is_empty(),
        "blocking": false,
        "findings": findings,
        "counts": counts,
        "checked": {"texts": read, "figures": n_figures, "claims": n_claims},
        "skipped": skipped,
        "ledger": {
            "from": ledger.from,
            "entries": ledger.entries,
            "sourced": ledger.facts.iter().filter(|f| !f.sources.is_empty()).count(),
            "unsourced": ledger.facts.iter().filter(|f| f.sources.is_empty()).count(),
            "unsourcedEntries": unsourced,
        },
    })
}

/// What a script in `script.timeline`'s markup shows: chapter titles, hints and paragraphs.
pub(crate) fn script_segments(text: &str) -> std::result::Result<Vec<Segment>, String> {
    let malayalam = filmcraft_tts::malayalam::detect(text);
    let chapters = crate::explainer::parse(text, malayalam)?;
    let mut out = Vec::new();
    for (ci, c) in chapters.iter().enumerate() {
        if !c.title.is_empty() {
            let mut seg = Segment::new("chapter titles", Kind::Title, c.title.clone());
            seg.at.insert("chapter".into(), json!(ci));
            seg.at.insert("line".into(), json!(c.line));
            out.push(seg);
        }
        for p in &c.paragraphs {
            for h in &p.hints {
                let mut seg = Segment::new("hints", Kind::Title, h.clone());
                seg.at.insert("chapter".into(), json!(ci));
                seg.at.insert("line".into(), json!(p.line));
                out.push(seg);
            }
            let mut seg = Segment::new("script", Kind::Narration, crate::narration::shown_text(&p.text));
            seg.at.insert("chapter".into(), json!(ci));
            seg.at.insert("line".into(), json!(p.line));
            out.push(seg);
        }
    }
    Ok(out)
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    if !p.is_object() && !p.is_null() {
        return Err(bad(CMD, "parameters must be an object"));
    }
    if let Some(k) = p.as_object().and_then(|o| o.keys().find(|k| !matches!(k.as_str(), "sequence" | "text" | "facts" | "claims")).cloned()) {
        return Err(bad(CMD, format!("unknown parameter `{}` (sequence, text, facts, claims)", k.chars().take(64).collect::<String>())));
    }
    let claims = match p.get("claims") {
        None | Some(Value::Null) => true,
        Some(v) => v.as_bool().ok_or_else(|| bad(CMD, "`claims` is true or false"))?,
    };
    let mut segments = Vec::new();
    let text = p.get("text").filter(|v| !v.is_null());
    if let Some(v) = text {
        let t = v.as_str().ok_or_else(|| bad(CMD, "`text` is a script, a string"))?;
        if t.len() > crate::explainer::MAX_TEXT_BYTES {
            return Err(bad(CMD, format!("the script is {} bytes; at most {}", t.len(), crate::explainer::MAX_TEXT_BYTES)));
        }
        segments.extend(script_segments(t).map_err(|e| bad(CMD, e))?);
    }
    let sequence = match p.get("sequence").filter(|v| !v.is_null()) {
        Some(v) => Some(ItemId(v.as_u64().ok_or_else(|| bad(CMD, "`sequence` is a sequence id"))?)),
        None if text.is_none() => Some(s.state.active_sequence.ok_or_else(|| bad(CMD, "give a script (`text`) or open a sequence"))?),
        None => None,
    };
    if let Some(id) = sequence {
        let q = s.project.sequence(id).ok_or_else(|| bad(CMD, "no such sequence"))?;
        segments.extend(crate::copy_lint::sequence_segments(&s.project, q));
    }
    let ledger = match crate::copy_lint::ledger(s, CMD, p)? {
        LedgerState::Found(l) => l,
        LedgerState::None(why) => return Err(bad(CMD, format!("{why}: the facts the film states, each with its source, as {{\"text\", \"source\"}}"))),
        LedgerState::Broken(why) => return Err(EngineError::Other(format!("{CMD}: the fact ledger could not be read: {why}"))),
    };
    let mut r = report(&segments, &ledger, claims);
    r["sequence"] = json!(sequence.map(|i| i.0));
    Ok(r)
}

#[cfg(test)]
#[path = "fact_ledger_tests.rs"]
mod tests;
