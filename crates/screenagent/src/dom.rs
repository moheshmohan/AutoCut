//! The picker read and driven through its own page instead of its pixels, where the browser
//! allows it (Chrome with View ▸ Developer ▸ Allow JavaScript from Apple Events ticked).
//!
//! Facts from the page (live, 2026-10-24, the 52-item album "akku tesla toy"):
//! - every media tile carries a `[role=checkbox]` whose `aria-label` is its kind, its shape and
//!   when it was taken: `Video - Landscape - Jul 28, 2025, 7:18:02 PM`, `Photo - Portrait - Mar
//!   10, 2024, 1:42:31 AM` — with a narrow no-break space (U+202F) before AM/PM. Its
//!   `aria-checked` is `true` once it is selected, and a click on it toggles it;
//! - the other checkboxes are not media: `Select` and one per day, `Select all photos from Tue,
//!   Jul 29, 2025` — never clicked here (a day's would select past what was asked);
//! - the grid is virtual: about 29 checkboxes are in the page at a time, more as its scroller
//!   (the element that scrolls furthest) moves on;
//! - Return on an album's exact title shows the album's items after a delay; the header then
//!   counts the selection ("52 selected").
//!
//! So [`wait_for_results`] waits until the grid's tiles are no longer the library's and have
//! settled, and [`tick_all`] walks the grid from the top, clicking every tile that is not
//! ticked yet, until a walk finds nothing left to tick and the header agrees. Each step is one
//! short script (Chrome's `execute javascript` does not wait for a promise): the waiting
//! between them is the agent's. The scripts are fixed text — nothing a page, a title or a model
//! says ever goes into one.

use std::collections::BTreeSet;

use crate::ScreenError;
use crate::act::Act;
use crate::tabs::Tab;

/// The page helpers every script starts with: `norm` (U+202F and U+00A0 made spaces), the
/// media tiles' checkboxes (`tiles`, by their label), the grid's scroller and `look` — what
/// the grid shows now.
const PRELUDE: &str = r#"var norm=function(s){return String(s||'').replace(/[  ]/g,' ').replace(/\s+/g,' ').trim();};var tileRe=/^[A-Z][A-Za-z ]{0,30} - .+ - [A-Z][a-z]{2,9}\.? \d{1,2}, \d{4}/;var tiles=function(){return Array.prototype.slice.call(document.querySelectorAll('[role=checkbox]')).filter(function(b){return tileRe.test(norm(b.getAttribute('aria-label')));});};var scroller=function(){var best=null;Array.prototype.forEach.call(document.querySelectorAll('*'),function(e){if(e.scrollHeight>e.clientHeight+50&&/(auto|scroll)/.test(getComputedStyle(e).overflowY)&&(!best||e.scrollHeight>best.scrollHeight)){best=e;}});return best;};var look=function(){var sc=scroller();var m=norm(document.body?document.body.innerText:'').match(/(\d[\d,]*) selected/);return {tiles:tiles().map(function(b){return {label:norm(b.getAttribute('aria-label')),checked:b.getAttribute('aria-checked')==='true'};}),header:m?m[0]:null,top:sc?Math.round(sc.scrollTop):-1,height:sc?Math.round(sc.scrollHeight):-1};};"#;

/// One step's script.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// What the grid shows; changes nothing.
    Look,
    /// Click every media tile in the page that is not ticked.
    Tick,
    /// The grid's scroller to its top.
    Top,
    /// The scroller on by 4/5 of its height (`moved`: it did move).
    Next,
}

impl Step {
    fn body(self) -> &'static str {
        match self {
            Step::Look => "return JSON.stringify(look());",
            Step::Tick => {
                "var n=0;tiles().forEach(function(b){if(b.getAttribute('aria-checked')!=='true'){b.click();n++;}});var l=look();l.clicked=n;return JSON.stringify(l);"
            }
            Step::Top => "var sc=scroller();if(sc){sc.scrollTop=0;}return JSON.stringify(look());",
            Step::Next => {
                "var sc=scroller();var before=sc?sc.scrollTop:0;if(sc){sc.scrollTop=before+Math.max(1,Math.round(sc.clientHeight*0.8));}var l=look();l.moved=sc?Math.abs(sc.scrollTop-before)>1:false;return JSON.stringify(l);"
            }
        }
    }
}

/// The script of one step: one expression, so its value is what Chrome answers.
pub fn script(step: Step) -> String {
    format!("(function(){{{PRELUDE}{}}})()", step.body())
}

/// One media tile as the page shows it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct Tile {
    pub label: String,
    pub checked: bool,
}

/// What a step answered.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(default)]
pub struct Look {
    pub tiles: Vec<Tile>,
    /// The header's "N selected", as it reads.
    pub header: Option<String>,
    pub top: i64,
    pub height: i64,
    pub moved: bool,
    pub clicked: u32,
}

impl Look {
    /// The header's count; `None` when it shows no selection.
    pub fn selected(&self) -> Option<u32> {
        self.header.as_deref().and_then(|h| crate::picker::selected_count(&h.replace(',', "")))
    }

    /// The tiles' labels, in page order.
    pub fn labels(&self) -> Vec<String> {
        self.tiles.iter().map(|t| t.label.clone()).collect()
    }
}

/// The most tiles one answer is read for.
const MAX_TILES: usize = 4000;

/// A media tile's label, normalised (U+202F and U+00A0 made spaces, white space collapsed), or
/// `None` for any other checkbox: `<kind> - <shape …> - <Mon D, YYYY, time>`, the kind a word
/// or two in letters ("Video", "Photo"), the last part starting with a date.
pub fn tile_label(raw: &str) -> Option<String> {
    let label = raw.replace(['\u{202f}', '\u{a0}'], " ").split_whitespace().collect::<Vec<_>>().join(" ");
    let parts: Vec<&str> = label.split(" - ").collect();
    let (kind, when) = (parts.first()?, parts.last()?);
    if parts.len() < 3 || kind.is_empty() || kind.len() > 31 || !kind.starts_with(|c: char| c.is_ascii_uppercase()) {
        return None;
    }
    if !kind.chars().all(|c| c.is_ascii_alphabetic() || c == ' ') {
        return None;
    }
    // "Jul 28, 2025, 7:18:02 PM": a month, a day, a four-digit year
    let mut words = when.split(' ');
    let month = words.next()?.trim_end_matches('.');
    let day = words.next()?.strip_suffix(',')?;
    let year = words.next()?.trim_end_matches(',');
    let ok = (3..=9).contains(&month.len())
        && month.chars().all(|c| c.is_ascii_alphabetic())
        && (1..=2).contains(&day.len())
        && day.chars().all(|c| c.is_ascii_digit())
        && year.len() == 4
        && year.chars().all(|c| c.is_ascii_digit());
    ok.then_some(label)
}

/// Run one step and read its answer; only media tiles are kept.
pub fn run<A: Act + ?Sized>(actor: &mut A, tab: &Tab, step: Step) -> Result<Look, ScreenError> {
    let answer = actor.run_js(tab, &script(step))?;
    let mut look: Look = serde_json::from_str(answer.trim())
        .map_err(|e| ScreenError::Parse(format!("the picker page answered `{}`: {e}", answer.chars().take(120).collect::<String>())))?;
    look.tiles.truncate(MAX_TILES);
    look.tiles = look.tiles.into_iter().filter_map(|t| tile_label(&t.label).map(|label| Tile { label, checked: t.checked })).collect();
    Ok(look)
}

/// How long the grid gets between two looks while a search's results load.
const RESULT_POLL_MS: u64 = 1000;
/// Looks at a grid that shows no tiles before that is taken for "no results" (it empties
/// while it loads).
const EMPTY_LOOKS: u32 = 5;

/// What the grid showed after a search was sent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Results {
    /// Tiles other than the library's, the same on two looks in a row.
    Shown(Look),
    /// No tiles, for [`EMPTY_LOOKS`] looks.
    Empty,
    /// After every look, the library's tiles as before the search (or nothing settled).
    Unchanged,
}

/// Wait up to `looks` looks, [`RESULT_POLL_MS`] apart, for the search's results: tiles that are
/// not `library` (the labels the grid showed before the search), the same twice in a row. The
/// album "akku tesla toy" answered after a delay that two looks 2.5 s apart missed, live.
pub fn wait_for_results<A: Act + ?Sized>(actor: &mut A, tab: &Tab, library: &[String], looks: u32) -> Result<Results, ScreenError> {
    let mut last: Option<Vec<String>> = None;
    let mut empty = 0u32;
    for _ in 0..looks {
        actor.sleep(RESULT_POLL_MS);
        let look = run(actor, tab, Step::Look)?;
        let labels = look.labels();
        if labels.is_empty() {
            empty += 1;
            if empty >= EMPTY_LOOKS {
                return Ok(Results::Empty);
            }
        } else {
            empty = 0;
            if labels != library && last.as_ref() == Some(&labels) {
                return Ok(Results::Shown(look));
            }
        }
        last = Some(labels);
    }
    Ok(Results::Unchanged)
}

/// How long a click on the tiles gets to take before the grid is looked at again, and a scroll
/// before the next step (the grid loads lazily).
const TICK_SETTLE_MS: u64 = 600;
const SCROLL_SETTLE_MS: u64 = 700;
/// Steps of one walk down the grid (a screen each: 2000 items are well under it), and walks.
const MAX_STEPS: u32 = 400;
const MAX_WALKS: u32 = 3;

/// What [`tick_all`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ticked {
    /// Different tiles the grid showed.
    pub tiles: usize,
    /// Clicks on tiles, all walks.
    pub clicked: u32,
    /// The header's count at the end; `None`: it shows no selection.
    pub selected: Option<u32>,
    /// Tiles still not ticked when the last walk ended.
    pub unticked: Vec<String>,
    pub walks: u32,
    /// The walk stopped early: more tiles than `cap_items` (a runaway grid).
    pub runaway: bool,
}

/// Tick every tile of the grid: walk it from the top, a screen at a time to its end (where the
/// scroller stops moving twice), clicking each tile not ticked yet; walk again until a walk
/// clicks nothing and leaves nothing unticked, and the header's count is not below the tiles
/// seen (two tiles can share a label: taken the same second). More than `cap_items` different
/// tiles ends the walk (`runaway`). `out_of_time` ends it with an error.
pub fn tick_all<A: Act + ?Sized>(
    actor: &mut A,
    tab: &Tab,
    cap_items: u32,
    out_of_time: &dyn Fn() -> bool,
    progress: &mut dyn FnMut(&str),
) -> Result<Ticked, ScreenError> {
    let cap = usize::try_from(cap_items).unwrap_or(usize::MAX);
    let mut out = Ticked::default();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for walk in 0..MAX_WALKS {
        out.walks = walk + 1;
        let mut unticked: BTreeSet<String> = BTreeSet::new();
        let mut clicked = 0u32;
        run(actor, tab, Step::Top)?;
        actor.sleep(SCROLL_SETTLE_MS);
        let mut still = 0u32;
        for _ in 0..MAX_STEPS {
            if out_of_time() {
                return Err(ScreenError::Stalled(format!(
                    "ran out of time ticking the album's tiles ({} of {} seen ticked)",
                    seen.len().saturating_sub(unticked.len()),
                    seen.len()
                )));
            }
            let tick = run(actor, tab, Step::Tick)?;
            clicked = clicked.saturating_add(tick.clicked);
            if tick.clicked > 0 {
                actor.sleep(TICK_SETTLE_MS);
            }
            let look = run(actor, tab, Step::Look)?;
            for t in &look.tiles {
                seen.insert(t.label.clone());
                if t.checked {
                    unticked.remove(&t.label);
                } else {
                    unticked.insert(t.label.clone());
                }
            }
            if seen.len() > cap {
                out.runaway = true;
                break;
            }
            let next = run(actor, tab, Step::Next)?;
            actor.sleep(SCROLL_SETTLE_MS);
            if next.moved {
                still = 0;
            } else {
                still += 1;
                if still >= 2 {
                    break;
                }
            }
        }
        out.clicked = out.clicked.saturating_add(clicked);
        out.tiles = seen.len();
        out.selected = run(actor, tab, Step::Look)?.selected();
        out.unticked = unticked.into_iter().collect();
        progress(&format!(
            "screenagent: walk {} of the grid: {} tile(s), {clicked} clicked, the header shows {}",
            walk + 1,
            out.tiles,
            out.selected.map_or_else(|| "nothing selected".to_string(), |n| format!("{n} selected"))
        ));
        let counted = out.selected.and_then(|n| usize::try_from(n).ok()).unwrap_or(0);
        if out.runaway || (clicked == 0 && out.unticked.is_empty()) || (out.unticked.is_empty() && counted >= out.tiles && out.tiles > 0) {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_tiles_are_told_from_the_other_checkboxes() {
        let narrow = "Video - Landscape - Jul 28, 2025, 7:18:02\u{202f}PM";
        assert_eq!(tile_label(narrow).as_deref(), Some("Video - Landscape - Jul 28, 2025, 7:18:02 PM"));
        for yes in
            ["Photo - Portrait - Mar 10, 2024, 1:42:31 AM", "Video - Landscape - Jun 7, 2026, 1:09:17\u{a0}AM", "Photo - Square - Sept. 3, 2023, 9:00:00 PM"]
        {
            assert!(tile_label(yes).is_some(), "{yes}");
        }
        for no in [
            "",
            "Select",
            "Select all photos from Tue, Jul 29, 2025",
            "Video",
            "Video - Landscape",
            "video - Landscape - Jul 28, 2025, 7:18:02 PM",
            "Video - Landscape - yesterday",
            "Video - Landscape - Jul 28, 25, 7:18 PM",
            "<img> - x - Jul 28, 2025",
        ] {
            assert!(tile_label(no).is_none(), "{no:?}");
        }
    }

    #[test]
    fn every_script_is_one_expression_of_fixed_text() {
        for step in [Step::Look, Step::Tick, Step::Top, Step::Next] {
            let s = script(step);
            assert!(s.starts_with("(function(){var norm=") && s.ends_with("})()"), "{s}");
            assert!(!s.chars().any(char::is_control), "one line, nothing AppleScript refuses: {step:?}");
            assert!(s.contains("JSON.stringify("));
        }
        // only Tick clicks, and only tiles not ticked; nothing selects a day or "Select"
        assert!(script(Step::Tick).contains("b.click()") && !script(Step::Look).contains("click"));
        assert!(!PRELUDE.contains("Select all"));
    }

    #[test]
    fn answers_are_parsed_and_only_media_tiles_kept() {
        let answer = r#"{"tiles":[{"label":"Select","checked":false},{"label":"Video - Landscape - Jul 28, 2025, 7:18:02 PM","checked":true}],"header":"1,234 selected","top":0,"height":900}"#;
        let look: Look = serde_json::from_str(answer).unwrap();
        assert_eq!(look.selected(), Some(1234), "a thousands comma is read");
        assert_eq!(Look { header: Some("Select up to 2000 items".into()), ..Look::default() }.selected(), None);
        assert_eq!(Look::default().selected(), None);
        struct One(String);
        impl Act for One {
            fn activate(&mut self) -> Result<(), ScreenError> {
                Ok(())
            }
            fn click_png(&mut self, _: &crate::capture::Capture, _: u32, _: u32) -> Result<(), ScreenError> {
                Ok(())
            }
            fn shift_click_png(&mut self, _: &crate::capture::Capture, _: u32, _: u32) -> Result<(), ScreenError> {
                Ok(())
            }
            fn type_text(&mut self, _: &str) -> Result<(), ScreenError> {
                Ok(())
            }
            fn key_return(&mut self) -> Result<(), ScreenError> {
                Ok(())
            }
            fn key_escape(&mut self) -> Result<(), ScreenError> {
                Ok(())
            }
            fn scroll(&mut self, _: i32) -> Result<(), ScreenError> {
                Ok(())
            }
            fn sleep(&mut self, _: u64) {}
            fn capture(&mut self) -> Result<crate::capture::Capture, ScreenError> {
                Err(ScreenError::Other("no screen".into()))
            }
            fn run_js(&mut self, _: &Tab, _: &str) -> Result<String, ScreenError> {
                Ok(self.0.clone())
            }
        }
        let tab = Tab { window: 1, id: 2, url: "https://photos.google.com/picker/x".into() };
        let look = run(&mut One(answer.into()), &tab, Step::Look).unwrap();
        assert_eq!(look.tiles.len(), 1, "`Select` is no tile: {look:?}");
        // garbage and hostile sizes are errors or bounded, never panics
        assert!(matches!(run(&mut One("missing value".into()), &tab, Step::Look), Err(ScreenError::Parse(_))));
        let many = format!(r#"{{"tiles":[{}]}}"#, vec![r#"{"label":"Photo - Portrait - Mar 10, 2024, 1:42:31 AM","checked":false}"#; MAX_TILES + 50].join(","));
        assert_eq!(run(&mut One(many), &tab, Step::Look).unwrap().tiles.len(), MAX_TILES);
    }
}
