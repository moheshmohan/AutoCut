//! What the edit is about: the **places** or the **people** (`focus`). A travel video shows the
//! places — streets, buildings, views, the crowd — and the person holding the camera only now
//! and then; a video of a party or a family day is about its people (the director's way before
//! this, and still its way when nothing says the video is about places).
//!
//! - **Which**: `focus` (`places`, `people`) names it; `auto` (the default) reads the brief (a
//!   trip, travel, a vlog, a holiday, a tour… is about places; a birthday, a wedding, a party,
//!   the family… about people), and else the footage: most moments outdoors and few people
//!   talking is a place.
//! - **The moments**, for places: ranked by how they look, how wide and whether outdoors, faces,
//!   smiles and speech counting for little ([`super::rank::Scores::rank_for_places`]); indoor
//!   moments (a shop, a lift) only as many as the outdoors cannot fill ([`places_pool`]).
//! - **Where a 360° shot looks**, for places: at what is not a person first, the farthest from
//!   the camera's holder first ([`is_person`]: a man, a woman, someone… — a crowd or people
//!   walking are the street, a place; the holder is the person below the lens or right by it);
//!   where no place was found away from the holder, the view the other way round. The looks keep
//!   their ids, so a plan names them as before.
//! - **Every part of the trip, in order** ([`cover_parts`], [`in_time_order`]): moments filmed
//!   hours apart are parts (a morning walk, the night out, the next day), each with its share of
//!   the shots, and one recording in at most [`MAX_PER_FILE`]; a shot over goes to a part that
//!   is short (or another recording), near where it falls in time; the
//!   shots between the opener and the ending keep filming order, and the ending is from the
//!   trip's last part.
//! - **How often the person is in it** ([`limit_people`]): the camera's holder never what a shot
//!   looks at (in 360° they are at its edges often enough), anyone else in at most a third
//!   ([`MAX_PEOPLE`]) — counting where a whip ends as well as where it starts. Their voice is
//!   heard over the music, and captioned (whole sentences), only where the shot shows who speaks
//!   ([`voiced`]). Past that, a 360° shot turns to its place: a follow
//!   drifts, a whip with no second place pushes in. An orbit, which sweeps past the holder,
//!   drifts too.

use serde::Serialize;

use super::{Camera, Candidate, Kind, LOOKS_APART, Look, Plan, PlannedShot, angle_between, cuts::TIME_JUMP, round2, window_from};

/// Shots of people a places edit keeps, a share of its shots (at least one).
pub const MAX_PEOPLE: f64 = 1.0 / 3.0;
/// A person this big on the sphere (degrees across) is right by the lens: the camera's holder.
const HOLDER_SIZE: f64 = 40.0;
/// A person this far below the horizon (degrees) is under the lens: the camera's holder.
const HOLDER_PITCH: f64 = -15.0;
/// A direction this close to a person is looking at them (degrees).
const NEAR: f64 = 45.0;
/// A place this far round from the holder is clear of them (degrees).
const CLEAR: f64 = 90.0;
/// Footage this much outdoors (a share of its moments) is about places, when people talk in
/// fewer than [`TALKING`] of them.
const OUTDOORS: f64 = 0.6;
const TALKING: f64 = 0.3;
/// Shots of one recording a places edit has at most: more places, not more of one walk.
pub const MAX_PER_FILE: usize = 2;
/// Indoor moments a places edit keeps when the outdoors fill it: a share of its shots (one at
/// least).
const INDOORS: f64 = 1.0 / 6.0;

/// What the edit is about.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Focus {
    /// Its people: faces, smiles and speech raise a moment, a 360° shot looks at who is there.
    #[default]
    People,
    /// The places: the view first, the person holding the camera now and then.
    Places,
}

impl Focus {
    pub fn name(self) -> &'static str {
        match self {
            Focus::People => "people",
            Focus::Places => "places",
        }
    }

    /// `places`, `people`, or none for `auto` (an error names anything else).
    pub fn parse(v: &str) -> Result<Option<Focus>, String> {
        match v.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(None),
            "places" | "place" | "travel" => Ok(Some(Focus::Places)),
            "people" | "person" => Ok(Some(Focus::People)),
            o => Err(format!("focus: auto, places or people, not `{o}`")),
        }
    }
}

fn words(text: &str) -> Vec<String> {
    text.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_string).collect()
}

/// What a brief says the video is about, if it says.
fn brief_focus(brief: &str) -> Option<Focus> {
    let w = words(brief);
    let has = |list: &[&str]| w.iter().any(|x| list.contains(&x.as_str()));
    if has(&["birthday", "wedding", "party", "family", "kids", "son", "daughter", "interview", "talk", "portrait", "friends"]) {
        Some(Focus::People)
    } else if has(&[
        "trip",
        "travel",
        "travels",
        "travelling",
        "traveling",
        "vlog",
        "holiday",
        "vacation",
        "tour",
        "sightseeing",
        "journey",
        "explore",
        "exploring",
        "city",
        "streets",
        "walk",
    ]) {
        Some(Focus::Places)
    } else {
        None
    }
}

/// What the edit is about, and why: `asked` (the video's `focus`) first, then the brief, then
/// the footage.
pub fn focus_of(asked: Option<Focus>, brief: &str, cands: &[Candidate]) -> (Focus, String) {
    if let Some(f) = asked {
        return (f, format!("{}: asked", f.name()));
    }
    if let Some(f) = brief_focus(brief) {
        return (f, format!("{}: the brief", f.name()));
    }
    if cands.is_empty() {
        return (Focus::People, "people: no moments to judge by".into());
    }
    let n = cands.len() as f64;
    let outdoors = cands.iter().filter(|c| !c.indoor).count() as f64 / n;
    let talking = cands.iter().filter(|c| c.talking).count() as f64 / n;
    if outdoors >= OUTDOORS && talking < TALKING {
        (Focus::Places, format!("places: {:.0} % of the moments are outdoors and few have people talking", outdoors * 100.0))
    } else {
        (Focus::People, format!("people: {:.0} % of the moments are outdoors, {:.0} % have people talking", outdoors * 100.0, talking * 100.0))
    }
}

/// Whether a look's description is a person (one or two of them, not the crowd or the street).
pub fn is_person(what: &str) -> bool {
    const ONE: &[&str] = &[
        "man", "men", "woman", "women", "person", "boy", "boys", "girl", "girls", "child", "children", "kid", "kids", "guy", "guys", "lady", "he", "she",
        "selfie", "face", "faces", "couple", "friend", "friends", "son", "daughter", "baby", "himself", "herself", "someone", "somebody", "his", "her",
    ];
    const MANY: &[&str] = &["crowd", "crowds", "pedestrians", "tourists", "passersby", "shoppers", "visitors", "people"];
    let w = words(what);
    w.iter().any(|x| ONE.contains(&x.as_str())) && !w.iter().any(|x| MANY.contains(&x.as_str()))
}

/// The camera's holder in a 360° shot: a person below the lens or right by it (else the biggest
/// person found).
fn holder(looks: &[Look]) -> Option<&Look> {
    let people = || looks.iter().filter(|l| is_person(&l.what));
    people()
        .find(|l| l.pitch <= HOLDER_PITCH || l.size.is_some_and(|s| s.0 >= HOLDER_SIZE))
        .or_else(|| people().max_by(|a, b| a.size.map_or(0.0, |s| s.0).total_cmp(&b.size.map_or(0.0, |s| s.0))))
}

/// Whether look `l` of `looks` is at a person: described as one, or close to one that is.
fn at_person(l: &Look, looks: &[Look]) -> bool {
    is_person(&l.what) || looks.iter().filter(|p| is_person(&p.what)).any(|p| angle_between(p.yaw, l.yaw) < NEAR)
}

/// Whether look `l` is at (or close to) the camera's holder.
fn at_holder(l: &Look, looks: &[Look]) -> bool {
    holder(looks).is_some_and(|h| h.id == l.id || angle_between(h.yaw, l.yaw) < NEAR)
}

fn wrap(yaw: f64) -> f64 {
    let y = (yaw + 180.0).rem_euclid(360.0) - 180.0;
    if y.is_finite() { y } else { 0.0 }
}

/// The candidates of a places edit: ranked for places, a 360° shot's place looks first —
/// farthest from the camera's holder first — and the view the other way round from the holder
/// where no place was found clear of them. Their looks keep their ids.
pub fn for_places(cands: &mut [Candidate]) {
    for c in cands.iter_mut() {
        c.rank = c.scores.rank_for_places(c.kind, c.wide, c.indoor);
        if c.kind != Kind::Vr || c.looks.is_empty() {
            continue;
        }
        let looks = c.looks.clone();
        let from_holder = |l: &Look| holder(&looks).map_or(180.0, |h| angle_between(h.yaw, l.yaw));
        let mut place: Vec<Look> = looks.iter().filter(|l| !at_person(l, &looks)).cloned().collect();
        place.sort_by(|a, b| from_holder(b).total_cmp(&from_holder(a)));
        if let Some(h) = holder(&looks)
            && !place.iter().any(|l| from_holder(l) >= CLEAR)
        {
            let id = format!("{}{}", c.id, char::from(b'a' + (looks.len().min(25) as u8)));
            let what = if h.pitch <= HOLDER_PITCH || h.size.is_some_and(|s| s.0 >= HOLDER_SIZE) {
                "the view away from the camera's holder"
            } else {
                "the view away from the person"
            };
            place.insert(0, Look { id, yaw: wrap(h.yaw + 180.0), pitch: 0.0, what: what.into(), named: false, size: None });
        }
        let rest: Vec<Look> = looks.iter().filter(|l| !place.iter().any(|p| p.id == l.id)).cloned().collect();
        c.looks = place.into_iter().chain(rest).collect();
    }
}

/// The moments a places edit plans from: indoor ones only as many as the outdoors cannot fill
/// (`need` moments), and a few (the best) when they can. Returns how many were set aside.
pub fn places_pool(cands: &mut Vec<Candidate>, need: usize, shots: usize) -> usize {
    let outdoors = cands.iter().filter(|c| !c.indoor).count();
    let keep_in = if outdoors >= need { ((shots as f64 * INDOORS).floor() as usize).max(1) } else { need.saturating_sub(outdoors).max(1) };
    let mut indoor: Vec<(f64, String)> = cands.iter().filter(|c| c.indoor).map(|c| (c.rank, c.id.clone())).collect();
    indoor.sort_by(|a, b| b.0.total_cmp(&a.0));
    let gone: Vec<String> = indoor.into_iter().skip(keep_in).map(|(_, id)| id).collect();
    cands.retain(|c| !gone.contains(&c.id));
    gone.len()
}

/// The parts of the trip: the moments' filming times split where they are [`TIME_JUMP`] apart,
/// each part its first and last time.
pub(super) fn parts(cands: &[Candidate]) -> Vec<(i64, i64)> {
    let mut times: Vec<i64> = cands.iter().filter_map(|c| c.captured).collect();
    times.sort_unstable();
    let mut out: Vec<(i64, i64)> = Vec::new();
    for t in times {
        match out.last_mut() {
            Some(p) if t.saturating_sub(p.1) < TIME_JUMP => p.1 = t,
            _ => out.push((t, t)),
        }
    }
    out
}

pub(super) fn part_of(parts: &[(i64, i64)], t: Option<i64>) -> Option<usize> {
    let t = t?;
    parts.iter().position(|(a, b)| t >= *a && t <= *b)
}

/// A shot of candidate `c` in place of `s` (same length): its camera kept where it fits, a
/// drift (360°) or a push in otherwise; its first look.
fn shot_of(s: &PlannedShot, c: &Candidate, beat: f64) -> Option<PlannedShot> {
    let secs = f64::from(s.beats) * beat;
    let fits = |m: Camera| c.fits(m) && m.min_seconds() <= secs + 1e-6 && !matches!(m, Camera::PlanetIn | Camera::PlanetOut | Camera::WhipPan);
    let camera = if fits(s.camera) {
        s.camera
    } else if c.kind == Kind::Vr && fits(Camera::Drift) {
        Camera::Drift
    } else {
        Camera::PushIn
    };
    let from = window_from(c, secs * camera.speed())?;
    Some(PlannedShot {
        clip: c.id.clone(),
        file: String::new(),
        from: round2(from),
        camera,
        look: c.looks.first().map(|l| l.id.clone()),
        to: None,
        ..s.clone()
    })
}

/// Every part of the trip in the edit, and many places: a part under its share of the body shots
/// (the shots between the opener and the ending, shared among the parts and one more), while it
/// has unused moments, or a recording in more than [`MAX_PER_FILE`] shots, gives a body shot to
/// the best unused moment outdoors (else any) of the part that is short, else of a recording not
/// over — one not in the edit yet first. The shot given up is the weakest of the recording with
/// too many, else of the part with the most, nearest where the new one falls in time. Returns
/// what changed.
pub fn cover_parts(plan: &mut Plan, cands: &[Candidate], beat: f64) -> Vec<String> {
    let ps = parts(cands);
    let mut out = Vec::new();
    let n = plan.shots.len();
    if n < 4 {
        return out;
    }
    let last = n - 1;
    let share = if ps.len() >= 2 { ((n - 2) / (ps.len() + 1)).max(1) } else { 0 };
    let cand = |id: &str| cands.iter().find(|c| c.id == id);
    for _ in 0..n {
        let part_n = |plan: &Plan, p: usize| plan.shots.iter().filter(|s| part_of(&ps, cand(&s.clip).and_then(|c| c.captured)) == Some(p)).count();
        let file_n = |plan: &Plan, item: u64| plan.shots.iter().filter(|s| cand(&s.clip).is_some_and(|c| c.item == item)).count();
        let used: Vec<String> = plan.shots.iter().map(|s| s.clip.clone()).collect();
        let unused = |c: &&Candidate| !used.contains(&c.id);
        // what is wrong: a part short of its share (with moments to give), else a recording over
        let short = (0..ps.len())
            .filter(|p| part_n(plan, *p) < share && cands.iter().filter(unused).any(|c| part_of(&ps, c.captured) == Some(*p)))
            .min_by_key(|p| part_n(plan, *p));
        let over = plan
            .shots
            .get(1..last)
            .unwrap_or_default()
            .iter()
            .filter_map(|s| cand(&s.clip).map(|c| c.item))
            .filter(|i| file_n(plan, *i) > MAX_PER_FILE)
            .max_by_key(|i| file_n(plan, *i));
        if short.is_none() && over.is_none() {
            break;
        }
        // the moment coming in
        let fresh = |c: &&Candidate| file_n(plan, c.item) == 0;
        let pool: Vec<&Candidate> = match short {
            Some(p) => cands.iter().filter(unused).filter(|c| part_of(&ps, c.captured) == Some(p)).collect(),
            None => cands.iter().filter(unused).filter(|c| Some(c.item) != over && file_n(plan, c.item) < MAX_PER_FILE).collect(),
        };
        let score = |c: &&&Candidate| (u8::from(!c.indoor), u8::from(fresh(c)), (c.rank * 1000.0) as i64);
        let Some(&best) = pool.iter().max_by_key(score) else { break };
        // the shot going out
        let biggest = (0..ps.len()).max_by_key(|p| part_n(plan, *p));
        let donors: Vec<usize> = (1..last)
            .filter(|i| {
                let c = plan.shots.get(*i).and_then(|s| cand(&s.clip));
                match over {
                    Some(item) => c.is_some_and(|c| c.item == item),
                    None => c.is_some_and(|c| part_of(&ps, c.captured) == biggest) && biggest.is_some_and(|p| part_n(plan, p) > share),
                }
            })
            .collect();
        let at = 1 + plan
            .shots
            .get(1..last)
            .unwrap_or_default()
            .iter()
            .filter(|s| cand(&s.clip).and_then(|c| c.captured).zip(best.captured).is_some_and(|(t, b)| t < b))
            .count();
        let weakest = donors.into_iter().min_by(|a, b| {
            let r = |i: &usize| plan.shots.get(*i).and_then(|s| cand(&s.clip)).map_or(0.0, |c| c.rank);
            r(a).total_cmp(&r(b)).then(a.abs_diff(at).cmp(&b.abs_diff(at)))
        });
        let Some(i) = weakest else { break };
        let Some(s) = plan.shots.get(i) else { break };
        let Some(new) = shot_of(s, best, beat) else { break };
        let why = if short.is_some() { "every part of the trip in the edit" } else { "at most two shots of one recording" };
        out.push(format!("shot {}: {} → {} ({}): {why}", i + 1, s.clip, best.id, best.caption.chars().take(60).collect::<String>()));
        if let Some(slot) = plan.shots.get_mut(i) {
            *slot = new;
        }
    }
    out
}

/// A places edit told in filming order: the shots between the opener (the hook, from anywhere)
/// and the ending sorted by when they were filmed (each keeps its length), and the ending from
/// the trip's last part — its best unused 360° moment there, when it is not already. Returns
/// what changed.
pub fn in_time_order(plan: &mut Plan, cands: &[Candidate], beat: f64) -> Vec<String> {
    let mut out = Vec::new();
    let n = plan.shots.len();
    if n < 3 {
        return out;
    }
    let time_of = |s: &PlannedShot| cands.iter().find(|c| c.id == s.clip).and_then(|c| c.captured);
    let before: Vec<String> = plan.shots.iter().map(|s| s.clip.clone()).collect();
    if let Some(body) = plan.shots.get_mut(1..n - 1) {
        body.sort_by_key(|s| time_of(s).unwrap_or(i64::MAX));
    }
    if plan.shots.iter().map(|s| &s.clip).ne(before.iter()) {
        out.push("the shots between the opener and the ending in filming order: a trip told as it went".into());
    }
    let ps = parts(cands);
    let last_part = ps.len().checked_sub(1);
    let Some(end) = plan.shots.last() else { return out };
    if last_part.is_some() && part_of(&ps, time_of(end)) != last_part {
        let used: Vec<&str> = plan.shots.iter().map(|s| s.clip.as_str()).collect();
        let best = cands
            .iter()
            .filter(|c| c.kind == Kind::Vr && !c.indoor && part_of(&ps, c.captured) == last_part && !used.contains(&c.id.as_str()))
            .max_by(|a, b| a.rank.total_cmp(&b.rank));
        if let Some(c) = best {
            let secs = f64::from(end.beats) * beat;
            if let Some(from) = window_from(c, secs * end.camera.speed()) {
                let was = end.clip.clone();
                let mut new = PlannedShot {
                    clip: c.id.clone(),
                    file: String::new(),
                    from: round2(from),
                    look: c.looks.first().map(|l| l.id.clone()),
                    to: None,
                    ..end.clone()
                };
                // its camera, where the new moment allows it: a whip needs a second look, a
                // follow a named subject
                if new.camera == Camera::WhipPan {
                    match c.second_look() {
                        Some(l) => new.to = Some(l.id.clone()),
                        None => new.camera = Camera::PushIn,
                    }
                }
                if new.camera == Camera::Follow && !c.looks.first().is_some_and(|l| l.named) {
                    new.camera = Camera::Drift;
                }
                if let Some(slot) = plan.shots.last_mut() {
                    *slot = new;
                }
                out.push(format!("the ending: {was} → {} ({}): from the trip's last part", c.id, c.caption.chars().take(60).collect::<String>()));
            }
        }
    }
    out
}

/// Whether a shot's own voice is heard over the music (and captioned): someone talks in it, and,
/// in a places edit, the shot shows who (not the street while the holder talks off-screen).
pub fn voiced(focus: Option<Focus>, c: &Candidate, s: &PlannedShot) -> bool {
    (c.talking || c.said.is_some()) && (focus != Some(Focus::Places) || shows(c, s, false))
}

/// Whether look `id` of `c` is at a person (`holder`: the camera's holder only).
fn on_person(c: &Candidate, id: Option<&str>, holder_only: bool) -> bool {
    let Some(l) = id.and_then(|id| c.looks.iter().find(|l| l.id == id)) else { return false };
    if holder_only { at_holder(l, &c.looks) } else { at_person(l, &c.looks) }
}

/// Whether shot `s` of candidate `c` shows a person (`holder`: the camera's holder): a 360°
/// shot looking at one where it starts or where its whip ends, or a flat shot of someone close.
fn shows(c: &Candidate, s: &PlannedShot, holder_only: bool) -> bool {
    match c.kind {
        Kind::Vr => on_person(c, s.look.as_deref(), holder_only) || on_person(c, s.to.as_deref(), holder_only),
        _ => !holder_only && c.scores.people >= 0.6 && c.subject.is_some_and(|b| b[2] * b[3] >= 0.12),
    }
}

/// A places edit with the camera's holder never the shot's subject and anyone else in at most a
/// third: past that, a 360° shot turns to its place — its whip on to another place, or
/// a push in; a follow drifts. Orbits drift too (they sweep past the holder). Returns what
/// changed.
pub fn limit_people(plan: &mut Plan, cands: &[Candidate]) -> Vec<String> {
    let n = plan.shots.len();
    let allowed = ((n as f64 * MAX_PEOPLE).floor() as usize).max(1);
    let mut people = 0usize;
    let mut out = Vec::new();
    for (k, s) in plan.shots.iter_mut().enumerate() {
        let Some(c) = cands.iter().find(|c| c.id == s.clip) else { continue };
        if c.kind == Kind::Vr && s.camera == Camera::Orbit {
            s.camera = Camera::Drift;
            out.push(format!("shot {}: an orbit sweeps past the camera's holder: a drift", k + 1));
        }
        if !shows(c, s, false) {
            continue;
        }
        // the camera's holder is never what a shot looks at (they are at its edges often enough);
        // other people, a third of the shots
        if people < allowed && !shows(c, s, true) {
            people += 1;
            continue;
        }
        let looks = &c.looks;
        let Some(place) = looks.iter().find(|l| !at_person(l, looks)) else { continue };
        let was = s.look.clone().unwrap_or_default();
        let ended = s.to.clone();
        s.look = Some(place.id.clone());
        if s.to.as_deref().is_some_and(|t| t == place.id || looks.iter().find(|l| l.id == t).is_none_or(|l| at_person(l, looks))) {
            s.to = looks.iter().find(|l| l.id != place.id && !at_person(l, looks) && angle_between(l.yaw, place.yaw) >= LOOKS_APART).map(|l| l.id.clone());
        }
        // a whip goes on to another place, or the shot just pushes in on this one
        if s.camera == Camera::WhipPan && s.to.is_none() {
            s.camera = Camera::PushIn;
        }
        // a follow keeps a named subject in view: a view away from the person drifts instead
        if s.camera == Camera::Follow && !place.named {
            s.camera = Camera::Drift;
        }
        if was == place.id {
            out.push(format!("shot {}: no longer ends on the person ({}): a travel video shows the places", k + 1, ended.unwrap_or_default()));
        } else {
            out.push(format!("shot {}: looks at {} ({}), not {was}: a travel video shows the places", k + 1, place.id, place.what));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn people_and_places_are_told_apart() {
        assert!(is_person("a man in a patterned shirt") && is_person("Someone close to the lens") && is_person("a woman waving"));
        assert!(!is_person("a crowd of people walking") && !is_person("the Taksim mosque") && !is_person("a red tram") && !is_person("something moving"));
        assert_eq!(brief_focus("a reel of our weekend in Istanbul, the city at night"), Some(Focus::Places));
        assert_eq!(brief_focus("our son's birthday party"), Some(Focus::People));
        assert_eq!(brief_focus("30 s, catchy"), None);
        assert_eq!(Focus::parse("auto"), Ok(None));
        assert_eq!(Focus::parse(" Places "), Ok(Some(Focus::Places)));
        assert!(Focus::parse("dogs").is_err());
        // the holder is the person below the lens, not the one at eye level
        let l = |id: &str, yaw: f64, pitch: f64, what: &str| Look { id: id.into(), yaw, pitch, what: what.into(), named: true, size: None };
        let looks = [l("a", 86.0, -2.0, "man in white shirt"), l("b", -126.0, -28.0, "man"), l("c", 60.0, 0.0, "a food cart")];
        assert_eq!(holder(&looks).map(|h| h.id.as_str()), Some("b"));
        assert!(at_person(&looks[2], &looks), "the cart is next to the man in white");
        assert!(!at_holder(&looks[2], &looks));
    }
}
