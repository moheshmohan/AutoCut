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
//!   smiles and speech counting for little ([`super::rank::Scores::rank_for_places`]).
//! - **Where a 360° shot looks**, for places: at what is not a person first ([`is_person`]: a
//!   man, a woman, someone close to the lens… — a crowd or people walking are the street, a
//!   place); where only a person was found, the view the other way round (away from the camera's
//!   holder). The looks keep their ids, so a plan names them as before.
//! - **How many shots have the person in them**, for places: at most a third ([`MAX_PEOPLE`]);
//!   past that, a 360° shot is turned to its place ([`limit_people`]), its camera kept if it
//!   still fits (a follow needs a named subject: else a drift; a whip needs a second place: else
//!   a push in).

use serde::Serialize;

use super::{Camera, Candidate, Kind, LOOKS_APART, Look, Plan, angle_between};

/// Shots of people a places edit keeps, a share of its shots (at least one).
pub const MAX_PEOPLE: f64 = 1.0 / 3.0;
/// A person this big on the sphere (degrees across) is right by the lens: the camera's holder.
const HOLDER_SIZE: f64 = 40.0;
/// A direction this close to a person is looking at them (degrees).
const NEAR: f64 = 40.0;
/// Footage this much outdoors (a share of its moments) is about places, when people talk in
/// fewer than [`TALKING`] of them.
const OUTDOORS: f64 = 0.6;
const TALKING: f64 = 0.3;

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

/// Whether look `l` of a shot is at a person: described as one, or close to one that is.
fn at_person(l: &Look, people: &[&Look]) -> bool {
    is_person(&l.what) || people.iter().any(|p| angle_between(p.yaw, l.yaw) < NEAR)
}

fn wrap(yaw: f64) -> f64 {
    let y = (yaw + 180.0).rem_euclid(360.0) - 180.0;
    if y.is_finite() { y } else { 0.0 }
}

/// The candidates of a places edit: ranked for places, a 360° shot's place looks first (and a
/// look away from the person where it had none). Their looks keep their ids.
pub fn for_places(cands: &mut [Candidate]) {
    for c in cands.iter_mut() {
        c.rank = c.scores.rank_for_places(c.kind, c.wide, c.indoor);
        if c.kind != Kind::Vr || c.looks.is_empty() {
            continue;
        }
        let people: Vec<Look> = c.looks.iter().filter(|l| is_person(&l.what)).cloned().collect();
        let refs: Vec<&Look> = people.iter().collect();
        let mut place: Vec<Look> = c.looks.iter().filter(|l| !at_person(l, &refs)).cloned().collect();
        if place.is_empty() {
            // only people found: the view the other way round from the nearest of them (the
            // biggest is the camera's holder)
            let holder = people.iter().max_by(|a, b| a.size.map_or(0.0, |s| s.0).total_cmp(&b.size.map_or(0.0, |s| s.0)));
            if let Some(h) = holder {
                let id = format!("{}{}", c.id, char::from(b'a' + (c.looks.len().min(25) as u8)));
                let what = if h.size.is_some_and(|s| s.0 >= HOLDER_SIZE) { "the view away from the camera's holder" } else { "the view away from the person" };
                place.push(Look { id, yaw: wrap(h.yaw + 180.0), pitch: 0.0, what: what.into(), named: false, size: None });
            }
        }
        let rest: Vec<Look> = c.looks.iter().filter(|l| !place.iter().any(|p| p.id == l.id)).cloned().collect();
        c.looks = place.into_iter().chain(rest).collect();
    }
}

/// Whether shot `look` of candidate `c` has the person in it: a 360° shot looking at one, or a
/// flat shot of someone close.
fn shows_person(c: &Candidate, look: Option<&str>) -> bool {
    match c.kind {
        Kind::Vr => {
            let people: Vec<&Look> = c.looks.iter().filter(|l| is_person(&l.what)).collect();
            look.and_then(|id| c.looks.iter().find(|l| l.id == id)).is_some_and(|l| at_person(l, &people))
        }
        _ => c.scores.people >= 0.6 && c.subject.is_some_and(|b| b[2] * b[3] >= 0.12),
    }
}

/// A places edit with the person in at most a third of its shots: past that, its 360° shots
/// looking at them turn to their place (the opener kept: it introduces the trip). Returns what
/// changed.
pub fn limit_people(plan: &mut Plan, cands: &[Candidate]) -> Vec<String> {
    let n = plan.shots.len();
    let allowed = ((n as f64 * MAX_PEOPLE).floor() as usize).max(1);
    let mut kept = 0usize;
    let mut out = Vec::new();
    for (k, s) in plan.shots.iter_mut().enumerate() {
        let Some(c) = cands.iter().find(|c| c.id == s.clip) else { continue };
        if !shows_person(c, s.look.as_deref()) {
            continue;
        }
        if kept < allowed {
            kept += 1;
            continue;
        }
        let people: Vec<&Look> = c.looks.iter().filter(|l| is_person(&l.what)).collect();
        let Some(place) = c.looks.iter().find(|l| !at_person(l, &people)) else { continue };
        let was = s.look.clone().unwrap_or_default();
        s.look = Some(place.id.clone());
        // a whip goes on to another place, or the shot just pushes in on this one
        let to_person = s.to.as_deref().is_some_and(|t| t == place.id || c.looks.iter().find(|l| l.id == t).is_none_or(|l| at_person(l, &people)));
        if to_person {
            s.to = c.looks.iter().find(|l| l.id != place.id && !at_person(l, &people) && angle_between(l.yaw, place.yaw) >= LOOKS_APART).map(|l| l.id.clone());
        }
        if s.camera == Camera::WhipPan && s.to.is_none() {
            s.camera = Camera::PushIn;
        }
        // a follow keeps a named subject in view: a view away from the person drifts instead
        if s.camera == Camera::Follow && !place.named {
            s.camera = Camera::Drift;
        }
        out.push(format!("shot {}: looks at {} ({}), not {was}: a travel video shows the places", k + 1, place.id, place.what));
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
    }
}
