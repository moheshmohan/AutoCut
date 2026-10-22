//! Reading a vision-language model's answer: the first balanced JSON object in it (models wrap
//! JSON in prose and code fences), else a lenient reading of what is not quite JSON (cut off at
//! the token limit, a list left open, a stray value), and every field of a [`ShotDescription`]
//! read leniently — a string or a list, a number or a word — and capped. A model's answer is hostile input: nothing in it can make reading fail, it can only
//! make the description emptier.

use filmcraft_project::scenes::{Energy, MAX_LIST, MAX_TEXT, ShotDescription, ShotType, ShotUse, Subject};
use serde_json::Value;

/// Longest answer looked at (characters).
pub const MAX_ANSWER: usize = 64 * 1024;
/// Deepest nesting the lenient reader follows.
const MAX_DEPTH: usize = 32;

/// The first `{`-balanced JSON object in `s` (models wrap JSON in prose or code fences despite
/// being asked not to); the whole string when it already parses.
pub fn extract_json(s: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str::<Value>(s.trim()) {
        return Some(v);
    }
    let start = s.find('{')?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (i, b) in s.as_bytes().get(start..)?.iter().enumerate() {
        let c = *b;
        if in_string {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_string = false;
            }
            continue;
        }
        match c {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str::<Value>(s.get(start..start + i + 1)?).ok();
                }
            }
            _ => {}
        }
    }
    None
}

/// The object an answer holds: [`extract_json`], or — for an answer that is not quite JSON —
/// what a lenient reading makes of it ([`lenient`]). `None` when there is no object to speak of.
pub fn extract_or_repair(s: &str) -> Option<Value> {
    let s: String = s.chars().take(MAX_ANSWER).collect();
    if let Some(v) = extract_json(&s).filter(Value::is_object) {
        return Some(v);
    }
    let start = s.find('{')?;
    lenient(s.get(start..)?).filter(|v| v.as_object().is_some_and(|o| !o.is_empty()))
}

/// A forgiving reader for what small models write when they mean JSON. It reads from the first
/// `{` and keeps going where a strict parser stops:
/// - the answer ends early (the token limit): open strings, lists and objects are closed, a key
///   without a value is dropped;
/// - a list is left open before the object's next key (`[{..}, {..}, "actions": [..]`): a string
///   followed by `:` inside a list ends the list;
/// - stray values inside an object (`{"name": "x", "...", "box": [..]}`), missing commas, stray
///   characters: skipped;
/// - a number JSON cannot hold (`1e309`): null.
///
/// Nesting is capped; the result is always a valid value.
pub fn lenient(s: &str) -> Option<Value> {
    let mut r = Lenient { b: s.as_bytes(), i: 0 };
    r.skip_to(b'{')?;
    match r.value(0) {
        Some(v @ Value::Object(_)) => Some(v),
        _ => None,
    }
}

struct Lenient<'a> {
    b: &'a [u8],
    i: usize,
}

impl Lenient<'_> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn ws(&mut self) {
        while self.peek().is_some_and(|c| c.is_ascii_whitespace()) {
            self.i += 1;
        }
    }

    fn skip_to(&mut self, c: u8) -> Option<()> {
        while self.peek()? != c {
            self.i += 1;
        }
        Some(())
    }

    /// A string at a `"` (the quote consumed); closed at the end of the input if need be.
    fn string(&mut self) -> String {
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        while let Some(c) = self.peek() {
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => match self.peek() {
                    Some(e) => {
                        self.i += 1;
                        match e {
                            b'n' => out.push(b'\n'),
                            b't' => out.push(b'\t'),
                            b'r' => out.push(b'\r'),
                            b'b' | b'f' => out.push(b' '),
                            b'u' => {
                                let hex =
                                    self.b.get(self.i..self.i + 4).and_then(|h| std::str::from_utf8(h).ok()).and_then(|h| u32::from_str_radix(h, 16).ok());
                                match hex.and_then(char::from_u32) {
                                    Some(ch) => {
                                        self.i += 4;
                                        let mut buf = [0u8; 4];
                                        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                                    }
                                    None => out.push(b'?'),
                                }
                            }
                            other => out.push(other),
                        }
                    }
                    None => break,
                },
                _ => out.push(c),
            }
            if out.len() > MAX_ANSWER {
                break;
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    /// Whether the string starting here is a key (a `:` follows it).
    fn at_key(&self) -> bool {
        if self.peek() != Some(b'"') {
            return false;
        }
        let mut j = self.i + 1;
        let mut escaped = false;
        while let Some(&c) = self.b.get(j) {
            j += 1;
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                break;
            }
        }
        while self.b.get(j).is_some_and(|c| c.is_ascii_whitespace()) {
            j += 1;
        }
        self.b.get(j) == Some(&b':')
    }

    fn number(&mut self) -> Value {
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit() || matches!(c, b'-' | b'+' | b'.' | b'e' | b'E')) {
            self.i += 1;
        }
        let text = std::str::from_utf8(self.b.get(start..self.i).unwrap_or_default()).unwrap_or_default();
        match text.parse::<f64>() {
            Ok(f) if f.is_finite() => {
                if f.fract() == 0.0 && f.abs() < 9.0e15 {
                    Value::from(f as i64)
                } else {
                    serde_json::Number::from_f64(f).map_or(Value::Null, Value::Number)
                }
            }
            _ => Value::Null,
        }
    }

    fn word(&mut self) -> Value {
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
            self.i += 1;
        }
        match self.b.get(start..self.i) {
            Some(b"true") => Value::Bool(true),
            Some(b"false") => Value::Bool(false),
            _ => Value::Null,
        }
    }

    /// The value here, or `None` when there is none (the end, or a closing bracket).
    fn value(&mut self, depth: usize) -> Option<Value> {
        self.ws();
        match self.peek()? {
            b'{' if depth < MAX_DEPTH => Some(self.object(depth + 1)),
            b'[' if depth < MAX_DEPTH => Some(self.list(depth + 1)),
            b'{' | b'[' => {
                self.i += 1;
                Some(Value::Null)
            }
            b'"' => Some(Value::String(self.string())),
            b'-' | b'0'..=b'9' => Some(self.number()),
            c if c.is_ascii_alphabetic() => Some(self.word()),
            _ => None,
        }
    }

    fn object(&mut self, depth: usize) -> Value {
        self.i += 1;
        let mut map = serde_json::Map::new();
        loop {
            self.ws();
            match self.peek() {
                None => break,
                Some(b'}') => {
                    self.i += 1;
                    break;
                }
                Some(b',') => self.i += 1,
                Some(b'"') => {
                    let key = self.string();
                    self.ws();
                    if self.peek() != Some(b':') {
                        // a stray string where a key should be
                        continue;
                    }
                    self.i += 1;
                    self.ws();
                    match self.value(depth) {
                        Some(v) => {
                            map.insert(key, v);
                        }
                        None => {
                            // no value (the end, or junk): the key goes
                            if self.peek().is_some_and(|c| c != b'}' && c != b',') {
                                self.i += 1;
                            }
                        }
                    }
                }
                Some(_) => self.i += 1,
            }
        }
        Value::Object(map)
    }

    /// A list at a `[`. It ends at its `]`, at the end of the input, at a `}` (the object around
    /// it closes; the list was never closed) or at a key (`"name":`: the object around it goes on
    /// from there).
    fn list(&mut self, depth: usize) -> Value {
        self.i += 1;
        let mut items = Vec::new();
        loop {
            self.ws();
            match self.peek() {
                None | Some(b'}') => break,
                Some(b']') => {
                    self.i += 1;
                    break;
                }
                Some(b',') => self.i += 1,
                Some(b'"') if self.at_key() => break,
                Some(_) => match self.value(depth) {
                    Some(v) => items.push(v),
                    None => self.i += 1,
                },
            }
        }
        Value::Array(items)
    }
}

/// One line of text from a model: control characters out, single spaces, capped.
pub fn clean(s: &str, max: usize) -> String {
    let s: String = s.chars().take(max.saturating_mul(4).max(16)).map(|c| if c.is_control() { ' ' } else { c }).collect();
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    let s: String = s.chars().take(max).collect();
    let low = s.trim().to_lowercase();
    if matches!(low.as_str(), "none" | "n/a" | "na" | "null" | "unknown" | "nothing" | "-" | "") { String::new() } else { s.trim().to_string() }
}

fn field<'a>(v: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    let o = v.as_object()?;
    keys.iter().find_map(|k| o.get(*k).filter(|x| !x.is_null()))
}

fn text(v: &Value, keys: &[&str], max: usize) -> String {
    match field(v, keys) {
        Some(Value::String(s)) => clean(s, max),
        Some(Value::Array(a)) => clean(&a.iter().take(MAX_LIST).filter_map(Value::as_str).collect::<Vec<_>>().join(", "), max),
        Some(Value::Number(n)) => n.to_string(),
        _ => String::new(),
    }
}

/// A list of short strings: a JSON list (of strings, or of objects with a `text`/`name`), or one
/// string split at commas and semicolons.
fn list(v: &Value, keys: &[&str]) -> Vec<String> {
    let items: Vec<String> = match field(v, keys) {
        Some(Value::Array(a)) => a
            .iter()
            .take(MAX_LIST * 4)
            .filter_map(|x| match x {
                Value::String(s) => Some(s.clone()),
                Value::Object(_) => field(x, &["text", "name", "action", "label"]).and_then(Value::as_str).map(str::to_string),
                Value::Number(n) => Some(n.to_string()),
                _ => None,
            })
            .collect(),
        Some(Value::String(s)) => s.split([',', ';']).map(str::to_string).collect(),
        _ => Vec::new(),
    };
    let mut out: Vec<String> = Vec::new();
    for s in items {
        let s = clean(&s, 200);
        if !s.is_empty() && !out.iter().any(|o| o.eq_ignore_ascii_case(&s)) && out.len() < MAX_LIST {
            out.push(s);
        }
    }
    out
}

/// A count: a number, or a number in words ("3", "three", "a few").
fn count(v: &Value, keys: &[&str]) -> u32 {
    let n = match field(v, keys) {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0),
        Some(Value::String(s)) => {
            let s = s.trim().to_lowercase();
            const WORDS: [&str; 11] = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
            s.parse::<f64>().ok().or_else(|| WORDS.iter().position(|w| s.starts_with(w)).map(|p| p as f64)).unwrap_or(match s.as_str() {
                "a few" | "few" | "several" => 3.0,
                "many" | "crowd" | "a crowd" | "lots" => 12.0,
                _ => 0.0,
            })
        }
        Some(Value::Array(a)) => a.len() as f64,
        _ => 0.0,
    };
    if n.is_finite() { n.clamp(0.0, 10_000.0) as u32 } else { 0 }
}

fn flag(v: &Value, keys: &[&str]) -> bool {
    match field(v, keys) {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => matches!(s.trim().to_lowercase().as_str(), "yes" | "true" | "y" | "1"),
        Some(Value::Number(n)) => n.as_f64().is_some_and(|x| x > 0.0),
        _ => false,
    }
}

/// A box from a model: `[x1, y1, x2, y2]` in Qwen's 0..1000 grid (or already 0..1), as
/// `[x, y, w, h]` shares of the picture; `None` when it makes no sense.
pub fn read_box(v: Option<&Value>) -> Option<[f32; 4]> {
    let a = v?.as_array()?;
    let c: Vec<f64> = a.iter().take(5).filter_map(Value::as_f64).collect();
    let [x1, y1, x2, y2] = c.as_slice() else { return None };
    if ![x1, y1, x2, y2].iter().all(|v| v.is_finite()) {
        return None;
    }
    let scale = if [x1, y1, x2, y2].iter().all(|v| **v <= 1.0) { 1.0 } else { 1000.0 };
    let n = |v: f64| (v / scale).clamp(0.0, 1.0);
    let (x1, x2) = (n(x1.min(*x2)), n(x1.max(*x2)));
    let (y1, y2) = (n(y1.min(*y2)), n(y1.max(*y2)));
    (x2 - x1 > 0.002 && y2 - y1 > 0.002).then_some([x1 as f32, y1 as f32, (x2 - x1) as f32, (y2 - y1) as f32])
}

fn subjects(v: &Value) -> Vec<Subject> {
    let mut out: Vec<Subject> = Vec::new();
    let split: Vec<Value>;
    let items: Vec<&Value> = match field(v, &["subjects", "subject", "objects"]) {
        Some(Value::Array(a)) => a.iter().take(MAX_LIST * 4).collect(),
        Some(Value::String(s)) => {
            split = s.split([',', ';']).take(MAX_LIST * 4).map(|p| Value::String(p.to_string())).collect();
            split.iter().collect()
        }
        Some(x @ Value::Object(_)) => vec![x],
        _ => Vec::new(),
    };
    for x in items {
        let s = match x {
            Value::String(s) => Subject { name: clean(s, 120), bbox: None },
            Value::Object(_) => Subject {
                name: text(x, &["name", "label", "subject", "what", "description"], 120),
                bbox: read_box(field(x, &["box", "bbox", "bbox_2d", "box_2d"])),
            },
            _ => continue,
        };
        if !s.name.is_empty() && out.len() < MAX_LIST && !out.iter().any(|o| o.name.eq_ignore_ascii_case(&s.name)) {
            out.push(s);
        }
    }
    out
}

fn uses(v: &Value) -> Vec<ShotUse> {
    let mut out = Vec::new();
    for s in list(v, &["uses", "use", "suggested_uses", "suggestedUses", "good_for"]) {
        if let Some(u) = ShotUse::from_words(&s)
            && !out.contains(&u)
        {
            out.push(u);
        }
    }
    out
}

/// A shot description from a model's answer (see [`crate::prompt`] for the fields asked for).
/// `detail`: keep the paragraph. Fails only when the answer holds no object at all.
pub fn parse_description(answer: &str, detail: bool) -> Result<ShotDescription, String> {
    let v = extract_or_repair(answer).ok_or_else(|| format!("no JSON object in the answer: {}", clean(answer, 120)))?;
    let mut d = ShotDescription {
        shot_type: ShotType::from_words(&text(&v, &["shot_type", "shotType", "shot", "framing"], 60)),
        subjects: subjects(&v),
        actions: list(&v, &["actions", "action", "activity"]),
        setting: text(&v, &["setting", "location", "place", "scene"], 200),
        mood: text(&v, &["mood", "tone", "atmosphere"], 80),
        energy: Energy::from_words(&text(&v, &["energy", "pace"], 40)),
        text: list(&v, &["text", "on_screen_text", "onScreenText", "visible_text"]),
        people: count(&v, &["people", "people_count", "peopleCount", "persons"]),
        faces_prominent: flag(&v, &["faces_prominent", "facesProminent", "faces"]),
        caption: text(&v, &["caption", "summary", "one_sentence"], 300),
        detail: if detail { text(&v, &["detail", "details", "description", "paragraph"], MAX_TEXT) } else { String::new() },
        uses: uses(&v),
    };
    // a caption the model left out: its first sentence of detail
    if d.caption.is_empty() {
        let from = text(&v, &["detail", "details", "description", "paragraph"], MAX_TEXT);
        d.caption = from.split_inclusive(['.', '!', '?']).next().map(|s| clean(s, 300)).unwrap_or_default();
    }
    if d.people == 0 && d.faces_prominent {
        d.people = 1;
    }
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_json_from_prose_and_fences() {
        assert_eq!(extract_json(r#"{"a":1}"#), Some(json!({"a": 1})));
        assert_eq!(extract_json("Sure! Here it is:\n```json\n{\"a\": {\"b\": 2}}\n```"), Some(json!({"a": {"b": 2}})));
        assert_eq!(extract_json(r#"the plan is {"a":"with } brace","b":3} hope it helps"#), Some(json!({"a": "with } brace", "b": 3})));
        assert_eq!(extract_json("no json here"), None);
        assert_eq!(extract_json(r#"{"unclosed": 1"#), None);
    }

    #[test]
    fn truncated_answers_keep_what_was_said() {
        assert_eq!(
            extract_or_repair(r#"{"caption": "A robot waves", "uses": ["hook", "re"#),
            Some(json!({"caption": "A robot waves", "uses": ["hook", "re"]}))
        );
        assert_eq!(extract_or_repair(r#"```json {"a": 1, "b":"#), Some(json!({"a": 1})), "a key without a value goes");
        assert_eq!(extract_or_repair(r#"{"a": 1, "b": 12"#), Some(json!({"a": 1, "b": 12})));
        assert_eq!(extract_or_repair(r#"{"a": {"b": [1, 2"#), Some(json!({"a": {"b": [1, 2]}})));
        assert_eq!(extract_or_repair(r#"{"a": "x\"#), Some(json!({"a": "x"})));
        // cut inside a key: the key goes
        assert_eq!(extract_or_repair(r#"{"a": 1, "cap"#), Some(json!({"a": 1})));
        assert_eq!(extract_or_repair("nothing"), None);
        assert_eq!(extract_or_repair("[1, 2]"), None, "an answer must be an object");
        assert_eq!(extract_or_repair("{]"), None, "an empty object is no answer");
        assert_eq!(extract_or_repair(r#"{"a": tru"#), Some(json!({"a": null})));
        assert_eq!(extract_or_repair(r#"{"a": "\u00e9t\u00e9", "b": [1 2 3]}"#), Some(json!({"a": "été", "b": [1, 2, 3]})), "escapes, missing commas");
        // a list left open before the next key
        assert_eq!(
            extract_or_repair(r#"{"subjects": [{"name": "a"}, {"name": "b"}, "actions": ["sit"], "caption": "x"}"#),
            Some(json!({"subjects": [{"name": "a"}, {"name": "b"}], "actions": ["sit"], "caption": "x"}))
        );
        assert_eq!(extract_or_repair(r#"{"t": ["a, b", "c"], "k": 1}"#), Some(json!({"t": ["a, b", "c"], "k": 1})), "commas in strings are not the list's");
    }

    #[test]
    fn a_full_answer_reads() {
        let a = r#"Here is the JSON:
```json
{"shot_type": "Medium shot", "camera": "static",
 "subjects": [{"name": "white humanoid robot", "box": [120, 80, 560, 990]}, "visitor"],
 "actions": ["robot raises its arm", "visitor films it"], "setting": "exhibition hall",
 "mood": "curious", "energy": "high", "text": ["GITEX 2024"], "people": 2, "faces_prominent": "yes",
 "caption": "A white humanoid robot raises its arm for a visitor.", "detail": "A long paragraph.",
 "uses": ["hook", "B-roll", "talking head", "nonsense"]}
```"#;
        let d = parse_description(a, true).unwrap();
        assert_eq!(d.shot_type, ShotType::Medium);
        assert_eq!(d.subjects.len(), 2);
        assert_eq!(d.subjects[0].name, "white humanoid robot");
        let b = d.subjects[0].bbox.unwrap();
        assert!((b[0] - 0.12).abs() < 1e-6 && (b[2] - 0.44).abs() < 1e-6 && (b[3] - 0.91).abs() < 1e-6, "{b:?}");
        assert_eq!(d.subjects[1].bbox, None);
        assert_eq!(d.actions.len(), 2);
        assert_eq!(d.energy, Energy::High);
        assert_eq!(d.text, vec!["GITEX 2024"]);
        assert_eq!(d.people, 2);
        assert!(d.faces_prominent);
        assert_eq!(d.uses, vec![ShotUse::Hook, ShotUse::BRoll, ShotUse::TalkingHead]);
        assert_eq!(d.detail, "A long paragraph.");
        // without `detail` the paragraph is not kept
        assert_eq!(parse_description(a, false).unwrap().detail, "");
    }

    #[test]
    fn hostile_and_partial_answers_never_fail_badly() {
        assert!(parse_description("I cannot describe this image.", false).is_err());
        let d = parse_description(r#"{"caption": 5, "people": "three", "subjects": "a dog, a cat", "uses": "hook; reveal", "actions": null}"#, false).unwrap();
        assert_eq!(d.caption, "5");
        assert_eq!(d.people, 3);
        assert_eq!(d.subjects.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["a dog", "a cat"]);
        assert_eq!(d.uses, vec![ShotUse::Hook, ShotUse::Reveal]);
        assert!(d.actions.is_empty());
        // absurd numbers, boxes and sizes
        let huge = "x".repeat(100_000);
        let a = format!(
            r#"{{"people": 1e300, "subjects": [{{"name": "a", "box": [0, 0, "x", 5]}}, {{"name": "b", "box": [900, 900, 100, 100]}}], "caption": "{huge}", "text": [{}]}}"#,
            (0..500).map(|i| format!("\"line {i}\"")).collect::<Vec<_>>().join(",")
        );
        let d = parse_description(&a, true).unwrap();
        assert_eq!(d.people, 10_000);
        assert_eq!(d.subjects[0].bbox, None, "a box that is not four numbers");
        let b = d.subjects[1].bbox.unwrap();
        assert!((b[0] - 0.1).abs() < 1e-6 && (b[2] - 0.8).abs() < 1e-6, "reversed corners are put in order: {b:?}");
        assert!(d.caption.chars().count() <= 300);
        assert!(d.text.len() <= MAX_LIST);
        // a number JSON cannot hold reads as nothing; the fields around it are kept
        let d = parse_description(r#"{"caption": "kept", "people": 1e309, "mood": "kept too"}"#, false).unwrap();
        assert_eq!((d.caption.as_str(), d.mood.as_str(), d.people), ("kept", "kept too", 0));
        // a stray value inside an object (seen from Qwen3-VL 2B)
        let d = parse_description(
            r#"{"shot_type": "wide", "subjects": [{"name": "man"}, {"name": "woman", "...", "box": [450, 380, 600, 500]}], "caption": "A selfie.", "people": 4}"#,
            false,
        )
        .unwrap();
        assert_eq!((d.subjects.len(), d.caption.as_str(), d.people), (2, "A selfie.", 4));
        assert!(d.subjects[1].bbox.is_some());
        // a caption taken from the paragraph; placeholders read as nothing
        let d = parse_description(r#"{"description": "A dog runs. Then it sits.", "setting": "N/A", "mood": "none"}"#, false).unwrap();
        assert_eq!(d.caption, "A dog runs.");
        assert_eq!((d.setting.as_str(), d.mood.as_str()), ("", ""));
        // faces without a count: someone is there
        assert_eq!(parse_description(r#"{"faces_prominent": true}"#, false).unwrap().people, 1);
    }

    #[test]
    fn mutated_answers_never_panic() {
        let base = r#"{"shot_type":"wide","subjects":[{"name":"robot","box":[1,2,300,400]}],"actions":["walks"],"setting":"street","mood":"calm","energy":"low","text":["SALE"],"people":4,"faces_prominent":false,"caption":"A robot walks down a street.","uses":["b-roll"]}"#;
        let bytes = base.as_bytes();
        for cut in 0..bytes.len() {
            let s = String::from_utf8_lossy(&bytes[..cut]);
            let _ = parse_description(&s, true);
        }
        let mut seed = 7u32;
        for _ in 0..2000 {
            let mut b = bytes.to_vec();
            for _ in 0..3 {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let i = (seed as usize >> 8) % b.len();
                b[i] = (seed >> 16) as u8;
            }
            let s = String::from_utf8_lossy(&b);
            let _ = parse_description(&s, true);
        }
        // deep nesting is refused, not recursed into
        let deep = format!("{}{}", "{\"a\":".repeat(10_000), "1");
        let _ = parse_description(&deep, false);
    }
}
