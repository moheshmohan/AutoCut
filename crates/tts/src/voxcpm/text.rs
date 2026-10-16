//! Text for VoxCPM: a narration script → utterances to speak and pauses.
//!
//! VoxCPM reads ordinary written text (punctuation guides its prosody), so the script is only
//! normalised where the written form is ambiguous to say: `filmcraft-tts-text`'s normaliser
//! spells numbers, years ("2022" → "twenty twenty-two"), money, times, dates, ordinals,
//! abbreviations, symbols and letter-by-letter acronyms, and its words are joined back into a
//! sentence with the punctuation in place ("Go1" → "Go one", "Dr. Smith" → "doctor Smith").
//!
//! Pause markers (`[pause 1s]`) become exact silence. Sentences are grouped into utterances of
//! at most [`MAX_UTTERANCE_CHARS`] (a long sentence is split at commas, else between words):
//! each utterance is generated on its own after the same speaker prompt, which keeps the voice
//! the same from the first sentence to the last and the model within its context.

use filmcraft_tts_text::{Lang, Token};

use crate::script::{self, Segment};
use crate::{TtsError, check_text};

/// Longest utterance generated in one pass (about 15 s of speech).
pub const MAX_UTTERANCE_CHARS: usize = 240;
/// Most utterances one script makes (a 64 KB script is at most a few hundred).
const MAX_PIECES: usize = 4096;

/// A piece of a script.
#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    /// Text to speak (normalised).
    Say(String),
    /// Silence, milliseconds.
    Pause(u32),
}

/// Longest style description handed to the model, in characters.
pub const MAX_STYLE_CHARS: usize = 120;

/// VoxCPM2's spoken-style control: a description of how to speak in parentheses directly in
/// front of the utterance ("(warm, friendly)Hello."). Only letters, digits, spaces and light
/// punctuation are kept (a bracket would end the description early), at most
/// [`MAX_STYLE_CHARS`]; empty when nothing usable is left.
pub fn style_prefix(style: &str) -> String {
    let kept: String = style.chars().filter(|c| c.is_alphanumeric() || matches!(c, ' ' | ',' | '-' | '\'' | '.')).take(MAX_STYLE_CHARS).collect();
    let words = kept.split_whitespace().collect::<Vec<_>>().join(" ");
    let words = words.trim_matches([',', '.', '-', ' ']);
    if words.chars().any(char::is_alphanumeric) { format!("({words})") } else { String::new() }
}

/// Normalise one stretch of speech into sentences.
pub fn sentences(text: &str) -> Result<Vec<String>, TtsError> {
    let tokens = filmcraft_tts_text::normalize(text, Lang::EnUs).map_err(|e| TtsError::Invalid(e.to_string()))?;
    let mut out = Vec::new();
    let mut cur = String::new();
    // an opening bracket or quote waits for the next word
    let mut open = String::new();
    let mut glue = false;
    let end = |cur: &mut String, out: &mut Vec<String>, mark: Option<char>| {
        let t = cur.trim().to_string();
        cur.clear();
        if t.chars().any(char::is_alphanumeric) {
            let mut t = t.trim_end_matches([',', ';', ':', '-']).to_string();
            if let Some(m) = mark {
                t.push(m);
            }
            out.push(t);
        }
    };
    for tok in tokens {
        match tok {
            Token::Word(w) => {
                if !cur.is_empty() && !glue {
                    cur.push(' ');
                }
                cur.push_str(&open);
                open.clear();
                cur.push_str(&w);
                glue = false;
            }
            Token::Punct(c) => match c {
                ',' | ';' | ':' | ')' | ']' | '}' | '%' | '…' => {
                    if !cur.is_empty() && !cur.ends_with(c) {
                        cur.push(c);
                    }
                    glue = false;
                }
                '(' | '[' | '{' | '“' | '‘' => open.push(c),
                '-' | '\'' | '’' => {
                    if !cur.is_empty() {
                        cur.push(c);
                        glue = true;
                    }
                }
                '—' | '–' => {
                    if !cur.is_empty() {
                        cur.push_str(" —");
                    }
                    glue = false;
                }
                // quotes and anything else carry no sound of their own
                _ => {}
            },
            Token::SentenceEnd => end(&mut cur, &mut out, Some('.')),
            Token::SentenceEndWith(c) => end(&mut cur, &mut out, Some(c)),
            Token::Pause { .. } => end(&mut cur, &mut out, None),
        }
    }
    end(&mut cur, &mut out, Some('.'));
    Ok(out)
}

/// Split one long sentence at commas (else between words) into parts of at most `max` chars.
fn split_long(s: &str, max: usize) -> Vec<String> {
    if s.chars().count() <= max {
        return vec![s.to_string()];
    }
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for word in s.split_whitespace() {
        let would = cur.chars().count() + 1 + word.chars().count();
        let at_comma = cur.ends_with([',', ';', ':']);
        if !cur.is_empty() && (would > max || (at_comma && cur.chars().count() > max / 2)) {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        // a single word longer than the limit is cut (no real word is)
        let w: String = word.chars().take(max).collect();
        cur.push_str(&w);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// A script → utterances and pauses (see the module docs).
pub fn pieces(text: &str) -> Result<Vec<Piece>, TtsError> {
    check_text(text)?;
    let mut out: Vec<Piece> = Vec::new();
    for seg in script::parse(text) {
        match seg {
            Segment::Pause { millis } => out.push(Piece::Pause(millis)),
            Segment::Speech(t) => {
                let mut cur = String::new();
                for s in sentences(&t)? {
                    for part in split_long(&s, MAX_UTTERANCE_CHARS) {
                        if !cur.is_empty() && cur.chars().count() + 1 + part.chars().count() > MAX_UTTERANCE_CHARS {
                            out.push(Piece::Say(std::mem::take(&mut cur)));
                        }
                        if !cur.is_empty() {
                            cur.push(' ');
                        }
                        cur.push_str(&part);
                    }
                }
                if !cur.is_empty() {
                    out.push(Piece::Say(cur));
                }
            }
        }
        if out.len() > MAX_PIECES {
            return Err(TtsError::TextTooLong);
        }
    }
    if !out.iter().any(|p| matches!(p, Piece::Say(_))) {
        return Err(TtsError::Empty);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn style_prefixes_are_clean_and_bounded() {
        assert_eq!(style_prefix("warm, friendly, smiling"), "(warm, friendly, smiling)");
        // brackets would close the description early; whitespace collapses
        assert_eq!(style_prefix("  excited)  (shout [laughing] "), "(excited shout laughing)");
        assert_eq!(style_prefix("()[]{}"), "");
        assert_eq!(style_prefix(""), "");
        assert!(style_prefix(&"calm ".repeat(1000)).chars().count() <= MAX_STYLE_CHARS + 2);
    }

    fn say(text: &str) -> Vec<String> {
        pieces(text)
            .unwrap()
            .into_iter()
            .filter_map(|p| match p {
                Piece::Say(s) => Some(s),
                Piece::Pause(_) => None,
            })
            .collect()
    }

    #[test]
    fn numbers_years_and_names_are_spelled() {
        assert_eq!(
            sentences("In October 2022 I met Spot, the yellow robot dog from Boston Dynamics, walking through the crowd at GITEX in Dubai.").unwrap(),
            vec!["In October twenty twenty-two I met Spot, the yellow robot dog from Boston Dynamics, walking through the crowd at GITEX in Dubai."]
        );
        assert_eq!(sentences("Unitree's Go1 costs $2,700.").unwrap(), vec!["Unitree's Go one costs two thousand seven hundred dollars."]);
        assert_eq!(sentences("Dr. Smith met the FBI (twice)!").unwrap(), vec!["doctor Smith met the F B I (twice)!"]);
        assert_eq!(sentences("Is it 3:30? Yes.").unwrap(), vec!["Is it three thirty?", "Yes."]);
        assert_eq!(sentences("A well-known fact").unwrap(), vec!["A well-known fact."]);
        assert_eq!(sentences("\"Quoted\" text — with a dash").unwrap(), vec!["Quoted text — with a dash."]);
    }

    #[test]
    fn scripts_become_utterances_and_pauses() {
        let p = pieces("Hello there. [pause 1s] How are you?").unwrap();
        assert_eq!(p, vec![Piece::Say("Hello there.".into()), Piece::Pause(1000), Piece::Say("How are you?".into())]);
        // short sentences share an utterance; long ones are split, never past the limit
        let long = "This sentence is short. ".repeat(40);
        let u = say(&long);
        assert!(u.len() > 1);
        assert!(u.iter().all(|s| s.chars().count() <= MAX_UTTERANCE_CHARS), "{u:?}");
        assert_eq!(u.join(" ").matches("short.").count(), 40, "nothing lost");
        let one = format!("{}.", (0..120).map(|i| format!("word{i},")).collect::<Vec<_>>().join(" "));
        let u = say(&one);
        assert!(u.len() > 1 && u.iter().all(|s| s.chars().count() <= MAX_UTTERANCE_CHARS));
        // a word longer than an utterance is cut, not looped on
        let u = say(&"x".repeat(1000));
        assert!(u.iter().all(|s| s.chars().count() <= MAX_UTTERANCE_CHARS));
    }

    #[test]
    fn empty_and_hostile_scripts() {
        assert_eq!(pieces(""), Err(TtsError::Empty));
        assert_eq!(pieces("[pause 1s] ... !!"), Err(TtsError::Empty));
        assert_eq!(pieces(&"a ".repeat(40_000)), Err(TtsError::TextTooLong));
        // odd characters never panic
        for t in ["\u{0}\u{1}abc", "🙂🙂 hi", "日本語のテキスト", "[[[pause", "1e999999", "-0.0.0.0", "$$$$", "(((((a"] {
            let _ = pieces(t);
        }
    }
}
