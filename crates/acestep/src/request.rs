//! What a generation asks for, checked, and the text the model is conditioned on.
//!
//! The layout is the official pipeline's for text-to-music without the LM planner
//! (`acestep/constants.py` `SFT_GEN_PROMPT`, `core/generation/handler/prompt_utils.py` and
//! `metadata_utils.py`): an instruction, the caption and the metadata block for the text encoder,
//! and a lyrics block (`[Instrumental]`, language `unknown`) whose token embeddings feed the
//! lyric encoder.

use crate::MusicError;

/// Shortest clip, seconds (the model works on at least 128 latent frames, 5.12 s).
pub const MIN_SECONDS: f64 = 5.0;
/// Longest clip, seconds (the model card says up to 10 minutes; capped to bound memory and time).
pub const MAX_SECONDS: f64 = 240.0;
/// Longest description, characters (the official limit).
pub const MAX_PROMPT_CHARS: usize = 512;
/// Tempo range accepted (the official one).
pub const BPM_RANGE: (u32, u32) = (30, 300);
/// Output sample rate (the VAE's).
pub const SAMPLE_RATE: u32 = 48_000;
/// Latent frames per second (48 kHz / 1920).
pub const FRAME_RATE: u32 = 25;
/// Samples per latent frame.
pub const HOP: usize = 1920;
/// Fewest latent frames generated (the official minimum); shorter clips are trimmed.
pub const MIN_FRAMES: usize = 128;

/// The DiT instruction for text to music.
pub const INSTRUCTION: &str = "Fill the audio semantic mask based on the given conditions:";
/// The lyrics block of an instrumental piece.
pub const INSTRUMENTAL_LYRICS: &str = "# Languages\nunknown\n\n# Lyric\n[Instrumental]<|endoftext|>";

/// One text-to-music generation.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    /// What the music is: genre, mood, instruments, tempo words ("calm lo-fi hip hop with soft
    /// piano and vinyl crackle").
    pub prompt: String,
    /// Length, [`MIN_SECONDS`]..=[`MAX_SECONDS`].
    pub seconds: f64,
    /// Seed of the diffusion noise: the same request and seed give the same audio.
    pub seed: u64,
    /// Tempo, beats per minute ([`BPM_RANGE`]); `None` lets the model choose.
    pub bpm: Option<u32>,
    /// Key, e.g. "A minor", "F# major", "Eb" (major), "Am"; `None` lets the model choose.
    pub key: Option<String>,
    /// Beats per bar: 2, 3, 4 or 6; `None` lets the model choose.
    pub time_signature: Option<u32>,
    /// Timestep shift of the turbo schedule: 1, 2 or 3 (3, the recommended value).
    pub shift: u32,
}

impl Request {
    /// A request with the defaults: model-chosen tempo, key and metre, shift 3.
    pub fn new(prompt: &str, seconds: f64, seed: u64) -> Request {
        Request { prompt: prompt.to_string(), seconds, seed, bpm: None, key: None, time_signature: None, shift: 3 }
    }

    /// Refuse what the model cannot do, with a message the user can act on.
    pub fn check(&self) -> Result<(), MusicError> {
        let p = clean_prompt(&self.prompt);
        if p.is_empty() {
            return Err(MusicError::Invalid("describe the music (genre, mood, instruments)".into()));
        }
        if self.prompt.chars().count() > MAX_PROMPT_CHARS {
            return Err(MusicError::Invalid(format!("the description is longer than {MAX_PROMPT_CHARS} characters")));
        }
        if !(self.seconds.is_finite() && (MIN_SECONDS..=MAX_SECONDS).contains(&self.seconds)) {
            return Err(MusicError::Invalid(format!("the length must be {MIN_SECONDS}–{MAX_SECONDS} seconds")));
        }
        if let Some(b) = self.bpm
            && !(BPM_RANGE.0..=BPM_RANGE.1).contains(&b)
        {
            return Err(MusicError::Invalid(format!("the tempo must be {}–{} BPM", BPM_RANGE.0, BPM_RANGE.1)));
        }
        if let Some(k) = &self.key {
            key_scale(k)?;
        }
        if let Some(t) = self.time_signature
            && ![2, 3, 4, 6].contains(&t)
        {
            return Err(MusicError::Invalid("the time signature must be 2, 3, 4 or 6 beats per bar".into()));
        }
        if !(1..=3).contains(&self.shift) {
            return Err(MusicError::Invalid("the shift must be 1, 2 or 3".into()));
        }
        Ok(())
    }

    /// Latent frames generated (at least [`MIN_FRAMES`]).
    pub fn frames(&self) -> usize {
        let f = (self.seconds.clamp(0.0, MAX_SECONDS) * f64::from(FRAME_RATE)).ceil() as usize;
        f.max(MIN_FRAMES)
    }

    /// Samples of the finished clip (the decoded frames trimmed to the requested length).
    pub fn samples(&self) -> usize {
        (self.seconds.clamp(0.0, MAX_SECONDS) * f64::from(SAMPLE_RATE)).round() as usize
    }

    /// The metadata block: `- bpm: … - timesignature: … - keyscale: … - duration: N seconds`.
    pub fn metas(&self) -> String {
        let bpm = self.bpm.map_or_else(|| "N/A".to_string(), |b| b.to_string());
        let ts = self.time_signature.map_or_else(|| "N/A".to_string(), |t| t.to_string());
        let key = self.key.as_deref().and_then(|k| key_scale(k).ok()).unwrap_or_else(|| "N/A".to_string());
        // the reference writes `int(duration)`
        let secs = self.seconds.clamp(0.0, MAX_SECONDS) as u64;
        format!("- bpm: {bpm}\n- timesignature: {ts}\n- keyscale: {key}\n- duration: {secs} seconds\n")
    }

    /// The text-encoder input (`SFT_GEN_PROMPT`).
    pub fn text_prompt(&self) -> String {
        format!("# Instruction\n{INSTRUCTION}\n\n# Caption\n{}\n\n# Metas\n{}<|endoftext|>\n", clean_prompt(&self.prompt), self.metas())
    }

    /// The lyric-encoder input: instrumental.
    pub fn lyrics_prompt(&self) -> &'static str {
        INSTRUMENTAL_LYRICS
    }
}

/// The description as the model reads it: control tokens (`<|…|>`) and control characters
/// removed, whitespace runs collapsed, trimmed, at most [`MAX_PROMPT_CHARS`] characters.
pub fn clean_prompt(s: &str) -> String {
    let mut t = s.to_string();
    // control tokens would steer the tokenizer: drop every `<|…|>`
    while let Some(a) = t.find("<|") {
        match t.get(a + 2..).and_then(|r| r.find("|>")) {
            Some(b) => t.replace_range(a..a + 2 + b + 2, " "),
            None => t.replace_range(a..a + 2, " "),
        }
    }
    let mut out = String::with_capacity(t.len());
    let mut space = false;
    for c in t.chars() {
        if c.is_whitespace() || c.is_control() {
            space = !out.is_empty();
        } else {
            if space {
                out.push(' ');
                space = false;
            }
            out.push(c);
        }
    }
    out.chars().take(MAX_PROMPT_CHARS).collect()
}

/// A key as the model's metadata writes it: "A minor", "F# major", "Eb major".
pub fn key_scale(k: &str) -> Result<String, MusicError> {
    let bad = || MusicError::Invalid(format!("unknown key `{k}` (a note A–G with # or b, then major or minor: \"A minor\", \"F# major\", \"Eb\")"));
    let s = k.trim();
    let mut chars = s.chars();
    let note = chars.next().filter(|c| ('A'..='G').contains(&c.to_ascii_uppercase())).ok_or_else(bad)?.to_ascii_uppercase();
    let rest: String = chars.collect();
    // right after the note, `b` is a flat (no mode word starts with it)
    let (acc, rest) = match rest.chars().next() {
        Some('#' | '♯') => ("#", rest.chars().skip(1).collect::<String>()),
        Some('b' | '♭') => ("b", rest.chars().skip(1).collect::<String>()),
        _ => ("", rest),
    };
    let mode = match rest.trim().to_ascii_lowercase().as_str() {
        "" | "maj" | "major" => "major",
        "m" | "min" | "minor" => "minor",
        _ => return Err(bad()),
    };
    Ok(format!("{note}{acc} {mode}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_follow_the_reference_layout() {
        let mut r = Request::new("Upbeat house, punchy drums", 24.0, 1);
        r.bpm = Some(124);
        r.key = Some("Am".into());
        r.time_signature = Some(4);
        assert_eq!(
            r.text_prompt(),
            "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n# Caption\nUpbeat house, punchy drums\n\n# Metas\n- bpm: 124\n- timesignature: 4\n- keyscale: A minor\n- duration: 24 seconds\n<|endoftext|>\n"
        );
        let d = Request::new("x", 10.6, 1);
        assert_eq!(d.metas(), "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: 10 seconds\n");
        assert_eq!(r.lyrics_prompt(), "# Languages\nunknown\n\n# Lyric\n[Instrumental]<|endoftext|>");
    }

    #[test]
    fn lengths_in_frames_and_samples() {
        assert_eq!(Request::new("x", 10.0, 1).frames(), 250);
        assert_eq!(Request::new("x", 10.0, 1).samples(), 480_000);
        assert_eq!(Request::new("x", 10.01, 1).frames(), 251);
        // short clips are generated at the minimum and trimmed
        assert_eq!(Request::new("x", 5.0, 1).frames(), MIN_FRAMES);
        assert_eq!(Request::new("x", 5.0, 1).samples(), 240_000);
        assert!(Request::new("x", 5.0, 1).samples() <= MIN_FRAMES * HOP);
        assert_eq!(Request::new("x", MAX_SECONDS, 1).frames(), 6000);
    }

    #[test]
    fn keys_normalise() {
        for (k, want) in [
            ("A minor", "A minor"),
            ("am", "A minor"),
            ("Am", "A minor"),
            ("F#", "F# major"),
            ("f# min", "F# minor"),
            ("Eb", "Eb major"),
            ("Bb minor", "Bb minor"),
            ("b minor", "B minor"),
            ("B", "B major"),
            ("C Major", "C major"),
            ("D♭ major", "Db major"),
            ("bb", "Bb major"),
        ] {
            assert_eq!(key_scale(k).unwrap(), want, "{k}");
        }
        for k in ["", "H", "X minor", "C dorian", "C##", "1", "minor"] {
            assert!(key_scale(k).is_err(), "{k}");
        }
    }

    #[test]
    fn hostile_requests_are_refused() {
        assert!(Request::new("calm piano", 30.0, 7).check().is_ok());
        let base = Request::new("calm piano", 30.0, 7);
        let cases = [
            Request { prompt: "".into(), ..base.clone() },
            Request { prompt: "  <|endoftext|>  \n\t".into(), ..base.clone() },
            Request { prompt: "x".repeat(MAX_PROMPT_CHARS + 1), ..base.clone() },
            Request { seconds: f64::NAN, ..base.clone() },
            Request { seconds: f64::INFINITY, ..base.clone() },
            Request { seconds: -3.0, ..base.clone() },
            Request { seconds: 1.0, ..base.clone() },
            Request { seconds: 1e9, ..base.clone() },
            Request { bpm: Some(0), ..base.clone() },
            Request { bpm: Some(10_000), ..base.clone() },
            Request { key: Some("Q".into()), ..base.clone() },
            Request { time_signature: Some(5), ..base.clone() },
            Request { shift: 0, ..base.clone() },
            Request { shift: 9, ..base.clone() },
        ];
        for r in cases {
            assert!(r.check().is_err(), "{r:?}");
            // never panics on hostile numbers
            let _ = (r.frames(), r.samples(), r.metas(), r.text_prompt());
        }
    }

    #[test]
    fn prompts_are_cleaned() {
        assert_eq!(clean_prompt("  lo-fi\n\nhip   hop\t"), "lo-fi hip hop");
        assert_eq!(clean_prompt("a<|endoftext|>b<|im_start|>c"), "a b c");
        assert_eq!(clean_prompt("broken <| tag"), "broken tag");
        assert_eq!(clean_prompt("ok |> fine"), "ok |> fine");
        assert_eq!(clean_prompt("x\u{0}y"), "x y");
        assert_eq!(clean_prompt(&"é".repeat(600)).chars().count(), MAX_PROMPT_CHARS);
        // every prefix of a multibyte string is handled without panicking
        let s = "ü<|ñ|>ç<|";
        for i in 0..=s.len() {
            if let Some(p) = s.get(..i) {
                let _ = clean_prompt(p);
            }
        }
    }
}
