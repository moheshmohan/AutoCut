//! What a generated track is, in the vocabulary editors search music libraries with:
//! **mood → genre → tempo → energy → use case**. Every generated track carries these as
//! metadata, and [`suggest`] ranks the styles for a brief ("calm, 30 s, for a tutorial").

/// The feeling of a track.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Mood {
    /// Minor, big drums and a heroic lead: trailers, showcases.
    Epic,
    /// Minor, sparse and pulsing: suspense.
    Tense,
    /// Major, bright and building: positive reveals.
    Uplifting,
    /// Major, soft pads and little percussion.
    Calm,
    /// Major, bouncy and light.
    Happy,
    /// Minor, low and ominous.
    Dark,
    /// Major sevenths, mellow and floating.
    Dreamy,
    /// Minor, fast and driving.
    Energetic,
    /// Minor, slow and tender.
    Sad,
}

impl Mood {
    pub const ALL: [Mood; 9] = [Mood::Epic, Mood::Tense, Mood::Uplifting, Mood::Calm, Mood::Happy, Mood::Dark, Mood::Dreamy, Mood::Energetic, Mood::Sad];

    pub fn id(self) -> &'static str {
        match self {
            Mood::Epic => "epic",
            Mood::Tense => "tense",
            Mood::Uplifting => "uplifting",
            Mood::Calm => "calm",
            Mood::Happy => "happy",
            Mood::Dark => "dark",
            Mood::Dreamy => "dreamy",
            Mood::Energetic => "energetic",
            Mood::Sad => "sad",
        }
    }

    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            Mood::Epic => "Epic",
            Mood::Tense => "Tense",
            Mood::Uplifting => "Uplifting",
            Mood::Calm => "Calm",
            Mood::Happy => "Happy",
            Mood::Dark => "Dark",
            Mood::Dreamy => "Dreamy",
            Mood::Energetic => "Energetic",
            Mood::Sad => "Sad",
        }
    }

    /// Other words for it ("suspense" is tense, "chill" is dreamy).
    pub fn synonyms(self) -> &'static [&'static str] {
        match self {
            Mood::Epic => &["heroic", "grand", "triumphant", "powerful", "majestic"],
            Mood::Tense => &["suspense", "suspenseful", "thriller", "tension", "anxious", "nervous"],
            Mood::Uplifting => &["inspiring", "inspirational", "hopeful", "positive", "motivational", "optimistic"],
            Mood::Calm => &["peaceful", "relaxed", "relaxing", "gentle", "serene", "soft", "quiet"],
            Mood::Happy => &["cheerful", "joyful", "fun", "playful", "upbeat", "bright", "sunny", "feel-good"],
            Mood::Dark => &["ominous", "sinister", "scary", "creepy", "eerie", "mysterious", "menacing"],
            Mood::Dreamy => &["chill", "chilled", "nostalgic", "mellow", "floaty", "hazy"],
            Mood::Energetic => &["driving", "intense", "hype", "aggressive", "fast", "pumping", "action-packed"],
            Mood::Sad => &["melancholic", "melancholy", "sombre", "somber", "emotional", "sentimental", "reflective", "tender"],
        }
    }

    /// The mood with this id (any case, surrounding space ignored).
    pub fn from_id(s: &str) -> Option<Mood> {
        Mood::ALL.into_iter().find(|m| m.id().eq_ignore_ascii_case(s.trim()))
    }

    /// The mood with this id or one of its [`synonyms`](Mood::synonyms).
    pub fn parse(s: &str) -> Option<Mood> {
        let t = s.trim();
        Mood::from_id(t).or_else(|| Mood::ALL.into_iter().find(|m| m.synonyms().iter().any(|w| w.eq_ignore_ascii_case(t))))
    }

    /// In a minor key.
    pub fn minor(self) -> bool {
        matches!(self, Mood::Epic | Mood::Tense | Mood::Dark | Mood::Energetic | Mood::Sad)
    }

    /// Close in feeling (a suggestion for one may offer the other).
    pub fn related(self, other: Mood) -> bool {
        use Mood::*;
        let pair = |a: Mood, b: Mood| (self == a && other == b) || (self == b && other == a);
        self == other
            || pair(Epic, Energetic)
            || pair(Epic, Tense)
            || pair(Tense, Dark)
            || pair(Uplifting, Happy)
            || pair(Uplifting, Epic)
            || pair(Calm, Dreamy)
            || pair(Calm, Sad)
            || pair(Dreamy, Sad)
            || pair(Happy, Energetic)
    }
}

/// The instrumentation and groove of a track.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Genre {
    /// Taiko and toms, strings, a brass-like lead, ostinati.
    Cinematic,
    /// Four-on-the-floor kick, supersaw chords, sub bass, side-chain pumping.
    Electronic,
    /// Backbeat drums, piano chords, plucked lead.
    Pop,
    /// Soft pads, bells, little or no percussion.
    Ambient,
    /// Swung boom-bap drums, electric piano, round bass.
    Lofi,
    /// Piano with soft strings, no drums until the energy rises.
    Piano,
}

impl Genre {
    pub const ALL: [Genre; 6] = [Genre::Cinematic, Genre::Electronic, Genre::Pop, Genre::Ambient, Genre::Lofi, Genre::Piano];

    pub fn id(self) -> &'static str {
        match self {
            Genre::Cinematic => "cinematic",
            Genre::Electronic => "electronic",
            Genre::Pop => "pop",
            Genre::Ambient => "ambient",
            Genre::Lofi => "lofi",
            Genre::Piano => "piano",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Genre::Cinematic => "Cinematic",
            Genre::Electronic => "Electronic",
            Genre::Pop => "Pop",
            Genre::Ambient => "Ambient",
            Genre::Lofi => "Lo-fi",
            Genre::Piano => "Piano",
        }
    }

    pub fn synonyms(self) -> &'static [&'static str] {
        match self {
            Genre::Cinematic => &["orchestral", "film", "score", "soundtrack", "epic orchestral"],
            Genre::Electronic => &["edm", "synth", "synthwave", "house", "techno", "dance", "electro"],
            Genre::Pop => &["corporate", "indie", "pop rock", "upbeat pop"],
            Genre::Ambient => &["atmospheric", "soundscape", "drone", "new age"],
            Genre::Lofi => &["lo-fi", "lo fi", "chillhop", "hip hop", "hip-hop", "beats"],
            Genre::Piano => &["solo piano", "keys", "classical", "neoclassical"],
        }
    }

    /// The genre with this id or one of its synonyms (any case).
    pub fn parse(s: &str) -> Option<Genre> {
        let t = s.trim();
        Genre::ALL.into_iter().find(|g| g.id().eq_ignore_ascii_case(t) || g.synonyms().iter().any(|w| w.eq_ignore_ascii_case(t)))
    }
}

/// What music is used for (the search vocabulary of a music library).
pub const USE_CASES: [&str; 30] = [
    "action",
    "corporate",
    "crime",
    "documentary",
    "drama",
    "family",
    "fashion",
    "food",
    "gaming",
    "horror",
    "kids",
    "lifestyle",
    "meditation",
    "memories",
    "nature",
    "podcast",
    "product",
    "promo",
    "showcase",
    "slideshow",
    "sports",
    "study",
    "suspense",
    "technology",
    "trailer",
    "travel",
    "tutorial",
    "vlog",
    "wedding",
    "workout",
];

/// The use case with this name (any case), as listed in [`USE_CASES`].
pub fn use_case(s: &str) -> Option<&'static str> {
    let t = s.trim();
    USE_CASES.into_iter().find(|u| u.eq_ignore_ascii_case(t))
}

/// A mood played in a genre: its tempo range, energy and what it suits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Style {
    pub mood: Mood,
    pub genre: Genre,
    /// Slowest, usual and fastest tempo, beats per minute.
    pub bpm: (f64, f64, f64),
    /// 0..1: how busy and loud it usually is.
    pub energy: f64,
    pub use_cases: &'static [&'static str],
    /// One line on how it sounds.
    pub feel: &'static str,
}

/// Every style, the usual genre of each mood first.
pub const STYLES: [Style; 18] = [
    Style {
        mood: Mood::Epic,
        genre: Genre::Cinematic,
        bpm: (80.0, 96.0, 112.0),
        energy: 0.8,
        use_cases: &["trailer", "showcase", "sports", "gaming", "action", "promo"],
        feel: "taiko drums, low strings and a brass-like lead; big and heroic",
    },
    Style {
        mood: Mood::Epic,
        genre: Genre::Electronic,
        bpm: (120.0, 128.0, 140.0),
        energy: 0.85,
        use_cases: &["gaming", "sports", "promo", "technology", "action"],
        feel: "four-on-the-floor drums, supersaw chords and a driving bass",
    },
    Style {
        mood: Mood::Tense,
        genre: Genre::Cinematic,
        bpm: (70.0, 92.0, 110.0),
        energy: 0.55,
        use_cases: &["suspense", "documentary", "crime", "trailer"],
        feel: "a pulsing low ostinato, ticking percussion and dark strings",
    },
    Style {
        mood: Mood::Tense,
        genre: Genre::Electronic,
        bpm: (85.0, 100.0, 120.0),
        energy: 0.6,
        use_cases: &["technology", "gaming", "crime", "suspense"],
        feel: "a sequenced bass pulse, tight hats and cold pads",
    },
    Style {
        mood: Mood::Uplifting,
        genre: Genre::Pop,
        bpm: (100.0, 112.0, 128.0),
        energy: 0.7,
        use_cases: &["corporate", "product", "travel", "vlog", "promo", "showcase"],
        feel: "piano chords, claps and a bright plucked melody",
    },
    Style {
        mood: Mood::Uplifting,
        genre: Genre::Cinematic,
        bpm: (84.0, 100.0, 116.0),
        energy: 0.7,
        use_cases: &["documentary", "wedding", "showcase", "nature", "corporate"],
        feel: "rising strings, warm horns and gentle percussion",
    },
    Style {
        mood: Mood::Calm,
        genre: Genre::Ambient,
        bpm: (60.0, 72.0, 90.0),
        energy: 0.3,
        use_cases: &["meditation", "nature", "tutorial", "podcast", "study"],
        feel: "slow pads and soft bells, barely any drums",
    },
    Style {
        mood: Mood::Calm,
        genre: Genre::Piano,
        bpm: (60.0, 76.0, 92.0),
        energy: 0.25,
        use_cases: &["wedding", "family", "memories", "tutorial"],
        feel: "broken piano chords over soft strings",
    },
    Style {
        mood: Mood::Happy,
        genre: Genre::Pop,
        bpm: (108.0, 120.0, 132.0),
        energy: 0.7,
        use_cases: &["vlog", "kids", "family", "food", "lifestyle", "travel"],
        feel: "bouncy bass, claps and a whistling lead",
    },
    Style {
        mood: Mood::Happy,
        genre: Genre::Electronic,
        bpm: (115.0, 124.0, 130.0),
        energy: 0.75,
        use_cases: &["lifestyle", "fashion", "promo", "workout"],
        feel: "feel-good house: offbeat bass, claps and stabs",
    },
    Style {
        mood: Mood::Dark,
        genre: Genre::Electronic,
        bpm: (80.0, 90.0, 110.0),
        energy: 0.55,
        use_cases: &["horror", "gaming", "technology", "crime"],
        feel: "a heavy sub, sparse beats and detuned drones",
    },
    Style {
        mood: Mood::Dark,
        genre: Genre::Cinematic,
        bpm: (60.0, 80.0, 100.0),
        energy: 0.5,
        use_cases: &["horror", "suspense", "documentary", "trailer"],
        feel: "low strings, distant drums and a cold lead",
    },
    Style {
        mood: Mood::Dreamy,
        genre: Genre::Lofi,
        bpm: (70.0, 82.0, 92.0),
        energy: 0.35,
        use_cases: &["study", "travel", "memories", "slideshow", "vlog", "lifestyle"],
        feel: "swung beats, electric piano sevenths and a round bass",
    },
    Style {
        mood: Mood::Dreamy,
        genre: Genre::Ambient,
        bpm: (60.0, 70.0, 85.0),
        energy: 0.25,
        use_cases: &["nature", "meditation", "memories", "slideshow"],
        feel: "floating pads and bell-like arpeggios",
    },
    Style {
        mood: Mood::Energetic,
        genre: Genre::Electronic,
        bpm: (120.0, 128.0, 150.0),
        energy: 0.95,
        use_cases: &["sports", "workout", "action", "promo", "gaming"],
        feel: "pumping kick, rolling bass and supersaw hooks",
    },
    Style {
        mood: Mood::Energetic,
        genre: Genre::Pop,
        bpm: (115.0, 126.0, 140.0),
        energy: 0.85,
        use_cases: &["sports", "vlog", "promo", "travel"],
        feel: "driving drums, eighth-note bass and a punchy lead",
    },
    Style {
        mood: Mood::Sad,
        genre: Genre::Piano,
        bpm: (56.0, 70.0, 84.0),
        energy: 0.25,
        use_cases: &["memories", "documentary", "drama", "family"],
        feel: "a slow piano melody over soft strings",
    },
    Style {
        mood: Mood::Sad,
        genre: Genre::Cinematic,
        bpm: (56.0, 68.0, 84.0),
        energy: 0.35,
        use_cases: &["documentary", "drama", "trailer", "memories"],
        feel: "long string chords and a lonely lead",
    },
];

impl Style {
    /// The style of `mood` in `genre`; without a genre (or for a pairing not in [`STYLES`]) the
    /// mood's usual one, in the asked genre.
    pub fn of(mood: Mood, genre: Option<Genre>) -> Style {
        let usual = STYLES.iter().find(|s| s.mood == mood).copied().unwrap_or(STYLES[0]);
        match genre {
            None => usual,
            Some(g) => STYLES.iter().find(|s| s.mood == mood && s.genre == g).copied().unwrap_or(Style { genre: g, ..usual }),
        }
    }
}

/// What a suggestion is asked for; every field optional.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Query {
    pub mood: Option<Mood>,
    pub genre: Option<Genre>,
    /// One of [`USE_CASES`].
    pub use_case: Option<&'static str>,
    /// 0..1.
    pub energy: Option<f64>,
    pub bpm: Option<f64>,
}

/// A style ranked for a [`Query`]: lower `cost` is a better match.
#[derive(Clone, Debug, PartialEq)]
pub struct Suggestion {
    pub style: Style,
    pub cost: f64,
    /// Why it was picked, in words.
    pub why: String,
}

/// The styles ranked for `q`, best first (ties keep [`STYLES`] order). Never empty.
pub fn suggest(q: &Query) -> Vec<Suggestion> {
    let mut out: Vec<Suggestion> = STYLES
        .iter()
        .map(|s| {
            let mut cost = 0.0;
            let mut why: Vec<String> = Vec::new();
            if let Some(m) = q.mood {
                if s.mood == m {
                    why.push(format!("{} mood", m.id()));
                } else if s.mood.related(m) {
                    cost += 1.5;
                    why.push(format!("{} is close to {}", s.mood.id(), m.id()));
                } else {
                    cost += 4.0;
                }
            }
            if let Some(g) = q.genre {
                if s.genre == g {
                    why.push(format!("{} genre", g.id()));
                } else {
                    cost += 2.0;
                }
            }
            if let Some(u) = q.use_case {
                if s.use_cases.contains(&u) {
                    why.push(format!("suits {u}"));
                } else {
                    cost += 1.5;
                }
            }
            if let Some(e) = q.energy.filter(|e| e.is_finite()) {
                let d = (s.energy - e.clamp(0.0, 1.0)).abs();
                cost += 3.0 * d;
                if d <= 0.15 {
                    why.push(format!("energy {:.2} close to {:.2}", s.energy, e));
                }
            }
            if let Some(b) = q.bpm.filter(|b| b.is_finite()) {
                let (lo, _, hi) = s.bpm;
                let off = if b < lo {
                    lo - b
                } else if b > hi {
                    b - hi
                } else {
                    0.0
                };
                cost += off / 10.0;
                if off == 0.0 {
                    why.push(format!("{b:.0} BPM is in its range"));
                }
            }
            if why.is_empty() {
                why.push(s.feel.to_string());
            }
            Suggestion { style: *s, cost, why: why.join("; ") }
        })
        .collect();
    out.sort_by(|a, b| a.cost.total_cmp(&b.cost));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_parses_ids_and_synonyms() {
        assert_eq!(Mood::parse(" EPIC "), Some(Mood::Epic));
        assert_eq!(Mood::parse("suspense"), Some(Mood::Tense));
        assert_eq!(Mood::parse("Chill"), Some(Mood::Dreamy));
        assert_eq!(Mood::parse("jazzy"), None);
        assert_eq!(Mood::from_id("suspense"), None, "ids only");
        assert_eq!(Genre::parse("EDM"), Some(Genre::Electronic));
        assert_eq!(Genre::parse("lo-fi"), Some(Genre::Lofi));
        assert_eq!(Genre::parse("polka"), None);
        assert_eq!(use_case("Vlog"), Some("vlog"));
        assert_eq!(use_case("party"), None);
        // no word means two moods or two genres
        for m in Mood::ALL {
            for w in m.synonyms() {
                assert_eq!(Mood::parse(w), Some(m), "{w}");
            }
        }
    }

    #[test]
    fn every_style_is_well_formed_and_every_mood_has_one() {
        for s in STYLES {
            let (lo, usual, hi) = s.bpm;
            assert!(crate::music::tempo::MIN_BPM <= lo && lo <= usual && usual <= hi && hi <= crate::music::tempo::MAX_BPM, "{s:?}");
            assert!((0.0..=1.0).contains(&s.energy));
            assert!(!s.use_cases.is_empty() && s.use_cases.iter().all(|u| USE_CASES.contains(u)), "{s:?}");
        }
        for m in Mood::ALL {
            assert_eq!(Style::of(m, None).mood, m);
        }
        // a pairing not in the table keeps the mood's tempo and energy, in the asked genre
        let s = Style::of(Mood::Calm, Some(Genre::Electronic));
        assert_eq!((s.mood, s.genre, s.bpm), (Mood::Calm, Genre::Electronic, Style::of(Mood::Calm, None).bpm));
        assert!(Mood::Epic.minor() && !Mood::Happy.minor());
    }

    #[test]
    fn suggestions_rank_the_brief_first() {
        let s = suggest(&Query { mood: Some(Mood::Calm), use_case: Some("tutorial"), ..Query::default() });
        assert_eq!(s.len(), STYLES.len());
        assert_eq!((s[0].style.mood, s[0].style.genre), (Mood::Calm, Genre::Ambient), "{:?}", s[0]);
        assert!(s[0].why.contains("calm mood") && s[0].why.contains("tutorial"));
        let s = suggest(&Query { energy: Some(0.95), bpm: Some(128.0), ..Query::default() });
        assert_eq!(s[0].style.mood, Mood::Energetic);
        let s = suggest(&Query { mood: Some(Mood::Sad), genre: Some(Genre::Cinematic), ..Query::default() });
        assert_eq!((s[0].style.mood, s[0].style.genre), (Mood::Sad, Genre::Cinematic));
        // nonsense numbers are ignored, never a panic
        let s = suggest(&Query { energy: Some(f64::NAN), bpm: Some(f64::INFINITY), ..Query::default() });
        assert_eq!(s.len(), STYLES.len());
    }
}
