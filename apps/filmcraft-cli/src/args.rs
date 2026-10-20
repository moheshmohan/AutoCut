//! Argument parsing: options may appear anywhere; everything else is positional.

use serde_json::{Map, Value};

/// Options that take a value (`--project p`); every other `--x` is a flag.
const VALUED: &[&str] = &[
    "--project",
    "--save-as",
    "--bridge",
    "--format",
    "--scale",
    "--quality",
    "--seconds",
    "--out",
    "--frames",
    "--preset",
    "--range",
    "--start",
    "--end",
    "--data-dir",
    "--settings",
    "--gpu-rendering",
    "--max-items",
    "--timeout",
    "--session",
    "--auto",
    "--since",
    "--items",
    "--relabel",
    "--compare",
    "--export",
    "--script-out",
    "--script-in",
    "--min-coverage",
    "--bpm",
    "--mood",
    "--voice",
    "--language",
    "--subtitle",
    "--title",
    "--report",
    "--description",
    "--album",
    "--music",
    "--target-minutes",
    "--resolution",
    // showcase look
    "--style",
    "--loudness",
    "--soundfont",
    // showcase branded outro
    "--outro-logo",
    "--outro-channel",
    "--outro-handle",
    "--outro-tagline",
    "--outro-cta",
    "--outro-cta-done",
    "--outro-terminal",
    "--outro-seconds",
    "--outro-background",
    "--screen",
    "--prompt",
    "--subject",
    "--upload-album",
    "--report-out",
    "--queries",
    "--sweep",
    "--month-days",
    "--slot",
    "--variant",
    "--name",
    // shorts
    "--count",
    "--aspect",
    "--min-seconds",
    "--max-seconds",
    "--motion",
    "--topic",
    "--analysis-fps",
    "--sequence",
    // youtube upload, showcase --upload-youtube
    "--channel",
    "--tags",
    "--category",
    "--privacy",
    "--publish-at",
    "--playlist",
    "--thumbnail",
    "--youtube-channel",
    "--youtube-privacy",
    "--youtube-playlist",
    "--youtube-tags",
    "--youtube-description",
    // redact (and showcase --redact)
    "--clips",
    "--kinds",
    "--strength",
    "--padding",
    "--min-face-height",
    "--min-plate-width",
    "--face-score",
    "--plate-score",
    "--exclude",
    "--redact",
    "--target",
    "--target-every",
    "--redact-target",
    // instagram setup / publish
    "--app-id",
    "--redirect-uri",
    "--caption",
    "--cover-at",
    "--cover-url",
    "--video-url",
    // narration numbers, photo locations (photos locations, showcase --places-from-locations)
    "--numbers",
    "--places-from-locations",
    "--album-url",
    "--per-item",
    // showcase-plan, sheet, frames; showcase in two languages
    "--base",
    "--patch",
    "--lines",
    "--lines-out",
    "--n",
    "--side",
    "--columns",
    "--at",
    "--languages",
    "--lines-ml",
    // locks hold
    "--minutes",
    "--note",
    // natural spoken Malayalam: ml-lint, ml-script
    "--max-words",
    "--model",
    "--examples",
    "--remember",
    "--batch",
];

/// Options that take a value in one subcommand only: `showcase --plan` is a flag, while
/// `ml-script --plan plan.json` names a file.
const VALUED_IN: &[(&str, &[&str])] = &[("ml-script", &["--plan"]), ("ml-lint", &["--plan"])];

fn valued_in(cmd: Option<&String>, opt: &str) -> bool {
    cmd.is_some_and(|c| VALUED_IN.iter().any(|(sub, opts)| sub == c && opts.contains(&opt)))
}

#[derive(Debug, Default)]
pub struct Args {
    pub positionals: Vec<String>,
    opts: Vec<(String, Option<String>)>,
}

impl Args {
    pub fn parse(it: impl IntoIterator<Item = String>) -> Self {
        let mut a = Args::default();
        let mut it = it.into_iter();
        while let Some(s) = it.next() {
            if s == "--" {
                a.positionals.extend(it.by_ref());
            } else if let Some((k, v)) = s.split_once('=').filter(|(k, _)| k.starts_with("--")) {
                a.opts.push((k.to_string(), Some(v.to_string())));
            } else if VALUED.contains(&s.as_str()) || valued_in(a.positionals.first(), &s) {
                let v = it.next();
                a.opts.push((s, v));
            } else if s.starts_with("--") && s.len() > 2 {
                a.opts.push((s, None));
            } else {
                a.positionals.push(s);
            }
        }
        a
    }

    pub fn pos(&self, i: usize) -> Option<&str> {
        self.positionals.get(i).map(String::as_str)
    }

    pub fn opt(&self, k: &str) -> Option<&str> {
        self.opts.iter().rev().find(|(n, _)| n == k).and_then(|(_, v)| v.as_deref())
    }

    /// Every value given to a repeatable option, in order (`--auto a --auto b`).
    pub fn all(&self, k: &str) -> Vec<&str> {
        self.opts.iter().filter(|(n, _)| n == k).filter_map(|(_, v)| v.as_deref()).collect()
    }

    pub fn flag(&self, k: &str) -> bool {
        self.opts.iter().any(|(n, _)| n == k)
    }

    /// A version query: `--version` anywhere, or `-V` in the subcommand position, so global
    /// options may come first (`--demo --version`, `--compact -V`).
    pub fn wants_version(&self) -> bool {
        self.flag("--version") || self.pos(0) == Some("-V")
    }

    /// The first option that is not in `allowed`, if any.
    pub fn unknown_opt(&self, allowed: &[&str]) -> Option<&str> {
        self.opts.iter().map(|(n, _)| n.as_str()).find(|n| !allowed.contains(n))
    }

    /// Command params from positionals `from..`: one JSON object, or `key=value` pairs
    /// (dotted keys nest: `color.r=1`).
    pub fn params_from(&self, from: usize) -> Result<Value, String> {
        let rest = self.positionals.get(from..).unwrap_or_default();
        if let [one] = rest
            && one.trim_start().starts_with('{')
        {
            return serde_json::from_str(one).map_err(|e| format!("params JSON: {e}"));
        }
        let mut m = Map::new();
        for kv in rest {
            let (k, v) = kv.split_once('=').ok_or_else(|| format!("expected key=value, got `{kv}`"))?;
            insert_dotted(&mut m, k, parse_value(v));
        }
        Ok(Value::Object(m))
    }
}

fn insert_dotted(m: &mut Map<String, Value>, key: &str, v: Value) {
    match key.split_once('.') {
        None => {
            m.insert(key.to_string(), v);
        }
        Some((head, tail)) => {
            let e = m.entry(head.to_string()).or_insert_with(|| Value::Object(Map::new()));
            if !e.is_object() {
                *e = Value::Object(Map::new());
            }
            if let Value::Object(child) = e {
                insert_dotted(child, tail, v);
            }
        }
    }
}

/// JSON when it parses (numbers, bools, null, arrays, objects, quoted strings), else a string.
pub fn parse_value(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(s: &str) -> Args {
        Args::parse(s.split_whitespace().map(str::to_string))
    }

    #[test]
    fn options_anywhere_and_positionals() {
        let a = args("exec --demo timeline.razor seconds=3.5 --project p.fcproj --save --compact");
        assert_eq!(a.positionals, ["exec", "timeline.razor", "seconds=3.5"]);
        assert_eq!(a.opt("--project"), Some("p.fcproj"));
        assert!(a.flag("--save") && a.flag("--demo") && a.flag("--compact"));
        assert!(!a.flag("--bridge"));
        let b = args("--bridge=127.0.0.1:9876 inspect sequence");
        assert_eq!(b.opt("--bridge"), Some("127.0.0.1:9876"));
        assert_eq!(b.pos(1), Some("sequence"));
        // regression: --language took no value, so `--language ml` was a flag and a positional
        let c = args("showcase --report r.json --language ml --plan");
        assert_eq!((c.opt("--language"), c.positionals.as_slice()), (Some("ml"), &["showcase".to_string()][..]));
    }

    #[test]
    fn key_value_params() {
        let a = args("exec x seconds=3.5 name=Selects on=true ids=[1,2] color.r=1 color.g=0.5 s=\"7\"");
        assert_eq!(a.params_from(2).unwrap(), json!({"seconds": 3.5, "name": "Selects", "on": true, "ids": [1, 2], "color": {"r": 1, "g": 0.5}, "s": "7"}));
        assert!(args("exec x novalue").params_from(2).is_err());
        assert_eq!(args("exec x").params_from(2).unwrap(), json!({}));
    }

    #[test]
    fn json_params() {
        let a = Args::parse(["exec".into(), "x".into(), r#"{"a": {"b": [1]}}"#.into()]);
        assert_eq!(a.params_from(2).unwrap(), json!({"a": {"b": [1]}}));
    }

    #[test]
    fn version_after_global_options() {
        for s in ["--version", "-V", "--demo --version", "--compact -V", "--data-dir d --version", "export --version"] {
            assert!(args(s).wants_version(), "{s}");
        }
        for s in ["version", "--data-dir -V", "exec x -V", "-- --version"] {
            assert!(!args(s).wants_version(), "{s}");
        }
    }

    #[test]
    fn unknown_opt_reports_first_unlisted_option() {
        let allowed = ["--bridge", "--project"];
        assert_eq!(args("mcp --automation-read-root X").unknown_opt(&allowed), Some("--automation-read-root"));
        assert_eq!(args("mcp --bridge=a --typo=1").unknown_opt(&allowed), Some("--typo"));
        assert_eq!(args("mcp --project p --bridge a").unknown_opt(&allowed), None);
    }

    #[test]
    fn repeatable_options_keep_every_value_in_order() {
        let a = Args::parse(["photos", "pick", "--auto", "robots", "--all", "--auto=drones 2022", "--auto", "cats"].map(str::to_string));
        assert_eq!(a.all("--auto"), ["robots", "drones 2022", "cats"]);
        assert_eq!(a.opt("--auto"), Some("cats"), "a single read still takes the last");
        assert!(a.all("--sweep").is_empty());
        // a valued option at the very end has no value: nothing to list
        assert!(args("photos pick --auto").all("--auto").is_empty());
        let b = args("photos pick --sweep 2017-10..2025-12 --month-days 10-21 --queries list.txt");
        assert_eq!((b.opt("--sweep"), b.opt("--month-days"), b.opt("--queries")), (Some("2017-10..2025-12"), Some("10-21"), Some("list.txt")));
        assert_eq!(b.positionals, ["photos", "pick"]);
    }

    /// Regression: `locks hold gpu --minutes 45 --note "…"` (as CLAUDE.md writes it) held for the
    /// default 30 minutes under the default note, its values taken as positionals.
    #[test]
    fn locks_hold_takes_its_minutes_and_note() {
        let a = Args::parse(["locks", "hold", "gpu", "--minutes", "45", "--note", "M3: a measurement"].map(str::to_string));
        assert_eq!(
            (a.opt("--minutes"), a.opt("--note"), a.positionals.as_slice()),
            (Some("45"), Some("M3: a measurement"), &["locks".to_string(), "hold".to_string(), "gpu".to_string()][..])
        );
    }

    #[test]
    fn plan_takes_a_file_in_the_malayalam_commands_only() {
        let a = args("ml-script --plan p.json --out s.json");
        assert_eq!((a.opt("--plan"), a.opt("--out"), a.positionals.as_slice()), (Some("p.json"), Some("s.json"), &["ml-script".to_string()][..]));
        let b = args("--data-dir d ml-lint s.json --plan p.json --strict");
        assert_eq!((b.opt("--plan"), b.pos(1), b.flag("--strict")), (Some("p.json"), Some("s.json"), true));
        // showcase --plan stays a flag
        let c = args("showcase --plan --report r.json");
        assert!(c.flag("--plan") && c.opt("--plan").is_none() && c.opt("--report") == Some("r.json"));
    }

    #[test]
    fn double_dash_ends_options() {
        let a = args("import -- --weird-name.mov a.mov");
        assert_eq!(a.positionals, ["import", "--weird-name.mov", "a.mov"]);
    }

    #[test]
    fn gpu_rendering_takes_a_space_separated_value() {
        let a = args("export --gpu-rendering off out.wav --start 0 --end 0.1");
        assert_eq!(a.positionals, ["export", "out.wav"]);
        assert_eq!(a.opt("--gpu-rendering"), Some("off"));
        let b = args("export out.wav --gpu-rendering auto");
        assert_eq!(b.positionals, ["export", "out.wav"]);
        assert_eq!(b.opt("--gpu-rendering"), Some("auto"));
    }
}
