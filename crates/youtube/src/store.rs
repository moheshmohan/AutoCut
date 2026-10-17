//! The connected channels: `<data dir>/youtube/channels.json` (no secret: ids, titles, handles,
//! which one is the default) and, per channel, its own sign-in in a [`TokenVault`] over
//! `<data dir>/youtube/channels/<channel id>/` — so the Keychain item "FilmCraft YouTube" has one
//! account per channel (the account is that directory) and channels never share or overwrite a
//! sign-in.

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use filmcraft_oauth::{SecretStore, TokenVault};
use serde::{Deserialize, Serialize};

use crate::YtError;

/// The channel list.
pub const FILE: &str = "channels.json";
/// The Keychain item's name (Keychain Access lists it so); one account per channel.
pub const KEYCHAIN_SERVICE: &str = "FilmCraft YouTube";
/// More channels than anyone connects.
pub const MAX_CHANNELS: usize = 64;
/// More than the file ever needs.
const MAX_FILE: u64 = 256 * 1024;

/// A connected channel (what `channels.list?mine=true` said when it was connected).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Channel {
    /// `UC…`
    pub id: String,
    pub title: String,
    /// `@name` (YouTube's `customUrl`), or empty.
    pub handle: String,
    /// `allowed`, `eligible`, `disallowed` (videos over 15 minutes), or empty when unknown.
    pub long_uploads: String,
    /// The scopes its sign-in granted, space-separated.
    pub scope: String,
    /// Unix seconds.
    pub connected_at: u64,
}

impl Channel {
    pub fn grants(&self, scope: &str) -> bool {
        self.scope.split_whitespace().any(|s| s == scope)
    }

    /// "H4hacks (@h4hacks)".
    pub fn label(&self) -> String {
        if self.handle.is_empty() { self.title.clone() } else { format!("{} ({})", self.title, self.handle) }
    }

    pub fn to_json(&self, default: bool) -> serde_json::Value {
        serde_json::json!({
            "id": self.id, "title": self.title, "handle": self.handle, "default": default,
            "url": format!("https://www.youtube.com/channel/{}", self.id),
            "longUploads": self.long_uploads, "playlists": self.grants(crate::auth::MANAGE_SCOPE), "connectedAt": self.connected_at,
        })
    }
}

/// The list as stored.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Book {
    /// The default channel's id.
    pub default: Option<String>,
    pub channels: Vec<Channel>,
}

impl Book {
    pub fn default_channel(&self) -> Option<&Channel> {
        let id = self.default.as_deref()?;
        self.channels.iter().find(|c| c.id == id)
    }

    /// The channel `reference` names: its id, its handle (with or without `@`, any case) or its
    /// title (any case); `None` is the default. An unknown or ambiguous name is an error that
    /// lists the connected channels.
    pub fn find(&self, reference: Option<&str>) -> Result<&Channel, YtError> {
        let list = || {
            if self.channels.is_empty() {
                "none is connected: run `filmcraft-cli youtube auth`".to_string()
            } else {
                format!("connected: {}", self.channels.iter().map(|c| format!("{} [{}]", c.label(), c.id)).collect::<Vec<_>>().join(", "))
            }
        };
        let Some(r) = reference.map(str::trim).filter(|r| !r.is_empty()) else {
            return match self.default_channel().or_else(|| if self.channels.len() == 1 { self.channels.first() } else { None }) {
                Some(c) => Ok(c),
                None if self.channels.is_empty() => {
                    Err(YtError::Channel("YouTube is not connected: run `filmcraft-cli youtube auth` and pick the channel in Google's account chooser".into()))
                }
                None => Err(YtError::Channel(format!("no default YouTube channel: name one (`filmcraft-cli youtube default <channel>`); {}", list()))),
            };
        };
        if let Some(c) = self.channels.iter().find(|c| c.id == r) {
            return Ok(c);
        }
        let bare = r.trim_start_matches('@');
        let by_handle: Vec<&Channel> =
            self.channels.iter().filter(|c| !c.handle.is_empty() && c.handle.trim_start_matches('@').eq_ignore_ascii_case(bare)).collect();
        if let [c] = by_handle.as_slice() {
            return Ok(c);
        }
        let by_title: Vec<&Channel> = self.channels.iter().filter(|c| c.title.trim().eq_ignore_ascii_case(r)).collect();
        match by_title.as_slice() {
            [c] => Ok(c),
            [] => Err(YtError::Channel(format!("no connected YouTube channel `{}`; {}", crate::clean(r, 60), list()))),
            _ => Err(YtError::Channel(format!("{} connected channels are titled `{}`: name one by its id; {}", by_title.len(), crate::clean(r, 60), list()))),
        }
    }
}

/// Where the channels' sign-ins are kept.
#[derive(Clone)]
pub enum Secrets {
    /// The macOS Keychain (unless `FILMCRAFT_SECRET_STORE=file`), else owner-only files.
    System,
    /// Owner-only `tokens.json` files only.
    Files,
    /// This store, with owner-only files as the fallback (tests).
    Store(Arc<dyn SecretStore>),
}

/// The channel list and the channels' sign-ins of one data directory (`<data dir>/youtube`).
#[derive(Clone)]
pub struct Channels {
    dir: PathBuf,
    secrets: Secrets,
}

impl Channels {
    pub fn new(dir: &Path, secrets: Secrets) -> Self {
        Self { dir: dir.to_path_buf(), secrets }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The list (empty when there is none). Entries with an unusable id are dropped; a damaged
    /// file is an error that says how to start again.
    pub fn load(&self) -> Result<Book, YtError> {
        let path = self.dir.join(FILE);
        let f = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Book::default()),
            Err(e) => return Err(YtError::Setup(format!("{}: {e}", path.display()))),
        };
        let mut b = Vec::new();
        f.take(MAX_FILE + 1).read_to_end(&mut b).map_err(|e| YtError::Setup(format!("{}: {e}", path.display())))?;
        if b.len() as u64 > MAX_FILE {
            return Err(YtError::Setup(format!("{} is larger than a channel list ever is; remove it and connect the channels again", path.display())));
        }
        let mut book: Book = serde_json::from_slice(&b).map_err(|e| {
            YtError::Setup(format!("{} is damaged ({e}); remove it and connect the channels again (`filmcraft-cli youtube auth`)", path.display()))
        })?;
        book.channels.retain(|c| valid_channel_id(&c.id));
        book.channels.truncate(MAX_CHANNELS);
        if book.default.as_deref().is_some_and(|d| !book.channels.iter().any(|c| c.id == d)) {
            book.default = None;
        }
        Ok(book)
    }

    /// Keep the list (atomically: a new file, then a rename).
    pub fn save(&self, book: &Book) -> Result<(), YtError> {
        std::fs::create_dir_all(&self.dir).map_err(|e| YtError::Setup(format!("{}: {e}", self.dir.display())))?;
        let path = self.dir.join(FILE);
        let tmp = self.dir.join(format!("{FILE}.tmp"));
        let bytes = serde_json::to_vec_pretty(book).map_err(|e| YtError::Setup(format!("cannot write the channel list: {e}")))?;
        std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, &path)).map_err(|e| YtError::Setup(format!("{}: {e}", path.display())))
    }

    /// The directory of a channel's sign-in.
    pub fn channel_dir(&self, id: &str) -> Result<PathBuf, YtError> {
        if !valid_channel_id(id) {
            return Err(YtError::Channel(format!("`{}` is not a YouTube channel id", crate::clean(id, 40))));
        }
        Ok(self.dir.join("channels").join(id))
    }

    /// A channel's sign-in.
    pub fn vault(&self, id: &str) -> Result<TokenVault, YtError> {
        let dir = self.channel_dir(id)?;
        Ok(match &self.secrets {
            Secrets::System => TokenVault::system(&dir, KEYCHAIN_SERVICE),
            Secrets::Files => TokenVault::file_only(&dir),
            Secrets::Store(s) => TokenVault::with_store(&dir, s.clone()),
        })
    }
}

/// A YouTube channel id as YouTube makes them (`UC` and 22 more), loosely: 1–64 letters, digits,
/// `-` and `_` (so it is also a safe directory name).
pub fn valid_channel_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ch(id: &str, title: &str, handle: &str) -> Channel {
        Channel { id: id.into(), title: title.into(), handle: handle.into(), ..Default::default() }
    }

    #[test]
    fn channels_are_found_by_id_handle_or_title() {
        let book = Book {
            default: Some("UCh4".into()),
            channels: vec![
                ch("UCh4", "H4hacks", "@H4hacks"),
                ch("UCmm", "Mohesh Mohan", "@MoheshMohan"),
                ch("UCs4", "S4Saranya", ""),
                ch("UCs5", "s4saranya", ""),
            ],
        };
        assert_eq!(book.find(None).unwrap().id, "UCh4");
        assert_eq!(book.find(Some("  ")).unwrap().id, "UCh4");
        assert_eq!(book.find(Some("UCmm")).unwrap().id, "UCmm");
        assert_eq!(book.find(Some("@moheshmohan")).unwrap().id, "UCmm");
        assert_eq!(book.find(Some("h4hacks")).unwrap().id, "UCh4");
        assert_eq!(book.find(Some("mohesh mohan")).unwrap().id, "UCmm");
        let e = book.find(Some("S4Saranya")).unwrap_err().to_string();
        assert!(e.contains("2 connected channels") && e.contains("UCs4"), "{e}");
        let e = book.find(Some("Life Of 'S'")).unwrap_err().to_string();
        assert!(e.contains("no connected YouTube channel") && e.contains("H4hacks (@H4hacks) [UCh4]"), "{e}");
        // no default: the only channel, else an error naming the command
        let one = Book { default: None, channels: vec![ch("UCh4", "H4hacks", "")] };
        assert_eq!(one.find(None).unwrap().id, "UCh4");
        let two = Book { default: None, channels: vec![ch("UCa", "A", ""), ch("UCb", "B", "")] };
        assert!(two.find(None).unwrap_err().to_string().contains("youtube default"));
        assert!(Book::default().find(None).unwrap_err().to_string().contains("youtube auth"));
    }

    #[test]
    fn the_list_survives_odd_files_and_ids_stay_names() {
        let dir = std::env::temp_dir().join(format!("filmcraft-yt-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = Channels::new(&dir, Secrets::Files);
        assert_eq!(store.load().unwrap(), Book::default());
        let book = Book { default: Some("UCh4".into()), channels: vec![ch("UCh4", "H4hacks", "@H4hacks")] };
        store.save(&book).unwrap();
        assert_eq!(store.load().unwrap(), book);
        std::fs::write(dir.join(FILE), r#"{"default":"../x","channels":[{"id":"../../etc"},{"id":"UCok","title":"Ok"},{"id":""},{"id":5}]}"#).unwrap();
        assert!(store.load().unwrap_err().to_string().contains("damaged"), "a wrong type is damage");
        std::fs::write(dir.join(FILE), r#"{"default":"../x","channels":[{"id":"../../etc"},{"id":"UCok","title":"Ok"},{"id":""}]}"#).unwrap();
        let b = store.load().unwrap();
        assert_eq!((b.channels.len(), b.default.clone()), (1, None), "{b:?}");
        std::fs::write(dir.join(FILE), "{").unwrap();
        assert!(store.load().unwrap_err().to_string().contains("connect the channels again"));
        std::fs::write(dir.join(FILE), vec![b' '; 300_000]).unwrap();
        assert!(store.load().is_err());
        for bad in ["", "..", "a/b", "UC x", &"a".repeat(65), "UC\u{e9}"] {
            assert!(store.channel_dir(bad).is_err() && store.vault(bad).is_err(), "{bad}");
        }
        assert_eq!(store.channel_dir("UC_a-1").unwrap(), dir.join("channels").join("UC_a-1"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
