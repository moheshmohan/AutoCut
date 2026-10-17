//! The macOS Keychain, through Apple's own `security` tool (`/usr/bin/security`).
//!
//! Why the tool and not an in-process binding (`security-framework`): a Keychain item trusts the
//! program that created it, by its code signature. FilmCraft's binaries are rebuilt all the time
//! (and the CLI, the app and test binaries are different programs), and an ad-hoc-signed program
//! that changed asks the user for their login password before it may read the item — or, with
//! interaction turned off, cannot read it at all. Items created by `security` trust `security`,
//! which Apple signs and never changes under us, so every FilmCraft build reads them without a
//! prompt. (GitHub's `gh` keeps its tokens the same way.) It also needs no `unsafe` and no new
//! dependency.
//!
//! The secret never appears in a command line (which other users could list): it is written to
//! `security -i`'s standard input, base64url-encoded so the line needs no quoting, and read back
//! with `find-generic-password -w`. Interactive lines are cut at 4096 bytes by `security`, so a
//! longer line is refused here ([`MAX_LINE`]). Every call is bounded ([`TIMEOUT`]): a locked
//! Keychain whose unlock dialog nobody answers ends in an error, not a hang.

use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::pkce::{base64url, base64url_decode};
use crate::store::{SecretStore, StoreError};

/// Apple's Keychain tool.
pub const SECURITY: &str = "/usr/bin/security";
/// The longest a Keychain call may take (an unlock dialog included).
pub const TIMEOUT: Duration = Duration::from_secs(120);
/// `security -i` reads at most 4096 bytes a line (a longer one becomes two commands).
pub const MAX_LINE: usize = 4000;
/// Marks a secret FilmCraft encoded.
const PREFIX: &str = "b64:";
/// The item's kind in Keychain Access.
const KIND: &str = "FilmCraft sign-in";
/// `security`'s exit status for "the specified item could not be found" (errSecItemNotFound).
const NOT_FOUND: i32 = 44;
/// What we keep of a tool's output.
const MAX_OUTPUT: u64 = 64 * 1024;

/// What a run of the tool printed and how it ended (`None`: killed by a signal).
#[derive(Debug, Clone, Default)]
pub struct ToolOutput {
    pub status: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// Runs `security` (a fake in tests).
pub trait Tool: Send + Sync {
    fn run(&self, args: &[&str], stdin: Option<&[u8]>) -> Result<ToolOutput, String>;
}

/// The real `/usr/bin/security`, each run bounded by [`TIMEOUT`].
#[derive(Debug, Clone)]
pub struct SecurityTool {
    pub path: PathBuf,
    pub timeout: Duration,
}

impl Default for SecurityTool {
    fn default() -> Self {
        Self { path: PathBuf::from(SECURITY), timeout: TIMEOUT }
    }
}

impl Tool for SecurityTool {
    fn run(&self, args: &[&str], stdin: Option<&[u8]>) -> Result<ToolOutput, String> {
        run_bounded(&self.path, args, stdin, self.timeout)
    }
}

/// A Keychain "generic password" item per account, under one service name.
pub struct KeychainStore {
    service: String,
    tool: Box<dyn Tool>,
}

impl KeychainStore {
    /// Items of `service` ("FilmCraft Google Photos", what Keychain Access lists them as).
    pub fn new(service: &str) -> Result<Self, StoreError> {
        Self::with_tool(service, Box::new(SecurityTool::default()))
    }

    pub fn with_tool(service: &str, tool: Box<dyn Tool>) -> Result<Self, StoreError> {
        if !quote_safe(service) {
            return Err(StoreError(format!("`{}` cannot name a Keychain item", crate::truncate(service, 40))));
        }
        Ok(Self { service: service.to_string(), tool })
    }

    pub fn service(&self) -> &str {
        &self.service
    }

    fn check(account: &str) -> Result<(), StoreError> {
        if quote_safe(account) { Ok(()) } else { Err(StoreError("that account name cannot name a Keychain item".into())) }
    }

    fn failure(&self, what: &str, out: &ToolOutput) -> StoreError {
        let said = String::from_utf8_lossy(&out.stderr);
        let said = said.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
        let status = out.status.map_or_else(|| "was stopped".to_string(), |s| format!("ended with status {s}"));
        StoreError(format!(
            "the macOS Keychain could not {what} ({SECURITY} {status}{}{})",
            if said.is_empty() { "" } else { ": " },
            crate::truncate(said, 200)
        ))
    }
}

impl SecretStore for KeychainStore {
    fn name(&self) -> String {
        format!("the macOS Keychain (\"{}\")", self.service)
    }

    fn get(&self, account: &str) -> Result<Option<Vec<u8>>, StoreError> {
        Self::check(account)?;
        let out = self.tool.run(&["find-generic-password", "-s", &self.service, "-a", account, "-w"], None).map_err(StoreError)?;
        match out.status {
            Some(0) => {}
            Some(NOT_FOUND) => return Ok(None),
            _ => return Err(self.failure("read the sign-in", &out)),
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let text = text.strip_suffix('\n').unwrap_or(&text);
        match text.strip_prefix(PREFIX) {
            Some(coded) => base64url_decode(coded).map(Some).ok_or_else(|| StoreError("the Keychain item is damaged (not FilmCraft's encoding)".into())),
            None => Ok(Some(text.as_bytes().to_vec())),
        }
    }

    fn set(&self, account: &str, secret: &[u8]) -> Result<(), StoreError> {
        Self::check(account)?;
        let line = format!("add-generic-password -U -s \"{}\" -a \"{account}\" -D \"{KIND}\" -w \"{PREFIX}{}\"\n", self.service, base64url(secret));
        if line.len() > MAX_LINE {
            return Err(StoreError(format!("the sign-in is too large for the macOS Keychain's command line ({} bytes)", secret.len())));
        }
        let out = self.tool.run(&["-i"], Some(line.as_bytes())).map_err(StoreError)?;
        if out.status == Some(0) { Ok(()) } else { Err(self.failure("save the sign-in", &out)) }
    }

    fn delete(&self, account: &str) -> Result<(), StoreError> {
        Self::check(account)?;
        let out = self.tool.run(&["delete-generic-password", "-s", &self.service, "-a", account], None).map_err(StoreError)?;
        match out.status {
            Some(0) | Some(NOT_FOUND) => Ok(()),
            _ => Err(self.failure("remove the sign-in", &out)),
        }
    }
}

/// Printable ASCII without `"` and `\`, at most 1024 bytes: safe inside the double quotes of a
/// `security -i` line.
fn quote_safe(s: &str) -> bool {
    !s.is_empty() && s.len() <= 1024 && s.bytes().all(|b| (0x20..0x7f).contains(&b) && b != b'"' && b != b'\\')
}

/// The Keychain account for the sign-in kept for directory `dir`: its path when that is plain
/// ASCII, else `sha256:` and a hash of it. Each data directory (`FILMCRAFT_DATA_DIR`) so has its
/// own sign-in, and one never overwrites another's.
pub fn account_for(dir: &Path) -> String {
    let path = dir.to_string_lossy();
    if quote_safe(&path) && !path.contains('\u{FFFD}') {
        return path.into_owned();
    }
    use sha2::Digest as _;
    let digest = sha2::Sha256::digest(dir.as_os_str().as_encoded_bytes());
    let hex: String = digest.iter().take(16).map(|b| format!("{b:02x}")).collect();
    format!("sha256:{hex}")
}

/// Whether this Mac has a Keychain FilmCraft can use: `security` is there and names a default
/// keychain. Asked once per process.
#[cfg(target_os = "macos")]
pub fn available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        Path::new(SECURITY).is_file()
            && run_bounded(Path::new(SECURITY), &["default-keychain"], None, Duration::from_secs(15)).is_ok_and(|o| o.status == Some(0) && !o.stdout.is_empty())
    })
}

/// Run `program` with `args`, `stdin` written to it, for at most `timeout`; the child is killed
/// after that. Output is kept up to 64 KiB a stream.
pub fn run_bounded(program: &Path, args: &[&str], stdin: Option<&[u8]>, timeout: Duration) -> Result<ToolOutput, String> {
    let name = program.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut child = Command::new(program)
        .args(args)
        .stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run {}: {e}", program.display()))?;
    if let (Some(input), Some(mut pipe)) = (stdin, child.stdin.take()) {
        // a few KB at most: the pipe takes it without the child reading first
        let written = pipe.write_all(input);
        drop(pipe);
        if let Err(e) = written {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("cannot talk to {name}: {e}"));
        }
    }
    let collect = |pipe: Option<Box<dyn std::io::Read + Send>>| {
        std::thread::spawn(move || {
            let mut out = Vec::new();
            if let Some(p) = pipe {
                let mut p = p.take(MAX_OUTPUT);
                let _ = p.read_to_end(&mut out);
                // keep draining so the child never blocks on a full pipe
                let _ = std::io::copy(&mut p.into_inner(), &mut std::io::sink());
            }
            out
        })
    };
    let stdout = collect(child.stdout.take().map(|p| Box::new(p) as Box<dyn std::io::Read + Send>));
    let stderr = collect(child.stderr.take().map(|p| Box::new(p) as Box<dyn std::io::Read + Send>));
    let deadline = Instant::now().checked_add(timeout).unwrap_or_else(Instant::now);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                // the readers are left to finish on their own: something the child started may
                // still hold its pipes
                return Err(format!("{name} did not answer within {} s (a locked Keychain waiting for its password?)", timeout.as_secs()));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("lost track of {name}: {e}"));
            }
        }
    };
    let stdout = stdout.join().unwrap_or_default();
    let stderr = stderr.join().unwrap_or_default();
    Ok(ToolOutput { status: status.code(), stdout, stderr })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Records every call; answers like `security` over an in-memory keychain.
    #[derive(Clone, Default)]
    struct FakeSecurity {
        calls: Arc<Mutex<Vec<(Vec<String>, Option<String>)>>>,
        items: Arc<Mutex<std::collections::HashMap<(String, String), String>>>,
        fail_with: Option<i32>,
    }

    impl Tool for FakeSecurity {
        fn run(&self, args: &[&str], stdin: Option<&[u8]>) -> Result<ToolOutput, String> {
            let input = stdin.map(|s| String::from_utf8_lossy(s).into_owned());
            self.calls.lock().unwrap().push((args.iter().map(|s| s.to_string()).collect(), input.clone()));
            if let Some(code) = self.fail_with {
                return Ok(ToolOutput {
                    status: Some(code),
                    stdout: vec![],
                    stderr: b"security: SecKeychainItemCopyContent: User interaction is not allowed.\n".to_vec(),
                });
            }
            let arg = |flag: &str, list: &[String]| list.iter().position(|a| a == flag).and_then(|i| list.get(i + 1)).cloned().unwrap_or_default();
            let mut items = self.items.lock().unwrap();
            match args.first().copied() {
                Some("-i") => {
                    // parse the one line the way `security -i` does: words, double quotes
                    let line = input.unwrap_or_default();
                    assert!(line.ends_with('\n') && line.matches('\n').count() == 1, "one command line");
                    let words: Vec<String> = line
                        .trim_end()
                        .split('"')
                        .enumerate()
                        .flat_map(|(i, part)| if i % 2 == 1 { vec![part.to_string()] } else { part.split_whitespace().map(str::to_string).collect() })
                        .collect();
                    assert_eq!(words.first().map(String::as_str), Some("add-generic-password"));
                    assert!(words.contains(&"-U".to_string()), "updates an existing item");
                    items.insert((arg("-s", &words), arg("-a", &words)), arg("-w", &words));
                    Ok(ToolOutput { status: Some(0), ..Default::default() })
                }
                Some("find-generic-password") => {
                    let list: Vec<String> = args.iter().map(|s| s.to_string()).collect();
                    assert!(list.contains(&"-w".to_string()));
                    match items.get(&(arg("-s", &list), arg("-a", &list))) {
                        Some(pw) => Ok(ToolOutput { status: Some(0), stdout: format!("{pw}\n").into_bytes(), stderr: vec![] }),
                        None => Ok(ToolOutput {
                            status: Some(NOT_FOUND),
                            stdout: vec![],
                            stderr: b"security: SecKeychainSearchCopyNext: The specified item could not be found in the keychain.\n".to_vec(),
                        }),
                    }
                }
                Some("delete-generic-password") => {
                    let list: Vec<String> = args.iter().map(|s| s.to_string()).collect();
                    let gone = items.remove(&(arg("-s", &list), arg("-a", &list))).is_some();
                    Ok(ToolOutput { status: Some(if gone { 0 } else { NOT_FOUND }), ..Default::default() })
                }
                other => panic!("unexpected security call {other:?}"),
            }
        }
    }

    #[test]
    fn round_trips_without_the_secret_on_a_command_line() {
        let fake = FakeSecurity::default();
        let k = KeychainStore::with_tool("FilmCraft Google Photos", Box::new(fake.clone())).unwrap();
        let account = "/Users/someone/Library/Application Support/FilmCraft/google-photos";
        assert_eq!(k.get(account).unwrap(), None);
        let secret = br#"{"access_token":"ya29.SECRET","refresh_token":"1//SECRET","expires_at":1,"scope":"a b"}"#;
        k.set(account, secret).unwrap();
        assert_eq!(k.get(account).unwrap().as_deref(), Some(&secret[..]));
        k.set(account, b"second").unwrap();
        assert_eq!(k.get(account).unwrap().as_deref(), Some(&b"second"[..]));
        // other accounts are other items
        assert_eq!(k.get("/tmp/other").unwrap(), None);
        k.delete(account).unwrap();
        k.delete(account).unwrap();
        assert_eq!(k.get(account).unwrap(), None);
        for (args, stdin) in fake.calls.lock().unwrap().iter() {
            assert!(!args.iter().any(|a| a.contains("SECRET") || a.contains("second") || a.contains(PREFIX)), "{args:?}");
            if let Some(line) = stdin {
                assert!(!line.contains("SECRET"), "the secret is encoded: {line}");
                assert!(line.len() <= MAX_LINE);
            }
        }
        assert!(k.name().contains("macOS Keychain") && k.name().contains("FilmCraft Google Photos"));
    }

    #[test]
    fn failures_are_errors_that_say_so() {
        let fake = FakeSecurity { fail_with: Some(36), ..Default::default() };
        let k = KeychainStore::with_tool("FilmCraft test", Box::new(fake)).unwrap();
        let e = k.get("acct").unwrap_err().to_string();
        assert!(e.contains("macOS Keychain could not read") && e.contains("status 36") && e.contains("User interaction is not allowed"), "{e}");
        assert!(k.set("acct", b"x").is_err());
        assert!(k.delete("acct").is_err());
        struct Broken;
        impl Tool for Broken {
            fn run(&self, _: &[&str], _: Option<&[u8]>) -> Result<ToolOutput, String> {
                Err("cannot run /usr/bin/security: not found".into())
            }
        }
        let k = KeychainStore::with_tool("FilmCraft test", Box::new(Broken)).unwrap();
        assert!(k.get("acct").unwrap_err().to_string().contains("cannot run"));
    }

    #[test]
    fn names_are_checked_and_large_secrets_refused() {
        let fake = FakeSecurity::default();
        for bad in ["", "a\"b", "a\\b", "line\nbreak", "tab\there", "é"] {
            assert!(KeychainStore::with_tool(bad, Box::new(fake.clone())).is_err(), "{bad:?}");
        }
        let k = KeychainStore::with_tool("FilmCraft test", Box::new(fake.clone())).unwrap();
        for bad in ["", "a\"; delete-keychain x", "a\\b", "x\ny"] {
            assert!(k.get(bad).is_err() && k.set(bad, b"x").is_err() && k.delete(bad).is_err(), "{bad:?}");
        }
        assert!(k.set("acct", &vec![b'x'; 2800]).is_ok(), "a large token set fits");
        let e = k.set("acct", &vec![b'x'; 3000]).unwrap_err();
        assert!(e.to_string().contains("too large"), "{e}");
        assert!(fake.calls.lock().unwrap().iter().all(|(_, stdin)| stdin.as_ref().is_none_or(|l| l.len() <= MAX_LINE)));
    }

    #[test]
    fn a_damaged_item_is_an_error() {
        let fake = FakeSecurity::default();
        fake.items.lock().unwrap().insert(("S".into(), "a".into()), "b64:%%%".into());
        fake.items.lock().unwrap().insert(("S".into(), "plain".into()), "{\"x\":1}".into());
        let k = KeychainStore::with_tool("S", Box::new(fake)).unwrap();
        assert!(k.get("a").unwrap_err().to_string().contains("damaged"));
        assert_eq!(k.get("plain").unwrap().as_deref(), Some(&b"{\"x\":1}"[..]), "an item written by hand is read as it is");
    }

    #[test]
    fn accounts_come_from_the_directory() {
        assert_eq!(
            account_for(Path::new("/Users/a/Library/Application Support/FilmCraft/google-photos")),
            "/Users/a/Library/Application Support/FilmCraft/google-photos"
        );
        let odd = account_for(Path::new("/Users/José/data\"x"));
        assert!(odd.starts_with("sha256:") && odd.len() == 7 + 32, "{odd}");
        assert_ne!(odd, account_for(Path::new("/Users/José/data\"y")));
        assert_ne!(account_for(Path::new("/a")), account_for(Path::new("/b")));
    }

    #[cfg(unix)]
    #[test]
    fn runs_are_bounded() {
        let out = run_bounded(Path::new("/bin/sh"), &["-c", "read x; echo \"got $x\"; echo oops >&2; exit 3"], Some(b"hi\n"), Duration::from_secs(20)).unwrap();
        assert_eq!((out.status, String::from_utf8_lossy(&out.stdout).trim(), String::from_utf8_lossy(&out.stderr).trim()), (Some(3), "got hi", "oops"));
        let t = Instant::now();
        let e = run_bounded(Path::new("/bin/sh"), &["-c", "exec sleep 30"], None, Duration::from_millis(200)).unwrap_err();
        assert!(e.contains("did not answer") && t.elapsed() < Duration::from_secs(10), "{e}");
        assert!(run_bounded(Path::new("/nonexistent/security"), &[], None, Duration::from_secs(1)).unwrap_err().contains("cannot run"));
        // a flood of output is cut, not buffered whole, and does not block the child
        let out = run_bounded(Path::new("/bin/sh"), &["-c", "head -c 1000000 /dev/zero"], None, Duration::from_secs(20)).unwrap();
        assert_eq!((out.status, out.stdout.len()), (Some(0), MAX_OUTPUT as usize));
    }
}
