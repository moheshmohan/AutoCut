//! The Elements library from the shell and over MCP: `filmcraft-cli exec elements.list` /
//! `elements.insert`, and the same commands through `command_run` on `filmcraft-cli mcp`.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Output, Stdio};

use serde_json::{Value, json};

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_filmcraft-cli")).args(args).output().expect("spawn filmcraft-cli")
}

fn json_out(o: &Output) -> Value {
    assert!(o.status.success(), "failed: {}", String::from_utf8_lossy(&o.stderr));
    serde_json::from_slice(&o.stdout).expect("JSON on stdout")
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/elements-tests").join(format!("{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn elements_from_the_shell() {
    let dir = tmp("cli");
    let data = dir.join("data");
    let data = data.to_str().unwrap();
    let all = json_out(&cli(&["--data-dir", data, "exec", "elements.list"]));
    assert_eq!(all.as_array().unwrap().len(), 36, "{all}");
    let stickers = json_out(&cli(&["--data-dir", data, "exec", "elements.list", "kind=sticker"]));
    assert_eq!(stickers.as_array().unwrap().len(), 10);
    // a sound and a text preset into the demo project, saved, then read back
    let project = dir.join("show.fcproj");
    let params = json!({"id": "sound.riser-2s", "seconds": 1.0, "dir": dir.join("audio")}).to_string();
    let r = json_out(&cli(&["--demo", "--data-dir", data, "exec", "elements.insert", &params, "--save-as", project.to_str().unwrap()]));
    assert_eq!(r["kind"], "sound", "{r}");
    assert!(std::path::Path::new(r["path"].as_str().unwrap()).exists(), "{r}");
    let r =
        json_out(&cli(&["--project", project.to_str().unwrap(), "--data-dir", data, "exec", "elements.insert", r#"{"id": "text.countdown", "seconds": 2}"#]));
    assert_eq!((r["kind"].as_str(), r["template"].as_str()), (Some("text"), Some("element:text.countdown")), "{r}");
    // a bad id or parameter is an error, not a crash
    for bad in [r#"{"id": "sound.nope"}"#, r#"{"id": "sticker.star", "duration": -1}"#, r#"{"id": 5}"#] {
        let o = cli(&["--demo", "--data-dir", data, "exec", "elements.insert", bad]);
        assert!(!o.status.success(), "{bad}");
        assert!(String::from_utf8_lossy(&o.stderr).contains("elements.insert"), "{bad}: {}", String::from_utf8_lossy(&o.stderr));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A minimal client for `filmcraft-cli mcp` (newline-delimited JSON-RPC over stdio).
struct Mcp {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    next: u64,
}

impl Mcp {
    fn start(data: &str) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_filmcraft-cli"))
            .args(["--demo", "--data-dir", data, "mcp"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn filmcraft-cli mcp");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut m = Mcp { child, stdin, stdout, next: 1 };
        let init =
            m.request("initialize", json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "elements-tests", "version": "1"}}));
        assert_eq!(init["result"]["serverInfo"]["name"], "filmcraft", "{init}");
        m.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
        m
    }

    fn send(&mut self, msg: &Value) {
        writeln!(self.stdin, "{msg}").unwrap();
        self.stdin.flush().unwrap();
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next;
        self.next += 1;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        loop {
            let mut line = String::new();
            assert!(self.stdout.read_line(&mut line).unwrap() > 0, "the MCP server closed stdout while `{method}` was pending");
            let v: Value = serde_json::from_str(&line).unwrap();
            if v["id"] == id {
                return v;
            }
        }
    }

    /// `command_run`: (is a tool error, the result or message).
    fn run(&mut self, id: &str, params: Value) -> (bool, Value) {
        let r = self.request("tools/call", json!({"name": "command_run", "arguments": {"id": id, "params": params}}));
        let text = r["result"]["content"][0]["text"].as_str().unwrap_or_default().to_string();
        (r["result"]["isError"] == true, serde_json::from_str(&text).unwrap_or(Value::String(text)))
    }
}

impl Drop for Mcp {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn elements_over_mcp() {
    let dir = tmp("mcp");
    let mut m = Mcp::start(dir.join("data").to_str().unwrap());
    let (err, sounds) = m.run("elements.list", json!({"kind": "sound"}));
    assert!(!err, "{sounds}");
    assert_eq!(sounds.as_array().unwrap().len(), 14);
    let (err, r) = m.run("elements.insert", json!({"id": "sound.boom", "seconds": 2.0, "dir": dir.join("audio")}));
    assert!(!err, "{r}");
    assert!(r["clip"].is_u64() && r["track"].as_str().is_some_and(|t| t.starts_with('A')), "{r}");
    let (err, r) = m.run("elements.insert", json!({"id": "sticker.bell", "seconds": 2.0, "values": {"badge": "12"}}));
    assert!(!err, "{r}");
    assert!(r["track"].as_str().is_some_and(|t| t.starts_with('V')), "{r}");
    let (err, msg) = m.run("elements.insert", json!({"id": "sticker.bell", "time": -5}));
    assert!(err, "a negative time is a tool error: {msg}");
    drop(m);
    let _ = std::fs::remove_dir_all(&dir);
}
