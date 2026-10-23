//! Client for the desktop app's JSON-lines control protocol. A connection per request in flight,
//! reused once its reply is in; reconnect on failure.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::Mutex;

use crate::{AutomationError, control_token};

/// How often a blocking export's app job is polled.
const JOB_POLL: Duration = Duration::from_millis(250);
/// Idle connections kept for reuse; the app serves at most 16 at once.
const MAX_IDLE: usize = 4;
/// How the app's reply to a request without its token begins (`apps/filmcraft/src/control_server.rs`).
const AUTH_FAILED: &str = "control authentication failed";

type Conn = (BufReader<tokio::net::tcp::OwnedReadHalf>, tokio::net::tcp::OwnedWriteHalf);

pub struct BridgeClient {
    addr: String,
    /// Inherited from the same owner-controlled environment as the desktop app.
    token: Option<String>,
    /// Without [`Self::token`], the owner-only file the app wrote its token to
    /// ([`crate::control_token`]); read for each request, so a restarted app's new token is used.
    token_file: Option<std::path::PathBuf>,
    /// Connections with no request in flight. A request takes one (or opens one) for its whole
    /// round trip and puts it back only once the reply is read, so a slow or abandoned request
    /// never holds up the others or leaves its late reply for the next one to read (#1538).
    idle: Mutex<Vec<Conn>>,
    next_id: AtomicU64,
}

impl BridgeClient {
    /// `addr` such as `127.0.0.1:9876` (loopback only). The desktop app requires its token on
    /// every request: `FILMCRAFT_CONTROL_TOKEN` from the environment, else the token the app
    /// wrote for its port ([`crate::control_token::path`]).
    pub fn new(addr: impl Into<String>) -> Result<Self, AutomationError> {
        let addr = addr.into();
        let host = addr.rsplit_once(':').map(|(h, _)| h).unwrap_or(&addr);
        if !matches!(host, "127.0.0.1" | "localhost" | "[::1]" | "::1") {
            return Err(AutomationError::BadRequest(format!("bridge address must be loopback, got `{addr}`")));
        }
        let token_file = control_token::port_of(&addr).and_then(control_token::path);
        Ok(Self { addr, token: control_token::from_env(), token_file, idle: Mutex::new(Vec::new()), next_id: AtomicU64::new(1) })
    }

    /// The token sent with each request: the environment's, else the one in the app's file.
    async fn credential(&self) -> Option<String> {
        if let Some(t) = &self.token {
            return Some(t.clone());
        }
        let file = tokio::fs::File::open(self.token_file.as_ref()?).await.ok()?;
        let mut text = String::new();
        file.take(256).read_to_string(&mut text).await.ok()?;
        control_token::parse(&text)
    }

    /// The app's refusal of a request's token, saying where the token came from and what to do.
    fn refused(&self, sent: bool, error: &str) -> AutomationError {
        let file = self.token_file.as_ref().map_or_else(|| "the app's token file".to_string(), |p| p.display().to_string());
        let source = match (&self.token, sent) {
            (Some(_), _) => format!("{} from the environment", control_token::ENV),
            (None, true) => file,
            (None, false) => {
                return AutomationError::Bridge(format!(
                    "{error}: no control token (no {} in the environment, none in {file}); start the app with `--control <port>` or FILMCRAFT_CONTROL_PORT, which writes it there",
                    control_token::ENV
                ));
            }
        };
        AutomationError::Bridge(format!(
            "{error}: the app at {} refused the token ({source}); restart the app, or unset a stale {}",
            self.addr,
            control_token::ENV
        ))
    }

    /// Run engine command `id` in the app. A blocking export (`file.exportMedia` with
    /// `wait: true`, see [`crate::long_job::LONG_COMMANDS`]) is started as an app job and polled
    /// here until it finishes, so the app keeps repainting (its status bar shows the progress) and
    /// answering other requests, and no request waits on the control server's 60 s reply limit
    /// (#91, #92). The reply is the same as a blocking export's: `{job, path, result}` (plus
    /// [`crate::long_job::JOB_EXTRAS`]: `offlineMedia`, an upload's `upload`), or the job's error.
    pub async fn execute(&self, id: &str, mut params: Value) -> Result<Value, AutomationError> {
        if !crate::long_job::is_long(id, &params) {
            return self.call("engine.execute", json!({"command": id, "params": params})).await;
        }
        if let Some(p) = params.as_object_mut() {
            p.insert("wait".into(), json!(false));
        }
        let mut start = self.call("engine.execute", json!({"command": id, "params": params})).await?;
        let Some(job) = start.get("job").and_then(Value::as_u64) else { return Ok(start) };
        let done = loop {
            tokio::time::sleep(JOB_POLL).await;
            let jobs = self.call("engine.execute", json!({"command": "jobs.list", "params": {}})).await?;
            let Some(j) = jobs.as_array().and_then(|a| a.iter().find(|j| j["id"].as_u64() == Some(job))).cloned() else {
                break Value::Null;
            };
            if j["finished"].as_bool() == Some(true) {
                break j;
            }
        };
        let result = done.get("result").cloned().unwrap_or(Value::Null);
        if let Some(e) = result.get("error").and_then(Value::as_str) {
            let what = if id == "file.exportMedia" { "export".to_string() } else { format!("`{id}`") };
            return Err(AutomationError::App(format!("{what} failed: {e}")));
        }
        if let Some(o) = start.as_object_mut() {
            o.insert("result".into(), result);
            for k in crate::long_job::JOB_EXTRAS {
                if let Some(v) = done.get(*k) {
                    o.insert((*k).into(), v.clone());
                }
            }
        }
        Ok(start)
    }

    /// Call a control method; returns `result` or the app's error.
    pub async fn call(&self, method: &str, params: Value) -> Result<Value, AutomationError> {
        for attempt in 0..2 {
            // A retry gets a fresh connection: the failed one may have been closed by an app restart.
            let idle = if attempt == 0 { self.idle.lock().await.pop() } else { None };
            let mut conn = match idle {
                Some(c) => c,
                None => self.connect().await?,
            };
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let token = self.credential().await;
            let line = format!("{}\n", json!({"id": id, "token": token, "method": method, "params": params}));
            let res: Result<Value, AutomationError> = async {
                conn.1.write_all(line.as_bytes()).await.map_err(|e| AutomationError::Bridge(e.to_string()))?;
                let mut buf = String::new();
                tokio::time::timeout(Duration::from_secs(90), conn.0.read_line(&mut buf))
                    .await
                    .map_err(|_| AutomationError::Bridge("timeout".into()))?
                    .map_err(|e| AutomationError::Bridge(e.to_string()))?;
                serde_json::from_str(&buf).map_err(|e| AutomationError::Bridge(format!("bad reply: {e}")))
            }
            .await;
            match res {
                Ok(v) => {
                    let error = v.get("error").and_then(Value::as_str).unwrap_or("error");
                    // the app closes a connection whose token it refused: it is not kept
                    if v.get("ok").and_then(Value::as_bool) != Some(true) && error.starts_with(AUTH_FAILED) {
                        return Err(self.refused(token.is_some(), error));
                    }
                    let mut idle = self.idle.lock().await;
                    if idle.len() < MAX_IDLE {
                        idle.push(conn);
                    }
                    drop(idle);
                    return if v.get("ok").and_then(Value::as_bool) == Some(true) {
                        Ok(v.get("result").cloned().unwrap_or(Value::Null))
                    } else {
                        Err(AutomationError::App(error.to_string()))
                    };
                }
                Err(_) if attempt == 0 => {}
                Err(e) => return Err(e),
            }
        }
        Err(AutomationError::Bridge("unreachable".into()))
    }

    async fn connect(&self) -> Result<Conn, AutomationError> {
        let s = tokio::time::timeout(Duration::from_secs(5), TcpStream::connect(&self.addr))
            .await
            .map_err(|_| AutomationError::Bridge(format!("timed out connecting to {}", self.addr)))?
            .map_err(|e| AutomationError::Bridge(format!("cannot connect to {} ({e}); start the app with `filmcraft --control <port>`", self.addr)))?;
        let (r, w) = s.into_split();
        Ok((BufReader::new(r), w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::net::TcpListener;

    /// A stand-in for the app's control server: the export starts job 7, which `jobs.list` reports
    /// running twice and then finished with `result`. Every request is recorded.
    async fn fake_app(result: Value) -> (String, Arc<Mutex<Vec<Value>>>) {
        fake_app_with(result, None).await
    }

    /// [`fake_app`] whose finished job also lists `extra` (`offlineMedia`, `upload`, …).
    async fn fake_app_with(result: Value, extra: Option<(&'static str, Value)>) -> (String, Arc<Mutex<Vec<Value>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        tokio::spawn(async move {
            let (s, _) = listener.accept().await.unwrap();
            let (r, mut w) = s.into_split();
            let mut lines = BufReader::new(r).lines();
            let mut polls = 0;
            while let Ok(Some(line)) = lines.next_line().await {
                let req: Value = serde_json::from_str(&line).unwrap();
                log.lock().await.push(req.clone());
                let reply = match req["params"]["command"].as_str() {
                    Some("file.exportMedia" | "photos.upload") => json!({"job": 7, "path": "/tmp/out.mov"}),
                    Some("jobs.list") => {
                        polls += 1;
                        let finished = polls >= 3;
                        let mut j = json!({"id": 7, "finished": finished, "result": if finished { result.clone() } else { Value::Null }});
                        if let Some((k, v)) = extra.clone().filter(|_| finished) {
                            j[k] = v;
                        }
                        json!([j])
                    }
                    _ => json!(null),
                };
                let line = format!("{}\n", json!({"id": req["id"], "ok": true, "result": reply}));
                w.write_all(line.as_bytes()).await.unwrap();
            }
        });
        (addr, seen)
    }

    /// A blocking bridge export runs as an app job (#91, #92): the app never sees `wait: true` (it
    /// would encode on its UI thread and the reply would time out after 60 s), and the caller still
    /// gets the finished result.
    #[tokio::test]
    async fn a_blocking_export_is_polled_as_a_job() {
        let (addr, seen) = fake_app(json!({"frames": 642})).await;
        let b = BridgeClient::new(addr).unwrap();
        let r = b.execute("file.exportMedia", json!({"path": "/tmp/out.mov", "wait": true})).await.unwrap();
        assert_eq!(r, json!({"job": 7, "path": "/tmp/out.mov", "result": {"frames": 642}}));
        let seen = seen.lock().await;
        assert_eq!(seen[0]["params"]["params"]["wait"], json!(false));
        assert!(seen.iter().all(|q| q["params"]["params"]["wait"] != json!(true)));
        assert_eq!(seen.iter().filter(|q| q["params"]["command"] == "file.exportMedia").count(), 1, "started once");
        assert_eq!(seen.iter().filter(|q| q["params"]["command"] == "jobs.list").count(), 3);
    }

    /// The offline media a polled export drew comes back as in a blocking export's reply (#1054).
    #[tokio::test]
    async fn a_polled_export_reports_its_offline_media() {
        let off = json!([{"item": 8, "status": "missing"}]);
        let (addr, _) = fake_app_with(json!({"frames": 3}), Some(("offlineMedia", off.clone()))).await;
        let b = BridgeClient::new(addr).unwrap();
        let r = b.execute("file.exportMedia", json!({"path": "/tmp/out.mov", "wait": true})).await.unwrap();
        assert_eq!(r["offlineMedia"], off, "{r}");
    }

    /// A Google Photos upload in the app is polled the same way; its media item comes back.
    #[tokio::test]
    async fn a_polled_upload_reports_its_media_item() {
        let item = json!({"mediaItemId": "M1", "productUrl": "https://photos.google.com/lr/photo/M1"});
        let (addr, seen) = fake_app_with(json!({"bytes": 10}), Some(("upload", item.clone()))).await;
        let b = BridgeClient::new(addr).unwrap();
        let r = b.execute("photos.upload", json!({"path": "/tmp/out.mov", "wait": true})).await.unwrap();
        assert_eq!(r, json!({"job": 7, "path": "/tmp/out.mov", "result": {"bytes": 10}, "upload": item}));
        assert_eq!(seen.lock().await[0]["params"]["params"]["wait"], json!(false), "the app runs it as a job");
    }

    #[tokio::test]
    async fn a_failed_blocking_export_is_an_error() {
        let (addr, _) = fake_app(json!({"error": "encode: disk full"})).await;
        let b = BridgeClient::new(addr).unwrap();
        let e = b.execute("file.exportMedia", json!({"path": "/tmp/out.mov", "wait": true})).await.unwrap_err();
        assert!(e.to_string().contains("export failed: encode: disk full"), "{e}");
    }

    /// A stand-in for the app that serves any number of connections and answers each request on
    /// its own, as the app does: `slow` after `slow_reply` (never when `None`), anything else at
    /// once with its command id. The ids of the requests it receives are sent to the returned channel.
    async fn slow_app(slow_reply: Option<Duration>) -> (String, tokio::sync::mpsc::UnboundedReceiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let (seen, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Ok((s, _)) = listener.accept().await {
                let seen = seen.clone();
                tokio::spawn(async move {
                    let (r, w) = s.into_split();
                    let w = Arc::new(Mutex::new(w));
                    let mut lines = BufReader::new(r).lines();
                    while let Ok(Some(line)) = lines.next_line().await {
                        let req: Value = serde_json::from_str(&line).unwrap();
                        let cmd = req["params"]["command"].as_str().unwrap_or_default().to_string();
                        let _ = seen.send(cmd.clone());
                        let delay = if cmd == "slow" { slow_reply } else { Some(Duration::ZERO) };
                        let w = w.clone();
                        tokio::spawn(async move {
                            let Some(d) = delay else { return };
                            tokio::time::sleep(d).await;
                            let line = format!("{}\n", json!({"id": req["id"], "ok": true, "result": cmd}));
                            let _ = w.lock().await.write_all(line.as_bytes()).await;
                        });
                    }
                });
            }
        });
        (addr, rx)
    }

    /// A command the app is slow to answer holds up only itself: a later call is answered at once
    /// instead of queueing behind it, and behind every call queued after it (#1538).
    #[tokio::test]
    async fn a_slow_command_does_not_block_later_calls() {
        let (addr, mut seen) = slow_app(None).await;
        let b = Arc::new(BridgeClient::new(addr).unwrap());
        let slow = tokio::spawn({
            let b = b.clone();
            async move { b.execute("slow", json!({})).await }
        });
        assert_eq!(seen.recv().await.as_deref(), Some("slow"));
        let fast = tokio::time::timeout(Duration::from_secs(5), b.execute("fast", json!({}))).await;
        assert_eq!(fast.expect("the later call waited on the slow one").unwrap(), json!("fast"));
        slow.abort();
    }

    /// A call given up while it waits leaves no late reply on a connection for the next call to read
    /// as its own (#1538).
    #[tokio::test]
    async fn an_abandoned_call_leaves_no_late_reply_behind() {
        let (addr, mut seen) = slow_app(Some(Duration::from_millis(100))).await;
        let b = Arc::new(BridgeClient::new(addr).unwrap());
        let slow = tokio::spawn({
            let b = b.clone();
            async move { b.execute("slow", json!({})).await }
        });
        assert_eq!(seen.recv().await.as_deref(), Some("slow"));
        slow.abort();
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(b.execute("fast", json!({})).await.unwrap(), json!("fast"));
    }

    /// Everything else is forwarded unchanged.
    #[tokio::test]
    async fn other_commands_are_forwarded() {
        let (addr, seen) = fake_app(Value::Null).await;
        let b = BridgeClient::new(addr).unwrap();
        b.execute("file.exportMedia", json!({"path": "/tmp/out.mov"})).await.unwrap();
        b.execute("sequence.inspect", json!({})).await.unwrap();
        let seen = seen.lock().await;
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[1]["params"]["command"], "sequence.inspect");
    }

    #[tokio::test]
    async fn credential_is_sent_on_every_request() {
        let (addr, seen) = fake_app(Value::Null).await;
        let mut b = BridgeClient::new(addr).unwrap();
        // Inject directly to avoid mutating the process environment in parallel tests.
        let token = "0123456789abcdef".repeat(4);
        b.token = Some(token.clone());
        b.call("ui.inspect", json!({})).await.unwrap();
        b.execute("file.saveAs", json!({"path": "/owner/project.fcproj"})).await.unwrap();
        let seen = seen.lock().await;
        assert_eq!(seen.len(), 2);
        assert!(seen.iter().all(|request| request["token"] == token));
        assert_eq!(seen[1]["params"]["params"]["path"], "/owner/project.fcproj");
    }

    /// A stand-in for the app's control server with its token check: a request with `token`
    /// is answered, any other gets the app's refusal and its connection is closed.
    async fn guarded_app(token: Arc<std::sync::Mutex<String>>) -> (String, Arc<Mutex<Vec<Value>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        tokio::spawn(async move {
            while let Ok((s, _)) = listener.accept().await {
                let (log, token) = (log.clone(), token.clone());
                tokio::spawn(async move {
                    let (r, mut w) = s.into_split();
                    let mut lines = BufReader::new(r).lines();
                    while let Ok(Some(line)) = lines.next_line().await {
                        let req: Value = serde_json::from_str(&line).unwrap();
                        log.lock().await.push(req.clone());
                        let expected = token.lock().unwrap().clone();
                        if req["token"].as_str() != Some(expected.as_str()) {
                            let _ = w
                                .write_all(format!("{}\n", json!({"ok": false, "error": "control authentication failed; closing the connection"})).as_bytes())
                                .await;
                            return;
                        }
                        let _ = w.write_all(format!("{}\n", json!({"id": req["id"], "ok": true, "result": req["method"]})).as_bytes()).await;
                    }
                });
            }
        });
        (addr, seen)
    }

    /// Without a token in the environment the client sends the one the app wrote for its port, and
    /// after the app restarts with a new token (and a new file) the new one.
    #[tokio::test]
    async fn the_token_comes_from_the_apps_file_and_follows_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("9877.token");
        let first = control_token::generate().unwrap();
        control_token::publish(&file, &first).unwrap();
        let token = Arc::new(std::sync::Mutex::new(first.clone()));
        let (addr, seen) = guarded_app(token.clone()).await;
        let mut b = BridgeClient::new(addr).unwrap();
        b.token = None;
        b.token_file = Some(file.clone());
        assert_eq!(b.call("ui.inspect", json!({})).await.unwrap(), json!("ui.inspect"));
        let second = control_token::generate().unwrap();
        *token.lock().unwrap() = second.clone();
        control_token::publish(&file, &second).unwrap();
        assert_eq!(b.call("ui.elements", json!({})).await.unwrap(), json!("ui.elements"));
        let seen = seen.lock().await;
        assert_eq!(seen[0]["token"], first);
        assert_eq!(seen.last().unwrap()["token"], second);
    }

    /// No token anywhere: the app refuses the request, and the error says where the token is
    /// looked for and how the app puts it there; a wrong one says it was refused.
    #[tokio::test]
    async fn a_missing_or_refused_token_says_what_to_do() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("9877.token");
        let token = Arc::new(std::sync::Mutex::new(control_token::generate().unwrap()));
        let (addr, seen) = guarded_app(token).await;
        let mut b = BridgeClient::new(addr).unwrap();
        b.token = None;
        b.token_file = Some(file.clone());
        let e = b.call("ui.inspect", json!({})).await.unwrap_err().to_string();
        assert!(e.contains("control authentication failed") && e.contains("no control token") && e.contains(&file.display().to_string()), "{e}");
        assert_eq!(seen.lock().await[0]["token"], Value::Null, "nothing to send");
        control_token::publish(&file, &control_token::generate().unwrap()).unwrap();
        let e = b.call("ui.inspect", json!({})).await.unwrap_err().to_string();
        assert!(e.contains("refused the token"), "{e}");
    }
}
