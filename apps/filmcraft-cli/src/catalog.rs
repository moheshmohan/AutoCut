//! Loopback-only FrameAtlas adapter. Retrieval receipts live with the edit, outside its cache.
use crate::{Backend, args::Args, fail, print};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

const MAX_JSON: u64 = 8 << 20;

struct Client {
    agent: ureq::Agent,
    base: String,
    token: String,
}
impl Client {
    fn new(port: u16) -> Result<Self, String> {
        let agent: ureq::Agent =
            ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(30))).max_redirects(0).http_status_as_error(false).build().into();
        let base = format!("http://127.0.0.1:{port}/api");
        let token = response(agent.get(format!("{base}/session")).call().map_err(|e| e.to_string())?)?["csrf"]
            .as_str()
            .ok_or("FrameAtlas returned no session")?
            .to_string();
        Ok(Self { agent, base, token })
    }
    fn call(&self, action: &str, args: &Value) -> Result<Value, String> {
        response(self.agent.post(format!("{}/{action}", self.base)).header("X-FrameAtlas-Token", &self.token).send_json(args).map_err(|e| e.to_string())?)
    }
}
fn response(mut r: ureq::http::Response<ureq::Body>) -> Result<Value, String> {
    let status = r.status();
    let value: Value = r.body_mut().with_config().limit(MAX_JSON).read_json().map_err(|e| e.to_string())?;
    if let Some(error) = value.get("error").and_then(Value::as_str).filter(|message| !message.trim().is_empty()) {
        return Err(error.to_string());
    }
    if !status.is_success() {
        return Err(format!("FrameAtlas returned {status}"));
    }
    Ok(value)
}
fn read_json(path: &Path) -> Result<Value, String> {
    let f = filmcraft_media::reader::open_regular_file(path).map_err(|e| e.to_string())?;
    if f.metadata().map_err(|e| e.to_string())?.len() > MAX_JSON {
        return Err("JSON exceeds 8 MiB".into());
    }
    let mut bytes = Vec::new();
    f.take(MAX_JSON + 1).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_JSON {
        return Err("JSON exceeds 8 MiB".into());
    }
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}
fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_JSON {
        return Err("handoff exceeds 8 MiB; split the selection into smaller chapters".into());
    }
    let temp = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut f = std::fs::File::create_new(&temp).map_err(|e| e.to_string())?;
    let result = (|| {
        f.write_all(&bytes).map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
        std::fs::rename(&temp, path).map_err(|e| e.to_string())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temp);
    }
    result
}
fn selection_requests(selection: &Value) -> Result<Vec<Value>, String> {
    let entries = selection["entries"].as_array().filter(|v| !v.is_empty() && v.len() <= 2000).ok_or("choose 1..2000 entries")?;
    entries
        .iter()
        .map(|entry| {
            let m = entry.as_object().ok_or("each selection must be an object")?;
            if m.keys().any(|k| !["id", "scene_key", "speech_key", "part_filename", "sourceIn", "duration", "mute", "dashcam"].contains(&k.as_str())) {
                return Err("unknown selection field".into());
            }
            if m.get("id").and_then(Value::as_str).is_none_or(|s| s.is_empty() || s.len() > 256) {
                return Err("each selection needs a catalogue id".into());
            }
            let mut request = serde_json::Map::new();
            for key in ["id", "scene_key", "speech_key", "part_filename"] {
                if let Some(v) = m.get(key) {
                    request.insert(key.into(), v.clone());
                }
            }
            Ok(Value::Object(request))
        })
        .collect()
}

/// Different scene/speech citations of one verified source file reuse its retained bytes.
/// Resolving every citation first preserves current timing and stale-reference checks.
fn reuse_source(state: &Value, request: &Value, resolved: &Value) -> Option<Value> {
    let reference = resolved.get("clip")?.as_object()?;
    let filename = reference.get("source_filename")?.as_str()?;
    let existing = state["results"].as_object()?.values().find(|value| {
        value["asset_id"] == request["id"]
            && value["clip"]["source_filename"] == filename
            && value["editing_sha256"].as_str().is_some()
            && matches!(value["indexed_source_verification"].as_str(), Some("matched" | "no_fingerprint"))
    })?;
    let path = existing["clip"]["source_path"].as_str()?;
    let mut result = existing.clone();
    let mut clip = Value::Object(reference.clone());
    clip["source_path"] = json!(path);
    result["clip"] = clip;
    result["retained_source_reused"] = json!(true);
    Some(result)
}
fn retrieve(client: &Client, selection: &Value, folder: &Path) -> Result<Value, String> {
    let requests = selection_requests(selection)?;
    let folder = folder.canonicalize().map_err(|e| e.to_string())?;
    if !folder.is_dir() {
        return Err("choose an existing editing directory".into());
    }
    let key = format!("{:x}", Sha256::digest(serde_json::to_vec(selection).map_err(|e| e.to_string())?));
    let receipt = folder.join(format!("autocut-frameatlas-{}.json", key.get(..16).ok_or("invalid selection fingerprint")?));
    let lock_path = receipt.with_extension("lock");
    let lock = std::fs::OpenOptions::new().create(true).truncate(false).read(true).write(true).open(lock_path).map_err(|e| e.to_string())?;
    lock.try_lock().map_err(|e| format!("this selection is already being retrieved: {e}"))?;
    let mut state = if receipt.exists() { read_json(&receipt)? } else { json!({"selection":selection,"jobs":{},"results":{}}) };
    if !state["jobs"].is_object() || !state["results"].is_object() {
        return Err("invalid retrieval receipt: jobs/results must be objects".into());
    }
    if state["selection"] != *selection {
        return Err("selection receipt does not match; choose another editing folder".into());
    }
    // Every reference must still resolve before any request is enqueued; expired picker access
    // is returned by the service with its recovery path, never silently substituted.
    let resolved = requests
        .iter()
        .map(|request| {
            let value = client.call("resolve", request)?;
            let clip = value.get("clip").cloned().unwrap_or(Value::Null);
            if serde_json::to_vec(&clip).map_err(|e| e.to_string())?.len() > 16384 {
                return Err("resolved clip exceeds 16 KiB".into());
            }
            Ok(json!({"clip":clip}))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut pending = Vec::new();
    let mut results = BTreeMap::new();
    let mut active_assets = std::collections::BTreeSet::new();
    let mut file_hashes = BTreeMap::<String, String>::new();
    for (request, reference) in requests.iter().zip(&resolved) {
        let ref_key = format!("{:x}", Sha256::digest(request.to_string()));
        if active_assets.contains(&request["id"].to_string()) {
            pending.push(json!({"asset":request["id"],"state":"waitingForOtherReference"}));
            continue;
        }
        let mut result = state["results"][&ref_key].clone();
        if result.is_null()
            && let Some(reused) = reuse_source(&state, request, reference)
        {
            result = reused;
        }
        if !result.is_null() {
            let path = result["clip"]["source_path"].as_str().or_else(|| result["path"].as_str()).ok_or("receipt has no source path")?;
            if !Path::new(path).is_file() {
                return Err(format!("retained editing source is missing: {path}; restore it or explicitly retry in a new folder"));
            }
        } else {
            let job = match state["jobs"].get(&ref_key) {
                Some(id) => id.clone(),
                None => {
                    let mut args = request.clone();
                    args["destination"] = json!(folder);
                    let job = client.call("retrieve", &args)?["id"].clone();
                    state["jobs"][&ref_key] = job.clone();
                    write_json(&receipt, &state)?;
                    job
                }
            };
            let status = client.call("job", &json!({"id":job}))?;
            match status["state"].as_str() {
                Some("done" | "complete" | "completed") => {
                    result = status["result"].clone();
                    state["results"][&ref_key] = result.clone();
                    write_json(&receipt, &state)?;
                }
                Some("failed" | "cancelled" | "error") => {
                    return Err(format!("retrieval {job}: {}. Receipt retained at {}", status["error"], receipt.display()));
                }
                _ => {
                    active_assets.insert(request["id"].to_string());
                    pending.push(json!({"job":job,"state":status["state"],"waitingForSelection":status["waiting_for_selection"]}));
                }
            }
        }
        if !result.is_null() {
            if !result.is_object() {
                return Err("FrameAtlas returned an invalid handoff".into());
            }
            if result["asset_id"] != request["id"] {
                return Err("retrieval returned another asset; receipt cannot be used".into());
            }
            let path = result["clip"]["source_path"].as_str().or_else(|| result["path"].as_str()).ok_or("handoff has no editing path")?.to_string();
            let hash = match file_hashes.get(&path) {
                Some(hash) => hash.clone(),
                None => {
                    let (hash, _) = filmcraft_engine::render_cache::digest_file(Path::new(&path))?;
                    file_hashes.insert(path, hash.clone());
                    hash
                }
            };
            if result.get("editing_sha256").is_some_and(|v| v.as_str() != Some(&hash)) {
                return Err("retained source changed since retrieval; restore the original or make a new selection".into());
            }
            result["editing_sha256"] = json!(hash);
            state["results"][&ref_key] = result.clone();
            write_json(&receipt, &state)?;
            results.insert(ref_key, result);
        }
    }
    if !pending.is_empty() {
        return Ok(
            json!({"pending":pending,"receipt":receipt,"resume":"Run the same catalog retrieve command again; completed downloads are reused. Cancel a job with catalog cancel <job id>."}),
        );
    }
    let mut entries = Vec::new();
    for (request, entry) in requests.iter().zip(selection["entries"].as_array().ok_or("no entries")?) {
        let ref_key = format!("{:x}", Sha256::digest(request.to_string()));
        let mut out = json!({"source":results.get(&ref_key).ok_or("missing completed retrieval")?});
        for key in ["sourceIn", "duration", "mute", "dashcam"] {
            if let Some(v) = entry.get(key) {
                out[key] = v.clone();
            }
        }
        entries.push(out);
    }
    let plan = json!({"format":"autocut.catalogEditPlan","version":1,"name":selection["name"].as_str().unwrap_or("Catalogue edit"),"sequence":selection.get("sequence").cloned().unwrap_or(json!({})),"entries":entries});
    let path = receipt.with_extension("edit-plan.json");
    write_json(&path, &plan)?;
    Ok(json!({"plan":path,"receipt":receipt,"entries":entries.len(),"ready":true}))
}

pub async fn run(a: &Args) {
    if let Some(option) = a.unknown_opt(&["--project", "--save", "--save-as", "--bridge", "--data-dir", "--json", "--compact", "--catalog-port"]) {
        fail(format!("catalog: unknown option {option}"));
    }
    let result: Result<Value, String> = async {
        let action = a.pos(1).ok_or("catalog search|resolve <JSON> | retrieve <selection.json> <editing-folder> | assemble <edit-plan.json> | cancel <job>")?;
        if action == "assemble" {
            let plan = read_json(&PathBuf::from(a.pos(2).ok_or("provide an edit-plan.json")?))?;
            let mut backend = Backend::open(a);
            let out = backend.exec("catalog.importEditPlan", json!({"plan":plan})).await?;
            backend.finish(a).await;
            return Ok(out);
        }
        let port = a.opt("--catalog-port").unwrap_or("7318").parse::<u16>().map_err(|e| e.to_string())?;
        if port == 0 {
            return Err("catalog port must be positive".into());
        }
        let client = Client::new(port)?;
        match action {
            "retrieve" => {
                retrieve(&client, &read_json(Path::new(a.pos(2).ok_or("provide selection.json")?))?, Path::new(a.pos(3).ok_or("provide an editing folder")?))
            }
            "cancel" => client.call("cancel", &json!({"id":a.pos(2).ok_or("provide job id")?})),
            "search" | "asset" | "resolve" | "job" | "stats" => {
                client.call(action, &serde_json::from_str::<Value>(a.pos(2).unwrap_or("{}")).map_err(|e| e.to_string())?)
            }
            _ => Err("unsupported catalogue action".into()),
        }
    }
    .await;
    match result {
        Ok(v) => print(a, &v),
        Err(e) => fail(format!("catalog: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hostile_selection_fails_before_network() {
        for value in [json!({"entries":[]}), json!({"entries":[{"id":"x","evil":true}]}), json!({"entries":[{}]})] {
            assert!(selection_requests(&value).is_err());
        }
    }
}

#[cfg(test)]
mod retrieval_tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    #[test]
    fn expired_selection_preserves_the_service_recovery_message() {
        use std::io::BufRead;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let worker = std::thread::spawn(move || {
            for (index, stream) in listener.incoming().take(2).enumerate() {
                let mut stream = stream.unwrap();
                stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                let mut input = std::io::BufReader::new(stream.try_clone().unwrap());
                let mut length = 0;
                loop {
                    let mut line = String::new();
                    assert!(input.read_line(&mut line).unwrap() > 0);
                    if line == "\r\n" {
                        break;
                    }
                    if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        length = value.trim().parse::<usize>().unwrap();
                    }
                }
                assert!(length < 32768);
                input.read_exact(&mut vec![0; length]).unwrap();
                let (status, value) = if index == 0 {
                    ("200 OK", json!({"csrf":"test"}))
                } else {
                    ("400 Bad Request", json!({"error":"Google Photos selection expired. Reopen the Picker and select this original again."}))
                };
                let body = value.to_string();
                write!(stream, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
                    .unwrap();
            }
        });
        let client = Client::new(port).unwrap();
        let error = client.call("resolve", &json!({"id":"expired"})).unwrap_err();
        assert!(error.contains("Reopen the Picker"), "{error}");
        worker.join().unwrap();
    }

    #[test]
    fn review_catalogue_scenes_reuse_one_verified_retained_original() {
        use std::{io::BufRead, sync::atomic::AtomicBool, time::Instant};
        let dir = std::env::temp_dir().join(format!(
            "autocut-multi-scene-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("original.mov");
        std::fs::write(&source, b"owned synthetic original").unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let count = Arc::new(AtomicUsize::new(0));
        let downloads = count.clone();
        let original = source.clone();
        let worker = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(20);
            let mut current = json!(null);
            let clip = |key: &str| json!({"kind":"scene","scene_key":key,"source_filename":"original.mov","source_start":if key=="first" {0.0}else{1.0},"source_end":if key=="first"{0.5}else{1.5}});
            while !stopping.load(Ordering::Relaxed) && Instant::now() < deadline {
                let (mut stream, _) = match listener.accept() {
                    Ok(s) => s,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    Err(e) => panic!("{e}"),
                };
                stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                let mut input = std::io::BufReader::new(stream.try_clone().unwrap());
                let mut first = String::new();
                input.read_line(&mut first).unwrap();
                let endpoint = first.split_whitespace().nth(1).unwrap();
                let mut length = 0;
                loop {
                    let mut line = String::new();
                    assert!(input.read_line(&mut line).unwrap() > 0);
                    if line == "\r\n" {
                        break;
                    }
                    if let Some(n) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        length = n.trim().parse::<usize>().unwrap();
                    }
                }
                assert!(length < 32768);
                let mut bytes = vec![0; length];
                input.read_exact(&mut bytes).unwrap();
                let args: Value = if bytes.is_empty() { json!({}) } else { serde_json::from_slice(&bytes).unwrap() };
                let value = match endpoint {
                    "/api/session" => json!({"csrf":"synthetic"}),
                    "/api/resolve" => json!({"clip":clip(args["scene_key"].as_str().unwrap())}),
                    "/api/retrieve" => {
                        current = clip(args["scene_key"].as_str().unwrap());
                        json!({"id":downloads.fetch_add(1,Ordering::SeqCst)+1})
                    }
                    "/api/job" => {
                        let mut reference = current.clone();
                        reference["source_path"] = json!(original);
                        json!({"state":"done","error":"","result":{"asset_id":"shared","path":original,"clip":reference,"indexed_source_verification":"matched"}})
                    }
                    _ => panic!("unexpected endpoint {endpoint}"),
                };
                let body = value.to_string();
                write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
                    .unwrap();
            }
        });
        let client = Client::new(port).unwrap();
        let selection = json!({"entries":[{"id":"shared","scene_key":"first"},{"id":"shared","scene_key":"second"}]});
        let result = retrieve(&client, &selection, &dir);
        stop.store(true, Ordering::Relaxed);
        worker.join().unwrap();
        let result = result.unwrap();
        assert_eq!(result["ready"], true);
        assert_eq!(count.load(Ordering::SeqCst), 1, "one original was requested again for its other scene");
        let plan = read_json(Path::new(result["plan"].as_str().unwrap())).unwrap();
        assert_eq!(plan["entries"][0]["source"]["clip"]["scene_key"], "first");
        assert_eq!(plan["entries"][1]["source"]["clip"]["scene_key"], "second");
        assert_eq!(plan["entries"][0]["source"]["editing_sha256"], plan["entries"][1]["source"]["editing_sha256"]);
    }
    #[test]
    fn pending_retrieval_resumes_without_a_second_download_and_can_cancel() {
        let dir = std::env::temp_dir().join(format!("autocut-catalog-retry-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("retained-original.bin");
        std::fs::write(&source, b"selected original").unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let enqueued = Arc::new(AtomicUsize::new(0));
        let count = enqueued.clone();
        let original = source.clone();
        let worker = std::thread::spawn(move || {
            let mut polls = 0;
            for stream in listener.incoming().take(8) {
                let mut stream = stream.unwrap();
                stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                let mut request = Vec::new();
                let mut chunk = [0u8; 1024];
                loop {
                    let n = stream.read(&mut chunk).unwrap();
                    if n == 0 {
                        break;
                    }
                    request.extend_from_slice(&chunk[..n]);
                    assert!(request.len() < 32768);
                    if let Some(split) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&request[..split]);
                        let length = head
                            .lines()
                            .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").and_then(|v| v.trim().parse::<usize>().ok()))
                            .unwrap_or(0);
                        if request.len() >= split + 4 + length {
                            break;
                        }
                    }
                }
                let first = String::from_utf8_lossy(&request);
                let path = first.split_whitespace().nth(1).unwrap();
                let value = match path {
                    "/api/session" => json!({"csrf":"synthetic-test-session"}),
                    "/api/resolve" => json!({"available":true}),
                    "/api/retrieve" => {
                        count.fetch_add(1, Ordering::SeqCst);
                        json!({"id":7})
                    }
                    "/api/job" => {
                        polls += 1;
                        if polls == 1 {
                            json!({"id":7,"state":"queued","result":{}})
                        } else {
                            json!({"id":7,"state":"done","result":{"asset_id":"selected","path":original,"provider":"test"}})
                        }
                    }
                    "/api/cancel" => json!({"id":7,"state":"cancelled"}),
                    _ => panic!("unexpected endpoint {path}"),
                };
                let body = value.to_string();
                write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body)
                    .unwrap();
            }
        });
        let client = Client::new(port).unwrap();
        let selection = json!({"name":"Retry test","entries":[{"id":"selected"}]});
        // Use a different selection key in case a previous failed test left a receipt.
        for file in std::fs::read_dir(&dir).unwrap().flatten() {
            if file.file_name().to_string_lossy().starts_with("autocut-frameatlas-") {
                std::fs::remove_file(file.path()).unwrap();
            }
        }
        assert!(retrieve(&client, &selection, &dir).unwrap()["pending"].is_array());
        let complete = retrieve(&client, &selection, &dir).unwrap();
        assert_eq!(complete["ready"], true);
        assert_eq!(retrieve(&client, &selection, &dir).unwrap()["ready"], true);
        assert_eq!(enqueued.load(Ordering::SeqCst), 1);
        assert_eq!(client.call("cancel", &json!({"id":7})).unwrap()["state"], "cancelled");
        worker.join().unwrap();
        let plan = read_json(Path::new(complete["plan"].as_str().unwrap())).unwrap();
        assert!(plan["entries"][0]["source"]["editing_sha256"].as_str().is_some());
    }
}
