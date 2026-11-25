//! ComfyUI entries against a stand-in server written here (std `TcpListener` on 127.0.0.1:0)
//! that answers ComfyUI's endpoints as the test says. No ComfyUI runs in the tests.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use serde_json::{Value, json};

use super::*;
use crate::generate_run::{self as run, Provider, Vars};

#[derive(Clone, Debug)]
struct Req {
    method: String,
    path: String,
    body: Vec<u8>,
}

impl Req {
    fn route(&self) -> &str {
        self.path.split('?').next().unwrap_or("")
    }
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }
}

/// What the stand-in does with a request.
enum Act {
    Json(u16, Value),
    Bytes(Vec<u8>),
    /// These bytes as they are (a broken answer).
    Raw(Vec<u8>),
    /// Close without answering.
    Close,
    /// Stop serving: every later connection is refused.
    Shutdown,
}

struct Stand {
    url: String,
    log: Arc<Mutex<Vec<Req>>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Stand {
    fn log(&self) -> Vec<Req> {
        self.log.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }
    fn routes(&self) -> Vec<String> {
        self.log().iter().map(|r| format!("{} {}", r.method, r.route())).collect()
    }
}

impl Drop for Stand {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn read_req(c: &mut TcpStream) -> Option<Req> {
    let mut got = Vec::new();
    let mut buf = [0u8; 8192];
    let head_end = loop {
        if let Some(i) = got.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        let n = c.read(&mut buf).ok()?;
        if n == 0 {
            return None;
        }
        got.extend_from_slice(&buf[..n]);
    };
    let head = String::from_utf8_lossy(&got[..head_end]).into_owned();
    let mut lines = head.lines();
    let mut first = lines.next()?.split(' ');
    let (method, path) = (first.next()?.to_string(), first.next()?.to_string());
    let length: usize =
        lines.filter_map(|l| l.split_once(':')).find(|(k, _)| k.eq_ignore_ascii_case("content-length")).and_then(|(_, v)| v.trim().parse().ok()).unwrap_or(0);
    let mut body = got[head_end..].to_vec();
    while body.len() < length {
        let n = c.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&buf[..n]);
    }
    Some(Req { method, path, body })
}

fn respond(c: &mut TcpStream, status: u16, kind: &str, body: &[u8]) {
    let head = format!("HTTP/1.1 {status} X\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
    let _ = c.write_all(head.as_bytes());
    let _ = c.write_all(body);
}

/// A stand-in ComfyUI: `handler` gets each request and how many times its method and route
/// were asked before.
fn stand(mut handler: impl FnMut(&Req, usize) -> Act + Send + 'static) -> Stand {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    l.set_nonblocking(true).unwrap();
    let url = format!("http://127.0.0.1:{}", l.local_addr().unwrap().port());
    let log = Arc::new(Mutex::new(Vec::new()));
    let stop = Arc::new(AtomicBool::new(false));
    let (log2, stop2) = (log.clone(), stop.clone());
    let thread = std::thread::spawn(move || {
        let mut seen: HashMap<String, usize> = HashMap::new();
        while !stop2.load(Ordering::Relaxed) {
            let Ok((mut c, _)) = l.accept() else {
                std::thread::sleep(Duration::from_millis(2));
                continue;
            };
            // (an accepted socket can inherit the listener's non-blocking mode)
            c.set_nonblocking(false).unwrap();
            c.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
            let Some(req) = read_req(&mut c) else { continue };
            let n = seen.entry(format!("{} {}", req.method, req.route())).or_insert(0);
            let count = *n;
            *n += 1;
            log2.lock().unwrap_or_else(PoisonError::into_inner).push(req.clone());
            match handler(&req, count) {
                Act::Json(status, v) => respond(&mut c, status, "application/json", v.to_string().as_bytes()),
                Act::Bytes(b) => respond(&mut c, 200, "application/octet-stream", &b),
                Act::Raw(b) => {
                    let _ = c.write_all(&b);
                }
                Act::Close => {}
                Act::Shutdown => break,
            }
        }
    });
    Stand { url, log, stop, thread: Some(thread) }
}

fn png() -> Vec<u8> {
    filmcraft_export::encode_png(vec![120; 64 * 36 * 4], 64, 36).unwrap()
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fc-comfy-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// An API-format workflow: a checkpoint, a prompt, a size, a sampler, a picture to start from
/// and a save node; `extra` adds nodes.
fn workflow(at: &Path, extra: Value) -> String {
    let mut w = json!({
        "3": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 20, "cfg": 7.0, "model": ["4", 0], "positive": ["6", 0], "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "open_model.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
        "10": {"class_type": "LoadImage", "inputs": {"image": "example.png"}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}},
    });
    if let (Some(w), Some(more)) = (w.as_object_mut(), extra.as_object()) {
        for (k, v) in more {
            w.insert(k.clone(), v.clone());
        }
    }
    let path = at.join("workflow_api.json");
    std::fs::write(&path, w.to_string()).unwrap();
    path.to_string_lossy().into_owned()
}

fn entry(server: &str, workflow: &str, extra: &str) -> String {
    format!(
        r#"{{"id":"cu","kind":"image","name":"Stand-in ComfyUI","comfyui":{{"server":"{server}","workflow":"{workflow}","inputs":{{"6.text":"{{prompt}}, film still","3.seed":"{{seed}}","5.width":"{{width}}","5.height":"{{height}}","10.image":"{{image}}","3.steps":8}},"output":"9","models":[{{"file":"open_model.safetensors","license":"Apache-2.0"}}]{extra}}},"estimateSeconds":5,"timeoutSeconds":30}}"#
    )
}

fn provider(json: &str) -> Provider {
    run::parse(&format!(r#"{{"providers":[{json}]}}"#)).unwrap().remove(0)
}

fn timing() -> Timing {
    Timing { poll: Duration::from_millis(15), give_up: Duration::from_millis(400), connect: Duration::from_secs(1), io: Duration::from_secs(3) }
}

fn done(id: &str, outputs: Value) -> Value {
    json!({id: {"prompt": [], "outputs": outputs, "status": {"status_str": "success", "completed": true, "messages": []}}})
}

/// Runs a prepared job against a stand-in; the notes it made.
fn go(p: &Provider, job: &Job, out: &Path, cancel: &AtomicBool) -> (Result<(), String>, Vec<String>, usize) {
    let (mut notes, mut ticks) = (Vec::new(), 0usize);
    let r = run_with(p, job, out, cancel, &mut |_, _| ticks += 1, &mut |n| notes.push(n), timing());
    (r, notes, ticks)
}

#[test]
fn a_workflow_is_filled_queued_followed_fetched_and_the_gpu_freed() {
    let d = dir("happy");
    let picture = d.join("start.png");
    std::fs::write(&picture, png()).unwrap();
    let result = png();
    let served = result.clone();
    let s = stand(move |r, n| match (r.method.as_str(), r.route()) {
        ("GET", "/system_stats") => Act::Json(200, json!({"system": {"comfyui_version": "0.0-test"}})),
        ("POST", "/upload/image") => Act::Json(200, json!({"name": "uploaded.png", "subfolder": "", "type": "input"})),
        ("POST", "/prompt") => Act::Json(200, json!({"prompt_id": "abc-123", "number": 1, "node_errors": {}})),
        ("GET", "/queue") if n == 0 => Act::Json(200, json!({"queue_running": [[0, "other", {}, {}, []]], "queue_pending": [[1, "abc-123", {}, {}, []]]})),
        ("GET", "/queue") => Act::Json(200, json!({"queue_running": [[1, "abc-123", {}, {}, []]], "queue_pending": []})),
        ("GET", "/history/abc-123") if n < 6 => Act::Json(200, json!({})),
        ("GET", "/history/abc-123") => {
            Act::Json(200, done("abc-123", json!({"9": {"images": [{"filename": "ComfyUI_00001_.png", "subfolder": "", "type": "output"}]}})))
        }
        ("GET", "/view") => Act::Bytes(served.clone()),
        ("POST", "/free") => Act::Json(200, json!({})),
        _ => Act::Json(404, json!({"error": "not here"})),
    });
    let p = provider(&entry(&s.url, &workflow(&d, json!({})), ""));
    assert!(run::installed(&p) && !run::takes_references(&p));
    let c = p.comfyui.clone().unwrap();
    let v = Vars {
        prompt: "a red kite; rm -rf / \"quoted\"".into(),
        seed: 42,
        width: Some(1024),
        height: Some(576),
        image: Some(picture.to_string_lossy().into_owned()),
        ..Default::default()
    };
    let job = prepare(&c, &v, "png").unwrap();
    // the request's values in a copy, each typed as the input wants it; the picture waits for upload
    let inputs = |n: &str| job.workflow[n]["inputs"].clone();
    assert_eq!(inputs("6")["text"], "a red kite; rm -rf / \"quoted\", film still");
    assert_eq!((inputs("3")["seed"].clone(), inputs("3")["steps"].clone()), (json!(42), json!(8)));
    assert_eq!((inputs("5")["width"].clone(), inputs("5")["height"].clone()), (json!(1024), json!(576)));
    assert_eq!(job.uploads.len(), 1);
    assert!(job.uploads[0].name.starts_with("autocut-") && job.uploads[0].name.ends_with(".png"), "{:?}", job.uploads);
    assert!(s.log().is_empty(), "preparing talks to nobody");
    let out = d.join("made").join("Generated Image 1.png");
    let (r, notes, ticks) = go(&p, &job, &out, &AtomicBool::new(false));
    r.unwrap();
    assert_eq!(std::fs::read(&out).unwrap(), result);
    // what was sent: the upload, the filled workflow naming the uploaded picture, the result fetched by name
    let log = s.log();
    let prompt = log.iter().find(|q| q.route() == "/prompt").unwrap().json();
    assert_eq!(prompt["prompt"]["10"]["inputs"]["image"], "uploaded.png");
    assert_eq!(prompt["prompt"]["6"]["inputs"]["text"], "a red kite; rm -rf / \"quoted\", film still");
    assert_eq!(prompt["prompt"]["4"]["inputs"]["ckpt_name"], "open_model.safetensors", "the workflow's own model");
    let upload = log.iter().find(|q| q.route() == "/upload/image").unwrap();
    let form = String::from_utf8_lossy(&upload.body);
    assert!(form.contains("name=\"image\"; filename=\"autocut-") && form.contains("name=\"type\"\r\n\r\ninput"), "{form}");
    assert!(!form.contains(&*d.to_string_lossy()), "no path of this computer reaches the server");
    let view = log.iter().find(|q| q.route() == "/view").unwrap();
    assert_eq!(view.path, "/view?filename=ComfyUI_00001_.png&subfolder=&type=output");
    // the GPU is freed after the job, last
    assert_eq!(s.routes().last().map(String::as_str), Some("POST /free"));
    // where it was, as it went
    assert!(notes.iter().any(|n| n == "Waiting in ComfyUI's queue (1 ahead)"), "{notes:?}");
    assert!(notes.iter().any(|n| n == "Running in ComfyUI"), "{notes:?}");
    assert!(ticks >= 5, "{ticks}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn a_job_past_its_time_or_stopped_is_withdrawn_and_the_gpu_freed() {
    let d = dir("timeout");
    let stuck = |running: bool| {
        move |r: &Req, _n: usize| match (r.method.as_str(), r.route()) {
            ("GET", "/system_stats") => Act::Json(200, json!({})),
            ("POST", "/prompt") => Act::Json(200, json!({"prompt_id": "slow-1", "number": 1})),
            ("GET", "/queue") if running => Act::Json(200, json!({"queue_running": [[1, "slow-1", {}, {}, []]], "queue_pending": []})),
            ("GET", "/queue") => Act::Json(200, json!({"queue_running": [[0, "other", {}, {}, []]], "queue_pending": [[1, "slow-1", {}, {}, []]]})),
            ("GET", "/history/slow-1") => Act::Json(200, json!({})),
            _ => Act::Json(200, json!({})),
        }
    };
    // past its time limit while it runs: interrupted (only it), then the GPU freed
    let s = stand(stuck(true));
    let p = provider(&entry(&s.url, &workflow(&d, json!({})), "").replace(r#""timeoutSeconds":30"#, r#""timeoutSeconds":1"#));
    let job = prepare(p.comfyui.as_ref().unwrap(), &Vars { prompt: "x".into(), ..Default::default() }, "png").unwrap();
    let t0 = std::time::Instant::now();
    let (r, _, _) = go(&p, &job, &d.join("a.png"), &AtomicBool::new(false));
    assert!(r.unwrap_err().contains("did not finish in 1 s"));
    assert!(t0.elapsed() < Duration::from_secs(8));
    let log = s.log();
    let interrupt = log.iter().find(|q| q.route() == "/interrupt").expect("the running job is interrupted");
    assert_eq!(interrupt.json()["prompt_id"], "slow-1");
    assert_eq!(s.routes().last().map(String::as_str), Some("POST /free"));
    assert!(!d.join("a.png").exists());
    // stopped while it waits in the queue: taken out of the queue, nothing interrupted
    let s = stand(stuck(false));
    let p = provider(&entry(&s.url, &workflow(&d, json!({})), ""));
    let cancel = AtomicBool::new(false);
    let (mut ticks, mut notes) = (0, Vec::new());
    let r = run_with(
        &p,
        &job_for(&p),
        &d.join("b.png"),
        &cancel,
        &mut |_, _| {
            ticks += 1;
            if ticks == 3 {
                cancel.store(true, Ordering::Relaxed);
            }
        },
        &mut |n| notes.push(n),
        timing(),
    );
    assert!(r.unwrap_err().contains("stopped"));
    let log = s.log();
    assert!(log.iter().all(|q| q.route() != "/interrupt"), "another client's job is never interrupted");
    let delete = log.iter().find(|q| q.method == "POST" && q.route() == "/queue").expect("taken out of the queue");
    assert_eq!(delete.json()["delete"], json!(["slow-1"]));
    assert_eq!(s.routes().last().map(String::as_str), Some("POST /free"));
    let _ = std::fs::remove_dir_all(&d);
}

fn job_for(p: &Provider) -> Job {
    prepare(p.comfyui.as_ref().unwrap(), &Vars { prompt: "x".into(), ..Default::default() }, "png").unwrap()
}

#[test]
fn a_server_that_goes_away_mid_job_is_given_up_on() {
    let d = dir("drop");
    // it stops serving once the job is queued: every later connection is refused
    let s = stand(|r, _| match (r.method.as_str(), r.route()) {
        ("GET", "/system_stats") => Act::Json(200, json!({})),
        ("POST", "/prompt") => Act::Json(200, json!({"prompt_id": "gone-1", "number": 1})),
        _ => Act::Shutdown,
    });
    let p = provider(&entry(&s.url, &workflow(&d, json!({})), ""));
    let t0 = std::time::Instant::now();
    let (r, _, _) = go(&p, &job_for(&p), &d.join("a.png"), &AtomicBool::new(false));
    let e = r.unwrap_err();
    assert!(e.contains("stopped answering"), "{e}");
    assert!(t0.elapsed() < Duration::from_secs(8));
    // it accepts and hangs up without a word
    let s = stand(|r, _| match (r.method.as_str(), r.route()) {
        ("GET", "/system_stats") => Act::Json(200, json!({})),
        ("POST", "/prompt") => Act::Json(200, json!({"prompt_id": "mute-1", "number": 1})),
        _ => Act::Close,
    });
    let p = provider(&entry(&s.url, &workflow(&d, json!({})), ""));
    let (r, _, _) = go(&p, &job_for(&p), &d.join("a.png"), &AtomicBool::new(false));
    assert!(r.unwrap_err().contains("stopped answering"));
    // nobody there at all: said before anything is sent, with how to start it
    let free = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let p = provider(&entry(&format!("http://127.0.0.1:{free}"), &workflow(&d, json!({})), ""));
    let (r, _, _) = go(&p, &job_for(&p), &d.join("a.png"), &AtomicBool::new(false));
    let e = r.unwrap_err();
    assert!(e.contains("not answering") && e.contains("--listen 127.0.0.1"), "{e}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn hostile_answers_are_errors_never_panics() {
    let d = dir("hostile");
    let picture = d.join("start.png");
    std::fs::write(&picture, png()).unwrap();
    let deep = format!("{}{}", "[".repeat(100_000), "]".repeat(100_000));
    let history = |entry: Value| {
        move |r: &Req, _n: usize| match (r.method.as_str(), r.route()) {
            ("POST", "/prompt") => Act::Json(200, json!({"prompt_id": "h-1"})),
            ("GET", "/history/h-1") => Act::Json(200, json!({"h-1": entry.clone()})),
            ("GET", "/view") => Act::Bytes(b"DATA".to_vec()),
            _ => Act::Json(200, json!({"name": "up.png"})),
        }
    };
    type Handler = Box<dyn FnMut(&Req, usize) -> Act + Send>;
    let cases: Vec<(Handler, &str)> = vec![
        (
            Box::new(|r: &Req, _| {
                if r.route() == "/prompt" { Act::Raw(b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\nnot json".to_vec()) } else { Act::Json(200, json!({})) }
            }),
            "not JSON",
        ),
        (
            Box::new(|r: &Req, _| if r.route() == "/prompt" { Act::Json(200, json!({"prompt_id": "../../etc"})) } else { Act::Json(200, json!({})) }),
            "names no job",
        ),
        (Box::new(|r: &Req, _| if r.route() == "/prompt" { Act::Json(200, json!({"prompt_id": 7})) } else { Act::Json(200, json!({})) }), "names no job"),
        (
            Box::new(|r: &Req, _| match r.route() {
                "/prompt" => Act::Json(
                    400,
                    json!({"error": {"message": "Prompt outputs failed validation"}, "node_errors": {"5": {"errors": [{"message": "Value too big", "details": "width 99999 is more than 16384"}]}}}),
                ),
                _ => Act::Json(200, json!({})),
            }),
            "width 99999",
        ),
        (
            Box::new(move |r: &Req, _| match r.route() {
                "/prompt" => Act::Json(200, json!({"prompt_id": "h-1"})),
                "/history/h-1" => Act::Raw(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{deep}", deep.len()).into_bytes()),
                _ => Act::Json(200, json!({})),
            }),
            "cannot be used",
        ),
        (Box::new(history(json!({"outputs": {"9": {"images": "x"}}}))), "saved no file"),
        (Box::new(history(json!({"outputs": {"9": {"images": [{"filename": 7}]}}}))), "saved no file"),
        (Box::new(history(json!({"outputs": []}))), "names no outputs"),
        (
            Box::new(history(
                json!({"outputs": {}, "status": {"status_str": "error", "messages": [["execution_error", {"node_type": "KSampler", "exception_message": "CUDA out of memory"}]]}}),
            )),
            "KSampler: CUDA out of memory",
        ),
        (Box::new(|_: &Req, _| Act::Json(200, json!({}))), "names no file"),
        (Box::new(|r: &Req, _| if r.route() == "/upload/image" { Act::Json(413, json!({})) } else { Act::Json(200, json!({})) }), "did not take"),
    ];
    for (handler, want) in cases {
        let s = stand(handler);
        let p = provider(&entry(&s.url, &workflow(&d, json!({})), ""));
        // (a picture to upload only where the upload is what is answered badly)
        let upload = matches!(want, "names no file" | "did not take");
        let v = Vars { prompt: "x".into(), image: upload.then(|| picture.to_string_lossy().into_owned()), ..Default::default() };
        let job = prepare(p.comfyui.as_ref().unwrap(), &v, "png").unwrap();
        let (r, _, _) = go(&p, &job, &d.join("a.png"), &AtomicBool::new(false));
        let e = r.unwrap_err();
        assert!(e.contains(want), "{want}: {e}");
        assert_eq!(s.routes().last().map(String::as_str), Some("POST /free"), "{want}: the GPU is freed whatever happened");
        assert!(!d.join("a.png").exists(), "{want}");
    }
    // a result's name goes to the server encoded, and only the file asked for is written
    let s = stand(history(json!({"outputs": {"9": {"images": [{"filename": "../../x.png", "subfolder": "a b", "type": "output"}]}}})));
    let p = provider(&entry(&s.url, &workflow(&d, json!({})), ""));
    let (r, _, _) = go(&p, &job_for(&p), &d.join("a.png"), &AtomicBool::new(false));
    r.unwrap();
    assert!(s.log().iter().any(|q| q.path == "/view?filename=..%2F..%2Fx.png&subfolder=a%20b&type=output"), "{:?}", s.routes());
    assert_eq!(std::fs::read(d.join("a.png")).unwrap(), b"DATA");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn a_workflow_that_loads_an_unlisted_model_is_refused_before_it_runs() {
    let d = dir("models");
    let s = stand(|_, _| Act::Json(200, json!({})));
    let wf = |extra: Value| workflow(&d, extra);
    for (extra, why) in [
        (json!({"4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "other_model.safetensors"}}}), "other_model.safetensors"),
        (json!({"20": {"class_type": "LoraLoader", "inputs": {"lora_name": "style.safetensors", "strength_model": 1.0}}}), "style.safetensors"),
        (json!({"21": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "open_model.safetensors", "clip_name2": "t5xxl"}}}), "t5xxl"),
        (json!({"22": {"class_type": "SomeCustomNode", "inputs": {"weights": "models/unet/q4.gguf"}}}), "q4.gguf"),
        (json!({"23": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x.pth"}}}), "4x.pth"),
    ] {
        let p = provider(&entry(&s.url, &wf(extra.clone()), ""));
        let e = prepare(p.comfyui.as_ref().unwrap(), &Vars::default(), "png").unwrap_err();
        assert!(e.contains(why) && e.contains("does not list"), "{extra}: {e}");
    }
    // a model listed with its folder is the same model
    let p = provider(&entry(&s.url, &wf(json!({"4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl/open_model.safetensors"}}})), ""));
    assert!(prepare(p.comfyui.as_ref().unwrap(), &Vars::default(), "png").is_ok());
    // a request can never choose the model, and inputs must be there
    for (inputs, why) in
        [(r#""4.ckpt_name":"{prompt}""#, "may not choose a model"), (r#""99.text":"{prompt}""#, "does not have"), (r#""6.txt":"{prompt}""#, "does not have")]
    {
        let e = entry(&s.url, &wf(json!({})), "").replace(r#""6.text":"{prompt}, film still""#, inputs);
        let p = provider(&e);
        let err = prepare(p.comfyui.as_ref().unwrap(), &Vars::default(), "png").unwrap_err();
        assert!(err.contains(why), "{inputs}: {err}");
    }
    // the editor's own format is not a workflow AutoCut can queue
    let ui = d.join("ui.json");
    std::fs::write(&ui, r#"{"nodes": [], "links": [], "version": 0.4}"#).unwrap();
    let p = provider(&entry(&s.url, &ui.to_string_lossy(), ""));
    assert!(prepare(p.comfyui.as_ref().unwrap(), &Vars::default(), "png").unwrap_err().contains("Export (API)"));
    for bad in ["[]", "{", r#"{"1": 5}"#, "{}"] {
        std::fs::write(&ui, bad).unwrap();
        assert!(prepare(p.comfyui.as_ref().unwrap(), &Vars::default(), "png").is_err(), "{bad}");
    }
    assert!(s.log().is_empty(), "nothing reached the server");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn only_allowed_licences_run() {
    let d = dir("licences");
    let wf = workflow(&d, json!({}));
    let with = |license: &str, allow: &str| {
        let e = entry(DEFAULT_SERVER, &wf, allow).replace(r#""license":"Apache-2.0""#, &format!(r#""license":"{license}""#));
        provider(&e)
    };
    for (license, why) in [
        ("CC-BY-NC-4.0", "non-commercial"),
        ("CC BY-NC-SA 4.0", "non-commercial"),
        ("CC-BY-ND-4.0", "no-derivatives"),
        ("FLUX.1 [dev] Non-Commercial License", "non-commercial"),
        ("Some Community License", "not on the allow-list"),
    ] {
        let p = with(license, "");
        let e = prepare(p.comfyui.as_ref().unwrap(), &Vars::default(), "png").unwrap_err();
        assert!(e.contains(why), "{license}: {e}");
        assert!(p.comfyui.as_ref().unwrap().licence_problem().is_some());
    }
    // open licences run as they are; another runs when the entry names it after reading it
    for license in ["Apache-2.0", "mit", "CC0-1.0", "cc by 4.0", "BSD_3_Clause"] {
        assert!(prepare(with(license, "").comfyui.as_ref().unwrap(), &Vars::default(), "png").is_ok(), "{license}");
    }
    let p = with("Some Community License", r#","allowLicenses":["some community license"]"#);
    assert!(prepare(p.comfyui.as_ref().unwrap(), &Vars::default(), "png").is_ok());
    // a listed model with a bad licence stops the workflow even if this one does not load it
    let e = entry(DEFAULT_SERVER, &wf, "").replace(
        r#""models":[{"file":"open_model.safetensors","license":"Apache-2.0"}]"#,
        r#""models":[{"file":"open_model.safetensors","license":"Apache-2.0"},{"file":"nc.safetensors","license":"CC-BY-NC-4.0"}]"#,
    );
    assert!(prepare(provider(&e).comfyui.as_ref().unwrap(), &Vars::default(), "png").unwrap_err().contains("nc.safetensors"));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn a_hostile_comfyui_entry_is_refused_when_the_list_is_read() {
    let wrap = |body: String| format!(r#"{{"providers":[{body}]}}"#);
    let wf = "/abs/workflow_api.json";
    for (bad, why) in [
        (entry("http://192.168.1.2:8188", wf, ""), "not on this computer"),
        (entry("https://127.0.0.1:8188", wf, ""), "http://"),
        (entry("http://127.0.0.1", wf, ""), "port"),
        (entry(DEFAULT_SERVER, "workflow_api.json", ""), "absolute"),
        (entry(DEFAULT_SERVER, "/abs/workflow.txt", ""), ".json"),
        (entry(DEFAULT_SERVER, wf, "").replace(r#""3.steps":8"#, r#""nodot":8"#), "<node id>.<input name>"),
        (entry(DEFAULT_SERVER, wf, "").replace(r#""3.steps":8"#, r#""3.steps":"{nope}""#), "unknown placeholder"),
        (entry(DEFAULT_SERVER, wf, "").replace(r#""10.image":"{image}""#, r#""10.image":"x {image}""#), "stands alone"),
        (entry(DEFAULT_SERVER, wf, "").replace(r#""3.steps":8"#, r#""3.steps":[8]"#), "takes a placeholder"),
        (entry(DEFAULT_SERVER, wf, "").replace(r#""output":"9""#, r#""output":"""#), "node id"),
        (entry(DEFAULT_SERVER, wf, "").replace(r#""license":"Apache-2.0""#, r#""license":"""#), "license"),
        (entry(DEFAULT_SERVER, wf, "").replace(r#""output":"9""#, r#""output":"9","shell":true"#), "unknown field"),
        (entry(DEFAULT_SERVER, wf, "").replace(r#""kind":"image","#, r#""kind":"image","command":["/bin/x","{out}"],"#), "no `command`"),
    ] {
        let e = run::parse(&wrap(bad.clone())).unwrap_err();
        assert!(e.contains(why), "{bad}: {e}");
    }
    let many: Vec<String> = (0..70).map(|k| format!(r#""{k}.text":"{{prompt}}""#)).collect();
    let e = entry(DEFAULT_SERVER, wf, "").replace(r#""3.steps":8"#, &many.join(","));
    assert!(run::parse(&wrap(e)).unwrap_err().contains("at most 64"));
    // a composition entry takes no workflow
    let e = run::parse(&wrap(format!(r#"{{"id":"raea","kind":"composition","name":"RAEA","comfyui":{{"workflow":"{wf}"}}}}"#))).unwrap_err();
    assert!(e.contains("comfyui"), "{e}");
    // what the entry says shows in what it can take
    let p =
        provider(&entry(DEFAULT_SERVER, wf, "").replace(r#""3.steps":8"#, r#""11.image":"{reference1}","12.image":"{reference2}","13.image":"{lastImage}""#));
    assert!(run::takes_references(&p) && run::max_references(&p) == 2 && run::takes_last_image(&p) && !run::takes_transparent(&p));
    assert!(!run::installed(&p) && run::missing(&p).contains("workflow"));
}

#[test]
fn the_place_of_a_job_in_the_queue_is_read() {
    let q = json!({"queue_running": [[3, "r"]], "queue_pending": [[9, "late"], [5, "mine"], [4, "early"]]});
    assert_eq!(place_in(&q, "r"), Some(Place::Running));
    assert_eq!(place_in(&q, "mine"), Some(Place::Waiting(2)), "the running one and the one numbered before it");
    assert_eq!(place_in(&q, "nobody"), None);
    for bad in [json!(null), json!([]), json!({"queue_pending": "x"}), json!({"queue_pending": [["x", "mine"]], "queue_running": 3})] {
        let _ = place_in(&bad, "mine");
    }
}

/// The provider list is one file per test binary (with the other generation tests).
#[test]
fn generate_image_runs_a_comfyui_workflow_end_to_end() {
    let _turn = run::LIST_TURN.lock().unwrap_or_else(PoisonError::into_inner);
    let d = dir("e2e");
    let result = png();
    let s = stand(move |r, n| match (r.method.as_str(), r.route()) {
        ("POST", "/prompt") => Act::Json(200, json!({"prompt_id": "e2e-1", "number": 1})),
        ("GET", "/history/e2e-1") if n < 2 => Act::Json(200, json!({})),
        ("GET", "/history/e2e-1") => Act::Json(200, done("e2e-1", json!({"9": {"images": [{"filename": "e2e.png", "subfolder": "", "type": "output"}]}}))),
        ("GET", "/view") => Act::Bytes(result.clone()),
        _ => Act::Json(200, json!({"system": {"comfyui_version": "0.0-test"}})),
    });
    let wf = workflow(&d, json!({}));
    let manifest = run::manifest_path().unwrap();
    std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    let caps = r#","capabilities":{"width":{"min":256,"max":2048,"step":64},"height":{"min":256,"max":2048,"step":64},"references":0}"#;
    std::fs::write(
        &manifest,
        format!(r#"{{"providers":[{}]}}"#, entry(&s.url, &wf, "").replace(r#","estimateSeconds":5"#, &format!("{caps},\"estimateSeconds\":5"))),
    )
    .unwrap();
    let mut session = crate::Session::default();
    session.execute("file.openDemoProject", json!({})).unwrap();
    let listed = session.execute("generate.providers", json!({})).unwrap();
    assert_eq!(listed["providers"][0]["runs"], "comfyui");
    assert_eq!(listed["providers"][0]["comfyui"]["server"], s.url.as_str());
    // checked without running: the workflow's models, the server answering
    let checked = session.execute("generate.checkTool", json!({"provider": "cu"})).unwrap();
    assert_eq!(checked["ok"], true, "{checked}");
    assert_eq!(checked["models"], json!(["open_model.safetensors"]));
    assert_eq!(checked["server"]["version"], "0.0-test");
    // what the entry cannot do is refused before ComfyUI hears of it
    let before = s.log().len();
    let e = session.execute("generate.image", json!({"prompt": "x", "width": 1000, "height": 1000})).unwrap_err().to_string();
    assert!(e.contains("nearest"), "{e}");
    assert_eq!(s.log().len(), before);
    let items0 = session.project.items.len();
    let v = session
        .execute("generate.image", json!({"prompt": "a lighthouse", "seed": 7, "dir": d.join("made").to_string_lossy(), "place": true, "wait": true}))
        .unwrap();
    assert_eq!(session.project.items.len(), items0 + 1, "{v}");
    let item = v["generated"][0]["item"].as_u64().unwrap();
    assert_eq!(session.project.item(filmcraft_project::ItemId(item)).unwrap().metadata.get("Generator").map(String::as_str), Some("cu"));
    assert!(v["generated"][0]["clip"].as_u64().is_some(), "{v}");
    let prompt = s.log().iter().find(|q| q.route() == "/prompt").unwrap().json();
    assert_eq!(prompt["prompt"]["3"]["inputs"]["seed"], 7);
    assert_eq!(prompt["prompt"]["6"]["inputs"]["text"], "a lighthouse, film still");
    assert_eq!(s.routes().last().map(String::as_str), Some("POST /free"));
    // a tool with problems says them all, and still runs nothing
    drop(s);
    let checked = session.execute("generate.checkTool", json!({"provider": "cu"})).unwrap();
    assert_eq!(checked["ok"], false);
    assert!(checked["problems"].as_array().unwrap().iter().any(|p| p.as_str().unwrap().contains("not answering")), "{checked}");
    assert!(session.execute("generate.checkTool", json!({"provider": "nope"})).is_err());
    assert!(session.execute("generate.checkTool", json!({})).is_err());
    std::fs::remove_file(&manifest).unwrap();
    let _ = std::fs::remove_dir_all(&d);
}
