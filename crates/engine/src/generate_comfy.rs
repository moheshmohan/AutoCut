//! A generation tool that is a workflow of a local ComfyUI server: the `comfyui` block of an
//! entry in the provider list ([`crate::generate_run`]).
//!
//! ComfyUI is a separate program the user installs and starts (`--listen 127.0.0.1`); AutoCut
//! only talks HTTP to it, on this computer: it never links, ships or reads its code. The user
//! keeps an API-format workflow file ("Export (API)" in ComfyUI); the entry names it and says
//! which node inputs take the request's values. For each job AutoCut fills a copy of the
//! workflow, uploads the pictures it needs (`/upload/image`), queues it (`POST /prompt`), follows
//! it (`/history/<id>`, `/queue`), fetches the result (`/view`) and then asks ComfyUI to unload
//! its models (`POST /free`), so the GPU lock the job held means the GPU is free afterwards.
//!
//! ```json
//! {"id": "sdxl", "kind": "image", "name": "SDXL (ComfyUI)",
//!  "comfyui": {
//!    "server": "http://127.0.0.1:8188",
//!    "workflow": "/absolute/path/sdxl_api.json",
//!    "inputs": {"6.text": "{prompt}", "7.text": "{negative}", "3.seed": "{seed}", "5.width": "{width}", "5.height": "{height}"},
//!    "output": "9",
//!    "models": [{"file": "sd_xl_base_1.0.safetensors", "license": "CreativeML-OpenRAIL-M"}],
//!    "allowLicenses": ["CreativeML-OpenRAIL-M"]
//!  }}
//! ```
//!
//! **Models and their licences.** Every model the workflow loads (a loader node's model input,
//! or any input naming a weights file) must be listed in `models` with its licence, and every
//! listed licence must be on the allow-list: open licences by default (MIT, Apache-2.0, BSD,
//! ISC, zlib, CC0, CC BY, CC BY-SA), plus what the entry adds in `allowLicenses` after its owner
//! read them. No non-commercial or no-derivatives licence is allowed unless the entry names it.
//! A request can never choose a model: `inputs` may not point at a model input.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::generate_run::Vars;

/// The server when the entry names none (ComfyUI's own default).
pub const DEFAULT_SERVER: &str = "http://127.0.0.1:8188";
/// The most a workflow file may weigh.
const WORKFLOW_MAX: u64 = 8 * 1024 * 1024;
const MAX_NODES: usize = 4000;
const MAX_INPUTS: usize = 64;
const MAX_MODELS: usize = 64;
const MAX_LICENSES: usize = 32;
/// The most reference pictures a workflow may take (`{reference1}` … `{reference16}`).
const MAX_REFS: usize = 16;

/// Licences allowed without the entry naming them (SPDX ids, compared without case).
const OPEN_LICENSES: [&str; 14] = [
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "0bsd",
    "isc",
    "zlib",
    "unlicense",
    "cc0-1.0",
    "cc-by-3.0",
    "cc-by-4.0",
    "cc-by-sa-3.0",
    "cc-by-sa-4.0",
    "mpl-2.0",
];

/// File endings of model weights.
const WEIGHTS: [&str; 10] = [".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft", ".onnx", ".pkl", ".pt2"];

/// File endings that may be uploaded as a workflow's picture, clip or sound.
const UPLOADABLE: [&str; 16] = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif", "exr", "mp4", "mov", "webm", "mkv", "wav", "flac", "mp3"];

/// The values a workflow input can take from a request.
const PLACEHOLDERS: [&str; 15] = [
    "prompt",
    "negative",
    "width",
    "height",
    "seed",
    "seconds",
    "frames",
    "fps",
    "steps",
    "strength",
    "transparent",
    "lastFrame",
    "image",
    "lastImage",
    "video",
];

/// The `comfyui` block of an entry.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Config {
    /// The server, on this computer (default `http://127.0.0.1:8188`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server: Option<String>,
    /// The API-format workflow (an absolute path to a `.json` file).
    pub workflow: String,
    /// `"<node id>.<input>"` → what it takes: a placeholder (`"{seed}"`, typed as the input
    /// needs), text with placeholders, or a fixed number, text or switch.
    #[serde(default)]
    pub inputs: BTreeMap<String, Value>,
    /// The node whose output is the result (default: the first saved file of the right type).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// The models the workflow loads, with their licences.
    #[serde(default)]
    pub models: Vec<Model>,
    /// Licences allowed besides the open ones (SPDX ids or the licence's own name).
    #[serde(default)]
    pub allow_licenses: Vec<String>,
}

/// A model a workflow loads.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Model {
    /// Its file as the workflow names it (`flux1-schnell.safetensors`, or with its folder).
    pub file: String,
    pub license: String,
}

/// A licence name compared without case, spaces or underscores.
fn norm(license: &str) -> String {
    license.trim().to_ascii_lowercase().replace([' ', '_'], "-")
}

/// Whether a licence says it is non-commercial or allows no derivatives.
pub fn restrictive(license: &str) -> bool {
    let n = norm(license);
    n.contains("-nc") || n.contains("noncommercial") || n.contains("non-commercial") || n.contains("-nd") || n.contains("noderiv") || n.contains("no-deriv")
}

impl Config {
    /// The server's address as given (or the default).
    pub fn server_url(&self) -> &str {
        self.server.as_deref().unwrap_or(DEFAULT_SERVER)
    }

    fn allowed(&self, license: &str) -> bool {
        let n = norm(license);
        OPEN_LICENSES.contains(&n.as_str()) || self.allow_licenses.iter().any(|a| norm(a) == n)
    }

    /// The licence problems of the listed models (the first one), without reading the workflow.
    pub fn licence_problem(&self) -> Option<String> {
        self.models.iter().find(|m| !self.allowed(&m.license)).map(|m| {
            let kind = if restrictive(&m.license) { "a non-commercial or no-derivatives licence" } else { "a licence not on the allow-list" };
            format!(
                "the model {} has {kind} ({}): AutoCut runs only open licences unless the entry's `allowLicenses` names it",
                short(&m.file, 120),
                short(&m.license, 80)
            )
        })
    }

    /// Whether any input takes the placeholder `name`.
    pub fn uses(&self, name: &str) -> bool {
        let tag = format!("{{{name}}}");
        self.inputs.values().any(|v| v.as_str().is_some_and(|s| s.contains(&tag)))
    }

    /// How many reference pictures the workflow takes: the highest `{referenceN}` it uses.
    pub fn references(&self) -> usize {
        (1..=MAX_REFS).rev().find(|k| self.uses(&format!("reference{k}"))).unwrap_or(0)
    }
}

fn short(t: &str, n: usize) -> String {
    t.chars().take(n).collect()
}

/// The placeholders in a text (`{name}`); an unknown one is an error.
fn placeholders(text: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find('{') {
        let after = rest.get(i + 1..).unwrap_or("");
        let Some(j) = after.find('}') else { break };
        let name = after.get(..j).unwrap_or("");
        if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric()) {
            let reference = name.strip_prefix("reference").and_then(|k| k.parse::<usize>().ok()).is_some_and(|k| (1..=MAX_REFS).contains(&k));
            if !PLACEHOLDERS.contains(&name) && !reference {
                return Err(format!("unknown placeholder {{{}}}", short(name, 40)));
            }
            out.push(name.to_string());
        }
        rest = after.get(j + 1..).unwrap_or("");
    }
    Ok(out)
}

/// `"<node>.<input>"` split, each part checked.
fn target(key: &str) -> Result<(&str, &str), String> {
    let bad = || format!("`comfyui.inputs` key `{}` must be \"<node id>.<input name>\"", short(key, 80));
    let (node, input) = key.rsplit_once('.').ok_or_else(bad)?;
    let ok = |s: &str, n: usize| !s.is_empty() && s.len() <= n && !s.chars().any(char::is_control);
    if !ok(node, 64) || !ok(input, 128) {
        return Err(bad());
    }
    Ok((node, input))
}

/// The block checked on its own (when the provider list is read): the server on this computer,
/// an absolute `.json` workflow, inputs and models bounded and well formed. Licences and the
/// workflow itself are checked when it is used.
pub fn check_config(c: &Config) -> Result<(), String> {
    #[cfg(not(target_arch = "wasm32"))]
    crate::loopback_http::Endpoint::parse(c.server_url()).map_err(|e| format!("`comfyui.server` {e}"))?;
    let wf = std::path::Path::new(&c.workflow);
    if !wf.is_absolute() || c.workflow.contains('\0') || c.workflow.len() > 4096 || wf.extension().and_then(|e| e.to_str()) != Some("json") {
        return Err("`comfyui.workflow` must be the absolute path of an API-format workflow (.json)".into());
    }
    if c.inputs.len() > MAX_INPUTS {
        return Err(format!("`comfyui.inputs` names at most {MAX_INPUTS} inputs"));
    }
    for (key, v) in &c.inputs {
        target(key)?;
        match v {
            Value::String(t) => {
                if t.len() > 8192 || t.contains('\0') {
                    return Err(format!("`comfyui.inputs` `{}` is too long", short(key, 80)));
                }
                let names = placeholders(t).map_err(|e| format!("`comfyui.inputs` `{}`: {e}", short(key, 80)))?;
                // a picture, clip or sound is uploaded and named alone
                let files = names.iter().filter(|n| is_file(n)).count();
                if files > 0 && (names.len() != 1 || t.trim() != format!("{{{}}}", names.first().map_or("", String::as_str))) {
                    return Err(format!("`comfyui.inputs` `{}`: a file placeholder stands alone", short(key, 80)));
                }
            }
            Value::Number(_) | Value::Bool(_) => {}
            _ => return Err(format!("`comfyui.inputs` `{}` takes a placeholder, text, a number or true/false", short(key, 80))),
        }
    }
    if let Some(o) = &c.output
        && (o.is_empty() || o.len() > 64 || o.chars().any(char::is_control))
    {
        return Err("`comfyui.output` is a node id".into());
    }
    if c.models.len() > MAX_MODELS || c.allow_licenses.len() > MAX_LICENSES {
        return Err(format!("`comfyui.models` lists at most {MAX_MODELS} models and `allowLicenses` at most {MAX_LICENSES}"));
    }
    let shape = |s: &str, n: usize| !s.trim().is_empty() && s.len() <= n && !s.chars().any(char::is_control);
    if c.models.iter().any(|m| !shape(&m.file, 512) || !shape(&m.license, 200)) || c.allow_licenses.iter().any(|l| !shape(l, 200)) {
        return Err("`comfyui.models` needs each model's `file` and `license`".into());
    }
    Ok(())
}

/// Placeholders whose value is a file to upload.
fn is_file(name: &str) -> bool {
    matches!(name, "image" | "lastImage" | "video") || name.starts_with("reference")
}

/// A string input that names model weights.
fn weights(v: &str) -> bool {
    let l = v.to_ascii_lowercase();
    WEIGHTS.iter().any(|w| l.ends_with(w))
}

/// Whether an input of a node is where a model is chosen: a loader's model field, or any value
/// naming a weights file.
fn model_input(class: &str, input: &str, value: &Value) -> bool {
    let named = value.as_str().is_some_and(weights);
    let loader = class.to_ascii_lowercase().contains("loader");
    let field = input.trim_end_matches(|c: char| c.is_ascii_digit());
    named || (loader && value.is_string() && (field.ends_with("_name") || matches!(field, "model" | "model_path" | "ckpt_path")))
}

/// The file name of a model as a workflow names it (after its folder).
fn base(file: &str) -> &str {
    file.rsplit(['/', '\\']).next().unwrap_or(file)
}

/// A workflow read from its file: an API-format object of nodes, bounded.
#[cfg(not(target_arch = "wasm32"))]
fn read_workflow(path: &str) -> Result<Map<String, Value>, String> {
    use std::io::Read as _;
    let f = std::fs::File::open(path).map_err(|e| format!("the workflow {path}: {e}"))?;
    let mut text = String::new();
    f.take(WORKFLOW_MAX + 1).read_to_string(&mut text).map_err(|e| format!("the workflow {path}: {e}"))?;
    if text.len() as u64 > WORKFLOW_MAX {
        return Err(format!("the workflow {path} is larger than 8 MiB"));
    }
    workflow_of(&text).map_err(|e| format!("the workflow {path}: {e}"))
}

#[cfg(target_arch = "wasm32")]
fn read_workflow(_: &str) -> Result<Map<String, Value>, String> {
    Err("ComfyUI workflows run on the desktop app".into())
}

/// The nodes of an API-format workflow.
pub fn workflow_of(text: &str) -> Result<Map<String, Value>, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("not valid JSON: {e}"))?;
    let Value::Object(nodes) = v else { return Err("not an API-format workflow (an object of nodes)".into()) };
    if nodes.contains_key("nodes") && nodes.contains_key("links") {
        return Err("this is the editor's format: save it with Export (API) in ComfyUI".into());
    }
    if nodes.is_empty() || nodes.len() > MAX_NODES {
        return Err(format!("a workflow has 1 to {MAX_NODES} nodes"));
    }
    for (id, node) in &nodes {
        let ok = node.get("class_type").and_then(Value::as_str).is_some() && node.get("inputs").is_some_and(Value::is_object);
        if !ok {
            return Err(format!("node {} has no class_type and inputs: not an API-format workflow (Export (API) in ComfyUI)", short(id, 40)));
        }
    }
    Ok(nodes)
}

/// The models a workflow loads: (node, input, file).
fn models_in(nodes: &Map<String, Value>) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for (id, node) in nodes {
        let class = node.get("class_type").and_then(Value::as_str).unwrap_or("");
        let Some(inputs) = node.get("inputs").and_then(Value::as_object) else { continue };
        for (name, v) in inputs {
            if model_input(class, name, v) {
                out.push((id.clone(), name.clone(), v.as_str().unwrap_or("").to_string()));
            }
        }
    }
    out
}

/// What a workflow is checked for before it runs: its inputs are there and choose no model, and
/// every model it loads is listed with an allowed licence. The models it loads, as found.
pub fn inspect(c: &Config, nodes: &Map<String, Value>) -> Result<Vec<String>, String> {
    if let Some(p) = c.licence_problem() {
        return Err(p);
    }
    for key in c.inputs.keys() {
        let (node, input) = target(key)?;
        let Some(n) = nodes.get(node) else { return Err(format!("`comfyui.inputs` names node {node}, which the workflow does not have")) };
        let class = n.get("class_type").and_then(Value::as_str).unwrap_or("");
        let Some(v) = n.get("inputs").and_then(|i| i.get(input)) else {
            return Err(format!("`comfyui.inputs` names {input} of node {node} ({class}), which it does not have"));
        };
        if model_input(class, input, v) {
            return Err(format!("`comfyui.inputs` may not choose a model ({input} of node {node}, {class}): a workflow's models are its own"));
        }
    }
    if let Some(o) = &c.output
        && !nodes.contains_key(o)
    {
        return Err(format!("`comfyui.output` names node {o}, which the workflow does not have"));
    }
    let mut found = BTreeSet::new();
    for (node, input, file) in models_in(nodes) {
        let listed = c.models.iter().find(|m| m.file == file || base(&m.file) == base(&file));
        if listed.is_none() {
            return Err(format!(
                "the workflow loads {} ({input} of node {node}), which `comfyui.models` does not list: list it with its licence",
                short(&file, 160)
            ));
        }
        found.insert(file);
    }
    Ok(found.into_iter().collect())
}

/// The workflow read and checked as a job would be (inputs, models, licences), without running
/// it: the models it loads.
pub fn check_workflow(c: &Config) -> Result<Vec<String>, String> {
    let nodes = read_workflow(&c.workflow)?;
    inspect(c, &nodes)
}

/// A file for the workflow: uploaded, then named in the inputs that take it.
#[derive(Clone, Debug, PartialEq)]
pub struct Upload {
    pub path: String,
    /// The name it is uploaded as.
    pub name: String,
    pub targets: Vec<(String, String)>,
}

/// A filled workflow, ready to queue.
#[derive(Clone, Debug)]
pub struct Job {
    pub server: String,
    pub workflow: Map<String, Value>,
    pub uploads: Vec<Upload>,
    pub output: Option<String>,
    /// The file type the result should have.
    pub extension: String,
}

fn file_value(v: &Vars, name: &str) -> Option<String> {
    match name {
        "image" => v.image.clone(),
        "lastImage" => v.last_image.clone(),
        "video" => v.video.clone(),
        _ => name.strip_prefix("reference").and_then(|k| k.parse::<usize>().ok()).and_then(|k| v.references.get(k.checked_sub(1)?).cloned()),
    }
    .filter(|s| !s.is_empty())
}

/// A request value typed as a workflow input wants it; `None` when the request did not give it.
fn typed(v: &Vars, name: &str) -> Option<Value> {
    use serde_json::json;
    match name {
        "prompt" => Some(json!(v.prompt)),
        "negative" => v.negative.clone().filter(|s| !s.is_empty()).map(Value::from),
        "width" => v.width.map(Value::from),
        "height" => v.height.map(Value::from),
        "seed" => Some(json!(v.seed)),
        "seconds" => v.seconds.filter(|x| x.is_finite()).map(Value::from),
        "frames" => v.frames.map(Value::from),
        "lastFrame" => v.frames.map(|n| Value::from(n.saturating_sub(1))),
        "fps" => v.fps.filter(|x| x.is_finite()).map(Value::from),
        "steps" => v.steps.map(Value::from),
        "strength" => v.strength.filter(|x| x.is_finite()).map(Value::from),
        "transparent" => v.transparent.then_some(Value::Bool(true)),
        _ => None,
    }
}

/// Text with its placeholders filled (a value the request did not give is empty).
fn fill_text(t: &str, v: &Vars) -> String {
    let mut out = String::with_capacity(t.len());
    let mut rest = t;
    while let Some(i) = rest.find('{') {
        out.push_str(rest.get(..i).unwrap_or(""));
        let after = rest.get(i + 1..).unwrap_or("");
        let Some(j) = after.find('}') else {
            out.push('{');
            rest = after;
            continue;
        };
        let name = after.get(..j).unwrap_or("");
        match typed(v, name) {
            Some(Value::String(s)) => out.push_str(&s),
            Some(Value::Number(n)) => out.push_str(&n.to_string()),
            Some(Value::Bool(b)) => out.push_str(if b { "true" } else { "false" }),
            Some(_) => {}
            None if PLACEHOLDERS.contains(&name) => {}
            None => {
                out.push('{');
                out.push_str(name);
                out.push('}');
            }
        }
        rest = after.get(j + 1..).unwrap_or("");
    }
    out.push_str(rest);
    out
}

/// The name a file is uploaded as: `autocut-<hash>.<ext>` (its path, size and time), so the
/// same file is not uploaded under many names and nothing of its path reaches the server.
fn upload_name(path: &str) -> Result<String, String> {
    let ext = std::path::Path::new(path).extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).unwrap_or_default();
    if !UPLOADABLE.contains(&ext.as_str()) {
        return Err(format!("{} cannot be given to a workflow (pictures, clips and sounds only: {})", short(path, 160), UPLOADABLE.join(", ")));
    }
    let meta = if cfg!(target_arch = "wasm32") { None } else { std::fs::metadata(path).ok() };
    let stamp = meta.as_ref().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in path.bytes().chain(meta.map_or(0, |m| m.len()).to_le_bytes()).chain(stamp.to_le_bytes()) {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(format!("autocut-{h:016x}.{ext}"))
}

/// A request turned into a job: the workflow read and checked, the request's values put in a
/// copy, the files it needs named for upload. Everything that can be refused is refused here,
/// before anything runs.
pub fn prepare(c: &Config, v: &Vars, extension: &str) -> Result<Job, String> {
    let mut nodes = read_workflow(&c.workflow)?;
    inspect(c, &nodes)?;
    let mut uploads: Vec<Upload> = Vec::new();
    for (key, given) in &c.inputs {
        let (node, input) = target(key)?;
        let value = match given {
            Value::String(t) => {
                let names = placeholders(t)?;
                match names.as_slice() {
                    [one] if t.trim() == format!("{{{one}}}") && is_file(one) => {
                        let Some(path) = file_value(v, one) else { continue };
                        let name = upload_name(&path)?;
                        match uploads.iter_mut().find(|u| u.path == path) {
                            Some(u) => u.targets.push((node.to_string(), input.to_string())),
                            None => uploads.push(Upload { path, name, targets: vec![(node.to_string(), input.to_string())] }),
                        }
                        continue;
                    }
                    // one placeholder alone takes the value's own type (a seed stays a number)
                    [one] if t.trim() == format!("{{{one}}}") => match typed(v, one) {
                        Some(x) => x,
                        None => continue,
                    },
                    _ => Value::String(fill_text(t, v)),
                }
            }
            other => other.clone(),
        };
        if let Some(slot) = nodes.get_mut(node).and_then(|n| n.get_mut("inputs")).and_then(Value::as_object_mut) {
            slot.insert(input.to_string(), value);
        }
    }
    Ok(Job { server: c.server_url().to_string(), workflow: nodes, uploads, output: c.output.clone(), extension: extension.to_ascii_lowercase() })
}

/// A saved file a finished job names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Saved {
    pub filename: String,
    pub subfolder: String,
    pub kind: String,
}

/// The result of a finished job's history entry: the output node's (or the first node's) saved
/// file of the wanted type, else its first saved file.
pub fn pick_output(entry: &Value, output: Option<&str>, extension: &str) -> Result<Saved, String> {
    let outputs = entry.get("outputs").and_then(Value::as_object).ok_or("the finished job names no outputs")?;
    let mut files: Vec<(bool, Saved)> = Vec::new();
    for (node, out) in outputs {
        if output.is_some_and(|o| o != node) {
            continue;
        }
        let Some(lists) = out.as_object() else { continue };
        for list in lists.values().filter_map(Value::as_array).take(16) {
            for f in list.iter().take(64) {
                let (Some(name), sub, kind) = (
                    f.get("filename").and_then(Value::as_str),
                    f.get("subfolder").and_then(Value::as_str).unwrap_or(""),
                    f.get("type").and_then(Value::as_str).unwrap_or("output"),
                ) else {
                    continue;
                };
                let ok = !name.is_empty() && name.len() <= 1024 && sub.len() <= 1024 && !name.contains('\0') && !sub.contains('\0');
                if !ok || !matches!(kind, "output" | "temp" | "input") {
                    continue;
                }
                // a preview (type temp) is a result only from the node the entry names
                if kind != "output" && output.is_none() {
                    continue;
                }
                let matches = std::path::Path::new(name).extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case(extension));
                files.push((matches, Saved { filename: name.to_string(), subfolder: sub.to_string(), kind: kind.to_string() }));
            }
        }
    }
    files.iter().find(|(m, _)| *m).or(files.first()).map(|(_, s)| s.clone()).ok_or_else(|| match output {
        Some(o) => format!("the finished job saved no file from node {o}"),
        None => "the finished job saved no file (is there a Save node in the workflow?)".to_string(),
    })
}

/// Why a finished job failed, from its history entry's status (`None`: it did not fail).
pub fn failure(entry: &Value) -> Option<String> {
    let status = entry.get("status")?;
    if status.get("status_str").and_then(Value::as_str) != Some("error") {
        return None;
    }
    let messages = status.get("messages").and_then(Value::as_array);
    let said = messages.into_iter().flatten().take(256).find_map(|m| {
        let pair = m.as_array()?;
        (pair.first()?.as_str()? == "execution_error").then_some(())?;
        let d = pair.get(1)?;
        let node = d.get("node_type").and_then(Value::as_str).unwrap_or("a node");
        let msg = d.get("exception_message").and_then(Value::as_str).unwrap_or("failed");
        Some(format!("{}: {}", short(node, 80), short(msg.trim(), 300)))
    });
    Some(said.unwrap_or_else(|| "the workflow failed".into()))
}

/// What ComfyUI said when it refused a workflow (`POST /prompt` answered 400).
pub fn refusal(body: &[u8]) -> String {
    let Ok(v) = serde_json::from_slice::<Value>(body) else { return "it gave no reason".into() };
    let top = v.get("error").and_then(|e| e.get("message").or(e.get("details"))).and_then(Value::as_str).unwrap_or("");
    let nodes = v.get("node_errors").and_then(Value::as_object).into_iter().flatten().take(4).filter_map(|(id, e)| {
        let first = e.get("errors")?.as_array()?.first()?;
        let msg = first.get("details").or(first.get("message")).and_then(Value::as_str)?;
        Some(format!("node {}: {}", short(id, 20), short(msg, 160)))
    });
    let all: Vec<String> = std::iter::once(short(top, 200)).filter(|t| !t.is_empty()).chain(nodes).collect();
    if all.is_empty() { "it gave no reason".into() } else { all.join("; ") }
}

/// How often a job is looked at and how long a silent server is waited for.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Timing {
    pub poll: std::time::Duration,
    /// How long ComfyUI may not answer before the job is given up.
    pub give_up: std::time::Duration,
    pub connect: std::time::Duration,
    pub io: std::time::Duration,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for Timing {
    fn default() -> Self {
        use std::time::Duration;
        Timing { poll: Duration::from_millis(500), give_up: Duration::from_secs(30), connect: Duration::from_secs(3), io: Duration::from_secs(60) }
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};

    use serde_json::{Value, json};

    use super::{Job, Timing, failure, pick_output, refusal, short};
    use crate::generate_run::Provider;
    use crate::loopback_http::{self as http, Body, Endpoint, Fail, Limits};

    /// The largest answer kept in memory (a job's history holds its workflow).
    const JSON_MAX: u64 = 32 * 1024 * 1024;
    /// The largest result fetched.
    const FILE_MAX: u64 = 8 * 1024 * 1024 * 1024;

    struct Server {
        ep: Endpoint,
        t: Timing,
    }

    impl Server {
        fn limits(&self, max: u64) -> Limits {
            Limits { connect: self.t.connect, io: self.t.io, max_body: max }
        }
        fn json(&self, method: &str, path: &str, body: Option<&Value>) -> Result<(u16, Value), Fail> {
            let text = body.map(Value::to_string);
            let b = text.as_deref().map_or(Body::Empty, |t| Body::Bytes(t.as_bytes(), "application/json"));
            let r = http::call(&self.ep, method, path, b, self.limits(JSON_MAX))?;
            if r.status != 200 {
                return Ok((r.status, serde_json::from_slice(&r.body).unwrap_or(Value::String(String::from_utf8_lossy(&r.body).chars().take(300).collect()))));
            }
            let v = serde_json::from_slice(&r.body).map_err(|e| Fail::Bad(format!("not JSON: {e}")))?;
            Ok((200, v))
        }
        /// Best effort, never long: unload the models and free the memory.
        fn free(&self) {
            let quick = Server { ep: self.ep.clone(), t: Timing { io: self.t.io.min(std::time::Duration::from_secs(10)), ..self.t } };
            let _ = quick.json("POST", "/free", Some(&json!({"unload_models": true, "free_memory": true})));
        }
        /// Takes a job out: interrupted when it runs, deleted from the queue when it waits.
        fn withdraw(&self, id: &str) {
            if let Ok((200, q)) = self.json("GET", "/queue", None) {
                match place_in(&q, id) {
                    Some(Place::Running) => {
                        let _ = self.json("POST", "/interrupt", Some(&json!({"prompt_id": id})));
                    }
                    Some(Place::Waiting(_)) => {
                        let _ = self.json("POST", "/queue", Some(&json!({"delete": [id]})));
                    }
                    None => {}
                }
            }
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    pub(crate) enum Place {
        Running,
        /// How many jobs are ahead of it.
        Waiting(usize),
    }

    /// Where a job is in ComfyUI's queue.
    pub(crate) fn place_in(queue: &Value, id: &str) -> Option<Place> {
        let is = |e: &Value| e.as_array().and_then(|a| a.get(1)).and_then(Value::as_str) == Some(id);
        let running = queue.get("queue_running").and_then(Value::as_array);
        if running.is_some_and(|r| r.iter().take(4096).any(is)) {
            return Some(Place::Running);
        }
        let pending = queue.get("queue_pending").and_then(Value::as_array)?;
        // ComfyUI lists pending jobs by their number, not in order
        let number = |e: &Value| e.as_array().and_then(|a| a.first()).and_then(Value::as_f64).unwrap_or(f64::MAX);
        let mine = pending.iter().take(4096).find(|e| is(e)).map(number)?;
        let ahead = pending.iter().take(4096).filter(|e| !is(e) && number(e) < mine).count() + running.map_or(0, Vec::len);
        Some(Place::Waiting(ahead))
    }

    /// Runs a job: uploads, queues, follows and fetches it to `out`, then frees the GPU. `tick`
    /// hears the seconds gone by (ComfyUI says nothing of how far a job is without its socket),
    /// `note` what is happening; `cancel` withdraws the job; the entry's `timeoutSeconds` too.
    pub fn run(
        p: &Provider,
        job: &Job,
        out: &Path,
        cancel: &AtomicBool,
        tick: &mut dyn FnMut(f64, Option<f64>),
        note: &mut dyn FnMut(String),
    ) -> Result<(), String> {
        run_with(p, job, out, cancel, tick, note, Timing::default())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_with(
        p: &Provider,
        job: &Job,
        out: &Path,
        cancel: &AtomicBool,
        tick: &mut dyn FnMut(f64, Option<f64>),
        note: &mut dyn FnMut(String),
        t: Timing,
    ) -> Result<(), String> {
        let ep = Endpoint::parse(&job.server).map_err(|e| format!("{}: {e}", p.name))?;
        let server = Server { ep, t };
        let url = server.ep.url();
        note("Connecting to ComfyUI".into());
        match server.json("GET", "/system_stats", None) {
            Ok(_) => {}
            Err(Fail::Unreachable(e)) => {
                return Err(format!(
                    "{}: ComfyUI is not answering at {url} ({e}): start it on this computer (`--listen 127.0.0.1`, see docs/generation.md)",
                    p.name
                ));
            }
            Err(e) => return Err(format!("{}: ComfyUI at {url}: {e}", p.name)),
        }
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        // a file that is made again goes first: what is there afterwards is what this job wrote
        let _ = std::fs::remove_file(out);
        let r = queue_and_follow(p, &server, job, out, cancel, tick, note);
        // after every job, whatever became of it: the models leave the GPU
        server.free();
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn queue_and_follow(
        p: &Provider,
        server: &Server,
        job: &Job,
        out: &Path,
        cancel: &AtomicBool,
        tick: &mut dyn FnMut(f64, Option<f64>),
        note: &mut dyn FnMut(String),
    ) -> Result<(), String> {
        let name = &p.name;
        let mut workflow = job.workflow.clone();
        for u in &job.uploads {
            if cancel.load(Ordering::Relaxed) {
                return Err("stopped: nothing was changed".into());
            }
            note(format!("Giving ComfyUI {}", short(&u.name, 60)));
            let boundary = format!("autocut{:016x}", u.name.bytes().fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b))));
            let form = Body::Form {
                boundary: &boundary,
                fields: &[("type", "input"), ("overwrite", "true")],
                file_field: "image",
                file_name: &u.name,
                file: Path::new(&u.path),
            };
            let r =
                http::call(&server.ep, "POST", "/upload/image", form, server.limits(64 * 1024)).map_err(|e| format!("{name}: uploading {}: {e}", u.name))?;
            if r.status != 200 {
                return Err(format!("{name}: ComfyUI did not take {} (status {})", u.name, r.status));
            }
            let v: Value = serde_json::from_slice(&r.body).map_err(|_| format!("{name}: ComfyUI's answer to an upload is not JSON"))?;
            let got = v.get("name").and_then(Value::as_str).filter(|n| !n.is_empty() && n.len() <= 255 && !n.contains(['\0', '\n']));
            let Some(got) = got else { return Err(format!("{name}: ComfyUI's answer to an upload names no file")) };
            let sub = v.get("subfolder").and_then(Value::as_str).filter(|s| s.len() <= 255 && !s.contains(['\0', '\n'])).unwrap_or("");
            let value = if sub.is_empty() { got.to_string() } else { format!("{sub}/{got}") };
            for (node, input) in &u.targets {
                if let Some(slot) = workflow.get_mut(node).and_then(|n| n.get_mut("inputs")).and_then(Value::as_object_mut) {
                    slot.insert(input.clone(), Value::String(value.clone()));
                }
            }
        }
        note("Queueing the workflow".into());
        let (status, answer) =
            server.json("POST", "/prompt", Some(&json!({"prompt": Value::Object(workflow), "client_id": "autocut"}))).map_err(|e| format!("{name}: {e}"))?;
        if status != 200 {
            let why = serde_json::to_vec(&answer).map(|b| refusal(&b)).unwrap_or_default();
            return Err(format!("{name}: ComfyUI refused the workflow (status {status}): {why}"));
        }
        let id = answer
            .get("prompt_id")
            .and_then(Value::as_str)
            .filter(|i| !i.is_empty() && i.len() <= 128 && i.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'));
        let Some(id) = id.map(str::to_string) else { return Err(format!("{name}: ComfyUI's answer names no job")) };
        let limit = p.timeout_seconds.unwrap_or(3600.0).clamp(1.0, 86_400.0);
        let t0 = web_time::Instant::now();
        let mut heard = t0;
        let mut turn = 0u32;
        let entry = loop {
            let gone = t0.elapsed().as_secs_f64();
            if cancel.load(Ordering::Relaxed) {
                server.withdraw(&id);
                return Err("stopped: nothing was changed".into());
            }
            if gone > limit {
                server.withdraw(&id);
                return Err(format!("{name} did not finish in {limit:.0} s and was stopped"));
            }
            match server.json("GET", &format!("/history/{id}"), None) {
                Ok((200, v)) => {
                    heard = web_time::Instant::now();
                    if let Some(e) = v.get(&id) {
                        break e.clone();
                    }
                }
                Ok((s, _)) => return Err(format!("{name}: ComfyUI answered status {s} about the job")),
                Err(Fail::Unreachable(_) | Fail::Dropped(_)) => {
                    if heard.elapsed() > server.t.give_up {
                        return Err(format!("{name}: ComfyUI stopped answering at {} while the job ran", server.ep.url()));
                    }
                }
                Err(e) => return Err(format!("{name}: ComfyUI's answer about the job cannot be used: {e}")),
            }
            // now and then, where the job is in the queue
            if turn.is_multiple_of(4)
                && let Ok((200, q)) = server.json("GET", "/queue", None)
            {
                match place_in(&q, &id) {
                    Some(Place::Running) => note("Running in ComfyUI".into()),
                    Some(Place::Waiting(0)) => note("Next in ComfyUI's queue".into()),
                    Some(Place::Waiting(n)) => note(format!("Waiting in ComfyUI's queue ({n} ahead)")),
                    None => {}
                }
            }
            turn = turn.wrapping_add(1);
            tick(gone, None);
            std::thread::sleep(server.t.poll);
        };
        if let Some(why) = failure(&entry) {
            return Err(format!("{name}: {why}"));
        }
        let saved = pick_output(&entry, job.output.as_deref(), &job.extension).map_err(|e| format!("{name}: {e}"))?;
        note("Fetching the result".into());
        let path = format!("/view?filename={}&subfolder={}&type={}", http::encode(&saved.filename), http::encode(&saved.subfolder), http::encode(&saved.kind));
        let part = out.with_extension("comfyui.part");
        let fetched = http::fetch_to(&server.ep, &path, &part, server.limits(FILE_MAX));
        match fetched {
            Ok((200, n)) if n > 0 => {}
            Ok((200, _)) => {
                let _ = std::fs::remove_file(&part);
                return Err(format!("{name}: the result {} is empty", short(&saved.filename, 120)));
            }
            Ok((s, _)) => return Err(format!("{name}: ComfyUI did not give the result {} (status {s})", short(&saved.filename, 120))),
            Err(e) => return Err(format!("{name}: fetching the result: {e}")),
        }
        std::fs::rename(&part, out).map_err(|e| {
            let _ = std::fs::remove_file(&part);
            format!("{}: {e}", out.display())
        })
    }

    /// Whether ComfyUI answers at `url`, and its version when it says.
    pub fn probe(url: &str) -> Result<Option<String>, String> {
        let ep = Endpoint::parse(url)?;
        let server = Server { ep, t: Timing { io: std::time::Duration::from_secs(3), connect: std::time::Duration::from_secs(2), ..Timing::default() } };
        match server.json("GET", "/system_stats", None) {
            Ok((200, v)) => Ok(v.pointer("/system/comfyui_version").and_then(Value::as_str).map(|s| short(s, 40))),
            Ok((s, _)) => Err(format!("it answered status {s}")),
            Err(e) => Err(e.to_string()),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::{probe, run};

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) use native::{Place, place_in, run_with};

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "generate_comfy_tests.rs"]
mod tests;
