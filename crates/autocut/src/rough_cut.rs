//! A clean talking edit from raw clips (`autocut.roughCut`): the files are imported (or the
//! given items taken), their speech understood (`media.understandSpeech`: words, speakers,
//! pauses, fillers, retakes), the kept speech assembled on a new sequence (`sequence.roughCut`:
//! pauses, fillers and abandoned takes cut, every cut between words) and, when asked, exported
//! (`file.exportMedia`). No language model: the same clips always cut the same way.

use serde_json::{Value, json};

use crate::{AutoCutError, Executor};

async fn run(ex: &dyn Executor, id: &str, p: Value) -> Result<Value, AutoCutError> {
    let step = crate::PlannedStep::new(id, p.clone(), "rough cut");
    crate::validate::validate_step(&step).map_err(AutoCutError::Invalid)?;
    ex.execute(id, p).await.map_err(|e| AutoCutError::Executor(format!("{id}: {e}")))
}

/// The export format a file name asks for (the default, H.264 in MP4, otherwise).
fn format_for(path: &str) -> Option<&'static str> {
    let ext = path.rsplit_once('.')?.1.to_ascii_lowercase();
    Some(match ext.as_str() {
        "mp4" | "m4v" => "h264",
        "mov" => "prores",
        "wav" => "wav",
        "aif" | "aiff" => "aiff",
        "m4a" | "aac" => "aac",
        "flac" => "flac",
        _ => return None,
    })
}

/// Make the rough cut (parameters as the engine checked them: `items` or `paths`, `language`,
/// `diarize`, the cut's `removePauses`, `removeFillers`, `removeRetakes`, `removeHesitations`,
/// `minPause`, `keepPause`, `name`, and `export`).
pub async fn make(ex: &dyn Executor, p: &Value, lines: &mut dyn FnMut(&str)) -> Result<Value, AutoCutError> {
    let mut items: Vec<u64> = p["items"].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
    if let Some(paths) = p["paths"].as_array().filter(|a| !a.is_empty()) {
        lines(&format!("[1/4] importing {} file(s)", paths.len()));
        let r = run(ex, "file.import", json!({"paths": paths, "reuse": true})).await?;
        items.extend(r["items"].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect::<Vec<_>>()).unwrap_or_default());
    }
    if items.is_empty() {
        return Err(AutoCutError::Other("no clips to cut: give `items` or `paths`".into()));
    }
    lines(&format!("[2/4] understanding the speech of {} clip(s)", items.len()));
    let mut up = json!({"items": items, "wait": true});
    for k in ["language", "diarize", "model"] {
        if let Some(v) = p.get(k).filter(|v| !v.is_null()) {
            up[k] = v.clone();
        }
    }
    let understood = run(ex, "media.understandSpeech", up).await?;
    let analyses = understood["speech"]["items"].clone();
    lines("[3/4] assembling the rough cut");
    let mut cut = json!({"items": items});
    for k in ["removePauses", "removeFillers", "removeRetakes", "removeHesitations", "minPause", "keepPause", "name"] {
        if let Some(v) = p.get(k).filter(|v| !v.is_null()) {
            cut[k] = v.clone();
        }
    }
    let made = run(ex, "sequence.roughCut", cut).await?;
    let mut export = Value::Null;
    if let Some(path) = p["export"].as_str() {
        lines(&format!("[4/4] exporting {path}"));
        let mut e = json!({"path": path, "wait": true});
        if let Some(f) = format_for(path) {
            e["format"] = json!(f);
        }
        export = run(ex, "file.exportMedia", e).await?;
    }
    let removed = &made["removed"];
    let status = format!(
        "Rough cut: {:.1} s of {:.1} s kept ({} retake(s), {} filler(s), {} pause(s) cut)",
        made["duration"].as_f64().unwrap_or(0.0),
        made["source"].as_f64().unwrap_or(0.0),
        removed["retakes"].as_u64().unwrap_or(0),
        removed["fillers"].as_u64().unwrap_or(0),
        removed["pauses"].as_u64().unwrap_or(0)
    );
    Ok(json!({"items": items, "speech": analyses, "roughCut": made, "export": export, "status": status}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Answers the job's commands and records them.
    struct Fake(Mutex<Vec<(String, Value)>>);

    impl Executor for Fake {
        fn execute(&self, id: &str, params: Value) -> crate::ExecFuture {
            self.0.lock().unwrap().push((id.to_string(), params));
            let out = match id {
                "file.import" => json!({"items": [7, 8]}),
                "media.understandSpeech" => json!({"job": 1, "speech": {"items": [{"item": 7}, {"item": 8}]}}),
                "sequence.roughCut" => json!({"sequence": 9, "duration": 30.0, "source": 42.5, "removed": {"retakes": 2, "fillers": 3, "pauses": 4}}),
                "file.exportMedia" => json!({"path": "out.mp4"}),
                _ => json!({}),
            };
            Box::pin(async move { Ok(out) })
        }
        fn render(&self, _: f64, _: u32) -> crate::RenderFuture {
            Box::pin(async { None })
        }
    }

    #[test]
    fn imports_understands_cuts_and_exports() {
        let ex = Fake(Mutex::new(Vec::new()));
        let p = json!({"paths": ["/a.mov", "/b.mov"], "language": "ml", "removeRetakes": true, "export": "/tmp/out.wav"});
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let mut said = Vec::new();
        let r = rt.block_on(make(&ex, &p, &mut |l| said.push(l.to_string()))).unwrap();
        let calls: Vec<String> = ex.0.lock().unwrap().iter().map(|c| c.0.clone()).collect();
        assert_eq!(calls, ["file.import", "media.understandSpeech", "sequence.roughCut", "file.exportMedia"]);
        let up = ex.0.lock().unwrap()[1].1.clone();
        assert_eq!((up["items"].clone(), up["language"].as_str(), up["wait"].as_bool()), (json!([7, 8]), Some("ml"), Some(true)));
        assert_eq!(ex.0.lock().unwrap()[2].1["removeRetakes"], true);
        assert_eq!(ex.0.lock().unwrap()[3].1["format"], "wav", "the format the file name asks for");
        assert!(r["status"].as_str().unwrap().contains("2 retake(s)"), "{r}");
        assert_eq!(said.len(), 4);
        // nothing to cut is an error
        let ex = Fake(Mutex::new(Vec::new()));
        assert!(rt.block_on(make(&ex, &json!({}), &mut |_| {})).is_err());
    }
}
