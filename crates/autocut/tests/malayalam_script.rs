//! `autocut.malayalamScript` with this crate as its host: a plan's English voice-over translated
//! by a (scripted) model, fixed, checked and revised, written as the lines file `showcase-plan
//! --lines` takes, and the result checked again by `script.lintMalayalam` — all engine commands.

use std::sync::Arc;

use filmcraft_autocut::host::{Models, ShowcaseHost};
use filmcraft_autocut::llm::FnModel;
use filmcraft_engine::Session;
use serde_json::json;

#[test]
fn a_plan_becomes_a_checked_malayalam_script() {
    // the writer: a formal draft, then (asked to fix it) the spoken line
    let writer = FnModel::new("scripted-malayalam", |_, user, _| {
        Ok(if user.contains("need fixing") {
            json!({"lines": ["ഇവിടെ നോക്കിയാലും വിസിറ്റേഴ്സിനെ വെൽക്കം ചെയ്യുന്ന പെപ്പർ. സോഫ്റ്റ്ബാങ്ക് 27,000 എണ്ണം ഉണ്ടാക്കി."]})
        } else {
            json!({"lines": [
                "എല്ലാ റോബോട്ടുകളും ഇതിലുണ്ട്.",
                "ഇവിടെ നോക്കിയാലും സന്ദർശകരെ സ്വാഗതം ചെയ്യുന്ന പെപ്പർ. സോഫ്റ്റ്ബാങ്ക് ഉണ്ടാക്കി.",
                "കണ്ടതിന് താങ്ക്സ്."
            ]})
        })
    });
    let mut s = Session::default();
    s.autocut.host = Some(Arc::new(ShowcaseHost::with_models(Models { analysis: Models::scripted().analysis, script: Arc::new(writer) })));
    let plan = json!({
        "entries": [{"label": "Pepper"}],
        "script": {"intro": "This is every robot.", "lines": ["Pepper greets visitors; SoftBank made twenty-seven thousand of them."], "outro": "Thanks for watching."}
    });
    let dir = std::env::temp_dir().join(format!("fc-ml-script-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("script-ml.json");
    let r = s.execute("autocut.malayalamScript", json!({"plan": plan, "out": out.to_string_lossy(), "wait": true})).unwrap();
    let o = &r["showcase"];
    assert_eq!(o["malayalam"]["lines"]["Pepper"], "ഇവിടെ നോക്കിയാലും വിസിറ്റേഴ്സിനെ വെൽക്കം ചെയ്യുന്ന പെപ്പർ. സോഫ്റ്റ്ബാങ്ക് 27,000 എണ്ണം ഉണ്ടാക്കി.", "{o}");
    assert_eq!(
        (o["summary"]["revised"].as_u64(), o["summary"]["factsMissing"].as_u64(), o["summary"]["translated"].as_u64()),
        (Some(1), Some(0), Some(3)),
        "{o}"
    );
    // the file is the lines file, and it checks clean against the plan
    let written: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(written, o["malayalam"]);
    let lint = s.execute("script.lintMalayalam", json!({"path": out.to_string_lossy(), "english": plan})).unwrap();
    assert_eq!((lint["summary"]["warnings"].as_u64(), lint["summary"]["factsMissing"].as_u64()), (Some(0), Some(0)), "{lint}");
    std::fs::remove_dir_all(dir).ok();
}
