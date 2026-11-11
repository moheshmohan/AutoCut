//! Durable FrameAtlas handoffs to editable timelines. Retrieval stays in the local catalogue
//! adapter; this command validates the complete handoff before a single undoable assembly.
use crate::commands::{CommandSpec, always, bad};
use crate::{EngineError, Result, Session};
use filmcraft_project::{ItemId, MediaRef};
use filmcraft_time::Tick;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const CMD: &str = "catalog.importEditPlan";
const MAX_ENTRIES: usize = 2000;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    /// Complete FrameAtlas retrieve/job result (including source verification and part mapping).
    pub source: Value,
    #[serde(default, rename = "sourceIn")]
    pub source_in: Option<i64>,
    #[serde(default)]
    pub duration: Option<i64>,
    #[serde(default)]
    pub mute: bool,
    #[serde(default)]
    pub dashcam: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EditPlan {
    pub format: String,
    pub version: u32,
    pub name: String,
    pub entries: Vec<Entry>,
    #[serde(default)]
    pub sequence: Value,
}

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Import catalogue edit plan",
        menu: &[],
        shortcut: None,
        params: r#"{"plan":{"format":"autocut.catalogEditPlan","version":1,"name":str,"sequence":{…file.newSequence},"entries":[{"source":FrameAtlasHandoff,"sourceIn":ticks?,"duration":ticks?,"mute":bool=false,"dashcam":bool=false}]}}"#,
        enabled: always,
        run: import,
        journal: true,
    }]
}

fn source_path(e: &Entry) -> Result<&str> {
    let source = &e.source;
    if source.get("asset_id").and_then(Value::as_str).is_none_or(|s| s.is_empty() || s.len() > 256) {
        return Err(bad(CMD, "a source needs its FrameAtlas asset_id"));
    }
    if source.get("clip").is_some() && !matches!(source["indexed_source_verification"].as_str(), Some("matched" | "no_fingerprint")) {
        return Err(bad(CMD, "a scene/speech source must be verified by FrameAtlas; resolve and retrieve it again"));
    }
    source
        .get("clip")
        .and_then(|c| c.get("source_path"))
        .or_else(|| source.get("path"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty() && s.len() <= 4096)
        .ok_or_else(|| bad(CMD, "source has no retained editing path"))
}

fn seconds(v: &Value) -> Option<Tick> {
    v.as_f64().filter(|v| v.is_finite() && (0.0..=12.0 * 3600.0).contains(v)).map(Tick::from_seconds_f64)
}

fn import(s: &mut Session, p: &Value) -> Result<Value> {
    let plan: EditPlan = serde_json::from_value(p.get("plan").cloned().ok_or_else(|| bad(CMD, "provide plan"))?).map_err(|e| bad(CMD, e.to_string()))?;
    if plan.format != "autocut.catalogEditPlan"
        || plan.version != 1
        || plan.name.is_empty()
        || plan.name.len() > 256
        || plan.entries.is_empty()
        || plan.entries.len() > MAX_ENTRIES
        || (!plan.sequence.is_null() && !plan.sequence.is_object())
    {
        return Err(bad(CMD, "version 1 plan needs a name, sequence object and 1..2000 entries"));
    }
    // Validate paths and every explicit timing before importing anything. A missing original
    // never becomes a placeholder in a supposedly successful catalogue handoff.
    let mut fingerprints = BTreeMap::new();
    for e in &plan.entries {
        let path = source_path(e)?;
        if e.source["clip"]["kind"] == "360_sample" && (e.source_in.is_none() || e.duration.is_none()) {
            return Err(bad(CMD, "a 360 sample is a point citation; choose explicit sourceIn and duration ticks"));
        }
        if let Some(expected) = e.source.get("editing_sha256").and_then(Value::as_str) {
            if expected.len() != 64 {
                return Err(bad(CMD, "invalid retained source digest"));
            }
            if !fingerprints.contains_key(path) {
                fingerprints.insert(path.to_string(), crate::render_cache::digest_file(std::path::Path::new(path)).map_err(|e| bad(CMD, e))?.0);
            }
            if fingerprints.get(path).map(String::as_str) != Some(expected) {
                return Err(bad(CMD, "retrieved source changed; resolve/retrieve it again"));
            }
        }
        s.services.file_size(path).map_err(|err| bad(CMD, format!("{path}: {err}")))?;
        for n in [e.source_in, e.duration].into_iter().flatten() {
            if n < 0 || n > Tick::from_seconds_f64(12.0 * 3600.0).0 {
                return Err(bad(CMD, "timing outside 0..12 hours"));
            }
        }
        if e.duration == Some(0) {
            return Err(bad(CMD, "duration must be positive"));
        }
    }
    s.edit_group("Assemble catalogue edit plan", |s| {
        let mut q = if plan.sequence.is_object() { plan.sequence.clone() } else { json!({}) };
        q["name"] = json!(plan.name);
        q["video"] = json!(1); q["audio"] = json!(1);
        let sequence = s.execute("file.newSequence", q)?;
        let seq = ItemId(sequence["sequence"].as_u64().ok_or_else(|| bad(CMD, "new sequence returned no identity"))?);
        let mut items = BTreeMap::<String, ItemId>::new();
        let mut entries = Vec::new();
        let mut cursor = Tick::ZERO;
        for e in &plan.entries {
            let path = source_path(e)?.to_string();
            let existing = s.project.items.values().find_map(|i| match &i.kind {
                filmcraft_project::ItemKind::Media(m) if matches!(&m.media, MediaRef::File { path: p } if p == &path) => Some(i.id), _ => None
            });
            let item = match items.get(&path).copied().or(existing) {
                Some(id) => id,
                None => crate::commands::import_streamed(s, &path, None)?,
            };
            s.media.remove(item);
            items.insert(path, item);
            if let Some(files)=e.source.get("related_files") {
                let deps=files.as_array().filter(|a|a.len()<=1000).ok_or_else(||bad(CMD,"too many related source files"))?;
                for path in deps { s.services.file_size(path.as_str().ok_or_else(||bad(CMD,"invalid related source path"))?).map_err(|e|bad(CMD,e.to_string()))?; }
                s.edit("Keep source dependencies",|p,_|{if let Some(i)=p.item_mut(item){i.metadata.insert("frameatlas.dependencies".into(),files.to_string());}Ok(())})?;
            }
            let media = s.project.item(item).and_then(|i| i.as_media()).ok_or_else(|| bad(CMD, "retrieved source is not media"))?;
            let still = media.info.kind == filmcraft_media::MediaKind::Still;
            let clip = &e.source["clip"];
            let sin = e.source_in.map(Tick).or_else(|| seconds(&clip["source_start"])).unwrap_or(Tick::ZERO);
            let duration = match e.duration.map(Tick) {
                Some(d) => d,
                None if still => Tick::from_seconds_f64(4.0),
                None => seconds(&clip["source_end"]).and_then(|end| end.0.checked_sub(sin.0).map(Tick))
                    .unwrap_or_else(|| Tick(media.info.duration.0.saturating_sub(sin.0))),
            };
            let end = sin.0.checked_add(duration.0).ok_or_else(|| bad(CMD, "source range overflows"))?;
            if duration.0 <= 0 || (!still && end > media.info.duration.0) {
                return Err(bad(CMD, "source range is empty or beyond the retrieved original; 360 sample citations need an explicit sourceIn and duration"));
            }
            let placed = s.execute("timeline.place", json!({"item":item.0,"track":"V1","audioTrack":"A1", "time":cursor.0,
                "sourceIn":sin.0,"duration":duration.0,"audio":!(e.mute || e.dashcam)}))?;
            let end = s.active_sequence().into_iter().flat_map(|q| q.video_tracks.iter().chain(q.audio_tracks.iter())).flat_map(|t| t.items.iter()).map(|i| i.end()).max().ok_or(EngineError::NoSequence)?;
            entries.push(json!({"source":e.source,"sourceIn":sin.0,"duration":duration.0,"clips":placed["clips"],"timelineIn":cursor.0,"timelineOut":end.0,"dashcam":e.dashcam,"sourceMuted":e.mute || e.dashcam}));
            cursor = end;
        }
        let provenance = json!({"format":"autocut.catalogProvenance","version":1,"entries":entries});
        s.edit("Retain catalogue provenance", |p,_| {
            if let Some(i) = p.item_mut(seq) { i.metadata.insert("frameatlas.provenance".into(), provenance.to_string()); }
            Ok(())
        })?;
        Ok(json!({"sequence":seq.0,"sources":items.len(),"entries":entries,"duration":cursor.0,"editable":true}))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn source() -> Value {
        let dir = crate::media_test_util::tmp_dir("catalog-portrait");
        let path = dir.join("portrait.png");
        image::save_buffer(&path, &[100_u8; 24 * 40 * 3], 24, 40, image::ColorType::Rgb8).unwrap();
        json!({"asset_id":"portrait-source","path":path,"provider":"google_photos","bytes":std::fs::metadata(&path).unwrap().len()})
    }
    #[test]
    fn portrait_photo_reuse_provenance_and_single_undo() {
        let mut s = Session::default();
        let source = source();
        let before = s.project.clone();
        let r=s.execute(CMD,json!({"plan":{"format":"autocut.catalogEditPlan","version":1,"name":"Trip","entries":[{"source":source},{"source":source,"duration":254016000000i64}]}})).unwrap();
        assert_eq!(r["sources"], 1);
        let q = s.active_sequence().unwrap();
        assert_eq!(q.video_tracks[0].items.len(), 2);
        let seq = s.project.item(ItemId(r["sequence"].as_u64().unwrap())).unwrap();
        assert!(seq.metadata["frameatlas.provenance"].contains("portrait-source"));
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(&*s.project, &*before);
    }
    #[test]
    fn bad_ranges_and_unverified_references_leave_project_and_history() {
        let mut s = Session::default();
        let source = source();
        let before = s.project.clone();
        for entry in [json!({"source":source,"duration":0}), json!({"source":{ "asset_id":"x","path":"/missing","clip":{"source_start":0,"source_end":1}}})] {
            assert!(s.execute(CMD, json!({"plan":{"format":"autocut.catalogEditPlan","version":1,"name":"Trip","entries":[entry]}})).is_err());
            assert_eq!(&*s.project, &*before);
        }
    }
    #[test]
    fn verified_multipart_uses_named_part_and_360_points_need_an_explicit_range() {
        let dir = crate::media_test_util::tmp_dir("catalog-parts");
        let first = dir.join("part-one.mov");
        let second = dir.join("part-two.mov");
        crate::media_test_util::make_movie(&second, filmcraft_media::DemoScene::Plasma, 64, 32, 24);
        std::fs::copy(&second, &first).unwrap();
        let source = json!({"asset_id":"recording","path":first,"provider":"insta360","related_files":[first,second],
            "indexed_source_verification":"matched","clip":{"kind":"scene","scene_key":"current-scene","source_filename":"part-two.mov","source_path":second,"source_start":0.25,"source_end":0.75}});
        let mut s = Session::default();
        let plan = |entry| json!({"plan":{"format":"autocut.catalogEditPlan","version":1,"name":"Parts","sequence":{"fps":24},"entries":[entry]}});
        let result = s.execute(CMD, plan(json!({"source":source,"dashcam":true}))).unwrap();
        let q = s.active_sequence().unwrap();
        assert!(q.audio_tracks.iter().all(|t| t.items.is_empty()));
        let c = &q.video_tracks[0].items[0];
        assert_eq!(c.source_in, Tick::from_seconds_f64(0.25));
        let item = s.project.item(c.item).unwrap();
        assert!(matches!(&item.as_media().unwrap().media, MediaRef::File{path} if path == second.to_str().unwrap()));
        assert!(item.metadata["frameatlas.dependencies"].contains("part-one.mov"));
        assert_eq!(result["entries"][0]["source"]["clip"]["source_filename"], "part-two.mov");
        let mut point = source.clone();
        point["clip"]["kind"] = json!("360_sample");
        point["clip"]["source_end"] = json!(0.25);
        let before = s.project.clone();
        assert!(s.execute(CMD, plan(json!({"source":point,"duration":Tick::from_seconds_f64(0.5).0}))).is_err());
        assert_eq!(&*s.project, &*before);
        s.execute(CMD, plan(json!({"source":point,"sourceIn":0,"duration":Tick::from_seconds_f64(0.5).0}))).unwrap();
    }
}
