//! The cloud supervisor: Claude plans or reviews when the local planner can't, against the real
//! command registry, with one repair round when its answer doesn't validate.

use serde_json::{Value, json};

use crate::llm::ChatModel;
use crate::{AutoCutError, PlannedStep};

/// Steps the supervisor returned.
#[derive(Debug, Clone)]
pub struct Supervision {
    pub steps: Vec<PlannedStep>,
    pub notes: Vec<String>,
}

const SYSTEM: &str = "\
You are the supervisor of AutoCut, an automated video editor. You turn a user's request into a \
batch of engine commands for the FilmCraft editor, or review such a batch.

Answer with ONE JSON object, nothing else:
{\"steps\": [{\"id\": string, \"params\": object, \"why\": string}], \"notes\": [string]}

Rules:
- Every `id` must come from the command catalog you are given; params must match its documented shape.
- Times: pass human-friendly `seconds` numbers where the catalog allows it (it usually does).
- Prefer transcript-driven commands when the request is about speech: transcript.removePauses, \
transcript.removeFillers, transcript.createCaptions, transcript.extract, transcript.search.
- Each step must be independently correct; they run in order, each an undoable edit.
- Fewer, higher-level steps beat many micro-steps. No playback, UI or prefs commands.
- If the request cannot be done with the catalog, say so in `notes` with empty `steps`.";

/// Ask the supervisor to plan a prompt the local planner couldn't.
pub async fn plan(model: &dyn ChatModel, prompt: &str, doc: &Value, registry: &str) -> Result<Supervision, AutoCutError> {
    let user = format!("COMMAND CATALOG (the only commands you may use):\n{registry}\n\nCURRENT DOCUMENT:\n{doc}\n\nUSER REQUEST:\n{prompt}");
    ask(model, &user).await
}

/// Ask the supervisor to review (and if needed fix) a locally planned batch.
pub async fn review(model: &dyn ChatModel, prompt: &str, doc: &Value, steps: &[PlannedStep]) -> Result<Supervision, AutoCutError> {
    let plan: Vec<Value> = steps.iter().map(|s| json!({"id": s.id, "params": s.params, "why": s.why})).collect();
    let user = format!(
        "CURRENT DOCUMENT:\n{doc}\n\nUSER REQUEST:\n{prompt}\n\nPROPOSED STEPS:\n{}\n\nReview these steps. If they accomplish the request, return them unchanged. If not, return corrected steps.",
        Value::Array(plan)
    );
    ask(model, &user).await
}

/// One exchange plus, when validation fails, one repair round naming the problems.
async fn ask(model: &dyn ChatModel, user: &str) -> Result<Supervision, AutoCutError> {
    let first = model.complete_json(SYSTEM, user, &[]).await.map_err(|e| AutoCutError::Llm(format!("supervisor ({}): {e}", model.id())))?;
    let mut parsed = parse(first);
    let mut errs = problems(&parsed);
    if !errs.is_empty() {
        let repair = format!("{user}\n\nYour previous answer was invalid:\n{}\n\nAnswer again with a corrected JSON object.", errs.join("\n"));
        let second = model.complete_json(SYSTEM, &repair, &[]).await.map_err(|e| AutoCutError::Llm(format!("supervisor repair ({}): {e}", model.id())))?;
        parsed = parse(second);
        errs = problems(&parsed);
    }
    if !errs.is_empty() {
        return Err(AutoCutError::Invalid(format!("the supervisor's plan does not validate: {}", errs.join("; "))));
    }
    let notes =
        parsed.as_ref().and_then(|v| v["notes"].as_array()).map(|a| a.iter().filter_map(|n| n.as_str().map(str::to_string)).collect()).unwrap_or_default();
    let steps = parsed.map(|v| steps_of(&v)).unwrap_or_default();
    Ok(Supervision { steps, notes })
}

/// Parse a model answer into `{steps, notes}`, tolerating a bare steps array.
fn parse(v: Value) -> Option<Value> {
    if v.get("steps").is_some() {
        return Some(v);
    }
    v.as_array().map(|a| json!({"steps": a, "notes": []}))
}

/// Structural problems of an answer (validation happens on the parsed steps).
fn problems(v: &Option<Value>) -> Vec<String> {
    let Some(v) = v else { return vec!["the answer is not JSON with a `steps` array".into()] };
    if !v["steps"].as_array().is_some_and(|a| !a.is_empty()) {
        return Vec::new(); // empty steps with notes is a valid "can't do this"
    }
    let mut out = Vec::new();
    for (n, s) in v["steps"].as_array().into_iter().flatten().enumerate() {
        let id = s["id"].as_str().unwrap_or_default().to_string();
        if id.is_empty() {
            out.push(format!("step {}: missing `id`", n + 1));
        }
        if !s["params"].is_object() && !s["params"].is_null() {
            out.push(format!("step {}: `params` must be an object", n + 1));
        }
        if filmcraft_engine::commands::find(&id).is_none() {
            out.push(format!("step {}: `{id}` is not a real command", n + 1));
        } else if !crate::digest::allowed(&id) {
            out.push(format!("step {}: `{id}` is outside AutoCut's command families", n + 1));
        }
    }
    out
}

fn steps_of(v: &Value) -> Vec<PlannedStep> {
    v["steps"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|s| PlannedStep {
            id: s["id"].as_str().unwrap_or_default().to_string(),
            params: s["params"].clone(),
            why: s["why"].as_str().unwrap_or("supervisor step").to_string(),
        })
        .filter(|s| !s.id.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract_json;
    use crate::llm::FixedModel;

    fn good() -> Value {
        json!({"steps": [{"id": "transcript.removePauses", "params": {"minSeconds": 0.6}, "why": "tight pauses"}], "notes": ["ok"]})
    }

    #[tokio::test]
    async fn parses_and_accepts_a_valid_plan() {
        let m = FixedModel::new(vec![good()]);
        let s = plan(&m, "tighten it", &json!({}), "catalog").await.unwrap();
        assert_eq!(s.steps.len(), 1);
        assert_eq!(s.steps[0].id, "transcript.removePauses");
        assert_eq!(s.notes, ["ok"]);
    }

    #[tokio::test]
    async fn repairs_once_then_fails() {
        let bad = json!({"steps": [{"id": "not.a.command", "params": {}, "why": ""}]});
        let m = FixedModel::new(vec![bad, good()]);
        let s = plan(&m, "tighten it", &json!({}), "catalog").await.unwrap();
        assert_eq!(s.steps[0].id, "transcript.removePauses");
        assert_eq!(m.asked.lock().unwrap().len(), 2, "exactly one repair round");

        let worse = FixedModel::new(vec![json!({"steps": [{"id": "still.not", "params": {}}]}), json!({"steps": [{"id": "again.no", "params": {}}]})]);
        let e = plan(&worse, "x", &json!({}), "catalog").await.unwrap_err();
        assert!(e.to_string().contains("does not validate"), "{e}");
    }

    #[tokio::test]
    async fn empty_steps_with_notes_means_cant() {
        let m = FixedModel::new(vec![json!({"steps": [], "notes": ["no such effect in the catalog"]})]);
        let s = plan(&m, "remove the background", &json!({}), "catalog").await.unwrap();
        assert!(s.steps.is_empty());
        assert_eq!(s.notes.len(), 1);
    }

    #[tokio::test]
    async fn bare_array_is_tolerated() {
        let m = FixedModel::new(vec![json!([{"id": "sequence.extract", "params": {}}])]);
        let s = plan(&m, "x", &json!({}), "catalog").await.unwrap();
        assert_eq!(s.steps[0].id, "sequence.extract");
    }

    #[test]
    fn extract_json_still_applies() {
        // the ChatModel contract already returns parsed JSON; parse() only normalizes shape
        let v = extract_json("junk {\"steps\": []} junk").unwrap();
        assert!(parse(v).is_some());
    }
}
