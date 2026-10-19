//! The planner behind recipe steps that are AutoCut intents (`recipe.run` in the engine): an
//! intent and its parameters become engine commands with the same deterministic handlers a prompt
//! uses ([`crate::planner`]), read from and checked against the session the recipe runs on —
//! through the job's command port, so every read is an ordinary command — and validated by
//! [`crate::validate`] before the engine runs any of them (which checks them once more against
//! the recipe allowlist).

use std::sync::Arc;

use filmcraft_engine::autocut::CommandPort;
use filmcraft_engine::recipes::{Planned, PlannedCommand, RecipePlanner};
use serde_json::Value;

use crate::digest::Digest;
use crate::planner::{self, Plan};
use crate::{ExecFuture, Executor, PlannedStep};

/// Intents a recipe step may name: the routine catalog (not `custom`, which needs a prompt).
pub fn intents() -> Vec<String> {
    crate::intent::INTENTS.iter().filter(|i| **i != "custom").map(|i| i.to_string()).collect()
}

/// [`Executor`] over a recipe job's command port (blocking calls, made when the future is
/// created: the planner runs on the job's own thread).
struct PortExecutor(CommandPort);

impl Executor for PortExecutor {
    fn execute(&self, id: &str, params: Value) -> ExecFuture {
        let r = self.0.call(id, params);
        Box::pin(async move { r })
    }
}

/// The AutoCut planner for recipes.
pub struct AutoCutRecipePlanner;

fn commands_of(steps: Vec<PlannedStep>) -> Vec<PlannedCommand> {
    steps.into_iter().map(|s| PlannedCommand { id: s.id, params: s.params, why: s.why }).collect()
}

impl RecipePlanner for AutoCutRecipePlanner {
    fn intents(&self) -> Vec<String> {
        intents()
    }

    fn plan(&self, intent: &str, params: &Value, port: &CommandPort) -> Result<Planned, String> {
        if !intents().iter().any(|i| i == intent) {
            return Err(format!("`{intent}` is not an AutoCut intent"));
        }
        let params = if params.is_object() { params.clone() } else { Value::Object(Default::default()) };
        let rt = tokio::runtime::Builder::new_current_thread().enable_time().build().map_err(|e| format!("planner: {e}"))?;
        let ex = PortExecutor(port.clone());
        rt.block_on(async {
            let digest = if intent == "showcase" {
                Digest { project: Value::Null, sequence: Value::Null, transcript: None }
            } else {
                Digest::read(&ex).await.map_err(|e| e.to_string())?
            };
            match planner::plan(intent, &params, &digest, &ex).await.map_err(|e| e.to_string())? {
                Plan::Steps { steps, notes } => {
                    crate::validate::validate(&steps)?;
                    Ok(Planned { commands: commands_of(steps), notes, replan: false })
                }
                Plan::GenerateThen { generate, .. } => {
                    crate::validate::validate_step(&generate)?;
                    Ok(Planned { commands: commands_of(vec![generate]), notes: vec!["a transcript is made first".into()], replan: true })
                }
            }
        })
    }
}

/// The planner the app, the CLI and the MCP server install (`session.recipes.planner`).
pub fn planner() -> Arc<dyn RecipePlanner> {
    Arc::new(AutoCutRecipePlanner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_engine::Session;
    use serde_json::json;

    /// A demo sequence with a transcript set, so the transcript intents plan without a model.
    fn session() -> Session {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        s.recipes.planner = Some(planner());
        s
    }

    #[test]
    fn intents_exclude_custom() {
        let i = intents();
        assert!(i.contains(&"remove_silence".to_string()) && i.contains(&"add_captions".to_string()));
        assert!(!i.contains(&"custom".to_string()));
    }

    #[test]
    fn a_recipe_of_intents_is_planned_and_checked_through_the_port() {
        let mut s = session();
        // a dry run plans nothing yet (intents are planned when their step runs), but checks them
        let plan = s.execute("recipe.run", json!({"name": "builtin:talking-head-clean-up", "dryRun": true})).unwrap();
        assert_eq!(plan["plan"].as_array().unwrap().len(), 4);
        // an unknown intent is refused before anything runs
        let bad = json!({"format": "filmcraft.recipe", "formatVersion": 1, "id": "user:x", "name": "X", "steps": [{"intent": "teleport"}]});
        let e = s.execute("recipe.run", json!({"recipe": bad})).unwrap_err().to_string();
        assert!(e.contains("not an AutoCut intent"), "{e}");
        // the colour of a voice: enhance_voice plans Essential Sound commands on the demo's audio
        let r = json!({"format": "filmcraft.recipe", "formatVersion": 1, "id": "user:voice", "name": "Voice",
            "steps": [{"label": "Clean the voice", "intent": "enhance_voice", "params": {"preset": "podcast", "loudness": false}}]});
        let before = s.history.undo.len();
        let out = s.execute("recipe.run", json!({"recipe": r, "wait": true})).unwrap();
        let run = &out["recipeRun"];
        assert_eq!(run["ok"], true, "{out}");
        let cmds: Vec<&str> = run["steps"][0]["commands"].as_array().unwrap().iter().map(|c| c["command"].as_str().unwrap()).collect();
        assert!(cmds.contains(&"essentialSound.applyPreset"), "{cmds:?}");
        assert!(s.history.undo.len() > before, "the planned commands edited the sequence");
    }
}
