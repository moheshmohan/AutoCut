//! Plan validation: every step must be a real, allowed command before anything executes.

use serde_json::Value;

use crate::PlannedStep;
use crate::digest::allowed;

/// Check one step against the real command registry and AutoCut's family allowlist.
pub fn validate_step(step: &PlannedStep) -> Result<(), String> {
    if filmcraft_engine::commands::find(&step.id).is_none() {
        return Err(format!("unknown command `{}`", step.id));
    }
    if !allowed(&step.id) {
        return Err(format!("`{}` is outside AutoCut's command families", step.id));
    }
    if !step.params.is_object() && !step.params.is_null() {
        return Err(format!("`{}` params must be an object, got {}", step.id, kind(&step.params)));
    }
    Ok(())
}

/// Validate every step; the first problem names its step.
pub fn validate(steps: &[PlannedStep]) -> Result<(), String> {
    for (n, s) in steps.iter().enumerate() {
        if let Err(e) = validate_step(s) {
            return Err(format!("step {} of {}: {e}", n + 1, steps.len()));
        }
    }
    Ok(())
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a bool",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn real_allowed_commands_pass() {
        assert!(validate_step(&PlannedStep::new("transcript.removePauses", json!({}), "")).is_ok());
        assert!(validate_step(&PlannedStep::new("sequence.extract", Value::Null, "")).is_ok());
    }

    #[test]
    fn rejects_unknown_and_disallowed() {
        let e = validate_step(&PlannedStep::new("no.such.command", json!({}), "")).unwrap_err();
        assert!(e.contains("unknown command"), "{e}");
        let e = validate_step(&PlannedStep::new("prefs.set", json!({}), "")).unwrap_err();
        assert!(e.contains("outside AutoCut's"), "{e}");
        let e = validate_step(&PlannedStep::new("timeline.razor", json!([1, 2]), "")).unwrap_err();
        assert!(e.contains("must be an object"), "{e}");
        let e = validate(&[PlannedStep::new("timeline.razor", json!({}), ""), PlannedStep::new("nope.x", json!({}), "")]).unwrap_err();
        assert!(e.starts_with("step 2 of 2"), "{e}");
    }
}
