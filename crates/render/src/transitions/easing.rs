//! Easing of a transition's progress: the curves live in [`filmcraft_project::easing`] (Penner's
//! families and a CSS-style Bézier, shared with the text animators); transitions use the clamped
//! [`ease`], evaluated on the CPU at the exact frame `Tick` so the shader (GPU) and its reference
//! (CPU) receive the same eased value.

pub use filmcraft_project::easing::{Bezier, COUNT, ease};
