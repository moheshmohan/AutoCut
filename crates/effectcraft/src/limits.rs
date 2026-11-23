//! The caps every template and render is held to, checked before anything is allocated.

/// The largest `.ectemplate` read (its ZIP reader caps the entries again).
pub const MAX_TEMPLATE_BYTES: u64 = 256 << 20;
/// The largest `manifest.json`.
pub const MAX_MANIFEST_BYTES: usize = 1 << 20;
/// The largest `project.ecproj`.
pub const MAX_PROJECT_BYTES: usize = 64 << 20;
/// The largest output side, pixels.
pub const MAX_SIDE: u32 = 8192;
/// The most output pixels in one frame (4096²: a 4K UHD frame fits, with room).
pub const MAX_PIXELS: u64 = 4096 * 4096;
/// The largest item (comp, solid, footage) a template may hold, pixels, at the render's scale:
/// each can become a full float buffer.
pub const MAX_ITEM_PIXELS: u64 = 8192 * 8192;
/// The most frames one render makes (10 minutes at 60 fps).
pub const MAX_FRAMES: u64 = 36_000;
/// The largest footage frame handed to EffectCraft, pixels (an 8K UHD frame fits).
pub const MAX_FOOTAGE_PIXELS: u64 = 40_000_000;
/// The most values one render takes.
pub const MAX_VALUES: usize = 256;
/// The longest control name or value key, bytes.
pub const MAX_KEY_BYTES: usize = 256;
/// The deepest a value's JSON may nest.
pub const MAX_VALUE_DEPTH: usize = 16;
/// The most JSON nodes in all the values together.
pub const MAX_VALUE_NODES: usize = 100_000;
/// The longest string in a value, characters.
pub const MAX_STRING_CHARS: usize = 100_000;
/// The largest text size a value may set, points.
pub const MAX_TEXT_SIZE: f64 = 4096.0;
/// The most controls, footage items and notes listed.
pub const MAX_LISTED: usize = 1000;
