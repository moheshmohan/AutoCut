//! Platform safe zones of a vertical (9:16) frame: the parts YouTube Shorts, Instagram Reels and
//! TikTok cover with their own interface. The numbers and their sources live in
//! [`filmcraft_project::safe_zones`] (caption burn-in keeps animated captions inside them too);
//! they are re-exported here for the renderer's users.

pub use filmcraft_project::safe_zones::*;
