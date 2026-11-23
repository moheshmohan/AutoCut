//! AutoCut's optional craft-fonts ([`filmcraft_text::fonts::CRAFT_FONTS`]) in EffectCraft's font
//! database, once per process, without copying (the faces point at the embedded bytes). Empty
//! when AutoCut was built without `CRAFT_FONTS_DIR`; EffectCraft's own bundled fonts are there
//! either way.

use std::sync::OnceLock;

use effectcraft_text::fonts::{FaceData, FaceInfo, FontSource};
use effectcraft_text::sfnt::read_faces_bytes;

/// The craft-fonts faces.
struct CraftFonts;

impl FontSource for CraftFonts {
    fn faces(&self) -> Vec<FaceInfo> {
        filmcraft_text::fonts::CRAFT_FONTS
            .iter()
            .flat_map(|f| {
                read_faces_bytes(f.bytes).into_iter().map(|n| FaceInfo {
                    family: n.family,
                    style: n.style,
                    weight: n.weight,
                    italic: n.italic,
                    index: n.index,
                    data: FaceData::Static(f.bytes),
                    origin: "craft-fonts",
                    native_family: n.native_family,
                    full_name: n.full_name,
                    postscript: n.postscript,
                    variable: n.variable,
                    instances: n.instances,
                })
            })
            .collect()
    }
}

/// Add the craft-fonts (once): how many faces were added.
pub(crate) fn ensure() -> usize {
    static ADDED: OnceLock<usize> = OnceLock::new();
    *ADDED.get_or_init(|| {
        if filmcraft_text::fonts::CRAFT_FONTS.is_empty() {
            return 0;
        }
        std::panic::catch_unwind(|| effectcraft_text::fonts::add_source(&CraftFonts)).unwrap_or_else(|_| {
            log::error!("effectcraft: adding the craft-fonts failed inside (caught); EffectCraft's own fonts remain");
            0
        })
    })
}
