# Elements

Previews of the bundled Elements library's stickers (`stickers/*.svg`). The app does not read
these files: it inserts the vector layers defined in `crates/project/src/elements.rs`
(`elements.insert {"id": "sticker.arrow"}`), and the SVGs are exported from the same layers by
`crates/engine/src/elements.rs`, so they cannot drift (`FILMCRAFT_BLESS=1 cargo test -p
filmcraft-engine bundled_sticker_svgs` regenerates them). The library's sounds are synthesized
when inserted and its animated text presets are templates in code, so neither has files here.

Every file has a `<file>.attribution` sidecar and a row in [`../../ATTRIBUTION.md`](../../ATTRIBUTION.md).
