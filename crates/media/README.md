# filmcraft-media

Media sources, still-image and image-sequence import, generated media, audio streams, and decoded
frame caching for FilmCraft.

## Photos

- A JPEG, PNG, WebP or TIFF is turned upright by its EXIF `Orientation` as it is decoded (a
  portrait phone photo is stored sideways with a tag saying so); its media size is the upright one.
  `still::probe_geometry` / `probe_geometry_reader` report that size and the orientation from the
  headers alone, without decoding (a large TIFF whose directory is at its end is read only where
  its headers are). The engine uses them when a project opens: one saved before photos were
  turned upright (before 2026-10-21 here) stored a portrait photo's unturned size and baked its
  clips' Motion anchors from it, and both are turned to match (`engine::still_migration`).
- Decoding is bounded: the `image` decoders may allocate at most `MAX_DECODE_BYTES` (512 MiB) for a
  picture and accept sides up to 65 535, so a header claiming more is refused before anything is
  allocated; a picture larger than `MAX_STILL_SIDE` (8192 on its long side — a 48-megapixel phone
  photo, 8064 × 6048, stays whole) is scaled down to it, since a still is held whole in memory.
- `exif` reads, clean-room from the TIFF 6.0 structure EXIF is (CIPA DC-008), the orientation,
  `DateTimeOriginal` with its `OffsetTimeOriginal`, and the GPS position; every offset is checked
  and each directory read once. HEIF / HEIC photos are decoded by `filmcraft-codecs` (`heif`);
  their extensions are importable here but never image sequences.

## Image-sequence filenames

`sequence::Numbered` identifies the last run of 1–18 ASCII frame-number digits in a filename's
stem when the extension is a recognized still-image format. `/` and `\` separate directory
components. A prefix may contain other UTF-8 characters, including accented letters, CJK and emoji:
`é0001.png`, `图0001.png` and `🎞️0001.png`. Parsing preserves the prefix, number padding and suffix
when generating sibling-frame paths. Sibling recognition compares the prefix's UTF-8 bytes exactly.
Stems without ASCII digits or with a final frame-number run longer than 18 characters are not numbered
image-sequence filenames.
