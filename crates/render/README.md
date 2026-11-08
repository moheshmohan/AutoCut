# filmcraft-render

The CPU reference compositor and GPU effect planning for FilmCraft/AutoCut.
See [architecture](../../docs/architecture.md) for colour, source coordinates,
intrinsic effects and the export pipeline.

## Upstream ports, AV17

- Lumetri's Luma vs Saturation and Saturation vs Saturation curves have straight
  axes: highlights no longer wrap into shadows. Hue curves remain periodic and
  meet at red. The UI uses the same curve helpers as rendering. Invalid sizes or
  non-finite points are refused by the new helpers.
- Shadow lift limits near-black chroma amplification while preserving the
  existing hue-preserving scale above the 0.02 encoded-luma floor. CPU/GPU
  calculations follow the same rule.
- Warp Stabilizer analysis prepares and tracks frames in blocks of 64,
  retaining the boundary frame between blocks. A bounded cache reuses smoothed
  corrections and cover scale for unchanged analysis/settings.
- Luma Key can key out bright pixels using thresholds measured from white.
  Long Shadow and Glint extents are bounded by the picture; preview-cost walks
  bound nested subclips.

AutoCut's 360/gyro pipeline, mask ordering, source-size point resolution and
existing effect parameter IDs are retained. Tests cover colour seams, untouched
shadow saturation, block boundaries, stabilizer cancellation/cache equivalence,
bright-key thresholds and hostile curve parameters.
