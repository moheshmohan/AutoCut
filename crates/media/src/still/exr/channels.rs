//! Which channels of an OpenEXR file make the picture shown.
//!
//! A channel name is `layer.channel`, where the layer is everything before the last dot and may
//! itself be nested (`ViewLayer.Combined.R`); a channel without a dot is in the unnamed layer.
//!
//! The picture is the file's main picture when it has one ([`main_picture`], the convention
//! compositing applications follow): the unprefixed `R`, `G`, `B` (and `A`) of the first part that
//! has all three, else the first part's unprefixed `Y` (and `A`). A file of named layers only (a
//! renderer's multilayer file, `ViewLayer.Combined.R`) has no main picture; then the richest
//! colour layer of any part is shown ([`color_channels`], [`best_layer`]) rather than nothing.

use std::collections::BTreeMap;

/// What the colour channels of a picture are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorLayout {
    /// Luminance: `Y`, or the only channel of a one-channel part.
    Y,
    /// `Y`, `A`.
    YA,
    /// `R`, `G`, `B`.
    Rgb,
    /// `R`, `G`, `B`, `A`.
    Rgba,
}

impl ColorLayout {
    /// How many channels the layout has, which is also its rank when choosing between groups.
    pub fn channels(self) -> usize {
        match self {
            ColorLayout::Y => 1,
            ColorLayout::YA => 2,
            ColorLayout::Rgb => 3,
            ColorLayout::Rgba => 4,
        }
    }

    pub fn has_alpha(self) -> bool {
        matches!(self, ColorLayout::YA | ColorLayout::Rgba)
    }
}

/// The picture of one part: its layout, the indices of its channels in layout order (R, G, B, A
/// or Y, A), and the layer they are in (`""` for the unnamed layer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Picture {
    pub layout: ColorLayout,
    pub channels: Vec<usize>,
    pub layer: String,
}

/// The main picture of a file by the OpenEXR convention: the channels without a layer prefix,
/// `R`, `G`, `B` (and `A` when present), of the first part that has all three; else the first
/// part's unprefixed `Y` (and `A`). Named layers never stand in for it. `parts` gives each part's
/// channel names as stored, in file order; returns the part's index and its picture. Names match
/// exactly, as the specification spells them; the first of duplicate names counts.
pub fn main_picture<'a, P, C>(parts: P) -> Option<(usize, Picture)>
where
    P: IntoIterator<Item = C>,
    C: IntoIterator<Item = &'a str>,
{
    let mut luminance: Option<(usize, Picture)> = None;
    for (part, names) in parts.into_iter().enumerate() {
        let (mut r, mut g, mut b, mut a, mut y) = (None, None, None, None, None);
        for (index, name) in names.into_iter().enumerate() {
            let slot = match name {
                "R" => &mut r,
                "G" => &mut g,
                "B" => &mut b,
                "A" => &mut a,
                "Y" => &mut y,
                _ => continue,
            };
            slot.get_or_insert(index);
        }
        let picture = |layout, channels| Picture { layout, channels, layer: String::new() };
        if let (Some(r), Some(g), Some(b)) = (r, g, b) {
            return Some(match a {
                Some(a) => (part, picture(ColorLayout::Rgba, vec![r, g, b, a])),
                None => (part, picture(ColorLayout::Rgb, vec![r, g, b])),
            });
        }
        if luminance.is_none()
            && let Some(y) = y
        {
            luminance = Some(match a {
                Some(a) => (part, picture(ColorLayout::YA, vec![y, a])),
                None => (part, picture(ColorLayout::Y, vec![y])),
            });
        }
    }
    luminance
}

/// The channels of one part's `names` that make a colour picture. RGB(A) or YA is never put
/// together from separate layers: the colour and alpha channels share a whole layer prefix (a
/// leading or trailing dot alone names no layer). The group with the most channels wins; equal
/// groups prefer the unnamed layer, then the first prefix alphabetically. A part of exactly one
/// channel, whatever its name, is luminance. `None` when there is no such group (`Z` beside `A`).
pub fn color_channels<'a>(names: impl IntoIterator<Item = &'a str>) -> Option<Picture> {
    #[derive(Default)]
    struct Group {
        r: Option<usize>,
        g: Option<usize>,
        b: Option<usize>,
        a: Option<usize>,
        y: Option<usize>,
    }

    // sorted prefixes make ties deterministic whatever the channel order, the unnamed group first
    let mut groups = BTreeMap::<&str, Group>::new();
    let mut count = 0usize;
    for (index, name) in names.into_iter().enumerate() {
        count = count.saturating_add(1);
        let (prefix, base) = name.rsplit_once('.').filter(|(prefix, base)| !prefix.is_empty() && !base.is_empty()).unwrap_or(("", name));
        if !matches!(base, "R" | "G" | "B" | "A" | "Y") {
            continue;
        }
        let group = groups.entry(prefix).or_default();
        let component = match base {
            "R" => &mut group.r,
            "G" => &mut group.g,
            "B" => &mut group.b,
            "A" => &mut group.a,
            "Y" => &mut group.y,
            _ => continue,
        };
        component.get_or_insert(index);
    }

    let mut best: Option<Picture> = None;
    for (layer, group) in &groups {
        let (layout, channels) = if let (Some(r), Some(g), Some(b)) = (group.r, group.g, group.b) {
            match group.a {
                Some(a) => (ColorLayout::Rgba, vec![r, g, b, a]),
                None => (ColorLayout::Rgb, vec![r, g, b]),
            }
        } else if let Some(y) = group.y {
            match group.a {
                Some(a) => (ColorLayout::YA, vec![y, a]),
                None => (ColorLayout::Y, vec![y]),
            }
        } else {
            continue;
        };
        if best.as_ref().is_none_or(|b| layout.channels() > b.layout.channels()) {
            best = Some(Picture { layout, channels, layer: (*layer).to_string() });
        }
    }
    best.or_else(|| (count == 1).then(|| Picture { layout: ColorLayout::Y, channels: vec![0], layer: String::new() }))
}

/// The part and picture to show when the file has no [`main_picture`]: the part whose
/// [`color_channels`] group has the most channels, the earliest part on a tie.
pub fn best_layer<'a, P, C>(parts: P) -> Option<(usize, Picture)>
where
    P: IntoIterator<Item = C>,
    C: IntoIterator<Item = &'a str>,
{
    let mut best: Option<(usize, Picture)> = None;
    for (part, names) in parts.into_iter().enumerate() {
        if let Some(p) = color_channels(names)
            && best.as_ref().is_none_or(|(_, b)| p.layout.channels() > b.layout.channels())
        {
            best = Some((part, p));
        }
    }
    best
}

/// The picture a file shows: its [`main_picture`], else its [`best_layer`].
pub fn shown_picture<'a, P, C>(parts: P) -> Option<(usize, Picture)>
where
    P: IntoIterator<Item = C> + Clone,
    C: IntoIterator<Item = &'a str>,
{
    main_picture(parts.clone()).or_else(|| best_layer(parts))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout_and_channels(p: Option<Picture>) -> Option<(ColorLayout, Vec<usize>)> {
        p.map(|p| (p.layout, p.channels))
    }

    #[test]
    fn colour_channels_come_from_one_layer() {
        let pick = |names: &[&str]| layout_and_channels(color_channels(names.iter().copied()));
        assert_eq!(pick(&["A", "B", "G", "R"]), Some((ColorLayout::Rgba, vec![3, 2, 1, 0])));
        assert_eq!(pick(&["B", "G", "R", "Z"]), Some((ColorLayout::Rgb, vec![2, 1, 0])));
        assert_eq!(pick(&["A", "Y"]), Some((ColorLayout::YA, vec![1, 0])));
        assert_eq!(pick(&["Y", "Z"]), Some((ColorLayout::Y, vec![0])));
        // never RGB from one layer and alpha from another
        assert_eq!(pick(&["A", "diffuse.B", "diffuse.G", "diffuse.R"]), Some((ColorLayout::Rgb, vec![3, 2, 1])));
        assert_eq!(pick(&["a.R", "b.G", "c.B"]), None);
        assert_eq!(pick(&["A", "Z"]), None);
        // the group with the most channels wins; ties go to the unnamed layer, then alphabetically
        assert_eq!(pick(&["R", "G", "B", "beauty.R", "beauty.G", "beauty.B", "beauty.A"]), Some((ColorLayout::Rgba, vec![3, 4, 5, 6])));
        assert_eq!(pick(&["z.R", "z.G", "z.B", "R", "G", "B"]), Some((ColorLayout::Rgb, vec![3, 4, 5])));
        assert_eq!(pick(&["z.R", "z.G", "z.B", "m.R", "m.G", "m.B"]), Some((ColorLayout::Rgb, vec![3, 4, 5])));
        // nested layers use the whole prefix; a leading or trailing dot names no layer
        assert_eq!(pick(&["a.b.R", "a.b.G", "a.b.B", "a.R"]), Some((ColorLayout::Rgb, vec![0, 1, 2])));
        assert_eq!(pick(&[".R", ".G", ".B"]), None);
        // the first of duplicate names is used, and a lone channel of any name is luminance
        assert_eq!(pick(&["Y", "Y"]), Some((ColorLayout::Y, vec![0])));
        assert_eq!(pick(&["depth.Z"]), Some((ColorLayout::Y, vec![0])));
        assert_eq!(pick(&[]), None);
        assert_eq!(pick(&["r", "g", "b"]), None);
        assert_eq!(
            color_channels(["ViewLayer.Combined.R", "ViewLayer.Combined.G", "ViewLayer.Combined.B"]).map(|p| p.layer).as_deref(),
            Some("ViewLayer.Combined")
        );
    }

    #[test]
    fn the_main_picture_is_the_unprefixed_colour_of_the_first_part_that_has_it() {
        let main = |parts: &[&[&str]]| main_picture(parts.iter().map(|p| p.iter().copied())).map(|(i, p)| (i, p.layout, p.channels));
        // one part: the bare channels, whatever else is there, also before them alphabetically
        assert_eq!(
            main(&[&["A", "B", "CryptoMaterial00.A", "CryptoMaterial00.B", "CryptoMaterial00.G", "CryptoMaterial00.R", "G", "R", "Z"]]),
            Some((0, ColorLayout::Rgba, vec![7, 6, 1, 0]))
        );
        assert_eq!(main(&[&["B", "G", "R", "diffuse.A", "diffuse.B", "diffuse.G", "diffuse.R"]]), Some((0, ColorLayout::Rgb, vec![2, 1, 0])));
        // named layers only: no main picture, however good a layer looks
        assert_eq!(main(&[&["ViewLayer.Combined.A", "ViewLayer.Combined.B", "ViewLayer.Combined.G", "ViewLayer.Combined.R"]]), None);
        // multi-part: the first part with bare R, G and B, not the first part
        assert_eq!(main(&[&["Z"], &["A", "B", "G", "R"], &["B", "G", "R"]]), Some((1, ColorLayout::Rgba, vec![3, 2, 1, 0])));
        assert_eq!(main(&[&["B", "G", "R"], &["A", "B", "G", "R"]]), Some((0, ColorLayout::Rgb, vec![2, 1, 0])));
        // colour wins over an earlier luminance part; luminance when there is no colour
        assert_eq!(main(&[&["Y"], &["B", "G", "R"]]), Some((1, ColorLayout::Rgb, vec![2, 1, 0])));
        assert_eq!(main(&[&["Z"], &["A", "Y"], &["Y"]]), Some((1, ColorLayout::YA, vec![1, 0])));
        // incomplete colour, other spellings and lone data channels are no main picture
        assert_eq!(main(&[&["G", "R"]]), None);
        assert_eq!(main(&[&["b", "g", "r"]]), None);
        assert_eq!(main(&[&["Z"]]), None);
        assert_eq!(main(&[]), None);
        assert_eq!(main(&[&["R", "G", "B", "R"]]), Some((0, ColorLayout::Rgb, vec![0, 1, 2])));
    }

    #[test]
    fn without_a_main_picture_the_richest_layer_is_shown() {
        let shown = |parts: &[&[&str]]| shown_picture(parts.iter().map(|p| p.iter().copied())).map(|(i, p)| (i, p.layout, p.channels, p.layer));
        // a renderer's multilayer file: the RGBA layer over the RGB ones
        assert_eq!(
            shown(&[&[
                "ViewLayer.AO.B",
                "ViewLayer.AO.G",
                "ViewLayer.AO.R",
                "ViewLayer.Combined.A",
                "ViewLayer.Combined.B",
                "ViewLayer.Combined.G",
                "ViewLayer.Combined.R"
            ]]),
            Some((0, ColorLayout::Rgba, vec![6, 5, 4, 3], "ViewLayer.Combined".to_string()))
        );
        // one part a pass: the earliest richest part
        assert_eq!(
            shown(&[&["ViewLayer.Depth.Z"], &["ViewLayer.Combined.B", "ViewLayer.Combined.G", "ViewLayer.Combined.R"], &["x.B", "x.G", "x.R"]]),
            Some((1, ColorLayout::Rgb, vec![2, 1, 0], "ViewLayer.Combined".to_string()))
        );
        // a lone depth channel is grey; the main picture wins over a richer layer
        assert_eq!(shown(&[&["Z"]]), Some((0, ColorLayout::Y, vec![0], String::new())));
        assert_eq!(shown(&[&["B", "G", "R", "beauty.A", "beauty.B", "beauty.G", "beauty.R"]]).map(|s| s.1), Some(ColorLayout::Rgb));
        assert_eq!(shown(&[&["A", "Z"]]), None);
    }
}
