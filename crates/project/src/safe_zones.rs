//! Platform safe zones of a vertical (9:16) frame: the parts each platform's own interface covers
//! when it plays a short — the title and search bar at the top; the channel, description, sound
//! and progress bar at the bottom; the like / comment / share buttons down the right — where
//! captions, titles and lower thirds should not go.
//!
//! The numbers come from the platforms' public guidance for vertical video, as shares of a
//! 1080×1920 frame, rounded so that each unsafe area is a little larger than published, never
//! smaller:
//!
//! | platform | top | bottom | left | right | buttons | published |
//! |---|---|---|---|---|---|---|
//! | YouTube Shorts | 15 % | 35 % | 5 % | 18 % | | 288, 672, 48, 192 px of 1080×1920 |
//! | Instagram Reels | 14 % | 35 % | 6 % | 6 % | | 14 %, 35 %, 6 % each side |
//! | TikTok | 13 % | 35 % | 12 % | 12 % | right 20 % from 40 % down | 160, 440, 80 px of 720×1280; a 120 px button column from 560 px down |
//! | Vertical | 15 % | 35 % | 12 % | 18 % | right 20 % from 40 % down | all three at once |
//!
//! Sources (read 2026-10-21):
//! - YouTube: Google's "Universal Video Ad Safe Zones" (the vertical, 1080×1920 diagram),
//!   <https://services.google.com/fh/files/misc/universalsafezones-youtube.pdf>, linked from
//!   "About video ad specs", <https://support.google.com/google-ads/answer/13547298>; vertical
//!   ads run in Shorts, and Google publishes no other Shorts numbers.
//! - Instagram: the Meta Ads Guide for Reels ("leave at least 14 % at the top, 35 % at the bottom
//!   and 6 % on each side free of text, logos and other key elements"),
//!   <https://www.facebook.com/business/ads-guide/update/video/instagram-reels>.
//! - TikTok: the In-Feed ads safe-zone files of TikTok's Ads help, "TikTok Auction In-Feed Ads",
//!   <https://ads.tiktok.com/help/article/tiktok-auction-in-feed-ads> (the page prints no numbers;
//!   the template's margins and button column as measured and published at
//!   <https://www.socialync.io/tools/safe-zones/tiktok>).
//!
//! The platforms change their interfaces and phones differ, so these are a production guide:
//! keep what matters inside [`Platform::safe_area`]. On a frame of another shape the shares
//! apply as they are (the interfaces are laid out for 9:16).

use serde::{Deserialize, Serialize};

/// A short-video platform whose interface covers part of a vertical frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Platform {
    /// YouTube Shorts.
    Youtube,
    /// Instagram Reels.
    Reels,
    TikTok,
    /// Safe on all three at once.
    Vertical,
}

/// A rectangle as shares (0..1) of the frame: left, top, right, bottom.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Area {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl Area {
    /// The rectangle between two corners (in either order); a corner that is not a number is 0.
    pub fn new(x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        let f = |v: f64| if v.is_finite() { v } else { 0.0 };
        let (x0, x1, y0, y1) = (f(x0), f(x1), f(y0), f(y1));
        Area { x0: x0.min(x1), y0: y0.min(y1), x1: x0.max(x1), y1: y0.max(y1) }
    }
    /// A rectangle of a `w`×`h` frame given in its pixels, as shares of it.
    pub fn from_pixels(r: [f64; 4], (w, h): (f64, f64)) -> Self {
        let (w, h) = (if w.is_finite() && w >= 1.0 { w } else { 1.0 }, if h.is_finite() && h >= 1.0 { h } else { 1.0 });
        Area::new(r[0] / w, r[1] / h, r[2] / w, r[3] / h)
    }
    /// In the pixels of a `w`×`h` frame: `[x0, y0, x1, y1]`.
    pub fn in_pixels(&self, (w, h): (f64, f64)) -> [f64; 4] {
        [self.x0 * w, self.y0 * h, self.x1 * w, self.y1 * h]
    }
    pub fn width(&self) -> f64 {
        (self.x1 - self.x0).max(0.0)
    }
    pub fn height(&self) -> f64 {
        (self.y1 - self.y0).max(0.0)
    }
    pub fn size(&self) -> f64 {
        self.width() * self.height()
    }
    /// Where the two overlap, if they do.
    pub fn intersect(&self, o: &Area) -> Option<Area> {
        let r = Area { x0: self.x0.max(o.x0), y0: self.y0.max(o.y0), x1: self.x1.min(o.x1), y1: self.y1.min(o.y1) };
        (r.x1 > r.x0 && r.y1 > r.y0).then_some(r)
    }
}

/// One part of the frame a platform's interface covers.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Zone {
    /// `top`, `bottom`, `left`, `right` or `buttons`.
    pub name: &'static str,
    /// What covers it there.
    pub what: &'static str,
    pub area: Area,
}

/// How far in from each edge (shares of the frame) a platform's interface reaches, and its button
/// column on the right, if it has one beyond the right margin: `(width, top)` shares.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Margins {
    pub top: f64,
    pub bottom: f64,
    pub left: f64,
    pub right: f64,
    pub buttons: Option<(f64, f64)>,
}

/// One that overlaps a zone: which, and how much of the thing (a share of its area) is in it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Overlap {
    pub zone: &'static str,
    pub share: f64,
}

/// Overlaps smaller than this share of the thing's own area do not count (an outline's edge).
pub const MIN_OVERLAP: f64 = 0.01;

impl Platform {
    pub const ALL: [Platform; 4] = [Platform::Youtube, Platform::Reels, Platform::TikTok, Platform::Vertical];

    /// `youtube` (also `shorts`, `yt`), `reels` (`instagram`, `ig`), `tiktok` (`tt`) or
    /// `vertical` (`all`, `any`, `generic`); case and `-` / `_` / spaces do not matter.
    pub fn from_name(s: &str) -> Option<Self> {
        let k: String = s.trim().chars().filter(|c| !matches!(c, '-' | '_' | ' ')).flat_map(char::to_lowercase).collect();
        match k.as_str() {
            "youtube" | "youtubeshorts" | "shorts" | "yt" => Some(Platform::Youtube),
            "reels" | "instagram" | "instagramreels" | "ig" => Some(Platform::Reels),
            "tiktok" | "tt" => Some(Platform::TikTok),
            "vertical" | "all" | "any" | "generic" => Some(Platform::Vertical),
            _ => None,
        }
    }
    /// The name parameters take.
    pub fn name(self) -> &'static str {
        match self {
            Platform::Youtube => "youtube",
            Platform::Reels => "reels",
            Platform::TikTok => "tiktok",
            Platform::Vertical => "vertical",
        }
    }
    /// The platform's own name, for labels (plain text).
    pub fn label(self) -> &'static str {
        match self {
            Platform::Youtube => "YouTube Shorts",
            Platform::Reels => "Instagram Reels",
            Platform::TikTok => "TikTok",
            Platform::Vertical => "All vertical platforms",
        }
    }
    /// The margins (see the module docs for where they come from).
    pub fn margins(self) -> Margins {
        let tiktok_buttons = Some((0.20, 0.40));
        match self {
            Platform::Youtube => Margins { top: 0.15, bottom: 0.35, left: 0.05, right: 0.18, buttons: None },
            Platform::Reels => Margins { top: 0.14, bottom: 0.35, left: 0.06, right: 0.06, buttons: None },
            Platform::TikTok => Margins { top: 0.13, bottom: 0.35, left: 0.12, right: 0.12, buttons: tiktok_buttons },
            Platform::Vertical => {
                let all = [Platform::Youtube, Platform::Reels, Platform::TikTok].map(Platform::margins);
                let most = |f: fn(&Margins) -> f64| all.iter().map(f).fold(0.0, f64::max);
                Margins { top: most(|m| m.top), bottom: most(|m| m.bottom), left: most(|m| m.left), right: most(|m| m.right), buttons: tiktok_buttons }
            }
        }
    }
    /// The parts of the frame the interface covers.
    pub fn zones(self) -> Vec<Zone> {
        let m = self.margins();
        let mut z = vec![
            Zone { name: "top", what: "title, search and menus", area: Area::new(0.0, 0.0, 1.0, m.top) },
            Zone { name: "bottom", what: "channel, description, sound and progress bar", area: Area::new(0.0, 1.0 - m.bottom, 1.0, 1.0) },
            Zone { name: "left", what: "the edge some phones crop", area: Area::new(0.0, m.top, m.left, 1.0 - m.bottom) },
            Zone { name: "right", what: "like, comment and share buttons", area: Area::new(1.0 - m.right, m.top, 1.0, 1.0 - m.bottom) },
        ];
        if let Some((w, top)) = m.buttons.filter(|(w, _)| *w > m.right) {
            z.push(Zone { name: "buttons", what: "the button column", area: Area::new(1.0 - w, top, 1.0 - m.right, 1.0 - m.bottom) });
        }
        z
    }
    /// The frame inside the margins (the button column may still reach into its lower right; see
    /// [`Platform::safe_span`]).
    pub fn safe_area(self) -> Area {
        let m = self.margins();
        Area::new(m.left, m.top, 1.0 - m.right, 1.0 - m.bottom)
    }
    /// The safe stretch across the frame (left, right shares) for something spanning rows `y0`
    /// to `y1` (shares): inside the side margins and clear of the button column where they meet.
    pub fn safe_span(self, y0: f64, y1: f64) -> (f64, f64) {
        let m = self.margins();
        let right = match m.buttons {
            Some((w, top)) if y0.max(y1) > top && w > m.right => 1.0 - w,
            _ => 1.0 - m.right,
        };
        (m.left, right)
    }
    /// The zones `area` (shares of the frame) reaches into, by more than [`MIN_OVERLAP`] of
    /// itself; empty when it is safe.
    pub fn overlaps(self, area: &Area) -> Vec<Overlap> {
        let own = area.size();
        if own.is_nan() || own <= 0.0 {
            return Vec::new();
        }
        self.zones()
            .iter()
            .filter_map(|z| {
                let share = z.area.intersect(area)?.size() / own;
                (share > MIN_OVERLAP).then_some(Overlap { zone: z.name, share })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_published_numbers_round_outwards() {
        // YouTube: 288 / 672 / 48 / 192 px of 1080×1920
        let y = Platform::Youtube.margins();
        assert!(y.top >= 288.0 / 1920.0 && y.bottom >= 672.0 / 1920.0 && y.left >= 48.0 / 1080.0 && y.right >= 192.0 / 1080.0);
        // Instagram Reels: 14 / 35 / 6 / 6 %
        let r = Platform::Reels.margins();
        assert!(r.top >= 0.14 && r.bottom >= 0.35 && r.left >= 0.06 && r.right >= 0.06);
        // TikTok: 160 / 440 / 80 px of 720×1280, a 120 px column from 560 px
        let t = Platform::TikTok.margins();
        assert!(t.top >= 160.0 / 1280.0 && t.bottom >= 440.0 / 1280.0 && t.left >= 80.0 / 720.0 && t.right >= 80.0 / 720.0);
        let (bw, btop) = t.buttons.unwrap();
        assert!(bw >= 120.0 / 720.0 && btop <= 560.0 / 1280.0);
        // the generic zone is at least as strict as every platform
        let v = Platform::Vertical.margins();
        for p in [Platform::Youtube, Platform::Reels, Platform::TikTok] {
            let m = p.margins();
            assert!(v.top >= m.top && v.bottom >= m.bottom && v.left >= m.left && v.right >= m.right, "{p:?}");
            assert!(p.safe_area().intersect(&Platform::Vertical.safe_area()) == Some(Platform::Vertical.safe_area()), "{p:?}");
        }
    }

    #[test]
    fn names_parse_and_zones_cover_the_edges() {
        for p in Platform::ALL {
            assert_eq!(Platform::from_name(p.name()), Some(p));
            assert_eq!(serde_json::to_value(p).unwrap(), serde_json::json!(p.name()));
            let safe = p.safe_area();
            assert!(safe.width() > 0.5 && safe.height() > 0.4, "{p:?}: {safe:?}");
            assert!(p.overlaps(&safe).iter().all(|o| o.zone == "buttons"), "{p:?}");
            // the four edges are covered
            for edge in [Area::new(0.0, 0.0, 1.0, 0.01), Area::new(0.0, 0.99, 1.0, 1.0), Area::new(0.0, 0.5, 0.01, 0.51), Area::new(0.99, 0.5, 1.0, 0.51)] {
                assert!(!p.overlaps(&edge).is_empty(), "{p:?}: {edge:?}");
            }
        }
        for (s, p) in [("YouTube Shorts", Platform::Youtube), ("ig", Platform::Reels), ("Tik-Tok", Platform::TikTok), ("ALL", Platform::Vertical)] {
            assert_eq!(Platform::from_name(s), Some(p), "{s}");
        }
        for bad in ["", "snapchat", "you tube shorts x", "\u{0}"] {
            assert_eq!(Platform::from_name(bad), None, "{bad}");
        }
    }

    #[test]
    fn a_bottom_caption_is_unsafe_and_a_middle_one_is_not() {
        let frame = (1080.0, 1920.0);
        // a caption 120 px tall, its bottom 24 % up the frame: in the bottom zone of all of them
        let low = Area::from_pixels([140.0, 1920.0 * 0.76 - 120.0, 940.0, 1920.0 * 0.76], frame);
        for p in Platform::ALL {
            let o = p.overlaps(&low);
            assert!(o.iter().any(|o| o.zone == "bottom" && o.share > 0.9), "{p:?}: {o:?}");
        }
        // above the bottom zone and inside the side margins: safe on YouTube
        let (l, r) = Platform::Youtube.safe_span(0.55, 0.62);
        let ok = Area::new(l + 0.01, 0.55, r - 0.01, 0.62);
        assert!(Platform::Youtube.overlaps(&ok).is_empty());
        // TikTok's button column narrows the span low in the frame, not high
        assert!(Platform::TikTok.safe_span(0.5, 0.6).1 < Platform::TikTok.safe_span(0.2, 0.3).1);
        assert!(!Platform::TikTok.overlaps(&Area::new(0.3, 0.5, 0.85, 0.6)).is_empty());
    }

    #[test]
    fn hostile_areas_never_panic() {
        for a in [
            Area::new(f64::NAN, f64::INFINITY, -1e300, 1e300),
            Area::new(0.5, 0.5, 0.5, 0.5),
            Area::from_pixels([f64::NAN; 4], (0.0, f64::NAN)),
            Area::from_pixels([-5.0, -5.0, 1e9, 1e9], (1080.0, 1920.0)),
        ] {
            for p in Platform::ALL {
                let o = p.overlaps(&a);
                assert!(o.iter().all(|o| o.share.is_finite() && o.share >= 0.0), "{a:?}: {o:?}");
                let _ = p.safe_span(f64::NAN, f64::INFINITY);
            }
            assert!(a.x0 <= a.x1 && a.y0 <= a.y1);
        }
    }
}
