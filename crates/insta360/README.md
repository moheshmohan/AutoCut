# filmcraft-insta360

Insta360 footage for the editor (layer L5): the user's 360° recordings on disk, and downloads
from their Insta360 cloud through Insta360 Studio. The engine's `insta360.*` commands
(`crates/engine/src/insta360.rs`) run it; the desktop app, the CLI (`filmcraft-cli insta360 …`)
and the MCP server (`insta360_list`, `insta360_import`, `insta360_download`) install it.

## Finding recordings (`library`)

An Insta360 camera names its files `VID_<yyyymmdd>_<hhmmss>_<ll>_<nnn>.insv` (video),
`IMG_….insp` (photos) and `LRV_…_01_<nnn>.lrv` (the low-resolution copy it records beside a
video, attached as its proxy). Studio's cloud download puts each one in
`~/Downloads/Insta360 Download/<stem>-Original/`; a camera card mounts with them in
`DCIM/Camera01`. A recording is complete when its file (and its proxy) ends with the camera's
trailer (`filmcraft_isobmff::insta360`): a download still being written has none yet. Photos
(`.insp`) are imported as 360° stills (`filmcraft_codecs::insp`).

## Downloading from the cloud (`studio`)

The Insta360 cloud has no public API; Insta360 Studio, signed in by the user, is the way to it.
Its media list is drawn by the app itself (no accessibility tree), so the local screen agent
(`filmcraft-screenagent`: Qwen2.5-VL through Ollama, Screen Recording and Accessibility granted)
reads the screen and clicks like a person, inside Studio's window only:

1. Studio in front, **Cloud Media** chosen, the list in **List View** (full names and dates). The
   view toggle is taken only as the leftmost of a clear row of the toolbar's icons (the trash is
   two along) and checked after; the list is not switched back.
2. The rows read; the wanted name found, scrolling the list in name order (budgeted).
3. Right-click → the menu item checked to say **Download** (the menu ends with "Remove from
   Insta360 Cloud", which is never clicked) → click.
4. The Download dialog's file name read: the wanted recording → **OK** (Studio's own folder);
   another → **Cancel**.
5. Wait for the file and its proxy to finish.

The mouse and keyboard are one agent's at a time: the `chrome` lock (`filmcraft-locks`) is held
while Studio is driven. Nothing reads Studio's own files, settings, tokens or caches.

The vision prompts were checked against Studio 6.0.7's screens (rows read exactly, the Download
item and OK button found within a few pixels) and the whole flow run live once.
