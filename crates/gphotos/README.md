# filmcraft-gphotos

Google Photos for the editor: open the Google Photos picker, let the user choose photos and
videos, download the originals into a cache, and import them into the project; and upload
finished exports back to the user's library — from the CLI, over MCP and in the app.

## Why a picker, not search

On 2025-03-31 Google removed the Library API scopes that let apps read a user's whole library
(`photoslibrary.readonly`, `photoslibrary`, `photoslibrary.sharing`). The Library API now only
returns media the app uploaded itself, and answers `403 PERMISSION_DENIED` for everything
else. The supported way to get at a user's existing photos is the
[Photos Picker API](https://developers.google.com/photos/picker): the app opens a *session*,
the user chooses items in Google Photos' own UI, and the app may then list and download
exactly those items.

Uploading is still the Library API's job, with the scope Google kept for it:
`photoslibrary.appendonly` (add media to the library, create albums, add to them). FilmCraft
asks for it only when you set up uploads (`photos auth --upload`), so picker-only users are
never asked. Albums FilmCraft makes also need `photoslibrary.readonly.appcreateddata` (read
back the albums and items *this app* created — the only Library read scope left), asked for
only by `photos auth --albums`. See [Albums](#albums).

## One-time setup (your Google account)

The Picker API needs an OAuth client of your own. For a private tool, Google's "Testing"
publishing mode is fine — you are the only user.

1. Open [Google Cloud Console](https://console.cloud.google.com/) → create a project (any name).
2. **APIs & Services → Library** → search **"Google Photos Picker API"** → **Enable**.
3. **APIs & Services → OAuth consent screen** → External → fill in an app name and your email
   → add yourself as a **Test user** (Users panel). Publishing stays **Testing**.
4. **APIs & Services → Credentials → Create credentials → OAuth client ID** → application type
   **Desktop app** → create → download the JSON. Desktop clients accept loopback redirects on
   any port, so no redirect URI needs adding.
5. Save it as `~/Library/Application Support/FilmCraft/google-photos/client_secret.json`
   (or the equivalent data dir on your platform).
6. Run `filmcraft-cli photos auth` — a browser opens, you consent, tokens land next to the
   secret (owner-only permissions). Done.

To **upload** exports as well:

7. **APIs & Services → Library** → search **"Photos Library API"** → **Enable** (in the same
   project). Without it every upload fails with a 403 that FilmCraft reports as "the Photos
   Library API is not enabled in your Google Cloud project".
8. Run `filmcraft-cli photos auth --upload` and leave the "add to your Google Photos library"
   permission ticked. This is incremental consent (`include_granted_scopes=true`): the new
   sign-in keeps the picker permission, and a later plain `photos auth` keeps the upload one.

To make **albums** (and upload into them):

9. Run `filmcraft-cli photos auth --albums` and leave both Google Photos permissions ticked
   ("add to your Google Photos library" and the one to see what FilmCraft added). Same
   incremental consent; it includes `--upload`.

`photos auth` requests only the read-only Picker scope (`photospicker.mediaitems.readonly`);
`--upload` adds `photoslibrary.appendonly`; `--albums` adds that and
`photoslibrary.readonly.appcreateddata`. Nothing else is ever requested (the retired
`photoslibrary`, `photoslibrary.readonly` and `photoslibrary.sharing` never are). Tokens never
leave this machine and are stored `0600`. A `tokens.json` from the old Library-API version of
this crate is refused with a "run `photos auth` again" error.

## Using it

```sh
filmcraft-cli photos pick --project my.fcproj --save   # pick in the browser → download → import
filmcraft-cli photos pick --max-items 5 --timeout 300   # cap the selection and the wait
filmcraft-cli photos pick --session <id>                # keep waiting on a timed-out session
filmcraft-cli photos cached                             # what earlier picks downloaded
filmcraft-cli photos add <id>… --bridge 127.0.0.1:9876  # import cached items again, no picker

# hands-free: a local vision agent (Qwen2.5-VL via Ollama) drives the picker itself —
# it types the query into the picker's search, selects matching videos, presses Done
filmcraft-cli photos pick --auto "beach sunset videos" --max-items 5 --bridge 127.0.0.1:9876

# a whole album as the seed: the agent opens the album titled "Robot showcase" in the picker
# and selects all of it (--videos-only keeps its videos)
filmcraft-cli photos pick --album "Robot showcase" --bridge 127.0.0.1:9876
```

`photos pick` opens the picker in your default browser (`--no-browser` just prints the link),
waits for you to press **Done** — 600 s by default, `--timeout` up to 3600 — then downloads
each item and runs `file.import` with `reuse`: a file the project already holds (picked
before) is reused, not imported twice. `--demo`/`--project`/`--bridge` choose where it imports,
as for every other command. Progress goes to stderr; the result JSON (`items`, `downloaded`,
`cached`, `failed`) to stdout.

Over MCP (`filmcraft-cli mcp --demo` / `--bridge`): `googlephotos_pick` (same flow,
`max_items`, `wait_seconds` default 300, `session` to resume, `query` for the
hands-free autopick agent below, or `album` (+ `videos_only`) to pick a whole album),
`googlephotos_cached` and `googlephotos_add` (re-import cached items). Without `query` or
`album`, an agent tells the user what to choose:

> "add the beach sunset videos from my Google Photos to the project" → the agent says "pick
> the beach sunset clips in the Google Photos tab that just opened" → `googlephotos_pick
> {max_items: 10}` → `timeline.place` with the imported clips.

If the agent's MCP client has short tool timeouts, it calls `googlephotos_pick
{wait_seconds: 0}` first (opens the picker, returns `status: pending` with a `session` and
`pickerUri`) and then `googlephotos_pick {session: …}` to wait for the result.

### Uploading exports

```sh
filmcraft-cli photos upload out/showcase.mp4 --description "Robots, 2020–2026"
filmcraft-cli showcase --report robots.json --export out/showcase.mp4 --upload
```

- **CLI:** `photos upload <file> [--description D]` prints progress on stderr and the new
  media item as JSON (`mediaItemId`, `productUrl`, `fileName`, `mimeType`, `bytes`,
  `resumed`). `showcase … --export out.mp4 --upload` uploads the export once the build
  succeeds (described with `--description`, else the title); a missing sign-in or consent is
  reported before the build starts.
- **Engine command:** `photos.upload {"path", "description"?, "wait"?}` runs the upload as a job
  (`jobs.list` shows bytes done / total and the status; `jobs.cancel` stops it between chunks;
  the finished entry carries `upload`: the media item). It is enabled only when the host
  installed an uploader (`Session::photos`, from `engine_uploader()`) and the sign-in has the
  upload permission; otherwise its reason names the command that fixes it.
- **MCP:** `googlephotos_upload {path, description?}` runs that job as a long call: MCP
  progress notifications in bytes, `notifications/cancelled` stops it (the local file is never
  touched), the answer carries `upload`. Bridge mode runs it in the app.
- **App:** Export mode shows **Upload to Google Photos** under the summary once an export has
  finished (automation id `export.uploadGooglePhotos`; `export.upload.status` is the status
  line, `export.upload.open` the link to the new item). While Google Photos is not connected the
  button is disabled and the status line says what to run.

Google Photos takes videos (MP4, MOV, M4V, MKV, AVI, MPG, MTS/M2TS, WMV, 3GP, …) up to 10 GB and
photos (JPEG, PNG, HEIC/HEIF, WebP, AVIF, GIF, TIFF, BMP, ICO) up to 200 MB; anything else is
refused before a byte is sent.

## Albums

An album is how a video and its next iteration meet: FilmCraft uploads the cut into an album,
you add the material for the next round to that album by hand in Google Photos, and the next
build reads the whole album back through the picker.

```sh
filmcraft-cli photos auth --albums                                   # once
filmcraft-cli photos album create "Robot showcase" --description "Every robot, 2020–2026"
filmcraft-cli photos upload out/showcase.mp4 --album "Robot showcase" # into it (made if missing)
filmcraft-cli showcase --report robots.json --export out/showcase.mp4 --upload --album "Robot showcase"
filmcraft-cli photos album add "Robot showcase" <mediaItemId>…       # earlier uploads
filmcraft-cli photos albums                                          # what FilmCraft made
# … add photos and videos to the album in Google Photos …
filmcraft-cli photos pick --album "Robot showcase" --bridge 127.0.0.1:9876   # the next round's seed
```

- **`photos album create <title> [--description D]`** prints the album as JSON (`id`, `title`,
  `productUrl`, `created`, `noteAdded`). It is idempotent: an album FilmCraft made with exactly
  that title is reused, never duplicated (titles are cleaned — control characters out, runs of
  white space one space — and at most 500 characters; longer is an error, never cut). The
  description becomes a text note at the top of a new album (`albums.addEnrichment`); it is not
  added again to an existing one.
- **`photos upload <file> --album <title|id>`** (and `showcase … --upload --album T`) resolves the
  album first — before any byte is sent — and creates it when missing; `batchCreate` then adds
  the new item to it (`albumId`). An id-like argument (40+ id characters) is only ever looked up,
  never made a title.
- **`photos album add <title|id> <mediaItemId>…`** adds items FilmCraft uploaded (ids from
  `photos upload`) to one of its albums, 50 per `albums.batchAddMediaItems` call, duplicates sent
  once; a refusal part-way says how many went in.
- **`photos albums [--json]`** lists the albums FilmCraft made (paged 50 at a time, at most 2000;
  `complete: false` when it stopped early — and then `create` refuses to make a title it might
  have missed).
- **Engine commands** (app, CLI `exec`, MCP `command_run`): `photos.createAlbum {title,
  description?}`, `photos.albums`, `photos.addToAlbum {album, mediaItemIds}`, and
  `photos.upload`'s `album`. They are short network calls answered directly (an upload into an
  album is the usual upload job). Without `photos auth --albums` they are disabled, and the
  reason names that command.
- **MCP:** `googlephotos_upload {…, album}`; `googlephotos_pick {album, videos_only?}`.

### What Google allows (Library API since 2025-03-31)

| call | scope it needs | reference |
|---|---|---|
| `albums.create` | `photoslibrary.appendonly` | [albums/create](https://developers.google.com/photos/library/reference/rest/v1/albums/create) |
| `albums.addEnrichment` | `photoslibrary.appendonly` (or `edit.appcreateddata`) | [albums/addEnrichment](https://developers.google.com/photos/library/reference/rest/v1/albums/addEnrichment) |
| `mediaItems.batchCreate` with `albumId` | `photoslibrary.appendonly` | [mediaItems/batchCreate](https://developers.google.com/photos/library/reference/rest/v1/mediaItems/batchCreate) |
| `albums.batchAddMediaItems` | `photoslibrary.appendonly` (or `edit.appcreateddata`) | [albums/batchAddMediaItems](https://developers.google.com/photos/library/reference/rest/v1/albums/batchAddMediaItems) |
| `albums.list`, `albums.get` | `photoslibrary.readonly.appcreateddata` only | [albums/list](https://developers.google.com/photos/library/reference/rest/v1/albums/list) |
| `mediaItems.search` | `photoslibrary.readonly.appcreateddata` only | [mediaItems/search](https://developers.google.com/photos/library/reference/rest/v1/mediaItems/search) |

- An app lists, searches and reads only the albums and media items it created
  ([updates](https://developers.google.com/photos/support/updates)); `albums.list` returns only
  FilmCraft's albums, never the user's own.
- Google adds only media items the app uploaded, and only into albums the app created ("The
  media items and albums must have been created by the developer via the API"); 50 per call, at
  most 20,000 in an album; one bad id fails the whole call.
- **Items you add to a FilmCraft album by hand are not readable through the Library API**
  (`mediaItems.search` returns only app-created items; Google does not say so of album searches
  in as many words, but nothing else is returned to apps). That is why reading an album goes
  through the picker: `photos pick --album`.
- Album titles: at most 500 characters. `isWriteable` says whether FilmCraft can still add to
  an album; `mediaItemsCount` is a string Google sends (read as a number when it is one).

### An album as the seed: `photos pick --album`

The picker has no album API, and Google's
[picking guide](https://developers.google.com/photos/picker/guides/picking-experience) says
albums "are not shown directly" — users search for an album's title. So the screen agent
([crates/screenagent](../screenagent/README.md)) does what a person would:

1. opens the picker; if it grounds an **Albums** view (a small element labelled "album" —
   never a thumbnail-sized box, which a click would select) it switches to it and pages
   through the albums; otherwise it types the exact title into the picker's search and looks
   at the suggestions, then (after Return) the results;
2. clicks only an element whose label names the album (the same words, case and punctuation
   aside, optionally followed by its item count). No such album, or two with that title, is an
   error — **nothing is selected**;
3. checks the screen that opened is headed by the title, then selects all of it (first tile,
   wheel to the end, Shift+click the last tile, read "N selected") and presses Done.

`--videos-only` grounds the range on video tiles and skips any photo in between when
downloading (it is listed in `skipped`). `--album` and `--auto` are exclusive. Selecting every
item only ever happens inside a search result or an opened album, never the whole library.

## How a pick works

1. `POST photospicker.googleapis.com/v1/sessions` (optionally `pickingConfig.maxItemCount`).
2. Open the session's `pickerUri` + `/autoclose` in the browser, so Google Photos closes its
   tab when the user is done.
3. Poll `GET /v1/sessions/{id}` at Google's `pollingConfig.pollInterval` (clamped to 1–30 s)
   until `mediaItemsSet`, Google's `timeoutIn` runs out, or our own wait does. Our own timeout
   leaves the session open so it can be resumed; Google's ends the pick (and the session is
   deleted).
4. `GET /v1/mediaItems?sessionId=…` (every page, at most 2000 items).
5. Download each `baseUrl` with the `Authorization: Bearer` header — `=d` for photos (the
   original with Exif minus location), `=dv` for videos — streamed to a `.part` file, then
   renamed into `<data dir>/google-photos/cache/<media item id>/<filename>`.
6. `DELETE /v1/sessions/{id}` (Google limits open sessions).

## How an upload works

1. `POST photoslibrary.googleapis.com/v1/uploads` with `X-Goog-Upload-Protocol: resumable`,
   `X-Goog-Upload-Command: start`, `X-Goog-Upload-Content-Type: <mime>` and
   `X-Goog-Upload-Raw-Size`. Google answers with the session URL (`X-Goog-Upload-URL`) and the
   chunk granularity (`X-Goog-Upload-Chunk-Granularity`, 256 KiB when absent or unreadable).
2. The file, streamed from disk in 8 MiB chunks (a multiple of the granularity; one chunk in
   memory at a time), each with `X-Goog-Upload-Command: upload` and its `X-Goog-Upload-Offset`,
   the last with `upload, finalize`. Its body is the upload token.
3. A dropped chunk (connection error, 408, 429, 5xx) waits (1 s, doubling, at most 32 s), asks
   `X-Goog-Upload-Command: query` where Google got to (`X-Goog-Upload-Status` and
   `X-Goog-Upload-Size-Received`) and resumes from there — or, if the final chunk had arrived,
   takes the token from the query. At most 5 failures in a row and 20 in all.
4. `POST /v1/mediaItems:batchCreate` with
   `{"newMediaItems":[{"description", "simpleMediaItem": {"uploadToken", "fileName"}}]}`; the
   item's own `status` is checked and the new item's id and `productUrl` returned.

Google's answers are treated as hostile: the session URL must be an `https://` Google API host
(it receives the bearer token — no user info, no port), a granularity above 64 MiB is refused,
a reported received size larger than what was sent or than the file is an error, upload tokens
must be short printable ASCII, and only `https://photos.google.com/` product links are passed on.
Errors name their fix: the Photos Library API not enabled (403 `SERVICE_DISABLED`), a sign-in
without the upload permission (401 / insufficient scopes → `photos auth --upload`), a file too
large or of a type Google Photos does not take, an expired upload session.

## Honest limits

- **No search, no browsing through an API.** The Picker exposes no albums, no queries, no
  filters: items are chosen in Google's UI — by the user, or by the local screen agent
  (`--auto`, `--album`) driving that UI. Items already in the cache can be re-imported by id
  without the picker.
- **Album picks depend on what the picker shows.** Google documents no Albums view in the
  picker; the agent searches the album's title when it finds none, and needs the album to come
  up as a suggestion or a result. A title shared by two albums is refused.
- **A person has to be there.** Each pick needs someone at a browser signed in to the Google
  account; there is no unattended import.
- **Download links last 60 minutes.** A very large selection over a slow line can outlive
  them; the late items fail with a 403 and can be picked again (cached ones are skipped).
- **Videos download whole** (`=dv`, Google's high-quality file, up to GBs) and only once Google
  has finished processing them; a still-processing video is reported in `failed`.
- **Photos come without location.** Google strips GPS from `=d` downloads.
- **Test-mode sign-ins lapse after 7 days** if the Google Cloud app stays in Testing; re-run
  `photos auth` (`--upload` if you upload). Promote the app to "In production" (still
  unverified, 100 users) to stop that.
- **Uploads count against your Google storage** and land in the library as Google stores them
  (in an album only with `--album`; Google may re-encode for playback). A cancelled upload leaves nothing in the
  library; Google drops the unfinished session on its own.

## Layout

- `auth.rs` — the installed-app OAuth flow (browser + loopback redirect on
  `http://127.0.0.1:8417`, waiting at most five minutes and ignoring stray connections), the
  picker-only or picker-and-upload consent (`Access`), token refresh, the `0600` token store,
  the granted-scope checks.
- `api.rs` — the Picker client: sessions (create / get / delete), the polling rule,
  `mediaItems.list` paging, Google's error messages with setup hints.
- `cache.rs` — streamed downloads into a per-item cache; ids and filenames sanitized so
  nothing lands outside it; the bearer token only ever goes to Google's media hosts.
- `upload.rs` — the resumable upload and `batchCreate` (with an `albumId` when asked) behind
  the `Transport` trait (ureq in the product, a fake Google in the tests; POST and GET), file
  checks, retries, error mapping.
- `albums.rs` — the albums FilmCraft makes: create (idempotent by title; a lost creation
  answer is followed by a fresh listing, never a duplicate), paged listing with caps, find by
  title or id, batched adds, a note at the top, readable errors; ids are URL-safe or refused.
- `lib.rs` — `GooglePhotos::pick` and `GooglePhotos::upload`, the engine uploader hosts
  install for `photos.upload` (`engine_uploader`, readiness read from disk without network),
  and the browser opener (https links only).

`cargo test -p filmcraft-gphotos` is fully offline: session, polling-interval and media item
JSON parsing (including hostile and truncated input), the polling rule, token store and scope
checks, consent URLs, the loopback redirect handling, cache path sanitizing and the
download-host check, and the whole upload protocol against a fake Google (chunking, resume
after dropped or half-kept chunks and lost final answers, retry caps, cancellation, hostile
session URLs, granularities, offsets and tokens, `batchCreate` answers, error mapping), and
the album calls against a fake library (request bodies and URLs, paging, repeated page tokens,
the 2000-album cap, idempotent and lost-answer creation, batches of 50, consent and
app-created-only errors, hostile pages: huge, missing fields, wrong types, non-JSON).

Real calls are opt-in — they change your library:

```sh
FILMCRAFT_GPHOTOS_LIVE_UPLOAD=/path/to/clip.mp4 cargo test -p filmcraft-gphotos --test live_upload -- --nocapture
FILMCRAFT_GPHOTOS_LIVE_ALBUM="FilmCraft test album" cargo test -p filmcraft-gphotos --test live_albums -- --nocapture
```
