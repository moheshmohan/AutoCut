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
6. Run `filmcraft-cli photos auth` — a browser opens, you consent, and the sign-in goes into
   the macOS Keychain (elsewhere: an owner-only `tokens.json` next to the secret). Done. See
   [Where the sign-in lives](#where-the-sign-in-lives).

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
leave this machine. A sign-in from the old Library-API version of this crate is refused with a
"run `photos auth` again" error.

### Where the sign-in lives

The sign-in runs on the shared OAuth module, [`filmcraft-oauth`](../oauth/README.md):
authorization code with PKCE (S256), the consent page in your default browser, the redirect
caught on `http://127.0.0.1:<a random port>` with its `state` checked, then a page saying you
can return to FilmCraft.

- **On a Mac** the tokens are a Keychain item named **FilmCraft Google Photos** (Keychain Access
  shows it; its account is the data directory, so each `FILMCRAFT_DATA_DIR` has its own).
  `<data dir>/google-photos/signin.json` beside the secret notes the granted scopes, without
  any token, so readiness checks never open the Keychain.
- **Elsewhere**, with `FILMCRAFT_SECRET_STORE=file`, or when the Keychain cannot take a save (a
  locked keychain nobody unlocks within two minutes, say), they are `tokens.json` there,
  readable only by you (`0600`); `photos auth` says which, and why.
- **Moving in:** a `tokens.json` from an earlier FilmCraft moves into the Keychain the first
  time it is used (`photos pick`, `upload`, an album command, the app's upload); the file is
  removed only after the Keychain item reads back the same. Nothing to do by hand.
- **Expiry:** while your Google Cloud app is in Testing mode, Google ends its sign-ins after
  7 days. The next call then says "Google no longer accepts the saved sign-in … sign in again:
  `filmcraft-cli photos auth`" (with `--upload` or `--albums` when the sign-in had them) —
  run that, and incremental consent keeps every permission you granted before.
- **Signing out:** `filmcraft-cli photos auth --sign-out` revokes the sign-in at Google and
  removes the Keychain item, `tokens.json` and `signin.json` (if Google cannot be reached it
  says so; remove FilmCraft's access at https://myaccount.google.com/connections).

## Using it

```sh
filmcraft-cli photos pick --project my.fcproj --save   # pick in the browser → download → import
filmcraft-cli photos pick --max-items 5 --timeout 300   # cap the selection and the wait
filmcraft-cli photos pick --session <id>                # keep waiting on a timed-out session
filmcraft-cli photos cached                             # what earlier picks downloaded
filmcraft-cli photos add <id>… --bridge 127.0.0.1:9877  # import cached items again, no picker

# hands-free: a local vision agent (Qwen2.5-VL via Ollama) drives the picker itself —
# it types the query into the picker's search, selects matching videos, presses Done
filmcraft-cli photos pick --auto "beach sunset videos" --max-items 5 --bridge 127.0.0.1:9877

# a whole album as the seed: the agent searches the picker for the exact title "Robot
# showcase" (the result is the album's items, after a delay it waits for) and selects all of
# it — tile by tile in the page when Chrome allows JavaScript from Apple Events (View ▸
# Developer), else by eyes (--videos-only downloads only its videos)
filmcraft-cli photos pick --album "Robot showcase" --bridge 127.0.0.1:9877

# many searches, ONE picking session and ONE tab: each typed in turn, every result selected
# (the selection accumulates), Done once, one download batch
filmcraft-cli photos pick --auto "videos Dubai robots" --auto "videos Tokyo robots" --all
filmcraft-cli photos pick --queries searches.txt --all          # one search per line, # comments

# photos as well as videos (a searched query otherwise gets "videos" added and only video
# tiles are clicked): everything from one day
filmcraft-cli photos pick --auto "October 30 2026" --all --photos --since 2026-10-30 --project my.fcproj --save

# a whole history in a handful of sessions: one "videos <Month> <Year>" search per month
# (--month-days 10-21: one per day of those days in each month instead); implies --all
filmcraft-cli photos pick --sweep 2017-10..2025-12 --bridge 127.0.0.1:9877

# close the picker tabs earlier runs left in Chrome (and nothing else)
filmcraft-cli photos tidy
```

`photos pick` opens the picker in a new tab of your default browser — when the agent drives
it (`--auto`, `--album`, many searches), on macOS, after closing Chrome's earlier picker tabs,
so picks do not pile up tabs (`--no-browser` just prints the link) — waits for you to press **Done** — 600 s by default, `--timeout` up to 3600 — then downloads
each item and runs `file.import` with `reuse`: a file the project already holds (picked
before) is reused, not imported twice. `--demo`/`--project`/`--bridge` choose where it imports,
as for every other command. Progress goes to stderr; the result JSON (`items`, `downloaded`,
`cached`, `failed`) to stdout.

Over MCP (`filmcraft-cli mcp --demo` / `--bridge`): `googlephotos_pick` (same flow,
`max_items`, `wait_seconds` default 300, `session` to resume, `query` — or `queries`, many
searches in one session, with `all` — for the hands-free autopick agent below, or `album`
(+ `videos_only`) to pick a whole album),
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
build reads the whole album back through the picker — `showcase --album "<title>" --prompt "…"`
makes the video straight from an album in one command ([crates/autocut: Album as the
seed](../autocut/README.md#album-as-the-seed)).

```sh
filmcraft-cli photos auth --albums                                   # once
filmcraft-cli photos album create "Robot showcase" --description "Every robot, 2020–2026"
filmcraft-cli photos upload out/showcase.mp4 --album "Robot showcase" # into it (made if missing)
filmcraft-cli showcase --report robots.json --export out/showcase.mp4 --upload --upload-album "Robot showcase"
filmcraft-cli photos album add "Robot showcase" <mediaItemId>…       # earlier uploads
filmcraft-cli photos albums                                          # what FilmCraft made
# … add photos and videos to the album in Google Photos …
filmcraft-cli photos pick --album "Robot showcase" --bridge 127.0.0.1:9877   # the next round's seed
filmcraft-cli showcase --album "Robot material" --prompt "every robot since 2020" \
    --export out/showcase.mp4 --upload --upload-album "Robot showcase"     # …or the whole video from it
```

- **`photos album create <title> [--description D]`** prints the album as JSON (`id`, `title`,
  `productUrl`, `created`, `noteAdded`). It is idempotent: an album FilmCraft made with exactly
  that title is reused, never duplicated (titles are cleaned — control characters out, runs of
  white space one space — and at most 500 characters; longer is an error, never cut). The
  description becomes a text note at the top of a new album (`albums.addEnrichment`); it is not
  added again to an existing one.
- **`photos upload <file> --album <title|id>`** (and `showcase … --upload --upload-album T`) resolves the
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
- **MCP:** `googlephotos_upload {…, album}`; `googlephotos_pick {album, videos_only?}`;
  `showcase_analyze {album, prompt}` (a showcase from the album's clips).

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

### Big libraries: many searches, few sessions, few tabs

Google gives apps no listing of a library, only its picker, whose search understands dates.
So a large library is pulled with searches — a list (`--auto` repeated, `--queries <file>`) or
a sweep of date searches (`--sweep FROM..TO`, built by [`sweep`](src/sweep.rs): `2017-10..2025-12`,
`2019..2021`, `2022`, `2022-10`; `--month-days 10-21` for days) — and the agent types them into
**one** picking session in **one** tab:

- it clears the search box and types each search in turn; the picker keeps the selection
  while the search changes, and the header's "N selected" must grow (a search whose count
  falls cleared some earlier selections: it and the searches before it are retried);
- every result is selected (`--all`, implied by a sweep): a date search by the picker's date
  groups — the round check left of each date heading selects that day — else by the range
  gesture; a search with no results is skipped;
- the session's item cap is `--max-items` (default and most 2000, Google's limit); before each
  search the agent checks the room left (guessing the next search's size from the biggest so
  far) and its budget, and leaves the rest for the next session. Google's picker does not stop
  a selection at the cap — date groups and ranges select past it ("143 selected · Select up to
  40 items", live) — but then refuses Done, so a date search whose groups pass the cap has the
  same days clicked once more (that unselects them) and goes to the next session; one too big
  even for an empty session is given up and listed as too big (split it by day,
  `--month-days 1-31`, or raise `--max-items`). A range past the cap cannot be undone, so the
  agent stops before a Done the picker would refuse;
- Done is pressed once and the selection downloads in one batch.

Sessions take up to 20 searches each; what one session left (or failed once) goes first into
the next, until nothing is left or a session gets nowhere. Searches still left at the end are
written to `<data dir>/google-photos/left-searches.txt` with the command to run them
(`photos pick --queries <that file> --all`). Everything downloads into the cache, so a search
picked twice downloads nothing twice.

### Tabs

- The agent closes the picker tab it drove when it finishes — after Done (the picker's finished
  page), or when it gives up — that very tab, named by its Chrome window and tab id when the
  agent began, and only while its address is still the picker (`photos.google.com/picker…` or
  `photos.google.com/integration/picker…`); never whatever tab is in front by then. When Chrome asks
  "Leave site?" (the picker holds its selection), the agent lists the dialog's buttons by their
  text and clicks the one that reads Leave — nothing else.
- A session the agent gave up on is deleted (`sessions.delete`), not left to linger.
- Before an agent-driven session opens its tab, Chrome's earlier picker tabs are closed (a tab
  holding a selection asks "Leave site?" and is closed once more, which Chrome takes as Leave),
  so its tab is the only one. Pointing the open picker tab at the new session was tried: a tab
  holding a selection asks "Leave site?" while Chrome already reports the new address, so the
  agent met the dialog instead of the picker. A pick by hand leaves open picker tabs be (you may
  be picking in one) and says how many there are.
- `photos tidy` closes every picker tab in Chrome — those tabs only, by window, tab id *and*
  address. A tab that asks "Leave site?" is closed once more, which Chrome takes as Leave (no
  model needed); one still open after that gets its Leave clicked as above (when the grounding
  model is in Ollama).

The first time, macOS asks once whether your terminal may control Google Chrome (Automation):
allow it, or the tabs are left as they are (and `photos tidy` cannot list them).

## How a pick works

1. `POST photospicker.googleapis.com/v1/sessions` (optionally `pickingConfig.maxItemCount`).
2. Open the session's `pickerUri` + `/autoclose` in a new browser tab (for an agent, after
   closing Chrome's earlier picker tabs) — so Google Photos closes its tab when the user is done (an
   agent closes it itself).
3. Poll `GET /v1/sessions/{id}` at Google's `pollingConfig.pollInterval` (clamped to 1–30 s)
   until `mediaItemsSet`, Google's `timeoutIn` runs out, or our own wait does. Our own timeout
   leaves the session open so it can be resumed — unless an agent drove it: then it is
   abandoned and deleted; Google's ends the pick (and the session is deleted).
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
- **Album picks depend on Google's search.** The picker has no Albums view and never suggests
  an album; Return on an album's exact title shows the album's items, and the agent selects
  that whole result. A title that is no album's gets ordinary search results instead, taken
  for the album up to the run's item cap (past it nothing is confirmed): give the exact title.
- **A person has to be there.** Each pick needs someone at a browser signed in to the Google
  account; there is no unattended import.
- **Download links last 60 minutes.** A very large selection over a slow line can outlive
  them; the late items fail with a 403 and can be picked again (cached ones are skipped).
- **Videos download whole** (`=dv`, Google's high-quality file, up to GBs) and only once Google
  has finished processing them; a still-processing video is reported in `failed`.
- **Photos come without location.** Google strips GPS from `=d` downloads (and from videos:
  none of 752 downloaded clips had a location atom). The places are still on the Google Photos
  page: `filmcraft-cli photos locations --album "<title>"` reads them from each item's Info panel
  with the screen agent (`crates/screenagent/README.md`).
- **Test-mode sign-ins lapse after 7 days** if the Google Cloud app stays in Testing; the error
  names the `photos auth` to re-run (`--upload` or `--albums` when the sign-in had them).
  Promote the app to "In production" (still unverified, 100 users) to stop that.
- **Uploads count against your Google storage** and land in the library as Google stores them
  (in an album only with `--album`; Google may re-encode for playback). A cancelled upload leaves nothing in the
  library; Google drops the unfinished session on its own.

## Layout

- `auth.rs` — Google Photos on [`filmcraft-oauth`](../oauth/README.md) (PKCE, the system
  browser, a loopback redirect on a random port, waiting at most five minutes and ignoring stray
  connections): the picker-only, upload or album consent (`Access`), refresh (a lapsed sign-in
  names the `photos auth` that grants what it had), the token vault (the Keychain item
  "FilmCraft Google Photos", else the `0600` `tokens.json`; the first-use move into the
  Keychain), the granted-scope checks (`Grants`, on the tokens or the on-disk summary), sign-out.
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
- `lib.rs` — `GooglePhotos::pick` (and `prepare_with`, for a caller that shows the picker its
  own way) and `GooglePhotos::upload`, the engine uploader hosts install for `photos.upload`
  (`engine_uploader`, readiness read from disk without network), and the browser opener (https
  links only).
- `sweep.rs` — the date searches of a sweep: month and day searches over a range, ranges
  parsed and bounded.

The tab handling lives in `filmcraft-screenagent` (`tabs.rs`) and the many-session loop in
`filmcraft-automation` (`photos.rs`), which drives this crate's sessions.

`cargo test -p filmcraft-gphotos` is fully offline: session, polling-interval and media item
JSON parsing (including hostile and truncated input), the polling rule, the token vault (files,
and a fake Keychain for the move of a `tokens.json` into it) and scope checks, consent URLs and
the command a lapsed sign-in names (the OAuth flow itself — PKCE, the loopback redirect, a mock
token endpoint — is tested in `filmcraft-oauth`), readiness read from disk, cache path sanitizing and the
download-host check, and the whole upload protocol against a fake Google (chunking, resume
after dropped or half-kept chunks and lost final answers, retry caps, cancellation, hostile
session URLs, granularities, offsets and tokens, `batchCreate` answers, error mapping), and
the album calls against a fake library (request bodies and URLs, paging, repeated page tokens,
the 2000-album cap, idempotent and lost-answer creation, batches of 50, consent and
app-created-only errors, hostile pages: huge, missing fields, wrong types, non-JSON), and the
sweep's searches (ranges, leap years, days a month lacks, the size bound).

Real calls are opt-in — they change your library:

```sh
FILMCRAFT_GPHOTOS_LIVE_UPLOAD=/path/to/clip.mp4 cargo test -p filmcraft-gphotos --test live_upload -- --nocapture
FILMCRAFT_GPHOTOS_LIVE_ALBUM="FilmCraft test album" cargo test -p filmcraft-gphotos --test live_albums -- --nocapture
```
