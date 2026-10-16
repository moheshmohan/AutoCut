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
`photoslibrary.appendonly` (add media to the library; read back only what the app added).
FilmCraft asks for it only when you set up uploads (`photos auth --upload`), so picker-only
users are never asked.

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

`photos auth` requests only the read-only Picker scope (`photospicker.mediaitems.readonly`);
`--upload` adds `photoslibrary.appendonly`, and nothing else is ever requested. Tokens never
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
```

`photos pick` opens the picker in your default browser (`--no-browser` just prints the link),
waits for you to press **Done** — 600 s by default, `--timeout` up to 3600 — then downloads
each item and runs `file.import`. `--demo`/`--project`/`--bridge` choose where it imports, as
for every other command. Progress goes to stderr; the result JSON (`items`, `downloaded`,
`cached`, `failed`) to stdout.

Over MCP (`filmcraft-cli mcp --demo` / `--bridge`): `googlephotos_pick` (same flow,
`max_items`, `wait_seconds` default 300, `session` to resume, and `query` for the
hands-free autopick agent below), `googlephotos_cached` and `googlephotos_add` (re-import
cached items). Without `query`, an agent tells the user what to choose:

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

- **No search, no browsing.** The Picker exposes no albums, no queries, no filters: the user
  chooses every item by hand in Google's UI. An agent can only ask the user to pick. Items
  already in the cache can be re-imported by id without the picker.
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
  (no album; Google may re-encode for playback). A cancelled upload leaves nothing in the
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
- `upload.rs` — the resumable upload and `batchCreate` behind the `Transport` trait (ureq in
  the product, a fake Google in the tests), file checks, retries, error mapping.
- `lib.rs` — `GooglePhotos::pick` and `GooglePhotos::upload`, the engine uploader hosts
  install for `photos.upload` (`engine_uploader`, readiness read from disk without network),
  and the browser opener (https links only).

`cargo test -p filmcraft-gphotos` is fully offline: session, polling-interval and media item
JSON parsing (including hostile and truncated input), the polling rule, token store and scope
checks, consent URLs, the loopback redirect handling, cache path sanitizing and the
download-host check, and the whole upload protocol against a fake Google (chunking, resume
after dropped or half-kept chunks and lost final answers, retry caps, cancellation, hostile
session URLs, granularities, offsets and tokens, `batchCreate` answers, error mapping).

A real upload is opt-in — it adds the file to your library:

```sh
FILMCRAFT_GPHOTOS_LIVE_UPLOAD=/path/to/clip.mp4 cargo test -p filmcraft-gphotos --test live_upload -- --nocapture
```
