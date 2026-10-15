# filmcraft-gphotos

Google Photos for the editor: open the Google Photos picker, let the user choose photos and
videos, download the originals into a cache, and import them into the project — from the CLI
and over MCP.

## Why a picker, not search

On 2025-03-31 Google removed the Library API scopes that let apps read a user's whole library
(`photoslibrary.readonly`, `photoslibrary`, `photoslibrary.sharing`). The Library API now only
returns media the app uploaded itself, and answers `403 PERMISSION_DENIED` for everything
else. The supported way to get at a user's existing photos is the
[Photos Picker API](https://developers.google.com/photos/picker): the app opens a *session*,
the user chooses items in Google Photos' own UI, and the app may then list and download
exactly those items. FilmCraft never uploads to Google Photos, so it uses the Picker only.

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

Only the read-only Picker scope (`photospicker.mediaitems.readonly`) is requested; tokens
never leave this machine and are stored `0600`. A `tokens.json` from the old Library-API
version of this crate is refused with a "run `photos auth` again" error.

## Using it

```sh
filmcraft-cli photos pick --project my.fcproj --save   # pick in the browser → download → import
filmcraft-cli photos pick --max-items 5 --timeout 300   # cap the selection and the wait
filmcraft-cli photos pick --session <id>                # keep waiting on a timed-out session
filmcraft-cli photos cached                             # what earlier picks downloaded
filmcraft-cli photos add <id>… --bridge 127.0.0.1:9876  # import cached items again, no picker
```

`photos pick` opens the picker in your default browser (`--no-browser` just prints the link),
waits for you to press **Done** — 600 s by default, `--timeout` up to 3600 — then downloads
each item and runs `file.import`. `--demo`/`--project`/`--bridge` choose where it imports, as
for every other command. Progress goes to stderr; the result JSON (`items`, `downloaded`,
`cached`, `failed`) to stdout.

Over MCP (`filmcraft-cli mcp --demo` / `--bridge`): `googlephotos_pick` (same flow,
`max_items`, `wait_seconds` default 300, `session` to resume), `googlephotos_cached` and
`googlephotos_add` (re-import cached items). An agent tells the user what to choose:

> "add the beach sunset videos from my Google Photos to the project" → the agent says "pick
> the beach sunset clips in the Google Photos tab that just opened" → `googlephotos_pick
> {max_items: 10}` → `timeline.place` with the imported clips.

If the agent's MCP client has short tool timeouts, it calls `googlephotos_pick
{wait_seconds: 0}` first (opens the picker, returns `status: pending` with a `session` and
`pickerUri`) and then `googlephotos_pick {session: …}` to wait for the result.

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
  `photos auth`. Promote the app to "In production" (still unverified, 100 users) to stop that.

## Layout

- `auth.rs` — the installed-app OAuth flow (browser + loopback redirect on
  `http://127.0.0.1:8417`, waiting at most five minutes and ignoring stray connections), token
  refresh, the `0600` token store, the granted-scope check.
- `api.rs` — the Picker client: sessions (create / get / delete), the polling rule,
  `mediaItems.list` paging, Google's error messages with setup hints.
- `cache.rs` — streamed downloads into a per-item cache; ids and filenames sanitized so
  nothing lands outside it; the bearer token only ever goes to Google's media hosts.
- `lib.rs` — `GooglePhotos::pick`, the whole bounded flow, and the browser opener (https
  links only).

`cargo test -p filmcraft-gphotos` is fully offline: session, polling-interval and media item
JSON parsing (including hostile and truncated input), the polling rule, token store and scope
check, the loopback redirect handling, cache path sanitizing and the download-host check.
