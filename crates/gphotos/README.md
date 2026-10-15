# filmcraft-gphotos

Google Photos for the editor: browse albums, search the library, download originals into a
cache, and import them into the project — from the CLI and over MCP.

## One-time setup (your Google account)

The Photos Library API needs an OAuth client of your own. For a private tool, Google's
"Testing" publishing mode is fine — you are the only user.

1. Open [Google Cloud Console](https://console.cloud.google.com/) → create a project (any name).
2. **APIs & Services → Library** → search **"Google Photos Library API"** → **Enable**.
3. **APIs & Services → OAuth consent screen** → External → fill in an app name and your email
   → add yourself as a **Test user** (Users panel). Publishing stays **Testing**.
4. **APIs & Services → Credentials → Create credentials → OAuth client ID** → application type
   **Desktop app** → create → download the JSON.
5. Save it as `~/Library/Application Support/FilmCraft/google-photos/client_secret.json`
   (or the equivalent data dir on your platform).
6. Run `filmcraft-cli photos auth` — a browser opens, you consent, tokens land next to the
   secret (owner-only permissions). Done.

Only the read-only scope (`photoslibrary.readonly`) is requested; tokens never leave this
machine and are stored `0600`.

## Using it

```sh
filmcraft-cli photos albums                                   # browse
filmcraft-cli photos search beach --category LANDSCAPES --after 2024-06-01
filmcraft-cli photos add AbC123 xyz789 --project my.fcproj    # download (cached) + import
```

Over MCP (`filmcraft-cli mcp --demo` / `--bridge`), same three as tools:
`googlephotos_browse`, `googlephotos_search`, `googlephotos_add` — so an agent can find
b-roll and drop it into the timeline:

> "search my Google Photos for beach sunset videos from last summer and add the best three to
> the project" → `googlephotos_search {category: LANDSCAPES, query: "beach sunset", after:
> "2025-06-01"}` → `googlephotos_add {ids: [...]}` → `timeline.place`.

## Honest limits

- **No semantic search.** Google's API filters by category, date and album; the free-text
  `query` matches filenames locally. Agents compose filters (and filenames often carry the
  words people remember), but "find the one where grandma laughs" is not answerable
  server-side. The local VLM could caption downloaded candidates — future work.
- **Videos download whole** (`=dv`, up to GBs); photos come down as originals (`=d`). Everything
  is cached under `<data dir>/google-photos/cache/<id>_<filename>` — imports are instant the
  second time.
- **Test-mode tokens expire after 7 days** of no use if your Google Cloud app stays in
  Testing; re-running `photos auth` refreshes consent. Promote the app to "In production"
  (still unverified, 100 users) to stop that.
- `file.import` does the editor-side ingestion; the bridge imports into the running app,
  headless modes into the open project.

## Layout

- `auth.rs` — the installed-app OAuth flow (browser + loopback redirect on
  `http://localhost:8417` — the same URI must be on the OAuth client), token refresh, the
  `0600` token store.
- `api.rs` — albums, `mediaItems.search` (Google filters + local filename matching), item
  lookup; categories validated against Google's list.
- `cache.rs` — download originals once, keyed by media item id.

`cargo test -p filmcraft-gphotos` is fully offline (query parsing, token store, category and
date validation, cache paths).
