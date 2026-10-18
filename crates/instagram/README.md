# filmcraft-instagram

Instagram for the editor (roadmap B4): connect the user's professional (Creator or Business)
account and publish finished exports as **Reels** — from the CLI, over MCP and in the app's
Export ▸ Destinations (layer L5, on [`filmcraft-oauth`](../oauth/README.md)'s loopback listener
and secret store).

It uses the **Instagram API with Instagram Login** (`graph.instagram.com`, "Business Login for
Instagram"): no Facebook Page is needed, only a professional account and the user's own Meta
app.

## What Meta's documentation says (read 2026-10-17)

| question | the docs | what FilmCraft does |
|---|---|---|
| Can a Reel be uploaded from a local file (resumable upload) with Instagram Login? | The content-publishing guide lists `upload_type=resumable` and `POST https://rupload.facebook.com/ig-api-upload/<container>` with the note "Only for apps that have implemented Facebook Login for Business", and its host table names `rupload.facebook.com` only in the Facebook Login column ([Publish Content](https://developers.facebook.com/docs/instagram-platform/content-publishing/), [Resumable Uploads](https://developers.facebook.com/docs/instagram-platform/content-publishing/resumable-uploads/)). For Instagram Login the documented source is `video_url`: "We cURL media used in publishing attempts, so the media must be hosted on a publicly accessible server". Third-party publishers' 2026 guides do show the resumable flow with Instagram Login tokens on `graph.instagram.com` ([upload-post.com](https://www.upload-post.com/instagram-api/)). | **Direct upload first** (local-first, nothing hosted): a container with `upload_type=resumable` on `graph.instagram.com`, the bytes to `rupload.facebook.com` with `Authorization: OAuth <token>`, `offset`, `file_size`. If Meta refuses it for this app, the error says so and names the documented way: `--video-url https://…` (`videoUrl`), a public address of the same file. `cargo xtask live --instagram` settles it for the user's app without publishing (below). |
| Redirect URIs for business login | `redirect_uri` must "exactly match one of the base URIs in your list of valid OAuth URIs" (App Dashboard ▸ Instagram ▸ API setup with Instagram login ▸ Set up Instagram business login; "we may have added a trailing slash") ([Business Login](https://developers.facebook.com/docs/instagram-platform/instagram-api-with-instagram-login/business-login), [OAuth Authorize](https://developers.facebook.com/docs/instagram-platform/reference/oauth-authorize)). Nothing says https-only or forbids `localhost`, and open-source schedulers register `http://localhost:<port>/…` for development ([Postiz](https://docs.postiz.com/providers/instagram)). No PKCE is documented. The code is valid an hour, once, and comes back with `#_` appended. | The redirect is **fixed** (exact match, port included): `http://localhost:8714/` by default, caught by a listener on that port. If the dashboard will not take an `http` URI, register any `https://` page you control and give it to `instagram setup --redirect-uri`: `instagram auth` then asks you to paste the address the browser ends on. No PKCE is sent (undocumented); `state` is checked. |
| Tokens | The code gives a **short-lived** token (1 hour) at `https://api.instagram.com/oauth/access_token` (form POST with the app secret); `GET graph.instagram.com/access_token?grant_type=ig_exchange_token&client_secret=…` gives a **long-lived** one (60 days, `expires_in` ≈ 5,184,000); `GET graph.instagram.com/refresh_access_token?grant_type=ig_refresh_token` renews one that is at least 24 hours old and not expired, for 60 more days; "Tokens that have not been refreshed in 60 days will expire". Tokens from the dashboard's **Generate token** are long-lived (60 days) ([Get started](https://developers.facebook.com/docs/instagram-platform/instagram-api-with-instagram-login/get-started), [Refresh Access Token](https://developers.facebook.com/docs/instagram-platform/reference/refresh_access_token)). The docs ask for the exchange "in server-side code" because it carries the app secret. | Both paths: the browser login (the user's own app secret, kept in the Keychain on the user's own machine — there is no FilmCraft server) or a pasted dashboard token (no secret on the desktop at all). Long-lived tokens are refreshed once they are a day old, whenever FilmCraft publishes or `instagram refresh` runs, so a used sign-in never lapses. |
| Publishing | `POST /<IG_ID>/media` (`media_type=REELS`, `caption`, `share_to_feed`, `thumb_offset` ms or `cover_url`), `GET /<container>?fields=status_code` (`EXPIRED`, `ERROR`, `FINISHED`, `IN_PROGRESS`, `PUBLISHED`; "once per minute, for no more than 5 minutes"), `POST /<IG_ID>/media_publish` (`creation_id`) ([IG User Media](https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-user/media/), [IG Container](https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-container)). | As documented; processing is polled after 5 s, then doubling to once a minute, 5 minutes at most. |
| Limits | "100 API-published posts within a 24-hour moving period" (the guide); the `content_publishing_limit` reference currently shows 50 ([reference](https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-user/content_publishing_limit)); 400 containers a day. | Counted here (`limits.json`), and Instagram's own `content_publishing_limit` read before every publish: its `quota_total` wins. |
| Reel specs | MOV/MP4, `moov` at the front, no edit lists; H.264/HEVC, progressive, closed GOP, 4:2:0; AAC ≤ 48 kHz mono/stereo, 128 kbps; 23–60 fps; ≤ 1920 px wide; 0.01:1–10:1 (9:16 recommended); ≤ 25 Mbps; 3 s–15 min; ≤ 300 MB. Cover: JPEG ≤ 8 MB, sRGB, 9:16. Caption ≤ 2,200 characters, 30 hashtags, 20 @mentions. | Checked before any byte leaves: here (file type and size, caption, limits) and by the shared pre-flight of the destinations (`publish.preflight`, roadmap B5). |

## One-time setup

The Meta app (**AutoCut**, its Instagram app **AutoCut-IG**) and the account's Instagram Tester
role are made in the Meta App Dashboard. Then, on the Mac:

```sh
# the simplest: a 60-day token from App Dashboard ▸ Instagram ▸ API setup with Instagram login ▸
# Generate access tokens ▸ (your account) Generate token — pasted at the prompt (not shown) or piped
filmcraft-cli instagram token

# or the browser login: the Instagram app ID and secret from the same page; the secret is read
# from standard input (a prompt that does not show it), never from the command line
filmcraft-cli instagram setup --app-id 2133883020551415 [--redirect-uri http://localhost:8714/]
#   …then add exactly that redirect URI under "Set up Instagram business login" ▸ OAuth redirect URIs
filmcraft-cli instagram auth

filmcraft-cli instagram account          # who is connected, when the sign-in runs out
```

- The app id and redirect URI are in `<data dir>/instagram/app.json`; the **app secret** in the
  Keychain item **FilmCraft Instagram app** (an owner-only `app_secret` file where there is no
  Keychain, or with `FILMCRAFT_SECRET_STORE=file`). `instagram setup --forget` removes them.
- The **token** is in the Keychain item **FilmCraft Instagram** (a `TokenVault` over
  `<data dir>/instagram/account/`); `account.json` beside it names the account, how it was
  connected, when the token was refreshed and when it runs out — no secret. Readiness checks
  (asked while the app draws) read only these files.
- Secrets never appear on a command line, in a URL FilmCraft logs, in an error or in the
  repository; transport errors are described without their URL (the two token swaps carry the
  secret and token in the query string, as Meta requires).
- `instagram disconnect` forgets the account here. Instagram has no revocation call for these
  tokens: remove the app in Instagram ▸ Settings ▸ Website permissions ▸ Apps and websites.

## Publishing

```sh
filmcraft-cli instagram publish out/show.mp4 [--caption "Every robot #robots"] \
    [--cover-at 2.5 | --cover-url https://…/cover.jpg] [--no-feed] [--upload-only]
filmcraft-cli instagram publish --video-url https://cdn.example.com/show.mp4 [--caption …]
filmcraft-cli instagram limits           # Instagram's count of posts in the last 24 h
```

1. **Checked first** (no network): the account and its token, the file (MP4/MOV, ≤ 300 MB) or URL
   (public `https://`, not a local or private address), the caption (2,200 characters, 30
   hashtags, 20 @mentions), the cover (`--cover-at` seconds → `thumb_offset`; `--cover-url` a
   public JPEG — Instagram takes covers only from a frame or a public URL), this machine's limit
   counts.
2. **The token** is refreshed if it is a day old (a refresh refused as expired or revoked says
   "connect again"; one failing for a passing reason leaves the current token in use).
3. **The container** (`media_type=REELS`, `upload_type=resumable`, caption, `share_to_feed`,
   cover). The upload URI must be on `rupload.facebook.com` (it receives the token); one
   Instagram names elsewhere is not used.
4. **The upload** streams the file from disk (never whole in memory). A dropped connection, a
   `5xx` or a retriable failure waits (1 s doubling to 30 s), asks how many bytes Instagram holds
   (`video_status.uploading_phase.bytes_transferred`, where it reports it) and sends the rest —
   or the whole file again when it does not say. At most 5 failures in a row, 12 in all.
5. **Across runs** the container is kept in `<data dir>/instagram/sessions/` under a key of
   (account, file, size, modification time, caption, cover, feed) for 23 hours: running the same
   publish again after a cancel, a crash, a processing timeout or `--upload-only` carries on
   with it — no second container, no second upload.
6. **Processing** is awaited at most 5 minutes; `ERROR` says what Instagram did not take (its
   subcode in words, e.g. 2207026 "it wants MP4 or MOV with H.264 or HEVC video and AAC audio"),
   `EXPIRED` that the container lapsed.
7. **The limit** as Instagram counts it (`content_publishing_limit`), then `media_publish`, then
   the permalink. `--upload-only` stops before publishing.

The answer (JSON): `published`, `mediaId`, `permalink`, `container`, `account`, `source`
(`upload` or `url`), `bytes`, `resumed`, `restarted`, `resumedEarlierUpload`,
`processingSeconds`, `notes`, `limits`.

### Errors in words

| Meta says | FilmCraft says |
|---|---|
| code 190 / 102, an expired or revoked token, a refused refresh | the sign-in of @account is no longer accepted; connect again: `instagram auth` (or `instagram token`) |
| code 10, 200–299 | the account has not allowed publishing (and, in development mode, must be an accepted Instagram Tester) |
| `PERSONAL` account | switch to a Creator or Business account |
| code 4, 17, 32, 613, HTTP 429 | the API rate limit, with the wait from `X-App-Usage` / `X-Business-Use-Case-Usage` |
| code 9 / 2207042 | the 24-hour publishing limit; publish again tomorrow (the upload is kept) |
| 2207026, 2207052, 2207057, 2207010, 2207004 | the format, the URL, the cover time, the caption, the cover size — what to change |
| 2207001, 2207003, 2207032, 2207053, `is_transient`, 5xx | retried, then "Instagram could not take it right now" |
| a refused direct upload (`unauthorized user request`) | direct uploads are documented only for Facebook Login apps: publish from `--video-url` |
| client secret / redirect refused at sign-in | check `instagram setup`; the redirect URI must match the dashboard exactly |

## Commands, CLI, MCP, app

- **Engine** (`filmcraft_engine::instagram`, installed by hosts as `Session::instagram.publisher`
  from `engine_publisher()`): `instagram.connect {wait?}` (a job: the browser; a loopback
  redirect only — a pasted one needs the CLI), `instagram.account`, `instagram.limits`,
  `instagram.disconnect`, `instagram.publish {path?, videoUrl?, caption?, shareToFeed?,
  coverSeconds?, coverUrl?, uploadOnly?, wait?}` (a job with byte progress; `jobs.cancel` stops
  it; the finished entry's `instagram` is the Reel). Every parameter is checked before a job
  starts; without a publisher or an account the commands are disabled with the reason.
- **CLI**: `instagram setup | token | auth | account | refresh | limits | disconnect | publish`.
- **MCP**: `instagram_publish` (a long call: progress notifications in bytes, cancellation);
  `command_run` for `instagram.account`, `instagram.limits`, `instagram.connect`.

## Destinations and the pre-flight (B5)

Export mode's **Destinations** column lists Media File, YouTube, YouTube Short and Instagram
Reel, each with a generic glyph drawn in `icons.rs` (a file, a play triangle in a rounded square,
an upright frame, a still camera — no platform logo) and its connect state. Picking a platform
shows its form instead of the Media File settings: the account (Connect), the metadata (title,
description, tags, privacy and channel for YouTube; caption with its counts, grid, cover frame
for Instagram), the file (the last export unless another is given), **Check**, the checklist,
**Export a Fixed Copy** when a copy can mend it, and **Publish** (disabled, with the reason, until
the account is connected and the file passes).

The pre-flight is the engine's (`crates/engine/src/publish.rs`), shared by every destination and
run before any byte leaves:

| check | Instagram Reel | YouTube Short | YouTube |
|---|---|---|---|
| container | MP4 / MOV (error) | — | — |
| video codec, chroma | H.264 / HEVC, 4:2:0 (error) | — | — |
| frame rate | 23–60 fps (error) | over 60: a note | over 60: a note |
| width | ≤ 1920 px (error) | — | — |
| aspect | 9:16 recommended (warning), 0.01:1–10:1 (error) | vertical or square (error) | — |
| bitrate | ≤ 25 Mbps (error) | — | — |
| duration | 3 s–15 min (error) | ≤ 3 min (error) | ≤ 12 h; over 15 min: a note |
| size | ≤ 300 MB (error) | ≤ 256 GB | ≤ 256 GB |
| audio | AAC, ≤ 48 kHz, mono/stereo (error) | — | — |
| `moov` at the front | warning (fast-start fix) | — | — |
| loudness, true peak | −14 LUFS ±2, ≤ −1 dBTP (warnings) | the same | the same |

Loudness is BS.1770 over the first 10 minutes. Every failure and warning carries its fix; where
an export preset can make a passing copy, `publish.fixCopy` writes `<name> (<Destination>).mp4`
beside it: re-encoded (H.264, AAC 48 kHz stereo, scaled to the width limit, 30 or 60 fps, the
bitrate that fits the size, loudness normalized to −14 LUFS with peaks at −1 dBTP, `moov` at the
front for Instagram) when a check needs it, else only the `moov` moved to the front — the same
samples, every chunk offset shifted. A length or a shape no copy can mend is said instead.

- **Engine**: `publish.destinations`, `publish.preflight {destination, path, loudness?}`,
  `publish.fixCopy {destination, path, out?, wait?}` (a job; `publish` in `jobs.list`),
  `publish.run {destination, path, …metadata, wait?}` (the pre-flight, then `youtube.upload` or
  `instagram.publish` as a job with progress and cancel — the MCP supervisor and the CLI chain
  export → publish with it).
- **CLI**: `publish destinations | check <destination> <file> | fix <destination> <file> | run
  <destination> <file> …`; `instagram publish` runs the pre-flight first (`--no-preflight` skips it).
- **MCP**: `publish_run` (a long call), `command_run` for `publish.preflight`, `publish.fixCopy`,
  `publish.destinations`.
- **App automation ids**: `export.destination.<id>`, `export.publish.*` (see
  `crates/ui-egui/src/panels/destinations.rs`).

## Layout

- `auth.rs` — the app (`app.json`, the secret in a `SecretStore`), redirect URIs, the consent URL,
  the pasted address, the token swaps and refresh, token answers of every shape, errors in words.
- `store.rs` — `account.json` and the token's `TokenVault`.
- `http.rs` — the `Transport` trait, the ureq transport (pure-Rust TLS, https only, no redirects
  followed, file bodies streamed with progress and cancel), capped answers.
- `api.rs` — Graph errors (codes, subcodes, usage headers), the retry policy, `/me`, containers,
  status, `media_publish`, `content_publishing_limit`, permalinks.
- `upload.rs` — the resumable upload, its resume, kept containers.
- `limits.rs` — the 24-hour counts.
- `lib.rs` — `Instagram` (setup, connect, token, refresh, listing, limits, preflight, publish) and
  the engine's `EnginePublisher` (readiness from disk, cached two seconds).

## Tests

`cargo test -p filmcraft-instagram` is offline: `tests/flow.rs` runs the adapter against a mock
Instagram on `127.0.0.1` (`tests/mock`) with a memory secret store (no Keychain): setup (the
secret only in the store), the browser sign-in over a loopback port and pasted back (a stale
`state`, a declined consent, a wrong secret), a pasted token (refreshed, or kept when the refresh
is refused; a personal account refused), a publish whose upload drops mid-way and resumes from
Instagram's count (every byte once, in order; OAuth header on the upload, bearer on the Graph
calls, no token in a URL), a cancelled upload resumed by the next run without a second container,
processing `ERROR` / `EXPIRED` / still busy (then finished by the next run without sending again),
`--upload-only` then publish, the limits (Instagram's count, 2207042, an app rate limit, 400
containers), a token refreshed once a day old and one refused, an expired sign-in (no call made),
a `video_url` publish, hostile answers (odd ids, huge bodies, non-JSON, an upload URI elsewhere, a
refused upload, no `video_status`, odd `/me`) — errors, never panics — and the engine commands
over the real publisher.

The live check reaches the user's real account, so it runs only when asked, and never signs in:
`cargo xtask live --instagram` checks the account and Instagram's count (`instagram limits`),
exports 4 s of the demo project at 1080×1920 and runs `instagram publish --upload-only`: the
container is made, the clip uploaded directly and processed — **nothing is published** (the
container lapses within 24 hours; it uses one of the day's 400). That answers whether direct
upload works for this app on Instagram Login. `--instagram-publish` publishes the clip as a Reel
(delete it in the Instagram app afterwards).
