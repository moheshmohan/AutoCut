# filmcraft-youtube

YouTube for the editor (roadmap B3): connect channels, choose the default, and upload finished
exports to one of them — resumably, with the title, description (and a showcase's chapters),
tags, category, privacy, a custom thumbnail and a playlist — from the CLI, over MCP and in the
app (layer L5, on [`filmcraft-oauth`](../oauth/README.md)).

## One-time setup

YouTube uses the same Google Cloud project and OAuth **Desktop** client as Google Photos
([crates/gphotos](../gphotos/README.md#one-time-setup-your-google-account)):

1. In that project, **APIs & Services → Library** → "YouTube Data API v3" → **Enable**. Without it
   every call fails with "the YouTube Data API v3 is not enabled in your Google Cloud project".
2. The client file is read from `<data dir>/youtube/client_secret.json`, else
   `<data dir>/google-photos/client_secret.json` (the Google Photos one) — nothing to copy.
3. In Testing mode the Google account signing in must be a test user. A Brand Account's channel
   is reached through the Google account that manages it; if Google refuses it at sign-in, add the
   Brand Account's address as a test user too.
4. `filmcraft-cli youtube auth` — a browser opens; pick the channel in Google's account chooser.

The consent asks for `youtube.upload` (upload videos, set their thumbnails) and
`youtube.readonly` (name the channel, read its playlists). `youtube auth --playlists` also asks for
`youtube.force-ssl` ("see, edit and permanently delete your YouTube videos…"): adding a video to a
playlist (`playlistItems.insert`) takes nothing narrower, so it is asked for only when wanted.
Captions upload (which needs the same scope) is not done yet.

## Channels and Brand Accounts

A YouTube sign-in acts as **one channel**: with YouTube scopes, Google's account chooser lists the
Google account and every channel it manages (Brand Accounts — `H4hacks`, say — included), and the
token it returns is that channel's. So:

- **Each sign-in connects one channel.** `youtube auth` again connects another; FilmCraft asks
  `channels.list?part=snippet,status&mine=true` which channel the new token is and keeps it under
  that channel's id. Connecting a channel again replaces its sign-in.
- **Tokens per channel.** Each channel's tokens are a `TokenVault` over
  `<data dir>/youtube/channels/<channel id>/`: on a Mac the Keychain item **FilmCraft YouTube**
  with one account per channel (the account is that directory, so it names the channel id and
  each `FILMCRAFT_DATA_DIR` has its own); elsewhere, or with `FILMCRAFT_SECRET_STORE=file`, an
  owner-only `tokens.json` there. Channels never share or overwrite a sign-in; `signin.json`
  beside each notes the scopes without a token, so listing and readiness never open the Keychain.
- **The list** is `<data dir>/youtube/channels.json` (ids, titles, handles, the scopes granted,
  `longUploadsStatus`, the default) — no secret.
- **The default.** The first channel connected is the default; `youtube auth --default`,
  `youtube default <channel>` or `youtube.setDefault` changes it. An upload goes to `--channel` /
  `channel` when given, else the default (else the only channel). A channel is named by its id
  (`UC…`), its handle (`@H4hacks`, any case) or its title (any case; two with one title must be
  named by id).
- **Disconnecting** (`youtube disconnect <channel>`) revokes that channel's grant at Google (best
  effort: if Google cannot be reached it says so, and how to remove the access by hand) and removes
  its Keychain account, its files and its entry; the next channel becomes the default.
- **Expiry.** In Testing mode Google ends sign-ins after 7 days; the next call says "Google no
  longer accepts the saved sign-in … sign in again: `filmcraft-cli youtube auth` (pick H4hacks
  (@h4hacks) in Google's account chooser)".

## Uploading

```sh
filmcraft-cli youtube upload out/show.mp4 --title "Every Robot I've Met" \
    [--channel @H4hacks] [--description D] [--tags robots,expo] [--category 28] \
    [--privacy private|unlisted|public] [--made-for-kids] [--publish-at 2026-11-01T18:00:00Z] \
    [--playlist "Robot showcases"] [--thumbnail t.jpg] [--no-sidecars] [--no-notify]
filmcraft-cli showcase --report r.json --export out/show.mp4 --upload-youtube [--youtube-channel C] \
    [--youtube-privacy P] [--youtube-playlist P] [--youtube-tags a,b] [--youtube-description D]
```

- **Checked before a byte is sent** (and before any quota is spent): the title (1–100
  characters, no `<` `>`), the description (≤ 5000 bytes, no `<` `>`), the tags (≤ 500
  characters as YouTube counts them: commas, and quotes around tags with spaces), the category (an
  id, or a name like "Science & Technology"; default 22, People & Blogs), the privacy (default
  **private**), `publishAt` (RFC 3339, private only), the file (a video type YouTube takes, ≤ 256
  GB), the thumbnail (JPEG or PNG by its first bytes, ≤ 50 MB), the channel and its sign-in, the
  playlist permission, and — for an MP4/MOV, from its header — a video over 15 minutes on a
  channel whose `longUploadsStatus` is not `allowed`.
- **Showcase sidecars.** `<file>.chapters.txt` beside the video (what `showcase --export` writes)
  is appended to the description after a blank line, when it is chapters (lines of `0:00 Title`,
  the first at 0:00, at least three) and fits in 5000 bytes; `<file>.thumbnail.jpg` becomes the
  thumbnail unless one is given. A file that is not usable is a note in the answer, not a failure.
  `--no-sidecars` (`sidecars: false`) leaves both out. `showcase --upload-youtube` titles the
  upload like the video and passes the build's thumbnail.
- **Privacy.** Uploads are private unless asked otherwise. YouTube locked uploads from Google Cloud
  projects it had not audited to private from July 2020; its `videos.insert` page now says such
  uploads are *not* restricted. Either way the answer reports YouTube's own `privacyStatus`, and
  when YouTube kept a video private that was asked to be unlisted or public, it says so plainly
  ("YouTube kept the video private … change it in YouTube Studio").
- **Shorts.** YouTube has no Shorts endpoint: a vertical or square video of 3 minutes or less
  becomes a Short. The answer's `shorts` says whether this one will (from the MP4/MOV header), or
  why not.
- **Thumbnails** (`thumbnails.set`) need a channel allowed custom thumbnails (a verified one); a
  refusal leaves the video up and says: "… cannot set custom thumbnails yet: YouTube turns them on
  once the channel is verified (https://www.youtube.com/verify) …".
- **Playlists**: the channel's playlist named by id or exact title (any case), found with
  `playlists.list?mine=true` (pages of 50) **before** the upload, so a wrong name leaves no video
  behind; the video is added after (`playlistItems.insert`). A missing playlist is not created.

The answer (JSON): `videoId`, `url`, `studioUrl`, `title`, `channel`, `privacy` (asked),
`privacyStatus` / `uploadStatus` (YouTube's), `bytes`, `resumed`, `restarted`,
`resumedEarlierUpload`, `chapters`, `shorts`, `thumbnail`, `playlist`, `notes`, `quota`.

### How an upload works (the resumable protocol)

1. `POST /upload/youtube/v3/videos?uploadType=resumable&part=snippet,status` with the metadata
   JSON, `X-Upload-Content-Length` and `X-Upload-Content-Type`; `Location` is the session URI.
   It must be on YouTube's upload host (it receives the bearer token): the endpoint's own origin or
   an `https://*.googleapis.com` host, no user info, no port.
2. `PUT <session>` with `Content-Range: bytes a-b/total`, 8 MiB chunks (multiples of 256 KiB,
   streamed from disk, one in memory); `308 Resume Incomplete` with `Range: bytes=0-n` until the
   last chunk's `200/201` and the video resource. A reported range must start at 0 and never pass
   what was sent.
3. A dropped connection, `5xx`, `429` or a rate limit waits (1 s doubling to 32 s, or
   `Retry-After` up to 60 s), asks `PUT` with `Content-Range: bytes */total` where YouTube got to,
   and resumes there. At most 5 failures in a row, 20 in all. A `401` refreshes the token once.
   A `404` means the session expired: the upload starts again from the first byte, once.
4. **Across runs**: the session URI is kept (owner-only) in `<data dir>/youtube/sessions/` under a
   key of (channel, file path, size, modification time, metadata) until the upload ends. A
   cancelled or interrupted upload run again asks where it got to and sends only the rest — no
   second `videos.insert`, no second upload from the day's 100. Kept sessions older than six days
   are not trusted.

## Quota

Google counts quota per Google Cloud project and per **Pacific-time day** (reset at midnight in
California): 10,000 units for most calls, and `videos.insert` in its own bucket of **100 calls a
day** (1 unit each). Costs: `channels.list`, `playlists.list` 1 unit; `thumbnails.set`,
`playlistItems.insert` 50 units. FilmCraft counts what it spends in
`<data dir>/youtube/quota.json` — `youtube quota`, `youtube channels` and every upload's answer
show today's units and uploads, what is left and when it resets. It is this machine's estimate:
other machines and apps on the project spend from the same quota, so it never blocks a call;
Google's answer is authoritative. Errors in words:

| Google says | FilmCraft says |
|---|---|
| `quotaExceeded` | the project's daily quota is used up; it resets at midnight Pacific, in N h M min; more can be asked for in Cloud Console ▸ Quotas (noted in `quota.json`) |
| `uploadLimitExceeded` | the channel has reached YouTube's upload limit for now; try tomorrow or another channel |
| `SERVICE_DISABLED` / `accessNotConfigured` | enable the YouTube Data API v3 in the project |
| `401` after a refresh, `invalid_grant` | sign in again: `filmcraft-cli youtube auth`, pick <channel> |
| `insufficientPermissions` | the permission is missing: `youtube auth` (or `--playlists`) |
| `youtubeSignupRequired`, no channel | the account has no channel: create one, or pick a channel |
| `invalidTitle`, `invalidTags`, `invalidPublishAt`, … | YouTube refused the video's title / tags / … |
| `5xx` after the retries | YouTube could not take it right now; run it again (it resumes) |

## Commands, CLI, MCP, app

- **Engine** (`filmcraft_engine::youtube`, installed by hosts as `Session::youtube.publisher` from
  `engine_publisher()`): `youtube.connect {default?, playlists?, wait?}` (a job: the browser),
  `youtube.channels`, `youtube.setDefault {channel}`, `youtube.disconnect {channel}`,
  `youtube.upload {path, title, channel?, description?, tags?, category?, privacy?, madeForKids?,
  publishAt?, playlist?, thumbnail?, sidecars?, notifySubscribers?, wait?}` (a job with byte
  progress; `jobs.cancel` stops it between chunks; the finished entry's `youtube` is the video).
  Every parameter is checked before a job starts; without a publisher or a channel the commands
  are disabled with the reason.
- **CLI**: `youtube auth [--default] [--playlists] | channels [--json] | default <channel> |
  disconnect <channel> | quota [--json] | upload <file> …`, and `showcase … --upload-youtube`.
- **MCP**: `youtube_upload` (a long call: progress notifications in bytes, cancellation);
  `command_run` for `youtube.connect` (long with `wait: true`), `youtube.channels`,
  `youtube.setDefault`, `youtube.disconnect`.
- **App**: Export mode, under Upload to Google Photos once an export has finished: a channel
  picker (`export.youtube.channel`, its options `export.youtube.channel.option.<i>`), Connect a
  channel (`export.youtube.connect`), Upload to YouTube (`export.uploadYouTube`, private, titled
  like the file; its label is the reason when disabled), the status line
  (`export.youtube.status`) and the link to the video (`export.youtube.open`). The full
  destinations form is Export ▸ Destinations ▸ YouTube / YouTube Short (B5: the channel, title,
  description, tags, privacy, the pre-flight and Publish through `publish.run`; see
  [crates/instagram](../instagram/README.md#destinations-and-the-pre-flight-b5)).

## Layout

- `auth.rs` — the client file (`youtube/`, then `google-photos/`), scopes, the consent request
  (`prompt=select_account consent`: the account chooser every time), granted-scope checks, OAuth
  errors in words.
- `store.rs` — `channels.json`, the per-channel `TokenVault`s, finding a channel by id, handle or
  title, channel-id checks (they are directory names).
- `http.rs` — the `Transport` trait, the ureq transport (pure-Rust TLS, https only, no redirects
  followed — `308` is an answer), capped bodies.
- `api.rs` — Google's errors (`ApiError`, `api_error`), the retry budget, `channels.list`,
  `playlists.list`, `playlistItems.insert`, `thumbnails.set`.
- `upload.rs` — the resumable upload, kept sessions, range and session-URI checks.
- `media.rs` — file, thumbnail and chapters checks, the MP4/MOV header read (filmcraft-isobmff)
  for the Shorts hint and the 15-minute limit.
- `quota.rs` — the day's count, Pacific time (US DST rule) without a time-zone database.
- `lib.rs` — `YouTube` (connect, listing, default, disconnect, preflight, upload) and the
  engine's `EnginePublisher` (readiness from disk, cached two seconds: it is asked while drawing).

## Tests

`cargo test -p filmcraft-youtube` is offline: `tests/flow.rs` runs the whole adapter against a
mock Google on `127.0.0.1` (`tests/mock`: token and revocation endpoints, `channels.list`,
playlists, thumbnails and the resumable protocol with scripted faults) and a stand-in browser that
"picks" a channel: two channels connected with their own sign-ins (a memory secret store: no
Keychain), the default, disconnecting one (its grant revoked, the other's kept); an upload with a
dropped connection mid-chunk and a 503, every byte arriving once and in order, the metadata,
chapters and thumbnail; an interrupted upload resumed by the next run without a second
`videos.insert`; an expired session started again (and twice: an error); quota counting;
`quotaExceeded`, `uploadLimitExceeded`, the API disabled, `invalidTitle`, a refused token, a lapsed
refresh token; the playlist permission and lookup (two pages); a thumbnail refused for an
unverified channel; hostile answers (a session URI elsewhere, ranges past what was sent, odd or
huge video answers, odd channel answers) — errors, never panics; the engine commands over the real
publisher. Unit tests cover the metadata limits, ranges, session URIs, Pacific days (DST
changes), chapters, thumbnails and odd files.

The live check uploads for real (one of the day's 100), so it runs only when asked:
`cargo xtask live --youtube` (or `--only youtube`) connects a channel if none is (the browser
opens), exports 5 s of the demo project and uploads it **private** to the default channel without
notifying subscribers.
