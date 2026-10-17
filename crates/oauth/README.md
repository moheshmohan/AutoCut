# filmcraft-oauth

OAuth 2.0 sign-ins for FilmCraft's desktop integrations, and where their tokens live (layer L5).
Google Photos ([`filmcraft-gphotos`](../gphotos/README.md)) signs in through it; the YouTube
adapter (roadmap B3) is meant to as well.

It follows the native-app profile of OAuth ([RFC 8252](https://www.rfc-editor.org/rfc/rfc8252)):
the authorization-code grant ([RFC 6749](https://www.rfc-editor.org/rfc/rfc6749) §4.1) with
PKCE S256 ([RFC 7636](https://www.rfc-editor.org/rfc/rfc7636)), consent in the system browser,
and the redirect caught on a loopback address.

## Signing in

```rust
let client = OAuthClient {
    provider: Provider::google(),
    client_id, client_secret: Some(secret),
    sign_in_command: "filmcraft-cli photos auth".into(), // named by every "sign in again" error
};
let req = SignInRequest { scopes, extra: vec![("access_type".into(), "offline".into())], wait, page };
let tokens = client.sign_in(&filmcraft_oauth::http_agent(), &req, &mut |url| open_in_browser(url), &mut |m| eprintln!("{m}"))?;
let stored = TokenVault::system(&data_dir, "FilmCraft Google Photos").save(&tokens)?;
```

1. A listener binds `127.0.0.1` on a port the OS picks (`loopback.rs`); the redirect URI is
   `http://127.0.0.1:<port>`. Google's Desktop clients take a loopback redirect on any port
   without registering it.
2. A fresh `state` (24 random bytes) and PKCE verifier (32 random bytes, 43 base64url
   characters) come from the OS's secure random source (`getrandom`); the consent URL carries
   `code_challenge = BASE64URL(SHA256(verifier))` and `code_challenge_method=S256`.
3. The consent page opens in the system browser (`open_in_browser`: `https://` links only);
   if it cannot, the link goes to the caller's progress for the user to open.
4. The listener waits (bounded; at most 64 stray connections such as favicon fetches) for a
   `GET` carrying `state`. A redirect whose `state` is not this sign-in's ends it, and its code
   is never exchanged. The tab gets a plain page: "Signed in to Google Photos — you can close
   this tab and return to FilmCraft" (or what went wrong).
5. The code is exchanged at the token endpoint with the verifier, as a
   `application/x-www-form-urlencoded` POST.

## Refresh, expiry and revocation

- `TokenSet::needs_refresh` is true a minute before the access token runs out.
- `OAuthClient::refresh` merges the answer onto the saved set (the refresh token and scopes
  stay unless the answer has new ones).
- A refused refresh token (`invalid_grant`: Google's Testing-mode sign-ins lapse after 7 days;
  a revoked access or changed password ends them too) is `OAuthError::SignInAgain`, which reads
  "Google no longer accepts the saved sign-in — it expired or was revoked (…); sign in again:
  `filmcraft-cli photos auth`" — never the raw answer. A refused code during a sign-in says to
  try the sign-in again; a refused client (`invalid_client`) says to check the client file.
- `OAuthClient::revoke` ([RFC 7009](https://www.rfc-editor.org/rfc/rfc7009)) ends a grant at
  the provider (`filmcraft-cli photos auth --sign-out`).

Token answers are hostile input: read at most 64 KiB, odd `expires_in` values (negative,
fractional, huge, strings) default or saturate, a non-bearer token is refused, and errors quote
at most 200 characters of the answer with control characters removed. `TokenSet`, `Pkce`,
`OAuthClient` and the gphotos `ClientSecret` print `<redacted>` for secrets in `Debug`.

## Where tokens live

`TokenVault` keeps one integration's tokens, per data directory:

| where | when |
|---|---|
| the macOS Keychain: a generic password named after the integration (`FilmCraft Google Photos`), its account the data directory's path | on a Mac with a default keychain |
| `<data dir>/tokens.json`, owner-only (`0600`), written atomically | elsewhere; with `FILMCRAFT_SECRET_STORE=file`; or when the Keychain cannot take a save (the note says why) |

Beside Keychain-held tokens, `<data dir>/signin.json` notes where they are and the scopes
granted — no token. Readiness checks (asked while the app draws) read it, or `tokens.json`,
and never touch the Keychain.

**Migration.** A `tokens.json` beside a usable Keychain is the newer copy: written by a
FilmCraft from before the Keychain (or by a save the Keychain refused). The first load moves it
into the Keychain, reads the item back, and removes the file only when the copy is identical;
otherwise the file stays and is used, and the note says why. A save to the Keychain removes a
stale `tokens.json` the same way.

**The Keychain through `/usr/bin/security`, not an in-process binding.** The roadmap first put
the Keychain in `crates/platform` behind a binding such as `security-framework` (MIT/Apache-2.0,
already in the lockfile through rustls). That binding works, but a Keychain item trusts the
program that created it by its code signature: FilmCraft's ad-hoc-signed binaries change with
every build, and the CLI, the app and test binaries are different programs, so each new build
would ask for the login password before reading the sign-in — or, with prompts turned off, could
not read it at all (an agent-driven or headless run cannot answer a dialog). Items created by
Apple's `security` tool trust `security`, which Apple signs, so every FilmCraft build reads them
without a prompt (GitHub's `gh` keeps its tokens the same way). It needs no `unsafe` (so nothing
for `crates/platform`, whose ADR keeps it to media FFI) and no new dependency. The trade-off:
any program running as the user could ask `security` for the item without a prompt, as it could
read a `0600` file — but the secret is encrypted at rest with the login keychain, is not in the
data directory (backups, copies, sync), and is unreadable while the keychain is locked.

How it talks to `security`:

- the secret never appears on a command line (other users can list those): it is written to
  `security -i`'s standard input as `add-generic-password -U … -w "b64:<base64url>"`, the
  encoding keeping the line free of quoting; service and account names are checked to be
  printable ASCII without quotes or backslashes (a data directory with other characters gets a
  `sha256:` account name);
- `security -i` cuts lines at 4096 bytes, so a line over 4000 is refused (a token set is well
  under a kilobyte; the vault then keeps it in the file);
- reads are `find-generic-password -s … -a … -w` (exit status 44 is "no such item");
- every call is bounded to two minutes, then killed: a locked keychain whose unlock dialog
  nobody answers is an error ("unlock the Keychain, or set `FILMCRAFT_SECRET_STORE=file` and
  sign in again"), never a hang;
- whether a Keychain is there is asked once per process (`security default-keychain`).

Windows Credential Manager and the Secret Service on Linux are not done; those systems use the
owner-only file.

## Layout

- `lib.rs` — `OAuthError`, the browser opener, the https-only agent, URL coding.
- `pkce.rs` — verifiers, S256 challenges, `state`, base64url (RFC 4648 §5).
- `loopback.rs` — the redirect listener and the page the browser gets.
- `client.rs` — `Provider` (`Provider::google()`), `OAuthClient`: consent URL, `sign_in`,
  code exchange, refresh, revocation, error mapping.
- `token.rs` — `TokenSet` (the stored JSON), token-endpoint answers.
- `store.rs` — the `SecretStore` trait; `FileStore` (owner-only files), `MemoryStore` (tests);
  `system_store` (the Keychain on macOS unless `FILMCRAFT_SECRET_STORE=file`).
- `keychain.rs` — `KeychainStore` over `/usr/bin/security` (any `Tool` in tests), bounded
  process runs, Keychain account names.
- `vault.rs` — `TokenVault`: load (with the migration), save (with the fallback), remove,
  summary.

## Tests

`cargo test -p filmcraft-oauth` is offline and never touches the Keychain:

- RFC 7636 Appendix B (verifier → challenge) and RFC 4648 base64url vectors; verifier rules;
- the loopback listener: strays, non-GET, a mismatched or missing `state`, a declined consent, an
  empty code, silent and garbage connections, the bounded wait, random ports;
- `tests/flow.rs`: a mock token endpoint on localhost and a stand-in browser — the whole
  sign-in (the exchange carries the verifier whose S256 is the URL's challenge, the same
  redirect URI, form encoding), a forged `state` (nothing exchanged), a declined consent, no
  browser, refresh (merging), `invalid_grant` on refresh (the "sign in again" message, no raw
  answer), `invalid_client`, a 500, unreadable and oversized answers, an unreachable endpoint,
  revocation;
- the Keychain store against a fake `security` (no secret in any argument, the one-line
  `-i` input, not-found, failures, damaged items, name checks, the size cap) and bounded runs
  of real processes (timeouts, floods of output);
- the vault with fake stores: the migration (file removed only after an identical read-back;
  kept when the store garbles, forgets or refuses), saves and their fallback, a locked store's
  error, file-only vaults, corrupt files, separate directories, removal.

The real Keychain is opt-in (it writes a throwaway item, removed at the end):

```sh
FILMCRAFT_LIVE_KEYCHAIN=1 cargo test -p filmcraft-oauth --test live_keychain -- --nocapture
```

## For YouTube (B3)

A second integration needs its own `Provider` (Google's, for YouTube), a `SignInRequest` with
its scopes, a `sign_in_command`, and a `TokenVault::system(<its data dir>, "FilmCraft YouTube")`.
Bring-your-own-client-ID mode (the roadmap's answer to Google's verification for the
`youtube.upload` scope) needs nothing more from this crate: the client id and secret are the
caller's. Per-account sign-ins (several channels) would key the vault by account as well as by
directory.
