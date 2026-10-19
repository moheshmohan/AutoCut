//! FilmCraft headless CLI: every engine command is one shell call away, for scripts and AI agents.
//!
//! Run `filmcraft-cli help` for the full reference (also in docs/agents.md).

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

/// `println!` that doesn't panic when stdout is closed (`filmcraft-cli commands | head`) with
/// "failed printing to stdout: Broken pipe (os error 32)". Once stdout has failed, later output is
/// dropped and the work goes on: see `stdout_failed`.
macro_rules! outln {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        if !$crate::stdout_gone() {
            if let Err(e) = writeln!(std::io::stdout(), $($arg)*) {
                $crate::stdout_failed(e);
            }
        }
    }};
}

/// `print!` counterpart of `outln!`.
macro_rules! out {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        if !$crate::stdout_gone() {
            if let Err(e) = write!(std::io::stdout(), $($arg)*) {
                $crate::stdout_failed(e);
            }
        }
    }};
}

mod args;
mod instagram;
mod probe;
mod publish;
mod youtube;

use args::{Args, parse_value};
use filmcraft_automation::BridgeClient;
use filmcraft_engine::Session;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicBool, Ordering};

/// Set once a write to stdout has failed; later output is dropped.
static STDOUT_GONE: AtomicBool = AtomicBool::new(false);
/// Set when stdout failed for a reason other than a closed reader: the exit status becomes 1.
static STDOUT_ERROR: AtomicBool = AtomicBool::new(false);

fn stdout_gone() -> bool {
    STDOUT_GONE.load(Ordering::Relaxed)
}

/// stdout went away. The program doesn't stop there: the rest of `exec`/`run`/`import` (later
/// script lines, `--save`, `--save-as`) still runs and the exit status still tells how it went, so
/// `exec … --save | head -1` keeps the edit. A reader that stopped early (a closed pipe) is not an
/// error, as in ripgrep; any other write error is reported and makes the exit status 1.
fn stdout_failed(e: std::io::Error) {
    if STDOUT_GONE.swap(true, Ordering::Relaxed) || e.kind() == std::io::ErrorKind::BrokenPipe {
        return;
    }
    diag(format_args!("filmcraft-cli: can't write to stdout: {e}"));
    STDOUT_ERROR.store(true, Ordering::Relaxed);
}

const HELP: &str = "\
filmcraft-cli: drive FilmCraft from the shell (headless engine, or the live app with --bridge)

USAGE
  filmcraft-cli [OPTIONS] <SUBCOMMAND> [ARGS]

SUBCOMMANDS
  exec <id> [key=value ...]     run one command; prints its result as JSON
  exec <id> '<json params>'     same, params as one JSON object
  run <script.jsonl | ->        run commands, one {\"id\",\"params\"} per line (# comments ok)
  commands [filter] [--json]    list commands (id, label, shortcut, params)
  describe <id>                 one command as JSON (menu, shortcut, params, enabled now)
  inspect [project|sequence]    project tree or active sequence as JSON (default: both)
  import <file>...              import media into the project
  export <out> [--preset name] [--format f] [--range r] [--start s --end s] [--settings json]
         [--gpu-rendering auto|off]
         [--scale f] [--quality 0-100] [--no-audio] [--queue] [--allow-offline]
                                export the active sequence and wait for it to finish: with an
                                export preset (`export --list-presets`; built-in or the user's), or
                                a format (h264|hevc|prores|dnxhr|apv|mjpeg|mxf-op1a|mxf-opatom|png|tiff|bmp|gif|wav|aiff|aac, guessed
                                from the extension); --range entire|inOut|workArea, or a custom
                                range in seconds; --settings is ExportSettings JSON merged over the
                                preset; --scale renders at a fraction of the frame size (0.5 =
                                half); --quality 0-100 for the formats that take one; --no-audio
                                leaves the sound out; --queue adds to the export queue and runs it
                                instead. Offline media (missing, made offline, unreadable) renders
                                as placeholder frames: the export then lists it in `offlineMedia`,
                                warns on stderr and exits 1, unless --allow-offline accepts it
  export --list-presets [query] list export presets (name, category, format) as JSON
  render --seconds S --out f.png [--scale 0.5] [--allow-offline]
                                render one Program frame to PNG (offline media: as for export)
  probe <media> [--image-sequence]
                                media info as JSON, with MXF / Ogg / BWF details; with
                                --image-sequence <media> is the first numbered still of a sequence
  showcase --album '<Google Photos album>' --prompt '<the video to make>' | --subject S
         [--since YYYY[-MM[-DD]]] [--frames N] [--screen N] [--compare N] [--report-out r.json] [--fresh]
         [any option of `showcase --report` below] [--project p.fcproj --save]
                                the whole video from one Google Photos album — the seed you fill
                                in Google Photos — in one command: the local screen agent opens
                                the album with that exact title in Google's picker and selects
                                all of it (Google lets apps read only albums they made, so the
                                picker is the way in: `photos auth` once, `ollama pull
                                qwen2.5vl:7b`, Screen Recording and Accessibility for this
                                terminal), its videos are downloaded (cached) and imported — a
                                file the project already holds is reused, not imported twice —
                                the subject (and a date and a title, when it names them) is read
                                from --prompt (built-in rules, else AUTOCUT_LLM_MODEL), only the
                                album's clips are analysed (as `analyze`; --report-out keeps the
                                report for a later `showcase --report`), and the show is built,
                                checked, exported and uploaded exactly as with --report (the
                                title: --title, the prompt's, else \"Every <Subject> I've
                                Filmed\"). The album's photos are skipped (not downloaded): a
                                showcase is built from video clips. --since leaves out items
                                created and clips filmed before it. Runs as the
                                `autocut.analyze` job (`album`), so --bridge makes it in the
                                running app. E.g. showcase --album \"Robot showcase\" --prompt
                                \"a video of every robot since 2020\" --voice clone:me --export
                                ~/Movies/robots.mp4 --upload --project robots.fcproj --save
  showcase --report analysis.json [--title T] [--subtitle S] [--voice clone:name] [--mood epic]
         [--bpm 100] [--music track.mp3] [--min-coverage 0.3] [--script-in s.json]
         [--script-out s.json] [--plan] [--stabilize] [--match-color] [--target-minutes N]
         [--no-fill] [--no-track] [--style tech] [--timeline] [--loudness social]
         [--soundfont generaluser-gs|oscillators|auto|<.sf2>] [--embed-captions]
         [--export out.mp4 [--upload [--upload-album <title>]]
         [--upload-youtube [--youtube-channel C] [--youtube-privacy P] [--youtube-playlist P]
         [--youtube-tags a,b] [--youtube-description D]]] [--qa | --no-qa]
                                a narrated 16:9 showcase of every clip the analysis found, oldest
                                first: title, a counter, name/date lower thirds that slide and
                                fade in and out, a slow push-in on every shot (portrait and 4:3
                                clips fill the frame, the crop following their robot with the
                                subject tracker, or a slow tilt across it where it is not found
                                or with --no-track; --no-fill shows them whole over a blurred copy), year cards,
                                dramatic transitions with whooshes, a generated music bed ducked
                                under the voice (cloned voices too), an outro; montage cuts and
                                entries land on the music's beat (the bed's --bpm grid, or with
                                --music your own track: imported, its beats measured
                                (media.beats), cut to length or remixed longer at its beats,
                                levelled and ducked like the bed); --stabilize puts Warp
                                Stabilizer on every shot (handheld footage; each shot is
                                analysed once, about its length in decoding); --match-color
                                matches an entry's other clips to its first full-frame one
                                (lumetri.applyMatch); --target-minutes aims for that length:
                                each entry breathes longer after its words (more and longer
                                shots, only while its robot is in view) or shorter, never padded
                                with out-of-view footage or held frames (the length it came to
                                is `seconds` in the output, next to `target_seconds`); --plan
                                prints the
                                entries and the local model's script without building;
                                --script-in takes a reviewed plan (entries and script, or entries
                                alone for the model to write); a plan's script may hold
                                \"pronounce\" ({\"Zaku II\": \"Zaku Two\"}: respellings the voice
                                reads, whole words) and \"say\" ({\"3\": \"...\", \"outro\": \"...\"}:
                                what the voice says instead of a part), while subtitles show the
                                text as written; the look (crates/autocut/README.md): --style
                                tech (or the plan's \"style\") themes the titles, lower thirds,
                                cards, chips and outro (DIN Condensed / DIN Alternate / Futura,
                                cyan, dark panels; Inter where a font is missing), --timeline (or
                                \"timeline\": true, \"events\") a timeline of the years along the
                                bottom with a marker gliding to each entry's date, the script's
                                \"chips\" spec chips beside a lower third, entries of \"kind\":
                                \"scene\" footage without a robot, the plan's \"transitions\"
                                catalogue transitions with sounds snapped to the beat;
                                --loudness social|voice|broadcast|<LUFS> normalises the mix
                                (audio.normalizeLoudness, -1 dBTP), --soundfont plays the bed on
                                those instruments, --embed-captions embeds the narration's
                                captions (CEA-608/708) in the export;
                                with --export, YouTube chapters go to <out>.chapters.txt and the
                                narration's subtitles to <out>.srt and <out>.vtt (at most 2 lines
                                of 42 characters, 1-7 s a cue; also on the sequence's Subtitles
                                caption track), a 1280x720 YouTube thumbnail to
                                <out>.thumbnail.jpg (the frame where a robot is largest, a
                                well-known one first, under the title in large outlined
                                capitals; composed in a \"<title> — Thumbnail\" sequence kept
                                in the project), and --upload then sends the export to
                                your Google Photos library (as `photos upload`; --upload-album
                                puts it into that album, made when missing, to add the next
                                round's material to by hand; with --report, --album still says
                                the same); --upload-youtube sends it to YouTube (as `youtube
                                upload`: the default channel or --youtube-channel, titled like
                                the video, the chapters appended to the description, the
                                thumbnail set, private unless --youtube-privacy). Spoken lines are
                                cached in the data directory, so a rebuild synthesizes only the
                                lines that changed. QA (on with --export, --qa without one, --no-qa
                                to skip) checks the built show: gaps and overlaps between shots and
                                sounds, black frames outside dips, loudness (-16..-14 LUFS), true
                                peak (<= -1 dBTP), the voice >= 4 dB above what plays between its
                                lines and, when AUTOCUT_ANALYSIS_MODEL answers, the subject in
                                every shot; findings in <out>.qa.json and a numbered contact
                                sheet in <out>.qa.png (and under `qa` in the output)
  analyze '<subject>' [--items 1,2] [--frames N] [--screen N] [--relabel earlier.json] [--compare 150]
         [--out report.json] [--fresh]
                                which video items show <subject> (e.g. robot): a local vision model
                                (AUTOCUT_ANALYSIS_MODEL, default qwen2.5vl:7b) locates it in sampled
                                frames, reads the text on and around it, and names it; with capture
                                dates (media.captureDate), the best moment per clip, oldest first,
                                and an identity per robot (clips of the same one, compared side by
                                side: at most --compare pairs, 0 = by name only), as JSON;
                                --screen N looks at N spread-out frames of a clip first and
                                skips the rest when none shows it (fast over many clips). Each
                                clip's analysis is kept in <data dir>/autocut/analysis-cache, by
                                the file (its Google Photos id, else its size and content), the
                                subject, the model and --frames/--screen: a clip analysed before
                                is not analysed again (`cached` counts them); --fresh analyses
                                every clip again (also for showcase --album)
  shorts <project.fcproj | media file> [--count 3] [--aspect 9:16] [--min-seconds 15]
         [--max-seconds 60] [--motion smooth|responsive] [--topic T] [--title-card]
         [--no-captions] [--analysis-fps 30] [--sequence id] [--export dir [--upload-youtube
         [--youtube-channel C] [--youtube-privacy P] [--youtube-playlist P] [--youtube-tags a,b]]]
                                long video → shorts (autocut.shorts): the transcript (Whisper's,
                                made first when there is none) is cut into sentences, runs of them
                                --min-seconds..--max-seconds long are scored for interest and the
                                strength of their opening (the local model rates the hooks when
                                Ollama has one), and the best --count that do not overlap each
                                become a vertical sequence: reframed to --aspect with the speaker
                                followed (sequence.autoReframe: the subject tracker, Motion
                                keyframes, letterboxed where nobody is found), big outlined
                                captions from the transcript, a title card with --title-card;
                                --export writes each to <dir>/NN-<title>.mp4 with the captions
                                burned in and --upload-youtube sends each to YouTube (private
                                unless --youtube-privacy). A media file is imported first; a
                                project's active sequence (or --sequence) is used. --save keeps
                                the shorts in the --project
  autocut '<prompt>' [--dry-run] [--frames]
                                plan (and run) a natural-language edit with AutoCut: a local
                                model (Ollama) classifies the prompt and deterministic planners
                                turn routine intents into engine commands. Prompts beyond the
                                routine catalog come back with needsSupervisor (nothing runs):
                                run those through an MCP client, which plans them with the
                                engine's own commands. --dry-run shows the validated steps
                                without running them; --frames adds captioned frames to the
                                handed-off document. Config: AUTOCUT_LLM_MODEL,
                                AUTOCUT_VLM_MODEL, AUTOCUT_OLLAMA_HOST,
                                AUTOCUT_CONFIDENCE_THRESHOLD
  photos auth [--upload | --albums | --sign-out]
                                 sign in to Google Photos (browser consent, one time; needs
                                 client_secret.json from Google Cloud — see crates/gphotos/README.md);
                                 --upload also asks to add to your library (for `photos upload`),
                                 --albums that and to see the albums FilmCraft makes (for the
                                 album commands), keeping what you granted before. The sign-in
                                 is kept in the macOS Keychain (elsewhere, or with
                                 FILMCRAFT_SECRET_STORE=file, in an owner-only tokens.json).
                                 --sign-out revokes it at Google and forgets it here
  photos upload <file> [--description D] [--album <title|id>]
                                 upload a file (an export: MP4/MOV/… video up to 10 GB, or a
                                 photo up to 200 MB) to your Google Photos library, resumably in
                                 8 MiB chunks; progress on stderr, the new media item (id, link)
                                 as JSON. --album puts it into that album too: the one FilmCraft
                                 made with exactly that title (or that id), created when missing.
                                 Needs `photos auth --upload` once (`--albums` for --album) and
                                 the Photos Library API enabled in your Google Cloud project
  photos album create <title> [--description D]
                                 make an album in your Google Photos library (titles up to 500
                                 characters; --description becomes a note at its top) and print
                                 its id and link; an album FilmCraft made earlier with exactly
                                 that title is reused, never duplicated. Add your own photos and
                                 videos to it in Google Photos, then `photos pick --album <title>`
  photos album add <title|id> <mediaItemId>...
                                 add media items FilmCraft uploaded (ids from `photos upload`) to
                                 one of its albums, 50 per request (Google adds only items an app
                                 uploaded, and only into albums it made)
  photos albums [--json]         the albums FilmCraft made: title, item count, link (Google shows
                                 apps no other albums, and not the items you add to them by hand)
  photos pick [--auto query]... [--queries file] [--sweep FROM..TO [--month-days D-D]] [--all]
              [--album <title> [--videos-only]] [--since YYYY-MM-DD] [--max-items n]
              [--timeout secs] [--session id] [--no-browser]
                                 open the Google Photos picker in the browser, wait (default
                                 600 s, at most 3600) for your selection, download it into the
                                 cache and import it (a file the project already holds is
                                 reused, not imported twice); --demo/--project/--bridge choose
                                 the target. A timed-out wait prints the session id: --session
                                 resumes it. When the agent drives the picker (below), Chrome's
                                 earlier picker tabs are closed first (macOS), so its tab is
                                 the only one.
                                 With --auto <query>, a local vision agent (Qwen2.5-VL via
                                 Ollama) drives the picker itself: it types the query into the
                                 picker's search, selects matching videos and presses Done
                                 (needs `ollama pull qwen2.5vl:7b`, Screen Recording and
                                 Accessibility for this terminal), then closes the picker tab
                                 (answering Chrome's \"Leave site?\" with Leave); a session it
                                 gave up on is deleted. --all selects every search result (a
                                 date search by the picker's date groups, else first tile, then
                                 Shift+click the last). Many searches — --auto repeated,
                                 --queries <file> (one per line, # comments), or --sweep
                                 2017-10..2025-12 (one \"videos <Month> <Year>\" search per
                                 month; --month-days 10-21 makes it one per day of those days
                                 in each month; a sweep implies --all) — go into ONE picking
                                 session and ONE tab: the agent clears the search box and types
                                 each in turn, the selection grows, Done is pressed once and
                                 everything downloads in one batch. A search with no results is
                                 skipped. Each session takes up to 20 searches and --max-items
                                 items (default and most 2000); searches that would pass it go
                                 to the next session (the picker selects past it but refuses
                                 Done: a date search's days past it are unselected again), one
                                 too big for any session is reported to split (--month-days),
                                 and those still left are written to
                                 <data dir>/google-photos/left-searches.txt for a next run.
                                 --since skips items created before that date (not downloaded).
                                 With --album <title> the same agent picks a whole album
                                 instead — the seed you curate in Google Photos: it searches
                                 the picker for that exact title (the picker has no Albums
                                 view; Return on an album's exact title shows its items),
                                 selects that whole result and presses Done. A search that
                                 shows nothing new, or a result too big for an album (more
                                 than --max-items, or a grid that never ends), selects
                                 nothing. --videos-only downloads only its videos (the album
                                 is selected whole, photos too; how many were left out is
                                 said). Items you add to an album by hand are reachable only
                                 this way (Google shows apps only what they uploaded)
  youtube auth [--default] [--playlists]
                                 connect a YouTube channel: the browser opens Google's consent,
                                 whose account chooser lists your Google account and the channels
                                 it manages (Brand Accounts); each sign-in connects the one picked
                                 (run it again for another channel). Uses the Google Photos OAuth
                                 client (client_secret.json; enable the YouTube Data API v3 in its
                                 project). The first channel is the default; --default makes this
                                 one the default; --playlists also asks to manage the channel's
                                 playlists (for --playlist). Each channel's sign-in is its own
                                 Keychain item account (\"FilmCraft YouTube\")
  youtube channels [--json]      the connected channels (* the default) and today's quota
  youtube default <channel>      uploads go there unless --channel says otherwise (id, @handle or
                                 title)
  youtube disconnect <channel>   revoke the channel's sign-in at Google and forget it here
  youtube quota [--json]         today's YouTube quota as FilmCraft spent it (10,000 units and 100
                                 uploads a Google Cloud project a day, reset at midnight Pacific)
  youtube upload <file> --title T [--channel C] [--description D] [--tags a,b] [--category 28]
         [--privacy private|unlisted|public] [--made-for-kids] [--publish-at 2026-11-01T18:00:00Z]
         [--playlist <title|id>] [--thumbnail f.jpg] [--no-sidecars] [--no-notify]
                                 upload a video to YouTube, resumably in 8 MiB chunks (a dropped
                                 connection resumes; running the same upload again after a crash
                                 or a cancel sends only the rest); progress on stderr, the video
                                 (videoId, url, studioUrl, privacyStatus, shorts hint) as JSON.
                                 Private by default (YouTube may keep an unaudited project's
                                 uploads private whatever is asked). <file>.chapters.txt beside it
                                 (a showcase export's) is appended to the description and
                                 <file>.thumbnail.jpg becomes the thumbnail (--no-sidecars: not).
                                 Vertical or square and up to 3 minutes: YouTube makes it a Short
  instagram setup --app-id <Instagram app ID> [--redirect-uri http://localhost:8714/] [--forget]
                                 set up your Meta app's Instagram business login once: its
                                 Instagram app ID (not the Meta app ID) and the redirect URI you
                                 registered under API setup with Instagram login; the app secret
                                 is read from standard input (piped, or typed at a prompt that
                                 does not show it) into the Keychain (\"FilmCraft Instagram app\")
  instagram auth                 connect your professional (Creator or Business) account: the
                                 browser opens Instagram's login; an http://localhost:<port>/
                                 redirect comes back by itself, an https:// one is pasted back.
                                 The 60-day token goes to the Keychain (\"FilmCraft Instagram\")
                                 and is refreshed once a day while FilmCraft is used
  instagram token                or connect with a token from App Dashboard ▸ Instagram ▸ API
                                 setup with Instagram login ▸ Generate access tokens, read from
                                 standard input (no app secret needed)
  instagram account [--json] | refresh | limits [--json] | disconnect
                                 the connected account and when its sign-in runs out; refresh it
                                 now; Instagram's count of posts in the last 24 h (100 posts and
                                 400 containers a day); forget the account here
  instagram publish <file> [--caption C] [--cover-at SECONDS | --cover-url https://…] [--no-feed]
         [--video-url https://…] [--upload-only]
                                 publish a video as a Reel: a container, the file sent to it with
                                 Meta's resumable upload (a dropped connection resumes; running the
                                 same publish again after a cancel or a crash carries on), its
                                 processing awaited (at most 5 minutes), then published; the Reel
                                 (mediaId, permalink) as JSON. --video-url publishes from a public
                                 address Instagram fetches instead. Caption: 2,200 characters, 30
                                 hashtags, 20 @mentions at most; --no-feed keeps it out of the grid;
                                 --upload-only stops before publishing (the container waits 23 h:
                                 the same command without it publishes without sending again)
  publish destinations           the destinations (Media File, YouTube, YouTube Short, Instagram
                                 Reel) and whether each is connected, as JSON
  publish check <destination> <file> [--no-loudness] [--json]
                                 the pre-flight of a file for youtube, youtubeShort or
                                 instagramReel: container, codecs, 4:2:0, frame rate, width,
                                 aspect, bitrate, duration, size, audio rate and channels, moov
                                 placement, loudness (−14 LUFS) and true peak — each failure with
                                 its fix; exits 1 when one fails
  publish fix <destination> <file> [--out F]
                                 a copy that passes: re-encoded (H.264 + AAC 48 kHz, the size,
                                 rate and bitrate it takes, loudness −14 LUFS / −1 dBTP) when a
                                 check needs it, else only the moov moved to the front
  publish run <destination> <file> [youtube upload / instagram publish options] [--no-preflight]
                                 the pre-flight, then `youtube upload` (youtube, youtubeShort; the
                                 title defaults to the file's name) or `instagram publish`
  photos tidy                    close every Google Photos picker tab left open in Chrome (and
                                 nothing else); a tab that asks \"Leave site?\" is closed once
                                 more, which Chrome takes as Leave (macOS)
  photos cached [--json]         list the Google Photos originals already downloaded
  locks [--json]                 who holds the shared resources (chrome: the browser an agent
                                 drives; gpu: on-device models) — picks, tidy and narration take
                                 them and wait their turn
  locks hold <chrome|gpu> [--minutes M] [--note N]
                                 hold a resource yourself (default 30 min, Ctrl-C to let go):
                                 e.g. before driving Chrome by hand or from another session
  photos add <mediaItemId>...    import already-downloaded items again (ids from `photos cached`)
  template list [query] [--json] the video templates: built-in starters and the user's
                                (<data dir>/Video Templates); --json for everything
  template inspect <id|name|path>
                                one template: slots and their placements, variants, beat grid
  template validate <path>      check a .fcvideo-template folder or .zip: every problem with
                                its field; exit 1 when invalid
  template apply <id|name|path> [--variant 9:16] [--slot <slot>=<file|item id|text>]...
         [--name N] [--save | --save-as p.fcproj]
                                a new sequence from a template: repeat --slot for each medium
                                (`--slot photos=a.jpg --slot photos=b.jpg --slot title=Lisbon`);
                                one undo step. E.g. template apply builtin:photo-beat-slideshow
                                --variant 9:16 --slot photos=~/p1.jpg --slot music=~/song.mp3
  template save <out.fcvideo-template | out.zip> [--name N] [--description D]
                                the active sequence as a template: clips flagged with
                                `exec template.markSlot` become slots
  bench-decode <media> [--frames N]
  mcp                           MCP server on stdio (headless, or --bridge to the live app)
  help                          this text
  --version                     print the version

OPTIONS
  --project <p.fcproj>          open this project first (headless)
  --demo                        open the built-in demo project (headless)
  --save                        save the project back to --project when done
  --save-as <p.fcproj>          save the project to this path when done
  --bridge <127.0.0.1:PORT>     send commands to the running app (`filmcraft --control PORT`)
  --data-dir <dir>              FilmCraft data directory for user export presets (headless;
                                default: the app's data directory)
  --keep-going                  `run`: report failing lines and continue
  --compact                     one-line JSON output

VALUES
  key=value parses value as JSON when it can (3.5, true, [1,2], {\"a\":1}), else as a string:
  `exec timeline.razor seconds=3.5`, `exec file.newBin name=Selects`.
  Times: 254016000000 ticks per second; most commands also take seconds=, frame= or timecode=.

EXIT STATUS
  0 success · 1 a command failed, or export/render drew offline media without --allow-offline
  · 2 usage error
";

/// Writes one diagnostic line to `w`, dropping any write error. `eprintln!` panics (exit 101) when stderr's
/// reader is gone, which would turn a usage error, a failure or a finished export into a crash.
fn write_diag(w: &mut dyn std::io::Write, msg: std::fmt::Arguments) {
    let _ = writeln!(w, "{msg}");
}

fn diag(msg: std::fmt::Arguments) {
    write_diag(&mut std::io::stderr(), msg);
}

fn usage(msg: impl std::fmt::Display) -> ! {
    diag(format_args!("filmcraft-cli: {msg}\nRun `filmcraft-cli help` for usage."));
    std::process::exit(2)
}

fn fail(msg: impl std::fmt::Display) -> ! {
    diag(format_args!("filmcraft-cli: {msg}"));
    std::process::exit(1)
}

/// Where commands go: an in-process session, or the desktop app's control channel.
enum Backend {
    Local(Box<Session>),
    Bridge(BridgeClient),
}

impl Backend {
    fn open(a: &Args) -> Self {
        if let Some(addr) = a.opt("--bridge") {
            if a.opt("--project").is_some() || a.flag("--demo") {
                usage("--bridge drives the running app; open projects there (exec file.open path=…)");
            }
            return Backend::Bridge(BridgeClient::new(addr).unwrap_or_else(|e| usage(e)));
        }
        let mut s = Session::default();
        // user export presets (and other per-user libraries) from the data directory, and the
        // narration cache there (a rebuilt showcase re-reads no line it has read before)
        if let Some(dir) = a.opt("--data-dir").map(std::path::PathBuf::from).or_else(filmcraft_engine::autosave::default_data_dir) {
            s.export_presets.set_dir(&dir);
            s.use_narration_cache(&dir);
        }
        // `photos.upload` (also over MCP): the Google Photos sign-in of `photos auth --upload`
        s.photos = Some(filmcraft_gphotos::engine_uploader());
        // `youtube.*` (also over MCP): the channels connected with `youtube auth`
        s.youtube.publisher = Some(filmcraft_youtube::engine_publisher());
        // `instagram.*` (also over MCP): the account connected with `instagram auth` / `token`
        s.instagram.publisher = Some(filmcraft_instagram::engine_publisher());
        // `autocut.analyze` / `autocut.showcasePlan` / `autocut.showcaseBuild` (also over MCP):
        // showcase jobs with the Ollama models of the AUTOCUT_* environment, which can take
        // their clips from a Google Photos `album` (picked by the screen agent)
        s.autocut.host = Some(filmcraft_autocut::host::engine_host_with_albums(filmcraft_automation::photos::album_source()));
        // `template.install {url}` (also over MCP)
        s.sharing.fetcher = Some(filmcraft_automation::template_fetch::fetcher());
        // `recipe.run` (also over MCP): intent steps planned by AutoCut's planner
        s.recipes.planner = Some(filmcraft_autocut::recipes::planner());
        if let Some(p) = a.opt("--project") {
            if let Err(e) = s.execute("file.open", json!({"path": p})) {
                fail(e);
            }
        } else if a.flag("--demo") {
            let _ = s.execute("file.openDemoProject", json!({}));
        }
        Backend::Local(Box::new(s))
    }

    async fn exec(&mut self, id: &str, params: Value) -> Result<Value, String> {
        match self {
            Backend::Local(s) => s.execute(id, params).map_err(|e| e.to_string()),
            // a blocking export runs as an app job polled by the client (#91, #92)
            Backend::Bridge(b) => b.execute(id, params).await.map_err(|e| e.to_string()),
        }
    }

    /// Save to the --project path (`--save`) or to `--save-as`. Bridge mode saves in the app.
    async fn finish(&mut self, a: &Args) {
        let r = if let Some(p) = a.opt("--save-as") {
            Some(self.exec("file.saveAs", json!({"path": p})).await)
        } else if a.flag("--save") {
            match a.opt("--project") {
                Some(p) => Some(self.exec("file.save", json!({"path": p})).await),
                None => Some(self.exec("file.save", json!({})).await),
            }
        } else {
            None
        };
        if let Some(Err(e)) = r {
            fail(format!("saving: {e}"));
        }
    }
}

/// `shorts` options as `autocut.shorts` parameters (with `wait: true`); a usage error says what
/// is wrong.
fn shorts_params(a: &Args) -> Result<Value, String> {
    let num = |k: &str| -> Result<Option<f64>, String> {
        a.opt(k).map(|v| v.parse::<f64>().ok().filter(|f| f.is_finite()).ok_or_else(|| format!("{k} wants a number, not `{v}`"))).transpose()
    };
    let mut p = json!({"wait": true, "count": 3});
    if let Some(n) = a.opt("--count") {
        let n: u64 = n.parse().ok().filter(|n| (1..=20).contains(n)).ok_or_else(|| format!("--count wants 1 to 20, not `{n}`"))?;
        p["count"] = json!(n);
    }
    for (flag, key) in [("--min-seconds", "minSeconds"), ("--max-seconds", "maxSeconds"), ("--analysis-fps", "analysisFps")] {
        if let Some(v) = num(flag)? {
            p[key] = json!(v);
        }
    }
    for (flag, key) in [("--aspect", "aspect"), ("--motion", "motion"), ("--topic", "topic")] {
        if let Some(v) = a.opt(flag) {
            p[key] = json!(v);
        }
    }
    if let Some(q) = a.opt("--sequence") {
        p["sequence"] = json!(q.parse::<u64>().map_err(|_| format!("--sequence wants a sequence id, not `{q}`"))?);
    }
    p["captions"] = json!(!a.flag("--no-captions"));
    p["title"] = json!(a.flag("--title-card"));
    if let Some(d) = a.opt("--export") {
        p["export"] = json!(absolute(d));
    }
    if a.flag("--upload-youtube") {
        if a.opt("--export").is_none() {
            return Err("--upload-youtube sends the exports to YouTube; add --export <dir>".into());
        }
        p["uploadYoutube"] = json!(true);
        let mut y = serde_json::Map::new();
        for (flag, key) in
            [("--youtube-channel", "channel"), ("--youtube-privacy", "privacy"), ("--youtube-playlist", "playlist"), ("--youtube-description", "description")]
        {
            if let Some(v) = a.opt(flag) {
                y.insert(key.into(), json!(v));
            }
        }
        if let Some(t) = a.opt("--youtube-tags") {
            y.insert("tags".into(), json!(t.split(',').map(str::trim).filter(|t| !t.is_empty()).collect::<Vec<_>>()));
        }
        p["youtube"] = Value::Object(y);
    }
    Ok(p)
}

fn print(a: &Args, v: &Value) {
    let s = if a.flag("--compact") { serde_json::to_string(v) } else { serde_json::to_string_pretty(v) };
    outln!("{}", s.unwrap_or_default());
}

/// QA of a built showcase (`showcase --qa`, on with `--export`): `model` grounds the subject.
async fn showcase_qa(
    ex: &dyn filmcraft_autocut::Executor,
    opts: &filmcraft_autocut::showcase::Options,
    built: &filmcraft_autocut::showcase::Built,
    model: Option<&filmcraft_autocut::ollama::Ollama>,
    progress: &mut dyn FnMut(&str),
) -> filmcraft_autocut::qa::QaReport {
    let q = filmcraft_autocut::qa::Options::for_showcase(opts, built);
    filmcraft_autocut::qa::check(ex, &q, model.map(|m| m as &dyn filmcraft_autocut::llm::ChatModel), progress).await
}

/// Export format from the output file's extension.
fn format_for(path: &str) -> Option<&'static str> {
    let ext = path.rsplit_once('.')?.1.to_ascii_lowercase();
    Some(match ext.as_str() {
        "mp4" | "m4v" => "h264",
        "mov" => "prores",
        "mxf" => "mxf-op1a",
        "png" => "png",
        "gif" => "gif",
        "wav" => "wav",
        "tif" | "tiff" => "tiff",
        "bmp" => "bmp",
        "aif" | "aiff" => "aiff",
        "m4a" | "aac" => "aac",
        _ => return None,
    })
}

/// OS hardware video decoders (VideoToolbox on macOS, Media Foundation on Windows) in front of our
/// own, as in the desktop app. Also what `mcp` and the headless commands decode with.
/// The GPU export frame renderer (filmcraft-gpu's off-screen compositor behind filmcraft-export's
/// frame-renderer hook): exports with GPU rendering Auto composite on the GPU and fall back to the
/// CPU reference renderer wherever it cannot.
struct GpuFrameRenderer(filmcraft_gpu::ExportRenderer);

impl filmcraft_export::FrameRenderer for GpuFrameRenderer {
    fn render(
        &mut self,
        project: &filmcraft_project::Project,
        seq: filmcraft_project::ItemId,
        t: filmcraft_time::Tick,
        opts: filmcraft_render::RenderOptions,
        sources: &dyn filmcraft_render::SourceProvider,
    ) -> Option<filmcraft_render::Image> {
        self.0.render(project, seq, t, opts, sources)
    }
}

fn register_gpu_frame_renderer() {
    filmcraft_export::register_frame_renderer(|| {
        filmcraft_gpu::ExportRenderer::new().map(|r| Box::new(GpuFrameRenderer(r)) as Box<dyn filmcraft_export::FrameRenderer>)
    });
}

fn register_hardware_decoders() -> filmcraft_platform::Availability {
    filmcraft_platform::register()
}

/// `path` made absolute against the working directory, as the project stores it. Unlike
/// `canonicalize`, `std::path::absolute` keeps symlinks and adds no Windows verbatim `\\?\`
/// prefix (#850). A path it cannot resolve is kept as given.
/// A template id or name as it is; a path made absolute.
fn template_key(k: &str) -> String {
    let lower = k.to_ascii_lowercase();
    if k.contains('/') || k.contains('\\') || k.starts_with('~') || lower.ends_with(".zip") || lower.ends_with(".fcvideo-template") || lower.ends_with(".json")
    {
        absolute(&filmcraft_engine::export_tools::expand_home(k))
    } else {
        k.to_string()
    }
}

fn absolute(path: &str) -> String {
    std::path::absolute(path).ok().and_then(|p| p.into_os_string().into_string().ok()).unwrap_or_else(|| path.to_string())
}

/// `flag <title|id>` (`--album`, `--upload-album`): absent, or a usable album title (a usage
/// error otherwise).
/// A finished pick, with what its sessions reported when it took many searches.
struct Picks {
    picked: filmcraft_gphotos::Picked,
    /// `sessions`, `left`, `stopped` of a many-search pick (empty for one session).
    report: Value,
    /// A many-search pick stopped early.
    stopped: bool,
}

/// The files of a pick to import, what to print of each item (no short-lived download link),
/// and how many came from the cache.
fn picked_json(picked: &filmcraft_gphotos::Picked) -> (Vec<String>, Vec<Value>, usize) {
    let paths = picked.fetched.iter().map(|f| f.path.to_string_lossy().into_owned()).collect();
    let items = picked
        .fetched
        .iter()
        .map(|f| {
            let mut v = f.item.summary();
            if let Some(o) = v.as_object_mut() {
                o.insert("path".into(), json!(f.path.to_string_lossy()));
            }
            v
        })
        .collect();
    (paths, items, picked.fetched.iter().filter(|f| f.was_cached).count())
}

/// The longest `--queries` file read.
const MAX_QUERIES_FILE: u64 = 1 << 20;

/// The searches of `photos pick`: every `--auto` in order, then the lines of `--queries <file>`
/// (blank lines and `#` comments skipped), then the months of `--sweep` (or with
/// `--month-days` their days) — cleaned: no blanks, no repeats in any case.
fn pick_searches_from(a: &Args) -> Result<Vec<String>, String> {
    let mut qs: Vec<String> = a.all("--auto").into_iter().map(str::to_string).collect();
    if let Some(path) = a.opt("--queries") {
        qs.extend(read_queries(path)?);
    }
    match (a.opt("--sweep"), a.opt("--month-days")) {
        (Some(spec), days) => {
            let (from, to) = filmcraft_gphotos::sweep::parse_months(spec).map_err(|e| format!("--sweep: {e}"))?;
            let days = days.map(filmcraft_gphotos::sweep::parse_days).transpose().map_err(|e| format!("--month-days: {e}"))?;
            qs.extend(filmcraft_gphotos::sweep::queries(from, to, days).map_err(|e| format!("--sweep: {e}"))?);
        }
        (None, Some(_)) => return Err("--month-days goes with --sweep".into()),
        (None, None) => {}
    }
    if qs.len() > filmcraft_gphotos::sweep::MAX_QUERIES {
        return Err(format!("{} searches is more than one run takes ({}); split them", qs.len(), filmcraft_gphotos::sweep::MAX_QUERIES));
    }
    Ok(filmcraft_screenagent::agent::clean_queries(qs))
}

/// One search per line of `path`; blank lines and `#` comments are skipped.
fn read_queries(path: &str) -> Result<Vec<String>, String> {
    let size = std::fs::metadata(path).map_err(|e| format!("--queries {path}: {e}"))?.len();
    if size > MAX_QUERIES_FILE {
        return Err(format!("--queries {path}: {size} bytes is more than a list of searches ({MAX_QUERIES_FILE} at most)"));
    }
    let text = std::fs::read_to_string(path).map_err(|e| format!("--queries {path}: {e}"))?;
    Ok(text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')).map(str::to_string).collect())
}

/// `photos tidy`: close the Google Photos picker tabs left open in Chrome, answering "Leave
/// site?" with Leave when the grounding model is there to see it.
#[cfg(target_os = "macos")]
fn tidy_picker_tabs() -> Result<filmcraft_screenagent::tabs::Tidied, String> {
    let eyes = filmcraft_screenagent::ground::QwenGrounder::from_config(&filmcraft_screenagent::Config::from_env());
    let have_eyes = eyes.model_available().unwrap_or(false);
    let mut hands = filmcraft_screenagent::act::MacActor::new(filmcraft_screenagent::tabs::CHROME, Box::new(filmcraft_screenagent::capture::OsShell));
    let mut pause = |ms: u64| std::thread::sleep(std::time::Duration::from_millis(ms.min(10_000)));
    let leave = have_eyes.then_some(&eyes as &dyn filmcraft_screenagent::ground::Ground);
    filmcraft_screenagent::tabs::tidy(&filmcraft_screenagent::capture::OsShell, &mut hands, leave, &mut pause, &mut |l| diag(format_args!("{l}")))
        .map_err(|e| e.to_string())
}

#[cfg(not(target_os = "macos"))]
fn tidy_picker_tabs() -> Result<filmcraft_screenagent::tabs::Tidied, String> {
    Err("closing picker tabs works with Chrome on macOS; close Google Photos picker tabs in your browser".into())
}

fn album_opt(a: &Args, cmd: &str, flag: &str) -> Option<String> {
    if !a.flag(flag) {
        return None;
    }
    let Some(t) = a.opt(flag).map(str::trim).filter(|t| !t.is_empty()) else {
        usage(format!("{cmd}: {flag} wants an album title (or id)"));
    };
    if let Err(e) = filmcraft_gphotos::albums::clean_title(t) {
        usage(format!("{cmd}: {flag}: {e}"));
    }
    Some(t.to_string())
}

/// Where `showcase` takes its clips from.
enum ShowcaseSource {
    /// An earlier analysis (`--report`, as `analyze --out` writes it).
    Report(String),
    /// A Google Photos album (`--album`): the `autocut.analyze` job's parameters.
    Album { title: String, params: Value },
}

/// `showcase`'s source of clips and the album its upload goes into, from its arguments. Every
/// usage error (exit 2) is here, before a sign-in, a model or a file is touched.
fn showcase_source(a: &Args) -> (ShowcaseSource, Option<String>) {
    let report = a.opt("--report").map(str::to_string);
    if a.flag("--report") && report.is_none() {
        usage("showcase: --report wants the analysis file (analyze --out)");
    }
    // with --report and --upload, `--album` is the album the upload goes into: its spelling
    // before --upload-album, kept working (--album now reads an album's clips instead)
    let legacy = report.is_some() && a.flag("--album") && a.flag("--upload") && !a.flag("--upload-album");
    let upload_album = album_opt(a, "showcase", if legacy { "--album" } else { "--upload-album" });
    if legacy {
        diag(format_args!(
            "showcase: --album with --report names the album the upload goes into; say --upload-album (--album <title> now reads the clips from a Google Photos album)"
        ));
    }
    if upload_album.is_some() && !a.flag("--upload") {
        usage("showcase: --upload-album names the album the upload goes into; add --upload (and --export out.mp4)");
    }
    let album = if legacy { None } else { album_opt(a, "showcase", "--album") };
    if album.is_some() && album == upload_album {
        diag(format_args!(
            "showcase: warning: the export goes into the album the clips come from, so the next `showcase --album` picks this video too; give --upload-album another album to keep them apart"
        ));
    }
    match (report, album) {
        (Some(_), Some(_)) => usage(
            "showcase: --album reads the clips from a Google Photos album, --report from an earlier analysis; give one of them (--upload-album names the album the upload goes into)",
        ),
        (None, None) => {
            usage("showcase --report analysis.json | --album '<Google Photos album>' --prompt '<the video>' [--title T] [--voice clone:name] [--plan] …")
        }
        (Some(r), None) => {
            for k in ["--prompt", "--subject", "--since", "--frames", "--screen", "--compare", "--report-out", "--with-photos", "--fresh"] {
                if a.flag(k) {
                    usage(format!("showcase: {k} goes with --album (a --report already says what it found)"));
                }
            }
            (ShowcaseSource::Report(r), upload_album)
        }
        (None, Some(title)) => {
            let text = |k: &str| -> Option<String> {
                if !a.flag(k) {
                    return None;
                }
                match a.opt(k).map(str::trim).filter(|t| !t.is_empty()) {
                    Some(t) => Some(t.to_string()),
                    None => usage(format!("showcase: {k} wants a value")),
                }
            };
            let (subject, prompt, since) = (text("--subject"), text("--prompt"), text("--since"));
            if subject.is_none() && prompt.is_none() {
                usage("showcase --album: say what the video is of: --prompt 'a video of every robot since 2020', or --subject robot");
            }
            if a.flag("--with-photos") {
                usage("showcase --album: a showcase is built from video clips, so the album's photos are skipped (not downloaded); there is no --with-photos");
            }
            let mut params = json!({"album": title, "wait": false});
            if let Some(t) = subject {
                params["subject"] = json!(t);
            }
            if let Some(t) = prompt {
                params["prompt"] = json!(t);
            }
            if let Some(d) = since {
                if let Err(e) = filmcraft_engine::autocut::parse_since(&json!(d)) {
                    usage(format!("showcase: --since: {e}"));
                }
                params["since"] = json!(d);
            }
            for (k, key) in [("--frames", "frames"), ("--screen", "screen"), ("--compare", "compare")] {
                if let Some(v) = text(k) {
                    let n: u64 = v.parse().unwrap_or_else(|_| usage(format!("showcase: {k} wants a whole number, not `{v}`")));
                    params[key] = json!(n);
                }
            }
            // analyse again the clips the analysis cache knows
            if a.flag("--fresh") {
                params["fresh"] = json!(true);
            }
            (ShowcaseSource::Album { title, params }, upload_album)
        }
    }
}

/// `showcase --album`: the engine's `autocut.analyze` job with `album`, on `b` — the whole
/// album picked through Google's picker (the screen agent selects it), its videos imported
/// without duplicating what the project holds, the subject read from `--prompt`, only those
/// clips analysed — with its progress on stderr. The report it kept, and its summary.
async fn album_analysis(b: &mut Backend, params: Value) -> Result<(Value, Value), String> {
    let start = b.exec("autocut.analyze", params).await?;
    let job = start["job"].as_u64().ok_or_else(|| format!("autocut.analyze started no job: {start}"))?;
    let mut last = String::new();
    let entry = loop {
        if let Backend::Local(s) = b {
            // nothing else runs the job's commands in this process: run them as the app's UI does
            s.pump_port(std::time::Duration::from_millis(50));
        }
        let jobs = b.exec("jobs.list", json!({})).await?;
        let j = jobs.as_array().and_then(|l| l.iter().find(|j| j["id"].as_u64() == Some(job))).cloned().ok_or_else(|| format!("job {job} is gone"))?;
        // (a failed job's status is its error, which the caller reports)
        let failed = j["result"]["error"].is_string();
        let status = j["status"].as_str().unwrap_or_default();
        if !failed && !status.is_empty() && status != last {
            diag(format_args!("showcase: {status}"));
            status.clone_into(&mut last);
        }
        if j["finished"].as_bool() == Some(true) {
            break j;
        }
        let pause = if matches!(b, Backend::Local(_)) { 10 } else { 250 };
        tokio::time::sleep(std::time::Duration::from_millis(pause)).await;
    };
    if let Some(e) = entry["result"]["error"].as_str() {
        return Err(e.to_string());
    }
    let kept = b.exec("autocut.showcaseInspect", json!({"report": true})).await?;
    let report = kept["report"].clone();
    if !report["clips"].is_array() {
        return Err("the analysis kept no report".into());
    }
    Ok((report, entry["showcase"]["summary"].clone()))
}

/// Whether the saved Google Photos sign-in can do what is asked, from disk (no network): uploads,
/// or uploads into albums.
fn photos_readiness(album: bool) -> Result<(), String> {
    let dir = filmcraft_gphotos::data_dir();
    if album { filmcraft_gphotos::album_readiness(&dir) } else { filmcraft_gphotos::upload_readiness(&dir) }
}

/// Upload `path` to the user's Google Photos library (`photos upload`, `showcase --upload`),
/// into `album` too when given (FilmCraft's album of that title or id, created when missing):
/// progress on stderr (stdout may be a protocol stream), the new media item as JSON.
async fn upload_to_photos(path: String, description: Option<String>, album: Option<String>) -> Result<Value, String> {
    tokio::task::spawn_blocking(move || {
        // the sign-in and its permissions from disk first: no token refresh just to refuse
        photos_readiness(album.is_some())?;
        let gp = filmcraft_gphotos::GooglePhotos::configured().map_err(|e| e.to_string())?;
        let opts = filmcraft_gphotos::upload::UploadOptions { description, album, ..Default::default() };
        let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
        // a line per status change, and at most two a second while the bytes go
        let mut last: Option<(u64, String, std::time::Instant)> = None;
        let mut progress = |sent: u64, total: u64, status: &str| {
            let pct = sent.saturating_mul(100).checked_div(total).unwrap_or(0);
            let show = last.as_ref().is_none_or(|(p, s, at)| s != status || (*p != pct && (pct == 100 || at.elapsed().as_millis() >= 500)));
            if show {
                if total == 0 {
                    diag(format_args!("photos upload: {status}"));
                } else {
                    diag(format_args!("photos upload: {pct:>3}% ({:.1} of {:.1} MiB) {status}", mib(sent), mib(total)));
                }
                last = Some((pct, status.to_string(), std::time::Instant::now()));
            }
            true
        };
        gp.upload(std::path::Path::new(&path), &opts, &mut progress).map(|u| u.to_json()).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("internal error: {e}"))?
}

/// Run album calls (`photos album …`, `photos albums`) off the async runtime, once the saved
/// sign-in is known to have the album permissions.
async fn with_albums<T: Send + 'static>(
    f: impl FnOnce(&filmcraft_gphotos::albums::Albums<'_>) -> Result<T, filmcraft_gphotos::GPhotosError> + Send + 'static,
) -> Result<T, String> {
    tokio::task::spawn_blocking(move || {
        photos_readiness(true)?;
        let gp = filmcraft_gphotos::GooglePhotos::configured().map_err(|e| e.to_string())?;
        gp.albums(f).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("internal error: {e}"))?
}

#[tokio::main]
async fn main() {
    cli().await;
    if STDOUT_ERROR.load(Ordering::Relaxed) {
        std::process::exit(1);
    }
}

async fn cli() {
    // `std::env::args` panics on an argument that is not Unicode (a Linux file name in Latin-1)
    let a = Args::parse(std::env::args_os().skip(1).map(|a| a.to_string_lossy().into_owned()));
    // `--help` anywhere (`filmcraft-cli --help`, `filmcraft-cli export --help`) prints the reference:
    // the parser takes it as a flag option, so it never reaches the subcommand match.
    if a.flag("--help") {
        out!("{HELP}");
        return;
    }
    // `--version` anywhere and `-V` as the subcommand, also after global options (`--demo --version`, #710).
    if a.wants_version() {
        outln!("filmcraft-cli {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    register_hardware_decoders();
    register_gpu_frame_renderer();
    let Some(cmd) = a.pos(0) else { usage("missing subcommand") };
    match cmd {
        "help" | "-h" => out!("{HELP}"),
        "version" => outln!("filmcraft-cli {}", env!("CARGO_PKG_VERSION")),
        "probe" => {
            let path = a.pos(1).unwrap_or_else(|| usage("probe <media> [--image-sequence]"));
            match probe::probe(path, a.flag("--image-sequence")) {
                Ok(v) => print(&a, &v),
                Err(e) => fail(e),
            }
        }
        "bench-decode" => {
            let path = a.pos(1).unwrap_or_else(|| usage("bench-decode <media>"));
            let n: i64 = a.opt("--frames").and_then(|v| v.parse().ok()).unwrap_or(120);
            let src = filmcraft_codecs::open_bytes(path, std::fs::read(path).unwrap_or_else(|e| fail(e)).into()).unwrap_or_else(|e| fail(e));
            let rate = src.info().frame_rate();
            let t0 = std::time::Instant::now();
            for f in 0..n {
                src.video_frame(filmcraft_media::FrameRequest::full(rate.tick_of(f))).unwrap_or_else(|e| fail(e));
            }
            let dt = t0.elapsed().as_secs_f64();
            outln!("{n} frames in {dt:.2}s → {:.1} fps (sequential, via Mp4Source)", n as f64 / dt);
            let t1 = std::time::Instant::now();
            let f = src.video_frame(filmcraft_media::FrameRequest::full(rate.tick_of(n / 2))).unwrap_or_else(|e| fail(e));
            let (w, h, _) = f.to_linear_f32_decimated(2);
            outln!("½-res linear conversion {w}x{h}: {:.1} ms", t1.elapsed().as_secs_f64() * 1000.0);
        }
        "locks" => match a.pos(1) {
            None => {
                let held: Vec<Value> = filmcraft_locks::Resource::ALL
                    .iter()
                    .map(|r| {
                        let h = filmcraft_locks::holder(*r);
                        json!({"resource": r.id(), "free": h.is_none(), "holder": h.as_ref().map(|h| json!({"pid": h.pid, "since": h.since, "what": h.what}))})
                    })
                    .collect();
                if a.flag("--json") {
                    outln!("{}", json!(held));
                } else {
                    for r in filmcraft_locks::Resource::ALL {
                        match filmcraft_locks::holder(r) {
                            Some(h) => outln!("{:7} busy: {h}", r.id()),
                            None => outln!("{:7} free", r.id()),
                        }
                    }
                }
            }
            Some("hold") => {
                let r = a.pos(2).and_then(filmcraft_locks::Resource::from_id).unwrap_or_else(|| usage("locks hold <chrome|gpu> [--minutes M] [--note N]"));
                let minutes = a
                    .opt("--minutes")
                    .map(|m| {
                        m.parse::<f64>()
                            .ok()
                            .filter(|m| m.is_finite() && *m > 0.0 && *m <= 24.0 * 60.0)
                            .unwrap_or_else(|| usage("locks hold: --minutes wants a number of minutes up to a day"))
                    })
                    .unwrap_or(30.0);
                let note = a.opt("--note").unwrap_or("held by hand (filmcraft-cli locks hold)").to_string();
                let _guard = filmcraft_locks::acquire(r, &note, std::time::Duration::from_secs(600), &mut |e| diag(format_args!("locks hold: waiting: {e}")))
                    .unwrap_or_else(|e| fail(format!("locks hold: {e}")));
                diag(format_args!("locks hold: holding {} for {minutes} min ({note}); Ctrl-C lets go", r.label()));
                tokio::time::sleep(std::time::Duration::from_secs_f64(minutes * 60.0)).await;
            }
            Some(other) => usage(format!("locks: unknown `{other}` (locks [--json] | locks hold <chrome|gpu>)")),
        },
        "photos" => {
            let what = a.pos(1).unwrap_or_else(|| usage("photos auth | pick | tidy | upload <file> | album create|add | albums | cached | add <id>..."));
            match what {
                "auth" => {
                    use filmcraft_gphotos::auth::Access;
                    let dir = filmcraft_gphotos::data_dir();
                    if a.flag("--sign-out") {
                        if a.flag("--upload") || a.flag("--albums") {
                            usage("photos auth: --sign-out goes alone");
                        }
                        match filmcraft_gphotos::auth::sign_out(&filmcraft_gphotos::api::http_agent(), &dir) {
                            Ok(out) if !out.was_signed_in => outln!("not signed in to Google Photos; nothing to sign out"),
                            Ok(out) => {
                                if let Err(e) = &out.revoked {
                                    diag(format_args!("photos auth: Google did not confirm the revocation: {e}"));
                                }
                                outln!("signed out of Google Photos{}", if out.revoked.is_ok() { " (revoked at Google)" } else { " here" });
                            }
                            Err(e) => fail(format!("photos auth --sign-out: {e}")),
                        }
                        return;
                    }
                    // --upload: also the permission to add to the library; --albums: that and
                    // reading back the albums FilmCraft made (incremental consent)
                    let access = if a.flag("--albums") {
                        Access::PickUploadAndAlbums
                    } else if a.flag("--upload") {
                        Access::PickAndUpload
                    } else {
                        Access::Pick
                    };
                    match filmcraft_gphotos::auth::ClientSecret::load(&dir) {
                        Ok(secret) => match filmcraft_gphotos::auth::consent(&secret, &dir, access) {
                            Ok((t, stored)) => {
                                if let Some(note) = &stored.note {
                                    diag(format_args!("photos auth: {note}"));
                                }
                                outln!(
                                    "signed in to Google Photos ({}); tokens in {}",
                                    match Access::for_grants(&t) {
                                        Access::PickUploadAndAlbums => "pick, upload and albums",
                                        Access::PickAndUpload => "pick and upload",
                                        Access::Pick => "pick",
                                    },
                                    stored.place
                                )
                            }
                            Err(e) => fail(format!("photos auth: {e}")),
                        },
                        Err(e) => {
                            diag(format_args!("photos auth: {e}"));
                            std::fs::create_dir_all(&dir).ok();
                            outln!("put client_secret.json in {}", dir.to_string_lossy());
                        }
                    }
                }
                "pick" => {
                    let parse_num = |k: &str| -> Option<u64> {
                        a.opt(k).map(|v| v.parse::<u64>().unwrap_or_else(|_| usage(format!("photos pick: {k} wants a whole number, not `{v}`"))))
                    };
                    // --auto (repeatable), --queries <file>, --sweep: the searches the agent types,
                    // many of them into one picking session and one tab
                    let searches = pick_searches_from(&a).unwrap_or_else(|e| usage(format!("photos pick: {e}")));
                    let searching = a.flag("--auto") || a.opt("--queries").is_some() || a.opt("--sweep").is_some();
                    let sweep = a.opt("--sweep").is_some();
                    // --album: the agent opens that album in the picker and picks all of it
                    let album = album_opt(&a, "photos pick", "--album");
                    if searching && album.is_some() {
                        usage("photos pick: --auto/--queries/--sweep search, --album picks a whole album; use one of them");
                    }
                    if (searching || album.is_some()) && a.flag("--no-browser") {
                        usage("photos pick: --auto, --queries, --sweep and --album drive the picker in a browser themselves; drop --no-browser");
                    }
                    // a sweep takes every result of each month (or day)
                    let select_all = a.flag("--all") || sweep;
                    if a.flag("--all") && !searching && album.is_none() {
                        usage("photos pick: --all is for the --auto agent (by hand, Shift+click the last tile)");
                    }
                    let videos_only = a.flag("--videos-only");
                    if videos_only && album.is_none() {
                        usage("photos pick: --videos-only is for --album (--auto always picks videos)");
                    }
                    let many = searches.len() > 1 || sweep;
                    if many && a.opt("--session").is_some() {
                        usage("photos pick: --session resumes one pick; it does not go with many searches");
                    }
                    if searching && searches.is_empty() && (a.opt("--queries").is_some() || sweep) {
                        usage("photos pick: no searches to run (the --queries file is empty?)");
                    }
                    let agent_drives = searching || album.is_some();
                    let since = a.opt("--since").map(str::to_string);
                    if let Some(d) = since.as_deref().filter(|d| !filmcraft_gphotos::is_day(d)) {
                        usage(format!("photos pick: --since wants a date like 2020-01-01, not `{d}`"));
                    }
                    let max_items = parse_num("--max-items").map(|n| u32::try_from(n).unwrap_or(u32::MAX));
                    let opts = filmcraft_gphotos::PickOptions {
                        max_items,
                        wait: std::time::Duration::from_secs(parse_num("--timeout").unwrap_or(if agent_drives { 900 } else { 600 })),
                        resume: a.opt("--session").map(str::to_string),
                        open_browser: !a.flag("--no-browser"),
                        since,
                        videos_only,
                    };
                    // what --max-items means to the agent: the session's item cap when it selects
                    // every result, else how many tiles it clicks
                    let agent_items = max_items.unwrap_or(if select_all { filmcraft_gphotos::api::MAX_PICK } else { 6 });
                    let album_title = album.clone();
                    // the arguments are fine: now the sign-in
                    let gp = filmcraft_gphotos::GooglePhotos::configured().unwrap_or_else(|e| fail(format!("photos: {e}")));
                    // the import target first, so a bad --project/--bridge fails before the user picks
                    let mut b = Backend::open(&a);
                    let picked = if many {
                        let req = filmcraft_automation::photos::Searches {
                            queries: searches,
                            select_all,
                            max_items: agent_items,
                            per_session: filmcraft_automation::photos::PER_SESSION,
                        };
                        diag(format_args!(
                            "photos pick: {} search(es), up to {} per picking session and {} item(s) per session",
                            req.queries.len(),
                            req.per_session,
                            req.max_items
                        ));
                        let swept = tokio::task::spawn_blocking(move || {
                            filmcraft_automation::photos::pick_searches(&gp, &opts, &req, &mut |line| diag(format_args!("{line}")), &|| false)
                        })
                        .await
                        .unwrap_or_else(|e| fail(format!("photos pick: {e}")));
                        for (k, s) in swept.sessions.iter().enumerate() {
                            diag(format_args!(
                                "photos pick: session {}: {}{}",
                                k + 1,
                                if s.note.is_empty() { "no report" } else { s.note.as_str() },
                                s.error.as_deref().map(|e| format!(" ({e})")).unwrap_or_default()
                            ));
                        }
                        if let Some(why) = &swept.stopped {
                            diag(format_args!("photos pick: stopped early: {why}"));
                        }
                        if !swept.too_big.is_empty() {
                            diag(format_args!(
                                "photos pick: {} search(es) matched more than a session holds ({} items): split them by day (a sweep: --month-days 1-31) or raise --max-items: {}",
                                swept.too_big.len(),
                                agent_items,
                                swept.too_big.join(" · ")
                            ));
                        }
                        if !swept.left.is_empty() {
                            let file = filmcraft_gphotos::data_dir().join("left-searches.txt");
                            let text: String = swept.left.iter().map(|q| format!("{q}\n")).collect();
                            match std::fs::create_dir_all(filmcraft_gphotos::data_dir()).and_then(|()| std::fs::write(&file, text)) {
                                Ok(()) => diag(format_args!(
                                    "photos pick: {} search(es) left for a next run (listed in {}): filmcraft-cli photos pick --queries '{}' --all",
                                    swept.left.len(),
                                    file.to_string_lossy(),
                                    file.to_string_lossy()
                                )),
                                Err(e) => {
                                    diag(format_args!("photos pick: {} search(es) left for a next run: {} ({e})", swept.left.len(), swept.left.join(" · ")))
                                }
                            }
                        }
                        let report = json!({"sessions": swept.sessions, "left": swept.left, "tooBig": swept.too_big, "stopped": swept.stopped});
                        Picks { picked: swept.picked, report, stopped: swept.stopped.is_some() }
                    } else {
                        // blocking HTTP and sleeps: keep them off the async runtime
                        let driven = tokio::task::spawn_blocking(move || {
                            // with --auto, a local vision agent (Qwen2.5-VL via Ollama) drives the
                            // picker: it types the query, clicks matching video tiles, presses Done;
                            // with --album it opens that album and selects all of it instead
                            let request = match (searching, album) {
                                (true, _) => Some(filmcraft_screenagent::agent::Request {
                                    queries: searches,
                                    max_items: agent_items,
                                    select_all,
                                    album: None,
                                    videos_only: true,
                                }),
                                (false, Some(title)) => Some(filmcraft_screenagent::agent::Request {
                                    max_items: opts.max_items.unwrap_or(filmcraft_gphotos::api::MAX_PICK),
                                    ..filmcraft_automation::photos::album_request(&title, videos_only)
                                }),
                                (false, None) => None,
                            };
                            filmcraft_automation::photos::pick_driven(&gp, &opts, request, &mut |line| diag(format_args!("{line}")), &|| false)
                        })
                        .await
                        .unwrap_or_else(|e| fail(format!("photos pick: {e}")))
                        .unwrap_or_else(|e| fail(format!("photos pick: {e}")));
                        match &driven.agent {
                            Some(Ok(s)) => diag(format_args!("photos pick: autopick: {}", s.note)),
                            Some(Err(e)) => diag(format_args!("photos pick: the autopick agent reported: {e}")),
                            None => {}
                        }
                        let nothing_to_pick = driven.nothing_to_pick();
                        // a search whose results alone pass the session's cap was unselected
                        let too_big = match &driven.agent {
                            Some(Ok(s)) if !s.too_big().is_empty() => Some(s.note.clone()),
                            _ => None,
                        };
                        let picked = match driven.outcome {
                            filmcraft_gphotos::PickOutcome::Pending(_) if too_big.is_some() => {
                                fail(format!("photos pick: nothing was picked: {}", too_big.unwrap_or_default()))
                            }
                            filmcraft_gphotos::PickOutcome::Done(p) => p,
                            filmcraft_gphotos::PickOutcome::Pending(_) if nothing_to_pick => match &album_title {
                                Some(t) => fail(format!("photos pick: the album `{t}` had nothing to pick")),
                                None => fail("photos pick: the search matched nothing to pick"),
                            },
                            filmcraft_gphotos::PickOutcome::Pending(_) if driven.abandoned => fail(
                                "photos pick: the agent did not finish the pick in time; its picking session was deleted and its tab closed (run it again)",
                            ),
                            filmcraft_gphotos::PickOutcome::Pending(s) => {
                                outln!("{}", json!({"status": "pending", "session": s.id, "pickerUri": s.autoclose_uri(), "expires": s.expire_time}));
                                fail(format!("photos pick: no selection yet; finish in the browser, then run `photos pick --session {}`", s.id));
                            }
                        };
                        Picks { picked, report: json!({}), stopped: false }
                    };
                    let Picks { picked, report, stopped } = picked;
                    for f in &picked.failed {
                        diag(format_args!("photos pick: {f}"));
                    }
                    // an album is selected whole, photos too (the picker cannot select by type)
                    let photos = filmcraft_gphotos::not_videos(&picked.skipped);
                    if photos > 0 && album_title.is_some() {
                        diag(format_args!("photos pick: {photos} photo(s) of the album were selected with it and not downloaded (--videos-only)"));
                    }
                    if picked.fetched.is_empty() {
                        if report.as_object().is_some_and(|o| !o.is_empty()) {
                            outln!("{report}");
                        }
                        fail(if !picked.failed.is_empty() {
                            "photos pick: nothing could be downloaded"
                        } else if photos > 0 && photos == picked.skipped.len() {
                            "photos pick: everything picked was a photo (--videos-only downloads only videos)"
                        } else if photos > 0 {
                            "photos pick: everything picked was a photo (--videos-only) or created before --since"
                        } else if !picked.skipped.is_empty() {
                            "photos pick: everything picked was created before --since"
                        } else {
                            "photos pick: nothing was picked"
                        });
                    }
                    let (paths, items, cached) = picked_json(&picked);
                    match b.exec("file.import", json!({"paths": paths, "reuse": true})).await {
                        Ok(v) => {
                            let mut out = json!({
                                "imported": v, "items": items, "downloaded": picked.fetched.len().saturating_sub(cached), "cached": cached,
                                "failed": picked.failed, "skipped": picked.skipped,
                            });
                            if let (Some(o), Some(r)) = (out.as_object_mut(), report.as_object()) {
                                o.extend(r.clone());
                            }
                            outln!("{out}");
                            b.finish(&a).await;
                        }
                        Err(e) => fail(format!("photos pick: import: {e}")),
                    }
                    if !picked.failed.is_empty() || stopped {
                        std::process::exit(1);
                    }
                }
                "tidy" => {
                    if a.positionals.len() > 2 {
                        usage("photos tidy (closes the Google Photos picker tabs left open in Chrome)");
                    }
                    // tidying drives Chrome too: not while an agent or a session holds it
                    let _browser = filmcraft_locks::acquire(filmcraft_locks::Resource::Chrome, "photos tidy", std::time::Duration::from_secs(120), &mut |e| {
                        diag(format_args!("photos tidy: waiting: {e}"))
                    })
                    .unwrap_or_else(|e| fail(format!("photos tidy: {e}")));
                    let tidied = tokio::task::spawn_blocking(tidy_picker_tabs).await.unwrap_or_else(|e| fail(format!("photos tidy: {e}")));
                    match tidied {
                        Ok(t) => {
                            for n in &t.notes {
                                diag(format_args!("photos tidy: {n}"));
                            }
                            diag(format_args!("photos tidy: closed {} of {} picker tab(s)", t.closed, t.found));
                            print(&a, &json!(t));
                            if !t.left.is_empty() {
                                std::process::exit(1);
                            }
                        }
                        Err(e) => fail(format!("photos tidy: {e}")),
                    }
                }
                "upload" => {
                    let path = a.pos(2).unwrap_or_else(|| usage("photos upload <file> [--description D] [--album <title|id>]"));
                    if a.positionals.len() > 3 {
                        usage("photos upload: one file at a time");
                    }
                    let description = a.opt("--description").map(str::to_string);
                    let album = album_opt(&a, "photos upload", "--album");
                    match upload_to_photos(absolute(path), description, album).await {
                        Ok(v) => print(&a, &v),
                        Err(e) => fail(format!("photos upload: {e}")),
                    }
                }
                "album" => match a.pos(2) {
                    Some("create") => {
                        let title = a.positionals.get(3..).map(|p| p.join(" ")).unwrap_or_default();
                        if title.trim().is_empty() {
                            usage("photos album create <title> [--description D]");
                        }
                        let title = filmcraft_gphotos::albums::clean_title(&title).unwrap_or_else(|e| usage(format!("photos album create: {e}")));
                        let description = a.opt("--description").map(str::to_string);
                        if let Err(e) = filmcraft_gphotos::albums::check_note(description.as_deref()) {
                            usage(format!("photos album create: {e}"));
                        }
                        match with_albums(move |al| al.ensure(&title, description.as_deref())).await {
                            Ok(e) => {
                                if let Some(why) = &e.note_error {
                                    diag(format_args!("photos album create: the album is made, but its description was not added: {why}"));
                                }
                                print(&a, &e.to_json());
                            }
                            Err(e) => fail(format!("photos album create: {e}")),
                        }
                    }
                    Some("add") => {
                        let (Some(album), Some(ids)) = (a.pos(3), a.positionals.get(4..).filter(|ids| !ids.is_empty())) else {
                            usage("photos album add <title|id> <mediaItemId>...  (ids from `photos upload`; Google adds only items FilmCraft uploaded)")
                        };
                        let (album, ids) = (album.to_string(), ids.to_vec());
                        if let Err(e) = filmcraft_gphotos::albums::check_item_ids(&ids) {
                            usage(format!("photos album add: {e}"));
                        }
                        match with_albums(move |al| al.add(&album, &ids)).await {
                            Ok(added) => print(&a, &added.to_json()),
                            Err(e) => fail(format!("photos album add: {e}")),
                        }
                    }
                    Some(other) => usage(format!("photos album: unknown `{other}` (create <title> | add <title|id> <mediaItemId>...)")),
                    None => usage("photos album create <title> [--description D] | photos album add <title|id> <mediaItemId>..."),
                },
                "albums" => {
                    if a.positionals.len() > 2 {
                        usage("photos albums [--json]");
                    }
                    match with_albums(|al| al.list()).await {
                        Ok(listed) if a.flag("--json") => print(&a, &listed.to_json()),
                        Ok(listed) => {
                            for al in &listed.albums {
                                let count = al.media_items.map(|n| n.to_string()).unwrap_or_else(|| "?".into());
                                outln!("{:<40} {count:>6} item(s)  {}", al.title, al.product_url.as_deref().unwrap_or(&al.id));
                            }
                            outln!(
                                "{} album(s) made by FilmCraft{}; Google shows apps no other albums (pick from those with `photos pick --album <title>`)",
                                listed.albums.len(),
                                if listed.complete { "" } else { " (more than could be listed)" }
                            );
                        }
                        Err(e) => fail(format!("photos albums: {e}")),
                    }
                }
                "cached" => {
                    let list = filmcraft_gphotos::cache::list();
                    if a.flag("--json") {
                        print(&a, &Value::Array(list));
                    } else {
                        for c in &list {
                            let mib = c["bytes"].as_u64().unwrap_or(0) as f64 / (1024.0 * 1024.0);
                            outln!("{:<48} {mib:>9.1} MiB  {}", c["id"].as_str().unwrap_or(""), c["filename"].as_str().unwrap_or(""));
                        }
                        outln!("{} cached item(s); `photos add <id>…` imports them again", list.len());
                    }
                }
                "add" => {
                    let ids: Vec<String> = a.positionals.get(2..).unwrap_or_default().to_vec();
                    if ids.is_empty() {
                        usage("photos add <mediaItemId>...  (ids from `photos cached`; new items come from `photos pick`)");
                    }
                    let mut paths = Vec::new();
                    for id in &ids {
                        match filmcraft_gphotos::cache::find(id) {
                            Some(p) => paths.push(p.to_string_lossy().into_owned()),
                            None => fail(format!("photos add {id}: not downloaded yet — Google only shares items picked in the picker; run `photos pick`")),
                        }
                    }
                    let mut b = Backend::open(&a);
                    match b.exec("file.import", json!({"paths": paths, "reuse": true})).await {
                        Ok(v) => {
                            outln!("{}", json!({"imported": v, "cached": ids.len()}));
                            b.finish(&a).await;
                        }
                        Err(e) => fail(format!("photos add: import: {e}")),
                    }
                }
                "search" => usage(
                    "photos search: Google retired library browsing and search for apps on 2025-03-31; use `photos pick` (or `photos pick --auto <query>`) to choose items in Google Photos",
                ),
                other => usage(format!("photos: unknown `{other}` (auth | pick | tidy | upload | album | albums | cached | add)")),
            }
        }
        "youtube" => youtube::run(&a).await,
        "instagram" => instagram::run(&a).await,
        "publish" => publish::run(&a).await,
        "showcase" => {
            // a narrated, counted showcase of every clip an analysis found (`--report`, from
            // analyze --out), or of a Google Photos album's clips (`--album`: picked, imported
            // and analysed first); every usage error comes before anything is read
            let (source, album) = showcase_source(&a);
            let mut opts = filmcraft_autocut::showcase::Options::default();
            if let Some(t) = a.opt("--title") {
                opts.title = t.to_string();
            }
            opts.subtitle = a.opt("--subtitle").map(str::to_string);
            opts.voice = a.opt("--voice").map(str::to_string);
            if let Some(m) = a.opt("--mood") {
                opts.mood = m.to_string();
            }
            if let Some(b) = a.opt("--bpm") {
                opts.bpm = b.parse().unwrap_or_else(|_| usage(format!("showcase: --bpm wants a number, not `{b}`")));
            }
            if let Some(c) = a.opt("--min-coverage") {
                opts.min_coverage = c.parse().unwrap_or_else(|_| usage(format!("showcase: --min-coverage wants 0..1, not `{c}`")));
            }
            opts.export = a.opt("--export").map(str::to_string);
            // the user's music under the show (absolute: the running app imports it over --bridge);
            // a missing file fails now, not after the script is written
            if let Some(m) = a.opt("--music") {
                if !std::path::Path::new(m).is_file() {
                    fail(format!("showcase: --music: {m}: no such file"));
                }
                opts.music = Some(absolute(m));
            }
            opts.stabilize = a.flag("--stabilize");
            opts.match_color = a.flag("--match-color");
            opts.fill_frame = !a.flag("--no-fill");
            // a filled shot's crop follows its robot (the subject tracker); --no-track: the tilt
            opts.follow_subject = !a.flag("--no-track");
            if let Some(m) = a.opt("--target-minutes") {
                match m.parse::<f64>() {
                    Ok(v) if v.is_finite() && v > 0.0 && v <= 24.0 * 60.0 => opts.target_minutes = Some(v),
                    _ => usage(format!("showcase: --target-minutes wants a length in minutes (e.g. 10), not `{m}`")),
                }
            }
            // the look's flags, checked now (they override the plan's fields, applied after it)
            let style_flag =
                a.opt("--style").map(|s| filmcraft_autocut::look::Style::from_plan(&json!(s)).unwrap_or_else(|e| usage(format!("showcase: --style: {e}"))));
            let loudness_flag = a
                .opt("--loudness")
                .map(|l| filmcraft_autocut::look::Loudness::from_plan(&json!(l)).unwrap_or_else(|e| usage(format!("showcase: --loudness: {e}"))));
            let soundfont_flag = a.opt("--soundfont").map(|f| {
                let ids = ["auto", "oscillators", "generaluser-gs", "generaluser", "gugs"];
                if !ids.contains(&f.trim().to_ascii_lowercase().as_str()) && !std::path::Path::new(f).is_file() {
                    fail(format!("showcase: --soundfont: {f}: neither auto, oscillators, generaluser-gs nor an .sf2 file"));
                }
                if ids.contains(&f.trim().to_ascii_lowercase().as_str()) { f.trim().to_string() } else { absolute(f) }
            });
            // --upload: the export goes to Google Photos once built; a missing sign-in or
            // consent fails now, not after the build. --upload-album: the upload also goes into
            // that album (FilmCraft's, created when missing), where the next iteration's
            // material can be added by hand
            let upload = a.flag("--upload") && !a.flag("--plan");
            if upload {
                if opts.export.is_none() {
                    usage("showcase: --upload sends the export to Google Photos; add --export out.mp4");
                }
                if let Err(e) = photos_readiness(album.is_some()) {
                    fail(format!("showcase: --upload: {e}"));
                }
            }
            // --upload-youtube: the export goes to YouTube once built; bad options and a missing
            // channel or sign-in fail now, not after the build
            let upload_youtube = a.flag("--upload-youtube") && !a.flag("--plan");
            if upload_youtube {
                if opts.export.is_none() {
                    usage("showcase: --upload-youtube sends the export to YouTube; add --export out.mp4");
                }
                // the options now (a usage error), then the channel and its sign-in from disk
                let req =
                    youtube::showcase_request(&a, "showcase.mp4", &opts.title, None).unwrap_or_else(|e| usage(format!("showcase: --upload-youtube: {e}")));
                if let Err(e) = youtube::readiness(&req) {
                    fail(format!("showcase: --upload-youtube: {e}"));
                }
            }
            // the clips: the report, or the album picked, imported and analysed in the project
            // the show is then built in
            let mut backend: Option<Backend> = None;
            let mut album_summary = Value::Null;
            let (v, from) = match source {
                ShowcaseSource::Report(report) => {
                    let bytes = std::fs::read(&report).unwrap_or_else(|e| fail(format!("showcase: {report}: {e}")));
                    let v: Value = serde_json::from_slice(&bytes).unwrap_or_else(|e| fail(format!("showcase: {report}: {e}")));
                    (v, report)
                }
                ShowcaseSource::Album { title, params } => {
                    // the picker's sign-in from disk first (the app has its own over --bridge)
                    if a.opt("--bridge").is_none()
                        && let Err(e) = filmcraft_automation::photos::pick_readiness()
                    {
                        fail(format!("showcase: --album: {e}"));
                    }
                    let mut b = Backend::open(&a);
                    diag(format_args!("showcase: making the video from the Google Photos album \"{title}\""));
                    let (report, summary) = album_analysis(&mut b, params).await.unwrap_or_else(|e| fail(format!("showcase: {e}")));
                    let al = &summary["album"];
                    diag(format_args!(
                        "showcase: album \"{title}\": {} clip(s) ({} downloaded, {} from the cache; {} imported, {} already in the project)",
                        al["picked"], al["downloaded"], al["cached"], al["imported"], al["reused"]
                    ));
                    for f in al["failed"].as_array().into_iter().flatten().filter_map(Value::as_str) {
                        diag(format_args!("showcase: album: {f}"));
                    }
                    if al["skipped"].as_u64().is_some_and(|n| n > 0) {
                        diag(format_args!("showcase: album: {} item(s) skipped (photos, or created before --since)", al["skipped"]));
                    }
                    if let Some(out) = a.opt("--report-out") {
                        std::fs::write(out, serde_json::to_string_pretty(&report).unwrap_or_default())
                            .unwrap_or_else(|e| fail(format!("showcase: {out}: {e}")));
                        diag(format_args!("showcase: wrote {out} (`showcase --report {out}` builds again without picking)"));
                    }
                    // the title: --title, else the prompt's, else "Every <Subject> I've Filmed"
                    if a.opt("--title").is_none() {
                        let subject = report["subject"].as_str().unwrap_or("robot");
                        opts.title = summary["title"].as_str().map_or_else(|| filmcraft_autocut::host::default_title(subject), str::to_string);
                    }
                    album_summary = summary["album"].clone();
                    backend = Some(b);
                    (report, format!("the album \"{title}\""))
                }
            };
            let mut facts: Vec<filmcraft_autocut::analyze::ClipFacts> = serde_json::from_value(v.get("clips").cloned().unwrap_or(Value::Null))
                .unwrap_or_else(|e| fail(format!("showcase: {from}: not an analysis report ({e})")));
            if let Some(subject) = v.get("subject").and_then(Value::as_str) {
                opts.subject = subject.to_string();
            }
            for f in &mut facts {
                f.label = filmcraft_autocut::analyze::sanitize_label_with(&f.label, &f.kind, &opts.subject, &f.text);
            }
            let mut entries = filmcraft_autocut::showcase::group(&facts, &opts.subject, opts.min_coverage);
            let cfg = filmcraft_autocut::config::Config::from_env();
            // a reviewed plan (`--script-in`): its (edited) entries replace the grouping, and its
            // script, when it has one, the model's
            let mut reviewed: Option<filmcraft_autocut::showcase::Script> = None;
            // the plan's look (theme, timeline, transitions, finishing), kept for the plan written out
            let mut look_fields = serde_json::Map::new();
            if let Some(path) = a.opt("--script-in") {
                let b = std::fs::read(path).unwrap_or_else(|e| fail(format!("showcase: {path}: {e}")));
                let sv: Value = serde_json::from_slice(&b).unwrap_or_else(|e| fail(format!("showcase: {path}: {e}")));
                opts.look = filmcraft_autocut::look::Look::from_plan(&sv).unwrap_or_else(|e| fail(format!("showcase: {path}: {e}")));
                look_fields = filmcraft_autocut::look::Look::plan_fields(&sv);
                if let Some(es) = sv.get("entries") {
                    entries = serde_json::from_value(es.clone()).unwrap_or_else(|e| fail(format!("showcase: {path}: bad entries ({e})")));
                    for (i, e) in entries.iter_mut().enumerate() {
                        e.number = i + 1;
                    }
                }
                let script = match (sv.get("script"), sv.get("entries")) {
                    (Some(s), _) => Some(s.clone()),
                    // entries alone: the model writes the words for them
                    (None, Some(_)) => None,
                    (None, None) => Some(sv),
                };
                if let Some(s) = script {
                    let script: filmcraft_autocut::showcase::Script =
                        serde_json::from_value(s).unwrap_or_else(|e| fail(format!("showcase: {path}: not a script ({e})")));
                    if script.lines.len() != entries.len() {
                        fail(format!("showcase: {path}: {} script lines for {} entries", script.lines.len(), entries.len()));
                    }
                    reviewed = Some(script);
                }
            }
            if let Some(st) = style_flag {
                opts.look.style = st;
            }
            if a.flag("--timeline") {
                opts.look.timeline = true;
            }
            if let Some(l) = loudness_flag {
                opts.look.loudness = l;
            }
            if a.flag("--embed-captions") {
                opts.look.embed_captions = true;
            }
            if soundfont_flag.is_some() {
                opts.look.soundfont = soundfont_flag;
            }
            let script = match reviewed {
                Some(s) => s,
                None => {
                    // the script model when it is installed, else the classifier
                    let mut model = filmcraft_autocut::ollama::Ollama::new(&cfg.ollama_host, &cfg.script_model, cfg.http_timeout);
                    if !model.available().unwrap_or(false) {
                        diag(format_args!("showcase: {} is not installed; writing with {}", cfg.script_model, cfg.llm_model));
                        model = filmcraft_autocut::ollama::Ollama::new(&cfg.ollama_host, &cfg.llm_model, cfg.http_timeout);
                    }
                    filmcraft_autocut::showcase::write_script(&model, &opts, &entries, &mut |l| diag(format_args!("showcase: {l}"))).await
                }
            };
            let mut plan = json!({"entries": entries, "script": script});
            if let Some(o) = plan.as_object_mut() {
                o.extend(look_fields);
            }
            if let Some(out) = a.opt("--script-out") {
                std::fs::write(out, serde_json::to_string_pretty(&plan).unwrap_or_default()).unwrap_or_else(|e| fail(format!("showcase: {out}: {e}")));
                diag(format_args!("showcase: wrote {out}"));
            }
            if a.flag("--plan") {
                print(&a, &plan);
                // the album's clips are in the project now: --save keeps them
                if let Some(mut b) = backend {
                    b.finish(&a).await;
                }
                return;
            }
            // QA of the built show: on with an export, `--qa` without one, `--no-qa` never
            let qa_on = !a.flag("--no-qa") && (a.flag("--qa") || opts.export.is_some());
            let qa_model = if qa_on {
                let m = filmcraft_autocut::ollama::Ollama::new(&cfg.ollama_host, &cfg.analysis_model, cfg.http_timeout);
                if m.available().unwrap_or(false) {
                    Some(m)
                } else {
                    diag(format_args!("showcase: QA: {} is not reachable; the subject check is skipped", cfg.analysis_model));
                    None
                }
            } else {
                None
            };
            let mut b = backend.unwrap_or_else(|| Backend::open(&a));
            let mut progress = |l: &str| diag(format_args!("showcase: {l}"));
            let (result, qa) = match &mut b {
                Backend::Local(s) => {
                    let session = std::mem::take(s);
                    let ex = filmcraft_autocut::SessionExecutor::from(*session);
                    let r = filmcraft_autocut::showcase::build(&ex, &opts, &entries, &script, &mut progress).await;
                    let qa = match &r {
                        Ok(built) if qa_on => Some(showcase_qa(&ex, &opts, built, qa_model.as_ref(), &mut progress).await),
                        _ => None,
                    };
                    if let Ok(inner) = std::sync::Arc::try_unwrap(ex.0) {
                        **s = inner.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
                    }
                    (r, qa)
                }
                Backend::Bridge(_) => {
                    let addr = a.opt("--bridge").unwrap_or_default();
                    let client = BridgeClient::new(addr).unwrap_or_else(|e| usage(e));
                    let ex = filmcraft_automation::AutocutExecutor::bridge(client);
                    let r = filmcraft_autocut::showcase::build(&ex, &opts, &entries, &script, &mut progress).await;
                    let qa = match &r {
                        Ok(built) if qa_on => Some(showcase_qa(&ex, &opts, built, qa_model.as_ref(), &mut progress).await),
                        _ => None,
                    };
                    (r, qa)
                }
            };
            match result {
                Ok(built) => {
                    for n in &built.notes {
                        diag(format_args!("showcase: {n}"));
                    }
                    // chapters for the YouTube description, next to the export
                    let chapters = filmcraft_autocut::showcase::youtube_chapters(&built.chapters, built.seconds);
                    if let Some(path) = &opts.export
                        && !chapters.is_empty()
                    {
                        let txt = std::path::Path::new(path).with_extension("chapters.txt");
                        match std::fs::write(&txt, format!("{chapters}\n")) {
                            Ok(()) => diag(format_args!("showcase: chapters in {}", txt.display())),
                            Err(e) => diag(format_args!("showcase: {}: {e}", txt.display())),
                        }
                    }
                    for file in &built.subtitle_files {
                        diag(format_args!("showcase: subtitles in {file}"));
                    }
                    if let Some(file) = &built.thumbnail {
                        diag(format_args!("showcase: thumbnail in {file}"));
                    }
                    let mut out = serde_json::to_value(&built).unwrap_or_default();
                    out["youtubeChapters"] = json!(chapters);
                    if !album_summary.is_null() {
                        out["album"] = album_summary;
                    }
                    if let Some(mut report) = qa {
                        diag(format_args!("showcase: QA: {}", report.summary()));
                        for f in report.findings.iter().filter(|f| f.severity >= filmcraft_autocut::qa::Severity::Warning).take(40) {
                            diag(format_args!("showcase: QA {:?} at {:.2}s ({}): {}", f.severity, f.seconds, f.check, f.message));
                        }
                        // the report and the contact sheet next to the export
                        if let Some(path) = &opts.export {
                            match filmcraft_autocut::qa::write_next_to(&mut report, std::path::Path::new(path)) {
                                Ok((j, png)) => {
                                    diag(format_args!("showcase: QA report in {}", j.display()));
                                    if let Some(p) = png {
                                        diag(format_args!("showcase: contact sheet in {}", p.display()));
                                    }
                                }
                                Err(e) => diag(format_args!("showcase: QA: {e}")),
                            }
                        }
                        out["qa"] = serde_json::to_value(&report).unwrap_or_default();
                    }
                    // --upload: the finished export to Google Photos, titled like the video
                    let mut upload_failed = None;
                    if upload && let Some(requested) = &opts.export {
                        let file = built.export.as_ref().and_then(|r| written_paths(r, requested).into_iter().next()).unwrap_or_else(|| requested.clone());
                        let description = a.opt("--description").map(str::to_string).or_else(|| Some(opts.title.clone()).filter(|t| !t.trim().is_empty()));
                        match upload_to_photos(absolute(&file), description, album.clone()).await {
                            Ok(v) => out["upload"] = v,
                            Err(e) => {
                                out["uploadError"] = json!(e);
                                upload_failed = Some(e);
                            }
                        }
                    }
                    // --upload-youtube: the export to YouTube, titled like the video, with its
                    // chapters and thumbnail
                    if upload_youtube && let Some(requested) = &opts.export {
                        let file = built.export.as_ref().and_then(|r| written_paths(r, requested).into_iter().next()).unwrap_or_else(|| requested.clone());
                        let r = match youtube::showcase_request(&a, &file, &opts.title, built.thumbnail.as_deref()) {
                            Ok(req) => youtube::upload(req).await,
                            Err(e) => Err(e),
                        };
                        match r {
                            Ok(v) => out["youtube"] = v,
                            Err(e) => {
                                out["youtubeError"] = json!(e);
                                upload_failed = Some(upload_failed.map_or(e.clone(), |p| format!("{p}; YouTube: {e}")));
                            }
                        }
                    }
                    print(&a, &out);
                    b.finish(&a).await;
                    if let Some(e) = upload_failed {
                        fail(format!("showcase: the video is built, but the upload failed: {e}"));
                    }
                }
                Err(e) => fail(format!("showcase: {e}")),
            }
        }
        "shorts" => {
            // long video → vertical shorts (autocut.shorts); every usage error before anything runs
            let src = a.pos(1).unwrap_or_else(|| usage("shorts <project.fcproj | media file> [--count N] [--aspect 9:16] [--export dir] [--upload-youtube]"));
            let params = shorts_params(&a).unwrap_or_else(|e| usage(format!("shorts: {e}")));
            if params["uploadYoutube"] == json!(true) {
                let req = youtube::showcase_request(&a, "short.mp4", "Short", None).unwrap_or_else(|e| usage(format!("shorts: --upload-youtube: {e}")));
                if let Err(e) = youtube::readiness(&req) {
                    fail(format!("shorts: --upload-youtube: {e}"));
                }
            }
            let is_project = std::path::Path::new(src).extension().is_some_and(|x| x.eq_ignore_ascii_case("fcproj"));
            if a.opt("--bridge").is_none() && !std::path::Path::new(src).exists() {
                fail(format!("shorts: {src}: no such file"));
            }
            let mut b = Backend::open(&a);
            let mut params = params;
            if is_project {
                b.exec("file.open", json!({"path": absolute(src)})).await.unwrap_or_else(|e| fail(format!("shorts: {src}: {e}")));
            } else {
                let r = b.exec("file.import", json!({"paths": [absolute(src)], "reuse": true})).await.unwrap_or_else(|e| fail(format!("shorts: {src}: {e}")));
                let item = r["items"]
                    .as_array()
                    .and_then(|i| i.first())
                    .and_then(Value::as_u64)
                    .unwrap_or_else(|| fail(format!("shorts: {src}: nothing was imported")));
                params["item"] = json!(item);
                if let Some(o) = params.as_object_mut() {
                    o.remove("sequence");
                }
            }
            diag(format_args!("shorts: finding the best {} passage(s) of {src}", params["count"]));
            match b.exec("autocut.shorts", params).await {
                Ok(r) => {
                    let made = &r["showcase"];
                    for s in made["shorts"].as_array().map(Vec::as_slice).unwrap_or_default() {
                        diag(format_args!(
                            "shorts: #{} \"{}\" {:.1}-{:.1}s{}{}",
                            s["number"],
                            s["title"].as_str().unwrap_or_default(),
                            s["start"].as_f64().unwrap_or(0.0),
                            s["end"].as_f64().unwrap_or(0.0),
                            s["export"].as_str().map(|p| format!(" → {p}")).unwrap_or_default(),
                            if s["followed"] == json!(true) { "" } else { " (letterboxed: nobody to follow)" }
                        ));
                        for n in s["notes"].as_array().map(Vec::as_slice).unwrap_or_default() {
                            diag(format_args!("shorts:   {}", n.as_str().unwrap_or_default()));
                        }
                    }
                    print(&a, made);
                    b.finish(&a).await;
                }
                Err(e) => fail(format!("shorts: {e}")),
            }
        }
        "analyze" => {
            // which clips show <subject>, what it is, when, and its best moment
            let subject = a.positionals.get(1..).map(|p| p.join(" ")).unwrap_or_default();
            if subject.trim().is_empty() {
                usage("analyze '<subject>' [--items 1,2,3] [--frames N] [--screen N] [--out report.json] [--fresh]");
            }
            let items: Option<Vec<u64>> = a.opt("--items").map(|v| {
                v.split(',')
                    .filter(|x| !x.trim().is_empty())
                    .map(|x| x.trim().parse::<u64>().unwrap_or_else(|_| usage(format!("analyze: --items wants ids like 3,4,5, not `{x}`"))))
                    .collect()
            });
            let frames =
                a.opt("--frames").map(|v| v.parse::<usize>().unwrap_or_else(|_| usage(format!("analyze: --frames wants a number, not `{v}`")))).unwrap_or(6);
            // --screen N: look at N spread-out frames first and skip the clip when none shows the
            // subject (fast over many clips that mostly do not; 0 = off)
            let screen = a
                .opt("--screen")
                .map(|v| v.parse::<usize>().unwrap_or_else(|_| usage(format!("analyze: --screen wants a number of frames, not `{v}`"))))
                .unwrap_or(0);
            // side-by-side comparisons for telling the same robot across clips (0: by name only)
            let compare = a
                .opt("--compare")
                .map(|v| v.parse::<usize>().unwrap_or_else(|_| usage(format!("analyze: --compare wants a number of pairs, not `{v}`"))))
                .unwrap_or(filmcraft_autocut::identity::MAX_COMPARISONS);
            let cfg = filmcraft_autocut::config::Config::from_env();
            let model = filmcraft_autocut::ollama::Ollama::new(&cfg.ollama_host, &cfg.analysis_model, cfg.http_timeout);
            if !model.available().unwrap_or(false) {
                fail(format!(
                    "analyze: the vision model `{}` is not in Ollama at {} — `ollama pull {}`",
                    cfg.analysis_model, cfg.ollama_host, cfg.analysis_model
                ));
            }
            let mut b = Backend::open(&a);
            // clips analysed before (same file, subject, model and settings) come from the
            // analysis cache in the data directory; --fresh analyses them again
            let cache = a
                .opt("--data-dir")
                .map(std::path::PathBuf::from)
                .or_else(filmcraft_engine::autosave::default_data_dir)
                .map_or_else(filmcraft_autocut::analysis_cache::AnalysisCache::off, |d| filmcraft_autocut::analysis_cache::AnalysisCache::in_data_dir(&d))
                .fresh(a.flag("--fresh"));
            let mut progress = |l: &str| diag(format_args!("analyze: {l}"));
            // --relabel report.json: read the text and describe again from the report's best
            // moments, no locating
            let earlier: Option<Vec<filmcraft_autocut::analyze::ClipFacts>> = a.opt("--relabel").map(|path| {
                let bytes = std::fs::read(path).unwrap_or_else(|e| fail(format!("analyze: {path}: {e}")));
                let v: Value = serde_json::from_slice(&bytes).unwrap_or_else(|e| fail(format!("analyze: {path}: {e}")));
                serde_json::from_value(v.get("clips").cloned().unwrap_or(Value::Null))
                    .unwrap_or_else(|e| fail(format!("analyze: {path}: not an analysis report ({e})")))
            });
            let result = match &mut b {
                Backend::Local(s) => {
                    let session = std::mem::take(s);
                    let ex = filmcraft_autocut::SessionExecutor::from(*session);
                    let r =
                        filmcraft_autocut::analyze::run(&model, &ex, &subject, items.as_deref(), frames, screen, earlier, compare, &cache, &mut progress).await;
                    if let Ok(inner) = std::sync::Arc::try_unwrap(ex.0) {
                        **s = inner.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
                    }
                    r
                }
                Backend::Bridge(_) => {
                    let addr = a.opt("--bridge").unwrap_or_default();
                    let client = BridgeClient::new(addr).unwrap_or_else(|e| usage(e));
                    let ex = filmcraft_automation::AutocutExecutor::bridge(client);
                    filmcraft_autocut::analyze::run(&model, &ex, &subject, items.as_deref(), frames, screen, earlier, compare, &cache, &mut progress).await
                }
            };
            match result {
                Ok(facts) => {
                    let v = json!({"subject": subject, "model": cfg.analysis_model, "clips": facts, "cached": cache.hits()});
                    if let Some(out) = a.opt("--out") {
                        let text = serde_json::to_string_pretty(&v).unwrap_or_default();
                        std::fs::write(out, text).unwrap_or_else(|e| fail(format!("analyze: {out}: {e}")));
                        diag(format_args!("analyze: wrote {out}"));
                    }
                    print(&a, &v);
                }
                Err(e) => fail(format!("analyze: {e}")),
            }
        }
        "autocut" => {
            let prompt = a.positionals.get(1..).map(|p| p.join(" ")).unwrap_or_default();
            if prompt.is_empty() {
                usage("autocut '<prompt>' [--dry-run] [--frames]");
            }
            let opts = filmcraft_autocut::runner::Options {
                mode: if a.flag("--dry-run") { filmcraft_autocut::runner::Mode::PlanOnly } else { filmcraft_autocut::runner::Mode::Execute },
                frames: a.flag("--frames"),
            };
            let cfg = filmcraft_autocut::config::Config::from_env();
            let mut b = Backend::open(&a);
            let report = match &mut b {
                Backend::Local(s) => {
                    // AutoCut plans through an Executor; lend it the session and take it back for --save
                    let session = std::mem::take(s);
                    let ex = filmcraft_autocut::SessionExecutor::from(*session);
                    let r = filmcraft_autocut::runner::run_with(&filmcraft_autocut::runner::Runners::from_config(&cfg), &prompt, &ex, &cfg, opts).await;
                    if let Ok(inner) = std::sync::Arc::try_unwrap(ex.0) {
                        **s = inner.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
                    }
                    r
                }
                Backend::Bridge(_) => {
                    // a second connection for the planner; the app serves 16, and the original
                    // stays with `b` for --save
                    let addr = a.opt("--bridge").unwrap_or_default();
                    let client = BridgeClient::new(addr).unwrap_or_else(|e| usage(e));
                    let ex = filmcraft_automation::AutocutExecutor::bridge(client);
                    filmcraft_autocut::runner::run_with(&filmcraft_autocut::runner::Runners::from_config(&cfg), &prompt, &ex, &cfg, opts).await
                }
            };
            match report {
                Ok(r) => {
                    for n in &r.notes {
                        diag(format_args!("autocut: {n}"));
                    }
                    if let Some(undo) = r.undo {
                        diag(format_args!("autocut: {undo} undoable edit(s); `exec edit.undo` reverts them"));
                    }
                    let v = serde_json::to_value(&r).unwrap_or_default();
                    print(&a, &v);
                    if r.needs_supervisor {
                        // the report (with the document) is on stdout; a caller that can plan —
                        // an agent over MCP — takes it from here
                        fail("autocut: this prompt needs the supervisor; run it through an MCP client (filmcraft-cli mcp) or rephrase as a routine edit");
                    }
                }
                Err(e) => fail(format!("autocut: {e}")),
            }
            b.finish(&a).await;
        }
        "commands" | "describe" => {
            let mut b = Backend::open(&a);
            let list = b.exec("command.list", json!({})).await.unwrap_or_else(|e| fail(e));
            let list = list.as_array().cloned().unwrap_or_default();
            if cmd == "describe" {
                let id = a.pos(1).unwrap_or_else(|| usage("describe <id>"));
                match list.iter().find(|c| c["id"] == id) {
                    Some(c) => print(&a, c),
                    None => fail(format!("unknown command `{id}` (try `filmcraft-cli commands {id}`)")),
                }
                return;
            }
            let f = a.pos(1).map(str::to_ascii_lowercase).unwrap_or_default();
            let hit = |c: &Value| {
                let id = c["id"].as_str().unwrap_or("").to_ascii_lowercase();
                let label = c["label"].as_str().unwrap_or("").to_ascii_lowercase();
                f.is_empty() || id.contains(&f) || label.contains(&f)
            };
            let list: Vec<Value> = list.into_iter().filter(hit).collect();
            if a.flag("--json") {
                print(&a, &Value::Array(list));
            } else {
                for c in &list {
                    let s = |k: &str| c[k].as_str().unwrap_or("").to_string();
                    outln!("{:<32} {:<34} {:<14} {}", s("id"), s("label"), s("shortcut"), s("params"));
                }
            }
        }
        "template" => {
            let mut b = Backend::open(&a);
            let key = |i: usize, what: &str| a.pos(i).map(str::to_string).unwrap_or_else(|| usage(format!("template {what}")));
            let (id, params) = match a.pos(1) {
                Some("list") => ("template.list", a.pos(2).map(|q| json!({"query": q})).unwrap_or(json!({}))),
                Some("inspect") => ("template.inspect", json!({"template": template_key(&key(2, "inspect <id|name|path>"))})),
                Some("validate") => ("template.validate", json!({"path": absolute(&key(2, "validate <path>"))})),
                Some("save") => {
                    let mut p = json!({"path": absolute(&key(2, "save <out.fcvideo-template | out.zip>"))});
                    if let Some(n) = a.opt("--name") {
                        p["name"] = json!(n);
                    }
                    if let Some(d) = a.opt("--description") {
                        p["description"] = json!(d);
                    }
                    ("template.save", p)
                }
                Some("apply") => {
                    let mut p = json!({"template": template_key(&key(2, "apply <id|name|path> [--variant V] [--slot id=value]..."))});
                    let mut slots = serde_json::Map::new();
                    for x in a.all("--slot") {
                        let Some((sid, v)) = x.split_once('=') else { usage(format!("--slot {x}: use <slot>=<file|item id|text>")) };
                        let v = if std::path::Path::new(&filmcraft_engine::export_tools::expand_home(v)).exists() {
                            json!(absolute(&filmcraft_engine::export_tools::expand_home(v)))
                        } else {
                            v.parse::<u64>().map(|n| json!(n)).unwrap_or_else(|_| json!(v))
                        };
                        match slots.entry(sid.to_string()).or_insert_with(|| json!([])) {
                            Value::Array(list) => list.push(v),
                            other => *other = json!([v]),
                        }
                    }
                    p["slots"] = Value::Object(slots);
                    if let Some(v) = a.opt("--variant") {
                        p["variant"] = json!(v);
                    }
                    if let Some(n) = a.opt("--name") {
                        p["name"] = json!(n);
                    }
                    ("template.apply", p)
                }
                _ => usage("template list | inspect <t> | validate <path> | apply <t> [--variant V] [--slot id=value]... | save <out>"),
            };
            match b.exec(id, params).await {
                Ok(v) if id == "template.list" && !a.flag("--json") => {
                    for t in v["templates"].as_array().cloned().unwrap_or_default() {
                        let slots: Vec<String> = t["slots"]
                            .as_array()
                            .cloned()
                            .unwrap_or_default()
                            .iter()
                            .map(|s| format!("{}×{}", s["id"].as_str().unwrap_or(""), s["count"]))
                            .collect();
                        outln!(
                            "{:<34} {:<26} {:>6.1} s  {:<10} {}",
                            t["id"].as_str().unwrap_or(""),
                            t["name"].as_str().unwrap_or(""),
                            t["seconds"].as_f64().unwrap_or(0.0),
                            t["variants"].as_array().map(|v| v.iter().filter_map(|x| x["id"].as_str()).collect::<Vec<_>>().join(" ")).unwrap_or_default(),
                            slots.join(", ")
                        );
                    }
                    for x in v["invalid"].as_array().cloned().unwrap_or_default() {
                        outln!("invalid: {} — {}", x["path"].as_str().unwrap_or(""), x["error"].as_str().unwrap_or(""));
                    }
                }
                Ok(v) => {
                    print(&a, &v);
                    if id == "template.validate" && v["valid"] != json!(true) {
                        std::process::exit(1);
                    }
                }
                Err(e) => fail(format!("{id}: {e}")),
            }
            b.finish(&a).await;
        }
        "exec" => {
            let id = a.pos(1).unwrap_or_else(|| usage("exec <id> [key=value ...]"));
            let params = a.params_from(2).unwrap_or_else(|e| usage(e));
            let mut b = Backend::open(&a);
            match b.exec(id, params).await {
                Ok(v) => print(&a, &v),
                Err(e) => fail(format!("{id}: {e}")),
            }
            b.finish(&a).await;
        }
        "run" => {
            let script = a.pos(1).unwrap_or_else(|| usage("run <script.jsonl | ->"));
            let text = if script == "-" {
                std::io::read_to_string(std::io::stdin()).unwrap_or_else(|e| fail(e))
            } else {
                std::fs::read_to_string(script).unwrap_or_else(|e| fail(format!("{script}: {e}")))
            };
            let mut b = Backend::open(&a);
            let mut failed = 0;
            for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty() && !l.trim_start().starts_with('#')) {
                let v: Value = serde_json::from_str(line).unwrap_or_else(|e| usage(format!("line {}: {e}", n + 1)));
                let id = v["id"].as_str().unwrap_or_else(|| usage(format!("line {}: missing \"id\"", n + 1)));
                match b.exec(id, v.get("params").cloned().unwrap_or(json!({}))).await {
                    Ok(r) => outln!("{}", json!({"line": n + 1, "id": id, "ok": true, "result": r})),
                    Err(e) => {
                        outln!("{}", json!({"line": n + 1, "id": id, "ok": false, "error": e}));
                        failed += 1;
                        if !a.flag("--keep-going") {
                            std::process::exit(1);
                        }
                    }
                }
            }
            b.finish(&a).await;
            if failed > 0 {
                std::process::exit(1);
            }
        }
        "inspect" => {
            let mut b = Backend::open(&a);
            let what = a.pos(1).unwrap_or("all");
            let mut get = async |id: &str| b.exec(id, json!({})).await.unwrap_or_else(|e| fail(e));
            let v = match what {
                "project" => get("project.inspect").await,
                "sequence" => get("sequence.inspect").await,
                "all" => json!({"project": get("project.inspect").await, "sequence": get("sequence.inspect").await}),
                other => usage(format!("inspect: unknown `{other}` (project | sequence)")),
            };
            print(&a, &v);
        }
        "import" => {
            let paths: Vec<String> = a.positionals.get(1..).unwrap_or_default().iter().map(|p| absolute(p)).collect();
            if paths.is_empty() {
                usage("import <file>...");
            }
            let mut b = Backend::open(&a);
            match b.exec("file.import", json!({"paths": paths})).await {
                Ok(v) => print(&a, &v),
                Err(e) => fail(format!("import: {e}")),
            }
            b.finish(&a).await;
        }
        "export" => {
            if a.flag("--list-presets") {
                let mut b = Backend::open(&a);
                let q = a.pos(1).map(|q| json!({"query": q})).unwrap_or(json!({}));
                match b.exec("export.presets.list", q).await {
                    Ok(v) => print(&a, &v["presets"]),
                    Err(e) => fail(format!("export: {e}")),
                }
                return;
            }
            let out = a.pos(1).unwrap_or_else(|| usage("export <out> [--preset name | --format f]"));
            let mut p = json!({"path": out, "wait": true});
            match (a.opt("--preset"), a.opt("--format")) {
                (Some(preset), f) => {
                    p["preset"] = json!(preset);
                    if let Some(f) = f {
                        p["format"] = json!(f);
                    }
                }
                (None, f) => {
                    let format = f.or_else(|| format_for(out)).unwrap_or_else(|| usage("export: give --preset or --format (unknown extension)"));
                    p["format"] = json!(format);
                }
            }
            for (k, key) in [("--scale", "scale"), ("--quality", "quality")] {
                if let Some(v) = a.opt(k) {
                    p[key] = parse_value(v);
                }
            }
            if let Some(r) = a.opt("--range") {
                p["range"] = json!(r);
            }
            set_custom_bounds(&mut p, a.opt("--start"), a.opt("--end"));
            if let Some(js) = a.opt("--settings") {
                p["settings"] = serde_json::from_str(js).unwrap_or_else(|e| usage(format!("--settings: {e}")));
            }
            if a.flag("--no-audio") {
                p["audio"] = json!(false);
            }
            if let Some(v) = a.opt("--gpu-rendering") {
                p["gpuRendering"] = json!(v); // off | auto (validated by the export command)
            }
            let mut b = Backend::open(&a);
            let t0 = std::time::Instant::now();
            let r = if a.flag("--queue") {
                let mut q = p.clone();
                q["start"] = json!(true);
                b.exec("export.queue.add", q).await.and_then(|v| {
                    let failed: Vec<&Value> = v["items"].as_array().map(|i| i.iter().filter(|x| x["status"] != "done").collect()).unwrap_or_default();
                    if failed.is_empty() { Ok(v) } else { Err(format!("queue: {}", Value::Array(failed.into_iter().cloned().collect()))) }
                })
            } else {
                b.exec("file.exportMedia", p).await
            };
            match r {
                Ok(v) => {
                    print(&a, &v);
                    let written = written_paths(&v, out).join(", ");
                    check_offline(&a, &offline_media(&v), &format!("export: {written}"));
                    diag(format_args!("exported {written} in {:.1}s", t0.elapsed().as_secs_f64()));
                }
                Err(e) => fail(format!("export: {e}")),
            }
        }
        "render" => {
            if a.opt("--bridge").is_some() {
                usage("render is headless; with --bridge use the MCP `render_frame` tool or `exec` ui commands");
            }
            let Backend::Local(mut s) = Backend::open(&a) else { usage("render is headless") };
            let secs: f64 = a.opt("--seconds").and_then(|v| v.parse().ok()).unwrap_or(0.0);
            let scale: f32 = a.opt("--scale").and_then(|v| v.parse().ok()).unwrap_or(1.0);
            let out = a.opt("--out").unwrap_or("frame.png");
            s.set_playhead(filmcraft_time::Tick::from_seconds_f64(secs));
            let t0 = std::time::Instant::now();
            let (img, offline) = render_at_playhead(&s, scale).unwrap_or_else(|e| fail(e));
            let dt = t0.elapsed();
            let png = filmcraft_automation::png_rgba(img.w as u32, img.h as u32, img.over_black_rgba8(), 0).unwrap_or_else(|e| fail(e));
            std::fs::write(out, png).unwrap_or_else(|e| fail(format!("{out}: {e}")));
            let offline: Vec<Value> = offline.iter().filter_map(|o| serde_json::to_value(o).ok()).collect();
            check_offline(&a, &offline, &format!("render: {out}"));
            diag(format_args!("rendered {}x{} in {:.1} ms → {out}", img.w, img.h, dt.as_secs_f64() * 1000.0));
        }
        "mcp" => {
            if let Some(opt) = a.unknown_opt(&["--bridge", "--project", "--demo", "--data-dir"]) {
                usage(format!("unknown mcp option `{opt}`"));
            }
            let server = match a.opt("--bridge") {
                Some(addr) => filmcraft_automation::FilmcraftMcp::bridge(addr).unwrap_or_else(|e| usage(e)),
                None => match Backend::open(&a) {
                    Backend::Local(s) => filmcraft_automation::FilmcraftMcp::headless(*s),
                    Backend::Bridge(_) => usage("mcp: use --bridge ADDR to drive a running app"),
                },
            };
            if let Err(e) = server.serve_stdio().await {
                fail(format!("mcp: {e}"));
            }
        }
        other => usage(format!("unknown subcommand `{other}`")),
    }
}

/// The files an export wrote, from its result: `path` of a direct export, or each queued item's
/// `path` (the format may have changed the extension: `out.mp4` exported as APV is `out.mp4.mov`).
/// `requested` when the result names none.
fn written_paths(result: &Value, requested: &str) -> Vec<String> {
    let paths: Vec<String> = match result.get("path").and_then(Value::as_str) {
        Some(p) => vec![p.to_string()],
        None => result
            .get("items")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|i| i.get("path").and_then(Value::as_str))
            .map(str::to_string)
            .collect(),
    };
    if paths.is_empty() { vec![requested.to_string()] } else { paths }
}

/// The offline items an export result reports (`offlineMedia` of a direct export, or of each
/// queued item), once each.
fn offline_media(result: &Value) -> Vec<Value> {
    let lists = std::iter::once(result).chain(result.get("items").and_then(Value::as_array).into_iter().flatten());
    let mut out: Vec<Value> = Vec::new();
    for o in lists.filter_map(|r| r.get("offlineMedia").and_then(Value::as_array)).flatten() {
        if !out.iter().any(|x| x.get("item") == o.get("item")) {
            out.push(o.clone());
        }
    }
    out
}

/// One stderr line per offline item that rendered as placeholder frames (#1054).
fn offline_warning(offline: &[Value]) -> String {
    let s = |o: &Value, k: &str| o.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    let mut msg = format!("filmcraft-cli: warning: {} offline media item(s) rendered as placeholder frames:", offline.len());
    for o in offline {
        let id = o.get("item").and_then(Value::as_u64).map(|i| i.to_string()).unwrap_or_default();
        msg.push_str(&format!("\n  item {id} {} ({}): {}", s(o, "name"), s(o, "status"), s(o, "path")));
    }
    msg
}

/// Warn about offline media on stderr; without `--allow-offline` that fails the run (exit 1), after
/// the output (`what`) was written, so scripts don't take placeholder frames for the real picture.
fn check_offline(a: &Args, offline: &[Value], what: &str) {
    if offline.is_empty() {
        return;
    }
    diag(format_args!("{}", offline_warning(offline)));
    if !a.flag("--allow-offline") {
        fail(format!("{what} has placeholder frames for offline media; relink it (exec media.relink …) or pass --allow-offline"));
    }
}

/// The active sequence at the playhead with the offline items it drew, or the reason it cannot be
/// rendered (no sequence, bad scale).
fn render_at_playhead(s: &Session, scale: f32) -> Result<(filmcraft_engine::render::Image, Vec<filmcraft_engine::media_pool::OfflineMedia>), String> {
    s.try_render_program_with_offline_at(scale, s.playhead()).map_err(|e| e.to_string())
}

/// `--start` / `--end` ask for a custom range. A lone bound is forwarded too, so the engine reports the missing one
/// instead of the export silently running over the whole sequence.
fn set_custom_bounds(p: &mut Value, start: Option<&str>, end: Option<&str>) {
    if start.is_none() && end.is_none() {
        return;
    }
    p["range"] = json!("custom");
    if let Some(s0) = start {
        p["startSeconds"] = parse_value(s0);
    }
    if let Some(s1) = end {
        p["endSeconds"] = parse_value(s1);
    }
}

#[cfg(test)]
mod album_tests {
    use std::sync::Arc;

    use filmcraft_autocut::album::FixedAlbum;
    use filmcraft_autocut::host::{Models, ShowcaseHost};
    use serde_json::json;

    /// `showcase --album`'s analysis runs the engine job in the CLI's own session, running the
    /// job's commands itself (nothing else does in this process): the album is picked (a
    /// `FixedAlbum` here), its clip imported, and only that clip analysed — not the demo
    /// project's six; a job that cannot start, or fails, is an error.
    #[tokio::test(flavor = "multi_thread")]
    async fn album_analysis_runs_the_job_in_the_cli_session() {
        let dir = std::env::temp_dir().join(format!("filmcraft-cli-album-analysis-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let clip = dir.join("robot.mov").to_string_lossy().into_owned();
        let mut s = filmcraft_engine::Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        // a real clip of the demo footage, from our own encoder
        let export =
            json!({"path": clip, "format": "prores", "range": "custom", "startSeconds": 0, "endSeconds": 0.5, "scale": 0.125, "audio": false, "wait": true});
        s.execute("file.exportMedia", export).unwrap();
        let album = Arc::new(FixedAlbum::new(vec![clip]));
        s.autocut.host = Some(Arc::new(ShowcaseHost::with_models(Models::scripted()).with_album_source(album.clone())));
        let mut b = super::Backend::Local(Box::new(s));
        let (report, summary) = super::album_analysis(&mut b, json!({"album": "Robot showcase", "subject": "robot", "wait": false})).await.unwrap();
        assert_eq!(report["clips"].as_array().map(Vec::len), Some(1), "{report}");
        assert_eq!((report["subject"].as_str(), summary["album"]["picked"].as_u64()), (Some("robot"), Some(1)), "{summary}");
        assert_eq!(album.asked().len(), 1);
        // refused before a job starts (no subject), and a failed job (an empty album)
        assert!(super::album_analysis(&mut b, json!({"album": "Robot showcase", "wait": false})).await.unwrap_err().contains("subject"));
        let super::Backend::Local(s) = &mut b else { panic!("local") };
        s.autocut.host = Some(Arc::new(ShowcaseHost::with_models(Models::scripted()).with_album_source(Arc::new(FixedAlbum::new(vec![])))));
        let e = super::album_analysis(&mut b, json!({"album": "Nothing", "subject": "robot", "wait": false})).await.unwrap_err();
        assert!(e.contains("has no video"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod shorts_tests {
    use super::{Args, shorts_params};

    fn args(v: &[&str]) -> Args {
        Args::parse(v.iter().map(|s| s.to_string()))
    }

    /// `shorts` options become `autocut.shorts` parameters; bad values are usage errors.
    #[test]
    fn shorts_options_become_job_params() {
        let p = shorts_params(&args(&[
            "shorts",
            "talk.mp4",
            "--count",
            "2",
            "--aspect",
            "1:1",
            "--min-seconds",
            "10",
            "--title-card",
            "--no-captions",
            "--export",
            "out",
            "--upload-youtube",
            "--youtube-privacy",
            "unlisted",
            "--youtube-tags",
            "a, b,",
        ]))
        .unwrap();
        assert_eq!((p["count"].as_u64(), p["aspect"].as_str(), p["minSeconds"].as_f64()), (Some(2), Some("1:1"), Some(10.0)));
        assert_eq!((p["title"].as_bool(), p["captions"].as_bool(), p["uploadYoutube"].as_bool()), (Some(true), Some(false), Some(true)));
        assert!(p["export"].as_str().is_some_and(|e| e.ends_with("out") && std::path::Path::new(e).is_absolute()), "{p}");
        assert_eq!(p["youtube"], serde_json::json!({"privacy": "unlisted", "tags": ["a", "b"]}));
        assert_eq!(p["wait"], true);
        let defaults = shorts_params(&args(&["shorts", "talk.mp4"])).unwrap();
        assert_eq!((defaults["count"].as_u64(), defaults["captions"].as_bool()), (Some(3), Some(true)));
        for bad in [
            &["shorts", "x", "--count", "0"][..],
            &["shorts", "x", "--count", "many"],
            &["shorts", "x", "--min-seconds", "NaN"],
            &["shorts", "x", "--sequence", "main"],
            &["shorts", "x", "--upload-youtube"],
        ] {
            assert!(shorts_params(&args(bad)).is_err(), "{bad:?}");
        }
    }
}

#[cfg(test)]
mod pick_tests {
    use super::{Args, pick_searches_from};

    fn args(v: &[&str]) -> Args {
        Args::parse(v.iter().map(|s| s.to_string()))
    }

    #[test]
    fn searches_come_from_every_auto_the_queries_file_and_the_sweep() {
        let dir = std::env::temp_dir().join(format!("filmcraft-cli-queries-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("searches.txt");
        std::fs::write(&file, "# robots I filmed\nvideos Dubai robots\n\n  ROBOTS  \nvideos Tokyo 2019\n").unwrap();
        let f = file.to_string_lossy().into_owned();
        let a = args(&["photos", "pick", "--auto", "robots", "--auto", "drones", "--queries", &f, "--sweep", "2022-11..2023-01"]);
        assert_eq!(
            pick_searches_from(&a).unwrap(),
            ["robots", "drones", "videos Dubai robots", "videos Tokyo 2019", "videos November 2022", "videos December 2022", "videos January 2023"],
            "in order, comments and blanks skipped, repeats once"
        );
        // days of each month
        let a = args(&["photos", "pick", "--sweep", "2022-10", "--month-days", "13-14"]);
        assert_eq!(pick_searches_from(&a).unwrap(), ["videos October 13 2022", "videos October 14 2022"]);
        // mistakes are said, not guessed at
        for (bad, says) in [
            (vec!["photos", "pick", "--month-days", "1-5"], "goes with --sweep"),
            (vec!["photos", "pick", "--sweep", "2025..2017"], "--sweep"),
            (vec!["photos", "pick", "--sweep", "2022", "--month-days", "40"], "--month-days"),
            (vec!["photos", "pick", "--queries", "/no/such/file.txt"], "--queries"),
            (vec!["photos", "pick", "--sweep", "1900..2100", "--month-days", "1-31"], "--sweep"),
        ] {
            let e = pick_searches_from(&args(&bad)).unwrap_err();
            assert!(e.contains(says), "{bad:?}: {e}");
        }
        // a huge file is refused before it is read
        let big = dir.join("big.txt");
        std::fs::write(&big, "x\n".repeat(600_000)).unwrap();
        assert!(pick_searches_from(&args(&["photos", "pick", "--queries", &big.to_string_lossy()])).unwrap_err().contains("bytes"));
        // nothing asked: nothing to search
        assert!(pick_searches_from(&args(&["photos", "pick"])).unwrap().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod format_tests {
    #[test]
    fn render_reports_the_real_reason() {
        let mut s = filmcraft_engine::Session::default();
        assert!(super::render_at_playhead(&s, 0.5).unwrap_err().contains("sequence"));
        s.execute("file.newSequence", serde_json::json!({"width": 16, "height": 16})).unwrap();
        for scale in [0.0, -1.0, f32::NAN] {
            let e = super::render_at_playhead(&s, scale).unwrap_err();
            assert!(e.contains("finite and positive"), "{e}");
        }
        assert_eq!(super::render_at_playhead(&s, 0.5).unwrap().0.w, 8);
    }

    /// The CLI (and MCP / headless runs, which share its entry point) registers the hardware
    /// decoders at start-up.
    #[test]
    fn startup_registers_the_hardware_decoders() {
        let hardware = super::register_hardware_decoders();
        // always on macOS and Windows; on Linux when a VA-API driver or NVIDIA's driver (NVDEC) is there
        let expected = cfg!(any(target_os = "macos", target_os = "windows"))
            || (cfg!(target_os = "linux") && matches!(hardware, filmcraft_platform::Availability::Available(_)));
        assert_eq!(filmcraft_platform::registered(), expected, "{hardware:?}");
    }

    #[test]
    fn a_failing_diagnostic_stream_does_not_panic() {
        struct Closed;
        impl std::io::Write for Closed {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
        }
        super::write_diag(&mut Closed, format_args!("exported {}", "a.wav"));
        let mut ok = Vec::new();
        super::write_diag(&mut ok, format_args!("hi"));
        assert_eq!(ok, b"hi\n");
    }

    #[test]
    fn export_message_names_the_written_files() {
        use serde_json::json;
        let direct = json!({"job": 1, "path": "out-apv.mp4.mov", "result": {"path": "out-apv.mp4.mov"}});
        assert_eq!(super::written_paths(&direct, "out-apv.mp4"), ["out-apv.mp4.mov"]);
        let queued = json!({"added": [1, 2], "items": [{"path": "a.mp4.mov"}, {"path": "b.wav"}]});
        assert_eq!(super::written_paths(&queued, "a.mp4"), ["a.mp4.mov", "b.wav"]);
        assert_eq!(super::written_paths(&json!({}), "x.mov"), ["x.mov"]);
        assert_eq!(super::written_paths(&json!({"items": []}), "x.mov"), ["x.mov"]);
    }

    #[test]
    fn offline_media_from_direct_and_queued_results() {
        use serde_json::json;
        let a = json!({"item": 8, "name": "a.mp4", "path": "/m/a.mp4", "status": "missing", "error": ""});
        let b = json!({"item": 9, "name": "b.mov", "path": "/m/b.mov", "status": "offline", "error": ""});
        assert_eq!(super::offline_media(&json!({"path": "x.mp4", "offlineMedia": [a]})), std::slice::from_ref(&a));
        let queued = json!({"items": [{"offlineMedia": [a, b]}, {"offlineMedia": [a]}, {"offlineMedia": []}]});
        assert_eq!(super::offline_media(&queued), [a.clone(), b.clone()]);
        assert!(super::offline_media(&json!({"path": "x.mp4"})).is_empty());
        let w = super::offline_warning(&[a, b]);
        assert!(w.contains("2 offline media item(s)") && w.contains("item 8 a.mp4 (missing): /m/a.mp4") && w.contains("item 9 b.mov (offline)"), "{w}");
    }

    #[test]
    fn export_lone_bound_is_forwarded_as_custom_range() {
        use serde_json::json;
        let mut p = json!({});
        super::set_custom_bounds(&mut p, Some("0.25"), None);
        assert_eq!(p["range"], "custom");
        assert_eq!(p["startSeconds"], 0.25);
        assert!(p.get("endSeconds").is_none());
        let mut p = json!({});
        super::set_custom_bounds(&mut p, None, Some("0.25"));
        assert_eq!(p["range"], "custom");
        assert_eq!(p["endSeconds"], 0.25);
        let mut p = json!({});
        super::set_custom_bounds(&mut p, None, None);
        assert!(p.get("range").is_none());
    }

    /// `import` used `canonicalize`, which resolves symlinks and on Windows stores `\\?\C:\…`
    /// verbatim paths in the project (#850).
    #[test]
    fn imported_paths_are_absolute_without_resolving_links() {
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(super::absolute("clip.mov"), cwd.join("clip.mov").to_string_lossy());
        assert!(!super::absolute("clip.mov").starts_with(r"\\?\"));
        assert_eq!(super::absolute(""), "");
        #[cfg(unix)]
        {
            let dir = std::env::temp_dir().join(format!("filmcraft-cli-abs-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("real.mov"), b"x").unwrap();
            let link = dir.join("link.mov");
            let _ = std::fs::remove_file(&link);
            std::os::unix::fs::symlink(dir.join("real.mov"), &link).unwrap();
            assert_eq!(super::absolute(&link.to_string_lossy()), link.to_string_lossy());
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }

    #[test]
    fn export_format_from_extension() {
        assert_eq!(super::format_for("out/clip.MXF"), Some("mxf-op1a"));
        assert_eq!(super::format_for("a.mov"), Some("prores"));
        assert_eq!(super::format_for("noext"), None);
    }
}
