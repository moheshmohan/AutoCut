# Recording a voice for cloning

A cloned voice copies its recording: the speaker's timbre, accent and rhythm, and also the room,
the microphone and whatever was playing in the background. FilmCraft's natural clones speak
English through VoxCPM2 (from a 10–30 s voice sample) and Malayalam through IndicF5 (from a
6–12 s clip and its exact words), so a better recording makes a better voice in both languages.
This page says how to record a seed for a future re-clone, and how to make the voice from it
with the CLI.

The first clone (`clone:mohesh-natural`) came from one phone clip of a conversation about AI
models (`IMG_3337.MOV`): Malayalam speech about 13 dB over a steady background, used as 24.5 s
for English and 9.3 s for Malayalam. It works, but every reading carries that room, and VoxCPM2
speaks English from Malayalam speech. The voice lab ([below](#better-references-from-clips-you-already-have))
finds and auditions better stretches in clips you already have — in the author's library (729
clips) it found his voice in only three, and the best cuts, other stretches of the same session,
gained little ([measured](../crates/tts/README.md#better-references-the-authors-library-measured-2026-10-21)).
A recording made for the purpose is the way to a better clone.

## The room and the phone

- **A quiet, soft room.** A bedroom with curtains, a bed, a rug and a full wardrobe is ideal;
  avoid bare walls, tiles, glass and empty rooms (they ring). Close the windows. Turn off fans,
  the air conditioner and anything that hums; no TV, no music, nobody talking nearby.
- **The phone about 20 cm from your mouth** (a hand's span), a little below and to the side of
  it, so your breath does not hit the microphone. A stand or a pile of books keeps the distance
  steady; do not hold it at arm's length.
- **The phone's own settings**: record video or a voice memo at its best quality, and turn off any
  voice isolation, studio or noise-reduction mode (they change the voice). Airplane mode, so no
  call or notification interrupts.
- **Level**: speak as loudly as you do on camera. Record ten seconds, play it back on headphones:
  your voice clear and close, no crackle on loud words, the pauses silent.

## How to speak

- **Natural and conversational**, the way you talk to a friend about robots — the delivery you
  want the narration to have. Not a flat reading, not a performance.
- **Whole sentences with a short pause (half a second to a second) after each.** The tools cut
  references at pauses where a sentence ends; a clip that stops mid-sentence garbles IndicF5's
  first words.
- **About 60 s of English, a 3 s pause, then about 60 s of Malayalam**, in one take. Then read
  both again (a second take): the finder picks the best stretches of either.
- If you stumble, pause, and say the whole sentence again.

## Reading script: English (about 60 s)

> Hi everyone, and welcome back. Today I'm walking around one of the biggest technology shows I
> have been to, and honestly, it's a lot to take in. There are robots everywhere. Some are small
> enough to sit on my palm, and some are industrial arms that could lift a small car. This one
> is a robot dog. It climbs stairs, it balances on three legs, and the engineers say it can open
> doors by itself. What surprised me most is how quickly people stop staring and start talking
> to these machines. Over there, a humanoid just waved at a group of kids, and they waved back.
> I'll be honest: I didn't expect to feel anything about a robot. Anyway, let's keep moving.
> There's a lot more to see before the hall closes, and I've saved my favourite for the end.

## Reading script: Malayalam (about 60 s)

Spoken Malayalam, as the narration is written ([malayalam-style.md](malayalam-style.md)): English
words as Malayali tech creators say them, in Malayalam script, and years said the English way.

> ഹായ് ഫ്രണ്ട്സ്, എല്ലാവർക്കും സുഖമല്ലേ? ഇന്ന് നമ്മൾ ഒരു അടിപൊളി റോബോട്ട് എക്സ്പോയിലേക്കാണ്
> പോകുന്നത്. ഇവിടെ ഒരുപാട് റോബോട്ടുകൾ ഉണ്ട്. ചിലത് ചെറുതാണ്, ചിലത് ഭയങ്കര വലുതും. ഈ റോബോട്ട്
> ഡോഗിനെ കണ്ടോ? ഇത് സ്റ്റെപ്പ് കയറും, ഡാൻസ് ചെയ്യും, പിന്നെ നമ്മളെ ഫോളോ ചെയ്യും. സത്യം
> പറഞ്ഞാൽ, ഞാൻ ഇത്രയും പ്രതീക്ഷിച്ചില്ല. ട്വന്റി ട്വന്റി ഫോറിലാണ് ഞാൻ ആദ്യമായി ഒരു റോബോട്ട്
> ഡോഗിനെ കാണുന്നത്. ഇതിന്റെ ബാറ്ററി ഏകദേശം രണ്ട് മണിക്കൂർ നിൽക്കും എന്നാണ് അവർ പറഞ്ഞത്. ഓക്കെ,
> ഇനി നമുക്ക് അടുത്ത സ്റ്റാളിലേക്ക് പോകാം. അവിടെ ഒരു ഹ്യൂമനോയിഡ് റോബോട്ട് ഉണ്ട്, അത് നമ്മളോട്
> സംസാരിക്കും. വീഡിയോ ഇഷ്ടപ്പെട്ടെങ്കിൽ ലൈക്ക് ചെയ്യണേ, സബ്സ്ക്രൈബ് ചെയ്യാൻ മറക്കല്ലേ!

Keep the script text: IndicF5 needs the exact words of its clip, and with a script you have them
(the speech model's transcript of spontaneous speech is only close).

## Making the voice from the recording

Copy the recording to the Mac (here `~/Movies/voice-seed/IMG_1234.MOV`). The voice lab finds the
best stretches, auditions them against your genuine speech, and puts the one you choose in place
— the voice keeps its name, its consent and the reference it had before
([`filmcraft_engine::voice_lab`](../crates/engine/src/voice_lab.rs)):

```sh
# 1. find: stretches by the voice's speaker, ranked per engine (refs.json, candidates/*.wav)
filmcraft-cli voice refs --voice clone:mohesh-natural ~/Movies/voice-seed/IMG_1234.MOV \
  --out ~/Movies/voice-seed/lab

# 2. audition: the current reference and the best candidates read a fixed test set; listen to the
#    files README.txt lists (WAV and M4A), compare the numbers
filmcraft-cli voice audition --voice clone:mohesh-natural --report ~/Movies/voice-seed/lab/refs.json --takes 3

# 3. keep the ones you prefer: the English voice sample, and the Malayalam clip with the exact
#    sentences of the script it covers (audition.json shows what the speech model heard)
filmcraft-cli voice promote --voice clone:mohesh-natural --report ~/Movies/voice-seed/lab/audition.json voxcpm-1
filmcraft-cli voice promote --voice clone:mohesh-natural --report ~/Movies/voice-seed/lab/audition.json indicf5-1 \
  --transcript "ഇവിടെ ഒരുപാട് റോബോട്ടുകൾ ഉണ്ട്. ചിലത് ചെറുതാണ്, ചിലത് ഭയങ്കര വലുതും."

# changed your mind: every earlier reference is kept
filmcraft-cli voice versions --voice clone:mohesh-natural
filmcraft-cli voice rollback --voice clone:mohesh-natural r0
```

A promoted reference is read again from the original recording at full band (44.1 kHz). The
narration cache keys every reading on the reference's bytes, so nothing read with the old
reference is reused, and a rollback finds its own readings again. Then check a Malayalam line
the usual way (`exec tts.preview`, the strict check; [malayalam-narration.md](malayalam-narration.md)).

A new voice from scratch (another speaker, with their consent) is made as before:
`tts.cloneVoice` with the English stretch as `segments`, then the voice lab as above for its
Malayalam reference.

## Better references from clips you already have

The same three steps work on your library: `voice refs` takes files, folders and `.fcproj`
projects (their source media), skips clips longer than `--max-clip-seconds` (600) and stops after
`--max-minutes` (240) of audio, and never looks at FilmCraft's own sound (narrations, generated
audio, the voices) or AutoCut's renders. Clips where the speaker does not match (tone-colour
similarity under `--min-similarity`, 0.85) are left out; everyone else's voice is never a
candidate. Heavy models run under the `gpu` lock (`filmcraft-cli locks`).

## Delivery styles: a reference for each way you speak

IndicF5 continues its reference clip, so the clip's delivery carries into every line read from
it: a flat clip reads flat. The style finder looks through your clips for Malayalam references
(whole phrases of 6–12 s by you, ending at a sentence end) and sorts them by how you speak in
them — **calm, plain, conversational, warm, excited, amazed, curious** — so each style of
narration can have its own reference
([`filmcraft_tts::voice_lab::styles`](../crates/tts/src/voice_lab/styles.rs)):

```sh
# the clips of a project (its media items: --items with --project) and any other files or folders
filmcraft-cli voice styles --voice clone:mohesh-natural --project "$HOME/Movies/AutoCut/3D Printing Saga/3D Printing Saga.fcproj" \
  --items 1662,1666,1668 ~/Movies/voice-seed --top 3 --out ~/Movies/voice-seed/styles
```

The speaker check is the voice lab's (tone-colour similarity to the voice's reference, 0.85),
and similarity follows the room and the microphone as well as the voice: the author's 3D
printing voice-overs and the current Malayalam reference's own source clip score 0.72–0.84
against a reference made from another session. Give stretches of such recordings as
`--speaker-ref file@start-end` (their tone colour is averaged with the voice's reference) and,
for clips you know are yours, a lower `--min-similarity`.

Each candidate's prosody is measured (`filmcraft_audio_dsp::prosody`: pitch level, pitch range
from the 10th to the 90th percentile, pitch movement, energy variation over syllable peaks,
syllable rate, the final contour) and placed against **your own spread** over all your candidates
(robust z-scores, leaning on your measured speech while there are few), so the styles are yours:
excited is lively *for you*. The rules (the module docs have the weights): excited = wide range,
much movement, punchy syllables, faster, higher; amazed = wide and high but slower; curious = a
rising end; calm = narrow, slow and even; plain = flat at an ordinary pace; warm = lower,
unhurried, still melodic; conversational = your middle. `dramatic` is not told apart by prosody
alone. Each candidate gets a share per style (they add up to 1) and its confidence is its own
style's share; a style your clips do not hold stays empty rather than offering a poor match.

The best `--top` per style are heard by both Malayalam speech models the strict check uses
(the Malayalam Whisper and IndicWhisper; `--no-hear` skips them) and written to
`styles/<style>-<rank>.wav`. `styles.json` lists each pick with what a style reference needs —
`path` (and `item` when it came from one), `start`, `end`, `transcript`, `style` — and why:
`confidence`, `features`, `z`, every style's `shares`, what each model heard and their
`agreement` (1 − the character error rate between them: low agreement means check the words
by ear before using them), the speaker `similarity` and signal `quality`. Listen, correct the
words if needed, and make it the voice's reference for that style (`tts.voiceReference` with
`style`).

`voice prosody` measures any recording the same way and can reshape its intonation (TD-PSOLA:
the same length and words, unvoiced sounds untouched):

```sh
# how lively is a narration, and the same with its pitch range scaled 1.3× and 1.6×, heard again
filmcraft-cli voice prosody "Narration 1.wav" --segment "IMG_4184.MOV@12.0-21.5" \
  --range 1.3,1.6 --out ~/Movies/voice-seed/shaped --hear
```

`--shift` (semitones), `--question` (semitones of rise over the last half second, ending at
`--question-end`) and `--emphasis` (dB on stressed syllables) shape it further. With `--hear` both
Malayalam models hear the original and each reshaped version, and the character error rate
between the two readings shows whether the words survived.

## Measures

- **Speaker similarity**: the cosine of OpenVoice V2 tone colours (1 = the same timbre). Over
  the author's library (729 clips, 569 minutes): stretches of his own speech scored 0.85–0.95
  against the voice's reference, other people at most 0.80 — and the cloned voice itself, in a
  render's narration, 0.82–0.86, which is why renders and their copies are never looked at.
- **Signal**: speech over the floor of the pauses around it (dB), clipping, a decay-time
  estimate after word ends (reverberation; measured only on a clean recording, about 25 dB of
  signal to noise or more), clarity (the contrast between syllables and their dips), tonal
  background in the pauses (music or a hum), bandwidth.
- **Whole sentences**: the edges where the words heard end a sentence, else at a long pause.
- **Audition**: each reading's similarity to held-out genuine speech (never the reference
  itself), what a speech model hears (character error rate; the first syllables of each
  sentence on their own, where IndicF5's garble is) and the signal scan (clipping, dropouts,
  bursts).
