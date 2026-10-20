# Natural spoken Malayalam

A Malayalam narration script written like a newspaper or a textbook sounds robotic when a voice
reads it, however good the voice. Write it the way a Malayali tech creator talks to friends on
YouTube: short sentences, the English tech words people actually say (written in Malayalam
script), spoken verb endings, numbers said the English way.

This page is the style. FilmCraft checks it for you (`filmcraft-cli ml-lint`, the
`script.lintMalayalam` command), can write a first draft from the English plan with the local
model (`filmcraft-cli ml-script`, `autocut.malayalamScript`), and a Malayalam showcase checks its
script before voicing it. The voice and the numbers' pronunciation are in
[malayalam-narration.md](malayalam-narration.md).

## The rules

1. **Short sentences.** One idea per sentence, at most about 12 spoken words. A year or a price
   said the English way ("ട്വന്റി ട്വന്റി വൺ") and an acronym spelled letter by letter ("ആർ ടി
   എ") count as one word. Split a long sentence at its commas; the lint flags more than 14.
2. **English tech words, in Malayalam script.** Where a Malayali creator says the English word, write
   it in Malayalam script:

   | written Malayalam | what people say |
   |---|---|
   | സ്വാഗതം ചെയ്യുന്നു | വെൽക്കം ചെയ്യുന്നു |
   | സന്ദർശകർ, സന്ദർശകരെ | വിസിറ്റേഴ്സ്, വിസിറ്റേഴ്സിനെ |
   | രൂപകൽപ്പന ചെയ്തു | ഡിസൈൻ ചെയ്തു |
   | അവതരിപ്പിച്ചു, പുറത്തിറക്കി | ലോഞ്ച് ചെയ്തു |
   | വികസിപ്പിക്കാൻ | ഡെവലപ്പ് ചെയ്യാൻ |
   | നിയന്ത്രിക്കാം | കൺട്രോൾ ചെയ്യാം |
   | തരംതിരിക്കും | സോർട്ട് ചെയ്യും |
   | വേഗത, പരമാവധി | സ്പീഡ്, മാക്സിമം |
   | മുഖഭാവങ്ങൾ, വികാരങ്ങൾ | എക്സ്പ്രഷൻസ്, ഇമോഷൻസ് |
   | നിർമ്മാണം (production) | പ്രൊഡക്ഷൻ |
   | ഗവേഷകർ | റിസർച്ചേഴ്സ് |
   | ഭക്ഷണം, തലച്ചോർ | ഫുഡ്, ബ്രെയിൻ |
   | റോബോട്ട് നായ, പടികൾ | റോബോട്ട് ഡോഗ്, സ്റ്റെപ്പുകൾ |
   | നന്ദി | താങ്ക്സ് |

   **When not to.** Everyday Malayalam words stay Malayalam: ആളുകൾ, കുട്ടികൾ, വെള്ളം, വഴി, ഉയരം,
   കണ്ണ്, കൈ, വീട്. "ആളുകളെ വെൽക്കം ചെയ്യുന്നു", never "പീപ്പിളിനെ". The test: would you say the
   English word to a friend in Kerala while showing them the video? Use it only then.
3. **Spoken forms, not written ones.**

   | written | spoken |
   |---|---|
   | നിർമ്മിച്ചു, നിർമ്മിച്ചതാണ് | ഉണ്ടാക്കി, ഉണ്ടാക്കിയതാണ് |
   | ലഭിക്കും, ലഭിച്ചു | കിട്ടും, കിട്ടി |
   | സാധിക്കും, സാധിക്കില്ല | പറ്റും, പറ്റില്ല |
   | ആരംഭിച്ചു | തുടങ്ങി |
   | യഥാർത്ഥ, യഥാർത്ഥത്തിൽ | ശരിക്കുള്ള, ശരിക്കും |
   | കണ്ടെത്തും | കണ്ടുപിടിക്കും |
   | ചുറ്റിനടന്നു | കറങ്ങി നടന്നു |
   | -ക്കായി, -ിനായി | -ക്ക് വേണ്ടി, -ിന് വേണ്ടി |
   | അതിനാൽ, എന്നാൽ | അതുകൊണ്ട്, പക്ഷേ |
   | പറയൂ, ചെയ്യൂ | പറയണേ, ചെയ്യണേ |

4. **Spoken endings and particles**, where they fit (not in every sentence): ഉണ്ടാക്കിയതാ (for
   ഉണ്ടാക്കിയതാണ്), "…എന്നേ തോന്നൂ", "…യേ ഉള്ളൂ", "ഇവനാണ്" (a robot as "this guy"), "കമന്റിൽ
   പറയണേ"; പിന്നെ (then, next), ഒക്കെ (and such: "ഡാൻസ് ഒക്കെ ചെയ്യും").
5. **Numbers as digits.** Years, prices and numbers from 100 up are said the English way, the way
   Malayali creators say them, and the voice does that from digits: 2021 → ട്വന്റി ട്വന്റി വൺ,
   $1,600 → സിക്സ്റ്റീൻ ഹണ്ട്രഡ് ഡോളർ, 170,000 → വൺ ലാഖ് സെവന്റി തൗസന്റ്. Small counts are fine as
   Malayalam words (രണ്ട്, പത്ത്). Never long Malayalam number words: "രണ്ടായിരത്തി
   ഇരുപത്തിയൊന്ന്" is hard to listen to and blurs the start of a line. A suffix goes after a
   hyphen: 2013-ൽ, 600-ലധികം. Writing the English words yourself ("ട്വന്റി ട്വന്റി വൺ") is fine too.
6. **Acronyms letter by letter**, with spaces: എ ഐ, ആർ ടി എ, യു എ ഇ, ക്യു ആർ. Codes take the
   digits' English names: 4K → ഫോർ കെ, M02 → എം സീറോ ടു, V.4 → വി ഫോർ.
7. **Names by their sound**, in Malayalam script: GITEX → ജൈടെക്സ് (it is said "Jai-tex"), Neolix
   → നിയോലിക്സ്, SoftBank → സോഫ്റ്റ്ബാങ്ക്. No Latin letters anywhere in the script: the voice
   reads Malayalam script.
8. **Facts never change.** Every number, name, place, unit and month of the English line stays,
   and nothing is added. Reordering for rhythm is good ("ട്വന്റി ട്വന്റി വൺ ഒക്ടോബറിൽ ജൈടെക്സിൽ
   വെച്ചാണ് … കണ്ടത്"); dropping a place or "an hour" is not.
9. **A sentence without a verb** is fine on purpose ("പന്ത്രണ്ട് വർഷം, നാൽപ്പത്തിയെട്ട്
   റോബോട്ടുകൾ."), not by accident: a run of them reads like a list.

## Before and after

| written (robotic) | spoken |
|---|---|
| ഈ റോബോട്ടിനെ ജപ്പാനിൽ രൂപകൽപ്പന ചെയ്ത് നിർമ്മിച്ചതാണ്, രണ്ടായിരത്തി ഇരുപത്തിരണ്ടിൽ ഏകദേശം മൂവായിരം ഡോളറിനാണ് ഇത് പുറത്തിറക്കിയത്. | ഈ റോബോട്ട് ജപ്പാനിലാണ് ഡിസൈൻ ചെയ്ത് ഉണ്ടാക്കിയത്. 2022-ൽ ലോഞ്ച് ചെയ്തു, വില ഏകദേശം 3,000 ഡോളർ. |
| എവിടെ നോക്കിയാലും സന്ദർശകരെ സ്വാഗതം ചെയ്യുന്ന ഈ റോബോട്ടിന് മുഖഭാവങ്ങൾ മനസ്സിലാക്കാൻ സാധിക്കും. | എവിടെ നോക്കിയാലും വിസിറ്റേഴ്സിനെ വെൽക്കം ചെയ്യുന്നത് ഇവനാണ്. നമ്മുടെ എക്സ്പ്രഷൻസ് ഒക്കെ ഇവന് മനസ്സിലാകും. |
| ലേസറുകളും ക്യാമറകളും ഉപയോഗിച്ച് വഴി കണ്ടെത്തി, പല ട്രേകളിലായി വിഭവങ്ങൾ കൊണ്ടുപോകുന്ന ഒരു സെർവിംഗ് റോബോട്ട് ആണ് ഇത്. | ഇതൊരു സെർവിംഗ് റോബോട്ടാണ്. പല ട്രേകളിലായി ഫുഡ് കൊണ്ടുപോകും. ലേസറും ക്യാമറയും വെച്ചാണ് വഴി കണ്ടുപിടിക്കുന്നത്. |
| ഇതിന്റെ AI, 4K ക്യാമറ ഉപയോഗിച്ച് മണിക്കൂറിൽ പതിനായിരത്തിലധികം പാഴ്സലുകൾ തരംതിരിക്കും. | ഇതിന്റെ എ ഐ, ഫോർ കെ ക്യാമറ വെച്ചാണ് പണിയെടുക്കുന്നത്. മണിക്കൂറിൽ 10,000-ലധികം പാഴ്സലുകൾ സോർട്ട് ചെയ്യും. |
| കണ്ടതിന് നന്ദി, ഏത് റോബോട്ടിനെയാണ് നിങ്ങൾക്ക് ഏറ്റവും ഇഷ്ടപ്പെട്ടത് എന്ന് കമന്റിൽ പറയൂ. | കണ്ടതിന് താങ്ക്സ്. ഇതിൽ ഏതാണ് നിങ്ങൾക്ക് ഏറ്റവും ഇഷ്ടപ്പെട്ടത്? കമന്റിൽ പറയണേ. |
| (English: "capped at twenty kilometres an hour") സിറ്റിയിൽ ഇതിന് മാക്സിമം ഇരുപത് കിലോമീറ്റർ സ്പീഡേ ഉള്ളൂ. — natural, but "an hour" is lost | സിറ്റിയിൽ ഇതിന് മണിക്കൂറിൽ മാക്സിമം ഇരുപത് കിലോമീറ്റർ സ്പീഡേ ഉള്ളൂ. |

With the English lines, the lint finds 14 warnings (and the lost "an hour") in the left column
and none in the right one.

## Checking a script: `ml-lint`

```sh
filmcraft-cli ml-lint script-ml.json --plan plan.json            # findings, and facts against the English
filmcraft-cli ml-lint script-ml.json --fix --out script-ml-fixed.json   # the safe replacements made
filmcraft-cli exec script.lintMalayalam '{"text": "സന്ദർശകരെ സ്വാഗതം ചെയ്യുന്നു."}'
```

The script is a lines file (`{"intro", "outro", "lines": {"<entry label>": "…"}}`, what
`showcase-plan --lines` takes), a Malayalam plan, or a text file. Each finding names its part
(intro, outro, an entry's label) and says what to say instead:

| kind | level | what |
|---|---|---|
| `longSentence` | warning | more than `--max-words` (14) spoken words; where to split |
| `formal` | warning or hint | a written form with a spoken one (the tables above; the weaker cases, like ഭക്ഷണം → ഫുഡ്, are hints) |
| `numberWords` | warning | a number from 100 up in Malayalam words; the digits, and how the voice says them |
| `latin` | warning | Latin letters; an acronym or code spelled out (`AI` → എ ഐ), a name to write by its sound |
| `noVerb` | hint | a sentence with no verb ending |

With `--plan` each line is also checked against its English line: every number (compared by
value, whichever way either side writes it), name (by sound: "Neolix" matches
"നിയോലിക്സിന്റെ"; the plan's `pronounce` gives the sound, "GITEX" → "Jai-tex"), acronym, unit
and month must be there, and no number may be added. `--fix` makes only the replacements that
keep the meaning and the word's suffix exactly (സന്ദർശകരെ → വിസിറ്റേഴ്സിനെ, നിർമ്മിച്ചതാണ് →
ഉണ്ടാക്കിയതാണ്, അറുന്നൂറിലധികം → 600-ലധികം, UAE-യിൽ → യു എ ഇ-യിൽ); the rest is for a person.
`--hints` lists the hints too; `--strict` exits 1 unless the script is clean.

Measured on a real 52-part showcase script (`cargo test -p filmcraft-engine ml_script -- --ignored`
with the files): the first, robotic draft got 54 warnings (19 long sentences, 30 formal words, 5
number words) and the hand rewrite that sounded natural got 7 (2 long sentences, 5 numbers from
150 up still in Malayalam words), no formal word. Against the English plan, the fact lock found
the three facts the rewrite had dropped (a state, a city, "an hour") and nothing else in either.

## A first draft: `ml-script`

```sh
filmcraft-cli ml-script --plan plan.json --out script-ml.json [--model qwen3:8b] [--report-out report.json]
filmcraft-cli exec autocut.malayalamScript '{"planPath": "plan.json", "out": "script-ml.json", "wait": true}'
```

The local script model (`AUTOCUT_SCRIPT_MODEL`, `qwen3:8b`) translates the plan's intro, lines
and outro a few at a time, given this style, built-in examples and your approved pairs, each line
with the facts it must keep. Every line it writes gets the safe fixes, the lint and the fact
check; a line with a warning or a lost or added fact goes back to the model once with the list of
what to fix, and the better version is kept. The output is the lines file; what is left to look at
is printed (and in `--report-out`). Parts the model could not write are left out, and
`showcase-plan --lines` names them.

**How good is the draft?** Measured 2026-10-20 on the 52-part showcase plan above, with
`qwen3:8b` on an M3 Max shared with another lane (25 minutes; 23 safe fixes made, 23 lines sent
back, 5 rewrites kept): every part came back, short sentences in the right register, digits and
spelled acronyms, no lint warning. But the Malayalam itself is weak. The fact lock flagged 19
parts (the hand-written script: 3), for real errors: fourteen kilograms became twelve, "seven or
eight kilos" seven or five, five-finger hands ten-finger ones, "six hundred" nonsense, a year or a
city gone. Other lines are fluent-looking but wrong where no fact is checked: the yellow robot
turned green, "Japan's first home-grown" robot became "Japan's first Indian" one, a garbled
sentence about NAO. By a native-reader standard, perhaps one line in five is usable after a light
edit. So treat it as a rough draft at most: the style guide, `ml-lint` and the fact lock are the
dependable parts. A capable writer (you, or the agent driving the editor) following this page,
then `ml-lint --plan` on the result, is the way to a script that sounds like you.

### Teaching it your voice

Approved lines make the next draft sound more like you:

```sh
filmcraft-cli ml-script --plan plan.json --remember script-ml.json
```

adds each English line and your reviewed Malayalam to `<data>/autocut/malayalam-examples.json`
(`<data>` is FilmCraft's data directory). The file is yours, on your machine, never in the
repository; the newest pairs are shown to the model first. `--examples pairs.json` uses another
file (`{"pairs": [{"en": "…", "ml": "…"}]}`).

## In a showcase

`showcase --language ml` and `showcase --languages en,ml --lines-ml script-ml.json` check the
Malayalam script before anything is voiced, against the English plan when the lines are read over
it, and print the findings as warnings. `--strict-script` makes them errors: the build stops
before the first line is synthesized.

## The table

The formal → spoken table is `FORMAL` in
[`crates/tts/src/ml_script.rs`](../crates/tts/src/ml_script.rs): a form, how it matches (the
whole word, its start for inflected forms, its end for a suffix), what people say, and the
replacement `--fix` may make. Keep it conservative: add a word only when the spoken form is what
people say in a tech video, make it a hint when it depends on context, and give it a fix only when
the replacement takes the same suffixes.
