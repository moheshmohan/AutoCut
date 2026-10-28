use super::*;

/// Lines of a robot showcase as first written (formal, read like an article: it sounded robotic)
/// and as rewritten in natural spoken Malayalam, the same facts. Places and people are left out.
const WRITTEN: [&str; 13] = [
    "എവിടെ നോക്കിയാലും ആളുകളെ സ്വാഗതം ചെയ്യുന്ന പെപ്പർ. മുഖത്തെയും ശബ്ദത്തിലെയും വികാരങ്ങൾ മനസ്സിലാക്കാൻ സോഫ്റ്റ്ബാങ്ക് ഉണ്ടാക്കിയ റോബോട്ട് ആണ്, ട്വന്റി ട്വന്റിയിൽ നിർമ്മാണം നിർത്തുന്നതിന് മുമ്പ് ഏകദേശം ഇരുപത്തിയേഴായിരം എണ്ണം ഉണ്ടാക്കി.",
    "പിന്നെ, സ്പോട്ട്! ബോസ്റ്റൺ ഡൈനാമിക്സിന്റെ ഈ റോബോട്ട് നായ പടികളും ചരിവുകളും കയറും, പുറത്ത് പതിനാല് കിലോ വരെ ഭാരം ചുമക്കാനും പറ്റും.",
    "ഈ മഞ്ഞ ഭീമൻ തെർമൈറ്റ് ഫയർഫൈറ്റിംഗ് റോബോട്ട് ആണ്. മിനിറ്റിൽ ടു തൗസന്റ് ഫൈവ് ഹണ്ട്രഡ് ഗാലൻ വെള്ളം പമ്പ് ചെയ്യും, ഡീസലിൽ ഇരുപത് മണിക്കൂർ ഓടും.",
    "കുഞ്ഞൻ നാവോയ്ക്ക് അറുപത് സെന്റിമീറ്ററിൽ താഴെ മാത്രം ഉയരം, ഇരുപത്തിയഞ്ച് ചലിക്കുന്ന ജോയിന്റുകൾ. ടു തൗസന്റ് എയ്റ്റ് മുതൽ കുട്ടികളെയും ഗവേഷകരെയും പഠിപ്പിക്കുന്നു.",
    "ഈ റോബോട്ട് യഥാർത്ഥ ബ്രഷുകൾ കൊണ്ട് ഒരു യഥാർത്ഥ ക്യാൻവാസിൽ ഓരോ കുത്തായി വരയ്ക്കുകയായിരുന്നു. ആം ഒരു യു ഫാക്ടറി എക്സ് ആം ആണ്, എന്ത് വരയ്ക്കണം എന്ന് തീരുമാനിക്കുന്നത് ഒരു എ ഐ ആണ്.",
    "ആപ്പ് വഴി ഓർഡർ ചെയ്തവർക്ക് ഒട്ടോണമിയുടെ ഒട്ടോബോട്ട് ഡ്രിങ്ക്സ് എത്തിച്ചു കൊടുത്തു. സിൻസിനാറ്റി പോലുള്ള എയർപോർട്ടുകളിൽ അപ്പോൾ തന്നെ ഇത് ആൾക്കൂട്ടത്തിനിടയിലൂടെ തനിയെ വഴി കണ്ടെത്തി ഡെലിവറി ചെയ്യുന്നുണ്ടായിരുന്നു.",
    "പിന്നെ അമേക്ക, ഞാൻ കണ്ടതിൽ വെച്ച് ഏറ്റവും ഭാവങ്ങൾ ഉള്ള റോബോട്ട് മുഖം. ബ്രിട്ടനിലെ എഞ്ചിനീയേർഡ് ആർട്സ് ട്വന്റി ട്വന്റി വണ്ണിന്റെ അവസാനത്തിലാണ് ഇതിനെ അവതരിപ്പിച്ചത്.",
    "ഒരു മാസം കഴിഞ്ഞ്, ഈ റോബോട്ട് ആം മണൽ ചേർത്ത ബയോഡീഗ്രേഡബിൾ പ്ലാസ്റ്റിക് കൊണ്ട് ശില്പങ്ങൾ ത്രീ ഡി പ്രിന്റ് ചെയ്യുകയായിരുന്നു. ബ്രണോ യൂണിവേഴ്സിറ്റി ഓഫ് ടെക്നോളജി ആണ് ഇത് വികസിപ്പിക്കാൻ സഹായിച്ചത്.",
    "തലയ്ക്കുള്ളിലെ ഒരു പ്രൊജക്ടർ ഒരു മാസ്കിലേക്ക് മുഖം പ്രൊജക്റ്റ് ചെയ്യും, അതുകൊണ്ട് ഒരു പേശി പോലും അനക്കാതെ മുഖം മാറ്റാം.",
    "ഇതുപോലുള്ള റോബോട്ടുകൾ പല ട്രേകളിലായി വിഭവങ്ങൾ കൊണ്ടുപോകും, ലേസറുകളും ക്യാമറകളും ഉപയോഗിച്ച് വഴി കണ്ടെത്തും.",
    "ഇത് ഓപ്റ്റി, ഓപ്പർച്യൂണിറ്റി പവലിയന്റെ മാസ്കോട്ട്. ഏകദേശം ഒരു മീറ്റർ ഉയരം, ടെർമിനസ് ആറ് മാസത്തിനുള്ളിൽ ഇതുപോലെ അൻപതെണ്ണം ഉണ്ടാക്കി.",
    "കണ്ടതിന് നന്ദി, ഇതിൽ ഏത് റോബോട്ടിനെ കാണാനാണ് നിങ്ങൾക്ക് ഏറ്റവും ഇഷ്ടം എന്ന് കമന്റിൽ പറയൂ.",
    "ചെറിയ ചുവന്ന റോബോട്ടുകൾ പാഴ്സലുകൾ അവിടെ മുഴുവൻ കൊണ്ടുനടന്നു. ഒരു റോബോട്ട് ഡിപ്പോ മണിക്കൂറിൽ പതിനായിരത്തിലധികം പാഴ്സലുകൾ തരംതിരിക്കും.",
];

const SPOKEN: [&str; 13] = [
    "എവിടെ നോക്കിയാലും ആളുകളെ വെൽക്കം ചെയ്യുന്ന പെപ്പർ. മുഖത്തെയും ശബ്ദത്തിലെയും ഇമോഷൻസ് മനസ്സിലാക്കാനാണ് സോഫ്റ്റ്ബാങ്ക് ഇതിനെ ഉണ്ടാക്കിയത്. ട്വന്റി ട്വന്റിയിൽ പ്രൊഡക്ഷൻ നിർത്തുന്നതിന് മുമ്പ് ഏകദേശം ഇരുപത്തിയേഴായിരം എണ്ണം ഇറക്കി.",
    "പിന്നെ, സ്പോട്ട്! ബോസ്റ്റൺ ഡൈനാമിക്സിന്റെ റോബോട്ട് ഡോഗ്. സ്റ്റെപ്പുകളും ചെരിവുകളും ഈസിയായി കേറും. പുറത്ത് പതിനാല് കിലോ വരെ സാധനവും കൊണ്ടുപോകും.",
    "ഈ മഞ്ഞ ഭീമനാണ് തെർമൈറ്റ് ഫയർഫൈറ്റിംഗ് റോബോട്ട്. ഒരു മിനിറ്റിൽ ടു തൗസന്റ് ഫൈവ് ഹണ്ട്രഡ് ഗാലൻ വെള്ളം പമ്പ് ചെയ്യും. ഡീസലിൽ ഇരുപത് മണിക്കൂർ തുടർച്ചയായി ഓടും.",
    "ഈ കുഞ്ഞൻ നാവോയ്ക്ക് അറുപത് സെന്റിമീറ്ററിൽ താഴെയേ ഉയരമുള്ളൂ, പക്ഷേ ഇരുപത്തിയഞ്ച് ജോയിന്റുകളുണ്ട്. ടു തൗസന്റ് എയ്റ്റ് മുതൽ കുട്ടികളെയും റിസർച്ചേഴ്സിനെയും പഠിപ്പിക്കുന്നുണ്ട്.",
    "ഈ റോബോട്ട് ശരിക്കുള്ള ബ്രഷ് കൊണ്ട് ശരിക്കുള്ള ക്യാൻവാസിൽ ഓരോ ഡോട്ടായി വരയ്ക്കുകയായിരുന്നു. ആം ഒരു യു ഫാക്ടറി എക്സ് ആം ആണ്. എന്ത് വരയ്ക്കണമെന്ന് തീരുമാനിക്കുന്നത് ഒരു എ ഐ.",
    "ആപ്പിൽ ഓർഡർ ചെയ്തവർക്ക് ഒട്ടോണമിയുടെ ഒട്ടോബോട്ട് ഡ്രിങ്ക്സ് കൊണ്ടുകൊടുക്കുന്നു. അന്നുതന്നെ സിൻസിനാറ്റി പോലുള്ള എയർപോർട്ടുകളിൽ ഇത് ആൾക്കൂട്ടത്തിനിടയിലൂടെ തനിയെ വഴി കണ്ടുപിടിച്ച് ഡെലിവറി ചെയ്യുന്നുണ്ടായിരുന്നു.",
    "പിന്നെ അമേക്ക. ഞാൻ കണ്ടതിൽ ഏറ്റവും എക്സ്പ്രസീവ് ആയ റോബോട്ട് മുഖം ഇതിന്റേതാണ്. ബ്രിട്ടനിലെ എഞ്ചിനീയേർഡ് ആർട്സ് ട്വന്റി ട്വന്റി വണ്ണിന്റെ അവസാനമാണ് ഇതിനെ ലോഞ്ച് ചെയ്തത്.",
    "ഒരു മാസം കഴിഞ്ഞ് വീണ്ടും എക്സ്പോയിൽ. ഈ റോബോട്ട് ആം മണൽ ചേർത്ത ബയോഡീഗ്രേഡബിൾ പ്ലാസ്റ്റിക് കൊണ്ട് ശില്പങ്ങൾ ത്രീ ഡി പ്രിന്റ് ചെയ്യുകയായിരുന്നു. ബ്രണോ യൂണിവേഴ്സിറ്റി ഓഫ് ടെക്നോളജിയും ഇതിൽ പങ്കാളിയായിരുന്നു.",
    "തലയ്ക്കുള്ളിലെ ഒരു പ്രൊജക്ടർ മാസ്കിലേക്ക് മുഖം പ്രൊജക്റ്റ് ചെയ്യും. അതുകൊണ്ട് ഒന്നും അനക്കാതെ തന്നെ മുഖം മാറ്റാം.",
    "ഇതുപോലുള്ളവ പല ട്രേകളിലായി ഫുഡ് കൊണ്ടുപോകും. ലേസറും ക്യാമറയും വെച്ചാണ് വഴി കണ്ടുപിടിക്കുന്നത്.",
    "ഇതാണ് ഓപ്റ്റി, ഓപ്പർച്യൂണിറ്റി പവലിയന്റെ മാസ്കോട്ട്. ഏകദേശം ഒരു മീറ്റർ ഉയരം. ആറ് മാസത്തിനുള്ളിൽ ടെർമിനസ് ഇതുപോലെ അൻപതെണ്ണം ഉണ്ടാക്കി.",
    "കണ്ടതിന് താങ്ക്സ്. ഇതിൽ ഏത് റോബോട്ടിനെ നേരിട്ട് കാണാനാണ് നിങ്ങൾക്ക് ഏറ്റവും ഇഷ്ടം? കമന്റിൽ പറയണേ.",
    "ചുവന്ന കുഞ്ഞു റോബോട്ടുകൾ പാഴ്സലുകളും കൊണ്ട് അവിടെ ഓടി നടക്കുന്നുണ്ടായിരുന്നു. ഒരു റോബോട്ട് ഡിപ്പോ മണിക്കൂറിൽ പതിനായിരത്തിലധികം പാഴ്സലുകൾ സോർട്ട് ചെയ്യും.",
];

fn count(lines: &[&str], kind: Option<Kind>, level: Level) -> usize {
    lines.iter().flat_map(|l| lint(l)).filter(|f| f.level == level && kind.is_none_or(|k| f.kind == k)).count()
}

#[test]
fn the_written_script_gets_many_findings_the_spoken_one_few() {
    let (written, spoken) = (count(&WRITTEN, None, Level::Warning), count(&SPOKEN, None, Level::Warning));
    assert!(written >= 12, "{written} warnings: {:#?}", WRITTEN.iter().flat_map(|l| lint(l)).collect::<Vec<_>>());
    assert!(spoken <= 3, "{spoken} warnings: {:#?}", SPOKEN.iter().flat_map(|l| lint(l)).filter(|f| f.level == Level::Warning).collect::<Vec<_>>());
    assert!(written >= 4 * spoken.max(1), "{written} vs {spoken}");
    // the spoken script uses no formal word of the table, and has no over-long sentence
    assert_eq!(count(&SPOKEN, Some(Kind::Formal), Level::Warning), 0);
    assert_eq!(count(&SPOKEN, Some(Kind::LongSentence), Level::Warning), 0);
    assert!(count(&WRITTEN, Some(Kind::Formal), Level::Warning) >= 9);
    assert!(count(&WRITTEN, Some(Kind::LongSentence), Level::Warning) >= 2);
    // both say 27,000 and 10,000 in Malayalam words: write digits
    assert_eq!(count(&SPOKEN, Some(Kind::NumberWords), Level::Warning), 2);
}

#[test]
fn formal_words_and_what_people_say() {
    let found = |t: &str| lint(t).into_iter().filter(|f| f.kind == Kind::Formal).map(|f| (f.text, f.fix, f.level)).collect::<Vec<_>>();
    let w = Level::Warning;
    assert_eq!(
        found("സന്ദർശകരെ സ്വാഗതം ചെയ്യുന്നു."),
        [("സന്ദർശകരെ".to_string(), Some("വിസിറ്റേഴ്സിനെ".to_string()), w), ("സ്വാഗതം ചെയ്യുന്നു".to_string(), Some("വെൽക്കം ചെയ്യുന്നു".to_string()), w)]
    );
    assert_eq!(found("ഇത് വികസിപ്പിക്കാൻ സഹായിച്ചു.")[0].1.as_deref(), Some("ഡെവലപ്പ് ചെയ്യാൻ"));
    assert_eq!(found("യു എ ഇ-യിൽ തന്നെ നിർമ്മിച്ചതാണ്.")[0].1.as_deref(), Some("ഉണ്ടാക്കിയതാണ്"));
    assert_eq!(found("ഞാൻ ഇത് നിർമ്മിച്ചു.")[0].1.as_deref(), Some("ഉണ്ടാക്കി"));
    assert_eq!(found("ഒറ്റ ചാർജിൽ എട്ട് മണിക്കൂർ ലഭിക്കും.")[0].1.as_deref(), Some("കിട്ടും"));
    assert_eq!(found("അത് ലഭിച്ചത് ഇന്നാണ്.")[0].1.as_deref(), Some("കിട്ടിയത്"));
    assert_eq!(found("ഇതിന് ചുമരിൽ കയറാൻ സാധിക്കില്ല.")[0].1.as_deref(), Some("പറ്റില്ല"));
    assert_eq!(found("ജോലികൾക്കായി ഉണ്ടാക്കിയതാണ്.")[0].1.as_deref(), Some("ജോലികൾക്ക് വേണ്ടി"));
    assert_eq!(found("മാപ്പിംഗിനായി ഫിറ്റ് ചെയ്തു.")[0].1.as_deref(), Some("മാപ്പിംഗിന് വേണ്ടി"));
    assert_eq!(found("അതിനാൽ ഇത് വേഗം ഓടും.")[0].1.as_deref(), Some("അതുകൊണ്ട്"));
    assert_eq!(found("ഗ്ലാസിനുള്ളിൽ ഒരു ആം നിയന്ത്രിച്ച് കളിക്കാം.")[0].1.as_deref(), Some("കൺട്രോൾ ചെയ്ത്"));
    assert_eq!(found("അത് ഒരു റോബോട്ട് നായകളുടെ കൂട്ടം.")[0].1.as_deref(), Some("റോബോട്ട് ഡോഗുകളുടെ"));
    assert_eq!(found("കമന്റിൽ പറയൂ.")[0].1.as_deref(), Some("പറയണേ"));
    // a hint: right in some places, but worth a look
    let h = found("ഇത് വഴി കണ്ടെത്തി.");
    assert_eq!((h[0].1.as_deref(), h[0].2), (None, Level::Hint));
    // inflected forms the table names only by their stem are found without a fix
    let g = found("കുട്ടികളെയും ഗവേഷകരെയും പഠിപ്പിക്കുന്നു.");
    assert_eq!((g[0].0.as_str(), g[0].1.as_deref(), g[0].2), ("ഗവേഷകരെയും", None, Level::Warning));
    // what people say is not flagged
    assert!(found("വിസിറ്റേഴ്സിനെ വെൽക്കം ചെയ്യുന്നു. ലോഞ്ച് ചെയ്തത് ട്വന്റി ട്വന്റി വണ്ണിലാണ്.").is_empty());
}

#[test]
fn the_table_is_written_canonically() {
    for e in FORMAL {
        assert_eq!(malayalam::canonical(e.form), e.form, "{}", e.form);
        assert!(!e.say.is_empty() && e.form.chars().all(|c| !c.is_ascii_alphabetic()), "{}", e.form);
        if let Fix::Prefix(t) | Fix::Present(t) | Fix::PastI(t) | Fix::Suffix(t) = e.fix {
            assert_eq!(malayalam::canonical(t), t, "{}", e.form);
        }
    }
}

#[test]
fn long_sentences_count_spoken_words() {
    // a year said the English way and a spelled acronym are one word each
    assert_eq!(spoken_words("ട്വന്റി ട്വന്റി ഫോർ ഏപ്രിലിൽ ആർ ടി എ-യുടെ പട്രോൾ റോബോട്ടിനെ കണ്ടു."), 6);
    assert_eq!(spoken_words("ഇതിന്റെ ഫോർ കെ നൈറ്റ് വിഷൻ ക്യാമറ."), 5);
    assert_eq!(spoken_words(""), 0);
    let long = "ട്വന്റി ട്വന്റി ഫോർ മാർച്ചിൽ, അബുദാബിയിൽ, ഒരു കിയോസ്കിൽ ഗ്ലാസിനുള്ളിൽ ഒരു റോബോട്ട് ആം ഓർഡറുകൾ തയ്യാറാക്കുന്നു, ഒരു പെപ്പർ റോബോട്ട് കസ്റ്റമേഴ്സിനെ സ്വാഗതം ചെയ്യുന്നു.";
    let f: Vec<Finding> = lint(long).into_iter().filter(|f| f.kind == Kind::LongSentence).collect();
    assert_eq!(f.len(), 1, "{f:?}");
    assert!(f[0].message.starts_with("17 spoken words"), "{}", f[0].message);
    assert!(f[0].suggestion.contains("തയ്യാറാക്കുന്നു"), "split at the comma nearest the middle: {}", f[0].suggestion);
    // the limit can be raised
    assert!(lint_with(long, &Options { max_words: 20 }).iter().all(|f| f.kind != Kind::LongSentence));
    // and is kept in range
    assert!(lint_with("ഒന്ന് രണ്ട് മൂന്ന് നാല് അഞ്ച്.", &Options { max_words: 0 }).iter().any(|f| f.kind == Kind::LongSentence));
}

#[test]
fn number_words_should_be_digits() {
    let nums = |t: &str| lint(t).into_iter().filter(|f| f.kind == Kind::NumberWords).map(|f| (f.text, f.fix)).collect::<Vec<_>>();
    assert_eq!(nums("അറുന്നൂറിലധികം എക്സ്പ്രഷൻസ് കാണിക്കാൻ പറ്റും."), [("അറുന്നൂറിലധികം".to_string(), Some("600-ലധികം".to_string()))]);
    assert_eq!(nums("മണിക്കൂറിൽ പതിനായിരത്തിലധികം പാഴ്സലുകൾ."), [("പതിനായിരത്തിലധികം".to_string(), Some("10,000-ലധികം".to_string()))]);
    assert_eq!(nums("ഏകദേശം ഇരുപത്തിയേഴായിരം എണ്ണം."), [("ഇരുപത്തിയേഴായിരം".to_string(), Some("27,000".to_string()))]);
    assert_eq!(nums("രണ്ടായിരത്തി ഇരുപത്തിയൊന്നിൽ കണ്ടു."), [("രണ്ടായിരത്തി ഇരുപത്തിയൊന്നിൽ".to_string(), Some("2021-ൽ".to_string()))]);
    assert_eq!(nums("ആയിരത്തോളം ആളുകൾ."), [("ആയിരത്തോളം".to_string(), Some("1,000-ഓളം".to_string()))]);
    // the suggestion says how the voice reads the digits
    let f = lint("നൂറ്റമ്പതിലധികം റോബോട്ടുകൾ ഉണ്ടായിരുന്നു.");
    let n = f.iter().find(|f| f.kind == Kind::NumberWords).unwrap();
    assert_eq!(n.suggestion, "150-ലധികം (said «വൺ ഹണ്ട്രഡ് ഫിഫ്റ്റിയിലധികം»)");
    // small counts and numbers already said the English way are fine
    assert!(nums("നാല് മീറ്റർ വരെ കാണും. പന്ത്രണ്ട് വർഷം. ടു തൗസന്റ് ഫൈവ് ഹണ്ട്രഡ് ഗാലൻ. 2,500 ഗാലൻ.").is_empty());
}

#[test]
fn latin_letters_are_spelled_or_transliterated() {
    let l = |t: &str| lint(t).into_iter().filter(|f| f.kind == Kind::Latin).map(|f| (f.text, f.fix)).collect::<Vec<_>>();
    assert_eq!(l("ദുബായ് പോലീസും UAE-യിൽ ഉണ്ടാക്കി."), [("UAE".to_string(), Some("യു എ ഇ-യിൽ".to_string()))]);
    assert_eq!(l("ഇതിന്റെ 4K ക്യാമറ."), [("4K".to_string(), Some("ഫോർ കെ".to_string()))]);
    assert_eq!(l("ഇത് M02 ആണ്."), [("M02".to_string(), Some("എം സീറോ ടു".to_string()))]);
    assert_eq!(l("പ്രൊമോബോട്ട് V.4."), [("V.4".to_string(), Some("വി ഫോർ".to_string()))]);
    // a name, or a long acronym that may be said as a word: written by its sound, no fix
    assert_eq!(l("GITEX-ൽ കണ്ടു."), [("GITEX".to_string(), None)]);
    assert_eq!(l("Neolix വാൻ."), [("Neolix".to_string(), None)]);
    assert_eq!(spell_code("AI").as_deref(), Some("എ ഐ"));
    assert_eq!(spell_code("RTA").as_deref(), Some("ആർ ടി എ"));
    assert_eq!(spell_code("Go1"), None);
}

#[test]
fn a_sentence_without_a_verb_is_a_hint() {
    let nv = |t: &str| lint(t).into_iter().filter(|f| f.kind == Kind::NoVerb).map(|f| (f.text, f.level)).collect::<Vec<_>>();
    assert_eq!(nv("പന്ത്രണ്ട് വർഷം, നാൽപ്പത്തിയെട്ട് റോബോട്ടുകൾ."), [("പന്ത്രണ്ട് വർഷം, നാൽപ്പത്തിയെട്ട് റോബോട്ടുകൾ.".to_string(), Level::Hint)]);
    // a verb anywhere, a copula, a spoken ending, an exclamation: fine
    for ok in ["ഇതാണ് ഓപ്റ്റി.", "കണ്ടാൽ ശരിക്കും മനുഷ്യനാണെന്നേ തോന്നൂ.", "ഞാൻ തന്നെ ഉണ്ടാക്കിയതാ.", "വീണ്ടും പെപ്പർ!", "കമന്റിൽ പറയണേ.", "ഇവനാണ്."]
    {
        assert!(nv(ok).is_empty(), "{ok}");
    }
}

#[test]
fn fixes_keep_the_rest_of_the_text() {
    let fixed = fix("എവിടെ നോക്കിയാലും സന്ദർശകരെ സ്വാഗതം ചെയ്യുന്ന പെപ്പർ. ഇത് UAE-യിൽ നിർമ്മിച്ചതാണ്, അറുന്നൂറിലധികം മുഖഭാവങ്ങൾ!");
    assert_eq!(fixed.text, "എവിടെ നോക്കിയാലും വിസിറ്റേഴ്സിനെ വെൽക്കം ചെയ്യുന്ന പെപ്പർ. ഇത് യു എ ഇ-യിൽ ഉണ്ടാക്കിയതാണ്, 600-ലധികം എക്സ്പ്രഷൻസ്!");
    assert_eq!(fixed.applied.len(), 6, "{:?}", fixed.applied);
    // nothing to fix: the text comes back as it was, zero-width joiners and all
    let t = "തലയുടെ സ്ഥാനത്ത് ഒരു ടാബ്\u{200C}ലെറ്റ്.";
    assert_eq!(fix(t).text, t);
    // fixing again changes nothing
    let again = fix(&fixed.text);
    assert_eq!(again.text, fixed.text);
}

#[test]
fn numbers_in_malayalam_words_read_back() {
    let read = |t: &str| numbers(t).into_iter().map(|n| n.value).collect::<Vec<_>>();
    // every number the Malayalam front end writes reads back as itself
    let mut ns: Vec<u64> = (1..=1200).collect();
    ns.extend((1201..=250_000).step_by(373));
    ns.extend([10_000, 11_000, 20_000, 25_000, 99_999, 100_000, 170_000, 250_000, 1_000_000, 10_000_000, 35_000_000, 123_456_789]);
    for n in ns {
        let w = malayalam::number(n);
        assert_eq!(read(&w), [n as f64], "{n}: {w}");
    }
    // as scripts write them: joined, with suffixes and sandhi
    for (t, v) in [
        ("നൂറ്റമ്പതിലധികം", 150.0),
        ("അറുന്നൂറിലധികം", 600.0),
        ("പതിനായിരത്തിലധികം", 10_000.0),
        ("ഇരുപത്തിയേഴായിരം", 27_000.0),
        ("അൻപതെണ്ണം", 50.0),
        ("പത്തെണ്ണത്തിൽ", 10.0),
        ("ആറിലധികം", 6.0),
        ("നൂറ്റെട്ട്", 108.0),
        ("നൂറ്റിപ്പത്ത്", 110.0),
        ("നാൽപ്പത്തിയെട്ട്", 48.0),
        ("ആയിരത്തിൽ", 1000.0),
        ("ഒരു ലക്ഷം", 100_000.0),
    ] {
        assert_eq!(read(t), [v], "{t}");
    }
    assert_eq!(read("ഏഴോ എട്ടോ കിലോ"), [7.0, 8.0]);
    // words that only look like numbers
    for t in ["വണ്ടി", "ടെൻഷൻ", "നൂറ്റാണ്ട്", "ആയിരക്കണക്കിന്", "എണ്ണ", "ഫോറം", "സിക്സർ", "ഓടും", "പോയിന്റുകൾ", "ആയിരുന്നു", "ആയിരിക്കും", "ലക്ഷ്യം"]
    {
        assert!(read(t).is_empty(), "{t}: {:?}", read(t));
    }
}

#[test]
fn numbers_said_the_english_way_read_back() {
    let read = |t: &str| numbers(t).into_iter().map(|n| n.value).collect::<Vec<_>>();
    for n in (0..=3000).chain((3001..=2_000_000).step_by(997)) {
        let w = malayalam::english_number(n);
        assert_eq!(read(&w), [n as f64], "{n}: {w}");
    }
    for y in 1100..=2099 {
        let w = malayalam::english_year(y);
        assert_eq!(read(&w), [y as f64], "{y}: {w}");
    }
    for (t, v) in [
        ("ട്വന്റി ട്വന്റി വണ്ണിന്റെ", 2021.0),
        ("ട്വന്റി തേർട്ടീനിൽ", 2013.0),
        ("ട്വന്റി ട്വന്റിയിലും", 2020.0),
        ("വൺ ലാഖ് സെവന്റി തൗസന്റിലധികം", 170_000.0),
        ("സിക്സ്റ്റീൻ ഹണ്ട്രഡ് ഡോളറിനാണ്", 1600.0),
        ("ടു പോയിന്റ് ഫൈവ്", 2.5),
        ("എം സീറോ ടു", 2.0),
    ] {
        assert_eq!(read(t), [v], "{t}");
    }
    // digits, with group commas, decimals and suffixes
    assert_eq!(read("2013-ൽ 1,600 ഡോളർ, 2.5 കിലോ, 85%"), [2013.0, 1600.0, 2.5, 85.0]);
    let n = numbers("ട്വന്റി ട്വന്റി ഫോർ മാർച്ചിൽ");
    assert_eq!((n[0].text.as_str(), n[0].said), ("ട്വന്റി ട്വന്റി ഫോർ", Said::English));
    assert_eq!(digits_of(170_000.0), "170,000");
    assert_eq!(digits_of(2021.0), "2021");
    assert_eq!(digits_of(1.8), "1.8");
}

#[test]
fn hostile_text_never_panics() {
    let sample = WRITTEN.join(" ");
    let chars: Vec<char> = sample.chars().collect();
    // every prefix, and every char dropped in turn (a corrupt file)
    for cut in (0..chars.len()).step_by(7) {
        let t: String = chars[..cut].iter().collect();
        let _ = (lint(&t), numbers(&t), fix(&t));
        let u: String = chars.iter().enumerate().filter(|(i, _)| *i != cut).map(|(_, c)| c).collect();
        let _ = (lint(&u), numbers(&u), fix(&u));
    }
    for t in [
        "",
        " ",
        "...",
        "്്്്",
        "\u{200D}\u{200C}",
        "ംംം",
        "ആയിരത്തിത്തിത്തി",
        "നൂറ്റ",
        "ഒരു ഒരു ഒരു ഒരു ഒരു ഒരു ഒരു ഒരു ഒരു",
        "99999999999999999999999999 1e308 1,2,3 ..5 5.",
        "ട്വന്റി ട്വന്റി ട്വന്റി ട്വന്റി ട്വന്റി ഹണ്ട്രഡ് ക്രോർ ക്രോർ ക്രോർ ക്രോർ",
        "😀 🤖 ‽ AAAAAAAAAAAAAAAAAAAA a1b2c3",
    ] {
        let _ = (lint(t), numbers(t), fix(t), spoken_words(t));
    }
    // a huge text is read up to the limit
    let big = "സ്വാഗതം ചെയ്യുന്നു. ".repeat(60_000);
    let f = lint(&big);
    assert!(f.len() <= MAX_FINDINGS && !f.is_empty());
}

#[test]
fn voice_markers_are_not_words_and_end_a_sentence() {
    // a pause marker, a style tag and a breath are read by the voice, never said: no Latin
    let marked = "പാർട്ട് വണ്ണിൽ നമ്മൾ കണ്ടതാണ്, [pause 0.3s] ഒരു ചീപ്പ് കിറ്റ്. [excited] ഇനി പാർട്ട് ടു! [breath] [Pause 500 ms] ശരി.";
    let f = lint(marked);
    assert!(f.iter().all(|f| f.kind != Kind::Latin), "{f:?}");
    // a pause's length is no number the script says: the numbers are those of the words alone
    let said = |t: &str| numbers(t).into_iter().map(|n| n.value).collect::<Vec<_>>();
    assert_eq!(said(marked), said(&crate::script::strip_marks(marked)));
    assert!(said("[pause 2.5s] ശരി.").is_empty());
    // one sentence of 16 words is too long; a pause marker in it makes two breaths of 8
    let words = "ഒന്ന് രണ്ട് മൂന്ന് നാല് അഞ്ച് ആറ് ഏഴ് എട്ട്";
    let long = format!("{words} {words}.");
    assert!(lint(&long).iter().any(|f| f.kind == Kind::LongSentence));
    for joined in [format!("{words}, [pause 0.25s] {words}."), format!("{words}[pause 0.25s]{words}."), format!("{words} [calm] {words}.")] {
        assert!(lint(&joined).iter().all(|f| f.kind != Kind::LongSentence), "{joined}");
    }
    // a bracket that is no marker is still text
    assert!(lint("ഇത് [hello] ആണ്.").iter().any(|f| f.kind == Kind::Latin));
    // fixes keep the markers byte for byte
    let fixed = fix("[pause 0.3s] നിങ്ങളെ സ്വാഗതം ചെയ്യുന്നു. [excited] ശരി!");
    assert!(fixed.text.starts_with("[pause 0.3s] ") && fixed.text.contains(" [excited] "), "{}", fixed.text);
    assert!(!fixed.applied.is_empty());
    assert_eq!(spoken_words("[pause 1s] ഒന്ന് രണ്ട്"), 2);
}
