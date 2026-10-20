use super::*;

fn lock(pronounce: &[(&str, &str)], texts: &[&str], labels: &[&str]) -> FactLock {
    let p: BTreeMap<String, String> = pronounce.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    FactLock::new(&p, texts, labels)
}

fn missing(c: &FactCheck) -> Vec<&str> {
    c.missing.iter().map(|f| f.text.as_str()).collect()
}

#[test]
fn a_faithful_line_is_clean() {
    let l = lock(&[("Robo-C", "Robo-See")], &[], &["Promobot Robo-C"]);
    let en = "Promobot's Robo-C makes over six hundred facial expressions, and can be built to look like anyone.";
    let ml = "പ്രൊമോബോട്ടിന്റെ റോബോ സി-ക്ക് അറുന്നൂറിലധികം എക്സ്പ്രഷൻസ് കാണിക്കാൻ പറ്റും, ആരുടെ മുഖം വേണമെങ്കിലും കൊടുക്കാം.";
    let facts = l.facts(en);
    let kinds: Vec<(FactKind, &str)> = facts.iter().map(|f| (f.kind, f.text.as_str())).collect();
    assert_eq!(kinds, [(FactKind::Name, "Promobot"), (FactKind::Name, "Robo-C"), (FactKind::Number, "six hundred")]);
    let c = l.check(en, ml);
    assert!(c.is_clean(), "{c:?}");
    // the same said with digits, or the English way, is just as faithful
    assert!(l.check(en, "പ്രൊമോബോട്ടിന്റെ റോബോ സി-ക്ക് 600-ലധികം എക്സ്പ്രഷൻസ് ഉണ്ട്.").is_clean());
    assert!(l.check(en, "പ്രൊമോബോട്ടിന്റെ റോബോ സി-ക്ക് സിക്സ് ഹണ്ട്രഡിലധികം എക്സ്പ്രഷൻസ് ഉണ്ട്.").is_clean());
}

#[test]
fn numbers_and_units_are_kept() {
    let en = "It pumps two and a half thousand gallons of water a minute and runs for twenty hours on diesel.";
    let ml = "ഒരു മിനിറ്റിൽ ടു തൗസന്റ് ഫൈവ് ഹണ്ട്രഡ് ഗാലൻ വെള്ളം പമ്പ് ചെയ്യും. ഡീസലിൽ ഇരുപത് മണിക്കൂർ തുടർച്ചയായി ഓടും.";
    let facts: Vec<(FactKind, String)> = check(en, ml).missing.iter().map(|f| (f.kind, f.text.clone())).collect();
    assert!(facts.is_empty(), "{facts:?}");
    let l = FactLock::default();
    let all: Vec<FactKind> = l.facts(en).iter().map(|f| f.kind).collect();
    assert_eq!(all, [FactKind::Number, FactKind::Unit, FactKind::Unit, FactKind::Number, FactKind::Unit]);
    // a changed number is missing, and the new one added
    let c = check(en, "ഒരു മിനിറ്റിൽ ടു തൗസന്റ് ഫൈവ് ഹണ്ട്രഡ് ഗാലൻ വെള്ളം പമ്പ് ചെയ്യും. ഡീസലിൽ പത്ത് മണിക്കൂർ ഓടും.");
    assert_eq!((missing(&c), c.added.clone()), (vec!["twenty"], vec!["10".to_string()]));
    assert_eq!(c.summary(), "missing: twenty; added: 10");
    // a lost unit
    let c = check("They're capped at twenty kilometres an hour.", "മാക്സിമം ഇരുപത് കിലോമീറ്റർ സ്പീഡേ ഉള്ളൂ.");
    assert_eq!(missing(&c), ["hour"]);
    // "a" and "one" alone are words, not facts; "ഒരു" is never added
    assert!(l.facts("A month later, one robot was there.").iter().all(|f| f.kind != FactKind::Number));
    assert!(check("A robot arm.", "ഒരു റോബോട്ട് ആം.").is_clean());
}

#[test]
fn years_codes_and_decimals() {
    let l = lock(&[("Expo 2020", "Expo twenty twenty"), ("M02", "M zero two"), ("M01", "M zero one")], &[], &[]);
    let en = "Expo 2020 had the M02 and the M01.";
    let c = l.check(en, "എക്സ്പോ ട്വന്റി ട്വന്റിയിൽ എം സീറോ ടുവും എം സീറോ വണ്ണും ഉണ്ടായിരുന്നു.");
    assert!(c.is_clean(), "{c:?}");
    assert!(l.facts(en).iter().any(|f| f.text == "Expo 2020" && f.kind == FactKind::Number));
    // the year lost
    let c = l.check(en, "എക്സ്പോയിൽ എം സീറോ ടുവും എം സീറോ വണ്ണും ഉണ്ടായിരുന്നു.");
    assert_eq!(missing(&c), ["Expo 2020"]);
    // digits and a decimal said in parts
    assert!(check("In 2013, it cost $1,600 and stood 1.8 metres tall.", "ട്വന്റി തേർട്ടീനിൽ 1,600 ഡോളറിന്, ഒരു മീറ്റർ എൺപത് സെന്റിമീറ്റർ ഉയരം.").is_clean());
    let c = check("It stood 1.8 metres tall.", "ഒരു മീറ്റർ ഉയരം.");
    assert_eq!(missing(&c), ["1.8"]);
    // codes in Latin letters: their digits and letters
    let c = check("Its 4K camera.", "ഇതിന്റെ ഫോർ കെ ക്യാമറ.");
    assert!(c.is_clean(), "{c:?}");
    let c = check("Its 4K camera.", "ഇതിന്റെ ക്യാമറ.");
    assert_eq!(c.missing.len(), 2, "{c:?}");
    // left in Latin letters, a fact is still there (the lint flags the letters)
    assert!(check("Its AI and 4K camera, by Neolix.", "ഇതിന്റെ AI, 4K ക്യാമറ, Neolix-ന്റെ.").is_clean());
}

#[test]
fn names_are_matched_by_sound() {
    // a dropped place is missing; the others are found under their Malayalam suffixes
    let en = "Their grippers come from OnRobot in Odense, Denmark, which makes hands for robots that work alongside people.";
    let ml = "ഇവയുടെ ഗ്രിപ്പറുകൾ ഡെന്മാർക്കിലെ ഓൺറോബോട്ടിന്റേതാണ്. മനുഷ്യരുടെ കൂടെ ജോലി ചെയ്യുന്ന റോബോട്ടുകൾക്ക് കൈകൾ ഉണ്ടാക്കുന്ന കമ്പനിയാണ് അവർ.";
    assert_eq!(missing(&check(en, ml)), ["Odense"]);
    // respellings give the sound; a sentence-initial word is a name when the plan says so
    let l = lock(&[("NAO", "Now")], &["Behind glass, a Kawasaki arm picked up metal parts."], &["NAO", "Kawasaki robot arm"]);
    let c = l.check(
        "Little NAO stands just under sixty centimetres tall, with twenty-five moving joints, and it's been teaching kids and researchers since 2008.",
        "ഈ കുഞ്ഞൻ നാവോയ്ക്ക് അറുപത് സെന്റിമീറ്ററിൽ താഴെയേ ഉയരമുള്ളൂ, പക്ഷേ ഇരുപത്തിയഞ്ച് ജോയിന്റുകളുണ്ട്. ടു തൗസന്റ് എയ്റ്റ് മുതൽ കുട്ടികളെയും റിസർച്ചേഴ്സിനെയും പഠിപ്പിക്കുന്നുണ്ട്.",
    );
    assert!(c.is_clean(), "{c:?}");
    let en = "Kawasaki built Japan's first home-grown industrial robot, back in 1969.";
    let ml = "ജപ്പാന്റെ ആദ്യത്തെ സ്വന്തം ഇൻഡസ്ട്രിയൽ റോബോട്ട് ഉണ്ടാക്കിയത് കവാസാക്കിയാണ്, നൈന്റീൻ സിക്സ്റ്റി നൈനിൽ.";
    let names: Vec<String> = l.facts(en).into_iter().filter(|f| f.kind == FactKind::Name).map(|f| f.text).collect();
    assert_eq!(names, ["Kawasaki", "Japan"]);
    assert!(l.check(en, ml).is_clean());
    assert_eq!(missing(&l.check(en, "ജപ്പാന്റെ ആദ്യത്തെ ഇൻഡസ്ട്രിയൽ റോബോട്ട്, നൈന്റീൻ സിക്സ്റ്റി നൈനിൽ.")), ["Kawasaki"]);
    // GITEX said as a word, by its respelling
    let l = lock(&[("GITEX", "Jai-tex")], &[], &[]);
    assert!(l.check("October 2021 at GITEX.", "ട്വന്റി ട്വന്റി വൺ ഒക്ടോബറിൽ ജൈടെക്സിൽ.").is_clean());
    assert_eq!(missing(&l.check("October 2021 at GITEX.", "ട്വന്റി ട്വന്റി വൺ ഒക്ടോബറിൽ.")), ["GITEX"]);
}

#[test]
fn acronyms_letters_months() {
    let en = "The UAE built it with the RTA, and its 4K camera spots riders without helmets with over 85 percent accuracy.";
    let ml = "യു എ ഇ-യും ആർ ടി എ-യും ചേർന്ന് ഉണ്ടാക്കിയതാണ്. ഇതിന്റെ ഫോർ കെ ക്യാമറ, ഹെൽമെറ്റ് ഇല്ലാതെ ഓടിക്കുന്നവരെ എൺപത്തിയഞ്ച് ശതമാനത്തിലധികം കൃത്യതയോടെ കണ്ടുപിടിക്കും.";
    let c = check(en, ml);
    assert!(c.is_clean(), "{c:?}");
    assert_eq!(missing(&check(en, "ആർ ടി എ ഉണ്ടാക്കിയതാണ്. ഫോർ കെ ക്യാമറ എൺപത്തിയഞ്ച് ശതമാനം കൃത്യത.")), ["UAE"]);
    // single capitals in a row are one acronym; an acronym said as a word is found by its sound
    let l = lock(&[("NodeMCU", "Node M C U")], &[], &[]);
    assert!(l.check("Its brain is a NodeMCU.", "ഇതിന്റെ ബ്രെയിൻ ഒരു നോഡ് എം സി യു ആണ്.").is_clean());
    assert!(check("At the NEMO science museum.", "നീമോ സയൻസ് മ്യൂസിയത്തിൽ.").is_clean());
    // months, and "May" only next to a number
    assert_eq!(missing(&check("In March 2024, in May 2025.", "ട്വന്റി ട്വന്റി ഫോർ മാർച്ചിൽ, ട്വന്റി ട്വന്റി ഫൈവ് മേയിൽ.")), Vec::<&str>::new());
    assert!(FactLock::default().facts("May I say something?").is_empty());
    assert_eq!(missing(&check("In April 2024.", "ട്വന്റി ട്വന്റി ഫോറിൽ.")), ["April"]);
}

#[test]
fn sounds() {
    assert!(skeleton_en("Neolix").contains(&"NLKS".to_string()));
    assert_eq!(skeleton_ml("നിയോലിക്സിന്റെ"), "NLKSNT");
    assert!(skeleton_en("Engineered").contains(&"NCNRT".to_string()));
    assert_eq!(skeleton_ml("എഞ്ചിനീയേർഡ്"), "NCNRT");
    assert!(skeleton_en("X-Arm").contains(&"KSM".to_string()), "{:?}", skeleton_en("X-Arm"));
    assert_eq!(skeleton_ml("ആംസ്റ്റർഡാമിലെ"), "~STRTML");
    assert!(approx_contains("~STRTML", "MSTRTM", 0));
    assert!(approx_contains("PRCNT", "PRTNT", 1) && !approx_contains("PRCNT", "PRTNT", 0));
    assert!(!approx_contains("", "K", 0) && approx_contains("ABC", "", 0));
}

#[test]
fn hostile_lines_never_panic() {
    let en = "Promobot's Robo-C makes over six hundred facial expressions, and can be built to look like anyone. In 2013 it cost $1,600.";
    let ml = "പ്രൊമോബോട്ടിന്റെ റോബോ സി-ക്ക് അറുന്നൂറിലധികം എക്സ്പ്രഷൻസ് കാണിക്കാൻ പറ്റും.";
    let l = lock(&[("Robo-C", "Robo-See"), ("", "x"), ("a", ""), ("$", "dollar")], &[en, ""], &["", "-", "A B"]);
    let ec: Vec<char> = en.chars().collect();
    let mc: Vec<char> = ml.chars().collect();
    for cut in 0..ec.len().max(mc.len()) {
        let e: String = ec.iter().take(cut).collect();
        let m: String = mc.iter().take(cut).collect();
        let _ = (l.check(&e, &m), l.check(&m, &e), l.facts(&e));
    }
    for t in ["", " ", "''", "'s", "- - -", "$", "%%%", "1,,,2", "..5", "9".repeat(400).as_str(), "A A A A A", "😀 Robo 😀", "half half half", "and and a an"]
    {
        let _ = (l.check(t, t), check(t, "ഒരു"), l.facts(t));
    }
    let big = "Promobot built 600 robots in Dubai. ".repeat(10_000);
    let c = l.check(&big, &big);
    assert!(c.missing.len() <= MAX_FACTS);
}
