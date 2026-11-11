//! Build inputs that change renderer semantics; unrelated OS environment need not be Unicode.
use std::{ffi::OsString, io};

pub fn selected(values: impl IntoIterator<Item = (OsString, OsString)>) -> io::Result<Vec<(String, String)>> {
    let mut result = Vec::new();
    for (key, value) in values {
        let Some(key) = key.to_str() else { continue };
        if key.starts_with("CARGO_FEATURE_")
            || ["TARGET", "PROFILE", "OPT_LEVEL", "DEBUG", "CARGO_CFG_TARGET_FEATURE", "CARGO_ENCODED_RUSTFLAGS"].contains(&key)
        {
            let value = value.into_string().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("non-Unicode build configuration: {key}")))?;
            result.push((key.to_string(), value));
        }
    }
    result.sort();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::ffi::OsStringExt;

    #[test]
    #[cfg(unix)]
    fn unrelated_non_unicode_environment_is_not_a_build_panic() {
        let result = selected([(OsString::from("UNRELATED"), OsString::from_vec(vec![255]))]).unwrap();
        assert!(result.is_empty());
        let error = selected([(OsString::from("CARGO_ENCODED_RUSTFLAGS"), OsString::from_vec(vec![255]))]).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("CARGO_ENCODED_RUSTFLAGS"));
    }

    #[test]
    fn target_features_and_flags_are_retained_in_stable_order() {
        let inputs =
            [("TARGET", "aarch64-apple-darwin"), ("CARGO_FEATURE_GPU", "1"), ("CARGO_ENCODED_RUSTFLAGS", "-Ctarget-cpu=native"), ("UNRELATED", "ignored")];
        let one = selected(inputs.map(|(k, v)| (k.into(), v.into()))).unwrap();
        let two = selected(inputs.into_iter().rev().map(|(k, v)| (k.into(), v.into()))).unwrap();
        assert_eq!(one, two);
        assert_eq!(one.len(), 3);
        assert!(one.iter().any(|(key, _)| key == "CARGO_FEATURE_GPU"));
    }
}
