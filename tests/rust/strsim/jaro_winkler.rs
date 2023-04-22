use anyhow::Result;
use grams::strsim::{CharacterTokenizer, JaroWinkler, SeqStrSim, StrSimWithTokenizer};
use std::str;

#[test]
fn test_similarity() -> Result<()> {
    let mut tokenizer = CharacterTokenizer {};
    let mut strsim = SeqStrSim::new(&mut tokenizer, JaroWinkler::default())?;

    let testcases = [
        ("MARTHA", "MARHTA", 0.9611111111111111),
        ("DWAYNE", "DUANE", 0.84),
        ("DIXON", "DICKSONX", 0.8133333333333332),
        ("László", "Lsáló", 0.8900000000000001),
        ("László", "Lsáló", 0.8900000000000001),
        (
            str::from_utf8(b"L\xc3\xa1szl\xc3\xb3").unwrap(),
            str::from_utf8(b"Ls\xc3\xa1l\xc3\xb3").unwrap(),
            0.8900000000000001,
        ),
        (
            "United Kingdom",
            "Sengenia (United Kingdom)",
            0.7342857142857143,
        ),
        (
            "United",
            "UK",
            0.5555555555555555, // if threshold is 0.7, otherwise check the testcases below
        ),
        ("United Kingdom", "", 0.0),
        ("", "United Kingdom", 0.0),
        ("", "", 1.0),
    ];
    for (k, q, sim) in testcases {
        assert_relative_eq!(strsim.similarity(k, q).unwrap(), sim);
    }

    let mut strsim = SeqStrSim::new(&mut tokenizer, JaroWinkler::new(Some(0.0), None, None))?;
    let testcases = [("United", "UK", 0.5999999999999999)];
    for (k, q, sim) in testcases {
        assert_relative_eq!(strsim.similarity(k, q).unwrap(), sim);
    }

    Ok(())
}
