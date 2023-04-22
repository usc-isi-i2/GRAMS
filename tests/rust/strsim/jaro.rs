use anyhow::Result;
use grams::strsim::{CharacterTokenizer, Jaro, SeqStrSim, StrSimWithTokenizer};
use std::str;

#[test]
fn test_similarity() -> Result<()> {
    let mut tokenizer = CharacterTokenizer {};
    let mut strsim = SeqStrSim::new(&mut tokenizer, Jaro {})?;

    let testcases = [
        ("MARTHA", "MARHTA", 0.9444444444444445),
        ("DWAYNE", "DUANE", 0.8222222222222223),
        ("DIXON", "DICKSONX", 0.7666666666666666),
        ("László", "Lsáló", 0.8777777777777779),
        ("László", "Lsáló", 0.8777777777777779),
        (
            str::from_utf8(b"L\xc3\xa1szl\xc3\xb3").unwrap(),
            str::from_utf8(b"Ls\xc3\xa1l\xc3\xb3").unwrap(),
            0.8777777777777779,
        ),
        ("United Kingdom", "", 0.0),
        ("", "United Kingdom", 0.0),
        ("", "", 1.0),
    ];
    for (k, q, sim) in testcases {
        assert_relative_eq!(strsim.similarity(k, q).unwrap(), sim);
    }

    Ok(())
}
