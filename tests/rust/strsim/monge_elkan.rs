use anyhow::Result;
use grams::strsim::{MongeElkan, SeqStrSim, StrSimWithTokenizer, WhitespaceCharSeqTokenizer};
use std::str;

#[test]
fn test_similarity() -> Result<()> {
    let mut tokenizer = WhitespaceCharSeqTokenizer {};
    let mut strsim = SeqStrSim::new(&mut tokenizer, MongeElkan::default())?;

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
        ("United Kingdom", "UK", 0.5515873015873015),
        (
            // extra space will be ignored during tokenization
            "distrito electoral  de la Cámara de los Comunes",
            "United Kingdom",
            0.17278439153439154,
        ),
    ];
    for (k, q, sim) in testcases {
        assert_relative_eq!(strsim.similarity(k, q).unwrap(), sim);
    }

    Ok(())
}
