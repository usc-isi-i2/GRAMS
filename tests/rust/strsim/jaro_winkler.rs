use anyhow::Result;
use grams::strsim::{
    CharacterTokenizer, JaroWinkler, StrSimWithTokenizer, StrSimWithValueTokenizer,
};

#[test]
fn test_similarity() -> Result<()> {
    let mut tokenizer = CharacterTokenizer {};
    let mut strsim = StrSimWithValueTokenizer::new(&mut tokenizer, JaroWinkler::default());

    let testcases = [
        ("MARTHA", "MARHTA", 0.9611111111111111),
        ("DWAYNE", "DUANE", 0.84),
        ("DIXON", "DICKSONX", 0.8133333333333332),
    ];
    for (k, q, sim) in testcases {
        assert_eq!(strsim.similarity(k, q).unwrap(), sim);
    }

    Ok(())
}
