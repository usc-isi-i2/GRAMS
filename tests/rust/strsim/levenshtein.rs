use anyhow::Result;
use grams::strsim::{CharacterTokenizer, Levenshtein, StrSimWithValueTokenizer};

#[test]
fn test_similarity() -> Result<()> {
    let mut tokenizer = CharacterTokenizer {};
    let strsim = StrSimWithValueTokenizer::new(&mut tokenizer, Levenshtein::default());

    let testcases = [("abc", "def", 0.0), ("aaa", "aaa", 1.0)];
    for (k, q, sim) in testcases {
        assert_eq!(strsim.similarity(k, q), sim);
    }
}
