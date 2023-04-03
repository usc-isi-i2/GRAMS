use anyhow::Result;
use grams::strsim::{
    CharacterTokenizer, Levenshtein, StrSimWithTokenizer, StrSimWithValueTokenizer,
};
use std::str;

#[test]
fn test_similarity() -> Result<()> {
    let mut tokenizer = CharacterTokenizer {};
    let mut strsim = StrSimWithValueTokenizer::new(&mut tokenizer, Levenshtein::default());

    let testcases = [("abc", "def", 0.0), ("aaa", "aaa", 1.0)];
    for (k, q, sim) in testcases {
        assert_eq!(strsim.similarity(k, q).unwrap(), sim);
    }

    Ok(())
}

#[test]
fn test_distance() -> Result<()> {
    let strsim = Levenshtein::default();

    let testcases = [
        ("a", "", 1),
        ("", "a", 1),
        ("abc", "", 3),
        ("", "abc", 3),
        ("", "", 0),
        ("a", "a", 0),
        ("abc", "abc", 0),
        ("a", "ab", 1),
        ("b", "ab", 1),
        ("ac", "abc", 1),
        ("abcdefg", "xabxcdxxefxgx", 6),
        ("ab", "a", 1),
        ("ab", "b", 1),
        ("abc", "ac", 1),
        ("xabxcdxxefxgx", "abcdefg", 6),
        ("a", "b", 1),
        ("ab", "ac", 1),
        ("ac", "bc", 1),
        ("abc", "axc", 1),
        ("xabxcdxxefxgx", "1ab2cd34ef5g6", 6),
        ("example", "samples", 3),
        ("sturgeon", "urgently", 6),
        ("levenshtein", "frankenstein", 6),
        ("distance", "difference", 5),
        ("java was neat", "scala is great", 7),
        ("ác", "áóc", 1),
        ("ác", "áóc", 1),
        (
            str::from_utf8(b"\xc3\xa1c").unwrap(),
            str::from_utf8(b"\xc3\xa1\xc3\xb3c").unwrap(),
            1,
        ),
    ];
    for (k, q, distance) in testcases {
        assert_eq!(
            strsim.distance(
                &k.chars().collect::<Vec<_>>(),
                &q.chars().collect::<Vec<_>>()
            ),
            distance as f64
        );
    }

    Ok(())
}
