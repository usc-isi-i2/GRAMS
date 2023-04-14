use super::SpecificStrSim;
use anyhow::Result;
use grams::strsim::HybridJaccard;

#[test]
fn test_similarity() -> Result<()> {
    let strsim = HybridJaccard::new(
        SpecificStrSim {
            map: hashmap! {
                "a" => hashmap!{
                        "p" => 0.7,
                        "q" => 0.8,
                    },
                "b" => hashmap!{
                        "p" => 0.5,
                        "q" => 0.9,
                    },
                "c" => hashmap!{
                        "p" => 0.2,
                        "q" => 0.1,
                    },
            },
        },
        None,
        None,
    );

    assert_relative_eq!(
        strsim
            .similarity(
                &vec![vec!['a'], vec!['b'], vec!['c']],
                &vec![vec!['p'], vec!['q']]
            )
            .unwrap(),
        0.5333333333333333
    );

    Ok(())
}
