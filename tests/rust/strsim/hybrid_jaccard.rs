use super::SpecificStrSim;
use anyhow::Result;
use grams::strsim::{
    hybrid_jaccard::find_max_lap_score, HybridJaccard, StrSim, WhitespaceCharSeqTokenizer,
};
use lapjv::lapjv;
use ndarray::array;

#[test]
fn test_similarity() -> Result<()> {
    let mut strsim = HybridJaccard::new(
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

#[test]
fn test_lap_solver() -> Result<()> {
    let C = 0.0;
    // let matrix = array![
    //     [8.0, 5.0, 9.0, 9.0],
    //     [4.0, 2.0, 6.0, 4.0],
    //     [7.0, 3.0, 7.0, 8.0],
    //     // [0.0, 0.0, 0.0, 0.0]
    //     [C, C, C, C]
    // ];
    let matrix = array![
        [2.0, 5.0, 1.0, 1.0],
        [6.0, 8.0, 4.0, 6.0],
        [3.0, 7.0, 3.0, 2.0],
        // [0.0, 0.0, 0.0, 0.0]
        [C, C, C, C]
    ];
    let tm = &matrix;
    // let tm = 10.0 - &matrix;
    let mut score = 0.0;
    println!("{:?}", tm);
    let result = lapjv(&tm).unwrap();
    for (i, j) in result.0.iter().zip(result.1.iter()) {
        if matrix[[*i, *j]] == C {
            continue;
        }
        score += matrix[[*i, *j]];
    }
    println!("{:?} {:?} {:?}", score, result.0, result.1);
    assert!(false);
    let score = find_max_lap_score(&-matrix);
    assert_relative_eq!(score, 15.0);
    Ok(())
}
