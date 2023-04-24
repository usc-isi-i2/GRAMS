use super::{db, get_table_cells, load_table};
use grams::{
    db::GramsDB,
    literal_matchers::{LiteralMatcherConfig, PyLiteralMatcher},
    steps::{data_matching::CellNode, python::data_matching::matching},
};
use itertools::Itertools;
use rstest::*;
use serde::{Deserialize, Deserializer};
use std::collections::{HashMap, HashSet};
use std::{error::Error, fs::File, path::Path, path::PathBuf};

#[derive(Debug, Deserialize)]
struct GoldMatches {
    #[serde(deserialize_with = "cell_from_text")]
    source: CellNode,
    #[serde(deserialize_with = "cell_from_text")]
    target: CellNode,
    entity: String,
    property: String,
    #[serde(deserialize_with = "split_commasep")]
    statements: Vec<String>,
    #[serde(deserialize_with = "split_commasep")]
    qualifiers: Vec<String>,
}

fn cell_from_text<'de, D>(deserializer: D) -> Result<CellNode, D::Error>
where
    D: Deserializer<'de>,
{
    let s: &str = Deserialize::deserialize(deserializer)?;
    let x = s
        .split("-")
        .map(|x| x.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    Ok(CellNode {
        row: x[0],
        col: x[1],
    })
}

fn split_commasep<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let s: &str = Deserialize::deserialize(deserializer)?;
    Ok(s.split(",")
        .map(|x| x.to_owned())
        .filter(|x| x.len() > 0)
        .collect::<Vec<_>>())
}

fn read_matches(
    file: &Path,
    delimiter: u8,
) -> HashMap<(CellNode, CellNode), HashMap<(String, String), GoldMatches>> {
    let file = File::open(file).unwrap();
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .from_reader(file);
    let rows: Vec<GoldMatches> = reader
        .deserialize()
        .map(|result| result.unwrap())
        .collect::<Vec<_>>();

    let mut matches = HashMap::new();
    for row in rows {
        let key = (row.source.clone(), row.target.clone());
        if !matches.contains_key(&key) {
            matches.insert(key.clone(), HashMap::new());
        }
        let keyl2 = (row.entity.clone(), row.property.clone());
        assert!(!matches.get(&key).unwrap().contains_key(&keyl2));

        matches.get_mut(&key).unwrap().insert(keyl2.clone(), row);
    }

    matches
}

#[fixture]
fn test_data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/resources/data_matching")
}

#[rstest]
#[case("list_of_highest_mountains.csv", true)]
#[case("Miss_Heritage_2014.csv", false)]
fn test_matching(
    db: GramsDB,
    test_data_dir: PathBuf,
    #[case] table_name: PathBuf,
    #[case] enable_entity_matching: bool,
) -> Result<(), Box<dyn Error>> {
    let table = load_table(&test_data_dir.join(&table_name).as_path())?;
    let cells = get_table_cells(&table);
    let gold_matches = read_matches(
        &test_data_dir
            .join(table_name.to_str().unwrap().replace(".csv", ".matches.tsv"))
            .as_path(),
        b'\t',
    );

    let mut context = db.get_algo_context(&table, 1)?;
    let literal_matcher = PyLiteralMatcher::new(&LiteralMatcherConfig {
        string: "string_exact_test".to_owned(),
        monolingual_text: "monolingual_exact_test".to_owned(),
        quantity: "quantity_test".to_owned(),
        time: "time_test".to_owned(),
        globecoordinate: "globecoordinate_test".to_owned(),
        entity: if enable_entity_matching {
            "entity_similarity_test".to_owned()
        } else {
            "always_fail_test".to_owned()
        },
    })?;

    let matches = matching(
        &table,
        cells,
        &mut context,
        &literal_matcher,
        Vec::new(),
        vec!["P31".to_owned()],
        false,
        true,
    )?;

    assert!(matches.edges.len() > 0);

    assert_eq!(matches.get_cell_node(0)?, CellNode { row: 0, col: 0 });
    assert_eq!(matches.get_cell_node(1)?, CellNode { row: 0, col: 1 });
    assert_eq!(matches.edges[0].source_id, 0);
    assert_eq!(matches.edges[0].target_id, 1);

    for edge in matches.edges.iter() {
        let source = matches.get_cell_node(edge.source_id)?;
        let target = matches.get_cell_node(edge.target_id)?;

        let keyl1 = (source.clone(), target.clone());

        // rels must be specified in the gold list
        assert!(
            gold_matches.contains_key(&keyl1),
            "missing {:?} {:?}",
            keyl1,
            edge.rels
        );

        let gold_rels = gold_matches.get(&keyl1).unwrap();
        let pred_rels = edge
            .rels
            .iter()
            .flat_map(|rel| {
                rel.statements.iter().map(|stmt| {
                    let keyl2 = (rel.source_entity_id.clone(), stmt.property.clone());
                    (
                        keyl2,
                        if stmt.property_matched_score.is_some() {
                            stmt.statement_index.to_string()
                        } else {
                            format!("[{}]", stmt.statement_index)
                        },
                        stmt.qualifier_matched_scores
                            .iter()
                            .map(|x| x.qualifier.clone())
                            .collect::<Vec<_>>(),
                    )
                })
            })
            .group_by(|x| x.0.clone())
            .into_iter()
            .map(|(k, g)| {
                let mut stmts = Vec::new();
                let mut quals = HashSet::new();

                for (_k2, s, q) in g.into_iter() {
                    stmts.push(s);
                    quals.extend(q);
                }
                (k, (stmts, quals.into_iter().sorted().collect::<Vec<_>>()))
            })
            .collect::<HashMap<_, _>>();

        let prk: HashSet<&(String, String)> = HashSet::from_iter(pred_rels.keys().into_iter());
        let grk: HashSet<&(String, String)> = HashSet::from_iter(gold_rels.keys().into_iter());
        let diffkeys = prk.symmetric_difference(&grk).collect::<Vec<_>>();

        assert_eq!(diffkeys.len(), 0, "({}, {}) {:?}", source, target, diffkeys);
        for key in pred_rels.keys() {
            assert_eq!(
                pred_rels.get(key).unwrap().0,
                gold_rels.get(key).unwrap().statements,
                "({}, {}) statements diff: {:?}",
                source,
                target,
                key
            );
            assert_eq!(
                pred_rels.get(key).unwrap().1,
                gold_rels.get(key).unwrap().qualifiers,
                "({}, {}) statements diff: {:?}",
                source,
                target,
                key
            );
        }
    }

    Ok(())
}
