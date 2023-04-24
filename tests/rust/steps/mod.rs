pub mod data_matching;

use rstest::*;

use std::{error::Error, fs::File, path::Path};

use grams::{
    db::GramsDB,
    literal_matchers::parsed_text_repr::{ParsedNumberRepr, ParsedTextRepr},
    table::{CandidateEntityId, Column, Context, EntityId, Link, LinkedTable},
};

fn read_csv(file: &str, delimiter: u8) -> Vec<Vec<String>> {
    if let Ok(file) = File::open(file) {
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(false)
            .from_reader(file);
        reader
            .records()
            .map(|result| {
                result
                    .unwrap()
                    .iter()
                    .map(|x| x.to_owned())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    } else {
        panic!("File not found: {}", file)
    }
}

/// Load table from files
fn load_table(path: &Path) -> Result<LinkedTable, Box<dyn Error>> {
    let rows = read_csv(path.to_str().unwrap(), b',');
    let columns = (0..rows[0].len())
        .map(|ci| Column {
            index: ci,
            name: Some(rows[0][ci].clone()),
            values: (1..rows.len())
                .map(|ri| rows[ri][ci].clone())
                .collect::<Vec<_>>(),
        })
        .collect::<Vec<_>>();

    let mut links: Vec<Vec<Vec<Link>>> = vec![vec![vec![]; columns.len()]; rows.len() - 1];
    read_csv(
        path.to_str()
            .unwrap()
            .replace(".csv", ".links.tsv")
            .as_str(),
        b'\t',
    )
    .into_iter()
    .for_each(|r| {
        let row = r[0].parse::<usize>().unwrap();
        let col = r[1].parse::<usize>().unwrap();
        links[row][col].push(Link {
            start: 0,
            end: rows[row][col].len(),
            url: None,
            entities: vec![EntityId(r[2].to_owned())],
            candidates: r[3..]
                .iter()
                .map(|x| CandidateEntityId {
                    id: EntityId(x.to_owned()),
                    probability: 1.0,
                })
                .collect(),
        });
    });

    Ok(LinkedTable {
        id: path.file_stem().unwrap().to_str().unwrap().to_owned(),
        links,
        columns,
        context: Context {
            page_title: None,
            page_url: None,
            page_entities: Vec::new(),
        },
    })
}

fn get_table_cells(table: &LinkedTable) -> Vec<Vec<ParsedTextRepr>> {
    let mut cells = Vec::new();
    for row in 0..table.links.len() {
        let mut row_cells = Vec::new();
        for col in 0..table.links[row].len() {
            let cell = ParsedTextRepr {
                origin: table.columns[col].values[row].clone(),
                normed_string: table.columns[col].values[row].clone(),
                number: match table.columns[col].values[row].parse::<f64>() {
                    Ok(n) => Some(ParsedNumberRepr {
                        number: n,
                        number_string: table.columns[col].values[row].clone(),
                        is_integer: table.columns[col].values[row].parse::<i64>().is_ok(),
                        unit: None,
                        prob: None,
                    }),
                    Err(_) => None,
                },
                datetime: None,
            };
            row_cells.push(cell);
        }
        cells.push(row_cells);
    }
    cells
}

#[fixture]
pub fn db() -> GramsDB {
    GramsDB::new("/Users/rook/workspace/sm-dev/data/home/databases").unwrap()
}
