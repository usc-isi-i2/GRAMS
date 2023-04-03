use hashbrown::HashSet;

use crate::error::GramsError;
use crate::table::{CandidateEntityId, Link};
use crate::{
    index::EntityTraversal,
    strsim::StrSimWithTokenizer,
    table::{EntityId, LinkedTable},
};
use kgdata::models::EntityMetadata;

/**
 * Adding more candidates to the table.
 *
 * The algorithm works by matching object properties of entities in other columns with value of the current cell.
 * If there is a match with score greater than a threshold, the value (entity id) of the matched object property
 * is added to the candidates of the current cell.
 *
 * Note: this function assume that the table only has one link per cell, as it does not which link to search and add
 * the new candidates to.
 *
 * @param table The table to augment
 * @param entity_traversal The entity traversal to use
 * @param strsim The string similarity function to use
 * @param threshold Candidate entities that have scores less than this threshold will not be added to the candidates
 * @param use_column_name If true, the column name will be used as a part of the queries to find candidates
 * @param language The language to use to match the cell value, None to use the default language, empty string to use all languages
 * @param search_all_columns If true, all columns will be used to find the candidates, otherwise, only columns that have at least one candidate entity will be used
 */
pub fn candidate_local_search<'t0: 't1, 't1, 't2, T, SS: StrSimWithTokenizer<T>>(
    table: &LinkedTable,
    entity_traversal: &'t1 mut Box<dyn EntityTraversal + 't0>,
    strsim: &mut SS,
    threshold: f64,
    use_column_name: bool,
    language: Option<&String>,
    search_all_columns: bool,
) -> Result<LinkedTable, GramsError> {
    let (nrows, ncols) = table.shape();

    for ri in 0..nrows {
        for ci in 0..ncols {
            if table.links[ri][ci].len() > 1 {
                return Err(GramsError::InvalidInputData(format!(
                    "Table has more than one link per cell at row {} column {}",
                    ri, ci
                )));
            }
        }
    }

    let mut newtable = table.clone();

    let mut entity_columns = Vec::new();
    for ci in 0..ncols {
        if search_all_columns
            || (0..nrows).any(|ri| table.links[ri][ci].iter().any(|l| l.candidates.len() > 0))
        {
            entity_columns.push(ci);
        }
    }
    // mapping from column index => row index => queries
    let cell2queries: Vec<Vec<Vec<T>>> = (0..ncols)
        .map(|ci| {
            if !entity_columns.contains(&ci) {
                return Vec::new();
            }

            if use_column_name {
                (0..nrows)
                    .map(|ri| {
                        let text = table.columns[ci].values[ri].trim();
                        if text.is_empty() {
                            return Vec::new();
                        }

                        let val = strsim.tokenize(text);
                        if let Some(header) = &table.columns[ci].name {
                            let valheader1 = strsim.tokenize(&format!(
                                "{} {}",
                                header.trim(),
                                table.columns[ci].values[ri].trim()
                            ));
                            let valheader2 = strsim.tokenize(&format!(
                                "{} {}",
                                table.columns[ci].values[ri].trim(),
                                header.trim()
                            ));
                            vec![valheader1, valheader2, val]
                        } else {
                            vec![val]
                        }
                    })
                    .collect()
            } else {
                (0..nrows)
                    .map(|ri| vec![strsim.tokenize(table.columns[ci].values[ri].trim())])
                    .collect()
            }
        })
        .collect();

    for ci in 0..ncols {
        for ri in 0..nrows {
            let links = &table.links[ri][ci];
            if links.len() == 0 {
                continue;
            }

            let entids: Vec<&str> = links[0]
                .candidates
                .iter()
                .map(|c| c.id.0.as_str())
                .collect();

            let next_ents = entity_traversal.get_outgoing_entity_metadata(&entids);

            for &oci in &entity_columns {
                if oci == ci {
                    continue;
                }

                let queries = &cell2queries[oci][ri];
                if queries.is_empty() {
                    continue;
                }

                // search for value in next_ents
                let matched_entity_ids = if newtable.links[ri][oci].len() > 0 {
                    let filtered_next_ents = {
                        let existing_candidates: HashSet<&String> = newtable.links[ri][oci][0]
                            .candidates
                            .iter()
                            .map(|c| &c.id.0)
                            .collect();

                        next_ents
                            .iter()
                            .map(|m| *m)
                            .filter(|k| !existing_candidates.contains(&k.id))
                            .collect::<Vec<_>>()
                    };

                    search_text(&queries, &filtered_next_ents, strsim, threshold, language)?
                } else {
                    search_text(&queries, &next_ents, strsim, threshold, language)?
                };

                if matched_entity_ids.len() > 0 {
                    if newtable.links[ri][oci].len() == 0 {
                        newtable.links[ri][oci].push(Link {
                            start: 0,
                            end: table.columns[ci].values[ri].len(),
                            url: None,
                            entities: Vec::new(),
                            candidates: matched_entity_ids,
                        });
                    } else {
                        newtable.links[ri][oci][0]
                            .candidates
                            .extend(matched_entity_ids);
                        newtable.links[ri][oci][0]
                            .candidates
                            .sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());
                    }
                }
            }
        }
    }

    Ok(newtable)
}

/**
 * Search for the given queries if it matches any entities
 */
pub fn search_text<'t, 't1, T, SS: StrSimWithTokenizer<T>>(
    queries: &[T],
    entities: &[&'t EntityMetadata],
    strsim: &mut SS,
    threshold: f64,
    language: Option<&String>,
) -> Result<Vec<CandidateEntityId>, GramsError> {
    let mut matched_ents = Vec::new();

    if let Some(lang) = language {
        if lang.is_empty() {
            // use all available languages
            for ent in entities {
                let mut score = cal_max_score(queries, ent.label.lang2value.values(), strsim)?;
                for values in ent.aliases.lang2values.values() {
                    score = score.max(cal_max_score(queries, values.iter(), strsim)?);
                }
                if score >= threshold {
                    matched_ents.push(CandidateEntityId {
                        id: EntityId(ent.id.clone()),
                        probability: score,
                    });
                }
            }
        } else {
            // use specific language
            for ent in entities {
                let mut score = -1.0;
                if let Some(val) = ent.label.lang2value.get(lang) {
                    score = cal_max_score_single_key(queries, val, strsim)?;
                }
                if let Some(vals) = ent.aliases.lang2values.get(lang) {
                    score = score.max(cal_max_score(queries, vals.iter(), strsim)?);
                }
                if score >= threshold {
                    matched_ents.push(CandidateEntityId {
                        id: EntityId(ent.id.clone()),
                        probability: score,
                    });
                }
            }
        }
    } else {
        // use default languages
        for ent in entities {
            let score =
                cal_max_score_single_key(queries, ent.label.get_default_value(), strsim)?.max(
                    cal_max_score(queries, ent.aliases.get_default_values().iter(), strsim)?,
                );
            if score >= threshold {
                matched_ents.push(CandidateEntityId {
                    id: EntityId(ent.id.clone()),
                    probability: score,
                });
            }
        }
    }

    Ok(matched_ents)
}

#[inline(always)]
fn cal_max_score<'t, 't1, I: Iterator<Item = &'t String>, T, SS: StrSimWithTokenizer<T>>(
    queries: &[T],
    keys: I,
    strsim: &mut SS,
) -> Result<f64, GramsError> {
    let mut max_score = f64::NEG_INFINITY;
    for k in keys {
        for q in queries {
            let score = strsim.similarity_pre_tok1(k, q)?;
            if score > max_score {
                max_score = score;
            }
        }
    }
    return Ok(max_score);
}

fn cal_max_score_single_key<'t, T, SS: StrSimWithTokenizer<T>>(
    queries: &[T],
    key: &String,
    strsim: &mut SS,
) -> Result<f64, GramsError> {
    let mut max_score = f64::NEG_INFINITY;
    for q in queries {
        let score = strsim.similarity_pre_tok1(key, q)?;
        if score > max_score {
            max_score = score;
        }
    }
    return Ok(max_score);
}
