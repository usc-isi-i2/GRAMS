use hashbrown::HashSet;

use crate::error::GramsError;
use crate::table::{CandidateEntityId, Link};
use crate::{
    index::EntityTraversal,
    strsim::StrSim,
    table::{EntityId, LinkedTable},
};
use kgdata::models::Entity;

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
 */
pub fn augment_candidates<'t0: 't1, 't1>(
    table: &LinkedTable,
    entity_traversal: &'t1 mut Box<dyn EntityTraversal + 't0>,
    strsim: &Box<dyn StrSim>,
    threshold: f64,
    use_column_name: bool,
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
        if (0..nrows).any(|ri| table.links[ri][ci].iter().any(|l| l.candidates.len() > 0)) {
            entity_columns.push(ci);
        }
    }
    let mut tmp_strs = [String::new(), String::new()];

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

            let next_ents = entity_traversal.get_outgoing_entities(&entids);

            for &oci in &entity_columns {
                if oci == ci {
                    continue;
                }

                let value = &table.columns[oci].values[ri];
                let queries: Vec<&str> = if use_column_name {
                    if let Some(header) = &table.columns[oci].name {
                        tmp_strs[0] = format!("{} {}", header, value);
                        tmp_strs[1] = format!("{} {}", value, header);
                        vec![value, tmp_strs[0].trim(), tmp_strs[1].trim()]
                    } else {
                        vec![value]
                    }
                } else {
                    vec![value]
                };

                // search for value in next_ents
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

                let matched_entity_ids =
                    search_text(&queries, &filtered_next_ents, strsim, threshold)?;
                if matched_entity_ids.len() > 0 {
                    if newtable.links[ri][oci].len() == 0 {
                        newtable.links[ri][oci].push(Link {
                            start: 0,
                            end: value.len(),
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
pub fn search_text<'t>(
    queries: &[&str],
    entities: &[&'t Entity],
    strsim: &Box<dyn StrSim>,
    threshold: f64,
) -> Result<Vec<CandidateEntityId>, GramsError> {
    let mut matched_ents = Vec::new();

    for ent in entities {
        let mut score = queries
            .iter()
            .map(|q| strsim.similarity(ent.label.get_default_value(), q))
            .try_fold(f64::NEG_INFINITY, |acc, x| {
                Ok::<f64, GramsError>(acc.max(x?))
            })?;

        score = score.max(
            ent.aliases
                .get_default_values()
                .iter()
                .map(|k| {
                    let res: Result<f64, GramsError> = queries
                        .iter()
                        .map(|q| strsim.similarity(k, q))
                        .try_fold(f64::NEG_INFINITY, |acc, x| Ok(acc.max(x?)));
                    res
                })
                .try_fold(f64::NEG_INFINITY, |acc, x| {
                    Ok::<f64, GramsError>(acc.max(x?))
                })?,
        );

        if score >= threshold {
            matched_ents.push(CandidateEntityId {
                id: EntityId(ent.id.clone()),
                probability: score,
            });
        }
    }

    Ok(matched_ents)
}
