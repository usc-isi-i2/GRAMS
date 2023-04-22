use crate::{
    cangraph::{CGEdge, CGNode, CGStatementNode},
    datagraph::node::{DGNode, DGNodeId},
    db::CacheRocksDBDict,
    error::GramsError,
};
use kgdata::models::{Entity, EntityLink, Value};
use log::warn;
use pyo3::prelude::*;

use super::{does_ent_have_data, FeatureExtractorContext};

/// Get the **number** of DG pairs that may contain contradicted information with the relationship inedge -> s -> outedge
///
/// A pair that contain contradicted information when both:
/// (1) It does not found in data graph
/// (2) The n-ary relationships (property and (optionally) qualifier) exist in the entity.
///
/// Because of (1), the (2) says the value in KG is different from the value in the table. However, we need to be
/// careful because of missing values. To combat this, we need to distinguish when we actually have missing values.
///
/// Also, we need to be careful to not use entities that the threshold is way too small (e.g., the cell is Bob but the candidate
/// is John).
///
/// For detecting missing values, we can use some information such as if the relationship has exactly one value, or
/// we can try to detect if we can find the information in some pages.
pub fn get_contradicted_information(
    cgu: &CGNode,
    s: &CGStatementNode,
    cgv: &CGNode,
    inedge: &CGEdge,
    outedge: &CGEdge,
    correct_entity_threshold: f64,
    context: &mut FeatureExtractorContext<'_>,
) -> Result<Vec<ContradictedInformation>, GramsError> {
    let uv_links = {
        context.cache.get_dg_pairs(&context.dg, s, outedge);
        context.cache.get_dg_pairs.get(&(s.id, outedge.id)).unwrap()
    };

    let is_outpred_qualifier = inedge.predicate != outedge.predicate;
    let is_outpred_data_predicate = context
        .context
        .get_or_fetch_property(&outedge.predicate, context.db)?
        .is_data_property();

    let entities = &context.context.entities;

    let mut contradicted_info = Vec::new();
    // let mut n_contradicted_info = 0;

    for (dgu, dgv) in context.dg.iter_dg_pair(cgu, cgv) {
        if uv_links.contains(&(dgu, dgv)) {
            continue;
        }

        let no_data = match context.dg.get_node(dgu) {
            DGNode::Cell(cell) => cell.entity_ids.iter().all(|eid| {
                cell.entity_probs[eid] < correct_entity_threshold
                    || !does_ent_have_data(
                        &entities[eid],
                        &inedge.predicate,
                        &outedge.predicate,
                        is_outpred_qualifier,
                    )
            }),
            DGNode::EntityValue(entity) => {
                entity.entity_prob < correct_entity_threshold
                    || !does_ent_have_data(
                        &entities[&entity.entity_id],
                        &inedge.predicate,
                        &outedge.predicate,
                        is_outpred_qualifier,
                    )
            }
            _ => {
                return Err(GramsError::IntegrityError(
                    "Source node must be cell or entity".to_owned(),
                ))
            }
        };
        if no_data {
            continue;
        }

        // at this moment, it should have some info. different than the one in the table
        // but it can be due to missing values, so we check it here.
        if is_outpred_data_predicate {
            let dgv_value = match context.dg.get_node(dgv) {
                DGNode::Cell(cell) => Some(&cell.value),
                DGNode::LiteralValue(literal) => match &literal.value {
                    Value::String(v) => Some(v),
                    Value::Quantity(v) => Some(&v.amount),
                    Value::MonolingualText(v) => Some(&v.text),
                    // Value::Time(v) => &v.time,
                    // Value::GlobeCoordinate(v) => &v.globe,
                    // Value::EntityId(v) => &v.id,
                    _ => {
                        // TODO: the other types weren't handled properly
                        None
                    }
                },
                DGNode::EntityValue(_entity) => {
                    // we do have this case that data predicate such as P2561
                    // that values link to entity value, we do not handle it for now so
                    // we set the value to None so it is skipped
                    None
                }
                _ => {
                    return Err(GramsError::IntegrityError(
                        "Target node must be cell, entity or literal".to_owned(),
                    ))
                }
            };

            let has_missing_info = if let Some(dgv_value) = dgv_value {
                match context.dg.get_node(dgu) {
                    DGNode::Cell(cell) => cell.entity_ids.iter().any(|eid| {
                        if cell.entity_probs[eid] < correct_entity_threshold {
                            return false;
                        }
                        let ent = &entities[eid];
                        does_ent_have_data(
                            ent,
                            &inedge.predicate,
                            &outedge.predicate,
                            is_outpred_qualifier,
                        ) && is_missing_data_info(
                            ent,
                            &inedge.predicate,
                            &outedge.predicate,
                            dgv_value,
                            // context,
                        )
                    }),
                    DGNode::EntityValue(e) => {
                        let ent = &entities[&e.entity_id];
                        // we do not need to check if this ent prob is above the threshold
                        // and if it has the data because we already checked it above (a single entity so if we do not have it, this code is not reachable)
                        is_missing_data_info(
                            ent,
                            &inedge.predicate,
                            &outedge.predicate,
                            dgv_value,
                            // context,
                        )
                    }
                    _ => false,
                }
            } else {
                false
            };

            if !has_missing_info {
                // n_contradicted_info += 1;
                contradicted_info.push(ContradictedInformation {
                    source: dgu,
                    target: dgv,
                    inedge: inedge.predicate.clone(),
                    outedge: outedge.predicate.clone(),
                });
            }
        } else {
            // object property, check an external db
            // but we need to filter out the case where we do not have the data
            let no_data = match context.dg.get_node(dgv) {
                DGNode::Cell(dgv_cell) => dgv_cell
                    .entity_probs
                    .values()
                    .all(|&p| p < correct_entity_threshold),
                DGNode::EntityValue(dgv_entity) => {
                    dgv_entity.entity_prob < correct_entity_threshold
                }
                DGNode::LiteralValue(dgv_lit) => {
                    warn!(
                        "Found a literal value {:?} for an object property {} -> {} in one of the entities of node {:?}",
                        dgv_lit,
                        &inedge.predicate,
                        &outedge.predicate,
                        &dgu,
                    );
                    true
                }
                _ => {
                    return Err(GramsError::IntegrityError(
                        "Target node must be cell, entity or literal".to_owned(),
                    ))
                }
            };
            if no_data {
                continue;
            }

            let has_missing_info = match context.dg.get_node(dgu) {
                DGNode::Cell(dgu_cell) => dgu_cell.entity_ids.iter().any(|eid| {
                    if dgu_cell.entity_probs[eid] < correct_entity_threshold {
                        return false;
                    }
                    let ent = &entities[eid];
                    if !does_ent_have_data(
                        ent,
                        &inedge.predicate,
                        &outedge.predicate,
                        is_outpred_qualifier,
                    ) {
                        return false;
                    }

                    match context.dg.get_node(dgv) {
                        DGNode::Cell(dgv_cell) => dgv_cell.entity_ids.iter().any(|v_eid| {
                            dgv_cell.entity_probs[v_eid] >= correct_entity_threshold
                                && is_missing_object_info(
                                    ent,
                                    &inedge.predicate,
                                    &outedge.predicate,
                                    v_eid,
                                    &mut context.cache.entity_links,
                                )
                        }),
                        DGNode::EntityValue(dgv_ent) => {
                            dgv_ent.entity_prob >= correct_entity_threshold
                                && is_missing_object_info(
                                    ent,
                                    &inedge.predicate,
                                    &outedge.predicate,
                                    &dgv_ent.entity_id,
                                    &mut context.cache.entity_links,
                                )
                        }
                        _ => unreachable!(),
                    }
                }),
                DGNode::EntityValue(dgu_ent) => {
                    let ent = &entities[&dgu_ent.entity_id];

                    // we do not need to check if this ent prob is above the threshold
                    // and if it has the data because we already checked it above (a single entity so if we do not have it, this code is not reachable)
                    match context.dg.get_node(dgv) {
                        DGNode::Cell(dgv_cell) => dgv_cell.entity_ids.iter().any(|v_eid| {
                            dgv_cell.entity_probs[v_eid] >= correct_entity_threshold
                                && is_missing_object_info(
                                    ent,
                                    &inedge.predicate,
                                    &outedge.predicate,
                                    v_eid,
                                    &mut context.cache.entity_links,
                                )
                        }),
                        DGNode::EntityValue(dgv_ent) => {
                            dgv_ent.entity_prob >= correct_entity_threshold
                                && is_missing_object_info(
                                    ent,
                                    &inedge.predicate,
                                    &outedge.predicate,
                                    &dgv_ent.entity_id,
                                    &mut context.cache.entity_links,
                                )
                        }
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            };

            if !has_missing_info {
                // n_contradicted_info += 1;
                contradicted_info.push(ContradictedInformation {
                    source: dgu,
                    target: dgv,
                    inedge: inedge.predicate.clone(),
                    outedge: outedge.predicate.clone(),
                });
            }
        }
    }
    Ok(contradicted_info)
}

fn is_missing_data_info(
    _ent: &Entity,
    _property: &str,
    _qualifier: &str,
    _value: &str,
    // context: &FeatureExtractorContext,
) -> bool {
    // TODO: implement this via infobox search
    false
}

fn is_missing_object_info(
    ent: &Entity,
    _property: &str,
    _qualifier: &str,
    target_ent_id: &str,
    // cache: &mut FeatureExtractorCache,
    entity_links: &mut CacheRocksDBDict<EntityLink>,
) -> bool {
    if let Some(link) = entity_links.get(&ent.id).unwrap() {
        link.targets.contains(target_ent_id)
    } else {
        false
    }
}

#[pyclass(module = "grams.core.features")]
/// Represents a pair of nodes that the relationship between them is contradicted by the data
pub struct ContradictedInformation {
    #[pyo3(get)]
    pub source: DGNodeId,
    #[pyo3(get)]
    pub target: DGNodeId,
    #[pyo3(get)]
    pub inedge: String,
    #[pyo3(get)]
    pub outedge: String,
}
