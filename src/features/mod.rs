use crate::{
    cangraph::{CGEdge, CGNode, CGNodeId, CGStatementNode},
    context::AlgoContext,
    datagraph::node::{CellNode, DGNode, DGNodeId},
    db::{CacheRocksDBDict, GramsDB},
    error::GramsError,
    table::LinkedTable,
};
use hashbrown::{HashMap, HashSet};
use kgdata::models::{Entity, EntityLink, Value};
use pyo3::prelude::*;

mod detect_contradicted_info;
mod dgproxy;
mod python;

use self::detect_contradicted_info::{get_contradicted_information, ContradictedInformation};
pub(crate) use self::python::register;

pub struct FeatureExtractorContext<'t> {
    table: &'t LinkedTable,
    db: &'t GramsDB,
    context: &'t mut AlgoContext,
    dg: &'t dgproxy::DGProxy,

    cache: &'t mut FeatureExtractorCache,
}

struct FeatureExtractorCache {
    pub get_dg_pairs: HashMap<(CGNodeId, usize), HashSet<(DGNodeId, DGNodeId)>>,
    pub entity_links: CacheRocksDBDict<EntityLink>,
}

impl FeatureExtractorCache {
    pub fn new(db: &GramsDB) -> Result<Self, GramsError> {
        Ok(Self {
            get_dg_pairs: HashMap::new(),
            entity_links: CacheRocksDBDict::new(db.open_entity_link_db()?),
        })
    }

    pub fn get_dg_pairs(
        &mut self,
        dg: &dgproxy::DGProxy,
        s: &CGStatementNode,
        outedge: &CGEdge,
    ) -> &HashSet<(DGNodeId, DGNodeId)> {
        if !self.get_dg_pairs.contains_key(&(s.id, outedge.id)) {
            self.get_dg_pairs
                .insert((s.id, outedge.id), dg.get_dg_pairs(s, outedge));
        }
        self.get_dg_pairs.get(&(s.id, outedge.id)).unwrap()
    }
}

/// Get number of discovered links that don't match due to value differences. This function do not count if:
/// - the link between two DG nodes is impossible
/// - the property/qualifier do not exist in the entity
fn get_unmatch_discovered_links(
    cgu: &CGNode,
    s: &CGStatementNode,
    cgv: &CGNode,
    inedge: &CGEdge,
    outedge: &CGEdge,
    context: &mut FeatureExtractorContext<'_>,
) -> Result<usize, GramsError> {
    let mut n_unmatch_links = 0;

    let uv_links = context.cache.get_dg_pairs(&context.dg, s, outedge);
    let is_outpred_qualifier = inedge.predicate != outedge.predicate;
    let is_outpred_data_predicate = context
        .context
        .get_or_fetch_property(&outedge.predicate, context.db)?
        .is_data_property();

    let entities = &context.context.entities;

    for (dgu, dgv) in context.dg.iter_dg_pair(cgu, cgv) {
        // if has link, then we don't have to count
        if uv_links.contains(&(dgu, dgv)) {
            continue;
        }

        // ignore pairs that can't have any links
        if !context
            .dg
            .dg_pair_has_possible_ent_links(dgu, dgv, is_outpred_data_predicate)
        {
            continue;
        }

        let no_data = match context.dg.get_node(dgu) {
            DGNode::Cell(cell) => cell.entity_ids.iter().all(|eid| {
                !does_ent_have_data(
                    &entities[eid],
                    &inedge.predicate,
                    &outedge.predicate,
                    is_outpred_qualifier,
                )
            }),
            DGNode::EntityValue(entity) => !does_ent_have_data(
                &entities[&entity.entity_id],
                &inedge.predicate,
                &outedge.predicate,
                is_outpred_qualifier,
            ),
            _ => {
                return Err(GramsError::IntegrityError(
                    "Source node must be cell or entity".to_owned(),
                ))
            }
        };
        if no_data {
            continue;
        }
        n_unmatch_links += 1;
    }

    Ok(n_unmatch_links)
}

#[inline(always)]
fn does_ent_have_data(
    ent: &Entity,
    inedge_predicate: &str,
    outedge_predicate: &str,
    is_qualifier: bool,
) -> bool {
    if let Some(stmts) = ent.props.get(inedge_predicate) {
        !is_qualifier
            || (stmts
                .iter()
                .any(|stmt| stmt.qualifiers.contains_key(outedge_predicate)))
    } else {
        false
    }
}
