use hashbrown::{HashMap, HashSet};
use kgdata::models::{Entity, EntityMetadata};

use crate::context::AlgoContext;

use super::{
    object_hop1_index::{MatchedStatement, ObjectHop1Index},
    EntityTraversal,
};

pub struct IndexTraversal<'t> {
    pub entities: &'t HashMap<String, Entity>,
    pub entity_metadata: &'t HashMap<String, EntityMetadata>,
    pub index: &'t ObjectHop1Index,
}

impl<'t> IndexTraversal<'t> {
    pub fn from_context(context: &'t mut AlgoContext) -> Self {
        context.init_object_1hop_index();
        Self {
            entities: &context.entities,
            entity_metadata: &context.entity_metadata,
            index: context.get_object_1hop_index(),
        }
    }
}

impl<'t> EntityTraversal for IndexTraversal<'t> {
    fn get_outgoing_entity_metadata<'t1>(
        &'t1 mut self,
        entity_ids: &[&str],
    ) -> Vec<&'t1 EntityMetadata> {
        let mut found_ents = HashSet::new();
        let mut next_entities = Vec::new();
        for entid in entity_ids {
            if let Some(next_ents) = self.index.index.get(*entid) {
                for eid in next_ents.keys() {
                    if !found_ents.contains(eid) {
                        next_entities.push(self.entity_metadata.get(eid).unwrap());
                        found_ents.insert(eid);
                    }
                }
            }
        }
        next_entities
    }

    fn iter_props_by_entity<'t1>(
        &'t1 mut self,
        source: &str,
        target: &str,
    ) -> core::slice::Iter<'t1, MatchedStatement> {
        if let Some(targets) = self.index.index.get(source) {
            if let Some(props) = targets.get(target) {
                return props.iter();
            }
        }
        return [].iter();
    }
}
