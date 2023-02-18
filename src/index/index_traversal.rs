use hashbrown::{HashMap, HashSet};
use kgdata::models::Entity;

use super::{
    object_hop1_index::{MatchedStatement, ObjectHop1Index},
    EntityTraversal,
};

struct IndexTraversal<'t> {
    pub entities: &'t HashMap<String, Entity>,
    pub index: ObjectHop1Index,
}

impl<'t> EntityTraversal for IndexTraversal<'t> {
    fn get_outgoing_entities<'t1>(&'t1 mut self, entity_ids: &[&str]) -> Vec<&'t1 Entity> {
        let mut found_ents = HashSet::new();
        let mut next_entities = Vec::new();
        for entid in entity_ids {
            if let Some(next_ents) = self.index.index.get(*entid) {
                for eid in next_ents.keys() {
                    if !found_ents.contains(eid) {
                        next_entities.push(self.entities.get(eid).unwrap());
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
    ) -> Box<dyn Iterator<Item = &'t1 MatchedStatement> + 't1> {
        if let Some(targets) = self.index.index.get(source) {
            if let Some(props) = targets.get(target) {
                return Box::new(props.iter());
            }
        }
        return Box::new(std::iter::empty::<&MatchedStatement>());
    }
}
