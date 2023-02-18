use hashbrown::{HashMap, HashSet};
use kgdata::models::{Entity, Value};

use super::{object_hop1_index::DiscoveredInfo, EntityTraversal};

pub struct CacheTraversal<'s> {
    pub entities: &'s HashMap<String, Entity>,
    outgoing_ents: HashMap<String, HashSet<String>>,
}

impl<'s> EntityTraversal<'s> for CacheTraversal<'s> {
    fn get_outgoing_entities(&mut self, entity_ids: &[&str]) -> HashMap<&'s String, &'s Entity> {
        let mut next_entities = HashMap::new();
        for entid in entity_ids {
            let ent = self.entities.get(*entid).unwrap();

            if !self.outgoing_ents.contains_key(*entid) {
                let mut next_tmp = HashSet::new();
                let ent = self.entities.get(*entid).unwrap();
                for stmts in ent.props.values() {
                    for stmt in stmts {
                        if let Value::EntityId(entid) = &stmt.value {
                            next_tmp.insert(entid.id.clone());
                        }

                        for qvals in stmt.qualifiers.values() {
                            for qval in qvals {
                                if let Value::EntityId(entid) = qval {
                                    next_tmp.insert(entid.id.clone());
                                }
                            }
                        }
                    }
                }
                self.outgoing_ents.insert((*entid).to_owned(), next_tmp);
            }

            for next_entid in self.outgoing_ents.get(*entid).unwrap() {
                if !next_entities.contains_key(next_entid) {
                    let ent = self.entities.get(next_entid).unwrap();
                    next_entities.insert(&ent.id, ent);
                }
            }
        }

        next_entities
    }

    fn get_connected_properties(
        &mut self,
        source: &str,
        target: &str,
    ) -> Box<dyn Iterator<Item = &DiscoveredInfo>> {
        todo!()
    }
}
