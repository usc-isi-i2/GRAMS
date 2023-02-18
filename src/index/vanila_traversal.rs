use hashbrown::{HashMap, HashSet};
use kgdata::models::{Entity, Value};

use super::{object_hop1_index::DiscoveredInfo, EntityTraversal};

pub struct VanilaTraversal<'s> {
    pub entities: &'s HashMap<String, Entity>,
}

impl<'s> EntityTraversal<'s> for VanilaTraversal<'s> {
    fn get_outgoing_entities(&mut self, entity_ids: &[&str]) -> HashMap<&'s String, &'s Entity> {
        let mut next_entities = HashMap::new();
        for entid in entity_ids {
            let ent = self.entities.get(*entid).unwrap();

            let ent = self.entities.get(*entid).unwrap();
            for stmts in ent.props.values() {
                for stmt in stmts {
                    if let Value::EntityId(entid) = &stmt.value {
                        if !next_entities.contains_key(&entid.id) {
                            next_entities.insert(&entid.id, self.entities.get(&entid.id).unwrap());
                        }
                    }

                    for qvals in stmt.qualifiers.values() {
                        for qval in qvals {
                            if let Value::EntityId(entid) = qval {
                                if !next_entities.contains_key(&entid.id) {
                                    next_entities
                                        .insert(&entid.id, self.entities.get(&entid.id).unwrap());
                                }
                            }
                        }
                    }
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
