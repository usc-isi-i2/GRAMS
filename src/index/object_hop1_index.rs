use hashbrown::{HashMap, HashSet};
use kgdata::models::{Entity, Value};

#[derive(Debug, Clone)]
pub struct MatchedQualifier {
    pub qualifier: String,
    pub qualifier_index: usize,
}

#[derive(Debug, Clone)]
pub struct MatchedStatement {
    pub property: String,
    pub statement_index: usize,
    pub is_property_matched: bool,
    pub qualifiers: Vec<MatchedQualifier>,
}

/// An index that storing target entities that an entity can reach and how to reach them.
pub struct ObjectHop1Index {
    /// Mapping from source entity to target entities and their associated list of matched statements that the
    /// target entities are discovered from. A target entity can be appeared in multiple matched statements with
    /// same or different properties, or same or different qualifiers.
    pub index: HashMap<String, HashMap<String, Vec<MatchedStatement>>>,
}

impl ObjectHop1Index {
    pub fn from_entities<S: AsRef<str>>(
        entity_ids: &[S],
        all_entities: &HashMap<String, Entity>,
    ) -> ObjectHop1Index {
        let mut index = HashMap::with_capacity(entity_ids.len());
        let filter_props = None;

        for entid in entity_ids {
            let ent = all_entities.get((*entid).as_ref()).unwrap();
            index.insert(
                (*entid).as_ref().to_owned(),
                Self::_build_outgoing_index(ent, filter_props),
            );
        }

        ObjectHop1Index { index }
    }

    fn _build_outgoing_index(
        ent: &Entity,
        filter_props: Option<&HashSet<String>>,
    ) -> HashMap<String, Vec<MatchedStatement>> {
        let mut index: HashMap<String, Vec<MatchedStatement>> = HashMap::new();
        let enable_filtering = filter_props.is_some();
        let empty_filter_props = HashSet::new();
        let filter_props = filter_props.unwrap_or(&empty_filter_props);
        for (prop, stmts) in ent.props.iter() {
            if enable_filtering && !filter_props.contains(prop) {
                continue;
            }

            for (stmt_i, stmt) in stmts.iter().enumerate() {
                let mut discovered_ents = HashMap::new();
                if let Value::EntityId(entid) = &stmt.value {
                    // add entid as target entity
                    discovered_ents.insert(
                        entid.id.clone(),
                        MatchedStatement {
                            property: prop.clone(),
                            statement_index: stmt_i,
                            is_property_matched: true,
                            qualifiers: vec![],
                        },
                    );
                }

                for (q, qvals) in stmt.qualifiers.iter() {
                    for (qual_i, qval) in qvals.iter().enumerate() {
                        if let Value::EntityId(entid) = qval {
                            if discovered_ents.contains_key(&entid.id) {
                                discovered_ents.get_mut(&entid.id).unwrap().qualifiers.push(
                                    MatchedQualifier {
                                        qualifier: q.clone(),
                                        qualifier_index: qual_i,
                                    },
                                );
                            } else {
                                discovered_ents.insert(
                                    entid.id.clone(),
                                    MatchedStatement {
                                        property: prop.clone(),
                                        statement_index: stmt_i,
                                        qualifiers: vec![MatchedQualifier {
                                            qualifier: q.clone(),
                                            qualifier_index: qual_i,
                                        }],
                                        is_property_matched: false,
                                    },
                                );
                            }
                        }
                    }
                }

                for (entid, info) in discovered_ents.into_iter() {
                    if index.contains_key(&entid) {
                        index.get_mut(&entid).unwrap().push(info);
                    } else {
                        index.insert(entid, vec![info]);
                    }
                }
            }
        }

        return index;
    }
}
