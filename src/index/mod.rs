// mod cache_traversal;
mod index_traversal;
mod object_hop1_index;
// mod vanila_traversal;

use kgdata::models::Entity;

pub use self::index_traversal::IndexTraversal;
pub use self::object_hop1_index::{MatchedStatement, ObjectHop1Index};

pub trait EntityTraversal {
    fn get_outgoing_entities<'t1>(&'t1 mut self, entity_ids: &[&str]) -> Vec<&'t1 Entity>;
    fn iter_props_by_entity<'t1>(
        &'t1 mut self,
        source_id: &str,
        target_id: &str,
    ) -> Box<dyn Iterator<Item = &'t1 MatchedStatement> + 't1>;
}
