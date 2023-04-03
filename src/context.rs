use hashbrown::HashMap;
use kgdata::models::{Entity, EntityMetadata};

use pyo3::prelude::*;

use crate::index::ObjectHop1Index;

#[pyclass(module = "grams.core", name = "AlgoContext")]
pub struct PyAlgoContext(pub AlgoContext);

/// A context object that contains the data needed for the algorithm to run for each table.
pub struct AlgoContext {
    pub entity_ids: Vec<String>,
    pub entities: HashMap<String, Entity>,
    pub entity_metadata: HashMap<String, EntityMetadata>,
    pub index_object1hop: Option<ObjectHop1Index>,
}

impl AlgoContext {
    pub fn new(
        entity_ids: Vec<String>,
        entities: HashMap<String, Entity>,
        entity_metadata: HashMap<String, EntityMetadata>,
    ) -> Self {
        Self {
            entity_ids,
            entities,
            entity_metadata,
            index_object1hop: None,
        }
    }

    /// Initialize the 1-hop index for quick lookup. Have to do this separate from get_object_1hop_index because
    /// it extends the lifetime of self mutably borrowed, preventing us from borrow self immutably again.
    pub fn init_object_1hop_index(&mut self) {
        if self.index_object1hop.is_none() {
            self.index_object1hop = Some(ObjectHop1Index::from_entities(
                &self.entity_ids,
                &self.entities,
            ));
        }
    }

    pub fn get_object_1hop_index(&self) -> &ObjectHop1Index {
        self.index_object1hop.as_ref().unwrap().clone()
    }
}
