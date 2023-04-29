use hashbrown::HashMap;
use kgdata::models::{python::entity::PyStatementView, Entity, EntityMetadata, Property};

use pyo3::prelude::*;

use crate::{db::GramsDB, error::GramsError, index::ObjectHop1Index};

#[pyclass(module = "grams.core", name = "AlgoContext")]
pub struct PyAlgoContext(pub AlgoContext);

/// A context object that contains the data needed for the algorithm to run for each table.
pub struct AlgoContext {
    pub entity_ids: Vec<String>,
    pub entities: HashMap<String, Entity>,
    pub entity_metadata: HashMap<String, EntityMetadata>,
    pub index_object1hop: Option<ObjectHop1Index>,
    pub props: HashMap<String, Property>,
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
            props: HashMap::new(),
        }
    }

    pub fn get_or_fetch_property(
        &mut self,
        pid: &str,
        db: &GramsDB,
    ) -> Result<&Property, GramsError> {
        if !self.props.contains_key(pid) {
            let prop = db
                .props
                .get(pid)?
                .ok_or_else(|| GramsError::DBIntegrityError(pid.to_owned()))?;
            self.props.insert(pid.to_string(), prop);
        }
        Ok(self.props.get(pid).unwrap())
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
        self.index_object1hop.as_ref().unwrap()
    }
}

#[pymethods]
impl PyAlgoContext {
    pub fn get_entity_statement(
        &self,
        entity_id: &str,
        prop: &str,
        stmt_index: usize,
    ) -> PyResult<PyStatementView> {
        let stmt = self
            .0
            .entities
            .get(entity_id)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                    "Entity {} not found",
                    entity_id
                ))
            })?
            .props
            .get(prop)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                    "Property {} not found",
                    prop
                ))
            })?
            .get(stmt_index)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Statement index {} out of bounds",
                    stmt_index
                ))
            })?;
        Ok(PyStatementView::new(stmt))
    }
}
