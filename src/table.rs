use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass]
pub struct PyLinkedTable {
    pub table: LinkedTable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkedTable {
    pub id: String,
    pub links: Vec<Vec<Vec<Link>>>,
    pub columns: Vec<Column>,
}

impl LinkedTable {
    pub fn shape(&self) -> (usize, usize) {
        if self.columns.len() == 0 {
            (0, 0)
        } else {
            (self.columns.len(), self.columns[0].values.len())
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    pub start: usize,
    pub end: usize,
    pub url: Option<String>,
    pub entities: Vec<EntityId>,
    pub candidates: Vec<CandidateEntityId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateEntityId {
    pub id: EntityId,
    pub probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column {
    pub index: usize,
    pub name: Option<String>,
    pub values: Vec<String>,
}
