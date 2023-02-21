use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass(module = "grams.core.table", name = "LinkedTable")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkedTable {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub links: Vec<Vec<Vec<Link>>>,
    #[pyo3(get)]
    pub columns: Vec<Column>,
    #[pyo3(get)]
    pub context: Context,
}

impl LinkedTable {
    pub fn shape(&self) -> (usize, usize) {
        if self.columns.len() == 0 {
            (0, 0)
        } else {
            (self.columns[0].values.len(), self.columns.len())
        }
    }
}

#[pyclass(module = "grams.core.table", name = "Context")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    #[pyo3(get)]
    pub page_title: Option<String>,
    #[pyo3(get)]
    pub page_url: Option<String>,
    #[pyo3(get)]
    pub page_entities: Vec<EntityId>,
}

#[pyclass(module = "grams.core.table", name = "Link")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    #[pyo3(get)]
    pub start: usize,
    #[pyo3(get)]
    pub end: usize,
    #[pyo3(get)]
    pub url: Option<String>,
    #[pyo3(get)]
    pub entities: Vec<EntityId>,
    #[pyo3(get)]
    pub candidates: Vec<CandidateEntityId>,
}

#[pyclass(module = "grams.core.table", name = "EntityId")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityId(pub String);

#[pyclass(module = "grams.core.table", name = "CandidateEntityId")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateEntityId {
    #[pyo3(get)]
    pub id: EntityId,
    #[pyo3(get)]
    pub probability: f64,
}

#[pyclass(module = "grams.core.table", name = "Column")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub name: Option<String>,
    #[pyo3(get)]
    pub values: Vec<String>,
}

#[pymethods]
impl LinkedTable {
    #[new]
    pub fn new(
        id: String,
        links: Vec<Vec<Vec<Link>>>,
        columns: Vec<Column>,
        context: Context,
    ) -> Self {
        Self {
            id,
            links,
            columns,
            context,
        }
    }

    pub fn get_links(&self, row: usize, col: usize) -> PyResult<Vec<Link>> {
        Ok(self
            .links
            .get(row)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Row index {} out of bounds",
                    row
                ))
            })?
            .get(col)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Column index {} out of bounds",
                    col
                ))
            })?
            .clone())
    }
}

#[pymethods]
impl Context {
    #[new]
    pub fn new(
        page_title: Option<String>,
        page_url: Option<String>,
        page_entities: Vec<EntityId>,
    ) -> Self {
        Self {
            page_title,
            page_url,
            page_entities,
        }
    }
}

#[pymethods]
impl Link {
    #[new]
    pub fn new(
        start: usize,
        end: usize,
        url: Option<String>,
        entities: Vec<EntityId>,
        candidates: Vec<CandidateEntityId>,
    ) -> Self {
        Self {
            start,
            end,
            url,
            entities,
            candidates,
        }
    }
}

#[pymethods]
impl EntityId {
    #[new]
    fn new(id: String) -> Self {
        Self(id)
    }

    #[getter]
    fn id(&self) -> &str {
        &self.0
    }
}

#[pymethods]
impl CandidateEntityId {
    #[new]
    pub fn new(id: EntityId, probability: f64) -> Self {
        Self { id, probability }
    }
}

#[pymethods]
impl Column {
    #[new]
    pub fn new(index: usize, name: Option<String>, values: Vec<String>) -> Self {
        Self {
            index,
            name,
            values,
        }
    }
}

pub(crate) fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "table")?;

    m.add_submodule(submodule)?;

    submodule.add_class::<LinkedTable>()?;
    submodule.add_class::<Context>()?;
    submodule.add_class::<Column>()?;
    submodule.add_class::<Link>()?;
    submodule.add_class::<CandidateEntityId>()?;
    submodule.add_class::<EntityId>()?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("grams.core.table", submodule)?;

    Ok(())
}
