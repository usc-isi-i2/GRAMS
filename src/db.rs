use std::path::PathBuf;

use hashbrown::{HashMap, HashSet};
use kgdata::{
    db::{
        open_entity_db, open_entity_link_db, open_entity_metadata_db, open_property_db,
        ReadonlyRocksDBDict,
    },
    error::KGDataError,
    models::{Entity, EntityLink, EntityMetadata, Property, Value},
};

use crate::{
    context::{AlgoContext, PyAlgoContext},
    error::{into_pyerr, GramsError},
    table::LinkedTable,
};
use once_cell::sync::OnceCell;
use pyo3::prelude::*;

// This is okay to use unsync once cell due to the GIL from Python
static DB_INSTANCE: OnceCell<Py<GramsDB>> = OnceCell::new();

#[pyclass(module = "grams.core", subclass)]
pub struct GramsDB {
    pub datadir: PathBuf,
    pub entities: ReadonlyRocksDBDict<String, Entity>,
    pub entity_metadata: ReadonlyRocksDBDict<String, EntityMetadata>,
    pub entity_link: ReadonlyRocksDBDict<String, EntityLink>,
    pub props: ReadonlyRocksDBDict<String, Property>,
}

impl std::fmt::Debug for GramsDB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GramsDB")
            .field("datadir", &self.datadir)
            .finish()
    }
}

impl GramsDB {
    pub fn new(datadir: &str) -> Result<Self, GramsError> {
        let datadir = PathBuf::from(datadir);
        Ok(Self {
            entities: open_entity_db(datadir.join("wdentities.db").as_os_str())?,
            entity_metadata: open_entity_metadata_db(
                datadir.join("wdentity_metadata.db").as_os_str(),
            )?,
            entity_link: open_entity_link_db(datadir.join("wdentity_wikilinks.db").as_os_str())?,
            props: open_property_db(datadir.join("wdprops.db").as_os_str())?,
            datadir,
        })
    }

    pub fn open_entity_link_db(
        &self,
    ) -> Result<ReadonlyRocksDBDict<String, EntityLink>, GramsError> {
        Ok(open_entity_link_db(
            self.datadir.join("wdentity_wikilinks.db").as_os_str(),
        )?)
    }

    fn get_table_entity_ids(&self, table: &LinkedTable) -> Vec<String> {
        let mut entity_ids = HashSet::new();
        for row in &table.links {
            for links in row {
                for link in links {
                    for candidate in &link.candidates {
                        entity_ids.insert(&candidate.id.0);
                    }
                    for entityid in &link.entities {
                        entity_ids.insert(&entityid.0);
                    }
                }
            }
        }
        for entityid in &table.context.page_entities {
            entity_ids.insert(&entityid.0);
        }

        entity_ids
            .into_iter()
            .map(|x| x.to_owned())
            .collect::<Vec<_>>()
    }

    fn get_entities(
        &self,
        entity_ids: &[String],
        n_hop: usize,
    ) -> Result<(HashMap<String, Entity>, HashMap<String, EntityMetadata>), GramsError> {
        let mut entities = HashMap::new();
        for eid in entity_ids {
            let entity = self
                .entities
                .get(eid)?
                .ok_or_else(|| GramsError::DBIntegrityError(eid.clone()))?;
            entities.insert(eid.to_owned(), entity);
        }

        if n_hop == 1 {
            let entity_metadata = self.get_entity_metadata(&entities)?;
            return Ok((entities, entity_metadata));
        }

        let mut next_hop_entities: HashMap<String, Entity>;
        let mut current_hop_entities: HashMap<String, Entity> = HashMap::new();

        for i in 2..=n_hop {
            // almost certainly that the number of entities in the next hop is bigger than the current hop
            next_hop_entities = HashMap::with_capacity(entity_ids.len());
            let it = if i == 2 {
                entities.values()
            } else {
                current_hop_entities.values()
            };

            for ent in it {
                for stmts in ent.props.values() {
                    for stmt in stmts {
                        if let Value::EntityId(eid) = &stmt.value {
                            if !entities.contains_key(&eid.id)
                                && !next_hop_entities.contains_key(&eid.id)
                            {
                                let next_entity = self
                                    .entities
                                    .get(&eid.id)?
                                    .ok_or_else(|| GramsError::DBIntegrityError(eid.id.clone()))?;
                                next_hop_entities.insert(eid.id.clone(), next_entity);
                            }
                        }

                        for qvals in stmt.qualifiers.values() {
                            for qval in qvals {
                                if let Value::EntityId(eid) = &qval {
                                    if !entities.contains_key(&eid.id)
                                        && !next_hop_entities.contains_key(&eid.id)
                                    {
                                        let next_entity =
                                            self.entities.get(&eid.id)?.ok_or_else(|| {
                                                GramsError::DBIntegrityError(eid.id.clone())
                                            })?;
                                        next_hop_entities.insert(eid.id.clone(), next_entity);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // integrate the current hop
            entities.extend(current_hop_entities.into_iter());
            current_hop_entities = next_hop_entities;
        }
        entities.extend(current_hop_entities.into_iter());
        let entity_metadata = self.get_entity_metadata(&entities)?;
        Ok((entities, entity_metadata))
    }

    fn get_entity_metadata(
        &self,
        entities: &HashMap<String, Entity>,
    ) -> Result<HashMap<String, EntityMetadata>, GramsError> {
        let mut entity_metadata = HashMap::with_capacity(entities.len());
        for ent in entities.values() {
            for stmts in ent.props.values() {
                for stmt in stmts {
                    if let Value::EntityId(eid) = &stmt.value {
                        if !entity_metadata.contains_key(&eid.id) {
                            let next_entity = self
                                .entity_metadata
                                .get(&eid.id)?
                                .ok_or_else(|| GramsError::DBIntegrityError(eid.id.clone()))?;
                            entity_metadata.insert(eid.id.clone(), next_entity);
                        }
                    }

                    for qvals in stmt.qualifiers.values() {
                        for qval in qvals {
                            if let Value::EntityId(eid) = &qval {
                                if !entity_metadata.contains_key(&eid.id) {
                                    let next_entity =
                                        self.entity_metadata.get(&eid.id)?.ok_or_else(|| {
                                            GramsError::DBIntegrityError(eid.id.clone())
                                        })?;
                                    entity_metadata.insert(eid.id.clone(), next_entity);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(entity_metadata)
    }
}

#[pymethods]
impl GramsDB {
    #[new]
    pub fn pynew(datadir: &str) -> PyResult<Self> {
        Self::new(datadir).map_err(into_pyerr)
    }

    #[staticmethod]
    pub fn init(py: Python<'_>, datadir: &str) -> PyResult<()> {
        if let Some(db) = DB_INSTANCE.get() {
            if !(&db.borrow(py)).datadir.as_os_str().eq(datadir) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "The database is already initialized with a different data directory. Call deinit first.",
                ));
            }
        } else {
            DB_INSTANCE
                .set(Py::new(py, Self::new(datadir).map_err(into_pyerr)?)?)
                .unwrap();
        }

        Ok(())
    }

    #[staticmethod]
    pub fn get_instance(py: Python<'_>) -> PyResult<Py<Self>> {
        Ok(DB_INSTANCE
            .get()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "The database is not initialized. Call init first.",
                )
            })?
            .clone_ref(py))
    }

    pub fn get_algo_context(&self, table: &LinkedTable, n_hop: usize) -> PyResult<PyAlgoContext> {
        let entity_ids = self.get_table_entity_ids(table);
        let (entities, entity_metadata) =
            self.get_entities(&entity_ids, n_hop).map_err(into_pyerr)?;

        Ok(PyAlgoContext(AlgoContext::new(
            entity_ids,
            entities,
            entity_metadata,
        )))
    }
}

pub struct CacheRocksDBDict<V>
where
    V: 'static,
{
    db: ReadonlyRocksDBDict<String, V>,
    cache: HashMap<String, Option<V>>,
}

impl<V> CacheRocksDBDict<V>
where
    V: 'static,
{
    pub fn new(db: ReadonlyRocksDBDict<String, V>) -> Self {
        Self {
            db,
            cache: HashMap::new(),
        }
    }

    pub fn get(&mut self, key: &str) -> Result<Option<&V>, KGDataError> {
        if !self.cache.contains_key(key) {
            self.cache.insert(key.to_owned(), self.db.get(key)?);
        }

        Ok(self.cache[key].as_ref())
    }
}
