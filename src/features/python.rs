use crate::cangraph::python::PyCGNode;
use crate::cangraph::CGEdge;
use crate::datagraph::node::DGNodeId;
use crate::datagraph::python::graphview::PyDGNode;
use crate::db::GramsDB;
use crate::error::into_pyerr;
use crate::{context::PyAlgoContext, table::LinkedTable};
use pyo3::prelude::*;

use super::{
    get_contradicted_information, get_unmatch_discovered_links, ContradictedInformation,
    FeatureExtractorCache, FeatureExtractorContext,
};

#[pyclass(module = "grams.core.features", name = "FeatureExtractorContext")]
struct PyFeatureExtractorContext {
    // table: Py<LinkedTable>,
    db: Py<GramsDB>,
    context: Py<PyAlgoContext>,
    dg: super::dgproxy::DGProxy,
    cache: FeatureExtractorCache,
}

#[pymethods]
impl PyFeatureExtractorContext {
    #[new]
    fn new(
        py: Python<'_>,
        table: Py<LinkedTable>,
        nodes: Vec<PyDGNode>,
        cg2dg: Vec<Option<DGNodeId>>,
        db: Py<GramsDB>,
        context: Py<PyAlgoContext>,
    ) -> PyResult<Self> {
        let dg = super::dgproxy::DGProxy::new(&table.as_ref(py).borrow(), nodes, cg2dg);
        let cache = FeatureExtractorCache::new(&db.as_ref(py).borrow()).map_err(into_pyerr)?;
        Ok(PyFeatureExtractorContext {
            // table,
            db,
            dg,
            context,
            cache,
        })
    }

    fn get_unmatch_discovered_links(
        &mut self,
        py: Python<'_>,
        cgu: &PyCGNode,
        s: &PyCGNode,
        cgv: &PyCGNode,
        inedge: &CGEdge,
        outedge: &CGEdge,
    ) -> PyResult<usize> {
        let mut context = FeatureExtractorContext {
            // table: &self.table.as_ref(py).borrow(),
            db: &self.db.as_ref(py).borrow(),
            context: &mut self.context.as_ref(py).borrow_mut().0,
            dg: &mut self.dg,
            cache: &mut self.cache,
        };
        get_unmatch_discovered_links(
            &cgu.0,
            s.0.as_statement().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>("s must be a statement node")
            })?,
            &cgv.0,
            inedge,
            outedge,
            &mut context,
        )
        .map_err(into_pyerr)
    }

    fn get_len_contradicted_information(
        &mut self,
        py: Python<'_>,
        cgu: &PyCGNode,
        s: &PyCGNode,
        cgv: &PyCGNode,
        inedge: &CGEdge,
        outedge: &CGEdge,
        correct_entity_threshold: f64,
    ) -> PyResult<usize> {
        Ok(self
            .get_contradicted_information(
                py,
                cgu,
                s,
                cgv,
                inedge,
                outedge,
                correct_entity_threshold,
            )?
            .len())
    }

    fn get_contradicted_information(
        &mut self,
        py: Python<'_>,
        cgu: &PyCGNode,
        s: &PyCGNode,
        cgv: &PyCGNode,
        inedge: &CGEdge,
        outedge: &CGEdge,
        correct_entity_threshold: f64,
    ) -> PyResult<Vec<ContradictedInformation>> {
        let mut context = FeatureExtractorContext {
            // table: &self.table.as_ref(py).borrow(),
            db: &self.db.as_ref(py).borrow(),
            context: &mut self.context.as_ref(py).borrow_mut().0,
            dg: &mut self.dg,
            cache: &mut self.cache,
        };
        get_contradicted_information(
            &cgu.0,
            s.0.as_statement().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>("s must be a statement node")
            })?,
            &cgv.0,
            inedge,
            outedge,
            correct_entity_threshold,
            &mut context,
        )
        .map_err(into_pyerr)
    }
}

pub(crate) fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "features")?;

    m.add_submodule(submodule)?;

    submodule.add_class::<PyFeatureExtractorContext>()?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("grams.core.features", submodule)?;

    Ok(())
}
