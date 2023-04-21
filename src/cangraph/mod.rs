mod node;
use pyo3::prelude::*;

pub use self::node::{CGNode, CGNodeId, CGStatementNode, ColumnNode};

#[pyclass(module = "grams.core.cangraph", name = "CGEdge")]
pub struct CGEdge {
    pub id: usize,
    pub source: CGNodeId,
    pub target: CGNodeId,
    pub predicate: String,
}

#[pymethods]
impl CGEdge {
    #[new]
    pub fn new(id: usize, source: CGNodeId, target: CGNodeId, predicate: String) -> Self {
        Self {
            id,
            source,
            target,
            predicate,
        }
    }
}

pub mod python {
    use crate::datagraph::node::ContextSpan;
    use kgdata::models::python::value::PyValue;
    use pyo3::prelude::*;

    use super::{
        node::{
            CGEdgeFlowSource, CGEdgeFlowTarget, CGNode, CGNodeId, CGStatementFlow, ColumnNode,
            EntityNode, LiteralNode,
        },
        CGEdge, CGStatementNode,
    };

    #[pyclass(module = "grams.core.cangraph", name = "CGNode")]
    pub struct PyCGNode(pub CGNode);

    #[pymethods]
    impl PyCGNode {
        #[staticmethod]
        pub fn column_node(id: usize, label: String, column: usize) -> PyCGNode {
            PyCGNode(CGNode::Column(ColumnNode {
                id: CGNodeId(id),
                label,
                column,
            }))
        }

        #[staticmethod]
        pub fn entity_node(
            id: usize,
            entity_id: String,
            context_span: Option<ContextSpan>,
        ) -> PyCGNode {
            PyCGNode(CGNode::Entity(EntityNode {
                id: CGNodeId(id),
                entity_id,
                context_span,
            }))
        }

        #[staticmethod]
        pub fn literal_node(
            id: usize,
            value: PyValue,
            context_span: Option<ContextSpan>,
        ) -> PyCGNode {
            PyCGNode(CGNode::Literal(LiteralNode {
                id: CGNodeId(id),
                value: value.0,
                context_span,
            }))
        }

        #[staticmethod]
        pub fn statement_node(id: usize, flow: Vec<CGStatementFlow>) -> Self {
            PyCGNode(CGNode::Statement(CGStatementNode {
                id: CGNodeId(id),
                flow,
            }))
        }
    }

    pub(crate) fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
        let submodule = PyModule::new(py, "cangraph")?;

        m.add_submodule(submodule)?;

        submodule.add_class::<PyCGNode>()?;
        submodule.add_class::<CGEdge>()?;
        submodule.add_class::<CGStatementFlow>()?;
        submodule.add_class::<CGEdgeFlowSource>()?;
        submodule.add_class::<CGEdgeFlowTarget>()?;

        py.import("sys")?
            .getattr("modules")?
            .set_item("grams.core.cangraph", submodule)?;

        Ok(())
    }
}
