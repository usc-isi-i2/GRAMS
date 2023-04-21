use pyo3::prelude::*;

use self::graphview::PyDGNode;

use super::node;
use super::statement;

pub mod graphview;

pub(crate) fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "datagraph")?;

    m.add_submodule(submodule)?;

    submodule.add_class::<graphview::PyDGraph>()?;
    submodule.add_class::<PyDGNode>()?;
    submodule.add_class::<node::Span>()?;
    submodule.add_class::<node::ContextSpan>()?;
    submodule.add_class::<node::CellNode>()?;
    submodule.add_class::<node::EntityValueNode>()?;
    submodule.add_class::<node::LiteralValueNode>()?;
    submodule.add_class::<statement::StatementNode>()?;
    submodule.add_class::<statement::EdgeFlowSource>()?;
    submodule.add_class::<statement::EdgeFlowTarget>()?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("grams.core.datagraph", submodule)?;

    Ok(())
}
