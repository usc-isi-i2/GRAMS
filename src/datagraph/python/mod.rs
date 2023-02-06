use pyo3::prelude::*;

pub mod graphview;

pub(crate) fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "datagraph")?;

    m.add_submodule(submodule)?;

    submodule.add_class::<graphview::PyDGraph>()?;
    // submodule.add_class::<PDGNode>()?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("grams.grams.datagraph", submodule)?;

    Ok(())
}
