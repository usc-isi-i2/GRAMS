use pyo3::prelude::*;

use crate::error::into_pyerr;

use super::{
    CharacterTokenizer, HybridJaccard, Jaro, JaroWinkler, Levenshtein, MongeElkan, StrSim,
    SymmetricMongeElkan, Tokenizer, WhitespaceCharSeqTokenizer,
};

#[pyclass(module = "grams.core.strsim", name = "WhitespaceCharSeqTokenizer")]
pub struct PyWhitespaceCharSeqTokenizer(WhitespaceCharSeqTokenizer);

#[pyclass(module = "grams.core.strsim", name = "CharacterTokenizer")]
pub struct PyCharacterTokenizer(CharacterTokenizer);

#[pyclass(module = "grams.core.strsim")]
pub struct VecVecChar(Vec<Vec<char>>);

#[pyclass(module = "grams.core.strsim")]
pub struct VecChar(Vec<char>);

#[pymethods]
impl PyWhitespaceCharSeqTokenizer {
    #[new]
    fn new() -> Self {
        PyWhitespaceCharSeqTokenizer(WhitespaceCharSeqTokenizer {})
    }

    fn tokenize(&mut self, s: &str) -> VecVecChar {
        VecVecChar(self.0.tokenize(s))
    }

    fn unique_tokenize(&mut self, s: &str) -> VecVecChar {
        VecVecChar(self.0.unique_tokenize(s))
    }
}

#[pymethods]
impl PyCharacterTokenizer {
    #[new]
    fn new() -> Self {
        PyCharacterTokenizer(CharacterTokenizer {})
    }

    fn tokenize(&mut self, s: &str) -> VecChar {
        VecChar(self.0.tokenize(s))
    }

    fn unique_tokenize(&mut self, s: &str) -> VecChar {
        VecChar(self.0.unique_tokenize(s))
    }
}

#[pyfunction]
pub fn hybrid_jaccard_similarity(key: &VecVecChar, query: &VecVecChar) -> PyResult<f64> {
    HybridJaccard::default()
        .similarity_pre_tok2(&key.0, &query.0)
        .map_err(into_pyerr)
}

#[pyfunction]
pub fn levenshtein_similarity(key: &VecChar, query: &VecChar) -> PyResult<f64> {
    Levenshtein::default()
        .similarity_pre_tok2(&key.0, &query.0)
        .map_err(into_pyerr)
}

#[pyfunction]
pub fn jaro_similarity(key: &VecChar, query: &VecChar) -> PyResult<f64> {
    (Jaro {})
        .similarity_pre_tok2(&key.0, &query.0)
        .map_err(into_pyerr)
}

#[pyfunction]
#[pyo3(signature = (key, query, threshold = 0.7, scaling_factor = 0.1, prefix_len = 4))]
pub fn jaro_winkler_similarity(
    key: &VecChar,
    query: &VecChar,
    threshold: f64,
    scaling_factor: f64,
    prefix_len: usize,
) -> PyResult<f64> {
    (JaroWinkler {
        threshold,
        scaling_factor,
        prefix_len,
    })
    .similarity_pre_tok2(&key.0, &query.0)
    .map_err(into_pyerr)
}

#[pyfunction]
pub fn monge_elkan_similarity(key: &VecVecChar, query: &VecVecChar) -> PyResult<f64> {
    MongeElkan::default()
        .similarity_pre_tok2(&key.0, &query.0)
        .map_err(into_pyerr)
}

#[pyfunction]
pub fn symmetric_monge_elkan_similarity(key: &VecVecChar, query: &VecVecChar) -> PyResult<f64> {
    SymmetricMongeElkan(MongeElkan::default())
        .similarity_pre_tok2(&key.0, &query.0)
        .map_err(into_pyerr)
}

pub(crate) fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "strsim")?;

    m.add_submodule(submodule)?;

    submodule.add_class::<PyWhitespaceCharSeqTokenizer>()?;
    submodule.add_class::<PyCharacterTokenizer>()?;
    submodule.add_class::<VecVecChar>()?;
    submodule.add_function(wrap_pyfunction!(levenshtein_similarity, m)?)?;
    submodule.add_function(wrap_pyfunction!(jaro_similarity, m)?)?;
    submodule.add_function(wrap_pyfunction!(jaro_winkler_similarity, m)?)?;
    submodule.add_function(wrap_pyfunction!(monge_elkan_similarity, m)?)?;
    submodule.add_function(wrap_pyfunction!(symmetric_monge_elkan_similarity, m)?)?;
    submodule.add_function(wrap_pyfunction!(hybrid_jaccard_similarity, m)?)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("grams.core.strsim", submodule)?;

    Ok(())
}
