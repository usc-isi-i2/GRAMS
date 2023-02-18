use postcard;
use pyo3::PyErr;
use serde_json;
use thiserror::Error;

/// Represent possible errors returned by this library.
#[derive(Error, Debug)]
pub enum GramsError {
    /// Represents errors that occur when the input data passing to the library is invalid.
    #[error("Invalid input data: {0}")]
    InvalidInputData(String),
    /// Represents errors that occur when the configuration data passing to the library is invalid.
    #[error("Invalid configuration: {0}")]
    InvalidConfigData(String),

    #[error(transparent)]
    PostcardError(#[from] postcard::Error),

    /// Represents all other cases of `std::io::Error`.
    #[error(transparent)]
    IOError(#[from] std::io::Error),

    /// serde_json error
    #[error(transparent)]
    SerdeJsonErr(#[from] serde_json::Error),

    /// PyO3 error
    #[error(transparent)]
    PyErr(#[from] pyo3::PyErr),
}

pub fn into_pyerr<E: Into<GramsError>>(err: E) -> PyErr {
    let hderr = err.into();
    if let GramsError::PyErr(e) = hderr {
        e
    } else {
        let anyerror: anyhow::Error = hderr.into();
        anyerror.into()
    }
}
