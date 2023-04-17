use pyo3::prelude::*;

/// A struct for storing parsing results of a text literal
/// for literal matching.
#[pyclass(module = "grams.core.literal_matchers", name = "ParsedTextRepr")]
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedTextRepr {
    pub origin: String,
    pub normed_string: String,
    pub number: Option<ParsedNumberRepr>,
    pub datetime: Option<ParsedDatetimeRepr>,
}

#[pyclass(module = "grams.core.literal_matchers", name = "ParsedDatetimeRepr")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedDatetimeRepr {
    pub year: Option<i64>,
    pub month: Option<i64>,
    pub day: Option<i64>,
    pub hour: Option<i64>,
    pub minute: Option<i64>,
    pub second: Option<i64>,
}

#[pyclass(module = "grams.core.literal_matchers", name = "ParsedNumberRepr")]
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedNumberRepr {
    pub number: f64,
    pub number_string: String,
    pub is_integer: bool,
    pub unit: Option<String>,
    pub prob: Option<f64>,
}

impl ParsedDatetimeRepr {
    pub fn has_only_year(&self) -> bool {
        self.year.is_some()
            && self.month.is_none()
            && self.day.is_none()
            && self.hour.is_none()
            && self.minute.is_none()
            && self.second.is_none()
    }

    pub fn first_day_of_year(&self) -> bool {
        self.year.is_some()
            && self.month == Some(1)
            && self.day == Some(1)
            && self.hour.is_none()
            && self.minute.is_none()
            && self.second.is_none()
    }
}

#[pymethods]
impl ParsedTextRepr {
    #[new]
    fn new(
        origin: String,
        normed_string: String,
        number: Option<ParsedNumberRepr>,
        datetime: Option<ParsedDatetimeRepr>,
    ) -> Self {
        Self {
            origin,
            normed_string,
            number,
            datetime,
        }
    }
}

#[pymethods]
impl ParsedNumberRepr {
    #[new]
    fn new(
        number: f64,
        number_string: String,
        is_integer: bool,
        unit: Option<String>,
        prob: Option<f64>,
    ) -> Self {
        Self {
            number,
            number_string,
            is_integer,
            unit,
            prob,
        }
    }
}

#[pymethods]
impl ParsedDatetimeRepr {
    #[new]
    fn new(
        year: Option<i64>,
        month: Option<i64>,
        day: Option<i64>,
        hour: Option<i64>,
        minute: Option<i64>,
        second: Option<i64>,
    ) -> Self {
        Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
        }
    }
}
