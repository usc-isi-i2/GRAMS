/// A struct for storing parsing results of a text literal
/// for literal matching.
pub struct ParsedTextRepr {
    pub origin: String,
    pub normed_string: String,
    pub number: Option<ParsedNumberRepr>,
    pub datetime: Option<ParsedDatetimeRepr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedDatetimeRepr {
    pub year: Option<i64>,
    pub month: Option<i64>,
    pub day: Option<i64>,
    pub hour: Option<i64>,
    pub minute: Option<i64>,
    pub second: Option<i64>,
}

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
