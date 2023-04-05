use crate::{context::AlgoContext, error::GramsError};
use anyhow::Result;
use kgdata::models::Value;

use super::{parsed_text_repr::ParsedTextRepr, SingleTypeMatcher};

pub struct GlobeCoordinateTest;

impl SingleTypeMatcher for GlobeCoordinateTest {
    fn get_name(&self) -> &'static str {
        "globecoordinate_test"
    }

    fn compare(
        &self,
        _query: &ParsedTextRepr,
        _key: &Value,
        _context: &AlgoContext,
    ) -> Result<(bool, f64), GramsError> {
        Ok((false, 0.0))
    }
}
