use crate::{context::AlgoContext, error::GramsError};
use anyhow::Result;
use kgdata::models::Value;

use super::{parsed_text_repr::ParsedTextRepr, SingleTypeMatcher};

pub struct QuantityTest;

impl SingleTypeMatcher for QuantityTest {
    fn get_name(&self) -> &'static str {
        "quantity_test"
    }

    fn compare(
        &self,
        query: &ParsedTextRepr,
        key: &Value,
        context: &AlgoContext,
    ) -> Result<(bool, f64), GramsError> {
        let num = match &query.number {
            None => return Ok((false, 0.0)),
            Some(num) => num.number,
        };

        let kgnum = match key.as_quantity().unwrap().amount.parse::<f64>() {
            Ok(num) => num,
            Err(_) => {
                return Err(GramsError::IntegrityError(format!(
                    "Invalid number in KG: {}",
                    key.as_quantity().unwrap().amount
                )));
            }
        };

        if (kgnum - num).abs() < 1e-5 {
            return Ok((true, 1.0));
        }

        let diff_percentage = if kgnum == 0.0 {
            (kgnum - num).abs() / 1e-7
        } else {
            ((kgnum - num) / kgnum).abs()
        };

        if diff_percentage < 0.05 {
            // within 5%
            return Ok((true, 0.95 - diff_percentage));
        } else {
            return Ok((false, 0.0));
        }
    }
}
