use kgdata::models::Value;

use crate::{context::AlgoContext, error::GramsError};

use self::parsed_text_repr::ParsedTextRepr;
use anyhow::Result;

pub mod entity_match;
pub mod globecoordinate_match;
pub mod monolingual_text_match;
pub mod parsed_text_repr;
pub mod quantity_match;
pub mod string_match;
pub mod time_match;
use pyo3::prelude::*;

#[pyclass(module = "grams.core.literal_matchers", name = "LiteralMatcherConfig")]
pub struct LiteralMatcherConfig {
    pub string: String,
    pub quantity: String,
    pub globecoordinate: String,
    pub time: String,
    pub monolingual_text: String,
    pub entity: String,
}

#[pyclass(
    module = "grams.core.literal_matchers",
    name = "LiteralMatcher",
    unsendable
)]
pub struct PyLiteralMatchers(pub LiteralMatcher);

pub struct LiteralMatcher {
    pub string_matcher: Box<dyn SingleTypeMatcher>,
    pub quantity_matcher: Box<dyn SingleTypeMatcher>,
    pub globecoordinate_matcher: Box<dyn SingleTypeMatcher>,
    pub time_matcher: Box<dyn SingleTypeMatcher>,
    pub monolingual_text_matcher: Box<dyn SingleTypeMatcher>,
    pub entity_matcher: Box<dyn SingleTypeMatcher>,
}

impl LiteralMatcher {
    pub fn new(cfg: LiteralMatcherConfig) -> Result<Self, GramsError> {
        let string_matcher = match cfg.string.as_str() {
            "string_exact_test" => Box::new(self::string_match::StringExactTest),
            _ => Err(GramsError::InvalidConfigData(format!(
                "Invalid string matcher: {}",
                cfg.string
            )))?,
        };
        let monolingual_text_matcher = match cfg.monolingual_text.as_str() {
            "monolingual_exact_test" => {
                Box::new(self::monolingual_text_match::MonolingualTextExactTest)
            }
            _ => Err(GramsError::InvalidConfigData(format!(
                "Invalid monolingual_text matcher: {}",
                cfg.monolingual_text
            )))?,
        };
        let quantity_matcher = match cfg.quantity.as_str() {
            "quantity_test" => Box::new(self::quantity_match::QuantityTest),
            _ => Err(GramsError::InvalidConfigData(format!(
                "Invalid quantity matcher: {}",
                cfg.quantity
            )))?,
        };
        let time_matcher = match cfg.time.as_str() {
            "time_test" => Box::new(self::time_match::TimeTest::default()),
            _ => Err(GramsError::InvalidConfigData(format!(
                "Invalid time matcher: {}",
                cfg.time
            )))?,
        };
        let globecoordinate_matcher = match cfg.globecoordinate.as_str() {
            "globecoordinate_test" => Box::new(self::globecoordinate_match::GlobeCoordinateTest),
            _ => Err(GramsError::InvalidConfigData(format!(
                "Invalid globecoordinate matcher: {}",
                cfg.globecoordinate
            )))?,
        };
        let entity_matcher = match cfg.entity.as_str() {
            "entity_similarity_test" => {
                Box::new(self::entity_match::EntitySimilarityTest::default())
            }
            _ => Err(GramsError::InvalidConfigData(format!(
                "Invalid entity matcher: {}",
                cfg.entity
            )))?,
        };

        Ok(LiteralMatcher {
            string_matcher,
            monolingual_text_matcher,
            quantity_matcher,
            time_matcher,
            globecoordinate_matcher,
            entity_matcher,
        })
    }

    /// Test if the query matches the key. Return a list of tuple of name of matched function and
    /// matched score.
    pub fn compare(
        &self,
        query: &ParsedTextRepr,
        key: &Value,
        context: &AlgoContext,
    ) -> Result<Option<(&'static str, f64)>, GramsError> {
        let matcher = match key {
            Value::String(_) => &self.string_matcher,
            Value::Quantity(_) => &self.quantity_matcher,
            Value::GlobeCoordinate(_) => &self.globecoordinate_matcher,
            Value::Time(_) => &self.time_matcher,
            Value::MonolingualText(_) => &self.monolingual_text_matcher,
            Value::EntityId(_) => &self.entity_matcher,
        };
        let (is_matched, score) = matcher.compare(query, key, context)?;
        if is_matched {
            Ok(Some((self.string_matcher.get_name(), score)))
        } else {
            Ok(None)
        }
    }
}

pub trait SingleTypeMatcher {
    /// Get the name of the matcher
    fn get_name(&self) -> &'static str;

    /// Test if the query matches the key and return the a tuple of (is_match, score), in which
    /// score is between 0 and 1 (inclusive)
    fn compare(
        &self,
        query: &ParsedTextRepr,
        key: &Value,
        context: &AlgoContext,
    ) -> Result<(bool, f64), GramsError>;
}
