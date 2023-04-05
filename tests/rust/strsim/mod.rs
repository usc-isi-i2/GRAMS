pub mod hybrid_jaccard;
pub mod jaro;
pub mod jaro_winkler;
pub mod levenshtein;
pub mod monge_elkan;

use grams::{
    error::GramsError,
    strsim::{ExpectTokenizerType, StrSim},
};
use std::collections::HashMap;

pub struct SpecificStrSim {
    map: HashMap<&'static str, HashMap<&'static str, f64>>,
}

impl StrSim<Vec<char>> for SpecificStrSim {
    fn similarity_pre_tok2(
        &self,
        tokenized_key: &Vec<char>,
        tokenized_query: &Vec<char>,
    ) -> Result<f64, GramsError> {
        let mut k1 = tokenized_key.into_iter().collect::<String>();
        let mut k2 = tokenized_query.into_iter().collect::<String>();

        if !self.map.contains_key(&k1.as_str()) {
            (k1, k2) = (k2, k1);
        }

        Ok(*self
            .map
            .get(&k1.as_str())
            .unwrap()
            .get(&k2.as_str())
            .unwrap())
    }
}

impl ExpectTokenizerType for SpecificStrSim {
    fn get_expected_tokenizer_type(&self) -> grams::strsim::TokenizerType {
        grams::strsim::TokenizerType::Seq(Box::new(None))
    }
}
