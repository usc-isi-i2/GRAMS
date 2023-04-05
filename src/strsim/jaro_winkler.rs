use crate::error::GramsError;

use super::{ExpectTokenizerType, Jaro, StrSim, TokenizerType};

use anyhow::Result;
use derive_more::Display;

#[derive(Display)]
#[display(fmt = "JaroWinkler")]
pub struct JaroWinkler {
    // Boost threshold, prefix bonus is only added when compared strings have a Jaro Distance above it. Defaults to 0.7.
    pub threshold: f64,
    // Scaling factor for how much the score is adjusted upwards for having common prefixes. Defaults to 0.1.
    pub scaling_factor: f64,
    pub prefix_len: usize,
}

impl JaroWinkler {
    pub fn default() -> Self {
        JaroWinkler {
            threshold: 0.7,
            scaling_factor: 0.1,
            prefix_len: 4,
        }
    }

    pub fn new(
        threshold: Option<f64>,
        scaling_factor: Option<f64>,
        prefix_len: Option<usize>,
    ) -> Self {
        JaroWinkler {
            threshold: threshold.unwrap_or(0.7),
            scaling_factor: scaling_factor.unwrap_or(0.1),
            prefix_len: prefix_len.unwrap_or(4),
        }
    }

    fn similarity(&self, s1: &[char], s2: &[char]) -> f64 {
        let mut jw_score = Jaro::similarity(s1, s2);
        if jw_score > self.threshold {
            // common prefix len
            let mut common_prefix_len = 0;

            let max_common_prefix_len = s1.len().min(s2.len()).min(self.prefix_len);
            while common_prefix_len < max_common_prefix_len
                && s1[common_prefix_len] == s2[common_prefix_len]
            {
                common_prefix_len += 1;
            }
            if common_prefix_len != 0 {
                jw_score += self.scaling_factor * (common_prefix_len as f64) * (1.0 - jw_score);
            }
        }

        jw_score
    }
}

impl StrSim<Vec<char>> for JaroWinkler {
    fn similarity_pre_tok2(
        &self,
        tokenized_key: &Vec<char>,
        tokenized_query: &Vec<char>,
    ) -> Result<f64, GramsError> {
        Ok(self.similarity(tokenized_key, tokenized_query))
    }
}

impl ExpectTokenizerType for JaroWinkler {
    fn get_expected_tokenizer_type(&self) -> TokenizerType {
        TokenizerType::Seq(Box::new(None))
    }
}
