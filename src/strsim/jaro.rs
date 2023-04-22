use crate::error::GramsError;

use super::{ExpectTokenizerType, StrSim, TokenizerType};

use anyhow::Result;
use derive_more::Display;

#[derive(Display)]
#[display(fmt = "Jaro")]
pub struct Jaro;

impl Jaro {
    pub fn similarity(s1: &[char], s2: &[char]) -> f64 {
        let max_len = s1.len().max(s2.len());

        if max_len == 0 {
            return 1.0;
        } else if s1.len() == 0 || s2.len() == 0 {
            return 0.0;
        }

        let search_range = ((max_len / 2) - 1).max(0); // equal floor(max_len as f64 / 2) - 1;

        let mut flags_s1 = vec![false; s1.len()];
        let mut flags_s2 = vec![false; s2.len()];

        // find number of matching characters (common_chars)
        let mut common_chars = 0;

        for i in 0..s1.len() {
            let low = if i > search_range {
                i - search_range
            } else {
                0
            };
            let high = (i + search_range).min(s2.len() - 1);
            for j in low..=high {
                if flags_s2[j] == false && s2[j] == s1[i] {
                    flags_s1[i] = true;
                    flags_s2[j] = true;
                    common_chars += 1;
                    break;
                }
            }
        }

        if common_chars == 0 {
            return 0.0;
        }

        // find the number of transpositions and jaro distance
        let mut trans_count = 0;
        let mut k = 0;

        for i in 0..s1.len() {
            if flags_s1[i] == true {
                let mut pivot = k;
                for j in k..s2.len() {
                    if flags_s2[j] == true {
                        k = j + 1;
                        pivot = j;
                        break;
                    }
                }
                if s1[i] != s2[pivot] {
                    trans_count += 1;
                }
            }
        }

        trans_count /= 2;
        return ((common_chars as f64) / (s1.len() as f64)
            + (common_chars as f64) / (s2.len() as f64)
            + ((common_chars - trans_count) as f64) / (common_chars as f64))
            / 3.0;
    }
}

impl StrSim<Vec<char>> for Jaro {
    fn similarity_pre_tok2(
        &self,
        tokenized_key: &Vec<char>,
        tokenized_query: &Vec<char>,
    ) -> Result<f64, GramsError> {
        Ok(Jaro::similarity(tokenized_key, tokenized_query))
    }
}

impl ExpectTokenizerType for Jaro {
    fn get_expected_tokenizer_type(&self) -> TokenizerType {
        TokenizerType::Seq(Box::new(None))
    }
}
