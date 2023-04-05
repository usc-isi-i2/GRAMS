use crate::error::GramsError;

use super::{ExpectTokenizerType, JaroWinkler, StrSim, TokenizerType};
use derive_more::Display;

#[derive(Display)]
#[display(fmt = "MongeElkan")]
pub struct MongeElkan<S: StrSim<Vec<char>>> {
    pub strsim: S,
    // This is for early exit. If the similarity is not possible to satisfy this value,
    // the function returns immediately with the return value 0.0. Defaults to None.
    pub lower_bound: f64,
}
#[derive(Display)]
#[display(fmt = "SymmetricMongeElkan")]
pub struct SymmetricMongeElkan<S: StrSim<Vec<char>>>(MongeElkan<S>);

impl MongeElkan<JaroWinkler> {
    pub fn default() -> Self {
        MongeElkan {
            strsim: JaroWinkler::default(),
            lower_bound: 0.0,
        }
    }
}

impl<S: StrSim<Vec<char>>> MongeElkan<S> {
    pub fn new(strsim: S, lower_bound: Option<f64>) -> Self {
        MongeElkan {
            strsim,
            lower_bound: lower_bound.unwrap_or(0.0),
        }
    }

    pub fn similarity(
        &self,
        bag1: &Vec<Vec<char>>,
        bag2: &Vec<Vec<char>>,
    ) -> Result<f64, GramsError> {
        if bag1.len() == 0 || bag2.len() == 0 {
            if bag1.len() == 0 && bag2.len() == 0 {
                return Ok(1.0);
            } else {
                return Ok(0.0);
            }
        }

        let mut score_sum = 0.0;
        for (idx, ele1) in bag1.iter().enumerate() {
            let mut max_score = self.strsim.similarity_pre_tok2(ele1, &bag2[0])?;
            for ele2 in &bag2[1..] {
                let score = self.strsim.similarity_pre_tok2(ele1, ele2)?;
                if score > max_score {
                    max_score = score;
                }
            }
            score_sum += max_score;

            // if it satisfies early exit condition
            if self.lower_bound > 0.0 {
                let rest_max = (bag1.len() - 1 - idx) as f64; // assume the rest scores are all 1
                if (score_sum + rest_max / bag1.len() as f64) < self.lower_bound {
                    return Ok(0.0);
                }
            }
        }

        let sim = score_sum / bag1.len() as f64;
        if self.lower_bound > 0.0 && sim < self.lower_bound {
            return Ok(0.0);
        }
        return Ok(sim);
    }

    pub fn symmetric_similarity(
        &self,
        bag1: &Vec<Vec<char>>,
        bag2: &Vec<Vec<char>>,
    ) -> Result<f64, GramsError> {
        let sim1 = self.similarity(bag1, bag2)?;
        if self.lower_bound > 0.0 && sim1 == 0.0 {
            return Ok(0.0);
        }
        let sim2 = self.similarity(bag2, bag1)?;
        if self.lower_bound > 0.0 && sim2 == 0.0 {
            return Ok(0.0);
        }

        return Ok((sim1 + sim2) / 2.0);
    }
}

impl<S: StrSim<Vec<char>>> StrSim<Vec<Vec<char>>> for MongeElkan<S> {
    fn similarity_pre_tok2(
        &self,
        bag1: &Vec<Vec<char>>,
        bag2: &Vec<Vec<char>>,
    ) -> Result<f64, GramsError> {
        self.similarity(bag1, bag2)
    }
}

impl<S: StrSim<Vec<char>>> StrSim<Vec<Vec<char>>> for SymmetricMongeElkan<S> {
    fn similarity_pre_tok2(
        &self,
        bag1: &Vec<Vec<char>>,
        bag2: &Vec<Vec<char>>,
    ) -> Result<f64, GramsError> {
        self.0.symmetric_similarity(bag1, bag2)
    }
}

impl<S: StrSim<Vec<char>> + ExpectTokenizerType> ExpectTokenizerType for MongeElkan<S> {
    fn get_expected_tokenizer_type(&self) -> TokenizerType {
        TokenizerType::Seq(Box::new(Some(self.strsim.get_expected_tokenizer_type())))
    }
}

impl<S: StrSim<Vec<char>> + ExpectTokenizerType> ExpectTokenizerType for SymmetricMongeElkan<S> {
    fn get_expected_tokenizer_type(&self) -> TokenizerType {
        TokenizerType::Seq(Box::new(Some(self.0.strsim.get_expected_tokenizer_type())))
    }
}
