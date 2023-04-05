use std::fmt::Display;

use super::super::{ExpectTokenizerType, StrSim, StrSimWithTokenizer, Tokenizer};
use crate::{error::GramsError, helper::ByValue};
use anyhow::Result;

pub struct SeqStrSim<
    't,
    T,
    SS: StrSim<T> + Display + ExpectTokenizerType,
    TK: Tokenizer<T, Return = ByValue> + Display,
> {
    pub tokenizer: &'t mut TK,
    pub strsim: SS,
    phantom: std::marker::PhantomData<T>,
}

impl<
        't,
        T,
        SS: StrSim<T> + Display + ExpectTokenizerType,
        TK: Tokenizer<T, Return = ByValue> + Display,
    > SeqStrSim<'t, T, SS, TK>
{
    pub fn new(tokenizer: &'t mut TK, strsim: SS) -> Result<Self, GramsError> {
        let expect_tok_type = strsim.get_expected_tokenizer_type();
        if !tokenizer.is_compatible(&expect_tok_type) || !expect_tok_type.is_outer_seq() {
            Err(GramsError::InvalidConfigData(format!(
                "StrSim {} expect {:?} tokenizer, but get {} which is not compatible",
                strsim, expect_tok_type, tokenizer,
            )))
        } else {
            Ok(Self {
                tokenizer,
                strsim,
                phantom: std::marker::PhantomData,
            })
        }
    }
}

impl<
        't,
        T,
        SS: StrSim<T> + Display + ExpectTokenizerType,
        TK: Tokenizer<T, Return = ByValue> + Display,
    > StrSimWithTokenizer<T> for SeqStrSim<'t, T, SS, TK>
{
    fn similarity(&mut self, key: &str, query: &str) -> Result<f64, GramsError> {
        let (s1, s2) = self.tokenizer.tokenize_pair(key, query);
        self.strsim.similarity_pre_tok2(&s1, &s2)
    }

    fn similarity_pre_tok1(&mut self, key: &str, tokenized_query: &T) -> Result<f64, GramsError> {
        let s1 = self.tokenizer.tokenize(key);
        self.strsim.similarity_pre_tok2(&s1, tokenized_query)
    }

    fn tokenize(&mut self, str: &str) -> T {
        self.tokenizer.tokenize(str)
    }

    fn tokenize_list(&mut self, strs: &[&str]) -> Vec<T> {
        strs.iter()
            .map(|s| self.tokenizer.tokenize(s))
            .collect::<Vec<T>>()
    }
}

impl<
        't,
        T,
        SS: StrSim<T> + Display + ExpectTokenizerType,
        TK: Tokenizer<T, Return = ByValue> + Display,
    > StrSim<T> for SeqStrSim<'t, T, SS, TK>
{
    fn similarity_pre_tok2(
        &self,
        tokenized_key: &T,
        tokenized_query: &T,
    ) -> Result<f64, GramsError> {
        self.strsim
            .similarity_pre_tok2(tokenized_key, tokenized_query)
    }
}
