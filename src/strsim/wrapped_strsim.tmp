use crate::{
    error::GramsError,
    helper::{ByReference, ByValue},
};

use super::{StrSim, StrSimWithTokenizer, Tokenizer};

pub struct StrSimWithValueTokenizer<'t, T, SS: StrSim<T>, TK: Tokenizer<T, Return = ByValue>> {
    pub tokenizer: &'t mut TK,
    pub strsim: SS,
    phantom: std::marker::PhantomData<T>,
}

impl<'t, T, SS: StrSim<T>, TK: Tokenizer<T, Return = ByValue>>
    StrSimWithValueTokenizer<'t, T, SS, TK>
{
    pub fn new(tokenizer: &'t mut TK, strsim: SS) -> Self {
        Self {
            tokenizer,
            strsim,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'t, T, SS: StrSim<T>, TK: Tokenizer<T, Return = ByValue>> StrSimWithTokenizer<T>
    for StrSimWithValueTokenizer<'t, T, SS, TK>
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

impl<'t, T, SS: StrSim<T>, TK: Tokenizer<T, Return = ByValue>> StrSim<T>
    for StrSimWithValueTokenizer<'t, T, SS, TK>
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

pub struct StrSimWithRefTokenizer<'t, T, SS: StrSim<T>, TK: Tokenizer<T, Return = ByReference>> {
    pub tokenizer: &'t mut TK,
    pub strsim: SS,
    phantom: std::marker::PhantomData<T>,
}

impl<'t, T, SS: StrSim<T>, TK: Tokenizer<T, Return = ByReference>>
    StrSimWithRefTokenizer<'t, T, SS, TK>
{
    pub fn new(tokenizer: &'t mut TK, strsim: SS) -> Self {
        Self {
            tokenizer,
            strsim,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'t, T: Clone, SS: StrSim<T>, TK: Tokenizer<T, Return = ByReference>> StrSimWithTokenizer<T>
    for StrSimWithRefTokenizer<'t, T, SS, TK>
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
        self.tokenizer.tokenize(str).clone()
    }

    fn tokenize_list(&mut self, strs: &[&str]) -> Vec<T> {
        strs.iter()
            .map(|s| self.tokenizer.tokenize(s).clone())
            .collect::<Vec<T>>()
    }
}

impl<'t, T, SS: StrSim<T>, TK: Tokenizer<T, Return = ByReference>> StrSim<T>
    for StrSimWithRefTokenizer<'t, T, SS, TK>
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
