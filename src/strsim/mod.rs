mod hybrid_jaccard;
mod jaro_winkler;
mod levenshtein;
mod monge_elkan;
mod strsim_wrapper;
mod tokenizers;
use anyhow::Result;

use crate::error::GramsError;
use crate::helper::ReturnKind;

pub use self::jaro_winkler::JaroWinkler;
pub use self::levenshtein::Levenshtein;
pub use self::monge_elkan::{MongeElkan, SymmetricMongeElkan};
pub use self::strsim_wrapper::{StrSimWithRefTokenizer, StrSimWithValueTokenizer};
pub use self::tokenizers::{CachedWhitespaceTokenizer, CharacterTokenizer, WhitespaceTokenizer};

pub trait StrSim<T> {
    /** Calculate the similarity with both key and query has already``` been pre-tokenized */
    fn similarity_pre_tok2(
        &self,
        tokenized_key: &T,
        tokenized_query: &T,
    ) -> Result<f64, GramsError>;
}

pub trait StrSimWithTokenizer<T>: StrSim<T> {
    /**
     * Calculate the similarity between two strings. Usually, the similarity function is symmetric so
     * key and query can be swapped. However, some functions such as monge-elkan are not symmetric, so
     * key and query takes specific meaning: key is the value in the database and query is the search
     * query from the user.
     *
     * The return value is a likelihood between 0 and 1.
     *
     * @param key the value in the database (e.g., entity label)
     * @param query the search query from the user (e.g., cell in the table)
     */
    fn similarity(&mut self, key: &str, query: &str) -> Result<f64, GramsError>;

    /** Calculate the similarity with the query's already been pre-tokenized */
    fn similarity_pre_tok1(&mut self, key: &str, tokenized_query: &T) -> Result<f64, GramsError>;

    /**
     * Tokenize a string into a tokens used for this method.
     */
    fn tokenize(&mut self, str: &str) -> T;

    /**
     * Tokenize a list of strings into a list of tokens used for this method.
     */
    fn tokenize_list(&mut self, strs: &[&str]) -> Vec<T>;
}

pub trait Tokenizer<T> {
    type Return: for<'t> ReturnKind<'t, T>;

    fn tokenize<'t>(&'t mut self, s: &str) -> <Self::Return as ReturnKind<'t, T>>::Type;
    fn tokenize_pair<'t>(
        &'t mut self,
        key: &str,
        query: &str,
    ) -> (
        <Self::Return as ReturnKind<'t, T>>::Type,
        <Self::Return as ReturnKind<'t, T>>::Type,
    );
    fn unique_tokenize<'t>(&'t mut self, s: &str) -> <Self::Return as ReturnKind<'t, T>>::Type;
    fn unique_tokenize_pair<'t>(
        &'t mut self,
        key: &str,
        query: &str,
    ) -> (
        <Self::Return as ReturnKind<'t, T>>::Type,
        <Self::Return as ReturnKind<'t, T>>::Type,
    );
}

impl<'t, T: Sized + 't> ReturnKind<'t, T> for Vec<char> {
    type Type = Vec<char>;
}

impl<'t, T: Sized + 't> ReturnKind<'t, T> for Vec<String> {
    type Type = Vec<String>;
}

impl<'t, T: Sized + 't> ReturnKind<'t, T> for &'t Vec<String> {
    type Type = &'t Vec<String>;
}
