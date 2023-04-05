use super::{Tokenizer, TokenizerType};
use crate::helper::{ByReference, ByValue};
use derive_more::Display;
use hashbrown::HashMap;
use itertools::Itertools;

#[derive(Display)]
#[display(fmt = "CharacterTokenizer")]
pub struct CharacterTokenizer;

#[derive(Display)]
#[display(fmt = "WhitespaceTokenizer")]
pub struct WhitespaceTokenizer;

#[derive(Display)]
#[display(fmt = "WhitespaceCharSeqTokenizer")]
pub struct WhitespaceCharSeqTokenizer;

pub struct CachedWhitespaceTokenizer {
    cache: HashMap<String, Vec<String>>,
    unique_cache: HashMap<String, Vec<String>>,
}

impl Tokenizer<Vec<char>> for CharacterTokenizer {
    type Return = ByValue;

    fn is_compatible(&self, tok_type: &TokenizerType) -> bool {
        !tok_type.has_nested()
    }

    fn tokenize<'t>(&'t mut self, s: &str) -> Vec<char> {
        s.chars().collect()
    }

    fn tokenize_pair<'t>(&'t mut self, key: &str, query: &str) -> (Vec<char>, Vec<char>) {
        (key.chars().collect(), query.chars().collect())
    }

    fn unique_tokenize<'t>(&'t mut self, s: &str) -> Vec<char> {
        s.chars().unique().collect::<Vec<char>>()
    }

    fn unique_tokenize_pair<'t>(&'t mut self, key: &str, query: &str) -> (Vec<char>, Vec<char>) {
        (self.unique_tokenize(key), self.unique_tokenize(query))
    }
}

impl Tokenizer<Vec<String>> for WhitespaceTokenizer {
    type Return = ByValue;

    fn is_compatible(&self, tok_type: &TokenizerType) -> bool {
        !tok_type.has_nested()
    }

    fn tokenize<'t>(&'t mut self, s: &str) -> Vec<String> {
        s.split_whitespace().map(|s| s.to_owned()).collect()
    }

    fn tokenize_pair<'t>(&'t mut self, key: &str, query: &str) -> (Vec<String>, Vec<String>) {
        (
            key.split_whitespace().map(|s| s.to_owned()).collect(),
            query.split_whitespace().map(|s| s.to_owned()).collect(),
        )
    }

    fn unique_tokenize<'t>(&'t mut self, s: &str) -> Vec<String> {
        s.split_whitespace()
            .unique()
            .map(|s| s.to_owned())
            .collect()
    }

    fn unique_tokenize_pair<'t>(
        &'t mut self,
        key: &str,
        query: &str,
    ) -> (Vec<String>, Vec<String>) {
        let key_tokens: Vec<_> = key
            .split_whitespace()
            .unique()
            .map(|s| s.to_owned())
            .collect();
        let query_tokens: Vec<_> = query
            .split_whitespace()
            .unique()
            .map(|s| s.to_owned())
            .collect();

        (key_tokens, query_tokens)
    }
}

impl Tokenizer<Vec<Vec<char>>> for WhitespaceCharSeqTokenizer {
    type Return = ByValue;

    fn is_compatible(&self, tok_type: &TokenizerType) -> bool {
        if let Some(nested_tok_type) = tok_type.get_nested() {
            nested_tok_type.is_outer_seq()
        } else {
            false
        }
    }

    fn tokenize<'t>(&'t mut self, s: &str) -> Vec<Vec<char>> {
        s.split_whitespace().map(|s| s.chars().collect()).collect()
    }

    fn tokenize_pair<'t>(&'t mut self, key: &str, query: &str) -> (Vec<Vec<char>>, Vec<Vec<char>>) {
        (
            key.split_whitespace()
                .map(|s| s.chars().collect())
                .collect(),
            query
                .split_whitespace()
                .map(|s| s.chars().collect())
                .collect(),
        )
    }

    fn unique_tokenize<'t>(&'t mut self, s: &str) -> Vec<Vec<char>> {
        s.split_whitespace()
            .unique()
            .map(|s| s.chars().collect())
            .collect()
    }

    fn unique_tokenize_pair<'t>(
        &'t mut self,
        key: &str,
        query: &str,
    ) -> (Vec<Vec<char>>, Vec<Vec<char>>) {
        let key_tokens: Vec<_> = key
            .split_whitespace()
            .unique()
            .map(|s| s.chars().collect())
            .collect();
        let query_tokens: Vec<_> = query
            .split_whitespace()
            .unique()
            .map(|s| s.chars().collect())
            .collect();

        (key_tokens, query_tokens)
    }
}

impl Tokenizer<Vec<String>> for CachedWhitespaceTokenizer {
    type Return = ByReference;

    fn is_compatible(&self, tok_type: &TokenizerType) -> bool {
        !tok_type.has_nested()
    }

    fn tokenize<'t>(&'t mut self, s: &str) -> &'t Vec<String> {
        if !self.cache.contains_key(s) {
            self.cache.insert(
                s.to_owned(),
                s.split_whitespace().map(|s| s.to_owned()).collect(),
            );
        }

        self.cache.get(s).unwrap()
    }

    fn tokenize_pair<'t>(
        &'t mut self,
        key: &str,
        query: &str,
    ) -> (&'t Vec<String>, &'t Vec<String>) {
        if !self.cache.contains_key(key) {
            self.cache.insert(
                key.to_owned(),
                key.split_whitespace().map(|s| s.to_owned()).collect(),
            );
        }

        if !self.cache.contains_key(query) {
            self.cache.insert(
                query.to_owned(),
                query.split_whitespace().map(|s| s.to_owned()).collect(),
            );
        }

        (self.cache.get(key).unwrap(), self.cache.get(query).unwrap())
    }

    fn unique_tokenize<'t>(&'t mut self, s: &str) -> &'t Vec<String> {
        if !self.unique_cache.contains_key(s) {
            self.unique_cache.insert(
                s.to_owned(),
                s.split_whitespace()
                    .unique()
                    .map(|s| s.to_owned())
                    .collect(),
            );
        }

        self.unique_cache.get(s).unwrap()
    }

    fn unique_tokenize_pair<'t>(
        &'t mut self,
        key: &str,
        query: &str,
    ) -> (&'t Vec<String>, &'t Vec<String>) {
        if !self.unique_cache.contains_key(key) {
            self.unique_cache.insert(
                key.to_owned(),
                key.split_whitespace()
                    .unique()
                    .map(|s| s.to_owned())
                    .collect(),
            );
        }

        if !self.unique_cache.contains_key(query) {
            self.unique_cache.insert(
                query.to_owned(),
                query
                    .split_whitespace()
                    .unique()
                    .map(|s| s.to_owned())
                    .collect(),
            );
        }

        (
            self.unique_cache.get(key).unwrap(),
            self.unique_cache.get(query).unwrap(),
        )
    }
}
