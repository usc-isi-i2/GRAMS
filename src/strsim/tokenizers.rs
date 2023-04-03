use super::Tokenizer;
use crate::helper::{ByReference, ByValue, ReturnKind};
use hashbrown::HashMap;

pub struct CharacterTokenizer;

pub struct WhitespaceTokenizer;

pub struct CachedWhitespaceTokenizer {
    cache: HashMap<String, Vec<String>>,
    unique_cache: HashMap<String, Vec<String>>,
}

impl Tokenizer<Vec<char>> for CharacterTokenizer {
    type Return = ByValue;
    fn tokenize<'t>(&'t mut self, s: &str) -> Vec<char> {
        s.chars().collect()
    }

    fn tokenize_pair<'t>(&'t mut self, key: &str, query: &str) -> (Vec<char>, Vec<char>) {
        (key.chars().collect(), query.chars().collect())
    }

    fn unique_tokenize<'t>(
        &'t mut self,
        s: &str,
    ) -> <Self::Return as ReturnKind<'t, Vec<char>>>::Type {
        let mut chars = s.chars().collect::<Vec<char>>();
        chars.dedup();
        chars
    }

    fn unique_tokenize_pair<'t>(
        &'t mut self,
        key: &str,
        query: &str,
    ) -> (
        <Self::Return as ReturnKind<'t, Vec<char>>>::Type,
        <Self::Return as ReturnKind<'t, Vec<char>>>::Type,
    ) {
        (self.unique_tokenize(key), self.unique_tokenize(query))
    }
}

impl Tokenizer<Vec<String>> for WhitespaceTokenizer {
    type Return = ByValue;

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
        let mut tokens: Vec<_> = s.split_whitespace().map(|s| s.to_owned()).collect();
        tokens.dedup();
        tokens
    }

    fn unique_tokenize_pair<'t>(
        &'t mut self,
        key: &str,
        query: &str,
    ) -> (Vec<String>, Vec<String>) {
        let mut key_tokens: Vec<_> = key.split_whitespace().map(|s| s.to_owned()).collect();
        key_tokens.dedup();
        let mut query_tokens: Vec<_> = query.split_whitespace().map(|s| s.to_owned()).collect();
        query_tokens.dedup();

        (key_tokens, query_tokens)
    }
}

impl Tokenizer<Vec<String>> for CachedWhitespaceTokenizer {
    type Return = ByReference;

    fn tokenize<'t>(&'t mut self, s: &str) -> &'t Vec<String> {
        if !self.cache.contains_key(s) {
            let tokens: Vec<String> = s.split_whitespace().map(|s| s.to_owned()).collect();
            self.cache.insert(s.to_owned(), tokens.clone());
        }

        self.cache.get(s).unwrap()
    }

    fn tokenize_pair<'t>(
        &'t mut self,
        key: &str,
        query: &str,
    ) -> (&'t Vec<String>, &'t Vec<String>) {
        if !self.cache.contains_key(key) {
            let tokens: Vec<String> = key.split_whitespace().map(|s| s.to_owned()).collect();
            self.cache.insert(key.to_owned(), tokens.clone());
        }

        if !self.cache.contains_key(query) {
            let tokens: Vec<String> = query.split_whitespace().map(|s| s.to_owned()).collect();
            self.cache.insert(query.to_owned(), tokens.clone());
        }

        (self.cache.get(key).unwrap(), self.cache.get(query).unwrap())
    }

    fn unique_tokenize<'t>(&'t mut self, s: &str) -> &'t Vec<String> {
        if !self.unique_cache.contains_key(s) {
            let mut tokens: Vec<String> = s.split_whitespace().map(|s| s.to_owned()).collect();
            tokens.dedup();
            self.unique_cache.insert(s.to_owned(), tokens.clone());
        }

        self.unique_cache.get(s).unwrap()
    }

    fn unique_tokenize_pair<'t>(
        &'t mut self,
        key: &str,
        query: &str,
    ) -> (&'t Vec<String>, &'t Vec<String>) {
        if !self.unique_cache.contains_key(key) {
            let mut tokens: Vec<String> = key.split_whitespace().map(|s| s.to_owned()).collect();
            tokens.dedup();
            self.unique_cache.insert(key.to_owned(), tokens.clone());
        }

        if !self.unique_cache.contains_key(query) {
            let mut tokens: Vec<String> = query.split_whitespace().map(|s| s.to_owned()).collect();
            tokens.dedup();
            self.unique_cache.insert(query.to_owned(), tokens.clone());
        }

        (
            self.unique_cache.get(key).unwrap(),
            self.unique_cache.get(query).unwrap(),
        )
    }
}
