use crate::strsim::{CharacterTokenizer, Levenshtein, Tokenizer};
use crate::{context::AlgoContext, error::GramsError};
use anyhow::Result;
use kgdata::models::Value;

use super::{parsed_text_repr::ParsedTextRepr, SingleTypeMatcher};
use itertools::Itertools;

pub struct EntitySimilarityTest {
    pub strsim: Levenshtein,
}

impl EntitySimilarityTest {
    pub fn default() -> Self {
        Self {
            strsim: Levenshtein::default(),
        }
    }

    fn tokenize_entity_labels(
        &self,
        entid: &str,
        tokenizer: &mut CharacterTokenizer,
        context: &AlgoContext,
    ) -> Vec<Vec<char>> {
        let it = if let Some(entity) = context.entities.get(entid) {
            entity
                .label
                .lang2value
                .values()
                .chain(entity.aliases.lang2values.values().flatten())
        } else {
            let entity = context.entity_metadata.get(entid).unwrap();
            entity
                .label
                .lang2value
                .values()
                .chain(entity.aliases.lang2values.values().flatten())
        };

        it.unique().map(|s| tokenizer.tokenize(s)).collect()
    }
}

impl SingleTypeMatcher for EntitySimilarityTest {
    fn get_name(&self) -> &'static str {
        "entity_similarity_test"
    }

    fn compare(
        &self,
        query: &ParsedTextRepr,
        key: &Value,
        context: &AlgoContext,
    ) -> Result<(bool, f64), GramsError> {
        let entid = &key.as_entity_id().unwrap().id;
        if !context.entities.contains_key(entid) && !context.entity_metadata.contains_key(entid) {
            return Err(GramsError::DBIntegrityError(entid.clone()));
        }

        let mut tokenizer = CharacterTokenizer {};
        let mut queries = query
            .normed_string
            .split(",")
            .map(|s| tokenizer.tokenize(s.trim()))
            .collect::<Vec<_>>();
        queries.push(tokenizer.tokenize(&query.normed_string));

        let entity_labels = self.tokenize_entity_labels(entid, &mut tokenizer, context);

        let res = entity_labels
            .iter()
            .map(|lbl| {
                queries
                    .iter()
                    .map(|q| string_match_similarity(lbl, q, &self.strsim))
                    .filter(|(m, _s)| *m)
                    .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
            })
            .filter(|x| x.is_some() && x.unwrap().0)
            .map(|x| x.unwrap().1)
            .max_by(|x, y| x.partial_cmp(&y).unwrap());

        match res {
            Some(s) => Ok((true, s)),
            None => Ok((false, 0.0)),
        }
    }
}

/// Compare if two strings are similar, this is a direct translation
/// from `grams.algorithm.literal_matchers.string_match`
fn string_match_similarity(s1: &Vec<char>, s2: &Vec<char>, strsim: &Levenshtein) -> (bool, f64) {
    if s1 == s2 {
        return (true, 1.0);
    }
    let distance = strsim.distance(s1, s2);
    if distance <= 1.0 {
        return (true, 0.95);
    }
    if distance > 1.0 && (distance as f64 / s1.len().min(s2.len()).max(1) as f64) < 0.03 {
        return (true, 0.85);
    }
    return (false, 0.0);
}
