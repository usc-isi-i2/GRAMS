use hashbrown::HashMap;

use crate::error::GramsError;

use super::StrSim;
use anyhow::Result;

pub struct Levenshtein {
    pub insertion: HashMap<char, f32>,
    pub insertion_default: f32,
    pub deletion: HashMap<char, f32>,
    pub deletion_default: f32,
    pub substitution: HashMap<char, HashMap<char, f32>>,
    pub substitution_default: f32,
    pub lowerbound: f64,
}

impl Levenshtein {
    pub fn default() -> Self {
        Levenshtein {
            insertion: HashMap::new(),
            insertion_default: 1.0,
            deletion: HashMap::new(),
            deletion_default: 1.0,
            substitution: HashMap::new(),
            substitution_default: 1.0,
            lowerbound: -1.0,
        }
    }

    pub fn compute_max_cost(&self, chars: &[char]) -> f32 {
        chars
            .iter()
            .map(|c| {
                self.insertion
                    .get(c)
                    .unwrap_or(&self.insertion_default)
                    .max(
                        self.deletion.get(c).unwrap_or(&self.deletion_default).max(
                            *self
                                .substitution
                                .get(c)
                                // RLTK has bug here, I haven't verified my fix
                                .map(|subs| {
                                    subs.values()
                                        .max_by(|&a, &b| a.partial_cmp(b).unwrap())
                                        .unwrap()
                                })
                                .unwrap_or(&self.substitution_default),
                        ),
                    )
            })
            .sum()
    }

    pub fn estimate_min_char_cost(&self, chars: &[char]) -> f32 {
        chars
            .iter()
            .map(|c| {
                self.insertion
                    .get(c)
                    .unwrap_or(&self.insertion_default)
                    .min(
                        self.deletion.get(c).unwrap_or(&self.deletion_default).min(
                            *self
                                .substitution
                                .get(c)
                                // RLTK has bug here, I haven't verified my fix
                                .map(|subs| {
                                    subs.values()
                                        .max_by(|&a, &b| a.partial_cmp(b).unwrap())
                                        .unwrap()
                                })
                                .unwrap_or(&self.substitution_default),
                        ),
                    )
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// The Levenshtein distance between two words is the minimum number of single-character edits (insertions,
    /// deletions or substitutions) required to change one word into the other.
    pub fn distance(&self, s1: &[char], s2: &[char]) -> f32 {
        let n1 = s1.len();
        let n2 = s2.len();
        if n1 == 0 && n2 == 0 {
            return 0.0;
        }

        let mut dp: Vec<Vec<f32>> = vec![vec![0.0; n2 + 1]; n1 + 1];
        for i in 0..=n1 {
            for j in 0..=n2 {
                if i == 0 && j == 0 {
                    continue;
                }

                if i == 0 {
                    // most top row
                    let c = &s2[j - 1];
                    dp[i][j] = *self.insertion.get(c).unwrap_or(&self.insertion_default);
                    dp[i][j] += dp[i][j - 1];
                } else if j == 0 {
                    // most left column
                    let c = &s1[i - 1];
                    dp[i][j] = *self.deletion.get(c).unwrap_or(&self.deletion_default);
                    dp[i][j] += dp[i - 1][j];
                } else {
                    let c1 = &s1[i - 1];
                    let c2 = &s2[j - 1];
                    let insert_cost = self.insertion.get(c2).unwrap_or(&self.insertion_default);
                    let delete_cost = self.deletion.get(c1).unwrap_or(&self.deletion_default);
                    let substitute_cost = self
                        .substitution
                        .get(c1)
                        .map(|subs| subs.get(c2).unwrap_or(&self.substitution_default))
                        .unwrap_or(&self.substitution_default);

                    if c1 == c2 {
                        dp[i][j] = dp[i - 1][j - 1];
                    } else {
                        dp[i][j] = (dp[i][j - 1] + insert_cost).min(
                            (dp[i - 1][j] + delete_cost).min(dp[i - 1][j - 1] + substitute_cost),
                        );
                    }
                }
            }
        }
        return dp[n1][n2];
    }
}

impl StrSim for Levenshtein {
    /**
     * Compute the Levenshtein similarity between two strings as
     * 1 - (levenshtein_distance / max_cost(key, query)).
     *
     * Directly translated from RLTK's implementation.
     */
    fn similarity(&self, key: &str, query: &str) -> Result<f64, GramsError> {
        let s1: Vec<char> = key.chars().collect();
        let s2: Vec<char> = query.chars().collect();

        let max_cost = self.compute_max_cost(&s1).max(self.compute_max_cost(&s2)) as f64;
        let min_lev: f64;

        if self.lowerbound > 0.0 {
            let diff = s1.len().abs_diff(s2.len()) as f64;
            if s1.len() == 0 && s2.len() == 0 {
                return Ok(1.0);
            }
            if s1.len() == 0 {
                min_lev = diff * self.estimate_min_char_cost(&s2) as f64;
            } else if s2.len() == 0 {
                min_lev = diff * self.estimate_min_char_cost(&s1) as f64;
            } else {
                min_lev = diff
                    * self
                        .estimate_min_char_cost(&s1)
                        .min(self.estimate_min_char_cost(&s2)) as f64;
            }
            let est_sim = 1.0 - (min_lev / max_cost);
            if est_sim < self.lowerbound {
                return Ok(0.0);
            }
        }

        let lev = self.distance(&s1, &s2) as f64;
        if max_cost < lev {
            return Err(GramsError::InvalidConfigData(
                "Illegal value of operation costs".to_owned(),
            ));
        }

        if max_cost == 0.0 {
            return Ok(1.0);
        }

        let lev_sim = 1.0 - (lev / max_cost);
        if self.lowerbound > 0.0 && lev_sim < self.lowerbound {
            return Ok(0.0);
        }
        Ok(lev_sim)
    }
}
