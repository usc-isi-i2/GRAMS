use crate::error::GramsError;

use super::StrSim;

use anyhow::Result;

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

    fn jaro_winkler(&self, s1: &[char], s2: &[char]) -> f64 {
        let mut jw_score = self.jaro_distance(s1, s2);
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

    fn jaro_distance(&self, s1: &[char], s2: &[char]) -> f64 {
        let max_len = s1.len().max(s2.len());
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
                let mut found = false;
                for j in k..s2.len() {
                    if flags_s2[j] == true {
                        k = j + 1;
                        found = true;
                        break;
                    }
                }
                if !found {
                    // equivalent to s1[i] != s2[j]
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

impl StrSim<Vec<char>> for JaroWinkler {
    fn similarity_pre_tok2(
        &self,
        tokenized_key: &Vec<char>,
        tokenized_query: &Vec<char>,
    ) -> Result<f64, GramsError> {
        Ok(self.jaro_winkler(tokenized_key, tokenized_query))
    }
}
