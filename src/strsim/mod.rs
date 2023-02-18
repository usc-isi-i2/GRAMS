mod levenshtein;
use anyhow::Result;

pub use self::levenshtein::Levenshtein;

pub trait StrSim {
    /**
     * Calculate the similarity between two strings. Usually, the similarity function is symmetric so
     * key and query can be swapped. However, some functions such as monge-elkan are not symmetric, so
     * key and query takes specific meaning: key is the value in the database and query is the search
     * query from the user.
     *
     * The return value is a likelihood between 0 and 1.
     */
    fn similarity(&self, key: &str, query: &str) -> Result<f64>;
}
