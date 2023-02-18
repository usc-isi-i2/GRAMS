use anyhow::Result;
use rocksdb::{self, Options};

#[pyclass(module = "grams.core.db", subclass)]
pub struct GramsDB {
    datadir: Path,
    entities: rocksdb::DB,
}

#[pymethods]
impl GramsDB {
    pub fn new(datadir: Path) -> Result<Self> {
        Self {
            datadir,
            entities: open_entity_db(&datadir.join("wdentities.db"))?,
        }
    }
}

/// A context object that contains the data needed for the algorithm to run for each table.
pub struct AlgoContext {
    entities: HashMap<String, Entity>,
}

fn open_entity_db(dbpath: &str) -> Result<rocksdb::DB> {
    let options = Options::default();
    options.create_if_missing(false);
    options.set_compression_type("zstd");
    options.set_compression_options(
        -14,       // window_bits
        6,         // level
        0,         // strategy
        16 * 1024, // max_dict_bytes
    );
    options.set_zstd_max_train_bytes(100 * 16 * 1024);

    rocksdb::DB::open_for_read_only(options, dbpath, false).map_err(into_pyerr)?
}
