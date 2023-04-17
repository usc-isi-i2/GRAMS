pub mod data_matching;
use grams::table::LinkedTable;

fn load_table(path: &str) -> Result<LinkedTable, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);

    for result in rdr.records() {
        let record = result?;
        println!("{:?}", record);
    }
}
