use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

#[derive(Debug)]
struct CsvEntry {
    source: String,
    values: Vec<f64>,
}

fn parse_csv(file_path: &str) -> Result<Vec<CsvEntry>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(BufReader::new(file));

    let mut entries = Vec::new();

    for result in reader.records() {
        let record = result?;

        const SOURCE_FIELD_INDEX: usize = 30;
        let source = record.get(SOURCE_FIELD_INDEX).unwrap().to_string();

        let values: Vec<f64> = record
            .iter()
            .filter_map(|x| x.parse::<f64>().ok())
            .collect();

        entries.push(CsvEntry { source, values });
    }

    Ok(entries)
}

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "data/data.csv";
    let entries = parse_csv(file_path)?;

    println!("{}", entries.len());

    Ok(())
}
