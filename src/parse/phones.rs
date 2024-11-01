use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

#[derive(Debug)]
pub struct CsvEntry {
    pub os: PhoneOs,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhoneOs {
    Android,
    IOs,
}

pub fn to_os(os: &str) -> PhoneOs {
    match os {
        "Android" => PhoneOs::Android,
        "iOS" => PhoneOs::IOs,
        // dataset only contains android and iphone
        val => panic!("unexpected os {val}"),
    }
}

pub fn normalize(data: &[f64]) -> Vec<f64> {
    let mean = data.iter().copied().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std_dev = variance.sqrt();

    data.iter().map(|&x| (x - mean) / std_dev).collect()
}

pub fn parse(file_path: &str) -> Result<Vec<CsvEntry>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(BufReader::new(file));

    let mut entries = Vec::new();
    let mut values_list = Vec::new();

    for result in reader.records() {
        const OS_FIELD_INDEX: usize = 2;
        const GENDER_FIELD_INDEX: usize = 9;
        const NUMERIC_FIELD_START: usize = 3;
        const NUMERIC_FIELD_END: usize = 8;

        let record = result?;

        let os = record.get(OS_FIELD_INDEX).unwrap().to_string();
        let gender = record.get(GENDER_FIELD_INDEX).unwrap().to_string();

        let mut values: Vec<f64> = record
            .iter()
            .enumerate()
            .filter_map(|(index, value)| {
                if (NUMERIC_FIELD_START..=NUMERIC_FIELD_END).contains(&index) {
                    value.parse::<f64>().ok()
                } else {
                    None
                }
            })
            .collect();

        values_list.push(values.clone());

        let gender_value = match gender.as_str() {
            "Female" => 0.0,
            "Male" => 1.0,
            // dataset contains only male and female
            val => panic!("unexpected gender {val}"),
        };

        values.push(gender_value);

        let phone_os = to_os(&os);
        entries.push(CsvEntry {
            os: phone_os,
            values,
        });
    }

    let normalized_values = normalize(&values_list.concat());

    let value_length = entries.first().map_or(0, |entry| entry.values.len());

    for (entry, new_values) in entries
        .iter_mut()
        .zip(normalized_values.chunks(value_length))
    {
        entry.values = new_values.to_vec();
    }

    Ok(entries)
}
