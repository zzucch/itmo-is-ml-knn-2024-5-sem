use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

#[derive(Debug)]
pub struct CsvEntry {
    pub source: Source,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Original,
    Manga,
    LightNovel,
    WebNovel,
    Novel,
    Anime,
    VisualNovel,
    VideoGame,
    Doujinshi,
    Comic,
    LiveAction,
    Game,
    MultimediaProject,
    Other,
}

pub fn to_source(source: &str) -> Result<Source, &str> {
    match source {
        "Original" => Ok(Source::Original),
        "Manga" => Ok(Source::Manga),
        "Light Novel" => Ok(Source::LightNovel),
        "Web Novel" => Ok(Source::WebNovel),
        "Novel" => Ok(Source::Novel),
        "Anime" => Ok(Source::Anime),
        "Visual Novel" => Ok(Source::VisualNovel),
        "Video Game" => Ok(Source::VideoGame),
        "Doujinshi" => Ok(Source::Doujinshi),
        "Comic" => Ok(Source::Comic),
        "Live Action" => Ok(Source::LiveAction),
        "Game" => Ok(Source::Game),
        "Multimedia Project" => Ok(Source::MultimediaProject),
        "Other" => Ok(Source::Other),
        "?" => Err("no source"),
        _ => panic!("unknown source: {source}"),
    }
}

pub fn parse(file_path: &str) -> Result<Vec<CsvEntry>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(BufReader::new(file));

    let mut entries = Vec::new();

    for result in reader.records() {
        const SOURCE_FIELD_INDEX: usize = 30;
        const FIRST_COMPANY_INDEX: usize = 37;
        const LAST_COMPANY_INDEX: usize = 970;

        let record = result?;
        let source = record.get(SOURCE_FIELD_INDEX).unwrap().to_string();

        let values: Vec<f64> = record
            .iter()
            .enumerate()
            .filter_map(|(index, value)| {
                if index <= SOURCE_FIELD_INDEX
                    || (FIRST_COMPANY_INDEX..=LAST_COMPANY_INDEX).contains(&index)
                {
                    None
                } else {
                    value.parse::<f64>().ok()
                }
            })
            .collect();

        if let Ok(source) = to_source(&source) {
            entries.push(CsvEntry { source, values });
        }
    }

    Ok(entries)
}
