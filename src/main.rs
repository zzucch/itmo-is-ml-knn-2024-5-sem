use std::error::Error;

use kiddo::{Manhattan, SquaredEuclidean};
use knn::{
    distance_metric::Chebyshev,
    kernel::*,
    knn::{Data, Knn, WindowType, DIMENSIONS},
    parse::breast_cancer::{parse, CsvEntry, Diagnosis},
};

fn csv_entries_to_data(entries: Vec<CsvEntry>) -> Vec<Data> {
    entries
        .into_iter()
        .map(|entry| Data {
            x: entry.values.try_into().unwrap(),
            y: entry.diagnosis,
        })
        .collect()
}

fn split_data(data: Vec<Data>, train_ratio: f64) -> (Vec<Data>, Vec<Data>) {
    let train_size = (data.len() as f64 * train_ratio) as usize;
    let (train_data, test_data) = data.split_at(train_size);

    (train_data.to_vec(), test_data.to_vec())
}

fn calculate_accuracy<M>(knn: &Knn<M>, test_data: &[Data]) -> f64
where
    M: kiddo::distance_metric::DistanceMetric<f64, DIMENSIONS>,
{
    let mut predictions = Vec::new();
    let actuals: Vec<Diagnosis> = test_data.iter().map(|test_point| test_point.y).collect();

    for test_point in test_data {
        match knn.predict(&test_point.x) {
            Ok(prediction) => predictions.push(Some(prediction)),
            Err(_) => predictions.push(None),
        }
    }

    let correct_predictions = predictions
        .iter()
        .zip(actuals.iter())
        .filter(|&(prediction, actual)| match prediction {
            Some(prediction) => prediction == actual,
            _ => false,
        })
        .count();

    let total_predictions = predictions.len();

    if total_predictions > 0 {
        (correct_predictions as f64 / total_predictions as f64) * 100.0
    } else {
        0.0
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    const FILE_PATH: &str = "data/breast-cancer.csv";

    let entries = parse(FILE_PATH)?;
    assert!(!entries.is_empty());
    assert_eq!(entries.first().unwrap().values.len(), DIMENSIONS);

    let data = csv_entries_to_data(entries);

    let (train_data, test_data) = split_data(data, 0.85);
    println!("train_data.len() : {}", train_data.len());
    println!("test_data.len() : {}", test_data.len());

    let kernel_functions: [(&str, fn(f64) -> f64); 4] = [
        ("uniform", uniform),
        ("triangular", triangular),
        ("epanechnikov", epanechnikov),
        ("gaussian", gaussian),
    ];
    let window_types = [
        ("fixed", WindowType::Fixed),
        ("unfixed", WindowType::Unfixed),
    ];

    let mut max_accuracy = 0.0;
    let mut count = 0;

    for radius in 1..10 {
        for neighbour_amount in 1..50 {
            for (window_name, window_type) in &window_types {
                for (kernel_name, kernel_function) in &kernel_functions {
                    let mut knn_manhattan: Knn<Manhattan> = Knn::new(
                        neighbour_amount,
                        radius as f64,
                        window_type,
                        *kernel_function,
                        train_data.len(),
                    );
                    knn_manhattan.fit(train_data.clone(), None);
                    let accuracy = calculate_accuracy(&knn_manhattan, &test_data);
                    count += 1;

                    if accuracy > max_accuracy {
                        max_accuracy = accuracy;
                        println!(
                            "{count}. kernel: {kernel_name}, window: {window_name}, neighbours: {neighbour_amount}, radius: {radius}, metric: Manhattan\taccuracy: {:.3}%",
                            accuracy
                        );
                    }

                    let mut knn_squared_euclidean: Knn<SquaredEuclidean> = Knn::new(
                        neighbour_amount,
                        radius as f64,
                        window_type,
                        *kernel_function,
                        train_data.len(),
                    );
                    knn_squared_euclidean.fit(train_data.clone(), None);
                    let accuracy = calculate_accuracy(&knn_squared_euclidean, &test_data);
                    count += 1;

                    if accuracy > max_accuracy {
                        max_accuracy = accuracy;
                        println!(
                            "{count}. kernel: {kernel_name}, window: {window_name}, neighbours: {neighbour_amount}, radius: {radius}, metric: SquaredEuclidean\taccuracy: {:.3}%",
                            accuracy
                        );
                    }

                    let mut knn_chebyshev: Knn<Chebyshev> = Knn::new(
                        neighbour_amount,
                        radius as f64,
                        window_type,
                        *kernel_function,
                        train_data.len(),
                    );
                    knn_chebyshev.fit(train_data.clone(), None);
                    let accuracy = calculate_accuracy(&knn_chebyshev, &test_data);
                    count += 1;

                    if accuracy > max_accuracy {
                        max_accuracy = accuracy;
                        println!(
                            "{count}. kernel: {kernel_name}, window: {window_name}, neighbours: {neighbour_amount}, radius: {radius}, metric: Chebyshev\taccuracy: {:.3}%",
                            accuracy
                        );
                    }
                }
            }
        }
    }

    Ok(())
}
