use kiddo::{Manhattan, SquaredEuclidean};
use knn::{
    distance_metric::Chebyshev,
    kernel::{epanechnikov, gaussian, triangular, uniform},
    knn::{Data, Knn, WindowType, DIMENSIONS},
    lowess::lowess,
    parse::breast_cancer::{opposite_diagnosis, parse, CsvEntry, Diagnosis},
};
use plotters::{
    chart::ChartBuilder,
    prelude::{BitMapBackend, IntoDrawingArea, PathElement},
    series::LineSeries,
    style::{IntoFont, BLACK, BLUE, RED, WHITE},
};
use std::error::Error;

fn csv_entries_to_data(entries: Vec<CsvEntry>) -> Vec<Data> {
    entries
        .into_iter()
        .map(|entry| Data {
            features: entry.values.try_into().unwrap(),
            label: entry.diagnosis,
        })
        .collect()
}

fn split_data(data: &[Data], train_ratio: f64) -> (Vec<Data>, Vec<Data>) {
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    let train_size = (data.len() as f64 * train_ratio) as usize;
    let (train_data, test_data) = data.split_at(train_size);

    (train_data.to_vec(), test_data.to_vec())
}

fn calculate_accuracy<M>(knn: &Knn<M>, test_data: &[Data]) -> f64
where
    M: kiddo::distance_metric::DistanceMetric<f64, DIMENSIONS>,
{
    let mut predictions = Vec::new();
    let actuals: Vec<Diagnosis> = test_data
        .iter()
        .map(|test_point| test_point.label)
        .collect();

    for test_point in test_data {
        match knn.predict(&test_point.features) {
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

#[allow(clippy::too_many_arguments)]
fn update_max_accuracy_and_print(
    accuracy: f64,
    max_accuracy: &mut f64,
    count: &mut usize,
    best_hyperparameters: &mut Hyperparameters,
    kernel_name: &str,
    kernel_function: fn(f64) -> f64,
    window_name: &str,
    window_type: WindowType,
    neighbour_amount: usize,
    radius: usize,
    metric: &str,
) {
    *count += 1;

    if accuracy > *max_accuracy {
        *max_accuracy = accuracy;

        best_hyperparameters.window = window_type;
        best_hyperparameters.k = neighbour_amount;
        best_hyperparameters.radius = radius as f64;
        best_hyperparameters.kernel = kernel_function;
        best_hyperparameters.metric = metric.to_string();

        println!(
            "{count}. kernel: {kernel_name}, window: {window_name}, neighbours: {neighbour_amount}, radius: {radius}, metric: {metric}\taccuracy: {accuracy:.3}%",
        );
    }
}

#[derive(Debug)]
struct Hyperparameters {
    k: usize,
    radius: f64,
    window: WindowType,
    kernel: fn(f64) -> f64,
    metric: String,
}

impl Hyperparameters {
    fn new() -> Self {
        Self {
            k: 0,
            radius: 0.0,
            window: WindowType::Fixed,
            kernel: uniform,
            metric: String::new(),
        }
    }
}

fn calculate_f1_score(data: &[Data], predictions: &[Diagnosis]) -> f64 {
    let mut true_positive_count = 0;
    let mut false_positive_count = 0;
    let mut false_negative_count = 0;

    for (actual, predicted) in data.iter().zip(predictions.iter()) {
        if actual.label == *predicted {
            true_positive_count += 1;
        } else {
            match predicted {
                Diagnosis::Malignant => {
                    false_positive_count += 1;
                }
                Diagnosis::Benign => {
                    false_negative_count += 1;
                }
            }
        }
    }

    let precision = if true_positive_count + false_positive_count > 0 {
        true_positive_count as f64 / (true_positive_count + false_positive_count) as f64
    } else {
        0.0
    };
    let recall = if true_positive_count + false_negative_count > 0 {
        true_positive_count as f64 / (true_positive_count + false_negative_count) as f64
    } else {
        0.0
    };

    if precision + recall > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else {
        0.0
    }
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn Error>> {
    const DATA_FILEPATH: &str = "data/breast-cancer.csv";
    const PLOT_FILENAME: &str = "plot.png";

    let entries = parse(DATA_FILEPATH)?;
    assert!(!entries.is_empty());
    assert_eq!(entries.first().unwrap().values.len(), DIMENSIONS);

    let data = csv_entries_to_data(entries);

    const TRAIN_RATIO: f64 = 0.6;
    const VALIDATION_RATIO: f64 = 0.6; // of data that is not train

    let (train_data, test_data) = split_data(&data, TRAIN_RATIO);
    let (test_data, validation_data) = split_data(&test_data, VALIDATION_RATIO);
    println!("train_data.len() : {}", train_data.len());
    println!("test_data.len() : {}", test_data.len());
    println!("validation_data.len() : {}", validation_data.len());

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
    let mut best_hyperparameters = Hyperparameters::new();

    for radius in 1..15 {
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
                    let accuracy = calculate_accuracy(&knn_manhattan, &validation_data);

                    update_max_accuracy_and_print(
                        accuracy,
                        &mut max_accuracy,
                        &mut count,
                        &mut best_hyperparameters,
                        kernel_name,
                        *kernel_function,
                        window_name,
                        *window_type,
                        neighbour_amount,
                        radius,
                        "manhattan",
                    );

                    let mut knn_squared_euclidean: Knn<SquaredEuclidean> = Knn::new(
                        neighbour_amount,
                        radius as f64,
                        window_type,
                        *kernel_function,
                        train_data.len(),
                    );
                    knn_squared_euclidean.fit(train_data.clone(), None);
                    let accuracy = calculate_accuracy(&knn_squared_euclidean, &validation_data);

                    update_max_accuracy_and_print(
                        accuracy,
                        &mut max_accuracy,
                        &mut count,
                        &mut best_hyperparameters,
                        kernel_name,
                        *kernel_function,
                        window_name,
                        *window_type,
                        neighbour_amount,
                        radius,
                        "squared euclidean",
                    );

                    let mut knn_chebyshev: Knn<Chebyshev> = Knn::new(
                        neighbour_amount,
                        radius as f64,
                        window_type,
                        *kernel_function,
                        train_data.len(),
                    );
                    knn_chebyshev.fit(train_data.clone(), None);
                    let accuracy = calculate_accuracy(&knn_chebyshev, &validation_data);

                    update_max_accuracy_and_print(
                        accuracy,
                        &mut max_accuracy,
                        &mut count,
                        &mut best_hyperparameters,
                        kernel_name,
                        *kernel_function,
                        window_name,
                        *window_type,
                        neighbour_amount,
                        radius,
                        "chebyshev",
                    );
                }
            }
        }
    }

    println!("best hyperparameters: {best_hyperparameters:?}");

    #[allow(clippy::items_after_statements)]
    const MAX_K: usize = 100;

    let mut f1_train_values = Vec::with_capacity(MAX_K);
    let mut f1_test_values = Vec::with_capacity(MAX_K);
    let mut k_values = Vec::with_capacity(MAX_K);

    for k in 1..MAX_K {
        let (train_predictions, test_predictions) = match best_hyperparameters.metric.as_str() {
            "manhattan" => {
                let mut knn_manhattan: Knn<Manhattan> = Knn::new(
                    k,
                    best_hyperparameters.radius,
                    &best_hyperparameters.window,
                    best_hyperparameters.kernel,
                    train_data.len(),
                );
                knn_manhattan.fit(train_data.clone(), None);

                let train_predictions: Vec<_> = train_data
                    .iter()
                    .map(|data| {
                        knn_manhattan
                            .predict(&data.features)
                            .unwrap_or(opposite_diagnosis(data.label))
                    })
                    .collect();

                let test_predictions: Vec<_> = test_data
                    .iter()
                    .map(|data| {
                        knn_manhattan
                            .predict(&data.features)
                            .unwrap_or(opposite_diagnosis(data.label))
                    })
                    .collect();

                (train_predictions, test_predictions)
            }
            "squared euclidean" => {
                let mut knn_squared_euclidean: Knn<SquaredEuclidean> = Knn::new(
                    k,
                    best_hyperparameters.radius,
                    &best_hyperparameters.window,
                    best_hyperparameters.kernel,
                    train_data.len(),
                );
                knn_squared_euclidean.fit(train_data.clone(), None);

                let train_predictions: Vec<_> = train_data
                    .iter()
                    .map(|data| {
                        knn_squared_euclidean
                            .predict(&data.features)
                            .unwrap_or(opposite_diagnosis(data.label))
                    })
                    .collect();

                let test_predictions: Vec<_> = test_data
                    .iter()
                    .map(|data| {
                        knn_squared_euclidean
                            .predict(&data.features)
                            .unwrap_or(opposite_diagnosis(data.label))
                    })
                    .collect();

                (train_predictions, test_predictions)
            }
            "chebyshev" => {
                let mut knn_chebyshev: Knn<Chebyshev> = Knn::new(
                    k,
                    best_hyperparameters.radius,
                    &best_hyperparameters.window,
                    best_hyperparameters.kernel,
                    train_data.len(),
                );
                knn_chebyshev.fit(train_data.clone(), None);

                let train_predictions: Vec<_> = train_data
                    .iter()
                    .map(|data| {
                        knn_chebyshev
                            .predict(&data.features)
                            .unwrap_or(opposite_diagnosis(data.label))
                    })
                    .collect();

                let test_predictions: Vec<_> = test_data
                    .iter()
                    .map(|data| {
                        knn_chebyshev
                            .predict(&data.features)
                            .unwrap_or(opposite_diagnosis(data.label))
                    })
                    .collect();

                (train_predictions, test_predictions)
            }
            _ => panic!("unexpected distance metric"),
        };

        let train_f1 = calculate_f1_score(&train_data, &train_predictions);
        let test_f1 = calculate_f1_score(&test_data, &test_predictions);

        f1_train_values.push(train_f1);
        f1_test_values.push(test_f1);
        k_values.push(k);
    }

    let root = BitMapBackend::new(PLOT_FILENAME, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("F1-score for k values", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(1..100, 0.0..1.0)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            k_values
                .iter()
                .copied()
                .map(|k_val| i32::try_from(k_val).unwrap())
                .zip(f1_train_values.iter().copied())
                .collect::<Vec<_>>(),
            RED,
        ))?
        .label("Train F1-score")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], RED));

    chart
        .draw_series(LineSeries::new(
            k_values
                .iter()
                .copied()
                .map(|k_val| i32::try_from(k_val).unwrap())
                .zip(f1_test_values.iter().copied())
                .collect::<Vec<_>>(),
            BLUE,
        ))?
        .label("Test F1-score")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], BLUE));

    chart.configure_series_labels().border_style(BLACK).draw()?;
    root.present()?;

    println!("plot saved to {PLOT_FILENAME}");

    // TODO: in case of dataset change add other distance metrics
    // for best_hyperparameters.metric
    // the amount of potential new code seems not justified for now
    let mut knn_manhattan: Knn<Manhattan> = Knn::new(
        best_hyperparameters.k,
        best_hyperparameters.radius,
        &best_hyperparameters.window,
        best_hyperparameters.kernel,
        train_data.len(),
    );

    let weights = lowess::<Manhattan>(
        best_hyperparameters.k,
        best_hyperparameters.radius,
        best_hyperparameters.window,
        best_hyperparameters.kernel,
        &train_data,
    );

    knn_manhattan.fit(train_data.clone(), None);

    let train_predictions: Vec<_> = train_data
        .iter()
        .map(|data| {
            knn_manhattan
                .predict(&data.features)
                .unwrap_or(opposite_diagnosis(data.label))
        })
        .collect();
    let test_predictions: Vec<_> = test_data
        .iter()
        .map(|data| {
            knn_manhattan
                .predict(&data.features)
                .unwrap_or(opposite_diagnosis(data.label))
        })
        .collect();

    let unweighted_accuracy = calculate_accuracy(&knn_manhattan, &test_data);
    let unweighted_train_f1 = calculate_f1_score(&train_data, &train_predictions);
    let unweighted_test_f1 = calculate_f1_score(&test_data, &test_predictions);

    println!("unweighted:");
    println!("accuracy: {unweighted_accuracy}, train f1 score: {unweighted_train_f1}, test f1 score: {unweighted_test_f1}");

    knn_manhattan.fit(train_data.clone(), Some(weights));

    let train_predictions: Vec<_> = train_data
        .iter()
        .map(|data| {
            knn_manhattan
                .predict(&data.features)
                .unwrap_or(opposite_diagnosis(data.label))
        })
        .collect();
    let test_predictions: Vec<_> = test_data
        .iter()
        .map(|data| {
            knn_manhattan
                .predict(&data.features)
                .unwrap_or(opposite_diagnosis(data.label))
        })
        .collect();

    let weighted_accuracy = calculate_accuracy(&knn_manhattan, &test_data);
    let weighted_train_f1 = calculate_f1_score(&train_data, &train_predictions);
    let weighted_test_f1 = calculate_f1_score(&test_data, &test_predictions);

    println!("weighted:");
    println!("accuracy: {weighted_accuracy}, train f1 score: {weighted_train_f1}, test f1 score: {weighted_test_f1}");

    Ok(())
}
