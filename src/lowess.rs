use crate::knn::{Data, Knn, WindowType, DIMENSIONS};

pub fn lowess<M>(
    neighbour_amount: usize,
    radius: f64,
    window_type: WindowType,
    kernel: fn(f64) -> f64,
    train_data: &[Data],
) -> Vec<f64>
where
    M: kiddo::distance_metric::DistanceMetric<f64, DIMENSIONS>,
{
    let mut weights = Vec::with_capacity(train_data.len());

    for (i, data_point) in train_data.iter().enumerate() {
        let mut modified_train_data = train_data.to_vec();
        modified_train_data.remove(i);

        let mut knn_instance: Knn<M> = Knn::new(
            neighbour_amount,
            radius,
            &window_type,
            kernel,
            modified_train_data.len(),
        );
        knn_instance.fit(modified_train_data, None);

        match knn_instance.predict(&data_point.features) {
            Ok(prediction) => {
                let weight = if prediction == data_point.label {
                    kernel(0.0)
                } else {
                    kernel(1.0)
                };
                weights.push(weight);
            }
            Err(_) => weights.push(0.0),
        }
    }
    weights
}
