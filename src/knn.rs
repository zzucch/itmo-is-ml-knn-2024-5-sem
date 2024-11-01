use std::{collections::HashMap, error::Error, marker::PhantomData};

use kiddo::{distance_metric::DistanceMetric, float::kdtree::KdTree};

use crate::parse::breast_cancer::Diagnosis;

pub const DIMENSIONS: usize = 30;

const BUCKET_SIZE: usize = 32;

#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    Fixed,
    Unfixed,
}

#[derive(Clone, Copy)]
pub struct Data {
    pub features: [f64; DIMENSIONS],
    pub label: Diagnosis,
}

#[derive(Clone)]
pub struct Knn<M: DistanceMetric<f64, DIMENSIONS>> {
    k: usize,
    radius: f64,
    kernel: fn(f64) -> f64,
    window: WindowType,
    kd_tree: KdTree<f64, usize, DIMENSIONS, BUCKET_SIZE, u32>,
    data: Vec<Data>,
    weights: Vec<f64>,
    _marker: PhantomData<M>,
}

impl<M: DistanceMetric<f64, DIMENSIONS>> Knn<M> {
    pub fn new(
        k: usize,
        radius: f64,
        window: &WindowType,
        kernel: fn(f64) -> f64,
        capacity: usize,
    ) -> Self {
        Knn {
            k,
            radius,
            kernel,
            window: *window,
            kd_tree: KdTree::with_capacity(capacity),
            data: Vec::new(),
            weights: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn fit(&mut self, data: Vec<Data>, weights: Option<Vec<f64>>) {
        self.data = data;
        self.weights = weights.unwrap_or_else(|| vec![1.0; self.data.len()]);

        for (idx, data_point) in self.data.iter().enumerate() {
            self.kd_tree.add(&data_point.features, idx);
        }
    }

    pub fn predict(&self, x: &[f64; DIMENSIONS]) -> Result<Diagnosis, Box<dyn Error>> {
        let (kernel_distances, targets, weights) = self.predict_with_neighbors(x);

        if targets.is_empty() || weights.is_empty() {
            return Err("no neighbors found for prediction".into());
        }

        let predicted_class = Self::predict_class(&kernel_distances, &targets, &weights);
        Ok(predicted_class)
    }

    fn predict_class(
        kernel_distances: &[f64],
        targets: &[Diagnosis],
        weights: &[f64],
    ) -> Diagnosis {
        let mut class_scores: HashMap<Diagnosis, f64> = HashMap::new();

        for (i, target) in targets.iter().enumerate() {
            let weighted_score = kernel_distances[i] * weights[i];
            *class_scores.entry(*target).or_insert(0.0) += weighted_score;
        }

        class_scores
            .into_iter()
            .max_by(|first, second| first.1.partial_cmp(&second.1).unwrap())
            .map(|(class, _)| class)
            .unwrap()
    }

    fn predict_with_neighbors(
        &self,
        x: &[f64; DIMENSIONS],
    ) -> (Vec<f64>, Vec<Diagnosis>, Vec<f64>) {
        let (distances, indices): (Vec<f64>, Vec<usize>) = match self.window {
            WindowType::Fixed => self.kd_tree.within::<M>(x, self.radius.powi(2)),
            WindowType::Unfixed => self.kd_tree.nearest_n::<M>(x, self.k),
        }
        .into_iter()
        .map(|neighbour| (neighbour.distance.sqrt(), neighbour.item))
        .unzip();

        let mut adjusted_distances = distances.clone();
        let mut weights = Vec::new();
        let mut targets = Vec::new();

        match self.window {
            WindowType::Fixed => {
                adjusted_distances
                    .iter_mut()
                    .for_each(|dist| *dist /= self.radius);
            }
            WindowType::Unfixed => {
                let adjusted_distance = *adjusted_distances.last().unwrap();
                adjusted_distances
                    .iter_mut()
                    .for_each(|distance| *distance /= adjusted_distance);
            }
        }

        for &index in &indices {
            targets.push(self.data[index].label);
            weights.push(self.weights[index]);
        }

        let kernel_distances: Vec<f64> = adjusted_distances
            .iter()
            .map(|&dist| (self.kernel)(dist))
            .collect();

        (kernel_distances, targets, weights)
    }
}
