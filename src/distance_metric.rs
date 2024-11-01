use kiddo::{distance_metric::DistanceMetric, float::kdtree::Axis};

pub struct Chebyshev {}

impl<A: Axis, const K: usize> DistanceMetric<A, K> for Chebyshev {
    #[inline]
    fn dist(first: &[A; K], second: &[A; K]) -> A {
        first
            .iter()
            .zip(second.iter())
            .map(|(&a_val, &b_val)| (a_val - b_val).abs())
            .fold(A::zero(), A::max)
    }

    #[inline]
    fn dist1(first: A, second: A) -> A {
        (first - second).abs()
    }
}
