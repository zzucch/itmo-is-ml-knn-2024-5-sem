use kiddo::{distance_metric::DistanceMetric, float::kdtree::Axis};

pub struct Chebyshev {}

impl<A: Axis, const K: usize> DistanceMetric<A, K> for Chebyshev {
    #[inline]
    fn dist(a: &[A; K], b: &[A; K]) -> A {
        a.iter()
            .zip(b.iter())
            .map(|(&a_val, &b_val)| (a_val - b_val).abs())
            .fold(A::zero(), |acc, x| acc.max(x))
    }

    #[inline]
    fn dist1(a: A, b: A) -> A {
        (a - b).abs()
    }
}
