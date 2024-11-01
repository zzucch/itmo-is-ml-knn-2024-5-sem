pub fn uniform(distance: f64) -> f64 {
    if distance < 1.0 {
        0.5
    } else {
        0.0
    }
}

pub fn triangular(distance: f64) -> f64 {
    if distance.abs() < 1.0 {
        1.0 - distance.abs()
    } else {
        0.0
    }
}

pub fn epanechnikov(distance: f64) -> f64 {
    if distance.abs() < 1.0 {
        (1.0 - distance.powi(2)) * 0.75
    } else {
        0.0
    }
}

pub fn gaussian(distance: f64) -> f64 {
    (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-distance.powi(2) / 2.0).exp()
}
