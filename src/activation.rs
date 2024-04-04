pub trait ReLU {
    fn derivative(val: Self) -> Self;
}

impl ReLU for f64 {
    fn derivative(val: Self) -> Self {
        if val > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}
