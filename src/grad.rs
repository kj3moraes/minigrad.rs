use crate::variable::Variable;

pub struct Gradient {
    derivatives: Vec<f64>,
}

impl Gradient {
    pub fn from(derivatives: &[f64]) -> Self {
        Self {
            derivatives: derivatives.to_vec(),
        }
    }

    pub fn wrt(&self, var: &Variable) -> f64 {
        self.derivatives[var.index()]
    }
}
