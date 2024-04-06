use crate::variable::Variable;

#[derive(Debug)]
pub struct Gradient {
    derivatives: Vec<f64>,
}

impl Gradient {
    pub fn from(derivatives: Vec<f64>) -> Self {
        Self { derivatives }
    }

    pub fn wrt(&self, var: &Variable) -> f64 {
        self.derivatives[var.index]
    }
}
