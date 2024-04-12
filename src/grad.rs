use crate::variable::Variable;
use candle_core::Tensor;

#[derive(Debug)]
pub struct Gradient {
    pub(crate) derivatives: Vec<Tensor>,
}

impl Gradient {
    pub fn from(derivatives: Vec<Tensor>) -> Self {
        Self { derivatives }
    }

    pub fn wrt(&self, var: &Variable) -> &Tensor {
        &self.derivatives[var.index]
    }
}
