use crate::activation::ReLU;
use crate::grad::Gradient;
use crate::tape::Tape;
use tensor_rs::tensor::Tensor;

#[derive(Clone, PartialEq)]
pub enum Value {
    Scalar(f64),
    Vector(Tensor),
}

impl From<Tensor> for Value {
    fn from(value: Tensor) -> Self {
        Self::Vector(value)
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::Scalar(value)
    }
}

impl Into<f64> for Value {
    fn into(self) -> f64 {
        match self {
            Self::Scalar(val) => val,
            Self::Vector(_) => panic!("Cannot convert a tensor to a float"),
        }
    }
}

impl Into<Tensor> for Value {
    fn into(self) -> Tensor {
        match self {
            Self::Scalar(_) => panic!("Cannot convert a float to a tensor"),
            Self::Vector(tns) => tns,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Variable<'a> {
    pub(crate) tape: Option<&'a Tape>,
    pub(crate) index: usize,
    pub value: f64,
}

/// Constructors
impl<'a> Variable<'a> {
    pub fn from(value: f64) -> Self {
        Self {
            tape: None,
            index: 0,
            value,
        }
    }

    pub(crate) fn new(t: &'a Tape, index: usize, value: f64) -> Self {
        Self {
            tape: Some(t),
            index,
            value,
        }
    }
}

/// Diffrentiation and Operations
impl<'a> Variable<'a> {
    pub fn grad(&self) -> Gradient {
        let nodes = self
            .tape
            .expect("This variable does not support automatic diffrentiation")
            .nodes
            .borrow();
        let len = self
            .tape
            .expect("This variable does not support automatic diffrentation")
            .len();
        let mut derivates = vec![0.0; len];

        // Set the derivative of the current variable wrt itself as 1
        derivates[self.index] = 1.0;

        for i in (0..len).rev() {
            let deriv = derivates[i];
            // Iterate over the two possible branches
            for j in 0..2 {
                derivates[nodes[i].deps[j]] += nodes[i].weight[j] * deriv;
            }
        }

        // Return a gradient object of all the derivates.
        Gradient::from(derivates)
    }

    pub fn relu(&self) -> Variable<'a> {
        let new_value = if self.value < 0.0 { 0.0 } else { self.value };
        let tape = self.tape.unwrap();
        // In practice, ReLU's derivative is 0 everywhere where value < 0.0 and
        Variable::new(
            tape,
            tape.push_unary(f64::derivative(self.value), self.index),
            new_value,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Tape;

    #[test]
    fn test_x_times_y() {
        let t = Tape::new();
        let x = t.var(0.5);
        let y = t.var(4.2);
        let z = 2.0 * x * y;
        let grad = z.grad();

        // Check that the calculated value is correct
        assert!((z.value - 4.2).abs() <= 1e-15);
        // Assert that the gradients calculated are correct as well.
        assert!((grad.wrt(&x) - 2.0 * y.value).abs() <= 1e-15);
        assert!((grad.wrt(&y) - 2.0 * x.value).abs() <= 1e-15);
    }

    #[test]
    fn test_x_plus_y() {
        let t = Tape::new();
        let x = t.var(0.5);
        let y = t.var(4.0);
        let z = x + y;
        let grad = z.grad();

        // Check that the calculated value is correct
        assert!((z.value - 4.5).abs() <= 1e-15);
        // Assert that the gradients calculated are correct as well.
        assert!((grad.wrt(&x) - 1.0).abs() <= 1e-15);
        assert!((grad.wrt(&y) - 1.0).abs() <= 1e-15);
    }
}
