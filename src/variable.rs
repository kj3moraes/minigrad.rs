use crate::activation::ReLU;
use crate::grad::Gradient;
use crate::tape::Tape;

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

    pub fn sin(&self) -> Variable {
        Variable::new(
            self.tape.unwrap(),
            self.tape.unwrap().push_unary(self.value.cos(), self.index),
            self.value.sin(),
        )
    }

    pub fn cos(&self) -> Variable {
        Variable::new(
            self.tape.unwrap(),
            self.tape
                .unwrap()
                .push_unary(-1.0 * self.value.sin(), self.index),
            self.value.cos(),
        )
    }

    pub fn pow(&self, power: f64) -> Variable {
        Variable::new(
            self.tape.unwrap(),
            self.tape
                .unwrap()
                .push_unary(power * self.value.powf(power - 1.0), self.index),
            self.value.powf(power),
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

    #[test]
    fn test_multiple_operations() {
        let t = Tape::new();
        let x = t.var(0.5);
        let y = t.var(4.2);
        let z = x * y - x.sin();
        let grad = z.grad();

        // Check that the calculated value is correct
        assert!((z.value - 1.620574461395797).abs() <= 1e-15);
        // Assert that the gradients calculated are correct as well.
        assert!((grad.wrt(&x) - (y - x.cos()).value).abs() <= 1e-15);
        assert!((grad.wrt(&y) - x.value).abs() <= 1e-15);
    }

    #[test]
    fn test_power() {
        let t = Tape::new();
        let x = t.var(2.0);
        let z = x.pow(3.0);
        let grad = z.grad();

        // Check that the calculated value is correct
        assert!((z.value - 8.0).abs() <= 1e-15);
        // Assert that the gradients calculated are correct as well.
        assert!((grad.wrt(&x) - (3.0 * (x.pow(2.0))).value).abs() <= 1e-15);
    }
}
