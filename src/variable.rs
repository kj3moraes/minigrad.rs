use crate::grad::Gradient;
use crate::tape::Tape;
use candle_core::{DType, Device, Tensor};

#[derive(Clone)]
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

#[derive(Clone, Debug)]
pub struct Variable<'a> {
    pub(crate) tape: Option<&'a Tape>,
    pub(crate) index: usize,
    pub value: Tensor,
}

/// Constructors
impl<'a> Variable<'a> {
    pub fn from(value: f64) -> Self {
        let tensor = Tensor::from_slice(&[value], (1, 1), &Device::Cpu).unwrap();
        Self {
            tape: None,
            index: 0,
            value: tensor,
        }
    }

    pub fn from_tensor(value: Tensor) -> Self {
        Self {
            tape: None,
            index: 0,
            value: value,
        }
    }

    pub(crate) fn new(t: &'a Tape, index: usize, value: f64) -> Self {
        let tensor = Tensor::from_slice(&[value], (1, 1), &Device::Cpu).unwrap();
        Self {
            tape: Some(t),
            index,
            value: tensor,
        }
    }

    pub(crate) fn new_tensor(t: &'a Tape, index: usize, value: Tensor) -> Self {
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

        let len = self.tape.unwrap().len();
        let mut derivates: Vec<Tensor> = vec![];

        // Set the derivative of the current variable wrt itself as 1
        derivates[self.index] = derivates[self.index].ones_like().unwrap();

        for i in (0..len).rev() {
            let node = &nodes[i];
            let deriv = derivates[i].clone();
            println!("The shape of deriv is {:?}", deriv.shape());
            println!(
                "The shape of lhs weight is {:?}",
                nodes[i].weight[0].t().unwrap().shape()
            );
            println!(
                "The shape of rhs weight is {:?}",
                nodes[i].weight[1].t().unwrap().shape()
            );
            // if y = xw
            // df/dx = x^t df/fy
            let lhs = node.weight[0].t().unwrap().matmul(&deriv).unwrap();
            derivates[node.deps[0]] = (&derivates[node.deps[0]] + lhs).unwrap();

            // df/dw = df/fy w^t
            let rhs = deriv.matmul(&node.weight[1].t().unwrap()).unwrap();
            derivates[node.deps[1]] = (&derivates[node.deps[1]] + rhs).unwrap();
        }

        // Return a gradient object of all the derivates.
        Gradient::from(derivates)
    }

    // pub fn relu(&self) -> Variable<'a> {
    //     let new_value = if self.value < 0.0 { 0.0 } else { self.value };
    //     let tape = self.tape.unwrap();
    //     // In practice, ReLU's derivative is 0 everywhere where value < 0.0 and
    //     Variable::new(
    //         tape,
    //         tape.push_unary(f64::derivative(self.value), self.index),
    //         new_value,
    //     )
    // }

    // pub fn sin(&self) -> Variable {
    //     Variable::new(
    //         self.tape.unwrap(),
    //         self.tape.unwrap().push_unary(self.value.cos(), self.index),
    //         self.value.sin(),
    //     )
    // }

    // pub fn cos(&self) -> Variable {
    //     Variable::new(
    //         self.tape.unwrap(),
    //         self.tape
    //             .unwrap()
    //             .push_unary(-1.0 * self.value.sin(), self.index),
    //         self.value.cos(),
    //     )
    // }

    // pub fn pow(&self, power: f64) -> Variable {
    //     Variable::new(
    //         self.tape.unwrap(),
    //         self.tape
    //             .unwrap()
    //             .push_unary(power * self.value.powf(power - 1.0), self.index),
    //         self.value.powf(power),
    //     )
    // }
}

#[cfg(test)]
mod tests {
    use candle_core::{IndexOp, Tensor};

    use crate::variable::Variable;

    use super::Tape;

    #[test]
    // fn test_x_times_y() {
    //     let t = Tape::new();
    //     let x = t.var(0.5);
    //     let y = t.var(4.2);
    //     let z = 2.0 * y.clone();
    //     let grad = z.grad();

    //     // Check that the calculated value is correct
    //     let z_value = z.value.i((0, 0)).unwrap().to_scalar::<f64>().unwrap();
    //     assert!((z_value - 8.4) <= 1e-15);

    //     let x_grad_value = grad.wrt(&x).i((0, 0)).unwrap().to_scalar::<f64>().unwrap();
    //     let y_grad_value = grad.wrt(&y).i((0, 0)).unwrap().to_scalar::<f64>().unwrap();
    //     // Assert that the gradients calculated are correct as well.
    //     assert!((x_grad_value - 0.0).abs() <= 1e-15);
    //     assert!((y_grad_value - 2.0).abs() <= 1e-15);
    // }
    #[test]
    fn test_x_times_y_2d() {
        const N: usize = 4;
        const M: usize = 3;
        const D: usize = 2;

        let t = Tape::new();
        let x = t
            .var(Tensor::ones((N, M), candle_core::DType::F64, &candle_core::Device::Cpu).unwrap());
        let y = t
            .var(Tensor::ones((M, D), candle_core::DType::F64, &candle_core::Device::Cpu).unwrap());

        let z = x.clone() * y.clone();
        let grad = z.grad();

        // Check that the calculated value is correct
        let z_value = z.value.i((0, 0)).unwrap().to_scalar::<f64>().unwrap();
        assert!((z_value - 8.4) <= 1e-15);

        // let x_grad_value = grad.wrt(&x).i((0, 0)).unwrap().to_scalar::<f64>().unwrap();
        // let y_grad_value = grad.wrt(&y).i((0, 0)).unwrap().to_scalar::<f64>().unwrap();
        // // Assert that the gradients calculated are correct as well.
        // assert!((x_grad_value - 0.0).abs() <= 1e-15);
        // assert!((y_grad_value - 2.0).abs() <= 1e-15);
    }
    // #[test]
    // fn test_x_plus_y() {
    //     let t = Tape::new();
    //     let x = t.var(0.5);
    //     let y = t.var(4.0);
    //     let z = x + y;
    //     let grad = z.grad();

    //     // Check that the calculated value is correct
    //     assert!((z.value - 4.5).abs() <= 1e-15);
    //     // Assert that the gradients calculated are correct as well.
    //     assert!((grad.wrt(&x) - 1.0).abs() <= 1e-15);
    //     assert!((grad.wrt(&y) - 1.0).abs() <= 1e-15);
    // }

    // #[test]
    // fn test_multiple_operations() {
    //     let t = Tape::new();
    //     let x = t.var(0.5);
    //     let y = t.var(4.2);
    //     let z = x * y - x.sin();
    //     let grad = z.grad();

    //     // Check that the calculated value is correct
    //     assert!((z.value - 1.620574461395797).abs() <= 1e-15);
    //     // Assert that the gradients calculated are correct as well.
    //     assert!((grad.wrt(&x) - (y - x.cos()).value).abs() <= 1e-15);
    //     assert!((grad.wrt(&y) - x.value).abs() <= 1e-15);
    // }

    // #[test]
    // fn test_power() {
    //     let t = Tape::new();
    //     let x = t.var(2.0);
    //     let z = x.pow(3.0);
    //     let grad = z.grad();

    //     // Check that the calculated value is correct
    //     assert!((z.value - 8.0).abs() <= 1e-15);
    //     // Assert that the gradients calculated are correct as well.
    //     assert!((grad.wrt(&x) - (3.0 * (x.pow(2.0))).value).abs() <= 1e-15);
    // }
}
