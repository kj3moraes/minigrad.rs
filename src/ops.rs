use candle_core::{DType, Device, Shape, Tensor};

use crate::tape::convert_to_tensor;
use crate::variable::Variable;
use std::ops::{Add, Mul, Sub};

impl<'a> Add for Variable<'a> {
    type Output = Variable<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        let new_value = (&self.value + &rhs.value).unwrap();
        let n = rhs.value.shape().dims()[1];
        println!("THe second dimension sizie is {}", n);
        let position = self.tape.unwrap().push_binary(
            Tensor::from_slice(&[1.0], (1, 1), &candle_core::Device::Cpu).unwrap(),
            self.index,
            Tensor::eye(n, DType::F64, &Device::Cpu).unwrap(),
            rhs.index,
            new_value.shape().clone(),
        );
        Variable::new_tensor(self.tape.unwrap(), position, new_value)
    }
}

// this is for self - rhs
impl<'a> Sub for Variable<'a> {
    type Output = Variable<'a>;
    fn sub(self, rhs: Self) -> Self::Output {
        let new_value = (&self.value - &rhs.value).unwrap();
        let position = self.tape.unwrap().push_binary(
            rhs.value.t().unwrap().ones_like().unwrap(),
            self.index,
            (-1.0 * self.value.t().unwrap().ones_like().unwrap()).unwrap(),
            rhs.index,
            new_value.shape().clone(),
        );
        Variable::new_tensor(self.tape.unwrap(), position, new_value)
    }
}

impl<'a> Mul for Variable<'a> {
    type Output = Variable<'a>;
    fn mul(self, rhs: Self) -> Self::Output {
        let new_value = self.value.matmul(&rhs.value).unwrap();
        let position = self.tape.unwrap().push_binary(
            rhs.value.t().unwrap().clone(),
            self.index,
            self.value.t().unwrap().clone(),
            rhs.index,
            new_value.shape().clone(),
        );
        Variable::new_tensor(self.tape.unwrap(), position, new_value)
    }
}

impl<'a> Mul<Variable<'a>> for f64 {
    type Output = Variable<'a>;
    fn mul(self, rhs: Variable<'a>) -> Self::Output {
        let new_value = (self * rhs.value).unwrap();
        let position = rhs.tape.unwrap().push_unary(
            convert_to_tensor(self),
            rhs.index,
            new_value.shape().clone(),
        );
        Variable::new_tensor(rhs.tape.unwrap(), position, new_value)
    }
}

// impl<'a> Mul<Variable<'a>> for i32 {
//     type Output = Variable<'a>;
//     fn mul(self, rhs: Variable<'a>) -> Self::Output {
//         let position = rhs.tape.unwrap().push_unary(self as f64, rhs.index);
//         let new_value = self as f64 * rhs.value;
//         Variable::new(rhs.tape.unwrap(), position, new_value)
//     }
// }
