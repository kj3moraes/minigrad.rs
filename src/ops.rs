use crate::tape::convert_to_tensor;
use crate::variable::Variable;
use std::ops::{Add, Mul, Sub};

impl<'a> Add for Variable<'a> {
    type Output = Variable<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        let position = self.tape.unwrap().push_binary(
            convert_to_tensor(1.0),
            self.index,
            convert_to_tensor(1.0),
            rhs.index,
        );
        let new_value = (self.value + rhs.value).unwrap();
        Variable::new_tensor(self.tape.unwrap(), position, new_value)
    }
}

// // this is for self - rhs
// impl<'a> Sub for Variable<'a> {
//     type Output = Variable<'a>;
//     fn sub(self, rhs: Self) -> Self::Output {
//         let position = self
//             .tape
//             .unwrap()
//             .push_binary(1.0, self.index, -1.0, rhs.index);
//         let new_value = self.value - rhs.value;
//         Variable::new(self.tape.unwrap(), position, new_value)
//     }
// }

impl<'a> Mul for Variable<'a> {
    type Output = Variable<'a>;
    fn mul(self, rhs: Self) -> Self::Output {
        let position = self.tape.unwrap().push_binary(
            rhs.value.t().unwrap().clone(),
            self.index,
            self.value.t().unwrap().clone(),
            rhs.index,
        );
        let new_value = self.value.matmul(&rhs.value).unwrap();
        Variable::new_tensor(self.tape.unwrap(), position, new_value)
    }
}

impl<'a> Mul<Variable<'a>> for f64 {
    type Output = Variable<'a>;
    fn mul(self, rhs: Variable<'a>) -> Self::Output {
        let position = rhs
            .tape
            .unwrap()
            .push_unary(convert_to_tensor(self), rhs.index);
        let new_value = (self * rhs.value).unwrap();
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
