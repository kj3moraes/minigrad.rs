use std::ops::{Add, Mul, Sub};

use crate::variable::Variable;

impl<'a> Add for Variable<'a> {
    type Output = Variable<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        let position = self
            .tape
            .unwrap()
            .push_binary(1.0, self.index, 1.0, rhs.index);
        let new_value = self.value + rhs.value;
        Variable::new(self.tape.unwrap(), position, new_value)
    }
}

// this is for self - rhs
impl<'a> Sub for Variable<'a> {
    type Output = Variable<'a>;
    fn sub(self, rhs: Self) -> Self::Output {
        let position = self
            .tape
            .unwrap()
            .push_binary(1.0, self.index, -1.0, rhs.index);
        let new_value = self.value - rhs.value;
        Variable::new(self.tape.unwrap(), position, new_value)
    }
}

impl<'a> Mul for Variable<'a> {
    type Output = Variable<'a>;
    fn mul(self, rhs: Self) -> Self::Output {
        let position = self
            .tape
            .unwrap()
            .push_binary(rhs.value, self.index, self.value, rhs.index);
        let new_value = self.value * rhs.value;
        Variable::new(self.tape.unwrap(), position, new_value)
    }
}

impl<'a> Mul<Variable<'a>> for f64 {
    type Output = Variable<'a>;
    fn mul(self, rhs: Variable<'a>) -> Self::Output {
        let position = rhs.tape.unwrap().push_unary(self, rhs.index);
        let new_value = self * rhs.value;
        Variable::new(rhs.tape.unwrap(), position, new_value)
    }
}

impl<'a> Mul<Variable<'a>> for i32 {
    type Output = Variable<'a>;
    fn mul(self, rhs: Variable<'a>) -> Self::Output {
        let position = rhs.tape.unwrap().push_unary(self as f64, rhs.index);
        let new_value = self as f64 * rhs.value;
        Variable::new(rhs.tape.unwrap(), position, new_value)
    }
}
