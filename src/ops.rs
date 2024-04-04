use std::ops::{Add, Mul};

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
