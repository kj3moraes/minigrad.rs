use crate::grad::Gradient;
use crate::tape::Tape;

pub struct Variable<'a> {
    tape: Option<&'a Tape>,
    index: usize,
    value: f64,
}

impl<'a> Variable<'a> {
    pub fn from(value: f64) -> Self {
        Self {
            tape: None,
            index: 0,
            value: value,
        }
    }

    pub fn item(&self) -> f64 {
        self.value
    }

    pub fn index(&self) -> usize {
        self.index
    }
}
