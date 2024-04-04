use crate::tape::Tape;

pub struct Variable<'a> {
    tape: &'a Tape,
    index: usize,
    value: f64,
}

impl<'a> Variable<'a> {
    pub fn index(&self) -> usize {
        self.index
    }
}
