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
}

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
}
