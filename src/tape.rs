use std::cell::RefCell;

struct Node {
    weight: [f64; 2],
    deps: [usize; 2],
}

impl Node {
    pub fn new(weight: [f64; 2], deps: [usize; 2]) -> Self {
        Self { weight, deps }
    }
}

pub(crate) struct Tape {
    nodes: RefCell<Vec<Node>>,
}
