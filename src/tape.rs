use std::cell::RefCell;

struct Node {
    weight: [f64; 2],
    deps: [usize; 2],
}

pub(crate) struct Tape {
    nodes: RefCell<Vec<Node>>,
}
