use std::cell::RefCell;

use crate::variable::Variable;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Node {
    pub(crate) weight: [f64; 2],
    pub(crate) deps: [usize; 2],
}

impl Node {
    pub fn from(weight: [f64; 2], deps: [usize; 2]) -> Self {
        Self { weight, deps }
    }
}

pub(crate) struct Tape {
    pub(crate) nodes: RefCell<Vec<Node>>,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            nodes: RefCell::new(Vec::new()),
        }
    }
}

impl Tape {
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    pub fn var(&self, value: f64) -> Variable {
        Variable::new(&self, self.push_leaf(), value)
    }

    pub fn push_leaf(&self) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node::from([0.0, 0.0], [len, len]));
        len
    }

    pub fn push_unary(&self, weight: f64, pos: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node::from([weight, 0.0], [pos, 0]));
        len
    }

    pub fn push_binary(&self, weight0: f64, pos0: usize, weight1: f64, pos1: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node::from([weight0, weight1], [pos0, pos1]));
        len
    }
}
