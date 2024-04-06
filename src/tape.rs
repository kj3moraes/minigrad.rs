use crate::variable::Variable;
use candle_core::{Device, Tensor};
use std::{cell::RefCell, fmt::Debug};

pub fn convert_to_tensor(value: f64) -> Tensor {
    Tensor::from_slice(&[value], (1, 1), &Device::Cpu).unwrap()
}

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub(crate) weight: [Tensor; 2],
    pub(crate) deps: [usize; 2],
}

impl Node {
    pub fn from(weight: [Tensor; 2], deps: [usize; 2]) -> Self {
        Self { weight, deps }
    }
}

#[derive(Debug)]
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

    pub fn var(&self, value: Tensor) -> Variable {
        Variable::new_tensor(&self, self.push_leaf(), value)
    }

    pub fn push_leaf(&self) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node::from(
            [convert_to_tensor(0.0), convert_to_tensor(0.0)],
            [len, len],
        ));
        len
    }

    pub fn push_unary(&self, weight: Tensor, pos: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node::from([weight, convert_to_tensor(0.0)], [pos, len]));
        len
    }

    pub fn push_binary(&self, weight0: Tensor, pos0: usize, weight1: Tensor, pos1: usize) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node::from([weight0, weight1], [pos0, pos1]));
        len
    }
}
