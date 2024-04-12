use crate::variable::Variable;
use candle_core::{Device, Shape, Tensor};
use std::{cell::RefCell, fmt::Debug};

pub fn convert_to_tensor(value: f64) -> Tensor {
    Tensor::from_slice(&[value], (1, 1), &Device::Cpu).unwrap()
}

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub(crate) weight: [Tensor; 2],
    pub(crate) deps: [usize; 2],

    pub(crate) is_leaf: bool,
    /// shape of the Variable this node symbolizes in the  
    pub(crate) shape: Shape,
}

impl Node {
    pub fn from(weight: [Tensor; 2], deps: [usize; 2], shape: Shape, is_leaf: bool) -> Self {
        Self {
            weight,
            deps,
            shape,
            is_leaf,
        }
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
        Variable::new_tensor(&self, self.push_leaf(value.shape().clone()), value)
    }

    pub fn push_leaf(&self, shape: Shape) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node::from(
            [convert_to_tensor(0.0), convert_to_tensor(0.0)],
            [len, len],
            shape,
            true,
        ));
        len
    }

    pub fn push_unary(&self, weight: Tensor, pos: usize, shape: Shape) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node::from(
            [weight, convert_to_tensor(0.0)],
            [pos, len],
            shape,
            false,
        ));
        len
    }

    pub fn push_binary(
        &self,
        weight0: Tensor,
        pos0: usize,
        weight1: Tensor,
        pos1: usize,
        shape: Shape,
    ) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let len = nodes.len();
        nodes.push(Node::from([weight0, weight1], [pos0, pos1], shape, false));
        len
    }
}
