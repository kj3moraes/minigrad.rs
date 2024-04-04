# minigrad.rs

An auto-differentiation engine written in Rust. This engine implements reverse-mode automatic diffrentiation using a tape-based tracking mechanism. A lot of the code is heavily inspired by Rufflewind's excellent post [here](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation).

```rust
let t = Tape::new();
let x = t.var(0.5);
let y = t.var(4.2);
let z = (x * y).relu();
let grad = z.grad();

// Check that the calculated value is correct
assert!((z.value - 2.1).abs() <= 1e-15);
// Assert that the gradients calculated are correct as well.
assert!((grad.wrt(&x) - y.value).abs() <= 1e-15);
assert!((grad.wrt(&y) - x.value).abs() <= 1e-15);
```

## References

[1] [Reverse-mode automatic differentiation: a tutorial](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation) by Rufflewind  
[2] [A Gentle Introduction to torch.autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) by PyTorch  
[3] [tiberiusferraira/Autograd-Experiments](https://github.com/tiberiusferreira/Autograd-Experiments/tree/master/src)  
[4] [Mostafa Samir's Blog](https://mostafa-samir.github.io/auto-diff-pt2/)  
