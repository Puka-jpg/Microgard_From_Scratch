# Building Micrograd

This repository contains my implementation of Micrograd, a small autograd engine inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). The project was built to deepen my understanding of automatic differentiation and neural networks.

## Features

- Custom `Value` class with automatic differentiation capabilities
- Implementation of basic operations (+, -, *, /, pow, etc.)
- Tanh and exp activation functions
- Simple neural network architecture with:
  - Neuron class
  - Layer class
  - MLP (Multi-Layer Perceptron) class
- Gradient-based optimization

## Example Usage

```python
# Create a simple neural network
n = MLP(3, [4, 4, 1])

# Training data
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

# Desired outputs
ys = [1.0, -1.0, -1.0, 1.0]

# Training loop
for k in range(100):
    # Forward pass
    ypred = [n(x) for x in xs]  
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # Backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # Gradient descent
    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(k, loss.data)
```

## Dependencies

- NumPy
- Matplotlib
- Graphviz (for visualizing computational graphs)

## Learning Outcomes

The primary goal of this project was to build a deep understanding of:
- Computational graphs
- Backpropagation
- Automatic differentiation
- Basic neural network architecture

This implementation helped me understand the inner workings of popular deep learning frameworks like PyTorch.
