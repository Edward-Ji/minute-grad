import numpy as np

from layer import Linear
from optimiser import AdamOptimiser, GradientDescentOptimiser
from tensor import Tensor

def linear_demo():
    linear = Linear(2, 1)
    X = Tensor(np.array([[1, 1], [1, 2], [2, 2]], dtype=float))
    w = Tensor(np.array([[2], [1]], dtype=float))
    b = Tensor(np.array(3))
    y = X @ w + b
    print(y)
    pred = linear(X)
    print(f"{pred = !s}")
    error = pred - y
    print(f"{error = !s}")
    abs_error = error.abs()
    print(f"{abs_error = !s}")
    loss = abs_error.sum()
    print(f"{loss = !s}")
    loss.backward()
    for tensor in [*linear.get_all_tensors(), error, abs_error, loss]:
        print(tensor)
        print(tensor.grad)
        print('---')

def gradient_descent_demo():
    # synthetic data
    X = Tensor(np.array([[1, 1], [1, 2], [2, 2]], dtype=float))
    w = Tensor(np.array([[2], [1]], dtype=float))
    b = Tensor(np.array(3))
    y = X @ w + b
    print(w)
    print(b)

    # linear model
    linear = Linear(2, 1)
    optimiser = GradientDescentOptimiser(linear.get_all_tensors(),
                                         learning_rate=1e-3)
    print(linear.weight)
    print(linear.bias)
    for _ in range(10_000):
        optimiser.zero_grad()
        pred = linear(X)
        loss = (pred - y).abs().sum()
        loss.backward()
        optimiser.optimise()
    print(linear.weight)
    print(linear.bias)


def adam_demo():
    # synthetic data
    X = Tensor(np.array([[1, 1], [1, 2], [2, 2]], dtype=float))
    w = Tensor(np.array([[2], [1]], dtype=float))
    b = Tensor(np.array(3))
    y = X @ w + b
    print(w)
    print(b)

    # linear model
    linear = Linear(2, 1)
    optimiser = AdamOptimiser(linear.get_all_tensors(), learning_rate=1e-3)
    print(linear.weight)
    print(linear.bias)
    for _ in range(10_000):
        optimiser.zero_grad()
        pred = linear(X)
        loss = (pred - y).abs().sum()
        loss.backward()
        optimiser.optimise()
    print(linear.weight)
    print(linear.bias)

def main():
    gradient_descent_demo()

if __name__ == "__main__":
    main()
