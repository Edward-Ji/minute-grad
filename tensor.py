from typing import Callable

import numpy as np
from numpy.typing import NDArray

from util import unbroadcast


class Tensor:
    """
    A simple tensor class that supports basic operations and automatic differentiation.
    This class is designed to be used for building and training neural networks.
    It supports operations like addition, subtraction, multiplication, and matrix
    multiplication. It also supports backpropagation through the computational graph
    to compute gradients for optimization.
    """
    def __init__(self, val: NDArray, requires_grad: bool = False):
        self.val: NDArray = np.array(val)

        self.requires_grad: bool = requires_grad
        self.grad: NDArray = np.zeros_like(self.val)

        self._backward: Callable[[], None] = lambda: None
        self._prev: set[Tensor] = set()

    def __len__(self):
        """
        Returns the length of the first dimension of the underlying numpy array.
        """
        return len(self.val)

    @property
    def shape(self):
        """
        Returns the shape of the underlying numpy array.
        """
        return self.val.shape

    def __str__(self):
        """
        Returns a string representation of the tensor in the format:
            Tensor(np.array([...]))
        """
        return f"Tensor({self.val})"

    def __matmul__(self, other):
        """
        Matrix multiplication of two tensors.
        """
        out = Tensor(self.val @ other.val, self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad @ other.val.T, self.shape)
            if other.requires_grad:
                other.grad = unbroadcast(self.val.T @ out.grad, other.shape)

        out._backward = _backward
        out._prev = {self, other}

        return out

    def __add__(self, other):
        """
        Element-wise addition of two tensors.
        """
        out = Tensor(self.val + other.val, self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.shape)

        out._backward = _backward
        out._prev = {self, other}

        return out

    def __neg__(self):
        """
        Negates the tensor element-wise.
        """
        out = Tensor(-self.val, self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad -= unbroadcast(out.grad, self.shape)

        out._backward = _backward
        out._prev = {self}

        return out

    def __sub__(self, other):
        """
        Element-wise subtraction of two tensors.
        """
        return self + -other

    def __mul__(self, other):
        """
        Element-wise multiplication of two tensors.
        """
        out = Tensor(self.val * other.val, self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad * other.val, self.shape)
            if other.requires_grad:
                other.grad += unbroadcast(self.val * out.grad, other.shape)

        out._backward = _backward
        out._prev = {self, other}

        return out

    def __getitem__(self, item):
        out = Tensor(self.val[item], self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad[item] += out.grad

        out._backward = _backward
        out._prev = {self}

        return out

    def abs(self):
        """
        Computes the absolute value of the tensor element-wise.
        """
        out = Tensor(np.abs(self.val), self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(np.sign(self.val) * out.grad, self.shape)

        out._backward = _backward
        out._prev = {self}

        return out

    def __pow__(self, power):
        """
        computes the power of the tensor element-wise.
        """
        out = Tensor(self.val**power, self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(
                    power * self.val ** (power - 1) * out.grad, self.shape
                )

        out._backward = _backward
        out._prev = {self}

        return out

    def sum(self):
        """
        Computes the sum of all elements in the tensor.
        """
        out = Tensor(np.sum(self.val), self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad

        out._backward = _backward
        out._prev = {self}

        return out

    def backward(self):
        """
        Backpropagation through the computational graph to compute gradients.
        This method traverses the graph in reverse topological sorting order, applying
        the chain rule to compute gradients for each tensor.
        """
        self.grad = np.ones_like(self.val)

        topo = []

        visited = set()

        def traverse(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for prev in tensor._prev:
                    traverse(prev)
                topo.append(tensor)

        traverse(self)

        for tensor in reversed(topo):
            tensor._backward()
