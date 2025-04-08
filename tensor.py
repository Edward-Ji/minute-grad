import numpy as np

from util import unbroadcast


def wrap_numpy(method):
    def wrapper(obj, *args, **kwargs):
        out = Tensor(method(obj.val, *args, **kwargs), obj.requires_grad)
        def _backward():
            if obj.requires_grad:
                raise NotImplementedError(
                    "Backward pass not implemented for this operation."
                )
        out._backward = _backward
        out._prev = {obj}
        return out
    return wrapper


class Tensor:
    def __init__(self, val, requires_grad=False):
        self.val = np.array(val)

        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.val)

        self._backward = lambda: None
        self._prev = set()

    def __len__(self):
        return len(self.val)

    @property
    def shape(self):
        return self.val.shape

    def __str__(self):
        return f"Tensor({self.val})"

    def __matmul__(self, other):
        out = Tensor(self.val @ other.val, self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad @ other.val.T, self.shape)
            if other.requires_grad:
                other.grad += unbroadcast(self.val.T @ out.grad, other.shape)

        out._backward = _backward
        out._prev = {self, other}

        return out

    def __add__(self, other):
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
        out = Tensor(-self.val, self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad -= unbroadcast(out.grad, self.shape)

        out._backward = _backward
        out._prev = {self}

        return out

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
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
        out = Tensor(np.abs(self.val), self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(np.sign(self.val) * out.grad, self.shape)

        out._backward = _backward
        out._prev = {self}

        return out

    def __pow__(self, power):
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
        out = Tensor(np.sum(self.val), self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad

        out._backward = _backward
        out._prev = {self}

        return out

    min = wrap_numpy(np.min)
    max = wrap_numpy(np.max)
    mean = wrap_numpy(np.mean)
    std = wrap_numpy(np.std)

    def backward(self):
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
