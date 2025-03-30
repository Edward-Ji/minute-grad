from abc import abstractmethod
from typing import Any

import numpy as np

from tensor import Tensor
from util import kaiming_uniform, unbroadcast


class Layer:

    def get_all_tensors(self):
        for attr in self.__dict__.values():
            if isinstance(attr, Tensor):
                yield attr
            elif isinstance(attr, Layer):
                yield from attr.get_all_tensors()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)


class Identity(Layer):

    def forward(self, x) -> Tensor:
        return x


class Linear(Layer):

    def __init__(self, no_input, no_output, initialise=kaiming_uniform):
        self.no_input = no_input
        self.no_output = no_output

        self.weight = Tensor(initialise(no_input, no_output), True)
        self.bias = Tensor(np.zeros((1, no_output)), True)

    def forward(self, x) -> Tensor:
        out = x @ self.weight + self.bias
        return out


class ReLU(Layer):

    def forward(self, x) -> Tensor:
        out = Tensor(np.maximum(x.val, 0), x.requires_grad)

        def _backward():
            x.grad += unbroadcast(out.grad * np.sign(out.val), x.shape)

        out._backward = _backward
        out._prev = {x}

        return out


class Dropout(Layer):

    def __init__(self, p=0.5):
        self.p = p

    def forward(self, x) -> Tensor:
        if self.p == 0:
            return x

        mask = np.random.binomial(1, 1 - self.p, x.shape)
        out = Tensor(x.val * mask, x.requires_grad)

        def _backward():
            x.grad += unbroadcast(out.grad * mask, x.shape)

        out._backward = _backward
        out._prev = {x}

        return out


class Sigmoid(Layer):

    def forward(self, x) -> Tensor:
        out = Tensor(1 / (1 + np.exp(-x.val)), x.requires_grad)

        def _backward():
            x.grad += unbroadcast(out.grad * out.val * (1 - out.val), x.shape)

        out._backward = _backward
        out._prev = {x}

        return out


class Softmax(Layer):

    def forward(self, x) -> Tensor:
        raise NotImplementedError


class BatchNormalisation(Layer):

    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        raise NotImplementedError


class MeanSquaredError(Layer):

    def forward(self, pred, truth) -> Tensor:
        loss = ((pred.val - truth.val) ** 2).sum() / len(truth)
        out = Tensor(loss, pred.requires_grad)

        def _backward():
            pred.grad += 2 * (pred.val - truth.val) / len(truth)

        out._backward = _backward
        out._prev = {pred}
        return out


class CrossEntropyLoss(Layer):

    def forward(self, logits, labels) -> Tensor:
        raise NotImplementedError


class Composite(Layer):

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def get_all_tensors(self):
        for attr in self.layers:
            yield from attr.get_all_tensors()

    def forward(self, x) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
