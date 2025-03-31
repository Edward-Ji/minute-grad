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
        # Note: calculating this may cause issues if the values in x get fairly large (even up to 1000 can cause issues)
        # May not be an issue for now, but might consider doing some normalisation to ensure consistency
        # See https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative for more details
        out = Tensor(np.exp(x.val) / np.sum(np.exp(x.val)), x.requires_grad)

        def _backward():
            x.grad += unbroadcast((np.diag(out.val) - np.outer(out.val, out.val)), x.shape)

        out._backward = _backward
        out._prev = {x}

        return out


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

    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def forward(self, logits, labels) -> Tensor:
        # for each training example, identify the correct class (generate an array of indices, where each index refers to the correct label)
        correct_classes = np.argmax(labels, axis=1)
        # using the indices of the correct labels, find the predicted probability of the correct label in each training example
        correct_values = np.array([logits[i][correct_classes[i]] for i in range(len(logits))])
        # take the negative log and average it out to obtain the loss value
        loss = (-np.log(correct_values)).sum() / len(labels)
        out = Tensor(loss, logits.requires_grad)

        def _backward():
            # note: we add epsilon here to avoid a divide by 0 error
            logits.grad += unbroadcast(-1 / (correct_values + self.epsilon), logits.shape)

        out._backward = _backward
        out._prev = {logits}
        return out

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
