from abc import abstractmethod
from typing import Any

import numpy as np

from tensor import Tensor
from util import kaiming_uniform, unbroadcast


class Layer:

    def __init__(self):
        self.training = True

    def train(self, training: bool):
        if self.training != training:
            self.training = training
            for attr in self.__dict__.values():
                if isinstance(attr, Layer):
                    attr.train(training)

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
        super().__init__()

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
        super().__init__()
        self.p = p

    def forward(self, x) -> Tensor:
        if not self.training or self.p == 0:
            return x

        mask = np.random.binomial(1, 1 - self.p, x.shape) / (1 - self.p)
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
    # DEPRECIATED CLASS
    # dont use this, the cross-entropy loss already has softmax built into it
    def forward(self, x) -> Tensor:
        # Note: calculating this may cause issues if the values in x get fairly large (even up to 1000 can cause issues)
        # May not be an issue for now, but might consider doing some normalisation to ensure consistency
        # See https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative for more details

        out = Tensor(np.exp(x.val) / np.sum(np.exp(x.val), axis=1).reshape(len(x.val), 1), x.requires_grad)
        def _backward():
            # do the following for each output value
            # this could probably be done more clearly with numpy operations but im just trying to survive rn
            for idx in range(len(out.val)):
                output_val = out.val[idx]
                # get the derivative of the softmax output w.r.t the softmax input logits
                dS_dz = np.diag(output_val.squeeze()) - np.outer(output_val, output_val)
                out.grad[idx] += np.sum(dS_dz, axis=0)

        out._backward = _backward
        out._prev = {x}

        return out


class BatchNormalisation(Layer):

    def __init__(self, features, epsilon=1e-5, initialise=kaiming_uniform):
        super().__init__()
        self.epsilon = epsilon
        self.weights = Tensor(np.ones((1, features)), True)
        self.bias = Tensor(np.zeros((1, features)), True)

    def forward(self, x) -> Tensor:
        mu = np.mean(x.val, axis=0)
        sigma_squared = np.var(x.val, axis=0)
        batch_size = len(x.val)
        normalised_data = (x.val - mu) / np.sqrt(sigma_squared + self.epsilon)
        # We do an elementwise multiplication here with the weights because normalised data is a tensor containing multiple input samples
        # Therefore, we want to multiply the weight array with each respective column in the tensor
        # e.g. if our input data is
        # input = [[1, 2, 3]
        #          [4, 5, 6]
        #          [7, 8, 9]]
        # (i.e. 3 samples in the batch with 3 features each)
        
        # and our weight/bias is
        # weights = [10, 20, 30]
        # 
        # then this represents that the weight for the first feature is 10, the weight for the second feature is 20, etc.
        # therefore, we want the result to be 
        # input * weights = [[10, 40, 90]
        #                    [40, 100, 180]
        #                    [70, 160, 270]]
        out = Tensor(normalised_data * self.weights.val + self.bias.val, x.requires_grad)

        def _backward():
            beta_gradient = np.sum(out.grad, axis=0)
            gamma_gradient = np.sum(out.grad * normalised_data, axis=0)
            self.bias.grad += beta_gradient
            self.weights.grad += gamma_gradient

            normalised_gradient = out.grad * self.weights.val
            var_inv = 1.0 / np.sqrt(sigma_squared + self.epsilon)
            var_gradient = np.sum(normalised_gradient * (x.val - mu) * -0.5 * (var_inv ** 3), axis=0)
            mu_gradient = np.sum(normalised_gradient * -var_inv, axis=0) + var_gradient * np.sum(-2 * (x.val - mu), axis=0) / batch_size
            x.grad += (normalised_gradient * var_inv) + (var_gradient * 2 * (x.val - mu) / batch_size) + (mu_gradient / batch_size)

        out._backward = _backward
        out._prev = {x}

        return out


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

    def __init__(self, epsilon=1e-15, label_smoothing=0.0):
        super().__init__()
        self.epsilon = epsilon
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        n_sample, n_class = logits.val.shape

        # THIS SECTION IS SOFTMAX
        exp_logits = np.exp(logits.val)
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
        probs = exp_logits / sum_exp

        # Create one-hot encoding for labels.
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n_sample), labels.val] = 1

        # Apply label smoothing if required.
        if self.label_smoothing > 0:
            smooth_labels = (
                one_hot * (1 - self.label_smoothing)
                + self.label_smoothing / n_class
            )
        else:
            smooth_labels = one_hot

        # Compute the per-sample losses and take the mean.
        losses = -np.sum(smooth_labels * np.log(probs + self.epsilon), axis=1)
        loss_value = np.mean(losses)
        out = Tensor(loss_value, logits.requires_grad)

        def _backward():
            if logits.requires_grad:
                # The gradient is (probs - smooth_labels) averaged over the samples.
                logits.grad += (probs - smooth_labels) / n_sample

        out._backward = _backward
        out._prev = {logits}
        return out


class Composite(Layer):

    def __init__(self, layers: list[Layer]):
        super().__init__()
        self.layers = layers

    def mode(self, training: bool):
        self.training = training
        for attr in self.layers:
            attr.train(training)

    def get_all_tensors(self):
        for attr in self.layers:
            yield from attr.get_all_tensors()

    def forward(self, x) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
