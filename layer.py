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

    def __init__(self, features, num_outputs, initialise=kaiming_uniform):
        self.weights = Tensor(initialise(num_outputs, features), True)
        self.bias = Tensor(np.zeros((num_outputs, 1)), True)

    def forward(self, x) -> Tensor:
        # Note: calculating this may cause issues if the values in x get fairly large (even up to 1000 can cause issues)
        # May not be an issue for now, but might consider doing some normalisation to ensure consistency
        # See https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative for more details
        weighted = self.weights.val @ x.val + self.bias.val

        out = Tensor(np.exp(x.val) / np.sum(np.exp(x.val), axis=1).reshape(len(x.val), 1), x.requires_grad)
        def _backward():
            # do the following for each output value
            for idx in len(out.val):
                output_val = out.val[idx]
                # get the derivative of the softmax output w.r.t the softmax input logits
                dS_dz = np.diag(output_val.squeeze()) - np.outer(output_val, output_val)

                # create the weight gradient update matrix, and fill it with the raw input values
                # this will have shape (num_softmax_outputs, num_softmax_outputs * num_raw_inputs), which is the number of weights we have
                # each row represents how each output wants to change the weights
                dW_rows = np.repeat(x.val[idx], num_outputs, axis=0)
                for i in range(num_outputs - 1):
                    dW_rows = np.concatenate((dW_rows, np.repeat(x.val[i], num_outputs, axis=0)), axis=1)

                # multiply each input value by each softmax derivative to calculate dS/dz * dz/dW = dS/dW
                for i in range(len(dS_dz)):
                    dW_rows[:, features * i: features * (i + 1)] *= dS_dz[i].reshape(len(dS_dz[i]), 1)

                # now, add up the gradients of each weight, where in each row t, the gradient for W_ij is given by i * features + j (assuming zero indexing)
                for output_t in range(num_outputs):
                    for i in range(num_outputs):
                        for j in range(features):
                            self.weights.grad[i, j] += dW_rows[output_t, (i * features) + j]

                # god damn finally, now just need to update the weight of x itself

            
            
            # x.grad += unbroadcast(x.grad @ self.weights.val, x.shape)

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
        normalised_data = Tensor((x.val - mu) / np.sqrt(sigma_squared + self.epsilon, axis=0), x.requires_grad)
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
        out = normalised_data * self.weight + self.bias 

        def _backward():
            beta_gradient = np.sum(out.grad, axis=0)
            gamma_gradient = np.sum(out.grad * normalised_data.val, axis=0)
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

    def __init__(self, epsilon=1e-15):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        n_sample = logits.val.shape[0]

        exp_logits = np.exp(logits.val)
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
        probs = exp_logits / sum_exp
        error = -np.log(probs[np.arange(n_sample), labels.val] + self.epsilon)
        out = Tensor(np.mean(error), logits.requires_grad)

        def _backward():
            if logits.requires_grad:
                true_probs = np.zeros_like(probs)
                true_probs[np.arange(n_sample), labels.val] = 1
                logits.grad += probs - true_probs

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
