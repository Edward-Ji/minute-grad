from abc import abstractmethod
from typing import Any

import numpy as np

from tensor import Tensor
from util import kaiming_uniform, unbroadcast


class Layer:
    """
    This is the base class for all layers in the neural network. It can also represent a
    model composed of multiple layers. Any custom layer or model should inherit from
    this class.

    The `forward` method must be implemented in subclasses to define the forward pass
    of the layer or model. The `train` method is used to set the layer to training or
    evaluation mode. The `get_all_tensors` method is used to retrieve all tensors
    associated with the layer or model. Alternatively, callers can use the Python
    function call syntax to invoke the `forward` method.
    """

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
    """
    This is a placeholder layer that returns the input as is.
    """

    def forward(self, x) -> Tensor:
        return x


class Linear(Layer):
    """
    A linear layer that applies a linear transformation to the input data. Formally, it
    represents the equation y = Wx + b, where W is the weight matrix, x is the input
    data, and b is the bias vector. See `util` for initialisation functions.
    """

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
    """
    The rectified linear unit (ReLU) activation function. It is defined as
    f(x) = max(0, x).
    """

    def forward(self, x) -> Tensor:
        out = Tensor(np.maximum(x.val, 0), x.requires_grad)

        def _backward():
            x.grad += unbroadcast(out.grad * np.sign(out.val), x.shape)

        out._backward = _backward
        out._prev = {x}

        return out


class LeakyReLU(Layer):
    """
    The leaky rectified linear unit (Leaky ReLU) activation function. It is defined as
    f(x) = x if x > 0 else negative_slope * x, where negative_slope is a small constant.
    """

    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x) -> Tensor:
        out = Tensor(
            np.where(x.val > 0, x.val, self.negative_slope * x.val), x.requires_grad
        )

        def _backward():
            if x.requires_grad:
                x.grad += unbroadcast(
                    out.grad * np.where(out.val > 0, 1, self.negative_slope), x.shape
                )

        out._backward = _backward
        out._prev = {x}

        return out


class Dropout(Layer):
    """
    The dropout layer randomly sets a fraction of the input units to 0 at each update
    during training time, which helps prevent overfitting. Here, p is the probability of
    dropping an input unit.
    """

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
    """
    The sigmoid activation function. It is defined as f(x) = 1 / (1 + exp(-x)).
    """

    def forward(self, x) -> Tensor:
        out = Tensor(1 / (1 + np.exp(-x.val)), x.requires_grad)

        def _backward():
            x.grad += unbroadcast(out.grad * out.val * (1 - out.val), x.shape)

        out._backward = _backward
        out._prev = {x}

        return out


class BatchNormalisation(Layer):
    """
    This layer normalises the input data using batch statistics. It computes the mean
    and variance of the input data over the batch dimension and scales the data feature
    wise. The input data is expected to be in the shape (batch_size, features).
    """

    def __init__(self, features, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.weights = Tensor(np.ones((1, features)), True)
        self.bias = Tensor(np.zeros((1, features)), True)
        self.averages = np.zeros((1, features))
        self.variances = np.zeros((1, features))
        self.num_examples = 0

    def forward(self, x) -> Tensor:
        batch_size = len(x.val)
        if self.training:
            # if we are training, we need to store a running mean and variance of the values
            mu = np.mean(x.val, axis=0)
            self.averages = ((self.averages * self.num_examples) + np.sum(x.val, axis=0)) / (self.num_examples + batch_size)
            sigma_squared = np.var(x.val, axis=0)
            self.variances = ((self.variances * self.num_examples) + (np.var(x.val, axis=0) * batch_size)) / (self.num_examples + batch_size)
            self.num_examples += batch_size
        else:
            mu = self.averages
            sigma_squared = self.variances
        
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
        out = Tensor(
            normalised_data * self.weights.val + self.bias.val, x.requires_grad
        )

        def _backward():
            beta_gradient = np.sum(out.grad, axis=0)
            gamma_gradient = np.sum(out.grad * normalised_data, axis=0)
            self.bias.grad += beta_gradient
            self.weights.grad += gamma_gradient

            normalised_gradient = out.grad * self.weights.val
            var_inv = 1.0 / np.sqrt(sigma_squared + self.epsilon)
            var_gradient = np.sum(
                normalised_gradient * (x.val - mu) * -0.5 * (var_inv**3), axis=0
            )
            mu_gradient = (
                np.sum(normalised_gradient * -var_inv, axis=0)
                + var_gradient * np.sum(-2 * (x.val - mu), axis=0) / batch_size
            )
            x.grad += (
                (normalised_gradient * var_inv)
                + (var_gradient * 2 * (x.val - mu) / batch_size)
                + (mu_gradient / batch_size)
            )

        out._backward = _backward
        out._prev = {x}

        return out


class MeanSquaredError(Layer):
    """
    The mean squared error (MSE) loss function. It is defined as the average of the
    squared differences between the predicted and true values. This is a common loss
    function used in regression tasks.
    """

    def forward(self, pred, truth) -> Tensor:
        loss = ((pred.val - truth.val) ** 2).sum() / len(truth)
        out = Tensor(loss, pred.requires_grad)

        def _backward():
            pred.grad += 2 * (pred.val - truth.val) / len(truth)

        out._backward = _backward
        out._prev = {pred}
        return out


class CrossEntropyLoss(Layer):
    """
    The cross-entropy loss function. It computes the difference between the predicted
    probability distribution and the true distribution. It expects logits (before
    softmax) as input and applies the softmax function internally. The labels are
    expected to be class indices [0, n_classes). The loss is computed as the negative
    log likelihood of the true class.

    A small epsilon value is added to the log function to prevent numerical instability
    when the predicted probabilities are very close to zero. The label smoothing
    parameter represents the degree of smoothing applied to the labels.
    """

    def __init__(self, epsilon=1e-15, label_smoothing=0.0):
        super().__init__()
        self.epsilon = epsilon
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        n_sample, n_class = logits.val.shape

        # Integrate numerically stable Softmax into the loss calculation
        shifted = logits.val - np.max(logits.val, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
        probs = exp_logits / sum_exp

        # Create one-hot encoding for labels.
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n_sample), labels.val] = 1

        # Apply label smoothing if required.
        if self.label_smoothing > 0:
            smooth_labels = (
                one_hot * (1 - self.label_smoothing) + self.label_smoothing / n_class
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
    """
    A composite layer that represents a model composed of multiple layers. It simply
    applies each layer in sequence to the input data.
    """

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
