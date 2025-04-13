from abc import abstractmethod

import numpy as np


class Optimiser:
    """
    Base class for all optimisers.
    """

    def __init__(self, tensors):
        self.tensors = list(tensors)

    def zero_grad(self):
        """
        Set all tensor gradients to zero.
        """
        for tensor in self.tensors:
            tensor.grad = np.zeros_like(tensor.grad)

    @abstractmethod
    def optimise(self):
        """
        Optimise the tensors.
        """
        raise NotImplementedError("Optimiser must implement optimise method.")


class GradientDescentOptimiser(Optimiser):
    """
    This is a simple gradient descent optimiser that updates the tensor values
    based on the gradients and a learning rate. It simply subtracts the
    gradients from the tensor values, scaled by the learning rate. An optional
    weight decay term can be added to the gradients before updating the tensor
    values. The weight decay term is simply the tensor value multiplied by the
    weight decay factor.
    """

    name = "SGD"

    def __init__(self, tensors, learning_rate=1e-3, weight_decay=0.0):
        super().__init__(tensors)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def optimise(self):
        for tensor in self.tensors:
            if self.weight_decay != 0:
                tensor.grad += self.weight_decay * tensor.val
            tensor.val = tensor.val - (self.learning_rate * tensor.grad)


class AdamOptimiser(Optimiser):
    """
    Adam is a stochastic gradient descent method that is based on the idea of
    adaptive learning rates. It uses the first and second moments of the
    gradients to adapt the learning rate for each parameter. The first moment
    is the mean of the gradients, and the second moment is the uncentered
    variance of the gradients. Adam uses a moving average of the first and
    second moments of the gradients to adapt the learning rate for each
    parameter. The moving averages are computed using exponential decay rates
    controlled by beta1 and beta2.
    """

    name = "Adam"

    def __init__(
        self,
        tensors,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(tensors)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.first_moment = {
            tensor: np.zeros_like(tensor.grad) for tensor in self.tensors
        }
        self.second_moment = {
            tensor: np.zeros_like(tensor.grad) for tensor in self.tensors
        }
        self.iterations = 1

    def optimise(self):
        for tensor in self.tensors:
            if self.weight_decay != 0:
                tensor.grad += self.weight_decay * tensor.val
            self.first_moment[tensor] = (
                self.beta1 * self.first_moment[tensor] + (1 - self.beta1) * tensor.grad
            )
            self.second_moment[tensor] = (
                self.beta2 * self.second_moment[tensor]
                + (1 - self.beta2) * tensor.grad**2
            )
            # Bias correction
            first_moment_corrected = self.first_moment[tensor] / (
                1 - pow(self.beta1, self.iterations)
            )
            second_moment_corrected = self.second_moment[tensor] / (
                1 - pow(self.beta2, self.iterations)
            )
            tensor.val -= (
                self.learning_rate
                * first_moment_corrected
                / (np.sqrt(second_moment_corrected) + self.epsilon)
            )
