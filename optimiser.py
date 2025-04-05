import numpy as np


class Optimiser:

    def __init__(self, tensors):
        self.tensors = list(tensors)

    def zero_grad(self):
        for tensor in self.tensors:
            tensor.grad = np.zeros_like(tensor.grad)


class GradientDescentOptimiser(Optimiser):

    def __init__(self, tensors, learning_rate=1e-3):
        super().__init__(tensors)
        self.learning_rate = learning_rate

    def optimise(self):
        for tensor in self.tensors:
            tensor.val = tensor.val - (self.learning_rate * tensor.grad)


class AdamOptimiser(Optimiser):

    def __init__(
        self, tensors, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
        super().__init__(tensors)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first_moment = {
            tensor: np.zeros_like(tensor.grad) for tensor in self.tensors
        }
        self.second_moment = {
            tensor: np.zeros_like(tensor.grad) for tensor in self.tensors
        }

    def optimise(self):
        for tensor in self.tensors:
            self.first_moment[tensor] = (
                self.beta1 * self.first_moment[tensor] + (1 - self.beta1) * tensor.grad
            )
            self.second_moment[tensor] = (
                self.beta2 * self.second_moment[tensor]
                + (1 - self.beta2) * tensor.grad**2
            )
            # Bias correction
            first_moment_corrected = self.first_moment[tensor] / (1 - self.beta1)
            second_moment_corrected = self.second_moment[tensor] / (1 - self.beta2)
            tensor.val -= (
                self.learning_rate
                * first_moment_corrected
                / (np.sqrt(second_moment_corrected) + self.epsilon)
            )
