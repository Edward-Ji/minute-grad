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

    def __init__(self,
                 tensors,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8):
        super().__init__(tensors)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iteration = 1
        self.tensors = list(tensors)

    def zero_grad(self)

    def optimise(self, tensor):
        # update the first moment
        tensor.first_moment = ((self.beta1 * tensor.first_moment) + ((1 - self.beta1) * tensor.grad))/(1 - (self.beta1 ** self.iteration))
        # update the second moment
        tensor.second_moment = (self.beta2 * tensor.second_moment) + ((1 - self.beta2) * np.square(tensor.grad))/(1 - (self.beta2 ** self.iteration))
        # update the parameters
        tensor.data = tensor.data - ((self.learning_rate/(np.sqrt(tensor.second_moment) + self.epsilon)) * self.first_moment)
