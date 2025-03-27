from abc import abstractmethod
from typing import Any
import numpy as np
from tensor import Tensor


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


class Linear(Layer):
    
    def __init__(self, no_input, no_output):
        self.no_input = no_input
        self.no_output = no_output

        self.weight = Tensor(np.random.randn(no_input, no_output), True)
        self.bias = Tensor(np.random.randn(1, no_output), True)

    def forward(self, x) -> Tensor:
        out = x @ self.weight + self.bias
        return out



class Softmax(Layer):
    
    def __init__(self, input_dim, output_dim):
        self.input = np.zeros(input_dim)
        self.output = np.zeros(output_dim)
        self.grad = 0

    def calculate_probabilities(self):
        self.output = np.exp(self.input) / np.sum(np.exp(self.input))
        return self.output

    def backward(self):
        #used for backprop. todo, idk what to do here
        pass



class BatchNormalisation(Layer):
    
    def __init__(self, input_dim, epsilon=1e-5):
        self.input = np.zeros(input_dim)
        self.output = np.zeros(input_dim)
        self.epsilon = epsilon

    def calculate_output(self):
        self.output = (self.input - np.mean(self.input)) / np.sqrt(np.var(self.input) + self.epsilon)
        return self.output


class CrossEntropyLoss(Layer):

    def __init__(self):
        self.num_examples = 0
        self.loss = 0
        self.grad = None

    def update_loss(self, prob_predictions, prob_true):
        # note: this function requires the true prediction to be in one-hot form (i.e. 0 for all the wrong labels and 1 for the right label)
        # use this function to calculate the loss for each training example's output before undertaking backpropagation
        self.loss = ((self.loss * self.num_examples) + (-np.sum(prob_true * np.log(prob_predictions)))) / (self.num_examples + 1)
        self.num_examples += 1
        # add this training example's error
        self.grad = np.append(self.error_matrix, prob_predictions - prob_true, axis=0) if self.error_matrix else np.array(prob_predictions - prob_true, ndmin=2)
        return self.loss

    def get_gradient():
        # use this after finishing a batch, and use the result in the backprop process
        return self.grad

    def clear_loss(self):
        # use this function after each batch of training
        self.loss = 0
        self.num_examples = 0
        self.grad = None

    def get_loss(self):
        # return the calculated loss value
        return self.loss
