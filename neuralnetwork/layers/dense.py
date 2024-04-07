from typing import Callable

import numpy as np

from neuralnetwork.layers import Layer
from neuralnetwork import losses
from neuralnetwork import activations


class Dense(Layer):

    def __init__(self, neurons: int, activation: Callable, weights: np.ndarray = None, bias: float = None):
        super().__init__(activation=activation,
                         weights=weights,
                         bias=bias)

        self.shape = (1, neurons)
        self.z: np.ndarray = np.zeros(neurons)
        self.a: np.ndarray = np.zeros(neurons)


    def build(self, a_: np.ndarray, loss: Callable) -> None:
        a_ = a_.reshape(1, -1)
        self.loss = loss
        _, neurons = self.shape
        if self.w is None:
            self.w = np.random.rand(neurons, len(a_[0])) * 2 - 1
        if self.b is None:
            self.b = np.random.random() * 2 - 1


    def feedforward(self, a_):
        self.z = (np.dot(self.w, a_.transpose()) + self.b).transpose()
        self.a = self.activation(self.z)


    def backpropagate(self, grad: np.ndarray, a_: np.ndarray, w_out: np.ndarray = None) -> tuple:
        """
        Calculate the Cost Function derivative over each weight and bias in order to determine its gradient.
        """
        # If the layer is hidden, use weights from the output to scale the gradient
        grad_b = grad if w_out is None else np.dot(grad, w_out)
        grad_b *= activations.d(self.activation, self.z)
        grad_w = [np.outer(g, a) for g, a in zip(grad_b, a_)]

        return grad_b, grad_w

