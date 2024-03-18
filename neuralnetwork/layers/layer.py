from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class Layer(ABC):

    def __init__(self, activation: Callable, weights: np.ndarray = None, bias: np.ndarray = None):
        # Initialize weights if given
        self.w: np.ndarray = weights
        self.b: np.ndarray = bias

        # Initialize neurons
        self.activation = activation
        self.z: np.ndarray = np.zeros([])
        self.a: np.ndarray = np.zeros([])

        # Initialize loss function
        self.loss: Callable = lambda: None


    @abstractmethod
    def compile(self, a_: np.ndarray, loss: Callable) -> None:
        pass


    @abstractmethod
    def feedforward(self, a_: np.ndarray) -> None:
        pass


    @abstractmethod
    def backpropagate(self,
                      l_prev: Layer,
                      delta_prev: np.ndarray = None,
                      l_next: Layer = None,
                      y: np.ndarray = None) -> tuple:
        pass


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # #   O V E R W R I T T E N   F U N C T I O N S   # # # # # # # # # # # # # # # # #
    def __str__(self) -> str:
        return 'Layer: [{}]'.format(self.a)

    def __repr__(self):
        return self.__str__()

