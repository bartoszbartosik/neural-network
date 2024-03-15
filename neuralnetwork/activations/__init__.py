from typing import Callable

import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


# Rectified Linear Unit
def relu(x):
    return 0 if x <= 0 else x


def linear(x):
    return x


# Return derivative value of a given function in a given point
def derivative(function: Callable, x: np.ndarray):
    match function.__name__:
        case 'sigmoid': return function(x)*(1-function(x))
        case 'relu': return 0 if x <= 0 else 1
        case 'linear': return 1