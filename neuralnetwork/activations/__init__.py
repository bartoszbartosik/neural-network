from typing import Callable

import numpy as np


# Return derivative value of a given function in a given point
def d(f: Callable, x: np.ndarray):
    match f.__name__:
        case 'sigmoid': return f(x)*(1 - f(x))
        case 'relu': return 1 * (x > 0)
        case 'linear': return 1


def sigmoid(x):
    return 1/(1+np.exp(-x))


# Rectified Linear Unit
def relu(x):
    return x * (x > 0)


def linear(x):
    return x