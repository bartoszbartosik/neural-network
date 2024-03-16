from typing import Callable

import numpy as np


def d(function: Callable, prediction: np.ndarray, target: np.ndarray):
    match function.__name__:
        case 'mse': return 2*(prediction - target)


def mse(prediction, target):
    """
    Return total Mean Squared Error
    """
    return np.mean((target - prediction)**2)
