from typing import Callable

import numpy as np


def d(function: Callable, prediction: np.ndarray, target: np.ndarray):
    match function.__name__:
        case 'mse': return 2*(prediction - target)
        case 'cross_entropy': return (1 - target) / (1 - prediction) - target / prediction


def mse(prediction, target):
    """
    Return total Mean Squared Error
    """
    return np.mean((target - prediction)**2)


def cross_entropy(prediction, target):
    return -np.mean(target * np.log(prediction) + (1 - target) * np.log(1-prediction))
