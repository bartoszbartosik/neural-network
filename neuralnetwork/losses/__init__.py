import numpy as np


def mse(target, prediction):
    """
    Return total Mean Squared Error
    """
    # Initialize error array
    errors = []

    # Calculate error for each training data
    for y, a in zip(target, prediction):
        # Compare it with the expected outcome
        error: np.ndarray = (y - a)**2

        # Append it to array error
        errors.append(error)

    errors = np.mean(np.array(errors), axis=0)
    errors = np.sum(errors)/errors.size

    return errors