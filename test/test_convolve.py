import unittest

import numpy as np

from neuralnetwork.layers.conv import convolve, crosscorrelate
from scipy.signal import convolve2d
from scipy.signal import correlate2d

class TestConvolve(unittest.TestCase):

    def test_convolve(self):
        x = np.array([
            [1, 2, 1, 0, 2],
            [2, 0, 0, 1, 0],
            [1, 0, 2, 1, 0],
            [0, 1, 0, 2, 1],
            [0, 2, 1, 0, 2],
        ])

        kernel = np.array([
            [1, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
        ])

        padding = 'same'

        expected = convolve2d(x, kernel, mode=padding)
        actual = convolve(x, kernel, padding=padding)

        print(expected)
        print(actual)

        np.testing.assert_equal(actual, expected)


    def test_crosscorrelate(self):
        x = np.array([
            [1, 2, 1, 0, 2],
            [2, 0, 0, 1, 0],
            [1, 0, 2, 1, 0],
            [0, 1, 0, 2, 1],
            [0, 2, 1, 0, 2],
        ])

        kernel = np.array([
            [1, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
        ])

        padding = 'same'

        expected = correlate2d(x, kernel, mode=padding)
        actual = crosscorrelate(x, kernel, padding=padding)

        print(expected)
        print(actual)

        np.testing.assert_equal(actual, expected)

