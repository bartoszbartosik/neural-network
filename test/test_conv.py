import unittest

import numpy as np

import neuralnetwork.losses
from neuralnetwork import ANN
from neuralnetwork import losses
from neuralnetwork.layers import InputLayer, Dense, Convolutional
from neuralnetwork.activations import sigmoid, linear, relu
import keras


class TestFeedforward(unittest.TestCase):

    def setUp(self):
        # Define input shape
        input_shape = (4,4)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Build model
        self.net = ANN()
        self.net.add_layer(InputLayer(input_shape=input_shape))
        self.net.add_layer(Convolutional(3, (2, 2), activation=relu))
        self.net.compile(neuralnetwork.losses.mse)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Build reference model
        self.kerasnet = keras.models.Sequential()
        self.kerasnet.add(keras.layers.Conv2D(input_shape=(None, input_shape[0])))
        self.kerasnet.compile()


    def prediction(self):
        x = np.array([
            [1.0, 0.5, 0.5, 1.0],
            [0.0, 0.5, 1.0, 0.5],
            [0.0, 0.5, 0.0, 0.5],
            [1.0, 0.5, 1.0, 0.0],
        ])

        self.net.predict(x)
        pass