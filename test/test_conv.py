import unittest

import numpy as np

import neuralnetwork.losses
from neuralnetwork import Network
from neuralnetwork import losses
from neuralnetwork.layers import InputLayer, Dense, Convolutional
from neuralnetwork.activations import sigmoid, linear, relu
import keras


class TestConv(unittest.TestCase):

    def setUp(self):
        # Define input shape
        input_shape = (1, 5, 5, 1)  # (batch_size, rows, cols, channels)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Build model
        self.net = Network()
        self.net.add_layer(InputLayer(input_shape=input_shape))
        self.net.add_layer(Convolutional(1, (3, 3), activation=relu, padding='same'))
        self.net.compile(neuralnetwork.losses.mse)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Build reference model
        self.kerasnet = keras.models.Sequential()
        self.kerasnet.add(keras.layers.Conv2D(2, (2, 2), input_shape=(5, 5, 2), activation='relu', padding='valid'))
        self.kerasnet.compile()


    def test_prediction(self):
        x = np.array([
            [1, 2, 1, 0, 2],
            [2, 0, 0, 1, 0],
            [1, 0, 2, 1, 0],
            [0, 1, 0, 2, 1],
            [0, 2, 1, 0, 2],
        ])

        self.net.layers[-1].kernels = np.array([[[
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ]]])
        self.net.layers[-1].biases = np.zeros_like(self.net.layers[-1].biases)

        a_net = self.net.predict(x.reshape(1, 5, 5, 1))
        print(a_net)

    def test_prediction_2(self):
        x = np.array([
            [[1, 2, 1, 0, 2],
             [2, 0, 0, 1, 0],
             [1, 0, 2, 1, 0],
             [0, 1, 0, 2, 1],
             [0, 2, 1, 0, 2]],
            [[1, 2, 1, 0, 2],
             [2, 0, 0, 1, 0],
             [1, 0, 2, 1, 0],
             [0, 1, 0, 2, 1],
             [0, 2, 1, 0, 2]],
        ])

        self.net.layers[0] = InputLayer(input_shape=(1, 5, 5, 2))
        self.net.layers[1] = Convolutional(2, (2, 2), activation=relu, padding='valid')
        self.net.compile(losses.mse)

        self.net.layers[1].kernels = np.array(self.kerasnet.layers[0].get_weights()[0].transpose())
        self.net.layers[1].biases = np.zeros_like(self.net.layers[1].biases)

        a = self.net.predict(x.reshape(1, 5, 5, 2))
        b = self.kerasnet.predict(x.reshape(1, 5, 5, 2))
        print(a)
        print(a.shape)


