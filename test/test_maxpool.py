import unittest

import numpy as np

import neuralnetwork.losses
from neuralnetwork import Network
from neuralnetwork import losses
from neuralnetwork.layers import InputLayer, Dense, Convolutional, MaxPooling
from neuralnetwork.activations import sigmoid, linear, relu
import keras


class TestConv(unittest.TestCase):

    def test_maxpooling(self):
        # Define input shape
        batch_size = 1
        input_shape = (4, 4, 1)  # (rows, cols, channels)

        x = np.array([
            [8, 3, 5, 9],
            [4, 7, 2, 4],
            [6, 5, 2, 1],
            [1, 7, 3, 6],
        ])

        # Define MaxPooling parameters
        pool_size = (2, 2)
        stride = 1

        # Build model
        net = Network()
        net.add_layer(InputLayer(input_shape=(batch_size, *input_shape)))
        net.add_layer(MaxPooling(pool_size, stride))
        net.compile(neuralnetwork.losses.mse)

        # Build reference model
        kerasnet = keras.models.Sequential()
        kerasnet.add(keras.layers.MaxPool2D(pool_size, stride))
        kerasnet.compile()

        # Predict
        a_net = net.predict(x.reshape(batch_size, *input_shape))
        a_kerasnet = kerasnet.predict(x.reshape(batch_size, *input_shape))
        print(a_net[0, :, :, 0])
        print(a_kerasnet[0, :, :, 0])

        np.testing.assert_allclose(a_net, a_kerasnet)


    def test_maxpooling_2(self):
        # Define input shape
        batch_size = 1
        input_shape = (4, 4, 1)  # (rows, cols, channels)

        x = np.array([
            [8, 3, 5, 9],
            [4, 7, 2, 4],
            [6, 5, 2, 1],
            [1, 7, 3, 6],
        ])

        # Define MaxPooling parameters
        pool_size = (2, 2)
        stride = 1

        # Build model
        net = Network()
        net.add_layer(InputLayer(input_shape=(batch_size, *input_shape)))
        net.add_layer(MaxPooling(pool_size, stride))
        net.compile(neuralnetwork.losses.mse)

        # Build reference model
        kerasnet = keras.models.Sequential()
        kerasnet.add(keras.layers.MaxPool2D(pool_size, stride))
        kerasnet.compile()

        # Predict
        a_net = net.predict(x.reshape(batch_size, *input_shape))
        a_kerasnet = kerasnet.predict(x.reshape(batch_size, *input_shape))
        print(a_net[0, :, :, 0])
        print(a_kerasnet[0, :, :, 0])

        np.testing.assert_allclose(a_net, a_kerasnet)


