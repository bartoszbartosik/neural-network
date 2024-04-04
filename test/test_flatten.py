import unittest

import numpy as np

import neuralnetwork.losses
from neuralnetwork import Network
from neuralnetwork import losses
from neuralnetwork.layers import InputLayer, Dense, Convolutional, MaxPooling, Flatten
from neuralnetwork.activations import sigmoid, linear, relu
import keras


class TestConv(unittest.TestCase):

    def test_flatten_1(self):
        # Define input shape
        batch_size = 1
        input_shape = (4, 4, 1)  # (rows, cols, channels)

        # Define MaxPooling parameters
        pool_size = (2, 2)
        stride = 2

        # Build model
        net = Network()
        net.add_layer(InputLayer(input_shape=(batch_size, *input_shape)))
        net.add_layer(MaxPooling(pool_size, stride))
        net.add_layer(Flatten())
        net.compile(neuralnetwork.losses.mse)

        # Build reference model
        kerasnet = keras.models.Sequential()
        kerasnet.add(keras.layers.MaxPool2D(pool_size, stride))
        kerasnet.add(keras.layers.Flatten())
        kerasnet.compile()

        x = np.array([
            [8, 3, 5, 9],
            [4, 7, 2, 4],
            [6, 5, 2, 1],
            [1, 7, 3, 6],
        ])

        a_net = net.predict(x.reshape(batch_size, *input_shape))
        a_kerasnet = kerasnet.predict(x.reshape(batch_size, *input_shape))
        print(a_net)
        print(a_kerasnet)

        np.testing.assert_allclose(a_net, a_kerasnet)


    def test_flatten_2(self):
        # Define input shape
        batch_size = 2
        input_shape = (5, 6, 1)  # (rows, cols, channels)

        # Kernels
        kernels = 1
        kernel_size = (3, 3)

        # Build model
        net = Network()
        net.add_layer(InputLayer(input_shape=(batch_size, *input_shape)))
        net.add_layer(Convolutional(kernels, kernel_size, activation=relu, padding='valid'))
        net.add_layer(Flatten())
        net.compile(losses.mse)

        # Build reference model
        kerasnet = keras.models.Sequential()
        kerasnet.add(keras.layers.Conv2D(kernels, kernel_size, input_shape=input_shape, activation='relu', padding='valid'))
        kerasnet.add(keras.layers.Flatten())
        kerasnet.compile()

        # Define sample input consisting of 2 samples/channels
        x = np.array([
            [[1, 2, 1, 0, 2, 2],
             [2, 0, 0, 1, 0, 1],
             [1, 0, 2, 1, 0, 0],
             [0, 1, 0, 2, 1, 0],
             [0, 2, 1, 0, 2, 1]],
            [[1, 2, 1, 0, 2, 2],
             [2, 0, 0, 1, 0, 1],
             [1, 0, 2, 1, 0, 0],
             [0, 1, 0, 2, 1, 0],
             [0, 2, 1, 0, 2, 1]],
        ])

        # Get reference model weights and assign to the tested one
        keras_weights = kerasnet.layers[0].get_weights()[0]
        _, _, channel, batch = keras_weights.shape
        for b in range(batch):
            for c in range(channel):
                net.layers[1].kernels[b, c, :, :] = keras_weights[:, :, c, b]
        net.layers[1].biases = np.zeros_like(net.layers[1].biases)

        # Predict actual and expected output, respectively
        a = net.predict(x.reshape(batch_size, *input_shape))
        b = kerasnet.predict(x.reshape(batch_size, *input_shape))
        print(a)
        print(b)

        np.testing.assert_allclose(a, b, rtol=1e-5)




