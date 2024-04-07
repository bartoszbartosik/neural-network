import unittest

import numpy as np

import neuralnetwork.losses
from neuralnetwork import Network
from neuralnetwork import losses
from neuralnetwork.layers import InputLayer, Dense, Convolutional, Flatten, MaxPooling
from neuralnetwork.activations import sigmoid, linear, relu
import keras


class TestConv(unittest.TestCase):

    def test_prediction_1(self):
        # Define input shape
        batch_size = 1
        input_shape = (5, 5, 1)  # (rows, cols, channels)

        # Kernel
        kernel_size = (3, 3)
        kernel = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ])
        bias = np.zeros((1,))

        # Build model
        net = Network()
        net.add_layer(InputLayer(input_shape=(batch_size, *input_shape)))
        net.add_layer(Convolutional(1, kernel_size, activation=relu, padding='valid'))
        net.compile(neuralnetwork.losses.mse)

        # Build reference model
        kerasnet = keras.models.Sequential()
        kerasnet.add(keras.layers.Conv2D(1, kernel_size, input_shape=input_shape, activation='relu', padding='valid'))
        kerasnet.compile()

        x = np.array([
            [1, 2, 1, 0, 2],
            [2, 0, 0, 1, 0],
            [1, 0, 2, 1, 0],
            [0, 1, 0, 2, 1],
            [0, 2, 1, 0, 2],
        ])

        net.layers[-1].w = kernel.reshape(net.layers[-1].w.shape)
        net.layers[-1].b = np.zeros_like(net.layers[-1].b)

        kerasnet.layers[-1].set_weights([kernel.reshape(3, 3, 1, 1), bias])

        a_net = net.predict(x.reshape(batch_size, *input_shape))
        a_kerasnet = kerasnet.predict(x.reshape(batch_size, *input_shape))
        print(a_net[0, :, :, 0])
        print(a_kerasnet[0, :, :, 0])

        np.testing.assert_allclose(a_net, a_kerasnet)

    def test_prediction_2(self):
        # Define input shape
        batch_size = 2
        input_shape = (5, 6, 1)  # (rows, cols, channels)

        # Kernels
        kernels = 2
        kernel_size = (3, 3)

        # Build model
        net = Network()
        net.add_layer(InputLayer(input_shape=(batch_size, *input_shape)))
        net.add_layer(Convolutional(kernels, kernel_size, activation=relu, padding='valid'))
        net.compile(losses.mse)

        # Build reference model
        kerasnet = keras.models.Sequential()
        kerasnet.add(keras.layers.Conv2D(kernels, kernel_size, input_shape=input_shape, activation='relu', padding='valid'))
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
                net.layers[1].w[b, c, :, :] = keras_weights[:, :, c, b]
        net.layers[1].b = np.zeros_like(net.layers[1].b)

        # Predict actual and expected output, respectively
        a = net.predict(x.reshape(batch_size, *input_shape))
        b = kerasnet.predict(x.reshape(batch_size, *input_shape))
        print(a[:, :, :, 0])
        print(b[:, :, :, 0])

        np.testing.assert_allclose(a, b, rtol=1e-5)

    def test_prediction_3(self):
        # Define input shape
        batch_size = 1
        input_shape = (4, 4, 1)  # (rows, cols, channels)

        # Define sample input
        x = np.array([
            [.7, -.3, -.7, .5],
            [.9, -.5, -.2, .9],
            [-.1, .8, -.3, -.5],
            [0, .2, -.1, .6]
        ])

        # Kernel
        kernel_size = (2, 2)
        kernel_1 = np.array([
            [.1, -.3],
            [-.5, .7]
        ])
        kernel_2 = np.array([
            [-.5, .1],
            [.3, .9]
        ])


        # Build model
        net = Network()
        net.add_layer(InputLayer(input_shape=(batch_size, *input_shape)))
        net.add_layer(Convolutional(1, kernel_size, activation=relu, padding='same'))
        net.add_layer(Convolutional(1, kernel_size, activation=relu, padding='valid'))
        net.add_layer(MaxPooling((2, 2), 1))
        net.add_layer(Flatten())
        net.add_layer(Dense(4, activation=sigmoid))
        net.compile(losses.mse)

        net.layers[1].w = kernel_1.reshape(net.layers[1].w.shape)
        net.layers[1].b = np.zeros_like(net.layers[1].b)
        net.layers[2].w = kernel_2.reshape(net.layers[2].w.shape)
        net.layers[2].b = np.zeros_like(net.layers[2].b)


        # Predict not trained model output
        a = net.predict(x.reshape(batch_size, *input_shape))
        print(a)

        # Expected output
        y = np.array([[0, 1, 0, 1]])

        # Train model
        net.fit(x.reshape(batch_size, *input_shape), y, epochs=20, batch_size=1, learning_rate=0.3)

        # Predict trained model output
        a = net.predict(x.reshape(batch_size, *input_shape))
        print(a)

