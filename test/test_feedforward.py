import unittest

import numpy as np

import neuralnetwork.losses
from neuralnetwork import ANN
from neuralnetwork import losses
from neuralnetwork.layers import InputLayer, Dense
from neuralnetwork.activations import sigmoid, linear
import keras


class TestFeedforward(unittest.TestCase):

    def setUp(self):
        # Define input shape
        input_shape = (4,)

        # Define weights
        weights_layer_1 = np.array([
            [0.5, 0.5, 0.5, 0.5],       # Neuron #1
            [0.25, 0.25, 0.25, 0.25],   # Neuron #2
            [0.35, 0.35, 0.35, 0.35]    # Neuron #3
        ])

        weights_layer_2 = np.array([
            [0.5, 0.5, 0.5],            # Neuron #1
            [0.25, 0.25, 0.25],         # Neuron #2
            [0.35, 0.35, 0.35]          # Neuron #3
        ])

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Build model
        self.net = ANN()
        self.net.add_layer(InputLayer(input_shape=input_shape))
        self.net.add_layer(Dense(3, activation=sigmoid))
        self.net.add_layer(Dense(3, activation=sigmoid))
        self.net.compile(neuralnetwork.losses.mse)

        # Assign weights
        self.net.layers[1].w = weights_layer_1
        self.net.layers[1].b = 0
        self.net.layers[2].w = weights_layer_2
        self.net.layers[2].b = 0

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Build reference model
        self.kerasnet = keras.models.Sequential()
        self.kerasnet.add(keras.layers.InputLayer(input_shape=(None, input_shape[0])))
        self.kerasnet.add(keras.layers.Dense(3, activation='sigmoid'))
        self.kerasnet.add(keras.layers.Dense(3, activation='sigmoid'))
        self.kerasnet.compile()

        # Assign weights
        self.kerasnet.layers[0].set_weights([np.transpose(weights_layer_1), np.array([0, 0, 0])])
        self.kerasnet.layers[1].set_weights([np.transpose(weights_layer_2), np.array([0, 0, 0])])


    def test_feedforward(self):
        # Define input
        x = np.array([-19, 4, 10, 0.11])

        # Predict
        keras_prediction = self.kerasnet.predict(x.reshape(1, -1))
        net_prediction = self.net.predict(x).reshape(1, -1)
        # Test
        np.testing.assert_almost_equal(np.float32(net_prediction), np.float32(keras_prediction))