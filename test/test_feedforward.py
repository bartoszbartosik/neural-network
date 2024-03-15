import unittest

import numpy as np

from neuralnetwork import ANN
from neuralnetwork.layers import InputLayer, Dense
from neuralnetwork.activations import sigmoid, linear
import keras


class TestFeedforward(unittest.TestCase):

    def setUp(self):

        # Input data
        self.x = np.array([1, 0, 1, 0])

        # Define shape
        input_shape = self.x.shape

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

        # Build model
        self.net = ANN()
        self.net.add_layer(InputLayer(input_shape=input_shape))
        self.net.add_layer(Dense(3, activation=linear))
        self.net.add_layer(Dense(3, activation=linear))
        self.net.compile()

        # Assign weights
        self.net.layers[1].weights = weights_layer_1
        self.net.layers[1].bias = 0
        self.net.layers[2].weights = weights_layer_2
        self.net.layers[2].bias = 0


        # Build reference model
        self.kerasnet = keras.models.Sequential()
        self.kerasnet.add(keras.layers.InputLayer(input_shape=(None, input_shape[0])))
        self.kerasnet.add(keras.layers.Dense(3, activation='linear'))
        self.kerasnet.add(keras.layers.Dense(3, activation='linear'))
        self.kerasnet.compile()

        # Assign weights
        self.kerasnet.layers[0].set_weights([np.transpose(weights_layer_1), np.array([0, 0, 0])])
        self.kerasnet.layers[1].set_weights([np.transpose(weights_layer_2), np.array([0, 0, 0])])



    def test_feedforward(self):
        # Predict
        keras_prediction = self.kerasnet.predict(self.x.reshape(1, -1))
        ann_prediction = self.net.predict(self.x).reshape(1, -1)
        # Test
        np.testing.assert_almost_equal(ann_prediction, keras_prediction)
