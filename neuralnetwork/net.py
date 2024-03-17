import json
import sys
from typing import List, Callable

import matplotlib.pyplot as plt
import numpy as np

from .layers import Layer
from neuralnetwork import losses
from neuralnetwork import activations


class Network:

    def __init__(self):
        # INITIALIZE LAYERS
        self.layers: List[Layer] = []
        self.loss = None

        self.summary = {
            'train_loss': []
        }


    def add_layer(self, layer: Layer) -> None:
        """
        Add new layer consisting of given number of neurons and their activation function.
        """
        self.layers.append(layer)


    def fit(self, x, y, batch_size: int, epochs: int , learning_rate: float):
        """
        Train the Neural Network.
        """
        # Prepare data divided on mini batches
        mini_batches = [[x[i:i+batch_size], y[i:i+batch_size]] for i in range(0, len(x), batch_size)]
        for i in range(epochs):
            # Apply SGD for averaged data from mini batch
            for xs, ys in mini_batches:
                # Train
                a = self.predict(xs)
                self.sgd(ys, learning_rate)

                loss = self.loss(a, ys)

            # Update summary
            self.summary['train_loss'].append(np.mean(loss))

            # Progress bar
            sys.stdout.write('\r')
            sys.stdout.write("Training in progress: [%-20s] %d%%" % ('=' * int(i/epochs*20), int(i/epochs*100)))
            sys.stdout.flush()


    def __build_layers(self):
        for i in range(1, len(self.layers[1:])+1):
            self.layers[i].compile(self.layers[i - 1].a)


    def compile(self, loss: Callable):
        self.loss = loss
        self.__build_layers()


    def feedforward(self, x):
        self.layers[0].a = x
        for i in range(1, len(self.layers[1:])+1):
            self.layers[i].feedforward(self.layers[i-1].a)


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict output of a neural network for a given input
        """
        self.feedforward(x)
        return self.layers[-1].a


    def backpropagate(self, y):
        """
        Calculate the Cost Function derivative over each weight and bias in order to determine its gradient.
        """
        # Initialize Loss Function gradient with respect to the weights matrix
        grad_w = [np.zeros_like(layer.w) for layer in self.layers[1:]]

        # Initialize Loss Function gradient with respect to the biases matrix
        grad_b = [np.zeros_like(layer.b) for layer in self.layers[1:]]

        # Output Layer
        delta = losses.d(self.loss, self.layers[-1].a, y) * activations.d(self.layers[-1].activation, self.layers[-1].z)
        grad_b[-1] = delta
        grad_w[-1] = [np.outer(d, a) for d, a in zip(delta, self.layers[-1].a)]

        # Hidden Layers
        for l in range(-2, -len(self.layers), -1):
            delta = np.dot(delta, self.layers[l+1].w) * activations.d(self.layers[l].activation, self.layers[l].z)
            grad_b[l] = delta
            grad_w[l] = [np.outer(d, a) for d, a in zip(delta, self.layers[l-1].a)]

        return grad_w, grad_b


    def sgd(self, y: np.ndarray, eta):
        """
        Update the weights and biases using backpropagation and Stochastic Gradient Descent.
        """
        # Calculate the Cost Function gradient with respect to the weights and biases
        grad_w, grad_b = self.backpropagate(y)

        # Compute new weights and biases using Stochastic Gradient Descent
        for l in range(len(self.layers[1:])):
            self.layers[l+1].w -= eta * np.mean(grad_w[l], axis=0)
            self.layers[l+1].b -= eta * np.mean(grad_b[l])


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # #   S A V E   &   L O A D   # # # # # # # # # # # # # # # # # # # # # #
    def save_network(self, filename, filepath = 'saved/'):
        """
        Save current Neural Network's architecture.
        """
        network = {
            'layers': [(layer.n, layer.activation.__name__) for layer in self.layers],
            'weights': [layer.get_weights().tolist() for layer in self.layers[1:]],
            'biases': [layer.get_biases().tolist() for layer in self.layers[1:]]
        }

        file = open('{}{}.json'.format(filepath, filename), 'w')
        json.dump(network, file)
        file.close()

    def load_network(self, filename, filepath = 'saved/'):
        """
        Load and overwrite Neural Network from a given file.
        """
        file = open('{}{}.json'.format(filepath, filename), 'r')
        network = json.load(file)
        file.close()

        layers = network['layers']

        weights = network['weights']
        biases = network['biases']

        self.layers.clear()
        for size, activation_function in layers:
            self.add_layer(size, activation_function)


        for layer, weights_l, biases_l in zip(self.layers[1:], weights, biases):
            for neuron, weight, bias in zip(layer.n, weights_l, biases_l):
                neuron.w = weight
                neuron.b = bias



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # #   V I S U A L I Z A T I O N   # # # # # # # # # # # # # # # # # # # # #
    def visualize(self):
        pass


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # #   P R I V A T E   F U N C T I O N S   # # # # # # # # # # # # # # # # # # #


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # #   O V E R W R I T T E N   F U N C T I O N S   # # # # # # # # # # # # # # # # #
    def __str__(self) -> str:
        return 'ANN: [Layers: {}]'.format([str(layer) for layer in self.layers])


    def __repr__(self):
        return self.__str__()
