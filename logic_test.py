import matplotlib.pyplot as plt
import numpy as np

from neuralnetwork.feedforwardneuralnetwork import FeedforwardNeuralNetwork


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #   T R A I N I N G   D A T A S   # # # # # # # # # # # # # # # # # # # # #


training_data = (
    [
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ] , [
        [0, 0],
        [0, 1],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 0]
    ]
)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #   N E U R A L   N E T W O R K   # # # # # # # # # # # # # # # # # # # # #
ann = FeedforwardNeuralNetwork()

# Input layer
ann.add_layer(3, activation_function='')

# Hidden layers
# ann.add_layer(5, activation_function='sigmoid')
# ann.add_layer(5, activation_function='sigmoid')

# Output layer
ann.add_layer(2, activation_function='sigmoid')


# Train neural network with given parameters
ann.train(training_data,
          epochs=500,
          learning_ratio=0.8,
          plot_cost=True)

ann.load_network('logic_test')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # #   V E R I F I C A T I O N   # # # # # # # # # # # # # # # # # # # # # #
input_array = [1, 0, 0]

print(ann.predict_output(input_array))
print('Neural network solution: ', np.round(ann.predict_output(input_array), 2))
