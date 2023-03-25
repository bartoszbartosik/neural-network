import matplotlib.pyplot as plt
import numpy as np

from feedforwardneuralnetwork.neuralnetwork import NeuralNetwork



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #   T R A I N I N G   D A T A S   # # # # # # # # # # # # # # # # # # # # #


# training_data = (
#     [
#         [1, 1, 0],
#         [1, 0, 1],
#         [1, 1, 1],
#         [0, 0, 0],
#         [0, 0, 1],
#         [0, 1, 0]
#     ] , [
#         [0, 0],
#         [0, 1],
#         [0, 0],
#         [1, 1],
#         [1, 1],
#         [1, 0]
#     ]
# )
#
# input_data = [1, 0, 0]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

input_array = np.random.uniform(low=0, high=4, size=(1000, 2))

print(np.array2string(input_array, separator=",").replace('[', '(').replace('],', ')').replace(']]', ')'))

output_array = [1 if (x-2)**2 + (y-2)**2 <= 2 else 0 for (x, y) in input_array ]

training_data = (
    input_array, output_array
)


for (x, y) in input_array:
    if (x - 2) ** 2 + (y - 2) ** 2 <= 2:
        plt.scatter(x, y, s=3, c='y')
    else:
        plt.scatter(x, y, s=3, c='b')

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.title('TRAINING DATA')
plt.xlabel('input 1')
plt.xlabel('input 2')
plt.grid()
plt.draw()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #   N E U R A L   N E T W O R K   # # # # # # # # # # # # # # # # # # # # #
ann = NeuralNetwork()

# Input layer
ann.add_layer(2, activation_function='')

# Hidden layers
ann.add_layer(20, activation_function='sigmoid')

# Output layer
ann.add_layer(1, activation_function='sigmoid')

# Train neural network with given parameters
ann.train(training_data,
          epochs=500,
          learning_ratio=0.05,
          plot_cost=True,
          plot_accuracy=True,
          direct_accuracy=True)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # #   V E R I F I C A T I O N   # # # # # # # # # # # # # # # # # # # # # #

test_array = np.random.uniform(low=0, high=4, size=(2000, 2))

for (x, y) in test_array:
    predicted_output = ann.predicted_output([x, y])
    if predicted_output >= 0.5:
        plt.scatter(x, y, s=3, c='y', alpha=abs(2*predicted_output-1))
    else:
        plt.scatter(x, y, s=3, c='b', alpha=abs(*predicted_output-1))

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.title('TEST NETWORK DATA')
plt.xlabel('input 1')
plt.xlabel('input 2')
plt.grid()
plt.show()
