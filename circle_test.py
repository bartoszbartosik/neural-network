import matplotlib.pyplot as plt
import numpy as np

from neuralnetwork.ann import ANN


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #   T R A I N I N G   D A T A   # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # #
# generate data #
# # # # # # # # #
input_array = np.random.uniform(low=0, high=4, size=(1000, 2))

output_array = [1 if (x-1)**2 + (y-3)**2 <= 0.5 or
                     (x-3)**2 + (y-1)**2 <= 0.5 or
                     (x-1)**2 + (y-1)**2 <= 0.5 or
                     (x-3)**2 + (y-3)**2 <= 0.5
                else 0 for (x, y) in input_array ]

np.savez('training_data/plane_test.npz', input_data=input_array, output_data=output_array)

# # # # # # #
# load data #
# # # # # # #
training_data_npz = np.load('training_data/plane_test.npz')

training_data = (
    training_data_npz['input_data'], training_data_npz['output_data']
)


# # # # # # # # # #
# visualize  data #
# # # # # # # # # #
for (x, y), out in zip(input_array, output_array):
    if out:
        plt.scatter(x, y, s=3, c='y')
    else:
        plt.scatter(x, y, s=3, c='b')

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.title('TRAINING DATA')
plt.xlabel('input 1')
plt.xlabel('input 2')
plt.grid()
plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #   N E U R A L   N E T W O R K   # # # # # # # # # # # # # # # # # # # # #

# Initialize neural network
ann = ANN()

# Input layer
ann.add_layer(2, activation_function='')

# Hidden layers
ann.add_layer(20, activation_function='sigmoid')
ann.add_layer(20, activation_function='sigmoid')

# Output layer
ann.add_layer(1, activation_function='sigmoid')

# Load network
ann.load_network('circle_test_H20-20')

# Train neural network with given parameters
# ann.train(training_data,
#           epochs=500,
#           learning_ratio=0.01,
#           plot_cost=True,
#           plot_accuracy=True,
#           discretize_accuracy=True)

# ann.save_network('circle_test_H20-20')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # #   V E R I F I C A T I O N   # # # # # # # # # # # # # # # # # # # # # #

test_array = np.random.uniform(low=0, high=4, size=(2000, 2))

for (x, y) in test_array:
    predicted_output = ann.predict_output([x, y])
    if predicted_output >= 0.5:
        plt.scatter(x, y, s=3, c='y', alpha=abs(2*predicted_output-1))
    else:
        plt.scatter(x, y, s=3, c='b', alpha=abs(2*predicted_output-1))

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.title('TEST DATA')
plt.xlabel('input 1')
plt.xlabel('input 2')
plt.grid()
plt.show()
