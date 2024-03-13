import numpy as np

from neuralnetwork.ann import ANN


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #   T R A I N I N G   D A T A   # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # #
# generate data #
# # # # # # # # #
input_data = np.random.choice([0, 1], size=(500, 9))
input_data = np.unique(input_data, axis=0)
print(len(input_data))

def check_sum(array):
    sum_array = np.sum(array, axis=1)
    sum_digit = np.zeros(shape=(len(sum_array), len(array[0]) + 1))

    for i, value in enumerate(sum_array):
        for j in range(len(sum_digit[i])):
            if j == value:
                sum_digit[i][j] = 1
                continue

    return sum_digit

np.savez('training_data/sum_test', input_data=input_data, output_data=check_sum(input_data))

# # # # # # #
# load data #
# # # # # # #
training_data_npz = np.load('training_data/sum_test.npz')

training_data = (
    training_data_npz['input_data'], training_data_npz['output_data']
)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #   N E U R A L   N E T W O R K   # # # # # # # # # # # # # # # # # # # # #
ann = ANN()

# Input layer
ann.add_layer(9, activation_function='')

# Hidden layers
ann.add_layer(50, activation_function='sigmoid')

# Output layer
ann.add_layer(10, activation_function='sigmoid')

# ann.load_network('sum_test_H-50')

# Train neural network with given parameters
ann.train(training_data,
          epochs=500,
          learning_ratio=0.5,
          plot_cost=True,
          plot_accuracy=True,
          discretize_accuracy=False)

# ann.save_network('sum_test_H-50')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # #   V E R I F I C A T I O N   # # # # # # # # # # # # # # # # # # # # # #

input_array = [0, 0, 1, 0, 0, 1, 0, 0, 0]    # 4

print(ann.predict_output(input_array))
print((np.round(ann.predict_output(input_array), 2)))
print('Neural network solution: ', np.argmax(ann.predict_output(input_array)))
