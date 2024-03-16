import numpy as np

from PIL import Image

from neuralnetwork.ann import ANN


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #   T R A I N I N G   D A T A   # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # #
# load  &&  process #
# # # # # # # # # # #
training_data_npz = np.load('training_data/mnist.npz')

input_data = [x.flatten()/255 for x in training_data_npz['x_test']]
print(len(input_data))

output_data = []
for value in training_data_npz['y_test']:
    out = np.zeros(10)
    out[value] = 1
    output_data.append(out)

training_data = (
    input_data, output_data
)


# # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # TEST DATA # # # # # # # # # # #
# From MNIST
test_input = [x.flatten()/255 for x in training_data_npz['x_train']]

test_output = []
for value in training_data_npz['y_train']:
    out = np.zeros(10)
    out[value] = 1
    test_output.append(out)

test_data = (
    test_input, test_output
)

# Own data
test_input_own = []
for i in range(10):
    img_file = Image.open('training_data/mnist_digits/{}_test.png'.format(i)).convert('L')
    img_np = np.asarray(img_file)
    img_np = np.array([arr.flatten()/255 for arr in img_np]).flatten()
    test_input_own.append(img_np)

test_output_own = np.identity(10)

test_own = (
    test_input_own, test_output_own
)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #   N E U R A L   N E T W O R K   # # # # # # # # # # # # # # # # # # # # #
ann = ANN()

# Input layer
ann.add_layer(784, activation_function='')

# Hidden layers
ann.add_layer(300, activation_function='sigmoid')
ann.add_layer(100, activation_function='sigmoid')

# Output layer
ann.add_layer(10, activation_function='sigmoid')


ann.load_network('mnist-300-100')

# Train neural network with given parameters
ann.fit(training_data,
        epochs=5,
        learning_ratio=0.1,
        plot_cost=True,
        plot_accuracy=True,
        discretize_accuracy=False)

ann.save_network('mnist-300-100')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # #   V E R I F I C A T I O N   # # # # # # # # # # # # # # # # # # # # # #

for i in range(10):
    input_array = test_own[0][i]
    print((np.round(ann.predict(input_array), 2)))
    print('Neural network solution: ', np.argmax(ann.predict(input_array)))
    print('Expected outcome: ', np.argmax(test_own[1][i]))

print('Accuracy for own test data: {}'.format(ann.accuracy(test_own, discretize=False)))
print('Accuracy for test data: {}'.format(ann.accuracy(test_data, discretize=False)))

