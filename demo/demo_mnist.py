import numpy as np

from PIL import Image

from neuralnetwork.net import Network
from neuralnetwork.layers import InputLayer, Convolutional, MaxPooling, Flatten, Dense
from neuralnetwork.activations import relu


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #   T R A I N I N G   D A T A   # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # #
# load  &&  process #
# # # # # # # # # # #
train_size = 1000
mnist = np.load('../datasets/mnist.npz')

x_train = mnist['x_train'][:train_size].reshape(train_size, 28, 28, 1) / 255
y_train_idx = mnist['y_train'][:train_size]
y_train = np.zeros((y_train_idx.size, y_train_idx.max() + 1))
y_train[np.arange(y_train_idx.size), y_train_idx] = 1

training_data = (
    x_train, y_train
)


# # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # TEST DATA # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #   N E U R A L   N E T W O R K   # # # # # # # # # # # # # # # # # # # # #
# Parameters
batch_size = 1
epochs = 10
learning_rate = 0.1

input_shape = (batch_size, 28, 28, 1)

cnn = Network()

cnn.add_layer(InputLayer(input_shape=input_shape))
cnn.add_layer(Convolutional(1, (2, 2), activation=relu, padding='same'))
cnn.add_layer(Convolutional(1, (2, 2), activation=relu, padding='same'))



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # #   V E R I F I C A T I O N   # # # # # # # # # # # # # # # # # # # # # #

