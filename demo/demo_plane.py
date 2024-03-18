import unittest

import numpy as np
from matplotlib import pyplot as plt, animation

from neuralnetwork import Network
from neuralnetwork import losses
from neuralnetwork import layers
from neuralnetwork.activations import sigmoid


def main():
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # #   T R A I N I N G   D A T A   # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # #
    # generate data #
    # # # # # # # # #
    train_samples = 1000
    input_array = np.random.uniform(low=0, high=4, size=(train_samples, 2))

    output_array = np.array([1 if (x-1)**2 + (y-3)**2 <= 0.5 or
                         (x-3)**2 + (y-1)**2 <= 0.5 or
                         (x-1)**2 + (y-1)**2 <= 0.5 or
                         (x-3)**2 + (y-3)**2 <= 0.5
                    else 0 for (x, y) in input_array ]).reshape(train_samples, -1)

    # np.savez('training_data/plane_test.npz', input_data=input_array, output_data=output_array)

    # # # # # # #
    # load data #
    # # # # # # #
    # training_data_npz = np.load('training_data/plane_test.npz')
    #
    # training_data = (
    #     training_data_npz['input_data'], training_data_npz['output_data']
    # )


    # # # # # # # # # #
    # visualize  data #
    # # # # # # # # # #
    for (x, y), out in zip(input_array, output_array):
        if out:
            plt.scatter(x, y, s=3, c='y')
        else:
            plt.scatter(x, y, s=3, c='b')

    ax_1 = plt.gca()
    ax_1.set_aspect('equal', adjustable='box')
    plt.title('TRAINING DATA')
    plt.xlabel('input 1')
    plt.xlabel('input 2')
    plt.grid()
    plt.show()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # #   N E U R A L   N E T W O R K   # # # # # # # # # # # # # # # # # # # #

    # Initialize neural network
    net = Network()

    # Input layer
    net.add_layer(layers.InputLayer((2,)))

    # Hidden layers
    net.add_layer(layers.Dense(50, activation=sigmoid))
    net.add_layer(layers.Dense(50, activation=sigmoid))
    net.add_layer(layers.Dense(50, activation=sigmoid))

    # Output layer
    net.add_layer(layers.Dense(1, activation=sigmoid))

    # Compile
    net.compile(losses.mse)

    # Train neural network
    epochs = 150
    history = np.zeros((epochs, train_samples))
    for e in range(epochs):
        net.fit(input_array, output_array, batch_size=1, epochs=1, learning_rate=0.1)
        history[e] = net.predict(input_array).reshape(-1,)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # #   V A L I D A T I O N   # # # # # # # # # # # # # # # # # # # # #
    # Loss function
    loss = net.summary['train_loss']

    plt.plot(range(epochs), loss, c='0.3')
    plt.title('training loss function')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.show()

    test_array = np.random.uniform(low=0, high=4, size=(2000, 2))
    prediction = net.predict(test_array)
    for v, (x, y) in zip(prediction, test_array):
        if v >= 0.5:
            plt.scatter(x, y, s=3, c='y', alpha=abs(2 * v - 1))
        else:
            plt.scatter(x, y, s=3, c='b', alpha=abs(2 * v - 1))

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.title('TEST DATA')
    plt.xlabel('input 1')
    plt.xlabel('input 2')
    plt.grid()
    plt.show()


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # #   P R O C E S S   V I S U A L I Z A T I O N   # # # # # # # # # # # # # # # #

    colours = np.empty((len(history), len(history[0]), 2), dtype=object)
    for epoch in range(epochs):
        for n, v in enumerate(history[epoch]):
            colours[epoch][n][0] = abs(2 * v - 1)
            if v >= 0.5:
                colours[epoch][n][1] = 'y'
            else:
                colours[epoch][n][1] = 'b'

    # ANIMATED PLOT
    fig, (ax_1, ax_2) = plt.subplots(1, 2)

    # Plot scatter
    ax_points = np.zeros(len(input_array), dtype=object)
    for i in range(len(input_array)):
        ax_points[i] = ax_1.scatter(input_array[i][0], input_array[i][1], s=3, c='0.5', alpha=0.8)

    epoch_template = 'epoch = %.0f'
    epoch_text = ax_1.text(0, -0.11, '', transform=ax_1.transAxes)
    ax_1.set_xlabel('x')
    ax_1.set_ylabel('y')
    ax_1.set_aspect('equal', adjustable='box')
    ax_1.grid()

    # Plot loss
    ax_2.plot(range(epochs), loss, c='0.3')
    ax_2.set_xlabel('epoch')
    ax_2.set_ylabel('loss')
    ax_2.figure.set_size_inches(20/1.8, 5/1.8)
    ax_2.grid()

    fig.set_figheight(5)
    fig.set_figwidth(10)

    # Animation function
    def update(epoch):
        for n, data in enumerate(colours[epoch]):
            ax_points[n].set_alpha(data[0])
            ax_points[n].set_color(data[1])

        ax_2.set_xlim(left=0, right=epoch+1)

        # Update text with iteration number
        epoch_text.set_text(epoch_template % (epoch + 1))

        return ax_points, epoch_text

    # Time in milliseconds per frame
    interval = 100

    # Show the plot
    anim = animation.FuncAnimation(fig=fig, func=update, frames=len(history), interval=interval, blit=False)

    # Save animation
    writergif = animation.PillowWriter(fps=len(history) / interval * 10)
    anim.save("anim3.gif", writer=writergif, dpi=120)


if __name__ == '__main__':
    main()
