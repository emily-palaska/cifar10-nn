import numpy as np
import math

from cnn.ConvLayer import ConvLayer
from cnn.FullyConnectedLayer import FullyConnectedLayer
from cnn.utils import *
from baselines.cifar10 import Cifar10, unpickle

class ConvNetwork:
    def __init__(self, input_size, num_classes, layer_config):
        self.layers = []
        self.layer_config = layer_config # (num_filters, kernel_size, stride, padding)
        self.softmax = Softmax()
        self.batch_size, self.height, self.width, self.input_chan = input_size
        self.num_classes = num_classes

        # Initialize layers according to given configuration
        current_size = input_size
        for num_filters, kernel_size, stride, padding in layer_config:
            layer = ConvLayer(num_filters, kernel_size, current_size, stride, padding)
            self.layers.append(layer)
            current_size = layer.get_out_size()

        # Add one final layer for the output
        flattened_size = math.prod(current_size[1:])
        layer = FullyConnectedLayer(flattened_size, num_classes)
        self.layers.append(layer)

    def train(self, x_train, y_train, epochs=1000, learning_rate=0.01):
        # x is num_samples, height, width, channels

        # Split into unique batches
        num_batches = len(x_train) // self.batch_size
        x_batches = np.array_split(x_train[:num_batches * self.batch_size], num_batches)
        y_batches = np.array_split(y_train[:num_batches * self.batch_size], num_batches)


        for epoch in range(epochs):
            x_batch = x_batches[epoch]
            y_batch = y_batches[epoch]

            activations = [x_batch]

            # forward pass layers
            for layer in self.layers:
                if layer.type == 'fc':
                    current_activations = activations[-1].reshape(activations[-1].shape[0], -1)
                else:
                    current_activations = activations[-1]
                z = layer.forward(current_activations)
                activations.append(z)

            # apply softmax
            activations.append(self.softmax.forward(activations[-1]))

            # calculate loss
            loss = cross_entropy_loss(activations[-1], y_batch)
            if epoch % 1 == 0:
                print(f"Epoch {epoch} loss: {loss}")

            # back propagation
            da = activations[-1] - y_batch  # Gradient for output layer
            for layer in reversed(self.layers):
                da = layer.backward(da, learning_rate=learning_rate)

def main():
    input_size = (100, 32, 32, 3)
    num_classes = 10
    layer_config = [(8, 3, 1, 1), (8, 3, 1, 1)]
    nn = ConvNetwork(input_size, num_classes, layer_config)

    # Load the CIFAR-10 dataset
    dataset = Cifar10(normalization='min-max')

    # Prepare the dataset
    x_train = dataset.images
    y_train = dataset.labels
    y_train = to_one_hot(y_train, 10)

    """
    # Load the test batch and normalize it with min-max
    test_batch = unpickle('./data/cifar-10-batches-py/test_batch')
    X_test = np.array(test_batch[b'data'])
    X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test))
    y_test = np.array(test_batch[b'labels'])
    y_test = to_one_hot(y_test, 10)
    """

    nn.train(x_train, y_train)

if __name__ == '__main__':
    main()