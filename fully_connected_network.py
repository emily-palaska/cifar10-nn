from torch.nn.functional import dropout

from scratch.FullyConnectedLayer import FullyConnectedLayer
from scratch.utils import *
from baselines.cifar10 import Cifar10
import time,json

class FullyConnectedNetwork:
    def __init__(self, input_size, num_classes, layer_config, init_type='he', dropout_rate=0.0):
        self.layers = []
        self.layer_config = layer_config
        self.softmax = Softmax()
        self.dropout_rate = dropout_rate

        # Initialize layers according to given configuration
        neurons_in = input_size
        for neurons_out in layer_config:
            layer = FullyConnectedLayer(neurons_in, neurons_out, init_type, dropout_rate)
            self.layers.append(layer)
            neurons_in = neurons_out

        # Add one final layer for the output
        layer = FullyConnectedLayer(neurons_in, num_classes)
        self.layers.append(layer)


    def train(self, x_train, y_train, x_test, y_test, epochs=100, learning_rate=0.001):
        # x is batch_size, input
        results = {"epochs": []}

        for epoch in range(epochs):
            start_time = time.time()
            activations = [x_train]
            # forward pass layers
            for layer in self.layers:
                z = layer.forward(activations[-1])
                activations.append(z)
            # apply softmax
            activations.append(self.softmax.forward(activations[-1]))

            # calculate loss
            loss = cross_entropy_loss(activations[-1], y_train)

            # back propagation
            da = activations[-1] - y_train  # Gradient for output layer
            for layer in reversed(self.layers):
                da = layer.backward(da, learning_rate)

            end_time = time.time()
            if epoch % 1 == 0:
                print(f"Epoch {epoch} loss: {loss} time: {end_time - start_time: .2f}s")

            results["epochs"].append({"epoch": epoch, "loss": loss})
            results["epochs"][-1]["time"] = end_time - start_time


        # Evaluate on test data
        test_metrics = self.evaluate(x_test, y_test)
        results["test_metrics"] = test_metrics

        # Save results to JSON file
        with open("training_results.json", "w") as f:
            json.dump(results, f, indent=4)

    def evaluate(self, x, y):
        activations = [x]
        for layer in self.layers:
            if layer.type == 'fc':
                current_activations = activations[-1].reshape(activations[-1].shape[0], -1)
            else:
                current_activations = activations[-1]
            z = layer.forward(current_activations, training=False)
            activations.append(z)

        predictions = np.argmax(self.softmax.forward(activations[-1]), axis=1)
        true_labels = np.argmax(y, axis=1)

        return metrics(predictions, true_labels)

def main():
    input_size = 32 * 32 * 3
    num_classes = 10
    layer_config = [2048, 1024, 512, 256, 128]
    nn = FullyConnectedNetwork(input_size, num_classes, layer_config, init_type='xavier', dropout_rate=0.5)

    # Load the CIFAR-10 dataset
    dataset = Cifar10(normalization='min-max')

    x_train = dataset.images
    x_train = x_train.reshape((x_train.shape[0], input_size))
    y_train = dataset.labels
    y_train = to_one_hot(y_train, 10)
    x_test = dataset.images_test
    x_test = x_test.reshape((x_test.shape[0], input_size))
    y_test = dataset.labels_test
    y_test = to_one_hot(y_test, 10)


    nn.train(x_train, y_train, x_test, y_test, epochs=30, learning_rate=0.001)

if __name__ == '__main__':
    main()