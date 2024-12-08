from sklearn.model_selection import KFold
from nn.scratch.FullyConnectedLayer import FullyConnectedLayer
from nn.scratch.utils import *
from baselines.cifar10 import Cifar10
import time, json


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

    def train(self, x_train, y_train, x_test, y_test, epochs=100, learning_rate=0.001, k_folds=1):
        """
        Train the fully connected network.
        :param x_train: Training data.
        :param y_train: Training labels (one-hot encoded).
        :param x_test: Test data.
        :param y_test: Test labels (one-hot encoded).
        :param epochs: Number of epochs to train.
        :param learning_rate: Learning rate for optimization.
        :param k_folds: Number of folds for cross-validation. Default is 1 (no cross-validation).
        """
        results = {"epochs": [], "test_metrics": [], "fold_results": []}

        if k_folds > 1:
            results["epochs"], results["fold_results"] = self._perform_k_fold_training(
                x_train, y_train, epochs, learning_rate, k_folds
            )
        else:
            results["epochs"] = self._train_on_data(x_train, y_train, epochs, learning_rate)
        results["test_metrics"] = self.evaluate(x_test, y_test)

        # Save results to JSON file
        self._save_results(results, "training_results.json")

    def _perform_k_fold_training(self, x_train, y_train, epochs, learning_rate, k_folds):
        """
        Perform k-fold cross-validation training.
        """
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results = []
        fold_epochs = []

        for fold, (train_index, val_index) in enumerate(kf.split(x_train)):
            print(f"Starting Fold {fold + 1}/{k_folds}")
            x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            # Train on the current fold
            fold_epochs.append(self._train_on_data(x_train_fold, y_train_fold, epochs, learning_rate))

            # Validate on the current fold
            val_metrics = self.evaluate(x_val_fold, y_val_fold)
            fold_results.append(val_metrics)

        return fold_epochs, fold_results

    def _train_on_data(self, x_train, y_train, epochs, learning_rate):
        """
        Train the network on the provided data for a set number of epochs.
        """
        epoch_results = []

        for epoch in range(epochs):
            start_time = time.time()
            activations = self._forward_pass(x_train)

            # Calculate loss
            loss = cross_entropy_loss(activations[-1], y_train)

            # Backward pass
            self._backward_pass(activations, y_train, learning_rate)

            end_time = time.time()
            print(f"Epoch {epoch} loss: {loss} time: {end_time - start_time:.2f}s")

            epoch_results.append({"epoch": epoch, "loss": loss, "time": end_time - start_time})

        return epoch_results

    def _forward_pass(self, x):
        """
        Perform a forward pass through all layers.
        """
        activations = [x]
        for layer in self.layers:
            z = layer.forward(activations[-1])
            activations.append(z)
        activations.append(self.softmax.forward(activations[-1]))
        return activations

    def _backward_pass(self, activations, y_train, learning_rate):
        """
        Perform a backward pass through all layers and update parameters.
        """
        da = activations[-1] - y_train  # Gradient for the output layer
        for layer in reversed(self.layers):
            da = layer.backward(da, learning_rate)

    def _save_results(self, results, filename):
        """
        Save training results to a JSON file.
        """
        with open(filename, "w") as f:
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
    layer_config = [1024, 512, 254]
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

    nn.train(x_train, y_train, x_test, y_test, epochs=30, learning_rate=0.0001, k_folds=5)


if __name__ == '__main__':
    main()