import numpy as np
import math, json, time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from cnn.ConvLayer import ConvLayer
from cnn.FullyConnectedLayer import FullyConnectedLayer
from cnn.utils import *
from baselines.cifar10 import Cifar10, unpickle

class ConvNetwork:
    def __init__(self, input_size, num_classes, layer_config):
        self.layers = []
        self.layer_config = layer_config  # (num_filters, kernel_size, stride, padding)
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

    def train(self, x_train, y_train, x_test, y_test, epochs=10, learning_rate=0.01):
        num_batches = len(x_train) // self.batch_size
        x_batches = np.array_split(x_train[:num_batches * self.batch_size], num_batches)
        y_batches = np.array_split(y_train[:num_batches * self.batch_size], num_batches)

        results = {"epochs": []}

        with open('output_cnn.txt', 'w') as tf:
            tf.write('Convolution Training Started\n')

        for epoch in range(epochs):
            start_time = time.time()
            x_batch = x_batches[epoch % num_batches]
            y_batch = y_batches[epoch % num_batches]

            activations = [x_batch]

            # Forward pass
            for layer in self.layers:
                if layer.type == 'fc':
                    current_activations = activations[-1].reshape(activations[-1].shape[0], -1)
                else:
                    current_activations = activations[-1]
                z = layer.forward(current_activations)
                activations.append(z)

            # Apply softmax
            activations.append(self.softmax.forward(activations[-1]))

            # Calculate loss
            loss = cross_entropy_loss(activations[-1], y_batch)
            results["epochs"].append({"epoch": epoch, "loss": loss})

            # Backward pass
            da = activations[-1] - y_batch
            for layer in reversed(self.layers):
                da = layer.backward(da, learning_rate=learning_rate)

            end_time = time.time()
            results["epochs"][-1]["time"] = end_time - start_time

            with open('output_cnn.txt', 'w') as tf:
                tf.write(f"Epoch {epoch}, Loss: {loss:.4f}, Time: {end_time - start_time:.4f} seconds\n")
            #print(f"Epoch {epoch}, Loss: {loss:.4f}, Time: {end_time - start_time:.4f} seconds")

        # Evaluate on test data
        metrics = self.evaluate(x_test, y_test)
        results["test_metrics"] = metrics

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
            z = layer.forward(current_activations)
            activations.append(z)

        predictions = np.argmax(self.softmax.forward(activations[-1]), axis=1)
        true_labels = np.argmax(y, axis=1)

        acc = accuracy_score(true_labels, predictions)
        precision, recall, fscore, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted")
        conf_matrix = confusion_matrix(true_labels, predictions)

        print("Evaluation Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {fscore:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": fscore,
            "confusion_matrix": conf_matrix.tolist()
        }


def main():
    input_size = (10000, 32, 32, 3)
    num_classes = 10
    layer_config = [(32, 3, 1, 1), (32, 3, 1, 1), (64, 3, 1, 1), (64, 3, 1, 1)]
    nn = ConvNetwork(input_size, num_classes, layer_config)

    # Load the CIFAR-10 dataset
    dataset = Cifar10(normalization='min-max')

    x_train = dataset.images
    y_train = dataset.labels
    y_train = to_one_hot(y_train, 10)
    x_test = dataset.images_test
    y_test = dataset.labels_test
    y_test = to_one_hot(y_test, 10)


    nn.train(x_train, y_train, x_test, y_test, epochs=3)

if __name__ == '__main__':
    main()