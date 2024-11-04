import numpy as np
from cifar10 import Cifar10, unpickle

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, init_type="xavier"):
        self.layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            self.layers.append({
                "weights": self.initialize_weights(layer_sizes[i], layer_sizes[i + 1], init_type),
                "biases": np.zeros((1, layer_sizes[i + 1]))
            })

    def initialize_weights(self, input_size, output_size, init_type="xavier"):
        if init_type == "xavier":
            return np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        elif init_type == "he":
            return np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        else:
            return np.random.randn(input_size, output_size) * 0.01

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def forward_propagation(self, X):
        activations = [X]
        for layer in self.layers[:-1]:
            z = np.dot(activations[-1], layer["weights"]) + layer["biases"]
            a = self.relu(z)
            activations.append(a)
        # Output layer with softmax
        z = np.dot(activations[-1], self.layers[-1]["weights"]) + self.layers[-1]["biases"]
        a = self.softmax(z)
        activations.append(a)
        return activations

    def cross_entropy_loss(self, predictions, labels):
        return -np.mean(np.sum(labels * np.log(predictions + 1e-9), axis=1))

    def backward_propagation(self, activations, labels):
        gradients = []
        m = labels.shape[0]
        da = activations[-1] - labels  # Gradient for output layer

        for i in reversed(range(len(self.layers))):
            if i < len(self.layers) - 1:  # Hidden layers
                dz = da * (activations[i + 1] > 0)  # Apply ReLU derivative
            else:
                dz = da  # For output layer

            dw = np.dot(activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            da = np.dot(dz, self.layers[i]["weights"].T)  # Compute da for the next layer
            gradients.insert(0, {"dw": dw, "db": db})

        return gradients

    def update_parameters(self, gradients, learning_rate=0.01):
        for i, layer in enumerate(self.layers):
            layer["weights"] -= learning_rate * gradients[i]["dw"]
            layer["biases"] -= learning_rate * gradients[i]["db"]

    def train(self, X_train, y_train, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            # Forward propagation
            activations = self.forward_propagation(X_train)
            loss = self.cross_entropy_loss(activations[-1], y_train)
            # Backward propagation
            gradients = self.backward_propagation(activations, y_train)
            # Parameter update
            self.update_parameters(gradients, learning_rate)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def evaluate(self, X_test, y_test):
        activations = self.forward_propagation(X_test)
        predictions = np.argmax(activations[-1], axis=1)
        accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

def to_one_hot(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

# add save/load function



# Example Usage
input_size = 3072 # number of pixels in one image
hidden_sizes = [128, 64] # neurons per layer
output_size = 10 # number of classes
network = NeuralNetwork(input_size, hidden_sizes, output_size)

# Load the CIFAR-10 dataset
dataset = Cifar10(normalization='min-max')

# Prepare the dataset
X_train = dataset.images
y_train = dataset.labels
y_train = to_one_hot(y_train, 10)

# Load the test batch and normalize it with min-max
test_batch = unpickle('./data/cifar-10-batches-py/test_batch')
X_test = np.array(test_batch[b'data'])
X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test))
y_test = np.array(test_batch[b'labels'])
y_test = to_one_hot(y_test, 10)

network.train(X_train, y_train, epochs=1000, learning_rate=0.01)
network.evaluate(X_test, y_test)
