from scratch.utils import *
import numpy as np

class FullyConnectedLayer:
    def __init__(self, input_size, output_size, init_type="xavier", dropout_rate=0.0):
        """
        Initializes a fully connected layer with weights, biases, and optional dropout.
        :param input_size: (int) The number of input neurons.
        :param output_size: (int) The number of output neurons.
        :param init_type: The type of weight initialization ("xavier" or "he").
        :param dropout_rate: (float) Probability of dropping a neuron during training (0.0 to 1.0).
        """
        self.type = 'fc'
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        # Initialize weights and biases
        self.weights = self.initialize_weights((input_size, output_size), init_type)
        self.biases = np.zeros(output_size)

        # ReLU activation function
        self.relu = ReLU()

        self.input = None  # Placeholder for the input during forward pass
        self.activations = None  # Placeholder for the activations
        self.dropout_mask = None  # Placeholder for the dropout mask

    def get_in_size(self):
        return self.input_size

    @staticmethod
    def initialize_weights(size, init_type="xavier"):
        """
        Initializes weights using the specified initialization strategy.
        """
        in_size, out_size = size[1], size[0]  # Get the input and output sizes from the shape tuple
        if init_type == "xavier":
            stddev = np.sqrt(2 / (in_size + out_size))
            return np.random.normal(0, stddev, size=size)
        elif init_type == "he":
            return np.random.randn(*size) * np.sqrt(2.0 / in_size)
        else:
            return np.random.randn(*size) * 0.01

    def forward(self, x, training=True):
        """
        Forward pass through the fully connected layer, with optional dropout.
        :param x: (np.ndarray) The input data (batch_size, input_size).
        :param training: (bool) Whether the layer is in training mode.
        :return: np.ndarray: The activations after applying weights, biases, activation, and dropout.
        """
        self.input = x
        self.activations = np.dot(x, self.weights) + self.biases
        self.activations = self.relu.forward(self.activations)

        if training and self.dropout_rate > 0.0:
            # Create dropout mask: 1 for keep, 0 for drop
            self.dropout_mask = (np.random.rand(*self.activations.shape) > self.dropout_rate).astype(np.float32)
            self.activations *= self.dropout_mask  # Apply the dropout mask
            self.activations /= (1 - self.dropout_rate)  # Scale the activations to maintain expected values

        return self.activations

    def backward(self, da, learning_rate=0.001):
        """
        Backward pass through the fully connected layer to update weights and biases, with dropout adjustment.
        """
        num_samples = self.input.shape[0]

        # Apply dropout mask to gradients if dropout was used during forward pass
        if self.dropout_rate > 0.0 and self.dropout_mask is not None:
            da *= self.dropout_mask
            da /= (1 - self.dropout_rate)

        dz = da * self.relu.backward(self.activations)
        dw = np.dot(self.input.T, dz) / num_samples
        db = np.sum(dz, axis=0, keepdims=True) / num_samples

        da_prev = np.dot(dz, self.weights.T)

        # Update weights and biases
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db.reshape(self.biases.shape)

        return da_prev
