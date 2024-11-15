import numpy as np
from cnn.utils import ReLU


class FullyConnectedLayer:
    def __init__(self, input_size, output_size, init_type="xavier"):
        """
        Initializes a fully connected layer with weights and biases.
        :param input_size: (int) The number of input neurons.
        :param output_size: (int) The number of output neurons.
        :param init_type: The type of weight initialization ("xavier" or "he").
        """
        self.type = 'fc'
        self.input_size = input_size
        self.output_size = output_size

        # Initialize weights randomly and biases as zeros
        self.weights = self.initialize_weights((input_size, output_size), init_type)
        self.biases = np.zeros(output_size)

        # ReLU activation function
        self.relu = ReLU()

        self.input = None  # Placeholder for the input during forward pass
        self.activations = None  # Placeholder for the activations

    def get_in_size(self):
        return self.input_size

    @staticmethod
    def initialize_weights(size, init_type="xavier"):
        """
        Initializes weights using the specified initialization strategy.
        :param size: (tuple) The shape of the weight matrix (input_size, output_size).
        :param init_type: (str) The weight initialization method ("xavier" or "he").
        :return: np.ndarray: Initialized weights.
        """

        in_size, out_size = size[1], size[0]  # Get the input and output sizes from the shape tuple
        if init_type == "xavier":
            # Xavier initialization: scales the weights to keep the variance the same across layers
            stddev = np.sqrt(2 / (in_size + out_size))  # Standard deviation for Xavier initialization
            return np.random.normal(0, stddev, size=size)  # Return normally distributed weights
        elif init_type == "he":
            # He initialization: scales weights for ReLU activation
            return np.random.randn(*size) * np.sqrt(
                2.0 / in_size)  # Return weights with standard deviation scaled for He
        else:
            # Default small random initialization
            return np.random.randn(size) * 0.01

    def forward(self, x):
        """
        Forward pass through the fully connected layer.
        :param x: (np.ndarray) The input data (batch_size, input_size).
        :return: np.ndarray: The activations after applying weights, biases, and activation function.
        """

        self.input = x  # Store the input for use in the backward pass
        self.activations = np.dot(x, self.weights) + self.biases  # Compute the linear transformation
        self.activations = self.relu.forward(self.activations)  # Apply the ReLU activation function
        return self.activations  # Return the activations after ReLU

    def backward(self, da, learning_rate=0.01):
        """
        Backward pass through the fully connected layer to update weights and biases.
        :param da: (np.ndarray) The gradient of the loss with respect to the output activations.
        :param learning_rate: (float) The learning rate for gradient descent.
        :return: np.ndarray: The gradient of the loss with respect to the input, for the previous layer.
        """
        num_samples = self.input.shape[0]
        dz = da * self.relu.backward(self.activations)  # Compute the gradient of the loss w.r.t the activations
        dw = np.dot(self.input.T, dz) / num_samples  # Compute the gradient w.r.t the weights
        db = np.sum(dz, axis=0, keepdims=True) / num_samples  # Compute the gradient w.r.t the biases
        da = np.dot(dz, self.weights.T)  # Compute the gradient w.r.t the input of the previous layer

        # Update weights and biases using gradient descent
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db.reshape(self.biases.shape)

        return da  # Return the gradient w.r.t the input, for the previous layer


def main():
    # Initialize the FullyConnected layer
    num_samples = 100
    input_size = 32*32*3
    output_size = 10  # arbitrary number of output neurons
    layer = FullyConnected(input_size, output_size)

    # Generate a random input
    x = np.random.rand(num_samples, input_size)

    # Forward pass
    output = layer.forward(x)
    print("Forward output:", output.shape)

    # Generate a random gradient for the backward pass (same shape as output)
    d_output = np.random.randn(output_size)

    # Backward pass
    d_input = layer.backward(d_output, num_samples, learning_rate=0.01)
    print("Backward output (gradient w.r.t input):", d_input.flatten())

if __name__ == "__main__":
    main()