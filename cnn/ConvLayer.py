import numpy as np
import math
from utils import ReLU
class ConvLayer:
    def __init__(self, num_filters, kernel_size, input_channels, stride=1, padding=0):
        """
        Initialize a convolutional layer.

        :param num_filters: number of filters
        :param kernel_size: kernel size (assumed square)
        :param input_channels: channels of input
        :param stride: stride of convolution
        :param padding: padding of convolution
        """
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_channels = input_channels
        self.relu = ReLU()
        self.weights = self.initialize_weights((num_filters, kernel_size, kernel_size, input_channels))
        self.biases = np.zeros((num_filters, 1))

    @staticmethod
    def initialize_weights(size, init_type="xavier"):
        in_size, out_size = size[3], size[0]
        if init_type == "xavier":
            stddev = np.sqrt(2 / (in_size + out_size))
            return np.random.normal(0, stddev, size=size)
        elif init_type == "he":
            return np.random.randn(size) * np.sqrt(2.0 / in_size)
        else:
            return np.random.randn(size) * 0.01

    def forward(self, x):
        activations = self.convolve(x)
        activations = self.relu.forward(activations)
        return activations

    def convolve(self, x):
        # Extract dimensions
        batch_size, in_height, in_width, in_channels = x.shape
        if not in_channels == self.input_channels:
            print("Dimension error in convolution")
            exit(1)

        # Calculate output dimensions
        out_height = math.floor((in_height - self.kernel_size + 2 * self.padding) / self.stride + 1)
        out_width = math.floor((in_width - self.kernel_size + 2 * self.padding) / self.stride + 1)

        # Initialize output tensor
        output = np.zeros((batch_size, out_height, out_width, self.num_filters))

        # Pad the input (pad height and width only)
        x_padded = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')

        # Perform convolution
        for n in range(batch_size):
            for i in range(out_height):
                for j in range(out_width):
                    # Define the slice of the input to be multiplied with filters
                    h_start, h_end = i * self.stride, i * self.stride + self.kernel_size
                    w_start, w_end = j * self.stride, j * self.stride + self.kernel_size

                    # Apply convolution for each filter
                    for k in range(self.num_filters):
                        x_slice = x_padded[n, h_start:h_end, w_start:w_end, :]  # Correctly shaped 3D slice (H, W, C)
                        output[n, i, j, k] = np.sum(x_slice * self.weights[k, :, :, :]).item() + self.biases[k].item()

        return output

    def backward(self, d_out, x):
        """
        Compute the backward pass for a convolutional layer.

        Parameters:
        - d_out: Gradient of the loss with respect to the output of this layer, shape (batch_size, out_height, out_width, num_filters)
        - x: Input to the forward pass of this layer, shape (batch_size, in_height, in_width, input_channels)

        Returns:
        - d_x: Gradient of the loss with respect to the input of this layer, shape (batch_size, in_height, in_width, input_channels)
        """
        # Initialize gradients for weights and biases
        d_weights = np.zeros_like(self.weights)
        d_biases = np.zeros_like(self.biases)

        # Calculate the dimensions for d_x
        batch_size, in_height, in_width, in_channels = x.shape
        _, out_height, out_width, num_filters = d_out.shape

        # Initialize the gradient for the input with padding for the backward convolution
        d_x = np.zeros((batch_size, in_height + 2 * self.padding, in_width + 2 * self.padding, in_channels))

        # Pad the input
        x_padded = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                          mode='constant')
        d_x_padded = np.zeros_like(x_padded)

        # Loop over the batch size, height, width, and filters to compute gradients
        for n in range(batch_size):
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size

                    for k in range(num_filters):
                        # Slice the region from x_padded that contributed to the output
                        x_slice = x_padded[n, h_start:h_end, w_start:w_end, :]

                        # Accumulate the gradient for the filter weights
                        d_weights[k] += x_slice * d_out[n, i, j, k]

                        # Accumulate the gradient for the biases
                        d_biases[k] += d_out[n, i, j, k]

                        # Accumulate the gradient for the input
                        d_x_padded[n, h_start:h_end, w_start:w_end, :] += self.weights[k] * d_out[n, i, j, k]

        # Remove padding from d_x_padded to get the gradient with respect to the original input
        if self.padding > 0:
            d_x = d_x_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            d_x = d_x_padded

        # update parameters here
        return d_x, d_weights, d_biases


# Define the main function
def main():
    # Example input parameters
    batch_size = 2
    in_height = 32
    in_width = 32
    input_channels = 3
    num_filters = 1
    kernel_size = 3
    stride = 1
    padding = 1

    # Initialize a ConvLayer with the specified parameters
    conv_layer = ConvLayer(num_filters=num_filters,
                           kernel_size=kernel_size,
                           input_channels=input_channels,
                           stride=stride,
                           padding=padding)

    # Generate a random input tensor of shape (batch_size, in_height, in_width, input_channels)
    x = np.random.randn(batch_size, in_height, in_width, input_channels)

    # Perform the forward pass
    print("Running forward pass...")
    activations = conv_layer.forward(x)
    print("Output of forward pass (activations):")
    print(activations)
    print("Activations shape:", activations.shape)

    # Generate a random gradient of the output (as if it came from the next layer in backpropagation)
    d_out = np.random.randn(*activations.shape)

    # Perform the backward pass
    print("\nRunning backward pass...")
    d_x, d_weights, d_biases = conv_layer.backward(d_out, x)

    # Display the shapes of the gradients
    print("Gradients with respect to input (d_x) shape:", d_x.shape)
    print("Gradients with respect to weights (d_weights) shape:", d_weights.shape)
    print("Gradients with respect to biases (d_biases) shape:", d_biases.shape)

    # Optionally, print actual gradient values
    print("\nGradients with respect to input (d_x):")
    print(d_x)
    print("\nGradients with respect to weights (d_weights):")
    print(d_weights)
    print("\nGradients with respect to biases (d_biases):")
    print(d_biases)


