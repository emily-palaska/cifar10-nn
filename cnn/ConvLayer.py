import numpy as np
import math
from cnn.utils import ReLU

class ConvLayer:
    def __init__(self, num_filters, kernel_size, input_size, stride=1, padding=0, init_type='xavier'):
        """
        Initialize a convolutional layer.

        :param num_filters: number of filters
        :param kernel_size: kernel size (assumed square)
        :param input_size: (batch_size, height, width, channels)
        :param stride: stride of convolution
        :param padding: padding of convolution
        """
        self.type = 'conv'
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # calculate input and output size
        self.batch_size, self.in_height, self.in_width, self.in_channels = input_size
        self.out_height = math.floor((self.in_height - kernel_size + 2 * padding) / stride + 1)
        self.out_width = math.floor((self.in_width - kernel_size + 2 * padding) / stride + 1)

        # initialize relu instance
        self.relu = ReLU()

        # initialize weights and biases with specified method
        self.weights = self.initialize_weights((num_filters, kernel_size, kernel_size, self.in_channels), init_type)
        self.biases = np.zeros((num_filters, 1))

        # placeholder for cached input and activations
        self.activations = None
        self.input = None

    def get_in_size(self):
        return self.batch_size, self.in_height, self.in_width, self.in_channels

    def get_out_size(self):
        return self.batch_size, self.out_height, self.out_width, self.num_filters

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
        self.activations = self.convolve(x)
        self.activations = self.relu.forward(self.activations)
        self.input = x
        return self.activations

    def convolve(self, x):
        # Extract dimensions
        batch_size, in_height, in_width, in_channels = x.shape
        if not in_channels == self.in_channels:
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

    def backward(self, da, learning_rate=0.01):
        # Get the last input
        a = self.input

        # Apply ReLU derivative
        da = da.reshape(*self.get_out_size())
        dz = da * self.relu.backward(self.activations)

        # Initialize new gradients for activations, weights and biases
        dw = np.zeros_like(self.weights)
        db = np.zeros_like(self.biases)
        da = np.zeros((self.batch_size, self.in_height, self.in_width, self.in_channels))

        # Pad the input
        a_pad = np.pad(a, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                          mode='constant')
        da_pad = np.pad(da, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                          mode='constant')

        # Loop over the batch size, height, width, and filters to compute gradients
        for n in range(self.batch_size):
            for i in range(self.out_height):
                for j in range(self.out_width):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size

                    for k in range(self.num_filters):
                        # Slice the region from x_padded that contributed to the output
                        a_slice = a_pad[n, h_start:h_end, w_start:w_end, :]

                        # Accumulate the gradient for the input
                        da_pad[n, h_start:h_end, w_start:w_end, :] += self.weights[k] * dz[n, i, j, k]

                        # Accumulate the gradient for the filter weights and biases
                        dw[k] += a_slice * dz[n, i, j, k]
                        db += dz[n, i, j, k]

        # Remove padding from d_x_padded to get the gradient with respect to the original input
        if self.padding > 0:
            da = da_pad[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            da = da_pad

        # Update parameters
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db

        return da


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


