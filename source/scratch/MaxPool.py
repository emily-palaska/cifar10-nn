import numpy as np

class MaxPool:
    def __init__(self, filter_size, stride):
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, input):
        self.last_input = input
        h, w = input.shape[1:3]
        output_dim = (h - self.filter_size) // self.stride + 1
        output = np.zeros((input.shape[0], output_dim, output_dim, input.shape[-1]))

        for i in range(output_dim):
            for j in range(output_dim):
                region = input[:,
                               i * self.stride:i * self.stride + self.filter_size,
                               j * self.stride:j * self.stride + self.filter_size]
                output[:, i, j] = np.max(region, axis=(1, 2))

        return output
