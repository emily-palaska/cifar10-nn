import numpy as np
import math
from utils import ReLU

class ConvNeuralNetwork:
    def __init__(self, input_shape, conv_layers, hidden_sizes, output_size, init_type="xavier", batch_size=100):
        self.batch_size = batch_size
        self.conv_layers = conv_layers  # List of convolutional layer configurations
        self.hidden_sizes = hidden_sizes  # List of hidden layers
        self.output_size = output_size  # Number of classes for classification
        self.init_type = init_type
        self.input_shape = input_shape  # Shape of the input (height, width, channels)

        self.conv_params = []  # List of dictionaries for conv layers parameters
        self.fc_params = []  # List of dictionaries for fully connected layers parameters
        self.build_network()
        self.relu = ReLU()

    def initialize_weights(self, size, init_type="xavier"):
        out_size = size[0]
        in_size = size[1] if len(size) == 2 else size[3]

        if init_type == "xavier":
            stddev = np.sqrt(2 / (in_size + out_size))
            return np.random.normal(0, stddev, size=size)
        elif init_type == "he":
            return np.random.randn(size) * np.sqrt(2.0 / in_size)
        else:
            return np.random.randn(size) * 0.01

    def build_network(self):
        # Convolutional layers initialization
        height = self.input_shape[0]
        width = self.input_shape[1]
        input_channels = self.input_shape[2]  # Number of input channels (e.g., 3 for RGB images)

        for i, (num_filters, kernel_size, stride, padding) in enumerate(self.conv_layers):
            filter_shape = (num_filters, kernel_size, kernel_size, input_channels)
            height = math.floor((height - kernel_size + 2 * padding) / stride + 1)
            width = math.floor((width - kernel_size + 2 * padding) / stride + 1)

            weights = self.initialize_weights(filter_shape, init_type=self.init_type)
            biases = np.zeros((num_filters, 1))

            self.conv_params.append({"weights": weights, "biases": biases, "stride": stride, "padding": padding})
            input_channels = num_filters  # The next layer takes the output from the current one

        # Fully connected layers initialization
        input_size = height * width * input_channels  # Flattened size after conv layers
        hidden_layers = [input_size] + self.hidden_sizes
        for i in range(len(hidden_layers) - 1):
            size = (hidden_layers[i], hidden_layers[i + 1])
            weights = self.initialize_weights(size, init_type=self.init_type)

            self.fc_params.append({
                "weights": weights,
                "biases": np.zeros((hidden_layers[i + 1], 1))
            })

        # Output layer initialization
        size = (self.hidden_sizes[-1], self.output_size)
        weights = self.initialize_weights(size, init_type=self.init_type)

        self.fc_params.append({
            "weights": weights,
            "biases": np.zeros((self.output_size, 1))
        })

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def conv2d(self, X, weights, biases, stride=1, padding=0):
        # Extract dimensions
        batch_size, in_height, in_width, in_channels = X.shape
        num_filters, _, kernel_size, _ = weights.shape

        # Calculate output dimensions
        out_height = math.floor((in_height - kernel_size + 2 * padding) / stride + 1)
        out_width = math.floor((in_width - kernel_size + 2 * padding) / stride + 1)

        # Initialize output tensor
        output = np.zeros((batch_size, out_height, out_width, num_filters))

        # Pad the input (pad height and width only)
        X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')

        # Perform convolution
        for n in range(batch_size):
            for i in range(out_height):
                for j in range(out_width):
                    # Define the slice of the input to be multiplied with filters
                    h_start, h_end = i * stride, i * stride + kernel_size
                    w_start, w_end = j * stride, j * stride + kernel_size

                    # Apply convolution for each filter
                    for k in range(num_filters):
                        X_slice = X_padded[n, h_start:h_end, w_start:w_end, :]  # Correctly shaped 3D slice (H, W, C)
                        output[n, i, j, k] = np.sum(X_slice * weights[k, :, :, :]).item() + biases[k].item()

        return output

    def forward_propagation(self, X):
        activations = []
        # Forward through convolutional layers
        conv_activations = X
        for param in self.conv_params:
            print(f'Convolution for {param["weights"].shape} layer')
            conv_activations = self.conv2d(conv_activations, param["weights"], param["biases"],
                                           param["stride"], param["padding"])
            conv_activations = self.relu(conv_activations)
            activations.append(conv_activations)
        # Flatten before fully connected layers
        flattened = conv_activations.reshape(X.shape[0], -1)
        fc_activations = flattened
        for param in self.fc_params[:-1]:
            print(f'Dot product for {param["weights"].shape} layer')
            fc_activations = np.dot(fc_activations, param["weights"]) + param["biases"].T
            fc_activations = self.relu(fc_activations)
            activations.append(fc_activations)

        # Output layer
        output = np.dot(fc_activations, self.fc_params[-1]["weights"]) + self.fc_params[-1]["biases"].T
        output = self.softmax(output)
        activations.append(output)

        return activations

    def cross_entropy_loss(self, predictions, labels):
        return -np.mean(np.sum(labels * np.log(predictions + 1e-9), axis=1))

    def backward_propagation(self, activations, labels):
        m = labels.shape[0]
        da = activations[-1] - labels  # Gradient for output layer

        # Backward through fully connected layers
        fc_gradients = []
        act_lvl = len(activations) - 1

        print('Activations shape:')
        for act in activations:
            print(act.shape)

        for i in reversed(range(len(self.fc_params))):
            print(f'Backward propagation of fc layer {i}, activation level {act_lvl}')
            dw = np.dot(activations[act_lvl].T, da) / m
            db = np.sum(da, axis=0, keepdims=True) / m
            print(f'Dot product of {da.shape} and {self.fc_params[i]["weights"].T.shape} with activations {activations[act_lvl - 1].shape}')
            if i > 0:
                da = np.dot(da, self.fc_params[i]["weights"].T) * (activations[act_lvl - 1] > 0)  # Applying ReLU derivative
                act_lvl -= 1
            else:
                da = np.dot(da, self.fc_params[i]["weights"].T)
                da = da * (activations[act_lvl - 1].reshape(da.shape) > 0)
            fc_gradients.insert(0, {"dw": dw, "db": db})


        # Backward through convolutional layers
        conv_gradients = []
        for i in reversed(range(len(self.conv_params))):
            print(f'Backward propagation of conv layer {act_lvl}')
            param = self.conv_params[i]
            print(f'Da {da.shape} activations {activations[act_lvl - 1].shape}')
            da = da.reshape(activations[act_lvl - 1].shape)  # Reshape to the shape before ReLU
            dw = np.zeros_like(param["weights"])
            db = np.zeros_like(param["biases"])

            for n in range(m):
                # For each sample in the batch, accumulate the gradients
                for k in range(param["weights"].shape[0]):  # For each filter
                    for c in range(da.shape[3]):  # For each input channel
                        h_end, w_end = param["weights"].shape[1], param["weights"].shape[2]
                        for h in range(da.shape[1] - h_end + 1):
                            for w in range(da.shape[2] - w_end + 1):
                                window = activations[act_lvl - 1][n, h:h + h_end, w:w + w_end, c]
                                dw[k, :, :, c] += window * da[n, h, w, k]
                    db[k] += np.sum(da[n, :, :, k])

            dw /= m
            db /= m
            conv_gradients.insert(0, {"dw": dw, "db": db})
            padding = param["padding"]
            #da = np.pad(da, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
            act_lvl -= 1

        return conv_gradients, fc_gradients

    def update_parameters(self, conv_gradients, fc_gradients, learning_rate=0.01):
        for i, param in enumerate(self.conv_params):
            param["weights"] -= learning_rate * conv_gradients[i]["dw"]
            param["biases"] -= learning_rate * conv_gradients[i]["db"]

        for i, param in enumerate(self.fc_params):
            param["weights"] -= learning_rate * fc_gradients[i]["dw"]
            param["biases"] -= learning_rate * fc_gradients[i]["db"]

    def train(self, X_train, y_train, epochs=1000, learning_rate=0.01):
        # Initialize output file
        with open("output_conv.txt", "w") as file:
            file.write(f"ConvNet training started.\n")

        n_samples = X_train.shape[0]
        loss_per_epoch = []

        for epoch in range(epochs):
            batch_inds = np.random.randint(0, n_samples, size=self.batch_size)
            X_train_batch = X_train[batch_inds]
            y_train_batch = y_train[batch_inds]

            print(f'\nEpoch {epoch} starting')
            start_time = time.time()
            activations = self.forward_propagation(X_train_batch)
            loss = self.cross_entropy_loss(activations[-1], y_train_batch)
            conv_gradients, fc_gradients = self.backward_propagation(activations, y_train_batch)
            self.update_parameters(conv_gradients, fc_gradients, learning_rate)
            end_time = time.time()

            # Write progress to file
            with open("output_conv.txt", "a") as file:
                file.write(f"Epoch {epoch}, Loss: {loss:.4f}, Time: {end_time - start_time:.2f}s\n")

            loss_per_epoch.append(loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return loss_per_epoch

    def evaluate(self, X_test, y_test):
        activations = self.forward_propagation(X_test)
        predictions = np.argmax(activations[-1], axis=1)
        accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
