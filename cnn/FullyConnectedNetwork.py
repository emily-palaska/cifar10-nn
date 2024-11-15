from FullyConnectedLayer import FullyConnectedLayer
from utils import *

class FullyConnectedNetwork:
    def __init__(self, input_size, num_classes, layer_config):
        self.layers = []
        self.layer_config = layer_config
        self.softmax = Softmax()

        # Initialize layers according to given configuration
        neurons_in = input_size
        for neurons_out in layer_config:
            layer = FullyConnectedLayer(neurons_in, neurons_out, init_type='he')
            self.layers.append(layer)
            neurons_in = neurons_out

        # Add one final layer for the output
        layer = FullyConnectedLayer(neurons_in, num_classes)
        self.layers.append(layer)


    def train(self, x_train, y_train, epochs=1000, learning_rate=0.01):
        # x is batch_size, input
        for epoch in range(epochs):
            activations = [x_train]
            # forward pass layers
            for layer in self.layers:
                z = layer.forward(activations[-1])
                activations.append(z)
            # apply softmax
            activations.append(self.softmax.forward(activations[-1]))

            # calculate loss
            loss = cross_entropy_loss(activations[-1], y_train)
            if epoch % 100 == 0:
                print(f"Epoch {epoch:03d} loss: {loss}")

            # back propagation
            num_samples = x_train.shape[0]
            da = activations[-1] - y_train  # Gradient for output layer
            for layer in reversed(self.layers):
                da = layer.backward(da, num_samples, learning_rate= learning_rate)

    def evaluate(self, X, Y):
        return NotImplemented

def main():
    batch_size = 100
    input_size = 32*32*3
    num_classes = 10
    layer_config = [256,256]
    nn = FullyConnectedNetwork(input_size, num_classes, layer_config)
    print("Shape of weights")
    for i in range(len(layer_config)):
        print(nn.layers[i].weights.shape)
    print("----------------------")
    # random input
    x = np.random.rand(batch_size, input_size)
    y = np.random.randint(0, num_classes, batch_size)
    y = to_one_hot(y, num_classes)
    nn.train(x,y)

if __name__ == '__main__':
    main()